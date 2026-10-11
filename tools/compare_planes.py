"""
Compare two gprMax reference planes recorded at different receiver spacings.

gprMax receivers are output points that do not enter the update equations, so a plane recorded
at 0.04 m holds the same values as a plane recorded at 0.01 m and decimated by four. This
script checks that on the receivers themselves: each file is decimated by its own step, the
recorded Position attributes of every compared pair of receivers must agree (which catches a
misaligned grid), and the field time series are then compared sample for sample.

compare_ref_planes.py, by contrast, compares two planes from different model domains with
accuracy metrics; here the two files are meant to hold the same numbers.

    reference   .out at 0.01 m, --ref-step 4   ->  226 x 226
    candidate   .out at 0.04 m, --cand-step 1  ->  226 x 226

Usage
-----
    python tools/compare_planes.py REFERENCE.out CANDIDATE.out \\
        --ref-step 4 --cand-step 1 --max-rel-l2 1e-6

Exit status
-----------
    0  planes pass
    1  planes differ beyond the tolerance, are misaligned, or cannot be compared
"""
from __future__ import annotations

import argparse
import sys

import h5py
import numpy as np

DEFAULT_COMPS = ("Ex", "Ey", "Ez", "Hx", "Hy", "Hz")


def plane_layout(f, path):
    """
    Infer the receiver grid of one gprMax .out, using the same rules as
    ra_hpo.load_gprmax_plane so that this check and the solver agree by construction.

    Returns (ny, nz, is_y_fast, x1, y0, z0, dy, dz).
    """
    nrx = int(f.attrs["nrx"])
    p1 = f["/rxs/rx1"].attrs["Position"]
    p2 = f["/rxs/rx2"].attrs["Position"]

    dy_native = float(p2[1] - p1[1])
    dz_native = float(p2[2] - p1[2])

    if abs(dy_native) > 1e-9:
        is_y_fast = True
        fast = dy_native
    elif abs(dz_native) > 1e-9:
        is_y_fast = False
        fast = dz_native
    else:
        raise ValueError(f"{path}: rx1 and rx2 share a position; not a receiver grid")

    pN = f[f"/rxs/rx{nrx}"].attrs["Position"]
    span = (p1[1] - pN[1]) if is_y_fast else (p1[2] - pN[2])
    fast_len = int(round(abs(span) / abs(fast))) + 1
    if fast_len <= 0 or nrx % fast_len:
        # The span is measured along the slow axis when the array is a single row, so fall
        # back to treating the whole file as one fast-axis run rather than guessing.
        fast_len = nrx
    slow_len = nrx // fast_len

    if is_y_fast:
        ny, nz = fast_len, slow_len
    else:
        nz, ny = fast_len, slow_len

    return dict(nrx=nrx, ny=ny, nz=nz, is_y_fast=is_y_fast,
                x1=float(p1[0]), dt=float(f.attrs["dt"]), Nt=int(f.attrs["Iterations"]))


def rx_index(layout, iy, iz):
    """0-based receiver index of grid point (iy, iz); dataset name is rx{index+1}."""
    if layout["is_y_fast"]:
        return iz * layout["ny"] + iy
    return iy * layout["nz"] + iz


def fmt_pos(p):
    """Positions print as plain metres; numpy scalar reprs make the log unreadable."""
    return "(" + ", ".join(f"{float(v):g}" for v in p) + ")"


def ulp_distance(a, b):
    """Distance in representable float32 steps, for arrays that are finite."""
    ia = a.astype(np.float32).view(np.int32).astype(np.int64)
    ib = b.astype(np.float32).view(np.int32).astype(np.int64)
    ia = np.where(ia < 0, np.int64(0x80000000) - ia, ia)
    ib = np.where(ib < 0, np.int64(0x80000000) - ib, ib)
    return np.abs(ia - ib)


def compare(ref_path, cand_path, ref_step, cand_step, comps, subsample, seed, max_rel_l2):
    with h5py.File(ref_path, "r") as fr, h5py.File(cand_path, "r") as fc:
        lr = plane_layout(fr, ref_path)
        lc = plane_layout(fc, cand_path)

        print(f"reference : {ref_path}")
        print(f"            {lr['ny']} x {lr['nz']} rx, {lr['Nt']} samples, "
              f"dt = {lr['dt']:.6e} s, step {ref_step}")
        print(f"candidate : {cand_path}")
        print(f"            {lc['ny']} x {lc['nz']} rx, {lc['Nt']} samples, "
              f"dt = {lc['dt']:.6e} s, step {cand_step}")

        # A difference in any of these is a different model, not a different recording, and
        # no tolerance should paper over it.
        for key, label in (("Nt", "iteration count"), ("dt", "time step"), ("x1", "plane x")):
            if lr[key] != lc[key]:
                print(f"  [FAIL] {label} differs: {lr[key]} vs {lc[key]}")
                return False

        ny_common = (lr["ny"] - 1) // ref_step + 1
        nz_common = (lr["nz"] - 1) // ref_step + 1
        if ((lc["ny"] - 1) // cand_step + 1 != ny_common or
                (lc["nz"] - 1) // cand_step + 1 != nz_common):
            print(f"  [FAIL] decimated grids differ: reference gives "
                  f"{ny_common} x {nz_common}, candidate gives "
                  f"{(lc['ny'] - 1) // cand_step + 1} x {(lc['nz'] - 1) // cand_step + 1}")
            return False
        print(f"common grid: {ny_common} x {nz_common} receivers")

        total = ny_common * nz_common
        rng = np.random.default_rng(seed)
        if subsample and subsample < total:
            picks = np.sort(rng.choice(total, size=subsample, replace=False))
            print(f"comparing {subsample} of {total} positions "
                  f"(seed {seed}), components {', '.join(comps)}")
        else:
            picks = np.arange(total)
            print(f"comparing all {total} positions, components {', '.join(comps)}")

        missing = [c for c in comps if f"/rxs/rx1/{c}" not in fr or f"/rxs/rx1/{c}" not in fc]
        if missing:
            print(f"  [FAIL] component(s) absent from one of the files: {', '.join(missing)}")
            return False

        n_bitwise = 0
        worst = dict(rel_l2=0.0, max_abs=0.0, max_ulp=0, pos=None)
        sum_sq_diff = {c: 0.0 for c in comps}
        sum_sq_ref = {c: 0.0 for c in comps}

        for n, flat in enumerate(picks):
            iy, iz = divmod(int(flat), nz_common)
            kr = rx_index(lr, iy * ref_step, iz * ref_step)
            kc = rx_index(lc, iy * cand_step, iz * cand_step)

            gr = fr[f"/rxs/rx{kr + 1}"]
            gc = fc[f"/rxs/rx{kc + 1}"]

            # The alignment check. Without it a grid off-by-one would show up only as a
            # small field difference, which is exactly what a tolerance would absorb.
            pr = np.asarray(gr.attrs["Position"], dtype=float)
            pc = np.asarray(gc.attrs["Position"], dtype=float)
            if np.max(np.abs(pr - pc)) > 1e-9:
                print(f"  [FAIL] position mismatch at grid ({iy}, {iz}): "
                      f"reference rx{kr + 1} at {fmt_pos(pr)}, "
                      f"candidate rx{kc + 1} at {fmt_pos(pc)}")
                return False

            for c in comps:
                a = np.asarray(gr[c][:], dtype=np.float32)
                b = np.asarray(gc[c][:], dtype=np.float32)
                if np.array_equal(a, b):
                    n_bitwise += 1
                    sum_sq_ref[c] += float(np.dot(a.astype(np.float64), a.astype(np.float64)))
                    continue
                d = b.astype(np.float64) - a.astype(np.float64)
                sum_sq_diff[c] += float(np.dot(d, d))
                sum_sq_ref[c] += float(np.dot(a.astype(np.float64), a.astype(np.float64)))
                norm = float(np.linalg.norm(a))
                rel = float(np.linalg.norm(d) / norm) if norm > 0 else float("inf")
                if rel > worst["rel_l2"]:
                    worst.update(rel_l2=rel, max_abs=float(np.max(np.abs(d))),
                                 max_ulp=int(np.max(ulp_distance(a, b))),
                                 pos=(iy, iz, c, fmt_pos(pr)))

            if n and n % 500 == 0:
                print(f"  ... {n}/{len(picks)} positions", flush=True)

    n_series = len(picks) * len(comps)
    print(f"\n  series compared    : {n_series}")
    print(f"  bitwise identical  : {n_bitwise}/{n_series}")

    tot_diff = sum(sum_sq_diff.values())
    tot_ref = sum(sum_sq_ref.values())
    rel_l2_all = (tot_diff ** 0.5 / tot_ref ** 0.5) if tot_ref > 0 else float("inf")
    print(f"  relative L2 (all)  : {rel_l2_all:.6e}")

    if n_bitwise == n_series:
        print("  [PASS] every compared series is bitwise identical")
        return True

    print(f"  worst series       : grid {worst['pos'][:2]}, {worst['pos'][2]}, "
          f"position {worst['pos'][3]}")
    print(f"    relative L2      : {worst['rel_l2']:.6e}")
    print(f"    max |difference| : {worst['max_abs']:.6e}")
    print(f"    max ULP distance : {worst['max_ulp']} (float32 unit roundoff 5.96e-08)")

    if max_rel_l2 > 0.0 and rel_l2_all <= max_rel_l2 and worst["rel_l2"] <= max_rel_l2:
        print(f"  [PASS] within the accepted relative L2 of {max_rel_l2:.1e}")
        print("  [WARN] not bitwise identical. gprMax receivers are output-only, so the "
              "retained\n         values were expected to match exactly; check that the two "
              "recordings\n         come from the same model.")
        return True

    if max_rel_l2 > 0.0:
        print(f"  [FAIL] relative L2 exceeds {max_rel_l2:.1e}")
    else:
        print("  [FAIL] not bitwise identical (pass --max-rel-l2 to allow a tolerance)")
    return False


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("reference", help="gprMax .out recorded at the finer receiver spacing")
    ap.add_argument("candidate", help="gprMax .out recorded at the decimated spacing")
    ap.add_argument("--ref-step", type=int, default=4,
                    help="receiver decimation applied to the reference (default 4)")
    ap.add_argument("--cand-step", type=int, default=1,
                    help="receiver decimation applied to the candidate (default 1)")
    ap.add_argument("--comps", nargs="+", default=list(DEFAULT_COMPS),
                    help="field components to compare (default all six)")
    ap.add_argument("--subsample", type=int, default=2000,
                    help="compare this many receiver positions, drawn without replacement; "
                         "0 compares all of them (default 2000)")
    ap.add_argument("--seed", type=int, default=0, help="seed for the subsample (default 0)")
    ap.add_argument("--max-rel-l2", type=float, default=0.0,
                    help="accept this relative L2 difference instead of requiring bitwise "
                         "equality (default 0, i.e. bitwise)")
    args = ap.parse_args()

    if args.ref_step < 1 or args.cand_step < 1:
        print("  [FAIL] decimation steps must be >= 1")
        sys.exit(1)

    try:
        ok = compare(args.reference, args.candidate, args.ref_step, args.cand_step,
                     tuple(args.comps), args.subsample, args.seed, args.max_rel_l2)
    except (OSError, KeyError, ValueError) as exc:
        print(f"  [FAIL] {exc}")
        sys.exit(1)
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
