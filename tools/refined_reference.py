"""
The finite planar FDTD reference on a refined grid (0.005 m) against the 0.01 m one.

First checks that the fracture written as a box on the 0.01 m grid reproduces the model that
reads it from the HDF5 volume (largest difference relative to the largest field, per
component), so that the two grids differ in the cell alone. Then compares, over the
evaluation window:
  * the direct waves of the two grids (0 to 60 ns, where the direct wave peaks), which checks
    the source normalization below;
  * the scattered reference at 0.005 m against the one at 0.01 m, i.e. the change of the
    reference itself;
  * the RA-HPO trace of the finite planar case against each reference, and the RA-HPO trace
    given the thickness of the refined slab (case refined_planar, 0.045 m) against the refined
    reference and against the finite planar trace.

gprMax gives a Hertzian dipole the length of one cell, so its moment, and every field it
radiates, halves with the cell. The 0.005 m traces are therefore multiplied by
H_COARSE / H_FINE = 2 to describe the same source.

Models: examples/gprMax_models/For_refined_reference/ (the 0.01 m box run and the 0.005 m fracture and
direct-wave runs) and the finite planar case of cases.py. Writes output/metrics/refined_reference.csv.

Usage:  python tools/refined_reference.py [--out-csv FILE]
"""
from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

import h5py
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parent))
from accuracy_metrics import misfits  # noqa: E402
from examples.cases import (BY_KEY, METRICS_DIR, REFINED, REFINED_BOX, REFINED_DIRECT,  # noqa: E402
                   TIME_RANGE, fdtd_output)
from plot_traces import (compare_traces, load_gprmax_ez, load_rahpo,  # noqa: E402
                         load_scattered_gprmax)

H_COARSE, H_FINE = 0.01, 0.005
DIRECT_WINDOW = (0.0, 60.0)   # the direct wave peaks near 27 ns, before the evaluation window
COMPONENTS = ("Ex", "Ey", "Ez", "Hx", "Hy", "Hz")


def box_check(model: str, box: str) -> dict[str, float]:
    """max |box - model| / max |model| for each field component at the receiver."""
    out = {}
    with h5py.File(fdtd_output(model), "r") as fm, h5py.File(fdtd_output(box), "r") as fb:
        for comp in COMPONENTS:
            a = np.asarray(fm[f"/rxs/rx1/{comp}"], float)
            b = np.asarray(fb[f"/rxs/rx1/{comp}"], float)
            if a.shape != b.shape:
                raise ValueError(f"{comp}: {a.shape} and {b.shape} samples")
            out[comp] = float(np.max(np.abs(b - a)) / np.max(np.abs(a)))
    return out


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[1])
    ap.add_argument("--out-csv", default=str(METRICS_DIR / "refined_reference.csv"))
    out_csv = Path(ap.parse_args().out_csv)

    c = BY_KEY["finite_planar"]
    print("Fracture as a box against the HDF5 volume, 0.01 m grid: max |difference| / max |field|")
    box = box_check(c.fdtd, REFINED_BOX)
    print("  " + "  ".join(f"{k} {v:.2g}" for k, v in box.items()))

    t1, s1 = load_scattered_gprmax(c)
    tf, ef = load_gprmax_ez(REFINED)
    td, ed = load_gprmax_ez(REFINED_DIRECT)
    if ef.size != ed.size or not np.allclose(tf, td):
        raise ValueError("time axes of the 0.005 m fracture and direct-wave runs differ")
    k = H_COARSE / H_FINE
    s2 = k * (ef - ed)
    tr, pr = load_rahpo(c.key)
    rows = []

    def add(label, ref, cand, ta, a, tb, b, window=TIME_RANGE):
        m = compare_traces(ta, a, tb, b, time_range=window)
        x = misfits(ta, a, tb, b, time_range=window)
        print(f"  {label:40} chi {m['Relative misfit (%)']:5.2f}%  r {m['Pearson r']:.5f}  "
              f"peak ratio {m['Peak ratio (ref/RA-HPO)']:.4f}  dt {m['Peak-time diff (ns)']:+.3f} ns  "
              f"chi_s {x['chi_s_pct']:.2f}% (a {x['alpha_opt']:.4f})")
        rows.append({"comparison": label, "reference": ref, "compared": cand,
                     "window_ns": f"{window[0]:g}-{window[1]:g}",
                     "chi_pct": m["Relative misfit (%)"], "nrmse_pct": m["NRMSE (%)"],
                     "r": m["Pearson r"], "peak_ref": m["Peak ref (V/m)"],
                     "peak_compared": m["Peak RA-HPO (V/m)"], "peak_ratio": m["Peak ratio (ref/RA-HPO)"],
                     "peak_dt_ns": m["Peak-time diff (ns)"], "chi_s_pct": x["chi_s_pct"],
                     "alpha_opt": x["alpha_opt"], "chi_st_pct": x["chi_st_pct"],
                     "tau_opt_ns": x["tau_opt_ns"]})

    print(f"Finite planar case, refined-grid reference (0.005 m traces x {k:g})")
    tdc, edc = load_gprmax_ez(c.direct)
    add("direct wave, 0.005 m against 0.01 m", c.direct, REFINED_DIRECT, tdc, edc, td, k * ed,
        window=DIRECT_WINDOW)
    add("reference, 0.005 m against 0.01 m", c.fdtd, REFINED, t1, s1, tf, s2)
    add("RA-HPO against the 0.01 m reference", c.fdtd, c.key, t1, s1, tr, pr)
    add("RA-HPO against the 0.005 m reference", REFINED, c.key, tf, s2, tr, pr)
    # RA-HPO given the thickness of the eight-cell 0.005 m slab (0.045 m) instead of 0.05 m
    r45 = BY_KEY["refined_planar"]
    tq, pq = load_rahpo(r45.key)
    add("RA-HPO 0.045 m against the 0.005 m reference", REFINED, r45.key, tf, s2, tq, pq)
    add("RA-HPO 0.045 m against RA-HPO 0.05 m", c.key, r45.key, tr, pr, tq, pq)

    out_csv.parent.mkdir(parents=True, exist_ok=True)
    with open(out_csv, "w", newline="") as fh:
        wr = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
        wr.writeheader()
        for r in rows:
            wr.writerow({k: (f"{v:.6g}" if isinstance(v, float) else v) for k, v in r.items()})
    box_csv = out_csv.with_name(out_csv.stem + "_box_check.csv")
    with open(box_csv, "w", newline="") as fh:
        wr = csv.writer(fh)
        wr.writerow(["component", "max_abs_difference_over_max_abs_field"])
        for k, v in box.items():
            wr.writerow([k, f"{v:.6g}"])
    print(f"wrote {out_csv} and {box_csv}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
