"""
Discretization convergence of the analytic stage (Supplementary Table S1).

Two sweeps, each run for the finite planar and the biaxial corrugated (A = 0.25 m) fracture
(cases.py, group "convergence"):
  * element size dl = 0.005 to 0.16 m, with the plane read at 0.04 m;
  * plane receiver spacing 0.02 to 0.16 m, with dl = 0.01 m.

For each level the script computes the misfit chi against the finest level of its sweep, the
observed order of convergence from successive differences, and the Richardson-extrapolated
error at the adopted settings (dl = 0.01 m, spacing 0.04 m). It also checks the statements of
the paper that rest on these numbers and writes metrics/convergence.csv.

The metric
----------
Both traces are linearly interpolated onto 2001 uniform points over 100 to 200 ns, then
    NRMSE      = rms(p - d) / (max d - min d) * 100 %
    Pearson r, peak ratio max|d|/max|p|, peak-time difference, and
    chi        = ||p - d|| / ||d|| * 100 %   (the value Table S1 reports)
The reference d is the finer trace.

Richardson extrapolation
------------------------
With error(h) = C h^p, the difference between levels h and h/2 is C h^p (1 - 2^-p), so the error
at h is that difference divided by (1 - 2^-p). The paper uses p = 2 (division by 0.75), which
the measured orders support over the whole range; the value at the locally measured p is
printed too. Observed order: with successive differences d_k between levels h_k and 2 h_k,
p_k = log2(d_{k+1} / d_k).

Usage
-----
    python tools/convergence.py            # print Table S1, the orders and the estimates
    python tools/convergence.py --check    # exit non-zero unless every value of the paper is reproduced
"""
from __future__ import annotations

import argparse
import csv
import math
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from rahpo.cases import ELEMENT_SIZES, METRICS_DIR, PLANE_SPACINGS, trace_file  # noqa: E402

WIN = (100.0, 200.0)
NGRID = 2001
SHAPES = ("planar", "corrugated")
# (axis, case-key prefix, levels finest first as (label, tag))
AXES = [
    ("element size", "convergence_dl", [(f"{v:.3f}", t) for t, v in ELEMENT_SIZES]),
    ("plane spacing", "convergence_res", [(f"{v:.2f}", t) for t, v in PLANE_SPACINGS]),
]
ADOPTED = {"element size": "0.010", "plane spacing": "0.04"}

# Values stated in the paper, so a drift in the traces or the metric fails loudly
PAPER = {
    "orders_element_size": (1.83, 2.18),                       # min, max over both shapes
    "orders_plane_spacing": {"planar": (1.97, 1.86), "corrugated": (2.00, 2.14)},
    "richardson_pct": {("element size", "planar"): 7.2e-3, ("element size", "corrugated"): 5.3e-3,
                       ("plane spacing", "planar"): 1.5e-1, ("plane spacing", "corrugated"): 2.2e-1},
    # chi of the method against FDTD (Table 2) for the two swept geometries, the yardstick of
    # "three orders of magnitude" (element size) and "at least thirty times" (plane spacing)
    "method_chi_pct": {"planar": 7.8, "corrugated": 7.3},
}
CHI = 4   # index of chi in metrics()


def load(key: str) -> tuple[np.ndarray, np.ndarray]:
    z = np.load(trace_file(key))
    return np.asarray(z["time_ns"], float), np.asarray(z["amplitude"], float)


def metrics(ref: str, cand: str) -> tuple[float, float, float, float, float]:
    """NRMSE %, Pearson r, peak ratio, peak-time difference ns, chi %; ref is d."""
    tr, ar = load(ref)
    tc, ac = load(cand)
    tg = np.linspace(*WIN, NGRID)
    d = np.interp(tg, tr, ar)
    p = np.interp(tg, tc, ac)
    return (np.sqrt(np.mean((p - d) ** 2)) / (d.max() - d.min()) * 100,
            np.corrcoef(p, d)[0, 1],
            np.max(np.abs(d)) / np.max(np.abs(p)),
            tg[np.argmax(np.abs(p))] - tg[np.argmax(np.abs(d))],
            np.linalg.norm(p - d) / np.linalg.norm(d) * 100)


def successive(pre: str, lv: list, shape: str) -> list[float]:
    """chi between each adjacent pair of levels, finest pair first."""
    return [metrics(f"{pre}_{shape}_{a[1]}", f"{pre}_{shape}_{b[1]}")[CHI]
            for a, b in zip(lv[:-1], lv[1:])]


def write_csv(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["axis", "shape", "level", "reference", "nrmse_pct", "pearson_r", "peak_ratio",
                    "dt_peak_ns", "chi_pct", "comparison"])
        for axis, pre, lv in AXES:
            for shape in SHAPES:
                for l in lv[1:]:
                    m = metrics(f"{pre}_{shape}_{lv[0][1]}", f"{pre}_{shape}_{l[1]}")
                    w.writerow([axis, shape, l[0], lv[0][0], *[f"{x:.6g}" for x in m], "vs_finest"])
                for a, b in zip(lv[:-1], lv[1:]):
                    m = metrics(f"{pre}_{shape}_{a[1]}", f"{pre}_{shape}_{b[1]}")
                    w.writerow([axis, shape, b[0], a[0], *[f"{x:.6g}" for x in m], "successive"])


def main() -> int:
    ap = argparse.ArgumentParser(description="Discretization convergence (Supplementary Table S1).")
    ap.add_argument("--check", action="store_true",
                    help="exit non-zero unless every value of the paper is reproduced")
    ap.add_argument("--out-csv", default=str(METRICS_DIR / "convergence.csv"))
    a = ap.parse_args()

    missing = [f"{pre}_{s}_{t}" for _, pre, lv in AXES for s in SHAPES for _, t in lv
               if not trace_file(f"{pre}_{s}_{t}").exists()]
    if missing:
        print(f"missing traces: {missing}", file=sys.stderr)
        return 2
    ok = True

    print("Supplementary Table S1: chi % against the finest level of each sweep\n")
    for axis, pre, lv in AXES:
        print(f"  {axis}")
        print(f"    {'level':>7}  {'planar':>9}  {'corrugated':>10}")
        for l in lv[1:]:
            vals = [metrics(f"{pre}_{s}_{lv[0][1]}", f"{pre}_{s}_{l[1]}")[CHI] for s in SHAPES]
            mark = "  <- adopted" if l[0] == ADOPTED[axis] else ""
            print(f"    {l[0]:>7}  {vals[0]:9.4f}  {vals[1]:10.4f}{mark}")
        print()

    print("Observed orders p_k = log2(d_{k+1} / d_k), successive differences finest first\n")
    orders = {}
    for axis, pre, lv in AXES:
        for shape in SHAPES:
            d = successive(pre, lv, shape)
            orders[(axis, shape)] = [math.log2(d[k + 1] / d[k]) for k in range(len(d) - 1)]
            print(f"  {axis:14s} {shape:11s} " + "  ".join(f"{p:.2f}" for p in orders[(axis, shape)]))
    el = [p for s in SHAPES for p in orders[("element size", s)]]
    lo, hi = PAPER["orders_element_size"]
    good = round(min(el), 2) == lo and round(max(el), 2) == hi
    ok &= good
    print(f"\n  element size, both shapes: {min(el):.2f} to {max(el):.2f}   paper {lo} to {hi}  "
          f"{'OK' if good else 'MISMATCH'}")
    for shape in SHAPES:
        want = PAPER["orders_plane_spacing"][shape]
        got = tuple(round(p, 2) for p in orders[("plane spacing", shape)])
        good = got == want
        ok &= good
        print(f"  plane spacing, {shape:11s}: {got}   paper {want}  {'OK' if good else 'MISMATCH'}")

    print("\nRichardson-extrapolated error at the adopted setting, error = d / (1 - 2^-p)\n")
    print(f"  {'axis':14s} {'shape':11s} {'d(h,h/2) %':>11} {'p=2':>10} {'local p':>10} {'p':>6}  paper")
    est = {}
    for axis, pre, lv in AXES:
        for shape in SHAPES:
            d = successive(pre, lv, shape)
            k = [l[0] for l in lv].index(ADOPTED[axis]) - 1   # the pair (adopted/2, adopted)
            nominal = d[k] / 0.75
            p_loc = orders[(axis, shape)][k] if k < len(orders[(axis, shape)]) else 2.0
            est[(axis, shape)] = nominal
            want = PAPER["richardson_pct"][(axis, shape)]
            good = f"{nominal:.1e}" == f"{want:.1e}"
            ok &= good
            print(f"  {axis:14s} {shape:11s} {d[k]:11.6f} {nominal:10.3e} "
                  f"{d[k] / (1 - 2 ** -p_loc):10.3e} {p_loc:6.2f}  {want:.1e} {'OK' if good else 'MISMATCH'}")

    print("\nHeadroom: chi of the method (Table 2) over the extrapolated error, same geometry\n")
    head = {key: PAPER["method_chi_pct"][key[1]] / e for key, e in est.items()}
    for (axis, shape), h in head.items():
        print(f"  {axis:14s} {shape:11s} {h:9.1f}x")
    worst_el = min(head[("element size", s)] for s in SHAPES)
    good = worst_el >= 10 ** 2.95
    ok &= good
    print(f"  element size: three orders of magnitude, {worst_el:.0f}x at worst  {'OK' if good else 'MISMATCH'}")
    worst_tr = min(head[("plane spacing", s)] for s in SHAPES)
    good = 30 <= worst_tr < 40
    ok &= good
    print(f"  plane spacing: at least thirty times, {worst_tr:.1f}x at worst  {'OK' if good else 'MISMATCH'}")

    # Midpoint quadrature over a cell of width dl integrates exp(i k x) with relative error
    # about (k dl)^2 / 24, with k the wavenumber at 3 f_c in the host (c = 3e8 m/s)
    k = 2 * math.pi * 3 * 100e6 * math.sqrt(9.0) / 3e8
    bound_pct = (k * 0.01) ** 2 / 24 * 100
    ratios = [bound_pct / est[("element size", s)] for s in SHAPES]
    good = f"{k:.1f}" == "18.8" and f"{bound_pct:.2f}" == "0.15" and all(18 <= r <= 32 for r in ratios)
    ok &= good
    print(f"\nQuadrature scale: k = {k:.2f} rad/m at 3 f_c, (k dl)^2/24 = {bound_pct:.3f} % at dl = 0.01 m,\n"
          f"  measured error {', '.join(f'{r:.0f}x' for r in ratios)} smaller (planar, corrugated)  "
          f"{'OK' if good else 'MISMATCH'}")

    write_csv(Path(a.out_csv))
    print(f"\nwrote {a.out_csv}")
    print("\nALL VALUES OF THE PAPER REPRODUCED" if ok else "\nMISMATCH: see above")
    return 0 if (ok or not a.check) else 1


if __name__ == "__main__":
    sys.exit(main())
