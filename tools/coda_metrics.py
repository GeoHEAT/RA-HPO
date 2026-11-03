"""
Late-window metrics of the heterogeneous fractures.

For each correlation length, and for the homogeneous parallel fracture for comparison:
  1. the energy arriving after the main lobe, as a percentage of the main-lobe energy;
  2. the number of local maxima of |trace| above 5% and above 10% of the peak, over the window
     that follows the main lobe.
Both are computed for the FDTD reference and for RA-HPO. Writes metrics/coda_metrics.csv.

Usage:  python tools/coda_metrics.py [--out-csv FILE]
"""
from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from rahpo.cases import BY_KEY, METRICS_DIR  # noqa: E402
from plot_traces import load_rahpo, load_scattered_gprmax  # noqa: E402

# The main lobe and the window that follows it, where the parallel fracture's edge
# diffractions arrive (about 160 ns).
MAIN = (121.0, 151.0)
LATE = (151.0, 171.0)
COUNT = (121.0, 200.0)      # window for counting arrivals
THRESHOLDS = (0.05, 0.10)   # fractions of the peak amplitude

# case key -> label
ORDER = [("heter_Lc4", "L_c = 4.0 m"), ("heter_Lc1", "L_c = 1.0 m"),
         ("heter_Lc02", "L_c = 0.2 m"), ("finite_planar", "homogeneous, parallel")]


def measure(t: np.ndarray, a: np.ndarray) -> dict:
    """Late-window energy as a percentage of the main lobe, and arrival counts."""
    ref = (t >= 100.0) & (t <= 200.0)
    peak = float(np.abs(a[ref]).max())
    m = (t >= MAIN[0]) & (t <= MAIN[1])
    l = (t >= LATE[0]) & (t <= LATE[1])
    out = {"late_pct": 100.0 * np.trapezoid(a[l] ** 2, t[l]) / np.trapezoid(a[m] ** 2, t[m])}
    w = (t >= COUNT[0]) & (t <= COUNT[1])
    aa = np.abs(a[w])
    interior = (aa[1:-1] > aa[:-2]) & (aa[1:-1] > aa[2:])
    for thr in THRESHOLDS:
        out[f"n_above_{int(thr * 100)}pct"] = int(np.sum(interior & (aa[1:-1] > thr * peak)))
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out-csv", default=str(METRICS_DIR / "coda_metrics.csv"))
    a = ap.parse_args()

    rows = []
    print(f"main lobe {MAIN[0]:.0f}-{MAIN[1]:.0f} ns, late window {LATE[0]:.0f}-{LATE[1]:.0f} ns, "
          f"arrivals counted over {COUNT[0]:.0f}-{COUNT[1]:.0f} ns\n")
    print(f"{'case':24s}{'source':8s}{'late % of main':>15}{'n >5%':>7}{'n >10%':>8}")
    for key, label in ORDER:
        c = BY_KEY[key]
        t, d = load_scattered_gprmax(c)
        tr, ra = load_rahpo(c.key)
        row = {"case": label}
        for src, (tt, aa) in (("FDTD", (t, d)), ("RA-HPO", (tr, ra))):
            m = measure(tt, aa)
            print(f"{label if src == 'FDTD' else '':24s}{src:8s}{m['late_pct']:15.2f}"
                  f"{m['n_above_5pct']:7d}{m['n_above_10pct']:8d}")
            row[f"late_pct_{src}"] = round(m["late_pct"], 2)
            row[f"n_above_5pct_{src}"] = m["n_above_5pct"]
            row[f"n_above_10pct_{src}"] = m["n_above_10pct"]
        rows.append(row)

    Path(a.out_csv).parent.mkdir(parents=True, exist_ok=True)
    with open(a.out_csv, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)
    print(f"\nwrote {a.out_csv}")

    late = [r["late_pct_FDTD"] for r in rows]
    n5 = [r["n_above_5pct_FDTD"] for r in rows[:3]]
    print(f"late energy spans {min(late):.1f} to {max(late):.1f} % across all four cases, so it "
          f"does not separate heterogeneity from edge diffraction;")
    print(f"arrivals above 5 % of the peak go {n5} as L_c falls, against "
          f"{rows[3]['n_above_5pct_FDTD']} for the homogeneous fracture.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
