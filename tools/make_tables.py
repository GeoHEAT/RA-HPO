"""Rows of Tables 2 and 3 of the paper, printed as LaTeX from metrics/accuracy_metrics.csv.

Table 2 (accuracy): chi, NRMSE and peak ratio of the nine single traces, plus the range of the
Pearson correlation. Table 3 (roughness): the corrugation amplitude A, the peak-to-peak
amplitude 2A/lambda, the RMS slope m_rms = A sqrt(k_y^2 + k_z^2) / 2, the largest local slope
atan(A max(k_y, k_z)) in degrees, the smallest radius of curvature R_c = 1 / (A k_y^2) and k R_c,
with k_y = pi/2, k_z = 3 pi/8 and lambda = 1.0 m, then peak ratio, r, chi and NRMSE. Also prints
the roughness limit, the largest amplitude up to which chi stays below twice the finite planar
value, and the limit the same rule gives against the refined planar chi of
metrics/refined_reference.csv when that file exists.

Usage
-----
    python tools/make_tables.py [--csv metrics/accuracy_metrics.csv]
"""
from __future__ import annotations

import argparse
import csv
import math
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from rahpo.cases import METRICS_DIR  # noqa: E402
ACCURACY = [  # Table 2 label -> case name in the CSV
    ("Infinite planar", "Infinite planar"),
    ("Finite planar", "Finite planar"),
    ("Finite planar, antenna arms", "Finite planar, antenna arms"),
    (r"Inclined ($8^\circ$)", "Inclined (8 deg)"),
    ("Piecewise concave", "Piecewise concave"),
    (r"Biaxial corrugated ($A = 0.25$~m)", "Biaxial corrugated, A = 0.25 m"),
    (r"Heterogeneous, $L_c = 0.2$~m", "Heterogeneous, Lc = 0.2 m"),
    (r"Heterogeneous, $L_c = 1.0$~m", "Heterogeneous, Lc = 1.0 m"),
    (r"Heterogeneous, $L_c = 4.0$~m", "Heterogeneous, Lc = 4.0 m"),
]
AMPLITUDES = ["0.0625", "0.125", "0.25", "0.30", "0.40", "0.50", "0.55", "0.60", "0.70"]
KY, KZ, LAM = math.pi / 2, 3 * math.pi / 8, 1.0


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", default=str(METRICS_DIR / "accuracy_metrics.csv"))
    rows = {r["case"]: r for r in csv.DictReader(open(ap.parse_args().csv))}

    print("% Table 2 rows")
    rs = []
    for label, key in ACCURACY:
        r = rows[key]
        rs.append(float(r["r"]))
        print(f"{label:40} & {float(r['chi_pct']):.1f} & {float(r['nrmse_pct']):.2f} & "
              f"{float(r['peak_ratio']):.2f} \\\\")
    print(f"% correlation range {min(rs):.4f} to {max(rs):.4f}\n")

    print("% Table 3 rows")
    planar = float(rows["Finite planar"]["chi_pct"])
    limit = None
    for a in AMPLITUDES:
        A = float(a)
        r = rows[f"Biaxial corrugated, A = {A:g} m"]
        rc = 1 / (A * KY ** 2)
        m = 0.5 * A * math.hypot(KY, KZ)
        theta = math.degrees(math.atan(A * max(KY, KZ)))
        chi = float(r["chi_pct"])
        if chi < 2 * planar and limit in (None, AMPLITUDES[AMPLITUDES.index(a) - 1]):
            limit = a
        ratio = f"{2 * A / LAM:g}" if 2 * A / LAM != int(2 * A / LAM) else f"{2 * A / LAM:.1f}"
        print(f"{a:6} & {ratio} & {m:.2f} & {theta:.0f} & {rc:.2f} & {2 * math.pi / LAM * rc:.1f} & "
              f"{float(r['peak_ratio']):.2f} & {float(r['r']):.4f} & {chi:.1f} & {float(r['nrmse_pct']):.2f} \\\\")
    A = float(limit)
    m = 0.5 * A * math.hypot(KY, KZ)
    within = [rows[f"Biaxial corrugated, A = {float(x):g} m"] for x in AMPLITUDES[:AMPLITUDES.index(limit) + 1]]
    print(f"\n% limit: chi < 2 x {planar:.2f} = {2 * planar:.1f} % up to A = {limit} m (2A/lambda = {2 * A / LAM:g}), "
          f"k R_c = {2 * math.pi / LAM / (A * KY ** 2):.2f}, m_rms = {m:.3f} ({math.degrees(math.atan(m)):.1f} deg), "
          f"peak ratio at most {max(float(w['peak_ratio']) for w in within):.3f}, "
          f"largest local slope {math.degrees(math.atan(A * max(KY, KZ))):.1f} deg")
    refined = Path(ap.parse_args().csv).with_name("refined_reference.csv")
    if refined.exists():
        fine = next(float(x["chi_pct"]) for x in csv.DictReader(open(refined))
                    if x["comparison"] == "RA-HPO against the 0.005 m reference")
        alt = None
        for a in AMPLITUDES:
            if float(rows[f"Biaxial corrugated, A = {float(a):g} m"]["chi_pct"]) >= 2 * fine:
                break
            alt = a
        print(f"% against the refined planar chi: 2 x {fine:.2f} = {2 * fine:.1f} % holds up to A = {alt} m")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
