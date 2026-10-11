"""
Two-medium equivalent-source test.

The same finite planar fracture and antenna model is solved three times with RA-HPO, each
time from a reference plane recorded in a *different* host rock: epsilon_r = 6, 9 and 12.
In every case the exterior medium used for propagation and scattering is the true host of
the full-domain benchmark, epsilon_r = 9. The epsilon_r = 9 run is the matched case, in
which the two-medium relations reduce to the single-medium ones. The epsilon_r = 6 and 12 runs
exercise the de-embedding of the plane field from its recording medium and its re-radiation
into the host (ra_hpo.py --plane_epsr).

The FDTD reference, the direct-wave subtraction, the evaluation window and the metrics are
those of plot_traces.py, so the numbers are comparable with Table 2.

Cases (cases.py): finite_planar_arms (matched), two_medium_eps6, two_medium_eps12.
Outputs: figures/Traces_two_medium.pdf and metrics/trace_metrics_two_medium.csv.

Usage:  python plot_two_medium.py
"""
from __future__ import annotations

import csv
import sys
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt

sys.path.insert(0, str(Path(__file__).resolve().parent))
from cases import BY_KEY, FIGURE_DIR, METRICS_DIR, TIME_RANGE  # noqa: E402
from plot_traces import (FONT_SIZE, compare_traces, load_rahpo,  # noqa: E402
                         load_scattered_gprmax, set_style)

METRICS_CSV = METRICS_DIR / "trace_metrics_two_medium.csv"
FIGURE = "Traces_two_medium"

# Wider than the paired trace panels (0.58 of the text width) because this figure stands
# alone and carries four curves; fonts, ticks, grid and line weights are unchanged.
FIGSIZE = (4.6, 2.7)

# (permittivity of the host in which the reference plane was recorded, case, color, linestyle)
RUNS = [
    (9,  "finite_planar_arms", "#fe0101", "--"),
    (6,  "two_medium_eps6", "#ff7f0e", (0, (1, 1.4))),
    (12, "two_medium_eps12", "#2ca02c", (0, (4, 1.4, 1, 1.4))),
]


def main():
    set_style()

    t_ref, d_ref = load_scattered_gprmax(BY_KEY["finite_planar_arms"])

    fig, ax = plt.subplots(figsize=FIGSIZE, constrained_layout=True)
    mask_ref = (t_ref >= TIME_RANGE[0]) & (t_ref <= TIME_RANGE[1])
    ax.plot(t_ref[mask_ref], d_ref[mask_ref], color="#0800ff", linestyle="-", linewidth=1.0,
            label=r"gprMax (host $\epsilon_r = 9$)")

    rows, box = [], r"$\chi$ / $r$ / peak ratio" + "\n"
    for epsr, npz, color, linestyle in RUNS:
        t, a = load_rahpo(npz)
        m = compare_traces(t_ref, d_ref, t, a, time_range=TIME_RANGE)
        mask = (t >= TIME_RANGE[0]) & (t <= TIME_RANGE[1])
        matched = " (matched)" if epsr == 9 else ""
        ax.plot(t[mask], a[mask], color=color, linestyle=linestyle, linewidth=1.0,
                label=rf"RA-HPO, plane $\epsilon_r = {epsr}${matched}")
        box += (rf"$\epsilon_r = {epsr}$: {m['Relative misfit (%)']:.1f}%, {m['Pearson r']:.3f}, "
                rf"{m['Peak ratio (ref/RA-HPO)']:.2f}" + "\n")
        rows.append({"plane_epsr": epsr, "exterior_epsr": 9, "reference": "FDTD",
                     "window (ns)": f"{TIME_RANGE[0]:g}-{TIME_RANGE[1]:g}", **m})
        print(f"  plane eps_r={epsr:2d}  chi={m['Relative misfit (%)']:5.2f}%  NRMSE={m['NRMSE (%)']:6.2f}%  r={m['Pearson r']:.4f}  "
              f"peak ratio={m['Peak ratio (ref/RA-HPO)']:.3f}  "
              f"dt_peak={m['Peak-time diff (ns)']:+.3f} ns")

    ax.set_xlim(*TIME_RANGE)
    ax.set_xlabel("Time (ns)")
    ax.set_ylabel(r"$E_z$ (V/m)")
    ax.grid(True, linestyle="--", linewidth=0.4, alpha=0.6)
    ax.legend(loc="upper right", frameon=True, edgecolor="gray", framealpha=0.9)
    ax.text(0.02, 0.96, box.rstrip("\n"), transform=ax.transAxes, va="top", ha="left",
            fontsize=FONT_SIZE - 1,
            bbox=dict(boxstyle="round,pad=0.3", facecolor="wheat", alpha=0.8, linewidth=0.5))

    FIGURE_DIR.mkdir(parents=True, exist_ok=True)
    fig.savefig(FIGURE_DIR / f"{FIGURE}.pdf")
    plt.close(fig)
    print(f"Saved {FIGURE_DIR / (FIGURE + '.pdf')}")

    METRICS_CSV.parent.mkdir(parents=True, exist_ok=True)
    with open(METRICS_CSV, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        for row in rows:
            writer.writerow({k: (f"{v:.6g}" if isinstance(v, float) else v) for k, v in row.items()})
    print(f"Wrote {len(rows)} rows to {METRICS_CSV}")


if __name__ == "__main__":
    main()
