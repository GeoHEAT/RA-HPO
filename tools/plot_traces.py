"""
Single-trace comparisons between full-domain gprMax and RA-HPO.

For every case of Tables 2 and 3 (cases.py, groups "accuracy" and "roughness") the script
  1. reads the gprMax receiver trace (Ez), passes it through the synthesis window of the solver
     (band_match, as every reference trace and the exact trace), and subtracts the direct-wave run of the same model
     without the fracture, giving the scattered response (skipped for the infinite case,
     which has no borehole and whose direct wave has decayed below 0.2% of the reflection
     peak inside the evaluation window);
  2. reads the RA-HPO trace <case>.npz;
  3. computes the metrics over the evaluation window TIME_RANGE with compare_traces;
  4. draws the comparison figure of the cases that have one, and writes all metrics to
     metrics/trace_metrics.csv.

Metrics (trace 1 = gprMax reference d, trace 2 = RA-HPO p; both linearly interpolated onto a
common uniform grid of N samples spanning TIME_RANGE):
  NRMSE      = sqrt( (1/N) sum_k (d_k - p_k)^2 ) / ( max_k d_k - min_k d_k ) x 100 %
  Pearson r  = correlation coefficient of d and p
  Peak ratio = max_k |d_k| / max_k |p_k|        (> 1: RA-HPO underestimates the peak)
  Peak-time difference = t(argmax_k |p_k|) - t(argmax_k |d_k|)
  Relative misfit chi = ||p - d|| / ||d|| x 100 %   (independent of the window once it holds
                        the response)

Usage:  python tools/plot_traces.py
"""
from __future__ import annotations

import csv
import sys
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
from matplotlib import font_manager

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from rahpo.cases import (ANALYTIC, CASES, FIGURE_DIR, METRICS_DIR, TIME_RANGE, Case, fdtd_output,
                         trace_file)
from rahpo.traces import (band_match, compare_traces, load_analytic, load_gprmax_ez,
                          load_rahpo, load_scattered_gprmax)

METRICS_CSV = METRICS_DIR / "trace_metrics.csv"

# -----------------------------
# Plotting
# -----------------------------
def set_style():
    arial = Path("/System/Library/Fonts/Supplemental/Arial.ttf")
    if arial.exists():
        font_manager.fontManager.addfont(str(arial))
        mpl.rcParams["font.family"] = "Arial"
    mpl.rcParams.update({
        "font.size": FONT_SIZE,
        "axes.labelsize": FONT_SIZE,
        "xtick.labelsize": FONT_SIZE - 1,
        "ytick.labelsize": FONT_SIZE - 1,
        "legend.fontsize": FONT_SIZE - 1,
        "axes.linewidth": 0.6,
        "xtick.major.width": 0.6,
        "ytick.major.width": 0.6,
        "xtick.direction": "in",
        "ytick.direction": "in",
        "xtick.top": True,
        "ytick.right": True,
        "pdf.fonttype": 42,
        "savefig.dpi": 300,
    })


def plot_case(case: Case, series: list[tuple[str, str, np.ndarray, np.ndarray]], metrics: dict):
    fig, ax = plt.subplots(figsize=FIGSIZE, constrained_layout=True)
    for kind, label, t, a in series:
        # Plot only the evaluation window, so that arrivals outside it (e.g. the direct wave
        # of the borehole-free model) do not set the vertical scale
        mask = (t >= TIME_RANGE[0]) & (t <= TIME_RANGE[1])
        y = a[mask] / np.max(np.abs(a[mask])) if case.normalized else a[mask]
        ax.plot(t[mask], y, color=COLORS[kind], linestyle=LINESTYLES[kind], linewidth=1.0, label=label)

    ax.set_xlim(*TIME_RANGE)
    ax.set_xlabel("Time (ns)")
    ax.set_ylabel(r"Normalized $E_z$" if case.normalized else r"$E_z$ (V/m)")
    ax.grid(True, linestyle="--", linewidth=0.4, alpha=0.6)
    ax.legend(loc="upper right", frameon=True, edgecolor="gray", framealpha=0.9)
    ax.text(0.02, 0.96,
            f"$\\chi$ = {metrics['Relative misfit (%)']:.1f}%\n"
            f"$r$ = {metrics['Pearson r']:.4f}\n"
            f"Peak ratio = {metrics['Peak ratio (ref/RA-HPO)']:.2f}",
            transform=ax.transAxes, va="top", ha="left", fontsize=FONT_SIZE - 1,
            bbox=dict(boxstyle="round,pad=0.3", facecolor="wheat", alpha=0.8, linewidth=0.5))

    FIGURE_DIR.mkdir(parents=True, exist_ok=True)
    fig.savefig(FIGURE_DIR / f"{case.figure}.pdf")
    plt.close(fig)


# -----------------------------
# Main
# -----------------------------
def main():
    set_style()
    rows = []
    for case in [c for c in CASES if c.group in ("accuracy", "roughness")]:
        needed = [fdtd_output(case.fdtd), trace_file(case.key)]
        if case.direct:
            needed.append(fdtd_output(case.direct))
        if case.analytic:
            needed.append(ANALYTIC / case.analytic)
        missing = [p.name for p in needed if not p.exists()]
        if missing:
            print(f"[skip] {case.key}: missing {', '.join(missing)}")
            continue

        t_ref, d_ref = load_scattered_gprmax(case)
        t_ra, d_ra = load_rahpo(case.key)
        metrics = compare_traces(t_ref, d_ref, t_ra, d_ra, time_range=TIME_RANGE)
        row = {"case": case.key, "label": case.label, "reference": "FDTD",
               "window (ns)": f"{TIME_RANGE[0]:g}-{TIME_RANGE[1]:g}", **metrics}
        rows.append(row)
        print(f"{case.key:22s} chi={metrics['Relative misfit (%)']:6.2f}%  "
              f"NRMSE={metrics['NRMSE (%)']:5.2f}%  r={metrics['Pearson r']:.4f}  "
              f"peak ratio={metrics['Peak ratio (ref/RA-HPO)']:.3f}  "
              f"dt_peak={metrics['Peak-time diff (ns)']:+.3f} ns")

        if case.figure:
            series = []
            if case.analytic:
                series.append(("analytic", "Analytic", *load_analytic(case.analytic)))
            series += [("gprmax", "gprMax", t_ref, d_ref), ("rahpo", "RA-HPO", t_ra, d_ra)]
            plot_case(case, series, metrics)

    if not rows:
        print("no case has both its FDTD output and its RA-HPO trace; nothing written")
        return
    METRICS_CSV.parent.mkdir(parents=True, exist_ok=True)
    with open(METRICS_CSV, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        for row in rows:
            writer.writerow({k: (f"{v:.6g}" if isinstance(v, float) else v) for k, v in row.items()})
    print(f"Wrote {len(rows)} rows to {METRICS_CSV}")


if __name__ == "__main__":
    main()
