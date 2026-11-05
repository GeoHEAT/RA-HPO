"""
Multi-offset gather: accuracy of each receiver against its offset from the source.

Reads output/metrics/gather.csv (one row per receiver, written by tools/gather.py) and writes
output/figures/Gather_metrics.pdf: two panels sharing the offset axis, the relative misfit chi and the
peak ratio (reference/RA-HPO) of each of the sixteen receivers. NRMSE is left out, since its
pattern over the gather follows that of chi (ratio 0.103 to 0.108). It shows the trend
with offset that the wiggle panels of tools/gather.py cannot resolve at one amplitude scale.

With --dipping the same panels also show the fractures dipping by 20 and 30 degrees, read from
output/metrics/dipping.csv (written by tools/dip_study.py, or the file given): RA-HPO and the
calibrated dipole in both panels, at the receivers whose RA-HPO trace has full plane coverage
(completeness horizon outside the evaluation window). The chi axis is then logarithmic, since
the dipole's chi lies well above the RA-HPO range.

Usage:  python tools/gather_metrics.py [--dipping [dipping.csv]]
"""
from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
sys.path.insert(0, str(Path(__file__).resolve().parent))
from examples.cases import FIGURE_DIR, METRICS_DIR, TIME_RANGE  # noqa: E402
from plot_traces import FONT_SIZE, set_style  # noqa: E402

FIGURE = "Gather_metrics"
PANELS = (("chi_pct", r"$\chi$ (%)"), ("peak_ratio", "peak ratio"))
DIP_STYLE = {20: ("tab:blue", "s"), 30: ("tab:red", "^")}


def load(path: Path) -> dict[str, np.ndarray]:
    """Columns of gather.csv as arrays, sorted by offset."""
    with open(path) as fh:
        rows = sorted(csv.DictReader(fh), key=lambda r: float(r["offset_m"]))
    return {k: np.array([float(r[k]) for r in rows]) for k in ("offset_m", "chi_pct", "nrmse_pct", "peak_ratio")}


def load_dip(path: Path) -> dict[tuple[int, str], dict[str, np.ndarray]]:
    """Rows of dipping.csv at the receivers with full plane coverage, grouped by (dip, model),
    each sorted by offset, with the column names of gather.csv."""
    with open(path) as fh:
        rows = list(csv.DictReader(fh))
    full = {(int(float(r["dip_deg"])), float(r["z_r"])) for r in rows if r["compared"] == "RA-HPO"
            and not (TIME_RANGE[0] <= float(r["t_c"]) <= TIME_RANGE[1])}
    rows = [r for r in rows if (int(float(r["dip_deg"])), float(r["z_r"])) in full]
    cols = {"offset_m": "offset_m", "chi_pct": "chi", "nrmse_pct": "nrmse", "peak_ratio": "peak_ratio"}
    out = {}
    for key in sorted({(int(float(r["dip_deg"])), r["compared"]) for r in rows}):
        sel = sorted((r for r in rows if (int(float(r["dip_deg"])), r["compared"]) == key),
                     key=lambda r: float(r["offset_m"]))
        out[key] = {k: np.array([float(r[c]) for r in sel]) for k, c in cols.items()}
    return out


def plot_sides(ax, x, y, **kw):
    """One line per side of the source, since the gather has no zero offset."""
    for i, side in enumerate((x < 0, x > 0)):
        if side.any():
            ax.plot(x[side], y[side], **(kw if i == 0 else {k: v for k, v in kw.items() if k != "label"}))


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--dipping", type=Path, nargs="?", const=METRICS_DIR / "dipping.csv", default=None,
                    help="add the dipping fractures (default file: output/metrics/dipping.csv)")
    args = ap.parse_args()
    d = load(METRICS_DIR / "gather.csv")
    dips = load_dip(args.dipping) if args.dipping else {}
    set_style()
    fig, axes = plt.subplots(1, 2, figsize=(6.0, 3.0 if dips else 2.3), sharex=True)
    for ax, (k, label) in zip(axes, PANELS):
        plot_sides(ax, d["offset_m"], d[k], color="k", lw=0.8, marker="o", ms=3,
                   label=r"RA-HPO, $0^\circ$" if dips else None)
        for (dip, model), g in dips.items():
            color, marker = DIP_STYLE[dip]
            if model == "RA-HPO":
                plot_sides(ax, g["offset_m"], g[k], color=color, lw=0.8, marker=marker, ms=3,
                           label=rf"RA-HPO, ${dip}^\circ$")
            else:
                plot_sides(ax, g["offset_m"], g[k], color=color, lw=0.8, ls="--", marker=marker, ms=3, mfc="none",
                           label=rf"calibrated dipole, ${dip}^\circ$")
        if k == "peak_ratio":
            ax.axhline(1.0, color="0.6", lw=0.6, ls="--", zorder=0)
        if k == "chi_pct" and dips:
            ax.set_yscale("log")
            ax.set_yticks([2, 5, 10, 20, 50])
            ax.set_yticklabels(["2", "5", "10", "20", "50"])
            ax.set_ylim(2, 60)
        ax.set_ylabel(label)
        ax.set_xlabel("offset (m)")
        ax.set_xticks(np.arange(-4, 5, 2))
    if dips:
        # RA-HPO in the first row, each dipole below the RA-HPO entry of its dip (the legend fills
        # column by column, so the first column gets a blank second entry)
        h = dict(zip(*reversed(axes[1].get_legend_handles_labels())))
        blank = plt.Line2D([], [], linestyle="none")
        order = [r"RA-HPO, $0^\circ$", None]
        for dip in sorted({dip for dip, _ in dips}):
            order += [rf"RA-HPO, ${dip}^\circ$", rf"calibrated dipole, ${dip}^\circ$"]
        fig.legend([h.get(k, blank) if k else blank for k in order], [k if k in h else "" for k in order],
                   loc="upper center", ncol=len(order) // 2, fontsize=FONT_SIZE, frameon=False,
                   handlelength=2.4, columnspacing=1.4)
        fig.tight_layout(pad=0.4, w_pad=1.2, rect=(0, 0, 1, 0.82))
    else:
        fig.tight_layout(pad=0.4, w_pad=1.2)
    FIGURE_DIR.mkdir(parents=True, exist_ok=True)
    out = FIGURE_DIR / f"{FIGURE}.pdf"
    fig.savefig(out)
    plt.close(fig)
    print(f"wrote {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
