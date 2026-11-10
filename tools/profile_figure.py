"""Figure of the constant-offset profile: scene, RA-HPO radargram, and its difference from the dipole.

(a) The three fractures of tools/profile_scene.py at true scale, with the source heights at which
each is valid shaded. (b) The RA-HPO traces along the profile. (c) RA-HPO minus the broadside-
calibrated dipole. Panels (b) and (c) have one row per fracture, from the highest down, each over
the source heights at which that fracture is valid, so the heights between the fractures are not
drawn; a dashed line joins each shaded band of (a) to its row. Each fracture is divided by the
largest |E_z| of its RA-HPO traces, in (b) and (c) alike, and labeled with its median peak against
fracture A in (b) and its median peak ratio of RA-HPO to the calibrated dipole in (c).

Reads the profile traces (profile_* cases, RAHPO_TRACE_DIR) and output/metrics/profile_peaks.csv
(written by python tools/profile_scene.py --sum --out output/metrics), and writes
output/figures/Profile_dipping.pdf.

    python tools/profile_figure.py [--peaks output/metrics/profile_peaks.csv]
"""
from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parent))

from examples.cases import FIGURE_DIR, METRICS_DIR  # noqa: E402

FIGURE = "Profile_dipping"


def radargram(items, window):
    """Stack traces (z, t, x) on the time grid of the first trace inside window [t0, t1] (ns),
    sorted by z. Returns (t, z, image[z, t])."""
    items = sorted(items, key=lambda it: it[0])
    dt = items[0][1][1] - items[0][1][0]
    n = int(np.floor((window[1] - window[0]) / dt + 1e-9)) + 1
    t = window[0] + np.arange(n) * dt
    img = np.array([np.interp(t, ti, xi, left=0.0, right=0.0) for _, ti, xi in items])
    return t, np.array([it[0] for it in items]), img


def relative_db(peaks, ref):
    """Median peak of each patch against that of patch ref, in dB."""
    m = {k: float(np.median(v)) for k, v in peaks.items()}
    return {k: 0.0 if k == ref else 20 * np.log10(v / m[ref]) for k, v in m.items()}


def panel(ax, letter):
    ax.text(0.01, 0.98, f"({letter})", transform=ax.transAxes, va="top", ha="left", fontweight="bold")


def scored_window(zs, step):
    """Source heights a row spans: half a step beyond the first and the last scored position."""
    return min(zs) - step / 2, max(zs) + step / 2


def row_order(scene):
    """Rows of panels (b) and (c), from the highest fracture down, as in panel (a)."""
    return sorted(scene, key=lambda p: -p.z_hi)


def tick_alignment(ticks, window):
    """Horizontal alignment of the time labels: a label on an edge of the window is set inside the panel,
    so the labels of panels (b) and (c), which meet at that edge, do not run together."""
    return ["left" if t == window[0] else "right" if t == window[1] else "center" for t in ticks]


def profile_images(pk, ds, ps, name, window):
    """Radargrams (RA-HPO, calibrated dipole) of one patch on the window (t0, t1) in ns."""
    patch = next(p for p in ps.SCENE if p.name == name)
    kb, kn = ps.calibration_names()[name]
    W = ds.calibration_filter(*ds.pair(kb, kn))
    rah, dip = [], []
    for z in ps.positions(patch):
        t, q, p = ds.pair(ps.case_name(name, z, "bh"), ps.case_name(name, z, "nobh"))
        rah.append((z, t, q)); dip.append((z, t, ds.apply_filter(W, p)))
    t, zs, R = radargram(rah, window)
    _, _, Dp = radargram(dip, window)
    return t, zs, R, Dp


def fig_profile(pk, ds, plt, ps, out, window=(100.0, 250.0)):
    """(a) scene, (b) RA-HPO radargram, (c) RA-HPO minus the calibrated dipole, along the profile.
    Panels (b) and (c) have one row per patch, over its own scored source heights. Each patch is shown
    on its own gain (its RA-HPO maximum), with its median peak against patch A in dB; (c) uses the gain
    of (b), so it shows the dipole's error as a fraction of the reflection."""
    from matplotlib.gridspec import GridSpec, GridSpecFromSubplotSpec
    from matplotlib.patches import ConnectionPatch, Rectangle
    db = relative_db({p.name: [float(r["peak_bh"]) for r in pk if r["patch"] == p.name] for p in ps.SCENE}, "A")
    ratio = {p.name: float(np.median([float(r["ratio"]) for r in pk if r["patch"] == p.name])) for p in ps.SCENE}
    fill = {ps.BRINE: ("#1b7f79", "brine"), ps.FRESH: ("#7fb8d8", "fresh\nwater")}
    fig = plt.figure(figsize=(7.2, 7.6))
    gs = GridSpec(1, 5, figure=fig, width_ratios=(0.72, 0.13, 1.3, 1.3, 0.05), wspace=0.12,
                  left=0.07, right=0.925, bottom=0.07, top=0.965)
    a0, cax = fig.add_subplot(gs[0]), fig.add_subplot(gs[4])
    rows_b = GridSpecFromSubplotSpec(len(ps.SCENE), 1, subplot_spec=gs[2], hspace=0.12)
    rows_c = GridSpecFromSubplotSpec(len(ps.SCENE), 1, subplot_spec=gs[3], hspace=0.12)
    xmax = 15.0
    zmax = max(p.z_hi for p in ps.SCENE) + 8
    a0.add_patch(Rectangle((0.425, 0), 0.15, zmax, color="#9ecae1", lw=0))
    for p in ps.SCENE:
        c, lab = fill[p.fill]
        z = np.array(ps.positions(p))       # contiguous on the 0.25 m grid
        a0.axhspan(*scored_window(z, ps.STEP), color="0.93", lw=0, zorder=0)
        a0.plot([p.x_at(p.z_lo), p.x_at(p.z_hi)], [p.z_lo, p.z_hi], color=c, lw=2.2, solid_capstyle="butt")
        a0.plot([0.5, 0.5], [z.min(), z.max()], color="k", lw=2.0, solid_capstyle="butt")
        a0.text(float(p.x_at(p.z_hi)) + 0.5, (p.z_lo + p.z_hi) / 2, f"{p.name}\n{p.dip:g}°\n{lab}",
                fontsize=7.5, va="center")
    a0.set_xlim(-0.5, xmax); a0.set_ylim(0, zmax); a0.set_aspect("equal", adjustable="box", anchor="C")
    a0.set_xlabel("$x$ (m)"); a0.set_ylabel("Source height (m)")
    a0.set_title("(a)", loc="left", fontweight="bold", fontsize=10, pad=3)
    im = None
    for i, p in enumerate(row_order(ps.SCENE)):
        t, zs, R, Dp = profile_images(pk, ds, ps, p.name, window)
        v = np.max(np.abs(R))
        lo, hi = scored_window(zs, ps.STEP)
        axb = fig.add_subplot(rows_b[i])
        axc = fig.add_subplot(rows_c[i], sharey=axb)
        for ax, img in ((axb, R), (axc, R - Dp)):
            im = ax.imshow(img / v, aspect="auto", extent=(t[0], t[-1], lo, hi), origin="lower", cmap="seismic",
                           vmin=-1, vmax=1, interpolation="bilinear")
            ticks = [100, 150, 200, 250]
            ax.set_xlim(*window); ax.set_xticks(ticks)
            for lab, ha in zip(ax.get_xticklabels(), tick_alignment(ticks, window)):
                lab.set_horizontalalignment(ha)
            if i < len(ps.SCENE) - 1:
                plt.setp(ax.get_xticklabels(), visible=False)
            else:
                ax.set_xlabel("Time (ns)")
        plt.setp(axc.get_yticklabels(), visible=False)
        if i == 1:
            axb.set_ylabel("Source height (m)")
        if i == 0:
            for ax, letter in ((axb, "b"), (axc, "c")):
                ax.set_title(f"({letter})", loc="left", fontweight="bold", fontsize=10, pad=3)
        # an empty corner of each row: upper left for a dipping patch (early times at the highest
        # sources), upper right for the parallel one (late times); the patch name in the opposite one
        tx, ha = (window[0] + 3, "left") if p.dip else (window[1] - 3, "right")
        axb.text(tx, zs[-1] - 0.3, "0 dB (reference)" if p.name == "A" else f"{db[p.name]:+.1f} dB",
                 ha=ha, va="top", fontsize=8)
        axc.text(tx, zs[-1] - 0.3, f"peak ratio {ratio[p.name]:.3f}", ha=ha, va="top", fontsize=8)
        axb.text(window[1] - 3 if p.dip else window[0] + 3, zs[0] + 0.3, p.name,
                 ha="right" if p.dip else "left", va="bottom", fontsize=9, fontweight="bold")
        for zz in (lo, hi):                 # join the shaded band of (a) to its row
            fig.add_artist(ConnectionPatch(xyA=(xmax, zz), coordsA=a0.transData, xyB=(window[0], zz),
                                           coordsB=axb.transData, color="0.65", lw=0.6, ls=(0, (2, 2))))
    # each patch divided by the largest |E_z| of its RA-HPO traces
    fig.colorbar(im, cax=cax, label="normalized $E_z$")
    fig.savefig(out)
    plt.close(fig)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--peaks", type=Path, default=METRICS_DIR / "profile_peaks.csv")
    a = ap.parse_args()
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import dip_study as ds
    import profile_scene as ps
    from plot_traces import set_style
    set_style()
    with open(a.peaks) as f:
        pk = list(csv.DictReader(f))
    FIGURE_DIR.mkdir(parents=True, exist_ok=True)
    out = FIGURE_DIR / f"{FIGURE}.pdf"
    fig_profile(pk, ds, plt, ps, out)
    print(f"wrote {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
