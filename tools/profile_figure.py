"""Figure of the constant-offset profile: scene, RA-HPO radargram, and its difference from the dipole.

(a) The three fractures of tools/profile_scene.py at true scale, with the source heights at which
each is valid. (b) The RA-HPO traces along the profile. (c) RA-HPO minus the broadside-calibrated
dipole. Each fracture is divided by the largest |E_z| of its RA-HPO traces, in (b) and (c) alike,
and labeled with its median peak against fracture A in (b) and its median peak ratio of RA-HPO
to the calibrated dipole in (c).

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
    Each patch is shown on its own gain (its RA-HPO maximum), with its median peak against patch A
    in dB; (c) uses the gain of (b), so it shows the dipole's error as a fraction of the reflection."""
    from matplotlib.gridspec import GridSpec
    from matplotlib.patches import Rectangle
    db = relative_db({p.name: [float(r["peak_bh"]) for r in pk if r["patch"] == p.name] for p in ps.SCENE}, "A")
    ratio = {p.name: float(np.median([float(r["ratio"]) for r in pk if r["patch"] == p.name])) for p in ps.SCENE}
    fill = {ps.BRINE: ("#1b7f79", "brine"), ps.FRESH: ("#7fb8d8", "fresh\nwater")}
    fig = plt.figure(figsize=(7.2, 5.0))
    gs = GridSpec(1, 4, figure=fig, width_ratios=(1.0, 1.25, 1.25, 0.06), wspace=0.2,
                  left=0.07, right=0.93, bottom=0.10, top=0.97)
    a0 = fig.add_subplot(gs[0])
    a1, a2 = fig.add_subplot(gs[1], sharey=a0), fig.add_subplot(gs[2], sharey=a0)
    cax = fig.add_subplot(gs[3])
    zmax = max(p.z_hi for p in ps.SCENE) + 8
    a0.add_patch(Rectangle((0.425, 0), 0.15, zmax, color="#9ecae1", lw=0))
    for p in ps.SCENE:
        c, lab = fill[p.fill]
        a0.plot([p.x_at(p.z_lo), p.x_at(p.z_hi)], [p.z_lo, p.z_hi], color=c, lw=2.2, solid_capstyle="butt")
        z = np.array(ps.positions(p))       # contiguous on the 0.25 m grid
        a0.plot([0.5, 0.5], [z.min(), z.max()], color="k", lw=2.0, solid_capstyle="butt")
        a0.text(float(p.x_at(p.z_hi)) + 0.8, (p.z_lo + p.z_hi) / 2, f"{p.name}\n{p.dip:g}°\n{lab}",
                fontsize=8, va="center")
    a0.set_xlim(-0.5, 17); a0.set_ylim(0, zmax); a0.set_aspect("equal")
    a0.set_xlabel("$x$ (m)"); a0.set_ylabel("Source height (m)")
    for p in ps.SCENE:
        t, zs, R, Dp = profile_images(pk, ds, ps, p.name, window)
        v = np.max(np.abs(R))
        ext = (t[0], t[-1], zs[0] - ps.STEP / 2, zs[-1] + ps.STEP / 2)
        for ax, img in ((a1, R), (a2, R - Dp)):
            im = ax.imshow(img / v, aspect="auto", extent=ext, origin="lower", cmap="seismic", vmin=-1, vmax=1,
                           interpolation="bilinear")
        # an empty corner of each patch's block: upper left for a dipping patch (early times at the
        # highest sources), upper right for the parallel one (late times)
        tx, ha = (window[0] + 3, "left") if p.dip else (window[1] - 3, "right")
        a1.text(tx, zs[-1], "0 dB (reference)" if p.name == "A" else f"{db[p.name]:+.1f} dB",
                ha=ha, va="top", fontsize=8)
        a2.text(tx, zs[-1], f"peak ratio {ratio[p.name]:.3f}", ha=ha, va="top", fontsize=8)
    for ax, letter in ((a0, "a"), (a1, "b"), (a2, "c")):
        panel(ax, letter)
    for ax in (a1, a2):
        ax.set_xlim(*window); ax.set_xlabel("Time (ns)")
        plt.setp(ax.get_yticklabels(), visible=False)
    a1.set_ylim(0, zmax)
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
