"""
Multi-offset gather: Supplementary Table S2 and the gather figure.

The finite planar fracture is computed for sixteen receivers on the borehole axis, every RA-HPO
trace from the one reference plane by axial shift (cases gather_z*), against one FDTD run that
records the same receivers (gprMax_models/For_gather/). The FDTD reference of each receiver is
the fractured run minus the no-fracture run at the same receiver.

Writes metrics/gather.csv (one row per receiver) and figures/Traces_gather.pdf: three panels
sharing both axes and one amplitude scale, the FDTD gather, the RA-HPO gather and their
difference, as wiggle traces against receiver depth over the evaluation window.

Usage:  python tools/gather.py [--no-figure]
"""
from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

import h5py
import numpy as np
import matplotlib.pyplot as plt

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from cases import (BY_KEY, FIGURE_DIR, GATHER_Z, METRICS_DIR, TIME_RANGE, Z_SOURCE,  # noqa: E402
                   fdtd_output)
from plot_traces import FONT_SIZE, compare_traces, load_rahpo, load_scattered_gprmax, set_style  # noqa: E402

FIGURE = "Traces_gather"


def key(z: float) -> str:
    return f"gather_z{str(z).replace('.', 'pt')}"


def check_positions() -> None:
    """The receivers of both gather runs must sit at GATHER_Z, in that order."""
    c = BY_KEY[key(GATHER_Z[0])]
    for model in (c.fdtd, c.direct):
        with h5py.File(fdtd_output(model), "r") as f:
            if len(f["/rxs"]) != len(GATHER_Z):
                raise ValueError(f"{model}: expected {len(GATHER_Z)} receivers, found {len(f['/rxs'])}")
            for i, z in enumerate(GATHER_Z, start=1):
                pz = float(f[f"/rxs/rx{i}"].attrs["Position"][2])
                if not np.isclose(pz, z):
                    raise ValueError(f"{model} rx{i} at z = {pz}, expected {z}")


def main() -> int:
    ap = argparse.ArgumentParser(description="Multi-offset gather: Table S2 and the figure.")
    ap.add_argument("--no-figure", action="store_true")
    a = ap.parse_args()
    check_positions()

    fdtd, rahpo, rows = {}, {}, []
    for i, z in enumerate(GATHER_Z, start=1):
        c = BY_KEY[key(z)]
        fdtd[z] = load_scattered_gprmax(c, rx=i)
        rahpo[z] = load_rahpo(c.key)
        m = compare_traces(*fdtd[z], *rahpo[z], time_range=TIME_RANGE)
        rows.append({"rx": i, "z_m": z, "offset_m": round(z - Z_SOURCE, 3),
                     "chi_pct": m["Relative misfit (%)"], "nrmse_pct": m["NRMSE (%)"],
                     "r": m["Pearson r"], "peak_ratio": m["Peak ratio (ref/RA-HPO)"],
                     "peak_dt_ns": m["Peak-time diff (ns)"]})

    METRICS_DIR.mkdir(parents=True, exist_ok=True)
    out_csv = METRICS_DIR / "gather.csv"
    with open(out_csv, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
        w.writeheader()
        for r in rows:
            w.writerow({k: (f"{v:.6g}" if isinstance(v, float) else v) for k, v in r.items()})
    print(f"{'offset':>7} {'chi %':>7} {'NRMSE %':>8} {'r':>8} {'peak ratio':>11} {'dt ns':>7}")
    for r in rows:
        print(f"{r['offset_m']:+7.1f} {r['chi_pct']:7.2f} {r['nrmse_pct']:8.3f} {r['r']:8.5f} "
              f"{r['peak_ratio']:11.4f} {r['peak_dt_ns']:7.3f}")
    print(f"wrote {out_csv}")
    if a.no_figure:
        return 0

    set_style()
    mf = max(np.max(np.abs(fdtd[z][1])) for z in GATHER_Z)
    gain = 0.42 * 0.5 / mf                        # wiggle gain for the 0.5 m receiver spacing
    fig, axes = plt.subplots(1, 3, figsize=(7.0, 3.3), sharex=True, sharey=True)
    for ax, which in zip(axes, ("FDTD", "RA-HPO", "FDTD $-$ RA-HPO")):
        for z in GATHER_Z:
            t_f, a_f = fdtd[z]
            t_r, a_r = rahpo[z]
            if which == "FDTD":
                t, s = t_f, a_f
            elif which == "RA-HPO":
                t, s = t_r, a_r
            else:   # difference on the FDTD time grid
                t, s = t_f, a_f - np.interp(t_f, t_r, a_r, left=0.0, right=0.0)
            m = (t >= TIME_RANGE[0]) & (t <= TIME_RANGE[1])
            x = z + gain * s[m]
            ax.plot(x, t[m], color="k", lw=0.5)
            ax.fill_betweenx(t[m], z, x, where=(x > z), color="k", alpha=0.65, lw=0)
        ax.set_title(which, fontsize=FONT_SIZE)
        ax.set_xlabel("receiver depth $z$ (m)")
        ax.set_ylim(TIME_RANGE[1], TIME_RANGE[0])
    axes[0].set_ylabel("time (ns)")
    for ax in axes:
        ax.plot([Z_SOURCE], [TIME_RANGE[0]], marker="v", color="crimson", ms=4, clip_on=False, zorder=5)
        ax.margins(x=0.04)
    fig.tight_layout(pad=0.4)
    FIGURE_DIR.mkdir(parents=True, exist_ok=True)
    out = FIGURE_DIR / f"{FIGURE}.pdf"
    fig.savefig(out)
    plt.close(fig)
    print(f"wrote {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
