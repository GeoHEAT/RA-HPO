"""
Geometry panels of the trace figures: fracture surface, reference plane, borehole,
transmitter and receiver positions and, for the antenna case, the antenna arms.

The panels are drawn in the style of the trace panels of plot_traces.py (8 pt Arial, no
title, vector PDF), so that a geometry panel and its trace panel match side by side. They are
written to output/figures/.

Fracture surfaces follow ra_hpo.fracture_surface_function, which evaluates x = f(y, z) in
local coordinates (y - ymin, z - zmin) on the [1, 9] m x [1, 9] m generation window.

Usage:
    python plot_surface.py                 # all panels
    python plot_surface.py finite_sinusoidal antenna_finite_borehole_fractures
"""
import sys
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib import font_manager
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from examples.cases import FIGURE_DIR as FIG_DIR
from rahpo.fracture import fracture_surface_function
from rahpo.parameters import GeometryParams

FIGSIZE = (3.4, 3.0)   # inches, before tight cropping (about 2.5 in wide after cropping)
FONT_SIZE = 8


# --- 1. Scene parameters (match the gprMax models and ra_hpo.GeometryParams) ---
@dataclass
class Scene:
    # Fracture generation window and base position
    ymin: float = 1.0
    ymax: float = 9.0
    zmin: float = 1.0
    zmax: float = 9.0
    x_base: float = 6.5
    y_center: float = 5.0
    z_center: float = 5.0
    # Reference plane
    x_plane: float = 3.5
    # Borehole
    x_bh: float = 0.5
    y_bh: float = 5.0
    r_bh: float = 0.075
    # Transmitter and receiver feed points
    z_tx: float = 5.5
    z_rx: float = 4.5
    # Antenna arms (gprMax PEC cylinders): arm length and half feed gap
    l_arm: float = 0.25
    halfgap: float = 0.01


# name: output basename; kind: ra_hpo fracture_type; extent: (y0, y1, z0, z1) of the drawn
# fracture; borehole / arms: draw the borehole cylinder / antenna arms
PANELS = [
    dict(name="infinite_vertical_planar", kind="planar", extent=(0.0, 10.0, 0.0, 10.0),
         borehole=False, arms=False),
    dict(name="finite_vertical_planar", kind="planar", extent=None, borehole=True, arms=False),
    dict(name="finite_inclined8deg_planar", kind="planar-tilted", extent=None, borehole=True, arms=False),
    dict(name="finite_parabolicbowl", kind="piecewise_yz", extent=None, borehole=True, arms=False),
    dict(name="finite_sinusoidal", kind="sinusoidal_3d", extent=None, borehole=True, arms=False),
    dict(name="antenna_finite_borehole_fractures", kind="planar", extent=None, borehole=True, arms=True),
]


# The drawn surface is the solver's surface, not a second copy of the formulas.
def fracture_surface(kind, Y, Z, s: Scene):
    """x = f(y, z) in local coordinates Y = y - ymin, Z = z - zmin."""
    geom = GeometryParams(
        fracture_type=kind, amplitude=0.25,
        ymin=s.ymin, ymax=s.ymax, zmin=s.zmin, zmax=s.zmax,
        y_center=s.y_center, z_center=s.z_center, x_base=s.x_base,
    )
    return fracture_surface_function(Y, Z, geom)


def create_cylinder(p0, p1, radius, num_theta=48, num_z=24):
    """Cylinder surface mesh between points p0 and p1."""
    p0 = np.array(p0, dtype=float)
    p1 = np.array(p1, dtype=float)
    v = p1 - p0
    L = np.linalg.norm(v)
    v_hat = v / L
    not_v = np.array([1.0, 0.0, 0.0])
    if np.allclose(np.abs(np.dot(v_hat, not_v)), 1.0):
        not_v = np.array([0.0, 1.0, 0.0])
    u = np.cross(v_hat, not_v)
    u /= np.linalg.norm(u)
    w = np.cross(v_hat, u)
    theta, z_local = np.meshgrid(np.linspace(0, 2 * np.pi, num_theta), np.linspace(0, L, num_z))
    X = p0[0] + v_hat[0] * z_local + radius * (u[0] * np.cos(theta) + w[0] * np.sin(theta))
    Y = p0[1] + v_hat[1] * z_local + radius * (u[1] * np.cos(theta) + w[1] * np.sin(theta))
    Z = p0[2] + v_hat[2] * z_local + radius * (u[2] * np.cos(theta) + w[2] * np.sin(theta))
    return X, Y, Z


# --- 3. Style and drawing ---
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
        "legend.fontsize": FONT_SIZE - 2,
        "axes.linewidth": 0.6,
        "grid.linewidth": 0.4,
        "pdf.fonttype": 42,
        "savefig.dpi": 300,
    })


def draw_panel(panel, s: Scene = Scene()):
    fig = plt.figure(figsize=FIGSIZE)
    ax = fig.add_subplot(111, projection="3d")

    # Fracture surface
    y0, y1, z0, z1 = panel["extent"] or (s.ymin, s.ymax, s.zmin, s.zmax)
    Y, Z = np.meshgrid(np.linspace(y0, y1, 81), np.linspace(z0, z1, 81))
    X = fracture_surface(panel["kind"], Y - s.ymin, Z - s.zmin, s)
    ax.plot_surface(X, Y, Z, color="tab:blue", rstride=4, cstride=4, shade=False,
                    linewidth=0.2, edgecolor="white", antialiased=True, label="fracture")

    # Reference plane
    Yr, Zr = np.meshgrid(np.linspace(1.0, 9.0, 2), np.linspace(1.0, 9.0, 2))
    ax.plot_surface(np.full_like(Yr, s.x_plane), Yr, Zr, color="lightgray", alpha=0.5,
                    linewidth=0, shade=False, label="reference plane")

    # Transmitter and receiver
    ax.scatter(s.x_bh, s.y_bh, s.z_tx, c="r", marker="x", s=10, linewidths=0.8, label="Tx")
    ax.scatter(s.x_bh, s.y_bh, s.z_rx, c="k", marker="*", s=10, linewidths=0.5, label="Rx")

    # Borehole
    if panel["borehole"]:
        Xc, Yc, Zc = create_cylinder([s.x_bh, s.y_bh, 0.0], [s.x_bh, s.y_bh, 10.0], s.r_bh)
        ax.plot_surface(Xc, Yc, Zc, color="brown", alpha=0.3, linewidth=0, shade=False,
                        label="borehole")

    # Antenna arms: drawn as thick lines so that 0.25 m arms remain visible at print size
    if panel["arms"]:
        for z_feed, col, label in ((s.z_tx, "r", "Tx arms"), (s.z_rx, "k", "Rx arms")):
            for sign in (1.0, -1.0):
                za = z_feed + sign * s.halfgap
                zb = z_feed + sign * (s.halfgap + s.l_arm)
                ax.plot([s.x_bh, s.x_bh], [s.y_bh, s.y_bh], [za, zb], color=col, linewidth=2.0,
                        solid_capstyle="butt", label=label if sign > 0 else None)

    ax.set_xlim(0, 8)
    ax.set_ylim(0, 10)
    ax.set_zlim(0, 10)
    ax.set_xticks([0, 4, 8])
    ax.set_yticks([0, 5, 10])
    ax.set_zticks([0, 5, 10])
    ax.tick_params(axis="both", which="major", pad=-2)
    ax.set_xlabel("x (m)", labelpad=-6)
    ax.set_ylabel("y (m)", labelpad=-6)
    ax.set_zlabel("z (m)", labelpad=-7)
    ax.set_box_aspect((1, 1, 1))
    ax.legend(loc="upper left", bbox_to_anchor=(-0.02, 1.02), frameon=True, edgecolor="gray",
              framealpha=0.9, handlelength=1.2, borderpad=0.3, labelspacing=0.2)

    FIG_DIR.mkdir(parents=True, exist_ok=True)
    out = FIG_DIR / f"{panel['name']}.pdf"
    fig.savefig(out, bbox_inches="tight", pad_inches=0.02)
    plt.close(fig)
    print(f"Saved {out}")


def main(names=None):
    set_style()
    for panel in PANELS:
        if not names or panel["name"] in names:
            draw_panel(panel)


if __name__ == "__main__":
    main(sys.argv[1:])
