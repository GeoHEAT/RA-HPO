"""
Fracture geometry volumes for the full-domain gprMax models.

Writes an HDF5 volume for gprMax's #geometry_objects_read: an int16 array in which the
fracture cells hold material 0 and every other cell -1 (left to the surrounding model). A cell
belongs to the fracture when its center lies within half the aperture of the fracture surface,
measured along the surface normal, so a curved or inclined fracture appears as a staircase on
the grid. Material 0 is defined in gprMax_models/materials.txt.

The volume is 8 x 8 x 8 m at 0.01 m (800^3 cells). The models place it at (0, 1, 1) m, so its
local y and z equal the solver's local fracture coordinates and the fracture center
(y, z) = (5, 5) m lies at local (4, 4) m.

Shapes (file names as the models in gprMax_models/ read them):

    planar          vertical plane x = 6.5 m
                    3d_domain8x8x8_frac_planar-<n>h_vertical_dx010.h5
    planar-tilted   plane inclined by 8 degrees, through x = 6.5 m at the fracture center
                    3d_domain8x8x8_frac_planar-<n>h_slope8deg_dx010.h5
    piecewise_yz    flat disc of radius 1.5 m inside a concave parabolic bowl
                    3d_domain8x8x8_frac_piecewise_yz-<n>h_cur3div18_inout1pt5_dx010.h5
    sinusoidal_3d   biaxial corrugation x = 6.5 + A sin(k_y y) sin(k_z z), periods 4 and 16/3 m
                    3d_domain8x8x8_frac_sinusoidal_3d-<n>h_A<tag>-ky1div2-kz2div3_dx010.h5

<n> is the aperture in cells. The surfaces are those of ra_hpo.fracture_surface_function.

Usage
-----
    python generate_geometry_h5.py --shape planar
    python generate_geometry_h5.py --shape sinusoidal_3d --amplitude 0.25
    python generate_geometry_h5.py --all --out-dir gprMax_models/For_full-domain_verifications/input_h5_geom

--all writes every volume read by the models of this repository. One volume needs about 6 GB
of memory.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import h5py
import numpy as np

H = 0.01            # cell size [m]
N = 800             # cells per axis
X_BASE = 6.5        # fracture plane position [m]
BACKGROUND_ID = -1
FRACTURE_ID = 0
# Every corrugation amplitude of the paper's roughness study [m]
AMPLITUDES = (0.0625, 0.125, 0.25, 0.30, 0.40, 0.50, 0.55, 0.60, 0.70)


def amplitude_tag(a: float) -> str:
    """0.25 -> '025', 0.0625 -> '00625', as in the model file names."""
    return f"{a:g}".replace(".", "")


def file_name(shape: str, cells: int, amplitude: float | None = None) -> str:
    stem = {
        "planar": f"planar-{cells}h_vertical",
        "planar-tilted": f"planar-{cells}h_slope8deg",
        "piecewise_yz": f"piecewise_yz-{cells}h_cur3div18_inout1pt5",
        "sinusoidal_3d": f"sinusoidal_3d-{cells}h_A{amplitude_tag(amplitude or 0.0)}-ky1div2-kz2div3",
    }[shape]
    return f"3d_domain8x8x8_frac_{stem}_dx{int(H * 1000):03d}.h5"


def distance_to_surface(shape: str, amplitude: float, n: int = N, h: float = H) -> np.ndarray:
    """Distance from every cell center to the fracture surface, shape (n, n, n).

    The coordinates are broadcast rather than expanded to full 3D arrays; the arithmetic per
    cell is the same.
    """
    c = (np.arange(n) + 0.5) * h
    X, Y, Z = np.meshgrid(c, c, c, indexing="ij", sparse=True)

    if shape in ("planar", "planar-tilted"):
        # Plane a x + b y + c z = d, with a unit normal after normalization
        if shape == "planar":
            a, b, cz, d = 1.0, 0.0, 0.0, X_BASE
        else:
            # x = (z - 50.25) / (-7.11537): slope 1/7.11537, i.e. 8 degrees, through x = 6.5 m
            # at local z = 4 m (the fracture center)
            a, b, cz, d = -7.11537, 0.0, -1.0, -50.25
        norm = np.sqrt(a**2 + b**2 + cz**2)
        a, b, cz, d = a / norm, b / norm, cz / norm, d / norm
        return np.abs(a * X + b * Y + cz * Z - d)

    if shape == "sinusoidal_3d":
        kz = 2 * np.pi / (n * h * 2 / 3)
        ky = 2 * np.pi / (n * h / 2)
        x_surf = amplitude * np.sin(kz * Z) * np.sin(ky * Y) + 0.0 * Z + 0.0 * Y + X_BASE
        dx_dz = amplitude * kz * np.cos(kz * Z) * np.sin(ky * Y) + 0.0
        dx_dy = amplitude * ky * np.sin(kz * Z) * np.cos(ky * Y) + 0.0
        norm = np.sqrt(1 + dx_dz**2 + dx_dy**2)
        return np.abs(X - x_surf) / norm

    if shape == "piecewise_yz":
        z0 = y0 = n * h / 2
        inner_radius = outer_radius = 1.5
        curvature = -3.0 / 18.0
        R = np.sqrt((Z - z0) ** 2 + (Y - y0) ** 2)          # shape (1, n, n)
        x_surf = np.full(R.shape, X_BASE)
        nrm = np.ones(R.shape)
        outer = R > outer_radius                             # parabolic bowl
        x_surf[outer] = (X_BASE - 0.1 * (outer_radius - inner_radius)) \
            + curvature * (R[outer] - outer_radius) ** 2
        nrm[outer] = np.sqrt(1 + (2 * curvature * (R[outer] - outer_radius)) ** 2)
        return np.abs(X - x_surf) / nrm

    raise ValueError(f"unknown shape: {shape}")


def build_volume(shape: str, cells: int, amplitude: float = 0.0, n: int = N,
                 h: float = H) -> np.ndarray:
    dist = distance_to_surface(shape, amplitude, n, h)
    data = np.full((n, n, n), BACKGROUND_ID, dtype=np.int16)
    data[dist <= cells * h / 2] = FRACTURE_ID
    return data


def write_volume(path: Path, data: np.ndarray, h: float = H) -> None:
    with h5py.File(path, "w") as f:
        f.attrs["dx_dy_dz"] = (h, h, h)
        f.create_dataset("data", data=data)
    n_frac = int(np.count_nonzero(data == FRACTURE_ID))
    print(f"wrote {path} ({n_frac} fracture cells, {100 * n_frac / data.size:.2f}% of the volume)")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--shape", choices=["planar", "planar-tilted", "piecewise_yz", "sinusoidal_3d"])
    ap.add_argument("--amplitude", type=float, default=0.25,
                    help="corrugation amplitude A in metres for sinusoidal_3d (default 0.25)")
    ap.add_argument("--cells", type=int, default=4,
                    help="aperture in cells (default 4, i.e. 0.04 m; the infinite planar case uses 2)")
    ap.add_argument("--all", action="store_true",
                    help="write every volume read by the models of this repository")
    ap.add_argument("--out-dir", default=".", help="output directory (default: current)")
    a = ap.parse_args()

    out = Path(a.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    if a.all:
        jobs = [("planar", 4, 0.0), ("planar", 2, 0.0), ("planar-tilted", 4, 0.0),
                ("piecewise_yz", 4, 0.0)] + [("sinusoidal_3d", 4, A) for A in AMPLITUDES]
    elif a.shape:
        jobs = [(a.shape, a.cells, a.amplitude)]
    else:
        ap.error("give --shape or --all")
    for shape, cells, amp in jobs:
        write_volume(out / file_name(shape, cells, amp if shape == "sinusoidal_3d" else None),
                     build_volume(shape, cells, amp))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
