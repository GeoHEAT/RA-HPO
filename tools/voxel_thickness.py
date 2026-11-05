"""Effective thickness of a voxel slab as gprMax 3.1.7 builds it.

gprMax 3.1.7 gives the material of a voxel to all twelve electric-field edges of its cell
(`build_voxel` in geometry_primitives_ext.pyx; `#geometry_objects_read` always builds without
averaging). Neighbouring cells share edges, so a slab of N cells across x carries the fill on
N + 1 planes of the tangential components Ey and Ez and on N planes of the normal component
Ex. For the field tangential to the slab it therefore acts as N + 1 cells: one voxel acts as
0.02 m at 0.01 m cells, and the 4-cell (0.04 m) fractures of the paper act as 0.05 m.

For a slab inclined by beta from the z axis, built by the rule of generate_geometry_h5.py (cell
centers within N h / 2 of the plane), the script counts the filled edges of each component in
the (x, z) section, divides their area by the length of the plane, and reports the effective
thickness per component and for the field tangential to the slab in the plane of incidence,
sin^2(beta) t_Ex + cos^2(beta) t_Ez. If gprMax 3.1.7 can be imported, the counts are also
taken from its compiled builder and must agree.

    python tools/voxel_thickness.py                 # the slabs of the paper's FDTD references
    python tools/voxel_thickness.py --cells 4 --dip 30
"""
from __future__ import annotations

import argparse
import sys

import numpy as np

H = 0.01


def slab_section(cells: int, dip: float, x0: float | None = None, z0: float = 5.0,
                 h: float = H, n: int | None = None) -> np.ndarray:
    """Boolean (x, z) section of the slab on an n x n grid of cell size h (default: 8 m)."""
    if n is None:
        n = int(round(8.0 / h))
    if x0 is None:
        x0 = 6.5 if cells % 2 == 0 else 6.505      # an odd slab is centered on a cell, not a face
    b = np.radians(dip)
    xc = (np.arange(n)[:, None] + 0.5) * h
    zc = (np.arange(n)[None, :] + 0.5) * h
    dist = (xc - x0) * np.cos(b) - (zc - z0) * np.sin(b)
    return np.abs(dist) <= cells * h / 2 + 1e-12


def edge_counts(section: np.ndarray, margin: int | None = 30) -> tuple[int, int, int, int]:
    """Filled Ex, Ey, Ez edges per y row of a slab uniform in y, and the number of z rows used.

    Ex edges of cell (i, k) lie at (i, k) and (i, k + 1) in the (x, z) section, Ez edges at
    (i, k) and (i + 1, k), Ey edges at the four corners. Rows within `margin` of the slab ends
    are left out, so the count measures the interior; margin None counts every cell.
    """
    iv, kv = np.nonzero(section)
    if margin is None:
        k1, k2 = int(kv.min()), int(kv.max()) + 1
    else:
        rows = np.nonzero(section.any(axis=0))[0]
        k1, k2 = rows[0] + margin, rows[-1] - margin
    keep = (kv >= k1) & (kv < k2)
    iv, kv = iv[keep], kv[keep]

    def count(offsets):
        filled = set()
        for di, dk in offsets:
            filled.update(zip((iv + di).tolist(), (kv + dk).tolist()))
        return len(filled)

    return (count([(0, 0), (0, 1)]), count([(0, 0), (1, 0), (0, 1), (1, 1)]),
            count([(0, 0), (1, 0)]), int(k2 - k1))


def effective_thickness(cells: int, dip: float, x0: float | None = None, h: float = H,
                        n: int | None = None) -> dict:
    """Effective thicknesses in metres: per component and tangential in the incidence plane."""
    nx, ny, nz, rows = edge_counts(slab_section(cells, dip, x0, h=h, n=n))
    b = np.radians(dip)
    length = rows * h / np.cos(b)                   # plane length covered by the counted rows
    t = {c: k * h * h / length for c, k in (("Ex", nx), ("Ey", ny), ("Ez", nz))}
    t["tangential"] = np.sin(b) ** 2 * t["Ex"] + np.cos(b) ** 2 * t["Ez"]
    return t


def gprmax_edge_counts(section: np.ndarray) -> tuple[int, int, int] | None:
    """Every Ex, Ey, Ez edge that gprMax's compiled builder fills in the middle y row, or None
    if gprMax cannot be imported. To be compared with edge_counts(section, margin=None)."""
    try:
        from gprMax.geometry_primitives_ext import build_voxels_from_array
    except ImportError:
        return None
    nxc, nzc = section.shape
    ny = 3                                           # the middle y row has no y-edge effects
    data = np.where(np.broadcast_to(section[:, None, :], (nxc, ny, nzc)), 0, -1).astype(np.int16)
    solid = np.zeros((nxc, ny, nzc), dtype=np.uint32)
    rigidE = np.zeros((12, nxc, ny, nzc), dtype=np.int8)
    rigidH = np.zeros((6, nxc, ny, nzc), dtype=np.int8)
    ID = np.zeros((6, nxc + 1, ny + 1, nzc + 1), dtype=np.uint32)
    fill = 7                                         # an ID that no other edge carries
    build_voxels_from_array(0, 0, 0, fill, False, np.ascontiguousarray(data), solid, rigidE, rigidH, ID)
    return tuple(int(np.count_nonzero(ID[c, :, 1, :] == fill)) for c in (0, 1, 2))


# (cells, dip [deg], cell size [m]): the slabs of the paper's FDTD references
TABLE = [(4, 0.0, H), (4, 8.0, H), (4, 20.0, H), (4, 21.0, H), (4, 30.0, H), (2, 0.0, H), (1, 0.0, H),
         (8, 0.0, 0.005)]


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--cells", type=int, help="slab thickness in cells")
    ap.add_argument("--dip", type=float, default=0.0, help="inclination from the z axis, degrees")
    ap.add_argument("--h", type=float, default=H, help="cell size in metres (default 0.01)")
    a = ap.parse_args()
    rows = [(a.cells, a.dip, a.h)] if a.cells else TABLE
    print(" N   dip      h    t_Ex   t_Ey   t_Ez  tangential  [cm]   nominal   gprMax 3.1.7 builder")
    ok = True
    for cells, dip, h in rows:
        t = effective_thickness(cells, dip, h=h)
        sec = slab_section(cells, dip, h=h)
        g = gprmax_edge_counts(sec)
        mine = edge_counts(sec, margin=None)[:3]
        check = "not importable" if g is None else ("same edges" if g == mine else f"DIFFERS {g} vs {mine}")
        ok &= g is None or g == mine
        print(f"{cells:2d} {dip:5.1f}  {h:5.3f}  {100 * t['Ex']:5.2f}  {100 * t['Ey']:5.2f}  {100 * t['Ez']:5.2f}"
              f"  {100 * t['tangential']:9.2f}         {100 * cells * h:5.1f}     {check}")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
