"""Test of the planar-dip volumes of generate_geometry_h5.py.

Four checks, on small grids:
  1. at dip 0 the planar-dip volume equals the planar volume bit for bit;
  2. a one-cell sheet at x = 6.505 m has exactly one voxel per (y, z) column inside its window,
     while at x = 6.5 m, on a cell face, the rule selects two (a floating-point tie);
  3. a 4-cell slab at 30 degrees has 4 or 5 voxels per column, on average 4 / cos(30 deg), and
     every voxel lies inside the window and the x cut;
  4. its voxels follow the solver's planar-dip surface: the mean x of each column lies within
     half a cell of x = x_ref + (z - z_ref) tan(dip).

Run from the repository root:  python tests/test_geometry_dip.py
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "tools"))

from generate_geometry_h5 import FRACTURE_ID, build_planar_dip, build_volume  # noqa: E402

H = 0.01


def test_dip_zero_is_planar():
    n, h = 80, 0.1                       # 8 m cube at 0.1 m: the shapes are given in metres
    a = build_volume("planar", 4, n=n, h=h)
    b = build_planar_dip(0.0, 6.5, 5.0, 4, (-1e9, 1e9, -1e9, 1e9), dims=(n, n, n),
                         origin=(0.0, 0.0, 0.0), h=h)
    assert np.array_equal(a, b)


def test_one_voxel_sheet():
    dims, origin = (800, 4, 20), (0.0, 1.0, 4.9)
    v = build_planar_dip(0.0, 6.505, 5.0, 1, (4.9, 5.1, 1.0, 9.0), dims=dims, origin=origin)
    per_column = (v == FRACTURE_ID).sum(axis=0)
    assert np.all(per_column == 1), np.unique(per_column)
    assert np.all(np.nonzero(v == FRACTURE_ID)[0] == 650)
    tie = build_planar_dip(0.0, 6.5, 5.0, 1, (4.9, 5.1, 1.0, 9.0), dims=dims, origin=origin)
    assert np.all((tie == FRACTURE_ID).sum(axis=0) == 2)


def check_thirty_degrees():
    dip, x_ref, z_ref = 30.0, 4.83, 3.0
    window, xmax = (1.04, 7.28, 1.0, 1.04), 7.3
    dims, origin = (800, 4, 1000), (0.0, 1.0, 0.0)
    v = build_planar_dip(dip, x_ref, z_ref, 4, window, xmax, dims=dims, origin=origin)
    frac = v == FRACTURE_ID
    i, j, k = np.nonzero(frac)
    x = origin[0] + (i + 0.5) * H
    y = origin[1] + (j + 0.5) * H
    z = origin[2] + (k + 0.5) * H
    assert z.min() >= window[0] and z.max() <= window[1]
    assert y.min() >= window[2] and y.max() <= window[3]
    assert x.max() <= xmax
    col = frac[:, 0, :].sum(axis=0)
    col = col[col > 0][3:-3]                      # interior columns
    assert set(np.unique(col)) <= {4, 5}, np.unique(col)
    mean = col.mean()
    assert abs(mean - 4 / np.cos(np.radians(dip))) < 0.02, mean
    # column centroids against the solver surface
    zc = origin[2] + (np.arange(dims[2]) + 0.5) * H
    sec = frac[:, 0, :]
    has = sec.any(axis=0)
    xs = origin[0] + (np.arange(dims[0]) + 0.5) * H
    centroid = (sec * xs[:, None]).sum(axis=0)[has] / sec.sum(axis=0)[has]
    surface = x_ref + (zc[has] - z_ref) * np.tan(np.radians(dip))
    interior = slice(3, -3)
    assert np.max(np.abs(centroid - surface)[interior]) <= H / 2 + 1e-12
    return mean


def test_thirty_degrees():
    check_thirty_degrees()


if __name__ == "__main__":
    test_dip_zero_is_planar()
    test_one_voxel_sheet()
    m = check_thirty_degrees()
    print(f"test_geometry_dip: 4 checks passed (mean column count at 30 degrees {m:.3f}, "
          f"4/cos30 = {4 / np.cos(np.radians(30)):.3f})")
