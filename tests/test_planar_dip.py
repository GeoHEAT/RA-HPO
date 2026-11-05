"""Test of the planar-dip fracture shape and of the guard before the reference plane.

`fracture_type='planar-dip'` is the plane x = x_base + (z - z_center) tan(dip), meshed like
the other shapes. Four checks:
  1. dip 0 gives the planar mesh bit for bit (all 11 rows of elempos);
  2. with the slope and anchor of planar-tilted (dip -atan(1/7.11537), x = 46.25/7.11537 m at
     z = 5 m) it gives the planar-tilted mesh to 1e-12 m. With the nominal dip of -8 degrees
     and x = 6.5 m at z = 5 m the two agree to 1.5e-5 m, because planar-tilted passes
     1.3e-5 m behind x = 6.5 m at the fracture center;
  3. at 30 degrees every element weight is ellength^2 / cos(30 deg), the normal is
     (cos 30, 0, -sin 30), and the front face lies half an aperture before the plane;
  4. elements at or in front of the reference plane are split off, and the completeness
     horizon is the earliest two-way time among the lost elements plus the source delay.

Run from the repository root:  python tests/test_planar_dip.py
"""
from __future__ import annotations

import sys
from dataclasses import replace
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from rahpo.constants import C0, FC  # noqa: E402
from rahpo.fracture import completeness_horizon, drop_elements_before_plane, get_fracture_mesh  # noqa: E402
from rahpo.parameters import GeometryParams, MediaParams  # noqa: E402

MEDIA = MediaParams()
BASE = GeometryParams(ellength=0.02)      # 8 x 8 m extent, coarser than the paper for speed


def mesh(geom):
    return get_fracture_mesh(geom, MEDIA)


def test_dip_zero_is_planar():
    a = mesh(replace(BASE, fracture_type="planar"))
    b = mesh(replace(BASE, fracture_type="planar-dip", dip_deg=0.0))
    assert np.array_equal(a, b), "dip 0 must reproduce the planar mesh bit for bit"


def check_matches_planar_tilted():
    t = mesh(replace(BASE, fracture_type="planar-tilted"))
    dip = -np.degrees(np.arctan(1.0 / 7.11537))
    d = mesh(replace(BASE, fracture_type="planar-dip", dip_deg=dip,
                     x_base=46.25 / 7.11537, z_center=5.0))
    np.testing.assert_allclose(d[0:3], t[0:3], rtol=0, atol=1e-12)     # positions
    np.testing.assert_allclose(d[3:6], t[3:6], rtol=0, atol=1e-12)     # normals
    np.testing.assert_allclose(d[6:], t[6:], rtol=1e-12)               # area and fill
    nominal = mesh(replace(BASE, fracture_type="planar-dip", dip_deg=-8.0))
    gap = float(np.max(np.abs(nominal[0] - t[0])))
    assert gap < 1.5e-5, gap
    return gap


def check_thirty_degrees():
    beta = np.radians(30.0)
    g = replace(BASE, fracture_type="planar-dip", dip_deg=30.0, x_base=4.83, z_center=3.0,
                zmin=1.04, zmax=7.28)
    e = mesh(g)
    np.testing.assert_allclose(e[6], g.ellength ** 2 / np.cos(beta), rtol=1e-12)
    np.testing.assert_allclose(e[3], np.cos(beta), rtol=0, atol=1e-12)
    np.testing.assert_allclose(e[4], 0.0, rtol=0, atol=1e-12)
    np.testing.assert_allclose(e[5], -np.sin(beta), rtol=0, atol=1e-12)
    # signed distance of the front face from the plane, along the unit normal
    dist = (e[0] - g.x_base) * np.cos(beta) - (e[2] - g.z_center) * np.sin(beta)
    np.testing.assert_allclose(dist, -g.planethick / 2, rtol=0, atol=1e-12)
    return e.shape[1]


def check_guard_and_horizon():
    elempos = np.zeros((11, 5))
    elempos[0] = [3.0, 3.5, 3.6, 6.5, 7.0]
    elempos[1] = 5.0
    elempos[2] = [5.0, 5.0, 5.0, 5.0, 6.0]
    kept, front = drop_elements_before_plane(elempos, 3.5)
    assert kept.shape == (11, 3) and np.array_equal(kept[0], [3.6, 6.5, 7.0])
    assert front.shape == (3, 2) and np.array_equal(front[0], [3.0, 3.5])
    src, rec = (0.5, 5.0, 5.5), (0.5, 5.0, 4.5)
    v = C0 / 3.0
    lost = np.array([[6.5], [5.0], [5.0]])
    expected = 2.0 * np.sqrt(36.0 + 0.25) / v + np.sqrt(2.0) / FC
    assert abs(completeness_horizon(lost, src, rec, v) - expected) <= 1e-15 * expected
    two = np.array([[6.5, 3.6], [5.0, 5.0], [5.0, 5.0]])           # the nearer element sets t_c
    near = (np.hypot(3.1, 0.5) * 2.0) / v + np.sqrt(2.0) / FC
    assert abs(completeness_horizon(two, src, rec, v) - near) <= 1e-15 * near
    assert completeness_horizon(np.zeros((3, 0)), src, rec, v) is None
    return expected


def test_matches_planar_tilted():
    check_matches_planar_tilted()


def test_thirty_degrees():
    check_thirty_degrees()


def test_guard_and_horizon():
    check_guard_and_horizon()


if __name__ == "__main__":
    test_dip_zero_is_planar()
    gap = check_matches_planar_tilted()
    n30 = check_thirty_degrees()
    t_c = check_guard_and_horizon()
    print(f"test_planar_dip: 4 checks passed (nominal 8 degree gap {gap:.2e} m, "
          f"{n30} elements at 30 degrees, t_c of the test element {t_c * 1e9:.2f} ns)")
