"""Test of the element weight (`--element_area`).

With `element_area='true'` (the default) each fracture element is weighted by the area of the
surface above its cell, ellength^2 * sqrt(1 + f_y^2 + f_z^2); with 'projected' by the cell
itself, ellength^2. Four checks:
  1. planar fracture: the true weight equals ellength^2 exactly, so both settings give the same
     trace bit for bit;
  2. planar-tilted fracture (8 degrees): the true weight is ellength^2 / cos(8 deg) at every
     element, so the trace scales by exactly sec(8 deg) between the two settings;
  3. biaxial corrugation, A = 0.25 m: the true weight matches the analytic surface element
     sqrt(1 + |grad f|^2) to the accuracy of the finite-difference gradient in the interior, and
     its mean excess over the projected cell is about 2.9 %;
  4. 'projected' gives ellength^2 everywhere, and an unknown mode is refused.

Run from the repository root:  python tests/test_element_area.py
"""
from __future__ import annotations

import sys
from dataclasses import replace
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from rahpo.fracture import get_fracture_mesh
from rahpo.parameters import GeometryParams, MediaParams

MEDIA = MediaParams()
BASE = GeometryParams()          # ellength 0.01 m, 8 x 8 m extent, as every bounded case


def area(geom):
    return get_fracture_mesh(geom, MEDIA)[6, :]


def test_planar_unchanged():
    g = replace(BASE, fracture_type="planar")
    a_true = area(replace(g, element_area="true"))
    a_proj = area(replace(g, element_area="projected"))
    assert np.array_equal(a_true, a_proj), "planar weights must be bit-identical"
    assert np.all(a_true == g.ellength ** 2)


def test_tilted_is_secant():
    g = replace(BASE, fracture_type="planar-tilted")
    a_true = area(replace(g, element_area="true"))
    sec = np.sqrt(1.0 + (1.0 / 7.11537) ** 2)          # slope of x = (z - 50.25) / -7.11537
    assert abs(np.degrees(np.arctan(1.0 / 7.11537)) - 8.0) < 1e-4
    np.testing.assert_allclose(a_true, g.ellength ** 2 * sec, rtol=1e-12)


def test_corrugation_matches_analytic():
    A = 0.25
    g = replace(BASE, fracture_type="sinusoidal_3d", amplitude=A)
    a_true = area(replace(g, element_area="true")).reshape(800, 800)
    # analytic gradient of the surface, in local coordinates
    ky, kz = np.pi / 2, 3 * np.pi / 8
    y = np.arange(g.ymin, g.ymax, g.ellength) + g.ellength / 2 - g.ymin
    z = np.arange(g.zmin, g.zmax, g.ellength) + g.ellength / 2 - g.zmin
    Y, Z = np.meshgrid(y, z, indexing="ij")
    fy = A * ky * np.cos(ky * Y) * np.sin(kz * Z)
    fz = A * kz * np.sin(ky * Y) * np.cos(kz * Z)
    exact = g.ellength ** 2 * np.sqrt(1 + fy ** 2 + fz ** 2)
    inner = (slice(1, -1), slice(1, -1))               # np.gradient is one-sided on the edges
    np.testing.assert_allclose(a_true[inner], exact[inner], rtol=1e-4)
    excess = a_true.mean() / g.ellength ** 2 - 1
    assert 0.028 < excess < 0.031, excess


def test_projected_and_unknown():
    g = replace(BASE, fracture_type="sinusoidal_3d", amplitude=0.25, element_area="projected")
    assert np.all(area(g) == g.ellength ** 2)
    try:
        area(replace(g, element_area="bogus"))
    except ValueError:
        return
    raise AssertionError("an unknown element_area must raise")


if __name__ == "__main__":
    test_planar_unchanged()
    test_tilted_is_secant()
    test_corrugation_matches_analytic()
    test_projected_and_unknown()
    print("test_element_area: 4 checks passed")
