"""Fracture surface and the specular point. No reference plane."""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from rahpo.fracture import fracture_surface_function, get_fracture_mesh, get_specular_point_numerical
from rahpo.parameters import GeometryParams, MediaParams


def test_planar_surface_is_flat():
    geom = GeometryParams(fracture_type="planar", x_base=6.5)
    y, z = np.meshgrid([0.0, 1.0, 4.0], [0.0, 2.0], indexing="ij")
    surface = fracture_surface_function(y, z, geom)
    assert np.all(surface == 6.5)


def test_tilted_surface_falls_as_z_increases():
    geom = GeometryParams(fracture_type="planar-tilted")
    z = np.array([0.0, 4.0])
    y = np.zeros_like(z)
    surface = fracture_surface_function(y, z, geom)
    assert surface[1] < surface[0]


def test_specular_point_of_the_paper_geometry():
    geom = GeometryParams(ellength=0.1, ymin=4.0, ymax=6.0, zmin=4.0, zmax=6.0, fracture_type="planar")
    elempos = get_fracture_mesh(geom, MediaParams())
    point, index = get_specular_point_numerical((0.5, 5.0, 5.5), (0.5, 5.0, 4.5), elempos)
    assert index >= 0
    np.testing.assert_allclose(point, (6.5, 5.0, 5.0), atol=geom.ellength / 2)
