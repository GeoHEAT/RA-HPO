"""Test of tools/voxel_thickness.py (gprMax 3.1.7 voxel slabs act about one cell thicker).

Two checks:
  1. the tangential effective thickness of a slab of N cells is 5.00, 4.98, 4.87 and 4.78 cm
     for N = 4 at 0, 8, 20 and 30 degrees, 3.00 cm for N = 2 and 2.00 cm for N = 1;
  2. if gprMax 3.1.7 can be imported, its compiled builder (build_voxels_from_array, without
     averaging, as #geometry_objects_read uses it) fills exactly the edges the tool counts.

Run from the repository root:  python tests/test_voxel_thickness.py
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "tools"))

from voxel_thickness import (edge_counts, effective_thickness, gprmax_edge_counts,  # noqa: E402
                             slab_section)

# (cells, dip [deg], cell size [m]): tangential thickness [cm]
EXPECTED_CM = {(4, 0.0, 0.01): 5.00, (4, 8.0, 0.01): 4.98, (4, 20.0, 0.01): 4.87, (4, 21.0, 0.01): 4.86,
               (4, 30.0, 0.01): 4.78, (2, 0.0, 0.01): 3.00, (1, 0.0, 0.01): 2.00, (8, 0.0, 0.005): 4.50}


def test_table():
    for (cells, dip, h), cm in EXPECTED_CM.items():
        t = effective_thickness(cells, dip, h=h)["tangential"]
        assert round(100 * t, 2) == cm, (cells, dip, h, 100 * t)


def check_against_gprmax_builder():
    checked = 0
    for cells, dip, h in EXPECTED_CM:
        sec = slab_section(cells, dip, h=h)
        g = gprmax_edge_counts(sec)
        if g is None:
            return "gprMax not importable, builder check skipped"
        assert g == edge_counts(sec, margin=None)[:3], (cells, dip, g)
        checked += 1
    return f"builder check on {checked} slabs"


def test_against_gprmax_builder():
    check_against_gprmax_builder()


if __name__ == "__main__":
    test_table()
    note = check_against_gprmax_builder()
    print(f"test_voxel_thickness: 2 checks passed ({note})")
