"""Test of the axial tool shift (`--tool_shift`).

Moving the source, the receiver and the recorded plane by +D along the borehole axis is the
same as moving the fracture by -D, for an axially uniform borehole and host. Three checks, on
the worked-example plane in examples/:
  1. with tool_shift = 0.0 the worked example reproduces its stored trace to a relative L2
     difference below 1e-15 (the stored trace is a cluster run; the default path is unchanged);
  2. a 10 degree planar-dip patch with the tool moved by +0.25 m gives the trace of the same
     patch moved by -0.25 m with the tool in place, to a relative L2 difference below 1e-12.
     The grid is binary-exact (element size 1/32 m), so the two meshes coincide exactly;
  3. a shift without no_viz is refused.

Run from the repository root:  python tests/test_tool_shift.py
"""
from __future__ import annotations

import logging
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from rahpo.parameters import GeometryParams, MediaParams, RuntimeParams, SourceParams  # noqa: E402
from rahpo.simulation import run_simulation  # noqa: E402

PLANE = str(ROOT / "examples" / "example_plane.h5")
REFERENCE = ROOT / "examples" / "example_reference.npz"
logging.disable(logging.CRITICAL)        # the solver logs every phase; the checks print a summary


def run(geom, shift=0.0, cores=2, **runtime):
    rt = RuntimeParams(tx_plane_file=PLANE, rx_plane_file=PLANE, cores=cores, no_viz=True,
                       tool_shift=shift, **runtime)
    return run_simulation(MediaParams(), SourceParams(), geom, rt)


def rel_l2(a, b):
    return float(np.linalg.norm(a - b) / np.linalg.norm(b))


def check_zero_shift_reproduces_example():
    geom = GeometryParams(fracture_type="planar", ymin=4.5, ymax=5.5, zmin=4.5, zmax=5.5)
    t, trace = run(geom, shift=0.0, cores=4)
    ref = np.load(REFERENCE)
    assert np.allclose(t * 1e9, ref["time_ns"], rtol=0, atol=1e-9)
    err = rel_l2(trace, ref["amplitude"])
    assert err < 1e-15, err
    return err


def check_tool_shift_equals_fracture_shift():
    h, D = 1.0 / 32.0, 0.25
    common = dict(fracture_type="planar-dip", dip_deg=10.0, x_base=6.5, ellength=h,
                  ymin=4.75, ymax=5.25)
    tool_moved = GeometryParams(z_center=5.0, zmin=4.75, zmax=5.25, **common)
    frac_moved = GeometryParams(z_center=5.0 - D, zmin=4.75 - D, zmax=5.25 - D, **common)
    t_a, a = run(tool_moved, shift=+D)
    t_b, b = run(frac_moved, shift=0.0)
    assert np.array_equal(t_a, t_b)
    assert np.max(np.abs(b)) > 0.0
    err = rel_l2(a, b)
    assert err < 1e-12, err
    return err


def test_shift_needs_no_viz():
    geom = GeometryParams(fracture_type="planar", ymin=4.75, ymax=5.25, zmin=4.75, zmax=5.25)
    rt = RuntimeParams(tx_plane_file=PLANE, rx_plane_file=PLANE, cores=1, no_viz=False,
                       tool_shift=0.25)
    try:
        run_simulation(MediaParams(), SourceParams(), geom, rt)
    except ValueError:
        return
    raise AssertionError("tool_shift without no_viz must raise")


def test_zero_shift_reproduces_example():
    check_zero_shift_reproduces_example()


def test_tool_shift_equals_fracture_shift():
    check_tool_shift_equals_fracture_shift()


if __name__ == "__main__":
    e0 = check_zero_shift_reproduces_example()
    e1 = check_tool_shift_equals_fracture_shift()
    test_shift_needs_no_viz()
    print(f"test_tool_shift: 3 checks passed (example rel L2 {e0:.1e}, "
          f"tool +0.25 m against fracture -0.25 m rel L2 {e1:.1e})")
