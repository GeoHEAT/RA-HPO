"""One synthetic trace, compared with the solver before this package split.

The numbers were recorded from ``run_simulation`` on this same pulse and mesh.
They lock the wiring of the modules: a change in the mathematics moves them.
"""
from __future__ import annotations

import sys
import tempfile
from pathlib import Path

import h5py
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from rahpo.parameters import GeometryParams, MediaParams, RuntimeParams, SourceParams
from rahpo.simulation import run_simulation

DT = 1e-10
NT = 256


def _plane(path: Path) -> None:
    yy = np.linspace(4.0, 6.0, 11)
    zz = np.linspace(4.0, 6.0, 11)
    t = np.arange(NT) * DT
    pulse = np.exp(-0.5 * ((t - 20e-9) / 4e-9) ** 2).astype(np.float32)
    with h5py.File(path, "w") as f:
        for comp in ("Ex", "Ey", "Ez", "Hx", "Hy", "Hz"):
            scale = 0.1 if comp.startswith("H") else 1.0
            data = np.broadcast_to(pulse * scale, (zz.size, yy.size, NT)).copy()
            f.create_dataset(comp, data=data.astype(np.float32))
        f.create_dataset("yy", data=yy)
        f.create_dataset("zz", data=zz)
        f.attrs.update(format_version=1, x1=3.5, dt=DT, Nt=NT,
                       target_res=float(yy[1] - yy[0]), step=1,
                       source_file="synthetic", source_size=0, source_mtime=0.0)


def test_synthetic_trace_matches_the_pre_split_solver():
    with tempfile.TemporaryDirectory() as directory:
        path = str(Path(directory) / "plane.h5")
        _plane(Path(path))
        geom = GeometryParams(ellength=0.5, ymin=4.0, ymax=6.0, zmin=4.0, zmax=6.0,
                              fracture_type="planar")
        run = RuntimeParams(tx_plane_file=path, rx_plane_file=path, cores=1, no_viz=True)
        _time, trace = run_simulation(MediaParams(), SourceParams(), geom, run)
    assert trace.size == 1126
    assert int(np.argmax(np.abs(trace))) == 710
    np.testing.assert_allclose(trace.sum(), -5.221568422524584, rtol=1e-12, atol=0.0)
    np.testing.assert_allclose(trace[710], -0.28107878314521706, rtol=1e-12, atol=0.0)
    np.testing.assert_allclose(
        trace[:8],
        [0.00309489, 0.00309915, 0.00310351, 0.00310787,
         0.00311209, 0.00311607, 0.00311969, 0.00312286],
        rtol=0.0, atol=5e-9)
    assert np.all(np.isfinite(trace))
