"""Test of tools/make_example_plane.py on a synthetic converted plane.

  1. the cropped, time-decimated plane loads through ra_hpo.load_gprmax_plane unchanged, with
     the requested receivers, dt multiplied by the factor and Nt divided by it;
  2. a 100 MHz Ricker pulse recorded at the FDTD time step survives decimation by 8: the
     decimated samples match the pulse at the new times to 1e-3 of its peak;
  3. a factor whose Nyquist frequency falls below twice the solver's 800 MHz cut-off is refused;
  4. an existing output file is not overwritten.

Run from the repository root:  python tests/test_make_example_plane.py
"""
from __future__ import annotations

import sys
import tempfile
from pathlib import Path

import h5py
import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))
sys.path.insert(0, str(HERE.parent / "tools"))

from make_example_plane import crop_and_decimate  # noqa: E402
from ra_hpo import load_gprmax_plane  # noqa: E402

DT = 0.01 / (299792458.0 * np.sqrt(3.0))    # gprMax time step of a 0.01 m cube
NT = 1600
FC = 100e6


def ricker(t, t0=20e-9):
    a = (np.pi * FC * (t - t0)) ** 2
    return (1 - 2 * a) * np.exp(-a)


def write_plane(path: Path):
    yy = np.round(np.arange(0.0, 9.0001, 0.04), 6)
    zz = np.round(np.arange(0.0, 9.0001, 0.04), 6)
    t = np.arange(NT) * DT
    s = ricker(t).astype(np.float32)
    with h5py.File(path, "w") as f:
        for i, c in enumerate(("Ex", "Ey", "Ez", "Hx", "Hy", "Hz")):
            f.create_dataset(c, data=np.broadcast_to(s * (i + 1), (zz.size, yy.size, NT)).astype(np.float32))
        f.create_dataset("yy", data=yy)
        f.create_dataset("zz", data=zz)
        f.attrs.update({"format_version": 1, "x1": 3.5, "dt": DT, "Nt": NT, "target_res": 0.04,
                        "step": 1, "source_file": "synthetic", "source_size": 0, "source_mtime": 0.0})


def main():
    with tempfile.TemporaryDirectory() as d:
        src, dst = Path(d) / "full.h5", Path(d) / "example.h5"
        write_plane(src)
        info = crop_and_decimate(src, dst, (4.4, 5.6), (4.6, 6.4), 8)
        fields, x1, yy, zz, dt, nt = load_gprmax_plane(str(dst))
        assert np.isclose(yy[0], 4.4) and np.isclose(yy[-1], 5.6) and yy.size == 31, yy[[0, -1]]
        assert np.isclose(zz[0], 4.6) and np.isclose(zz[-1], 6.4) and zz.size == 46, zz[[0, -1]]
        assert fields["Ez"].shape == (46, 31, nt) and nt == NT // 8, fields["Ez"].shape
        assert np.isclose(dt, 8 * DT) and x1 == 3.5
        err = np.max(np.abs(fields["Ez"][0, 0, :] / 3.0 - ricker(np.arange(nt) * dt)))
        assert err < 1e-3, f"decimated pulse differs by {err:.2e} of its peak"
        try:
            crop_and_decimate(src, Path(d) / "aliased.h5", (4.4, 5.6), (4.6, 6.4), 40)
            raise AssertionError("factor 40 should be refused")
        except ValueError:
            pass
        try:
            crop_and_decimate(src, dst, (4.4, 5.6), (4.6, 6.4), 8)
            raise AssertionError("existing output should not be overwritten")
        except FileExistsError:
            pass
    print(f"test_make_example_plane: 4 checks passed (46 x 31 receivers, Nt {nt}, "
          f"pulse error {err:.1e} of peak)")


if __name__ == "__main__":
    main()
