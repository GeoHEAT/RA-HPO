"""Test of examples/analytic/thin_layers.py (exact infinite layers of any aperture and fill).

Three checks:
  1. the defaults of exxgprt_check.py (2 cm, eps_r 80, 0.5 S/m, front face at 5.99 m)
     reproduce the stored trace vertical_plannar_2cm_analy_trace.csv to 1e-11 of its peak;
  2. the stored 1 mm brine reference in examples/analytic/thin_layers/ matches a fresh
     computation to 1e-12 of its peak;
  3. two layers of equal sheet conductance, 1 mm at 10 S/m and 0.4 mm at 25 S/m, have peaks
     within 0.3 % of each other, and 0.1 mm at 10 S/m peaks at 0.153 of 1 mm (the sheet law
     |R| = x / (1 + x), x = Z sigma d / 2, gives 0.059 / 0.386 = 0.153).

Run from the repository root:  python tests/test_thin_layers.py   (about 30 s)
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent.parent / "examples" / "analytic"
sys.path.insert(0, str(HERE))

from thin_layers import FINE, LAYERS, check_infinite, check_stored, exact_trace  # noqa: E402


def stored(name):
    return np.loadtxt(HERE / "thin_layers" / f"{name}.csv", delimiter=",")


def check_stored_two_cm():
    err = check_stored()
    assert err < 1e-11, err
    return err


def test_reference_file():
    d, er, s = LAYERS["brine10_1mm"]
    t, e = exact_trace(d, er, s, **FINE)
    ref = stored("brine10_1mm")
    assert np.allclose(ref[:, 0], t, rtol=0, atol=1e-9)
    err = np.max(np.abs(ref[:, 1] - e)) / np.max(np.abs(e))
    assert err < 1e-12, err


def check_sheet_law():
    peak = {n: np.max(np.abs(stored(n)[:, 1])) for n in ("brine10_1mm", "brine25_0p4mm", "brine10_0p1mm")}
    same = peak["brine25_0p4mm"] / peak["brine10_1mm"]
    assert abs(same - 1) < 3e-3, same
    thin = peak["brine10_0p1mm"] / peak["brine10_1mm"]
    assert abs(thin - 0.153) < 0.003, thin
    return same, thin


def test_stored_two_cm():
    check_stored_two_cm()


def test_stored_three_cm():
    # the reference of the infinite planar case: the 3 cm layer its two-cell FDTD slab acts as
    err = check_infinite()
    assert err < 1e-11, err


def test_sheet_law():
    check_sheet_law()


if __name__ == "__main__":
    err = check_stored_two_cm()
    test_reference_file()
    same, thin = check_sheet_law()
    print(f"test_thin_layers: 3 checks passed (stored 2 cm {err:.1e}; equal sigma*d peak ratio {same:.4f}; "
          f"0.1 mm / 1 mm {thin:.3f})")
