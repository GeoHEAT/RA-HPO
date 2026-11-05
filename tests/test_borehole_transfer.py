"""Test of tools/borehole_transfer.py (cylinder solution of the borehole dipole).

Five checks:
  1. a cylinder filled with the host gives F = 1 at every angle and frequency;
  2. F tends to 1 as the radius tends to zero;
  3. F is even in the elevation psi;
  4. at 100 MHz the paper's borehole gives |F| = 0.405, 0.423, 0.484, 0.614 and 0.766 at
     psi = 0, 15, 30, 45 and 55 degrees;
  5. the two-way broadside loss 1 / |F(0)|^2 is 15.7 dB for 0.5 S/m water and 2.4 dB for
     0.05 S/m;
  6. the axial-wavenumber integral of the exact near field reproduces the closed-form dipole
     field to 1e-11, gives a ratio of 1 for a host-filled borehole, and tends to F with
     distance (within 1 % at 30 m and 0.1 % at 300 m).

Run from the repository root:  python tests/test_borehole_transfer.py
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "tools"))

from borehole_transfer import (HOST, axial_integral, dipole_ez, near_field_ratio,  # noqa: E402
                               transfer)

F = np.array([50e6, 100e6, 200e6])


def test_host_filled():
    for psi in (0.0, 20.0, 45.0, 60.0):
        np.testing.assert_allclose(transfer(psi, F, fluid=HOST), 1.0, rtol=1e-10)


def test_small_radius():
    np.testing.assert_allclose(transfer(30.0, F, a=1e-5), 1.0, rtol=1e-4)


def test_even_in_psi():
    np.testing.assert_allclose(transfer(-25.0, F), transfer(25.0, F), rtol=1e-12)


def check_reference_values():
    got = [abs(transfer(p, 1e8)[0]) for p in (0, 15, 30, 45, 55)]
    np.testing.assert_allclose(got, [0.405, 0.423, 0.484, 0.614, 0.766], atol=5e-4)
    return got


def check_broadside_loss():
    db = lambda s: -20 * np.log10(abs(transfer(0.0, 1e8, fluid=(80.0, s))[0]) ** 2)
    assert round(db(0.5), 1) == 15.7, db(0.5)
    assert round(db(0.05), 1) == 2.4, db(0.05)
    return db(0.5)


def check_near_field():
    rho = 3.0
    dz = rho * np.tan(np.radians([0.0, 20.0, 45.0, -59.0]))
    for f in (60e6, 200e6):
        np.testing.assert_allclose(axial_integral(rho, dz, f), dipole_ez(rho, dz, f), rtol=1e-11)
        np.testing.assert_allclose(near_field_ratio(rho, dz, f, fluid=HOST), 1.0, rtol=1e-12)
    psi = np.array([0.0, 30.0, 45.0])
    worst = {}
    for R, tol in ((30.0, 1e-2), (300.0, 1e-3)):
        r = near_field_ratio(R, R * np.tan(np.radians(psi)), 1e8, n_prop=8192, n_evan=1024)
        F = np.array([transfer(p, 1e8)[0] for p in psi])
        worst[R] = float(np.max(np.abs(r / F - 1)))
        assert worst[R] < tol, (R, worst[R])
    return worst


def test_reference_values():
    check_reference_values()


def test_broadside_loss():
    check_broadside_loss()


def test_near_field():
    check_near_field()


if __name__ == "__main__":
    test_host_filled()
    test_small_radius()
    test_even_in_psi()
    got = check_reference_values()
    loss = check_broadside_loss()
    far = check_near_field()
    print(f"test_borehole_transfer: 6 checks passed (|F| at 100 MHz {', '.join(f'{v:.3f}' for v in got)}; "
          f"broadside loss {loss:.2f} dB; near field against F {far[30.0]:.1e} at 30 m, {far[300.0]:.1e} at 300 m)")
