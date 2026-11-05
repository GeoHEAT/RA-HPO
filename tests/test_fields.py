"""Parts of the propagation stage that do not need a reference plane."""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from rahpo.constants import EPS0, FC, MU0
from rahpo.fields import (
    analytical_ricker_spectrum,
    edge_taper_weights,
    gamma_of_omega,
    next_pow2,
)


def test_next_pow2():
    assert next_pow2(1) == 1
    assert next_pow2(5) == 8
    assert next_pow2(8) == 8
    assert next_pow2(9) == 16


def test_lossless_propagation_constant():
    omega = 2.0 * np.pi * np.array([50e6, 100e6, 200e6])
    eps = EPS0 * 9.0
    gamma = gamma_of_omega(omega, MU0, eps, 0.0)
    expected = 1j * omega * np.sqrt(MU0 * eps)
    np.testing.assert_allclose(gamma, expected, rtol=1e-12)
    assert np.all(np.real(gamma) >= -1e-8)


def test_ricker_spectrum_peaks_at_the_center_frequency():
    freq = np.linspace(1e6, 400e6, 4000)
    spectrum = analytical_ricker_spectrum(2.0 * np.pi * freq, FC)
    assert spectrum[0] >= 0.0
    assert np.all(spectrum >= 0.0)
    assert abs(freq[np.argmax(spectrum)] - FC) / FC < 0.01


def test_edge_taper_is_one_inside_and_zero_on_the_boundary():
    weights = edge_taper_weights(
        np.array([5.0, 1.0]), np.array([5.0, 5.0]), 1.0, (1.0, 9.0, 1.0, 9.0))
    assert weights[0] == 1.0
    assert weights[1] == 0.0
