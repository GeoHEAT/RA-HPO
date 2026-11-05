"""Test of the dipole calibration of tools/dip_study.py.

Four checks, on synthetic traces and rows (no data needed):
  1. calibrating a trace on itself and applying the filter returns it, chi < 1e-3 (0.1 %);
  2. a smooth complex filter H(f), a loss and a delay like the borehole's, is recovered: a
     trace filtered by H is reproduced from the unfiltered one to chi < 1e-3 (0.1 %). The
     regularization eps = 1e-6 max|P|^2 leaves about 0.03 % in both;
  3. the completeness horizon is read from a solver log line: a time, "none" (every element
     contributes, inf) and a missing line or log (nan);
  4. a receiver whose t_c falls inside the evaluation window is left out of the full-coverage
     set, and one with t_c after the window or none is kept.

Run from the repository root:  python tests/test_dip_study.py
"""
from __future__ import annotations

import sys
import tempfile
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "tools"))

from dip_study import apply_filter, calibration_filter, full_coverage, horizon  # noqa: E402

DT = 0.0192583                      # ns, the planes' time step
T = np.arange(12000) * DT


def ricker(t, t0, fc=0.1):          # fc in GHz, t in ns
    a = (np.pi * fc * (t - t0)) ** 2
    return (1 - 2 * a) * np.exp(-a)


def chi(a, b):
    return 100 * np.linalg.norm(a - b) / np.linalg.norm(b)


def check_self_calibration():
    p = ricker(T, 130.0)
    W = calibration_filter(T, p, p)
    c = chi(apply_filter(W, p), p)
    assert c < 0.1, c
    return c


def check_known_filter():
    p = ricker(T, 130.0)
    n = 4 * T.size
    f = np.fft.rfftfreq(n, DT)                                  # GHz
    H = 0.16 * np.exp(-2j * np.pi * f * 3.5) * (1 + 0.3 * f)    # loss, 3.5 ns delay, slow dispersion
    q = np.fft.irfft(np.fft.rfft(p, n) * H, n)[: T.size]
    W = calibration_filter(T, q, p)
    c = chi(apply_filter(W, p), q)
    assert c < 0.1, c
    return c


def test_horizon():
    with tempfile.TemporaryDirectory() as d:
        Path(d, "cut.log").write_text(
            "2026-10-20 INFO [DROPPED] 0 of 10 elements\n"
            "2026-10-20 INFO [HORIZON] completeness horizon t_c = 135.0 ns on the trace time axis: "
            "earliest two-way time of the 12 non-contributing elements at v = 9.99e+07 m/s, plus the "
            "source delay 14.1 ns\n")
        Path(d, "full.log").write_text("2026-10-20 INFO [HORIZON] completeness horizon t_c: none, every "
                                       "element contributes\n")
        Path(d, "old.log").write_text("2026-10-20 INFO no horizon line\n")
        assert horizon("cut", d) == 135.0
        assert horizon("full", d) == np.inf
        assert np.isnan(horizon("old", d)) and np.isnan(horizon("absent", d))


def test_full_coverage():
    rows = [{"dip_deg": 30, "compared": "RA-HPO", "z_r": z, "t_c": tc}
            for z, tc in ((4.5, np.inf), (6.5, 131.2), (7.0, 250.0))]
    rows.append({"dip_deg": 20, "compared": "RA-HPO", "z_r": 4.5, "t_c": 140.0})
    assert [r["z_r"] for r in full_coverage(rows, 30, "RA-HPO")] == [4.5, 7.0]
    assert full_coverage(rows, 20, "RA-HPO") == []


def test_self_calibration():
    check_self_calibration()


def test_known_filter():
    check_known_filter()


if __name__ == "__main__":
    c0 = check_self_calibration()
    c = check_known_filter()
    test_horizon()
    test_full_coverage()
    print(f"test_dip_study: 4 checks passed (self-calibration chi {c0:.3f} %, known filter {c:.3f} %)")
