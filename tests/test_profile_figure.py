"""Test of the helpers of tools/profile_figure.py (no data needed).

Two checks:
  1. radargram: traces of different lengths on one time axis are cut to a common window and
     stacked by source height, and a trace shifted by k samples appears shifted by k rows;
  2. relative_db: the median peak of each patch against that of the reference patch in dB
     (a patch ten times weaker gives -20 dB, the reference 0 dB, the median, not the mean).
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "tools"))

from profile_figure import radargram, relative_db  # noqa: E402


def test_radargram():
    dt = 0.02
    t1 = np.arange(1000) * dt
    t2 = np.arange(1200) * dt
    a = np.exp(-((t1 - 10.0) / 0.5) ** 2)
    b = np.exp(-((t2 - 10.0 - 5 * dt) / 0.5) ** 2)
    t, z, img = radargram([(2.0, t1, a), (1.0, t2, b)], (5.0, 15.0))
    assert list(z) == [1.0, 2.0], z
    assert img.shape == (2, t.size) and abs(t[0] - 5.0) < 1e-9 and t[-1] <= 15.0 + 1e-9
    assert int(np.argmax(img[0])) - int(np.argmax(img[1])) == 5


def test_relative_db():
    peaks = {"A": [1.0, 2.0, 100.0], "B": [0.1, 0.2, 0.3], "C": [2.0, 2.0, 2.0]}
    db = relative_db(peaks, "A")
    assert db["A"] == 0.0, db
    assert abs(db["B"] - 20 * np.log10(0.2 / 2.0)) < 1e-12, db
    assert abs(db["C"] - 0.0) < 1e-12, db
