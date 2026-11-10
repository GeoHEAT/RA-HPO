"""Test of the helpers of tools/profile_figure.py (no data needed).

Five checks:
  1. radargram: traces of different lengths on one time axis are cut to a common window and
     stacked by source height, and a trace shifted by k samples appears shifted by k rows;
  2. relative_db: the median peak of each patch against that of the reference patch in dB
     (a patch ten times weaker gives -20 dB, the reference 0 dB, the median, not the mean);
  3. scored_window: the source heights a row of panels (b) and (c) spans, half a step beyond the
     first and the last scored position, as the image extent of the radargram;
  4. row_order: the rows of panels (b) and (c) run from the highest fracture down, as in panel (a);
  5. tick_alignment: a time label on an edge of the window is set inside the panel, so the labels
     of panels (b) and (c), which meet there, do not run together.
"""
from __future__ import annotations

import sys
from collections import namedtuple
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "tools"))

from profile_figure import radargram, relative_db, row_order, scored_window, tick_alignment  # noqa: E402


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


def test_scored_window():
    lo, hi = scored_window([10.5, 10.0, 10.25], 0.25)
    assert (lo, hi) == (9.875, 10.625), (lo, hi)


def test_row_order():
    P = namedtuple("P", "name z_hi")
    rows = row_order((P("A", 22.1), P("B", 51.1), P("C", 78.2)))
    assert [p.name for p in rows] == ["C", "B", "A"], rows


def test_tick_alignment():
    ha = tick_alignment([100, 150, 200, 250], (100.0, 250.0))
    assert ha == ["left", "center", "center", "right"], ha
