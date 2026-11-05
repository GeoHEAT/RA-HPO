"""Test of the profile reduction of tools/profile_scene.py (--sum), on synthetic traces.

Every profile case and calibration pair is written as a Ricker pulse into a temporary trace
folder (no data needed). The calibration pairs differ by a gain g, so the calibrated dipole of a
position is g times its borehole-free trace. The borehole traces are set so that:
  1. patch A is 1.40 times its calibrated dipole, patch B 1.00 times (B on A's pair) and C 1.40;
  2. patch C is 10 times weaker than A, so its median peak lies 20.0 dB below A's, and B's lies
     20 log10(1.00 / 1.40) = -2.9 dB below.
--sum must return these numbers.

Run from the repository root:  python tests/test_profile_sum.py
"""
from __future__ import annotations

import os
import sys
import tempfile
from pathlib import Path

import numpy as np

TMP = tempfile.mkdtemp()
os.environ["RAHPO_TRACE_DIR"] = TMP                 # read by rahpo.paths at import
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "tools"))

import examples.cases  # noqa: E402
import profile_scene as ps  # noqa: E402

DT = 0.0192583
T = np.arange(16000) * DT
G = 0.16                                           # gain of the calibration pairs


def ricker(t0, fc=0.1):
    a = (np.pi * fc * (T - t0)) ** 2
    return (1 - 2 * a) * np.exp(-a)


def save(name, x):
    np.savez(Path(TMP) / f"{name}.npz", time_ns=T, amplitude=x)


def write_scene():
    for patch, ratio in ((ps.A, 1.40), (ps.B, 1.00), (ps.C, 1.40)):
        amp = 0.1 if patch is ps.C else 1.0
        for z in ps.positions(patch):
            t0 = 120.0 if patch is ps.B else 90.0 + 10.0 * (z - patch.z_lo - 4.0)
            save(ps.case_name(patch.name, z, "nobh"), amp * ricker(t0))
            save(ps.case_name(patch.name, z, "bh"), ratio * G * amp * ricker(t0))
    for kb, kn in set(ps.calibration_names().values()):
        save(kn, ricker(100.0))
        save(kb, G * ricker(100.0))


def test_sum():
    # rahpo.paths may already have been imported by another test, so the folder is set directly
    saved, examples.cases.TRACE_DIR = examples.cases.TRACE_DIR, Path(TMP)
    try:
        write_scene()
        out = "\n".join(ps.summarize())
    finally:
        examples.cases.TRACE_DIR = saved
    lines = {ln.split(":")[0]: ln for ln in out.splitlines()}
    assert "median peak ratio RA-HPO / dipole 1.400" in lines["patch A"], out
    assert "median peak ratio RA-HPO / dipole 1.000" in lines["patch B"], out
    assert "median peak ratio RA-HPO / dipole 1.400" in lines["patch C"], out
    assert "-2.9 dB" in lines["patch B against A"], out
    assert "-20.0 dB" in lines["patch C against A"], out


if __name__ == "__main__":
    test_sum()
    print("test_profile_sum: 1 check passed")
