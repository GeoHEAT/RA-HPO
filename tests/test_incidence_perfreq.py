"""Test of the per-frequency incidence diagnostic (`--incidence perfreq`).

By default the solver applies one incidence direction per element, from the band-averaged real
Poynting vector, at every frequency. The diagnostic recomputes the direction per frequency.
Three checks:
  1. if every frequency's direction equals the band direction, the per-frequency path gives the
     band path's contribution to round-off, so the two paths implement the same reflection;
  2. with a field whose direction does change across the band, the diagnostic returns a finite
     contribution that differs from the band one, and a non-zero angular spread;
  3. the default, `incidence="band"`, returns a bare spectrum.

Run from the repository root:  python tests/test_incidence_perfreq.py
"""
from __future__ import annotations

import sys
from dataclasses import replace
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import ra_hpo  # noqa: E402
from ra_hpo import MediaParams, WorkerParams, element_contribution  # noqa: E402

NFREQ = 16


def build(y: float, z: float, rotate: bool):
    """One element at (6.5, y, z) facing the borehole, with a uniform six-component plane.

    With rotate=True the plane's Hx grows across the band, which turns Re(E x H*) from -x
    toward +y as frequency rises, so the incidence direction changes with frequency.
    """
    media = MediaParams()
    yy = np.linspace(0.0, 10.0, 21)
    zz = np.linspace(0.0, 10.0, 21)
    omega = 2 * np.pi * np.linspace(50e6, 200e6, NFREQ)
    v = 3e8 / np.sqrt(media.epsr2)
    gamma = 1j * omega / v
    shape = (len(yy), len(zz), NFREQ)
    plane = {c: np.zeros(shape, dtype=complex) for c in ("Ex", "Ey", "Ez", "Hx", "Hy", "Hz")}
    plane["Ez"][:] = 1.0
    plane["Hy"][:] = 1.0 / 120.0
    if rotate:
        plane["Hx"][:] = np.linspace(0.0, 1.0 / 120.0, NFREQ)[None, None, :]

    elempos = np.zeros((11, 1))
    elempos[0:3, 0] = (6.5, y, z)
    elempos[3:6, 0] = (-1.0, 0.0, 0.0)
    elempos[6, 0] = 0.01 ** 2
    elempos[7, 0] = 0.04
    elempos[8, 0] = media.epsr_plane
    elempos[9, 0] = media.epsr2
    elempos[10, 0] = media.sigma_water

    p = WorkerParams(
        Epos_tx=plane, Epos_rx=plane,
        yy=yy, zz=zz, x1=3.5, r_yy=yy, r_zz=zz, r_x1=3.5,
        omega_pos=omega, gamma_pos=gamma, gamma_pos1=gamma, eamp=np.ones(NFREQ),
        dt=1e-11, Npre=0, Nt_out=64, Nfft=128, v=v,
        epsr2=media.epsr2, eps0_local=8.854187817e-12,
        srcx=0.5, srcy=5.0, srcz=5.5, recx=0.5, recy=5.0, recz=4.5,
        spec_x=6.5, spec_y=5.0, spec_z=5.0,
        ellength=0.01, tapx=np.ones(1), tapy=np.ones(1), critdist=3.0,
        use_shift=False,
    )
    return elempos, p, media


def rel(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.linalg.norm(a - b) / np.linalg.norm(b))


def main() -> int:
    ok = True

    # 1. per-frequency path with every direction forced to the band direction
    elempos, p, media = build(5.6, 6.2, rotate=True)
    band = element_contribution(0, elempos, p, media)
    real_dirs = ra_hpo.per_frequency_directions
    ra_hpo.per_frequency_directions = lambda S, k: (np.repeat(k[:, None], S.shape[1], axis=1), 0.0)
    try:
        forced, _ = element_contribution(0, elempos, replace(p, incidence="perfreq"), media)
    finally:
        ra_hpo.per_frequency_directions = real_dirs
    e1 = rel(forced, band)
    print(f"  constant direction: perfreq vs band, relative difference {e1:.2e}")
    ok &= e1 < 1e-12

    # 2. a field whose direction turns across the band
    pf, spread = element_contribution(0, elempos, replace(p, incidence="perfreq"), media)
    e2 = rel(pf, band)
    finite = bool(np.all(np.isfinite(pf)))
    print(f"  turning direction:  perfreq vs band, relative difference {e2:.2e}, "
          f"spread {np.degrees(spread):.2f} deg, all finite: {finite}")
    ok &= finite and e2 > 1e-6 and spread > 0.0

    # 3. the default returns a bare spectrum
    c = element_contribution(0, elempos, p, media)
    bare = isinstance(c, np.ndarray) and c.shape == (NFREQ,)
    print(f"  default mode returns a bare spectrum: {bare}")
    ok &= bare

    print("\nPASS" if ok else "\nFAIL")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
