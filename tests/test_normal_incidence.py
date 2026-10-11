"""Test of exactly normal incidence in element_contribution.

`element_contribution` splits the incident field into TE and TM components using
`e_te = (k_dir x n_hat)/|k_dir x n_hat|`, which is undefined when the ray is exactly along the
surface normal, so the code takes a separate branch there. No case of the paper reaches that
branch: element centers sit at `ymin + (k + 1/2) * ellength`, half a cell off the grid lines
the source lies on. The test therefore builds the geometry directly.

Three checks:
  1. an element on the source normal returns a finite contribution;
  2. an off-normal element returns a non-zero contribution;
  3. an element whose ray leaves the recorded plane returns exactly zero.

Run from the repository root:  python tests/test_normal_incidence.py
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from ra_hpo import MediaParams, WorkerParams, element_contribution  # noqa: E402

NFREQ = 8


def build(n_el_y: float, n_el_z: float) -> tuple[np.ndarray, WorkerParams, MediaParams]:
    """One element at (6.5, y, z) with an x-directed normal, source and receiver on x = 0.5."""
    media = MediaParams()
    # A small recorded plane at x1 = 3.5 m, wide enough that the rays cross inside it
    yy = np.linspace(0.0, 10.0, 21)
    zz = np.linspace(0.0, 10.0, 21)
    omega = 2 * np.pi * np.linspace(50e6, 200e6, NFREQ)
    v = 3e8 / np.sqrt(media.epsr2)
    gamma = 1j * omega / v
    plane = {c: np.zeros((len(yy), len(zz), NFREQ), dtype=complex) for c in ("Ex", "Ey", "Ez")}
    plane["Ez"][:] = 1.0 + 0.0j       # uniform, non-zero, so a dropped element is visible as 0

    # elempos rows, as get_fracture_mesh builds them: 0:3 position, 3:6 normal, 6 projected
    # area, 7 aperture, 8 fill permittivity, 9 host permittivity, 10 fill conductivity
    elempos = np.zeros((11, 1))
    elempos[0:3, 0] = (6.5, n_el_y, n_el_z)
    elempos[3:6, 0] = (-1.0, 0.0, 0.0)     # facing the borehole
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
        srcx=0.5, srcy=5.0, srcz=5.5, recx=0.5, recy=5.0, recz=5.5,
        spec_x=6.5, spec_y=5.0, spec_z=5.5,
        ellength=0.01, tapx=np.ones(1), tapy=np.ones(1), critdist=3.0,
        use_shift=False,
    )
    return elempos, p, media


def main() -> int:
    ok = True

    # 1. exactly normal incidence: element at the source's y and z, so k_dir = -n_hat
    elempos, p, media = build(5.0, 5.5)
    try:
        c = element_contribution(0, elempos, p, media)
        finite = bool(np.all(np.isfinite(c)))
        print(f"  normal incidence      -> returned shape {c.shape}, all finite: {finite}")
        ok &= finite
    except UnboundLocalError as exc:
        print(f"  normal incidence      -> FAILED, {type(exc).__name__}: {exc}")
        ok = False

    # 2. off-normal element
    elempos, p, media = build(5.6, 6.2)
    c_off = element_contribution(0, elempos, p, media)
    nonzero = bool(np.any(c_off != 0))
    print(f"  off-normal incidence  -> non-zero contribution: {nonzero}")
    ok &= nonzero

    # 3. an element whose ray leaves the recorded plane returns exactly zero, which the solver
    #    counts as a dropped element
    elempos, p, media = build(5.0, 60.0)
    c_out = element_contribution(0, elempos, p, media)
    dropped = bool(not np.any(c_out))
    print(f"  ray outside the plane -> exactly zero (counted as dropped): {dropped}")
    ok &= dropped

    print("\nPASS" if ok else "\nFAIL")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
