"""Exact reflected trace of an infinite planar layer of any aperture and fill.

The same Sommerfeld-integral solution as exxgprt.m and exxgprt_check.py (a z-directed dipole and
a z-directed receiver 1 m apart in a host of eps_r 9 and 1e-4 S/m, facing an infinite layer),
written as functions of the aperture d, the fill (eps_r, sigma) and the distance of the
layer's front face. The antennas lie 6.0 m from the layer's center plane, as in the infinite
planar case of the paper, so the front face lies at 6.0 - d/2. The trace carries the 1/fc
wavelet delay of exxgprt.m; examples.traces.load_analytic shifts it to gprMax's sqrt(2)/fc.

    python examples/analytic/thin_layers.py --check      # reproduce the stored 2 cm and 3 cm traces
    python examples/analytic/thin_layers.py --write      # the five thin-layer references
    python examples/analytic/thin_layers.py --infinite   # the 3 cm reference of the infinite case

--infinite writes infinite_planar_3cm_analy_trace.csv: 3 cm of water (eps_r 80, 0.5 S/m) with the
front face at 5.985 m, the layer that the two-cell slab of the infinite planar FDTD model acts as
in gprMax 3.1.7 (tools/voxel_thickness.py), evaluated with the integration of the stored 2 cm
trace.

--write puts the traces of the thin layers compared in the paper in examples/analytic/thin_layers/:

    brine10_0p1mm.csv   0.1 mm, eps_r 43, 10 S/m       sigma d = 0.001 S
    brine10_1mm.csv     1 mm,   eps_r 43, 10 S/m       sigma d = 0.01 S
    brine25_0p4mm.csv   0.4 mm, eps_r 43, 25 S/m       sigma d = 0.01 S
    fresh_1mm.csv       1 mm,   eps_r 80, 0.01 S/m
    water_5mm.csv       5 mm,   eps_r 80, 0.5 S/m

Each takes about a minute.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
from scipy.integrate import quad
from scipy.special import jv

HERE = Path(__file__).resolve().parent
C0 = 299792458.0
MU = 4e-7 * np.pi
EPS0 = 1 / (MU * C0 ** 2)
FC = 1e8
NF, MF = 256, 8
CENTER = 6.0                  # antenna axis to the layer's center plane [m]
OFFSET = 1.0                  # transmitter-receiver offset [m]

LAYERS = {   # name: (aperture [m], eps_r, sigma [S/m])
    "brine10_0p1mm": (1e-4, 43.0, 10.0),
    "brine10_1mm": (1e-3, 43.0, 10.0),
    "brine25_0p4mm": (4e-4, 43.0, 25.0),
    "fresh_1mm": (1e-3, 80.0, 0.01),
    "water_5mm": (5e-3, 80.0, 0.5),
}


def axes():
    freq = np.arange(1, NF + 1) * 6 * FC / NF
    dt = 1 / (MF * freq.max())
    return freq, np.arange(MF * NF) * dt * 1e9


def _kern(kappa, e0, e1, e2, sm, d, z, r):
    g0 = np.sqrt(sm * e0 + kappa ** 2)
    g1 = np.sqrt(sm * e1 + kappa ** 2)
    g2 = np.sqrt(sm * e2 + kappa ** 2)
    r0te = (g0 - g1) / (g0 + g1)
    r1te = (g1 - g2) / (g1 + g2)
    r0tm = (e1 * g0 - e0 * g1) / (e1 * g0 + e0 * g1)
    r1tm = (e2 * g1 - e1 * g2) / (e2 * g1 + e1 * g2)
    x = np.exp(-2 * g1 * d)
    glre = (r0te + r1te * x) / (1 + r0te * r1te * x)
    glrm = (r0tm + r1tm * x) / (1 + r0tm * r1tm * x)
    arg = g0 - 1j * np.imag(np.sqrt(sm * e0))
    return jv(0, kappa * r) * kappa * np.exp(-arg * z) * (g0 * glrm / e0 - sm * glre / g0)


def _cquad(f, a, b, pieces=1, epsrel=1e-10):
    """Complex integral over [a, b], split into equal pieces."""
    edges = np.linspace(a, b, pieces + 1)
    total = 0j
    for lo, hi in zip(edges[:-1], edges[1:]):
        total += quad(lambda k: f(k).real, lo, hi, epsrel=epsrel, limit=4000)[0]
        total += 1j * quad(lambda k: f(k).imag, lo, hi, epsrel=epsrel, limit=4000)[0]
    return total


# Integration of the five thin layers: the interval between the branch points of host and fill
# is long for a brine fill, so it is split; (8, 40, 10) pieces at epsrel 1e-12 converge without
# warnings, and doubling them changes the 0.4 mm and 0.1 mm traces by 5e-15 of their peak. With
# single intervals the 0.4 mm, 25 S/m layer warned and was off by 1.1e-4 of its peak. The
# stored 2 cm check keeps the single intervals of exxgprt.m, which it reproduces.
FINE = {"pieces": (8, 40, 10), "epsrel": 1e-12}


def exact_trace(d: float, epsr_fill: float, sigma_fill: float, front: float | None = None,
                r: float = OFFSET, epsr_host: float = 9.0, sigma_host: float = 1e-4,
                pieces=(1, 1, 1), epsrel: float = 1e-10):
    """(time_ns, Ez) of the reflection from the layer; front defaults to CENTER - d / 2."""
    if front is None:
        front = CENTER - d / 2
    z = 2 * front
    freq, time = axes()
    wav = 2 * (freq / FC) ** 2 * np.exp(-(freq / FC) ** 2) / (FC * np.sqrt(np.pi))
    smu = 2j * np.pi * freq * MU
    eta0 = sigma_host + 2j * np.pi * freq * EPS0 * epsr_host
    eta1 = sigma_fill + 2j * np.pi * freq * EPS0 * epsr_fill
    eta2 = eta0
    vc1 = np.zeros(NF, complex)
    for j in range(NF):
        a = np.imag(np.sqrt(eta0[j] * smu[j]))
        b = np.imag(np.sqrt(eta1[j] * smu[j]))
        mx = np.sqrt((40 / abs(z)) ** 2 + abs(eta1[j] * smu[j]))
        f = lambda k: _kern(k, eta0[j], eta1[j], eta2[j], smu[j], d, z, r)
        v = _cquad(f, 0, a, pieces[0], epsrel) + _cquad(f, a, b, pieces[1], epsrel)
        if mx > b:
            v += _cquad(f, b, mx, pieces[2], epsrel)
        v *= np.exp(-1j * np.imag(np.sqrt(smu[j] * eta0[j])) * z)
        vc1[j] = v
    exx0 = np.zeros(MF * NF, complex)
    exx0[1:NF + 1] = vc1 * wav * np.exp(-2j * np.pi * freq / FC) / (8 * np.pi)
    return time, 2 * MF * NF * np.real(np.fft.ifft(exx0, MF * NF)) * freq[0]


INFINITE = (0.03, 80.0, 0.5)              # aperture [m], eps_r, sigma [S/m] of the infinite case
INFINITE_CSV = "infinite_planar_3cm_analy_trace.csv"


def check_infinite() -> float:
    """Max difference to the stored 3 cm trace of the infinite case relative to its peak."""
    csv = np.loadtxt(HERE / INFINITE_CSV, delimiter=",")
    t, e = exact_trace(*INFINITE)
    if not np.allclose(csv[:, 0], t, rtol=0, atol=1e-9):
        raise ValueError("time axes differ")
    return float(np.max(np.abs(csv[:, 1] - e)) / np.max(np.abs(csv[:, 1])))


def check_stored() -> float:
    """Max difference to the stored 2 cm trace relative to its peak (exxgprt_check.py defaults)."""
    csv = np.loadtxt(HERE / "vertical_plannar_2cm_analy_trace.csv", delimiter=",")
    t, e = exact_trace(0.02, 80.0, 0.5, front=6.0 - 0.01)
    if not np.allclose(csv[:, 0], t, rtol=0, atol=1e-9):
        raise ValueError("time axes differ")
    return float(np.max(np.abs(csv[:, 1] - e)) / np.max(np.abs(csv[:, 1])))


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--check", action="store_true", help="reproduce the stored 2 cm and 3 cm traces")
    ap.add_argument("--write", action="store_true", help="write the five thin-layer traces")
    ap.add_argument("--infinite", action="store_true", help="write the 3 cm trace of the infinite case")
    ap.add_argument("--out-dir", default=None,
                    help="output folder (default: thin_layers/ for --write, this folder for --infinite)")
    a = ap.parse_args()
    if not (a.check or a.write or a.infinite):
        ap.error("give --check, --write and/or --infinite")
    if a.check:
        for label, fn in (("stored 2 cm trace", check_stored), ("stored 3 cm trace", check_infinite)):
            err = fn()
            print(f"{label}: max|diff|/peak = {err:.2e} -> {'PASS' if err < 1e-11 else 'FAIL'}")
            if err >= 1e-11:
                return 1
    if a.infinite:
        t, e = exact_trace(*INFINITE)
        path = Path(a.out_dir or HERE) / INFINITE_CSV
        np.savetxt(path, np.column_stack([t, e]), delimiter=",")
        print(f"wrote {path}")
    if a.write:
        import warnings
        out = Path(a.out_dir or HERE / "thin_layers")
        out.mkdir(parents=True, exist_ok=True)
        for name, (d, er, s) in LAYERS.items():
            with warnings.catch_warnings(record=True) as w:
                warnings.simplefilter("always")
                t, e = exact_trace(d, er, s, **FINE)
            np.savetxt(out / f"{name}.csv", np.column_stack([t, e]), delimiter=",")
            print(f"wrote {out / name}.csv  (d {d * 1e3:g} mm, eps_r {er:g}, {s:g} S/m, "
                  f"peak {np.max(np.abs(e)):.4e}, integration warnings {len(w)})")
    return 0


if __name__ == "__main__":
    sys.exit(main())
