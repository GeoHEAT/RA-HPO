"""Exact angular transfer of a fluid-filled borehole (cylinder solution).

A z-directed Hertzian dipole sits on the axis of an infinite circular cylinder of radius a
(fluid eps_f, sigma_f) in an unbounded host (eps_h, sigma_h). Each axial wavenumber beta
carries one TM cylindrical wave of order 0; matching E_z and H_phi at rho = a gives the wave
that leaves the cylinder. In the far field the wave leaving at elevation psi above the plane
normal to the borehole has beta = k_h sin(psi) (stationary phase), so the far field with the
borehole divided by the far field of the same dipole in the host alone is

    F(psi, f) = (2j eps_h / (pi a)) / [eps_h kap_h H1(kap_h a) J0(kap_f a)
                                       - eps_f (kap_h^2 / kap_f) J1(kap_f a) H0(kap_h a)],

with kap = sqrt(k^2 - beta^2), H the Hankel functions of the second kind, the complex
permittivities eps = eps_r - j sigma / (omega eps0), and the exp(j omega t) convention
(Chew, 1995). F = 1 when the fluid equals the host. A reflection whose transmit leg leaves at
psi_t and whose receive leg arrives at psi_r carries F(psi_t) F(psi_r); a dipole model whose
wavelet was calibrated at broadside carries F(0)^2 instead.

    python tools/borehole_transfer.py                  # tables of F and the broadside loss
"""
from __future__ import annotations

import argparse
import sys

import numpy as np
from scipy.special import hankel2, jv

C0 = 299792458.0
MU0 = 4e-7 * np.pi
EPS0 = 1.0 / (C0 * C0 * MU0)
FC = 100e6
A_BOREHOLE = 0.075            # borehole radius [m]
WATER = (80.0, 0.5)           # borehole fluid of the paper (eps_r, sigma [S/m])
HOST = (9.0, 1e-4)


def transfer(psi_deg, f, a=A_BOREHOLE, fluid=WATER, host=HOST) -> np.ndarray:
    """One-way far-field ratio F(psi, f), the borehole dipole over the dipole in the host."""
    f = np.atleast_1d(np.asarray(f, float))
    w = 2 * np.pi * f
    eh = host[0] - 1j * host[1] / (w * EPS0)
    ef = fluid[0] - 1j * fluid[1] / (w * EPS0)
    kh = w * np.sqrt(MU0 * EPS0 * eh)
    kf = w * np.sqrt(MU0 * EPS0 * ef)
    beta = kh * np.sin(np.radians(psi_deg))
    kap_h = np.sqrt(kh ** 2 - beta ** 2 + 0j)
    kap_h = np.where(kap_h.imag > 0, -kap_h, kap_h)           # outgoing, decaying
    kap_f = np.sqrt(kf ** 2 - beta ** 2 + 0j)                 # even in kap_f: either root
    uh, uf = kap_h * a, kap_f * a
    den = eh * kap_h * hankel2(1, uh) * jv(0, uf) - ef * (kap_h ** 2 / kap_f) * jv(1, uf) * hankel2(0, uh)
    return (2j * eh / (np.pi * a)) / den


def _krho(k, kz):
    """sqrt(k^2 - kz^2) on the branch with Im <= 0 (outgoing or decaying)."""
    kr = np.sqrt(k ** 2 - kz ** 2 + 0j)
    return np.where(kr.imag > 0, -kr, kr)


def transfer_kz(kz, f, a=A_BOREHOLE, fluid=WATER, host=HOST) -> np.ndarray:
    """F at the axial wavenumbers kz (array) for one frequency f; evanescent for |kz| > Re k_h.
    transfer(psi, f) is this function at kz = k_h sin(psi)."""
    w = 2 * np.pi * f
    eh = host[0] - 1j * host[1] / (w * EPS0)
    ef = fluid[0] - 1j * fluid[1] / (w * EPS0)
    kh = w * np.sqrt(MU0 * EPS0 * eh)
    kf = w * np.sqrt(MU0 * EPS0 * ef)
    kap_h = _krho(kh, kz)
    kap_f = np.sqrt(kf ** 2 - kz ** 2 + 0j)
    uh, uf = kap_h * a, kap_f * a
    den = eh * kap_h * hankel2(1, uh) * jv(0, uf) - ef * (kap_h ** 2 / kap_f) * jv(1, uf) * hankel2(0, uh)
    return (2j * eh / (np.pi * a)) / den


def axial_integral(rho, dz, f, weight=None, host=HOST, n_prop=2048, n_evan=512):
    """I(rho, dz) = integral over kz of weight(kz) kap^2 H0(kap rho) exp(-j kz dz), host medium.

    With weight 1 this is 2j (k^2 + d^2/dz^2) exp(-j k r) / r, the E_z of a z-directed dipole
    up to a constant (Sommerfeld identity); with weight = transfer_kz it is the E_z outside the
    borehole with the same normalization. rho is a scalar, dz an array of heights above the
    dipole. The propagating part is integrated in phi with kz = Re(k) sin(phi), the evanescent
    parts in tau with kz = +-Re(k) cosh(tau), which removes the square-root behaviour at the
    branch points; Gauss-Legendre nodes, cut where the evanescent wave has decayed by exp(-40).
    """
    w = 2 * np.pi * f
    k = w * np.sqrt(MU0 * EPS0 * (host[0] - 1j * host[1] / (w * EPS0)))
    kr = k.real
    x, wx = np.polynomial.legendre.leggauss(n_prop)
    phi = x * np.pi / 2
    kz_p, jac_p = kr * np.sin(phi), kr * np.cos(phi) * wx * np.pi / 2
    tmax = np.arcsinh(40.0 / (kr * rho))
    t, wt = np.polynomial.legendre.leggauss(n_evan)
    tau = (t + 1) * tmax / 2
    kz_e, jac_e = kr * np.cosh(tau), kr * np.sinh(tau) * wt * tmax / 2
    kz = np.concatenate([kz_p, kz_e, -kz_e])
    jac = np.concatenate([jac_p, jac_e, jac_e])
    kap = _krho(k, kz)
    g = kap ** 2 * jac * (1.0 if weight is None else weight(kz))
    dz = np.atleast_1d(np.asarray(dz, float))
    return (np.exp(-1j * np.outer(dz, kz)) * hankel2(0, kap * rho)[None, :]) @ g


def near_field_ratio(rho, dz, f, a=A_BOREHOLE, fluid=WATER, host=HOST, **quad):
    """Exact E_z with the borehole over E_z of the dipole in the host alone, at radius rho and
    heights dz above the dipole. Tends to F(psi, f) far from the borehole."""
    num = axial_integral(rho, dz, f, lambda kz: transfer_kz(kz, f, a, fluid, host), host, **quad)
    return num / axial_integral(rho, dz, f, None, host, **quad)


def dipole_ez(rho, dz, f, host=HOST):
    """Closed form of 2j (k^2 + d^2/dz^2) exp(-j k r) / r, the reference for axial_integral."""
    w = 2 * np.pi * f
    k = w * np.sqrt(MU0 * EPS0 * (host[0] - 1j * host[1] / (w * EPS0)))
    dz = np.atleast_1d(np.asarray(dz, float))
    r = np.hypot(rho, dz)
    c2 = (dz / r) ** 2
    g = np.exp(-1j * k * r) / r
    return 2j * g * (k ** 2 * (1 - c2) + (2j * k / r + 2 / r ** 2) * c2 - (1j * k / r + 1 / r ** 2) * (1 - c2))


def ricker_spectrum(f, fc=FC):
    return (f / fc) ** 2 * np.exp(-(f / fc) ** 2)


def peak_ratio_two_way(psi_t, psi_r, fluid=WATER, n=1 << 14, fmax=1.2e9):
    """Ricker peak with F(psi_t) F(psi_r) over the Ricker peak with F(0)^2."""
    f = np.linspace(0, fmax, n // 2 + 1)[1:]
    r = ricker_spectrum(f)
    def peak(spec):
        full = np.zeros(n // 2 + 1, complex)
        full[1:] = spec
        return np.max(np.abs(np.fft.irfft(full, n)))
    f0 = transfer(0.0, f, fluid=fluid) ** 2
    return peak(r * transfer(psi_t, f, fluid=fluid) * transfer(psi_r, f, fluid=fluid)) / peak(r * f0)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.parse_args()

    print("One-way F(psi, f) of the paper's borehole (radius 0.075 m, water 80 / 0.5 S/m, host 9 / 1e-4 S/m)")
    print("  |F| / phase [deg] at psi = 0, 15, 30, 45, 55 deg")
    for f in (60e6, 100e6, 150e6, 200e6):
        row = "  ".join(f"{abs(v):5.3f}/{np.degrees(np.angle(v)):6.1f}"
                        for v in (transfer(p, f)[0] for p in (0, 15, 30, 45, 55)))
        print(f"  {f / 1e6:5.0f} MHz  {row}")
    print("  |F| cos(psi) at 100 MHz, psi = 0, 15, 30, 45, 55: " +
          "  ".join(f"{abs(transfer(p, 1e8)[0]) * np.cos(np.radians(p)):.3f}" for p in (0, 15, 30, 45, 55)))

    print("\nTwo-way broadside loss at 100 MHz, 1 / |F(0)|^2 (dB) for the borehole fluid conductivity:")
    for s in (0.01, 0.05, 0.1, 0.5, 1.0, 2.0):
        g = abs(transfer(0.0, 1e8, fluid=(80.0, s))[0]) ** 2
        print(f"  sigma_b = {s:5.2f} S/m   {1 / g:7.2f} x   {-20 * np.log10(g):5.1f} dB")

    print("\nTwo-way gain over broadside, Ricker peak, both legs at the dip (water 0.5 S/m and fresh 0.05 S/m):")
    for d in (8, 15, 20, 30, 40):
        print(f"  dip {d:2d} deg   {peak_ratio_two_way(d, d):5.3f}   {peak_ratio_two_way(d, d, fluid=(80.0, 0.05)):5.3f}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
