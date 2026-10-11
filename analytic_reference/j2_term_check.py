"""Size of the J2 term that exxgprt.m omits from the exact infinite-layer trace.

The reflected field of a horizontal electric dipole above a layered medium has two Sommerfeld
terms, a J0 term and a cos(2 phi) J2 term:

    E_xx = 1/(8 pi) Int kappa e^{-Gamma0 z} [ J0(kappa r) (Gamma0 R_TM / y0 - j w mu R_TE / Gamma0)
                                             - cos(2 phi) J2(kappa r) (Gamma0 R_TM / y0 + j w mu R_TE / Gamma0) ] dkappa

exxgprt.m evaluates only the J0 term. Near normal incidence R_TM = -R_TE and Gamma0/y0 = j w mu/Gamma0,
so the J2 bracket vanishes, and J2(kappa r) is small for the small kappa that dominate when the
offset r = 1 m is small against the two-way distance z = 11.98 m. This script evaluates both
terms with the integration of exxgprt_check.py and reports the peak of the J0 trace, the peak of
the J2 trace and the peak of their sum, for the configuration of exxgprt.m (inline, phi = 0).

    python analytic_reference/j2_term_check.py
"""
import numpy as np
from scipy.integrate import quad
from scipy.special import jv

d, ncells = 0.02, 1
zsrc = zrec = 6.0 - 0.01 * ncells
z, r, phi = zrec + zsrc, 1.0, 0.0
nf, mf = 256, 8
c0, mu = 299792458.0, 4e-7 * np.pi
eps0 = 1 / (mu * c0 ** 2)
fc = 1e8
freq = np.arange(1, nf + 1) * 6 * fc / nf
wav = 2 * (freq / fc) ** 2 * np.exp(-(freq / fc) ** 2) / (fc * np.sqrt(np.pi))
smu = 2j * np.pi * freq * mu
eta0 = 1e-4 + 2j * np.pi * freq * eps0 * 9.0
eta1 = 0.5 + 2j * np.pi * freq * eps0 * 80.0


def kern(kappa, e0, e1, sm, order):
    e2 = e0
    g0, g1, g2 = (np.sqrt(sm * e + kappa ** 2) for e in (e0, e1, e2))
    r0te, r1te = (g0 - g1) / (g0 + g1), (g1 - g2) / (g1 + g2)
    r0tm, r1tm = (e1 * g0 - e0 * g1) / (e1 * g0 + e0 * g1), (e2 * g1 - e1 * g2) / (e2 * g1 + e1 * g2)
    x = np.exp(-2 * g1 * d)
    glre = (r0te + r1te * x) / (1 + r0te * r1te * x)
    glrm = (r0tm + r1tm * x) / (1 + r0tm * r1tm * x)
    arg = g0 - 1j * np.imag(np.sqrt(sm * e0))
    if order == 0:
        return jv(0, kappa * r) * kappa * np.exp(-arg * z) * (g0 * glrm / e0 - sm * glre / g0)
    return -np.cos(2 * phi) * jv(2, kappa * r) * kappa * np.exp(-arg * z) * (g0 * glrm / e0 + sm * glre / g0)


def cquad(f, a, b):
    re = quad(lambda k: f(k).real, a, b, epsrel=1e-10, limit=2000)[0]
    im = quad(lambda k: f(k).imag, a, b, epsrel=1e-10, limit=2000)[0]
    return re + 1j * im


def trace(order):
    v = np.zeros(nf, complex)
    for j in range(nf):
        a = np.imag(np.sqrt(eta0[j] * smu[j]))
        b = np.imag(np.sqrt(eta1[j] * smu[j]))
        mx = np.sqrt((40 / abs(z)) ** 2 + abs(eta1[j] * smu[j]))
        f = lambda k: kern(k, eta0[j], eta1[j], smu[j], order)
        s = cquad(f, 0, a) + cquad(f, a, b)
        if mx > b:
            s += cquad(f, b, mx)
        v[j] = s * np.exp(-1j * np.imag(np.sqrt(smu[j] * eta0[j])) * z)
    e = np.zeros(mf * nf, complex)
    e[1:nf + 1] = v * wav * np.exp(-2j * np.pi * freq / fc) / (8 * np.pi)
    return 2 * mf * nf * np.real(np.fft.ifft(e, mf * nf)) * freq[0]


t0, t2 = trace(0), trace(2)
p0, p2, ps = (np.max(np.abs(x)) for x in (t0, t2, t0 + t2))
print(f"peak |J0 trace| {p0:.6e}  peak |J2 trace| {p2:.6e}  ratio {p2 / p0:.3e}")
print(f"peak of J0 + J2 relative to J0 alone: {ps / p0 - 1:+.3e}")
print(f"max |J2 trace| / peak |J0 trace| over the record: {np.max(np.abs(t2)) / p0:.3e}")
