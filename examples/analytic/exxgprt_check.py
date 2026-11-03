"""Python transcription of exxgprt.m and hfhorxrall.m, to check the analytic reference without MATLAB.

Recomputes the exact reflected trace of the infinite planar case and compares it with a stored
CSV. With no arguments it checks vertical_plannar_2cm_analy_trace.csv in this folder:

    python examples/analytic/exxgprt_check.py
    python examples/analytic/exxgprt_check.py D NCELLS CSV

Arguments: layer thickness D (m), NCELLS with zsrc = zrec = 6 - 0.01 * NCELLS (half the
aperture in 0.01 m cells), and the CSV to compare with. Prints whether the time axes agree and
the largest difference relative to the peak (about 1.5e-12 for the stored trace). Needs numpy
and scipy; about a minute.
"""
import sys

import numpy as np
from scipy.integrate import quad
from scipy.special import jv

from pathlib import Path

d = float(sys.argv[1]) if len(sys.argv) > 1 else 0.02
ncells = int(sys.argv[2]) if len(sys.argv) > 2 else 1
ref = sys.argv[3] if len(sys.argv) > 3 else str(Path(__file__).with_name("vertical_plannar_2cm_analy_trace.csv"))
zsrc = zrec = 6.0 - 0.01 * ncells
z, r = zrec + zsrc, 1.0
nf, mf = 256, 8
epsr0 = 9.0
epsr1, sigma = 80.0, 0.5
c0 = 299792458.0
mu = 4e-7 * np.pi
eps0 = 1 / (mu * c0 ** 2)
fc = 1e8
freq = np.arange(1, nf + 1) * 6 * fc / nf
dt = 1 / (mf * freq.max())
time = np.arange(mf * nf) * dt * 1e9
wav = 2 * (freq / fc) ** 2 * np.exp(-(freq / fc) ** 2) / (fc * np.sqrt(np.pi))
smu = 2j * np.pi * freq * mu
eta0 = 1e-4 + 2j * np.pi * freq * eps0 * epsr0
eta1 = sigma + 2j * np.pi * freq * eps0 * epsr1
eta2 = eta0


def kern(kappa, e0, e1, e2, sm):
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


def cquad(f, a, b):
    re = quad(lambda k: f(k).real, a, b, epsrel=1e-10, limit=2000)[0]
    im = quad(lambda k: f(k).imag, a, b, epsrel=1e-10, limit=2000)[0]
    return re + 1j * im


vc1 = np.zeros(nf, complex)
for j in range(nf):
    a = np.imag(np.sqrt(eta0[j] * smu[j]))
    b = np.imag(np.sqrt(eta1[j] * smu[j]))
    mx = np.sqrt((40 / abs(z)) ** 2 + abs(eta1[j] * smu[j]))
    f = lambda k: kern(k, eta0[j], eta1[j], eta2[j], smu[j])
    v = cquad(f, 0, a) + cquad(f, a, b)
    if mx > b:
        v += cquad(f, b, mx)
    v *= np.exp(-1j * np.imag(np.sqrt(smu[j] * eta0[j])) * z)
    vc1[j] = v
exx0 = np.zeros(mf * nf, complex)
exx0[1:nf + 1] = vc1 * wav * np.exp(-2j * np.pi * freq / fc) / (8 * np.pi)
exx0t = 2 * mf * nf * np.real(np.fft.ifft(exx0, mf * nf)) * freq[0]
csv = np.loadtxt(ref, delimiter=",")
print(f"d={d} ncells={ncells}: time axis equal {np.allclose(csv[:, 0], time, rtol=0, atol=1e-9)}; "
      f"max|diff|/peak = {np.max(np.abs(csv[:, 1] - exx0t)) / np.max(np.abs(csv[:, 1])):.2e}")
