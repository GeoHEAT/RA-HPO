"""Load traces and compare them over a time window.

The metric definitions are those used for the tables of the paper. Reference
traces pass through the same Gaussian synthesis window as the solver.
"""
from __future__ import annotations

import h5py
import numpy as np
from scipy.interpolate import interp1d

from rahpo.cases import ANALYTIC, Case, fdtd_output, trace_file
from rahpo.constants import FC

F_SIGMA = 8.0 * FC  # width of the Gaussian synthesis window, Hz


def band_match(t_ns: np.ndarray, d: np.ndarray) -> np.ndarray:
    """Apply the synthesis window of the solver, W(f) = exp(-(f/F_SIGMA)^2 / 2), to a reference trace.

    The solver multiplies the deconvolved RA-HPO spectrum by W. Every reference trace passes
    through the same window, so that both traces of a comparison cover the same band. The
    trace is zero-padded to four times its length, so the filter does not wrap around.
    """
    n = 4 * d.size
    f = np.fft.rfftfreq(n, (t_ns[1] - t_ns[0]) * 1e-9)
    return np.fft.irfft(np.fft.rfft(d, n) * np.exp(-0.5 * (f / F_SIGMA) ** 2), n)[: d.size]

def load_gprmax_ez(model: str, rx: int = 1) -> tuple[np.ndarray, np.ndarray]:
    """Return (time_ns, Ez) of one receiver of the gprMax output of a model, band-matched."""
    with h5py.File(fdtd_output(model), "r") as f:
        dt = float(f.attrs["dt"])
        ez = f[f"/rxs/rx{rx}/Ez"][:]
    t = np.arange(ez.size) * dt * 1e9
    return t, band_match(t, ez)

def load_scattered_gprmax(case: Case, rx: int = 1) -> tuple[np.ndarray, np.ndarray]:
    """gprMax trace minus the direct-wave run (both on the same time axis)."""
    t, ez = load_gprmax_ez(case.fdtd, rx)
    if case.direct is None:
        return t, ez
    t_ref, ez_ref = load_gprmax_ez(case.direct, rx)
    n = min(ez.size, ez_ref.size)
    if not np.allclose(t[:n], t_ref[:n]):
        raise ValueError(f"Time axes of {case.fdtd} and {case.direct} differ.")
    out = np.zeros_like(ez)
    out[:n] = ez[:n] - ez_ref[:n]
    return t, out

def load_rahpo(key: str) -> tuple[np.ndarray, np.ndarray]:
    """RA-HPO trace of a case, as written by ra_hpo.py --output_name <key>."""
    data = np.load(trace_file(key))
    return data["time_ns"], data["amplitude"]

def load_analytic(filename: str) -> tuple[np.ndarray, np.ndarray]:
    """Exact trace (analytic_reference/), shifted from its 1/fc wavelet delay to gprMax's sqrt(2)/fc,
    band-matched."""
    data = np.loadtxt(ANALYTIC / filename, delimiter=",")
    t = data[:, 0] - 1.0 / FC * 1e9 + np.sqrt(2.0) / FC * 1e9
    return t, band_match(t, data[:, 1])

def compare_traces(time1, data1, time2, data2, time_range=(0, 180)):
    """Metrics of trace 2 against reference trace 1 over time_range [ns] (see module docstring)."""
    mask1 = (time1 >= time_range[0]) & (time1 <= time_range[1])
    mask2 = (time2 >= time_range[0]) & (time2 <= time_range[1])

    t1_filtered = time1[mask1]
    d1_filtered = data1[mask1]
    t2_filtered = time2[mask2]
    d2_filtered = data2[mask2]

    t_common = np.linspace(time_range[0], time_range[1], max(len(t1_filtered), len(t2_filtered)))

    interp1 = interp1d(t1_filtered, d1_filtered, kind="linear", bounds_error=False, fill_value=0)
    interp2 = interp1d(t2_filtered, d2_filtered, kind="linear", bounds_error=False, fill_value=0)

    d1_interp = interp1(t_common)
    d2_interp = interp2(t_common)

    diff = d1_interp - d2_interp

    rmse = np.sqrt(np.mean(diff**2))
    chi = 100 * np.linalg.norm(diff) / np.linalg.norm(d1_interp) if np.any(d1_interp) else np.inf
    range1 = np.max(d1_interp) - np.min(d1_interp)
    nrmse = (rmse / range1 * 100) if range1 != 0 else np.inf

    corr = np.corrcoef(d1_interp, d2_interp)[0, 1]

    peak1 = np.max(np.abs(d1_interp))
    peak2 = np.max(np.abs(d2_interp))
    peak_ratio = peak1 / peak2 if peak2 != 0 else np.inf
    peak_dt = t_common[np.argmax(np.abs(d2_interp))] - t_common[np.argmax(np.abs(d1_interp))]

    return {
        "RMSE (V/m)": rmse,
        "NRMSE (%)": nrmse,
        "Pearson r": corr,
        "Peak ref (V/m)": peak1,
        "Peak RA-HPO (V/m)": peak2,
        "Peak ratio (ref/RA-HPO)": peak_ratio,
        "Peak-time diff (ns)": peak_dt,
        "Relative misfit (%)": chi,
    }
