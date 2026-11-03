"""
Accuracy metrics and the closed-form numbers of the paper.

Trace metrics
-------------
For every trace pair the script computes the metrics of plot_traces.compare_traces (NRMSE, r,
peak ratio, peak-time difference) and three relative misfits on the same common time grid:

    chi     = ||p - d|| / ||d||                     relative misfit
    chi_s   = min_a ||a p - d|| / ||d||             after one amplitude scaling (optimal a)
    chi_st  = min_a,tau ||a p(t - tau) - d|| / ||d||  after scaling and a time shift, |tau| <= 0.3 ns

d is the reference (FDTD unless stated) and p the compared trace. chi_s equals sqrt(1 - c^2),
with c the uncentered correlation of p and d. chi is also evaluated over 100 to 250 ns, to show
that it does not depend on the window once the window contains the response.

Groups (metrics/accuracy_metrics.csv): the single traces of Table 2 ("accuracy") and the
corrugation amplitudes of Table 3 ("roughness"); the plane recorded without the borehole
("control"); the sixteen receivers of the gather; the two-medium runs; FDTD and RA-HPO against
the exact solution of the infinite case ("exact"); a pure delay or amplitude offset of the
finite planar FDTD trace, which shows the scale of chi ("calibration"); the plane computed on a
4.0 m domain against the 8.0 m one ("truncation"); and one incidence direction per frequency
against the band-averaged one ("incidence").

Further tables
--------------
    two_medium.csv           where the two-medium misfit sits: the best chi a pure time shift
                             reaches, and chi over the main lobe, the late window and the coda
    exact_solution.csv       each solution's misfit against the exact trace split into amplitude
                             and delay; the FDTD direct wave against the analytic whole-space
                             field; the Yee dispersion delay; the delay of a one-cell path change
    no_borehole_control.csv  envelope maxima, the delay of the best cross-correlation and the
                             best constant phase rotation of the plane recorded without the
                             borehole against FDTD
    incidence_spread.csv     how far the per-frequency incidence directions spread around the
                             band average (from incidence_perfreq_incidence.npz)
    closed_form.csv          R_c = 1/(A k_y^2) and k R_c of the corrugation; |R|/|r12| of a
                             water-filled slab; the skin depth of the heterogeneous fill

Usage:  python tools/accuracy_metrics.py [--out-dir metrics]
"""
from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

import numpy as np
from scipy.interpolate import interp1d
from scipy.signal import hilbert

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from rahpo.cases import BY_KEY, CASES, GATHER_Z, METRICS_DIR, TIME_RANGE, TRACE_DIR  # noqa: E402
from plot_traces import (band_match, compare_traces, load_analytic, load_rahpo,  # noqa: E402
                         load_scattered_gprmax)

SHIFTS_NS = np.arange(-0.3, 0.3 + 1e-9, 0.001)


def common_grid(t1, d1, t2, d2, time_range):
    """Reference and compared trace on the grid compare_traces builds, plus an interpolant of the compared trace."""
    m1 = (t1 >= time_range[0]) & (t1 <= time_range[1])
    m2 = (t2 >= time_range[0]) & (t2 <= time_range[1])
    tc = np.linspace(time_range[0], time_range[1], max(m1.sum(), m2.sum()))
    d = interp1d(t1[m1], d1[m1], kind="linear", bounds_error=False, fill_value=0)(tc)
    p_full = interp1d(t2, d2, kind="linear", bounds_error=False, fill_value=0)   # for shifting
    p = interp1d(t2[m2], d2[m2], kind="linear", bounds_error=False, fill_value=0)(tc)
    return tc, d, p, p_full


def misfits(t1, d1, t2, d2, time_range=TIME_RANGE):
    tc, d, p, p_full = common_grid(t1, d1, t2, d2, time_range)
    nd = np.linalg.norm(d)
    chi = np.linalg.norm(p - d) / nd
    a = (p @ d) / (p @ p)
    chi_s = np.linalg.norm(a * p - d) / nd
    best = (np.inf, 0.0, a)
    for tau in SHIFTS_NS:
        ps = p_full(tc - tau)
        aa = (ps @ d) / (ps @ ps)
        e = np.linalg.norm(aa * ps - d) / nd
        if e < best[0]:
            best = (e, tau, aa)
    c = (p @ d) / (np.linalg.norm(p) * nd)
    return {"chi_pct": 100 * chi, "chi_s_pct": 100 * chi_s, "alpha_opt": a,
            "chi_st_pct": 100 * best[0], "tau_opt_ns": best[1], "alpha_st": best[2],
            "sqrt_1_minus_c2_pct": 100 * np.sqrt(max(0.0, 1 - c * c))}


def row(group, label, reference, compared, t1, d1, t2, d2):
    m = compare_traces(t1, d1, t2, d2, time_range=TIME_RANGE)
    x = misfits(t1, d1, t2, d2)
    hi = min(250.0, t1.max(), t2.max())
    x_wide = misfits(t1, d1, t2, d2, time_range=(TIME_RANGE[0], hi))
    r = m["Pearson r"]
    return {"group": group, "case": label, "reference": reference, "compared": compared,
            "nrmse_pct": m["NRMSE (%)"], "r": r, "peak_ratio": m["Peak ratio (ref/RA-HPO)"],
            "peak_dt_ns": m["Peak-time diff (ns)"], **x,
            "sqrt_1_minus_r2_pct": 100 * np.sqrt(max(0.0, 1 - r * r)),
            "chi_pct_100_to_hi": x_wide["chi_pct"], "hi_ns": hi}


def trace_rows():
    rows = []
    for c in CASES:
        if c.group in ("accuracy", "roughness", "control", "two_medium"):
            t, d = load_scattered_gprmax(c)
            tp, p = load_rahpo(c.key)
            rows.append(row(c.group, c.label, c.fdtd, c.key, t, d, tp, p))
    for i, z in enumerate(GATHER_Z, start=1):
        c = BY_KEY[f"gather_z{str(z).replace('.', 'pt')}"]
        t, d = load_scattered_gprmax(c, rx=i)
        tp, p = load_rahpo(c.key)
        rows.append(row("gather", c.label, f"{c.fdtd} rx{i}", c.key, t, d, tp, p))
    inf = BY_KEY["infinite_planar"]
    te, de = load_analytic(inf.analytic)
    t, d = load_scattered_gprmax(inf)
    tp, p = load_rahpo(inf.key)
    rows.append(row("exact", "Infinite planar, FDTD against exact", inf.analytic, inf.fdtd, te, de, t, d))
    rows.append(row("exact", "Infinite planar, RA-HPO against exact", inf.analytic, inf.key, te, de, tp, p))
    fp = BY_KEY["finite_planar"]
    t, d = load_scattered_gprmax(fp)
    for dt in (0.05, 0.10):
        rows.append(row("calibration", f"FDTD finite planar delayed by {dt:.2f} ns", fp.fdtd,
                        "same, shifted", t, d, t + dt, d))
    for x in (1.04, 1.05, 1.08):
        rows.append(row("calibration", f"FDTD finite planar divided by {x:.2f}", fp.fdtd,
                        "same, scaled", t, d, t, d / x))
    t8, a8 = load_rahpo("finite_planar")
    t4, a4 = load_rahpo("truncated_domain")
    rows.append(row("truncation", "Plane from a 4.0 m domain against 8.0 m (RA-HPO traces)",
                    "finite_planar", "truncated_domain", t8, a8, t4, a4))
    tb, ab = load_rahpo("incidence_band")
    tq, aq = load_rahpo("incidence_perfreq")
    rows.append(row("incidence", "Piecewise concave, incidence per frequency against band-averaged",
                    "incidence_band", "incidence_perfreq", tb, ab, tq, aq))
    return rows


def twomedium_rows():
    """Where the two-medium misfit sits: the best chi any pure time shift within +-1 ns reaches,
    and chi over the main lobe (100-145 ns), the late window (145-200 ns) and the coda (150-170 ns),
    each normalized by the reference over that window. The matched eps_r = 9 run is included."""
    t, d = load_scattered_gprmax(BY_KEY["finite_planar_arms"])
    rows = []
    for er, key in ((9, "finite_planar_arms"), (6, "two_medium_eps6"), (12, "two_medium_eps12")):
        tp, p = load_rahpo(key)
        tc, dd, pp, pf = common_grid(t, d, tp, p, TIME_RANGE)
        nd = np.linalg.norm(dd)
        best = min((np.linalg.norm(pf(tc - tau) - dd) / nd, tau) for tau in np.arange(-1.0, 1.0 + 1e-9, 0.002))

        def win(lo, hi):
            k = (tc >= lo) & (tc < hi)
            return 100 * np.linalg.norm(pp[k] - dd[k]) / np.linalg.norm(dd[k])
        rows.append({"plane_epsr": er, "chi_pct": 100 * np.linalg.norm(pp - dd) / nd,
                     "chi_best_pure_shift_pct": 100 * best[0], "best_shift_ns": best[1],
                     "chi_100_145_pct": win(100, 145), "chi_145_200_pct": win(145, 200),
                     "chi_150_170_pct": win(150, 170)})
    return rows


def no_borehole_rows():
    """The finite planar case from a plane recorded without the borehole, against FDTD with it.

    Both traces on a 0.02 ns grid over the evaluation window: the maxima of their envelopes
    (|analytic signal|) and where they occur, the lag and value of the largest normalized
    cross-correlation (positive: the no-borehole trace arrives earlier), and the constant phase
    rotation of the no-borehole analytic signal that best matches the FDTD trace."""
    t, d = load_scattered_gprmax(BY_KEY["finite_planar"])
    tn, pn = load_rahpo("no_borehole_plane")
    tg = np.arange(TIME_RANGE[0], TIME_RANGE[1] + 1e-9, 0.02)
    q, p = np.interp(tg, t, d), np.interp(tg, tn, pn)
    eq, ep = np.abs(hilbert(q)), np.abs(hilbert(p))
    c = np.correlate(q, p, mode="full") / (np.linalg.norm(q) * np.linalg.norm(p))
    lags = (np.arange(c.size) - (p.size - 1)) * 0.02
    k = int(np.argmax(np.abs(c)))
    ap = hilbert(p)
    phis = np.radians(np.arange(-180.0, 180.0, 0.5))
    cor = [np.corrcoef(np.real(np.exp(1j * f) * ap), q)[0, 1] for f in phis]
    j = int(np.argmax(cor))
    return [
        {"quantity": "envelope maximum, FDTD (V/m)", "value": eq.max(), "time_ns": tg[eq.argmax()]},
        {"quantity": "envelope maximum, plane without borehole (V/m)", "value": ep.max(),
         "time_ns": tg[ep.argmax()]},
        {"quantity": "envelope ratio, without / with borehole", "value": ep.max() / eq.max(), "time_ns": ""},
        {"quantity": "largest normalized cross-correlation", "value": c[k], "time_ns": lags[k]},
        {"quantity": "best constant phase rotation (deg)", "value": np.degrees(phis[j]), "time_ns": ""},
        {"quantity": "correlation at that rotation", "value": cor[j], "time_ns": ""},
    ]


def incidence_rows():
    """Power-weighted rms spread of the per-frequency incidence directions around the band
    average, over the elements that contribute (written by ra_hpo.py --incidence perfreq)."""
    inc = np.load(TRACE_DIR / "incidence_perfreq_incidence.npz")
    sp, w = np.degrees(inc["spread_rad"]), inc["contribution"]
    ok = np.isfinite(sp) & (w > 0)
    return [{"quantity": "contributing elements", "value": int(ok.sum())},
            {"quantity": "median spread (deg)", "value": float(np.median(sp[ok]))},
            {"quantity": "95th percentile (deg)", "value": float(np.percentile(sp[ok], 95))},
            {"quantity": "maximum (deg)", "value": float(sp[ok].max())},
            {"quantity": "contribution-weighted mean (deg)",
             "value": float(np.sum(w[ok] * sp[ok]) / np.sum(w[ok]))}]


def exact_rows():
    """Why both solutions exceed 5 % against the exact solution of the infinite case.

    For FDTD and RA-HPO against the exact trace: chi; chi after one amplitude fit; after one time
    shift (within +-0.5 ns) only; after both. Then the direct wave: the first 2 to 40 ns of the
    same FDTD run (no borehole, so the direct wave is the whole-space field of the dipole on its
    axis 1 m away) against the analytic whole-space field of exxgprt.m with gprMax's Ricker delay
    sqrt(2)/f_c: peak ratio, chi and the best shift. Last, the phase delay that Yee dispersion
    predicts over the two-way path along a grid axis (eps_r = 9, dx = 0.01 m, dt = dx/(c sqrt 3)),
    weighted by the Ricker energy spectrum, and the two-way delay of a 0.01 m change of path."""
    inf = BY_KEY["infinite_planar"]
    te, de = load_analytic(inf.analytic)
    tf, df = load_scattered_gprmax(inf)
    tr, dr = load_rahpo(inf.key)
    rows = []
    for name, tt, xx in (("FDTD", tf, df), ("RA-HPO", tr, dr)):
        tc, q, p, pf = common_grid(te, de, tt, xx, TIME_RANGE)
        nq = np.linalg.norm(q)
        a = (p @ q) / (p @ p)
        shift = min((np.linalg.norm(pf(tc - s) - q) / nq, s) for s in np.arange(-0.5, 0.5 + 1e-9, 0.001))
        both = min(((lambda ps: np.linalg.norm((ps @ q) / (ps @ ps) * ps - q) / nq)(pf(tc - s)), s)
                   for s in np.arange(-0.5, 0.5 + 1e-9, 0.001))
        rows.append({"item": f"{name} against exact, reflection 100-200 ns",
                     "chi_pct": 100 * np.linalg.norm(p - q) / nq,
                     "chi_amplitude_fitted_pct": 100 * np.linalg.norm(a * p - q) / nq, "amplitude_scale": a,
                     "chi_shift_only_pct": 100 * shift[0], "shift_ns": -shift[1],
                     "chi_shift_and_amplitude_pct": 100 * both[0], "peak_ratio_exact_over_trace":
                     np.max(np.abs(q)) / np.max(np.abs(p))})
    c0, mu = 299792458.0, 4e-7 * np.pi
    eps0, fc, nf, mf = 1 / (mu * c0 ** 2), 1e8, 4096, 8
    freq = np.arange(1, nf + 1) * 6 * fc / 256 / 16
    dt = 1 / (mf * freq.max())
    wav = 2 * (freq / fc) ** 2 * np.exp(-(freq / fc) ** 2) / (fc * np.sqrt(np.pi))
    eta0 = 1e-4 + 2j * np.pi * freq * eps0 * 9.0
    g = np.sqrt(2j * np.pi * freq * mu * eta0)
    G = 2 * (1 + g) * np.exp(-g) / (4 * np.pi * eta0)           # whole space, on the dipole axis, R = 1 m
    spec = np.zeros(mf * nf, complex)
    spec[1:nf + 1] = G * wav * np.exp(-2j * np.pi * freq * np.sqrt(2) / fc)
    ta = np.arange(mf * nf) * dt * 1e9
    ea = band_match(ta, 2 * np.real(np.fft.ifft(spec)) / dt)   # the FDTD trace is band-matched too
    tg = np.linspace(2.0, 40.0, 20001)
    q, p = np.interp(tg, ta, ea), np.interp(tg, tf, df)
    nq = np.linalg.norm(q)
    shift = min((np.linalg.norm(np.interp(tg + s, tf, df) - q) / nq, s) for s in np.arange(-0.3, 0.3 + 1e-9, 0.0005))
    rows.append({"item": "FDTD direct wave against analytic whole-space field, 2-40 ns",
                 "chi_pct": 100 * np.linalg.norm(p - q) / nq, "chi_amplitude_fitted_pct": "",
                 "amplitude_scale": "", "chi_shift_only_pct": 100 * shift[0], "shift_ns": shift[1],
                 "chi_shift_and_amplitude_pct": "", "peak_ratio_exact_over_trace": np.max(np.abs(q)) / np.max(np.abs(p))})
    v, dx = c0 / 3, 0.01
    dtg = dx / (c0 * np.sqrt(3))
    f = np.linspace(5e6, 600e6, 2000)
    w = 2 * np.pi * f
    kn = 2 / dx * np.arcsin(dx / (v * dtg) * np.sin(w * dtg / 2))
    L = 2 * np.hypot(6.0, 0.5)
    tau = (kn - w / v) * L / w
    S2 = (f ** 2 * np.exp(-(f / fc) ** 2)) ** 2
    rows.append({"item": f"Yee dispersion delay over the {L:.2f} m two-way path, along an axis (ns)",
                 "chi_pct": "", "chi_amplitude_fitted_pct": "", "amplitude_scale": "", "chi_shift_only_pct": "",
                 "shift_ns": np.sum(tau * S2) / np.sum(S2) * 1e9, "chi_shift_and_amplitude_pct": "",
                 "peak_ratio_exact_over_trace": ""})
    rows.append({"item": "Two-way delay of a 0.01 m change of reflector distance (ns)", "chi_pct": "",
                 "chi_amplitude_fitted_pct": "", "amplitude_scale": "", "chi_shift_only_pct": "",
                 "shift_ns": 2 * 0.01 / v * 1e9, "chi_shift_and_amplitude_pct": "", "peak_ratio_exact_over_trace": ""})
    return rows


def analytic_rows():
    rows = []
    ky, kz, lam = np.pi / 2, 3 * np.pi / 8, 1.0
    for A in (0.0625, 0.125, 0.25, 0.30, 0.40, 0.50, 0.55, 0.60, 0.70):
        rc = 1 / (A * ky ** 2)
        rows.append({"quantity": "R_c = 1/(A ky^2) (m)", "parameter": f"A = {A} m", "value": rc})
        rows.append({"quantity": "k R_c", "parameter": f"A = {A} m", "value": 2 * np.pi / lam * rc})
    eps0, mu0, w = 8.854187817e-12, 4e-7 * np.pi, 2 * np.pi * 100e6

    def gamma(er, sig):
        return np.sqrt(1j * w * mu0 * (sig + 1j * w * eps0 * er))
    g1, g2 = gamma(9.0, 1e-4), gamma(80.0, 0.5)
    r12 = (g1 - g2) / (g1 + g2)
    for dap in (0.001, 0.005, 0.01, 0.02, 0.04):
        e = np.exp(-2 * g2 * dap)
        R = (r12 - r12 * e) / (1 - r12 ** 2 * e)
        rows.append({"quantity": "|R|/|r12|, water slab in host, normal incidence, 100 MHz",
                     "parameter": f"d = {dap * 100:g} cm", "value": abs(R) / abs(r12)})
    em, ef = 9.0 - 1j * 1e-4 / (w * eps0), 43.0 - 1j * 10.0 / (w * eps0)
    for phi in (0.1, 0.3, 0.5, 0.7, 0.9):
        e = ((1 - phi) * np.sqrt(em) + phi * np.sqrt(ef)) ** 2
        alpha = -(w * np.sqrt(mu0 * eps0 * e)).imag
        rows.append({"quantity": "skin depth of CRIM fill, 100 MHz (cm)",
                     "parameter": f"phi = {phi}", "value": 100 / alpha})
    return rows


def write(path: Path, rows):
    with open(path, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
        w.writeheader()
        for r in rows:
            w.writerow({k: (f"{v:.6g}" if isinstance(v, (float, np.floating)) else v) for k, v in r.items()})


def main() -> int:
    ap = argparse.ArgumentParser(description="Accuracy metrics and closed-form numbers.")
    ap.add_argument("--out-dir", default=str(METRICS_DIR))
    a = ap.parse_args()
    out = Path(a.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    tr = trace_rows()
    write(out / "accuracy_metrics.csv", tr)
    write(out / "two_medium.csv", twomedium_rows())
    write(out / "exact_solution.csv", exact_rows())
    write(out / "no_borehole_control.csv", no_borehole_rows())
    write(out / "incidence_spread.csv", incidence_rows())
    write(out / "closed_form.csv", analytic_rows())
    print(f"{'group':12} {'case':38} {'chi':>6} {'chi_s':>6} {'NRMSE':>6} {'r':>7} {'peak':>6}")
    for r in tr:
        print(f"{r['group']:12} {r['case'][:38]:38} {r['chi_pct']:6.2f} {r['chi_s_pct']:6.2f} "
              f"{r['nrmse_pct']:6.2f} {r['r']:7.4f} {r['peak_ratio']:6.3f}")
    print(f"\nwrote accuracy_metrics.csv, two_medium.csv, exact_solution.csv, "
          f"no_borehole_control.csv, incidence_spread.csv and closed_form.csv in {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
