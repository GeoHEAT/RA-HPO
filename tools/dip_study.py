"""Dipping fractures and thin layers: RA-HPO and a broadside-calibrated dipole against the references.

Dipping fractures (cases dip20_* and dip30_* of examples/cases.py). For each receiver of the
gather, RA-HPO on the borehole plane and the calibrated dipole are compared with the scattered
full-domain FDTD trace over the evaluation window. Each trace also carries its completeness
horizon t_c, read from the [HORIZON] line of its solver log: the earliest two-way time at which
an element that contributes nothing (in front of the plane, or with a ray that misses the
recorded plane) would arrive. A receiver with t_c inside the window lacks part of the fracture
and is listed apart from those with full plane coverage.

The calibrated dipole is the trace from the borehole-free plane times one filter
W(f) = Q P* / (|P|^2 + eps), eps = 1e-6 max|P|^2, taken in the window of the specular reflection
(the peak +- 15 ns, with cosine tapers) from a pair of traces for a fracture parallel to the
well, Q on the borehole plane and P on the borehole-free plane, with the same fill and
distance (cases dipcal20_* and dipcal30_*). W is then applied unchanged at every receiver.

Thin layers (cases thin_*): RA-HPO against the exact infinite layer of
examples/analytic/thin_layers.py.

Traces are read from RAHPO_TRACE_DIR, solver logs from RAHPO_LOG_DIR, full-domain outputs from
RAHPO_FDTD_DIR (default: next to the models). Missing traces are reported and skipped.

    python tools/dip_study.py [--out output/metrics]

Writes dipping.csv (one row per receiver and model) and thin_layers.csv, and prints the ranges
over the receivers with full plane coverage.
"""
from __future__ import annotations

import argparse
import csv
import re
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from examples.cases import BY_KEY, GATHER_Z, METRICS_DIR, THIN_LAYERS, TIME_RANGE, Z_SOURCE, trace_file, ztag  # noqa: E402
from examples.traces import compare_traces, load_analytic, load_rahpo, load_scattered_gprmax  # noqa: E402
from rahpo.paths import LOGS  # noqa: E402

DIPS = (30, 20)
HORIZON = re.compile(r"\[HORIZON\] completeness horizon t_c(?: = ([0-9.]+) ns|: none)")


def have(key):
    return trace_file(key).exists()


def horizon(key, log_dir=None):
    """t_c [ns] from the solver log of a case: inf when every element contributes, nan when the
    log or its [HORIZON] line is missing."""
    path = Path(log_dir or LOGS) / f"{key}.log"
    if not path.exists():
        return np.nan
    m = HORIZON.search(path.read_text(errors="replace"))
    if m is None:
        return np.nan
    return float(m.group(1)) if m.group(1) else np.inf


# ----------------------------------------------------------------------------- calibration
def taper_window(t, center, half=15.0, ramp=5.0):
    w = np.zeros_like(t)
    inner = np.abs(t - center) <= half
    w[inner] = 1.0
    edge = (np.abs(t - center) > half) & (np.abs(t - center) <= half + ramp)
    w[edge] = 0.5 * (1 + np.cos(np.pi * (np.abs(t[edge] - center) - half) / ramp))
    return w


def calibration_filter(t, q, p, eps_rel=1e-6):
    """W(f) from the broadside pair (q borehole, p borehole-free) on a common time axis."""
    w = taper_window(t, t[np.argmax(np.abs(q))])
    n = 4 * t.size
    Q, P = np.fft.rfft(q * w, n), np.fft.rfft(p * w, n)
    return Q * np.conj(P) / (np.abs(P) ** 2 + eps_rel * np.max(np.abs(P) ** 2))


def apply_filter(W, p):
    n = 2 * (W.size - 1)
    return np.fft.irfft(np.fft.rfft(p, n) * W, n)[: p.size]


def pair(key_bh, key_nobh):
    tb, qb = load_rahpo(key_bh)
    tn, pn = load_rahpo(key_nobh)
    if not np.allclose(tb, tn):
        raise ValueError(f"time axes of {key_bh} and {key_nobh} differ")
    return tb, qb, pn


# ----------------------------------------------------------------------------- metrics
def metrics(t_ref, ref, t, x, time_range=TIME_RANGE):
    m = compare_traces(t_ref, ref, t, x, time_range=time_range)
    return {"chi": m["Relative misfit (%)"], "nrmse": m["NRMSE (%)"], "peak_ratio": m["Peak ratio (ref/RA-HPO)"],
            "r": m["Pearson r"], "dt_peak": m["Peak-time diff (ns)"]}


def fdtd(dip, z):
    """Scattered FDTD trace of the dipping slab at the receiver at height z."""
    return load_scattered_gprmax(BY_KEY[f"dip{dip}_bh_{ztag(z)}"], rx=GATHER_Z.index(z) + 1)


def score_dip(dip, rows):
    """RA-HPO and the calibrated dipole at every receiver of the slab dipping by dip degrees."""
    W = None
    cb, cn = f"dipcal{dip}_bh", f"dipcal{dip}_nobh"
    if have(cb) and have(cn):
        W = calibration_filter(*pair(cb, cn))
    else:
        print(f"  {dip} deg: calibration pair missing, dipole not computed")
    for z in GATHER_Z:
        kb, kn = f"dip{dip}_bh_{ztag(z)}", f"dip{dip}_nobh_{ztag(z)}"
        if not have(kb):
            continue
        try:
            tf, qf = fdtd(dip, z)
        except (FileNotFoundError, OSError):
            print(f"  {dip} deg: FDTD output missing")
            return
        tr, xr = load_rahpo(kb)
        base = {"dip_deg": dip, "z_r": z, "offset_m": round(z - Z_SOURCE, 1)}
        rows.append({**base, "compared": "RA-HPO", **metrics(tf, qf, tr, xr), "t_c": horizon(kb)})
        if W is not None and have(kn):
            tn, pn = load_rahpo(kn)
            rows.append({**base, "compared": "calibrated dipole", **metrics(tf, qf, tn, apply_filter(W, pn)),
                         "t_c": horizon(kn)})


def score_thin_layers(rows):
    """RA-HPO against the exact infinite layer, per layer."""
    for name in THIN_LAYERS:
        k = f"thin_{name}"
        if not have(k):
            continue
        ta, qa = load_analytic(BY_KEY[k].analytic)
        tr, xr = load_rahpo(k)
        rows.append({"layer": name, **metrics(ta, qa, tr, xr)})


def full_coverage(rows, dip, compared):
    """Rows of one slab and model whose t_c lies outside the evaluation window."""
    return [r for r in rows if r["dip_deg"] == dip and r["compared"] == compared
            and not (TIME_RANGE[0] <= r["t_c"] <= TIME_RANGE[1])]


def summary(rows):
    lines = []
    for dip in DIPS:
        ra = [r for r in rows if r["dip_deg"] == dip and r["compared"] == "RA-HPO"]
        if not ra:
            continue
        full = full_coverage(rows, dip, "RA-HPO")
        cut = [r for r in ra if r not in full]
        lines.append(f"{dip} deg: {len(full)} receivers with full plane coverage, offsets "
                     f"{min(r['offset_m'] for r in full):+.1f} to {max(r['offset_m'] for r in full):+.1f} m")
        if cut:
            lines.append("  left out (t_c inside the window): " + ", ".join(
                f"{r['offset_m']:+.1f} m ({r['t_c']:.1f} ns)" for r in cut))
        z_full = {r["z_r"] for r in full}         # the receivers RA-HPO sees completely
        for compared in ("RA-HPO", "calibrated dipole"):
            sel = [r for r in rows if r["dip_deg"] == dip and r["compared"] == compared and r["z_r"] in z_full]
            if not sel:
                continue
            chi = [r["chi"] for r in sel]
            pk = [r["peak_ratio"] for r in sel]
            lines.append(f"  {compared:17s} chi {min(chi):.2f}-{max(chi):.2f} % (median {np.median(chi):.2f}), "
                         f"NRMSE <= {max(r['nrmse'] for r in sel):.3f} %, peak ratio {min(pk):.4f}-{max(pk):.4f} "
                         f"(median {np.median(pk):.4f}), r >= {min(r['r'] for r in sel):.5f}")
    return lines


def write_csv(path, rows):
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", default=str(METRICS_DIR))
    a = ap.parse_args()
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    dip_rows, thin_rows = [], []
    for dip in DIPS:
        score_dip(dip, dip_rows)
    score_thin_layers(thin_rows)
    if dip_rows:
        write_csv(out / "dipping.csv", dip_rows)
        print("\n".join(summary(dip_rows)))
        print(f"per receiver: {out / 'dipping.csv'} ({len(dip_rows)} rows)")
    if thin_rows:
        write_csv(out / "thin_layers.csv", thin_rows)
        print("Thin layers, RA-HPO against the exact layer:")
        for r in thin_rows:
            print(f"  {r['layer']:14s} chi {r['chi']:.2f} %, NRMSE {r['nrmse']:.3f} %, peak ratio {r['peak_ratio']:.4f}")
        print(f"per layer: {out / 'thin_layers.csv'}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
