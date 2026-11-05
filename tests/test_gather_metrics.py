"""Test of tools/gather_metrics.py, the gather figure of accuracy against offset.

Three checks, on synthetic tables (no data needed):
  1. load() returns the rows of a shuffled gather.csv sorted by offset, with every value of the
     row it came from;
  2. main() writes the figure from the gather.csv in RAHPO_METRICS_DIR into RAHPO_FIGURE_DIR,
     with the two panels chi and peak ratio;
  3. load_dip() reads a dipping.csv as written by tools/dip_study.py, keeps only the receivers
     whose RA-HPO trace has full plane coverage (t_c outside the evaluation window), for RA-HPO
     and the calibrated dipole alike, groups them by dip and model sorted by offset, and main()
     --dipping writes the figure.
"""
from __future__ import annotations

import csv
import os
import random
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path[:0] = [str(ROOT), str(ROOT / "tools")]

import gather_metrics as gm  # noqa: E402

FIELDS = ["rx", "z_m", "offset_m", "chi_pct", "nrmse_pct", "r", "peak_ratio", "peak_dt_ns"]
DIP_FIELDS = ["dip_deg", "z_r", "offset_m", "compared", "chi", "nrmse", "peak_ratio", "r", "dt_peak", "t_c"]
Z = [1.5 + 0.5 * k for k in range(17) if k != 8]


def write_gather(path, zs=Z):
    rows = [{"rx": i + 1, "z_m": z, "offset_m": z - 5.5, "chi_pct": 3 + i, "nrmse_pct": 0.3 + 0.01 * i,
             "r": 0.999, "peak_ratio": 1 + 0.001 * i, "peak_dt_ns": 0} for i, z in enumerate(zs)]
    random.Random(1).shuffle(rows)
    with open(path, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=FIELDS)
        w.writeheader()
        w.writerows(rows)
    return rows


def run_main(tmp, *args):
    env = dict(os.environ, RAHPO_METRICS_DIR=str(tmp), RAHPO_FIGURE_DIR=str(tmp / "fig"),
               PYTHONPATH=str(ROOT), MPLBACKEND="Agg")
    subprocess.run([sys.executable, str(ROOT / "tools" / "gather_metrics.py"), *args], check=True, env=env,
                   capture_output=True, text=True)
    pdf = tmp / "fig" / f"{gm.FIGURE}.pdf"
    assert pdf.exists() and pdf.stat().st_size > 1000, pdf


def test_load_sorts_and_keeps_values(tmp_path):
    rows = write_gather(tmp_path / "gather.csv")
    d = gm.load(tmp_path / "gather.csv")
    assert list(d["offset_m"]) == sorted(r["offset_m"] for r in rows)
    by_off = {r["offset_m"]: r for r in rows}
    for k in ("chi_pct", "nrmse_pct", "peak_ratio"):
        assert all(abs(v - by_off[o][k]) < 1e-12 for o, v in zip(d["offset_m"], d[k])), k


def test_main_writes_figure(tmp_path):
    write_gather(tmp_path / "gather.csv")
    run_main(tmp_path)
    assert [k for k, _ in gm.PANELS] == ["chi_pct", "peak_ratio"]


def test_dipping(tmp_path):
    rows = []
    for dip in (20, 30):
        for compared in ("RA-HPO", "calibrated dipole"):
            for z in (4.5, 1.5, 3.0, 6.0, 8.0):
                # at 30 deg the receiver at z = 8.0 m loses part of the fracture inside the window
                t_c = 140.0 if (dip == 30 and z == 8.0) else float("inf")
                rows.append({"dip_deg": dip, "z_r": z, "offset_m": round(z - 5.5, 1), "compared": compared,
                             "chi": dip / 3 + z, "nrmse": 0.5, "peak_ratio": 1 + dip / 100 * (compared != "RA-HPO"),
                             "r": 0.999, "dt_peak": 0.0, "t_c": t_c if compared == "RA-HPO" else float("nan")})
    random.Random(2).shuffle(rows)
    with open(tmp_path / "dipping.csv", "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=DIP_FIELDS)
        w.writeheader()
        w.writerows(rows)
    g = gm.load_dip(tmp_path / "dipping.csv")
    assert sorted(g) == [(20, "RA-HPO"), (20, "calibrated dipole"), (30, "RA-HPO"), (30, "calibrated dipole")]
    for (dip, _), d in g.items():
        expect = [-4.0, -2.5, -1.0, 0.5] + ([] if dip == 30 else [2.5])
        assert list(d["offset_m"]) == expect, (dip, d["offset_m"])
        assert all(abs(c - (dip / 3 + o + 5.5)) < 1e-12 for o, c in zip(d["offset_m"], d["chi_pct"]))
    write_gather(tmp_path / "gather.csv", (1.5, 4.5, 6.5, 9.5))
    run_main(tmp_path, "--dipping", str(tmp_path / "dipping.csv"))
