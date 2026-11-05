"""
Compare two gprMax reference planes recorded at the same plane position.

Domain-truncation test of the near-field model. The reference plane is produced by a
fracture-free near-field FDTD model. Truncating that model's domain just beyond the plane
should leave the recorded plane unchanged. This script measures the change directly, on the
plane itself, before any propagation.

The decimated loader and the metric definitions are imported from validate_propagation.py,
so the numbers are computed by the same code as the plane-to-plane propagation test.

Metrics, per pixel of the decimated plane, with the reference plane as d and the truncated
plane as p (same convention as compare_traces and validate_propagation):
    NRMSE      = sqrt( mean_k (p_k - d_k)^2 ) / ( max_k d_k - min_k d_k )
    Pearson r  = correlation of p and d
    Peak ratio = max_k |d_k| / max_k |p_k|
    Peak-time difference = t(argmax_k |p_k|) - t(argmax_k |d_k|)

Usage:
    python tools/compare_ref_planes.py TRUNCATED.out REFERENCE.out
    python tools/compare_ref_planes.py plane_4m.out plane_8m.out --comps Ez Ex Ey

Metric maps are written to output/comparisons/ unless --outdir is set.

Both files must record the same plane position, the same receiver grid and the same time
step; the script checks all three and refuses to compare mismatched planes.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib import font_manager

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from rahpo.paths import COMPARISONS  # noqa: E402
from validate_propagation import compute_metrics, load_gprmax_plane_decimated  # noqa: E402

# Acceptance thresholds of the test
NRMSE_MEDIAN_MAX = 0.005      # 0.5 %
PEAK_RATIO_TOL = 0.01         # |median peak ratio - 1|
PEAK_TIME_TOL_S = 0.02e-9     # one FDTD time step of the 0.01 m grid

FIGSIZE = (6.6, 2.3)
FONT_SIZE = 8


def set_style():
    arial = Path("/System/Library/Fonts/Supplemental/Arial.ttf")
    if arial.exists():
        font_manager.fontManager.addfont(str(arial))
        mpl.rcParams["font.family"] = "Arial"
    else:
        # Servers usually lack Arial; fall back to a metric-compatible sans-serif if present
        mpl.rcParams["font.family"] = ["Arial", "Liberation Sans", "Helvetica", "DejaVu Sans"]
    mpl.rcParams.update({
        "font.size": FONT_SIZE,
        "axes.labelsize": FONT_SIZE,
        "xtick.labelsize": FONT_SIZE - 1,
        "ytick.labelsize": FONT_SIZE - 1,
        "axes.linewidth": 0.6,
        "pdf.fonttype": 42,
        "savefig.dpi": 300,
    })


def check_planes_match(a, b, name_a, name_b, x_tol=1e-3):
    """Refuse to compare planes that are not the same grid at the same position."""
    (x_a, yy_a, zz_a, dt_a), (x_b, yy_b, zz_b, dt_b) = a, b
    if abs(x_a - x_b) > x_tol:
        raise ValueError(f"Plane positions differ: {name_a} at x={x_a}, {name_b} at x={x_b}.")
    if yy_a.shape != yy_b.shape or zz_a.shape != zz_b.shape:
        raise ValueError(
            f"Receiver grids differ: {name_a} is {zz_a.size}x{yy_a.size}, "
            f"{name_b} is {zz_b.size}x{yy_b.size}.")
    if not (np.allclose(yy_a, yy_b, rtol=0, atol=1e-9)
            and np.allclose(zz_a, zz_b, rtol=0, atol=1e-9)):
        raise ValueError(f"Receiver coordinates differ between {name_a} and {name_b}.")
    if not np.isclose(dt_a, dt_b, rtol=1e-9, atol=0.0):
        raise ValueError(f"Time steps differ: {name_a} dt={dt_a}, {name_b} dt={dt_b}.")


def plot_maps(metrics, yy, zz, comp, out_png):
    """Three maps over the plane: NRMSE, peak ratio, peak-time difference."""
    panels = [
        (metrics["NRMSE_map"] * 100.0, f"NRMSE (%), ${comp}$", "viridis", None),
        (metrics["peak_ratio_map"], "Peak ratio (ref / truncated)", "coolwarm", None),
        (metrics["dT_peak_map"] * 1e9, "Peak-time difference (ns)", "coolwarm", None),
    ]
    fig, axes = plt.subplots(1, 3, figsize=FIGSIZE, constrained_layout=True)
    for ax, (data, label, cmap, _) in zip(axes, panels):
        # Diverging maps are centered on their physical neutral value
        if cmap == "coolwarm":
            center = 1.0 if "ratio" in label else 0.0
            span = max(abs(np.nanmin(data) - center), abs(np.nanmax(data) - center), 1e-12)
            im = ax.pcolormesh(yy, zz, data, cmap=cmap, shading="nearest",
                               vmin=center - span, vmax=center + span)
        else:
            im = ax.pcolormesh(yy, zz, data, cmap=cmap, shading="nearest")
        ax.set_xlabel("y (m)")
        ax.set_ylabel("z (m)")
        ax.set_aspect("equal")
        cb = fig.colorbar(im, ax=ax, shrink=0.9, pad=0.02)
        cb.set_label(label)
        cb.ax.tick_params(labelsize=FONT_SIZE - 1, width=0.6, length=2)
        cb.outline.set_linewidth(0.6)
    fig.savefig(out_png)
    plt.close(fig)


def compare(trunc_file, ref_file, comps, y_step, z_step, outdir):
    trunc_name, ref_name = Path(trunc_file).stem, Path(ref_file).stem

    print(f"Loading {trunc_file} (decimated {y_step} x {z_step}) ...")
    f_trunc, x_t, yy_t, zz_t, dt_t, nt_t = load_gprmax_plane_decimated(
        trunc_file, comps=tuple(comps), y_step=y_step, z_step=z_step)
    print(f"Loading {ref_file} (decimated {y_step} x {z_step}) ...")
    f_ref, x_r, yy_r, zz_r, dt_r, nt_r = load_gprmax_plane_decimated(
        ref_file, comps=tuple(comps), y_step=y_step, z_step=z_step)

    check_planes_match((x_t, yy_t, zz_t, dt_t), (x_r, yy_r, zz_r, dt_r), trunc_name, ref_name)
    print(f"Both planes at x = {x_r:.3f} m, {zz_r.size} x {yy_r.size} pixels, dt = {dt_r:.4e} s")

    t_trunc = np.arange(nt_t) * dt_t
    t_ref = np.arange(nt_r) * dt_r

    all_pass = True
    for comp in comps:
        if comp not in f_trunc or comp not in f_ref:
            print(f"[skip] {comp}: not present in both files")
            continue

        # align_method='none': the two models share a grid and a time window, so any shift
        # would be a real difference, not a sampling artifact, and must not be removed.
        metrics, _, _, _ = compute_metrics(
            f_trunc[comp], f_ref[comp], t_trunc, t_ref, align_method="none")

        med_nrmse = float(np.median(metrics["NRMSE_map"]))
        med_ratio = float(np.median(metrics["peak_ratio_map"]))
        med_dt = float(np.median(metrics["dT_peak_map"]))
        checks = [
            ("median NRMSE", med_nrmse, NRMSE_MEDIAN_MAX, f"{med_nrmse*100:.3f}% < 0.5%"),
            ("median |peak ratio - 1|", abs(med_ratio - 1.0), PEAK_RATIO_TOL,
             f"{abs(med_ratio-1.0):.4f} < {PEAK_RATIO_TOL}"),
            ("median |peak-time diff|", abs(med_dt), PEAK_TIME_TOL_S,
             f"{abs(med_dt)*1e9:.4f} ns <= {PEAK_TIME_TOL_S*1e9:.2f} ns"),
        ]
        comp_pass = all(value <= limit for _, value, limit, _ in checks)
        all_pass &= comp_pass

        print(f"\n=== {trunc_name} vs {ref_name} ({comp}) ===")
        print(metrics["summary"])
        print("--- acceptance ---")
        for label, value, limit, text in checks:
            print(f"  [{'PASS' if value <= limit else 'FAIL'}] {label}: {text}")
        print(f"  {comp}: {'PASS' if comp_pass else 'FAIL'}")

        stem = f"compare_planes_{trunc_name}_vs_{ref_name}_{comp}"
        np.savez(Path(outdir) / f"{stem}.npz",
                 NRMSE_map=metrics["NRMSE_map"], corr_map=metrics["corr_map"],
                 peak_ratio_map=metrics["peak_ratio_map"], dT_peak_map=metrics["dT_peak_map"],
                 yy=yy_r, zz=zz_r, x_plane=x_r, window_s=np.array(metrics["window_s"]))
        plot_maps(metrics, yy_r, zz_r, comp, Path(outdir) / f"{stem}.png")
        print(f"  Saved {stem}.npz and {stem}.png")

    print(f"\nOVERALL ({trunc_name} vs {ref_name}): {'PASS' if all_pass else 'FAIL'}")
    return all_pass


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("truncated", help="gprMax .out of the truncated-domain plane")
    ap.add_argument("reference", help="gprMax .out of the plane from the full domain")
    ap.add_argument("--comps", nargs="+", default=["Ez", "Ex", "Ey"],
                    choices=["Ex", "Ey", "Ez"], help="field components to compare")
    ap.add_argument("--y-step", type=int, default=4, help="receiver decimation in y")
    ap.add_argument("--z-step", type=int, default=4, help="receiver decimation in z")
    ap.add_argument("--outdir", default=str(COMPARISONS),
                    help="directory for the .npz and .png outputs")
    args = ap.parse_args()

    set_style()
    Path(args.outdir).mkdir(parents=True, exist_ok=True)
    ok = compare(args.truncated, args.reference, args.comps,
                 args.y_step, args.z_step, args.outdir)
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
