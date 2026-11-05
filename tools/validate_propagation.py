"""
Validate the ray-aligned propagation operator in isolation.

The field recorded by gprMax on a reference plane (x = x1) is propagated along rays to a
target plane (x = x2) and compared with the field simulated directly by gprMax on x2.

The default inputs are the outputs of the two models in
examples/gprMax_models/For_ray-align_propagation_verification/: the plane at x = 3.5 m is propagated to
x = 6.5 m, where the second model records the field directly.

Usage:
    python tools/validate_propagation.py Ez
    python tools/validate_propagation.py Ex --plane PLANE.out --target TARGET.out

Products are written to output/propagation/.

Metrics (per pixel (i, j) of the decimated plane; d = FDTD, p = propagated field; sums and
means run over the common time window of N samples):
  * NRMSE (same definition as compare_traces in plot_traces.py):
        NRMSE_ij = sqrt( (1/N) sum_k (p_ijk - d_ijk)^2 ) / ( max_k d_ijk - min_k d_ijk )
  * Pearson correlation coefficient between p_ij and d_ij.
  * Peak ratio (FDTD / propagated): max_k |d_ijk| / max_k |p_ijk|.
  * Peak-time difference: t(argmax_k |p_ijk|) - t(argmax_k |d_ijk|).
  * Additionally: the global relative L2 error and the RMS-normalized NRMSE
    sqrt(mean e^2 / mean d^2).

Both planes are decimated at load: only the receivers kept by (y_step, z_step) are read, and
the coordinate axes are built from the Position attributes of the first row and column.
"""
import argparse
import sys
from pathlib import Path

import h5py
import numpy as np
import matplotlib.pyplot as plt

_REPO = Path(__file__).resolve().parents[1]
if str(_REPO) not in sys.path:
    sys.path.insert(0, str(_REPO))
from rahpo.paths import PROPAGATION  # noqa: E402
from matplotlib.animation import FuncAnimation, PillowWriter
from scipy.interpolate import interp1d
import matplotlib.ticker as mticker

# ========== Test configuration (models in For_ray-align_propagation_verification/) ==========
_MODELS = _REPO / "examples" / "gprMax_models" / "For_ray-align_propagation_verification"
PLANE_FILE = str(_MODELS / "3D_8x10x10-shifted_nofrac_tx5pt0_planex3_h0pt01.out")   # recorded plane, x = 3.5 m
TARGET_FILE = str(_MODELS / "3D_8x10x10-shifted_nofrac_tx5pt0_planex6_h0pt01.out")  # simulated plane at x2
x2 = 6.5  # target plane x [m]
x_tol = 1e-3  # meters tolerance on the plane position
src_pos = (0.5, 5.0, 5.0)  # (x_s, y_s, z_s) in meters

eps_r = 9.0
mu_r  = 1.0
sigma = 1e-4  # S/m
pattern_regularizer = 1e-6  # stabilizes pattern ratio near nulls

# Spatial decimation (keep every y_step-th and z_step-th trace)
y_step = 4
z_step = 4

# =============================================================================================

# Physical constants
c0 = 299792458.0
mu0 = 4e-7 * np.pi
eps0 = 1.0 / (mu0 * c0**2)
mu  = mu0 * mu_r
eps = eps0 * eps_r


def _read_plane_axes(f, round_decimals=9, x_tol=1e-3):
    """
    Infer the receiver-grid layout of a single constant-x rx_array plane from the Position
    attributes of rx1, rx2, rx{nrx}, the first row, and the first column.

    Returns:
      x_plane: plane x [m]
      yy, zz: ascending coordinate axes (rounded as in np.unique of rounded positions)
      y_order, z_order: native grid index of each ascending coordinate
      rx_index: function (iy_native, iz_native) -> 1-based receiver index
    """
    nrx = int(f.attrs['nrx'])

    def pos(k):
        return np.asarray(f[f"/rxs/rx{k}"].attrs['Position'], dtype=float)

    p1, pN = pos(1), pos(nrx)
    if nrx < 2:
        raise ValueError("A plane needs at least two receivers.")
    p2 = pos(2)
    if max(abs(p2[0] - p1[0]), abs(pN[0] - p1[0])) > x_tol:
        raise ValueError("Receivers are not on a single constant-x plane.")

    # Same layout inference as ra_hpo.load_gprmax_plane
    if abs(p2[1] - p1[1]) > 1e-9:      # y varies fastest
        ny = int(np.round(abs(pN[1] - p1[1]) / abs(p2[1] - p1[1]))) + 1
        nz = nrx // ny

        def rx_index(iy, iz):
            return iz * ny + iy + 1
    elif abs(p2[2] - p1[2]) > 1e-9:    # z varies fastest
        nz = int(np.round(abs(pN[2] - p1[2]) / abs(p2[2] - p1[2]))) + 1
        ny = nrx // nz

        def rx_index(iy, iz):
            return iy * nz + iz + 1
    else:
        raise ValueError("Cannot infer the receiver-grid layout from rx1 and rx2.")
    if ny * nz != nrx:
        raise ValueError(f"Inferred grid {ny} x {nz} does not match nrx = {nrx}.")

    y_native = np.array([pos(rx_index(iy, 0))[1] for iy in range(ny)])
    z_native = np.array([pos(rx_index(0, iz))[2] for iz in range(nz)])
    y_round = np.round(y_native, round_decimals)
    z_round = np.round(z_native, round_decimals)
    if np.unique(y_round).size != ny or np.unique(z_round).size != nz:
        raise ValueError("Receiver coordinates along the first row/column are not unique.")

    y_order = np.argsort(y_round, kind='stable')
    z_order = np.argsort(z_round, kind='stable')
    x_plane = float(p1[0])
    return x_plane, y_round[y_order], z_round[z_order], y_order, z_order, rx_index


def load_gprmax_plane_decimated(filename, comps=('Ez',), y_step=1, z_step=1,
                                round_decimals=9, x_tol=1e-3, n_spot=10):
    """
    Load Ex/Ey/Ez time series on a constant-x plane from a gprMax HDF5 .out file, reading
    only every y_step-th / z_step-th receiver (from offset 0 on the ascending axes).

    Returns:
      fields: dict comp -> array (nz_ds, ny_ds, Nt), float64
      x_plane: plane x [m]
      yy, zz: decimated coordinate axes (ny_ds,), (nz_ds,)
      dt: time step [s]
      Nt: number of time samples
    """
    if isinstance(comps, str):
        comps = (comps,)
    if y_step < 1 or z_step < 1:
        raise ValueError("y_step and z_step must be >= 1.")

    with h5py.File(filename, 'r') as f:
        dt = float(f.attrs['dt'])
        Nt = int(f.attrs['Iterations'])
        x_plane, yy_all, zz_all, y_order, z_order, rx_index = _read_plane_axes(
            f, round_decimals=round_decimals, x_tol=x_tol)

        yy = yy_all[::y_step]
        zz = zz_all[::z_step]
        sel_y = y_order[::y_step]
        sel_z = z_order[::z_step]

        # (1-based receiver index, iz_ds, iy_ds), read in increasing receiver index
        sel = sorted((rx_index(int(iy_nat), int(iz_nat)), iz_ds, iy_ds)
                     for iz_ds, iz_nat in enumerate(sel_z)
                     for iy_ds, iy_nat in enumerate(sel_y))

        # Spot check: stored positions of a few selected receivers match the inferred axes
        for i in np.unique(np.linspace(0, len(sel) - 1, n_spot).astype(int)):
            k, iz, iy = sel[i]
            p = np.asarray(f[f"/rxs/rx{k}"].attrs['Position'], dtype=float)
            if (abs(p[0] - x_plane) > x_tol
                    or not np.isclose(np.round(p[1], round_decimals), yy[iy], rtol=0, atol=1e-9)
                    or not np.isclose(np.round(p[2], round_decimals), zz[iz], rtol=0, atol=1e-9)):
                raise ValueError(f"Position check failed for rx{k} in {filename}: {p}")

        fields = {c: np.empty((zz.size, yy.size, Nt), dtype=np.float64)
                  for c in comps if f"/rxs/rx1/{c}" in f}
        for k, iz, iy in sel:
            g = f[f"/rxs/rx{k}"]
            for c in fields:
                fields[c][iz, iy, :] = g[c][:]

    return fields, x_plane, yy, zz, dt, Nt


def gamma_of_omega(omega, mu, eps, sigma):
    # gamma = sqrt(j ω μ (σ + j ω ε)), principal branch with Re(gamma) > 0
    eta = sigma + 1j * omega * eps
    return np.sqrt(1j * omega * mu * eta)

def next_pow2(n):
    return 1 << (int(n - 1).bit_length())


def propagate_ff_component(E1_t, dt, x1, x2, yy, zz, src_pos, mu, eps, sigma, comp, reg=1e-6):
    """
    Same-ray propagation of one component from x=x1 to x=x2.
    For each target (y2,z2) at x2, we pick the field at the intersection point
    (y1,z1) where the ray from the source through (x2,y2,z2) hits x1, and then
    apply the exact along-ray ratio (near/induction/radiation terms included).

    Inputs:
      E1_t: (nz, ny, Nt) at plane x1 (on grid zz x yy)
      dt: time step [s]
      x1, x2: planes [m]
      yy, zz: coordinate axes (ny,), (nz,) (sorted ascending)
      src_pos: (xs, ys, zs)
      comp: 'Ex'|'Ey'|'Ez'
      reg: unused here (kept for signature compatibility)

    Returns:
      E2_t: (nz, ny, Nt_out) propagated field
      t2:   (Nt_out,) time axis (aligned to original t=0)
    """
    xs, ys, zs = src_pos
    if np.isclose(x2, xs):
        raise ValueError("x2 coincides with source x; ray parameter λ is undefined.")

    nz, ny, Nt = E1_t.shape
    # Target plane coordinates (x = x2)
    Y2, Z2 = np.meshgrid(yy, zz, indexing='xy')  # (nz, ny)

    # Same-ray mapping: intersection with plane x1
    lam = (x1 - xs) / (x2 - xs)  # constant across the plane
    Y1m = ys + lam * (Y2 - ys)   # (nz, ny)
    Z1m = zs + lam * (Z2 - zs)   # (nz, ny)

    # Ranges along the same ray
    R2 = np.sqrt((x2 - xs)**2 + (Y2 - ys)**2 + (Z2 - zs)**2)   # (nz, ny)
    R1 = np.sqrt((x1 - xs)**2 + (Y1m - ys)**2 + (Z1m - zs)**2) # (nz, ny)
    dR = R2 - R1

    # Direction cosines along the ray (use target point to define the ray)
    nx_ray = (x2 - xs) / R2
    ny_ray = (Y2 - ys) / R2
    nz_ray = (Z2 - zs) / R2  # constant along the ray

    # Time padding based on dR bounds (prevent circular wrap)
    c_med = 1.0 / np.sqrt(mu * eps)
    tau_min = np.min(dR) / c_med
    tau_max = np.max(dR) / c_med
    Npre  = int(np.ceil(max(0.0, -tau_min) / dt))
    Npost = int(np.ceil(max(0.0,  tau_max) / dt))

    # Zero-mean and time padding at x1
    E1_0 = E1_t - E1_t.mean(axis=2, keepdims=True)
    Nt_out = Nt + Npre + Npost
    E1_pad = np.zeros((nz, ny, Nt_out), dtype=np.float64)
    E1_pad[:, :, Npre:Npre+Nt] = E1_0

    # Bilinear interpolation on the x1 plane to sample E1 at (Y1m, Z1m)
    # Compute bracketing indices and weights along y and z
    def interp_weights(axis, coords):
        # axis: (N,), coords: (nz, ny)
        idx0 = np.searchsorted(axis, coords) - 1
        idx0 = np.clip(idx0, 0, len(axis) - 2)
        a0 = axis[idx0]
        a1 = axis[idx0 + 1]
        w = (coords - a0) / (a1 - a0 + np.finfo(float).eps)
        w = np.clip(w, 0.0, 1.0)
        return idx0, w

    iy0, wy = interp_weights(yy, Y1m)
    iz0, wz = interp_weights(zz, Z1m)

    # Gather four neighbors and blend
    # Shapes: (nz, ny, Nt_out)
    E00 = E1_pad[iz0,     iy0,     :]
    E01 = E1_pad[iz0,     iy0 + 1, :]
    E10 = E1_pad[iz0 + 1, iy0,     :]
    E11 = E1_pad[iz0 + 1, iy0 + 1, :]

    w00 = (1.0 - wz) * (1.0 - wy)
    w01 = (1.0 - wz) * wy
    w10 = wz * (1.0 - wy)
    w11 = wz * wy

    E1_ray_pad = (w00[..., None] * E00 +
                  w01[..., None] * E01 +
                  w10[..., None] * E10 +
                  w11[..., None] * E11)

    # Frequency axis and complex wavenumber
    Nfft = next_pow2(Nt_out)
    fpos = np.fft.rfftfreq(Nfft, dt)
    omega = 2.0 * np.pi * fpos
    gamma = gamma_of_omega(omega, mu, eps, sigma)  # (Npos,)

    # Build along-ray ratio G(y,z,omega)
    R13 = R1[..., None]
    R23 = R2[..., None]
    dR3 = dR[..., None]
    g3  = gamma[None, None, :]

    attph = np.exp(-g3 * dR3)

    if comp in ('Ex', 'Ey'):
        # Exact transverse components along same ray (pattern cancels):
        # G = (R1/R2)^3 * [P(gR2)/P(gR1)] * e^{-g(R2-R1)}
        gR1 = g3 * R13
        gR2 = g3 * R23
        P1 = 3.0 + 3.0 * gR1 + gR1 * gR1
        P2 = 3.0 + 3.0 * gR2 + gR2 * gR2
        G = ((R13 / R23) ** 3) * (P2 / P1) * attph

    elif comp == 'Ez':
        # Ez shape along same ray (use nz from the ray, identical at R1 and R2)
        nzr = nz_ray[..., None]
        gR1 = g3 * R13
        gR2 = g3 * R23

        S1_num = ((3.0 * nzr * nzr - 1.0) * (1.0 + gR1)) + ((nzr * nzr - 1.0) * (gR1 * gR1))
        S2_num = ((3.0 * nzr * nzr - 1.0) * (1.0 + gR2)) + ((nzr * nzr - 1.0) * (gR2 * gR2))
        S1 = S1_num / (R13 ** 3)
        S2 = S2_num / (R23 ** 3)

        G = (S2 / S1) * attph
    else:
        raise ValueError("Component must be 'Ex', 'Ey', or 'Ez'.")

    # Apply propagation in frequency domain
    E1_pos = np.fft.rfft(E1_ray_pad, n=Nfft, axis=2)  # (nz, ny, Npos)
    E2_pos = E1_pos * G
    E2_full = np.fft.irfft(E2_pos, n=Nfft, axis=2)

    # Crop to padded record length and time axis
    E2_t = E2_full[:, :, :Nt_out]
    t2 = (np.arange(Nt_out) - Npre) * dt
    return E2_t, t2

def animate_plane(E_t, yy, zz, dt=None, comp='Ez', compLabel='E_z',
                  gif_name='plane_field.gif', nFrames=60, fps=20, time=None):
    """
    E_t: (nz, ny, Nt) array
    Provide either dt or a full time vector 'time' of length Nt.
    """
    Nt = E_t.shape[2]
    if time is None:
        if dt is None:
            raise ValueError("Provide either 'dt' or 'time'.")
        time = np.arange(Nt) * dt

    t_frames = np.linspace(time[0], time[-1], nFrames)
    idx_frames = np.clip(np.searchsorted(time, t_frames), 0, Nt - 1)

    vals = np.abs(E_t[:, :, idx_frames])
    vmax = np.percentile(vals, 99.0)
    if vmax == 0:
        vmax = np.max(np.abs(E_t))
    clim = (-3.0, 3.0)

    fig, ax = plt.subplots(figsize=(3, 4.5))
    im = ax.imshow(E_t[:, :, idx_frames[0]],
                   extent=[yy[0], yy[-1], zz[0], zz[-1]],
                   origin='lower', aspect='equal', cmap='turbo',
                   vmin=clim[0], vmax=clim[1])
    fig.colorbar(im, ax=ax)
    ax.set_xlabel('y [m]')
    ax.set_ylabel('z [m]')
    title = ax.set_title(f'{compLabel} at t = {time[idx_frames[0]]*1e9:.3f} ns')

    def update(frame_idx):
        idxk = idx_frames[frame_idx]
        im.set_data(E_t[:, :, idxk])
        title.set_text(f'{compLabel} at t = {time[idxk]*1e9:.3f} ns')
        return im, title

    ani = FuncAnimation(fig, update, frames=len(idx_frames), blit=False)
    ani.save(gif_name, writer=PillowWriter(fps=fps))
    plt.close(fig)
    print(f"Saved animation to {gif_name} (component {comp})")

def regrid_nearest_plane(E_src, yy_src, zz_src, yy_tgt, zz_tgt):
    """
    Nearest-neighbor regridding of a 3D field from (zz_src, yy_src, Nt) grid
    to (zz_tgt, yy_tgt, Nt). Fast and no SciPy dependency.

    E_src: (nz_src, ny_src, Nt)
    Returns: E_tgt: (nz_tgt, ny_tgt, Nt)
    """
    nz_src, ny_src, Nt = E_src.shape

    # Build nearest index maps along y and z
    def nearest_indices(x_src, x_tgt):
        # assumes x_src is sorted; returns array of indices in src for each tgt
        idx = np.searchsorted(x_src, x_tgt)
        idx = np.clip(idx, 1, len(x_src)-1)
        left = idx - 1
        right = idx
        choose_right = (np.abs(x_tgt - x_src[left]) > np.abs(x_tgt - x_src[right]))
        idx = np.where(choose_right, right, left)
        return idx

    y_idx = nearest_indices(yy_src, yy_tgt)  # shape (ny_tgt,)
    z_idx = nearest_indices(zz_src, zz_tgt)  # shape (nz_tgt,)

    # Advanced indexing to pick nearest samples on both axes
    E_tgt = E_src[z_idx[:, None], y_idx[None, :], :]
    return E_tgt

def resample_to_time(E_src, t_src, t_tgt):
    f = interp1d(t_src, E_src, kind='linear', axis=-1,
                 bounds_error=False, fill_value=0.0, assume_sorted=True)
    return f(t_tgt)

def compute_metrics(E_pred, E_meas, t_pred, t_meas, eps=1e-12, align_method='auto', max_shift_s=2e-7):
    """
    Align time windows and compute metrics (definitions in the module docstring).
    align_method:
      - 'none' : no global shift
      - 'xcorr': estimate global integer-sample shift by cross-correlating spatial averages
      - 'auto' : use 'xcorr' if dt differs, else 'none'
    """
    # 1) Build overlapping time window
    t0 = max(t_pred[0], t_meas[0])
    t1 = min(t_pred[-1], t_meas[-1])
    if t1 <= t0:
        raise ValueError("No overlapping time window between predicted and measured time axes.")

    i0_pred = int(np.searchsorted(t_pred, t0, side='left'))
    i1_pred = int(np.searchsorted(t_pred, t1, side='right'))
    i0_meas = int(np.searchsorted(t_meas, t0, side='left'))
    i1_meas = int(np.searchsorted(t_meas, t1, side='right'))

    t_pred_c = t_pred[i0_pred:i1_pred]
    t_meas_c = t_meas[i0_meas:i1_meas]

    # 2) If dt differs, resample measured onto predicted grid for fair comparison
    dtp = float(np.median(np.diff(t_pred_c)))
    dtm = float(np.median(np.diff(t_meas_c)))
    E_pred_c = E_pred[:, :, i0_pred:i1_pred]
    E_meas_c = E_meas[:, :, i0_meas:i1_meas]

    if not np.isclose(dtp, dtm, rtol=1e-6, atol=1e-12):
        print(f"Resampling measured from dt={dtm} to dt={dtp} for comparison.")
        E_meas_c = resample_to_time(E_meas_c, t_meas_c, t_pred_c)
        t_common = t_pred_c
    else:
        # Make same length by trimming to min length
        L = min(len(t_pred_c), len(t_meas_c))
        t_pred_c = t_pred_c[:L]
        t_meas_c = t_meas_c[:L]
        E_pred_c = E_pred_c[:, :, :L]
        E_meas_c = E_meas_c[:, :, :L]
        t_common = t_pred_c  # same as t_meas_c

    # 3) Optional global time shift (integer samples) via xcorr of spatial averages
    if align_method == 'auto':
        align_method = 'xcorr' if not np.isclose(dtp, dtm) else 'none'

    if align_method == 'xcorr':
        # Spatial average of envelopes
        pe = np.mean(np.abs(E_pred_c), axis=(0, 1))
        me = np.mean(np.abs(E_meas_c), axis=(0, 1))
        K = int(max_shift_s / dtp)
        # Cross-correlation limited to ±K lags
        lags = np.arange(-K, K + 1)
        # Compute correlation at each lag
        corr_vals = []
        for k in lags:
            if k >= 0:
                a = pe[k:]
                b = me[:len(a)]
            else:
                a = pe[:k]
                b = me[-k:len(me)]
            if len(a) == 0:
                corr_vals.append(0.0)
            else:
                corr_vals.append(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b) + eps))
        k_best = lags[int(np.argmax(corr_vals))]

        # Apply integer-sample shift to measured and re-trim to equal length
        if k_best > 0:
            E_meas_c = E_meas_c[:, :, k_best:]
            E_pred_c = E_pred_c[:, :, :E_meas_c.shape[2]]
            t_common = t_common[:E_meas_c.shape[2]]
        elif k_best < 0:
            E_pred_c = E_pred_c[:, :, -k_best:]
            E_meas_c = E_meas_c[:, :, :E_pred_c.shape[2]]
            t_common = t_common[:E_pred_c.shape[2]]
        print(f"Applied global time shift (meas vs pred): {k_best} samples ({k_best*dtp*1e9:.3f} ns)")

    # 4) Metrics
    err2 = (E_pred_c - E_meas_c) ** 2
    mse_map = err2.mean(axis=2)

    # NRMSE (primary): RMSE normalized by the peak-to-peak range of the FDTD trace
    range_map = E_meas_c.max(axis=2) - E_meas_c.min(axis=2)
    nrmse_map = np.sqrt(mse_map) / (range_map + eps)

    # Peak ratio FDTD / propagated (same direction as compare_traces)
    peak_ratio_map = np.max(np.abs(E_meas_c), axis=2) / (np.max(np.abs(E_pred_c), axis=2) + eps)

    # Additional global metrics
    num = np.linalg.norm((E_pred_c - E_meas_c).ravel())
    den = np.linalg.norm(E_meas_c.ravel()) + eps
    relL2 = num / den
    power_map = (E_meas_c ** 2).mean(axis=2) + eps
    nrmse_rms_map = np.sqrt(mse_map / power_map)

    nz, ny, Lc = E_pred_c.shape
    P = E_pred_c.reshape(nz*ny, Lc)
    M = E_meas_c.reshape(nz*ny, Lc)
    Pz = P - P.mean(axis=1, keepdims=True)
    Mz = M - M.mean(axis=1, keepdims=True)
    num_c = np.sum(Pz * Mz, axis=1)
    den_c = np.sqrt(np.sum(Pz**2, axis=1) * np.sum(Mz**2, axis=1)) + eps
    corr = (num_c / den_c).reshape(nz, ny)

    k_pred = np.argmax(np.abs(E_pred_c), axis=2)
    k_meas = np.argmax(np.abs(E_meas_c), axis=2)
    dT_peak_map = t_common[k_pred] - t_common[k_meas]

    metrics = {
        'NRMSE_map': nrmse_map,
        'peak_ratio_map': peak_ratio_map,
        'corr_map': corr,
        'dT_peak_map': dT_peak_map,
        'relL2': relL2,
        'NRMSE_rms_map': nrmse_rms_map,
        'window_s': (float(t_common[0]), float(t_common[-1])),
        'summary': (
            f"Window: {t_common[0]*1e9:.3f} to {t_common[-1]*1e9:.3f} ns ({Lc} samples), "
            f"{nz} x {ny} pixels\n"
            f"NRMSE (range-normalized) map: median={np.median(nrmse_map)*100:.3f}%, "
            f"90th pct={np.percentile(nrmse_map,90)*100:.3f}%, "
            f"99th pct={np.percentile(nrmse_map,99)*100:.3f}%\n"
            f"Correlation map: median={np.median(corr):.4f}, 10th pct={np.percentile(corr,10):.4f}, "
            f"1st pct={np.percentile(corr,1):.4f}\n"
            f"Peak ratio (FDTD/propagated) map: median={np.median(peak_ratio_map):.4f}, "
            f"10th pct={np.percentile(peak_ratio_map,10):.4f}, "
            f"90th pct={np.percentile(peak_ratio_map,90):.4f}\n"
            f"Peak-time diff (ns): median={np.median(dT_peak_map)*1e9:.3f}, "
            f"90th pct={np.percentile(np.abs(dT_peak_map),90)*1e9:.3f}\n"
            f"--- additional ---\n"
            f"Global relative L2 error: {relL2:.4e}\n"
            f"NRMSE map: median={np.median(nrmse_rms_map):.3f}, 90th pct={np.percentile(nrmse_rms_map,90):.3f}\n"
            f"Correlation map: median={np.median(corr):.3f}, 10th pct={np.percentile(corr,10):.3f}\n"
            f"Peak-time diff (ns): median={np.median(dT_peak_map)*1e9:.3f}, "
            f"90th pct={np.percentile(np.abs(dT_peak_map),90)*1e9:.3f}"
        )
    }
    return metrics, E_pred_c, E_meas_c, t_common

# --------------- Main ---------------
if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="Validate the ray-aligned propagation operator "
                                             "plane to plane.")
    ap.add_argument("comp", nargs="?", default="Ez", choices=["Ex", "Ey", "Ez"],
                    help="field component (default Ez)")
    ap.add_argument("--plane", default=PLANE_FILE, help="recorded reference plane (gprMax .out)")
    ap.add_argument("--target", default=TARGET_FILE,
                    help="directly simulated plane at x2 (gprMax .out)")
    _args = ap.parse_args()
    infile, infile_x2, COMP = _args.plane, _args.target, _args.comp
    COMP_LABEL = {'Ex': 'E_x', 'Ey': 'E_y', 'Ez': 'E_z'}[COMP]
    YLABEL = f"{COMP_LABEL} [V/m] per A·m"

    # Load the decimated reference plane (only the receivers that are used)
    fields_ds, x1, yy_ds, zz_ds, dt, Nt = load_gprmax_plane_decimated(
        infile, comps=(COMP,), y_step=y_step, z_step=z_step, x_tol=x_tol)

    fs = 1.0 / dt
    Npos = Nt // 2 + 1
    df = fs / Nt
    print(f"Loaded plane at x1={x1:.3f} m with {list(fields_ds.keys())}, dt={dt*1e9:.3f} ns, Nt={Nt}")
    print(f"Frequency axis: fs={fs/1e6:.3f} MHz, df={df/1e3:.3f} kHz, Npos={Npos}")
    print(f"Decimated at load: ny={yy_ds.size}, nz={zz_ds.size} (y_step={y_step}, z_step={z_step})")
    if COMP not in fields_ds:
        raise ValueError(f"Reference-plane file does not contain {COMP} component.")

    # Propagate each loaded component
    propagated = {}
    t2 = None
    for comp, E1_t in fields_ds.items():
        print(f"Propagating {comp}: plane x1={x1:.3f} m -> x2={x2:.3f} m")
        E2_t, t2 = propagate_ff_component(
            E1_t=E1_t,
            dt=dt,
            x1=x1,
            x2=x2,
            yy=yy_ds,
            zz=zz_ds,
            src_pos=src_pos,
            mu=mu,
            eps=eps,
            sigma=sigma,
            comp=comp,
            reg=pattern_regularizer
        )
        propagated[comp] = E2_t

    # Zero padding added by the propagation
    Nt2 = next(iter(propagated.values())).shape[2]
    Npre_est = int(round(max(0.0, -t2[0] / dt)))
    Npost_est = Nt2 - Nt - Npre_est
    print(f"Padding used (estimated): Npre={Npre_est}, Npost={Npost_est}, Nt_out={Nt2}")

    # Propagated field with its time axis, one file per component
    out = {
        'x1': x1,
        'x2': x2,
        'yy': yy_ds,
        'zz': zz_ds,
        # input-plane time metadata (provenance)
        'dt_in': dt,
        'Nt_in': Nt,
        # propagated-plane time axis
        't2': t2,
        'Nt2': Nt2,
        # medium and source
        'src_pos': np.array(src_pos),
        'eps_r': eps_r,
        'mu_r': mu_r,
        'sigma': sigma,
        'pattern_regularizer': pattern_regularizer,
        'y_step': y_step,
        'z_step': z_step,
    }
    for k, v in propagated.items():
        out[f"{k}_x2"] = v.astype(np.float32)

    prop_npz = PROPAGATION / f"propagation_{COMP}_field_x2.npz"
    np.savez_compressed(prop_npz, **out)
    print(f"Saved propagated fields and time axis to {prop_npz}")

    # ===================== Comparison with measured x2 plane =====================
    meas_fields, x2_loaded, yy2, zz2, dt_meas, Nt_meas = load_gprmax_plane_decimated(
        infile_x2, comps=(COMP,), y_step=y_step, z_step=z_step, x_tol=x_tol)
    print(f"Loaded measured x2-plane from {infile_x2}: x_loaded={x2_loaded:.4f} (target {x2}), "
          f"dt_meas={dt_meas*1e9:.3f} ns, Nt_meas={Nt_meas}, ny={yy2.size}, nz={zz2.size}")
    if abs(x2_loaded - x2) > x_tol:
        print(f"Warning: measured plane is at x={x2_loaded}, not within {x_tol} m of x2={x2}.")
    if COMP not in meas_fields:
        raise ValueError(f"Measured x2-plane file does not contain {COMP} component.")

    if np.array_equal(yy2, yy_ds) and np.array_equal(zz2, zz_ds):
        # Same receiver grid on both planes: decimated axes coincide exactly
        E2_meas_on_ds = meas_fields[COMP]
        print("Measured and propagated planes share the decimated grid (no regridding).")
    else:
        # Different grids: reload the measured plane in full and regrid by nearest neighbor
        print("Grids differ: reloading the full measured plane for nearest-neighbor regridding.")
        meas_full, _, yy2, zz2, dt_meas, Nt_meas = load_gprmax_plane_decimated(
            infile_x2, comps=(COMP,), y_step=1, z_step=1, x_tol=x_tol)
        E2_meas_on_ds = regrid_nearest_plane(meas_full[COMP], yy2, zz2, yy_ds, zz_ds)
    t_meas = np.arange(Nt_meas) * dt_meas

    # Align time and compute metrics vs propagated field
    E2_pred = propagated[COMP]
    t_pred = t2
    metrics, E_pred_c, E_meas_c, t_common = compute_metrics(E2_pred, E2_meas_on_ds, t_pred, t_meas)

    print(f"\n=== Quantitative comparison ({COMP}) ===")
    print(metrics['summary'])

    # Visualize NRMSE, correlation, and peak-time difference maps
    extent = [yy_ds[0], yy_ds[-1], zz_ds[0], zz_ds[-1]]
    fig1, ax1 = plt.subplots(1, 3, figsize=(9, 3), constrained_layout=True)
    nrmse_vmax = float(np.percentile(metrics['NRMSE_map'], 99))
    im0 = ax1[0].imshow(metrics['NRMSE_map'], extent=extent,
                        origin='lower', aspect='equal', cmap='magma', vmin=0.0, vmax=nrmse_vmax)
    ax1[0].set_title("NRMSE (range-normalized)")
    ax1[0].set_xlabel("y [m]"); ax1[0].set_ylabel("z [m]")
    cb = fig1.colorbar(im0, ax=ax1[0], shrink=0.6, extend='max')
    cb.ax.yaxis.set_major_formatter(
        mticker.FuncFormatter(lambda x, _: f"{x*100:.2f}%")
    )

    corr_vmin = min(0.99, np.floor(np.percentile(metrics['corr_map'], 1) * 100) / 100)
    im1 = ax1[1].imshow(metrics['corr_map'], extent=extent,
                        origin='lower', aspect='equal', cmap='viridis', vmin=corr_vmin, vmax=1.0)
    ax1[1].set_title("Pearson correlation")
    ax1[1].set_xlabel("y [m]"); ax1[1].set_ylabel("z [m]")
    fig1.colorbar(im1, ax=ax1[1], shrink=0.6, extend='min')

    im2 = ax1[2].imshow(metrics['dT_peak_map']*1e9, extent=extent,
                     origin='lower', aspect='equal', cmap='coolwarm')
    ax1[2].set_title("Peak-time difference [ns]")
    ax1[2].set_xlabel("y [m]"); ax1[2].set_ylabel("z [m]")
    fig1.colorbar(im2, ax=ax1[2], shrink=0.6)

    out_png = PROPAGATION / f"propagation_{COMP}_metric_maps.png"
    fig1.savefig(out_png, dpi=200, bbox_inches='tight')
    plt.close(fig1)
    print(f"Saved comparison metrics figure to {out_png}")


    # -----------------------------
    # Plot traces at (y, z) = (5.0, 5.0) and (5.0, 9.5) on the target plane
    # -----------------------------
    for y_pick, z_pick in ((5.0, 5.0), (5.0, 9.5)):
        iy0 = int(np.argmin(np.abs(yy_ds - y_pick)))
        iz0 = int(np.argmin(np.abs(zz_ds - z_pick)))
        plt.figure(figsize=(7, 4))
        plt.plot(t_common * 1e9, E_meas_c[iz0, iy0, :], 'k', lw=1.2, label=f'FDTD {COMP_LABEL}')
        plt.plot(t_common * 1e9, E_pred_c[iz0, iy0, :], 'r--', lw=1.2, label=f'Ray-aligned {COMP_LABEL}')
        plt.grid(True, alpha=0.3)
        plt.xlabel('Time [ns]')
        plt.ylabel(YLABEL)
        plt.title(f'{COMP_LABEL} at y={yy_ds[iy0]:.2f} m, z={zz_ds[iz0]:.2f} m (x={x2:g} m)')
        plt.legend()
        plt.tight_layout()
        tag = f"x{x2:g}_y{y_pick:g}z{z_pick:g}".replace('.', 'p')
        plt.savefig(PROPAGATION / f"propagation_{COMP}_trace_{tag}.png", dpi=300)
        plt.close()

    metrics_npz = PROPAGATION / f"propagation_{COMP}_metrics.npz"
    np.savez_compressed(
        metrics_npz,
        yy=yy_ds, zz=zz_ds,
        window_s=np.array(metrics['window_s']),
        NRMSE_map=metrics['NRMSE_map'].astype(np.float32),
        peak_ratio_map=metrics['peak_ratio_map'].astype(np.float32),
        corr_map=metrics['corr_map'].astype(np.float32),
        dT_peak_map=metrics['dT_peak_map'].astype(np.float32),
        relL2_global=np.array(metrics['relL2']),
        NRMSE_rms_map=metrics['NRMSE_rms_map'].astype(np.float32),
    )
    print(f"Saved quantitative metrics to {metrics_npz}")
