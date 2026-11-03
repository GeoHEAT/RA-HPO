"""One RA-HPO trace.

``run_simulation`` meshes the fracture, loads the reference plane, integrates
``element_contribution`` over the mesh, and synthesises the receiver trace.
"""
from __future__ import annotations

import logging
import os
import time

import matplotlib
matplotlib.use("Agg")
import matplotlib.cm as cm
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import TwoSlopeNorm
from matplotlib.lines import Line2D
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401  (registers the 3D projection)

try:
    from joblib import Parallel, delayed
    HAVE_JOBLIB = True
except ImportError:
    HAVE_JOBLIB = False

from rahpo.constants import EPS0, FC, MU0
from rahpo.fields import (
    analytical_ricker_spectrum,
    build_padded_plane_spectrum,
    build_taper,
    edge_taper_weights,
    gamma_of_omega,
    load_gprmax_plane,
)
from rahpo.fracture import get_fracture_mesh, get_specular_point_numerical
from rahpo.parameters import GeometryParams, MediaParams, RuntimeParams, SourceParams, WorkerParams
from rahpo.scattering import element_contribution

def visualize_setup(geom, srcpar, run, elempos, x1, yy, zz):
    """Visualizes the GPR Tx/Rx, the Input Plane (x1), and Scattering Elements."""
    fig = plt.figure(figsize=(12, 10))
    ax = fig.add_subplot(111, projection='3d')

    # Plot Source and Receiver
    xs, ys, zs = srcpar.src_pos
    ax.scatter(xs, ys, zs, c='r', marker='*', s=200, label='Tx(Source)', zorder=5)

    xr, yr, zr = srcpar.rec_pos
    ax.scatter(xr, yr, zr, c='b', marker='*', s=200, label='Rx(Receiver)', zorder=5)
    
    # Plot the Input Plane (x1)
    Y_plane, Z_plane = np.meshgrid(yy[::10], zz[::10]) 
    X_plane = np.full_like(Y_plane, x1)
    ax.plot_surface(X_plane, Y_plane, Z_plane, alpha=0.2, color='blue', linewidth=0)

    # Plot the Scattering Elements as a Colored Surface
    # Reconstruct the 2D grid dimensions based on the geometry parameters
    ny = len(np.arange(geom.ymin, geom.ymax, geom.ellength))
    nz = len(np.arange(geom.zmin, geom.zmax, geom.ellength))
    
    # Reshape the flattened 1D arrays back into 2D matrices
    X_surf = elempos[0, :].reshape(ny, nz)
    Y_surf = elempos[1, :].reshape(ny, nz)
    Z_surf = elempos[2, :].reshape(ny, nz)
    epsr_surf = elempos[8, :].reshape(ny, nz)

    # Setup Diverging Colormap centered at the mean
    mean_epsr = np.mean(epsr_surf)
    vmin = np.min(epsr_surf)
    vmax = np.max(epsr_surf)
    
    if np.isclose(vmin, vmax):
        color_norm = plt.Normalize(vmin - 1, vmax + 1)
    else:
        color_norm = TwoSlopeNorm(vmin=vmin, vcenter=mean_epsr, vmax=vmax)

    cmap = plt.get_cmap('RdYlBu')
    
    surf = ax.plot_surface(X_surf, Y_surf, Z_surf, 
                           facecolors=cmap(color_norm(epsr_surf)), 
                           shade=False, alpha=0.9, 
                           linewidth=0, antialiased=False)
                           
    m = cm.ScalarMappable(cmap=cmap, norm=color_norm)
    m.set_array([])
    cbar = fig.colorbar(m, ax=ax, shrink=0.5, aspect=12, pad=0.1)
    cbar.set_label(r"Relative Permittivity ($\epsilon_r$)", fontsize=12)

    # Plot a few Ray Paths (Verification)
    indices = np.linspace(0, elempos.shape[1]-1, 3, dtype=int)
    for i in indices:
        xe, ye, ze = elempos[0, i], elempos[1, i], elempos[2, i]
        denom = (xe - xs)
        if abs(denom) > 1e-9:
            lam = (x1 - xs) / denom
            yi = ys + lam * (ye - ys)
            zi = zs + lam * (ze - zs)
            ax.plot([xs, x1], [ys, yi], [zs, zi], 'g--', lw=1)
            ax.plot([x1, xe], [yi, ye], [zi, ze], 'g-', lw=1)

    ax.set_xlabel('X [m]')
    ax.set_ylabel('Y [m]')
    ax.set_zlabel('Z [m]')
    ax.set_title(f'RA-HPO Setup: Input x={x1:.2f}, Source at {srcpar.src_pos}')
    
    custom_lines = [
        Line2D([0], [0], color='blue', lw=4, alpha=0.2),
        Line2D([0], [0], color='g', linestyle='--')
    ]
    ax.legend(custom_lines, [f'Input Plane x={x1:.2f}', 'Ray Paths'], loc='upper left')

    plt.tight_layout()
    plt.savefig('images/RA_HPO_Geometry_Surface.png', dpi=300)


def run_simulation(media: MediaParams, srcpar: SourceParams, geom: GeometryParams, run: RuntimeParams) -> tuple[np.ndarray, np.ndarray]:
    """Compute one scattered trace.

    Returns
    -------
    tuple[np.ndarray, np.ndarray]
        Time axis in seconds and the scattered Ez at the receiver, V/m.
    """
    start_t = time.time()
    _p0 = time.perf_counter()   # per-phase timing

    xs, ys, zs = srcpar.src_pos
    xr, yr, zr = srcpar.rec_pos

    logging.info("Generating fracture mesh...")
    elempos = get_fracture_mesh(geom, media)
    numelem = elempos.shape[1]
    logging.info(f"Generated {numelem} elements, element area '{geom.element_area}' "
                 f"(total {np.sum(elempos[6, :]):.6f} m^2).")

    logging.info("Finding specular point...")
    P_spec, idx_spec = get_specular_point_numerical(srcpar.src_pos, srcpar.rec_pos, elempos)
    logging.info(f"Specular Point found at: {P_spec}")
    _p_mesh = time.perf_counter()   # end phase: mesh + specular

    logging.info("Loading Field Planes...")
    fields_t, x1, yy, zz, dt, Nt = load_gprmax_plane(run.tx_plane_file, run.comps,
                                                     target_res=run.target_res)

    # Identical TX/RX files -> reciprocal field via axial shift of the forward field
    # Different files -> load the reciprocal plane from its own file.
    use_shift = (os.path.abspath(run.rx_plane_file) == os.path.abspath(run.tx_plane_file))
    if use_shift:
        logging.info("TX and RX plane files identical -> reciprocal field via axial shift of forward field.")
        reciprocal_fields_t, r_x1, r_yy, r_zz, r_dt, r_Nt = fields_t, x1, yy, zz, dt, Nt
    else:
        reciprocal_fields_t, r_x1, r_yy, r_zz, r_dt, r_Nt = load_gprmax_plane(
            run.rx_plane_file, run.comps, target_res=run.target_res)

    if not np.isclose(dt, r_dt):
        raise ValueError("Time steps of forward and reciprocal planes must match.")

    logging.info(f"Loaded TX plane at x1={x1:.3f} m: comps={list(fields_t.keys())}, dt={dt*1e9:.3f} ns, Nt={Nt}, ny={yy.size}, nz={zz.size}")
    logging.info(f"Loaded RX plane at r_x1={r_x1:.3f} m: comps={list(reciprocal_fields_t.keys())}, dt={r_dt*1e9:.3f} ns, Nt={r_Nt}, ny={r_yy.size}, nz={r_zz.size}")
    _p_load = time.perf_counter()   # end phase: plane load / I/O

    if not run.no_viz:
        visualize_setup(geom, srcpar, run, elempos, x1, yy, zz)

    eps = EPS0 * media.epsr2
    mu  = MU0
    sigma = media.sigma_rock
    v = 1.0 / np.sqrt(mu * eps)
    c_med = v

    eps1 = EPS0 * media.epsr1
    sigma1 = media.sigma_rock1
    v1 = 1.0 / np.sqrt(mu * eps1) 

    # TX Delays
    denom_tx = elempos[0,:] - xs
    valid_tx = np.abs(denom_tx) > 1e-9
    el_v_tx = elempos[:, valid_tx]
    lam_tx = (x1 - xs) / (el_v_tx[0,:] - xs)
    y1m_tx = ys + lam_tx * (el_v_tx[1,:] - ys)
    z1m_tx = zs + lam_tx * (el_v_tx[2,:] - zs)
    R2_tx = np.sqrt((el_v_tx[0,:] - xs)**2 + (el_v_tx[1,:] - ys)**2 + (el_v_tx[2,:] - zs)**2)
    R1_tx = np.sqrt((x1 - xs)**2 + (y1m_tx - ys)**2 + (z1m_tx - zs)**2)
    # effective dR so that dR/c_med equals the true delay R2/v2 - R1/v1
    dR_vals_tx = c_med * (R2_tx / v - R1_tx / v1)

    # RX Delays
    denom_rx = elempos[0,:] - xr
    valid_rx = np.abs(denom_rx) > 1e-9
    el_v_rx = elempos[:, valid_rx]
    lam_rx = (r_x1 - xr) / (el_v_rx[0,:] - xr)
    y1m_rx = yr + lam_rx * (el_v_rx[1,:] - yr)
    z1m_rx = zr + lam_rx * (el_v_rx[2,:] - zr)
    R2_rx = np.sqrt((el_v_rx[0,:] - xr)**2 + (el_v_rx[1,:] - yr)**2 + (el_v_rx[2,:] - zr)**2)
    R1_rx = np.sqrt((r_x1 - xr)**2 + (y1m_rx - yr)**2 + (z1m_rx - zr)**2)
    # effective dR so that dR/c_med equals the true delay R2/v2 - R1/v1
    dR_vals_rx = c_med * (R2_rx / v - R1_rx / v1)
    
    dR_min = float(min(np.min(dR_vals_tx), np.min(dR_vals_rx)))
    dR_max = float(max(np.max(dR_vals_tx), np.max(dR_vals_rx)))

    
    # The cut-off is applied inside, so the spectra come back already reduced.
    cutoff_high = 8.0 * FC
    Epos_small_tx, omega_pos, valid_mask_freqs, Npre, Nt_out, Nfft = \
        build_padded_plane_spectrum(fields_t, dt, dR_min, dR_max, c_med,
                                    cutoff_hz=cutoff_high)
    if use_shift:
        Epos_small_rx = Epos_small_tx
    else:
        Epos_small_rx, _, _, _, _, _ = build_padded_plane_spectrum(
            reciprocal_fields_t, r_dt, dR_min, dR_max, c_med, cutoff_hz=cutoff_high)


    gamma_pos = gamma_of_omega(omega_pos, mu, eps, sigma)

    gamma_pos1 = gamma_of_omega(omega_pos, mu, eps1, sigma1)
    eta1_pos = sigma1 + 1j * omega_pos * eps1   # y1 = sigma1 + j w eps1
    eta2_pos = sigma  + 1j * omega_pos * eps    # y2 = sigma2 + j w eps2
    eamp_pos = np.divide(eta1_pos, eta2_pos,
                         out=np.zeros_like(eta1_pos),
                         where=(eta2_pos != 0))  # y1/y2 (safe at w=0)

    
    # build_padded_plane_spectrum already applied the mask, so only the scalar frequency axes
    # still need reducing. valid_mask_freqs is kept because the
    # deconvolution below scatters the result back into the full-length spectrum.
    omega_pos_small = omega_pos[valid_mask_freqs]
    gamma_pos_small = gamma_pos[valid_mask_freqs]
    gamma_pos1_small = gamma_pos1[valid_mask_freqs]
    eamp_small = eamp_pos[valid_mask_freqs].astype(np.complex64)



    logging.info(f"Original Freqs: {len(omega_pos)} points (up to {omega_pos[-1]/2/np.pi/1e9:.1f} GHz), Nfft={Nfft}")
    logging.info(f"Reduced Freqs:  {len(omega_pos_small)} points (up to {cutoff_high/1e6:.1f} MHz)")

    tapx, tapy = build_taper(run.critdist, geom.ellength, run.tapw)

    params = WorkerParams(
        Epos_tx=Epos_small_tx, Epos_rx=Epos_small_rx,
        
        yy=yy, zz=zz, x1=x1, r_yy=r_yy, r_zz=r_zz, r_x1=r_x1,
        omega_pos=omega_pos_small, gamma_pos=gamma_pos_small, 
        gamma_pos1=gamma_pos1_small, eamp=eamp_small,
        dt=dt,
        Npre=Npre, Nt_out=Nt_out, Nfft=Nfft, v=v,
        epsr2=media.epsr2, eps0_local=EPS0,
        srcx=xs, srcy=ys, srcz=zs,
        recx=xr, recy=yr, recz=zr,
        spec_x=P_spec[0], spec_y=P_spec[1], spec_z=P_spec[2],
        ellength=geom.ellength, tapx=tapx, tapy=tapy, critdist=run.critdist,
        doradpat=0, pws_interpor=None, use_shift=use_shift,
        incidence=run.incidence,
    )
    _p_fft = time.perf_counter()   # end phase: FFT / spectrum build (+ ray-delay prep)

    logging.info("Starting parallel reaction integral computation...")
    if HAVE_JOBLIB and run.cores != 1:
        results = Parallel(n_jobs=run.cores, verbose=0)(
            delayed(element_contribution)(iel, elempos, params, media) for iel in range(numelem)
        )
    else:
        results = [element_contribution(iel, elempos, params, media) for iel in range(numelem)]

    if run.incidence == "perfreq":
        # Elements that return early give a bare zero spectrum, with no spread to report.
        spread = np.array([r[1] if isinstance(r, tuple) else np.nan for r in results])
        results = [r[0] if isinstance(r, tuple) else r for r in results]

    results_arr = np.array(results)

    if run.incidence == "perfreq":
        # Weight each element's spread by its contribution, so that elements carrying little
        # of the trace do not dominate the summary.
        w = np.sum(np.abs(results_arr), axis=1)
        good = np.isfinite(spread) & (w > 0)
        deg = np.degrees(spread[good])
        wmean = float(np.sum(w[good] * deg) / np.sum(w[good])) if np.any(good) else float("nan")
        logging.info(
            f"[INCIDENCE] per-frequency directions, power-weighted rms spread around the band "
            f"average over {int(np.count_nonzero(good))} elements: median {np.median(deg):.3f} deg, "
            f"95th percentile {np.percentile(deg, 95):.3f} deg, max {np.max(deg):.3f} deg, "
            f"contribution-weighted mean {wmean:.3f} deg")
        run.incidence_spread = (elempos[0:3, :].copy(), spread, w)

    # Dropped-element diagnostic. An element contributes exactly zero only
    # where the kernel returns early: degenerate source- or receiver-to-element distances, a
    # vanishing plane-crossing denominator, or a ray whose crossing point falls outside the
    # recorded plane, which `same_ray_propagate_point` reports by returning a zero field. All of
    # those are silent, and the last one is a real modeling limit rather than a numerical corner:
    # it bounds how far a receiver can sit from the plane before part of the fracture stops
    # contributing. Reported as a count so a run cannot quietly integrate over fewer elements
    # than the mesh contains.
    n_dropped = int(np.count_nonzero(~results_arr.any(axis=1)))
    logging.info(
        f"[DROPPED] {n_dropped} of {numelem} elements ({100.0 * n_dropped / numelem:.3f}%) "
        f"contributed exactly zero (degenerate ray geometry, or ray outside the recorded plane)"
    )

    # Optional edge taper, skipped entirely at the default edge_taper = 0.0
    if run.edge_taper > 0.0:
        box = (geom.ymin, geom.ymax, geom.zmin, geom.zmax)
        w_tap = edge_taper_weights(elempos[1, :], elempos[2, :], run.edge_taper, box)
        results_arr *= w_tap[:, None]
        logging.info(
            f"[TAPER] cos^2 edge taper, width {run.edge_taper:g} m, mesh y [{box[0]:g}, {box[1]:g}] "
            f"z [{box[2]:g}, {box[3]:g}] m: {int(np.count_nonzero(w_tap < 1.0))} of {numelem} "
            f"elements weighted below 1, mean weight {float(np.mean(w_tap)):.4f}")

    Erec_raw_small = np.sum(results_arr, axis=0)
    del results_arr
    logging.debug(f"Max raw spectrum amplitude: {np.max(np.abs(Erec_raw_small))}")
    _p_integ = time.perf_counter()   # end phase: parallel element integration

    scaling_factor = dt
    Erec_raw_small *= scaling_factor

    S_mag = analytical_ricker_spectrum(omega_pos_small, FC, amp=1.0)
    t0_source = np.sqrt(2) / FC
    I0_spectrum = S_mag.astype(complex) * np.exp(-1j * omega_pos_small * t0_source)
    S_peak = np.max(np.abs(I0_spectrum))
    S_normalized = I0_spectrum / S_peak
    Erec_nrml_small = Erec_raw_small / S_peak

    max_pwr = np.max(np.abs(S_normalized)**2)
    water_level = 1e-6 * max_pwr  
    deconvolved_pos = Erec_nrml_small * np.conj(S_normalized) / (np.abs(S_normalized)**2 + water_level)

    freqs_small = omega_pos_small / (2*np.pi)
    f_sigma = 8.0 * FC
    gaussian_window = np.exp(-0.5 * (freqs_small / f_sigma)**2)
    deconvolved_pos *= gaussian_window
    
    deconvolved_full = np.zeros(Nfft // 2 + 1, dtype=complex)
    deconvolved_full[valid_mask_freqs] = deconvolved_pos
    E_time_full = np.fft.irfft(deconvolved_full, n=Nfft)
    trace = E_time_full[:Nt_out].real
    tvec = (np.arange(Nt_out) - Npre) * dt
    _p_synth = time.perf_counter()   # end phase: deconvolution + iFFT + synthesis

    logging.info(f"Shape of Erec_raw_small: {Erec_raw_small.shape}")
    logging.info(f"Size of trace={trace.size}, tvec[0]={tvec[0]*1e9:.3f} ns, tvec[-1]={tvec[-1]*1e9:.3f} ns")
    # Per-phase breakdown: (i) plane load/IO, (ii) FFT/spectrum build,
    # (iii) parallel element integration, (iv) deconvolution + iFFT + synthesis. Only (iii)
    # varies with the number of workers, so the constant phases dominate as workers are added.
    logging.info(
        f"[PHASE] cores={run.cores} mesh={_p_mesh - _p0:.3f} load={_p_load - _p_mesh:.3f} "
        f"fft={_p_fft - _p_load:.3f} integ={_p_integ - _p_fft:.3f} synth={_p_synth - _p_integ:.3f} "
        f"total={_p_synth - _p0:.3f}s"
    )
    logging.info(f"Simulation completed in {time.time() - start_t:.2f}s")

    return tvec, trace


def save_trace(filename: str, tvec: np.ndarray, trace: np.ndarray) -> None:
    """Write ``time_ns`` and ``amplitude`` to a NumPy ``.npz`` file."""
    np.savez(filename, time_ns=tvec*1e9, amplitude=trace)
    logging.info(f"Trace saved to {filename}")

