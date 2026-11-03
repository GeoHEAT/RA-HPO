"""Physical-optics scattering of one fracture element.

``element_contribution`` propagates the reference-plane field to the element,
applies the three-layer TE/TM reflection, and returns that element's part of
the reaction integral. ``per_frequency_directions`` and
``reflect_per_frequency`` are the per-frequency incidence diagnostic.
"""
from __future__ import annotations

import numpy as np

from rahpo.constants import MU0
from rahpo.fields import same_ray_propagate_H_dipole, same_ray_propagate_point
from rahpo.parameters import MediaParams, WorkerParams

def element_contribution(iel: int, elempos: np.ndarray, p: WorkerParams, media: MediaParams) -> np.ndarray | tuple:
    """Reaction-integral contribution of fracture element ``iel``.

    Returns the spectrum at the kept frequencies. In the ``perfreq`` diagnostic
    the return is ``(spectrum, spread_rad)``. An element whose ray misses the
    recorded plane contributes zero.
    """
    pos_el = elempos[0:3, iel]
    pos_tx = np.array([p.srcx, p.srcy, p.srcz])
    pos_rx = np.array([p.recx, p.recy, p.recz])
    
    vec_tx_el = pos_el - pos_tx
    dist_tx_el = np.linalg.norm(vec_tx_el)
    if dist_tx_el < 1e-12: return np.zeros_like(p.omega_pos, dtype=complex)
    k_inc_hat = vec_tx_el / dist_tx_el
    
    vec_rx_el = pos_el - pos_rx
    dist_rx_el = np.linalg.norm(vec_rx_el)
    if dist_rx_el < 1e-12: return np.zeros_like(p.omega_pos, dtype=complex)
    
    # Retrieve Normal Vector
    n_hat = elempos[3:6, iel] # [nx, ny, nz]
    
    # Forward Fields (Ray-Aligned)
    E_inc = same_ray_propagate_point(
        p.Epos_tx, p.yy, p.zz, p.x1, p.srcx, p.srcy, p.srcz,
        pos_el[0], pos_el[1], pos_el[2], p.gamma_pos,
        ('Ex' in p.Epos_tx, 'Ey' in p.Epos_tx, 'Ez' in p.Epos_tx),
        gamma_pos1=p.gamma_pos1, eamp=p.eamp   # TWO-MEDIUM
    )
    H_inc = same_ray_propagate_H_dipole(
        p.Epos_tx, p.yy, p.zz, p.x1, p.srcx, p.srcy, p.srcz, 
        pos_el[0], pos_el[1], pos_el[2], p.gamma_pos,
        ('Hx' in p.Epos_tx, 'Hy' in p.Epos_tx, 'Hz' in p.Epos_tx),
        gamma_pos1=p.gamma_pos1   # TWO-MEDIUM (no eamp for H)
    )

    # Reciprocal Fields
    if p.use_shift:
        # SHIFT MODE: TX and RX plane files are identical.
        # Translation invariance: the field of a source at rec observed at pos_el equals
        # the forward field (source at src) observed at pos_el - (rec - src). We therefore
        # reuse the FORWARD plane/spectrum (p.Epos_tx, p.yy, p.zz, p.x1) with the SOURCE
        # coordinates and evaluate at the virtual element pos_el_rx. No second FFT needed.
        offset_vec = np.array([p.recx - p.srcx, p.recy - p.srcy, p.recz - p.srcz])
        pos_el_rx = pos_el - offset_vec
        rx_Epos, rx_yy, rx_zz, rx_x1 = p.Epos_tx, p.yy, p.zz, p.x1
        rx_sx, rx_sy, rx_sz = p.srcx, p.srcy, p.srcz
    else:
        # LOAD MODE (new-version behaviour): use the separately loaded RX plane.
        pos_el_rx = pos_el
        rx_Epos, rx_yy, rx_zz, rx_x1 = p.Epos_rx, p.r_yy, p.r_zz, p.r_x1
        rx_sx, rx_sy, rx_sz = p.recx, p.recy, p.recz

    denom_rx = (pos_el_rx[0] - rx_sx)
    if np.isclose(denom_rx, 0.0): return np.zeros_like(p.omega_pos, dtype=complex)

    E_rx = same_ray_propagate_point(
        rx_Epos, rx_yy, rx_zz, rx_x1, rx_sx, rx_sy, rx_sz,
        pos_el_rx[0], pos_el_rx[1], pos_el_rx[2], p.gamma_pos, 
        ('Ex' in rx_Epos, 'Ey' in rx_Epos, 'Ez' in rx_Epos),
        gamma_pos1=p.gamma_pos1, eamp=p.eamp  
    )
    H_rx = same_ray_propagate_H_dipole(
        rx_Epos, rx_yy, rx_zz, rx_x1, rx_sx, rx_sy, rx_sz,
        pos_el_rx[0], pos_el_rx[1], pos_el_rx[2], p.gamma_pos, 
        ('Hx' in rx_Epos, 'Hy' in rx_Epos, 'Hz' in rx_Epos),
        gamma_pos1=p.gamma_pos1   
    )

    # Reflection (Airy/Sommerfeld Slab)
    S_vec = np.cross(E_inc.T, np.conj(H_inc.T)).T 
    avg_S = np.mean(np.real(S_vec), axis=1)
    norm_S = np.linalg.norm(avg_S)
    k_dir = avg_S / norm_S if norm_S > 1e-15 else k_inc_hat

    cos_theta = -np.dot(k_dir, n_hat)
    if cos_theta < 0:
        n_hat = -n_hat
        cos_theta = -cos_theta
    cos_theta = np.clip(cos_theta, 0.0, 1.0)
    if p.incidence == "perfreq":
        # Diagnostic: one direction per frequency. n_hat keeps the orientation chosen from the
        # band average above, and cos_theta becomes an array, which the coefficient formulas
        # below accept unchanged because they are elementwise in frequency.
        k_f, spread = per_frequency_directions(np.real(S_vec), k_dir)
        cos_theta = np.clip(-(n_hat @ k_f), 0.0, 1.0)
    sin_theta = np.sqrt(1.0 - cos_theta**2)

    eta2_scalar = np.sqrt(MU0 / (p.eps0_local * p.epsr2))
    omega = p.omega_pos
    safe_omega = np.where(omega == 0, 1e-12, omega)
    smu = 1j * safe_omega * MU0
    eta_rock = p.eps0_local * p.epsr2 * (1j * safe_omega) + media.sigma_rock
    
    eps_real_frac = elempos[8, iel] 
    sigma_frac = elempos[10, iel]
    eta_frac = p.eps0_local * eps_real_frac * (1j * safe_omega) + sigma_frac
    
    k_rock_sq = -smu * eta_rock
    k_rock = np.sqrt(k_rock_sq)
    kappa = k_rock * sin_theta
    
    gamma_1 = np.sqrt(smu * eta_rock + kappa**2)
    gamma_2 = np.sqrt(smu * eta_frac + kappa**2)
    gamma_3 = gamma_1 
    
    r12_te = (gamma_1 - gamma_2) / (gamma_1 + gamma_2)
    r23_te = (gamma_2 - gamma_3) / (gamma_2 + gamma_3)
    r12_tm = (eta_frac * gamma_1 - eta_rock * gamma_2) / (eta_frac * gamma_1 + eta_rock * gamma_2)
    r23_tm = (eta_rock * gamma_2 - eta_frac * gamma_3) / (eta_rock * gamma_2 + eta_frac * gamma_3)
    
    thickness = elempos[7, iel]
    exp_term = np.exp(-2 * gamma_2 * thickness)
    R_TE = (r12_te + r23_te * exp_term) / (1.0 + r12_te * r23_te * exp_term)
    R_TM = (r12_tm + r23_tm * exp_term) / (1.0 + r12_tm * r23_tm * exp_term)

    # Reconstruct Fields
    if p.incidence == "perfreq":
        E_ref, H_ref = reflect_per_frequency(E_inc, k_f, n_hat, R_TE, R_TM, eta2_scalar)
    else:
        te_vec_raw = np.cross(k_dir, n_hat)
        te_len = np.linalg.norm(te_vec_raw)

        # Specular direction, needed by H_ref in both branches: at normal incidence (te_len -> 0)
        # the TE/TM split degenerates and is skipped, but the reflected ray is k_ref = -k_dir.
        k_ref = k_dir - 2 * np.dot(k_dir, n_hat) * n_hat

        if te_len < 1e-9:
            # Normal incidence: TE and TM coincide, so one coefficient applies to the whole field.
            E_ref = E_inc * R_TE
        else:
            e_te = te_vec_raw / te_len
            e_tm_inc = np.cross(e_te, k_dir)
            E_inc_TE = np.dot(e_te, E_inc)
            E_inc_TM = np.dot(e_tm_inc, E_inc)
            e_tm_ref = np.cross(e_te, k_ref)
            E_ref = (e_te[:, None] * (E_inc_TE * R_TE)[None, :]) + \
                    (e_tm_ref[:, None] * (E_inc_TM * R_TM)[None, :])

        H_ref = np.cross(k_ref, E_ref.T).T / eta2_scalar
    Js = np.cross(n_hat, H_ref.T).T
    Ms = np.cross(E_ref.T, n_hat).T

    term1 = np.sum(E_rx * Js, axis=0) 
    term2 = np.sum(H_rx * Ms, axis=0) 
    reaction = 1.0 * (term1 - term2) * elempos[6, iel] 

    # No taper is applied here. The optional edge taper (--edge_taper) multiplies each
    # element's contribution by its weight in run_simulation, after this loop. The radial taper
    # of build_taper() (critdist, tapw) is off by default.

    if p.incidence == "perfreq":
        return reaction, spread
    return reaction


def per_frequency_directions(S_re: np.ndarray, k_band: np.ndarray):
    """Per-frequency incidence directions for the "perfreq" diagnostic.

    S_re is the real Poynting vector Re(E x H*) at the element, shape (3, Nf), and k_band the
    unit band-averaged direction that the method applies at every frequency. Each
    frequency's direction is S_re normalized. Where |S_re| is below 1e-6 of its largest value
    over the band, the direction is noise and carries no energy, so the band direction is kept.

    Returns the directions, shape (3, Nf), and the angular spread around k_band in radians: the
    root-mean-square angle weighted by |S_re|, the power at each frequency.
    """
    mag = np.linalg.norm(S_re, axis=0)
    k_f = np.repeat(k_band[:, None], S_re.shape[1], axis=1)
    peak = mag.max() if mag.size else 0.0
    if peak <= 0.0:
        return k_f, 0.0
    ok = mag > 1e-6 * peak
    k_f[:, ok] = S_re[:, ok] / mag[ok]
    ang = np.arccos(np.clip(k_band @ k_f, -1.0, 1.0))
    spread = float(np.sqrt(np.sum(mag * ang**2) / np.sum(mag)))
    return k_f, spread


def reflect_per_frequency(E_inc, k_f, n_hat, R_TE, R_TM, eta2_scalar):
    """Reflected E and H with a separate incidence direction at each frequency.

    The same construction as the band path of element_contribution, with k_dir, the TE/TM
    unit vectors and the specular direction k_ref carried as (3, Nf) arrays. Where a
    frequency's ray is normal to the element, TE and TM coincide and R_TE applies to the
    whole field, as in the band path.
    """
    k_ref = k_f - 2.0 * (n_hat @ k_f)[None, :] * n_hat[:, None]
    te_raw = np.cross(k_f.T, n_hat).T
    te_len = np.linalg.norm(te_raw, axis=0)
    normal = te_len < 1e-9
    e_te = te_raw / np.where(normal, 1.0, te_len)[None, :]
    e_tm_inc = np.cross(e_te.T, k_f.T).T
    e_tm_ref = np.cross(e_te.T, k_ref.T).T
    E_inc_TE = np.sum(e_te * E_inc, axis=0)
    E_inc_TM = np.sum(e_tm_inc * E_inc, axis=0)
    E_ref = e_te * (E_inc_TE * R_TE)[None, :] + e_tm_ref * (E_inc_TM * R_TM)[None, :]
    if np.any(normal):
        E_ref[:, normal] = E_inc[:, normal] * R_TE[normal]
    H_ref = np.cross(k_ref.T, E_ref.T).T / eta2_scalar
    return E_ref, H_ref

