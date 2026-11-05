"""Reference planes and ray-aligned propagation.

``load_gprmax_plane`` reads a gprMax ``.out`` file or a plane converted to the
contiguous layout. ``same_ray_propagate_point`` and
``same_ray_propagate_H_dipole`` carry one field component from that plane to a
point along the straight ray.
"""
from __future__ import annotations

import logging
from typing import Dict, Tuple

import h5py
import numpy as np

def next_pow2(n: int) -> int:
    """Smallest power of two greater than or equal to ``n``."""
    return 1 << (int(n - 1).bit_length())


def interp_weights_sorted(axis: np.ndarray, coords: float) -> tuple[int, float]:
    """Linear-interpolation index and weight of ``coords`` on a sorted axis.

    The index is clipped to the last interval of ``axis``, and the weight to
    ``[0, 1]``, so a coordinate outside the plane does not extrapolate.
    """
    idx0 = np.searchsorted(axis, coords) - 1
    idx0 = int(np.clip(idx0, 0, len(axis) - 2))
    a0, a1 = axis[idx0], axis[idx0 + 1]
    w = (coords - a0) / (a1 - a0 + np.finfo(float).eps)
    return idx0, float(np.clip(w, 0.0, 1.0))


def gamma_of_omega(omega: np.ndarray, mu: float, eps: float, sigma: float) -> np.ndarray:
    """Propagation constant branched so that its real part is non-negative.

    Parameters
    ----------
    omega : np.ndarray
        Angular frequency, rad/s.
    mu, eps, sigma : float
        Absolute permeability, permittivity, and conductivity.
    """
    eta = sigma + 1j * omega * eps
    gam = np.sqrt(1j * omega * mu * eta)
    return np.where(np.real(gam) < 0, -gam, gam)


def build_taper(critdist: float, ellength: float, tapw: float) -> tuple[np.ndarray, np.ndarray]:
    """Radial taper samples used when ``RuntimeParams.critdist`` is non-zero.

    Returns the radial coordinate ``tapx`` and the cosine-squared weight
    ``tapy``. The default run leaves this taper off.
    """
    n_points = max(1, int(round(critdist / (0.6 * ellength))))
    tapx = np.linspace(0.0, critdist, n_points)
    tapy = np.ones_like(tapx)
    mask = tapx >= (critdist - tapw)
    if np.any(mask):
        slopstartel = np.searchsorted(mask, True)
        if slopstartel < len(tapx):
            slop_len = len(tapx) - slopstartel
            slopx = np.linspace(0, np.pi, slop_len)
            tapy[slopstartel:] = ((np.cos(slopx) + 1.0) / 2.0)**2
    return tapx, tapy


def edge_taper_weights(y: np.ndarray, z: np.ndarray, width: float, box: tuple[float, float, float, float]) -> np.ndarray:
    """Cosine-squared edge taper weight per element, for approximating an unbounded fracture.

    y, z are the element coordinates (elempos rows 1 and 2), width is the taper width in
    metres and box = (ymin, ymax, zmin, zmax) is the meshed fracture extent. With d the
    distance from an element to the nearest mesh edge, the weight is sin^2(pi d / (2 width))
    for d < width and 1 otherwise: one over the interior, falling smoothly to zero at the
    mesh boundary.
    """
    ymin, ymax, zmin, zmax = box
    d = np.minimum(np.minimum(np.abs(y - ymin), np.abs(y - ymax)),
                   np.minimum(np.abs(z - zmin), np.abs(z - zmax)))
    w = np.ones_like(d)
    inside = d < width
    ratio = d[inside] / width          # same order of operations as ra_hpo_taper.py
    w[inside] = np.sin(0.5 * np.pi * ratio) ** 2
    return w


def analytical_ricker_spectrum(omega: np.ndarray, fc: float, amp: float = 1.0) -> np.ndarray:
    """Analytic spectrum of a Ricker wavelet of center frequency ``fc`` (Hz)."""
    zeta = (np.pi ** 2) * (fc ** 2)
    pre = (amp / (2.0 * zeta)) * np.sqrt(np.pi / zeta)
    return pre * (omega**2) * np.exp(-(omega**2) / (4.0 * zeta))


def load_gprmax_plane(filename, comps=("Ex", "Ey", "Ez", "Hx", "Hy", "Hz"),
                      target_res=None):
    """Load field components from a gprMax HDF5 output file.

    target_res is the receiver spacing to read at, in metres. None means use the plane
    as recorded, which is the default: decimation belongs in the near-field model's
    #rx_array spacing, not here. A value decimates a finely recorded plane to it.
    """
    if isinstance(comps, str): comps = (comps,)
    
    fields = {}
    with h5py.File(filename, 'r') as f:
        # Converted plane (see convert_plane.py): one contiguous dataset per component, so the
        # whole plane arrives in six sequential reads. A gprMax .out has no 'format_version'
        # attribute and is read receiver by receiver below.
        if 'format_version' in f.attrs:
            stored_res = float(f.attrs['target_res'])
            if target_res is not None and not np.isclose(stored_res, target_res):
                raise ValueError(
                    f"{filename} was converted at target_res={stored_res} m but "
                    f"target_res={target_res} m was requested; convert again or pass "
                    f"the matching value")
            yy = f['yy'][:]
            zz = f['zz'][:]
            fields = {c: f[c][:] for c in comps if c in f}
            logging.info(f"Converted plane: {zz.size}x{yy.size}, step={int(f.attrs['step'])}, "
                         f"source={f.attrs['source_file']}")
            return (fields, float(f.attrs['x1']), yy, zz,
                    float(f.attrs['dt']), int(f.attrs['Nt']))

        # Read Global Attributes
        nrx = int(f.attrs['nrx'])

        dt = float(f.attrs['dt'])
        
        # Infer Geometry and Sorting
        p1 = f['/rxs/rx1'].attrs['Position']
        p2 = f['/rxs/rx2'].attrs['Position']
        pN = f[f'/rxs/rx{nrx}'].attrs['Position']
        
        x1 = float(p1[0])
        
        dx = p2[0] - p1[0]
        dy_native = p2[1] - p1[1]
        dz_native = p2[2] - p1[2]
        
        is_y_fast = False
        if abs(dy_native) > 1e-9:
            is_y_fast = True
        elif abs(dz_native) > 1e-9:
            is_y_fast = False # Z-Fast
        else:
            if nrx > 1: raise ValueError("Invalid grid geometry.")
            is_y_fast = True # Single point

        # Calculate Dimensions and Decimation
        y_min, y_max = min(p1[1], pN[1]), max(p1[1], pN[1])
        z_min, z_max = min(p1[2], pN[2]), max(p1[2], pN[2])
        
        if is_y_fast:
            step_native = dy_native
            fast_len = int(np.round((y_max - y_min) / abs(step_native))) + 1
            slow_len = nrx // fast_len
            ny_native, nz_native = fast_len, slow_len
            step = (1 if target_res is None
                    else max(1, int(np.round(target_res / abs(step_native)))))
            
            native_yy = np.linspace(p1[1], pN[1] if p1[1]<pN[1] else p1[1]-abs(step_native)*(ny_native-1), ny_native)
            if nz_native > 1:
                native_zz = np.linspace(z_min, z_max, nz_native)
            else:
                native_zz = np.array([p1[2]])
        else: # Z-Fast
            step_native = dz_native
            fast_len = int(np.round((z_max - z_min) / abs(step_native))) + 1
            slow_len = nrx // fast_len
            nz_native, ny_native = fast_len, slow_len
            step = (1 if target_res is None
                    else max(1, int(np.round(target_res / abs(step_native)))))
            
            native_zz = np.linspace(p1[2], pN[2] if p1[2]<pN[2] else p1[2]-abs(step_native)*(nz_native-1), nz_native)
            if ny_native > 1:
                native_yy = np.linspace(y_min, y_max, ny_native)
            else:
                native_yy = np.array([p1[1]])

        yy = native_yy[::step]
        zz = native_zz[::step]
        ny, nz = yy.size, zz.size
        
        logging.info(f"Grid Decimation: Factor={step}. Size: {ny}x{nz}")
        
        # Sparse Data Loading
        Nt = int(f.attrs['Iterations'])
        
        # A gprMax .out stores one dataset per receiver per component. Each receiver group is
        # opened once and all components are read from it, in increasing receiver index, so
        # the reads move forward through the file.
        present = [c for c in comps if f"/rxs/rx1/{c}" in f]
        for c in present:
            fields[c] = np.zeros((nz, ny, Nt), dtype=np.float32)

        visits = []
        if is_y_fast:
            for iz_new, iz_old in enumerate(range(0, nz_native, step)):
                row_start = iz_old * ny_native
                for iy_new, iy_old in enumerate(range(0, ny_native, step)):
                    visits.append((iz_new, iy_new, row_start + iy_old))
        else: # Z-Fast
            for iy_new, iy_old in enumerate(range(0, ny_native, step)):
                row_start = iy_old * nz_native
                for iz_new, iz_old in enumerate(range(0, nz_native, step)):
                    visits.append((iz_new, iy_new, row_start + iz_old))
        visits.sort(key=lambda v: v[2])

        for iz_new, iy_new, k in visits:
            group = f[f"/rxs/rx{k+1}"]      # path resolved once per receiver, not per read
            for c in present:
                # read_direct fills the destination row without allocating a temporary
                group[c].read_direct(fields[c], dest_sel=np.s_[iz_new, iy_new, :])

    return fields, x1, yy, zz, dt, Nt


def build_padded_plane_spectrum(fields_t: Dict[str, np.ndarray],
                                dt: float,
                                dR_min: float,
                                dR_max: float,
                                c_med: float,
                                cutoff_hz: float = None,
                                row_block: int = 16):

    """
    Zero-mean -> pad (global) -> rFFT for each component present on the plane.
    Padding avoids circular wrap for delays in [dR_min/c_med, dR_max/c_med].
    Returns:
      Epos: comp -> (nz, ny, Npos)
      omega_pos: (Npos,)
      Npre, Nt_out, Nfft
    """
    
    any_comp = next(iter(fields_t.values()))
    nz, ny, Nt = any_comp.shape
    tau_min = dR_min / c_med
    tau_max = dR_max / c_med
    Npre  = int(np.ceil(max(0.0, -tau_min) / dt))
    Npost = int(np.ceil(max(0.0,  tau_max) / dt))
    Nt_out = Npre + (Nt + Npost)*2
    Nfft = 2 * next_pow2(Nt_out)

    omega_pos = 2.0 * np.pi * np.fft.rfftfreq(Nfft, dt)

    # Only the retained frequency bins are kept, and the plane is transformed one block of
    # rows at a time, so the full-band spectrum of the whole plane is never held in memory.
    # rfft and the mean act along axis 2 only, so blocking along iz does not change the result.
    keep = (np.ones(omega_pos.size, dtype=bool) if cutoff_hz is None
            else (omega_pos / (2.0 * np.pi)) <= cutoff_hz)
    n_keep = int(keep.sum())

    Epos = {}
    for c, E in fields_t.items():
        out = np.empty((nz, ny, n_keep), dtype=np.complex64)
        for iz0 in range(0, nz, row_block):
            iz1 = min(iz0 + row_block, nz)
            block = E[iz0:iz1]
            b0 = block - np.mean(block, axis=2, keepdims=True)
            pad = np.zeros((iz1 - iz0, ny, Nt_out), dtype=np.float32)
            pad[:, :, Npre:Npre+Nt] = b0
            out[iz0:iz1] = np.fft.rfft(pad, n=Nfft, axis=2)[..., keep].astype(np.complex64)
        Epos[c] = out

    return Epos, omega_pos, keep, Npre, Nt_out, Nfft


def same_ray_propagate_point(Epos_plane: Dict[str, np.ndarray],
                             yy: np.ndarray, zz: np.ndarray, x1: float,
                             xs: float, ys: float, zs: float,
                             xe: float, ye: float, ze: float,
                             gamma_pos: np.ndarray,
                             comps_present: Tuple[bool, bool, bool],
                             gamma_pos1: np.ndarray = None,   # TWO-MEDIUM: gamma_1 (acquisition medium)
                             eamp: np.ndarray = None) -> np.ndarray:  # TWO-MEDIUM: y1/y2 for E
    """
    Ray-aligned sampling and along-ray propagation from plane x1 to element (xe,ye,ze).
    Returns E_cart(ω) as a 3 x Npos complex array [Ex; Ey; Ez];
    Missing components are treated as zeros.
    """
    hasEx, hasEy, hasEz = comps_present
    Npos = gamma_pos.size
    E_out = np.zeros((3, Npos), dtype=complex)

    denom = (xe - xs)
    if np.isclose(denom, 0.0, atol=1e-12): return E_out

    lam = (x1 - xs) / denom
    y1m = ys + lam * (ye - ys)
    z1m = zs + lam * (ze - zs)
    if not (yy[0] <= y1m <= yy[-1]) or not (zz[0] <= z1m <= zz[-1]):
        return E_out
    
    # Bilinear interpolation indices/weights
    iy0, wy = interp_weights_sorted(yy, y1m)
    iz0, wz = interp_weights_sorted(zz, z1m)

    # Distances along the ray
    R2 = np.sqrt((xe - xs)**2 + (ye - ys)**2 + (ze - zs)**2)
    R1 = np.sqrt((x1 - xs)**2 + (y1m - ys)**2 + (z1m - zs)**2)
    dR = R2 - R1

    # ----------------------------------------------------------------------------
    if gamma_pos1 is None: gamma_pos1 = gamma_pos
    if eamp is None: eamp = 1.0
    # gR1 uses gamma_1 (Medium-1 near-field shape at R1); gR2 uses gamma_2 (Medium-2 at R2)
    gR1 = gamma_pos1 * R1
    gR2 = gamma_pos  * R2
    # Phase/attenuation: remove Medium-1 phase up to R1, add Medium-2 phase to R2
    attph = np.exp(-(gamma_pos * R2 - gamma_pos1 * R1))
    # ------------------------------------------------------------------------------

    # Helpers
    def blend(comp: str) -> np.ndarray:
        E = Epos_plane[comp]
        E00 = E[iz0,     iy0,     :]
        E01 = E[iz0,     iy0 + 1, :]
        E10 = E[iz0 + 1, iy0,     :]
        E11 = E[iz0 + 1, iy0 + 1, :]
        w00 = (1.0 - wz) * (1.0 - wy)
        w01 = (1.0 - wz) * wy
        w10 = wz * (1.0 - wy)
        w11 = wz * wy
        return w00 * E00 + w01 * E01 + w10 * E10 + w11 * E11  

    # Ex, Ey: along-ray exact ratio
    P1 = 3.0 + 3.0 * gR1 + gR1 * gR1
    P2 = 3.0 + 3.0 * gR2 + gR2 * gR2
    ratio_T = ((R1 / R2)**3) * (P2 / P1) * attph * eamp  # TWO-MEDIUM: * (y1/y2)

    if hasEx:
        E1 = blend('Ex')
        E_out[0, :] = E1 * ratio_T
    if hasEy:
        E1 = blend('Ey')
        E_out[1, :] = E1 * ratio_T

    # Ez: shape function along same ray
    nzr = (ze - zs) / R2
    S1_num = ((3.0 * nzr * nzr - 1.0) * (1.0 + gR1)) + ((nzr * nzr - 1.0) * (gR1 * gR1))
    S2_num = ((3.0 * nzr * nzr - 1.0) * (1.0 + gR2)) + ((nzr * nzr - 1.0) * (gR2 * gR2))
    S1 = S1_num / (R1**3)
    S2 = S2_num / (R2**3)
    ratio_Z = (S2 / S1) * attph * eamp # TWO-MEDIUM: * (y1/y2)
    if hasEz:
        E1z = blend('Ez')
        E_out[2, :] = E1z * ratio_Z

    return E_out


def same_ray_propagate_H_dipole(Hpos_plane: Dict[str, np.ndarray],
                                yy: np.ndarray, zz: np.ndarray, x1: float,
                                xs: float, ys: float, zs: float,
                                xe: float, ye: float, ze: float,
                                gamma_pos: np.ndarray,
                                comps_present: Tuple[bool, bool, bool],
                                gamma_pos1: np.ndarray = None) -> np.ndarray:
    """
    Ray-aligned propagation specifically for the MAGNETIC FIELD (H) of a Z-directed Electric Dipole.
    
    Physics differences from E-field:
    H-field has no radial/longitudinal component (Hz = 0 theoretically).
    H-field decay polynomial is (1/r + 1/r^2), lacking the 1/r^3 static term.
    """
    hasHx, hasHy, hasHz = comps_present
    Npos = gamma_pos.size
    H_out = np.zeros((3, Npos), dtype=complex)

    denom = (xe - xs)
    if np.isclose(denom, 0.0, atol=1e-12):
        return H_out

    lam = (x1 - xs) / denom
    y1m = ys + lam * (ye - ys)
    z1m = zs + lam * (ze - zs)

    if not (yy[0] <= y1m <= yy[-1]) or not (zz[0] <= z1m <= zz[-1]):
        return H_out

    # Bilinear interpolation
    iy0, wy = interp_weights_sorted(yy, y1m)
    iz0, wz = interp_weights_sorted(zz, z1m)

    R2 = np.sqrt((xe - xs)**2 + (ye - ys)**2 + (ze - zs)**2)
    R1 = np.sqrt((x1 - xs)**2 + (y1m - ys)**2 + (z1m - zs)**2)
    dR = R2 - R1
    # --- split gamma; H carries NO y1/y2 amplitude factor ---
    if gamma_pos1 is None: gamma_pos1 = gamma_pos
    gR1 = gamma_pos1 * R1
    gR2 = gamma_pos  * R2
    attph = np.exp(-(gamma_pos * R2 - gamma_pos1 * R1))

    # --- H-FIELD SPECIFIC POLYNOMIALS ---
    # Theory: H ~ (1/r + 1/gr^2) ~ (gr + 1)/r^2
    # We factor out the 1/r^2 geometric term explicitly, 
    # leaving the polynomial Q = (1 + gR)
    
    Q1 = 1.0 + gR1
    Q2 = 1.0 + gR2
    
    # The Ratio: (R1/R2)^2 * (Q2/Q1) * phase
    # Note: E-field used (R1/R2)^3. H-field uses (R1/R2)^2 because it lacks the 1/r^3 term.
    ratio_H = ((R1 / R2)**2) * (Q2 / Q1) * attph

    # Helpers
    def blend(comp: str) -> np.ndarray:
        H = Hpos_plane[comp]
        H00 = H[iz0,     iy0,     :]
        H01 = H[iz0,     iy0 + 1, :]
        H10 = H[iz0 + 1, iy0,     :]
        H11 = H[iz0 + 1, iy0 + 1, :]
        w00 = (1.0 - wz) * (1.0 - wy)
        w01 = (1.0 - wz) * wy
        w10 = wz * (1.0 - wy)
        w11 = wz * wy
        return w00 * H00 + w01 * H01 + w10 * H10 + w11 * H11

    # Propagate Transverse Components (Hx, Hy)
    if hasHx:
        H_out[0, :] = blend('Hx') * ratio_H
    if hasHy:
        H_out[1, :] = blend('Hy') * ratio_H
        
    # Hz Component
    # For a Z-directed Electric Dipole, Hz is theoretically ZERO.
    # We generally do NOT propagate it to avoid amplifying numerical noise.
    # If you absolutely must (e.g., dipole is tilted), use the same ratio_H.
    if hasHz:
         # H_out[2, :] = blend('Hz') * ratio_H # Optional: Uncomment if source is tilted
         pass 

    return H_out

