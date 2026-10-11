#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
RA-HPO: Ray-Aligned Hybrid Physical Optics forward solver for borehole radar.

A single FDTD simulation (gprMax) resolves the near-borehole region and records the field on a
reference plane. This solver propagates that field along rays to a meshed fracture surface,
scatters it by physical optics with three-layer TE/TM reflection coefficients, and couples the
scattered field back to the receiver through electromagnetic reciprocity.

Copyright (c) 2026 Jinqiang Chen, Alexis Shakas, Evert Slob. MIT License (see LICENSE).
"""

import os
import time
import sys
import argparse
import logging
import numpy as np
from scipy.ndimage import gaussian_filter
from scipy.interpolate import RegularGridInterpolator
from scipy.stats import norm
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from matplotlib.colors import TwoSlopeNorm
from matplotlib.lines import Line2D
from mpl_toolkits.mplot3d import Axes3D
import h5py
from dataclasses import dataclass
from typing import Dict, Any, Tuple, Optional

# Attempt to import joblib for parallel processing
try:
    from joblib import Parallel, delayed
    HAVE_JOBLIB = True
except ImportError:
    HAVE_JOBLIB = False

# -----------------------------------------------------------------------------
# GLOBAL CONSTANTS
# -----------------------------------------------------------------------------
C0  = 299_792_458.0
MU0 = 4.0 * np.pi * 1e-7
EPS0 = 1.0 / (C0 * C0 * MU0)
FC  = 100e6  # center frequency

# -----------------------------------------------------------------------------
# DATACLASSES (Configuration Structures)
# -----------------------------------------------------------------------------
@dataclass
class MediaParams:
    epsr2: float = 9.0       # background (rock)
    epsr_plane: float = 80.0 # fracture fluid
    sigma_rock: float = 1e-4
    sigma_water: float = 0.5
    # Medium 1: the host in which the reference plane was recorded. It equals the background
    # (medium 2) unless the plane is re-radiated into a different medium (two-medium mode).
    epsr1: float = 9.0
    sigma_rock1: float = 1e-4

@dataclass
class SourceParams:
    # Source/Rec positions (Must match gprMax simulation)
    src_pos: Tuple[float, float, float] = (0.5, 5.0, 5.5)
    rec_pos: Tuple[float, float, float] = (0.5, 5.0, 4.5)

@dataclass
class GeometryParams:
    ellength: float = 0.01    # Discretization size
    planethick: float = 0.04  # Fracture aperture
    amplitude: float = 0.25   # Corrugation amplitude A, used by fracture_type='sinusoidal_3d'
    
    # Grid bounds for the fracture generation (y-z plane)
    ymin: float = 1.0
    ymax: float = 9.0
    zmin: float = 1.0
    zmax: float = 9.0
    
    # Fracture Center for the piecewise logic
    y_center: float = 5.0
    z_center: float = 5.0
    
    # Base X position
    x_base: float = 6.5

    # Element area in the reaction integral: 'true' weights each element by its surface area,
    # ellength**2 / |n_x|; 'projected' by the projected cell ellength**2. The two are identical
    # on a planar fracture (n_x = 1).
    element_area: str = 'true'

    # Flag for heterogeneity
    is_heterogeneous: bool = False
    fracture_type: str = 'planar'  # Keep track of geometry
    sigma_gpr: float = 10.0        # Add dynamic sigma
    phi_min: float = 0.1           # Add dynamic phi
    phi_max: float = 0.9           # Add dynamic phi
    thickness_var: float = 0.0

    load_from_file: bool = False
    # Paths will be populated dynamically via argparse in __main__
    syz_path: str = ""
    h5_path: str = ""
    mat_path: str = ""

@dataclass
class RuntimeParams:
    cores: int = 24
    critdist: float = 0.0 
    tapw: float = 0.0  
    tx_plane_file: str = "" # Path populated via argparse
    rx_plane_file: str = "" # Path populated via argparse
    comps: Tuple[str, ...] = ("Ex", "Ey", "Ez", "Hx", "Hy", "Hz")
    no_viz: bool = False  # skip the diagnostic 3D geometry plot
    # Receiver spacing to read the plane at, in metres. None reads the plane as recorded. Set
    # it to read a finely recorded plane at a coarser spacing.
    target_res: Optional[float] = None
    # Cosine-squared edge taper on the element contributions, width in metres. 0.0 means no
    # taper (bounded fractures); the infinite planar case uses it. See edge_taper_weights().
    edge_taper: float = 0.0
    # Incidence direction used for the reflection at each element. "band" (the method) takes
    # one direction per element, from the band-averaged real Poynting vector, and applies it at
    # every frequency. "perfreq" is a diagnostic: it recomputes the direction, the TE/TM basis
    # and the reflection angle at each frequency, and logs how far the per-frequency directions
    # spread around the band average. See per_frequency_directions().
    incidence: str = "band"
    # Filled by run_simulation in "perfreq" mode: (element positions, spread in rad, contribution)
    incidence_spread: Optional[tuple] = None

@dataclass
class WorkerParams:
    # State payload passed to parallel worker nodes
    
    Epos_tx: Dict[str, np.ndarray]
    Epos_rx: Dict[str, np.ndarray]

    yy: np.ndarray; zz: np.ndarray; x1: float
    r_yy: np.ndarray; r_zz: np.ndarray; r_x1: float
    omega_pos: np.ndarray; gamma_pos: np.ndarray
    gamma_pos1: np.ndarray; eamp: np.ndarray # Medium-1 propagation constant and E amplitude factor (y1/y2)
    dt: float; Npre: int; Nt_out: int; Nfft: int; v: float
    epsr2: float; eps0_local: float
    
    # Standardized naming to avoid confusion
    srcx: float; srcy: float; srcz: float
    recx: float; recy: float; recz: float
    
    # Specular point coords for tapering
    spec_x: float; spec_y: float; spec_z: float
    ellength: float; tapx: np.ndarray; tapy: np.ndarray; critdist: float
    
    # Optional placeholders
    doradpat: int = 0
    pws_interpor: Optional[Any] = None
    use_shift: bool = False  # reciprocal field by axial shift of the forward field
    incidence: str = "band"  # "band" or the "perfreq" diagnostic, see RuntimeParams.incidence

# -----------------------------------------------------------------------------
# GEOMETRY GENERATION
# -----------------------------------------------------------------------------
def fracture_surface_function(y_grid, z_grid, geom: GeometryParams):
    """
    Defines the fracture shape x = f(y, z).
    y_grid and z_grid representing the **local** coordinates.
    """
    # Local coordinates relative to center
    Y = y_grid
    Z = z_grid

    if geom.fracture_type == 'planar':
        x_surf = np.full_like(Y, geom.x_base) 
    elif geom.fracture_type == 'planar-tilted':
        x_surf = (Z - 50.25)/(-7.11537)
    elif geom.fracture_type == 'piecewise_yz':
        y0 = geom.y_center-geom.ymin
        z0 = geom.z_center-geom.zmin
        R = np.sqrt((Z - z0)**2 + (Y - y0)**2)
        inner_radius = 1.5
        outer_radius = 1.5 
        
        x_surf = np.zeros_like(R)
        x_flat = geom.x_base
    
        # Region 1: Flat
        mask1 = R <= inner_radius
        x_surf[mask1] = x_flat
        
        # Region 2: Transition (Linear)
        mask2 = (R > inner_radius) & (R <= outer_radius)
        if np.any(mask2):
            t = (R[mask2] - inner_radius) / (outer_radius - inner_radius)
            drop = 0.1 * (outer_radius - inner_radius)
            x_surf[mask2] = x_flat - drop * t
            
        # Region 3: Curved (Parabolic Bowl)
        mask3 = R > outer_radius
        if np.any(mask3):
            curvature = -3.0/18.0
            drop_at_boundary = 0.1 * (outer_radius - inner_radius)
            x_start_curved = x_flat - drop_at_boundary
            x_surf[mask3] = x_start_curved + curvature * (R[mask3] - outer_radius)**2
    elif geom.fracture_type == 'sinusoidal_3d':
        # Biaxial corrugation; the lateral periods follow the meshed extent.
        A = geom.amplitude
        kz = 2 * np.pi / ((geom.zmax - geom.zmin) * 2 / 3) 
        ky = 2 * np.pi / ((geom.ymax - geom.ymin) / 2)
        x0 = geom.x_base 
        slope_z = 0.0 
        slope_y = 0.0 
        
        x_surf = A * np.sin(kz * Z) * np.sin(ky * Y) + slope_z * Z + slope_y * Y + x0
    else:
        raise ValueError(f"Unknown fracture type: {geom.fracture_type}")

    return x_surf

def compute_manifold_uv(X_surf: np.ndarray, dy: float, dz: float) -> tuple:
    """Computes 2D unwrapped arc-length coordinates (U, V) by integrating the surface gradient."""
    dX_dY = np.gradient(X_surf, dy, axis=0)
    dX_dZ = np.gradient(X_surf, dz, axis=1)
    
    ds_y = np.sqrt(1.0 + dX_dY**2) * dy
    ds_z = np.sqrt(1.0 + dX_dZ**2) * dz
    
    # Cumulative numerical integration
    U = np.cumsum(ds_y, axis=0) - ds_y[0:1, :] / 2.0
    V = np.cumsum(ds_z, axis=1) - ds_z[:, 0:1] / 2.0
    
    # Anchor to 0,0
    return U - U[0:1, :], V - V[:, 0:1]

def load_fracture_properties_from_files(Y_local, Z_local, geom, media):
    """Directly extracts 2D fracture properties from the 3D gprMax HDF5 grid."""
    logging.info(f"Loading exact FDTD properties from: {geom.h5_path}")
    
    mat_dict = {}
    with open(geom.mat_path, 'r') as f:
        for line in f:
            if line.startswith("#material"):
                parts = line.split()
                epsr, sigma = float(parts[1]), float(parts[2])
                mat_id = int(parts[5].replace('frc_mat', ''))
                mat_dict[mat_id] = (epsr, sigma)

    with h5py.File(geom.h5_path, 'r') as f:
        h = f.attrs['dx_dy_dz'][0]
        data = np.array(f['data'], dtype=np.int16)

    with h5py.File(geom.syz_path, 'r') as f:
        S_yz = np.array(f['S_yz'])
        
    nx, ny, nz = data.shape
    y_1d = (np.arange(ny) + 0.5) * h
    z_1d = (np.arange(nz) + 0.5) * h

    # Collapse the 3D curved fracture into a 2D property map
    # Since background is -1 and materials are >= 0, max() along X isolates the fracture!
    mat_2d = np.max(data, axis=0) 
    
    epsr_2d = np.full((ny, nz), media.epsr2)
    sigma_2d = np.full((ny, nz), media.sigma_rock)
    
    # Map IDs to properties
    for j in range(ny):
        for k in range(nz):
            m_id = mat_2d[j, k]
            if m_id in mat_dict:
                epsr_2d[j, k] = mat_dict[m_id][0]
                sigma_2d[j, k] = mat_dict[m_id][1]

    # Map FDTD coordinates to RA-HPO Local Coordinates
    # Using method='nearest' perfectly preserves the FDTD integer step staircasing
    interp_epsr = RegularGridInterpolator((y_1d, z_1d), epsr_2d, method='nearest', bounds_error=False, fill_value=media.epsr2)
    interp_sigma = RegularGridInterpolator((y_1d, z_1d), sigma_2d, method='nearest', bounds_error=False, fill_value=media.sigma_rock)
    interp_pert = RegularGridInterpolator((y_1d, z_1d), S_yz, method='nearest', bounds_error=False, fill_value=None)

    # Clip the local coordinates strictly to the FDTD cell centers
    Y_local = np.clip(Y_local, y_1d[0], y_1d[-1])
    Z_local = np.clip(Z_local, z_1d[0], z_1d[-1])
    pts_local = np.stack((Y_local.ravel(), Z_local.ravel()), axis=-1)
    
    epsr_out = interp_epsr(pts_local).reshape(Y_local.shape)
    sigma_out = interp_sigma(pts_local).reshape(Z_local.shape)

    perturbation = interp_pert(pts_local).reshape(Y_local.shape)
    thick_var = getattr(geom, 'thickness_var', 0.0)
    thick_out = geom.planethick * (1.0 + thick_var * perturbation)
    
    return thick_out, epsr_out, sigma_out

def fracture_properties_function(Y, Z, geom, media):
    """Returns spatially varying thickness, permittivity, and conductivity."""
    shape = Y.shape
    if getattr(geom, 'is_heterogeneous', False) == False:
        thick = np.full(shape, geom.planethick)
        epsr = np.full(shape, media.epsr_plane)
        sigma = np.full(shape, media.sigma_water)
        return thick, epsr, sigma
    
    # First, map the RA-HPO local elements (Y, Z) to their unwrapped (U, V) coordinates
    Y_local = Y - geom.ymin
    Z_local = Z - geom.zmin

    if getattr(geom, 'load_from_file', False) and os.path.exists(geom.h5_path):
        logging.info("Loading fracture properties directly from gprMax files...")
        return load_fracture_properties_from_files(Y_local, Z_local, geom, media)

    # -------------------------------------------------------------------
    # Without files: recreate the gprMax (U, V) unwrapped field analytically
    # -------------------------------------------------------------------
    ny_gpr, nz_gpr = 800, 800
    h_gpr = 0.01
    sigma_gpr = getattr(geom, 'sigma_gpr', 10.0)
    phi_min = getattr(geom, 'phi_min', 0.1)
    phi_max = getattr(geom, 'phi_max', 0.9)
    
    y_1d = (np.arange(ny_gpr) + 0.5) * h_gpr
    z_1d = (np.arange(nz_gpr) + 0.5) * h_gpr
    Y_gpr, Z_gpr = np.meshgrid(y_1d, z_1d, indexing='ij')
    
    # Replicate the exact FDTD surface to get identical U, V base matrices
    X_surf_gpr = fracture_surface_function(Y_gpr, Z_gpr, geom)
    U_gpr, V_gpr = compute_manifold_uv(X_surf_gpr, h_gpr, h_gpr)

    u_max, v_max = np.max(U_gpr), np.max(V_gpr)
    nu = int(np.ceil(u_max / h_gpr)) + 1
    nv = int(np.ceil(v_max / h_gpr)) + 1
    u_vec = np.arange(nu) * h_gpr
    v_vec = np.arange(nv) * h_gpr

    interp_Y_to_U = RegularGridInterpolator((y_1d, z_1d), U_gpr, bounds_error=False, fill_value=None)
    interp_Z_to_V = RegularGridInterpolator((y_1d, z_1d), V_gpr, bounds_error=False, fill_value=None)

    # If using the unwrapped U, V logic, must interpolate these global Y,Z coordinates 
    # to find the corresponding local u_spec, v_spec
    y_spec_local = ny_gpr * h_gpr / 2.0
    z_spec_local = nz_gpr * h_gpr / 2.0
    u_spec = interp_Y_to_U(np.array([[y_spec_local, z_spec_local]]))[0]
    v_spec = interp_Z_to_V(np.array([[y_spec_local, z_spec_local]]))[0]

    idx_u_spec = np.argmin(np.abs(u_vec - u_spec))
    idx_v_spec = np.argmin(np.abs(v_vec - v_spec))
    
    np.random.seed(43)
    Lc_pixels = 2.0 * sigma_gpr
    buffer = int(4.0 * Lc_pixels)
    pan_margin = 0.0 #int(max(20, sigma_gpr))
    total_pad = buffer + pan_margin
    padded_shape = (nu + 2 * total_pad, nv + 2 * total_pad)
    padded_noise = np.random.normal(loc=0.0, scale=1.0, size=padded_shape)
    
    # correlated_padded = gaussian_filter(padded_noise, sigma=sigma_gpr) 
    # FFT spectral synthesis of an exponential-covariance field
    noise_fft = np.fft.fft2(padded_noise)
    
    # Create normalized frequency grids
    kx = np.fft.fftfreq(padded_shape[0])
    ky = np.fft.fftfreq(padded_shape[1])
    KX, KY = np.meshgrid(kx, ky, indexing='ij')
    K_sq = KX**2 + KY**2
    
    # Power Spectral Density (PSD) of a 2D Exponential Covariance function
    # S(k) is proportional to (1 + (2*pi*k*L_c)^2)^(-3/2)
    psd = (1.0 + (2.0 * np.pi * Lc_pixels)**2 * K_sq)**(-1.5)
    
    # The required Fourier domain filter is the square root of the PSD
    filter_response = np.sqrt(psd)
    
    # Apply filter and transform back to spatial domain
    field_fft = noise_fft * filter_response
    correlated_padded = np.real(np.fft.ifft2(field_fft))
    #----------------------------------------------------------------------------------------

    f_mean = np.mean(correlated_padded)
    f_std = np.std(correlated_padded)
    std_large = (correlated_padded - f_mean) / f_std if f_std > 1e-10 else correlated_padded - f_mean
    
    # Define search zone strictly using pan_margin (Protects the 4*sigma filter_pad)
    center_u = idx_u_spec + total_pad
    center_v = idx_v_spec + total_pad
    
    align_specular = False
    if align_specular:
        valid_zone = std_large[center_u - pan_margin : center_u + pan_margin,
                            center_v - pan_margin : center_v + pan_margin]
        
        # Find the value naturally closest to 0.0 in this safe zone
        idx_zero_flat = np.argmin(np.abs(valid_zone))
        local_u_zero, local_v_zero = np.unravel_index(idx_zero_flat, valid_zone.shape)
        
        # Map back to global coordinates within std_large
        i_zero = (center_u - pan_margin) + local_u_zero
        j_zero = (center_v - pan_margin) + local_v_zero
    else:
        # Bypass alignment: strictly use the geometric center of the pad
        i_zero = center_u
        j_zero = center_v
    
    # Calculate crop boundaries
    u_start = i_zero - idx_u_spec
    v_start = j_zero - idx_v_spec
    
    # Extract the field over the fracture
    std_field_cropped = std_large[u_start : u_start + nu, v_start : v_start + nv]

    # Map through CDF
    uniform_field = norm.cdf(std_field_cropped)
    stochastic_uv = 2.0 * uniform_field - 1.0
    
    # ---------------------------------------------------------
    # DOUBLE INTERPOLATION: (Y,Z) -> (U,V) -> STOCHASTIC VALUE
    # ---------------------------------------------------------
    # Clip the local coordinates strictly to the FDTD cell centers
    Y_local = np.clip(Y_local, y_1d[0], y_1d[-1])
    Z_local = np.clip(Z_local, z_1d[0], z_1d[-1])
    
    pts_local = np.stack((Y_local.ravel(), Z_local.ravel()), axis=-1)
    U_local = interp_Y_to_U(pts_local)
    V_local = interp_Z_to_V(pts_local)

    U_local = np.clip(U_local, u_vec[0], u_vec[-1])
    V_local = np.clip(V_local, v_vec[0], v_vec[-1])
    
    # Now query the rigorously unwrapped field using the exact arc-length coordinates
    interp_uv = RegularGridInterpolator((u_vec, v_vec), stochastic_uv, bounds_error=False, fill_value=0.0)
    
    pts_uv = np.stack((U_local, V_local), axis=-1)
    perturbation = interp_uv(pts_uv).reshape(shape)
    
    # ---------------------------------------------------------
    # APPLY GEOMETRY AND PETROPHYSICS (CRIM)
    # ---------------------------------------------------------
    # Variant Aperture/Thickness (+/- 50% variation) - 0.0 multiplier keeps it constant here
    thick = geom.planethick * (1.0 + 0.0 * perturbation)
    
    # Map perturbation [-1.0, 1.0] to porosity 
    phi = phi_min + (phi_max - phi_min) * ((perturbation + 1.0) / 2.0)
    
    omega = 2.0 * np.pi * FC
    
    epsr_matrix = media.epsr2      
    sigma_matrix = media.sigma_rock
    epsr_fluid = media.epsr_plane
    sigma_fluid = media.sigma_water 
    
    eps_m_star = epsr_matrix - 1j * (sigma_matrix / (omega * EPS0))
    eps_f_star = epsr_fluid - 1j * (sigma_fluid / (omega * EPS0))
    
    sqrt_eff = (1.0 - phi) * np.sqrt(eps_m_star) + phi * np.sqrt(eps_f_star)
    eps_eff_star = sqrt_eff**2
    
    epsr = np.real(eps_eff_star)
    sigma = -np.imag(eps_eff_star) * omega * EPS0
    
    return thick, epsr, sigma


def get_fracture_mesh(geom: GeometryParams, media: MediaParams):
    """Generates mesh for an arbitrary surface and computes Local Normal Vectors."""
    y_vec = np.arange(geom.ymin, geom.ymax, geom.ellength) + (geom.ellength / 2.0)
    z_vec = np.arange(geom.zmin, geom.zmax, geom.ellength) + (geom.ellength / 2.0)
    Y, Z = np.meshgrid(y_vec, z_vec, indexing='ij')
    # SHIFT TO LOCAL BEFORE CALLING
    Y_local = Y - geom.ymin
    Z_local = Z - geom.zmin
    
    # Compute X coordinates
    X = fracture_surface_function(Y_local, Z_local, geom)
    
    # Compute Normals (Gradient Method)
    # Surface F(x,y,z) = x - f(y,z) = 0
    # Normal ~ Gradient(F) = (1, -df/dy, -df/dz)
    
    dy = geom.ellength
    dz = geom.ellength
    
    # df/dy (axis 0), df/dz (axis 1)
    df_dy = np.gradient(X, dy, axis=0)
    df_dz = np.gradient(X, dz, axis=1)
    
    nx = np.ones_like(X)
    ny = -df_dy
    nz = -df_dz
    
    # Normalize
    mag = np.sqrt(nx**2 + ny**2 + nz**2)
    mag[mag == 0] = 1 
    nx /= mag
    ny /= mag
    nz /= mag
    
    X_front = X - (geom.planethick / 2.0) * nx
    Y_front = Y - (geom.planethick / 2.0) * ny
    Z_front = Z - (geom.planethick / 2.0) * nz
    # Flatten and Pack into elempos
    # Structure:
    # 0,1,2: x,y,z | 3,4,5: nx, ny, nz
    # 6: Area | 7: Thick | 8: eps_plane | 9: eps_back | 10: sigma_plane
    
    numel = X_front.size
    elempos = np.zeros((11, numel))
    
    elempos[0, :] = X_front.ravel()
    elempos[1, :] = Y_front.ravel()
    elempos[2, :] = Z_front.ravel()
    
    elempos[3, :] = nx.ravel()
    elempos[4, :] = ny.ravel()
    elempos[5, :] = nz.ravel()
    
    # Area of the surface element above each projected cell: dS = sqrt(1 + f_y^2 + f_z^2) dy dz =
    # ellength^2 * mag, with mag the norm of (1, -f_y, -f_z) above.
    if geom.element_area == 'true':
        elempos[6, :] = (geom.ellength ** 2.0) * mag.ravel()
    elif geom.element_area == 'projected':
        elempos[6, :] = geom.ellength ** 2.0
    else:
        raise ValueError(f"Unknown element_area: {geom.element_area}")

    thick, epsr, sigma = fracture_properties_function(Y_front, Z_front, geom, media)
    elempos[7, :] = thick.ravel()
    elempos[8, :] = epsr.ravel()
    elempos[9, :] = media.epsr2
    elempos[10, :] = sigma.ravel()
    
    return elempos

def get_specular_point_numerical(src_pos, rec_pos, elempos):
    """Finds the specular point on an arbitrary mesh by minimizing path length."""
    src = np.array(src_pos)[:, None] 
    rec = np.array(rec_pos)[:, None] 
    els = elempos[0:3, :]            
    
    dist_tx = np.sqrt(np.sum((els - src)**2, axis=0))
    dist_rx = np.sqrt(np.sum((els - rec)**2, axis=0))
    
    total_dist = dist_tx + dist_rx
    idx_min = np.argmin(total_dist)
    
    P_spec = els[:, idx_min]
    return P_spec, idx_min

# -----------------------------------------------------------------------------
# STANDARD UTILS
# -----------------------------------------------------------------------------
def next_pow2(n): return 1 << (int(n - 1).bit_length())

def interp_weights_sorted(axis, coords):
    idx0 = np.searchsorted(axis, coords) - 1
    idx0 = int(np.clip(idx0, 0, len(axis) - 2))
    a0, a1 = axis[idx0], axis[idx0 + 1]
    w = (coords - a0) / (a1 - a0 + np.finfo(float).eps)
    return idx0, float(np.clip(w, 0.0, 1.0))

def gamma_of_omega(omega, mu, eps, sigma):
    eta = sigma + 1j * omega * eps
    gam = np.sqrt(1j * omega * mu * eta)
    return np.where(np.real(gam) < 0, -gam, gam)

def build_taper(critdist, ellength, tapw):
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

def edge_taper_weights(y, z, width, box):
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

def analytical_ricker_spectrum(omega, fc, amp=1.0):
    zeta = (np.pi ** 2) * (fc ** 2)
    pre = (amp / (2.0 * zeta)) * np.sqrt(np.pi / zeta)
    return pre * (omega**2) * np.exp(-(omega**2) / (4.0 * zeta))

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

# -----------------------------------------------------------------------------
# GPRMAX LOADER
# -----------------------------------------------------------------------------
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

# -----------------------------------------------------------------------------
# FIELD PROPAGATORS
# -----------------------------------------------------------------------------

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

# -----------------------------------------------------------------------------
# KERNEL: Element Contribution
# -----------------------------------------------------------------------------
def element_contribution(iel, elempos, p: WorkerParams, media: MediaParams):
    """Calculates the scattering contribution (reaction integral) of a single fracture element."""
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

# -----------------------------------------------------------------------------
# MAIN DRIVER
# -----------------------------------------------------------------------------
def run_simulation(media: MediaParams, srcpar: SourceParams, geom: GeometryParams, run: RuntimeParams):
    """Executes the high-level RA-HPO computational flow."""
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

def save_trace(filename, tvec, trace):
    """Saves the output traces to standard npz format."""
    np.savez(filename, time_ns=tvec*1e9, amplitude=trace)
    logging.info(f"Trace saved to {filename}")

# -----------------------------------------------------------------------------
# ENTRY POINT
# -----------------------------------------------------------------------------
if __name__ == "__main__":
    # --- Output directories, created after the arguments are parsed ---
    image_dir = "./images"
    log_dir = "./logs"
    data_dir = "./npz_data"

    # Logging is configured after the arguments are parsed, so that each run writes its own log
    # file, named after --output_name. Nothing may log before that.

    # --- Setup Arguments ---
    parser = argparse.ArgumentParser(
        description="RA-HPO: Ray-Aligned Hybrid Physical Optics forward solver for borehole radar. "
                    "Every optional flag defaults to the configuration used in the paper.")
    g = parser.add_argument_group("reference planes")
    g.add_argument("--tx_plane_file", type=str, required=True,
                   help="Reference plane of the transmitter: a gprMax .out file or a plane "
                        "converted with convert_plane.py.")
    g.add_argument("--rx_plane_file", type=str, required=True,
                   help="Reference plane of the receiver. Pass the same file as --tx_plane_file "
                        "to obtain the reciprocal field by axial shift of the forward field.")
    g.add_argument("--target_res", type=float, default=None,
                   help="Receiver spacing in metres to read the plane at. Default: the plane "
                        "as recorded. Set it to read a finely recorded plane at a coarser "
                        "spacing (0.04 for the 0.01 m planes of the paper).")
    g.add_argument("--plane_epsr", type=float, default=None,
                   help="Relative permittivity of the host in which the plane was recorded, "
                        "for re-radiation into a different host (two-medium mode). Default: "
                        "the background permittivity, 9.0.")

    g = parser.add_argument_group("fracture")
    g.add_argument("--fracture_type", type=str, default=None,
                   choices=["planar", "planar-tilted", "piecewise_yz", "sinusoidal_3d"],
                   help="Fracture shape (default planar). planar-tilted is the 8 degree plane, "
                        "piecewise_yz the concave bowl with a flat center, sinusoidal_3d the "
                        "biaxial corrugation.")
    g.add_argument("--amplitude", type=float, default=None,
                   help="Corrugation amplitude A in metres for sinusoidal_3d (default 0.25).")
    g.add_argument("--planethick", type=float, default=None,
                   help="Fracture aperture in metres (default 0.04).")
    g.add_argument("--fracture_extent", type=float, nargs=4, default=None,
                   metavar=("YMIN", "YMAX", "ZMIN", "ZMAX"),
                   help="Meshed extent of the fracture in metres (default 1 9 1 9). For the "
                        "non-planar shapes it also sets the lateral periods.")
    g.add_argument("--ellength", type=float, default=None,
                   help="Fracture element size in metres (default 0.01).")
    g.add_argument("--element_area", type=str, default=None, choices=["true", "projected"],
                   help="Element weight in the reaction integral: 'true' (default) is the "
                        "surface area of each element, ellength^2/|n_x|; 'projected' is "
                        "ellength^2. Identical on planar fractures.")
    g.add_argument("--edge_taper", type=float, default=0.0,
                   help="Width in metres of a cosine-squared taper at the edges of the meshed "
                        "fracture, to approximate an unbounded fracture (default 0, no taper).")

    g = parser.add_argument_group("fracture fill")
    g.add_argument("--fluid_epsr", type=float, default=None,
                   help="Relative permittivity of the fracture fluid (default 80.0, water).")
    g.add_argument("--fluid_sigma", type=float, default=None,
                   help="Conductivity of the fracture fluid in S/m (default 0.5).")
    g.add_argument("--heterogeneous", action="store_true",
                   help="Read a heterogeneous fill from --h5_geom, --syz_geom and --mat_file, "
                        "as written by generate_heter_geometry_h5_and_materials.py.")
    g.add_argument("--h5_geom", type=str, default="",
                   help="Geometry volume of the heterogeneous fill (with --heterogeneous).")
    g.add_argument("--syz_geom", type=str, default="",
                   help="Fracture-plane property map of the heterogeneous fill (with --heterogeneous).")
    g.add_argument("--mat_file", type=str, default="",
                   help="Materials file of the heterogeneous fill (with --heterogeneous).")

    g = parser.add_argument_group("acquisition")
    g.add_argument("--src_pos", type=float, nargs=3, default=None, metavar=("X", "Y", "Z"),
                   help="Source position in metres (default 0.5 5.0 5.5). The reference plane "
                        "is recorded for this position, so move it only along the borehole axis.")
    g.add_argument("--rec_pos", type=float, nargs=3, default=None, metavar=("X", "Y", "Z"),
                   help="Receiver position in metres (default 0.5 5.0 4.5). On the borehole "
                        "axis the reciprocal field follows by axial shift.")

    g = parser.add_argument_group("run")
    g.add_argument("--output_name", type=str, default="rahpo_trace",
                   help="Base name of the .npz trace, the PNG and the log file.")
    g.add_argument("--cores", type=int, default=24,
                   help="Number of worker processes for the element integration (default 24).")
    g.add_argument("--incidence", type=str, default="band", choices=["band", "perfreq"],
                   help="'band' (default): one incidence direction per element from the "
                        "band-averaged Poynting vector. 'perfreq': diagnostic that recomputes "
                        "it at every frequency and also writes <output_name>_incidence.npz.")
    g.add_argument("--no_viz", action="store_true",
                   help="Skip the diagnostic 3D geometry plot.")

    args = parser.parse_args()
    for d in [image_dir, log_dir, data_dir]:
        os.makedirs(d, exist_ok=True)

    # --- Logging, named after this run ---
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
        handlers=[
            logging.FileHandler(os.path.join(log_dir, f"{args.output_name}.log"), mode="w"),
            logging.StreamHandler(sys.stdout),
        ],
    )
    logging.info(f"RA-HPO run '{args.output_name}', log written to "
                 f"{os.path.join(log_dir, args.output_name + '.log')}")

    if not HAVE_JOBLIB:
        logging.warning("Joblib not found. Running in strictly serial mode.")

    # --- Initialize Parameters ---
    if args.heterogeneous:
        missing = [n for n in (args.h5_geom, args.syz_geom, args.mat_file)
                   if not n or not os.path.exists(n)]
        if missing:
            parser.error("--heterogeneous needs existing --h5_geom, --syz_geom and --mat_file; "
                         f"missing: {missing}")

    # A dataclass default is overridden only when its flag is given, so omitting a flag keeps
    # the configuration of the paper.
    media_overrides = {}
    if args.fluid_epsr is not None:
        media_overrides["epsr_plane"] = args.fluid_epsr
    if args.fluid_sigma is not None:
        media_overrides["sigma_water"] = args.fluid_sigma
    if args.plane_epsr is not None:
        media_overrides["epsr1"] = args.plane_epsr
    media = MediaParams(**media_overrides)

    src_overrides = {}
    if args.src_pos is not None:
        src_overrides["src_pos"] = tuple(args.src_pos)
    if args.rec_pos is not None:
        src_overrides["rec_pos"] = tuple(args.rec_pos)
    srcpar = SourceParams(**src_overrides)
    
    geom_overrides = {}
    if args.heterogeneous:
        geom_overrides["is_heterogeneous"] = True
        geom_overrides["load_from_file"] = True
    if args.ellength is not None:
        geom_overrides["ellength"] = args.ellength
    if args.fracture_type is not None:
        geom_overrides["fracture_type"] = args.fracture_type
    if args.amplitude is not None:
        geom_overrides["amplitude"] = args.amplitude
    if args.planethick is not None:
        geom_overrides["planethick"] = args.planethick
    if args.element_area is not None:
        geom_overrides["element_area"] = args.element_area
    if args.fracture_extent is not None:
        (geom_overrides["ymin"], geom_overrides["ymax"],
         geom_overrides["zmin"], geom_overrides["zmax"]) = args.fracture_extent

    geom = GeometryParams(
        h5_path=args.h5_geom,
        syz_path=args.syz_geom,
        mat_path=args.mat_file,
        **geom_overrides
    )
    
    run = RuntimeParams(
        tx_plane_file=args.tx_plane_file,
        rx_plane_file=args.rx_plane_file,
        cores=args.cores,
        no_viz=args.no_viz,
        target_res=args.target_res,
        edge_taper=args.edge_taper,
        incidence=args.incidence,
    )

    # --- Run Kernel ---
    try:
        tvec, trace = run_simulation(media, srcpar, geom, run)
        
        # --- Handle Output ---
        base_filename = args.output_name
        save_trace(os.path.join(data_dir, f"{base_filename}.npz"), tvec, trace)
        if run.incidence == "perfreq":
            pos, spread, w = run.incidence_spread
            np.savez(os.path.join(data_dir, f"{base_filename}_incidence.npz"),
                     elem_pos=pos, spread_rad=spread, contribution=w)
            logging.info(f"Incidence spread saved to {base_filename}_incidence.npz")
        
        plt.figure()
        plt.plot(tvec*1e9, trace, 'r-')
        plt.xlabel("Time [ns]")
        plt.ylabel("Normalized amplitude")
        plt.title("Trace (RA + HPO)")
        plt.grid(True)
        plt.savefig(os.path.join(image_dir, f"{base_filename}.png"), dpi=300)
        logging.info(f"Plot saved to {os.path.join(image_dir, f'{base_filename}.png')}")
        
    except Exception as e:
        logging.error("Simulation failed with exception:", exc_info=True)
        sys.exit(1)