#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
RA-HPO: Ray-Aligned Hybrid Physical Optics Method
Version: Updated for ARBITRARY CURVED FRACTURES (Meshed Surface Approach).
Copyright (c) 2025 Jinqiang Chen
Deliverable for the GeoHEAT project

Description:
This script models the response of a borehole radar for deep geothermal exploration.
It uses a semi-analytical method, propagating numerical near-field data (from gprMax)
to a far-field fracture using ray-aligned propagation, and computes the "re-entry"
coupling back to the receiver using the Electromagnetic Reciprocity Theorem.
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
    epsr_plane: float = 43.0 # fracture fluid
    sigma_rock: float = 1e-4
    sigma_water: float = 10.0

@dataclass
class SourceParams:
    # Source/Rec positions (Must match gprMax simulation)
    src_pos: Tuple[float, float, float] = (0.5, 5.0, 5.5)
    rec_pos: Tuple[float, float, float] = (0.5, 5.0, 4.5)

@dataclass
class GeometryParams:
    ellength: float = 0.01    # Discretization size
    planethick: float = 0.04  # Fracture aperture
    
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

    # Flag for heterogeneity
    is_heterogeneous: bool = True
    fracture_type: str = 'planar'  # Keep track of geometry
    sigma_gpr: float = 10.0         # Add dynamic sigma
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
    plane_file: str = "" # Path populated via argparse
    comps: Tuple[str, ...] = ("Ex", "Ey", "Ez", "Hx", "Hy", "Hz")

@dataclass
class WorkerParams:
    # State payload passed to parallel worker nodes
    Epos: Dict[str, np.ndarray]
    E_time: Dict[str, np.ndarray]
    yy: np.ndarray; zz: np.ndarray; x1: float
    omega_pos: np.ndarray; gamma_pos: np.ndarray
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

# -----------------------------------------------------------------------------
# GEOMETRY GENERATION (Core Mathematics - Untouched)
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
        A = 0.25 
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
        return load_fracture_properties_from_files(Y_local, Z_local, geom, media)

    # -------------------------------------------------------------------
    # OPTIONAL: recreat exact gprMax (U, V) UNWRAPPED FIELD analytically
    # -------------------------------------------------------------------
    ny_gpr, nz_gpr = 800, 800
    h_gpr = 0.01
    sigma_gpr = getattr(geom, 'sigma_gpr', 5.0)
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
    y_spec_local = 4.0
    z_spec_local = 4.0
    u_spec = interp_Y_to_U(np.array([[y_spec_local, z_spec_local]]))[0]
    v_spec = interp_Z_to_V(np.array([[y_spec_local, z_spec_local]]))[0]

    idx_u_spec = np.argmin(np.abs(u_vec - u_spec))
    idx_v_spec = np.argmin(np.abs(v_vec - v_spec))
    
    np.random.seed(43)
    buffer = int(4.0 * sigma_gpr)
    pan_margin = int(max(20, sigma_gpr))
    total_pad = buffer + pan_margin
    padded_shape = (nu + 2 * total_pad, nv + 2 * total_pad)
    padded_noise = np.random.normal(loc=0.0, scale=1.0, size=padded_shape)
    
    correlated_padded = gaussian_filter(padded_noise, sigma=sigma_gpr)

    f_mean = np.mean(correlated_padded)
    f_std = np.std(correlated_padded)
    std_large = (correlated_padded - f_mean) / f_std if f_std > 1e-10 else correlated_padded - f_mean
    
    # Define search zone strictly using pan_margin (Protects the 4*sigma filter_pad)
    center_u = idx_u_spec + total_pad
    center_v = idx_v_spec + total_pad
    
    valid_zone = std_large[center_u - pan_margin : center_u + pan_margin,
                           center_v - pan_margin : center_v + pan_margin]
    
    # Find the value naturally closest to 0.0 in this safe zone
    idx_zero_flat = np.argmin(np.abs(valid_zone))
    local_u_zero, local_v_zero = np.unravel_index(idx_zero_flat, valid_zone.shape)
    
    # Map back to global coordinates within std_large
    i_zero = (center_u - pan_margin) + local_u_zero
    j_zero = (center_v - pan_margin) + local_v_zero
    
    # Calculate crop boundaries
    u_start = i_zero - idx_u_spec
    v_start = j_zero - idx_v_spec
    
    # Extract the final perfectly continuous, edge-artifact-free field!
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
    
    elempos[6, :] = geom.ellength ** 2.0 # need to be improved with actual area calculation for dip elements

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
# STANDARD UTILS (Core Math - Untouched)
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
# GPRMAX LOADER (Core Math - Untouched)
# -----------------------------------------------------------------------------
def load_gprmax_plane(filename, comps=("Ex", "Ey", "Ez", "Hx", "Hy", "Hz"), target_res=0.04):
    """Loads and decimates specified field components from a gprMax HDF5 output file."""
    if isinstance(comps, str): comps = (comps,)
    fields = {}
    with h5py.File(filename, 'r') as f:
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
            step = max(1, int(np.round(target_res / abs(step_native))))
            
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
            step = max(1, int(np.round(target_res / abs(step_native))))
            
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
        
        for c in comps:
            if f"/rxs/rx1/{c}" in f:
                field_data = np.zeros((nz, ny, Nt), dtype=np.float32)
                
                if is_y_fast:
                    for iz_new, iz_old in enumerate(range(0, nz_native, step)):
                        row_start = iz_old * ny_native
                        for iy_new, iy_old in enumerate(range(0, ny_native, step)):
                            k = row_start + iy_old
                            field_data[iz_new, iy_new, :] = f[f"/rxs/rx{k+1}/{c}"][:]
                else: # Z-Fast
                    for iy_new, iy_old in enumerate(range(0, ny_native, step)):
                        row_start = iy_old * nz_native
                        for iz_new, iz_old in enumerate(range(0, nz_native, step)):
                            k = row_start + iz_old
                            field_data[iz_new, iy_new, :] = f[f"/rxs/rx{k+1}/{c}"][:]
                
                fields[c] = field_data

    return fields, x1, yy, zz, dt, Nt

# -----------------------------------------------------------------------------
# FIELD PROPAGATORS (Core Math - Untouched)
# -----------------------------------------------------------------------------
def build_padded_plane_spectrum(fields_t: Dict[str, np.ndarray],
                                dt: float,
                                dR_min: float,
                                dR_max: float,
                                c_med: float):
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

    Epos = {}
    E_time_padded = {}
    for c, E in fields_t.items():
        E0 = E - np.mean(E, axis=2, keepdims=True)
        Epad = np.zeros((nz, ny, Nt_out), dtype=np.float32)
        Epad[:, :, Npre:Npre+Nt] = E0
        E_time_padded[c] = Epad
        Epos[c] = np.fft.rfft(Epad, n=Nfft, axis=2) 

    fpos = np.fft.rfftfreq(Nfft, dt)
    omega_pos = 2.0 * np.pi * fpos

    return Epos, E_time_padded, omega_pos, Npre, Nt_out, Nfft

def same_ray_propagate_point(Epos_plane: Dict[str, np.ndarray],
                             yy: np.ndarray, zz: np.ndarray, x1: float,
                             xs: float, ys: float, zs: float,
                             xe: float, ye: float, ze: float,
                             gamma_pos: np.ndarray,
                             comps_present: Tuple[bool, bool, bool]) -> np.ndarray:
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

    # Bilinear interpolation indices/weights
    iy0, wy = interp_weights_sorted(yy, y1m)
    iz0, wz = interp_weights_sorted(zz, z1m)

    # Distances along the ray
    R2 = np.sqrt((xe - xs)**2 + (ye - ys)**2 + (ze - zs)**2)
    R1 = np.sqrt((x1 - xs)**2 + (y1m - ys)**2 + (z1m - zs)**2)
    dR = R2 - R1

    attph = np.exp(-gamma_pos * dR) 
    gR1 = gamma_pos * R1
    gR2 = gamma_pos * R2

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
    ratio_T = ((R1 / R2)**3) * (P2 / P1) * attph

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
    ratio_Z = (S2 / S1) * attph
    if hasEz:
        E1z = blend('Ez')
        E_out[2, :] = E1z * ratio_Z

    return E_out  

def same_ray_propagate_H_dipole(Hpos_plane: Dict[str, np.ndarray],
                                yy: np.ndarray, zz: np.ndarray, x1: float,
                                xs: float, ys: float, zs: float,
                                xe: float, ye: float, ze: float,
                                gamma_pos: np.ndarray,
                                comps_present: Tuple[bool, bool, bool]) -> np.ndarray:
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
    attph = np.exp(-gamma_pos * dR)
    gR1 = gamma_pos * R1
    gR2 = gamma_pos * R2

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
# KERNEL: Element Contribution (Core Math - Untouched)
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
        p.Epos, p.yy, p.zz, p.x1, p.srcx, p.srcy, p.srcz,
        pos_el[0], pos_el[1], pos_el[2], p.gamma_pos,
        ('Ex' in p.Epos, 'Ey' in p.Epos, 'Ez' in p.Epos)
    )
    H_inc = same_ray_propagate_H_dipole(
        p.Epos, p.yy, p.zz, p.x1, p.srcx, p.srcy, p.srcz, 
        pos_el[0], pos_el[1], pos_el[2], p.gamma_pos,
        ('Hx' in p.Epos, 'Hy' in p.Epos, 'Hz' in p.Epos)
    )

    # Reciprocal Fields
    offset_vec = pos_rx - pos_tx
    pos_el_virtual = pos_el - offset_vec
    
    # FIX: Use p.srcx instead of p.xs
    denom = (pos_el_virtual[0] - p.srcx)
    if np.isclose(denom, 0.0): return np.zeros_like(p.omega_pos, dtype=complex)
    
    E_rx = same_ray_propagate_point(
        p.Epos, p.yy, p.zz, p.x1, p.srcx, p.srcy, p.srcz,
        pos_el_virtual[0], pos_el_virtual[1], pos_el_virtual[2], p.gamma_pos, 
        ('Ex' in p.Epos, 'Ey' in p.Epos, 'Ez' in p.Epos)
    )
    H_rx = same_ray_propagate_H_dipole(
        p.Epos, p.yy, p.zz, p.x1, p.srcx, p.srcy, p.srcz,
        pos_el_virtual[0], pos_el_virtual[1], pos_el_virtual[2], p.gamma_pos, 
        ('Hx' in p.Epos, 'Hy' in p.Epos, 'Hz' in p.Epos)
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
    te_vec_raw = np.cross(k_dir, n_hat)
    te_len = np.linalg.norm(te_vec_raw)
    
    if te_len < 1e-9:
        E_ref = E_inc * R_TE
    else:
        e_te = te_vec_raw / te_len
        e_tm_inc = np.cross(e_te, k_dir)
        E_inc_TE = np.dot(e_te, E_inc)
        E_inc_TM = np.dot(e_tm_inc, E_inc)
        k_ref = k_dir - 2 * np.dot(k_dir, n_hat) * n_hat
        e_tm_ref = np.cross(e_te, k_ref)
        E_ref = (e_te[:, None] * (E_inc_TE * R_TE)[None, :]) + \
                (e_tm_ref[:, None] * (E_inc_TM * R_TM)[None, :])

    H_ref = np.cross(k_ref, E_ref.T).T / eta2_scalar
    Js = np.cross(n_hat, H_ref.T).T
    Ms = np.cross(E_ref.T, n_hat).T

    term1 = np.sum(E_rx * Js, axis=0) 
    term2 = np.sum(H_rx * Ms, axis=0) 
    reaction = 1.0 * (term1 - term2) * elempos[6, iel] 

    # Optional Tapering----------------------------------------------------------
    # reaction *= tapval
    # ---------------------------------------------------------------------------

    return reaction

# -----------------------------------------------------------------------------
# MAIN DRIVER (Refactored for flow and logging)
# -----------------------------------------------------------------------------
def run_simulation(media: MediaParams, srcpar: SourceParams, geom: GeometryParams, run: RuntimeParams):
    """Executes the high-level RA-HPO computational flow."""
    start_t = time.time()
    
    xs, ys, zs = srcpar.src_pos
    xr, yr, zr = srcpar.rec_pos
    
    logging.info("Generating fracture mesh...")
    elempos = get_fracture_mesh(geom, media)
    numelem = elempos.shape[1]
    logging.info(f"Generated {numelem} elements.")

    logging.info("Finding specular point...")
    P_spec, idx_spec = get_specular_point_numerical(srcpar.src_pos, srcpar.rec_pos, elempos)
    logging.info(f"Specular Point found at: {P_spec}")
    
    logging.info("Loading Field Planes...")
    fields_t, x1, yy, zz, dt, Nt = load_gprmax_plane(run.plane_file, run.comps)
    logging.info(f"Loaded plane at x1={x1:.3f} m: comps={list(fields_t.keys())}, dt={dt*1e9:.3f} ns, Nt={Nt}, ny={yy.size}, nz={zz.size}")

    visualize_setup(geom, srcpar, run, elempos, x1, yy, zz)

    eps = EPS0 * media.epsr2
    mu  = MU0
    sigma = media.sigma_rock
    v = 1.0 / np.sqrt(mu * eps)
    c_med = v

    denom = elempos[0,:] - xs
    valid = np.abs(denom) > 1e-9
    el_v = elempos[:, valid]
    lam = (x1 - xs) / (el_v[0,:] - xs)
    y1m = ys + lam * (el_v[1,:] - ys)
    z1m = zs + lam * (el_v[2,:] - zs)
    
    R2 = np.sqrt((el_v[0,:] - xs)**2 + (el_v[1,:] - ys)**2 + (el_v[2,:] - zs)**2)
    R1 = np.sqrt((x1 - xs)**2 + (y1m - ys)**2 + (z1m - zs)**2)
    dR_vals = R2 - R1
    
    dR_min = float(np.min(dR_vals))
    dR_max = float(np.max(dR_vals))

    Epos, E_time_padded, omega_pos, Npre, Nt_out, Nfft = build_padded_plane_spectrum(fields_t, dt, dR_min, dR_max, c_med)
    gamma_pos = gamma_of_omega(omega_pos, mu, eps, sigma)

    cutoff_high = 8.0 * FC
    freqs = omega_pos / (2*np.pi)
    valid_mask_freqs = freqs <= cutoff_high

    omega_pos_small = omega_pos[valid_mask_freqs]
    gamma_pos_small = gamma_pos[valid_mask_freqs]

    Epos_small = {}
    for comp, data in Epos.items():
        Epos_small[comp] = data[:, :, valid_mask_freqs].astype(np.complex64)

    logging.info(f"Original Freqs: {len(omega_pos)} points (up to {omega_pos[-1]/2/np.pi/1e9:.1f} GHz), Nfft={Nfft}")
    logging.info(f"Reduced Freqs:  {len(omega_pos_small)} points (up to {cutoff_high/1e6:.1f} MHz)")

    tapx, tapy = build_taper(run.critdist, geom.ellength, run.tapw)

    params = WorkerParams(
        Epos=Epos_small, E_time=E_time_padded, yy=yy, zz=zz, x1=x1,
        omega_pos=omega_pos_small, gamma_pos=gamma_pos_small, dt=dt,
        Npre=Npre, Nt_out=Nt_out, Nfft=Nfft, v=v,
        epsr2=media.epsr2, eps0_local=EPS0,
        srcx=xs, srcy=ys, srcz=zs,
        recx=xr, recy=yr, recz=zr,
        spec_x=P_spec[0], spec_y=P_spec[1], spec_z=P_spec[2],
        ellength=geom.ellength, tapx=tapx, tapy=tapy, critdist=run.critdist,
        doradpat=0, pws_interpor=None
    )

    logging.info("Starting parallel reaction integral computation...")
    if HAVE_JOBLIB and run.cores != 1:
        results = Parallel(n_jobs=run.cores, verbose=0)(
            delayed(element_contribution)(iel, elempos, params, media) for iel in range(numelem)
        )
    else:
        results = [element_contribution(iel, elempos, params, media) for iel in range(numelem)]

    Erec_raw_small = np.sum(np.array(results), axis=0)  
    logging.debug(f"Max raw spectrum amplitude: {np.max(np.abs(Erec_raw_small))}")

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
    f_sigma = 5.0 * FC
    gaussian_window = np.exp(-0.5 * (freqs_small / f_sigma)**2)
    deconvolved_pos *= gaussian_window
    
    deconvolved_full = np.zeros(Nfft // 2 + 1, dtype=complex)
    deconvolved_full[valid_mask_freqs] = deconvolved_pos
    E_time_full = np.fft.irfft(deconvolved_full, n=Nfft)
    trace = E_time_full[:Nt_out].real
    tvec = (np.arange(Nt_out) - Npre) * dt

    logging.info(f"Shape of Erec_raw_small: {Erec_raw_small.shape}")
    logging.info(f"Size of trace={trace.size}, tvec[0]={tvec[0]*1e9:.3f} ns, tvec[-1]={tvec[-1]*1e9:.3f} ns")
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
    # --- Standardize Environment & Logging ---
    image_dir = "./images"
    log_dir = "./logs"
    data_dir = "./npz_data"
    for d in [image_dir, log_dir, data_dir]:
        os.makedirs(d, exist_ok=True)

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
        handlers=[logging.FileHandler(os.path.join(log_dir, "ra_hpo.log")), logging.StreamHandler(sys.stdout)]
    )

    if not HAVE_JOBLIB:
        logging.warning("Joblib not found. Running in strictly serial mode.")

    # --- Setup Arguments ---
    parser = argparse.ArgumentParser(description="RA-HPO: Ray-Aligned Hybrid Physical Optics Method")
    parser.add_argument("--plane_file", type=str, default="3D_8x10x10-shifted_nofrac_planex3_h0pt01.out",
                        help="Path to the gprMax reference plane HDF5 file.")
    parser.add_argument("--h5_geom", type=str, default="/home/nfs/jchen18/chandler/BH_gprMax/shifted_domain8x10x10_fracs/input_h5_geom/geom_OnHeter_sig10_highVar.h5",
                        help="Path to geometry HDF5 file.")
    parser.add_argument("--syz_geom", type=str, default="/home/nfs/jchen18/chandler/BH_gprMax/shifted_domain8x10x10_fracs/input_h5_geom/Syz_OnHeter_sig10_highVar.h5",
                        help="Path to stochastic perturbation HDF5 file.")
    parser.add_argument("--mat_file", type=str, default="/home/nfs/jchen18/chandler/BH_gprMax/shifted_domain8x10x10_fracs/input_h5_geom/mat_OnHeter_sig10_highVar.txt",
                        help="Path to the materials text file.")
    parser.add_argument("--cores", type=int, default=24, help="Number of CPU cores for parallel execution.")
    parser.add_argument("--output_name", type=str, default="RA_HPO_BH_trace", help="Base name for the output files.")
    
    args = parser.parse_args()

    # --- Initialize Parameters ---
    media = MediaParams()
    srcpar = SourceParams()
    
    geom = GeometryParams(
        h5_path=args.h5_geom,
        syz_path=args.syz_geom,
        mat_path=args.mat_file
    )
    
    run = RuntimeParams(
        plane_file=args.plane_file,
        cores=args.cores
    )

    # --- Run Kernel ---
    try:
        tvec, trace = run_simulation(media, srcpar, geom, run)
        
        # --- Handle Output ---
        base_filename = args.output_name
        save_trace(os.path.join(data_dir, f"{base_filename}.npz"), tvec, trace)
        
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