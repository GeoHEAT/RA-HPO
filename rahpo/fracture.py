"""Fracture surface, fill properties, and the surface mesh.

``fracture_surface_function`` is x = f(y, z). ``get_fracture_mesh`` samples it,
builds the normals, and attaches the fill. Nothing in this module propagates
a field.
"""
from __future__ import annotations

import logging
import os

import h5py
import numpy as np
from scipy.interpolate import RegularGridInterpolator
from scipy.stats import norm

from rahpo.constants import EPS0, FC
from rahpo.parameters import GeometryParams, MediaParams

def fracture_surface_function(y_grid: np.ndarray, z_grid: np.ndarray, geom: GeometryParams) -> np.ndarray:
    """Fracture surface ``x = f(y, z)`` in local coordinates.

    Parameters
    ----------
    y_grid, z_grid : np.ndarray
        Local coordinates, measured from the lower corner of the meshed extent.
    geom : GeometryParams
        ``fracture_type`` selects the surface: ``planar``, ``planar-tilted``
        (8 degrees), ``planar-dip`` (a plane at ``geom.dip_deg`` from the borehole
        axis through ``x_base`` at ``z_center``), ``piecewise_yz`` (concave bowl
        with a flat center), or ``sinusoidal_3d`` (biaxial corrugation of
        amplitude ``geom.amplitude``).

    Returns
    -------
    np.ndarray
        x coordinate of the surface, same shape as ``y_grid``.
    """
    # Local coordinates relative to center
    Y = y_grid
    Z = z_grid

    if geom.fracture_type == 'planar':
        x_surf = np.full_like(Y, geom.x_base) 
    elif geom.fracture_type == 'planar-tilted':
        x_surf = (Z - 50.25)/(-7.11537)
    elif geom.fracture_type == 'planar-dip':
        # x = x_base + (z - z_center) tan(dip) in absolute z; Z is local, z = Z + zmin
        x_surf = geom.x_base + (Z + geom.zmin - geom.z_center) * np.tan(np.radians(geom.dip_deg))
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


def get_fracture_mesh(geom: GeometryParams, media: MediaParams) -> np.ndarray:
    """Sample the fracture and return one column per element.

    Rows 0:3 are the position, 3:6 the unit normal, 6 the element weight,
    7 the aperture, 8 the fill permittivity, 9 the host permittivity, and
    10 the fill conductivity.
    """
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


def drop_elements_before_plane(elempos: np.ndarray, x_plane: float) -> tuple[np.ndarray, np.ndarray]:
    """Split off the elements at or in front of the reference plane x = x_plane.

    Such an element would need the plane field propagated backwards along the ray (R1 > R2),
    which the method does not model. Returns the elements behind the plane and the positions
    (rows 0 to 2) of those removed.
    """
    front = elempos[0, :] <= x_plane
    return elempos[:, ~front], elempos[0:3, front].copy()


def completeness_horizon(lost_pos: np.ndarray, src_pos, rec_pos, v: float) -> float | None:
    """Completeness horizon t_c in seconds, or None if no element is lost.

    lost_pos holds the positions (3, n) of the elements that contribute nothing, because they
    were dropped before the plane or their ray misses the recorded plane. t_c is the earliest
    two-way time source -> element -> receiver among them at speed v, plus the source delay
    sqrt(2)/fc that the trace time axis carries. The trace may lack contributions after t_c.
    """
    if lost_pos.shape[1] == 0:
        return None
    two_way = (np.linalg.norm(lost_pos - np.asarray(src_pos, float)[:, None], axis=0)
               + np.linalg.norm(lost_pos - np.asarray(rec_pos, float)[:, None], axis=0))
    return float(np.min(two_way)) / v + np.sqrt(2.0) / FC


def get_specular_point_numerical(src_pos, rec_pos, elempos: np.ndarray) -> tuple[np.ndarray, int]:
    """Element that minimises the transmitter-element-receiver path length.

    Returns
    -------
    tuple[np.ndarray, int]
        Position of that element and its column index in ``elempos``.
    """
    src = np.array(src_pos)[:, None] 
    rec = np.array(rec_pos)[:, None] 
    els = elempos[0:3, :]            
    
    dist_tx = np.sqrt(np.sum((els - src)**2, axis=0))
    dist_rx = np.sqrt(np.sum((els - rec)**2, axis=0))
    
    total_dist = dist_tx + dist_rx
    idx_min = np.argmin(total_dist)
    
    P_spec = els[:, idx_min]
    return P_spec, idx_min

