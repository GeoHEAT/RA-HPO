"""
Heterogeneous fracture fills for the full-domain gprMax models and for RA-HPO.

A porosity field with an exponential covariance is synthesized on the fracture plane by FFT
spectral synthesis (fixed seed), mapped to permittivity and conductivity by the CRIM mixing
law between the host rock and a geothermal brine, and quantized into a set of materials. For
each model three files are written:

    geom_<name>.h5   gprMax geometry volume (#geometry_objects_read)
    mat_<name>.txt   its materials file
    Syz_<name>.h5    the fill on the fracture plane, read by ra_hpo.py --heterogeneous

The models of the paper have correlation lengths of 0.2, 1.0 and 4.0 m and porosity from 0.1
to 0.9.

Usage
-----
    python generate_heter_geometry_h5_and_materials.py --out-dir gprMax_models/For_full-domain_verifications/input_h5_geom
"""
import argparse
import os

import numpy as np
import h5py
from scipy.ndimage import gaussian_filter
from scipy.interpolate import RegularGridInterpolator
from scipy.stats import norm

class PetrophysicalModel:
    """Handles the physical mixing models for the subsurface."""
    # Default fluid end-member: a saline brine at geothermal temperature. The values are
    # representative, not derived from a stated temperature or salinity: eps_r = 43.0 is
    # liquid water well above ambient (Archer & Wang, 1990), and sigma = 10 S/m lies within
    # the brine conductivities of Ucok et al. (1980), 22-375 C and 3-26 wt% NaCl.
    def __init__(self, epsr_matrix=9.0, sigma_matrix=1e-4, 
                 epsr_fluid=43.0, sigma_fluid=10.0, freq=100e6):
        self.omega = 2.0 * np.pi * freq
        self.eps0 = 8.8541878128e-12
        
        # Precompute complex properties of end-members
        self.eps_m_star = epsr_matrix - 1j * (sigma_matrix / (self.omega * self.eps0))
        self.eps_f_star = epsr_fluid - 1j * (sigma_fluid / (self.omega * self.eps0))

    def calculate_crim(self, porosity: float) -> tuple:
        """Calculates effective permittivity and conductivity using the CRIM equation."""
        sqrt_eff = (1.0 - porosity) * np.sqrt(self.eps_m_star) + porosity * np.sqrt(self.eps_f_star)
        eps_eff_star = sqrt_eff**2
        
        epsr_eff = np.real(eps_eff_star)
        sigma_eff = -np.imag(eps_eff_star) * self.omega * self.eps0
        return epsr_eff, sigma_eff

def generate_oversized_field(shape: tuple, sigma: float = 4.0, seed: int = 66) -> tuple:
    """Generates an oversized field, decoupled into filter padding and pan margin, by FFT
    spectral synthesis of an exponential covariance model."""
    np.random.seed(seed)
    
    # Correlation length in pixels (Lc = 2*sigma)
    L_c_pixels = 2.0 * sigma
    
    # 1. Padding to completely absorb FFT periodic boundary wrap-around artifacts
    # For an exponential field, 4 correlation lengths ensures safe decorrelation at boundaries
    filter_pad = int(4.0 * L_c_pixels)
    # 2. Panning margin to guarantee we can slide the window to find a 0.0 crossing
    pan_margin = 0 #int(max(20, sigma))
    
    total_pad = filter_pad + pan_margin
    padded_shape = (shape[0] + 2 * total_pad, shape[1] + 2 * total_pad)
    
    # Generate Gaussian white noise
    noise = np.random.normal(loc=0.0, scale=1.0, size=padded_shape)
    
    # FFT Spectral Synthesis for Exponential Covariance
    noise_fft = np.fft.fft2(noise)
    
    # Create normalized frequency grids
    kx = np.fft.fftfreq(padded_shape[0])
    ky = np.fft.fftfreq(padded_shape[1])
    KX, KY = np.meshgrid(kx, ky, indexing='ij')
    K_sq = KX**2 + KY**2
    
    # Power Spectral Density (PSD) of a 2D Exponential Covariance function
    # S(k) is proportional to (1 + (2*pi*k*L_c)^2)^(-3/2)
    psd = (1.0 + (2.0 * np.pi * L_c_pixels)**2 * K_sq)**(-1.5)
    
    # The required Fourier domain filter is the square root of the PSD
    filter_response = np.sqrt(psd)
    
    # Apply filter and transform back to spatial domain
    field_fft = noise_fft * filter_response
    large_field = np.real(np.fft.ifft2(field_fft))
    
    # Normalize to mean 0.0 and std 1.0 to ensure safe CDF mapping downstream
    large_field = (large_field - np.mean(large_field)) / np.std(large_field)
    
    # Return the field, the total padding, and the specific margin we are allowed to search in
    return large_field, total_pad, pan_margin

def compute_distance_field(X: np.ndarray, Y: np.ndarray, Z: np.ndarray, 
                           fracture_type: str, nx: int, ny: int, nz: int, h: float) -> np.ndarray:
    """Calculates the orthogonal distance from various defined geometries."""
    if fracture_type == 'planar':
        # Planar Fracture: Define by equation ax + by + cz = d 
        a, b, c, d = 1.0, 0.0, 0.0, 6.5
        # 8 deg: a, b, c, d = -7.11537, 0.0, -1.0, -50.25
        norm = np.sqrt(a**2 + b**2 + c**2)
        return np.abs((a/norm)*X + (b/norm)*Y + (c/norm)*Z - (d/norm))
        
    elif fracture_type == 'cylindrical':
        x0, y0 = 0.5, 0.5
        radius = 3.5 / 1.414
        return np.abs(np.sqrt((X - x0)**2 + (Y - y0)**2) - radius)
        
    elif fracture_type == 'spherical_shell':
        x0, y0, z0 = nx*h/2, ny*h/2, nz*h/2
        radius = 2.0
        return np.abs(np.sqrt((X - x0)**2 + (Y - y0)**2 + (Z - z0)**2) - radius)
        
    elif fracture_type == 'saddle':
        x0, y0, z0 = nx*h/2, ny*h/2, nz*h/2
        a, b = 2.0, 2.0
        Z_surf = a * (X - x0)**2 - b * (Y - y0)**2 + z0
        dZ_dX, dZ_dY = 2 * a * (X - x0), -2 * b * (Y - y0)
        norm = np.sqrt(1 + dZ_dX**2 + dZ_dY**2)
        return np.abs(Z - Z_surf) / norm
        
    elif fracture_type == 'sinusoidal_3d':
        A = 0.25
        kz = 2 * np.pi / (nz * h * 2 / 3)
        ky = 2 * np.pi / (ny * h / 2)
        x0, slope_z, slope_y = 6.5, 0.0, 0.0
        
        X_surf = A * np.sin(kz * Z) * np.sin(ky * Y) + slope_z * Z + slope_y * Y + x0
        dX_dZ = A * kz * np.cos(kz * Z) * np.sin(ky * Y) + slope_z
        dX_dY = A * ky * np.sin(kz * Z) * np.cos(ky * Y) + slope_y
        norm = np.sqrt(1 + dX_dZ**2 + dX_dY**2)
        return np.abs(X - X_surf) / norm
        
    elif fracture_type == 'piecewise_yz':
        z0, y0 = nz*h/2, ny*h/2
        inner_radius, outer_radius = 1.5, 1.5
        R = np.sqrt((Z - z0)**2 + (Y - y0)**2)
        dist = np.full_like(Z, np.inf)
        
        mask1 = R <= inner_radius
        x_flat = 6.5
        dist[mask1] = np.abs(X[mask1] - x_flat)
        
        mask2 = (R > inner_radius) & (R <= outer_radius)
        t = (R[mask2] - inner_radius) / (outer_radius - inner_radius)
        x_transition = x_flat * (1 - t) + (x_flat - 0.1 * (R[mask2] - inner_radius)) * t
        dist[mask2] = np.abs(X[mask2] - x_transition)
        
        mask3 = R > outer_radius
        curvature = -3.0 / 18.0
        x_curved = (x_flat - 0.1 * (outer_radius - inner_radius)) + curvature * (R[mask3] - outer_radius)**2
        dx_dr = 2 * curvature * (R[mask3] - outer_radius)
        norm3 = np.sqrt(1 + dx_dr**2)
        dist[mask3] = np.abs(X[mask3] - x_curved) / norm3
        return dist
        
    else:
        raise ValueError(f"Unknown fracture_type: {fracture_type}")

def compute_manifold_uv(X_surf: np.ndarray, dy: float, dz: float) -> tuple:
    """Computes the 2D unwrapped arc-length coordinates (U, V) by integrating the surface gradient."""
    dX_dY = np.gradient(X_surf, dy, axis=0)
    dX_dZ = np.gradient(X_surf, dz, axis=1)
    
    ds_y = np.sqrt(1.0 + dX_dY**2) * dy
    ds_z = np.sqrt(1.0 + dX_dZ**2) * dz
    
    # Cumulative numerical integration
    U = np.cumsum(ds_y, axis=0) - ds_y[0:1, :] / 2.0
    V = np.cumsum(ds_z, axis=1) - ds_z[:, 0:1] / 2.0
    
    # Anchor to 0,0
    return U - U[0:1, :], V - V[:, 0:1]

def get_fracture_surface_2d(Y, Z, fracture_type, h):
    """Extracts the X elevation of the fracture to compute arc-length."""
    if fracture_type == 'planar':
        return np.full_like(Y, 6.5)
    elif fracture_type == 'sinusoidal_3d':
        A, x0 = 0.25, 6.5
        nz_phys, ny_phys = Z.shape[1], Y.shape[0]
        kz = 2 * np.pi / (nz_phys * h * 2 / 3)
        ky = 2 * np.pi / (ny_phys * h / 2)
        return A * np.sin(kz * Z) * np.sin(ky * Y) + x0
    elif fracture_type == 'piecewise_yz':
        nz_phys, ny_phys = Z.shape[1], Y.shape[0]
        z0, y0 = nz_phys*h/2, ny_phys*h/2
        inner_radius, outer_radius = 1.5, 1.5
        R = np.sqrt((Z - z0)**2 + (Y - y0)**2)
        X_surf = np.full_like(Y, 6.5)
        # Simplified piecewise assignment (mirror your main logic here)
        mask3 = R > outer_radius
        curvature = -3.0 / 18.0
        X_surf[mask3] = 6.5 + curvature * (R[mask3] - outer_radius)**2
        return X_surf
    else:
        return np.full_like(Y, 6.5)


def write_materials_file(filename: str, petro_model: PetrophysicalModel, 
                         num_bins: int, phi_min: float, phi_max: float) -> None:
    """Writes the gprMax discrete materials mapping file."""
    with open(filename, 'w') as f_out:
        for i in range(num_bins):
            # If phi_min == phi_max (Baseline), this safely evaluates to a constant phi
            phi = phi_min + (phi_max - phi_min) * (i / (max(1, num_bins - 1)))
            epsr, sigma = petro_model.calculate_crim(phi)
            f_out.write(f"#material: {epsr:.3f} {sigma:.4f} 1 0 frc_mat{i}\n")

def build_fracture_model(model_name: str, fracture_type: str, sigma_f: float, phi_min: float, phi_max: float, align_specular: bool = False):
    """Main orchestrator for building the gprMax HDF5 and materials file."""
    # 1. Grid Parameters
    h = 0.01
    nx, ny, nz = 800, 800, 800
    n_thickness = 4
    base_thickness = n_thickness * h
    num_materials = 500
    bg_id = -1
    
    print(f"\n--- Generating {model_name} ---")
    x = (np.arange(nx) + 0.5) * h
    y = (np.arange(ny) + 0.5) * h
    z = (np.arange(nz) + 0.5) * h
    X, Y, Z = np.meshgrid(x, y, z, indexing='ij')

    # Geometry & Physics calculation
    dist = compute_distance_field(X, Y, Z, fracture_type, nx, ny, nz, h)
    #==================================================

    # Extract 2D grid
    Y2, Z2 = np.meshgrid(y, z, indexing='ij')
    
    # Get Surface and Arc-Lengths
    X_surf_2d = get_fracture_surface_2d(Y2, Z2, fracture_type, h)
    U_2d, V_2d = compute_manifold_uv(X_surf_2d, h, h)
    
    # Define the unwrapped domain and generate the rigorous field
    u_max, v_max = np.max(U_2d), np.max(V_2d)
    nu = int(np.ceil(u_max / h)) + 1
    nv = int(np.ceil(v_max / h)) + 1
    u_vec = np.arange(nu) * h
    v_vec = np.arange(nv) * h

    # If using the unwrapped U, V logic, you must interpolate these global 
    # Y,Z coordinates to find the corresponding local u_spec, v_spec
    y_spec_local = ny * h / 2
    z_spec_local = nz * h / 2
    interp_Y_to_U = RegularGridInterpolator((y, z), U_2d, bounds_error=False, fill_value=None)
    interp_Z_to_V = RegularGridInterpolator((y, z), V_2d, bounds_error=False, fill_value=None)
    u_spec = interp_Y_to_U(np.array([[y_spec_local, z_spec_local]]))[0]
    v_spec = interp_Z_to_V(np.array([[y_spec_local, z_spec_local]]))[0]

    idx_u_spec = np.argmin(np.abs(u_vec - u_spec))
    idx_v_spec = np.argmin(np.abs(v_vec - v_spec))
    
    large_field, total_pad, pan_margin = generate_oversized_field((nu, nv), sigma=sigma_f, seed=43)
    
    f_mean = np.mean(large_field)
    f_std = np.std(large_field)
    std_large = (large_field - f_mean) / f_std if f_std > 1e-10 else large_field - f_mean
    
    # 2. Define search zone strictly using pan_margin (Protects the 4*sigma filter_pad)
    center_u = idx_u_spec + total_pad
    center_v = idx_v_spec + total_pad
    
    if align_specular:
        valid_zone = std_large[center_u - pan_margin : center_u + pan_margin,
                               center_v - pan_margin : center_v + pan_margin]
        
        # 3. Find the value naturally closest to 0.0 in this safe zone
        idx_zero_flat = np.argmin(np.abs(valid_zone))
        local_u_zero, local_v_zero = np.unravel_index(idx_zero_flat, valid_zone.shape)
        
        # 4. Map back to global coordinates within std_large
        i_zero = (center_u - pan_margin) + local_u_zero
        j_zero = (center_v - pan_margin) + local_v_zero
    else:
        # Bypass alignment: strictly use the geometric center of the pad
        i_zero = center_u
        j_zero = center_v
    
    # 5. Calculate crop boundaries
    u_start = i_zero - idx_u_spec
    v_start = j_zero - idx_v_spec
    
    # 6. Extract the final perfectly continuous, edge-artifact-free field!
    std_field_cropped = std_large[u_start : u_start + nu, v_start : v_start + nv]

    # Map through CDF
    uniform_field = norm.cdf(std_field_cropped)
    S_uv = 2.0 * uniform_field - 1.0
    
    # Interpolate the (u,v) field back to the physical (y,z) coordinate grid
    interp_uv = RegularGridInterpolator((u_vec, v_vec), S_uv, bounds_error=False, fill_value=0.0)
    U_2d_clipped = np.clip(U_2d, u_vec[0], u_vec[-1])
    V_2d_clipped = np.clip(V_2d, v_vec[0], v_vec[-1])
    points_uv = np.stack((U_2d_clipped.ravel(), V_2d_clipped.ravel()), axis=-1)
    S_yz = interp_uv(points_uv).reshape((ny, nz))
    
    # Broadcast along X
    stochastic_field = np.broadcast_to(S_yz[np.newaxis, :, :], (nx, ny, nz))
    #==================================================
    
    # Apply Thresholding and Discretization
    local_thickness = base_thickness * (1.0 + 0.0 * stochastic_field)
    is_fracture = dist <= (local_thickness / 2.0)
    
    # Map stochastic field to integer material bins
    material_indices = np.round((stochastic_field + 1.0) / 2.0 * (num_materials - 1)).astype(np.int16)
    material_indices = np.clip(material_indices, 0, num_materials - 1)
    
    # Initialize background and apply fracture materials
    data = np.full((nx, ny, nz), bg_id, dtype=np.int16)
    data[is_fracture] = material_indices[is_fracture]
    
    total_frac_cells = np.sum(is_fracture)
    
    # Output Generation
    petro = PetrophysicalModel()
    materials_filename = f'mat_{model_name}.txt'
    write_materials_file(materials_filename, petro, num_materials, phi_min, phi_max)
    
    hdf5file_Syz = f'Syz_{model_name}.h5'
    with h5py.File(hdf5file_Syz, 'w') as fout_Syz:
        fout_Syz.attrs['dx_dy_dz'] = (h, h, h)
        fout_Syz.create_dataset('S_yz', data=S_yz)

    hdf5file = f'geom_{model_name}.h5'
    with h5py.File(hdf5file, 'w') as fout:
        fout.attrs['dx_dy_dz'] = (h, h, h)
        fout.create_dataset('data', data=data)
        
    print(f"Written: {hdf5file} and {materials_filename}")

if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="Heterogeneous fracture fills (geometry volume, "
                                             "materials file and fracture-plane map).")
    ap.add_argument("--out-dir", default=".", help="output directory (default: current)")
    out_dir = ap.parse_args().out_dir
    os.makedirs(out_dir, exist_ok=True)
    os.chdir(out_dir)

    # sigma sets the correlation length (L_c = 2 sigma cells of 0.01 m); phi the porosity range
    simulation_matrix = [
        {"name": "ExpHeter_Lc02_highVar", "sigma": 10.0,  "phi": [0.1, 0.9]},
        {"name": "ExpHeter_Lc1_highVar",  "sigma": 50.0,  "phi": [0.1, 0.9]},
        {"name": "ExpHeter_Lc4_highVar",  "sigma": 200.0, "phi": [0.1, 0.9]},
    ]
    for model in simulation_matrix:
        build_fracture_model(
            model_name=model["name"],
            fracture_type="planar",
            sigma_f=model["sigma"],
            phi_min=model["phi"][0],
            phi_max=model["phi"][1],
            align_specular=False,
        )
    print(f"Written to {os.getcwd()}")
