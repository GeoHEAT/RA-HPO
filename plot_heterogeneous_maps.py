"""
Porosity, permittivity and conductivity maps of the stochastically heterogeneous fractures.

Reads the gprMax geometry (geom_<model>.h5) and materials (mat_<model>.txt) files written
by generate_heter_geometry_h5_and_materials.py, maps the material IDs back to physical
properties, and plots each property on the fracture plane.

Style matches plot_surface.py and plot_traces.py: 8 pt Arial, no title,
axis labels with units in parentheses, vector PDF (the dense surface itself is rasterized
inside the PDF). Two size presets set the print scale:
  "side": panel next to a trace (about 0.40 of the text width)
  "row" : one of three panels in a row (about 0.32 of the text width)
"""
import os
from pathlib import Path

import h5py
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib import cm
from matplotlib import colors
from matplotlib import font_manager

# Figure size (inches, before tight cropping) for each print width
PANEL_SIZES = {"side": (2.76, 2.45), "row": (2.12, 1.9)}
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
        "grid.linewidth": 0.4,
        "pdf.fonttype": 42,
        "savefig.dpi": 300,
    })


def reverse_map_properties(hdf5_filename, materials_filename):
    """
    Reads a gprMax HDF5 grid and reverse-maps integer IDs back to physical properties.
    """
    print(f"  Loading HDF5 data: {hdf5_filename}...")
    with h5py.File(hdf5_filename, 'r') as f:
        dx, dy, dz = f.attrs['dx_dy_dz']
        data_ids = np.array(f['data'], dtype=np.int16)

    nx, ny, nz = data_ids.shape

    material_map = {}
    with open(materials_filename, 'r') as f:
        for line in f:
            if line.startswith("#material"):
                parts = line.split()
                epsr = float(parts[1])
                sigma = float(parts[2])
                mat_id = int(parts[5].replace('frc_mat', ''))
                material_map[mat_id] = (epsr, sigma)

    epsr_3d = np.full((nx, ny, nz), -1.0, dtype=np.float32)
    sigma_3d = np.full((nx, ny, nz), -1.0, dtype=np.float32)

    fracture_mask = data_ids >= 0
    unique_ids = np.unique(data_ids[fracture_mask])

    for mat_id in unique_ids:
        if mat_id in material_map:
            epsr, sigma = material_map[mat_id]
            epsr_3d[data_ids == mat_id] = epsr
            sigma_3d[data_ids == mat_id] = sigma

    # These MUST match the deep geothermal brine used in generation
    epsr_matrix = 9.0
    sigma_matrix = 1e-4
    epsr_fluid = 43.0       # representative, water well above ambient temperature
    sigma_fluid = 10.0      # representative saline brine; no salinity implied
    freq = 100e6
    omega = 2.0 * np.pi * freq
    eps0 = 8.8541878128e-12

    eps_m_star = epsr_matrix - 1j * (sigma_matrix / (omega * eps0))
    eps_f_star = epsr_fluid - 1j * (sigma_fluid / (omega * eps0))
    sqrt_m = np.sqrt(eps_m_star)
    sqrt_f = np.sqrt(eps_f_star)

    eps_eff_star_3d = epsr_3d - 1j * (sigma_3d / (omega * eps0))

    phi_3d = np.full((nx, ny, nz), -1.0, dtype=np.complex64)
    phi_3d[fracture_mask] = (np.sqrt(eps_eff_star_3d[fracture_mask]) - sqrt_m) / (sqrt_f - sqrt_m)
    phi_3d = np.real(phi_3d).astype(np.float32)

    x = np.arange(nx) * dx
    y = np.arange(ny) * dy
    z = np.arange(nz) * dz

    return x, y, z, phi_3d, epsr_3d, sigma_3d


def plot_fracture_property_v2(X_surf, Y_mesh, Z_mesh, Values, color_label, cmap, filename,
                              size="side"):
    """3D surface plot of one property on the fracture plane, robust to zero-variance fields."""
    fig = plt.figure(figsize=PANEL_SIZES[size])
    ax = fig.add_subplot(111, projection='3d')

    # Visual downsampling stride: avoids aliasing artifacts on dense 800 x 800 grids
    stride = 4
    X_s = X_surf[::stride, ::stride]
    Y_s = Y_mesh[::stride, ::stride]
    Z_s = Z_mesh[::stride, ::stride]
    Val_s = Values[::stride, ::stride]

    v_min = np.nanmin(Values)
    v_max = np.nanmax(Values)
    v_mean = np.nanmean(Values)

    if np.isclose(v_min, v_max, atol=1e-4):
        delta = max(abs(v_mean) * 0.05, 0.05)
        v_min = v_mean - delta
        v_max = v_mean + delta

    norm = colors.TwoSlopeNorm(vmin=v_min, vcenter=v_mean, vmax=v_max)
    fcolors = cmap(norm(Val_s))
    ax.plot_surface(X_s, Y_s, Z_s, facecolors=fcolors, linewidth=0, antialiased=True,
                    shade=False, rasterized=True)

    ax.set_xlim(4, 8)
    ax.set_xticks([4, 6, 8])
    ax.set_ylim(np.min(Y_mesh), np.max(Y_mesh))
    ax.set_zlim(np.min(Z_mesh), np.max(Z_mesh))
    ax.locator_params(axis='y', nbins=3)
    ax.locator_params(axis='z', nbins=3)
    ax.tick_params(axis='both', which='major', pad=-2)
    ax.set_xlabel('x (m)', labelpad=-6)
    ax.set_ylabel('y (m)', labelpad=-5)
    ax.set_zlabel('z (m)', labelpad=-6)
    ax.set_box_aspect((1, 2, 2))
    ax.view_init(elev=20, azim=-45)

    m = cm.ScalarMappable(cmap=cmap, norm=norm)
    m.set_array(Values)
    cbar = fig.colorbar(m, ax=ax, shrink=0.55, aspect=14, pad=0.12)
    cbar.set_label(color_label)
    cbar.ax.tick_params(labelsize=FONT_SIZE - 1, width=0.6, length=2)
    cbar.outline.set_linewidth(0.6)

    # The tight bounding box of a 3D axes omits its axis labels, so pass them explicitly;
    # otherwise the x and y labels below the axes box are cropped away
    fig.savefig(filename, bbox_inches="tight", pad_inches=0.02,
                bbox_extra_artists=[ax.xaxis.label, ax.yaxis.label, ax.zaxis.label])
    plt.close(fig)


def generate_standard_images(model_name, input_dir=".", output_dir="."):
    """Processes a single model configuration and generates its 3 plots."""
    hdf5_file = os.path.join(input_dir, f'geom_{model_name}.h5')
    materials_file = os.path.join(input_dir, f'mat_{model_name}.txt')

    # Check if files exist to avoid crashing the batch loop
    if not os.path.exists(hdf5_file) or not os.path.exists(materials_file):
        print(f"  [ERROR] Files for {model_name} not found. Skipping.")
        return

    x, y, z, phi_3d, epsr_3d, sigma_3d = reverse_map_properties(hdf5_file, materials_file)

    Y, Z = np.meshgrid(y, z, indexing='ij')

    phi_2d = np.max(phi_3d, axis=0)
    epsr_2d = np.max(epsr_3d, axis=0)
    sigma_2d = np.max(sigma_3d, axis=0)

    x0 = 6.5
    # X_surf must be a 2D array matching Y and Z dimensions for plot_surface
    X_surf = np.full_like(Y, x0)

    fracture_mask_2d = phi_2d >= 0

    phi_2d[~fracture_mask_2d] = np.nan
    epsr_2d[~fracture_mask_2d] = np.nan
    sigma_2d[~fracture_mask_2d] = np.nan

    # 1: Porosity (three panels in a row)
    plot_fracture_property_v2(
        X_surf, Y, Z, phi_2d,
        color_label=r"Porosity $\phi$",
        cmap=cm.coolwarm,
        filename=os.path.join(output_dir, f"fig_porosity_{model_name}.pdf"),
        size="row",
    )

    # 2: Permittivity at 100 MHz (panel next to the trace)
    plot_fracture_property_v2(
        X_surf, Y, Z, epsr_2d,
        color_label=r"Permittivity $\varepsilon_r'$",
        cmap=cm.Spectral,
        filename=os.path.join(output_dir, f"fig_permittivity_{model_name}.pdf"),
        size="side",
    )

    # 3: Conductivity at 100 MHz
    plot_fracture_property_v2(
        X_surf, Y, Z, sigma_2d,
        color_label=r"Conductivity $\sigma_{\mathrm{eff}}$ (S/m)",
        cmap=cm.Spectral,
        filename=os.path.join(output_dir, f"fig_conductivity_{model_name}.pdf"),
        size="side",
    )
    print(f"  Successfully saved figures for {model_name}.")


if __name__ == "__main__":
    from cases import FIGURE_DIR, GEOMETRY

    set_style()
    os.makedirs(FIGURE_DIR, exist_ok=True)
    for model in ("ExpHeter_Lc02_highVar", "ExpHeter_Lc1_highVar", "ExpHeter_Lc4_highVar"):
        print(f"\nProcessing: {model}")
        generate_standard_images(model, input_dir=str(GEOMETRY), output_dir=str(FIGURE_DIR))
