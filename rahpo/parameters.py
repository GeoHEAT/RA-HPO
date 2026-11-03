"""Run configuration.

One trace is fixed by four dataclasses. ``MediaParams`` is the host and the
fracture fill, ``SourceParams`` the antenna positions, ``GeometryParams`` the
meshed fracture, and ``RuntimeParams`` the reference planes and the run.
``WorkerParams`` is the payload sent to each element of the reaction integral.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Optional, Tuple

import numpy as np

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

