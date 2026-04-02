# RA-HPO: Ray-Aligned Hybrid Physical Optics Method

[![Python Version](https://img.shields.io/badge/python-3.9%2B-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Project: GeoHEAT](https://img.shields.io/badge/Project-GeoHEAT-success)](https://www.geoheat.org/)

**RA-HPO** is an open-source, semi-analytical Python framework designed for modeling borehole radar responses in deep geothermal exploration. 

By coupling high-frequency asymptotic methods (Physical Optics) with the Electromagnetic Reciprocity Theorem, RA-HPO efficiently computes the response of the borehole radar accounting for effects of electromagnetic waves interacting with fractures and the near-borehole enviroment. This provides a highly computationally efficient alternative to full-domain brute-force FDTD simulations for large-scale far-field problems.

## 🚀 Key Features

* **Ray-Aligned Propagation:** Accurately propagates electric and magnetic fields from a near-field reference plane (e.g., extracted from `gprMax`) to arbitrary far-field locations.
* **Curved Fractures:** Supports complex 3D fracture geometries (planar, tilted, sinusoidal, and piecewise) via a robust meshed surface approach.
* **Stochastic Petrophysics Mapping:** Seamlessly maps exact 2D unwrapped arc-length coordinates from 3D FDTD HDF5 grids onto analytical meshes, preserving complex properties using the Complex Refractive Index Method (CRIM).
* **High-Performance:** Built-in parallel processing via `joblib` for rapid calculation of reaction integrals over large fracture surfaces.

## 🛠️ Installation

We recommend using the `conda` package manager to ensure all dependencies are correctly configured.

1. **Clone the repository:**
   ```bash
   git clone https://github.com/GeoHEAT/WP5T1_borehole_radar_forward_solver.git
   cd RA-HPO
   ```

2.  **Create the Conda environment:**

    ```bash
    conda env create -f environment.yml
    ```

3.  **Activate the environment:**

    ```bash
    conda activate rahpo-env
    ```

## 💻 Quick Start

*(Note: Ensure you have your `gprMax` HDF5 output files and geometry matrices placed in the correct input directories before running.)*

Here is an example of the input file for gprMax to obtain the reference-plane field:
```text
#title: A simplified BH environment
#num_threads: 128
#domain: 8.0 10.0 10.0
#dx_dy_dz: 0.01 0.01 0.01
#time_window: 90e-9

#material: 9.0 1e-4 1 0 rock
#material: 80.0 0.5 1 0 my_water

#waveform: ricker 100 100e6 my_ricker
#hertzian_dipole: z 0.5 5.0 5.5 my_ricker
#rx_array: 3.5 0.5 0.5 3.5 9.5 9.5 0.01 0.01 0.01

#box: 0 0 0 8.0 10.0 10.0 rock
#cylinder: 0.5 5.0 0 0.5 5.0 10.0 0.075 my_water
```


To run a standard simulation, execute the main Python script:

```bash
python RA-HPO.py 
```

Outputs, including the processed time-domain traces (`.npz`) and normalized amplitude plots (`.png`), will be saved to the `npz_data/` and `images/` directories, respectively.

## 📂 Repository Structure

  * `run_rahpo.py`: Main driver script containing the RA-HPO kernel and processing logic.
  * `environment.yml`: Conda dependency configuration.
  * `images/`: Output directory for generated trace plots and geometry visualizations.
  * `logs/`: Output directory for standard logging outputs.
  * `npz_data/`: Output directory for saved numerical traces.

## 🧠 Theory & Methodology

The core philosophy of RA-HPO is to split the massive computational domain of deep borehole radar exploration.

1.  **Near-field (gprMax):** FDTD is used to model the highly complex source-borehole-fluid interactions up to a reference plane (e.g., x = 3.5m).
2.  **Far-field (RA-HPO):** Fields are analytically propagated to a fracture surface (e.g., x \> 3.5m). The scattered field's coupling back into the receiver is approximated using the **Electromagnetic Reciprocity Theorem**, calculating the reaction integral over the fracture without needing to explicitly model the "re-entry" through the complex borehole environment.

