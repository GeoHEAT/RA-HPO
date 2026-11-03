"""Command line for one RA-HPO trace.

The flags and their defaults are the configuration used in the paper. Run
``python ra_hpo.py --help`` or ``python -m rahpo --help``.
"""
from __future__ import annotations

import argparse
import logging
import os
import sys

import numpy as np

from rahpo.parameters import GeometryParams, MediaParams, RuntimeParams, SourceParams
# simulation selects the non-interactive backend before pyplot is imported
from rahpo.simulation import HAVE_JOBLIB, run_simulation, save_trace
import matplotlib.pyplot as plt


def main(argv: list[str] | None = None) -> int:
    """Parse ``argv`` (default: the process arguments) and write one trace.

    Returns
    -------
    int
        0 when the trace is written, 1 when the run raises.
    """
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
                        "converted with tools/convert_plane.py.")
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

    args = parser.parse_args(argv)
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
        return 1

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
