"""
The cases of the paper: models, reference planes, solver settings and file locations.

This catalogue is not part of the solver. It lives next to the worked example so that
``rahpo`` stays the method, and the experiments of the paper stay in ``examples``.

Every script that runs or evaluates a case takes its configuration from here. A case writes
its RA-HPO trace as ``output/traces/<key>.npz``, and the analysis scripts read it under
that name.

gprMax input files stay in ``examples/gprMax_models/``; gprMax writes each ``.out`` next to
its input. Converted planes, traces, figures and tables go under ``output/`` (see
:mod:`rahpo.paths`). ``RAHPO_FDTD_DIR``, when set, is searched for the full-domain gprMax
outputs instead of the model directories.
"""
from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

from rahpo.paths import FIGURES, METRICS, PLANES, PROJECT_DIR, TRACES

REPO = PROJECT_DIR
MODELS = REPO / "examples" / "gprMax_models"
GEOMETRY = MODELS / "For_full-domain_verifications" / "input_h5_geom"
ANALYTIC = REPO / "examples" / "analytic"
PLANE_DIR = PLANES
TRACE_DIR = TRACES
FDTD_DIR = os.environ.get("RAHPO_FDTD_DIR")
FIGURE_DIR = FIGURES
METRICS_DIR = METRICS

TIME_RANGE = (100.0, 200.0)   # evaluation window of every comparison [ns]
PLANE_RES = 0.04              # receiver spacing at which the solver reads the planes [m]

# Reference-plane models (paths below examples/gprMax_models/, without extension)
_P = "For_referenced_plane"
PLANE_BH = f"{_P}/For_including_BH_only/3D_8x10x10-shifted_nofrac_planex3_h0pt01"
PLANE_NOBH = f"{_P}/For_infinite_withoutBH/3D_8x10x10-shifted_nobh_nofrac_planex3_h0pt01"
PLANE_ARMS_TX = f"{_P}/For_including_BH_and_antenna_arms/3D_8x10x10-shifted-noNearBH_nofrac_tx5pt5_planex3_h0pt01"
PLANE_ARMS_RX = f"{_P}/For_including_BH_and_antenna_arms/3D_8x10x10-shifted-noNearBH_nofrac_tx4pt5_planex3_h0pt01"
# The plane of PLANE_BH from a domain truncated at x = 4.0 m, recorded at 0.01 m and at 0.04 m
PLANE_TRUNCATED = f"{_P}/For_truncated_domain/3D_4x10x10-shifted_nofrac_planex3_h0pt01"
PLANE_TRUNCATED_RX04 = f"{_P}/For_truncated_domain/3D_4x10x10-shifted_nofrac_planex3_h0pt01_rx0pt04"


def plane_two_medium(epsr: int, side: str) -> str:
    z = {"tx": "5pt5", "rx": "4pt5"}[side]
    return (f"{_P}/For_including_BH_and_antenna_arms/"
            f"3D_8x10x10-shifted-antena_noNearFracs_tx{z}_planex3_espr{epsr}_h0pt01")


# Full-domain models
_F = "For_full-domain_verifications"
DIRECT = f"{_F}/3D8x10x10-shifted_nofrac_directwave_h0pt01"
DIRECT_ARMS = f"{_F}/3D8x10x10-shifted_witharms_nofrac_directwave_h0pt01"
FDTD_ARMS = f"{_F}/3D8x10x10-shifted-noNearBH_frac-dist6-4h_planar_h0pt01"
FDTD_PLANAR = f"{_F}/3D8x10x10-shifted_frac-dist6-4h_planar_h0pt01"
GATHER_FRAC = "For_gather/gather_frac"
GATHER_NOFRAC = "For_gather/gather_nofrac"
REFINED = "For_refined_reference/3D8x10x10-shifted_frac-dist6-4h_planar_h0pt005"
REFINED_DIRECT = "For_refined_reference/3D8x10x10-shifted_nofrac_directwave_h0pt005"
REFINED_BOX = "For_refined_reference/3D8x10x10-shifted_frac-dist6-4h_planar-box_h0pt01"

# Corrugation amplitudes of the roughness study: (file tag, A [m])
AMPLITUDES = [("00625", 0.0625), ("0125", 0.125), ("025", 0.25), ("03", 0.30), ("04", 0.40),
              ("05", 0.50), ("055", 0.55), ("06", 0.60), ("07", 0.70)]
# Receiver positions of the multi-offset gather, in the order of rx1..rx16 of the gather models
GATHER_Z = [1.5, 2.0, 2.5, 3.0, 3.5, 4.0, 4.5, 5.0, 6.0, 6.5, 7.0, 7.5, 8.0, 8.5, 9.0, 9.5]
Z_SOURCE = 5.5


@dataclass(frozen=True)
class Case:
    key: str
    label: str
    group: str                      # accuracy, roughness, two_medium, control, gather, ...
    plane_tx: str                   # reference-plane model (transmitter)
    flags: tuple = ()               # ra_hpo.py flags besides planes, output name and cores
    plane_rx: Optional[str] = None  # receiver plane model; None: same as plane_tx (axial shift)
    fdtd: Optional[str] = None      # full-domain model of the reference trace
    direct: Optional[str] = None    # its direct-wave model, subtracted; None: no subtraction
    figure: Optional[str] = None    # figure written by plot_traces.py
    analytic: Optional[str] = None  # exact trace in examples/analytic/ (infinite planar case)
    normalized: bool = False        # plot each trace normalized by its peak
    raw_plane: bool = False         # read the gprMax .out directly instead of a converted plane
    target_res: float = PLANE_RES


def _heterogeneous(lc_tag: str, lc: str) -> Case:
    name = f"ExpHeter_{lc_tag}_highVar"
    geom = GEOMETRY.relative_to(REPO)
    return Case(
        f"heter_{lc_tag}", f"Heterogeneous, Lc = {lc} m", "accuracy", PLANE_BH,
        ("--fracture_type", "planar", "--heterogeneous", "--fluid_epsr", "43.0",
         "--fluid_sigma", "10.0", "--h5_geom", f"{geom}/geom_{name}.h5",
         "--syz_geom", f"{geom}/Syz_{name}.h5", "--mat_file", f"{geom}/mat_{name}.txt"),
        fdtd=f"{_F}/{name}", direct=DIRECT,
        figure=f"Traces_heterogeneous_{lc_tag}")


def _corrugated(tag: str, a: float) -> Case:
    return Case(
        f"corrugated_A{tag}", f"Biaxial corrugated, A = {a:g} m", "roughness", PLANE_BH,
        ("--fracture_type", "sinusoidal_3d", "--amplitude", f"{a:g}"),
        fdtd=f"{_F}/3D8x10x10-shifted_frac-dist6-4h_A{tag}sin_h0pt01", direct=DIRECT,
        figure="Traces_corrugated_A025" if tag == "025" else None)


CASES: list[Case] = [
    # Single-trace validation (Table 2)
    Case("infinite_planar", "Infinite planar", "accuracy", PLANE_NOBH,
         ("--planethick", "0.02", "--fracture_extent", "-3", "13", "-3", "13",
          "--edge_taper", "3"),
         fdtd=f"{_F}/3D8x8x8-shifted_nobh_frac-dist6-2h_h0pt010", direct=None,
         figure="Traces_infinite_planar", analytic="vertical_plannar_2cm_analy_trace.csv",
         normalized=True),
    Case("finite_planar", "Finite planar", "accuracy", PLANE_BH, ("--fracture_type", "planar"),
         fdtd=FDTD_PLANAR, direct=DIRECT, figure="Traces_finite_planar"),
    Case("finite_planar_arms", "Finite planar, antenna arms", "accuracy", PLANE_ARMS_TX,
         ("--fracture_type", "planar"), plane_rx=PLANE_ARMS_RX,
         fdtd=FDTD_ARMS, direct=DIRECT_ARMS, figure="Traces_antenna_arms"),
    Case("inclined_8deg", "Inclined (8 deg)", "accuracy", PLANE_BH,
         ("--fracture_type", "planar-tilted"),
         fdtd=f"{_F}/3D8x10x10-shifted_frac-dist6-4h_planar-8deg_h0pt01", direct=DIRECT,
         figure="Traces_inclined_8deg"),
    Case("piecewise_concave", "Piecewise concave", "accuracy", PLANE_BH,
         ("--fracture_type", "piecewise_yz"),
         fdtd=f"{_F}/3D8x10x10-shifted_frac-dist6-4h_piecewiseyx_h0pt01", direct=DIRECT,
         figure="Traces_piecewise_concave"),
    _heterogeneous("Lc02", "0.2"),
    _heterogeneous("Lc1", "1.0"),
    _heterogeneous("Lc4", "4.0"),
]
# Roughness study (Table 3); A = 0.25 m is also the corrugated row of Table 2
CASES += [_corrugated(tag, a) for tag, a in AMPLITUDES]
# Two-medium propagation: planes recorded in eps_r = 6 and 12, re-radiated into eps_r = 9
CASES += [Case(f"two_medium_eps{e}", f"Plane recorded in eps_r = {e}", "two_medium",
               plane_two_medium(e, "tx"), ("--fracture_type", "planar", "--plane_epsr", f"{e:.1f}"),
               plane_rx=plane_two_medium(e, "rx"), fdtd=FDTD_ARMS, direct=DIRECT_ARMS)
          for e in (6, 12)]
# Control: the finite planar case from a plane recorded without the borehole
CASES += [Case("no_borehole_plane", "Finite planar, plane recorded without the borehole",
               "control", PLANE_NOBH, ("--fracture_type", "planar"),
               fdtd=FDTD_PLANAR, direct=DIRECT)]
# Multi-offset gather: sixteen receivers on the borehole axis, one plane
CASES += [Case(f"gather_z{str(z).replace('.', 'pt')}", f"Gather, offset {z - Z_SOURCE:+.1f} m",
               "gather", PLANE_BH,
               ("--fracture_type", "planar", "--rec_pos", "0.5", "5.0", f"{z:g}"),
               fdtd=GATHER_FRAC, direct=GATHER_NOFRAC)
          for z in GATHER_Z]
# Domain truncation: the finite planar case from a plane computed on a 4.0 m domain
CASES += [Case("truncated_domain", "Finite planar, plane from a 4.0 m domain", "truncation",
               PLANE_TRUNCATED, ("--fracture_type", "planar"))]
# Incidence diagnostic: one direction per element against one per frequency (projected areas)
CASES += [Case("incidence_band", "Piecewise concave, band-averaged incidence", "incidence",
               PLANE_BH, ("--fracture_type", "piecewise_yz", "--element_area", "projected")),
          Case("incidence_perfreq", "Piecewise concave, incidence per frequency", "incidence",
               PLANE_BH, ("--fracture_type", "piecewise_yz", "--element_area", "projected",
                          "--incidence", "perfreq"))]
# Discretization convergence (Supplementary Table S1), projected element areas
CONVERGENCE_SHAPES = {"planar": ("--fracture_type", "planar"),
                      "corrugated": ("--fracture_type", "sinusoidal_3d", "--amplitude", "0.25")}
ELEMENT_SIZES = [("0pt005", 0.005), ("0pt01", 0.01), ("0pt02", 0.02), ("0pt04", 0.04),
                 ("0pt08", 0.08), ("0pt16", 0.16)]
PLANE_SPACINGS = [("0pt02", 0.02), ("0pt04", 0.04), ("0pt08", 0.08), ("0pt16", 0.16)]
for shape, sflags in CONVERGENCE_SHAPES.items():
    CASES += [Case(f"convergence_dl_{shape}_{tag}", f"Element size {dl:g} m, {shape}",
                   "convergence", PLANE_BH,
                   sflags + ("--ellength", f"{dl:g}", "--element_area", "projected"))
              for tag, dl in ELEMENT_SIZES]
    CASES += [Case(f"convergence_res_{shape}_{tag}", f"Plane spacing {r:g} m, {shape}",
                   "convergence", PLANE_BH, sflags + ("--element_area", "projected"),
                   raw_plane=True, target_res=r)
              for tag, r in PLANE_SPACINGS]

BY_KEY = {c.key: c for c in CASES}
if len(BY_KEY) != len(CASES):
    raise RuntimeError("duplicate case keys")


def fdtd_output(model: str) -> Path:
    """Path of a gprMax output: next to its input file, or in RAHPO_FDTD_DIR when set."""
    name = Path(model).name + ".out"
    return Path(FDTD_DIR) / name if FDTD_DIR else MODELS / f"{model}.out"


def plane_input(case: Case, side: str = "tx") -> Path:
    """The plane file the solver reads: converted (output/planes/<model>.h5) or the gprMax .out."""
    model = case.plane_tx if side == "tx" or case.plane_rx is None else case.plane_rx
    return MODELS / f"{model}.out" if case.raw_plane else PLANE_DIR / f"{Path(model).name}.h5"


def trace_file(key: str) -> Path:
    return TRACE_DIR / f"{key}.npz"
