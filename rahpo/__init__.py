"""Ray-aligned hybrid physical optics for borehole radar.

The numerical stages can be imported on their own:

- ``rahpo.parameters`` — dataclasses that fix one run
- ``rahpo.fracture`` — surface, fill, and mesh
- ``rahpo.fields`` — reference-plane loading and ray propagation
- ``rahpo.scattering`` — physical-optics reaction integral
- ``rahpo.simulation`` — one scattered trace
- ``rahpo.cli`` — the command line used by ``python ra_hpo.py``

The cases of the paper and the trace comparisons live in ``examples``, not in
this package. Runs write under ``output/``, which is created on import.
"""
from __future__ import annotations

from rahpo.paths import (
    COMPARISONS,
    FIGURES,
    IMAGES,
    LOGS,
    METRICS,
    OUTPUT_DIR,
    PACKAGE_DIR,
    PLANES,
    PROJECT_DIR,
    PROPAGATION,
    TRACES,
)

__version__ = "1.0.0"

__all__ = [
    "COMPARISONS",
    "FIGURES",
    "IMAGES",
    "LOGS",
    "METRICS",
    "OUTPUT_DIR",
    "PACKAGE_DIR",
    "PLANES",
    "PROJECT_DIR",
    "PROPAGATION",
    "TRACES",
]
