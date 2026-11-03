"""Ray-aligned hybrid physical optics for borehole radar.

The numerical stages can be imported on their own:

- ``rahpo.parameters`` — dataclasses that fix one run
- ``rahpo.fracture`` — surface, fill, and mesh
- ``rahpo.fields`` — reference-plane loading and ray propagation
- ``rahpo.scattering`` — physical-optics reaction integral
- ``rahpo.simulation`` — one scattered trace
- ``rahpo.cases`` — the cases of the paper and where their files live
- ``rahpo.traces`` — loading traces and the comparison metrics
- ``rahpo.cli`` — the command line used by ``python ra_hpo.py``
"""
from __future__ import annotations

__version__ = "1.0.0"
