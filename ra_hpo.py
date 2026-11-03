#!/usr/bin/env python3
"""Command line for the RA-HPO solver.

The implementation lives in the ``rahpo`` package. This file is the command
used throughout the paper, so a run is still::

    python ra_hpo.py --tx_plane_file PLANE --rx_plane_file PLANE --output_name NAME

Import the stages from ``rahpo`` (``rahpo.fracture``, ``rahpo.fields``,
``rahpo.scattering``, ``rahpo.simulation``) rather than from this module.
"""
from __future__ import annotations

import sys

from rahpo.cli import main

if __name__ == "__main__":
    raise SystemExit(main())
