"""``python -m rahpo`` runs the same command line as ``python ra_hpo.py``."""
from __future__ import annotations

import sys

from rahpo.cli import main

if __name__ == "__main__":
    raise SystemExit(main())
