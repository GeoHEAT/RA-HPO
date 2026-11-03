"""Where runs write their files.

``OUTPUT_DIR`` sits next to the repository. Importing :mod:`rahpo` creates it
and the folders below. A script takes a path from here and writes with
``pathlib``; it does not invent a folder of its own.

An environment variable replaces a folder when the files are too large to keep
next to the source tree: ``RAHPO_TRACE_DIR``, ``RAHPO_LOG_DIR``,
``RAHPO_IMAGE_DIR``, ``RAHPO_FIGURE_DIR``, ``RAHPO_METRICS_DIR``,
``RAHPO_PLANE_DIR``, ``RAHPO_PROPAGATION_DIR``, ``RAHPO_COMPARISON_DIR``.
"""
from __future__ import annotations

import os
from pathlib import Path

PACKAGE_DIR = Path(__file__).resolve().parent
PROJECT_DIR = PACKAGE_DIR.parent
OUTPUT_DIR = PROJECT_DIR / "output"


def _folder(name: str, env_name: str) -> Path:
    """Return ``output/<name>``, or the directory named by ``env_name``."""
    override = os.environ.get(env_name)
    path = Path(override) if override else OUTPUT_DIR / name
    path.mkdir(parents=True, exist_ok=True)
    return path


TRACES = _folder("traces", "RAHPO_TRACE_DIR")
LOGS = _folder("logs", "RAHPO_LOG_DIR")
IMAGES = _folder("images", "RAHPO_IMAGE_DIR")
FIGURES = _folder("figures", "RAHPO_FIGURE_DIR")
METRICS = _folder("metrics", "RAHPO_METRICS_DIR")
PLANES = _folder("planes", "RAHPO_PLANE_DIR")
PROPAGATION = _folder("propagation", "RAHPO_PROPAGATION_DIR")
COMPARISONS = _folder("comparisons", "RAHPO_COMPARISON_DIR")
