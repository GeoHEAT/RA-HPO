"""The case catalogue and the comparison metrics."""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from examples.cases import BY_KEY, CASES, MODELS, REPO, plane_two_medium
from examples.traces import band_match, compare_traces


def test_case_keys_are_unique_and_the_model_tree_is_the_repository():
    assert len(BY_KEY) == len(CASES)
    assert REPO == Path(__file__).resolve().parents[1]
    assert MODELS == REPO / "gprMax_models"
    assert MODELS.is_dir()


def test_two_medium_plane_name_carries_the_permittivity():
    assert "espr12" in plane_two_medium(12, "tx")
    assert "tx4pt5" in plane_two_medium(6, "rx")


def test_identical_traces_have_no_misfit():
    time = np.linspace(0.0, 180.0, 400)
    data = np.sin(2 * np.pi * time / 40.0)
    metrics = compare_traces(time, data, time, data, time_range=(0.0, 180.0))
    assert metrics["Relative misfit (%)"] == 0.0
    assert metrics["Pearson r"] == 1.0
    assert metrics["Peak ratio (ref/RA-HPO)"] == 1.0
    assert metrics["Peak-time diff (ns)"] == 0.0


def test_band_match_of_zeros_is_zero():
    time = np.arange(64, dtype=float)
    matched = band_match(time, np.zeros(64))
    assert matched.shape == (64,)
    assert np.all(matched == 0.0)


def test_output_directories_are_created_on_import():
    import os

    from examples.cases import FIGURE_DIR, METRICS_DIR, PLANE_DIR, TRACE_DIR
    from rahpo.paths import (
        COMPARISONS,
        FIGURES,
        IMAGES,
        LOGS,
        METRICS,
        OUTPUT_DIR,
        PLANES,
        PROPAGATION,
        TRACES,
    )

    assert OUTPUT_DIR == REPO / "output"
    assert (TRACE_DIR, PLANE_DIR, FIGURE_DIR, METRICS_DIR) == (TRACES, PLANES, FIGURES, METRICS)
    folders = {
        "RAHPO_TRACE_DIR": TRACES,
        "RAHPO_LOG_DIR": LOGS,
        "RAHPO_IMAGE_DIR": IMAGES,
        "RAHPO_FIGURE_DIR": FIGURES,
        "RAHPO_METRICS_DIR": METRICS,
        "RAHPO_PLANE_DIR": PLANES,
        "RAHPO_PROPAGATION_DIR": PROPAGATION,
        "RAHPO_COMPARISON_DIR": COMPARISONS,
    }
    for env_name, folder in folders.items():
        assert folder.is_dir()
        if env_name not in os.environ:
            assert folder.is_relative_to(OUTPUT_DIR)
