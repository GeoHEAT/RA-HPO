"""The figure scripts of the paper import and find their plot settings."""
from __future__ import annotations

import importlib
import sys
from pathlib import Path

import pytest

TOOLS = Path(__file__).resolve().parents[1] / "tools"
sys.path.insert(0, str(TOOLS.parent))
sys.path.insert(0, str(TOOLS))


@pytest.mark.parametrize("name", ["plot_traces", "gather", "plot_two_medium"])
def test_figure_script_imports(name):
    importlib.import_module(name)


def test_plot_traces_keeps_its_plot_settings():
    plot_traces = importlib.import_module("plot_traces")
    assert plot_traces.FIGSIZE == (3.6, 2.3)
    assert plot_traces.FONT_SIZE == 8
    assert set(plot_traces.COLORS) == {"analytic", "gprmax", "rahpo"}
    assert set(plot_traces.LINESTYLES) == {"analytic", "gprmax", "rahpo"}
    plot_traces.set_style()
