# Using the solver

`rahpo` is the method. The experiments of the paper are not part of it: they live in
[`examples/`](../examples/) and are run by the scripts in [`tools/`](../tools/).

## What to import

| Module | Role |
|---|---|
| `rahpo.parameters` | `MediaParams`, `SourceParams`, `GeometryParams`, `RuntimeParams` |
| `rahpo.fracture` | Surface `x = f(y, z)`, fill properties, mesh, specular point |
| `rahpo.fields` | Read a reference plane and propagate one field along a ray |
| `rahpo.scattering` | Physical-optics contribution of one fracture element |
| `rahpo.simulation` | One scattered trace, `run_simulation` |
| `rahpo.cli` | The command line behind `python ra_hpo.py` |
| `examples.cases` | Every case of the paper: models, flags, output names |
| `examples.traces` | Load a trace and compute the comparison metrics |

A script that only needs the surface does not import the plane reader, and a script
that reproduces a table does not live inside `rahpo`.

## One trace from Python

From the repository root, with the package installed (`pip install -e .`) or with
the root on `PYTHONPATH`:

```python
from rahpo.parameters import GeometryParams, MediaParams, RuntimeParams, SourceParams
from rahpo.simulation import run_simulation

time_s, ez = run_simulation(
    MediaParams(),
    SourceParams(),
    GeometryParams(fracture_type="planar", ymin=4.5, ymax=5.5, zmin=4.5, zmax=5.5),
    RuntimeParams(
        tx_plane_file="examples/example_plane.h5",
        rx_plane_file="examples/example_plane.h5",
        cores=4,
        no_viz=True,
    ),
)
```

`time_s` is in seconds and `ez` is the scattered field at the receiver, in V/m.
The same run from the command line is the quick start in the [README](../README.md).

## A case of the paper

```python
from examples.cases import BY_KEY, plane_input

case = BY_KEY["finite_planar"]
print(case.label, case.flags)
print(plane_input(case, "tx"))
```

`python tools/run_case.py finite_planar` prints the gprMax commands, the plane
conversion, and the RA-HPO command for that case. The gprMax inputs are
`examples/gprMax_models/`. The exact infinite-layer trace is
`examples/analytic/`. `examples/cases.json` is the same catalogue as
`examples/cases.py`.

## Figures

`python examples/showcase.py` redraws the two pictures in
[`docs/images/`](images/). Those are the pictures embedded in the README. They are
written there on purpose, so the README can show them.

Every other product of a run goes under [`output/`](../output/), which `import rahpo`
creates and git ignores:

| Folder | Contents |
|---|---|
| `output/traces/` | RA-HPO traces, `<name>.npz` |
| `output/logs/` | one log per run |
| `output/images/` | the diagnostic trace plot and the geometry plot |
| `output/figures/` | PDF figures of the paper |
| `output/metrics/` | CSV tables |
| `output/planes/` | converted reference planes |
| `output/propagation/` | the ray-propagation test |
| `output/comparisons/` | the truncated-domain plane comparison |

`examples.cases` points `TRACE_DIR`, `PLANE_DIR`, `FIGURE_DIR` and `METRICS_DIR` at
those folders. A script builds a path with `pathlib` from one of them.
