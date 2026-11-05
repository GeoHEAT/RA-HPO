"""Slice test: does one voxel act as 0.02 m in gprMax 3.1.7?

Without material averaging, gprMax 3.1.7 gives a filled cell's material to all twelve
electric-field edges of the cell, so a slab of N cells carries the fill on N + 1 grid planes of
the field tangential to it (tools/voxel_thickness.py). This test checks the rule on one voxel.
Three small models, 3 x 3 x 3 m at 0.01 m and 60 ns, host eps_r 9 and 1e-4 S/m, a z-directed
dipole at (0.8, 1.5, 1.8) m (the paper's Ricker, ricker 100 100e6) and one receiver 0.6 m
below it:

    a       no slab;
    b       one voxel at x = 2.30-2.31 m across the domain (eps_r 9, 0.5 S/m), read from an HDF5
            volume with #geometry_objects_read and built by generate_geometry_h5.build_planar_dip;
    b_box   the same slab as #box with averaging off (n).

The reflection is each trace minus trace a. It is compared with the exact infinite layer of
eps_r 9 and 0.5 S/m (examples/analytic/thin_layers.py) centered on the voxel, 0.02 m thick
(the voxel acting as two cells) and 0.01 m thick (the voxel as built). The exact traces carry
the 1/fc wavelet delay and are shifted to gprMax's sqrt(2)/fc. The script also reports whether
b and b_box give the same trace.

    python tools/slice_test.py --write WORKDIR
    (cd WORKDIR && for m in a b b_box; do python -m gprMax $m.in; done)
    python tools/slice_test.py --analyze WORKDIR

Run gprMax 3.1.7 (the version of the paper's outputs). Each model takes a few minutes on a
laptop and about 2 GB of memory.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import h5py
import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(ROOT / "examples" / "analytic"))

L, H = 3.0, 0.01
SOURCE, RECEIVER = (0.8, 1.5, 1.8), (0.8, 1.5, 1.2)
VOXEL_X = 2.305                      # center of the voxel at x = 2.30-2.31 m
FILL = (9.0, 0.5)
WINDOW_NS = (25.0, 60.0)
FC = 1e8

HEADER = """#title: Slice test {name}: {what}
#domain: {L} {L} {L}
#dx_dy_dz: {H} {H} {H}
#time_window: 60e-9

#material: 9.0 1e-4 1 0 rock
#waveform: ricker 100 100e6 my_ricker
#hertzian_dipole: z {sx} {sy} {sz} my_ricker
#rx: {rx} {ry} {rz}

#box: 0 0 0 {L} {L} {L} rock
"""
MODELS = {
    "a": ("no slab", ""),
    "b": ("one voxel from an HDF5 volume",
          "#geometry_objects_read: 0 0 0 slice_voxel.h5 slice_materials.txt\n"),
    "b_box": ("the same slab as #box, averaging off",
              f"#material: {FILL[0]} {FILL[1]} 1 0 sheet\n#box: 2.30 0 0 2.31 {L} {L} sheet n\n"),
}


def write(workdir: Path) -> None:
    from generate_geometry_h5 import build_planar_dip, write_volume
    workdir.mkdir(parents=True, exist_ok=True)
    for name, (what, extra) in MODELS.items():
        text = HEADER.format(name=name, what=what, L=L, H=H, sx=SOURCE[0], sy=SOURCE[1], sz=SOURCE[2],
                             rx=RECEIVER[0], ry=RECEIVER[1], rz=RECEIVER[2]) + extra
        (workdir / f"{name}.in").write_text(text)
    n = int(round(L / H))
    vol = build_planar_dip(0.0, VOXEL_X, 1.5, 1, (-1e9, 1e9, -1e9, 1e9), dims=(n, n, n),
                           origin=(0.0, 0.0, 0.0))
    frac = np.nonzero((vol == 0).any(axis=(1, 2)))[0]
    if frac.tolist() != [230]:
        raise RuntimeError(f"the voxel is not the single x index 230: {frac}")
    write_volume(workdir / "slice_voxel.h5", vol)
    (workdir / "slice_materials.txt").write_text(f"#material: {FILL[0]} {FILL[1]} 1 0 frc_sheet\n")
    print(f"wrote {', '.join(m + '.in' for m in MODELS)}, slice_voxel.h5, slice_materials.txt in {workdir}")


def read_ez(path: Path):
    with h5py.File(path, "r") as f:
        dt = float(f.attrs["dt"])
        ez = f["rxs/rx1/Ez"][:].astype(float)
    return np.arange(ez.size) * dt * 1e9, ez


def analyze(workdir: Path) -> int:
    from thin_layers import FINE, exact_trace
    t, a = read_ez(workdir / "a.out")
    refl, raw = {}, {}
    for m in ("b", "b_box"):
        p = workdir / f"{m}.out"
        if p.exists():
            tm, e = read_ez(p)
            if not np.allclose(tm, t):
                raise ValueError(f"time axes of a and {m} differ")
            refl[m] = e - a
            raw[m] = e
    front_dist = VOXEL_X - SOURCE[0]                       # source plane to the voxel center
    offset = SOURCE[2] - RECEIVER[2]
    shift = (np.sqrt(2.0) - 1.0) / FC * 1e9
    ex = {}
    for label, d in (("0.02 m (two cells)", 0.02), ("0.01 m (as built)", 0.01)):
        te, e = exact_trace(d, FILL[0], FILL[1], front=front_dist - d / 2, r=offset, **FINE)
        ex[label] = np.interp(t, te + shift, e)
    win = (t >= WINDOW_NS[0]) & (t <= WINDOW_NS[1])
    print(f"Reflection over {WINDOW_NS[0]:g}-{WINDOW_NS[1]:g} ns; peak ratio FDTD / exact, chi normalized by the exact trace")
    if not refl:
        print("models b and b_box missing")
        return 2
    for m, r in refl.items():
        cells = []
        for label, e in ex.items():
            pk = np.max(np.abs(r[win])) / np.max(np.abs(e[win]))
            chi = 100 * np.linalg.norm(r[win] - e[win]) / np.linalg.norm(e[win])
            cells.append(f"{label}: peak {pk:.3f}, chi {chi:5.1f} %")
        print(f"  {m:6s} " + "   ".join(cells))
    if len(raw) == 2:
        same = np.array_equal(raw["b"], raw["b_box"])
        print(f"b and b_box traces {'bitwise identical' if same else 'differ'}")
    return 0


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    g = ap.add_mutually_exclusive_group(required=True)
    g.add_argument("--write", metavar="WORKDIR")
    g.add_argument("--analyze", metavar="WORKDIR")
    a = ap.parse_args()
    if a.write:
        write(Path(a.write))
        return 0
    return analyze(Path(a.analyze))


if __name__ == "__main__":
    sys.exit(main())
