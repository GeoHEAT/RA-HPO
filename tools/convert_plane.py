"""
Convert a gprMax reference plane into the contiguous layout the solver reads fastest.

A gprMax .out file stores one dataset per receiver and field component, so reading a full
reference plane costs one small read per receiver and component. This script reads the plane
once, through the solver's own loader, and writes each component as a single contiguous
float32 dataset. The solver then reads the plane in six sequential blocks.

Usage
-----
    python convert_plane.py --in plane.out --out plane.h5
    python convert_plane.py --in plane.out --out plane.h5 --target_res 0.04

The first form keeps the plane as recorded. The second also decimates a finely recorded plane
to the given receiver spacing (the 0.01 m reference planes of the paper are read at 0.04 m).
Pass the converted file to ra_hpo.py in place of the .out file. Passing the same file as
--tx_plane_file and --rx_plane_file selects the axial shift for the reciprocal field.

Sizes are reported in decimal GB (bytes / 1e9).
"""
from __future__ import annotations

import argparse
import os
import sys
import time
from pathlib import Path

import h5py
import numpy as np

# The solver is imported for its loader, so that decimation is shared rather than
# reimplemented here. Its command-line parsing sits under an "if __name__ == '__main__'" guard.
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from rahpo.fields import load_gprmax_plane  # noqa: E402

# Bump this if the on-disk layout ever changes, so that an old file is rejected rather than
# silently misread.
FORMAT_VERSION = 1


def convert(in_path: str, out_path: str, target_res: float) -> None:
    """Read one gprMax .out plane through the solver's loader and write it back contiguously."""
    src = Path(in_path)
    if not src.is_file():
        raise FileNotFoundError(in_path)

    print(f"Reading {in_path} (target_res = {target_res} m)")
    t0 = time.perf_counter()
    fields, x1, yy, zz, dt, Nt = load_gprmax_plane(in_path, target_res=target_res)
    t_read = time.perf_counter() - t0

    nz, ny = zz.size, yy.size
    payload = sum(a.nbytes for a in fields.values())
    print(f"  read {len(fields)} components, {nz} x {ny} receivers, {Nt} samples "
          f"in {t_read:.1f} s ({payload / 1e9:.2f} GB in memory)")

    # Spacing of the grid actually delivered, which is what a later run must match. With
    # target_res=None the plane is used as recorded, so that spacing is the recording spacing
    # and no decimation was applied.
    eff_res = float(abs(yy[1] - yy[0])) if yy.size > 1 else 0.0
    if target_res is None:
        stored_res, step = eff_res, 1
    else:
        stored_res = float(target_res)
        step = int(round(target_res / eff_res)) if eff_res > 0 else 1

    print(f"Writing {out_path}")
    t0 = time.perf_counter()
    with h5py.File(out_path, "w") as f:
        # One contiguous dataset per component: no chunking, no compression, so that each
        # component is a single sequential read.
        for comp, arr in fields.items():
            f.create_dataset(comp, data=np.ascontiguousarray(arr, dtype=np.float32))
        f.create_dataset("yy", data=np.asarray(yy, dtype=np.float64))
        f.create_dataset("zz", data=np.asarray(zz, dtype=np.float64))

        f.attrs["format_version"] = FORMAT_VERSION
        f.attrs["x1"] = float(x1)
        f.attrs["dt"] = float(dt)
        f.attrs["Nt"] = int(Nt)
        f.attrs["target_res"] = stored_res
        f.attrs["step"] = step
        # Provenance of the converted plane.
        f.attrs["source_file"] = str(src.resolve())
        f.attrs["source_size"] = src.stat().st_size
        f.attrs["source_mtime"] = src.stat().st_mtime
    t_write = time.perf_counter() - t0

    out_size = Path(out_path).stat().st_size
    print(f"  wrote {out_size / 1e9:.2f} GB in {t_write:.1f} s")
    print(f"  source {src.stat().st_size / 1e9:.1f} GB -> converted {out_size / 1e9:.2f} GB "
          f"({src.stat().st_size / out_size:.1f}x smaller)")


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--in", dest="in_path", required=True, help="gprMax .out reference plane")
    ap.add_argument("--out", dest="out_path", required=True, help="converted .h5 to write")
    ap.add_argument("--target_res", type=float, default=None,
                    help="receiver spacing in metres to read the plane at. Default: the plane "
                         "as recorded. Use 0.04 for a plane recorded at 0.01 m.")
    args = ap.parse_args()

    if os.path.exists(args.out_path):
        raise SystemExit(f"refusing to overwrite existing file: {args.out_path}")
    convert(args.in_path, args.out_path, args.target_res)


if __name__ == "__main__":
    main()
