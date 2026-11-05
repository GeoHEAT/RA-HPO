"""Cut a small reference plane for the worked example.

A converted reference plane of the paper (convert_plane.py) is 5.73 GB: 226 x 226 receivers,
six components, about 4 700 time samples. The worked example needs only the part of the plane
that the rays to a small fracture cross, and only the band the solver keeps (below 8 f_c =
800 MHz), so this script

  1. crops the plane to a rectangle in (y, z), and
  2. decimates it in time by an integer factor with an anti-aliasing polyphase filter
     (scipy.signal.resample_poly), refusing any factor whose new Nyquist frequency is below
     twice the solver's cut-off,

and writes a converted plane in the same format, which ra_hpo.py reads unchanged. The attributes
record the crop, the factor and the source file.

Which rectangle. With the plane at x1 = 3.5 m, the source at (0.5, 5.0, 5.5) and the fracture at
x = 6.5 m, a ray from the source to an element crosses the plane half-way, so a fracture
y, z in [4.5, 5.5] is seen at y in [4.75, 5.25], z in [5.0, 5.5]. The reciprocal field comes from
the same plane by the axial shift: the receiver 1.0 m below the source is evaluated at the
element moved up by 1.0 m, which crosses the plane at z in [5.5, 6.0]. An element whose ray
misses the plane contributes nothing, so the default rectangle y in [4.4, 5.6], z in [4.6, 6.4]
keeps a margin. The log line "[DROPPED] 0 of ... elements" of ra_hpo.py confirms that no ray
misses it.

Usage
-----
    python tools/make_example_plane.py --in output/planes/3D_8x10x10-shifted_nofrac_planex3_h0pt01.h5 \\
        --out examples/example_plane.h5 [--y 4.4 5.6] [--z 4.6 6.4] [--decimate 8]
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import h5py
import numpy as np
from scipy.signal import resample_poly

COMPS = ("Ex", "Ey", "Ez", "Hx", "Hy", "Hz")
CUTOFF_HZ = 8 * 100e6        # the solver keeps frequencies below 8 f_c, f_c = 100 MHz


def crop_and_decimate(src: Path, dst: Path, yr, zr, factor: int) -> dict:
    if dst.exists():
        raise FileExistsError(f"{dst} exists; this script does not overwrite")
    with h5py.File(src, "r") as f:
        if "format_version" not in f.attrs:
            raise ValueError(f"{src} is not a converted plane (run convert_plane.py first)")
        yy, zz = f["yy"][:], f["zz"][:]
        iy = np.where((yy >= yr[0] - 1e-9) & (yy <= yr[1] + 1e-9))[0]
        iz = np.where((zz >= zr[0] - 1e-9) & (zz <= zr[1] + 1e-9))[0]
        if iy.size < 2 or iz.size < 2:
            raise ValueError(f"crop y {yr}, z {zr} leaves {iy.size} x {iz.size} receivers")
        dt, nt = float(f.attrs["dt"]), int(f.attrs["Nt"])
        nyq_new = 1.0 / (2.0 * dt * factor)
        if nyq_new < 2.0 * CUTOFF_HZ:
            raise ValueError(f"decimation {factor} gives Nyquist {nyq_new / 1e9:.2f} GHz, below "
                             f"twice the {CUTOFF_HZ / 1e6:.0f} MHz cut-off")
        out = {}
        for c in COMPS:
            a = f[c][iz[0]:iz[-1] + 1, iy[0]:iy[-1] + 1, :].astype(np.float64)   # (nz, ny, Nt)
            if a.shape[-1] != nt:
                raise ValueError(f"{c}: time axis {a.shape[-1]} != Nt {nt}")
            out[c] = resample_poly(a, up=1, down=factor, axis=-1) if factor > 1 else a
        attrs = dict(f.attrs)
    nt_new = out["Ez"].shape[-1]
    with h5py.File(dst, "w") as g:
        for c in COMPS:
            g.create_dataset(c, data=np.ascontiguousarray(out[c], dtype=np.float32))
        g.create_dataset("yy", data=yy[iy[0]:iy[-1] + 1].astype(np.float64))
        g.create_dataset("zz", data=zz[iz[0]:iz[-1] + 1].astype(np.float64))
        for k, v in attrs.items():
            g.attrs[k] = v
        g.attrs["dt"] = dt * factor
        g.attrs["Nt"] = nt_new
        g.attrs["example_crop_y"] = np.array(yr, dtype=float)
        g.attrs["example_crop_z"] = np.array(zr, dtype=float)
        g.attrs["example_time_decimation"] = factor
        g.attrs["example_derived_from"] = str(src)
    return {"ny": iy.size, "nz": iz.size, "Nt": nt_new, "dt_ns": dt * factor * 1e9,
            "nyquist_GHz": nyq_new / 1e9, "bytes": dst.stat().st_size}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--in", dest="src", required=True)
    ap.add_argument("--out", dest="dst", required=True)
    ap.add_argument("--y", type=float, nargs=2, default=(4.4, 5.6))
    ap.add_argument("--z", type=float, nargs=2, default=(4.6, 6.4))
    ap.add_argument("--decimate", type=int, default=8)
    a = ap.parse_args()
    info = crop_and_decimate(Path(a.src), Path(a.dst), a.y, a.z, a.decimate)
    print(f"wrote {a.dst}: {info['nz']} x {info['ny']} receivers, Nt {info['Nt']}, "
          f"dt {info['dt_ns']:.4f} ns (Nyquist {info['nyquist_GHz']:.2f} GHz), "
          f"{info['bytes'] / 1e6:.1f} MB")
    return 0


if __name__ == "__main__":
    sys.exit(main())
