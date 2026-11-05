"""Constant-offset profile past three fractures: scene, validity audit, cases, trace sum.

The tool moves along a vertical well (x = 0.5 m, y = 5 m) with the receiver 1 m below the
source. The reference plane moves with it (ra_hpo.py --tool_shift), which is exact for an
axially uniform borehole and host. Three fractures (patches) lie beside the well, each 14 m along dip by
8 m along strike (y = 1 to 9 m), 15 m apart along the well:

    A   dipping 30 degrees, 1 mm brine (eps_r 43, 10 S/m), down-dip edge at x = 3.7 m
    B   parallel to the well (0 degrees), 1 mm brine, at x = 6.5 m
    C   as A, but 1 mm fresh water (eps_r 80, 0.01 S/m)

A tool position is valid for a patch when the first Fresnel zone of its specular reflection
(the points whose two-way path exceeds the specular one by at most half a wavelength, 1 m at
100 MHz in the host) lies inside the patch, every ray from the source and from the receiver to
that zone crosses the recorded plane inside its extent, and no such ray meets another patch
(shadow audit: a 1 mm, 10 S/m sheet transmits only 0.61 one way). Positions are spaced
0.25 m. Each valid position gives one case on the borehole plane and one on the
borehole-free plane; a calibration pair for each dipping patch (0 degrees, same fill, at the
specular distance of its middle position) gives the broadside-calibrated dipole. Patch B has no
pair of its own: it is calibrated on A's pair, which has the same fill.

--sum reduces the traces. The peak of a trace is max |trace| over the whole trace, since each
profile case meshes one patch only. Per patch: the median over its valid positions of
peak(borehole plane) / peak(calibrated dipole), and its median peak on the borehole plane
against that of A, in dB.

    python tools/profile_scene.py --audit                 # summary of the valid positions
    python tools/profile_scene.py --cases cases_profile.tsv
    python tools/profile_scene.py --sum [--out DIR]       # per-patch numbers and per-position CSV
"""
from __future__ import annotations

import argparse
import sys
from dataclasses import dataclass
from pathlib import Path

import numpy as np

SOURCE_X, WELL_Y, Z_SOURCE = 0.5, 5.0, 5.5     # the plane is recorded for the source at z = 5.5 m
OFFSET = 1.0                                   # receiver below the source [m]
X_PLANE, PLANE_Y, PLANE_Z = 3.5, (0.5, 9.5), (0.5, 9.5)
LAMBDA = 1.0                                   # host wavelength at 100 MHz [m]
STEP = 0.25
DL = 0.02
BRINE, FRESH = (43.0, 10.0), (80.0, 0.01)
APERTURE = 0.001


@dataclass(frozen=True)
class Patch:
    name: str
    dip: float          # degrees from the borehole axis
    x_ref: float        # the plane passes x_ref at z_ref
    z_ref: float
    z_lo: float         # extent along the well [m]
    z_hi: float
    fill: tuple

    def x_at(self, z):
        return self.x_ref + (np.asarray(z) - self.z_ref) * np.tan(np.radians(self.dip))

    def flags(self):
        f = (["--fracture_type", "planar", "--fracture_x", f"{self.x_ref:g}"] if self.dip == 0 else
             ["--fracture_type", "planar-dip", "--dip", f"{self.dip:g}", "--fracture_x", f"{self.x_ref:g}",
              "--fracture_z", f"{self.z_ref:g}"])
        return f + ["--fracture_extent", "1", "9", f"{self.z_lo:g}", f"{self.z_hi:g}",
                    "--planethick", f"{APERTURE:g}", "--fluid_epsr", f"{self.fill[0]:g}",
                    "--fluid_sigma", f"{self.fill[1]:g}", "--ellength", f"{DL:g}"]


def _dipping(name, z_lo, fill, dip=30.0, x_lo=3.7, along=14.0):
    z_hi = z_lo + along * np.cos(np.radians(dip))
    return Patch(name, dip, x_lo, z_lo, z_lo, round(z_hi, 3), fill)


A = _dipping("A", 10.0, BRINE)
B = Patch("B", 0.0, 6.5, 0.0, A.z_hi + 15.0, A.z_hi + 15.0 + 14.0, BRINE)
C = _dipping("C", B.z_hi + 15.0, FRESH)
SCENE = (A, B, C)


def _unit(dip):
    b = np.radians(dip)
    return np.array([np.cos(b), -np.sin(b)]), np.array([np.sin(b), np.cos(b)])   # normal, along dip


def specular(patch, z_s):
    """Specular point (x, z) on the infinite plane of the patch for source z_s, receiver z_s - 1."""
    n, _ = _unit(patch.dip)
    c = n @ np.array([patch.x_ref, patch.z_ref])
    s, r = np.array([SOURCE_X, z_s]), np.array([SOURCE_X, z_s - OFFSET])
    image = r + 2 * (c - n @ r) * n
    d = image - s
    return s + (c - n @ s) / (n @ d) * d


def fresnel_zone(patch, z_s, h=0.05):
    """Points (x, y, z) of the plane within half a wavelength of the specular path length."""
    n, t = _unit(patch.dip)
    p0 = specular(patch, z_s)
    src = np.array([SOURCE_X, WELL_Y, z_s])
    rec = np.array([SOURCE_X, WELL_Y, z_s - OFFSET])
    s = np.arange(-4.0, 4.0 + h / 2, h)
    y = np.arange(WELL_Y - 4.0, WELL_Y + 4.0 + h / 2, h)
    S, Y = np.meshgrid(s, y, indexing="ij")
    P = np.stack([p0[0] + S * t[0], Y, p0[1] + S * t[1]], axis=-1).reshape(-1, 3)
    L = np.linalg.norm(P - src, axis=1) + np.linalg.norm(P - rec, axis=1)
    L0 = np.linalg.norm(np.array([p0[0], WELL_Y, p0[1]]) - src) + np.linalg.norm(np.array([p0[0], WELL_Y, p0[1]]) - rec)
    return P[L <= L0 + LAMBDA / 2], L0


def _crosses_plane(P, origin, z_shift):
    """Rays from origin to the points P cross x = X_PLANE inside the plane shifted by z_shift."""
    lam = (X_PLANE - origin[0]) / (P[:, 0] - origin[0])
    y = origin[1] + lam * (P[:, 1] - origin[1])
    z = origin[2] + lam * (P[:, 2] - origin[2])
    return ((y >= PLANE_Y[0]) & (y <= PLANE_Y[1]) & (z >= PLANE_Z[0] + z_shift) & (z <= PLANE_Z[1] + z_shift)
            & (lam > 0) & (lam < 1))


def _segment_hits(patch, a, b):
    """Does any segment a[i] -> b[i] (points (x, y, z)) cross the patch rectangle?"""
    n, _ = _unit(patch.dip)
    c = n @ np.array([patch.x_ref, patch.z_ref])
    fa = a[:, 0] * n[0] + a[:, 2] * n[1] - c
    fb = b[:, 0] * n[0] + b[:, 2] * n[1] - c
    cross = fa * fb < 0
    if not np.any(cross):
        return False
    w = fa[cross] / (fa[cross] - fb[cross])
    p = a[cross] + w[:, None] * (b[cross] - a[cross])
    inside = (p[:, 1] >= 1.0) & (p[:, 1] <= 9.0) & (p[:, 2] >= patch.z_lo) & (p[:, 2] <= patch.z_hi)
    return bool(np.any(inside))


def audit(patch, z_s):
    """(valid, reason) for one tool position and one patch."""
    zone, L0 = fresnel_zone(patch, z_s)
    if zone[:, 2].min() < patch.z_lo or zone[:, 2].max() > patch.z_hi:
        return False, "Fresnel zone leaves the patch along dip"
    if zone[:, 1].min() < 1.0 or zone[:, 1].max() > 9.0:
        return False, "Fresnel zone leaves the patch along strike"
    if zone[:, 0].min() <= X_PLANE + 0.1:
        return False, "Fresnel zone reaches the reference plane"
    shift = z_s - Z_SOURCE
    src = np.array([SOURCE_X, WELL_Y, z_s])
    if not np.all(_crosses_plane(zone, src, shift)):
        return False, "a source ray misses the recorded plane"
    # the receiver's field is the source's shifted by the offset: virtual points zone + offset
    virt = zone + np.array([0.0, 0.0, OFFSET])
    if not np.all(_crosses_plane(virt, src, shift)):
        return False, "a receiver ray misses the recorded plane"
    rec = np.array([SOURCE_X, WELL_Y, z_s - OFFSET])
    for other in SCENE:
        if other is patch:
            continue
        if _segment_hits(other, np.repeat(src[None], len(zone), 0), zone) or \
           _segment_hits(other, zone, np.repeat(rec[None], len(zone), 0)):
            return False, f"shadowed by patch {other.name}"
    return True, f"two-way path {L0:.2f} m"


def positions(patch):
    """Valid source heights for the patch, on the 0.25 m grid."""
    lo = patch.z_lo - 2.0
    hi = patch.z_hi + 12.0
    zs = np.round(np.arange(np.floor(lo / STEP) * STEP, hi + STEP / 2, STEP), 3)
    return [z for z in zs if audit(patch, z)[0]]


def calibration_patch(patch, z_s):
    """The 0-degree patch with the same fill at the specular distance of z_s (calibration pair)."""
    n, _ = _unit(patch.dip)
    p = specular(patch, z_s)
    d = n @ (p - np.array([SOURCE_X, z_s]))          # distance from the source to the plane
    return Patch(f"{patch.name}cal", 0.0, round(SOURCE_X + d, 3), 0.0, z_s - 7.0, z_s + 7.0, patch.fill)


def case_rows():
    rows = []
    for patch in SCENE:
        zs = positions(patch)
        for z in zs:
            for plane in ("bh", "nobh"):
                rows.append((case_name(patch.name, z, plane), plane,
                             ["--tool_shift", f"{z - Z_SOURCE:g}"] + patch.flags()))
        if patch.dip != 0 and zs:
            mid = zs[len(zs) // 2]
            cal = calibration_patch(patch, mid)
            for plane in ("bh", "nobh"):
                rows.append((case_name(f"{patch.name}cal", mid, plane), plane,
                             ["--tool_shift", f"{mid - Z_SOURCE:g}"] + cal.flags()))
    return rows


def case_name(patch_name, z, plane):
    return f"profile_{patch_name}_z{z:07.3f}_{plane}".replace(".", "p")


def calibration_names():
    """Calibration pair (bh, nobh) per patch name; B uses A's (same fill)."""
    out = {}
    for patch in SCENE:
        zs = positions(patch)
        if patch.dip != 0 and zs:
            mid = zs[len(zs) // 2]
            out[patch.name] = (case_name(f"{patch.name}cal", mid, "bh"), case_name(f"{patch.name}cal", mid, "nobh"))
    out["B"] = out["A"]
    return out


def peak(t, x):
    """(|peak|, time of the peak) over the whole trace."""
    i = int(np.argmax(np.abs(x)))
    return float(np.abs(x[i])), float(t[i])


def summarize(out_dir=None):
    """Per-patch numbers from the profile traces; returns the report lines."""
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    from dip_study import apply_filter, calibration_filter, have, pair
    cal = calibration_names()
    lines, table = [], []
    per = {}
    for patch in SCENE:
        kb_cal, kn_cal = cal[patch.name]
        if not (have(kb_cal) and have(kn_cal)):
            lines.append(f"patch {patch.name}: calibration pair {kb_cal[:-3]} missing, not scored")
            continue
        t, q, p = pair(kb_cal, kn_cal)
        W = calibration_filter(t, q, p)
        rec = []
        for z in positions(patch):
            kb, kn = case_name(patch.name, z, "bh"), case_name(patch.name, z, "nobh")
            if not (have(kb) and have(kn)):
                continue
            t, q, p = pair(kb, kn)
            d = apply_filter(W, p)
            (pq, tq), (pd, td) = peak(t, q), peak(t, d)
            rec.append((z, pq, tq, pd, td, pq / pd))
            table.append((patch.name, z, pq, tq, pd, td, pq / pd))
        per[patch.name] = rec
        n_all = len(positions(patch))
        if len(rec) < n_all:
            lines.append(f"patch {patch.name}: {len(rec)} of {n_all} positions available")
        if not rec:
            continue
        r = np.median([x[5] for x in rec])
        lines.append(f"patch {patch.name}: median peak ratio RA-HPO / dipole {r:.3f} over {len(rec)} positions")
    if per.get("A"):
        ref = np.median([x[1] for x in per["A"]])
        for name in ("B", "C"):
            if per.get(name):
                db = 20 * np.log10(np.median([x[1] for x in per[name]]) / ref)
                lines.append(f"patch {name} against A: median peak {db:+.1f} dB")
    if out_dir and table:
        out = Path(out_dir)
        out.mkdir(parents=True, exist_ok=True)
        with open(out / "profile_peaks.csv", "w") as f:
            f.write("patch,z_source,peak_bh,t_peak_bh_ns,peak_dipole,t_peak_dipole_ns,ratio\n")
            for row in table:
                f.write(",".join([row[0]] + [f"{v:.6g}" for v in row[1:]]) + "\n")
        lines.append(f"per-position peaks: {out / 'profile_peaks.csv'}")
    return lines


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--audit", action="store_true")
    ap.add_argument("--cases", metavar="TSV")
    ap.add_argument("--sum", action="store_true", help="reduce the traces in RAHPO_TRACE_DIR")
    ap.add_argument("--out", default=None, help="folder for profile_peaks.csv")
    a = ap.parse_args()
    if a.audit:
        for p in SCENE:
            zs = positions(p)
            print(f"patch {p.name}: dip {p.dip:g} deg, fill {p.fill}, x {p.x_at(p.z_lo):.2f}-{p.x_at(p.z_hi):.2f} m, "
                  f"z {p.z_lo:g}-{p.z_hi:g} m: {len(zs)} valid positions"
                  + (f", source z {zs[0]:g}-{zs[-1]:g} m" if zs else ""))
            for z in (zs[0] - STEP, zs[-1] + STEP) if zs else ():
                print(f"   just outside, z = {z:g}: {audit(p, z)[1]}")
        print(f"cases: {len(case_rows())} (both planes, calibration pairs included)")
    if a.cases:
        with open(a.cases, "w") as f:
            f.write("name\tplane\tflags\n")
            for name, plane, flags in case_rows():
                f.write(f"{name}\t{plane}\t{' '.join(flags)}\n")
        print(f"wrote {a.cases}")
    if a.sum:
        print("\n".join(summarize(a.out)))
    return 0


if __name__ == "__main__":
    sys.exit(main())
