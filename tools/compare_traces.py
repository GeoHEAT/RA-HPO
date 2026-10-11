"""
Compare two RA-HPO traces sample by sample.

An RA-HPO trace is a .npz written by ra_hpo.save_trace, holding:
    time_ns   (N,) float64, the time axis in nanoseconds
    amplitude (N,) float64, the trace

Usage
-----
    python tools/compare_traces.py REFERENCE.npz CANDIDATE.npz
    python tools/compare_traces.py ref.npz cand.npz --max-rel-l2 1e-9

By default the test is bitwise: the amplitudes must be identical, sample for sample. Pass
--max-rel-l2 to accept a small relative L2 difference instead, for example between runs whose
floating-point summation order differs.

Exit status
-----------
    0  traces pass
    1  traces differ beyond the tolerance, or cannot be compared
"""
from __future__ import annotations

import argparse
import sys

import numpy as np

REQUIRED_KEYS = ("time_ns", "amplitude")


def load_trace(path):
    """Load one RA-HPO .npz trace, failing loudly if it is not the expected shape."""
    data = np.load(path)
    missing = [k for k in REQUIRED_KEYS if k not in data]
    if missing:
        raise KeyError(f"{path}: missing key(s) {', '.join(missing)}")
    t = np.asarray(data["time_ns"])
    a = np.asarray(data["amplitude"])
    if t.ndim != 1 or a.ndim != 1:
        raise ValueError(f"{path}: expected 1-D arrays, got {t.shape} and {a.shape}")
    if t.size != a.size:
        raise ValueError(f"{path}: time_ns has {t.size} samples but amplitude has {a.size}")
    return t, a


def compare(ref_path, cand_path, max_rel_l2=0.0):
    """
    Compare two traces and print a short report.

    Returns True if the candidate is acceptable. The criterion is bitwise equality of the
    amplitudes unless max_rel_l2 > 0, in which case the relative L2 difference is used.
    """
    t_ref, a_ref = load_trace(ref_path)
    t_cand, a_cand = load_trace(cand_path)

    print(f"reference : {ref_path}  ({a_ref.size} samples)")
    print(f"candidate : {cand_path}  ({a_cand.size} samples)")

    # A length change means the padding or the frequency window moved, which is a real
    # difference and not something a tolerance should paper over.
    if a_ref.size != a_cand.size:
        print(f"  [FAIL] sample count differs: {a_ref.size} vs {a_cand.size}")
        return False

    # The time axis is derived from dt and the transform length, so a change here points at
    # the padding or the transform rather than at the physics. Report it separately.
    if not np.array_equal(t_ref, t_cand):
        dt_max = float(np.max(np.abs(t_ref - t_cand)))
        print(f"  [WARN] time axes are not bitwise equal (max |dt| = {dt_max:.3e} ns)")

    bitwise = np.array_equal(a_ref, a_cand)
    diff = a_cand - a_ref
    max_abs = float(np.max(np.abs(diff))) if diff.size else 0.0
    ref_norm = float(np.linalg.norm(a_ref))
    rel_l2 = float(np.linalg.norm(diff) / ref_norm) if ref_norm > 0.0 else float("inf")

    print(f"  bitwise identical : {bitwise}")
    print(f"  max |difference|  : {max_abs:.6e} V/m")
    print(f"  relative L2       : {rel_l2:.6e}")

    if bitwise:
        print("  [PASS] bitwise identical")
        return True
    if max_rel_l2 > 0.0 and rel_l2 <= max_rel_l2:
        print(f"  [PASS] within the accepted relative L2 of {max_rel_l2:.1e}")
        return True

    if max_rel_l2 > 0.0:
        print(f"  [FAIL] relative L2 {rel_l2:.6e} exceeds {max_rel_l2:.1e}")
    else:
        print("  [FAIL] not bitwise identical (pass --max-rel-l2 to allow a tolerance)")
    return False


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("reference", help="trace produced before the change")
    ap.add_argument("candidate", help="trace produced after the change")
    ap.add_argument("--max-rel-l2", type=float, default=0.0,
                    help="accept this relative L2 difference instead of requiring bitwise "
                         "equality (default 0, i.e. bitwise)")
    args = ap.parse_args()

    try:
        ok = compare(args.reference, args.candidate, args.max_rel_l2)
    except (OSError, KeyError, ValueError) as exc:
        print(f"  [FAIL] {exc}")
        sys.exit(1)
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
