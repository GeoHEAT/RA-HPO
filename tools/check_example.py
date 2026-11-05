"""Check the worked-example trace against the stored reference trace.

    python tools/check_example.py examples/example_reference.npz output/traces/example.npz [--max-chi 1.0]

Both traces are interpolated onto one grid over their common time span. Prints the relative
misfit chi = ||c - f|| / ||f|| (f the reference), the peak ratio and the peak-time difference,
and exits non-zero if chi exceeds --max-chi percent.
"""
import argparse
import sys

import numpy as np


def load(path):
    d = np.load(path)
    return np.asarray(d["time_ns"], float), np.asarray(d["amplitude"], float)


def main() -> int:
    ap = argparse.ArgumentParser(description="Check the worked-example trace against the reference.")
    ap.add_argument("reference", help="stored reference trace (.npz)")
    ap.add_argument("trace", help="trace to check (.npz)")
    ap.add_argument("--max-chi", type=float, default=1.0, help="acceptance, percent")
    a = ap.parse_args()
    tf, f = load(a.reference)
    tc, c = load(a.trace)
    t = np.linspace(max(tf[0], tc[0]), min(tf[-1], tc[-1]), 20001)
    fi, ci = np.interp(t, tf, f), np.interp(t, tc, c)
    chi = 100 * np.linalg.norm(ci - fi) / np.linalg.norm(fi)
    pr = np.max(np.abs(fi)) / np.max(np.abs(ci))
    dtp = t[np.argmax(np.abs(ci))] - t[np.argmax(np.abs(fi))]
    ok = chi <= a.max_chi
    print(f"example check: chi {chi:.4f} %  peak ratio (reference/trace) {pr:.5f}  peak-time diff {dtp:+.4f} ns  "
          f"over {t[0]:.1f}-{t[-1]:.1f} ns -> {'PASS' if ok else 'FAIL'} (max {a.max_chi} %)")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
