"""
Commands that reproduce a case of the paper.

For a case of cases.py the script prints the gprMax runs of its reference plane(s) and of its
full-domain reference, the conversion of the plane, and the RA-HPO command. With --run it
executes the RA-HPO command (the planes must already exist). Every command is meant to be run
from the repository root.

Usage
-----
    python tools/run_case.py --list
    python tools/run_case.py finite_planar                 # print the steps
    python tools/run_case.py finite_planar --run --cores 8  # run RA-HPO
    python tools/run_case.py --commands --group roughness  # RA-HPO commands of a group, one per line
    python tools/run_case.py --write-json                  # configs/cases.json
"""
from __future__ import annotations

import argparse
import json
import shlex
import subprocess
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from examples.cases import (CASES, BY_KEY, MODELS, PLANE_RES, REPO, Case,  # noqa: E402
                            plane_input, trace_file)


def rel(p: Path) -> str:
    return str(Path(p).relative_to(REPO)) if Path(p).is_relative_to(REPO) else str(p)


def solver_command(c: Case, cores: int) -> list[str]:
    return (["python", "ra_hpo.py",
             "--tx_plane_file", rel(plane_input(c, "tx")),
             "--rx_plane_file", rel(plane_input(c, "rx")),
             "--target_res", f"{c.target_res:g}"]
            + list(c.flags)
            + ["--cores", str(cores), "--no_viz", "--output_name", c.key])


def plane_models(c: Case) -> list[str]:
    return [c.plane_tx] + ([c.plane_rx] if c.plane_rx else [])


def steps(c: Case, cores: int) -> list[str]:
    out = [f"# {c.label}  [{c.group}]"]
    out.append("# 1. reference plane(s): gprMax, then conversion")
    for m in plane_models(c):
        out.append(f"python -m gprMax {rel(MODELS / m)}.in")
        if not c.raw_plane:
            out.append(f"python tools/convert_plane.py --in {rel(MODELS / m)}.out "
                       f"--out {rel(plane_input(Case('_', '', '', m)))} --target_res {PLANE_RES:g}")
    if c.fdtd:
        out.append("# 2. full-domain reference (gprMax)")
        for m in (c.fdtd, c.direct):
            if m:
                out.append(f"python -m gprMax {rel(MODELS / m)}.in")
    out.append("# 3. RA-HPO, writes " + rel(trace_file(c.key)))
    out.append(shlex.join(solver_command(c, cores)))
    return out


def as_dict(c: Case) -> dict:
    return {"label": c.label, "group": c.group,
            "reference_plane_tx": f"gprMax_models/{c.plane_tx}.in",
            "reference_plane_rx": f"gprMax_models/{c.plane_rx or c.plane_tx}.in",
            "full_domain_model": f"gprMax_models/{c.fdtd}.in" if c.fdtd else None,
            "direct_wave_model": f"gprMax_models/{c.direct}.in" if c.direct else None,
            "solver_flags": ["--target_res", f"{c.target_res:g}"] + list(c.flags),
            "plane_file": "gprMax .out" if c.raw_plane else "converted (convert_plane.py)",
            "trace": rel(trace_file(c.key)), "figure": c.figure}


def main() -> int:
    ap = argparse.ArgumentParser(description="Commands that reproduce a case of the paper.")
    ap.add_argument("case", nargs="?", help="case key (see --list)")
    ap.add_argument("--list", action="store_true", help="list the cases")
    ap.add_argument("--commands", action="store_true", help="print the RA-HPO commands of a group")
    ap.add_argument("--group", help="restrict --list or --commands to one group")
    ap.add_argument("--run", action="store_true", help="run the RA-HPO command of the case")
    ap.add_argument("--cores", type=int, default=24)
    ap.add_argument("--write-json", action="store_true", help="write configs/cases.json")
    a = ap.parse_args()

    selected = [c for c in CASES if not a.group or c.group == a.group]
    if a.write_json:
        p = REPO / "configs" / "cases.json"
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(json.dumps({c.key: as_dict(c) for c in CASES}, indent=2) + "\n")
        print(f"wrote {rel(p)} with {len(CASES)} cases")
        return 0
    if a.commands:
        for c in selected:
            print(shlex.join(solver_command(c, a.cores)))
        return 0
    if a.list or not a.case:
        print(f"{'case':30s} {'group':12s} label")
        for c in selected:
            print(f"{c.key:30s} {c.group:12s} {c.label}")
        return 0
    if a.case not in BY_KEY:
        print(f"unknown case '{a.case}'; see --list", file=sys.stderr)
        return 2
    c = BY_KEY[a.case]
    if a.run:
        cmd = solver_command(c, a.cores)
        missing = [p for p in {plane_input(c, "tx"), plane_input(c, "rx")} if not p.exists()]
        if missing:
            print("missing reference plane(s); run the steps of: python tools/run_case.py "
                  f"{c.key}\n  " + "\n  ".join(rel(p) for p in missing), file=sys.stderr)
            return 3
        print(shlex.join(cmd))
        return subprocess.call(cmd, cwd=REPO)
    print("\n".join(steps(c, a.cores)))
    return 0


if __name__ == "__main__":
    sys.exit(main())
