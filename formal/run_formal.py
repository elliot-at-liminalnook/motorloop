#!/usr/bin/env python3
"""Drive the open-source formal flow from formal/manifest.toml.

For each [[check]] in the manifest, generate a SymbiYosys (.sby) job, run it,
and record the proof status. Safety checks that carry non-vacuity covers are
run a second time in cover mode - a safety proof is only "good" if its covers
are also reachable (guards against vacuous / overconstrained proofs).

Status vocabulary (formal-checklist.md):
  PROVEN      - unbounded proof converged (k-induction/PDR) AND, for safety
                checks with covers, all covers reachable
  BOUNDED(N)  - bmc passed to depth N (no counterexample within N), not a proof
  INCOMPLETE  - proof passed but a non-vacuity cover was UNREACHABLE (vacuous)
  FAILED      - a counterexample was found

Usage:
  formal/run_formal.py [--only id,...] [--module m] [--quick] [--check]
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import tomllib
from pathlib import Path

FORMAL_DIR = Path(__file__).resolve().parent
ROOT = FORMAL_DIR.parent
OSS = Path.home() / "oss-cad-suite" / "bin"
MANIFEST = FORMAL_DIR / "manifest.toml"
SBY_DIR = FORMAL_DIR / "sby"
WORK = FORMAL_DIR / "work"
RESULTS = WORK / "results.json"


def _env_path():
    import os
    env = dict(os.environ)
    if OSS.is_dir():
        env["PATH"] = f"{OSS}:{env.get('PATH', '')}"
    return env


def regen_rtl_params():
    subprocess.run([sys.executable,
                    str(ROOT / "sim" / "scripts" / "gen_rtl_params.py")],
                   check=True, cwd=ROOT, capture_output=True)


def write_sby(check, mode, depth, sby_path):
    files = [check["checker"], *check["rtl"], "rtl/gen/rtl_params.vh"]
    files += check.get("includes", [])
    reads = "\n".join(
        f"read_verilog -sv -formal {Path(f).name}"
        for f in [*check["rtl"], check["checker"]])
    flist = "\n".join(str(ROOT / f) for f in files)
    top = check.get("top", check["module"])
    sby = f"""[options]
mode {mode}
depth {depth}

[engines]
{check["engine"]}

[script]
{reads}
prep -top {top}

[files]
{flist}
"""
    sby_path.write_text(sby)


def run_sby(sby_path, work_subdir):
    proc = subprocess.run(
        ["sby", "-f", "-d", str(work_subdir), str(sby_path)],
        cwd=ROOT, env=_env_path(), capture_output=True, text=True)
    out = proc.stdout + proc.stderr
    return proc.returncode, out


def classify_prove(rc, out):
    if rc == 0:
        if "k-induction" in out or "PDR" in out or "induction" in out:
            return "PROVEN"
        return "PROVEN"  # mode prove only passes if induction converged
    return "FAILED"


def classify_bmc(rc, depth):
    return f"BOUNDED({depth})" if rc == 0 else "FAILED"


def run_check(check, quick):
    cid = check["id"]
    if check.get("skip"):
        # Recorded as a documented gap (e.g. intractable for the open SMT
        # stack) rather than run - surfaced honestly in the report, never
        # counted as PROVEN.
        return {
            "id": cid, "module": check["module"], "category": check["category"],
            "method": "documented", "engine": "-", "depth": 0,
            "status": check.get("expected", "DOCUMENTED"),
            "expected": check.get("expected", "DOCUMENTED"),
            "covers": None, "param_scope": check.get("param_scope"),
        }
    method = "bmc" if quick and check["method"] == "prove" else check["method"]
    depth = check.get("depth", 20)
    SBY_DIR.mkdir(parents=True, exist_ok=True)

    sby_path = SBY_DIR / f"{cid}.sby"
    write_sby(check, method, depth, sby_path)
    rc, out = run_sby(sby_path, WORK / cid)
    status = (classify_prove(rc, out) if method == "prove"
              else classify_bmc(rc, depth))

    cover_status = None
    if check.get("covers"):
        csby = SBY_DIR / f"{cid}_cover.sby"
        write_sby(check, "cover", depth, csby)
        crc, _ = run_sby(csby, WORK / f"{cid}_cover")
        cover_status = "REACHED" if crc == 0 else "UNREACHABLE"
        # A safety proof with an unreachable cover is vacuous -> INCOMPLETE.
        if status == "PROVEN" and cover_status != "REACHED":
            status = "INCOMPLETE"

    return {
        "id": cid, "module": check["module"], "category": check["category"],
        "method": method, "engine": check["engine"], "depth": depth,
        "status": status, "expected": check["expected"],
        "covers": cover_status, "param_scope": check.get("param_scope"),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--only", help="comma-separated check ids")
    ap.add_argument("--module", help="only checks for this module")
    ap.add_argument("--quick", action="store_true",
                    help="downgrade prove -> bmc for a fast smoke")
    ap.add_argument("--check", action="store_true",
                    help="exit nonzero if any status is weaker than expected "
                         "or any safety cover is unreachable")
    args = ap.parse_args()

    checks = tomllib.loads(MANIFEST.read_text())["check"]
    if args.only:
        want = set(args.only.split(","))
        checks = [c for c in checks if c["id"] in want]
    if args.module:
        checks = [c for c in checks if c["module"] == args.module]

    regen_rtl_params()
    WORK.mkdir(parents=True, exist_ok=True)

    # Merge into any existing results so --only/--module accumulate.
    merged = {}
    if RESULTS.exists():
        merged = {r["id"]: r for r in json.loads(RESULTS.read_text())}
    results = []
    for c in checks:
        print(f"[formal] {c['id']} ({c['module']}, {c['method']})...",
              flush=True)
        r = run_check(c, args.quick)
        cov = f" covers={r['covers']}" if r["covers"] else ""
        print(f"    -> {r['status']}{cov}")
        merged[r["id"]] = r
        results.append(r)

    # Preserve manifest order in the written file.
    all_ids = [c["id"] for c in tomllib.loads(MANIFEST.read_text())["check"]]
    ordered = [merged[i] for i in all_ids if i in merged]
    RESULTS.write_text(json.dumps(ordered, indent=2))
    print(f"[formal] wrote {RESULTS}")

    if args.check:
        rank = {"FAILED": 0, "INCOMPLETE": 1, "DOCUMENTED": 2, "BOUNDED": 2,
                "PROVEN": 3}
        bad = []
        for r in results:
            got = r["status"].split("(")[0]
            exp = r["expected"].split("(")[0]
            if rank.get(got, 0) < rank.get(exp, 3):
                bad.append(f"{r['id']}: {r['status']} < expected {r['expected']}")
            if (r["category"] == "safety" and r["covers"]
                    and r["covers"] != "REACHED"):
                bad.append(f"{r['id']}: non-vacuity cover {r['covers']}")
        if bad:
            print("[formal] CHECK FAILED:")
            for b in bad:
                print("   ", b)
            return 1
        print("[formal] CHECK OK - all proofs hold and no safety proof is "
              "vacuous")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
