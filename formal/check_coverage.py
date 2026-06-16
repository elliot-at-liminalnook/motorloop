#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
"""Coverage gate (robotics-ip-checklist stage 2): every reusable block in
cores/ must be EITHER formally proven (formal/manifest.toml) OR explicitly
declared sim-only (formal/sim_only.toml). Fails loudly on an undeclared block,
so a module cannot quietly ship "reusable" with no stated verification.

  python3 formal/check_coverage.py
"""

from __future__ import annotations

import sys
import tomllib
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
MANIFEST = ROOT / "formal" / "manifest.toml"
SIM_ONLY = ROOT / "formal" / "sim_only.toml"

# The system core (motorloop.core) is verified by composition, not as a leaf.
SYSTEM = {"motorloop"}


def core_block_names() -> set[str]:
    # Leaf IP cores live at the repo root, one per block; skip the system core.
    return {core.stem for core in ROOT.glob("*.core")
            if core.stem not in SYSTEM}


def main():
    blocks = core_block_names()
    manifest = tomllib.loads(MANIFEST.read_text())
    proven = {c["module"] for c in manifest.get("check", [])}
    proven |= {c["id"] for c in manifest.get("check", [])}
    sim_only = set(tomllib.loads(SIM_ONLY.read_text()).keys())

    declared = proven | sim_only
    undeclared = sorted(b for b in blocks if b not in declared)

    print(f"[coverage] {len(blocks)} leaf cores; "
          f"{len(blocks & proven)} proven, {len(blocks & sim_only)} sim-only")
    if undeclared:
        for b in undeclared:
            print(f"  UNDECLARED: {b} (add a manifest proof or a "
                  f"formal/sim_only.toml entry)")
        sys.exit(1)
    # Reverse check: sim_only entries that are actually also proven (stale).
    stale = sorted(sim_only & proven)
    for b in stale:
        print(f"  note: '{b}' is in sim_only.toml but is also proven - "
              f"consider removing the sim-only declaration")
    print("[coverage] OK - every reusable block has a stated verification")


if __name__ == "__main__":
    main()
