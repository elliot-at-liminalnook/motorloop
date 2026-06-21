# SPDX-License-Identifier: MIT
"""Build-progress ledger — the single source of truth for the framework BUILD (not its runs).

The implementation is itself a multi-stage run, so it gets the same resume/backup property we give
training: a JSON ledger of phases + per-phase tar snapshots. `make fw-status / fw-snapshot /
fw-restore` drive this. To resume the build: read `status`, do the next phase, verify, snapshot.

  python -m arena._ledger status
  python -m arena._ledger seed                 # (re)create the ledger with all phases = todo
  python -m arena._ledger mark <phase> <todo|doing|done|verified> [snapshot]
"""

from __future__ import annotations

import json, sys, time
from pathlib import Path

HERE = Path(__file__).resolve().parent
LEDGER = HERE / "BUILD_STATE.json"

# (phase, title, verify_cmd) — verify_cmd uses $MJX_PY (the local CPU venv python) where it runs code.
PHASES = [
    (0, "Scaffold + build-resumability tooling",
     "$MJX_PY -c \"import json; json.load(open('sim/robot/arena/BUILD_STATE.json')); print('ledger ok')\""),
    (1, "Layer 0: trace spine (arena/trace.py)",
     "cd sim/robot && $MJX_PY -m arena.trace --selftest"),
    (2, "Kernel + watcher emit into the universal stream",
     "cd sim/robot && $MJX_PY -m arena.kernel_emit --selftest"),
    (3, "Layer 1: Stage (arena/stage.py) round-trips kernel CLIs",
     "cd sim/robot && $MJX_PY -m arena.stage --selftest"),
    (4, "Layer 2: unified engine + schedules (collapse the two drivers)",
     "cd sim/robot && $MJX_PY -m arena.engine --selftest"),
    (5, "Layer 3: Runner (Local + Pod lifecycle unified)",
     "cd sim/robot && $MJX_PY -m arena.runner --selftest"),
    (6, "Layer 4: Run/Pipeline sugar + observability",
     "cd sim/robot && $MJX_PY -m arena.run --selftest"),
    (7, "Migrate: arena.cli unified entrypoint + drivers point at arena",
     "cd sim/robot && $MJX_PY -m py_compile curriculum_drive.py selfplay_drive.py && $MJX_PY -m arena.cli --selftest"),
    (8, "PodRunner integration (offline) + real-GPU exercise = self-play transition",
     "cd sim/robot && $MJX_PY -m arena.pod_smoke --selftest"),
    (9, "Coach — automatic competency controller (adaptive reward, replaces brittle weights)",
     "cd sim/robot && $MJX_PY -m arena.coach --selftest"),
]


def load():
    if LEDGER.exists():
        return json.loads(LEDGER.read_text())
    return None


def seed(force=False):
    if LEDGER.exists() and not force:
        return load()
    led = {"created": time.strftime("%Y-%m-%dT%H:%M:%S"), "phases": [
        {"phase": p, "title": t, "verify_cmd": v, "status": "todo", "snapshot": None,
         "ts": None, "notes": None} for (p, t, v) in PHASES]}
    LEDGER.write_text(json.dumps(led, indent=2))
    return led


def mark(phase, status, snapshot=None, notes=None):
    led = load() or seed()
    for ph in led["phases"]:
        if ph["phase"] == int(phase):
            ph["status"] = status; ph["ts"] = time.strftime("%Y-%m-%dT%H:%M:%S")
            if snapshot is not None: ph["snapshot"] = snapshot
            if notes is not None: ph["notes"] = notes
    LEDGER.write_text(json.dumps(led, indent=2))
    return led


def status():
    led = load() or seed()
    mark_ch = {"todo": "·", "doing": "▸", "done": "○", "verified": "✓"}
    print(f"arena BUILD_STATE  (created {led['created']})")
    nxt = None
    for ph in led["phases"]:
        ch = mark_ch.get(ph["status"], "?")
        print(f"  [{ch}] phase {ph['phase']}  {ph['title']}")
        if nxt is None and ph["status"] != "verified":
            nxt = ph
    if nxt:
        print(f"\nNEXT: phase {nxt['phase']} — {nxt['title']}")
        print(f"  verify: {nxt['verify_cmd']}")
    else:
        print("\nALL PHASES VERIFIED ✓")
    return led


def verify_cmd(phase):
    led = load() or seed()
    for ph in led["phases"]:
        if ph["phase"] == int(phase):
            return ph["verify_cmd"]
    return ""


def main(argv):
    if not argv or argv[0] == "status":
        status()
    elif argv[0] == "seed":
        seed(force=True); status()
    elif argv[0] == "mark":
        mark(argv[1], argv[2], argv[3] if len(argv) > 3 else None); status()
    elif argv[0] == "verify-cmd":
        print(verify_cmd(argv[1]))
    else:
        print(__doc__)


if __name__ == "__main__":
    main(sys.argv[1:])
