#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
"""V.7a golden-trajectory regression test for the commanded PD env.

First run (no golden on disk): computes the fingerprint, WRITES
golden_commanded_pd.json, and passes with a "golden created" note.
Every later run recomputes the fingerprint and pins each field to the golden
within rtol 1e-3 — a mismatch means physics/obs semantics drifted.

CPU-only and slow to jit (a few minutes) by design: determinism beats speed
for a drift tripwire.
"""
from __future__ import annotations

import json
import os
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

os.environ.setdefault("MUJOCO_GL", "")
os.environ.setdefault("CODESIGN_OUT", "/tmp/v67out")

GOLDEN = HERE / "golden_commanded_pd.json"
RTOL = 1e-3
FIELDS = ("qpos_sum", "qpos_abs_sum", "final_z", "reward_sum")


def test_golden_commanded_pd():
    from golden_traj import fingerprint
    import numpy as np

    fp = fingerprint("commanded_pd")
    assert set(fp) == set(FIELDS), f"fingerprint fields changed: {sorted(fp)}"

    if not GOLDEN.exists():
        GOLDEN.write_text(json.dumps(fp, indent=2, sort_keys=True) + "\n")
        print(f"golden created: {GOLDEN} -> {fp}")
        return

    golden = json.loads(GOLDEN.read_text())
    assert set(golden) == set(FIELDS), (
        f"golden file {GOLDEN} has unexpected fields {sorted(golden)} — "
        "delete it and rerun to regenerate"
    )
    for k in FIELDS:
        want, got = float(golden[k]), float(fp[k])
        assert np.isclose(got, want, rtol=RTOL, atol=1e-6), (
            f"golden trajectory mismatch on {k!r}: golden={want} current={got} "
            f"(rtol {RTOL}) — physics/obs semantics drifted — if intentional, "
            f"delete the golden ({GOLDEN.name}) and rerun"
        )
    print(f"golden trajectory holds: {fp}")


if __name__ == "__main__":
    test_golden_commanded_pd()
