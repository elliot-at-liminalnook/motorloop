# SPDX-License-Identifier: MIT
"""Materialize a locomotion/scaffold checkpoint as a combat-compatible fighter seed.

`train_adversarial.warm_start()` already knows how to pad locomotion observations
and grow the action head for striker DOFs. This utility makes that conversion an
explicit artifact so the same scaffold can be benchmarked and rendered before
combat curriculum training.
"""

from __future__ import annotations

import argparse
import json
import os
import pickle
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from train_adversarial import AdversarialEnv, warm_start  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", required=True, help="source locomotion/fighter checkpoint")
    ap.add_argument("--out", required=True, help="combat-compatible output checkpoint")
    ap.add_argument("--frame-skip", type=int, default=5)
    ap.add_argument("--lean-contacts", action="store_true")
    ap.add_argument("--no-striker", action="store_true")
    ap.add_argument("--bench-sep-lo", type=float, default=0.25)
    ap.add_argument("--bench-sep-hi", type=float, default=0.7)
    ap.add_argument("--bench-az", type=float, default=3.14159)
    args = ap.parse_args()

    src = Path(args.src)
    if not src.exists():
        raise SystemExit(f"missing source checkpoint: {src}")
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)

    striker = False if args.no_striker else None
    env = AdversarialEnv(
        self_collision=not args.lean_contacts,
        frame_skip=args.frame_skip,
        sep_lo=args.bench_sep_lo,
        sep_hi=args.bench_sep_hi,
        azimuth=args.bench_az,
        striker=striker,
    )
    params = warm_start(str(src), env.observation_size, env.action_size)
    if params is None:
        raise SystemExit(f"could not convert checkpoint for obs={env.observation_size} act={env.action_size}: {src}")
    with open(out, "wb") as f:
        pickle.dump(params, f)

    rec = {
        "src": str(src),
        "out": str(out),
        "obs": int(env.observation_size),
        "act": int(env.action_size),
        "striker": bool(env._has_striker),
        "lean_contacts": bool(args.lean_contacts),
        "frame_skip": int(args.frame_skip),
    }
    sidecar = out.with_suffix(".json")
    sidecar.write_text(json.dumps(rec, indent=2))
    print(
        f"[prepare-fighter-seed] {src.name} -> {out} "
        f"obs={rec['obs']} act={rec['act']} striker={int(rec['striker'])}",
        flush=True,
    )


if __name__ == "__main__":
    # Keep render-only GL settings from leaking into a non-rendering MJX import.
    if os.environ.get("MUJOCO_GL") == "":
        os.environ.pop("MUJOCO_GL", None)
    main()
