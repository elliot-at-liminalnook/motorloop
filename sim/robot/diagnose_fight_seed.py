# SPDX-License-Identifier: MIT
"""Per-episode fighter benchmark diagnosis for a fixed seed."""

from __future__ import annotations

import argparse
import json
import os
import pickle
import sys
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

from train_adversarial import AdversarialEnv, load_opponent  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--seed", type=int, required=True)
    ap.add_argument("--episodes", type=int, default=8)
    ap.add_argument("--steps", type=int, default=80)
    ap.add_argument("--sep-lo", type=float, default=0.25)
    ap.add_argument("--sep-hi", type=float, default=0.70)
    ap.add_argument("--azimuth", type=float, default=3.14159)
    ap.add_argument("--lean-contacts", action="store_true")
    ap.add_argument("--engage-obs", action="store_true")
    ap.add_argument("--out-json", default="")
    args = ap.parse_args()

    infer = load_opponent(args.ckpt)
    env = AdversarialEnv(
        self_collision=not args.lean_contacts,
        frame_skip=5,
        sep_lo=args.sep_lo,
        sep_hi=args.sep_hi,
        azimuth=args.azimuth,
        striker=None,
        opponent="passive",
        engage_obs=args.engage_obs,
    )
    step = jax.jit(env.step)
    keys = jax.random.split(jax.random.PRNGKey(args.seed), args.episodes)
    rows = []
    for epi, key0 in enumerate(keys):
        key = key0
        state = env.reset(key)
        rel0 = (state.pipeline_state.xpos[env._Bt] - state.pipeline_state.xpos[env._At])[:2]
        sep0 = float(jnp.linalg.norm(rel0))
        theta0 = float(jnp.arctan2(rel0[1], rel0[0]))
        sums = {k: 0.0 for k in ("sparc", "dealt", "taken", "closing", "fleeing", "fire")}
        first_dealt_step = None
        min_dist = sep0
        final_dist = sep0
        done_step = None
        for t in range(args.steps):
            key, sk = jax.random.split(key)
            action, _ = infer(state.obs, sk)
            state = step(state, action)
            alive = 1.0 - float(state.done)
            metrics = state.metrics
            for k in sums:
                sums[k] += float(metrics[k]) * alive
            dist = float(metrics["dist"])
            min_dist = min(min_dist, dist)
            final_dist = dist
            if first_dealt_step is None and float(metrics["dealt"]) > 1e-6:
                first_dealt_step = t + 1
            if done_step is None and float(state.done) > 0.5:
                done_step = t + 1
        rows.append({
            "episode": epi,
            "sep0": sep0,
            "theta0": theta0,
            "rel0": [float(rel0[0]), float(rel0[1])],
            "min_dist": min_dist,
            "final_dist": final_dist,
            "first_dealt_step": first_dealt_step,
            "done_step": done_step,
            **sums,
        })
    rec = {
        "ckpt": args.ckpt,
        "seed": args.seed,
        "episodes": args.episodes,
        "steps": args.steps,
        "rows": rows,
        "mean": {k: float(np.mean([r[k] for r in rows])) for k in rows[0] if isinstance(rows[0][k], (int, float))},
    }
    text = json.dumps(rec, indent=2)
    print(text)
    if args.out_json:
        Path(args.out_json).write_text(text)


if __name__ == "__main__":
    os.environ.pop("MUJOCO_GL", None)
    main()
