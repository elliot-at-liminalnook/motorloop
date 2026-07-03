#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
"""V.7a golden-trajectory fingerprint: a 4-number summary of a fixed rollout
(PRNGKey(0) reset, 50 zero-action steps) of the commanded PD env. Any silent
change to physics, model generation, reset semantics, PD mapping, or reward
moves at least one of the numbers; test_golden_traj.py pins them to a stored
golden JSON.

Pure CPU (JAX_PLATFORMS=cpu is forced before jax import) — the 50-step jit
takes a few minutes, which is expected.
"""
from __future__ import annotations

import os
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

# environment must be pinned BEFORE commanded_env/jax import: the fingerprint
# is only meaningful for the pd control mode on the CPU backend.
os.environ.setdefault("MUJOCO_GL", "")
os.environ.setdefault("JAX_PLATFORMS", "cpu")
os.environ.setdefault("CODESIGN_OUT", "/tmp/v67out")
os.environ["CMD_CONTROL_MODE"] = "pd"

N_STEPS = 50
COMMAND = (0.4, 0.0)   # forward walk command, deploy-style (remote=True, no resample)


def fingerprint(env_name: str = "commanded_pd") -> dict:
    """Deterministic trajectory fingerprint, every field rounded to 6 decimals."""
    if env_name != "commanded_pd":
        raise ValueError(f"unknown env name {env_name!r} (only 'commanded_pd')")
    import jax
    import jax.numpy as jnp
    import numpy as np
    import commanded_env as C

    assert C.CMD_CONTROL_MODE == "pd", (
        "commanded_env was imported with CMD_CONTROL_MODE="
        f"{C.CMD_CONTROL_MODE!r} before golden_traj could pin 'pd'"
    )
    env = C._build()()
    key = jax.random.PRNGKey(0)
    state = env.reset_with_command(key, jnp.array(COMMAND))
    step = jax.jit(env.step)
    zero = jnp.zeros(env.action_size)
    reward_sum = 0.0
    for _ in range(N_STEPS):
        state = step(state, zero)
        reward_sum += float(state.reward)
    qpos = np.asarray(state.pipeline_state.qpos, dtype=np.float64)
    return {
        "qpos_sum": round(float(qpos.sum()), 6),
        "qpos_abs_sum": round(float(np.abs(qpos).sum()), 6),
        "final_z": round(float(qpos[2]), 6),
        "reward_sum": round(float(reward_sum), 6),
    }


if __name__ == "__main__":
    import json
    print(json.dumps(fingerprint(), indent=2, sort_keys=True))
