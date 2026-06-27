# SPDX-License-Identifier: MIT
"""Verify the shared CPG teacher is exactly what CommandedEnv deploys."""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

import jax
import jax.numpy as jnp

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from commanded_env import CPG_RESIDUAL_SCALE, PD_SCALE, VMAX, _build  # noqa: E402
from cpg_teacher import cpg_pd_step_target  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--cmd", default="0.35,0.0")
    ap.add_argument("--tol", type=float, default=1e-6)
    args = ap.parse_args()

    cmd = jnp.asarray([float(x.strip()) for x in args.cmd.replace(";", ",").split(",") if x.strip()])
    if cmd.shape != (2,):
        raise ValueError("--cmd expects vx,vy")
    Env = _build()
    env = Env()
    key = jax.random.PRNGKey(args.seed)
    st = env.reset_with_command(key, cmd)
    raw = jax.random.uniform(key, (env.action_size,), minval=-1.0, maxval=1.0)
    target, motor_action, prior = cpg_pd_step_target(
        env._stand, env._jr, st.info["phase"], cmd, raw, env._cpg_idx, env.action_size,
        VMAX, CPG_RESIDUAL_SCALE, PD_SCALE, directional=env._cpg,
        prev_command=st.info["prev_cmd"], xp=jnp)
    st2 = jax.jit(env.step)(st, raw)
    err_action = float(jnp.max(jnp.abs(st2.info["prev_action"] - motor_action)))
    target2 = jnp.clip(env._stand + PD_SCALE * st2.info["prev_action"], env._jr[:, 0], env._jr[:, 1])
    err_target = float(jnp.max(jnp.abs(target2 - target)))
    ok = err_action <= args.tol and err_target <= args.tol and bool(jnp.all(jnp.isfinite(prior)))
    print(f"teacher_action_max_err={err_action:.3e} teacher_target_max_err={err_target:.3e} ok={ok}")
    if not ok:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
