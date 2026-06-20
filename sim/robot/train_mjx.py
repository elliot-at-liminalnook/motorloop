# SPDX-License-Identifier: MIT
"""MJX training skeleton for the parametric body (GPU target).

Loads the generated MJCF, defines the env logic (obs/reward) and the
domain-randomization hook, and sketches the MJX (JAX) vectorized PPO loop that runs
thousands of envs on one GPU. DR over the part parameters is the feature that makes
"swap a part" need *no* retrain - the policy already trained across the range.

Locally (no JAX) `--smoke` validates the env on the generated body with plain
MuJoCo, proving the training env is wired before you rent a GPU. On a CUDA box:
  pip install "jax[cuda12]" mujoco-mjx brax
  python train_mjx.py            # runs the MJX PPO loop
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

os.environ.setdefault("MUJOCO_GL", "osmesa")

import mujoco  # noqa: E402
import numpy as np  # noqa: E402

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from gen_robot_mjcf import build_mjcf, load_spec  # noqa: E402
import sparc_score as sparc  # noqa: E402

SPEC = load_spec(HERE / "robot.toml")


def make_model(overrides: dict | None = None) -> mujoco.MjModel:
    return mujoco.MjModel.from_xml_string(build_mjcf(SPEC, overrides))


def sample_dr(rng: np.random.Generator) -> dict:
    """Sample part parameters from [domain_randomization] -> generator overrides.
    Train across these and a part swap within range needs NO retrain. (Topology DR
    - leg count - is a regen, not a field randomization.)"""
    dr = SPEC["domain_randomization"]
    u = lambda lo_hi: float(rng.uniform(*lo_hi))
    return {
        "torso": {"mass": u(dr["torso_mass"])},
        "leg_defaults": {"thigh_len": u(dr["thigh_len"]),
                         "calf_len": u(dr["calf_len"]),
                         "joint_stiffness": u(dr["joint_stiffness"])},
        "actuator": {"gear": u(dr["gear"])},
    }


def obs(model, data) -> np.ndarray:
    """Proprioception: joint pos/vel + torso orientation/vel + height. (The dodge
    perception + attacker track from combat_env.py plugs in here for the real task.)"""
    return np.concatenate([
        data.qpos[7:], data.qvel[6:],         # actuated joints
        data.qpos[3:7], data.qvel[:6],        # torso quat + lin/ang vel
        [data.qpos[2]],                       # height
    ]).astype(np.float32)


def reward(model, data, opp=None) -> float:
    """SPARC-differential objective (sparc_score.step_reward): win the decision =
    maximize OUR points, minimize the opponent's. Stay-up + effort are shaping; the
    match terms (damage dealt/taken, aggression=closing-not-fleeing, control) come
    from `opp` (the attacker state) in the real self-play env. With no opponent
    (locomotion warm-up) only the shaping + a forward-aggression proxy apply."""
    up = 1.0 - 2.0 * (data.qpos[4] ** 2 + data.qpos[5] ** 2)   # torso upright (shaping)
    shaping = 1.0 * up - 0.001 * float(np.square(data.ctrl).sum())
    if opp is None:
        closing = max(0.0, float(data.qvel[0]))                # forward = proxy aggression
        return shaping + sparc.step_reward(closing=min(closing, 1.0))
    # self-play: dealt/taken from weapon<->body contacts, closing from rel velocity,
    # control from weapon-denial/position - all computed against `opp` on the GPU env.
    return shaping + sparc.step_reward(dealt=opp["dealt"], taken=opp["taken"],
                                       closing=opp["closing"], fleeing=opp["fleeing"],
                                       control=opp["control"])


def smoke():
    """Prove the training env on the generated body with plain MuJoCo (no GPU)."""
    rng = np.random.default_rng(0)
    model = make_model(sample_dr(rng))            # a domain-randomized body
    data = mujoco.MjData(model)
    mujoco.mj_resetData(model, data)
    data.qpos[2] = SPEC["torso"]["spawn_height"]
    mujoco.mj_forward(model, data)
    o0 = obs(model, data)
    moved, rsum = 0.0, 0.0
    for _ in range(200):
        data.ctrl[:] = rng.uniform(-1, 1, model.nu)            # exercise actuators
        mujoco.mj_step(model, data)
        rsum += reward(model, data)
    o1 = obs(model, data)
    moved = float(np.linalg.norm(data.qpos[:2]))
    print(f"smoke: DR body act_dim={model.nu}, obs_dim={o0.shape[0]}, "
          f"obs finite={np.isfinite(o1).all()}, torque moved torso {moved:.3f} m, "
          f"reward/step~{rsum/200:.2f}")
    print("env wired on the generated body. JAX/MJX absent -> run the loop on a GPU box.")


def train():
    try:
        import jax  # noqa: F401
        from mujoco import mjx  # noqa: F401
    except Exception:
        print("JAX/MJX not installed (this box is the local scaffold).")
        print("On a CUDA box:  pip install 'jax[cuda12]' mujoco-mjx brax")
        print("Then this runs the vectorized PPO loop below. Smoke-validating instead:\n")
        return smoke()
    # --- GPU path (sketch): put the generated model on device, vmap reset/step ---
    # mx = mjx.put_model(make_model())
    # batched DR: sample_dr per env -> randomize mx fields (body_mass, dof_damping,
    #   actuator forcerange, geom_friction) across the batch, then vmap the rollout.
    # Plug obs()/reward() (JAX-rewritten) into a brax/PPO trainer over N=4096 envs.
    raise SystemExit("MJX present: wire the brax PPO trainer here (see docstring).")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--smoke", action="store_true", help="validate env locally (no GPU)")
    ap.add_argument("--selfplay", action="store_true",
                    help="two-robot SPARC self-play match (match_env.py)")
    args = ap.parse_args()
    if args.selfplay:
        from match_env import selfplay
        selfplay()
    elif args.smoke:
        smoke()
    else:
        train()


if __name__ == "__main__":
    main()
