# SPDX-License-Identifier: MIT
"""Evaluate + record the dodge-balance quadruped (notes/rl-dodge-balance-checklist §5).

Loads the torch policy BEFORE the mujoco env (heap-order fix), runs the
DodgeQuadrupedEnv, reports return + hit count, and records the full MuJoCo state
(qpos/qvel, incl. projectiles) for torch-free rendering by render_rollout.py.
  python eval_dodge.py [--model ...] [--random] [--difficulty 0.6] [--video --tag dodge]
"""

from __future__ import annotations

import argparse
import os
from pathlib import Path

os.environ.setdefault("MUJOCO_GL", "osmesa")

from stable_baselines3 import PPO  # noqa: E402  (torch first)
import numpy as np  # noqa: E402
# NOTE: dodge_env (mujoco) is imported lazily in main(), AFTER the torch model is
# loaded - mujoco loaded before torch tensor construction corrupts the heap.

OUT = Path(__file__).resolve().parents[2] / "sim" / "build" / "rl"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default=None)
    ap.add_argument("--random", action="store_true")
    ap.add_argument("--motor", default="db42s03")
    ap.add_argument("--difficulty", type=float, default=0.6)
    ap.add_argument("--episodes", type=int, default=5)
    ap.add_argument("--video", action="store_true")
    ap.add_argument("--max-steps", type=int, default=900)
    ap.add_argument("--tag", default="dodge")
    args = ap.parse_args()

    use_model = args.model and not args.random
    model = PPO.load(args.model) if use_model else None
    norm = None
    if use_model:
        import pickle
        vp = Path(args.model).with_name(Path(args.model).name
                                        .replace("ppo_", "vecnorm_").replace(".zip", ".pkl"))
        if vp.is_file():
            with open(vp, "rb") as f:
                vn = pickle.load(f)
            norm = (vn.obs_rms.mean, np.sqrt(vn.obs_rms.var + vn.epsilon), vn.clip_obs)

    from dodge_env import DodgeQuadrupedEnv   # lazy: mujoco AFTER the torch model
    env = DodgeQuadrupedEnv(motor=args.motor, difficulty=args.difficulty)
    returns, hits_per_ep = [], []
    qpos_rec, qvel_rec = [], []
    for ep in range(args.episodes):
        obs, _ = env.reset(seed=ep)
        done = trunc = False
        ret = 0.0; hits = 0; steps = 0
        while not (done or trunc) and steps < args.max_steps:
            if use_model:
                o = obs if norm is None else np.clip((obs - norm[0]) / norm[1],
                                                     -norm[2], norm[2])
                a = model.predict(o, deterministic=True)[0]
            else:
                a = env.action_space.sample()
            obs, r, done, trunc, info = env.step(a)
            ret += r; hits += int(info["hit"]); steps += 1
            if args.video and ep == 0:
                qpos_rec.append(env.data.qpos.copy())
                qvel_rec.append(env.data.qvel.copy())
        returns.append(ret); hits_per_ep.append(hits)
    print(f"{args.tag}: return mean={np.mean(returns):.1f} | hits/ep mean="
          f"{np.mean(hits_per_ep):.1f} | steps survived ~{steps}")
    if args.video and qpos_rec:
        OUT.mkdir(parents=True, exist_ok=True)
        p = OUT / f"{args.tag}_traj.npz"
        np.savez(p, qpos=np.array(qpos_rec), qvel=np.array(qvel_rec), robot="dodge")
        print(f"recorded {len(qpos_rec)} steps -> {p}")


if __name__ == "__main__":
    main()
