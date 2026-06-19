# SPDX-License-Identifier: MIT
"""Evaluate + render a trained policy (notes/rl-environment-checklist.md §4).

Reports mean episode return and writes a headless rollout video (MUJOCO_GL=osmesa)
to figures/rl/. With --random it renders the untrained baseline for a before/after
comparison. Usage:
  python eval.py --model sim/build/rl/ppo_Ant-v5_db42s03.zip --episodes 5 --video
"""

from __future__ import annotations

import argparse
import os
from pathlib import Path

os.environ.setdefault("MUJOCO_GL", "osmesa")

# Import the torch/SB3 stack BEFORE mujoco/gymnasium: creating a mujoco context
# and then loading torch in the same process corrupts the heap (native lib order).
from stable_baselines3 import PPO  # noqa: E402
import numpy as np  # noqa: E402
from foc_env import make_env  # noqa: E402

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "sim" / "build" / "rl"


def rollout(env, policy, episodes, record=False, max_steps=900):
    """Run episodes for return stats; if record, also capture the first
    episode's MuJoCo state (qpos/qvel) per step for offline rendering (rendering
    in-process alongside torch is fragile - render_rollout.py does it separately)."""
    returns = []
    traj_qpos, traj_qvel = [], []
    for ep in range(episodes):
        obs, _ = env.reset(seed=ep)
        done = trunc = False
        ret = 0.0
        while not (done or trunc):
            a = policy(obs)
            obs, r, done, trunc, _ = env.step(a)
            ret += r
            if record and ep == 0 and len(traj_qpos) < max_steps:
                d = env.unwrapped.data
                traj_qpos.append(d.qpos.copy()); traj_qvel.append(d.qvel.copy())
        returns.append(ret)
    return np.array(returns), np.array(traj_qpos), np.array(traj_qvel)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--robot", default="HalfCheetah-v5")
    ap.add_argument("--motor", default="db42s03")
    ap.add_argument("--gear", type=float, default=2.0)
    ap.add_argument("--model", default=None)
    ap.add_argument("--random", action="store_true")
    ap.add_argument("--episodes", type=int, default=5)
    ap.add_argument("--video", action="store_true")
    ap.add_argument("--tag", default=None)
    args = ap.parse_args()

    # Load the torch policy BEFORE creating the mujoco env: a live mujoco context
    # followed by torch tensor construction (PPO.load) corrupts the heap.
    use_model = args.model and not args.random
    model = PPO.load(args.model) if use_model else None
    norm = None
    if use_model:
        import pickle
        vpath = Path(args.model).with_name(
            Path(args.model).name.replace("ppo_", "vecnorm_").replace(".zip", ".pkl"))
        if vpath.is_file():
            with open(vpath, "rb") as f:
                vn = pickle.load(f)          # VecNormalize: obs_rms, clip, epsilon
            norm = (vn.obs_rms.mean, np.sqrt(vn.obs_rms.var + vn.epsilon),
                    vn.clip_obs)
    env = make_env(args.robot, args.motor, args.gear)

    if use_model:
        def policy(o):
            if norm is not None:
                o = np.clip((o - norm[0]) / norm[1], -norm[2], norm[2])
            return model.predict(o, deterministic=True)[0]
        tag = args.tag or Path(args.model).stem
    else:
        def policy(o):
            return env.action_space.sample()
        tag = args.tag or f"{args.robot}_{args.motor}_random"

    returns, qpos, qvel = rollout(env, policy, args.episodes, record=args.video)
    print(f"{tag}: return mean={returns.mean():.1f} std={returns.std():.1f} "
          f"min={returns.min():.1f} max={returns.max():.1f}")
    if args.video and len(qpos):
        OUT.mkdir(parents=True, exist_ok=True)
        traj = OUT / f"{tag}_traj.npz"
        np.savez(traj, qpos=qpos, qvel=qvel, robot=args.robot)
        print(f"recorded {len(qpos)} steps -> {traj}")
        print(f"render with: MUJOCO_GL=osmesa python sim/rl/render_rollout.py "
              f"--traj {traj} --tag {tag}")


if __name__ == "__main__":
    main()
