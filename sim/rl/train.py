# SPDX-License-Identifier: MIT
"""Train a policy on the motor-enveloped locomotion env with SB3 PPO
(notes/rl-environment-checklist.md §3).

The NN is a high-level policy over the FOC inner loop (actuator.py): it outputs
per-joint commands that the motor envelope realizes. Vectorized CPU training.
Step budget is a CLI arg - report it honestly (a CPU demo, not a converged SOTA
policy). Usage:
  python train.py --robot Ant-v5 --motor db42s03 --steps 600000 --n-envs 12
"""

from __future__ import annotations

import argparse
import os
from pathlib import Path

os.environ.setdefault("MUJOCO_GL", "osmesa")

from foc_env import make_env  # noqa: E402
from stable_baselines3 import PPO  # noqa: E402
from stable_baselines3.common.env_util import make_vec_env  # noqa: E402
from stable_baselines3.common.vec_env import (SubprocVecEnv,  # noqa: E402
                                              VecNormalize)

OUT = Path(__file__).resolve().parents[2] / "sim" / "build" / "rl"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--robot", default="HalfCheetah-v5")
    ap.add_argument("--motor", default="db42s03")
    ap.add_argument("--gear", type=float, default=2.0)
    ap.add_argument("--steps", type=int, default=600_000)
    ap.add_argument("--n-envs", type=int, default=12)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--tag", default=None)
    args = ap.parse_args()
    tag = args.tag or f"{args.robot}_{args.motor}"
    OUT.mkdir(parents=True, exist_ok=True)

    venv = make_vec_env(
        lambda: make_env(args.robot, args.motor, args.gear),
        n_envs=args.n_envs, seed=args.seed, vec_env_cls=SubprocVecEnv)
    # VecNormalize (obs + reward) is essential for PPO on MuJoCo locomotion.
    venv = VecNormalize(venv, norm_obs=True, norm_reward=True, clip_obs=10.0)
    model = PPO("MlpPolicy", venv, verbose=1, seed=args.seed,
                n_steps=2048, batch_size=256, n_epochs=10, gae_lambda=0.95,
                gamma=0.99, ent_coef=0.0, learning_rate=3e-4, clip_range=0.2,
                policy_kwargs=dict(net_arch=[256, 256]),
                tensorboard_log=str(OUT / "tb"))
    print(f"training {tag}: {args.steps:,} steps x {args.n_envs} envs (motor "
          f"envelope = {args.motor}, gear {args.gear})")
    model.learn(total_timesteps=args.steps, progress_bar=False)
    model.save(OUT / f"ppo_{tag}.zip")
    venv.save(str(OUT / f"vecnorm_{tag}.pkl"))   # eval needs the obs stats
    print(f"saved {OUT / f'ppo_{tag}.zip'} (+ vecnorm)")


if __name__ == "__main__":
    main()
