# SPDX-License-Identifier: MIT
"""Train the dodge-balance quadruped with a difficulty curriculum
(notes/rl-dodge-balance-checklist.md §3-§4).

PPO + VecNormalize (the MuJoCo recipe). A curriculum ramps the threat difficulty
from 0 (balance only) to a target over the first part of training - dodge-from-
scratch with a falling robot rarely learns, so it learns to stand first.
  python train_dodge.py --steps 1500000 --n-envs 16 --max-difficulty 0.6
"""

from __future__ import annotations

import argparse
from pathlib import Path

# The MAIN process must NOT load mujoco (dodge_env imports it): mujoco loaded +
# torch tensor construction (the PPO policy) corrupts the heap. So only the SB3
# stack is imported here; mujoco is loaded lazily, inside the spawned env workers.
from stable_baselines3 import PPO  # noqa: E402
from stable_baselines3.common.callbacks import BaseCallback  # noqa: E402
from stable_baselines3.common.env_util import make_vec_env  # noqa: E402
from stable_baselines3.common.vec_env import SubprocVecEnv, VecNormalize  # noqa: E402

OUT = Path(__file__).resolve().parents[2] / "sim" / "build" / "rl"


def _make_env(motor="db42s03"):
    from dodge_env import make_dodge_env   # lazy: mujoco loads in the worker only
    return make_dodge_env(motor=motor, difficulty=0.0)


class Curriculum(BaseCallback):
    """Ramp threat difficulty 0 -> max over the first `ramp` fraction."""
    def __init__(self, total, max_d, ramp=0.6):
        super().__init__()
        self.total, self.max_d, self.ramp = total, max_d, ramp

    def _on_rollout_start(self):
        frac = self.num_timesteps / (self.total * self.ramp)
        d = min(self.max_d, frac * self.max_d)
        self.training_env.env_method("set_difficulty", d)

    def _on_step(self):
        return True


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--steps", type=int, default=1_500_000)
    ap.add_argument("--n-envs", type=int, default=16)
    ap.add_argument("--max-difficulty", type=float, default=0.6)
    ap.add_argument("--motor", default="db42s03")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--tag", default="dodge")
    args = ap.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)

    # 'spawn' (not fork): dodge_env loads mujoco at import, so a forked worker
    # would inherit torch+mujoco in the heap-corrupting order; spawn starts fresh
    # and loads torch (SB3 worker) before mujoco (the env).
    venv = make_vec_env(_make_env, n_envs=args.n_envs, seed=args.seed,
                        env_kwargs=dict(motor=args.motor),
                        vec_env_cls=SubprocVecEnv,
                        vec_env_kwargs=dict(start_method="spawn"))
    venv = VecNormalize(venv, norm_obs=True, norm_reward=True, clip_obs=10.0)
    model = PPO("MlpPolicy", venv, verbose=1, seed=args.seed,
                n_steps=2048, batch_size=256, n_epochs=10, gae_lambda=0.95,
                gamma=0.99, ent_coef=0.0, learning_rate=3e-4, clip_range=0.2,
                policy_kwargs=dict(net_arch=[256, 256]),
                tensorboard_log=str(OUT / "tb"))
    print(f"training {args.tag}: {args.steps:,} steps, curriculum 0->"
          f"{args.max_difficulty} (motor {args.motor})")
    model.learn(total_timesteps=args.steps,
                callback=Curriculum(args.steps, args.max_difficulty))
    model.save(OUT / f"ppo_{args.tag}.zip")
    venv.save(str(OUT / f"vecnorm_{args.tag}.pkl"))
    print(f"saved {OUT / f'ppo_{args.tag}.zip'} (+ vecnorm)")


if __name__ == "__main__":
    main()
