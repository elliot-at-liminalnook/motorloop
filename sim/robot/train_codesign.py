# SPDX-License-Identifier: MIT
"""Phase 1 — train a baseline policy on the MJX body with brax PPO (GPU).

  python train_codesign.py [--steps 3000000 --envs 2048]
Saves params to /root/proj/out/baseline.pkl and prints the learning curve.
"""

from __future__ import annotations

import argparse
import functools
import pickle
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from gen_robot_mjcf import build_mjcf, load_spec  # noqa: E402
from mjx_env import CodesignEnv  # noqa: E402
from brax.training.agents.ppo import train as ppo  # noqa: E402

OUT = Path("/root/proj/out"); OUT.mkdir(parents=True, exist_ok=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--steps", type=int, default=3_000_000)
    ap.add_argument("--envs", type=int, default=2048)
    ap.add_argument("--tag", default="baseline")
    args = ap.parse_args()

    xml = build_mjcf(load_spec(HERE / "robot.toml"))
    env = CodesignEnv(xml)
    print(f"obs={env.observation_size} act={env.action_size} envs={args.envs} steps={args.steps:,}")

    t0 = time.time(); curve = []
    def prog(step, m):
        r = float(m.get("eval/episode_reward", 0.0))
        curve.append((int(step), r))
        print(f"  step {int(step):>9,}  reward {r:8.2f}  ({time.time()-t0:5.0f}s)", flush=True)

    make_inf, params, _ = ppo.train(
        environment=env, num_timesteps=args.steps, num_evals=8, episode_length=200,
        num_envs=args.envs, batch_size=1024, num_minibatches=16, unroll_length=20,
        num_updates_per_batch=4, learning_rate=3e-4, entropy_cost=1e-2,
        discounting=0.97, reward_scaling=0.1, normalize_observations=True,
        seed=0, progress_fn=prog)

    with open(OUT / f"{args.tag}.pkl", "wb") as f:
        pickle.dump(params, f)
    print(f"saved {OUT / f'{args.tag}.pkl'} | first {curve[0][1]:.1f} -> last {curve[-1][1]:.1f}")


if __name__ == "__main__":
    main()
