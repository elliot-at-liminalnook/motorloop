# SPDX-License-Identifier: MIT
"""Throughput comparison across every active MuJoCo-Warp geometry."""

import argparse
import time

import torch

from combat_warp_env import CombatWarpEnv
from mesh_warp_env import MeshWarpEnv
from walker_warp_env import WalkerWarpEnv


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--envs", type=int, default=256)
    parser.add_argument("--steps", type=int, default=100)
    parser.add_argument("--device", default=None)
    args = parser.parse_args()
    for name, cls in (("walker", WalkerWarpEnv), ("mesh", MeshWarpEnv),
                      ("combat", CombatWarpEnv)):
        env = cls(args.envs, seed=0, device=args.device, episode_length=800)
        action = torch.zeros((args.envs, env.act_dim), device=env.device)
        for _ in range(5):
            env.step(action)
        if env.device.type == "cuda":
            torch.cuda.synchronize()
        start = time.time()
        for _ in range(args.steps):
            env.step(action)
        if env.device.type == "cuda":
            torch.cuda.synchronize()
        wall = time.time() - start
        print(f"RESULT geometry={name} device={env.device} "
              f"env_steps_per_s={args.envs * args.steps / wall:.1f}")


if __name__ == "__main__":
    main()
