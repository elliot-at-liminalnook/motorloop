# SPDX-License-Identifier: MIT
"""Collect rollouts and train offline helpers from MuJoCo-Warp tensors."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import torch

from combat_warp_env import CombatWarpEnv
from mesh_warp_env import MeshWarpEnv
from train_mesh_warp import Actor, load_policy
from walker_warp_env import WalkerWarpEnv

ENVIRONMENTS = {"walker": WalkerWarpEnv, "mesh": MeshWarpEnv, "combat": CombatWarpEnv}


@torch.no_grad()
def collect(geometry="walker", checkpoint=None, steps=1000, nworld=16,
            seed=0, device=None):
    env = ENVIRONMENTS[geometry](nworld, seed=seed, device=device, episode_length=800)
    policy = (load_policy(checkpoint, env.obs_dim, env.act_dim, env.device)
              if checkpoint else lambda obs: torch.zeros((len(obs), env.act_dim), device=env.device))
    obs = env.reset()
    data = {key: [] for key in ("obs", "action", "next_obs", "reward", "done")}
    for _ in range(steps):
        action = policy(obs)
        next_obs, reward, done, _ = env.step(action)
        for key, value in (("obs", obs), ("action", action), ("next_obs", next_obs),
                           ("reward", reward), ("done", done)):
            data[key].append(value.detach().cpu())
        obs = next_obs
    return {key: torch.stack(value).numpy() for key, value in data.items()}


def train_behavior_clone(dataset, output, epochs=50, hidden="256,128", lr=3e-4,
                         device=None):
    device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
    raw = np.load(dataset)
    obs = torch.as_tensor(raw["obs"].reshape(-1, raw["obs"].shape[-1]),
                          dtype=torch.float32, device=device)
    action = torch.as_tensor(raw["action"].reshape(-1, raw["action"].shape[-1]),
                             dtype=torch.float32, device=device)
    sizes = tuple(map(int, hidden.split(",")))
    actor = Actor(obs.shape[-1], action.shape[-1], sizes).to(device)
    optimizer = torch.optim.Adam(actor.parameters(), lr=lr)
    for epoch in range(epochs):
        prediction = torch.tanh(actor(obs))
        loss = ((prediction - action) ** 2).mean()
        optimizer.zero_grad(); loss.backward(); optimizer.step()
    torch.save({"actor": actor.state_dict(), "obs_dim": obs.shape[-1],
                "act_dim": action.shape[-1], "hidden": hidden,
                "loss": float(loss.detach())}, output)
    return float(loss.detach())


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("mode", nargs="?", choices=("collect", "clone"), default="collect")
    parser.add_argument("--geometry", choices=tuple(ENVIRONMENTS), default="walker")
    parser.add_argument("--checkpoint", "--ckpt")
    parser.add_argument("--steps", type=int, default=1000)
    parser.add_argument("--envs", type=int, default=16)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--device")
    parser.add_argument("--dataset")
    parser.add_argument("--out", required=True)
    parser.add_argument("--epochs", type=int, default=50)
    args, _ = parser.parse_known_args(argv)
    if args.mode == "collect":
        np.savez_compressed(args.out, **collect(args.geometry, args.checkpoint, args.steps,
                                                args.envs, args.seed, args.device))
        print(f"saved {args.out}")
    else:
        if not args.dataset:
            parser.error("clone mode requires --dataset")
        loss = train_behavior_clone(args.dataset, args.out, args.epochs, device=args.device)
        print(f"saved {args.out} loss={loss:.6g}")


if __name__ == "__main__":
    main()
