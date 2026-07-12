# SPDX-License-Identifier: MIT
"""Random/CEM-style searches evaluated through actual MuJoCo-Warp rollouts."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import numpy as np
import torch

from codesign_warp_env import CodesignWarpEnv
from cpg_teacher import DEFAULT_RAW, cpg_action, decode_params
from train_mesh_warp import load_policy
from walker_warp_env import WalkerWarpEnv


@torch.no_grad()
def score_cpg(raw, steps=120, nworld=8, seed=0, device=None):
    env = WalkerWarpEnv(nworld, seed=seed, device=device, episode_length=None)
    params = decode_params(np.asarray(raw), xp=np)
    indices = np.arange(12).reshape(4, 3)
    start = env.qpos[:, 0].clone()
    reward = torch.zeros(nworld, device=env.device)
    for step in range(steps):
        phase = 2.0 * math.pi * step * env._dt * float(params.freq)
        action = cpg_action(phase, params, indices, 12, xp=np)
        actions = torch.as_tensor(action, dtype=torch.float32, device=env.device).expand(nworld, -1)
        _, current, _, _ = env.step(actions)
        reward += current
    displacement = env.qpos[:, 0] - start
    return float((displacement + 0.01 * reward).mean())


@torch.no_grad()
def score_design(design, checkpoint=None, steps=120, nworld=8, seed=0, device=None):
    env = CodesignWarpEnv(nworld, seed=seed, device=device, episode_length=None,
                          design=design)
    policy = (load_policy(checkpoint, env.obs_dim, env.act_dim, env.device)
              if checkpoint else lambda obs: torch.zeros((len(obs), env.act_dim), device=env.device))
    start = env.qpos[:, 0].clone()
    obs = env.reset()
    total = torch.zeros(nworld, device=env.device)
    for _ in range(steps):
        obs, reward, _, _ = env.step(policy(obs))
        total += reward
    return float((env.qpos[:, 0] - start + 0.01 * total).mean())


def search(mode="cpg", candidates=32, generations=4, seed=0, **kwargs):
    rng = np.random.default_rng(seed)
    dim = len(DEFAULT_RAW) if mode == "cpg" else 3
    mean = np.asarray(DEFAULT_RAW, dtype=float) if mode == "cpg" else np.full(dim, 0.5)
    std = np.full(dim, 0.15 if mode == "cpg" else 0.3)
    best = (float("-inf"), mean.copy())
    for generation in range(generations):
        population = mean + rng.normal(size=(candidates, dim)) * std
        if mode != "cpg":
            population = population.clip(0, 1)
        scores = np.asarray([
            score_cpg(item, seed=seed + index, **kwargs) if mode == "cpg"
            else score_design(item, seed=seed + index, **kwargs)
            for index, item in enumerate(population)
        ])
        elite = population[np.argsort(scores)[-max(2, candidates // 4):]]
        mean, std = elite.mean(0), elite.std(0).clip(min=1e-3)
        index = int(scores.argmax())
        if scores[index] > best[0]:
            best = float(scores[index]), population[index].copy()
        print(f"SEARCH generation={generation} best={scores.max():.6f}", flush=True)
    return {"mode": mode, "score": best[0], "parameters": best[1].tolist()}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("mode", nargs="?", choices=("cpg", "design"), default="cpg")
    parser.add_argument("--candidates", "--pop", type=int, default=16)
    parser.add_argument("--generations", "--gens", type=int, default=4)
    parser.add_argument("--steps", type=int, default=120)
    parser.add_argument("--envs", type=int, default=8)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--device", default=None)
    parser.add_argument("--checkpoint", "--ckpt")
    parser.add_argument("--out")
    args, _ = parser.parse_known_args(argv)
    result = search(args.mode, args.candidates, args.generations, args.seed,
                    steps=args.steps, nworld=args.envs, device=args.device,
                    **({"checkpoint": args.checkpoint} if args.mode == "design" else {}))
    text = json.dumps(result, indent=2)
    print(text)
    if args.out:
        Path(args.out).write_text(text + "\n")
    return result


if __name__ == "__main__":
    main()
