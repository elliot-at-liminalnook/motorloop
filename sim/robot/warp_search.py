# SPDX-License-Identifier: MIT
"""Random/CEM-style searches evaluated through actual MuJoCo-Warp rollouts."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import numpy as np
import torch

from cpg_teacher import DEFAULT_RAW, cpg_action, decode_params
from ladder_warp_env import UniversalControlWarpEnv
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
    env = UniversalControlWarpEnv(
        nworld, rung=30, seed=seed, device=device, episode_length=None,
        designs=[design])
    policy = (load_policy(
        checkpoint, env.obs_dim, env.act_dim, env.device,
        task_dim=env.architecture_task_dim, morphology_source=env)
              if checkpoint else lambda obs: torch.zeros((len(obs), env.act_dim), device=env.device))
    obs = env.reset()
    start = env.qpos[:, 0].clone()
    total = torch.zeros(nworld, device=env.device)
    for _ in range(steps):
        obs, reward, _, _ = env.step(policy(obs))
        total += reward
    return float((env.qpos[:, 0] - start + 0.01 * total).mean())


def search(mode="cpg", candidates=32, generations=4, seed=0,
           repeats=3, heldout_repeats=5, **kwargs):
    rng = np.random.default_rng(seed)
    dim = len(DEFAULT_RAW) if mode == "cpg" else 3
    mean = np.asarray(DEFAULT_RAW, dtype=float) if mode == "cpg" else np.full(dim, 0.5)
    std = np.full(dim, 0.15 if mode == "cpg" else 0.3)
    best = (float("-inf"), mean.copy())
    paired_seeds = [seed + 10_000 + 1_009 * index for index in range(repeats)]
    heldout_seeds = [seed + 1_000_000 + 10_007 * index
                     for index in range(heldout_repeats)]
    baseline = (float(np.mean([
        score_design(mean, seed=value, **kwargs) for value in paired_seeds]))
                if mode == "design" else None)
    for generation in range(generations):
        population = mean + rng.normal(size=(candidates, dim)) * std
        if mode != "cpg":
            population = population.clip(0, 1)
        if mode == "cpg":
            scores = np.asarray([
                score_cpg(item, seed=seed + index, **kwargs)
                for index, item in enumerate(population)])
        else:
            # Common random numbers: every design sees the exact same physics
            # seeds, so the ranking cannot be a lucky-seed ranking.
            scores = np.asarray([float(np.mean([
                score_design(item, seed=value, **kwargs) for value in paired_seeds
            ])) for item in population])
        elite = population[np.argsort(scores)[-max(2, candidates // 4):]]
        mean, std = elite.mean(0), elite.std(0).clip(min=1e-3)
        index = int(scores.argmax())
        if scores[index] > best[0]:
            best = float(scores[index]), population[index].copy()
        print(f"SEARCH generation={generation} best={scores.max():.6f}", flush=True)
    result = {"mode": mode, "score": best[0], "parameters": best[1].tolist()}
    if baseline is not None:
        heldout_baseline_scores = [
            score_design(np.full(dim, 0.5), seed=value, **kwargs)
            for value in heldout_seeds]
        heldout_best_scores = [
            score_design(best[1], seed=value, **kwargs) for value in heldout_seeds]
        heldout_baseline = float(np.mean(heldout_baseline_scores))
        heldout_best = float(np.mean(heldout_best_scores))
        result.update(
            selection_baseline_score=float(baseline),
            selection_improvement=float(best[0] - baseline),
            paired_selection_seeds=paired_seeds,
            heldout_seeds=heldout_seeds,
            heldout_baseline_scores=heldout_baseline_scores,
            heldout_best_scores=heldout_best_scores,
            baseline_score=heldout_baseline,
            heldout_score=heldout_best,
            improvement=float(heldout_best - heldout_baseline))
    return result


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("mode", nargs="?", choices=("cpg", "design"), default="cpg")
    parser.add_argument("--candidates", "--pop", type=int, default=16)
    parser.add_argument("--generations", "--gens", type=int, default=4)
    parser.add_argument("--steps", type=int, default=120)
    parser.add_argument("--envs", type=int, default=8)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--repeats", type=int, default=3,
                        help="paired common-random-number seeds during search")
    parser.add_argument("--heldout-repeats", type=int, default=5,
                        help="unseen seed bank used only for the reported gate")
    parser.add_argument("--device", default=None)
    parser.add_argument("--checkpoint", "--ckpt")
    parser.add_argument("--out")
    args = parser.parse_args(argv)
    result = search(args.mode, args.candidates, args.generations, args.seed,
                    repeats=args.repeats, heldout_repeats=args.heldout_repeats,
                    steps=args.steps, nworld=args.envs, device=args.device,
                    **({"checkpoint": args.checkpoint} if args.mode == "design" else {}))
    text = json.dumps(result, indent=2)
    print(text)
    if args.out:
        Path(args.out).write_text(text + "\n")
    return result


if __name__ == "__main__":
    main()
