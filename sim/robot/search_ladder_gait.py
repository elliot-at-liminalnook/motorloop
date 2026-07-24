# SPDX-License-Identifier: MIT
"""Search a clean periodic residual around the rung-6/legacy walk hybrid.

The legacy walker knows useful state-conditioned fore/aft coordination but was
trained before strict contact constraints.  The accepted rung-6 policy supplies
balance, and the ladder clock supplies real foot lift.  This CEM search adds a
small per-leg Fourier residual to yaw and pitch, scoring forward velocity only
on non-slipping transitions and explicitly rejecting resets/falls.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import torch

from ladder_warp_env import LadderLocomotionWarpEnv, WALKER_OBS_DIM
from train_mesh_warp import load_policy


PARAMS_PER_AXIS = 3  # bias, sin, cos
CONTROLLED_AXES = 8  # four yaw + four pitch
PARAM_DIM = PARAMS_PER_AXIS * CONTROLLED_AXES


def _sha256(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _residual(parameters: torch.Tensor, phase: torch.Tensor) -> torch.Tensor:
    """Return (world, 4, 2) yaw/pitch residuals."""
    coeff = parameters.reshape(-1, 4, 2, PARAMS_PER_AXIS)
    s = torch.sin(phase)[:, None, None]
    c = torch.cos(phase)[:, None, None]
    return coeff[..., 0] + coeff[..., 1] * s + coeff[..., 2] * c


@torch.no_grad()
def evaluate_population(
    env: LadderLocomotionWarpEnv,
    rung6,
    legacy,
    population: torch.Tensor,
    repeats: int,
    steps: int,
    blend: float,
    seed: int,
    cat_weight: float,
    slip_weight: float,
) -> dict[str, torch.Tensor]:
    pop = len(population)
    expanded = population.repeat_interleave(repeats, dim=0)
    env._gen.manual_seed(seed)
    obs = env.reset()
    sums = {key: torch.zeros(pop * repeats, device=env.device) for key in (
        "x", "safe_x", "lateral", "cat", "cat_slip", "fall", "duty", "step")}
    for _ in range(steps):
        transfer_obs = obs.clone()
        transfer_obs[:, 47:50] = 0.0
        transfer_obs[:, -23:] = 0.0
        transfer_obs[:, -23 + 5] = 1.0
        base = rung6(transfer_obs)
        old = legacy(obs[:, :WALKER_OBS_DIM])
        phase = torch.atan2(
            obs[:, env._phase_sin_obs_index],
            obs[:, env._phase_sin_obs_index + 1],
        )
        action = base.clone()
        action[:, 0::3].lerp_(old[:, 0::3], blend)
        action[:, 1::3].lerp_(old[:, 1::3], blend)
        correction = _residual(expanded, phase)
        action[:, 0::3].add_(correction[..., 0])
        action[:, 1::3].add_(correction[..., 1])
        lift_wave = torch.sin(phase)[:, None] * (env._pair_a - env._pair_b)
        action[:, 2::3] = 0.95 * lift_wave
        obs, _, _, info = env.step(action.clamp(-0.95, 0.95), alpha=1.0,
                                   imit_anneal=0.0)
        clean = (info["cat_slip"] <= 0.0).to(obs.dtype) * (1.0 - info["fallrate"])
        sums["x"].add_(info["xprogress"])
        sums["safe_x"].add_(info["xprogress"] * clean)
        sums["lateral"].add_(info["lateral"])
        sums["cat"].add_(info["cat_done"])
        sums["cat_slip"].add_(info["cat_slip"])
        sums["fall"].add_(info["fallrate"])
        sums["duty"].add_(info["contact"].mean(dim=-1))
        sums["step"].add_(info["ladder_step_clock"])
    result = {}
    for key, value in sums.items():
        result[key] = (value / float(steps)).reshape(pop, repeats).mean(dim=-1)
    result["score"] = (
        result["safe_x"]
        - cat_weight * result["cat"]
        - slip_weight * result["cat_slip"]
        - 0.10 * result["lateral"]
        + 0.02 * result["step"]
        - 4.0 * result["fall"]
        - 1.5 * (result["duty"] - 0.95).clamp_min(0.0)
    )
    return result


def search(args) -> dict:
    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    worlds = args.population * args.repeats
    env = LadderLocomotionWarpEnv(worlds, rung=7, seed=args.seed, device=device,
                                  episode_length=800)
    rung6 = load_policy(args.rung6, env.obs_dim, env.act_dim, env.device)
    legacy = load_policy(args.legacy, WALKER_OBS_DIM, env.act_dim, env.device)
    generator = torch.Generator(device=env.device)
    generator.manual_seed(args.seed + 991)
    mean = torch.zeros(PARAM_DIM, device=env.device)
    if args.init_json:
        initial = json.loads(Path(args.init_json).read_text())
        values = initial.get("best", initial).get("parameters")
        if values is None or len(values) != PARAM_DIM:
            raise ValueError(f"--init-json must contain {PARAM_DIM} best parameters")
        mean.copy_(torch.as_tensor(values, dtype=mean.dtype, device=mean.device))
    std = torch.full_like(mean, args.initial_std)
    best: dict | None = None
    history = []
    elite_count = max(2, round(args.population * args.elite_fraction))
    for generation in range(args.generations):
        population = mean + std * torch.randn(
            (args.population, PARAM_DIM), generator=generator, device=env.device)
        population.clamp_(-args.residual_limit, args.residual_limit)
        population[0].zero_() if generation == 0 else population[0].copy_(mean)
        metrics = evaluate_population(
            env, rung6, legacy, population, args.repeats, args.steps,
            args.blend, args.seed + 10_000 * generation,
            args.cat_weight, args.slip_weight)
        elite_index = torch.topk(metrics["score"], elite_count).indices
        elite = population[elite_index]
        mean = elite.mean(dim=0)
        std = elite.std(dim=0, unbiased=False).clamp_min(args.min_std)
        index = int(metrics["score"].argmax())
        row = {key: float(value[index]) for key, value in metrics.items()}
        row.update(generation=generation)
        candidate = {**row, "parameters": population[index].cpu().tolist()}
        history.append(candidate)
        if best is None or candidate["score"] > best["score"]:
            best = candidate
        print("SEARCH " + " ".join(
            f"{key}={value:.5f}" for key, value in row.items()
            if key != "generation") + f" generation={generation}", flush=True)
    assert best is not None
    pareto = []
    for candidate in history:
        dominated = any(
            other is not candidate
            and other["safe_x"] >= candidate["safe_x"]
            and other["cat"] <= candidate["cat"]
            and other["fall"] <= candidate["fall"]
            and (other["safe_x"] > candidate["safe_x"]
                 or other["cat"] < candidate["cat"]
                 or other["fall"] < candidate["fall"])
            for other in history
        )
        if not dominated:
            pareto.append(candidate)
    pareto.sort(key=lambda item: item["safe_x"])
    output = {
        "parameterization": "per-leg yaw/pitch bias+sin+cos",
        "source_checkpoints": {
            "rung6": {"path": str(Path(args.rung6).resolve()),
                      "sha256": _sha256(args.rung6)},
            "legacy": {"path": str(Path(args.legacy).resolve()),
                        "sha256": _sha256(args.legacy)},
        },
        "blend": args.blend,
        "population": args.population,
        "repeats": args.repeats,
        "steps": args.steps,
        "seed": args.seed,
        "cat_weight": args.cat_weight,
        "slip_weight": args.slip_weight,
        "best": best,
        "pareto": pareto,
        "history": history,
    }
    if args.out:
        Path(args.out).write_text(json.dumps(output, indent=2, sort_keys=True) + "\n")
    return output


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--rung6", required=True)
    parser.add_argument("--legacy", required=True)
    parser.add_argument("--population", type=int, default=64)
    parser.add_argument("--repeats", type=int, default=4)
    parser.add_argument("--generations", type=int, default=12)
    parser.add_argument("--steps", type=int, default=400)
    parser.add_argument("--blend", type=float, default=0.65)
    parser.add_argument("--cat-weight", type=float, default=1.5)
    parser.add_argument("--slip-weight", type=float, default=0.5)
    parser.add_argument("--init-json")
    parser.add_argument("--initial-std", type=float, default=0.15)
    parser.add_argument("--min-std", type=float, default=0.015)
    parser.add_argument("--residual-limit", type=float, default=0.55)
    parser.add_argument("--elite-fraction", type=float, default=0.20)
    parser.add_argument("--seed", type=int, default=20260712)
    parser.add_argument("--device", default=None)
    parser.add_argument("--out")
    args = parser.parse_args(argv)
    result = search(args)
    print(json.dumps(result["best"], indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
