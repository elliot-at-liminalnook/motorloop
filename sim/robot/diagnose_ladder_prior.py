# SPDX-License-Identifier: MIT
"""Measure the safe policy/prior interpolation frontier for ladder rung 7.

The searched walking prior can be fast but constraint-mismatched, while PPO can
settle on a safe policy that moves mostly sideways.  This diagnostic evaluates
paired interpolation families in one batched MuJoCo-Warp run and reports the
full rung-7 gate margin for each blend.  It is deliberately read-only: a result
artifact can justify the next training adaptation without mutating a checkpoint.
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import torch

from ladder_warp_env import LadderLocomotionWarpEnv, WALKER_OBS_DIM
from train_mesh_warp import (Actor, RunningNorm, gate_diagnostics, load_policy,
                             sha256_file)


GATES = (
    ("xprogress", ">=", 0.10),
    ("lateral", "<=", 0.15),
    ("duty", "<=", 0.95),
    ("up", ">=", 0.85),
    ("ladder_step_clock", ">=", 0.65),
    ("ladder_swing_clearance", ">=", 0.25),
    ("catrate", "<=", 0.001),
    ("fallrate", "<=", 0.0001),
)
AXIS_INDEX = {"yaw": (0,), "pitch": (1,), "lift": (2,),
              "yaw_pitch": (0, 1), "all": (0, 1, 2)}


def _load_raw_policy(path: str | Path, obs_dim: int, act_dim: int, device):
    checkpoint = torch.load(path, map_location=device, weights_only=True)
    saved = checkpoint.get("args", {})
    hidden = tuple(int(value) for value in saved.get("hidden", "512,256,128").split(","))
    architecture = saved.get("architecture", "mlp")
    prediction_decoder = saved.get("prediction_decoder", "recurrent")
    task_dim = int(saved.get("actor_task_dim", 0))
    actor = Actor(obs_dim, act_dim, hidden, architecture=architecture,
                  task_dim=task_dim,
                  prediction_decoder=prediction_decoder).to(device)
    norm = RunningNorm(obs_dim).to(device)
    actor.load_state_dict(checkpoint["actor"])
    norm.load_state_dict(checkpoint["obs_norm"])
    actor.eval(); norm.eval()
    return checkpoint, actor, norm


def _axis_mask(mode: str, act_dim: int, device) -> torch.Tensor:
    mask = torch.zeros(act_dim, device=device)
    for axis in AXIS_INDEX[mode]:
        mask[axis::3] = 1.0
    return mask


def _restore_duals(env: LadderLocomotionWarpEnv, checkpoint: dict) -> dict:
    tensors = checkpoint.get("runtime", {}).get("env", {}).get("tensors", {})
    restored = {}
    for field, destination in (
        ("_constraint_duals", env.constraint_duals),
        ("_competence_duals", env.competence_duals),
    ):
        if field not in tensors:
            continue
        value = tensors[field].to(device=destination.device, dtype=destination.dtype)
        destination.copy_(value)
        restored[field] = destination.detach().cpu().tolist()
    return restored


@torch.no_grad()
def sweep(args) -> dict:
    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    blends = tuple(float(value) for value in args.blends.split(","))
    modes = tuple(value.strip() for value in args.modes.split(",") if value.strip())
    if not blends or any(not 0.0 <= value <= 1.0 for value in blends):
        raise ValueError("--blends must contain values in [0, 1]")
    if not modes or any(value not in AXIS_INDEX for value in modes):
        raise ValueError(f"--modes must be drawn from {tuple(AXIS_INDEX)}")

    combinations = [(mode, blend) for mode in modes for blend in blends]
    rows = len(combinations)
    worlds = rows * args.repeats
    env = LadderLocomotionWarpEnv(
        worlds, rung=7, seed=args.seed, device=device, episode_length=args.episode_length)
    env.configure_action_prior(args.prior_json)
    checkpoint, actor, norm = _load_raw_policy(
        args.checkpoint, env.obs_dim, env.act_dim, env.device)
    _, anchor, anchor_norm = _load_raw_policy(
        args.anchor, env.obs_dim, env.act_dim, env.device)
    transfer = load_policy(args.transfer, WALKER_OBS_DIM, env.act_dim, env.device)
    restored_duals = _restore_duals(env, checkpoint)

    blend_world = torch.as_tensor(
        [blend for _, blend in combinations], device=env.device,
        dtype=torch.float32).repeat_interleave(args.repeats)[:, None]
    mode_masks = torch.stack([
        _axis_mask(mode, env.act_dim, env.device) for mode, _ in combinations
    ]).repeat_interleave(args.repeats, dim=0)
    metric_names = (
        "xprogress", "lateral", "speed", "align", "catrate", "cat_slip",
        "fallrate", "duty", "up", "ladder_step_clock",
        "ladder_swing_clearance", "ladder_safe_progress", "reward",
    )
    sums = {name: torch.zeros(worlds, device=env.device) for name in metric_names}
    prior_axis_error = {
        axis: torch.zeros(worlds, device=env.device) for axis in ("yaw", "pitch", "lift")}
    obs = env.reset()
    for _ in range(args.steps):
        learner_mean = actor(norm(obs))
        anchor_obs = obs.clone()
        anchor_obs[:, -23:] = 0.0
        anchor_obs[:, -23 + 5] = 1.0
        anchor_obs[:, 47:50] = 0.0
        prior_base = anchor(anchor_norm(anchor_obs))
        transfer_action = transfer(obs[:, :WALKER_OBS_DIM])
        prior_mean, prior_mask = env.policy_mean_prior(
            obs, prior_base, transfer_action=transfer_action)
        interpolation = mode_masks * (prior_mask.abs() > 0.0)
        mean = learner_mean + blend_world * interpolation * (prior_mean - learner_mean)
        obs, reward, _, info = env.step(torch.tanh(mean), alpha=1.0, imit_anneal=0.0)
        values = {
            "xprogress": info["xprogress"],
            "lateral": info["lateral"],
            "speed": info["speed"],
            "align": info["align"],
            "catrate": info["cat_done"],
            "cat_slip": info["cat_slip"],
            "fallrate": info["fallrate"],
            "duty": info["contact"].mean(dim=-1),
            "up": info["up"],
            "ladder_step_clock": info["ladder_step_clock"],
            "ladder_swing_clearance": info["ladder_swing_clearance"],
            "ladder_safe_progress": info["ladder_safe_progress"],
            "reward": reward,
        }
        for name, value in values.items():
            sums[name].add_(value)
        difference = learner_mean - prior_mean
        for axis_name, axis in (("yaw", 0), ("pitch", 1), ("lift", 2)):
            prior_axis_error[axis_name].add_(difference[:, axis::3].square().mean(dim=-1))

    samples = {
        name: (value / args.steps).reshape(rows, args.repeats)
        for name, value in sums.items()
    }
    results = []
    for index, (mode, blend) in enumerate(combinations):
        metrics = {name: float(value[index].mean()) for name, value in samples.items()}
        metrics["xprogress_std"] = float(samples["xprogress"][index].std(unbiased=False))
        metrics["xprogress_stderr"] = metrics["xprogress_std"] / math.sqrt(args.repeats)
        metrics["catrate_stderr"] = float(
            samples["catrate"][index].std(unbiased=False) / math.sqrt(args.repeats))
        metrics["forward_speed_fraction"] = (
            metrics["xprogress"] / max(metrics["speed"], 1.0e-6))
        metrics["lateral_forward_ratio"] = (
            metrics["lateral"] / max(abs(metrics["xprogress"]), 1.0e-3))
        metrics["prior_rmse_by_axis"] = {
            axis: float(torch.sqrt(
                (value / args.steps).reshape(rows, args.repeats)[index].mean()))
            for axis, value in prior_axis_error.items()
        }
        gates = gate_diagnostics(GATES, metrics)
        results.append({"mode": mode, "blend": blend, **metrics, "gates": gates})

    feasible = [row for row in results if row["gates"]["all_pass"]]
    if feasible:
        selected = max(feasible, key=lambda row: (row["xprogress"], -row["blend"]))
        selection_reason = "fastest fully feasible interpolation"
    else:
        selected = max(results, key=lambda row: (
            row["gates"]["worst_relative_margin"], row["xprogress"]))
        selection_reason = "largest worst-gate headroom; no interpolation fully feasible"
    output = {
        "schema_version": 1,
        "checkpoint": str(args.checkpoint),
        "checkpoint_sha256": sha256_file(args.checkpoint),
        "anchor": str(args.anchor),
        "anchor_sha256": sha256_file(args.anchor),
        "transfer": str(args.transfer),
        "transfer_sha256": sha256_file(args.transfer),
        "prior_json": str(args.prior_json),
        "prior_sha256": sha256_file(args.prior_json),
        "seed": args.seed,
        "steps": args.steps,
        "repeats": args.repeats,
        "restored_duals": restored_duals,
        "results": results,
        "selected": selected,
        "selection_reason": selection_reason,
    }
    if args.out:
        Path(args.out).write_text(json.dumps(output, indent=2, sort_keys=True) + "\n")
    return output


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--anchor", required=True)
    parser.add_argument("--transfer", required=True)
    parser.add_argument("--prior-json", required=True)
    parser.add_argument("--blends", default="0,0.125,0.25,0.375,0.5,0.625,0.75,0.875,1")
    parser.add_argument("--modes", default="yaw,pitch,yaw_pitch,all")
    parser.add_argument("--repeats", type=int, default=16)
    parser.add_argument("--steps", type=int, default=400)
    parser.add_argument("--episode-length", type=int, default=800)
    parser.add_argument("--seed", type=int, default=20267777)
    parser.add_argument("--device", default=None)
    parser.add_argument("--out")
    args = parser.parse_args(argv)
    if min(args.repeats, args.steps, args.episode_length) <= 0:
        raise ValueError("--repeats/--steps/--episode-length must be positive")
    output = sweep(args)
    print(json.dumps({
        "selection_reason": output["selection_reason"],
        "selected": output["selected"],
    }, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
