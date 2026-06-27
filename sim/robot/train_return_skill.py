# SPDX-License-Identifier: MIT
"""Train the local return-to-origin skill.

This intentionally trains only the final local skill:

    active-WP3 / return handoff state -> origin

It uses ``ReturnSkillEnv`` with a 5-D action head and does not try to solve the
whole checkpoint square during training.
"""

from __future__ import annotations

import argparse
import json
import os
import pickle
import sys
import time
from pathlib import Path

import jax
import jax.numpy as jnp

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
OUT = Path(os.environ.get("CODESIGN_OUT", "/root/proj/out"))


def resolve_path(text: str) -> Path:
    path = Path(text)
    candidates = [path]
    if not path.is_absolute():
        candidates.extend([ROOT / path, OUT / path])
        if path.parts and path.parts[0] == "out":
            candidates.append(OUT.joinpath(*path.parts[1:]))
    for candidate in candidates:
        if candidate.exists():
            return candidate
    raise FileNotFoundError(text)


def apply_route_env(path_text: str) -> dict[str, str]:
    if not path_text:
        return {}
    data = json.loads(resolve_path(path_text).read_text())
    deploy_env = data.get("deploy_env") or {}
    if not deploy_env:
        raise ValueError(f"{path_text!r} has no deploy_env")
    for key, value in deploy_env.items():
        os.environ[str(key)] = str(value)
    return {str(k): str(v) for k, v in deploy_env.items()}


def infer_policy_obs(policy) -> int | None:
    pp = policy.get("params", policy) if isinstance(policy, dict) else {}
    try:
        return int(pp["hidden_0"]["kernel"].shape[0])
    except Exception:
        return None


def infer_policy_output(policy) -> int | None:
    pp = policy.get("params", policy) if isinstance(policy, dict) else {}
    try:
        layers = [
            (name, layer)
            for name, layer in pp.items()
            if isinstance(layer, dict) and "bias" in layer
        ]
        if not layers:
            return None
        # Flax dense layers are named hidden_0...hidden_n here; the final
        # policy head emits mean and log-std, so width is usually 2 * action_dim.
        name, layer = sorted(layers, key=lambda item: int(item[0].split("_")[-1]))[-1]
        return int(layer["bias"].shape[0])
    except Exception:
        return None


def warm_start(path: str | None, obs_dim: int, action_dim: int):
    if not path:
        return None
    p = resolve_path(path)
    try:
        parts = list(pickle.load(open(p, "rb")))
        if len(parts) < 2:
            return None
        old_obs = infer_policy_obs(parts[1])
        old_output = infer_policy_output(parts[1])
        expected_outputs = {action_dim, action_dim * 2}
        if old_obs != obs_dim or old_output not in expected_outputs:
            print(f"WARM-START skipped: checkpoint obs/output {old_obs}/{old_output} "
                  f"not compatible with env {obs_dim}/{action_dim}",
                  flush=True)
            return None
        return tuple(parts)
    except Exception as exc:
        print(f"WARM-START skipped ({type(exc).__name__}: {exc})", flush=True)
        return None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tag", default="return_skill")
    ap.add_argument("--route-candidate", default="out/cpg_route_full_softmax2_wp2_focuscross_route_gait.json")
    ap.add_argument("--resume", default="")
    ap.add_argument("--starts", default="")
    ap.add_argument("--start-jitter", type=float, default=0.015)
    ap.add_argument("--phase-jitter", type=float, default=0.03)
    ap.add_argument("--vel-jitter", type=float, default=0.03)
    ap.add_argument("--steps", type=int, default=500_000)
    ap.add_argument("--envs", type=int, default=1024)
    ap.add_argument("--batch", type=int, default=512)
    ap.add_argument("--minibatches", type=int, default=8)
    ap.add_argument("--unroll", type=int, default=10)
    ap.add_argument("--updates", type=int, default=2)
    ap.add_argument("--evals", type=int, default=5)
    ap.add_argument("--episode-length", type=int, default=260)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--entropy", type=float, default=0.01)
    args = ap.parse_args()

    OUT.mkdir(parents=True, exist_ok=True)
    deploy_env = apply_route_env(args.route_candidate)
    os.environ.update({
        "CODESIGN_OUT": str(OUT),
        "CMD_CONTROL_MODE": "cpg_pd",
        "CMD_CPG_RESIDUAL_SCALE": "0.0",
        "CMD_CPG_RESIDUAL_SCALE_TRANSITION": "0.0",
        "CMD_CPG_RESIDUAL_SCALE_WP2": "",
        "CMD_CPG_RESIDUAL_SCALE_WP3": "",
        "CMD_TRAIN_MODE": "route",
        "CMD_REWARD_MODE": "route",
        "CMD_VMAX": os.environ.get("CMD_VMAX", "0.35"),
        "CMD_TRACK_SIGMA": "0.05",
        "CMD_RESET_NOISE": "0.0",
        "RETURN_SKILL_STARTS": args.starts,
        "RETURN_SKILL_START_JITTER": str(args.start_jitter),
        "RETURN_SKILL_PHASE_JITTER": str(args.phase_jitter),
        "RETURN_SKILL_VEL_JITTER": str(args.vel_jitter),
    })
    sys.path.insert(0, str(HERE))
    from brax.training.agents.ppo import train as ppo  # noqa: E402
    from return_skill_env import _build  # noqa: E402

    Env = _build()
    env = Env()
    restore = warm_start(args.resume, env.observation_size, env.action_size)
    meta = {
        "tag": args.tag,
        "route_candidate": str(resolve_path(args.route_candidate)) if args.route_candidate else "",
        "route_candidate_env_keys": sorted(deploy_env),
        "starts": args.starts,
        "start_jitter": args.start_jitter,
        "phase_jitter": args.phase_jitter,
        "vel_jitter": args.vel_jitter,
        "obs": env.observation_size,
        "action": env.action_size,
        "steps": args.steps,
        "envs": args.envs,
        "episode_length": args.episode_length,
    }
    (OUT / f"{args.tag}_train_meta.json").write_text(json.dumps(meta, indent=2))
    print(f"return skill env: obs={env.observation_size} act={env.action_size}", flush=True)
    t0 = time.time()
    hist = []

    def progress(step, metrics):
        def g(name):
            return float(metrics.get(f"eval/episode_{name}", 0.0))
        row = {
            "step": int(step),
            "sec": round(time.time() - t0, 1),
            "reward": g("reward"),
            "return_dist": g("return_dist"),
            "return_progress": g("return_progress"),
            "return_hit": g("return_hit"),
        }
        hist.append(row)
        with open(OUT / f"{args.tag}_train.jsonl", "a") as f:
            f.write(json.dumps(row) + "\n")
        print(
            f"[return-skill] step {row['step']:>8,} reward={row['reward']:.3f} "
            f"dist={row['return_dist']:.3f} progress={row['return_progress']:.3f} "
            f"hit={row['return_hit']:.3f}",
            flush=True,
        )

    def checkpoint(*items):
        try:
            with open(OUT / f"{args.tag}_ckpt.pkl", "wb") as f:
                pickle.dump(items[-1], f)
        except Exception as exc:
            print(f"[return-skill] checkpoint failed: {exc}", flush=True)

    _, params, _ = ppo.train(
        environment=env,
        num_timesteps=args.steps,
        num_evals=args.evals,
        episode_length=args.episode_length,
        num_envs=args.envs,
        batch_size=args.batch,
        num_minibatches=args.minibatches,
        unroll_length=args.unroll,
        num_updates_per_batch=args.updates,
        learning_rate=args.lr,
        entropy_cost=args.entropy,
        discounting=0.97,
        reward_scaling=0.1,
        normalize_observations=True,
        seed=args.seed,
        progress_fn=progress,
        policy_params_fn=checkpoint,
        restore_params=restore,
    )
    out = OUT / f"{args.tag}.pkl"
    pickle.dump(params, open(out, "wb"))
    report = {**meta, "artifact": str(out), "history": hist, "train_s": round(time.time() - t0, 1)}
    (OUT / f"{args.tag}_report.json").write_text(json.dumps(report, indent=2))
    print(json.dumps(report, indent=2), flush=True)
    print(f"saved {out}", flush=True)


if __name__ == "__main__":
    main()
