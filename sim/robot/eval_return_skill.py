# SPDX-License-Identifier: MIT
"""Evaluate a trained return-to-origin skill on local WP3 starts."""

from __future__ import annotations

import argparse
import json
import os
import pickle
import sys
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np

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


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tag", default="return_skill_eval")
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--route-candidate", default="out/cpg_route_full_softmax2_wp2_focuscross_route_gait.json")
    ap.add_argument("--starts", default="")
    ap.add_argument("--start-jitter", type=float, default=0.015)
    ap.add_argument("--phase-jitter", type=float, default=0.03)
    ap.add_argument("--vel-jitter", type=float, default=0.03)
    ap.add_argument("--episodes", type=int, default=8)
    ap.add_argument("--steps", type=int, default=260)
    ap.add_argument("--seed", type=int, default=53)
    args = ap.parse_args()

    OUT.mkdir(parents=True, exist_ok=True)
    deploy_env = apply_route_env(args.route_candidate)
    os.environ.update({
        "CODESIGN_OUT": str(OUT),
        "CMD_CONTROL_MODE": "cpg_pd",
        "CMD_CPG_RESIDUAL_SCALE": "0.0",
        "CMD_CPG_RESIDUAL_SCALE_TRANSITION": "0.0",
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
    from brax.training.acme import running_statistics  # noqa: E402
    import ppo_nets as ppo_networks  # shared (512,256,128) + small-final-init factory (audit item 5)  # noqa: E402
    from commanded_env import FALL_Z, MIN_UP_Z  # noqa: E402
    from return_skill_env import _build  # noqa: E402

    params = pickle.load(open(resolve_path(args.ckpt), "rb"))
    Env = _build()
    env = Env()
    net = ppo_networks.make_ppo_networks(
        env.observation_size,
        env.action_size,
        preprocess_observations_fn=running_statistics.normalize,
    )
    inf = ppo_networks.make_inference_fn(net)(params, deterministic=True)
    step_fn = jax.jit(env.step)
    rows = []
    summaries = []
    for ep in range(args.episodes):
        key = jax.random.PRNGKey(args.seed + ep)
        state = env.reset(key)
        fall_reason = ""
        hit_step = -1
        closest = float("inf")
        for t in range(args.steps):
            key, act_key = jax.random.split(key)
            action, _ = inf(state.obs, act_key)
            state = step_fn(state, action)
            pos = np.asarray(state.pipeline_state.qpos[:2], dtype=np.float32)
            dist = float(np.linalg.norm(pos))
            closest = min(closest, dist)
            up = 1.0 - 2.0 * (float(state.pipeline_state.qpos[4]) ** 2 + float(state.pipeline_state.qpos[5]) ** 2)
            hit = dist <= float(os.environ.get("RETURN_SKILL_RADIUS", "0.07"))
            if hit and hit_step < 0:
                hit_step = t
            rows.append([
                ep,
                t,
                float(pos[0]),
                float(pos[1]),
                dist,
                float(state.pipeline_state.qpos[2]),
                up,
                float(np.asarray(action[0])),
                float(np.asarray(action[1])),
                float(np.asarray(action[2])),
                float(np.asarray(action[3])),
                float(np.asarray(action[4])),
            ])
            if float(state.pipeline_state.qpos[2]) < FALL_Z:
                fall_reason = f"height<{FALL_Z:.3f}"
                break
            if up < MIN_UP_Z:
                fall_reason = f"up<{MIN_UP_Z:.3f}"
                break
        summaries.append({
            "episode": ep,
            "samples": t + 1,
            "final_dist": dist,
            "closest_dist": closest,
            "hit": bool(hit_step >= 0),
            "hit_step": int(hit_step),
            "fall_reason": fall_reason,
        })
    R = np.asarray(rows, dtype=np.float32)
    npz = OUT / f"{args.tag}_return_eval.npz"
    js = OUT / f"{args.tag}_return_eval_summary.json"
    if R.size:
        np.savez(
            npz,
            episode=R[:, 0],
            t=R[:, 1],
            x=R[:, 2],
            y=R[:, 3],
            dist=R[:, 4],
            z=R[:, 5],
            up=R[:, 6],
            action=R[:, 7:12],
            tag=args.tag,
        )
    else:
        np.savez(npz, tag=args.tag)
    final = np.asarray([s["final_dist"] for s in summaries], dtype=np.float32)
    closest = np.asarray([s["closest_dist"] for s in summaries], dtype=np.float32)
    summary = {
        "tag": args.tag,
        "ckpt": str(resolve_path(args.ckpt)),
        "route_candidate": str(resolve_path(args.route_candidate)) if args.route_candidate else "",
        "route_candidate_env_keys": sorted(deploy_env),
        "episodes": int(args.episodes),
        "steps": int(args.steps),
        "hit_count": int(sum(1 for s in summaries if s["hit"])),
        "fall_count": int(sum(1 for s in summaries if s["fall_reason"])),
        "final_dist_mean": float(final.mean()) if len(final) else float("nan"),
        "final_dist_min": float(final.min()) if len(final) else float("nan"),
        "closest_dist_mean": float(closest.mean()) if len(closest) else float("nan"),
        "closest_dist_min": float(closest.min()) if len(closest) else float("nan"),
        "summaries": summaries,
    }
    js.write_text(json.dumps(summary, indent=2))
    print(
        f"[return-skill-eval] hits={summary['hit_count']}/{args.episodes} "
        f"falls={summary['fall_count']} final_mean={summary['final_dist_mean']:.6f} "
        f"closest_mean={summary['closest_dist_mean']:.6f}",
        flush=True,
    )
    print(json.dumps(summary, indent=2), flush=True)
    print(f"saved {npz} and {js}", flush=True)


if __name__ == "__main__":
    main()
