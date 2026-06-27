# SPDX-License-Identifier: MIT
"""Closed-loop route replay with a compact correction policy.

This evaluates artifacts produced by ``train_route_corrector.py`` without using
fixed WP2/WP3 schedule JSONs.  The corrector observes route-state features,
predicts compact coefficients, reconstructs a residual action, and lets
``CommandedEnv`` apply it through the normal CPG/PD residual channel.
"""

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
sys.path.insert(0, str(HERE))


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
    data = json.loads(resolve_path(path_text).read_text())
    deploy_env = data.get("deploy_env") or {}
    if not deploy_env:
        raise ValueError(f"{path_text!r} has no deploy_env")
    for key, value in deploy_env.items():
        os.environ[str(key)] = str(value)
    return deploy_env


def parse_waypoints(text: str) -> np.ndarray:
    pts = []
    for chunk in text.split(";"):
        chunk = chunk.strip()
        if not chunk:
            continue
        x, y = chunk.split(",")
        pts.append((float(x), float(y)))
    if not pts:
        raise ValueError("at least one waypoint is required")
    return np.asarray(pts, dtype=np.float32)


def waypoint_command(pos: np.ndarray, reached: int, waypoints: np.ndarray, radius: float, gain: float, vmax: float):
    target = waypoints[min(reached, len(waypoints) - 1)]
    delta = target - pos
    dist = float(np.linalg.norm(delta))
    if dist < radius and reached < len(waypoints):
        reached += 1
        if reached >= len(waypoints):
            return reached, target, dist, np.zeros(2, dtype=np.float32)
        target = waypoints[reached]
        delta = target - pos
        dist = float(np.linalg.norm(delta))
    cmd = gain * delta
    norm = float(np.linalg.norm(cmd))
    if norm > vmax:
        cmd = cmd * (vmax / max(norm, 1e-6))
    return reached, target, dist, np.asarray(cmd, dtype=np.float32)


def apply_mlp(params, x):
    for layer in params[:-1]:
        x = jnp.tanh(x @ layer["w"] + layer["b"])
    return x @ params[-1]["w"] + params[-1]["b"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tag", default="route_corrector")
    ap.add_argument("--artifact", required=True)
    ap.add_argument("--wp2-artifact", default="",
                    help="optional artifact used only while active route waypoint 2 is corrected")
    ap.add_argument("--wp3-artifact", default="",
                    help="optional artifact used only while active route waypoint 3 is corrected")
    ap.add_argument("--wp3-command-artifact", default="",
                    help="optional low-dimensional command-head artifact used while active waypoint 3 is corrected")
    ap.add_argument("--wp3-command-blend", type=float, default=1.0,
                    help="multiplier for normalized command corrections predicted by --wp3-command-artifact")
    ap.add_argument("--wp3-command-clip", type=float, default=1.0,
                    help="clip normalized command correction before multiplying by vmax")
    ap.add_argument("--wp3-command-residual-gain", type=float, default=1.0,
                    help="multiplier for the predicted residual-strength head when scaling WP3 residual actions")
    ap.add_argument("--wp3-command-residual-max", type=float, default=1.5,
                    help="max multiplier applied to WP3 residual actions from the command-head strength output")
    ap.add_argument("--route-candidate", required=True)
    ap.add_argument("--waypoints", default="0.35,0;0.35,0.35;0,0.35;0,0")
    ap.add_argument("--radius", type=float, default=0.07)
    ap.add_argument("--steps-per-waypoint", type=int, default=260)
    ap.add_argument("--gain", type=float, default=2.0)
    ap.add_argument("--vmax", type=float, default=0.35)
    ap.add_argument("--wp2-residual-scale", type=float, default=0.64)
    ap.add_argument("--wp3-residual-scale", type=float, default=0.64)
    ap.add_argument("--seed", type=int, default=53)
    args = ap.parse_args()

    deploy_env = apply_route_env(args.route_candidate)
    os.environ.update({
        "CODESIGN_OUT": str(OUT),
        "CMD_TRAIN_MODE": "route",
        "CMD_REWARD_MODE": "route",
        "CMD_CONTROL_MODE": "cpg_pd",
        "CMD_CPG_RESIDUAL_SCALE": "0.0",
        "CMD_CPG_RESIDUAL_SCALE_TRANSITION": "0.0",
        "CMD_CPG_RESIDUAL_SCALE_WP2": str(args.wp2_residual_scale),
        "CMD_CPG_RESIDUAL_SCALE_WP3": str(args.wp3_residual_scale),
        "CMD_ROUTE_WAYPOINTS": args.waypoints,
        "CMD_ROUTE_RADIUS": str(args.radius),
        "CMD_ROUTE_GAIN": str(args.gain),
        "CMD_VMAX": str(args.vmax),
        "CMD_TRACK_SIGMA": "0.05",
        "CMD_RESET_NOISE": "0.0",
    })

    from commanded_env import FALL_Z, MIN_UP_Z, _build

    def load_artifact(path_text: str):
        artifact = pickle.load(open(resolve_path(path_text), "rb"))
        return {
            "path": str(resolve_path(path_text)),
            "params": artifact["params"],
            "feature_mean": jnp.asarray(artifact["feature_mean"]),
            "feature_std": jnp.asarray(artifact["feature_std"]),
            "basis": jnp.asarray(artifact["basis"]),
            "action_mean": jnp.asarray(artifact["action_mean"]),
        }

    wp2_artifact = load_artifact(args.wp2_artifact or args.artifact)
    wp3_artifact = load_artifact(args.wp3_artifact or args.artifact)
    default_artifact = str(resolve_path(args.artifact))

    def load_command_artifact(path_text: str):
        artifact = pickle.load(open(resolve_path(path_text), "rb"))
        return {
            "path": str(resolve_path(path_text)),
            "params": artifact["params"],
            "feature_mean": jnp.asarray(artifact["feature_mean"]),
            "feature_std": jnp.asarray(artifact["feature_std"]),
            "head_names": tuple(artifact.get("head_names", ())),
        }

    wp3_command_artifact = load_command_artifact(args.wp3_command_artifact) if args.wp3_command_artifact else None

    @jax.jit
    def predict_wp2_action(features):
        coeff = apply_mlp(
            wp2_artifact["params"],
            (features - wp2_artifact["feature_mean"]) / wp2_artifact["feature_std"],
        )
        return jnp.clip(wp2_artifact["action_mean"] + coeff @ wp2_artifact["basis"], -1.0, 1.0)

    @jax.jit
    def predict_wp3_action(features):
        coeff = apply_mlp(
            wp3_artifact["params"],
            (features - wp3_artifact["feature_mean"]) / wp3_artifact["feature_std"],
        )
        return jnp.clip(wp3_artifact["action_mean"] + coeff @ wp3_artifact["basis"], -1.0, 1.0)

    if wp3_command_artifact is not None:
        @jax.jit
        def predict_wp3_command_head(features):
            return apply_mlp(
                wp3_command_artifact["params"],
                (features - wp3_command_artifact["feature_mean"]) / wp3_command_artifact["feature_std"],
            )
    else:
        predict_wp3_command_head = None

    waypoints = parse_waypoints(args.waypoints)
    route_scale = float(max(float(np.max(np.linalg.norm(waypoints, axis=1))), args.vmax, 1e-6))
    Env = _build()
    env = Env()
    step = jax.jit(env.step)
    state = env.reset_with_command(jax.random.PRNGKey(args.seed), jnp.zeros(2))
    reached = 0
    active_steps = {2: 0, 3: 0}
    prev_cmd = np.zeros(2, dtype=np.float32)
    prev_dist = None
    closest = [float("inf")] * len(waypoints)
    rec = []
    fall_reason = ""
    total_steps = int(args.steps_per_waypoint * len(waypoints))
    for t in range(total_steps):
        pos = np.asarray(state.pipeline_state.qpos[:2], dtype=np.float32)
        reached, target, dist, cmd = waypoint_command(pos, reached, waypoints, args.radius, args.gain, args.vmax)
        if reached >= len(waypoints):
            break
        route_wp = min(reached, len(waypoints) - 1)
        state = state.replace(
            info={
                **state.info,
                "cmd": jnp.asarray(cmd),
                "remote": jnp.array(True),
                "route_wp": jnp.asarray(route_wp, dtype=jnp.int32),
                "route_prev_dist": jnp.asarray(dist, dtype=jnp.float32),
            },
            obs=state.obs.at[-2:].set(jnp.asarray(cmd)),
        )
        delta = target - pos
        vxy = np.asarray(state.pipeline_state.qvel[:2], dtype=np.float32)
        phase = float(np.asarray(state.info["phase"]))
        progress = 0.0 if prev_dist is None else float(prev_dist - dist)
        residual_scale = 0.0
        since_switch = 0
        if route_wp == 2:
            residual_scale = float(args.wp2_residual_scale)
            since_switch = active_steps[2]
        elif route_wp == 3:
            residual_scale = float(args.wp3_residual_scale)
            since_switch = active_steps[3]
        prior_strength = 1.0 / (1.0 + max(residual_scale, 0.0))
        active_onehot = np.zeros((len(waypoints),), dtype=np.float32)
        active_onehot[route_wp] = 1.0
        features = np.concatenate([
            active_onehot,
            np.asarray([route_wp / max(len(waypoints) - 1, 1)], dtype=np.float32),
            delta.astype(np.float32) / route_scale,
            np.asarray([dist / route_scale], dtype=np.float32),
            cmd.astype(np.float32) / max(args.vmax, 1e-6),
            prev_cmd.astype(np.float32) / max(args.vmax, 1e-6),
            vxy / max(args.vmax, 1e-6),
            np.asarray([np.sin(phase), np.cos(phase)], dtype=np.float32),
            np.asarray([since_switch / max(args.steps_per_waypoint, 1)], dtype=np.float32),
            np.asarray([progress / max(args.vmax, 1e-6)], dtype=np.float32),
            np.asarray([prior_strength, residual_scale], dtype=np.float32),
        ]).astype(np.float32)
        command_head = np.zeros((6,), dtype=np.float32)
        wp3_action_mult = 1.0
        if route_wp == 3 and predict_wp3_command_head is not None:
            command_head = np.asarray(predict_wp3_command_head(jnp.asarray(features)), dtype=np.float32)
            correction = np.clip(
                command_head[:2],
                -float(args.wp3_command_clip),
                float(args.wp3_command_clip),
            ) * float(args.vmax) * float(args.wp3_command_blend)
            cmd = cmd + correction.astype(np.float32)
            cmd_norm = float(np.linalg.norm(cmd))
            if cmd_norm > args.vmax:
                cmd = cmd * (args.vmax / max(cmd_norm, 1e-6))
            wp3_action_mult = float(np.clip(
                command_head[2] * float(args.wp3_command_residual_gain),
                0.0,
                float(args.wp3_command_residual_max),
            ))
            state = state.replace(
                info={**state.info, "cmd": jnp.asarray(cmd)},
                obs=state.obs.at[-2:].set(jnp.asarray(cmd)),
            )
        if route_wp == 2:
            action = predict_wp2_action(jnp.asarray(features))
        elif route_wp == 3:
            action = predict_wp3_action(jnp.asarray(features)) * wp3_action_mult
        else:
            action = jnp.zeros(env.action_size)
        state = step(state, action)
        xy = np.asarray(state.pipeline_state.qpos[:2], dtype=np.float32)
        for i, wp in enumerate(waypoints):
            closest[i] = min(closest[i], float(np.linalg.norm(xy - wp)))
        up = 1.0 - 2.0 * (float(state.pipeline_state.qpos[4]) ** 2 + float(state.pipeline_state.qpos[5]) ** 2)
        rec.append([t, reached, float(target[0]), float(target[1]), float(cmd[0]), float(cmd[1]),
                    float(state.pipeline_state.qvel[0]), float(state.pipeline_state.qvel[1]),
                    float(xy[0]), float(xy[1]), float(state.pipeline_state.qpos[2]), up, dist,
                    float(jnp.linalg.norm(action)), float(command_head[0]), float(command_head[1]),
                    float(command_head[2]), float(wp3_action_mult)])
        if route_wp in active_steps:
            active_steps[route_wp] += 1
        prev_cmd = cmd
        prev_dist = dist
        if float(state.pipeline_state.qpos[2]) < FALL_Z:
            fall_reason = f"height<{FALL_Z:.3f}"
            break
        if up < MIN_UP_Z:
            fall_reason = f"up<{MIN_UP_Z:.3f}"
            break

    R = np.asarray(rec, dtype=np.float32)
    OUT.mkdir(parents=True, exist_ok=True)
    npz = OUT / f"{args.tag}_nav.npz"
    js = OUT / f"{args.tag}_nav_summary.json"
    if R.size:
        np.savez(npz, t=R[:, 0], waypoint=R[:, 1], target_x=R[:, 2], target_y=R[:, 3],
                 cmd_vx=R[:, 4], cmd_vy=R[:, 5], vx=R[:, 6], vy=R[:, 7],
                 x=R[:, 8], y=R[:, 9], z=R[:, 10], up=R[:, 11], dist=R[:, 12],
                 action_norm=R[:, 13], head_cmd_dx=R[:, 14], head_cmd_dy=R[:, 15],
                 head_residual_strength=R[:, 16], wp3_action_mult=R[:, 17],
                 waypoints=waypoints, radius=args.radius,
                 artifact=default_artifact, wp2_artifact=wp2_artifact["path"],
                 wp3_artifact=wp3_artifact["path"],
                 wp3_command_artifact=wp3_command_artifact["path"] if wp3_command_artifact else "",
                 tag=args.tag)
        final_dist = float(R[-1, 12])
        z_min = float(R[:, 10].min())
        up_min = float(R[:, 11].min())
    else:
        np.savez(npz, waypoints=waypoints, radius=args.radius, artifact=default_artifact,
                 wp2_artifact=wp2_artifact["path"], wp3_artifact=wp3_artifact["path"],
                 wp3_command_artifact=wp3_command_artifact["path"] if wp3_command_artifact else "",
                 tag=args.tag)
        final_dist = float("nan")
        z_min = float("nan")
        up_min = float("nan")
    closest_rows = []
    if R.size:
        xy = R[:, 8:10]
        for i, wp in enumerate(waypoints):
            d = np.linalg.norm(xy - wp, axis=1)
            j = int(d.argmin())
            closest_rows.append({
                "waypoint": int(i),
                "min_dist": float(d[j]),
                "closest_xy": xy[j].tolist(),
                "step": j,
                "inside_radius": bool(d[j] <= args.radius),
            })
    summary = {
        "tag": args.tag,
        "artifact": default_artifact,
        "wp2_artifact": wp2_artifact["path"],
        "wp3_artifact": wp3_artifact["path"],
        "wp3_command_artifact": wp3_command_artifact["path"] if wp3_command_artifact else "",
        "wp3_command_blend": float(args.wp3_command_blend),
        "wp3_command_clip": float(args.wp3_command_clip),
        "wp3_command_residual_gain": float(args.wp3_command_residual_gain),
        "wp3_command_residual_max": float(args.wp3_command_residual_max),
        "route_candidate": str(resolve_path(args.route_candidate)),
        "deploy_env_keys": sorted(deploy_env),
        "reached": int(reached),
        "total_waypoints": int(len(waypoints)),
        "success": bool(reached >= len(waypoints) and not fall_reason),
        "samples": int(len(R)),
        "budget": total_steps,
        "final_dist": final_dist,
        "z_min": z_min,
        "up_min": up_min,
        "fall_reason": fall_reason,
        "closest_waypoints": closest_rows,
        "waypoint2_min_dist": float(closest_rows[2]["min_dist"]) if len(closest_rows) > 2 else float("nan"),
        "active_steps": {str(k): int(v) for k, v in active_steps.items()},
    }
    js.write_text(json.dumps(summary, indent=2))
    print(f"[route-corrector-eval] reached {summary['reached']}/{summary['total_waypoints']} "
          f"success={int(summary['success'])} samples={summary['samples']} fall={fall_reason or 'none'}")
    print(json.dumps(summary, indent=2), flush=True)
    print(f"saved {npz} and {js}", flush=True)


if __name__ == "__main__":
    main()
