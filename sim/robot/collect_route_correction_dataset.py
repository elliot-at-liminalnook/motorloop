# SPDX-License-Identifier: MIT
"""Collect route-state correction labels from scheduled residual rollouts.

The current CPG scaffold plus searched waypoint schedules is useful as an
oracle, but brittle as a controller.  This collector turns those schedules into
closed-loop training data for a feedback corrector:

    route state features -> low-dimensional residual coefficients

The labels include both the full residual action and compact PCA coefficients
over the schedule-action basis.  A later trainer can learn the compact
state-conditioned correction while validation remains closed-loop rollout based.
"""

from __future__ import annotations

import argparse
import json
import os
import pickle
import sys
from pathlib import Path

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


def parse_seeds(text: str) -> list[int]:
    vals = []
    for chunk in text.replace(";", ",").split(","):
        chunk = chunk.strip()
        if chunk:
            vals.append(int(chunk))
    if not vals:
        raise ValueError("at least one seed is required")
    return vals


def parse_start_states(text: str) -> list[tuple[int, float, float]]:
    vals = []
    for chunk in text.split(";"):
        chunk = chunk.strip()
        if not chunk:
            continue
        parts = [p.strip() for p in chunk.split(",")]
        if len(parts) != 3:
            raise ValueError("--start-states entries must be 'waypoint,x,y'")
        vals.append((int(parts[0]), float(parts[1]), float(parts[2])))
    return vals


def load_schedule(path_text: str, action_dim: int) -> tuple[np.ndarray, int]:
    if not path_text:
        return np.zeros((0, action_dim), dtype=np.float32), 1
    data = json.loads(resolve_path(path_text).read_text())
    action = np.asarray(data.get("best_action"), dtype=np.float32)
    if action.ndim == 1:
        action = action[None, :]
    if action.ndim != 2 or action.shape[1] != action_dim:
        raise ValueError(f"{path_text!r} best_action shape {action.shape} incompatible with action_dim={action_dim}")
    segment_steps = int(data.get("best", {}).get("segment_steps", 0) or 1)
    return np.clip(action, -1.0, 1.0).astype(np.float32), max(1, segment_steps)


def make_basis(actions: list[np.ndarray], rank: int) -> tuple[np.ndarray, np.ndarray]:
    rows = [a for a in actions if a.size]
    if not rows:
        raise ValueError("at least one non-empty schedule is required to build a correction basis")
    mat = np.concatenate(rows, axis=0).astype(np.float32)
    mean = mat.mean(axis=0)
    centered = mat - mean
    _, _, vh = np.linalg.svd(centered, full_matrices=False)
    rank = max(1, min(int(rank), vh.shape[0]))
    return mean.astype(np.float32), vh[:rank].astype(np.float32)


def apply_mlp(params, x):
    import jax.numpy as jnp

    for layer in params[:-1]:
        x = jnp.tanh(x @ layer["w"] + layer["b"])
    return x @ params[-1]["w"] + params[-1]["b"]


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


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tag", default="route_correction")
    ap.add_argument("--route-candidate", required=True)
    ap.add_argument("--wp2-action-json", required=True)
    ap.add_argument("--wp3-action-json", default="")
    ap.add_argument("--out-dir", default=str(OUT / "route_correction_dataset"))
    ap.add_argument("--waypoints", default="0.35,0;0.35,0.35;0,0.35;0,0")
    ap.add_argument("--radius", type=float, default=0.07)
    ap.add_argument("--steps-per-waypoint", type=int, default=260)
    ap.add_argument("--gain", type=float, default=2.0)
    ap.add_argument("--vmax", type=float, default=0.35)
    ap.add_argument("--wp2-residual-scale", type=float, default=0.64)
    ap.add_argument("--wp3-residual-scale", type=float, default=0.64)
    ap.add_argument("--seeds", default="53")
    ap.add_argument("--basis-rank", type=int, default=4)
    ap.add_argument("--reset-noise", type=float, default=0.0,
                    help="motor reset noise passed through CMD_RESET_NOISE")
    ap.add_argument("--xy-noise", type=float, default=0.0,
                    help="uniform base XY reset perturbation in meters")
    ap.add_argument("--phase-jitter", type=float, default=0.0,
                    help="uniform initial CPG phase perturbation in radians")
    ap.add_argument("--start-states", default="",
                    help="optional semicolon list of 'waypoint,x,y' reset states for targeted handoff collection")
    ap.add_argument("--start-jitter", type=float, default=0.0,
                    help="uniform XY jitter around --start-states in meters")
    ap.add_argument("--rollout-policy", choices=("schedule", "zero", "corrector"), default="schedule",
                    help="policy used to generate visited states; labels still come from the schedule oracle")
    ap.add_argument("--rollout-artifact", default="",
                    help="compact corrector artifact for --rollout-policy=corrector")
    ap.add_argument("--rollout-wp2-artifact", default="",
                    help="optional compact corrector artifact used only at active waypoint 2 during rollout")
    ap.add_argument("--rollout-wp3-artifact", default="",
                    help="optional compact corrector artifact used only at active waypoint 3 during rollout")
    args = ap.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    deploy_env = apply_route_env(args.route_candidate)
    os.environ.pop("CMD_CPG_WP2_ACTION_JSON", None)
    os.environ.pop("CMD_CPG_WP3_ACTION_JSON", None)
    os.environ.pop("CMD_CPG_WP2_ACTION_SEGMENT_STEPS", None)
    os.environ.pop("CMD_CPG_WP3_ACTION_SEGMENT_STEPS", None)
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
        "CMD_RESET_NOISE": str(args.reset_noise),
    })
    if args.rollout_policy == "schedule":
        os.environ["CMD_CPG_WP2_ACTION_JSON"] = str(resolve_path(args.wp2_action_json))
    if args.rollout_policy == "schedule" and args.wp3_action_json:
        os.environ["CMD_CPG_WP3_ACTION_JSON"] = str(resolve_path(args.wp3_action_json))

    import jax
    import jax.numpy as jnp
    from commanded_env import _build

    waypoints = parse_waypoints(args.waypoints)
    seeds = parse_seeds(args.seeds)
    env = _build()()
    step = jax.jit(env.step)
    zero = jnp.zeros(env.action_size, dtype=jnp.float32)
    wp2_actions, wp2_steps = load_schedule(args.wp2_action_json, env.action_size)
    wp3_actions, wp3_steps = load_schedule(args.wp3_action_json, env.action_size)
    action_mean, basis = make_basis([wp2_actions, wp3_actions], args.basis_rank)

    def load_artifact(path_text: str):
        artifact = pickle.load(open(resolve_path(path_text), "rb"))
        return {
            "params": artifact["params"],
            "feature_mean": jnp.asarray(artifact["feature_mean"]),
            "feature_std": jnp.asarray(artifact["feature_std"]),
            "basis": jnp.asarray(artifact["basis"]),
            "action_mean": jnp.asarray(artifact["action_mean"]),
            "path": str(resolve_path(path_text)),
        }

    wp2_artifact = None
    wp3_artifact = None
    if args.rollout_policy == "corrector":
        wp2_path = args.rollout_wp2_artifact or args.rollout_artifact
        wp3_path = args.rollout_wp3_artifact or args.rollout_artifact
        if not wp2_path or not wp3_path:
            raise ValueError("--rollout-policy=corrector requires --rollout-artifact or both waypoint artifacts")
        wp2_artifact = load_artifact(wp2_path)
        wp3_artifact = load_artifact(wp3_path)

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

    start_states = parse_start_states(args.start_states)
    route_scale = float(max(float(np.max(np.linalg.norm(waypoints, axis=1))), args.vmax, 1e-6))
    total_steps = int(args.steps_per_waypoint * len(waypoints))
    rows = {
        "features": [],
        "coeff": [],
        "residual_action": [],
        "rollout_action": [],
        "command_target": [],
        "command_correction": [],
        "head_target": [],
        "active": [],
        "xy": [],
        "target": [],
        "cmd": [],
        "vel": [],
        "phase": [],
        "dist": [],
        "progress": [],
        "alive": [],
        "seed": [],
        "step": [],
    }
    summaries = []

    for seed_i, seed in enumerate(seeds):
        key = jax.random.PRNGKey(seed)
        state = env.reset_with_command(key, jnp.zeros(2))
        reached = 0
        if start_states:
            start_wp, start_x, start_y = start_states[seed_i % len(start_states)]
            reached = int(np.clip(start_wp, 0, len(waypoints) - 1))
            pipe = state.pipeline_state.replace(
                qpos=state.pipeline_state.qpos.at[:2].set(jnp.asarray([start_x, start_y], dtype=jnp.float32))
            )
            state = state.replace(pipeline_state=pipe)
        if args.xy_noise > 0.0 or args.start_jitter > 0.0 or args.phase_jitter > 0.0:
            xy_key, start_key, phase_key = jax.random.split(jax.random.fold_in(key, 911), 3)
            xy_delta = jnp.zeros((2,), dtype=jnp.float32)
            if args.xy_noise > 0.0:
                xy_delta = xy_delta + jax.random.uniform(
                    xy_key, (2,), minval=-float(args.xy_noise), maxval=float(args.xy_noise)
                )
            if start_states and args.start_jitter > 0.0:
                xy_delta = xy_delta + jax.random.uniform(
                    start_key, (2,), minval=-float(args.start_jitter), maxval=float(args.start_jitter)
                )
            if args.xy_noise > 0.0 or (start_states and args.start_jitter > 0.0):
                pipe = state.pipeline_state.replace(
                    qpos=state.pipeline_state.qpos.at[:2].add(xy_delta)
                )
                state = state.replace(pipeline_state=pipe)
            if args.phase_jitter > 0.0:
                phase_delta = jax.random.uniform(
                    phase_key, (), minval=-float(args.phase_jitter), maxval=float(args.phase_jitter)
                )
                state = state.replace(info={**state.info, "phase": state.info["phase"] + phase_delta})
        active_steps = {2: 0, 3: 0}
        prev_cmd = np.zeros(2, dtype=np.float32)
        prev_dist = None
        fall = ""
        closest = [float("inf")] * len(waypoints)
        for t in range(total_steps):
            pos = np.asarray(state.pipeline_state.qpos[:2], dtype=np.float32)
            reached, target, dist, cmd = waypoint_command(
                pos, reached, waypoints, args.radius, args.gain, args.vmax)
            if reached >= len(waypoints):
                break
            route_wp = min(reached, len(waypoints) - 1)
            info = {
                **state.info,
                "cmd": jnp.asarray(cmd),
                "remote": jnp.array(True),
                "route_wp": jnp.asarray(route_wp, dtype=jnp.int32),
                "route_prev_dist": jnp.asarray(dist, dtype=jnp.float32),
            }
            state = state.replace(info=info, obs=state.obs.at[-2:].set(jnp.asarray(cmd)))

            label = np.zeros((env.action_size,), dtype=np.float32)
            residual_scale = 0.0
            since_switch = 0
            if route_wp == 2 and len(wp2_actions):
                idx = min(len(wp2_actions) - 1, active_steps[2] // wp2_steps)
                label = wp2_actions[idx]
                residual_scale = float(args.wp2_residual_scale)
                since_switch = active_steps[2]
            elif route_wp == 3 and len(wp3_actions):
                idx = min(len(wp3_actions) - 1, active_steps[3] // wp3_steps)
                label = wp3_actions[idx]
                residual_scale = float(args.wp3_residual_scale)
                since_switch = active_steps[3]

            delta = target - pos
            vxy = np.asarray(state.pipeline_state.qvel[:2], dtype=np.float32)
            phase = float(np.asarray(state.info["phase"]))
            progress = 0.0 if prev_dist is None else float(prev_dist - dist)
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
            coeff = (label - action_mean) @ basis.T
            rollout_action = zero
            if args.rollout_policy == "corrector":
                if route_wp == 2:
                    rollout_action = predict_wp2_action(jnp.asarray(features))
                elif route_wp == 3:
                    rollout_action = predict_wp3_action(jnp.asarray(features))

            state = step(state, rollout_action)
            xy = np.asarray(state.pipeline_state.qpos[:2], dtype=np.float32)
            if args.rollout_policy == "schedule":
                # Schedule rollouts provide a local oracle motion target.  Store it as a
                # command-style label so a later learner can steer the CPG interface
                # directly instead of only imitating 12-D residual motor actions.
                dt = float(getattr(env, "_dt", 1.0))
                desired_cmd = (xy - pos) / max(dt, 1e-6)
                desired_norm = float(np.linalg.norm(desired_cmd))
                if desired_norm > args.vmax:
                    desired_cmd = desired_cmd * (args.vmax / max(desired_norm, 1e-6))
            else:
                # For learner-state rollouts we label the same state with the local
                # schedule action but do not simulate a separate oracle step.  Fall back
                # to the waypoint command for the command head.
                desired_cmd = cmd
            command_correction = desired_cmd.astype(np.float32) - cmd.astype(np.float32)
            prev_norm = float(np.linalg.norm(prev_cmd))
            cmd_norm = float(np.linalg.norm(cmd))
            turn_bias = 0.0
            if prev_norm > 1e-6 and cmd_norm > 1e-6:
                turn_bias = float((prev_cmd[0] * cmd[1] - prev_cmd[1] * cmd[0]) / (prev_norm * cmd_norm))
                turn_bias = float(np.clip(turn_bias, -1.0, 1.0))
            max_residual_scale = max(float(args.wp2_residual_scale), float(args.wp3_residual_scale), 1e-6)
            residual_strength = float(np.clip(residual_scale / max_residual_scale, 0.0, 1.0))
            handoff_timing = float(np.clip(since_switch / max(args.steps_per_waypoint, 1), 0.0, 2.0))
            phase_speed_adjust = float(np.clip(np.linalg.norm(cmd - prev_cmd) / max(args.vmax, 1e-6), 0.0, 2.0))
            head_target = np.asarray([
                command_correction[0] / max(args.vmax, 1e-6),
                command_correction[1] / max(args.vmax, 1e-6),
                residual_strength,
                turn_bias,
                phase_speed_adjust,
                handoff_timing,
            ], dtype=np.float32)
            rows["features"].append(features)
            rows["coeff"].append(coeff.astype(np.float32))
            rows["residual_action"].append(label.astype(np.float32))
            rows["rollout_action"].append(np.asarray(rollout_action, dtype=np.float32))
            rows["command_target"].append(desired_cmd.astype(np.float32))
            rows["command_correction"].append(command_correction.astype(np.float32))
            rows["head_target"].append(head_target)
            rows["active"].append(route_wp)
            rows["xy"].append(pos.astype(np.float32))
            rows["target"].append(target.astype(np.float32))
            rows["cmd"].append(cmd.astype(np.float32))
            rows["vel"].append(vxy.astype(np.float32))
            rows["phase"].append(phase)
            rows["dist"].append(dist)
            rows["progress"].append(progress)
            rows["alive"].append(1.0)
            rows["seed"].append(seed)
            rows["step"].append(t)
            for i, wp in enumerate(waypoints):
                closest[i] = min(closest[i], float(np.linalg.norm(xy - wp)))
            if route_wp in active_steps:
                active_steps[route_wp] += 1
            prev_cmd = cmd
            prev_dist = dist
            up = 1.0 - 2.0 * (float(state.pipeline_state.qpos[4]) ** 2 + float(state.pipeline_state.qpos[5]) ** 2)
            if float(state.pipeline_state.qpos[2]) < 0.10:
                fall = "height"
                break
            if up < 0.25:
                fall = "up"
                break
        summaries.append({
            "seed": int(seed),
            "reached": int(reached),
            "closest": [float(x) for x in closest],
            "fall": fall,
            "samples": int(sum(1 for s in rows["seed"] if s == seed)),
        })

    arrays = {k: np.asarray(v, dtype=np.float32) for k, v in rows.items()
              if k not in ("active", "seed", "step")}
    arrays["active"] = np.asarray(rows["active"], dtype=np.int32)
    arrays["seed"] = np.asarray(rows["seed"], dtype=np.int32)
    arrays["step"] = np.asarray(rows["step"], dtype=np.int32)
    arrays["basis"] = basis.astype(np.float32)
    arrays["action_mean"] = action_mean.astype(np.float32)
    out_npz = out_dir / f"{args.tag}.npz"
    np.savez(out_npz, **arrays)
    manifest = {
        "tag": args.tag,
        "out_npz": str(out_npz),
        "route_candidate": str(resolve_path(args.route_candidate)),
        "deploy_env_keys": sorted(deploy_env),
        "wp2_action_json": str(resolve_path(args.wp2_action_json)),
        "wp3_action_json": str(resolve_path(args.wp3_action_json)) if args.wp3_action_json else "",
        "feature_dim": int(arrays["features"].shape[1]) if len(rows["features"]) else 0,
        "coeff_dim": int(basis.shape[0]),
        "head_dim": 6,
        "action_dim": int(env.action_size),
        "samples": int(len(rows["active"])),
        "seeds": seeds,
        "reset_noise": float(args.reset_noise),
        "xy_noise": float(args.xy_noise),
        "phase_jitter": float(args.phase_jitter),
        "start_states": [{"waypoint": int(wp), "x": float(x), "y": float(y)} for wp, x, y in start_states],
        "start_jitter": float(args.start_jitter),
        "rollout_policy": args.rollout_policy,
        "rollout_artifact": str(resolve_path(args.rollout_artifact)) if args.rollout_artifact else "",
        "rollout_wp2_artifact": wp2_artifact["path"] if wp2_artifact is not None else "",
        "rollout_wp3_artifact": wp3_artifact["path"] if wp3_artifact is not None else "",
        "summaries": summaries,
    }
    out_json = out_dir / f"{args.tag}_manifest.json"
    out_json.write_text(json.dumps(manifest, indent=2))
    print(json.dumps(manifest, indent=2), flush=True)
    print(f"saved {out_npz} and {out_json}", flush=True)


if __name__ == "__main__":
    main()
