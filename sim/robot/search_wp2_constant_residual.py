# SPDX-License-Identifier: MIT
"""Search a constant or segmented residual action for one active route waypoint.

This is a cheap diagnostic between pure CPG search and full PPO.  It preserves
the route scaffold before the target waypoint, then applies a searched residual
action only while that waypoint is active.  If this improves the active
waypoint closest distance, the residual interface has a reachable correction
target; if it cannot improve, PPO is probably fighting an action/scale/objective
bottleneck rather than just a missing initialization.
"""

from __future__ import annotations

import argparse
import json
import os
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


def apply_route_env(path_text: str):
    data = json.loads(resolve_path(path_text).read_text())
    deploy_env = data.get("deploy_env") or {}
    if not deploy_env:
        raise ValueError(f"{path_text!r} has no deploy_env")
    for key, val in deploy_env.items():
        os.environ[str(key)] = str(val)
    return deploy_env


def parse_waypoints(text: str) -> np.ndarray:
    pts = []
    for chunk in text.split(";"):
        chunk = chunk.strip()
        if not chunk:
            continue
        x, y = chunk.split(",")
        pts.append((float(x), float(y)))
    return np.asarray(pts, dtype=np.float32)


def load_init_action(path_text: str, action_dim: int, segments: int) -> np.ndarray | None:
    if not path_text:
        return None
    data = json.loads(resolve_path(path_text).read_text())
    action = np.asarray(data.get("best_action"), dtype=np.float32)
    if action.shape == (action_dim,):
        return np.tile(action[None, :], (segments, 1))
    if action.ndim == 2 and action.shape[1] == action_dim:
        if action.shape[0] == segments:
            return action
        src_idx = np.linspace(0, action.shape[0] - 1, num=segments)
        return action[np.rint(src_idx).astype(int)]
    if action.shape == (segments, action_dim):
        return action
    if action.size == segments * action_dim:
        return action.reshape(segments, action_dim)
    raise ValueError(f"{path_text!r} best_action shape {action.shape} is incompatible with segments={segments}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tag", default="wp2_constant_residual")
    ap.add_argument("--route-candidate", required=True)
    ap.add_argument("--waypoints", default="0.35,0;0.35,0.35;0,0.35;0,0")
    ap.add_argument("--radius", type=float, default=0.07)
    ap.add_argument("--steps-per-waypoint", type=int, default=260)
    ap.add_argument("--gain", type=float, default=2.0)
    ap.add_argument("--vmax", type=float, default=0.35)
    ap.add_argument("--wp2-residual-scale", type=float, default=0.08,
                    help="legacy target residual scale; also used as the waypoint-2 prefix scale unless overridden")
    ap.add_argument("--target-residual-scale", type=float, default=None,
                    help="residual scale for the searched active target waypoint")
    ap.add_argument("--prefix-wp2-residual-scale", type=float, default=None,
                    help="residual scale for a preloaded waypoint-2 schedule while searching later waypoints")
    ap.add_argument("--prefix-wp2-action-json", default="",
                    help="preloaded waypoint-2 schedule used as prefix while searching later waypoints")
    ap.add_argument("--target-waypoint", type=int, default=2,
                    help="active waypoint index whose residual action is searched")
    ap.add_argument("--target-step-w", type=float, default=0.0,
                    help="penalty per control step spent on the active target; useful for earlier handoff searches")
    ap.add_argument("--stop-after-target-hit", action="store_true",
                    help="stop candidate evaluation after the searched target waypoint has been reached")
    ap.add_argument("--segments", type=int, default=1,
                    help="number of piecewise-constant residual actions across the waypoint-2 segment")
    ap.add_argument("--segment-steps", type=int, default=0,
                    help="steps per segment; default divides remaining route budget evenly")
    ap.add_argument("--init-action-json", default="",
                    help="previous *_wp2_constant_residual.json whose best_action seeds CEM")
    ap.add_argument("--pop", type=int, default=32)
    ap.add_argument("--gens", type=int, default=5)
    ap.add_argument("--elite-frac", type=float, default=0.25)
    ap.add_argument("--init-std", type=float, default=0.35)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    OUT.mkdir(parents=True, exist_ok=True)
    deploy_env = apply_route_env(args.route_candidate)
    target_scale = float(args.wp2_residual_scale if args.target_residual_scale is None else args.target_residual_scale)
    prefix_wp2_scale = float(
        args.wp2_residual_scale if args.prefix_wp2_residual_scale is None else args.prefix_wp2_residual_scale
    )
    wp2_scale = target_scale if int(args.target_waypoint) == 2 else prefix_wp2_scale
    wp3_scale = target_scale if int(args.target_waypoint) == 3 else 0.0
    os.environ.update({
        "CODESIGN_OUT": str(OUT),
        "CMD_TRAIN_MODE": "route",
        "CMD_REWARD_MODE": "route",
        "CMD_CONTROL_MODE": "cpg_pd",
        "CMD_CPG_RESIDUAL_SCALE": "0.0",
        "CMD_CPG_RESIDUAL_SCALE_TRANSITION": "0.0",
        "CMD_CPG_RESIDUAL_SCALE_WP2": str(wp2_scale),
        "CMD_CPG_RESIDUAL_SCALE_WP3": str(wp3_scale),
        "CMD_ROUTE_WAYPOINTS": args.waypoints,
        "CMD_ROUTE_RADIUS": str(args.radius),
        "CMD_ROUTE_GAIN": str(args.gain),
        "CMD_VMAX": str(args.vmax),
        "CMD_TRACK_SIGMA": "0.05",
        "CMD_RESET_NOISE": "0.0",
    })
    if args.prefix_wp2_action_json:
        os.environ["CMD_CPG_WP2_ACTION_JSON"] = str(resolve_path(args.prefix_wp2_action_json))

    # Import after env is set: commanded_env/cpg_teacher read CPG globals at import time.
    import jax
    import jax.numpy as jnp
    from commanded_env import _build

    waypoints = parse_waypoints(args.waypoints)
    target_waypoint = int(np.clip(args.target_waypoint, 0, max(0, len(waypoints) - 1)))
    total_steps = int(args.steps_per_waypoint * len(waypoints))
    env = _build()()
    step = jax.jit(env.step)
    zero = jnp.zeros(env.action_size, dtype=jnp.float32)
    rng = np.random.default_rng(args.seed)

    def waypoint_command(pos: np.ndarray, reached: int):
        target = waypoints[min(reached, len(waypoints) - 1)]
        delta = target - pos
        dist = float(np.linalg.norm(delta))
        if dist < args.radius and reached < len(waypoints):
            reached += 1
            if reached >= len(waypoints):
                return reached, target, dist, np.zeros(2, dtype=np.float32)
            target = waypoints[reached]
            delta = target - pos
            dist = float(np.linalg.norm(delta))
        cmd = args.gain * delta
        norm = float(np.linalg.norm(cmd))
        if norm > args.vmax:
            cmd = cmd * (args.vmax / norm)
        return reached, target, dist, np.asarray(cmd, dtype=np.float32)

    def sync_remote_state(state, reached: int, target: np.ndarray, dist: float, cmd: np.ndarray):
        # Keep the env's route bookkeeping aligned with eval_checkpoint_navigation's
        # external waypoint state so route context and waypoint-2 residual gates fire.
        return state.replace(
            info={
                **state.info,
                "cmd": jnp.asarray(cmd),
                "remote": jnp.array(True),
                "route_wp": jnp.asarray(min(reached, len(waypoints) - 1), dtype=jnp.int32),
                "route_prev_dist": jnp.asarray(dist, dtype=jnp.float32),
            },
            obs=state.obs.at[-2:].set(jnp.asarray(cmd)),
        )

    def replay_prefix():
        state = env.reset_with_command(jax.random.PRNGKey(args.seed), jnp.zeros(2))
        reached = 0
        closest = [float("inf")] * len(waypoints)
        prefix_steps = 0
        for t in range(total_steps):
            pos = np.asarray(state.pipeline_state.qpos[:2], dtype=np.float32)
            reached, target, dist, cmd = waypoint_command(pos, reached)
            if reached >= target_waypoint:
                prefix_steps = t
                break
            state = sync_remote_state(state, reached, target, dist, cmd)
            state = step(state, zero)
            xy = np.asarray(state.pipeline_state.qpos[:2], dtype=np.float32)
            for i, wp in enumerate(waypoints):
                closest[i] = min(closest[i], float(np.linalg.norm(xy - wp)))
            prefix_steps = t + 1
            if float(state.done) > 0.5:
                break
        return state, closest, prefix_steps, reached

    prefix_state, prefix_closest, prefix_steps, prefix_reached = replay_prefix()

    segments = max(1, int(args.segments))

    def eval_action(action_np: np.ndarray):
        actions_np = np.clip(np.asarray(action_np, dtype=np.float32), -1.0, 1.0).reshape(segments, env.action_size)
        actions = jnp.asarray(actions_np, dtype=jnp.float32)
        state = prefix_state
        closest = list(prefix_closest)
        reached = int(prefix_reached)
        reached_max = reached
        alive = True
        target_steps = 0
        target_min_dist = float("inf")
        default_segment_steps = max(1, int(np.ceil(max(1, total_steps - prefix_steps) / segments)))
        segment_steps = int(args.segment_steps) if args.segment_steps > 0 else default_segment_steps
        for t in range(prefix_steps, total_steps):
            pos = np.asarray(state.pipeline_state.qpos[:2], dtype=np.float32)
            reached, target, dist, cmd = waypoint_command(pos, reached)
            state = sync_remote_state(state, reached, target, dist, cmd)
            active_target = reached == target_waypoint
            if active_target:
                target_min_dist = min(target_min_dist, dist)
            seg_i = min(segments - 1, int(target_steps // segment_steps))
            act = actions[seg_i] if active_target else zero
            state = step(state, act)
            xy = np.asarray(state.pipeline_state.qpos[:2], dtype=np.float32)
            for i, wp in enumerate(waypoints):
                closest[i] = min(closest[i], float(np.linalg.norm(xy - wp)))
            if active_target:
                target_min_dist = min(target_min_dist, float(np.linalg.norm(xy - waypoints[target_waypoint])))
            reached_max = max(reached_max, reached)
            target_steps += int(active_target)
            if args.stop_after_target_hit and reached > target_waypoint:
                break
            if float(state.done) > 0.5:
                alive = False
                break
        final_xy = np.asarray(state.pipeline_state.qpos[:2], dtype=np.float32)
        final_dist = float(np.linalg.norm(final_xy - waypoints[min(reached_max, len(waypoints) - 1)]))
        if not np.isfinite(target_min_dist):
            target_min_dist = float(np.linalg.norm(final_xy - waypoints[target_waypoint]))
        # Primary score: reach the target waypoint and reduce its active distance;
        # secondary: avoid falls/final drift.
        score = (
            100.0 * float(reached_max)
            - 120.0 * target_min_dist
            - 8.0 * final_dist
            - float(args.target_step_w) * float(target_steps)
            - (0.0 if alive else 100.0)
        )
        return {
            "score": float(score),
            "reached_proxy": int(reached_max),
            "closest": [float(x) for x in closest],
            "target_waypoint": int(target_waypoint),
            "target_min_dist": float(target_min_dist),
            "waypoint2_min_dist": float(closest[2]) if len(closest) > 2 else float("nan"),
            "final_dist": final_dist,
            "alive": bool(alive),
            "target_steps": int(target_steps),
            "wp2_steps": int(target_steps if target_waypoint == 2 else 0),
            "final_xy": [float(final_xy[0]), float(final_xy[1])],
            "segment_steps": int(segment_steps),
        }

    baseline = eval_action(np.zeros((segments, env.action_size), dtype=np.float32))
    init_action = load_init_action(args.init_action_json, env.action_size, segments)
    mean = init_action if init_action is not None else np.zeros((segments, env.action_size), dtype=np.float32)
    std = np.full((segments, env.action_size), float(args.init_std), dtype=np.float32)
    elite_n = max(2, int(args.pop * args.elite_frac))
    best_action = mean.copy()
    best = baseline
    history = []
    for gen in range(args.gens):
        cand = mean[None, :, :] + std[None, :, :] * rng.standard_normal(
            (args.pop, segments, env.action_size)
        ).astype(np.float32)
        cand = np.clip(cand, -1.0, 1.0)
        cand[0] = mean
        if args.pop > 1:
            cand[1] = np.zeros((segments, env.action_size), dtype=np.float32)
        rows = [eval_action(c) for c in cand]
        order = np.argsort([-r["score"] for r in rows])
        if rows[int(order[0])]["score"] > best["score"]:
            best = rows[int(order[0])]
            best_action = cand[int(order[0])].copy()
        elites = cand[order[:elite_n]]
        mean = elites.mean(axis=0).astype(np.float32)
        std = elites.std(axis=0).astype(np.float32) + 0.02
        row = rows[int(order[0])]
        history.append({
            "gen": int(gen),
            "best_score": row["score"],
            "best_target_dist": row["target_min_dist"],
            "best_wp2": row["waypoint2_min_dist"],
            "best_reached_proxy": row["reached_proxy"],
            "all_best_target_dist": best["target_min_dist"],
            "all_best_wp2": best["waypoint2_min_dist"],
            "all_best_score": best["score"],
        })
        print(
            f"[wp2-const] gen {gen:02d} best score={row['score']:+.3f} "
            f"target={row['target_min_dist']:.4f} wp2={row['waypoint2_min_dist']:.4f} "
            f"reached~{row['reached_proxy']} allbest={best['target_min_dist']:.4f}",
            flush=True,
        )

    report = {
        "tag": args.tag,
        "route_candidate": str(resolve_path(args.route_candidate)),
        "deploy_env_keys": sorted(deploy_env),
        "wp2_residual_scale": float(args.wp2_residual_scale),
        "target_residual_scale": float(target_scale),
        "prefix_wp2_residual_scale": float(prefix_wp2_scale),
        "prefix_wp2_action_json": str(resolve_path(args.prefix_wp2_action_json)) if args.prefix_wp2_action_json else "",
        "target_waypoint": int(target_waypoint),
        "target_step_w": float(args.target_step_w),
        "stop_after_target_hit": bool(args.stop_after_target_hit),
        "segments": int(segments),
        "segment_steps_arg": int(args.segment_steps),
        "init_action_json": str(args.init_action_json or ""),
        "prefix_steps": int(prefix_steps),
        "prefix_route_wp": int(np.asarray(prefix_state.info["route_wp"])),
        "baseline": baseline,
        "best": best,
        "best_action": np.asarray(best_action, dtype=np.float32).tolist(),
        "history": history,
        "prefix_reached": int(prefix_reached),
    }
    out = OUT / f"{args.tag}_wp2_constant_residual.json"
    out.write_text(json.dumps(report, indent=2))
    print(json.dumps(report, indent=2), flush=True)
    print(f"saved {out}", flush=True)


if __name__ == "__main__":
    main()
