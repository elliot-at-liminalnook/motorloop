# SPDX-License-Identifier: MIT
"""Search piecewise residual schedules against the full checkpoint route.

This is a deliberately small black-box diagnostic between hand-authored CPG
search and full residual PPO.  Unlike ``search_wp2_constant_residual.py``, this
optimizes multiple waypoint-local schedules together, so waypoint-2 timing and
the waypoint-3 return correction can trade off under the same route score.
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


def apply_route_env(path_text: str) -> dict[str, str]:
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
    if not pts:
        raise ValueError("at least one waypoint is required")
    return np.asarray(pts, dtype=np.float32)


def parse_ints(text: str) -> list[int]:
    vals = []
    for chunk in text.split(","):
        chunk = chunk.strip()
        if chunk:
            vals.append(int(chunk))
    if not vals:
        raise ValueError("at least one waypoint index is required")
    return vals


def load_action_schedule(path_text: str, action_dim: int, segments: int) -> tuple[np.ndarray, int]:
    data = json.loads(resolve_path(path_text).read_text())
    action = np.asarray(data.get("best_action"), dtype=np.float32)
    if action.ndim == 1:
        action = action[None, :]
    if action.ndim != 2 or action.shape[1] != action_dim:
        raise ValueError(f"{path_text!r} best_action shape {action.shape} incompatible with action_dim={action_dim}")
    if action.shape[0] != segments:
        src_idx = np.linspace(0, action.shape[0] - 1, num=segments)
        action = action[np.rint(src_idx).astype(int)]
    segment_steps = int(data.get("best", {}).get("segment_steps", 0) or 1)
    return np.clip(action, -1.0, 1.0).astype(np.float32), max(1, segment_steps)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tag", default="route_residual_schedule")
    ap.add_argument("--route-candidate", required=True)
    ap.add_argument("--waypoints", default="0.35,0;0.35,0.35;0,0.35;0,0")
    ap.add_argument("--search-waypoints", default="2,3",
                    help="comma-separated active waypoint indices whose residual schedules are optimized")
    ap.add_argument("--radius", type=float, default=0.07)
    ap.add_argument("--steps-per-waypoint", type=int, default=260)
    ap.add_argument("--gain", type=float, default=2.0)
    ap.add_argument("--vmax", type=float, default=0.35)
    ap.add_argument("--wp2-residual-scale", type=float, default=0.64)
    ap.add_argument("--wp3-residual-scale", type=float, default=0.64)
    ap.add_argument("--segments", type=int, default=3)
    ap.add_argument("--segment-steps", type=int, default=0,
                    help="steps per segment for all searched waypoints; default uses init schedule values or route budget")
    ap.add_argument("--init-wp2-action-json", default="")
    ap.add_argument("--init-wp3-action-json", default="")
    ap.add_argument("--pop", type=int, default=16)
    ap.add_argument("--gens", type=int, default=4)
    ap.add_argument("--elite-frac", type=float, default=0.25)
    ap.add_argument("--init-std", type=float, default=0.25)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    OUT.mkdir(parents=True, exist_ok=True)
    deploy_env = apply_route_env(args.route_candidate)
    search_waypoints = parse_ints(args.search_waypoints)
    segments = max(1, int(args.segments))
    total_steps = int(args.steps_per_waypoint * len(parse_waypoints(args.waypoints)))
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

    # Import after env is set: commanded_env/cpg_teacher read globals at import time.
    import jax
    import jax.numpy as jnp
    from commanded_env import _build

    waypoints = parse_waypoints(args.waypoints)
    env = _build()()
    step = jax.jit(env.step)
    zero = jnp.zeros(env.action_size, dtype=jnp.float32)
    rng = np.random.default_rng(args.seed)
    wp_to_slot = {wp: i for i, wp in enumerate(search_waypoints)}

    segment_steps = {
        wp: int(args.segment_steps) if args.segment_steps > 0 else max(1, int(np.ceil(total_steps / len(waypoints) / segments)))
        for wp in search_waypoints
    }
    mean = np.zeros((len(search_waypoints), segments, env.action_size), dtype=np.float32)
    init_paths = {2: args.init_wp2_action_json, 3: args.init_wp3_action_json}
    for wp, path in init_paths.items():
        if wp in wp_to_slot and path:
            action, steps = load_action_schedule(path, env.action_size, segments)
            mean[wp_to_slot[wp]] = action
            if args.segment_steps <= 0:
                segment_steps[wp] = steps

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

    def sync_remote_state(state, reached: int, dist: float, cmd: np.ndarray):
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

    def eval_actions(actions_np: np.ndarray):
        actions_np = np.clip(np.asarray(actions_np, dtype=np.float32), -1.0, 1.0)
        actions_np = actions_np.reshape(len(search_waypoints), segments, env.action_size)
        actions = jnp.asarray(actions_np, dtype=jnp.float32)
        state = env.reset_with_command(jax.random.PRNGKey(args.seed), jnp.zeros(2))
        reached = 0
        reached_max = 0
        closest = [float("inf")] * len(waypoints)
        active_closest = [float("inf")] * len(waypoints)
        active_steps = {wp: 0 for wp in search_waypoints}
        hit_steps: dict[str, int] = {}
        alive = True
        success = False
        last_dist = float("inf")
        last_target = waypoints[0]
        final_xy = np.zeros(2, dtype=np.float32)

        for t in range(total_steps):
            pos = np.asarray(state.pipeline_state.qpos[:2], dtype=np.float32)
            before = reached
            reached, target, dist, cmd = waypoint_command(pos, reached)
            if reached > before:
                hit_steps[str(before)] = int(t)
            if reached >= len(waypoints):
                success = True
                reached_max = max(reached_max, reached)
                break
            last_dist = dist
            last_target = target
            state = sync_remote_state(state, reached, dist, cmd)
            if reached < len(active_closest):
                active_closest[reached] = min(active_closest[reached], dist)
            if reached in wp_to_slot:
                slot = wp_to_slot[reached]
                seg_i = min(segments - 1, active_steps[reached] // max(segment_steps[reached], 1))
                act = actions[slot, seg_i]
                active_steps[reached] += 1
            else:
                act = zero
            state = step(state, act)
            xy = np.asarray(state.pipeline_state.qpos[:2], dtype=np.float32)
            final_xy = xy
            for i, wp in enumerate(waypoints):
                closest[i] = min(closest[i], float(np.linalg.norm(xy - wp)))
            if reached < len(active_closest):
                active_closest[reached] = min(active_closest[reached], float(np.linalg.norm(xy - waypoints[reached])))
            reached_max = max(reached_max, reached)
            if float(state.done) > 0.5:
                alive = False
                break

        if not success:
            final_dist = float(np.linalg.norm(final_xy - last_target)) if np.isfinite(last_dist) else float("inf")
        else:
            final_dist = 0.0
        active_score_dist = active_closest[min(reached_max, len(waypoints) - 1)]
        if not np.isfinite(active_score_dist):
            active_score_dist = final_dist
        wp2_steps = int(active_steps.get(2, 0))
        wp3_steps = int(active_steps.get(3, 0))
        # Lexicographic intent in scalar form: reach more waypoints first, then
        # shrink the active target miss, prefer earlier waypoint-2 handoff, and
        # keep the body alive.
        score = (
            1000.0 * float(reached_max)
            + 500.0 * float(success)
            - 260.0 * float(active_score_dist)
            - 30.0 * float(final_dist)
            - 0.03 * float(wp2_steps)
            - (0.0 if alive else 500.0)
        )
        return {
            "score": float(score),
            "reached": int(reached_max),
            "total_waypoints": int(len(waypoints)),
            "success": bool(success and alive),
            "closest": [float(x) for x in closest],
            "active_closest": [float(x) if np.isfinite(x) else None for x in active_closest],
            "final_dist": float(final_dist),
            "final_xy": [float(final_xy[0]), float(final_xy[1])],
            "alive": bool(alive),
            "active_steps": {str(k): int(v) for k, v in active_steps.items()},
            "hit_steps": hit_steps,
            "segment_steps": {str(k): int(v) for k, v in segment_steps.items()},
        }

    baseline = eval_actions(mean)
    zero_baseline = eval_actions(np.zeros_like(mean))
    std = np.full_like(mean, float(args.init_std), dtype=np.float32)
    elite_n = max(2, int(args.pop * args.elite_frac))
    best_action = mean.copy()
    best = baseline
    history = []
    for gen in range(args.gens):
        cand = mean[None, :, :, :] + std[None, :, :, :] * rng.standard_normal(
            (args.pop, len(search_waypoints), segments, env.action_size)
        ).astype(np.float32)
        cand = np.clip(cand, -1.0, 1.0)
        cand[0] = mean
        if args.pop > 1:
            cand[1] = np.zeros_like(mean)
        rows = [eval_actions(c) for c in cand]
        order = np.argsort([-r["score"] for r in rows])
        top = rows[int(order[0])]
        if top["score"] > best["score"]:
            best = top
            best_action = cand[int(order[0])].copy()
        elites = cand[order[:elite_n]]
        mean = elites.mean(axis=0).astype(np.float32)
        std = elites.std(axis=0).astype(np.float32) + 0.02
        history.append({
            "gen": int(gen),
            "best_score": float(top["score"]),
            "best_reached": int(top["reached"]),
            "best_success": bool(top["success"]),
            "best_final_dist": float(top["final_dist"]),
            "all_best_score": float(best["score"]),
            "all_best_reached": int(best["reached"]),
            "all_best_final_dist": float(best["final_dist"]),
        })
        print(
            f"[route-sched] gen {gen:02d} best score={top['score']:+.3f} "
            f"reached={top['reached']}/{len(waypoints)} success={int(top['success'])} "
            f"final={top['final_dist']:.4f} allbest={best['final_dist']:.4f}",
            flush=True,
        )

    action_by_wp = {
        str(wp): np.asarray(best_action[wp_to_slot[wp]], dtype=np.float32).tolist()
        for wp in search_waypoints
    }
    report = {
        "tag": args.tag,
        "route_candidate": str(resolve_path(args.route_candidate)),
        "deploy_env_keys": sorted(deploy_env),
        "search_waypoints": [int(wp) for wp in search_waypoints],
        "segments": int(segments),
        "steps_per_waypoint": int(args.steps_per_waypoint),
        "radius": float(args.radius),
        "wp2_residual_scale": float(args.wp2_residual_scale),
        "wp3_residual_scale": float(args.wp3_residual_scale),
        "init_wp2_action_json": str(args.init_wp2_action_json or ""),
        "init_wp3_action_json": str(args.init_wp3_action_json or ""),
        "baseline": baseline,
        "zero_baseline": zero_baseline,
        "best": best,
        "best_action_by_waypoint": action_by_wp,
        "history": history,
    }
    report_path = OUT / f"{args.tag}_route_residual_schedule.json"
    report_path.write_text(json.dumps(report, indent=2))
    for wp in search_waypoints:
        slot = wp_to_slot[wp]
        schedule = {
            "tag": args.tag,
            "source_report": str(report_path),
            "waypoint": int(wp),
            "best_action": np.asarray(best_action[slot], dtype=np.float32).tolist(),
            "best": {
                "segment_steps": int(segment_steps[wp]),
                "route_score": float(best["score"]),
                "reached": int(best["reached"]),
                "final_dist": float(best["final_dist"]),
            },
        }
        schedule_path = OUT / f"{args.tag}_wp{wp}_schedule.json"
        schedule_path.write_text(json.dumps(schedule, indent=2))
    print(json.dumps(report, indent=2), flush=True)
    print(f"saved {report_path}", flush=True)


if __name__ == "__main__":
    main()
