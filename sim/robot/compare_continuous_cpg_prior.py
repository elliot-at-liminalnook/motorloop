# SPDX-License-Identifier: MIT
"""Compare cardinal routing against continuous CPG-parameter routing.

This is a diagnostic, not a search pass.  It reuses a frozen route candidate and
asks whether different router interfaces make the same CPG priors compose
better on the square route and on short command-transition probes.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import sys
from pathlib import Path

import jax.numpy as jnp
import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import cpg_teacher as cpg_teacher_mod  # noqa: E402
from collect_transition_dataset import (  # noqa: E402
    COMMAND_TRANSITIONS,
    DIR_CMDS,
    build_evaler,
    command_sequence,
    load_candidate,
)
from search_cpg_route_mjx import load_route  # noqa: E402

OUT = Path(os.environ.get("CODESIGN_OUT", "/root/proj/out"))

ROUTE_STAT_NAMES = (
    "score",
    "reached",
    "route_progress",
    "final_dist",
    "survived_frac",
    "mean_up",
    "min_z",
    "saturation",
    "mean_cross",
    "max_cross",
    "mean_backtrack",
    "final_x",
    "final_y",
    "active_final",
    "focus_progress",
    "focus_dist",
    "focus_cross",
)


def route_stats_dict(stats) -> dict[str, float | int]:
    arr = np.asarray(stats, dtype=float)
    out = {name: float(arr[i]) for i, name in enumerate(ROUTE_STAT_NAMES)}
    out["reached"] = int(round(out["reached"]))
    out["active_final"] = int(round(out["active_final"]))
    return out


def unit(v: np.ndarray) -> np.ndarray:
    n = float(np.linalg.norm(v))
    if n <= 1e-9:
        return np.asarray([1.0, 0.0], dtype=np.float32)
    return (v / n).astype(np.float32)


def transition_summary(evaler, z_flat, prev_name: str, next_name: str, phases: np.ndarray, args) -> dict:
    cmds = command_sequence(
        prev_name,
        next_name,
        args.transition_speed,
        args.transition_warmup_steps,
        args.transition_steps,
    )
    direction = unit(DIR_CMDS[next_name])
    side = np.asarray([-direction[1], direction[0]], dtype=np.float32)
    rows = []
    for phase in phases:
        hist = evaler.eval_command_dataset(
            z_flat,
            jnp.asarray(cmds, dtype=jnp.float32),
            jnp.asarray(float(phase), dtype=jnp.float32),
        )
        pre_xy = np.asarray(hist[2])
        post_xy = np.asarray(hist[3])
        alive = np.asarray(hist[7])
        sat = np.asarray(hist[6])
        dist_before = np.asarray(hist[15])
        dist_after = np.asarray(hist[16])
        step_progress = np.asarray(hist[17])
        cross = np.asarray(hist[19])
        start = min(max(int(args.transition_warmup_steps), 0), pre_xy.shape[0] - 1)
        sl = slice(start, None)
        net_delta = post_xy[-1] - pre_xy[start]
        rows.append({
            "survived_frac": float(alive[sl].mean()),
            "mean_saturation": float(sat[sl].mean()),
            "mean_step_progress": float(step_progress[sl].mean()),
            "positive_progress_frac": float((step_progress[sl] > 0.0).mean()),
            "mean_cross": float(cross[sl].mean()),
            "distance_reduction_mean": float((dist_before[sl] - dist_after[sl]).mean()),
            "net_progress": float(np.dot(net_delta, direction)),
            "net_cross": float(abs(np.dot(net_delta, side))),
        })
    keys = rows[0].keys()
    summary = {k: float(np.mean([r[k] for r in rows])) for k in keys}
    summary.update({
        f"{k}_min": float(np.min([r[k] for r in rows]))
        for k in ("survived_frac", "net_progress", "positive_progress_frac")
    })
    summary["phase_samples"] = int(len(phases))
    summary["samples"] = int(len(phases) * args.transition_steps)
    summary["prev_command"] = prev_name
    summary["next_command"] = next_name
    return summary


def args_namespace(args, router: str, candidate: dict):
    class Obj:
        pass

    ns = Obj()
    ns.router = router
    ns.gate_power = float(candidate.get("gate_power", 1.0) if args.gate_power is None else args.gate_power)
    ns.waypoints = args.waypoints
    ns.radius = args.radius
    ns.steps_per_waypoint = args.steps_per_waypoint
    ns.frame_skip = args.frame_skip
    ns.kp = args.kp
    ns.kd = args.kd
    ns.scale = args.scale
    ns.stand_flex = args.stand_flex
    ns.stand_knee = args.stand_knee
    ns.spawn_height = args.spawn_height
    ns.fast_design = args.fast_design
    ns.max_saturation = args.max_saturation
    ns.vmax = args.vmax
    ns.gain = args.gain
    ns.axis_deadband = args.axis_deadband
    ns.command_mode = args.command_mode
    ns.cross_w = args.cross_w
    ns.backtrack_w = args.backtrack_w
    ns.progress_w = args.progress_w
    ns.final_dist_w = args.final_dist_w
    ns.waypoint_bonus_w = args.waypoint_bonus_w
    ns.focus_active = args.focus_active
    ns.focus_progress_w = args.focus_progress_w
    ns.focus_dist_w = args.focus_dist_w
    ns.focus_cross_w = args.focus_cross_w
    ns.steps_per_waypoint = args.steps_per_waypoint or candidate.get("steps_per_waypoint", 260)
    return ns


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--route-candidate", required=True)
    ap.add_argument("--tag", default="continuous_prior_v1")
    ap.add_argument("--routers", default="blend,winner,softmax,param_rbf")
    ap.add_argument("--phase-samples", type=int, default=2)
    ap.add_argument("--transition-speed", type=float, default=0.35)
    ap.add_argument("--transition-warmup-steps", type=int, default=80)
    ap.add_argument("--transition-steps", type=int, default=120)
    ap.add_argument("--rbf-sigma", type=float, default=0.85)
    ap.add_argument("--prev-command-blend", type=float, default=0.25)
    ap.add_argument("--gate-power", type=float, default=None)
    ap.add_argument("--steps-per-waypoint", type=int, default=None)
    ap.add_argument("--waypoints", default="0.35,0;0.35,0.35;0,0.35;0,0")
    ap.add_argument("--radius", type=float, default=0.07)
    ap.add_argument("--frame-skip", type=int, default=5)
    ap.add_argument("--kp", type=float, default=30.0)
    ap.add_argument("--kd", type=float, default=1.0)
    ap.add_argument("--scale", type=float, default=1.0)
    ap.add_argument("--stand-flex", type=float, default=0.15)
    ap.add_argument("--stand-knee", type=float, default=-1.25)
    ap.add_argument("--spawn-height", type=float, default=None)
    ap.add_argument("--fast-design", default=(0.5, 0.08, 1.0 / 3.0))
    ap.add_argument("--max-saturation", type=float, default=0.75)
    ap.add_argument("--vmax", type=float, default=0.35)
    ap.add_argument("--gain", type=float, default=2.0)
    ap.add_argument("--axis-deadband", type=float, default=0.02)
    ap.add_argument("--command-mode", choices=["segment-axis", "axis-priority", "full"], default="full")
    ap.add_argument("--cross-w", type=float, default=16.0)
    ap.add_argument("--backtrack-w", type=float, default=10.0)
    ap.add_argument("--progress-w", type=float, default=35.0)
    ap.add_argument("--final-dist-w", type=float, default=22.0)
    ap.add_argument("--waypoint-bonus-w", type=float, default=0.0)
    ap.add_argument("--focus-active", type=int, default=2)
    ap.add_argument("--focus-progress-w", type=float, default=0.0)
    ap.add_argument("--focus-dist-w", type=float, default=0.0)
    ap.add_argument("--focus-cross-w", type=float, default=0.0)
    args = ap.parse_args()

    route_path = Path(args.route_candidate)
    candidate = load_candidate(route_path)
    route = load_route(str(route_path))
    z_flat = jnp.asarray(route.reshape(-1), dtype=jnp.float32)
    phases = np.linspace(0.0, 2.0 * math.pi, max(1, args.phase_samples), endpoint=False)
    routers = [r.strip().lower() for r in args.routers.replace(";", ",").split(",") if r.strip()]

    report = {
        "tag": args.tag,
        "route_candidate": str(route_path),
        "routers": routers,
        "phase_samples": int(len(phases)),
        "rbf_sigma": float(args.rbf_sigma),
        "prev_command_blend": float(args.prev_command_blend),
        "gate_power": float(candidate.get("gate_power", 1.0) if args.gate_power is None else args.gate_power),
        "results": {},
    }
    for router in routers:
        gate_power = float(candidate.get("gate_power", 1.0) if args.gate_power is None else args.gate_power)
        candidate_for_router = {**candidate, "router": router, "gate_power": gate_power}
        cpg_teacher_mod.CPG_RBF_SIGMA = float(args.rbf_sigma)
        cpg_teacher_mod.CPG_PREV_CMD_BLEND = float(args.prev_command_blend)
        evaler = build_evaler(candidate_for_router, args_namespace(args, router, candidate))
        stats = route_stats_dict(evaler.eval_one_stats(z_flat))
        transitions = {
            name: transition_summary(evaler, z_flat, prev_name, next_name, phases, args)
            for name, (prev_name, next_name) in COMMAND_TRANSITIONS.items()
        }
        report["results"][router] = {
            "route": stats,
            "transitions": transitions,
        }
        print(
            f"[continuous-prior] {router}: reached={stats['reached']}/{evaler.nwp} "
            f"final={stats['final_dist']:.3f} focus={stats['focus_dist']:.3f} "
            f"surv={stats['survived_frac']:.2f}",
            flush=True,
        )

    OUT.mkdir(parents=True, exist_ok=True)
    out_path = OUT / f"{args.tag}_continuous_prior_compare.json"
    out_path.write_text(json.dumps(report, indent=2))
    print(json.dumps(report, indent=2), flush=True)
    print(f"saved {out_path}", flush=True)


if __name__ == "__main__":
    main()
