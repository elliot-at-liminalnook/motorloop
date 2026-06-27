# SPDX-License-Identifier: MIT
"""Search transition-only CPG controls for a frozen route scaffold.

The route gait vectors stay fixed.  CEM mutates only:

  phase_reset_amount, phase_speed_boost, stance_height_boost,
  turn_bias, cross_axis_correction

This keeps transition tuning separate from steady-state CPG gait parameters.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

import jax.numpy as jnp
import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import cpg_teacher as cpg_teacher_mod  # noqa: E402
from collect_transition_dataset import build_evaler, load_candidate  # noqa: E402
from search_cpg_route_mjx import load_route  # noqa: E402

OUT = Path(os.environ.get("CODESIGN_OUT", "/root/proj/out"))

CONTROL_NAMES = (
    "phase_reset_amount",
    "phase_speed_boost",
    "stance_height_boost",
    "turn_bias",
    "cross_axis_correction",
)

BOUNDS = np.asarray([
    [0.0, 1.0],     # phase reset amount
    [-0.5, 2.0],    # phase speed boost
    [-0.15, 0.25],  # stance height boost
    [-0.45, 0.45],  # turn bias
    [-0.45, 0.45],  # cross-axis correction
], dtype=np.float32)


def route_stats_dict(stats) -> dict[str, float | int]:
    names = (
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
    arr = np.asarray(stats, dtype=float)
    out = {name: float(arr[i]) for i, name in enumerate(names)}
    out["reached"] = int(round(out["reached"]))
    out["active_final"] = int(round(out["active_final"]))
    return out


def controls_dict(values: np.ndarray) -> dict[str, float]:
    return {name: float(values[i]) for i, name in enumerate(CONTROL_NAMES)}


def clip_controls(values: np.ndarray) -> np.ndarray:
    return np.clip(values, BOUNDS[:, 0], BOUNDS[:, 1]).astype(np.float32)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--route-candidate", required=True)
    ap.add_argument("--tag", default="transition_controls")
    ap.add_argument("--pop", type=int, default=32)
    ap.add_argument("--gens", type=int, default=8)
    ap.add_argument("--elite-frac", type=float, default=0.25)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--init-std", default="0.25,0.45,0.08,0.16,0.16")
    ap.add_argument("--min-reached", type=int, default=3)
    # build_evaler compatibility defaults
    ap.add_argument("--waypoints", default="0.35,0;0.35,0.35;0,0.35;0,0")
    ap.add_argument("--radius", type=float, default=0.07)
    ap.add_argument("--steps-per-waypoint", type=int, default=None)
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
    ap.add_argument("--command-mode", default="full")
    ap.add_argument("--router", default="softmax")
    ap.add_argument("--gate-power", type=float, default=2.0)
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
    cpg_teacher_mod.CPG_ROUTER = str(candidate.get("router", args.router))
    cpg_teacher_mod.CPG_GATE_POWER = float(candidate.get("gate_power", args.gate_power))
    evaler = build_evaler(candidate, args)
    z_flat = jnp.asarray(route.reshape(-1), dtype=jnp.float32)

    rng = np.random.default_rng(args.seed)
    std = np.asarray([float(x) for x in args.init_std.replace(";", ",").split(",") if x.strip()], dtype=np.float32)
    if std.shape != (len(CONTROL_NAMES),):
        raise ValueError(f"--init-std expected {len(CONTROL_NAMES)} values")
    zero_controls = np.zeros(len(CONTROL_NAMES), dtype=np.float32)
    mean = zero_controls.copy()
    elite_n = max(2, int(args.pop * args.elite_frac))
    # Warm up the JIT, then use a pinned zero-control baseline in every
    # generation so transition-control search is always compared to no controls.
    evaler.eval_one_stats_with_controls(z_flat, jnp.asarray(zero_controls))
    baseline_stats = route_stats_dict(evaler.eval_one_stats_with_controls(z_flat, jnp.asarray(zero_controls)))
    best_controls = zero_controls.copy()
    best_stats = baseline_stats

    history = []
    for gen in range(args.gens):
        cand = mean[None, :] + std[None, :] * rng.standard_normal((args.pop, len(CONTROL_NAMES)))
        cand = np.asarray([clip_controls(c) for c in cand], dtype=np.float32)
        cand[0] = zero_controls
        rows = [baseline_stats]
        for c in cand[1:]:
            stats = route_stats_dict(evaler.eval_one_stats_with_controls(z_flat, jnp.asarray(c)))
            rows.append(stats)
        scores = np.asarray([r["score"] for r in rows], dtype=float)
        scores = np.nan_to_num(scores, nan=-1e9, posinf=-1e9, neginf=-1e9)
        order = np.argsort(-scores)
        if scores[order[0]] > float(best_stats["score"]):
            best_controls = cand[order[0]].copy()
            best_stats = rows[int(order[0])]
        elites = cand[order[:elite_n]]
        mean = elites.mean(axis=0).astype(np.float32)
        std = np.maximum(elites.std(axis=0).astype(np.float32), 1e-3)
        row = rows[int(order[0])]
        history.append({
            "gen": gen,
            "best_controls": controls_dict(cand[order[0]]),
            "best_stats": row,
            "allbest_controls": controls_dict(best_controls),
            "allbest_stats": best_stats,
        })
        print(
            f"[transition-controls] gen {gen:02d} best reached={row['reached']}/{evaler.nwp} "
            f"score={row['score']:+.3f} focus={row['focus_dist']:.3f} "
            f"allbest={best_stats['reached']}/{evaler.nwp}:{best_stats['score']:+.3f}",
            flush=True,
        )

    passed = (
        int(best_stats["reached"]) >= int(args.min_reached)
        and float(best_stats["survived_frac"]) >= 0.999
        and float(best_stats["saturation"]) <= float(candidate.get("max_saturation", args.max_saturation))
    )
    final = {
        "tag": args.tag,
        "route_candidate": str(route_path),
        "control_names": CONTROL_NAMES,
        "bounds": BOUNDS.tolist(),
        "best_controls": controls_dict(best_controls),
        "best_stats": best_stats,
        "baseline_controls": controls_dict(np.zeros(len(CONTROL_NAMES), dtype=np.float32)),
        "baseline_stats": baseline_stats,
        "passes_gate": bool(passed),
        "min_reached": int(args.min_reached),
        "history": history,
        "deploy_env": {
            "CMD_CPG_PHASE_RESET_AMOUNT": str(float(best_controls[0])),
            "CMD_CPG_PHASE_SPEED_BOOST": str(float(best_controls[1])),
            "CMD_CPG_STANCE_HEIGHT_BOOST": str(float(best_controls[2])),
            "CMD_CPG_TURN_BIAS": str(float(best_controls[3])),
            "CMD_CPG_CROSS_AXIS_CORRECTION": str(float(best_controls[4])),
        },
    }
    OUT.mkdir(parents=True, exist_ok=True)
    out = OUT / f"{args.tag}_transition_controls.json"
    out.write_text(json.dumps(final, indent=2))
    print(json.dumps(final, indent=2), flush=True)
    print(f"saved {out}", flush=True)


if __name__ == "__main__":
    main()
