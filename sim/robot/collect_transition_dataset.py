# SPDX-License-Identifier: MIT
"""Collect transition-aware CPG teacher samples from a saved route candidate.

This fills the gap between isolated cardinal gait search and BC/residual
training.  It replays a deployable route CPG, phase-shifts the oscillator, then
saves BC-compatible `(obs, action)` pairs for the route-transition states where
cardinal priors fail to compose cleanly.
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
from search_cpg_route_mjx import MjxRouteEval, load_route, parse_waypoints  # noqa: E402

OUT = Path(os.environ.get("CODESIGN_OUT", "/root/proj/out"))

BUCKETS = (
    "partial_route",
    "forward_to_left",
    "left_to_backward",
    "backward_to_right",
    "right_to_forward",
    "diagonal_correction",
    "overshoot_recovery",
    "cross_axis_drift",
    "waypoint2_focus",
)

DIR_CMDS = {
    "forward": np.asarray([1.0, 0.0], dtype=np.float32),
    "backward": np.asarray([-1.0, 0.0], dtype=np.float32),
    "left": np.asarray([0.0, 1.0], dtype=np.float32),
    "right": np.asarray([0.0, -1.0], dtype=np.float32),
    "diag_fl": np.asarray([1.0, 1.0], dtype=np.float32) / np.sqrt(2.0),
    "diag_fr": np.asarray([1.0, -1.0], dtype=np.float32) / np.sqrt(2.0),
    "diag_bl": np.asarray([-1.0, 1.0], dtype=np.float32) / np.sqrt(2.0),
    "diag_br": np.asarray([-1.0, -1.0], dtype=np.float32) / np.sqrt(2.0),
    "hold": np.asarray([0.0, 0.0], dtype=np.float32),
}

COMMAND_TRANSITIONS = {
    "forward_to_left": ("forward", "left"),
    "left_to_backward": ("left", "backward"),
    "backward_to_right": ("backward", "right"),
    "right_to_forward": ("right", "forward"),
}

COMMAND_FAMILIES = (
    "forward",
    "backward",
    "left",
    "right",
    "diag_fl",
    "diag_fr",
    "diag_bl",
    "diag_br",
    "hold",
)


def as_fast_design(value):
    if value is None:
        return None
    if isinstance(value, str):
        if value.strip().lower() in ("", "none", "raw"):
            return None
        return tuple(float(x.strip()) for x in value.replace(";", ",").split(",") if x.strip())
    return tuple(float(x) for x in value)


def load_candidate(path: Path) -> dict:
    data = json.loads(path.read_text())
    if "raw_route" not in data and not all(k in data for k in ("raw_forward", "raw_backward", "raw_left", "raw_right")):
        raise ValueError(f"{path} is not a route candidate JSON")
    return data


def build_evaler(candidate: dict, args) -> MjxRouteEval:
    cpg_teacher_mod.CPG_ROUTER = str(candidate.get("router", args.router))
    cpg_teacher_mod.CPG_GATE_POWER = float(candidate.get("gate_power", args.gate_power))
    waypoints = np.asarray(candidate.get("waypoints", parse_waypoints(args.waypoints)), dtype=np.float32)
    return MjxRouteEval(
        waypoints=waypoints,
        radius=float(candidate.get("radius", args.radius)),
        steps_per_waypoint=int(args.steps_per_waypoint or candidate.get("steps_per_waypoint", 260)),
        frame_skip=int(args.frame_skip),
        kp=float(candidate.get("kp", args.kp)),
        kd=float(candidate.get("kd", args.kd)),
        scale=float(candidate.get("scale", args.scale)),
        stand_flex=float(candidate.get("stand_flex", args.stand_flex)),
        stand_knee=float(candidate.get("stand_knee", args.stand_knee)),
        spawn_height=candidate.get("spawn_height", args.spawn_height),
        fast_design=as_fast_design(candidate.get("fast_design", args.fast_design)),
        max_saturation=float(candidate.get("max_saturation", args.max_saturation)),
        vmax=float(candidate.get("vmax", args.vmax)),
        gain=float(candidate.get("gain", args.gain)),
        axis_deadband=float(candidate.get("axis_deadband", args.axis_deadband)),
        command_mode=str(candidate.get("command_mode", args.command_mode)),
        cross_w=float(candidate.get("cross_w", args.cross_w)),
        backtrack_w=float(candidate.get("backtrack_w", args.backtrack_w)),
        progress_w=float(candidate.get("progress_w", args.progress_w)),
        final_dist_w=float(candidate.get("final_dist_w", args.final_dist_w)),
        waypoint_bonus_w=float(candidate.get("waypoint_bonus_w", args.waypoint_bonus_w)),
        focus_active=int(candidate.get("focus_active", args.focus_active)),
        focus_progress_w=float(candidate.get("focus_progress_w", args.focus_progress_w)),
        focus_dist_w=float(candidate.get("focus_dist_w", args.focus_dist_w)),
        focus_cross_w=float(candidate.get("focus_cross_w", args.focus_cross_w)),
    )


def concat_hist(rows: list[tuple]) -> dict[str, np.ndarray]:
    names = (
        "obs",
        "action",
        "pre_xy",
        "post_xy",
        "z",
        "up",
        "saturation",
        "alive",
        "reached",
        "active",
        "segment",
        "phase",
        "cmd",
        "target",
        "prev_waypoint",
        "dist_before",
        "dist_after",
        "segment_progress",
        "progress_frac",
        "cross",
        "backtrack",
        "action_scale",
    )
    cols = []
    for i in range(len(names)):
        cols.append(np.concatenate([np.asarray(row[i]) for row in rows], axis=0))
    out = dict(zip(names, cols))
    out["active"] = out["active"].astype(np.int32)
    out["segment"] = out["segment"].astype(np.int32)
    out["reached"] = out["reached"].astype(np.int32)
    out["phase_sample"] = np.concatenate([
        np.full((np.asarray(row[0]).shape[0],), i, dtype=np.int32)
        for i, row in enumerate(rows)
    ])
    return out


def add_labels(data: dict[str, np.ndarray], max_saturation: float) -> dict[str, np.ndarray]:
    labels = {
        "distance_reduction": data["dist_before"] - data["dist_after"],
        "y_drift": data["post_xy"][:, 1] - data["pre_xy"][:, 1],
        "x_delta": data["post_xy"][:, 0] - data["pre_xy"][:, 0],
        "fall": (data["alive"] < 0.5).astype(np.float32),
        "high_saturation": (data["saturation"] > max_saturation).astype(np.float32),
    }
    # The known failing transition is active waypoint 2, whose desired progress is
    # negative X.  Positive X delta is therefore an X-progress loss there.
    labels["x_loss"] = np.where(data["active"] == 2, np.maximum(labels["x_delta"], 0.0), 0.0)
    labels["positive_y_drift"] = (labels["y_drift"] > 0.0).astype(np.float32)
    return labels


def bucket_masks(data: dict[str, np.ndarray], labels: dict[str, np.ndarray], cross_threshold: float) -> dict[str, np.ndarray]:
    active = data["active"]
    cmd = data["cmd"]
    return {
        "partial_route": np.ones(active.shape, dtype=bool),
        "forward_to_left": active == 1,
        "left_to_backward": active == 2,
        "backward_to_right": active == 3,
        "right_to_forward": active == 0,
        "diagonal_correction": (np.abs(cmd[:, 0]) > 1e-3) & (np.abs(cmd[:, 1]) > 1e-3),
        "overshoot_recovery": (labels["distance_reduction"] < 0.0) | (data["backtrack"] > 0.02),
        "cross_axis_drift": data["cross"] > cross_threshold,
        "waypoint2_focus": active == 2,
    }


def save_bucket(out_dir: Path, name: str, mask: np.ndarray, data: dict[str, np.ndarray],
                labels: dict[str, np.ndarray], metadata: dict) -> dict:
    count = int(mask.sum())
    entry = {"name": name, "count": count, "ok": count > 0}
    if count <= 0:
        return entry
    path = out_dir / f"transition_{name}.npz"
    np.savez(
        path,
        obs=data["obs"][mask],
        action=data["action"][mask],
        command=data["cmd"][mask],
        target=data["target"][mask],
        pre_xy=data["pre_xy"][mask],
        post_xy=data["post_xy"][mask],
        z=data["z"][mask],
        up=data["up"][mask],
        saturation=data["saturation"][mask],
        alive=data["alive"][mask],
        active=data["active"][mask],
        segment=data["segment"][mask],
        phase=data["phase"][mask],
        reached=data["reached"][mask],
        progress_frac=data["progress_frac"][mask],
        cross=data["cross"][mask],
        backtrack=data["backtrack"][mask],
        phase_sample=data["phase_sample"][mask],
        distance_reduction=labels["distance_reduction"][mask],
        y_drift=labels["y_drift"][mask],
        x_loss=labels["x_loss"][mask],
        fall=labels["fall"][mask],
        high_saturation=labels["high_saturation"][mask],
        metadata=json.dumps(metadata),
    )
    entry.update({
        "path": str(path),
        "distance_reduction_mean": float(labels["distance_reduction"][mask].mean()),
        "distance_reduction_max": float(labels["distance_reduction"][mask].max()),
        "positive_progress_samples": int((labels["distance_reduction"][mask] > 0.0).sum()),
        "mean_y_drift": float(labels["y_drift"][mask].mean()),
        "mean_x_loss": float(labels["x_loss"][mask].mean()),
        "survived_frac": float(data["alive"][mask].mean()),
        "mean_saturation": float(data["saturation"][mask].mean()),
    })
    return entry


def command_sequence(prev_name: str, next_name: str, speed: float, warmup_steps: int,
                     transition_steps: int) -> np.ndarray:
    prev = DIR_CMDS[prev_name] * float(speed)
    nxt = DIR_CMDS[next_name] * float(speed)
    return np.vstack([
        np.repeat(prev[None, :], int(warmup_steps), axis=0),
        np.repeat(nxt[None, :], int(transition_steps), axis=0),
    ]).astype(np.float32)


def steady_command(name: str, speed: float, steps: int) -> np.ndarray:
    scale = 0.0 if name == "hold" else float(speed)
    cmd = DIR_CMDS[name] * scale
    return np.repeat(cmd[None, :], int(steps), axis=0).astype(np.float32)


def collect_command_transition(evaler: MjxRouteEval, z_flat, name: str, prev_name: str, next_name: str,
                               phases: np.ndarray, args) -> tuple[dict[str, np.ndarray], dict[str, np.ndarray]]:
    cmds = command_sequence(prev_name, next_name, args.transition_speed,
                            args.transition_warmup_steps, args.transition_steps)
    rows = []
    cmd_jnp = jnp.asarray(cmds, dtype=jnp.float32)
    for i, phase in enumerate(phases):
        print(
            f"[transition-dataset] command {name} phase {i + 1}/{len(phases)} "
            f"phase0={phase:.3f}",
            flush=True,
        )
        rows.append(evaler.eval_command_dataset(z_flat, cmd_jnp, jnp.asarray(phase, dtype=jnp.float32)))
    data = concat_hist(rows)
    labels = add_labels(data, float(args.max_saturation))
    return data, labels


def collect_steady_command(evaler: MjxRouteEval, z_flat, name: str, phases: np.ndarray,
                           args) -> tuple[dict[str, np.ndarray], dict[str, np.ndarray]]:
    cmds = steady_command(name, args.transition_speed, args.steady_steps)
    rows = []
    cmd_jnp = jnp.asarray(cmds, dtype=jnp.float32)
    for i, phase in enumerate(phases):
        print(
            f"[transition-dataset] command_family {name} phase {i + 1}/{len(phases)} "
            f"phase0={phase:.3f}",
            flush=True,
        )
        rows.append(evaler.eval_command_dataset(z_flat, cmd_jnp, jnp.asarray(phase, dtype=jnp.float32)))
    data = concat_hist(rows)
    labels = add_labels(data, float(args.max_saturation))
    return data, labels


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--route-candidate", required=True,
                    help="saved *_route_gait.json candidate with deployable raw vectors")
    ap.add_argument("--out-dir", default=str(OUT / "gait_dataset"))
    ap.add_argument("--tag", default="transition")
    ap.add_argument("--phase-samples", type=int, default=8)
    ap.add_argument("--command-phase-samples", type=int, default=8)
    ap.add_argument("--steady-phase-samples", type=int, default=8)
    ap.add_argument("--no-command-transitions", action="store_true")
    ap.add_argument("--no-steady-commands", action="store_true")
    ap.add_argument("--transition-speed", type=float, default=0.35)
    ap.add_argument("--transition-warmup-steps", type=int, default=80)
    ap.add_argument("--transition-steps", type=int, default=120)
    ap.add_argument("--steady-steps", type=int, default=200)
    ap.add_argument("--cross-threshold", type=float, default=0.08)
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
    ap.add_argument(
        "--router",
        choices=["blend", "softmax", "winner", "param_softmax", "param_rbf", "continuous"],
        default="softmax",
    )
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
    evaler = build_evaler(candidate, args)
    phases = np.linspace(0.0, 2.0 * math.pi, max(1, args.phase_samples), endpoint=False)
    rows = []
    z_flat = jnp.asarray(route.reshape(-1), dtype=jnp.float32)
    for i, phase in enumerate(phases):
        print(f"[transition-dataset] phase {i + 1}/{len(phases)} phase0={phase:.3f}", flush=True)
        rows.append(evaler.eval_one_dataset(z_flat, jnp.asarray(phase, dtype=jnp.float32)))

    data = concat_hist(rows)
    labels = add_labels(data, float(candidate.get("max_saturation", args.max_saturation)))
    masks = bucket_masks(data, labels, args.cross_threshold)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    metadata = {
        "tag": args.tag,
        "route_candidate": str(route_path),
        "phase_samples": int(len(phases)),
        "total_samples": int(data["obs"].shape[0]),
        "waypoints": np.asarray(evaler.waypoints_np).tolist(),
        "radius": float(evaler.radius),
        "command_mode": str(candidate.get("command_mode", args.command_mode)),
        "router": str(candidate.get("router", args.router)),
        "gate_power": float(candidate.get("gate_power", args.gate_power)),
    }
    manifest = {
        **metadata,
        "buckets": [],
        "kept": 0,
        "attempted": len(BUCKETS),
    }
    for name in BUCKETS:
        entry = save_bucket(out_dir, name, masks[name], data, labels, metadata)
        if entry["ok"]:
            manifest["kept"] += 1
        manifest["buckets"].append(entry)
        print(f"[transition-dataset] {name}: count={entry['count']}", flush=True)

    command_entries = []
    if not args.no_command_transitions:
        cmd_phases = np.linspace(
            0.0,
            2.0 * math.pi,
            max(1, args.command_phase_samples),
            endpoint=False,
        )
        for name, (prev_name, next_name) in COMMAND_TRANSITIONS.items():
            cmd_data, cmd_labels = collect_command_transition(
                evaler, z_flat, name, prev_name, next_name, cmd_phases, args
            )
            entry = save_bucket(
                out_dir,
                f"command_{name}",
                np.ones(cmd_data["obs"].shape[0], dtype=bool),
                cmd_data,
                cmd_labels,
                {**metadata, "source": "command_sequence", "transition_family": name},
            )
            entry.update({
                "transition_family": name,
                "source": "command_sequence",
                "prev_command": prev_name,
                "next_command": next_name,
            })
            command_entries.append(entry)
            print(f"[transition-dataset] command_{name}: count={entry['count']}", flush=True)

    command_family_entries = []
    if not args.no_steady_commands:
        steady_phases = np.linspace(
            0.0,
            2.0 * math.pi,
            max(1, args.steady_phase_samples),
            endpoint=False,
        )
        for name in COMMAND_FAMILIES:
            cmd_data, cmd_labels = collect_steady_command(evaler, z_flat, name, steady_phases, args)
            entry = save_bucket(
                out_dir,
                f"command_family_{name}",
                np.ones(cmd_data["obs"].shape[0], dtype=bool),
                cmd_data,
                cmd_labels,
                {**metadata, "source": "steady_command", "command_family": name},
            )
            entry.update({
                "command_family": name,
                "source": "steady_command",
            })
            command_family_entries.append(entry)
            print(f"[transition-dataset] command_family_{name}: count={entry['count']}", flush=True)

    focus = next((b for b in manifest["buckets"] if b["name"] == "waypoint2_focus"), {})
    manifest["focused_transition_nonzero_progress"] = int(focus.get("positive_progress_samples", 0))
    command_ok = {
        e["transition_family"]: bool(e["ok"])
        for e in command_entries
    }
    manifest["covers_transition_families"] = {
        b["name"]: bool(b["ok"]) or command_ok.get(b["name"], False)
        for b in manifest["buckets"]
        if b["name"] != "partial_route"
    }
    manifest["command_transition_buckets"] = command_entries
    manifest["command_transition_samples"] = int(sum(e.get("count", 0) for e in command_entries))
    manifest["command_family_buckets"] = command_family_entries
    manifest["command_family_samples"] = int(sum(e.get("count", 0) for e in command_family_entries))
    manifest["covers_command_families"] = {
        e["command_family"]: bool(e["ok"]) for e in command_family_entries
    }
    path = out_dir / f"{args.tag}_transition_manifest.json"
    path.write_text(json.dumps(manifest, indent=2))
    print(json.dumps(manifest, indent=2), flush=True)
    print(f"saved {path}", flush=True)


if __name__ == "__main__":
    main()
