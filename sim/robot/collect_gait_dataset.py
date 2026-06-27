# SPDX-License-Identifier: MIT
"""Collect a CPG-teacher gait library for behavior cloning.

For each command bucket this runs MJX CEM over `CPGParams`, keeps only
physically valid rollouts, and saves supervised pairs:

    obs + command -> clipped CPG/PD motor action
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
from search_cpg_gait_mjx import MjxGaitEval, cem, parse_design, resolve_stance  # noqa: E402
from cpg_teacher import params_to_dict, decode_params  # noqa: E402

OUT = Path(os.environ.get("CODESIGN_OUT", "/root/proj/out"))


DIRS = {
    "forward": np.asarray([1.0, 0.0], dtype=float),
    "backward": np.asarray([-1.0, 0.0], dtype=float),
    "left": np.asarray([0.0, 1.0], dtype=float),
    "right": np.asarray([0.0, -1.0], dtype=float),
    "diag_fl": np.asarray([1.0, 1.0], dtype=float) / np.sqrt(2.0),
    "diag_fr": np.asarray([1.0, -1.0], dtype=float) / np.sqrt(2.0),
    "diag_bl": np.asarray([-1.0, 1.0], dtype=float) / np.sqrt(2.0),
    "diag_br": np.asarray([-1.0, -1.0], dtype=float) / np.sqrt(2.0),
    "hold": np.asarray([0.0, 0.0], dtype=float),
}


def parse_csv_floats(text: str) -> list[float]:
    return [float(x.strip()) for x in text.replace(";", ",").split(",") if x.strip()]


def parse_names(text: str) -> list[str]:
    vals = [x.strip() for x in text.replace(";", ",").split(",") if x.strip()]
    bad = [x for x in vals if x not in DIRS]
    if bad:
        raise ValueError(f"unknown command bucket(s): {bad}; valid={sorted(DIRS)}")
    return vals


def build_buckets(names: list[str], speeds: list[float]) -> list[tuple[str, np.ndarray]]:
    buckets = []
    for name in names:
        if name == "hold":
            buckets.append(("hold", DIRS[name].copy()))
            continue
        for speed in speeds:
            buckets.append((f"{name}_{speed:.2f}", DIRS[name] * float(speed)))
    return buckets


def pass_gate(command: np.ndarray, stats: np.ndarray, min_survival: float, min_progress: float,
              max_lateral: float, max_saturation: float) -> tuple[bool, dict]:
    direction = command / max(float(np.linalg.norm(command)), 1e-9)
    side = np.asarray([-direction[1], direction[0]], dtype=float)
    delta = np.asarray([float(stats[2]), float(stats[3])], dtype=float)
    progress = float(np.dot(delta, direction))
    lateral = float(abs(np.dot(delta, side)))
    survived = float(stats[4])
    saturation = float(stats[8])
    hold = float(np.linalg.norm(command)) < 1e-9
    ok = survived >= min_survival and saturation <= max_saturation
    if hold:
        ok = ok and float(np.linalg.norm(delta)) <= max_lateral
    else:
        ok = ok and progress >= min_progress and lateral <= max_lateral
    return ok, {
        "progress": progress,
        "lateral": lateral,
        "survived_frac": survived,
        "saturation": saturation,
        "hold": hold,
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out-dir", default=str(OUT / "gait_dataset"))
    ap.add_argument("--buckets", default="forward,backward,left,right,diag_fl,diag_fr,diag_bl,diag_br,hold")
    ap.add_argument("--speeds", default="0.12,0.24,0.35")
    ap.add_argument("--steps", type=int, default=240)
    ap.add_argument("--frame-skip", type=int, default=5)
    ap.add_argument("--pop", type=int, default=128)
    ap.add_argument("--gens", type=int, default=24)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--elite-frac", type=float, default=0.2)
    ap.add_argument("--kp", type=float, default=30.0)
    ap.add_argument("--kd", type=float, default=1.0)
    ap.add_argument("--scale", type=float, default=1.0)
    ap.add_argument("--stand-flex", type=float, default=None)
    ap.add_argument("--stand-knee", type=float, default=None)
    ap.add_argument("--spawn-height", type=float, default=None)
    ap.add_argument("--fast-design", default="0.5,0.08,0.3333333333333333")
    ap.add_argument("--track-sigma", type=float, default=0.05)
    ap.add_argument("--progress-w", type=float, default=8.0)
    ap.add_argument("--lateral-w", type=float, default=6.0)
    ap.add_argument("--track-w", type=float, default=3.0)
    ap.add_argument("--align-w", type=float, default=1.0)
    ap.add_argument("--vel-progress-w", type=float, default=8.0)
    ap.add_argument("--vel-lateral-w", type=float, default=4.0)
    ap.add_argument("--max-saturation", type=float, default=0.75)
    ap.add_argument("--min-survival", type=float, default=0.999)
    ap.add_argument("--min-progress", type=float, default=0.02)
    ap.add_argument("--max-lateral", type=float, default=0.25)
    args = ap.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    fast_design = parse_design(args.fast_design)
    stand_flex, stand_knee = resolve_stance(args.stand_flex, args.stand_knee)
    buckets = build_buckets(parse_names(args.buckets), parse_csv_floats(args.speeds))
    manifest = {
        "out_dir": str(out_dir),
        "steps": args.steps,
        "seed": args.seed,
        "buckets": [],
        "kept": 0,
        "attempted": len(buckets),
    }

    for i, (name, command) in enumerate(buckets):
        print(f"[gait-dataset] bucket {i + 1}/{len(buckets)} {name} command={command.tolist()}", flush=True)
        evaler = MjxGaitEval(
            command, args.steps, args.frame_skip, args.kp, args.kd, args.scale,
            stand_flex, stand_knee, args.spawn_height, fast_design,
            args.track_sigma, args.progress_w, args.lateral_w, args.track_w, args.align_w,
            args.vel_progress_w, args.vel_lateral_w, args.max_saturation,
        )
        best_z, best_stats = cem(evaler, args.pop, args.gens, args.seed + i, args.elite_frac)
        ok, gate = pass_gate(command, best_stats, args.min_survival, args.min_progress,
                             args.max_lateral, args.max_saturation)
        params = decode_params(best_z, xp=np)
        entry = {
            "name": name,
            "command": command.tolist(),
            "ok": bool(ok),
            "gate": gate,
            "score": float(best_stats[0]),
            "raw": best_z.tolist(),
            "params": params_to_dict(params),
        }
        if ok:
            obs, action, xy, z, up, sat, alive = evaler.eval_one_dataset(jnp.asarray(best_z, dtype=jnp.float32))
            path = out_dir / f"{i:03d}_{name}.npz"
            np.savez(
                path,
                obs=np.asarray(obs),
                action=np.asarray(action),
                command=np.asarray(command, dtype=np.float32),
                xy=np.asarray(xy),
                z=np.asarray(z),
                up=np.asarray(up),
                saturation=np.asarray(sat),
                alive=np.asarray(alive),
                raw=np.asarray(best_z, dtype=np.float32),
                params=json.dumps(params_to_dict(params)),
            )
            entry["path"] = str(path)
            manifest["kept"] += 1
            print(f"[gait-dataset] kept {name}: {gate}", flush=True)
        else:
            print(f"[gait-dataset] rejected {name}: {gate}", flush=True)
        manifest["buckets"].append(entry)
        (out_dir / "manifest.json").write_text(json.dumps(manifest, indent=2))

    print(json.dumps(manifest, indent=2), flush=True)
    print(f"saved manifest {out_dir / 'manifest.json'}", flush=True)


if __name__ == "__main__":
    main()
