# SPDX-License-Identifier: MIT
"""Residual locomotion curriculum over the shared CPG teacher.

This is a serial driver around `train_commanded.py`:

    action = CPG_teacher_action + residual_scale * policy_action

Each residual scale is trained, evaluated, and validated before the next scale is
allowed to start.  The "best" checkpoint is only promoted when held-out
checkpoint navigation improves.
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
OUT = Path(os.environ.get("CODESIGN_OUT", "/root/proj/out"))


def parse_scales(text: str) -> list[float]:
    vals = [float(x.strip()) for x in text.replace(";", ",").split(",") if x.strip()]
    if not vals:
        raise ValueError("at least one residual scale is required")
    return vals


def run(cmd: list[str], env: dict[str, str]):
    print("+ " + " ".join(cmd), flush=True)
    subprocess.run(cmd, cwd=str(HERE), env=env, check=True)


def python_exe() -> str:
    return sys.executable or "python"


def resolve_path(path: Path) -> Path | None:
    """Resolve user-provided output paths from either repo root or OUT.

    The driver runs from sim/robot, while most reports are passed as
    out/foo.json from the repository root.  A missing baseline must not silently
    turn promotion into "compare against nothing".
    """
    candidates = [path]
    if not path.is_absolute():
        candidates.extend([ROOT / path, OUT / path])
        if path.parts and path.parts[0] == "out":
            candidates.append(OUT.joinpath(*path.parts[1:]))
    for candidate in candidates:
        if candidate.exists():
            return candidate
    return None


def read_json(path: Path) -> dict | None:
    resolved = resolve_path(path)
    return json.loads(resolved.read_text()) if resolved else None


def load_route_candidate_env(path_text: str) -> tuple[dict[str, str], str]:
    if not path_text:
        return {}, ""
    resolved = resolve_path(Path(path_text))
    if resolved is None:
        raise FileNotFoundError(f"could not resolve --route-candidate {path_text!r}")
    data = json.loads(resolved.read_text())
    deploy_env = data.get("deploy_env") or {}
    if not deploy_env:
        raise ValueError(f"--route-candidate {path_text!r} has no deploy_env")
    return {str(k): str(v) for k, v in deploy_env.items()}, str(resolved)


def safe_scale(scale: float) -> str:
    return str(scale).replace(".", "p").replace("-", "m")


def eval_stage(tag: str, ckpt: Path, args, env: dict[str, str]) -> dict:
    py = python_exe()
    mode_summaries = {}
    for mode in ("forward", "backward", "left", "right", "square"):
        eval_tag = f"{tag}_{mode}"
        run([
            py, "eval_commanded.py",
            "--tag", eval_tag,
            "--ckpt", str(ckpt),
            "--mode", mode,
            "--hold", str(args.eval_hold),
            "--speed", str(args.eval_speed),
            "--seed", str(args.seed),
        ], env)
        mode_summaries[mode] = read_json(OUT / f"{eval_tag}_eval_summary.json")

    nav_tag = f"{tag}_nav"
    run([
        py, "eval_checkpoint_navigation.py",
        "--tag", nav_tag,
        "--ckpt", str(ckpt),
        "--waypoints", args.waypoints,
        "--radius", str(args.nav_radius),
        "--gain", str(args.nav_gain),
        "--steps-per-waypoint", str(args.nav_steps),
        "--seed", str(args.seed),
    ], env)
    nav = read_json(OUT / f"{nav_tag}_nav_summary.json")
    return {"modes": mode_summaries, "nav": nav}


def gate_eval(summary: dict, min_survival_frac: float, require_nav_success: bool) -> tuple[bool, list[str]]:
    failures = []
    for mode, row in summary["modes"].items():
        if not row:
            failures.append(f"{mode}: missing eval summary")
            continue
        frac = row.get("samples", 0) / max(row.get("total_steps", 1), 1)
        if frac < min_survival_frac:
            failures.append(f"{mode}: survival {frac:.2f} < {min_survival_frac}")
    nav = summary.get("nav") or {}
    if nav.get("fall_reason"):
        failures.append(f"nav fall {nav.get('fall_reason')}")
    if require_nav_success and not nav.get("success"):
        failures.append(f"nav success required, reached {nav.get('reached', 0)}/{nav.get('total_waypoints', 1)}")
    return not failures, failures


def nav_score(nav: dict | None) -> tuple[int, float, int, float]:
    if not nav:
        return (0, -float("inf"), 0, -float("inf"))
    wp2 = nav.get("waypoint2_min_dist")
    if wp2 is None and "closest_waypoints" in nav and len(nav["closest_waypoints"]) > 2:
        wp2 = nav["closest_waypoints"][2].get("min_dist")
    wp2 = float(wp2) if wp2 is not None else float("inf")
    final_dist = float(nav.get("final_dist", 1e9))
    return (int(nav.get("reached", 0)), -wp2, int(nav.get("samples", 0)), -final_dist)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tag", default="residual_locomotion")
    ap.add_argument("--resume", required=True)
    ap.add_argument("--scales", default="0.1,0.25,0.5,1.0")
    ap.add_argument("--steps", type=int, default=1_800_000)
    ap.add_argument("--envs", type=int, default=2048)
    ap.add_argument("--batch", type=int, default=512)
    ap.add_argument("--minibatches", type=int, default=8)
    ap.add_argument("--unroll", type=int, default=10)
    ap.add_argument("--updates", type=int, default=2)
    ap.add_argument("--evals", type=int, default=6)
    ap.add_argument("--episode-length", type=int, default=500)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--entropy", type=float, default=0.006)
    ap.add_argument("--vmax", type=float, default=0.35)
    ap.add_argument("--track-sigma", type=float, default=0.05)
    ap.add_argument("--transition-scale-mult", type=float, default=1.0,
                    help="multiply residual scale during short command-transition windows")
    ap.add_argument("--transition-hold-steps", type=int, default=0,
                    help="number of env steps to use transition residual scale after command changes")
    ap.add_argument("--transition-delta-threshold", type=float, default=0.08)
    ap.add_argument("--wp2-residual-scale", type=float, default=None,
                    help="optional minimum residual scale while active route waypoint 2 is being corrected")
    ap.add_argument("--obs-prior-strength", action="store_true",
                    help="append current prior strength to the observation before the command tail")
    ap.add_argument("--obs-route-context", action="store_true",
                    help="append route waypoint/transition context to the observation before the command tail")
    ap.add_argument("--route-start-wp", type=int, default=0,
                    help="when --train-mode route, start episodes with this active waypoint")
    ap.add_argument("--route-start-xy", default="",
                    help="optional x,y reset position for focused route residual training")
    ap.add_argument("--route-starts", default="",
                    help="semicolon-separated waypoint,x,y reset mix for route residual training")
    ap.add_argument("--wp2-y-overshoot-w", type=float, default=0.0,
                    help="route-mode penalty for y above the waypoint-2 target during the backward segment")
    ap.add_argument("--wp2-y-vel-w", type=float, default=0.0,
                    help="route-mode penalty for positive y velocity during active waypoint 2")
    ap.add_argument("--wp2-x-progress-w", type=float, default=0.0,
                    help="route-mode reward for negative-x progress during active waypoint 2")
    ap.add_argument("--wp2-x-remaining-w", type=float, default=0.0,
                    help="route-mode penalty for remaining too far right of the waypoint-2 target")
    ap.add_argument("--train-mode", default="cardinal")
    ap.add_argument("--reward-mode", default="command")
    ap.add_argument("--hold-steps", type=int, default=80)
    ap.add_argument("--eval-hold", type=int, default=240)
    ap.add_argument("--eval-speed", type=float, default=0.35)
    ap.add_argument("--waypoints", default="0.35,0;0.35,0.35;0,0.35;0,0")
    ap.add_argument("--nav-radius", type=float, default=0.07)
    ap.add_argument("--nav-gain", type=float, default=2.0)
    ap.add_argument("--nav-steps", type=int, default=220)
    ap.add_argument("--min-survival-frac", type=float, default=0.95)
    ap.add_argument("--baseline-nav", default=None,
                    help="optional previous nav JSON; residual stages must beat this before promotion")
    ap.add_argument("--route-candidate", default="",
                    help="optional route gait JSON whose deploy_env CPG parameters seed training/eval")
    ap.add_argument("--require-nav-success", action="store_true",
                    help="stop unless a stage reaches all nav waypoints; otherwise promote on improvement")
    args = ap.parse_args()

    OUT.mkdir(parents=True, exist_ok=True)
    py = python_exe()
    resume = Path(args.resume)
    best_ckpt = resume
    baseline_nav = read_json(Path(args.baseline_nav)) if args.baseline_nav else None
    if args.baseline_nav and baseline_nav is None:
        raise FileNotFoundError(f"could not resolve --baseline-nav {args.baseline_nav!r}")
    route_env, route_candidate_path = load_route_candidate_env(args.route_candidate)
    best_score = nav_score(baseline_nav) if baseline_nav else (-1, -float("inf"), 0, -float("inf"))
    report = {
        "tag": args.tag,
        "baseline_nav": str(args.baseline_nav or ""),
        "baseline_score": best_score,
        "route_candidate": route_candidate_path,
        "route_candidate_env_keys": sorted(route_env),
        "stages": [],
        "best": {"ckpt": str(best_ckpt), "score": best_score},
    }

    for i, scale in enumerate(parse_scales(args.scales)):
        stage_tag = f"{args.tag}_r{safe_scale(scale)}"
        env = os.environ.copy()
        env.update(route_env)
        env.update({
            "CODESIGN_OUT": str(OUT),
            "CMD_CONTROL_MODE": "cpg_pd",
            "CMD_CPG_RESIDUAL_SCALE": str(scale),
            "CMD_CPG_RESIDUAL_SCALE_TRANSITION": str(scale * args.transition_scale_mult),
            "CMD_CPG_RESIDUAL_SCALE_WP2": "" if args.wp2_residual_scale is None else str(args.wp2_residual_scale),
            "CMD_CPG_TRANSITION_HOLD_STEPS": str(args.transition_hold_steps),
            "CMD_CPG_TRANSITION_DELTA_THRESH": str(args.transition_delta_threshold),
            "CMD_OBS_PRIOR_STRENGTH": "1" if args.obs_prior_strength else "0",
            "CMD_OBS_ROUTE_CONTEXT": "1" if args.obs_route_context else "0",
            "CMD_VMAX": str(args.vmax),
            "CMD_TRACK_SIGMA": str(args.track_sigma),
            "CMD_TRAIN_MODE": args.train_mode,
            "CMD_REWARD_MODE": args.reward_mode,
            "CMD_HOLD_STEPS": str(args.hold_steps),
            "CMD_ROUTE_WAYPOINTS": args.waypoints,
            "CMD_ROUTE_RADIUS": str(args.nav_radius),
            "CMD_ROUTE_GAIN": str(args.nav_gain),
            "CMD_ROUTE_START_WP": str(args.route_start_wp),
            "CMD_ROUTE_START_XY": args.route_start_xy,
            "CMD_ROUTE_STARTS": args.route_starts,
            "CMD_ROUTE_WP2_Y_OVERSHOOT_W": str(args.wp2_y_overshoot_w),
            "CMD_ROUTE_WP2_Y_VEL_W": str(args.wp2_y_vel_w),
            "CMD_ROUTE_WP2_X_PROGRESS_W": str(args.wp2_x_progress_w),
            "CMD_ROUTE_WP2_X_REMAINING_W": str(args.wp2_x_remaining_w),
        })
        run([
            py, "train_commanded.py",
            "--tag", stage_tag,
            "--resume", str(best_ckpt),
            "--steps", str(args.steps),
            "--envs", str(args.envs),
            "--batch", str(args.batch),
            "--minibatches", str(args.minibatches),
            "--unroll", str(args.unroll),
            "--updates", str(args.updates),
            "--evals", str(args.evals),
            "--episode-length", str(args.episode_length),
            "--seed", str(args.seed + i),
            "--lr", str(args.lr),
            "--entropy", str(args.entropy),
        ], env)
        ckpt = OUT / f"{stage_tag}.pkl"
        summary = eval_stage(stage_tag, ckpt, args, env)
        ok, failures = gate_eval(summary, args.min_survival_frac, args.require_nav_success)
        score = nav_score(summary.get("nav"))
        improved = score > best_score
        promoted = bool(ok and improved)
        if promoted:
            best_score = score
            best_ckpt = ckpt
            shutil.copyfile(ckpt, OUT / f"{args.tag}_best.pkl")
        stage = {
            "tag": stage_tag,
            "scale": scale,
            "ckpt": str(ckpt),
            "ok": ok,
            "failures": failures,
            "nav_score": score,
            "promoted": promoted,
            "summary": summary,
        }
        report["stages"].append(stage)
        report["best"] = {"ckpt": str(best_ckpt), "score": best_score}
        (OUT / f"{args.tag}_residual_report.json").write_text(json.dumps(report, indent=2))
        print(json.dumps(stage, indent=2), flush=True)
        if not ok:
            print(f"[residual] stopping before next scale; gate failed: {failures}", flush=True)
            break

    print(f"saved {OUT / f'{args.tag}_residual_report.json'}", flush=True)


if __name__ == "__main__":
    main()
