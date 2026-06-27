# SPDX-License-Identifier: MIT
"""Calibrate whether the CPG prior is useful as a learner scaffold.

This report separates "the prior solves navigation" from "the prior reduces the
learner's burden".  It can read a route-search JSON, a deployed navigation NPZ,
and a transition-dataset manifest.
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

import numpy as np

OUT = Path(os.environ.get("CODESIGN_OUT", "/root/proj/out"))


def read_json(path: str | None) -> dict | None:
    if not path:
        return None
    p = Path(path)
    return json.loads(p.read_text()) if p.exists() else None


def finite_float(value, default=None):
    try:
        out = float(value)
    except (TypeError, ValueError):
        return default
    return out if np.isfinite(out) else default


def route_prior_summary(route: dict | None) -> dict:
    if not route:
        return {"source": None, "available": False}
    reached = int(route.get("reached", route.get("search_stats", {}).get("reached", 0)))
    total = int(route.get("total_waypoints", 4))
    survived = finite_float(route.get("survived_frac"), 0.0)
    saturation = finite_float(route.get("saturation"), None)
    return {
        "available": True,
        "tag": route.get("tag"),
        "router": route.get("router"),
        "gate_power": route.get("gate_power"),
        "reached": reached,
        "total_waypoints": total,
        "route_progress": finite_float(route.get("route_progress"), None),
        "final_dist": finite_float(route.get("final_dist"), None),
        "survived_frac": survived,
        "min_z": finite_float(route.get("min_z"), None),
        "mean_up": finite_float(route.get("mean_up"), None),
        "saturation": saturation,
        "passes_gate": bool(route.get("passes_gate", False)),
        "keeps_body_alive": bool(survived >= 0.999),
        "moves_roughly_toward_route": bool(reached >= 1 or finite_float(route.get("route_progress"), 0.0) > 1.0),
    }


def nav_prior_summary(nav_npz: str | None, radius_default: float) -> dict:
    if not nav_npz or not Path(nav_npz).exists():
        return {"source": nav_npz, "available": False}
    data = np.load(nav_npz, allow_pickle=True)
    if not all(k in data.files for k in ("x", "y", "z", "up", "waypoints")):
        return {"source": nav_npz, "available": False, "error": "missing navigation arrays"}
    xy = np.column_stack([data["x"], data["y"]])
    z = np.asarray(data["z"])
    up = np.asarray(data["up"])
    waypoints = np.asarray(data["waypoints"], dtype=float)
    radius = finite_float(np.asarray(data["radius"]).item() if "radius" in data.files else radius_default, radius_default)
    active = np.asarray(data["waypoint"], dtype=int) if "waypoint" in data.files else np.zeros(len(xy), dtype=int)
    closest = []
    for i, wp in enumerate(waypoints):
        dist = np.linalg.norm(xy - wp, axis=1)
        j = int(dist.argmin())
        closest.append({
            "waypoint": int(i),
            "min_dist": float(dist[j]),
            "closest_xy": xy[j].tolist(),
            "step": j,
            "inside_radius": bool(dist[j] <= radius),
        })
    # `waypoint` records the sequential route state from the evaluator.  Do not
    # infer reached count from closest distance alone: the square route's final
    # waypoint is the origin, so the robot starts near it before that waypoint is
    # actually active.
    reached = int(active.max()) if len(active) else 0
    wp2 = closest[2] if len(closest) > 2 else None
    residual_vector = None
    if wp2 is not None:
        residual_vector = (waypoints[2] - np.asarray(wp2["closest_xy"], dtype=float)).tolist()
    return {
        "source": nav_npz,
        "available": True,
        "samples": int(len(xy)),
        "radius": float(radius),
        "reached_estimate": int(reached),
        "total_waypoints": int(len(waypoints)),
        "survival": {
            "z_min": float(z.min()) if len(z) else None,
            "up_min": float(up.min()) if len(up) else None,
            "keeps_body_alive": bool(len(z) > 0 and z.min() >= 0.08 and up.min() >= 0.3),
        },
        "closest_waypoints": closest,
        "waypoint2_burden": {
            "min_dist": wp2["min_dist"] if wp2 else None,
            "required_xy_correction": residual_vector,
            "inside_radius": wp2["inside_radius"] if wp2 else False,
        },
        "final_xy": xy[-1].tolist() if len(xy) else None,
    }


def transition_summary(manifest: dict | None) -> dict:
    if not manifest:
        return {"available": False}
    coverage = manifest.get("covers_transition_families", {})
    missing = [k for k, ok in coverage.items() if not ok]
    focus_progress = int(manifest.get("focused_transition_nonzero_progress", 0))
    route_samples = int(manifest.get("total_samples", 0))
    command_samples = int(manifest.get("command_transition_samples", 0))
    command_family_samples = int(manifest.get("command_family_samples", 0))
    command_family_coverage = manifest.get("covers_command_families", {})
    command_family_missing = [k for k, ok in command_family_coverage.items() if not ok]
    return {
        "available": True,
        "route_samples": route_samples,
        "command_transition_samples": command_samples,
        "command_family_samples": command_family_samples,
        "total_samples": route_samples + command_samples + command_family_samples,
        "kept_buckets": int(manifest.get("kept", 0)),
        "attempted_buckets": int(manifest.get("attempted", 0)),
        "coverage": coverage,
        "missing": missing,
        "command_family_coverage": command_family_coverage,
        "command_family_missing": command_family_missing,
        "focused_transition_nonzero_progress": focus_progress,
        "focused_transition_has_signal": bool(focus_progress > 0),
    }


def residual_comparison(nav: dict, residual_nav: dict | None) -> dict:
    if not residual_nav:
        return {
            "available": False,
            "status": "unmeasured",
            "note": "provide --residual-nav-json after BC/residual rollout to measure overconstraint directly",
        }
    residual_nav = residual_nav.get("nav", residual_nav)
    prior_reached = int(nav.get("reached_estimate", 0)) if nav.get("available") else 0
    residual_reached = int(residual_nav.get("reached", 0))
    prior_wp2 = ((nav.get("waypoint2_burden") or {}).get("min_dist")
                 if nav.get("available") else None)
    residual_wp2 = residual_nav.get("waypoint2_min_dist")
    residual_final = finite_float(residual_nav.get("final_dist"), None)
    return {
        "available": True,
        "residual_reached": residual_reached,
        "prior_reached": prior_reached,
        "improved_waypoint_count": bool(residual_reached > prior_reached),
        "prior_waypoint2_min_dist": prior_wp2,
        "residual_waypoint2_min_dist": finite_float(residual_wp2, None),
        "improved_waypoint2_distance": bool(
            prior_wp2 is not None
            and residual_wp2 is not None
            and finite_float(residual_wp2, float("inf")) < finite_float(prior_wp2, -float("inf"))
        ),
        "residual_final_dist": residual_final,
        "overconstraint_suspected": bool(residual_reached <= prior_reached and residual_final is not None),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tag", default="prior_calibration")
    ap.add_argument("--route-json", default=None)
    ap.add_argument("--nav-npz", default=None)
    ap.add_argument("--transition-manifest", default=None)
    ap.add_argument("--residual-nav-json", default=None)
    ap.add_argument("--radius", type=float, default=0.07)
    ap.add_argument("--out-json", default=None)
    args = ap.parse_args()

    route = read_json(args.route_json)
    transitions = read_json(args.transition_manifest)
    residual_nav = read_json(args.residual_nav_json)
    route_report = route_prior_summary(route)
    nav_report = nav_prior_summary(args.nav_npz, args.radius)
    transition_report = transition_summary(transitions)
    overconstraint = residual_comparison(nav_report, residual_nav)

    prior_survival = bool(
        route_report.get("keeps_body_alive", False)
        or (nav_report.get("survival") or {}).get("keeps_body_alive", False)
    )
    prior_progress = bool(
        route_report.get("moves_roughly_toward_route", False)
        or nav_report.get("reached_estimate", 0) >= 1
    )
    composability_gap = None
    if nav_report.get("available"):
        composability_gap = {
            "reached_estimate": nav_report["reached_estimate"],
            "total_waypoints": nav_report["total_waypoints"],
            "fails_before_full_route": bool(nav_report["reached_estimate"] < nav_report["total_waypoints"]),
            "waypoint2_min_dist": (nav_report.get("waypoint2_burden") or {}).get("min_dist"),
        }
    scaffold_ready = bool(
        prior_survival
        and prior_progress
        and route_report.get("reached", 0) >= 2
        and transition_report.get("focused_transition_has_signal", False)
        and not transition_report.get("missing", ["missing"])
        and not transition_report.get("command_family_missing", ["missing"])
    )
    report = {
        "tag": args.tag,
        "prior_survival": {
            "ok": prior_survival,
            "route": route_report,
            "nav": nav_report.get("survival"),
        },
        "prior_progress": {
            "ok": prior_progress,
            "route_progress": route_report.get("route_progress"),
            "reached": route_report.get("reached"),
            "nav_reached_estimate": nav_report.get("reached_estimate"),
        },
        "prior_residual_burden": nav_report.get("waypoint2_burden"),
        "prior_composability": composability_gap,
        "prior_overconstraint": overconstraint,
        "transition_dataset": transition_report,
        "bc_scaffold_ready": scaffold_ready,
    }
    out = Path(args.out_json) if args.out_json else OUT / f"{args.tag}_prior_calibration.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, indent=2))
    print(json.dumps(report, indent=2), flush=True)
    print(f"saved {out}", flush=True)


if __name__ == "__main__":
    main()
