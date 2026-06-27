# SPDX-License-Identifier: MIT
"""Validation summary/gate for command-conditioned locomotion artifacts."""

from __future__ import annotations

import argparse, json, os, sys
from pathlib import Path

OUT = Path(os.environ.get("CODESIGN_OUT", "/root/proj/out"))


def read_json(path: Path):
    return json.loads(path.read_text()) if path.exists() else None


def read_last_jsonl(path: Path):
    if not path.exists():
        return None
    rows = [json.loads(l) for l in path.read_text().splitlines() if l.strip()]
    return rows[-1] if rows else None


def nav_score(nav: dict):
    wp2 = nav.get("waypoint2_min_dist")
    if wp2 is None and "closest_waypoints" in nav and len(nav["closest_waypoints"]) > 2:
        wp2 = nav["closest_waypoints"][2].get("min_dist")
    wp2 = float(wp2) if wp2 is not None else float("inf")
    return (nav.get("reached", 0), -wp2, -nav.get("final_dist", 1e9))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tag", default="cmd")
    ap.add_argument("--out", default=str(OUT))
    ap.add_argument("--require-modes", default="",
                    help="comma-separated eval modes to require, e.g. forward,backward,left,right,square")
    ap.add_argument("--min-track", type=float, default=0.55)
    ap.add_argument("--min-align", type=float, default=0.40)
    ap.add_argument("--max-err", type=float, default=0.75)
    ap.add_argument("--min-survival-frac", type=float, default=0.90)
    ap.add_argument("--min-nav-frac", type=float, default=0.75)
    ap.add_argument("--baseline-nav", default=None,
                    help="optional previous *_nav_summary.json; candidate must beat it when --require-nav-improvement")
    ap.add_argument("--require-nav-improvement", action="store_true")
    ap.add_argument("--render", default=None, help="optional rendered video/GIF path that must exist and be non-empty")
    ap.add_argument("--check", action="store_true", help="exit nonzero when gates fail")
    args = ap.parse_args()

    root = Path(args.out)
    train = read_last_jsonl(root / f"{args.tag}_train.jsonl")
    deploy = read_json(root / f"{args.tag}_eval_summary.json")
    nav = read_json(root / f"{args.tag}_nav_summary.json")
    modes = [m.strip() for m in args.require_modes.replace(";", ",").split(",") if m.strip()]
    mode_deploy = {m: read_json(root / f"{args.tag}_{m}_eval_summary.json") for m in modes}
    failures = []

    if train is None:
        failures.append("missing train jsonl")
    if deploy is None and not modes:
        failures.append("missing eval summary")
    if nav is None:
        failures.append("missing nav summary")
    for mode, row in mode_deploy.items():
        if row is None:
            failures.append(f"missing {mode} eval summary")

    if train is not None and train.get("track_mean", 0.0) < args.min_track:
        failures.append(f"train track_mean {train.get('track_mean')} < {args.min_track}")
    deploy_rows = mode_deploy if modes else {"deploy": deploy}
    for name, row in deploy_rows.items():
        if row is None:
            continue
        surv = row.get("samples", 0) / max(row.get("total_steps", 1), 1)
        if surv < args.min_survival_frac:
            failures.append(f"{name} eval survival {surv:.2f} < {args.min_survival_frac}")
        if row.get("mean_alignment", 0.0) < args.min_align:
            failures.append(f"{name} eval alignment {row.get('mean_alignment')} < {args.min_align}")
        if row.get("mean_vector_error", 999.0) > args.max_err:
            failures.append(f"{name} eval vector_error {row.get('mean_vector_error')} > {args.max_err}")
    if nav is not None:
        frac = nav.get("reached", 0) / max(nav.get("total_waypoints", 1), 1)
        if frac < args.min_nav_frac:
            failures.append(f"nav reached {frac:.2f} < {args.min_nav_frac}")
        if nav.get("fall_reason"):
            failures.append(f"nav fall {nav.get('fall_reason')}")
        if args.require_nav_improvement:
            if not args.baseline_nav:
                failures.append("nav improvement required but --baseline-nav missing")
            else:
                base = read_json(Path(args.baseline_nav))
                if base is None:
                    failures.append(f"baseline nav missing: {args.baseline_nav}")
                else:
                    cand_score = nav_score(nav)
                    base_score = nav_score(base)
                    if cand_score <= base_score:
                        failures.append(f"nav did not improve: candidate {cand_score} <= baseline {base_score}")
    if args.render:
        rp = Path(args.render)
        if not rp.exists() or rp.stat().st_size <= 0:
            failures.append(f"missing/non-empty render: {rp}")

    summary = dict(tag=args.tag, train=train, deploy=deploy, mode_deploy=mode_deploy,
                   nav=nav, render=args.render, failures=failures,
                   ok=not failures)
    (root / f"{args.tag}_validation.json").write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary, indent=2))
    if args.check and failures:
        sys.exit(1)


if __name__ == "__main__":
    main()
