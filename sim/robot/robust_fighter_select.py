# SPDX-License-Identifier: MIT
"""Robust paired selection for fighter/self-play checkpoint promotion.

This is deliberately stricter than a single keep-best curve: every candidate is
evaluated on the same benchmark slices and seeds as the baseline, then promoted
only if it preserves real damage, avoids early-contact artifacts, and improves
without paired margin/judge regression.
"""

from __future__ import annotations

import argparse
import json
import os
import pickle
import sys
from dataclasses import dataclass
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from train_adversarial import AdversarialEnv, BENCH_KEYS, build_benchmark, load_opponent  # noqa: E402

OUT = Path(os.environ.get("CODESIGN_OUT", "/root/proj/out"))
OUT.mkdir(parents=True, exist_ok=True)


@dataclass(frozen=True)
class Slice:
    name: str
    sep_lo: float
    sep_hi: float
    az: float
    steps: int
    gate: bool = True


def parse_slice(text: str) -> Slice:
    parts = text.split(":")
    if len(parts) not in (5, 6):
        raise argparse.ArgumentTypeError("slice must be name:sep_lo:sep_hi:az:steps[:gate]")
    gate = True if len(parts) == 5 else parts[5].lower() not in ("0", "false", "no", "report")
    return Slice(parts[0], float(parts[1]), float(parts[2]), float(parts[3]), int(parts[4]), gate)


def add_scores(rec: dict) -> dict:
    rec["bench_ratio"] = rec["dealt"] / max(rec["taken"], 1e-6)
    rec["bench_margin"] = rec["dealt"] - rec["taken"]
    rec["idle_penalty"] = 10.0 * max(0.0, rec["ac_idle"] - 0.3)
    rec["bench_judge"] = (
        100.0 * rec["win_rate"] + rec["sparc"] + 20.0 * rec["bench_margin"] - rec["idle_penalty"]
    )
    return rec


def eval_params(params, bench, *, tag: str, ckpt: str, sl: Slice, seed: int) -> dict:
    vals = np.asarray(bench(params))
    rec = {k: float(vals[i]) for i, k in enumerate(BENCH_KEYS)}
    rec.update(tag=tag, ckpt=ckpt, slice=sl.name, seed=seed, gate=sl.gate,
               bench_sep_lo=sl.sep_lo, bench_sep_hi=sl.sep_hi, bench_az=sl.az,
               bench_steps=sl.steps)
    return add_scores(rec)


def summarize(rows: list[dict]) -> dict:
    gate = [r for r in rows if r.get("gate", True)]
    src = gate or rows
    def vals(k):
        return np.array([float(r[k]) for r in src], dtype=np.float64)
    return {
        "n": len(src),
        "mean_judge": float(vals("bench_judge").mean()),
        "min_judge": float(vals("bench_judge").min()),
        "mean_margin": float(vals("bench_margin").mean()),
        "min_margin": float(vals("bench_margin").min()),
        "mean_dealt": float(vals("dealt").mean()),
        "min_dealt": float(vals("dealt").min()),
        "mean_taken": float(vals("taken").mean()),
        "max_early": float(vals("ac_dmg_early").max()),
        "max_airborne": float(vals("ac_airborne").max()),
        "mean_win": float(vals("win_rate").mean()),
        "min_survival": float(vals("survival_rate").min()),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--baseline", required=True)
    ap.add_argument("--candidate", action="append", default=[],
                    help="checkpoint path, optionally name=path; may be repeated")
    ap.add_argument("--opponent", default=None,
                    help="fixed opponent checkpoint; default is baseline")
    ap.add_argument("--slice", type=parse_slice, action="append", default=[],
                    help="benchmark slice; may be repeated. Defaults to clean100 if omitted")
    ap.add_argument("--seeds", default="20240601,20240602,20240603")
    ap.add_argument("--bench-epis", type=int, default=4)
    ap.add_argument("--frame-skip", type=int, default=5)
    ap.add_argument("--lean-contacts", action="store_true")
    ap.add_argument("--min-dealt", type=float, default=0.02)
    ap.add_argument("--max-early", type=float, default=0.6)
    ap.add_argument("--max-airborne", type=float, default=0.05)
    ap.add_argument("--min-survival", type=float, default=1.0)
    ap.add_argument("--margin-tol", type=float, default=0.0)
    ap.add_argument("--judge-tol", type=float, default=0.0)
    ap.add_argument("--min-mean-margin-delta", type=float, default=0.0)
    ap.add_argument("--min-mean-judge-delta", type=float, default=0.0)
    ap.add_argument("--out-json", default=str(OUT / "robust_fighter_select.json"))
    args = ap.parse_args()
    if not args.slice:
        args.slice = [Slice("clean100", 0.18, 0.45, 0.8, 100, True)]

    seeds = [int(x) for x in args.seeds.split(",") if x.strip()]
    baseline = str(Path(args.baseline))
    opponent = str(Path(args.opponent or args.baseline))
    ckpts = [("baseline", baseline)]
    for i, item in enumerate(args.candidate):
        if "=" in item:
            name, path = item.split("=", 1)
        else:
            path = item
            name = Path(path).stem or f"candidate{i}"
        ckpts.append((name, str(Path(path))))

    params = {name: pickle.load(open(path, "rb")) for name, path in ckpts}
    opp = load_opponent(opponent)
    rows_by_name = {name: [] for name, _ in ckpts}

    for sl in args.slice:
        env = AdversarialEnv(self_collision=not args.lean_contacts,
                             frame_skip=args.frame_skip,
                             sep_lo=sl.sep_lo, sep_hi=sl.sep_hi, azimuth=sl.az,
                             opponent="frozen", opp_infer=opp)
        for seed in seeds:
            bench = build_benchmark(env, args.bench_epis, sl.steps, seed=seed)
            for name, path in ckpts:
                rows_by_name[name].append(eval_params(params[name], bench, tag=name, ckpt=path, sl=sl, seed=seed))

    baseline_rows = rows_by_name["baseline"]
    base_summary = summarize(baseline_rows)
    candidates = []
    best = None
    for name, path in ckpts[1:]:
        rows = rows_by_name[name]
        summary = summarize(rows)
        paired = []
        for b, c in zip(baseline_rows, rows):
            if not c.get("gate", True):
                continue
            paired.append({
                "slice": c["slice"],
                "seed": c["seed"],
                "margin_delta": c["bench_margin"] - b["bench_margin"],
                "judge_delta": c["bench_judge"] - b["bench_judge"],
                "dealt": c["dealt"],
                "early": c["ac_dmg_early"],
                "survival": c["survival_rate"],
            })
        damage_ok = summary["min_dealt"] >= args.min_dealt
        early_ok = summary["max_early"] <= args.max_early
        air_ok = summary["max_airborne"] <= args.max_airborne
        survival_ok = summary["min_survival"] >= args.min_survival
        paired_margin_ok = all(p["margin_delta"] >= -args.margin_tol for p in paired)
        paired_judge_ok = all(p["judge_delta"] >= -args.judge_tol for p in paired)
        mean_margin_delta = summary["mean_margin"] - base_summary["mean_margin"]
        mean_judge_delta = summary["mean_judge"] - base_summary["mean_judge"]
        improved = (mean_margin_delta >= args.min_mean_margin_delta and
                    mean_judge_delta >= args.min_mean_judge_delta and
                    (mean_margin_delta > 0.0 or mean_judge_delta > 0.0))
        accepted = all([damage_ok, early_ok, air_ok, survival_ok,
                        paired_margin_ok, paired_judge_ok, improved])
        rec = {
            "name": name,
            "ckpt": path,
            "summary": summary,
            "rows": rows,
            "mean_margin_delta": mean_margin_delta,
            "mean_judge_delta": mean_judge_delta,
            "accepted": bool(accepted),
            "checks": {
                "damage_ok": damage_ok,
                "early_ok": early_ok,
                "air_ok": air_ok,
                "survival_ok": survival_ok,
                "paired_margin_ok": paired_margin_ok,
                "paired_judge_ok": paired_judge_ok,
                "improved": improved,
            },
            "paired": paired,
        }
        candidates.append(rec)
        if accepted and (best is None or summary["mean_judge"] > best["summary"]["mean_judge"]):
            best = rec

    out = {
        "baseline": {"ckpt": baseline, "summary": base_summary, "rows": baseline_rows},
        "opponent": opponent,
        "slices": [sl.__dict__ for sl in args.slice],
        "seeds": seeds,
        "candidates": candidates,
        "accepted": best,
    }
    Path(args.out_json).write_text(json.dumps(out, indent=2))
    print(json.dumps({
        "baseline": base_summary,
        "accepted": None if best is None else {
            "name": best["name"],
            "ckpt": best["ckpt"],
            "summary": best["summary"],
            "mean_margin_delta": best["mean_margin_delta"],
            "mean_judge_delta": best["mean_judge_delta"],
        },
        "candidates": [
            {"name": c["name"], "accepted": c["accepted"],
             "mean_margin_delta": c["mean_margin_delta"],
             "mean_judge_delta": c["mean_judge_delta"],
             "checks": c["checks"]}
            for c in candidates
        ],
        "out_json": args.out_json,
    }, indent=2))


if __name__ == "__main__":
    main()
