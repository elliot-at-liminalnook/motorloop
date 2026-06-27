# SPDX-License-Identifier: MIT
"""Evaluate many fighter checkpoints with one compiled benchmark."""

from __future__ import annotations

import argparse
import glob
import json
import os
from pathlib import Path

import numpy as np

from train_adversarial import AdversarialEnv, BENCH_KEYS, build_benchmark, warm_start


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=os.environ.get("CODESIGN_OUT", "/root/proj/out"))
    ap.add_argument("--bench-sep-lo", type=float, default=0.25)
    ap.add_argument("--bench-sep-hi", type=float, default=0.70)
    ap.add_argument("--bench-az", type=float, default=3.14159)
    ap.add_argument("--bench-epis", type=int, default=4)
    ap.add_argument("--bench-steps", type=int, default=80)
    ap.add_argument("--seed", type=int, default=20240601)
    ap.add_argument("--lean-contacts", action="store_true")
    ap.add_argument("--pattern", action="append", default=[])
    ap.add_argument("--result-json", default=None)
    args = ap.parse_args()

    out = Path(args.out)
    patterns = args.pattern or [
        "spr*_best.pkl", "spr*_ckpt.pkl", "robust*_best.pkl", "robust*_ckpt.pkl",
        "dmg*_best.pkl", "dmg*_ckpt.pkl", "clean*_best.pkl", "margin*_best.pkl",
        "target*_best.pkl", "head*_best.pkl", "head_clean*_best.pkl",
        "selfplay_best.pkl", "tfix*_best.pkl", "tbase*_best.pkl", "tjudge*_best.pkl",
        "scaffold_combat_contact_short/*best.pkl", "sparc_push*_best.pkl",
    ]
    paths: list[str] = []
    for pat in patterns:
        paths.extend(glob.glob(str(out / pat)))
    paths = sorted(set(paths))
    print(f"candidate_count={len(paths)}", flush=True)

    env = AdversarialEnv(
        self_collision=not args.lean_contacts,
        frame_skip=5,
        sep_lo=args.bench_sep_lo,
        sep_hi=args.bench_sep_hi,
        azimuth=args.bench_az,
        striker=None,
    )
    bench = build_benchmark(env, args.bench_epis, args.bench_steps, seed=args.seed)
    rows = []
    for path in paths:
        try:
            params = warm_start(path, env.observation_size, env.action_size)
            if params is None:
                continue
            vals = np.asarray(bench(params))
            rec = {k: float(vals[i]) for i, k in enumerate(BENCH_KEYS)}
            rec["path"] = path
            rec["name"] = os.path.basename(path)
            rec["margin"] = rec["dealt"] - rec["taken"]
            rec["judge"] = (
                100.0 * rec["win_rate"]
                + rec["sparc"]
                + 20.0 * rec["margin"]
                - 10.0 * max(0.0, rec["ac_idle"] - 0.3)
            )
            rows.append(rec)
            print(json.dumps({k: rec[k] for k in [
                "name", "sparc", "dealt", "taken", "closing", "fleeing",
                "win_rate", "survival_rate", "safe_rate", "margin", "judge",
            ]}), flush=True)
        except Exception as exc:
            print(json.dumps({
                "name": os.path.basename(path),
                "error": type(exc).__name__,
                "msg": str(exc)[:200],
            }), flush=True)

    rows.sort(key=lambda r: r["sparc"], reverse=True)
    result_path = Path(args.result_json) if args.result_json else out / "fighter_checkpoint_sparc_sweep.json"
    result_path.write_text(json.dumps(rows, indent=2))
    print("TOP_BY_SPARC", flush=True)
    for rec in rows[:10]:
        print(json.dumps({k: rec[k] for k in [
            "name", "path", "sparc", "dealt", "taken", "closing",
            "fleeing", "win_rate", "safe_rate", "margin", "judge",
        ]}), flush=True)
    print(f"saved {result_path}", flush=True)


if __name__ == "__main__":
    main()
