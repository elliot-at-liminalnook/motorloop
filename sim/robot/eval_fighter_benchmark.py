# SPDX-License-Identifier: MIT
"""Evaluate a fighter checkpoint on the held-out combat benchmark without training."""

from __future__ import annotations

import argparse
import json
import os
import pickle
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from train_adversarial import AdversarialEnv, BENCH_KEYS, build_benchmark, load_opponent  # noqa: E402

OUT = Path(os.environ.get("CODESIGN_OUT", "/root/proj/out"))
OUT.mkdir(parents=True, exist_ok=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tag", default="bench_eval")
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--opponent", choices=["passive", "frozen"], default="passive")
    ap.add_argument("--opp-ckpt", default=None)
    ap.add_argument("--bench-sep-lo", type=float, default=0.4)
    ap.add_argument("--bench-sep-hi", type=float, default=1.2)
    ap.add_argument("--bench-az", type=float, default=3.14159)
    ap.add_argument("--bench-epis", type=int, default=16)
    ap.add_argument("--bench-steps", type=int, default=200)
    ap.add_argument("--seed", type=int, default=20240601)
    ap.add_argument("--frame-skip", type=int, default=5)
    ap.add_argument("--lean-contacts", action="store_true")
    ap.add_argument("--no-striker", action="store_true")
    ap.add_argument("--stochastic", action="store_true",
                    help="sample the policy action distribution instead of using deterministic mean actions")
    ap.add_argument("--face-opponent", action="store_true",
                    help="spawn A and B yawed toward each other for randomized bearings")
    ap.add_argument("--engage-obs", action="store_true",
                    help="append opponent-direction and radial/lateral engagement features to observations")
    ap.add_argument("--out-json", default=None)
    args = ap.parse_args()

    ckpt = Path(args.ckpt)
    params = pickle.load(open(ckpt, "rb"))
    striker = False if args.no_striker else None
    opp = None
    if args.opponent == "frozen":
        if not args.opp_ckpt:
            raise SystemExit("--opponent frozen requires --opp-ckpt")
        opp = load_opponent(args.opp_ckpt)
    env = AdversarialEnv(
        self_collision=not args.lean_contacts,
        frame_skip=args.frame_skip,
        sep_lo=args.bench_sep_lo,
        sep_hi=args.bench_sep_hi,
        azimuth=args.bench_az,
        striker=striker,
        opponent=args.opponent,
        opp_infer=opp,
        face_opponent=args.face_opponent,
        engage_obs=args.engage_obs,
    )
    bench = build_benchmark(env, args.bench_epis, args.bench_steps, seed=args.seed,
                            deterministic=not args.stochastic)
    vals = np.asarray(bench(params))
    rec = {k: float(vals[i]) for i, k in enumerate(BENCH_KEYS)}
    rec.update(
        tag=args.tag,
        ckpt=str(ckpt),
        opponent=args.opponent,
        opp_ckpt=str(args.opp_ckpt or ""),
        bench_sep_lo=args.bench_sep_lo,
        bench_sep_hi=args.bench_sep_hi,
        bench_az=args.bench_az,
        bench_epis=args.bench_epis,
        bench_steps=args.bench_steps,
        seed=args.seed,
        frame_skip=args.frame_skip,
        lean_contacts=bool(args.lean_contacts),
        stochastic=bool(args.stochastic),
        face_opponent=bool(args.face_opponent),
        engage_obs=bool(args.engage_obs),
    )
    rec["bench_ratio"] = rec["dealt"] / max(rec["taken"], 1e-6)
    rec["bench_margin"] = rec["dealt"] - rec["taken"]
    rec["idle_penalty"] = 10.0 * max(0.0, rec["ac_idle"] - 0.3)
    rec["bench_judge"] = (
        100.0 * rec["win_rate"] + rec["sparc"] + 20.0 * rec["bench_margin"] - rec["idle_penalty"]
    )
    out = Path(args.out_json) if args.out_json else OUT / f"{args.tag}_benchmark_eval.json"
    out.write_text(json.dumps(rec, indent=2))
    print(
        f"[fighter-bench] tag={args.tag} win={rec['win_rate']:.3f} "
        f"surv={rec['survival_rate']:.3f} safe={rec['safe_rate']:.3f} "
        f"sparc={rec['sparc']:.2f} ratio={rec['bench_ratio']:.2f} "
        f"opp={args.opponent} stochastic={args.stochastic} ckpt={ckpt.name}",
        flush=True,
    )
    print("METRIC " + " ".join(f"{k}={v}" for k, v in dict(
        stage="fighter_benchmark_eval",
        tag=args.tag,
        win=f"{rec['win_rate']:.4f}",
        survival=f"{rec['survival_rate']:.4f}",
        safe=f"{rec['safe_rate']:.4f}",
        ratio=f"{rec['bench_ratio']:.4f}",
        sparc=f"{rec['sparc']:.3f}",
    ).items()), flush=True)
    print(f"saved {out}", flush=True)


if __name__ == "__main__":
    main()
