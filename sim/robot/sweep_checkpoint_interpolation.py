# SPDX-License-Identifier: MIT
"""Sweep linear interpolation between two fighter checkpoints."""

from __future__ import annotations

import argparse
import copy
import json
import os
import pickle
import sys
from pathlib import Path

import jax
import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

import train_adversarial as ta  # noqa: E402
from search_policy_bias import build_multiseed_benchmark  # noqa: E402

OUT = Path(os.environ.get("CODESIGN_OUT", "/root/proj/out"))
OUT.mkdir(parents=True, exist_ok=True)


def _blend_tree(base, other, alpha: float):
    def blend(a, b):
        if hasattr(a, "shape") and hasattr(b, "shape") and a.shape == b.shape:
            return (1.0 - alpha) * a + alpha * b
        return a

    return jax.tree_util.tree_map(blend, base, other)


def blend_checkpoint(base, other, alpha: float, *, blend_normalizer: bool, blend_value: bool):
    base = list(base)
    other = list(other)
    out = list(base)
    if blend_normalizer:
        out[0] = _blend_tree(base[0], other[0], alpha)
    out[1] = _blend_tree(base[1], other[1], alpha)
    if blend_value and len(base) > 2 and len(other) > 2:
        out[2] = _blend_tree(base[2], other[2], alpha)
    return tuple(out)


def _mean_rows(seed_rows: list[dict], seeds: list[int], subset: set[int] | None) -> dict:
    rows = [r for s, r in zip(seeds, seed_rows) if subset is None or s in subset]
    if not rows:
        return {}
    rec = {}
    for key in ta.BENCH_KEYS:
        rec[key] = float(np.mean([r[key] for r in rows]))
    rec["margin"] = rec["dealt"] - rec["taken"]
    rec["ratio"] = rec["dealt"] / max(rec["taken"], 1e-6)
    rec["min_margin"] = float(min(r["dealt"] - r["taken"] for r in rows))
    rec["min_dealt"] = float(min(r["dealt"] for r in rows))
    rec["min_survival"] = float(min(r["survival_rate"] for r in rows))
    rec["max_peak_pen"] = float(max(r["ac_peak_pen"] for r in rows))
    return rec


def _apply_overrides(args) -> None:
    if not any(x is not None for x in (
        args.striker_rod_len,
        args.striker_stroke,
        args.striker_rod_radius,
        args.contact_solref_timeconst,
        args.floor_calf_solref_timeconst,
    )) and not args.disable_calf_floor:
        return
    ta.SPEC = copy.deepcopy(ta.SPEC)
    ta.SPEC.setdefault("striker", {})
    ta.SPEC.setdefault("contact", {})
    if args.striker_rod_len is not None:
        ta.SPEC["striker"]["rod_len"] = float(args.striker_rod_len)
    if args.striker_stroke is not None:
        ta.SPEC["striker"]["stroke"] = float(args.striker_stroke)
    if args.striker_rod_radius is not None:
        ta.SPEC["striker"]["rod_radius"] = float(args.striker_rod_radius)
    if args.contact_solref_timeconst is not None:
        ta.SPEC["contact"]["solref"] = [
            float(args.contact_solref_timeconst),
            float(args.contact_solref_dampratio),
        ]
    if args.floor_calf_solref_timeconst is not None:
        ta.SPEC["contact"]["floor_calf_solref"] = [
            float(args.floor_calf_solref_timeconst),
            float(args.floor_calf_solref_dampratio),
        ]
    if args.disable_calf_floor:
        ta.SPEC["contact"]["calf_floor"] = False


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--tag", default="checkpoint_interpolation")
    ap.add_argument("--base", required=True)
    ap.add_argument("--other", required=True)
    ap.add_argument("--alphas", default="0,0.25,0.5,0.75,1")
    ap.add_argument("--bench-seeds", default="20240601")
    ap.add_argument("--strict-seeds", default="20240601,20240611,20240621,20240631,20240641")
    ap.add_argument("--heldout-seeds", default="20240651,20240661")
    ap.add_argument("--all7-seeds", default="20240601,20240611,20240621,20240631,20240641,20240671,20240681")
    ap.add_argument("--bench-epis", type=int, default=8)
    ap.add_argument("--bench-steps", type=int, default=80)
    ap.add_argument("--bench-sep-lo", type=float, default=0.25)
    ap.add_argument("--bench-sep-hi", type=float, default=0.70)
    ap.add_argument("--bench-az", type=float, default=3.14159)
    ap.add_argument("--frame-skip", type=int, default=5)
    ap.add_argument("--lean-contacts", action="store_true")
    ap.add_argument("--engage-obs", action="store_true")
    ap.add_argument("--contact-obs", action="store_true")
    ap.add_argument("--striker-rod-len", type=float, default=None)
    ap.add_argument("--striker-stroke", type=float, default=None)
    ap.add_argument("--striker-rod-radius", type=float, default=None)
    ap.add_argument("--contact-solref-timeconst", type=float, default=None)
    ap.add_argument("--contact-solref-dampratio", type=float, default=1.0)
    ap.add_argument("--floor-calf-solref-timeconst", type=float, default=None)
    ap.add_argument("--floor-calf-solref-dampratio", type=float, default=1.0)
    ap.add_argument("--disable-calf-floor", action="store_true")
    ap.add_argument("--blend-normalizer", action="store_true")
    ap.add_argument("--blend-value", action="store_true")
    args = ap.parse_args()

    _apply_overrides(args)
    seeds = [int(x) for x in args.bench_seeds.split(",") if x.strip()]
    strict = {int(x) for x in args.strict_seeds.split(",") if x.strip()}
    heldout = {int(x) for x in args.heldout_seeds.split(",") if x.strip()}
    all7 = {int(x) for x in args.all7_seeds.split(",") if x.strip()}
    alphas = [float(x) for x in args.alphas.split(",") if x.strip()]

    env = ta.AdversarialEnv(
        self_collision=not args.lean_contacts,
        frame_skip=args.frame_skip,
        sep_lo=args.bench_sep_lo,
        sep_hi=args.bench_sep_hi,
        azimuth=args.bench_az,
        striker=None,
        opponent="passive",
        engage_obs=args.engage_obs,
        contact_obs=args.contact_obs,
    )
    base = ta.warm_start(args.base, env.observation_size, env.action_size)
    other = ta.warm_start(args.other, env.observation_size, env.action_size)
    if base is None:
        base = pickle.load(open(args.base, "rb"))
    if other is None:
        other = pickle.load(open(args.other, "rb"))

    bench = build_multiseed_benchmark(env, args.bench_epis, args.bench_steps, seeds)
    rows = []
    for alpha in alphas:
        params = blend_checkpoint(
            base,
            other,
            alpha,
            blend_normalizer=args.blend_normalizer,
            blend_value=args.blend_value,
        )
        vals, vals_by_seed = bench(params)
        vals = np.asarray(vals)
        vals_by_seed = np.asarray(vals_by_seed)
        seed_rows = [{k: float(row[i]) for i, k in enumerate(ta.BENCH_KEYS)} for row in vals_by_seed]
        rec = {k: float(vals[i]) for i, k in enumerate(ta.BENCH_KEYS)}
        rec.update({
            "alpha": alpha,
            "margin": rec["dealt"] - rec["taken"],
            "ratio": rec["dealt"] / max(rec["taken"], 1e-6),
            "strict": _mean_rows(seed_rows, seeds, strict),
            "heldout": _mean_rows(seed_rows, seeds, heldout),
            "all7": _mean_rows(seed_rows, seeds, all7),
            "seed_rows": seed_rows,
        })
        rows.append(rec)
        print(
            f"alpha={alpha:.3f} all9={rec['sparc']:.3f} "
            f"all7={rec['all7'].get('sparc', 0.0):.3f} "
            f"strict={rec['strict'].get('sparc', 0.0):.3f} "
            f"heldout={rec['heldout'].get('sparc', 0.0):.3f} "
            f"surv={rec['all7'].get('min_survival', 0.0):.2f} "
            f"pen={rec['all7'].get('max_peak_pen', 0.0):.4f}",
            flush=True,
        )

    rows.sort(key=lambda r: (
        min(r["all7"].get("sparc", -1e9), r["heldout"].get("sparc", -1e9)),
        r["all7"].get("sparc", -1e9) + r["heldout"].get("sparc", -1e9),
    ), reverse=True)
    best = rows[0]
    best_params = blend_checkpoint(
        base,
        other,
        best["alpha"],
        blend_normalizer=args.blend_normalizer,
        blend_value=args.blend_value,
    )
    ckpt_out = OUT / f"{args.tag}_best.pkl"
    pickle.dump(best_params, open(ckpt_out, "wb"))
    out_json = OUT / f"{args.tag}_checkpoint_interpolation.json"
    out_json.write_text(json.dumps({
        "tag": args.tag,
        "base": args.base,
        "other": args.other,
        "seeds": seeds,
        "rows": rows,
        "best_alpha": best["alpha"],
        "artifact": str(ckpt_out),
    }, indent=2))
    print(f"BEST alpha={best['alpha']:.3f} saved {ckpt_out}", flush=True)
    print(f"saved {out_json}", flush=True)


if __name__ == "__main__":
    main()
