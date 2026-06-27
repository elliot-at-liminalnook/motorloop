# SPDX-License-Identifier: MIT
"""Sweep striker rod/stroke geometry for a fixed fighter checkpoint."""

from __future__ import annotations

import argparse
import copy
import json
import os
import pickle
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

import train_adversarial as ta  # noqa: E402

OUT = Path(os.environ.get("CODESIGN_OUT", "/root/proj/out"))
OUT.mkdir(parents=True, exist_ok=True)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--tag", default="striker_geometry_sweep")
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--rod-lens", default="0.10,0.16,0.22")
    ap.add_argument("--strokes", default="0.10")
    ap.add_argument("--radii", default="")
    ap.add_argument("--contact-timeconsts", default="",
                    help="optional comma-separated MuJoCo contact solref time constants to sweep")
    ap.add_argument("--contact-dampratio", type=float, default=1.0)
    ap.add_argument("--floor-calf-timeconsts", default="",
                    help="optional comma-separated floor-vs-calf pair solref time constants to sweep")
    ap.add_argument("--floor-calf-dampratio", type=float, default=1.0)
    ap.add_argument("--disable-calf-floor", action="store_true",
                    help="in lean-contact matches, let calf capsules collide with opponents but not the floor")
    ap.add_argument("--bench-seeds", default="20240681")
    ap.add_argument("--bench-epis", type=int, default=8)
    ap.add_argument("--bench-steps", type=int, default=80)
    ap.add_argument("--bench-sep-lo", type=float, default=0.25)
    ap.add_argument("--bench-sep-hi", type=float, default=0.70)
    ap.add_argument("--bench-az", type=float, default=3.14159)
    ap.add_argument("--stochastic", action="store_true",
                    help="sample the policy action distribution instead of evaluating deterministic means")
    ap.add_argument("--lean-contacts", action="store_true")
    ap.add_argument("--engage-obs", action="store_true")
    ap.add_argument("--contact-obs", action="store_true")
    args = ap.parse_args()

    seeds = [int(x) for x in args.bench_seeds.split(",") if x.strip()]
    rod_lens = [float(x) for x in args.rod_lens.split(",") if x.strip()]
    strokes = [float(x) for x in args.strokes.split(",") if x.strip()]
    radii = [float(x) for x in args.radii.split(",") if x.strip()]
    if not radii:
        radii = [float(ta.SPEC.get("striker", {}).get("rod_radius", 0.008))]
    contact_timeconsts = [float(x) for x in args.contact_timeconsts.split(",") if x.strip()]
    if not contact_timeconsts:
        contact_timeconsts = [None]
    floor_calf_timeconsts = [float(x) for x in args.floor_calf_timeconsts.split(",") if x.strip()]
    if not floor_calf_timeconsts:
        floor_calf_timeconsts = [None]

    rows = []
    base_spec = copy.deepcopy(ta.SPEC)
    for floor_calf_timeconst in floor_calf_timeconsts:
        for contact_timeconst in contact_timeconsts:
            for radius in radii:
                for stroke in strokes:
                    for rod_len in rod_lens:
                        spec = copy.deepcopy(base_spec)
                        spec.setdefault("striker", {})
                        spec["striker"]["rod_len"] = rod_len
                        spec["striker"]["stroke"] = stroke
                        spec["striker"]["rod_radius"] = radius
                        if contact_timeconst is not None:
                            spec.setdefault("contact", {})
                            spec["contact"]["solref"] = [contact_timeconst, float(args.contact_dampratio)]
                        if floor_calf_timeconst is not None:
                            spec.setdefault("contact", {})
                            spec["contact"]["floor_calf_solref"] = [
                                floor_calf_timeconst,
                                float(args.floor_calf_dampratio),
                            ]
                        if args.disable_calf_floor:
                            spec.setdefault("contact", {})
                            spec["contact"]["calf_floor"] = False
                        ta.SPEC = spec
                        env = ta.AdversarialEnv(
                            self_collision=not args.lean_contacts,
                            frame_skip=5,
                            sep_lo=args.bench_sep_lo,
                            sep_hi=args.bench_sep_hi,
                            azimuth=args.bench_az,
                            striker=None,
                            opponent="passive",
                            engage_obs=args.engage_obs,
                            contact_obs=args.contact_obs,
                        )
                        params = ta.warm_start(args.ckpt, env.observation_size, env.action_size)
                        if params is None:
                            params = pickle.load(open(args.ckpt, "rb"))
                        vals_by_seed = []
                        for seed in seeds:
                            bench = ta.build_benchmark(
                                env,
                                args.bench_epis,
                                args.bench_steps,
                                seed=seed,
                                deterministic=not args.stochastic,
                            )
                            vals_by_seed.append(np.asarray(bench(params)))
                        vals_by_seed = np.stack(vals_by_seed, axis=0)
                        vals = vals_by_seed.mean(axis=0)
                        rec = {k: float(vals[i]) for i, k in enumerate(ta.BENCH_KEYS)}
                        rec.update({
                            "rod_len": rod_len,
                            "stroke": stroke,
                            "rod_radius": radius,
                            "contact_solref": (
                                [contact_timeconst, float(args.contact_dampratio)]
                                if contact_timeconst is not None else None
                            ),
                            "floor_calf_solref": (
                                [floor_calf_timeconst, float(args.floor_calf_dampratio)]
                                if floor_calf_timeconst is not None else None
                            ),
                            "calf_floor": bool(spec.get("contact", {}).get("calf_floor", True)),
                            "stochastic": bool(args.stochastic),
                            "ratio": rec["dealt"] / max(rec["taken"], 1e-6),
                            "margin": rec["dealt"] - rec["taken"],
                            "seed_rows": [
                                {k: float(row[i]) for i, k in enumerate(ta.BENCH_KEYS)}
                                for row in vals_by_seed
                            ],
                        })
                        rows.append(rec)
                        print(
                            f"rod_len={rod_len:.3f} stroke={stroke:.3f} radius={radius:.4f} "
                            f"solref={contact_timeconst if contact_timeconst is not None else 'default'} "
                            f"floor_calf={floor_calf_timeconst if floor_calf_timeconst is not None else 'default'} "
                            f"calf_floor={spec.get('contact', {}).get('calf_floor', True)} "
                            f"sparc={rec['sparc']:.3f} dealt={rec['dealt']:.4f} "
                            f"taken={rec['taken']:.4f} margin={rec['margin']:+.4f} "
                            f"pen={rec['ac_peak_pen']:.4f} early={rec['ac_dmg_early']:.3f} "
                            f"surv={rec['survival_rate']:.2f} safe={rec['safe_rate']:.2f}",
                            flush=True,
                        )
    ta.SPEC = base_spec
    rows.sort(key=lambda r: (r["dealt"], r["sparc"]), reverse=True)
    out = OUT / f"{args.tag}_striker_geometry_sweep.json"
    out.write_text(json.dumps({"tag": args.tag, "ckpt": args.ckpt, "seeds": seeds, "rows": rows}, indent=2))
    best = rows[0]
    print(
        f"BEST rod_len={best['rod_len']:.3f} stroke={best['stroke']:.3f} "
        f"sparc={best['sparc']:.3f} dealt={best['dealt']:.4f} taken={best['taken']:.4f} "
        f"pen={best['ac_peak_pen']:.4f}",
        flush=True,
    )
    print(f"saved {out}", flush=True)


if __name__ == "__main__":
    os.environ.pop("MUJOCO_GL", None)
    main()
