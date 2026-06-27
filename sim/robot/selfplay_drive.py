# SPDX-License-Identifier: MIT
"""STEP 2/F — self-play LEAGUE driver (the open-ended "keeps improving" engine).

NOTE: superseded by the `arena` framework (`python -m arena.cli league/pipeline ...`, which reuses
this module's RW/BENCH as its config source). Kept working + as that source.

The skill curriculum (`curriculum_drive`) trains the striker vs a PASSIVE opponent → it saturates
(you can only get so good at hitting a dummy). Self-play supplies an EVER-improving opponent (an
arms race), so there's always a harder problem to learn. Seeded from the skill fighter, each round
trains A vs a FROZEN snapshot drawn from a Hall of Fame, then snapshots A back in.

Stability (DeepMind 1v1 soccer recipe, the proven analog):
  * the TRAINING opponent is sampled from the FIRST QUARTER of the snapshot pool (oldest 25%) —
    sampling the archive, not just the latest, is what stops the arms race CYCLING;
  * the BENCHMARK opponent is a FIXED reference (the seed) so the keep-best curve is comparable
    across rounds — best-so-far stays monotone by construction even as the training opponent rotates.
  (Opponent-ID critic conditioning is a further refinement, noted but not yet wired.)

Resume-safe (`selfplay_state.json`) → continue a short league into a longer one.

  python selfplay_drive.py --seed-ckpt out/curriculum_best.pkl --rounds 12 --round-steps 10000000 --envs 8192 --lean-contacts
  python selfplay_drive.py --resume
  python selfplay_drive.py --tiny
"""

from __future__ import annotations

import argparse, json, math, os, shutil, subprocess, sys, time
from pathlib import Path

HERE = Path(__file__).resolve().parent
OUT = Path(os.environ.get("CODESIGN_OUT", "/root/proj/out")); OUT.mkdir(parents=True, exist_ok=True)
STATE = OUT / "selfplay_state.json"

# self-play reward weights (win the exchange vs a fighting opponent) + spawn spread. Range MATCHED
# to THIS body's engagement envelope: max reach 0.62 m and gear-12 (half-speed) legs that close at
# ~0.02/step, so the old 0.4-1.2 m left fighters unable to make contact in 200 steps (dealt=taken=0).
# 0.25-0.7 spans in-reach (0.25, guaranteed contact) to a short close (0.7); approach/shaping up to drive it.
RW = dict(sep_lo=0.25, sep_hi=0.7, approach=3.0, azimuth=3.14159, shaping=0.8,
          clean=10.0, trade=8.0, taken=0.0, disengage=1.5, fire=1.0, early=0.0, min_hit=0)
BENCH = dict(sep_lo=0.25, sep_hi=0.7, az=3.14159, epis=16, steps=200)


def load_state():
    if STATE.exists():
        return json.loads(STATE.read_text())
    return None


def save_state(st):
    STATE.write_text(json.dumps(st, indent=2))


def first_quarter(hof):
    """The oldest 25% of snapshots (the stable pool the paper samples opponents from)."""
    k = max(1, (len(hof) + 3) // 4)
    return hof[:k]


def run_round(py, rd, train_opp, bench_opp, warm, steps, cum_base, lean, tiny, envs, batch,
              frame_skip, bench_epis, bench_steps, accept_metric, early_hit_penalty, min_hit_step,
              min_keep_dealt, max_keep_early, taken_weight, bench_seeds):
    tag = f"spr{rd}"
    log = OUT / f"sp_{tag}.log"
    for stale in (OUT / f"{tag}_best.pkl", OUT / f"{tag}_state.json",
                  OUT / f"{tag}_ckpt.pkl", OUT / f"{tag}_prev.pkl"):
        try:
            stale.unlink()
        except FileNotFoundError:
            pass
    cmd = [py, "-u", str(HERE / "train_adversarial.py"),
           "--tag", tag, "--steps", str(steps), "--cum-base", str(cum_base),
           "--opponent", "frozen", "--opp-ckpt", str(train_opp), "--bench-opp-ckpt", str(bench_opp),
           "--resume", str(warm),
           "--sep-lo", str(RW["sep_lo"]), "--sep-hi", str(RW["sep_hi"]),
           "--approach-weight", str(RW["approach"]), "--azimuth", str(RW["azimuth"]),
           "--shaping", str(RW["shaping"]), "--clean-weight", str(RW["clean"]),
           "--trade-weight", str(RW["trade"]), "--disengage-weight", str(RW["disengage"]),
           "--taken-weight", str(taken_weight),
           "--fire-shaping", str(RW["fire"]),
           "--early-hit-penalty", str(early_hit_penalty),
           "--min-hit-step", str(min_hit_step),
           "--bench-sep-lo", str(BENCH["sep_lo"]), "--bench-sep-hi", str(BENCH["sep_hi"]),
           "--bench-az", str(BENCH["az"])]
    cmd += ["--frame-skip", str(frame_skip), "--bench-epis", str(bench_epis),
            "--bench-steps", str(bench_steps), "--keep-metric", str(accept_metric),
            "--bench-seeds", str(bench_seeds),
            "--min-keep-dealt", str(min_keep_dealt),
            "--max-keep-early-dmg", str(max_keep_early)]
    if envs: cmd += ["--envs", str(envs)]
    if batch: cmd += ["--batch", str(batch)]
    if lean: cmd += ["--lean-contacts"]
    if tiny: cmd += ["--tiny"]
    print(f"\n=== ROUND {rd}: train-opp={Path(train_opp).name} bench-opp={Path(bench_opp).name} "
          f"warm={Path(warm).name} steps={steps:,} ===", flush=True)
    t0 = time.time()
    with open(log, "w") as lf:
        rc = subprocess.run(cmd, stdout=lf, stderr=subprocess.STDOUT, env={**os.environ}).returncode
    sf = OUT / f"{tag}_state.json"
    best_path = OUT / f"{tag}_best.pkl"
    if rc != 0 or not sf.exists() or not best_path.exists():
        print(f"ROUND {rd} FAILED rc={rc} (see {log})", flush=True); return None
    stt = json.loads(sf.read_text())
    print(f"ROUND {rd} done in {time.time()-t0:.0f}s: best_bench={stt['best_bench']} "
          f"last_ratio={stt.get('last_ratio')} cum_step={stt['cum_step']}", flush=True)
    return stt, str(best_path)


def eval_seed_benchmark(py, seed, lean, frame_skip, bench_epis, bench_steps, bench_seeds):
    """Evaluate the seed against its fixed frozen copy before any round is accepted."""
    seed_list = [int(x) for x in str(bench_seeds).split(",") if x.strip()]
    recs = []
    for bench_seed in seed_list:
        out = OUT / f"selfplay_seed_benchmark_eval_{bench_seed}.json"
        log = OUT / f"selfplay_seed_benchmark_eval_{bench_seed}.log"
        try:
            out.unlink()
        except FileNotFoundError:
            pass
        cmd = [py, "-u", str(HERE / "eval_fighter_benchmark.py"),
               "--tag", f"selfplay_seed_{bench_seed}",
               "--ckpt", str(seed),
               "--opponent", "frozen",
               "--opp-ckpt", str(seed),
               "--bench-sep-lo", str(BENCH["sep_lo"]),
               "--bench-sep-hi", str(BENCH["sep_hi"]),
               "--bench-az", str(BENCH["az"]),
               "--bench-epis", str(bench_epis),
               "--bench-steps", str(bench_steps),
               "--seed", str(bench_seed),
               "--frame-skip", str(frame_skip),
               "--out-json", str(out)]
        if lean:
            cmd.append("--lean-contacts")
        with open(log, "w") as lf:
            rc = subprocess.run(cmd, stdout=lf, stderr=subprocess.STDOUT, env={**os.environ}).returncode
        if rc != 0 or not out.exists():
            print(f"SEED BENCHMARK FAILED rc={rc} seed={bench_seed} (see {log})", flush=True)
            return None
        recs.append(json.loads(out.read_text()))
    out = OUT / "selfplay_seed_benchmark_eval.json"
    log = OUT / "selfplay_seed_benchmark_eval.log"
    log.write_text("\n".join(json.dumps(r, sort_keys=True) for r in recs) + "\n")
    numeric = [k for k, v in recs[0].items() if isinstance(v, (int, float))]
    rec = {k: float(sum(float(r[k]) for r in recs) / len(recs)) for k in numeric}
    rec.update(recs[0])
    for k in numeric:
        rec[k] = float(sum(float(r[k]) for r in recs) / len(recs))
    margins = [float(r["bench_margin"]) for r in recs]
    judges = [float(r["bench_judge"]) for r in recs]
    dealt = [float(r["dealt"]) for r in recs]
    early = [float(r["ac_dmg_early"]) for r in recs]
    rec["bench_min_margin"] = min(margins)
    rec["bench_min_judge"] = min(judges)
    rec["bench_min_dealt"] = min(dealt)
    rec["bench_max_early"] = max(early)
    rec["bench_seeds"] = seed_list
    out.write_text(json.dumps(rec, indent=2))
    print(f"SEED BENCHMARK: win={rec['win_rate']:.3f} sparc={rec['sparc']:.2f} "
          f"ratio={rec['bench_ratio']:.2f} margin={rec['bench_margin']:+.3f} "
          f"minMargin={rec['bench_min_margin']:+.3f} surv={rec['survival_rate']:.2f} "
          f"safe={rec['safe_rate']:.2f}",
          flush=True)
    return rec


def delta_diagnostic(py, tag, before, after, ref_deltas):
    """Task-vector diagnostic for adapter-style continual learning decisions."""
    out_json = OUT / f"{tag}_delta.json"
    out_npz = OUT / f"{tag}_delta.npz"
    log = OUT / f"{tag}_delta.log"
    cmd = [py, "-u", str(HERE / "skill_delta.py"),
           "--before", str(before), "--after", str(after),
           "--out-json", str(out_json), "--out-npz", str(out_npz)]
    for ref in ref_deltas or []:
        if ref and os.path.exists(ref):
            cmd += ["--ref-delta", str(ref)]
    with open(log, "w") as lf:
        rc = subprocess.run(cmd, stdout=lf, stderr=subprocess.STDOUT, env={**os.environ}).returncode
    if rc != 0 or not out_json.exists() or not out_npz.exists():
        print(f"DELTA {tag}: failed rc={rc} (see {log})", flush=True)
        return dict(ok=False, json=str(out_json), npz=str(out_npz), log=str(log))
    rec = json.loads(out_json.read_text())
    dec = rec.get("decision", {})
    print(f"DELTA {tag}: {dec.get('action', 'unknown')} rank={dec.get('rank_hint', '?')} "
          f"rel={rec.get('relative_norm', 0):.3e} refs={len(rec.get('refs', []))} "
          f"reason={dec.get('reason', '')}", flush=True)
    return dict(ok=True, json=str(out_json), npz=str(out_npz),
                relative_norm=rec.get("relative_norm"),
                decision=dec,
                refs=rec.get("refs", []))


def run_robust_promotion(py, tag, baseline, candidate, opponent, slices, seeds, bench_epis,
                         frame_skip, lean, min_dealt, max_early, margin_tol, judge_tol,
                         min_mean_margin_delta, min_mean_judge_delta):
    """Independent paired selector for Hall-of-Fame promotion."""
    out_json = OUT / f"{tag}_robust_select.json"
    log = OUT / f"{tag}_robust_select.log"
    cmd = [py, "-u", str(HERE / "robust_fighter_select.py"),
           "--baseline", str(baseline),
           "--candidate", f"{tag}={candidate}",
           "--opponent", str(opponent),
           "--seeds", str(seeds),
           "--bench-epis", str(bench_epis),
           "--frame-skip", str(frame_skip),
           "--min-dealt", str(min_dealt),
           "--max-early", str(max_early),
           "--margin-tol", str(margin_tol),
           "--judge-tol", str(judge_tol),
           "--min-mean-margin-delta", str(min_mean_margin_delta),
           "--min-mean-judge-delta", str(min_mean_judge_delta),
           "--out-json", str(out_json)]
    for sl in slices:
        cmd += ["--slice", str(sl)]
    if lean:
        cmd.append("--lean-contacts")
    with open(log, "w") as lf:
        rc = subprocess.run(cmd, stdout=lf, stderr=subprocess.STDOUT, env={**os.environ}).returncode
    if rc != 0 or not out_json.exists():
        print(f"ROBUST {tag}: failed rc={rc} (see {log})", flush=True)
        return dict(ok=False, accepted=False, json=str(out_json), log=str(log), reason="selector_failed")
    rec = json.loads(out_json.read_text())
    best = rec.get("accepted")
    cand = rec.get("candidates", [{}])[0]
    checks = cand.get("checks", {})
    summary = cand.get("summary", {})
    accepted = best is not None
    print(f"ROBUST {tag}: {'ACCEPT' if accepted else 'REJECT'} "
          f"mean_margin_delta={cand.get('mean_margin_delta', 0.0):+.4f} "
          f"mean_judge_delta={cand.get('mean_judge_delta', 0.0):+.3f} "
          f"min_margin={summary.get('min_margin', 0.0):+.4f} "
          f"checks={checks}", flush=True)
    return dict(ok=True, accepted=accepted, json=str(out_json), log=str(log),
                summary=summary, checks=checks,
                mean_margin_delta=cand.get("mean_margin_delta"),
                mean_judge_delta=cand.get("mean_judge_delta"))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed-ckpt", default=str(OUT / "curriculum_best.pkl"),
                    help="the skill-curriculum fighter that seeds the league (HoF[0] + fixed benchmark ref)")
    ap.add_argument("--rounds", type=int, default=12)
    ap.add_argument("--round-steps", type=int, default=10_000_000)
    ap.add_argument("--tol", type=float, default=0.0,
                    help="win-rate regression tolerance for accepting a round into the HoF")
    ap.add_argument("--envs", type=int, default=0)
    ap.add_argument("--batch", type=int, default=0)
    ap.add_argument("--frame-skip", type=int, default=5)
    ap.add_argument("--bench-epis", type=int, default=BENCH["epis"])
    ap.add_argument("--bench-steps", type=int, default=BENCH["steps"])
    ap.add_argument("--bench-seeds", default="20240601")
    ap.add_argument("--train-sep-lo", type=float, default=RW["sep_lo"])
    ap.add_argument("--train-sep-hi", type=float, default=RW["sep_hi"])
    ap.add_argument("--train-azimuth", type=float, default=RW["azimuth"])
    ap.add_argument("--bench-sep-lo", type=float, default=BENCH["sep_lo"])
    ap.add_argument("--bench-sep-hi", type=float, default=BENCH["sep_hi"])
    ap.add_argument("--bench-az", type=float, default=BENCH["az"])
    ap.add_argument("--archive-non-improvements", action="store_true",
                    help="archive non-regressing but not improved rounds into HoF")
    ap.add_argument("--accept-metric", choices=["win", "sparc", "ratio", "margin", "judge",
                                                "min_margin", "min_judge"], default="win",
                    help="round acceptance metric; win is strict, sparc is useful for early self-play shaping")
    ap.add_argument("--min-accepted-dealt", type=float, default=0.0,
                    help="minimum held-out damage dealt required before a round may enter the HoF")
    ap.add_argument("--max-accepted-early-dmg", type=float, default=1.0,
                    help="maximum best-checkpoint damage fraction allowed before the early-hit window")
    ap.add_argument("--margin-tol", type=float, default=0.0,
                    help="allowed held-out damage-margin regression for self-play acceptance")
    ap.add_argument("--early-hit-penalty", type=float, default=RW["early"],
                    help="training penalty for damage before --min-hit-step")
    ap.add_argument("--min-hit-step", type=int, default=RW["min_hit"],
                    help="training reward only counts dealt damage at/after this env step")
    ap.add_argument("--taken-weight", type=float, default=RW["taken"],
                    help="extra training penalty on damage taken")
    ap.add_argument("--max-stale-rounds", type=int, default=2,
                    help="stop after this many accepted-but-not-improved rounds")
    ap.add_argument("--no-robust-promotion", dest="robust_promotion", action="store_false",
                    help="skip independent paired selector before HoF promotion")
    ap.set_defaults(robust_promotion=True)
    ap.add_argument("--robust-slice", action="append", default=[],
                    help="robust promotion slice name:sep_lo:sep_hi:az:steps[:gate]; "
                         "default: clean100 and bridge100")
    ap.add_argument("--robust-margin-tol", type=float, default=0.0,
                    help="paired margin regression tolerance for robust promotion")
    ap.add_argument("--robust-judge-tol", type=float, default=0.0,
                    help="paired judge regression tolerance for robust promotion")
    ap.add_argument("--robust-min-mean-margin-delta", type=float, default=0.0,
                    help="minimum mean margin improvement required by robust promotion")
    ap.add_argument("--robust-min-mean-judge-delta", type=float, default=0.0,
                    help="minimum mean judge improvement required by robust promotion")
    ap.add_argument("--lean-contacts", action="store_true")
    ap.add_argument("--resume", action="store_true")
    ap.add_argument("--tiny", action="store_true")
    args = ap.parse_args()
    py = sys.executable
    if args.tiny:
        args.rounds, args.round_steps = 2, 8_000
        args.bench_epis, args.bench_steps = min(args.bench_epis, 4), min(args.bench_steps, 80)
    RW["sep_lo"], RW["sep_hi"], RW["azimuth"] = args.train_sep_lo, args.train_sep_hi, args.train_azimuth
    BENCH["sep_lo"], BENCH["sep_hi"], BENCH["az"] = args.bench_sep_lo, args.bench_sep_hi, args.bench_az
    robust_slices = args.robust_slice or [
        "clean100:0.18:0.45:0.8:100",
        "bridge100:0.25:0.45:0.5:100",
    ]

    import numpy as np
    rng = np.random.default_rng(0)
    st = load_state() if args.resume else None
    if st is None:
        if not os.path.exists(args.seed_ckpt):
            print(f"SEED MISSING: {args.seed_ckpt} (run the skill curriculum first)"); return
        seed = str(OUT / "selfplay_seed.pkl"); shutil.copy(args.seed_ckpt, seed)
        seed_eval = eval_seed_benchmark(py, seed, args.lean_contacts, args.frame_skip,
                                        args.bench_epis, args.bench_steps, args.bench_seeds)
        if seed_eval is None:
            return
        seed_margin = float(seed_eval.get("bench_margin", seed_eval["dealt"] - seed_eval["taken"]))
        st = dict(round=0, hof=[seed], seed=seed, global_best_ckpt=seed,
                  global_best_bench=float({"win": seed_eval["win_rate"],
                                           "sparc": seed_eval["sparc"],
                                           "ratio": seed_eval["bench_ratio"],
                                           "margin": seed_eval["dealt"] - seed_eval["taken"],
                                           "judge": seed_eval.get(
                                               "bench_judge",
                                               100.0 * seed_eval["win_rate"] + seed_eval["sparc"]
                                               + 20.0 * (seed_eval["dealt"] - seed_eval["taken"])
                                               - 10.0 * max(0.0, seed_eval.get("ac_idle", 0.0) - 0.3)),
                                           "min_margin": seed_eval.get("bench_min_margin", seed_margin),
                                           "min_judge": seed_eval.get("bench_min_judge", seed_eval.get("bench_judge", 0.0))}[args.accept_metric]),
                  global_best_sparc=float(seed_eval["sparc"]),
                  global_best_dealt=float(seed_eval["dealt"]),
                  global_best_taken=float(seed_eval["taken"]),
                  global_best_margin=seed_margin,
                  global_best_min_margin=float(seed_eval.get("bench_min_margin", seed_margin)),
                  global_best_min_judge=float(seed_eval.get("bench_min_judge", seed_eval.get("bench_judge", 0.0))),
                  accept_metric=args.accept_metric, stale_rounds=0,
                  seed_eval=seed_eval, cum_step=0)
        save_state(st)
    if "global_best_margin" not in st:
        seed_eval = st.get("seed_eval", {})
        st["global_best_dealt"] = float(seed_eval.get("dealt", 0.0))
        st["global_best_taken"] = float(seed_eval.get("taken", 0.0))
        st["global_best_margin"] = float(seed_eval.get("bench_margin",
                                                       st["global_best_dealt"] - st["global_best_taken"]))
        st["global_best_min_margin"] = float(seed_eval.get("bench_min_margin", st["global_best_margin"]))
        st["global_best_min_judge"] = float(seed_eval.get("bench_min_judge", st.get("global_best_bench", 0.0)))
    print(f"LEAGUE start: round={st['round']} HoF={len(st['hof'])} best_bench={st['global_best_bench']:.2f} "
          f"cum_step={st['cum_step']:,}", flush=True)

    for rd in range(st["round"], args.rounds):
        # training opponent: sampled from the FIRST QUARTER of the HoF (stable); benchmark: the SEED
        pool = first_quarter(st["hof"])
        train_opp = pool[int(rng.integers(len(pool)))]
        warm = st["global_best_ckpt"]
        out = run_round(py, rd, train_opp, st["seed"], warm, args.round_steps,
                        st["cum_step"], args.lean_contacts, args.tiny, args.envs, args.batch,
                        args.frame_skip, args.bench_epis, args.bench_steps, args.accept_metric,
                        args.early_hit_penalty, args.min_hit_step,
                        args.min_accepted_dealt, args.max_accepted_early_dmg,
                        args.taken_weight, args.bench_seeds)
        if out is None:
            save_state(st); print("LEAGUE STOP: round failed.", flush=True); return
        res, round_best = out
        st["cum_step"] = res["cum_step"]
        delta = delta_diagnostic(py, f"spr{rd}", warm, round_best, st.get("delta_refs", []))
        st.setdefault("skill_deltas", []).append(dict(round=rd, warm=warm, ckpt=round_best, **delta))
        robust = None
        if args.robust_promotion:
            robust = run_robust_promotion(
                py, f"spr{rd}", st["global_best_ckpt"], round_best, st["seed"], robust_slices,
                args.bench_seeds, args.bench_epis, args.frame_skip, args.lean_contacts,
                args.min_accepted_dealt, args.max_accepted_early_dmg,
                args.robust_margin_tol, args.robust_judge_tol,
                args.robust_min_mean_margin_delta, args.robust_min_mean_judge_delta)
            st.setdefault("robust_promotions", []).append(dict(round=rd, ckpt=round_best, **robust))
        bench = float(res["best_bench"])
        best_dealt = float(res.get("best_dealt", 0.0))
        best_taken = float(res.get("best_taken", 0.0))
        best_margin = float(res.get("best_margin", best_dealt - best_taken))
        best_min_margin = float(res.get("best_min_margin", best_margin))
        best_min_judge = float(res.get("best_min_judge", res.get("best_judge", bench)))
        best_min_dealt = float(res.get("best_min_dealt", best_dealt))
        best_early = float(res.get("best_max_early", res.get("best_ac_dmg_early", 0.0)))
        guard_margin = best_min_margin if args.accept_metric == "min_margin" else best_margin
        global_guard_margin = (float(st.get("global_best_min_margin", st["global_best_margin"]))
                               if args.accept_metric == "min_margin"
                               else float(st["global_best_margin"]))
        score_ok = bench >= float(st["global_best_bench"]) - args.tol
        damage_ok = best_min_dealt >= args.min_accepted_dealt
        margin_ok = guard_margin >= global_guard_margin - args.margin_tol
        early_ok = best_early <= args.max_accepted_early_dmg
        robust_ok = True if robust is None else bool(robust.get("accepted"))
        accepted = score_ok and damage_ok and margin_ok and early_ok and robust_ok
        if not accepted:
            reasons = []
            if not score_ok:
                reasons.append(f"score {bench:.3f} < {float(st['global_best_bench'])-args.tol:.3f}")
            if not damage_ok:
                reasons.append(f"min_dealt {best_min_dealt:.4f} < {args.min_accepted_dealt:.4f}")
            if not margin_ok:
                reasons.append(
                    f"margin_guard {guard_margin:+.4f} < {global_guard_margin-args.margin_tol:+.4f}"
                )
            if not early_ok:
                reasons.append(f"early_dmg {best_early:.3f} > {args.max_accepted_early_dmg:.3f}")
            if not robust_ok:
                reasons.append("independent robust selector rejected candidate")
            print(f"ROUND {rd} REJECT: {', '.join(reasons)}; "
                  f"do not add {Path(round_best).name} to HoF; keep global best "
                  f"{Path(st['global_best_ckpt']).name}", flush=True)
            save_state(st)
            print("LEAGUE STOP: strict self-play gate rejected a regressed round.", flush=True)
            return
        improved = bench > float(st["global_best_bench"])
        if improved:
            st["global_best_bench"] = bench; st["global_best_ckpt"] = round_best
            st["global_best_sparc"] = float(res.get("best_sparc", st.get("global_best_sparc", 0.0)))
            st["global_best_dealt"] = best_dealt
            st["global_best_taken"] = best_taken
            st["global_best_margin"] = best_margin
            st["global_best_min_margin"] = best_min_margin
            st["global_best_min_judge"] = best_min_judge
            st["hof"].append(round_best)
            if delta.get("ok"):
                st.setdefault("delta_refs", []).append(delta["npz"])
            st["stale_rounds"] = 0
        elif args.archive_non_improvements:
            st["hof"].append(round_best)
            if delta.get("ok"):
                st.setdefault("delta_refs", []).append(delta["npz"])
            st["stale_rounds"] = int(st.get("stale_rounds", 0)) + 1
        else:
            st["stale_rounds"] = int(st.get("stale_rounds", 0)) + 1
        print(f"ROUND {rd} ACCEPT: bench {bench:.3f} "
              f"(global best {float(st['global_best_bench']):.3f})"
              f"{'  *NEW-BEST*' if improved else '  (not archived; no improvement)'}", flush=True)
        st["round"] = rd + 1; save_state(st)
        if st["global_best_ckpt"] and os.path.exists(st["global_best_ckpt"]):
            shutil.copy(st["global_best_ckpt"], OUT / "selfplay_best.pkl")
        if not improved and args.max_stale_rounds >= 0 and int(st.get("stale_rounds", 0)) >= args.max_stale_rounds:
            print(f"LEAGUE STOP: {st['stale_rounds']} stale rounds without {args.accept_metric} improvement.", flush=True)
            return

    print(f"\nLEAGUE DONE: rounds={st['round']} HoF={len(st['hof'])} "
          f"global_best_bench={st['global_best_bench']:.2f} cum_step={st['cum_step']:,} -> selfplay_best.pkl", flush=True)
    print("METRIC " + " ".join(f"{k}={v}" for k, v in dict(
        stage="selfplay_drive", rounds=st["round"], hof=len(st["hof"]),
        global_best_bench=f"{st['global_best_bench']:.2f}", cum_step=st["cum_step"]).items()), flush=True)


if __name__ == "__main__":
    main()
