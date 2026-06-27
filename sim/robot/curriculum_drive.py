# SPDX-License-Identifier: MIT
"""STEP 2 curriculum DRIVER — gate + rollback + keep-best chain, resume-safe.

NOTE: superseded by the `arena` framework (`python -m arena.cli curriculum ...`, which reuses this
module's PHASES/BENCH as its config source). This standalone driver is kept working + as that source.

Turns a sequence of training phases into a run whose HONEST held-out benchmark only goes UP — or
we roll back. The mechanism that makes "improves with more training time" hold by construction:

  * each phase WARM-STARTS from the best checkpoint so far (the chain never starts from worse);
  * `train_adversarial` keeps the per-phase BEST by the fixed held-out benchmark (`{tag}_best.pkl`),
    so a phase's best ≥ its warm-start point (≈ the global best) up to eval noise;
  * after a phase we GATE on its benchmark (read from `{tag}_state.json`): accept only if it did not
    regress beyond tolerance; otherwise ROLLBACK (keep the previous best) and retry the phase with
    gentler widening (narrower `sep_hi`), up to `--retries`, else stop with an honest report.

Resume-safe: progress is persisted to `curriculum_state.json`, so a SHORT run continues into a
LONGER one — re-invoke with `--resume` to pick up at the next phase, or `--extend N` to train the
FINAL phase longer (warm-started from the global best). This is the "save a short run and continue"
requirement.

  python curriculum_drive.py --warm out/universal_ckpt.pkl --steps-per-phase 4000000 --lean-contacts
  python curriculum_drive.py --resume                 # continue where it left off
  python curriculum_drive.py --extend 8000000         # train the final phase longer, from its best
  python curriculum_drive.py --tiny                   # fast CPU plumbing test (no GPU)
"""

from __future__ import annotations

import argparse, json, os, shutil, subprocess, sys, time
from pathlib import Path

HERE = Path(__file__).resolve().parent
OUT = Path(os.environ.get("CODESIGN_OUT", "/root/proj/out")); OUT.mkdir(parents=True, exist_ok=True)
STATE = OUT / "curriculum_state.json"

# Spawn-curriculum phases (the contact-forcing reverse curriculum + win-reward asymmetry that
# ramps in as separation widens). sep widens; approach/shaping/FIRE-shaping anneal DOWN; clean-
# hit/trade/disengage asymmetry ramps UP (early phases learn to land hits + FIRE THE ROD in
# range; later phases learn to win exchanges without trading). `fire` = firing-shaping weight (the
# fix for the SPARSE firing reward — a close `strk0` phase manufactures the first successful fires,
# then it anneals as real rod hits take over). Phase 0 also RE-STABILIZES balance with the heavier
# rod body (warm-started from the 12-action locomotor via the action-head grow). Opponent is
# passive here; moving/attacking (D/E) + self-play (F) extend this once the scripted opponent/HoF
# are wired.
PHASES = [
    dict(name="strk0", sep_lo=0.20, sep_hi=0.35, approach=4.0, azimuth=0.3, shaping=1.2,
         clean=2.0, trade=0.5, disengage=0.0, fire=1.0),    # FORCE engagement: high approach (anti-flee),
                                                            # near-facing spawn, strong close→strike shaping
    dict(name="cval", sep_lo=0.30, sep_hi=0.45, approach=4.0, azimuth=0.6, shaping=1.0,
         clean=2.0, trade=1.0, disengage=0.0, fire=0.8),
    dict(name="c1", sep_lo=0.40, sep_hi=0.70, approach=3.5, azimuth=1.2, shaping=0.8,
         clean=3.0, trade=2.0, disengage=0.5, fire=0.6),
    dict(name="c2", sep_lo=0.40, sep_hi=1.00, approach=1.5, azimuth=2.0, shaping=0.6,
         clean=4.0, trade=3.0, disengage=1.0, fire=0.4),
    dict(name="c3", sep_lo=0.40, sep_hi=1.40, approach=1.0, azimuth=3.14159, shaping=0.4,
         clean=5.0, trade=3.0, disengage=1.0, fire=0.2),
]
# FIXED benchmark config (comparable across ALL phases — never changes). Range matched to the body's
# engagement envelope (max reach 0.62 m, gear-12 half-speed legs): the old 0.4-1.2 m measured a range
# this body can't close in 200 steps, so dealt read 0 even when it could strike up close. 0.25-0.7.
BENCH = dict(sep_lo=0.25, sep_hi=0.7, az=3.14159, epis=16, steps=200)


def load_state():
    if STATE.exists():
        return json.loads(STATE.read_text())
    return dict(completed=[], global_best_ckpt=None, global_best_bench=-1e30, cum_step=0)


def save_state(st):
    STATE.write_text(json.dumps(st, indent=2))


def run_phase(py, ph, warm, steps, cum_base, lean, tiny, tag_suffix="", envs=0, batch=0,
              keep_metric="win", min_keep_dealt=0.0, max_keep_early_dmg=1.0,
              flee_penalty=0.0, close_bonus=0.0, close_radius=0.45, damage_bonus=0.0):
    """Run one training phase as a subprocess; return its {tag}_state.json dict (or None on failure)."""
    tag = ph["name"] + tag_suffix
    log = OUT / f"curr_{tag}.log"
    cmd = [py, "-u", str(HERE / "train_adversarial.py"),
           "--tag", tag, "--steps", str(steps), "--cum-base", str(cum_base),]
    if envs: cmd += ["--envs", str(envs)]
    if batch: cmd += ["--batch", str(batch)]
    cmd += [
           "--sep-lo", str(ph["sep_lo"]), "--sep-hi", str(ph["sep_hi"]),
           "--approach-weight", str(ph["approach"]), "--azimuth", str(ph["azimuth"]),
           "--shaping", str(ph["shaping"]), "--clean-weight", str(ph["clean"]),
           "--trade-weight", str(ph["trade"]), "--disengage-weight", str(ph["disengage"]),
           "--fire-shaping", str(ph.get("fire", 0.0)),
           "--bench-sep-lo", str(BENCH["sep_lo"]), "--bench-sep-hi", str(BENCH["sep_hi"]),
           "--bench-az", str(BENCH["az"]), "--bench-epis", str(BENCH["epis"]),
           "--bench-steps", str(BENCH["steps"]),
           "--keep-metric", str(keep_metric),
           "--min-keep-dealt", str(min_keep_dealt),
           "--max-keep-early-dmg", str(max_keep_early_dmg),
           "--flee-penalty", str(ph.get("flee_penalty", flee_penalty)),
           "--close-bonus", str(ph.get("close_bonus", close_bonus)),
           "--close-radius", str(ph.get("close_radius", close_radius)),
           "--damage-bonus", str(ph.get("damage_bonus", damage_bonus))]
    if warm: cmd += ["--resume", str(warm)]
    if lean: cmd += ["--lean-contacts"]
    if tiny: cmd += ["--tiny"]
    print(f"\n=== PHASE {tag}: sep {ph['sep_lo']}-{ph['sep_hi']} steps {steps:,} "
          f"warm={Path(warm).name if warm else 'scratch'} ===", flush=True)
    t0 = time.time()
    with open(log, "w") as lf:
        rc = subprocess.run(cmd, stdout=lf, stderr=subprocess.STDOUT, env={**os.environ}).returncode
    sf = OUT / f"{tag}_state.json"
    if rc != 0 or not sf.exists():
        print(f"PHASE {tag} FAILED rc={rc} (see {log}); no state file.", flush=True)
        return None
    stt = json.loads(sf.read_text())
    print(f"PHASE {tag} done in {time.time()-t0:.0f}s: best_bench={stt['best_bench']} "
          f"last_ratio={stt.get('last_ratio')} cum_step={stt['cum_step']}", flush=True)
    return stt


def main():
    global PHASES
    ap = argparse.ArgumentParser()
    ap.add_argument("--warm", default=str(OUT / "universal_ckpt.pkl"), help="phase-0 warm-start (locomotor or fighter)")
    ap.add_argument("--steps-per-phase", type=int, default=4_000_000)
    ap.add_argument("--phases", type=int, default=0,
                    help="run only the first N curriculum phases (0=all phases)")
    ap.add_argument("--bench-sep-lo", type=float, default=BENCH["sep_lo"])
    ap.add_argument("--bench-sep-hi", type=float, default=BENCH["sep_hi"])
    ap.add_argument("--bench-az", type=float, default=BENCH["az"])
    ap.add_argument("--bench-epis", type=int, default=BENCH["epis"])
    ap.add_argument("--bench-steps", type=int, default=BENCH["steps"])
    ap.add_argument("--keep-metric", choices=["win", "sparc", "ratio", "margin", "judge",
                                              "min_margin", "min_judge"], default="win")
    ap.add_argument("--min-keep-dealt", type=float, default=0.0)
    ap.add_argument("--max-keep-early-dmg", type=float, default=1.0)
    ap.add_argument("--flee-penalty", type=float, default=float(os.environ.get("FLEE_PENALTY", "0")))
    ap.add_argument("--close-bonus", type=float, default=float(os.environ.get("CLOSE_BONUS", "0")))
    ap.add_argument("--close-radius", type=float, default=float(os.environ.get("CLOSE_RADIUS", "0.45")))
    ap.add_argument("--damage-bonus", type=float, default=float(os.environ.get("DAMAGE_BONUS", "0")))
    ap.add_argument("--envs", type=int, default=0, help="num_envs per phase (0=train_adversarial default; 8192 saturates an A100)")
    ap.add_argument("--batch", type=int, default=0, help="PPO batch_size per phase (0=default)")
    ap.add_argument("--tol", type=float, default=2.0, help="benchmark regression tolerance before rollback")
    ap.add_argument("--retries", type=int, default=1, help="rollback retries (gentler widening) per phase")
    ap.add_argument("--lean-contacts", action="store_true")
    ap.add_argument("--resume", action="store_true", help="continue from curriculum_state.json")
    ap.add_argument("--extend", type=int, default=0, help="train the FINAL phase this many more steps, from global best")
    ap.add_argument("--tiny", action="store_true")
    args = ap.parse_args()
    if args.phases > 0:
        PHASES = PHASES[:args.phases]
    BENCH.update(sep_lo=args.bench_sep_lo, sep_hi=args.bench_sep_hi, az=args.bench_az,
                 epis=args.bench_epis, steps=args.bench_steps)
    py = sys.executable
    if args.tiny:
        args.steps_per_phase = 8_000
        PHASES = PHASES[:2]
        BENCH.update(epis=min(BENCH["epis"], 4), steps=min(BENCH["steps"], 40))

    st = load_state() if (args.resume or args.extend) else dict(
        completed=[], global_best_ckpt=None, global_best_bench=-1e30, cum_step=0)
    if not st["completed"] and not args.extend:
        st["global_best_ckpt"] = args.warm if os.path.exists(args.warm) else None
    print(f"DRIVER start: completed={st['completed']} global_best_bench={st['global_best_bench']:.2f} "
          f"cum_step={st['cum_step']:,}", flush=True)

    # --- normal forward pass over remaining phases (gate + rollback) ---
    if not args.extend:
        for ph in PHASES:
            if ph["name"] in st["completed"]:
                continue
            sep_hi = ph["sep_hi"]
            for attempt in range(args.retries + 1):
                ph_try = {**ph, "sep_hi": sep_hi}
                res = run_phase(py, ph_try, st["global_best_ckpt"], args.steps_per_phase,
                                st["cum_step"], args.lean_contacts, args.tiny,
                                tag_suffix="" if attempt == 0 else f"_r{attempt}",
                                envs=args.envs, batch=args.batch,
                                keep_metric=args.keep_metric,
                                min_keep_dealt=args.min_keep_dealt,
                                max_keep_early_dmg=args.max_keep_early_dmg,
                                flee_penalty=args.flee_penalty,
                                close_bonus=args.close_bonus,
                                close_radius=args.close_radius,
                                damage_bonus=args.damage_bonus)
                if res is None:
                    save_state(st); print("DRIVER STOP: phase subprocess failed.", flush=True); return
                st["cum_step"] = res["cum_step"]
                bench = res["best_bench"]
                if bench >= st["global_best_bench"] - args.tol:        # GATE: accept
                    if bench > st["global_best_bench"]:
                        st["global_best_bench"] = bench
                    st["global_best_ckpt"] = str(OUT / f"{res['tag']}_best.pkl")
                    st["completed"].append(ph["name"]); save_state(st)
                    print(f"GATE PASS {ph['name']}: bench {bench:.2f} >= "
                          f"{st['global_best_bench']-args.tol:.2f} -> adopt {Path(st['global_best_ckpt']).name}",
                          flush=True)
                    break
                else:                                                  # ROLLBACK
                    sep_hi = ph["sep_lo"] + 0.5 * (sep_hi - ph["sep_lo"])
                    print(f"GATE FAIL {ph['name']} (bench {bench:.2f} < {st['global_best_bench']-args.tol:.2f}): "
                          f"ROLLBACK to {Path(str(st['global_best_ckpt'])).name}, retry sep_hi={sep_hi:.2f}",
                          flush=True)
                    save_state(st)
            else:
                print(f"DRIVER STOP: {ph['name']} kept regressing after {args.retries} retries "
                      f"(global best preserved at {st['global_best_bench']:.2f}).", flush=True)
                break

    # --- optional EXTEND: train the final phase longer from the global best (continue a short run) ---
    if args.extend:
        ph = {**PHASES[-1]}
        res = run_phase(py, ph, st["global_best_ckpt"], args.extend, st["cum_step"],
                        args.lean_contacts, args.tiny, tag_suffix="_ext",
                        envs=args.envs, batch=args.batch,
                        keep_metric=args.keep_metric,
                        min_keep_dealt=args.min_keep_dealt,
                        max_keep_early_dmg=args.max_keep_early_dmg,
                        flee_penalty=args.flee_penalty,
                        close_bonus=args.close_bonus,
                        close_radius=args.close_radius,
                        damage_bonus=args.damage_bonus)
        if res is not None:
            st["cum_step"] = res["cum_step"]
            if res["best_bench"] > st["global_best_bench"]:
                st["global_best_bench"] = res["best_bench"]
                st["global_best_ckpt"] = str(OUT / f"{res['tag']}_best.pkl")
            save_state(st)
            print(f"EXTEND done: bench {res['best_bench']:.2f}", flush=True)

    # --- finalize: publish the global best ---
    if st["global_best_ckpt"] and os.path.exists(st["global_best_ckpt"]):
        shutil.copy(st["global_best_ckpt"], OUT / "curriculum_best.pkl")
    print(f"\nDRIVER DONE: completed={st['completed']} global_best_bench={st['global_best_bench']:.2f} "
          f"cum_step={st['cum_step']:,} -> curriculum_best.pkl", flush=True)
    print("METRIC " + " ".join(f"{k}={v}" for k, v in dict(
        stage="curriculum_drive", phases=len(st["completed"]),
        global_best_bench=f"{st['global_best_bench']:.2f}", cum_step=st["cum_step"]).items()), flush=True)


if __name__ == "__main__":
    main()
