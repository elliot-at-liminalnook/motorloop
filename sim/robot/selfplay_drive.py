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

# self-play reward weights (win the exchange vs a fighting opponent) + spawn spread (medium range,
# where tactics matter). Firing-shaping low (the striker skill is already learned by the seed).
RW = dict(sep_lo=0.4, sep_hi=1.2, approach=1.0, azimuth=3.14159, shaping=0.3,
          clean=5.0, trade=3.0, disengage=1.0, fire=0.3)
BENCH = dict(sep_lo=0.4, sep_hi=1.2, az=3.14159, epis=16, steps=200)


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


def run_round(py, rd, train_opp, bench_opp, warm, steps, cum_base, lean, tiny, envs, batch):
    tag = f"spr{rd}"
    log = OUT / f"sp_{tag}.log"
    cmd = [py, "-u", str(HERE / "train_adversarial.py"),
           "--tag", tag, "--steps", str(steps), "--cum-base", str(cum_base),
           "--opponent", "frozen", "--opp-ckpt", str(train_opp), "--bench-opp-ckpt", str(bench_opp),
           "--resume", str(warm),
           "--sep-lo", str(RW["sep_lo"]), "--sep-hi", str(RW["sep_hi"]),
           "--approach-weight", str(RW["approach"]), "--azimuth", str(RW["azimuth"]),
           "--shaping", str(RW["shaping"]), "--clean-weight", str(RW["clean"]),
           "--trade-weight", str(RW["trade"]), "--disengage-weight", str(RW["disengage"]),
           "--fire-shaping", str(RW["fire"]),
           "--bench-sep-lo", str(BENCH["sep_lo"]), "--bench-sep-hi", str(BENCH["sep_hi"]),
           "--bench-az", str(BENCH["az"]), "--bench-epis", str(BENCH["epis"]),
           "--bench-steps", str(BENCH["steps"])]
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
    if rc != 0 or not sf.exists():
        print(f"ROUND {rd} FAILED rc={rc} (see {log})", flush=True); return None
    stt = json.loads(sf.read_text())
    print(f"ROUND {rd} done in {time.time()-t0:.0f}s: best_bench={stt['best_bench']} "
          f"last_ratio={stt.get('last_ratio')} cum_step={stt['cum_step']}", flush=True)
    return stt, str(OUT / f"{tag}_best.pkl")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed-ckpt", default=str(OUT / "curriculum_best.pkl"),
                    help="the skill-curriculum fighter that seeds the league (HoF[0] + fixed benchmark ref)")
    ap.add_argument("--rounds", type=int, default=12)
    ap.add_argument("--round-steps", type=int, default=10_000_000)
    ap.add_argument("--tol", type=float, default=3.0, help="benchmark regression tolerance for keep-best gate")
    ap.add_argument("--envs", type=int, default=0)
    ap.add_argument("--batch", type=int, default=0)
    ap.add_argument("--lean-contacts", action="store_true")
    ap.add_argument("--resume", action="store_true")
    ap.add_argument("--tiny", action="store_true")
    args = ap.parse_args()
    py = sys.executable
    if args.tiny:
        args.rounds, args.round_steps = 2, 8_000

    import numpy as np
    rng = np.random.default_rng(0)
    st = load_state() if args.resume else None
    if st is None:
        if not os.path.exists(args.seed_ckpt):
            print(f"SEED MISSING: {args.seed_ckpt} (run the skill curriculum first)"); return
        seed = str(OUT / "selfplay_seed.pkl"); shutil.copy(args.seed_ckpt, seed)
        st = dict(round=0, hof=[seed], seed=seed, global_best_ckpt=seed,
                  global_best_bench=-1e30, cum_step=0)
        save_state(st)
    print(f"LEAGUE start: round={st['round']} HoF={len(st['hof'])} best_bench={st['global_best_bench']:.2f} "
          f"cum_step={st['cum_step']:,}", flush=True)

    for rd in range(st["round"], args.rounds):
        # training opponent: sampled from the FIRST QUARTER of the HoF (stable); benchmark: the SEED
        pool = first_quarter(st["hof"])
        train_opp = pool[int(rng.integers(len(pool)))]
        out = run_round(py, rd, train_opp, st["seed"], st["global_best_ckpt"], args.round_steps,
                        st["cum_step"], args.lean_contacts, args.tiny, args.envs, args.batch)
        if out is None:
            save_state(st); print("LEAGUE STOP: round failed.", flush=True); return
        res, round_best = out
        st["cum_step"] = res["cum_step"]
        st["hof"].append(round_best)                       # archive this round's best as an opponent
        if res["best_bench"] >= st["global_best_bench"] - args.tol:
            if res["best_bench"] > st["global_best_bench"]:
                st["global_best_bench"] = res["best_bench"]; st["global_best_ckpt"] = round_best
            print(f"ROUND {rd} ACCEPT: bench {res['best_bench']:.2f} (global best {st['global_best_bench']:.2f})", flush=True)
        else:
            print(f"ROUND {rd} regressed (bench {res['best_bench']:.2f} < {st['global_best_bench']-args.tol:.2f}); "
                  f"keep global best {Path(st['global_best_ckpt']).name}", flush=True)
        st["round"] = rd + 1; save_state(st)
        if st["global_best_ckpt"] and os.path.exists(st["global_best_ckpt"]):
            shutil.copy(st["global_best_ckpt"], OUT / "selfplay_best.pkl")

    print(f"\nLEAGUE DONE: rounds={st['round']} HoF={len(st['hof'])} "
          f"global_best_bench={st['global_best_bench']:.2f} cum_step={st['cum_step']:,} -> selfplay_best.pkl", flush=True)
    print("METRIC " + " ".join(f"{k}={v}" for k, v in dict(
        stage="selfplay_drive", rounds=st["round"], hof=len(st["hof"]),
        global_best_bench=f"{st['global_best_bench']:.2f}", cum_step=st["cum_step"]).items()), flush=True)


if __name__ == "__main__":
    main()
