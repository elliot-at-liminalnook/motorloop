# SPDX-License-Identifier: MIT
"""Validate the locomotion->fighting co-design boundary (design-review feedback).

The walker-policy co-design (Station 3) ranks bodies by how *easy to drive* they are,
which is NOT the battlebot goal (winning SPARC matches). This script answers the
empirical question: does a body's WALKER fitness predict its FIGHTER (SPARC match)
fitness? Verdict rule:
  high rank-correlation -> walker co-design is a valid prefilter for combat bodies.
  low/none             -> walker co-design is ONLY a prefilter/warm-start; FIGHT
                          return must take over design ranking.

Two phases (run at the walker->fighter boundary, then after fighter training):
  --build-pack    after walker: eval pack [default + randoms + walker-best], walker fit
  --score-fighter after fighter: add fighter (SPARC) fit, report the correlation + verdict
"""

from __future__ import annotations

import argparse, os, pickle, sys, time
from pathlib import Path
import jax, jax.numpy as jnp, numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from gen_robot_mjcf import build_mjcf, load_spec       # noqa: E402
from mjx_env import UniversalEnv, DESIGN_DIM           # noqa: E402
from train_adversarial import AdversarialEnv           # noqa: E402
from brax.training.agents.ppo import networks as ppo_networks   # noqa: E402
from brax.training.acme import running_statistics      # noqa: E402

OUT = Path(os.environ.get("CODESIGN_OUT", "/root/proj/out")); SPEC = load_spec(HERE / "robot.toml")
PACK = OUT / "eval_pack.pkl"
EPIS, STEPS, N_RAND = 64, 250, 10        # overridden by --tiny / CLI


def METRIC(**kw):
    print("METRIC " + " ".join(f"{k}={v}" for k, v in kw.items()), flush=True)


def load_policy(path, obs, act):
    params = pickle.load(open(path, "rb"))
    net = ppo_networks.make_ppo_networks(obs, act, preprocess_observations_fn=running_statistics.normalize)
    return ppo_networks.make_inference_fn(net)(params, deterministic=True)


def make_ret(env, inf, steps=None, episodes=None):
    steps = steps or STEPS; episodes = episodes or EPIS
    @jax.jit
    def ret(rng, design):
        def ep(k):
            st = env.reset_with(k, design)
            def stp(c, _):
                s, key = c; key, a = jax.random.split(key); act, _ = inf(s.obs, a)
                s = env.step(s, act); return (s, key), s.reward * (1.0 - s.done)
            (_, _), rs = jax.lax.scan(stp, (st, rng), None, length=steps)
            return rs.sum()
        return jax.vmap(ep)(jax.random.split(rng, episodes)).mean()
    return lambda d: float(ret(jax.random.PRNGKey(0), jnp.asarray(d)))


def build_pack():
    t0 = time.time()
    env = UniversalEnv(build_mjcf(SPEC))
    walker = load_policy(OUT / "universal_ckpt.pkl", env.observation_size, env.action_size)
    wret = make_ret(env, walker)
    t_c = time.time(); _ = wret(np.full(DESIGN_DIM, 0.5)); compile_s = time.time() - t_c  # eval-rollout compile
    rng = np.random.default_rng(0)
    designs = [np.full(DESIGN_DIM, 0.5)] + [rng.uniform(0, 1, DESIGN_DIM) for _ in range(N_RAND)]
    wfit = [wret(d) for d in designs]
    best = designs[int(np.argmax(wfit))]                 # the "best walker-selected body"
    designs.append(best); wfit.append(wret(best))
    labels = ["default"] + [f"rand{i}" for i in range(N_RAND)] + ["walker-best"]
    pickle.dump({"designs": designs, "walker_fit": wfit, "labels": labels}, open(PACK, "wb"))
    METRIC(stage="build_pack", t_s=f"{time.time()-t0:.1f}", compile_s=f"{compile_s:.1f}",
           n_bodies=len(designs), wfit_lo=f"{min(wfit):.1f}", wfit_hi=f"{max(wfit):.1f}",
           epis=EPIS, steps=STEPS)
    print(f"[build-pack] {len(designs)} bodies; walker_fit range {min(wfit):.1f}..{max(wfit):.1f}; "
          f"walker-best fit {wfit[-1]:.1f}. saved {PACK}", flush=True)


def score_fighter():
    t0 = time.time()
    p = pickle.load(open(PACK, "rb"))
    env = AdversarialEnv()
    fighter = load_policy(OUT / "adv_ckpt.pkl", env.observation_size, env.action_size)
    fret = make_ret(env, fighter)
    t_c = time.time(); _ = fret(p["designs"][0]); compile_s = time.time() - t_c
    ffit = [fret(d) for d in p["designs"]]
    w = np.array(p["walker_fit"]); f = np.array(ffit)
    rk = lambda x: np.argsort(np.argsort(x))
    rho = float(np.corrcoef(rk(w), rk(f))[0, 1])
    print("\n  body          walker_fit  fighter_fit")
    for lab, wi, fi in zip(p["labels"], w, f):
        print(f"  {lab:12s} {wi:10.1f} {fi:11.2f}")
    wb = p["labels"].index("walker-best")
    wb_rank = int((f > f[wb]).sum())                     # 0 = walker-best is the top fighter
    verdict = ("walker co-design PREDICTS combat fitness -> valid prefilter"
               if rho >= 0.5 else
               "walker fitness does NOT predict combat -> walker = prefilter/warm-start ONLY; "
               "FIGHT-return must rank designs")
    print(f"\n[score-fighter] Spearman(walker_fit, fighter_fit) = {rho:+.2f} over {len(f)} bodies. "
          f"walker-best is fighter-rank #{wb_rank+1}/{len(f)}.")
    print(f"VERDICT: {verdict}.", flush=True)
    METRIC(stage="score_fighter", t_s=f"{time.time()-t0:.1f}", compile_s=f"{compile_s:.1f}",
           n_bodies=len(f), rho=f"{rho:+.3f}", walker_best_rank=wb_rank + 1)
    pickle.dump({**p, "fighter_fit": ffit, "rho": rho}, open(OUT / "validation.pkl", "wb"))


def main():
    global EPIS, STEPS, N_RAND
    ap = argparse.ArgumentParser()
    ap.add_argument("--build-pack", action="store_true")
    ap.add_argument("--score-fighter", action="store_true")
    ap.add_argument("--epis", type=int, default=0)
    ap.add_argument("--steps", type=int, default=0)
    ap.add_argument("--n-rand", type=int, default=0)
    ap.add_argument("--tiny", action="store_true", help="lightweight plumbing run (e2e harness)")
    a = ap.parse_args()
    if a.tiny: EPIS, STEPS, N_RAND = 8, 60, 4
    if a.epis: EPIS = a.epis
    if a.steps: STEPS = a.steps
    if a.n_rand: N_RAND = a.n_rand
    if a.build_pack: build_pack()
    elif a.score_fighter: score_fighter()


if __name__ == "__main__":
    main()
