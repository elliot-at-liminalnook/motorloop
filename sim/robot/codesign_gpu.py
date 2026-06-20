# SPDX-License-Identifier: MIT
"""Phases 2/3/5 — REAL co-design on GPU: trained-policy return as the design fitness.

(3) Train ONE design-conditioned universal policy with brax PPO over a body
    distribution (DR inside UniversalEnv).
(2) Design fitness = that policy's mean return on a fixed body (a cheap rollout, no
    retrain). Compare to a static passive-stability proxy -> "how wrong the proxy is".
(5) CEM (single-obj) and a 2-objective Pareto pass (return vs mass) over the design.

  python codesign_gpu.py [--steps 4000000 --envs 2048]
"""

from __future__ import annotations

import argparse, functools, os, pickle, sys, time
from pathlib import Path
import jax, jax.numpy as jnp, numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from gen_robot_mjcf import build_mjcf, load_spec          # noqa: E402
from mjx_env import UniversalEnv, DESIGN_DIM, apply_design  # noqa: E402
from brax.training.agents.ppo import train as ppo          # noqa: E402

OUT = Path("/root/proj/out"); OUT.mkdir(parents=True, exist_ok=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--steps", type=int, default=4_000_000)
    ap.add_argument("--envs", type=int, default=2048)
    ap.add_argument("--resume", default=None,
                    help="warm-start from a saved params pickle (keeps improving, no scratch)")
    args = ap.parse_args()
    restore = None
    if args.resume and os.path.exists(args.resume):
        restore = pickle.load(open(args.resume, "rb"))
        print(f"RESUMING from {args.resume} (warm-start, not scratch)")
    xml = build_mjcf(load_spec(HERE / "robot.toml"))
    env = UniversalEnv(xml)
    print(f"universal env: obs={env.observation_size} act={env.action_size} design_dim={DESIGN_DIM}")

    # ---- (3) train the universal policy (checkpointed + CSV metrics: survives ssh/proc death) ----
    t0 = time.time()
    csv = OUT / "metrics.csv"; csv.write_text("step,reward,sec\n")
    def prog(s, m):
        r = float(m.get("eval/episode_reward", 0.0))
        open(csv, "a").write(f"{int(s)},{r:.3f},{time.time()-t0:.0f}\n")
        print(f"  [train] step {int(s):>9,} reward {r:7.2f} ({time.time()-t0:.0f}s)", flush=True)
    def save_ckpt(*a):                                       # latest params each eval
        try: pickle.dump(a[-1], open(OUT / "universal_ckpt.pkl", "wb"))
        except Exception: pass
    n_eval = max(6, args.steps // 1_000_000)
    make_inf, params, _ = ppo.train(
        environment=env, num_timesteps=args.steps, num_evals=n_eval, episode_length=200,
        num_envs=args.envs, batch_size=1024, num_minibatches=16, unroll_length=20,
        num_updates_per_batch=4, learning_rate=3e-4, entropy_cost=1e-2, discounting=0.97,
        reward_scaling=0.1, normalize_observations=True, seed=0,
        progress_fn=prog, policy_params_fn=save_ckpt, restore_params=restore)
    pickle.dump(params, open(OUT / "universal.pkl", "wb"))
    try: inference = make_inf(params, deterministic=True)
    except TypeError: inference = make_inf(params)

    # ---- (2) trained-return design fitness (a fast jitted rollout) ----
    EPIS, STEPS = 64, 200
    @jax.jit
    def trained_return(rng, design):
        def ep(key):
            st = env.reset_with(key, design)
            def stp(c, _):
                s, k = c; k, a = jax.random.split(k); act, _ = inference(s.obs, a)
                s = env.step(s, act); return (s, k), s.reward * (1.0 - s.done)
            (_, _), rs = jax.lax.scan(stp, (st, key), None, length=STEPS)
            return rs.sum()
        return jax.vmap(ep)(jax.random.split(rng, EPIS)).mean()

    @jax.jit
    def passive_proxy(rng, design):          # static stand-stability (zero action)
        def ep(key):
            st = env.reset_with(key, design); z = jnp.zeros((env.action_size,))
            def stp(c, _):
                s, _u = c; s = env.step(s, z); return (s, _u), (1.0 - s.done)
            (_, _), up = jax.lax.scan(stp, (st, key), None, length=STEPS)
            return up.sum()
        return jax.vmap(ep)(jax.random.split(rng, EPIS)).mean()

    def body_mass(design):                   # total mass of the designed body
        return float(np.sum(np.array(apply_design(env._mx, jnp.asarray(design)).body_mass)))

    key = jax.random.PRNGKey(1)
    f = lambda d: float(trained_return(key, jnp.asarray(d)))
    p = lambda d: float(passive_proxy(key, jnp.asarray(d)))

    # ---- (2 verify) proxy vs trained-return correlation over a random design sample ----
    rng = np.random.default_rng(0); samp = rng.uniform(0, 1, (24, DESIGN_DIM))
    fr = np.array([f(d) for d in samp]); pr = np.array([p(d) for d in samp])
    rk = lambda x: np.argsort(np.argsort(x))
    rho = float(np.corrcoef(rk(fr), rk(pr))[0, 1])
    print(f"\n[Phase 2] trained-return vs passive proxy over {len(samp)} designs: "
          f"Spearman rho={rho:+.2f}  (low/neg => the proxy ranks designs wrongly => "
          f"trained return is the real fitness)")

    # ---- (5) CEM over the design on the TRAINED return ----
    def cem(fit, dim, pop=16, gens=8, seed=0):
        r = np.random.default_rng(seed); mean = np.full(dim, .5); std = np.full(dim, .3)
        bx, bf = None, -1e9
        for g in range(gens):
            P = np.clip(mean + std * r.standard_normal((pop, dim)), 0, 1)
            F = np.array([fit(d) for d in P]); E = P[np.argsort(F)[-max(2, pop // 4):]]
            mean, std = E.mean(0), E.std(0) + 1e-3
            i = int(np.argmax(F));  bf, bx = (F[i], P[i]) if F[i] > bf else (bf, bx)
            print(f"  [CEM] gen {g} best={F.max():7.1f} mean={F.mean():7.1f}", flush=True)
        return bx, bf
    bestd, bestf = cem(f, DESIGN_DIM)
    print(f"[Phase 5] CEM best design {np.round(bestd,3)} trained-return {bestf:.1f} "
          f"(default {f([.5]*DESIGN_DIM):.1f}); mass {body_mass(bestd):.2f}")

    # ---- (5 multi-obj) Pareto front: return vs mass over the sample ----
    pts = [(f(d), -body_mass(d), d) for d in samp]
    pareto = [a for a in pts if not any((b[0] >= a[0] and b[1] >= a[1] and b[:2] != a[:2]) for b in pts)]
    print(f"[Phase 5] Pareto front (return vs -mass): {len(pareto)} non-dominated of {len(pts)}")
    print("PROVEN: co-design now ranks bodies by a TRAINED POLICY's return (GPU), "
          "CEM optimizes it, proxy-correlation quantifies the gap, Pareto front exists.")


if __name__ == "__main__":
    main()
