# SPDX-License-Identifier: MIT
"""Phases 2/3/5 — REAL co-design on GPU: trained-policy return as the design fitness.

(3) Train ONE design-conditioned universal policy with brax PPO over a body
    distribution (DR inside UniversalEnv).
(2) Design fitness = that policy's mean return on a fixed body (a cheap rollout, no
    retrain). Compare to a static passive-stability proxy -> "how wrong the proxy is".
(5) CEM (single-obj) and a 2-objective Pareto pass (return vs mass) over the design.

  python codesign_gpu.py [--steps 4000000 --envs 2048]
  python codesign_gpu.py --tiny        # EXTREMELY lightweight plumbing run (e2e harness)

Emits machine-readable `METRIC stage=... k=v ...` lines the e2e harness parses to
profile where wall-clock goes (compile vs train vs CEM vs Pareto).
"""

from __future__ import annotations

import argparse, os, pickle, sys, time
from pathlib import Path
import jax, jax.numpy as jnp, numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from gen_robot_mjcf import build_mjcf, load_spec          # noqa: E402
from mjx_env import UniversalEnv, DESIGN_DIM, apply_design  # noqa: E402
from brax.training.agents.ppo import train as ppo          # noqa: E402

OUT = Path(os.environ.get("CODESIGN_OUT", "/root/proj/out")); OUT.mkdir(parents=True, exist_ok=True)


def METRIC(**kw):
    print("METRIC " + " ".join(f"{k}={v}" for k, v in kw.items()), flush=True)


def policy_fitness_direct(overrides, K=200_000, envs=1024, eval_epis=64, eval_steps=200,
                          restore_path=None, seed=0):
    """Phase 2 (DIRECT, costly form): generate the body from a full-design override dict,
    warm-start a baseline policy, fine-tune it K steps ON THAT BODY, and return the mean
    SPARC/locomotion episode return. This is the honest #1 — a trained policy's return as
    the design fitness, one fine-tune per candidate — that #2 (the universal policy) later
    makes cheap. Imported by optimize_design.py behind `--fitness policy` (GPU only)."""
    from mjx_env import CodesignEnv  # noqa: E402
    xml = build_mjcf(load_spec(HERE / "robot.toml"), overrides)
    env = CodesignEnv(xml)
    restore = None
    if restore_path and os.path.exists(restore_path):
        restore = pickle.load(open(restore_path, "rb"))
    make_inf, params, _ = ppo.train(
        environment=env, num_timesteps=K, num_evals=1, episode_length=eval_steps,
        num_envs=envs, batch_size=min(512, envs), num_minibatches=8, unroll_length=10,
        num_updates_per_batch=4, learning_rate=3e-4, entropy_cost=1e-2, discounting=0.97,
        reward_scaling=0.1, normalize_observations=True, seed=seed, restore_params=restore)
    try: inf = make_inf(params, deterministic=True)
    except TypeError: inf = make_inf(params)

    @jax.jit
    def ret(rng):
        def ep(k):
            st = env.reset(k)
            def stp(c, _):
                s, key = c; key, a = jax.random.split(key); act, _ = inf(s.obs, a)
                s = env.step(s, act); return (s, key), s.reward * (1.0 - s.done)
            (_, _), rs = jax.lax.scan(stp, (st, k), None, length=eval_steps)
            return rs.sum()
        return jax.vmap(ep)(jax.random.split(rng, eval_epis)).mean()
    return float(ret(jax.random.PRNGKey(seed + 1)))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--steps", type=int, default=4_000_000)
    ap.add_argument("--envs", type=int, default=2048)
    ap.add_argument("--batch", type=int, default=1024)
    ap.add_argument("--minibatches", type=int, default=16)
    ap.add_argument("--unroll", type=int, default=20)
    ap.add_argument("--evals", type=int, default=0, help="0 => max(6, steps//1e6)")
    ap.add_argument("--cem-pop", type=int, default=16)
    ap.add_argument("--cem-gens", type=int, default=8)
    ap.add_argument("--n-designs", type=int, default=24)
    ap.add_argument("--eval-epis", type=int, default=64)
    ap.add_argument("--eval-steps", type=int, default=200)
    ap.add_argument("--tiny", action="store_true",
                    help="EXTREMELY lightweight plumbing run for the e2e harness")
    ap.add_argument("--resume", default=None,
                    help="warm-start from a saved params pickle (keeps improving, no scratch)")
    args = ap.parse_args()
    if args.tiny:                         # smallest config that still exercises every link
        args.steps, args.envs = 8_000, 256
        args.batch, args.minibatches, args.unroll = 256, 8, 5     # PPO floor = 256*8*5 = 10,240
        args.evals = 2
        args.cem_pop, args.cem_gens, args.n_designs = 4, 2, 6
        args.eval_epis, args.eval_steps = 8, 60
    n_eval = args.evals or max(6, args.steps // 1_000_000)

    restore = None
    if args.resume and os.path.exists(args.resume):
        restore = pickle.load(open(args.resume, "rb"))
        print(f"RESUMING from {args.resume} (warm-start, not scratch)")
    xml = build_mjcf(load_spec(HERE / "robot.toml"))
    t_env = time.time()
    env = UniversalEnv(xml)
    METRIC(stage="env_build", t_s=f"{time.time()-t_env:.1f}",
           obs=env.observation_size, act=env.action_size, design_dim=DESIGN_DIM)
    print(f"universal env: obs={env.observation_size} act={env.action_size} design_dim={DESIGN_DIM}")

    # ---- (3) train the universal policy (checkpointed + CSV metrics: survives ssh/proc death) ----
    t0 = time.time()
    tm = {"first_eval": None}
    csv = OUT / "metrics.csv"; csv.write_text("step,reward,sec\n")
    def prog(s, m):
        if tm["first_eval"] is None:                         # ~compile + first iteration
            tm["first_eval"] = time.time() - t0
        r = float(m.get("eval/episode_reward", 0.0))
        open(csv, "a").write(f"{int(s)},{r:.3f},{time.time()-t0:.0f}\n")
        print(f"  [train] step {int(s):>9,} reward {r:7.2f} ({time.time()-t0:.0f}s)", flush=True)
    def save_ckpt(*a):                                        # latest params each eval
        try: pickle.dump(a[-1], open(OUT / "universal_ckpt.pkl", "wb"))
        except Exception: pass
    last_reward = {"r": float("nan"), "step": 0}
    def prog2(s, m):
        prog(s, m); last_reward["r"] = float(m.get("eval/episode_reward", 0.0)); last_reward["step"] = int(s)
    make_inf, params, _ = ppo.train(
        environment=env, num_timesteps=args.steps, num_evals=n_eval, episode_length=200,
        num_envs=args.envs, batch_size=args.batch, num_minibatches=args.minibatches,
        unroll_length=args.unroll, num_updates_per_batch=4, learning_rate=3e-4,
        entropy_cost=1e-2, discounting=0.97, reward_scaling=0.1, normalize_observations=True,
        seed=0, progress_fn=prog2, policy_params_fn=save_ckpt, restore_params=restore)
    train_s = time.time() - t0
    pickle.dump(params, open(OUT / "universal.pkl", "wb"))
    try: inference = make_inf(params, deterministic=True)
    except TypeError: inference = make_inf(params)
    METRIC(stage="walker_train", train_s=f"{train_s:.1f}", compile_s=f"{tm['first_eval'] or 0:.1f}",
           env_steps=last_reward["step"], throughput=f"{last_reward['step']/max(train_s,1e-6):.0f}",
           final_reward=f"{last_reward['r']:.2f}", evals=n_eval)

    # ---- (2) trained-return design fitness (a fast jitted rollout) ----
    EPIS, STEPS = args.eval_epis, args.eval_steps
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
    # time the FIRST trained_return call separately = the eval-rollout compile cost
    t_c = time.time(); _ = float(trained_return(key, jnp.full(DESIGN_DIM, 0.5))); eval_compile_s = time.time() - t_c
    f = lambda d: float(trained_return(key, jnp.asarray(d)))
    p = lambda d: float(passive_proxy(key, jnp.asarray(d)))
    METRIC(stage="eval_compile", t_s=f"{eval_compile_s:.1f}", epis=EPIS, steps=STEPS)

    # ---- (2 verify) proxy vs trained-return correlation over a random design sample ----
    t_p2 = time.time()
    rng = np.random.default_rng(0); samp = rng.uniform(0, 1, (args.n_designs, DESIGN_DIM))
    fr = np.array([f(d) for d in samp]); pr = np.array([p(d) for d in samp])
    rk = lambda x: np.argsort(np.argsort(x))
    rho = float(np.corrcoef(rk(fr), rk(pr))[0, 1])
    p2_s = time.time() - t_p2
    print(f"\n[Phase 2] trained-return vs passive proxy over {len(samp)} designs: "
          f"Spearman rho={rho:+.2f}  (low/neg => the proxy ranks designs wrongly => "
          f"trained return is the real fitness)")
    METRIC(stage="phase2_corr", t_s=f"{p2_s:.1f}", n_designs=len(samp), rho=f"{rho:+.3f}")

    # ---- (5) CEM over the design on the TRAINED return ----
    t_cem = time.time()
    def cem(fit, dim, pop, gens, seed=0):
        r = np.random.default_rng(seed); mean = np.full(dim, .5); std = np.full(dim, .3)
        bx, bf = None, -1e9
        for g in range(gens):
            P = np.clip(mean + std * r.standard_normal((pop, dim)), 0, 1)
            F = np.array([fit(d) for d in P]); E = P[np.argsort(F)[-max(2, pop // 4):]]
            mean, std = E.mean(0), E.std(0) + 1e-3
            i = int(np.argmax(F));  bf, bx = (F[i], P[i]) if F[i] > bf else (bf, bx)
            print(f"  [CEM] gen {g} best={F.max():7.1f} mean={F.mean():7.1f}", flush=True)
        return bx, bf
    bestd, bestf = cem(f, DESIGN_DIM, args.cem_pop, args.cem_gens)
    cem_s = time.time() - t_cem; default_f = f([.5] * DESIGN_DIM)
    print(f"[Phase 5] CEM best design {np.round(bestd,3)} trained-return {bestf:.1f} "
          f"(default {default_f:.1f}); mass {body_mass(bestd):.2f}")
    METRIC(stage="cem", t_s=f"{cem_s:.1f}", pop=args.cem_pop, gens=args.cem_gens,
           best=f"{bestf:.1f}", default=f"{default_f:.1f}", evals=args.cem_pop * args.cem_gens)

    # ---- (5 multi-obj) Pareto front: return vs mass over the sample ----
    t_pf = time.time()
    pts = [(f(d), -body_mass(d), d) for d in samp]
    pareto = [a for a in pts if not any((b[0] >= a[0] and b[1] >= a[1] and b[:2] != a[:2]) for b in pts)]
    METRIC(stage="pareto", t_s=f"{time.time()-t_pf:.1f}", n=len(pareto), of=len(pts))
    print(f"[Phase 5] Pareto front (return vs -mass): {len(pareto)} non-dominated of {len(pts)}")
    print("PROVEN: co-design now ranks bodies by a TRAINED POLICY's return (GPU), "
          "CEM optimizes it, proxy-correlation quantifies the gap, Pareto front exists.")


if __name__ == "__main__":
    main()
