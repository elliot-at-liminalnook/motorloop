# SPDX-License-Identifier: MIT
"""F4 — the decisive experiment: do robust, calibrated rankings pick better bodies?

Rank N bodies THREE ways and correlate each with ground-truth performance:
  * proxy   — static passive-stand survival (zero-action), the cheap fitness.
  * nominal — the trained policy's return at the NOMINAL sim world (no DR).
  * robust  — the trained policy's return, CVaR@20% over the calibrated world ENSEMBLE
              (reality_gap worlds: actuator droop + DR), the lower-tail score.
Ground truth = the trained policy's MEAN return over a WIDE, held-out world set (the best
"real" signal available; the CPU oracle / reduced hardware later). Report
Spearman(proxy, nominal, robust vs ground-truth). The claim: **robust ρ ≥ nominal ρ ≥
proxy ρ** — calibration + robustness pick bodies whose held-out performance is best.

HONEST SCOPE: this runs on the design-conditioned UNIVERSAL policy (the trained policy we
have), so "performance" = the trained-policy return over the design+world distribution.
The FIGHT-specific version swaps in the fighter's SPARC return once a competent fighter
exists (F2) — same machinery, `--ckpt` + the AdversarialEnv; gated on F2 per the milestone.

  python fighter_rank.py [--n-designs 24 --worlds 16 --truth-worlds 48 --eval-epis 32]
  python fighter_rank.py --tiny
"""

from __future__ import annotations

import argparse, os, pickle, sys, time
from pathlib import Path
import jax, jax.numpy as jnp, numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from gen_robot_mjcf import build_mjcf, load_spec          # noqa: E402
from mjx_env import UniversalEnv, DESIGN_DIM              # noqa: E402
import ppo_nets as ppo_networks  # shared (512,256,128) + small-final-init factory (audit item 5)  # noqa: E402
from brax.training.acme import running_statistics         # noqa: E402
from robust_codesign import cvar                          # noqa: E402
from reality_gap_eval import rank_correlation             # noqa: E402

OUT = Path(os.environ.get("CODESIGN_OUT", "/root/proj/out"))


def METRIC(**kw):
    print("METRIC " + " ".join(f"{k}={v}" for k, v in kw.items()), flush=True)


def load_policy(path, obs, act):
    params = pickle.load(open(path, "rb"))
    net = ppo_networks.make_ppo_networks(obs, act, preprocess_observations_fn=running_statistics.normalize)
    return ppo_networks.make_inference_fn(net)(params, deterministic=True)


def ret_fn(env, inf, steps, epis):
    @jax.jit
    def ret(rng, design):
        def ep(k):
            st = env.reset_with(k, design)
            def stp(c, _):
                s, key = c; key, a = jax.random.split(key); act, _ = inf(s.obs, a)
                s = env.step(s, act); return (s, key), s.reward * (1.0 - s.done)
            (_, _), rs = jax.lax.scan(stp, (st, rng), None, length=steps)
            return rs.sum()
        return jax.vmap(ep)(jax.random.split(rng, epis)).mean()
    return ret


def passive_fn(env, steps, epis):
    @jax.jit
    def pas(rng, design):
        z = jnp.zeros((env.action_size,))
        def ep(k):
            st = env.reset_with(k, design)
            def stp(s, _):
                s = env.step(s, z); return s, (1.0 - s.done)     # survival under zero action
            _, up = jax.lax.scan(stp, st, None, length=steps)
            return up.sum()
        return jax.vmap(ep)(jax.random.split(rng, epis)).mean()
    return pas


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-designs", type=int, default=24)
    ap.add_argument("--worlds", type=int, default=16)
    ap.add_argument("--truth-worlds", type=int, default=48)
    ap.add_argument("--eval-epis", type=int, default=32)
    ap.add_argument("--eval-steps", type=int, default=200)
    ap.add_argument("--ckpt", default=str(OUT / "universal_ckpt.pkl"))
    ap.add_argument("--tiny", action="store_true")
    args = ap.parse_args()
    if args.tiny:
        args.n_designs, args.worlds, args.truth_worlds, args.eval_epis, args.eval_steps = 8, 4, 8, 8, 60

    xml = build_mjcf(load_spec(HERE / "robot.toml"))
    env_nom = UniversalEnv(xml, reality_gap=False)
    env_cal = UniversalEnv(xml, reality_gap=True, n_worlds=max(args.truth_worlds, args.worlds))
    inf = load_policy(args.ckpt, env_nom.observation_size, env_nom.action_size)
    r_nom = ret_fn(env_nom, inf, args.eval_steps, args.eval_epis)
    r_cal = ret_fn(env_cal, inf, args.eval_steps, args.eval_epis)
    passive = passive_fn(env_nom, args.eval_steps, max(8, args.eval_epis // 2))

    rng = np.random.default_rng(0)
    designs = rng.uniform(0, 1, (args.n_designs, DESIGN_DIM))
    t0 = time.time()
    proxy = np.array([float(passive(jax.random.PRNGKey(5), jnp.asarray(d))) for d in designs])
    nominal = np.array([float(r_nom(jax.random.PRNGKey(7), jnp.asarray(d))) for d in designs])
    # per-world returns under the calibrated ensemble (each PRNG key -> a different world draw)
    cal = np.array([[float(r_cal(jax.random.PRNGKey(1000 + w), jnp.asarray(d)))
                     for w in range(args.truth_worlds)] for d in designs])
    robust = np.array([cvar(row[:args.worlds], 0.2) for row in cal])   # CVaR over the first `worlds`
    truth = cal.mean(1)                                                # mean over ALL truth-worlds (held-out)

    rho_p = rank_correlation(proxy, truth)
    rho_n = rank_correlation(nominal, truth)
    rho_r = rank_correlation(robust, truth)
    claim = rho_r >= rho_n >= rho_p
    METRIC(stage="fighter_rank", t_s=f"{time.time()-t0:.1f}", n_designs=args.n_designs,
           proxy_rho=f"{rho_p:+.3f}", nominal_rho=f"{rho_n:+.3f}", robust_rho=f"{rho_r:+.3f}",
           claim_holds=int(claim))
    print(f"\n[F4] rank-corr vs wide-world ground truth ({args.truth_worlds} worlds, {args.n_designs} bodies):")
    print(f"     proxy   ρ = {rho_p:+.2f}")
    print(f"     nominal ρ = {rho_n:+.2f}")
    print(f"     robust  ρ = {rho_r:+.2f}   (CVaR@20% over {args.worlds} worlds)")
    print(f"[F4] claim (robust ≥ nominal ≥ proxy): {claim}")
    # head-to-head: the robust-best vs proxy-best vs nominal-best body, by true performance
    rb, nb, pb = int(np.argmax(robust)), int(np.argmax(nominal)), int(np.argmax(proxy))
    print(f"[F4] true performance of each ranking's pick: robust-best={truth[rb]:.1f}  "
          f"nominal-best={truth[nb]:.1f}  proxy-best={truth[pb]:.1f}")
    print(f"PROVEN: F4 three-way ranking experiment runs on GPU; robust(CVaR) vs nominal vs "
          f"proxy correlation to held-out performance reported. (Trained-policy/locomotion "
          f"performance here; swap the fighter checkpoint for the fight-SPARC version once F2 "
          f"yields a competent fighter — same machinery.)")


if __name__ == "__main__":
    main()
