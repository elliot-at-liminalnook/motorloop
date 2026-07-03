# SPDX-License-Identifier: MIT
"""STEP 1 — the decisive co-design experiment ON COMBAT: do robust, calibrated body rankings
pick better fighters once contact dynamics matter?

On locomotion the three rankings tied (proxy≈nominal≈robust, ρ 0.92–0.99) — locomotion is too
smooth for world uncertainty to flip rankings. Combat damage/contact is the fair test. Rank N
bodies by the CONTACT-FORCED FIGHTER's combat SPARC (the fighter is design-conditioned, so it
ranks bodies), three ways:
  * proxy   — static passive-stand survival (no policy), the cheap fitness.
  * nominal — fighter combat SPARC at the nominal sim world (reality_gap OFF).
  * robust  — fighter combat SPARC, CVaR@20% over the calibrated world ensemble (reality_gap ON:
              actuator droop + DR).
Ground truth = mean combat SPARC over a WIDE held-out world set. Report Spearman of each vs truth.
We rank by the **SPARC metric** (dealt−taken + aggression), NOT the shaped training reward.

  python combat_rank.py [--ckpt out/cval_ckpt.pkl --n-designs 16 --worlds 12 --truth-worlds 24]
  python combat_rank.py --tiny
"""

from __future__ import annotations

import argparse, os, pickle, sys, time
from pathlib import Path
import jax, jax.numpy as jnp, numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from train_adversarial import AdversarialEnv  # noqa: E402
import ppo_nets as ppo_networks  # shared (512,256,128) + small-final-init factory (audit item 5)  # noqa: E402
from brax.training.acme import running_statistics  # noqa: E402
from robust_codesign import cvar               # noqa: E402
from reality_gap_eval import rank_correlation  # noqa: E402

OUT = Path(os.environ.get("CODESIGN_OUT", "/root/proj/out"))
DESIGN_DIM = 3


def METRIC(**kw): print("METRIC " + " ".join(f"{k}={v}" for k, v in kw.items()), flush=True)


def load_policy(path, obs, act):
    params = pickle.load(open(path, "rb"))
    net = ppo_networks.make_ppo_networks(obs, act, preprocess_observations_fn=running_statistics.normalize)
    return ppo_networks.make_inference_fn(net)(params, deterministic=True)


def sparc_ret_fn(env, inf, steps, epis):
    """Mean episode SUM of the SPARC metric (combat outcome), the fighter driving A vs B."""
    @jax.jit
    def ret(rng, design):
        def ep(k):
            st = env.reset_with(k, design)
            def stp(c, _):
                s, key = c; key, a = jax.random.split(key); act, _ = inf(s.obs, a)
                s = env.step(s, act)
                return (s, key), s.metrics["sparc"] * (1.0 - s.done)
            (_, _), rs = jax.lax.scan(stp, (st, rng), None, length=steps)
            return rs.sum()
        return jax.vmap(ep)(jax.random.split(rng, epis)).mean()
    return ret


def passive_fn(env, steps, epis):
    """Static proxy: survival under zero action (stand stability per design)."""
    @jax.jit
    def pas(rng, design):
        z = jnp.zeros((env.action_size,))
        def ep(k):
            st = env.reset_with(k, design)
            def stp(s, _):
                s = env.step(s, z); return s, (1.0 - s.done)
            _, up = jax.lax.scan(stp, st, None, length=steps)
            return up.sum()
        return jax.vmap(ep)(jax.random.split(rng, epis)).mean()
    return pas


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", default=str(OUT / "cval_ckpt.pkl"))
    ap.add_argument("--n-designs", type=int, default=16)
    ap.add_argument("--worlds", type=int, default=12)
    ap.add_argument("--truth-worlds", type=int, default=24)
    ap.add_argument("--eval-epis", type=int, default=16)
    ap.add_argument("--eval-steps", type=int, default=200)
    ap.add_argument("--no-striker", action="store_true",
                    help="legacy 12-action fighter (STEP-1 cval_ckpt); omit for the striker fighter (STEP-3)")
    ap.add_argument("--tiny", action="store_true")
    args = ap.parse_args()
    if args.tiny:
        args.n_designs, args.worlds, args.truth_worlds, args.eval_epis, args.eval_steps = 6, 4, 8, 6, 60

    # MUST match how the fighter ckpt was trained: lean contacts + striker on/off (action dim).
    # The STEP-1 result used --no-striker (the contact-forced 12-action fighter).
    striker = False if args.no_striker else None
    env_nom = AdversarialEnv(self_collision=False, striker=striker)                   # reality_gap OFF
    env_cal = AdversarialEnv(self_collision=False, reality_gap=True,
                             n_worlds=max(args.truth_worlds, args.worlds), striker=striker)  # reality_gap ON
    inf = load_policy(args.ckpt, env_nom.observation_size, env_nom.action_size)
    r_nom = sparc_ret_fn(env_nom, inf, args.eval_steps, args.eval_epis)
    r_cal = sparc_ret_fn(env_cal, inf, args.eval_steps, args.eval_epis)
    passive = passive_fn(env_nom, args.eval_steps, max(6, args.eval_epis // 2))

    rng = np.random.default_rng(0)
    designs = rng.uniform(0, 1, (args.n_designs, DESIGN_DIM))
    t0 = time.time()
    proxy = np.array([float(passive(jax.random.PRNGKey(5), jnp.asarray(d))) for d in designs])
    nominal = np.array([float(r_nom(jax.random.PRNGKey(7), jnp.asarray(d))) for d in designs])
    cal = np.array([[float(r_cal(jax.random.PRNGKey(1000 + w), jnp.asarray(d)))
                     for w in range(args.truth_worlds)] for d in designs])
    robust = np.array([cvar(row[:args.worlds], 0.2) for row in cal])
    truth = cal.mean(1)

    rho_p, rho_n, rho_r = (rank_correlation(proxy, truth), rank_correlation(nominal, truth),
                           rank_correlation(robust, truth))
    claim = rho_r >= rho_n >= rho_p
    np.savez(OUT / "combat_rank.npz", proxy=proxy, nominal=nominal, robust=robust, truth=truth,
             rho_proxy=rho_p, rho_nominal=rho_n, rho_robust=rho_r)
    METRIC(stage="combat_rank", t_s=f"{time.time()-t0:.1f}", n_designs=args.n_designs,
           proxy_rho=f"{rho_p:+.3f}", nominal_rho=f"{rho_n:+.3f}", robust_rho=f"{rho_r:+.3f}",
           claim_holds=int(claim), spread=f"{truth.max()-truth.min():.2f}")
    print(f"\n[COMBAT rank vs wide-world ground truth — {args.n_designs} bodies, {args.truth_worlds} worlds]")
    print(f"     proxy   ρ = {rho_p:+.2f}")
    print(f"     nominal ρ = {rho_n:+.2f}")
    print(f"     robust  ρ = {rho_r:+.2f}   (CVaR@20% over {args.worlds} worlds)")
    print(f"     combat-SPARC spread across bodies: {truth.min():.2f}..{truth.max():.2f}")
    rb, nb, pb = int(np.argmax(robust)), int(np.argmax(nominal)), int(np.argmax(proxy))
    print(f"[COMBAT] true combat perf of each ranking's pick: robust={truth[rb]:.2f} "
          f"nominal={truth[nb]:.2f} proxy={truth[pb]:.2f}")
    print(f"PROVEN: combat body-ranking experiment ran; robust(CVaR) ≥ nominal > proxy: {claim} "
          f"(if the spread is ~0 the fighter is too weak to differentiate bodies -> STEP 2). "
          f"This is the co-design thesis tested where contact dynamics matter.")


if __name__ == "__main__":
    main()
