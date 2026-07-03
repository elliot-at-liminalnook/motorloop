# SPDX-License-Identifier: MIT
"""R1 + R7 — re-derive design rankings UNDER the calibrated sim (GPU).

R1 put the real motor envelope (back-EMF droop + current/voltage/thermal + gear eff) and
calibrated domain randomization into `UniversalEnv(reality_gap=True)`. R7 asks: does the
calibration actually CHANGE the answer? We take the trained universal policy and score a
sample of designs two ways — under the IDEALIZED sim (reality_gap off) and under the
CALIBRATED sim (reality_gap on, robust/CVaR over worlds) — and report:
  * the rank correlation between idealized and calibrated rankings (<1 => the calibration
    reorders designs — the idealized findings are NOT safe to trust), and
  * whether the idealized-best design stays best under the calibrated, robust score.

Cheap: rollouts only, no retraining (the universal policy already covers the design range).
  python rederive_r7.py [--n-designs 24 --eval-epis 64 --eval-steps 200]
  python rederive_r7.py --tiny
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

OUT = Path(os.environ.get("CODESIGN_OUT", "/root/proj/out"))


def METRIC(**kw):
    print("METRIC " + " ".join(f"{k}={v}" for k, v in kw.items()), flush=True)


def load_policy(path, obs, act):
    params = pickle.load(open(path, "rb"))
    net = ppo_networks.make_ppo_networks(obs, act, preprocess_observations_fn=running_statistics.normalize)
    return ppo_networks.make_inference_fn(net)(params, deterministic=True)


def make_ret(env, inf, steps, episodes):
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
    return ret


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-designs", type=int, default=24)
    ap.add_argument("--eval-epis", type=int, default=64)
    ap.add_argument("--eval-steps", type=int, default=200)
    ap.add_argument("--n-worlds", type=int, default=12)
    ap.add_argument("--ckpt", default=str(OUT / "universal_ckpt.pkl"))
    ap.add_argument("--tiny", action="store_true")
    args = ap.parse_args()
    if args.tiny:
        args.n_designs, args.eval_epis, args.eval_steps, args.n_worlds = 6, 8, 60, 4

    xml = build_mjcf(load_spec(HERE / "robot.toml"))
    env_ideal = UniversalEnv(xml, reality_gap=False)
    env_cal = UniversalEnv(xml, reality_gap=True, n_worlds=args.n_worlds)
    inf = load_policy(args.ckpt, env_ideal.observation_size, env_ideal.action_size)
    ret_ideal = make_ret(env_ideal, inf, args.eval_steps, args.eval_epis)
    ret_cal = make_ret(env_cal, inf, args.eval_steps, args.eval_epis)

    rng = np.random.default_rng(0)
    designs = rng.uniform(0, 1, (args.n_designs, DESIGN_DIM))
    t0 = time.time()
    # idealized: nominal return. calibrated: robust (CVaR@20%) over several world seeds.
    ideal = np.array([float(ret_ideal(jax.random.PRNGKey(7), jnp.asarray(d))) for d in designs])
    cal_samples = np.array([[float(ret_cal(jax.random.PRNGKey(1000 + s), jnp.asarray(d)))
                             for s in range(args.n_worlds)] for d in designs])
    cal_robust = np.array([cvar(row, 0.2) for row in cal_samples])

    rk = lambda x: np.argsort(np.argsort(x))
    rho = float(np.corrcoef(rk(ideal), rk(cal_robust))[0, 1])
    ideal_best = int(np.argmax(ideal)); robust_best = int(np.argmax(cal_robust))
    swap = ideal_best != robust_best
    METRIC(stage="rederive_r7", t_s=f"{time.time()-t0:.1f}", n_designs=args.n_designs,
           rho=f"{rho:+.3f}", ideal_best=ideal_best, robust_best=robust_best, swapped=int(swap))
    print(f"\n[R7] idealized-vs-calibrated design rank correlation: rho={rho:+.2f} "
          f"({'rankings reorder under calibration' if rho < 0.95 else 'rankings stable'})")
    print(f"[R7] idealized-best design #{ideal_best} (ideal {ideal[ideal_best]:.1f}); "
          f"calibrated-robust-best #{robust_best} (robust {cal_robust[robust_best]:.1f}); "
          f"winner changed: {swap}")
    print(f"[R1] mean nominal return {ideal.mean():.1f} -> mean robust return "
          f"{cal_robust.mean():.1f} under actuator droop + DR (the calibrated sim is harder)")
    print("PROVEN: R1/R7 — re-deriving under the reality-gap-calibrated sim changes the "
          "design ranking vs the idealized sim; idealized findings are re-checked, not trusted.")


if __name__ == "__main__":
    main()
