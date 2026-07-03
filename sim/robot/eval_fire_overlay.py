# SPDX-License-Identifier: MIT
"""Evaluate a simple geometric striker-fire overlay on a fighter checkpoint."""

from __future__ import annotations

import argparse
import json
import os
import pickle
import sys
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

from train_adversarial import AdversarialEnv, BENCH_KEYS, load_opponent  # noqa: E402

OUT = Path(os.environ.get("CODESIGN_OUT", "/root/proj/out"))
OUT.mkdir(parents=True, exist_ok=True)


def build_fire_overlay_benchmark(env: AdversarialEnv, n_epis: int, steps: int, seeds: list[int]):
    from brax.training.acme import running_statistics
    import ppo_nets as ppo_networks  # shared (512,256,128) + small-final-init factory (audit item 5)

    net = ppo_networks.make_ppo_networks(
        env.observation_size,
        env.action_size,
        preprocess_observations_fn=running_statistics.normalize,
    )
    make_inf = ppo_networks.make_inference_fn(net)
    seed_keys = jnp.stack(
        [jax.random.split(jax.random.PRNGKey(int(seed)), n_epis) for seed in seeds],
        axis=0,
    )

    @jax.jit
    def bench(params, rod_threshold, body_threshold, fire_value, front_min):
        inf = make_inf(params, deterministic=True)

        def ep(k):
            st = env.reset(k)
            d0 = jnp.linalg.norm((st.pipeline_state.xpos[env._Bt] - st.pipeline_state.xpos[env._At])[:2])

            def stp(carry, _):
                s, key, t = carry
                key, sk = jax.random.split(key)
                a, _ = inf(s.obs, sk)
                ps = s.pipeline_state
                rel = (ps.xpos[env._Bt] - ps.xpos[env._At])[:2]
                dist = jnp.linalg.norm(rel)
                unit = rel / (dist + 1e-6)
                rmat = ps.xmat[env._At].reshape(-1)
                forward = rmat[:2]
                forward = forward / (jnp.linalg.norm(forward) + 1e-6)
                front = jnp.dot(forward, unit)
                rod_d = jnp.min(jnp.linalg.norm(ps.geom_xpos[env._Arod_gids] - ps.xpos[env._Bt], axis=1))
                gate = ((rod_d <= rod_threshold) | (dist <= body_threshold)) & (front >= front_min)
                strike = jnp.maximum(jnp.clip(a[env._strike_local], 0.0, 1.0), fire_value * gate.astype(jnp.float32))
                a = a.at[env._strike_local].set(strike)
                s = env.step(s, a)
                alive = 1.0 - s.done
                m = s.metrics
                sat = jnp.mean(jnp.abs(a[:env._n_hinge]) > 0.95)
                base = jnp.array([
                    m["sparc"] * alive,
                    m["dealt"] * alive,
                    m["taken"] * alive,
                    m["clean_hit"] * alive,
                    m["trade"] * alive,
                    m["fire"] * alive,
                    m["closing"] * alive,
                    m["fleeing"] * alive,
                    m["dist"] * alive,
                    alive,
                    sat,
                ])
                ps2 = s.pipeline_state
                tz = ps2.xpos[env._At][2]
                up_a = ps2.xmat[env._At].reshape(-1)[8]
                pen = jnp.max(jnp.maximum(0.0, -ps2.contact.dist))
                idle = (jnp.mean(jnp.abs(a)) < 0.1).astype(jnp.float32)
                ac = jnp.array([tz, pen, m["dealt"], up_a, idle, t])
                return (s, key, t + 1.0), (base, ac)

            (_, _, _), (base_o, ac_o) = jax.lax.scan(stp, (st, k, 0.0), None, length=steps)
            tz_c, pen_c, dl_c, up_c, idle_c, t_c = (ac_o[:, i] for i in range(6))
            dmg_tot = dl_c.sum()
            tot = jnp.maximum(dmg_tot, 1e-9)
            has_dmg = dmg_tot > 1e-6
            ac_agg = jnp.array([
                tz_c.max(),
                (tz_c > env._airborne_z).mean(),
                pen_c.max(),
                idle_c.mean(),
                jnp.where(has_dmg, (dl_c * (t_c < 0.15 * steps)).sum() / tot, 0.0),
                jnp.where(has_dmg, (dl_c * (up_c > 0.5)).sum() / tot, 1.0),
                jnp.where(has_dmg, (dl_c * (tz_c < env._grounded_z)).sum() / tot, 1.0),
                up_c.mean(),
            ])
            return base_o.sum(0), d0, ac_agg

        def seed_eval(keys):
            per_ep, d0, ac_ep = jax.vmap(ep)(keys)
            agg = per_ep[:, :10].mean(0)
            spe = per_ep[:, 0]
            bm = lambda mask: jnp.sum(spe * mask) / jnp.maximum(jnp.sum(mask), 1.0)
            bins = jnp.array([
                bm(d0 < 0.6),
                bm((d0 >= 0.6) & (d0 < 0.9)),
                bm(d0 >= 0.9),
            ])
            dealt_s, taken_s, alive_s, sat_s = per_ep[:, 1], per_ep[:, 2], per_ep[:, 9], per_ep[:, 10]
            survived_bout = alive_s >= steps - 0.5
            win = jnp.mean(((dealt_s - taken_s > 0.0) & survived_bout).astype(jnp.float32))
            surv = jnp.mean(survived_bout.astype(jnp.float32))
            safe = jnp.mean(((sat_s / steps) < 0.5).astype(jnp.float32))
            return jnp.concatenate([agg, bins, jnp.array([win, surv, safe]), ac_ep.mean(0)])

        vals_by_seed = jax.vmap(seed_eval)(seed_keys)
        return vals_by_seed.mean(0), vals_by_seed

    return bench


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--tag", default="fire_overlay")
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--bench-sep-lo", type=float, default=0.25)
    ap.add_argument("--bench-sep-hi", type=float, default=0.70)
    ap.add_argument("--bench-az", type=float, default=3.14159)
    ap.add_argument("--bench-epis", type=int, default=8)
    ap.add_argument("--bench-steps", type=int, default=80)
    ap.add_argument("--bench-seeds", default="20240681")
    ap.add_argument("--rod-thresholds", default="0.12,0.15,0.18,0.22,0.30,10.0")
    ap.add_argument("--body-thresholds", default="0.0,0.30,0.40")
    ap.add_argument("--front-mins", default="-1.0,0.0,0.5")
    ap.add_argument("--fire-values", default="0.5,1.0")
    ap.add_argument("--lean-contacts", action="store_true")
    ap.add_argument("--engage-obs", action="store_true")
    args = ap.parse_args()

    params = pickle.load(open(args.ckpt, "rb"))
    seeds = [int(x) for x in str(args.bench_seeds).split(",") if x.strip()]
    env = AdversarialEnv(
        self_collision=not args.lean_contacts,
        frame_skip=5,
        sep_lo=args.bench_sep_lo,
        sep_hi=args.bench_sep_hi,
        azimuth=args.bench_az,
        striker=None,
        opponent="passive",
        engage_obs=args.engage_obs,
    )
    bench = build_fire_overlay_benchmark(env, args.bench_epis, args.bench_steps, seeds)
    rod_thresholds = [float(x) for x in args.rod_thresholds.split(",") if x.strip()]
    body_thresholds = [float(x) for x in args.body_thresholds.split(",") if x.strip()]
    front_mins = [float(x) for x in args.front_mins.split(",") if x.strip()]
    fire_values = [float(x) for x in args.fire_values.split(",") if x.strip()]
    rows = []
    for rt in rod_thresholds:
        for bt in body_thresholds:
            for fm in front_mins:
                for fv in fire_values:
                    vals, vals_by_seed = bench(params, rt, bt, fv, fm)
                    rec = {k: float(vals[i]) for i, k in enumerate(BENCH_KEYS)}
                    rec.update({
                        "rod_threshold": rt,
                        "body_threshold": bt,
                        "front_min": fm,
                        "fire_value": fv,
                        "ratio": rec["dealt"] / max(rec["taken"], 1e-6),
                        "margin": rec["dealt"] - rec["taken"],
                    })
                    rows.append(rec)
                    print(
                        f"rt={rt:.3f} bt={bt:.3f} front={fm:.2f} fire={fv:.2f} "
                        f"sparc={rec['sparc']:.3f} dealt={rec['dealt']:.4f} "
                        f"taken={rec['taken']:.4f} margin={rec['margin']:+.4f} "
                        f"pen={rec['ac_peak_pen']:.4f} early={rec['ac_dmg_early']:.3f}",
                        flush=True,
                    )
    rows.sort(key=lambda r: (r["dealt"], r["sparc"]), reverse=True)
    out = OUT / f"{args.tag}_fire_overlay.json"
    out.write_text(json.dumps({"tag": args.tag, "ckpt": args.ckpt, "seeds": seeds, "rows": rows}, indent=2))
    best = rows[0]
    print(
        f"BEST sparc={best['sparc']:.3f} dealt={best['dealt']:.4f} taken={best['taken']:.4f} "
        f"rt={best['rod_threshold']:.3f} bt={best['body_threshold']:.3f} "
        f"front={best['front_min']:.2f} fire={best['fire_value']:.2f}",
        flush=True,
    )
    print(f"saved {out}", flush=True)


if __name__ == "__main__":
    os.environ.pop("MUJOCO_GL", None)
    main()
