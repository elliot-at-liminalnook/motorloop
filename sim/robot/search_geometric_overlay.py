# SPDX-License-Identifier: MIT
"""Search a low-dimensional geometric residual overlay for fighter contact conversion.

This is a diagnostic/prototyping tool, not a checkpoint format: it keeps the
policy fixed and adds a compact feedback residual based on benchmark geometry.
If it finds contact where the policy cannot, the overlay tells us what feedback
features or adapter structure the trainable policy needs.
"""

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

from train_adversarial import AdversarialEnv, BENCH_KEYS  # noqa: E402

OUT = Path(os.environ.get("CODESIGN_OUT", "/root/proj/out"))
OUT.mkdir(parents=True, exist_ok=True)


def build_overlay_benchmark(env: AdversarialEnv, n_epis: int, steps: int, seeds: list[int],
                            n_feat: int):
    from brax.training.acme import running_statistics
    from brax.training.agents.ppo import networks as ppo_networks

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
    def bench(params, overlay_vec):
        inf = make_inf(params, deterministic=True)
        weights = overlay_vec.reshape(n_feat, env.action_size)

        def features(ps, t):
            rel = (ps.xpos[env._Bt] - ps.xpos[env._At])[:2]
            dist = jnp.linalg.norm(rel)
            unit = rel / (dist + 1e-6)
            rmat = ps.xmat[env._At].reshape(-1)
            forward = rmat[:2]
            side_axis = rmat[3:5]
            forward = forward / (jnp.linalg.norm(forward) + 1e-6)
            side_axis = side_axis / (jnp.linalg.norm(side_axis) + 1e-6)
            front = jnp.dot(forward, unit)
            side = jnp.dot(side_axis, unit)
            rod_d = jnp.min(jnp.linalg.norm(ps.geom_xpos[env._Arod_gids] - ps.xpos[env._Bt], axis=1))
            close = jnp.clip((0.75 - dist) / 0.50, 0.0, 1.0)
            near = jnp.clip((0.45 - dist) / 0.25, 0.0, 1.0)
            rod = jnp.clip((0.50 - rod_d) / 0.40, 0.0, 1.0)
            late = jnp.clip(t / jnp.maximum(float(steps), 1.0), 0.0, 1.0)
            return jnp.array([
                1.0,
                close,
                near,
                rod,
                front,
                side,
                close * side,
                rod * side,
                rod * front,
                late,
            ], dtype=jnp.float32)

        def ep(k):
            st = env.reset(k)
            d0 = jnp.linalg.norm((st.pipeline_state.xpos[env._Bt] - st.pipeline_state.xpos[env._At])[:2])

            def stp(carry, _):
                s, key, t = carry
                key, sk = jax.random.split(key)
                a, _ = inf(s.obs, sk)
                feat = features(s.pipeline_state, t)
                residual = feat @ weights
                a = jnp.clip(a + residual, -1.0, 1.0)
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
                ps = s.pipeline_state
                tz = ps.xpos[env._At][2]
                up_a = ps.xmat[env._At].reshape(-1)[8]
                pen = jnp.max(jnp.maximum(0.0, -ps.contact.dist))
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

        seed_vals = jax.vmap(seed_eval)(seed_keys)
        return seed_vals.mean(0), seed_vals

    return bench


def score(vals: np.ndarray, *, keep_metric: str, min_survival: float,
          max_peak_pen: float, max_early: float) -> tuple[float, dict]:
    rec = {k: float(vals[i]) for i, k in enumerate(BENCH_KEYS)}
    rec["ratio"] = rec["dealt"] / max(rec["taken"], 1e-6)
    rec["margin"] = rec["dealt"] - rec["taken"]
    judge = 100.0 * rec["win_rate"] + rec["sparc"] + 20.0 * rec["margin"]
    rec["judge"] = judge
    valid = (
        rec["survival_rate"] >= min_survival
        and rec["safe_rate"] >= min_survival
        and rec["ac_peak_pen"] <= max_peak_pen
        and rec["ac_dmg_early"] <= max_early
    )
    if keep_metric == "dealt":
        raw = 100.0 * rec["dealt"] + rec["sparc"] + 10.0 * rec["margin"]
    elif keep_metric == "sparc":
        raw = rec["sparc"]
    elif keep_metric == "judge":
        raw = judge
    elif keep_metric == "margin":
        raw = 100.0 * rec["margin"] + rec["sparc"]
    else:
        raise ValueError(keep_metric)
    rec["valid"] = bool(valid)
    rec["selected_score"] = float(raw if valid else -1e9 + raw)
    return rec["selected_score"], rec


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--tag", default="geometric_overlay")
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--bench-sep-lo", type=float, default=0.25)
    ap.add_argument("--bench-sep-hi", type=float, default=0.70)
    ap.add_argument("--bench-az", type=float, default=3.14159)
    ap.add_argument("--bench-epis", type=int, default=8)
    ap.add_argument("--bench-steps", type=int, default=80)
    ap.add_argument("--bench-seeds", default="20240681")
    ap.add_argument("--lean-contacts", action="store_true")
    ap.add_argument("--engage-obs", action="store_true")
    ap.add_argument("--gens", type=int, default=8)
    ap.add_argument("--pop", type=int, default=48)
    ap.add_argument("--elite", type=int, default=8)
    ap.add_argument("--sigma", type=float, default=0.20)
    ap.add_argument("--min-sigma", type=float, default=0.02)
    ap.add_argument("--max-abs", type=float, default=0.85)
    ap.add_argument("--seed", type=int, default=123)
    ap.add_argument("--keep-metric", choices=["dealt", "sparc", "judge", "margin"], default="dealt")
    ap.add_argument("--min-survival", type=float, default=1.0)
    ap.add_argument("--max-peak-pen", type=float, default=0.05)
    ap.add_argument("--max-early", type=float, default=0.5)
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
    feature_names = [
        "bias", "close", "near", "rod", "front", "side", "close_side",
        "rod_side", "rod_front", "late",
    ]
    n_feat = len(feature_names)
    search_dim = n_feat * env.action_size
    bench = build_overlay_benchmark(env, args.bench_epis, args.bench_steps, seeds, n_feat)

    def evaluate(vec: np.ndarray) -> tuple[float, dict]:
        vals, _ = bench(params, jnp.asarray(vec, dtype=jnp.float32))
        return score(
            np.asarray(vals),
            keep_metric=args.keep_metric,
            min_survival=args.min_survival,
            max_peak_pen=args.max_peak_pen,
            max_early=args.max_early,
        )

    rng = np.random.default_rng(args.seed)
    mean = np.zeros(search_dim, dtype=np.float32)
    sigma = np.full(search_dim, args.sigma, dtype=np.float32)
    max_abs = np.full(search_dim, args.max_abs, dtype=np.float32)
    best_score, best_rec = evaluate(mean)
    best_vec = mean.copy()
    print(
        f"[geom-overlay] base score={best_score:.3f} sparc={best_rec['sparc']:.3f} "
        f"dealt={best_rec['dealt']:.4f} taken={best_rec['taken']:.4f} "
        f"margin={best_rec['margin']:+.4f} pen={best_rec['ac_peak_pen']:.4f} "
        f"valid={best_rec['valid']}",
        flush=True,
    )
    hist = []
    for gen in range(args.gens):
        samples = rng.normal(mean, sigma, size=(args.pop, search_dim)).astype(np.float32)
        samples = np.clip(samples, -max_abs[None, :], max_abs[None, :])
        samples[0] = mean
        if gen == 0:
            samples[0] = 0.0
        rows = []
        for i, sample in enumerate(samples):
            sc, rec = evaluate(sample)
            rows.append((sc, sample.copy(), rec))
            if sc > best_score:
                best_score = sc
                best_vec = sample.copy()
                best_rec = {**rec, "gen": gen, "rank": i}
        rows.sort(key=lambda r: r[0], reverse=True)
        elites = np.stack([r[1] for r in rows[: max(1, args.elite)]], axis=0)
        mean = elites.mean(axis=0)
        sigma = np.maximum(elites.std(axis=0) * 0.9, args.min_sigma)
        top = rows[0][2]
        row = {
            "gen": gen,
            "gen_score": float(rows[0][0]),
            "best_score": float(best_score),
            "sparc": top["sparc"],
            "dealt": top["dealt"],
            "taken": top["taken"],
            "margin": top["margin"],
            "closing": top["closing"],
            "fleeing": top["fleeing"],
            "peak_pen": top["ac_peak_pen"],
            "early": top["ac_dmg_early"],
            "valid": top["valid"],
            "sigma_mean": float(sigma.mean()),
        }
        hist.append(row)
        print(
            f"[geom-overlay] gen={gen:02d} score={row['gen_score']:.3f} "
            f"sparc={row['sparc']:.3f} dealt={row['dealt']:.4f} "
            f"taken={row['taken']:.4f} margin={row['margin']:+.4f} "
            f"close={row['closing']:.3f} flee={row['fleeing']:.3f} "
            f"pen={row['peak_pen']:.4f} valid={row['valid']} best={best_score:.3f}",
            flush=True,
        )

    report = {
        "tag": args.tag,
        "ckpt": args.ckpt,
        "seeds": seeds,
        "feature_names": feature_names,
        "action_size": int(env.action_size),
        "search_dim": search_dim,
        "base": best_rec if len(hist) == 0 else None,
        "best": best_rec,
        "best_vector": best_vec.tolist(),
        "best_matrix": best_vec.reshape(n_feat, env.action_size).tolist(),
        "history": hist,
    }
    out = OUT / f"{args.tag}_geometric_overlay.json"
    out.write_text(json.dumps(report, indent=2))
    print(f"[geom-overlay] saved {out}", flush=True)


if __name__ == "__main__":
    os.environ.pop("MUJOCO_GL", None)
    main()
