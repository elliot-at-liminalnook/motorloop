# SPDX-License-Identifier: MIT
"""Search an observation-gated action blend between two fighter checkpoints."""

from __future__ import annotations

import argparse
import copy
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

import train_adversarial as ta  # noqa: E402
from search_policy_bias import score_record  # noqa: E402

OUT = Path(os.environ.get("CODESIGN_OUT", "/root/proj/out"))
OUT.mkdir(parents=True, exist_ok=True)

FEATURE_NAMES = [
    "bias", "dist", "unit_x", "unit_y", "radial", "lateral",
    "front", "abs_side", "min_rod", "rod_close", "body_close",
]
TIME_FEATURE_NAMES = ["time", "time_sq"]


def _feature_names(use_time_features: bool) -> list[str]:
    return FEATURE_NAMES + (TIME_FEATURE_NAMES if use_time_features else [])


def _apply_overrides(args) -> None:
    if not any(x is not None for x in (
        args.striker_rod_len,
        args.striker_stroke,
        args.striker_rod_radius,
        args.contact_solref_timeconst,
        args.floor_calf_solref_timeconst,
    )) and not args.disable_calf_floor:
        return
    ta.SPEC = copy.deepcopy(ta.SPEC)
    ta.SPEC.setdefault("striker", {})
    ta.SPEC.setdefault("contact", {})
    if args.striker_rod_len is not None:
        ta.SPEC["striker"]["rod_len"] = float(args.striker_rod_len)
    if args.striker_stroke is not None:
        ta.SPEC["striker"]["stroke"] = float(args.striker_stroke)
    if args.striker_rod_radius is not None:
        ta.SPEC["striker"]["rod_radius"] = float(args.striker_rod_radius)
    if args.contact_solref_timeconst is not None:
        ta.SPEC["contact"]["solref"] = [
            float(args.contact_solref_timeconst),
            float(args.contact_solref_dampratio),
        ]
    if args.floor_calf_solref_timeconst is not None:
        ta.SPEC["contact"]["floor_calf_solref"] = [
            float(args.floor_calf_solref_timeconst),
            float(args.floor_calf_solref_dampratio),
        ]
    if args.disable_calf_floor:
        ta.SPEC["contact"]["calf_floor"] = False


def _make_infer(params, obs_dim: int, act_dim: int):
    from brax.training.acme import running_statistics
    from brax.training.agents.ppo import networks as ppo_networks

    net = ppo_networks.make_ppo_networks(
        obs_dim, act_dim, preprocess_observations_fn=running_statistics.normalize
    )
    return ppo_networks.make_inference_fn(net)(params, deterministic=True)


def _gate_features(obs, t_norm=0.0, use_time_features: bool = False):
    # Requires --engage-obs --contact-obs: engage tail starts at 44, contact tail at 52.
    dist = obs[44]
    unit_x = obs[45]
    unit_y = obs[46]
    radial = obs[47]
    lateral = obs[48]
    front = obs[56]
    side_abs = jnp.abs(obs[57])
    min_rod = obs[54]
    rod_close = obs[58]
    body_close = obs[59]
    base = jnp.array([1.0, dist, unit_x, unit_y, radial, lateral, front, side_abs, min_rod, rod_close, body_close])
    if not use_time_features:
        return base
    t = jnp.asarray(t_norm, dtype=base.dtype)
    return jnp.concatenate([base, jnp.array([t, t * t], dtype=base.dtype)])


def base_router_vector_dim(router_hidden: int, action_size: int, constant_action_gate: bool,
                           action_state_boost: bool, action_state_pull: bool,
                           feature_dim: int) -> int:
    if constant_action_gate:
        return (
            int(action_size)
            + (feature_dim if action_state_boost else 0)
            + (feature_dim if action_state_pull else 0)
        )
    return feature_dim if router_hidden <= 0 else feature_dim + router_hidden * (feature_dim + 2) + 1


def router_vector_dim(router_hidden: int, action_size: int, constant_action_gate: bool,
                      action_state_boost: bool, action_state_pull: bool,
                      c_state_gate: bool, feature_dim: int) -> int:
    dim = base_router_vector_dim(
        router_hidden, action_size, constant_action_gate, action_state_boost,
        action_state_pull, feature_dim
    )
    return dim + (feature_dim if c_state_gate else 0)


def build_router_benchmark(env, params_a, params_b, params_c, n_epis: int, steps: int, seeds: list[int],
                           router_hidden: int, constant_action_gate: bool,
                           action_state_boost: bool, action_state_pull: bool,
                           c_state_gate: bool,
                           use_time_features: bool):
    infer_a = _make_infer(params_a, env.observation_size, env.action_size)
    infer_b = _make_infer(params_b, env.observation_size, env.action_size)
    infer_c = None if params_c is None else _make_infer(params_c, env.observation_size, env.action_size)
    feature_dim = len(_feature_names(use_time_features))
    base_dim = base_router_vector_dim(
        router_hidden, env.action_size, constant_action_gate, action_state_boost,
        action_state_pull, feature_dim
    )
    seed_keys = jnp.stack(
        [jax.random.split(jax.random.PRNGKey(int(seed)), n_epis) for seed in seeds],
        axis=0,
    )
    step_env = jax.jit(env.step)

    @jax.jit
    def bench(gate_vec):
        def blended_action(obs, key, t):
            ka, kb, kc = jax.random.split(key, 3)
            a, _ = infer_a(obs, ka)
            b, _ = infer_b(obs, kb)
            feat = _gate_features(obs, t / jnp.maximum(float(steps - 1), 1.0), use_time_features)
            base_vec = gate_vec[:base_dim]
            if constant_action_gate:
                base_g = jax.nn.sigmoid(base_vec[:env.action_size])
                off = env.action_size
                if action_state_boost:
                    boost = jax.nn.sigmoid(jnp.dot(base_vec[off:off + feature_dim], feat))
                    off += feature_dim
                    g = base_g + (1.0 - base_g) * boost
                else:
                    g = base_g
                if action_state_pull:
                    pull = jax.nn.sigmoid(jnp.dot(base_vec[off:off + feature_dim], feat))
                    g = g * (1.0 - pull)
                mixed = (1.0 - g) * a + g * b
                gate_metric = jnp.mean(g)
            else:
                logit = jnp.dot(base_vec[:len(FEATURE_NAMES)], feat)
                if router_hidden > 0:
                    off = len(FEATURE_NAMES)
                    w1 = base_vec[off:off + feature_dim * router_hidden].reshape(
                        feature_dim, router_hidden
                    )
                    off += feature_dim * router_hidden
                    b1 = base_vec[off:off + router_hidden]
                    off += router_hidden
                    w2 = base_vec[off:off + router_hidden]
                    off += router_hidden
                    b2 = base_vec[off]
                    logit = logit + jnp.dot(jnp.tanh(jnp.dot(feat, w1) + b1), w2) + b2
                g = jax.nn.sigmoid(logit)
                mixed = (1.0 - g) * a + g * b
                gate_metric = g
            if c_state_gate:
                c, _ = infer_c(obs, kc)
                cg = jax.nn.sigmoid(jnp.dot(gate_vec[base_dim:base_dim + feature_dim], feat))
                mixed = (1.0 - cg) * mixed + cg * c
                gate_metric = 0.5 * (gate_metric + cg)
            return mixed, gate_metric

        def ep(k):
            st = env.reset(k)
            d0 = jnp.linalg.norm((st.pipeline_state.xpos[env._Bt] - st.pipeline_state.xpos[env._At])[:2])

            def stp(carry, _):
                s, key, t = carry
                key, sk = jax.random.split(key)
                a, gate = blended_action(s.obs, sk, t)
                feat = _gate_features(s.obs, t / jnp.maximum(float(steps - 1), 1.0), use_time_features)
                s = step_env(s, a)
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
                    gate * alive,
                ])
                ps = s.pipeline_state
                tz = ps.xpos[env._At][2]
                up_a = ps.xmat[env._At].reshape(-1)[8]
                pen = jnp.max(jnp.maximum(0.0, -ps.contact.dist))
                idle = (jnp.mean(jnp.abs(a)) < 0.1).astype(jnp.float32)
                ac = jnp.array([tz, pen, m["dealt"], up_a, idle, t])
                return (s, key, t + 1.0), (base, ac, feat)

            (_, _, _), (base_o, ac_o, feat_o) = jax.lax.scan(stp, (st, k, 0.0), None, length=steps)
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
            return base_o.sum(0), d0, ac_agg, feat_o.mean(0)

        def seed_eval(keys):
            per_ep, d0, ac_ep, feat_ep = jax.vmap(ep)(keys)
            agg = per_ep[:, :10].mean(0)
            spe = per_ep[:, 0]
            bm = lambda mask: jnp.sum(spe * mask) / jnp.maximum(jnp.sum(mask), 1.0)
            bins = jnp.array([bm(d0 < 0.6), bm((d0 >= 0.6) & (d0 < 0.9)), bm(d0 >= 0.9)])
            dealt_s, taken_s, alive_s, sat_s = per_ep[:, 1], per_ep[:, 2], per_ep[:, 9], per_ep[:, 10]
            survived_bout = alive_s >= steps - 0.5
            win = jnp.mean(((dealt_s - taken_s > 0.0) & survived_bout).astype(jnp.float32))
            surv = jnp.mean(survived_bout.astype(jnp.float32))
            safe = jnp.mean(((sat_s / steps) < 0.5).astype(jnp.float32))
            row = jnp.concatenate([agg, bins, jnp.array([win, surv, safe]), ac_ep.mean(0)])
            gate_mean = per_ep[:, 11].mean() / steps
            return row, gate_mean, feat_ep.mean(0)

        seed_vals, gate_means, feature_means = jax.vmap(seed_eval)(seed_keys)
        return seed_vals.mean(0), seed_vals, gate_means, feature_means

    return bench


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--tag", default="policy_router")
    ap.add_argument("--ckpt-a", required=True)
    ap.add_argument("--ckpt-b", required=True)
    ap.add_argument("--ckpt-c", default=None)
    ap.add_argument("--bench-seeds", default="20240601")
    ap.add_argument("--bench-epis", type=int, default=8)
    ap.add_argument("--bench-steps", type=int, default=80)
    ap.add_argument("--bench-sep-lo", type=float, default=0.25)
    ap.add_argument("--bench-sep-hi", type=float, default=0.70)
    ap.add_argument("--bench-az", type=float, default=3.14159)
    ap.add_argument("--frame-skip", type=int, default=5)
    ap.add_argument("--lean-contacts", action="store_true")
    ap.add_argument("--engage-obs", action="store_true")
    ap.add_argument("--contact-obs", action="store_true")
    ap.add_argument("--striker-rod-len", type=float, default=None)
    ap.add_argument("--striker-stroke", type=float, default=None)
    ap.add_argument("--striker-rod-radius", type=float, default=None)
    ap.add_argument("--contact-solref-timeconst", type=float, default=None)
    ap.add_argument("--contact-solref-dampratio", type=float, default=1.0)
    ap.add_argument("--floor-calf-solref-timeconst", type=float, default=None)
    ap.add_argument("--floor-calf-solref-dampratio", type=float, default=1.0)
    ap.add_argument("--disable-calf-floor", action="store_true")
    ap.add_argument("--gens", type=int, default=5)
    ap.add_argument("--pop", type=int, default=24)
    ap.add_argument("--elite", type=int, default=6)
    ap.add_argument("--sigma", type=float, default=0.8)
    ap.add_argument("--min-sigma", type=float, default=0.05)
    ap.add_argument("--max-abs", type=float, default=6.0)
    ap.add_argument("--seed", type=int, default=123)
    ap.add_argument("--eval-router-json", default=None,
                    help="Evaluate the best_vector from a saved router JSON and exit.")
    ap.add_argument("--init-router-json", default=None,
                    help="Initialize CEM mean from the best_vector in a saved router JSON.")
    ap.add_argument("--router-hidden", type=int, default=0,
                    help="Number of tanh hidden units added on top of the linear gate.")
    ap.add_argument("--constant-action-gate", action="store_true",
                    help="Search one constant branch-B gate per action dimension instead of one state gate.")
    ap.add_argument("--action-state-boost", action="store_true",
                    help="With --constant-action-gate, append a scalar state gate that boosts every action toward branch B.")
    ap.add_argument("--action-state-pull", action="store_true",
                    help="With --constant-action-gate, append a scalar state gate that pulls every action back toward branch A.")
    ap.add_argument("--c-state-gate", action="store_true",
                    help="Append a scalar state gate that blends the A/B mix toward --ckpt-c.")
    ap.add_argument("--time-features", action="store_true",
                    help="Append normalized episode time and time^2 to router gate features.")
    ap.add_argument("--keep-metric", choices=["sparc", "judge", "margin", "dealt", "min_judge", "min_margin"], default="judge")
    ap.add_argument("--min-dealt", type=float, default=0.8)
    ap.add_argument("--per-seed-min-dealt", type=float, default=0.0)
    ap.add_argument("--max-peak-pen", type=float, default=0.05)
    ap.add_argument("--max-early", type=float, default=0.8)
    ap.add_argument("--min-survival", type=float, default=1.0)
    ap.add_argument("--min-safe", type=float, default=1.0)
    ap.add_argument("--per-seed-gates", action="store_true")
    args = ap.parse_args()

    if not (args.engage_obs and args.contact_obs):
        raise SystemExit("router features require --engage-obs --contact-obs")
    if args.action_state_boost and not args.constant_action_gate:
        raise SystemExit("--action-state-boost requires --constant-action-gate")
    if args.action_state_pull and not args.constant_action_gate:
        raise SystemExit("--action-state-pull requires --constant-action-gate")
    if args.c_state_gate and not args.ckpt_c:
        raise SystemExit("--c-state-gate requires --ckpt-c")

    _apply_overrides(args)
    seeds = [int(x) for x in args.bench_seeds.split(",") if x.strip()]
    env = ta.AdversarialEnv(
        self_collision=not args.lean_contacts,
        frame_skip=args.frame_skip,
        sep_lo=args.bench_sep_lo,
        sep_hi=args.bench_sep_hi,
        azimuth=args.bench_az,
        striker=None,
        opponent="passive",
        engage_obs=args.engage_obs,
        contact_obs=args.contact_obs,
    )
    params_a = ta.warm_start(args.ckpt_a, env.observation_size, env.action_size)
    params_b = ta.warm_start(args.ckpt_b, env.observation_size, env.action_size)
    params_c = None if args.ckpt_c is None else ta.warm_start(args.ckpt_c, env.observation_size, env.action_size)
    if params_a is None:
        params_a = pickle.load(open(args.ckpt_a, "rb"))
    if params_b is None:
        params_b = pickle.load(open(args.ckpt_b, "rb"))
    if args.ckpt_c is not None and params_c is None:
        params_c = pickle.load(open(args.ckpt_c, "rb"))
    feature_names = _feature_names(args.time_features)
    bench = build_router_benchmark(
        env, params_a, params_b, params_c, args.bench_epis, args.bench_steps, seeds,
        args.router_hidden, args.constant_action_gate, args.action_state_boost,
        args.action_state_pull, args.c_state_gate, args.time_features
    )

    def evaluate(vec):
        vals, seed_vals, gate_means, feature_means = bench(jnp.asarray(vec, dtype=jnp.float32))
        score, rec = score_record(
            np.asarray(vals),
            min_dealt=args.min_dealt,
            max_peak_pen=args.max_peak_pen,
            max_early=args.max_early,
            keep_metric=args.keep_metric,
            seed_vals=np.asarray(seed_vals),
            per_seed_gates=args.per_seed_gates,
            per_seed_min_dealt=args.per_seed_min_dealt,
            min_survival=args.min_survival,
            min_safe=args.min_safe,
        )
        rec["gate_means"] = [float(x) for x in np.asarray(gate_means)]
        rec["gate_mean"] = float(np.mean(rec["gate_means"]))
        rec["feature_means"] = [
            {name: float(row[i]) for i, name in enumerate(feature_names)}
            for row in np.asarray(feature_means)
        ]
        return score, rec

    feat_dim = len(feature_names)
    base_dim = base_router_vector_dim(
        args.router_hidden, env.action_size, args.constant_action_gate,
        args.action_state_boost, args.action_state_pull, feat_dim
    )
    dim = router_vector_dim(
        args.router_hidden, env.action_size, args.constant_action_gate,
        args.action_state_boost, args.action_state_pull, args.c_state_gate, feat_dim
    )

    def load_router_vector(path: str) -> np.ndarray:
        src = Path(path)
        data = json.loads(src.read_text())
        if "best_vector" not in data:
            raise SystemExit(f"{src} does not contain best_vector")
        vec = np.asarray(data["best_vector"], dtype=np.float32)
        if not args.constant_action_gate and vec.shape == (feat_dim,) and dim > feat_dim:
            out = np.zeros(dim, dtype=np.float32)
            out[:feat_dim] = vec
            return out
        if args.constant_action_gate and args.action_state_boost and vec.shape == (env.action_size,):
            out = np.zeros(dim, dtype=np.float32)
            out[:env.action_size] = vec
            if args.c_state_gate:
                out[base_dim] = -args.max_abs
            return out
        if args.constant_action_gate and args.action_state_pull:
            old_base_dim = base_router_vector_dim(
                args.router_hidden, env.action_size, args.constant_action_gate,
                args.action_state_boost, False, feat_dim
            )
            old_dim = old_base_dim + (feat_dim if args.c_state_gate else 0)
            if vec.shape == (old_dim,):
                out = np.zeros(dim, dtype=np.float32)
                out[:old_base_dim] = vec[:old_base_dim]
                pull_start = old_base_dim
                out[pull_start] = -args.max_abs
                if args.c_state_gate:
                    out[base_dim:base_dim + feat_dim] = vec[old_base_dim:old_base_dim + feat_dim]
                return out
        if args.c_state_gate and vec.shape == (base_dim,):
            out = np.zeros(dim, dtype=np.float32)
            out[:base_dim] = vec
            out[base_dim] = -args.max_abs
            return out
        if vec.shape != (dim,):
            raise SystemExit(f"expected vector shape {(dim,)}, got {vec.shape}")
        return vec

    if args.eval_router_json:
        src = Path(args.eval_router_json)
        vec = load_router_vector(args.eval_router_json)
        score, rec = evaluate(vec)
        out = OUT / f"{args.tag}_policy_router_eval.json"
        out.write_text(json.dumps({
            "tag": args.tag,
            "mode": "eval",
            "source_router_json": str(src),
            "ckpt_a": args.ckpt_a,
            "ckpt_b": args.ckpt_b,
            "ckpt_c": args.ckpt_c,
            "seeds": seeds,
            "feature_names": [
                *feature_names,
            ],
            "router_hidden": args.router_hidden,
            "constant_action_gate": args.constant_action_gate,
            "action_state_boost": args.action_state_boost,
            "action_state_pull": args.action_state_pull,
            "c_state_gate": args.c_state_gate,
            "time_features": args.time_features,
            "action_size": env.action_size,
            "score": float(score),
            "vector": vec.tolist(),
            "result": rec,
        }, indent=2))
        print(
            f"[router-eval] score={score:.3f} sparc={rec['sparc']:.3f} "
            f"dealt={rec['dealt']:.4f} taken={rec['taken']:.4f} "
            f"margin={rec['bench_margin']:+.4f} gate={rec['gate_mean']:.3f} "
            f"valid={rec['valid']} saved={out}",
            flush=True,
        )
        return

    rng = np.random.default_rng(args.seed)
    init_vec = load_router_vector(args.init_router_json) if args.init_router_json else None
    mean = np.zeros(dim, dtype=np.float32) if init_vec is None else init_vec.copy()
    sigma = np.full(dim, args.sigma, dtype=np.float32)
    max_abs = np.full(dim, args.max_abs, dtype=np.float32)
    best_score = -1e30
    best_vec = np.zeros(dim, dtype=np.float32)
    best_rec = {}
    hist = []

    seeds_to_try = [np.zeros(dim, dtype=np.float32)]
    a_only = np.zeros(dim, dtype=np.float32)
    b_only = np.zeros(dim, dtype=np.float32)
    if args.constant_action_gate:
        a_only[:] = -args.max_abs
        b_only[:] = args.max_abs
    else:
        a_only[0] = -args.max_abs
        b_only[0] = args.max_abs
    if args.c_state_gate:
        a_only[base_dim] = -args.max_abs
        b_only[base_dim] = -args.max_abs
        c_only = np.zeros(dim, dtype=np.float32)
        c_only[base_dim] = args.max_abs
    else:
        c_only = None
    seeds_to_try += [a_only, b_only]
    if c_only is not None:
        seeds_to_try.append(c_only)
    if init_vec is not None:
        seeds_to_try.append(np.clip(init_vec, -max_abs, max_abs).astype(np.float32))
    for i, vec in enumerate(seeds_to_try):
        score, rec = evaluate(vec)
        if score > best_score:
            best_score, best_vec, best_rec = score, vec.copy(), {**rec, "gen": -1, "rank": i}
        print(
            f"[router] seed={i} score={score:.3f} sparc={rec['sparc']:.3f} "
            f"margin={rec['bench_margin']:+.3f} gate={rec['gate_mean']:.3f} valid={rec['valid']}",
            flush=True,
        )
    for gen in range(args.gens):
        samples = rng.normal(mean, sigma, size=(args.pop, dim)).astype(np.float32)
        samples = np.clip(samples, -max_abs[None, :], max_abs[None, :])
        samples[0] = mean
        rows = []
        for i, sample in enumerate(samples):
            score, rec = evaluate(sample)
            rows.append((score, sample.copy(), rec))
            if score > best_score:
                best_score, best_vec, best_rec = score, sample.copy(), {**rec, "gen": gen, "rank": i}
        rows.sort(key=lambda x: x[0], reverse=True)
        elites = np.stack([r[1] for r in rows[: max(1, args.elite)]], axis=0)
        mean = elites.mean(axis=0)
        sigma = np.maximum(elites.std(axis=0) * 0.9, args.min_sigma)
        top = rows[0][2]
        hist.append({
            "gen": gen,
            "score": float(rows[0][0]),
            "sparc": top["sparc"],
            "margin": top["bench_margin"],
            "gate_mean": top["gate_mean"],
            "valid": top["valid"],
        })
        print(
            f"[router] gen={gen:02d} score={rows[0][0]:.3f} sparc={top['sparc']:.3f} "
            f"margin={top['bench_margin']:+.3f} gate={top['gate_mean']:.3f} "
            f"valid={top['valid']} best={best_score:.3f}",
            flush=True,
        )

    out = OUT / f"{args.tag}_policy_router.json"
    out.write_text(json.dumps({
        "tag": args.tag,
        "ckpt_a": args.ckpt_a,
        "ckpt_b": args.ckpt_b,
        "ckpt_c": args.ckpt_c,
        "seeds": seeds,
        "feature_names": feature_names,
        "router_hidden": args.router_hidden,
        "constant_action_gate": args.constant_action_gate,
        "action_state_boost": args.action_state_boost,
        "action_state_pull": args.action_state_pull,
        "c_state_gate": args.c_state_gate,
        "time_features": args.time_features,
        "action_size": env.action_size,
        "best_score": float(best_score),
        "best_vector": best_vec.tolist(),
        "best": best_rec,
        "history": hist,
    }, indent=2))
    print(f"[router] saved {out}", flush=True)


if __name__ == "__main__":
    main()
