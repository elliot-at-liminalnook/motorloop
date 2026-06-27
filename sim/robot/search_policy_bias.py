# SPDX-License-Identifier: MIT
"""Search a low-dimensional final-action bias for a fighter checkpoint.

This is a direct SPARC polish tool: keep the trained policy fixed, perturb only
the final policy-head mean bias, and score candidates with the same held-out
combat benchmark used by ``train_adversarial.py``.
"""

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

OUT = Path(os.environ.get("CODESIGN_OUT", "/root/proj/out"))
OUT.mkdir(parents=True, exist_ok=True)


def _policy_params(params):
    policy = params[1]
    return policy.get("params", policy)


def _output_layer_name(pp: dict) -> str:
    hids = [k for k in pp if k.startswith("hidden_")]
    if not hids:
        raise ValueError("policy has no hidden_* dense layers")
    return max(hids, key=lambda k: int(k.split("_")[-1]))


def infer_action_dim(params) -> int:
    pp = _policy_params(params)
    out = _output_layer_name(pp)
    return int(pp[out]["bias"].shape[0]) // 2


def infer_first_hidden_dim(params) -> int:
    pp = _policy_params(params)
    if "hidden_0" not in pp:
        raise ValueError("policy has no hidden_0 dense layer")
    return int(pp["hidden_0"]["kernel"].shape[1])


def infer_output_input_dim(params) -> int:
    pp = _policy_params(params)
    out = _output_layer_name(pp)
    return int(pp[out]["kernel"].shape[0])


def with_mean_bias_delta(base_params, delta: np.ndarray):
    """Returns a checkpoint tuple with only the policy mean bias shifted."""
    parts = list(base_params)
    policy = copy.deepcopy(parts[1])
    pp0 = policy.get("params", policy)
    pp = dict(pp0)
    out = _output_layer_name(pp)
    layer = dict(pp[out])
    bias = layer["bias"]
    act_dim = bias.shape[0] // 2
    d = jnp.asarray(delta, dtype=bias.dtype)
    if d.shape[0] != act_dim:
        raise ValueError(f"delta length {d.shape[0]} != action dim {act_dim}")
    layer["bias"] = bias.at[:act_dim].add(d)
    pp[out] = layer
    if "params" in policy:
        policy = {**policy, "params": pp}
    else:
        policy = pp
    parts[1] = policy
    return tuple(parts)


def with_adapter_delta(base_params, action_delta: np.ndarray,
                       obs_shift: np.ndarray | None = None,
                       obs_log_std_delta: np.ndarray | None = None,
                       engage_kernel_uv: tuple[np.ndarray, np.ndarray] | None = None,
                       output_kernel_uv: tuple[np.ndarray, np.ndarray] | None = None,
                       engage_start: int = 44):
    """Returns a checkpoint with action bias and optional engage-observation norm edits.

    The engage tail is `[dist, unit_x, unit_y, radial, lateral, rel_radial, closing, fleeing]`.
    Shifting/scaling its normalizer changes how strongly the existing policy reacts to the
    opponent state without changing the checkpoint interface or network architecture.
    """
    params = list(with_mean_bias_delta(base_params, action_delta))
    if (obs_shift is None and obs_log_std_delta is None and engage_kernel_uv is None
            and output_kernel_uv is None):
        return tuple(params)
    norm = params[0]
    mean = norm.mean
    std = norm.std
    n = int(mean.shape[0])
    if n < engage_start + 8:
        raise ValueError(f"checkpoint obs dim {n} does not contain engage tail starting at {engage_start}")
    sl = slice(engage_start, engage_start + 8)
    if obs_shift is not None or obs_log_std_delta is not None:
        if obs_shift is not None:
            s = jnp.asarray(obs_shift, dtype=mean.dtype)
            if s.shape[0] != 8:
                raise ValueError(f"obs_shift length {s.shape[0]} != 8")
            # shift is expressed in normalized-observation units.
            mean = mean.at[sl].add(s * std[sl])
        if obs_log_std_delta is not None:
            g = jnp.asarray(obs_log_std_delta, dtype=std.dtype)
            if g.shape[0] != 8:
                raise ValueError(f"obs_log_std_delta length {g.shape[0]} != 8")
            std = std.at[sl].set(jnp.maximum(std[sl] * jnp.exp(g), 1e-6))
        params[0] = norm.replace(mean=mean, std=std)
    if engage_kernel_uv is not None:
        u_np, v_np = engage_kernel_uv
        policy = copy.deepcopy(params[1])
        pp0 = policy.get("params", policy)
        pp = dict(pp0)
        layer = dict(pp["hidden_0"])
        kernel = layer["kernel"]
        u = jnp.asarray(u_np, dtype=kernel.dtype)
        v = jnp.asarray(v_np, dtype=kernel.dtype)
        if u.ndim != 2 or u.shape[1] != 8:
            raise ValueError(f"engage kernel u shape {u.shape} must be (rank, 8)")
        if v.ndim != 2 or v.shape[0] != u.shape[0] or v.shape[1] != kernel.shape[1]:
            raise ValueError(f"engage kernel v shape {v.shape} incompatible with u {u.shape} and kernel {kernel.shape}")
        delta = jnp.einsum("ri,rh->ih", u, v)
        layer["kernel"] = kernel.at[sl, :].add(delta)
        pp["hidden_0"] = layer
        if "params" in policy:
            policy = {**policy, "params": pp}
        else:
            policy = pp
        params[1] = policy
    if output_kernel_uv is not None:
        u_np, v_np = output_kernel_uv
        policy = copy.deepcopy(params[1])
        pp0 = policy.get("params", policy)
        pp = dict(pp0)
        out = _output_layer_name(pp)
        layer = dict(pp[out])
        kernel = layer["kernel"]
        act_dim = layer["bias"].shape[0] // 2
        u = jnp.asarray(u_np, dtype=kernel.dtype)
        v = jnp.asarray(v_np, dtype=kernel.dtype)
        if u.ndim != 2 or u.shape[1] != kernel.shape[0]:
            raise ValueError(f"output kernel u shape {u.shape} incompatible with kernel {kernel.shape}")
        if v.ndim != 2 or v.shape[0] != u.shape[0] or v.shape[1] != act_dim:
            raise ValueError(f"output kernel v shape {v.shape} incompatible with act_dim {act_dim}")
        delta = jnp.einsum("rh,ra->ha", u, v)
        layer["kernel"] = kernel.at[:, :act_dim].add(delta)
        pp[out] = layer
        if "params" in policy:
            policy = {**policy, "params": pp}
        else:
            policy = pp
        params[1] = policy
    return tuple(params)


def _seed_gate_summary(seed_vals) -> dict:
    if seed_vals is None:
        return {}
    rows = [{k: float(row[i]) for i, k in enumerate(ta.BENCH_KEYS)} for row in seed_vals]
    seed_judges = [
        100.0 * r["win_rate"]
        + r["sparc"]
        + 20.0 * (r["dealt"] - r["taken"])
        - 10.0 * max(0.0, r["ac_idle"] - 0.3)
        for r in rows
    ]
    return {
        "seed_min_sparc": float(min(r["sparc"] for r in rows)),
        "seed_min_judge": float(min(seed_judges)),
        "seed_min_dealt": float(min(r["dealt"] for r in rows)),
        "seed_min_margin": float(min(r["dealt"] - r["taken"] for r in rows)),
        "seed_min_survival": float(min(r["survival_rate"] for r in rows)),
        "seed_min_safe": float(min(r["safe_rate"] for r in rows)),
        "seed_max_peak_pen": float(max(r["ac_peak_pen"] for r in rows)),
        "seed_max_early": float(max(r["ac_dmg_early"] for r in rows)),
        "seed_rows": rows,
    }


def score_record(vals, *, min_dealt: float, max_peak_pen: float,
                 max_early: float, keep_metric: str, seed_vals=None,
                 per_seed_gates: bool = False,
                 per_seed_min_dealt: float = 0.0,
                 min_survival: float = 1.0,
                 min_safe: float = 1.0) -> tuple[float, dict]:
    rec = {k: float(vals[i]) for i, k in enumerate(ta.BENCH_KEYS)}
    rec.update(_seed_gate_summary(seed_vals))
    rec["bench_ratio"] = rec["dealt"] / max(rec["taken"], 1e-6)
    rec["bench_margin"] = rec["dealt"] - rec["taken"]
    rec["idle_penalty"] = 10.0 * max(0.0, rec["ac_idle"] - 0.3)
    rec["bench_judge"] = (
        100.0 * rec["win_rate"]
        + rec["sparc"]
        + 20.0 * rec["bench_margin"]
        - rec["idle_penalty"]
    )
    valid = (
        rec["dealt"] >= min_dealt
        and rec["ac_peak_pen"] <= max_peak_pen
        and rec["ac_dmg_early"] <= max_early
        and rec["survival_rate"] >= min_survival
        and rec["safe_rate"] >= min_safe
    )
    if per_seed_gates:
        valid = valid and (
            rec["seed_min_dealt"] >= per_seed_min_dealt
            and rec["seed_min_survival"] >= min_survival
            and rec["seed_min_safe"] >= min_safe
            and rec["seed_max_peak_pen"] <= max_peak_pen
            and rec["seed_max_early"] <= max_early
        )
    if keep_metric == "sparc":
        raw = rec["sparc"]
    elif keep_metric == "judge":
        raw = rec["bench_judge"]
    elif keep_metric == "margin":
        raw = rec["bench_margin"]
    elif keep_metric == "dealt":
        raw = rec["dealt"]
    elif keep_metric == "min_dealt":
        raw = rec["seed_min_dealt"]
    elif keep_metric == "min_margin":
        raw = rec["seed_min_margin"]
    elif keep_metric == "min_sparc":
        raw = rec["seed_min_sparc"]
    elif keep_metric == "min_judge":
        raw = rec["seed_min_judge"]
    else:
        raise ValueError(keep_metric)
    score = raw if valid else -1e9 + raw
    rec["valid"] = bool(valid)
    rec["selected_score"] = float(score)
    return float(score), rec


def build_multiseed_benchmark(bench_env, n_epis: int, steps: int, seeds: list[int],
                              deterministic: bool = True):
    """Build one compiled benchmark that returns aggregate and per-seed rows.

    `train_adversarial.build_benchmark` is convenient, but constructing one jitted
    benchmark per seed makes search iteration spend minutes compiling before any
    candidate score. This keeps the same metrics and gates while batching seeds
    inside one jitted function.
    """
    from brax.training.acme import running_statistics
    from brax.training.agents.ppo import networks as ppo_networks

    net = ppo_networks.make_ppo_networks(
        bench_env.observation_size,
        bench_env.action_size,
        preprocess_observations_fn=running_statistics.normalize,
    )
    make_inf = ppo_networks.make_inference_fn(net)
    seed_keys = jnp.stack(
        [jax.random.split(jax.random.PRNGKey(int(seed)), n_epis) for seed in seeds],
        axis=0,
    )

    @jax.jit
    def bench(params):
        inf = make_inf(params, deterministic=deterministic)

        def ep(k):
            st = bench_env.reset(k)
            d0 = jnp.linalg.norm(
                (st.pipeline_state.xpos[bench_env._Bt] - st.pipeline_state.xpos[bench_env._At])[:2]
            )

            def stp(carry, _):
                s, key, t = carry
                key, sk = jax.random.split(key)
                a, _ = inf(s.obs, sk)
                s = bench_env.step(s, a)
                alive = 1.0 - s.done
                m = s.metrics
                sat = jnp.mean(jnp.abs(a[:bench_env._n_hinge]) > 0.95)
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
                tz = ps.xpos[bench_env._At][2]
                up_a = ps.xmat[bench_env._At].reshape(-1)[8]
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
                (tz_c > bench_env._airborne_z).mean(),
                pen_c.max(),
                idle_c.mean(),
                jnp.where(has_dmg, (dl_c * (t_c < 0.15 * steps)).sum() / tot, 0.0),
                jnp.where(has_dmg, (dl_c * (up_c > 0.5)).sum() / tot, 1.0),
                jnp.where(has_dmg, (dl_c * (tz_c < bench_env._grounded_z)).sum() / tot, 1.0),
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


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--tag", default="policy_bias_search")
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--bench-sep-lo", type=float, default=0.25)
    ap.add_argument("--bench-sep-hi", type=float, default=0.70)
    ap.add_argument("--bench-az", type=float, default=3.14159)
    ap.add_argument("--bench-epis", type=int, default=8)
    ap.add_argument("--bench-steps", type=int, default=80)
    ap.add_argument("--bench-seeds", default="20240601")
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
    ap.add_argument("--opponent", choices=["passive", "frozen"], default="passive")
    ap.add_argument("--opp-ckpt", default="")
    ap.add_argument("--gens", type=int, default=6)
    ap.add_argument("--pop", type=int, default=24)
    ap.add_argument("--elite", type=int, default=6)
    ap.add_argument("--sigma", type=float, default=0.15)
    ap.add_argument("--min-sigma", type=float, default=0.02)
    ap.add_argument("--max-abs", type=float, default=0.8)
    ap.add_argument("--seed", type=int, default=123)
    ap.add_argument("--search-strike", action="store_true",
                    help="also search pneumatic striker action bias; default searches hinge actions only")
    ap.add_argument("--search-engage-norm", action="store_true",
                    help="also search shift/gain for the last 8 engage observation normalizer features")
    ap.add_argument("--engage-start", type=int, default=44,
                    help="start index of engage tail in observations; default is LOCO_OBS(38)+opponent(6)")
    ap.add_argument("--norm-shift-sigma", type=float, default=0.20,
                    help="initial CEM std for engage normalizer mean shifts, in normalized units")
    ap.add_argument("--norm-logstd-sigma", type=float, default=0.20,
                    help="initial CEM std for engage normalizer log-std deltas")
    ap.add_argument("--norm-shift-max", type=float, default=1.5,
                    help="absolute clamp for engage normalizer mean shifts, in normalized units")
    ap.add_argument("--norm-logstd-max", type=float, default=1.0,
                    help="absolute clamp for engage normalizer log-std deltas")
    ap.add_argument("--search-engage-kernel-rank", type=int, default=0,
                    help="rank of low-rank delta added to hidden_0 input weights for engage features")
    ap.add_argument("--engage-kernel-sigma", type=float, default=0.05,
                    help="initial CEM std for low-rank engage kernel factors")
    ap.add_argument("--engage-kernel-max", type=float, default=0.25,
                    help="absolute clamp for low-rank engage kernel factors")
    ap.add_argument("--search-output-kernel-rank", type=int, default=0,
                    help="rank of low-rank delta added to final policy mean output kernel")
    ap.add_argument("--output-kernel-sigma", type=float, default=0.05,
                    help="initial CEM std for low-rank output mean kernel factors")
    ap.add_argument("--output-kernel-max", type=float, default=0.25,
                    help="absolute clamp for low-rank output mean kernel factors")
    ap.add_argument("--keep-metric", choices=["sparc", "judge", "margin", "dealt", "min_dealt",
                                              "min_margin", "min_sparc", "min_judge"],
                    default="sparc")
    ap.add_argument("--min-dealt", type=float, default=0.8)
    ap.add_argument("--max-peak-pen", type=float, default=0.05)
    ap.add_argument("--max-early", type=float, default=0.5)
    ap.add_argument("--min-survival", type=float, default=1.0)
    ap.add_argument("--min-safe", type=float, default=1.0)
    ap.add_argument("--per-seed-gates", action="store_true",
                    help="require each benchmark seed to pass survival/safe/penetration/early-damage gates")
    ap.add_argument("--per-seed-min-dealt", type=float, default=0.0,
                    help="minimum dealt damage required for each seed when --per-seed-gates is set")
    args = ap.parse_args()

    ckpt = Path(args.ckpt)
    base_params = pickle.load(open(ckpt, "rb"))
    act_dim = infer_action_dim(base_params)
    first_hidden = infer_first_hidden_dim(base_params)
    output_input_dim = infer_output_input_dim(base_params)
    action_search_dim = act_dim if args.search_strike else max(1, act_dim - 2)
    norm_dim = 16 if args.search_engage_norm else 0
    engage_rank = max(0, int(args.search_engage_kernel_rank))
    engage_kernel_dim = engage_rank * (8 + first_hidden)
    output_rank = max(0, int(args.search_output_kernel_rank))
    output_kernel_dim = output_rank * (output_input_dim + act_dim)
    search_dim = action_search_dim + norm_dim + engage_kernel_dim + output_kernel_dim
    seeds = [int(x) for x in str(args.bench_seeds).split(",") if x.strip()]
    opp = None
    if args.opponent == "frozen":
        if not args.opp_ckpt:
            raise SystemExit("--opponent frozen requires --opp-ckpt")
        opp = ta.load_opponent(args.opp_ckpt)
    if any(x is not None for x in (
        args.striker_rod_len,
        args.striker_stroke,
        args.striker_rod_radius,
        args.contact_solref_timeconst,
        args.floor_calf_solref_timeconst,
    )) or args.disable_calf_floor:
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
    env = ta.AdversarialEnv(
        self_collision=not args.lean_contacts,
        frame_skip=args.frame_skip,
        sep_lo=args.bench_sep_lo,
        sep_hi=args.bench_sep_hi,
        azimuth=args.bench_az,
        striker=None,
        opponent=args.opponent,
        opp_infer=opp,
        engage_obs=args.engage_obs,
        contact_obs=args.contact_obs,
    )
    bench = build_multiseed_benchmark(env, args.bench_epis, args.bench_steps, seeds)

    def decode(vec: np.ndarray) -> tuple[
        np.ndarray,
        np.ndarray | None,
        np.ndarray | None,
        tuple[np.ndarray, np.ndarray] | None,
        tuple[np.ndarray, np.ndarray] | None,
    ]:
        delta = np.zeros(act_dim, dtype=np.float32)
        delta[:action_search_dim] = vec[:action_search_dim]
        obs_shift = None
        obs_log_std = None
        uv = None
        off = action_search_dim
        if args.search_engage_norm:
            obs_shift = vec[off:off + 8].astype(np.float32)
            obs_log_std = vec[off + 8:off + 16].astype(np.float32)
            off += 16
        if engage_rank > 0:
            u = vec[off:off + engage_rank * 8].reshape(engage_rank, 8).astype(np.float32)
            off += engage_rank * 8
            v = vec[off:off + engage_rank * first_hidden].reshape(engage_rank, first_hidden).astype(np.float32)
            uv = (u, v)
            off += engage_rank * first_hidden
        out_uv = None
        if output_rank > 0:
            u = vec[off:off + output_rank * output_input_dim].reshape(output_rank, output_input_dim).astype(np.float32)
            off += output_rank * output_input_dim
            v = vec[off:off + output_rank * act_dim].reshape(output_rank, act_dim).astype(np.float32)
            out_uv = (u, v)
        return delta, obs_shift, obs_log_std, uv, out_uv

    def evaluate(vec: np.ndarray) -> tuple[float, dict]:
        delta_full, obs_shift, obs_log_std, uv, out_uv = decode(vec)
        params = with_adapter_delta(
            base_params,
            delta_full,
            obs_shift,
            obs_log_std,
            uv,
            out_uv,
            engage_start=args.engage_start,
        )
        vals, vals_by_seed = bench(params)
        vals = np.asarray(vals)
        vals_by_seed = np.asarray(vals_by_seed)
        return score_record(
            vals,
            min_dealt=args.min_dealt,
            max_peak_pen=args.max_peak_pen,
            max_early=args.max_early,
            keep_metric=args.keep_metric,
            seed_vals=vals_by_seed,
            per_seed_gates=args.per_seed_gates,
            per_seed_min_dealt=args.per_seed_min_dealt,
            min_survival=args.min_survival,
            min_safe=args.min_safe,
        )

    rng = np.random.default_rng(args.seed)
    mean = np.zeros(search_dim, dtype=np.float32)
    sigma = np.full(search_dim, args.sigma, dtype=np.float32)
    max_abs = np.full(search_dim, args.max_abs, dtype=np.float32)
    if args.search_engage_norm:
        sigma[action_search_dim:action_search_dim + 8] = args.norm_shift_sigma
        sigma[action_search_dim + 8:action_search_dim + 16] = args.norm_logstd_sigma
        max_abs[action_search_dim:action_search_dim + 8] = args.norm_shift_max
        max_abs[action_search_dim + 8:action_search_dim + 16] = args.norm_logstd_max
    if engage_kernel_dim:
        k0 = action_search_dim + norm_dim
        sigma[k0:k0 + engage_kernel_dim] = args.engage_kernel_sigma
        max_abs[k0:k0 + engage_kernel_dim] = args.engage_kernel_max
    if output_kernel_dim:
        k0 = action_search_dim + norm_dim + engage_kernel_dim
        sigma[k0:k0 + output_kernel_dim] = args.output_kernel_sigma
        max_abs[k0:k0 + output_kernel_dim] = args.output_kernel_max
    best_score = -1e30
    best_vec = np.zeros(search_dim, dtype=np.float32)
    best_delta = np.zeros(act_dim, dtype=np.float32)
    best_obs_shift = None
    best_obs_log_std = None
    best_engage_kernel_u = None
    best_engage_kernel_v = None
    best_output_kernel_u = None
    best_output_kernel_v = None
    best_rec: dict = {}
    hist = []

    base_score, base_rec = evaluate(np.zeros(search_dim, dtype=np.float32))
    best_score, best_rec = base_score, {**base_rec, "gen": -1, "rank": 0}
    print(
        f"[bias-search] base score={base_score:.3f} sparc={base_rec['sparc']:.3f} "
        f"dealt={base_rec['dealt']:.3f} taken={base_rec['taken']:.3f} "
        f"close={base_rec['closing']:.3f} flee={base_rec['fleeing']:.3f} "
        f"pen={base_rec['ac_peak_pen']:.4f} valid={base_rec['valid']}",
        flush=True,
    )

    for gen in range(args.gens):
        samples = rng.normal(mean, sigma, size=(args.pop, search_dim)).astype(np.float32)
        samples = np.clip(samples, -max_abs[None, :], max_abs[None, :])
        samples[0] = mean
        if gen == 0:
            samples[0] = 0.0
        rows = []
        for i, sample in enumerate(samples):
            score, rec = evaluate(sample)
            rows.append((score, sample.copy(), rec))
            if score > best_score:
                best_score = score
                best_vec = sample.copy()
                best_delta, best_obs_shift, best_obs_log_std, best_uv, best_out_uv = decode(best_vec)
                if best_uv is None:
                    best_engage_kernel_u = None
                    best_engage_kernel_v = None
                else:
                    best_engage_kernel_u, best_engage_kernel_v = best_uv
                if best_out_uv is None:
                    best_output_kernel_u = None
                    best_output_kernel_v = None
                else:
                    best_output_kernel_u, best_output_kernel_v = best_out_uv
                best_rec = {**rec, "gen": gen, "rank": i}
                out_ckpt = OUT / f"{args.tag}_best.pkl"
                pickle.dump(
                    with_adapter_delta(
                        base_params,
                        best_delta,
                        best_obs_shift,
                        best_obs_log_std,
                        None if best_engage_kernel_u is None else (best_engage_kernel_u, best_engage_kernel_v),
                        None if best_output_kernel_u is None else (best_output_kernel_u, best_output_kernel_v),
                        engage_start=args.engage_start,
                    ),
                    open(out_ckpt, "wb"),
                )
        rows.sort(key=lambda x: x[0], reverse=True)
        elites = np.stack([r[1] for r in rows[: max(1, args.elite)]], axis=0)
        mean = elites.mean(axis=0)
        sigma = np.maximum(elites.std(axis=0) * 0.9, args.min_sigma)
        top = rows[0][2]
        row = {
            "gen": gen,
            "best_score": float(best_score),
            "gen_score": float(rows[0][0]),
            "sparc": top["sparc"],
            "dealt": top["dealt"],
            "taken": top["taken"],
            "margin": top["bench_margin"],
            "closing": top["closing"],
            "fleeing": top["fleeing"],
            "peak_pen": top["ac_peak_pen"],
            "seed_min_sparc": top.get("seed_min_sparc"),
            "seed_min_judge": top.get("seed_min_judge"),
            "seed_min_survival": top.get("seed_min_survival"),
            "seed_min_safe": top.get("seed_min_safe"),
            "seed_max_peak_pen": top.get("seed_max_peak_pen"),
            "seed_max_early": top.get("seed_max_early"),
            "valid": top["valid"],
            "sigma_mean": float(sigma.mean()),
        }
        hist.append(row)
        print(
            f"[bias-search] gen={gen:02d} score={row['gen_score']:.3f} "
            f"sparc={row['sparc']:.3f} margin={row['margin']:+.3f} "
            f"close={row['closing']:.3f} flee={row['fleeing']:.3f} "
            f"pen={row['peak_pen']:.4f} valid={row['valid']} "
            f"best={best_score:.3f}",
            flush=True,
        )

    if best_score <= base_score:
        pickle.dump(base_params, open(OUT / f"{args.tag}_best.pkl", "wb"))
        best_vec = np.zeros(search_dim, dtype=np.float32)
        best_delta = np.zeros(act_dim, dtype=np.float32)
        best_obs_shift = None
        best_obs_log_std = None
        best_engage_kernel_u = None
        best_engage_kernel_v = None
        best_output_kernel_u = None
        best_output_kernel_v = None
        best_rec = {**base_rec, "gen": -1, "rank": 0}

    report = {
        "tag": args.tag,
        "ckpt": str(ckpt),
        "act_dim": act_dim,
        "first_hidden": first_hidden,
        "output_input_dim": output_input_dim,
        "action_search_dim": action_search_dim,
        "search_dim": search_dim,
        "search_engage_norm": bool(args.search_engage_norm),
        "search_engage_kernel_rank": engage_rank,
        "search_output_kernel_rank": output_rank,
        "engage_start": args.engage_start,
        "bench_seeds": seeds,
        "per_seed_gates": bool(args.per_seed_gates),
        "base": base_rec,
        "best": best_rec,
        "best_vector": best_vec.tolist(),
        "best_delta": best_delta.tolist(),
        "best_obs_shift": None if best_obs_shift is None else best_obs_shift.tolist(),
        "best_obs_log_std_delta": None if best_obs_log_std is None else best_obs_log_std.tolist(),
        "best_engage_kernel_u": None if best_engage_kernel_u is None else best_engage_kernel_u.tolist(),
        "best_engage_kernel_v": None if best_engage_kernel_v is None else best_engage_kernel_v.tolist(),
        "best_output_kernel_u": None if best_output_kernel_u is None else best_output_kernel_u.tolist(),
        "best_output_kernel_v": None if best_output_kernel_v is None else best_output_kernel_v.tolist(),
        "history": hist,
        "artifact": str(OUT / f"{args.tag}_best.pkl"),
    }
    out_json = OUT / f"{args.tag}_policy_bias_search.json"
    out_json.write_text(json.dumps(report, indent=2))
    print(f"[bias-search] saved {out_json}", flush=True)


if __name__ == "__main__":
    main()
