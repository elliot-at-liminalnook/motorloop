# SPDX-License-Identifier: MIT
"""Diagnose why a fighter policy reaches proximity without scoring contact."""

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


def _make_infer(params, obs_dim: int, act_dim: int):
    from brax.training.acme import running_statistics
    from brax.training.agents.ppo import networks as ppo_networks

    net = ppo_networks.make_ppo_networks(
        obs_dim, act_dim, preprocess_observations_fn=running_statistics.normalize
    )
    return ppo_networks.make_inference_fn(net)(params, deterministic=True)


def _as_float(x) -> float:
    return float(np.asarray(x))


def _as_int(x) -> int:
    return int(np.asarray(x))


def _contact_rows(env: ta.AdversarialEnv, ps, top_k: int, min_pen: float) -> list[dict]:
    pen = np.maximum(0.0, -np.asarray(ps.contact.dist))
    geom = np.asarray(ps.contact.geom)
    if pen.size == 0 or geom.size == 0:
        return []
    order = np.argsort(-pen)
    rows = []
    for ci in order[:top_k]:
        p = float(pen[ci])
        if p < min_pen:
            continue
        g0 = int(geom[ci, 0])
        g1 = int(geom[ci, 1])
        if g0 < 0 or g1 < 0:
            continue
        b0 = int(env._geom_body[g0])
        b1 = int(env._geom_body[g1])
        rows.append({
            "penetration": p,
            "geom0": env._geom_names[g0],
            "geom1": env._geom_names[g1],
            "body0": env._body_names[b0],
            "body1": env._body_names[b1],
            "geom0_id": g0,
            "geom1_id": g1,
        })
    return rows


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--seed", type=int, required=True)
    ap.add_argument("--episodes", type=int, default=8)
    ap.add_argument("--steps", type=int, default=80)
    ap.add_argument("--sep-lo", type=float, default=0.25)
    ap.add_argument("--sep-hi", type=float, default=0.70)
    ap.add_argument("--azimuth", type=float, default=3.14159)
    ap.add_argument("--lean-contacts", action="store_true")
    ap.add_argument("--engage-obs", action="store_true")
    ap.add_argument("--contact-obs", action="store_true")
    ap.add_argument("--striker-rod-len", type=float, default=None)
    ap.add_argument("--striker-stroke", type=float, default=None)
    ap.add_argument("--striker-rod-radius", type=float, default=None)
    ap.add_argument("--floor-calf-solref-timeconst", type=float, default=None)
    ap.add_argument("--floor-calf-solref-dampratio", type=float, default=1.0)
    ap.add_argument("--disable-calf-floor", action="store_true")
    ap.add_argument("--top-k", type=int, default=5)
    ap.add_argument("--contact-min-pen", type=float, default=1e-4)
    ap.add_argument("--out-json", default="")
    args = ap.parse_args()

    if any(v is not None for v in (
        args.striker_rod_len,
        args.striker_stroke,
        args.striker_rod_radius,
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
        if args.floor_calf_solref_timeconst is not None:
            ta.SPEC["contact"]["floor_calf_solref"] = [
                float(args.floor_calf_solref_timeconst),
                float(args.floor_calf_solref_dampratio),
            ]
        if args.disable_calf_floor:
            ta.SPEC["contact"]["calf_floor"] = False

    env = ta.AdversarialEnv(
        self_collision=not args.lean_contacts,
        frame_skip=5,
        sep_lo=args.sep_lo,
        sep_hi=args.sep_hi,
        azimuth=args.azimuth,
        striker=None,
        opponent="passive",
        engage_obs=args.engage_obs,
        contact_obs=args.contact_obs,
    )
    params = ta.warm_start(args.ckpt, env.observation_size, env.action_size)
    if params is None:
        params = pickle.load(open(args.ckpt, "rb"))
    infer = _make_infer(params, env.observation_size, env.action_size)
    step = jax.jit(env.step)
    keys = jax.random.split(jax.random.PRNGKey(args.seed), args.episodes)
    rows = []
    for epi, key0 in enumerate(keys):
        key = key0
        state = env.reset(key)
        rel0 = (state.pipeline_state.xpos[env._Bt] - state.pipeline_state.xpos[env._At])[:2]
        step_rows = []
        contact_events = []
        contact_pair_stats: dict[str, dict] = {}
        max_pen = 0.0
        max_pen_step = None
        first_done_step = None
        sums = {k: 0.0 for k in ("sparc", "dealt", "taken", "closing", "fleeing", "fire")}
        first_dealt_step = None
        for t in range(args.steps):
            key, sk = jax.random.split(key)
            action, _ = infer(state.obs, sk)
            state = step(state, action)
            ps = state.pipeline_state
            rel = (ps.xpos[env._Bt] - ps.xpos[env._At])[:2]
            dist = jnp.linalg.norm(rel)
            unit = rel / (dist + 1e-6)
            rmat = ps.xmat[env._At].reshape(-1)
            forward = rmat[:2]
            side = rmat[3:5]
            forward = forward / (jnp.linalg.norm(forward) + 1e-6)
            side = side / (jnp.linalg.norm(side) + 1e-6)
            fwd_dot = jnp.dot(forward, unit)
            lat_dot = jnp.dot(side, unit)
            rod_pos = ps.geom_xpos[env._Arod_gids]
            rod_dists = jnp.linalg.norm(rod_pos - ps.xpos[env._Bt], axis=1)
            min_rod = jnp.min(rod_dists)
            min_rod_idx = jnp.argmin(rod_dists)
            strike = jnp.clip(action[env._strike_local], 0.0, 1.0)
            fire_mean = jnp.mean(strike)
            fire_minrod = strike[min_rod_idx]
            alive = 1.0 - _as_float(state.done)
            if first_done_step is None and alive <= 0.0:
                first_done_step = t + 1
            for k in sums:
                sums[k] += _as_float(state.metrics[k]) * alive
            dealt = _as_float(state.metrics["dealt"]) * alive
            if first_dealt_step is None and dealt > 1e-6:
                first_dealt_step = t + 1
            contacts = _contact_rows(env, ps, args.top_k, args.contact_min_pen)
            if contacts:
                step_max = contacts[0]["penetration"]
                if step_max > max_pen:
                    max_pen = step_max
                    max_pen_step = t + 1
                for c in contacts:
                    pair = f"{c['geom0']}|{c['geom1']}"
                    stat = contact_pair_stats.setdefault(pair, {
                        "geom0": c["geom0"],
                        "geom1": c["geom1"],
                        "body0": c["body0"],
                        "body1": c["body1"],
                        "max_penetration": 0.0,
                        "count": 0,
                    })
                    stat["count"] += 1
                    stat["max_penetration"] = max(stat["max_penetration"], c["penetration"])
                if len(contact_events) < args.top_k * 4 or step_max >= 0.045:
                    contact_events.append({
                        "step": t + 1,
                        "top_contacts": contacts,
                    })
            step_rows.append({
                "step": t + 1,
                "dist": _as_float(dist),
                "min_rod_dist": _as_float(min_rod),
                "min_rod_index": _as_int(min_rod_idx),
                "front_dot": _as_float(fwd_dot),
                "side_dot": _as_float(lat_dot),
                "fire_mean": _as_float(fire_mean),
                "fire_min_rod": _as_float(fire_minrod),
                "strike_action": [float(x) for x in np.asarray(strike)],
                "dealt": dealt,
                "taken": _as_float(state.metrics["taken"]) * alive,
                "closing": _as_float(state.metrics["closing"]) * alive,
                "fleeing": _as_float(state.metrics["fleeing"]) * alive,
            })
        by_body = sorted(step_rows, key=lambda r: r["dist"])[: args.top_k]
        by_rod = sorted(step_rows, key=lambda r: r["min_rod_dist"])[: args.top_k]
        close_rows = [r for r in step_rows if r["dist"] < 0.40]
        rows.append({
            "episode": epi,
            "sep0": _as_float(jnp.linalg.norm(rel0)),
            "theta0": _as_float(jnp.arctan2(rel0[1], rel0[0])),
            "first_done_step": first_done_step,
            "first_dealt_step": first_dealt_step,
            "sum": sums,
            "min_dist": min(r["dist"] for r in step_rows),
            "min_rod_dist": min(r["min_rod_dist"] for r in step_rows),
            "max_penetration": max_pen,
            "max_penetration_step": max_pen_step,
            "top_contact_pairs": sorted(
                contact_pair_stats.values(),
                key=lambda r: (r["max_penetration"], r["count"]),
                reverse=True,
            )[: args.top_k],
            "contact_events": contact_events[: args.top_k * 8],
            "max_front_dot": max(r["front_dot"] for r in step_rows),
            "min_abs_side_dot": min(abs(r["side_dot"]) for r in step_rows),
            "max_fire_mean": max(r["fire_mean"] for r in step_rows),
            "mean_fire_close": float(np.mean([r["fire_mean"] for r in close_rows])) if close_rows else 0.0,
            "top_by_body_dist": by_body,
            "top_by_rod_dist": by_rod,
        })
    rec = {
        "ckpt": args.ckpt,
        "seed": args.seed,
        "episodes": args.episodes,
        "steps": args.steps,
        "lean_contacts": bool(args.lean_contacts),
        "engage_obs": bool(args.engage_obs),
        "contact_obs": bool(args.contact_obs),
        "striker": {
            "rod_len": ta.SPEC.get("striker", {}).get("rod_len"),
            "stroke": ta.SPEC.get("striker", {}).get("stroke"),
            "rod_radius": ta.SPEC.get("striker", {}).get("rod_radius"),
            "calf_floor": ta.SPEC.get("contact", {}).get("calf_floor", True),
            "floor_calf_solref": ta.SPEC.get("contact", {}).get("floor_calf_solref"),
        },
        "rows": rows,
    }
    text = json.dumps(rec, indent=2)
    print(text)
    if args.out_json:
        Path(args.out_json).write_text(text)


if __name__ == "__main__":
    os.environ.pop("MUJOCO_GL", None)
    main()
