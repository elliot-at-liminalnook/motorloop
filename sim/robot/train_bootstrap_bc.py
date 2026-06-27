# SPDX-License-Identifier: MIT
"""Behavior-clone the CPG gait library.

The dataset contains `(obs, action)` pairs produced by the CPG teacher.  This
trainer learns a compact command-conditioned policy:

    obs + command -> clipped CPG/PD motor action

Use `CMD_CONTROL_MODE=pd` for rollout evaluation so the cloned action is treated
as the motor action, not as a residual on top of another CPG prior.
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
import optax

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

OUT = Path(os.environ.get("CODESIGN_OUT", "/root/proj/out"))
TRANSITION_FIXED_EXTRA = 6  # prev_cmd(2), cmd_delta(2), prior_strength(1), active(1)
TRANSITION_PHASE_EXTRA = 2  # sin(phase), cos(phase)


def parse_sizes(text: str) -> tuple[int, ...]:
    return tuple(int(x.strip()) for x in text.replace(";", ",").split(",") if x.strip())


def dataset_files(dataset: Path) -> list[Path]:
    if dataset.is_file():
        return [dataset]
    return sorted(p for p in dataset.glob("*.npz") if p.name != "manifest.npz")


def augment_obs(obs: np.ndarray, action: np.ndarray, data, feature_mode: str) -> np.ndarray:
    if feature_mode == "base":
        return obs
    if feature_mode != "transition":
        raise ValueError(f"unknown --feature-mode {feature_mode}")
    n = obs.shape[0]
    cmd = np.asarray(data["command"], dtype=np.float32) if "command" in data else obs[:, -2:]
    if cmd.ndim == 1:
        cmd = np.repeat(cmd[None, :], n, axis=0)
    prev_cmd = np.vstack([cmd[:1], cmd[:-1]])
    cmd_delta = cmd - prev_cmd
    prev_action = np.vstack([np.zeros_like(action[:1]), action[:-1]])
    prior_strength = np.clip(np.linalg.norm(cmd, axis=1, keepdims=True) / 0.35, 0.0, 1.0).astype(np.float32)
    active = np.asarray(data["active"], dtype=np.float32).reshape(-1, 1) if "active" in data else np.zeros((n, 1), dtype=np.float32)
    phase = np.asarray(data["phase"], dtype=np.float32).reshape(-1) if "phase" in data else np.zeros((n,), dtype=np.float32)
    phase_feat = np.stack([np.sin(phase), np.cos(phase)], axis=1).astype(np.float32)
    return np.concatenate([obs, prev_cmd, cmd_delta, prev_action, prior_strength, active, phase_feat], axis=1)


def sample_weights(data, n: int, mode: str, positive_w: float, bad_w: float,
                   waypoint2_w: float, drift_w: float) -> np.ndarray:
    w = np.ones((n,), dtype=np.float32)
    if mode == "uniform":
        return w
    if mode != "useful":
        raise ValueError(f"unknown --sample-weight-mode {mode}")
    cmd = np.asarray(data["command"], dtype=np.float32) if "command" in data else None
    if cmd is not None and cmd.ndim == 1:
        cmd = np.repeat(cmd[None, :], n, axis=0)
    moving = np.ones((n,), dtype=bool) if cmd is None else np.linalg.norm(cmd, axis=1) > 0.05
    if "distance_reduction" in data:
        dr = np.asarray(data["distance_reduction"], dtype=np.float32).reshape(-1)
        positive = np.clip(dr / 0.003, 0.0, 1.0)
        w *= np.where(moving, 1.0 + (positive_w - 1.0) * positive, 1.0)
        w *= np.where(moving & (dr < -0.001), bad_w, 1.0)
    if "active" in data:
        active = np.asarray(data["active"], dtype=np.int32).reshape(-1)
        w *= np.where(active == 2, waypoint2_w, 1.0)
    if "y_drift" in data and "active" in data:
        y_drift = np.asarray(data["y_drift"], dtype=np.float32).reshape(-1)
        active = np.asarray(data["active"], dtype=np.int32).reshape(-1)
        w *= np.where((active == 2) & (y_drift > 0.0), drift_w, 1.0)
    if "x_loss" in data:
        x_loss = np.asarray(data["x_loss"], dtype=np.float32).reshape(-1)
        w *= np.where(x_loss > 0.001, drift_w, 1.0)
    if "fall" in data:
        w *= np.where(np.asarray(data["fall"], dtype=np.float32).reshape(-1) > 0.5, bad_w, 1.0)
    if "high_saturation" in data:
        w *= np.where(np.asarray(data["high_saturation"], dtype=np.float32).reshape(-1) > 0.5, bad_w, 1.0)
    return np.clip(w, 0.05, 20.0).astype(np.float32)


def load_split(dataset: Path, val_frac: float, seed: int, alive_only: bool, feature_mode: str,
               weight_mode: str, positive_w: float, bad_w: float, waypoint2_w: float, drift_w: float):
    files = dataset_files(dataset)
    if not files:
        raise FileNotFoundError(f"no .npz dataset files under {dataset}")
    rng = np.random.default_rng(seed)
    order = np.arange(len(files))
    rng.shuffle(order)
    val_n = max(1, int(round(len(files) * val_frac))) if len(files) > 1 else 1
    val_idx = set(order[:val_n].tolist())
    train_files = [f for i, f in enumerate(files) if i not in val_idx] or [files[order[-1]]]
    val_files = [f for i, f in enumerate(files) if i in val_idx]

    def load_many(paths):
        xs, ys, ws = [], [], []
        for path in paths:
            d = np.load(path, allow_pickle=True)
            obs = np.asarray(d["obs"], dtype=np.float32)
            action = np.asarray(d["action"], dtype=np.float32)
            weights = sample_weights(d, obs.shape[0], weight_mode, positive_w, bad_w, waypoint2_w, drift_w)
            obs = augment_obs(obs, action, d, feature_mode)
            if alive_only and "alive" in d:
                mask = np.asarray(d["alive"]) > 0.5
                obs = obs[mask]
                action = action[mask]
                weights = weights[mask]
            xs.append(obs)
            ys.append(action)
            ws.append(weights)
        return np.concatenate(xs, axis=0), np.concatenate(ys, axis=0), np.concatenate(ws, axis=0)

    return load_many(train_files), load_many(val_files), train_files, val_files


def init_mlp(key, obs_dim: int, action_dim: int, hidden: tuple[int, ...]):
    sizes = (obs_dim,) + hidden + (action_dim,)
    params = []
    keys = jax.random.split(key, len(sizes) - 1)
    for k, din, dout in zip(keys, sizes[:-1], sizes[1:]):
        scale = np.sqrt(2.0 / max(din + dout, 1))
        params.append({
            "w": scale * jax.random.normal(k, (din, dout)),
            "b": jnp.zeros((dout,), dtype=jnp.float32),
        })
    return params


def apply_mlp(params, obs):
    x = obs
    for layer in params[:-1]:
        x = jnp.tanh(x @ layer["w"] + layer["b"])
    y = x @ params[-1]["w"] + params[-1]["b"]
    return jnp.tanh(y)


def corrupt_transition_context(xb: np.ndarray, base_dim: int, action_dim: int, rng,
                               cmd_jitter: float, prev_action_noise: float,
                               prev_action_dropout: float, stale_cmd_prob: float,
                               wrong_prev_action_prob: float, phase_jitter: float) -> np.ndarray:
    if xb.shape[1] < base_dim + 2 + 2 + action_dim + 2:
        return xb
    out = xb.copy()
    cmd_slice = slice(base_dim - 2, base_dim)
    prev_cmd_slice = slice(base_dim, base_dim + 2)
    cmd_delta_slice = slice(base_dim + 2, base_dim + 4)
    prev_action_slice = slice(base_dim + 4, base_dim + 4 + action_dim)
    phase_start = base_dim + 4 + action_dim + 2
    phase_slice = slice(phase_start, phase_start + 2)
    if cmd_jitter > 0.0:
        out[:, cmd_slice] += rng.normal(0.0, cmd_jitter, size=(len(out), 2)).astype(np.float32)
        out[:, prev_cmd_slice] += rng.normal(0.0, cmd_jitter, size=(len(out), 2)).astype(np.float32)
    if stale_cmd_prob > 0.0:
        stale = rng.random(len(out)) < stale_cmd_prob
        out[stale, cmd_slice] = out[stale, prev_cmd_slice]
    out[:, cmd_delta_slice] = out[:, cmd_slice] - out[:, prev_cmd_slice]
    if prev_action_noise > 0.0:
        out[:, prev_action_slice] += rng.normal(
            0.0, prev_action_noise, size=(len(out), action_dim)
        ).astype(np.float32)
    if wrong_prev_action_prob > 0.0:
        wrong = rng.random(len(out)) < wrong_prev_action_prob
        perm = rng.permutation(len(out))
        out[wrong, prev_action_slice] = out[perm[wrong], prev_action_slice]
    if prev_action_dropout > 0.0:
        drop = rng.random(len(out)) < prev_action_dropout
        out[drop, prev_action_slice] = 0.0
    if phase_jitter > 0.0 and out.shape[1] >= phase_start + 2:
        phase = np.arctan2(out[:, phase_slice.start], out[:, phase_slice.start + 1])
        phase = phase + rng.normal(0.0, phase_jitter, size=len(out)).astype(np.float32)
        out[:, phase_slice.start] = np.sin(phase)
        out[:, phase_slice.start + 1] = np.cos(phase)
    return out


def make_batcher(x, y, w, batch_size: int, seed: int, feature_mode: str,
                 base_dim: int, action_dim: int, cmd_jitter: float,
                 prev_action_noise: float, prev_action_dropout: float,
                 stale_cmd_prob: float, wrong_prev_action_prob: float,
                 phase_jitter: float):
    rng = np.random.default_rng(seed)
    n = len(x)
    while True:
        idx = rng.integers(0, n, size=batch_size)
        xb = x[idx]
        if feature_mode == "transition":
            xb = corrupt_transition_context(
                xb, base_dim, action_dim, rng, cmd_jitter,
                prev_action_noise, prev_action_dropout, stale_cmd_prob,
                wrong_prev_action_prob, phase_jitter,
            )
        yield jnp.asarray(xb), jnp.asarray(y[idx]), jnp.asarray(w[idx])


def weighted_mse(pred, y, w):
    per = jnp.mean((pred - y) ** 2, axis=-1)
    return jnp.sum(per * w) / jnp.maximum(jnp.sum(w), 1e-6)


def eval_loss(params, x, y, w, mean, std, batch_size=8192):
    losses = []
    for i in range(0, len(x), batch_size):
        xb = jnp.asarray(x[i:i + batch_size])
        yb = jnp.asarray(y[i:i + batch_size])
        wb = jnp.asarray(w[i:i + batch_size])
        pred = apply_mlp(params, (xb - mean) / std)
        losses.append(float(weighted_mse(pred, yb, wb)))
    return float(np.mean(losses)) if losses else float("nan")


def command_program(mode: str, hold: int, speed: float) -> np.ndarray:
    if mode == "forward":
        legs = [(speed, 0.0)]
    elif mode == "backward":
        legs = [(-abs(speed), 0.0)]
    elif mode == "left":
        legs = [(0.0, abs(speed))]
    elif mode == "right":
        legs = [(0.0, -abs(speed))]
    else:
        s = abs(speed)
        legs = [(s, 0.0), (0.0, s), (-s, 0.0), (0.0, -s), (0.0, 0.0)]
    return np.array([c for c in legs for _ in range(hold)], dtype=np.float32)


def summarize_rollout(rows: np.ndarray, total_steps: int):
    if rows.size == 0:
        return {"samples": 0, "total_steps": total_steps, "survived_full": False}
    cmd = rows[:, 1:3]
    vel = rows[:, 3:5]
    cmd_norm = np.linalg.norm(cmd, axis=1)
    vel_norm = np.linalg.norm(vel, axis=1)
    moving = cmd_norm > 1e-6
    align = np.sum(cmd[moving] * vel[moving], axis=1) / (cmd_norm[moving] * vel_norm[moving] + 1e-6) if moving.any() else []
    err = np.linalg.norm(vel - cmd, axis=1)
    return {
        "samples": int(len(rows)),
        "total_steps": int(total_steps),
        "survived_full": bool(len(rows) == total_steps),
        "mean_alignment": float(np.mean(align)) if len(align) else 0.0,
        "mean_vector_error": float(err[moving].mean()) if moving.any() else 0.0,
        "x_delta": float(rows[-1, 5] - rows[0, 5]),
        "y_delta": float(rows[-1, 6] - rows[0, 6]),
        "z_min": float(rows[:, 7].min()),
        "up_min": float(rows[:, 8].min()),
    }


def policy_obs(artifact: dict, obs, prev_cmd, prev_action, active=0.0, phase=None):
    if artifact.get("feature_mode", "base") != "transition":
        return obs
    cmd = obs[-2:]
    cmd_delta = cmd - prev_cmd
    prior_strength = jnp.asarray([jnp.clip(jnp.linalg.norm(cmd) / 0.35, 0.0, 1.0)], dtype=jnp.float32)
    active_arr = jnp.asarray([active], dtype=jnp.float32)
    parts = [obs, prev_cmd, cmd_delta, prev_action, prior_strength, active_arr]
    base_len = int(obs.shape[0] + prev_cmd.shape[0] + cmd_delta.shape[0] + prev_action.shape[0] + 2)
    want = int(artifact.get("obs_dim", base_len))
    if want >= base_len + TRANSITION_PHASE_EXTRA:
        ph = jnp.asarray(0.0 if phase is None else phase, dtype=jnp.float32)
        parts.append(jnp.asarray([jnp.sin(ph), jnp.cos(ph)], dtype=jnp.float32))
    out = jnp.concatenate(parts)
    if int(out.shape[0]) < want:
        out = jnp.pad(out, (0, want - int(out.shape[0])))
    if int(out.shape[0]) > want:
        out = out[:want]
    return out


def read_json(path: str | None) -> dict | None:
    if not path:
        return None
    p = Path(path)
    return json.loads(p.read_text()) if p.exists() else None


def promotion_result(nav: dict, gate: dict, baseline: dict | None, min_wp2_improvement: float) -> dict:
    if baseline is None:
        return {
            "baseline": None,
            "closed_loop_improved": bool(gate["checkpoint_route_success"]),
            "promote": bool(gate["ok"]),
            "reason": "no baseline supplied; full checkpoint success is required",
        }
    base_nav = baseline.get("nav", baseline)
    reached = int(nav.get("reached", 0))
    base_reached = int(base_nav.get("reached", 0))
    wp2 = float(nav.get("waypoint2_min_dist", float("inf")))
    base_wp2 = float(base_nav.get("waypoint2_min_dist", float("inf")))
    if not np.isfinite(base_wp2) and "closest_waypoints" in base_nav and len(base_nav["closest_waypoints"]) > 2:
        base_wp2 = float(base_nav["closest_waypoints"][2].get("min_dist", float("inf")))
    improved = reached > base_reached
    if reached == base_reached and np.isfinite(wp2) and np.isfinite(base_wp2):
        improved = improved or (base_wp2 - wp2 >= float(min_wp2_improvement))
    return {
        "baseline_reached": base_reached,
        "candidate_reached": reached,
        "baseline_waypoint2_min_dist": base_wp2,
        "candidate_waypoint2_min_dist": wp2,
        "min_waypoint2_improvement": float(min_wp2_improvement),
        "closed_loop_improved": bool(improved),
        "promote": bool(gate["fixed_direction_survival"] and improved),
        "reason": "promotion is based on closed-loop nav, not BC validation loss",
    }


def rollout_eval(artifact: dict, tag: str, modes: list[str], hold: int, speed: float, seed: int,
                 min_survival_frac: float, nav_radius: float, nav_gain: float, nav_steps: int,
                 promotion_baseline: dict | None = None, min_wp2_improvement: float = 0.0):
    os.environ.setdefault("CMD_CONTROL_MODE", "pd")
    from commanded_env import FALL_Z, MIN_UP_Z, VMAX, _build  # noqa: E402

    params = artifact["params"]
    mean = jnp.asarray(artifact["obs_mean"])
    std = jnp.asarray(artifact["obs_std"])
    Env = _build()
    env = Env()
    step = jax.jit(env.step)

    @jax.jit
    def policy(obs):
        return apply_mlp(params, (obs - mean) / std)

    summaries = {}
    key = jax.random.PRNGKey(seed)
    for mode in modes:
        cmds = command_program(mode, hold, speed)
        st = env.reset_with_command(key, cmds[0])
        prev_cmd = jnp.asarray(cmds[0])
        prev_action = jnp.zeros(env.action_size)
        rec = []
        for i, cmd in enumerate(cmds):
            st = st.replace(info={**st.info, "cmd": jnp.asarray(cmd), "remote": jnp.array(True)})
            st = st.replace(obs=st.obs.at[-2:].set(jnp.asarray(cmd)))
            po = policy_obs(artifact, st.obs, prev_cmd, prev_action, phase=st.info.get("phase", 0.0))
            act = policy(po)
            st = step(st, act)
            prev_cmd = jnp.asarray(cmd)
            prev_action = act
            dx = st.pipeline_state
            up = 1.0 - 2.0 * (float(dx.qpos[4]) ** 2 + float(dx.qpos[5]) ** 2)
            rec.append([i, float(cmd[0]), float(cmd[1]), float(dx.qvel[0]), float(dx.qvel[1]),
                        float(dx.qpos[0]), float(dx.qpos[1]), float(dx.qpos[2]), up])
            if float(dx.qpos[2]) < FALL_Z or up < MIN_UP_Z:
                break
        summaries[mode] = summarize_rollout(np.asarray(rec, dtype=np.float32), len(cmds))

    waypoints = np.asarray([[0.35, 0.0], [0.35, 0.35], [0.0, 0.35], [0.0, 0.0]], dtype=np.float32)
    st = env.reset_with_command(key, jnp.zeros(2))
    prev_cmd = jnp.zeros(2)
    prev_action = jnp.zeros(env.action_size)
    reached = 0
    nav_rec = []
    for t in range(nav_steps * len(waypoints)):
        pos = np.asarray(st.pipeline_state.qpos[:2])
        target = waypoints[min(reached, len(waypoints) - 1)]
        delta = target - pos
        dist = float(np.linalg.norm(delta))
        if dist < nav_radius and reached < len(waypoints):
            reached += 1
            if reached >= len(waypoints):
                break
            target = waypoints[reached]
            delta = target - pos
        cmd = nav_gain * delta
        n = float(np.linalg.norm(cmd))
        if n > VMAX:
            cmd = cmd * (VMAX / n)
        st = st.replace(info={**st.info, "cmd": jnp.asarray(cmd), "remote": jnp.array(True)})
        st = st.replace(obs=st.obs.at[-2:].set(jnp.asarray(cmd)))
        po = policy_obs(artifact, st.obs, prev_cmd, prev_action, float(reached), phase=st.info.get("phase", 0.0))
        act = policy(po)
        st = step(st, act)
        prev_cmd = jnp.asarray(cmd)
        prev_action = act
        dx = st.pipeline_state
        up = 1.0 - 2.0 * (float(dx.qpos[4]) ** 2 + float(dx.qpos[5]) ** 2)
        nav_rec.append([
            t, reached, float(target[0]), float(target[1]), float(cmd[0]), float(cmd[1]),
            float(dx.qpos[0]), float(dx.qpos[1]), float(dx.qpos[2]), up, dist,
        ])
        if float(dx.qpos[2]) < FALL_Z or up < MIN_UP_Z:
            break
    nav_arr = np.asarray(nav_rec, dtype=np.float32)
    closest = []
    if len(nav_arr):
        xy = nav_arr[:, 6:8]
        for i, wp in enumerate(waypoints):
            d = np.linalg.norm(xy - wp, axis=1)
            j = int(d.argmin())
            closest.append({
                "waypoint": int(i),
                "min_dist": float(d[j]),
                "closest_xy": xy[j].tolist(),
                "step": j,
                "inside_radius": bool(d[j] <= nav_radius),
            })
    nav = {
        "reached": int(reached),
        "total_waypoints": int(len(waypoints)),
        "success": bool(reached >= len(waypoints)),
        "samples": int(len(nav_rec)),
        "final_dist": float(nav_arr[-1, 10]) if len(nav_arr) else float("nan"),
        "z_min": float(nav_arr[:, 8].min()) if len(nav_arr) else float("nan"),
        "up_min": float(nav_arr[:, 9].min()) if len(nav_arr) else float("nan"),
        "closest_waypoints": closest,
        "waypoint2_min_dist": float(closest[2]["min_dist"]) if len(closest) > 2 else float("nan"),
    }
    survival_ok = all(v["samples"] / max(v["total_steps"], 1) >= min_survival_frac for v in summaries.values())
    gate = {
        "fixed_direction_survival": bool(survival_ok),
        "checkpoint_route_reached": int(reached),
        "checkpoint_route_success": bool(nav["success"]),
        "no_survival_regression_below_teacher": bool(survival_ok),
        "ok": bool(survival_ok and nav["success"]),
    }
    promotion = promotion_result(nav, gate, promotion_baseline, min_wp2_improvement)
    report = {"modes": summaries, "nav": nav, "gate": gate, "promotion": promotion}
    out = OUT / f"{tag}_bc_rollout_eval.json"
    out.write_text(json.dumps(report, indent=2))
    if len(nav_arr):
        np.savez(
            OUT / f"{tag}_bc_nav_trace.npz",
            t=nav_arr[:, 0],
            waypoint=nav_arr[:, 1],
            target_x=nav_arr[:, 2],
            target_y=nav_arr[:, 3],
            cmd_vx=nav_arr[:, 4],
            cmd_vy=nav_arr[:, 5],
            x=nav_arr[:, 6],
            y=nav_arr[:, 7],
            z=nav_arr[:, 8],
            up=nav_arr[:, 9],
            dist=nav_arr[:, 10],
            waypoints=waypoints,
            radius=nav_radius,
            tag=tag,
        )
    print(json.dumps(report, indent=2), flush=True)
    print(f"saved {out}", flush=True)
    return report


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", default=str(OUT / "gait_dataset"))
    ap.add_argument("--tag", default="bootstrap_bc")
    ap.add_argument("--hidden", default="256,256")
    ap.add_argument("--steps", type=int, default=20000)
    ap.add_argument("--batch", type=int, default=4096)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--val-frac", type=float, default=0.2)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--alive-only", action="store_true", default=True)
    ap.add_argument("--feature-mode", choices=["base", "transition"], default="base")
    ap.add_argument("--sample-weight-mode", choices=["uniform", "useful"], default="uniform")
    ap.add_argument("--positive-progress-weight", type=float, default=3.0)
    ap.add_argument("--bad-sample-weight", type=float, default=0.35)
    ap.add_argument("--waypoint2-weight", type=float, default=2.0)
    ap.add_argument("--drift-sample-weight", type=float, default=0.5)
    ap.add_argument("--cmd-jitter-std", type=float, default=0.0)
    ap.add_argument("--prev-action-noise-std", type=float, default=0.0)
    ap.add_argument("--prev-action-dropout", type=float, default=0.0)
    ap.add_argument("--stale-command-prob", type=float, default=0.0)
    ap.add_argument("--wrong-prev-action-prob", type=float, default=0.0)
    ap.add_argument("--phase-jitter-std", type=float, default=0.0,
                    help="radian stddev applied to transition sin/cos phase features during training")
    ap.add_argument("--eval-every", type=int, default=1000)
    ap.add_argument("--rollout-eval", action="store_true")
    ap.add_argument("--eval-only", action="store_true")
    ap.add_argument("--artifact", default=None)
    ap.add_argument("--rollout-modes", default="forward,backward,left,right,square")
    ap.add_argument("--rollout-hold", type=int, default=240)
    ap.add_argument("--rollout-speed", type=float, default=0.35)
    ap.add_argument("--min-survival-frac", type=float, default=0.95)
    ap.add_argument("--nav-radius", type=float, default=0.07)
    ap.add_argument("--nav-gain", type=float, default=2.0)
    ap.add_argument("--nav-steps", type=int, default=220)
    ap.add_argument("--promotion-baseline-json", default=None,
                    help="closed-loop rollout/nav JSON to beat before marking a BC variant promotable")
    ap.add_argument("--promotion-min-waypoint2-improvement", type=float, default=0.0)
    args = ap.parse_args()

    dataset = Path(args.dataset)
    if args.eval_only:
        if not args.artifact:
            raise ValueError("--eval-only requires --artifact")
        artifact = pickle.load(open(args.artifact, "rb"))
        modes = [x.strip() for x in args.rollout_modes.replace(";", ",").split(",") if x.strip()]
        rollout_eval(artifact, args.tag, modes, args.rollout_hold, args.rollout_speed, args.seed,
                     args.min_survival_frac, args.nav_radius, args.nav_gain, args.nav_steps,
                     read_json(args.promotion_baseline_json), args.promotion_min_waypoint2_improvement)
        return

    (train, val, train_files, val_files) = load_split(
        dataset,
        args.val_frac,
        args.seed,
        args.alive_only,
        args.feature_mode,
        args.sample_weight_mode,
        args.positive_progress_weight,
        args.bad_sample_weight,
        args.waypoint2_weight,
        args.drift_sample_weight,
    )
    x_train, y_train, w_train = train
    x_val, y_val, w_val = val
    mean = jnp.asarray(x_train.mean(axis=0), dtype=jnp.float32)
    std = jnp.asarray(x_train.std(axis=0) + 1e-6, dtype=jnp.float32)
    hidden = parse_sizes(args.hidden)
    params = init_mlp(jax.random.PRNGKey(args.seed), x_train.shape[1], y_train.shape[1], hidden)
    opt = optax.adam(args.lr)
    opt_state = opt.init(params)
    action_dim = int(y_train.shape[1])
    base_obs_dim = int(x_train.shape[1])
    if args.feature_mode == "transition":
        extra = action_dim + TRANSITION_FIXED_EXTRA + TRANSITION_PHASE_EXTRA
        base_obs_dim = int(x_train.shape[1] - extra)
    batcher = make_batcher(
        x_train, y_train, w_train, args.batch, args.seed + 1,
        args.feature_mode, base_obs_dim, action_dim,
        args.cmd_jitter_std, args.prev_action_noise_std,
        args.prev_action_dropout, args.stale_command_prob,
        args.wrong_prev_action_prob, args.phase_jitter_std,
    )

    @jax.jit
    def train_step(params, opt_state, xb, yb, wb):
        def loss_fn(p):
            pred = apply_mlp(p, (xb - mean) / std)
            return weighted_mse(pred, yb, wb)
        loss, grads = jax.value_and_grad(loss_fn)(params)
        updates, opt_state2 = opt.update(grads, opt_state, params)
        params2 = optax.apply_updates(params, updates)
        return params2, opt_state2, loss

    best = {"val_loss": float("inf"), "step": 0}
    best_params = params
    for step_i in range(1, args.steps + 1):
        xb, yb, wb = next(batcher)
        params, opt_state, loss = train_step(params, opt_state, xb, yb, wb)
        if step_i % args.eval_every == 0 or step_i == args.steps:
            train_n = min(len(x_train), 50000)
            train_loss = eval_loss(params, x_train[:train_n], y_train[:train_n], w_train[:train_n], mean, std)
            val_loss = eval_loss(params, x_val, y_val, w_val, mean, std)
            if val_loss < best["val_loss"]:
                best = {"val_loss": val_loss, "step": step_i, "train_loss": train_loss}
                best_params = params
            print(f"[bc] step {step_i} loss={float(loss):.6f} train={train_loss:.6f} val={val_loss:.6f}", flush=True)

    artifact = {
        "type": "cpg_teacher_bc_v1",
        "params": best_params,
        "obs_mean": np.asarray(mean),
        "obs_std": np.asarray(std),
        "obs_dim": int(x_train.shape[1]),
        "action_dim": int(y_train.shape[1]),
        "hidden": hidden,
        "feature_mode": args.feature_mode,
        "base_obs_dim": base_obs_dim,
        "sample_weight_mode": args.sample_weight_mode,
        "sample_weight_config": {
            "positive_progress_weight": args.positive_progress_weight,
            "bad_sample_weight": args.bad_sample_weight,
            "waypoint2_weight": args.waypoint2_weight,
            "drift_sample_weight": args.drift_sample_weight,
        },
        "sample_weight_stats": {
            "train_mean": float(np.mean(w_train)),
            "train_min": float(np.min(w_train)),
            "train_max": float(np.max(w_train)),
            "val_mean": float(np.mean(w_val)),
            "val_min": float(np.min(w_val)),
            "val_max": float(np.max(w_val)),
        },
        "context_corruption": {
            "cmd_jitter_std": args.cmd_jitter_std,
            "prev_action_noise_std": args.prev_action_noise_std,
            "prev_action_dropout": args.prev_action_dropout,
            "stale_command_prob": args.stale_command_prob,
            "wrong_prev_action_prob": args.wrong_prev_action_prob,
            "phase_jitter_std": args.phase_jitter_std,
        },
        "transition_phase_features": bool(args.feature_mode == "transition"),
        "best": best,
        "dataset": str(dataset),
        "train_files": [str(p) for p in train_files],
        "val_files": [str(p) for p in val_files],
    }
    OUT.mkdir(parents=True, exist_ok=True)
    pkl = OUT / f"{args.tag}.pkl"
    meta = OUT / f"{args.tag}_bc_meta.json"
    pickle.dump(artifact, open(pkl, "wb"))
    meta.write_text(json.dumps({k: v for k, v in artifact.items() if k not in ("params", "obs_mean", "obs_std")}, indent=2))
    print(f"saved {pkl} and {meta}", flush=True)
    if args.rollout_eval:
        modes = [x.strip() for x in args.rollout_modes.replace(";", ",").split(",") if x.strip()]
        rollout_eval(artifact, args.tag, modes, args.rollout_hold, args.rollout_speed, args.seed,
                     args.min_survival_frac, args.nav_radius, args.nav_gain, args.nav_steps,
                     read_json(args.promotion_baseline_json), args.promotion_min_waypoint2_improvement)


if __name__ == "__main__":
    main()
