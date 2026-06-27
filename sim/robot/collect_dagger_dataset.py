# SPDX-License-Identifier: MIT
"""Collect learner-visited states and relabel them with a teacher/corrector.

This is a lightweight DAgger-style data pass:

    learned BC policy rolls out -> visited states -> corrective action labels

The resulting `.npz` is BC-compatible (`obs`, `action`, optional labels) and can
be mixed back into `train_bootstrap_bc.py`.
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
from train_bootstrap_bc import apply_mlp, policy_obs  # noqa: E402

OUT = Path(os.environ.get("CODESIGN_OUT", "/root/proj/out"))


def parse_waypoints(text: str) -> np.ndarray:
    pts = []
    for chunk in text.split(";"):
        chunk = chunk.strip()
        if not chunk:
            continue
        x, y = chunk.split(",")
        pts.append((float(x), float(y)))
    if not pts:
        raise ValueError("at least one waypoint is required")
    return np.asarray(pts, dtype=np.float32)


def apply_route_env(route_json: str | None):
    if not route_json:
        return None
    data = json.loads(Path(route_json).read_text())
    for key, value in data.get("deploy_env", {}).items():
        os.environ[str(key)] = str(value)
    return data


def one_step_oracle_label(step, st, teacher_label, policy_action, target, active,
                          rng: np.random.Generator, candidates: int, noise_std: float,
                          fall_z: float, min_up_z: float, action_penalty: float,
                          cross_penalty: float, fall_penalty: float) -> tuple[np.ndarray, dict]:
    """Pick the one-step action that most reduces waypoint error from this learner state."""
    teacher_np = np.asarray(teacher_label, dtype=np.float32)
    policy_np = np.asarray(policy_action, dtype=np.float32)
    target_np = np.asarray(target, dtype=np.float32)
    pre_xy = np.asarray(st.pipeline_state.qpos[:2], dtype=np.float32)
    pre_dist = float(np.linalg.norm(target_np - pre_xy))
    centers = [teacher_np, policy_np, 0.5 * (teacher_np + policy_np)]
    acts = [np.clip(c, -1.0, 1.0).astype(np.float32) for c in centers]
    for i in range(max(0, int(candidates) - len(acts))):
        center = centers[i % len(centers)]
        acts.append(np.clip(center + rng.normal(0.0, noise_std, size=center.shape), -1.0, 1.0).astype(np.float32))

    best_action = acts[0]
    best_score = -float("inf")
    best_meta = {}
    pre_cross = abs(float(pre_xy[1] - target_np[1]))
    for idx, action in enumerate(acts):
        st2 = step(st, jnp.asarray(action))
        post_xy = np.asarray(st2.pipeline_state.qpos[:2], dtype=np.float32)
        post_dist = float(np.linalg.norm(target_np - post_xy))
        up = 1.0 - 2.0 * (float(st2.pipeline_state.qpos[4]) ** 2 + float(st2.pipeline_state.qpos[5]) ** 2)
        alive = float(float(st2.pipeline_state.qpos[2]) >= fall_z and up >= min_up_z)
        cross_growth = max(abs(float(post_xy[1] - target_np[1])) - pre_cross, 0.0)
        x_loss = max(float(post_xy[0] - pre_xy[0]), 0.0) if int(active) == 2 else 0.0
        score = (
            (pre_dist - post_dist)
            - cross_penalty * cross_growth
            - cross_penalty * x_loss
            - action_penalty * float(np.mean(action ** 2))
            - fall_penalty * (1.0 - alive)
        )
        if score > best_score:
            best_score = score
            best_action = action
            best_meta = {
                "oracle_candidate": idx,
                "oracle_score": score,
                "oracle_dist_before": pre_dist,
                "oracle_dist_after": post_dist,
                "oracle_alive": alive,
            }
    return best_action.astype(np.float32), best_meta


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--artifact", required=True, help="BC artifact .pkl to roll out")
    ap.add_argument("--route-json", default=None,
                    help="route candidate JSON whose deploy_env defines the CPG scaffold labels")
    ap.add_argument("--out-dir", default=str(OUT / "gait_dataset"))
    ap.add_argument("--tag", default="dagger_bc")
    ap.add_argument("--waypoints", default="0.35,0;0.35,0.35;0,0.35;0,0")
    ap.add_argument("--radius", type=float, default=0.07)
    ap.add_argument("--gain", type=float, default=2.0)
    ap.add_argument("--steps-per-waypoint", type=int, default=260)
    ap.add_argument("--episodes", type=int, default=4)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--fall-z", type=float, default=None)
    ap.add_argument("--min-up-z", type=float, default=None)
    ap.add_argument("--label-mode", choices=["cpg", "one_step_oracle"], default="cpg",
                    help="cpg uses the scaffold label; one_step_oracle locally searches for a corrective action")
    ap.add_argument("--oracle-candidates", type=int, default=17)
    ap.add_argument("--oracle-noise-std", type=float, default=0.20)
    ap.add_argument("--oracle-action-penalty", type=float, default=0.01)
    ap.add_argument("--oracle-cross-penalty", type=float, default=2.0)
    ap.add_argument("--oracle-fall-penalty", type=float, default=5.0)
    args = ap.parse_args()

    apply_route_env(args.route_json)
    os.environ.setdefault("CMD_CONTROL_MODE", "pd")

    from commanded_env import FALL_Z, MIN_UP_Z, PD_SCALE, VMAX, _build  # noqa: E402
    from cpg_teacher import cpg_pd_step_target  # noqa: E402

    fall_z = FALL_Z if args.fall_z is None else float(args.fall_z)
    min_up_z = MIN_UP_Z if args.min_up_z is None else float(args.min_up_z)
    artifact = pickle.load(open(args.artifact, "rb"))
    params = artifact["params"]
    mean = jnp.asarray(artifact["obs_mean"])
    std = jnp.asarray(artifact["obs_std"])

    Env = _build()
    env = Env()
    step = jax.jit(env.step)

    @jax.jit
    def policy(obs):
        return apply_mlp(params, (obs - mean) / std)

    @jax.jit
    def teacher_action(state_obs, phase, cmd):
        _, motor_action, _ = cpg_pd_step_target(
            env._stand,
            env._jr,
            phase,
            cmd,
            jnp.zeros(env.action_size, dtype=jnp.float32),
            env._cpg_idx,
            env.action_size,
            VMAX,
            0.0,
            PD_SCALE,
            directional=env._cpg,
            xp=jnp,
        )
        return motor_action

    waypoints = parse_waypoints(args.waypoints)
    total_steps = int(args.steps_per_waypoint * len(waypoints))
    rows = {
        "obs": [],
        "action": [],
        "policy_action": [],
        "command": [],
        "target": [],
        "xy": [],
        "z": [],
        "up": [],
        "alive": [],
        "active": [],
        "phase": [],
        "dist": [],
        "distance_reduction": [],
        "y_drift": [],
        "x_loss": [],
        "oracle_score": [],
        "oracle_dist_before": [],
        "oracle_dist_after": [],
        "oracle_alive": [],
        "oracle_candidate": [],
    }
    summaries = []
    oracle_rng = np.random.default_rng(args.seed + 10_000)

    for ep in range(args.episodes):
        key = jax.random.PRNGKey(args.seed + ep)
        st = env.reset_with_command(key, jnp.zeros(2))
        prev_cmd = jnp.zeros(2)
        prev_action = jnp.zeros(env.action_size)
        reached = 0
        ep_rec = []
        for t in range(total_steps):
            pos = np.asarray(st.pipeline_state.qpos[:2])
            target = waypoints[min(reached, len(waypoints) - 1)]
            delta = target - pos
            dist = float(np.linalg.norm(delta))
            if dist < args.radius and reached < len(waypoints):
                reached += 1
                if reached >= len(waypoints):
                    break
                target = waypoints[reached]
                delta = target - pos
                dist = float(np.linalg.norm(delta))
            cmd = args.gain * delta
            norm = float(np.linalg.norm(cmd))
            if norm > VMAX:
                cmd = cmd * (VMAX / norm)
            cmd = np.asarray(cmd, dtype=np.float32)
            st = st.replace(info={**st.info, "cmd": jnp.asarray(cmd), "remote": jnp.array(True)})
            st = st.replace(obs=st.obs.at[-2:].set(jnp.asarray(cmd)))

            obs0 = st.obs
            phase = st.info["phase"]
            teacher_label = teacher_action(obs0, phase, jnp.asarray(cmd))
            po = policy_obs(artifact, obs0, prev_cmd, prev_action, float(reached), phase=phase)
            act = policy(po)
            oracle_meta = {
                "oracle_score": float("nan"),
                "oracle_dist_before": dist,
                "oracle_dist_after": float("nan"),
                "oracle_alive": float("nan"),
                "oracle_candidate": -1,
            }
            if args.label_mode == "one_step_oracle":
                label_np, found_meta = one_step_oracle_label(
                    step,
                    st,
                    teacher_label,
                    act,
                    target,
                    reached,
                    oracle_rng,
                    args.oracle_candidates,
                    args.oracle_noise_std,
                    fall_z,
                    min_up_z,
                    args.oracle_action_penalty,
                    args.oracle_cross_penalty,
                    args.oracle_fall_penalty,
                )
                oracle_meta.update(found_meta)
                label = jnp.asarray(label_np)
            else:
                label = teacher_label
            pre_xy = np.asarray(st.pipeline_state.qpos[:2], dtype=np.float32)
            st = step(st, act)
            post_xy = np.asarray(st.pipeline_state.qpos[:2], dtype=np.float32)
            dx = st.pipeline_state
            up = 1.0 - 2.0 * (float(dx.qpos[4]) ** 2 + float(dx.qpos[5]) ** 2)
            alive = float(float(dx.qpos[2]) >= fall_z and up >= min_up_z)
            dist_after = float(np.linalg.norm(target - post_xy))

            rows["obs"].append(np.asarray(obs0, dtype=np.float32))
            rows["action"].append(np.asarray(label, dtype=np.float32))
            rows["policy_action"].append(np.asarray(act, dtype=np.float32))
            rows["command"].append(cmd)
            rows["target"].append(target)
            rows["xy"].append(post_xy)
            rows["z"].append(float(dx.qpos[2]))
            rows["up"].append(up)
            rows["alive"].append(alive)
            rows["active"].append(float(reached))
            rows["phase"].append(float(np.asarray(phase)))
            rows["dist"].append(dist_after)
            rows["distance_reduction"].append(dist - dist_after)
            rows["y_drift"].append(float(post_xy[1] - pre_xy[1]))
            rows["x_loss"].append(float(max(post_xy[0] - pre_xy[0], 0.0)) if reached == 2 else 0.0)
            rows["oracle_score"].append(float(oracle_meta["oracle_score"]))
            rows["oracle_dist_before"].append(float(oracle_meta["oracle_dist_before"]))
            rows["oracle_dist_after"].append(float(oracle_meta["oracle_dist_after"]))
            rows["oracle_alive"].append(float(oracle_meta["oracle_alive"]))
            rows["oracle_candidate"].append(float(oracle_meta["oracle_candidate"]))
            ep_rec.append([t, reached, float(post_xy[0]), float(post_xy[1]), float(dx.qpos[2]), up, dist_after])

            prev_cmd = jnp.asarray(cmd)
            prev_action = act
            if not alive:
                break

        arr = np.asarray(ep_rec, dtype=np.float32)
        summaries.append({
            "episode": ep,
            "samples": int(len(ep_rec)),
            "reached": int(reached),
            "final_dist": float(arr[-1, 6]) if len(arr) else float("nan"),
            "z_min": float(arr[:, 4].min()) if len(arr) else float("nan"),
            "up_min": float(arr[:, 5].min()) if len(arr) else float("nan"),
        })

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    arrays = {k: np.asarray(v, dtype=np.float32) for k, v in rows.items()}
    path = out_dir / f"dagger_{args.tag}.npz"
    np.savez(
        path,
        **arrays,
        fall=(arrays["alive"] < 0.5).astype(np.float32),
        high_saturation=np.zeros_like(arrays["alive"], dtype=np.float32),
        route_json=str(args.route_json or ""),
        artifact=str(args.artifact),
    )
    manifest = {
        "tag": args.tag,
        "path": str(path),
        "artifact": str(args.artifact),
        "route_json": str(args.route_json or ""),
        "episodes": args.episodes,
        "label_mode": args.label_mode,
        "samples": int(arrays["obs"].shape[0]),
        "positive_progress_samples": int((arrays["distance_reduction"] > 0.0).sum()),
        "oracle_labeled_samples": int(np.isfinite(arrays["oracle_score"]).sum()),
        "summaries": summaries,
    }
    manifest_path = out_dir / f"dagger_{args.tag}_manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2))
    print(json.dumps(manifest, indent=2), flush=True)
    print(f"saved {path} and {manifest_path}", flush=True)


if __name__ == "__main__":
    main()
