# SPDX-License-Identifier: MIT
"""Waypoint/checkpoint validation for a command-conditioned locomotor.

Loads the velocity-command policy, points its command vector at successive XY
checkpoints, and records whether the robot reaches them without falling.
"""

from __future__ import annotations

import argparse, json, os, pickle, sys
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from commanded_env import FALL_Z, MIN_UP_Z, VMAX, _build  # noqa: E402
from eval_commanded import resolve_ckpt  # noqa: E402
from brax.training.acme import running_statistics  # noqa: E402
from brax.training.agents.ppo import networks as ppo_networks  # noqa: E402

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


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tag", default="cmd")
    ap.add_argument("--ckpt", default=None)
    ap.add_argument("--waypoints", default="0.35,0;0.35,0.35;0,0.35;0,0",
                    help="semicolon-separated XY waypoints, e.g. '0.4,0;0.4,0.4;0,0'")
    ap.add_argument("--radius", type=float, default=0.07)
    ap.add_argument("--gain", type=float, default=2.0)
    ap.add_argument("--axis-priority", action="store_true",
                    help="for axis-aligned waypoint courses, command only the dominant remaining axis")
    ap.add_argument("--segment-axis", action="store_true",
                    help="for axis-aligned waypoint courses, command along the active waypoint segment axis")
    ap.add_argument("--axis-deadband", type=float, default=0.02)
    ap.add_argument("--steps-per-waypoint", type=int, default=140)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out-npz", default=None)
    ap.add_argument("--out-json", default=None)
    ap.add_argument("--fall-z", type=float, default=FALL_Z)
    ap.add_argument("--min-up-z", type=float, default=MIN_UP_Z)
    args = ap.parse_args()

    ckpt = resolve_ckpt(args.tag, args.ckpt)
    params = pickle.load(open(ckpt, "rb"))
    Env = _build(); env = Env()
    net = ppo_networks.make_ppo_networks(env.observation_size, env.action_size,
                                         preprocess_observations_fn=running_statistics.normalize)
    inf = ppo_networks.make_inference_fn(net)(params, deterministic=True)
    step = jax.jit(env.step)

    waypoints = parse_waypoints(args.waypoints)
    total_budget = int(args.steps_per_waypoint * len(waypoints))
    key = jax.random.PRNGKey(args.seed)
    st = env.reset_with_command(key, jnp.zeros(2))
    reached = 0
    fall_reason = ""
    rec = []

    for t in range(total_budget):
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
        if args.segment_axis:
            prev = np.zeros(2, dtype=np.float32) if reached == 0 else waypoints[reached - 1]
            seg = target - prev
            cmd2 = np.zeros_like(cmd)
            ax = int(np.argmax(np.abs(seg)))
            remain = target[ax] - pos[ax]
            if abs(float(remain)) > args.axis_deadband:
                cmd2[ax] = args.gain * remain
            cmd = cmd2
        elif args.axis_priority:
            cmd2 = np.zeros_like(cmd)
            ax = int(np.argmax(np.abs(delta)))
            if abs(float(delta[ax])) > args.axis_deadband:
                cmd2[ax] = cmd[ax]
            cmd = cmd2
        n = float(np.linalg.norm(cmd))
        if n > VMAX:
            cmd = cmd * (VMAX / n)
        cmd = np.asarray(cmd, dtype=np.float32)
        info = {**st.info, "cmd": jnp.asarray(cmd), "remote": jnp.array(True)}
        if "route_wp" in info:
            info["route_wp"] = jnp.asarray(min(reached, len(waypoints) - 1), dtype=jnp.int32)
        if "route_prev_dist" in info:
            info["route_prev_dist"] = jnp.asarray(dist, dtype=jnp.float32)
        st = st.replace(info=info)
        st = st.replace(obs=st.obs.at[-2:].set(jnp.asarray(cmd)))
        key, k = jax.random.split(key)
        act, _ = inf(st.obs, k)
        st = step(st, act)
        dx = st.pipeline_state
        up = 1.0 - 2.0 * (float(dx.qpos[4]) ** 2 + float(dx.qpos[5]) ** 2)
        rec.append([t, reached, float(target[0]), float(target[1]), float(cmd[0]), float(cmd[1]),
                    float(dx.qvel[0]), float(dx.qvel[1]), float(dx.qpos[0]), float(dx.qpos[1]),
                    float(dx.qpos[2]), up, dist])
        if float(dx.qpos[2]) < args.fall_z:
            fall_reason = f"height<{args.fall_z:.3f}"
            break
        if up < args.min_up_z:
            fall_reason = f"up<{args.min_up_z:.3f}"
            break

    R = np.asarray(rec, dtype=np.float32)
    npz = Path(args.out_npz) if args.out_npz else OUT / f"{args.tag}_nav.npz"
    js = Path(args.out_json) if args.out_json else OUT / f"{args.tag}_nav_summary.json"
    if R.size:
        np.savez(npz, t=R[:, 0], waypoint=R[:, 1], target_x=R[:, 2], target_y=R[:, 3],
                 cmd_vx=R[:, 4], cmd_vy=R[:, 5], vx=R[:, 6], vy=R[:, 7],
                 x=R[:, 8], y=R[:, 9], z=R[:, 10], up=R[:, 11], dist=R[:, 12],
                 waypoints=waypoints, radius=args.radius, ckpt=str(ckpt), tag=args.tag)
        final_dist = float(R[-1, 12])
        z_min = float(R[:, 10].min())
        up_min = float(R[:, 11].min())
        closest = []
        xy = R[:, 8:10]
        for i, wp in enumerate(waypoints):
            d = np.linalg.norm(xy - wp, axis=1)
            j = int(d.argmin())
            closest.append({
                "waypoint": int(i),
                "min_dist": float(d[j]),
                "closest_xy": xy[j].tolist(),
                "step": j,
                "inside_radius": bool(d[j] <= args.radius),
            })
    else:
        np.savez(npz, waypoints=waypoints, radius=args.radius, ckpt=str(ckpt), tag=args.tag)
        final_dist = float("nan"); z_min = float("nan"); up_min = float("nan")
        closest = []
    summary = dict(tag=args.tag, ckpt=str(ckpt), waypoints=waypoints.tolist(),
                   reached=int(reached), total_waypoints=int(len(waypoints)),
                   success=bool(reached >= len(waypoints) and not fall_reason),
                   samples=int(len(R)), budget=total_budget, final_dist=final_dist,
                   z_min=z_min, up_min=up_min, fall_reason=fall_reason,
                   closest_waypoints=closest,
                   waypoint2_min_dist=float(closest[2]["min_dist"]) if len(closest) > 2 else float("nan"))
    js.write_text(json.dumps(summary, indent=2))
    print(f"[nav] reached {summary['reached']}/{summary['total_waypoints']} "
          f"success={int(summary['success'])} samples={summary['samples']} "
          f"fall={fall_reason or 'none'} ckpt={ckpt.name}")
    print("METRIC " + " ".join(f"{k}={v}" for k, v in dict(
        stage="cmd_nav", tag=args.tag, reached=summary["reached"],
        total=summary["total_waypoints"], success=int(summary["success"]),
        samples=summary["samples"]).items()))
    print(f"saved {npz} and {js}")


if __name__ == "__main__":
    main()
