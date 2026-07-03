# SPDX-License-Identifier: MIT
"""Render a trained mesh-robot policy (mesh_commanded_env) to MP4.

  MUJOCO_GL=egl python render_mesh_video.py --ckpt out/meshwalk1.pkl --out out/meshwalk1.mp4
Rolls the policy under a fixed forward command; also writes a gait-signature
JSON next to the video (V.7: the eyeball stays the gate, the numbers make it
a characterization).
"""
from __future__ import annotations

import argparse
import json
import pickle
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

import jax
import jax.numpy as jnp
import mujoco
import numpy as np

import ppo_nets as ppo_networks
from brax.training.acme import running_statistics
from gen_mesh_robot_mjcf import build_mesh_robot
from mesh_commanded_env import MESH_VMAX, _build


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--steps", type=int, default=300)
    ap.add_argument("--speed", type=float, default=0.6 * MESH_VMAX)
    ap.add_argument("--w", type=int, default=480)
    ap.add_argument("--h", type=int, default=320)
    ap.add_argument("--fps", type=int, default=30)
    args = ap.parse_args()

    env = _build()()
    params = pickle.load(open(args.ckpt, "rb"))
    net = ppo_networks.make_ppo_networks(
        env.observation_size, env.action_size,
        preprocess_observations_fn=running_statistics.normalize)
    infer = ppo_networks.make_inference_fn(net)(tuple(params), deterministic=True)

    s = env.reset(jax.random.PRNGKey(0))
    step = jax.jit(env.step)
    infer_j = jax.jit(infer)
    cmd = jnp.array([args.speed, 0.0, 0.0])

    m = mujoco.MjModel.from_xml_string(build_mesh_robot())
    d = mujoco.MjData(m)
    ren = mujoco.Renderer(m, height=args.h, width=args.w)
    frames, contacts, x0 = [], [], None
    render_every = max(1, int(1.0 / (args.fps * env._dt)))
    for t in range(args.steps):
        s = s.replace(info={**s.info, "cmd": cmd, "cmd_timer": jnp.zeros((), dtype=jnp.int32)})
        a, _ = infer_j(s.obs, jax.random.PRNGKey(0))
        s = step(s, a)
        contacts.append(np.asarray(
            s.pipeline_state.geom_xpos[env._feet][:, 2] < 0.03))
        if x0 is None:
            x0 = float(s.pipeline_state.qpos[0])
        if t % render_every == 0:
            d.qpos[:] = np.asarray(s.pipeline_state.qpos)
            d.qvel[:] = np.asarray(s.pipeline_state.qvel)
            mujoco.mj_forward(m, d)
            ren.update_scene(d, camera=-1)
            frames.append(ren.render())
        if float(s.done):
            print(f"episode ended (fall) at t={t}")
            break

    dx = float(s.pipeline_state.qpos[0]) - x0
    dur = len(contacts) * env._dt
    print(f"rollout: {len(contacts)} steps, forward displacement {dx:+.3f} m "
          f"({dx / max(dur, 1e-6):+.3f} m/s vs commanded {args.speed})")

    import imageio
    imageio.mimsave(args.out, frames, fps=args.fps)
    print(f"wrote {args.out} ({len(frames)} frames)")
    try:
        from gait_signature import gait_signature
        sig = gait_signature(np.asarray(contacts), env._dt)
        sig.update(displacement_m=dx, mean_speed=dx / max(dur, 1e-6),
                   commanded=args.speed, steps=len(contacts))
        Path(args.out).with_suffix(".gait.json").write_text(json.dumps(sig, indent=2, default=float))
        print("gait signature:", sig)
    except Exception as e:
        print(f"gait signature skipped: {e}")


if __name__ == "__main__":
    main()
