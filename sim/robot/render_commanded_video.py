# SPDX-License-Identifier: MIT
"""Render a command-conditioned walking rollout to mp4/GIF."""

from __future__ import annotations

import argparse, os, pickle, sys
from pathlib import Path

os.environ.setdefault("MUJOCO_GL", "egl")

import imageio
import jax
import jax.numpy as jnp
import mujoco
import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from commanded_env import FALL_Z, MIN_UP_Z, VMAX, _build  # noqa: E402
from eval_commanded import command_program, resolve_ckpt  # noqa: E402
from gen_robot_mjcf import build_mjcf, load_spec  # noqa: E402
from brax.training.acme import running_statistics  # noqa: E402
import ppo_nets as ppo_networks  # shared (512,256,128) + small-final-init factory (audit item 5)  # noqa: E402

OUT = Path(os.environ.get("CODESIGN_OUT", "/root/proj/out"))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tag", default="cmd")
    ap.add_argument("--ckpt", default=None)
    ap.add_argument("--out", default=None)
    ap.add_argument("--mode", choices=["square", "forward", "sweep"], default="forward")
    ap.add_argument("--hold", type=int, default=160)
    ap.add_argument("--speed", type=float, default=VMAX)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--fps", type=int, default=30)
    ap.add_argument("--w", type=int, default=720)
    ap.add_argument("--h", type=int, default=480)
    args = ap.parse_args()

    ckpt = resolve_ckpt(args.tag, args.ckpt)
    params = pickle.load(open(ckpt, "rb"))
    Env = _build(); env = Env()
    net = ppo_networks.make_ppo_networks(env.observation_size, env.action_size,
                                         preprocess_observations_fn=running_statistics.normalize)
    inf = ppo_networks.make_inference_fn(net)(params, deterministic=True)
    step = jax.jit(env.step)
    cmds = command_program(args.mode, args.hold, args.speed)
    key = jax.random.PRNGKey(args.seed)
    st = env.reset_with_command(key, cmds[0])
    qpos_log = []
    for i, cmd in enumerate(cmds):
        st = st.replace(info={**st.info, "cmd": jnp.asarray(cmd), "remote": jnp.array(True)})
        st = st.replace(obs=st.obs.at[-2:].set(jnp.asarray(cmd)))
        key, k = jax.random.split(key)
        act, _ = inf(st.obs, k)
        st = step(st, act)
        dx = st.pipeline_state
        qpos_log.append(np.asarray(dx.qpos))
        up = 1.0 - 2.0 * (float(dx.qpos[4]) ** 2 + float(dx.qpos[5]) ** 2)
        if float(dx.qpos[2]) < FALL_Z or up < MIN_UP_Z:
            print(f"rollout stopped at step {i}: z={float(dx.qpos[2]):.3f} up={up:.3f}")
            break

    model = mujoco.MjModel.from_xml_string(build_mjcf(load_spec(HERE / "robot.toml")))
    data = mujoco.MjData(model)
    renderer = mujoco.Renderer(model, args.h, args.w)
    cam = mujoco.MjvCamera()
    cam.lookat[:] = [0.0, 0.0, 0.20]; cam.distance = 1.8; cam.elevation = -18; cam.azimuth = 125
    frames = []
    for q in qpos_log:
        data.qpos[:] = q
        mujoco.mj_forward(model, data)
        cam.lookat[0] = data.qpos[0]
        cam.lookat[1] = data.qpos[1]
        renderer.update_scene(data, camera=cam)
        frames.append(renderer.render())
    out = args.out or str(OUT / f"{args.tag}_{args.mode}.mp4")
    try:
        imageio.mimsave(out, frames, fps=args.fps, codec="libx264", quality=8)
        print(f"wrote {out} ({len(frames)} frames @ {args.fps}fps)")
    except Exception as e:
        gif = os.path.splitext(out)[0] + ".gif"
        imageio.mimsave(gif, frames[::2], duration=2.0 / args.fps)
        print(f"mp4 failed ({e}); wrote {gif} instead")


if __name__ == "__main__":
    main()
