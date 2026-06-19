# SPDX-License-Identifier: MIT
"""Render a recorded rollout trajectory to a video (notes/rl-environment-checklist.md §4).

Torch-free by design: eval.py records the MuJoCo state (qpos/qvel) per step in the
policy process; this process replays + renders it with mujoco.Renderer (OSMesa),
avoiding the fragile torch+renderer interaction. The record->render split is the
same philosophy as the kinetic-realm viz plan. Usage:
  MUJOCO_GL=osmesa python render_rollout.py --traj sim/build/rl/<tag>_traj.npz --tag <tag>
"""

from __future__ import annotations

import argparse
import os
from pathlib import Path

os.environ.setdefault("MUJOCO_GL", "osmesa")

import gymnasium as gym  # noqa: E402
import imageio  # noqa: E402
import mujoco  # noqa: E402
import numpy as np  # noqa: E402

ROOT = Path(__file__).resolve().parents[2]
FIG = ROOT / "figures" / "rl"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--traj", required=True)
    ap.add_argument("--tag", default="rollout")
    ap.add_argument("--fps", type=int, default=30)
    ap.add_argument("--w", type=int, default=480)
    ap.add_argument("--h", type=int, default=360)
    args = ap.parse_args()

    data = np.load(args.traj)
    qpos, qvel = data["qpos"], data["qvel"]
    robot = str(data["robot"]) if "robot" in data else "Ant-v5"
    if robot == "dodge":
        import sys
        sys.path.insert(0, str(Path(__file__).resolve().parent))
        from dodge_env import build_scene
        m = mujoco.MjModel.from_xml_string(build_scene())
        d = mujoco.MjData(m)
    else:
        env = gym.make(robot)
        env.reset(seed=0)
        m, d = env.unwrapped.model, env.unwrapped.data

    cam = mujoco.MjvCamera()
    cam.type = mujoco.mjtCamera.mjCAMERA_TRACKING
    cam.trackbodyid = max(1, mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, "torso"))
    cam.distance, cam.elevation, cam.azimuth = 8.0, -20.0, 90.0

    renderer = mujoco.Renderer(m, args.h, args.w)
    frames = []
    for t in range(len(qpos)):
        d.qpos[:] = qpos[t]; d.qvel[:] = qvel[t]
        mujoco.mj_forward(m, d)
        renderer.update_scene(d, camera=cam)
        frames.append(renderer.render())
    renderer.close()

    FIG.mkdir(parents=True, exist_ok=True)
    out = FIG / f"{args.tag}.mp4"
    imageio.mimsave(out, frames, fps=args.fps)
    print(f"wrote {out} ({len(frames)} frames, {robot})")


if __name__ == "__main__":
    main()
