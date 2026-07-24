# SPDX-License-Identifier: MIT
"""render_walker_video.py — roll a trained walker policy (walker_warp_env) to MP4.

  MUJOCO_GL=egl python render_walker_video.py --ckpt walker_step.pt --out walker.mp4

Rebuilds the actor from the checkpoint, drives a plain-MuJoCo walker_improved
model under a fixed forward command through the SAME PD servo (with torque-speed
derating) the env uses, and writes a gait signature next to the video.
"""
from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

import mujoco
import torch

from walker_improved import build_walker, DEFAULTS, LEGS, _DESIGN
from walker_warp_env import WALKER_KP


def _actor_from_ckpt(path, obs_dim, act_dim, priv_dim):
    from train_mesh_warp import Actor          # the exact trained architecture
    ck = torch.load(path, map_location="cpu", weights_only=True)
    actor = Actor(obs_dim, act_dim, (512, 256, 128))
    actor.load_state_dict(ck["actor"])         # strict: must match exactly
    actor.eval()
    norm = ck.get("obs_norm")
    return actor, norm, ck


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--steps", type=int, default=500)
    ap.add_argument("--speed", type=float, default=0.12)
    ap.add_argument("--w", type=int, default=480)
    ap.add_argument("--h", type=int, default=320)
    ap.add_argument("--fps", type=int, default=30)
    args = ap.parse_args()

    m = mujoco.MjModel.from_xml_string(build_walker())
    d = mujoco.MjData(m)
    aj = [int(m.actuator_trnid[a, 0]) for a in range(m.nu)]
    qa = np.array([m.jnt_qposadr[j] for j in aj])
    da = np.array([m.jnt_dofadr[j] for j in aj])
    gear = m.actuator_gear[:m.nu, 0].copy()
    kp = np.array(list(WALKER_KP) * 4)
    # Match WalkerWarpEnv exactly: no-load speeds come from robot_design.TARGET,
    # not from the older mesh robot constants.
    wfree = np.array(_DESIGN.wfrees())
    jr = np.array([m.jnt_range[j] for j in aj])
    frac = np.array([0.6, 0.6, 1.0] * 4)
    authority = frac * 0.5 * (jr[:, 1] - jr[:, 0])
    stand = np.zeros(12)
    for k in range(4):
        stand[3 * k + 2] = DEFAULTS["lift_nom"]
    feet = [mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_GEOM, f"{L}_foot") for L in LEGS]
    floor = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_GEOM, "floor")

    obs_dim, priv_dim = 50, 34
    actor, norm, ck = _actor_from_ckpt(args.ckpt, obs_dim, 12, priv_dim)

    def yaw_rot():
        w, x, y, z = d.qpos[3:7]
        yaw = math.atan2(2 * (w * z + x * y), 1 - 2 * (y * y + z * z))
        c, s = math.cos(yaw), math.sin(yaw)
        return np.array([[c, s], [-s, c]])

    cmd = np.array([args.speed, 0.0, 0.0])
    prev_a = np.zeros(12)

    def obs():
        R = yaw_rot()
        rv = np.concatenate([R @ d.qvel[0:2], d.qvel[2:6]])
        return np.concatenate([d.qpos[qa], d.qvel[da], d.qpos[3:7], rv,
                               d.qpos[2:3], prev_a,
                               np.concatenate([R @ cmd[:2], cmd[2:3]])]).astype(np.float32)

    def apply_norm(o):
        if norm is None:
            return o
        mean = norm["mean"].numpy()
        var = norm["var"].numpy()
        return np.clip((o - mean) / np.sqrt(var + 1e-8), -5.0, 5.0)   # RunningNorm clip

    d.qpos[:] = 0
    d.qpos[2] = DEFAULTS["stance_h"]
    d.qpos[qa] = stand
    mujoco.mj_forward(m, d)
    for _ in range(int(0.4 / m.opt.timestep)):
        d.ctrl[:] = np.clip(kp * (stand - d.qpos[qa]) / gear, -1, 1)
        mujoco.mj_step(m, d)

    ren = mujoco.Renderer(m, height=args.h, width=args.w)
    frames, contacts = [], []
    render_every = max(1, int(1.0 / (args.fps * m.opt.timestep * 5)))
    x0 = float(d.qpos[0])
    for t in range(args.steps):
        o = apply_norm(obs())
        with torch.no_grad():
            a = actor(torch.from_numpy(o).float()).numpy()
        a = np.clip(a, -1, 1)
        prev_a[:] = 0.6 * prev_a + 0.4 * a
        target = np.clip(stand + prev_a * authority, jr[:, 0], jr[:, 1])
        for _ in range(5):
            tau = kp * (target - d.qpos[qa])
            qd = d.qvel[da]
            der = np.clip(1 - np.abs(qd) / wfree, 0, 1)
            lim = np.where(tau * qd > 0, der, 1.0)
            d.ctrl[:] = np.clip(tau / gear, -lim, lim)
            mujoco.mj_step(m, d)
        c = np.array([any((d.contact.geom1[i] == fg and d.contact.geom2[i] == floor) or
                          (d.contact.geom2[i] == fg and d.contact.geom1[i] == floor)
                          for i in range(d.ncon)) for fg in feet])
        contacts.append(c)
        if t % render_every == 0:
            ren.update_scene(d, camera=-1)
            frames.append(ren.render())
        if d.qpos[2] < 0.25:
            print(f"fell at t={t}")
            break

    dx = float(d.qpos[0]) - x0
    dur = len(contacts) * m.opt.timestep * 5
    duty = float(np.mean(contacts))
    print(f"rollout {len(contacts)} steps: dx={dx*1000:+.0f}mm ({dx/max(dur,1e-6):+.3f} m/s "
          f"vs cmd {args.speed}), duty={duty:.3f}")
    import imageio
    imageio.mimsave(args.out, frames, fps=args.fps)
    print(f"wrote {args.out} ({len(frames)} frames)")
    sig = dict(duty_factor=duty, displacement_m=dx, mean_speed=dx / max(dur, 1e-6),
               commanded=args.speed, steps=len(contacts),
               per_foot_duty=np.mean(contacts, axis=0).tolist())
    Path(args.out).with_suffix(".gait.json").write_text(json.dumps(sig, indent=2))
    print("gait:", sig)


if __name__ == "__main__":
    main()
