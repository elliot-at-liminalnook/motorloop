# SPDX-License-Identifier: MIT
"""Velocity-command locomotion for the REAL leg mechanism (mesh_robot.xml).

The proven Go2-style recipe (commanded_env) re-targeted at the mesh body:
PD position targets around the settled stance, exp-kernel velocity tracking
with the A.9 de-exploits (progress clamped at the commanded magnitude,
capped air-time on the pushrod tips, body-frame obs), P-ONLY servo through
the actuator torque path (the harness lesson: explicit -kd*qvel through
qfrc_applied is unstable against this mechanism's tiny smooth inertias; the
model's dof damping is the damping path).

Mechanism-specific rules honored:
  * resets place toe/pushrod via loop_consistent_pose(knee) — the exported
    hard requirement (naive knee placement violates the loop by ~26 mm and
    the near-hard couplings answer with ~kN forces);
  * per-axis action authority: yaw ±45 deg is the stride engine, pitch is
    ±25 deg, knee -90..+10 (kept mostly stowed by the pose term — walking
    happens on pushrod tips; the 3 N.m knee cannot body-lift at extension);
  * PD gains = the behavior suite's proven P-only set (2, 40, 6).

First-run caveats (documented, accepted): torso mass + SEA yaw stiffness +
yaw ROM are design placeholders; a policy trained today answers "can RL find
a gait in this workspace AT ALL", not "what will the hardware do".

  MESH_VMAX=0.6 python train_commanded.py --env mesh ...
"""
from __future__ import annotations

import os
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

MESH_VMAX = float(os.environ.get("MESH_VMAX", "0.6"))   # modest until the speed
                                                        # envelope is measured
TRACK_W = 5.0
TRACK_SIGMA = float(os.environ.get("MESH_TRACK_SIGMA", "0.25"))
PROGRESS_W = 12.0
ALIGN_W = 1.0
BACKWARD_W = 6.0
UPRIGHT_W = 0.5
AIRTIME_W = 1.0
AIRTIME_TARGET = 0.25          # s — slower cadence for the yaw-stride gait
AIRTIME_CAP = 2 * AIRTIME_TARGET
ACTRATE_W = 0.05
VELZ_W = 0.5
ANGXY_W = 0.1
POSE_W = 0.15
FOOT_CONTACT_Z = 0.03          # pushrod tip sphere r=0.012 on the floor + slack
FALL_Z = 0.25                  # settle band is 0.33-0.52; below 0.25 = collapsed
MIN_UP_Z = 0.4
CMD_HOLD_STEPS = 80
RESET_NOISE = 0.03             # rad on ACTUATED joints only (loop follows knee)
KP = (2.0, 40.0, 6.0)          # yaw, swing, knee — behavior-suite proven, P-only
AUTHORITY_FRAC = 0.6           # action ±1 commands this fraction of half-range


def _build():
    import jax
    import jax.numpy as jnp
    import mujoco
    from brax.envs.base import Env, State
    from mujoco import mjx

    from gen_mesh_robot_mjcf import build_mesh_robot

    class MeshCommandedEnv(Env):
        def __init__(self, frame_skip=5):
            m = mujoco.MjModel.from_xml_string(build_mesh_robot())
            self._mx = mjx.put_model(m)
            self._fs = frame_skip
            self._dt = frame_skip * m.opt.timestep     # 0.02 s at fleet dt=0.004
            self._nu = int(m.nu)                       # 12: (yaw, swing, knee) x 4
            jname = lambda j: mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_JOINT, j) or ""
            # actuated joint addressing, in actuator order
            aj = [int(m.actuator_trnid[a, 0]) for a in range(m.nu)]
            self._qa = jnp.array([int(m.jnt_qposadr[j]) for j in aj])
            self._da = jnp.array([int(m.jnt_dofadr[j]) for j in aj])
            self._jr = jnp.array([list(m.jnt_range[j]) for j in aj])
            self._gear = jnp.array([float(m.actuator_gear[a, 0]) for a in range(m.nu)])
            self._kp = jnp.array([KP[0], KP[1], KP[2]] * 4)   # actuator order is per-leg triples
            # passive loop joints (toe, slide) per leg — set from knee at reset
            self._knee_q = jnp.array([int(m.jnt_qposadr[j]) for j in aj[2::3]])
            legs = ("FL", "FR", "RL", "RR")
            jid = {jname(j): j for j in range(m.njnt)}
            self._toe_q = jnp.array([int(m.jnt_qposadr[jid[f"{L}_toe_hinge"]]) for L in legs])
            self._slide_q = jnp.array([int(m.jnt_qposadr[jid[f"{L}_pushrod_slide"]]) for L in legs])
            self._feet = jnp.array([mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_GEOM, f"{L}_foot")
                                    for L in legs])
            self._torso = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, "torso")
            self._q0 = jnp.array(m.qpos0)
            self._stand = self._q0[self._qa]
            self._authority = AUTHORITY_FRAC * 0.5 * (self._jr[:, 1] - self._jr[:, 0])
            self._obs_size = 2 * self._nu + 4 + 6 + 1 + self._nu + 3

        @property
        def observation_size(self): return self._obs_size
        @property
        def action_size(self): return self._nu
        @property
        def backend(self): return "mjx"

        @staticmethod
        def _yaw_rot(dx):
            w, x, y, z = dx.qpos[3], dx.qpos[4], dx.qpos[5], dx.qpos[6]
            yaw = jnp.arctan2(2.0 * (w * z + x * y), 1.0 - 2.0 * (y * y + z * z))
            c, s = jnp.cos(yaw), jnp.sin(yaw)
            return jnp.array([[c, s], [-s, c]])

        def _obs(self, dx, cmd, prev_action):
            R = self._yaw_rot(dx)
            return jnp.concatenate([
                dx.qpos[self._qa], dx.qvel[self._da],
                dx.qpos[3:7],
                jnp.concatenate([R @ dx.qvel[0:2], dx.qvel[2:6]]),
                dx.qpos[2:3],
                prev_action,
                jnp.concatenate([R @ cmd[:2], cmd[2:3]]),
            ])

        def _sample_cmd(self, rng):
            a, s, h = jax.random.split(rng, 3)
            dirs = jnp.array([[1.0, 0.0], [0.0, 1.0], [-1.0, 0.0], [0.0, -1.0]])
            d = dirs[jax.random.randint(a, (), 0, 4)]
            spd = jax.random.uniform(s, (), minval=0.3 * MESH_VMAX, maxval=MESH_VMAX)
            hold = (jax.random.uniform(h, ()) < 0.15).astype(jnp.float32)
            return jnp.concatenate([d * spd, jnp.zeros(1)]) * (1.0 - hold)

        def _metrics0(self):
            z = lambda: jnp.zeros(())
            return {"track": z(), "verr": z(), "align": z(), "speed": z(),
                    "progress": z(), "up": z(), "height": z()}

        def reset(self, rng):
            rng, nr, cr = jax.random.split(rng, 3)
            noise = jax.random.uniform(nr, (self._nu,), minval=-RESET_NOISE, maxval=RESET_NOISE)
            qpos = self._q0.at[self._qa].add(noise)
            # HARD RULE: passive loop joints from the knee via the model-exact
            # polynomials — anything else violates the near-hard couplings.
            for i in range(4):
                phi = qpos[self._knee_q[i]]
                qpos = qpos.at[self._toe_q[i]].set(_POLY_TOE_J(phi))
                qpos = qpos.at[self._slide_q[i]].set(_POLY_SLIDE_J(phi))
            dx = mjx.forward(self._mx, mjx.make_data(self._mx).replace(qpos=qpos))
            cmd = self._sample_cmd(cr)
            info = {"cmd": cmd, "rng": rng, "cmd_timer": jnp.zeros((), dtype=jnp.int32),
                    "air_time": jnp.zeros(4), "prev_action": jnp.zeros(self._nu),
                    "prev_feet_xy": dx.geom_xpos[self._feet][:, :2]}
            return State(dx, self._obs(dx, cmd, info["prev_action"]),
                         jnp.zeros(()), jnp.zeros(()), self._metrics0(), info)

        def step(self, state, action):
            cmd = state.info["cmd"]
            rng, cr = jax.random.split(state.info["rng"])
            timer = state.info["cmd_timer"] + 1
            resample = timer >= CMD_HOLD_STEPS
            cmd = jnp.where(resample, self._sample_cmd(cr), cmd)
            timer = jnp.where(resample, 0, timer)
            a = jnp.clip(action, -1.0, 1.0)
            target = jnp.clip(self._stand + a * self._authority,
                              self._jr[:, 0], self._jr[:, 1])
            dx = state.pipeline_state

            def sub(_, x):
                tau = self._kp * (target - x.qpos[self._qa])   # P-ONLY (see docstring)
                ctrl = jnp.clip(tau / self._gear, -1.0, 1.0)
                return mjx.step(self._mx, x.replace(ctrl=ctrl))
            dx = jax.lax.fori_loop(0, self._fs, sub, dx)

            v = dx.qvel[0:2]
            verr = jnp.sum((v - cmd[:2]) ** 2)
            track = jnp.exp(-verr / TRACK_SIGMA)
            up = 1.0 - 2.0 * (dx.qpos[4] ** 2 + dx.qpos[5] ** 2)
            cmd_norm = jnp.linalg.norm(cmd[:2])
            speed = jnp.linalg.norm(v)
            progress = jnp.dot(v, cmd[:2]) / (cmd_norm + 1e-6)
            align = jnp.dot(v, cmd[:2]) / (speed * cmd_norm + 1e-6)
            active = (cmd_norm > 0.05).astype(jnp.float32)
            foot_z = dx.geom_xpos[self._feet][:, 2]
            contact = foot_z < FOOT_CONTACT_Z
            air = state.info["air_time"]
            first_c = jnp.logical_and(contact, air > 0.0)
            air_rwd = jnp.sum((jnp.minimum(air, AIRTIME_CAP) - AIRTIME_TARGET)
                              * first_c.astype(jnp.float32))
            new_air = jnp.where(contact, 0.0, air + self._dt)
            act_rate = jnp.sum((a - state.info["prev_action"]) ** 2)
            pose_dev = jnp.sum((dx.qpos[self._qa] - self._stand) ** 2)
            progress_c = jnp.clip(progress, -cmd_norm, cmd_norm) / MESH_VMAX
            reward = (TRACK_W * track + UPRIGHT_W * up + 0.1
                      + ALIGN_W * active * jnp.clip(align, -1.0, 1.0)
                      + PROGRESS_W * active * progress_c
                      + AIRTIME_W * air_rwd * jnp.clip(cmd_norm / MESH_VMAX, 0, 1)
                      - BACKWARD_W * active * jnp.maximum(0.0, -progress)
                      - POSE_W * pose_dev - ACTRATE_W * act_rate
                      - VELZ_W * dx.qvel[2] ** 2
                      - ANGXY_W * (dx.qvel[3] ** 2 + dx.qvel[4] ** 2))
            done = jnp.where((dx.xpos[self._torso][2] < FALL_Z) | (up < MIN_UP_Z), 1.0, 0.0)
            # MERGE into existing metrics — brax's Evaluator injects a 'reward' key;
            # a fresh dict drops it and breaks the scan-carry pytree structure.
            metrics = {**state.metrics, "track": track, "verr": jnp.sqrt(verr), "align": align,
                       "speed": speed, "progress": progress, "up": up,
                       "height": dx.xpos[self._torso][2]}
            info = {**state.info, "cmd": cmd, "rng": rng, "cmd_timer": timer,
                    "air_time": new_air, "prev_action": a,
                    "prev_feet_xy": dx.geom_xpos[self._feet][:, :2]}
            return state.replace(pipeline_state=dx, obs=self._obs(dx, cmd, a),
                                 reward=reward, done=done, metrics=metrics, info=info)

    # jax-traceable evaluations of the SAME loop polynomials the model compiles
    import jax.numpy as jnp
    from gen_mesh_robot_mjcf import loop_polycoefs
    _cs, _cp, _, _ = loop_polycoefs()

    def _POLY_SLIDE_J(phi):
        return ((((_cs[4] * phi + _cs[3]) * phi + _cs[2]) * phi + _cs[1]) * phi + _cs[0])

    def _POLY_TOE_J(phi):
        return ((((_cp[4] * phi + _cp[3]) * phi + _cp[2]) * phi + _cp[1]) * phi + _cp[0])

    return MeshCommandedEnv
