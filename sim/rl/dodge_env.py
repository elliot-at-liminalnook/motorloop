# SPDX-License-Identifier: MIT
"""Dodge-and-balance env (notes/rl-dodge-balance-checklist.md): a quadruped that
perceives objects accelerating toward its legs (rangefinder "lidar"), dodges by
moving the threatened leg, and balances on the others.

Built on the gymnasium Ant body (torso + 4 legs, 8 joints) with rangefinder rays
+ projectile free-bodies injected, and the joints driven through the FOC motor
envelope (actuator.py). Perception = a horizontal fan of rangefinders + their
closing rate (cheap, no camera render). Reward = balance + survive + dodge -
hit - effort, with a difficulty curriculum.
"""

from __future__ import annotations

import sys
from pathlib import Path

import glob
import os

import gymnasium as gym
import mujoco
import numpy as np
from gymnasium import spaces

sys.path.insert(0, str(Path(__file__).resolve().parent))
from actuator import make_actuator  # noqa: E402

N_RAYS = 12
N_PROJ = 4
RAY_MAX = 6.0
FOOT_GEOMS = ["left_ankle_geom", "right_ankle_geom", "third_ankle_geom",
              "fourth_ankle_geom"]
LEG_GEOMS = FOOT_GEOMS + ["aux_1_geom", "aux_2_geom", "aux_3_geom", "aux_4_geom",
                          "left_leg_geom", "right_leg_geom", "back_leg_geom",
                          "rightback_leg_geom"]


def _ant_xml() -> str:
    import gymnasium
    return open(glob.glob(os.path.dirname(gymnasium.__file__)
                          + "/envs/mujoco/assets/ant.xml")[0]).read()


def build_scene() -> str:
    """Inject rangefinder sites/sensors + projectile bodies into the Ant scene."""
    xml = _ant_xml()
    # rangefinder sites on the torso, a horizontal fan slightly tilted down
    sites = []
    for i in range(N_RAYS):
        th = 2 * np.pi * i / N_RAYS
        sites.append(f'<site name="rf{i}" pos="0 0 0" size="0.01" '
                     f'zaxis="{np.cos(th):.4f} {np.sin(th):.4f} -0.15"/>')
    xml = xml.replace('<geom name="torso_geom" pos="0 0 0" size="0.25" type="sphere"/>',
                      '<geom name="torso_geom" pos="0 0 0" size="0.25" type="sphere"/>\n      '
                      + "\n      ".join(sites))
    # projectile free-bodies, parked far away; contype/conaffinity to hit legs
    proj = []
    for i in range(N_PROJ):
        proj.append(
            f'<body name="proj{i}" pos="{20+i} 0 0.3">'
            f'<freejoint name="pj{i}"/>'
            f'<geom name="pjg{i}" type="sphere" size="0.12" density="300" '
            f'contype="1" conaffinity="1" rgba="0.9 0.2 0.2 1"/></body>')
    xml = xml.replace("</worldbody>", "\n    ".join(proj) + "\n  </worldbody>")
    # rangefinder sensors
    sensors = "<sensor>\n" + "\n".join(
        f'    <rangefinder name="rfs{i}" site="rf{i}"/>' for i in range(N_RAYS)) \
        + "\n  </sensor>"
    xml = xml.replace("</mujoco>", sensors + "\n</mujoco>")
    return xml


class DodgeQuadrupedEnv(gym.Env):
    metadata = {"render_modes": []}

    def __init__(self, motor: str = "db42s03", gear_ratio: float = 2.0,
                 vbus: float = 12.0, difficulty: float = 0.3,
                 frame_skip: int = 5, seed: int | None = None):
        self.model = mujoco.MjModel.from_xml_string(build_scene())
        self.data = mujoco.MjData(self.model)
        self.frame_skip = frame_skip
        self.difficulty = float(difficulty)
        self.act = make_actuator(motor, gear_ratio=gear_ratio, vbus=vbus)
        self._rng = np.random.default_rng(seed)

        m = self.model
        self._foot_gid = [mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_GEOM, g)
                          for g in FOOT_GEOMS]
        self._leg_gid = {mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_GEOM, g)
                         for g in LEG_GEOMS}
        self._proj_gid = [mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_GEOM, f"pjg{i}")
                          for i in range(N_PROJ)]
        self._proj_qadr = [m.jnt_qposadr[mujoco.mj_name2id(
            m, mujoco.mjtObj.mjOBJ_JOINT, f"pj{i}")] for i in range(N_PROJ)]
        self._proj_dadr = [m.jnt_dofadr[mujoco.mj_name2id(
            m, mujoco.mjtObj.mjOBJ_JOINT, f"pj{i}")] for i in range(N_PROJ)]
        # 8 actuated hinge joints' dof addresses (for the motor envelope)
        self._act_dof = [int(m.jnt_dofadr[int(m.actuator_trnid[a, 0])])
                         for a in range(m.nu)]
        self._proj_active = [False] * N_PROJ
        self._proj_target = [0] * N_PROJ
        self._cooldown = 0
        self._prev_rays = np.full(N_RAYS, RAY_MAX)

        self.action_space = spaces.Box(-1.0, 1.0, (m.nu,), np.float32)
        obs = self._get_obs()
        self.observation_space = spaces.Box(-np.inf, np.inf, obs.shape, np.float32)

    # ---- threats ----
    def _launch(self, i):
        leg = int(self._rng.integers(4))
        foot = self.data.geom_xpos[self._foot_gid[leg]].copy()
        # spawn offset out from the foot, at leg height, then aim at the foot
        ang = self._rng.uniform(0, 2 * np.pi)
        d = 2.2
        spawn = foot + np.array([d * np.cos(ang), d * np.sin(ang), 0.25])
        spawn[2] = max(spawn[2], 0.12)
        speed = 3.0 + 6.0 * self.difficulty
        vel = (foot - spawn); vel /= (np.linalg.norm(vel) + 1e-6); vel *= speed
        q, dq = self._proj_qadr[i], self._proj_dadr[i]
        self.data.qpos[q:q + 3] = spawn
        self.data.qpos[q + 3:q + 7] = [1, 0, 0, 0]
        self.data.qvel[dq:dq + 3] = vel
        self.data.qvel[dq + 3:dq + 6] = 0
        self._proj_active[i] = True
        self._proj_target[i] = leg

    def _park(self, i):
        q = self._proj_qadr[i]
        self.data.qpos[q:q + 3] = [20 + i, 0, -5]
        self.data.qvel[self._proj_dadr[i]:self._proj_dadr[i] + 6] = 0
        self._proj_active[i] = False

    def _update_threats(self):
        # despawn projectiles that passed/landed; launch new ones on cooldown
        for i in range(N_PROJ):
            if self._proj_active[i]:
                p = self.data.qpos[self._proj_qadr[i]:self._proj_qadr[i] + 3]
                if p[2] < 0.05 or np.linalg.norm(p[:2]) > 5.0:
                    self._park(i)
        self._cooldown -= 1
        rate = 0.2 + 0.8 * self.difficulty
        if self._cooldown <= 0 and self._rng.random() < rate:
            free = [i for i in range(N_PROJ) if not self._proj_active[i]]
            if free:
                self._launch(free[0])
                self._cooldown = int(30 * (1.2 - self.difficulty))

    # ---- obs ----
    def _rays(self):
        r = np.array([self.data.sensordata[
            self.model.sensor_adr[mujoco.mj_name2id(
                self.model, mujoco.mjtObj.mjOBJ_SENSOR, f"rfs{i}")]]
            for i in range(N_RAYS)])
        r = np.where((r < 0) | (r > RAY_MAX), RAY_MAX, r)
        return r

    def _get_obs(self):
        d, m = self.data, self.model
        qpos = d.qpos[7:7 + m.nu]            # 8 hinge angles
        qvel = d.qvel[6:6 + m.nu]            # 8 hinge speeds
        torso_quat = d.qpos[3:7]
        torso_vel = d.qvel[0:6]              # lin+ang
        rays = self._rays()
        closing = (self._prev_rays - rays)   # +ve = approaching
        self._prev_rays = rays
        return np.concatenate([qpos, qvel, torso_quat, torso_vel,
                               [d.qpos[2]], rays / RAY_MAX, closing]).astype(np.float32)

    def _up(self):
        # torso z-axis world component (1 = upright)
        q = self.data.qpos[3:7]
        return 1 - 2 * (q[1] ** 2 + q[2] ** 2)

    def _hit(self):
        for c in range(self.data.ncon):
            g1, g2 = self.data.contact[c].geom1, self.data.contact[c].geom2
            if (g1 in self._proj_gid and g2 in self._leg_gid) or \
               (g2 in self._proj_gid and g1 in self._leg_gid):
                return True
        return False

    def _foot_clearances(self):
        # min distance from each active projectile to its target foot
        cl = []
        for i in range(N_PROJ):
            if self._proj_active[i]:
                p = self.data.qpos[self._proj_qadr[i]:self._proj_qadr[i] + 3]
                f = self.data.geom_xpos[self._foot_gid[self._proj_target[i]]]
                cl.append(np.linalg.norm(p - f))
        return cl

    # ---- gym API ----
    def reset(self, *, seed=None, options=None):
        if seed is not None:
            self._rng = np.random.default_rng(seed)
        mujoco.mj_resetData(self.model, self.data)
        self.data.qpos[2] = 0.55
        self.data.qpos[7:7 + self.model.nu] += self._rng.uniform(
            -0.1, 0.1, self.model.nu)
        for i in range(N_PROJ):
            self._park(i)
        self._cooldown = 20
        mujoco.mj_forward(self.model, self.data)
        self._prev_rays = self._rays()
        return self._get_obs(), {}

    def step(self, action):
        action = np.clip(action, -1.0, 1.0)
        w = np.array([self.data.qvel[i] for i in self._act_dof])
        self.data.ctrl[:] = action * self.act.torque_scale(w)   # motor envelope
        for _ in range(self.frame_skip):
            self._update_threats()
            mujoco.mj_step(self.model, self.data)
        up, h = self._up(), self.data.qpos[2]
        hit = self._hit()
        clear = self._foot_clearances()
        reward = (1.0                                   # alive
                  + 1.0 * up                            # stay upright
                  - 2.0 * abs(h - 0.55)                 # hold height
                  - 0.005 * float(np.square(action).sum())   # effort
                  - 25.0 * hit                          # being hit (strong: dodge!)
                  + 0.8 * sum(min(c, 1.0) for c in clear))   # keep legs clear
        fell = (up < 0.2) or (h < 0.25)
        terminated = bool(fell)
        return self._get_obs(), float(reward), terminated, False, {"hit": hit}

    def set_difficulty(self, d):
        self.difficulty = float(np.clip(d, 0.0, 1.0))


def make_dodge_env(**kw):
    return DodgeQuadrupedEnv(**kw)


try:
    from gymnasium.envs.registration import register
    register(id="MotorloopDodge-v0", entry_point="dodge_env:make_dodge_env",
             max_episode_steps=1000)
except Exception:
    pass
