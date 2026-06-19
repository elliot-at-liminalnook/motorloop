# SPDX-License-Identifier: MIT
"""Combat-dodge env (notes/rl-combat-dodge-checklist.md): a quadruped that evades a
*weaponized ground robot* — the combat-robotics meta of a low spinner (a fast
horizontal kill-disk near the floor) or a low hammer-bot (an overhead descending
arc). Unlike the projectile env (dodge_env.py), the threat is a *persistent mobile
pursuer*; the task is to maintain a safe geometric relationship: keep the feet/legs
above the strike band when in the bite radius (high-step / tuck), stand off scaled
to how fast it closes, ride high so the blade passes under, and leap the plane when
cornered — all under the FOC motor envelope.

Built on the gymnasium Ant body (torso + 4 legs, 8 joints) with a low rangefinder
ring + a privileged pursuer track injected, the pursuer a kinematic mocap body that
chases the robot's CoM. Registers MotorloopCombat-v0.
"""

from __future__ import annotations

import glob
import os
import sys
from pathlib import Path

import gymnasium as gym
import mujoco
import numpy as np
from gymnasium import spaces

sys.path.insert(0, str(Path(__file__).resolve().parent))
from actuator import make_actuator  # noqa: E402

N_RAYS = 12
RAY_MAX = 6.0
R_W = 0.30                       # blade radius
BAND_Z = 0.07                   # blade centre height (spinner)
BLADE_HALF = 0.05               # blade half-height -> strike band [0.02, 0.12]
Z_HI = BAND_Z + BLADE_HALF      # top of the strike band the feet must clear
REACH = R_W + 0.10              # bite radius used by the reward
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
    """Inject a low rangefinder ring + a kinematic weaponized pursuer (mocap)."""
    xml = _ant_xml()
    # low rangefinder ring on the torso, tilted down to catch a ground-hugging bot
    sites = []
    for i in range(N_RAYS):
        th = 2 * np.pi * i / N_RAYS
        sites.append(f'<site name="rf{i}" pos="0 0 0" size="0.01" '
                     f'zaxis="{np.cos(th):.4f} {np.sin(th):.4f} -0.30"/>')
    xml = xml.replace('<geom name="torso_geom" pos="0 0 0" size="0.25" type="sphere"/>',
                      '<geom name="torso_geom" pos="0 0 0" size="0.25" type="sphere"/>\n      '
                      + "\n      ".join(sites))
    # the pursuer: a kinematic mocap body (chassis box + low blade disk) that we
    # teleport toward the robot each substep; mocap geoms collide with + push the
    # robot (knockback) but are themselves immovable - a heavy spinner.
    adv = (f'<body name="adv" mocap="true" pos="3 0 {BAND_Z}">'
           f'<geom name="adv_blade" type="cylinder" size="{R_W} {BLADE_HALF}" '
           f'pos="0 0 0" contype="1" conaffinity="1" rgba="0.9 0.1 0.1 1"/>'
           f'<geom name="adv_chassis" type="box" size="0.18 0.18 0.08" '
           f'pos="0 0 0.13" contype="1" conaffinity="1" rgba="0.2 0.2 0.25 1"/>'
           f'</body>')
    xml = xml.replace("</worldbody>", adv + "\n  </worldbody>")
    sensors = "<sensor>\n" + "\n".join(
        f'    <rangefinder name="rfs{i}" site="rf{i}"/>' for i in range(N_RAYS)) \
        + "\n  </sensor>"
    xml = xml.replace("</mujoco>", sensors + "\n</mujoco>")
    return xml


class CombatDodgeEnv(gym.Env):
    metadata = {"render_modes": []}

    def __init__(self, motor: str = "db42s03", gear_ratio: float = 2.0,
                 vbus: float = 12.0, difficulty: float = 0.3,
                 weapon: str = "spinner", frame_skip: int = 5,
                 seed: int | None = None):
        self.model = mujoco.MjModel.from_xml_string(build_scene())
        self.data = mujoco.MjData(self.model)
        self.frame_skip = frame_skip
        self.difficulty = float(difficulty)
        self.weapon = weapon
        self.act = make_actuator(motor, gear_ratio=gear_ratio, vbus=vbus)
        self._rng = np.random.default_rng(seed)
        self._dt = self.model.opt.timestep

        m = self.model
        self._foot_gid = [mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_GEOM, g)
                          for g in FOOT_GEOMS]
        self._leg_gid = {mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_GEOM, g)
                         for g in LEG_GEOMS}
        self._blade_gid = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_GEOM, "adv_blade")
        self._chassis_gid = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_GEOM, "adv_chassis")
        self._weapon_gid = {self._blade_gid, self._chassis_gid}
        self._adv_bid = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, "adv")
        self._act_dof = [int(m.jnt_dofadr[int(m.actuator_trnid[a, 0])])
                         for a in range(m.nu)]

        self._adv_pos = np.array([3.0, 0.0, BAND_Z])
        self._adv_vel = np.zeros(3)
        self._prev_rays = np.full(N_RAYS, RAY_MAX)
        self._prev_action = np.zeros(m.nu)

        self.action_space = spaces.Box(-1.0, 1.0, (m.nu,), np.float32)
        obs = self._get_obs()
        self.observation_space = spaces.Box(-np.inf, np.inf, obs.shape, np.float32)

    # ---- pursuer ----
    def _adv_speed(self):
        return 1.0 + 3.0 * self.difficulty          # m/s, curriculum

    def _drive_adv(self):
        """Move the mocap pursuer one substep toward the robot (pure pursuit)."""
        com = self.data.qpos[0:3].copy()
        if self.weapon == "hammer":
            target = np.array([com[0], com[1], 0.55])        # hover above, then slam
            phase = (self.data.time * (1.0 + self.difficulty)) % 1.0
            target[2] = 0.55 - 0.45 * max(0.0, np.sin(np.pi * phase))  # descend arc
        else:                                                 # spinner: low chase
            target = np.array([com[0], com[1], BAND_Z])
        step = self._adv_speed() * self._dt
        delta = target - self._adv_pos
        dn = np.linalg.norm(delta)
        if dn > step:
            delta = delta / dn * step
        new = self._adv_pos + delta
        self._adv_vel = (new - self._adv_pos) / self._dt
        self._adv_pos = new
        self.data.mocap_pos[0] = self._adv_pos

    # ---- obs ----
    def _rays(self):
        r = np.array([self.data.sensordata[
            self.model.sensor_adr[mujoco.mj_name2id(
                self.model, mujoco.mjtObj.mjOBJ_SENSOR, f"rfs{i}")]]
            for i in range(N_RAYS)])
        return np.where((r < 0) | (r > RAY_MAX), RAY_MAX, r)

    def _geom_low_z(self, gid):
        """Lowest world-z point of a capsule geom (its frame z-axis is the long
        axis): centre - |axis_z|*half_length - radius. The ankle-geom *centre*
        sits high on a diagonal leg; the strike happens at the foot tip."""
        R = self.data.geom_xmat[gid].reshape(3, 3)
        rad, half = self.model.geom_size[gid][0], self.model.geom_size[gid][1]
        return self.data.geom_xpos[gid][2] - abs(R[2, 2]) * half - rad

    def _foot_state(self):
        """per foot: (tip height, horizontal distance to pursuer, in-reach flag)."""
        out = []
        for g in self._foot_gid:
            p = self.data.geom_xpos[g]
            d = np.linalg.norm(p[:2] - self._adv_pos[:2])
            out.append((self._geom_low_z(g), d, 1.0 if d < REACH else 0.0))
        return out

    def _get_obs(self):
        d, m = self.data, self.model
        qpos = d.qpos[7:7 + m.nu]
        qvel = d.qvel[6:6 + m.nu]
        torso_quat = d.qpos[3:7]
        torso_vel = d.qvel[0:6]
        rays = self._rays()
        closing = self._prev_rays - rays
        self._prev_rays = rays
        rel = self._adv_pos[:2] - d.qpos[0:2]                  # pursuer rel pos
        rel_v = self._adv_vel[:2] - d.qvel[0:2]
        weap = [1.0, 0.0] if self.weapon == "spinner" else [0.0, 1.0]
        track = np.concatenate([rel, rel_v, [R_W, Z_HI], weap])
        feet = np.array(self._foot_state()).reshape(-1)        # 4 x 3
        return np.concatenate([
            qpos, qvel, torso_quat, torso_vel, [d.qpos[2]],
            rays / RAY_MAX, closing, track, feet]).astype(np.float32)

    def _up(self):
        q = self.data.qpos[3:7]
        return 1 - 2 * (q[1] ** 2 + q[2] ** 2)

    def _strike(self):
        for c in range(self.data.ncon):
            g1, g2 = self.data.contact[c].geom1, self.data.contact[c].geom2
            if (g1 in self._weapon_gid and g2 in self._leg_gid) or \
               (g2 in self._weapon_gid and g1 in self._leg_gid):
                return True
        return False

    # ---- gym API ----
    def reset(self, *, seed=None, options=None):
        if seed is not None:
            self._rng = np.random.default_rng(seed)
        mujoco.mj_resetData(self.model, self.data)
        self.data.qpos[2] = 0.55
        self.data.qpos[7:7 + self.model.nu] += self._rng.uniform(
            -0.1, 0.1, self.model.nu)
        ang = self._rng.uniform(0, 2 * np.pi)
        self._adv_pos = np.array([3.0 * np.cos(ang), 3.0 * np.sin(ang), BAND_Z])
        self._adv_vel = np.zeros(3)
        self.data.mocap_pos[0] = self._adv_pos
        self._prev_action = np.zeros(self.model.nu)
        mujoco.mj_forward(self.model, self.data)
        self._prev_rays = self._rays()
        return self._get_obs(), {}

    def _ss(self, a, b, x):
        t = np.clip((x - a) / (b - a + 1e-9), 0.0, 1.0)
        return t * t * (3 - 2 * t)

    def step(self, action):
        action = np.clip(action, -1.0, 1.0)
        w = np.array([self.data.qvel[i] for i in self._act_dof])
        self.data.ctrl[:] = action * self.act.torque_scale(w)   # motor envelope
        struck = False
        for _ in range(self.frame_skip):
            self._drive_adv()
            mujoco.mj_step(self.model, self.data)
            if self._strike():
                struck = True

        com = self.data.qpos[0:3]
        com_v = self.data.qvel[0:3]
        up, h = self._up(), com[2]
        d_com = np.linalg.norm(com[:2] - self._adv_pos[:2])
        rel = com[:2] - self._adv_pos[:2]
        rel_v = com_v[:2] - self._adv_vel[:2]
        closing = max(0.0, -float(rel @ rel_v) / (d_com + 1e-6))   # pursuer gaining
        urg = closing * self._ss(REACH + 0.6, REACH, d_com)        # urgency

        # 1. height clearance: keep each foot TIP above the strike band when in reach
        r_clear = 0.0
        foot_low_inreach = False
        for g in self._foot_gid:
            dg = np.linalg.norm(self.data.geom_xpos[g][:2] - self._adv_pos[:2])
            inr = self._ss(REACH + 0.3, REACH - 0.1, dg)
            margin = self._geom_low_z(g) - Z_HI
            r_clear += inr * (4.0 * np.clip(margin, 0, 0.15) / 0.15
                              - 8.0 * max(0.0, -margin))
            if inr > 0.5 and margin < 0:
                foot_low_inreach = True
        # 2. standoff (run with margin), 3. belly clearance, 4. leap the plane
        r_standoff = 1.5 * urg * np.clip(d_com - REACH, 0.0, 1.0)
        r_body = 1.0 * urg * np.clip(h - 0.45, 0.0, 0.25) / 0.25
        r_leap = 2.0 * urg * max(0.0, com_v[2]) if foot_low_inreach else 0.0
        if self.weapon == "hammer":   # hammer threatens from above: retreat laterally
            r_standoff += 1.0 * urg * np.clip(d_com - 0.3, 0.0, 1.0)
            r_leap = 0.0
        # 5. graceful + alive
        calm = float(np.exp(-urg))
        r_settle = 0.6 * calm * (up - 0.15 * float(np.linalg.norm(self.data.qvel[3:6])))
        r_jerk = -0.03 * float(np.square(action - self._prev_action).sum())
        self._prev_action = action.copy()

        reward = (1.0 + r_clear + r_standoff + r_body + r_leap
                  + r_settle + r_jerk - 50.0 * float(struck))
        fell = (up < 0.2) or (h < 0.25)
        terminated = bool(fell or struck)
        info = {"strike": struck, "d_com": d_com,
                "min_foot_clear": min(self._geom_low_z(g) - Z_HI
                                      for g in self._foot_gid),
                "in_reach": any(np.linalg.norm(self.data.geom_xpos[g][:2]
                                - self._adv_pos[:2]) < REACH for g in self._foot_gid)}
        return self._get_obs(), float(reward), terminated, False, info

    def set_difficulty(self, d):
        self.difficulty = float(np.clip(d, 0.0, 1.0))


def make_combat_env(**kw):
    return CombatDodgeEnv(**kw)


try:
    from gymnasium.envs.registration import register
    register(id="MotorloopCombat-v0", entry_point="combat_env:make_combat_env",
             max_episode_steps=1000)
except Exception:
    pass
