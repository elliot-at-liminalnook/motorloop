# SPDX-License-Identifier: MIT
"""Gymnasium env: a standard MuJoCo locomotion robot driven through the motor +
FOC inner-loop envelope (notes/rl-environment-checklist.md §2).

A standard Gymnasium MuJoCo env (Ant-v5 = 8-joint quadruped locomotion) provides
the multibody robot, dynamics and reward; a thin action wrapper rolls off the
policy's per-joint torque command by the chosen motor's torque-speed envelope
(actuator.py) at each joint's current speed. So the policy learns under the real
motor's limits, and swapping the motor profile changes what the robot can do.
"""

from __future__ import annotations

import sys
from pathlib import Path

import gymnasium as gym
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from actuator import MotorActuator, make_actuator  # noqa: E402


class MotorEnvelopeWrapper(gym.Wrapper):
    """Scale each actuator's command by its motor's speed-dependent torque
    envelope (>= back-EMF rolloff + current limit). action stays in [-1,1]; the
    delivered torque is action * torque_scale(joint_speed)."""

    def __init__(self, env: gym.Env, actuator: MotorActuator):
        super().__init__(env)
        m = env.unwrapped.model
        # joint dof address feeding each actuator (for joint speed lookup)
        self._dof = [int(m.jnt_dofadr[int(m.actuator_trnid[a, 0])])
                     for a in range(m.nu)]
        self._act = actuator
        self.last_scale = np.ones(m.nu)

    def step(self, action):
        qvel = self.env.unwrapped.data.qvel
        w = np.array([qvel[i] for i in self._dof])
        scale = self._act.torque_scale(w)
        self.last_scale = scale
        return self.env.step(np.clip(action, -1.0, 1.0) * scale)


def make_env(robot: str = "HalfCheetah-v5", motor: str = "db42s03",
             gear_ratio: float = 2.0, vbus: float = 12.0,
             render_mode: str | None = None, **gym_kw):
    """A motor-enveloped locomotion env. `motor` selects the GM2804/DB42S03/EC 45
    profile (the env's actuator limits)."""
    env = gym.make(robot, render_mode=render_mode, **gym_kw)
    act = make_actuator(motor, gear_ratio=gear_ratio, vbus=vbus)
    return MotorEnvelopeWrapper(env, act)


# Register a stable id (notes/rl-environment-checklist.md §2): gym.make(
# "MotorloopLocomotion-v0", motor=..., gear_ratio=...) builds the enveloped env.
try:
    from gymnasium.envs.registration import register
    register(id="MotorloopLocomotion-v0", entry_point="foc_env:make_env",
             kwargs=dict(robot="HalfCheetah-v5", motor="db42s03", gear_ratio=2.0),
             max_episode_steps=1000)
except Exception:
    pass
