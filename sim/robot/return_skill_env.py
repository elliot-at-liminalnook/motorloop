# SPDX-License-Identifier: MIT
"""Low-dimensional return-to-origin control over the MuJoCo-Warp walker."""

from __future__ import annotations

import os

import numpy as np
import torch

from locomotion_spec import VMAX
from walker_warp_env import WalkerWarpEnv

ACTION_DIM = 5
OBS_DIM = 16
DEFAULT_STARTS = ((0.05442, 0.38643), (0.06719, 0.38646), (0.04125, 0.36983),
                  (0.04066, 0.35510), (0.02561, 0.33790), (0.00388, 0.32333),
                  (0.01506, 0.32812), (-0.00865, 0.31477))


def parse_xy_list(text):
    rows = []
    for chunk in text.split(";"):
        if chunk.strip():
            values = [float(value) for value in chunk.split(",")]
            rows.append(values[-2:])
    return np.asarray(rows or DEFAULT_STARTS, dtype=np.float32)


class ReturnSkillWarpEnv:
    def __init__(self, nworld=1, seed=0, device=None, episode_length=260, **kwargs):
        self._base = WalkerWarpEnv(nworld, seed=seed, device=device,
                                   episode_length=episode_length, **kwargs)
        self.nworld, self.device = self._base.nworld, self._base.device
        self.obs_dim, self.priv_dim, self.act_dim = OBS_DIM, self._base.priv_dim, ACTION_DIM
        self._dt, self._gen = self._base._dt, self._base._gen
        self._target = torch.zeros((nworld, 2), device=self.device)
        self._prev_distance = torch.zeros(nworld, device=self.device)
        self._previous_action = torch.zeros((nworld, ACTION_DIM), device=self.device)
        self.model_hash, self.gait_loaded = self._base.model_hash, self._base.gait_loaded
        self._wm, self._wd = self._base._wm, self._base._wd
        self.qpos, self.qvel = self._base.qpos, self._base.qvel
        self.qacc_warmstart, self.sim_time = self._base.qacc_warmstart, self._base.sim_time

    @property
    def observation_size(self): return self.obs_dim
    @property
    def action_size(self): return self.act_dim
    @property
    def backend(self): return "mujoco_warp"
    def privileged(self): return self._base.privileged()

    def observe(self):
        delta = self._target - self.qpos[:, :2]
        distance = torch.linalg.vector_norm(delta, dim=-1, keepdim=True)
        velocity = self.qvel[:, :2] / VMAX
        base = self._base.observe()
        return torch.cat((delta, distance, velocity, self._previous_action,
                          base[:, 24:28], base[:, 30:31], base[:, 34:35]), dim=-1)

    def reset(self):
        self._base.reset()
        starts = torch.as_tensor(parse_xy_list(os.environ.get("RETURN_SKILL_STARTS", "")),
                                 device=self.device)
        indices = torch.arange(self.nworld, device=self.device) % len(starts)
        self.qpos[:, :2] = starts[indices]
        self._prev_distance = torch.linalg.vector_norm(self.qpos[:, :2], dim=-1)
        self._previous_action.zero_()
        return self.observe()

    def step(self, action, alpha=1.0, imit_anneal=0.0):
        action = action.clamp(-1, 1)
        direction = -self.qpos[:, :2]
        norm = torch.linalg.vector_norm(direction, dim=-1, keepdim=True).clamp(min=1e-6)
        command = direction / norm * (0.25 + 0.75 * (action[:, 4:5] + 1.0) * 0.5) * VMAX
        command += action[:, :2] * 0.25 * VMAX
        self._base._cmd[:, :2] = command
        self._base._cmd[:, 2] = action[:, 2] * 0.35
        self._base._timer.zero_()
        motor = torch.zeros((self.nworld, self._base.act_dim), device=self.device)
        _, _, done, info = self._base.step(motor, alpha=alpha, imit_anneal=imit_anneal)
        distance = torch.linalg.vector_norm(self.qpos[:, :2], dim=-1)
        progress = self._prev_distance - distance
        reward = 18.0 * progress - 1.5 * distance - 0.04 * (action ** 2).sum(-1)
        reward += 4.0 * (distance < 0.07).float()
        self._prev_distance = distance
        self._previous_action = action
        info.update(return_dist=distance, return_progress=progress)
        return self.observe(), reward, done, info


def _build():
    return ReturnSkillWarpEnv
