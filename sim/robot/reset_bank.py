# SPDX-License-Identifier: MIT
"""Torch-native banked reset wrapper for batched MuJoCo-Warp environments."""

from __future__ import annotations

import math

import torch
import warp as wp
import mujoco_warp as mjwp

from constants import LAUNCH_SPEED


class BankedAutoResetWrapper:
    """Replace post-terminal states with a deterministic diverse reset bank."""

    def __init__(self, env, bank_key=0, bank_size=256, canonical_frac=0.3,
                 launch_speed=LAUNCH_SPEED, root_dof=0):
        self.env = env
        self._K = int(bank_size)
        self._stride = next(value for value in (17, 13, 11, 7) if math.gcd(value, self._K) == 1)
        generator = torch.Generator(device=env.device).manual_seed(int(bank_key))
        q0 = getattr(env, "_q0_64", getattr(env, "_q0", None))
        if q0 is None:
            q0 = env.qpos[0].clone()
        q0 = q0.float().to(env.device)
        self._bank_qpos = q0.expand(self._K, -1).clone()
        self._bank_qvel = torch.zeros((self._K, env.qvel.shape[1]), device=env.device)
        if hasattr(env, "_qa"):
            noise = torch.rand((self._K, len(env._qa)), generator=generator,
                               device=env.device) * 0.1 - 0.05
            self._bank_qpos[:, env._qa] += noise
        launch_count = int(self._K * (1.0 - canonical_frac))
        if launch_count and root_dof is not None:
            speed = torch.empty(launch_count, device=env.device).uniform_(
                float(launch_speed[0]), float(launch_speed[1]), generator=generator)
            angle = torch.empty(launch_count, device=env.device).uniform_(
                -torch.pi, torch.pi, generator=generator)
            self._bank_qvel[:launch_count, root_dof] = speed * angle.cos()
            self._bank_qvel[:launch_count, root_dof + 1] = speed * angle.sin()
        self._cursor = torch.arange(env.nworld, device=env.device) % self._K

    def __getattr__(self, name):
        return getattr(self.env, name)

    def reset(self):
        self._cursor = (self._cursor + self._stride) % self._K
        _ = self.env.reset()
        self._apply(torch.ones(self.env.nworld, dtype=torch.bool, device=self.env.device))
        return self.env.observe()

    def _apply(self, mask):
        selected_qpos = self._bank_qpos[self._cursor]
        selected_qvel = self._bank_qvel[self._cursor]
        self.env.qpos.copy_(torch.where(mask[:, None], selected_qpos, self.env.qpos))
        self.env.qvel.copy_(torch.where(mask[:, None], selected_qvel, self.env.qvel))
        with wp.ScopedDevice(self.env._wp_device):
            mjwp.forward(self.env._wm, self.env._wd)
        if hasattr(self.env, "_refresh_outputs"):
            self.env._refresh_outputs()

    def step(self, action, **kwargs):
        obs, reward, done, info = self.env.step(action, **kwargs)
        mask = done.bool()
        self._cursor = torch.where(mask, (self._cursor + self._stride) % self._K,
                                   self._cursor)
        self._apply(mask)
        info["priv"] = self.env.privileged()
        return self.env.observe(), reward, done, info


def make_wrap_fn(bank_key=0, bank_size=256, canonical_frac=0.3,
                 launch_speed=LAUNCH_SPEED, root_dof=0):
    return lambda env, **_: BankedAutoResetWrapper(
        env, bank_key, bank_size, canonical_frac, launch_speed, root_dof)
