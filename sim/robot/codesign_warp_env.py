# SPDX-License-Identifier: MIT
"""Design-conditioned MuJoCo-Warp environments without per-world model fiction."""

from __future__ import annotations

import hashlib

import numpy as np
import torch

from design_codec import DESIGN_DIM, apply_fast_mujoco
from walker_warp_env import WalkerWarpEnv


DEFAULT_DESIGN_BANK = np.asarray([
    (0.0, 0.0, 0.0),
    (1.0, 0.0, 1.0),
    (0.0, 1.0, 1.0),
    (1.0, 1.0, 0.0),
    (0.5, 0.5, 0.5),
], dtype=np.float32)


class CodesignWarpEnv(WalkerWarpEnv):
    """A real fixed design: model fields are edited before Warp upload."""

    architecture_task_dim = DESIGN_DIM

    def __init__(self, nworld: int, seed: int = 0, device: str | None = None,
                 episode_length: int | None = 800, design=(0.5, 0.5, 0.5), **kwargs):
        self.design = torch.as_tensor(design, dtype=torch.float32,
                                      device=torch.device("cuda:0" if (
                                          (device is None and torch.cuda.is_available())
                                          or str(device).startswith("cuda")) else "cpu"))
        design_np = self.design.detach().cpu().numpy()
        super().__init__(nworld, seed=seed, device=device, episode_length=episode_length,
                         model_transform=lambda model: apply_fast_mujoco(model, design_np),
                         **kwargs)

    @property
    def obs_dim(self):
        return super().obs_dim + DESIGN_DIM

    def observe(self):
        base = super().observe()
        return torch.cat((base, self.design.expand(self.nworld, -1)), dim=-1)


class DesignEnsembleWarpEnv:
    """One policy over several actual compiled models, grouped on one device.

    MuJoCo-Warp models are shared across worlds, so pretending to vary masses in
    a single model would not simulate the requested designs. This wrapper uses
    one captured model per design and concatenates their device tensors for the
    learner. It costs one graph launch per design but preserves real physics.
    """

    action_semantics = "pd_target@50hz:lowpass+torque_speed_v1"
    reward_semantics = "design_ensemble:velocity_command:v1"
    architecture_task_dim = DESIGN_DIM

    def __init__(self, nworld: int, seed: int = 0, device: str | None = None,
                 episode_length: int | None = 800, designs=DEFAULT_DESIGN_BANK):
        designs = np.asarray(designs, dtype=np.float32).reshape(-1, DESIGN_DIM)
        if nworld < len(designs):
            designs = designs[:nworld]
        counts = np.full(len(designs), nworld // len(designs), dtype=int)
        counts[:nworld % len(designs)] += 1
        self.envs = [CodesignWarpEnv(int(count), seed=seed + i, device=device,
                                     episode_length=episode_length, design=design)
                     for i, (count, design) in enumerate(zip(counts, designs)) if count]
        first = self.envs[0]
        self.nworld = nworld
        self.device = first.device
        self.obs_dim, self.priv_dim, self.act_dim = first.obs_dim, first.priv_dim, first.act_dim
        self._dt = first._dt
        self._episode_length = episode_length
        self.gait_loaded = all(env.gait_loaded for env in self.envs)
        self.model_hash = hashlib.sha256(
            "|".join(env.model_hash for env in self.envs).encode()).hexdigest()[:16]
        self._gen = first._gen

    def observe(self):
        return torch.cat([env.observe() for env in self.envs], dim=0)

    def privileged(self):
        return torch.cat([env.privileged() for env in self.envs], dim=0)

    def reset(self):
        return torch.cat([env.reset() for env in self.envs], dim=0)

    def step(self, action, alpha=1.0, imit_anneal=0.0):
        chunks = action.split([env.nworld for env in self.envs], dim=0)
        results = [env.step(chunk, alpha=alpha, imit_anneal=imit_anneal)
                   for env, chunk in zip(self.envs, chunks)]
        obs = torch.cat([result[0] for result in results], dim=0)
        reward = torch.cat([result[1] for result in results], dim=0)
        done = torch.cat([result[2] for result in results], dim=0)
        keys = results[0][3].keys()
        info = {key: torch.cat([result[3][key] for result in results], dim=0)
                for key in keys}
        return obs, reward, done, info
