# SPDX-License-Identifier: MIT
"""Torch-native Random Network Distillation for on-device Warp rollouts."""

from __future__ import annotations

import copy
from dataclasses import dataclass

import torch
import torch.nn as nn


class RandomTargetNetwork(nn.Module):
    def __init__(self, feature_dim, hidden_dim=256, output_dim=128):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(feature_dim, hidden_dim), nn.ReLU(),
                                 nn.Linear(hidden_dim, output_dim))

    def forward(self, value):
        return self.net(value)


class PredictorNetwork(nn.Module):
    def __init__(self, feature_dim, hidden_dim=256, output_dim=128):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(feature_dim, hidden_dim), nn.ReLU(),
                                 nn.Linear(hidden_dim, hidden_dim), nn.ReLU(),
                                 nn.Linear(hidden_dim, output_dim))

    def forward(self, value):
        return self.net(value)


@dataclass
class RND:
    feature_dim: int
    target: nn.Module
    predictor: nn.Module
    optimizer: torch.optim.Optimizer

    @torch.no_grad()
    def novelty(self, feature):
        single = feature.ndim == 1
        feature = feature[None] if single else feature
        result = ((self.target(feature) - self.predictor(feature)) ** 2).mean(-1)
        return result[0] if single else result

    def update(self, feature):
        feature = feature[None] if feature.ndim == 1 else feature
        with torch.no_grad():
            target = self.target(feature)
        loss = ((self.predictor(feature) - target) ** 2).mean()
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        return loss.detach()


def make_rnd(feature_dim: int, hidden_dim: int = 128, output_dim: int = 64,
             lr: float = 1e-3, key=None, device="cpu") -> RND:
    seed = int(key) if isinstance(key, int) else 0
    with torch.random.fork_rng():
        torch.manual_seed(seed)
        target = RandomTargetNetwork(feature_dim, hidden_dim, output_dim).to(device)
        predictor = PredictorNetwork(feature_dim, hidden_dim, output_dim).to(device)
    for parameter in target.parameters():
        parameter.requires_grad_(False)
    return RND(feature_dim, target, predictor, torch.optim.Adam(predictor.parameters(), lr=lr))


class RNDPredictor:
    def __init__(self, obs_dim: int, hidden_dim: int = 256, output_dim: int = 128,
                 lr: float = 1e-3, obs_start: int = 0, obs_end: int | None = None,
                 seed: int = 0, device="cpu"):
        self.obs_start, self.obs_end = obs_start, obs_end or obs_dim
        self.rnd = make_rnd(self.obs_end - self.obs_start, hidden_dim, output_dim,
                            lr, seed, device)
        self._rnd_var = torch.tensor(1.0, device=device)
        self._rnd_count = 1e-4

    def _features(self, obs):
        if isinstance(obs, dict):
            obs = obs["state"]
        return torch.as_tensor(obs)[..., self.obs_start:self.obs_end]

    def raw_novelty(self, obs):
        return self.rnd.novelty(self._features(obs))

    def novelty(self, obs):
        return self.raw_novelty(obs) / (self._rnd_var.sqrt() + 1e-8)

    def update(self, obs_batch):
        features = self._features(obs_batch)
        raw = self.rnd.novelty(features)
        self._rnd_var.mul_(0.99).add_(0.01 * raw.var(unbiased=False).clamp(min=1e-6))
        return float(self.rnd.update(features))

    def get_state(self):
        return {"predictor": copy.deepcopy(self.rnd.predictor.state_dict()),
                "optimizer": copy.deepcopy(self.rnd.optimizer.state_dict()),
                "rnd_var": self._rnd_var.clone(), "rnd_count": self._rnd_count}

    def set_state(self, state):
        self.rnd.predictor.load_state_dict(state["predictor"])
        self.rnd.optimizer.load_state_dict(state["optimizer"])
        self._rnd_var.copy_(state["rnd_var"])
        self._rnd_count = state["rnd_count"]
