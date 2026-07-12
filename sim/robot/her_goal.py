# SPDX-License-Identifier: MIT
"""Torch-native future-goal relabeling for MuJoCo-Warp rollouts."""

from __future__ import annotations

from collections import deque
import random

import torch

GOAL_DIM = 4
GOAL_WEIGHTS = torch.tensor((1.0, 0.5, 0.3, 1.0))


def goal_reward(achieved, goal, sigma: float = 0.15, weights=GOAL_WEIGHTS):
    achieved = torch.as_tensor(achieved)
    goal = torch.as_tensor(goal, device=achieved.device, dtype=achieved.dtype)
    weights = torch.as_tensor(weights, device=achieved.device, dtype=achieved.dtype)
    distance_sq = (((achieved - goal) * weights) ** 2).sum(dim=-1)
    return torch.exp(-distance_sq / (2.0 * sigma ** 2))


def _future_indices(generator, n_steps: int, n_envs: int, fraction: float, device):
    mask = torch.rand((n_steps, n_envs), generator=generator, device=device) < fraction
    time = torch.arange(n_steps, device=device)[:, None]
    span = n_steps - time
    offset = torch.floor(torch.rand((n_steps, n_envs), generator=generator,
                                    device=device) * span).long()
    return mask, (time + offset).clamp(max=n_steps - 1)


def relabel_goal_arrays(obs, next_obs, reward, achieved, goal, generator,
                        her_coeff: float, sigma: float = 0.15, fraction: float = 0.5,
                        her_dim: int = GOAL_DIM, weights=GOAL_WEIGHTS):
    """Relabel `(time, world, feature)` tensors with future achieved goals."""
    n_steps, n_envs = reward.shape
    mask, future = _future_indices(generator, n_steps, n_envs, fraction, reward.device)
    world = torch.arange(n_envs, device=reward.device).expand(n_steps, -1)
    future_goal = achieved[future, world]
    new_goal = torch.where(mask[..., None], future_goal, goal)
    delta = (goal_reward(achieved, new_goal, sigma, weights)
             - goal_reward(achieved, goal, sigma, weights)) * mask
    reward2 = reward + her_coeff * delta

    def replace_goal(value):
        return torch.cat((value[..., :-her_dim],
                          torch.where(mask[..., None], new_goal,
                                      value[..., -her_dim:])), dim=-1)

    return replace_goal(obs), replace_goal(next_obs), reward2, {
        "mask": mask, "new_goal": new_goal, "future": future,
    }


class HERGoal:
    DIM = GOAL_DIM
    goal_reward = staticmethod(goal_reward)


def sample_goal(generator: torch.Generator, device="cpu"):
    values = torch.rand(4, generator=generator, device=device)
    return torch.stack((0.1 + 0.9 * values[0],
                        -torch.pi + 2.0 * torch.pi * values[1],
                        -1.0 + 2.0 * values[2], 0.05 + 0.45 * values[3]))


class HERReplayBuffer:
    def __init__(self, capacity: int = 100000, her_fraction: float = 0.4):
        self.buffer = deque(maxlen=capacity)
        self.her_fraction = her_fraction
        self._episode = []

    def add(self, obs, action, reward, next_obs, done, goal, achieved):
        self._episode.append(dict(obs=obs, action=action, reward=reward,
                                  next_obs=next_obs, done=done, goal=goal, achieved=achieved))
        if done:
            self._finish_episode()

    def _finish_episode(self):
        self.buffer.extend(self._episode)
        count = int(len(self._episode) * self.her_fraction)
        for index in random.sample(range(len(self._episode)), min(count, len(self._episode))):
            future = random.randint(index, len(self._episode) - 1)
            transition = self._episode[index]
            goal = self._episode[future]["achieved"]
            self.buffer.append({**transition, "goal": goal,
                                "reward": float(goal_reward(transition["achieved"], goal))})
        self._episode.clear()

    def __len__(self):
        return len(self.buffer)
