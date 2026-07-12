# SPDX-License-Identifier: MIT
"""Frozen-opponent plumbing for symmetric Warp combat."""

import sys
from pathlib import Path

import pytest

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
torch = pytest.importorskip("torch")
pytest.importorskip("mujoco_warp")

from combat_warp_env import CombatWarpEnv


def test_frozen_opponent_consumes_mirrored_observation():
    env = CombatWarpEnv(3, seed=4, device="cpu")
    seen = []

    def opponent(obs):
        seen.append(obs.clone())
        return torch.zeros((len(obs), env.act_dim))

    env.set_opponent(opponent)
    env.step(torch.zeros((3, env.act_dim)))
    assert seen and seen[0].shape == (3, env.obs_dim)
    assert torch.isfinite(seen[0]).all()
