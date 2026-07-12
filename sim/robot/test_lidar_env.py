# SPDX-License-Identifier: MIT
"""Fused Warp lidar integration contracts."""

import sys
from pathlib import Path

import pytest

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
torch = pytest.importorskip("torch")
pytest.importorskip("mujoco_warp")

from combat_warp_env import CombatWarpEnv


def test_lidar_expands_observation_and_stays_finite():
    plain = CombatWarpEnv(1, seed=0, device="cpu", lidar=False)
    lidar = CombatWarpEnv(1, seed=0, device="cpu", lidar=True)
    assert lidar.obs_dim > plain.obs_dim
    obs, reward, _, _ = lidar.step(torch.zeros((1, lidar.act_dim)))
    assert torch.isfinite(obs).all() and torch.isfinite(reward).all()
    scan = obs[:, -lidar.layer.nray:]
    assert bool(((scan >= 0.0) & (scan <= 1.0)).all())
