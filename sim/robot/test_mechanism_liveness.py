# SPDX-License-Identifier: MIT
"""End-to-end liveness checks for the active MuJoCo-Warp mechanisms."""

import sys
from pathlib import Path

import pytest

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
torch = pytest.importorskip("torch")
pytest.importorskip("mujoco_warp")

from commanded_env import _build
from combat_warp_env import CombatWarpEnv


def test_commanded_adapter_is_real_warp_environment():
    env = _build()(nworld=2, seed=4, device="cpu", episode_length=4)
    assert env.backend == "mujoco_warp"
    before = env.qpos.clone()
    env.step(torch.full((2, env.act_dim), 0.2))
    assert not torch.equal(before, env.qpos)


def test_combat_b_policy_changes_b_actuators():
    env = CombatWarpEnv(1, seed=3, device="cpu")
    env.set_opponent(lambda obs: torch.full((len(obs), env.act_dim), 0.25))
    env.step(torch.zeros((1, env.act_dim)))
    assert bool((env.actions_b == 0.25).all())
