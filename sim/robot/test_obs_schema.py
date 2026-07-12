# SPDX-License-Identifier: MIT
"""Observation schema contracts for locomotion, co-design, and combat."""

import sys
from pathlib import Path

import pytest

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
torch = pytest.importorskip("torch")
pytest.importorskip("mujoco_warp")

from codesign_warp_env import CodesignWarpEnv
from combat_warp_env import CombatWarpEnv
from walker_warp_env import WalkerWarpEnv


@pytest.mark.parametrize("env", (
    lambda: WalkerWarpEnv(2, device="cpu"),
    lambda: CodesignWarpEnv(2, device="cpu", design=(0.2, 0.4, 0.6)),
    lambda: CombatWarpEnv(2, device="cpu"),
))
def test_observation_and_privileged_shapes_are_stable(env):
    instance = env()
    assert instance.observe().shape == (2, instance.obs_dim)
    assert instance.privileged().shape == (2, instance.priv_dim)
    assert torch.isfinite(instance.observe()).all()
    assert torch.isfinite(instance.privileged()).all()


def test_codesign_vector_is_in_actor_observation():
    env = CodesignWarpEnv(2, device="cpu", design=(0.2, 0.4, 0.6))
    torch.testing.assert_close(env.observe()[:, -3:],
                               torch.tensor((0.2, 0.4, 0.6)).expand(2, -1))
