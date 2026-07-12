# SPDX-License-Identifier: MIT
"""Transition-info contracts shared by all MuJoCo-Warp environments."""

import sys
from pathlib import Path

import pytest

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
torch = pytest.importorskip("torch")
pytest.importorskip("mujoco_warp")

from combat_warp_env import CombatWarpEnv
from walker_warp_env import WalkerWarpEnv

REQUIRED = {"truncated", "terminal_obs", "terminal_priv", "priv", "track",
            "verr", "align", "speed", "progress", "up", "height"}


@pytest.mark.parametrize("factory", (WalkerWarpEnv, CombatWarpEnv))
def test_transition_info_has_truncation_and_terminal_snapshots(factory):
    env = factory(2, seed=1, device="cpu", episode_length=1)
    _, _, done, info = env.step(torch.zeros((2, env.act_dim)))
    assert REQUIRED <= info.keys()
    assert bool(done.all()) and bool((info["truncated"] == 1).all())
    assert info["terminal_obs"].shape == (2, env.obs_dim)
    assert info["terminal_priv"].shape == (2, env.priv_dim)
