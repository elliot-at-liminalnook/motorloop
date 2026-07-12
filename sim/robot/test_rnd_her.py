# SPDX-License-Identifier: MIT
"""Torch RND and HER deterministic contracts."""

import sys
from pathlib import Path

import torch

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

from her_goal import goal_reward, relabel_goal_arrays
from rnd_curiosity import RNDPredictor


def test_rnd_novelty_decreases_on_repeated_state():
    rnd = RNDPredictor(4, hidden_dim=16, output_dim=8, seed=2)
    batch = torch.ones((64, 4))
    before = float(rnd.raw_novelty(batch).mean())
    for _ in range(80):
        rnd.update(batch)
    after = float(rnd.raw_novelty(batch).mean())
    assert after < 0.5 * before


def test_her_future_indices_and_goal_columns():
    generator = torch.Generator().manual_seed(9)
    obs = torch.zeros((4, 2, 7))
    achieved = torch.rand((4, 2, 4), generator=generator)
    goal = torch.zeros_like(achieved)
    out, _, reward, info = relabel_goal_arrays(
        obs, obs.clone(), torch.zeros((4, 2)), achieved, goal,
        generator, her_coeff=1.0, fraction=1.0)
    time = torch.arange(4)[:, None]
    assert bool((info["future"] >= time).all())
    torch.testing.assert_close(out[..., -4:], info["new_goal"])
    assert torch.isfinite(reward).all()
    assert float(goal_reward(achieved, achieved).min()) == 1.0
