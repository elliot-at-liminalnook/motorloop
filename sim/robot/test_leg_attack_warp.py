# SPDX-License-Identifier: MIT
"""Contracts for the runtime-selectable leg attack/kick training rung."""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
torch = pytest.importorskip("torch")
pytest.importorskip("mujoco_warp")

from leg_attack_warp_env import (  # noqa: E402
    LegAttackConfig,
    LegAttackWarpEnv,
    leg_attack_reward,
    reduce_leg_contacts,
)


def test_contact_reducer_credits_each_foot_and_floor_separately():
    geom = torch.tensor(((1, 3), (2, 3), (0, 1), (2, 0)), dtype=torch.int32)
    dist = torch.tensor((-0.01, -0.02, -0.003, -0.004))
    world = torch.zeros(4, dtype=torch.int32)
    nacon = torch.tensor((4,), dtype=torch.int32)
    geom_to_leg = torch.tensor((-1, 0, 1, -1), dtype=torch.long)
    target = torch.tensor((False, False, False, True))
    hit, support = reduce_leg_contacts(
        geom, dist, world, nacon, geom_to_leg, target, 0, 1)
    torch.testing.assert_close(hit, torch.tensor(((0.01, 0.02, 0.0, 0.0),)))
    torch.testing.assert_close(support, torch.tensor(((1.0, 1.0, 0.0, 0.0),)))


def test_reward_prefers_selected_hit_and_three_leg_support():
    cfg = LegAttackConfig()
    support = torch.tensor(((0.0, 1.0, 1.0, 1.0),) * 2)
    extension = torch.tensor(((0.02, 0.0, 0.0, 0.0),) * 2)
    selected = torch.zeros(2, dtype=torch.long)
    active = torch.ones(2)
    phase = torch.full((2,), 0.4)
    action = torch.zeros((2, 14))
    selected_hit = torch.tensor(((0.02, 0.0, 0.0, 0.0),
                                 (0.0, 0.02, 0.0, 0.0)))
    reward, metrics = leg_attack_reward(
        selected_hit, support, extension, selected, active, phase, action, 0.02, cfg)
    assert reward[0] > reward[1]
    assert metrics["attack_selected_hit"][0] == 1
    assert metrics["attack_wrong_hit"][1] == 1
    assert metrics["attack_support"].eq(1).all()


def test_attack_credit_is_invariant_to_prescribed_phase():
    cfg = LegAttackConfig()
    hit = torch.tensor(((0.02, 0.0, 0.0, 0.0),) * 2)
    support = torch.tensor(((0.0, 1.0, 1.0, 1.0),) * 2)
    extension = torch.tensor(((0.01, 0.0, 0.0, 0.0),) * 2)
    selected = torch.zeros(2, dtype=torch.long)
    active = torch.ones(2)
    action = torch.zeros((2, 14))
    reward, metrics = leg_attack_reward(
        hit, support, extension, selected, active,
        torch.tensor((0.05, 0.80)), action, 0.02, cfg)
    torch.testing.assert_close(reward[0], reward[1])
    torch.testing.assert_close(
        metrics["attack_kick_speed"][0], metrics["attack_kick_speed"][1])


def test_controller_switch_changes_command_not_physical_state():
    env = LegAttackWarpEnv(4, seed=7, device="cpu", episode_length=8)
    env.set_attack_command(torch.arange(4), True)
    first = env.observe().clone()
    torch.testing.assert_close(first[:, -6:-2], torch.eye(4))
    physical = first[:, :44].clone()
    env.set_attack_command("RR", True)
    second = env.observe()
    torch.testing.assert_close(second[:, :44], physical)
    torch.testing.assert_close(second[:, -6:-2],
                               torch.tensor((0.0, 0.0, 0.0, 1.0)).expand(4, -1))


def test_leg_attack_step_contract_and_autoreset():
    env = LegAttackWarpEnv(2, seed=9, device="cpu", episode_length=1)
    obs, reward, done, info = env.step(torch.zeros((2, env.act_dim)))
    assert obs.shape == info["terminal_obs"].shape == (2, 51)
    assert done.eq(1).all()
    assert torch.isfinite(obs).all() and torch.isfinite(reward).all()
    for key in ("attack_selected_hit", "attack_wrong_hit", "attack_support",
                "attack_kick_speed", "attack_task_reward"):
        assert info[key].shape == (2,) and torch.isfinite(info[key]).all()
