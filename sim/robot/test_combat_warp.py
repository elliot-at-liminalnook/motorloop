# SPDX-License-Identifier: MIT
"""Deterministic contracts for the fused MuJoCo-Warp combat environment."""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

torch = pytest.importorskip("torch")
wp = pytest.importorskip("warp")
pytest.importorskip("mujoco_warp")

from combat_warp_env import CombatWarpEnv  # noqa: E402
from train_mesh_warp import build_args, train  # noqa: E402


def test_combat_model_has_twelve_leg_servos_per_robot():
    env = CombatWarpEnv(1, seed=0, device="cpu", episode_length=8)
    i = env.layer.idx
    assert len(i.Aqa) == len(i.Bqa) == 12
    assert env.act_dim == 14  # twelve selected servos plus two pneumatic valves
    assert env.obs_dim == 44 and env.priv_dim == 32


def test_two_policy_actions_scatter_entirely_on_device():
    env = CombatWarpEnv(2, seed=1, device="cpu", episode_length=8)
    a = torch.linspace(-1.5, 1.5, 2 * env.act_dim).reshape(2, env.act_dim)
    b = -a
    env.set_opponent(lambda _: b)
    env.step(a)
    ctrl = wp.to_torch(env.layer.d.ctrl)
    ia = torch.as_tensor(env.layer.idx.actA)
    ib = torch.as_tensor(env.layer.idx.actB)
    torch.testing.assert_close(ctrl[:, ia], a.clamp(-1, 1), rtol=0, atol=0)
    torch.testing.assert_close(ctrl[:, ib], b.clamp(-1, 1), rtol=0, atol=0)


def test_same_seed_action_tape_is_exactly_repeatable():
    first = CombatWarpEnv(2, seed=22, device="cpu", episode_length=20)
    second = CombatWarpEnv(2, seed=22, device="cpu", episode_length=20)
    gen = torch.Generator().manual_seed(7)
    for _ in range(5):
        action = torch.rand((2, first.act_dim), generator=gen) * 0.4 - 0.2
        out_a = first.step(action)
        out_b = second.step(action)
        torch.testing.assert_close(out_a[0], out_b[0], rtol=0, atol=0)
        torch.testing.assert_close(out_a[1], out_b[1], rtol=0, atol=0)
        torch.testing.assert_close(out_a[2], out_b[2], rtol=0, atol=0)


@pytest.mark.gpu
def test_randomized_combat_stays_below_constraint_pool_capacity(gpu_device):
    env = CombatWarpEnv(256, seed=19, device=gpu_device, episode_length=None)
    generator = torch.Generator(device=gpu_device).manual_seed(91)
    peak = None
    for _ in range(100):
        action = torch.rand(
            (env.nworld, env.act_dim), generator=generator, device=env.device
        ) * 2.0 - 1.0
        _, _, _, info = env.step(action)
        current = info["constraint_rows"].max()
        peak = current if peak is None else torch.maximum(peak, current)
    assert peak is not None
    peak_value = int(peak)
    assert peak_value < env.layer.njmax, (
        f"combat constraint pool exhausted: peak={peak_value}, capacity={env.layer.njmax}")
    assert env.layer.njmax >= 192 and env.layer.nconmax >= 64


def test_time_limit_autoreset_returns_fresh_state():
    env = CombatWarpEnv(2, seed=3, device="cpu", episode_length=1)
    initial = env.qpos.clone()
    obs, reward, done, info = env.step(torch.zeros((2, env.act_dim)))
    assert bool(done.all()) and bool((info["truncated"] == 1).all())
    torch.testing.assert_close(env.qpos, initial, rtol=0, atol=0)
    assert torch.isfinite(obs).all() and torch.isfinite(reward).all()
    assert torch.isfinite(info["priv"]).all()


@pytest.mark.gpu
def test_combat_tiny_ppo_cycle(tmp_path, gpu_device):
    args = build_args([
        "--geometry", "combat", "--device", gpu_device, "--steps", "4",
        "--envs", "2", "--horizon", "2", "--episode-length", "8",
        "--hidden", "8,8", "--epochs", "1", "--minibatches", "1",
        "--evals", "1", "--eval-envs", "1", "--eval-steps", "2",
        "--preflight", "off", "--tag", str(tmp_path / "combat"),
    ])
    result = train(args)
    assert result["updates"] and result["evals"]
    assert Path(result["ckpt"]).exists()
