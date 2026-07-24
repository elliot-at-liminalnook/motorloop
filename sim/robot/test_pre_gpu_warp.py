# SPDX-License-Identifier: MIT
"""Deterministic launch gates for the active Warp walker and PPO trainer.

These tests cover the gaps that a finite smoke rollout cannot: dynamic tracking
of the selected reference, MuJoCo/Warp trajectory agreement, coherent autoresets,
hand-checked GAE semantics, fixed evaluation scenarios, checkpoint compatibility,
and restoration of every state that affects the next transition.
"""

from __future__ import annotations

import json
import math
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

torch = pytest.importorskip("torch")
pytest.importorskip("warp")
pytest.importorskip("mujoco_warp")

import mujoco  # noqa: E402
import mujoco_warp as mjwp  # noqa: E402
import warp as wp  # noqa: E402

import walker_warp_env as W  # noqa: E402
from walker_improved import DEFAULTS, build_walker  # noqa: E402
from train_mesh_warp import (  # noqa: E402
    Actor,
    Critic,
    RunningNorm,
    capture_runtime_state,
    checkpoint_contract,
    compute_gae,
    evaluate,
    load_ckpt,
    restore_runtime_state,
    save_ckpt,
    validate_training_args,
)


def _disable_terminations(monkeypatch):
    monkeypatch.setattr(W, "CAT_ON", False)


def _walker_fingerprint():
    env = W.WalkerWarpEnv(1, seed=101, device="cpu", episode_length=None)
    reward_sum = 0.0
    joint_phase = torch.arange(env.act_dim, dtype=torch.float32) * 0.31
    for step in range(80):
        env._cmd[:] = torch.tensor([[0.15, 0.0, 0.0]])
        env._timer.zero_()
        action = (0.25 * torch.sin(0.17 * step + joint_phase)).unsqueeze(0)
        _, reward, done, _ = env.step(action, alpha=1.0, imit_anneal=0.0)
        assert not bool(done.any())
        reward_sum += float(reward[0])
    qpos = env.qpos[0].double().cpu().numpy()
    return {
        "final_z": round(float(qpos[2]), 6),
        "final_x": round(float(qpos[0]), 6),
        "qpos_abs_sum": round(float(np.abs(qpos).sum()), 6),
        "qpos_sum": round(float(qpos.sum()), 6),
        "reward_sum": round(reward_sum, 6),
    }


def test_walker_warp_golden_trajectory(monkeypatch):
    """Pin the complete Warp physics/action/reward path to a checked-in fingerprint."""
    _disable_terminations(monkeypatch)
    got = _walker_fingerprint()
    expected = json.loads((HERE / "golden_walker_warp.json").read_text())
    assert got.keys() == expected.keys()
    for key in got:
        assert np.isclose(got[key], expected[key], rtol=1e-4, atol=1e-6), (
            f"Warp golden drift on {key}: expected {expected[key]}, got {got[key]}")


@pytest.mark.gpu
def test_reference_gait_tracks_dynamically_at_full_servo_derating(monkeypatch, gpu_device):
    """Track two gait cycles as a batched CUDA proof before imitation RL."""
    _disable_terminations(monkeypatch)
    env = W.WalkerWarpEnv(64, seed=0, device=gpu_device, episode_length=None)
    env._reset_noise.zero_()
    env.reset()
    raw = json.loads((HERE / "reference_gait_walker.json").read_text())
    targets = torch.as_tensor(raw["qpos_targets"], dtype=torch.float32, device=env.device)
    period, frames, dt = float(raw["period_s"]), len(targets), env._dt

    nsteps = round(2.0 * period / dt)
    done_seen = torch.zeros(env.nworld, dtype=torch.bool, device=env.device)
    z_min = torch.full((env.nworld,), float("inf"), device=env.device)
    up_min = torch.full((env.nworld,), float("inf"), device=env.device)
    foot_max = torch.full((env.nworld, 4), -float("inf"), device=env.device)
    error_sum = torch.zeros((), device=env.device)
    error_max = torch.zeros((), device=env.device)
    saturated_sum = torch.zeros((), device=env.device)
    x_start = env.qpos[:, 0].clone()
    for step in range(nsteps):
        phase = ((step * dt) % period) / period
        x = phase * frames
        i0, frac = int(math.floor(x)) % frames, x - math.floor(x)
        ref = (targets[i0] * (1.0 - frac) + targets[(i0 + 1) % frames] * frac)
        ref = ref.unsqueeze(0).expand(env.nworld, -1)
        desired_used = (ref - env._stand) / env._authority
        command = (desired_used - W.ACT_LP * env._prev_a) / (1.0 - W.ACT_LP)
        saturated_sum += (command.abs() > 1.0).sum()
        env._cmd.zero_()
        env._timer.zero_()
        _, _, done, _ = env.step(command.clamp(-1.0, 1.0), alpha=1.0, imit_anneal=0.0)
        done_seen |= done.bool()
        z_min = torch.minimum(z_min, env.xpos[:, env._torso, 2])
        up = 1.0 - 2.0 * (env.qpos[:, 4] ** 2 + env.qpos[:, 5] ** 2)
        up_min = torch.minimum(up_min, up)
        foot_max = torch.maximum(foot_max, env.geom_xpos[:, env._feet, 2])
        error = torch.sqrt(((env.qpos[:, env._qa] - ref) ** 2).mean(dim=1))
        error_sum += error.sum()
        error_max = torch.maximum(error_max, error.max())

    progress = env.qpos[:, 0] - x_start
    per_foot_clearance = foot_max.amin(dim=0)
    mean_error = error_sum / (nsteps * env.nworld)
    saturation_rate = saturated_sum / (nsteps * env.nworld * env.act_dim)
    assert not bool(done_seen.any()), "reference gait fell or hit a hard constraint"
    assert float(z_min.min()) > 0.30
    assert float(up_min.min()) > 0.90
    assert float(progress.min()) > 0.02, "two crawl cycles made no meaningful forward progress"
    assert bool((per_foot_clearance > 0.03).all()), (
        f"not every foot cleared 30 mm: {per_foot_clearance.cpu().tolist()}")
    assert float(mean_error) < 0.03 and float(error_max) < 0.08
    assert float(saturation_rate) < 0.15, "reference relies excessively on clipped commands"


def _cpu_pd_step(m, d, target, qadr, dadr, kp, wfree, gear, frame_skip):
    for _ in range(frame_skip):
        tau = kp * (target - d.qpos[qadr])
        qd = d.qvel[dadr]
        limit = np.where(tau * qd > 0.0,
                         np.clip(1.0 - np.abs(qd) / wfree, 0.0, 1.0), 1.0)
        d.ctrl[:] = np.clip(tau / gear, -limit, limit)
        mujoco.mj_step(m, d)


@pytest.mark.parametrize("airborne,steps", [(True, 20), (False, 10)])
def test_active_walker_mujoco_warp_trajectory_parity(monkeypatch, airborne, steps):
    """The exact training body follows the CPU reference under one action tape."""
    _disable_terminations(monkeypatch)
    env = W.WalkerWarpEnv(1, seed=4, device="cpu", episode_length=None)
    env._reset_noise.zero_()
    env.reset()
    if airborne:
        env.qpos[:, 2] = 1.0
        env.qvel.zero_()
        with wp.ScopedDevice(env._wp_device):
            mjwp.forward(env._wm, env._wd)

    m = mujoco.MjModel.from_xml_string(build_walker(DEFAULTS, floor=True))
    d = mujoco.MjData(m)
    d.qpos[:] = env.qpos[0].double().cpu().numpy()
    d.qvel[:] = env.qvel[0].double().cpu().numpy()
    mujoco.mj_forward(m, d)
    qadr, dadr = env._qa.cpu().numpy(), env._da.cpu().numpy()
    kp = np.asarray(list(W.WALKER_KP) * 4)
    gear = m.actuator_gear[:m.nu, 0].copy()
    wfree = np.asarray(W._DESIGN.wfrees())
    prev = np.zeros(m.nu)
    tape = np.random.default_rng(12).uniform(-0.25, 0.25, (steps, m.nu))

    for action in tape:
        used = W.ACT_LP * prev + (1.0 - W.ACT_LP) * action
        target = np.clip(env._stand.cpu().numpy() + used * env._authority.cpu().numpy(),
                         env._jr_lo.cpu().numpy(), env._jr_hi.cpu().numpy())
        _cpu_pd_step(m, d, target, qadr, dadr, kp, wfree, gear, env._fs)
        env._cmd.zero_()
        env._timer.zero_()
        env.step(torch.as_tensor(action, dtype=torch.float32).unsqueeze(0),
                 alpha=1.0, imit_anneal=0.0)
        prev = used

    warp_q = env.qpos[0].double().cpu().numpy()
    mean_err = float(np.mean(np.abs(warp_q - d.qpos)))
    max_err = float(np.max(np.abs(warp_q - d.qpos)))
    if airborne:
        assert mean_err < 5e-3 and max_err < 5e-2
    else:
        assert mean_err < 2e-2 and max_err < 0.15


def test_autoreset_returns_fresh_kinematics_and_privileged_state(monkeypatch):
    _disable_terminations(monkeypatch)
    env = W.WalkerWarpEnv(2, seed=9, device="cpu", episode_length=1)
    _, _, done, info = env.step(torch.zeros((2, env.act_dim)), alpha=1.0,
                                imit_anneal=0.0)
    assert bool((done == 1).all()) and bool((info["truncated"] == 1).all())

    m = mujoco.MjModel.from_xml_string(build_walker(DEFAULTS, floor=True))
    for world in range(2):
        d = mujoco.MjData(m)
        d.qpos[:] = env.qpos[world].double().cpu().numpy()
        d.qvel[:] = env.qvel[world].double().cpu().numpy()
        mujoco.mj_forward(m, d)
        expected_feet = d.geom_xpos[env._feet.cpu().numpy(), 2]
        np.testing.assert_allclose(
            env.geom_xpos[world, env._feet, 2].cpu().numpy(), expected_feet,
            atol=2e-4, rtol=2e-4)
        np.testing.assert_allclose(
            info["priv"][world, 4:8].cpu().numpy(), expected_feet,
            atol=2e-4, rtol=2e-4)


def test_command_reward_is_rotation_equivariant(monkeypatch):
    """Rotating body, velocity, and command together must not change reward."""
    _disable_terminations(monkeypatch)
    env = W.WalkerWarpEnv(2, seed=3, device="cpu", episode_length=None)
    env._reset_noise.zero_()
    env.reset()
    env.qpos[:, 2] = 1.0                         # airborne: isolate frame semantics
    half = math.pi / 4.0                         # world 1 yaw = +90 degrees
    env.qpos[0, 3:7] = torch.tensor([1.0, 0.0, 0.0, 0.0])
    env.qpos[1, 3:7] = torch.tensor([math.cos(half), 0.0, 0.0, math.sin(half)])
    env.qvel.zero_()
    env.qvel[0, 0] = 0.10
    env.qvel[1, 1] = 0.10
    env._cmd[0] = torch.tensor([0.20, 0.0, 0.0])
    env._cmd[1] = torch.tensor([0.0, 0.20, 0.0])
    with wp.ScopedDevice(env._wp_device):
        mjwp.forward(env._wm, env._wd)
    obs = env.observe().cpu().numpy()
    # World orientation is intentionally present in quaternion slots 24:28;
    # every other zero-angular-velocity feature should agree in the body frame.
    np.testing.assert_allclose(np.delete(obs[0], np.s_[24:28]),
                               np.delete(obs[1], np.s_[24:28]), atol=2e-5)

    env._timer.zero_()
    _, reward, done, info = env.step(torch.zeros((2, env.act_dim)), alpha=1.0,
                                     imit_anneal=0.0)
    assert not bool(done.any())
    torch.testing.assert_close(reward[0], reward[1], rtol=2e-5, atol=2e-5)
    for key in ("track", "verr", "align", "speed", "progress", "up", "height"):
        torch.testing.assert_close(info[key][0], info[key][1], rtol=2e-5, atol=2e-5)


def test_gae_matches_hand_calculation_and_stops_at_done():
    rewards = torch.tensor([[1.0], [2.0], [3.0]])
    dones = torch.tensor([[0.0], [1.0], [0.0]])
    values = torch.tensor([[0.5], [0.25], [1.0]])
    adv, ret = compute_gae(rewards, dones, values, torch.tensor([0.75]),
                           gamma=0.9, lam=0.8)
    torch.testing.assert_close(adv[:, 0], torch.tensor([1.985, 1.75, 2.675]))
    torch.testing.assert_close(ret[:, 0], torch.tensor([2.485, 2.0, 3.675]))

    # A time-limit transition carries gamma*V_terminal in its reward, then done
    # stops the recursive tail. A true terminal transition does not bootstrap.
    terminal_adv, _ = compute_gae(torch.tensor([[1.0]]), torch.ones((1, 1)),
                                  torch.tensor([[0.4]]), torch.zeros(1), gamma=0.9)
    trunc_adv, _ = compute_gae(torch.tensor([[1.0 + 0.9 * 2.0]]), torch.ones((1, 1)),
                               torch.tensor([[0.4]]), torch.zeros(1), gamma=0.9)
    torch.testing.assert_close(terminal_adv, torch.tensor([[0.6]]))
    torch.testing.assert_close(trunc_adv, torch.tensor([[2.4]]))


def test_batch_geometry_rejects_discarded_samples():
    args = SimpleNamespace(envs=3, horizon=5, minibatches=4, epochs=1, steps=100,
                           preflight="off")
    env = SimpleNamespace(_dt=0.02, obs_dim=50)
    with pytest.raises(ValueError, match="silently discard"):
        validate_training_args(args, env, (32, 32))


def test_fixed_evaluation_bank_is_repeatable(monkeypatch):
    _disable_terminations(monkeypatch)
    env = W.WalkerWarpEnv(2, seed=44, device="cpu", episode_length=20)
    torch.manual_seed(2)
    actor = Actor(env.obs_dim, env.act_dim, (16, 16))
    norm = RunningNorm(env.obs_dim)
    first = evaluate(env, actor, norm, 1.0, 0.0, 12, reset_seed=1234)
    second = evaluate(env, actor, norm, 1.0, 0.0, 12, reset_seed=1234)
    assert first == second


def test_runtime_restore_reproduces_rng_and_next_transition(monkeypatch):
    _disable_terminations(monkeypatch)
    env = W.WalkerWarpEnv(2, seed=5, device="cpu", episode_length=30)
    actions = torch.linspace(-0.2, 0.2, 24).reshape(2, 12)
    for _ in range(3):
        env.step(actions, alpha=1.0, imit_anneal=0.0)
    torch.manual_seed(77)
    state = capture_runtime_state(env)

    expected_random = torch.randn(8)
    expected = env.step(-actions, alpha=1.0, imit_anneal=0.0)
    expected_qpos = env.qpos.clone()

    restored = W.WalkerWarpEnv(2, seed=999, device="cpu", episode_length=30)
    restore_runtime_state(restored, state)
    torch.testing.assert_close(torch.randn(8), expected_random, rtol=0, atol=0)
    got = restored.step(-actions, alpha=1.0, imit_anneal=0.0)
    torch.testing.assert_close(restored.qpos, expected_qpos, rtol=0, atol=2e-6)
    torch.testing.assert_close(got[1], expected[1], rtol=0, atol=2e-5)
    torch.testing.assert_close(got[2], expected[2], rtol=0, atol=0)


def test_checkpoint_contract_rejects_model_drift(tmp_path, monkeypatch):
    _disable_terminations(monkeypatch)
    env = W.WalkerWarpEnv(1, seed=0, device="cpu", episode_length=10)
    args = SimpleNamespace(geometry="walker", marker="test")
    actor = Actor(env.obs_dim, env.act_dim, (8,))
    critic = Critic(env.obs_dim + env.priv_dim, (8,))
    on, pn = RunningNorm(env.obs_dim), RunningNorm(env.priv_dim)
    opt = torch.optim.Adam(list(actor.parameters()) + list(critic.parameters()))
    path = tmp_path / "policy.pt"
    contract = checkpoint_contract(env, args)
    save_ckpt(path, 10, actor, critic, on, pn, opt, args,
              contract=contract, runtime=capture_runtime_state(env))
    bad = {**contract, "model_hash": "different-body"}
    with pytest.raises(ValueError, match="incompatible"):
        load_ckpt(path, actor, critic, on, pn, opt, "cpu", expected_contract=bad)


def test_reward_only_migration_preserves_actor_but_resets_critic(tmp_path, monkeypatch):
    _disable_terminations(monkeypatch)
    env = W.WalkerWarpEnv(1, seed=0, device="cpu", episode_length=10)
    args = SimpleNamespace(geometry="walker", marker="test", steps=20)
    actor = Actor(env.obs_dim, env.act_dim, (8,))
    critic = Critic(env.obs_dim + env.priv_dim, (8,))
    on, pn = RunningNorm(env.obs_dim), RunningNorm(env.priv_dim)
    opt = torch.optim.Adam(
        list(actor.parameters()) + list(critic.parameters()), lr=1.23e-5)
    path = tmp_path / "policy.pt"
    contract = checkpoint_contract(env, args)
    save_ckpt(path, 10, actor, critic, on, pn, opt, args,
              contract=contract, runtime={
                  "canary": True,
                  "env": {"tensors": {
                      "_constraint_duals": torch.tensor((2.0,)),
                      "_competence_duals": torch.tensor((3.0,)),
                      "_competence_error_square": torch.tensor((4.0,)),
                  }},
              })
    saved_actor = {name: value.detach().clone()
                   for name, value in actor.state_dict().items()}
    with torch.no_grad():
        for value in actor.parameters():
            value.add_(1.0)
        for value in critic.parameters():
            value.add_(2.0)
    migrated_critic = {name: value.detach().clone()
                       for name, value in critic.state_dict().items()}
    opt.param_groups[0]["lr"] = 0.5
    new_contract = {**contract, "reward_semantics": "reward:v2"}

    step, runtime = load_ckpt(
        path, actor, critic, on, pn, opt, "cpu",
        expected_contract=new_contract, allow_reward_migration=True)

    assert step == 10 and runtime["canary"] is True
    assert runtime["schedule_progress"] == 0.5
    assert "_constraint_duals" in runtime["env"]["tensors"]
    assert "_competence_duals" not in runtime["env"]["tensors"]
    assert "_competence_error_square" not in runtime["env"]["tensors"]
    for name, value in actor.state_dict().items():
        torch.testing.assert_close(value, saved_actor[name])
    for name, value in critic.state_dict().items():
        torch.testing.assert_close(value, migrated_critic[name])
    assert opt.state == {}
    assert opt.param_groups[0]["lr"] == pytest.approx(1.23e-5)


def test_default_training_budget_has_enough_updates():
    from train_mesh_warp import build_args
    args = build_args(["--geometry", "walker", "--tag", "/tmp/pre-gpu-default"])
    updates = math.ceil(args.steps / (args.envs * args.horizon))
    assert updates >= 200, f"default long run has only {updates} PPO updates"
