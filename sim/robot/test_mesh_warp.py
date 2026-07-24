# SPDX-License-Identifier: MIT
"""Warp-path env + trainer gates (mesh_warp_env / train_mesh_warp), CPU-runnable.

1. Obs-parity anchor: the env's 50-obs vector vs a hand-built NumPy mirror using
   the same MjModel name lookups; the independent mirror is the oracle.
2. Reset validity: per-world loop residual |q_slide - poly_s(q_knee)| < 1e-6
   (and toe), i.e. resets honor loop_consistent_pose on noised knees.
3. Env smoke: nworld=8, 200 random-action steps — finite, autoreset fires,
   telemetry sane.
4. PPO smoke: tiny net, 3 updates — losses finite, entropy coef schedule moves,
   checkpoint save + resume roundtrip exact.
5. Determinism-ish: two runs, same seed -> identical first-update losses.

Runs in the canonical `.venv-warp` environment.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

torch = pytest.importorskip("torch")
pytest.importorskip("warp")
pytest.importorskip("mujoco_warp")

import mujoco  # noqa: E402

from gen_mesh_robot_mjcf import build_mesh_robot, loop_consistent_pose, loop_polycoefs, poly_eval  # noqa: E402
from mesh_warp_env import EvalTelemetry, MeshWarpEnv  # noqa: E402


def _finite_telemetry_tree(value):
    """Structured telemetry may contain mappings, sequences, and empty-bin None values."""
    if isinstance(value, dict):
        return all(_finite_telemetry_tree(item) for item in value.values())
    if isinstance(value, (list, tuple)):
        return all(_finite_telemetry_tree(item) for item in value)
    if value is None:
        return True
    return bool(np.isfinite(value).all())


@pytest.fixture(scope="module")
def small_env():
    return MeshWarpEnv(nworld=4, seed=3, device="cpu", episode_length=None)


def _addressing(m):
    aj = [int(m.actuator_trnid[a, 0]) for a in range(m.nu)]
    qa = np.array([m.jnt_qposadr[j] for j in aj])
    da = np.array([m.jnt_dofadr[j] for j in aj])
    return aj, qa, da


# ---------------------------------------------------------------------------
# 1. obs parity vs numpy mirror of mesh_commanded_env._obs (lines 127-136)
# ---------------------------------------------------------------------------
def test_obs_parity_numpy_mirror(small_env):
    env = small_env
    m = mujoco.MjModel.from_xml_string(build_mesh_robot())
    aj, qa, da = _addressing(m)
    knee_q = qa[2::3]
    legs = ("FL", "FR", "RL", "RR")
    jid = {mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_JOINT, j): j for j in range(m.njnt)}
    toe_q = np.array([m.jnt_qposadr[jid[f"{L}_toe_hinge"]] for L in legs])
    slide_q = np.array([m.jnt_qposadr[jid[f"{L}_pushrod_slide"]] for L in legs])

    rng = np.random.default_rng(7)
    n = env.nworld
    qpos = np.tile(m.qpos0, (n, 1))
    qpos[:, qa] += rng.uniform(-0.3, 0.3, (n, m.nu))
    for k in range(4):                                    # keep the loop honest
        for w in range(n):
            toe, slide = loop_consistent_pose(qpos[w, knee_q[k]])
            qpos[w, toe_q[k]], qpos[w, slide_q[k]] = toe, slide
    # distinct root pose per world: yaw + small tilt exercises the R rotation
    half = rng.uniform(-np.pi, np.pi, n) / 2
    tilt = rng.uniform(-0.1, 0.1, (n, 2))
    quat = np.stack([np.cos(half), tilt[:, 0], tilt[:, 1], np.sin(half)], axis=1)
    quat /= np.linalg.norm(quat, axis=1, keepdims=True)
    qpos[:, 3:7] = quat
    qpos[:, 0:3] = rng.uniform(-1, 1, (n, 3))
    qpos[:, 2] = 0.45 + rng.uniform(-0.05, 0.05, n)
    qvel = rng.uniform(-1.0, 1.0, (n, m.nv))
    cmd = rng.uniform(-0.6, 0.6, (n, 3))
    prev_a = rng.uniform(-1.0, 1.0, (n, m.nu))

    env.qpos.copy_(torch.as_tensor(qpos, dtype=torch.float32))
    env.qvel.copy_(torch.as_tensor(qvel, dtype=torch.float32))
    env._cmd = torch.as_tensor(cmd, dtype=torch.float32)
    env._prev_a = torch.as_tensor(prev_a, dtype=torch.float32)
    obs = env.observe().cpu().numpy()
    assert obs.shape == (n, 50)

    qp = env.qpos.cpu().numpy().astype(np.float64)        # exactly what the env holds (f32)
    qv = env.qvel.cpu().numpy().astype(np.float64)
    for w in range(n):
        # mirror of mesh_commanded_env.py lines 120-136:
        #   R from root-quat yaw (lines 121-125), then
        #   [qpos[qa], qvel[da], qpos[3:7], R@qvel[0:2], qvel[2:6], qpos[2:3],
        #    prev_action, R@cmd[:2], cmd[2:3]]           (lines 127-136)
        qw, qx, qy, qz = qp[w, 3:7]
        yaw = np.arctan2(2.0 * (qw * qz + qx * qy), 1.0 - 2.0 * (qy * qy + qz * qz))
        c, s = np.cos(yaw), np.sin(yaw)
        R = np.array([[c, s], [-s, c]])
        ref = np.concatenate([
            qp[w, qa], qv[w, da], qp[w, 3:7],
            R @ qv[w, 0:2], qv[w, 2:6], qp[w, 2:3],
            prev_a[w].astype(np.float32),
            R @ cmd[w, :2].astype(np.float32).astype(np.float64),
            cmd[w, 2:3].astype(np.float32),
        ])
        np.testing.assert_allclose(obs[w], ref, atol=1e-5)


# ---------------------------------------------------------------------------
# 2. reset validity: loop residual against the exported quartics
# ---------------------------------------------------------------------------
def test_reset_loop_residual(small_env):
    env = small_env
    obs = env.reset()
    assert obs.shape == (env.nworld, 50) and torch.isfinite(obs).all()
    cs, cp, _, _ = loop_polycoefs()
    qp = env.qpos.cpu().numpy().astype(np.float64)
    knee = qp[:, env._knee_q.cpu().numpy()]
    slide = qp[:, env._slide_q.cpu().numpy()]
    toe = qp[:, env._toe_q.cpu().numpy()]
    slide_ref = np.vectorize(lambda x: poly_eval(cs, x))(knee)
    toe_ref = np.vectorize(lambda x: poly_eval(cp, x))(knee)
    assert np.abs(slide - slide_ref).max() < 1e-6, "reset violated the slide loop coupling"
    assert np.abs(toe - toe_ref).max() < 1e-6, "reset violated the toe loop coupling"
    # noise actually landed on the actuated joints (not a frozen qpos0 copy)
    qa = env._qa.cpu().numpy()
    assert np.abs(qp[:, qa] - qp[0, qa]).max() > 1e-4


# ---------------------------------------------------------------------------
# 3. CUDA env smoke: a training-scale batch over 200 random steps
# ---------------------------------------------------------------------------
@pytest.mark.gpu
def test_env_smoke_random_actions(gpu_device):
    env = MeshWarpEnv(nworld=256, seed=0, device=gpu_device, episode_length=50)
    gen = torch.Generator(device=gpu_device).manual_seed(1)
    tel = EvalTelemetry(env.device)
    dones = torch.zeros((), device=env.device)
    finite = torch.ones((), dtype=torch.bool, device=env.device)
    reset_t_ok = torch.ones((), dtype=torch.bool, device=env.device)
    for _ in range(200):
        a = torch.rand((env.nworld, env.act_dim), generator=gen, device=env.device) * 2 - 1
        obs, rew, done, info = env.step(a, alpha=0.5, imit_anneal=0.5)
        finite &= torch.isfinite(obs).all() & torch.isfinite(rew).all()
        finite &= torch.isfinite(info["priv"]).all()
        assert info["priv"].shape == (env.nworld, env.priv_dim)
        dones += done.sum()
        reset_t_ok &= torch.where(done.bool(), env._t == 0, True).all()
        tel.add(rew, info)
    assert bool(finite), "non-finite CUDA rollout state"
    assert int(dones) > 0, "no autoreset fired in 200 steps with episode_length=50"
    assert bool(reset_t_ok), "done world kept its episode clock — autoreset did not fire per-world"
    m = tel.result()
    assert 0.0 <= m["duty"] <= 1.0
    assert m["air"] >= 0.0
    assert -1.0 <= m["diagsync"] <= 1.0
    assert _finite_telemetry_tree(m)


# ---------------------------------------------------------------------------
# 3b. imitation hook: reference_gait.json wiring + anneal gating
# ---------------------------------------------------------------------------
def test_reference_gait_hook(tmp_path):
    import json
    n_frames = 20
    m = mujoco.MjModel.from_xml_string(build_mesh_robot())
    aj, qa, _ = _addressing(m)
    names = [mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_JOINT, j) for j in aj]
    targets = np.tile(m.qpos0[qa], (n_frames, 1))
    targets[:, 0] += 0.2 * np.sin(np.linspace(0, 2 * np.pi, n_frames, endpoint=False))
    swing = np.zeros((n_frames, 4))
    swing[: n_frames // 2, [0, 3]] = 1.0             # pair A swings first half
    swing[n_frames // 2:, [1, 2]] = 1.0
    gait = {"period_s": 2.0, "n": n_frames, "joint_order": names,
            "qpos_targets": targets.tolist(), "feet_swing": swing.tolist()}
    path = tmp_path / "reference_gait.json"
    path.write_text(json.dumps(gait))

    env = MeshWarpEnv(nworld=4, seed=5, device="cpu", gait_path=path)
    assert env.gait_loaded
    a = torch.zeros((4, env.act_dim))
    _, r_on, _, info_on = env.step(a, imit_anneal=1.0)
    assert torch.isfinite(r_on).all()
    assert bool((info_on["imit"] > 0).all()), "imitation term should be positive near qpos0"
    env2 = MeshWarpEnv(nworld=4, seed=5, device="cpu", gait_path=path)
    _, r_off, _, info_off = env2.step(a, imit_anneal=0.0)
    assert bool((info_off["imit"] == 0).all())
    assert bool((r_on > r_off).all()), "annealed-off imitation must not add reward"
    # absent file -> hook disabled, env still steps
    env3 = MeshWarpEnv(nworld=2, seed=5, device="cpu", gait_path=tmp_path / "missing.json")
    assert not env3.gait_loaded
    obs, rew, _, _ = env3.step(torch.zeros((2, env3.act_dim)))
    assert torch.isfinite(obs).all() and torch.isfinite(rew).all()


# ---------------------------------------------------------------------------
# 4. PPO smoke + checkpoint roundtrip
# ---------------------------------------------------------------------------
def _tiny_args(tmp_path, tag, steps, seed=0, resume=None, device="cpu"):
    from train_mesh_warp import build_args
    argv = ["--steps", str(steps), "--envs", "8", "--horizon", "8",
            "--episode-length", "50", "--hidden", "32,32", "--seed", str(seed),
            "--device", device, "--evals", "1", "--eval-envs", "4",
            "--eval-steps", "10", "--preflight", "off",
            "--tag", str(tmp_path / tag)]
    if resume:
        argv += ["--resume", str(resume)]
    return build_args(argv)


def _flat(state):
    return {k: v for k, v in state.items()}


@pytest.mark.gpu
def test_ppo_smoke_and_ckpt_roundtrip(tmp_path, gpu_device):
    from train_mesh_warp import train
    args = _tiny_args(tmp_path, "smoke", steps=3 * 8 * 8,
                      device=gpu_device)                            # exactly 3 updates
    stats = train(args)
    ups = stats["updates"]
    assert len(ups) == 3
    for u in ups:
        assert np.isfinite([u["pi_loss"], u["v_loss"], u["entropy"]]).all()
    assert ups[0]["ent_coef"] != ups[2]["ent_coef"], "entropy schedule did not move"
    assert ups[0]["alpha"] != ups[2]["alpha"], "alpha curriculum did not move"
    ck1 = torch.load(stats["ckpt"], map_location="cpu", weights_only=True)
    assert ck1["step"] == 3 * 8 * 8
    assert ck1["contract"]["geometry"] == "mesh"
    assert ck1["runtime"] is not None

    # resume with the same --steps: loads everything, runs 0 updates, saves again
    args2 = _tiny_args(tmp_path, "smoke_resumed", steps=3 * 8 * 8,
                       resume=stats["ckpt"], device=gpu_device)
    stats2 = train(args2)
    assert len(stats2["updates"]) == 0
    ck2 = torch.load(stats2["ckpt"], map_location="cpu", weights_only=True)
    assert ck2["step"] == ck1["step"]
    for part in ("actor", "critic", "obs_norm", "priv_norm"):
        for k, v in ck1[part].items():
            assert torch.equal(v, ck2[part][k]), f"{part}.{k} changed across resume roundtrip"
    s1, s2 = ck1["opt"]["state"], ck2["opt"]["state"]
    assert list(s1.keys()) == list(s2.keys())
    for k in s1:
        for kk, v in s1[k].items():
            same = torch.equal(v, s2[k][kk]) if torch.is_tensor(v) else v == s2[k][kk]
            assert same, f"optimizer state {k}.{kk} changed across resume roundtrip"


# ---------------------------------------------------------------------------
# 5. determinism-ish: same seed -> same first-update losses
# ---------------------------------------------------------------------------
def test_determinism_same_seed_same_losses(tmp_path):
    from train_mesh_warp import train
    torch.set_num_threads(1)
    a = train(_tiny_args(tmp_path, "det_a", steps=2 * 8 * 8, seed=11))
    b = train(_tiny_args(tmp_path, "det_b", steps=2 * 8 * 8, seed=11))
    ua, ub = a["updates"][0], b["updates"][0]
    assert ua["pi_loss"] == ub["pi_loss"]
    assert ua["v_loss"] == ub["v_loss"]
    assert ua["entropy"] == ub["entropy"]
