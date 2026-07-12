# SPDX-License-Identifier: MIT
"""Warp-path walker env + trainer gates (walker_warp_env / train_mesh_warp
--geometry walker), CPU-runnable — the WalkerWarpEnv analog of test_mesh_warp.py.

1. Obs/priv shape + layout: the 50-obs vector matches a hand-built numpy mirror
   of the mesh_commanded_env._obs layout (lines 127-136) on the walker's 12
   actuated joints, atol 1e-5; privileged() is (n, 34).
2. Reset validity: torso starts at stance_h and drop-settles to an UPRIGHT
   stance (torso z > 0.30, up_z > 0.9) with the feet on the floor.
3. Env smoke: nworld=8, 200 random-action steps — finite, autoreset fires,
   telemetry sane.
4. Imitation hook: reference_gait_walker.json wiring; imit finite + positive at
   the reset stance, zero when annealed off, and absent file disables cleanly.
5. PPO smoke: train(--geometry walker), 3 updates — losses finite, schedules
   move, checkpoint saves.
6. Determinism: two runs, same seed -> identical rollout rewards.
7. Mechanism checks: known escape hatches trigger CaT violations; a plausible
   crawl step does not.

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

from walker_improved import DEFAULTS, LEGS, build_walker  # noqa: E402
from walker_warp_env import EvalTelemetry, WalkerWarpEnv  # noqa: E402


@pytest.fixture(scope="module")
def small_env():
    return WalkerWarpEnv(nworld=4, seed=3, device="cpu", episode_length=None)


def _addressing(m):
    aj = [int(m.actuator_trnid[a, 0]) for a in range(m.nu)]
    qa = np.array([m.jnt_qposadr[j] for j in aj])
    da = np.array([m.jnt_dofadr[j] for j in aj])
    return aj, qa, da


# ---------------------------------------------------------------------------
# 1. obs parity vs numpy mirror of the mesh _obs layout + priv shape
# ---------------------------------------------------------------------------
def test_obs_parity_numpy_mirror(small_env):
    env = small_env
    m = mujoco.MjModel.from_xml_string(build_walker(DEFAULTS, floor=True))
    aj, qa, da = _addressing(m)

    rng = np.random.default_rng(7)
    n = env.nworld
    qpos = np.tile(m.qpos0, (n, 1))
    qpos[:, qa] += rng.uniform(-0.1, 0.1, (n, m.nu))
    half = rng.uniform(-np.pi, np.pi, n) / 2
    tilt = rng.uniform(-0.1, 0.1, (n, 2))
    quat = np.stack([np.cos(half), tilt[:, 0], tilt[:, 1], np.sin(half)], axis=1)
    quat /= np.linalg.norm(quat, axis=1, keepdims=True)
    qpos[:, 3:7] = quat
    qpos[:, 0:3] = rng.uniform(-1, 1, (n, 3))
    qpos[:, 2] = 0.42 + rng.uniform(-0.05, 0.05, n)
    qvel = rng.uniform(-1.0, 1.0, (n, m.nv))
    cmd = rng.uniform(-0.6, 0.6, (n, 3))
    prev_a = rng.uniform(-1.0, 1.0, (n, m.nu))

    env.qpos.copy_(torch.as_tensor(qpos, dtype=torch.float32))
    env.qvel.copy_(torch.as_tensor(qvel, dtype=torch.float32))
    env._cmd = torch.as_tensor(cmd, dtype=torch.float32)
    env._prev_a = torch.as_tensor(prev_a, dtype=torch.float32)
    obs = env.observe().cpu().numpy()
    assert obs.shape == (n, 50)
    assert env.privileged().shape == (n, 34)

    qp = env.qpos.cpu().numpy().astype(np.float64)
    qv = env.qvel.cpu().numpy().astype(np.float64)
    for w in range(n):
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
# 2. reset validity: drop-settle to an upright stance with feet on the floor
# ---------------------------------------------------------------------------
def test_reset_drop_settles_upright(monkeypatch):
    # This is a physical stance test, not a command-progress/CaT test. With CaT
    # enabled, zero action under a sampled nonzero command is intentionally reset
    # before it can settle (termination behavior is covered below).
    import walker_warp_env as W
    monkeypatch.setattr(W, "CAT_ON", False)
    env = WalkerWarpEnv(nworld=8, seed=1, device="cpu", episode_length=None)
    obs = env.reset()
    assert obs.shape == (env.nworld, 50) and torch.isfinite(obs).all()
    # noise actually landed on the actuated joints (not a frozen qpos0 copy)
    qa = env._qa.cpu().numpy()
    qp0 = env.qpos.cpu().numpy()
    assert np.abs(qp0[:, qa] - qp0[0, qa]).max() > 1e-4
    # lift slides are parked near the deep-knee nominal, inside range
    lift = qp0[:, env._lift_q.cpu().numpy()]
    assert np.abs(lift - DEFAULTS["lift_nom"]).max() < 0.02
    assert (lift >= 0.0).all() and (lift <= DEFAULTS["lift_range"]).all()
    # hold nominal (zero action) and let it settle onto its feet
    a = torch.zeros((env.nworld, env.act_dim))
    for _ in range(60):
        env.step(a, alpha=1.0, imit_anneal=0.0)
    zt = env.xpos[:, env._torso, 2]
    up = 1.0 - 2.0 * (env.qpos[:, 4] ** 2 + env.qpos[:, 5] ** 2)
    footz = env.geom_xpos[:, env._feet, 2]
    assert (zt > 0.30).all(), f"torso collapsed: min z {float(zt.min()):.3f}"
    assert (up > 0.9).all(), f"not upright: min up {float(up.min()):.3f}"
    assert (footz < 0.03).float().mean() > 0.8, "feet not settled on the floor"


# ---------------------------------------------------------------------------
# 3. CUDA env smoke: a training-scale batch over 200 random steps
# ---------------------------------------------------------------------------
@pytest.mark.gpu
def test_env_smoke_random_actions(gpu_device):
    env = WalkerWarpEnv(nworld=256, seed=0, device=gpu_device, episode_length=50)
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
    assert np.isfinite(list(m.values())).all()


# ---------------------------------------------------------------------------
# 4. imitation hook: reference_gait_walker.json wiring + anneal gating
# ---------------------------------------------------------------------------
def test_reference_gait_hook(tmp_path):
    # the shipped reference loads by default and is positive at the reset stance
    env = WalkerWarpEnv(nworld=4, seed=5, device="cpu")
    assert env.gait_loaded, "reference_gait_walker.json did not load"
    a = torch.zeros((4, env.act_dim))
    _, r_on, _, info_on = env.step(a, imit_anneal=1.0)
    assert torch.isfinite(info_on["imit"]).all()
    assert bool((info_on["imit"] > 0).all()), "imitation term should be positive at stance"
    env2 = WalkerWarpEnv(nworld=4, seed=5, device="cpu")
    _, r_off, _, info_off = env2.step(a, imit_anneal=0.0)
    assert bool((info_off["imit"] == 0).all())
    assert bool((r_on > r_off).all()), "annealed-off imitation must not add reward"
    # absent file -> hook disabled, env still steps
    env3 = WalkerWarpEnv(nworld=2, seed=5, device="cpu", gait_path=tmp_path / "missing.json")
    assert not env3.gait_loaded
    obs, rew, _, _ = env3.step(torch.zeros((2, env3.act_dim)))
    assert torch.isfinite(obs).all() and torch.isfinite(rew).all()


# ---------------------------------------------------------------------------
# 5. PPO smoke via --geometry walker
# ---------------------------------------------------------------------------
def _tiny_args(tmp_path, tag, steps, seed=0, device="cpu"):
    from train_mesh_warp import build_args
    return build_args(["--geometry", "walker", "--steps", str(steps), "--envs", "8",
                       "--horizon", "8", "--episode-length", "50", "--hidden", "32,32",
                       "--seed", str(seed), "--device", device, "--evals", "1",
                       "--eval-envs", "4", "--eval-steps", "8", "--preflight", "off",
                       "--tag", str(tmp_path / tag)])


@pytest.mark.gpu
def test_ppo_smoke_walker(tmp_path, gpu_device):
    from train_mesh_warp import train
    stats = train(_tiny_args(tmp_path, "wsmoke", steps=3 * 8 * 8, device=gpu_device))
    ups = stats["updates"]
    assert len(ups) == 3
    for u in ups:
        assert np.isfinite([u["pi_loss"], u["v_loss"], u["entropy"]]).all()
    assert ups[0]["ent_coef"] != ups[2]["ent_coef"], "entropy schedule did not move"
    assert ups[0]["alpha"] != ups[2]["alpha"], "alpha curriculum did not move"
    ck = torch.load(stats["ckpt"], map_location="cpu", weights_only=False)
    assert ck["step"] == 3 * 8 * 8 and ck["args"]["geometry"] == "walker"
    assert ck["contract"]["geometry"] == "walker"
    assert ck["runtime"] is not None


# ---------------------------------------------------------------------------
# 6. determinism-ish: same seed -> same first-update losses
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


# ---------------------------------------------------------------------------
# 7. anti-hack rewards + CaT — the direct anti-creep mechanism (targeted check)
# ---------------------------------------------------------------------------
def test_anti_hack_slip_and_clearance():
    """A world DRAGGING a planted foot (in contact, moving) earns a negative slip
    contribution AND a nonzero CaT slip violation; a world LIFTING a foot cleanly
    earns a positive clearance reward with ~zero CaT violation. Exercises the exact
    quantities step() assembles, on hand-set Data (mjwp.forward, no physics fight)."""
    import mujoco_warp as mjwp
    import walker_warp_env as W
    env = WalkerWarpEnv(nworld=2, seed=0, device="cpu", episode_length=None)
    qpos = env.qpos.clone()
    qvel = torch.zeros_like(env.qvel)
    qpos[0, 2] = 0.36          # world 0: drop torso so all feet are IN CONTACT
    qvel[0, 0] = 0.5           #          base vx -> planted feet DRAG at ~0.5 m/s
    qpos[1, 2] = 0.44          # world 1: raise the whole robot -> feet SWING (clear)
    env.qpos.copy_(qpos)
    env.qvel.copy_(qvel)
    mjwp.forward(env._wm, env._wd)       # refresh cvel/geom_xpos/subtree_com

    foot_z = env.geom_xpos[:, env._feet, 2]
    cf = (foot_z < W.WALKER_CONTACT_Z).float()
    hs = env._foot_hspeed()
    assert torch.isfinite(hs).all()

    # world 0: dragging planted foot -> positive slip, negative reward, CaT violated
    assert bool((cf[0] > 0).all()), "drag world feet should register as in contact"
    slip0 = float((hs[0] * cf[0]).sum() / cf[0].sum().clamp(min=1.0))
    assert slip0 > 0.1, f"dragging a planted foot must have positive slip, got {slip0}"
    assert -W.SLIP_W * slip0 < 0.0, "slip must subtract from reward"
    v_slip0 = float(((hs[0] * cf[0]).amax() - W.CAT_SLIP_LIMIT).clamp(min=0.0))
    assert v_slip0 > 0.0, "dragging must produce a nonzero CaT slip violation"

    # world 1: clean lift -> positive clearance reward, ~zero CaT slip/orient violation
    swing1 = 1.0 - cf[1]
    assert bool((swing1 > 0).any()), "lift world should have swing feet"
    clear1 = float((foot_z[1].clamp(max=W.CLEAR_TARGET) * swing1).sum()
                   / swing1.sum().clamp(min=1.0))
    assert clear1 > 0.0, "lifting a foot cleanly must earn clearance reward"
    up1 = float(1.0 - 2.0 * (env.qpos[1, 4] ** 2 + env.qpos[1, 5] ** 2))
    v_slip1 = float(((hs[1] * cf[1]).amax() - W.CAT_SLIP_LIMIT).clamp(min=0.0))
    v_orient1 = max(W.CAT_UP_MIN - up1, 0.0)
    assert v_slip1 == 0.0 and v_orient1 == 0.0, "clean upright lift => low CaT violation"


# ---------------------------------------------------------------------------
# 8. escape-hatch matrix: each known degenerate behavior has a structural
#    non-tradeoff violation; a plausible crawl step clears the constraints.
# ---------------------------------------------------------------------------
def test_cat_escape_hatch_matrix():
    import walker_warp_env as W
    env = WalkerWarpEnv(nworld=6, seed=0, device="cpu", episode_length=None)
    env.qvel.zero_()
    env.qvel[3, int(env._da[0])] = W.CAT_QVEL_LIMIT * 1.5
    env.qvel[3, 2] = W.CAT_BODY_VZ_LIMIT * 1.5

    # rows: 0 standing, 1 foot-drag/slide, 2 in-place hop, 3 flail,
    #       4 clean step, 5 two-foot-anchor wheelbarrow
    cf = torch.tensor([[1, 1, 1, 1],
                       [1, 1, 1, 1],
                       [0, 0, 0, 0],
                       [1, 1, 1, 1],
                       [1, 1, 1, 0],
                       [1, 1, 0, 0]], dtype=torch.float32)
    foot_hspeed = torch.tensor([[0.00, 0.00, 0.00, 0.00],
                                [0.35, 0.30, 0.32, 0.31],
                                [0.00, 0.00, 0.00, 0.00],
                                [0.00, 0.00, 0.00, 0.00],
                                [0.02, 0.03, 0.01, 0.60],
                                [0.02, 0.02, 0.20, 0.20]], dtype=torch.float32)
    up = torch.ones(6)
    cmd_norm = torch.full((6,), 0.25)
    active = torch.ones(6)
    mature = torch.ones(6, dtype=torch.bool)
    progress_ema = torch.tensor([0.00, 0.08, 0.00, 0.08, 0.08, 0.12], dtype=torch.float32)
    duty_ema = torch.tensor([1.00, 0.98, 0.00, 1.00, 0.80, 0.70], dtype=torch.float32)
    foot_duty_ema = torch.tensor([[1.00, 1.00, 1.00, 1.00],
                                  [0.98, 0.98, 0.98, 0.98],
                                  [0.00, 0.00, 0.00, 0.00],
                                  [0.90, 0.90, 0.90, 0.90],
                                  [0.80, 0.75, 0.70, 0.40],
                                  [0.99, 1.00, 0.40, 0.40]], dtype=torch.float32)

    terms, _ = env._cat_violations(cf, foot_hspeed, up, cmd_norm, active,
                                   mature, mature, progress_ema, duty_ema, foot_duty_ema)

    assert float(terms["cat_progress"][0]) > 0.5, "standing must violate min progress"
    assert float(terms["cat_duty"][0]) > 0.0, "standing must violate commanded duty cap"
    assert float(terms["cat_foot_duty"][0]) > 0.0, "standing anchors every foot"
    assert float(terms["cat_slip"][1]) > 0.0, "foot-drag creep/sliding must violate slip"
    assert float(terms["cat_duty"][1]) > 0.0, "high-duty crawl must violate duty cap"
    assert float(terms["cat_progress"][2]) > 0.5, "in-place hopping must violate progress"
    assert float(terms["cat_support"][2]) > 0.0, "all-air hopping must violate support"
    assert float(terms["cat_qvel"][3]) > 0.0, "flailing joints must violate qvel"
    assert float(terms["cat_body"][3]) > 0.0, "flailing body motion must violate body limits"
    for k in W.CAT_TERM_KEYS:
        assert float(terms[k][4]) == 0.0, f"clean crawl step should not violate {k}"
    assert float(terms["cat_duty"][5]) == 0.0, "aggregate duty alone misses two-foot anchoring"
    assert float(terms["cat_foot_duty"][5]) > 0.0, "anchored individual feet must violate CaT"


# ---------------------------------------------------------------------------
# 9. CaT terminations fire, OR into done, and drive the catrate telemetry;
#    MESH_CAT off disables them cleanly
# ---------------------------------------------------------------------------
@pytest.mark.gpu
def test_cat_terminations_and_telemetry(monkeypatch, gpu_device):
    import walker_warp_env as W
    env = WalkerWarpEnv(nworld=256, seed=2, device=gpu_device, episode_length=200)
    gen = torch.Generator(device=gpu_device).manual_seed(5)
    tel = EvalTelemetry(env.device)
    cat_terms = torch.zeros((), device=env.device)
    contracts_ok = torch.ones((), dtype=torch.bool, device=env.device)
    for _ in range(120):
        a = torch.rand((env.nworld, env.act_dim), generator=gen, device=env.device) * 2 - 1
        _, rew, done, info = env.step(a, alpha=0.6, imit_anneal=0.3)
        contracts_ok &= ((info["cat_delta"] >= 0) & (info["cat_delta"] <= 1)).all()
        contracts_ok &= (done.bool() | ~info["cat_done"].bool()).all()
        cat_terms += info["cat_done"].sum()
        tel.add(rew, info)
    m = tel.result()
    assert bool(contracts_ok), "CaT delta or termination contract failed"
    assert 0.0 <= m["catrate"] <= 1.0 and np.isfinite(m["catrate"])
    assert int(cat_terms) > 0, "random actions should trigger some CaT terminations"

    # MESH_CAT off -> no CaT terminations ever, catrate telemetry stays 0
    monkeypatch.setattr(W, "CAT_ON", False)
    env2 = WalkerWarpEnv(nworld=256, seed=2, device=gpu_device, episode_length=None)
    tel2 = EvalTelemetry(env2.device)
    cat_off = torch.ones((), dtype=torch.bool, device=env2.device)
    for _ in range(50):
        a = torch.rand((env2.nworld, env2.act_dim), generator=gen, device=env2.device) * 2 - 1
        _, rew, _, info = env2.step(a, alpha=0.6, imit_anneal=0.3)
        cat_off &= (info["cat_done"] == 0).all()
        tel2.add(rew, info)
    assert bool(cat_off), "CaT off must never terminate"
    assert tel2.result()["catrate"] == 0.0


# ---------------------------------------------------------------------------
# 10. determinism: same seed -> identical CaT terminations + rewards
# ---------------------------------------------------------------------------
def test_cat_determinism_same_seed():
    torch.set_num_threads(1)

    def run(seed):
        env = WalkerWarpEnv(nworld=6, seed=seed, device="cpu", episode_length=40)
        gen = torch.Generator().manual_seed(123)     # identical action stream each run
        cats, rews = [], []
        for _ in range(60):
            a = torch.rand((6, env.act_dim), generator=gen) * 2 - 1
            _, rew, _, info = env.step(a, alpha=0.5, imit_anneal=0.5)
            cats.append(info["cat_done"].clone())
            rews.append(rew.clone())
        return torch.stack(cats), torch.stack(rews)

    c1, r1 = run(7)
    c2, r2 = run(7)
    assert bool((c1 == c2).all()), "same seed must give identical CaT terminations"
    assert torch.equal(r1, r2), "same seed must give identical rewards"
    # a different env seed changes the stochastic draws (RNG actually bites)
    c3, _ = run(8)
    assert not bool((c1 == c3).all()), "different seed should change CaT terminations"
