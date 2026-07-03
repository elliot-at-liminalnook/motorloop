# SPDX-License-Identifier: MIT
"""M3 component (iv) gate: fused obs/reward kernels vs the numpy reference on
randomized states of the real fight scene.

The numpy reference (obsreward.obs_reference / reward_reference) mirrors
train_adversarial.AdversarialEnv line-by-line (citations in obsreward.py) and
is ALSO the baseline-mode computation in fused.py — so kernel-vs-reference
parity here certifies both "kernels are right" and "baseline == fused". A
direct comparison against the jax env itself needs .venv-sim (jax/brax) and is
out of scope for this venv; layout compatibility is pinned instead via
constants.LOCO_OBS and the obs_schema conventions.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest
import warp as wp

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parents[1]))      # sim/robot: warplayer + gen_robot_mjcf + constants

from warplayer.fused import FightLayer  # noqa: E402
from warplayer.obsreward import RewardConfig, reward_reference, obs_reference  # noqa: E402

wp.init()

NWORLD = 4


def _randomized_layer(mode, lidar, seed=0, cfg=None, steps=3, close=False):
    """A FightLayer advanced a few control steps under random actions so the
    state (incl. contacts) is nontrivial. Pool sizes are raised over the
    rest-state defaults: flailing at ±1 torque makes far more contacts/rows
    than the settled spawn put_data sizes for. `close=True` re-spawns B right
    next to A so weapon<->body contacts actually occur."""
    rng = np.random.default_rng(seed)
    lay = FightLayer(nworld=NWORLD, mode=mode, lidar=lidar, seed=seed, cfg=cfg,
                     design=rng.normal(0, 0.5, (NWORLD, 3)),
                     nconmax=128, njmax=256)
    if close:
        import mujoco
        qpos = lay.d.qpos.numpy().copy()
        brq = int(lay.mjm.jnt_qposadr[mujoco.mj_name2id(
            lay.mjm, mujoco.mjtObj.mjOBJ_JOINT, "B_root")])
        qpos[:, brq] = qpos[:, brq] - 0.85          # B spawns at +0.6; bring to ~0.25 m gap
        lay.d.qpos.assign(qpos)
        lay._mjwp.forward(lay.m, lay.d)
    lay.set_actions(rng.uniform(-1, 1, (NWORLD, lay.idx.nuA)))
    for _ in range(steps):
        lay.step()
    return lay


def test_obs_layout_matches_env_constants():
    from constants import LOCO_OBS
    lay = FightLayer(nworld=2, mode="fused", lidar=False)
    assert lay.obs_dim == LOCO_OBS + 6          # _obs: loco + privileged opp tail
    lay_l = FightLayer(nworld=2, mode="fused", lidar=True)
    assert lay_l.nray == 144                    # 128 horizontal + 16 vertical
    assert lay_l.obs_dim == LOCO_OBS + 144      # _lidar_obs actor, stack=1


def test_module_constants_pin_train_adversarial():
    """DAMAGE_REF/STRIKE_KINETIC are copied (train_adversarial imports jax, not
    importable here) — pin them against the source text so drift fails loudly."""
    from warplayer import obsreward
    src = (HERE.parents[1] / "train_adversarial.py").read_text()
    assert f"DAMAGE_REF = {obsreward.DAMAGE_REF}" in src
    assert f"STRIKE_KINETIC = {obsreward.STRIKE_KINETIC}" in src


@pytest.mark.parametrize("lidar", [False, True])
def test_obs_kernel_matches_reference(lidar):
    lay = _randomized_layer("fused", lidar=lidar, seed=1)
    got = lay.obs.numpy()
    h = lay._host_pull()
    scan = lay.lidar.scan.numpy() if lidar else None
    want = obs_reference(h, lay.idx, lay.design.numpy(), scan)
    assert got.shape == want.shape == (NWORLD, lay.obs_dim)
    np.testing.assert_allclose(got, want, atol=1e-5)
    # worlds are decorrelated (different qvel noise -> different obs)
    assert not np.allclose(got[0], got[1])


def test_reward_kernel_matches_reference_default_config():
    lay = _randomized_layer("fused", lidar=False, seed=2, steps=1)
    # replay ONE more step with recorded pre-step carries, then compare
    prev_dist = lay.prev_dist.numpy().copy().astype(np.float64)
    prev_dealt = lay.prev_dealt.numpy().copy().astype(np.float64)
    vel_ema = lay.vel_ema.numpy().copy().astype(np.float64)
    t = lay.t.numpy().copy().astype(np.float64)
    lay.step()
    h = lay._host_pull()
    want_r, want_d, want_pd, want_ps, want_ema, want_t = reward_reference(
        h, lay.idx, lay.cfg, prev_dist, prev_dealt, vel_ema, t)
    np.testing.assert_allclose(lay.reward.numpy(), want_r, atol=2e-5)
    np.testing.assert_allclose(lay.done.numpy(), want_d)
    np.testing.assert_allclose(lay.prev_dist.numpy(), want_pd, atol=1e-5)
    np.testing.assert_allclose(lay.prev_dealt.numpy(), want_ps, atol=1e-5)
    np.testing.assert_allclose(lay.vel_ema.numpy(), want_ema, atol=1e-5)
    np.testing.assert_allclose(lay.t.numpy(), want_t)


def test_reward_kernel_matches_reference_dense_weights_on():
    """Every dense term exercised: nonzero weights for the whole shaping set."""
    cfg = RewardConfig.from_constants(
        shaping=0.7, approach_w=1.3, upright_w=0.3, alive=0.1, energy_w=0.02,
        airborne_w=0.5, height_w=0.4, move_w=0.6, close_bonus_w=0.8,
        face_w=0.5, flee_w=0.3, taken_w=0.9, clean_w=1.1, trade_w=0.4,
        dis_w=0.6, damage_bonus_w=2.0, loco_speed=0.4, loco_track_w=8.0,
        early_hit_penalty=0.5, min_hit_step=2.0, require_closing=1.0,
        stationary_pen=0.7, oscillation_pen=0.2, penalty_w=1.5,
        fire_shaping=0.6, fire_cost=0.05)
    lay = _randomized_layer("fused", lidar=False, seed=3, cfg=cfg, steps=2)
    prev_dist = lay.prev_dist.numpy().copy().astype(np.float64)
    prev_dealt = lay.prev_dealt.numpy().copy().astype(np.float64)
    vel_ema = lay.vel_ema.numpy().copy().astype(np.float64)
    t = lay.t.numpy().copy().astype(np.float64)
    lay.step()
    h = lay._host_pull()
    want_r, want_d, *_ = reward_reference(h, lay.idx, cfg, prev_dist, prev_dealt, vel_ema, t)
    np.testing.assert_allclose(lay.reward.numpy(), want_r, atol=5e-5)
    np.testing.assert_allclose(lay.done.numpy(), want_d)


def test_damage_terms_seen_on_contact():
    """Spawn the robots nearly touching and flail: leg->body damage must
    register, proving the contact-pool kernel actually accumulates (not
    vacuously zero) — and it must match the numpy pool reference."""
    from warplayer.obsreward import damage_reference
    lay = _randomized_layer("fused", lidar=False, seed=5, steps=0, close=True)
    rng = np.random.default_rng(5)
    seen = 0.0
    for k in range(40):
        lay.set_actions(rng.uniform(-1, 1, (NWORLD, lay.idx.nuA)))
        lay.step()
        dealt = lay.dealt_leg.numpy()
        taken = lay.taken_leg.numpy()
        h = lay._host_pull()
        want = damage_reference(h, lay.idx, NWORLD)
        np.testing.assert_allclose(dealt, want[0], atol=1e-6,
                                   err_msg=f"dealt_leg mismatch at step {k}")
        np.testing.assert_allclose(taken, want[2], atol=1e-6,
                                   err_msg=f"taken_leg mismatch at step {k}")
        seen = max(seen, float(dealt.max() + taken.max()))
    assert seen > 0.0, ("40 flailing control steps at ~0.25 m produced zero leg damage "
                        "— masks or pool wiring broken")


def test_baseline_equals_fused_outputs():
    """The M3 wrapper-way baseline and the fused kernels produce the same
    obs/reward stream on the same trajectory (float32 tolerance)."""
    a = _randomized_layer("fused", lidar=True, seed=7, steps=0)
    b = _randomized_layer("baseline", lidar=True, seed=7, steps=0)
    rng = np.random.default_rng(7)
    for k in range(5):
        act = rng.uniform(-1, 1, (NWORLD, a.idx.nuA))
        a.set_actions(act)
        b.set_actions(act)
        a.step()
        b.step()
        np.testing.assert_allclose(a.obs.numpy(), b.obs.numpy(), atol=1e-5,
                                   err_msg=f"obs diverged at control step {k}")
        np.testing.assert_allclose(a.reward.numpy(), b.reward.numpy(), atol=2e-5,
                                   err_msg=f"reward diverged at control step {k}")
        np.testing.assert_allclose(a.done.numpy(), b.done.numpy())


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-q"]))
