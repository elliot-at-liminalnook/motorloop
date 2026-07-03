# SPDX-License-Identifier: MIT
"""M3(a) gate: trajectory parity — appending our kernels to the mujoco_warp
step MUST NOT perturb the physics.

Three exact (bitwise) checks at matched settings on the fight scene:
  1. plain mjwp.step loop vs FightLayer.step_fused on the same model/state ->
     identical qpos sequences (our kernels only READ Data and WRITE our own
     buffers);
  2. engine-sensors-enabled vs -disabled models -> identical qpos (sensors
     never touch dynamics), which makes the baseline/fused physics identical
     by construction;
  3. the fused layer's step count bookkeeping (t) advances per control step.
"""
from __future__ import annotations

import sys
from pathlib import Path

import mujoco
import numpy as np
import pytest
import warp as wp

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parents[1]))      # sim/robot: warplayer + gen_robot_mjcf + constants

import mujoco_warp as mjwp  # noqa: E402
from warplayer.fused import FightLayer, build_fight_model  # noqa: E402

wp.init()

NWORLD = 3
CTRL_STEPS = 6


def _plain_rollout(mjm, qvel0, ctrl, n_ctrl, frame_skip):
    """Reference: bare mujoco_warp, no bespoke kernels."""
    mjd = mujoco.MjData(mjm)
    mujoco.mj_resetData(mjm, mjd)
    mujoco.mj_forward(mjm, mjd)
    m = mjwp.put_model(mjm)
    d = mjwp.put_data(mjm, mjd, nworld=NWORLD)
    d.qvel.assign(qvel0)
    mjwp.forward(m, d)
    d.ctrl.assign(ctrl)
    traj = []
    for _ in range(n_ctrl):
        for _ in range(frame_skip):
            mjwp.step(m, d)
        traj.append(d.qpos.numpy().copy())
    return traj


def test_fused_kernels_do_not_perturb_the_step():
    lay = FightLayer(nworld=NWORLD, mode="fused", lidar=True, seed=11)
    rng = np.random.default_rng(11)
    act = rng.uniform(-0.5, 0.5, (NWORLD, lay.idx.nuA))
    lay.set_actions(act)

    ctrl = np.zeros((NWORLD, lay.mjm.nu), dtype=np.float32)
    ctrl[:, lay.idx.actA] = np.clip(act, -1, 1).astype(np.float32)
    ref = _plain_rollout(lay.mjm, lay._qvel0, ctrl, CTRL_STEPS, lay.frame_skip)

    for k in range(CTRL_STEPS):
        lay.step_fused()
        got = lay.d.qpos.numpy()
        assert np.array_equal(got, ref[k]), (
            f"qpos diverged at control step {k}: max|Δ|="
            f"{np.abs(got - ref[k]).max():.3e} — the appended kernels perturbed the step")


def test_sensor_stage_disable_is_physics_neutral():
    """baseline model (rangefinders on) and fused model (mjDSBL_SENSOR) march
    through IDENTICAL qpos — so the M3 benchmark compares equal physics."""
    mjm_on, _ = build_fight_model(lidar=True, disable_sensors=False)
    mjm_off, _ = build_fight_model(lidar=True, disable_sensors=True)
    rng = np.random.default_rng(13)
    qvel0 = rng.uniform(-0.05, 0.05, (NWORLD, mjm_on.nv)).astype(np.float32)
    ctrl = np.zeros((NWORLD, mjm_on.nu), dtype=np.float32)
    ta = _plain_rollout(mjm_on, qvel0, ctrl, CTRL_STEPS, 5)
    tb = _plain_rollout(mjm_off, qvel0, ctrl, CTRL_STEPS, 5)
    for k, (a, b) in enumerate(zip(ta, tb)):
        assert np.array_equal(a, b), f"sensor stage changed physics at control step {k}"


def test_carried_state_advances():
    lay = FightLayer(nworld=NWORLD, mode="fused", lidar=False, seed=17)
    assert np.all(lay.t.numpy() == 0)
    for _ in range(4):
        lay.step_fused()
    assert np.all(lay.t.numpy() == 4)
    assert np.all(lay.prev_dist.numpy() > 0)     # spawn separation ~1.2 m
    lay.reset(seed=18)
    assert np.all(lay.t.numpy() == 0)


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-q"]))
