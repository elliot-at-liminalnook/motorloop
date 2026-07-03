# SPDX-License-Identifier: MIT
"""M3 component (iii) gate: the 144-ray lidar kernel vs the MuJoCo C reference.

Oracles, on the ACTUAL fight scene (build_match(..., lidar=True), 128
horizontal + 16 vertical rays on A_torso):
  1. the C rangefinder sensordata (mj_forward -> mj_ray per sensor) at
     randomized robot poses — exact-ish (float32 ray math vs float64 C);
  2. mujoco.mj_ray called directly with the same origin/direction/bodyexclude;
  3. batched nworld=4 with per-world states — each world matches ITS OWN C
     reference (the (nworld, ray) thread grid carries independent worlds).
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
from warplayer.fused import build_fight_model  # noqa: E402
from warplayer.lidar import Lidar  # noqa: E402

wp.init()

ATOL = 5e-4        # meters, over rays up to 2 m — float32 quadratic-root noise


def _random_state(mjm, seed):
    """A randomized but sane fight pose: hinge perturbations + torso shifts."""
    rng = np.random.default_rng(seed)
    mjd = mujoco.MjData(mjm)
    mujoco.mj_resetData(mjm, mjd)
    for r in ("A_root", "B_root"):
        q = mjm.jnt_qposadr[mujoco.mj_name2id(mjm, mujoco.mjtObj.mjOBJ_JOINT, r)]
        mjd.qpos[q:q + 3] += rng.uniform(-0.15, 0.15, 3)
        quat = rng.normal(size=4) * 0.15 + np.array([1.0, 0, 0, 0])
        mjd.qpos[q + 3:q + 7] = quat / np.linalg.norm(quat)
    hinge = [j for j in range(mjm.njnt) if mjm.jnt_type[j] == mujoco.mjtJoint.mjJNT_HINGE]
    for j in hinge:
        a = mjm.jnt_qposadr[j]
        lo, hi = mjm.jnt_range[j]
        mjd.qpos[a] = rng.uniform(lo, hi)
    mujoco.mj_forward(mjm, mjd)
    return mjd


@pytest.fixture(scope="module")
def scene():
    mjm, _ = build_fight_model(lidar=True)
    return mjm


def test_lidar_vs_c_rangefinder_randomized(scene):
    mjm = scene
    for seed in range(5):
        mjd = _random_state(mjm, seed)
        m = mjwp.put_model(mjm)
        d = mjwp.put_data(mjm, mjd, nworld=1)
        lid = Lidar(mjm, nworld=1)
        lid.launch(m, d)
        ours = lid.dist.numpy()[0]
        ref = mjd.sensordata.copy()
        assert np.array_equal(ours < 0, ref < 0), f"seed {seed}: hit/miss pattern differs"
        hits = ref >= 0
        assert hits.sum() > 0, f"seed {seed}: degenerate pose (no hits) — bad test state"
        err = np.abs(ours[hits] - ref[hits]).max()
        assert err <= ATOL, f"seed {seed}: max ray error {err:.2e} m > {ATOL}"


def test_lidar_vs_mj_ray_direct(scene):
    """Bypass the sensor pipeline: same rays through mujoco.mj_ray directly
    (flg_static=1, geomgroup=None, bodyexclude=torso — the rangefinder call,
    engine_sensor.c / mujoco_warp sensor.py:832-845)."""
    mjm = scene
    mjd = _random_state(mjm, 42)
    m = mjwp.put_model(mjm)
    d = mjwp.put_data(mjm, mjd, nworld=1)
    lid = Lidar(mjm, nworld=1)
    lid.launch(m, d)
    ours = lid.dist.numpy()[0]

    torso = mujoco.mj_name2id(mjm, mujoco.mjtObj.mjOBJ_BODY, "A_torso")
    R = mjd.xmat[torso].reshape(3, 3)
    origin = mjd.xpos[torso] + R @ np.array([0.0, 0.0, 0.03])
    dirs = lid.dirs_local.numpy() @ R.T
    geomid = np.zeros(1, dtype=np.int32)
    for i in range(0, lid.nray, 7):            # subsample: 21 rays is plenty
        dist = mujoco.mj_ray(mjm, mjd, origin, dirs[i].astype(np.float64),
                             None, 1, torso, geomid)
        if geomid[0] < 0:
            assert ours[i] == -1.0, f"ray {i}: kernel hit, mj_ray missed"
        else:
            want = min(dist, 2.0)              # positive-sensor cutoff clamp
            assert abs(ours[i] - want) <= ATOL, f"ray {i}: {ours[i]} vs mj_ray {want}"


def test_lidar_batched_worlds_independent(scene):
    mjm = scene
    nworld = 4
    states = [_random_state(mjm, 100 + w) for w in range(nworld)]
    m = mjwp.put_model(mjm)
    d = mjwp.put_data(mjm, states[0], nworld=nworld)
    d.qpos.assign(np.stack([s.qpos for s in states]).astype(np.float32))
    mjwp.forward(m, d)                          # batched FK for geom/site poses
    lid = Lidar(mjm, nworld=nworld)
    lid.launch(m, d)
    ours = lid.dist.numpy()
    for w, s in enumerate(states):
        ref = s.sensordata
        hits = ref >= 0
        assert np.array_equal(ours[w] < 0, ref < 0), f"world {w}: hit/miss differs"
        err = np.abs(ours[w][hits] - ref[hits]).max()
        assert err <= ATOL, f"world {w}: max ray error {err:.2e} m"
    # different worlds genuinely see different scans
    assert not np.allclose(ours[0], ours[1])


def test_scan_normalization_matches_env_rule(scene):
    """scan == train_adversarial._lidar_scan clean branch: miss -> 1.0, else d/max."""
    mjm = scene
    mjd = _random_state(mjm, 7)
    m = mjwp.put_model(mjm)
    d = mjwp.put_data(mjm, mjd, nworld=1)
    lid = Lidar(mjm, nworld=1, max_range=2.0)
    lid.launch(m, d)
    raw = lid.dist.numpy()[0]
    scan = lid.scan.numpy()[0]
    want = np.where(raw < 0, 2.0, np.clip(raw, 0, 2.0)) / 2.0
    np.testing.assert_allclose(scan, want, atol=1e-7)
    assert scan.min() >= 0.0 and scan.max() <= 1.0


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-q"]))
