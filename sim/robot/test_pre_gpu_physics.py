# SPDX-License-Identifier: MIT
"""Cheap physical invariants that must hold before launching GPU training."""

from __future__ import annotations

import math
import sys
from pathlib import Path

import mujoco
import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

import robot_design as rd  # noqa: E402
from gen_mesh_robot_mjcf import (  # noqa: E402
    SERVO_COUNT,
    SERVO_MASS,
    SERVO_NO_LOAD_CURRENT_A,
    SERVO_STALL_CURRENT_A,
)
from gen_robot_mjcf import load_spec  # noqa: E402
from validate_body import validate  # noqa: E402
from walker_improved import DEFAULTS, build_walker  # noqa: E402


def test_complete_servo_bank_mass_and_current_are_explicit():
    assert SERVO_COUNT == 12
    assert math.isclose(SERVO_COUNT * SERVO_MASS, 0.816, abs_tol=1e-12)
    assert math.isclose(SERVO_COUNT * SERVO_NO_LOAD_CURRENT_A, 2.88, abs_tol=1e-12)
    assert math.isclose(SERVO_COUNT * SERVO_STALL_CURRENT_A, 28.8, abs_tol=1e-12)
    assert rd.TARGET.total_mass() <= rd.ROBOT_MASS_LIMIT_KG


def test_compiled_inertias_are_positive_and_physically_valid():
    m = mujoco.MjModel.from_xml_string(build_walker(DEFAULTS, floor=True))
    inertia = np.asarray(m.body_inertia[1:])
    assert np.all(inertia > 0.0)
    # Principal moments of a physical rigid body obey the triangle inequalities.
    assert np.all(inertia[:, 0] <= inertia[:, 1] + inertia[:, 2] + 1e-12)
    assert np.all(inertia[:, 1] <= inertia[:, 0] + inertia[:, 2] + 1e-12)
    assert np.all(inertia[:, 2] <= inertia[:, 0] + inertia[:, 1] + 1e-12)


def _addresses(m):
    joints = np.asarray(m.actuator_trnid[:m.nu, 0], dtype=int)
    return (np.asarray(m.jnt_qposadr[joints], dtype=int),
            np.asarray(m.jnt_dofadr[joints], dtype=int))


def _run_airborne(dt, seconds=1.0):
    m = mujoco.MjModel.from_xml_string(build_walker(DEFAULTS, floor=True))
    m.opt.timestep = dt
    d = mujoco.MjData(m)
    d.qpos[2] = 1.0
    qadr, dadr = _addresses(m)
    gear = np.asarray(m.actuator_gear[:m.nu, 0])
    kp = np.asarray([30.0, 80.0, 1000.0] * 4)
    wfree = np.asarray(rd.TARGET.wfrees())
    mujoco.mj_forward(m, d)
    control_dt = 0.02
    ncontrol = round(seconds / control_dt)
    substeps = round(control_dt / dt)
    assert math.isclose(substeps * dt, control_dt, abs_tol=1e-12)
    for tick in range(ncontrol):
        phase = 2.0 * math.pi * tick / ncontrol
        target = np.asarray(m.qpos0[qadr]) + np.asarray(
            [0.08 * math.sin(phase), 0.04 * math.cos(phase),
             0.002 * math.sin(phase)] * 4)
        for _ in range(substeps):
            tau = kp * (target - d.qpos[qadr])
            qd = d.qvel[dadr]
            limit = np.where(tau * qd > 0.0,
                             np.clip(1.0 - np.abs(qd) / wfree, 0.0, 1.0), 1.0)
            d.ctrl[:] = np.clip(tau / gear, -limit, limit)
            mujoco.mj_step(m, d)
    return d.qpos.copy(), d.qvel.copy()


def test_active_walker_is_stable_under_timestep_halving():
    """A policy should not be rewarded for a coarse-integrator artifact."""
    q4, v4 = _run_airborne(0.004)
    q2, v2 = _run_airborne(0.002)
    assert np.isfinite(q4).all() and np.isfinite(q2).all()
    assert np.mean(np.abs(q4 - q2)) < 5e-3
    assert np.max(np.abs(q4 - q2)) < 5e-2
    assert np.mean(np.abs(v4 - v2)) < 5e-2


def test_complete_body_trainability_proof_executes_on_pinned_mujoco():
    result = validate(load_spec(HERE / "robot.toml"), verbose=False)
    assert result["ok"]
    assert 0.3 < result["effort"] < 3.0
    assert all(check[1] for check in result["checks"])
