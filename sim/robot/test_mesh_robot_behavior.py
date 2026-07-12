#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
"""Behavioral validation of the mesh robot: it STANDS, STRIKES, HOLDS, and
rolls out deterministically through the actuator path (d.ctrl), never applied forces.

Stand pose: knee -50 deg via loop_consistent_pose (initializing the knee without
the matched toe/slide values is a 26 mm constraint violation — see generator).
"""
from __future__ import annotations

import math
import sys
from pathlib import Path

import mujoco
import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

from gen_mesh_robot_mjcf import (GEARS, build_mesh_robot, loop_consistent_pose,  # noqa: E402
                                 slider_crank_s)

LEGS = ("FL", "FR", "RL", "RR")
STAND_KNEE = math.radians(-50)
XML = build_mesh_robot(floor=True)
M = mujoco.MjModel.from_xml_string(XML)
DT = M.opt.timestep                      # 0.004 (fleet standard — quartic loop couplings)


def jid(n):
    return mujoco.mj_name2id(M, mujoco.mjtObj.mjOBJ_JOINT, n)


def aid(n):
    return mujoco.mj_name2id(M, mujoco.mjtObj.mjOBJ_ACTUATOR, n)


GEAR = {"yaw": GEARS["hip_yaw"], "swing": GEARS["leg_swing"],
        "knee": GEARS["knee_blade"]}


def make_data(stand=True, drop_h=0.0):
    d = mujoco.MjData(M)
    if stand:
        psi, s = loop_consistent_pose(STAND_KNEE)
        for L in LEGS:
            d.qpos[M.jnt_qposadr[jid(f"{L}_knee_blade")]] = STAND_KNEE
            d.qpos[M.jnt_qposadr[jid(f"{L}_toe_hinge")]] = psi
            d.qpos[M.jnt_qposadr[jid(f"{L}_pushrod_slide")]] = s
    d.qpos[2] += drop_h
    mujoco.mj_forward(M, d)
    return d


def pd_ctrl(d, targets, kp=(2.0, 40.0, 6.0), kd=None):
    """Stance P-control through the ACTUATORS (ctrl in [-1,1], honest force limits).

    P-ONLY by design (2026-07-03): the old explicit -kd*qvel term through the
    clipped ctrl was a bounded version of the same explicit-damping instability
    that blew the contract sweep (c/m >> 2/dt on the light crank dofs once the
    polycoef loop removed the connect's reflected inertia) — under contact load
    it chattered the knees at 3 N.m bang-bang and tripped QACC warnings. Joint
    damping (implicit under implicitfast, unconditionally stable) + actuator
    saturation provide the damping path the hardware actually has. `kd` is kept
    in the signature (ignored) so older call sites don't break."""
    for L in LEGS:
        for k, (j, g, p) in enumerate((("hip_yaw", "yaw", kp[0]),
                                       ("leg_swing", "swing", kp[1]),
                                       ("knee_blade", "knee", kp[2]))):
            q = d.qpos[M.jnt_qposadr[jid(f"{L}_{j}")]]
            tau = p * (targets[k] - q)
            d.ctrl[aid(f"{L}_{('yaw_m', 'swing_m', 'knee_m')[k]}")] = \
                np.clip(tau / GEAR[g], -1, 1)


# ------------------------------------------------------------------ 1. drop+stand
def test_drop_settles_standing():
    d = make_data(stand=True, drop_h=0.02)
    peak_pen = 0.0
    for _ in range(int(2.0 / DT)):                       # 2 s
        pd_ctrl(d, (0.0, 0.0, STAND_KNEE))
        mujoco.mj_step(M, d)
        if d.ncon:
            peak_pen = max(peak_pen, float(np.maximum(0, -d.contact.dist).max()))
    assert np.all(np.isfinite(d.qpos)), "exploded during settle"
    z = float(d.qpos[2])
    up = float(d.xmat[mujoco.mj_name2id(M, mujoco.mjtObj.mjOBJ_BODY, "torso")].reshape(3, 3)[2, 2])
    assert 0.33 < z < 0.52, f"torso settled at z={z:.3f}"
    assert up > 0.9, f"toppled: up_z={up:.2f}"
    assert peak_pen < 0.05, f"peak penetration {peak_pen:.3f}"


# --------------------------------------------------------------- 2. powered strike
def test_powered_strike_full_stroke_unloaded():
    """Zero-g on a local model copy (a 0.5 s 'airborne' test under gravity FALLS A
    METRE and slams the blades into the floor — measured: the landing rams the
    pistons to +27 mm; this test is about the MECHANISM, not the crash):
    the 3 N.m knee must complete the full stroke unloaded."""
    m2 = mujoco.MjModel.from_xml_string(XML)
    m2.opt.gravity[:] = 0.0
    d = mujoco.MjData(m2)
    psi, s0 = loop_consistent_pose(STAND_KNEE)
    for L in LEGS:
        d.qpos[m2.jnt_qposadr[jid(f"{L}_knee_blade")]] = STAND_KNEE
        d.qpos[m2.jnt_qposadr[jid(f"{L}_toe_hinge")]] = psi
        d.qpos[m2.jnt_qposadr[jid(f"{L}_pushrod_slide")]] = s0
    d.qpos[2] += 1.0
    mujoco.mj_forward(m2, d)
    for _ in range(int(0.5 / DT)):
        pd_ctrl(d, (0.0, 0.0, math.radians(-90)), kp=(2.0, 40.0, 30.0))
        mujoco.mj_step(m2, d)
    assert np.all(np.isfinite(d.qpos))
    for L in LEGS:
        s = float(d.qpos[m2.jnt_qposadr[jid(f"{L}_pushrod_slide")]])
        assert s <= -0.040, f"{L}: unloaded strike reached {s * 1000:.1f} mm (want <= -40)"


def test_powered_strike_lifts_the_body_loaded():
    """From stance, a 4-leg stomp must EXTEND and LIFT the robot. Full -41 mm is
    NOT expected loaded: at -90 the toggle is at its weakest advantage — physics.
    The selected ST3215-HS supplies 1.96 N.m at the crank and the model carries
    the full 6 lb class mass. The outcome gates below are measured behavior, not
    an attribute-level claim."""
    d = make_data(stand=True)
    for _ in range(int(1.0 / DT)):                       # settle 1 s
        pd_ctrl(d, (0.0, 0.0, STAND_KNEE))
        mujoco.mj_step(M, d)
    z0 = float(d.qpos[2])
    for _ in range(int(0.5 / DT)):                       # strike 0.5 s
        pd_ctrl(d, (0.0, 0.0, math.radians(-90)), kp=(2.0, 40.0, 30.0))
        mujoco.mj_step(M, d)
    assert np.all(np.isfinite(d.qpos))
    lift = float(d.qpos[2]) - z0
    assert lift > 0.012, f"stomp lifted the torso only {lift * 1000:.1f} mm (want > 12)"
    for L in LEGS:
        s = float(d.qpos[M.jnt_qposadr[jid(f"{L}_pushrod_slide")]])
        assert s <= -0.015, f"{L}: loaded strike reached {s * 1000:.1f} mm (want <= -15, servo-true)"


# ------------------------------------------------------------------ 3. worm holds
def test_worm_selflocking_holds_unpowered():
    d = make_data(stand=True)
    d.qpos[2] += 1.0                                     # airborne: leg dangles
    sw = M.jnt_qposadr[jid("FL_leg_swing")]
    d.qpos[sw] = math.radians(20)
    mujoco.mj_forward(M, d)
    d.ctrl[:] = 0.0                                      # UNPOWERED
    for _ in range(int(2.0 / DT)):
        mujoco.mj_step(M, d)
    drift = abs(math.degrees(d.qpos[sw]) - 20.0)
    assert drift < 5.0, (
        f"swing drifted {drift:.1f} deg unpowered — frictionloss approximation of the "
        f"self-locking worm is not holding (raise WORM_FRICTIONLOSS or bench-measure)")


# ---------------------------------------------------------- 4. deterministic CPU anchor
def test_mujoco_rollout_is_deterministic():
    """A fixed control tape produces an exact repeatable CPU reference."""
    tape = np.random.default_rng(7).uniform(-0.2, 0.2, (20, M.nu))

    def rollout():
        d = make_data(stand=True)
        out = []
        for ctrl in tape:
            d.ctrl[:] = ctrl
            for _ in range(5):
                mujoco.mj_step(M, d)
            out.append(d.qpos.copy())
        return np.asarray(out)

    first, second = rollout(), rollout()
    assert np.isfinite(first).all()
    np.testing.assert_array_equal(first, second)


# -------------------------------------------------------------------- 5. yaw sweep
def test_yaw_tracks_through_belt_placeholder():
    d = make_data(stand=True)
    d.qpos[2] += 1.0                                     # airborne: isolate the axis
    mujoco.mj_forward(M, d)
    yq = M.jnt_qposadr[jid("FL_hip_yaw")]
    for tgt_deg in (30.0, -30.0):
        for _ in range(int(1.5 / DT)):
            pd_ctrl(d, (math.radians(tgt_deg), 0.0, STAND_KNEE))
            mujoco.mj_step(M, d)
        err = abs(math.degrees(d.qpos[yq]) - tgt_deg)
        assert err < 8.0, f"yaw settled {err:.1f} deg from {tgt_deg} target"
    # NOTE: v1 models the belt as rigid + placeholder damping. Once the rubber
    # pulley's torsional stiffness is bench-measured, replace with a rotor+spring
    # SEA and re-tune this tolerance (steering will get springier, not stiffer).


if __name__ == "__main__":
    import pytest
    raise SystemExit(pytest.main([__file__, "-q", "-s"]))
