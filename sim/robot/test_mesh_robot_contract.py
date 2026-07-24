#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
"""T0 contract for the MESH robot (gen_mesh_robot_mjcf): the compiled model
delivers the physics the assembly json claims — outcomes, not attributes.

Census note: the json's 6-joint chain lists {L}_heel_pin as a revolute; in the
MJCF that dof IS the loop equality (a tree joint there would double-count the
loop dof). The loop is closed by TWO 1-row quartic joint couplings per leg
(secret-sauce §8: full-rank at the toggle's dead center, unlike the retired
<connect>, which is what re-enabled the fleet dt=0.004). Census = 5 tree
joints/leg + 2 couplings/leg, asserted exactly.
"""
from __future__ import annotations

import math
import re
import sys
from pathlib import Path

import mujoco
import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

from gen_mesh_robot_mjcf import (ARMATURE, MAX_ROBOT_MASS_KG, MAX_ROBOT_MASS_LB,  # noqa: E402
                                 SERVO_COUNT, SERVO_FREE_RAD_S, SERVO_MASS,
                                 SERVO_MODEL, SERVO_NO_LOAD_CURRENT_A,
                                 SERVO_STALL_CURRENT_A, SERVO_STALL_NM,
                                 build_mesh_robot, load_assembly,
                                 loop_consistent_pose, slider_crank_s)

ASM = load_assembly()
LEGS = ("FL", "FR", "RL", "RR")
TREE_JOINTS = ("hip_yaw", "leg_swing", "knee_blade", "toe_hinge", "pushrod_slide")
XML = build_mesh_robot(ASM, floor=False)
M = mujoco.MjModel.from_xml_string(XML)


def jid(m, n):
    return mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_JOINT, n)


# ------------------------------------------------------------------- 1. census
def test_census_matches_json_chains():
    names = {mujoco.mj_id2name(M, mujoco.mjtObj.mjOBJ_JOINT, j) for j in range(M.njnt)}
    want = {"root"} | {f"{L}_{j}" for L in LEGS for j in TREE_JOINTS}
    assert names == want, names ^ want
    assert M.nu == 12 and M.neq == 8
    eq_names = {mujoco.mj_id2name(M, mujoco.mjtObj.mjOBJ_EQUALITY, i) for i in range(M.neq)}
    assert eq_names == ({f"{L}_loop_slide" for L in LEGS}
                        | {f"{L}_loop_toe" for L in LEGS})
    assert M.nv == 6 + 4 * len(TREE_JOINTS)


def test_timestep_is_fleet_standard():
    """THE payoff of the quartic couplings (secret-sauce §8): the mesh robot runs
    at the fleet's dt=0.004 again. If someone reverts to a loop closure that needs
    dt=0.002, this fails loudly instead of silently doubling every rollout cost."""
    assert M.opt.timestep == 0.004


def test_quartic_fit_matches_closed_form():
    """Gate (a): the emitted polynomials track the analytic slider-crank over the
    full knee ROM — independently re-derived here from the closed forms."""
    from gen_mesh_robot_mjcf import (KNEE_ROM, POLY_SLIDE, POLY_TOE, conrod_psi,
                                     poly_eval)
    lo, hi = KNEE_ROM
    phis = np.linspace(lo, hi, 1001)
    rs = max(abs(poly_eval(POLY_SLIDE, p) - slider_crank_s(p)) for p in phis)
    rp = max(abs(poly_eval(POLY_TOE, p) - conrod_psi(p)) for p in phis)
    assert rs < 5e-4, f"slide fit residual {rs * 1e3:.3f} mm"
    assert rp < math.radians(0.3), f"toe fit residual {math.degrees(rp):.3f} deg"


# ------------------------------------------------------- 2. ctrl->qfrc outcome
def test_ctrl_one_delivers_design_torque():
    from gen_mesh_robot_mjcf import GEARS
    want = {"yaw_m": GEARS["hip_yaw"], "swing_m": GEARS["leg_swing"],
            "knee_m": GEARS["knee_blade"]}   # single source: servo-true generator
    d = mujoco.MjData(M)
    for i in range(M.nu):
        name = mujoco.mj_id2name(M, mujoco.mjtObj.mjOBJ_ACTUATOR, i)
        mujoco.mj_resetData(M, d)
        d.ctrl[i] = 1.0
        mujoco.mj_forward(M, d)
        dof = M.jnt_dofadr[M.actuator_trnid[i, 0]]
        tau = float(d.qfrc_actuator[dof])
        expect = next(v for k, v in want.items() if name.endswith(k))
        assert abs(tau - expect) < 0.05 * expect, (name, tau, expect)


def test_all_twelve_actuators_use_the_selected_st3215_hs_envelope():
    """The part choice is a compiled-model contract, not just a catalog comment."""
    assert SERVO_MODEL == "waveshare_st3215_hs"
    assert M.nu == SERVO_COUNT == 12
    assert math.isclose(SERVO_STALL_NM, 20.0 * 0.0980665, rel_tol=1e-9)
    assert math.isclose(SERVO_FREE_RAD_S, 106.0 * 2 * math.pi / 60, rel_tol=1e-9)
    assert SERVO_MASS == 0.068
    assert SERVO_NO_LOAD_CURRENT_A == 0.240
    assert SERVO_STALL_CURRENT_A == 2.4
    contract = ASM["hardware_contract"]
    assert contract["motor_count"] == 12
    assert contract["motor_model"] == "Waveshare ST3215-HS"
    acts = [a for L in LEGS for a in ASM["legs"][L]["actuators"]]
    assert len(acts) == 12
    assert {a["motor_model"] for a in acts} == {"Waveshare ST3215-HS"}
    assert all(math.isclose(a["motor_mass_kg"], SERVO_MASS) for a in acts)


# ----------------------------------------------------------------- 3. armature
def test_armature_matches_json_guidance():
    for L in LEGS:
        for j, key in (("hip_yaw", "hip_yaw"), ("leg_swing", "leg_swing"),
                       ("knee_blade", "knee_blade")):
            dof = M.jnt_dofadr[jid(M, f"{L}_{j}")]
            assert np.isclose(M.dof_armature[dof], ARMATURE[key], rtol=0.01), (L, j)


# --------------------------------------------------------------------- 4. mass
def test_mass_census():
    assert math.isclose(SERVO_COUNT * SERVO_MASS, 0.816, abs_tol=1e-12)
    for sea in (False, True):
        model = mujoco.MjModel.from_xml_string(build_mesh_robot(ASM, floor=False, sea=sea))
        got = float(model.body_mass.sum())
        assert got <= MAX_ROBOT_MASS_KG + 1e-9, (
            f"sea={sea}: model is {got / 0.45359237:.6f} lb, "
            f"above {MAX_ROBOT_MASS_LB:.1f} lb")
        # The simulation intentionally carries the maximum legal mass so gravity
        # and inertia are conservative. Placeholder structure fills the remainder.
        assert math.isclose(got, MAX_ROBOT_MASS_KG, abs_tol=1e-9), (
            sea, got, MAX_ROBOT_MASS_KG)


# ----------------------------------------------------------- 5. limits + axes
def test_limits_and_axes_match_json():
    for L in LEGS:
        for j in ASM["legs"][L]["joints"]:
            short = j["name"].split("_", 1)[1]
            if short == "heel_pin":
                continue                                       # the connect (census test)
            jj = jid(M, j["name"])
            assert jj >= 0, j["name"]
            if "limit_deg" in j:
                want = np.radians(j["limit_deg"])
                assert np.allclose(M.jnt_range[jj], want, atol=1e-6), (j["name"],)
            elif "travel_m" in j:
                # slide range carries documented margins beyond the working stroke
                lo, hi = M.jnt_range[jj]
                assert lo <= j["travel_m"][0] and hi >= j["travel_m"][1], (j["name"],)
            assert np.allclose(M.jnt_axis[jj], j["axis_leg_local"], atol=1e-9), (j["name"],)


# --------------------------------------------- 6. loop closure tracks physics
def _servo_sweep(m, leg, targets_deg, steps_per=240):   # 240 x dt=0.004 = 0.96 s per target
    """Continuously servo the knee via the ACTUATOR (the training-causal path);
    return worst |slide - closed_form| in m.

    HARNESS LESSON (2026-07-03, proven by bisection): the old qfrc_applied servo
    with an explicit -0.3*qvel term was ITSELF unstable on this dof — with the
    polycoef loop the knee's smooth inertia is just crank+armature (~2.8e-4
    kg.m2), so the explicit damping eigenvalue c/m ~ 1000/s violates the
    dt*lambda <= 2 explicit limit at BOTH dt=0.004 and 0.002 (the old connect
    masked it by reflecting anchor inertia into the dof). Free system, P-only,
    and actuator-path servos are all stable; kd=0.3 alone reproduced the
    -587 deg windup. Actuator saturation + joint damping are the honest
    damping path."""
    d = mujoco.MjData(m)
    m.opt.gravity[:] = 0
    mujoco.mj_forward(m, d)
    kq = m.jnt_qposadr[jid(m, f"{leg}_knee_blade")]
    sq = m.jnt_qposadr[jid(m, f"{leg}_pushrod_slide")]
    act = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_ACTUATOR, f"{leg}_knee_m")
    worst = 0.0
    for x in targets_deg:
        tgt = math.radians(x)
        for _ in range(steps_per):
            if act >= 0:
                d.ctrl[act] = float(np.clip(4.0 * (tgt - d.qpos[kq]), -1, 1))
            mujoco.mj_step(m, d)
        assert np.all(np.isfinite(d.qpos)), f"{leg}: NaN during sweep"
        worst = max(worst, abs(d.qpos[sq] - slider_crank_s(d.qpos[kq])))
    return worst


SWEEP = list(range(0, -91, -5)) + list(range(-90, 11, 5))


def test_loop_tracks_closed_form_FL_and_mirrored_FR():
    for leg in ("FL", "FR"):                                   # FR is the mirrored one
        worst = _servo_sweep(M, leg, SWEEP)
        assert worst < 0.002, f"{leg}: loop err {worst * 1000:.2f} mm > 2 mm"


def test_consistent_pose_helper_closes_loop_exactly():
    d = mujoco.MjData(M)
    for phi_deg in (-90, -45, 10):
        phi = math.radians(phi_deg)
        psi, s = loop_consistent_pose(phi)
        mujoco.mj_resetData(M, d)
        for L in LEGS:
            d.qpos[M.jnt_qposadr[jid(M, f"{L}_knee_blade")]] = phi
            d.qpos[M.jnt_qposadr[jid(M, f"{L}_toe_hinge")]] = psi
            d.qpos[M.jnt_qposadr[jid(M, f"{L}_pushrod_slide")]] = s
        mujoco.mj_forward(M, d)
        eq_rows = d.efc_type == mujoco.mjtConstraint.mjCNSTR_EQUALITY
        resid = float(np.abs(d.efc_pos[eq_rows]).max()) if eq_rows.any() else 0.0
        assert resid < 1e-6, (phi_deg, resid)


# ------------------------------------------------- 7. the test sees the loop
def test_loop_test_fires_without_the_equality():
    xml = re.sub(r"<equality>.*?</equality>", "", XML, flags=re.S)
    m2 = mujoco.MjModel.from_xml_string(xml)
    assert m2.neq == 0, "strip failed; test is vacuous"
    worst = _servo_sweep(m2, "FL", list(range(0, -91, -10)))
    assert worst > 0.002, (
        f"loop check passed WITHOUT the constraint (err {worst * 1000:.2f} mm) — "
        "the tracking test has gone blind to the mechanism it verifies")


if __name__ == "__main__":
    import pytest
    raise SystemExit(pytest.main([__file__, "-q"]))


# --------------------------------------------- 8. L-R2 regressions (2026-07-03)
def test_no_transient_blowup():
    """The -587 deg first-hold windup, as a permanent regression: holding the
    first sweep target through the actuator path must keep the knee inside its
    ROM (+/-5 deg margin) and leave the rest of the robot quiet."""
    m = mujoco.MjModel.from_xml_string(XML)
    d = mujoco.MjData(m)
    m.opt.gravity[:] = 0
    act = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_ACTUATOR, "FL_knee_m")
    kq = m.jnt_qposadr[jid(m, "FL_knee_blade")]
    for _ in range(240):
        d.ctrl[act] = float(np.clip(4.0 * (0.0 - d.qpos[kq]), -1, 1))
        mujoco.mj_step(m, d)
    assert abs(math.degrees(d.qpos[kq])) < 5.0, (
        f"knee wandered to {math.degrees(d.qpos[kq]):.1f} deg holding 0")
    assert float(np.abs(d.qvel).max()) < 1.0, "robot not quiet after first hold"


def test_dt004_loop_stability():
    """dt=0.004 (fleet standard) end-to-end: model timestep IS 0.004, the free
    system holds loop consistency to numerical zero, and the full ROM sweep
    tracks the closed form. This is the claim that justified replacing the
    <connect> (dt=0.002-bound) with quartic couplings."""
    m = mujoco.MjModel.from_xml_string(XML)
    assert abs(m.opt.timestep - 0.004) < 1e-12, "fleet dt regressed"
    d = mujoco.MjData(m)
    m.opt.gravity[:] = 0
    mujoco.mj_forward(m, d)
    for _ in range(500):
        mujoco.mj_step(m, d)
    assert float(np.abs(d.qvel).max()) < 1e-9, "free system drifts at consistency"
    worst = _servo_sweep(mujoco.MjModel.from_xml_string(XML), "FL", SWEEP)
    assert worst < 0.002, f"dt=0.004 sweep err {worst * 1000:.2f} mm"
