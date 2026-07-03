#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
"""DR.2 cross-engine contract: MuJoCo-measured vs Drake-derived, no shared fate.

The MJCF path (gen_robot_mjcf.py) and the Drake path (gen_robot_drake.py) compile
the same robot.toml through two implementations that share no physical derivation
— only the raw spec load and the motors.py primitives. For both engines to agree
on OUTCOMES, the spec->model compilation has to be right twice. That converts the
single-simulator self-consistency bug class (the 2026-07 gear bug: a coherently
weak model every within-engine check confirmed) from undetectable-in-principle to
red-CI-the-same-day.

Agreement bands (fixed BEFORE the asserts, per the plan — failures are decisions
already made; contact-free quantities only, per the kill-criterion note):
  1. deliverable torque   — ctrl=1 -> mj_forward -> qfrc_actuator at the hinge dof
                            (post-gear joint space; NOT actuator_force, which is
                            pre-gear and reads 1.0 even on a correct model)
                            == Drake JointActuator effort limit, within 1%
  2. total mass           — sum(MuJoCo body_mass) == Drake CalcTotalMass, within 1%
  3. gravity stance torque— canonical stance (stand_abd/stand_flex/stand_knee),
                            base held (MuJoCo: mj_inverse with qacc=qvel=0,
                            contact disabled, feet in the air — the free base held
                            at zero acceleration is exactly the welded base;
                            Drake: -CalcGravityGeneralizedForces on a
                            weld_base=True plant): per-hinge within 3%, PLUS the
                            stance-hold feasibility bar — no hinge's gravity
                            torque may exceed 60% of the deliverable torque, in
                            BOTH engines' numbers
  4. firing demos         — the contract must demonstrably catch (a) the
                            historical gear bug (stripped model: MuJoCo collapses
                            to 1 N·m, engines disagree >10x, contract 1 fires) and
                            (b) a drifted spec (+10% torso mass on the MuJoCo
                            build only: contract 2 fires)

Adjudication policy: a firing contract is a STOP-THE-LINE event. The tiebreaker
is a hand calculation or a bench-measured number — never "trust the engine we
train on."
"""
from __future__ import annotations

import copy
import re
import sys
from pathlib import Path

import mujoco
import numpy as np
import pytest

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

from gen_robot_mjcf import build_mjcf, load_spec  # noqa: E402
from gen_robot_drake import build_plant, hinge_effort_limits  # noqa: E402

SPEC = load_spec(HERE / "robot.toml")

HINGE_SUFFIXES = ("_abd", "_flex", "_knee")
TORQUE_RTOL = 0.01          # deliverable torque band (contract 1)
MASS_RTOL = 0.01            # total mass band (contract 2)
GRAVITY_RTOL = 0.03         # per-hinge gravity-statics band (contract 3)
GRAVITY_ATOL = 0.02         # N·m floor so a near-zero hinge can't fail on noise
STANCE_HOLD_FRAC = 0.60     # gravity torque <= 60% of deliverable (plan A.3 bar)


# ------------------------------------------------------------ MuJoCo-side probes
def _mj_model(spec: dict) -> mujoco.MjModel:
    # build_mjcf default = striker OFF: the same 12-hinge contact-free body the
    # Drake plant models (DR.1 scope excludes the striker).
    return mujoco.MjModel.from_xml_string(build_mjcf(spec))


def _mj_hinge_actuators(m: mujoco.MjModel) -> dict[str, int]:
    """hinge joint name -> actuator id (matched by name, never by index order)."""
    out: dict[str, int] = {}
    for i in range(m.nu):
        jid = m.actuator_trnid[i, 0]
        jname = mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_JOINT, jid) or ""
        if jname.endswith(HINGE_SUFFIXES):
            out[jname] = i
    return out


def mj_measured_hinge_torque(m: mujoco.MjModel) -> dict[str, float]:
    """Joint-space torque each hinge delivers at full command (the trainer's path).

    ctrl=1 -> mj_forward -> qfrc_actuator at the hinge's dof. Post-gear, so a
    missing gear attribute (the historical bug) collapses this to 1.0 N·m while
    every attribute-level field still documents the design intent.
    """
    d = mujoco.MjData(m)
    acts = _mj_hinge_actuators(m)
    for i in acts.values():
        d.ctrl[i] = 1.0
    mujoco.mj_forward(m, d)
    return {j: float(d.qfrc_actuator[m.jnt_dofadr[m.actuator_trnid[i, 0]]])
            for j, i in acts.items()}


def mj_gravity_stance_torque(spec: dict) -> dict[str, float]:
    """Per-hinge gravity-compensation torque at the canonical stance, base held.

    mj_inverse with qacc=0, qvel=0 asks: what applied generalized force yields
    zero acceleration everywhere? Holding the free base at zero acceleration is
    mechanically identical to welding it, so the hinge entries are the welded-
    base gravity torques. Feet contact is ignored (torso high + contact disabled);
    the parallel springs are neutral at stance (springref == stance), damping is
    zero at qvel=0, so qfrc_inverse reduces to pure gravity statics.
    """
    m = _mj_model(spec)
    m.opt.disableflags |= mujoco.mjtDisableBit.mjDSBL_CONTACT
    d = mujoco.MjData(m)
    stance = {"abd": float(spec["leg_defaults"]["stand_abd"]),
              "flex": float(spec["leg_defaults"]["stand_flex"]),
              "knee": float(spec["leg_defaults"]["stand_knee"])}
    d.qpos[:] = 0.0
    d.qpos[2] = 1.0          # feet well off the floor
    d.qpos[3] = 1.0          # identity quaternion
    hinges: dict[str, int] = {}
    for j in range(m.njnt):
        name = mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_JOINT, j) or ""
        if name.endswith(HINGE_SUFFIXES):
            d.qpos[m.jnt_qposadr[j]] = stance[name.rsplit("_", 1)[1]]
            hinges[name] = m.jnt_dofadr[j]
    d.qvel[:] = 0.0
    d.qacc[:] = 0.0
    mujoco.mj_inverse(m, d)
    return {name: float(d.qfrc_inverse[dof]) for name, dof in hinges.items()}


# ------------------------------------------------------------- Drake-side probes
def drake_gravity_stance_torque(spec: dict) -> dict[str, float]:
    """Per-hinge gravity-compensation torque from the WELDED-base Drake plant.

    The default context IS the canonical stance (build_plant sets stand_* as the
    default angles). Compensation = -tau_g from CalcGravityGeneralizedForces.
    """
    from pydrake.multibody.tree import JointIndex
    plant = build_plant(spec, weld_base=True)
    ctx = plant.CreateDefaultContext()
    tau_g = plant.CalcGravityGeneralizedForces(ctx)
    out: dict[str, float] = {}
    for i in range(plant.num_joints()):
        j = plant.get_joint(JointIndex(i))
        if j.name().endswith(HINGE_SUFFIXES):
            out[j.name()] = float(-tau_g[j.velocity_start()])
    return out


# --------------------------------------------------- the contracts (as helpers,
# so the firing demos can prove each one has teeth by catching its AssertionError)
def _assert_deliverable_torque_agrees(mj: dict[str, float], dk: dict[str, float]) -> None:
    assert set(mj) == set(dk), (
        f"hinge sets differ: mj-only {set(mj) - set(dk)}, drake-only {set(dk) - set(mj)}")
    for j in sorted(mj):
        assert abs(mj[j] - dk[j]) <= TORQUE_RTOL * dk[j], (
            f"{j}: MuJoCo delivers {mj[j]:.4f} N·m vs Drake effort limit "
            f"{dk[j]:.4f} N·m (band {TORQUE_RTOL:.0%}) — the engines disagree on "
            f"actuation; stop the line")


def _assert_total_mass_agrees(mj_spec: dict, drake_spec: dict) -> None:
    mj_mass = float(_mj_model(mj_spec).body_mass.sum())
    plant = build_plant(drake_spec)
    dk_mass = float(plant.CalcTotalMass(plant.CreateDefaultContext()))
    assert abs(mj_mass - dk_mass) <= MASS_RTOL * dk_mass, (
        f"total mass: MuJoCo {mj_mass:.4f} kg vs Drake {dk_mass:.4f} kg "
        f"(band {MASS_RTOL:.0%}) — the models drifted; stop the line")


# ------------------------------------------------------------ 1. deliverable torque
def test_deliverable_torque_matches_across_engines():
    mj = mj_measured_hinge_torque(_mj_model(SPEC))
    dk = hinge_effort_limits(build_plant(SPEC))
    _assert_deliverable_torque_agrees(mj, dk)


# ------------------------------------------------------------------ 2. total mass
def test_total_mass_matches_across_engines():
    _assert_total_mass_agrees(SPEC, SPEC)


# -------------------------------------------------------- 3. gravity stance torque
def test_gravity_stance_torque_matches_across_engines():
    mj = mj_gravity_stance_torque(SPEC)
    dk = drake_gravity_stance_torque(SPEC)
    assert set(mj) == set(dk)
    for j in sorted(mj):
        tol = max(GRAVITY_RTOL * abs(dk[j]), GRAVITY_ATOL)
        assert abs(mj[j] - dk[j]) <= tol, (
            f"{j}: stance gravity torque MuJoCo {mj[j]:+.4f} vs Drake {dk[j]:+.4f} "
            f"N·m (band {GRAVITY_RTOL:.0%}) — mass/geometry assembly diverged")


def test_stance_hold_feasible_in_both_engines():
    """Plan DR.2 assertion 3: gravity torque < 60% of deliverable, both engines."""
    limits = hinge_effort_limits(build_plant(SPEC))
    for label, torques in [("MuJoCo", mj_gravity_stance_torque(SPEC)),
                           ("Drake", drake_gravity_stance_torque(SPEC))]:
        for j, g in torques.items():
            assert abs(g) <= STANCE_HOLD_FRAC * limits[j], (
                f"{label} {j}: holding stance needs {abs(g):.3f} N·m — more than "
                f"{STANCE_HOLD_FRAC:.0%} of the {limits[j]:.3f} N·m deliverable")


# --------------------------------------- 4. firing demos: the contract has teeth
def test_contract_fires_on_gear_stripped_model():
    """(a) The historical bug, as a day-one detection — demonstrated, not argued.

    Strip the gear attribute (the exact pre-2026-07 defect): the MuJoCo side
    collapses to the gear-default 1 N·m while Drake's independent derivation
    still says ~12.97 — a >10x cross-engine disagreement contract 1 must catch.
    """
    xml = build_mjcf(SPEC)
    stripped = re.sub(r' gear="[^"]*"', "", xml)
    assert 'gear=' not in stripped.split("<actuator>", 1)[1], (
        "strip failed; the demo is vacuous")
    mj = mj_measured_hinge_torque(mujoco.MjModel.from_xml_string(stripped))
    dk = hinge_effort_limits(build_plant(SPEC))
    for j in sorted(mj):
        assert abs(mj[j] - 1.0) <= 0.01, (
            f"{j}: expected the historical 1 N·m signature, measured {mj[j]:.4f}")
        assert dk[j] / mj[j] > 10.0, (
            f"{j}: engines no longer disagree >10x on the bugged model "
            f"(mj {mj[j]:.3f} vs drake {dk[j]:.3f}) — the demo lost its teeth")
    with pytest.raises(AssertionError):
        _assert_deliverable_torque_agrees(mj, dk)   # the contract MUST fire


def test_contract_fires_on_drifted_torso_mass():
    """(b) Spec drift between builders becomes loud: +10% torso mass on the
    MuJoCo build only (Drake still compiles the pristine spec) must trip the
    1% total-mass contract."""
    drifted = copy.deepcopy(SPEC)
    drifted["torso"]["mass"] = 1.10 * float(drifted["torso"]["mass"])
    with pytest.raises(AssertionError):
        _assert_total_mass_agrees(mj_spec=drifted, drake_spec=SPEC)


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-q"]))
