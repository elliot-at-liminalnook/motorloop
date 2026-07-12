#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
"""T0 model-build contract: the compiled MjModel delivers the physics the spec claims.

Every check exercises OUTCOMES through the same causal path training uses
(write ctrl -> forward -> measure joint-space force), never a metadata field.
This is the test that makes the gear-bug class impossible: the pre-2026-07
generator emitted <motor forcerange="±tau"/> with no gear attribute, so every
hinge silently maxed at gear-default 1 N·m while the
forcerange attribute documented an intent no actuator could deliver. Every
check here would have been green under that model EXCEPT the outcome checks —
test_contract_fires_on_gear_stripped_model proves they fire on exactly that bug.

  1. gear attribute actually emitted (build_mjcf AND build_match paths)
  2. ctrl=1 -> measured joint torque == tau  (qfrc_actuator, post-gear joint
     space — NOT d.actuator_force, which is pre-gear and reads 1.0 even on a
     correct model)
  3. total mass matches the spec's explicit masses (±5%)
  4. joint ranges match robot.toml
  5. timestep × frame_skip == the 0.02 s control dt the reward math assumes
  6. foot geoms exist and can collide with the floor
  7. the suite catches the historical bug (gear-stripped model must FAIL #2)
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

from gen_robot_mjcf import (actuator_unit_mass, build_match, build_mjcf,  # noqa: E402
                            joint_torque_limit, load_spec)
from gen_mesh_robot_mjcf import MAX_ROBOT_MASS_KG  # noqa: E402

SPEC = load_spec(HERE / "robot.toml")
TAU = joint_torque_limit(SPEC)
# V.1: the contract asserts the COMPILED MODEL agrees with the shared constants
# module — if either side drifts, this is where it breaks loud.
from constants import CONTROL_DT, FRAME_SKIP, TIMESTEP  # noqa: E402

HINGE_SUFFIXES = ("_abd_m", "_flex_m", "_knee_m")


def _model(xml: str) -> mujoco.MjModel:
    return mujoco.MjModel.from_xml_string(xml)


def _hinges(m: mujoco.MjModel) -> list[int]:
    return [i for i in range(m.nu)
            if (mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_ACTUATOR, i) or "").endswith(HINGE_SUFFIXES)]


def measured_hinge_torque(m: mujoco.MjModel) -> np.ndarray:
    """Joint-space torque each hinge actually delivers at full command.

    The causal path the trainer uses: write ctrl -> mj_forward -> read
    qfrc_actuator at the hinge's dof. Post-gear, so it exposes a missing gear
    attribute that every attribute-level check (forcerange, spec, comments)
    is blind to.
    """
    d = mujoco.MjData(m)
    hinges = _hinges(m)
    d.ctrl[hinges] = 1.0
    mujoco.mj_forward(m, d)
    dofs = [m.jnt_dofadr[m.actuator_trnid[i, 0]] for i in hinges]
    return d.qfrc_actuator[dofs]


# ---------------------------------------------------------------- 1. attribute
def test_gear_emitted_on_every_hinge_both_build_paths():
    for label, xml in [("build_mjcf", build_mjcf(SPEC)),
                       ("build_match", build_match(SPEC, SPEC, sep=1.2,
                                                   striker=True, striker_b=True))]:
        m = _model(xml)
        hinges = _hinges(m)
        assert hinges, f"{label}: no hinge actuators found"
        gears = m.actuator_gear[hinges, 0]
        assert np.allclose(gears, TAU, rtol=0.01), (
            f"{label}: hinge gear {np.unique(gears)} != tau {TAU:.3f} — "
            f"gear attribute missing or wrong (the historical ~8%-torque bug)")


# ------------------------------------------------------------------ 2. outcome
def test_ctrl_one_delivers_tau_at_the_joint():
    for label, xml in [("build_mjcf", build_mjcf(SPEC)),
                       ("build_match", build_match(SPEC, SPEC, sep=1.2,
                                                   striker=True, striker_b=True))]:
        tq = measured_hinge_torque(_model(xml))
        assert np.all(np.abs(tq - TAU) < 0.05 * TAU), (
            f"{label}: measured joint torque {np.unique(np.round(tq, 3))} != "
            f"tau {TAU:.3f} — actuators cannot deliver the design torque")


# --------------------------------------------------------------------- 3. mass
def _expected_mass(spec: dict, striker: bool) -> float:
    d = spec["leg_defaults"]
    legs = spec["leg"]
    total = float(spec["torso"]["mass"])
    total += len(legs) * (float(d.get("hip_mass", 0.3))
                          + float(d["thigh_mass"]) + float(d["calf_mass"])
                          + float(d.get("foot_mass", 0.05))
                          + 3 * actuator_unit_mass(spec))
    if striker:
        s = spec["striker"]
        r, ln = float(s["rod_radius"]), float(s["rod_len"])
        rod_mass = float(s["rod_density"]) * (math.pi * r * r * ln + 4 / 3 * math.pi * r ** 3)
        n_front = sum(1 for leg in legs if leg["pos"][0] > 0)
        total += n_front * rod_mass
    return total


def test_total_mass_matches_spec():
    for striker in (False, True):
        m = _model(build_mjcf(SPEC, striker=striker))
        want = _expected_mass(SPEC, striker)
        got = float(m.body_mass.sum())
        assert abs(got - want) <= 0.05 * want, (
            f"striker={striker}: model mass {got:.3f} kg vs spec {want:.3f} kg")
        assert got <= MAX_ROBOT_MASS_KG + 1e-9, (
            f"striker={striker}: {got / 0.45359237:.3f} lb exceeds the 6 lb limit")
        if striker:
            assert math.isclose(got, MAX_ROBOT_MASS_KG, abs_tol=1e-8), (
                "armed model should exercise the conservative maximum-mass envelope")


def test_all_twelve_hinges_are_selected_bus_servos():
    assert SPEC["actuator"]["motor"] == "waveshare_st3215_hs"
    assert SPEC["actuator"]["voltage"] == 12.0
    m = _model(build_mjcf(SPEC))
    assert len(_hinges(m)) == 12
    assert math.isclose(12 * actuator_unit_mass(SPEC), 0.816, abs_tol=1e-12)


# ------------------------------------------------------------------- 4. ranges
def test_joint_ranges_match_spec():
    m = _model(build_mjcf(SPEC))
    d = SPEC["leg_defaults"]
    want = {"abd": d["abd_range"], "flex": d["flex_range"], "knee": d["knee_range"]}
    for j in range(m.njnt):
        name = mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_JOINT, j) or ""
        for suffix, rng in want.items():
            if name.endswith("_" + suffix):
                assert np.allclose(m.jnt_range[j], rng, atol=1e-9), (
                    f"{name}: range {m.jnt_range[j]} != spec {rng}")


# ----------------------------------------------------------------------- 5. dt
def test_timestep_and_control_dt():
    m = _model(build_mjcf(SPEC))
    assert abs(m.opt.timestep - TIMESTEP) < 1e-12
    assert abs(m.opt.timestep * FRAME_SKIP - CONTROL_DT) < 1e-12, (
        "control dt drifted from the 0.02 s the reward math assumes")


# ------------------------------------------------------------------- 6. contact
def test_foot_geoms_exist_and_collide_with_floor():
    m = _model(build_mjcf(SPEC))
    floor = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_GEOM, "floor")
    assert floor >= 0
    feet = [g for g in range(m.ngeom)
            if (mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_GEOM, g) or "").endswith("_foot")]
    assert len(feet) == len(SPEC["leg"]), "one foot geom per leg"
    for g in feet:
        ok = ((m.geom_contype[g] & m.geom_conaffinity[floor])
              or (m.geom_contype[floor] & m.geom_conaffinity[g]))
        name = mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_GEOM, g)
        assert ok, f"{name} cannot collide with the floor (contype/conaffinity mask)"


# ------------------------------------------- 7. the test can catch the real bug
def strip_gear(xml: str) -> str:
    """Reproduce the historical defect: motors with forcerange but no gear."""
    return re.sub(r'(<motor [^>]*?) gear="[^"]*"', r"\1", xml)


def test_contract_fires_on_gear_stripped_model():
    xml = strip_gear(build_mjcf(SPEC))
    assert 'gear=' not in xml.split("<actuator>")[1], "strip failed; test is vacuous"
    tq = measured_hinge_torque(_model(xml))
    # On the bugged model the outcome check MUST fail loudly: torque collapses
    # to the gear-default 1 N·m, nowhere near tau. If this assert ever trips,
    # the contract has gone blind to the very bug it exists to catch.
    # It only needs to exceed test #2's 5% acceptance band. With the selected
    # 1.96 N.m servo, MuJoCo's default 1 N.m is about half the requested torque.
    assert np.all(np.abs(tq - TAU) > 0.05 * TAU), (
        f"gear-stripped model still delivers {np.unique(np.round(tq, 3))} N·m — "
        "the outcome check no longer discriminates; fix the test")
    assert np.allclose(tq, 1.0, atol=0.01), "historical signature is 1 N·m exactly"


if __name__ == "__main__":
    import pytest
    raise SystemExit(pytest.main([__file__, "-q"]))


# ------------------------------------------------------- 8. reflected inertia
def test_armature_is_derived_not_hardcoded():
    """dof armature == J_rotor x gear^2 from the motor entry (the hidden-actuator-
    property class the gear bug belonged to; 0.01-for-everything was luck)."""
    import sys as _s; _s.path.insert(0, str(HERE.parents[1]))
    from sim.tests.motors import MOTORS, SERVOS
    m = _model(build_mjcf(SPEC))
    a = SPEC["actuator"]
    if a["motor"] in SERVOS:
        want = SERVOS[a["motor"]].output_inertia_kg_m2_est * float(a["gear"]) ** 2
    else:
        want = MOTORS[a["motor"]].inertia_kg_m2 * float(a["gear"]) ** 2
    hinges = [m.jnt_dofadr[m.actuator_trnid[i, 0]] for i in _hinges(m)]
    got = m.dof_armature[hinges]
    assert np.allclose(got, want, rtol=0.01), (got[:3], want)
