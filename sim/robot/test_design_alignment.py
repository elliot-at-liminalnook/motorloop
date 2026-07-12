# SPDX-License-Identifier: MIT
"""test_design_alignment.py — enforce that every model describes the SAME robot.

The gear-bug class of error survives when models drift silently. `robot_design.py`
is the single source; these contract tests assert that every model describes the
same 6 lb robot with twelve Waveshare ST3215-HS bus servos. The walker, model checks and
reference-gait paths use TARGET geometry; CURRENT retains the CAD transmission
geometry but uses the same selected actuator BOM.

Run: .venv-warp/bin/python -m pytest sim/robot/test_design_alignment.py -q
"""
import math
import sys
from pathlib import Path

import pytest

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

import robot_design as rd
import walker_improved as wi
from gen_mesh_robot_mjcf import GEARS, WFREE


# --- geometry is single-sourced: the RL env now sources TARGET (Level C step 3) --
def test_walker_geometry_is_the_spec():
    assert wi.DEFAULTS == rd.TARGET.geometry()


def test_walker_masses_are_the_spec():
    assert wi.LEG_MASS == rd.LEG_MASS
    assert wi.TORSO_MASS == rd.TORSO_MASS


def test_reference_gait_uses_the_walker_joint_contract():
    """The checked-in reference is consumed directly by the Warp environment."""
    import json
    raw = json.loads((HERE / "reference_gait_walker.json").read_text())
    expected = [f"{leg}_{joint}" for leg in ("FL", "FR", "RL", "RR")
                for joint in ("hip_yaw", "pitch", "lift")]
    assert raw["joint_order"] == expected
    assert raw["gait_class"] == "forward_static_crawl_walker"


# --- CURRENT faithfully reproduces the CAD-derived mesh actuation --------------
def test_current_reproduces_live_actuation():
    g = rd.CURRENT.gears()          # [yaw, pitch, knee] x4
    w = rd.CURRENT.wfrees()
    # yaw/pitch are the gen_mesh GEARS/WFREE (ST3215-HS belt6 / worm20)
    assert math.isclose(g[0], GEARS["hip_yaw"], rel_tol=1e-3)
    assert math.isclose(g[1], GEARS["leg_swing"], rel_tol=1e-3)
    assert math.isclose(w[0], WFREE["hip_yaw"], rel_tol=1e-3)
    assert math.isclose(w[1], WFREE["leg_swing"], rel_tol=1e-3)
    # knee = direct servo on the crank: force/speed are gen_mesh's crank torque/speed
    # referred through the lift crank (N·m -> N, rad/s -> m/s)
    assert math.isclose(g[2], GEARS["knee_blade"] / rd.LIFT_CRANK, rel_tol=1e-3)
    assert math.isclose(w[2], WFREE["knee_blade"] * rd.LIFT_CRANK, rel_tol=1e-3)


# --- the spec's actuation is internally consistent ----------------------------
def test_target_actuation_self_consistent():
    for axis in (rd.TARGET.yaw, rd.TARGET.pitch, rd.TARGET.knee):
        assert axis.kind.startswith("servo")
        assert axis.part == "waveshare_st3215_hs"
    assert rd.TARGET.servo_count() == 12
    assert math.isclose(rd.TARGET.actuator_mass(), 12 * 0.068, abs_tol=1e-12)


def test_target_is_all_servo_with_the_yaw_belt_sea():
    assert rd.TARGET.knee.kind == "servo"
    assert rd.TARGET.yaw.kind == "servo_sea"
    assert rd.TARGET.pitch.kind == "servo"
    assert math.isclose(rd.TARGET.total_mass(), rd.ROBOT_MASS_LIMIT_KG, abs_tol=1e-12)


# --- the two gaps Level C (step 3) CLOSED — the RL env == TARGET ---------------
def test_rl_geometry_matches_target():
    # walker_improved.build_walker (what the env builds) is on the TARGET geometry
    # (lever 16 cm, stance 40x56): DEFAULTS is sourced from robot_design.TARGET.
    assert wi.DEFAULTS == rd.TARGET.geometry()


def test_rl_actuation_matches_target():
    # The env's per-actuator gear and torque-speed no-load speed come from TARGET.
    import mujoco
    m = mujoco.MjModel.from_xml_string(wi.build_walker())   # from spec, never a disk path
    gears = [float(m.actuator_gear[a, 0]) for a in range(m.nu)]
    assert gears == pytest.approx(rd.TARGET.gears(), rel=1e-3)   # XML emits 6 sig figs
    assert wi._DESIGN.wfrees() == pytest.approx(rd.TARGET.wfrees())
    assert wi._DESIGN is rd.TARGET      # active RL design is the selected hardware target


def test_compiled_walker_obeys_six_pound_limit():
    import mujoco
    m = mujoco.MjModel.from_xml_string(wi.build_walker())
    got = float(m.body_mass.sum())
    assert got <= rd.ROBOT_MASS_LIMIT_KG + 1e-9
    assert got == pytest.approx(rd.TARGET.total_mass(), abs=1e-9)
