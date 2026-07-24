# SPDX-License-Identifier: MIT
"""walker_improved.py — the three approved geometry changes, one parametric model.

Changes vs the combat leg (notes/gait-feasibility-verdict.md CORRECTION):
  1. DEEP-KNEE re-bias   — nominal foot sits mid-lift-range (knee proxy = lift
                           prismatic parked at lift_nom), leaving clearance room.
  2. LONGER YAW LEVER    — foot offset yaw_lever from the vertical yaw axis, so a
                           yaw sweep translates the foot (stride ∝ lever). Legs are
                           SPLAYED LATERALLY so the yaw axis (1.85 rad/s,
                           11.77 N·m stall) strides FORE-AFT, not sideways — the fix for the
                           "yaw stride is lateral" finding.
  3. WIDER STANCE        — foot at ±(hy+yaw_lever): the lever widens the support
                           polygon for free, shrinking the COM shift a step needs.

ST3215-HS limits + the 6 lb mass budget are shared with gen_mesh_robot_mjcf so
the feasibility result transfers to the selected twelve-servo build.
This is the WALKING model (not the slider-crank strike model — irrelevant here).
Consumed by both plain MuJoCo checks and the MuJoCo-Warp runtime.
"""
from __future__ import annotations

import math
import os
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from gen_mesh_robot_mjcf import (SEA_YAW_EFFECTIVE_INERTIA_EST,  # noqa: E402
                                 SEA_YAW_STIFFNESS_EST)
# SINGLE SOURCE OF TRUTH: geometry + masses + per-axis actuation come from
# robot_design so the RL env, model checks, co-design and reference can't drift.
# TARGET uses the improved lever-16 cm / stance-40x56 geometry and the same
# ST3215-HS servo on yaw, pitch and knee for every leg. The active design remains
# selectable via MESH_DESIGN for geometry comparisons; motor choice is identical.
from robot_design import (CURRENT, TARGET, LEG_MASS, SERVO_MASS,  # noqa: E402
                          TORSO_MASS)

_DESIGNS = {"current": CURRENT, "target": TARGET}
_DESIGN = _DESIGNS[os.environ.get("MESH_DESIGN", "target").strip().lower()]

LEGS = ("FL", "FR", "RL", "RR")
SIGN = {"FL": (+1, +1), "FR": (+1, -1), "RL": (-1, +1), "RR": (-1, -1)}
FOOT_R = 0.012
LIFT_FORCE = _DESIGN.knee.gear()         # knee actuator force (N) — from the spec
LIFT_SPEED = _DESIGN.knee.wfree()        # knee no-load foot speed (m/s) — from the spec

# The knee servo's output-inertia estimate is referred through the 40 mm crank to
# an equivalent slide mass by Axis.armature(). This is explicitly an estimate:
# Waveshare publishes neither output inertia nor continuous-duty torque.
_BASE_LIFT_ARMATURE = 0.02               # structural/bearing floor on the lift SLIDE


def _lift_armature(design=_DESIGN):
    return _BASE_LIFT_ARMATURE + design.knee.armature()


# --- yaw SEA belt series stiffness (TARGET yaw is servo+SEA) --------------------
# The co-design tags the belt with a DIMENSIONLESS Level-B stiffness k_tilde
# (TARGET.yaw.sea_k_tilde = 25). We map it to a physical rotary stiffness by
# calibrating the design point to the belt-leg resonance intent: a physical k of
# ~15 N·m/rad puts the resonance f = sqrt(k/I_eff)/2pi ~1.0 Hz near stride cadence,
# so the belt can store/return stride energy (Level B). That coincides with
# gen_mesh_robot_mjcf.SEA_YAW_STIFFNESS_EST = 15 N·m/rad (an INDEPENDENT bench
# estimate), so the two paths agree. Physical stiffness scales linearly with the
# dimensionless tag.
#
# MuJoCo realizes a soft <equality> as k = I_couple/tc^2 (dampratio 1), so tc is
# derived from k and the constraint's effective inertia. The 6 lb model uses an
# explicit estimate; bench-identify the real pulley and coupled inertia together.
_SEA_K_TILDE_REF = 25.0
_SEA_I_EFF_YAW = SEA_YAW_EFFECTIVE_INERTIA_EST


def sea_k_phys(k_tilde=_SEA_K_TILDE_REF):
    """Physical yaw SEA stiffness (N·m/rad) from the co-design's dimensionless tag."""
    return (float(k_tilde) / _SEA_K_TILDE_REF) * SEA_YAW_STIFFNESS_EST


# --- design defaults (the three approved changes) — SOURCED from robot_design ---
DEFAULTS = _DESIGN.geometry()


def stance_width(p=DEFAULTS):
    return 2 * (p["half_wid"] + p["yaw_lever"])   # foot-to-foot lateral span


def foot_nominal(L, p=DEFAULTS):
    """Nominal world foot position at zero yaw (stance)."""
    sx, sy = SIGN[L]
    return (sx * p["half_len"], sy * (p["half_wid"] + p["yaw_lever"]),
            0.0)


def build_walker(p=DEFAULTS, floor=True, sea=None):
    """Parametric walker MJCF. Actuation (per-axis gear + knee armature + yaw SEA)
    is sourced from the active robot_design (_DESIGN); geometry from `p`.

    sea: yaw series-elastic actuator. None -> auto (on iff the design's yaw is
    servo_sea, i.e. TARGET). When on, each yaw MOTOR drives a rotor DOF that is
    coupled to the hip_yaw joint by a soft equality (the rubber belt) — the
    series-elastic model ported from gen_mesh_robot_mjcf. Kinematic consumers
    Kinematic checks pass sea=False: the SEA is an actuation detail, not foot placement.
    """
    if sea is None:
        sea = (_DESIGN.yaw.kind == "servo_sea")
    hx, hy, lev, H = p["half_len"], p["half_wid"], p["yaw_lever"], p["stance_h"]
    shin = H - 0.02
    lift_arm = _lift_armature(_DESIGN)
    yaw_armature = _DESIGN.yaw.armature()
    pitch_armature = _DESIGN.pitch.armature()
    # per-axis forcerange from the spec: yaw/pitch torque (N·m), lift force (N)
    gy, gp, gk = _DESIGN.yaw.gear(), _DESIGN.pitch.gear(), _DESIGN.knee.gear()
    k_sea = sea_k_phys(_DESIGN.yaw.sea_k_tilde or _SEA_K_TILDE_REF)
    tc_sea = math.sqrt(_SEA_I_EFF_YAW / k_sea)   # solref timeconst (k = I_eff/tc^2, zeta=1)
    # Preserve the old walker's relative structural distribution while its total
    # per-leg structure is normalized by robot_design to the 6 lb budget.
    raw_parts = (0.30, 0.655, 0.655, 0.05)  # yaw link, upper, lower, foot
    yaw_structure, upper_structure, lower_structure, foot_structure = [
        LEG_MASS * x / sum(raw_parts) for x in raw_parts]
    legs, acts, rotors, eqs = [], [], [], []
    for L in LEGS:
        sx, sy = SIGN[L]
        mx, my = sx * hx, sy * hy
        legs.append(f'''
    <body name="{L}_yaw" pos="{mx:.4f} {my:.4f} 0">
      <joint name="{L}_hip_yaw" type="hinge" axis="0 0 1" range="-0.785 0.785"
             armature="{0.0 if sea else yaw_armature:.6g}" damping="0.08"/>
      <geom type="capsule" fromto="0 0 0 0 {sy*lev:.4f} 0" size="0.018"
            mass="{yaw_structure + SERVO_MASS:.9f}" contype="0" conaffinity="0"/>
      <body name="{L}_hip" pos="0 {sy*lev:.4f} 0">
        <joint name="{L}_pitch" type="hinge" axis="1 0 0" range="-0.5 0.5"
               armature="{pitch_armature:.6g}" damping="0.12"/>
        <geom type="capsule" fromto="0 0 0 0 0 {-shin*0.45:.3f}" size="0.015"
              mass="{upper_structure + SERVO_MASS:.9f}" contype="0" conaffinity="0"/>
        <body name="{L}_shin" pos="0 0 {-shin*0.45:.3f}">
          <joint name="{L}_lift" type="slide" axis="0 0 1"
                 range="0 {p['lift_range']:.4f}" armature="{lift_arm:.5g}" damping="1.5"/>
          <geom type="capsule" fromto="0 0 0 0 0 {-shin*0.55:.3f}" size="0.012"
                mass="{lower_structure:.9f}" contype="0" conaffinity="0"/>
          <geom name="{L}_foot" type="sphere" pos="0 0 {-shin*0.55:.3f}"
                size="{FOOT_R}" mass="{foot_structure:.9f}"/>
        </body>
      </body>
    </body>''')
        # yaw drives the rotor (motor side of the belt) when SEA is on
        yaw_joint = f"{L}_yaw_rotor" if sea else f"{L}_hip_yaw"
        acts.append(
            f'    <motor name="{L}_yaw_m" joint="{yaw_joint}" gear="{gy:.6g}" '
            f'forcerange="-{gy:.6g} {gy:.6g}" ctrlrange="-1 1"/>\n'
            f'    <motor name="{L}_pitch_m" joint="{L}_pitch" gear="{gp:.6g}" '
            f'forcerange="-{gp:.6g} {gp:.6g}" ctrlrange="-1 1"/>\n'
            f'    <motor name="{L}_lift_m" joint="{L}_lift" gear="{gk:.6g}" '
            f'forcerange="-{gk:.6g} {gk:.6g}" ctrlrange="-1 1"/>')
        if sea:
            # rotor: a flywheel on the yaw axis, coupled to hip_yaw by the belt.
            # RANGE matches hip_yaw so the RL env's range-based authority/clamp on
            # the (now rotor) actuated joint is unchanged; armature = the yaw servo
            # reflected inertia from the selected servo profile.
            rotors.append(f'''
    <body name="{L}_yaw_rotor_body" pos="{mx:.4f} {my:.4f} 0">
      <joint name="{L}_yaw_rotor" type="hinge" axis="0 0 1" range="-0.785 0.785"
             armature="{yaw_armature:.6g}" damping="0.01"/>
      <geom type="sphere" size="0.012" mass="{SERVO_MASS}" contype="0" conaffinity="0"
            rgba="0.9 0.6 0.2 0.5"/>
    </body>''')
            # belt = soft equality q_rotor = q_hip_yaw; softness (solref timeconst)
            # realizes k_sea N·m/rad over I_eff (Level-B SEA).
            eqs.append(f'    <joint name="{L}_yaw_sea" joint1="{L}_yaw_rotor" '
                       f'joint2="{L}_hip_yaw" polycoef="0 1 0 0 0" '
                       f'solref="{tc_sea:.4f} 1"/>')
    floor_xml = ('    <geom name="floor" type="plane" size="0 0 0.1" '
                 'rgba="0.4 0.5 0.4 1"/>\n') if floor else ""
    eq_xml = f'''  <equality>
{chr(10).join(eqs)}
  </equality>
''' if sea else ""
    return f'''<mujoco model="walker_improved">
  <compiler angle="radian" autolimits="true"/>
  <option timestep="0.004" integrator="implicitfast"/>
  <default><joint damping="0.02"/>
    <geom friction="1.2 0.1 0.1" contype="1" conaffinity="1"/></default>
  <worldbody>
{floor_xml}    <body name="torso" pos="0 0 {H:.3f}">
      <freejoint/>
      <geom name="torso" type="box" size="{hx+0.04:.3f} {hy+0.03:.3f} 0.035"
            mass="{TORSO_MASS + (0 if sea else 4 * SERVO_MASS):.9f}" rgba="0.3 0.3 0.6 1"/>
{''.join(legs)}{''.join(rotors)}
    </body>
  </worldbody>
{eq_xml}  <actuator>
{chr(10).join(acts)}
  </actuator>
</mujoco>'''


if __name__ == "__main__":
    import mujoco
    p = DEFAULTS
    print(f"stance: {2*p['half_len']*1000:.0f}mm long x {stance_width()*1000:.0f}mm wide "
          f"(was 200x162); yaw lever {p['yaw_lever']*1000:.0f}mm (was 31); "
          f"lift range {p['lift_range']*1000:.0f}mm, nom {p['lift_nom']*1000:.0f}mm")
    m = mujoco.MjModel.from_xml_string(build_walker())
    print(f"model: nq={m.nq} nu={m.nu} mass={m.body_mass.sum():.2f}kg")
    # yaw-stride check: how far does a foot move fore-aft per yaw sweep now?
    d = mujoco.MjData(m)
    fg = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_GEOM, "FL_foot")
    ya = m.jnt_qposadr[mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_JOINT, "FL_hip_yaw")]
    xs = []
    for yaw in (-30, 0, 30):
        d.qpos[:] = 0
        d.qpos[2] = p["stance_h"]
        d.qpos[ya] = math.radians(yaw)
        mujoco.mj_forward(m, d)
        xs.append(d.geom_xpos[fg][0])
    print(f"YAW STRIDE: foot fore-aft travel over ±30° yaw = {(xs[2]-xs[0])*1000:+.0f}mm "
          f"(was +0mm / 31mm lateral) — yaw now strides FORE-AFT")
