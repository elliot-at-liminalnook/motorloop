#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
"""Independent Drake build of the parametric quadruped: robot.toml -> MultibodyPlant.

DR.1 (notes/uplift-execution-plan.md, Phase DR / V.8): a SECOND engine whose model
is derived independently from the same spec, so the two compilations can referee
each other (test_cross_engine.py). A single simulator can never disagree with
itself — the pre-2026-07 gear bug survived precisely because every check consulted
the one model (or code shared with its generator). This module breaks that shared
fate.

INDEPENDENCE RULES (load-bearing; violating them recreates shared fate):
  1. NEVER read model.xml — the plant is built programmatically from robot.toml.
  2. NEVER import `joint_torque_limit` or any physical derivation from
     gen_robot_mjcf.py. Torque is RE-DERIVED here from sim/tests/motors.py
     primitives (kt x peak_factor x rated_current_a x gear); inertias are
     re-assembled from Drake's own solid-geometry helpers. Sharing the raw
     tomllib load (`gen_robot_mjcf.load_spec`) is allowed — sharing
     *interpretation* is what recreates shared fate.

Scope (DR.1): contact-free quantities only — bodies, joints, limits, actuators,
masses, inertias. Striker, parallel springs, and contact tuning are deliberately
NOT modelled: DR.2 asserts only mass / kinematics / actuation / gravity-statics,
per the plan's contact-adjudication kill criterion.

Structure per robot.toml (mirrors the MJCF topology, independently expressed):
  floating-base box torso; per leg: hip capsule (abd hinge about x at the leg
  attachment), thigh capsule (flex hinge about y at the lateral hip offset),
  calf capsule (knee hinge about y at the thigh end), foot sphere WELDED at the
  calf end. One JointActuator per hinge with the independently derived effort
  limit.

  ../../.venv-sim/bin/python gen_robot_drake.py --config robot.toml --summary
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parents[1] / "sim" / "tests"))

from motors import MOTORS  # noqa: E402  (the shared PRIMITIVES — bench-profiled motors)
from gen_robot_mjcf import load_spec  # noqa: E402  (raw tomllib load ONLY — allowed share)

from pydrake.math import RigidTransform  # noqa: E402
from pydrake.multibody.plant import MultibodyPlant  # noqa: E402
from pydrake.multibody.tree import (  # noqa: E402
    FixedOffsetFrame, JointActuatorIndex, RevoluteJoint, RigidBody,
    RotationalInertia, SpatialInertia, UnitInertia)

# The MJCF generator hard-codes these two masses as geom attributes rather than
# spec fields. They are RE-STATED here as named constants (not imported): if the
# two builders ever drift, DR.2's 1% total-mass contract fires — that loudness is
# the point of stating them twice.
HIP_MASS_KG = 0.3
FOOT_MASS_KG = 0.05


def joint_effort_limit_nm(spec: dict) -> float:
    """Independent re-derivation of the per-hinge torque limit.

    tau = kt x peak_factor x rated_current_a x gear, straight from the motors.py
    primitives — deliberately NOT gen_robot_mjcf.joint_torque_limit (rule 2).
    """
    a = spec["actuator"]
    m = MOTORS[a["motor"]]
    return m.kt * float(a["peak_factor"]) * m.rated_current_a * float(a["gear"])


def _spatial_inertia(mass: float, p_BoBcm, G_cm: UnitInertia) -> SpatialInertia:
    """SpatialInertia about the body origin from a CENTRAL unit inertia.

    G_cm is the unit inertia of the solid about its own centroid (Drake helper);
    p_BoBcm places that centroid in the body frame (whose origin sits at the
    inboard joint, matching the MJCF body-frame convention).
    """
    full = float(mass) * G_cm.CopyToFullMatrix3()
    I_cm = RotationalInertia(full[0, 0], full[1, 1], full[2, 2],
                             full[0, 1], full[0, 2], full[1, 2])
    return SpatialInertia.MakeFromCentralInertia(
        mass=float(mass), p_PScm_E=np.asarray(p_BoBcm, dtype=float), I_SScm_E=I_cm)


def _add_leg(plant: MultibodyPlant, torso: RigidBody, leg: dict, d: dict,
             tau: float) -> None:
    """One 3-DOF mammalian leg chain: abd(x) -> flex(y) -> knee(y) -> welded foot."""
    n = leg["name"]
    px, py, pz = (float(v) for v in leg["pos"])
    sy = 1.0 if py >= 0 else -1.0                       # abduction stand-off direction
    r = float(d["link_radius"])
    tl, cl, ho = float(d["thigh_len"]), float(d["calf_len"]), float(d["hip_offset"])
    damping = float(d["joint_damping"])

    def _hinge(name: str, parent_frame, child: RigidBody, axis, rng, default: float):
        joint = plant.AddJoint(RevoluteJoint(
            name, parent_frame, child.body_frame(), np.asarray(axis, dtype=float),
            damping=damping))
        joint.set_position_limits(np.array([float(rng[0])]), np.array([float(rng[1])]))
        joint.set_default_angle(float(default))
        plant.AddJointActuator(f"{name}_m", joint, effort_limit=tau)

    # hip: capsule along +/-y from the joint to the lateral stand-off
    hip = plant.AddRigidBody(f"{n}_hip", _spatial_inertia(
        HIP_MASS_KG, [0.0, sy * ho / 2.0, 0.0], UnitInertia.SolidCapsule(r, ho, [0.0, 1.0, 0.0])))
    hip_frame = plant.AddFrame(FixedOffsetFrame(
        f"{n}_abd_parent", torso.body_frame(), RigidTransform([px, py, pz])))
    _hinge(f"{n}_abd", hip_frame, hip, [1.0, 0.0, 0.0],
           d["abd_range"], d.get("stand_abd", 0.0))

    # thigh: capsule straight down from the hip-flex joint
    thigh = plant.AddRigidBody(f"{n}_thigh", _spatial_inertia(
        d["thigh_mass"], [0.0, 0.0, -tl / 2.0], UnitInertia.SolidCapsule(r, tl, [0.0, 0.0, 1.0])))
    thigh_frame = plant.AddFrame(FixedOffsetFrame(
        f"{n}_flex_parent", hip.body_frame(), RigidTransform([0.0, sy * ho, 0.0])))
    _hinge(f"{n}_flex", thigh_frame, thigh, [0.0, 1.0, 0.0],
           d["flex_range"], d.get("stand_flex", -0.4))

    # calf: capsule straight down from the knee
    calf = plant.AddRigidBody(f"{n}_calf", _spatial_inertia(
        d["calf_mass"], [0.0, 0.0, -cl / 2.0], UnitInertia.SolidCapsule(r, cl, [0.0, 0.0, 1.0])))
    calf_frame = plant.AddFrame(FixedOffsetFrame(
        f"{n}_knee_parent", thigh.body_frame(), RigidTransform([0.0, 0.0, -tl])))
    _hinge(f"{n}_knee", calf_frame, calf, [0.0, 1.0, 0.0],
           d["knee_range"], d.get("stand_knee", -1.1))

    # foot: sphere welded at the calf end (a mass point for statics; contact is
    # out of DR.1 scope)
    foot = plant.AddRigidBody(f"{n}_foot", _spatial_inertia(
        FOOT_MASS_KG, [0.0, 0.0, 0.0], UnitInertia.SolidSphere(float(d["foot_radius"]))))
    plant.WeldFrames(calf.body_frame(), foot.body_frame(),
                     RigidTransform([0.0, 0.0, -cl]))


def build_plant(spec: dict, weld_base: bool = False,
                time_step: float = 0.0) -> MultibodyPlant:
    """robot.toml spec dict -> finalized MultibodyPlant.

    weld_base=True fixes the torso to the world at spawn height (the DR.2
    gravity-statics variant); default is a floating base. Default configuration
    is the canonical stance (leg_defaults stand_*) at torso spawn_height.
    """
    plant = MultibodyPlant(time_step=time_step)
    t, d = spec["torso"], spec["leg_defaults"]
    tau = joint_effort_limit_nm(spec)
    hx, hy, hz = (float(v) for v in t["half_extents"])

    torso = plant.AddRigidBody("torso", _spatial_inertia(
        t["mass"], [0.0, 0.0, 0.0], UnitInertia.SolidBox(2 * hx, 2 * hy, 2 * hz)))
    for leg in spec["leg"]:
        _add_leg(plant, torso, leg, d, tau)

    X_W_torso = RigidTransform([0.0, 0.0, float(t["spawn_height"])])
    if weld_base:
        plant.WeldFrames(plant.world_frame(), torso.body_frame(), X_W_torso)
    else:
        plant.SetDefaultFloatingBaseBodyPose(torso, X_W_torso)
    plant.Finalize()
    return plant


def hinge_effort_limits(plant: MultibodyPlant) -> dict[str, float]:
    """joint name -> actuator effort limit (N·m), for every actuated hinge."""
    out: dict[str, float] = {}
    for i in range(plant.num_actuators()):
        act = plant.get_joint_actuator(JointActuatorIndex(i))
        out[act.joint().name()] = float(act.effort_limit())
    return out


def summarize(spec: dict) -> str:
    plant = build_plant(spec)
    context = plant.CreateDefaultContext()
    tau = joint_effort_limit_nm(spec)
    total_mass = plant.CalcTotalMass(context)
    com = plant.CalcCenterOfMassPositionInWorld(context)

    # DR.1 done-when: mass/CoM asserted against the spec, not eyeballed.
    d, legs = spec["leg_defaults"], spec["leg"]
    spec_mass = float(spec["torso"]["mass"]) + len(legs) * (
        HIP_MASS_KG + float(d["thigh_mass"]) + float(d["calf_mass"]) + FOOT_MASS_KG)
    assert abs(total_mass - spec_mass) <= 0.01 * spec_mass, (
        f"plant mass {total_mass:.4f} kg != spec sum {spec_mass:.4f} kg (>1%)")
    assert com[2] < float(spec["torso"]["spawn_height"]), (
        "CoM at stance must sit below the torso (legs hang down)")

    lines = [
        f"{spec['meta']['name']} (Drake {'.'.join(map(str, _drake_version()))}, "
        f"independent build from robot.toml)",
        f"  DoF: nq={plant.num_positions()} nv={plant.num_velocities()} "
        f"(floating base 6 + {plant.num_actuators()} hinges), "
        f"actuators={plant.num_actuators()}",
        f"  derived effort limit: kt x peak_factor x rated_current x gear = "
        f"{tau:.4f} N·m",
    ]
    for name, lim in hinge_effort_limits(plant).items():
        lines.append(f"    {name:<10s} effort limit {lim:.4f} N·m")
    lines += [
        f"  total mass: {total_mass:.4f} kg (spec sum {spec_mass:.4f} kg)",
        f"  CoM at default config (stance, torso z={spec['torso']['spawn_height']}): "
        f"[{com[0]:.4f}, {com[1]:.4f}, {com[2]:.4f}] m -> height {com[2]:.4f} m",
    ]
    return "\n".join(lines)


def _drake_version() -> tuple:
    try:
        from pydrake.common import __version__ as v  # type: ignore
        return tuple(v.split(".")[:3])
    except Exception:
        import importlib.metadata
        try:
            return tuple(importlib.metadata.version("drake").split(".")[:3])
        except Exception:
            return ("?",)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--config", default=str(HERE / "robot.toml"))
    ap.add_argument("--summary", action="store_true",
                    help="print DoF count, per-joint effort limits, total mass, "
                         "and CoM height at the default (stance) configuration")
    args = ap.parse_args()
    spec = load_spec(args.config)
    if args.summary:
        print(summarize(spec))
    else:
        plant = build_plant(spec)
        print(f"{spec['meta']['name']}: Drake plant finalized — "
              f"nq={plant.num_positions()} nv={plant.num_velocities()} "
              f"nu={plant.num_actuators()}, effort limit "
              f"{joint_effort_limit_nm(spec):.4f} N·m per hinge")


if __name__ == "__main__":
    main()
