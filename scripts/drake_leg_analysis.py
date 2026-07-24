#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
"""Drake multibody analysis of the 7-1-2026 robotic foot (leg swing + knee slider-crank).

Run with the pinned venv:
  build/foot_rig/drake/.venv/bin/python scripts/drake_leg_analysis.py

Model matches scripts/rig_foot_7_1.py exactly:
  world -> (weld) base housing
        -> [leg_swing]  revolute Z at the gear axle pin (-0.16269, 0, 0), +/-25 deg,
                        worm-driven 20:1
        -> [knee]       revolute Z at the knee pin (leg frame +0.16269 X), -90..+10 deg,
                        rotates the blade UPPER length (crank, r = 75 mm)
        -> [toe hinge]  revolute Z at (0, +0.075) on the crank -> blade LOWER length
                        (conrod, L = 100 mm)
        -> [piston]     prismatic along leg-local Y through the distal bushing
  loop closure: ball constraint pinning the conrod's heel ear to the piston pin.

Masses are the physics.json placeholders; COMs/inertias are thin-rod/box estimates.
Gravity acts along world -Y (the leg hangs down -Y as modeled).

Analyses:
  1. Piston kinematics + mechanical advantage across the knee ROM (closed form,
     cross-checked against Drake forward kinematics through the ball constraint).
  2. Static holding torques via dV/dq using plant.CalcPotentialEnergy:
     knee motor vs blade angle; hip (swing) joint + worm-side torque vs swing angle.
  3. Passive-drop SAP simulation with joint limits: release the blade near vertical,
     confirm it sweeps out front and the heel pin lowers by weight to the -90 deg stop.
"""

from __future__ import annotations

import json
import math
from pathlib import Path

import numpy as np
from pydrake.math import RigidTransform
from pydrake.multibody.plant import DiscreteContactApproximation
from pydrake.multibody.tree import (
    FixedOffsetFrame,
    PrismaticJoint,
    RevoluteJoint,
    RotationalInertia,
    SpatialInertia,
)
from pydrake.systems.analysis import Simulator
from pydrake.systems.framework import DiagramBuilder
from pydrake.multibody.plant import AddMultibodyPlantSceneGraph

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "build" / "foot_rig" / "drake"
OUT.mkdir(parents=True, exist_ok=True)

# ---- measured geometry (m), identical to rig_foot_7_1.py ----------------------
SWING_PIVOT_W = np.array([-0.16269, 0.0, 0.0])
KNEE_IN_LEG = np.array([0.16269, 0.0, 0.0])       # knee pin in the leg-link frame
CRANK_R = 0.075                                    # knee pin -> toe hinge
CONROD_L = 0.100                                   # toe hinge -> heel pin
HEEL_NEUTRAL_IN_LEG = np.array([0.16269, -0.025, 0.0])
SWING_LIM = math.radians(25.0)
KNEE_LIM = (math.radians(-90.0), math.radians(10.0))
WORM_RATIO = 20.0

# ---- placeholder mass properties (physics.json) --------------------------------
# (mass kg, COM in body frame, thin-rod/box half-lengths for inertia estimate)
LEG_MASS, LEG_COM = 0.55, np.array([0.09, -0.06, 0.0])       # rails+carriers+shin, spread out
UPPER_MASS, UPPER_COM = 0.05, np.array([0.0, 0.0375, 0.0])   # strip knee->toe
LOWER_MASS, LOWER_COM = 0.07, np.array([0.012, -0.05, 0.0])  # curved plates toe->heel
ROD_MASS, ROD_COM = 0.08, np.array([0.0, -0.13, 0.0])        # rod hangs below the pin
G = 9.81


def rod_inertia(mass: float, length: float, axis: str) -> RotationalInertia:
    """Thin-rod rotational inertia about the COM, rod along `axis`."""
    i = mass * length * length / 12.0
    eps = max(i * 0.01, 1e-7)
    if axis == "y":
        return RotationalInertia(i, eps, i)
    return RotationalInertia(eps, i, i)


def spatial(mass: float, com: np.ndarray, length: float, axis: str) -> SpatialInertia:
    return SpatialInertia.MakeFromCentralInertia(mass, com, rod_inertia(mass, length, axis))


def build_plant(time_step: float, with_loop: bool, lock_swing: bool):
    """Build the mechanism. Continuous (time_step=0) plants skip the ball constraint
    and are posed analytically; discrete plants get SAP + the loop constraint."""
    builder = DiagramBuilder()
    plant, _ = AddMultibodyPlantSceneGraph(builder, time_step=time_step)
    if time_step > 0:
        plant.set_discrete_contact_approximation(DiscreteContactApproximation.kSap)
    plant.mutable_gravity_field().set_gravity_vector([0.0, -G, 0.0])

    leg = plant.AddRigidBody("leg_swing_link", spatial(LEG_MASS, LEG_COM, 0.30, "x"))
    upper = plant.AddRigidBody("blade_upper", spatial(UPPER_MASS, UPPER_COM, CRANK_R, "y"))
    lower = plant.AddRigidBody("blade_lower", spatial(LOWER_MASS, LOWER_COM, CONROD_L, "y"))
    rod = plant.AddRigidBody("pushrod", spatial(ROD_MASS, ROD_COM, 0.26, "y"))

    z = [0.0, 0.0, 1.0]
    f_swing = plant.AddFrame(FixedOffsetFrame(
        "swing_anchor", plant.world_frame(), RigidTransform(SWING_PIVOT_W)))
    if lock_swing:
        plant.WeldFrames(f_swing, leg.body_frame())
        swing = None
    else:
        swing = plant.AddJoint(RevoluteJoint(
            "leg_swing", f_swing, leg.body_frame(), z, damping=0.02))
        swing.set_position_limits([-SWING_LIM], [SWING_LIM])

    f_knee = plant.AddFrame(FixedOffsetFrame(
        "knee_anchor", leg.body_frame(), RigidTransform(KNEE_IN_LEG)))
    knee = plant.AddJoint(RevoluteJoint(
        "knee_blade", f_knee, upper.body_frame(), z, damping=0.01))
    knee.set_position_limits([KNEE_LIM[0]], [KNEE_LIM[1]])

    f_toe = plant.AddFrame(FixedOffsetFrame(
        "toe_hinge_anchor", upper.body_frame(), RigidTransform([0.0, CRANK_R, 0.0])))
    toe = plant.AddJoint(RevoluteJoint(
        "toe_hinge", f_toe, lower.body_frame(), z, damping=0.005))

    f_bush = plant.AddFrame(FixedOffsetFrame(
        "bushing_anchor", leg.body_frame(), RigidTransform(HEEL_NEUTRAL_IN_LEG)))
    piston = plant.AddJoint(PrismaticJoint(
        "piston", f_bush, rod.body_frame(), [0.0, 1.0, 0.0], damping=0.05))

    if with_loop:
        plant.AddBallConstraint(lower, np.array([0.0, -CONROD_L, 0.0]), rod, np.zeros(3))

    plant.Finalize()
    diagram = builder.Build()
    joints = {"swing": swing, "knee": knee, "toe": toe, "piston": piston}
    return diagram, plant, joints


# ---- closed-form slider-crank (identical to the rig script) --------------------
def heel_y(phi: float) -> float:
    tx, ty = -CRANK_R * math.sin(phi), CRANK_R * math.cos(phi)
    return ty - math.sqrt(CONROD_L**2 - tx**2)


def conrod_angle(phi: float) -> float:
    tx, ty = -CRANK_R * math.sin(phi), CRANK_R * math.cos(phi)
    dx, dy = 0.0 - tx, heel_y(phi) - ty
    return math.atan2(dx, -dy)


def pose_context(plant, joints, context, sigma: float, phi: float) -> None:
    """Set all four joint coordinates consistently with the loop closure."""
    if joints["swing"] is not None:
        joints["swing"].set_angle(context, sigma)
    joints["knee"].set_angle(context, phi)
    # toe-hinge coordinate is the conrod angle RELATIVE to the crank
    joints["toe"].set_angle(context, conrod_angle(phi) - phi)
    joints["piston"].set_translation(context, heel_y(phi) - HEEL_NEUTRAL_IN_LEG[1])


def main() -> None:
    results: dict = {}

    # ---------- kinematic cross-check: Drake FK vs closed form ----------
    diagram, plant, joints = build_plant(0.0, with_loop=False, lock_swing=False)
    ctx = diagram.CreateDefaultContext()
    pctx = plant.GetMyContextFromRoot(ctx)
    lower_body = plant.GetBodyByName("blade_lower")
    rod_body = plant.GetBodyByName("pushrod")
    worst_gap = 0.0
    for phi_deg in np.linspace(-90, 10, 51):
        pose_context(plant, joints, pctx, 0.0, math.radians(phi_deg))
        p_heel = plant.EvalBodyPoseInWorld(pctx, lower_body) @ np.array([0.0, -CONROD_L, 0.0])
        p_pin = plant.EvalBodyPoseInWorld(pctx, rod_body).translation()
        worst_gap = max(worst_gap, float(np.linalg.norm(p_heel - p_pin)))
    results["fk_worst_loop_gap_m"] = worst_gap
    print(f"[check] worst loop-closure gap across ROM (Drake FK): {worst_gap:.2e} m")
    assert worst_gap < 1e-9, "loop closure inconsistent"

    # ---------- 1. piston kinematics + mechanical advantage ----------
    phis = np.linspace(math.radians(-90), math.radians(10), 201)
    drop_mm = np.array([1000.0 * (heel_y(p) - heel_y(0.0)) for p in phis])
    dh_dphi = np.gradient([heel_y(p) for p in phis], phis)  # m/rad
    results["piston_drop_mm_at_-90"] = float(drop_mm[0])
    results["dh_dphi_max_m_per_rad"] = float(np.max(np.abs(dh_dphi)))

    # ---------- 2. static holding torques via dV/dq ----------
    def potential(sigma: float, phi: float) -> float:
        pose_context(plant, joints, pctx, sigma, phi)
        return plant.CalcPotentialEnergy(pctx)

    eps = 1e-5
    knee_tau = np.array([(potential(0.0, p + eps) - potential(0.0, p - eps)) / (2 * eps)
                         for p in phis])
    results["knee_holding_tau_max_nm"] = float(np.max(np.abs(knee_tau)))
    results["knee_holding_tau_at_-90_nm"] = float(knee_tau[0])

    sigmas = np.linspace(-SWING_LIM, SWING_LIM, 101)
    swing_curves = {}
    for phi_deg in (10.0, -40.0, -90.0):
        p = math.radians(phi_deg)
        tau = np.array([(potential(s + eps, p) - potential(s - eps, p)) / (2 * eps)
                        for s in sigmas])
        swing_curves[phi_deg] = tau
    tau_all = np.concatenate(list(swing_curves.values()))
    results["swing_holding_tau_max_nm"] = float(np.max(np.abs(tau_all)))
    results["worm_side_tau_max_nm"] = float(np.max(np.abs(tau_all)) / WORM_RATIO)

    # ---------- 3. passive-drop SAP simulation (loop constraint active) ----------
    diagram_d, plant_d, joints_d = build_plant(1e-3, with_loop=True, lock_swing=True)
    sim = Simulator(diagram_d)
    ctx_d = sim.get_mutable_context()
    pctx_d = plant_d.GetMyContextFromRoot(ctx_d)
    pose_context(plant_d, joints_d, pctx_d, 0.0, math.radians(-5.0))  # released near vertical
    lower_d = plant_d.GetBodyByName("blade_lower")
    rod_d = plant_d.GetBodyByName("pushrod")
    t_log, phi_log, drop_log, gap_log = [], [], [], []
    sim.Initialize()
    for t in np.arange(0.0, 2.5001, 0.01):
        sim.AdvanceTo(t)
        phi = joints_d["knee"].get_angle(pctx_d)
        p_heel = plant_d.EvalBodyPoseInWorld(pctx_d, lower_d) @ np.array([0.0, -CONROD_L, 0.0])
        p_pin = plant_d.EvalBodyPoseInWorld(pctx_d, rod_d).translation()
        t_log.append(t)
        phi_log.append(math.degrees(phi))
        drop_log.append(1000.0 * (joints_d["piston"].get_translation(pctx_d)
                                  + HEEL_NEUTRAL_IN_LEG[1] - heel_y(0.0)))
        gap_log.append(1000.0 * float(np.linalg.norm(p_heel - p_pin)))
    results["drop_settle_deg"] = phi_log[-1]
    results["drop_settle_mm"] = drop_log[-1]
    results["sim_worst_loop_gap_mm"] = float(np.max(gap_log))
    print(f"[sim] settled at knee {phi_log[-1]:+.1f} deg, heel drop {drop_log[-1]:+.1f} mm, "
          f"worst loop gap {np.max(gap_log):.3f} mm")

    # ---------- plots (dataviz conventions: single axis, thin marks, direct labels) ----------
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    SURF, INK, INK2, GRID = "#fcfcfb", "#0b0b0b", "#52514e", "#e8e8e6"
    BLUE, AQUA, VIOLET, RED = "#2a78d6", "#1baf7a", "#4a3aa7", "#e34948"
    plt.rcParams.update({
        "figure.facecolor": SURF, "axes.facecolor": SURF, "savefig.facecolor": SURF,
        "text.color": INK, "axes.edgecolor": GRID, "axes.labelcolor": INK2,
        "xtick.color": INK2, "ytick.color": INK2, "axes.grid": True,
        "grid.color": GRID, "grid.linewidth": 0.6, "axes.spines.top": False,
        "axes.spines.right": False, "font.size": 11, "axes.titlesize": 12,
    })
    phis_deg = np.degrees(phis)

    # fig 1: piston kinematics (two panels — different units, never dual-axis)
    fig, (a1, a2) = plt.subplots(2, 1, figsize=(7.2, 6.4), sharex=True)
    a1.plot(phis_deg, drop_mm, color=BLUE, lw=2)
    a1.set_ylabel("heel-pin drop (mm)")
    a1.set_title("Piston kinematics across the knee ROM (slider-crank closure)")
    a1.annotate(f"{drop_mm[0]:.1f} mm at -90°", (phis_deg[0], drop_mm[0]),
                textcoords="offset points", xytext=(8, -12), color=INK2)
    a2.plot(phis_deg, 1000.0 * dh_dphi, color=AQUA, lw=2)
    a2.set_ylabel("dh/dφ (mm per rad)")
    a2.set_xlabel("knee blade angle φ (deg; − = toe out front)")
    a2.annotate("top dead center: dh/dφ = 0\n(infinite mech. advantage)",
                (0.0, 0.0), textcoords="offset points", xytext=(-170, 40), color=INK2,
                arrowprops={"arrowstyle": "-", "color": GRID})
    a2.axvline(0.0, color=GRID, lw=1)
    fig.tight_layout()
    fig.savefig(OUT / "piston_kinematics.png", dpi=130)

    # fig 2: knee holding torque
    fig, ax = plt.subplots(figsize=(7.2, 4.2))
    ax.plot(phis_deg, 1000.0 * knee_tau, color=BLUE, lw=2)
    ax.set_xlabel("knee blade angle φ (deg)")
    ax.set_ylabel("holding torque (mN·m)")
    ax.set_title("Knee-motor torque to hold the blade static under gravity")
    imax = int(np.argmax(np.abs(knee_tau)))
    ax.annotate(f"peak {1000.0 * knee_tau[imax]:+.1f} mN·m at {phis_deg[imax]:.0f}°",
                (phis_deg[imax], 1000.0 * knee_tau[imax]),
                textcoords="offset points", xytext=(14, -6), color=INK2)
    fig.tight_layout()
    fig.savefig(OUT / "knee_holding_torque.png", dpi=130)

    # fig 3: swing holding torque, three blade poses + worm-side scale note
    fig, ax = plt.subplots(figsize=(7.2, 4.4))
    for (phi_deg, tau), color in zip(swing_curves.items(), (BLUE, AQUA, VIOLET)):
        ax.plot(np.degrees(sigmas), tau, color=color, lw=2, label=f"blade at {phi_deg:+.0f}°")
        ax.annotate(f"{phi_deg:+.0f}°", (np.degrees(sigmas)[-1], tau[-1]),
                    textcoords="offset points", xytext=(4, -3), color=color)
    ax.set_xlabel("leg swing σ (deg)")
    ax.set_ylabel("holding torque at swing joint (N·m)")
    ax.set_title("Hip holding torque vs swing angle (worm sees 1/20th of this)")
    ax.legend(frameon=False, loc="upper left")
    fig.tight_layout()
    fig.savefig(OUT / "swing_holding_torque.png", dpi=130)

    # fig 4: passive drop simulation (two panels)
    fig, (a1, a2) = plt.subplots(2, 1, figsize=(7.2, 6.0), sharex=True)
    a1.plot(t_log, phi_log, color=BLUE, lw=2)
    a1.axhline(-90, color=RED, lw=1, ls="--")
    a1.annotate("-90° joint stop", (t_log[-1], -90), textcoords="offset points",
                xytext=(-80, 6), color=RED)
    a1.set_ylabel("knee blade angle (deg)")
    a1.set_title("Passive drop from -5°: blade sweeps out front by weight (SAP, loop closed)")
    a2.plot(t_log, drop_log, color=AQUA, lw=2)
    a2.set_ylabel("heel-pin drop (mm)")
    a2.set_xlabel("time (s)")
    fig.tight_layout()
    fig.savefig(OUT / "passive_drop_sim.png", dpi=130)

    (OUT / "results.json").write_text(json.dumps(results, indent=2) + "\n")
    print(json.dumps(results, indent=2))
    print("PLOTS:", *[str(OUT / n) for n in
                      ("piston_kinematics.png", "knee_holding_torque.png",
                       "swing_holding_torque.png", "passive_drop_sim.png")], sep="\n  ")


if __name__ == "__main__":
    main()
