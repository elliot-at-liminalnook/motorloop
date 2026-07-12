# SPDX-License-Identifier: MIT
"""robot_design.py — the ONE source of truth for the walker's design vector.

Every model (RL env, MuJoCo checks, co-design, reference gait) must read the geometry AND
actuation from here so they cannot silently drift — the same single-source discipline
`constants.py` (V.1) and the pydantic `RobotSpec` (V.2) applied to the parametric quad,
now extended to cover the ACTUATOR/transmission design vector and the co-design loop.

Two named designs share the selected hardware:
  * CURRENT  — CAD-derived transmission geometry (belt 6, worm 20, direct knee).
  * TARGET   — the improved walking geometry with the same twelve ST3215-HS
               servos. The optional yaw SEA changes compliance, not motor choice.

Both designs carry the full 6 lb mass envelope. `test_design_alignment.py` asserts
that the compiled RL model uses this actuator BOM and stays within that limit.
"""
from __future__ import annotations

from dataclasses import dataclass
import math

# Low-level hardware and mass-budget constants are shared with the CAD-derived
# generator so training and mechanism validation cannot describe different robots.
from gen_mesh_robot_mjcf import (BELT_RATIO, LEG_STRUCTURE_MASS,  # noqa: E402
                                 MAX_ROBOT_MASS_KG,
                                 SERVO_FREE_RAD_S, SERVO_MASS,
                                 SERVO_OUTPUT_INERTIA_KG_M2_EST,
                                 SERVO_STALL_NM, STRIKER_MASS,
                                 TORSO_STRUCTURE_MASS, WORM_RATIO)

V_BUS = 48.0
LIFT_CRANK = 0.04            # m, rotary->foot-vertical crank radius (lift axis)

# --- shared mass constants ----------------------------------------------------
# The walker omits the cosmetic striker geom, so its mass is lumped into the torso
# to retain the conservative full-robot 6 lb gravity/inertia envelope.
TORSO_MASS = TORSO_STRUCTURE_MASS + STRIKER_MASS
LEG_MASS = LEG_STRUCTURE_MASS
ROBOT_MASS_LIMIT_KG = MAX_ROBOT_MASS_KG

# --- motor catalog: the single source consumed by codesign_* too --------------
# servos: (stall N·m, no-load rad/s, mass kg, cost $, output inertia estimate kg·m²)
SERVOS = {
    "waveshare_st3215_hs": (SERVO_STALL_NM, SERVO_FREE_RAD_S, SERVO_MASS,
                             27.81, SERVO_OUTPUT_INERTIA_KG_M2_EST),
    # Historical comparison candidates; neither is used by CURRENT or TARGET.
    "gobilda_25_3": (1.059, 15.18, 0.060, 50, 2.73375e-3),
    "reefs_400sc":  (2.86, (math.pi / 3) / 0.074, 0.060, 50, 2.7e-3),
    "savox_2274":   (2.45, (math.pi / 3) / 0.080, 0.060, 55, 2.7e-3),
}
# BLDCs: (Kv, I_cont A, I_peak A, mass kg, cost $)  — Kt = 9.55/Kv
BLDCS = {
    "odrive_d5312s": (330, 30, 60, 0.25, 129),
    "mjbots_mj5208": (330, 18, 58, 0.19, 190),
    "odrive_d6374":  (150, 50, 90, 0.80, 119),
}


def kt(kv):
    return 9.55 / kv


@dataclass(frozen=True)
class Axis:
    """One leg axis: actuator kind + part + reduction (+ SEA stiffness on yaw).
    `linear` axes (the lift/knee, via a crank) report force+lin-speed; rotary axes
    report torque+ang-speed. `gear`/`wfree` are what the RL env consumes (the
    forcerange multiplier and the torque-speed derating no-load speed)."""
    kind: str                 # "servo" | "servo_sea" | "bldc"
    part: str                 # key into SERVOS or BLDCS
    reduction: float
    linear: bool = False      # True for the lift/knee (prismatic via crank)
    sea_k_tilde: float | None = None    # Level-B dimensionless spring stiffness (yaw)

    def _rotary_stall_free(self):
        if self.kind.startswith("servo"):
            stall, free = SERVOS[self.part][:2]
            return stall * self.reduction, free / self.reduction
        kv, ic, ip, _, _ = BLDCS[self.part]
        cont = 0.9 * kt(kv) * ic * self.reduction        # continuous joint torque
        free = (kv * V_BUS * 2 * math.pi / 60) / self.reduction
        return cont, free

    def gear(self) -> float:
        """RL-env forcerange: joint torque (N·m) or, for a linear axis, force (N)."""
        t, _ = self._rotary_stall_free()
        return t / LIFT_CRANK if self.linear else t

    def peak(self) -> float:
        if self.kind == "bldc":
            kv, ic, ip, _, _ = BLDCS[self.part]
            t = 0.9 * kt(kv) * ip * self.reduction
        else:
            t = SERVOS[self.part][0] * self.reduction
        return t / LIFT_CRANK if self.linear else t

    def wfree(self) -> float:
        """RL-env derating no-load speed: rad/s (rotary) or m/s (linear)."""
        _, f = self._rotary_stall_free()
        return f * LIFT_CRANK if self.linear else f

    def mass(self) -> float:
        return SERVOS[self.part][2] if self.kind.startswith("servo") else BLDCS[self.part][3]

    def armature(self) -> float:
        """Reflected actuator inertia at this joint (kg·m² or kg for a slide)."""
        if self.kind.startswith("servo"):
            rotor = SERVOS[self.part][4]
        else:
            rotor = 1.5e-5       # 5312-class estimate; historical BLDC studies only
        reflected = rotor * self.reduction ** 2
        return reflected / LIFT_CRANK ** 2 if self.linear else reflected


@dataclass(frozen=True)
class RobotDesign:
    name: str
    # geometry (the walker_improved vector)
    half_len: float
    half_wid: float
    yaw_lever: float
    stance_h: float
    lift_range: float
    lift_nom: float
    # per-axis actuation
    yaw: Axis
    pitch: Axis
    knee: Axis
    torso_mass: float = TORSO_MASS

    def geometry(self) -> dict:
        """Geometry dict for walker_improved.build_walker."""
        return dict(half_len=self.half_len, half_wid=self.half_wid,
                    yaw_lever=self.yaw_lever, stance_h=self.stance_h,
                    lift_range=self.lift_range, lift_nom=self.lift_nom)

    def gears(self) -> list[float]:
        """Per-actuator forcerange in actuator order [yaw,pitch,knee]×4 (RL env)."""
        return [self.yaw.gear(), self.pitch.gear(), self.knee.gear()] * 4

    def wfrees(self) -> list[float]:
        return [self.yaw.wfree(), self.pitch.wfree(), self.knee.wfree()] * 4

    def actuator_mass(self) -> float:
        return 4 * (self.yaw.mass() + self.pitch.mass() + self.knee.mass())

    def total_mass(self) -> float:
        return self.torso_mass + 4 * LEG_MASS + self.actuator_mass()

    def servo_count(self) -> int:
        return 4 * sum(axis.kind.startswith("servo")
                       for axis in (self.yaw, self.pitch, self.knee))


# =============================================================================
# CURRENT — CAD-derived transmission layout with the selected servo on every axis.
# Reproduces GEARS/WFREE from gen_mesh_robot_mjcf exactly.
# =============================================================================
CURRENT = RobotDesign(
    name="current",
    half_len=0.15, half_wid=0.08, yaw_lever=0.11, stance_h=0.42,
    lift_range=0.045, lift_nom=0.022,
    yaw=Axis("servo", "waveshare_st3215_hs", reduction=BELT_RATIO),
    pitch=Axis("servo", "waveshare_st3215_hs", reduction=WORM_RATIO),
    knee=Axis("servo", "waveshare_st3215_hs", reduction=1.0, linear=True),
)

# =============================================================================
# TARGET — improved walking geometry, fixed user-selected hardware on all 12 axes.
# The yaw retains the rubber-belt SEA model; pitch uses the existing self-locking
# worm and knee/lift is direct through the 40 mm crank abstraction.
# =============================================================================
TARGET = RobotDesign(
    name="target",
    half_len=0.20, half_wid=0.12, yaw_lever=0.16, stance_h=0.42,
    lift_range=0.047, lift_nom=0.0235,
    yaw=Axis("servo_sea", "waveshare_st3215_hs", reduction=BELT_RATIO,
             sea_k_tilde=25.0),
    pitch=Axis("servo", "waveshare_st3215_hs", reduction=WORM_RATIO),
    knee=Axis("servo", "waveshare_st3215_hs", reduction=1.0, linear=True),
)


def summary(d: RobotDesign) -> str:
    return (f"{d.name}: stance {2*d.half_len*100:.0f}x{2*(d.half_wid+d.yaw_lever)*100:.0f}cm "
            f"lever {d.yaw_lever*100:.0f}cm | yaw {d.yaw.kind}/{d.yaw.part} N{d.yaw.reduction:.0f} "
            f"| pitch {d.pitch.kind} N{d.pitch.reduction:.0f} "
            f"| knee {d.knee.kind}/{d.knee.part} N{d.knee.reduction:.0f} "
            f"| actuators {d.actuator_mass():.2f}kg total {d.total_mass():.2f}kg")


if __name__ == "__main__":
    for d in (CURRENT, TARGET):
        print(summary(d))
        print(f"    gears  {[round(g, 1) for g in d.gears()[:3]]}  "
              f"wfree {[round(w, 2) for w in d.wfrees()[:3]]}")
