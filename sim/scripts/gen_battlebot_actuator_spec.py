# SPDX-License-Identifier: MIT
"""First-cut actuator spec for the leg-weapon battlebot (motorloop-battlebot-design).

The sizing motion is M1 = vertical retract: lift the leg/spear clear of a low
spinner's strike plane FAST enough to dodge (and the same axis drives the spear).
We compute the kinematics -> peak force/power for a range of dodge-time budgets,
then check each profiled motor's power ceiling against it. Power is the feasibility
gate: a belt ratio can trade torque<->speed freely, but F*v <= P_motor*eff always.

Assumptions are provenance-tagged (decided/assumed) - refine with real CAD masses
and the confirmed weight class. Reuses sim/tests/motors.py (the profiled motors).
Run: python sim/scripts/gen_battlebot_actuator_spec.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "sim" / "tests"))
from motors import MOTORS, TIERS  # noqa: E402

FIG = ROOT / "figures" / "battlebot"

# --- motion requirement (provenance-tagged first-cut) ---
LIFT_M = 0.20            # decided: ~8 in retract (the 6-10 in spec)
MASS_KG = 1.5           # assumed: moving leg + spear + tip (refine from CAD)
G = 9.81
DRIVE_EFF = 0.85       # assumed: belt drive efficiency
T_BUDGET = np.array([0.08, 0.10, 0.125, 0.15, 0.20, 0.25])   # s, dodge-time sweep


def motion_demand(d, m, t):
    """Triangular accel/decel velocity profile: peak v, accel, force, power."""
    v_pk = 2 * d / t
    a_pk = 4 * d / t ** 2
    f_pk = m * (a_pk + G)            # worst case: accelerate up against gravity
    p_pk = f_pk * v_pk              # peak mechanical power (~mid-stroke)
    return v_pk, a_pk, f_pk, p_pk


def motor_power_ceiling(mot, vbus):
    """Max mechanical power at the optimal operating point (back-EMF = V/2):
    P = V^2/(4 R_ll), drawing I* = V/(2 R_ll). Returns (P_W, I*_A, burst x rated)."""
    p = vbus ** 2 / (4.0 * mot.r_line_line)
    i_star = vbus / (2.0 * mot.r_line_line)
    return p, i_star, i_star / mot.rated_current_a


def main():
    print(f"Leg-weapon retract spec - lift {LIFT_M*39.37:.0f} in ({LIFT_M} m), "
          f"moving mass {MASS_KG} kg, belt eff {DRIVE_EFF}\n")
    print("dodge   peak v   peak a    peak F   peak P   (P/eff at motor shaft)")
    print("time s   m/s     g         N        W        W")
    demands = {}
    for t in T_BUDGET:
        v, a, f, p = motion_demand(LIFT_M, MASS_KG, t)
        demands[t] = p / DRIVE_EFF
        print(f"{t:5.3f}  {v:6.2f}  {a/G:6.1f}    {f:7.1f}  {p:7.0f}   {p/DRIVE_EFF:7.0f}")

    print("\nMotor power ceilings  P=V^2/(4R_ll)  (burst current I*=V/(2R), x rated):")
    print("motor                24V bus              48V bus (rule ceiling)")
    ceilings = {}
    for key in TIERS:
        m = MOTORS[key]
        p24, i24, x24 = motor_power_ceiling(m, 24.0)
        p48, i48, x48 = motor_power_ceiling(m, 48.0)
        ceilings[key] = (p24, p48)
        print(f"{m.label:20s} {p24:5.0f} W ({i24:4.1f}A {x24:4.1f}x)   "
              f"{p48:5.0f} W ({i48:4.1f}A {x48:4.1f}x)")

    print("\nVerdict - fastest dodge each motor supports at 24V / 48V (nominal mass):")
    for key in TIERS:
        m = MOTORS[key]
        p24, p48 = ceilings[key]
        def fastest(cap):
            ok = [t for t in T_BUDGET if demands[t] <= cap]
            return f"{min(ok):.3f}s" if ok else "none"
        print(f"  {m.label:20s} 24V: {fastest(p24):>6s}   48V: {fastest(p48):>6s}")

    # figure: required power vs dodge time, with motor ceilings
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        tt = np.linspace(0.07, 0.27, 200)
        pp = np.array([motion_demand(LIFT_M, MASS_KG, t)[3] / DRIVE_EFF for t in tt])
        fig, ax = plt.subplots(figsize=(7, 4.5))
        ax.plot(tt * 1000, pp, "k-", lw=2, label=f"required ({MASS_KG} kg, {LIFT_M} m)")
        for key in TIERS:
            m = MOTORS[key]
            p24 = ceilings[key][0]
            ax.axhline(p24, ls="--", lw=1.2, label=f"{m.label} ceiling @24V ({p24:.0f} W)")
        ax.set_xlabel("dodge time budget (ms)")
        ax.set_ylabel("peak shaft power (W)")
        ax.set_title("Leg retract: power needed to dodge vs motor ceilings")
        ax.set_ylim(0, max(pp) * 1.05)
        ax.grid(alpha=0.3); ax.legend(fontsize=8)
        FIG.mkdir(parents=True, exist_ok=True)
        out = FIG / "retract_power.png"
        fig.tight_layout(); fig.savefig(out, dpi=110)
        print(f"\nwrote {out}")
    except Exception as e:
        print(f"(figure skipped: {e})")


if __name__ == "__main__":
    main()
