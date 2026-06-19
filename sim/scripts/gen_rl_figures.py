#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
"""RL motor-coupling figures (notes/rl-environment-checklist.md §5).

The motor-envelope figure is closed-form from actuator.py (system python, no
torch/mujoco) - it shows how each motor profile constrains the joint torque the
RL policy can command vs joint speed (the RL<->motor coupling). If a cross-motor
return summary exists (sim/build/rl/coupling_returns.json, written by the eval
sweep), a return-per-motor bar is added.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

ROOT = Path(__file__).resolve().parents[2]
for sub in ("sim/rl", "sim/tests"):
    sys.path.insert(0, str(ROOT / sub))
from actuator import MotorActuator  # noqa: E402
from motors import MOTORS, TIERS  # noqa: E402

FIG = ROOT / "figures" / "rl"
COLORS = {"gm2804": "tab:red", "db42s03": "tab:green", "maxon_ec45": "tab:blue"}
CAVEAT = ("closed-form from the motor profiles (actuator.py); the RL policy's "
          "torque command is scaled by this envelope at each joint")


def caveat(fig):
    fig.text(0.99, 0.004, CAVEAT, ha="right", va="bottom", fontsize=6.5,
             color="0.45", style="italic")


def fig_motor_envelope(gear=2.0, vbus=12.0):
    w = np.linspace(0, 60, 200)
    fig, ax = plt.subplots(figsize=(8.6, 5.0), constrained_layout=True)
    for n in TIERS:
        a = MotorActuator(MOTORS[n], gear_ratio=gear, vbus=vbus)
        ax.plot(w, a.torque_scale(w) * 100, color=COLORS[n], lw=1.8,
                label=f"{MOTORS[n].label} (knee {a.base_joint_speed_rad_s():.0f} rad/s)")
    ax.set_xlabel("joint speed (rad/s)")
    ax.set_ylabel("available torque (% of stall)")
    ax.set_title(f"RL motor coupling: torque envelope the policy must respect "
                 f"(gear {gear:g}:1, {vbus:g} V)")
    ax.legend(loc="lower left"); ax.grid(alpha=0.3)
    ax.text(0.97, 0.9, "the gimbal starves at speed -> caps how fast the robot\n"
            "can move; the servo holds torque much further",
            transform=ax.transAxes, ha="right", fontsize=8, color="0.35")
    caveat(fig)
    FIG.mkdir(parents=True, exist_ok=True)
    fig.savefig(FIG / "motor_envelope.png", dpi=130)
    plt.close(fig)
    print(f"wrote {FIG / 'motor_envelope.png'}")


def fig_coupling_returns():
    p = ROOT / "sim" / "build" / "rl" / "coupling_returns.json"
    if not p.is_file():
        print("(no coupling_returns.json yet — run the eval sweep)")
        return
    data = json.loads(p.read_text())
    motors = [m for m in TIERS if m in data]
    fig, ax = plt.subplots(figsize=(8.0, 5.0), constrained_layout=True)
    ax.bar(range(len(motors)), [data[m] for m in motors],
           color=[COLORS[m] for m in motors])
    ax.set_xticks(range(len(motors)))
    ax.set_xticklabels([MOTORS[m].label for m in motors], fontsize=8, rotation=15)
    ax.set_ylabel("mean episode return (same policy, swapped motor)")
    ax.set_title("RL motor coupling: the same trained policy under each motor")
    ax.grid(alpha=0.3, axis="y")
    caveat(fig)
    fig.savefig(FIG / "coupling_returns.png", dpi=130)
    plt.close(fig)
    print(f"wrote {FIG / 'coupling_returns.png'}")


def main():
    fig_motor_envelope()
    fig_coupling_returns()


if __name__ == "__main__":
    main()
