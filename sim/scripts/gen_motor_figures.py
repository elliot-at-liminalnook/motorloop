#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
"""Motor-comparison figures (notes/motor-selection-checklist.md §4): the three
motors head-to-head. Metrics are closed-form from the datasheet params
(motors.py), so these render fast and rebuild-free.

Figures (-> figures/motors/):
  torque_speed.png   torque-speed envelope at 12 V (M1/M2)
  dynamics.png       electrical L/R + mechanical acceleration (M3/M4)
  efficiency.png     efficiency vs speed at rated current (M6)
  latency_coupling.png  angle-latency torque loss vs speed per motor (M8 - the
                        motor<->sensor coupling, ties to the part-comparison)
  summary.png        at-a-glance bars (Kt, stall torque, efficiency, price)

Caveat: datasheet/estimate provenance (motors.py), 12 V bench - relative,
illustrative, not a rated-power hardware spec.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

PROJECT_ROOT = Path(__file__).resolve().parents[2]
for sub in ("sim/scripts", "sim/tests"):
    sys.path.insert(0, str(PROJECT_ROOT / sub))
from motors import MOTORS, SENSOR_LATENCY_S, TIERS  # noqa: E402

VBUS = 12.0
COLORS = {"gm2804": "tab:red", "db42s03": "tab:green", "maxon_ec45": "tab:blue"}
CAVEAT = ("closed-form from datasheet/estimate params (motors.py), 12 V bench - "
          "relative, not a rated-power hardware spec")


def caveat(fig):
    fig.text(0.99, 0.004, CAVEAT, ha="right", va="bottom", fontsize=6.5,
             color="0.45", style="italic")


def finish(fig, out, name):
    caveat(fig)
    p = Path(out) / f"{name}.png"
    fig.savefig(p, dpi=130)
    plt.close(fig)
    print(f"  wrote {p}")


def fig_torque_speed(out):
    fig, ax = plt.subplots(figsize=(8.6, 5.0), constrained_layout=True)
    for n in TIERS:
        m = MOTORS[n]
        w0 = m.no_load_speed_rad_s(VBUS) * 60 / (2 * np.pi)   # rpm
        tstall = m.stall_torque_nm(VBUS) * 1e3                # mN*m
        ax.plot([0, w0], [tstall, 0], "-o", color=COLORS[n],
                label=f"{m.label} ({m.pole_pairs} pp)")
    ax.set_xlabel("speed (rev/min)"); ax.set_ylabel("torque (mN*m)")
    ax.set_title("M1/M2  Torque-speed envelope at 12 V (idealized linear)")
    ax.legend(loc="upper right"); ax.grid(alpha=0.3)
    finish(fig, out, "torque_speed")


def fig_dynamics(out):
    fig, (a0, a1) = plt.subplots(1, 2, figsize=(11.0, 4.6),
                                 constrained_layout=True)
    x = np.arange(len(TIERS))
    cols = [COLORS[n] for n in TIERS]
    a0.bar(x, [MOTORS[n].elec_tau_s * 1e3 for n in TIERS], color=cols)
    a0.set_xticks(x); a0.set_xticklabels([MOTORS[n].label for n in TIERS],
                                         fontsize=7, rotation=15)
    a0.set_ylabel("electrical time constant L/R (ms)")
    a0.set_title("M3  Current-loop dynamics (L/R)"); a0.grid(alpha=0.3, axis="y")
    a1.bar(x, [MOTORS[n].accel_rad_s2(MOTORS[n].rated_current_a) for n in TIERS],
           color=cols)
    a1.set_xticks(x); a1.set_xticklabels([MOTORS[n].label for n in TIERS],
                                         fontsize=7, rotation=15)
    a1.set_ylabel("acceleration at rated current (rad/s^2)")
    a1.set_title("M4  Mechanical dynamics (Kt*I/J)"); a1.grid(alpha=0.3, axis="y")
    finish(fig, out, "dynamics")


def fig_efficiency(out):
    fig, ax = plt.subplots(figsize=(8.6, 5.0), constrained_layout=True)
    w = np.linspace(20, 400, 60)
    for n in TIERS:
        m = MOTORS[n]
        eff = [m.efficiency(m.rated_current_a, wi) * 100 for wi in w]
        ax.plot(w, eff, color=COLORS[n], label=m.label)
    ax.set_xlabel("speed (rad/s)"); ax.set_ylabel("efficiency at rated current (%)")
    ax.set_title("M6  Efficiency: the gimbal's 9 Ohm vs the characterized motors")
    ax.legend(loc="lower right"); ax.grid(alpha=0.3)
    finish(fig, out, "efficiency")


def fig_latency_coupling(out):
    fig, ax = plt.subplots(figsize=(8.6, 5.0), constrained_layout=True)
    w = np.linspace(0, 1500, 80)
    for n in TIERS:
        m = MOTORS[n]
        loss = [m.latency_torque_loss(wi, SENSOR_LATENCY_S["AS5600"]) * 100
                for wi in w]
        ax.plot(w, loss, color=COLORS[n],
                label=f"{m.label} ({m.pole_pairs} pp) + AS5600")
    ax.axhline(0.3, ls=":", color="0.5")
    ax.text(50, 0.6, "AS5047P keeps every motor < 0.01% (flat near 0)",
            fontsize=8, color="0.35")
    ax.set_xlabel("speed (rad/s)")
    ax.set_ylabel("torque lost to angle latency (%)")
    ax.set_title("M8  Motor<->sensor coupling: more pole pairs -> more "
                 "latency-sensitive")
    ax.legend(loc="upper left"); ax.grid(alpha=0.3)
    finish(fig, out, "latency_coupling")


def fig_summary(out):
    fig, axes = plt.subplots(2, 2, figsize=(10.5, 7.0), constrained_layout=True)
    x = np.arange(len(TIERS))
    labels = [MOTORS[n].label for n in TIERS]
    cols = [COLORS[n] for n in TIERS]

    def bar(ax, vals, title, ylabel, fmt="%.0f"):
        ax.bar(x, vals, color=cols)
        ax.set_xticks(x); ax.set_xticklabels(labels, fontsize=7, rotation=15)
        ax.set_title(title, fontsize=10); ax.set_ylabel(ylabel)
        ax.grid(alpha=0.3, axis="y")
        for i, v in enumerate(vals):
            ax.text(i, v, fmt % v, ha="center", va="bottom", fontsize=8)

    bar(axes[0, 0], [MOTORS[n].kt * 1e3 for n in TIERS], "Torque constant Kt",
        "mN*m/A", "%.0f")
    bar(axes[0, 1], [MOTORS[n].stall_torque_nm(VBUS) * 1e3 for n in TIERS],
        "Stall torque @ 12 V", "mN*m", "%.0f")
    bar(axes[1, 0], [MOTORS[n].efficiency(MOTORS[n].rated_current_a, 200) * 100
                     for n in TIERS], "Efficiency @ rated", "%", "%.0f")
    bar(axes[1, 1], [MOTORS[n].price_usd for n in TIERS], "Price", "USD", "%.0f")
    fig.suptitle("Motor comparison: budget GM2804 / mid DB42S03 / premium "
                 "EC 45 flat", fontsize=12)
    finish(fig, out, "summary")


FIGURES = {"torque_speed": fig_torque_speed, "dynamics": fig_dynamics,
           "efficiency": fig_efficiency, "latency_coupling": fig_latency_coupling,
           "summary": fig_summary}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=str(PROJECT_ROOT / "figures" / "motors"))
    ap.add_argument("--only", default="")
    args = ap.parse_args()
    out = Path(args.out); out.mkdir(parents=True, exist_ok=True)
    names = [n.strip() for n in args.only.split(",") if n.strip()] or FIGURES
    for name in names:
        print(f"rendering {name} ...")
        FIGURES[name](out)
    print(f"done -> {out}")


if __name__ == "__main__":
    main()
