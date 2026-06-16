#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
"""Generate the README and gallery figures from live bench runs.

Every image embedded in README.md, sim/README.md, and figures/gallery.md is
rendered by this script from actual bench traces - no mockups. Figures:

  startup.png       closed-loop sensored six-step startup (clean config)
  commutation.png   steady-state commutation zoom from the same run
  brownout.png      emergent brownout chain (S6a: 1 A supply realism)
  regen.png         regen pump-up bounded by the duty down-slew guard (S6b)
  adc_chain.png     what the RTL believes vs the plant's truth (ADC chain)
  thermal.png       locked-rotor winding heating + R(T) current droop (S9b)
  cogging.png       startup from a cogging detent vs the clean model (S7)
  eccentricity.png  AS5600 eccentricity signature + closed-loop ripple (S10)
  pwm_ripple.png    per-cycle switching ripple (fine trace interval)
  stall_raster.png  stall detect / safe-off / acknowledge / re-stall (E21)
  deadtime.png      gate handoff microscopy at 200 ns resolution
  parity.png        three-way plant parity: C++ / Python / Modelica
  motorloop.gif     animated rotor + phase currents loop

Everything is parameterized from sim/config/params.toml (thresholds and
annotations included), so the figures regenerate mechanically when
parameters change. Each figure carries the standing caveat: motor
parameters are placeholders, so these are illustrations of system
behavior, not hardware predictions.

Usage:
  python3 sim/scripts/gen_readme_figures.py [--out figures]
      [--only startup,regen,...]   # any subset of the names above
"""

from __future__ import annotations

import argparse
import math
import shutil
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from matplotlib import animation  # noqa: E402

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT / "sim" / "scripts"))
sys.path.insert(0, str(PROJECT_ROOT / "sim" / "tests"))
sys.path.insert(0, str(PROJECT_ROOT / "sim" / "build" / "cpp"))

import sim_params  # noqa: E402
from bench_factory import (bench_config, expected_init_time, foc,  # noqa: E402
                           freq_word, realism)

CAVEAT = ("simulated output with placeholder motor parameters "
          "(provenance: sim/config/params.toml) - illustrative, "
          "not a hardware prediction")

SECTOR_COLORS = plt.get_cmap("tab10")
PHASE_COLORS = ("tab:blue", "tab:orange", "tab:green")


def caveat(fig):
    fig.text(0.99, 0.005, CAVEAT, ha="right", va="bottom",
             fontsize=6.5, color="0.45", style="italic")


def arr(trace, key):
    return np.asarray(trace[key], dtype=float)


def wrap_pi(x):
    return (np.asarray(x) + np.pi) % (2 * np.pi) - np.pi


def shade_sectors(ax, t, sector):
    """Translucent background band per commutation sector."""
    edges = np.flatnonzero(np.diff(sector) != 0) + 1
    bounds = np.concatenate(([0], edges, [len(t) - 1]))
    for lo, hi in zip(bounds[:-1], bounds[1:]):
        s = int(sector[lo])
        ax.axvspan(t[lo], t[hi], color=SECTOR_COLORS(s % 10), alpha=0.10,
                   linewidth=0)


def plot_phases(ax, t, trace, mask=None, lw=0.5):
    for ph, color in zip("abc", PHASE_COLORS):
        v = arr(trace, f"i{ph}_a")
        if mask is not None:
            ax.plot(t, v[mask], lw=lw, color=color, label=f"i{ph}")
        else:
            ax.plot(t, v, lw=lw, color=color, label=f"i{ph}")


def steady_bench(params, bldcsim, cfg=None, target=80, seconds=1.2):
    b = bldcsim.Bench(cfg if cfg is not None else bench_config(params))
    b.run_for(expected_init_time(params))
    assert b.configured, "DRV init did not complete"
    b.set_align_offset(int(params.value("rtl.sensored_align_offset")))
    b.set_target_speed(target)
    b.set_mode(2)
    b.run_for(seconds)
    assert b.shoot_through_violations == 0
    return b


# ---------------------------------------------------------------------------
# startup + commutation + gif (one shared run)
# ---------------------------------------------------------------------------

def run_startup(params, bldcsim, target=80, seconds=1.2):
    t_init = expected_init_time(params)
    b = steady_bench(params, bldcsim, target=target, seconds=seconds)
    return b.trace, t_init


def fig_startup(trace, t_init, params, out):
    t = arr(trace, "time_s")
    fig, axes = plt.subplots(3, 1, sharex=True, figsize=(9.5, 6.4),
                             constrained_layout=True)
    fig.suptitle("Closed-loop startup: Verilog speed control of the "
                 "simulated motor", fontsize=12)

    ax = axes[0]
    ax.plot(t, arr(trace, "omega_rad_s"), label="plant speed (truth)",
            color="tab:blue", lw=1.4)
    ax.plot(t, arr(trace, "speed_rad_s"), label="RTL speed estimate",
            color="tab:orange", lw=1.0, ls="--")
    ax.axhline(80, color="0.4", ls=":", lw=1, label="target (80 rad/s)")
    ax.axvspan(0, t_init, color="0.5", alpha=0.15, linewidth=0)
    ax.annotate("DRV8301 init + DC-cal\n(SPI config, EN_GATE)",
                xy=(t_init, 8), xytext=(t_init + 0.06, 18), fontsize=8,
                arrowprops=dict(arrowstyle="->", color="0.3"), color="0.25")
    ax.set_ylabel("speed [rad/s]")
    ax.legend(loc="lower right", fontsize=8)
    ax.grid(alpha=0.25)

    ax = axes[1]
    plot_phases(ax, t, trace)
    ax.set_ylabel("phase current [A]")
    ax.legend(loc="upper right", fontsize=8, ncol=3)
    ax.grid(alpha=0.25)

    ax = axes[2]
    ax.plot(t, arr(trace, "duty") / 625.0 * 100.0, color="tab:purple",
            lw=1.0)
    ax.set_ylabel("PWM duty [%]")
    ax.set_xlabel("time [s]")
    ax.set_ylim(0, 100)
    ax.grid(alpha=0.25)

    caveat(fig)
    fig.savefig(out, dpi=160, facecolor="white")
    plt.close(fig)


def fig_commutation(trace, out, window_s=0.06):
    t = arr(trace, "time_s")
    sector = arr(trace, "sector")
    hi = t[-1] - 0.02
    lo = hi - window_s
    m = (t >= lo) & (t <= hi)
    t_ms = (t[m] - lo) * 1e3

    fig, axes = plt.subplots(2, 1, sharex=True, figsize=(9.5, 4.6),
                             constrained_layout=True,
                             height_ratios=[3, 1])
    fig.suptitle("Six-step commutation at steady state "
                 "(sector bands from the RTL's own debug port)",
                 fontsize=12)

    ax = axes[0]
    shade_sectors(ax, t_ms, sector[m])
    for ph, color in zip("abc", PHASE_COLORS):
        ax.plot(t_ms, arr(trace, f"i{ph}_a")[m], lw=1.2, color=color,
                label=f"i{ph}")
    ax.set_ylabel("phase current [A]")
    ax.legend(loc="upper right", fontsize=8, ncol=3)
    ax.grid(alpha=0.25)

    ax = axes[1]
    ax.step(t_ms, sector[m], where="post", color="0.3", lw=1.2)
    ax.set_ylabel("sector")
    ax.set_yticks(range(6))
    ax.set_xlabel(f"time [ms] (window of {window_s*1e3:.0f} ms)")
    ax.grid(alpha=0.25)

    caveat(fig)
    fig.savefig(out, dpi=160, facecolor="white")
    plt.close(fig)


def fig_gif(trace, params, out, frames=60, fps=12):
    """Animated rotor + sliding phase-current window, one mechanical rev."""
    t = arr(trace, "time_s")
    theta = arr(trace, "theta_rad")
    sector = arr(trace, "sector")
    omega = float(np.mean(arr(trace, "omega_rad_s")[-2000:]))
    rev_s = 2 * math.pi / max(omega, 1.0)
    t1 = t[-1] - 0.01
    t0 = t1 - rev_s
    idx = np.linspace(np.searchsorted(t, t0), np.searchsorted(t, t1),
                      frames).astype(int)

    fig, (axr, axc) = plt.subplots(
        1, 2, figsize=(8.0, 3.2), dpi=100,
        gridspec_kw={"width_ratios": [1, 2.2]})
    fig.suptitle("motorloop: Verilog RTL spinning a simulated motor",
                 fontsize=10)
    fig.subplots_adjust(left=0.02, right=0.97, top=0.86, bottom=0.16,
                        wspace=0.15)

    axr.set_aspect("equal")
    axr.set_xlim(-1.25, 1.25)
    axr.set_ylim(-1.25, 1.25)
    axr.axis("off")
    ring = plt.Circle((0, 0), 1.0, fill=False, color="0.6", lw=2)
    axr.add_patch(ring)
    arrow, = axr.plot([], [], lw=3, solid_capstyle="round")
    hub = plt.Circle((0, 0), 0.08, color="0.3")
    axr.add_patch(hub)
    sector_txt = axr.text(0, -1.18, "", ha="center", fontsize=8,
                          color="0.3")

    win = 0.025
    cur_lines = [axc.plot([], [], lw=0.9, color=c, label=f"i{p}")[0]
                 for p, c in zip("abc", PHASE_COLORS)]
    time_marker = axc.axvline(0, color="0.3", lw=0.8, ls=":")
    iall = np.concatenate([arr(trace, f"i{p}_a")[idx[0]:idx[-1]]
                           for p in "abc"])
    pad = 1.2 * np.max(np.abs(iall))
    axc.set_ylim(-pad, pad)
    axc.set_xlabel("time [ms]", fontsize=8)
    axc.set_ylabel("phase current [A]", fontsize=8)
    axc.tick_params(labelsize=7)
    axc.legend(loc="upper right", fontsize=7, ncol=3)
    axc.grid(alpha=0.25)

    def draw(k):
        i = idx[k]
        th = theta[i]
        s = int(sector[i])
        color = SECTOR_COLORS(s % 10)
        arrow.set_data([0, math.cos(th)], [0, math.sin(th)])
        arrow.set_color(color)
        sector_txt.set_text(f"sector {s}   "
                            f"θ = {math.degrees(th) % 360:5.1f}°")
        m = (t >= t[i] - win) & (t <= t[i] + win)
        tm = (t[m] - t[i]) * 1e3
        for p, line in zip("abc", cur_lines):
            line.set_data(tm, arr(trace, f"i{p}_a")[m])
        axc.set_xlim(-win * 1e3, win * 1e3)
        time_marker.set_xdata([0, 0])
        return [arrow, sector_txt, time_marker, *cur_lines]

    anim = animation.FuncAnimation(fig, draw, frames=frames, blit=True)
    anim.save(out, writer=animation.PillowWriter(fps=fps))
    plt.close(fig)


# ---------------------------------------------------------------------------
# brownout (S6a) and regen pump-up (S6b)
# ---------------------------------------------------------------------------

def run_brownout(params, bldcsim, target=80, run_s=1.5, idle_s=0.5):
    b = bldcsim.Bench(realism(params, "supply"))
    b.run_for(expected_init_time(params))
    b.set_align_offset(int(params.value("rtl.sensored_align_offset")))
    b.set_target_speed(target)
    b.set_mode(2)
    b.run_for(run_s)
    t_idle = b.time_s
    b.set_mode(0)
    b.run_for(idle_s)
    assert b.pvdd_uv_events >= 1, "no emergent UVLO - figure premise broken"
    assert b.configured, "watchdog did not recover - figure premise broken"
    return b.trace, t_idle


def fig_brownout(trace, t_idle, params, out):
    t = arr(trace, "time_s")
    vbus = arr(trace, "vbus_v")
    v_falling = params.value("drv8301_uvlo.pvdd_falling")
    i_limit = params.value("bus.supply_current_limit")
    uv = np.flatnonzero(vbus < v_falling)
    t_uv = t[uv[0]] if uv.size else None

    fig, axes = plt.subplots(3, 1, sharex=True, figsize=(9.5, 6.4),
                             constrained_layout=True)
    fig.suptitle("Emergent brownout: nothing here is injected - a "
                 f"{i_limit:.0f} A bench supply is simply not enough",
                 fontsize=12)

    ax = axes[0]
    ax.plot(t, vbus, color="tab:red", lw=1.0, label="bus voltage")
    ax.axhline(v_falling, color="0.2", ls="--", lw=1,
               label=f"DRV8301 UVLO falling ({v_falling} V, datasheet)")
    if t_uv is not None:
        ax.annotate("motor load drags the current-limited\nsupply below "
                    "UVLO -> registers silently reset",
                    xy=(t_uv, v_falling), xytext=(t_uv + 0.12, 2.5),
                    fontsize=8, color="0.25",
                    arrowprops=dict(arrowstyle="->", color="0.3"))
    ax.axvline(t_idle, color="0.4", ls=":", lw=1)
    ax.annotate("controller idled ->\nbus recovers, watchdog\nre-runs init",
                xy=(t_idle, 9), xytext=(t_idle + 0.05, 4.0), fontsize=8,
                color="0.25",
                arrowprops=dict(arrowstyle="->", color="0.3"))
    ax.set_ylabel("bus voltage [V]")
    ax.legend(loc="center right", fontsize=8)
    ax.grid(alpha=0.25)

    ax = axes[1]
    ax.plot(t, arr(trace, "omega_rad_s"), color="tab:blue", lw=1.2)
    ax.set_ylabel("speed [rad/s]")
    ax.grid(alpha=0.25)

    ax = axes[2]
    ax.step(t, arr(trace, "fault_count"), where="post", color="tab:red",
            lw=1.2, label="nFAULT events seen by RTL")
    ax2 = ax.twinx()
    ax2.fill_between(t, arr(trace, "configured"), step="post",
                     color="tab:green", alpha=0.25, linewidth=0)
    ax2.set_ylabel("configured", color="tab:green")
    ax2.set_yticks([0, 1])
    ax.set_ylabel("fault count")
    ax.set_xlabel("time [s]")
    ax.legend(loc="upper left", fontsize=8)
    ax.grid(alpha=0.25)

    caveat(fig)
    fig.savefig(out, dpi=160, facecolor="white")
    plt.close(fig)


def fig_regen(params, bldcsim, out):
    b = bldcsim.Bench(realism(params, "supply",
                              supply={"i_limit_a": 20.0}))
    b.run_for(expected_init_time(params))
    b.set_align_offset(int(params.value("rtl.sensored_align_offset")))
    b.set_target_speed(110)
    b.set_mode(2)
    b.run_for(1.2)
    t_decel = b.time_s
    b.set_target_speed(10)
    b.run_for(0.8)
    assert b.shoot_through_violations == 0
    trace = b.trace

    t = arr(trace, "time_s")
    vbus = arr(trace, "vbus_v")
    v_set = params.value("bus.vbus")
    target = np.where(t < t_decel, 110.0, 10.0)
    i_peak = int(np.argmax(vbus))

    fig, axes = plt.subplots(3, 1, sharex=True, figsize=(9.5, 6.4),
                             constrained_layout=True)
    fig.suptitle("Regenerative braking pumps the bus - the supply cannot "
                 "sink current, the RTL's down-slew guard bounds it",
                 fontsize=12)

    ax = axes[0]
    ax.plot(t, vbus, color="tab:red", lw=1.0, label="bus voltage")
    ax.axhline(v_set, color="0.2", ls="--", lw=1,
               label=f"supply setpoint ({v_set:.0f} V)")
    ax.annotate(f"regen charges the bus cap\n(peak {vbus[i_peak]:.2f} V); "
                "the supply only\nsources, so it stays up",
                xy=(t[i_peak], vbus[i_peak]),
                xytext=(0.45, 0.55), textcoords="axes fraction",
                fontsize=8, color="0.25",
                arrowprops=dict(arrowstyle="->", color="0.3"))
    ax.set_ylabel("bus voltage [V]")
    ax.legend(loc="lower left", fontsize=8)
    ax.grid(alpha=0.25)

    ax = axes[1]
    ax.plot(t, arr(trace, "omega_rad_s"), color="tab:blue", lw=1.2,
            label="plant speed")
    ax.plot(t, target, color="0.4", ls=":", lw=1.2, label="target")
    ax.axvline(t_decel, color="0.4", ls=":", lw=1)
    ax.annotate("hard decel commanded", xy=(t_decel, 60),
                xytext=(t_decel + 0.1, 75), fontsize=8, color="0.25",
                arrowprops=dict(arrowstyle="->", color="0.3"))
    ax.set_ylabel("speed [rad/s]")
    ax.legend(loc="upper right", fontsize=8)
    ax.grid(alpha=0.25)

    ax = axes[2]
    ax.plot(t, arr(trace, "duty") / 625.0 * 100.0, color="tab:purple",
            lw=1.0, label="PWM duty")
    ax.annotate("duty falls at the guard's slew limit,\nnot a step - "
                "bounding the regen current",
                xy=(t_decel + 0.05, 30), xytext=(t_decel + 0.2, 55),
                fontsize=8, color="0.25",
                arrowprops=dict(arrowstyle="->", color="0.3"))
    ax.set_ylabel("PWM duty [%]")
    ax.set_xlabel("time [s]")
    ax.grid(alpha=0.25)

    caveat(fig)
    fig.savefig(out, dpi=160, facecolor="white")
    plt.close(fig)


# ---------------------------------------------------------------------------
# adc chain: truth vs what the RTL decodes
# ---------------------------------------------------------------------------

def fig_adc_chain(params, bldcsim, out):
    b = steady_bench(params, bldcsim,
                     cfg=realism(params, "supply",
                                 supply={"i_limit_a": 20.0}),
                     target=80, seconds=1.0)
    trace = b.trace
    t = arr(trace, "time_s")
    hi = t[-1] - 0.002
    lo = hi - 0.012
    m = (t >= lo) & (t <= hi)
    t_ms = (t[m] - lo) * 1e3

    vref = params.value("adc.vref")
    offset = params.value("feedback.current.offset")
    gain = params.value("drv8301.amp_gain")
    shunt = params.value("feedback.current.shunt")
    divider = params.value("feedback.bus_voltage.divider_ratio")

    i_rtl = (arr(trace, "cur_code") / 4096.0 * vref - offset) / (gain * shunt)
    v_rtl = arr(trace, "vbus_code") / 4096.0 * vref / divider

    fig, axes = plt.subplots(2, 1, sharex=True, figsize=(9.5, 5.6),
                             constrained_layout=True)
    fig.suptitle("What the RTL believes vs what is true: the modeled "
                 "feedback chain (divider, RC, aperture, quantization)",
                 fontsize=12)

    ax = axes[0]
    plot_phases(ax, t_ms, trace, mask=m, lw=0.8)
    ax.step(t_ms, i_rtl[m], where="post", color="0.15", lw=1.3,
            label="RTL's decoded shunt current")
    ax.set_title("the staircase rides whichever low-side shunt the ADC "
                 "sequencer is sampling that conversion", fontsize=9,
                 color="0.25")
    ax.set_ylabel("current [A]")
    ax.legend(loc="upper right", fontsize=8, ncol=4)
    ax.grid(alpha=0.25)

    ax = axes[1]
    ax.plot(t_ms, arr(trace, "vbus_v")[m], color="tab:red", lw=1.0,
            label="bus voltage (truth)")
    ax.step(t_ms, v_rtl[m], where="post", color="0.15", lw=1.3,
            label="RTL's decoded bus voltage")
    ax.set_ylabel("bus voltage [V]")
    ax.set_xlabel("time [ms]")
    ax.legend(loc="upper right", fontsize=8)
    ax.grid(alpha=0.25)

    caveat(fig)
    fig.savefig(out, dpi=160, facecolor="white")
    plt.close(fig)


# ---------------------------------------------------------------------------
# thermal: locked rotor heats the winding, R(T) droops the current (S9b)
# ---------------------------------------------------------------------------

def fig_thermal(params, bldcsim, out):
    b = bldcsim.Bench(realism(params, "thermal",
                              motor={"inertia_kg_m2": 1e9}))
    b.run_for(expected_init_time(params))
    clk = params.value("rtl.clock_frequency")
    b.set_open_loop(freq_word(0.01, clk), 2**31)
    b.set_duty(int(0.8 * 625))
    b.set_mode(1)
    b.run_for(2.5)
    trace = b.trace

    t = arr(trace, "time_s")
    i_mag = np.max(np.abs(np.stack(
        [arr(trace, f"i{p}_a") for p in "abc"])), axis=0)
    motor_t = arr(trace, "motor_t_c")
    drive = i_mag > 0.05
    i0 = float(np.percentile(i_mag[drive], 95))
    i1 = float(i_mag[-1])

    fig, axes = plt.subplots(2, 1, sharex=True, figsize=(9.5, 5.6),
                             constrained_layout=True)
    fig.suptitle("Locked rotor at fixed duty: the winding lump heats and "
                 "rising resistance visibly droops the stall current",
                 fontsize=12)

    ax = axes[0]
    ax.plot(t, motor_t, color="tab:red", lw=1.4, label="winding temperature")
    ax.plot(t, arr(trace, "fet_tj_c"), color="tab:orange", lw=1.0,
            label="hottest FET junction")
    ax.set_ylabel("temperature [°C]")
    ax.legend(loc="center right", fontsize=8)
    ax.grid(alpha=0.25)

    ax = axes[1]
    ax.plot(t, i_mag, color="tab:blue", lw=1.0)
    ax.set_ylim(i1 - 0.35, i0 + 0.15)
    ax.annotate("R rises with winding temperature ->\nstall current falls "
                f"({i0:.2f} A -> {i1:.2f} A) with no\ncontrol action at all",
                xy=(t[-1] - 0.05, i1), xytext=(0.55, 0.18),
                textcoords="axes fraction", fontsize=8, color="0.25",
                arrowprops=dict(arrowstyle="->", color="0.3"))
    ax.set_ylabel("|phase current| [A] (zoomed)")
    ax.set_xlabel("time [s]")
    ax.grid(alpha=0.25)

    caveat(fig)
    fig.savefig(out, dpi=160, facecolor="white")
    plt.close(fig)


# ---------------------------------------------------------------------------
# cogging detent startup (S7)
# ---------------------------------------------------------------------------

def fig_cogging(params, bldcsim, out):
    t_init = expected_init_time(params)
    clean = steady_bench(params, bldcsim, target=60, seconds=1.2)
    rough = steady_bench(params, bldcsim,
                         cfg=realism(params, "mechanical"),
                         target=60, seconds=1.2)

    fig, axes = plt.subplots(1, 2, figsize=(9.5, 4.0),
                             constrained_layout=True,
                             width_ratios=[2, 1])
    fig.suptitle("Cogging + stiction: the closed loop hides them (left), "
                 "the physics does not (right)", fontsize=12)

    ax = axes[0]
    for b, label, color in ((clean, "clean model", "0.55"),
                            (rough, "cogging + stiction", "tab:blue")):
        tr = b.trace
        ax.plot(arr(tr, "time_s"), arr(tr, "omega_rad_s"), lw=1.2,
                color=color, label=label)
    ax.axhline(60, color="0.4", ls=":", lw=1, label="target (60 rad/s)")
    ax.set_xlabel("time [s]")
    ax.set_ylabel("speed [rad/s]")
    ax.legend(loc="lower right", fontsize=8)
    ax.grid(alpha=0.25)

    # Right panel: the detent itself, plant-only. The RTL's alignment
    # phase winds far more torque than stiction needs, so the closed loop
    # genuinely masks the detents (an honest finding - the left curves
    # nearly coincide). Give both rotors an identical shove with the gates
    # off and watch where they park instead.
    ax = axes[1]
    order = int(params.value("motor.cogging_order"))
    pitch_deg = 360.0 / order
    final_deg = {}
    for which, label, color in (("clean", "clean (coasts on)", "0.55"),
                                ("rough", "cogging + stiction",
                                 "tab:blue")):
        cfg = (bench_config(params) if which == "clean"
               else realism(params, "mechanical"))
        plant = bldcsim.ThreePhasePlant(cfg["motor"], cfg["bridge"],
                                        cfg["plant"])
        plant.set_gates([False] * 3, [False] * 3)
        plant.set_load_torque(0.012)
        plant.advance(0.08)
        plant.set_load_torque(0.0)
        ts, th = [], []
        for k in range(500):
            plant.advance(2e-3)
            ts.append((k + 1) * 2e-3)
            th.append(math.degrees(plant.theta_rad))
        ax.plot(ts, th, lw=1.4, color=color, label=label)
        final_deg[which] = th[-1]
    lo = min(0.0, final_deg["rough"] - 2 * pitch_deg)
    hi = max(final_deg["clean"], final_deg["rough"]) + pitch_deg
    for k in range(int(lo // pitch_deg), int(hi // pitch_deg) + 2):
        ax.axhline(k * pitch_deg, color="0.85", lw=0.6, zorder=0)
    ax.annotate(f"parks exactly on a detent\n({order} per rev, "
                f"{pitch_deg:.0f}° pitch - gray gridlines)",
                xy=(0.6, final_deg["rough"]),
                xytext=(0.35, 0.45), textcoords="axes fraction",
                fontsize=8, color="0.25",
                arrowprops=dict(arrowstyle="->", color="0.3"))
    ax.set_xlabel("time [s] (gates off, identical shove)")
    ax.set_ylabel("rotor angle [deg]")
    ax.legend(loc="upper left", fontsize=8)
    ax.grid(alpha=0.15, axis="x")

    caveat(fig)
    fig.savefig(out, dpi=160, facecolor="white")
    plt.close(fig)


# ---------------------------------------------------------------------------
# sensor eccentricity signature (S10)
# ---------------------------------------------------------------------------

def fig_eccentricity(params, bldcsim, out):
    b = steady_bench(params, bldcsim, cfg=realism(params, "sensor"),
                     target=80, seconds=0.9)
    theta_m, err = [], []
    for _ in range(700):
        b.run_for(2.5e-4)
        theta_m.append(b.theta % (2 * math.pi))
        err.append(b.encoder_angle_rad - b.theta)
    theta_m = np.asarray(theta_m)
    err = wrap_pi(np.asarray(err))
    err = err - np.mean(err)

    # Least-squares fit of the 1/rev + 2/rev signature the model implements.
    A = np.column_stack([np.sin(theta_m), np.cos(theta_m),
                         np.sin(2 * theta_m), np.cos(2 * theta_m),
                         np.ones_like(theta_m)])
    coef, *_ = np.linalg.lstsq(A, err, rcond=None)
    amp1 = math.hypot(coef[0], coef[1])
    amp2 = math.hypot(coef[2], coef[3])
    e1 = params.value("sensor_imperfection.eccentricity_e1")
    e2 = params.value("sensor_imperfection.eccentricity_e2")
    grid = np.linspace(0, 2 * math.pi, 400)
    fit = (coef[0] * np.sin(grid) + coef[1] * np.cos(grid)
           + coef[2] * np.sin(2 * grid) + coef[3] * np.cos(2 * grid)
           + coef[4])

    trace = b.trace
    t = arr(trace, "time_s")
    m = t >= t[-1] - 0.25

    fig, axes = plt.subplots(1, 2, figsize=(9.5, 4.0),
                             constrained_layout=True)
    fig.suptitle("An off-center sensor magnet leaves a once-per-rev "
                 "signature - and the loop tolerates it", fontsize=12)

    ax = axes[0]
    ax.plot(theta_m, err * 1e3, ".", ms=2.5, color="tab:blue", alpha=0.5,
            label="measured - true angle")
    ax.plot(grid, fit * 1e3, color="tab:red", lw=1.5,
            label=f"fit: {amp1*1e3:.1f} mrad @1/rev, "
                  f"{amp2*1e3:.1f} @2/rev")
    ax.text(0.03, 0.04,
            f"configured: e1={e1*1e3:.1f} mrad, e2={e2*1e3:.1f} mrad "
            f"(assumed, Q20)", transform=ax.transAxes, fontsize=7.5,
            color="0.35")
    ax.set_xlabel("mechanical angle [rad]")
    ax.set_ylabel("angle error [mrad]")
    ax.legend(loc="upper right", fontsize=8)
    ax.grid(alpha=0.25)

    ax = axes[1]
    ax.plot(t[m], arr(trace, "omega_rad_s")[m], lw=1.0, color="tab:blue",
            label="plant speed")
    ax.plot(t[m], arr(trace, "speed_rad_s")[m], lw=0.9, ls="--",
            color="tab:orange", label="RTL estimate")
    ax.axhline(80, color="0.4", ls=":", lw=1)
    ripple = float(np.std(arr(trace, "omega_rad_s")[m]))
    ax.set_title(f"closed-loop ripple stays bounded "
                 f"(σ = {ripple:.2f} rad/s)", fontsize=9)
    ax.set_xlabel("time [s]")
    ax.set_ylabel("speed [rad/s]")
    ax.legend(loc="upper right", fontsize=8)
    ax.grid(alpha=0.25)

    caveat(fig)
    fig.savefig(out, dpi=160, facecolor="white")
    plt.close(fig)


# ---------------------------------------------------------------------------
# pwm ripple microscopy
# ---------------------------------------------------------------------------

def fig_pwm_ripple(params, bldcsim, out):
    # Settle well past the startup overshoot (where duty clamps to zero
    # and there is no PWM to see), then capture 2 ms at 1 us resolution.
    b = bldcsim.Bench(bench_config(params, trace_interval_s=1e-6))
    b.run_for(expected_init_time(params))
    b.set_align_offset(int(params.value("rtl.sensored_align_offset")))
    b.set_target_speed(80)
    b.set_mode(2)
    b.run_for(0.95)
    assert b.duty > 0, "not at steady drive - PWM ripple premise broken"
    b.clear_trace()
    b.run_for(0.002)
    trace = b.trace
    t_us = (arr(trace, "time_s") - arr(trace, "time_s")[0]) * 1e6
    pwm_period_us = 1e6 / params.value("pwm.frequency")

    fig, (ax, axz) = plt.subplots(2, 1, figsize=(9.5, 5.6),
                                  constrained_layout=True)
    fig.suptitle("What the averaged model cannot show: per-cycle switching "
                 "ripple from the event-resolved bridge", fontsize=12)
    plot_phases(ax, t_us, trace, lw=0.9)
    ax.set_ylabel("phase current [A]")
    ax.set_xlabel("time [µs]")
    ax.legend(loc="center right", fontsize=8, ncol=3)
    ax.grid(alpha=0.25)

    # Zoom: the conducting phase with the largest mean |i| in the window.
    means = {p: float(np.mean(np.abs(arr(trace, f"i{p}_a"))))
             for p in "abc"}
    ph = max(means, key=means.get)
    color = PHASE_COLORS["abc".index(ph)]
    mz = t_us <= 6.2 * pwm_period_us
    iz = arr(trace, f"i{ph}_a")[mz]
    axz.plot(t_us[mz], iz, lw=1.1, color=color, label=f"i{ph} (zoom)")
    x0 = 2 * pwm_period_us
    y0 = float(np.max(iz)) + 0.12 * float(np.ptp(iz))
    axz.annotate("", xy=(x0, y0), xytext=(x0 + pwm_period_us, y0),
                 arrowprops=dict(arrowstyle="<->", color="0.3", lw=1))
    axz.text(x0 + pwm_period_us / 2, y0 + 0.06 * float(np.ptp(iz)),
             f"one PWM period ({pwm_period_us:.0f} µs @ "
             f"{params.value('pwm.frequency')/1e3:.0f} kHz)",
             ha="center", fontsize=8, color="0.25")
    axz.set_xlabel("time [µs]")
    axz.set_ylabel(f"i{ph} [A] (zoomed)")
    axz.legend(loc="lower right", fontsize=8)
    axz.grid(alpha=0.25)

    caveat(fig)
    fig.savefig(out, dpi=160, facecolor="white")
    plt.close(fig)


# ---------------------------------------------------------------------------
# stall raster (E21)
# ---------------------------------------------------------------------------

def fig_stall_raster(params, bldcsim, out):
    b = bldcsim.Bench(bench_config(
        params, motor={"coulomb_friction_nm": 2.0,
                       "stiction_omega_eps": 0.5}))
    b.run_for(expected_init_time(params))
    b.set_align_offset(int(params.value("rtl.sensored_align_offset")))
    b.set_target_speed(60)

    rec = {k: [] for k in ("t", "run_cmd", "stalled", "active", "duty")}

    def act(mode, seconds):
        b.set_mode(mode)
        for _ in range(int(seconds / 2e-3)):
            b.run_for(2e-3)
            rec["t"].append(b.time_s)
            rec["run_cmd"].append(mode == 2)
            rec["stalled"].append(bool(b.stalled))
            rec["active"].append(max(abs(i) for i in b.currents) > 0.05)
            rec["duty"].append(b.duty)

    act(2, 0.7)
    assert rec["stalled"][-1], "stall never latched - figure premise broken"
    act(0, 0.1)
    act(2, 0.7)
    assert rec["stalled"][-1], "re-stall not detected"

    t = np.asarray(rec["t"])
    stall_timeout = params.value("rtl.stall_timeout")

    fig, (ax0, ax1) = plt.subplots(2, 1, sharex=True, figsize=(9.5, 5.2),
                                   constrained_layout=True,
                                   height_ratios=[1.2, 1])
    fig.suptitle("Jammed rotor: stall detected, gates killed, host "
                 "acknowledges, re-stall re-detected", fontsize=12)

    ax0.plot(t, np.asarray(rec["duty"]) / 625.0 * 100.0,
             color="tab:purple", lw=1.2, label="PWM duty")
    ax0.annotate(f"duty winds to the rail while the rotor\nnever moves; "
                 f"after {stall_timeout:.1f} s the detector latches",
                 xy=(float(t[0]) + stall_timeout, 80),
                 xytext=(float(t[0]) + stall_timeout + 0.15, 45),
                 fontsize=8, color="0.25",
                 arrowprops=dict(arrowstyle="->", color="0.3"))
    ax0.set_ylabel("PWM duty [%]")
    ax0.legend(loc="center right", fontsize=8)
    ax0.grid(alpha=0.25)

    lanes = (("run commanded", "run_cmd", "tab:blue"),
             ("power stage active", "active", "tab:green"),
             ("STALL latched", "stalled", "tab:red"))
    for row, (label, key, color) in enumerate(lanes):
        y0 = len(lanes) - 1 - row
        v = np.asarray(rec[key], dtype=float)
        ax1.fill_between(t, y0 + 0.1, y0 + 0.1 + 0.8 * v, step="post",
                         color=color, alpha=0.7, linewidth=0)
        ax1.text(t[0] - 0.015, y0 + 0.5, label, ha="right", va="center",
                 fontsize=8.5)
    ax1.set_ylim(0, len(lanes))
    ax1.set_yticks([])
    ax1.set_xlabel("time [s]")
    ax1.set_xlim(left=t[0] - 0.35)
    ax1.grid(alpha=0.25, axis="x")

    caveat(fig)
    fig.savefig(out, dpi=160, facecolor="white")
    plt.close(fig)


# ---------------------------------------------------------------------------
# dead-time microscopy
# ---------------------------------------------------------------------------

def fig_deadtime(params, bldcsim, out):
    # Six-step drive is not complementary within a sector (the freewheel
    # path is the body diode), so leg handoffs - and their dead-time gaps -
    # happen at sector boundaries. Capture 4 ms at 200 ns resolution to
    # guarantee at least one, and search both handoff directions.
    b = steady_bench(params, bldcsim, target=80, seconds=0.95)
    step_cycles = 5
    clk = params.value("rtl.clock_frequency")
    dt_s = step_cycles / clk
    n = 20000  # 4 ms at 200 ns resolution: spans a sector boundary
    gh = np.zeros((3, n), dtype=bool)
    gl = np.zeros((3, n), dtype=bool)
    for i in range(n):
        b.run_cycles(step_cycles)
        hi, lo = b.gates
        for p in range(3):
            gh[p, i] = hi[p]
            gl[p, i] = lo[p]

    # A handoff = one gate of a leg falls, the opposite gate rises shortly
    # after, with an all-off gap in between.
    max_gap = int(round(12e-6 / dt_s))
    evt = None  # (phase, fall_idx, rise_idx)
    for p in range(3):
        for first, second in ((gh, gl), (gl, gh)):
            falls = np.flatnonzero(first[p, :-1] & ~first[p, 1:])
            for f in falls:
                seg = second[p, f + 1:f + 1 + max_gap]
                rises = np.flatnonzero(seg)
                if rises.size:
                    cand = (p, int(f), int(f) + 1 + int(rises[0]))
                    if evt is None or cand[2] - cand[1] < evt[2] - evt[1]:
                        evt = cand
                    break
    assert evt is not None, "no leg handoff captured in 4 ms"
    p, f, r = evt
    lo_i = max(0, f - 20)
    hi_i = min(n, r + 30)
    t_us = (np.arange(lo_i, hi_i) - f) * dt_s * 1e6

    fig, ax = plt.subplots(figsize=(9.5, 4.6), constrained_layout=True)
    gap_us = (r - f) * dt_s * 1e6
    fig.suptitle("Gate handoff under the microscope: the dead-time gap the "
                 "shoot-through checker polices every edge", fontsize=12)

    names = []
    for row, (sig, label) in enumerate(
            [(gh[0], "AH"), (gl[0], "AL"), (gh[1], "BH"), (gl[1], "BL"),
             (gh[2], "CH"), (gl[2], "CL")]):
        y0 = 10 - 2 * row
        ax.step(t_us, y0 + 1.5 * sig[lo_i:hi_i], where="post",
                color=SECTOR_COLORS(row // 2), lw=1.4)
        names.append((y0, label))
    for y0, label in names:
        ax.text(t_us[0] - 0.4, y0 + 0.75, label, ha="right", va="center",
                fontsize=9)
    ax.axvspan(0, gap_us, color="tab:red", alpha=0.12, linewidth=0)
    ax.annotate(
        f"both gates low: {gap_us:.1f} µs gap\n(config "
        f"{params.value('pwm.rtl_dead_time')*1e6:.1f} µs; checker-"
        f"measured min this run: {b.min_dead_time_s*1e6:.2f} µs)",
        xy=(gap_us / 2, 6), xytext=(gap_us + 1.5, 5.2), fontsize=8,
        color="0.25", arrowprops=dict(arrowstyle="->", color="0.3"))
    ax.set_yticks([])
    ax.set_xlabel(f"µs relative to the phase-{'ABC'[p]} leg handoff "
                  "(sector boundary)")
    ax.grid(alpha=0.25, axis="x")

    caveat(fig)
    fig.savefig(out, dpi=160, facecolor="white")
    plt.close(fig)


# ---------------------------------------------------------------------------
# FOC figures (foc-checklist stage 8)
# ---------------------------------------------------------------------------

def _foc_align(params):
    return int(params.value("foc.align_offset"))


def _lsb_per_amp(params):
    return (params.value("drv8301.amp_gain")
            * params.value("feedback.current.shunt")
            * 4096.0 / params.value("adc.vref"))


def fig_foc_startup(params, bldcsim, out):
    import foc_reference as fr
    pp = int(params.value("motor.pole_pairs"))
    lpa = _lsb_per_amp(params)
    b = bldcsim.Bench(bench_config(params, trace_interval_s=2e-4,
                                   motor={"trapezoid_blend": 0.0}))
    b.run_for(expected_init_time(params))
    t_init = b.time_s
    b.set_align_offset(_foc_align(params))
    b.set_id_target(0)
    b.set_foc_speed_loop(True)
    b.set_target_speed(80)
    b.set_mode(3)
    # Sample id/iq alongside the trace by stepping in small windows.
    ts, omega, idq_d, idq_q = [], [], [], []
    for _ in range(600):
        b.run_for(2e-3)
        ts.append(b.time_s)
        omega.append(b.omega)
        id_, iq = fr.park(*fr.clarke(b.foc_cur_a / lpa, b.foc_cur_b / lpa),
                          pp * b.theta)
        idq_d.append(id_)
        idq_q.append(iq)
    t = np.asarray(ts)

    fig, axes = plt.subplots(2, 1, sharex=True, figsize=(9.5, 5.6),
                             constrained_layout=True)
    fig.suptitle("Field-oriented control spinning the simulated PMSM: speed "
                 "loop -> iq*, inner current loop holds id = 0", fontsize=12)
    ax = axes[0]
    ax.plot(t, omega, color="tab:blue", lw=1.4, label="plant speed")
    ax.axhline(80, color="0.4", ls=":", lw=1, label="target (80 rad/s)")
    ax.axvspan(0, t_init, color="0.5", alpha=0.15, linewidth=0)
    ax.set_ylabel("speed [rad/s]")
    ax.legend(loc="lower right", fontsize=8)
    ax.grid(alpha=0.25)
    ax = axes[1]
    ax.plot(t, idq_q, color="tab:green", lw=1.0, label="iq (torque current)")
    ax.plot(t, idq_d, color="tab:red", lw=1.0, label="id (flux current)")
    ax.axhline(0, color="0.4", ls=":", lw=1)
    ax.set_ylabel("dq current [A]")
    ax.set_xlabel("time [s]")
    ax.legend(loc="upper right", fontsize=8)
    ax.grid(alpha=0.25)
    fig.text(0.99, 0.005,
             "FOC: id held at 0 (all current makes torque). " + CAVEAT,
             ha="right", va="bottom", fontsize=6.5, color="0.45",
             style="italic")
    fig.savefig(out, dpi=160, facecolor="white")
    plt.close(fig)


def fig_foc_sampling(params, bldcsim, out):
    """Q21: dq measurement error of the two sampling schemes over a window."""
    import foc_reference as fr
    pp = int(params.value("motor.pole_pairs"))
    clk = params.value("rtl.clock_frequency")
    lpa = _lsb_per_amp(params)

    def collect(scheme):
        b = bldcsim.Bench(foc(params, sample_scheme=scheme))
        b.run_for(expected_init_time(params))
        b.set_open_loop(freq_word(20.0, clk), 1 << 20)
        b.set_duty(int(0.45 * 625))
        b.set_mode(1)
        b.set_foc_sample(True)
        b.run_for(0.25)
        ts, err = [], []
        for _ in range(400):
            b.run_for(2e-4)
            te = pp * b.theta
            tid, tiq = fr.park(*fr.clarke(*b.currents[:2]), te)
            mid, miq = fr.park(*fr.clarke(b.foc_cur_a / lpa,
                                          b.foc_cur_b / lpa), te)
            ts.append(b.time_s)
            err.append(math.hypot(mid - tid, miq - tiq))
        return np.asarray(ts), np.asarray(err)

    t0, e0 = collect(0)
    t1, e1 = collect(1)
    fig, ax = plt.subplots(figsize=(9.5, 4.0), constrained_layout=True)
    fig.suptitle("Why FOC needs simultaneous current sampling (Q21): dq "
                 "measurement error per scheme", fontsize=12)
    ax.plot((t1 - t1[0]) * 1e3, e1, color="tab:red", lw=0.9,
            label=f"sequential single-ADC (mean {np.mean(e1):.2f} A)")
    ax.plot((t0 - t0[0]) * 1e3, e0, color="tab:blue", lw=0.9,
            label=f"simultaneous (dual ADC / S&H, mean {np.mean(e0):.2f} A)")
    ax.set_ylabel("dq current error [A]")
    ax.set_xlabel("time [ms]")
    ax.legend(loc="upper right", fontsize=8)
    ax.grid(alpha=0.25)
    ax.set_title("the sequential ADC samples phase B ~22 us late, after its "
                 "low-side conduction window closes", fontsize=9, color="0.3")
    caveat(fig)
    fig.savefig(out, dpi=160, facecolor="white")
    plt.close(fig)


def fig_foc_latency(params, bldcsim, out):
    """Q22: torque vs speed, AS5600 angle latency, extrapolation off vs on."""
    ke = params.value("motor.Ke")
    pp = int(params.value("motor.pole_pairs"))

    def torque_at(omega, extrap):
        b = bldcsim.Bench(foc(params, "sensor"))
        b.run_for(expected_init_time(params))
        b.set_speed_clamp(True, omega)
        b.set_align_offset(_foc_align(params))
        b.set_foc_extrap(extrap)
        b.set_id_target(0)
        b.set_iq_target(80)
        b.set_mode(3)
        b.run_for(0.1)
        ts = []
        for _ in range(150):
            b.run_for(2e-4)
            te = pp * b.theta
            ia, ib, ic = b.currents
            ts.append(ke * (math.sin(te) * ia
                            + math.sin(te - 2 * math.pi / 3) * ib
                            + math.sin(te - 4 * math.pi / 3) * ic))
        return float(np.mean(ts))

    speeds = [20, 40, 60, 80, 100, 120]
    t_off = [torque_at(w, False) for w in speeds]
    t_on = [torque_at(w, True) for w in speeds]
    fig, ax = plt.subplots(figsize=(9.5, 4.0), constrained_layout=True)
    fig.suptitle("AS5600 angle latency costs FOC torque as speed rises - and "
                 "extrapolation recovers it (Q22)", fontsize=12)
    ax.plot(speeds, t_off, "o-", color="tab:red", lw=1.4,
            label="raw sensor angle")
    ax.plot(speeds, t_on, "s-", color="tab:blue", lw=1.4,
            label="omega-latency extrapolation")
    ax.set_xlabel("clamped speed [rad/s]")
    ax.set_ylabel("developed torque [N·m]\n(fixed iq* = 80 LSB)")
    ax.legend(loc="lower left", fontsize=8)
    ax.grid(alpha=0.25)
    caveat(fig)
    fig.savefig(out, dpi=160, facecolor="white")
    plt.close(fig)


# ---------------------------------------------------------------------------
# three-way plant parity
# ---------------------------------------------------------------------------

def fig_parity(params, bldcsim, out):
    from run_three_phase_reference import (BridgeParams, motor_from_config,
                                           scenario_from_config, simulate)

    prefix = "scenario.three_phase_open_loop"
    motor = {
        "resistance_ohm": params.value("motor.R"),
        "inductance_h": params.value("motor.L"),
        "ke_v_s_per_rad": params.value("motor.Ke"),
        "inertia_kg_m2": params.value("motor.J"),
        "damping_n_m_s_per_rad": params.value("motor.B"),
        "pole_pairs": int(params.value("motor.pole_pairs")),
        "trapezoid_blend": params.value("motor.emf_trapezoid_blend"),
        "load_torque_n_m": params.value(f"{prefix}.load_torque"),
    }
    bridge = {
        "vbus_v": params.value("bus.vbus"),
        "fet_rds_on_ohm": params.value("inverter.fet_rds_on"),
        "diode_vf_v": params.value("inverter.body_diode_vf"),
    }
    config = {
        "current_epsilon_a": params.value("sim.current_epsilon"),
        "max_substep_s": params.value(f"{prefix}.max_substep"),
    }
    cpp = bldcsim.simulate_three_phase_open_loop(
        motor, bridge, config,
        params.value(f"{prefix}.duty"),
        params.value(f"{prefix}.f_elec_final"),
        params.value(f"{prefix}.ramp_time"),
        params.value(f"{prefix}.stop"),
        params.value(f"{prefix}.dt"))

    rows = simulate(motor_from_config(params),
                    BridgeParams(vbus_v=params.value("bus.vbus")),
                    **scenario_from_config(params))
    py = {k: [r[k] for r in rows] for k in rows[0]}

    oracle = None
    if shutil.which("omc"):
        from run_three_phase_oracle import run_oracle
        oracle = run_oracle(params, PROJECT_ROOT / "sim" / "build"
                            / "oracle_fig")
    else:
        print("WARNING: omc not found - parity figure drawn without the "
              "Modelica oracle")

    tc = arr(cpp, "time_s")
    ia_c = arr(cpp, "ia_a")
    rms = float(np.sqrt(np.mean(ia_c ** 2)))
    res_py = np.abs(arr(py, "ia_a") - ia_c) / rms * 100.0
    if oracle is not None:
        ia_o = np.interp(tc, arr(oracle, "time_s"), arr(oracle, "ia_a"))
        res_or = np.abs(ia_o - ia_c) / rms * 100.0

    fig, axes = plt.subplots(3, 1, figsize=(9.5, 7.0),
                             constrained_layout=True)
    fig.suptitle("The same physics three times on purpose: C++ bench vs "
                 "Python executable spec vs Modelica oracle", fontsize=12)

    ax = axes[0]
    ax.plot(tc, arr(cpp, "omega_rad_s"), color="tab:blue", lw=1.6,
            label="C++ (bench)")
    ax.plot(tc, arr(py, "omega_rad_s"), color="tab:orange", lw=1.0,
            ls="--", label="Python (spec)")
    if oracle is not None:
        ax.plot(arr(oracle, "time_s"), arr(oracle, "omega_rad_s"),
                color="tab:green", lw=1.0, ls=":", label="Modelica (dassl)")
    ax.set_ylabel("speed [rad/s]")
    ax.legend(loc="lower right", fontsize=8)
    ax.set_title("one indistinguishable trajectory (open-loop ramp "
                 "scenario)", fontsize=9)
    ax.grid(alpha=0.25)

    ax = axes[1]
    mwin = (tc >= 0.10) & (tc <= 0.13)
    ax.plot(tc[mwin] * 1e3, ia_c[mwin], color="tab:blue", lw=1.6)
    ax.plot(tc[mwin] * 1e3, arr(py, "ia_a")[mwin], color="tab:orange",
            lw=1.0, ls="--")
    if oracle is not None:
        ax.plot(tc[mwin] * 1e3, ia_o[mwin], color="tab:green", lw=1.0,
                ls=":")
    ax.set_ylabel("ia [A]")
    ax.set_xlabel("time [ms] (zoom)")
    ax.grid(alpha=0.25)

    ax = axes[2]
    ax.semilogy(tc, np.maximum(res_py, 1e-10), color="tab:orange", lw=0.9,
                label="|Python - C++| (shared integrator)")
    if oracle is not None:
        ax.semilogy(tc, np.maximum(res_or, 1e-10), color="tab:green",
                    lw=0.9, label="|Modelica - C++| (independent tool, "
                                  "event-located dassl)")
    ax.set_ylabel("ia residual [% of RMS]")
    ax.set_xlabel("time [s]")
    ax.set_ylim(1e-10, 50)
    ax.legend(loc="upper right", fontsize=8)
    ax.grid(alpha=0.25, which="both")

    caveat(fig)
    fig.savefig(out, dpi=160, facecolor="white")
    plt.close(fig)


# ---------------------------------------------------------------------------
# main
# ---------------------------------------------------------------------------

def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", type=Path,
                        default=PROJECT_ROOT / "figures")
    parser.add_argument("--only", type=str, default=None,
                        help="comma-separated subset of figure names")
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)

    all_names = ["startup", "commutation", "gif", "brownout", "regen",
                 "adc_chain", "thermal", "cogging", "eccentricity",
                 "pwm_ripple", "stall_raster", "deadtime", "parity",
                 "foc_startup", "foc_sampling", "foc_latency"]
    only = set(args.only.split(",")) if args.only else set(all_names)
    unknown = only - set(all_names)
    if unknown:
        raise SystemExit(f"unknown figure name(s): {sorted(unknown)}")

    params = sim_params.load(None)
    print(params.banner_text())
    try:
        import bldcsim
    except ImportError:
        raise SystemExit("bldcsim module not built; run "
                         "sim/scripts/build_bench.sh first")

    done = []

    if only & {"startup", "commutation", "gif"}:
        print("running closed-loop startup scenario...")
        trace, t_init = run_startup(params, bldcsim)
        if "startup" in only:
            fig_startup(trace, t_init, params, args.out / "startup.png")
            done.append("startup.png")
        if "commutation" in only:
            fig_commutation(trace, args.out / "commutation.png")
            done.append("commutation.png")
        if "gif" in only:
            print("rendering animation...")
            fig_gif(trace, params, args.out / "motorloop.gif")
            done.append("motorloop.gif")

    if "brownout" in only:
        print("running brownout scenario (supply realism)...")
        trace, t_idle = run_brownout(params, bldcsim)
        fig_brownout(trace, t_idle, params, args.out / "brownout.png")
        done.append("brownout.png")

    simple = [("regen", fig_regen, "regen pump-up"),
              ("adc_chain", fig_adc_chain, "ADC chain"),
              ("thermal", fig_thermal, "locked-rotor thermal"),
              ("cogging", fig_cogging, "cogging startup"),
              ("eccentricity", fig_eccentricity, "sensor eccentricity"),
              ("pwm_ripple", fig_pwm_ripple, "PWM ripple"),
              ("stall_raster", fig_stall_raster, "stall raster"),
              ("deadtime", fig_deadtime, "dead-time microscopy"),
              ("parity", fig_parity, "three-way parity"),
              ("foc_startup", fig_foc_startup, "FOC startup"),
              ("foc_sampling", fig_foc_sampling, "FOC sampling (Q21)"),
              ("foc_latency", fig_foc_latency, "FOC angle latency (Q22)")]
    for name, fn, desc in simple:
        if name in only:
            print(f"running {desc} scenario...")
            fn(params, bldcsim, args.out / f"{name}.png")
            done.append(f"{name}.png")

    for f in done:
        print(f"wrote {args.out / f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
