#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
"""Render the part-comparison study figures from live bench runs
(notes/part-comparison-checklist.md §3).

Every figure is produced by re-running the experiments in
sim/tests/part_compare.py (the same runners the pytest suite asserts on) and
plotting the result - no mockups. Figures (-> figures/comparison/):

  t1_latency.png        angle error vs speed, AS5600 vs AS5047P
  t2_reversal.png       reversal waveform + the max-reversal-speed cliff
  t3_skew.png           dq current ripple vs di/dt, MCP3208 vs ADS9224R
  t4_noise_floor.png    measurement-noise spectra (FFT), sequential vs simultaneous
  t5_snap.png           commutation integrity through a full-speed snap reversal
  t6_phase_margin.png   commutation lag vs speed + max locked speed
  t7_resolution.png     delivered angular-resolution staircase + LSB
  t8_penalty.png        torque efficiency lost to angle latency
  t9_dirty.png          realism layers: what an IC upgrade does (and doesn't) fix
  t10_envelope.png      operating envelope, speed x reversal abruptness

Standing caveat (every figure): placeholder motor parameters (Q1), simulation
against the device models - relative comparisons, not hardware predictions.

Usage:
  python3 sim/scripts/gen_comparison_figures.py [--out figures/comparison]
      [--only t1_latency,t10_envelope]
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from matplotlib.colors import ListedColormap  # noqa: E402

PROJECT_ROOT = Path(__file__).resolve().parents[2]
for sub in ("sim/scripts", "sim/tests", "sim/build/cpp"):
    sys.path.insert(0, str(PROJECT_ROOT / sub))

import bldcsim  # noqa: E402
import part_compare as PC  # noqa: E402
import part_metrics as M  # noqa: E402
import sim_params  # noqa: E402

CAVEAT = ("simulated output, placeholder motor parameters (Q1) - relative "
          "model-vs-model comparison, not a hardware prediction")
A_COLOR, B_COLOR = "tab:red", "tab:green"     # cheaper part / premium part
RPM = 60.0 / (2.0 * np.pi)                      # rad/s -> rev/min


def caveat(fig):
    fig.text(0.99, 0.004, CAVEAT, ha="right", va="bottom", fontsize=6.5,
             color="0.45", style="italic")


def finish(fig, out, name):
    caveat(fig)
    path = Path(out) / f"{name}.png"
    fig.savefig(path, dpi=130)
    plt.close(fig)
    print(f"  wrote {path}")


# --------------------------------------------------------------------------- #
def fig_t1_latency(params, out):
    om = [20, 60, 120, 240, 400, 600, 800, 1000, 1300]
    a = PC.run_speed_sweep(params, bldcsim, PC.SENSOR_A, om, n=200)
    s = PC.run_speed_sweep(params, bldcsim, PC.SENSOR_B, om, n=200)
    fig, ax = plt.subplots(figsize=(8.4, 5.0), constrained_layout=True)
    rpm = a["omega"] * RPM
    ax.plot(rpm, a["rms_deg"], "-o", color=A_COLOR, label="AS5600 (PWM, ~90 us)")
    ax.plot(rpm, s["rms_deg"], "-o", color=B_COLOR,
            label="AS5047P (SPI 14-bit, DAEC)")
    ax.axhline(30, ls="--", color="0.5", lw=0.9)
    ax.axhline(90, ls="--", color="0.3", lw=0.9)
    ax.text(rpm[0], 32, "30 deg: degraded commutation", fontsize=7, color="0.4")
    ax.text(rpm[0], 95, "90 deg: torque inverts", fontsize=7, color="0.3")
    ax.set_xlabel("mechanical speed (rev/min, placeholder motor)")
    ax.set_ylabel("commutation angle error, RMS (electrical deg)")
    ax.set_title("T1  Angle latency: measured-vs-true error grows with speed")
    ax.legend(loc="upper left")
    ax.grid(alpha=0.3)
    finish(fig, out, "t1_latency")


def fig_t2_reversal(params, out):
    speeds = [100, 200, 400, 700, 1000, 1400, 1800]
    a = PC.run_reversal(params, bldcsim, PC.SENSOR_A, speeds, transition_s=0.01)
    s = PC.run_reversal(params, bldcsim, PC.SENSOR_B, speeds, transition_s=0.01)
    wa = PC.run_reversal_waveform(params, bldcsim, PC.SENSOR_A, omega=400.0)
    wb = PC.run_reversal_waveform(params, bldcsim, PC.SENSOR_B, omega=400.0)
    fig, (axw, axc) = plt.subplots(1, 2, figsize=(11.5, 4.8),
                                   constrained_layout=True)
    axw.plot(wa["t"] * 1e3, wa["err_deg"], color=A_COLOR, label="AS5600")
    axw.plot(wb["t"] * 1e3, wb["err_deg"], color=B_COLOR, label="AS5047P")
    axw.axhline(90, ls="--", color="0.3", lw=0.9)
    axw.axhline(-90, ls="--", color="0.3", lw=0.9)
    axw.set_xlabel("time through a +/-400 rad/s reversal (ms)")
    axw.set_ylabel("commutation error (electrical deg)")
    axw.set_title("Reversal waveform")
    axw.legend(loc="upper right")
    axw.grid(alpha=0.3)
    axc.bar([0, 1], [a["cliff_speed"] * RPM, s["cliff_speed"] * RPM],
            color=[A_COLOR, B_COLOR])
    axc.set_xticks([0, 1])
    axc.set_xticklabels(["AS5600", "AS5047P"])
    axc.set_ylabel("max reversal speed before inversion (rev/min)")
    axc.set_title("The reversal cliff")
    for i, v in enumerate([a["cliff_speed"], s["cliff_speed"]]):
        axc.text(i, v * RPM, f"{v * RPM:.0f}", ha="center", va="bottom")
    fig.suptitle("T2  How fast can you slam forward->reverse without breaking",
                 fontsize=12)
    finish(fig, out, "t2_reversal")


def fig_t3_skew(params, out):
    om = [40, 80, 160, 320, 480, 640]
    sim = PC.run_skew_sweep(params, bldcsim, PC.SCHEME_SIMULTANEOUS, om, n=400)
    seq = PC.run_skew_sweep(params, bldcsim, PC.SCHEME_SEQUENTIAL, om, n=400)
    fig, ax = plt.subplots(figsize=(8.4, 5.0), constrained_layout=True)
    ax.plot(sim["omega"] * RPM, sim["ripple"], "-o", color=B_COLOR,
            label=PC.SCHEME_LABEL[PC.SCHEME_SIMULTANEOUS])
    ax.plot(seq["omega"] * RPM, seq["ripple"], "-o", color=A_COLOR,
            label=PC.SCHEME_LABEL[PC.SCHEME_SEQUENTIAL])
    ax.set_xlabel("speed (rev/min) -> phase-current di/dt grows with speed")
    ax.set_ylabel("dq current ripple injected by sampling (FOC LSB, RMS)")
    ax.set_title("T3  Current-sample skew: sequential single-ADC vs simultaneous")
    ax.legend(loc="upper left")
    ax.grid(alpha=0.3)
    finish(fig, out, "t3_skew")


def fig_t4_noise_floor(params, out):
    sim = PC.run_skew_spectrum(params, bldcsim, PC.SCHEME_SIMULTANEOUS, n=4096)
    seq = PC.run_skew_spectrum(params, bldcsim, PC.SCHEME_SEQUENTIAL, n=4096)
    fig, ax = plt.subplots(figsize=(8.4, 5.0), constrained_layout=True)
    ax.semilogx(seq["freqs"][1:], seq["psd_db"][1:], color=A_COLOR, lw=0.8,
                label=PC.SCHEME_LABEL[PC.SCHEME_SEQUENTIAL])
    ax.semilogx(sim["freqs"][1:], sim["psd_db"][1:], color=B_COLOR, lw=0.8,
                label=PC.SCHEME_LABEL[PC.SCHEME_SIMULTANEOUS])
    ax.axhline(seq["floor_db"], ls="--", color=A_COLOR, lw=0.8)
    ax.axhline(sim["floor_db"], ls="--", color=B_COLOR, lw=0.8)
    ax.set_xlabel("frequency (Hz)")
    ax.set_ylabel("measured id current PSD (dB)")
    ax.set_title("T4  Measurement-noise floor of the current feedback")
    ax.legend(loc="upper right")
    ax.grid(alpha=0.3, which="both")
    finish(fig, out, "t4_noise_floor")


def fig_t5_snap(params, out):
    a = PC.run_snap(params, bldcsim, PC.SENSOR_A, omega=300.0)
    s = PC.run_snap(params, bldcsim, PC.SENSOR_B, omega=300.0)
    fig, (ax0, ax1) = plt.subplots(2, 1, figsize=(9.0, 6.4), sharex=True,
                                   constrained_layout=True)
    for r, c, lab in ((a, A_COLOR, "AS5600"), (s, B_COLOR, "AS5047P")):
        ax0.plot(r["t"] * 1e3, r["err_deg"], color=c, label=lab)
        ax1.plot(r["t"] * 1e3, r["ia"], color=c, label=lab, lw=0.8)
    for ax in (ax0, ax1):
        ax.axvline(a["t_flip"] * 1e3, ls=":", color="0.4")
        ax.grid(alpha=0.3)
    ax0.axhline(90, ls="--", color="0.3", lw=0.8)
    ax0.axhline(-90, ls="--", color="0.3", lw=0.8)
    ax0.set_ylabel("commutation error (elec deg)")
    ax0.set_title("T5  Full-speed snap reversal: commutation integrity")
    ax0.legend(loc="upper right")
    ax1.set_ylabel("phase-A current (A)")
    ax1.set_xlabel("time (ms); dotted = the +300 -> -300 rad/s snap")
    note = (f"peak |I|: AS5600 {a['peak_current']:.1f} A, "
            f"AS5047P {s['peak_current']:.1f} A   |   faults latched: "
            f"AS5600 {a['faulted']}, AS5047P {s['faulted']} "
            f"(stiff-supply default)")
    ax1.text(0.01, 0.02, note, transform=ax1.transAxes, fontsize=7,
             color="0.35")
    finish(fig, out, "t5_snap")


def fig_t6_phase_margin(params, out):
    om = [20, 60, 120, 240, 400, 600, 800, 1000, 1300]
    a = PC.run_speed_sweep(params, bldcsim, PC.SENSOR_A, om, n=200)
    s = PC.run_speed_sweep(params, bldcsim, PC.SENSOR_B, om, n=200)
    fig, (axm, axb) = plt.subplots(1, 2, figsize=(11.5, 4.8),
                                   constrained_layout=True)
    axm.plot(a["omega"] * RPM, a["mean_deg"], "-o", color=A_COLOR,
             label="AS5600")
    axm.plot(s["omega"] * RPM, s["mean_deg"], "-o", color=B_COLOR,
             label="AS5047P")
    axm.axhline(-90, ls="--", color="0.3", lw=0.9)
    axm.set_xlabel("speed (rev/min)")
    axm.set_ylabel("mean commutation lag (electrical deg)")
    axm.set_title("Phase-margin erosion")
    axm.legend(loc="lower left")
    axm.grid(alpha=0.3)
    mla, mlb = PC.max_locked_speed(a), PC.max_locked_speed(s)
    axb.bar([0, 1], [mla * RPM, mlb * RPM], color=[A_COLOR, B_COLOR])
    axb.set_xticks([0, 1])
    axb.set_xticklabels(["AS5600", "AS5047P"])
    axb.set_ylabel("max speed holding commutation (rev/min)")
    axb.set_title("Max locked speed")
    for i, v in enumerate([mla, mlb]):
        axb.text(i, v * RPM, f"{v * RPM:.0f}", ha="center", va="bottom")
    fig.suptitle("T6  Commutation phase margin vs speed", fontsize=12)
    finish(fig, out, "t6_phase_margin")


def fig_t7_resolution(params, out):
    a = PC.run_resolution(params, bldcsim, PC.SENSOR_A, n=4000)
    s = PC.run_resolution(params, bldcsim, PC.SENSOR_B, n=4000)
    fig, (axs, axb) = plt.subplots(1, 2, figsize=(11.5, 4.8),
                                   constrained_layout=True)
    w = slice(0, 400)
    axs.plot(a["t"][w] * 1e3, a["enc_deg"][w] - a["enc_deg"][w][0],
             color=A_COLOR, label="AS5600 measured", lw=0.9)
    axs.plot(s["t"][w] * 1e3, s["enc_deg"][w] - s["enc_deg"][w][0],
             color=B_COLOR, label="AS5047P measured", lw=0.9)
    axs.plot(a["t"][w] * 1e3, a["truth_deg"][w] - a["truth_deg"][w][0],
             color="0.3", ls=":", label="true angle", lw=0.9)
    axs.set_xlabel("time (ms), slow rotation")
    axs.set_ylabel("angle (mechanical deg, zeroed)")
    axs.set_title("Delivered-resolution staircase")
    axs.legend(loc="upper left")
    axs.grid(alpha=0.3)
    axb.bar([0, 1], [a["lsb_deg"], s["lsb_deg"]], color=[A_COLOR, B_COLOR])
    axb.set_xticks([0, 1])
    axb.set_xticklabels(["AS5600", "AS5047P"])
    axb.set_ylabel("delivered angular LSB (mechanical deg)")
    axb.set_title(f"~{a['lsb_deg'] / max(s['lsb_deg'], 1e-9):.1f}x finer "
                  "(not the full 4x: RTL truncates 14->12 bit)")
    for i, v in enumerate([a["lsb_deg"], s["lsb_deg"]]):
        axb.text(i, v, f"{v:.4f}", ha="center", va="bottom")
    fig.suptitle("T7  Smallest resolvable move", fontsize=12)
    finish(fig, out, "t7_resolution")


def fig_t8_penalty(params, out):
    om = [20, 60, 120, 240, 400, 600, 800, 1000, 1300]
    a = PC.run_speed_sweep(params, bldcsim, PC.SENSOR_A, om, n=200)
    s = PC.run_speed_sweep(params, bldcsim, PC.SENSOR_B, om, n=200)
    fig, ax = plt.subplots(figsize=(8.4, 5.0), constrained_layout=True)
    ax.plot(a["omega"] * RPM, a["torque_eff"] * 100, "-o", color=A_COLOR,
            label="AS5600")
    ax.plot(s["omega"] * RPM, s["torque_eff"] * 100, "-o", color=B_COLOR,
            label="AS5047P")
    ax.axhline(0, color="0.3", lw=0.9)
    ax.fill_between(a["omega"] * RPM, a["torque_eff"] * 100, 0,
                    where=(a["torque_eff"] < 0), color=A_COLOR, alpha=0.12)
    ax.set_xlabel("speed (rev/min)")
    ax.set_ylabel("torque efficiency  mean(cos(angle error))  (%)")
    ax.set_title("T8  Torque lost to angle latency (cos<0 = the loop brakes)")
    ax.legend(loc="lower left")
    ax.grid(alpha=0.3)
    finish(fig, out, "t8_penalty")


def fig_t9_dirty(params, out):
    a = PC.run_dirty_bench(params, bldcsim, PC.SENSOR_A, n=2000)
    s = PC.run_dirty_bench(params, bldcsim, PC.SENSOR_B, n=2000)
    layers = ["clean", "eccentricity", "disturbance", "all"]
    av = [a["layers"][k] for k in layers]
    sv = [s["layers"][k] for k in layers]
    x = np.arange(len(layers))
    fig, ax = plt.subplots(figsize=(8.8, 5.0), constrained_layout=True)
    ax.bar(x - 0.18, av, 0.36, color=A_COLOR, label="AS5600")
    ax.bar(x + 0.18, sv, 0.36, color=B_COLOR, label="AS5047P")
    ax.set_xticks(x)
    ax.set_xticklabels(["clean", "eccentricity\n(mechanical)",
                        "disturbance\n(electrical)", "all"])
    ax.set_ylabel("angle-error std (electrical deg)")
    ax.set_title("T9  What a better sensor IC fixes - and what it can't")
    ax.legend(loc="upper left")
    ax.grid(alpha=0.3, axis="y")
    ax.text(0.99, 0.97, "eccentricity hits BOTH ~equally (a mounting problem);\n"
            "the digital angle path is immune to the electrical disturbance",
            transform=ax.transAxes, ha="right", va="top", fontsize=7.5,
            color="0.3")
    finish(fig, out, "t9_dirty")


def fig_t10_envelope(params, out):
    speeds = [100, 300, 500, 800, 1100, 1500]
    trans = [0.03, 0.012, 0.005, 0.002]
    a = PC.run_envelope(params, bldcsim, PC.SENSOR_A, speeds, trans)
    s = PC.run_envelope(params, bldcsim, PC.SENSOR_B, speeds, trans)
    cmap = ListedColormap(["#c44", "#4a4"])   # broke / locked
    fig, axes = plt.subplots(1, 2, figsize=(11.5, 4.8),
                             constrained_layout=True)
    for ax, r, title in ((axes[0], a, "AS5600"), (axes[1], s, "AS5047P")):
        ax.imshow(r["locked"].astype(int), cmap=cmap, vmin=0, vmax=1,
                  aspect="auto", origin="lower")
        ax.set_xticks(range(len(speeds)))
        ax.set_xticklabels([f"{int(v * RPM)}" for v in speeds], fontsize=8)
        ax.set_yticks(range(len(trans)))
        ax.set_yticklabels([f"{t * 1e3:.0f}" for t in trans])
        ax.set_xlabel("reversal peak speed (rev/min)")
        ax.set_ylabel("reversal time (ms)")
        ax.set_title(f"{title}  - locked {r['locked_frac'] * 100:.0f}%")
    fig.suptitle("T10  Operating envelope: green = reverses cleanly, red = "
                 "commutation breaks", fontsize=12)
    finish(fig, out, "t10_envelope")


FIGURES = {
    "t1_latency": fig_t1_latency, "t2_reversal": fig_t2_reversal,
    "t3_skew": fig_t3_skew, "t4_noise_floor": fig_t4_noise_floor,
    "t5_snap": fig_t5_snap, "t6_phase_margin": fig_t6_phase_margin,
    "t7_resolution": fig_t7_resolution, "t8_penalty": fig_t8_penalty,
    "t9_dirty": fig_t9_dirty, "t10_envelope": fig_t10_envelope,
}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=str(PROJECT_ROOT / "figures" / "comparison"))
    ap.add_argument("--only", default="")
    args = ap.parse_args()
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    params = sim_params.load()
    names = [n.strip() for n in args.only.split(",") if n.strip()] or FIGURES
    for name in names:
        if name not in FIGURES:
            raise SystemExit(f"unknown figure {name!r}; known: {list(FIGURES)}")
        print(f"rendering {name} ...")
        FIGURES[name](params, out)
    print(f"done -> {out}")


if __name__ == "__main__":
    main()
