#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
"""Render the open ADS9224R module figures (board checklist §8).

Four figures -> figures/ads9224r-module/:
  signal_chain.png   shunt -> THS4551 FDA -> RC bucket -> ADS9224R + reference
  simultaneity.png   one CONVST samples both channels at the same instant (Q21)
  scaling.png        DC transfer (codes/A) from the ngspice front-end model
  settling.png       acquisition-window settling from the ngspice transient

scaling.png + settling.png are rendered from live ngspice runs of
sim/circuits/ads9224r_*.cir (the same models the derivation tests assert on).
Standing caveat: designed + simulated against the device models, values
EVM-baseline / datasheet-typical (Q23) - not a hardware measurement.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch  # noqa: E402

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT / "sim" / "scripts"))
import sim_params  # noqa: E402
import spice_runner  # noqa: E402

CAVEAT = ("designed + simulated vs the device models; values EVM-baseline / "
          "datasheet-typical (Q23) - not a hardware measurement")


def caveat(fig):
    fig.text(0.99, 0.004, CAVEAT, ha="right", va="bottom", fontsize=6.5,
             color="0.45", style="italic")


def finish(fig, out, name):
    caveat(fig)
    path = Path(out) / f"{name}.png"
    fig.savefig(path, dpi=130)
    plt.close(fig)
    print(f"  wrote {path}")


def fig_signal_chain(params, out):
    m = params.circuit_values("circuit.ads9224r_module")
    gain = params.value("feedback.current_ads9224r.fda_gain")
    cpa = params.value("feedback.current_ads9224r.codes_per_amp")
    fs = params.value("feedback.current_ads9224r.full_scale_a")
    fig, ax = plt.subplots(figsize=(11.0, 4.2))
    ax.set_xlim(0, 104); ax.set_ylim(0, 40); ax.axis("off")
    blocks = [
        (4, "Low-side\nshunt\n%g mOhm" % (m["shunt"] * 1e3), "#cde"),
        (24, "THS4551 FDA\ngain = %g\n(Rf/Rg=%gk/%g)" % (
            gain, m["fda_rf"] / 1e3, m["fda_rg"]), "#dec"),
        (46, "RC bucket\nRflt %gohm\nCflt %.1fnF" % (
            m["flt_r"], m["flt_c"] * 1e9), "#edc"),
        (68, "ADS9224R\n16-bit dual\nsimultaneous", "#ddd"),
        (89, "FPGA header\nCONVST/CS/SCLK\nSDOx/READY 3V3", "#cdd"),
    ]
    for x, label, color in blocks:
        ax.add_patch(FancyBboxPatch((x, 14), 13, 12,
                     boxstyle="round,pad=0.3", fc=color, ec="0.4"))
        ax.text(x + 6.5, 20, label, ha="center", va="center", fontsize=8.5)
    for x0 in (17, 39, 61, 83):
        ax.add_patch(FancyArrowPatch((x0, 20), (x0 + 5, 20),
                     arrowstyle="-|>", mutation_scale=14, color="0.3"))
    # reference feeding the ADC
    ax.add_patch(FancyBboxPatch((57, 1), 24, 8, boxstyle="round,pad=0.3",
                 fc="#fed", ec="0.4"))
    ax.text(69, 5, "REF6041 %.3f V + buffer + %guF reservoir"
            % (m["ref_v"], m["ref_reservoir_c"] * 1e6),
            ha="center", va="center", fontsize=8)
    ax.add_patch(FancyArrowPatch((69, 9), (74, 14), arrowstyle="-|>",
                 mutation_scale=12, color="0.3"))
    ax.text(50, 36, "Open ADS9224R current-sense module: one CONVST samples "
            "both phase currents at the same instant (Q21)",
            ha="center", fontsize=11, weight="bold")
    ax.text(50, 31, "full-scale +/-%.0f A   |   %.0f codes/A   |   "
            "16-bit two's-complement" % (fs, cpa), ha="center", fontsize=9,
            color="0.3")
    finish(fig, out, "signal_chain")


def fig_simultaneity(params, out):
    fig, (ax0, ax1) = plt.subplots(2, 1, figsize=(9.0, 5.2), sharex=True)
    t = np.linspace(0, 10, 500)
    convst = np.where((t > 2) & (t < 2.4), 1.0, 0.0)
    for ax, title in ((ax0, "ADS9224R - simultaneous (this module)"),
                      (ax1, "Single sequential ADC (MCP3208) - skew")):
        ax.plot(t, convst + 3.2, color="0.3", lw=1.2)
        ax.text(0.2, 3.5, "CONVST", fontsize=8, color="0.3")
        ax.set_yticks([]); ax.set_ylim(-0.3, 4.6)
        ax.set_title(title, fontsize=10)
    # simultaneous: both sample at t=2
    for y, lab, c in ((2.0, "phase A", "tab:blue"), (0.6, "phase B", "tab:red")):
        ax0.axvline(2.0, color=c, ls="--", lw=1.0)
        ax0.plot([2.0], [y + 0.4], "o", color=c)
        ax0.text(2.3, y + 0.4, f"{lab} sampled @ t0", color=c, fontsize=8)
    # sequential: A at t=2, B at t=2+skew
    ax1.axvline(2.0, color="tab:blue", ls="--", lw=1.0)
    ax1.plot([2.0], [2.4], "o", color="tab:blue")
    ax1.text(2.3, 2.4, "phase A @ t0", color="tab:blue", fontsize=8)
    ax1.axvline(5.5, color="tab:red", ls="--", lw=1.0)
    ax1.plot([5.5], [1.0], "o", color="tab:red")
    ax1.text(5.8, 1.0, "phase B @ t0 + skew", color="tab:red", fontsize=8)
    ax1.annotate("", xy=(5.5, 0.1), xytext=(2.0, 0.1),
                 arrowprops=dict(arrowstyle="<->", color="0.4"))
    ax1.text(3.75, 0.25, "inter-channel skew -> dq error (Q21)",
             ha="center", fontsize=8, color="0.4")
    ax1.set_xlabel("time (us, illustrative)")
    fig.suptitle("Why simultaneous sampling: the module removes the skew the "
                 "part-comparison study (T3) measures", fontsize=11)
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    finish(fig, out, "simultaneity")


def fig_scaling(params, out):
    dc = spice_runner.run_netlist("ads9224r_frontend", params)["ads9224r_dc.out"]
    cur = np.array([r[0] for r in dc]); vod = np.array([r[1] for r in dc])
    ref = params.value("circuit.ads9224r_module.ref_v")
    fs = params.value("feedback.current_ads9224r.full_scale_a")
    cpa = params.value("feedback.current_ads9224r.codes_per_amp")
    codes = vod / ref * 32768.0
    fig, ax = plt.subplots(figsize=(8.4, 5.0))
    ax.plot(cur, codes, color="tab:green", lw=1.6)
    ax.axhline(32768, ls="--", color="0.5", lw=0.8)
    ax.axhline(-32768, ls="--", color="0.5", lw=0.8)
    ax.axvline(fs, ls=":", color="0.5"); ax.axvline(-fs, ls=":", color="0.5")
    ax.text(0, 34000, "+full-scale (clamp)", fontsize=7, color="0.4")
    ax.set_xlabel("phase current (A)")
    ax.set_ylabel("ADC code (16-bit signed)")
    ax.set_title("ADS9224R module scaling (ngspice): %.0f codes/A, "
                 "+/-%.0f A full-scale" % (cpa, fs))
    ax.grid(alpha=0.3)
    finish(fig, out, "scaling")


def fig_settling(params, out):
    rows = spice_runner.run_netlist("ads9224r_settle", params)["ads9224r_settle.out"]
    t = np.array([r[0] for r in rows]); v = np.array([r[1] for r in rows])
    t_acq = params.value("adc.ads9224r_acq_window_s")
    resid = np.abs(4.0 - v) / 3.0
    fig, (ax0, ax1) = plt.subplots(1, 2, figsize=(11.0, 4.4))
    ax0.plot(t * 1e9, v, color="tab:purple", lw=1.4)
    ax0.axvline(t_acq * 1e9, ls="--", color="0.4")
    ax0.text(t_acq * 1e9 + 5, 2.5, "acquisition\nwindow end", fontsize=8,
             color="0.4")
    ax0.set_xlabel("time (ns)"); ax0.set_ylabel("bucket voltage (V)")
    ax0.set_title("Charge-bucket settling (worst-case step)")
    ax0.grid(alpha=0.3); ax0.set_xlim(0, t_acq * 1e9 * 2)
    ax1.semilogy(t * 1e9, np.maximum(resid, 1e-9), color="tab:purple", lw=1.4)
    ax1.axvline(t_acq * 1e9, ls="--", color="0.4")
    ax1.axhline(1.0 / 65536, ls=":", color="tab:red")
    ax1.text(5, 1.0 / 65536 * 1.3, "0.5 LSB", color="tab:red", fontsize=8)
    ax1.set_xlabel("time (ns)"); ax1.set_ylabel("settling residual (fraction)")
    ax1.set_title("Settles below 0.5 LSB within the window")
    ax1.grid(alpha=0.3, which="both"); ax1.set_xlim(0, t_acq * 1e9 * 2)
    fig.suptitle("ADS9224R front-end acquisition settling (ngspice; flt_r * "
                 "flt_c)", fontsize=11)
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    finish(fig, out, "settling")


FIGURES = {"signal_chain": fig_signal_chain, "simultaneity": fig_simultaneity,
           "scaling": fig_scaling, "settling": fig_settling}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out",
                    default=str(PROJECT_ROOT / "figures" / "ads9224r-module"))
    ap.add_argument("--only", default="")
    args = ap.parse_args()
    out = Path(args.out); out.mkdir(parents=True, exist_ok=True)
    params = sim_params.load()
    names = [n.strip() for n in args.only.split(",") if n.strip()] or FIGURES
    for name in names:
        print(f"rendering {name} ...")
        FIGURES[name](params, out)
    print(f"done -> {out}")


if __name__ == "__main__":
    main()
