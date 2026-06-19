#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
"""Stress-test / extreme-scenario figures: push the system + code to the limit
and show the protections (and the math) holding (notes/stress-test-report.md).

Four axes -> figures/stress/:
  A system failure boundaries: thermal runaway, brownout cascade, regen
    overvoltage, overcurrent clamp + shoot-through guard, fault inject/detect
  B control & sensor limits:   reversal cliff to loss-of-lock, extreme load step
  C analog / ADC extremes:     acquisition-settling failure boundary, full-scale
    code clipping
  D numerical / code edges:    fixed-point rails stay bounded, circle-limiter
    saturation

Every scenario is a live bench / ngspice / RTL-math run. Standing caveat:
placeholder motor parameters (Q1) - relative, illustrative, not hardware specs.

Usage: python3 sim/scripts/gen_stress_figures.py [--out figures/stress] [--only A1_thermal,...]
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
for sub in ("sim/scripts", "sim/tests", "sim/build/cpp"):
    sys.path.insert(0, str(PROJECT_ROOT / sub))

import bldcsim  # noqa: E402
import focmath  # noqa: E402
import part_compare as PC  # noqa: E402
import sim_params  # noqa: E402
import spice_runner  # noqa: E402
from bench_factory import expected_init_time, foc, realism  # noqa: E402

CAVEAT = ("live bench/ngspice/RTL run, placeholder motor params (Q1) - "
          "illustrative stress behaviour, not a hardware spec")
P = sim_params.load()
ALIGN = int(P.value("foc.align_offset"))


def caveat(fig):
    fig.text(0.99, 0.004, CAVEAT, ha="right", va="bottom", fontsize=6.5,
             color="0.45", style="italic")


def finish(fig, out, name):
    caveat(fig)
    path = Path(out) / f"{name}.png"
    fig.savefig(path, dpi=130)
    plt.close(fig)
    print(f"  wrote {path}")


def _init(cfg, iq=60, omega=None, mode=3):
    b = bldcsim.Bench(cfg)
    b.run_for(expected_init_time(P))
    b.set_align_offset(ALIGN)
    b.set_id_target(0)
    b.set_iq_target(int(iq))
    if omega is not None:
        b.set_speed_clamp(True, float(omega))
    b.set_mode(mode)
    return b


# --------------------------------------------------------------------------- A
def fig_A1_thermal(out):
    b = _init(foc(P, "thermal"), iq=250, omega=0.0)
    t, mt, ft, ipk = [], [], [], []
    for k in range(48):
        b.run_for(0.25)
        t.append(k * 0.25); mt.append(b.motor_t_c); ft.append(b.fet_tj_max_c)
        ipk.append(np.abs(b.currents).max())
    fig, (a0, a1) = plt.subplots(2, 1, figsize=(9.0, 6.2), sharex=True,
                                 constrained_layout=True)
    a0.plot(t, mt, color="tab:red", label="motor winding")
    a0.plot(t, ft, color="tab:orange", label="FET junction")
    a0.set_ylabel("temperature (C)"); a0.legend(loc="upper left")
    a0.set_title("A1  Thermal runaway: locked-rotor cook (iq held, rotor blocked)")
    a0.grid(alpha=0.3)
    a1.plot(t, ipk, color="tab:blue")
    a1.set_ylabel("peak phase current (A)"); a1.set_xlabel("time (s)")
    a1.set_ylim(0, max(ipk) * 1.3)
    a1.text(0.02, 0.12, "the current loop faithfully holds the commanded current "
            "-> nothing throttles the heating\n(a pure current loop has no thermal "
            "foldback) -> temperature runs away", transform=a1.transAxes,
            fontsize=8, color="0.35")
    a1.grid(alpha=0.3)
    finish(fig, out, "A1_thermal")


def fig_A2_brownout(out):
    # Emergent brownout: a weak current-limited supply driven by the closed-loop
    # six-step controller. Read the C++ trace (records vbus from t=0 at 10 us).
    b = bldcsim.Bench(realism(P, "supply", supply={"i_limit_a": 0.7}))
    b.run_for(expected_init_time(P))
    b.set_target_speed(95)
    b.set_mode(2)
    b.run_for(1.2)
    tr = b.trace
    t = np.asarray(tr["time_s"]) * 1e3
    vbus = np.asarray(tr["vbus_v"]); fc = np.asarray(tr["fault_count"])
    vfall = P.value("drv8301_uvlo.pvdd_falling")
    fig, (a0, a1) = plt.subplots(2, 1, figsize=(9.5, 6.2), sharex=True,
                                 constrained_layout=True)
    a0.plot(t, vbus, color="tab:red", lw=0.9, label="bus voltage")
    a0.axhline(vfall, ls="--", color="0.3",
               label="DRV8301 UVLO falling (%.1f V)" % vfall)
    a0.set_ylabel("DC bus (V)"); a0.legend(loc="lower left", fontsize=8)
    a0.set_title("A2  Brownout cascade: a 0.7 A supply is dragged below UVLO "
                 "(nothing injected)")
    a0.grid(alpha=0.3)
    a1.step(t, fc, where="post", color="tab:purple")
    a1.set_ylabel("nFAULT events\n(RTL)"); a1.set_xlabel("time (ms)")
    a1.grid(alpha=0.3)
    finish(fig, out, "A2_brownout")


def fig_A3_regen(out):
    # Clean regen: spin up under six-step closed loop, then command a hard decel
    # (110 -> 10). The rotor's kinetic energy dumps into the bus cap, pumping it
    # up - bounded by the controller's duty down-slew guard. Read the trace.
    b = bldcsim.Bench(realism(P, "supply",
                              supply={"i_limit_a": 20.0, "capacitance": 3.0e-4}))
    b.run_for(expected_init_time(P))
    b.set_align_offset(int(P.value("rtl.sensored_align_offset")))
    b.set_target_speed(110); b.set_mode(2); b.run_for(1.2)
    b.set_target_speed(10); b.run_for(0.8)
    tr = b.trace
    t = np.asarray(tr["time_s"]) * 1e3
    vbus = np.asarray(tr["vbus_v"]); om = np.asarray(tr["omega_rad_s"])
    fig, (a0, a1) = plt.subplots(2, 1, figsize=(9.5, 6.0), sharex=True,
                                 constrained_layout=True)
    a0.plot(t, vbus, color="tab:red", lw=0.9, label="DC bus")
    a0.axhline(P.value("supply.v_alarm"), ls="--", color="0.3",
               label="overvoltage alarm (%.0f V)" % P.value("supply.v_alarm"))
    a0.axhline(P.value("bus.vbus"), ls=":", color="0.5", label="nominal 12 V")
    a0.set_ylabel("DC bus (V)"); a0.legend(loc="upper left", fontsize=8)
    a0.set_title("A3  Regen overvoltage: a hard decel dumps kinetic energy into "
                 "the bus cap")
    a0.text(0.97, 0.5, "pump bounded by the duty\ndown-slew guard -> stays\nunder "
            "the alarm", transform=a0.transAxes, ha="right", fontsize=8,
            color="0.35")
    a0.grid(alpha=0.3)
    a1.plot(t, om, color="tab:blue"); a1.set_ylabel("speed (rad/s)")
    a1.set_xlabel("time (ms)"); a1.grid(alpha=0.3)
    finish(fig, out, "A3_regen")


def fig_A4_overcurrent(out):
    cmds = [40, 100, 200, 300, 500, 800, 1200, 1800]
    delivered, peak, shoot = [], [], []
    for c in cmds:
        b = _init(foc(P), iq=c, omega=40.0)
        for _ in range(8):
            b.run_for(0.02)
        delivered.append(b.foc_iq); peak.append(np.abs(b.currents).max())
        shoot.append(b.shoot_through_violations)
    fig, (a0, a1) = plt.subplots(1, 2, figsize=(11.0, 4.6),
                                 constrained_layout=True)
    a0.plot(cmds, cmds, ls=":", color="0.6", label="commanded")
    a0.plot(cmds, delivered, "-o", color="tab:red", label="delivered (clamped)")
    a0.axhline(P.value("foc.iq_max"), ls="--", color="0.4",
               label="iq_max clamp (%d)" % P.value("foc.iq_max"))
    a0.set_xlabel("commanded iq (LSB)"); a0.set_ylabel("iq (LSB)")
    a0.set_title("Torque-current clamp holds"); a0.legend(loc="upper left")
    a0.grid(alpha=0.3)
    a1.bar(range(len(cmds)), shoot, color="tab:green")
    a1.set_xticks(range(len(cmds))); a1.set_xticklabels(cmds, fontsize=7, rotation=45)
    a1.set_ylim(-1, 1); a1.set_yticks([0, 1])
    a1.set_xlabel("commanded iq (LSB)"); a1.set_ylabel("shoot-through events")
    a1.set_title("Dead-time guard: 0 shoot-through at every command")
    a1.grid(alpha=0.3, axis="y")
    fig.suptitle("A4  Overcurrent command: the protections, not the silicon, set "
                 "the limit", fontsize=12)
    finish(fig, out, "A4_overcurrent")


def fig_A5_fault(out):
    b = _init(foc(P), iq=60, omega=60.0)
    rec = {"t": [], "fc": [], "dead": [], "locked": []}
    dt = 1e-3
    inj = 60
    for k in range(160):
        if k == inj:
            b.inject_drv_latched_fault()
        if k == 110:
            b.inject_drv_otw(True)
        b.run_for(dt)
        rec["t"].append(k * dt); rec["fc"].append(b.fault_count)
        rec["dead"].append(int(b.drv_dead)); rec["locked"].append(int(b.locked_out))
    t = np.array(rec["t"]) * 1e3
    fig, ax = plt.subplots(figsize=(9.0, 5.0), constrained_layout=True)
    ax.plot(t, rec["fc"], color="tab:red", drawstyle="steps-post",
            label="fault_count")
    ax.plot(t, np.array(rec["dead"]) + 0.05, color="tab:orange",
            drawstyle="steps-post", label="drv_dead")
    ax.plot(t, np.array(rec["locked"]) - 0.05, color="tab:purple",
            drawstyle="steps-post", label="locked_out")
    ax.axvline(inj * dt * 1e3, ls=":", color="0.5")
    ax.text(inj * dt * 1e3 + 1, 0.5, "latched fault injected", fontsize=8,
            color="0.4")
    ax.set_xlabel("time (ms)"); ax.set_ylabel("event / flag")
    ax.set_title("A5  Fault injection: the controller detects and flags it")
    ax.legend(loc="center right"); ax.grid(alpha=0.3)
    finish(fig, out, "A5_fault")


# --------------------------------------------------------------------------- B
def fig_B1_reversal_cliff(out):
    speeds = [100, 300, 600, 900, 1300, 1800]
    a = PC.run_reversal(P, bldcsim, PC.SENSOR_A, speeds, transition_s=0.008)
    s = PC.run_reversal(P, bldcsim, PC.SENSOR_B, speeds, transition_s=0.008)
    fig, ax = plt.subplots(figsize=(9.0, 5.0), constrained_layout=True)
    rpm = a["peak_speed"] * 60 / (2 * np.pi)
    ax.plot(rpm, a["peak_err_deg"], "-o", color="tab:red", label="AS5600")
    ax.plot(rpm, s["peak_err_deg"], "-o", color="tab:green", label="AS5047P")
    ax.axhline(90, ls="--", color="0.3")
    ax.text(rpm[0], 95, "90 deg: commutation inverts (loss of lock)", fontsize=8,
            color="0.3")
    ax.set_xlabel("reversal peak speed (rev/min)")
    ax.set_ylabel("peak commutation error (elec deg)")
    ax.set_title("B1  Reversal slam to failure: where each sensor loses lock")
    ax.legend(loc="upper left"); ax.grid(alpha=0.3)
    finish(fig, out, "B1_reversal_cliff")


def fig_B2_load_step(out):
    b = _init(foc(P, "mechanical"), iq=120, omega=None, mode=3)
    b.set_foc_speed_loop(False)
    b.set_speed_clamp(True, 80.0)
    rec = {"t": [], "om": [], "i": []}
    dt = 2e-4
    for k in range(300):
        if k == 60:
            b.set_load_torque(0.8)     # instant heavy load
        if k == 180:
            b.set_load_torque(0.0)
        b.run_for(dt)
        rec["t"].append(k * dt); rec["om"].append(b.omega)
        rec["i"].append(np.abs(b.currents).max())
    t = np.array(rec["t"]) * 1e3
    fig, (a0, a1) = plt.subplots(2, 1, figsize=(9.0, 6.0), sharex=True,
                                 constrained_layout=True)
    a0.plot(t, rec["om"], color="tab:blue"); a0.set_ylabel("speed (rad/s)")
    a0.set_title("B2  Extreme load step: instant 0->0.8 N*m and back")
    a0.grid(alpha=0.3)
    for kx in (60, 180):
        a0.axvline(kx * dt * 1e3, ls=":", color="0.6")
        a1.axvline(kx * dt * 1e3, ls=":", color="0.6")
    a1.plot(t, rec["i"], color="tab:red"); a1.set_ylabel("peak current (A)")
    a1.set_xlabel("time (ms)"); a1.grid(alpha=0.3)
    finish(fig, out, "B2_load_step")


# --------------------------------------------------------------------------- C
def fig_C1_settle_limit(out):
    caps = [("1 nF (design)", 1.0e-9), ("4.7 nF", 4.7e-9),
            ("22 nF", 22.0e-9), ("47 nF (oversized)", 47.0e-9)]
    t_acq = P.value("adc.ads9224r_acq_window_s")
    ref = P.value("circuit.ads9224r_module.ref_v")
    fig, ax = plt.subplots(figsize=(9.0, 5.0), constrained_layout=True)
    for label, c in caps:
        rows = spice_runner.run_netlist("ads9224r_acq", P,
                                        overrides={"bcap": c})["ads9224r_acq.out"]
        tt = np.array([r[0] for r in rows]); v = np.array([r[1] for r in rows])
        resid = np.abs(ref - v) / ref
        ax.semilogy(tt * 1e9, np.maximum(resid, 1e-9), label=label)
    ax.axvline(t_acq * 1e9, ls="--", color="0.3")
    ax.text(t_acq * 1e9 + 3, 1e-2, "tACQ %.0f ns" % (t_acq * 1e9), fontsize=8,
            color="0.4")
    ax.axhline(1 / 65536, ls=":", color="tab:red")
    ax.text(5, 1 / 65536 * 1.4, "0.5 LSB", color="tab:red", fontsize=8)
    ax.set_xlim(0, t_acq * 1e9 * 2.2)
    ax.set_xlabel("time (ns)"); ax.set_ylabel("settling residual (fraction)")
    ax.set_title("C1  Settling failure boundary: oversize the charge bucket and "
                 "it misses tACQ")
    ax.legend(loc="upper right", fontsize=8); ax.grid(alpha=0.3, which="both")
    finish(fig, out, "C1_settle_limit")


def fig_C2_fullscale_clip(out):
    dc = spice_runner.run_netlist("ads9224r_frontend", P)["ads9224r_dc.out"]
    cur = np.array([r[0] for r in dc]); v = np.array([r[1] for r in dc])
    ref = P.value("circuit.ads9224r_module.ref_v")
    fs = P.value("feedback.current_ads9224r.full_scale_a")
    codes = v / ref * 32768.0
    fig, ax = plt.subplots(figsize=(9.0, 5.0), constrained_layout=True)
    ax.plot(cur, codes, color="tab:blue")
    ax.axhline(32768, ls="--", color="tab:red"); ax.axhline(-32768, ls="--", color="tab:red")
    ax.axvspan(fs, cur.max(), color="tab:red", alpha=0.1)
    ax.axvspan(cur.min(), -fs, color="tab:red", alpha=0.1)
    ax.text(fs * 1.02, 0, "railed\n(beyond FS)", fontsize=8, color="tab:red")
    ax.set_xlabel("phase current (A)"); ax.set_ylabel("ADC code (signed 16-bit)")
    ax.set_title("C2  Full-scale clipping: beyond +/-%.0f A the code rails (no "
                 "wrap)" % fs)
    ax.grid(alpha=0.3)
    finish(fig, out, "C2_fullscale_clip")


# --------------------------------------------------------------------------- D
def fig_D1_numeric_rails(out):
    fm = focmath.FocMath()
    M = (1 << 17) - 1
    mags = np.linspace(0, M, 60)
    da, db, dc, idv, iqv = [], [], [], [], []
    for m in mags:
        o = fm.eval(16384, int(m), int(-m), int(m), int(m), int(m), int(-m))
        da.append(o["duty_a"]); db.append(o["duty_b"]); dc.append(o["duty_c"])
        idv.append(o["id"]); iqv.append(o["iq"])
    half = int(P.value("pwm.frequency") and 625)  # PWM_HALF_PERIOD
    fig, (a0, a1) = plt.subplots(1, 2, figsize=(11.0, 4.6),
                                 constrained_layout=True)
    a0.plot(mags, da, label="duty_a"); a0.plot(mags, db, label="duty_b")
    a0.plot(mags, dc, label="duty_c")
    a0.axhline(625, ls="--", color="0.4"); a0.axhline(0, ls="--", color="0.4")
    a0.set_title("Duties stay in [0, 625] (no overflow wrap)")
    a0.set_xlabel("input magnitude (-> 18-bit rail)"); a0.set_ylabel("duty (counts)")
    a0.legend(fontsize=8); a0.grid(alpha=0.3)
    a1.plot(mags, idv, label="id"); a1.plot(mags, iqv, label="iq")
    a1.axvline(32768, ls="--", color="0.4")
    a1.text(33000, -55000, "physical input\nrange (<=FS code)", fontsize=7.5,
            color="0.4")
    a1.set_title("dq monotone within range; wraps only past the 18-bit rail")
    a1.set_xlabel("input magnitude"); a1.set_ylabel("dq current (LSB)")
    a1.legend(fontsize=8, loc="upper left"); a1.grid(alpha=0.3)
    fig.suptitle("D1  Fixed-point limits: the duties (bridge commands) always "
                 "clamp to [0,625]; intermediate dq wraps only at 2x-beyond-"
                 "physical inputs", fontsize=11)
    finish(fig, out, "D1_numeric_rails")


def fig_D2_circle_sat(out):
    fm = focmath.FocMath()
    M = (1 << 17) - 1
    cmds = np.linspace(0, M, 60)
    mag = []
    for c in cmds:
        o = fm.eval(8000, 0, 0, int(c * 0.6), int(c), 0, 0)
        mag.append(np.hypot(o["valpha"], o["vbeta"]))
    fig, ax = plt.subplots(figsize=(9.0, 5.0), constrained_layout=True)
    ax.plot(cmds, mag, color="tab:blue")
    ax.set_xlabel("commanded voltage magnitude (-> rail)")
    ax.set_ylabel("limited |v_alpha,beta| (LSB)")
    ax.set_title("D2  Circle limiter saturates the voltage vector (no overflow)")
    ax.grid(alpha=0.3)
    ax.text(0.4, 0.2, "output flattens at the voltage-circle radius\n"
            "regardless of how large the command grows",
            transform=ax.transAxes, fontsize=8, color="0.35")
    finish(fig, out, "D2_circle_sat")


FIGURES = {
    "A1_thermal": fig_A1_thermal, "A2_brownout": fig_A2_brownout,
    "A3_regen": fig_A3_regen, "A4_overcurrent": fig_A4_overcurrent,
    "A5_fault": fig_A5_fault, "B1_reversal_cliff": fig_B1_reversal_cliff,
    "B2_load_step": fig_B2_load_step, "C1_settle_limit": fig_C1_settle_limit,
    "C2_fullscale_clip": fig_C2_fullscale_clip, "D1_numeric_rails": fig_D1_numeric_rails,
    "D2_circle_sat": fig_D2_circle_sat,
}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=str(PROJECT_ROOT / "figures" / "stress"))
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
