# SPDX-License-Identifier: MIT
"""codesign_bldc.py — BLDC motor + belt co-design for the dynamic-gait path.

The servo shortlist (codesign.py servo_report) topped out; the honest path to a
dynamic gait is a BLDC leg actuator with your own FOC. This evaluates the motor
catalog the way you asked: TORQUE-AT-SPEED under THERMAL (continuous-current)
limits, not vendor max-power/max-torque headlines.

Model per motor (Kt = 9.55/Kv N·m/A), driven at bus V through a belt reduction N
(eff eta). The OUTPUT torque-speed envelope is voltage- AND current-limited:
    tau_out(w_out) = N*eta * Kt * min( I_lim, (V - Kt*N*w_out)/R )
i.e. flat thermal-torque N*eta*Kt*I up to base speed ~V/(Kt*N), then it rolls off
as back-EMF eats the voltage. We report the CONTINUOUS envelope (I_cont, the
thermal number that matters for a walking gait) and the PEAK (I_peak, seconds-long
bursts for the stomp).

Leg targets (output/joint side), from Level A/B:
  * dynamic stride: joint speed for Fr>=0.10 is ~2*v/lever with v=sqrt(Fr*g*L);
  * stance/drag CONTINUOUS torque ~6 N·m; combat stomp PEAK ~15 N·m.
Per motor we pick the belt ratio that clears the dynamic speed with headroom, then
report the continuous + peak torque there and the max thermally-sustainable Froude.

  .venv-warp/bin/python codesign_bldc.py
Currents/R marked (est) are class estimates, not datasheet — flagged in output.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from walker_improved import DEFAULTS  # noqa: E402

G = 9.81
L = DEFAULTS["stance_h"]           # leg length ~0.42 m
LEVER = DEFAULTS["yaw_lever"]      # 0.11 m stride lever
V_BUS = 48.0                       # design bus voltage (user: "48 V, tons of headroom")
ETA = 0.90                         # belt efficiency
STANCE_TORQUE = 6.0               # N·m continuous the leg must hold/drag
COMBAT_TORQUE = 15.0              # N·m peak stomp
FR_DYNAMIC = 0.10                 # dynamic-walk threshold

# catalog: (Kv, I_cont A, I_peak A, R_phase ohm, mass g, cost $, note, est_flags)
# est_flags: which of (I_cont, I_peak, R) are class-estimates vs datasheet.
BLDC = {
    "ODrive D6374 150Kv":   (150, 50, 90, 0.039, 800, 119, "reference", ""),
    "ODrive D5312s 330Kv":  (330, 30, 60, 0.050, 250, 129, "compact", ""),
    "mjbots mj5208 330Kv":  (330, 18, 58, 0.100, 193, 190, "light/distal", "Icont,R est"),
    "Maytech 6374 170Kv":   (170, 40, 65, 0.030, 800,  87, "cheap 6374", "Icont est"),
    "Flipsky 6384 190Kv":   (190, 50, 95, 0.025, 1000, 129, "cheap hi-torque", "Icont est"),
    "Flipsky 7070 110Kv":   (110, 50, 100, 0.020, 1080, 108, "big cheap", "Icont est"),
    "MAD 8118 100Kv":       (100, 30, 45, 0.070, 620, 250, "low-Kv QDD-ish", "I est"),
    "T-Motor V807 170Kv":   (170, 60, 150, 0.025, 650, 365, "heavy power", "Icont est"),
    "T-Motor V10L 170Kv":   (170, 90, 196, 0.018, 980, 497, "extreme", "Icont est"),
}


def kt(kv):
    return 9.55 / kv          # N·m/A


def out_envelope(kv, I, R, N, w_out):
    """Output torque (N·m) at output speed w_out (rad/s), current-limit I, ratio N."""
    ktv = kt(kv)
    w_m = N * w_out                                  # motor speed
    i_volt = max(0.0, (V_BUS - ktv * w_m) / R)       # current the bus allows here
    return N * ETA * ktv * min(I, i_volt)


def evaluate(name):
    kv, Ic, Ip, R, mg, cost, note, est = BLDC[name]
    # target joint no-load speed for dynamic stride (Fr just past threshold, +50% headroom)
    v_dyn = np.sqrt(FR_DYNAMIC * G * L)
    w_joint_dyn = 1.5 * (2 * v_dyn / LEVER)          # rad/s, with headroom
    # pick belt ratio so motor no-load / N clears w_joint_dyn; but N also sets torque.
    # choose N to just meet the CONTINUOUS stance torque at low speed, then see how
    # fast we can still go (max Fr) — the torque-at-speed-under-thermal optimum.
    w_m0 = kv * V_BUS * 2 * np.pi / 60                # motor no-load speed rad/s
    # smallest N giving stance torque at ~0 speed:  N*eta*Kt*Ic >= STANCE_TORQUE
    N_min_torque = STANCE_TORQUE / (ETA * kt(kv) * Ic)
    # largest N keeping output no-load above the dynamic joint speed:
    N_max_speed = (w_m0 / w_joint_dyn)
    if N_min_torque > N_max_speed:
        # can't both hold stance torque AND reach dynamic speed — flag it
        N = N_min_torque
        feasible = False
    else:
        N = np.sqrt(N_min_torque * N_max_speed)      # geometric mid: balance
        feasible = True
    w_out0 = w_m0 / N
    # max sustainable joint speed under CONTINUOUS current while still delivering
    # a modest driving torque (say 2 N·m to move): solve envelope >= 2
    w_grid = np.linspace(0.1, w_out0, 200)
    ok = [w for w in w_grid if out_envelope(kv, Ic, R, N, w) >= 2.0]
    w_sust = max(ok) if ok else 0.0
    v_max = w_sust * LEVER / 2                        # invert stride relation
    fr = v_max ** 2 / (G * L)
    tau_cont0 = out_envelope(kv, Ic, R, N, 0.2)      # continuous torque near stall
    tau_peak0 = out_envelope(kv, Ip, R, N, 0.2)      # peak (stomp) near stall
    return dict(name=name, kt=kt(kv), N=N, w_out0=w_out0, fr=fr, v=v_max,
                tau_cont=tau_cont0, tau_peak=tau_peak0, mass=mg, cost=cost,
                note=note, est=est, feasible=feasible,
                combat_ok=tau_peak0 >= COMBAT_TORQUE, dyn_ok=fr >= FR_DYNAMIC)


def report():
    v_dyn = np.sqrt(FR_DYNAMIC * G * L)
    print(f"leg L={L:.2f}m lever={LEVER*100:.0f}cm  bus={V_BUS:.0f}V  "
          f"dynamic threshold Fr>={FR_DYNAMIC} (v>{v_dyn:.2f} m/s)")
    print(f"targets: stance {STANCE_TORQUE:.0f} N·m cont, stomp {COMBAT_TORQUE:.0f} N·m peak\n")
    print(f"  {'motor':<22} {'belt':>5} {'Fr_max':>6} {'v[m/s]':>6} {'τcont':>6} "
          f"{'τpeak':>6} {'mass':>5} {'$':>4}  verdict")
    rows = sorted((evaluate(n) for n in BLDC), key=lambda r: -r["fr"])
    for r in rows:
        dyn = "DYNAMIC" if r["dyn_ok"] else "quasi-stat"
        cbt = "+stomp" if r["combat_ok"] else "weak-stomp"
        flag = f"  [{r['est']}]" if r["est"] else ""
        print(f"  {r['name']:<22} {r['N']:>4.0f}:1 {r['fr']:>6.2f} {r['v']:>6.2f} "
              f"{r['tau_cont']:>5.0f}N {r['tau_peak']:>5.0f}N {r['mass']/1000:>4.1f}kg "
              f"{r['cost']:>4.0f}  {dyn} {cbt}{flag}")
    print("\n  All comfortably clear the dynamic threshold (BLDC continuous torque-at-")
    print("  speed dwarfs the servo): the choice is mass / cost / thermal margin, not")
    print("  capability. Fr shown is thermally SUSTAINABLE (continuous current), not a")
    print("  burst. Combat stomp uses the peak-current column.")
    # recommendation
    good = [r for r in rows if r["dyn_ok"] and r["combat_ok"]]
    light = min(good, key=lambda r: r["mass"]) if good else None
    cheap = min(good, key=lambda r: r["cost"]) if good else None
    if light:
        print(f"\n  Lightest that does both: {light['name']} ({light['mass']/1000:.1f}kg, "
              f"${light['cost']}, Fr {light['fr']:.2f})")
    if cheap:
        print(f"  Cheapest that does both: {cheap['name']} (${cheap['cost']}, "
              f"{cheap['mass']/1000:.1f}kg, Fr {cheap['fr']:.2f})")
    print("\n  NOTE: currents/R marked (est) are class estimates — refine from datasheets")
    print("  before committing. Optimize torque-at-speed under YOUR thermal envelope")
    print("  (continuous current with your cooling), per your guidance. Level C (RL on")
    print("  the BLDC-driven sprung leg) is the pre-hardware verification.")


if __name__ == "__main__":
    report()


# ---------------------------------------------------------------------------
# PROVENANCE + AVAILABILITY SCORECARD (2026-07-05) — for a self-FOC-on-FPGA build.
# These are CURATED ENGINEERING JUDGMENTS (documentation maturity, datasheet
# trustworthiness, distributor availability, FOC/FPGA ecosystem prior-art), NOT
# datasheet-derived numbers. Scores 1-5. Weighting favors provenance (doc+data)
# with availability a strong secondary, per the optimization ask.
# ---------------------------------------------------------------------------
W_PROV = dict(doc=0.30, data=0.30, avail=0.20, eco=0.20)   # doc/data = provenance

# (doc, datasheet-validation, availability, FOC/FPGA-ecosystem, why)
MOTOR_PROV = {
    "ODrive D6374 150Kv":  (5, 5, 4, 5, "robotics vendor publishes Kt/R/I/thermal; dual-shaft+thermistor; ODrive ecosystem"),
    "ODrive D5312s 330Kv": (5, 5, 4, 5, "same ODrive provenance, compact"),
    "mjbots mj5208 330Kv": (4, 5, 3, 5, "open robotics actuator (Pieper); documented for legs; smaller vendor stock"),
    "Maytech 6374 170Kv":  (2, 2, 5, 3, "e-skate: everywhere & cheap, but optimistic/incomplete specs — characterize yourself"),
    "Flipsky 6384/7070":   (2, 2, 5, 3, "same e-skate provenance gap; huge availability"),
    "MAD 8118 100Kv":      (3, 3, 3, 2, "UAV: thrust curves not Kt/torque; awkward for belt legs"),
    "T-Motor V807/V10L":   (3, 3, 4, 2, "reputable UAV maker but prop-oriented docs, pricey"),
}
ENCODER_PROV = {
    "AS5047P (Infineon)":  (5, 5, 5, 5, "industry-standard 14-bit magnetic; SPI/ABI/PWM; ODrive/VESC/SimpleFOC all use it; Mouser/Digikey stock"),
    "MA732 (MPS)":         (4, 4, 4, 3, "good datasheet, high-speed; less FOC prior-art than AS5047P"),
    "CUI AMT212B":         (4, 4, 3, 4, "absolute RS485, ODrive-sold; more specialized/output-side"),
}
POWER_PROV = {
    "TI DRV8353RS + ext FETs": (5, 5, 5, 5, "SPI gate driver + integrated current-sense; DRV8353RS-EVM; TI app notes; ideal FPGA-drives-PWM split"),
    "TI TIDA-010956 (ref)":    (5, 5, 4, 4, "documented 85A/24-60V 3-phase inverter REFERENCE DESIGN (schematic/BOM/test) to adapt"),
    "VESC 75/300 (open HW)":   (4, 4, 4, 3, "open-source schematics, battle-tested — but it's a CONTROLLER; FPGA FOC must bypass its MCU"),
    "ST EVSPIN32G4":           (4, 4, 4, 3, "documented eval stage (embedded MCU you'd ignore)"),
}


def _score(row):
    d, v, a, e, _ = row
    return W_PROV["doc"] * d + W_PROV["data"] * v + W_PROV["avail"] * a + W_PROV["eco"] * e


def provenance_report():
    print("PROVENANCE + AVAILABILITY SCORECARD (curated judgment, not datasheet math)")
    print(f"weights: doc {W_PROV['doc']}, datasheet-validation {W_PROV['data']}, "
          f"availability {W_PROV['avail']}, FOC/FPGA-ecosystem {W_PROV['eco']}\n")
    for title, cat in (("MOTOR", MOTOR_PROV), ("ENCODER", ENCODER_PROV),
                       ("GATE DRIVER / POWER STAGE", POWER_PROV)):
        print(f"  --- {title} ---")
        for name, row in sorted(cat.items(), key=lambda kv: -_score(kv[1])):
            d, v, a, e, why = row
            print(f"    {_score(row):.1f}  {name:<26} doc{d} data{v} avail{a} eco{e}"
                  f"  — {why}")
        print()
    print("  RECOMMENDED PROVENANCE STACK (FPGA does FOC math; parts do the analog):")
    print("    motor    : ODrive D6374 150Kv   (published params, dual-shaft encoder mount)")
    print("    encoder  : AS5047P               (motor-side; +optional output-side after belt)")
    print("    power    : TI DRV8353RS + ext FETs, adapting TIDA-010956 as the reference")
    print("    fpga     : AMD/Xilinx Zynq class (most documented FOC prior-art & motor kits)")
    print("    cheap-iterate fallback: Maytech/Flipsky 6374 — accept you bench-characterize it")
