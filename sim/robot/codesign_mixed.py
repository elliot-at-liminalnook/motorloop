# SPDX-License-Identifier: MIT
"""codesign_mixed.py — co-design for the LIGHTWEIGHT mixed config: 1 BLDC + 2 servos
per leg (notes/bldc-build-spec.md §1).

The all-servo and all-BLDC problems are single-actuator. The mixed config's physics
is different and more interesting: each of the three leg axes (yaw stride, pitch
reach, lift/knee foot-clearance) can be either a fast BLDC or a slow servo, and the
achievable gait speed is CAPPED BY THE SLOWEST GAIT-CRITICAL AXIS — so where you spend
the one BLDC decides the ceiling. This sets that up as the optimization and reports it.

Per-axis torque-speed envelopes:
  * BLDC (D5312s-class): no-load Kv*V/N, continuous torque N*eta*Kt*I_cont — fast+strong.
  * servo (torque-class, e.g. REEFS/Savox): no-load servo_free/N, stall servo_stall*N.
  * SEA spring on the YAW (your rubber belt): if yaw is a SERVO, the spring lets the
    stride oscillate near its resonance without the servo moving fast (Level B), so a
    servo+spring yaw is NOT as cadence-capped as a bare servo.

Gait-speed model (dynamic walk): stride s = 2*lever*sin(A); the cadence f is capped by
  - stride axis (yaw): peak yaw rate ~4*A*f <= yaw_noload  (or the SEA resonance if sprung)
  - lift axis: retract+extend 2*lift_range per swing -> lift rate <= lift_noload
  v = s*min(f_stride, f_lift);  Fr = v^2/(g*L). Propulsion power (BLDC ample) is checked.

  .venv-warp/bin/python codesign_mixed.py     # optimize each BLDC placement + recommend
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from codesign import static_margin, DEFAULTS  # noqa: E402  reuse geometry+static
from walker_improved import LEG_MASS, TORSO_MASS  # noqa: E402

G = 9.81
V_BUS = 48.0
YAW_AMP = 0.45            # rad half yaw sweep
DUTY = 0.6               # dynamic-walk duty
CRANK = 0.04            # m, lift crank radius (servo/BLDC rotary -> foot vertical)
FR_DYNAMIC = 0.10

# --- actuators — SINGLE-SOURCED from robot_design (2026-07-05) ------------------
from robot_design import BLDCS, SERVOS  # noqa: E402
_BMAP = {"D5312s": "odrive_d5312s", "mj5208": "mjbots_mj5208", "D6374": "odrive_d6374"}
BLDC = {k: dict(kv=BLDCS[v][0], kt=9.55 / BLDCS[v][0], icont=BLDCS[v][1],
                ipeak=BLDCS[v][2], mass=BLDCS[v][3], cost=BLDCS[v][4])
        for k, v in _BMAP.items()}
_S = SERVOS["reefs_400sc"]
SERVO = dict(free=_S[1], stall=_S[0], mass=_S[2], cost=_S[3])
SEA_FNAT = 3.85          # Hz, Level B resonant stride freq the belt spring enables

AXES = ["yaw", "pitch", "lift"]


def axis_noload(kind, motor, N):
    """Joint no-load angular speed (rad/s) for an axis at reduction N."""
    if kind == "bldc":
        return (BLDC[motor]["kv"] * V_BUS * 2 * np.pi / 60) / N
    return SERVO["free"] / N


def axis_torque(kind, motor, N, peak=False):
    """Joint stall/continuous torque (N·m)."""
    if kind == "bldc":
        m = BLDC[motor]
        return 0.9 * N * m["kt"] * (m["ipeak"] if peak else m["icont"])
    return SERVO["stall"] * N


def gait_froude(geom, placement, motor, Nyaw, Nlift):
    """Achievable dynamic-walk Froude for a BLDC placement + reductions."""
    lev, L, lift = geom["yaw_lever"], geom["stance_h"], geom["lift_range"]
    stride = 2 * lev * np.sin(YAW_AMP)
    yaw_kind = "bldc" if placement == "yaw" else "servo"
    lift_kind = "bldc" if placement == "lift" else "servo"
    # stride cadence (yaw): peak yaw rate ~4*A*f <= no-load; a SERVO yaw with the SEA
    # belt can instead oscillate at the spring resonance f_nat (Level B).
    yaw_nl = axis_noload(yaw_kind, motor, Nyaw)
    f_stride = yaw_nl / (4 * YAW_AMP)
    if yaw_kind == "servo":
        f_stride = max(f_stride, SEA_FNAT)          # spring rescue on the sprung axis
    # lift cadence: retract+extend 2*lift per swing; swing=(1-DUTY)/f
    lift_nl_lin = axis_noload(lift_kind, motor, Nlift) * CRANK      # m/s foot
    f_lift = lift_nl_lin * (1 - DUTY) / (2 * lift)
    f = min(f_stride, f_lift)
    v = stride * f
    return v ** 2 / (G * L), v, ("lift" if f_lift < f_stride else "stride")


def mass_mixed(placement, motor):
    base = TORSO_MASS + 4 * LEG_MASS
    return base + 4 * BLDC[motor]["mass"] + 8 * SERVO["mass"]


def evaluate(theta, placement, motor):
    """theta = (half_len, half_wid, yaw_lever, stance_h, lift_range, Nyaw, Nlift)."""
    hl, hw, lev, H, lift, Nyaw, Nlift = theta
    geom = dict(half_len=hl, half_wid=hw, yaw_lever=lev, stance_h=H, lift_range=lift)
    margin = static_margin([hl, hw, lev, H, lift, 12, 6]) * 1000   # geom-only
    fr, v, limiter = gait_froude(geom, placement, motor, Nyaw, Nlift)
    # combat stomp = knee/lift peak torque (servo unless lift is the BLDC)
    stomp = axis_torque("bldc" if placement == "lift" else "servo", motor, Nlift, peak=True)
    return dict(margin=margin, fr=fr, v=v, limiter=limiter, stomp=stomp,
                mass=mass_mixed(placement, motor), cost=4*BLDC[motor]["cost"]+8*SERVO["cost"])


BOUNDS = [(0.10, 0.20), (0.06, 0.12), (0.05, 0.16), (0.35, 0.50),
          (0.03, 0.06), (6.0, 16.0), (6.0, 16.0)]   # geom + Nyaw + Nlift


def optimize(placement, motor, w):
    from scipy.optimize import differential_evolution

    def obj(theta):
        e = evaluate(theta, placement, motor)
        pen = 0.0
        if e["margin"] < 5:
            pen += 1e3 * (5 - e["margin"])
        if e["stomp"] < 6:                     # retain a plant/stomp
            pen += 1e3 * (6 - e["stomp"])
        return -(w["dyn"]*e["fr"]*100 + w["stab"]*e["margin"]/10 - w["mass"]*e["mass"]) + pen
    r = differential_evolution(obj, BOUNDS, seed=0, popsize=15, maxiter=40, tol=1e-6, polish=True)
    return r.x, evaluate(r.x, placement, motor)


def report():
    print(f"MIXED CONFIG co-design: 1 BLDC + 2 servos/leg, bus {V_BUS:.0f}V, "
          f"dynamic threshold Fr>={FR_DYNAMIC}")
    print(f"servo: REEFS-class {SERVO['free']:.0f} rad/s free, {SERVO['stall']:.1f} N·m stall; "
          f"SEA belt resonance {SEA_FNAT:.1f} Hz\n")
    w = dict(dyn=1.0, stab=0.6, mass=0.5)
    motor = "D5312s"
    print(f"  Where to spend the ONE BLDC (motor={motor}):")
    print(f"  {'placement':<10} {'Fr':>6} {'v[m/s]':>6} {'limiter':>8} {'stomp':>6} "
          f"{'mass':>6} {'Nyaw':>5} {'Nlift':>5}  verdict")
    best = None
    for placement in AXES:
        th, e = optimize(placement, motor, w)
        tag = "DYNAMIC" if e["fr"] >= FR_DYNAMIC else "quasi-static"
        print(f"  {placement:<10} {e['fr']:>6.3f} {e['v']:>6.2f} {e['limiter']:>8} "
              f"{e['stomp']:>5.0f}N {e['mass']:>5.1f}kg {th[5]:>4.0f}:1 {th[6]:>4.0f}:1  {tag}")
        if best is None or e["fr"] > best[1]["fr"]:
            best = (placement, e, th)
    bp, be, bt = best
    print(f"\n  BEST BLDC PLACEMENT: {bp}  (Fr {be['fr']:.3f}, v {be['v']:.2f} m/s, "
          f"{be['limiter']}-limited, {be['mass']:.1f} kg)")
    print(f"  geometry: stance {2*bt[0]*100:.0f}x{2*(bt[1]+bt[2])*100:.0f}cm, lever "
          f"{bt[2]*100:.0f}cm, lift {bt[4]*1000:.0f}mm; Nyaw {bt[5]:.0f}:1 Nlift {bt[6]:.0f}:1")
    print("\n  READ: the gait ceiling is set by the SLOWEST gait-critical axis. The one")
    print("  BLDC should go on that limiter. With the SEA belt on the yaw, a SERVO yaw")
    print("  is rescued by the spring (Level B), which frees the BLDC for the OTHER")
    print("  limiter (the lift/knee cadence) — often the higher-Fr placement. This is")
    print("  the quantitative answer to 'yaw vs lift for the single BLDC'.")
    # light-vs-capable across motors on the best placement
    print(f"\n  Motor choice on the '{bp}' axis (mass vs headroom):")
    for m in BLDC:
        th, e = optimize(bp, m, w)
        print(f"    {m:<8} Fr {e['fr']:.3f}  mass {e['mass']:.1f}kg  ${e['cost']:.0f}")
    print("\n  Level C sim (walker env: BLDC torque on the chosen axis + servo PD on the")
    print("  other two, obs/action unchanged) verifies before hardware.")


if __name__ == "__main__":
    report()


# ---------------------------------------------------------------------------
# HISTORICAL MIXED-ACTUATOR OPTIMUM (2026-07-05). This remains useful as a design
# comparison, but it is no longer emitted as robot_design.TARGET: the hardware BOM
# is now fixed to twelve Waveshare ST3215-HS servos and a 6 lb total-mass limit.
# ---------------------------------------------------------------------------
OPTIMUM = dict(
    placement="lift", bldc="D5312s", servo="reefs_400sc",
    half_len=0.20, half_wid=0.12, yaw_lever=0.16, stance_h=0.42,
    lift_range=0.047, lift_nom=0.0235,
    Nyaw=11.0, Npitch=10.0, Nlift=15.0, sea_k_tilde=25.0, fr=0.084,
)


def emit_historical_mixed_optimum():
    """Package the superseded mixed BLDC/servo optimum for comparison runs."""
    from robot_design import RobotDesign, Axis
    o = OPTIMUM
    return RobotDesign(
        name="historical_mixed", half_len=o["half_len"], half_wid=o["half_wid"],
        yaw_lever=o["yaw_lever"], stance_h=o["stance_h"],
        lift_range=o["lift_range"], lift_nom=o["lift_nom"],
        yaw=Axis("servo_sea", "reefs_400sc", o["Nyaw"], sea_k_tilde=o["sea_k_tilde"]),
        pitch=Axis("servo", "reefs_400sc", o["Npitch"]),
        knee=Axis("bldc", "odrive_d5312s", o["Nlift"], linear=True),
    )
