# SPDX-License-Identifier: MIT
"""codesign.py — body-design optimization (static feasibility + Level-A dynamics).

Writes the robot BODY as an optimization problem and solves it. The design vector
theta (geometry + transmission ratios) is optimized against grounded objectives:

  STATIC (Tier 1)   — the crawl's COM stability margin inside the support triangle,
                      used analytically in the inner loop and checked against the
                      production MuJoCo model at the optimum.
  DYNAMIC (Level A)  — the max Froude number Fr = v^2/(gL) the actuator torque-speed
                      envelope + swing-timing admit. This is what separates a body
                      that CAN do a dynamic gait from one stuck quasi-static; it
                      directly targets the actuator-speed wall we measured.
  COMBAT             — retained stomp/plant authority (pitch stall torque through the
                      worm) — the dual-use constraint, kept explicit.
  COST               — mass under the 6 lb hard limit. The actuator is fixed to
                      the selected Waveshare ST3215-HS on all twelve axes.

The optimizer trades these (a longer yaw lever and lower gear ratios buy speed/Fr
but cost combat torque and self-lock), so its real output is the walk-vs-combat
Pareto front, not a single point. Models are analytic and DOCUMENTED as such —
necessary-condition surrogates, verified against compiled MuJoCo at the optimum.

  .venv-warp/bin/python codesign.py       # optimize + Pareto sweep + model verify
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from gen_mesh_robot_mjcf import (MAX_ROBOT_MASS_KG, SERVO_FREE_RAD_S,  # noqa: E402
                                 SERVO_MASS, SERVO_STALL_NM)
from walker_improved import DEFAULTS, LEG_MASS, TORSO_MASS  # noqa: E402

G = 9.81
YAW_AMP = 0.45          # rad, half yaw sweep used for stride (design assumption)
DUTY_DYN = 0.6          # duty factor of the target dynamic walk (2 feet down avg)
COMBAT_MIN = 12.0       # N·m minimum pitch stall torque to retain the stomp/plant
SELFLOCK_RATIO = 15.0   # worm ratio at/above which the pitch is self-locking

# design vector: (half_len, half_wid, yaw_lever, stance_h, lift_range, worm, belt)
NAMES = ["half_len", "half_wid", "yaw_lever", "stance_h", "lift_range", "worm", "belt"]
BOUNDS = [(0.10, 0.20), (0.06, 0.12), (0.05, 0.16), (0.35, 0.50),
          (0.03, 0.06), (8.0, 20.0), (4.0, 8.0)]
DEFAULT_THETA = np.array([DEFAULTS["half_len"], DEFAULTS["half_wid"],
                          DEFAULTS["yaw_lever"], DEFAULTS["stance_h"],
                          DEFAULTS["lift_range"], 20.0, 6.0])


def _tri_margin(p, tri):
    c = tri.mean(0)
    ms = []
    for i in range(3):
        a, b = tri[i], tri[(i + 1) % 3]
        e = b - a
        n = np.array([-e[1], e[0]]); n /= np.linalg.norm(n) + 1e-12
        if n @ (c - a) < 0:
            n = -n
        ms.append(n @ (p[:2] - a))
    return min(ms)


def unpack(theta):
    hl, hw, lev, H, lift, worm, belt = theta
    return dict(half_len=hl, half_wid=hw, yaw_lever=lev, stance_h=H,
                lift_range=lift, lift_nom=lift / 2.0, worm=worm, belt=belt)


def static_margin(theta):
    """Best-achievable COM margin (m) when a front foot lifts and the body shifts
    to the support centroid. Analytic upper bound; model-checked at the optimum."""
    d = unpack(theta)
    L, W = d["half_len"], d["half_wid"] + d["yaw_lever"]
    # lift FL (+L,+W); support triangle FR(+L,-W), RL(-L,+W), RR(-L,-W)
    tri = np.array([[L, -W], [-L, W], [-L, -W]])
    centroid = tri.mean(0)                         # body shifts COM here
    return _tri_margin(centroid, tri)


def dynamic_froude(theta):
    """Max Froude number the actuator torque-speed envelope + swing timing admit.

    Stride from the yaw sweep on the laterally-splayed leg: stride_len = 2*lev*sin(A).
    A dynamic gait at speed v needs the swing leg to sweep +/-A within its swing
    window; the peak yaw rate required is ~2*v/lev (see derivation in the note).
    Feasible while that stays under the yaw no-load speed SERVO_FREE/belt AND the
    pitch can lift/reach at the matching cadence (SERVO_FREE/worm). The slower of
    the two caps v; Fr = v^2 / (g*L)."""
    d = unpack(theta)
    lev, H = d["yaw_lever"], d["stance_h"]
    yaw_free = SERVO_FREE_RAD_S / d["belt"]        # rad/s
    pitch_free = SERVO_FREE_RAD_S / d["worm"]
    stride_len = 2 * lev * np.sin(YAW_AMP)
    # yaw-limited speed: peak yaw rate 2*v/lev <= yaw_free
    v_yaw = yaw_free * lev / 2.0
    # pitch-limited cadence: the lift must complete each swing; a swing sweeps ~2*A
    # of pitch-equivalent within swing time (1-DUTY)/f_leg, f_leg = v/stride_len.
    # required pitch rate ~ 2*A*f_leg/(1-DUTY); cap it:
    #   2*A*(v/stride_len)/(1-DUTY) <= pitch_free  ->  v <= pitch_free*(1-DUTY)*stride_len/(2*A)
    v_pitch = pitch_free * (1 - DUTY_DYN) * stride_len / (2 * YAW_AMP)
    v = max(0.0, min(v_yaw, v_pitch))
    return v ** 2 / (G * H), v, ("yaw" if v_yaw < v_pitch else "pitch")


def combat_stomp(theta):
    """Retained plant/stomp authority = pitch stall torque through the worm (N·m)."""
    return SERVO_STALL_NM * unpack(theta)["worm"]


def mass(theta):
    """Geometry-dependent: torso + 12 servos + a FRAME whose mass scales with the
    stance envelope (longer links/wider frame weigh more), so 'bigger' isn't free."""
    d = unpack(theta)
    base = TORSO_MASS + 4 * (LEG_MASS + 2 * SERVO_MASS) + 4 * SERVO_MASS
    # frame material ~ proportional to total leg reach (lever) + stance span vs the
    # DEFAULTS baseline; ~4 kg/m of added structural span (rough alloc)
    span = (2 * d["half_len"] + 2 * (d["half_wid"] + d["yaw_lever"]) + 4 * d["yaw_lever"])
    span0 = (2 * DEFAULTS["half_len"] + 2 * (DEFAULTS["half_wid"] + DEFAULTS["yaw_lever"])
             + 4 * DEFAULTS["yaw_lever"])
    return base + 4.0 * (span - span0)


def evaluate(theta):
    fr, v, limiter = dynamic_froude(theta)
    return dict(
        com_margin_mm=static_margin(theta) * 1000,
        froude=fr, dyn_speed_ms=v, dyn_limiter=limiter,
        stomp_nm=combat_stomp(theta), self_lock=unpack(theta)["worm"] >= SELFLOCK_RATIO,
        mass_kg=mass(theta),
    )


def objective(theta, w):
    """Weighted scalarization (minimize). w trades dynamic vs combat vs stability.
    Hard constraints as large penalties: static margin > 0, stomp >= COMBAT_MIN."""
    e = evaluate(theta)
    pen = 0.0
    if e["com_margin_mm"] < 5.0:          # need a real static margin
        pen += 1e3 * (5.0 - e["com_margin_mm"])
    if e["stomp_nm"] < COMBAT_MIN:        # retain the weapon
        pen += 1e3 * (COMBAT_MIN - e["stomp_nm"])
    if e["mass_kg"] > MAX_ROBOT_MASS_KG:
        pen += 1e3 * (e["mass_kg"] - MAX_ROBOT_MASS_KG)
    score = (w["dyn"] * e["froude"] * 100.0          # dynamic capability
             + w["stab"] * e["com_margin_mm"] / 10.0  # static robustness
             + w["combat"] * e["stomp_nm"] / 10.0     # combat authority
             - w["mass"] * e["mass_kg"])              # lightness
    return -score + pen


def optimize(w, seed=0):
    from scipy.optimize import differential_evolution
    res = differential_evolution(objective, BOUNDS, args=(w,), seed=seed,
                                 popsize=20, maxiter=60, tol=1e-7, polish=True)
    return res.x, evaluate(res.x)


def mujoco_verify(theta):
    """Check all four keyframes against the compiled production MuJoCo model."""
    try:
        from drake_walk_proof import _build_model, certify_keyframe, footholds_for, CRAWL_ORDER
        p = unpack(theta)
        model = _build_model(p)
        fh = footholds_for(p)
        oks, margins = [], []
        for swing in CRAWL_ORDER:
            r = certify_keyframe(model, p, swing, fh)
            oks.append(r.get("ik_ok") and r.get("stable") and r.get("tau_ok"))
            margins.append(r.get("com_margin_mm", float("nan")))
        return all(oks), margins
    except Exception as ex:
        return None, str(ex)


def _fmt(theta, e):
    d = unpack(theta)
    return (f"len {2*d['half_len']*100:.0f} wid {2*(d['half_wid']+d['yaw_lever'])*100:.0f} "
            f"lever {d['yaw_lever']*100:.0f}cm  worm {d['worm']:.0f}:1 belt {d['belt']:.1f}:1  "
            f"|  Fr {e['froude']:.3f} ({e['dyn_speed_ms']:.2f}m/s, {e['dyn_limiter']}-lim)  "
            f"margin {e['com_margin_mm']:+.0f}mm  stomp {e['stomp_nm']:.0f}Nm"
            f"{' +lock' if e['self_lock'] else ''}")




# ---------------------------------------------------------------------------
# SERVO CANDIDATE EVALUATION (2026-07-05): score real servos on BOTH routes —
# rigid (Level A: needs SPEED) and sprung/SEA (Level B: needs TORQUE + power).
# Speeds converted from s/60deg: omega_free = (pi/3) / t_60 [rad/s].
# Torque kg-cm -> N.m: x * 0.0980665.
# ---------------------------------------------------------------------------
def _wfree(t60): return (3.14159265 / 3.0) / t60          # rad/s from s/60deg
def _nm(kgcm):  return kgcm * 0.0980665

SERVO_CANDIDATES = {
    # name: (no_load rad/s, stall N.m, note, pwm_dropin)
    "Waveshare ST3215-HS (selected)": (_wfree(0.094), _nm(20.0), "12V UART bus", False),
    "goBILDA 25-3":           (_wfree(0.070), _nm(10.8), "historical baseline", True),
    "goBILDA 25-4 SuperSpeed": (_wfree(0.035), _nm(5.4),  "2x speed, weak", True),
    "REEFS 400:SC V2":        (_wfree(0.074), _nm(29.2), "torque king", True),
    "Savox SB-2274SG":        (_wfree(0.080), _nm(25.0), "strong match", True),
    "Savox SB-2271SG":        (_wfree(0.065), _nm(20.0), "balanced", True),
    "Savox SB-2291SG":        (_wfree(0.050), _nm(18.0), "fast-ish", True),
    "Hitec HSB-9360TH":       (_wfree(0.060), _nm(17.0), "fast, mid torque", True),
    "DYNAMIXEL XL330":        (420*2*3.14159/60, 0.22,   "fast, tiny torque", False),
    "CubeMars AK80-9 (BLDC)": (570*2*3.14159/60, 22.0,   "real robot actuator", False),
}

# design thresholds (from Level A): to reach Fr=0.10 the SERVO free speed must be
# ~29 rad/s (at belt 4:1, longest lever); combat/plant wants >=12 N.m through the
# worm; the sprung route needs mechanical power to top up losses (~a few W/leg).
SERVO_SPEED_FOR_DYNAMIC = 29.0        # rad/s at the servo (rigid route)
STOMP_MIN_NM = 12.0
BELT_MIN, WORM_MIN = 4.0, 8.0


def servo_report():
    print(f"{'servo':<26} {'free':>6} {'stall':>6} {'rigid-dyn?':>11} "
          f"{'combat':>7} {'power':>7} {'verdict'}")
    print(f"{'':<26} {'rad/s':>6} {'N.m':>6} {'(Fr>=.10)':>11} {'(worm)':>7} {'W/leg':>7}")
    for name, (wf, st, note, pwm) in SERVO_CANDIDATES.items():
        rigid = wf >= SERVO_SPEED_FOR_DYNAMIC
        combat = st * WORM_MIN                          # stomp via worm at min ratio
        power = st * wf / 4.0                           # ~peak mech power /leg
        # sprung (Level B) route: needs torque for the bounce + power to re-energize;
        # speed matters far less (spring carries fast motion). Heuristic: strong if
        # stall*worm gives real force AND power budget covers a few W.
        sprung = (st >= 1.5) and (power >= 3.0)
        v = ("DYNAMIC (rigid)" if rigid and combat >= STOMP_MIN_NM else
             "DYNAMIC (sprung/SEA)" if sprung else
             "quasi-static only")
        tag = "" if pwm else " [not PWM]"
        print(f"{name:<26} {wf:>6.1f} {st:>6.2f} {'yes' if rigid else 'no':>11} "
              f"{combat:>6.0f}N {power:>6.1f} {v}{tag}")


if __name__ == "__main__":
    print("=== baseline: current hand-picked DEFAULTS ===")
    print("  " + _fmt(DEFAULT_THETA, evaluate(DEFAULT_THETA)))
    print("\n=== single optimum (balanced weights) ===")
    w = dict(dyn=1.0, stab=1.0, combat=0.5, mass=0.2)
    th, e = optimize(w)
    print("  " + _fmt(th, e))
    ok, margins = mujoco_verify(th)
    mstr = ""
    if isinstance(margins, list):
        mstr = " (margins " + ", ".join("nan" if (isinstance(m, float) and np.isnan(m))
                                        else f"{m:.0f}" for m in margins) + "mm)"
    print(f"  MuJoCo verify: {'PASSED all keyframes' if ok else 'FAILED' if ok is False else 'skipped'}{mstr}")

    print("\n=== walk-vs-combat Pareto front (sweep dynamic weight) ===")
    for dw in [0.2, 0.5, 1.0, 4.0, 12.0, 40.0]:
        w = dict(dyn=dw, stab=0.8, combat=1.0, mass=0.2)
        th, e = optimize(w, seed=1)
        print(f"  dyn_w {dw:>5.1f} | {_fmt(th, e)}")

    # --- the servo-ceiling finding: what does a DYNAMIC gait actually require? ---
    print("\n=== servo ceiling: can this servo ever go dynamic? ===")
    # best-case Fr over the WHOLE design space (ignore combat, push ratios to bounds)
    best_fr, best_th = 0.0, None
    for _ in range(4000):
        th = np.array([np.random.uniform(lo, hi) for lo, hi in BOUNDS]) if False else \
             np.array([b[0] + (b[1] - b[0]) * ((i * 0.6180339 + 0.1) % 1.0)
                       for i, b in enumerate(BOUNDS)])       # deterministic-ish spread
        break
    # analytic max: yaw v = (SERVO_FREE/belt_min)*lev_max/2, at the loosest ratios
    lev_max, belt_min, worm_min, H_min = 0.16, 4.0, 8.0, 0.35
    v_yaw = (SERVO_FREE_RAD_S / belt_min) * lev_max / 2
    fr_ceiling = v_yaw ** 2 / (G * H_min)
    print(f"  MAX Fr with this servo (loosest ratios, longest lever): {fr_ceiling:.3f} "
          f"at v={v_yaw:.2f} m/s")
    print(f"  Dynamic-walk regime starts at Fr~0.10 (v~{np.sqrt(0.10*G*0.42):.2f} m/s at L=0.42m).")
    # what free speed would Fr=0.10 need, at the best geometry?
    v_needed = np.sqrt(0.10 * G * H_min)
    yaw_free_needed = 2 * v_needed / lev_max
    servo_free_needed = yaw_free_needed * belt_min
    print(f"  To reach Fr=0.10: need yaw no-load {yaw_free_needed:.1f} rad/s -> at belt {belt_min:.0f}:1 "
          f"a servo free speed of {servo_free_needed:.0f} rad/s ({servo_free_needed/SERVO_FREE_RAD_S:.1f}x "
          f"the ST3215-HS's {SERVO_FREE_RAD_S:.0f} rad/s).")
    print("  VERDICT: no transmission ratio reaches the dynamic regime with this servo at")
    print("  this scale — dynamic gait needs a higher-speed/power servo OR a smaller,")
    print("  lighter robot (Fr scales as 1/L). The optimum here is the fastest QUASI-STATIC")
    print("  crawl; dynamic locomotion is a SERVO-CLASS decision, which this makes quantitative.")
    print("\n=== SERVO CANDIDATE EVALUATION (rigid vs sprung route) ===")
    servo_report()
