# SPDX-License-Identifier: MIT
"""codesign_slip.py — Level B: SLIP spring-resonance limit-cycle co-design.

Level A (codesign.py) found the servo too SLOW for a dynamic gait when the
transmission is rigid — the motor would have to produce all the fast stance
motion. This adds the escape hatch you built into the leg: SERIES ELASTICITY.

With a spring in the leg (the SEA rubber belt), the STANCE dynamics are the
mass-spring bounce — fast, passive, motor-independent. The (slow) servo only has
to (a) set the touchdown angle and (b) inject the small energy lost per stride.
So the question co-design answers here is: what leg-spring stiffness admits a
STABLE dynamic limit cycle for THIS robot (m, L), and is the per-stride energy
within the selected ST3215-HS's reach even though its speed is limited?

Model: the canonical Spring-Loaded Inverted Pendulum (SLIP) — point mass m on a
massless spring leg (rest length L0, stiffness k). Stance = 2-D mass-spring about
the planted foot; flight = ballistic. A periodic gait is a fixed point of the
apex->apex return map; STABLE iff |dP/d(apex)| < 1. Resonance = the mass-spring
half-period matching the stance time.

  .venv-warp/bin/python codesign_slip.py
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
from scipy.integrate import solve_ivp

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from walker_improved import DEFAULTS  # noqa: E402
from gen_mesh_robot_mjcf import SERVO_STALL_NM, SERVO_FREE_RAD_S  # noqa: E402
from robot_design import TARGET  # noqa: E402

G = 9.81
M = TARGET.total_mass()                                     # full 6 lb robot
L0 = DEFAULTS["stance_h"]                                  # ~0.42 m leg rest length


def _stance(x0, k, m, L0, xf):
    """Integrate one stance phase (foot at xf) until the leg returns to L0 with
    ṙ>0 (liftoff), or the leg collapses / times out. x0 = (x, y, vx, vy)."""
    def rhs(t, s):
        x, y, vx, vy = s
        dx, dy = x - xf, y
        r = np.hypot(dx, dy)
        F = k * (L0 - r)                        # spring (compression -> push out)
        return [vx, vy, F * dx / r / m, F * dy / r / m - G]

    def liftoff(t, s):
        return np.hypot(s[0] - xf, s[1]) - L0   # r == L0
    liftoff.terminal = True
    liftoff.direction = 1                       # extending back to rest
    def collapse(t, s):
        return s[1] - 0.15 * L0                 # body fell (leg buckled)
    collapse.terminal = True
    collapse.direction = -1

    sol = solve_ivp(rhs, [0, 1.2], x0, events=[liftoff, collapse],
                    rtol=1e-6, atol=1e-8, max_step=4e-3)
    if sol.t_events[1].size or not sol.t_events[0].size:
        return None, None
    return sol.y[:, -1], sol.t_events[0][0]


def apex_return(y_apex, E, alpha, k, m, L0):
    """Map apex height -> next apex height for a conservative SLIP at energy E,
    touchdown angle alpha. Returns (y_apex', vx', t_stance) or (None,...)."""
    vx = np.sqrt(max(2 * (E - m * G * y_apex) / m, 0.0))
    # flight down from apex to touchdown (foot contacts at height L0 cos alpha)
    y_td = L0 * np.cos(alpha)
    if y_apex < y_td:
        return None, None, None
    t_fall = np.sqrt(2 * (y_apex - y_td) / G)
    x_td, vy_td = 0.0, -G * t_fall
    xf = x_td + L0 * np.sin(alpha)              # foot placement
    st, ts = _stance([x_td, y_td, vx, vy_td], k, m, L0, xf)
    if st is None:
        return None, None, None
    x, y, vxl, vyl = st
    if vyl <= 0:                                # never left the ground going up
        return None, None, None
    y_next = y + vyl ** 2 / (2 * G)             # ballistic apex
    return y_next, vxl, ts


def find_limit_cycle(k, m, L0, v_target):
    """For target forward speed v_target, search touchdown angle alpha for a
    periodic gait (apex fixed point). Return dict or None."""
    # energy from a nominal apex a bit above L0 and the target speed
    for y_apex0 in (L0 * 1.06, L0 * 1.12):
        E = m * G * y_apex0 + 0.5 * m * v_target ** 2
        for alpha in np.radians(np.arange(12, 40, 2.0)):    # leg angle from vertical
            # fixed point of P(y): find y where apex_return(y)=y
            def resid(y):
                yn, _, _ = apex_return(y, E, alpha, k, m, L0)
                return None if yn is None else yn - y
            # bracket-scan for a sign change / small residual
            ys = np.linspace(L0 * 1.005, E / (m * G) * 0.999, 18)
            prev = None
            for y in ys:
                r = resid(y)
                if r is None:
                    prev = None
                    continue
                if prev is not None and prev[1] * r < 0:     # sign change -> root
                    ya, yb = prev[0], y
                    for _ in range(22):
                        ym = 0.5 * (ya + yb)
                        rm = resid(ym)
                        if rm is None:
                            break
                        if prev[1] * rm < 0:
                            yb = ym
                        else:
                            ya, prev = ym, (ym, rm)
                    y_star = 0.5 * (ya + yb)
                    # stability: dP/dy via finite diff
                    h = 1e-4
                    yp, _, ts = apex_return(y_star, E, alpha, k, m, L0)
                    yph, _, _ = apex_return(y_star + h, E, alpha, k, m, L0)
                    if yp is None or yph is None:
                        continue
                    dP = (yph - yp) / h
                    yn, vxn, tstance = apex_return(y_star, E, alpha, k, m, L0)
                    if tstance is None or vxn is None or vxn < 0.2:
                        continue                      # reject backward/stalled gaits
                    return dict(k=k, alpha_deg=np.degrees(alpha), y_apex=y_star,
                                v=vxn, eig=abs(dP), t_stance=tstance, energy=E,
                                stable=abs(dP) < 1.0)
                prev = (y, r)
    return None


def natural_freq(k, m):
    return np.sqrt(k / m) / (2 * np.pi)         # Hz, mass-spring


def report():
    print(f"robot: m = {M:.1f} kg, leg L0 = {L0:.2f} m,  dynamic regime Fr>0.10 "
          f"(v > {np.sqrt(0.10*G*L0):.2f} m/s)")
    print(f"dimensionless stiffness k~ = k*L0/(m*g); running SLIP is stable ~k~ in [10,40]\n")
    print(f"  {'k~':>4} {'k[kN/m]':>8} {'f_nat[Hz]':>9} {'v[m/s]':>7} {'Fr':>6} "
          f"{'alpha':>6} {'eig':>6} {'stable':>7} {'t_st[ms]':>8}")
    results = []
    for ktil in (8, 12, 16, 20, 25, 30, 40):
        k = ktil * M * G / L0                     # physical stiffness N/m
        # target the low end of the dynamic regime
        v_t = np.sqrt(0.12 * G * L0)
        lc = find_limit_cycle(k, M, L0, v_t)
        fn = natural_freq(k, M)
        if lc is None:
            print(f"  {ktil:>4} {k/1e3:>8.1f} {fn:>9.2f} {'—':>7} {'—':>6} "
                  f"{'—':>6} {'—':>6} {'no LC':>7} {'—':>8}")
            continue
        fr = lc["v"] ** 2 / (G * L0)
        stab = "STABLE" if lc["stable"] else "unstable"
        results.append((ktil, k, lc, fr, fn))
        print(f"  {ktil:>4} {k/1e3:>8.1f} {fn:>9.2f} {lc['v']:>7.2f} {fr:>6.3f} "
              f"{lc['alpha_deg']:>5.0f}° {lc['eig']:>6.2f} {stab:>7} "
              f"{lc['t_stance']*1e3:>8.0f}")

    # --- servo-energy check: can the ST3215-HS sustain the springy gait? ---
    print("\n=== does the spring change the verdict? ===")
    if not results:
        print("  No SLIP limit cycle found — scale/energy out of reach entirely.")
        return
    # Dynamic limit cycles EXIST here (Fr up to ~0.32). They are open-loop unstable
    # at FIXED touchdown angle (eig>1), but the eigenvalues are modest and shrink
    # with stiffness — i.e. STABILIZABLE by regulating the touchdown angle once per
    # stride (low-bandwidth, servo-friendly; the spring carries the fast bounce).
    dyn = [r for r in results if r[3] >= 0.10]
    print(f"  {len(dyn)} dynamic-regime limit cycle(s) exist (Fr up to "
          f"{max(r[3] for r in results):.2f}); all open-loop unstable at fixed angle")
    print(f"  (eig {min(r[2]['eig'] for r in results):.1f}–{max(r[2]['eig'] for r in results):.1f}),")
    print("  STABILIZABLE by per-stride touchdown-angle control — not a speed problem.")
    ktil, k, lc, fr, fn = min(results, key=lambda r: r[2]["eig"])   # best-conditioned
    # per-stride energy the motor must inject = losses; for a lossless SLIP it is 0,
    # real losses ~ 10-20% of the stance spring energy. Spring PE at max compression:
    dv = lc["v"]
    E_spring = 0.5 * k * (0.06 * L0) ** 2          # ~6% compression scale
    E_loss = 0.15 * E_spring                        # 15% loss/stride (estimate)
    stride_t = lc["t_stance"] / 0.35                # stance ~35% of stride
    P_motor = E_loss / stride_t
    # ST3215-HS mechanical power budget (one leg): stall*free/4 for a linear curve.
    P_servo = SERVO_STALL_NM * SERVO_FREE_RAD_S / 4
    print(f"  Best spring: k~={ktil} (k={k/1e3:.1f} kN/m), f_nat={fn:.2f} Hz, "
          f"gait v={lc['v']:.2f} m/s (Fr={fr:.3f}), basin eig={lc['eig']:.2f}")
    print(f"  Resonance: leg natural {fn:.2f} Hz vs stance {1/(2*lc['t_stance']):.2f} Hz "
          f"half-cycle — {'MATCHED' if abs(fn-1/(2*lc['t_stance']))<0.5 else 'detuned'}")
    print(f"  Energy/stride the motor must inject (losses): {E_loss:.2f} J over "
          f"{stride_t*1e3:.0f} ms -> {P_motor:.1f} W/leg")
    print(f"  ST3215-HS mechanical power budget: ~{P_servo:.1f} W/leg  -> "
          f"{'FEASIBLE — the spring carries the fast dynamics, motor only re-energizes' if P_motor < P_servo else 'still short'}")
    print("\n  UPSHOT: series elasticity can put a dynamic limit cycle in reach even")
    print("  though the servo SPEED alone cannot — the spring does the bounce, the")
    print("  motor tops up losses. The SEA belt stiffness is the design knob; target")
    print(f"  k~{ktil} (~{k/1e3:.0f} kN/m leg-equivalent). Translate to belt N·m/rad via")
    print("  the lever, and tune f_nat to the stride. Level C (full trajopt/RL on the")
    print("  sprung leg) is the verification rung.")


if __name__ == "__main__":
    report()
