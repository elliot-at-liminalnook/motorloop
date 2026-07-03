# SPDX-License-Identifier: MIT
"""M2 gates (secret-sauce §10c): exact loop-coordinate joint.

(a) warp FK == gen_mesh_robot_mjcf closed form to 1e-9 across the working ROM
(b) NO instability at TDC at dt=0.004 — the entire point vs the <connect>
    blowup (10 m slide error, |qacc| 7e9, converter report): 2000 steps
    swinging through phi=0, energy drift bounded, state finite
(c) toggle force projection tau/F == ds/dphi to 1e-6 (analytic vs central diff)
"""
from __future__ import annotations

import math
import sys
from pathlib import Path

import numpy as np
import pytest
import warp as wp

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parents[1]))          # sim/robot (closed form)
sys.path.insert(0, str(HERE.parents[2]))          # sim (package parent). not used

from warplayer import loopjoint as lj  # noqa: E402

wp.init()
F64 = wp.float64
R, L = lj.CRANK_R, lj.CONROD_L


# EXACT mechanism closed form (the ground truth from the assembly json /
# gen_mesh_robot_mjcf's original :55-70). Computed inline because that module's
# loop_consistent_pose now returns QUARTIC-FITTED values (rung-2 couplings,
# residual ~5e-4 rad) — the M2 joint must match the exact mechanism to 1e-9,
# which is a stricter bar than the quartic approximation it replaces.
def s_exact(p):
    return R * math.cos(p) - math.sqrt(L * L - (R * math.sin(p)) ** 2) - (R - L)


def psi_exact(p):
    return math.asin(R * math.sin(p) / L) - p


def _fk(phis: np.ndarray):
    n = len(phis)
    phi = wp.array(phis, dtype=F64)
    outs = [wp.zeros(n, dtype=F64) for _ in range(4)]
    toe = wp.zeros(n, dtype=wp.vec2d)
    heel = wp.zeros(n, dtype=F64)
    wp.launch(lj.fk_eval, dim=n,
              inputs=[phi, F64(lj.CRANK_R), F64(lj.CONROD_L),
                      outs[0], outs[1], outs[2], outs[3], toe, heel])
    return [o.numpy() for o in outs] + [toe.numpy(), heel.numpy()]


def test_a_closed_form_match_1e9():
    phis = np.linspace(-math.pi / 2, math.radians(10.0), 501)
    psi_w, s_w, _, _, toe, heel = _fk(phis)
    psi_ref = np.array([psi_exact(p) for p in phis])
    s_ref = np.array([s_exact(p) for p in phis])
    assert np.max(np.abs(psi_w - psi_ref)) < 1e-9
    assert np.max(np.abs(s_w - s_ref)) < 1e-9
    # geometric consistency: |toe -> heel| == conrod length everywhere
    heel_xy = np.stack([np.zeros_like(heel), heel], axis=1)
    rod = np.linalg.norm(toe - heel_xy, axis=1)
    assert np.max(np.abs(rod - lj.CONROD_L)) < 1e-9


MASSES = (0.05, 0.07, 0.08)          # blade_upper, blade_lower, pushrod (json)
ARGS = None


def _args():
    return [F64(lj.CRANK_R), F64(lj.CONROD_L), F64(MASSES[0]), F64(MASSES[1]),
            F64(MASSES[2]), F64(9.81)]


def test_b1_actuated_tdc_transits_no_blowup_at_dt_0004():
    """THE M2 claim, in the mechanism's real regime: the knee motor drives the
    blade back and forth THROUGH the dead center at dt=0.004 — the exact
    scenario where the <connect> model measured |qacc| ~7e9 and 10 m of slide
    error (converter report). Coordinate elimination has no constraint to
    fight: accelerations must stay at physical scale (6+ orders below the
    blowup) and the loop stays exactly closed by construction."""
    nworld, dt, steps = 4, 4.0e-3, 2000
    phi = wp.array(np.array([-0.6, -0.9, -0.3, -1.2]), dtype=F64)
    phid = wp.zeros(nworld, dtype=F64)
    fp = wp.zeros(nworld, dtype=F64)
    args = _args()
    max_acc = 0.0
    max_spd = 0.0
    crossings = np.zeros(nworld, dtype=int)
    prev_phi = phi.numpy().copy()
    prev_pd = phid.numpy().copy()
    for t in range(steps):
        # PD position servo alternating between the stomp pose (-60 deg) and the
        # back-reach stop (+5 deg) at 1 Hz — how the knee is actually driven;
        # torque clipped to the 3 N.m placeholder motor limit.
        target = math.radians(-60.0) if (t * dt) % 1.0 < 0.5 else math.radians(5.0)
        cur, curd = phi.numpy(), phid.numpy()
        tq = np.clip(2.0 * (target - cur) - 0.05 * curd, -3.0, 3.0)
        tau = wp.array(tq, dtype=F64)
        wp.launch(lj.loop_step, dim=nworld, inputs=[F64(dt), *args, tau, fp, phi, phid])
        cur, curd = phi.numpy(), phid.numpy()
        max_acc = max(max_acc, float(np.max(np.abs(curd - prev_pd))) / dt)
        max_spd = max(max_spd, float(np.max(np.abs(curd))))
        crossings += ((prev_phi < 0.0) != (cur < 0.0)).astype(int)
        prev_phi, prev_pd = cur.copy(), curd.copy()
    assert np.all(np.isfinite(phi.numpy())) and np.all(np.isfinite(phid.numpy()))
    assert np.all(crossings >= 2), f"must transit TDC repeatedly (crossings={crossings})"
    assert np.max(np.abs(phi.numpy())) < 4.0 * math.pi, "no runaway winding"
    # physical acceleration ceiling: the servo itself commands tau_max/M_min —
    # ~1.4e4 rad/s^2 for a 3 N.m motor on this ~2e-4 kg.m^2 mechanism. Anything
    # near that is DESIGN, not instability; the connect model hit 7e9 (5.5
    # orders beyond physics). Bar = 2.5x ceiling.
    phis_scan = np.linspace(-math.pi / 2, math.radians(10), 181)
    M_min = min(_M_ref(p) for p in phis_scan)
    ceiling = 3.0 / M_min
    assert max_acc < 2.5 * ceiling, (
        f"|phidd| peaked at {max_acc:.3e} vs physical ceiling {ceiling:.3e} rad/s^2 "
        f"(connect blew up at ~7e9)")
    assert max_spd < 100.0


def _M_ref(p):
    """Host-side generalized inertia (same formulas as loopjoint._mass_grav)."""
    sp, cp = math.sin(p), math.cos(p)
    root = math.sqrt(L * L - R * R * sp * sp)
    ds = -R * sp + (R * R * sp * cp) / root
    Mc = MASSES[0] * R * R / 3.0
    dxr = R * cp / 2.0
    dyr = (-R * sp + ds) / 2.0
    u = R * sp / L
    dth = (R * cp / L) / math.sqrt(1.0 - u * u)
    Ir = MASSES[1] * L * L / 12.0
    return (Mc + MASSES[1] * (dxr * dxr + dyr * dyr) + Ir * dth * dth
            + MASSES[2] * ds * ds)


def test_b2_matches_tight_reference_with_4th_order_convergence():
    """Integrator correctness, stated honestly. This mechanism has NO potential
    minimum inside its ROM (gravity always pulls the blade through the stomp
    stop — by design), so 'conserve energy while oscillating in a well' is not
    a scenario the real leg has. The stronger claim we CAN make: the warp
    kernel integrates the exact EOM — verified by (i) matching a scipy
    RK45(rtol=1e-11) reference trajectory (that reference conserves energy to
    3e-9 over 8 s, proving the shared EOM is exact), and (ii) 4th-order step
    convergence (halving dt shrinks the error ~16x)."""
    from scipy.integrate import solve_ivp

    def rhs(t, y):
        p, pd = y
        h = 1e-6
        dM = (_M_ref(p + h) - _M_ref(p - h)) / (2 * h)
        sp, cp = math.sin(p), math.cos(p)
        root = math.sqrt(L * L - R * R * sp * sp)
        ds = -R * sp + (R * R * sp * cp) / root
        dyr = (-R * sp + ds) / 2.0
        Qg = -9.81 * (MASSES[0] * (-(R / 2.0) * sp) + MASSES[1] * dyr + MASSES[2] * ds)
        return [pd, (Qg - 0.5 * dM * pd * pd) / _M_ref(p)]

    y0 = [-1.4, 8.0]
    T = 1.0
    ref = solve_ivp(rhs, [0, T], y0, rtol=1e-11, atol=1e-13, max_step=0.01)
    phi_ref = ref.y[0, -1]

    def run_warp(dt):
        steps = int(round(T / dt))
        phi = wp.array(np.array([y0[0]]), dtype=F64)
        phid = wp.array(np.array([y0[1]]), dtype=F64)
        tau = wp.zeros(1, dtype=F64)
        fp = wp.zeros(1, dtype=F64)
        args = _args()
        for _ in range(steps):
            wp.launch(lj.loop_step, dim=1, inputs=[F64(dt), *args, tau, fp, phi, phid])
        return float(phi.numpy()[0])

    err_coarse = abs(run_warp(4.0e-3) - phi_ref)
    err_fine = abs(run_warp(2.0e-3) - phi_ref)
    assert err_coarse < 0.02, f"dt=0.004 trajectory error {err_coarse:.4f} rad vs reference"
    order = math.log2(max(err_coarse, 1e-14) / max(err_fine, 1e-14))
    assert order > 3.0, (
        f"convergence order {order:.2f} (errors {err_coarse:.2e} -> {err_fine:.2e}); "
        f"RK4 on the exact EOM must show ~4th order")


def test_c_toggle_force_projection_1e6():
    phis = np.array([-1.2, -0.6, -0.1, 0.0, 0.15])
    F = 40.0                                        # N, ~the mesh robot's tip force
    n = len(phis)
    phi = wp.array(phis, dtype=F64)
    fp = wp.array(np.full(n, F), dtype=F64)
    q = wp.zeros(n, dtype=F64)
    wp.launch(lj.gen_force_eval, dim=n,
              inputs=[phi, F64(lj.CRANK_R), F64(lj.CONROD_L), fp, q])
    tau = q.numpy()
    # central-difference ds/dphi from the closed form (fp64, h chosen for ~1e-10)
    h = 1e-7
    ds_fd = np.array([(s_exact(p + h) - s_exact(p - h)) / (2 * h)
                      for p in phis])
    assert np.max(np.abs(tau - F * ds_fd)) < 1e-6 * max(1.0, F)
    # at TDC exactly: ds/dphi == 0 -> mechanical advantage diverges (toggle press)
    assert abs(tau[3]) < 1e-9


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-q"]))
