# SPDX-License-Identifier: MIT
"""warplayer.loopjoint — M2: the EXACT loop-coordinate joint (secret-sauce §8, §10c(i)).

The slider-crank blade foot (crank r=75 mm, conrod L=100 mm — constants and
closed form identical to sim/robot/gen_mesh_robot_mjcf.py:55-70) is a 1-DOF
mechanism. Every constraint-based engine represents it as extra DOFs + an
equality (3-row <connect> in MuJoCo — whose Jacobian goes rank-deficient at the
toggle's dead center, forcing dt=0.002; report §8). Here we do what the report
calls "fundamentally better": COORDINATE ELIMINATION. psi(phi), s(phi) and their
derivatives are evaluated analytically *inside kinematics*; the passive bodies
are placed kinematically from the crank angle phi; dynamics run in the reduced
1-DOF coordinate with the projected generalized inertia

    M(phi) = I_crank + m_rod |dCOM_rod/dphi|^2 + I_rod (dtheta_rod/dphi)^2
             + m_piston (ds/dphi)^2

so there is NO constraint row, NO singular direction, and the toggle-press
force amplification tau = F * ds/dphi appears exactly, by construction.
Float64 throughout: the tests pin the closed form to 1e-9 and the force
projection to 1e-6 (production fp32 variant is a mechanical port).

Frame (planar test rig): crank pivot at origin, +y up, gravity -y. phi=0 =
blade vertical, toe UP (top dead center); toe = (r sin phi, r cos phi); heel
slides on the x=0 lane at heel_y = r cos phi - sqrt(L^2 - r^2 sin^2 phi);
s(phi) = heel_y(phi) - heel_y(0)  ==  gen_mesh_robot_mjcf.slider_crank_s.
"""
from __future__ import annotations

import warp as wp

CRANK_R = 0.075
CONROD_L = 0.100
F64 = wp.float64


@wp.func
def crank_kin(phi: F64, r: F64, L: F64) -> wp.vec4d:
    """(psi, s, dpsi_dphi, ds_dphi) — the whole loop, analytically."""
    sp = wp.sin(phi)
    cp = wp.cos(phi)
    u = r * sp / L
    root = wp.sqrt(L * L - r * r * sp * sp)          # = L*sqrt(1-u^2)
    psi = wp.asin(u) - phi
    s = r * cp - root - (r - L)
    dpsi = (r * cp / L) / wp.sqrt(F64(1.0) - u * u) - F64(1.0)
    ds = -r * sp + (r * r * sp * cp) / root
    return wp.vec4d(psi, s, dpsi, ds)


@wp.func
def _mass_grav(phi: F64, r: F64, L: F64, m_c: F64, m_r: F64, m_p: F64,
               g: F64) -> wp.vec2d:
    """(M(phi), Qg(phi)): projected generalized inertia + gravity force."""
    k = crank_kin(phi, r, L)
    ds = k[3]
    sp = wp.sin(phi)
    cp = wp.cos(phi)
    # crank: uniform rod about its pivot
    Mc = m_c * r * r / F64(3.0)
    # conrod COM = midpoint(toe, heel); dCOM/dphi
    dxr = r * cp / F64(2.0)
    dyr = (-r * sp + ds) / F64(2.0)
    u = r * sp / L
    dtheta = (r * cp / L) / wp.sqrt(F64(1.0) - u * u)     # conrod tilt rate
    Ir = m_r * L * L / F64(12.0)
    Mr = m_r * (dxr * dxr + dyr * dyr) + Ir * dtheta * dtheta
    Mp = m_p * ds * ds
    # gravity: Qg = -g * d(m_i y_i)/dphi ; y_c' = -(r/2) sp ; y_r' = dyr ; y_p' = ds
    Qg = -g * (m_c * (-(r / F64(2.0)) * sp) + m_r * dyr + m_p * ds)
    return wp.vec2d(Mc + Mr + Mp, Qg)


@wp.kernel
def fk_eval(phi: wp.array(dtype=F64), r: F64, L: F64,
            out_psi: wp.array(dtype=F64), out_s: wp.array(dtype=F64),
            out_dpsi: wp.array(dtype=F64), out_ds: wp.array(dtype=F64),
            out_toe: wp.array(dtype=wp.vec2d), out_heel_y: wp.array(dtype=F64)):
    i = wp.tid()
    p = phi[i]
    k = crank_kin(p, r, L)
    out_psi[i] = k[0]
    out_s[i] = k[1]
    out_dpsi[i] = k[2]
    out_ds[i] = k[3]
    out_toe[i] = wp.vec2d(r * wp.sin(p), r * wp.cos(p))
    out_heel_y[i] = (r - L) + k[1]                    # heel_y(0) = r - L


@wp.func
def _accel(q: F64, qd: F64, tau: F64, fp: F64,
           r: F64, L: F64, m_c: F64, m_r: F64, m_p: F64, g: F64) -> F64:
    """Reduced EOM:  M(q) qdd + (1/2)M'(q) qd^2 + V'(q) = tau + F*ds/dq."""
    mg = _mass_grav(q, r, L, m_c, m_r, m_p, g)
    h = F64(1.0e-6)
    mp_ = _mass_grav(q + h, r, L, m_c, m_r, m_p, g)
    mm_ = _mass_grav(q - h, r, L, m_c, m_r, m_p, g)
    dM = (mp_[0] - mm_[0]) / (F64(2.0) * h)
    k = crank_kin(q, r, L)
    return (tau + fp * k[3] + mg[1] - F64(0.5) * dM * qd * qd) / mg[0]


@wp.kernel
def loop_step(dt: F64, r: F64, L: F64, m_c: F64, m_r: F64, m_p: F64, g: F64,
              tau: wp.array(dtype=F64), f_piston: wp.array(dtype=F64),
              phi: wp.array(dtype=F64), phid: wp.array(dtype=F64)):
    """RK4 step of the reduced 1-DOF dynamics. One thread per world.

    There is no constraint here to make stiff — the loop is closed by
    coordinates — so a classic explicit RK4 resolves the mechanism at dt=0.004
    with energy drift ~1e-6 in the leg's operating regime (measured in the
    tests). The earlier (phi, phid) semi-implicit and momentum-symplectic
    variants both accumulated visible drift once transits sampled the
    configuration-dependent inertia coarsely; RK4's 4 evaluations/step remove
    that at trivial cost (~12 _mass_grav calls, all closed-form)."""
    w = wp.tid()
    q = phi[w]
    qd = phid[w]
    t = tau[w]
    f = f_piston[w]
    k1q = qd
    k1v = _accel(q, qd, t, f, r, L, m_c, m_r, m_p, g)
    k2q = qd + F64(0.5) * dt * k1v
    k2v = _accel(q + F64(0.5) * dt * k1q, k2q, t, f, r, L, m_c, m_r, m_p, g)
    k3q = qd + F64(0.5) * dt * k2v
    k3v = _accel(q + F64(0.5) * dt * k2q, k3q, t, f, r, L, m_c, m_r, m_p, g)
    k4q = qd + dt * k3v
    k4v = _accel(q + dt * k3q, k4q, t, f, r, L, m_c, m_r, m_p, g)
    phi[w] = q + dt / F64(6.0) * (k1q + F64(2.0) * k2q + F64(2.0) * k3q + k4q)
    phid[w] = qd + dt / F64(6.0) * (k1v + F64(2.0) * k2v + F64(2.0) * k3v + k4v)


@wp.kernel
def gen_force_eval(phi: wp.array(dtype=F64), r: F64, L: F64,
                   f_piston: wp.array(dtype=F64), out_q: wp.array(dtype=F64)):
    """Generalized crank torque produced by a piston force — THE toggle profile:
    Q = F * ds/dphi (diverging mechanical advantage F/tau = 1/(ds/dphi) at TDC)."""
    i = wp.tid()
    k = crank_kin(phi[i], r, L)
    out_q[i] = f_piston[i] * k[3]


@wp.kernel
def energy_eval(phi: wp.array(dtype=F64), phid: wp.array(dtype=F64),
                r: F64, L: F64, m_c: F64, m_r: F64, m_p: F64, g: F64,
                out_e: wp.array(dtype=F64)):
    """Total mechanical energy of the reduced system (for the TDC stability test)."""
    i = wp.tid()
    p = phi[i]
    mg = _mass_grav(p, r, L, m_c, m_r, m_p, g)
    k = crank_kin(p, r, L)
    sp = wp.sin(p)
    y_c = (r / F64(2.0)) * wp.cos(p)
    heel_y = (r - L) + k[1]
    toe_y = r * wp.cos(p)
    y_r = (toe_y + heel_y) / F64(2.0)
    V = g * (m_c * y_c + m_r * y_r + m_p * heel_y)
    out_e[i] = F64(0.5) * mg[0] * phid[i] * phid[i] + V
    # keep sp referenced (warp requires no unused vars? benign either way)
    if sp > F64(1.0e30):
        out_e[i] = sp
