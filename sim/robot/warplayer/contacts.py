# SPDX-License-Identifier: MIT
"""warplayer.contacts — analytic narrowphase + MuJoCo-style soft contact solve.

Implements secret-sauce §10c(ii): a curated static pair table with analytic
primitive kernels — capsule-plane and capsule-capsule are segment/point distance
problems (~30 lines each), no GJK. Contacts are appended to the flat cross-world
pool with ``wp.atomic_add`` (mujoco_warp convention, §4).

The solve ports MuJoCo's soft-constraint math (§1 of the report; formulas from
the MuJoCo "Solver parameters" docs / engine_core_constraint.c):
  impedance d(x): two-branch power sigmoid over x = |r|/width (solimp),
  aref = -b*rdot - k*r  with b = 2/(dmax*tc), k = d /(dmax^2 tc^2 dr^2),
  regularization R = (1-d)/d * A_ii,
  f_n solved from (A + R) f = aref - a0, f_n >= 0,
performed as fixed-count Jacobi sweeps at the acceleration level: each sweep
re-evaluates contact-point accelerations from the currently applied forces, so
contacts sharing a body converge to the coupled solution without assembling A
(the diagonal-only shortcut visibly underestimates penetration for a two-ended
capsule; the sweeps recover parity). Friction: per-axis clamped tangential
damping (|f_t| <= mu*f_n) — a box approximation of MuJoCo's pyramidal cone,
adequate for the M1 drop/stack scenarios; noted divergence for sliding-heavy
scenes.
"""
from __future__ import annotations

import warp as wp


@wp.func
def _impedance(depth: float, d0: float, dmax: float, width: float,
               mid: float, power: float) -> float:
    x = wp.min(wp.abs(depth) / width, 1.0)
    y = float(0.0)
    if x <= mid:
        y = wp.pow(x / mid, power) * mid
    else:
        y = 1.0 - wp.pow((1.0 - x) / (1.0 - mid), power) * (1.0 - mid)
    return d0 + y * (dmax - d0)


@wp.func
def _inv_inertia_world(q: wp.quat, inertia: wp.vec3) -> wp.mat33:
    R = wp.quat_to_matrix(q)
    di = wp.mat33(1.0 / inertia[0], 0.0, 0.0,
                  0.0, 1.0 / inertia[1], 0.0,
                  0.0, 0.0, 1.0 / inertia[2])
    return R @ di @ wp.transpose(R)


@wp.kernel
def narrowphase(pair_a: wp.array(dtype=wp.int32), pair_b: wp.array(dtype=wp.int32),
                cap_radius: wp.array2d(dtype=wp.float32),
                cap_halflen: wp.array2d(dtype=wp.float32),
                xpos: wp.array2d(dtype=wp.vec3), xquat: wp.array2d(dtype=wp.quat),
                naconmax: int,
                nacon: wp.array(dtype=wp.int32),
                con_world: wp.array(dtype=wp.int32), con_a: wp.array(dtype=wp.int32),
                con_b: wp.array(dtype=wp.int32), con_pos: wp.array(dtype=wp.vec3),
                con_n: wp.array(dtype=wp.vec3), con_depth: wp.array(dtype=wp.float32)):
    """One thread per (world, pair). Curated static table; atomic compaction."""
    world, pid = wp.tid()
    mw = cap_radius.shape[0]
    wm = world % mw
    a = pair_a[pid]
    b = pair_b[pid]
    ra = cap_radius[wm, a]
    ha = cap_halflen[wm, a]
    qa = xquat[world, a]
    pa = xpos[world, a]
    axa = wp.quat_rotate(qa, wp.vec3(0.0, 0.0, 1.0))
    if b < 0:
        # capsule vs plane z=0: test both endcap centers
        for cap_end in range(2):
            end_sign = 1.0 - 2.0 * float(cap_end)
            cpt = pa + axa * (ha * end_sign)
            depth = ra - cpt[2]
            if depth > 0.0:
                idx = wp.atomic_add(nacon, 0, 1)
                if idx < naconmax:
                    con_world[idx] = world
                    con_a[idx] = a
                    con_b[idx] = -1
                    con_n[idx] = wp.vec3(0.0, 0.0, -1.0)   # A -> B(ground): down
                    con_pos[idx] = wp.vec3(cpt[0], cpt[1], cpt[2] - ra + depth * 0.5)
                    con_depth[idx] = depth
    else:
        rb = cap_radius[wm, b]
        hb = cap_halflen[wm, b]
        qb = xquat[world, b]
        pb = xpos[world, b]
        axb = wp.quat_rotate(qb, wp.vec3(0.0, 0.0, 1.0))
        # closest points between segments (Ericson RTCD 5.1.9)
        d1 = axa * (2.0 * ha)
        d2 = axb * (2.0 * hb)
        p1 = pa - axa * ha
        p2 = pb - axb * hb
        rseg = p1 - p2
        segA = wp.dot(d1, d1)
        segE = wp.dot(d2, d2)
        segF = wp.dot(d2, rseg)
        segC = wp.dot(d1, rseg)
        bb = wp.dot(d1, d2)
        den = segA * segE - bb * bb
        ps = float(0.0)
        if den > 1.0e-9:
            ps = wp.clamp((bb * segF - segC * segE) / den, 0.0, 1.0)
        pt = float(0.0)
        if segE > 1.0e-9:
            pt = wp.clamp((bb * ps + segF) / segE, 0.0, 1.0)
        ps = wp.clamp((bb * pt - segC) / wp.max(segA, 1.0e-9), 0.0, 1.0)
        c1 = p1 + d1 * ps
        c2 = p2 + d2 * pt
        dvec = c2 - c1
        dist = wp.length(dvec)
        depth = ra + rb - dist
        if depth > 0.0 and dist > 1.0e-9:
            idx = wp.atomic_add(nacon, 0, 1)
            if idx < naconmax:
                n = dvec / dist
                con_world[idx] = world
                con_a[idx] = a
                con_b[idx] = b
                con_n[idx] = n
                con_pos[idx] = c1 + n * (ra - depth * 0.5)
                con_depth[idx] = depth


@wp.func
def _point_acc(world: int, body: int, pos: wp.vec3, gravity: float,
               body_mass: wp.array2d(dtype=wp.float32),
               body_inertia: wp.array2d(dtype=wp.vec3),
               xpos: wp.array2d(dtype=wp.vec3), xquat: wp.array2d(dtype=wp.quat),
               angvel: wp.array2d(dtype=wp.vec3),
               frc: wp.array2d(dtype=wp.vec3), trq: wp.array2d(dtype=wp.vec3)) -> wp.vec3:
    """Acceleration of a body-fixed point under gravity + currently-applied
    contact forces (used by the Jacobi sweeps to couple same-body contacts)."""
    mw = body_mass.shape[0]
    wm = world % mw
    m = body_mass[wm, body]
    iw = _inv_inertia_world(xquat[world, body], body_inertia[wm, body])
    om = angvel[world, body]
    rr = pos - xpos[world, body]
    alin = frc[world, body] / m + wp.vec3(0.0, 0.0, gravity)
    Iw_om = wp.inverse(iw) @ om
    alpha = iw @ (trq[world, body] - wp.cross(om, Iw_om))
    return alin + wp.cross(alpha, rr) + wp.cross(om, wp.cross(om, rr))


@wp.func
def _point_vel(world: int, body: int, pos: wp.vec3,
               xpos: wp.array2d(dtype=wp.vec3),
               linvel: wp.array2d(dtype=wp.vec3),
               angvel: wp.array2d(dtype=wp.vec3)) -> wp.vec3:
    rr = pos - xpos[world, body]
    return linvel[world, body] + wp.cross(angvel[world, body], rr)


@wp.func
def _ainv_contrib(world: int, body: int, pos: wp.vec3, n: wp.vec3,
                  body_mass: wp.array2d(dtype=wp.float32),
                  body_inertia: wp.array2d(dtype=wp.vec3),
                  xpos: wp.array2d(dtype=wp.vec3),
                  xquat: wp.array2d(dtype=wp.quat)) -> float:
    """Body's contribution to A_ii = J M^-1 J^T along direction n at pos."""
    mw = body_mass.shape[0]
    wm = world % mw
    rr = pos - xpos[world, body]
    rxn = wp.cross(rr, n)
    iw = _inv_inertia_world(xquat[world, body], body_inertia[wm, body])
    return 1.0 / body_mass[wm, body] + wp.dot(rxn, iw @ rxn)


@wp.kernel
def contact_sweep(gravity: float, timeconst: float, dampratio: float,
                  d0: float, dmax: float, width: float, mid: float, power: float,
                  mu: float,
                  body_mass: wp.array2d(dtype=wp.float32),
                  body_inertia: wp.array2d(dtype=wp.vec3),
                  xpos: wp.array2d(dtype=wp.vec3), xquat: wp.array2d(dtype=wp.quat),
                  linvel: wp.array2d(dtype=wp.vec3), angvel: wp.array2d(dtype=wp.vec3),
                  nacon: wp.array(dtype=wp.int32),
                  con_world: wp.array(dtype=wp.int32), con_a: wp.array(dtype=wp.int32),
                  con_b: wp.array(dtype=wp.int32), con_pos: wp.array(dtype=wp.vec3),
                  con_n: wp.array(dtype=wp.vec3), con_depth: wp.array(dtype=wp.float32),
                  con_f: wp.array(dtype=wp.vec3),
                  frc: wp.array2d(dtype=wp.vec3), trq: wp.array2d(dtype=wp.vec3)):
    """One Jacobi sweep: one thread per contact; graph-safe fixed launch dim
    with early exit past the live count (mujoco_warp pattern, §4)."""
    cid = wp.tid()
    if cid >= nacon[0]:
        return
    world = con_world[cid]
    a = con_a[cid]
    b = con_b[cid]
    n = con_n[cid]                       # A -> B
    pos = con_pos[cid]
    depth = con_depth[cid]
    # current relative accel/vel of B w.r.t. A along n (B may be the static plane)
    acc_a = _point_acc(world, a, pos, gravity, body_mass, body_inertia,
                       xpos, xquat, angvel, frc, trq)
    vel_a = _point_vel(world, a, pos, xpos, linvel, angvel)
    acc_b = wp.vec3(0.0, 0.0, 0.0)
    vel_b = wp.vec3(0.0, 0.0, 0.0)
    if b >= 0:
        acc_b = _point_acc(world, b, pos, gravity, body_mass, body_inertia,
                           xpos, xquat, angvel, frc, trq)
        vel_b = _point_vel(world, b, pos, xpos, linvel, angvel)
    # constraint r = -depth (negative when penetrating); rdot = separating speed
    rel_v = vel_b - vel_a
    rel_a = acc_b - acc_a
    rdot = wp.dot(n, rel_v)
    r = -depth
    d = _impedance(depth, d0, dmax, width, mid, power)
    bcoef = 2.0 / (dmax * timeconst)
    kcoef = d / (dmax * dmax * timeconst * timeconst * dampratio * dampratio)
    aref = -bcoef * rdot - kcoef * r
    Aii = _ainv_contrib(world, a, pos, n, body_mass, body_inertia, xpos, xquat)
    mw2 = body_mass.shape[0]
    A0 = 1.0 / body_mass[world % mw2, a]        # diagApprox-style: translation-only
    if b >= 0:
        Aii += _ainv_contrib(world, b, pos, n, body_mass, body_inertia, xpos, xquat)
        A0 += 1.0 / body_mass[world % mw2, b]
    # MuJoCo regularizes with the qpos0 DIAGONAL APPROXIMATION, not the true
    # contact A (engine_core_constraint diagApprox; secret-sauce §8 flags exactly
    # this mismatch near singular configs). Using true-A here made contacts ~3.6x
    # softer than the reference; A0 restores rest-penetration parity.
    Rreg = A0 * (1.0 - d) / d
    a0 = wp.dot(n, rel_a)
    f_old = con_f[cid]
    fn_old = wp.dot(n, f_old)
    # regularized optimality (MuJoCo dual): a = aref - R*f  =>  residual includes -R*f_old
    # (dropping it converges to pen=0 instead of MuJoCo's pen = R*f/k rest penetration)
    dfn = (aref - a0 - Rreg * fn_old) / (Aii + Rreg)
    fn = wp.max(fn_old + dfn, 0.0)
    # friction: clamped tangential damping toward zero slip
    t1 = wp.normalize(wp.cross(n, wp.vec3(0.57735, 0.57735, 0.57735) + n * 0.1))
    t2 = wp.cross(n, t1)
    ft1_old = wp.dot(t1, f_old)
    ft2_old = wp.dot(t2, f_old)
    a01 = wp.dot(t1, rel_a)
    a02 = wp.dot(t2, rel_a)
    v1 = wp.dot(t1, rel_v)
    v2 = wp.dot(t2, rel_v)
    A1 = _ainv_contrib(world, a, pos, t1, body_mass, body_inertia, xpos, xquat)
    A2 = _ainv_contrib(world, a, pos, t2, body_mass, body_inertia, xpos, xquat)
    if b >= 0:
        A1 += _ainv_contrib(world, b, pos, t1, body_mass, body_inertia, xpos, xquat)
        A2 += _ainv_contrib(world, b, pos, t2, body_mass, body_inertia, xpos, xquat)
    R1 = A0 * (1.0 - d) / d
    R2 = A0 * (1.0 - d) / d
    ft1 = ft1_old + (-bcoef * v1 - a01 - R1 * ft1_old) / (A1 + R1)
    ft2 = ft2_old + (-bcoef * v2 - a02 - R2 * ft2_old) / (A2 + R2)
    lim = mu * fn
    ft1 = wp.clamp(ft1, -lim, lim)
    ft2 = wp.clamp(ft2, -lim, lim)
    f_new = n * fn + t1 * ft1 + t2 * ft2
    df = f_new - f_old
    con_f[cid] = f_new
    # apply the force DELTA to the bodies (force on B at pos; reaction on A)
    if b >= 0:
        wp.atomic_add(frc, world, b, df)
        wp.atomic_add(trq, world, b, wp.cross(pos - xpos[world, b], df))
    wp.atomic_add(frc, world, a, -df)
    wp.atomic_add(trq, world, a, wp.cross(pos - xpos[world, a], -df))
