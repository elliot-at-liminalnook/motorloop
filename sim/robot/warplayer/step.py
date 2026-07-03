# SPDX-License-Identifier: MIT
"""warplayer.step — the M1 stepping loop (secret-sauce §10c).

Semi-implicit Euler (velocity then position — the same scheme as MuJoCo's
``integrator="Euler"`` for these free bodies, which have no joint damping):
zero accumulators -> narrowphase (atomic pool) -> K Jacobi contact sweeps ->
integrate. Every launch has fixed dims and no host branches, so the whole step
is CUDA-graph-capturable (the §3 primitive we reuse rather than rebuild);
`wp.capture_begin/end` works on CUDA devices — on CPU the same call sequence
just runs eagerly.
"""
from __future__ import annotations

import warp as wp

from .contacts import contact_sweep, narrowphase
from .types import Data, Model


@wp.kernel
def _zero_step(frc: wp.array2d(dtype=wp.vec3), trq: wp.array2d(dtype=wp.vec3),
               nacon: wp.array(dtype=wp.int32), con_f: wp.array(dtype=wp.vec3)):
    world, body = wp.tid()
    frc[world, body] = wp.vec3(0.0, 0.0, 0.0)
    trq[world, body] = wp.vec3(0.0, 0.0, 0.0)
    if world == 0 and body == 0:
        nacon[0] = 0
    # contact forces persist only within a step's sweeps; clear the pool slots
    # this thread owns (strided) so warm-start artifacts can't leak across steps
    i = world * frc.shape[1] + body
    n = frc.shape[0] * frc.shape[1]
    j = i
    while j < con_f.shape[0]:
        con_f[j] = wp.vec3(0.0, 0.0, 0.0)
        j += n


@wp.kernel
def _integrate(dt: float, gravity: float,
               body_mass: wp.array2d(dtype=wp.float32),
               body_inertia: wp.array2d(dtype=wp.vec3),
               frc: wp.array2d(dtype=wp.vec3), trq: wp.array2d(dtype=wp.vec3),
               xpos: wp.array2d(dtype=wp.vec3), xquat: wp.array2d(dtype=wp.quat),
               linvel: wp.array2d(dtype=wp.vec3), angvel: wp.array2d(dtype=wp.vec3)):
    world, body = wp.tid()
    mw = body_mass.shape[0]
    wm = world % mw
    m = body_mass[wm, body]
    q = xquat[world, body]
    R = wp.quat_to_matrix(q)
    Ib = body_inertia[wm, body]
    Iw = R @ wp.mat33(Ib[0], 0.0, 0.0, 0.0, Ib[1], 0.0, 0.0, 0.0, Ib[2]) @ wp.transpose(R)
    om = angvel[world, body]
    v = linvel[world, body] + dt * (frc[world, body] / m + wp.vec3(0.0, 0.0, gravity))
    alpha = wp.inverse(Iw) @ (trq[world, body] - wp.cross(om, Iw @ om))
    om = om + dt * alpha
    linvel[world, body] = v
    angvel[world, body] = om
    xpos[world, body] = xpos[world, body] + dt * v
    wnorm = wp.length(om)
    if wnorm > 1.0e-12:
        dq = wp.quat_from_axis_angle(om / wnorm, wnorm * dt)
        xquat[world, body] = wp.normalize(wp.mul(dq, q))


def step(m: Model, d: Data):
    """Advance every world by one dt. Launch sequence is graph-capturable."""
    wp.launch(_zero_step, dim=(m.nworld, m.nbody),
              inputs=[d.frc, d.trq, d.nacon, d.con_f])
    wp.launch(narrowphase, dim=(m.nworld, m.npair),
              inputs=[m.pair_a, m.pair_b, m.cap_radius, m.cap_halflen,
                      d.xpos, d.xquat, m.naconmax, d.nacon, d.con_world,
                      d.con_a, d.con_b, d.con_pos, d.con_n, d.con_depth])
    for _ in range(m.solver_iters):
        wp.launch(contact_sweep, dim=m.naconmax,
                  inputs=[m.gravity, m.timeconst, m.dampratio,
                          m.imp_d0, m.imp_dmax, m.imp_width, m.imp_mid, m.imp_power,
                          m.friction,
                          m.body_mass, m.body_inertia, d.xpos, d.xquat,
                          d.linvel, d.angvel, d.nacon, d.con_world, d.con_a,
                          d.con_b, d.con_pos, d.con_n, d.con_depth, d.con_f,
                          d.frc, d.trq])
    wp.launch(_integrate, dim=(m.nworld, m.nbody),
              inputs=[m.dt, m.gravity, m.body_mass, m.body_inertia,
                      d.frc, d.trq, d.xpos, d.xquat, d.linvel, d.angvel])
