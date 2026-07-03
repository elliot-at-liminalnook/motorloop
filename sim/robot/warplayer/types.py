# SPDX-License-Identifier: MIT
"""warplayer.types — Model/Data containers (secret-sauce §10c data-layout mandate).

Layout follows mujoco_warp conventions (§4): Data arrays carry an explicit
``nworld`` leading dimension; Model arrays are broadcastable — kernels index
``worldid % model_leading_dim`` so per-world domain randomization needs no
recompile; contacts live in ONE flat cross-world pool, compacted with
``wp.atomic_add`` and tagged with a per-contact worldid.

Bodies in M1 are free capsules over a single ground plane at z=0; the pair
table is CURATED AND STATIC (§10c(ii)): at this project's geom counts,
hand-culled enumeration is optimal — MJX's sin was uncurated all-pairs plus
dense efc rows, not staticness itself.
"""
from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np
import warp as wp


def capsule_inertia(mass: float, r: float, hl: float) -> tuple[float, float]:
    """(I_axial, I_perp) of a solid capsule about its COM (MuJoCo's formula:
    cylinder + two hemispheres, same as user_objects.cc)."""
    L = 2.0 * hl
    v_cyl = math.pi * r * r * L
    v_sph = 4.0 / 3.0 * math.pi * r ** 3
    rho = mass / (v_cyl + v_sph)
    m_c, m_s = rho * v_cyl, rho * v_sph
    i_ax = m_c * r * r / 2.0 + m_s * 2.0 * r * r / 5.0
    i_perp = (m_c * (L * L / 12.0 + r * r / 4.0)
              + m_s * (2.0 * r * r / 5.0 + L * L / 4.0 + 3.0 * L * r / 8.0))
    return i_ax, i_perp


@dataclass
class Model:
    nworld: int
    nbody: int
    npair: int
    naconmax: int
    # broadcastable (leading dim 1 or nworld)
    body_mass: wp.array2d      # (mw, nbody) float32
    body_inertia: wp.array2d   # (mw, nbody) vec3  (body-frame diagonal, capsule: [ax, perp, perp])
    cap_radius: wp.array2d     # (mw, nbody) float32
    cap_halflen: wp.array2d    # (mw, nbody) float32
    # curated static pair table: pair -> (bodyA, bodyB); bodyB == -1 => ground plane
    pair_a: wp.array           # (npair,) int32
    pair_b: wp.array           # (npair,) int32
    # scalar params (host-side constants baked at launch)
    gravity: float = -9.81
    dt: float = 0.002
    # MuJoCo solref/solimp (shared by all contacts in M1)
    timeconst: float = 0.02
    dampratio: float = 1.0
    imp_d0: float = 0.9
    imp_dmax: float = 0.95
    imp_width: float = 0.001
    imp_mid: float = 0.5
    imp_power: float = 2.0
    friction: float = 1.0
    solver_iters: int = 8


@dataclass
class Data:
    # state, (nworld, nbody)
    xpos: wp.array2d           # vec3
    xquat: wp.array2d          # quat
    linvel: wp.array2d         # vec3
    angvel: wp.array2d         # vec3 (world frame)
    # per-step accumulators, (nworld, nbody)
    frc: wp.array2d            # vec3
    trq: wp.array2d            # vec3
    # cross-world atomic contact pool, flat (naconmax,)
    nacon: wp.array            # (1,) int32
    con_world: wp.array        # int32
    con_a: wp.array            # int32
    con_b: wp.array            # int32 (-1 = plane)
    con_pos: wp.array          # vec3
    con_n: wp.array            # vec3 (A -> B)
    con_depth: wp.array        # float32
    con_f: wp.array            # vec3 (solved force on B at pos; -f on A) — persists across solver iters


def _z2(nw, nb, dtype):
    return wp.zeros((nw, nb), dtype=dtype)


def make_capsules_scene(nworld: int, caps: list[dict], pairs: list[tuple[int, int]],
                        dt=0.002, solref=(0.02, 1.0),
                        solimp=(0.9, 0.95, 0.001, 0.5, 2.0), friction=1.0,
                        naconmax_per_world=8) -> tuple[Model, Data]:
    """Build Model+Data for capsules over a plane. caps: [{mass, radius, halflen,
    pos, quat, ...}]; identical across worlds (per-world ICs set on Data after)."""
    nb = len(caps)
    mass = np.array([[c["mass"] for c in caps]], dtype=np.float32)
    inr = np.zeros((1, nb, 3), dtype=np.float32)
    rad = np.array([[c["radius"] for c in caps]], dtype=np.float32)
    hln = np.array([[c["halflen"] for c in caps]], dtype=np.float32)
    for i, c in enumerate(caps):
        ax, pp = capsule_inertia(c["mass"], c["radius"], c["halflen"])
        inr[0, i] = (pp, pp, ax)          # capsule axis = local z (MuJoCo convention)
    m = Model(
        nworld=nworld, nbody=nb, npair=len(pairs),
        naconmax=naconmax_per_world * nworld,
        body_mass=wp.array(mass, dtype=wp.float32),
        body_inertia=wp.array(inr, dtype=wp.vec3),
        cap_radius=wp.array(rad, dtype=wp.float32),
        cap_halflen=wp.array(hln, dtype=wp.float32),
        pair_a=wp.array(np.array([p[0] for p in pairs], dtype=np.int32), dtype=wp.int32),
        pair_b=wp.array(np.array([p[1] for p in pairs], dtype=np.int32), dtype=wp.int32),
        dt=dt, timeconst=solref[0], dampratio=solref[1],
        imp_d0=solimp[0], imp_dmax=solimp[1], imp_width=solimp[2],
        imp_mid=solimp[3], imp_power=solimp[4], friction=friction)
    xpos = np.zeros((nworld, nb, 3), dtype=np.float32)
    xquat = np.zeros((nworld, nb, 4), dtype=np.float32)
    for i, c in enumerate(caps):
        xpos[:, i] = c["pos"]
        xquat[:, i] = c.get("quat", (0, 0, 0, 1))   # warp quat = (x,y,z,w)
    d = Data(
        xpos=wp.array(xpos, dtype=wp.vec3), xquat=wp.array(xquat, dtype=wp.quat),
        linvel=_z2(nworld, nb, wp.vec3), angvel=_z2(nworld, nb, wp.vec3),
        frc=_z2(nworld, nb, wp.vec3), trq=_z2(nworld, nb, wp.vec3),
        nacon=wp.zeros(1, dtype=wp.int32),
        con_world=wp.zeros(m.naconmax, dtype=wp.int32),
        con_a=wp.zeros(m.naconmax, dtype=wp.int32),
        con_b=wp.zeros(m.naconmax, dtype=wp.int32),
        con_pos=wp.zeros(m.naconmax, dtype=wp.vec3),
        con_n=wp.zeros(m.naconmax, dtype=wp.vec3),
        con_depth=wp.zeros(m.naconmax, dtype=wp.float32),
        con_f=wp.zeros(m.naconmax, dtype=wp.vec3))
    return m, d
