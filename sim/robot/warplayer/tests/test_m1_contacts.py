# SPDX-License-Identifier: MIT
"""M1 gate (secret-sauce §10c): contact parity vs the MuJoCo C reference.

Kill criterion: rest-pose contact behavior > 1% off MuJoCo at matched
solref/solimp/friction/dt (scale = capsule radius). Scenarios: a horizontal
capsule dropped onto the plane (two endcap contacts — exercises the same-body
coupling the Jacobi sweeps exist for), and a two-capsule cross stack
(capsule-capsule + capsule-plane). nworld=4 with different drop heights proves
the (nworld, ...) layout carries independent worlds.
"""
from __future__ import annotations

import sys
from pathlib import Path

import mujoco
import numpy as np
import pytest
import warp as wp

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parents[1]))
from warplayer import make_capsules_scene, step  # noqa: E402

wp.init()

R, HL, MASS = 0.05, 0.15, 0.4
SOLREF = (0.02, 1.0)
SOLIMP = (0.9, 0.95, 0.001, 0.5, 2.0)
DT, SECONDS = 0.002, 1.5
QY90 = (0.0, np.sin(np.pi / 4), 0.0, np.cos(np.pi / 4))   # warp xyzw: z-axis -> x


def _mj_ref(caps, seconds=SECONDS):
    geoms = ""
    for i, c in enumerate(caps):
        # mujoco quat wxyz from warp xyzw
        q = c.get("quat", (0, 0, 0, 1))
        geoms += f"""
        <body name="b{i}" pos="{c['pos'][0]} {c['pos'][1]} {c['pos'][2]}"
              quat="{q[3]} {q[0]} {q[1]} {q[2]}">
          <freejoint/>
          <geom type="capsule" size="{c['radius']} {c['halflen']}" mass="{c['mass']}"
                solref="{SOLREF[0]} {SOLREF[1]}"
                solimp="{SOLIMP[0]} {SOLIMP[1]} {SOLIMP[2]} {SOLIMP[3]} {SOLIMP[4]}"
                friction="1 0.005 0.0001" condim="3"/>
        </body>"""
    xml = f"""<mujoco><option integrator="Euler" timestep="{DT}" gravity="0 0 -9.81"/>
      <worldbody><geom name="floor" type="plane" size="0 0 1"
        solref="{SOLREF[0]} {SOLREF[1]}"
        solimp="{SOLIMP[0]} {SOLIMP[1]} {SOLIMP[2]} {SOLIMP[3]} {SOLIMP[4]}"
        friction="1 0.005 0.0001"/>{geoms}</worldbody></mujoco>"""
    m = mujoco.MjModel.from_xml_string(xml)
    d = mujoco.MjData(m)
    for _ in range(int(seconds / DT)):
        mujoco.mj_step(m, d)
    return np.array([d.qpos[7 * i:7 * i + 3] for i in range(len(caps))])


def _warp_run(caps, pairs, nworld=1, z_offsets=None, seconds=SECONDS):
    m, d = make_capsules_scene(nworld, caps, pairs, dt=DT,
                               solref=SOLREF, solimp=SOLIMP, friction=1.0)
    if z_offsets is not None:
        xp = d.xpos.numpy()
        for w, dz in enumerate(z_offsets):
            xp[w, :, 2] += dz
        d.xpos = wp.array(xp, dtype=wp.vec3)
    for _ in range(int(seconds / DT)):
        step(m, d)
    return d.xpos.numpy()


def test_flat_drop_rest_parity():
    caps = [dict(mass=MASS, radius=R, halflen=HL, pos=(0.0, 0.0, 0.08), quat=QY90)]
    zw = _warp_run(caps, pairs=[(0, -1)])[0, 0, 2]
    zm = _mj_ref(caps)[0, 2]
    err = abs(zw - zm)
    assert err <= 0.01 * R, f"rest z: warp {zw:.6f} vs mujoco {zm:.6f} (|Δ|={err * 1e3:.3f} mm > 1% of r)"
    # penetration depths comparable (both sub-mm, same order)
    pen_w, pen_m = R - zw, R - zm
    assert pen_w > 0 and pen_m > 0
    assert abs(pen_w - pen_m) <= 0.35 * max(pen_m, 1e-5)


def test_cross_stack_parity():
    qx90 = (np.sin(np.pi / 4), 0.0, 0.0, np.cos(np.pi / 4))   # z-axis -> -y
    caps = [dict(mass=MASS, radius=R, halflen=HL, pos=(0.0, 0.0, 0.06), quat=QY90),
            dict(mass=MASS, radius=R, halflen=HL, pos=(0.0, 0.0, 0.17), quat=qx90)]
    got = _warp_run(caps, pairs=[(0, -1), (1, -1), (0, 1)])[0]
    ref = _mj_ref(caps)
    for i in range(2):
        err = abs(got[i, 2] - ref[i, 2])
        assert err <= 0.01 * R, (
            f"body {i} rest z: warp {got[i, 2]:.6f} vs mujoco {ref[i, 2]:.6f} "
            f"(|Δ|={err * 1e3:.3f} mm)")


def test_nworld_independent_ics():
    caps = [dict(mass=MASS, radius=R, halflen=HL, pos=(0.0, 0.0, 0.08), quat=QY90)]
    offs = [0.0, 0.01, 0.02, 0.03]
    got = _warp_run(caps, pairs=[(0, -1)], nworld=4, z_offsets=offs)
    for w, dz in enumerate(offs):
        ref = _mj_ref([dict(caps[0], pos=(0.0, 0.0, 0.08 + dz))])
        err = abs(got[w, 0, 2] - ref[0, 2])
        assert err <= 0.01 * R, f"world {w}: |Δ|={err * 1e3:.3f} mm"
    assert np.all(np.isfinite(got))


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-q"]))
