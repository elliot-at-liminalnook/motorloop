# SPDX-License-Identifier: MIT
"""Local CPU diagnostic: does the fighter body topple ON ITS OWN (no action)?

The live self-play run shows `alive` pinned at ~80/200 across every reward lever
(upright 2.15->2.705, fire 0.26->1.14, ratio 4.76->1.3). Question: is the fall
STRUCTURAL (body falls passively) or BEHAVIORAL (only when the policy drives it)?

Roll out the single-robot body with ZERO action under the training decimation
(frame_skip=5), striker ON vs OFF, and report when the torso first drops below the
done threshold (z<0.18). If a do-nothing body falls at ~80, it's structural.
"""
import sys
from pathlib import Path
import numpy as np
import mujoco

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from gen_robot_mjcf import build_mjcf, load_spec  # noqa: E402

SPEC = load_spec(HERE / "robot.toml")
FRAME_SKIP = 5
STEPS = 200
FALL_Z = 0.18


def rollout(striker: bool, hold_default=True, perturb=0.0, seed=0):
    xml = build_mjcf(SPEC, self_collision=True, striker=striker)
    m = mujoco.MjModel.from_xml_string(xml)
    d = mujoco.MjData(m)
    tid = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, "torso")
    # let it settle to rest pose, then read the standing height
    mujoco.mj_forward(m, d)
    rng = np.random.default_rng(seed)
    # neutral ctrl = 0 (position/torque actuators centered); optional small perturbation
    zs = []
    fell_at = None
    for t in range(STEPS):
        if perturb:
            d.ctrl[:] = rng.normal(0, perturb, size=m.nu)
        else:
            d.ctrl[:] = 0.0
        for _ in range(FRAME_SKIP):
            mujoco.mj_step(m, d)
        z = float(d.xpos[tid][2])
        zs.append(z)
        if fell_at is None and z < FALL_Z:
            fell_at = t
    zs = np.array(zs)
    return dict(striker=striker, nu=m.nu, nq=m.nq, mass=float(m.body_subtreemass[tid]),
                z0=zs[0], zmin=float(zs.min()), zmax=float(zs.max()), zend=float(zs[-1]),
                fell_at=fell_at)


def main():
    print(f"{'cfg':<22} {'nu':>3} {'mass(kg)':>8} {'z_start':>7} {'z_min':>6} {'z_end':>6} {'fell@':>6}")
    for striker in (False, True):
        for perturb in (0.0, 0.05):
            r = rollout(striker, perturb=perturb)
            tag = f"striker={'ON ' if striker else 'OFF'} perturb={perturb}"
            print(f"{tag:<22} {r['nu']:>3} {r['mass']:>8.3f} {r['z0']:>7.3f} "
                  f"{r['zmin']:>6.3f} {r['zend']:>6.3f} {str(r['fell_at']):>6}")
    print(f"\nfall threshold z<{FALL_Z}; STEPS={STEPS} (frame_skip={FRAME_SKIP}); "
          "fell@=control-step torso first dropped below threshold (None=stood the whole bout)")


if __name__ == "__main__":
    main()
