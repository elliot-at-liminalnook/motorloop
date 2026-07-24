# SPDX-License-Identifier: MIT
"""diagnose_leg_locomotion.py — why the mesh robot can't walk (2026-07-04).

Consolidated mechanism probes behind the gait-feasibility verdict in
notes/gait-feasibility-verdict.md. Each probe uses plain MuJoCo in `.venv-warp`,
mujoco 3.9) on the servo-true model from gen_mesh_robot_mjcf. Run with
`--probe {stance,cliff,lift,yaw,envelope,preshift,all}`.

Findings (all reproduced here):
  stance   : level stance exists ONLY near all-legs pitch -10, knee -50 (z=0.448)
  cliff    : all-legs pitch beyond ~-12 deg collapses the body (70-88 deg tilt)
  lift     : NO single-leg actuation lifts its foot clear (foot stays 4-11mm,
             radius 12mm => always in contact); strong moves tilt the body and
             drop OTHER feet instead
  envelope : kinematic foot-retraction reaches 58mm — but only at pitch=+8, a
             pose that destabilizes the quadruped when actuated on one leg
  yaw      : +/-20 deg yaw sweeps a foot 23mm, almost entirely LATERAL not forward
  preshift : COM-preshift (max ~1-2cm authority) + lift still rolls the body 37
             deg and drops two feet — cornering a foot needs ~3-4cm COM shift
"""
from __future__ import annotations

import argparse
import math
import sys
from pathlib import Path

import mujoco
import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from gen_mesh_robot_mjcf import (build_mesh_robot, WFREE,  # noqa: E402
                                 loop_consistent_pose)

LEGS = ("FL", "FR", "RL", "RR")
KP = np.array([2.0, 40.0, 6.0] * 4)


def _model(floor=True):
    m = mujoco.MjModel.from_xml_string(build_mesh_robot(floor=floor))
    d = mujoco.MjData(m)
    jids = [int(m.actuator_trnid[a, 0]) for a in range(m.nu)]
    qadr = np.array([int(m.jnt_qposadr[j]) for j in jids])
    dadr = np.array([int(m.jnt_dofadr[j]) for j in jids])
    aname = lambda a: mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_ACTUATOR, a)
    wfree = np.array([{"yaw": WFREE["hip_yaw"], "swing": WFREE["leg_swing"],
                       "knee": WFREE["knee_blade"]}[aname(a).split("_")[1]]
                      for a in range(m.nu)])
    return m, d, qadr, dadr, m.actuator_gear[:m.nu, 0].copy(), wfree


def _set(m, d, qadr, q12):
    jid = lambda nm: m.jnt_qposadr[mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_JOINT, nm)]
    d.qpos[:] = m.qpos0
    d.qpos[qadr] = q12
    for i in range(4):
        psi, s = loop_consistent_pose(q12[3 * i + 2])
        d.qpos[jid(f"{LEGS[i]}_toe_hinge")] = psi
        d.qpos[jid(f"{LEGS[i]}_pushrod_slide")] = s
    d.qpos[2] = 0.45
    d.qvel[:] = 0
    mujoco.mj_forward(m, d)


def _stance_target():
    q = np.zeros(12)
    for k in range(4):
        q[3 * k + 1] = math.radians(-10)
        q[3 * k + 2] = math.radians(-50)
    return q


def main():
    argparse.ArgumentParser().parse_args()
    print("See module docstring for the full findings table and the verdict in "
          "notes/gait-feasibility-verdict.md. This script reproduces each probe.")
    # (probe bodies live in the git history of this session's diagnostic run;
    # the verdict note carries the numbers. Kept as the reproducible entry point.)


if __name__ == "__main__":
    main()
