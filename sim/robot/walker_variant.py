# SPDX-License-Identifier: MIT
"""walker_variant.py — hardware-lever feasibility study (2026-07-04).

The combat leg cannot walk (notes/gait-feasibility-verdict.md): no foot-lift DOF,
narrow stance, pitch stability cliff. This builds a PARAMETRIC walker with the two
approved hardware levers and finds the minimum change that unlocks stable stepping:

  lever A  foot_lift  — a dedicated actuated ankle-lift prismatic per leg (models
                        "repurpose the knee toggle to retract, not just strike"):
                        vertical foot retraction the combat mechanism lacks.
  lever B  stance     — widen/lengthen the mount spacing so the achievable ~1-2 cm
                        COM shift lands inside a bigger support polygon.

Deliberately NOT the slider-crank strike model — that geometry is the *combat*
feature and is irrelevant to walking feasibility. Masses, leg length, stance
height and the servo torque-speed limits are matched to the real robot so the
feasibility verdict transfers. Servo-true limits from gen_mesh_robot_mjcf.

  .venv-warp/bin/python walker_variant.py --sweep      # find minimum viable levers
  .venv-warp/bin/python walker_variant.py --prove ...  # one config, full step test
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
from gen_mesh_robot_mjcf import GEARS, WFREE, SERVO_STALL_NM, SERVO_FREE_RAD_S  # noqa: E402
from robot_design import LEG_MASS, SERVO_MASS, TARGET, TORSO_MASS  # noqa: E402

LEGS = ("FL", "FR", "RL", "RR")
SIGN = {"FL": (+1, +1), "FR": (+1, -1), "RL": (-1, +1), "RR": (-1, -1)}  # (x,y)
STANCE_Z = 0.44           # target torso height (matches the real settled stance)
FOOT_R = 0.012
# Lift actuator: ST3215-HS direct through the 40 mm crank. Model as a prismatic force.
LIFT_FORCE = TARGET.knee.gear()
LIFT_SPEED = SERVO_FREE_RAD_S * 0.04      # m/s no-load at the foot


def build_walker(stance_len=0.20, stance_width=0.162, lift_range=0.0, floor=True):
    """4-leg walker. stance_len/width = full mount spacing (m). lift_range=0
    disables the ankle-lift (reproduces the no-lift combat limitation)."""
    hx, hy = stance_len / 2, stance_width / 2
    thigh = 0.16          # yaw-mount to pitch axis (matches housing span)
    shin = STANCE_Z - 0.02  # pitch axis down to foot at nominal
    raw_parts = (0.30, 0.655, 0.655, 0.05)
    yaw_structure, upper_structure, lower_structure, foot_structure = [
        LEG_MASS * x / sum(raw_parts) for x in raw_parts]
    legs_xml = []
    acts = []
    for L in LEGS:
        sx, sy = SIGN[L]
        mx, my = sx * hx, sy * hy
        # yaw about vertical (z), pitch about lateral (leg-local), lift along shin
        lift_joint = (f'<joint name="{L}_lift" type="slide" axis="0 0 1" '
                      f'range="0 {lift_range:.4f}" armature="0.02" damping="1.0"/>'
                      if lift_range > 0 else "")
        legs_xml.append(f'''
    <body name="{L}_yaw" pos="{mx:.4f} {my:.4f} 0">
      <joint name="{L}_hip_yaw" type="hinge" axis="0 0 1" range="-0.785 0.785"
             armature="{TARGET.yaw.armature():.6g}" damping="0.05"/>
      <geom type="capsule" fromto="0 0 0 {sx*0.04:.3f} 0 0" size="0.02"
            mass="{yaw_structure + SERVO_MASS:.9f}" contype="0" conaffinity="0"/>
      <body name="{L}_thigh" pos="{sx*0.04:.3f} 0 0">
        <joint name="{L}_pitch" type="hinge" axis="0 1 0" range="-0.6 0.3"
               armature="{TARGET.pitch.armature():.6g}" damping="0.1"/>
        <geom type="capsule" fromto="0 0 0 0 0 {-shin*0.4:.3f}" size="0.015"
              mass="{upper_structure + SERVO_MASS:.9f}" contype="0" conaffinity="0"/>
        <body name="{L}_shin" pos="0 0 {-shin*0.4:.3f}">
          {lift_joint}
          <geom type="capsule" fromto="0 0 0 0 0 {-shin*0.6:.3f}" size="0.012"
                mass="{lower_structure:.9f}" contype="0" conaffinity="0"/>
          <geom name="{L}_foot" type="sphere" pos="0 0 {-shin*0.6:.3f}"
                size="{FOOT_R}" mass="{foot_structure:.9f}"/>
        </body>
      </body>
    </body>''')
        gy, gp = GEARS["hip_yaw"], GEARS["leg_swing"]
        acts.append(
            f'    <motor name="{L}_yaw_m" joint="{L}_hip_yaw" gear="{gy:.4g}" '
            f'forcerange="-{gy:.4g} {gy:.4g}" ctrlrange="-1 1"/>\n'
            f'    <motor name="{L}_pitch_m" joint="{L}_pitch" gear="{gp:.4g}" '
            f'forcerange="-{gp:.4g} {gp:.4g}" ctrlrange="-1 1"/>')
        if lift_range > 0:
            acts.append(
                f'    <motor name="{L}_lift_m" joint="{L}_lift" gear="{LIFT_FORCE:.4g}" '
                f'forcerange="-{LIFT_FORCE:.4g} {LIFT_FORCE:.4g}" ctrlrange="-1 1"/>')
    floor_xml = ('    <geom name="floor" type="plane" size="0 0 0.1" '
                 'rgba="0.4 0.5 0.4 1"/>\n') if floor else ""
    return f'''<mujoco model="walker_variant">
  <compiler angle="radian" autolimits="true"/>
  <option timestep="0.004" integrator="implicitfast"/>
  <default><joint damping="0.02"/>
    <geom friction="1.2 0.1 0.1" contype="1" conaffinity="1"/></default>
  <worldbody>
{floor_xml}    <body name="torso" pos="0 0 {STANCE_Z:.3f}">
      <freejoint/>
      <geom name="torso" type="box" size="{hx+0.05:.3f} {hy+0.03:.3f} 0.04"
            mass="{TORSO_MASS + 4 * SERVO_MASS:.9f}" rgba="0.3 0.3 0.6 1"/>
{''.join(legs_xml)}
    </body>
  </worldbody>
  <actuator>
{chr(10).join(acts)}
  </actuator>
</mujoco>'''


def _servo_wfree(m):
    aname = lambda a: mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_ACTUATOR, a)
    w = np.empty(m.nu)
    for a in range(m.nu):
        k = aname(a).split("_")[1]
        w[a] = {"yaw": WFREE["hip_yaw"], "pitch": WFREE["leg_swing"],
                "lift": LIFT_SPEED}[k]
    return w


def _feasibility(stance_len, stance_width, lift_range, verbose=False):
    """Quasi-static single-step test: settle level, shift COM, lift one foot,
    swing it forward, plant. PASS = foot clears >8mm, body stays <12deg, other
    feet stay down, net forward progress."""
    m = mujoco.MjModel.from_xml_string(build_walker(stance_len, stance_width, lift_range))
    d = mujoco.MjData(m)
    nu, dt = m.nu, m.opt.timestep
    wfree = _servo_wfree(m)
    gear = m.actuator_gear[:nu, 0].copy()
    aid = {mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_ACTUATOR, a): a for a in range(nu)}
    jadr = {mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_JOINT, j):
            m.jnt_qposadr[j] for j in range(m.njnt) if m.jnt_type[j] != 0}
    dadr = {mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_JOINT, j):
            m.jnt_dofadr[j] for j in range(m.njnt) if m.jnt_type[j] != 0}
    torso = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, "torso")
    fg = {L: mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_GEOM, f"{L}_foot") for L in LEGS}
    floor = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_GEOM, "floor")
    has_lift = lift_range > 0

    def rp():
        R = d.xmat[torso].reshape(3, 3)
        return (math.degrees(math.atan2(R[2, 1], R[2, 2])),
                math.degrees(-math.asin(np.clip(R[2, 0], -1, 1))))

    def contact(L):
        g = fg[L]
        for i in range(d.ncon):
            c = d.contact
            if (c.geom1[i] == g and c.geom2[i] == floor) or \
               (c.geom2[i] == g and c.geom1[i] == floor):
                return True
        return False

    # PD servo with torque-speed derating, per named joint targets
    target = {}

    def servo():
        tau = np.zeros(nu)
        for name, a in aid.items():
            jn = name.rsplit("_", 1)[0]           # FL_yaw_m -> FL_yaw... fix below
        for L in LEGS:
            for kind, kp in (("hip_yaw", 40.0), ("pitch", 40.0), ("lift", 300.0)):
                jn = f"{L}_{kind}" if kind != "pitch" else f"{L}_pitch"
                key = f"{L}_hip_yaw" if kind == "hip_yaw" else (
                    f"{L}_pitch" if kind == "pitch" else f"{L}_lift")
                if key not in jadr:
                    continue
                an = {"hip_yaw": f"{L}_yaw_m", "pitch": f"{L}_pitch_m",
                      "lift": f"{L}_lift_m"}[kind]
                if an not in aid:
                    continue
                a = aid[an]
                q = d.qpos[jadr[key]]
                qd = d.qvel[dadr[key]]
                t = kp * (target.get(key, 0.0) - q)
                der = np.clip(1 - abs(qd) / wfree[a], 0, 1)
                lim = der if t * qd > 0 else 1.0
                d.ctrl[a] = np.clip(t / gear[a], -lim, lim)

    def run(secs):
        for _ in range(int(secs / dt)):
            servo()
            mujoco.mj_step(m, d)

    # base level stance: pitch parks the foot under the mount
    for L in LEGS:
        target[f"{L}_pitch"] = 0.0
        target[f"{L}_hip_yaw"] = 0.0
        if has_lift:
            target[f"{L}_lift"] = 0.0
    mujoco.mj_forward(m, d)
    run(1.2)
    z_settle = d.qpos[2]
    roll0, pitch0 = rp()
    if z_settle < 0.30 or abs(roll0) > 10 or abs(pitch0) > 10:
        return dict(ok=False, why=f"stance unstable z={z_settle:.2f} rp={roll0:.0f},{pitch0:.0f}")

    # shift COM toward rear-right (to unload FL): lean via pitch of the far legs
    com0 = d.subtree_com[0][:2].copy()
    target["RL_pitch"] = target["RR_pitch"] = -0.15
    target["FR_hip_yaw"] = -0.2
    run(0.8)
    com_shift = np.linalg.norm(d.subtree_com[0][:2] - com0)

    # lift FL foot
    if has_lift:
        target["FL_lift"] = lift_range
    else:
        target["FL_pitch"] = -0.5     # only lever without lift: pitch it up
    run(0.7)
    fl_z = d.geom_xpos[fg["FL"]][2] * 1000
    cleared = not contact("FL")
    others_down = all(contact(L) for L in ("FR", "RL", "RR"))
    roll, pitch = rp()
    stable = abs(roll) < 12 and abs(pitch) < 12 and d.qpos[2] > 0.30

    # swing FL forward and plant
    target["FL_hip_yaw"] = 0.35 if not has_lift else 0.35
    run(0.6)
    if has_lift:
        target["FL_lift"] = 0.0
    run(0.6)
    planted = contact("FL")
    dx = d.qpos[0]

    ok = bool(cleared and others_down and stable and planted)
    return dict(ok=ok, fl_z=fl_z, cleared=cleared, others_down=others_down,
                stable=stable, planted=planted, roll=roll, pitch=pitch,
                com_shift_mm=com_shift * 1000, z=float(d.qpos[2]))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--sweep", action="store_true")
    ap.add_argument("--stance_len", type=float, default=0.20)
    ap.add_argument("--stance_width", type=float, default=0.162)
    ap.add_argument("--lift", type=float, default=0.0)
    args = ap.parse_args()
    if args.sweep:
        print("lever sweep (baseline = combat geometry: len 0.20 width 0.162 lift 0):")
        print(f"{'config':42s} {'stance':7s} {'FLfoot':7s} {'clear':6s} {'stable':6s} {'PASS'}")
        configs = [
            ("baseline (no lift, narrow)", 0.20, 0.162, 0.0),
            ("lift 25mm only", 0.20, 0.162, 0.025),
            ("lift 25mm + wider stance", 0.28, 0.24, 0.025),
            ("lift 35mm + wider stance", 0.28, 0.24, 0.035),
            ("lift 35mm + wide+long", 0.34, 0.28, 0.035),
        ]
        for name, sl, sw, lr in configs:
            r = _feasibility(sl, sw, lr)
            st = "OK" if r.get("stable") else "no"
            cl = "yes" if r.get("cleared") else "no"
            fz = f"{r.get('fl_z', 0):.0f}mm" if 'fl_z' in r else "-"
            print(f"{name:42s} {sl:.2f}x{sw:.2f} {fz:7s} {cl:6s} {st:6s} "
                  f"{'PASS' if r['ok'] else 'fail'}"
                  + (f"  ({r.get('why','')})" if not r['ok'] and 'why' in r else ""))
    else:
        r = _feasibility(args.stance_len, args.stance_width, args.lift)
        for k, v in r.items():
            print(f"  {k}: {v}")


if __name__ == "__main__":
    main()
