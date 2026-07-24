# SPDX-License-Identifier: MIT
"""prove_reference_gait.py — open-loop validation of sim/robot/reference_gait.json.

Plays the reference PD position targets through the SAME P-only servo model the
suite/env uses (test_mesh_robot_behavior.py pd_ctrl pattern with kp=(2,40,6),
plus mesh_commanded_env's torque-speed derating: drive torque limit =
stall*(1-|qvel|/wfree) clipped [0,1]; braking keeps full stall) in plain mujoco,
from the settled qpos0 stance, for >= 3 cycles.

PASS requires:
  * finite state, torso z > 0.30 and up_z > 0.85 throughout the cycles,
  * net forward displacement > 0 with every cycle advancing (consistent),
  * every foot actually breaks floor contact during its intended swing window.

CLI:
  .venv-warp/bin/python prove_reference_gait.py [--json reference_gait.json]
      [--cycles 4] [--mu 1.2] [--belt4] [--quiet]
`--mu` overrides sliding friction on ALL geoms (mujoco combines by max, so this
sets the contact mu); `--belt4` rebuilds yaw at belt ratio 4 (4.24 N.m, 3.80
rad/s) instead of the shipped ratio 6 for the pulley-decision sensitivity check.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import mujoco
import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

from gen_mesh_robot_mjcf import (build_mesh_robot, SERVO_STALL_NM,  # noqa: E402
                                 SERVO_FREE_RAD_S, WFREE)

LEGS = ("FL", "FR", "RL", "RR")
KP = (2.0, 40.0, 6.0)          # the suite's P-only gains (yaw, swing, knee)
CTRL_HZ = 50.0                 # fleet control rate: targets held for frame_skip=5


def servo_ctrl(d, target, kp, gear, wfree):
    """P-only torque through the actuators with the servo torque-speed line."""
    tau = kp * (target - d.qpos[QADR])
    qd = d.qvel[DADR]
    derate = np.clip(1.0 - np.abs(qd) / wfree, 0.0, 1.0)
    lim = np.where(tau * qd > 0.0, derate, 1.0)     # drive derated, braking full
    d.ctrl[:] = np.clip(tau / gear, -lim, lim)


def foot_contacts(m, d, foot_gids, floor_gid):
    c = np.zeros(4, dtype=bool)
    for i in range(d.ncon):
        g1, g2 = d.contact.geom1[i], d.contact.geom2[i]
        for k, fg in enumerate(foot_gids):
            if (g1 == fg and g2 == floor_gid) or (g2 == fg and g1 == floor_gid):
                c[k] = True
    return c


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--json", default=str(HERE / "reference_gait.json"))
    ap.add_argument("--cycles", type=int, default=4)
    ap.add_argument("--startup_cycles", type=int, default=2)
    ap.add_argument("--mu", type=float, default=None)
    ap.add_argument("--belt4", action="store_true")
    ap.add_argument("--quiet", action="store_true")
    args = ap.parse_args()

    ref = json.loads(Path(args.json).read_text())
    T, n = float(ref["period_s"]), int(ref["n"])
    qt = np.asarray(ref["qpos_targets"], dtype=float)          # (n, 12)
    fsw = np.asarray(ref["feet_swing"], dtype=bool)            # (n, 4)
    assert qt.shape == (n, 12) and fsw.shape == (n, 4)

    m = mujoco.MjModel.from_xml_string(build_mesh_robot())
    if args.mu is not None:
        m.geom_friction[:, 0] = args.mu
    yaw_ids = []
    aname = lambda a: mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_ACTUATOR, a)
    wfree = np.empty(m.nu)
    for a in range(m.nu):
        kind = aname(a).split("_")[1]                          # yaw|swing|knee
        wfree[a] = {"yaw": WFREE["hip_yaw"], "swing": WFREE["leg_swing"],
                    "knee": WFREE["knee_blade"]}[kind]
        if kind == "yaw":
            yaw_ids.append(a)
    if args.belt4:                                             # pulley sensitivity
        g4 = SERVO_STALL_NM * 4.0
        for a in yaw_ids:
            m.actuator_gear[a, 0] = g4
            m.actuator_forcerange[a] = (-g4, g4)
            wfree[a] = SERVO_FREE_RAD_S / 4.0
    gear = m.actuator_gear[:m.nu, 0].copy()
    kp = np.array(list(KP) * 4)

    global QADR, DADR
    jids = [int(m.actuator_trnid[a, 0]) for a in range(m.nu)]
    QADR = np.array([int(m.jnt_qposadr[j]) for j in jids])
    DADR = np.array([int(m.jnt_dofadr[j]) for j in jids])
    # sanity: model actuator order must match the reference joint_order
    jname = [mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_JOINT, j) for j in jids]
    assert jname == ref["joint_order"], f"joint order mismatch: {jname}"

    foot_gids = [mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_GEOM, f"{L}_foot") for L in LEGS]
    floor_gid = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_GEOM, "floor")
    torso = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, "torso")

    d = mujoco.MjData(m)
    dt = m.opt.timestep
    steps_per_ctrl = max(1, round(1.0 / (CTRL_HZ * dt)))

    def ref_target(phase):
        x = (phase % 1.0) * n
        i0 = int(x) % n
        i1 = (i0 + 1) % n
        w = x - int(x)
        return (1.0 - w) * qt[i0] + w * qt[i1]

    # soft-start base pose: per-joint cycle mean (pitch/yaw), stance value (knee).
    # Blending into a mid-cycle pose directly DRAGS the pinned feet to staggered
    # stance positions and the asymmetric drag yaws/tips the robot (measured);
    # instead the startup cycles ramp the gait amplitude 0 -> 1 about this base
    # so every foot is STEPPED to its position by the gait's own swing slots.
    base = qt.mean(axis=0)
    knee_cols = [2, 5, 8, 11]
    base[knee_cols] = qt[:, knee_cols].min(axis=0)
    # the fore-aft COM-shift modulation baked into the targets must NOT be
    # scaled away at low amplitude (it is what keeps the COM inside the support
    # triangle when a leg lifts) — restore the (1-a) remainder from the aux data
    aux_dx = np.asarray(ref.get("aux_delta_x", np.zeros(n)), dtype=float)
    aux_dx -= aux_dx.mean()                  # bias is positional, safe to scale
    dfx = ref.get("aux_dfx_dth", {})
    pitch_cols = [1, 4, 7, 10]
    dth_mod = np.zeros((n, 12))
    for li, L in enumerate(LEGS):
        if L in dfx and dfx[L]:
            dth_mod[:, pitch_cols[li]] = -aux_dx / float(dfx[L])

    def mod_at(phase):
        x = (phase % 1.0) * n
        i0 = int(x) % n
        return dth_mod[i0]

    def target_at(phase, a):
        full = ref_target(phase)
        out = base + a * (full - base) + (1.0 - a) * mod_at(phase)
        ak = min(1.0, a / 0.3)               # knee (swing lift) ramps in faster
        out[knee_cols] = base[knee_cols] + ak * (full[knee_cols] - base[knee_cols])
        return out

    def run(t_end, target_fn):
        t = 0.0
        while t < t_end:
            tgt = target_fn(t)
            for _ in range(steps_per_ctrl):
                servo_ctrl(d, tgt, kp, gear, wfree)
                mujoco.mj_step(m, d)
            t += steps_per_ctrl * dt
        return t

    # ---- 1. settle at qpos0 (the settled stance; drop test z ~= 0.41)
    stand = d.qpos[QADR].copy()                                # zeros
    run(2.0, lambda t: stand)
    z_settle = float(d.qpos[2])

    # ---- 2. blend from stance into the (symmetric) gait base pose
    run(3.0, lambda t: stand + (base - stand) * min(1.0, t / 2.5))
    # ---- 2b. startup: ramp gait amplitude over startup_cycles (legs step
    #          themselves into the staggered crawl; not scored)
    t_su = args.startup_cycles * T
    if t_su:
        run(t_su, lambda t: target_at(t / T, min(1.0, t / t_su) ** 2 *
                                      (3 - 2 * min(1.0, t / t_su))))

    # ---- 3. play cycles open-loop, collect evidence
    up_of = lambda: float(d.xmat[torso].reshape(3, 3)[2, 2])
    rec = {"t": [], "x": [], "y": [], "z": [], "up": [], "con": [], "phase": []}
    t, t_end = 0.0, args.cycles * T
    while t < t_end - 1e-9:
        phase = t / T
        tgt = ref_target(phase)
        for _ in range(steps_per_ctrl):
            servo_ctrl(d, tgt, kp, gear, wfree)
            mujoco.mj_step(m, d)
        t += steps_per_ctrl * dt
        rec["t"].append(t); rec["phase"].append((t / T) % 1.0)
        rec["x"].append(float(d.qpos[0])); rec["y"].append(float(d.qpos[1]))
        rec["z"].append(float(d.qpos[2])); rec["up"].append(up_of())
        rec["con"].append(foot_contacts(m, d, foot_gids, floor_gid))
    for k in rec:
        rec[k] = np.asarray(rec[k])

    finite = bool(np.all(np.isfinite(d.qpos)))
    zmin, upmin = float(rec["z"].min()), float(rec["up"].min())
    # per-cycle displacement
    cyc = (rec["t"] / T - 1e-9).astype(int)
    dx_cyc, duty, swing_air = [], [], []
    for c in range(args.cycles):
        sel = cyc == c
        xs = rec["x"][sel]
        dx_cyc.append(xs[-1] - xs[0])
        duty.append(rec["con"][sel].mean(axis=0))
        # airborne fraction inside each foot's intended swing window
        ph = rec["phase"][sel]
        idx = np.minimum((ph * n).astype(int), n - 1)
        air = []
        for li in range(4):
            w = fsw[idx, li]
            air.append(1.0 - rec["con"][sel][w, li].mean() if w.any() else 0.0)
        swing_air.append(air)
    dx_cyc = np.asarray(dx_cyc)
    duty = np.asarray(duty)
    swing_air = np.asarray(swing_air)
    v = float(dx_cyc.sum() / (args.cycles * T))

    ok = (finite and zmin > 0.30 and upmin > 0.85
          and bool(np.all(dx_cyc > 0.005))
          and bool(np.all(swing_air[1:] > 0.5)))   # cycle 0 may carry blend-in transient

    if not args.quiet:
        print(f"settled z={z_settle:.3f}; played {args.cycles} cycles of T={T}s "
              f"(mu={'model 1.2' if args.mu is None else args.mu}"
              f"{', belt4' if args.belt4 else ''})")
        print(f"  finite={finite}  min torso z={zmin:.3f}  min up_z={upmin:.3f}")
        print(f"  dx per cycle [mm]: {[round(1e3 * v, 1) for v in dx_cyc]}  "
              f"(mean {1e3 * dx_cyc.mean():.1f}, expected "
              f"{1e3 * float(ref['expected_speed_ms']) * T:.1f})")
        print(f"  speed {1e3 * v:.1f} mm/s   net dy {1e3 * (rec['y'][-1] - rec['y'][0]):.1f} mm")
        for c in range(args.cycles):
            print(f"  cycle {c}: duty {[round(x, 2) for x in duty[c]]}  "
                  f"swing-airborne frac {[round(x, 2) for x in swing_air[c]]}")
        print("PASS" if ok else "FAIL")
    else:
        print(f"{'PASS' if ok else 'FAIL'} v={1e3 * v:.1f}mm/s zmin={zmin:.3f} "
              f"upmin={upmin:.2f} dx={[round(1e3 * q, 1) for q in dx_cyc]} "
              f"air_min={swing_air[1:].min():.2f}")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
