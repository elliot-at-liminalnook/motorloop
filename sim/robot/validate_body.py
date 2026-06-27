# SPDX-License-Identifier: MIT
"""validate_body.py — fail-fast body-trainability check. ONE command, PASS/FAIL per property +
an overall verdict, BEFORE any GPU time. Run after every body change (motor/gear/spring/geometry).

Checks (each with a healthy threshold):
  stands       — tallest STABLE stand (PD hold 200 steps, real feet-only match model)
  upright      — up-axis·world-up at the stand (xmat); >0.85 = vertical, not leaning
  torque_margin— actuator saturation while holding the stand; <0.4 = real headroom (no straining)
  leg_speed    — joint no-load speed (rad/s) from the motor envelope; higher = faster legs
  striker_aim  — does the rod point FORWARD (toward a front opponent), not into the floor?

  CODESIGN_OUT=/tmp/cd python validate_body.py            # validates the current robot.toml
"""
from __future__ import annotations
import numpy as np, mujoco
from gen_robot_mjcf import build_match, load_spec, joint_torque_limit

FS = 5


def _idx(m):
    At = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, "A_torso")
    names = [mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_ACTUATOR, i) for i in range(m.nu)]
    A = [i for i, n in enumerate(names) if n.startswith("A_") and not n.endswith("strike_m")]
    trn = m.actuator_trnid[:, 0]
    jq = [m.jnt_qposadr[trn[i]] for i in A]; jd = [m.jnt_dofadr[trn[i]] for i in A]
    jr = np.array([m.jnt_range[trn[i]] for i in A]); tmax = m.actuator_forcerange[A, 1].copy()
    return At, names, A, jq, jd, jr, tmax


def _hold(m, knee, flex, steps=200, kp=22, kd=1.1):
    At, names, A, jq, jd, jr, tmax = _idx(m)
    d = mujoco.MjData(m); mujoco.mj_forward(m, d)
    qt = d.qpos[jq].copy()
    for k, i in enumerate(A):
        if "knee" in names[i]: qt[k] = knee
        elif "flex" in names[i]: qt[k] = flex
    qt = np.clip(qt, jr[:, 0], jr[:, 1]); zs = []; ups = []; sat = []; fell = None
    for t in range(steps):
        for _ in range(FS):
            tau = np.clip(kp * (qt - d.qpos[jq]) - kd * d.qvel[jd], -tmax, tmax)
            d.ctrl[A] = tau / np.where(tmax > 0, tmax, 1); mujoco.mj_step(m, d)
        z = float(d.xpos[At][2]); upz = float(d.xmat[At].reshape(-1)[8])
        zs.append(z); ups.append(upz); sat.append(float(np.mean(np.abs(tau) >= tmax - 1e-6)))
        if fell is None and (z < 0.1 or upz < 0.3): fell = t
    return dict(z=float(np.mean(zs[-40:])), up=float(np.mean(ups[-40:])),
                sat=float(np.mean(sat)), stable=(fell is None))


def validate(spec, verbose=True):
    m = mujoco.MjModel.from_xml_string(build_match(spec, spec, sep=1.2, self_collision=True,
                                                   striker=True, striker_b=True))
    tau = joint_torque_limit(spec)
    ld = spec.get("leg_defaults", {})
    stand_flex = float(ld.get("stand_flex", -0.4))
    stand_knee = float(ld.get("stand_knee", -1.1))
    # best STABLE stance: tallest upright one that holds the full bout
    best = None
    for knee in (stand_knee, -0.8, -1.1, -1.4):
        for flex in (stand_flex, -0.2, 0.0, 0.2):
            r = _hold(m, knee, flex)
            if r["stable"] and (best is None or r["up"] > best["up"]):   # MOST UPRIGHT stable stance
                best = r
    best = best or _hold(m, -1.2, -0.4)
    # leg no-load speed from the motor envelope
    import sys; sys.path.insert(0, str(__import__("pathlib").Path(__file__).resolve().parents[1]))
    from reality_gap import default_uncertainty
    unc = default_uncertainty(spec["actuator"]["motor"], spec["actuator"]["gear"])
    dp = {f.name: float(getattr(unc["act"], f.name)[0]) for f in __import__("dataclasses").fields(unc["act"])}
    leg_speed = dp["vbus"] / (dp["ke_phase"] * float(spec["actuator"]["gear"]))
    # striker aim: rod tip vs foot in the body frame — forward (x) component should dominate down (z)
    d = mujoco.MjData(m); mujoco.mj_forward(m, d)
    aim = _striker_forward(m, d)
    # spawn-launch: BOTH robots spawned at spawn_height + the _q0 stance must NOT catapult. The feet-
    # underground bug (spawn_height too low for the leg length) penetrates the floor and the solver
    # launches the body to ~1 m → survival 0, uprightness <0. This check (env-faithful: both robots)
    # is what a PD-hold-from-rest misses. knee/flex below MUST match train_adversarial._q0.
    roots = {p: m.jnt_qposadr[mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_JOINT, f"{p}_root")] for p in ("A", "B")}
    knj = [m.jnt_qposadr[j] for j in range(m.njnt) if (mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_JOINT, j) or "").endswith("_knee")]
    fxj = [m.jnt_qposadr[j] for j in range(m.njnt) if (mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_JOINT, j) or "").endswith("_flex")]
    spawn = float(spec["torso"]["spawn_height"])
    dd = mujoco.MjData(m); mujoco.mj_resetData(m, dd)
    for p in roots: dd.qpos[roots[p] + 2] = spawn
    for a in knj: dd.qpos[a] = stand_knee
    for a in fxj: dd.qpos[a] = stand_flex
    mujoco.mj_forward(m, dd)
    spawn_pen = float(np.maximum(0, -dd.contact.dist).max()) if dd.ncon else 0.0
    At2 = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, "A_torso"); pk = float(dd.xpos[At2][2])
    for _ in range(120):
        mujoco.mj_step(m, dd); pk = max(pk, float(dd.xpos[At2][2]))
    # peak_z (no upward catapult) is the hard gate; a small static pen that resolves DOWNWARD is harmless
    no_launch = spawn_pen < 0.09 and pk < spawn + 0.12

    checks = [
        ("stands", best["stable"], f"stable stand torso-z={best['z']:.3f}", best["stable"]),
        # upright = passive PD-hold (a LOWER bound; a trained policy holds more vertical). The real
        # trainability predictor is torque_margin: the db42s03 bodies leaned because they SATURATED
        # (no authority to balance); a body that stands ~upright with torque headroom CAN be trained up.
        ("upright", best["up"] > 0.80, f"up_z={best['up']:.2f} passive (want >0.80; trained does better)", best["up"] > 0.80),
        ("torque_margin", best["sat"] < 0.25, f"saturation={best['sat']:.2f} (want <0.25 — authority to balance)", best["sat"] < 0.25),
        ("leg_speed", leg_speed > 8.0, f"no-load joint speed={leg_speed:.1f} rad/s (want >8)", leg_speed > 8.0),
        ("striker_aim", aim > 0.0, f"rod forward/down ratio={aim:.2f} (>0 = points forward not floor)", aim > 0.0),
        ("spawn_launch", no_launch, f"spawn_pen={spawn_pen:.3f} peak_z={pk:.2f} vs spawn {spawn:.2f} (no catapult)", no_launch),
    ]
    ok = all(c[1] for c in checks if c[0] in ("stands", "upright", "torque_margin", "spawn_launch"))   # hard gates
    if verbose:
        print(f"BODY VALIDATE  motor={spec['actuator']['motor']} gear={spec['actuator']['gear']} "
              f"thigh={spec['leg_defaults']['thigh_len']} stiff={spec['leg_defaults']['joint_stiffness']} "
              f"joint-τ=±{tau:.2f}N·m")
        for name, good, detail, _ in checks:
            print(f"  {'✓' if good else '✗'} {name:14s} {detail}")
        print(f"  => {'TRAINABLE (stands upright)' if ok else 'NOT READY — fix stand/upright first'}")
    return dict(ok=ok, checks=checks, best=best, leg_speed=leg_speed, aim=aim)


def _striker_forward(m, d):
    """Rod direction in the torso frame: + if forward-dominant (toward a front opponent), - if down."""
    try:
        rod = [g for g in range(m.ngeom) if (mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_GEOM, g) or "").startswith("A_") and "rod" in (mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_GEOM, g) or "")]
        if not rod: return -1.0
        g = rod[0]; rod_dir = d.geom_xmat[g].reshape(3, 3)[:, 2]   # rod capsule local-z axis in world
        # forward (world +x toward opponent) vs down (world -z)
        return float(abs(rod_dir[0]) - abs(rod_dir[2]))
    except Exception:
        return -1.0


if __name__ == "__main__":
    from pathlib import Path
    validate(load_spec(Path(__file__).resolve().parent / "robot.toml"))
