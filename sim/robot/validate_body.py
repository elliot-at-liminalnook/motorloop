# SPDX-License-Identifier: MIT
"""validate_body.py — fail-fast body-trainability check. ONE command, PASS/FAIL per property +
an overall verdict, BEFORE any GPU time. Run after every body change (motor/gear/spring/geometry).

T1 rules (post gear-bug rewrite; see notes/rl-verification-playbook.md):
  * torque is MEASURED through the training causal path (ctrl -> qfrc_actuator),
    never read from actuator_forcerange — the pre-2026-07 validator trusted that
    attribute and green-lit a body with 8% of its design torque.
  * the stance bar is ABSOLUTE: the body must hold THE canonical training stance
    (leg_defaults stand_flex/stand_knee — what the envs PD around), not the best
    stance a grid search can find. Bars never adapt downward.

Checks:
  stands       — canonical stance PD-hold (env-faithful two-robot match model)
  upright      — up-axis·world-up at the stand (xmat); >0.80 = vertical, not leaning
  torque_margin— max per-joint steady holding torque / MEASURED deliverable; <0.60
  drop_settle  — release from spawn height, PD-hold stance: settles standing
  spawn_launch — no catapult / floor penetration at spawn (feet-underground bug)
  max_effort   — full ctrl on one knee moves it at the spec-torque scale
                 (catches BOTH a weak body and a mis-scaled catapult-strong one)
  leg_speed    — joint no-load speed (rad/s) from the motor envelope
  striker_aim  — does the rod point FORWARD (toward a front opponent), not the floor?

  CODESIGN_OUT=/tmp/cd python validate_body.py            # validates the current robot.toml
  python validate_body.py --strip-gear                    # regression demo: gear-bug body must FAIL
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
    jr = np.array([m.jnt_range[trn[i]] for i in A])
    return At, names, A, jq, jd, jr, measured_tmax(m, A, jd)


def measured_tmax(m, A, jd):
    """Deliverable joint torque per actuator, MEASURED (ctrl=1 -> qfrc_actuator).

    This is the same causal path training uses. actuator_forcerange is not
    consulted anywhere in this file: it documents intent, it does not deliver
    torque (gear does), and trusting it is how the 8%-torque body passed.
    """
    d = mujoco.MjData(m)
    d.ctrl[A] = 1.0
    mujoco.mj_forward(m, d)
    return np.abs(d.qfrc_actuator[jd])


def _hold(m, knee, flex, steps=200, kp=None, kd=None, start_spawn=None):
    # gains default to constants.PD_KP/PD_KD — the validator must hold the stance
    # the way TRAINING holds it, or its sag bar measures the wrong loop (V.1).
    from constants import PD_KD, PD_KP
    kp = PD_KP if kp is None else kp
    kd = PD_KD if kd is None else kd
    """PD-hold the given stance; ctrl scaling uses MEASURED tmax so the commanded
    torque equals the delivered torque on any model, bugged or not."""
    At, names, A, jq, jd, jr, tmax = _idx(m)
    d = mujoco.MjData(m)
    if start_spawn is not None:                      # drop test: start at spawn height
        root = m.jnt_qposadr[mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_JOINT, "A_root")]
        d.qpos[root + 2] = start_spawn
    mujoco.mj_forward(m, d)
    qt = d.qpos[jq].copy()
    for k, i in enumerate(A):
        if "knee" in names[i]: qt[k] = knee
        elif "flex" in names[i]: qt[k] = flex
    qt = np.clip(qt, jr[:, 0], jr[:, 1]); zs = []; ups = []; taus = []; fell = None; pk = 0.0
    for t in range(steps):
        for _ in range(FS):
            tau = np.clip(kp * (qt - d.qpos[jq]) - kd * d.qvel[jd], -tmax, tmax)
            d.ctrl[A] = tau / np.where(tmax > 0, tmax, 1); mujoco.mj_step(m, d)
        z = float(d.xpos[At][2]); upz = float(d.xmat[At].reshape(-1)[8])
        zs.append(z); ups.append(upz); taus.append(np.abs(tau)); pk = max(pk, z)
        if fell is None and (z < 0.1 or upz < 0.3): fell = t
    hold_tau = np.mean(np.asarray(taus)[-40:], axis=0)          # steady per-joint holding torque
    return dict(z=float(np.mean(zs[-40:])), up=float(np.mean(ups[-40:])),
                margin=float(np.max(hold_tau / np.where(tmax > 0, tmax, 1))),
                peak_z=pk, stable=(fell is None))


def _max_effort_ratio(m, spec):
    """Full ctrl on one knee, free-floating at the CANONICAL stance pose: does the
    joint accelerate at the SPEC-torque scale?  ratio = Δqacc / (M⁻¹·τ_spec)_j —
    ~1 when the actuator delivers the design torque, ~1/13 on the gear-bug body,
    >>1 on a mis-scaled catapult-strong one. Band [0.3, 3] passes; outside fails.
    Pose matters: at qpos0 every knee sits at 0 rad, OUTSIDE its [-2.6,-0.4] range,
    and the joint-limit constraint silently absorbs the torque being measured."""
    tau_spec = joint_torque_limit(spec)
    ld = spec["leg_defaults"]
    At, names, A, jq, jd, jr, tmax = _idx(m)
    knee = next(k for k, i in enumerate(A) if "knee" in names[i])
    d = mujoco.MjData(m)
    root = m.jnt_qposadr[mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_JOINT, "A_root")]
    d.qpos[root + 2] = 1.5                                        # airborne: pure inertial response
    for k, i in enumerate(A):                                     # inside joint limits
        d.qpos[jq[k]] = float(ld["stand_knee"]) if "knee" in names[i] else (
            float(ld["stand_flex"]) if "flex" in names[i] else float(ld.get("stand_abd", 0.0)))
    mujoco.mj_forward(m, d); qacc0 = d.qacc[jd[knee]]
    d.ctrl[A[knee]] = 1.0
    mujoco.mj_forward(m, d); dq = d.qacc[jd[knee]] - qacc0
    M = np.zeros((m.nv, m.nv)); mujoco.mj_fullM(m, M, d.qM)
    tau_vec = np.zeros(m.nv); tau_vec[jd[knee]] = tau_spec        # prediction uses SPEC torque:
    pred = np.linalg.solve(M, tau_vec)[jd[knee]]                  # measured/spec must NOT self-normalize
    return float(abs(dq) / abs(pred))


def validate(spec, verbose=True, xml_transform=None):
    xml = build_match(spec, spec, sep=1.2, self_collision=True, striker=True, striker_b=True)
    if xml_transform is not None:                    # test hook: e.g. strip gear (regression demo)
        xml = xml_transform(xml)
    m = mujoco.MjModel.from_xml_string(xml)
    tau = joint_torque_limit(spec)
    ld = spec.get("leg_defaults", {})
    stand_flex = float(ld.get("stand_flex", -0.4))
    stand_knee = float(ld.get("stand_knee", -1.1))
    # ABSOLUTE bar: the canonical training stance (what the envs PD around), no search-down.
    best = _hold(m, stand_knee, stand_flex)
    # kinematic stance height (leg geometry at the canonical angles) -> sag bar
    tl, cl = float(ld["thigh_len"]), float(ld["calf_len"])
    z_kin = tl * np.cos(stand_flex) + cl * np.cos(stand_flex + stand_knee) + float(ld["foot_radius"])
    # drop test: release at spawn height while PD-holding the stance; must settle standing
    drop = _hold(m, stand_knee, stand_flex, steps=300, start_spawn=float(spec["torso"]["spawn_height"]))
    effort = _max_effort_ratio(m, spec)
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

    stands = best["stable"] and best["z"] > 0.75 * z_kin
    # settle bar is self-consistent: return to (within 5% of) the static stance height.
    # peak bar spawn+0.25: above a soft-contact rebound, far below a catapult (~1 m rise).
    settles = drop["stable"] and drop["z"] > 0.95 * best["z"] and drop["up"] > 0.80 \
        and drop["peak_z"] < spawn + 0.25
    checks = [
        ("stands", stands,
         f"CANONICAL stance (flex={stand_flex:.2f} knee={stand_knee:.2f}) torso-z={best['z']:.3f} "
         f"(want >{0.75 * z_kin:.3f} = 75% of kinematic {z_kin:.3f})", stands),
        # upright = passive PD-hold (a LOWER bound; a trained policy holds more vertical). The real
        # trainability predictor is torque_margin: the db42s03 bodies leaned because they SATURATED
        # (no authority to balance); a body that stands ~upright with torque headroom CAN be trained up.
        ("upright", best["up"] > 0.80, f"up_z={best['up']:.2f} passive (want >0.80; trained does better)", best["up"] > 0.80),
        ("torque_margin", best["margin"] < 0.60,
         f"max joint holding/MEASURED deliverable={best['margin']:.2f} (want <0.60 — headroom to move)",
         best["margin"] < 0.60),
        ("drop_settle", settles,
         f"drop from {spawn:.2f}m: z={drop['z']:.3f} up={drop['up']:.2f} peak={drop['peak_z']:.2f} (settles standing)",
         settles),
        ("spawn_launch", no_launch, f"spawn_pen={spawn_pen:.3f} peak_z={pk:.2f} vs spawn {spawn:.2f} (no catapult)", no_launch),
        ("max_effort", 0.3 < effort < 3.0,
         f"knee Δqacc/(M⁻¹τ_spec)={effort:.2f} (want 0.3–3; ~{1/12.97:.2f} = gear-bug signature)",
         0.3 < effort < 3.0),
        ("leg_speed", leg_speed > 8.0, f"no-load joint speed={leg_speed:.1f} rad/s (want >8)", leg_speed > 8.0),
        ("striker_aim", aim > 0.0, f"rod forward/down ratio={aim:.2f} (>0 = points forward not floor)", aim > 0.0),
    ]
    hard = ("stands", "upright", "torque_margin", "drop_settle", "spawn_launch", "max_effort")
    ok = all(c[1] for c in checks if c[0] in hard)
    if verbose:
        print(f"BODY VALIDATE  motor={spec['actuator']['motor']} gear={spec['actuator']['gear']} "
              f"thigh={spec['leg_defaults']['thigh_len']} stiff={spec['leg_defaults']['joint_stiffness']} "
              f"joint-τ=±{tau:.2f}N·m (spec) measured=±{_idx(m)[6].max():.2f}N·m")
        for name, good, detail, _ in checks:
            print(f"  {'✓' if good else '✗'} {name:14s} {detail}")
        print(f"  => {'TRAINABLE (stands upright)' if ok else 'NOT READY — fix stand/upright first'}")
    return dict(ok=ok, checks=checks, best=best, drop=drop, effort=effort,
                leg_speed=leg_speed, aim=aim)


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
    import sys as _sys
    from pathlib import Path
    _transform = None
    if "--strip-gear" in _sys.argv:                  # regression demo: the gear-bug body must FAIL
        from test_model_contract import strip_gear as _transform
    r = validate(load_spec(Path(__file__).resolve().parent / "robot.toml"), xml_transform=_transform)
    raise SystemExit(0 if r["ok"] else 1)
