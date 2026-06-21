# SPDX-License-Identifier: MIT
"""arena/feasibility.py — PHYSICAL FEASIBILITY PRE-FLIGHT (catch infeasible designs in seconds, on the
CPU, BEFORE any GPU training).

THE LESSON (2026-06-21): a long self-play GPU run was burned to discover that the reward demanded
something the BODY physically could not do — stand above the fall threshold. The db42s03 was ~3×
undersized, so the max stable stand (0.185 m) sat BELOW the `done` threshold (0.18 m) → survival was
geometrically impossible, and every reward lever / extra training step left the metric dead-flat. A
flat curve looks identical whether the target is HARD or IMPOSSIBLE; that ambiguity cost the run.

GENERALIZED PRINCIPLE: separate "can't because physics" from "can't because not-yet-learned" — cheaply,
up front. RL cannot beat a torque/geometry wall. For each success-condition the reward/termination
*implies*, ask: "is there ANY controller — even a hand-scripted one — that achieves it within the
actuator envelope?" That is a physics question, answerable in seconds without learning.

Probes (each returns a Verdict PASS/WARN/FAIL + the limiting number):
  * stand        — max STABLE stand height on the REAL contact model vs the fall threshold (+margin).
                   This is the one that would have caught the survival=0 bug in one comparison.
  * torque_margin— gravity-comp torque to hold a nominal stance vs the actuator limit (τ_avail/τ_req).
  * scripted_floor— best `alive` a hand controller (zero / PD-to-stance) reaches WITHOUT learning —
                   the model-free achievability floor. If a scripted controller can't, RL is a gamble.
  * reach        — max limb/striker extension vs the benchmark's closest spawn (can it even connect?).

  python -m arena.feasibility              # report on the current robot.toml
  python -m arena.feasibility --selftest   # proves it FAILs the old body, PASSes the fixed one
"""

from __future__ import annotations

import dataclasses as dc
import sys
from pathlib import Path

import numpy as np

ROBOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROBOT))


@dc.dataclass
class Verdict:
    name: str
    status: str            # "PASS" | "WARN" | "FAIL"
    value: float
    limit: float
    detail: str

    def __str__(self):
        mark = {"PASS": "✓", "WARN": "▲", "FAIL": "✗"}[self.status]
        return f"  {mark} {self.name:<14} {self.status:<4} {self.detail}"


# ----------------------------------------------------------------------------- model helpers
def _match_model(spec, striker=True):
    import mujoco
    from gen_robot_mjcf import build_match
    xml = build_match(spec, spec, sep=2.4, self_collision=True, striker=striker, striker_b=False)
    return mujoco.MjModel.from_xml_string(xml)


def _A_joints(m):
    import mujoco
    names = [mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_ACTUATOR, i) for i in range(m.nu)]
    A = [i for i, n in enumerate(names) if n.startswith("A_") and not n.endswith("strike_m")]
    trn = m.actuator_trnid[:, 0]
    jq = [m.jnt_qposadr[trn[i]] for i in A]
    jd = [m.jnt_dofadr[trn[i]] for i in A]
    jr = np.array([m.jnt_range[trn[i]] for i in A])
    tmax = m.actuator_forcerange[A, 1].copy()
    return names, A, jq, jd, jr, tmax


def _hold(m, At, names, A, jq, jd, jr, tmax, knee, flex, steps, fall, fs=5, noise=0.0,
          kp=22, kd=1.1, seed=0):
    """Roll out a fixed-stance PD controller; return (alive_steps, mean_stand_z, mean_saturation)."""
    import mujoco
    rng = np.random.default_rng(seed)
    d = mujoco.MjData(m); mujoco.mj_forward(m, d)
    qt = d.qpos[jq].copy()
    for k, i in enumerate(A):
        if "knee" in names[i]: qt[k] = knee
        elif "flex" in names[i]: qt[k] = flex
    qt = np.clip(qt, jr[:, 0], jr[:, 1])
    fell, zs, sat = None, [], []
    for t in range(steps):
        for _ in range(fs):
            tau = np.clip(kp * (qt - d.qpos[jq]) - kd * d.qvel[jd], -tmax, tmax)
            a = tau / np.where(tmax > 0, tmax, 1.0)
            if noise: a = a + rng.normal(0, noise, len(A))
            d.ctrl[A] = np.clip(a, -1, 1)
            mujoco.mj_step(m, d)
        z = float(d.xpos[At][2]); zs.append(z)
        sat.append(float(np.mean(np.abs(tau) >= tmax - 1e-6)))
        if fell is None and z < fall: fell = t
    return (steps if fell is None else fell), float(np.mean(zs[-30:])), float(np.mean(sat))


# ----------------------------------------------------------------------------- the probes
_STANCES = [(-1.6, -0.5), (-1.2, -0.4), (-0.9, -0.3), (-0.6, -0.5), (-0.6, -0.2), (-0.5, -0.1)]


def probe_stand(spec, fall_threshold, steps=200, margin=0.04) -> Verdict:
    """Max STABLE stand height on the real (feet-only) contact model vs the fall threshold."""
    import mujoco
    m = _match_model(spec); At = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, "A_torso")
    h = _A_joints(m)
    best_z, best_alive = 0.0, 0
    for knee, flex in _STANCES:
        alive, z, _ = _hold(m, At, *h, knee, flex, steps, fall_threshold)
        if alive >= steps and z > best_z:        # only count STABLE (full-bout) stances
            best_z, best_alive = z, alive
    if best_alive < steps:                        # nothing held a full bout
        # report the tallest height reached at all, to show how far short
        tall = max(_hold(m, At, *h, k, f, steps, 0.0)[1] for k, f in _STANCES)
        return Verdict("stand", "FAIL", tall, fall_threshold,
                       f"NO stance holds a full bout above fall={fall_threshold:.3f}; "
                       f"tallest reachable torso-z={tall:.3f} → survival is INFEASIBLE "
                       f"(raise torque-to-weight or lower the threshold)")
    head = best_z - fall_threshold
    status = "PASS" if head >= margin else "WARN"
    return Verdict("stand", status, best_z, fall_threshold,
                   f"stable stand torso-z={best_z:.3f}, fall={fall_threshold:.3f}, "
                   f"headroom={head:.3f} (want ≥{margin:.2f})")


def probe_torque_margin(spec, steps=200) -> Verdict:
    """Available joint torque vs the torque needed to hold a stance (scaled-ceiling bisection)."""
    import mujoco
    m = _match_model(spec); At = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, "A_torso")
    names, A, jq, jd, jr, tmax0 = _A_joints(m)
    # find the smallest torque-scale at which a nominal stance holds a full bout
    nominal = (-0.6, -0.5)
    need = None
    for ts in (0.5, 0.75, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0):
        alive, _, _ = _hold(m, At, names, A, jq, jd, jr, tmax0 * ts, *nominal, steps, 0.09)
        if alive >= steps:
            need = ts; break
    if need is None:
        return Verdict("torque_margin", "FAIL", 0.0, 1.0,
                       "no stance holds even at 6× torque — geometry/CoM problem, not just motor")
    margin = 1.0 / need                            # τ_available / τ_required at nominal stance
    status = "PASS" if margin >= 1.3 else ("WARN" if margin >= 1.0 else "FAIL")
    return Verdict("torque_margin", status, margin, 1.0,
                   f"τ_avail/τ_req ≈ {margin:.2f} (holds at {need:.2g}× torque; want ≥1.3 for control authority)")


def probe_scripted_floor(spec, fall_threshold, steps=200) -> Verdict:
    """Best `alive` a hand controller reaches WITHOUT learning, robustness-checked under noise."""
    import mujoco
    m = _match_model(spec); At = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, "A_torso")
    h = _A_joints(m)
    best = max((_hold(m, At, *h, k, f, steps, fall_threshold)[0] for k, f in _STANCES), default=0)
    # robustness of the best stance under perturbation (knee -0.6 family)
    rob = min(_hold(m, At, *h, -0.6, -0.5, steps, fall_threshold, noise=0.15, seed=s)[0] for s in range(3))
    frac = best / steps
    status = "PASS" if (frac >= 0.95 and rob >= 0.9 * steps) else ("WARN" if frac >= 0.5 else "FAIL")
    return Verdict("scripted_floor", status, frac, 1.0,
                   f"best scripted alive={best}/{steps} ({frac:.0%}), noisy-min={rob}/{steps} "
                   f"→ {'a basin RL will find' if status=='PASS' else 'RL is a gamble here'}")


def probe_reach(spec, bench_min_sep=0.20) -> Verdict:
    """Max limb+striker extension vs the closest benchmark spawn — can it physically connect?"""
    ld = spec["leg_defaults"]; reach = ld["thigh_len"] + ld["calf_len"]
    s = spec.get("striker", {})
    if s.get("enabled"):
        reach += s.get("stroke", 0) + s.get("rod_len", 0)
    status = "PASS" if reach >= bench_min_sep else "WARN"
    return Verdict("reach", status, reach, bench_min_sep,
                   f"max reach ≈ {reach:.2f} m vs closest spawn {bench_min_sep:.2f} m")


# ----------------------------------------------------------------------------- top-level report
def feasibility_report(spec, fall_threshold, bench_min_sep=0.20, verbose=True) -> dict:
    """Run every probe; return {ok, verdicts, fails}. `ok=False` if any core probe FAILs."""
    verdicts = [
        probe_stand(spec, fall_threshold),
        probe_torque_margin(spec),
        probe_scripted_floor(spec, fall_threshold),
        probe_reach(spec, bench_min_sep),
    ]
    fails = [v for v in verdicts if v.status == "FAIL"]
    warns = [v for v in verdicts if v.status == "WARN"]
    ok = not fails
    if verbose:
        print(f"FEASIBILITY PRE-FLIGHT  (fall_threshold={fall_threshold}, body="
              f"{spec['torso']['mass']}kg, gear {spec['actuator']['gear']}):")
        for v in verdicts:
            print(v)
        tag = "FEASIBLE ✓" if ok else "INFEASIBLE ✗ — do NOT spend GPU on this"
        print(f"  => {tag}" + (f"  ({len(warns)} warning(s))" if warns and ok else ""))
    return {"ok": ok, "verdicts": verdicts, "fails": fails, "warns": warns}


def preflight_gate(spec, fall_threshold, bench_min_sep=0.20):
    """Pipeline gate: raise if the design is physically infeasible for the reward's demands."""
    rep = feasibility_report(spec, fall_threshold, bench_min_sep, verbose=True)
    if not rep["ok"]:
        raise RuntimeError("feasibility pre-flight FAILED: "
                           + "; ".join(f.detail for f in rep["fails"]))
    return rep


# ----------------------------------------------------------------------------- selftest
def _selftest():
    import copy
    from gen_robot_mjcf import load_spec
    fixed = load_spec(ROBOT / "robot.toml")                      # the fixed body (gear 12, 3.5 kg)

    # (1) the FIXED body PASSes at the new threshold 0.09
    rep_fixed = feasibility_report(fixed, fall_threshold=0.09, verbose=True)
    assert rep_fixed["ok"], "fixed body should be feasible"
    print()

    # (2) the OLD body at the OLD threshold FAILs — i.e. this WOULD have caught the bug pre-flight
    old = copy.deepcopy(fixed)
    old["torso"]["mass"] = 6.0
    old["actuator"]["gear"] = 6.0
    old["torso"]["spawn_height"] = 0.34
    rep_old = feasibility_report(old, fall_threshold=0.18, verbose=True)
    assert not rep_old["ok"], "old body @0.18 must be flagged INFEASIBLE"
    assert any(f.name == "stand" for f in rep_old["fails"]), "the stand probe must be the failure"

    print("\nPROVEN: feasibility pre-flight PASSes the fixed body (gear12/3.5kg @0.09) and FAILs the old "
          "body (gear6/6kg @0.18) on the `stand` probe — it would have caught survival=0 in seconds, "
          "no GPU. Generalized probes: stand / torque_margin / scripted_floor / reach.")


if __name__ == "__main__":
    if "--selftest" in sys.argv:
        _selftest()
    else:
        from gen_robot_mjcf import load_spec
        spec = load_spec(ROBOT / "robot.toml")
        feasibility_report(spec, fall_threshold=0.09, verbose=True)
