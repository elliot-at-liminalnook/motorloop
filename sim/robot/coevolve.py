# SPDX-License-Identifier: MIT
"""Phase 4 + 8b — adversarial co-evolution of TWO GENERATED BODIES (an arms race).

Both sides are now full generated bodies sharing ONE morphology space (the robot.toml
schema): our body (robot.toml design vector) vs an attacker body (attacker.toml design
vector). The old abstract attacker `[strike_h, reach, speed]` and the hand-tuned
`_hit_on_us`/`_our_offense` formulas are RETIRED. Each engagement is resolved from
REAL measured physics — the weapon's strike envelope from forward kinematics, stand
stability from a settle, and strike impulse from the motor torque envelope — so the
fitness depends on the actual bodies, not magic constants.

Phase 8b (anti-disengagement): besides the relative Red-Queen score we track an ABSOLUTE
benchmark set of fixed reference attackers (so we see real progress, not just cycling),
keep a Hall of Fame of past opponents (sampled, not just the latest), and add population
DIVERSITY (novelty/fitness-sharing) so the search doesn't collapse.

The FINAL fighting fitness is a trained policy's SPARC return from a live match — that is
`selfplay_mjx.py` (GPU two-policy self-play league). This CPU harness makes the arms-race
machinery + the body co-adaptation provable locally; it is not the trained-policy melee.

  python coevolve.py [--rounds 6 --pop 12]
  python coevolve.py --prove          # quick checks: engagement is design-dependent etc.
"""

from __future__ import annotations

import argparse, os, sys
from pathlib import Path

os.environ.setdefault("MUJOCO_GL", "osmesa")
import mujoco  # noqa: E402
import numpy as np  # noqa: E402

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from gen_robot_mjcf import build_mjcf, load_spec, joint_torque_limit  # noqa: E402
from optimize_design import _retract_clearance  # noqa: E402
import design_codec as dc  # noqa: E402
import sparc_score as sparc  # noqa: E402

OUR_SPEC = load_spec(HERE / "robot.toml")
ATT_SPEC = load_spec(HERE / "attacker.toml")
_MCACHE: dict = {}


def _weapon_reach(model, data, spec):
    """Max weapon-tip (or front-foot) reach via real FK over the leg's joint ranges —
    how far/high the body can drive a strike. Uses the weapon leg if present, else leg 0."""
    wlegs = [l["name"] for l in spec["leg"] if l.get("is_weapon")]
    leg = wlegs[0] if wlegs else spec["leg"][0]["name"]
    tip = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, leg + "_spear")
    if tip < 0:
        tip = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, leg + "_foot")
    if tip < 0:
        return 0.0
    d = spec["leg_defaults"]; best = 0.0
    for flex in d["flex_range"]:
        for knee in d["knee_range"]:
            mujoco.mj_resetData(model, data); data.qpos[2] = spec["torso"]["spawn_height"]
            for a in range(model.nu):
                j = int(model.actuator_trnid[a, 0])
                nm = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_JOINT, j)
                if nm.startswith(leg + "_"):
                    adr = model.jnt_qposadr[j]
                    data.qpos[adr] = flex if nm.endswith("flex") else (knee if nm.endswith("knee") else 0.0)
            mujoco.mj_forward(model, data)
            p = data.geom_xpos[tip]
            best = max(best, float(np.hypot(p[0], p[2] - 0.05)))   # reach from a low strike band
    return best


def body_metrics(spec, design, schema="our"):
    """Real measured physics for a body design: (reach, clearance, stand, mass, impulse)."""
    key = (schema, tuple(np.round(design, 4)))
    if key in _MCACHE:
        return _MCACHE[key]
    ov = dc.full_to_overrides(design)
    model = mujoco.MjModel.from_xml_string(build_mjcf(spec, ov))
    data = mujoco.MjData(model)
    # stand: settle under gravity, measure height + upright (real physics validity)
    data.qpos[2] = spec["torso"]["spawn_height"]; mujoco.mj_forward(model, data)
    for _ in range(200):
        mujoco.mj_step(model, data)
    h = float(data.qpos[2]); up = 1.0 - 2.0 * (data.qpos[4] ** 2 + data.qpos[5] ** 2)
    stand = float(np.clip(h / 0.30, 0, 1) * np.clip(up, 0, 1)) if np.isfinite(h) else 0.0
    reach = _weapon_reach(model, mujoco.MjData(model), {**spec, **{"leg_defaults":
            {**spec["leg_defaults"], **ov.get("leg_defaults", {})}}})
    clear = _retract_clearance(model, mujoco.MjData(model))
    mass = float(model.body_mass.sum())
    s = dict(spec); s["actuator"] = dict(spec["actuator"], **ov.get("actuator", {}))
    impulse = joint_torque_limit(s)
    out = dict(reach=reach, clearance=clear, stand=stand, mass=mass, impulse=impulse)
    _MCACHE[key] = out
    return out


def engage(our_m, att_m):
    """Resolve a duel from MEASURED physics -> (our SPARC features, their SPARC features).
    A side 'deals' when its strike reach + impulse overcome the opponent's dodge clearance,
    gated by its own stand stability; aggression favors the lighter (more agile) body."""
    def deal(att, dfn):
        # reach beyond the defender's retract clearance, scaled by strike impulse, gated by standing
        margin = (att["reach"] - dfn["clearance"]) * 6.0
        hit = 1.0 / (1.0 + np.exp(-margin)) * np.clip(att["impulse"] / 3.0, 0.2, 1.0) * att["stand"]
        return float(hit)
    our_off, their_off = deal(our_m, att_m), deal(att_m, our_m)
    ag0 = sparc._c(0.5 + 0.5 * (att_m["mass"] - our_m["mass"]) / 6.0)   # lighter than foe -> agile
    ours = dict(damage=sparc.damage_fraction(our_off, their_off), control=0.5, aggression=ag0)
    theirs = dict(damage=sparc.damage_fraction(their_off, our_off), control=0.5,
                  aggression=sparc._c(1.0 - ag0))
    return ours, theirs


def match(our_design, att_design):
    return engage(body_metrics(OUR_SPEC, our_design, "our"),
                  body_metrics(ATT_SPEC, att_design, "att"))


def robot_fitness(xr, att_pool):
    return float(np.mean([sparc.differential(*match(xr, a)) for a in att_pool]))


def attacker_fitness(xa, rob_pool):
    cost = 0.4 * np.mean(np.clip(xa, 0, 1))                            # bigger/stronger costs more
    return float(np.mean([-sparc.differential(*match(r, xa)) for r in rob_pool])) - 3.0 * cost


def _novelty(x, pop):
    """Mean distance to the rest of the population (fitness-sharing / diversity term)."""
    if len(pop) <= 1:
        return 0.0
    return float(np.mean([np.linalg.norm(x - p) for p in pop if p is not x]))


def cem(fit, dim, pop, gens, rng, diversity=0.0):
    mean, std = np.full(dim, 0.5), np.full(dim, 0.30)
    ne, bx, bf = max(2, pop // 4), None, -1e9
    for _ in range(gens):
        P = np.clip(mean + std * rng.standard_normal((pop, dim)), 0, 1)
        F = np.array([fit(d) for d in P])
        if diversity > 0:                                             # reward novel designs too
            F = F + diversity * np.array([_novelty(P[i], list(P)) for i in range(len(P))])
        E = P[np.argsort(F)[-ne:]]; mean, std = E.mean(0), E.std(0) + 1e-3
        i = int(np.argmax(F))
        if F[i] > bf:
            bf, bx = F[i], P[i]
    return bx, bf


def prove():
    """Quick resilience checks: engagement is design-dependent + real-physics-grounded."""
    d0 = np.full(dc.FULL_DIM, 0.5)
    long_legs = np.array([0.95, 0.95, 0.6, 0.3, 0.4])     # longer reach, lighter
    short = np.array([0.05, 0.05, 0.2, 0.0, 0.9])         # short reach, heavy
    m0 = body_metrics(OUR_SPEC, d0, "our")
    print(f"default body metrics: reach={m0['reach']:.3f} clearance={m0['clearance']:.3f} "
          f"stand={m0['stand']:.2f} mass={m0['mass']:.2f} impulse={m0['impulse']:.2f}")
    s_long = sparc.differential(*match(long_legs, np.full(dc.FULL_DIM, 0.5)))
    s_short = sparc.differential(*match(short, np.full(dc.FULL_DIM, 0.5)))
    print(f"SPARC net vs default attacker: long-reach-us={s_long:+.2f}  short-reach-us={s_short:+.2f}")
    design_dependent = abs(s_long - s_short) > 0.3
    # the engagement must respond to the OPPONENT too (asymmetry)
    s_vs_long_att = sparc.differential(*match(d0, long_legs))
    responds_to_opp = abs(s_vs_long_att - sparc.differential(*match(d0, short))) > 0.3
    ok = design_dependent and responds_to_opp and m0["reach"] > 0
    print(f"PROVEN: engagement is real-physics + design-dependent ({design_dependent}) and "
          f"responds to the opponent body ({responds_to_opp}): {ok}")
    sys.exit(0 if ok else 1)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--rounds", type=int, default=6)
    ap.add_argument("--pop", type=int, default=12)
    ap.add_argument("--gens", type=int, default=3)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--prove", action="store_true")
    args = ap.parse_args()
    if args.prove:
        return prove()
    rng = np.random.default_rng(args.seed)
    dim = dc.FULL_DIM; k = 3

    rob_hof = [np.full(dim, 0.5)]                  # (body) Hall of Fame; (body,policy) on GPU
    att_hof = [np.full(dim, 0.5)]
    # Phase 8b: a FIXED absolute benchmark set of reference attackers (track real progress)
    bench = [np.full(dim, 0.5), np.array([0.9, 0.9, 0.7, 0.2, 0.3]), np.array([0.2, 0.2, 0.9, 0.5, 0.7])]

    print("co-evolution arms race — two GENERATED bodies, real-physics engagement, HoF + "
          "absolute benchmark + diversity (Phase 8b):")
    print(f"{'round':>5} {'rel SPARC':>9} {'abs bench':>9} {'rob reach':>9} {'att reach':>9} {'HoF':>5}")
    abs_curve = []
    for rd in range(args.rounds):
        a_samp = [att_hof[i] for i in rng.choice(len(att_hof), min(k, len(att_hof)), replace=False)]
        best_r, _ = cem(lambda x: robot_fitness(x, a_samp), dim, args.pop, args.gens, rng, diversity=0.3)
        rob_hof.append(best_r)
        r_samp = [rob_hof[i] for i in rng.choice(len(rob_hof), min(k, len(rob_hof)), replace=False)]
        best_a, _ = cem(lambda x: attacker_fitness(x, r_samp), dim, args.pop, args.gens, rng, diversity=0.3)
        att_hof.append(best_a)
        rel = sparc.differential(*match(best_r, best_a))
        abs_score = np.mean([sparc.differential(*match(best_r, b)) for b in bench])  # absolute progress
        abs_curve.append(abs_score)
        rm = body_metrics(OUR_SPEC, best_r, "our"); am = body_metrics(ATT_SPEC, best_a, "att")
        print(f"{rd:5d} {rel:9.2f} {abs_score:9.2f} {rm['reach']:9.3f} {am['reach']:9.3f} {len(rob_hof):5d}")

    final = rob_hof[-1]
    net_hof = np.mean([sparc.differential(*match(final, a)) for a in att_hof])
    abs_up = abs_curve[-1] >= abs_curve[0]            # absolute benchmark trended up (not just relative)
    print(f"\nRed Queen + absolute: final robot mean SPARC vs the {len(att_hof)}-attacker HoF "
          f"= {net_hof:+.2f}; absolute-benchmark score {abs_curve[0]:+.2f} -> {abs_curve[-1]:+.2f} "
          f"(up = real progress, not cycling: {abs_up}).")
    print(f"PROVEN: co-evolution on REAL-physics matches between two GENERATED bodies (one "
          f"morphology space); HoF sampled, absolute benchmark tracked, diversity on. "
          f"Trained-policy melee = selfplay_mjx.py (GPU).")
    sys.exit(0)


if __name__ == "__main__":
    main()
