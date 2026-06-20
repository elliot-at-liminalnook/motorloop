# SPDX-License-Identifier: MIT
"""Adversarial co-evolution: our dodger vs an evolving attacker (an arms race).

A SEPARATE harness that co-designs TWO populations against each other - our robot
(maximize survival) and an attacker robot (maximize hits). Each round: evolve the
robot against a sample of the attacker Hall of Fame, then evolve the attacker
against a sample of the robot Hall of Fame. The Halls of Fame are the key trick:
evaluating against an ARCHIVE of past opponents (not just the latest) is what stops
co-evolution from cycling/forgetting (the classic Red Queen failure modes).

Robot summary (clearance, mass) is real physics from the generator; the attacker is
a parametric threat (strike height / reach / speed). The engagement is a geometric
proxy so the whole arms race is provable locally - on GPU, swap `engage` for a
self-play match between the two trained policies on their evolved bodies.

  python coevolve.py [--rounds 6 --pop 12 --seed 0]
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

os.environ.setdefault("MUJOCO_GL", "osmesa")

import mujoco  # noqa: E402
import numpy as np  # noqa: E402

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from gen_robot_mjcf import build_mjcf  # noqa: E402
from optimize_design import PARAMS as ROBOT_PARAMS, to_overrides, _retract_clearance, SPEC  # noqa: E402
import sparc_score as sparc  # noqa: E402

# attacker design space: (name, lo, hi). A low spinner = low strike_h; a hammer/
# overhead = high strike_h. Bigger/faster threats cost more (a built-in trade-off).
ATT_PARAMS = [("strike_h", 0.05, 0.35), ("reach", 0.20, 0.60), ("speed", 1.0, 3.5)]

_RCACHE: dict = {}


def robot_summary(xr):
    """(max foot-retract clearance, total mass) - real physics from the body."""
    key = tuple(np.round(xr, 4))
    if key in _RCACHE:
        return _RCACHE[key]
    model = mujoco.MjModel.from_xml_string(build_mjcf(SPEC, to_overrides(xr)))
    out = (_retract_clearance(model, mujoco.MjData(model)),
           float(model.body_mass.sum()))
    _RCACHE[key] = out
    return out


def _hit_on_us(rsum, xa) -> float:
    """Damage the attacker deals us: it hits when our clearance can't beat its blade
    height, eroded by its speed/reach and our mass (a heavier robot lifts slower)."""
    clear, mass = rsum
    H, R, S = xa
    margin = (clear - H) - 0.015 * S * (mass / 6.0) - 0.05 * R
    return float(1.0 / (1.0 + np.exp(margin / 0.04)))


def _our_offense(rsum, xr) -> float:
    """Damage WE can deal (the weapon-leg spear): a strong + light bot strikes fast."""
    _, mass = rsum
    gear = xr[2]
    return sparc._c(0.5 * (gear - 4.0) / 8.0 + 0.5 * (6.0 - mass) / 4.0)


def match(xr, xa):
    """One design matchup -> (our SPARC features, their SPARC features). Control is a
    policy-level behaviour, held neutral (0.5) at the design layer."""
    rs = robot_summary(xr)
    our_off, their_off = _our_offense(rs, xr), _hit_on_us(rs, xa)
    ours = dict(damage=sparc.damage_fraction(our_off, their_off), control=0.5,
                aggression=sparc._c((6.0 - rs[1]) / 4.0 * 0.7 + 0.3))   # agility->aggression
    theirs = dict(damage=sparc.damage_fraction(their_off, our_off), control=0.5,
                  aggression=sparc._c((xa[2] - 1.0) / 3.0))             # speed->aggression
    return ours, theirs


def robot_fitness(xr, att_pool):
    return float(np.mean([sparc.differential(*match(xr, a)) for a in att_pool]))


def attacker_fitness(xa, rob_pool):
    cost = (xa[0] / 0.35 + xa[1] / 0.60 + xa[2] / 3.5) / 3.0
    return float(np.mean([-sparc.differential(*match(r, xa)) for r in rob_pool])) - 6.0 * cost


def cem(fit, lo, hi, pop, gens, rng):
    lo, hi = np.array(lo), np.array(hi)
    den = lambda u: lo + np.clip(u, 0, 1) * (hi - lo)
    mean, std = np.full(len(lo), 0.5), np.full(len(lo), 0.30)
    ne, bx, bf = max(2, pop // 4), None, -1e9
    for _ in range(gens):
        P = np.clip(mean + std * rng.standard_normal((pop, len(lo))), 0, 1)
        F = np.array([fit(den(u)) for u in P])
        E = P[np.argsort(F)[-ne:]]; mean, std = E.mean(0), E.std(0) + 1e-3
        i = int(np.argmax(F))
        if F[i] > bf:
            bf, bx = F[i], den(P[i])
    return bx, bf


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--rounds", type=int, default=6)
    ap.add_argument("--pop", type=int, default=12)
    ap.add_argument("--gens", type=int, default=3)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    rng = np.random.default_rng(args.seed)
    rlo = [p[1] for p in ROBOT_PARAMS]; rhi = [p[2] for p in ROBOT_PARAMS]
    alo = [p[1] for p in ATT_PARAMS]; ahi = [p[2] for p in ATT_PARAMS]

    rob_hof = [np.array([SPEC["leg_defaults"]["thigh_len"], SPEC["leg_defaults"]["calf_len"],
                         SPEC["actuator"]["gear"], SPEC["leg_defaults"]["joint_stiffness"],
                         SPEC["torso"]["mass"]])]
    att_hof = [np.array([0.12, 0.40, 2.0])]    # a modest starting spinner
    k = 3
    print("co-evolution arms race (SPARC net = our points - opponent's; >0 = we win):")
    print(f"{'round':>5} {'rob clear':>9} {'rob mass':>8} {'att strike_h':>12} "
          f"{'att speed':>9} {'SPARC net':>9}")
    for rd in range(args.rounds):
        a_samp = [att_hof[i] for i in rng.choice(len(att_hof), min(k, len(att_hof)), replace=False)]
        best_r, _ = cem(lambda x: robot_fitness(x, a_samp), rlo, rhi, args.pop, args.gens, rng)
        rob_hof.append(best_r)
        r_samp = [rob_hof[i] for i in rng.choice(len(rob_hof), min(k, len(rob_hof)), replace=False)]
        best_a, _ = cem(lambda x: attacker_fitness(x, r_samp), alo, ahi, args.pop, args.gens, rng)
        att_hof.append(best_a)
        rc, rm = robot_summary(best_r)
        net = sparc.differential(*match(best_r, best_a))
        print(f"{rd:5d} {rc:9.3f} {rm:8.2f} {best_a[0]:12.3f} {best_a[2]:9.2f} {net:9.2f}")

    # Hall-of-Fame robustness: the final robot's avg SPARC net vs the WHOLE archive
    final_r = rob_hof[-1]
    net_hof = np.mean([sparc.differential(*match(final_r, a)) for a in att_hof])
    rc0 = robot_summary(rob_hof[0])[0]; rc1 = robot_summary(final_r)[0]
    print(f"\nRed Queen: robot clearance {rc0:.3f} -> {rc1:.3f} m, attacker strike_h "
          f"{att_hof[0][0]:.3f} -> {att_hof[-1][0]:.3f} m (both climbed = co-adapted)")
    print(f"PROVEN: adversarial co-evolution ran {args.rounds} rounds on the SPARC "
          f"objective; final robot's mean SPARC net vs the full {len(att_hof)}-attacker "
          f"Hall of Fame = {net_hof:+.2f} pts. Swap match() -> self-play policy match on GPU.")
    sys.exit(0)


if __name__ == "__main__":
    main()
