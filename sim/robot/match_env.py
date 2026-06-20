# SPDX-License-Identifier: MIT
"""Self-play match: two robots fight, scored by SPARC (win the decision).

Our robot (A) and the attacker (B) are BOTH full controllable bodies in one scene,
each with a weapon-leg. Damage is a weapon-geom touching the opponent's body; the
per-step reward and the final match score are the SPARC differential (sparc_score)
- our points minus the opponent's - so each policy is trained to WIN, not survive.
This pairs with coevolve.py: the bodies come from the co-evolution, the policies
from self-play on those bodies.

Locally `--prove` validates the match mechanics (scene, weapon->body damage
classification, closing/aggression, SPARC scoring + winner). `selfplay()` is the
GPU-target PPO self-play skeleton (run on a CUDA/MJX box).
  python match_env.py --prove
"""

from __future__ import annotations

import argparse
import copy
import os
import sys
from pathlib import Path

os.environ.setdefault("MUJOCO_GL", "osmesa")

import mujoco  # noqa: E402
import numpy as np  # noqa: E402

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from gen_robot_mjcf import build_match, load_spec  # noqa: E402
import sparc_score as sparc  # noqa: E402

SPEC = load_spec(HERE / "robot.toml")
DAMAGE_REF, RAM_REF = 150.0, 150.0      # N: scale impact force -> [0,1] severity (SPARC tiers)


def weapon_spec(base=SPEC):
    """A body with a weapon-leg (the spear) added - both fighters get one."""
    s = copy.deepcopy(base)
    s["leg"] = base["leg"] + [{"name": "WP", "pos": [0.24, 0.0, 0.0], "is_weapon": True}]
    return s


class MatchEnv:
    def __init__(self, spec_a, spec_b, frame_skip=5, max_steps=400, seed=0, sep=2.4):
        self.model = mujoco.MjModel.from_xml_string(build_match(spec_a, spec_b, sep))
        self.data = mujoco.MjData(self.model)
        self.fs, self.max_steps = frame_skip, max_steps
        self.rng = np.random.default_rng(seed)
        m = self.model
        gname = lambda g: mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_GEOM, g) or ""
        self.floor = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_GEOM, "floor")
        self.weapon, self.body = {}, {}
        for s in ("A", "B"):
            gs = [g for g in range(m.ngeom) if gname(g).startswith(s + "_")]
            self.weapon[s] = {g for g in gs if gname(g).endswith("_spear")}
            self.body[s] = {g for g in gs if not gname(g).endswith("_spear")}
        aname = lambda a: mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_ACTUATOR, a) or ""
        self.act = {s: [a for a in range(m.nu) if aname(a).startswith(s + "_")]
                    for s in ("A", "B")}
        self.jadr, self.root_q, self.root_d, self.torso = {}, {}, {}, {}
        for s in ("A", "B"):
            self.jadr[s] = [(m.jnt_qposadr[int(m.actuator_trnid[a, 0])],
                             m.jnt_dofadr[int(m.actuator_trnid[a, 0])]) for a in self.act[s]]
            rj = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_JOINT, s + "_root")
            self.root_q[s], self.root_d[s] = m.jnt_qposadr[rj], m.jnt_dofadr[rj]
            self.torso[s] = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, s + "_torso")
        self.reset()

    def reset(self, seed=None):
        if seed is not None:
            self.rng = np.random.default_rng(seed)
        mujoco.mj_resetData(self.model, self.data)
        for s in ("A", "B"):
            self.data.qpos[self.root_q[s] + 2] = SPEC["torso"]["spawn_height"]
        mujoco.mj_forward(self.model, self.data)
        self.tally = dict(dealt_A=0.0, dealt_B=0.0, close_A=0.0, flee_A=0.0,
                          close_B=0.0, flee_B=0.0, ram_A=0.0, ram_B=0.0, steps=0)
        return self.obs("A", "B"), self.obs("B", "A")

    def _contact_force(self, c):
        f = np.zeros(6)
        mujoco.mj_contactForce(self.model, self.data, c, f)
        return float(abs(f[0]))                              # normal force magnitude (N)

    def _classify(self, g1, g2):
        """(A-deals, B-deals) for one contact: a weapon touching the OPPONENT body."""
        a = ((g1 in self.weapon["A"] and g2 in self.body["B"]) or
             (g2 in self.weapon["A"] and g1 in self.body["B"]))
        b = ((g1 in self.weapon["B"] and g2 in self.body["A"]) or
             (g2 in self.weapon["B"] and g1 in self.body["A"]))
        return float(a), float(b)

    def _damage(self):
        """Force-weighted contact: weapon->opponent-body = damage (impact force, so a
        hard thrust >> a glancing touch, matching SPARC severity tiers); body<->body =
        ram force (a control/aggression signal - shoving the opponent around)."""
        a_dmg = b_dmg = ram = 0.0
        for c in range(self.data.ncon):
            g1, g2 = self.data.contact[c].geom1, self.data.contact[c].geom2
            x, y = self._classify(g1, g2)
            if x or y:
                f = self._contact_force(c)
                a_dmg += f * x; b_dmg += f * y
            elif (g1 in self.body["A"] and g2 in self.body["B"]) or \
                 (g2 in self.body["A"] and g1 in self.body["B"]):
                ram += self._contact_force(c)                # robot-to-robot ram
        return a_dmg, b_dmg, ram

    def _closing(self, s, opp):
        rel = self.data.xpos[self.torso[opp]][:2] - self.data.xpos[self.torso[s]][:2]
        n = np.linalg.norm(rel) + 1e-9
        v = self.data.qvel[self.root_d[s]:self.root_d[s] + 2]
        toward = float(v @ rel) / n
        return max(0.0, toward) / 2.0, max(0.0, -toward) / 2.0   # closing, fleeing (norm)

    def obs(self, s, opp):
        d = self.data
        jq = np.array([d.qpos[qa] for qa, _ in self.jadr[s]])
        jv = np.array([d.qvel[da] for _, da in self.jadr[s]])
        rd = self.root_d[s]
        rel = d.xpos[self.torso[opp]] - d.xpos[self.torso[s]]
        return np.concatenate([jq, jv, d.xquat[self.torso[s]], d.qvel[rd:rd + 6],
                               [d.xpos[self.torso[s]][2]], rel,
                               d.qvel[self.root_d[opp]:self.root_d[opp] + 3]]).astype(np.float32)

    def step(self, act_a, act_b):
        for a, v in zip(self.act["A"], act_a):
            self.data.ctrl[a] = np.clip(v, -1, 1)
        for a, v in zip(self.act["B"], act_b):
            self.data.ctrl[a] = np.clip(v, -1, 1)
        da = db = ram = 0.0
        for _ in range(self.fs):
            mujoco.mj_step(self.model, self.data)
            x, y, r = self._damage(); da += x; db += y; ram += r
        ca, fa = self._closing("A", "B"); cb, fb = self._closing("B", "A")
        a_ram, b_ram = (ram, 0.0) if ca >= cb else (0.0, ram)   # ram credit to who drives in
        t = self.tally
        t["dealt_A"] += da; t["dealt_B"] += db; t["steps"] += 1
        t["close_A"] += ca; t["flee_A"] += fa; t["close_B"] += cb; t["flee_B"] += fb
        t["ram_A"] += a_ram; t["ram_B"] += b_ram
        # damage scaled by impact force (severity); control credited to the rammer
        r_a = sparc.step_reward(dealt=min(da / DAMAGE_REF, 1), taken=min(db / DAMAGE_REF, 1),
                                closing=min(ca, 1), fleeing=min(fa, 1), control=min(a_ram / RAM_REF, 1))
        r_b = sparc.step_reward(dealt=min(db / DAMAGE_REF, 1), taken=min(da / DAMAGE_REF, 1),
                                closing=min(cb, 1), fleeing=min(fb, 1), control=min(b_ram / RAM_REF, 1))
        za, zb = self.data.xpos[self.torso["A"]][2], self.data.xpos[self.torso["B"]][2]
        done = bool(t["steps"] >= self.max_steps or za < 0.1 or zb < 0.1
                    or not np.isfinite(self.data.qpos).all())
        return (self.obs("A", "B"), self.obs("B", "A")), (r_a, r_b), done

    def score(self):
        """Final SPARC points for each side from the match tally -> winner. Damage and
        control are force-graded (cumulative impact + ram force, relative)."""
        t = self.tally
        ramtot = t["ram_A"] + t["ram_B"]
        feats = lambda me, opp: dict(
            damage=sparc.damage_fraction(t[f"dealt_{me}"], t[f"dealt_{opp}"]),
            control=(t[f"ram_{me}"] / ramtot) if ramtot > 1e-9 else 0.5,
            aggression=sparc.aggression_fraction(t[f"close_{me}"], t[f"flee_{me}"], t["steps"]))
        fa, fb = feats("A", "B"), feats("B", "A")
        return sparc.points(**fa), sparc.points(**fb)


def run_match(env, pol_a, pol_b, max_steps=400):
    (oa, ob) = env.reset()
    done = False
    while not done:
        (oa, ob), _, done = env.step(pol_a(oa, env, "A"), pol_b(ob, env, "B"))
    pa, pb = env.score()
    return pa, pb, ("A" if pa > pb else "B" if pb > pa else "tie")


# scripted controllers (for the proof; the real ones are trained policies)
def random_ctrl(o, env, s):
    return env.rng.uniform(-1, 1, len(env.act[s]))


def selfplay():
    """GPU self-play (sketch): two PPO policies on the evolved bodies, scored by the
    SPARC differential, trading off against each other (and the coevolve Hall of
    Fame as a league). Run on a CUDA/MJX box."""
    try:
        import jax  # noqa: F401
        from mujoco import mjx  # noqa: F401
    except Exception:
        print("JAX/MJX absent - this is the GPU-target self-play skeleton.")
        print("On a CUDA box: vmap MatchEnv over N envs in MJX, train pol_A and pol_B")
        print("with PPO on sparc_score.step_reward, league vs coevolve.py's Hall of Fame.")
        return
    raise SystemExit("MJX present: wire the two-policy PPO self-play loop here.")


def prove():
    env = MatchEnv(weapon_spec(), weapon_spec(), seed=0)
    print(f"match scene: A act={len(env.act['A'])}, B act={len(env.act['B'])}, "
          f"A weapon geoms={len(env.weapon['A'])}, A body geoms={len(env.body['A'])}")
    # 1. damage classification logic (synthetic geom pairs)
    aw = next(iter(env.weapon["A"])); bb = next(iter(env.body["B"]))
    bw = next(iter(env.weapon["B"])); ab = next(iter(env.body["A"]))
    c1 = env._classify(aw, bb); c2 = env._classify(bw, ab); c3 = env._classify(aw, bw)
    ok_dmg = c1 == (1.0, 0.0) and c2 == (0.0, 1.0) and c3 == (0.0, 0.0)
    print(f"damage classify: A-weapon->B-body={c1}, B-weapon->A-body={c2}, "
          f"weapon-vs-weapon={c3}  OK={ok_dmg}")
    # 2. closing/aggression: give A a shove toward B, confirm it reads as closing
    env.reset(seed=1)
    env.data.qvel[env.root_d["A"]] = 2.0           # +x velocity, B is at +x
    ca, fa = env._closing("A", "B")
    ok_close = ca > fa
    print(f"aggression: A pushed toward B -> closing={ca:.2f} > fleeing={fa:.2f}  OK={ok_close}")
    # 2b. force on robot-to-robot contact: overlap the torsos, confirm ram force reads
    env.reset(seed=2)
    env.data.qpos[env.root_q["A"]] = -0.15      # x
    env.data.qpos[env.root_q["B"]] = 0.15
    mujoco.mj_forward(env.model, env.data)
    mujoco.mj_step(env.model, env.data)
    _, _, ram = env._damage()
    ok_force = ram > 0.0
    print(f"contact force: torsos overlapped -> ram impact {ram:.1f} N measured "
          f"(damage now force-weighted, SPARC severity)  OK={ok_force}")
    # 3. full match runs + scores + declares a winner (random vs random)
    pa, pb, win = run_match(env, random_ctrl, random_ctrl)
    ok_match = np.isfinite(pa) and np.isfinite(pb)
    print(f"random match: A={pa:.2f} pts, B={pb:.2f} pts, winner={win}  OK={ok_match}")
    # 4. SPARC scoring sanity: a dominant-A tally beats B
    ok_score = sparc.differential(dict(damage=0.9, control=0.6, aggression=0.8),
                                  dict(damage=0.1, control=0.4, aggression=0.2)) > 0
    allok = ok_dmg and ok_close and ok_force and ok_match and ok_score
    print(f"\nPROVEN: two-robot self-play match - scene + force-weighted weapon->body "
          f"damage + robot-robot ram force + closing/aggression + SPARC scoring/winner "
          f"all work: {allok}. Swap random_ctrl -> trained policies on GPU (selfplay()).")
    sys.exit(0 if allok else 1)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--prove", action="store_true")
    args = ap.parse_args()
    prove() if args.prove else selfplay()


if __name__ == "__main__":
    main()
