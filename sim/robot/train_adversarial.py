# SPDX-License-Identifier: MIT
"""Stage B — adversarial, design-conditioned policy (GPU), warm-started from the
locomotion universal policy (Stage A). The skill-ladder our combat-dodge work proved:
learn to move the body first, THEN learn to fight, so we don't relearn balance under
adversarial pressure (which collapses).

Same default body as Stage A (warm-start-compatible), TWO robots in one scene,
LEGS-AS-WEAPONS (a leg/foot contacting the opponent's body = penetration-weighted
damage). Obs = [Stage-A locomotion obs (38)] + [opponent rel pos/vel (6)] so the
first 38 dims match Stage A; warm-start pads the input layer 38->44 (opponent inputs
init ~0 -> starts as the locomotor). Reward = SPARC (damage dealt - taken +
aggression) + a small locomotion anchor (don't forget to stand). B is passive here
(single-agent attack); self-play league = the stretch.

  python train_adversarial.py [--steps 12000000 --resume <loco_ckpt>]
"""

from __future__ import annotations

import argparse, copy, os, pickle, sys, time
from pathlib import Path
import jax, jax.numpy as jnp, numpy as np, mujoco
from mujoco import mjx
from brax.envs.base import Env, State
from brax.training.agents.ppo import train as ppo

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from gen_robot_mjcf import build_match, load_spec  # noqa: E402
try:
    from arena import kernel_emit as _ke           # universal trace emit (no-op if ARENA_SINK unset)
except Exception:                                  # arena absent ⇒ standalone kernel, unchanged
    class _ke:                                     # noqa: E301
        emit_metric = emit_event = emit_error = staticmethod(lambda *a, **k: None)

SPEC = load_spec(HERE / "robot.toml")
OUT = Path(os.environ.get("CODESIGN_OUT", "/root/proj/out")); OUT.mkdir(parents=True, exist_ok=True)
LOCO_OBS = 38; DAMAGE_REF = 0.05
STRIKE_KINETIC = 0.1  # rod damage multiplier per m/s of slide speed: hit at ~11 m/s ≈ ×2.1 damage


def METRIC(**kw):
    print("METRIC " + " ".join(f"{k}={v}" for k, v in kw.items()), flush=True)
    _ke.emit_metric(str(kw.get("stage", "metric")), **{k: v for k, v in kw.items() if k != "stage"})


class AdversarialEnv(Env):
    def __init__(self, frame_skip=5, shaping=1.0, sep=1.0, self_collision=True,
                 sep_lo=None, sep_hi=None, approach_weight=0.0, azimuth=0.0,
                 reality_gap=False, n_worlds=64,
                 clean_weight=0.0, trade_weight=0.0, disengage_weight=0.0, striker=None,
                 fire_shaping=0.0, rod_reach=0.30, opponent="passive", opp_infer=None, opp_params=None,
                 upright_weight=0.3, energy_penalty=0.0, airborne_penalty=0.0, height_weight=0.0,
                 move_weight=0.0, early_hit_penalty=0.0, min_hit_step=0, taken_weight=0.0,
                 flee_penalty=0.0, close_bonus=0.0, close_radius=0.45, damage_bonus=0.0,
                 face_opponent=False, engage_obs=False, contact_obs=False, face_weight=0.0,
                 penetration_penalty=0.0, penetration_tol=0.045,
                 reset_bank_seed=None, reset_bank_epis=0):
        # REAL-ROBOT competency knobs (the Coach drives these, like it drives clean/fire):
        #   upright_weight — BALANCE/survival anchor (was a fixed 0.3); raise it when the fighter
        #     FALLS (survival verdict lagging) — a battlebot that lands a hit but topples loses.
        #   energy_penalty — ENERGY/actuator-safety: penalize hinge effort; raise it when the policy
        #     SLAMS the actuators (safe verdict lagging) — ties to the real motor/torque envelope.
        self._upright_w = float(upright_weight); self._energy_w = float(energy_penalty)
        self._airborne_w = float(airborne_penalty)        # anti-cheat: discourage jump-to-strike
        self._height_w = float(height_weight)             # reward standing TALL (vs a low sprawl)
        self._move_w = float(move_weight)                 # LOCOMOTION pretrain: reward torso planar SPEED
        self._early_hit_penalty = float(early_hit_penalty)
        self._min_hit_step = float(min_hit_step)
        self._taken_w = float(taken_weight)
        self._flee_w = float(flee_penalty)
        self._close_bonus_w = float(close_bonus)
        self._close_radius = float(close_radius)
        self._damage_bonus_w = float(damage_bonus)
        self._face_w = float(face_weight)
        self._penalty_w = float(penetration_penalty)
        self._penalty_tol = float(penetration_tol)
        self._reset_keys = None
        if reset_bank_seed is not None and int(reset_bank_epis) > 0:
            self._reset_keys = jax.random.split(
                jax.random.PRNGKey(int(reset_bank_seed)),
                int(reset_bank_epis),
            )
        self._face_opponent = bool(face_opponent)
        self._engage_obs = bool(engage_obs)
        self._contact_obs = bool(contact_obs)
        spawn_z = float(SPEC.get("torso", {}).get("spawn_height", 0.35))
        self._airborne_z = spawn_z + 0.07
        self._grounded_z = spawn_z + 0.03
        #   (learn to WALK this body before fighting). A loco stage sets move high + combat off so the
        #   fighter inherits real mobility instead of parking in a stable crouch. Same env/obs => the
        #   combat stages warm-start from it seamlessly (no cross-env obs mismatch).
        # OPPONENT (B): "passive" (skill curriculum — B limp, default, byte-identical) | "frozen"
        # (B driven by a FROZEN policy snapshot of OUR fighter — self-play, makes `taken` truly
        # adversarial). A frozen opponent is armed too (striker_b) so it can strike back. The
        # opponent sees a MIRRORED obs (B-centric, same layout as A's) so an A-snapshot plugs in.
        self._opp = str(opponent); self._opp_infer = opp_infer; self._opp_params = opp_params
        self._armed_b = self._opp != "passive"
        # WIN-EXCHANGES reward asymmetry (the dealt≈taken "trading" fix — gives headroom ABOVE
        # the trading plateau so the curve keeps rising with more training):
        #   clean_weight    + w·dealt·(1−taken)  — reward hits landed WHILE NOT being hit.
        #   trade_weight     − w·min(dealt,taken) — punish mutual contact (trading blows).
        #   disengage_weight + w·prev_dealt·outward_vel — reward retreating right AFTER a hit
        #                      (gated on having just dealt; ANNEAL so it can't become fleeing).
        # All default 0 ⇒ byte-identical to the contact-forced fighter's reward when unset.
        self._clean_w = float(clean_weight); self._trade_w = float(trade_weight)
        self._dis_w = float(disengage_weight)
        # shaping = weight on the dense close→strike potential; sep = base start separation.
        # CURRICULUM: each reset samples the A–B start separation from [sep_lo, sep_hi]
        # (default = fixed `sep`). A close low end GUARANTEES some envs spawn in striking
        # range every batch (so the dealt reward signal always exists); widening the high end
        # over phases teaches closing. This is the reverse/contact-forcing curriculum.
        # approach_weight: potential-based APPROACH reward = w·(dist_{t-1} − dist_t) — pays for
        # CLOSING distance to the opponent each step (approach velocity). Dense + always
        # available before any hit lands, so it "forces" learning to close the gap; a velocity
        # curriculum step (high early when the policy can't reach, anneal as strikes take over).
        self._approach_w = float(approach_weight)
        self._shaping = float(shaping); self._sep = float(sep)
        self._sep_lo = float(sep_lo if sep_lo is not None else sep)
        self._sep_hi = float(sep_hi if sep_hi is not None else sep)
        self._azimuth = float(azimuth)        # ±range (rad) of opponent BEARING — varied attack angles
        m = mujoco.MjModel.from_xml_string(
            build_match(SPEC, SPEC, sep, self_collision=self_collision, striker=striker,
                        striker_b=self._armed_b))
        self._mj_model = m
        self._geom_names = [
            mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_GEOM, g) or f"geom_{g}"
            for g in range(m.ngeom)
        ]
        self._body_names = [
            mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_BODY, b) or f"body_{b}"
            for b in range(m.nbody)
        ]
        self._geom_body = np.asarray(m.geom_bodyid, dtype=np.int32)
        self._mx = mjx.put_model(m); self._fs = frame_skip; self._nu = m.nu
        self._q0 = jnp.array(m.qpos0)
        _ld = SPEC.get("leg_defaults", {})
        _stand_abd = float(_ld.get("stand_abd", 0.0))
        _stand_flex = float(_ld.get("stand_flex", -0.4))
        _stand_knee = float(_ld.get("stand_knee", -1.1))
        # BENT default leg pose so feet rest near the floor at spawn. The straight default (knee
        # clamped to -0.4) spawned the body ~0.2 m UNDERGROUND → the contact solver catapulted the
        # torso to 1.5 m (the launch exploit anti_cheat.py caught). Bent knees → ~0.04 m clearance.
        for _j in range(m.njnt):
            _nm = mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_JOINT, _j) or ""
            _a = int(m.jnt_qposadr[_j])
            if _nm.endswith("_knee"):
                self._q0 = self._q0.at[_a].set(_stand_knee)
            elif _nm.endswith("_flex"):
                self._q0 = self._q0.at[_a].set(_stand_flex)
            elif _nm.endswith("_abd"):
                self._q0 = self._q0.at[_a].set(_stand_abd)
        # PNEUMATIC striker params (kinetic damage scale + per-fire energy cost). The rod is a
        # separate striking mask: rod damage scales with the slide-joint SPEED (a fast strike hits
        # harder); firing costs reward (air/energy) so the policy fires only to connect.
        _ss = SPEC.get("striker", {})
        self._fire_cost = float(_ss.get("fire_cost", 0.0)); self._kin = STRIKE_KINETIC
        # firing-SHAPING: dense reward for firing a rod WHEN its tip is aimed/in-range at B — the
        # firing analog of leg-proximity shaping that cracked dealt=0 (without it the fire_cost
        # alone teaches "never fire"). rod_reach = the distance within which a fire is "good".
        self._fire_shaping = float(fire_shaping); self._rod_reach = float(rod_reach)
        gn = lambda g: mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_GEOM, g) or ""
        an = lambda a: mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_ACTUATOR, a) or ""
        mk = lambda p: jnp.array([p(gn(g)) for g in range(m.ngeom)])
        leg_geom = lambda n, s: n.startswith(s + "_") and (
            n.endswith("_hipg") or n.endswith("_thighg") or
            n.endswith("_calfg") or n.endswith("_foot") or n.endswith("_spear"))
        leg_weapon = lambda n, s: n.startswith(s + "_") and (n.endswith("_foot") or n.endswith("_spear"))
        leg_target = lambda n, s: (
            n == s + "_torso" or
            (n.startswith(s + "_") and (
                n.endswith("_hipg") or n.endswith("_thighg") or n.endswith("_calfg"))))
        # legs-as-weapons: foot/spear geoms damage the opponent's torso/upper-leg targets. Do not
        # count foot/rod weapon-on-weapon binds as damage to both sides; that made every clash a trade.
        # The pneumatic ROD is a SEPARATE weapon mask (its damage gets the tip-speed multiplier).
        self._Aleg = mk(lambda n: leg_weapon(n, "A"))
        self._Bleg = mk(lambda n: leg_weapon(n, "B"))
        self._Arod = mk(lambda n: n.startswith("A_") and n.endswith("_rod"))
        self._Brod = mk(lambda n: n.startswith("B_") and n.endswith("_rod"))
        self._Abody = mk(lambda n: leg_target(n, "A"))
        self._Bbody = mk(lambda n: leg_target(n, "B"))
        self._Arod_gids = jnp.array([g for g in range(m.ngeom)
                                     if gn(g).startswith("A_") and gn(g).endswith("_rod")], dtype=int)
        self._Brod_gids = jnp.array([g for g in range(m.ngeom)
                                     if gn(g).startswith("B_") and gn(g).endswith("_rod")], dtype=int)
        # action = ALL A actuators (hinge motors THEN pneumatic strikes); obs/back-EMF use the
        # HINGE actuators only (the slide DOFs are excluded so LOCO_OBS stays 38).
        self._actA = jnp.array([a for a in range(m.nu) if an(a).startswith("A_")])
        self._nuA = int(self._actA.shape[0])
        A_acts = [a for a in range(m.nu) if an(a).startswith("A_")]
        A_hinge = [a for a in A_acts if not an(a).endswith("_strike_m")]
        A_strike = [a for a in A_acts if an(a).endswith("_strike_m")]
        self._n_hinge = len(A_hinge); self._has_striker = len(A_strike) > 0
        # local positions of the strike actuators WITHIN the A action vector (for the firing cost)
        self._strike_local = jnp.array([A_acts.index(a) for a in A_strike], dtype=int)
        Aj = [int(m.actuator_trnid[a, 0]) for a in A_hinge]                      # hinge joints only
        self._Aqa = jnp.array([int(m.jnt_qposadr[j]) for j in Aj])
        self._Ada = jnp.array([int(m.jnt_dofadr[j]) for j in Aj])
        # A's slide (strike) DOF addresses — the rod tip speed used to scale kinetic damage
        self._strike_dofs = jnp.array([int(m.jnt_dofadr[m.actuator_trnid[a, 0]]) for a in A_strike], dtype=int)
        # B (opponent) actuator/joint indices — a MIRROR of A; only used when opponent != passive
        # (a frozen A-snapshot drives B → needs B's hinge joints for the mirrored obs + B's
        # actuators/strike split for the control scatter).
        B_acts = [a for a in range(m.nu) if an(a).startswith("B_")]
        self._actB = jnp.array(B_acts, dtype=int); self._nuB = len(B_acts)
        B_hinge = [a for a in B_acts if not an(a).endswith("_strike_m")]
        B_strike = [a for a in B_acts if an(a).endswith("_strike_m")]
        self._has_striker_b = len(B_strike) > 0
        self._n_hinge_b = len(B_hinge)
        self._B_strike_local = jnp.array([B_acts.index(a) for a in B_strike], dtype=int)
        Bj = [int(m.actuator_trnid[a, 0]) for a in B_hinge]
        self._Bqa = jnp.array([int(m.jnt_qposadr[j]) for j in Bj], dtype=int)
        self._Bda = jnp.array([int(m.jnt_dofadr[j]) for j in Bj], dtype=int)
        self._strike_dofs_b = jnp.array([int(m.jnt_dofadr[m.actuator_trnid[a, 0]]) for a in B_strike], dtype=int)
        self._ArD = int(m.jnt_dofadr[mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_JOINT, "A_root")])
        self._BrD = int(m.jnt_dofadr[mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_JOINT, "B_root")])
        self._Arq = int(m.jnt_qposadr[mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_JOINT, "A_root")])
        self._Brq = int(m.jnt_qposadr[mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_JOINT, "B_root")])
        self._At = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, "A_torso")
        self._Bt = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, "B_torso")
        # A's striking bodies (calf/foot of each leg) — for the "aim a limb at B" shaping
        bn = lambda b: mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_BODY, b) or ""
        _strike = [b for b in range(m.nbody) if bn(b).startswith("A_")
                   and (bn(b).endswith("_calf") or bn(b).endswith("_foot"))]
        self._Astrike = jnp.array(_strike if _strike else [self._At])
        _strike_b = [b for b in range(m.nbody) if bn(b).startswith("B_")
                     and (bn(b).endswith("_calf") or bn(b).endswith("_foot"))]
        self._Bstrike = jnp.array(_strike_b if _strike_b else [self._Bt])
        self._hinge = jnp.array(m.jnt_type == mujoco.mjtJoint.mjJNT_HINGE)
        self._obs_size = LOCO_OBS + 6 + (8 if self._engage_obs else 0) + (8 if self._contact_obs else 0)
        # reality-gap: per-episode calibrated WORLD (actuator droop + DR), NOT in obs (the
        # policy must be robust to it). Mirrors UniversalEnv; used by the decisive combat
        # body-ranking experiment (nominal=rg off, robust=CVaR over worlds). obs unchanged.
        self._rg = bool(reality_gap); self._nworlds = int(n_worlds)
        if self._rg:
            from reality_gap import (sample_domain_params, default_uncertainty,
                                     actuator_scale, apply_to_mjx_model)
            unc = default_uncertainty()
            dps = [sample_domain_params(i, unc) for i in range(n_worlds)]
            self._bank = {k: jnp.asarray([float(dp[k]) for dp in dps])
                          for k in dps[0] if isinstance(dps[0][k], (int, float))}
            self._act_scale = staticmethod(actuator_scale).__func__
            self._apply_dp = staticmethod(apply_to_mjx_model).__func__

    @property
    def observation_size(self): return self._obs_size
    @property
    def action_size(self): return self._nuA
    @property
    def backend(self): return "mjx"

    def _design_model(self, d, dp=None):
        # one design codec (design_codec.apply_fast); hinge_mask keeps the spring off
        # the free-joint root. Identical maps to the universal env (worldbody mass 0).
        from design_codec import apply_fast
        mxd = apply_fast(self._mx, d, hinge_mask=self._hinge)
        if self._rg and dp is not None:       # calibrated DR on top of the design (no stiffness clash)
            mxd = self._apply_dp(mxd, dp, hinge_mask=None)
        return mxd

    def _world(self, rng):
        k = jax.random.randint(rng, (), 0, self._nworlds)
        return {f: self._bank[f][k] for f in self._bank}

    def _ctrl_scale(self, action, qvel, dp):
        # back-EMF torque-speed envelope on A's HINGE joints (the real motorloop droop). The
        # pneumatic strike dims (if any) are appended unscaled — gas force, not a back-EMF motor.
        if self._rg and dp is not None:
            scale = self._act_scale(qvel[self._Ada], dp)
            if self._has_striker:
                scale = jnp.concatenate([scale, jnp.ones(self._nuA - self._n_hinge)])
            return action * scale
        return action

    def _engage_tail(self, dx, me_t, opp_t, me_root_d, opp_root_d):
        rel = (dx.xpos[opp_t] - dx.xpos[me_t])[:2]
        dist = jnp.linalg.norm(rel)
        unit = rel / (dist + 1e-6)
        v_me = dx.qvel[me_root_d:me_root_d + 2]
        v_opp = dx.qvel[opp_root_d:opp_root_d + 2]
        radial = jnp.dot(v_me, unit)
        lateral = v_me[0] * (-unit[1]) + v_me[1] * unit[0]
        rel_radial = jnp.dot(v_me - v_opp, unit)
        clos = jnp.clip(radial / 2, 0, 1)
        flee = jnp.clip(-radial / 2, 0, 1)
        return jnp.array([dist, unit[0], unit[1], radial, lateral, rel_radial, clos, flee])

    def _contact_tail(self, dx, me_t, opp_t, rod_gids, strike_bodies):
        rel = (dx.xpos[opp_t] - dx.xpos[me_t])[:2]
        dist = jnp.linalg.norm(rel)
        unit = rel / (dist + 1e-6)
        rmat = dx.xmat[me_t].reshape(-1)
        forward = rmat[:2]
        side_axis = rmat[3:5]
        forward = forward / (jnp.linalg.norm(forward) + 1e-6)
        side_axis = side_axis / (jnp.linalg.norm(side_axis) + 1e-6)
        front = jnp.dot(forward, unit)
        side = jnp.dot(side_axis, unit)
        rod_d = jnp.linalg.norm(dx.geom_xpos[rod_gids] - dx.xpos[opp_t], axis=1)
        rod0 = rod_d[0] if rod_d.shape[0] > 0 else dist
        rod1 = rod_d[1] if rod_d.shape[0] > 1 else rod0
        min_rod = jnp.min(rod_d) if rod_d.shape[0] > 0 else dist
        limb_dist = jnp.min(jnp.linalg.norm(dx.xpos[strike_bodies] - dx.xpos[opp_t], axis=1))
        rod_close = jnp.clip((0.50 - min_rod) / 0.40, 0.0, 1.0)
        body_close = jnp.clip((0.45 - dist) / 0.30, 0.0, 1.0)
        return jnp.array([rod0, rod1, min_rod, limb_dist, front, side, rod_close, body_close])

    def _obs(self, dx, d):
        loco = jnp.concatenate([dx.qpos[self._Aqa], dx.qvel[self._Ada], dx.xquat[self._At],
                                dx.qvel[self._ArD:self._ArD + 6], dx.xpos[self._At][2:3], d])
        opp = jnp.concatenate([dx.xpos[self._Bt] - dx.xpos[self._At], dx.qvel[self._BrD:self._BrD + 3]])
        if self._engage_obs:
            opp = jnp.concatenate([opp, self._engage_tail(dx, self._At, self._Bt, self._ArD, self._BrD)])
        if self._contact_obs:
            opp = jnp.concatenate([opp, self._contact_tail(dx, self._At, self._Bt,
                                                           self._Arod_gids, self._Astrike)])
        return jnp.concatenate([loco, opp])

    def _obsB(self, dx, d):
        """Opponent's obs — the SAME layout as A's but B-centric (B=me, A=opponent), so a frozen
        snapshot of OUR fighter drives B unchanged (symmetric self-play)."""
        loco = jnp.concatenate([dx.qpos[self._Bqa], dx.qvel[self._Bda], dx.xquat[self._Bt],
                                dx.qvel[self._BrD:self._BrD + 6], dx.xpos[self._Bt][2:3], d])
        opp = jnp.concatenate([dx.xpos[self._At] - dx.xpos[self._Bt], dx.qvel[self._ArD:self._ArD + 3]])
        if self._engage_obs:
            opp = jnp.concatenate([opp, self._engage_tail(dx, self._Bt, self._At, self._BrD, self._ArD)])
        if self._contact_obs:
            opp = jnp.concatenate([opp, self._contact_tail(dx, self._Bt, self._At,
                                                           self._Brod_gids, self._Bstrike)])
        return jnp.concatenate([loco, opp])

    _MET0 = None
    def _metrics0(self):
        return {"dealt": jnp.zeros(()), "taken": jnp.zeros(()), "closing": jnp.zeros(()),
                "fleeing": jnp.zeros(()), "sparc": jnp.zeros(()), "dist": jnp.zeros(()),
                "approach": jnp.zeros(()), "close_term": jnp.zeros(()),
                "clean_hit": jnp.zeros(()), "trade": jnp.zeros(()),
                "disengage": jnp.zeros(()), "fire": jnp.zeros(()), "face": jnp.zeros(()),
                "penalty": jnp.zeros(())}

    def _planar_dist(self, dx):
        return jnp.linalg.norm((dx.xpos[self._Bt] - dx.xpos[self._At])[:2])

    def _yaw_quat(self, yaw):
        return jnp.array([jnp.cos(0.5 * yaw), 0.0, 0.0, jnp.sin(0.5 * yaw)])

    def _place(self, qpos, sep, theta):
        # A at origin; B at BEARING theta, distance sep -> the policy must approach + strike
        # from a varied angle (encourages different angles of attack, not just head-on).
        bx = sep * jnp.cos(theta); by = sep * jnp.sin(theta)
        qpos = (qpos.at[self._Arq].set(0.0).at[self._Arq + 1].set(0.0)
                    .at[self._Brq].set(bx).at[self._Brq + 1].set(by))
        if self._face_opponent:
            qpos = qpos.at[self._Arq + 3:self._Arq + 7].set(self._yaw_quat(theta))
            qpos = qpos.at[self._Brq + 3:self._Brq + 7].set(self._yaw_quat(theta + jnp.pi))
        return qpos

    def _info(self, d, dx, dp):
        # prev_dealt: damage dealt last step — gates the post-hit disengage bonus (only reward
        # retreating right AFTER landing a hit, not idle fleeing).
        info = {"design": d, "prev_dist": self._planar_dist(dx),
                "prev_dealt": jnp.zeros(()), "t": jnp.zeros(())}
        if self._rg:
            info["dp"] = dp
        return info

    def reset(self, rng):
        if self._reset_keys is not None:
            idx = jax.random.randint(rng, (), 0, self._reset_keys.shape[0])
            rng = self._reset_keys[idx]
        rng, dr, nr, sr, tr, wr = jax.random.split(rng, 6)
        d = jax.random.uniform(dr, (3,))
        dp = self._world(wr) if self._rg else None
        sep = jax.random.uniform(sr, (), minval=self._sep_lo, maxval=self._sep_hi)
        theta = jax.random.uniform(tr, (), minval=-self._azimuth, maxval=self._azimuth)
        qpos = self._q0.at[7:].add(jax.random.uniform(nr, (self._q0.shape[0] - 7,), minval=-0.03, maxval=0.03))
        qpos = self._place(qpos, sep, theta)
        dx = mjx.forward(self._design_model(d, dp), mjx.make_data(self._mx).replace(qpos=qpos))
        return State(dx, self._obs(dx, d), jnp.zeros(()), jnp.zeros(()), self._metrics0(), self._info(d, dx, dp))

    def reset_with(self, rng, design):
        """Reset with a GIVEN design (eval). With reality_gap on, draws a calibrated world too
        (a fresh PRNG key per call samples a different world -> CVaR over keys = robust score)."""
        nr, wr = jax.random.split(rng)
        dp = self._world(wr) if self._rg else None
        qpos = self._q0.at[7:].add(jax.random.uniform(nr, (self._q0.shape[0] - 7,), minval=-0.03, maxval=0.03))
        dx = mjx.forward(self._design_model(design, dp), mjx.make_data(self._mx).replace(qpos=qpos))
        return State(dx, self._obs(dx, design), jnp.zeros(()), jnp.zeros(()), self._metrics0(),
                     self._info(design, dx, dp))

    def step(self, state, action):
        d = state.info["design"]
        dp = state.info["dp"] if self._rg else None
        mxd = self._design_model(d, dp)
        clip_a = jnp.clip(action, -1, 1)
        a = self._ctrl_scale(clip_a, state.pipeline_state.qvel, dp)
        ctrl = jnp.zeros(self._nu).at[self._actA].set(a)
        if self._opp == "frozen" and self._opp_infer is not None:   # B driven by a frozen snapshot
            b_obs = self._obsB(state.pipeline_state, d)
            b_raw, _ = self._opp_infer(b_obs, jax.random.PRNGKey(0))  # deterministic ⇒ key unused
            ctrl = ctrl.at[self._actB].set(jnp.clip(b_raw, -1, 1))
        dx = state.pipeline_state.replace(ctrl=ctrl)
        dx = jax.lax.fori_loop(0, self._fs, lambda i, x: mjx.step(mxd, x), dx)
        pen = jnp.maximum(0.0, -dx.contact.dist); g0, g1 = dx.contact.geom[:, 0], dx.contact.geom[:, 1]
        peak_pen_step = jnp.max(pen)
        dealt = jnp.sum(pen * ((self._Aleg[g0] & self._Bbody[g1]) | (self._Aleg[g1] & self._Bbody[g0])))
        taken = jnp.sum(pen * ((self._Bleg[g0] & self._Abody[g1]) | (self._Bleg[g1] & self._Abody[g0])))
        # pneumatic ROD damage scales with tip (slide) SPEED — a fast strike hits harder. Rod
        # always deals its base penetration; firing it fast multiplies it. (No-striker body: the
        # rod masks are all-zero and _has_striker is False ⇒ this is a no-op, byte-identical.)
        fire_cost = 0.0; fire_aim = 0.0; fire_act = 0.0
        if self._has_striker:
            rod_dealt = jnp.sum(pen * ((self._Arod[g0] & self._Bbody[g1]) | (self._Arod[g1] & self._Bbody[g0])))
            rod_speed = jnp.max(jnp.abs(dx.qvel[self._strike_dofs]))
            dealt = dealt + rod_dealt * (1.0 + self._kin * rod_speed)
            fire_i = jnp.clip(clip_a[self._strike_local], 0.0, 1.0)        # per-rod fire command
            fire_cost = self._fire_cost * jnp.sum(fire_i)
            fire_act = jnp.mean(fire_i)
            # dense firing-shaping: reward firing a rod when ITS tip is near B (aimed/in-range).
            d_rb = jnp.linalg.norm(dx.geom_xpos[self._Arod_gids] - dx.xpos[self._Bt], axis=1)
            fire_aim = jnp.sum(fire_i * jnp.maximum(0.0, 1.0 - d_rb / self._rod_reach))
        if self._has_striker_b:
            rod_taken = jnp.sum(pen * ((self._Brod[g0] & self._Abody[g1]) | (self._Brod[g1] & self._Abody[g0])))
            rod_speed_b = jnp.max(jnp.abs(dx.qvel[self._strike_dofs_b]))
            taken = taken + rod_taken * (1.0 + self._kin * rod_speed_b)
        dealt_f = jnp.clip(dealt / DAMAGE_REF, 0, 1); taken_f = jnp.clip(taken / DAMAGE_REF, 0, 1)
        late_hit = (state.info["t"] >= self._min_hit_step).astype(jnp.float32)
        scored_dealt = dealt_f * late_hit
        early_dealt = dealt_f * (1.0 - late_hit)
        rel = (dx.xpos[self._Bt] - dx.xpos[self._At])[:2]; dist = jnp.linalg.norm(rel); n = dist + 1e-6
        toward = jnp.dot(dx.qvel[self._ArD:self._ArD + 2], rel) / n
        clos = jnp.clip(toward / 2, 0, 1); flee = jnp.clip(-toward / 2, 0, 1)
        close_term = jnp.clip(1.0 - dist / jnp.maximum(self._close_radius, 1e-6), 0.0, 1.0)
        rmat_face = dx.xmat[self._At].reshape(-1)
        fwd_xy = rmat_face[:2]
        fwd_xy = fwd_xy / (jnp.linalg.norm(fwd_xy) + 1e-6)
        face = jnp.dot(fwd_xy, rel / n)
        # UPRIGHTNESS via the torso's three PERPENDICULAR body axes (rotation matrix `xmat`, row-major
        # body->world), not a quaternion proxy. Third row = the world-Z component of the body's
        # [forward, side, up] axes. A perfectly upright torso has forward & side axes HORIZONTAL
        # (their world-Z ~ 0) and the up axis VERTICAL (world-Z ~ 1). This directly catches a BACKWARD
        # collapse onto the backside: the forward axis tips up (|fwd_z|↑) and the up axis falls
        # (up_z → 0 then negative) — the old `1-2(qx²+qy²)` proxy barely moved for that failure.
        R = dx.xmat[self._At].reshape(-1)          # MJX xmat is (3,3); flatten to row-major (9,)
        fwd_z, side_z, up_z = R[6], R[7], R[8]     # world-Z of body forward / side / up axes
        pitch_pen = jnp.abs(fwd_z)                 # tipping forward/backward (the backside-collapse axis)
        roll_pen = jnp.abs(side_z)                 # tipping sideways
        up = up_z - 0.6 * pitch_pen - 0.4 * roll_pen      # rich uprightness: 1=perfectly upright, <0=toppled
        # stance-height reward component (pay for standing TALL toward ~0.24 m, not a low sprawl)
        height = jnp.clip((dx.xpos[self._At][2] - 0.16) / 0.08, 0.0, 1.0)
        # the real SPARC objective (force/penetration-weighted damage + aggression):
        sparc = 6.0 * (scored_dealt - taken_f) + 5.0 * (clos - flee)
        # dense close→strike SHAPING (annealed via self._shaping; legs-as-weapons so getting
        # close + a limb on B already scores): close the bodies, AIM A LIMB at B's torso, and
        # a hit accelerator. The leg-proximity term gives the missing gradient to "land a hit".
        legdist = jnp.min(jnp.linalg.norm(dx.xpos[self._Astrike] - dx.xpos[self._Bt], axis=1))
        shaped = self._shaping * (-0.15 * dist - 0.20 * legdist + 3.0 * scored_dealt)
        # potential-based APPROACH reward: pay for distance CLOSED this step (approach
        # velocity). Dense + available before any hit -> "forces" learning to close the gap.
        approach = state.info["prev_dist"] - dist                 # >0 when the gap shrank
        # WIN-EXCHANGES asymmetry (headroom above the dealt≈taken plateau): a CLEAN hit (landed
        # while not being hit) pays more than a TRADE (mutual contact), and retreating right
        # after a hit (gated on prev_dealt, so it isn't idle fleeing) is rewarded.
        clean = scored_dealt * (1.0 - taken_f)          # landed WHILE NOT being hit
        trade = jnp.minimum(dealt_f, taken_f)           # mutual contact (drive DOWN)
        outward = jnp.clip(-toward / 2, 0, 1)           # moving AWAY from the opponent
        disengage = state.info["prev_dealt"] * outward  # retreat right after landing a scored hit
        energy = jnp.mean(jnp.abs(clip_a[:self._n_hinge]))      # hinge actuator effort (energy/safety)
        # ANTI-CHEAT lever: penalize being airborne (torso above stance) so the policy can't develop
        # a jump-to-strike tendency (the mild cousin of the launch exploit the anti-cheat metrics flag).
        airborne = jnp.maximum(0.0, dx.xpos[self._At][2] - self._airborne_z)
        # LOCOMOTION reward: torso planar SPEED (learn to translate the body while staying upright —
        # the topple-done + upright term keep it from lunge-and-fall, so it learns a controlled gait).
        move = jnp.linalg.norm(dx.qvel[self._ArD:self._ArD + 2])
        reward = (sparc + shaped + self._approach_w * approach + self._upright_w * up + 0.1
                  + self._height_w * height + self._move_w * move
                  + self._clean_w * clean - self._trade_w * trade + self._dis_w * disengage
                  - self._flee_w * flee + self._close_bonus_w * close_term
                  + self._face_w * close_term * face
                  + self._damage_bonus_w * scored_dealt
                  + self._fire_shaping * fire_aim - fire_cost - self._energy_w * energy
                  - self._airborne_w * airborne - self._early_hit_penalty * early_dealt
                  - self._taken_w * taken_f
                  - self._penalty_w * jnp.maximum(0.0, peak_pen_step - self._penalty_tol))
        # FALL = torso below 0.09 m. The 3.5 kg / gear-12 body holds a stable controllable stance at
        # torso-z ~0.15 (crouch ~0.11), measured; 0.09 sits below the crouch so dodging/crouching
        # survives but a real topple (torso ~0.05-0.07) is caught. The old 0.18 sat ABOVE this body's
        # max standing height (0.185, singular straight-leg) → survival was geometrically impossible.
        # FALL = torso below 0.09 m OR TOPPLED (up-axis tilted past ~70° from vertical). The height
        # check alone missed the backside collapse (torso stays ~0.1 m while lying on its back), so a
        # sprawl counted as "alive"; up_z<0.3 catches the topple even when the torso is still off the floor.
        done = jnp.where((dx.xpos[self._At][2] < 0.09) | (up_z < 0.3), 1.0, 0.0)
        # MERGE into the existing metrics dict (brax's Evaluator injects a 'reward' key —
        # replacing the dict drops it and breaks the scan-carry pytree).
        metrics = {**state.metrics, "dealt": dealt_f, "taken": taken_f, "closing": clos,
                   "fleeing": flee, "sparc": sparc, "dist": dist, "approach": approach,
                   "close_term": close_term, "clean_hit": clean, "trade": trade,
                   "disengage": disengage, "fire": fire_act, "face": face,
                   "penalty": jnp.maximum(0.0, peak_pen_step - self._penalty_tol)}
        return state.replace(pipeline_state=dx, obs=self._obs(dx, d), reward=reward, done=done,
                             metrics={**metrics},
                             info={**state.info, "prev_dist": dist,
                                   "prev_dealt": scored_dealt, "t": state.info["t"] + 1.0})


def _grow_action_head(policy, act_dim):
    """Grow the policy net's OUTPUT layer when the body gained DOFs (e.g. + pneumatic striker).
    The head emits [mean(act) | log_std(act)] along the last axis (brax NormalTanhDistribution
    splits it in half). Split each half and insert neutral (zero) columns for the new actions, so
    the EXISTING actions are preserved EXACTLY and the new ones start 'no-fire' but explorable."""
    pp = policy.get("params", policy)
    hids = [int(k.split("_")[1]) for k in pp if k.startswith("hidden_")]
    if not hids:
        return policy
    out = f"hidden_{max(hids)}"                       # the output layer
    ob = pp[out]["bias"]; old_act = ob.shape[0] // 2
    if act_dim is None or act_dim <= old_act:
        return policy
    gpad = act_dim - old_act
    def grow(x):
        mean, logstd = x[..., :old_act], x[..., old_act:]
        z = jnp.zeros(x.shape[:-1] + (gpad,), x.dtype)
        return jnp.concatenate([mean, z, logstd, z], axis=-1)
    pp[out] = {**pp[out], "kernel": grow(pp[out]["kernel"]), "bias": grow(ob)}
    print(f"WARM-START action head {old_act}->{act_dim} (+{gpad} striker dims, neutral init)", flush=True)
    return policy


def warm_start(path, obs_dim, act_dim=None):
    """Pad a saved (normalizer, policy, value) tuple to the target body. Input layers + normalizer
    grow 38->obs_dim (the opponent inputs init ~0) when warm-starting the LOCOMOTOR; the policy
    ACTION HEAD grows when the body gained DOFs (the striker, `act_dim`). Best-effort with a
    scratch fall-back. Idempotent: a same-shape checkpoint passes through untouched."""
    try:
        parts = list(pickle.load(open(path, "rb")))      # (normalizer, policy_dict, value_dict, ...)
        norm, nets = parts[0], parts[1:]
        pp = nets[0].get("params", nets[0]) if nets else {}
        old_obs = int(pp.get("hidden_0", {}).get("kernel", jnp.zeros((LOCO_OBS, 1))).shape[0])
        if old_obs == obs_dim:
            pad = 0
        elif old_obs in (LOCO_OBS, LOCO_OBS + 2):
            pad = obs_dim - LOCO_OBS
        else:
            pad = obs_dim - old_obs
        if pad < 0:
            raise ValueError(f"checkpoint obs {old_obs} is wider than target obs {obs_dim}")
        c = norm.count                                   # brax UInt64 = {hi, lo}: value = hi*2^32 + lo
        cval = float(jnp.asarray(c.hi)) * (2.0 ** 32) + float(jnp.asarray(c.lo))
        keep = LOCO_OBS if old_obs == LOCO_OBS + 2 and obs_dim >= LOCO_OBS + 6 else old_obs
        def remap_vec(v, fill):
            out = jnp.full((obs_dim,), fill, dtype=v.dtype)
            n = min(keep, obs_dim, v.shape[0])
            return out.at[:n].set(v[:n])
        nkw = {}
        for fn in ("mean", "std", "summed_variance"):
            v = getattr(norm, fn, None)
            if hasattr(v, "ndim") and v.ndim >= 1 and v.shape[0] == old_obs and old_obs != obs_dim:
                # new (opponent) dims start standardized: mean 0, std 1, summed_variance=count (var~1)
                fill = 0.0 if fn == "mean" else 1.0 if fn == "std" else max(cval, 1.0)
                nkw[fn] = remap_vec(v, fill)
        norm = norm.replace(**nkw)
        def pad_leaf(x):
            if not (hasattr(x, "ndim") and x.ndim >= 1 and x.shape[0] == old_obs and old_obs != obs_dim):
                return x
            out = jnp.zeros((obs_dim,) + x.shape[1:], dtype=x.dtype)
            n = min(keep, obs_dim, x.shape[0])
            return out.at[:n].set(x[:n])
        nets = [jax.tree_util.tree_map(pad_leaf, n) for n in nets]
        if nets:                                         # grow the policy (nets[0]) action head only
            nets[0] = _grow_action_head(nets[0], act_dim)
        print(f"WARM-START ok: obs {old_obs}->{obs_dim} keep={keep} "
              f"(count={cval:.0f}, normalizer + {len(nets)} nets)", flush=True)
        return tuple([norm] + nets)
    except Exception as e:
        print(f"warm-start failed ({type(e).__name__}: {e}) -> training Stage B from scratch", flush=True)
        return None


def load_opponent(path, obs=None, act=None):
    """Frozen opponent (deterministic) inference fn from a saved striker ckpt — drives B in
    self-play. obs/act are INFERRED from the policy net if not given (input-layer width = obs;
    output-layer width / 2 = act). Returns a bound `policy(obs, key) -> (action, extra)`."""
    from brax.training.agents.ppo import networks as ppo_networks
    from brax.training.acme import running_statistics
    params = pickle.load(open(path, "rb"))
    pol = params[1]["params"]
    if obs is None:
        obs = int(pol["hidden_0"]["kernel"].shape[0])
    if act is None:
        hids = [int(k.split("_")[1]) for k in pol if k.startswith("hidden_")]
        act = int(pol[f"hidden_{max(hids)}"]["bias"].shape[0]) // 2
    net = ppo_networks.make_ppo_networks(obs, act, preprocess_observations_fn=running_statistics.normalize)
    return ppo_networks.make_inference_fn(net)(params, deterministic=True)


def head_only_network_factory(observation_size, action_size, preprocess_observations_fn):
    """PPO network factory that stops policy gradients outside the final dense layer."""
    from brax.training import networks as brax_networks
    from brax.training.agents.ppo import networks as ppo_networks
    base = ppo_networks.make_ppo_networks(
        observation_size, action_size,
        preprocess_observations_fn=preprocess_observations_fn)
    base_policy = base.policy_network

    def freeze_non_output(policy):
        pp = policy.get("params", policy)
        hids = [k for k in pp if k.startswith("hidden_")]
        if not hids:
            return policy
        out = max(hids, key=lambda k: int(k.split("_")[1]))
        pp2 = {k: (v if k == out else jax.tree_util.tree_map(jax.lax.stop_gradient, v))
               for k, v in pp.items()}
        return {**policy, "params": pp2} if "params" in policy else pp2

    def apply(norm, policy, obs):
        return base_policy.apply(norm, freeze_non_output(policy), obs)

    return ppo_networks.PPONetworks(
        policy_network=brax_networks.FeedForwardNetwork(init=base_policy.init, apply=apply),
        value_network=base.value_network,
        parametric_action_distribution=base.parametric_action_distribution,
    )


def build_benchmark(bench_env, n_epis, steps, seed=20240601, deterministic=True):
    """Honest held-out benchmark: mean over FIXED-seed episodes (fixed designs/spawns, comparable
    across curriculum phases) of the episode-summed [SPARC, dealt, taken] COMBAT metrics — read
    from `state.metrics`, so independent of the shaped training reward. Network reconstructed the
    same way `combat_rank.load_policy` does (proven to consume these checkpoints). Returns a jitted
    `bench(params) -> [sparc, dealt, taken]`; keep-best on `sparc` makes best-so-far monotone."""
    from brax.training.agents.ppo import networks as ppo_networks
    from brax.training.acme import running_statistics
    net = ppo_networks.make_ppo_networks(bench_env.observation_size, bench_env.action_size,
                                         preprocess_observations_fn=running_statistics.normalize)
    make_inf = ppo_networks.make_inference_fn(net)
    keys = jax.random.split(jax.random.PRNGKey(seed), n_epis)        # FIXED held-out configs

    @jax.jit
    def bench(params):
        inf = make_inf(params, deterministic=deterministic)
        def ep(k):
            st = bench_env.reset(k)
            d0 = jnp.linalg.norm((st.pipeline_state.xpos[bench_env._Bt]
                                  - st.pipeline_state.xpos[bench_env._At])[:2])   # initial separation
            def stp(carry, _):
                s, key, t = carry; key, sk = jax.random.split(key)
                a, _ = inf(s.obs, sk); s = bench_env.step(s, a); alive = 1.0 - s.done
                m = s.metrics
                sat = jnp.mean(jnp.abs(a[:bench_env._n_hinge]) > 0.95)   # actuator saturation (slamming)
                base = jnp.array([m["sparc"] * alive, m["dealt"] * alive, m["taken"] * alive,
                                  m["clean_hit"] * alive, m["trade"] * alive, m["fire"] * alive,
                                  m["closing"] * alive, m["fleeing"] * alive, m["dist"] * alive,
                                  alive, sat])
                # ANTI-CHEAT raw per-step (so the eval EXPOSES launch/clash-idle exploits live):
                ps = s.pipeline_state; qa = ps.qpos
                tz = ps.xpos[bench_env._At][2]                           # A torso height
                up_a = ps.xmat[bench_env._At].reshape(-1)[8]             # up-axis world-Z (1=upright, <0=toppled)
                pen = jnp.max(jnp.maximum(0.0, -ps.contact.dist))        # deepest contact penetration
                idle = (jnp.mean(jnp.abs(a)) < 0.1).astype(jnp.float32)
                ac = jnp.array([tz, pen, m["dealt"], up_a, idle, t])
                return (s, key, t + 1.0), (base, ac)
            (_, _, _), (base_o, ac_o) = jax.lax.scan(stp, (st, k, 0.0), None, length=steps)
            tz_c, pen_c, dl_c, up_c, idle_c, t_c = (ac_o[:, i] for i in range(6))
            dmg_tot = dl_c.sum()
            tot = jnp.maximum(dmg_tot, 1e-9)
            has_dmg = dmg_tot > 1e-6
            ac_agg = jnp.array([
                tz_c.max(),                                    # peak_torso_z   (launch: >0.45)
                (tz_c > bench_env._airborne_z).mean(),          # airborne_frac  (>0.05)
                pen_c.max(),                                   # peak_penetration (>0.05 = solver exploit)
                idle_c.mean(),                                 # idle_frac      (>0.5 = freezing)
                jnp.where(has_dmg, (dl_c * (t_c < 0.15 * steps)).sum() / tot, 0.0),  # spawn-clash
                jnp.where(has_dmg, (dl_c * (up_c > 0.5)).sum() / tot, 1.0),  # upright_dmg_frac
                jnp.where(has_dmg, (dl_c * (tz_c < bench_env._grounded_z)).sum() / tot, 1.0),  # grounded_dmg_frac
                up_c.mean(),                                   # uprightness (mean up-axis·world-up; ~1=stands, <0.5=topples)
            ])
            return base_o.sum(0), d0, ac_agg
        per_ep, d0, ac_ep = jax.vmap(ep)(keys)                # per_ep:(n,11); ac_ep:(n,7)
        agg = per_ep[:, :10].mean(0)                          # the 10 dense decomposition signals
        spe = per_ep[:, 0]                                    # per-episode SPARC sum, for range bins
        bm = lambda mask: jnp.sum(spe * mask) / jnp.maximum(jnp.sum(mask), 1.0)
        bins = jnp.array([bm(d0 < 0.6), bm((d0 >= 0.6) & (d0 < 0.9)), bm(d0 >= 0.9)])  # close/med/far SPARC
        # SPARSE VERDICT — unshaped per-bout pass/fail (the honest judgment the coach CAN'T reach):
        dealt_s, taken_s, alive_s, sat_s = per_ep[:, 1], per_ep[:, 2], per_ep[:, 9], per_ep[:, 10]
        survived_bout = alive_s >= steps - 0.5                            # didn't fall the whole bout
        # CLEAN WIN = out-damaged the opponent AND stayed upright. keep-best selects on THIS, so a
        # topple-after-a-hit no longer counts as a win — the judgment now matches the goal, and the
        # Coach's `upright` lever has the selection pressure AGREEING with it (not fighting it).
        win = jnp.mean(((dealt_s - taken_s > 0.0) & survived_bout).astype(jnp.float32))
        surv = jnp.mean(survived_bout.astype(jnp.float32))
        safe = jnp.mean(((sat_s / steps) < 0.5).astype(jnp.float32))      # didn't slam actuators most of the bout
        return jnp.concatenate([agg, bins, jnp.array([win, surv, safe]), ac_ep.mean(0)])  # 16 + 7 anti-cheat
    return bench


# build_benchmark()'s 16-value vector: dense decomposition (for the Coach) + range profile + the
# SPARSE VERDICT (win_rate/survival_rate/safe_rate — the unshaped judgment; keep-best selects on win_rate).
BENCH_KEYS = ["sparc", "dealt", "taken", "clean", "trade", "fire", "closing", "fleeing", "dist",
              "alive", "sparc_close", "sparc_med", "sparc_far", "win_rate", "survival_rate", "safe_rate",
              # anti-cheat block (catch degenerate strategies live — see anti_cheat.py for the full 24):
              "ac_peak_z", "ac_airborne", "ac_peak_pen", "ac_idle", "ac_dmg_early",
              "ac_upright_dmg", "ac_grounded_dmg", "ac_uprightness"]


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--steps", type=int, default=12_000_000)
    ap.add_argument("--envs", type=int, default=2048); ap.add_argument("--resume", default=None)
    ap.add_argument("--batch", type=int, default=1024); ap.add_argument("--minibatches", type=int, default=16)
    ap.add_argument("--unroll", type=int, default=20); ap.add_argument("--evals", type=int, default=0)
    ap.add_argument("--shaping", type=float, default=1.0, help="dense close→strike shaping weight (anneal to 0)")
    ap.add_argument("--sep", type=float, default=1.0, help="base start separation")
    ap.add_argument("--sep-lo", type=float, default=None, help="curriculum: min start separation per reset")
    ap.add_argument("--sep-hi", type=float, default=None, help="curriculum: max start separation per reset")
    ap.add_argument("--approach-weight", type=float, default=0.0,
                    help="velocity/approach reward: weight on distance CLOSED per step (anneal over phases)")
    ap.add_argument("--azimuth", type=float, default=0.0,
                    help="±opponent bearing range in rad (0=head-on; 3.14=any angle) — varied attack angles")
    ap.add_argument("--updates", type=int, default=4, help="num_updates_per_batch (more = more SGD/iteration)")
    ap.add_argument("--frame-skip", type=int, default=5, help="control decimation (higher = fewer mjx.steps)")
    ap.add_argument("--lr", type=float, default=3e-4); ap.add_argument("--entropy", type=float, default=1e-2)
    ap.add_argument("--clip-epsilon", type=float, default=0.3,
                    help="PPO policy-ratio clip; lower values make resumed/head-only probes more conservative")
    ap.add_argument("--desired-kl", type=float, default=0.01,
                    help="target KL used by Brax PPO, especially with --lr-schedule ADAPTIVE_KL")
    ap.add_argument("--max-grad-norm", type=float, default=None,
                    help="optional PPO gradient clipping norm")
    ap.add_argument("--lr-schedule", choices=["NONE", "ADAPTIVE_KL"], default=None,
                    help="optional Brax PPO learning-rate schedule")
    ap.add_argument("--lean-contacts", action="store_true", help="F-SPEED: reduced-collision fight scene")
    # WIN-EXCHANGES reward asymmetry (STEP 2b) — default 0 = the contact-forced fighter's reward.
    ap.add_argument("--clean-weight", type=float, default=0.0, help="+w·dealt·(1−taken): reward un-traded hits")
    ap.add_argument("--trade-weight", type=float, default=0.0, help="−w·min(dealt,taken): punish mutual contact")
    ap.add_argument("--disengage-weight", type=float, default=0.0,
                    help="+w·prev_dealt·outward_vel: reward retreat right AFTER a hit (anneal — don't make it flee)")
    ap.add_argument("--fire-shaping", type=float, default=0.0,
                    help="dense reward for firing a rod when its tip is aimed/in-range at B (anneal as hits take over)")
    ap.add_argument("--rod-reach", type=float, default=0.30,
                    help="reach radius for dense striker fire shaping")
    ap.add_argument("--striker-rod-len", type=float, default=None,
                    help="override physical striker rod length for this run")
    ap.add_argument("--striker-stroke", type=float, default=None,
                    help="override physical striker slide stroke for this run")
    ap.add_argument("--striker-rod-radius", type=float, default=None,
                    help="override physical striker rod radius for this run")
    ap.add_argument("--contact-solref-timeconst", type=float, default=None,
                    help="override MuJoCo geom contact solref time constant for this run")
    ap.add_argument("--contact-solref-dampratio", type=float, default=1.0,
                    help="MuJoCo geom contact solref damping ratio used with --contact-solref-timeconst")
    ap.add_argument("--floor-calf-solref-timeconst", type=float, default=None,
                    help="override only floor-vs-calf pair contact solref time constant for this run")
    ap.add_argument("--floor-calf-solref-dampratio", type=float, default=1.0,
                    help="MuJoCo damping ratio used with --floor-calf-solref-timeconst")
    ap.add_argument("--disable-calf-floor", action="store_true",
                    help="in lean-contact matches, let calf capsules collide with opponents but not the floor")
    ap.add_argument("--upright-weight", type=float, default=0.1, help="BALANCE anchor weight (Coach raises it when survival lags). Low default: the gear-12 body stands nearly for free, so a high survival reward just teaches stand-and-avoid; engagement (approach/dealt) must dominate.")
    ap.add_argument("--energy-penalty", type=float, default=0.0, help="ENERGY/actuator-safety: penalize hinge effort (Coach raises it when slamming)")
    ap.add_argument("--airborne-penalty", type=float, default=float(os.environ.get("AIRBORNE_PENALTY", "0")), help="ANTI-CHEAT: penalize torso above 0.35m (discourage jump-to-strike). Env default AIRBORNE_PENALTY so it applies to every arena-spawned stage too.")
    ap.add_argument("--height-weight", type=float, default=float(os.environ.get("HEIGHT_WEIGHT", "0")), help="reward standing TALL (torso-z toward ~0.24m) — stops the low-sprawl posture. Env default HEIGHT_WEIGHT for arena stages.")
    ap.add_argument("--move-weight", type=float, default=float(os.environ.get("MOVE_WEIGHT", "0")), help="LOCOMOTION pretrain: reward torso planar SPEED (learn to WALK this body). A loco stage sets this high + combat off so the fighter inherits mobility instead of parking. Env default MOVE_WEIGHT for arena stages.")
    ap.add_argument("--early-hit-penalty", type=float, default=0.0,
                    help="penalize damage dealt before --min-hit-step so close curricula do not learn spawn clashes")
    ap.add_argument("--min-hit-step", type=int, default=0,
                    help="training reward only counts dealt damage at or after this environment step")
    ap.add_argument("--taken-weight", type=float, default=0.0,
                    help="extra penalty on damage taken; useful for avoid-trade and hit-reset phases")
    ap.add_argument("--flee-penalty", type=float, default=float(os.environ.get("FLEE_PENALTY", "0")),
                    help="extra per-step penalty on moving away from the opponent; SPARC-focused anti-avoidance lever")
    ap.add_argument("--close-bonus", type=float, default=float(os.environ.get("CLOSE_BONUS", "0")),
                    help="reward being inside --close-radius; use with damage/anti-flee shaping to avoid idle hugging")
    ap.add_argument("--close-radius", type=float, default=float(os.environ.get("CLOSE_RADIUS", "0.45")),
                    help="distance radius for --close-bonus")
    ap.add_argument("--damage-bonus", type=float, default=float(os.environ.get("DAMAGE_BONUS", "0")),
                    help="extra reward on scored dealt damage; useful when held-out SPARC is contact-starved")
    ap.add_argument("--face-opponent", action="store_true",
                    help="spawn A and B yawed toward each other for randomized bearings")
    ap.add_argument("--face-weight", type=float, default=0.0,
                    help="reward close-range torso/front-striker alignment toward the opponent")
    ap.add_argument("--penetration-penalty", type=float, default=0.0,
                    help="penalize per-step peak contact penetration above --penetration-tol")
    ap.add_argument("--penetration-tol", type=float, default=0.045,
                    help="tolerance for --penetration-penalty, in meters")
    ap.add_argument("--reset-bank-seed", type=int, default=None,
                    help="sample training resets from the fixed benchmark episode keys for this seed")
    ap.add_argument("--reset-bank-epis", type=int, default=8,
                    help="number of fixed episode keys in --reset-bank-seed")
    ap.add_argument("--engage-obs", action="store_true",
                    help="append opponent-direction and radial/lateral engagement features to observations")
    ap.add_argument("--contact-obs", action="store_true",
                    help="append rod distance, limb distance, and torso-to-opponent alignment features")
    ap.add_argument("--policy-train-scope", choices=["full", "head"], default="full",
                    help="head trains only the final policy dense layer; value net still trains normally")
    ap.add_argument("--freeze-normalizer", action="store_true",
                    help="benchmark/save checkpoints with the warm-start normalizer to avoid policy drift from obs stats")
    ap.add_argument("--milestone-gap", type=int, default=int(os.environ.get("MILESTONE_GAP", "0")), help="save a numbered milestone checkpoint+sidecar every N cum-steps (0=off). Env default MILESTONE_GAP so arena stages save milestones too — the render daemon turns each into a 1v1 evolution video")
    # Held-out BENCHMARK eval (the honest monotone-improvement curve + keep-best selection). Run
    # at every eval on a FIXED config (comparable across curriculum phases), independent of the
    # shaped training reward — reads the SPARC/dealt/taken metrics, not `reward`.
    ap.add_argument("--bench-epis", type=int, default=16, help="benchmark episodes (fixed held-out seeds)")
    ap.add_argument("--bench-steps", type=int, default=200, help="benchmark steps/episode")
    ap.add_argument("--bench-seeds", default="20240601",
                    help="comma-separated fixed benchmark seeds averaged for keep-best selection")
    ap.add_argument("--bench-sep-lo", type=float, default=0.4, help="benchmark fixed start-sep low")
    ap.add_argument("--bench-sep-hi", type=float, default=1.2, help="benchmark fixed start-sep high")
    ap.add_argument("--bench-az", type=float, default=3.14159, help="benchmark azimuth range (all angles)")
    ap.add_argument("--no-benchmark", action="store_true", help="disable benchmark eval / keep-best")
    ap.add_argument("--keep-metric", choices=["win", "sparc", "ratio", "margin", "judge",
                                              "min_margin", "min_judge"], default="win",
                    help="held-out metric for keep-best selection; default win is strict, sparc helps early self-play before win-rate is nonzero")
    ap.add_argument("--min-keep-dealt", type=float, default=0.0,
                    help="minimum held-out dealt damage required for a checkpoint to become keep-best")
    ap.add_argument("--max-keep-early-dmg", type=float, default=1.0,
                    help="maximum early-damage fraction allowed for a checkpoint to become keep-best")
    ap.add_argument("--max-keep-peak-pen", type=float, default=1.0,
                    help="maximum held-out peak penetration allowed for a checkpoint to become keep-best")
    ap.add_argument("--min-keep-margin", type=float, default=-1e30,
                    help="minimum per-seed dealt-minus-taken margin required for keep-best")
    ap.add_argument("--min-keep-survival", type=float, default=0.0,
                    help="minimum held-out survival rate required for keep-best")
    ap.add_argument("--min-keep-safe", type=float, default=0.0,
                    help="minimum held-out actuator safe rate required for keep-best")
    ap.add_argument("--no-striker", action="store_true", help="disable the pneumatic striker (legacy 12-action body)")
    ap.add_argument("--opponent", choices=["passive", "frozen"], default="passive",
                    help="B opponent: passive (skill curriculum) or frozen (self-play vs a snapshot)")
    ap.add_argument("--opp-ckpt", default=None, help="frozen opponent ckpt (striker snapshot) driving B")
    ap.add_argument("--bench-opp-ckpt", default=None,
                    help="FIXED reference opponent for the benchmark (default: same as --opp-ckpt) — comparable across self-play rounds")
    ap.add_argument("--cum-base", type=int, default=0, help="cumulative env-step base (resume: prior total steps)")
    ap.add_argument("--tag", default="adv", help="checkpoint/metrics tag")
    ap.add_argument("--tiny", action="store_true", help="lightweight plumbing run (e2e harness)")
    args = ap.parse_args()
    if args.tiny:
        args.steps, args.envs = 8_000, 256
        args.batch, args.minibatches, args.unroll = 256, 8, 5
        args.evals = 2
        args.bench_epis, args.bench_steps = 4, 40
    n_eval = args.evals or max(6, args.steps // 1_000_000)
    global SPEC
    if any(x is not None for x in (
        args.striker_rod_len,
        args.striker_stroke,
        args.striker_rod_radius,
        args.contact_solref_timeconst,
        args.floor_calf_solref_timeconst,
        args.disable_calf_floor,
    )):
        SPEC = copy.deepcopy(SPEC)
        SPEC.setdefault("striker", {})
        SPEC.setdefault("contact", {})
        if args.striker_rod_len is not None:
            SPEC["striker"]["rod_len"] = float(args.striker_rod_len)
        if args.striker_stroke is not None:
            SPEC["striker"]["stroke"] = float(args.striker_stroke)
        if args.striker_rod_radius is not None:
            SPEC["striker"]["rod_radius"] = float(args.striker_rod_radius)
        if args.contact_solref_timeconst is not None:
            SPEC["contact"]["solref"] = [
                float(args.contact_solref_timeconst),
                float(args.contact_solref_dampratio),
            ]
        if args.floor_calf_solref_timeconst is not None:
            SPEC["contact"]["floor_calf_solref"] = [
                float(args.floor_calf_solref_timeconst),
                float(args.floor_calf_solref_dampratio),
            ]
        if args.disable_calf_floor:
            SPEC["contact"]["calf_floor"] = False
        print(
            "geometry/contact override: "
            f"rod_len={SPEC['striker'].get('rod_len')} "
            f"stroke={SPEC['striker'].get('stroke')} "
            f"rod_radius={SPEC['striker'].get('rod_radius')} "
            f"contact_solref={SPEC.get('contact', {}).get('solref')} "
            f"floor_calf_solref={SPEC.get('contact', {}).get('floor_calf_solref')} "
            f"calf_floor={SPEC.get('contact', {}).get('calf_floor', True)}",
            flush=True,
        )
    striker = False if args.no_striker else None     # None = spec default (ON)
    # self-play: the TRAINING opponent (rotates through the HoF) and the BENCHMARK opponent (a
    # FIXED reference so the benchmark is comparable across rounds). passive => skill curriculum.
    opp_infer = (load_opponent(args.opp_ckpt) if args.opponent == "frozen"
                 and args.opp_ckpt and os.path.exists(args.opp_ckpt) else None)
    bench_opp_path = args.bench_opp_ckpt or args.opp_ckpt
    bench_opp = (load_opponent(bench_opp_path) if args.opponent == "frozen"
                 and bench_opp_path and os.path.exists(bench_opp_path) else None)
    t_env = time.time(); env = AdversarialEnv(shaping=args.shaping, sep=args.sep,
                                              self_collision=not args.lean_contacts,
                                              frame_skip=args.frame_skip,
                                              sep_lo=args.sep_lo, sep_hi=args.sep_hi,
                                              approach_weight=args.approach_weight, azimuth=args.azimuth,
                                              clean_weight=args.clean_weight, trade_weight=args.trade_weight,
                                              disengage_weight=args.disengage_weight, striker=striker,
                                              fire_shaping=args.fire_shaping, rod_reach=args.rod_reach,
                                              upright_weight=args.upright_weight,
                                              energy_penalty=args.energy_penalty,
                                              airborne_penalty=args.airborne_penalty,
                                              height_weight=args.height_weight,
                                              move_weight=args.move_weight,
                                              early_hit_penalty=args.early_hit_penalty,
                                              min_hit_step=args.min_hit_step,
                                              taken_weight=args.taken_weight,
                                              flee_penalty=args.flee_penalty,
                                              close_bonus=args.close_bonus,
                                              close_radius=args.close_radius,
                                              damage_bonus=args.damage_bonus,
                                              face_opponent=args.face_opponent,
                                              engage_obs=args.engage_obs,
                                              contact_obs=args.contact_obs,
                                              face_weight=args.face_weight,
                                              penetration_penalty=args.penetration_penalty,
                                              penetration_tol=args.penetration_tol,
                                              reset_bank_seed=args.reset_bank_seed,
                                              reset_bank_epis=args.reset_bank_epis,
                                              opponent=args.opponent, opp_infer=opp_infer)
    METRIC(stage="adv_env_build", t_s=f"{time.time()-t_env:.1f}",
           obs=env.observation_size, act=env.action_size, striker=int(env._has_striker),
           opponent=args.opponent, flee_penalty=args.flee_penalty,
           close_bonus=args.close_bonus, close_radius=args.close_radius,
           damage_bonus=args.damage_bonus, engage_obs=int(args.engage_obs),
           contact_obs=int(args.contact_obs),
           face_weight=args.face_weight,
           penetration_penalty=args.penetration_penalty,
           penetration_tol=args.penetration_tol,
           striker_rod_len=SPEC.get("striker", {}).get("rod_len"),
           striker_stroke=SPEC.get("striker", {}).get("stroke"),
           striker_rod_radius=SPEC.get("striker", {}).get("rod_radius"),
           contact_solref=SPEC.get("contact", {}).get("solref"),
           floor_calf_solref=SPEC.get("contact", {}).get("floor_calf_solref"),
           calf_floor=SPEC.get("contact", {}).get("calf_floor", True),
           reset_bank_seed=args.reset_bank_seed,
           reset_bank_epis=args.reset_bank_epis)
    print(f"adversarial env: obs={env.observation_size} act(A)={env.action_size} "
          f"striker={env._has_striker} opponent={args.opponent} "
          f"flee_penalty={args.flee_penalty} close_bonus={args.close_bonus} "
          f"damage_bonus={args.damage_bonus}", flush=True)
    # held-out benchmark (fixed config + fixed reference opponent) → honest improvement curve + keep-best
    benches = []
    bench_seeds = []
    if not args.no_benchmark:
        bench_opponent = "frozen" if bench_opp is not None else "passive"
        bench_env = AdversarialEnv(self_collision=not args.lean_contacts, frame_skip=args.frame_skip,
                                   sep_lo=args.bench_sep_lo, sep_hi=args.bench_sep_hi, azimuth=args.bench_az,
                                   striker=striker, opponent=bench_opponent, opp_infer=bench_opp,
                                   face_opponent=args.face_opponent, engage_obs=args.engage_obs,
                                   contact_obs=args.contact_obs,
                                   face_weight=args.face_weight,
                                   penetration_penalty=args.penetration_penalty,
                                   penetration_tol=args.penetration_tol,
                                   rod_reach=args.rod_reach)
        bench_seeds = [int(x) for x in str(args.bench_seeds).split(",") if x.strip()]
        benches = [build_benchmark(bench_env, args.bench_epis, args.bench_steps, seed=s)
                   for s in bench_seeds]
    restore = (warm_start(args.resume, env.observation_size, env.action_size)
               if args.resume and os.path.exists(args.resume) else None)
    frozen_norm = restore[0] if args.freeze_normalizer and restore is not None else None
    METRIC(stage="warm_start", ok=int(restore is not None),
           resume=os.path.basename(args.resume) if args.resume else "none")
    import json
    t0 = time.time(); csv = OUT / "adv_metrics.csv"; csv.write_text("step,reward,sec\n")
    fjson = OUT / "fight_metrics.jsonl"; fjson.write_text("")          # F0: the six trackers
    bjson = OUT / f"{args.tag}_benchmark.jsonl"; bjson.write_text("")  # the honest monotone curve
    for stale in (OUT / f"{args.tag}_best.pkl", OUT / f"{args.tag}_state.json"):
        try:
            stale.unlink()
        except FileNotFoundError:
            pass
    tm = {"first_eval": None}; last = {"r": float("nan"), "step": 0, "dealt": 0.0, "taken": 0.0}
    best = {"score": -1e30, "win": -1.0, "sparc": -1e30, "dealt": 0.0, "taken": 0.0,
            "ratio": 0.0, "margin": -1e30,
            "min_margin": -1e30, "min_judge": -1e30, "min_dealt": 0.0,
            "max_early": 0.0, "max_peak_pen": 0.0,
            "judge": -1e30, "early": 0.0, "airborne": 0.0, "idle": 0.0,
            "uprightness": 1.0, "step": -1, "last_ms": -10**18}
    def g(m, k): return float(m.get(f"eval/episode_{k}", 0.0))
    def save(obj, name):
        try: pickle.dump(obj, open(OUT / name, "wb"))
        except Exception as e: print(f"  [ck] save {name} failed: {e}", flush=True)
    def eval_params(params):
        if frozen_norm is None:
            return params
        return tuple([frozen_norm] + list(params[1:]))
    def prog(s, m):
        if tm["first_eval"] is None: tm["first_eval"] = time.time() - t0
        r = g(m, "reward"); dealt = g(m, "dealt"); taken = g(m, "taken")
        clos = g(m, "closing"); flee = g(m, "fleeing"); sparc = g(m, "sparc"); dist = g(m, "dist")
        clean = g(m, "clean_hit"); trade = g(m, "trade"); dis = g(m, "disengage"); fire = g(m, "fire")
        open(csv, "a").write(f"{int(s)},{r:.3f},{time.time()-t0:.0f}\n")
        rec = dict(step=int(s), sec=round(time.time()-t0, 0), reward=round(r, 3), sparc=round(sparc, 3),
                   dealt=round(dealt, 4), taken=round(taken, 4), closing=round(clos, 4),
                   fleeing=round(flee, 4), dist=round(dist, 3), clean_hit=round(clean, 4),
                   trade=round(trade, 4), disengage=round(dis, 4), fire=round(fire, 4), tag=args.tag,
                   shaping=args.shaping, sep=args.sep)
        open(fjson, "a").write(json.dumps(rec) + "\n")
        last.update(r=r, step=int(s), dealt=dealt, taken=taken)
        print(f"  [{args.tag}] step {int(s):>9,} sparc {sparc:6.2f} dealt {dealt:.3f} taken {taken:.3f} "
              f"clean {clean:.3f} fire {fire:.3f} close {clos:.2f} dist {dist:.2f} ({time.time()-t0:.0f}s)",
              flush=True)
    def ck(*a):
        step, params = int(a[0]), a[-1]
        p_latest = OUT / f"{args.tag}_ckpt.pkl"
        try:                                            # rotate latest→prev (in-run rollback point)
            if p_latest.exists(): os.replace(p_latest, OUT / f"{args.tag}_prev.pkl")
        except Exception: pass
        params_eval = eval_params(params)
        save(params_eval, f"{args.tag}_ckpt.pkl")
        if not benches: return
        try:
            vals_by_seed = np.stack([np.asarray(bench(params_eval)) for bench in benches], axis=0)
            vals = np.mean(vals_by_seed, axis=0)
            v = {k: float(vals[i]) for i, k in enumerate(BENCH_KEYS)}   # full held-out decomposition
            bsparc, bdealt, btaken = v["sparc"], v["dealt"], v["taken"]
        except Exception as e:
            print(f"  [bench] failed: {type(e).__name__}: {e}", flush=True); return
        seed_rows = [{k: float(row[i]) for i, k in enumerate(BENCH_KEYS)} for row in vals_by_seed]
        seed_margins = np.array([r["dealt"] - r["taken"] for r in seed_rows], dtype=np.float64)
        seed_judges = np.array([
            100.0 * r["win_rate"] + r["sparc"] + 20.0 * (r["dealt"] - r["taken"])
            - 10.0 * max(0.0, r["ac_idle"] - 0.3)
            for r in seed_rows
        ], dtype=np.float64)
        seed_dealt = np.array([r["dealt"] for r in seed_rows], dtype=np.float64)
        seed_early = np.array([r["ac_dmg_early"] for r in seed_rows], dtype=np.float64)
        seed_peak_pen = np.array([r["ac_peak_pen"] for r in seed_rows], dtype=np.float64)
        bmin_margin = float(seed_margins.min())
        bmin_judge = float(seed_judges.min())
        bmin_dealt = float(seed_dealt.min())
        bmax_early = float(seed_early.max())
        bmax_peak_pen = float(seed_peak_pen.max())
        bratio = bdealt / max(btaken, 1e-6)
        bmargin = bdealt - btaken
        bwin = v["win_rate"]                                  # the SPARSE VERDICT (unshaped)
        idle_pen = 10.0 * max(0.0, v["ac_idle"] - 0.3)
        bjudge = 100.0 * bwin + bsparc + 20.0 * bmargin - idle_pen
        bscore = {"win": bwin, "sparc": bsparc, "ratio": bratio,
                  "margin": bmargin, "judge": bjudge,
                  "min_margin": bmin_margin, "min_judge": bmin_judge}[args.keep_metric]
        keep_ok = ((bmin_dealt >= args.min_keep_dealt)
                   and (bmax_early <= args.max_keep_early_dmg)
                   and (bmax_peak_pen <= args.max_keep_peak_pen)
                   and (bmin_margin >= args.min_keep_margin)
                   and (v["survival_rate"] >= args.min_keep_survival)
                   and (v["safe_rate"] >= args.min_keep_safe))
        improved = keep_ok and bscore > best["score"]
        if improved:
            best.update(score=bscore, win=bwin, sparc=bsparc, ratio=bratio,
                        dealt=bdealt, taken=btaken, margin=bmargin, judge=bjudge,
                        min_margin=bmin_margin, min_judge=bmin_judge,
                        min_dealt=bmin_dealt, max_early=bmax_early,
                        max_peak_pen=bmax_peak_pen, step=step)
            best.update(early=v["ac_dmg_early"], airborne=v["ac_airborne"],
                        idle=v["ac_idle"], uprightness=v["ac_uprightness"])
            save(params_eval, f"{args.tag}_best.pkl")
        # MILESTONE checkpoints (for evolution videos): save a numbered ckpt + metric sidecar every
        # `--milestone-gap` cum-steps. A render daemon turns each into a 1v1 mp4 (downloaded locally).
        cs = args.cum_base + step
        if args.milestone_gap > 0 and cs - best["last_ms"] >= args.milestone_gap:
            best["last_ms"] = cs
            save(params_eval, f"{args.tag}_ms_{cs:09d}.pkl")
            json.dump(dict(tag=args.tag, step=cs, win=round(bwin, 3), ratio=round(bratio, 3),
                           sparc=round(bsparc, 2), survival=round(v["survival_rate"], 3),
                           ac_peak_z=round(v["ac_peak_z"], 3), ac_airborne=round(v["ac_airborne"], 3),
                           ac_idle=round(v["ac_idle"], 3), ac_grounded_dmg=round(v["ac_grounded_dmg"], 3)),
                      open(OUT / f"{args.tag}_ms_{cs:09d}.json", "w"))
            print(f"  [milestone] ms_{cs:09d} (win {bwin:.2f} ratio {bratio:.2f}) -> render queue", flush=True)
        rec = dict(step=step, cum_step=args.cum_base + step,
                   win_rate=round(bwin, 4), survival_rate=round(v["survival_rate"], 4),
                   safe_rate=round(v["safe_rate"], 4), best=round(best["score"], 4),
                   keep_metric=args.keep_metric, selected_score=round(bscore, 4),
                   bench_seeds=bench_seeds,
                   keep_ok=int(keep_ok), improved=int(improved),
                   bench_sparc=round(bsparc, 3), bench_dealt=round(bdealt, 4), bench_taken=round(btaken, 4),
                   bench_ratio=round(bratio, 3), bench_margin=round(bmargin, 4),
                   bench_judge=round(bjudge, 4), bench_min_margin=round(bmin_margin, 4),
                   bench_min_judge=round(bmin_judge, 4), bench_min_dealt=round(bmin_dealt, 4),
                   bench_max_early=round(bmax_early, 4), bench_max_peak_pen=round(bmax_peak_pen, 4),
                   idle_penalty=round(idle_pen, 4),
                   clean=round(v["clean"], 4), trade=round(v["trade"], 4), fire=round(v["fire"], 3),
                   closing=round(v["closing"], 3), fleeing=round(v["fleeing"], 3),
                   dist=round(v["dist"], 2), alive=round(v["alive"], 1),
                   sparc_close=round(v["sparc_close"], 2), sparc_med=round(v["sparc_med"], 2),
                   sparc_far=round(v["sparc_far"], 2),
                   ac_peak_z=round(v["ac_peak_z"], 3), ac_airborne=round(v["ac_airborne"], 3),
                   ac_peak_pen=round(v["ac_peak_pen"], 3), ac_idle=round(v["ac_idle"], 3),
                   ac_dmg_early=round(v["ac_dmg_early"], 3), ac_upright_dmg=round(v["ac_upright_dmg"], 3),
                   ac_grounded_dmg=round(v["ac_grounded_dmg"], 3), ac_uprightness=round(v["ac_uprightness"], 3))
        open(bjson, "a").write(json.dumps(rec) + "\n")
        _ke.emit_metric("benchmark", **rec)
        json.dump(dict(tag=args.tag, cum_step=args.cum_base + step, wall_s=round(time.time()-t0, 0),
                       keep_metric=args.keep_metric, best_bench=round(best["score"], 4),
                       best_score=round(best["score"], 4), best_win=round(best["win"], 4),
                       best_sparc=round(best["sparc"], 3),
                       best_dealt=round(best["dealt"], 4), best_taken=round(best["taken"], 4),
                       best_margin=round(best["margin"], 4),
                       best_judge=round(best["judge"], 4),
                       best_min_margin=round(best["min_margin"], 4),
                       best_min_judge=round(best["min_judge"], 4),
                       best_min_dealt=round(best["min_dealt"], 4),
                       best_max_early=round(best["max_early"], 4),
                       best_max_peak_pen=round(best["max_peak_pen"], 4),
                       best_ac_dmg_early=round(best["early"], 4),
                       best_ac_airborne=round(best["airborne"], 4),
                       best_ac_idle=round(best["idle"], 4),
                       best_ac_uprightness=round(best["uprightness"], 4),
                       best_step=best["step"],
                       last_score=round(bscore, 4), last_win=round(bwin, 4),
                       last_bench=round(bsparc, 3), last_dealt=round(bdealt, 4),
                       last_taken=round(btaken, 4), last_ratio=round(bratio, 3),
                       last_margin=round(bmargin, 4), last_judge=round(bjudge, 4),
                       last_min_margin=round(bmin_margin, 4), last_min_judge=round(bmin_judge, 4),
                       last_min_dealt=round(bmin_dealt, 4), last_max_early=round(bmax_early, 4),
                       last_max_peak_pen=round(bmax_peak_pen, 4)),
                  open(OUT / f"{args.tag}_state.json", "w"))
        print(f"  [bench] step {step:>9,} WIN {bwin:.2f} surv {v['survival_rate']:.2f} safe {v['safe_rate']:.2f} | "
              f"sparc {bsparc:6.2f} ratio {bratio:.2f} margin {bmargin:+.3f} "
              f"minMargin {bmin_margin:+.3f} judge {bjudge:+.2f} minJudge {bmin_judge:+.2f} "
              f"fire {v['fire']:.2f} "
              f"sparc[c/m/f] {v['sparc_close']:.0f}/{v['sparc_med']:.0f}/{v['sparc_far']:.0f} | "
              f"keep-{args.keep_metric} {bscore:.2f} best {best['score']:.2f}"
              f"{'  *NEW-BEST*' if improved else ''}"
              f"{'' if keep_ok else '  (keep-gated)'}", flush=True)
        cheats = ([] + (["LAUNCH"] if v["ac_peak_z"] > 0.45 else [])
                  + (["airborne"] if v["ac_airborne"] > 0.05 else [])
                  + (["penetration"] if v["ac_peak_pen"] > 0.05 else [])
                  + (["idle"] if v["ac_idle"] > 0.5 else [])
                  + (["clash-early"] if v["dealt"] > 1e-4 and v["ac_dmg_early"] > 0.7 else [])
                  + (["airborne-dmg"] if v["dealt"] > 1e-4 and v["ac_grounded_dmg"] < 0.5 else []))
        if v["ac_uprightness"] < 0.5:
            cheats.append("sprawled")
        print(f"  [anti-cheat] upright {v['ac_uprightness']:.2f} peak_z {v['ac_peak_z']:.2f} "
              f"airb {v['ac_airborne']:.2f} pen {v['ac_peak_pen']:.3f} idle {v['ac_idle']:.2f} "
              f"earlyDmg {v['ac_dmg_early']:.2f} grndDmg {v['ac_grounded_dmg']:.2f}"
              + (f"  ## CHEAT-FLAGS: {cheats}" if cheats else "  -> clean"), flush=True)
    def _head_factory(observation_size, action_size, preprocess_observations_fn, **_):
        return head_only_network_factory(observation_size, action_size, preprocess_observations_fn)
    network_factory = _head_factory if args.policy_train_scope == "head" else None
    train_kwargs = {}
    if network_factory is not None:
        train_kwargs["network_factory"] = network_factory
    ppo.train(environment=env, num_timesteps=args.steps, num_evals=n_eval,
              episode_length=300, num_envs=args.envs, batch_size=args.batch,
              num_minibatches=args.minibatches, unroll_length=args.unroll, num_updates_per_batch=args.updates,
              learning_rate=args.lr, entropy_cost=args.entropy, discounting=0.97, reward_scaling=1.0,
              clipping_epsilon=args.clip_epsilon, desired_kl=args.desired_kl,
              max_grad_norm=args.max_grad_norm, learning_rate_schedule=args.lr_schedule,
              normalize_observations=True, seed=0, progress_fn=prog, policy_params_fn=ck,
              restore_params=restore, **train_kwargs)
    train_s = time.time() - t0
    ratio = last["dealt"] / max(last["taken"], 1e-6)
    competent = last["dealt"] > last["taken"] and last["dealt"] > 0.02
    METRIC(stage="fighter_train", train_s=f"{train_s:.1f}", compile_s=f"{tm['first_eval'] or 0:.1f}",
           env_steps=last["step"], cum_step=args.cum_base + last["step"],
           throughput=f"{last['step']/max(train_s,1e-6):.0f}",
           final_sparc=f"{last['r']:.2f}", dealt=f"{last['dealt']:.4f}", taken=f"{last['taken']:.4f}",
           dealt_taken_ratio=f"{ratio:.2f}", competent=int(competent), warm=int(restore is not None),
           keep_metric=args.keep_metric,
           best_score=f"{best['score']:.3f}" if benches else "off",
           best_win=f"{best['win']:.3f}" if benches else "off",
           best_margin=f"{best['margin']:.4f}" if benches else "off",
           best_judge=f"{best['judge']:.3f}" if benches else "off",
           best_step=best["step"])
    print(f"FIGHTER: final dealt {last['dealt']:.4f} vs taken {last['taken']:.4f} (ratio {ratio:.2f}); "
          f"best held-out {args.keep_metric.upper()} {best['score']:.3f} "
          f"(win {best['win']:.3f}, sparc {best['sparc']:.2f}) @ step {best['step']:,} "
          f"(-> {args.tag}_best.pkl). The sparse verdict (win-rate) is the judgment; the dense "
          f"decomposition is the coach's signal.", flush=True)


if __name__ == "__main__":
    main()
