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

import argparse, os, pickle, sys, time
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
                 fire_shaping=0.0, rod_reach=0.30, opponent="passive", opp_infer=None, opp_params=None):
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
        self._mx = mjx.put_model(m); self._fs = frame_skip; self._nu = m.nu
        self._q0 = jnp.array(m.qpos0)
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
        # legs-as-weapons: any A leg geom vs any B body geom (and vice-versa). The pneumatic ROD
        # is a SEPARATE mask (its damage gets the tip-speed multiplier).
        self._Aleg = mk(lambda n: leg_geom(n, "A"))
        self._Abody = mk(lambda n: n.startswith("A_") and n != "floor")
        self._Bleg = mk(lambda n: leg_geom(n, "B"))
        self._Bbody = mk(lambda n: n.startswith("B_") and n != "floor")
        self._Arod = mk(lambda n: n.startswith("A_") and n.endswith("_rod"))
        self._Brod = mk(lambda n: n.startswith("B_") and n.endswith("_rod"))
        self._Arod_gids = jnp.array([g for g in range(m.ngeom)
                                     if gn(g).startswith("A_") and gn(g).endswith("_rod")], dtype=int)
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
        self._n_hinge_b = len(B_hinge)
        self._B_strike_local = jnp.array([B_acts.index(a) for a in B_strike], dtype=int)
        Bj = [int(m.actuator_trnid[a, 0]) for a in B_hinge]
        self._Bqa = jnp.array([int(m.jnt_qposadr[j]) for j in Bj], dtype=int)
        self._Bda = jnp.array([int(m.jnt_dofadr[j]) for j in Bj], dtype=int)
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
        self._hinge = jnp.array(m.jnt_type == mujoco.mjtJoint.mjJNT_HINGE)
        self._obs_size = LOCO_OBS + 6
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

    def _obs(self, dx, d):
        loco = jnp.concatenate([dx.qpos[self._Aqa], dx.qvel[self._Ada], dx.xquat[self._At],
                                dx.qvel[self._ArD:self._ArD + 6], dx.xpos[self._At][2:3], d])
        opp = jnp.concatenate([dx.xpos[self._Bt] - dx.xpos[self._At], dx.qvel[self._BrD:self._BrD + 3]])
        return jnp.concatenate([loco, opp])

    def _obsB(self, dx, d):
        """Opponent's obs — the SAME layout as A's but B-centric (B=me, A=opponent), so a frozen
        snapshot of OUR fighter drives B unchanged (symmetric self-play)."""
        loco = jnp.concatenate([dx.qpos[self._Bqa], dx.qvel[self._Bda], dx.xquat[self._Bt],
                                dx.qvel[self._BrD:self._BrD + 6], dx.xpos[self._Bt][2:3], d])
        opp = jnp.concatenate([dx.xpos[self._At] - dx.xpos[self._Bt], dx.qvel[self._ArD:self._ArD + 3]])
        return jnp.concatenate([loco, opp])

    _MET0 = None
    def _metrics0(self):
        return {"dealt": jnp.zeros(()), "taken": jnp.zeros(()), "closing": jnp.zeros(()),
                "fleeing": jnp.zeros(()), "sparc": jnp.zeros(()), "dist": jnp.zeros(()),
                "approach": jnp.zeros(()), "clean_hit": jnp.zeros(()), "trade": jnp.zeros(()),
                "disengage": jnp.zeros(()), "fire": jnp.zeros(())}

    def _planar_dist(self, dx):
        return jnp.linalg.norm((dx.xpos[self._Bt] - dx.xpos[self._At])[:2])

    def _place(self, qpos, sep, theta):
        # A at origin; B at BEARING theta, distance sep -> the policy must approach + strike
        # from a varied angle (encourages different angles of attack, not just head-on).
        bx = sep * jnp.cos(theta); by = sep * jnp.sin(theta)
        return (qpos.at[self._Arq].set(0.0).at[self._Arq + 1].set(0.0)
                    .at[self._Brq].set(bx).at[self._Brq + 1].set(by))

    def _info(self, d, dx, dp):
        # prev_dealt: damage dealt last step — gates the post-hit disengage bonus (only reward
        # retreating right AFTER landing a hit, not idle fleeing).
        info = {"design": d, "prev_dist": self._planar_dist(dx), "prev_dealt": jnp.zeros(())}
        if self._rg:
            info["dp"] = dp
        return info

    def reset(self, rng):
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
        dealt_f = jnp.clip(dealt / DAMAGE_REF, 0, 1); taken_f = jnp.clip(taken / DAMAGE_REF, 0, 1)
        rel = (dx.xpos[self._Bt] - dx.xpos[self._At])[:2]; dist = jnp.linalg.norm(rel); n = dist + 1e-6
        toward = jnp.dot(dx.qvel[self._ArD:self._ArD + 2], rel) / n
        clos = jnp.clip(toward / 2, 0, 1); flee = jnp.clip(-toward / 2, 0, 1)
        up = 1.0 - 2.0 * (dx.qpos[4] ** 2 + dx.qpos[5] ** 2)
        # the real SPARC objective (force/penetration-weighted damage + aggression):
        sparc = 6.0 * (dealt_f - taken_f) + 5.0 * (clos - flee)
        # dense close→strike SHAPING (annealed via self._shaping; legs-as-weapons so getting
        # close + a limb on B already scores): close the bodies, AIM A LIMB at B's torso, and
        # a hit accelerator. The leg-proximity term gives the missing gradient to "land a hit".
        legdist = jnp.min(jnp.linalg.norm(dx.xpos[self._Astrike] - dx.xpos[self._Bt], axis=1))
        shaped = self._shaping * (-0.15 * dist - 0.20 * legdist + 3.0 * dealt_f)
        # potential-based APPROACH reward: pay for distance CLOSED this step (approach
        # velocity). Dense + available before any hit -> "forces" learning to close the gap.
        approach = state.info["prev_dist"] - dist                 # >0 when the gap shrank
        # WIN-EXCHANGES asymmetry (headroom above the dealt≈taken plateau): a CLEAN hit (landed
        # while not being hit) pays more than a TRADE (mutual contact), and retreating right
        # after a hit (gated on prev_dealt, so it isn't idle fleeing) is rewarded.
        clean = dealt_f * (1.0 - taken_f)               # landed WHILE NOT being hit
        trade = jnp.minimum(dealt_f, taken_f)           # mutual contact (drive DOWN)
        outward = jnp.clip(-toward / 2, 0, 1)           # moving AWAY from the opponent
        disengage = state.info["prev_dealt"] * outward  # retreat right after landing a hit
        reward = (sparc + shaped + self._approach_w * approach + 0.3 * up + 0.1
                  + self._clean_w * clean - self._trade_w * trade + self._dis_w * disengage
                  + self._fire_shaping * fire_aim - fire_cost)
        done = jnp.where(dx.xpos[self._At][2] < 0.18, 1.0, 0.0)
        # MERGE into the existing metrics dict (brax's Evaluator injects a 'reward' key —
        # replacing the dict drops it and breaks the scan-carry pytree).
        metrics = {**state.metrics, "dealt": dealt_f, "taken": taken_f, "closing": clos,
                   "fleeing": flee, "sparc": sparc, "dist": dist, "approach": approach,
                   "clean_hit": clean, "trade": trade, "disengage": disengage, "fire": fire_act}
        return state.replace(pipeline_state=dx, obs=self._obs(dx, d), reward=reward, done=done,
                             metrics={**metrics},
                             info={**state.info, "prev_dist": dist, "prev_dealt": dealt_f})


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
        pad = obs_dim - LOCO_OBS
        c = norm.count                                   # brax UInt64 = {hi, lo}: value = hi*2^32 + lo
        cval = float(jnp.asarray(c.hi)) * (2.0 ** 32) + float(jnp.asarray(c.lo))
        nkw = {}
        for fn in ("mean", "std", "summed_variance"):
            v = getattr(norm, fn, None)
            if hasattr(v, "ndim") and v.ndim >= 1 and v.shape[0] == LOCO_OBS:
                # new (opponent) dims start standardized: mean 0, std 1, summed_variance=count (var~1)
                fill = (jnp.zeros(pad) if fn == "mean" else
                        jnp.ones(pad) if fn == "std" else
                        jnp.full((pad,), max(cval, 1.0)))
                nkw[fn] = jnp.concatenate([v, fill])
        norm = norm.replace(**nkw)
        pad_leaf = lambda x: (jnp.concatenate([x, jnp.zeros((pad,) + x.shape[1:])], 0)
                              if (hasattr(x, "ndim") and x.ndim >= 1 and x.shape[0] == LOCO_OBS) else x)
        nets = [jax.tree_util.tree_map(pad_leaf, n) for n in nets]
        if nets:                                         # grow the policy (nets[0]) action head only
            nets[0] = _grow_action_head(nets[0], act_dim)
        print(f"WARM-START ok: obs {LOCO_OBS}->{obs_dim} (count={cval:.0f}, normalizer + {len(nets)} nets)", flush=True)
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


def build_benchmark(bench_env, n_epis, steps, seed=20240601):
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
        inf = make_inf(params, deterministic=True)
        def ep(k):
            st = bench_env.reset(k)
            d0 = jnp.linalg.norm((st.pipeline_state.xpos[bench_env._Bt]
                                  - st.pipeline_state.xpos[bench_env._At])[:2])   # initial separation
            def stp(carry, _):
                s, key = carry; key, sk = jax.random.split(key)
                a, _ = inf(s.obs, sk); s = bench_env.step(s, a); alive = 1.0 - s.done
                m = s.metrics
                return (s, key), jnp.array([m["sparc"] * alive, m["dealt"] * alive, m["taken"] * alive,
                                            m["clean_hit"] * alive, m["trade"] * alive, m["fire"] * alive,
                                            m["closing"] * alive, m["fleeing"] * alive, m["dist"] * alive, alive])
            (_, _), outs = jax.lax.scan(stp, (st, k), None, length=steps)
            return outs.sum(0), d0
        per_ep, d0 = jax.vmap(ep)(keys)                       # per_ep:(n,10) ep-totals; d0:(n,) initial sep
        agg = per_ep.mean(0)                                  # the 10 decomposition signals
        spe = per_ep[:, 0]                                    # per-episode SPARC sum, for range bins
        bm = lambda mask: jnp.sum(spe * mask) / jnp.maximum(jnp.sum(mask), 1.0)
        bins = jnp.array([bm(d0 < 0.6), bm((d0 >= 0.6) & (d0 < 0.9)), bm(d0 >= 0.9)])  # close/med/far SPARC
        return jnp.concatenate([agg, bins])                  # 13 values (see BENCH_KEYS)
    return bench


# names for build_benchmark()'s 13-value vector (the held-out combat decomposition + range profile)
BENCH_KEYS = ["sparc", "dealt", "taken", "clean", "trade", "fire", "closing", "fleeing", "dist",
              "alive", "sparc_close", "sparc_med", "sparc_far"]


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
    ap.add_argument("--lean-contacts", action="store_true", help="F-SPEED: reduced-collision fight scene")
    # WIN-EXCHANGES reward asymmetry (STEP 2b) — default 0 = the contact-forced fighter's reward.
    ap.add_argument("--clean-weight", type=float, default=0.0, help="+w·dealt·(1−taken): reward un-traded hits")
    ap.add_argument("--trade-weight", type=float, default=0.0, help="−w·min(dealt,taken): punish mutual contact")
    ap.add_argument("--disengage-weight", type=float, default=0.0,
                    help="+w·prev_dealt·outward_vel: reward retreat right AFTER a hit (anneal — don't make it flee)")
    ap.add_argument("--fire-shaping", type=float, default=0.0,
                    help="dense reward for firing a rod when its tip is aimed/in-range at B (anneal as hits take over)")
    # Held-out BENCHMARK eval (the honest monotone-improvement curve + keep-best selection). Run
    # at every eval on a FIXED config (comparable across curriculum phases), independent of the
    # shaped training reward — reads the SPARC/dealt/taken metrics, not `reward`.
    ap.add_argument("--bench-epis", type=int, default=16, help="benchmark episodes (fixed held-out seeds)")
    ap.add_argument("--bench-steps", type=int, default=200, help="benchmark steps/episode")
    ap.add_argument("--bench-sep-lo", type=float, default=0.4, help="benchmark fixed start-sep low")
    ap.add_argument("--bench-sep-hi", type=float, default=1.2, help="benchmark fixed start-sep high")
    ap.add_argument("--bench-az", type=float, default=3.14159, help="benchmark azimuth range (all angles)")
    ap.add_argument("--no-benchmark", action="store_true", help="disable benchmark eval / keep-best")
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
                                              fire_shaping=args.fire_shaping,
                                              opponent=args.opponent, opp_infer=opp_infer)
    METRIC(stage="adv_env_build", t_s=f"{time.time()-t_env:.1f}",
           obs=env.observation_size, act=env.action_size, striker=int(env._has_striker),
           opponent=args.opponent)
    print(f"adversarial env: obs={env.observation_size} act(A)={env.action_size} "
          f"striker={env._has_striker} opponent={args.opponent}", flush=True)
    # held-out benchmark (fixed config + fixed reference opponent) → honest improvement curve + keep-best
    bench = None
    if not args.no_benchmark:
        bench_opponent = "frozen" if bench_opp is not None else "passive"
        bench_env = AdversarialEnv(self_collision=not args.lean_contacts, frame_skip=args.frame_skip,
                                   sep_lo=args.bench_sep_lo, sep_hi=args.bench_sep_hi, azimuth=args.bench_az,
                                   striker=striker, opponent=bench_opponent, opp_infer=bench_opp)
        bench = build_benchmark(bench_env, args.bench_epis, args.bench_steps)
    restore = (warm_start(args.resume, env.observation_size, env.action_size)
               if args.resume and os.path.exists(args.resume) else None)
    METRIC(stage="warm_start", ok=int(restore is not None),
           resume=os.path.basename(args.resume) if args.resume else "none")
    import json
    t0 = time.time(); csv = OUT / "adv_metrics.csv"; csv.write_text("step,reward,sec\n")
    fjson = OUT / "fight_metrics.jsonl"; fjson.write_text("")          # F0: the six trackers
    bjson = OUT / f"{args.tag}_benchmark.jsonl"; bjson.write_text("")  # the honest monotone curve
    tm = {"first_eval": None}; last = {"r": float("nan"), "step": 0, "dealt": 0.0, "taken": 0.0}
    best = {"bench": -1e30, "step": -1}                                # keep-best (monotone by construction)
    def g(m, k): return float(m.get(f"eval/episode_{k}", 0.0))
    def save(obj, name):
        try: pickle.dump(obj, open(OUT / name, "wb"))
        except Exception as e: print(f"  [ck] save {name} failed: {e}", flush=True)
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
        save(params, f"{args.tag}_ckpt.pkl")
        if bench is None: return
        try:
            vals = np.asarray(bench(params))
            v = {k: float(vals[i]) for i, k in enumerate(BENCH_KEYS)}   # full held-out decomposition
            bsparc, bdealt, btaken = v["sparc"], v["dealt"], v["taken"]
        except Exception as e:
            print(f"  [bench] failed: {type(e).__name__}: {e}", flush=True); return
        bratio = bdealt / max(btaken, 1e-6); improved = bsparc > best["bench"]
        if improved:
            best.update(bench=bsparc, step=step); save(params, f"{args.tag}_best.pkl")
        rec = dict(step=step, cum_step=args.cum_base + step, bench_sparc=round(bsparc, 3),
                   bench_dealt=round(bdealt, 4), bench_taken=round(btaken, 4), bench_ratio=round(bratio, 3),
                   best=round(best["bench"], 3), improved=int(improved),
                   clean=round(v["clean"], 4), trade=round(v["trade"], 4), fire=round(v["fire"], 3),
                   closing=round(v["closing"], 3), fleeing=round(v["fleeing"], 3),
                   dist=round(v["dist"], 2), alive=round(v["alive"], 1),
                   sparc_close=round(v["sparc_close"], 2), sparc_med=round(v["sparc_med"], 2),
                   sparc_far=round(v["sparc_far"], 2))
        open(bjson, "a").write(json.dumps(rec) + "\n")
        _ke.emit_metric("benchmark", **rec)
        json.dump(dict(tag=args.tag, cum_step=args.cum_base + step, wall_s=round(time.time()-t0, 0),
                       best_bench=round(best["bench"], 3), best_step=best["step"],
                       last_bench=round(bsparc, 3), last_ratio=round(bratio, 3)),
                  open(OUT / f"{args.tag}_state.json", "w"))
        print(f"  [bench] step {step:>9,} sparc {bsparc:7.2f} ratio {bratio:.2f} "
              f"clean {v['clean']:.3f} trade {v['trade']:.3f} fire {v['fire']:.2f} "
              f"sparc[c/m/f] {v['sparc_close']:.1f}/{v['sparc_med']:.1f}/{v['sparc_far']:.1f} | "
              f"best {best['bench']:7.2f}{'  *NEW-BEST*' if improved else ''}", flush=True)
    ppo.train(environment=env, num_timesteps=args.steps, num_evals=n_eval,
              episode_length=300, num_envs=args.envs, batch_size=args.batch,
              num_minibatches=args.minibatches, unroll_length=args.unroll, num_updates_per_batch=args.updates,
              learning_rate=args.lr, entropy_cost=args.entropy, discounting=0.97, reward_scaling=1.0,
              normalize_observations=True, seed=0, progress_fn=prog, policy_params_fn=ck,
              restore_params=restore)
    train_s = time.time() - t0
    ratio = last["dealt"] / max(last["taken"], 1e-6)
    competent = last["dealt"] > last["taken"] and last["dealt"] > 0.02
    METRIC(stage="fighter_train", train_s=f"{train_s:.1f}", compile_s=f"{tm['first_eval'] or 0:.1f}",
           env_steps=last["step"], cum_step=args.cum_base + last["step"],
           throughput=f"{last['step']/max(train_s,1e-6):.0f}",
           final_sparc=f"{last['r']:.2f}", dealt=f"{last['dealt']:.4f}", taken=f"{last['taken']:.4f}",
           dealt_taken_ratio=f"{ratio:.2f}", competent=int(competent), warm=int(restore is not None),
           best_bench=f"{best['bench']:.2f}" if bench is not None else "off", best_step=best["step"])
    print(f"FIGHTER: final dealt {last['dealt']:.4f} vs taken {last['taken']:.4f} (ratio {ratio:.2f}); "
          f"best benchmark SPARC {best['bench']:.2f} @ step {best['step']:,} "
          f"(-> {args.tag}_best.pkl). competent = {competent}. Decomposition, not the scalar, "
          f"is the verdict (a survivor has dealt≈0).", flush=True)


if __name__ == "__main__":
    main()
