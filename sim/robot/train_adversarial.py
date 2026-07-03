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
import ppo_nets as ppo_networks  # shared (512,256,128) + small-final-init factory (audit item 5)

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
from constants import LOCO_OBS  # V.1/V.4: layout constant lives in constants.py
DAMAGE_REF = 0.05
STRIKE_KINETIC = 0.1  # rod damage multiplier per m/s of slide speed: hit at ~11 m/s ≈ ×2.1 damage


def METRIC(**kw):
    print("METRIC " + " ".join(f"{k}={v}" for k, v in kw.items()), flush=True)
    _ke.emit_metric(str(kw.get("stage", "metric")), **{k: v for k, v in kw.items() if k != "stage"})


class AdversarialEnv(Env):
    def __init__(self, frame_skip=5, shaping=1.0, sep=1.0, self_collision=True,
                 sep_lo=None, sep_hi=None, approach_weight=0.0, azimuth=0.0,
                 reality_gap=False, n_worlds=64,
                 action_mode="pd", pd_action_scale=0.4, history_len=3,
                 combat_scale=1.0, loco_speed=0.0, loco_track_w=8.0, loco_drill_frac=0.0,
                 alive_bonus=0.1, gait_airtime_w=0.0, gait_slip_w=0.0, gait_pose_w=0.0,
                 ko_weight=0.0, ko_alpha=1.0, ko_done=False, ko_min_dealt=0.02,
                 clean_weight=0.0, trade_weight=0.0, disengage_weight=0.0, striker=None,
                 fire_shaping=0.0, rod_reach=0.30, opponent="passive", opp_infer=None, opp_params=None,
                 walker_infer=None, walker_speed=0.25,
                 upright_weight=0.3, energy_penalty=0.0, airborne_penalty=0.0, height_weight=0.0,
                 move_weight=0.0, early_hit_penalty=0.0, min_hit_step=0, taken_weight=0.0,
                 flee_penalty=0.0, close_bonus=0.0, close_radius=0.45, damage_bonus=0.0,
                 face_opponent=False, engage_obs=False, contact_obs=False, face_weight=0.0,
                 penetration_penalty=0.0, penetration_tol=0.045,
                 reset_bank_seed=None, reset_bank_epis=0,
                 lidar=False, lidar_n_rays=128, lidar_n_vertical=16,
                 lidar_max_range=2.0, lidar_noise_sigma=0.015,
                 lidar_dropout_rate=0.02, lidar_latency_steps=0,
                 lidar_frame_stack=3,
                 hierarchical=False, gate_weight=1.0, gate_threshold=0.3,
                 her_coefficient=0.0, her_sigma=0.15, her_fraction=0.5,
                 rnd_coefficient=0.0, rnd_hidden_dim=128, rnd_output_dim=64,
                 rnd_lr=1e-3, rnd_clip=10.0, rnd_seed=0, rnd_feature="tactical",
                 require_closing=False, closing_eps=0.05,
                 stationary_damage_penalty=0.0, oscillation_penalty=0.0,
                 move_eps=0.1, opponent_script=0.0,
                 cpg_control=False, cpg_speed=0.9, cpg_residual_scale=0.5):
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
        # LIDAR sensor configuration: when enabled, the actor observation contains
        # a simulated lidar depth scan instead of privileged opponent state. The
        # critic still receives privileged engage/contact features (asymmetric
        # actor-critic for sim-to-real).
        self._lidar = bool(lidar)
        self._lidar_n_rays = int(lidar_n_rays)
        self._lidar_n_vertical = int(lidar_n_vertical)
        self._lidar_n_total = int(lidar_n_rays + lidar_n_vertical)
        self._lidar_max_range = float(lidar_max_range)
        self._lidar_noise_sigma = float(lidar_noise_sigma)
        self._lidar_dropout_rate = float(lidar_dropout_rate)
        self._lidar_latency = int(lidar_latency_steps)
        self._lidar_stack = int(lidar_frame_stack)
        self._lidar_scan_dim = self._lidar_n_total * max(1, self._lidar_stack)
        # Noise/dropout are only applied when configured; with both off the scan
        # is a deterministic clean depth image (benchmark/eval determinism).
        self._lidar_stochastic = (self._lidar_noise_sigma > 0) or (self._lidar_dropout_rate > 0)
        # HIERARCHICAL policy: adds a gate logit to the action space that modulates
        # the striker DOFs. When the gate is closed (sigmoid(gate)~0), only approach
        # (hinge) actions are expressed. When open (~1), strike actions fire.
        # This creates a natural approach/strike decomposition in the learned policy.
        self._hierarchical = bool(hierarchical)
        self._gate_weight = float(gate_weight)
        self._gate_threshold = float(gate_threshold)
        # Goal-conditioned obs + achievement reward. The hindsight RELABELING that
        # makes this true HER is applied over collected rollouts in main() via
        # her_goal.install_her_relabel(); the env supplies her_goal/her_achieved
        # (in info) so the relabel pass can run. See her_goal.py.
        self._her_coeff = float(her_coefficient)
        self._her_sigma = float(her_sigma)
        self._her_fraction = float(her_fraction)
        # RND intrinsic motivation (TRUE RND): a functional predictor whose params
        # + Adam state are carried PER-ENV in state.info and updated every step on
        # the next-state proprioceptive features. The novelty bonus therefore drops
        # as states become familiar — wired directly into the env reward.
        self._rnd_coeff = float(rnd_coefficient)
        self._rnd_clip = float(rnd_clip)
        # RUNG 3: RND on TACTICAL descriptors (distance/bearing/approach/lateral/contact/
        # tip-speed) instead of raw proprioception — so curiosity rewards reaching new
        # tactical SITUATIONS, not twitching joints. Tactical feature dim = 8.
        self._rnd_feature = str(rnd_feature)
        self._rnd_feat_dim = 8 if self._rnd_feature == "tactical" else LOCO_OBS
        self._rnd_coeff = float(rnd_coefficient)
        self._rnd = None
        if self._rnd_coeff > 0:
            from rnd_curiosity import make_rnd
            self._rnd = make_rnd(feature_dim=self._rnd_feat_dim, hidden_dim=int(rnd_hidden_dim),
                                 output_dim=int(rnd_output_dim), lr=float(rnd_lr),
                                 key=jax.random.PRNGKey(int(rnd_seed)))
        # RUNG 2b: outcome-grounded reward shaping to kill the stand-still exploit.
        self._require_closing = bool(require_closing)   # damage only credited while closing
        self._closing_eps = float(closing_eps)          # min approach velocity to count a hit
        self._stationary_pen = float(stationary_damage_penalty)  # penalize damage while not moving
        self._oscillation_pen = float(oscillation_penalty)       # penalize effort while not moving
        self._move_eps = float(move_eps)                # planar speed below this = "not moving"
        # RUNG 4: scripted ACTIVE opponent — B pursues A (+ strikes when close) so the
        # benchmark/curriculum is no longer a limp dummy. 0 = off (passive).
        self._opponent_script = float(opponent_script)
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
        # C.2 walker-pursuer: a frozen COMMANDED-ENV walker (pdval lineage) drives B —
        # it sees the commanded obs layout built B-centrically with cmd = unit(A−B)·speed,
        # so standing still now LOSES structurally (the opponent walks over and hits you)
        # instead of via gate patches. Replaces the open-loop sinusoid "pursuer".
        self._walker_infer = walker_infer
        self._walker_speed = float(walker_speed)
        if self._opp == "walker":
            from commanded_env import DEFAULT_FAST_DESIGN  # the design pdval trained under
            self._walker_design = jnp.array(DEFAULT_FAST_DESIGN, dtype=jnp.float32)
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
                        striker_b=self._armed_b,
                        lidar=self._lidar, lidar_n_rays=self._lidar_n_rays,
                        lidar_n_vertical=self._lidar_n_vertical,
                        lidar_max_range=self._lidar_max_range))
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
        # PD ACTION MODE (B.1, audit item 2b): hinge actions are position TARGETS around
        # the stand pose, turned into torque by a PD recomputed EVERY physics substep
        # (250 Hz) inside the frame_skip loop — the proven legged-RL action space.
        # Direct 50 Hz torque (the old default) is the best-documented wrong one
        # (Peng & van de Panne 2017; every Go1/Go2 MJX recipe is PD). Striker + gate
        # channels stay direct. `torque` mode preserves the legacy pathway.
        self._action_mode = str(action_mode)
        self._pd_act_scale = float(pd_action_scale)
        self._hist_len = int(history_len)
        self._hinge_localA = jnp.array([A_acts.index(a) for a in A_hinge], dtype=int)
        self._standA = self._q0[self._Aqa]         # canonical stance (targets are offsets from it)
        self._jrA = jnp.array([list(m.jnt_range[j]) for j in Aj])
        # torque mapping divisor = GEAR (delivered torque = gear × ctrl); forcerange
        # only documents intent — trusting it is the pattern that hid the 8%-torque bug
        _gA = np.array([float(m.actuator_gear[a, 0]) for a in A_hinge])
        _fA = np.array([float(m.actuator_forcerange[a, 1]) for a in A_hinge])
        self._gearA = jnp.array(np.where(_gA > 0, _gA, _fA))
        from constants import PD_KD, PD_KP  # V.1 single read point
        self._pd_kp_act = PD_KP
        self._pd_kd_act = PD_KD
        # A's foot geoms — critic-only contact booleans (a real robot has no foot
        # sensors; the actor must infer stance/swing from its history obs instead)
        self._Afeet_gids = jnp.array([g for g in range(m.ngeom)
                                      if gn(g).startswith("A_") and gn(g).endswith("_foot")], dtype=int)
        # WALK-THEN-FIGHT curriculum (B.3, audit item 4). Defaults preserve the legacy
        # reward exactly: combat_scale=1, no loco tracking, no gait terms, no drills.
        # The curriculum anneals: loco_speed 0.10→0.6 (phase-0 velocity tracking toward
        # the opponent, exp-kernel — maximized only at SUSTAINED matched velocity, so
        # oscillation scores ~0), combat_scale 0→1 gated on the behavior benchmark,
        # alive_bonus down as combat comes in, and keeps loco_drill_frac≈0.25 forever
        # (a quarter of episodes stay pure walking drills so combat gradients can't
        # erase the gait).
        self._combat_scale = float(combat_scale)
        self._loco_speed = float(loco_speed)
        self._loco_track_w = float(loco_track_w)
        self._drill_frac = float(loco_drill_frac)
        self._alive_bonus = float(alive_bonus)
        self._airtime_w = float(gait_airtime_w)
        self._slip_w = float(gait_slip_w)
        self._pose_w = float(gait_pose_w)
        # C.1 sparse zero-sum KO (audit item 9): the trained objective finally contains
        # the outcome keep-best selects on. KO pays ONLY if we actually dealt damage
        # (dealt_cum gate — the passive B sags on its springs; an ungated KO pays for
        # waiting). ko_done MUST stay False in benchmark envs or survived_bout inverts
        # win_rate. total = α·dense + (1−α)·outcome (Bansal et al.: anneal α 1→0.2).
        self._ko_w = float(ko_weight)
        self._ko_alpha = float(ko_alpha)
        self._ko_done = bool(ko_done)
        self._ko_min_dealt = float(ko_min_dealt)
        # CPG-PD LOCOMOTION CONTROL (verified the body walks under the CPG): the policy outputs a
        # RESIDUAL; the legs are driven by the directional CPG gait toward B (the prior walks) + the
        # residual via PD; the policy controls the striker/gate directly and learns to steer + strike.
        self._cpg_control = bool(cpg_control)
        self._cpg_speed = float(cpg_speed)
        self._cpg_res_scale = float(cpg_residual_scale)
        if self._cpg_control:
            from cpg_teacher import make_directional_params_from_env
            self._cpg = make_directional_params_from_env()
            self._cpg_base_freq = float(np.asarray(self._cpg.backward.freq))
            hinge_order = {an(a): i for i, a in enumerate(A_hinge)}        # name -> 0..n_hinge-1
            self._cpg_idx = jnp.array([[hinge_order[f"A_{lg}_{jt}_m"] for jt in ("abd", "flex", "knee")]
                                       for lg in ("FL", "FR", "RL", "RR")], dtype=int)
            self._hinge_local = jnp.array([A_acts.index(a) for a in A_hinge], dtype=int)  # action-vec slots
            self._cpg_stand = self._q0[self._Aqa]                          # (n_hinge,) leg stand pose
            self._cpg_jr = jnp.array([list(m.jnt_range[j]) for j in Aj])   # (n_hinge,2) leg joint ranges
            self._cpg_tmax = jnp.array([float(m.actuator_forcerange[a, 1]) for a in A_hinge])  # (n_hinge,)
            self._cpg_dt = float(frame_skip) * float(m.opt.timestep)
            self._cpg_vmax = float(os.environ.get("CMD_VMAX", "1.2"))
            self._pd_kp = float(os.environ.get("CMD_PD_KP", "30.0"))
            self._pd_kd = float(os.environ.get("CMD_PD_KD", "1.0"))
            self._pd_scale = float(os.environ.get("CMD_PD_SCALE", "1.0"))
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
        # B mirror of the PD action pathway (frozen snapshots are A-policies: same semantics)
        self._hinge_localB = jnp.array([B_acts.index(a) for a in B_hinge], dtype=int)
        self._standB = self._q0[self._Bqa]
        self._jrB = jnp.array([list(m.jnt_range[j]) for j in Bj])
        _gB = np.array([float(m.actuator_gear[a, 0]) for a in B_hinge])
        _fB = np.array([float(m.actuator_forcerange[a, 1]) for a in B_hinge])
        self._gearB = jnp.array(np.where(_gB > 0, _gB, _fB))
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
        # LIDAR sensor addresses: rangefinder data is in sensordata at sensor_adr
        self._lidar_adr = jnp.array(m.sensor_adr[:m.nsensor]) if self._lidar and m.nsensor > 0 else None
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
    def _hist_dim(self):
        """History block width: H frames of (hinge qpos + hinge qvel) + prev action."""
        return (self._hist_len * 2 * self._n_hinge + self.action_size) if self._hist_len > 0 else 0

    @property
    def observation_size(self):
        her_dim = 4 if self._her_coeff > 0 else 0
        if self._lidar:
            # Asymmetric actor-critic: actor sees loco + lidar (+ history), critic sees
            # loco + lidar + privileged (+ history + foot contacts). HER goal stays LAST.
            return {
                "state": LOCO_OBS + self._lidar_scan_dim + self._hist_dim + her_dim,
                "value_state": LOCO_OBS + self._lidar_scan_dim + 6
                               + (8 if self._engage_obs else 0)
                               + (8 if self._contact_obs else 0)
                               + self._hist_dim + int(self._Afeet_gids.shape[0]) + her_dim,
            }
        return self._obs_size + self._hist_dim + her_dim
    @property
    def obsB_size(self):
        """Dim of the MIRRORED opponent observation (what a frozen B-snapshot consumes).

        B has no lidar sensors, so a frozen opponent always sees this flat layout;
        a snapshot trained on lidar/asymmetric obs is INCOMPATIBLE and is rejected
        early (see main). Mirrors A's flat layout including the history block (B's
        own hinges + B's previous action) so post-B.1 snapshots stay self-play-able."""
        hist = (self._hist_len * 2 * self._n_hinge_b + self._nuB) if self._hist_len > 0 else 0
        return LOCO_OBS + 6 + (8 if self._engage_obs else 0) + (8 if self._contact_obs else 0) + hist

    @property
    def action_size(self):
        base = self._nuA
        return base + 1 if self._hierarchical and self._has_striker else base
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

    def _lidar_scan(self, dx, rng=None):
        """Extract rangefinder sensordata, normalize to [0,1], apply DR.

        ``rng=None`` (or noise/dropout disabled) returns a clean, DETERMINISTIC
        scan — used by the benchmark/eval. Otherwise Gaussian range noise and
        random ray dropout are applied with the per-env key threaded in.
        Misses (rangefinder returns -1) and dropped rays read 1.0 (= max range).
        """
        raw = dx.sensordata[self._lidar_adr] if self._lidar_adr is not None else jnp.zeros(self._lidar_n_total)
        hits = jnp.where(raw < 0, self._lidar_max_range, jnp.clip(raw, 0.0, self._lidar_max_range))
        scan = hits / self._lidar_max_range
        if rng is not None and self._lidar_stochastic:
            k_noise, k_drop = jax.random.split(rng)
            if self._lidar_noise_sigma > 0:
                noise = jax.random.normal(k_noise, scan.shape, dtype=scan.dtype)
                scan = scan + noise * (self._lidar_noise_sigma / self._lidar_max_range)
            if self._lidar_dropout_rate > 0:
                drop = jax.random.bernoulli(k_drop, self._lidar_dropout_rate, scan.shape)
                scan = jnp.where(drop, 1.0, scan)
            scan = jnp.clip(scan, 0.0, 1.0)
        return scan

    def _lidar_obs(self, dx, d, info, her_goal):
        """Build the asymmetric lidar observation dict and advance lidar info.

        Order: per-env noise/dropout -> sensor LATENCY -> frame STACK -> obs.
        Applying latency BEFORE building the obs means a delayed scan reaches the
        actor AND critic even when ``frame_stack == 1`` (the old code only rebuilt
        the obs when stacking, so latency was a no-op at stack 1). Returns
        (obs_dict, new_info).
        """
        loco = self._loco(dx, d)
        rng = info["lidar_rng"]
        rng, sub = jax.random.split(rng)
        scan = self._lidar_scan(dx, rng=sub)            # per-env, per-step noise/dropout
        new_info = {**info, "lidar_rng": rng}
        # sensor LATENCY: the policy observes the scan from `latency` steps ago.
        if self._lidar_latency > 0:
            hist = new_info["lidar_scan_history"]       # (latency, n_total) FIFO
            observed = hist[0]
            new_info["lidar_scan_history"] = jnp.concatenate([hist[1:], scan[None]], axis=0)
        else:
            observed = scan
        # frame STACK on the (post-latency) observed scan for temporal velocity.
        if self._lidar_stack > 1:
            prev = new_info["lidar_prev_scans"]         # (stack-1, n_total)
            stacked = jnp.concatenate([prev.reshape(-1), observed])
            new_info["lidar_prev_scans"] = jnp.concatenate([prev[1:], observed[None]], axis=0)
        else:
            stacked = observed
        hist_parts = ([info["prop_hist"].reshape(-1), info["prev_act"]]
                      if self._hist_len > 0 else [])
        actor = jnp.concatenate([loco, stacked, *hist_parts])
        # critic additionally sees per-foot contact booleans (privileged: no foot
        # sensors exist on the actor side) — inserted BEFORE the HER goal tail.
        critic = jnp.concatenate([loco, stacked, self._privileged_tail(dx), *hist_parts,
                                  self._foot_contacts(dx)])
        if her_goal is not None and self._her_coeff > 0:
            actor = jnp.concatenate([actor, her_goal])
            critic = jnp.concatenate([critic, her_goal])
        return {"state": actor, "value_state": critic}, new_info

    def _her_extract_achieved(self, dx):
        """Extract the achieved goal (4D) from pipeline state for HER."""
        if self._her_coeff <= 0:
            return jnp.zeros(4)
        rel = (dx.xpos[self._Bt] - dx.xpos[self._At])[:2]
        dist = jnp.linalg.norm(rel)
        bearing = jnp.arctan2(rel[1], rel[0])
        rmat = dx.xmat[self._At].reshape(-1)
        fwd = rmat[:2]
        fwd = fwd / (jnp.linalg.norm(fwd) + 1e-6)
        front = jnp.dot(fwd, rel / (dist + 1e-6))
        rod_d = jnp.linalg.norm(dx.geom_xpos[self._Arod_gids] - dx.xpos[self._Bt], axis=1) \
            if self._Arod_gids.shape[0] > 0 else jnp.array([dist])
        min_rod = jnp.min(rod_d)
        return jnp.array([dist, bearing, front, min_rod])

    def _her_goal_reward(self, dx, her_goal):
        """Per-step goal-achievement reward for the active goal (pre-relabel)."""
        if self._her_coeff <= 0 or her_goal is None:
            return jnp.zeros(())
        achieved = self._her_extract_achieved(dx)
        diff = (achieved - her_goal) * jnp.array([1.0, 0.5, 0.3, 1.0])
        dist_sq = jnp.sum(diff ** 2)
        return self._her_coeff * jnp.exp(-dist_sq / (2.0 * self._her_sigma ** 2))

    def _loco(self, dx, d):
        """Stage-A-compatible proprioceptive observation (LOCO_OBS dims)."""
        return jnp.concatenate([dx.qpos[self._Aqa], dx.qvel[self._Ada], dx.xquat[self._At],
                                dx.qvel[self._ArD:self._ArD + 6], dx.xpos[self._At][2:3], d])

    def _rnd_feat(self, dx, d):
        """RND feature vector. RUNG 3: 'tactical' = engagement descriptors (distance,
        bearing, approach/lateral velocity, rod distance, front-alignment, tip speed) so
        novelty rewards reaching new tactical SITUATIONS, not new joint twitches. The
        proprioceptive default rewards any new joint config -> in-place jitter."""
        if self._rnd_feature == "tactical":
            eng = self._engage_tail(dx, self._At, self._Bt, self._ArD, self._BrD)   # [dist,ux,uy,radial,lateral,...]
            con = self._contact_tail(dx, self._At, self._Bt, self._Arod_gids, self._Astrike)  # [..,min_rod,..,front,..]
            tip = jnp.max(jnp.abs(dx.qvel[self._strike_dofs])) if self._has_striker else jnp.zeros(())
            return jnp.array([eng[0], eng[1], eng[2], eng[3], eng[4], con[2], con[4], tip])
        return self._loco(dx, d)

    def _privileged_tail(self, dx):
        """Privileged opponent state (critic-only when lidar): opp rel pos/vel + engage + contact."""
        opp = jnp.concatenate([dx.xpos[self._Bt] - dx.xpos[self._At], dx.qvel[self._BrD:self._BrD + 3]])
        if self._engage_obs:
            opp = jnp.concatenate([opp, self._engage_tail(dx, self._At, self._Bt, self._ArD, self._BrD)])
        if self._contact_obs:
            opp = jnp.concatenate([opp, self._contact_tail(dx, self._At, self._Bt,
                                                           self._Arod_gids, self._Astrike)])
        return opp

    def _foot_contacts(self, dx):
        """A's per-foot contact booleans (foot-geom height proxy, same rule as
        commanded_env's air-time term)."""
        from constants import FOOT_CONTACT_Z
        return (dx.geom_xpos[self._Afeet_gids][:, 2] < FOOT_CONTACT_Z).astype(jnp.float32)

    def _obs(self, dx, d, her_goal=None, info=None):
        """Non-lidar observation: [loco, privileged_opp, hist?, prev_act?, her_goal?].
        No actor/critic split here, so the privileged foot contacts are NOT included
        (they'd leak into the actor)."""
        parts = [self._loco(dx, d), self._privileged_tail(dx)]
        if self._hist_len > 0 and info is not None:
            parts += [info["prop_hist"].reshape(-1), info["prev_act"]]
        obs = jnp.concatenate(parts)
        if her_goal is not None and self._her_coeff > 0:
            obs = jnp.concatenate([obs, her_goal])
        return obs

    def _obsB(self, dx, d, info=None):
        """Opponent's obs — the SAME layout as A's but B-centric (B=me, A=opponent), so a frozen
        snapshot of OUR fighter drives B unchanged (symmetric self-play). Post-B.1 the
        layout mirror includes B's own proprio history + previous action."""
        loco = jnp.concatenate([dx.qpos[self._Bqa], dx.qvel[self._Bda], dx.xquat[self._Bt],
                                dx.qvel[self._BrD:self._BrD + 6], dx.xpos[self._Bt][2:3], d])
        opp = jnp.concatenate([dx.xpos[self._At] - dx.xpos[self._Bt], dx.qvel[self._ArD:self._ArD + 3]])
        if self._engage_obs:
            opp = jnp.concatenate([opp, self._engage_tail(dx, self._Bt, self._At, self._BrD, self._ArD)])
        if self._contact_obs:
            opp = jnp.concatenate([opp, self._contact_tail(dx, self._Bt, self._At,
                                                           self._Brod_gids, self._Bstrike)])
        parts = [loco, opp]
        if self._hist_len > 0 and info is not None and "prop_hist_B" in info:
            parts += [info["prop_hist_B"].reshape(-1), info["prev_act_B"]]
        return jnp.concatenate(parts)

    def _obs_walker(self, dx, prev_act):
        """COMMANDED-env obs layout (53 dims), built B-centrically: what the pdval
        walker was trained on. Order mirrors commanded_env._obs exactly: [hinge qpos,
        hinge qvel, quat, body-frame planar vel + vz + angular, torso z, design,
        prev_action, body-frame cmd]. cmd = unit(A−B)·walker_speed, yaw-rate 0."""
        quat = dx.qpos[self._Brq + 3:self._Brq + 7]
        w, x, y, z = quat[0], quat[1], quat[2], quat[3]
        yaw = jnp.arctan2(2.0 * (w * z + x * y), 1.0 - 2.0 * (y * y + z * z))
        c, s = jnp.cos(yaw), jnp.sin(yaw)
        R = jnp.array([[c, s], [-s, c]])
        v_body = R @ dx.qvel[self._BrD:self._BrD + 2]
        rel = (dx.xpos[self._At] - dx.xpos[self._Bt])[:2]
        cmd_world = rel / (jnp.linalg.norm(rel) + 1e-6) * self._walker_speed
        return jnp.concatenate([
            dx.qpos[self._Bqa], dx.qvel[self._Bda], quat,
            jnp.concatenate([v_body, dx.qvel[self._BrD + 2:self._BrD + 6]]),
            dx.qpos[self._Brq + 2:self._Brq + 3],
            self._walker_design,
            prev_act,
            jnp.concatenate([R @ cmd_world, jnp.zeros(1)]),
        ])

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

    def _info(self, d, dx, dp, lidar_rng=None):
        info = {"design": d, "prev_dist": self._planar_dist(dx),
                "prev_dealt": jnp.zeros(()), "t": jnp.zeros(()),
                # EMA of torso planar VELOCITY (vector, not magnitude) for the not_moving
                # gate: in-place oscillation averages to ~0, real locomotion doesn't. The
                # old instantaneous-speed gate went blind on the full-torque body (any
                # vigorous jitter clears 0.1 m/s instantaneously, every step).
                "vel_ema": jnp.zeros(2)}
        if self._rg:
            info["dp"] = dp
        if self._lidar:
            # Per-env lidar RNG seeded from the reset key (so noise/dropout differ
            # across envs AND episodes), advanced every step.
            info["lidar_rng"] = lidar_rng if lidar_rng is not None else jax.random.PRNGKey(0)
            if self._lidar_stack > 1:
                info["lidar_prev_scans"] = jnp.ones((self._lidar_stack - 1, self._lidar_n_total))
            if self._lidar_latency > 0:
                info["lidar_scan_history"] = jnp.ones((self._lidar_latency, self._lidar_n_total))
        if self._her_coeff > 0:
            # her_goal: active goal (set by reset); her_achieved: achieved goal at
            # this state — both exposed so the HER relabel pass can collect them.
            info["her_goal"] = jnp.zeros(4)
            info["her_achieved"] = self._her_extract_achieved(dx)
        if self._rnd is not None:
            # Per-env RND predictor + Adam state, identical at reset, trained per step.
            info["rnd_predictor"] = self._rnd.init_predictor_params
            info["rnd_opt_state"] = self._rnd.init_opt_state
        if self._cpg_control:
            info["phase"] = jnp.zeros(())                 # CPG gait phase, advanced each step
        if self._airtime_w > 0 or self._slip_w > 0:
            info["air_time_A"] = jnp.zeros(4)
            info["prev_feet_A"] = dx.geom_xpos[self._Afeet_gids][:, :2]
        if self._ko_w > 0:
            info["dealt_cum"] = jnp.zeros(())
        if self._drill_frac > 0:
            info["loco_drill"] = jnp.zeros(())    # reset() draws the real flag per episode
        if self._hist_len > 0:
            # B.1 (audit item 7): proprio/action history — at 50 Hz with parallel springs
            # and no foot sensors, stance/swing is unobservable from one frame. Seeded
            # with the current frame tiled (no fake zero-history transient at reset).
            frame = jnp.concatenate([dx.qpos[self._Aqa], dx.qvel[self._Ada]])
            info["prop_hist"] = jnp.tile(frame, (self._hist_len, 1))
            info["prev_act"] = jnp.zeros(self.action_size)
            if self._opp == "frozen":
                frameB = jnp.concatenate([dx.qpos[self._Bqa], dx.qvel[self._Bda]])
                info["prop_hist_B"] = jnp.tile(frameB, (self._hist_len, 1))
                info["prev_act_B"] = jnp.zeros(self._nuB)
        if self._opp == "walker":
            info["walker_prev_act"] = jnp.zeros(self._n_hinge_b)
        return info

    def _finish_reset(self, dx, d, info):
        """Build the observation (lidar dict or flat array) and finalize info."""
        her_goal = info.get("her_goal")
        if self._lidar:
            obs, info = self._lidar_obs(dx, d, info, her_goal)
        else:
            obs = self._obs(dx, d, her_goal=her_goal, info=info)
        return State(dx, obs, jnp.zeros(()), jnp.zeros(()), self._metrics0(), info)

    def reset(self, rng):
        if self._reset_keys is not None:
            idx = jax.random.randint(rng, (), 0, self._reset_keys.shape[0])
            rng = self._reset_keys[idx]
        rng, dr, nr, sr, tr, wr, gr, lr = jax.random.split(rng, 8)
        d = jax.random.uniform(dr, (3,))
        dp = self._world(wr) if self._rg else None
        sep = jax.random.uniform(sr, (), minval=self._sep_lo, maxval=self._sep_hi)
        theta = jax.random.uniform(tr, (), minval=-self._azimuth, maxval=self._azimuth)
        qpos = self._q0.at[7:].add(jax.random.uniform(nr, (self._q0.shape[0] - 7,), minval=-0.03, maxval=0.03))
        qpos = self._place(qpos, sep, theta)
        dx = mjx.forward(self._design_model(d, dp), mjx.make_data(self._mx).replace(qpos=qpos))
        info = self._info(d, dx, dp, lidar_rng=lr)
        if self._drill_frac > 0:
            # 25%-forever loco-drill rider: this episode ignores combat reward entirely
            # and just walks — combat gradients can never fully erase the gait.
            info["loco_drill"] = (jax.random.uniform(jax.random.fold_in(rng, 7), ())
                                  < self._drill_frac).astype(jnp.float32)
        if self._her_coeff > 0:
            g1, g2, g3, g4 = jax.random.split(gr, 4)
            info["her_goal"] = jnp.array([
                jax.random.uniform(g1, (), minval=0.1, maxval=1.0),
                jax.random.uniform(g2, (), minval=-jnp.pi, maxval=jnp.pi),
                jax.random.uniform(g3, (), minval=-1.0, maxval=1.0),
                jax.random.uniform(g4, (), minval=0.05, maxval=0.5)])
        return self._finish_reset(dx, d, info)

    def reset_with(self, rng, design):
        """Reset with a GIVEN design (eval). With reality_gap on, draws a calibrated world too
        (a fresh PRNG key per call samples a different world -> CVaR over keys = robust score)."""
        nr, wr, lr = jax.random.split(rng, 3)
        dp = self._world(wr) if self._rg else None
        qpos = self._q0.at[7:].add(jax.random.uniform(nr, (self._q0.shape[0] - 7,), minval=-0.03, maxval=0.03))
        dx = mjx.forward(self._design_model(design, dp), mjx.make_data(self._mx).replace(qpos=qpos))
        info = self._info(design, dx, dp, lidar_rng=lr)
        return self._finish_reset(dx, design, info)

    def step(self, state, action):
        d = state.info["design"]
        dp = state.info["dp"] if self._rg else None
        mxd = self._design_model(d, dp)
        clip_a = jnp.clip(action, -1, 1)
        # HIERARCHICAL: extract gate logit and modulate strike actions
        gate_val = 1.0  # default: no gating
        if self._hierarchical and self._has_striker:
            gate_logit = clip_a[-1]  # last action dim is the gate
            gate_val = jax.nn.sigmoid(gate_logit)
            clip_a = clip_a[:-1]  # drop gate dim for ctrl
            if self._strike_local.shape[0] > 0:
                strike_mask = jnp.zeros(self._nuA, dtype=clip_a.dtype)
                strike_mask = strike_mask.at[self._strike_local].set(1.0)
                clip_a = clip_a * (1.0 - strike_mask) + clip_a * strike_mask * gate_val
        if self._cpg_control:
            # CPG-PD: legs follow the directional CPG gait toward B (prior) + policy residual; the
            # striker/gate come straight from the policy. The CPG prior makes the body WALK from step 0.
            from cpg_teacher import cpg_pd_step_target, transition_phase_delta
            ps = state.pipeline_state
            rel = (ps.xpos[self._Bt] - ps.xpos[self._At])[:2]
            cmd = rel / (jnp.linalg.norm(rel) + 1e-6) * self._cpg_speed       # world-frame, toward B
            phase = state.info["phase"]
            leg_res = clip_a[self._hinge_local]
            target, _, _ = cpg_pd_step_target(
                self._cpg_stand, self._cpg_jr, phase, cmd, leg_res, self._cpg_idx, self._n_hinge,
                self._cpg_vmax, self._cpg_res_scale, self._pd_scale, directional=self._cpg, xp=jnp)
            tau = self._pd_kp * (target - ps.qpos[self._Aqa]) - self._pd_kd * ps.qvel[self._Ada]
            leg_ctrl = jnp.clip(tau / jnp.maximum(self._cpg_tmax, 1e-6), -1.0, 1.0)
            ctrl = jnp.zeros(self._nu).at[self._actA[self._hinge_local]].set(leg_ctrl)
            if self._has_striker:
                ctrl = ctrl.at[self._actA[self._strike_local]].set(clip_a[self._strike_local])
            cpg_phase_next = phase + transition_phase_delta(self._cpg_base_freq, self._cpg_dt,
                                                            jnp.zeros(()), xp=jnp)
        elif self._action_mode == "pd":
            # B.1 PD action mode: hinge TARGETS from the policy (offsets on the stance);
            # the torque PD runs per-substep inside the frame_skip loop below (250 Hz).
            # Striker/gate channels stay direct (pneumatic valve command, held 20 ms).
            ctrl = jnp.zeros(self._nu)
            if self._has_striker:
                ctrl = ctrl.at[self._actA[self._strike_local]].set(clip_a[self._strike_local])
            pd_target_A = jnp.clip(self._standA + self._pd_act_scale * clip_a[self._hinge_localA],
                                   self._jrA[:, 0], self._jrA[:, 1])
            cpg_phase_next = None
        else:
            a = self._ctrl_scale(clip_a, state.pipeline_state.qvel, dp)
            ctrl = jnp.zeros(self._nu).at[self._actA].set(a)
            cpg_phase_next = None
        pd_target_B = None
        if self._opp == "frozen" and self._opp_infer is not None:   # B driven by a frozen snapshot
            b_obs = self._obsB(state.pipeline_state, d, info=state.info)
            b_raw, _ = self._opp_infer(b_obs, jax.random.PRNGKey(0))  # deterministic ⇒ key unused
            b_clip = jnp.clip(b_raw, -1, 1)
            if self._action_mode == "pd" and not self._cpg_control:
                # frozen snapshots are A-policies saved under PD semantics (T7 sidecar
                # enforces this) — drive B's hinges through the SAME per-substep PD
                pd_target_B = jnp.clip(self._standB + self._pd_act_scale * b_clip[self._hinge_localB],
                                       self._jrB[:, 0], self._jrB[:, 1])
                if self._has_striker_b:
                    ctrl = ctrl.at[self._actB[self._B_strike_local]].set(b_clip[self._B_strike_local])
            else:
                ctrl = ctrl.at[self._actB].set(b_clip)
        elif self._opp == "walker" and self._walker_infer is not None:
            # C.2: commanded-env walker pursues A. Its actions are PD targets at the
            # COMMANDED scale (PD_SCALE=1.0, its training contract) — NOT the fighter's
            # 0.4 — applied through the same per-substep PD loop. B's striker stays 0.
            w_obs = self._obs_walker(state.pipeline_state, state.info["walker_prev_act"])
            w_raw, _ = self._walker_infer(w_obs, jax.random.PRNGKey(0))
            w_clip = jnp.clip(w_raw, -1.0, 1.0)
            pd_target_B = jnp.clip(self._standB + w_clip, self._jrB[:, 0], self._jrB[:, 1])
        elif self._opponent_script > 0:
            # RUNG 4: scripted ACTIVE opponent — B lunges toward A (crude pursuer) so the
            # judge/curriculum isn't a limp dummy; standing still then gets approached + hit.
            ps = state.pipeline_state
            relAB = (ps.xpos[self._At] - ps.xpos[self._Bt])[:2]
            dir_to_A = relAB / (jnp.linalg.norm(relAB) + 1e-6)
            Rb = ps.xmat[self._Bt].reshape(-1)
            fwd_b = Rb[:2] / (jnp.linalg.norm(Rb[:2]) + 1e-6)
            ahead = jnp.dot(fwd_b, dir_to_A)                      # +1 when A is in front of B
            # phase-cycled forward leg drive toward A (alternating sign per actuator = gait-ish)
            phase = jnp.sin(0.6 * state.info["t"] + jnp.pi * jnp.arange(self._nuB))
            b_ctrl = self._opponent_script * jnp.clip(ahead + 0.5, 0.0, 1.0) * phase
            ctrl = ctrl.at[self._actB].set(jnp.clip(b_ctrl, -1, 1))
        dx = state.pipeline_state.replace(ctrl=ctrl)
        if self._action_mode == "pd" and not self._cpg_control:
            actA_h = self._actA[self._hinge_localA]
            actB_h = self._actB[self._hinge_localB]
            def _pd_substep(i, x):
                tauA = (self._pd_kp_act * (pd_target_A - x.qpos[self._Aqa])
                        - self._pd_kd_act * x.qvel[self._Ada])
                c = x.ctrl.at[actA_h].set(jnp.clip(tauA / self._gearA, -1.0, 1.0))
                if pd_target_B is not None:
                    tauB = (self._pd_kp_act * (pd_target_B - x.qpos[self._Bqa])
                            - self._pd_kd_act * x.qvel[self._Bda])
                    c = c.at[actB_h].set(jnp.clip(tauB / self._gearB, -1.0, 1.0))
                return mjx.step(mxd, x.replace(ctrl=c))
            dx = jax.lax.fori_loop(0, self._fs, _pd_substep, dx)
        else:
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
        rel = (dx.xpos[self._Bt] - dx.xpos[self._At])[:2]; dist = jnp.linalg.norm(rel); n = dist + 1e-6
        toward = jnp.dot(dx.qvel[self._ArD:self._ArD + 2], rel) / n
        move = jnp.linalg.norm(dx.qvel[self._ArD:self._ArD + 2])   # torso planar speed
        clos = jnp.clip(toward / 2, 0, 1); flee = jnp.clip(-toward / 2, 0, 1)
        # RUNG 2b: damage is CREDITED only while CLOSING on the opponent (toward > eps),
        # so a stationary policy's incidental contact earns nothing. Off => byte-identical.
        closing_credit = jnp.where(self._require_closing,
                                   (toward > self._closing_eps).astype(jnp.float32), 1.0)
        scored_dealt = dealt_f * late_hit * closing_credit
        early_dealt = dealt_f * (1.0 - late_hit)
        # not_moving gates on SMOOTHED displacement velocity (β=0.04 EMA ≈ 0.5 s at 50 Hz):
        # a vibrating body has instantaneous speed > eps every step but vector-mean ~0.
        from constants import VEL_EMA_BETA
        vel_ema = ((1.0 - VEL_EMA_BETA) * state.info["vel_ema"]
                   + VEL_EMA_BETA * dx.qvel[self._ArD:self._ArD + 2])
        not_moving = (jnp.linalg.norm(vel_ema) < self._move_eps).astype(jnp.float32)
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
        # stance-height reward component (pay for standing TALL toward the CANONICAL stance,
        # not a low sprawl). Saturates at 0.285 = the validated settled stance z (validate_body,
        # full-torque body); the old 0.24 saturation point couldn't tell a crouch from standing.
        height = jnp.clip((dx.xpos[self._At][2] - 0.17) / 0.115, 0.0, 1.0)
        # the real SPARC objective (force/penetration-weighted damage + aggression):
        if self._loco_speed > 0:
            # B.3: the exp-kernel tracking term below replaces the farmable instantaneous
            # closing term (clip(toward/2) pays for velocity SPIKES toward B; the kernel
            # pays only for sustained matched velocity).
            sparc = 6.0 * (scored_dealt - taken_f)
        else:
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
        # LOCOMOTION reward: torso planar SPEED (move computed above; learn to translate the
        # body while staying upright — topple-done + upright keep it from lunge-and-fall).
        # HIERARCHICAL gate reward: reward opening the gate when close enough
        # (close_term >= gate_threshold), penalize opening when too far.
        gate_reward = 0.0
        if self._hierarchical and self._has_striker:
            in_range = (close_term >= self._gate_threshold).astype(jnp.float32)
            gate_reward = self._gate_weight * gate_val * in_range * close_term
            gate_reward -= 0.5 * self._gate_weight * gate_val * (1.0 - in_range)
        # ---- B.3 walk-then-fight: phase-0 velocity tracking + contact gait terms ----
        loco_track = jnp.zeros(())
        if self._loco_speed > 0:
            # command: walk toward the opponent at v_des, tapering to a stop inside 0.5 m
            # (later, "walk toward your opponent" is just this command — audit item 2's
            # velocity-command design carried into the arena).
            v_des = self._loco_speed * jnp.minimum(1.0, dist / 0.5)
            cmd_vec = (rel / n) * v_des
            vA = dx.qvel[self._ArD:self._ArD + 2]
            loco_track = jnp.exp(-jnp.sum((vA - cmd_vec) ** 2) / 0.25)
        gait = jnp.zeros(())
        if self._airtime_w > 0 or self._slip_w > 0:
            from constants import AIRTIME_CAP, AIRTIME_TARGET, FOOT_CONTACT_Z, GAIT_DISP_GATE
            foot_z = dx.geom_xpos[self._Afeet_gids][:, 2]
            fcontact = foot_z < FOOT_CONTACT_Z
            air_t = state.info["air_time_A"]
            first_c = jnp.logical_and(fcontact, air_t > 0.0)
            # capped air-time credit, gated on genuine displacement (EMA velocity —
            # jitter can't open the gate); a long hop can't out-earn a cadence.
            disp_gate = jnp.clip(jnp.linalg.norm(vel_ema) / GAIT_DISP_GATE, 0.0, 1.0)
            air_rwd = jnp.sum((jnp.minimum(air_t, AIRTIME_CAP) - AIRTIME_TARGET)
                              * first_c.astype(jnp.float32)) * disp_gate
            feet_xy = dx.geom_xpos[self._Afeet_gids][:, :2]
            ctrl_dt = self._fs * 0.004
            slip = jnp.sum(fcontact.astype(jnp.float32)
                           * jnp.sum((feet_xy - state.info["prev_feet_A"]) ** 2, axis=1)) / (ctrl_dt ** 2)
            pose_dev = jnp.sum((dx.qpos[self._Aqa] - self._standA) ** 2)
            gait = self._airtime_w * air_rwd - self._slip_w * slip - self._pose_w * pose_dev
        # combat terms bundled so the curriculum can anneal them (k_c) and the loco-drill
        # rider can zero them per-episode without touching the gait/tracking terms.
        drill = state.info.get("loco_drill", jnp.zeros(()))
        k_c = self._combat_scale * (1.0 - drill)
        combat = (sparc + shaped
                  + self._clean_w * clean - self._trade_w * trade + self._dis_w * disengage
                  - self._flee_w * flee + self._close_bonus_w * close_term
                  + self._face_w * close_term * face
                  + self._damage_bonus_w * scored_dealt
                  + self._fire_shaping * fire_aim + gate_reward)
        reward = (k_c * combat + self._loco_track_w * loco_track + gait
                  + self._approach_w * approach + self._upright_w * up + self._alive_bonus
                  + self._height_w * height + self._move_w * move
                  - fire_cost - self._energy_w * energy
                  - self._airborne_w * airborne - self._early_hit_penalty * early_dealt
                  - self._taken_w * taken_f
                  - self._penalty_w * jnp.maximum(0.0, peak_pen_step - self._penalty_tol)
                  + self._her_goal_reward(dx, state.info.get("her_goal"))
                  # RUNG 2b: penalize damage dealt while NOT moving (stationary jab) and
                  # actuator effort spent while NOT moving (in-place oscillation).
                  - self._stationary_pen * dealt_f * not_moving
                  - self._oscillation_pen * energy * not_moving)
        new_info = {**state.info, "prev_dist": dist, "vel_ema": vel_ema,
                    "prev_dealt": scored_dealt, "t": state.info["t"] + 1.0}
        if self._airtime_w > 0 or self._slip_w > 0:
            new_info["air_time_A"] = jnp.where(fcontact, 0.0,
                                               state.info["air_time_A"] + self._fs * 0.004)
            new_info["prev_feet_A"] = feet_xy
        if self._cpg_control:
            new_info["phase"] = cpg_phase_next
        # TRUE RND intrinsic bonus: novelty of the NEXT proprioceptive state under
        # the per-env predictor carried in info; the predictor then takes one Adam
        # step toward the random target, so revisited states yield less bonus over
        # time (novelty decreases on familiar states — the defining RND property).
        if self._rnd is not None:
            feat = self._rnd_feat(dx, d)
            pred, opt = state.info["rnd_predictor"], state.info["rnd_opt_state"]
            novelty = jnp.clip(self._rnd.novelty(pred, feat), 0.0, self._rnd_clip)
            reward = reward + self._rnd_coeff * novelty
            new_pred, new_opt, _ = self._rnd.update(pred, opt, feat)
            new_info["rnd_predictor"] = new_pred
            new_info["rnd_opt_state"] = new_opt
        # FALL = torso below 0.09 m. The 3.5 kg / gear-12 body holds a stable controllable stance at
        # torso-z ~0.15 (crouch ~0.11), measured; 0.09 sits below the crouch so dodging/crouching
        # survives but a real topple (torso ~0.05-0.07) is caught. The old 0.18 sat ABOVE this body's
        # max standing height (0.185, singular straight-leg) → survival was geometrically impossible.
        # FALL = torso below 0.09 m OR TOPPLED (up-axis tilted past ~70° from vertical). The height
        # check alone missed the backside collapse (torso stays ~0.1 m while lying on its back), so a
        # sprawl counted as "alive"; up_z<0.3 catches the topple even when the torso is still off the floor.
        a_fell = ((dx.xpos[self._At][2] < 0.09) | (up_z < 0.3)).astype(jnp.float32)
        done = a_fell
        if self._ko_w > 0:
            RB = dx.xmat[self._Bt].reshape(-1)
            b_down = ((dx.xpos[self._Bt][2] < 0.09) | (RB[8] < 0.3)).astype(jnp.float32)
            dealt_cum = state.info.get("dealt_cum", jnp.zeros(())) + scored_dealt
            ko_valid = b_down * (dealt_cum > self._ko_min_dealt).astype(jnp.float32)
            new_info["dealt_cum"] = dealt_cum
            reward = (self._ko_alpha * reward
                      + (1.0 - self._ko_alpha) * self._ko_w * (ko_valid - a_fell))
            if self._ko_done:
                done = jnp.maximum(done, ko_valid)
        # MERGE into the existing metrics dict (brax's Evaluator injects a 'reward' key —
        # replacing the dict drops it and breaks the scan-carry pytree).
        metrics = {**state.metrics, "dealt": dealt_f, "taken": taken_f, "closing": clos,
                   "fleeing": flee, "sparc": sparc, "dist": dist, "approach": approach,
                   "close_term": close_term, "clean_hit": clean, "trade": trade,
                   "disengage": disengage, "fire": fire_act, "face": face,
                   "penalty": jnp.maximum(0.0, peak_pen_step - self._penalty_tol)}
        if self._her_coeff > 0:
            # achieved goal at the NEXT state (her_goal carried unchanged) — collected
            # as a transition extra so the HER relabel pass can use it.
            new_info["her_achieved"] = self._her_extract_achieved(dx)
        if self._hist_len > 0:
            frame = jnp.concatenate([dx.qpos[self._Aqa], dx.qvel[self._Ada]])
            new_info["prop_hist"] = jnp.concatenate([state.info["prop_hist"][1:], frame[None]], axis=0)
            new_info["prev_act"] = jnp.clip(action, -1.0, 1.0)
            if self._opp == "frozen" and self._opp_infer is not None:
                frameB = jnp.concatenate([dx.qpos[self._Bqa], dx.qvel[self._Bda]])
                new_info["prop_hist_B"] = jnp.concatenate(
                    [state.info["prop_hist_B"][1:], frameB[None]], axis=0)
                new_info["prev_act_B"] = b_clip
        if self._opp == "walker" and self._walker_infer is not None:
            new_info["walker_prev_act"] = w_clip
        if self._lidar:
            new_obs, new_info = self._lidar_obs(dx, d, new_info, new_info.get("her_goal"))
        else:
            new_obs = self._obs(dx, d, her_goal=new_info.get("her_goal"), info=new_info)
        return state.replace(pipeline_state=dx, obs=new_obs, reward=reward, done=done,
                             metrics=metrics, info=new_info)


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


def _net_obs_dim(net):
    """Input-layer width of a saved policy/value net dict."""
    pp = net.get("params", net)
    return int(pp.get("hidden_0", {}).get("kernel", jnp.zeros((LOCO_OBS, 1))).shape[0])


def _keep_count(old_obs, target_dim):
    """How many leading obs dims to copy (locomotor 40-dim drops its 2 extra height dims)."""
    if old_obs == LOCO_OBS + 2 and target_dim >= LOCO_OBS + 6:
        return LOCO_OBS
    return min(old_obs, target_dim)


def _pad_net_input(net, old_obs, target_dim, keep):
    """Grow a net's INPUT layer old_obs->target_dim (leading rows kept, rest zero)."""
    if old_obs == target_dim:
        return net

    def pad_leaf(x):
        if hasattr(x, "ndim") and x.ndim >= 1 and x.shape[0] == old_obs:
            out = jnp.zeros((target_dim,) + x.shape[1:], dtype=x.dtype)
            n = min(keep, target_dim, x.shape[0])
            return out.at[:n].set(x[:n])
        return x

    return jax.tree_util.tree_map(pad_leaf, net)


def warm_start(path, obs_dim, act_dim=None):
    """Pad a saved (normalizer, policy, value) tuple to the target body/obs.

    Handles BOTH flat obs (int ``obs_dim``) and the asymmetric lidar dict obs
    (``{'state': actor_dim, 'value_state': critic_dim}``). The POLICY net + its
    normalizer head grow to ``state`` (actor) width; the VALUE net + its head grow
    to ``value_state`` (critic) width — so an asymmetric checkpoint restores
    correctly and a flat locomotor warm-starts BOTH heads on its shared loco
    prefix. The policy ACTION HEAD grows when the body gained DOFs (``act_dim``).
    Idempotent on a same-shape checkpoint; best-effort with a scratch fall-back."""
    try:
        parts = list(pickle.load(open(path, "rb")))      # (normalizer, policy_dict, value_dict, ...)
        norm, nets = parts[0], list(parts[1:])
        if not nets:
            raise ValueError("checkpoint has no policy net")
        is_dict = isinstance(obs_dim, dict)
        state_dim = int(obs_dim["state"]) if is_dict else int(obs_dim)
        value_dim = int(obs_dim["value_state"]) if is_dict else int(obs_dim)
        pol_old = _net_obs_dim(nets[0])
        val_old = _net_obs_dim(nets[1]) if len(nets) > 1 else pol_old
        if state_dim < pol_old or value_dim < val_old:
            raise ValueError(f"checkpoint wider than target "
                             f"(policy {pol_old}->{state_dim}, value {val_old}->{value_dim})")
        pol_keep, val_keep = _keep_count(pol_old, state_dim), _keep_count(val_old, value_dim)
        c = norm.count                                   # brax UInt64 = {hi, lo}: value = hi*2^32 + lo
        cval = float(jnp.asarray(c.hi)) * (2.0 ** 32) + float(jnp.asarray(c.lo))

        def remap1d(v, target, keep, fill):
            out = jnp.full((target,), fill, dtype=v.dtype)
            n = min(keep, target, v.shape[0])
            return out.at[:n].set(v[:n])

        # Normalizer: new dims start standardized (mean 0, std 1, summed_variance=count).
        nkw = {}
        for fn in ("mean", "std", "summed_variance"):
            v = getattr(norm, fn, None)
            if v is None:
                continue
            fill = 0.0 if fn == "mean" else 1.0 if fn == "std" else max(cval, 1.0)
            if isinstance(v, dict):                        # asymmetric source normalizer
                sv, vv = v["state"], v["value_state"]
            elif hasattr(v, "ndim") and v.ndim >= 1:       # flat source: share loco prefix
                sv = vv = v
            else:
                continue
            if is_dict:
                nkw[fn] = {"state": remap1d(sv, state_dim, _keep_count(sv.shape[0], state_dim), fill),
                           "value_state": remap1d(vv, value_dim, _keep_count(vv.shape[0], value_dim), fill)}
            elif sv.shape[0] != state_dim:
                nkw[fn] = remap1d(sv, state_dim, _keep_count(sv.shape[0], state_dim), fill)
        if nkw:
            norm = norm.replace(**nkw)

        nets[0] = _pad_net_input(nets[0], pol_old, state_dim, pol_keep)
        if len(nets) > 1:
            nets[1] = _pad_net_input(nets[1], val_old, value_dim, val_keep)
        for i in range(2, len(nets)):
            oi = _net_obs_dim(nets[i])
            nets[i] = _pad_net_input(nets[i], oi, value_dim, _keep_count(oi, value_dim))
        nets[0] = _grow_action_head(nets[0], act_dim)
        print(f"WARM-START ok: policy {pol_old}->{state_dim} value {val_old}->{value_dim} "
              f"({'dict' if is_dict else 'flat'} obs, count={cval:.0f}, {len(nets)} nets)", flush=True)
        return tuple([norm] + nets)
    except Exception as e:
        print(f"warm-start failed ({type(e).__name__}: {e}) -> training Stage B from scratch", flush=True)
        return None


def opponent_obs_act(path):
    """Return the (obs_dim, act_dim) a saved opponent checkpoint expects."""
    params = pickle.load(open(path, "rb"))
    pol = params[1]["params"]
    obs = int(pol["hidden_0"]["kernel"].shape[0])
    hids = [int(k.split("_")[1]) for k in pol if k.startswith("hidden_")]
    act = int(pol[f"hidden_{max(hids)}"]["bias"].shape[0]) // 2
    return obs, act


def validate_frozen_opponent(env, ckpt_path, role="opponent", allow_legacy=False):
    """Fail EARLY (before training) if a frozen opponent snapshot is incompatible.

    B is driven by the MIRRORED flat observation (``env.obsB_size``) and its action
    scatters onto B's ``_nuB`` actuators. A snapshot trained with lidar/asymmetric
    obs or a hierarchical gate has the wrong obs/act width and would only fail
    deep inside the jitted rollout — so we check the checkpoint's shapes here and
    raise a precise error instead.

    T7: shapes are NOT identity. The sidecar (.meta.json) must also agree on
    ACTION SEMANTICS — a torque-trained opponent driven under PD semantics (or a
    pre-gear-fix opponent on the full-torque body) passes every shape check and
    plays garbage. Missing sidecar = retired pre-2026-07 artifact = rejected."""
    import ckpt_meta
    ckpt_meta.check_semantics(
        ckpt_path,
        expected_semantics=ckpt_meta.fighter_semantics(
            getattr(env, "_action_mode", "torque"), getattr(env, "_pd_act_scale", 0.4)),
        expected_model_hash=ckpt_meta.current_model_hash(
            build_match(SPEC, SPEC, sep=1.2, striker=True, striker_b=True)),
        role=f"frozen {role}", allow_legacy=allow_legacy)
    exp_obs, exp_act = env.obsB_size, env._nuB
    got_obs, got_act = opponent_obs_act(ckpt_path)
    if got_obs != exp_obs:
        raise ValueError(
            f"frozen {role} '{os.path.basename(ckpt_path)}' expects obs dim {got_obs}, but the "
            f"mirrored B observation is {exp_obs}. B has no lidar sensors, so a lidar/asymmetric "
            f"snapshot cannot drive it — retrain the {role} WITHOUT --lidar-obs (and with matching "
            f"--engage-obs/--contact-obs), or provide a compatible snapshot.")
    if got_act != exp_act:
        raise ValueError(
            f"frozen {role} '{os.path.basename(ckpt_path)}' has action dim {got_act}, but B has "
            f"{exp_act} actuators. A hierarchical-gate snapshot (act={exp_act}+1) cannot drive B; "
            f"provide a non-hierarchical snapshot.")
    print(f"frozen {role} ok: obs={got_obs} act={got_act} (matches mirrored B)", flush=True)


def load_policy(path, observation_size, action_size):
    """Deterministic inference fn matching a checkpoint's obs structure.

    For a dict (asymmetric lidar) observation_size, builds the network with
    ``policy_obs_key='state'`` / ``value_obs_key='value_state'`` so the dict
    normalizer + dict obs are consumed correctly (unlike :func:`load_opponent`,
    which assumes a flat obs). Used by the renderer to roll out a lidar policy."""
    import ppo_nets as ppo_networks  # shared (512,256,128) + small-final-init factory (audit item 5)
    from brax.training.acme import running_statistics
    params = pickle.load(open(path, "rb"))
    if isinstance(observation_size, dict):
        net = ppo_networks.make_ppo_networks(
            observation_size, action_size,
            preprocess_observations_fn=running_statistics.normalize,
            policy_obs_key="state", value_obs_key="value_state")
    else:
        net = ppo_networks.make_ppo_networks(
            observation_size, action_size,
            preprocess_observations_fn=running_statistics.normalize)
    return ppo_networks.make_inference_fn(net)(params, deterministic=True)


def load_opponent(path, obs=None, act=None):
    """Frozen opponent (deterministic) inference fn from a saved striker ckpt — drives B in
    self-play. obs/act are INFERRED from the policy net if not given (input-layer width = obs;
    output-layer width / 2 = act). Returns a bound `policy(obs, key) -> (action, extra)`."""
    import ppo_nets as ppo_networks  # shared (512,256,128) + small-final-init factory (audit item 5)
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
    import ppo_nets as ppo_networks  # shared (512,256,128) + small-final-init factory (audit item 5)
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


def behavior_keep_ok(v, min_closed=-1e30, min_approach=-1e30, min_disp=-1e30, min_far_sparc=-1e30):
    """RUNG 2a/4 selection gate: a checkpoint may become keep-best only if it actually
    engaged — closed the gap, approached, moved, and performed from FAR spawns (not just
    incidental close-range contact). Pure function so it is unit-testable without a GPU."""
    return bool(v.get("bh_closed", 0.0) >= min_closed
                and v.get("bh_approach", 0.0) >= min_approach
                and v.get("bh_disp", 0.0) >= min_disp
                and v.get("sparc_far", 0.0) >= min_far_sparc)


def build_benchmark(bench_env, n_epis, steps, seed=20240601, deterministic=True):
    """Honest held-out benchmark: mean over FIXED-seed episodes (fixed designs/spawns, comparable
    across curriculum phases) of the episode-summed [SPARC, dealt, taken] COMBAT metrics — read
    from `state.metrics`, so independent of the shaped training reward. Network reconstructed the
    same way `combat_rank.load_policy` does (proven to consume these checkpoints). Returns a jitted
    `bench(params) -> [sparc, dealt, taken]`; keep-best on `sparc` makes best-so-far monotone."""
    from brax.training.acme import running_statistics
    obs_size = bench_env.observation_size
    if isinstance(obs_size, dict):
        net = ppo_networks.make_ppo_networks(
            obs_size, bench_env.action_size,
            preprocess_observations_fn=running_statistics.normalize,
            policy_obs_key="state", value_obs_key="value_state")
    else:
        net = ppo_networks.make_ppo_networks(
            obs_size, bench_env.action_size,
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
                # BEHAVIOR metrics — expose the "win by standing still" exploit live:
                Axy = ps.xpos[bench_env._At][:2]; rel_xy = ps.xpos[bench_env._Bt][:2] - Axy
                dist_xy = jnp.linalg.norm(rel_xy); unit_xy = rel_xy / (dist_xy + 1e-6)
                vel = ps.qvel[bench_env._ArD:bench_env._ArD + 2]
                approach = jnp.dot(vel, unit_xy)                              # >0 closing on opponent
                lateral = jnp.abs(vel[0] * (-unit_xy[1]) + vel[1] * unit_xy[0])  # circling/strafe
                gate_open = ((jax.nn.sigmoid(a[-1]) > 0.5).astype(jnp.float32)
                             if (bench_env._hierarchical and bench_env._has_striker) else 1.0)
                tip = (jnp.max(jnp.abs(ps.qvel[bench_env._strike_dofs]))
                       if bench_env._has_striker else 0.0)
                beh = jnp.array([Axy[0], Axy[1], dist_xy, lateral * alive, approach * alive,
                                 gate_open * alive, tip * alive])
                return (s, key, t + 1.0), (base, ac, beh)
            A0 = st.pipeline_state.xpos[bench_env._At][:2]                     # start torso xy
            (_, _, _), (base_o, ac_o, beh_o) = jax.lax.scan(stp, (st, k, 0.0), None, length=steps)
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
            # BEHAVIOR aggregates: net torso displacement, path length, how much it
            # CLOSED from the start, mean lateral/strafe + approach speed, gate-open
            # fraction, rod tip speed. A standing-still policy reads ~0 on movement.
            pos = jnp.concatenate([A0[None, :], beh_o[:, :2]], axis=0)
            path = jnp.sum(jnp.linalg.norm(pos[1:] - pos[:-1], axis=1))
            displacement = jnp.linalg.norm(beh_o[-1, :2] - A0)
            closed = d0 - jnp.min(beh_o[:, 2])
            beh_agg = jnp.array([displacement, path, closed, beh_o[:, 3].mean(),
                                 beh_o[:, 4].mean(), beh_o[:, 5].mean(), beh_o[:, 6].mean()])
            return base_o.sum(0), d0, ac_agg, beh_agg
        per_ep, d0, ac_ep, beh_ep = jax.vmap(ep)(keys)        # per_ep:(n,11); ac_ep:(n,8); beh_ep:(n,7)
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
        return jnp.concatenate([agg, bins, jnp.array([win, surv, safe]), ac_ep.mean(0),
                                beh_ep.mean(0)])  # 16 + 8 anti-cheat + 7 behavior
    return bench


# build_benchmark()'s 16-value vector: dense decomposition (for the Coach) + range profile + the
# SPARSE VERDICT (win_rate/survival_rate/safe_rate — the unshaped judgment; keep-best selects on win_rate).
BENCH_KEYS = ["sparc", "dealt", "taken", "clean", "trade", "fire", "closing", "fleeing", "dist",
              "alive", "sparc_close", "sparc_med", "sparc_far", "win_rate", "survival_rate", "safe_rate",
              # anti-cheat block (catch degenerate strategies live — see anti_cheat.py for the full 24):
              "ac_peak_z", "ac_airborne", "ac_peak_pen", "ac_idle", "ac_dmg_early",
              "ac_upright_dmg", "ac_grounded_dmg", "ac_uprightness",
              # BEHAVIOR block (rung 1): movement/engagement signals — a stand-still policy reads ~0
              "bh_disp", "bh_path", "bh_closed", "bh_lateral", "bh_approach", "bh_gate_open", "bh_tip_speed"]


def main():
    # Defaults follow the 2026-07 uplift audit (notes/training-uplift-audit.md item 3):
    # 100M steps at batch 512×8×20 (81,920 steps/iter) ≈ 1,220 PPO iterations — the old
    # 12M @ 1024×16×20 was ~37 iterations, an order of magnitude below any published
    # from-scratch quadruped result.
    ap = argparse.ArgumentParser(); ap.add_argument("--steps", type=int, default=100_000_000)
    ap.add_argument("--envs", type=int, default=2048); ap.add_argument("--resume", default=None)
    ap.add_argument("--batch", type=int, default=512); ap.add_argument("--minibatches", type=int, default=8)
    ap.add_argument("--unroll", type=int, default=20); ap.add_argument("--evals", type=int, default=0)
    ap.add_argument("--episode-length", type=int, default=600,
                    help="control steps/episode (600 @ 50 Hz = 12 s; was hardcoded 300 = 6 s, "
                         "barely 2 strides of credit at the old γ)")
    # B.3 walk-then-fight curriculum knobs (defaults = legacy combat reward exactly).
    # A locomotion phase: --loco-speed 0.1..0.6 --combat-scale 0..1 --loco-drill-frac 0.25
    # --gait-airtime-w 1.0 --gait-slip-w 0.1 --gait-pose-w 0.2 --alive-bonus 0.1→0.02.
    ap.add_argument("--combat-scale", type=float, default=1.0,
                    help="k_c: scales ALL combat reward terms (curriculum anneals 0→1, "
                         "gated on the behavior benchmark)")
    ap.add_argument("--loco-speed", type=float, default=0.0,
                    help="v_des (m/s) for exp-kernel velocity tracking toward the opponent "
                         "(0=off). Replaces the farmable instantaneous closing term when on.")
    ap.add_argument("--loco-track-w", type=float, default=8.0,
                    help="weight of the exp-kernel tracking term")
    ap.add_argument("--loco-drill-frac", type=float, default=0.0,
                    help="fraction of episodes that are PURE loco drills (combat reward "
                         "zeroed) for the whole run — the gait-retention rider")
    ap.add_argument("--alive-bonus", type=float, default=0.1,
                    help="per-step alive constant (anneal down as combat comes in)")
    ap.add_argument("--gait-airtime-w", type=float, default=0.0,
                    help="capped per-foot air-time reward, gated on real displacement")
    ap.add_argument("--gait-slip-w", type=float, default=0.0,
                    help="penalty on planar drift of feet IN CONTACT")
    ap.add_argument("--gait-pose-w", type=float, default=0.0,
                    help="penalty on hinge deviation from the stand pose")
    ap.add_argument("--ko-weight", type=float, default=0.0,
                    help="C.1: ±W terminal outcome reward (KO gated on cumulative dealt; "
                         "zero-sum vs own fall). 0=off.")
    ap.add_argument("--ko-alpha", type=float, default=1.0,
                    help="C.1: total = α·dense + (1−α)·outcome; anneal 1→0.2")
    ap.add_argument("--ko-done", action="store_true",
                    help="C.1: end the episode on a valid KO (NEVER in benchmark envs)")
    ap.add_argument("--diverse-resets", type=int, default=256,
                    help="B.4: size of the banked auto-reset pool (0 = stock brax replay "
                         "of ONE cached state per env — the audited defect). ~70%% of the "
                         "bank are launch states (root vel U(0.1,0.5) m/s, random heading).")
    ap.add_argument("--action-mode", choices=["pd", "torque"], default="pd",
                    help="B.1: pd = hinge actions are stance-relative position targets, "
                         "torque PD per physics substep (250 Hz) — the proven legged-RL "
                         "action space. torque = legacy direct 50 Hz ctrl write.")
    ap.add_argument("--pd-action-scale", type=float, default=0.4,
                    help="rad of hinge-target authority per unit action in pd mode")
    ap.add_argument("--history-len", type=int, default=3,
                    help="B.1: control steps of (qpos,qvel) history + prev action in the "
                         "actor obs (0 = single-frame legacy; stance/swing is unobservable "
                         "from one frame at 50 Hz)")
    ap.add_argument("--preflight", choices=["strict", "warn", "off"], default="strict",
                    help="T2 config sanity gate before training (strict=refuse red lines; "
                         "pbt_train passes warn for its per-cycle slices)")
    ap.add_argument("--regression-abort", action="store_true",
                    help="T6 tripwire: abort after 3 consecutive benches with judge >30%% "
                         "below peak (keep-best has the peak; stop paying for the decay — "
                         "the cpglong collapse burned ~40M steps post-peak)")
    ap.add_argument("--stagnation-disp-floor", type=float, default=0.0,
                    help="T6 tripwire: abort if bench bh_disp (m) is below this once 30%% of "
                         "--steps is spent (0=off; enable for locomotion-first runs — the "
                         "0.18m/12M run should have self-terminated at ~2M)")
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
    # Both flags below existed since the lidar work but NO script ever passed them (audit
    # item 3) — every run trained unclipped at fixed lr. On by default now; --lr-schedule
    # NONE / --max-grad-norm -1 restore the old behavior.
    ap.add_argument("--max-grad-norm", type=float, default=1.0,
                    help="PPO gradient clipping norm (pass -1 to disable)")
    ap.add_argument("--lr-schedule", choices=["NONE", "ADAPTIVE_KL"], default="ADAPTIVE_KL",
                    help="Brax PPO learning-rate schedule (ADAPTIVE_KL targets --desired-kl; "
                         "NOTE: reduces PBT's lr dimension to initial-lr-only)")
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
    # LIDAR sim-to-real sensor simulation
    ap.add_argument("--lidar-obs", action="store_true",
                    help="enable simulated lidar rangefinder observations (asymmetric actor-critic)")
    ap.add_argument("--lidar-n-rays", type=int, default=128,
                    help="number of horizontal lidar rays (360-degree sweep)")
    ap.add_argument("--lidar-n-vertical", type=int, default=16,
                    help="number of forward-facing vertical lidar rays")
    ap.add_argument("--lidar-max-range", type=float, default=2.0,
                    help="maximum lidar range in meters (misses map to this)")
    ap.add_argument("--lidar-noise-sigma", type=float, default=0.015,
                    help="Gaussian noise sigma on lidar distances (meters)")
    ap.add_argument("--lidar-dropout-rate", type=float, default=0.02,
                    help="fraction of lidar rays randomly dropped per step")
    ap.add_argument("--lidar-latency-steps", type=int, default=0,
                    help="number of control steps to delay lidar scan (sensor latency)")
    ap.add_argument("--lidar-frame-stack", type=int, default=3,
                    help="number of consecutive scans to stack for temporal velocity info")
    # Hierarchical policy: approach brain + strike brain with learned gate
    ap.add_argument("--hierarchical", action="store_true",
                    help="hierarchical policy: add a learned gate that modulates striker actions")
    ap.add_argument("--gate-weight", type=float, default=1.0,
                    help="weight for the gate reward (reward opening gate near opponent)")
    ap.add_argument("--gate-threshold", type=float, default=0.3,
                    help="close_term threshold for gate reward (below this = far)")
    # TRUE RND intrinsic motivation: per-env predictor trained inside env.step on
    # next-state loco features; novelty bonus added to reward (decreases on familiar states).
    ap.add_argument("--rnd-coefficient", type=float, default=0.0,
                    help="weight for the RND novelty bonus (0=disabled). Predictor IS trained per-env each step.")
    ap.add_argument("--rnd-lr", type=float, default=1e-3,
                    help="learning rate for the RND predictor (Adam)")
    ap.add_argument("--rnd-hidden-dim", type=int, default=128,
                    help="hidden layer size for the RND target/predictor networks")
    ap.add_argument("--rnd-output-dim", type=int, default=64,
                    help="output embedding dimension of the RND networks")
    ap.add_argument("--rnd-seed", type=int, default=0,
                    help="seed for the fixed RND target network + predictor init")
    # TRUE HER: on-policy hindsight relabeling pass over each PPO rollout window
    # (future-goal relabel of obs + reward). See her_goal.install_her_relabel.
    ap.add_argument("--her-coefficient", type=float, default=0.0,
                    help="weight for the goal-achievement reward AND enables hindsight relabeling (0=disabled)")
    ap.add_argument("--her-sigma", type=float, default=0.15,
                    help="sigma for goal-achievement Gaussian kernel")
    ap.add_argument("--her-fraction", type=float, default=0.5,
                    help="fraction of rollout transitions relabeled with a future achieved goal (hindsight)")
    # RUNG 3: RND feature space (tactical descriptors vs raw proprioception)
    # tactical is the default (audit C19): proprio-RND paid novelty for unseen joint
    # configurations, i.e. it FUNDED the jitter exploit; tactical descriptors pay for
    # new engagement situations. rung-3 test asserts the sensitivity gap.
    ap.add_argument("--rnd-feature", choices=["proprio", "tactical"], default="tactical",
                    help="RND novelty feature space: 'tactical' (distance/bearing/approach/contact/tip — rewards new situations) or 'proprio' (raw joints — rewards twitching)")
    # RUNG 2b: outcome-grounded reward shaping (kill the stand-still exploit)
    ap.add_argument("--require-closing", action="store_true",
                    help="credit dealt damage only while CLOSING on the opponent (toward > closing-eps)")
    ap.add_argument("--closing-eps", type=float, default=0.05,
                    help="min approach velocity (m/s) for a hit to count under --require-closing")
    ap.add_argument("--stationary-damage-penalty", type=float, default=0.0,
                    help="penalty on damage dealt while the torso is not moving (move < move-eps)")
    ap.add_argument("--oscillation-penalty", type=float, default=0.0,
                    help="penalty on hinge effort spent while the torso is not moving (in-place jitter)")
    ap.add_argument("--move-eps", type=float, default=0.1,
                    help="torso planar speed (m/s) below which the policy counts as 'not moving'")
    # RUNG 4: scripted active opponent (passive B is a curriculum phase, not the judge)
    ap.add_argument("--opponent-script", type=float, default=0.0,
                    help="strength of a scripted pursuer opponent B (0=passive limp dummy)")
    # CPG-PD locomotion control: the body walks via the CPG gait prior (verified), policy = residual
    ap.add_argument("--cpg-control", action="store_true",
                    help="drive legs with the CPG gait toward B + policy residual (gives walking from step 0)")
    ap.add_argument("--cpg-speed", type=float, default=0.9,
                    help="commanded CPG walk speed toward the opponent (m/s)")
    ap.add_argument("--cpg-residual-scale", type=float, default=0.5,
                    help="how much the policy residual can modulate the CPG gait (0=pure CPG)")
    # RUNG 2a / 4: behavior + range-balanced keep-best gates (benchmark can't be farmed by standing still)
    ap.add_argument("--min-keep-closed", type=float, default=-1e30,
                    help="min held-out gap CLOSED (m) required for keep-best (rejects stand-still wins)")
    ap.add_argument("--min-keep-approach", type=float, default=-1e30,
                    help="min held-out mean approach velocity required for keep-best")
    ap.add_argument("--min-keep-disp", type=float, default=-1e30,
                    help="min held-out torso displacement (m) required for keep-best")
    ap.add_argument("--min-keep-far-sparc", type=float, default=-1e30,
                    help="min held-out FAR-range SPARC required for keep-best (must perform from far spawns, not only close)")
    ap.add_argument("--policy-train-scope", choices=["full", "head"], default="full",
                    help="head trains only the final policy dense layer; value net still trains normally")
    ap.add_argument("--freeze-normalizer", action="store_true",
                    help="benchmark/save checkpoints with the warm-start normalizer to avoid policy drift from obs stats")
    ap.add_argument("--milestone-gap", type=int, default=int(os.environ.get("MILESTONE_GAP", "0")), help="save a numbered milestone checkpoint+sidecar every N cum-steps (0=off). Env default MILESTONE_GAP so arena stages save milestones too — the render daemon turns each into a 1v1 evolution video")
    # Held-out BENCHMARK eval (the honest monotone-improvement curve + keep-best selection). Run
    # at every eval on a FIXED config (comparable across curriculum phases), independent of the
    # shaped training reward — reads the SPARC/dealt/taken metrics, not `reward`.
    ap.add_argument("--bench-epis", type=int, default=16, help="benchmark episodes (fixed held-out seeds)")
    ap.add_argument("--bench-steps", type=int, default=600,
                    help="benchmark steps/episode (matches --episode-length so the judge "
                         "sees full episodes, not the first third)")
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
    ap.add_argument("--walker-ckpt", default=None,
                    help="C.2: commanded-env walker checkpoint (pdval lineage) that drives B "
                         "as a PURSUER when --opponent walker (T7 sidecar must say "
                         "commanded-PD semantics)")
    ap.add_argument("--walker-speed", type=float, default=0.25,
                    help="C.2: pursuer's commanded speed toward A (anneal 0.1→0.35)")
    ap.add_argument("--opponent", choices=["passive", "frozen", "walker"], default="passive",
                    help="B opponent: passive (skill curriculum) or frozen (self-play vs a snapshot)")
    ap.add_argument("--opp-ckpt", default=None, help="frozen opponent ckpt (striker snapshot) driving B")
    ap.add_argument("--allow-legacy-opponent", action="store_true",
                    help="T7 override: load a sidecar-less (pre-2026-07 gear-fix) opponent "
                         "checkpoint anyway. Those trained on the 8%%-torque body; you almost "
                         "never want this.")
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
    walker_infer = None
    if args.opponent == "walker":
        if args.action_mode != "pd":
            raise SystemExit("--opponent walker requires --action-mode pd (the walker's "
                             "targets ride the per-substep PD loop)")
        if not (args.walker_ckpt and os.path.exists(args.walker_ckpt)):
            raise SystemExit(f"--opponent walker needs --walker-ckpt (got {args.walker_ckpt!r})")
        import ckpt_meta
        ckpt_meta.check_semantics(args.walker_ckpt,
                                  expected_semantics=ckpt_meta.COMMANDED_PD_SEMANTICS,
                                  expected_model_hash=None,   # walker trained single-body build
                                  role="walker opponent")
        walker_infer = load_opponent(args.walker_ckpt)
        print(f"walker-pursuer opponent: {args.walker_ckpt} @ {args.walker_speed} m/s", flush=True)
    t_env = time.time(); env = AdversarialEnv(shaping=args.shaping, sep=args.sep,
                                              self_collision=not args.lean_contacts,
                                              frame_skip=args.frame_skip,
                                              action_mode=args.action_mode,
                                              pd_action_scale=args.pd_action_scale,
                                              history_len=args.history_len,
                                              combat_scale=args.combat_scale,
                                              loco_speed=args.loco_speed,
                                              loco_track_w=args.loco_track_w,
                                              loco_drill_frac=args.loco_drill_frac,
                                              alive_bonus=args.alive_bonus,
                                              gait_airtime_w=args.gait_airtime_w,
                                              gait_slip_w=args.gait_slip_w,
                                              gait_pose_w=args.gait_pose_w,
                                              ko_weight=args.ko_weight, ko_alpha=args.ko_alpha,
                                              ko_done=args.ko_done,
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
                                              opponent=args.opponent, opp_infer=opp_infer,
                                              walker_infer=walker_infer,
                                              walker_speed=args.walker_speed,
                                              lidar=args.lidar_obs,
                                              lidar_n_rays=args.lidar_n_rays,
                                              lidar_n_vertical=args.lidar_n_vertical,
                                              lidar_max_range=args.lidar_max_range,
                                              lidar_noise_sigma=args.lidar_noise_sigma,
                                              lidar_dropout_rate=args.lidar_dropout_rate,
                                              lidar_latency_steps=args.lidar_latency_steps,
                                              lidar_frame_stack=args.lidar_frame_stack,
                                              hierarchical=args.hierarchical,
                                              gate_weight=args.gate_weight,
                                              gate_threshold=args.gate_threshold,
                                              her_coefficient=args.her_coefficient,
                                              her_sigma=args.her_sigma,
                                              her_fraction=args.her_fraction,
                                              rnd_coefficient=args.rnd_coefficient,
                                              rnd_hidden_dim=args.rnd_hidden_dim,
                                              rnd_output_dim=args.rnd_output_dim,
                                              rnd_lr=args.rnd_lr,
                                              rnd_seed=args.rnd_seed,
                                              rnd_feature=args.rnd_feature,
                                              require_closing=args.require_closing,
                                              closing_eps=args.closing_eps,
                                              stationary_damage_penalty=args.stationary_damage_penalty,
                                              oscillation_penalty=args.oscillation_penalty,
                                              move_eps=args.move_eps,
                                              opponent_script=args.opponent_script,
                                              cpg_control=args.cpg_control,
                                              cpg_speed=args.cpg_speed,
                                              cpg_residual_scale=args.cpg_residual_scale)
    # TRUE RND (predictor trained per-env inside env.step on next-state features)
    # is built INSIDE AdversarialEnv when rnd_coefficient>0 — no wrapper needed.
    if args.rnd_coefficient > 0:
        print(f"RND enabled (true, env-integrated): coeff={args.rnd_coefficient} "
              f"feature_dim={LOCO_OBS} hidden={args.rnd_hidden_dim} "
              f"output={args.rnd_output_dim} lr={args.rnd_lr}", flush=True)
    # TRUE HER: install the on-policy hindsight relabel pass over PPO rollouts.
    if args.her_coefficient > 0:
        from her_goal import install_her_relabel
        install_her_relabel(args.her_coefficient, sigma=args.her_sigma,
                            fraction=args.her_fraction)
    # Frozen-opponent compatibility: reject incompatible snapshots BEFORE training.
    if args.opponent == "frozen" and opp_infer is not None:
        validate_frozen_opponent(env, args.opp_ckpt, role="opponent",
                                 allow_legacy=args.allow_legacy_opponent)
    if args.opponent == "frozen" and bench_opp is not None:
        validate_frozen_opponent(env, bench_opp_path, role="bench-opponent",
                                 allow_legacy=args.allow_legacy_opponent)
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
        # bench mirrors the walker opponent when training against it (WIN vs a mover
        # is the meaningful number); frozen/passive logic unchanged otherwise.
        bench_opponent = ("walker" if walker_infer is not None
                          else "frozen" if bench_opp is not None else "passive")
        bench_env = AdversarialEnv(self_collision=not args.lean_contacts, frame_skip=args.frame_skip,
                                   action_mode=args.action_mode,
                                   pd_action_scale=args.pd_action_scale,
                                   history_len=args.history_len,
                                   # bench stays at combat defaults (judge scores fights,
                                   # not curriculum phases); obs layout must match though.
                                   sep_lo=args.bench_sep_lo, sep_hi=args.bench_sep_hi, azimuth=args.bench_az,
                                   striker=striker, opponent=bench_opponent, opp_infer=bench_opp,
                                   walker_infer=walker_infer, walker_speed=args.walker_speed,
                                   face_opponent=args.face_opponent, engage_obs=args.engage_obs,
                                   contact_obs=args.contact_obs,
                                   face_weight=args.face_weight,
                                   penetration_penalty=args.penetration_penalty,
                                   penetration_tol=args.penetration_tol,
                                   rod_reach=args.rod_reach,
                                   lidar=args.lidar_obs,
                                   lidar_n_rays=args.lidar_n_rays,
                                   lidar_n_vertical=args.lidar_n_vertical,
                                   lidar_max_range=args.lidar_max_range,
                                   lidar_noise_sigma=0.0,  # deterministic clean scan in benchmark
                                   lidar_dropout_rate=0.0,
                                   lidar_latency_steps=0,  # no latency: held-out eval is deterministic
                                   lidar_frame_stack=args.lidar_frame_stack,
                                   hierarchical=args.hierarchical,
                                   gate_weight=args.gate_weight,
                                   gate_threshold=args.gate_threshold,
                                   # Match the TRAINING obs structure so the policy loads: the goal
                                   # dims must be present. The benchmark goal is deterministic per
                                   # fixed seed and only affects `reward`, not the SPARC/dealt/taken
                                   # metrics the benchmark reads. RND stays OFF in the benchmark.
                                   her_coefficient=args.her_coefficient,
                                   her_sigma=args.her_sigma,
                                   rnd_coefficient=0.0,
                                   # RUNG 4: the JUDGE faces the same scripted active opponent
                                   # (passive B is a training curriculum phase, not the benchmark).
                                   opponent_script=args.opponent_script,
                                   cpg_control=args.cpg_control,
                                   cpg_speed=args.cpg_speed,
                                   cpg_residual_scale=args.cpg_residual_scale)
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
    regression = {"n": 0}          # consecutive benches >30% below peak judge (T6 tripwire)
    def g(m, k): return float(m.get(f"eval/episode_{k}", 0.0))
    import ckpt_meta
    # body identity = the canonical two-robot build from the current spec (captures
    # gear/masses/geometry; per-run scene tweaks like lean-contacts don't change WHO
    # the policy is — they're recorded in the resolved-config JSON instead)
    _model_hash = ckpt_meta.current_model_hash(
        build_match(SPEC, SPEC, sep=1.2, striker=True, striker_b=True))
    def save(obj, name):
        try:
            pickle.dump(obj, open(OUT / name, "wb"))
            ckpt_meta.write_meta(OUT / name,
                                 action_semantics=ckpt_meta.fighter_semantics(
                                     args.action_mode, args.pd_action_scale),
                                 obs_size=env.observation_size, model_hash=_model_hash,
                                 behavior={k: best.get(k) for k in ("score", "win", "judge", "step")},
                                 extra=dict(tag=args.tag, lidar=bool(args.lidar_obs),
                                            history_len=args.history_len))
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
                   and (v["safe_rate"] >= args.min_keep_safe)
                   # RUNG 2a/4: behavior + range gates — a stand-still or close-only policy
                   # cannot become keep-best (the benchmark can't be farmed by standing still).
                   and behavior_keep_ok(v, args.min_keep_closed, args.min_keep_approach,
                                        args.min_keep_disp, args.min_keep_far_sparc))
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
        # T6 stagnation tripwire (behavioral, unfarmable: displacement can't be earned
        # by twitching in place under the EMA gate, and can't be zero while walking).
        if (args.stagnation_disp_floor > 0 and step >= 0.3 * args.steps
                and v["bh_disp"] < args.stagnation_disp_floor):
            print(f"  [{args.tag}] TRIPWIRE-STAGNATION: bh_disp {v['bh_disp']:.3f} m < "
                  f"{args.stagnation_disp_floor} at {step:,}/{args.steps:,} — aborting.",
                  flush=True)
            os._exit(3)
        # T6 regression tripwire: judge > 30% below peak for 3 consecutive benches ⇒
        # self-play is likely cycling (the cpglong collapse burned ~40M steps post-peak).
        # keep-best already preserves the peak params; this stops paying for the decay.
        if best["judge"] > -1e29 and bjudge < 0.7 * best["judge"] and best["judge"] > 0:
            regression["n"] += 1
            print(f"  [tripwire] judge {bjudge:+.2f} < 70% of peak {best['judge']:+.2f} "
                  f"({regression['n']}/3 consecutive)", flush=True)
            if regression["n"] >= 3 and args.regression_abort:
                print(f"  [{args.tag}] TRIPWIRE-REGRESSION: aborting; peak preserved in "
                      f"{args.tag}_best.pkl @ step {best['step']:,}", flush=True)
                os._exit(4)
        else:
            regression["n"] = 0
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
                   ac_grounded_dmg=round(v["ac_grounded_dmg"], 3), ac_uprightness=round(v["ac_uprightness"], 3),
                   bh_disp=round(v["bh_disp"], 3), bh_path=round(v["bh_path"], 3),
                   bh_closed=round(v["bh_closed"], 3), bh_lateral=round(v["bh_lateral"], 3),
                   bh_approach=round(v["bh_approach"], 3), bh_gate_open=round(v["bh_gate_open"], 3),
                   bh_tip_speed=round(v["bh_tip_speed"], 3))
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
        cheats = ([] + (["LAUNCH"] if v["ac_peak_z"] > 0.60 else [])  # matches anti_cheat.py bar
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
        # BEHAVIOR (rung 1): the real stand-still tell is winning WITHOUT engaging —
        # it never closes the gap (bh_closed ~0) and doesn't approach (bh_approach <=0),
        # even if it twitches in place (small bh_disp). Flag that against a nonzero win.
        stuck = (v["win_rate"] > 0.05 and v["bh_closed"] < 0.1
                 and v["bh_approach"] < 0.05 and v["bh_disp"] < 0.2)
        print(f"  [behavior] disp {v['bh_disp']:.2f}m path {v['bh_path']:.2f}m closed {v['bh_closed']:+.2f}m "
              f"lateral {v['bh_lateral']:.2f} approach {v['bh_approach']:+.3f} "
              f"gate_open {v['bh_gate_open']:.2f} tip_speed {v['bh_tip_speed']:.2f}"
              + ("  ## STAND-STILL (no movement)" if stuck else ""), flush=True)
    def _head_factory(observation_size, action_size, preprocess_observations_fn, **_):
        return head_only_network_factory(observation_size, action_size, preprocess_observations_fn)
    network_factory = _head_factory if args.policy_train_scope == "head" else None
    # Asymmetric PPO network for lidar: actor sees 'state' (loco+lidar), critic sees 'value_state' (loco+lidar+privileged)
    if network_factory is None and args.lidar_obs:
        def _asymmetric_factory(observation_size, action_size, preprocess_observations_fn, **_):
            return ppo_networks.make_ppo_networks(
                observation_size, action_size,
                preprocess_observations_fn=preprocess_observations_fn,
                policy_obs_key="state",
                value_obs_key="value_state",
            )
        network_factory = _asymmetric_factory
    if network_factory is None:
        # non-lidar path: still route through the shared factory — the brax default
        # (4x32) is exactly the never-overridden bottleneck audit item 5 retired.
        network_factory = ppo_networks.make_ppo_networks
    train_kwargs = {"network_factory": network_factory}
    if args.diverse_resets > 0:
        # B.4: banked auto-reset via the wrap_env_fn hook — stock brax replays ONE
        # cached reset state per env forever (audit item 6).
        from reset_bank import make_wrap_fn
        train_kwargs["wrap_env_fn"] = make_wrap_fn(
            jax.random.PRNGKey(args.reset_bank_seed if args.reset_bank_seed is not None else 0),
            bank_size=args.diverse_resets, canonical_frac=0.3,
            root_dof=env._ArD)
    if args.preflight != "off":
        from preflight import preflight_check
        obs_size = env.observation_size
        obs_dim = obs_size.get("state") if isinstance(obs_size, dict) else obs_size
        preflight_check(steps=args.steps, batch=args.batch, minibatches=args.minibatches,
                        unroll=args.unroll, episode_length=args.episode_length,
                        discounting=0.99, control_dt=0.004 * args.frame_skip,
                        obs_dim=obs_dim, hidden0=512, from_scratch=(restore is None),
                        mode=args.preflight, tag=args.tag, run_dir=OUT, resolved=vars(args))
    # γ=0.99 -> 2 s credit horizon at 50 Hz (0.97 was 0.66 s — shorter than one stride);
    # reward_scaling 0.1 keeps advantage magnitudes sane under the single shared Adam
    # (ship together with grad clipping, per audit item 3).
    ppo.train(environment=env, num_timesteps=args.steps, num_evals=n_eval,
              episode_length=args.episode_length, num_envs=args.envs, batch_size=args.batch,
              num_minibatches=args.minibatches, unroll_length=args.unroll, num_updates_per_batch=args.updates,
              learning_rate=args.lr, entropy_cost=args.entropy, discounting=0.99, reward_scaling=0.1,
              clipping_epsilon=args.clip_epsilon, desired_kl=args.desired_kl,
              max_grad_norm=(None if args.max_grad_norm is not None and args.max_grad_norm < 0
                             else args.max_grad_norm),
              learning_rate_schedule=args.lr_schedule,
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
