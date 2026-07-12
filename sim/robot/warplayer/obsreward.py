# SPDX-License-Identifier: MIT
"""warplayer.obsreward — component (iv) of the thin bespoke layer (§10c).

Obs + reward kernels that mirror train_adversarial.AdversarialEnv and are
launched in the SAME sequence as mujoco_warp's step (graph-capturable on
CUDA), eliminating the per-control-step device->host->device round-trip that
bounds runtime observation latency.

OBS (obs_kernel) mirrors AdversarialEnv._obs / _lidar_obs exactly for the
base config (history_len=0, her off, engage/contact obs off, frame_stack=1,
latency=0, noise off — every omitted feature is a HOST-side training-time
add-on, see the split below):
  flat  (has_lidar=0): [loco(38), opp(6)]           == _obs        (:627-637)
  lidar (has_lidar=1): [loco(38), scan(nray)]       == _lidar_obs actor (:529-566)
  loco (38 = constants.LOCO_OBS) == _loco (:594-597):
    [qpos[Aqa](12), qvel[Ada](12), xquat[At](4), qvel[ArD:ArD+6](6),
     xpos[At].z(1), design(3)]
  opp == _privileged_tail base (:614-621):
    [xpos[Bt]-xpos[At](3), qvel[BrD:BrD+3](3)]

REWARD (damage_kernel + reward_kernel) implements every DENSE term of
AdversarialEnv.step that is a pure function of (state, action, carried
scalars) — line references inline. Device-carried state: prev_dist,
prev_dealt, vel_ema, t (all (nworld,) buffers updated in-kernel, so the
whole thing lives inside one captured graph).

HOST-SIDE SPLIT (documented per the M3 spec): sparse KO logic (dealt_cum
accounting + ko_done, :1074-1083), HER goal reward (:585-601), RND intrinsic
bonus (:1055-1064), gait air-time/slip/pose (per-foot carried state,
:1015-1027, weights default 0), CPG prior, curriculum/drill riders, episode
resets, and lidar noise/dropout/latency/stacking DR (:518-527) stay host-side:
they are either sparse/episodic bookkeeping or training-time randomization,
not per-step dense compute.

mujoco_warp Data fields read IN-PLACE by these kernels (the explicit list the
GPU graph capture needs — capture is a flag, not a rewrite):
  obs_kernel:    d.qpos, d.qvel, d.xpos, d.xquat
  damage_kernel: d.nacon, d.contact.dist, d.contact.geom, d.contact.worldid
  reward_kernel: d.qvel, d.xpos, d.xmat, d.geom_xpos
plus layer-owned buffers (act, scan, damage accumulators, carries).
Everything indexed (nworld, ...) per the §4 layout mandate.

The numpy reference implementations at the bottom double as the BASELINE
(wrapper-way) computation in fused.py and as the test oracle: baseline
correctness and kernel parity are the same code path.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import warp as wp

# train_adversarial.py:40-41 — module constants there; that module imports
# Keep these local to the Warp kernel module. Values are asserted in tests.
DAMAGE_REF = 0.05
STRIKE_KINETIC = 0.1


# ---------------------------------------------------------------------------
# host-side index extraction (mirrors AdversarialEnv.__init__ :252-292,380-387)
# ---------------------------------------------------------------------------

@dataclass
class FightIndices:
    n_hinge: int
    nuA: int
    At: int
    Bt: int
    ArD: int
    BrD: int
    Aqa: np.ndarray            # (n_hinge,) qpos addresses of A's hinge joints
    Ada: np.ndarray            # (n_hinge,) dof addresses
    actA: np.ndarray           # (nuA,) actuator ids of A (ctrl scatter)
    Bqa: np.ndarray            # mirrored B addresses for self-play
    Bda: np.ndarray
    actB: np.ndarray
    strike_local: np.ndarray   # strike slots within the A action vector
    strike_local_b: np.ndarray
    strike_dofs: np.ndarray    # A slide (rod) dof addresses
    strike_dofs_b: np.ndarray  # B slide dof addresses
    Astrike: np.ndarray        # A calf/foot body ids (limb-proximity shaping)
    Bstrike: np.ndarray
    Arod_gids: np.ndarray      # A rod geom ids (fire aim shaping)
    Brod_gids: np.ndarray
    mask_Aleg: np.ndarray      # (ngeom,) int32 0/1 weapon/target masks
    mask_Bleg: np.ndarray
    mask_Arod: np.ndarray
    mask_Brod: np.ndarray
    mask_Abody: np.ndarray
    mask_Bbody: np.ndarray


def fight_indices(mjm) -> FightIndices:
    """Extract the env's index tables from the fight MjModel with the same
    name-based rules as train_adversarial.py:252-292 (weapon/target masks),
    :279-289 (A actuators/joints), :380-386 (roots, torsos)."""
    import mujoco

    gn = lambda g: mujoco.mj_id2name(mjm, mujoco.mjtObj.mjOBJ_GEOM, g) or ""
    an = lambda a: mujoco.mj_id2name(mjm, mujoco.mjtObj.mjOBJ_ACTUATOR, a) or ""
    bn = lambda b: mujoco.mj_id2name(mjm, mujoco.mjtObj.mjOBJ_BODY, b) or ""
    mk = lambda p: np.array([1 if p(gn(g)) else 0 for g in range(mjm.ngeom)], dtype=np.int32)
    leg_weapon = lambda n, s: n.startswith(s + "_") and (n.endswith("_foot") or n.endswith("_spear"))
    leg_target = lambda n, s: (n == s + "_torso" or (n.startswith(s + "_") and (
        n.endswith("_hipg") or n.endswith("_thighg") or n.endswith("_calfg"))))

    A_acts = [a for a in range(mjm.nu) if an(a).startswith("A_")]
    A_hinge = [a for a in A_acts if not an(a).endswith("_strike_m")]
    A_strike = [a for a in A_acts if an(a).endswith("_strike_m")]
    B_acts = [a for a in range(mjm.nu) if an(a).startswith("B_")]
    B_hinge = [a for a in B_acts if not an(a).endswith("_strike_m")]
    B_strike = [a for a in B_acts if an(a).endswith("_strike_m")]
    Aj = [int(mjm.actuator_trnid[a, 0]) for a in A_hinge]
    Bj = [int(mjm.actuator_trnid[a, 0]) for a in B_hinge]
    jid = lambda name: mujoco.mj_name2id(mjm, mujoco.mjtObj.mjOBJ_JOINT, name)
    bid = lambda name: mujoco.mj_name2id(mjm, mujoco.mjtObj.mjOBJ_BODY, name)
    _strike_bodies = [b for b in range(mjm.nbody) if bn(b).startswith("A_")
                      and (bn(b).endswith("_calf") or bn(b).endswith("_foot"))]
    _strike_bodies_b = [b for b in range(mjm.nbody) if bn(b).startswith("B_")
                        and (bn(b).endswith("_calf") or bn(b).endswith("_foot"))]
    At = bid("A_torso")
    return FightIndices(
        n_hinge=len(A_hinge), nuA=len(A_acts), At=At, Bt=bid("B_torso"),
        ArD=int(mjm.jnt_dofadr[jid("A_root")]), BrD=int(mjm.jnt_dofadr[jid("B_root")]),
        Aqa=np.array([int(mjm.jnt_qposadr[j]) for j in Aj], dtype=np.int32),
        Ada=np.array([int(mjm.jnt_dofadr[j]) for j in Aj], dtype=np.int32),
        actA=np.array(A_acts, dtype=np.int32),
        Bqa=np.array([int(mjm.jnt_qposadr[j]) for j in Bj], dtype=np.int32),
        Bda=np.array([int(mjm.jnt_dofadr[j]) for j in Bj], dtype=np.int32),
        actB=np.array(B_acts, dtype=np.int32),
        strike_local=np.array([A_acts.index(a) for a in A_strike], dtype=np.int32),
        strike_local_b=np.array([B_acts.index(a) for a in B_strike], dtype=np.int32),
        strike_dofs=np.array([int(mjm.jnt_dofadr[mjm.actuator_trnid[a, 0]]) for a in A_strike], dtype=np.int32),
        strike_dofs_b=np.array([int(mjm.jnt_dofadr[mjm.actuator_trnid[a, 0]]) for a in B_strike], dtype=np.int32),
        Astrike=np.array(_strike_bodies if _strike_bodies else [At], dtype=np.int32),
        Bstrike=np.array(_strike_bodies_b if _strike_bodies_b else [bid("B_torso")], dtype=np.int32),
        Arod_gids=np.array([g for g in range(mjm.ngeom)
                            if gn(g).startswith("A_") and gn(g).endswith("_rod")], dtype=np.int32),
        Brod_gids=np.array([g for g in range(mjm.ngeom)
                            if gn(g).startswith("B_") and gn(g).endswith("_rod")], dtype=np.int32),
        mask_Aleg=mk(lambda n: leg_weapon(n, "A")), mask_Bleg=mk(lambda n: leg_weapon(n, "B")),
        mask_Arod=mk(lambda n: n.startswith("A_") and n.endswith("_rod")),
        mask_Brod=mk(lambda n: n.startswith("B_") and n.endswith("_rod")),
        mask_Abody=mk(lambda n: leg_target(n, "A")), mask_Bbody=mk(lambda n: leg_target(n, "B")))


# ---------------------------------------------------------------------------
# reward configuration (env-default weights; train_adversarial.py:52-80 kwargs)
# ---------------------------------------------------------------------------

@wp.struct
class RewardParams:
    shaping: float
    approach_w: float
    upright_w: float
    alive: float
    energy_w: float
    airborne_w: float
    airborne_z: float
    height_w: float
    move_w: float
    close_bonus_w: float
    close_radius: float
    face_w: float
    flee_w: float
    taken_w: float
    clean_w: float
    trade_w: float
    dis_w: float
    damage_bonus_w: float
    combat_scale: float
    loco_speed: float
    loco_track_w: float
    early_hit_penalty: float
    min_hit_step: float
    require_closing: float
    closing_eps: float
    stationary_pen: float
    oscillation_pen: float
    move_eps: float
    vel_ema_beta: float
    penalty_w: float
    penalty_tol: float
    fire_cost: float
    fire_shaping: float
    rod_reach: float
    kin: float
    damage_ref: float
    fall_z: float
    topple_up_z: float


@dataclass
class RewardConfig:
    """Python-side mirror of RewardParams with the env's DEFAULT kwargs
    (train_adversarial.py:52-80). Shared constants come from sim/robot/constants
    at runtime (the V.1 single-read-point convention)."""
    shaping: float = 1.0
    approach_w: float = 0.0
    upright_w: float = 0.3
    alive: float = 0.1
    energy_w: float = 0.0
    airborne_w: float = 0.0
    airborne_z: float = 0.42       # spawn_height(0.35) + 0.07 (:164-165)
    height_w: float = 0.0
    move_w: float = 0.0
    close_bonus_w: float = 0.0
    close_radius: float = 0.45
    face_w: float = 0.0
    flee_w: float = 0.0
    taken_w: float = 0.0
    clean_w: float = 0.0
    trade_w: float = 0.0
    dis_w: float = 0.0
    damage_bonus_w: float = 0.0
    combat_scale: float = 1.0
    loco_speed: float = 0.0
    loco_track_w: float = 8.0
    early_hit_penalty: float = 0.0
    min_hit_step: float = 0.0
    require_closing: float = 0.0
    closing_eps: float = 0.05
    stationary_pen: float = 0.0
    oscillation_pen: float = 0.0
    move_eps: float = 0.1          # constants.MOVE_EPS (checked in from_constants)
    vel_ema_beta: float = 0.04     # constants.VEL_EMA_BETA
    penalty_w: float = 0.0
    penalty_tol: float = 0.045
    fire_cost: float = 0.0         # SPEC striker.fire_cost (:246-247)
    fire_shaping: float = 0.0
    rod_reach: float = 0.30
    kin: float = STRIKE_KINETIC
    damage_ref: float = DAMAGE_REF
    fall_z: float = 0.09           # constants.FALL_Z
    topple_up_z: float = 0.3       # constants.TOPPLE_UP_Z

    @classmethod
    def from_constants(cls, spec: dict | None = None, **overrides) -> "RewardConfig":
        """Pull the shared values from sim/robot/constants (must be importable —
        fused.py puts sim/robot on sys.path) + the striker spec, then apply overrides."""
        from constants import FALL_Z, MOVE_EPS, TOPPLE_UP_Z, VEL_EMA_BETA
        kw = dict(move_eps=MOVE_EPS, vel_ema_beta=VEL_EMA_BETA,
                  fall_z=FALL_Z, topple_up_z=TOPPLE_UP_Z)
        if spec is not None:
            kw["fire_cost"] = float(spec.get("striker", {}).get("fire_cost", 0.0))
            kw["airborne_z"] = float(spec.get("torso", {}).get("spawn_height", 0.35)) + 0.07
        kw.update(overrides)
        return cls(**kw)

    def to_struct(self) -> RewardParams:
        p = RewardParams()
        for name in RewardParams.vars:  # noqa: B007 — warp struct field registry
            setattr(p, name, float(getattr(self, name)))
        return p


# ---------------------------------------------------------------------------
# kernels
# ---------------------------------------------------------------------------

@wp.kernel
def obs_kernel(
    # Data (read in-place):
    qpos: wp.array2d(dtype=wp.float32),
    qvel: wp.array2d(dtype=wp.float32),
    xpos: wp.array2d(dtype=wp.vec3),
    xquat: wp.array2d(dtype=wp.quat),
    # In:
    Aqa: wp.array(dtype=wp.int32),
    Ada: wp.array(dtype=wp.int32),
    design: wp.array2d(dtype=wp.float32),
    scan: wp.array2d(dtype=wp.float32),
    At: int,
    Bt: int,
    ArD: int,
    BrD: int,
    has_lidar: int,
    # Out:
    obs: wp.array2d(dtype=wp.float32),
):
    w = wp.tid()
    nh = Aqa.shape[0]
    for i in range(nh):                       # _loco :594-597
        obs[w, i] = qpos[w, Aqa[i]]
        obs[w, nh + i] = qvel[w, Ada[i]]
    o = 2 * nh
    q = xquat[w, At]                          # stored (w,x,y,z): put_data keeps MjData order
    obs[w, o + 0] = q[0]
    obs[w, o + 1] = q[1]
    obs[w, o + 2] = q[2]
    obs[w, o + 3] = q[3]
    o += 4
    for i in range(6):                        # root lin+ang velocity
        obs[w, o + i] = qvel[w, ArD + i]
    o += 6
    obs[w, o] = xpos[w, At][2]                # torso z
    o += 1
    for i in range(design.shape[1]):          # design vector (3)
        obs[w, o + i] = design[w, i]
    o += design.shape[1]
    if has_lidar == 1:                        # _lidar_obs actor layout :558
        for i in range(scan.shape[1]):
            obs[w, o + i] = scan[w, i]
    else:                                     # _privileged_tail base :614-616
        dp = xpos[w, Bt] - xpos[w, At]
        obs[w, o + 0] = dp[0]
        obs[w, o + 1] = dp[1]
        obs[w, o + 2] = dp[2]
        for i in range(3):
            obs[w, o + 3 + i] = qvel[w, BrD + i]


@wp.kernel
def damage_zero_kernel(
    dealt_leg: wp.array(dtype=wp.float32),
    dealt_rod: wp.array(dtype=wp.float32),
    taken_leg: wp.array(dtype=wp.float32),
    taken_rod: wp.array(dtype=wp.float32),
    pen_peak: wp.array(dtype=wp.float32),
):
    w = wp.tid()
    dealt_leg[w] = 0.0
    dealt_rod[w] = 0.0
    taken_leg[w] = 0.0
    taken_rod[w] = 0.0
    pen_peak[w] = 0.0


@wp.kernel
def damage_kernel(
    # Data (read in-place; the compacted cross-world contact pool):
    nacon: wp.array(dtype=wp.int32),
    con_dist: wp.array(dtype=wp.float32),
    con_geom: wp.array(dtype=wp.vec2i),
    con_worldid: wp.array(dtype=wp.int32),
    # In (0/1 geom masks, train_adversarial.py:264-270):
    mask_Aleg: wp.array(dtype=wp.int32),
    mask_Bleg: wp.array(dtype=wp.int32),
    mask_Arod: wp.array(dtype=wp.int32),
    mask_Brod: wp.array(dtype=wp.int32),
    mask_Abody: wp.array(dtype=wp.int32),
    mask_Bbody: wp.array(dtype=wp.int32),
    # Out (per-world accumulators, pre-zeroed):
    dealt_leg: wp.array(dtype=wp.float32),
    dealt_rod: wp.array(dtype=wp.float32),
    taken_leg: wp.array(dtype=wp.float32),
    taken_rod: wp.array(dtype=wp.float32),
    pen_peak: wp.array(dtype=wp.float32),
):
    i = wp.tid()
    if i >= nacon[0]:
        return
    g = con_geom[i]
    g0 = g[0]
    g1 = g[1]
    if g0 < 0 or g1 < 0:                       # flex contact: no geoms
        return
    pen = wp.max(0.0, -con_dist[i])            # :900 pen = max(0, -dist)
    w = con_worldid[i]
    wp.atomic_max(pen_peak, w, pen)            # :901 peak_pen_step
    if pen <= 0.0:
        return
    # :902-903 legs-as-weapons damage; :909,:919 rod damage (separate masks)
    if (mask_Aleg[g0] == 1 and mask_Bbody[g1] == 1) or (mask_Aleg[g1] == 1 and mask_Bbody[g0] == 1):
        wp.atomic_add(dealt_leg, w, pen)
    if (mask_Bleg[g0] == 1 and mask_Abody[g1] == 1) or (mask_Bleg[g1] == 1 and mask_Abody[g0] == 1):
        wp.atomic_add(taken_leg, w, pen)
    if (mask_Arod[g0] == 1 and mask_Bbody[g1] == 1) or (mask_Arod[g1] == 1 and mask_Bbody[g0] == 1):
        wp.atomic_add(dealt_rod, w, pen)
    if (mask_Brod[g0] == 1 and mask_Abody[g1] == 1) or (mask_Brod[g1] == 1 and mask_Abody[g0] == 1):
        wp.atomic_add(taken_rod, w, pen)


@wp.kernel
def reward_kernel(
    # Data (read in-place):
    qvel: wp.array2d(dtype=wp.float32),
    xpos: wp.array2d(dtype=wp.vec3),
    xmat: wp.array2d(dtype=wp.mat33),
    geom_xpos: wp.array2d(dtype=wp.vec3),
    # In:
    act: wp.array2d(dtype=wp.float32),         # (nworld, nuA) clipped action
    dealt_leg: wp.array(dtype=wp.float32),
    dealt_rod: wp.array(dtype=wp.float32),
    taken_leg: wp.array(dtype=wp.float32),
    taken_rod: wp.array(dtype=wp.float32),
    pen_peak: wp.array(dtype=wp.float32),
    Astrike: wp.array(dtype=wp.int32),
    Arod_gids: wp.array(dtype=wp.int32),
    strike_dofs: wp.array(dtype=wp.int32),
    strike_dofs_b: wp.array(dtype=wp.int32),
    strike_local: wp.array(dtype=wp.int32),
    At: int,
    Bt: int,
    ArD: int,
    BrD: int,
    n_hinge: int,
    p: RewardParams,
    # In/out device-carried state:
    prev_dist: wp.array(dtype=wp.float32),
    prev_dealt: wp.array(dtype=wp.float32),
    vel_ema: wp.array(dtype=wp.vec2),
    t: wp.array(dtype=wp.float32),
    # Out:
    reward: wp.array(dtype=wp.float32),
    done: wp.array(dtype=wp.float32),
):
    w = wp.tid()

    # :910,920 rod tip speeds
    rod_speed = float(0.0)
    for i in range(strike_dofs.shape[0]):
        rod_speed = wp.max(rod_speed, wp.abs(qvel[w, strike_dofs[i]]))
    rod_speed_b = float(0.0)
    for i in range(strike_dofs_b.shape[0]):
        rod_speed_b = wp.max(rod_speed_b, wp.abs(qvel[w, strike_dofs_b[i]]))
    # :902-921 dealt/taken with the kinetic rod multiplier
    dealt = dealt_leg[w] + dealt_rod[w] * (1.0 + p.kin * rod_speed)
    taken = taken_leg[w] + taken_rod[w] * (1.0 + p.kin * rod_speed_b)
    dealt_f = wp.clamp(dealt / p.damage_ref, 0.0, 1.0)     # :922
    taken_f = wp.clamp(taken / p.damage_ref, 0.0, 1.0)
    late_hit = wp.where(t[w] >= p.min_hit_step, 1.0, 0.0)  # :923

    # :924-927 engagement geometry
    pa = xpos[w, At]
    pb = xpos[w, Bt]
    rel = wp.vec2(pb[0] - pa[0], pb[1] - pa[1])
    dist = wp.length(rel)
    n = dist + 1.0e-6
    vA = wp.vec2(qvel[w, ArD + 0], qvel[w, ArD + 1])
    toward = wp.dot(vA, rel) / n
    move = wp.length(vA)
    clos = wp.clamp(toward / 2.0, 0.0, 1.0)
    flee = wp.clamp(-toward / 2.0, 0.0, 1.0)
    # :930-933 closing-gated damage credit
    credit = wp.where(p.require_closing != 0.0, wp.where(toward > p.closing_eps, 1.0, 0.0), 1.0)
    scored_dealt = dealt_f * late_hit * credit
    early_dealt = dealt_f * (1.0 - late_hit)
    # :937-939 velocity EMA + not_moving gate
    ema = (1.0 - p.vel_ema_beta) * vel_ema[w] + p.vel_ema_beta * vA
    vel_ema[w] = ema
    not_moving = wp.where(wp.length(ema) < p.move_eps, 1.0, 0.0)
    # :940-943 close/face (env quirk preserved: fwd = xmat.reshape(-1)[:2] = row 0 xy)
    close_term = wp.clamp(1.0 - dist / wp.max(p.close_radius, 1.0e-6), 0.0, 1.0)
    R = xmat[w, At]
    fw = wp.vec2(R[0, 0], R[0, 1])
    fw = fw / (wp.length(fw) + 1.0e-6)
    face = wp.dot(fw, rel / n)
    # :952-955 uprightness from the torso's world-Z axis components
    fwd_z = R[2, 0]
    side_z = R[2, 1]
    up_z = R[2, 2]
    up = up_z - 0.6 * wp.abs(fwd_z) - 0.4 * wp.abs(side_z)
    height = wp.clamp((pa[2] - 0.17) / 0.115, 0.0, 1.0)    # :959
    # :963-967 SPARC
    sparc = 6.0 * (scored_dealt - taken_f)
    if p.loco_speed <= 0.0:
        sparc += 5.0 * (clos - flee)
    # :971-972 close->strike shaping
    legdist = float(3.4e38)
    for i in range(Astrike.shape[0]):
        legdist = wp.min(legdist, wp.length(xpos[w, Astrike[i]] - pb))
    shaped = p.shaping * (-0.15 * dist - 0.20 * legdist + 3.0 * scored_dealt)
    approach = prev_dist[w] - dist                          # :975
    # :979-982 win-exchange asymmetry
    clean = scored_dealt * (1.0 - taken_f)
    trade = wp.min(dealt_f, taken_f)
    outward = wp.clamp(-toward / 2.0, 0.0, 1.0)
    disengage = prev_dealt[w] * outward
    # :983 energy over the FIRST n_hinge action slots (env layout convention)
    energy = float(0.0)
    for i in range(n_hinge):
        energy += wp.abs(act[w, i])
    energy /= float(n_hinge)
    airborne = wp.max(0.0, pa[2] - p.airborne_z)            # :986
    # :912-917 pneumatic fire cost + aim shaping
    fire_cost = float(0.0)
    fire_aim = float(0.0)
    for i in range(strike_local.shape[0]):
        fi = wp.clamp(act[w, strike_local[i]], 0.0, 1.0)
        fire_cost += p.fire_cost * fi
        d_rb = wp.length(geom_xpos[w, Arod_gids[i]] - pb)
        fire_aim += fi * wp.max(0.0, 1.0 - d_rb / p.rod_reach)
    # :1002-1005 phase-0 velocity tracking toward the opponent
    loco_track = float(0.0)
    if p.loco_speed > 0.0:
        v_des = p.loco_speed * wp.min(1.0, dist / 0.5)
        cmd = rel * (v_des / n)
        dv = vA - cmd
        loco_track = wp.exp(-wp.dot(dv, dv) / 0.25)
    # :1028-1045 assemble (gait/HER/RND/gate are host-side, see module docstring)
    combat = (sparc + shaped
              + p.clean_w * clean - p.trade_w * trade + p.dis_w * disengage
              - p.flee_w * flee + p.close_bonus_w * close_term
              + p.face_w * close_term * face
              + p.damage_bonus_w * scored_dealt
              + p.fire_shaping * fire_aim)
    r = (p.combat_scale * combat + p.loco_track_w * loco_track
         + p.approach_w * approach + p.upright_w * up + p.alive
         + p.height_w * height + p.move_w * move
         - fire_cost - p.energy_w * energy
         - p.airborne_w * airborne - p.early_hit_penalty * early_dealt
         - p.taken_w * taken_f
         - p.penalty_w * wp.max(0.0, pen_peak[w] - p.penalty_tol)
         - p.stationary_pen * dealt_f * not_moving
         - p.oscillation_pen * energy * not_moving)
    reward[w] = r
    # :1073 fall/topple termination (KO variant is host-side)
    fell = float(0.0)
    if pa[2] < p.fall_z or up_z < p.topple_up_z:
        fell = 1.0
    done[w] = fell
    prev_dist[w] = dist                                     # :1047 info carry
    prev_dealt[w] = scored_dealt
    t[w] = t[w] + 1.0


# ---------------------------------------------------------------------------
# numpy references — the test oracle AND the baseline (wrapper-way) compute
# ---------------------------------------------------------------------------

def normalize_scan(raw: np.ndarray, max_range: float) -> np.ndarray:
    """train_adversarial._lidar_scan:515-517 (clean/deterministic branch)."""
    hits = np.where(raw < 0, max_range, np.clip(raw, 0.0, max_range))
    return hits / max_range


def obs_reference(h: dict, idx: FightIndices, design: np.ndarray,
                  scan: np.ndarray | None = None) -> np.ndarray:
    """Vectorized numpy mirror of obs_kernel over h = dict of (nworld, ...) arrays."""
    loco = np.concatenate([
        h["qpos"][:, idx.Aqa],
        h["qvel"][:, idx.Ada],
        h["xquat"][:, idx.At],
        h["qvel"][:, idx.ArD:idx.ArD + 6],
        h["xpos"][:, idx.At, 2:3],
        design,
    ], axis=1)
    if scan is not None:
        return np.concatenate([loco, scan], axis=1).astype(np.float32)
    opp = np.concatenate([
        h["xpos"][:, idx.Bt] - h["xpos"][:, idx.At],
        h["qvel"][:, idx.BrD:idx.BrD + 3],
    ], axis=1)
    return np.concatenate([loco, opp], axis=1).astype(np.float32)


def damage_reference(h: dict, idx: FightIndices, nworld: int):
    """Per-world (dealt_leg, dealt_rod, taken_leg, taken_rod, pen_peak) from the
    contact pool arrays (numpy mirror of damage_kernel)."""
    nacon = int(h["nacon"])
    out = np.zeros((5, nworld), dtype=np.float64)
    geom = h["con_geom"][:nacon]
    dist = h["con_dist"][:nacon]
    wid = h["con_worldid"][:nacon]
    ok = (geom[:, 0] >= 0) & (geom[:, 1] >= 0)
    pen = np.maximum(0.0, -dist) * ok
    g0, g1 = geom[:, 0], geom[:, 1]
    def pair(a, b):
        return ((idx.__dict__[a][np.clip(g0, 0, None)] & idx.__dict__[b][np.clip(g1, 0, None)])
                | (idx.__dict__[a][np.clip(g1, 0, None)] & idx.__dict__[b][np.clip(g0, 0, None)])).astype(bool) & ok
    for row, (a, b) in enumerate((("mask_Aleg", "mask_Bbody"), ("mask_Arod", "mask_Bbody"),
                                  ("mask_Bleg", "mask_Abody"), ("mask_Brod", "mask_Abody"))):
        sel = pair(a, b)
        np.add.at(out[row], wid[sel], pen[sel])
    np.maximum.at(out[4], wid, pen)
    return out[0], out[1], out[2], out[3], out[4]


def reward_reference(h: dict, idx: FightIndices, cfg: RewardConfig,
                     prev_dist: np.ndarray, prev_dealt: np.ndarray,
                     vel_ema: np.ndarray, t: np.ndarray):
    """Vectorized numpy mirror of damage_kernel + reward_kernel. Returns
    (reward, done, new_prev_dist, new_prev_dealt, new_vel_ema, new_t)."""
    nworld = h["qvel"].shape[0]
    dealt_leg, dealt_rod, taken_leg, taken_rod, pen_peak = damage_reference(h, idx, nworld)
    qvel, xpos, xmat, act = h["qvel"], h["xpos"], h["xmat"], h["act"]

    rod_speed = (np.abs(qvel[:, idx.strike_dofs]).max(axis=1)
                 if idx.strike_dofs.size else np.zeros(nworld))
    rod_speed_b = (np.abs(qvel[:, idx.strike_dofs_b]).max(axis=1)
                   if idx.strike_dofs_b.size else np.zeros(nworld))
    dealt = dealt_leg + dealt_rod * (1.0 + cfg.kin * rod_speed)
    taken = taken_leg + taken_rod * (1.0 + cfg.kin * rod_speed_b)
    dealt_f = np.clip(dealt / cfg.damage_ref, 0, 1)
    taken_f = np.clip(taken / cfg.damage_ref, 0, 1)
    late_hit = (t >= cfg.min_hit_step).astype(np.float64)

    rel = (xpos[:, idx.Bt] - xpos[:, idx.At])[:, :2]
    dist = np.linalg.norm(rel, axis=1)
    n = dist + 1e-6
    vA = qvel[:, idx.ArD:idx.ArD + 2]
    toward = np.einsum("wi,wi->w", vA, rel) / n
    move = np.linalg.norm(vA, axis=1)
    clos = np.clip(toward / 2, 0, 1)
    flee = np.clip(-toward / 2, 0, 1)
    credit = np.where(cfg.require_closing != 0.0,
                      (toward > cfg.closing_eps).astype(np.float64), 1.0)
    scored_dealt = dealt_f * late_hit * credit
    early_dealt = dealt_f * (1.0 - late_hit)
    new_ema = (1.0 - cfg.vel_ema_beta) * vel_ema + cfg.vel_ema_beta * vA
    not_moving = (np.linalg.norm(new_ema, axis=1) < cfg.move_eps).astype(np.float64)
    close_term = np.clip(1.0 - dist / max(cfg.close_radius, 1e-6), 0.0, 1.0)
    Rm = xmat[:, idx.At].reshape(nworld, 9)
    fw = Rm[:, :2]
    fw = fw / (np.linalg.norm(fw, axis=1, keepdims=True) + 1e-6)
    face = np.einsum("wi,wi->w", fw, rel / n[:, None])
    fwd_z, side_z, up_z = Rm[:, 6], Rm[:, 7], Rm[:, 8]
    up = up_z - 0.6 * np.abs(fwd_z) - 0.4 * np.abs(side_z)
    height = np.clip((xpos[:, idx.At, 2] - 0.17) / 0.115, 0.0, 1.0)
    sparc = 6.0 * (scored_dealt - taken_f)
    if cfg.loco_speed <= 0:
        sparc = sparc + 5.0 * (clos - flee)
    legdist = np.linalg.norm(xpos[:, idx.Astrike] - xpos[:, idx.Bt][:, None], axis=2).min(axis=1)
    shaped = cfg.shaping * (-0.15 * dist - 0.20 * legdist + 3.0 * scored_dealt)
    approach = prev_dist - dist
    clean = scored_dealt * (1.0 - taken_f)
    trade = np.minimum(dealt_f, taken_f)
    outward = np.clip(-toward / 2, 0, 1)
    disengage = prev_dealt * outward
    energy = np.mean(np.abs(act[:, :idx.n_hinge]), axis=1)
    airborne = np.maximum(0.0, xpos[:, idx.At, 2] - cfg.airborne_z)
    fire_cost = np.zeros(nworld)
    fire_aim = np.zeros(nworld)
    if idx.strike_local.size:
        fi = np.clip(act[:, idx.strike_local], 0.0, 1.0)
        fire_cost = cfg.fire_cost * fi.sum(axis=1)
        d_rb = np.linalg.norm(h["geom_xpos"][:, idx.Arod_gids] - xpos[:, idx.Bt][:, None], axis=2)
        fire_aim = (fi * np.maximum(0.0, 1.0 - d_rb / cfg.rod_reach)).sum(axis=1)
    loco_track = np.zeros(nworld)
    if cfg.loco_speed > 0:
        v_des = cfg.loco_speed * np.minimum(1.0, dist / 0.5)
        cmd = rel / n[:, None] * v_des[:, None]
        loco_track = np.exp(-np.sum((vA - cmd) ** 2, axis=1) / 0.25)
    combat = (sparc + shaped
              + cfg.clean_w * clean - cfg.trade_w * trade + cfg.dis_w * disengage
              - cfg.flee_w * flee + cfg.close_bonus_w * close_term
              + cfg.face_w * close_term * face
              + cfg.damage_bonus_w * scored_dealt
              + cfg.fire_shaping * fire_aim)
    reward = (cfg.combat_scale * combat + cfg.loco_track_w * loco_track
              + cfg.approach_w * approach + cfg.upright_w * up + cfg.alive
              + cfg.height_w * height + cfg.move_w * move
              - fire_cost - cfg.energy_w * energy
              - cfg.airborne_w * airborne - cfg.early_hit_penalty * early_dealt
              - cfg.taken_w * taken_f
              - cfg.penalty_w * np.maximum(0.0, pen_peak - cfg.penalty_tol)
              - cfg.stationary_pen * dealt_f * not_moving
              - cfg.oscillation_pen * energy * not_moving)
    done = ((xpos[:, idx.At, 2] < cfg.fall_z) | (up_z < cfg.topple_up_z)).astype(np.float64)
    return reward, done, dist, scored_dealt, new_ema, t + 1.0
