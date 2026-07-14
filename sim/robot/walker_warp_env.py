# SPDX-License-Identifier: MIT
"""Batched MuJoCo-Warp velocity-command environment for the 12-servo walker.

The warp-path RL stack of mesh_warp_env.py, re-pointed from the combat leg
(mesh_robot.xml, slider-crank strike mechanism) at the CLEAN walking geometry
walker_improved.build_walker (notes/gait-feasibility-verdict.md CORRECTION):
12 actuators, per leg a hip_yaw HINGE, a pitch HINGE, and a lift SLIDE — and NO
loop joints. Everything structural is inherited from mesh_warp_env (the spec):

  * physics: mesh_warp_env._pd_ctrl (a GENERIC P-only servo kernel with the
    servo torque-speed derating + curriculum alpha) x frame_skip substeps of
    mjwp.step, CUDA-graph-captured on GPU / eager on CPU — same code both ways;
  * obs / reward / command / autoreset: torch on dlpack views of the mjwarp Data
    arrays, zero-copy; reward weights are mesh_commanded_env's module constants
    (imported as SPEC, one home for every gain); the imitation hook, action
    low-pass, and curriculum knobs are mesh_warp_env's (IMIT_W etc.);
  * asymmetric critic: privileged() analogous to the mesh env (foot contact /
    height / penetration, qfrc_actuator, TRUE root vel) with the four passive
    slide positions REPLACED by the four actuated lift positions (the walker's
    deep-knee mechanism state).

Adaptations vs mesh_warp_env, all because the walker has no slider-crank loop:
  1. model = build_walker(floor=True) — feet need ground contact; all 12 motors are
     Waveshare ST3215-HS units: yaw servo+SEA 11.77 N.m / 1.85 rad/s, pitch through
     the worm 39.23 N.m / 0.56 rad/s, and lift through the crank 49.03 N / 0.44 m/s.
     Gears come from the built model and wfree from robot_design.TARGET.wfrees().
     The yaw is a series-elastic belt:
     the motor drives a rotor DOF coupled to hip_yaw by a soft equality, so the
     actuated yaw joint is {L}_yaw_rotor (nq/nv grow by 4). PD gains WALKER_KP.
  2. reset = nominal stance (yaw 0, pitch 0, lift parked at DEFAULTS['lift_nom'])
     + small per-joint noise, clamped to range. NO loop_consistent_pose. Torso
     starts at stance_h (~0.42) and DROP-SETTLES onto its feet (settled z~0.38).
  3. imitation target = dynamically validated reference_gait_walker.json;
     joint_order is already the walker actuator order, so its permutation is identity.

Autoreset immediately reruns forward kinematics after replacing qpos/qvel. This
makes returned observations, privileged contact/force features, and the next
critic input describe the same post-reset state.
"""
from __future__ import annotations

import os
import sys
import time
import hashlib
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

import mujoco  # noqa: E402
import torch  # noqa: E402
import warp as wp  # noqa: E402
import mujoco_warp as mjwp  # noqa: E402

import mesh_locomotion_spec as SPEC  # noqa: E402
from constants import FRAME_SKIP, TIMESTEP  # noqa: E402
import walker_improved as WALK  # noqa: E402
from walker_improved import DEFAULTS, LEGS, build_walker  # noqa: E402
# shared warp-path machinery — the P-only servo kernel, imitation hook, telemetry,
# and the annealable imitation / action-low-pass knobs all come from the mesh env
# so both geometries read them from one place (MESH_IMIT_W / MESH_ACT_LP / ...).
from mesh_warp_env import (  # noqa: E402
    ACT_LP, IMIT_FEET_W, IMIT_SIGMA, IMIT_W, EvalTelemetry, _pd_ctrl,
    load_reference_gait)

# --- walker-only knobs, read ONCE (same discipline as the spec module) --------
# PD gains: yaw ~30 N·m/rad, pitch ~80 N·m/rad, lift ~1000 N/m position servo on
# the SLIDE — validated to hold an upright stance (torso z 0.378, up_z 1.0, all
# four feet in contact) under the servo torque-speed model.
WALKER_KP = (float(os.environ.get("WALKER_KP_YAW", "30.0")),
             float(os.environ.get("WALKER_KP_PITCH", "80.0")),
             float(os.environ.get("WALKER_KP_LIFT", "1000.0")))
# action authority as a fraction of each joint's half-range: yaw/pitch 0.6 (the
# mesh AUTHORITY_FRAC), lift 1.0 so ±1 reaches the full 0..lift_range travel.
WALKER_AUTH = (0.6, 0.6, 1.0)
RESET_NOISE_HINGE = float(os.environ.get("WALKER_RESET_NOISE", "0.05"))  # rad, yaw+pitch
RESET_NOISE_LIFT = float(os.environ.get("WALKER_RESET_NOISE_LIFT", "0.005"))  # m, slide
CMD_MODE = os.environ.get("WALKER_CMD_MODE", "sample").strip().lower()
FIXED_CMD_X = float(os.environ.get("WALKER_CMD_X", str(SPEC.MESH_VMAX)))
FIXED_CMD_Y = float(os.environ.get("WALKER_CMD_Y", "0.0"))
FIXED_CMD_YAW = float(os.environ.get("WALKER_CMD_YAW", "0.0"))

# --- anti-hack reward + Constraints-as-Terminations knobs, read ONCE ----------
# The velocity-tracking reward alone admits a family of hacks: standing still
# (good upright/no slip), creep/foot-drag (smooth velocity without real swing),
# in-place hopping (feet lift but no task progress), sliding, and flailing. The
# walker path treats those as constraint failures first, reward tradeoffs second:
#   (A) a bounded reward still teaches task preference and a nonzero reference
#       gait prior keeps the final policy inside the walking state distribution;
#   (B) CaT (Chane-Sane et al. 2403.18765) expresses non-negotiable requirements
#       as stochastic terminations, so a violating world cannot buy its way out
#       with task reward.
SLIP_W = float(os.environ.get("MESH_SLIP_W", "2.0"))       # (A1) drag-a-planted-foot penalty
CLEAR_W = float(os.environ.get("MESH_CLEAR_W", "1.0"))     # (A2) lift-the-swing-foot bonus
CLEAR_TARGET = float(os.environ.get("MESH_CLEAR_TARGET", "0.04"))  # m, clearance cap
WALKER_CONTACT_Z = float(os.environ.get("WALKER_FOOT_CONTACT_Z", "0.05"))  # conservative true-air proxy
MOTION_PRIOR_W = float(os.environ.get("MESH_MOTION_PRIOR_W", "2.0"))  # non-annealed gait prior
CAT_ON = int(os.environ.get("MESH_CAT", "1")) != 0         # (B) master switch
CAT_SLIP_LIMIT = float(os.environ.get("MESH_CAT_SLIP_LIMIT", "0.15"))  # m/s, THE anti-creep limit
CAT_PMAX = float(os.environ.get("MESH_CAT_PMAX", "1.0"))   # max per-step termination prob
CAT_UP_MIN = float(os.environ.get("MESH_CAT_UP_MIN", "0.85"))  # upright-constraint floor on up_z
CAT_QVEL_LIMIT = float(os.environ.get("MESH_CAT_QVEL_LIMIT", "40.0"))  # rad/s, generous joint-speed cap
CAT_PROGRESS_FRAC = float(os.environ.get("MESH_CAT_PROGRESS_FRAC", "0.20"))
CAT_PROGRESS_MIN = float(os.environ.get("MESH_CAT_PROGRESS_MIN", "0.025"))  # m/s, active-command floor
CAT_PROGRESS_EMA = float(os.environ.get("MESH_CAT_PROGRESS_EMA", "0.95"))   # 50 Hz low-pass
CAT_PROGRESS_GRACE_STEPS = int(os.environ.get("MESH_CAT_PROGRESS_GRACE_STEPS", "10"))
CAT_DUTY_MAX = float(os.environ.get("MESH_CAT_DUTY_MAX", "0.90"))            # commanded gait must step
CAT_FOOT_DUTY_MAX = float(os.environ.get("MESH_CAT_FOOT_DUTY_MAX", "0.95"))  # every foot must lift
CAT_DUTY_EMA = float(os.environ.get("MESH_CAT_DUTY_EMA", "0.90"))            # rolling duty window
CAT_DUTY_GRACE_STEPS = int(os.environ.get("MESH_CAT_DUTY_GRACE_STEPS", "10"))
CAT_MIN_CONTACTS = float(os.environ.get("MESH_CAT_MIN_CONTACTS", "1.0"))    # no all-air hopping
CAT_BODY_VZ_LIMIT = float(os.environ.get("MESH_CAT_BODY_VZ_LIMIT", "1.5"))  # m/s, anti-flail
CAT_BODY_ANGXY_LIMIT = float(os.environ.get("MESH_CAT_BODY_ANGXY_LIMIT", "8.0"))  # rad/s
# orientation violation relu(CAT_UP_MIN - up_z) normalizes over the band down to
# the hard-fall floor (SPEC.MIN_UP_Z): reaching the fall boundary => full violation.
CAT_UP_SCALE = max(CAT_UP_MIN - SPEC.MIN_UP_Z, 1e-6)
CAT_TERM_KEYS = ("cat_slip", "cat_orient", "cat_qvel", "cat_progress", "cat_duty",
                 "cat_foot_duty", "cat_support", "cat_body")
REWARD_MODE = os.environ.get("WALKER_REWARD_MODE", "walk").strip().lower()
SPRINT_CAT_TERM_KEYS = ("cat_orient", "cat_qvel", "cat_body")
HOP_CAT_TERM_KEYS = ("cat_orient", "cat_qvel")
SPRINT_SPEED_W = float(os.environ.get("WALKER_SPRINT_SPEED_W", "10.0"))
SPRINT_UPRIGHT_W = float(os.environ.get("WALKER_SPRINT_UPRIGHT_W", "0.5"))
SPRINT_ALIGN_W = float(os.environ.get("WALKER_SPRINT_ALIGN_W", "0.25"))
SPRINT_BACKWARD_W = float(os.environ.get("WALKER_SPRINT_BACKWARD_W", "15.0"))
SPRINT_LATERAL_W = float(os.environ.get("WALKER_SPRINT_LATERAL_W", "1.0"))
SPRINT_ACTRATE_W = float(os.environ.get("WALKER_SPRINT_ACTRATE_W", str(SPEC.ACTRATE_W)))
SPRINT_POSE_W = float(os.environ.get("WALKER_SPRINT_POSE_W", str(SPEC.POSE_W)))
SPRINT_VELZ_W = float(os.environ.get("WALKER_SPRINT_VELZ_W", str(SPEC.VELZ_W)))
SPRINT_ANGXY_W = float(os.environ.get("WALKER_SPRINT_ANGXY_W", str(SPEC.ANGXY_W)))
HOP_HEIGHT_W = float(os.environ.get("WALKER_HOP_HEIGHT_W", "80.0"))       # pay only new peak z
HOP_UPVEL_W = float(os.environ.get("WALKER_HOP_UPVEL_W", "1.5"))          # takeoff impulse
HOP_AIR_W = float(os.environ.get("WALKER_HOP_AIR_W", "0.05"))             # small true-air bonus
HOP_LAND_W = float(os.environ.get("WALKER_HOP_LAND_W", "8.0"))            # one-shot stable landing
HOP_BAD_LAND_W = float(os.environ.get("WALKER_HOP_BAD_LAND_W", "4.0"))
HOP_FALL_W = float(os.environ.get("WALKER_HOP_FALL_W", "10.0"))
HOP_UPRIGHT_W = float(os.environ.get("WALKER_HOP_UPRIGHT_W", "0.05"))
HOP_TIME_W = float(os.environ.get("WALKER_HOP_TIME_W", "0.02"))
HOP_DRIFT_W = float(os.environ.get("WALKER_HOP_DRIFT_W", "1.0"))
HOP_ACTRATE_W = float(os.environ.get("WALKER_HOP_ACTRATE_W", str(SPEC.ACTRATE_W)))
HOP_POSE_W = float(os.environ.get("WALKER_HOP_POSE_W", str(SPEC.POSE_W)))
HOP_ANGXY_W = float(os.environ.get("WALKER_HOP_ANGXY_W", "0.1"))
HOP_AIR_CONTACTS_MAX = float(os.environ.get("WALKER_HOP_AIR_CONTACTS_MAX", "0.5"))
HOP_LAND_CONTACTS = float(os.environ.get("WALKER_HOP_LAND_CONTACTS", "2.0"))
HOP_AIR_MIN_Z = float(os.environ.get("WALKER_HOP_AIR_MIN_Z", str(DEFAULTS["stance_h"] + 0.02)))
HOP_LAND_UP_MIN = float(os.environ.get("WALKER_HOP_LAND_UP_MIN", "0.90"))
HOP_LAND_VZ_MAX = float(os.environ.get("WALKER_HOP_LAND_VZ_MAX", "0.7"))
HOP_LAND_ANGXY_MAX = float(os.environ.get("WALKER_HOP_LAND_ANGXY_MAX", "4.0"))

REFERENCE_GAIT = HERE / "reference_gait_walker.json"
OBS_DIM = 50   # 12 qpos + 12 qvel + 4 quat + 6 root vel + 1 z + 12 prev_a + 3 cmd
PRIV_DIM = 34  # 4 contact + 4 foot_z + 4 penetration + 12 qfrc + 6 root vel + 4 lift


class WalkerWarpEnv:
    """Batched (nworld) walker velocity-command env; reward semantics = mesh_commanded_env."""

    def __init__(self, nworld: int, seed: int = 0, device: str | None = None,
                 frame_skip: int = FRAME_SKIP, episode_length: int | None = None,
                 nconmax: int | None = None, njmax: int | None = 128,
                 gait_path: Path | str = REFERENCE_GAIT, model_transform=None,
                 model_xml_transform=None):
        wp.init()
        use_cuda = torch.cuda.is_available() if device is None else str(device).startswith("cuda")
        self.device = torch.device("cuda:0" if use_cuda else "cpu")
        self._wp_device = wp.get_device("cuda:0" if use_cuda else "cpu")
        if self._wp_device.is_cuda:
            # torch adopts warp's stream: eager torch and graph launches are totally
            # ordered on ONE stream — no cross-stream syncs anywhere.
            torch.cuda.set_stream(torch.cuda.ExternalStream(
                wp.get_stream(self._wp_device).cuda_stream, device=self.device))

        model_xml = build_walker(DEFAULTS, floor=True)
        if model_xml_transform is not None:
            model_xml = model_xml_transform(model_xml)
        m = mujoco.MjModel.from_xml_string(model_xml)  # from spec, never a disk path
        if model_transform is not None:
            model_transform(m)
        self.model_hash = hashlib.sha256(
            model_xml.encode() + m.body_mass.tobytes() + m.body_inertia.tobytes()
            + m.dof_damping.tobytes() + m.jnt_stiffness.tobytes()).hexdigest()[:16]
        assert abs(m.opt.timestep - TIMESTEP) < 1e-12, "walker dt drifted from constants.TIMESTEP"
        assert m.nu == 12, f"walker must have 12 actuators, got {m.nu}"
        self.mjm = m
        self.nworld, self._fs = int(nworld), int(frame_skip)
        self._dt = frame_skip * m.opt.timestep                   # 0.02 s control step
        self._nu = int(m.nu)
        self._episode_length = episode_length
        dev = self.device

        # --- actuated-joint addressing in actuator order (yaw, pitch, lift x 4) --
        aj = [int(m.actuator_trnid[a, 0]) for a in range(m.nu)]
        jname = lambda j: mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_JOINT, j) or ""  # noqa: E731
        self.joint_names = [jname(j) for j in aj]
        lt = lambda x: torch.as_tensor(np.asarray(x), dtype=torch.long, device=dev)      # noqa: E731
        ft = lambda x: torch.as_tensor(np.asarray(x), dtype=torch.float32, device=dev)   # noqa: E731
        self._qa = lt([m.jnt_qposadr[j] for j in aj])
        self._da = lt([m.jnt_dofadr[j] for j in aj])
        jr = np.array([m.jnt_range[j] for j in aj])
        self._jr_lo, self._jr_hi = ft(jr[:, 0]), ft(jr[:, 1])
        gear = np.array([float(m.actuator_gear[a, 0]) for a in range(m.nu)])
        kp = np.array(list(WALKER_KP) * 4)
        # Per-actuator torque-speed no-load speeds from TARGET: yaw/pitch rad/s,
        # lift slide m/s. Gear and speed are emitted from the same ST3215-HS spec.
        wfree = np.array(WALK._DESIGN.wfrees())
        self._kp_t, self._wfree_t, self._gear_t = ft(kp), ft(wfree), ft(gear)
        # per-leg lift SLIDE addressing (the deep-knee mechanism state / priv tensor)
        self._lift_q = lt([m.jnt_qposadr[j] for j in aj[2::3]])
        feet_gid = [mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_GEOM, f"{L}_foot") for L in LEGS]
        self._feet = lt(feet_gid)
        # foot geom -> parent body -> kinematic-tree root, for the cvel->point
        # velocity transform (mjwarp cvel is a 6D spatial velocity [ang; lin] about
        # subtree_com[body_rootid], global orientation): v_foot = v_lin + w x (foot - ref).
        feet_bid = [int(m.geom_bodyid[g]) for g in feet_gid]
        self._feet_body = lt(feet_bid)
        self._feet_root = lt([int(m.body_rootid[b]) for b in feet_bid])
        self._torso = int(mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, "torso"))
        self._foot_r = float(WALK.FOOT_R)

        # nominal stance: qpos0 (torso at stance_h, joints 0) with the lift SLIDEs
        # parked at DEFAULTS['lift_nom'] (mid-range deep-knee, leaves clearance both
        # ways). _stand is the PD target center AND the pose-deviation reference.
        q0 = m.qpos0.copy()
        q0[self._lift_q.cpu().numpy()] = DEFAULTS["lift_nom"]
        self._q0_64 = torch.as_tensor(q0, dtype=torch.float64, device=dev)
        self._stand = ft(q0[self._qa.cpu().numpy()])
        frac = np.array(list(WALKER_AUTH) * 4)
        self._authority = ft(frac * 0.5 * (jr[:, 1] - jr[:, 0]))
        # per-joint reset noise (rad on the two hinges, m on the lift slide)
        self._reset_noise = torch.as_tensor(
            np.array([RESET_NOISE_HINGE, RESET_NOISE_HINGE, RESET_NOISE_LIFT] * 4),
            dtype=torch.float64, device=dev)

        # --- mjwarp model/data + zero-copy torch views -------------------------
        mjd0 = mujoco.MjData(m)
        mujoco.mj_resetData(m, mjd0)
        mujoco.mj_forward(m, mjd0)
        with wp.ScopedDevice(self._wp_device):
            self._wm = mjwp.put_model(m)
            self._wd = mjwp.put_data(m, mjd0, nworld=nworld, nconmax=nconmax, njmax=njmax)
            self._target_wp = wp.zeros((nworld, m.nu), dtype=wp.float32)
            self._alpha_wp = wp.zeros(1, dtype=wp.float32)
            wpf = lambda x: wp.array(np.asarray(x, dtype=np.float32), dtype=wp.float32)  # noqa: E731
            wpi = lambda x: wp.array(np.asarray(x, dtype=np.int32), dtype=wp.int32)      # noqa: E731
            self._kp_wp, self._wfree_wp, self._gear_wp = wpf(kp), wpf(wfree), wpf(gear)
            self._qa_wp = wpi([m.jnt_qposadr[j] for j in aj])
            self._da_wp = wpi([m.jnt_dofadr[j] for j in aj])
        self.qpos = wp.to_torch(self._wd.qpos)             # (nworld, nq) — ALIASES mjwarp Data
        self.qvel = wp.to_torch(self._wd.qvel)             # (nworld, nv)
        self.xpos = wp.to_torch(self._wd.xpos)             # (nworld, nbody, 3)
        self.geom_xpos = wp.to_torch(self._wd.geom_xpos)   # (nworld, ngeom, 3)
        self.cvel = wp.to_torch(self._wd.cvel)             # (nworld, nbody, 6) [ang; lin]
        self.subtree_com = wp.to_torch(self._wd.subtree_com)      # (nworld, nbody, 3)
        self.qfrc_actuator = wp.to_torch(self._wd.qfrc_actuator)  # (nworld, nv)
        self.qacc_warmstart = wp.to_torch(self._wd.qacc_warmstart)
        self.sim_time = wp.to_torch(self._wd.time)
        self.constraint_rows = wp.to_torch(self._wd.nefc)
        self.solver_iterations = wp.to_torch(self._wd.solver_niter)
        # The island-index buffer can legitimately be zero-width; Data.njmax
        # is the configured constraint-row capacity used by the solver.
        self.constraint_capacity = int(self._wd.njmax)
        self._target_t = wp.to_torch(self._target_wp)
        self._alpha_t = wp.to_torch(self._alpha_wp)

        # --- per-world state buffers (torch-owned) ------------------------------
        self._gen = torch.Generator(device=dev)
        self._gen.manual_seed(seed)
        self._cmd = torch.zeros((nworld, 3), device=dev)
        self._timer = torch.zeros(nworld, dtype=torch.long, device=dev)
        self._t = torch.zeros(nworld, dtype=torch.long, device=dev)
        self._air = torch.zeros((nworld, 4), device=dev)
        self._prev_a = torch.zeros((nworld, self._nu), device=dev)
        self._prev_xy = torch.zeros((nworld, 2), device=dev)
        self._progress_ema = torch.zeros(nworld, device=dev)
        self._duty_ema = torch.ones(nworld, device=dev)
        self._foot_duty_ema = torch.ones((nworld, 4), device=dev)
        self._hop_peak_z = torch.full((nworld,), float(DEFAULTS["stance_h"]), device=dev)
        self._hop_airborne = torch.zeros(nworld, dtype=torch.bool, device=dev)
        self._dirs = ft([[1.0, 0.0], [0.0, 1.0], [-1.0, 0.0], [0.0, -1.0]])
        self._pair_a = ft([1.0, 0.0, 0.0, 1.0])            # diagonal pair A = (FL, RR)
        self._pair_b = ft([0.0, 1.0, 1.0, 0.0])            # diagonal pair B = (FR, RL)
        # the reference gait names hip_yaw; with the TARGET SEA belt the ACTUATED yaw
        # joint is the rotor (motor side), so map rotor -> hip_yaw for the imitation
        # lookup. The joint-side crawl reference is applied to the rotor target; the
        # belt keeps rotor ~= joint, so the imitation is consistent. Identity when SEA
        # is off (CURRENT), where the actuated yaw joint IS hip_yaw.
        gait_names = [n.replace("_yaw_rotor", "_hip_yaw") for n in self.joint_names]
        self._gait = load_reference_gait(gait_path, gait_names, dev)

        # --- CUDA graph capture (CPU runs the same sequence eagerly) ------------
        self._graph = None
        if self._wp_device.is_cuda:
            with wp.ScopedDevice(self._wp_device):
                self._substep()                            # load modules before capture
                wp.synchronize_device(self._wp_device)
                with wp.ScopedCapture() as cap:
                    for _ in range(self._fs):
                        self._substep()
                self._graph = cap.graph
        self.reset()                                       # also repairs the warmup step above

    # ------------------------------------------------------------------ plumbing
    @property
    def obs_dim(self):
        return OBS_DIM

    @property
    def priv_dim(self):
        return PRIV_DIM

    @property
    def act_dim(self):
        return self._nu

    @property
    def dt(self):
        return self._dt

    @property
    def gait_loaded(self):
        return self._gait is not None

    def _substep(self):
        wp.launch(_pd_ctrl, dim=(self.nworld, self._nu),
                  inputs=[self._wd.qpos, self._wd.qvel, self._target_wp,
                          self._qa_wp, self._da_wp, self._kp_wp, self._wfree_wp,
                          self._gear_wp, self._alpha_wp, self._wd.ctrl],
                  device=self._wp_device)
        mjwp.step(self._wm, self._wd)

    def _run_physics(self):
        with wp.ScopedDevice(self._wp_device):
            if self._graph is not None:
                wp.capture_launch(self._graph)
            else:
                for _ in range(self._fs):
                    self._substep()

    # ------------------------------------------------------------------ commands
    def _sample_cmd(self) -> torch.Tensor:
        """Spec lines 138-144: 4 cardinal dirs, |v| ~ U(0.3, 1)*MESH_VMAX, 15% hold."""
        n, dev = self.nworld, self.device
        if CMD_MODE == "fixed":
            cmd = torch.tensor([FIXED_CMD_X, FIXED_CMD_Y, FIXED_CMD_YAW],
                               dtype=torch.float32, device=dev)
            return cmd.expand(n, -1)
        idx = torch.randint(0, 4, (n,), generator=self._gen, device=dev)
        spd = (torch.rand(n, generator=self._gen, device=dev)
               * (0.7 * SPEC.MESH_VMAX) + 0.3 * SPEC.MESH_VMAX)
        hold = (torch.rand(n, generator=self._gen, device=dev) < 0.15).float()
        xy = self._dirs[idx] * (spd * (1.0 - hold)).unsqueeze(-1)
        return torch.cat([xy, torch.zeros(n, 1, device=dev)], dim=-1)

    # ------------------------------------------------------------------ reset
    def reset(self, mask: torch.Tensor | None = None) -> torch.Tensor:
        """Reset all worlds (mask=None) or the masked slice, then refresh
        kinematics (mjwp.forward) so xpos/geom_xpos/privileged() are exact."""
        if mask is None:
            mask = torch.ones(self.nworld, dtype=torch.bool, device=self.device)
        self._reset_worlds(mask)
        with wp.ScopedDevice(self._wp_device):
            mjwp.forward(self._wm, self._wd)
        return self.observe()

    def _reset_worlds(self, mask: torch.Tensor):
        """Branchless in-place per-world reset through the Data views (no host
        syncs — an all-False mask is a no-op). Nominal stance (yaw 0, pitch 0,
        lift parked at DEFAULTS['lift_nom']) + per-joint noise on the ACTUATED
        joints, CLAMPED to joint range; qvel = 0; torso left at stance_h to
        DROP-SETTLE onto its feet. No loop joints -> no loop_consistent_pose."""
        n, dev = self.nworld, self.device
        mf = mask.unsqueeze(-1)
        noise = ((torch.rand((n, self._nu), generator=self._gen, device=dev,
                             dtype=torch.float64) * 2.0 - 1.0) * self._reset_noise)
        q = self._q0_64.expand(n, -1).clone()
        qa64 = q[:, self._qa] + noise
        qa64 = torch.clamp(qa64, self._jr_lo.to(torch.float64), self._jr_hi.to(torch.float64))
        q[:, self._qa] = qa64
        qf = q.to(torch.float32)
        self.qpos.copy_(torch.where(mf, qf, self.qpos))
        self.qvel.copy_(torch.where(mf, torch.zeros_like(self.qvel), self.qvel))
        self.qacc_warmstart.copy_(torch.where(
            mf, torch.zeros_like(self.qacc_warmstart), self.qacc_warmstart))
        self.sim_time.copy_(torch.where(mask, torch.zeros_like(self.sim_time), self.sim_time))
        self._cmd = torch.where(mf, self._sample_cmd(), self._cmd)
        zl = torch.zeros_like(self._t)
        self._timer = torch.where(mask, zl, self._timer)
        self._t = torch.where(mask, zl, self._t)
        self._air = torch.where(mf, torch.zeros_like(self._air), self._air)
        self._prev_a = torch.where(mf, torch.zeros_like(self._prev_a), self._prev_a)
        self._prev_xy = torch.where(mask.unsqueeze(-1), qf[:, 0:2], self._prev_xy)
        self._progress_ema = torch.where(mask, torch.zeros_like(self._progress_ema),
                                         self._progress_ema)
        self._duty_ema = torch.where(mask, torch.ones_like(self._duty_ema), self._duty_ema)
        self._foot_duty_ema = torch.where(mf, torch.ones_like(self._foot_duty_ema),
                                          self._foot_duty_ema)
        self._hop_peak_z = torch.where(mask, torch.full_like(self._hop_peak_z, float(DEFAULTS["stance_h"])),
                                       self._hop_peak_z)
        self._hop_airborne = torch.where(mask, torch.zeros_like(self._hop_airborne),
                                         self._hop_airborne)

    # ------------------------------------------------------------------ obs
    def _yaw_cs(self):
        """Yaw rotation of spec lines 120-125: R = [[c, s], [-s, c]]."""
        q = self.qpos
        w, x, y, z = q[:, 3], q[:, 4], q[:, 5], q[:, 6]
        yaw = torch.atan2(2.0 * (w * z + x * y), 1.0 - 2.0 * (y * y + z * z))
        return torch.cos(yaw), torch.sin(yaw)

    def observe(self) -> torch.Tensor:
        """The 50-obs layout of spec lines 127-136, verbatim order (nu=12 both
        geometries, so the vector is byte-for-byte the same shape as the mesh env)."""
        q, v, cmd = self.qpos, self.qvel, self._cmd
        c, s = self._yaw_cs()
        v_loc = torch.stack([c * v[:, 0] + s * v[:, 1], -s * v[:, 0] + c * v[:, 1]], dim=-1)
        c_loc = torch.stack([c * cmd[:, 0] + s * cmd[:, 1], -s * cmd[:, 0] + c * cmd[:, 1]], dim=-1)
        return torch.cat([q[:, self._qa], v[:, self._da], q[:, 3:7],
                          v_loc, v[:, 2:6], q[:, 2:3],
                          self._prev_a, c_loc, cmd[:, 2:3]], dim=-1)

    def _foot_hspeed(self) -> torch.Tensor:
        """(nworld, 4) world-frame HORIZONTAL speed of each foot geom, read fresh
        from the mjwarp velocity fields (no finite differencing, so it is exact
        even for a just-reset world). mjwarp cvel[b] is the 6D spatial velocity
        [angular; linear] of body b about subtree_com[body_rootid[b]] in the world
        frame; the linear velocity of the foot geom point is
        v_foot = v_lin + w x (geom_xpos_foot - subtree_com[root])  (validated exact
        vs mujoco.mj_objectVelocity). Horizontal = xy norm."""
        w = self.cvel[:, self._feet_body, 0:3]
        v_lin = self.cvel[:, self._feet_body, 3:6]
        ref = self.subtree_com[:, self._feet_root, :]
        arm = self.geom_xpos[:, self._feet, :] - ref
        v_foot = v_lin + torch.linalg.cross(w, arm, dim=-1)
        return torch.linalg.vector_norm(v_foot[..., :2], dim=-1)

    def _cat_violations(self, cf: torch.Tensor, foot_hspeed: torch.Tensor,
                        up: torch.Tensor, cmd_norm: torch.Tensor,
                        active: torch.Tensor, progress_mature: torch.Tensor,
                        duty_mature: torch.Tensor,
                        progress_ema: torch.Tensor,
                        duty_ema: torch.Tensor,
                        foot_duty_ema: torch.Tensor) -> tuple[dict[str, torch.Tensor], torch.Tensor]:
        """Normalized CaT violation magnitudes. Zero means the constraint is
        satisfied; values around 1 mean a full-scale violation for stochastic
        termination. Kept as one helper so tests exercise the exact runtime
        constraint definitions."""
        z = torch.zeros_like(cmd_norm)
        active_progress = (active > 0.0) & progress_mature
        active_duty = (active > 0.0) & duty_mature
        progress_req = torch.maximum(torch.full_like(cmd_norm, CAT_PROGRESS_MIN),
                                     CAT_PROGRESS_FRAC * cmd_norm)

        v_slip = ((foot_hspeed * cf).amax(-1) - CAT_SLIP_LIMIT).clamp(min=0.0)
        v_orient = (CAT_UP_MIN - up).clamp(min=0.0)
        v_qvel = (self.qvel[:, self._da].abs().amax(-1) - CAT_QVEL_LIMIT).clamp(min=0.0)
        v_progress = torch.where(active_progress,
                                 (progress_req - progress_ema).clamp(min=0.0), z)
        v_duty = torch.where(active_duty, (duty_ema - CAT_DUTY_MAX).clamp(min=0.0), z)
        v_foot_duty = torch.where(active_duty,
                                  (foot_duty_ema.amax(-1) - CAT_FOOT_DUTY_MAX).clamp(min=0.0), z)
        contact_count = cf.sum(-1)
        if CAT_MIN_CONTACTS > 0.0:
            v_support = torch.where(active_duty,
                                    (CAT_MIN_CONTACTS - contact_count).clamp(min=0.0), z)
            support_scale = CAT_MIN_CONTACTS
        else:
            v_support = z
            support_scale = 1.0
        v_body_vz = (self.qvel[:, 2].abs() - CAT_BODY_VZ_LIMIT).clamp(min=0.0)
        v_body_ang = (self.qvel[:, 3:5].abs().amax(-1) - CAT_BODY_ANGXY_LIMIT).clamp(min=0.0)
        terms = {
            "cat_slip": v_slip / max(CAT_SLIP_LIMIT, 1e-6),
            "cat_orient": v_orient / CAT_UP_SCALE,
            "cat_qvel": v_qvel / max(CAT_QVEL_LIMIT, 1e-6),
            "cat_progress": v_progress / progress_req.clamp(min=1e-6),
            "cat_duty": v_duty / max(1.0 - CAT_DUTY_MAX, 1e-6),
            "cat_foot_duty": v_foot_duty / max(1.0 - CAT_FOOT_DUTY_MAX, 1e-6),
            "cat_support": v_support / max(support_scale, 1e-6),
            "cat_body": torch.maximum(v_body_vz / max(CAT_BODY_VZ_LIMIT, 1e-6),
                                      v_body_ang / max(CAT_BODY_ANGXY_LIMIT, 1e-6)),
        }
        return terms, progress_req

    def privileged(self) -> torch.Tensor:
        """(nworld, 34) critic-only tensor: per-foot contact / height / penetration,
        qfrc_actuator at the actuated dofs, TRUE root velocity, and the four lift
        SLIDE positions (the walker's deep-knee mechanism state — the analog of the
        mesh env's four passive loop slides). Never enters the 50-obs policy input."""
        foot_z = self.geom_xpos[:, self._feet, 2]
        contact = (foot_z < WALKER_CONTACT_Z).float()
        pen = (self._foot_r - foot_z).clamp(min=0.0)
        return torch.cat([contact, foot_z, pen,
                          self.qfrc_actuator[:, self._da],
                          self.qvel[:, 0:6], self.qpos[:, self._lift_q]], dim=-1)

    # ------------------------------------------------------------------ step
    def step(self, action: torch.Tensor, alpha: float = 1.0, imit_anneal: float = 1.0):
        """One 50 Hz control step for every world. Returns (obs, reward, done, info);
        obs/priv in info are POST-autoreset, info['terminal_obs'/'terminal_priv']
        are the pre-reset values for truncation bootstrapping.

        alpha: curriculum scalar in [0,1] — 0 disables the servo torque-speed
        derating (lim=1 always), 1 is fully servo-true (see _pd_ctrl).
        imit_anneal: multiplies MESH_IMIT_W (trainer schedules 1 -> 0)."""
        n, dev, dt = self.nworld, self.device, self._dt
        # command hold/resample (spec lines 171-176)
        self._timer = self._timer + 1
        resample = self._timer >= SPEC.CMD_HOLD_STEPS
        self._cmd = torch.where(resample.unsqueeze(-1), self._sample_cmd(), self._cmd)
        self._timer = torch.where(resample, torch.zeros_like(self._timer), self._timer)
        self._progress_ema = torch.where(resample, torch.zeros_like(self._progress_ema),
                                         self._progress_ema)
        # action low-pass; the USED action is what obs reports as prev_action
        a = ACT_LP * self._prev_a + (1.0 - ACT_LP) * action.clamp(-1.0, 1.0)
        raw_target = self._stand + a * self._authority
        target_clamped = ((raw_target < self._jr_lo) | (raw_target > self._jr_hi)).float()
        target = raw_target.clamp(self._jr_lo, self._jr_hi)
        self._target_t.copy_(target)
        self._alpha_t.fill_(float(alpha))
        self._run_physics()

        # --- reward, verbatim port of spec lines 195-230 ------------------------
        q, v = self.qpos, self.qvel
        cmd = self._cmd
        vxy = v[:, 0:2]
        verr = ((vxy - cmd[:, :2]) ** 2).sum(-1)
        track = torch.exp(-verr / SPEC.TRACK_SIGMA)
        up = 1.0 - 2.0 * (q[:, 4] ** 2 + q[:, 5] ** 2)
        cmd_norm = torch.linalg.vector_norm(cmd[:, :2], dim=-1)
        speed = torch.linalg.vector_norm(vxy, dim=-1)
        dot = (vxy * cmd[:, :2]).sum(-1)
        progress = dot / (cmd_norm + 1e-6)
        xdelta = q[:, 0:2] - self._prev_xy
        xprogress = (xdelta * cmd[:, :2]).sum(-1) / ((cmd_norm + 1e-6) * dt)
        align = dot / (speed * cmd_norm + 1e-6)
        cmd_dir = cmd[:, :2] / cmd_norm.clamp(min=1e-6).unsqueeze(-1)
        lateral = torch.linalg.vector_norm(vxy - cmd_dir * progress.unsqueeze(-1), dim=-1)
        active = (cmd_norm > 0.05).float()
        progress_ema = CAT_PROGRESS_EMA * self._progress_ema + (1.0 - CAT_PROGRESS_EMA) * (
            active * xprogress)
        foot_z = self.geom_xpos[:, self._feet, 2]
        contact = foot_z < WALKER_CONTACT_Z
        air = self._air
        first_c = contact & (air > 0.0)
        fcf = first_c.float()
        air_rwd = ((air.clamp(max=SPEC.AIRTIME_CAP) - SPEC.AIRTIME_TARGET) * fcf).sum(-1)
        new_air = torch.where(contact, torch.zeros_like(air), air + dt)
        action_delta = (a - self._prev_a).abs()
        act_rate = action_delta.square().sum(-1)
        pose_dev = ((q[:, self._qa] - self._stand) ** 2).sum(-1)
        progress_c = progress.clamp(-cmd_norm, cmd_norm) / SPEC.MESH_VMAX
        phase = (self._t.float() * dt * SPEC.CLOCK_HZ) % 1.0
        swing_a = (phase < 0.5).float().unsqueeze(-1)
        want = self._pair_a * swing_a + self._pair_b * (1.0 - swing_a)
        cf = contact.float()
        contact_count = cf.sum(-1)
        clock_bonus = (want * (1.0 - cf) + (1.0 - want) * cf).mean(-1)
        height = self.xpos[:, self._torso, 2]
        hop_peak_delta = torch.zeros(n, device=dev)
        hop_airborne = torch.zeros(n, dtype=torch.bool, device=dev)
        hop_landed = torch.zeros(n, dtype=torch.bool, device=dev)
        hop_stable_landing = torch.zeros(n, device=dev)
        new_hop_peak_z = self._hop_peak_z
        new_hop_airborne = self._hop_airborne
        reward_components: dict[str, torch.Tensor]
        if REWARD_MODE == "sprint":
            reward_components = {
                "sprint_progress": SPRINT_SPEED_W * active * xprogress,
                "upright": SPRINT_UPRIGHT_W * up.clamp(min=0.0),
                "alignment": SPRINT_ALIGN_W * active * align.clamp(-1.0, 1.0),
                "backward_penalty": -SPRINT_BACKWARD_W * active
                    * (-xprogress).clamp(min=0.0),
                "lateral_penalty": -SPRINT_LATERAL_W * active * lateral.square(),
                "pose_penalty": -SPRINT_POSE_W * pose_dev,
                "action_rate_penalty": -SPRINT_ACTRATE_W * act_rate,
                "vertical_speed_penalty": -SPRINT_VELZ_W * v[:, 2].square(),
                "angular_speed_penalty": -SPRINT_ANGXY_W
                    * (v[:, 3].square() + v[:, 4].square()),
            }
            reward = sum(reward_components.values())
        elif REWARD_MODE == "hop":
            hop_peak_delta = (height - self._hop_peak_z).clamp(min=0.0)
            new_hop_peak_z = torch.maximum(self._hop_peak_z, height)
            hop_airborne = (contact_count <= HOP_AIR_CONTACTS_MAX) & (height > HOP_AIR_MIN_Z)
            hop_landed = self._hop_airborne & (contact_count >= HOP_LAND_CONTACTS)
            body_angxy = torch.linalg.vector_norm(v[:, 3:5], dim=-1)
            stable_mask = ((up > HOP_LAND_UP_MIN)
                           & (v[:, 2].abs() < HOP_LAND_VZ_MAX)
                           & (body_angxy < HOP_LAND_ANGXY_MAX))
            hop_stable_landing = (hop_landed & stable_mask).float()
            bad_landing = (hop_landed.float() * (1.0 - stable_mask.float()))
            reward_components = {
                "hop_peak": HOP_HEIGHT_W * hop_peak_delta,
                "hop_upward_speed": HOP_UPVEL_W * v[:, 2].clamp(min=0.0),
                "hop_airborne": HOP_AIR_W * hop_airborne.float(),
                "hop_landing": HOP_LAND_W * hop_stable_landing,
                "upright": HOP_UPRIGHT_W * up.clamp(min=0.0),
                "bad_landing_penalty": -HOP_BAD_LAND_W * bad_landing,
                "time_penalty": torch.full_like(speed, -HOP_TIME_W),
                "drift_penalty": -HOP_DRIFT_W * speed.square(),
                "pose_penalty": -HOP_POSE_W * pose_dev,
                "action_rate_penalty": -HOP_ACTRATE_W * act_rate,
                "angular_speed_penalty": -HOP_ANGXY_W
                    * (v[:, 3].square() + v[:, 4].square()),
            }
            reward = sum(reward_components.values())
            new_hop_airborne = torch.where(hop_landed, torch.zeros_like(self._hop_airborne),
                                           self._hop_airborne | hop_airborne)
        else:
            reward_components = {
                "tracking": SPEC.TRACK_W * track,
                "upright": SPEC.UPRIGHT_W * up,
                "alive": torch.full_like(track, 0.1),
                "gait_clock": SPEC.CLOCK_W * active * clock_bonus,
                "alignment": SPEC.ALIGN_W * active * align.clamp(-1.0, 1.0),
                "progress": SPEC.PROGRESS_W * active * progress_c,
                "airtime": SPEC.AIRTIME_W * air_rwd
                    * (cmd_norm / SPEC.MESH_VMAX).clamp(0.0, 1.0),
                "backward_penalty": -SPEC.BACKWARD_W * active
                    * (-progress).clamp(min=0.0),
                "pose_penalty": -SPEC.POSE_W * pose_dev,
                "action_rate_penalty": -SPEC.ACTRATE_W * act_rate,
                "vertical_speed_penalty": -SPEC.VELZ_W * v[:, 2].square(),
                "angular_speed_penalty": -SPEC.ANGXY_W
                    * (v[:, 3].square() + v[:, 4].square()),
            }
            reward = sum(reward_components.values())
        imit = torch.zeros(n, device=dev)
        motion_prior = torch.zeros(n, device=dev)
        if (REWARD_MODE == "walk" and self._gait is not None
                and (MOTION_PRIOR_W > 0.0 or (imit_anneal > 0.0 and IMIT_W > 0.0))):
            g = self._gait
            gp = (self._t.float() * dt / g["period"]) % 1.0
            x = gp * g["n"]
            i0 = x.floor().long() % g["n"]
            i1 = (i0 + 1) % g["n"]
            frac = (x - x.floor()).unsqueeze(-1)
            ref_q = g["qpos"][i0] * (1.0 - frac) + g["qpos"][i1] * frac
            err2 = ((q[:, self._qa] - ref_q) ** 2).sum(-1)
            want_ref = g["swing"][i0]
            want_swing_n = want_ref.sum(-1).clamp(min=1.0)
            want_stance = 1.0 - want_ref
            want_stance_n = want_stance.sum(-1).clamp(min=1.0)
            swing_recall = (want_ref * (1.0 - cf)).sum(-1) / want_swing_n
            stance_recall = (want_stance * cf).sum(-1) / want_stance_n
            feet_agree = 0.7 * swing_recall + 0.3 * stance_recall
            motion_prior = torch.exp(-err2 / IMIT_SIGMA ** 2) + IMIT_FEET_W * feet_agree
            if imit_anneal > 0.0 and IMIT_W > 0.0:
                imit = motion_prior
            prior_reward = ((IMIT_W * imit_anneal) + MOTION_PRIOR_W * active) * motion_prior
            reward = reward + prior_reward
            reward_components["motion_prior"] = prior_reward

        # --- (A) anti-loophole term-level rewards (folded into the sum) ----------
        # Fresh, exact world-frame horizontal foot speed (cvel-derived; reused by
        # the CaT slip constraint below). cf/foot_z are the env's own contact proxy.
        foot_hspeed = self._foot_hspeed()                            # (n, 4)
        duty_ema = CAT_DUTY_EMA * self._duty_ema + (1.0 - CAT_DUTY_EMA) * cf.mean(-1)
        foot_duty_ema = CAT_DUTY_EMA * self._foot_duty_ema + (1.0 - CAT_DUTY_EMA) * cf
        n_contact = contact_count.clamp(min=1.0)
        n_swing = (1.0 - cf).sum(-1).clamp(min=1.0)
        # (A1) FOOT-SLIP PENALTY: dragging a foot that is IN CONTACT costs reward —
        # the direct anti-creep term (a creep drags planted feet forward).
        slip = (foot_hspeed * cf).sum(-1) / n_contact                # mean over contacting feet
        stance_foot_speed = (foot_hspeed * cf).amax(-1)
        # (A2) FOOT-CLEARANCE REWARD: a SWING foot is rewarded for reaching a target
        # ground clearance (capped, so it need only lift high enough, not fly).
        clearance = (foot_z.clamp(max=CLEAR_TARGET) * (1.0 - cf)).sum(-1) / n_swing
        if REWARD_MODE == "walk":
            reward_components["slip_penalty"] = -SLIP_W * slip
            reward_components["clearance"] = CLEAR_W * clearance
            reward = reward + reward_components["slip_penalty"] + reward_components["clearance"]

        fall = (height < SPEC.FALL_Z) | (up < SPEC.MIN_UP_Z)      # spec line 231
        if REWARD_MODE == "hop":
            reward_components["fall_penalty"] = -HOP_FALL_W * fall.float()
            reward = reward + reward_components["fall_penalty"]

        # --- (B) CaT: constraints enforced by stochastic termination ------------
        # Per-step nonneg violations (0 = satisfied), each normalized by ~its own
        # scale; delta = clip(max_c(violation_c/scale_c) * P_MAX, 0, 1) is the
        # per-world termination probability; a fresh uniform from the env RNG (so
        # it is deterministic under a seed) OR-s into done, cutting PPO's future-
        # reward bootstrap on violation (the CaT down-scaling mechanism).
        cat_done = torch.zeros(n, dtype=torch.bool, device=dev)
        cat_delta = torch.zeros(n, device=dev)
        cat_terms, progress_req = self._cat_violations(
            cf, foot_hspeed, up, cmd_norm, active,
            self._timer >= CAT_PROGRESS_GRACE_STEPS,
            self._timer >= CAT_DUTY_GRACE_STEPS,
            progress_ema, duty_ema, foot_duty_ema)
        if CAT_ON:
            cat_keys = (SPRINT_CAT_TERM_KEYS if REWARD_MODE == "sprint"
                        else HOP_CAT_TERM_KEYS if REWARD_MODE == "hop"
                        else CAT_TERM_KEYS)
            norm = torch.stack([cat_terms[k] for k in cat_keys], dim=-1).amax(-1)
            cat_delta = (norm * CAT_PMAX).clamp(0.0, 1.0)
            u = torch.rand(n, generator=self._gen, device=dev)
            cat_done = u < cat_delta

        # --- bookkeeping, terminal snapshot, per-world autoreset ----------------
        air_pre = air.clone()
        self._air = new_air
        self._prev_a = a
        self._prev_xy.copy_(q[:, 0:2])
        self._progress_ema = progress_ema
        self._duty_ema = duty_ema
        self._foot_duty_ema = foot_duty_ema
        self._hop_peak_z = new_hop_peak_z
        self._hop_airborne = new_hop_airborne
        self._t = self._t + 1
        # fall AND cat_done are true TERMINATIONS (no bootstrap); only the time
        # limit is a truncation. CaT worlds are excluded from trunc so their future
        # rewards are correctly cut.
        term = fall | cat_done
        if self._episode_length is not None:
            trunc = (self._t >= self._episode_length) & ~term
        else:
            trunc = torch.zeros(n, dtype=torch.bool, device=dev)
        done = term | trunc
        q_actuated = q[:, self._qa]
        qd_actuated = v[:, self._da]
        requested_effort = self._kp_t * (target - q_actuated) / self._gear_t
        drive_derating = (requested_effort * qd_actuated) > 0.0
        speed_limit = (1.0 - qd_actuated.abs() / self._wfree_t).clamp(0.0, 1.0)
        effective_limit = 1.0 - float(alpha) * (1.0 - speed_limit)
        effective_limit = torch.where(drive_derating, effective_limit,
                                      torch.ones_like(effective_limit))
        effort_ratio = requested_effort.abs() / effective_limit.clamp_min(1.0e-6)
        joint_margin = torch.minimum(q_actuated - self._jr_lo,
                                     self._jr_hi - q_actuated)
        joint_range = (self._jr_hi - self._jr_lo).clamp_min(1.0e-6)
        foot_penetration = (self._foot_r - foot_z).clamp_min(0.0)
        actuator_diagnostics = {
            "target_clamped": target_clamped,
            "effort_saturated": (effort_ratio >= 1.0).float(),
            "effort_ratio": effort_ratio,
            "requested_effort_abs": requested_effort.abs(),
            "available_effort": effective_limit,
            "effort_shortfall": (
                requested_effort.abs() - effective_limit).clamp_min(0.0),
            "speed_derated": (drive_derating & (effective_limit < 0.999)).float(),
            "joint_limit_near": (joint_margin < 0.01 * joint_range).float(),
            "action_delta": action_delta,
        }
        simulation_diagnostics = {
            "constraint_rows": self.constraint_rows.float(),
            "constraint_capacity": torch.full_like(
                self.constraint_rows, float(self.constraint_capacity), dtype=torch.float32),
            "solver_iterations": self.solver_iterations.float(),
            "foot_penetration": foot_penetration.amax(-1),
            "state_nonfinite": ((~torch.isfinite(q)).sum(-1)
                                + (~torch.isfinite(v)).sum(-1)).float(),
        }
        obs_pre = self.observe()
        priv_pre = self.privileged()
        self._reset_worlds(done)                 # branchless per-world autoreset, no host sync
        # Refresh derived kinematics/contact/forces after qpos/qvel replacement.
        # Unconditional execution avoids a done.any() host synchronization on CUDA.
        with wp.ScopedDevice(self._wp_device):
            mjwp.forward(self._wm, self._wd)
        info = {"truncated": trunc.float(), "terminal_obs": obs_pre, "terminal_priv": priv_pre,
                "priv": self.privileged(),
                "cat_done": cat_done.float(), "cat_delta": cat_delta,
                "reward_components": reward_components,
                "actuator_diagnostics": actuator_diagnostics,
                "simulation_diagnostics": simulation_diagnostics,
                "gait_phase": phase,
                "slip": slip, "stance_foot_speed": stance_foot_speed,
                "clearance": clearance,
                "foot_hspeed": foot_hspeed, "foot_height": foot_z,
                "contact": cf, "first_contact": fcf, "air_pre": air_pre,
                "track": track, "verr": torch.sqrt(verr), "align": align, "speed": speed,
                "command_speed": cmd_norm, "forward_efficiency": align,
                "lateral_speed_fraction": lateral / speed.clamp_min(1.0e-3),
                "progress": progress, "xprogress": xprogress, "lateral": lateral,
                "progress_ema": progress_ema,
                "progress_req": progress_req, "duty_ema": duty_ema,
                "foot_duty_ema_by_leg": foot_duty_ema,
                "foot_duty_ema": foot_duty_ema.amax(-1),
                "hop_peak": new_hop_peak_z, "hop_peak_delta": hop_peak_delta,
                "hop_airborne": hop_airborne.float(), "hop_landed": hop_landed.float(),
                "hop_stable_landing": hop_stable_landing,
                "up": up, "height": height, "imit": imit,
                "motion_prior": motion_prior, "fallrate": fall.float(), **cat_terms}
        return self.observe(), reward, done.float(), info


def main():
    """Throughput probe: random actions, prints one RESULT line."""
    import argparse
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--nworld", type=int, default=8)
    ap.add_argument("--steps", type=int, default=100)
    ap.add_argument("--warmup", type=int, default=10)
    ap.add_argument("--device", default=None)
    args = ap.parse_args()
    env = WalkerWarpEnv(args.nworld, seed=0, device=args.device, episode_length=800)
    act = torch.zeros((args.nworld, env.act_dim), device=env.device)
    for _ in range(args.warmup):
        env.step(act.uniform_(-1, 1))
    if env.device.type == "cuda":
        torch.cuda.synchronize()
    t0 = time.time()
    for _ in range(args.steps):
        obs, reward, done, info = env.step(act.uniform_(-1, 1))
    if env.device.type == "cuda":
        torch.cuda.synchronize()
    wall = time.time() - t0
    assert torch.isfinite(obs).all() and torch.isfinite(reward).all()
    print(f"RESULT bench=walker_warp_env nworld={args.nworld} steps={args.steps} "
          f"device={env.device} env_steps_per_s={args.nworld * args.steps / wall:.1f} "
          f"wall_s={wall:.3f}", flush=True)


if __name__ == "__main__":
    main()
