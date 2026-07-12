# SPDX-License-Identifier: MIT
"""mesh_warp_env.py — batched velocity-command mesh-robot env on mujoco_warp + torch.

Warp-path port of sim/robot/mesh_commanded_env.py (THE spec: its module-level
constants are imported here so every reward weight, gain, and MESH_* env knob is
read from one place; obs layout / reward terms / PD path / reset rule cite its
line numbers inline). Split of labor (the M4 pattern, warplayer/m4_train_demo.py):

  * mujoco_warp does physics: frame_skip substeps of (PD-ctrl kernel -> mjwp.step),
    a fixed launch sequence that is CUDA-graph-captured when a GPU is present and
    runs eagerly on CPU — same code both ways;
  * everything else (obs, reward, command logic, autoreset) is torch on dlpack
    views of the mjwarp Data arrays (qpos/qvel/xpos/geom_xpos/qfrc_actuator) —
    zero-copy, no host round-trips. Per-world autoreset writes only the done
    worlds' qpos/qvel slices through the views (branchless torch.where), honoring
    loop_consistent_pose on the noised knees via the exported quartics.

Runtime mechanisms (all annealable from the trainer):
  * imitation hook — if sim/robot/reference_gait.json exists (schema: period_s,
    n, joint_order matching actuator order, qpos_targets [n][12], feet_swing
    [n][4]) the reward gains IMIT_W*anneal * exp(-|qpos_act - ref(phase)|^2/s^2)
    plus a feet_swing agreement bonus; phase = (t*dt/period) mod 1 per world;
  * curriculum alpha in step(): 0 = derating OFF (lim=1), 1 = servo-true
    torque-speed line; lim_eff = 1 - alpha*(1 - lim) (mass stays fixed);
  * action low-pass a_used = ACT_LP*a_prev + (1-ACT_LP)*a_new; obs's prev_action
    slot reports the USED (filtered) action;
  * privileged() critic tensor (foot contact/height/penetration proxies,
    qfrc_actuator, true root vel, loop slide positions) — critic-only, obs
    stays 50.

Per-world autoreset reruns batched forward kinematics after rewriting qpos/qvel.
Returned observations, privileged contact/force features, and the next critic
input therefore all describe the same post-reset state.
"""
from __future__ import annotations

import json
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
from gen_mesh_robot_mjcf import FOOT_R, WFREE, build_mesh_robot, loop_polycoefs  # noqa: E402

# --- new-env knobs, read ONCE (same discipline as the spec module) -------------
IMIT_W = float(os.environ.get("MESH_IMIT_W", "2.0"))
IMIT_SIGMA = float(os.environ.get("MESH_IMIT_SIGMA", "0.5"))     # rad, on the 12-joint L2 error
IMIT_FEET_W = float(os.environ.get("MESH_IMIT_FEET_W", "0.5"))   # feet_swing agreement fraction
ACT_LP = float(os.environ.get("MESH_ACT_LP", "0.6"))             # a_used = LP*prev + (1-LP)*new
REFERENCE_GAIT = HERE / "reference_gait.json"
LEGS = ("FL", "FR", "RL", "RR")
OBS_DIM = 50            # 12 qpos + 12 qvel + 4 quat + 6 root vel + 1 z + 12 prev_a + 3 cmd
PRIV_DIM = 34           # 4 contact + 4 foot_z + 4 penetration + 12 qfrc + 6 root vel + 4 slide


@wp.kernel
def _pd_ctrl(
    qpos: wp.array2d(dtype=wp.float32),
    qvel: wp.array2d(dtype=wp.float32),
    target: wp.array2d(dtype=wp.float32),
    qadr: wp.array(dtype=wp.int32),
    dadr: wp.array(dtype=wp.int32),
    kp: wp.array(dtype=wp.float32),
    wfree: wp.array(dtype=wp.float32),
    gear: wp.array(dtype=wp.float32),
    alpha: wp.array(dtype=wp.float32),
    ctrl: wp.array2d(dtype=wp.float32),
):
    """P-ONLY servo torque per substep (mesh_commanded_env lines 183-191): DRIVE
    torque derates linearly to zero at the joint no-load speed, BRAKING torque
    keeps full stall authority. Curriculum: lim_eff = 1 - alpha*(1 - lim), so
    alpha=0 disables derating entirely and alpha=1 is servo-true."""
    w, i = wp.tid()
    tau = kp[i] * (target[w, i] - qpos[w, qadr[i]])
    qd = qvel[w, dadr[i]]
    lim = wp.float32(1.0)
    if tau * qd > 0.0:
        lim = wp.clamp(1.0 - wp.abs(qd) / wfree[i], 0.0, 1.0)
    lim = 1.0 - alpha[0] * (1.0 - lim)
    ctrl[w, i] = wp.clamp(tau / gear[i], -lim, lim)


def _poly(c, x):
    """Horner eval of ascending polycoef c0..c4 (mesh_commanded_env lines 249-253)."""
    return (((c[4] * x + c[3]) * x + c[2]) * x + c[1]) * x + c[0]


def load_reference_gait(path: Path, joint_names: list[str], device) -> dict | None:
    """Imitation reference hook. Returns None if the file is absent (another
    agent produces it). joint_order may name joints or actuators; qpos_targets
    are permuted into OUR actuator order. feet_swing rows are (FL, FR, RL, RR),
    1.0 = foot should be swinging at that frame."""
    path = Path(path)
    if not path.exists():
        return None
    raw = json.loads(path.read_text())
    order = list(raw["joint_order"])
    if all(nm in order for nm in joint_names):
        perm = [order.index(nm) for nm in joint_names]
    else:  # tolerate actuator-name spelling ({L}_{yaw,swing,knee}_m)
        alt = [nm.replace("hip_yaw", "yaw_m").replace("leg_swing", "swing_m")
                 .replace("knee_blade", "knee_m") for nm in joint_names]
        perm = [order.index(nm) for nm in alt]
    n = int(raw["n"])
    q = np.asarray(raw["qpos_targets"], dtype=np.float32)[:, perm]
    fs = np.asarray(raw["feet_swing"], dtype=np.float32)
    if q.shape != (n, len(joint_names)) or fs.shape != (n, 4):
        raise ValueError(f"reference_gait.json shape mismatch: qpos {q.shape}, feet {fs.shape}, n={n}")
    return {"period": float(raw["period_s"]), "n": n,
            "qpos": torch.as_tensor(q, device=device),
            "swing": torch.as_tensor(fs, device=device)}


class MeshWarpEnv:
    """Batched (nworld) mesh-robot velocity-command env; semantics = mesh_commanded_env."""

    def __init__(self, nworld: int, seed: int = 0, device: str | None = None,
                 frame_skip: int = FRAME_SKIP, episode_length: int | None = None,
                 nconmax: int | None = None, njmax: int | None = 128,
                 gait_path: Path | str = REFERENCE_GAIT):
        wp.init()
        use_cuda = torch.cuda.is_available() if device is None else str(device).startswith("cuda")
        self.device = torch.device("cuda:0" if use_cuda else "cpu")
        self._wp_device = wp.get_device("cuda:0" if use_cuda else "cpu")
        if self._wp_device.is_cuda:
            # torch adopts warp's stream: eager torch ops and graph launches are
            # totally ordered on ONE stream — no cross-stream syncs anywhere.
            torch.cuda.set_stream(torch.cuda.ExternalStream(
                wp.get_stream(self._wp_device).cuda_stream, device=self.device))

        model_xml = build_mesh_robot()
        self.model_hash = hashlib.sha256(model_xml.encode()).hexdigest()[:16]
        m = mujoco.MjModel.from_xml_string(model_xml)   # V.6: built from spec, never a disk path
        assert abs(m.opt.timestep - TIMESTEP) < 1e-12, "model dt drifted from constants.TIMESTEP"
        self.mjm = m
        self.nworld, self._fs = int(nworld), int(frame_skip)
        self._dt = frame_skip * m.opt.timestep                   # 0.02 s control step
        self._nu = int(m.nu)
        self._episode_length = episode_length
        dev = self.device

        # --- actuated-joint addressing in actuator order (spec lines 87-111) ---
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
        kp = np.array(list(SPEC.KP) * 4)
        wfree = np.array([WFREE["hip_yaw"], WFREE["leg_swing"], WFREE["knee_blade"]] * 4)
        self._knee_q = lt([m.jnt_qposadr[j] for j in aj[2::3]])
        jid = {jname(j): j for j in range(m.njnt)}
        self._toe_q = lt([m.jnt_qposadr[jid[f"{L}_toe_hinge"]] for L in LEGS])
        self._slide_q = lt([m.jnt_qposadr[jid[f"{L}_pushrod_slide"]] for L in LEGS])
        self._feet = lt([mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_GEOM, f"{L}_foot") for L in LEGS])
        self._torso = int(mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, "torso"))
        self._q0_64 = torch.as_tensor(m.qpos0.copy(), dtype=torch.float64, device=dev)
        self._stand = ft(m.qpos0[self._qa.cpu().numpy()])
        frac = np.array([SPEC.YAW_AUTHORITY, SPEC.AUTHORITY_FRAC, SPEC.AUTHORITY_FRAC] * 4)
        self._authority = ft(frac * 0.5 * (jr[:, 1] - jr[:, 0]))
        # loop quartics — SAME polynomials the model compiles (spec lines 244-253)
        self._cs, self._cp, _, _ = loop_polycoefs()

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
        self.qfrc_actuator = wp.to_torch(self._wd.qfrc_actuator)  # (nworld, nv)
        self.qacc_warmstart = wp.to_torch(self._wd.qacc_warmstart)
        self.sim_time = wp.to_torch(self._wd.time)
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
        self._dirs = ft([[1.0, 0.0], [0.0, 1.0], [-1.0, 0.0], [0.0, -1.0]])
        self._pair_a = ft([1.0, 0.0, 0.0, 1.0])            # diagonal pair A = (FL, RR)
        self._pair_b = ft([0.0, 1.0, 1.0, 0.0])            # diagonal pair B = (FR, RL)
        self._gait = load_reference_gait(gait_path, self.joint_names, dev)

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
        if self._graph is not None:
            wp.capture_launch(self._graph)
        else:
            for _ in range(self._fs):
                self._substep()

    # ------------------------------------------------------------------ commands
    def _sample_cmd(self) -> torch.Tensor:
        """Spec lines 138-144: 4 cardinal dirs, |v| ~ U(0.3, 1)*MESH_VMAX, 15% hold."""
        n, dev = self.nworld, self.device
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
        mjwp.forward(self._wm, self._wd)
        return self.observe()

    def _reset_worlds(self, mask: torch.Tensor):
        """Branchless in-place per-world reset through the Data views (no host
        syncs — an all-False mask is a no-op). Spec lines 151-168: qpos0 +
        U(-0.03, 0.03) on ACTUATED joints only, then toe/slide from the knee via
        the model-exact quartics (the exported HARD RULE), qvel = 0, fresh
        command, all clocks zeroed."""
        n, dev = self.nworld, self.device
        mf = mask.unsqueeze(-1)
        noise = (torch.rand((n, self._nu), generator=self._gen, device=dev,
                            dtype=torch.float64) * 2.0 - 1.0) * SPEC.RESET_NOISE
        q = self._q0_64.expand(n, -1).clone()
        q[:, self._qa] += noise
        knee = q[:, self._knee_q]
        q[:, self._toe_q] = _poly(self._cp, knee)
        q[:, self._slide_q] = _poly(self._cs, knee)
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

    # ------------------------------------------------------------------ obs
    def _yaw_cs(self):
        """Yaw rotation of spec lines 120-125: R = [[c, s], [-s, c]]."""
        q = self.qpos
        w, x, y, z = q[:, 3], q[:, 4], q[:, 5], q[:, 6]
        yaw = torch.atan2(2.0 * (w * z + x * y), 1.0 - 2.0 * (y * y + z * z))
        return torch.cos(yaw), torch.sin(yaw)

    def observe(self) -> torch.Tensor:
        """The 50-obs layout of spec lines 127-136, verbatim order."""
        q, v, cmd = self.qpos, self.qvel, self._cmd
        c, s = self._yaw_cs()
        v_loc = torch.stack([c * v[:, 0] + s * v[:, 1], -s * v[:, 0] + c * v[:, 1]], dim=-1)
        c_loc = torch.stack([c * cmd[:, 0] + s * cmd[:, 1], -s * cmd[:, 0] + c * cmd[:, 1]], dim=-1)
        return torch.cat([q[:, self._qa], v[:, self._da], q[:, 3:7],
                          v_loc, v[:, 2:6], q[:, 2:3],
                          self._prev_a, c_loc, cmd[:, 2:3]], dim=-1)

    def privileged(self) -> torch.Tensor:
        """(nworld, 34) critic-only tensor: per-foot contact/height/penetration
        force proxies, qfrc_actuator at the actuated dofs, TRUE root velocity,
        and the four loop slide positions. Never enters the 50-obs policy input."""
        foot_z = self.geom_xpos[:, self._feet, 2]
        contact = (foot_z < SPEC.FOOT_CONTACT_Z).float()
        pen = (FOOT_R - foot_z).clamp(min=0.0)
        return torch.cat([contact, foot_z, pen,
                          self.qfrc_actuator[:, self._da],
                          self.qvel[:, 0:6], self.qpos[:, self._slide_q]], dim=-1)

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
        # action low-pass; the USED action is what obs reports as prev_action
        a = ACT_LP * self._prev_a + (1.0 - ACT_LP) * action.clamp(-1.0, 1.0)
        target = (self._stand + a * self._authority).clamp(self._jr_lo, self._jr_hi)
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
        align = dot / (speed * cmd_norm + 1e-6)
        active = (cmd_norm > 0.05).float()
        foot_z = self.geom_xpos[:, self._feet, 2]
        contact = foot_z < SPEC.FOOT_CONTACT_Z
        air = self._air
        first_c = contact & (air > 0.0)
        fcf = first_c.float()
        air_rwd = ((air.clamp(max=SPEC.AIRTIME_CAP) - SPEC.AIRTIME_TARGET) * fcf).sum(-1)
        new_air = torch.where(contact, torch.zeros_like(air), air + dt)
        act_rate = ((a - self._prev_a) ** 2).sum(-1)
        pose_dev = ((q[:, self._qa] - self._stand) ** 2).sum(-1)
        progress_c = progress.clamp(-cmd_norm, cmd_norm) / SPEC.MESH_VMAX
        phase = (self._t.float() * dt * SPEC.CLOCK_HZ) % 1.0
        swing_a = (phase < 0.5).float().unsqueeze(-1)
        want = self._pair_a * swing_a + self._pair_b * (1.0 - swing_a)
        cf = contact.float()
        clock_bonus = (want * (1.0 - cf) + (1.0 - want) * cf).mean(-1)
        reward = (SPEC.TRACK_W * track + SPEC.UPRIGHT_W * up + 0.1
                  + SPEC.CLOCK_W * active * clock_bonus
                  + SPEC.ALIGN_W * active * align.clamp(-1.0, 1.0)
                  + SPEC.PROGRESS_W * active * progress_c
                  + SPEC.AIRTIME_W * air_rwd * (cmd_norm / SPEC.MESH_VMAX).clamp(0.0, 1.0)
                  - SPEC.BACKWARD_W * active * (-progress).clamp(min=0.0)
                  - SPEC.POSE_W * pose_dev - SPEC.ACTRATE_W * act_rate
                  - SPEC.VELZ_W * v[:, 2] ** 2
                  - SPEC.ANGXY_W * (v[:, 3] ** 2 + v[:, 4] ** 2))
        imit = torch.zeros(n, device=dev)
        if self._gait is not None and imit_anneal > 0.0 and IMIT_W > 0.0:
            g = self._gait
            gp = (self._t.float() * dt / g["period"]) % 1.0
            x = gp * g["n"]
            i0 = x.floor().long() % g["n"]
            i1 = (i0 + 1) % g["n"]
            frac = (x - x.floor()).unsqueeze(-1)
            ref_q = g["qpos"][i0] * (1.0 - frac) + g["qpos"][i1] * frac
            err2 = ((q[:, self._qa] - ref_q) ** 2).sum(-1)
            want_ref = g["swing"][i0]
            feet_agree = (want_ref * (1.0 - cf) + (1.0 - want_ref) * cf).mean(-1)
            imit = torch.exp(-err2 / IMIT_SIGMA ** 2) + IMIT_FEET_W * feet_agree
            reward = reward + (IMIT_W * imit_anneal) * imit
        height = self.xpos[:, self._torso, 2]
        fall = (height < SPEC.FALL_Z) | (up < SPEC.MIN_UP_Z)      # spec line 231

        # --- bookkeeping, terminal snapshot, per-world autoreset ----------------
        air_pre = air.clone()
        self._air = new_air
        self._prev_a = a
        self._t = self._t + 1
        if self._episode_length is not None:
            trunc = (self._t >= self._episode_length) & ~fall
        else:
            trunc = torch.zeros(n, dtype=torch.bool, device=dev)
        done = fall | trunc
        obs_pre = self.observe()
        priv_pre = self.privileged()
        self._reset_worlds(done)                 # branchless per-world autoreset, no host sync
        # Keep post-reset qpos/qvel and derived kinematics/contact/forces coherent.
        # Unconditional execution avoids a done.any() host synchronization on CUDA.
        mjwp.forward(self._wm, self._wd)
        info = {"truncated": trunc.float(), "terminal_obs": obs_pre, "terminal_priv": priv_pre,
                "priv": self.privileged(),
                "contact": cf, "first_contact": fcf, "air_pre": air_pre,
                "track": track, "verr": torch.sqrt(verr), "align": align, "speed": speed,
                "progress": progress, "up": up, "height": height, "imit": imit}
        return self.observe(), reward, done.float(), info


class EvalTelemetry:
    """Cheap running-mean telemetry over an eval window (device scalars only):
    duty factor, mean air-time at touchdown, diag_sync = mean Pearson corr of the
    two diagonal-pair contact patterns over all (world, step) samples, plus the
    standard tracking metrics."""

    KEYS = ("track", "verr", "align", "speed", "progress", "up", "height")
    OPTIONAL_KEYS = ("xprogress", "lateral", "progress_ema", "progress_req", "duty_ema",
                     "foot_duty_ema", "motion_prior",
                     "hop_peak", "hop_peak_delta", "hop_airborne", "hop_landed",
                     "hop_stable_landing",
                     "cat_slip", "cat_orient", "cat_qvel", "cat_progress",
                     "cat_duty", "cat_foot_duty", "cat_support", "cat_body")

    def __init__(self, device):
        z = lambda: torch.zeros((), device=device)  # noqa: E731
        self._sums = {k: z() for k in self.KEYS}
        self._optional_sums = {k: z() for k in self.OPTIONAL_KEYS}
        self._optional_counts = {k: 0 for k in self.OPTIONAL_KEYS}
        self._reward, self._duty, self._n = z(), z(), 0
        self._catrate = z()          # mean CaT termination rate (0 if env reports none)
        self._td_air, self._td_cnt = z(), z()
        # per diagonal pair (a=FLxRR, b=FRxRL): Sx, Sy, Sxy, Sxx, Syy, count
        self._pa = [z() for _ in range(5)]
        self._pb = [z() for _ in range(5)]
        self._pn = 0

    def add(self, reward, info):
        cf = info["contact"]
        n = cf.shape[0]
        self._n += 1
        self._reward += reward.mean()
        self._duty += cf.mean()
        cat = info.get("cat_done")           # walker env reports it; mesh env does not
        if cat is not None:
            self._catrate += cat.mean()
        for k in self.KEYS:
            self._sums[k] += info[k].mean()
        for k in self.OPTIONAL_KEYS:
            if k in info:
                self._optional_sums[k] += info[k].mean()
                self._optional_counts[k] += 1
        self._td_air += (info["air_pre"] * info["first_contact"]).sum()
        self._td_cnt += info["first_contact"].sum()
        for sums, (i, j) in ((self._pa, (0, 3)), (self._pb, (1, 2))):
            x, y = cf[:, i], cf[:, j]
            sums[0] += x.sum(); sums[1] += y.sum(); sums[2] += (x * y).sum()
            sums[3] += (x * x).sum(); sums[4] += (y * y).sum()
        self._pn += n

    @staticmethod
    def _corr(sums, n):
        sx, sy, sxy, sxx, syy = (s.item() for s in sums)
        mx, my = sx / n, sy / n
        var = (sxx / n - mx * mx) * (syy / n - my * my)
        return (sxy / n - mx * my) / (var ** 0.5) if var > 1e-12 else 0.0

    def result(self) -> dict:
        n = max(self._n, 1)
        out = {k: (self._sums[k] / n).item() for k in self.KEYS}
        for k in self.OPTIONAL_KEYS:
            cnt = self._optional_counts[k]
            if cnt:
                out[k] = (self._optional_sums[k] / cnt).item()
        out["reward"] = (self._reward / n).item()
        out["duty"] = (self._duty / n).item()
        out["catrate"] = (self._catrate / n).item()
        cnt = self._td_cnt.item()
        out["air"] = (self._td_air.item() / cnt) if cnt > 0 else 0.0
        pn = max(self._pn, 1)
        out["diagsync"] = 0.5 * (self._corr(self._pa, pn) + self._corr(self._pb, pn))
        return out


def main():
    """Throughput probe: random actions, prints one RESULT line."""
    import argparse
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--nworld", type=int, default=8)
    ap.add_argument("--steps", type=int, default=100)
    ap.add_argument("--warmup", type=int, default=10)
    ap.add_argument("--device", default=None)
    args = ap.parse_args()
    env = MeshWarpEnv(args.nworld, seed=0, device=args.device, episode_length=800)
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
    print(f"RESULT bench=mesh_warp_env nworld={args.nworld} steps={args.steps} "
          f"device={env.device} env_steps_per_s={args.nworld * args.steps / wall:.1f} "
          f"wall_s={wall:.3f}", flush=True)


if __name__ == "__main__":
    main()
