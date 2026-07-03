# SPDX-License-Identifier: MIT
"""warplayer.fused — M3: the two-robot fight scene stepped via mujoco_warp AS A
LIBRARY with our lidar/obs/reward kernels appended to the same launch sequence
(secret-sauce §10c(iv)).

Two modes, the exact comparison the >=2x kill criterion is defined over:

  BASELINE (`mode="baseline"`) — the wrapper way, as train_adversarial does it:
    mujoco_warp steps the physics (engine rangefinder sensors ENABLED — the env's
    lidar is 144 <rangefinder> sensors computed inside every physics substep,
    exactly like the MJX pipeline it mirrors), then per CONTROL step the state
    is pulled device->host (qpos, qvel, xpos, xquat, xmat, geom_xpos,
    sensordata, contact pool), obs/reward are computed in numpy (the same
    reference code the tests use), and obs/reward are pushed host->device for
    the learner. That device->host->device round-trip is the seam being killed.

  FUSED (`mode="fused"`) — ours: the SAME physics (engine sensor stage disabled
    via mjDSBL_SENSOR; sensors never affect dynamics, asserted by the parity
    test) + our lidar/obs/reward kernels launched right after the step, writing
    (nworld, obs_dim) / (nworld,) buffers consumed zero-copy via dlpack /
    wp.array.numpy(). No host branch, no sync, fixed launch dims: on CUDA the
    whole control step (FRAME_SKIP physics steps + 4 kernels) captures into ONE
    graph (`use_graph=True`); on CPU the identical call sequence runs eagerly —
    capture is a flag, not a rewrite.

Scene: gen_robot_mjcf.build_match(spec, spec, sep=1.2, striker=True,
striker_b=True[, lidar=True]) — the same builder train_adversarial and
bench_warp_vs_mjx use; model built from spec via from_xml_string (V.6
guardrail: no from_xml_path). dt and control cadence come from sim/robot/
constants (TIMESTEP, FRAME_SKIP) and are asserted against the compiled model.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import warp as wp

HERE = Path(__file__).resolve().parent
if str(HERE.parent) not in sys.path:
    sys.path.insert(0, str(HERE.parent))          # sim/robot: gen_robot_mjcf, constants

from .lidar import Lidar, lidar_kernel  # noqa: E402,F401
from .obsreward import (  # noqa: E402
    FightIndices,
    RewardConfig,
    damage_kernel,
    damage_zero_kernel,
    fight_indices,
    normalize_scan,
    obs_kernel,
    obs_reference,
    reward_kernel,
    reward_reference,
)

QVEL_NOISE = 0.05   # same decorrelation as bench_warp_vs_mjx.py:42,68-70


def build_fight_model(lidar: bool = False, lidar_n_rays: int = 128,
                      lidar_n_vertical: int = 16, lidar_max_range: float = 2.0,
                      disable_sensors: bool = False):
    """(mjm, spec) for the two-robot fight scene, built from the spec."""
    import mujoco
    from gen_robot_mjcf import build_match, load_spec

    spec = load_spec(HERE.parent / "robot.toml")
    xml = build_match(spec, spec, sep=1.2, striker=True, striker_b=True,
                      lidar=lidar, lidar_n_rays=lidar_n_rays,
                      lidar_n_vertical=lidar_n_vertical, lidar_max_range=lidar_max_range)
    mjm = mujoco.MjModel.from_xml_string(xml)
    if disable_sensors:
        mjm.opt.disableflags |= mujoco.mjtDisableBit.mjDSBL_SENSOR
    return mjm, spec


class FightLayer:
    """mujoco_warp-as-library + the bespoke 10%: one object per benchmark mode."""

    def __init__(self, nworld: int, mode: str = "fused", lidar: bool = True,
                 lidar_max_range: float = 2.0, seed: int = 0,
                 design: np.ndarray | None = None, cfg: RewardConfig | None = None,
                 mjm=None, spec=None, nconmax: int | None = None, njmax: int | None = None):
        import mujoco
        import mujoco_warp as mjwp
        from constants import FRAME_SKIP, LOCO_OBS, TIMESTEP

        assert mode in ("baseline", "fused")
        self.mode = mode
        self.nworld = int(nworld)
        self.has_lidar = bool(lidar)
        self._mjwp = mjwp
        if mjm is None:
            # baseline keeps engine sensors (the wrapper computes lidar that way);
            # fused disables the engine sensor stage and computes lidar itself.
            mjm, spec = build_fight_model(lidar=lidar, lidar_max_range=lidar_max_range,
                                          disable_sensors=(mode == "fused"))
        self.mjm, self.spec = mjm, spec
        assert abs(mjm.opt.timestep - TIMESTEP) < 1e-12, "model dt != constants.TIMESTEP"
        self.frame_skip = int(FRAME_SKIP)
        self.idx = fight_indices(mjm)
        self.cfg = cfg if cfg is not None else RewardConfig.from_constants(spec)
        self._params = self.cfg.to_struct()

        mjd = mujoco.MjData(mjm)
        mujoco.mj_resetData(mjm, mjd)
        mujoco.mj_forward(mjm, mjd)
        self.m = mjwp.put_model(mjm)
        self.d = mjwp.put_data(mjm, mjd, nworld=self.nworld, nconmax=nconmax, njmax=njmax)
        rng = np.random.default_rng(seed)
        self._qvel0 = rng.uniform(-QVEL_NOISE, QVEL_NOISE,
                                  size=(self.nworld, mjm.nv)).astype(np.float32)
        self._qpos0 = np.tile(mjd.qpos.astype(np.float32), (self.nworld, 1))
        self.d.qvel.assign(self._qvel0)
        mjwp.forward(self.m, self.d)     # consistent derived state for step 0 obs

        # ---- layer-owned device buffers (the fused-graph outputs) ----
        nray = 0
        self.lidar = None
        if lidar:
            if mode == "fused":
                self.lidar = Lidar(mjm, self.nworld, max_range=lidar_max_range)
                nray = self.lidar.nray
            else:
                nray = int(mjm.nsensor)
        self.nray = nray
        self.max_range = float(lidar_max_range)
        design = np.zeros((self.nworld, 3), dtype=np.float32) if design is None \
            else np.asarray(design, dtype=np.float32).reshape(self.nworld, 3)
        self.design = wp.array(design, dtype=wp.float32)
        self.obs_dim = LOCO_OBS + (nray if lidar else 6)
        self.obs = wp.zeros((self.nworld, self.obs_dim), dtype=wp.float32)
        self.reward = wp.zeros(self.nworld, dtype=wp.float32)
        self.done = wp.zeros(self.nworld, dtype=wp.float32)
        self.act = wp.zeros((self.nworld, self.idx.nuA), dtype=wp.float32)
        self.prev_dist = wp.zeros(self.nworld, dtype=wp.float32)
        self.prev_dealt = wp.zeros(self.nworld, dtype=wp.float32)
        self.vel_ema = wp.zeros(self.nworld, dtype=wp.vec2)
        self.t = wp.zeros(self.nworld, dtype=wp.float32)
        self.dealt_leg = wp.zeros(self.nworld, dtype=wp.float32)
        self.dealt_rod = wp.zeros(self.nworld, dtype=wp.float32)
        self.taken_leg = wp.zeros(self.nworld, dtype=wp.float32)
        self.taken_rod = wp.zeros(self.nworld, dtype=wp.float32)
        self.pen_peak = wp.zeros(self.nworld, dtype=wp.float32)
        # device-side index tables for the kernels
        i = self.idx
        self._wAqa = wp.array(i.Aqa, dtype=wp.int32)
        self._wAda = wp.array(i.Ada, dtype=wp.int32)
        self._wAstrike = wp.array(i.Astrike, dtype=wp.int32)
        self._wArod = wp.array(i.Arod_gids, dtype=wp.int32)
        self._wsd = wp.array(i.strike_dofs, dtype=wp.int32)
        self._wsdb = wp.array(i.strike_dofs_b, dtype=wp.int32)
        self._wsl = wp.array(i.strike_local, dtype=wp.int32)
        self._wmasks = [wp.array(m, dtype=wp.int32) for m in
                        (i.mask_Aleg, i.mask_Bleg, i.mask_Arod, i.mask_Brod,
                         i.mask_Abody, i.mask_Bbody)]
        # host carries for the baseline path (numpy mirrors of the device carries)
        self._h_prev_dist = np.zeros(self.nworld)
        self._h_prev_dealt = np.zeros(self.nworld)
        self._h_vel_ema = np.zeros((self.nworld, 2))
        self._h_t = np.zeros(self.nworld)
        self._graph = None
        # baseline scan slice: rangefinder sensordata addresses (env _lidar_adr, :399)
        self._sensor_adr = mjm.sensor_adr[:mjm.nsensor].copy() if lidar and mode == "baseline" else None

    # ------------------------------------------------------------------ fused
    def _launch_outputs(self, include_reward: bool = True):
        """Append the bespoke kernels to the step's launch sequence (no host sync).
        include_reward=False computes obs only (reset-time refresh: the reward
        kernel advances the carried state prev_dist/prev_dealt/vel_ema/t)."""
        d, i = self.d, self.idx
        wp.launch(damage_zero_kernel, dim=self.nworld,
                  inputs=[self.dealt_leg, self.dealt_rod, self.taken_leg,
                          self.taken_rod, self.pen_peak])
        wp.launch(damage_kernel, dim=d.naconmax,
                  inputs=[d.nacon, d.contact.dist, d.contact.geom, d.contact.worldid,
                          *self._wmasks],
                  outputs=[self.dealt_leg, self.dealt_rod, self.taken_leg,
                           self.taken_rod, self.pen_peak])
        if self.lidar is not None:
            self.lidar.launch(self.m, d)
        scan = self.lidar.scan if self.lidar is not None else self.obs  # dummy when no lidar
        wp.launch(obs_kernel, dim=self.nworld,
                  inputs=[d.qpos, d.qvel, d.xpos, d.xquat,
                          self._wAqa, self._wAda, self.design, scan,
                          i.At, i.Bt, i.ArD, i.BrD,
                          1 if self.lidar is not None else 0],
                  outputs=[self.obs])
        if not include_reward:
            return
        wp.launch(reward_kernel, dim=self.nworld,
                  inputs=[d.qvel, d.xpos, d.xmat, d.geom_xpos,
                          self.act, self.dealt_leg, self.dealt_rod,
                          self.taken_leg, self.taken_rod, self.pen_peak,
                          self._wAstrike, self._wArod, self._wsd, self._wsdb, self._wsl,
                          i.At, i.Bt, i.ArD, i.BrD, i.n_hinge, self._params,
                          self.prev_dist, self.prev_dealt, self.vel_ema, self.t],
                  outputs=[self.reward, self.done])

    def _control_step_fused(self):
        self._physics_block()
        self._launch_outputs()

    def _physics_block(self):
        for _ in range(self.frame_skip):
            self._mjwp.step(self.m, self.d)

    def capture(self):
        """CUDA: capture one control step into a graph.

        fused    -> ONE graph = FRAME_SKIP steps + our kernels (the §10c(iv)
                    artifact: OUR obs/reward inside THEIR graph).
        baseline -> graph over the physics block ONLY — the wrapper way also
                    captures/jits the step; what it CANNOT capture is its host
                    obs/reward, which stays outside. Capturing both keeps the
                    >=2x comparison honest: the measured delta is the seam,
                    not uncaptured physics launch overhead."""
        assert wp.get_device().is_cuda, "graph capture needs a CUDA device"
        target = self._control_step_fused if self.mode == "fused" else self._physics_block
        target()                                  # load modules before capture
        wp.synchronize()
        with wp.ScopedCapture() as cap:
            target()
        self._graph = cap.graph

    def step_fused(self):
        if self._graph is not None:
            wp.capture_launch(self._graph)
        else:
            self._control_step_fused()

    # --------------------------------------------------------------- baseline
    def _host_pull(self) -> dict:
        """The wrapper-way device->host round-trip payload."""
        d = self.d
        h = {
            "qpos": d.qpos.numpy().copy(), "qvel": d.qvel.numpy().copy(),
            "xpos": d.xpos.numpy().copy(), "xquat": d.xquat.numpy().copy(),
            "xmat": d.xmat.numpy().copy(), "geom_xpos": d.geom_xpos.numpy().copy(),
            "nacon": int(d.nacon.numpy()[0]),
            "con_dist": d.contact.dist.numpy().copy(),
            "con_geom": d.contact.geom.numpy().copy(),
            "con_worldid": d.contact.worldid.numpy().copy(),
            "act": self.act.numpy().copy(),
        }
        if self._sensor_adr is not None:
            h["sensordata"] = d.sensordata.numpy().copy()
        return h

    def step_baseline(self):
        if self._graph is not None:               # captured physics block (CUDA)
            wp.capture_launch(self._graph)
        else:
            self._physics_block()
        wp.synchronize()
        h = self._host_pull()                      # device -> host
        scan = None
        if self._sensor_adr is not None:
            scan = normalize_scan(h["sensordata"][:, self._sensor_adr], self.max_range)
        obs = obs_reference(h, self.idx, self.design.numpy(), scan)
        rew, done, self._h_prev_dist, self._h_prev_dealt, self._h_vel_ema, self._h_t = \
            reward_reference(h, self.idx, self.cfg, self._h_prev_dist,
                             self._h_prev_dealt, self._h_vel_ema, self._h_t)
        self.obs.assign(obs.astype(np.float32))    # host -> device (learner consumes device buffers)
        self.reward.assign(rew.astype(np.float32))
        self.done.assign(done.astype(np.float32))

    # ------------------------------------------------------------------ common
    def step(self):
        if self.mode == "fused":
            self.step_fused()
        else:
            self.step_baseline()

    def refresh_outputs(self):
        """Fill obs for the CURRENT state without advancing physics (used at
        reset; identical kernels/reference as the step path). Reward is NOT
        recomputed: its kernel advances the carried state (t, prev_*)."""
        if self.mode == "fused":
            self._launch_outputs(include_reward=False)
        else:
            h = self._host_pull()
            scan = None
            if self._sensor_adr is not None:
                scan = normalize_scan(h["sensordata"][:, self._sensor_adr], self.max_range)
            self.obs.assign(obs_reference(h, self.idx, self.design.numpy(), scan).astype(np.float32))

    def set_actions(self, actions: np.ndarray):
        """Write the policy's A-robot actions (nworld, nuA), clipped to [-1,1],
        into d.ctrl (direct-torque action mode, train_adversarial.py:846-848).
        Host->device by design in the demo: the policy lives on host. In
        production the policy writes these buffers via dlpack on-device."""
        a = np.clip(np.asarray(actions, dtype=np.float32), -1.0, 1.0)
        self.act.assign(a)
        ctrl = np.zeros((self.nworld, self.mjm.nu), dtype=np.float32)
        ctrl[:, self.idx.actA] = a
        self.d.ctrl.assign(ctrl)

    def reset(self, seed: int | None = None):
        if seed is not None:
            rng = np.random.default_rng(seed)
            self._qvel0 = rng.uniform(-QVEL_NOISE, QVEL_NOISE,
                                      size=(self.nworld, self.mjm.nv)).astype(np.float32)
        self.d.qpos.assign(self._qpos0)
        self.d.qvel.assign(self._qvel0)
        self.d.ctrl.zero_()
        self.act.zero_()
        for b in (self.prev_dist, self.prev_dealt, self.vel_ema, self.t,
                  self.reward, self.done):
            b.zero_()
        self._h_prev_dist = np.zeros(self.nworld)
        self._h_prev_dealt = np.zeros(self.nworld)
        self._h_vel_ema = np.zeros((self.nworld, 2))
        self._h_t = np.zeros(self.nworld)
        self._mjwp.forward(self.m, self.d)
        self.refresh_outputs()

    # zero-copy consumption (M4): on CPU wp.array.numpy() aliases the buffer;
    # on CUDA hand `self.obs` / `self.reward` to torch.from_dlpack / jax dlpack.
    def obs_numpy(self) -> np.ndarray:
        return self.obs.numpy()

    def reward_numpy(self) -> np.ndarray:
        return self.reward.numpy()
