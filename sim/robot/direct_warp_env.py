# SPDX-License-Identifier: MIT
"""Generic batched MuJoCo-Warp environment for generated single-robot MJCF."""

from __future__ import annotations

import hashlib

import mujoco
import mujoco_warp as mjwp
import torch
import warp as wp


class DirectWarpEnv:
    """Direct normalized actuator control for arbitrary generated MJCF.

    This is the replacement for the old functional design environment. It keeps
    model construction and physics in MuJoCo/MuJoCo-Warp and exposes zero-copy
    Torch tensors to the shared PPO implementation.
    """

    gait_loaded = False
    action_semantics = "direct_normalized_actuator:v1"
    reward_semantics = "upright+forward-energy:v1"

    def __init__(self, xml: str, nworld: int = 1, seed: int = 0,
                 device: str | None = None, frame_skip: int = 5,
                 episode_length: int | None = 200, design=None,
                 model_transform=None, nconmax=None, njmax=128):
        wp.init()
        use_cuda = torch.cuda.is_available() if device is None else str(device).startswith("cuda")
        self.device = torch.device("cuda:0" if use_cuda else "cpu")
        self._wp_device = wp.get_device("cuda:0" if use_cuda else "cpu")
        if self._wp_device.is_cuda:
            torch.cuda.set_stream(torch.cuda.ExternalStream(
                wp.get_stream(self._wp_device).cuda_stream, device=self.device))
        model = mujoco.MjModel.from_xml_string(xml)
        if model_transform is not None:
            model_transform(model)
        self.model_hash = hashlib.sha256(
            xml.encode() + model.body_mass.tobytes() + model.body_inertia.tobytes()
            + model.dof_damping.tobytes() + model.jnt_stiffness.tobytes()).hexdigest()[:16]
        self.mjm = model
        self.nworld, self._fs, self._nu = int(nworld), int(frame_skip), int(model.nu)
        self._dt = self._fs * float(model.opt.timestep)
        self._episode_length = episode_length
        self._design = None if design is None else torch.as_tensor(
            design, dtype=torch.float32, device=self.device)

        joints = [int(model.actuator_trnid[a, 0]) for a in range(model.nu)]
        self._qa = torch.as_tensor([model.jnt_qposadr[j] for j in joints],
                                   dtype=torch.long, device=self.device)
        self._da = torch.as_tensor([model.jnt_dofadr[j] for j in joints],
                                   dtype=torch.long, device=self.device)
        q0 = model.qpos0.copy()
        stand_abd, stand_flex, stand_knee = 0.0, -0.4, -1.1
        for joint in range(model.njnt):
            name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_JOINT, joint) or ""
            address = int(model.jnt_qposadr[joint])
            if name.endswith("_abd"):
                q0[address] = stand_abd
            elif name.endswith("_flex"):
                q0[address] = stand_flex
            elif name.endswith("_knee"):
                q0[address] = stand_knee
        self._q0 = torch.as_tensor(q0, dtype=torch.float32, device=self.device)
        self._gen = torch.Generator(device=self.device).manual_seed(seed)

        data = mujoco.MjData(model)
        data.qpos[:] = q0
        mujoco.mj_forward(model, data)
        with wp.ScopedDevice(self._wp_device):
            self._wm = mjwp.put_model(model)
            self._wd = mjwp.put_data(model, data, nworld=self.nworld,
                                     nconmax=nconmax, njmax=njmax)
        self.qpos = wp.to_torch(self._wd.qpos)
        self.qvel = wp.to_torch(self._wd.qvel)
        self.ctrl = wp.to_torch(self._wd.ctrl)
        self.xpos = wp.to_torch(self._wd.xpos)
        self.qfrc_actuator = wp.to_torch(self._wd.qfrc_actuator)
        self.qacc_warmstart = wp.to_torch(self._wd.qacc_warmstart)
        self.sim_time = wp.to_torch(self._wd.time)
        self._t = torch.zeros(self.nworld, dtype=torch.long, device=self.device)
        self._graph = None
        if self._wp_device.is_cuda:
            with wp.ScopedDevice(self._wp_device):
                self._physics_block()
                wp.synchronize_device(self._wp_device)
                with wp.ScopedCapture() as capture:
                    self._physics_block()
                self._graph = capture.graph
        self.reset()

    @property
    def act_dim(self):
        return self._nu

    @property
    def obs_dim(self):
        return 2 * self._nu + 11 + (0 if self._design is None else len(self._design))

    @property
    def priv_dim(self):
        return 6 + self._nu

    @property
    def observation_size(self):
        return self.obs_dim

    @property
    def action_size(self):
        return self.act_dim

    @property
    def backend(self):
        return "mujoco_warp"

    def _physics_block(self):
        for _ in range(self._fs):
            mjwp.step(self._wm, self._wd)

    def observe(self):
        parts = (self.qpos[:, self._qa], self.qvel[:, self._da], self.qpos[:, 3:7],
                 self.qvel[:, :6], self.qpos[:, 2:3])
        if self._design is not None:
            parts += (self._design.expand(self.nworld, -1),)
        return torch.cat(parts, dim=-1)

    def privileged(self):
        return torch.cat((self.qvel[:, :6], self.qfrc_actuator[:, self._da]), dim=-1)

    def _reset_worlds(self, done):
        mask = done.bool()
        noise = (torch.rand((self.nworld, self._nu), generator=self._gen,
                            device=self.device) * 0.1 - 0.05)
        reset_qpos = self._q0.expand(self.nworld, -1).clone()
        reset_qpos[:, self._qa] += noise
        self.qpos.copy_(torch.where(mask[:, None], reset_qpos, self.qpos))
        self.qvel.masked_fill_(mask[:, None], 0.0)
        self.ctrl.masked_fill_(mask[:, None], 0.0)
        self.qacc_warmstart.masked_fill_(mask[:, None], 0.0)
        self.sim_time.masked_fill_(mask, 0.0)
        self._t.masked_fill_(mask, 0)

    def reset(self):
        self._reset_worlds(torch.ones(self.nworld, dtype=torch.bool, device=self.device))
        with wp.ScopedDevice(self._wp_device):
            mjwp.forward(self._wm, self._wd)
        return self.observe()

    def step(self, action, alpha=1.0, imit_anneal=0.0):
        del alpha, imit_anneal
        clipped = action.to(self.device).clamp(-1.0, 1.0)
        self.ctrl.copy_(clipped)
        with wp.ScopedDevice(self._wp_device):
            if self._graph is None:
                self._physics_block()
            else:
                wp.capture_launch(self._graph)
        up = 1.0 - 2.0 * (self.qpos[:, 4] ** 2 + self.qpos[:, 5] ** 2)
        reward = 1.0 + up + self.qvel[:, 0] - 0.001 * (clipped ** 2).sum(-1)
        self._t.add_(1)
        terminated = self.qpos[:, 2] < 0.10
        truncated = (self._t >= self._episode_length) & ~terminated \
            if self._episode_length is not None else torch.zeros_like(terminated)
        done = terminated | truncated
        terminal_obs, terminal_priv = self.observe(), self.privileged()
        zero4 = torch.zeros((self.nworld, 4), device=self.device)
        speed = torch.linalg.vector_norm(self.qvel[:, :2], dim=-1)
        info = {
            "truncated": truncated.float(), "terminal_obs": terminal_obs,
            "terminal_priv": terminal_priv, "priv": terminal_priv,
            "contact": zero4, "first_contact": zero4, "air_pre": zero4,
            "track": speed, "verr": torch.zeros_like(speed), "align": speed,
            "speed": speed, "progress": self.qvel[:, 0], "up": up,
            "height": self.qpos[:, 2],
        }
        self._reset_worlds(done)
        with wp.ScopedDevice(self._wp_device):
            mjwp.forward(self._wm, self._wd)
        info["priv"] = self.privileged()
        return self.observe(), reward, done.float(), info
