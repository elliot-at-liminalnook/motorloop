# SPDX-License-Identifier: MIT
"""Torch-facing batched combat environment over fused MuJoCo-Warp kernels."""

from __future__ import annotations

import hashlib
from typing import Callable

import mujoco
import numpy as np
import torch
import warp as wp
import mujoco_warp as mjwp

from warplayer.fused import FightLayer
from predictive_control import (InteractionTrajectoryTarget,
                                model_morphology_tokens)


class CombatWarpEnv:
    """Two-policy combat with zero-copy Torch views and per-world autoreset.

    The 12 leg motors per robot and optional pneumatic striker actions are kept
    distinct in the compiled model. Physics, contact damage, observations, and
    both reward streams execute on the selected Warp device.
    """

    gait_loaded = False
    action_semantics = "direct_actuator:A12_servos+2_pneumatic:v1"
    reward_semantics = "combat:symmetric_damage+locomotion:v1"

    def __init__(self, nworld: int, seed: int = 0, device: str | None = None,
                 episode_length: int | None = 800, lidar: bool = False, cfg=None):
        wp.init()
        use_cuda = torch.cuda.is_available() if device is None else str(device).startswith("cuda")
        self.device = torch.device("cuda:0" if use_cuda else "cpu")
        self._wp_device = wp.get_device("cuda:0" if use_cuda else "cpu")
        if self._wp_device.is_cuda:
            torch.cuda.set_stream(torch.cuda.ExternalStream(
                wp.get_stream(self._wp_device).cuda_stream, device=self.device))
        with wp.ScopedDevice(self._wp_device):
            self.layer = FightLayer(nworld=nworld, mode="fused", lidar=lidar, seed=seed,
                                    cfg=cfg)

        self.nworld = int(nworld)
        self.act_dim = int(self.layer.idx.nuA)
        self.obs_dim = int(self.layer.obs_dim)
        self.priv_dim = 32
        self._dt = self.layer.frame_skip * float(self.layer.mjm.opt.timestep)
        self._episode_length = episode_length
        self.model_hash = hashlib.sha256(
            bytes(self.layer.mjm.names) + self.layer.mjm.body_mass.tobytes()
            + self.layer.mjm.actuator_gear.tobytes()).hexdigest()[:16]
        self._gen = torch.Generator(device=self.device).manual_seed(seed)
        self._opponent: Callable[[torch.Tensor], torch.Tensor] | None = None
        self.mjm = self.layer.mjm

        # All tensors alias Warp/MuJoCo-Warp storage.
        self.qpos = wp.to_torch(self.layer.d.qpos)
        self.qvel = wp.to_torch(self.layer.d.qvel)
        self.qacc_warmstart = wp.to_torch(self.layer.d.qacc_warmstart)
        self.sim_time = wp.to_torch(self.layer.d.time)
        self.xpos = wp.to_torch(self.layer.d.xpos)
        self.geom_xpos = wp.to_torch(self.layer.d.geom_xpos)
        self.xquat = wp.to_torch(self.layer.d.xquat)
        self.cvel = wp.to_torch(self.layer.d.cvel)
        self.qfrc_actuator = wp.to_torch(self.layer.d.qfrc_actuator)
        self.obs_a = wp.to_torch(self.layer.obs)
        self.obs_b = wp.to_torch(self.layer.obs_b)
        self.reward_a = wp.to_torch(self.layer.reward)
        self.reward_b = wp.to_torch(self.layer.reward_b)
        self.done_a = wp.to_torch(self.layer.done)
        self.done_b = wp.to_torch(self.layer.done_b)
        self.constraint_rows = wp.to_torch(self.layer.d.nefc)
        self.actions_a = wp.to_torch(self.layer.act)
        self.actions_b = wp.to_torch(self.layer.act_b)
        self._prev_dist = wp.to_torch(self.layer.prev_dist)
        self._prev_dealt = wp.to_torch(self.layer.prev_dealt)
        self._vel_ema = wp.to_torch(self.layer.vel_ema)
        self._combat_time = wp.to_torch(self.layer.t)
        self._prev_dist_b = wp.to_torch(self.layer.prev_dist_b)
        self._prev_dealt_b = wp.to_torch(self.layer.prev_dealt_b)
        self._vel_ema_b = wp.to_torch(self.layer.vel_ema_b)
        self._combat_time_b = wp.to_torch(self.layer.t_b)
        self._t = torch.zeros(self.nworld, dtype=torch.long, device=self.device)
        self._qpos0 = torch.as_tensor(self.layer._qpos0, device=self.device)
        self._qvel0 = torch.as_tensor(self.layer._qvel0, device=self.device)
        self._wm, self._wd = self.layer.m, self.layer.d
        (self.morphology_tokens, self.morphology_token_types,
         self.morphology_token_mask) = model_morphology_tokens(
            self.mjm, device=self.device, batch=self.nworld)
        self.morphology_token_count = int(self.morphology_tokens.shape[1])
        foot_geoms = []
        for leg in ("FL", "FR", "RL", "RR"):
            name = f"A_{leg}_foot"
            geom = mujoco.mj_name2id(
                self.mjm, mujoco.mjtObj.mjOBJ_GEOM, name)
            if geom < 0:
                raise ValueError(f"combat morphology is missing required foot geom {name}")
            foot_geoms.append(geom)
        self._prediction_feet = torch.as_tensor(
            foot_geoms, dtype=torch.long, device=self.device)
        actuator_joint_ids = self.mjm.actuator_trnid[
            np.asarray(self.layer.idx.actA, dtype=np.int64), 0]
        self._prediction_actuator_dofs = torch.as_tensor(
            self.mjm.jnt_dofadr[actuator_joint_ids], dtype=torch.long, device=self.device)
        self.reset()
        if self._wp_device.is_cuda:
            with wp.ScopedDevice(self._wp_device):
                self.layer.capture()

    def set_opponent(self, policy: Callable[[torch.Tensor], torch.Tensor] | None) -> None:
        self._opponent = policy
        self._reset_opponent_state(None)

    def _reset_opponent_state(self, mask: torch.Tensor | None) -> None:
        """A recurrent opponent's hidden state must not cross episodes."""
        reset = getattr(self._opponent, "reset", None)
        if callable(reset):
            reset(mask)

    def observe(self) -> torch.Tensor:
        return self.obs_a

    def observe_opponent(self) -> torch.Tensor:
        return self.obs_b

    def privileged(self) -> torch.Tensor:
        i = self.layer.idx
        rel = self.xpos[:, i.Bt] - self.xpos[:, i.At]
        damage = torch.stack([
            wp.to_torch(self.layer.dealt_leg), wp.to_torch(self.layer.dealt_rod),
            wp.to_torch(self.layer.taken_leg), wp.to_torch(self.layer.taken_rod),
            wp.to_torch(self.layer.pen_peak),
        ], dim=-1)
        return torch.cat((self.qvel[:, i.ArD:i.ArD + 6],
                          self.qvel[:, i.BrD:i.BrD + 6], rel, damage,
                          self.qpos[:, torch.as_tensor(i.Aqa, device=self.device)]), dim=-1)

    def trajectory_state(self) -> torch.Tensor:
        """Standard world-frame snapshot for self-supervised future prediction."""
        i = self.layer.idx
        quat = self.xquat[:, i.At]
        up = wp.to_torch(self.layer.d.xmat)[:, i.At, 2, 2]
        feet = self.geom_xpos[:, self._prediction_feet]
        # The fused combat path does not yet expose per-foot contacts, so use the
        # same conservative height proxy as locomotion rather than inventing a
        # zero target that would teach the predictor that combat has no support.
        contact = (feet[..., 2] < 0.05).float()
        effort = torch.zeros((self.nworld, 14), device=self.device)
        count = min(self.act_dim, 14)
        effort[:, :count] = self.qfrc_actuator[
            :, self._prediction_actuator_dofs[:count]]
        interaction = torch.zeros((self.nworld, 4), device=self.device)
        return torch.cat((
            self.xpos[:, i.At], quat, self.cvel[:, i.At, 3:6],
            self.cvel[:, i.At, 0:3], feet.reshape(self.nworld, 12), contact,
            interaction, effort, up[:, None], self.xpos[:, i.At, 2:3]), dim=-1)

    def interaction_target(self, horizon: int) -> InteractionTrajectoryTarget:
        """Approach and optional selected-leg contact intent in a local world frame."""
        target = InteractionTrajectoryTarget.empty(
            horizon, self.nworld, device=self.device, dtype=self.xpos.dtype,
            dt=self._dt)
        i = self.layer.idx
        root = self.xpos[:, i.At]
        opponent = self.xpos[:, i.Bt]
        quat = self.xquat[:, i.At]
        yaw = torch.atan2(2.0 * (quat[:, 0] * quat[:, 3]
                                + quat[:, 1] * quat[:, 2]),
                          1.0 - 2.0 * (quat[:, 2].square() + quat[:, 3].square()))
        c, s = torch.cos(yaw), torch.sin(yaw)

        def local(value: torch.Tensor) -> torch.Tensor:
            return torch.stack((c * value[:, 0] + s * value[:, 1],
                                -s * value[:, 0] + c * value[:, 1],
                                value[:, 2]), dim=-1)

        relative = local(opponent - root)
        distance = torch.linalg.vector_norm(relative[:, :2], dim=-1)
        direction_xy = relative[:, :2] / distance[:, None].clamp_min(1.0e-5)
        travel = (distance - 0.34).clamp_min(0.0)
        duration = max(int(horizon) * self._dt, self._dt)
        approach_speed = torch.minimum(travel / duration,
                                       torch.full_like(travel, 0.35))
        time_fraction = (torch.arange(
            1, int(horizon) + 1, device=self.device, dtype=self.xpos.dtype)
                         / float(max(int(horizon), 1)))[:, None]
        target.root_delta[..., :2] = (time_fraction[..., None] * travel[None, :, None]
                                            * direction_xy[None, :, :])
        target.root_delta_mask[..., :2] = 1.0
        target.velocity[..., :2] = (approach_speed[:, None]
                                     * direction_xy).unsqueeze(0)
        target.velocity_mask[..., :2] = (travel > 0.0)[None, :, None].to(self.xpos.dtype)

        # Leg-conditioned environments expose a runtime-selected attacking leg.
        # The target asks that leg to reach and contact a broad opponent region
        # sometime in the horizon.  It does not prescribe a strike phase or
        # joint trajectory, and two of the other three legs are sufficient
        # support rather than demanding one fixed stance pattern.
        if hasattr(self, "_attack_leg") and hasattr(self, "_attack_active"):
            active = self._attack_active.to(self.xpos.dtype)
            selected = torch.nn.functional.one_hot(
                self._attack_leg, num_classes=4).to(self.xpos.dtype)
            selected_active = selected * active[:, None]
            target.effector_position.copy_(relative[:, None, :].expand(-1, 4, -1))
            target.effector_mask.copy_(selected_active)
            target.effector_radius.fill_(0.18)
            foot_relative = self.geom_xpos[:, self._prediction_feet] - root[:, None, :]
            current_feet = torch.stack((
                c[:, None] * foot_relative[..., 0] + s[:, None] * foot_relative[..., 1],
                -s[:, None] * foot_relative[..., 0] + c[:, None] * foot_relative[..., 1],
                foot_relative[..., 2]), dim=-1)
            direction = target.effector_position - current_feet
            target.effector_direction.copy_(torch.nn.functional.normalize(
                direction, dim=-1, eps=1.0e-6))
            target.effector_speed_min.copy_(0.25 * selected_active)
            target.interaction_event.copy_(selected_active)
            # During an active attack all four interaction channels matter:
            # selected leg should hit, non-selected legs should not.
            target.interaction_event_mask.copy_(active[:, None].expand(-1, 4))
            target.support_leg_mask.copy_(1.0 - selected_active)
            target.min_support.copy_(torch.where(
                active > 0.5, torch.full_like(active, 2.0),
                torch.full_like(active, 3.0)))
        return target

    def _refresh_outputs(self) -> None:
        with wp.ScopedDevice(self._wp_device):
            self.layer.refresh_outputs()

    def _reset_worlds(self, mask: torch.Tensor) -> None:
        mask = mask.bool()
        m2 = mask[:, None]
        self.qpos.copy_(torch.where(m2, self._qpos0, self.qpos))
        self.qvel.copy_(torch.where(m2, self._qvel0, self.qvel))
        self.qacc_warmstart.masked_fill_(m2, 0.0)
        self.sim_time.masked_fill_(mask, 0.0)
        self._t.masked_fill_(mask, 0)
        for tensor in (self.actions_a, self.actions_b, self._prev_dist, self._prev_dealt,
                       self._vel_ema, self._combat_time, self._prev_dist_b,
                       self._prev_dealt_b, self._vel_ema_b, self._combat_time_b):
            tensor.masked_fill_(mask.reshape((-1,) + (1,) * (tensor.ndim - 1)), 0.0)

    def reset(self, seed: int | None = None) -> torch.Tensor:
        if seed is not None:
            self._gen.manual_seed(seed)
        self._reset_worlds(torch.ones(self.nworld, dtype=torch.bool, device=self.device))
        self._reset_opponent_state(None)
        with wp.ScopedDevice(self._wp_device):
            mjwp.forward(self._wm, self._wd)
            self.layer.refresh_outputs()
        return self.observe()

    def step(self, action: torch.Tensor, alpha: float = 1.0,
             imit_anneal: float = 0.0):
        del alpha, imit_anneal
        self.actions_a.copy_(action.to(self.device).clamp(-1.0, 1.0))
        if self._opponent is None:
            self.actions_b.zero_()
        else:
            with torch.no_grad():
                self.actions_b.copy_(self._opponent(self.obs_b).clamp(-1.0, 1.0))
        with wp.ScopedDevice(self._wp_device):
            self.layer.step_fused()

        reward = self.reward_a.clone()
        terminal_obs = self.obs_a.clone()
        terminal_priv = self.privileged().clone()
        terminal_b = self.obs_b.clone()
        self._t.add_(1)
        terminated = (self.done_a > 0) | (self.done_b > 0)
        truncated = (self._t >= self._episode_length) & ~terminated \
            if self._episode_length is not None else torch.zeros_like(terminated)
        done = terminated | truncated
        if bool(done.any()):
            # Autoreset replaces these worlds' physical state below; the
            # opponent's GRU memory must not leak across that boundary.
            self._reset_opponent_state(done)

        i = self.layer.idx
        rel = self.xpos[:, i.Bt] - self.xpos[:, i.At]
        dist = torch.linalg.vector_norm(rel[:, :2], dim=-1)
        speed = torch.linalg.vector_norm(self.qvel[:, i.ArD:i.ArD + 2], dim=-1)
        up = self.layer.d.xmat
        up_t = wp.to_torch(up)[:, i.At, 2, 2]
        height = self.xpos[:, i.At, 2]
        zero4 = torch.zeros((self.nworld, 4), device=self.device)
        info = {
            "truncated": truncated.float(), "terminal_obs": terminal_obs,
            "terminal_priv": terminal_priv, "terminal_opponent_obs": terminal_b,
            "opponent_reward": self.reward_b.clone(), "priv": terminal_priv,
            "contact": zero4, "first_contact": zero4, "air_pre": zero4,
            "track": torch.exp(-dist), "verr": dist, "align": torch.zeros_like(dist),
            "speed": speed, "progress": -dist, "up": up_t, "height": height,
            "fallrate": terminated.to(torch.float32),
            "constraint_rows": self.constraint_rows.clone(),
            "constraint_capacity": self.layer.njmax,
            "reward_components": {"combat_base": reward},
            "actuator_diagnostics": {
                "command_saturated": (self.actions_a.abs() > 0.95).to(torch.float32),
                "action_magnitude": self.actions_a.abs(),
            },
            "simulation_diagnostics": {
                "constraint_rows": self.constraint_rows.to(torch.float32),
                "constraint_capacity": torch.full_like(
                    self.constraint_rows, float(self.layer.njmax), dtype=torch.float32),
                "state_nonfinite": ((~torch.isfinite(self.qpos)).sum(dim=-1)
                                    + (~torch.isfinite(self.qvel)).sum(dim=-1)).to(torch.float32),
            },
        }
        self._reset_worlds(done)
        with wp.ScopedDevice(self._wp_device):
            mjwp.forward(self._wm, self._wd)
            self.layer.refresh_outputs()
        info["priv"] = self.privileged()
        return self.observe(), reward, done.float(), info


def main() -> None:
    import argparse
    import time

    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--nworld", type=int, default=32)
    ap.add_argument("--steps", type=int, default=100)
    ap.add_argument("--warmup", type=int, default=10)
    ap.add_argument("--device", default=None)
    args = ap.parse_args()
    env = CombatWarpEnv(args.nworld, device=args.device)
    action = torch.zeros((args.nworld, env.act_dim), device=env.device)
    for _ in range(args.warmup):
        env.step(action)
    if env.device.type == "cuda":
        torch.cuda.synchronize()
    start = time.time()
    for _ in range(args.steps):
        obs, reward, _, _ = env.step(action)
    if env.device.type == "cuda":
        torch.cuda.synchronize()
    wall = time.time() - start
    assert torch.isfinite(obs).all() and torch.isfinite(reward).all()
    print(f"RESULT bench=combat_warp_env nworld={args.nworld} steps={args.steps} "
          f"device={env.device} env_steps_per_s={args.nworld * args.steps / wall:.1f}")


if __name__ == "__main__":
    main()
