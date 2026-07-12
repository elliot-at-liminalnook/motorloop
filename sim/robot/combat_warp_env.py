# SPDX-License-Identifier: MIT
"""Torch-facing batched combat environment over fused MuJoCo-Warp kernels."""

from __future__ import annotations

import hashlib
from typing import Callable

import torch
import warp as wp
import mujoco_warp as mjwp

from warplayer.fused import FightLayer


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

        # All tensors alias Warp/MuJoCo-Warp storage.
        self.qpos = wp.to_torch(self.layer.d.qpos)
        self.qvel = wp.to_torch(self.layer.d.qvel)
        self.qacc_warmstart = wp.to_torch(self.layer.d.qacc_warmstart)
        self.sim_time = wp.to_torch(self.layer.d.time)
        self.xpos = wp.to_torch(self.layer.d.xpos)
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
        self.reset()
        if self._wp_device.is_cuda:
            with wp.ScopedDevice(self._wp_device):
                self.layer.capture()

    def set_opponent(self, policy: Callable[[torch.Tensor], torch.Tensor] | None) -> None:
        self._opponent = policy

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
            "constraint_rows": self.constraint_rows.clone(),
            "constraint_capacity": self.layer.njmax,
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
