# SPDX-License-Identifier: MIT
"""Compatibility entry point for command-conditioned MuJoCo-Warp locomotion."""

from __future__ import annotations

import argparse
import math

import torch

from locomotion_spec import *  # noqa: F401,F403 - compatibility re-export


def sample_command(generator: torch.Generator, nworld: int = 1,
                   device: str | torch.device = "cpu") -> torch.Tensor:
    """Sample batched velocity commands using the canonical Torch RNG."""
    device = torch.device(device)
    if CMD_TRAIN_MODE == "forward":
        direction = torch.tensor((1.0, 0.0), device=device).repeat(nworld, 1)
    elif CMD_TRAIN_MODE == "backward":
        direction = torch.tensor((-1.0, 0.0), device=device).repeat(nworld, 1)
    elif CMD_TRAIN_MODE == "cardinal":
        choices = torch.tensor(((1.0, 0.0), (0.0, 1.0), (-1.0, 0.0), (0.0, -1.0)),
                               device=device)
        direction = choices[torch.randint(4, (nworld,), generator=generator, device=device)]
    else:
        angle = (torch.rand(nworld, generator=generator, device=device) * 2.0 - 1.0) * math.pi
        direction = torch.stack((angle.cos(), angle.sin()), dim=-1)
    speed = (0.35 + 0.65 * torch.rand(nworld, generator=generator, device=device)) * VMAX
    yaw = torch.zeros(nworld, device=device)
    if CMD_TRAIN_MODE not in ("forward", "backward", "cardinal"):
        yaw = (torch.rand(nworld, generator=generator, device=device) * 2.0 - 1.0) * YAW_MAX
    hold = torch.rand(nworld, generator=generator, device=device) < 0.15
    command = torch.cat((direction * speed[:, None], yaw[:, None]), dim=-1)
    return torch.where(hold[:, None], torch.zeros_like(command), command)


def _build():
    """Return the supported command environment class for compatibility."""
    from walker_warp_env import WalkerWarpEnv

    class CommandedWarpEnv(WalkerWarpEnv):
        def __init__(self, nworld: int = 1, **kwargs):
            super().__init__(nworld=nworld, **kwargs)

        @property
        def observation_size(self):
            return self.obs_dim

        @property
        def action_size(self):
            return self.act_dim

        @property
        def backend(self):
            return "mujoco_warp"

    return CommandedWarpEnv


def prove() -> None:
    def track(velocity, command):
        delta = torch.as_tensor(velocity) - torch.as_tensor(command)
        return float(torch.exp(-(delta * delta).sum() / TRACK_SIGMA))

    command = (VMAX, 0.0)
    matched = track(command, command)
    wrong = track((-VMAX, 0.0), command)
    stopped = track((0.0, 0.0), command)
    hold = track((0.0, 0.0), (0.0, 0.0))
    hold_moving = track((VMAX, 0.0), (0.0, 0.0))
    assert matched > 0.9 and matched > 5 * wrong and matched > stopped
    assert hold > 0.9 and hold > 5 * hold_moving
    print("PASS: command tracking and hold-command ordering")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--prove", action="store_true")
    args = parser.parse_args()
    if args.prove:
        prove()
    else:
        print("Use train_mesh_warp.py --geometry walker for command-conditioned training.")
