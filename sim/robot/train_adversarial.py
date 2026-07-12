# SPDX-License-Identifier: MIT
"""Fused MuJoCo-Warp combat training and compatibility API."""

from __future__ import annotations

import os
import sys
from pathlib import Path

import torch
import warp as wp

from combat_warp_env import CombatWarpEnv
from constants import LOCO_OBS
from gen_robot_mjcf import load_spec
from train_mesh_warp import load_policy as _load_torch_policy
from warplayer.obsreward import RewardConfig
from warp_train_cli import run

HERE = Path(__file__).resolve().parent
SPEC = load_spec(HERE / "robot.toml")
OUT = Path(os.environ.get("CODESIGN_OUT", str(HERE.parent / "build/gpu/out")))
OUT.mkdir(parents=True, exist_ok=True)
DAMAGE_REF = 0.05
STRIKE_KINETIC = 0.1

BENCH_KEYS = [
    "sparc", "dealt", "taken", "clean", "trade", "fire", "closing", "fleeing",
    "dist", "alive", "sparc_close", "sparc_med", "sparc_far", "win_rate",
    "survival_rate", "safe_rate", "ac_peak_z", "ac_airborne", "ac_peak_pen",
    "ac_idle", "ac_dmg_early", "ac_upright_dmg", "ac_grounded_dmg",
    "ac_uprightness", "bh_disp", "bh_path", "bh_closed", "bh_lateral",
    "bh_approach", "bh_gate_open", "bh_tip_speed",
]


def METRIC(**values):
    print("METRIC " + " ".join(f"{key}={value}" for key, value in values.items()),
          flush=True)


class AdversarialEnv(CombatWarpEnv):
    """Compatibility name over the supported fused combat environment."""

    def __init__(self, nworld: int = 1, frame_skip: int = 5,
                 episode_length: int | None = 800, lidar: bool = False,
                 seed: int = 0, device: str | None = None, **kwargs):
        del frame_skip
        aliases = {
            "approach_weight": "approach_w", "upright_weight": "upright_w",
            "alive_bonus": "alive", "energy_penalty": "energy_w",
            "airborne_penalty": "airborne_w", "height_weight": "height_w",
            "move_weight": "move_w", "close_bonus": "close_bonus_w",
            "face_weight": "face_w", "flee_penalty": "flee_w",
            "taken_weight": "taken_w", "clean_weight": "clean_w",
            "trade_weight": "trade_w", "disengage_weight": "dis_w",
            "damage_bonus": "damage_bonus_w", "loco_speed": "loco_speed",
            "loco_track_w": "loco_track_w", "early_hit_penalty": "early_hit_penalty",
            "min_hit_step": "min_hit_step", "stationary_damage_penalty": "stationary_pen",
            "oscillation_penalty": "oscillation_pen", "move_eps": "move_eps",
            "penetration_penalty": "penalty_w", "penetration_tol": "penalty_tol",
            "combat_scale": "combat_scale", "shaping": "shaping",
        }
        cfg = RewardConfig.from_constants(SPEC)
        for source, target in aliases.items():
            if source in kwargs:
                setattr(cfg, target, float(kwargs[source]))
        super().__init__(nworld=nworld, seed=seed, device=device,
                         episode_length=episode_length, lidar=lidar, cfg=cfg)

    @property
    def observation_size(self):
        return self.obs_dim

    @property
    def action_size(self):
        return self.act_dim

    @property
    def backend(self):
        return "mujoco_warp"


def load_policy(path, observation_size=None, action_size=None, device=None):
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    checkpoint = torch.load(path, map_location=device, weights_only=False)
    observation_size = observation_size or checkpoint["obs_norm"]["mean"].numel()
    action_size = action_size or checkpoint["actor"]["log_std"].numel()
    return _load_torch_policy(path, observation_size, action_size, device)


def load_opponent(path, obs=None, act=None, device=None):
    return load_policy(path, obs, act, device)


def warm_start(path, obs_dim, act_dim=None):
    del obs_dim, act_dim
    return torch.load(path, map_location="cpu", weights_only=False)


def behavior_keep_ok(values, min_closed=-1e30, min_approach=-1e30,
                     min_disp=-1e30, min_far_sparc=-1e30):
    return bool(values.get("bh_closed", 0.0) >= min_closed
                and values.get("bh_approach", 0.0) >= min_approach
                and values.get("bh_disp", 0.0) >= min_disp
                and values.get("sparc_far", 0.0) >= min_far_sparc)


def build_benchmark(env: CombatWarpEnv, n_epis: int, steps: int,
                    seed: int = 20240601, deterministic: bool = True):
    """Return a fixed-scenario Torch benchmark callable over a checkpoint path."""
    del deterministic
    dealt_view = wp.to_torch(env.layer.dealt_leg)
    taken_view = wp.to_torch(env.layer.taken_leg)

    def benchmark(checkpoint_path):
        policy = load_policy(checkpoint_path, env.obs_dim, env.act_dim, env.device)
        totals = torch.zeros(len(BENCH_KEYS), device=env.device)
        for episode in range(n_epis):
            obs = env.reset(seed + episode)
            start = env.xpos[:, env.layer.idx.At, :2].clone()
            dealt = torch.zeros(env.nworld, device=env.device)
            taken = torch.zeros_like(dealt)
            reward = torch.zeros_like(dealt)
            done = torch.zeros_like(dealt)
            for _ in range(steps):
                obs, current, done, _ = env.step(policy(obs))
                reward += current
                dealt += dealt_view
                taken += taken_view
            displacement = torch.linalg.vector_norm(
                env.xpos[:, env.layer.idx.At, :2] - start, dim=-1).mean()
            totals[0] += reward.mean()
            totals[1] += dealt.mean()
            totals[2] += taken.mean()
            totals[9] += (1.0 - done).mean()
            totals[24] += displacement
        return totals / max(n_epis, 1)

    return benchmark


if __name__ == "__main__":
    run("combat", sys.argv[1:], default_tag="combat_warp")
