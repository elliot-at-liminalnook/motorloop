# SPDX-License-Identifier: MIT
"""Leg-conditioned attack/kick curriculum over the fused combat environment.

The low-level policy receives an explicit command

    [attack_active, FL, FR, RL, RR, sin(kick_phase), cos(kick_phase)]

after the ordinary combat observation.  A controller (learned or manual) can
switch the commanded leg without changing the actuator action layout.  During
training the command is randomized, including mid-episode switches, so the
policy must lift/strike/recover with that leg while the other three remain the
support polygon.

This is deliberately a separate checkpoint geometry.  Existing combat policies
keep their 44-value observation contract and cannot be silently loaded as a
leg-conditioned policy merely because the 14 actuator actions still match.
"""

from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Sequence

import mujoco
import torch
import torch.nn.functional as F
import warp as wp
import mujoco_warp as mjwp

from combat_warp_env import CombatWarpEnv
from predictive_control import TRAJECTORY_RAW_INTERACTION
from warplayer.obsreward import RewardConfig


LEG_NAMES = ("FL", "FR", "RL", "RR")
LEG_TO_INDEX = {name: index for index, name in enumerate(LEG_NAMES)}
ATTACK_COMMAND_DIM = 7


@dataclass(frozen=True)
class LegAttackConfig:
    """Task-level curriculum and reward constants.

    The base fused reward contributes only upright/alive/energy terms.  All hit
    credit below is resolved per foot, so a non-commanded leg cannot collect the
    commanded-leg reward.
    """

    command_hold_steps: int = 80
    kick_period_steps: int = 40
    attack_probability: float = 0.85
    target_radius_min: float = 0.38
    target_radius_max: float = 0.62
    hit_scale_m: float = 0.020
    max_extension_speed_m_s: float = 1.5
    selected_hit_weight: float = 12.0
    wrong_leg_hit_weight: float = 10.0
    strike_speed_weight: float = 1.5
    recovery_speed_weight: float = 0.75
    support_weight: float = 1.5
    idle_support_weight: float = 0.75
    idle_action_weight: float = 0.02

    def __post_init__(self):
        if self.command_hold_steps <= 0 or self.kick_period_steps <= 1:
            raise ValueError("command_hold_steps must be positive and kick_period_steps > 1")
        if not 0.0 <= self.attack_probability <= 1.0:
            raise ValueError("attack_probability must be in [0, 1]")
        if not 0.0 < self.target_radius_min <= self.target_radius_max:
            raise ValueError("target radius bounds must be positive and ordered")


def reduce_leg_contacts(
    contact_geom: torch.Tensor,
    contact_dist: torch.Tensor,
    contact_world: torch.Tensor,
    nacon: torch.Tensor,
    geom_to_leg: torch.Tensor,
    target_mask: torch.Tensor,
    floor_geom: int,
    nworld: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Return `(opponent_hit_penetration, floor_support)` per world and leg.

    `geom_to_leg[g]` is -1 for non-foot geoms and 0..3 for A's four feet.
    The implementation uses fixed-size tensor reductions, so it remains on the
    learner device and does not synchronize on the dynamic contact count.
    """

    if contact_geom.ndim != 2 or contact_geom.shape[1] != 2:
        raise ValueError("contact_geom must have shape (pool, 2)")
    device = contact_dist.device
    pool = contact_dist.shape[0]
    slots = torch.arange(pool, device=device)
    valid_slot = slots < nacon.reshape(-1)[0].to(device=device, dtype=torch.long)
    g0 = contact_geom[:, 0].to(dtype=torch.long)
    g1 = contact_geom[:, 1].to(dtype=torch.long)
    valid_geom = (g0 >= 0) & (g1 >= 0)
    safe0 = g0.clamp(0, geom_to_leg.numel() - 1)
    safe1 = g1.clamp(0, geom_to_leg.numel() - 1)
    leg0 = geom_to_leg[safe0]
    leg1 = geom_to_leg[safe1]
    target0 = target_mask[safe0]
    target1 = target_mask[safe1]
    hit0 = (leg0 >= 0) & target1
    hit1 = (leg1 >= 0) & target0
    hit_leg = torch.where(hit0, leg0, torch.where(hit1, leg1, torch.zeros_like(leg0)))
    penetration = (-contact_dist).clamp_min(0.0)
    hit_valid = valid_slot & valid_geom & (hit0 | hit1) & (penetration > 0.0)

    world = contact_world.to(dtype=torch.long).clamp(0, max(int(nworld) - 1, 0))
    hit_flat = torch.zeros(int(nworld) * len(LEG_NAMES), device=device,
                            dtype=contact_dist.dtype)
    hit_index = world * len(LEG_NAMES) + hit_leg.clamp(0, len(LEG_NAMES) - 1)
    hit_flat.scatter_add_(0, hit_index, torch.where(
        hit_valid, penetration, torch.zeros_like(penetration)))

    floor0 = g0 == int(floor_geom)
    floor1 = g1 == int(floor_geom)
    support0 = (leg0 >= 0) & floor1
    support1 = (leg1 >= 0) & floor0
    support_leg = torch.where(
        support0, leg0, torch.where(support1, leg1, torch.zeros_like(leg0)))
    support_valid = valid_slot & valid_geom & (support0 | support1) & (contact_dist <= 0.0)
    support_flat = torch.zeros_like(hit_flat)
    support_index = world * len(LEG_NAMES) + support_leg.clamp(0, len(LEG_NAMES) - 1)
    support_flat.scatter_add_(0, support_index, support_valid.to(contact_dist.dtype))
    return (hit_flat.reshape(int(nworld), len(LEG_NAMES)),
            support_flat.reshape(int(nworld), len(LEG_NAMES)).clamp_max(1.0))


def leg_attack_reward(
    hit_by_leg: torch.Tensor,
    support_by_leg: torch.Tensor,
    extension_delta_m: torch.Tensor,
    selected_leg: torch.Tensor,
    attack_active: torch.Tensor,
    phase: torch.Tensor,
    action: torch.Tensor,
    control_dt: float,
    cfg: LegAttackConfig,
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    """Compute the command-conditioned task reward and diagnostic components."""

    row = torch.arange(hit_by_leg.shape[0], device=hit_by_leg.device)
    leg = selected_leg.to(dtype=torch.long)
    selected_hit = hit_by_leg[row, leg]
    wrong_hit = (hit_by_leg.sum(dim=-1) - selected_hit).clamp_min(0.0)
    selected_support = support_by_leg[row, leg]
    support_other = ((support_by_leg.sum(dim=-1) - selected_support) / 3.0).clamp(0.0, 1.0)
    all_support = support_by_leg.mean(dim=-1)

    extension_speed = extension_delta_m[row, leg] / float(control_dt)
    # Phase remains an optional high-level timing cue and a diagnostic, but it
    # does not decide when a physically useful motion receives credit.  A hit,
    # extension, or recovery has the same meaning whenever the controller finds
    # it effective.
    del phase
    strike_speed = (extension_speed.clamp_min(0.0) /
                    cfg.max_extension_speed_m_s).clamp(0.0, 1.0)
    recovery_speed = ((-extension_speed).clamp_min(0.0) /
                      cfg.max_extension_speed_m_s).clamp(0.0, 1.0)
    selected_hit_score = (selected_hit / cfg.hit_scale_m).clamp(0.0, 1.0)
    wrong_hit_score = (wrong_hit / cfg.hit_scale_m).clamp(0.0, 1.0)
    active = attack_active.to(hit_by_leg.dtype).clamp(0.0, 1.0)
    hinge_effort = action[:, :12].abs().mean(dim=-1)

    attacking = (
        cfg.selected_hit_weight * selected_hit_score
        - cfg.wrong_leg_hit_weight * wrong_hit_score
        + cfg.strike_speed_weight * strike_speed
        + cfg.recovery_speed_weight * recovery_speed
        + cfg.support_weight * support_other
    )
    idle = cfg.idle_support_weight * all_support - cfg.idle_action_weight * hinge_effort
    reward = active * attacking + (1.0 - active) * idle
    metrics = {
        "attack_selected_hit": selected_hit_score,
        "attack_wrong_hit": wrong_hit_score,
        "attack_support": support_other,
        "attack_selected_ground": selected_support,
        "attack_kick_speed": strike_speed,
        "attack_recovery_speed": recovery_speed,
        "attack_task_reward": reward,
    }
    return reward, metrics


class LegAttackWarpEnv(CombatWarpEnv):
    """Passive-target kick rung with a runtime-switchable attacking leg."""

    gait_loaded = False
    action_semantics = "direct_actuator:A12_servos+2_pneumatic:v1"
    observation_semantics = "combat44+attack(active1+leg4+phase_sincos2):v1"
    reward_semantics = "leg_attack:selected_foot_hit+three_leg_support:phase_free:v2"

    def __init__(
        self,
        nworld: int,
        seed: int = 0,
        device: str | None = None,
        episode_length: int | None = 800,
        lidar: bool = False,
        cfg: RewardConfig | None = None,
        task_cfg: LegAttackConfig | None = None,
    ):
        self._leg_attack_ready = False
        self.task_cfg = task_cfg or LegAttackConfig()
        base_cfg = cfg or RewardConfig.from_constants(
            shaping=0.0,
            combat_scale=0.0,
            approach_w=0.0,
            upright_w=0.7,
            alive=0.1,
            energy_w=0.02,
            loco_speed=0.0,
            loco_track_w=0.0,
            fire_shaping=0.0,
        )
        super().__init__(nworld=nworld, seed=seed, device=device,
                         episode_length=episode_length, lidar=lidar, cfg=base_cfg)

        self._base_obs_dim = int(self.layer.obs_dim)
        self.obs_dim = self._base_obs_dim + ATTACK_COMMAND_DIM
        self._attack_leg = torch.zeros(self.nworld, dtype=torch.long, device=self.device)
        self._attack_active = torch.ones(self.nworld, dtype=torch.float32, device=self.device)
        self._attack_override_leg = torch.full(
            (self.nworld,), -1, dtype=torch.long, device=self.device)
        self._attack_override_active = torch.full(
            (self.nworld,), -1, dtype=torch.long, device=self.device)
        self._attack_timer = torch.zeros(self.nworld, dtype=torch.long, device=self.device)
        self._attack_phase_step = torch.zeros(
            self.nworld, dtype=torch.long, device=self.device)

        self._geom_xpos = wp.to_torch(self.layer.d.geom_xpos)
        self._contact_geom = wp.to_torch(self.layer.d.contact.geom)
        self._contact_dist = wp.to_torch(self.layer.d.contact.dist)
        self._contact_world = wp.to_torch(self.layer.d.contact.worldid)
        self._nacon = wp.to_torch(self.layer.d.nacon)

        mjm = self.layer.mjm
        gid = lambda name: mujoco.mj_name2id(mjm, mujoco.mjtObj.mjOBJ_GEOM, name)
        jid = lambda name: mujoco.mj_name2id(mjm, mujoco.mjtObj.mjOBJ_JOINT, name)
        foot_ids = [gid(f"A_{leg}_foot") for leg in LEG_NAMES]
        if min(foot_ids) < 0:
            raise ValueError(f"leg-attack task requires all four A foot geoms: {foot_ids}")
        self._foot_gids = torch.as_tensor(foot_ids, dtype=torch.long, device=self.device)
        geom_to_leg = torch.full((mjm.ngeom,), -1, dtype=torch.long, device=self.device)
        geom_to_leg[self._foot_gids] = torch.arange(len(LEG_NAMES), device=self.device)
        self._geom_to_leg = geom_to_leg
        self._target_mask = torch.as_tensor(
            self.layer.idx.mask_Bbody.astype(bool), device=self.device)
        self._floor_gid = gid("floor")
        self._A_root_qadr = int(mjm.jnt_qposadr[jid("A_root")])
        self._B_root_qadr = int(mjm.jnt_qposadr[jid("B_root")])
        self._base_b_height = self._qpos0[:, self._B_root_qadr + 2].clone()
        self._prev_extension = torch.zeros(
            (self.nworld, len(LEG_NAMES)), device=self.device)
        self._prev_support = torch.zeros_like(self._prev_extension)
        self._leg_attack_ready = True
        self.reset(seed=seed)

    # ---------------------------------------------------------- command API
    @staticmethod
    def leg_index(leg: str | int) -> int:
        if isinstance(leg, str):
            key = leg.upper()
            if key not in LEG_TO_INDEX:
                raise ValueError(f"unknown leg {leg!r}; choose one of {LEG_NAMES}")
            return LEG_TO_INDEX[key]
        value = int(leg)
        if value < 0 or value >= len(LEG_NAMES):
            raise ValueError(f"leg index must be in [0, {len(LEG_NAMES) - 1}]")
        return value

    def _leg_values(self, leg: str | int | Sequence[int] | torch.Tensor) -> torch.Tensor:
        if isinstance(leg, (str, int)):
            return torch.full((self.nworld,), self.leg_index(leg), dtype=torch.long,
                              device=self.device)
        values = torch.as_tensor(leg, dtype=torch.long, device=self.device)
        if values.ndim == 0:
            values = values.expand(self.nworld)
        if values.shape != (self.nworld,):
            raise ValueError(f"per-world leg command must have shape ({self.nworld},)")
        if bool(((values < 0) | (values >= len(LEG_NAMES))).any()):
            raise ValueError("per-world leg command contains an invalid leg index")
        return values

    def set_attack_leg(self, leg: str | int | Sequence[int] | torch.Tensor | None) -> None:
        """Lock the attacking leg, or pass `None` to resume curriculum sampling."""
        if leg is None:
            self._attack_override_leg.fill_(-1)
            return
        values = self._leg_values(leg)
        self._attack_override_leg.copy_(values)
        self._attack_leg.copy_(values)
        self._attack_phase_step.zero_()

    def set_attack_enabled(self, active: bool | Sequence[bool] | torch.Tensor | None) -> None:
        """Lock attack on/off, or pass `None` to resume curriculum sampling."""
        if active is None:
            self._attack_override_active.fill_(-1)
            return
        values = torch.as_tensor(active, dtype=torch.long, device=self.device)
        if values.ndim == 0:
            values = values.expand(self.nworld)
        if values.shape != (self.nworld,):
            raise ValueError(f"per-world attack enable must have shape ({self.nworld},)")
        values = values.clamp(0, 1)
        self._attack_override_active.copy_(values)
        self._attack_active.copy_(values.to(torch.float32))

    def set_attack_command(self, leg: str | int | Sequence[int] | torch.Tensor,
                           active: bool | Sequence[bool] | torch.Tensor = True) -> None:
        """Controller hook: switch the leg command without resetting the robot."""
        self.set_attack_leg(leg)
        self.set_attack_enabled(active)

    def clear_attack_command(self) -> None:
        """Return leg and attack-enable channels to the randomized curriculum."""
        self.set_attack_leg(None)
        self.set_attack_enabled(None)

    # -------------------------------------------------------- observations
    def _phase(self) -> torch.Tensor:
        return torch.remainder(
            self._attack_phase_step, self.task_cfg.kick_period_steps).to(torch.float32) \
            / float(self.task_cfg.kick_period_steps)

    def _command_features(self) -> torch.Tensor:
        if not self._leg_attack_ready:
            return torch.empty((self.nworld, 0), device=self.device)
        phase = self._phase()
        angle = 2.0 * math.pi * phase
        one_hot = F.one_hot(self._attack_leg, num_classes=len(LEG_NAMES)).to(torch.float32)
        return torch.cat((self._attack_active[:, None], one_hot,
                          torch.sin(angle)[:, None], torch.cos(angle)[:, None]), dim=-1)

    def observe(self) -> torch.Tensor:
        base = super().observe()
        if not self._leg_attack_ready:
            return base
        return torch.cat((base, self._command_features()), dim=-1)

    def observe_opponent(self) -> torch.Tensor:
        base = super().observe_opponent()
        if not self._leg_attack_ready:
            return base
        return torch.cat((base, self._command_features()), dim=-1)

    # ------------------------------------------------------------- resets
    def _sample_commands(self, mask: torch.Tensor) -> None:
        mask = mask.bool()
        random_leg = torch.randint(
            0, len(LEG_NAMES), (self.nworld,), generator=self._gen, device=self.device)
        random_active = (torch.rand(
            self.nworld, generator=self._gen, device=self.device)
            < self.task_cfg.attack_probability).to(torch.float32)
        next_leg = torch.where(
            self._attack_override_leg >= 0,
            self._attack_override_leg,
            random_leg,
        )
        next_active = torch.where(
            self._attack_override_active >= 0,
            self._attack_override_active.to(torch.float32),
            random_active,
        )
        self._attack_leg.copy_(torch.where(mask, next_leg, self._attack_leg))
        self._attack_active.copy_(torch.where(mask, next_active, self._attack_active))
        self._attack_timer.copy_(torch.where(
            mask,
            torch.full_like(self._attack_timer, self.task_cfg.command_hold_steps),
            self._attack_timer,
        ))
        self._attack_phase_step.copy_(torch.where(
            mask, torch.zeros_like(self._attack_phase_step), self._attack_phase_step))

    def _randomize_target_layout(self, mask: torch.Tensor) -> None:
        mask = mask.bool()
        radius = self.task_cfg.target_radius_min + (
            self.task_cfg.target_radius_max - self.task_cfg.target_radius_min
        ) * torch.rand(self.nworld, generator=self._gen, device=self.device)
        bearing = (2.0 * math.pi) * torch.rand(
            self.nworld, generator=self._gen, device=self.device) - math.pi
        aq, bq = self._A_root_qadr, self._B_root_qadr
        ax = self._qpos0[:, aq + 0]
        ay = self._qpos0[:, aq + 1]
        bx = ax + radius * torch.cos(bearing)
        by = ay + radius * torch.sin(bearing)
        facing = bearing + math.pi
        values = (
            bx,
            by,
            self._base_b_height,
            torch.cos(0.5 * facing),
            torch.zeros_like(facing),
            torch.zeros_like(facing),
            torch.sin(0.5 * facing),
        )
        for offset, value in enumerate(values):
            self._qpos0[:, bq + offset].copy_(torch.where(
                mask, value.to(self._qpos0.dtype), self._qpos0[:, bq + offset]))

    def _current_extension(self) -> torch.Tensor:
        i = self.layer.idx
        torso = self.xpos[:, i.At]
        target = self.xpos[:, i.Bt]
        direction = target[:, :2] - torso[:, :2]
        direction = direction / (torch.linalg.vector_norm(
            direction, dim=-1, keepdim=True) + 1.0e-6)
        foot_rel = self._geom_xpos[:, self._foot_gids, :2] - torso[:, None, :2]
        return (foot_rel * direction[:, None, :]).sum(dim=-1)

    def reset(self, seed: int | None = None) -> torch.Tensor:
        if not getattr(self, "_leg_attack_ready", False):
            return super().reset(seed)
        if seed is not None:
            self._gen.manual_seed(seed)
        all_worlds = torch.ones(self.nworld, dtype=torch.bool, device=self.device)
        self._sample_commands(all_worlds)
        self._randomize_target_layout(all_worlds)
        super().reset(seed=None)
        self._prev_extension.copy_(self._current_extension())
        self._prev_support.zero_()
        return self.observe()

    # --------------------------------------------------------------- step
    def _contact_metrics(self) -> tuple[torch.Tensor, torch.Tensor]:
        return reduce_leg_contacts(
            self._contact_geom,
            self._contact_dist,
            self._contact_world,
            self._nacon,
            self._geom_to_leg,
            self._target_mask,
            self._floor_gid,
            self.nworld,
        )

    def trajectory_state(self) -> torch.Tensor:
        """Add real per-leg opponent contact to the predictive state target."""
        state = super().trajectory_state()
        hit_by_leg, _ = self._contact_metrics()
        state[:, TRAJECTORY_RAW_INTERACTION] = (hit_by_leg > 0.0).to(state.dtype)
        return state

    def step(self, action: torch.Tensor, alpha: float = 1.0,
             imit_anneal: float = 0.0):
        del alpha, imit_anneal
        if action.shape != (self.nworld, self.act_dim):
            raise ValueError(
                f"leg-attack action must have shape ({self.nworld}, {self.act_dim})")
        command_before = self._command_features().clone()
        leg_before = self._attack_leg.clone()
        active_before = self._attack_active.clone()
        phase_before = self._phase().clone()
        self.actions_a.copy_(action.to(self.device).clamp(-1.0, 1.0))
        if self._opponent is None:
            self.actions_b.zero_()
        else:
            with torch.no_grad():
                self.actions_b.copy_(self._opponent(
                    self.observe_opponent()).clamp(-1.0, 1.0))
        with wp.ScopedDevice(self._wp_device):
            self.layer.step_fused()

        base_reward = self.reward_a.clone()
        terminal_obs = torch.cat((self.obs_a.clone(), command_before), dim=-1)
        terminal_priv = self.privileged().clone()
        terminal_b = torch.cat((self.obs_b.clone(), command_before), dim=-1)
        current_extension = self._current_extension()
        extension_delta = current_extension - self._prev_extension
        hit_by_leg, support_by_leg = self._contact_metrics()
        task_reward, attack_metrics = leg_attack_reward(
            hit_by_leg,
            support_by_leg,
            extension_delta,
            leg_before,
            active_before,
            phase_before,
            self.actions_a,
            self._dt,
            self.task_cfg,
        )
        reward = base_reward + task_reward

        self._t.add_(1)
        terminated = (self.done_a > 0) | (self.done_b > 0)
        truncated = (self._t >= self._episode_length) & ~terminated \
            if self._episode_length is not None else torch.zeros_like(terminated)
        done = terminated | truncated
        i = self.layer.idx
        selected_foot = self._geom_xpos[
            torch.arange(self.nworld, device=self.device), self._foot_gids[leg_before]]
        target_distance = torch.linalg.vector_norm(
            selected_foot - self.xpos[:, i.Bt], dim=-1)
        selected_extension_delta = extension_delta[
            torch.arange(self.nworld, device=self.device), leg_before]
        up_t = wp.to_torch(self.layer.d.xmat)[:, i.At, 2, 2]
        height = self.xpos[:, i.At, 2]
        first_contact = ((support_by_leg > 0.0) &
                         (self._prev_support <= 0.0)).to(support_by_leg.dtype)
        zero4 = torch.zeros_like(support_by_leg)
        info = {
            "truncated": truncated.float(),
            "terminal_obs": terminal_obs,
            "terminal_priv": terminal_priv,
            "terminal_opponent_obs": terminal_b,
            "opponent_reward": self.reward_b.clone(),
            "priv": terminal_priv,
            "contact": support_by_leg,
            "first_contact": first_contact,
            "air_pre": zero4,
            "track": attack_metrics["attack_selected_hit"],
            "verr": target_distance,
            "align": attack_metrics["attack_support"],
            "speed": attack_metrics["attack_kick_speed"],
            "progress": selected_extension_delta,
            "up": up_t,
            "height": height,
            "fallrate": (self.done_a > 0).to(torch.float32),
            "constraint_rows": self.constraint_rows.clone(),
            "constraint_capacity": self.layer.njmax,
            "attack_leg": leg_before.to(torch.float32),
            "attack_active": active_before,
            "attack_phase": phase_before,
            "gait_phase": phase_before,
            "reward_components": {
                "combat_base": base_reward,
                "attack_task": task_reward,
            },
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
            "attack_hit_by_leg": hit_by_leg,
            "attack_support_by_leg": support_by_leg,
            **attack_metrics,
        }

        self._attack_timer.sub_(1)
        self._attack_phase_step.add_(1)
        curriculum_owned = ((self._attack_override_leg < 0) |
                            (self._attack_override_active < 0))
        resample = done | ((self._attack_timer <= 0) & curriculum_owned)
        self._sample_commands(resample)
        self._randomize_target_layout(done)
        if bool(done.any()):
            self._reset_opponent_state(done)
        if bool(done.any()):
            self._reset_opponent_state(done)
        self._reset_worlds(done)
        with wp.ScopedDevice(self._wp_device):
            mjwp.forward(self._wm, self._wd)
            self.layer.refresh_outputs()
        self._prev_extension.copy_(self._current_extension())
        self._prev_support.copy_(torch.where(
            done[:, None], torch.zeros_like(support_by_leg), support_by_leg))
        info["priv"] = self.privileged()
        return self.observe(), reward, done.float(), info


def main() -> None:
    import argparse

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--nworld", type=int, default=8)
    parser.add_argument("--steps", type=int, default=100)
    parser.add_argument("--device", default=None)
    parser.add_argument("--leg", choices=LEG_NAMES)
    args = parser.parse_args()
    env = LegAttackWarpEnv(args.nworld, device=args.device)
    if args.leg:
        env.set_attack_command(args.leg, True)
    action = torch.zeros((env.nworld, env.act_dim), device=env.device)
    totals = torch.zeros(env.nworld, device=env.device)
    for _ in range(args.steps):
        _, reward, _, info = env.step(action)
        totals += reward
    print(
        f"RESULT leg_attack nworld={env.nworld} steps={args.steps} "
        f"return={float(totals.mean()):.3f} "
        f"support={float(info['attack_support'].mean()):.3f}"
    )


if __name__ == "__main__":
    main()
