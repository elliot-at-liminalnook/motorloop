# SPDX-License-Identifier: MIT
"""Task-conditioned MuJoCo-Warp environments for the executable RL ladder.

The HTML training ladder is deliberately not implemented as 31 aliases for the
same reward.  Locomotion rungs 2..23 share one fixed observation/action contract
so an accepted policy can warm-start the next rung, while each rung changes one
physical demand: target pose/height, command family, impulses, collision
geometry, gravity-relative slope, payload, route state, or degraded ranging.

Combat rungs 24..29 likewise share the 51-value leg-command contract.  Rung 26
is the runtime-selectable kick policy; the adjacent rungs progressively add
approach, rod strike, pursuit, a frozen opponent, and finally PFSP orchestration.
"""

from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path
import xml.etree.ElementTree as ET

import mujoco
import mujoco_warp as mjwp
import torch
import torch.nn.functional as F
import warp as wp

from leg_attack_warp_env import LegAttackConfig, LegAttackWarpEnv
from codesign_warp_env import DesignEnsembleWarpEnv
from walker_warp_env import (CAT_DUTY_GRACE_STEPS, CAT_DUTY_MAX,
                             CAT_FOOT_DUTY_MAX, CAT_PROGRESS_GRACE_STEPS,
                             CAT_SLIP_LIMIT, OBS_DIM as WALKER_OBS_DIM,
                             WALKER_CONTACT_Z, WalkerWarpEnv)


LOCOMOTION_RUNG_MIN = 2
LOCOMOTION_RUNG_MAX = 23
COMBAT_RUNG_MIN = 24
COMBAT_RUNG_MAX = 29
LIDAR_RAYS = 144
LOCOMOTION_TASK_CHANNELS = 23
LOCOMOTION_EXTRA_DIM = 12 + 1 + 2 + 2 + LIDAR_RAYS + LOCOMOTION_TASK_CHANNELS
LADDER_LOCOMOTION_OBS_DIM = WALKER_OBS_DIM + LOCOMOTION_EXTRA_DIM
GAIT_PERIOD_STEPS = 50
ADAPTIVE_DUAL_MAX = 10.0
UNIVERSAL_OBS_DIM = 256
UNIVERSAL_PHYSICAL_DIM = 225
UNIVERSAL_TASK_CHANNELS = 31
UNIVERSAL_ACT_DIM = 14
UNIVERSAL_PRIV_DIM = 34

# Circles conservatively enclosing the boxes inserted for degraded-lidar nav.
NAV_OBSTACLES = ((0.45, 0.00, 0.16), (0.78, 0.28, 0.16),
                 (0.78, -0.28, 0.16), (1.10, 0.00, 0.16))


def normalized_duty_costs(info: dict[str, torch.Tensor],
                          swing_clearance: torch.Tensor | None = None,
                          ) -> tuple[torch.Tensor, torch.Tensor]:
    """Dense, scale-free costs for the same contracts used by CaT.

    CaT must terminate on the worst foot because every foot has to lift. Using
    that same maximum as a dense reward, however, gives no credit for the first
    three successful lifts: one not-yet-lifted foot keeps the maximum pinned at
    one. Averaging independently normalized per-foot excesses supplies an
    immediate signal for every successful swing while strict worst-foot
    termination remains unchanged. When physical swing clearance is available,
    its remaining normalized debt is the most timely proxy: it reaches zero at
    the exact height used by the duty contract rather than many actions later.
    """
    duty_value = info.get("cycle_duty", info["duty_ema"])
    foot_value = info.get("cycle_duty_by_leg", info["foot_duty_ema_by_leg"])
    duty_cost = ((duty_value - CAT_DUTY_MAX)
                 / max(1.0 - CAT_DUTY_MAX, 1.0e-6)).clamp(0.0, 1.0)
    per_foot = ((foot_value - CAT_FOOT_DUTY_MAX)
                / max(1.0 - CAT_FOOT_DUTY_MAX, 1.0e-6)).clamp(0.0, 1.0)
    foot_cost = per_foot.mean(dim=-1)
    if swing_clearance is not None:
        foot_cost = (1.0 - swing_clearance).clamp(0.0, 1.0)
    return duty_cost, foot_cost


def swing_clearance_scores(foot_z: torch.Tensor, desired_swing: torch.Tensor,
                           foot_radius: float,
                           ) -> tuple[torch.Tensor, torch.Tensor]:
    """Return mean and worst required-foot clearance on the contract scale.

    Mean clearance admits a half-gait loophole: one foot in each commanded
    diagonal can remain planted while its partner earns half credit forever.
    The lower envelope automatically focuses learning on whichever required
    swing foot currently has the least clearance, without leg-specific weights.
    """
    floor = foot_radius + 0.002
    span = max(WALKER_CONTACT_Z - floor, 1.0e-6)
    by_leg = ((foot_z - floor) / span).clamp(0.0, 1.0)
    count = desired_swing.sum(dim=-1).clamp_min(1.0)
    mean = (by_leg * desired_swing).sum(dim=-1) / count
    required = desired_swing > 0.5
    worst = torch.where(required, by_leg, torch.ones_like(by_leg)).amin(dim=-1)
    return mean, worst

LOCOMOTION_RUNG_NAMES = {
    2: "stand_and_settle", 3: "balance_under_push", 4: "track_pose_vector",
    5: "height_control", 6: "step_in_place", 7: "walk_forward",
    8: "velocity_tracking", 9: "turn_in_place", 10: "omnidirectional",
    11: "heading_hold", 12: "stop_on_command", 13: "servo_true_motors",
    14: "stumble_recovery", 15: "push_recovery", 16: "rough_terrain",
    17: "slope_walking", 18: "payload_carry", 19: "return_to_origin",
    20: "go_to_waypoint", 21: "route_following", 22: "route_correction",
    23: "degraded_lidar_nav",
}
COMBAT_RUNG_NAMES = {
    24: "approach_dummy", 25: "strike_target", 26: "commanded_leg_kick",
    27: "pursue_evader", 28: "beat_frozen_self", 29: "pfsp_league",
}


def _scenario_obstacles(variant: float) -> tuple[tuple[float, float, float], ...]:
    return tuple((x + (variant - 0.5) * 0.10 * (-1) ** index,
                  y + (variant - 0.5) * 0.08 * (-1) ** (index + 1), radius)
                 for index, (x, y, radius) in enumerate(NAV_OBSTACLES))


def _scenario_xml(xml: str, rung: int, variant: float = 0.5) -> str:
    """Insert the collision geometry that distinguishes terrain rungs."""
    root = ET.fromstring(xml)
    world = root.find("worldbody")
    if world is None:
        raise ValueError("walker MJCF has no worldbody")
    floor = next((g for g in world.findall("geom") if g.get("name") == "floor"), None)
    if rung == 14:
        bar_x = 0.50 + 0.25 * variant
        bar_height = 0.015 + 0.020 * variant
        ET.SubElement(world, "geom", name="ladder_trip_bar", type="box",
                      pos=f"{bar_x} 0 {0.5 * bar_height}",
                      size=f"0.025 0.34 {0.5 * bar_height}",
                      rgba="0.75 0.35 0.16 1", friction="1.2 0.1 0.1")
    elif rung == 16:
        # Raised tiles form a repeatable height field with both fore/aft and
        # left/right height changes. The plane remains beneath them (no holes).
        tiles = ((0.38, -0.18, 0.018), (0.38, 0.18, 0.035),
                 (0.72, -0.18, 0.045), (0.72, 0.18, 0.024),
                 (1.06, -0.18, 0.030), (1.06, 0.18, 0.050))
        height_scale = 0.70 + 0.60 * variant
        for index, (x, y, h) in enumerate(tiles):
            h *= height_scale
            ET.SubElement(world, "geom", name=f"rough_tile_{index}", type="box",
                          pos=f"{x} {y} {0.5 * h}", size=f"0.16 0.16 {0.5 * h}",
                          rgba="0.38 0.31 0.24 1",
                          friction=f"{0.55 + 0.30 * variant + 0.10 * (index % 3):.2f} "
                                   "0.08 0.08")
        if floor is not None:
            floor.set("friction", "0.85 0.08 0.08")
    elif rung == 17 and floor is not None:
        angle = math.radians(5.0 + 8.0 * variant)
        floor.set("quat", f"{math.cos(angle / 2)} 0 {math.sin(angle / 2)} 0")
    elif rung == 23:
        for index, (x, y, radius) in enumerate(_scenario_obstacles(variant)):
            side = radius / math.sqrt(2.0)
            ET.SubElement(world, "geom", name=f"nav_obstacle_{index}", type="box",
                          pos=f"{x} {y} 0.16", size=f"{side} {side} 0.16",
                          rgba="0.58 0.22 0.16 1", friction="1.0 0.1 0.1")
    return ET.tostring(root, encoding="unicode")


def _scenario_model(model: mujoco.MjModel, rung: int, variant: float = 0.5) -> None:
    """Apply model-field domain changes before the model is uploaded to Warp."""
    if rung == 18:
        torso = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "torso")
        factor = 1.15 + 0.30 * variant
        model.body_mass[torso] *= factor
        model.body_inertia[torso] *= factor


class LadderLocomotionWarpEnv(WalkerWarpEnv):
    """One stable policy contract for ladder rungs 2 through 23."""

    action_semantics = "pd_target@50hz:lowpass+torque_speed_v1"
    architecture_task_dim = LOCOMOTION_TASK_CHANNELS
    _phase_sin_obs_index = WALKER_OBS_DIM + 12 + 1 + 2

    def __init__(self, nworld: int, rung: int, seed: int = 0,
                 device: str | None = None, episode_length: int | None = 800,
                 **kwargs):
        rung = int(rung)
        if rung not in LOCOMOTION_RUNG_NAMES:
            raise ValueError(f"locomotion rung must be in [{LOCOMOTION_RUNG_MIN}, "
                             f"{LOCOMOTION_RUNG_MAX}], got {rung}")
        self.rung = rung
        self.rung_name = LOCOMOTION_RUNG_NAMES[rung]
        self.scenario_variant = float(kwargs.pop(
            "scenario_variant", ((seed * 2_654_435_761 + rung * 97_531) % 1_000_003)
            / 1_000_002.0))
        self._ladder_ready = False
        super().__init__(
            nworld=nworld, seed=seed, device=device, episode_length=episode_length,
            model_xml_transform=lambda xml: _scenario_xml(
                xml, rung, self.scenario_variant),
            model_transform=lambda model: _scenario_model(
                model, rung, self.scenario_variant),
            **kwargs,
        )
        # The base walker loads a phase-locked reference gait for its standalone
        # imitation curriculum.  Ladder training deliberately owns every
        # scaffold through its explicit action-prior path, so discard that
        # hidden teacher entirely.  This also keeps startup diagnostics honest:
        # a direct outcome-only rung reports imitation=off.
        self._gait = None
        dev = self.device
        self._pose_command = torch.zeros((nworld, 12), device=dev)
        self._height_command = torch.full((nworld,), 0.38, device=dev)
        self._goal = torch.zeros((nworld, 2), device=dev)
        self._heading_command = torch.zeros(nworld, device=dev)
        self._velocity_command = torch.zeros((nworld, 3), device=dev)
        self._route_index = torch.zeros(nworld, dtype=torch.long, device=dev)
        self._task_t = torch.zeros(nworld, dtype=torch.long, device=dev)
        self._constraint_age = torch.zeros(nworld, dtype=torch.long, device=dev)
        self._prev_goal_distance = torch.zeros(nworld, device=dev)
        self._lidar = torch.ones((nworld, LIDAR_RAYS), device=dev)
        self._lidar_previous = torch.ones_like(self._lidar)
        self._routes = self._route_tensor()
        self._obstacles = torch.as_tensor(
            _scenario_obstacles(self.scenario_variant), dtype=torch.float32, device=dev)
        self._disturbance_impulse = torch.zeros(nworld, device=dev)
        self._disturbance_fire = torch.zeros(nworld, dtype=torch.long, device=dev)
        self._walk_prior_coefficients: torch.Tensor | None = None
        self._walk_prior_blend = 0.65
        # The dual controller owns only genuine physical constraints.  Earlier
        # versions also treated progress, duty cycle, and support pattern as
        # catastrophes; that terminated unconventional-but-valid locomotion and
        # silently turned gait style into a safety law.
        constraint_count = 1
        self._constraint_duals = torch.zeros(constraint_count, device=dev)
        self._constraint_error_square = torch.zeros(constraint_count, device=dev)
        self._constraint_limits = torch.full((constraint_count,), 0.001, device=dev)
        self._competence_duals = torch.zeros(1, device=dev)
        self._competence_error_square = torch.zeros(1, device=dev)
        competence_target = 0.40 if rung == 6 else 0.10
        self._competence_targets = torch.full(
            (1,), competence_target, device=dev)
        # Exact episode-cycle contact accounting.  An EMA sampled at the end of
        # a period unfairly forgets the pair that swung early in that period.
        self._cycle_contact_sum = torch.zeros((nworld, 4), device=dev)
        self._cycle_contact_steps = torch.zeros(nworld, device=dev)
        self._ladder_ready = True
        self.reset()

    @property
    def obs_dim(self):
        return LADDER_LOCOMOTION_OBS_DIM if self._ladder_ready else WALKER_OBS_DIM

    @property
    def observation_semantics(self):
        return ("ladder_locomotion234:pose12+height1+goal2+"
                "heading_or_phase2+lidar144+task23:v3")

    @property
    def external_clock_observation_indices(self) -> tuple[int, ...]:
        """Actor-observation fields that encode the hand-authored gait clock.

        Feed-forward checkpoints retain these channels for compatibility.  The
        recurrent architecture masks them and must infer temporal phase from
        physical history.  Rung 11 moves the clock into the first two pose
        command channels because its heading channels carry the real command.
        """
        if self.rung < 6:
            return ()
        start = WALKER_OBS_DIM if self.rung == 11 else self._phase_sin_obs_index
        return (start, start + 1)

    @property
    def reward_semantics(self):
        version = {3: 2, 6: 13, 7: 15, 12: 2, 15: 2, 16: 2, 22: 2}.get(
            self.rung, 1)
        objective = "objective4" if self.rung in (6, 7) else "objective3"
        return (f"ladder:{self.rung:02d}:{self.rung_name}:v{version}:{objective}"
                "+catprog")

    @property
    def cat_term_keys(self) -> tuple[str, ...]:
        """Physical boundaries plus ONE outcome termination.

        cat_progress terminates a world that sustains near-zero progress while
        a motion command is active (command-gated, EMA-matured, grace period).
        It is an OUTCOME constraint — failure to do the commanded task — not a
        style: the contract's termination ban targets gait-shape preferences.
        Restored 2026-07-24 after pressure-only duals lost to the standing
        attractor on both curriculum orderings (rung 6 stand-first, rung 8
        walk-first): a policy whose bootstrap survives standing under command
        out-earns every first step toward motion.
        """
        return ("cat_slip", "cat_orient", "cat_qvel", "cat_body",
                "cat_progress")

    def configure_action_prior(self, path: str | Path) -> None:
        """Load a searched behavior prior artifact without embedding it in code."""
        artifact = json.loads(Path(path).read_text())
        best = artifact.get("best", artifact)
        values = best.get("parameters")
        if not isinstance(values, list) or len(values) != 24:
            raise ValueError("walk prior must contain 24 yaw/pitch Fourier parameters")
        self._walk_prior_coefficients = torch.as_tensor(
            values, dtype=torch.float32, device=self.device).reshape(4, 2, 3)
        self._walk_prior_blend = float(artifact.get("blend", 0.65))
        if not 0.0 <= self._walk_prior_blend <= 1.0:
            raise ValueError("walk prior blend must be in [0, 1]")

    @property
    def adaptive_constraint_names(self) -> tuple[str, ...]:
        return ("cat_slip",)

    @property
    def constraint_duals(self) -> torch.Tensor:
        return self._constraint_duals

    @property
    def adaptive_constraint_limits(self) -> tuple[float, ...]:
        return tuple(float(value) for value in self._constraint_limits)

    @property
    def adaptive_dual_max(self) -> float:
        return ADAPTIVE_DUAL_MAX

    @property
    def action_prior_suppression_pressure(self) -> torch.Tensor:
        """Only pressure that says the motion teacher itself is unsafe.

        Slip is caused by the yaw/pitch traveling prior and should make that
        teacher yield. Duty pressure means feet are not lifting reliably, so it
        must instead strengthen the lift clock inside ``policy_mean_prior``.
        Collapsing both meanings with ``max()`` suppresses the very scaffold
        needed to resolve a duty violation.
        """
        return self._constraint_duals[0]

    @property
    def action_prior_lift_pressure(self) -> torch.Tensor:
        return self._constraint_duals.new_zeros(())

    @property
    def duty_constraint_grace_steps(self) -> int:
        return max(CAT_DUTY_GRACE_STEPS, GAIT_PERIOD_STEPS)

    @property
    def reset_phase_randomization_probability(self) -> torch.Tensor:
        """Pressure-driven phase coverage without a hand-authored curriculum.

        Forward walking keeps full phase coverage while its acquisition teacher
        exists. Rung 6 has no target phase at all: arbitrary foot order is part
        of the behavior space rather than a randomized version of one gait.
        """
        if self.rung == 7:
            return self._constraint_duals.new_ones(())
        return self._constraint_duals.new_zeros(())

    @staticmethod
    def _update_dual_state(duals: torch.Tensor, error_square: torch.Tensor,
                           normalized_violation: torch.Tensor) -> None:
        """Scale-free ascent on violation and multiplicative release on success."""
        error_square.lerp_(normalized_violation.square(), 0.05)
        normalized_direction = normalized_violation / (
            error_square.sqrt().clamp_min(1.0))
        violating = normalized_violation > 0.0
        duals.mul_(torch.where(
            violating, torch.ones_like(duals), torch.full_like(duals, 0.95)))
        duals.add_(0.02 * normalized_direction.clamp_min(0.0)).clamp_(
            0.0, ADAPTIVE_DUAL_MAX)

    def update_constraint_duals(self, observed: torch.Tensor) -> None:
        """Projected RMS-normalized dual ascent against physical limits.

        A raw dual update is extremely sensitive to the units and initial size
        of a constraint violation.  In particular, a new walking policy can be
        tens of times over a strict slip budget during its first few rollouts;
        letting that transient set the multiplier immediately destroys the
        locomotion objective before the policy can respond.  The running second
        moment makes the controller scale-free: persistent violation controls
        how *long* the multiplier rises, while one large transient controls only
        its normalization.
        """
        if observed.shape != self._constraint_duals.shape:
            raise ValueError("constraint observation shape does not match dual state")
        normalized_violation = observed / self._constraint_limits - 1.0
        # RMS-normalized ascent is deliberately conservative on a transient.
        # Once a constraint is actually satisfied, however, stale pressure must
        # not pin the policy at a safe-but-motionless solution.
        self._update_dual_state(
            self._constraint_duals, self._constraint_error_square,
            normalized_violation)

    @property
    def adaptive_competence_names(self) -> tuple[str, ...]:
        if self.rung == 6:
            return ("ladder_foot_activity",)
        return ("xprogress",) if self.rung == 7 else ()

    @property
    def competence_duals(self) -> torch.Tensor:
        return self._competence_duals

    @property
    def adaptive_competence_targets(self) -> tuple[float, ...]:
        return tuple(float(value) for value in self._competence_targets)

    def update_competence_duals(self, observed: torch.Tensor) -> None:
        """Dual ascent for lower-bound skill contracts such as progress."""
        if observed.shape != self._competence_duals.shape:
            raise ValueError("competence observation shape does not match dual state")
        normalized_shortfall = 1.0 - observed / self._competence_targets
        self._update_dual_state(
            self._competence_duals, self._competence_error_square,
            normalized_shortfall)

    @property
    def action_prior_scale(self) -> float:
        # A prior is an explicit, versioned acquisition aid, never an implicit
        # property of a ladder rung.  In particular, the accepted rung-6 anchor
        # is a retention teacher, not authorization to invent a phase-locked
        # walking target.  Keep its optimizer weight exactly zero until the
        # runner supplies and configures a searched prior artifact.
        if self._walk_prior_coefficients is None:
            return 0.0
        return 10.0 if self.rung == 7 else 2.0

    @property
    def action_prior_floor_fraction(self) -> float:
        # Rung 7 otherwise forgets propulsion as soon as the transfer loss reaches
        # zero.  A small floor preserves the searched gait scaffold while PPO is
        # still free to alter it in response to slip/fall constraints.
        return 0.10 if self.rung == 7 else 0.0

    @property
    def action_prior_competence_metric(self) -> str | None:
        return "xprogress" if self.rung == 7 else None

    def _route_tensor(self) -> torch.Tensor:
        routes = {
            19: ((0.0, 0.0),),
            20: ((0.70, 0.15),),
            21: ((0.55, 0.0), (0.55, 0.50), (0.05, 0.50), (0.05, 0.05)),
            22: ((0.55, 0.0), (0.55, 0.50), (0.05, 0.50), (0.05, 0.05)),
            23: ((0.35, 0.38), (0.92, 0.46), (1.20, -0.34), (0.35, -0.42)),
        }
        points = routes.get(self.rung, ((0.0, 0.0),))
        return torch.as_tensor(points, dtype=torch.float32, device=self.device)

    def _yaw(self) -> torch.Tensor:
        q = self.qpos
        return torch.atan2(2.0 * (q[:, 3] * q[:, 6] + q[:, 4] * q[:, 5]),
                           1.0 - 2.0 * (q[:, 5] ** 2 + q[:, 6] ** 2))

    def _goal_local(self) -> torch.Tensor:
        delta = self._goal - self.qpos[:, :2]
        c, s = torch.cos(self._yaw()), torch.sin(self._yaw())
        return torch.stack((c * delta[:, 0] + s * delta[:, 1],
                            -s * delta[:, 0] + c * delta[:, 1]), dim=-1)

    def _clean_lidar(self) -> torch.Tensor:
        if self.rung != 23:
            return torch.ones((self.nworld, LIDAR_RAYS), device=self.device)
        origin = self.qpos[:, :2]
        angles = self._yaw()[:, None] + torch.linspace(
            -math.pi, math.pi, LIDAR_RAYS + 1, device=self.device)[:-1]
        direction = torch.stack((torch.cos(angles), torch.sin(angles)), dim=-1)
        center = self._obstacles[:, :2]
        radius = self._obstacles[:, 2]
        offset = center[None, None, :, :] - origin[:, None, None, :]
        projection = (offset * direction[:, :, None, :]).sum(dim=-1)
        perpendicular2 = (offset * offset).sum(dim=-1) - projection ** 2
        inside = radius[None, None, :] ** 2 - perpendicular2
        distance = projection - torch.sqrt(inside.clamp_min(0.0))
        valid = (inside >= 0.0) & (distance > 0.0)
        distance = torch.where(valid, distance, torch.full_like(distance, 2.0))
        return (distance.amin(dim=-1) / 2.0).clamp(0.0, 1.0)

    def _update_lidar(self) -> None:
        clean = self._clean_lidar()
        if self.rung != 23:
            self._lidar.copy_(clean)
            self._lidar_previous.copy_(clean)
            return
        world = torch.arange(self.nworld, device=self.device)[:, None]
        ray = torch.arange(LIDAR_RAYS, device=self.device)[None, :]
        phase = world * 17.17 + ray * 3.11 + self._task_t[:, None] * 0.73
        noise = 0.025 * torch.sin(phase)
        dropout = torch.frac(torch.abs(torch.sin(phase * 1.913)) * 43758.5453) > 0.94
        degraded = torch.where(dropout, torch.ones_like(clean), (clean + noise).clamp(0.0, 1.0))
        # One-control-step latency, deliberately exposed only on the final nav rung.
        self._lidar.copy_(self._lidar_previous)
        self._lidar_previous.copy_(degraded)

    def observe(self) -> torch.Tensor:
        base = super().observe()
        if not self._ladder_ready:
            return base
        pose = (self._pose_command - self._stand) / self._authority.clamp_min(1.0e-5)
        heading_error = torch.atan2(
            torch.sin(self._heading_command - self._yaw()),
            torch.cos(self._heading_command - self._yaw()))
        heading = torch.stack((torch.sin(heading_error), torch.cos(heading_error)), dim=-1)
        task = F.one_hot(torch.full(
            (self.nworld,), self.rung - 1, device=self.device, dtype=torch.long),
            num_classes=LOCOMOTION_TASK_CHANNELS).to(torch.float32)
        return torch.cat((base, pose, self._height_command[:, None], self._goal_local(),
                          heading, self._lidar, task), dim=-1)

    def policy_mean_prior(self, raw_obs: torch.Tensor, base_mean: torch.Tensor | None = None,
                          transfer_action: torch.Tensor | None = None
                          ) -> tuple[torch.Tensor, torch.Tensor] | None:
        """Verified phase-to-lift demonstration for the first stepping rung.

        Targets are pre-tanh actor means. Only the four lift channels are
        supervised; PPO remains responsible for yaw/pitch balance and the
        physical contact outcome.
        """
        if self.rung < 7 or self._walk_prior_coefficients is None:
            return None
        # Heading channels carry phase on most movement rungs. Rung 11 needs
        # them for the heading target, so its first pose-command channel carries
        # the same clock instead.
        phase_index = WALKER_OBS_DIM if self.rung == 11 else self._phase_sin_obs_index
        phase_sin = raw_obs[:, phase_index]
        pair_a = (phase_sin >= 0.0).to(raw_obs.dtype).unsqueeze(-1)
        desired_swing = self._pair_a * pair_a + self._pair_b * (1.0 - pair_a)
        target = (base_mean.clone() if base_mean is not None else torch.zeros(
            (len(raw_obs), self.act_dim), dtype=raw_obs.dtype, device=raw_obs.device))
        mask = torch.full_like(target, 0.25 if base_mean is not None else 0.0)
        if self.rung == 7:
            # A square lift target reverses instantaneously at half-cycle and
            # creates a repeatable high-speed touchdown.  The sinusoid carries
            # the same diagonal gait clock but has continuous position and
            # velocity, giving PPO a physically realizable landing trajectory.
            lift_wave = phase_sin[:, None] * (self._pair_a - self._pair_b)
            target[:, 2::3] = 1.85 * lift_wave
        else:
            target[:, 2::3] = 1.85 * (2.0 * desired_swing - 1.0)
        if self.rung == 7 and base_mean is not None:
            action_target = torch.tanh(target)
            if transfer_action is not None:
                if transfer_action.shape != action_target.shape:
                    raise ValueError("transfer action must match the ladder action shape")
                # A legacy state-conditioned walker supplies substantially more
                # useful fore/aft coordination than the open-loop residual below.
                # Search writes the Fourier correction as a versioned JSON
                # artifact.  This environment only defines the abstraction and
                # validates its shape; no discovered gait coefficients live here.
                blend = self._walk_prior_blend
                action_target[:, 0::3].lerp_(transfer_action[:, 0::3], blend)
                action_target[:, 1::3].lerp_(transfer_action[:, 1::3], blend)
                if self._walk_prior_coefficients is not None:
                    coefficients = self._walk_prior_coefficients.to(raw_obs.dtype)
                    phase_cos = raw_obs[:, self._phase_sin_obs_index + 1]
                    correction = (coefficients[None, ..., 0]
                                  + coefficients[None, ..., 1] * phase_sin[:, None, None]
                                  + coefficients[None, ..., 2] * phase_cos[:, None, None])
                    action_target[:, 0::3].add_(correction[..., 0])
                    action_target[:, 1::3].add_(correction[..., 1])
            else:
                # Reproducible fallback used by unit tests and installations that
                # do not ship the legacy walker diagnostic checkpoint.
                pitch_amp = raw_obs.new_tensor((-0.0190, 0.0290, -0.0823, -0.3907))
                pitch_bias = raw_obs.new_tensor((0.2098, 0.0807, -0.2969, 0.0134))
                yaw_amp = raw_obs.new_tensor((-0.2981, 0.2250, -0.1583, 0.0340))
                yaw_bias = raw_obs.new_tensor((-0.3628, 0.1626, -0.1992, -0.0939))
                phase_sign = 2.0 * desired_swing - 1.0
                action_target[:, 1::3].add_(pitch_amp * phase_sign + pitch_bias)
                action_target[:, 0::3].add_(yaw_amp * phase_sign + yaw_bias)
            target = torch.atanh(action_target.clamp(-0.95, 0.95))
            mask[:, 0::3] = 1.0
            mask[:, 1::3] = 1.0
        command = raw_obs[:, WALKER_OBS_DIM - 3:WALKER_OBS_DIM]
        active = (torch.linalg.vector_norm(command, dim=-1, keepdim=True) > 0.05) \
            .to(raw_obs.dtype)
        lift_pressure = self.action_prior_lift_pressure
        residual_scale = 1.0 / (1.0 + lift_pressure)
        # The open-loop clock is an acquisition scaffold, not ground truth for
        # actuator timing.  Once physical duty pressure rises, yaw/pitch teacher
        # channels yield and the lift target stays finite instead of becoming
        # more rigid.  PPO can then learn the phase lead required by measured
        # motor/contact dynamics from the robust clearance objective.
        mask[:, 2::3] = active
        mask[:, 0::3] *= active * residual_scale
        mask[:, 1::3] *= active * residual_scale
        return target, mask

    def _sample_task(self, mask: torch.Tensor, *, place_robot: bool) -> None:
        mask = mask.bool()
        m2 = mask[:, None]
        n, dev = self.nworld, self.device
        pose_noise = torch.rand((n, 12), generator=self._gen, device=dev) * 1.4 - 0.7
        pose = (self._stand + pose_noise * self._authority).clamp(self._jr_lo, self._jr_hi)
        height = 0.34 + 0.09 * torch.rand(n, generator=self._gen, device=dev)
        heading = (torch.rand(n, generator=self._gen, device=dev) * 2.0 - 1.0) * math.pi
        angle = (torch.rand(n, generator=self._gen, device=dev) * 2.0 - 1.0) * math.pi
        speed = 0.18 + 0.35 * torch.rand(n, generator=self._gen, device=dev)
        velocity = torch.stack((speed * torch.cos(angle), speed * torch.sin(angle),
                                0.6 * torch.sin(2.0 * angle)), dim=-1)
        push_sign = torch.where(
            torch.rand(n, generator=self._gen, device=dev) > 0.5, 1.0, -1.0)
        push_scale = 0.70 + 0.60 * torch.rand(n, generator=self._gen, device=dev)
        base_impulse = {3: 0.55, 15: 0.85, 22: 0.75}.get(self.rung, 0.0)
        disturbance = push_sign * push_scale * base_impulse
        fire = torch.randint(35, 85, (n,), generator=self._gen, device=dev)
        self._pose_command.copy_(torch.where(m2, pose, self._pose_command))
        self._height_command.copy_(torch.where(mask, height, self._height_command))
        self._heading_command.copy_(torch.where(mask, heading, self._heading_command))
        self._velocity_command.copy_(torch.where(m2, velocity, self._velocity_command))
        self._disturbance_impulse.copy_(torch.where(
            mask, disturbance, self._disturbance_impulse))
        self._disturbance_fire.copy_(torch.where(mask, fire, self._disturbance_fire))
        self._route_index.masked_fill_(mask, 0)
        if self.rung == 7 and place_robot:
            # This only broadens the temporary acquisition teacher. Promotion is
            # outcome-based and does not score agreement with this phase.
            phase = torch.randint(
                0, GAIT_PERIOD_STEPS, (n,), generator=self._gen, device=dev)
            probability = self.reset_phase_randomization_probability
            randomize = (torch.rand(
                n, generator=self._gen, device=dev) < probability) & mask
            next_phase = torch.where(randomize, phase, torch.zeros_like(phase))
            self._task_t.copy_(torch.where(mask, next_phase, self._task_t))
        else:
            self._task_t.masked_fill_(mask, 0)
        if place_robot:
            self._constraint_age.masked_fill_(mask, 0)
            self._cycle_contact_sum.masked_fill_(m2, 0.0)
            self._cycle_contact_steps.masked_fill_(mask, 0.0)

        if self.rung == 19 and place_robot:
            start_angle = (torch.rand(n, generator=self._gen, device=dev) * 2.0 - 1.0) * math.pi
            start_radius = 0.30 + 0.25 * torch.rand(n, generator=self._gen, device=dev)
            start = torch.stack((start_radius * torch.cos(start_angle),
                                 start_radius * torch.sin(start_angle)), dim=-1)
            self.qpos[:, :2].copy_(torch.where(m2, start, self.qpos[:, :2]))
            self._prev_xy.copy_(torch.where(m2, start, self._prev_xy))
        if self.rung >= 19:
            first = self._routes[0].expand(n, -1)
            self._goal.copy_(torch.where(m2, first, self._goal))
        else:
            self._goal.copy_(torch.where(m2, self.qpos[:, :2], self._goal))
        distance = torch.linalg.vector_norm(self._goal - self.qpos[:, :2], dim=-1)
        self._prev_goal_distance.copy_(torch.where(mask, distance, self._prev_goal_distance))

    def reset(self, mask: torch.Tensor | None = None) -> torch.Tensor:
        if not getattr(self, "_ladder_ready", False):
            return super().reset(mask)
        if mask is None:
            mask = torch.ones(self.nworld, dtype=torch.bool, device=self.device)
        super().reset(mask)
        self._sample_task(mask, place_robot=True)
        with wp.ScopedDevice(self._wp_device):
            mjwp.forward(self._wm, self._wd)
        self._update_lidar()
        return self.observe()

    def _set_task_command(self, alpha: float = 1.0) -> None:
        t = self._task_t
        cmd = self._velocity_command.clone()
        if self.rung >= 6:
            phase = (2.0 * math.pi * torch.remainder(
                t, GAIT_PERIOD_STEPS).to(torch.float32) / GAIT_PERIOD_STEPS)
            if self.rung == 11:
                self._pose_command[:, 0] = self._stand[0] + self._authority[0] * torch.sin(phase)
                self._pose_command[:, 1] = self._stand[1] + self._authority[1] * torch.cos(phase)
            else:
                self._heading_command.copy_(self._yaw() + phase)
        if self.rung <= 6:
            cmd.zero_()
        elif self.rung == 7:
            # Self-paced command curriculum: servo authority opens the upper
            # speed bound, while learned constraint pressure temporarily closes
            # it again.  Once contacts become clean the dual falls and speed
            # returns automatically, without a per-attempt command schedule.
            cmd.zero_()
            cmd[:, 0] = 0.10 + 0.22 * float(alpha) / (
                1.0 + self._constraint_duals.max())
        elif self.rung == 8:
            cmd[:, 0] = self._velocity_command[:, :2].norm(dim=-1)
            cmd[:, 1:].zero_()
            # Walk-first acquisition: a fixed stripe of worlds holds v=0 so
            # commanded standing is learned WITH locomotion, never as a prior
            # attractor stepping must later escape. Deterministic (no RNG
            # draw) to keep replay exactness untouched.
            cmd[::8, 0] = 0.0
        elif self.rung == 9:
            cmd.zero_(); cmd[:, 2] = self._velocity_command[:, 2].sign() * 0.55
        elif self.rung == 10:
            pass
        elif self.rung == 11:
            cmd[:, :2] *= 0.55
            err = torch.atan2(torch.sin(self._heading_command - self._yaw()),
                              torch.cos(self._heading_command - self._yaw()))
            cmd[:, 2] = (1.6 * err).clamp(-0.8, 0.8)
        elif self.rung == 12:
            moving = (torch.remainder(t, 120) < 70).to(torch.float32)
            cmd[:, 0] = 0.32 * moving
            cmd[:, 1:].zero_()
        elif self.rung == 13:
            pass
        elif self.rung == 16:
            cmd[:] = torch.tensor((0.28, 0.0, 0.0), device=self.device)
        elif self.rung in (14, 15, 17, 18):
            cmd[:] = torch.tensor((0.28, 0.0, 0.0), device=self.device)
        elif self.rung >= 19:
            delta = self._goal - self.qpos[:, :2]
            distance = torch.linalg.vector_norm(delta, dim=-1, keepdim=True).clamp_min(1.0e-5)
            cmd.zero_()
            cmd[:, :2] = delta / distance * torch.minimum(
                torch.full_like(distance, 0.38), 1.2 * distance)
            desired = torch.atan2(delta[:, 1], delta[:, 0])
            yaw_error = torch.atan2(torch.sin(desired - self._yaw()),
                                    torch.cos(desired - self._yaw()))
            cmd[:, 2] = (1.2 * yaw_error).clamp(-0.7, 0.7)
        self._cmd.copy_(cmd)
        self._timer.zero_()  # the rung owns command timing, not the base sampler

    def _inject_disturbance(self) -> None:
        t = self._task_t
        period = {3: 140, 15: 100, 22: 160}.get(self.rung)
        if period is not None:
            fire = torch.remainder(t, period) == self._disturbance_fire.clamp_max(period - 1)
            self.qvel[:, 1].add_(
                fire.to(self.qvel.dtype) * self._disturbance_impulse)

    def _cat_violations(self, cf: torch.Tensor, foot_hspeed: torch.Tensor,
                        up: torch.Tensor, cmd_norm: torch.Tensor,
                        active: torch.Tensor, progress_mature: torch.Tensor,
                        duty_mature: torch.Tensor, progress_ema: torch.Tensor,
                        duty_ema: torch.Tensor,
                        foot_duty_ema: torch.Tensor):
        """Exclude known exogenous impulse ticks from only the slip constraint.

        A scripted base-velocity impulse can instantaneously move a planted foot
        in world coordinates even when the controller did nothing wrong.  All
        other constraints remain live, and slip becomes active again after a
        short recovery grace window.
        """
        # The ladder clears the base command timer every tick so its task-owned
        # command cannot be overwritten.  Constraint grace must therefore use a
        # separate per-episode clock; using the command timer permanently disabled
        # progress, duty, per-foot-duty, and support constraints.
        progress_mature = self._constraint_age >= CAT_PROGRESS_GRACE_STEPS
        # Duty is a cycle-level property. Starting this check before a complete
        # commanded gait period guarantees false failures for legs whose swing
        # phase has not occurred yet.
        duty_mature = self._constraint_age >= self.duty_constraint_grace_steps
        # Rung 6 intentionally has zero translational command, but stepping is
        # still an active gait task.  Treat its duty/support contracts as active
        # while keeping progress disabled; tying both notions to command speed
        # previously let a rear-only shuffle pass the prerequisite rung.
        constraint_active = (torch.ones_like(active) if self.rung == 6 else active)
        if self.rung == 6:
            progress_mature = torch.zeros_like(progress_mature)
        self._cycle_contact_sum.add_(cf)
        self._cycle_contact_steps.add_(1.0)
        cycle_duty_by_leg = self._cycle_contact_sum / self._cycle_contact_steps[:, None]
        cycle_duty = cycle_duty_by_leg.mean(dim=-1)
        terms, progress_req = super()._cat_violations(
            cf, foot_hspeed, up, cmd_norm, constraint_active, progress_mature,
            duty_mature, progress_ema, cycle_duty, cycle_duty_by_leg)
        # Extra terms are carried into ``info`` by WalkerWarpEnv but are not part
        # of its CAT_TERM_KEYS maximum.  They make the exact contract observable.
        terms["cycle_duty"] = cycle_duty
        terms["cycle_duty_by_leg"] = cycle_duty_by_leg
        terms["foot_cycle_duty"] = cycle_duty_by_leg.amax(dim=-1)
        period = {3: 140, 15: 100, 22: 160}.get(self.rung)
        if period is not None:
            offset = (torch.remainder(self._task_t, period)
                      - self._disturbance_fire.clamp_max(period - 1))
            grace = (offset >= 0) & (offset < 4)
            terms["cat_slip"] = torch.where(
                grace, torch.zeros_like(terms["cat_slip"]), terms["cat_slip"])
        return terms, progress_req

    def _task_reward(self, info: dict, action: torch.Tensor
                     ) -> tuple[torch.Tensor, dict[str, torch.Tensor],
                                dict[str, torch.Tensor]]:
        q = self.qpos
        up = info["up"]
        speed = torch.linalg.vector_norm(self.qvel[:, :2], dim=-1)
        pose_error = torch.sqrt(((q[:, self._qa] - self._pose_command) ** 2).mean(dim=-1))
        pose_score = torch.exp(-pose_error / 0.20)
        height_error = (info["height"] - self._height_command).abs()
        height_score = torch.exp(-height_error / 0.035)
        yaw_error = (self.qvel[:, 5] - self._cmd[:, 2]).abs()
        yaw_score = torch.exp(-yaw_error / 0.35)
        heading_error = torch.atan2(torch.sin(self._heading_command - self._yaw()),
                                    torch.cos(self._heading_command - self._yaw())).abs()
        heading_score = torch.exp(-heading_error / 0.40)
        goal_distance = torch.linalg.vector_norm(self._goal - q[:, :2], dim=-1)
        goal_progress = self._prev_goal_distance - goal_distance
        goal_hit = (goal_distance < 0.10).to(torch.float32)
        stop_phase = (torch.remainder(self._task_t, 120) >= 70).to(torch.float32)
        stop_score = torch.exp(-speed / 0.08)
        move_progress = (1.0 - stop_phase) * info["xprogress"]
        stop_compliance = stop_phase * stop_score
        # Competence pressure and the public physical gate must observe the same
        # event over compatible windows.  A short EMA starts each episode at
        # "all feet planted" and severely under-reports activity in PPO's
        # 64-step rollout even when a full held-out episode already passes.
        # Exact episode contact coverage removes that false shortfall.  Keep the
        # EMA-derived value only as a diagnostic so a window mismatch is visible.
        activity_duty_by_leg = info.get(
            "cycle_duty_by_leg", info["foot_duty_ema_by_leg"])
        per_foot_activity = ((1.0 - activity_duty_by_leg)
                             / max(1.0 - CAT_FOOT_DUTY_MAX, 1.0e-6)).clamp(0.0, 1.0)
        foot_activity_mean = per_foot_activity.mean(dim=-1)
        foot_activity = per_foot_activity.amin(dim=-1)
        ema_per_foot_activity = ((1.0 - info["foot_duty_ema_by_leg"])
                                 / max(1.0 - CAT_FOOT_DUTY_MAX, 1.0e-6)).clamp(0.0, 1.0)
        foot_activity_ema = ema_per_foot_activity.amin(dim=-1)
        step_phase_a = (torch.remainder(self._task_t, GAIT_PERIOD_STEPS)
                        < GAIT_PERIOD_STEPS // 2).to(torch.float32).unsqueeze(-1)
        desired_swing = (self._pair_a * step_phase_a
                         + self._pair_b * (1.0 - step_phase_a))
        foot_z = self.geom_xpos[:, self._feet, 2]
        # Use the exact same airborne threshold as the public duty contract.
        # Previously shaping declared full success at 39 mm while CaT required
        # 50 mm, creating an 11 mm region rewarded as success but terminated as
        # failure.
        physical_contact = info["contact"]
        step_clock = (desired_swing * (1.0 - physical_contact)
                      + (1.0 - desired_swing) * physical_contact).mean(dim=-1)
        swing_clearance, worst_swing_clearance = swing_clearance_scores(
            foot_z, desired_swing, self._foot_r)
        if self.rung == 7:
            smooth_phase = torch.sin(
                2.0 * math.pi * torch.remainder(
                    self._task_t, GAIT_PERIOD_STEPS).to(torch.float32)
                / GAIT_PERIOD_STEPS)
            desired_lift_action = smooth_phase[:, None] * (self._pair_a - self._pair_b)
        else:
            desired_lift_action = 2.0 * desired_swing - 1.0
        step_action_score = torch.exp(
            -((action[:, 2::3] - desired_lift_action) ** 2).mean(dim=-1) / 0.25)
        normalized_slip = (info["cat_slip"] / self._constraint_limits[0]).clamp_max(20.0)
        # Dense contact-consistency cost. CaT only fires after the physical
        # threshold is crossed, which is the right termination contract but a
        # sparse learning signal. Stance feet should be stationary in the world
        # at *any* speed, so this normalized squared speed is a parameter-free
        # control-barrier proxy whose strength comes from the learned dual.
        stance_slip_ratio = (info["stance_foot_speed"] / CAT_SLIP_LIMIT).clamp_max(3.0)
        stance_slip_cost = stance_slip_ratio.square()
        task_outcome = 0.25 * up.clamp_min(0.0)
        task_constraint = torch.zeros_like(task_outcome)
        task_scaffold = torch.zeros_like(task_outcome)

        if self.rung in (2, 3):
            task_outcome = 2.0 * up.clamp_min(0.0) + 0.8 * stop_score
        elif self.rung == 4:
            task_outcome = 4.0 * pose_score + up.clamp_min(0.0)
        elif self.rung == 5:
            task_outcome = 4.0 * height_score + up.clamp_min(0.0)
        elif self.rung == 6:
            # Outcome: every foot has recently unloaded, the body remains still,
            # and the robot remains upright. Nothing specifies order, pairing,
            # phase, cadence, symmetry, or joint trajectory. The per-leg mean is
            # a temporary dense discovery scaffold; learned competence pressure
            # makes it disappear once worst-leg coverage is sufficient.
            still_score = torch.exp(-speed / 0.08)
            stable_score = ((up - 0.85) / 0.15).clamp(0.0, 1.0)
            outcome = torch.pow(
                (foot_activity * still_score * stable_score).clamp_min(1.0e-8),
                1.0 / 3.0)
            competence_pressure = self._competence_duals[0].clamp_min(0.0)
            discovery_scale = competence_pressure / (1.0 + competence_pressure)
            # The permanent physical outcome keeps a fixed scale.  Only the
            # dense discovery scaffold adapts, and its bounded coefficient
            # cannot make the critic chase an ever-growing return distribution.
            task_outcome = outcome
            task_scaffold = discovery_scale * foot_activity_mean
        elif self.rung == 7:
            progress_target = 0.10
            normalized_progress = info["xprogress"].clamp(-0.4, 0.4) / progress_target
            safe_progress = info["xprogress"] * (info["cat_slip"] <= 0.0).to(q.dtype)
            competence_pressure = self._competence_duals[0].clamp_min(0.0)
            competence_scale = competence_pressure / (1.0 + competence_pressure)
            task_outcome = ((1.0 + competence_scale) * normalized_progress
                            - info["lateral"] / 0.15
                            + 0.25 * up.clamp_min(0.0))
            task_constraint = (-self._constraint_duals[0]
                               * (normalized_slip + stance_slip_cost))
        elif self.rung == 9:
            task_outcome = 4.0 * yaw_score - 1.5 * speed
        elif self.rung == 11:
            task_outcome = 3.0 * heading_score
        elif self.rung == 12:
            task_outcome = 3.0 * stop_compliance + 40.0 * move_progress
        elif self.rung >= 19:
            task_outcome = 35.0 * goal_progress + 5.0 * goal_hit - 0.4 * goal_distance

        obstacle_clearance = torch.full_like(goal_distance, 2.0)
        if self.rung == 23:
            delta = q[:, None, :2] - self._obstacles[None, :, :2]
            obstacle_clearance = (torch.linalg.vector_norm(delta, dim=-1)
                                  - self._obstacles[None, :, 2]).amin(dim=-1)
            task_constraint = (task_constraint
                               - 4.0 * (0.12 - obstacle_clearance).clamp_min(0.0))
        if self.rung != 7:
            task_constraint = (task_constraint
                               - self._constraint_duals[0] * normalized_slip)
        task = task_outcome + task_constraint + task_scaffold
        metrics = {
            "ladder_pose_error": pose_error,
            "ladder_pose_score": pose_score,
            "ladder_height_error": height_error,
            "ladder_height_score": height_score,
            "ladder_yaw_error": yaw_error,
            "ladder_yaw_score": yaw_score,
            "ladder_heading_error": heading_error,
            "ladder_heading_score": heading_score,
            "ladder_goal_distance": goal_distance,
            "ladder_goal_progress": goal_progress,
            "ladder_goal_hit": goal_hit,
            "ladder_stop_speed": speed * stop_phase,
            "ladder_stop_score": stop_compliance,
            "ladder_move_progress": move_progress,
            "ladder_step_clock": step_clock,
            "ladder_swing_clearance": swing_clearance,
            "ladder_worst_swing_clearance": worst_swing_clearance,
            "ladder_step_action_score": step_action_score,
            "ladder_foot_activity": foot_activity,
            "ladder_foot_activity_mean": foot_activity_mean,
            "ladder_foot_activity_ema": foot_activity_ema,
            "ladder_safe_progress": (safe_progress if self.rung == 7
                                      else info["xprogress"]),
            "ladder_stance_slip_ratio": stance_slip_ratio,
            "ladder_obstacle_clearance": obstacle_clearance,
            "ladder_task_reward": task,
        }
        reward_components = {
            "ladder_task_outcome": task_outcome,
            "ladder_task_constraint": task_constraint,
            "ladder_task_scaffold": task_scaffold,
        }
        return task, metrics, reward_components

    def _advance_route(self, hit: torch.Tensor) -> None:
        if self.rung < 19:
            return
        count = len(self._routes)
        advance = hit.bool()
        self._route_index.copy_(torch.where(
            advance, torch.remainder(self._route_index + 1, count), self._route_index))
        self._goal.copy_(torch.where(
            advance[:, None], self._routes[self._route_index], self._goal))

    def step(self, action: torch.Tensor, alpha: float = 1.0, imit_anneal: float = 0.0):
        if action.shape != (self.nworld, self.act_dim):
            raise ValueError(f"ladder locomotion action must have shape "
                             f"({self.nworld}, {self.act_dim})")
        self._set_task_command(alpha)
        self._inject_disturbance()
        # Ladder objectives own their temporary scaffolds explicitly.  Letting
        # the base walker apply its reference-gait imitation schedule here would
        # silently reintroduce phase/joint choreography even when rung 7 is
        # launched without an action prior.  Optional fallback guidance is
        # handled by the trainer's visible, annealed action-prior mechanism.
        obs, reward, done, info = super().step(
            action, alpha=alpha, imit_anneal=0.0)
        # Rung 6 establishes physical stepping without prescribing an order;
        # every later locomotion skill inherits that same freedom.  The generic
        # walker reward otherwise smuggles in a diagonal contact clock, a fixed
        # airtime target, a nominal joint pose, and foot-clearance shaping.  They
        # are useful standalone walker scaffolds but are style specifications,
        # not ladder outcomes.  Remove them before applying the rung-owned
        # objective while leaving their zero-valued telemetry fields explicit.
        if self.rung >= 6:
            components = info.setdefault("reward_components", {})
            for name in ("gait_clock", "airtime", "pose_penalty", "clearance"):
                component = components.get(name)
                if component is not None:
                    reward = reward - component
                    components[name] = torch.zeros_like(component)
            if self.rung == 7:
                # At the deliberately gentle discovery command (0.10 m/s), the
                # generic Gaussian velocity score gives standing still more than
                # 90% credit. In the first outcome-only evaluation it supplied
                # 78.4% of all reward while direct progress supplied 5.4%, making
                # contact collapse optimal. Rung 7 already owns a target-scaled,
                # signed progress outcome, so remove this saturated proxy rather
                # than hand-increasing another coefficient.
                component = components.get("tracking")
                if component is not None:
                    reward = reward - component
                    components["tracking"] = torch.zeros_like(component)
        task_reward, metrics, task_components = self._task_reward(info, action)
        reward = reward + task_reward
        info.update(metrics)
        info.setdefault("reward_components", {}).update(task_components)
        info["gait_phase"] = (torch.remainder(
            self._task_t, GAIT_PERIOD_STEPS).to(torch.float32) / GAIT_PERIOD_STEPS)
        self._prev_goal_distance.copy_(metrics["ladder_goal_distance"])
        self._advance_route(metrics["ladder_goal_hit"])
        self._task_t.add_(1)
        self._constraint_age.add_(1)
        done_mask = done.bool()
        self._sample_task(done_mask, place_robot=True)
        # Pose and height targets change periodically without resetting dynamics.
        periodic = self.rung in (4, 5, 8, 9, 10, 11, 13, 16)
        resample = ((torch.remainder(self._task_t, 120) == 0) & ~done_mask
                    if periodic else torch.zeros_like(done_mask))
        self._sample_task(resample, place_robot=False)
        if self.rung == 19:
            with wp.ScopedDevice(self._wp_device):
                mjwp.forward(self._wm, self._wd)
            info["priv"] = self.privileged()
        self._update_lidar()
        return self.observe(), reward, done, info


class LadderCombatWarpEnv(LegAttackWarpEnv):
    """Shared-policy combat ladder with a per-rung opponent/damage objective."""

    architecture_task_dim = len(COMBAT_RUNG_NAMES)

    def __init__(self, nworld: int, rung: int, seed: int = 0,
                 device: str | None = None, episode_length: int | None = 800,
                 **kwargs):
        rung = int(rung)
        if rung not in COMBAT_RUNG_NAMES:
            raise ValueError(f"combat rung must be in [{COMBAT_RUNG_MIN}, {COMBAT_RUNG_MAX}]")
        self.rung = rung
        self.rung_name = COMBAT_RUNG_NAMES[rung]
        self._ladder_combat_ready = False
        super().__init__(nworld=nworld, seed=seed, device=device,
                         episode_length=episode_length,
                         task_cfg=kwargs.pop("task_cfg", LegAttackConfig()), **kwargs)
        self.obs_dim += self.architecture_task_dim
        self._ladder_combat_ready = True
        self._ladder_prev_distance = torch.zeros(nworld, device=self.device)
        self._ladder_prev_rod = torch.zeros(nworld, device=self.device)
        self._ladder_prev_taken = torch.zeros(nworld, device=self.device)
        self._reset_ladder_metrics(torch.ones(nworld, dtype=torch.bool, device=self.device))
        if rung in (24, 25):
            self.set_attack_enabled(False)

    @property
    def reward_semantics(self):
        return f"ladder:{self.rung:02d}:{self.rung_name}:v1"

    @property
    def observation_semantics(self):
        return "combat44+attack7+combat_task6:v1"

    def _combat_task(self) -> torch.Tensor:
        return F.one_hot(torch.full(
            (self.nworld,), self.rung - COMBAT_RUNG_MIN,
            device=self.device, dtype=torch.long),
            num_classes=self.architecture_task_dim).to(torch.float32)

    def observe(self) -> torch.Tensor:
        base = super().observe()
        if not self._ladder_combat_ready:
            return base
        return torch.cat((base, self._combat_task()), dim=-1)

    def observe_opponent(self) -> torch.Tensor:
        base = super().observe_opponent()
        if not self._ladder_combat_ready:
            return base
        return torch.cat((base, self._combat_task()), dim=-1)

    def _reset_ladder_metrics(self, mask: torch.Tensor) -> None:
        i = self.layer.idx
        distance = torch.linalg.vector_norm(
            (self.xpos[:, i.Bt] - self.xpos[:, i.At])[:, :2], dim=-1)
        rod = wp.to_torch(self.layer.dealt_rod)
        taken = wp.to_torch(self.layer.taken_leg) + wp.to_torch(self.layer.taken_rod)
        self._ladder_prev_distance.copy_(torch.where(mask, distance, self._ladder_prev_distance))
        self._ladder_prev_rod.copy_(torch.where(mask, rod, self._ladder_prev_rod))
        self._ladder_prev_taken.copy_(torch.where(mask, taken, self._ladder_prev_taken))

    def reset(self, seed: int | None = None) -> torch.Tensor:
        obs = super().reset(seed)
        if hasattr(self, "_ladder_prev_distance"):
            self._reset_ladder_metrics(torch.ones(
                self.nworld, dtype=torch.bool, device=self.device))
        return obs

    def _script_evader(self) -> None:
        if self.rung != 27 or self._opponent is not None:
            return
        i = self.layer.idx
        delta = self.xpos[:, i.Bt, :2] - self.xpos[:, i.At, :2]
        direction = delta / torch.linalg.vector_norm(delta, dim=-1, keepdim=True).clamp_min(1e-5)
        self.qvel[:, i.BrD:i.BrD + 2].copy_(0.28 * direction)

    def step(self, action: torch.Tensor, alpha: float = 1.0, imit_anneal: float = 0.0):
        self._script_evader()
        obs, reward, done, info = super().step(action, alpha=alpha, imit_anneal=imit_anneal)
        task = self._combat_task()
        info["terminal_obs"] = torch.cat((info["terminal_obs"], task), dim=-1)
        info["terminal_opponent_obs"] = torch.cat(
            (info["terminal_opponent_obs"], task), dim=-1)
        i = self.layer.idx
        distance = torch.linalg.vector_norm(
            (self.xpos[:, i.Bt] - self.xpos[:, i.At])[:, :2], dim=-1)
        approach = self._ladder_prev_distance - distance
        rod = wp.to_torch(self.layer.dealt_rod)
        taken = wp.to_torch(self.layer.taken_leg) + wp.to_torch(self.layer.taken_rod)
        rod_delta = (rod - self._ladder_prev_rod).clamp_min(0.0)
        taken_delta = (taken - self._ladder_prev_taken).clamp_min(0.0)
        if self.rung == 24:
            addition = 25.0 * approach - 0.2 * distance
        elif self.rung == 25:
            addition = 20.0 * rod_delta + 8.0 * approach
        elif self.rung == 26:
            addition = torch.zeros_like(reward)
        elif self.rung == 27:
            addition = 18.0 * approach - 0.1 * distance
        else:
            addition = 18.0 * rod_delta - 16.0 * taken_delta + 5.0 * approach
        reward = reward + addition
        info.setdefault("reward_components", {})["ladder_combat_task"] = addition
        info.update(
            ladder_approach=approach,
            ladder_target_distance=distance,
            ladder_rod_hit=rod_delta,
            ladder_taken=taken_delta,
            ladder_combat_margin=rod_delta - taken_delta,
            ladder_task_reward=info["attack_task_reward"] + addition,
        )
        done_mask = done.bool()
        self._ladder_prev_distance.copy_(distance)
        self._ladder_prev_rod.copy_(rod)
        self._ladder_prev_taken.copy_(taken)
        self._reset_ladder_metrics(done_mask)
        return obs, reward, done, info


class UniversalControlWarpEnv:
    """One actor contract spanning locomotion, combat, and morphology tasks.

    The underlying physics environments retain their natural observations and
    actuators.  This adapter removes family-local task one-hots, pads physical
    features into a stable namespace, appends a 14-channel actuator-availability
    mask, and finishes with one global 31-rung command.  The actor therefore has
    one 256-value input and 14-value output throughout the entire ladder.
    """

    architecture_task_dim = UNIVERSAL_TASK_CHANNELS
    obs_dim = UNIVERSAL_OBS_DIM
    priv_dim = UNIVERSAL_PRIV_DIM
    act_dim = UNIVERSAL_ACT_DIM
    action_semantics = "universal14:A12_servos+2_optional_pneumatic:v1"
    observation_semantics = (
        "universal256:physical211+actuator_mask14+task31:v1")

    def __init__(self, nworld: int, rung: int, seed: int = 0,
                 device: str | None = None, episode_length: int | None = 800,
                 **kwargs):
        self.rung = int(rung)
        if (COMBAT_RUNG_MIN <= self.rung <= COMBAT_RUNG_MAX
                and kwargs.get("power_model", "off") != "off"):
            # The fused combat layer owns its own actuation kernel and does not
            # yet model the shared bus; dropping silently would misreport the
            # physics, so say it once and keep the combat rung on v1 semantics.
            print(f"universal rung {self.rung}: shared-bus power model is not "
                  "yet implemented in the fused combat layer; combat physics "
                  "stays on v1 action semantics", flush=True)
            kwargs = {key: value for key, value in kwargs.items()
                      if key not in ("power_model", "power_model_params")}
        if LOCOMOTION_RUNG_MIN <= self.rung <= LOCOMOTION_RUNG_MAX:
            self.domain = "locomotion"
            self.env = LadderLocomotionWarpEnv(
                nworld, rung=self.rung, seed=seed, device=device,
                episode_length=episode_length, **kwargs)
            self._source_task_dim = LOCOMOTION_TASK_CHANNELS
            available = 12
        elif COMBAT_RUNG_MIN <= self.rung <= COMBAT_RUNG_MAX:
            self.domain = "combat"
            self.env = LadderCombatWarpEnv(
                nworld, rung=self.rung, seed=seed, device=device,
                episode_length=episode_length, **kwargs)
            self._source_task_dim = len(COMBAT_RUNG_NAMES)
            available = 14
        elif self.rung == 30:
            self.domain = "codesign"
            self.env = DesignEnsembleWarpEnv(
                nworld, seed=seed, device=device, episode_length=episode_length,
                **kwargs)
            # The trailing three values are physical design parameters, not a
            # task selector, and remain visible in the padded physical block.
            self._source_task_dim = 0
            available = 12
        else:
            raise ValueError("universal control supports ladder rungs 2 through 30")
        self.nworld = self.env.nworld
        self.device = self.env.device
        self._dt = self.env._dt
        self._episode_length = episode_length
        self._gen = self.env._gen
        self.gait_loaded = self.env.gait_loaded
        self.model_hash = hashlib.sha256(
            f"universal-v1|{self.env.model_hash}".encode()).hexdigest()[:16]
        self.policy_action_mask = torch.zeros(
            UNIVERSAL_ACT_DIM, dtype=torch.float32, device=self.device)
        self.policy_action_mask[:available] = 1.0
        if str(getattr(self.env, "action_semantics", "")).endswith(
                "shared_bus_v2"):
            self.action_semantics = f"{self.action_semantics}+shared_bus_v2"
        # The legacy codesign environment appends three search coordinates to
        # its physical observation. Predictive policies explicitly mask these
        # channels and condition on compiled physical tokens instead.
        self.morphology_parameter_observation_indices = (
            (50, 51, 52) if self.domain == "codesign" else ())

    def __getattr__(self, name):
        if name == "env":
            raise AttributeError(name)
        return getattr(self.env, name)

    @property
    def reward_semantics(self):
        return f"universal:{self.rung:02d}:{self.env.reward_semantics}"

    def _observation(self, value: torch.Tensor) -> torch.Tensor:
        physical = (value[..., :-self._source_task_dim]
                    if self._source_task_dim else value)
        if physical.shape[-1] > 211:
            raise ValueError(
                f"universal physical observation overflow: {physical.shape[-1]} > 211")
        shape = (*physical.shape[:-1], UNIVERSAL_PHYSICAL_DIM)
        padded = physical.new_zeros(shape)
        padded[..., :physical.shape[-1]] = physical
        padded[..., 211:225] = self.policy_action_mask
        task = physical.new_zeros((*physical.shape[:-1], UNIVERSAL_TASK_CHANNELS))
        task[..., self.rung - 1] = 1.0
        return torch.cat((padded, task), dim=-1)

    def _privileged(self, value: torch.Tensor) -> torch.Tensor:
        if value.shape[-1] > UNIVERSAL_PRIV_DIM:
            raise ValueError(
                f"universal privileged observation overflow: {value.shape[-1]}")
        padded = value.new_zeros((*value.shape[:-1], UNIVERSAL_PRIV_DIM))
        padded[..., :value.shape[-1]] = value
        return padded

    def observe(self) -> torch.Tensor:
        return self._observation(self.env.observe())

    def privileged(self) -> torch.Tensor:
        return self._privileged(self.env.privileged())

    def policy_mean_prior(self, raw_obs: torch.Tensor,
                          base_mean: torch.Tensor | None = None,
                          transfer_action: torch.Tensor | None = None):
        if not hasattr(self.env, "policy_mean_prior"):
            return None
        local_base = None if base_mean is None else base_mean[:, :self.env.act_dim]
        local_transfer = (None if transfer_action is None
                          else transfer_action[:, :self.env.act_dim])
        result = self.env.policy_mean_prior(
            raw_obs, local_base, transfer_action=local_transfer)
        if result is None:
            return None
        target, mask = result
        padded_target = raw_obs.new_zeros((len(raw_obs), UNIVERSAL_ACT_DIM))
        padded_mask = torch.zeros_like(padded_target)
        padded_target[:, :target.shape[-1]] = target
        padded_mask[:, :mask.shape[-1]] = mask
        return padded_target, padded_mask

    def set_opponent(self, policy) -> None:
        if self.domain != "combat":
            raise ValueError("opponents are only valid in the combat domain")

        def universal_opponent(local_obs):
            return policy(self._observation(local_obs))[:, :self.env.act_dim]

        reset = getattr(policy, "reset", None)
        if callable(reset):
            # the wrapped env drives episode-boundary state resets through this
            universal_opponent.reset = reset
        self.env.set_opponent(universal_opponent)

    def reset(self, *args, **kwargs) -> torch.Tensor:
        return self._observation(self.env.reset(*args, **kwargs))

    def step(self, action: torch.Tensor, alpha: float = 1.0,
             imit_anneal: float = 0.0):
        if action.shape != (self.nworld, UNIVERSAL_ACT_DIM):
            raise ValueError(
                f"universal action must have shape ({self.nworld}, {UNIVERSAL_ACT_DIM})")
        local = action if self.domain == "combat" else action[:, :12]
        obs, reward, done, info = self.env.step(
            local, alpha=alpha, imit_anneal=imit_anneal)
        info["terminal_obs"] = self._observation(info["terminal_obs"])
        info["terminal_priv"] = self._privileged(info["terminal_priv"])
        if "terminal_opponent_obs" in info:
            info["terminal_opponent_obs"] = self._observation(
                info["terminal_opponent_obs"])
        info["priv"] = self._privileged(info["priv"])
        return self._observation(obs), reward, done, info


UNIVERSAL_COMMAND_CHANNELS = 31
# Command-block slices (within the trailing 31-channel conditioning block).
COMMAND_VELOCITY = slice(0, 3)
COMMAND_VELOCITY_ACTIVE = 3
COMMAND_HEADING = slice(4, 6)
COMMAND_HEADING_ACTIVE = 6
COMMAND_HEIGHT = 7
COMMAND_POSE = slice(8, 20)
COMMAND_GOAL = slice(20, 22)
COMMAND_GOAL_ACTIVE = 22
COMMAND_LEG = slice(23, 27)
COMMAND_ENGAGE = 27
COMMAND_CADENCE = 28
COMMAND_CADENCE_ACTIVE = 29

# Static activity table: the executable form of the per-rung command
# signatures in notes/universal-command-contract.md.  Rungs sharing a
# signature assert that identical commands demand identical behavior;
# everything else about a rung is environment distribution.
_COMMAND_ACTIVITY = {
    # rung: (velocity, heading, goal, cadence)
    **{rung: (1.0, 0.0, 0.0, 0.0) for rung in (2, 3, 5, 7, 8, 9, 10, 12,
                                               13, 14, 15, 16, 17, 18, 30)},
    4: (0.0, 0.0, 0.0, 0.0),
    6: (1.0, 0.0, 0.0, 1.0),
    11: (1.0, 1.0, 0.0, 0.0),
    **{rung: (0.0, 0.0, 1.0, 0.0) for rung in (19, 20, 21, 22, 23)},
    **{rung: (0.0, 0.0, 1.0, 0.0) for rung in (24, 25, 27, 28, 29)},
    26: (0.0, 0.0, 0.0, 0.0),
}

# Locomotion physical-namespace indices (walker base is 50 wide).
_LOCO_VELOCITY = slice(47, 50)
_LOCO_POSE = slice(50, 62)
_LOCO_HEIGHT = 62
_LOCO_GOAL = slice(63, 65)
_LOCO_HEADING = slice(65, 67)
# Combat physical-namespace indices (44 fused channels + attack command 7).
_COMBAT_OPPONENT = slice(38, 40)
_COMBAT_ATTACK_ACTIVE = 44
_COMBAT_LEG = slice(45, 49)


class UniversalCommandWarpEnv(UniversalControlWarpEnv):
    """Rung-invisible universal contract: commands carry all task semantics.

    Identical to :class:`UniversalControlWarpEnv` except the trailing 31
    conditioning channels are an explicit command block instead of a rung
    one-hot.  Every command value is derived from the underlying environment's
    observation tensor, so terminal observations keep the pre-reset commands
    they were experienced under.  The rung number still selects the wrapped
    environment, reward, and gates — it is simply no longer observable.
    """

    observation_semantics = (
        "universal256:physical211+actuator_mask14+command31:v2")
    task_conditioning_kind = "command"

    def __init__(self, nworld: int, rung: int, **kwargs):
        super().__init__(nworld, rung, **kwargs)
        if int(rung) not in _COMMAND_ACTIVITY:
            raise ValueError(
                f"rung {rung} has no command signature; see "
                "notes/universal-command-contract.md")
        self.model_hash = hashlib.sha256(
            f"universal-command-v2|{self.env.model_hash}".encode()).hexdigest()[:16]

    def _observation(self, value: torch.Tensor) -> torch.Tensor:
        with_task = super()._observation(value)
        physical = with_task[..., :UNIVERSAL_PHYSICAL_DIM]
        command = with_task.new_zeros(
            (*with_task.shape[:-1], UNIVERSAL_COMMAND_CHANNELS))
        velocity_active, heading_active, goal_active, cadence_active = \
            _COMMAND_ACTIVITY[self.rung]
        if self.domain in ("locomotion", "codesign"):
            command[..., COMMAND_VELOCITY] = physical[..., _LOCO_VELOCITY]
            command[..., COMMAND_HEIGHT] = (
                physical[..., _LOCO_HEIGHT] if self.domain == "locomotion"
                else 0.0)
            if self.domain == "locomotion":
                command[..., COMMAND_POSE] = physical[..., _LOCO_POSE]
                command[..., COMMAND_GOAL] = (
                    goal_active * physical[..., _LOCO_GOAL])
                # Heading channels double as the gait phase clock on movement
                # rungs; only a rung that commands heading may publish them.
                command[..., COMMAND_HEADING] = (
                    heading_active * physical[..., _LOCO_HEADING])
        else:
            command[..., COMMAND_GOAL] = (
                goal_active * physical[..., _COMBAT_OPPONENT])
            engage = physical[..., _COMBAT_ATTACK_ACTIVE]
            command[..., COMMAND_ENGAGE] = engage
            command[..., COMMAND_LEG] = (
                engage.unsqueeze(-1) * physical[..., _COMBAT_LEG])
        command[..., COMMAND_VELOCITY_ACTIVE] = velocity_active
        command[..., COMMAND_HEADING_ACTIVE] = heading_active
        command[..., COMMAND_GOAL_ACTIVE] = goal_active
        command[..., COMMAND_CADENCE] = cadence_active
        command[..., COMMAND_CADENCE_ACTIVE] = cadence_active
        return torch.cat(
            (with_task[..., :UNIVERSAL_PHYSICAL_DIM], command), dim=-1)


def main() -> None:
    import argparse
    import time

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--rung", type=int, default=23)
    parser.add_argument("--nworld", type=int, default=8)
    parser.add_argument("--steps", type=int, default=100)
    parser.add_argument("--warmup", type=int, default=5)
    parser.add_argument("--device", default=None)
    args = parser.parse_args()
    cls = LadderLocomotionWarpEnv if args.rung <= LOCOMOTION_RUNG_MAX else LadderCombatWarpEnv
    env = cls(args.nworld, rung=args.rung, device=args.device, episode_length=800)
    action = torch.zeros((env.nworld, env.act_dim), device=env.device)
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
    print(f"RESULT bench=ladder_warp_env rung={args.rung} nworld={args.nworld} "
          f"steps={args.steps} device={env.device} "
          f"env_steps_per_s={args.nworld * args.steps / wall:.1f}", flush=True)


if __name__ == "__main__":
    main()
