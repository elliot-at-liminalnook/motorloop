# SPDX-License-Identifier: MIT
"""Morphology tokens and locally stabilized predictive-control utilities.

The reactive policy should not need a hand-authored vector for every new robot.
Instead, a MuJoCo model is serialized into a padded sequence of body, joint, and
actuator tokens.  The trajectory decoder consumes the resulting morphology
embedding and predicts physical outcomes in a yaw-stabilized local world frame.
"""
from __future__ import annotations

from dataclasses import dataclass
import math

import mujoco
import numpy as np
import torch
from torch import nn
import torch.nn.functional as F


MORPH_NUMERIC_DIM = 20
MORPH_TOKEN_PAD = 0
MORPH_TOKEN_BODY = 1
MORPH_TOKEN_JOINT = 2
MORPH_TOKEN_ACTUATOR = 3
MORPH_TOKEN_TYPES = 4

# Raw physical snapshot: root xyz, root quaternion, world spatial velocity,
# four world foot positions, floor-support flags, task-interaction contact flags,
# padded actuator force, up, height.  Support and interaction contact are
# deliberately separate: a kick should not have to pretend that hitting an
# opponent is the same physical event as standing on the floor.
TRAJECTORY_RAW_DIM = 3 + 4 + 6 + 12 + 4 + 4 + 14 + 1 + 1
TRAJECTORY_RAW_SUPPORT = slice(25, 29)
TRAJECTORY_RAW_INTERACTION = slice(29, 33)
TRAJECTORY_RAW_EFFORT = slice(33, 47)
# Decoder target: local root delta, continuous 6-D orientation, local velocity,
# local foot positions, support, interaction contact, signed-log actuator force,
# and fall risk.
TRAJECTORY_TARGET_DIM = 3 + 6 + 6 + 12 + 4 + 4 + 14 + 1


@dataclass(frozen=True)
class TrajectorySlices:
    root_delta: slice = slice(0, 3)
    rotation_6d: slice = slice(3, 9)
    velocity: slice = slice(9, 15)
    feet: slice = slice(15, 27)
    contact: slice = slice(27, 31)
    interaction: slice = slice(31, 35)
    effort: slice = slice(35, 49)
    fall: slice = slice(49, 50)


TRAJECTORY_SLICES = TrajectorySlices()


@dataclass
class InteractionTrajectoryTarget:
    """Broad, masked physical intent for a short future trajectory.

    A target can request root motion, an end-effector region, a contact event,
    and aggregate support without specifying a gait or joint trajectory.  Every
    field has an explicit mask, so a task states only what physically matters.
    """

    root_delta: torch.Tensor
    root_delta_mask: torch.Tensor
    velocity: torch.Tensor
    velocity_mask: torch.Tensor
    effector_position: torch.Tensor
    effector_mask: torch.Tensor
    effector_radius: torch.Tensor
    effector_direction: torch.Tensor
    effector_speed_min: torch.Tensor
    interaction_event: torch.Tensor
    interaction_event_mask: torch.Tensor
    support_leg_mask: torch.Tensor
    min_support: torch.Tensor
    task_weight: torch.Tensor
    dt: float

    @classmethod
    def empty(cls, horizon: int, batch: int, *, device, dtype,
              dt: float) -> "InteractionTrajectoryTarget":
        h, b = int(horizon), int(batch)
        z_h3 = torch.zeros((h, b, 3), device=device, dtype=dtype)
        z_h6 = torch.zeros((h, b, 6), device=device, dtype=dtype)
        z_b4 = torch.zeros((b, 4), device=device, dtype=dtype)
        return cls(
            root_delta=z_h3,
            root_delta_mask=torch.zeros_like(z_h3),
            velocity=z_h6,
            velocity_mask=torch.zeros_like(z_h6),
            effector_position=torch.zeros((b, 4, 3), device=device, dtype=dtype),
            effector_mask=z_b4.clone(),
            effector_radius=torch.full_like(z_b4, 0.10),
            effector_direction=torch.zeros((b, 4, 3), device=device, dtype=dtype),
            effector_speed_min=z_b4.clone(),
            interaction_event=z_b4.clone(),
            interaction_event_mask=z_b4.clone(),
            support_leg_mask=z_b4.clone(),
            min_support=torch.zeros(b, device=device, dtype=dtype),
            task_weight=torch.ones(b, device=device, dtype=dtype),
            dt=float(dt),
        )

    @classmethod
    def cat(cls, values: list["InteractionTrajectoryTarget"]
            ) -> "InteractionTrajectoryTarget":
        if not values:
            raise ValueError("cannot concatenate an empty interaction-target list")
        dt = values[0].dt
        if any(abs(value.dt - dt) > 1.0e-12 for value in values[1:]):
            raise ValueError("interaction targets must share one control timestep")
        time_fields = {"root_delta", "root_delta_mask", "velocity", "velocity_mask"}
        kwargs = {}
        for name in cls.__dataclass_fields__:
            if name == "dt":
                continue
            kwargs[name] = torch.cat(
                [getattr(value, name) for value in values],
                dim=1 if name in time_fields else 0)
        return cls(**kwargs, dt=dt)


def _safe_range(values: np.ndarray) -> np.ndarray:
    values = np.asarray(values, dtype=np.float32)
    return np.nan_to_num(values, nan=0.0, posinf=1.0e4, neginf=-1.0e4)


def model_morphology_tokens(model: mujoco.MjModel, *,
                            actuator_wfree: np.ndarray | None = None,
                            actuator_kp: np.ndarray | None = None,
                            max_tokens: int | None = None,
                            device: torch.device | str = "cpu",
                            batch: int = 1) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Serialize actual compiled model parameters into typed numeric tokens.

    Values are deliberately physical rather than design-code coordinates.  A
    changed mass, link offset, joint limit, gear, or motor speed is visible even
    when it arose from an unseen co-design parameterization.
    """
    rows: list[np.ndarray] = []
    kinds: list[int] = []

    for body in range(1, model.nbody):  # omit MuJoCo's inertial world body
        row = np.zeros(MORPH_NUMERIC_DIM, dtype=np.float32)
        row[0:3] = model.body_pos[body]
        row[3:7] = model.body_quat[body]
        row[7] = model.body_mass[body]
        row[8:11] = model.body_inertia[body]
        row[11] = float(model.body_parentid[body]) / max(model.nbody - 1, 1)
        row[12] = float(model.body_dofnum[body])
        row[13] = float(model.body_jntnum[body])
        rows.append(_safe_range(row)); kinds.append(MORPH_TOKEN_BODY)

    for joint in range(model.njnt):
        row = np.zeros(MORPH_NUMERIC_DIM, dtype=np.float32)
        row[0:3] = model.jnt_pos[joint]
        row[3:6] = model.jnt_axis[joint]
        row[6:8] = model.jnt_range[joint]
        dof = int(model.jnt_dofadr[joint])
        if dof >= 0:
            row[8] = model.dof_damping[dof]
            row[9] = model.dof_armature[dof]
            row[10] = model.dof_frictionloss[dof]
        row[11] = model.jnt_stiffness[joint]
        row[12] = float(model.jnt_type[joint])
        row[13] = float(model.jnt_limited[joint])
        row[14] = float(model.jnt_bodyid[joint]) / max(model.nbody - 1, 1)
        rows.append(_safe_range(row)); kinds.append(MORPH_TOKEN_JOINT)

    wfree = np.zeros(model.nu, dtype=np.float32) if actuator_wfree is None \
        else np.asarray(actuator_wfree, dtype=np.float32)
    kp = np.zeros(model.nu, dtype=np.float32) if actuator_kp is None \
        else np.asarray(actuator_kp, dtype=np.float32)
    for actuator in range(model.nu):
        row = np.zeros(MORPH_NUMERIC_DIM, dtype=np.float32)
        row[0:6] = model.actuator_gear[actuator, :6]
        row[6:8] = model.actuator_ctrlrange[actuator]
        row[8:10] = model.actuator_forcerange[actuator]
        row[10] = wfree[actuator] if actuator < len(wfree) else 0.0
        row[11] = kp[actuator] if actuator < len(kp) else 0.0
        row[12] = float(model.actuator_trntype[actuator])
        row[13] = float(model.actuator_trnid[actuator, 0]) / max(model.njnt - 1, 1)
        row[14] = float(model.actuator_ctrllimited[actuator])
        row[15] = float(model.actuator_forcelimited[actuator])
        rows.append(_safe_range(row)); kinds.append(MORPH_TOKEN_ACTUATOR)

    count = len(rows)
    width = count if max_tokens is None else int(max_tokens)
    if width < count:
        raise ValueError(f"max_tokens={width} cannot hold {count} morphology tokens")
    numeric = np.zeros((width, MORPH_NUMERIC_DIM), dtype=np.float32)
    token_types = np.zeros(width, dtype=np.int64)
    mask = np.zeros(width, dtype=np.bool_)
    numeric[:count] = np.stack(rows) if rows else 0.0
    token_types[:count] = np.asarray(kinds, dtype=np.int64)
    mask[:count] = True
    numeric_t = torch.as_tensor(numeric, device=device).unsqueeze(0).expand(batch, -1, -1)
    types_t = torch.as_tensor(token_types, device=device).unsqueeze(0).expand(batch, -1)
    mask_t = torch.as_tensor(mask, device=device).unsqueeze(0).expand(batch, -1)
    return numeric_t, types_t, mask_t


class MorphologyTokenEncoder(nn.Module):
    """Permutation-aware typed token encoder with a learned pooling query."""

    def __init__(self, width: int, layers: int = 2, heads: int = 4):
        super().__init__()
        if width % heads:
            raise ValueError("morphology encoder width must be divisible by attention heads")
        self.numeric = nn.Sequential(
            nn.LayerNorm(MORPH_NUMERIC_DIM), nn.Linear(MORPH_NUMERIC_DIM, width), nn.SiLU())
        self.kind = nn.Embedding(MORPH_TOKEN_TYPES, width, padding_idx=MORPH_TOKEN_PAD)
        layer = nn.TransformerEncoderLayer(
            width, heads, dim_feedforward=2 * width, dropout=0.0,
            activation="gelu", batch_first=True, norm_first=True)
        self.encoder = nn.TransformerEncoder(layer, layers, nn.LayerNorm(width))
        self.query = nn.Parameter(torch.zeros(width))
        nn.init.normal_(self.query, std=0.02)

    def forward(self, numeric: torch.Tensor, token_types: torch.Tensor,
                mask: torch.Tensor) -> torch.Tensor:
        if numeric.ndim != 3 or token_types.shape != numeric.shape[:2] \
                or mask.shape != numeric.shape[:2]:
            raise ValueError("morphology tensors must be [batch,tokens,numeric], [batch,tokens]")
        x = self.numeric(numeric) + self.kind(token_types)
        x = self.encoder(x, src_key_padding_mask=~mask.bool())
        score = (x * self.query).sum(-1) / self.query.numel() ** 0.5
        score = score.masked_fill(~mask.bool(), -torch.inf)
        weight = torch.softmax(score, dim=-1)
        return (x * weight.unsqueeze(-1)).sum(1)


def quaternion_to_matrix(quaternion: torch.Tensor) -> torch.Tensor:
    """Convert scalar-first unit quaternions to rotation matrices."""
    q = F.normalize(quaternion, dim=-1)
    w, x, y, z = q.unbind(-1)
    return torch.stack((
        1 - 2 * (y*y + z*z), 2 * (x*y - z*w), 2 * (x*z + y*w),
        2 * (x*y + z*w), 1 - 2 * (x*x + z*z), 2 * (y*z - x*w),
        2 * (x*z - y*w), 2 * (y*z + x*w), 1 - 2 * (x*x + y*y),
    ), dim=-1).reshape(*q.shape[:-1], 3, 3)


def stabilized_trajectory_target(anchor: torch.Tensor, future: torch.Tensor,
                                 *, fall_height: float = 0.16,
                                 fall_up: float = 0.35) -> torch.Tensor:
    """Express future physical state in anchor-position/anchor-yaw coordinates."""
    if anchor.shape[-1] != TRAJECTORY_RAW_DIM or future.shape[-1] != TRAJECTORY_RAW_DIM:
        raise ValueError("unexpected raw trajectory-state width")
    anchor_pos, anchor_q = anchor[..., 0:3], anchor[..., 3:7]
    future_pos, future_q = future[..., 0:3], future[..., 3:7]
    aq = anchor_q
    yaw = torch.atan2(2 * (aq[..., 0] * aq[..., 3] + aq[..., 1] * aq[..., 2]),
                      1 - 2 * (aq[..., 2].square() + aq[..., 3].square()))
    c, s = torch.cos(yaw), torch.sin(yaw)
    local_rotation = torch.stack((c, s, torch.zeros_like(c),
                                  -s, c, torch.zeros_like(c),
                                  torch.zeros_like(c), torch.zeros_like(c),
                                  torch.ones_like(c)), dim=-1).reshape(*c.shape, 3, 3)

    def rotate(value: torch.Tensor) -> torch.Tensor:
        return torch.matmul(local_rotation, value.unsqueeze(-1)).squeeze(-1)

    root_delta = rotate(future_pos - anchor_pos)
    future_rot = torch.matmul(local_rotation, quaternion_to_matrix(future_q))
    rotation_6d = future_rot[..., :, :2].transpose(-1, -2).reshape(*future.shape[:-1], 6)
    velocity = torch.cat((rotate(future[..., 7:10]), rotate(future[..., 10:13])), -1)
    feet_world = future[..., 13:25].reshape(*future.shape[:-1], 4, 3)
    feet_local = torch.matmul(
        local_rotation.unsqueeze(-3),
        (feet_world - anchor_pos.unsqueeze(-2)).unsqueeze(-1)).squeeze(-1).reshape(
            *future.shape[:-1], 12)
    contact = future[..., TRAJECTORY_RAW_SUPPORT]
    interaction = future[..., TRAJECTORY_RAW_INTERACTION]
    raw_effort = future[..., TRAJECTORY_RAW_EFFORT]
    effort = torch.sign(raw_effort) * torch.log1p(raw_effort.abs())
    up, height = future[..., 47:48], future[..., 48:49]
    fall = ((up < fall_up) | (height < fall_height)).to(future.dtype)
    return torch.cat((root_delta, rotation_6d, velocity, feet_local,
                      contact, interaction, effort, fall), dim=-1)


def trajectory_prediction_loss(prediction: torch.Tensor, target: torch.Tensor,
                               valid: torch.Tensor | None = None) -> tuple[torch.Tensor, dict]:
    """Balanced multi-head self-supervision; returns scalar and diagnostics."""
    s = TRAJECTORY_SLICES
    weight = torch.ones(prediction.shape[:-1], device=prediction.device,
                        dtype=prediction.dtype) if valid is None else valid.to(prediction.dtype)
    denominator = weight.sum().clamp_min(1.0)

    def mse(part: slice) -> torch.Tensor:
        per = (prediction[..., part] - target[..., part]).square().mean(-1)
        return (per * weight).sum() / denominator

    components = {
        "body_position": mse(s.root_delta),
        "body_rotation": mse(s.rotation_6d),
        "body_velocity": mse(s.velocity),
        "feet": mse(s.feet),
        "contact": (F.binary_cross_entropy_with_logits(
            prediction[..., s.contact], target[..., s.contact], reduction="none").mean(-1)
            * weight).sum() / denominator,
        "interaction": (F.binary_cross_entropy_with_logits(
            prediction[..., s.interaction], target[..., s.interaction],
            reduction="none").mean(-1) * weight).sum() / denominator,
        "effort": F.smooth_l1_loss(
            prediction[..., s.effort], target[..., s.effort], reduction="none").mean(-1)
            .mul(weight).sum() / denominator,
        "fall": (F.binary_cross_entropy_with_logits(
            prediction[..., s.fall], target[..., s.fall], reduction="none").squeeze(-1)
            * weight).sum() / denominator,
    }
    loss = (components["body_position"] + 0.5 * components["body_rotation"]
            + 0.25 * components["body_velocity"] + components["feet"]
            + 0.5 * components["contact"] + 0.5 * components["interaction"]
            + 0.1 * components["effort"]
            + components["fall"])
    return loss, components


def trajectory_calibration_metrics(prediction: torch.Tensor, target: torch.Tensor,
                                   valid: torch.Tensor | None = None) -> dict[str, torch.Tensor]:
    """Dimensionless held-out errors used to decide planning authority.

    Unlike the auxiliary training objective, every continuous head is divided
    by a physical error scale and every binary head is a Brier error.  The
    resulting number therefore has a stable meaning across tasks and reward
    changes instead of inheriting arbitrary loss weights.
    """
    s = TRAJECTORY_SLICES
    weight = torch.ones(prediction.shape[:-1], device=prediction.device,
                        dtype=prediction.dtype) if valid is None else valid.to(prediction.dtype)
    denominator = weight.sum().clamp_min(1.0)

    def nrmse(part: slice, scale: float) -> torch.Tensor:
        per = ((prediction[..., part] - target[..., part]) / float(scale)).square().mean(-1)
        return torch.sqrt((per * weight).sum() / denominator)

    def brier(part: slice) -> torch.Tensor:
        per = (torch.sigmoid(prediction[..., part]) - target[..., part]).square().mean(-1)
        return (per * weight).sum() / denominator

    metrics = {
        "body_position": nrmse(s.root_delta, 0.25),
        "body_rotation": nrmse(s.rotation_6d, 1.0),
        "body_velocity": nrmse(s.velocity, 1.0),
        "feet": nrmse(s.feet, 0.25),
        "contact_brier": brier(s.contact),
        "interaction_brier": brier(s.interaction),
        "effort": nrmse(s.effort, 3.0),
        "fall_brier": brier(s.fall),
    }
    metrics["overall"] = torch.stack(tuple(metrics.values())).mean()
    return metrics


def interaction_trajectory_cost(prediction: torch.Tensor,
                                target: InteractionTrajectoryTarget
                                ) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    """Compare a predicted future with broad task-space physical intent."""
    if prediction.shape[:2] != target.root_delta.shape[:2]:
        raise ValueError("prediction and interaction target must share [horizon,batch]")
    s = TRAJECTORY_SLICES
    per_world_weight = target.task_weight.clamp_min(0.0)
    weight_sum = per_world_weight.sum().clamp_min(1.0)

    def masked_error(predicted: torch.Tensor, desired: torch.Tensor,
                     mask: torch.Tensor, scale: float) -> torch.Tensor:
        error = F.smooth_l1_loss(
            predicted / scale, desired / scale, reduction="none") * mask
        per_world = error.sum(dim=(0, 2)) / mask.sum(dim=(0, 2)).clamp_min(1.0)
        return (per_world * per_world_weight).sum() / weight_sum

    root = masked_error(
        prediction[..., s.root_delta], target.root_delta,
        target.root_delta_mask, 0.25)
    velocity = masked_error(
        prediction[..., s.velocity], target.velocity,
        target.velocity_mask, 1.0)

    feet = prediction[..., s.feet].reshape(*prediction.shape[:2], 4, 3)
    distance = torch.linalg.vector_norm(
        feet - target.effector_position.unsqueeze(0), dim=-1)
    # A soft minimum lets any timing within the horizon satisfy the event while
    # retaining gradients for near-miss frames.  No fixed strike phase is imposed.
    temperature = 0.04
    closest = -temperature * torch.logsumexp(-distance / temperature, dim=0)
    reach_debt = ((closest - target.effector_radius).clamp_min(0.0) / 0.25).square()
    reach = (reach_debt * target.effector_mask * per_world_weight[:, None]).sum() \
        / (target.effector_mask * per_world_weight[:, None]).sum().clamp_min(1.0)

    if prediction.shape[0] > 1:
        foot_velocity = (feet[1:] - feet[:-1]) / max(target.dt, 1.0e-6)
        direction = F.normalize(target.effector_direction, dim=-1, eps=1.0e-6)
        directed_speed = (foot_velocity * direction.unsqueeze(0)).sum(-1).amax(dim=0)
    else:
        directed_speed = torch.zeros_like(target.effector_speed_min)
    speed_debt = ((target.effector_speed_min - directed_speed).clamp_min(0.0) / 1.0).square()
    speed = (speed_debt * target.effector_mask * per_world_weight[:, None]).sum() \
        / (target.effector_mask * per_world_weight[:, None]).sum().clamp_min(1.0)

    interaction_probability = torch.sigmoid(prediction[..., s.interaction])
    any_interaction = 1.0 - torch.prod(1.0 - interaction_probability.clamp(0.0, 1.0), dim=0)
    event_error = F.binary_cross_entropy(
        any_interaction.clamp(1.0e-6, 1.0 - 1.0e-6),
        target.interaction_event, reduction="none")
    event = (event_error * target.interaction_event_mask
             * per_world_weight[:, None]).sum() / (
                 target.interaction_event_mask * per_world_weight[:, None]
             ).sum().clamp_min(1.0)

    support_probability = torch.sigmoid(prediction[..., s.contact])
    support_count = (support_probability * target.support_leg_mask.unsqueeze(0)).sum(-1)
    support_debt = (target.min_support.unsqueeze(0) - support_count).clamp_min(0.0).square()
    support_active = (target.min_support > 0.0).to(prediction.dtype)
    support = (support_debt.mean(0) * support_active * per_world_weight).sum() / (
        support_active * per_world_weight).sum().clamp_min(1.0)

    parts = {"task_root": root, "task_velocity": velocity,
             "task_effector_reach": reach, "task_effector_speed": speed,
             "task_interaction": event, "task_support": support}
    parts["task"] = sum(parts.values())
    return parts["task"], parts


class RecurrentTrajectoryDecoder(nn.Module):
    """Predict a future physical rollout from policy memory and future actions."""

    def __init__(self, context_dim: int, act_dim: int, morph_dim: int):
        super().__init__()
        self.act_dim = int(act_dim)
        self.initial = nn.Linear(context_dim + morph_dim, context_dim)
        self.action = nn.Sequential(nn.Linear(act_dim, context_dim), nn.SiLU())
        self.cell = nn.GRUCell(context_dim, context_dim)
        self.head = nn.Sequential(nn.LayerNorm(context_dim),
                                  nn.Linear(context_dim, TRAJECTORY_TARGET_DIM))

    def forward(self, context: torch.Tensor, morphology: torch.Tensor,
                future_actions: torch.Tensor) -> torch.Tensor:
        if future_actions.ndim != 3:
            raise ValueError("future_actions must be [horizon,batch,action]")
        state = torch.tanh(self.initial(torch.cat((context, morphology), -1)))
        outputs = []
        for action in future_actions:
            state = self.cell(self.action(action), state)
            outputs.append(self.head(state))
        return torch.stack(outputs)


class TemporalTransformerTrajectoryDecoder(nn.Module):
    """Predict a complete causal future in one parallel Transformer pass.

    The full candidate action chunk is known to the planner, but a physical
    state at frame ``t`` must not depend on actions after ``t``.  A causal
    attention mask preserves that contract while still letting the GPU process
    every time position in parallel during training and guidance.
    """

    def __init__(self, context_dim: int, act_dim: int, morph_dim: int,
                 layers: int = 2, heads: int = 4):
        super().__init__()
        if context_dim % heads:
            raise ValueError("trajectory decoder width must be divisible by attention heads")
        self.act_dim = int(act_dim)
        self.action = nn.Linear(act_dim, context_dim)
        self.condition = nn.Linear(context_dim + morph_dim, 2 * context_dim)
        layer = nn.TransformerEncoderLayer(
            context_dim, heads, dim_feedforward=2 * context_dim, dropout=0.0,
            activation="gelu", batch_first=True, norm_first=True)
        self.encoder = nn.TransformerEncoder(
            layer, layers, nn.LayerNorm(context_dim))
        self.head = nn.Linear(context_dim, TRAJECTORY_TARGET_DIM)

    @staticmethod
    def _position_encoding(length: int, width: int, *, device, dtype) -> torch.Tensor:
        position = torch.arange(length, device=device, dtype=torch.float32).unsqueeze(1)
        frequency = torch.exp(
            torch.arange(0, width, 2, device=device, dtype=torch.float32)
            * (-math.log(10_000.0) / max(width, 1)))
        encoding = torch.zeros(length, width, device=device, dtype=torch.float32)
        encoding[:, 0::2] = torch.sin(position * frequency)
        if width > 1:
            encoding[:, 1::2] = torch.cos(position * frequency[:encoding[:, 1::2].shape[1]])
        return encoding.to(dtype=dtype)

    def forward(self, context: torch.Tensor, morphology: torch.Tensor,
                future_actions: torch.Tensor) -> torch.Tensor:
        if future_actions.ndim != 3:
            raise ValueError("future_actions must be [horizon,batch,action]")
        horizon = future_actions.shape[0]
        actions = future_actions.transpose(0, 1)
        x = self.action(actions)
        gamma, beta = self.condition(torch.cat((context, morphology), -1)).chunk(2, -1)
        x = (x * (1.0 + 0.25 * torch.tanh(gamma).unsqueeze(1))
             + 0.25 * beta.unsqueeze(1))
        x = x + self._position_encoding(
            horizon, x.shape[-1], device=x.device, dtype=x.dtype).unsqueeze(0)
        causal = torch.triu(torch.ones(
            horizon, horizon, device=x.device, dtype=torch.bool), diagonal=1)
        return self.head(self.encoder(x, mask=causal)).transpose(0, 1)


def guided_action_sequence(decoder: nn.Module, context: torch.Tensor,
                           morphology: torch.Tensor, initial_actions: torch.Tensor,
                           *, steps: int = 2, learning_rate: float = 0.08,
                           fall_weight: float = 4.0, effort_weight: float = 0.03,
                           feasibility_weight: float = 0.2,
                           interaction_target: InteractionTrajectoryTarget | None = None,
                           task_weight: float = 1.0) -> tuple[torch.Tensor, dict]:
    """Differentiate through predicted physics and improve a candidate action plan.

    The prior term keeps guidance a small correction to the reactive controller;
    fall, effort, and discontinuity costs provide embodiment-aware feasibility.
    """
    base = initial_actions.detach()
    candidate = base.clone().requires_grad_(True)
    def plan_cost(value: torch.Tensor) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        prediction = decoder(context.detach(), morphology.detach(), torch.tanh(value))
        fall = torch.sigmoid(prediction[..., TRAJECTORY_SLICES.fall]).mean()
        effort = prediction[..., TRAJECTORY_SLICES.effort].square().mean()
        predicted_contacts = torch.sigmoid(prediction[..., TRAJECTORY_SLICES.contact])
        support_shortage = (1.0 - predicted_contacts.sum(-1)).clamp_min(0.0).mean()
        smooth = (torch.tanh(candidate[1:]) - torch.tanh(candidate[:-1])).square().mean() \
            if len(candidate) > 1 else candidate.new_zeros(())
        prior = (value - base).square().mean()
        task = value.new_zeros(())
        task_parts: dict[str, torch.Tensor] = {}
        if interaction_target is not None:
            task, task_parts = interaction_trajectory_cost(prediction, interaction_target)
        cost = task_weight * task + fall_weight * fall + effort_weight * effort \
            + feasibility_weight * (smooth + support_shortage) + prior
        parts = {"total": cost, "task": task, "fall": fall, "effort": effort,
                 "support_shortage": support_shortage,
                 "smoothness": smooth, "prior": prior, **task_parts}
        return cost, parts

    # Record whether optimization is meaningful rather than merely recording
    # that the planner was called.
    with torch.enable_grad():
        _, initial_costs = plan_cost(candidate)
    gradient_norms = []
    for _ in range(max(int(steps), 0)):
        cost, _ = plan_cost(candidate)
        # Costs are means over worlds.  Multiplying by batch makes the per-world
        # action correction invariant to evaluation batch size instead of
        # silently shrinking toward zero as parallelism grows.
        gradient, = torch.autograd.grad(cost * candidate.shape[1], candidate)
        gradient_norms.append(gradient.square().mean().sqrt().detach())
        candidate = (candidate - learning_rate * gradient).clamp(-5.0, 5.0).detach() \
            .requires_grad_(True)
    with torch.enable_grad():
        _, final_costs = plan_cost(candidate)
    base_action = torch.tanh(base)
    planned_action = torch.tanh(candidate.detach())
    diagnostics = {
        **{f"before_{name}": value.detach() for name, value in initial_costs.items()},
        **{f"after_{name}": value.detach() for name, value in final_costs.items()},
        "gradient_rms": (torch.stack(gradient_norms).mean() if gradient_norms
                         else candidate.new_zeros(())),
        "action_delta_rms": (planned_action - base_action).square().mean().sqrt(),
        "action_delta_max": (planned_action - base_action).abs().amax(),
    }
    return planned_action, diagnostics
