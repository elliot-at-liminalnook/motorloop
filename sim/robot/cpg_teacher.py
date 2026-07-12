# SPDX-License-Identifier: MIT
"""Shared CPG teacher used by search, training, dataset collection, and eval.

This module owns the low-level sinusoidal gait parameterization.  Keep the
actual gait math here so CPG search and `CommandedEnv` cannot drift apart again.
"""

from __future__ import annotations

from dataclasses import dataclass
import json
import math
import os
from pathlib import Path
from typing import Any

import numpy as np

PARAM_DIM = 35
LEG_NAMES = ("FL", "FR", "RL", "RR")
JOINT_NAMES = ("abd", "flex", "knee")
DEFAULT_RAW = (
    1.6, 0.0, math.pi, math.pi, 0.0,
    0.0, 0.45, 0.0,
    0.0, -0.35, 0.35,
    0.0, 0.05, 0.0,
    0.5, 0.0, 0.0, 0.0, 0.0,
    0.0, 0.0, 0.0, 0.0,
    1.0, 1.0, 1.0, 1.0,
    1.0, 1.0, 1.0, 1.0,
    1.0, 1.0, 1.0, 1.0,
)


def _tuple_env(name: str, default: tuple[float, ...]) -> tuple[float, ...]:
    raw = os.environ.get(name)
    if not raw:
        return default
    vals = tuple(float(x.strip()) for x in raw.replace(";", ",").split(",") if x.strip())
    if len(vals) != len(default):
        raise ValueError(f"{name} expected {len(default)} comma-separated floats, got {len(vals)}")
    return vals


def _raw_env(name: str) -> np.ndarray | None:
    raw = os.environ.get(name)
    if not raw:
        return None
    vals = [
        float(x.strip())
        for x in raw.replace("[", "").replace("]", "").replace(";", ",").split(",")
        if x.strip()
    ]
    if len(vals) not in (14, PARAM_DIM):
        raise ValueError(f"{name} expected 14 or {PARAM_DIM} comma-separated floats, got {len(vals)}")
    return np.asarray(vals, dtype=float)


@dataclass(frozen=True)
class CPGParams:
    freq: Any
    phase_offsets: Any
    flex: Any
    knee: Any
    abd: Any
    duty_factor: Any = 0.5
    foot_clearance: Any = 0.0
    stance_height: Any = 0.0
    lateral_bias: Any = 0.0
    yaw_bias: Any = 0.0
    leg_phase_skew: Any = (0.0, 0.0, 0.0, 0.0)
    flex_scale: Any = (1.0, 1.0, 1.0, 1.0)
    knee_scale: Any = (1.0, 1.0, 1.0, 1.0)
    abd_scale: Any = (1.0, 1.0, 1.0, 1.0)


@dataclass(frozen=True)
class DirectionalCPGParams:
    backward: CPGParams
    forward: CPGParams
    right: CPGParams
    left: CPGParams


CPG_FREQ = float(os.environ.get("CMD_CPG_FREQ", "2.3311830770273883"))
CPG_PHASES = _tuple_env("CMD_CPG_PHASES", (2.490359217242055, 4.3063642666711734, 3.7875659070859546, 3.5594085789810963))
CPG_FLEX = _tuple_env("CMD_CPG_FLEX", (0.45966979961573434, 0.5497073611969452, -0.5387807457533359))
CPG_KNEE = _tuple_env("CMD_CPG_KNEE", (0.09117364413977733, -1.0, 0.3400696236889605))
CPG_ABD = _tuple_env("CMD_CPG_ABD", (-0.2095847593411988, 0.021138716435718723, 0.0354906730851447))

CPG_FWD_FREQ = float(os.environ.get("CMD_CPG_FWD_FREQ", "0.4034733945173041"))
CPG_FWD_PHASES = _tuple_env("CMD_CPG_FWD_PHASES", (2.679208967599348, 2.6900453791931644, 2.849267611638736, 2.811809625670169))
CPG_FWD_FLEX = _tuple_env("CMD_CPG_FWD_FLEX", (0.5606508678163025, 0.6747202723347252, -0.23512752136986062))
CPG_FWD_KNEE = _tuple_env("CMD_CPG_FWD_KNEE", (-0.6016914652902376, -0.3706009759021222, 0.5615922584762738))
CPG_FWD_ABD = _tuple_env("CMD_CPG_FWD_ABD", (0.2518272366809658, 0.011602467376185427, 0.011748098844599732))

CPG_RIGHT_FREQ = float(os.environ.get("CMD_CPG_RIGHT_FREQ", "1.8870274536837222"))
CPG_RIGHT_PHASES = _tuple_env("CMD_CPG_RIGHT_PHASES", (2.642200148774274, 3.965355779706484, 2.959659549814987, 3.498519023232316))
CPG_RIGHT_FLEX = _tuple_env("CMD_CPG_RIGHT_FLEX", (0.2819041385543283, 0.6240818454310288, 0.18302879494100793))
CPG_RIGHT_KNEE = _tuple_env("CMD_CPG_RIGHT_KNEE", (-0.27455635237924947, 0.7863625942078021, 0.5847941509807889))
CPG_RIGHT_ABD = _tuple_env("CMD_CPG_RIGHT_ABD", (-0.0036915103541357297, -0.5, 0.4696551703800028))

CPG_LEFT_FREQ = float(os.environ.get("CMD_CPG_LEFT_FREQ", "2.0554187420291914"))
CPG_LEFT_PHASES = _tuple_env("CMD_CPG_LEFT_PHASES", (3.3073506905771186, 3.337085176644598, 3.606809107543834, 2.6421262575939863))
CPG_LEFT_FLEX = _tuple_env("CMD_CPG_LEFT_FLEX", (0.19981689965346552, 0.5368382195056368, 0.6116957367106818))
CPG_LEFT_KNEE = _tuple_env("CMD_CPG_LEFT_KNEE", (-0.4007722530900349, 0.28517371031967237, 0.654341017109891))
CPG_LEFT_ABD = _tuple_env("CMD_CPG_LEFT_ABD", (-0.1618802156253249, 0.45177879755062594, -0.08083421040237325))
CPG_ROUTER = os.environ.get("CMD_CPG_ROUTER", "blend").strip().lower()
CPG_GATE_POWER = float(os.environ.get("CMD_CPG_GATE_POWER", "1.0"))
CPG_RBF_SIGMA = float(os.environ.get("CMD_CPG_RBF_SIGMA", "0.85"))
CPG_PREV_CMD_BLEND = float(os.environ.get("CMD_CPG_PREV_CMD_BLEND", "0.25"))
CPG_MAP_JSON = os.environ.get("CMD_CPG_MAP_JSON", "").strip()
CPG_PHASE_RESET_AMOUNT = float(os.environ.get("CMD_CPG_PHASE_RESET_AMOUNT", "0.0"))
CPG_PHASE_SPEED_BOOST = float(os.environ.get("CMD_CPG_PHASE_SPEED_BOOST", "0.0"))
CPG_STANCE_HEIGHT_BOOST = float(os.environ.get("CMD_CPG_STANCE_HEIGHT_BOOST", "0.0"))
CPG_TURN_BIAS = float(os.environ.get("CMD_CPG_TURN_BIAS", "0.0"))
CPG_CROSS_AXIS_CORRECTION = float(os.environ.get("CMD_CPG_CROSS_AXIS_CORRECTION", "0.0"))


def _load_linear_map(path: str):
    if not path:
        return None
    try:
        data = json.loads(Path(path).read_text())
        coef = np.asarray(data["coef"], dtype=np.float32)
        if coef.shape[1] != 4:
            raise ValueError(f"coef expected 4 output weights, got {coef.shape}")
        return {
            "coef": coef,
            "feature_names": tuple(data.get("feature_names", ())),
        }
    except Exception as exc:
        raise RuntimeError(f"failed to load CMD_CPG_MAP_JSON={path}: {exc}") from exc


CPG_LINEAR_MAP = _load_linear_map(CPG_MAP_JSON)


def make_directional_params_from_env() -> DirectionalCPGParams:
    backward = CPGParams(CPG_FREQ, CPG_PHASES, CPG_FLEX, CPG_KNEE, CPG_ABD)
    forward = CPGParams(CPG_FWD_FREQ, CPG_FWD_PHASES, CPG_FWD_FLEX, CPG_FWD_KNEE, CPG_FWD_ABD)
    right = CPGParams(CPG_RIGHT_FREQ, CPG_RIGHT_PHASES, CPG_RIGHT_FLEX, CPG_RIGHT_KNEE, CPG_RIGHT_ABD)
    left = CPGParams(CPG_LEFT_FREQ, CPG_LEFT_PHASES, CPG_LEFT_FLEX, CPG_LEFT_KNEE, CPG_LEFT_ABD)
    raw_backward = _raw_env("CMD_CPG_RAW")
    raw_forward = _raw_env("CMD_CPG_FWD_RAW")
    raw_right = _raw_env("CMD_CPG_RIGHT_RAW")
    raw_left = _raw_env("CMD_CPG_LEFT_RAW")
    return DirectionalCPGParams(
        backward=decode_params(raw_backward, xp=np) if raw_backward is not None else backward,
        forward=decode_params(raw_forward, xp=np) if raw_forward is not None else forward,
        right=decode_params(raw_right, xp=np) if raw_right is not None else right,
        left=decode_params(raw_left, xp=np) if raw_left is not None else left,
    )


def decode_params(raw, xp=None) -> CPGParams:
    """Decode the searchable vector into clipped CPGParams.

    Legacy 14-D vectors are accepted and padded with neutral rich-gait values.
    """
    if xp is None:
        xp = np
    z = xp.asarray(raw)
    if int(z.shape[0]) < PARAM_DIM:
        default = xp.asarray(DEFAULT_RAW, dtype=z.dtype)
        z = xp.concatenate([z, default[int(z.shape[0]):]], axis=0)
    return CPGParams(
        freq=xp.clip(z[0], 0.4, 4.0),
        phase_offsets=xp.mod(z[1:5], 2.0 * math.pi),
        flex=xp.asarray([xp.clip(z[5], -0.7, 0.7), xp.clip(z[6], -1.0, 1.0), xp.clip(z[7], -1.0, 1.0)]),
        knee=xp.asarray([xp.clip(z[8], -0.7, 0.7), xp.clip(z[9], -1.0, 1.0), xp.clip(z[10], -1.0, 1.0)]),
        abd=xp.asarray([xp.clip(z[11], -0.4, 0.4), xp.clip(z[12], -0.5, 0.5), xp.clip(z[13], -0.5, 0.5)]),
        duty_factor=xp.clip(z[14], 0.25, 0.85),
        foot_clearance=xp.clip(z[15], -0.6, 0.6),
        stance_height=xp.clip(z[16], -0.5, 0.5),
        lateral_bias=xp.clip(z[17], -0.4, 0.4),
        yaw_bias=xp.clip(z[18], -0.4, 0.4),
        leg_phase_skew=xp.clip(z[19:23], -math.pi, math.pi),
        flex_scale=xp.clip(z[23:27], 0.2, 1.8),
        knee_scale=xp.clip(z[27:31], 0.2, 1.8),
        abd_scale=xp.clip(z[31:35], 0.2, 1.8),
    )


def encode_params(params: CPGParams) -> np.ndarray:
    return np.asarray([
        float(np.asarray(params.freq)),
        *np.asarray(params.phase_offsets, dtype=float).tolist(),
        *np.asarray(params.flex, dtype=float).tolist(),
        *np.asarray(params.knee, dtype=float).tolist(),
        *np.asarray(params.abd, dtype=float).tolist(),
        float(np.asarray(params.duty_factor)),
        float(np.asarray(params.foot_clearance)),
        float(np.asarray(params.stance_height)),
        float(np.asarray(params.lateral_bias)),
        float(np.asarray(params.yaw_bias)),
        *np.asarray(params.leg_phase_skew, dtype=float).tolist(),
        *np.asarray(params.flex_scale, dtype=float).tolist(),
        *np.asarray(params.knee_scale, dtype=float).tolist(),
        *np.asarray(params.abd_scale, dtype=float).tolist(),
    ], dtype=float)


def params_to_dict(params: CPGParams) -> dict[str, Any]:
    return {
        "freq": float(np.asarray(params.freq)),
        "phase": np.asarray(params.phase_offsets, dtype=float).tolist(),
        "flex_bias": float(np.asarray(params.flex)[0]),
        "flex_sin": float(np.asarray(params.flex)[1]),
        "flex_cos": float(np.asarray(params.flex)[2]),
        "knee_bias": float(np.asarray(params.knee)[0]),
        "knee_sin": float(np.asarray(params.knee)[1]),
        "knee_cos": float(np.asarray(params.knee)[2]),
        "abd_bias": float(np.asarray(params.abd)[0]),
        "abd_sin": float(np.asarray(params.abd)[1]),
        "abd_cos": float(np.asarray(params.abd)[2]),
        "duty_factor": float(np.asarray(params.duty_factor)),
        "foot_clearance": float(np.asarray(params.foot_clearance)),
        "stance_height": float(np.asarray(params.stance_height)),
        "lateral_bias": float(np.asarray(params.lateral_bias)),
        "yaw_bias": float(np.asarray(params.yaw_bias)),
        "leg_phase_skew": np.asarray(params.leg_phase_skew, dtype=float).tolist(),
        "flex_scale": np.asarray(params.flex_scale, dtype=float).tolist(),
        "knee_scale": np.asarray(params.knee_scale, dtype=float).tolist(),
        "abd_scale": np.asarray(params.abd_scale, dtype=float).tolist(),
    }


def _asarray(x, xp):
    return xp.asarray(x, dtype=xp.float32 if hasattr(xp, "float32") else None)


def transition_controls_array(controls=None, xp=None):
    if xp is None:
        xp = np
    if controls is None:
        return xp.asarray([
            CPG_PHASE_RESET_AMOUNT,
            CPG_PHASE_SPEED_BOOST,
            CPG_STANCE_HEIGHT_BOOST,
            CPG_TURN_BIAS,
            CPG_CROSS_AXIS_CORRECTION,
        ], dtype=xp.float32)
    return xp.asarray(controls, dtype=xp.float32)


def transition_strength(command, prev_command, vmax: float, xp=None):
    """Return 0..1 command-switch strength for transition-only CPG controls."""
    if xp is None:
        xp = np
    if prev_command is None:
        return xp.asarray(0.0, dtype=xp.float32)
    cmd = xp.asarray(command)
    prev = xp.asarray(prev_command)
    return xp.clip(xp.linalg.norm(cmd - prev) / max(float(vmax), 1e-6), 0.0, 1.0)


def transition_turn_signal(command, prev_command, xp=None):
    """Signed left/right turn signal from previous command to current command."""
    if xp is None:
        xp = np
    if prev_command is None:
        return xp.asarray(0.0, dtype=xp.float32)
    cmd = xp.asarray(command)
    prev = xp.asarray(prev_command)
    denom = xp.maximum(xp.linalg.norm(cmd) * xp.linalg.norm(prev), 1e-6)
    cross = prev[0] * cmd[1] - prev[1] * cmd[0]
    return xp.clip(cross / denom, -1.0, 1.0)


def linear_map_features(command, prev_command, vmax: float, xp=None):
    if xp is None:
        xp = np
    cmd = xp.asarray(command)
    prev = xp.zeros_like(cmd) if prev_command is None else xp.asarray(prev_command)
    cmd_n = cmd / max(float(vmax), 1e-6)
    prev_n = prev / max(float(vmax), 1e-6)
    speed = xp.clip(xp.linalg.norm(cmd) / max(float(vmax), 1e-6), 0.0, 1.0)
    change = transition_strength(cmd, prev, vmax, xp=xp)
    # Feature order must match fit_cpg_param_map.py.
    return xp.asarray([
        1.0,
        cmd_n[0],
        cmd_n[1],
        speed,
        prev_n[0],
        prev_n[1],
        change,
        0.0,
    ], dtype=xp.float32)


def _linear_map_weight_vector(command, vmax: float, xp, prev_command=None):
    if CPG_LINEAR_MAP is None:
        return None
    coef = xp.asarray(CPG_LINEAR_MAP["coef"], dtype=xp.float32)
    feat = linear_map_features(command, prev_command, vmax, xp=xp)
    weights = xp.maximum(0.0, feat @ coef)
    weights = weights / xp.maximum(xp.sum(weights), 1e-6)
    cmd = xp.asarray(command)
    speed_scale = xp.clip(xp.linalg.norm(cmd) / vmax, 0.0, 1.0)
    weights = xp.where(speed_scale > 1e-6, weights, xp.zeros_like(weights))
    return weights, speed_scale


def transition_phase_for_action(phase, strength, controls=None, xp=None):
    """Apply the transition phase-reset control without mutating caller state."""
    if xp is None:
        xp = np
    ctrl = transition_controls_array(controls, xp=xp)
    reset = xp.clip(ctrl[0], 0.0, 1.0)
    return phase * (1.0 - reset * xp.clip(strength, 0.0, 1.0))


def transition_phase_delta(base_freq, dt: float, strength, controls=None, xp=None):
    """Phase increment with optional transition speed boost."""
    if xp is None:
        xp = np
    ctrl = transition_controls_array(controls, xp=xp)
    boost = 1.0 + ctrl[1] * xp.clip(strength, 0.0, 1.0)
    boost = xp.maximum(boost, 0.05)
    return 2.0 * math.pi * base_freq * float(dt) * boost


def apply_transition_controls(params: CPGParams, command, prev_command, strength, controls=None, xp=None) -> CPGParams:
    """Return params with transition-only height/turn/cross-axis controls applied."""
    if xp is None:
        xp = np
    ctrl = transition_controls_array(controls, xp=xp)
    s = xp.clip(strength, 0.0, 1.0)
    turn = transition_turn_signal(command, prev_command, xp=xp)
    return CPGParams(
        freq=params.freq,
        phase_offsets=params.phase_offsets,
        flex=params.flex,
        knee=params.knee,
        abd=params.abd,
        duty_factor=params.duty_factor,
        foot_clearance=params.foot_clearance,
        stance_height=params.stance_height + ctrl[2] * s,
        lateral_bias=params.lateral_bias + ctrl[4] * turn * s,
        yaw_bias=params.yaw_bias + ctrl[3] * turn * s,
        leg_phase_skew=params.leg_phase_skew,
        flex_scale=params.flex_scale,
        knee_scale=params.knee_scale,
        abd_scale=params.abd_scale,
    )


def _directional_weight_vector(command, vmax: float, xp, prev_command=None):
    cmd = xp.asarray(command)
    norm = xp.linalg.norm(cmd)
    speed_scale = xp.clip(norm / vmax, 0.0, 1.0)
    gates = xp.asarray([
        xp.clip(cmd[0] / vmax, 0.0, 1.0),
        xp.clip(-cmd[0] / vmax, 0.0, 1.0),
        xp.clip(cmd[1] / vmax, 0.0, 1.0),
        xp.clip(-cmd[1] / vmax, 0.0, 1.0),
    ], dtype=xp.float32)
    if CPG_ROUTER == "winner":
        idx = xp.argmax(gates)
        gate_sum = xp.sum(gates)
        weights = xp.where(xp.arange(4) == idx, 1.0, 0.0)
        weights = xp.where(gate_sum > 1e-6, weights, xp.zeros_like(weights))
        return weights, xp.minimum(1.0, gate_sum)
    if CPG_ROUTER in ("softmax", "param_softmax"):
        power = max(1.0, CPG_GATE_POWER)
        powered = xp.power(gates, power)
        weights = powered / xp.maximum(xp.sum(powered), 1e-6)
        weights = xp.where(xp.sum(gates) > 1e-6, weights, xp.zeros_like(weights))
        return weights, xp.minimum(1.0, xp.sum(gates))
    if CPG_ROUTER in ("param_rbf", "continuous"):
        centers = xp.asarray([[1.0, 0.0], [-1.0, 0.0], [0.0, 1.0], [0.0, -1.0]], dtype=xp.float32)
        direction = xp.where(norm > 1e-6, cmd / xp.maximum(norm, 1e-6), xp.asarray([1.0, 0.0], dtype=xp.float32))
        d2 = xp.sum((centers - direction[None, :]) ** 2, axis=1)
        weights = xp.exp(-d2 / (2.0 * max(CPG_RBF_SIGMA, 1e-3) ** 2))
        weights = weights / xp.maximum(xp.sum(weights), 1e-6)
        if prev_command is not None and CPG_PREV_CMD_BLEND > 0.0:
            prev = xp.asarray(prev_command)
            prev_norm = xp.linalg.norm(prev)
            prev_direction = xp.where(
                prev_norm > 1e-6,
                prev / xp.maximum(prev_norm, 1e-6),
                direction,
            )
            prev_d2 = xp.sum((centers - prev_direction[None, :]) ** 2, axis=1)
            prev_weights = xp.exp(-prev_d2 / (2.0 * max(CPG_RBF_SIGMA, 1e-3) ** 2))
            prev_weights = prev_weights / xp.maximum(xp.sum(prev_weights), 1e-6)
            change = xp.clip(xp.linalg.norm(cmd - prev) / vmax, 0.0, 1.0)
            keep = xp.clip(CPG_PREV_CMD_BLEND * change, 0.0, 0.75)
            weights = (1.0 - keep) * weights + keep * prev_weights
            weights = weights / xp.maximum(xp.sum(weights), 1e-6)
        return weights, speed_scale
    if CPG_ROUTER in ("param_linear", "linear", "fitted_linear"):
        out = _linear_map_weight_vector(command, vmax, xp, prev_command=prev_command)
        if out is not None:
            return out
        return gates, xp.minimum(1.0, xp.sum(gates))
    return gates, xp.minimum(1.0, xp.sum(gates))


def _stack_params(directional: DirectionalCPGParams, name: str, xp):
    return xp.stack([
        _asarray(getattr(directional.forward, name), xp),
        _asarray(getattr(directional.backward, name), xp),
        _asarray(getattr(directional.left, name), xp),
        _asarray(getattr(directional.right, name), xp),
    ], axis=0)


def _blend_linear(directional: DirectionalCPGParams, name: str, weights, xp):
    vals = _stack_params(directional, name, xp)
    return xp.sum(weights.reshape((-1,) + (1,) * (vals.ndim - 1)) * vals, axis=0)


def _blend_angles(directional: DirectionalCPGParams, name: str, weights, xp):
    vals = _stack_params(directional, name, xp)
    s = xp.sum(weights.reshape((-1,) + (1,) * (vals.ndim - 1)) * xp.sin(vals), axis=0)
    c = xp.sum(weights.reshape((-1,) + (1,) * (vals.ndim - 1)) * xp.cos(vals), axis=0)
    return xp.mod(xp.arctan2(s, c), 2.0 * math.pi)


def blend_directional_params(directional: DirectionalCPGParams, weights, xp=None) -> CPGParams:
    if xp is None:
        xp = np
    return CPGParams(
        freq=_blend_linear(directional, "freq", weights, xp),
        phase_offsets=_blend_angles(directional, "phase_offsets", weights, xp),
        flex=_blend_linear(directional, "flex", weights, xp),
        knee=_blend_linear(directional, "knee", weights, xp),
        abd=_blend_linear(directional, "abd", weights, xp),
        duty_factor=_blend_linear(directional, "duty_factor", weights, xp),
        foot_clearance=_blend_linear(directional, "foot_clearance", weights, xp),
        stance_height=_blend_linear(directional, "stance_height", weights, xp),
        lateral_bias=_blend_linear(directional, "lateral_bias", weights, xp),
        yaw_bias=_blend_linear(directional, "yaw_bias", weights, xp),
        leg_phase_skew=_blend_angles(directional, "leg_phase_skew", weights, xp),
        flex_scale=_blend_linear(directional, "flex_scale", weights, xp),
        knee_scale=_blend_linear(directional, "knee_scale", weights, xp),
        abd_scale=_blend_linear(directional, "abd_scale", weights, xp),
    )


def cpg_action(phase, params: CPGParams, cpg_idx, nu: int, xp=None):
    """Return a motor-action prior in [-unclipped, unclipped] joint-action order."""
    if xp is None:
        xp = np
    phases = _asarray(params.phase_offsets, xp)
    flex_p = _asarray(params.flex, xp)
    knee_p = _asarray(params.knee, xp)
    abd_p = _asarray(params.abd, xp)
    skew = _asarray(params.leg_phase_skew, xp)
    flex_scale = _asarray(params.flex_scale, xp)
    knee_scale = _asarray(params.knee_scale, xp)
    abd_scale = _asarray(params.abd_scale, xp)
    ph = phase + phases + skew
    s = xp.sin(ph)
    c = xp.cos(ph)
    side = xp.asarray([1.0, -1.0, 1.0, -1.0], dtype=xp.float32)
    front_rear = xp.asarray([1.0, 1.0, -1.0, -1.0], dtype=xp.float32)
    cycle = xp.mod(ph, 2.0 * math.pi) / (2.0 * math.pi)
    swing = 0.5 * (1.0 + xp.tanh(12.0 * (cycle - params.duty_factor)))
    abd = side * (abd_p[0] + params.lateral_bias) + abd_scale * (abd_p[1] * s + abd_p[2] * c)
    flex = (flex_p[0] + params.stance_height + params.yaw_bias * front_rear
            + flex_scale * (flex_p[1] * s + flex_p[2] * c))
    knee = (knee_p[0] - params.stance_height + params.foot_clearance * swing
            + knee_scale * (knee_p[1] * s + knee_p[2] * c))
    rows = xp.stack([abd, flex, knee], axis=1)
    out = xp.zeros(nu, dtype=xp.float32)
    if hasattr(out, "at"):
        return out.at[cpg_idx].set(rows).reshape((nu,))
    out[np.asarray(cpg_idx)] = np.asarray(rows)
    return out.reshape((nu,))


def blend_directional_priors(
    phase,
    command,
    cpg_idx,
    nu: int,
    vmax: float,
    directional: DirectionalCPGParams | None = None,
    prev_command=None,
    transition_amount=None,
    transition_controls=None,
    xp=None,
):
    """Blend cardinal CPG priors by the command direction."""
    if xp is None:
        xp = np
    directional = directional or make_directional_params_from_env()
    base_freq = directional.backward.freq
    amount = (
        transition_strength(command, prev_command, vmax, xp=xp)
        if transition_amount is None
        else xp.asarray(transition_amount, dtype=xp.float32)
    )
    phase = transition_phase_for_action(phase, amount, controls=transition_controls, xp=xp)
    weights, speed_scale = _directional_weight_vector(command, vmax, xp, prev_command=prev_command)
    if CPG_ROUTER in ("param_rbf", "continuous", "param_softmax", "param_linear", "linear", "fitted_linear"):
        params = blend_directional_params(directional, weights, xp=xp)
        params = apply_transition_controls(
            params, command, prev_command, amount, controls=transition_controls, xp=xp
        )
        freq_ratio = params.freq / xp.maximum(base_freq, 1e-6)
        return speed_scale * cpg_action(phase * freq_ratio, params, cpg_idx, nu, xp=xp)
    backward = apply_transition_controls(
        directional.backward, command, prev_command, amount, controls=transition_controls, xp=xp
    )
    forward = apply_transition_controls(
        directional.forward, command, prev_command, amount, controls=transition_controls, xp=xp
    )
    right_p = apply_transition_controls(
        directional.right, command, prev_command, amount, controls=transition_controls, xp=xp
    )
    left_p = apply_transition_controls(
        directional.left, command, prev_command, amount, controls=transition_controls, xp=xp
    )
    back = cpg_action(phase, backward, cpg_idx, nu, xp=xp)
    fwd = cpg_action(phase * (forward.freq / base_freq), forward, cpg_idx, nu, xp=xp)
    right = cpg_action(phase * (right_p.freq / base_freq), right_p, cpg_idx, nu, xp=xp)
    left = cpg_action(phase * (left_p.freq / base_freq), left_p, cpg_idx, nu, xp=xp)
    priors = xp.stack([fwd, back, left, right], axis=0)
    if CPG_ROUTER in ("winner", "softmax"):
        return speed_scale * xp.sum(weights[:, None] * priors, axis=0)
    gates = weights
    denom = xp.maximum(1.0, xp.sum(gates))
    return xp.sum(gates[:, None] * priors, axis=0) / denom


def cpg_pd_step_target(
    stand,
    joint_range,
    phase,
    command,
    raw_action,
    cpg_idx,
    nu: int,
    vmax: float,
    residual_scale: float,
    pd_scale: float,
    directional: DirectionalCPGParams | None = None,
    prev_command=None,
    transition_amount=None,
    transition_controls=None,
    xp=None,
):
    """Map command + residual policy action to PD targets.

    Returns `(target, motor_action, prior)` so callers can use the same action for
    PD control, action-rate penalties, and dataset labels.
    """
    if xp is None:
        xp = np
    prior = blend_directional_priors(
        phase, command, cpg_idx, nu, vmax,
        directional=directional,
        prev_command=prev_command,
        transition_amount=transition_amount,
        transition_controls=transition_controls,
        xp=xp,
    )
    motor_action = xp.clip(prior + residual_scale * xp.clip(raw_action, -1.0, 1.0), -1.0, 1.0)
    target = xp.clip(stand + pd_scale * motor_action, joint_range[:, 0], joint_range[:, 1])
    return target, motor_action, prior


def phase_advance(dt: float, directional: DirectionalCPGParams | None = None) -> float:
    directional = directional or make_directional_params_from_env()
    return 2.0 * math.pi * float(directional.backward.freq) * float(dt)


def command_vector(name: str, speed: float) -> np.ndarray:
    table = {
        "forward": (1.0, 0.0),
        "backward": (-1.0, 0.0),
        "left": (0.0, 1.0),
        "right": (0.0, -1.0),
    }
    if name not in table:
        raise KeyError(name)
    return float(speed) * np.asarray(table[name], dtype=float)
