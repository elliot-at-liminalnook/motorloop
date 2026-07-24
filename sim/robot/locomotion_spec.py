# SPDX-License-Identifier: MIT
"""Backend-independent command, route, and reward constants for locomotion."""

from __future__ import annotations

import json
import os
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
DEFAULT_FAST_DESIGN = (0.5, 0.08, 1.0 / 3.0)
VMAX = float(os.environ.get("CMD_VMAX", "1.2"))
TRACK_W = 5.0
TRACK_SIGMA = float(os.environ.get("CMD_TRACK_SIGMA", "0.35"))
UPRIGHT_W = 0.1
ALIGN_W = 1.0
BACKWARD_W = 6.0
PROGRESS_W = 12.0
CMD_HOLD_STEPS = int(os.environ.get("CMD_HOLD_STEPS", "80"))
FALL_Z = 0.10
MIN_UP_Z = 0.25
CMD_TRAIN_MODE = os.environ.get("CMD_TRAIN_MODE", "cardinal")
CMD_REWARD_MODE = os.environ.get("CMD_REWARD_MODE", "command")
CMD_CONTROL_MODE = os.environ.get("CMD_CONTROL_MODE", "pd")
RESET_NOISE = float(os.environ.get("CMD_RESET_NOISE", "0.05"))
CPG_WP2_ACTION_JSON = os.environ.get("CMD_CPG_WP2_ACTION_JSON", "").strip()
CPG_WP2_ACTION_SEGMENT_STEPS = int(os.environ.get("CMD_CPG_WP2_ACTION_SEGMENT_STEPS", "0") or "0")
CPG_WP3_ACTION_JSON = os.environ.get("CMD_CPG_WP3_ACTION_JSON", "").strip()
CPG_WP3_ACTION_SEGMENT_STEPS = int(os.environ.get("CMD_CPG_WP3_ACTION_SEGMENT_STEPS", "0") or "0")
CMD_ROUTE_WAYPOINTS = os.environ.get("CMD_ROUTE_WAYPOINTS", "0.35,0;0.35,0.35;0,0.35;0,0")
CMD_ROUTE_RADIUS = float(os.environ.get("CMD_ROUTE_RADIUS", "0.07"))
CMD_ROUTE_GAIN = float(os.environ.get("CMD_ROUTE_GAIN", "2.0"))
CMD_ROUTE_START_WP = int(os.environ.get("CMD_ROUTE_START_WP", "0"))
CMD_ROUTE_START_XY = os.environ.get("CMD_ROUTE_START_XY", "")
CMD_ROUTE_STARTS = os.environ.get("CMD_ROUTE_STARTS", "")
AIRTIME_W = 1.0
ACTRATE_W = 0.05
VELZ_W = 0.5
ANGXY_W = 0.1
POSE_W = 0.2
SLIP_W = 0.1
YAW_MAX = float(os.environ.get("CMD_YAW_MAX", "0.8"))


def resolve_optional_json(path_text: str) -> Path:
    path = Path(path_text)
    candidates = [path]
    if not path.is_absolute():
        candidates.extend((HERE.parents[1] / path,
                           Path(os.environ.get("CODESIGN_OUT", "/root/proj/out")) / path))
    for candidate in candidates:
        if candidate.exists():
            return candidate
    raise FileNotFoundError(path_text)


def load_waypoint_action_schedule(path_text: str, steps_override: int,
                                  env_name: str) -> tuple[np.ndarray | None, int]:
    if not path_text:
        return None, 0
    data = json.loads(resolve_optional_json(path_text).read_text())
    action = np.asarray(data.get("best_action"), dtype=np.float32)
    if action.ndim == 1:
        action = action[None, :]
    if action.ndim != 2:
        raise ValueError(f"{env_name} best_action must be 1-D or 2-D, got {action.shape}")
    steps = int(steps_override or data.get("best", {}).get("segment_steps", 0) or 1)
    return np.clip(action, -1.0, 1.0), max(1, steps)


CPG_WP2_ACTIONS, CPG_WP2_ACTION_STEPS = load_waypoint_action_schedule(
    CPG_WP2_ACTION_JSON, CPG_WP2_ACTION_SEGMENT_STEPS, "CMD_CPG_WP2_ACTION_JSON")
CPG_WP3_ACTIONS, CPG_WP3_ACTION_STEPS = load_waypoint_action_schedule(
    CPG_WP3_ACTION_JSON, CPG_WP3_ACTION_SEGMENT_STEPS, "CMD_CPG_WP3_ACTION_JSON")


def parse_route_waypoints(text: str) -> np.ndarray:
    points = [tuple(map(float, chunk.split(","))) for chunk in text.split(";") if chunk.strip()]
    return np.asarray(points or [(0.35, 0.0), (0.35, 0.35), (0.0, 0.35), (0.0, 0.0)],
                      dtype=np.float32)


def parse_xy(text: str):
    return None if not text.strip() else np.asarray(tuple(map(float, text.split(","))), dtype=np.float32)


def parse_route_starts(text: str):
    starts = []
    for chunk in text.split(";"):
        if chunk.strip():
            waypoint, x, y = chunk.split(",")
            starts.append((int(waypoint), float(x), float(y)))
    return starts
