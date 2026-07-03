# SPDX-License-Identifier: MIT
"""V.1: the one home for every load-bearing number (notes/uplift-execution-plan.md).

Each constant carries its derivation and what breaks if it moves. Env-var
overrides are read ONCE, here — a knob read in four places is four bugs waiting.
test_model_contract asserts the compiled model agrees with this module.
"""
from __future__ import annotations

import os

# --- simulation clock ---------------------------------------------------------
TIMESTEP = 0.004          # s, emitted by gen_robot_mjcf._wrap; implicitfast-stable
FRAME_SKIP = 5            # physics substeps per control step (train_* default)
CONTROL_DT = TIMESTEP * FRAME_SKIP   # 0.02 s = 50 Hz control; air-time/progress math assumes it

# --- PD action pathway (B.1/A.9): single read point for the gains -------------
PD_KP = float(os.environ.get("CMD_PD_KP", "30.0"))    # N·m/rad; validated by A.3 stance-hold
PD_KD = float(os.environ.get("CMD_PD_KD", "1.0"))     # N·m·s/rad
PD_ACTION_SCALE = 0.4     # rad of target authority per unit action (fighter pd mode)

# --- contact / stance geometry (full-torque body, validate_body 2026-07-02) ---
FOOT_CONTACT_Z = 0.04     # m, foot-sphere height counting as ground contact (r=0.03 + slack)
FALL_Z = 0.09             # m, torso below this = fallen (crouch ~0.11 survives)
TOPPLE_UP_Z = 0.3         # up-axis·ẑ below this = toppled (~70° tilt)
STANCE_Z = 0.286          # m, measured settled canonical stance (validator, stiffness 0.5)
MAX_EXTENSION_Z = 0.49    # m, kinematic ceiling at knee limit −0.4 (anti-cheat launch bar 0.60)

# --- gait terms (Go2-recipe calibration, shared by commanded_env + fighter B.3) -
AIRTIME_TARGET = 0.2      # s, desired swing duration
AIRTIME_CAP = 2 * AIRTIME_TARGET   # credit saturates: hops must not out-earn cadence
GAIT_DISP_GATE = 0.05     # m/s of EMA velocity that fully opens displacement-gated gait credit

# --- motion gates (A.5: EMA over the velocity VECTOR, not instantaneous speed) -
VEL_EMA_BETA = 0.04       # per-step; τ ≈ CONTROL_DT/β = 0.5 s smoothing window
MOVE_EPS = 0.1            # m/s of smoothed velocity below which "not moving"

# --- reset diversity (B.4) ----------------------------------------------------
LAUNCH_SPEED = (0.1, 0.5)  # m/s root-velocity range for launch states (RSI-lite;
                           # revisit from DR.5's region-of-attraction if adopted)

# --- commands -----------------------------------------------------------------
VMAX = float(os.environ.get("CMD_VMAX", "1.2"))   # m/s command ceiling; DR.3 (V.8)
                                                  # calibrates this against trajopt envelope

# --- observation layout -------------------------------------------------------
LOCO_OBS = 38             # proprio prefix: 12 qpos + 12 qvel + 4 quat + 6 root vel
                          # + 1 torso z + 3 design; stage-A checkpoint compat pins it.
                          # obs_schema.py describes everything appended after it.
