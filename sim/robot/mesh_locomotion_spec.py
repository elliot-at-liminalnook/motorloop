# SPDX-License-Identifier: MIT
"""Backend-independent reward and actuator constants for the mesh robot."""

import os

MESH_VMAX = float(os.environ.get("MESH_VMAX", "0.6"))
TRACK_W = 5.0
TRACK_SIGMA = float(os.environ.get("MESH_TRACK_SIGMA", "0.25"))
PROGRESS_W = 12.0
ALIGN_W = 1.0
BACKWARD_W = 6.0
UPRIGHT_W = 0.5
AIRTIME_W = 1.0
AIRTIME_TARGET = float(os.environ.get("MESH_AIRTIME_TARGET", "1.2"))
AIRTIME_CAP = 2 * AIRTIME_TARGET
ACTRATE_W = 0.05
VELZ_W = 0.5
ANGXY_W = 0.1
POSE_W = 0.15
FOOT_CONTACT_Z = 0.03
FALL_Z = 0.25
MIN_UP_Z = 0.4
CMD_HOLD_STEPS = 80
RESET_NOISE = 0.03
KP = (2.0, 40.0, 6.0)
AUTHORITY_FRAC = 0.6
YAW_AUTHORITY = float(os.environ.get("MESH_YAW_AUTH", str(AUTHORITY_FRAC)))
CLOCK_HZ = float(os.environ.get("MESH_CLOCK_HZ", "0.26"))
