#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
"""Plain-MuJoCo oracle versus uploaded MuJoCo-Warp model contracts."""

import sys
from pathlib import Path

import mujoco
import numpy as np
import warp as wp

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

from walker_improved import DEFAULTS, build_walker
from walker_warp_env import WalkerWarpEnv


def test_uploaded_model_mass_and_inertia_match_plain_mujoco():
    reference = mujoco.MjModel.from_xml_string(build_walker(DEFAULTS, floor=True))
    env = WalkerWarpEnv(1, seed=0, device="cpu")
    np.testing.assert_allclose(env._wm.body_mass.numpy()[0], reference.body_mass,
                               rtol=1e-6, atol=1e-7)
    np.testing.assert_allclose(env._wm.body_inertia.numpy()[0], reference.body_inertia,
                               rtol=1e-6, atol=1e-7)


def test_uploaded_actuator_gears_match_plain_mujoco():
    reference = mujoco.MjModel.from_xml_string(build_walker(DEFAULTS, floor=True))
    env = WalkerWarpEnv(1, seed=0, device="cpu")
    uploaded = env._wm.actuator_gear.numpy()[0]
    np.testing.assert_allclose(uploaded, reference.actuator_gear, rtol=1e-6, atol=1e-7)
    assert reference.nu == 12


def test_full_command_reaches_same_generalized_force_path():
    model = mujoco.MjModel.from_xml_string(build_walker(DEFAULTS, floor=True))
    data = mujoco.MjData(model)
    data.ctrl[:] = 1.0
    mujoco.mj_forward(model, data)
    actuator_joints = model.actuator_trnid[:, 0]
    dofs = model.jnt_dofadr[actuator_joints]
    expected = np.abs(model.actuator_gear[:, 0])
    np.testing.assert_allclose(np.abs(data.qfrc_actuator[dofs]), expected,
                               rtol=1e-6, atol=1e-7)
