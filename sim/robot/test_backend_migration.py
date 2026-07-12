# SPDX-License-Identifier: MIT
"""Contracts for legacy commands converted to the MuJoCo-Warp stack."""

from __future__ import annotations

import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

from drake_walk_proof import (  # noqa: E402
    CRAWL_ORDER,
    _build_model,
    certify_keyframe,
    footholds_for,
)
from gen_robot_drake import build_model, hinge_effort_limits  # noqa: E402
from gen_robot_mjcf import load_spec  # noqa: E402
from reference_gait_gen import generate  # noqa: E402
from walker_improved import DEFAULTS  # noqa: E402


def test_legacy_model_inspector_compiles_production_mujoco():
    model = build_model(load_spec(HERE / "robot.toml"))
    assert model.nu == 12
    assert len(hinge_effort_limits(model)) == 12


def test_reference_export_validates_all_twelve_targets():
    reference, targets = generate()
    assert targets.shape == (reference["n"], 12)


def test_compiled_walker_passes_every_single_support_check():
    model = _build_model(DEFAULTS)
    footholds = footholds_for(DEFAULTS)
    for swing_leg in CRAWL_ORDER:
        result = certify_keyframe(model, DEFAULTS, swing_leg, footholds)
        assert result["ik_ok"] and result["stable"] and result["tau_ok"]
