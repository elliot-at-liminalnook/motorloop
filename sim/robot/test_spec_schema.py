#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
"""Spec-validation gate tests (plan V.2): robot.toml is typed-checked at model build.

The real robot.toml must validate clean, and each class of nonsense spec must die
with a NAMED error (the field in the message) instead of surfacing mid-training:
zero gear, a two-way knee, a DR bracket that misses its center, a spawn height
below the kinematic stance.
"""
from __future__ import annotations

import sys
from pathlib import Path

import pytest

try:
    import tomllib                       # py3.11+ stdlib
except ModuleNotFoundError:              # py3.10: pip install tomli
    import tomli as tomllib

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

from pydantic import ValidationError  # noqa: E402

from spec_schema import validate_spec  # noqa: E402


def _spec() -> dict:
    with open(HERE / "robot.toml", "rb") as f:
        return tomllib.load(f)


def test_current_robot_toml_validates_clean():
    spec = _spec()
    out = validate_spec(spec)
    assert out is spec               # gate returns the ORIGINAL dict, untouched
    assert out == _spec()            # ... and does not mutate it


def test_zero_gear_fails_with_named_error():
    spec = _spec()
    spec["actuator"]["gear"] = 0.0
    with pytest.raises(ValidationError) as ei:
        validate_spec(spec)
    assert "gear" in str(ei.value)


def test_two_way_knee_range_fails():
    spec = _spec()
    spec["leg_defaults"]["knee_range"] = [-2.6, 0.4]
    with pytest.raises(ValidationError) as ei:
        validate_spec(spec)
    assert "knee_range" in str(ei.value)


def test_dr_bracket_must_contain_center():
    spec = _spec()
    # leg_defaults.joint_stiffness center is 0.5; a [1.0, 2.0] bracket misses it.
    assert spec["leg_defaults"]["joint_stiffness"] == 0.5
    spec["domain_randomization"]["joint_stiffness"] = [1.0, 2.0]
    with pytest.raises(ValidationError) as ei:
        validate_spec(spec)
    assert "joint_stiffness" in str(ei.value)


def test_spawn_below_kinematic_stance_fails():
    spec = _spec()
    spec["torso"]["spawn_height"] = 0.1
    with pytest.raises(ValidationError) as ei:
        validate_spec(spec)
    assert "spawn_height" in str(ei.value)


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-q"]))
