# SPDX-License-Identifier: MIT
"""Unit contracts for target-GPU repeatability acceptance bounds."""

from __future__ import annotations

import sys
from pathlib import Path

import torch

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

from gpu_determinism_canary import (  # noqa: E402
    METRIC_TOLERANCES,
    _metric_failures,
    _tensor_tolerance,
)


def test_non_contact_metric_bounds_remain_near_bitwise():
    atol, rtol = METRIC_TOLERANCES["mesh"]["updates"]
    assert not _metric_failures([{"loss": 100.0}], [{"loss": 100.001}], atol, rtol)
    assert _metric_failures([{"loss": 100.0}], [{"loss": 100.02}], atol, rtol)


def test_combat_metric_bounds_allow_roundoff_but_reject_behavioral_drift():
    atol, rtol = METRIC_TOLERANCES["combat"]["evals"]
    assert not _metric_failures([{"reward": 0.4}], [{"reward": 0.4008}], atol, rtol)
    assert _metric_failures([{"reward": 0.4}], [{"reward": 0.41}], atol, rtol)


def test_combat_policy_tensors_are_tighter_than_contact_normalizers():
    policy_tol = _tensor_tolerance("combat", "actor", "trunk.0.weight")
    variance_tol = _tensor_tolerance("combat", "obs_norm", "var")
    assert policy_tol[0] < variance_tol[0]
    baseline = torch.ones(8)
    assert torch.allclose(baseline, baseline + 1e-3, atol=policy_tol[0], rtol=policy_tol[1])
    assert not torch.allclose(baseline, baseline + 0.2, atol=policy_tol[0], rtol=policy_tol[1])
