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


def test_mesh_evaluation_allows_quantile_roundoff_but_not_behavioral_drift():
    atol, rtol = METRIC_TOLERANCES["mesh"]["evals"]
    assert not _metric_failures([{"p50": 0.1}], [{"p50": 0.100008}], atol, rtol)
    assert _metric_failures([{"p50": 0.1}], [{"p50": 0.10002}], atol, rtol)


def test_combat_metric_bounds_allow_roundoff_but_reject_behavioral_drift():
    atol, rtol = METRIC_TOLERANCES["combat"]["evals"]
    assert not _metric_failures([{"reward": 0.4}], [{"reward": 0.4008}], atol, rtol)
    assert _metric_failures([{"reward": 0.4}], [{"reward": 0.41}], atol, rtol)


def test_artifact_hashes_are_provenance_not_repeatability_metrics():
    atol, rtol = METRIC_TOLERANCES["ground"]["evals"]
    first = [{"reward": 0.4, "wall_seconds": 1.0,
              "env_steps_per_second": 100.0,
              "learner": {"v_loss": 64.0},
              "checkpoint_sha256": "first-artifact",
              "diagnostics": {"solver_iterations_std": 0.75},
              "simulation_diagnostics": {"solver_iterations": {"mean": 1.4}},
              "checkpoint_replay": {"before_sha256": "first-trajectory",
                                     "max_tolerance_ratio": 0.06,
                                     "metric_tolerance_ratios": {"reward": 0.04},
                                     "pass": True}}]
    second = [{"reward": 0.4, "wall_seconds": 9.0,
               "env_steps_per_second": 20.0,
               "learner": {"v_loss": 67.0},
               "checkpoint_sha256": "second-artifact",
               "diagnostics": {"solver_iterations_std": 0.77},
               "simulation_diagnostics": {"solver_iterations": {"mean": 1.1}},
               "checkpoint_replay": {"before_sha256": "second-trajectory",
                                      "max_tolerance_ratio": 0.17,
                                      "metric_tolerance_ratios": {"reward": 0.12},
                                      "pass": True}}]
    assert not _metric_failures(first, second, atol, rtol)


def test_combat_policy_tensors_are_tighter_than_contact_normalizers():
    policy_tol = _tensor_tolerance("combat", "actor", "trunk.0.weight")
    variance_tol = _tensor_tolerance("combat", "obs_norm", "var")
    assert policy_tol[0] < variance_tol[0]
    baseline = torch.ones(8)
    assert torch.allclose(baseline, baseline + 1e-3, atol=policy_tol[0], rtol=policy_tol[1])
    assert not torch.allclose(baseline, baseline + 0.2, atol=policy_tol[0], rtol=policy_tol[1])
