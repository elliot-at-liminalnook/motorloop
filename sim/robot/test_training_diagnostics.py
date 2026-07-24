# SPDX-License-Identifier: MIT
"""Unit contracts for scale-aware training diagnostics."""

from __future__ import annotations

import math

import pytest

torch = pytest.importorskip("torch")

from training_diagnostics import (  # noqa: E402
    checkpoint_replay_comparison,
    checkpoint_replay_tolerances,
    critic_calibration,
    diagnostic_alerts,
    gradient_clip_diagnostics,
    multi_seed_summary,
    normalization_diagnostics,
    normalization_snapshot,
    objective_gradient_diagnostics,
    optimizer_diagnostics,
    parameter_snapshot,
    parameter_update_diagnostics,
    policy_trust_region_diagnostics,
)


class _Actor(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = torch.nn.Linear(2, 2, bias=False)
        self.log_std = torch.nn.Parameter(torch.zeros(2))
        with torch.no_grad():
            self.linear.weight.copy_(torch.eye(2))

    def forward(self, value):
        return self.linear(value)


def test_checkpoint_replay_comparison_is_scale_aware():
    comparison = checkpoint_replay_comparison(
        {"reward": 19.0, "clock": 0.698},
        {"reward": 19.000002, "clock": 0.698020},
        ("reward", "clock"))
    assert not comparison["pass"]
    assert comparison["failed_metrics"] == ["clock"]
    assert comparison["metric_tolerance_ratios"]["reward"] < 1.0
    assert comparison["metric_tolerance_ratios"]["clock"] > 1.0


def test_checkpoint_replay_tolerance_tracks_contact_solver_family():
    mesh = checkpoint_replay_tolerances("mesh")
    ground = checkpoint_replay_tolerances("walker")
    combat = checkpoint_replay_tolerances("combat")
    assert mesh[0] < ground[0] < combat[0]
    assert checkpoint_replay_tolerances("ladder_combat") == combat


class _Norm(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.register_buffer("mean", torch.zeros(2))
        self.register_buffer("var", torch.ones(2))
        self.register_buffer("count", torch.tensor(1.0))
        self.clip = 2.0


def _gaussian_logp(z, mean, log_std):
    return (-0.5 * ((z - mean) / log_std.exp()).square() - log_std).sum(-1)


def test_parameter_update_reports_actual_relative_step():
    module = torch.nn.Linear(2, 1, bias=False)
    with torch.no_grad():
        module.weight.fill_(2.0)
    before = parameter_snapshot(module)
    with torch.no_grad():
        module.weight.add_(0.2)
    result = parameter_update_diagnostics(module, before)
    assert result["relative_update"] == pytest.approx(0.1)
    assert result["largest_relative_updates"][0]["name"] == "weight"


def test_clip_summary_keeps_actor_and_critic_pressure_separate():
    result = gradient_clip_diagnostics([0.5, 2.0], [100.0, 200.0])
    assert result["actor"]["clipped_fraction"] == 0.5
    assert result["critic"]["clipped_fraction"] == 1.0
    assert result["critic"]["scale"]["max"] == pytest.approx(0.01)


def test_optimizer_diagnostics_expose_adam_state_and_step_health():
    module = torch.nn.Linear(2, 1)
    optimizer = torch.optim.Adam(module.parameters(), lr=1.0e-3)
    module(torch.ones((1, 2))).sum().backward()
    optimizer.step()
    result = optimizer_diagnostics(optimizer)
    assert result["state_fraction"] == 1.0
    assert result["nonfinite_state_count"] == 0
    assert result["groups"][0]["step_min"] == 1
    assert not result["groups"][0]["step_desynchronized"]
    assert result["groups"][0]["predicted_effective_step_abs"]["max"] > 0.0


def test_whole_rollout_trust_region_has_ratio_and_ess():
    actor = _Actor()
    obs = torch.tensor([[0.1, -0.2], [0.2, 0.3]])
    z = actor(obs).detach()
    old_logp = _gaussian_logp(z, actor(obs).detach(), actor.log_std.detach())
    result = policy_trust_region_diagnostics(
        actor, obs, z, old_logp, _gaussian_logp, 0.2)
    assert result["approx_kl"] == pytest.approx(0.0, abs=1e-7)
    assert result["effective_sample_fraction"] == pytest.approx(1.0)
    assert result["importance_ratio"]["p50"] == pytest.approx(1.0)


def test_objective_gradients_expose_conflict():
    parameter = torch.nn.Parameter(torch.tensor([1.0, -2.0]))
    forward = parameter.sum()
    backward = -parameter.sum()
    result = objective_gradient_diagnostics(
        {"forward": forward, "backward": backward}, [parameter])
    assert result["norms"]["forward"] == pytest.approx(math.sqrt(2.0))
    assert result["cosines"]["forward__backward"] == pytest.approx(-1.0)


def test_critic_calibration_is_exact_for_perfect_prediction():
    target = torch.tensor([1.0, 2.0, 4.0])
    result = critic_calibration(target, target)
    assert result["rmse"] == 0.0
    assert result["explained_variance"] == pytest.approx(1.0)
    assert result["calibration_slope"] == pytest.approx(1.0)


def test_normalizer_drift_and_outliers_are_visible():
    norm = _Norm()
    before = normalization_snapshot(norm)
    norm.mean.copy_(torch.tensor([1.0, 0.0]))
    norm.var.copy_(torch.tensor([4.0, 1.0]))
    norm.count.fill_(5.0)
    result = normalization_diagnostics(
        before, norm, torch.tensor([[10.0, 0.0], [0.0, 0.0]]))
    assert result["count_after"] == 5.0
    assert result["standardized_mean_drift"]["max"] == pytest.approx(1.0)
    assert result["current_outlier_fraction"] > 0.0


def test_seed_uncertainty_and_alerts_are_machine_readable():
    summary = multi_seed_summary(
        [{"x": 0.0}, {"x": 1.0}, {"x": 2.0}], ["x"])
    assert summary["metrics"]["x"]["mean"] == 1.0
    alerts = diagnostic_alerts({
        "integrity": {"nonfinite_count": 1},
        "trust_region": {"clip_fraction": 0.5,
                         "effective_sample_fraction": 0.5},
        "gradient_clipping": {
            "actor": {"clipped_fraction": 1.0},
            "critic": {"clipped_fraction": 1.0},
        },
        "checkpoint_replay": {"pass": False},
    })
    codes = {row["code"] for row in alerts}
    assert {"nonfinite", "ppo_clip_pressure", "low_effective_sample_size",
            "actor_always_clipped", "critic_always_clipped",
            "checkpoint_replay_mismatch"} <= codes


def test_saturated_adaptive_contracts_are_not_silently_treated_as_self_tuning():
    alerts = diagnostic_alerts({
        "integrity": {}, "trust_region": {}, "gradient_clipping": {},
        "adaptive_contracts": {
            "dual_max": 10.0,
            "constraints": [{
                "name": "cat_slip", "observed": 0.02,
                "target": 0.001, "comparison": "<=", "dual": 10.0,
            }],
            "competence": [{
                "name": "ladder_foot_activity", "observed": 0.20,
                "target": 0.40, "comparison": ">=", "dual": 10.0,
            }],
        },
    })
    by_code = {row["code"]: row for row in alerts}
    assert by_code["adaptive_constraint_saturated"]["value"]["name"] \
        == "cat_slip"
    assert by_code["adaptive_competence_saturated"]["value"]["name"] \
        == "ladder_foot_activity"


def test_adaptive_metric_window_disagreement_is_not_hidden():
    alerts = diagnostic_alerts({
        "integrity": {}, "trust_region": {}, "gradient_clipping": {},
        "adaptive_contracts": {
            "dual_max": 10.0,
            "constraints": [],
            "competence": [{
                "name": "ladder_foot_activity", "observed": 0.18,
                "target": 0.40, "comparison": ">=", "dual": 1.0,
            }],
        },
    }, evaluation={"ladder_foot_activity": 0.42})
    row = next(row for row in alerts
               if row["code"] == "adaptive_metric_window_mismatch")
    assert row["value"]["training"] == pytest.approx(0.18)
    assert row["value"]["evaluation"] == pytest.approx(0.42)


def test_scaffold_dominance_is_reported_as_an_objective_design_problem():
    alerts = diagnostic_alerts({
        "integrity": {}, "trust_region": {}, "gradient_clipping": {},
    }, evaluation={
        "reward_role_shares": {
            "outcome": 0.20, "constraint": 0.10,
            "efficiency": 0.05, "scaffold": 0.65,
        },
    })
    assert "scaffold_reward_dominance" in {row["code"] for row in alerts}


def test_failed_gate_lowers_reward_component_dominance_tripwire():
    alerts = diagnostic_alerts({
        "integrity": {}, "trust_region": {}, "gradient_clipping": {},
    }, gates={"all_pass": False, "worst_metric": "xprogress"}, evaluation={
        "reward_components": {
            "tracking": {"absolute_mean_share": 0.78},
            "direct_progress": {"absolute_mean_share": 0.22},
        },
    })
    row = next(row for row in alerts
               if row["code"] == "reward_component_dominance")
    assert row["value"] == pytest.approx(0.78)
    assert "60%" in row["message"]


def test_physics_actuator_and_per_leg_alerts_surface_real_world_failures():
    alerts = diagnostic_alerts({
        "integrity": {},
        "trust_region": {},
        "gradient_clipping": {},
    }, evaluation={
        "duty_fl": 0.99, "duty_fr": 0.99, "duty_rl": 0.99, "duty_rr": 0.99,
        "actuator_diagnostics": {
            "effort_saturated": {"overall": {"mean": 0.30}},
        },
        "simulation_diagnostics": {
            "constraint_rows": {"p99": 95.0},
            "constraint_capacity": {"p10": 100.0},
            "state_nonfinite": {"max": 1.0},
        },
        "termination_ledger": {"fall": {"rate": 0.02}},
    })
    codes = {row["code"] for row in alerts}
    assert {"all_legs_planted", "actuator_effort_saturation",
            "solver_constraint_capacity", "simulation_state_nonfinite",
            "frequent_falls"} <= codes
