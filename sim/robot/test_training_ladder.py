# SPDX-License-Identifier: MIT
"""Executable training-ladder manifest and environment contracts."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
torch = pytest.importorskip("torch")
pytest.importorskip("mujoco_warp")

from ladder_warp_env import (GAIT_PERIOD_STEPS, LadderCombatWarpEnv,
                             LadderLocomotionWarpEnv,
                             normalized_duty_costs,
                             swing_clearance_scores)  # noqa: E402
from train_mesh_warp import (action_prior_weight, adaptive_ppo_learning_rate,
                             clip_actor_critic_gradients,
                             duty_stagnation_tripwire_enabled,
                             early_gates_pass, evaluation_trends, gate_diagnostics,
                             incremental_eval_interval,
                             kl_epoch_should_stop,
                             parse_early_gates,
                             prior_competence_pressure)  # noqa: E402
from training_ladder import (LadderRunner, RUNGS, Gate, make_parser,
                             merge_candidate_archives, validate_manifest)  # noqa: E402


def test_manifest_is_the_complete_html_ladder():
    validate_manifest()
    assert len(RUNGS) == 31
    assert [r.number for r in RUNGS] == list(range(1, 32))
    assert RUNGS[25].slug == "commanded_leg_kick"
    assert RUNGS[-1].slug == "codesign_loop"


def test_gate_directions_are_explicit():
    assert Gate("score", ">=", 0.5).passes({"score": 0.5})
    assert not Gate("score", ">=", 0.5).passes({"score": 0.49})
    assert Gate("error", "<=", 0.2).passes({"error": 0.1})
    assert not Gate("missing", "<=", 1).passes({})
    early = parse_early_gates(["up,>=,0.85", "catrate,<=,0.001"])
    assert early_gates_pass(early, {"up": 0.9, "catrate": 0.0})
    assert not early_gates_pass(early, {"up": 0.9, "catrate": 0.01})


def test_ladder_promotion_uses_worst_deterministic_seed(tmp_path):
    args = make_parser().parse_args(["run", "--out", str(tmp_path)])
    runner = LadderRunner(args)
    rung = RUNGS[5]
    metrics = {
        "duty": 0.80,
        "foot_cycle_duty": 0.86,
        "speed": 0.06,
        "ladder_step_clock": 0.71,
        "ladder_swing_clearance": 0.63,
        "up": 1.0,
        "catrate": 0.0001,
        "fallrate": 0.0,
        "diagnostics": {"multi_seed_evaluation": {"metrics": {
            # The means pass both contracts, but one held-out seed on each
            # boundary does not.  Promotion must preserve those failures.
            "ladder_step_clock": {"values": [0.6995, 0.7005, 0.7006]},
            "catrate": {"values": [0.0001, 0.0011, 0.0002]},
        }}},
    }

    passed, details = runner._gate(rung, metrics)

    assert not passed
    assert any("FAIL ladder_step_clock" in row and "worst of 3" in row
               for row in details)
    assert any("FAIL catrate" in row and "worst of 3" in row
               for row in details)


def test_ladder_gate_falls_back_to_standalone_fixed_seed(tmp_path):
    args = make_parser().parse_args(["run", "--out", str(tmp_path)])
    runner = LadderRunner(args)
    gate = Gate("score", ">=", 0.7)

    value, source, values = runner._gate_observation(gate, {"score": 0.8})

    assert value == pytest.approx(0.8)
    assert source == "direct fixed-seed observation"
    assert values == [0.8]


def test_gate_diagnostics_expose_normalized_headroom_and_learning_slopes():
    gates = parse_early_gates(["xprogress,>=,0.1", "catrate,<=,0.001"])
    report = gate_diagnostics(gates, {"xprogress": 0.04, "catrate": 0.0002})
    assert not report["all_pass"]
    assert report["worst_metric"] == "xprogress"
    checks = {row["metric"]: row for row in report["checks"]}
    assert checks["xprogress"]["relative_margin"] == pytest.approx(-0.6)
    assert checks["catrate"]["relative_margin"] == pytest.approx(0.8)

    trend = evaluation_trends(
        {"step": 1_000_000, "xprogress": 0.02, "catrate": 0.0004},
        {"xprogress": 0.05, "catrate": 0.0001},
        1_500_000,
    )
    assert trend["xprogress_per_million"] == pytest.approx(0.06)
    assert trend["catrate_per_million"] == pytest.approx(-0.0006)


def test_resumed_eval_schedule_covers_only_new_experience():
    rollout = 128 * 64
    interval = incremental_eval_interval(12_001_280, 14_000_000, 4, rollout)
    assert interval == 499_712
    assert interval < 14_000_000 // 4
    assert interval % rollout == 0
    assert incremental_eval_interval(99, 100, 4, rollout) == rollout


def test_interrupted_ladder_attempt_resumes_at_next_durable_target(tmp_path):
    tag = tmp_path / "rung_06_step_in_place"
    Path(str(tag) + ".stats.json").write_text(json.dumps({"evals": [
        {"step": 8_822_784},
        {"step": 10_002_432},
    ]}))

    completed = LadderRunner._durable_completed_attempts(
        tag, base_steps=2_000_000, recorded_attempts=6)

    assert completed == 5
    assert 2_000_000 * (completed + 1) == 12_000_000


def test_resume_retries_recorded_attempt_when_legacy_stats_are_missing(tmp_path):
    completed = LadderRunner._durable_completed_attempts(
        tmp_path / "missing", base_steps=2_000_000, recorded_attempts=6)

    assert completed == 5


def test_actor_and_critic_gradient_clipping_are_disjoint():
    actor = torch.nn.Linear(2, 1, bias=False)
    critic = torch.nn.Linear(2, 1, bias=False)
    actor.weight.grad = torch.ones_like(actor.weight)
    critic.weight.grad = torch.full_like(critic.weight, 1_000.0)

    actor_norm, critic_norm = clip_actor_critic_gradients(actor, critic)

    assert float(actor_norm) == pytest.approx(2.0 ** 0.5)
    assert float(critic_norm) == pytest.approx(2.0 ** 0.5 * 1_000.0)
    assert float(actor.weight.grad.norm()) == pytest.approx(1.0)
    assert float(critic.weight.grad.norm()) == pytest.approx(1.0)


def test_kl_controller_stops_oversized_epochs_and_self_tunes_lr():
    assert not kl_epoch_should_stop(0.029, 0.02, 1.5)
    assert kl_epoch_should_stop(0.031, 0.02, 1.5)
    assert adaptive_ppo_learning_rate(3.0e-4, 3.0e-4, 0.05, 0.02) \
        == pytest.approx(1.5e-4)
    assert adaptive_ppo_learning_rate(1.5e-4, 3.0e-4, 0.005, 0.02) \
        == pytest.approx(1.65e-4)
    assert adaptive_ppo_learning_rate(3.0e-4, 3.0e-4, 0.02, 0.02) \
        == pytest.approx(3.0e-4)


def test_changed_gate_contract_reaudits_and_invalidates_dependent_prefix(
        tmp_path, monkeypatch):
    args = make_parser().parse_args(["run", "--out", str(tmp_path), "--resume"])
    runner = LadderRunner(args)
    checkpoints = {}
    for number in (2, 3):
        path = tmp_path / f"accepted_{number}.pt"
        path.touch()
        checkpoints[str(number)] = str(path)
    runner.state.update(completed=[2, 3], checkpoints=checkpoints,
                        metrics={}, attempts={"2": 1, "3": 1},
                        retention_baselines={})
    invalidated_path = Path(checkpoints["3"])

    def evaluate(rung, checkpoint, label):
        metrics = {"up": 0.99 if rung.number == 2 else 0.10,
                   "speed": 0.01, "catrate": 0.0, "fallrate": 0.0}
        return 0, metrics

    monkeypatch.setattr(runner, "_evaluate_checkpoint", evaluate)
    runner._audit_completed_contracts()
    assert runner.state["completed"] == [2]
    assert "3" not in runner.state["checkpoints"]
    assert not invalidated_path.exists()
    assert list((tmp_path / "diagnostics").glob("contract_invalidated_rung_03_*.pt"))
    assert runner.state["failed"]["rung"] == 3
    assert runner.state["retention_baselines"]["2"]["fallrate"] == 0.0


def test_accepting_relearned_rung_archives_stale_successor_candidates(tmp_path):
    args = make_parser().parse_args(["run", "--out", str(tmp_path), "--resume"])
    runner = LadderRunner(args)
    runner.state.update(
        completed=[2, 3, 4, 5],
        attempts={"6": 4, "7": 17, "8": 2},
        metrics={"7": {"xprogress": 0.12}, "8": {"xprogress": 0.14}},
        checkpoints={}, retention_baselines={},
    )
    stale_seven = tmp_path / "rung_07_walk_forward.pt"
    stale_seven_best = tmp_path / "rung_07_walk_forward.best.pt"
    stale_eight = tmp_path / "rung_08_velocity_tracking.pt"
    for path in (stale_seven, stale_seven_best, stale_eight):
        path.touch()
    prior = tmp_path / "priors" / "rung_07_walk_prior.json"
    prior.parent.mkdir()
    prior.write_text("{}")

    archived = runner._quarantine_stale_successor_candidates(6)

    assert len(archived) == 4
    assert not stale_seven.exists()
    assert not stale_seven_best.exists()
    assert not stale_eight.exists()
    assert not prior.exists()
    assert runner.state["attempts"] == {"6": 4}
    assert runner.state["metrics"] == {}
    assert all(Path(path).parent == tmp_path / "diagnostics" for path in archived)


def test_regression_matrix_persists_current_and_replayed_skill_margins(
        tmp_path, monkeypatch):
    args = make_parser().parse_args(["run", "--out", str(tmp_path)])
    runner = LadderRunner(args)
    learned, candidate = RUNGS[1], RUNGS[2]
    checkpoint = tmp_path / "candidate.pt"
    checkpoint.touch()
    runner.state.update(
        completed=[learned.number],
        checkpoints={str(learned.number): str(tmp_path / "accepted.pt")},
        metrics={}, attempts={str(candidate.number): 1},
        retention_baselines={str(learned.number): {
            "up": 0.95, "speed": 0.10, "catrate": 0.0, "fallrate": 0.0,
        }},
    )
    candidate_metrics = {"up": 0.90, "catrate": 0.0, "fallrate": 0.0}
    candidate_checks = runner._current_task_regression_checks(
        candidate, candidate_metrics)
    runner._record_regression_task(
        candidate, str(checkpoint), candidate, candidate_checks,
        True, "candidate_task")

    replay_metrics = {
        "up": 0.90, "speed": 0.12, "catrate": 0.0, "fallrate": 0.0,
    }
    monkeypatch.setattr(
        runner, "_evaluate_checkpoint",
        lambda rung, path, label: (0, replay_metrics))
    passed, report = runner._retention_gate(candidate, str(checkpoint))

    assert passed and report[0]["pass"]
    matrix = json.loads((tmp_path / "regression_matrix.json").read_text())
    row = next(iter(matrix["candidates"].values()))
    assert set(row["tasks"]) == {"2", "3"}
    up = next(check for check in row["tasks"]["2"]["checks"]
              if check["metric"] == "up")
    assert up["baseline"] == pytest.approx(0.95)
    assert up["current"] == pytest.approx(0.90)
    assert up["acceptance_boundary"] == pytest.approx(0.85)
    assert up["margin"] == pytest.approx(0.05)
    assert row["all_pass"]


def test_walk_prior_retargets_itself_from_gate_pressure(tmp_path):
    args = make_parser().parse_args(["run", "--out", str(tmp_path), "--resume"])
    runner = LadderRunner(args)
    prior = tmp_path / "priors" / "rung_07_walk_prior.json"
    prior.parent.mkdir()
    fast = {"generation": 1, "x": 0.16, "safe_x": 0.15, "cat": 0.02,
            "cat_slip": 0.02, "fall": 0.0, "parameters": [0.1] * 24}
    safe = {"generation": 2, "x": 0.11, "safe_x": 0.10, "cat": 0.004,
            "cat_slip": 0.003, "fall": 0.0, "parameters": [0.2] * 24}
    prior.write_text(json.dumps({"pareto": [fast, safe], "best": fast}))
    runner._retarget_walk_prior({"xprogress": 0.05, "catrate": 0.02})
    artifact = json.loads(prior.read_text())
    assert artifact["best"]["generation"] == 2
    assert artifact["selection_history"][-1]["reason"] == "constraint pressure"


def test_walk_prior_archive_keeps_distinct_old_tradeoffs():
    old = {"pareto": [{"parameters": [0.1] * 24, "x": 0.16}]}
    refined = {"history": [
        {"parameters": [0.1] * 24, "x": 0.16},
        {"parameters": [0.2] * 24, "x": 0.03},
    ]}
    archive = merge_candidate_archives(old, refined)
    assert [candidate["x"] for candidate in archive] == [0.16, 0.03]


def test_static_ladder_rungs_do_not_trigger_the_locomotion_duty_tripwire():
    for rung in range(2, 24):
        assert not duty_stagnation_tripwire_enabled("ladder_locomotion", rung)
    assert duty_stagnation_tripwire_enabled("walker", None)


@pytest.mark.parametrize("rung,period,fire", ((3, 140, 60), (15, 100, 45),
                                               (22, 160, 80)))
def test_scripted_push_has_a_narrow_slip_constraint_grace(rung, period, fire):
    env = LadderLocomotionWarpEnv(2, rung=rung, seed=5, device="cpu", episode_length=3)
    ones = torch.ones(2)
    contacts = torch.ones((2, 4))
    fast_feet = torch.ones((2, 4))
    zeros = torch.zeros(2)
    zero_feet = torch.zeros((2, 4))

    def slip_at(step):
        env._task_t.fill_(step)
        terms, _ = env._cat_violations(
            contacts, fast_feet, ones, zeros, zeros,
            torch.ones(2, dtype=torch.bool), torch.ones(2, dtype=torch.bool),
            zeros, zeros, zero_feet)
        return terms["cat_slip"]

    assert bool((slip_at(fire - 1) > 0.0).all())
    assert bool((slip_at(fire) == 0.0).all())
    assert bool((slip_at(fire + 3) == 0.0).all())
    assert bool((slip_at(fire + 4) > 0.0).all())


def test_step_rung_exposes_only_verified_lift_action_prior(tmp_path):
    env = LadderLocomotionWarpEnv(2, rung=6, seed=7, device="cpu", episode_length=3)
    target, mask = env.policy_mean_prior(env.observe())
    assert target.shape == mask.shape == (2, 12)
    assert torch.equal(mask[:, 2::3], torch.ones((2, 4)))
    assert torch.count_nonzero(mask[:, [0, 1, 3, 4, 6, 7, 9, 10]]) == 0
    assert torch.allclose(target[:, 2::3].abs(), torch.full((2, 4), 1.85))
    standing = LadderLocomotionWarpEnv(1, rung=5, seed=7, device="cpu", episode_length=3)
    assert standing.policy_mean_prior(standing.observe()) is None

    moving = LadderLocomotionWarpEnv(2, rung=7, seed=7, device="cpu", episode_length=3)
    moving._set_task_command(0.0)
    assert torch.allclose(moving._cmd[:, 0], torch.full((2,), 0.10))
    moving._set_task_command(1.0)
    assert torch.allclose(moving._cmd[:, 0], torch.full((2,), 0.32))
    moving._constraint_duals.fill_(3.0)
    moving._set_task_command(1.0)
    assert torch.allclose(moving._cmd[:, 0], torch.full((2,), 0.155))
    moving._constraint_duals.zero_()
    moving_obs = moving.observe()
    moving_target, moving_mask = moving.policy_mean_prior(
        moving_obs, torch.zeros((2, moving.act_dim)))
    assert torch.equal(moving_mask[:, 2::3], torch.ones((2, 4)))
    assert torch.equal(moving_mask[:, 0::3], torch.ones((2, 4)))
    assert torch.equal(moving_mask[:, 1::3], torch.ones((2, 4)))
    assert torch.isfinite(moving_target).all()
    prior = tmp_path / "walk_prior.json"
    prior.write_text(json.dumps({"blend": 0.55, "best": {
        "parameters": [0.1] + [0.0] * 23}}))
    moving.configure_action_prior(prior)
    transfer = torch.full((2, moving.act_dim), 0.4)
    transferred_target, _ = moving.policy_mean_prior(
        moving_obs, torch.zeros((2, moving.act_dim)), transfer_action=transfer)
    transferred_action = torch.tanh(transferred_target)
    assert not torch.allclose(transferred_action[:, 0::3], torch.full((2, 4), 0.26))
    assert not torch.allclose(transferred_action[:, 1::3], torch.full((2, 4), 0.26))
    assert bool((transferred_action[:, :].abs() <= 0.950001).all())
    assert bool((transferred_action[:, 2::3].abs() <= 0.950001).all())
    assert torch.allclose(transferred_action[:, 2], transferred_action[:, 11])
    assert torch.allclose(transferred_action[:, 5], transferred_action[:, 8])
    assert torch.allclose(transferred_action[:, 2], -transferred_action[:, 5])

    moving._task_t.fill_(24)
    moving._set_task_command(1.0)
    before = torch.tanh(moving.policy_mean_prior(
        moving.observe(), torch.zeros((2, moving.act_dim)), transfer_action=transfer)[0])
    moving._task_t.fill_(25)
    moving._set_task_command(1.0)
    after = torch.tanh(moving.policy_mean_prior(
        moving.observe(), torch.zeros((2, moving.act_dim)), transfer_action=transfer)[0])
    assert float((after[:, 2::3] - before[:, 2::3]).abs().max()) < 0.25
    moving._constraint_duals.copy_(torch.tensor([0.0, 0.0, 3.0]))
    _, pressure_mask = moving.policy_mean_prior(
        moving.observe(), torch.zeros((2, moving.act_dim)), transfer_action=transfer)
    # Physical duty pressure must not make a possibly lagging open-loop clock
    # increasingly rigid; the robust contact objective owns final timing.
    assert torch.allclose(pressure_mask[:, 2::3], torch.ones((2, 4)))
    assert bool((pressure_mask[:, 2::3] > pressure_mask[:, 0::3]).all())
    assert float(moving.action_prior_suppression_pressure) == 0.0
    moving._constraint_duals.copy_(torch.tensor([3.0, 0.0, 0.0]))
    _, slip_pressure_mask = moving.policy_mean_prior(
        moving.observe(), torch.zeros((2, moving.act_dim)), transfer_action=transfer)
    assert torch.allclose(slip_pressure_mask[:, 2::3], torch.ones((2, 4)))
    assert float(moving.action_prior_suppression_pressure) == 3.0
    moving.update_constraint_duals(torch.tensor([0.002, 0.0, 0.0]))
    assert float(moving.constraint_duals[0]) > 0.0


def test_constraint_dual_controller_is_scale_normalized():
    env = LadderLocomotionWarpEnv(1, rung=7, seed=9, device="cpu", episode_length=3)

    def response(observed: float) -> float:
        env._constraint_duals.zero_()
        env._constraint_error_square.zero_()
        for _ in range(100):
            env.update_constraint_duals(torch.tensor([observed, 0.0, 0.0]))
        return float(env.constraint_duals[0])

    small_violation = response(0.002)
    large_violation = response(0.020)
    # A tenfold transient changes the normalizer, not the long-run controller
    # timescale. Both persistent violations therefore create comparable pressure.
    assert 0.75 <= large_violation / small_violation <= 1.25

    before_recovery = float(env.constraint_duals[0])
    for _ in range(100):
        env.update_constraint_duals(torch.tensor([0.0, 0.0, 0.0]))
    assert float(env.constraint_duals[0]) < 0.01 * before_recovery

    env._constraint_duals.zero_()
    for _ in range(20):
        env.update_constraint_duals(torch.tensor([0.0, 0.02, 0.02]))
    assert bool((env.constraint_duals[1:] > 0.0).all())

    for _ in range(100):
        env.update_competence_duals(torch.tensor([0.02]))
    competence_pressure = float(env.competence_duals[0])
    assert competence_pressure > 0.0
    for _ in range(100):
        env.update_competence_duals(torch.tensor([0.12]))
    assert float(env.competence_duals[0]) < 0.01 * competence_pressure


def test_step_clock_competence_pressure_self_tunes_to_rung_six_gate():
    env = LadderLocomotionWarpEnv(1, rung=6, seed=9, device="cpu", episode_length=3)
    assert env.adaptive_competence_names == ("ladder_step_clock",)
    assert float(env._competence_targets[0]) == pytest.approx(0.70)

    for _ in range(100):
        env.update_competence_duals(torch.tensor([0.60]))
    shortfall_pressure = float(env.competence_duals[0])
    assert shortfall_pressure > 0.0

    for _ in range(100):
        env.update_competence_duals(torch.tensor([0.75]))
    assert float(env.competence_duals[0]) < 0.01 * shortfall_pressure


def test_per_foot_duty_cost_credits_each_successful_lift():
    all_planted = {
        "duty_ema": torch.tensor([1.0]),
        "foot_duty_ema_by_leg": torch.ones((1, 4)),
    }
    half_lifted = {
        "duty_ema": torch.tensor([0.95]),
        # The worst-foot value is still one, but two independently successful
        # lifts must halve the dense learning cost.
        "foot_duty_ema_by_leg": torch.tensor([[1.0, 1.0, 0.90, 0.90]]),
    }
    every_foot_lifted = {
        "duty_ema": torch.tensor([0.90]),
        "foot_duty_ema_by_leg": torch.full((1, 4), 0.90),
    }

    aggregate, feet = normalized_duty_costs(all_planted)
    assert float(aggregate) == pytest.approx(1.0)
    assert float(feet) == pytest.approx(1.0)
    aggregate, feet = normalized_duty_costs(half_lifted)
    assert float(aggregate) == pytest.approx(0.5)
    assert float(feet) == pytest.approx(0.5)
    _, timely_debt = normalized_duty_costs(
        half_lifted, swing_clearance=torch.tensor([0.25]))
    assert float(timely_debt) == pytest.approx(0.75)
    aggregate, feet = normalized_duty_costs(every_foot_lifted)
    assert float(aggregate) == pytest.approx(0.0)
    assert float(feet) == pytest.approx(0.0)


def test_worst_swing_clearance_closes_half_gait_loophole():
    desired = torch.tensor([[1.0, 0.0, 0.0, 1.0]])
    # FL fully clears while the other required diagonal foot remains planted.
    foot_z = torch.tensor([[0.050, 0.014, 0.014, 0.014]])
    mean, worst = swing_clearance_scores(foot_z, desired, foot_radius=0.012)
    assert float(mean) == pytest.approx(0.5)
    assert float(worst) == pytest.approx(0.0)

    foot_z[:, 3] = 0.050
    mean, worst = swing_clearance_scores(foot_z, desired, foot_radius=0.012)
    assert float(mean) == pytest.approx(1.0)
    assert float(worst) == pytest.approx(1.0)


def test_ladder_constraint_grace_uses_episode_age_not_command_timer():
    env = LadderLocomotionWarpEnv(1, rung=7, seed=10, device="cpu", episode_length=100)
    cf = torch.ones((1, 4))
    foot_hspeed = torch.zeros_like(cf)
    scalar = torch.ones(1)
    duty = torch.full((1,), 0.99)
    foot_duty = torch.full((1, 4), 0.99)

    def violations():
        terms, _ = env._cat_violations(
            cf, foot_hspeed, scalar, torch.full((1,), 0.25), scalar,
            # Opposite values prove the override owns grace maturity.
            torch.ones(1, dtype=torch.bool), torch.ones(1, dtype=torch.bool),
            torch.full((1,), 0.10), duty, foot_duty)
        return terms

    env._timer.zero_()
    env._constraint_age.zero_()
    assert float(violations()["cat_foot_duty"]) == 0.0

    env._constraint_age.fill_(env.duty_constraint_grace_steps - 1)
    assert float(violations()["cat_foot_duty"]) == 0.0
    env._constraint_age.fill_(env.duty_constraint_grace_steps)
    assert float(violations()["cat_foot_duty"]) > 0.0
    assert float(violations()["cat_duty"]) > 0.0

    env.reset()
    assert int(env._constraint_age.item()) == 0


def test_step_in_place_enforces_duty_without_translation_command():
    env = LadderLocomotionWarpEnv(1, rung=6, seed=11, device="cpu", episode_length=100)
    assert env.adaptive_constraint_names == (
        "cat_slip", "cat_duty", "cat_foot_duty")
    env._constraint_age.fill_(env.duty_constraint_grace_steps)
    cf = torch.ones((1, 4))
    scalar = torch.ones(1)
    terms, _ = env._cat_violations(
        cf, torch.zeros_like(cf), scalar, torch.zeros(1), torch.zeros(1),
        torch.zeros(1, dtype=torch.bool), torch.ones(1, dtype=torch.bool),
        torch.zeros(1), torch.ones(1), torch.ones((1, 4)))
    assert float(terms["cat_progress"]) == 0.0
    assert float(terms["cat_duty"]) == pytest.approx(1.0)
    assert float(terms["cat_foot_duty"]) == pytest.approx(1.0)


def test_cycle_duty_treats_early_and_late_swing_pairs_equally():
    env = LadderLocomotionWarpEnv(1, rung=6, seed=12, device="cpu", episode_length=100)
    env._constraint_age.fill_(env.duty_constraint_grace_steps)
    scalar = torch.ones(1)
    terms = None
    # Pair A swings for the first half and pair B for the second. Every foot has
    # identical 50% cycle duty even though an endpoint EMA would favor pair B.
    for step in range(GAIT_PERIOD_STEPS):
        contact = (1.0 - (env._pair_a if step < GAIT_PERIOD_STEPS // 2
                          else env._pair_b))[None, :]
        terms, _ = env._cat_violations(
            contact, torch.zeros_like(contact), scalar, torch.zeros(1),
            torch.zeros(1), torch.zeros(1, dtype=torch.bool),
            torch.ones(1, dtype=torch.bool), torch.zeros(1),
            torch.ones(1), torch.ones((1, 4)))
    assert terms is not None
    assert torch.allclose(terms["cycle_duty_by_leg"], torch.full((1, 4), 0.5))
    assert float(terms["cat_duty"]) == 0.0
    assert float(terms["cat_foot_duty"]) == 0.0


def test_action_prior_yields_to_learned_constraint_pressure():
    unconstrained = action_prior_weight(5.0, 0.1, 500_000, 2_000_000, 0.0)
    constrained = action_prior_weight(5.0, 0.1, 500_000, 2_000_000, 3.0)
    assert constrained == pytest.approx(unconstrained / 4.0)
    # The scaffold remains recoverable after its normal annealing horizon.
    assert action_prior_weight(5.0, 0.1, 2_000_000, 2_000_000, 0.0) == 0.5
    assert prior_competence_pressure(0.1, 0.02) == pytest.approx(5.0)
    assert prior_competence_pressure(0.1, 0.1) == 1.0
    # At the normal annealing horizon, a 5x competence pressure encodes an 80%
    # normalized shortfall. Safety pressure still divides that scaffold.
    assert action_prior_weight(5.0, 0.1, 2_000_000, 2_000_000,
                               3.0, 5.0) == pytest.approx(1.0)
    assert action_prior_weight(5.0, 0.1, 2_000_000, 2_000_000,
                               0.0, 1.0) == pytest.approx(0.5)


@pytest.mark.parametrize("rung", (2, 4, 9, 14, 16, 17, 18, 19, 23))
def test_locomotion_rungs_have_one_warm_start_contract(rung):
    env = LadderLocomotionWarpEnv(2, rung=rung, seed=11, device="cpu", episode_length=3)
    obs, reward, _, info = env.step(torch.zeros((2, env.act_dim)))
    assert obs.shape == info["terminal_obs"].shape == (2, 234)
    assert env.act_dim == 12 and env.priv_dim == 34
    assert torch.isfinite(obs).all() and torch.isfinite(reward).all()
    assert info["ladder_task_reward"].shape == (2,)
    assert info["ladder_step_clock"].shape == (2,)
    assert info["ladder_swing_clearance"].shape == (2,)
    assert info["ladder_step_action_score"].shape == (2,)
    assert info["ladder_safe_progress"].shape == (2,)
    assert info["ladder_stance_slip_ratio"].shape == (2,)
    assert info["ladder_move_progress"].shape == (2,)
    assert info["fallrate"].shape == (2,)


@pytest.mark.parametrize("rung", range(24, 30))
def test_combat_rungs_have_one_warm_start_contract(rung):
    env = LadderCombatWarpEnv(2, rung=rung, seed=13, device="cpu", episode_length=3)
    obs, reward, _, info = env.step(torch.zeros((2, env.act_dim)))
    assert obs.shape == info["terminal_obs"].shape == (2, 57)
    assert env.act_dim == 14 and env.priv_dim == 32
    assert torch.isfinite(obs).all() and torch.isfinite(reward).all()
    assert info["ladder_task_reward"].shape == (2,)
    assert info["ladder_combat_margin"].shape == (2,)
    assert info["fallrate"].shape == (2,)
