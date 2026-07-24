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

import training_ladder  # noqa: E402
from ladder_warp_env import (GAIT_PERIOD_STEPS, LadderCombatWarpEnv,
                             LadderLocomotionWarpEnv,
                             UniversalCommandWarpEnv,
                             UniversalControlWarpEnv,
                             normalized_duty_costs,
                             swing_clearance_scores)  # noqa: E402
from train_mesh_warp import (Actor, RunningNorm, action_prior_weight,
                             adaptive_ppo_learning_rate,
                             clip_actor_critic_gradients,
                             gate_margin_projection,
                             partition_policy_and_predictor_parameters,
                             duty_stagnation_tripwire_enabled,
                             early_gates_pass, evaluation_trends, gate_diagnostics,
                             incremental_eval_interval,
                             inherit_task_conditioning,
                             kl_epoch_should_stop,
                             load_replay_artifacts,
                             parse_early_gates,
                             policy_observation,
                             prior_competence_pressure, schedule_progress,
                             robust_gate_diagnostics,
                             scale_invariant_value_loss, schedules)  # noqa: E402
from training_ladder import (DIAGNOSTIC_ONLY_METRICS, LadderRunner, RUNGS, Gate, make_parser,
                             merge_candidate_archives, validate_manifest)  # noqa: E402
from walker_warp_env import CAT_FOOT_DUTY_MAX  # noqa: E402
from warp_eval import inherit_policy_checkpoint  # noqa: E402


def test_manifest_is_the_complete_html_ladder():
    validate_manifest()
    assert len(RUNGS) == 31
    assert [r.number for r in RUNGS] == list(range(1, 32))
    assert RUNGS[25].slug == "commanded_leg_kick"
    assert RUNGS[-1].slug == "codesign_loop"
    assert not any(gate.metric in DIAGNOSTIC_ONLY_METRICS
                   for rung in RUNGS for gate in rung.gates)
    assert {gate.role for rung in RUNGS for gate in rung.gates} == {
        "outcome", "constraint"}


def test_walk_teacher_is_an_optional_fallback_not_a_prerequisite(tmp_path):
    args = make_parser().parse_args(["run", "--out", str(tmp_path)])
    runner = LadderRunner(args)
    assert args.walk_prior_mode == "fallback"
    rung6 = tmp_path / "rung_06_step_in_place.pt"
    rung6.touch()
    runner.state["checkpoints"] = {"6": str(rung6)}
    # The repository-local teacher may or may not exist in a deployment.  The
    # fallback path must remain safe when it is absent, while the strict opt-in
    # mode is the only mode allowed to require it.
    original = training_ladder.LEGACY_WALK_TEACHER
    try:
        training_ladder.LEGACY_WALK_TEACHER = tmp_path / "missing_teacher.pt"
        assert runner._ensure_walk_prior() is None
        runner.args.walk_prior_mode = "always"
        with pytest.raises(RuntimeError, match="legacy walk teacher"):
            runner._ensure_walk_prior()
    finally:
        training_ladder.LEGACY_WALK_TEACHER = original


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
        "foot_air_fraction_min": 0.051,
        "speed": 0.06,
        "up": 1.0,
        "catrate": 0.0001,
        "fallrate": 0.0,
        "diagnostics": {"multi_seed_evaluation": {"metrics": {
            # The means pass both contracts, but one held-out seed on each
            # boundary does not.  Promotion must preserve those failures.
            "foot_air_fraction_min": {"values": [0.0495, 0.0505, 0.0506]},
            "catrate": {"values": [0.0001, 0.0011, 0.0002]},
        }}},
    }

    passed, details = runner._gate(rung, metrics)

    assert not passed
    assert any("FAIL [outcome] foot_air_fraction_min" in row and "worst of 3" in row
               for row in details)
    assert any("FAIL [constraint] catrate" in row and "worst of 3" in row
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


def test_robust_gate_diagnostics_use_adverse_seed_tail():
    gates = parse_early_gates(["score,>=,0.7", "error,<=,0.1"])
    report = robust_gate_diagnostics(gates, {"metrics": {
        "score": {"values": [0.8, 0.69, 0.9]},
        "error": {"values": [0.02, 0.11, 0.03]},
    }})
    assert not report["all_pass"]
    assert report["adverse_metrics"] == {"score": 0.69, "error": 0.11}


def test_retention_defaults_to_one_full_episode():
    args = make_parser().parse_args(["run"])
    assert args.retention_steps == args.episode_length == 800


def test_real_replay_artifact_validates_and_preserves_pressure(tmp_path):
    path = tmp_path / "replay.pt"
    torch.save({"rung": 4, "observations": torch.zeros(3, 256),
                "actions": torch.zeros(3, 14)}, path)
    bank = load_replay_artifacts([f"{path},2.5"], 256, 14)
    assert bank[0]["rung"] == 4
    assert bank[0]["pressure"] == pytest.approx(2.5)


def test_feedforward_command_actor_consumes_time_major_replay(tmp_path):
    """Real replay is [time, env, feature], even for feed-forward actors."""
    from train_mesh_warp import build_args, train
    replay = tmp_path / "replay.pt"
    torch.save({
        "rung": 2,
        "observation_semantics": (
            "universal256:physical211+actuator_mask14+command31:v2"),
        "observations": torch.zeros(3, 4, 256),
        "actions": torch.zeros(3, 4, 14),
        "dones": torch.zeros(3, 4, dtype=torch.bool),
    }, replay)
    args = build_args([
        "--geometry", "universal_command", "--rung", "6",
        "--steps", str(8 * 8), "--envs", "8", "--horizon", "8",
        "--episode-length", "50", "--hidden", "32,32", "--seed", "5",
        "--device", "cpu", "--evals", "1", "--eval-envs", "4",
        "--eval-steps", "8", "--diagnostic-eval-seeds", "1",
        "--checkpoint-replay-steps", "4", "--epochs", "1", "--minibatches", "1",
        "--architecture", "task_film", "--distill-weight", "0.05",
        "--replay-artifact", str(replay),
        "--preflight", "off", "--tag", str(tmp_path / "replay_smoke")])
    stats = train(args)
    assert len(stats["updates"]) == 1
    assert stats["updates"][0]["distill_loss"] >= 0.0


@pytest.mark.parametrize("rung,active", ((2, 12), (26, 14), (30, 12)))
def test_universal_controller_contract_is_stable_across_domains(rung, active):
    env = UniversalControlWarpEnv(2, rung=rung, seed=17, device="cpu",
                                  episode_length=2)
    obs = env.reset()
    assert obs.shape == (2, 256)
    assert env.act_dim == 14 and env.priv_dim == 34
    assert int(env.policy_action_mask.sum()) == active
    assert bool((obs[:, -31 + rung - 1] == 1.0).all())


def test_future_interaction_targets_are_broad_and_leg_conditioned():
    locomotion = UniversalControlWarpEnv(
        2, rung=7, seed=17, device="cpu", episode_length=2)
    locomotion.env._set_task_command(alpha=1.0)
    walk_target = locomotion.interaction_target(8)
    assert walk_target.root_delta.shape == (8, 2, 3)
    assert bool((walk_target.root_delta_mask[..., :2] == 1.0).all())
    assert float(walk_target.effector_mask.sum()) == 0.0

    combat = UniversalControlWarpEnv(
        2, rung=26, seed=19, device="cpu", episode_length=2)
    combat.env.set_attack_command(torch.tensor([0, 3]), True)
    kick_target = combat.interaction_target(8)
    torch.testing.assert_close(
        kick_target.effector_mask,
        torch.tensor([[1., 0., 0., 0.], [0., 0., 0., 1.]]))
    torch.testing.assert_close(kick_target.interaction_event,
                               kick_target.effector_mask)
    assert bool((kick_target.min_support == 2.0).all())


@pytest.mark.parametrize("rung", (2, 6, 11, 20, 24, 26, 30))
def test_universal_command_block_is_rung_invisible_and_spec_true(rung):
    """v2 conditioning must be exactly the documented command layout — no
    channel may encode rung identity."""
    env = UniversalCommandWarpEnv(2, rung=rung, seed=17, device="cpu",
                                  episode_length=2)
    obs = env.reset()
    assert obs.shape == (2, 256)
    assert "command31:v2" in env.observation_semantics
    physical, block = obs[:, :225], obs[:, 225:]
    assert float(block[:, 30].abs().max()) == 0.0  # reserved stays zero
    if rung in (2, 6, 11, 30):
        torch.testing.assert_close(block[:, 0:3], physical[:, 47:50])
        assert bool((block[:, 3] == 1.0).all())
    if rung == 11:
        torch.testing.assert_close(block[:, 4:6], physical[:, 65:67])
        assert bool((block[:, 6] == 1.0).all())
    else:
        assert float(block[:, 4:7].abs().max()) == 0.0
    if rung == 6:
        assert bool((block[:, 28] == 1.0).all())
        assert bool((block[:, 29] == 1.0).all())
    else:
        assert float(block[:, 28:30].abs().max()) == 0.0
    if rung == 30:
        assert float(block[:, 8:20].abs().max()) == 0.0  # no pose in codesign
    if rung == 20:
        torch.testing.assert_close(block[:, 20:22], physical[:, 63:65])
        assert bool((block[:, 22] == 1.0).all())
        assert float(block[:, 3].abs().max()) == 0.0
    if rung == 24:
        torch.testing.assert_close(block[:, 20:22], physical[:, 38:40])
        assert bool((block[:, 22] == 1.0).all())
        assert float(block[:, 27].abs().max()) == 0.0  # attack disarmed on 24
    if rung == 26:
        engage = physical[:, 44]
        torch.testing.assert_close(block[:, 27], engage)
        torch.testing.assert_close(
            block[:, 23:27], engage[:, None] * physical[:, 45:49])
        assert float(block[:, 20:23].abs().max()) == 0.0


def test_hold_rungs_share_one_command_signature():
    """Rungs 2 and 12 at a zero command must be observationally compatible:
    same activity flags, so they assert the same demanded behavior."""
    stand = UniversalCommandWarpEnv(1, rung=2, seed=5, device="cpu",
                                    episode_length=2)
    stop = UniversalCommandWarpEnv(1, rung=12, seed=5, device="cpu",
                                   episode_length=2)
    activity = (3, 6, 22, 29)  # velocity, heading, goal, cadence actives
    stand_flags = stand.reset()[:, 225:][:, activity]
    stop_flags = stop.reset()[:, 225:][:, activity]
    torch.testing.assert_close(stand_flags, stop_flags)


def test_command_contract_semantics_and_ladder_geometry_swap(tmp_path):
    from train_mesh_warp import build_args, checkpoint_contract
    env = UniversalCommandWarpEnv(1, rung=2, seed=3, device="cpu",
                                  episode_length=2)
    args = build_args(["--geometry", "universal_command", "--rung", "2",
                       "--architecture", "task_film_gru", "--tag", "x"])
    contract = checkpoint_contract(env, args)
    assert contract["task_conditioning_semantics"] == "command_film_v2"
    assert contract["observation_semantics"].endswith("command31:v2")

    commanded = LadderRunner(make_parser().parse_args(
        ["run", "--out", str(tmp_path / "a"), "--command-observations"]))
    legacy = LadderRunner(make_parser().parse_args(
        ["run", "--out", str(tmp_path / "b")]))
    rung = RUNGS[5]
    assert rung.geometry == "universal_control"
    assert commanded._geometry(rung) == "universal_command"
    assert legacy._geometry(rung) == "universal_control"


def test_power_model_reaches_universal_contract_but_not_combat():
    """Enabling the shared bus must show up in the universal action-semantics
    contract for locomotion rungs, and combat rungs must stay on v1 (the fused
    combat layer does not yet model the bus) without crashing."""
    powered = UniversalCommandWarpEnv(1, rung=2, seed=3, device="cpu",
                                      episode_length=2,
                                      power_model="shared_bus")
    assert powered.action_semantics.endswith("shared_bus_v2")
    assert powered.power_model_record["model"] == "shared_bus_v2"
    combat = UniversalCommandWarpEnv(1, rung=26, seed=3, device="cpu",
                                     episode_length=2,
                                     power_model="shared_bus")
    assert not combat.action_semantics.endswith("shared_bus_v2")


def test_walk_first_ordering_certifies_stand_rungs_after_acquisition():
    from training_ladder import WALK_FIRST_ORDER, ordered_rungs
    numbers = [rung.number for rung in ordered_rungs(1, 31, walk_first=True)]
    assert numbers == list(WALK_FIRST_ORDER)
    assert numbers[1] == 8, "velocity-tracking acquisition must come first"
    assert numbers.index(8) < numbers.index(2), "stand certifies after walking"
    assert numbers.index(8) < numbers.index(6), "stepping follows a rhythmic parent"
    default = [rung.number for rung in ordered_rungs(1, 31, walk_first=False)]
    assert default == list(range(1, 32))
    windowed = [rung.number for rung in ordered_rungs(2, 8, walk_first=True)]
    assert windowed == [8, 7, 2, 3, 4, 5, 6]


def test_previous_checkpoint_follows_acceptance_order_not_numbering(tmp_path):
    """Under walk-first ordering rung 2 certifies AFTER rung 8; its warm/parent
    checkpoint must be the latest ACCEPTED one, not the largest number below 2."""
    runner = LadderRunner(make_parser().parse_args(
        ["run", "--out", str(tmp_path), "--command-observations"]))
    eight = tmp_path / "rung_08.pt"
    eight.touch()
    runner.state["completed"] = [8]
    runner.state["checkpoints"]["8"] = str(eight)
    rung2 = next(rung for rung in RUNGS if rung.number == 2)
    assert runner._previous_checkpoint(rung2) == str(eight)
    # classic numeric runs are unchanged: latest accepted is still the answer
    five = tmp_path / "rung_05.pt"
    five.touch()
    runner.state["completed"] = [2, 3, 4, 5]
    runner.state["checkpoints"] = {"5": str(five)}
    rung6 = next(rung for rung in RUNGS if rung.number == 6)
    assert runner._previous_checkpoint(rung6) == str(five)


def test_walk_first_requires_command_observations(tmp_path):
    from training_ladder import main
    with pytest.raises(SystemExit, match="command-observations"):
        main(["run", "--out", str(tmp_path), "--walk-first", "--dry-run"])


def test_walk_first_dry_run_completes_in_acquisition_order(tmp_path):
    from training_ladder import main
    rc = main(["run", "--out", str(tmp_path), "--walk-first",
               "--command-observations", "--dry-run", "--from", "1",
               "--to", "12"])
    assert rc == 0
    state = json.loads((tmp_path / "ladder_state.json").read_text())
    assert state["completed"] == [1, 8, 10, 9, 11, 12, 7, 2, 3, 4, 5, 6]


def test_rung8_command_distribution_includes_standing_stripe():
    env = LadderLocomotionWarpEnv(16, rung=8, seed=11, device="cpu",
                                  episode_length=4)
    env.reset()
    env._set_task_command(alpha=1.0)
    speeds = env._velocity_command.clone()
    # the published command (not the raw sample) is what the policy tracks
    cmd = env._cmd if hasattr(env, "_cmd") else None
    obs = env.observe()
    forward = obs[:, 47]
    assert float(forward[0].abs()) == 0.0, "stripe world 0 holds at zero"
    assert float(forward[8].abs()) == 0.0, "stripe world 8 holds at zero"
    assert float(forward.abs().max()) > 0.0, "non-stripe worlds still move"


def test_scenario_seed_changes_physical_model_variant():
    low = LadderLocomotionWarpEnv(
        1, rung=17, seed=1, device="cpu", episode_length=2,
        scenario_variant=0.0)
    high = LadderLocomotionWarpEnv(
        1, rung=17, seed=1, device="cpu", episode_length=2,
        scenario_variant=1.0)
    assert low.model_hash != high.model_hash


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


def test_resume_extends_past_durable_checkpoint_without_terminal_failure(
        tmp_path, monkeypatch):
    args = make_parser().parse_args([
        "run", "--out", str(tmp_path), "--resume", "--attempts", "1",
        "--steps-per-rung", "2000000",
    ])
    runner = LadderRunner(args)
    rung = RUNGS[5]
    tag = tmp_path / "rung_06_step_in_place"
    Path(str(tag) + ".pt").touch()
    Path(str(tag) + ".stats.json").write_text(json.dumps({"evals": [
        {"step": 4_005_888},
    ]}))
    runner.state["attempts"]["6"] = 2
    anchor = tmp_path / "rung_05_height_control.pt"
    anchor.touch()
    commands = []
    monkeypatch.setattr(
        runner, "_run",
        lambda argv, log, dry_run=False: commands.append(argv) or 0)
    monkeypatch.setattr(runner, "_metrics", lambda _: {
        "duty": 0.8, "foot_cycle_duty": 0.8, "speed": 0.1,
        "ladder_step_clock": 0.8, "ladder_swing_clearance": 0.5,
        "up": 1.0, "catrate": 0.0, "fallrate": 0.0,
    })
    monkeypatch.setattr(runner, "_candidate_paths", lambda _: [])

    runner._run_ppo(rung, warm_override=str(anchor))

    command = commands[0]
    assert command[command.index("--steps") + 1] == "6000000"
    assert command[command.index("--resume") + 1] == str(tag) + ".pt"


def test_gate_margin_projection_strikes_only_doomed_attempts():
    total = 10_000_000
    # Healthy improvement crossing well within budget: no strike.
    improving = [(1_000_000 * i, -0.5 + 0.1 * i) for i in range(1, 5)]
    assert not gate_margin_projection(
        improving, total, slack=2.0, window=4)["strike"]
    # Failing and flat: never crosses, strike.
    flat = [(1_000_000 * i, -0.4) for i in range(1, 5)]
    assert gate_margin_projection(flat, total, slack=2.0, window=4)["strike"]
    # Failing and worsening: strike.
    worsening = [(1_000_000 * i, -0.2 - 0.05 * i) for i in range(1, 5)]
    assert gate_margin_projection(
        worsening, total, slack=2.0, window=4)["strike"]
    # Failing but improving too slowly: crossing far beyond slack * remaining.
    crawling = [(1_000_000 * i, -1.0 + 1.0e-9 * i) for i in range(1, 5)]
    projection = gate_margin_projection(crawling, total, slack=2.0, window=4)
    assert projection["strike"]
    assert projection["projected_crossing_step"] > total
    # Already passing: the early-gate stop owns this case.
    passing = [(1_000_000 * i, 0.05) for i in range(1, 5)]
    assert not gate_margin_projection(
        passing, total, slack=2.0, window=4)["strike"]
    # Too little history: no verdict.
    assert not gate_margin_projection(
        flat[:2], total, slack=2.0, window=4)["strike"]


def test_entropy_boost_rewinds_but_never_exceeds_start():
    from train_mesh_warp import ENT_END, ENT_START, build_args, schedules
    base = ["--geometry", "walker", "--steps", "1000000", "--tag", "x"]
    plain = build_args(base)
    boosted = build_args(base + ["--entropy-boost", "1.5"])
    huge = build_args(base + ["--entropy-boost", "100.0"])
    step = 500_000
    ent_plain, _, _ = schedules(step, plain)
    ent_boosted, _, _ = schedules(step, boosted)
    ent_huge, _, _ = schedules(step, huge)
    assert ent_boosted == pytest.approx(1.5 * ent_plain)
    assert ent_huge == pytest.approx(ENT_START)
    assert ENT_END < ent_plain < ENT_START


def test_plateau_aborted_retry_gets_exploration_intervention(tmp_path, monkeypatch):
    args = make_parser().parse_args([
        "run", "--out", str(tmp_path), "--resume", "--attempts", "1",
        "--steps-per-rung", "2000000",
    ])
    runner = LadderRunner(args)
    rung = RUNGS[5]
    tag = tmp_path / "rung_06_step_in_place"
    Path(str(tag) + ".pt").touch()
    Path(str(tag) + ".stats.json").write_text(json.dumps({
        "evals": [{"step": 4_005_888}],
        "plateau_abort": {"step": 4_005_888, "consecutive_strikes": 3,
                          "worst_relative_margin": -0.2},
    }))
    runner.state["attempts"]["6"] = 2
    anchor = tmp_path / "rung_05_height_control.pt"
    anchor.touch()
    commands = []
    monkeypatch.setattr(
        runner, "_run",
        lambda argv, log, dry_run=False: commands.append(argv) or 0)
    monkeypatch.setattr(runner, "_metrics", lambda _: {
        "duty": 0.8, "foot_cycle_duty": 0.8, "speed": 0.1,
        "ladder_step_clock": 0.8, "ladder_swing_clearance": 0.5,
        "up": 1.0, "catrate": 0.0, "fallrate": 0.0,
    })
    monkeypatch.setattr(runner, "_candidate_paths", lambda _: [])

    runner._run_ppo(rung, warm_override=str(anchor))

    command = commands[0]
    assert command[command.index("--entropy-boost") + 1] == "1.5"
    assert "--learning-rate-restart" in command
    # Without a plateau record the same retry stays untouched.
    Path(str(tag) + ".stats.json").write_text(json.dumps({
        "evals": [{"step": 4_005_888}]}))
    commands.clear()
    runner._run_ppo(rung, warm_override=str(anchor))
    assert "--entropy-boost" not in commands[0]
    assert "--learning-rate-restart" not in commands[0]


def test_actor_and_critic_gradient_clipping_are_disjoint():
    actor = torch.nn.Linear(2, 1, bias=False)
    critic = torch.nn.Linear(2, 1, bias=False)
    actor.weight.grad = torch.ones_like(actor.weight)
    critic.weight.grad = torch.full_like(critic.weight, 1_000.0)

    actor_norm, critic_norm, predictor_norm = clip_actor_critic_gradients(actor, critic)

    assert float(actor_norm) == pytest.approx(2.0 ** 0.5)
    assert float(critic_norm) == pytest.approx(2.0 ** 0.5 * 1_000.0)
    assert float(predictor_norm) == 0.0
    assert float(actor.weight.grad.norm()) == pytest.approx(1.0)
    assert float(critic.weight.grad.norm()) == pytest.approx(1.0)


def _predictive_actor() -> Actor:
    return Actor(24, 14, (32,), architecture="predictive_token_gru", task_dim=8)


def test_predictor_parameters_form_their_own_optimizer_subspace():
    """The trajectory decoder must never share the adaptive PPO optimizer."""
    actor = _predictive_actor()
    policy, predictor = partition_policy_and_predictor_parameters(actor)
    decoder_ids = {id(p) for p in actor.trajectory_decoder.parameters()}
    assert {id(p) for p in predictor} == decoder_ids
    assert decoder_ids.isdisjoint({id(p) for p in policy})
    assert len(policy) + len(predictor) == len(list(actor.parameters()))

    plain = Actor(11, 3, (16, 8), architecture="task_film_gru", task_dim=4)
    plain_policy, plain_predictor = partition_policy_and_predictor_parameters(plain)
    assert plain_predictor == []
    assert len(plain_policy) == len(list(plain.parameters()))


def test_heldout_calibration_degradation_freezes_and_recovery_unfreezes():
    actor = _predictive_actor()
    actor.prediction_freeze_tolerance = 0.15
    actor.prediction_freeze_patience = 3
    # Warmup: the first ten observations may never freeze, mirroring authority.
    for _ in range(10):
        actor.observe_prediction_calibration(torch.tensor(0.10))
    assert actor.prediction_training_enabled
    best = float(actor.prediction_best_calibration)
    assert best == pytest.approx(float(actor.prediction_calibration_ema))
    # Sustained degradation beyond tolerance freezes after patience runs out.
    for _ in range(40):
        actor.observe_prediction_calibration(torch.tensor(1.0))
        if not actor.prediction_training_enabled:
            break
    assert not actor.prediction_training_enabled
    assert float(actor.prediction_best_calibration) == pytest.approx(best)
    # Calibration keeps being observed while frozen; recovery reopens training.
    for _ in range(60):
        actor.observe_prediction_calibration(torch.tensor(0.05))
        if actor.prediction_training_enabled:
            break
    assert actor.prediction_training_enabled


def test_freeze_disabled_by_nonpositive_tolerance():
    actor = _predictive_actor()
    actor.prediction_freeze_tolerance = 0.0
    for _ in range(30):
        actor.observe_prediction_calibration(torch.tensor(0.10))
    for _ in range(30):
        actor.observe_prediction_calibration(torch.tensor(5.0))
    assert actor.prediction_training_enabled


def test_predictive_actor_loads_pre_freeze_checkpoints():
    torch.manual_seed(3)
    old = _predictive_actor()
    state = old.state_dict()
    for name in ("prediction_best_calibration", "prediction_degraded_streak",
                 "prediction_frozen"):
        state.pop(name)
    fresh = _predictive_actor()
    fresh.load_state_dict(state)
    for a, b in zip(old.parameters(), fresh.parameters()):
        torch.testing.assert_close(a, b)
    assert fresh.prediction_training_enabled


def test_predictive_smoke_separate_optimizer_and_unseen_eval_calibration(tmp_path):
    """CPU proof: decoder Adam is separate and constant, its state checkpoints,
    and evaluation measures held-out predictor calibration on the eval env."""
    from train_mesh_warp import build_args, train
    torch.set_num_threads(1)
    args = build_args([
        "--geometry", "universal_control", "--rung", "2",
        "--steps", str(2 * 8 * 8), "--envs", "8", "--horizon", "8",
        "--episode-length", "50", "--hidden", "32,32", "--seed", "5",
        "--device", "cpu", "--evals", "1", "--eval-envs", "4",
        "--eval-steps", "8", "--diagnostic-eval-seeds", "1",
        "--checkpoint-replay-steps", "4", "--epochs", "1", "--minibatches", "1",
        "--architecture", "predictive_token_gru",
        "--prediction-horizon", "4", "--prediction-anchors", "2",
        "--prediction-lr", "1e-3", "--guidance-horizon", "4",
        "--preflight", "off", "--tag", str(tmp_path / "pred_smoke")])
    stats = train(args)

    update = stats["updates"][0]
    assert update["trajectory_prediction_loss"] > 0.0
    assert update["predictor_learning_rate"] == pytest.approx(1e-3)
    assert update["predictor_gradient_norm_before_clip"] > 0.0
    assert update["trajectory_prediction_frozen"] is False

    eval_record = stats["evals"][-1]
    calibration = eval_record["eval_predictor_calibration"]
    assert calibration["overall"] > 0.0
    assert "body_position" in calibration

    ck = torch.load(stats["ckpt"], map_location="cpu", weights_only=True)
    assert ck["prediction_opt"] is not None
    from train_mesh_warp import Critic
    actor = Actor(256, 14, (32, 32), architecture="predictive_token_gru",
                  task_dim=31)
    critic = Critic(256 + UniversalControlWarpEnv.priv_dim, (32, 32))
    policy, predictor = partition_policy_and_predictor_parameters(actor)
    decoder_parameters = sum(
        len(group["params"]) for group in ck["prediction_opt"]["param_groups"])
    assert decoder_parameters == len(predictor)
    # The main optimizer holds policy + critic only: no decoder overlap.
    main_parameters = sum(
        len(group["params"]) for group in ck["opt"]["param_groups"])
    assert main_parameters == len(policy) + len(list(critic.parameters()))


def test_predictive_family_trains_on_command_observations(tmp_path):
    """The commands-only v2 contract must train end-to-end with the predictive
    family and record the command conditioning in its checkpoint contract."""
    from train_mesh_warp import build_args, train
    torch.set_num_threads(1)
    args = build_args([
        "--geometry", "universal_command", "--rung", "2",
        "--steps", str(2 * 8 * 8), "--envs", "8", "--horizon", "8",
        "--episode-length", "50", "--hidden", "32,32", "--seed", "5",
        "--device", "cpu", "--evals", "1", "--eval-envs", "4",
        "--eval-steps", "8", "--diagnostic-eval-seeds", "1",
        "--checkpoint-replay-steps", "4", "--epochs", "1", "--minibatches", "1",
        "--architecture", "predictive_token_gru",
        "--prediction-horizon", "4", "--prediction-anchors", "2",
        "--guidance-horizon", "4",
        "--preflight", "off", "--tag", str(tmp_path / "cmd_smoke")])
    stats = train(args)
    assert stats["updates"][0]["trajectory_prediction_loss"] > 0.0
    ck = torch.load(stats["ckpt"], map_location="cpu", weights_only=True)
    assert ck["contract"]["task_conditioning_semantics"] == "command_film_v2"
    assert ck["contract"]["observation_semantics"].endswith("command31:v2")


def test_width_blocks_are_the_honest_architecture_parameters():
    from train_mesh_warp import build_args, resolved_architecture, Critic
    base = ["--geometry", "walker", "--tag", "x"]
    args = build_args(base + ["--width", "256", "--blocks", "2"])
    assert args.hidden == "256,256"
    args = build_args(base + ["--width", "128"])
    assert args.hidden == "128,128,128"
    with pytest.raises(SystemExit):
        build_args(base + ["--blocks", "2"])

    # The audit's core finding, now provable from the recorded shapes: a
    # tapering --hidden constructs the SAME FiLM network as the honest
    # constant-width spelling.
    torch.manual_seed(0)
    tapered = Actor(24, 14, (32, 16, 8), architecture="task_film_gru", task_dim=8)
    torch.manual_seed(0)
    honest = Actor(24, 14, (32, 32, 32), architecture="task_film_gru", task_dim=8)
    critic = Critic(30, (32, 32))
    tapered_record = resolved_architecture(tapered, critic)
    honest_record = resolved_architecture(honest, critic)
    assert tapered_record == honest_record
    assert tapered_record["actor"]["parameters"] == sum(
        parameter.numel() for parameter in tapered.parameters())
    predictive = _predictive_actor()
    record = resolved_architecture(predictive, critic)
    assert record["trajectory_decoder"]["parameters"] == sum(
        parameter.numel()
        for parameter in predictive.trajectory_decoder.parameters())


def test_task_inheritance_exactly_preserves_predecessor_behavior():
    torch.manual_seed(17)
    actor = Actor(11, 3, (16, 8), architecture="task_film", task_dim=4)
    norm = RunningNorm(11)
    norm.mean[-4:].copy_(torch.tensor((0.20, 0.55, 0.0, 0.0)))
    norm.var[-4:].copy_(torch.tensor((0.16, 0.24, 1.0e-12, 1.0e-12)))
    physical = torch.randn(9, 7)

    def output(task_index: int):
        task = torch.zeros((len(physical), 4))
        task[:, task_index] = 1.0
        return actor(norm(torch.cat((physical, task), dim=-1)))

    source_before = output(1).detach()
    unrelated_before = output(0).detach()
    assert not torch.allclose(output(2), source_before)
    residual = inherit_task_conditioning(actor, norm, 1, 2)
    assert residual < 1.0e-5
    assert torch.allclose(output(2), source_before, atol=2.0e-6, rtol=2.0e-6)
    assert torch.equal(output(0), unrelated_before)


def test_recurrent_actor_sequence_matches_step_loop_and_real_resets():
    torch.manual_seed(23)
    actor = Actor(11, 3, (16, 8), architecture="task_film_gru", task_dim=4)
    observations = torch.randn(7, 5, 11)
    reset_before = torch.zeros((7, 5), dtype=torch.bool)
    reset_before[3, 1] = True
    reset_before[5, 4] = True
    initial = torch.randn(5, actor.recurrent_state_dim)

    sequence_mean, sequence_state = actor.sequence(
        observations, initial.clone(), reset_before)
    state = initial.clone()
    manual = []
    for time_index in range(len(observations)):
        state *= (~reset_before[time_index]).float().unsqueeze(-1)
        mean, state = actor.step(observations[time_index], state)
        manual.append(mean)

    assert torch.allclose(sequence_mean, torch.stack(manual), atol=1.0e-6)
    assert torch.allclose(sequence_state, state, atol=1.0e-6)


def test_recurrent_actor_builds_internal_phase_without_external_clock():
    torch.manual_seed(29)
    actor = Actor(11, 3, (16, 8), architecture="task_film_gru", task_dim=4)
    observation = torch.randn(4, 11)
    first, state = actor.step(observation, actor.initial_state(4))
    second, state = actor.step(observation, state)

    assert not torch.allclose(first, second)
    assert float(state.detach().norm()) > 0.0
    raw = torch.randn(2, 11)
    masked = policy_observation(raw, (3, 4))
    assert torch.equal(masked[:, 3:5], torch.zeros_like(masked[:, 3:5]))
    assert torch.equal(masked[:, :3], raw[:, :3])
    assert torch.equal(masked[:, 5:], raw[:, 5:])
    morphology_masked = policy_observation(raw, (), (5, 6, 7))
    assert torch.equal(morphology_masked[:, 5:8], torch.zeros_like(raw[:, 5:8]))
    assert torch.equal(morphology_masked[:, :5], raw[:, :5])


def test_inherited_checkpoint_exactly_clones_predecessor_task(tmp_path):
    torch.manual_seed(31)
    actor = Actor(11, 3, (16, 8), architecture="task_film_gru", task_dim=4)
    norm = RunningNorm(11)
    source = tmp_path / "source.pt"
    output = tmp_path / "inherited.pt"
    torch.save({
        "actor": actor.state_dict(),
        "obs_norm": norm.state_dict(),
        "args": {"hidden": "16,8", "architecture": "task_film_gru",
                 "actor_task_dim": 4},
        # candidate manufacture refuses contract-less sources by design
        "contract": {"geometry": "test", "observation_semantics": "test:v1"},
    }, source)

    metadata = inherit_policy_checkpoint(source, output, 1, 2)
    inherited = Actor(11, 3, (16, 8), architecture="task_film_gru", task_dim=4)
    inherited.load_state_dict(torch.load(
        output, map_location="cpu", weights_only=True)["actor"])
    physical = torch.randn(6, 7)
    source_task = torch.zeros(6, 4); source_task[:, 1] = 1.0
    target_task = torch.zeros(6, 4); target_task[:, 2] = 1.0

    source_mean = inherited(norm(torch.cat((physical, source_task), dim=-1)))
    target_mean = inherited(norm(torch.cat((physical, target_task), dim=-1)))
    assert torch.allclose(source_mean, target_mean, atol=2.0e-6, rtol=2.0e-6)
    assert metadata["max_pre_activation_error"] < 1.0e-5


def test_zero_shot_exam_accepts_only_with_margin_and_retention(tmp_path, monkeypatch):
    args = make_parser().parse_args([
        "run", "--out", str(tmp_path), "--architecture", "task_film_gru"])
    runner = LadderRunner(args)
    parent = tmp_path / "rung_02.pt"; parent.write_bytes(b"parent")
    candidate = tmp_path / "test_out.pt"; candidate.write_bytes(b"candidate")
    runner.state["completed"] = [2]
    runner.state["checkpoints"] = {"2": str(parent)}
    rung = RUNGS[2]
    promotion = {"up": 0.99, "catrate": 0.0, "fallrate": 0.0}
    monkeypatch.setattr(runner, "_prepare_test_out_checkpoint",
                        lambda *_: (str(candidate), {"kind": "test"}))
    monkeypatch.setattr(runner, "_promotion_gate",
                        lambda *_: (True, promotion, ["all fresh seeds passed"]))
    monkeypatch.setattr(runner, "_retention_gate", lambda *_: (True, [{"pass": True}]))
    monkeypatch.setattr(runner, "_record_regression_task", lambda *_: None)

    passed, metrics, checkpoint = runner._test_out(rung)

    assert passed
    assert checkpoint == str(candidate)
    assert metrics["test_out_pass"]
    assert metrics["retention_pass"]
    assert runner.state["test_out"]["3"]["decision"] == "accept"


def test_zero_shot_exam_falls_back_to_training_without_extra_margin(
        tmp_path, monkeypatch):
    args = make_parser().parse_args([
        "run", "--out", str(tmp_path), "--architecture", "task_film_gru",
        "--test-out-margin", "0.10"])
    runner = LadderRunner(args)
    parent = tmp_path / "rung_02.pt"; parent.write_bytes(b"parent")
    candidate = tmp_path / "test_out.pt"; candidate.write_bytes(b"candidate")
    runner.state["completed"] = [2]
    runner.state["checkpoints"] = {"2": str(parent)}
    rung = RUNGS[2]
    # The ordinary up gate passes at 0.72, but this is only 4.2% above it.
    promotion = {"up": 0.75, "catrate": 0.0, "fallrate": 0.0}
    monkeypatch.setattr(runner, "_prepare_test_out_checkpoint",
                        lambda *_: (str(candidate), {"kind": "test"}))
    monkeypatch.setattr(runner, "_promotion_gate",
                        lambda *_: (True, promotion, ["ordinary gate passed"]))
    retention_called = []
    monkeypatch.setattr(runner, "_retention_gate",
                        lambda *_: retention_called.append(True) or (True, []))
    monkeypatch.setattr(runner, "_record_regression_task", lambda *_: None)

    passed, metrics, checkpoint = runner._test_out(rung)

    assert not passed
    assert checkpoint is None
    assert not retention_called
    assert metrics["test_out_worst_relative_margin"] < 0.10
    assert runner.state["test_out"]["3"]["decision"] == "train"


def test_running_norm_can_freeze_categorical_task_suffix():
    norm = RunningNorm(5)
    norm.mean.copy_(torch.tensor((0.0, 0.0, 0.0, 0.3, 0.7)))
    norm.var.copy_(torch.tensor((1.0, 1.0, 1.0, 0.2, 0.2)))
    task_mean = norm.mean[-2:].clone()
    task_var = norm.var[-2:].clone()
    batch = torch.tensor(((4.0, 3.0, 2.0, 1.0, 0.0),
                          (2.0, 1.0, 0.0, 1.0, 0.0)))
    norm.update(batch, frozen_suffix=2)
    assert not torch.equal(norm.mean[:3], torch.zeros(3))
    assert torch.equal(norm.mean[-2:], task_mean)
    assert torch.equal(norm.var[-2:], task_var)


def test_value_loss_is_invariant_to_raw_reward_units():
    prediction = torch.tensor([1.0, 3.0])
    target = torch.tensor([2.0, 5.0])
    scale = target.std(unbiased=False)
    original = scale_invariant_value_loss(prediction, target, scale)
    rescaled = scale_invariant_value_loss(
        prediction * 100.0, target * 100.0, scale * 100.0)
    assert float(rescaled) == pytest.approx(float(original))


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
    assert not duty_stagnation_tripwire_enabled("universal_control", 2)


@pytest.mark.parametrize("rung,period,fire", ((3, 140, 60), (15, 100, 45),
                                               (22, 160, 80)))
def test_scripted_push_has_a_narrow_slip_constraint_grace(rung, period, fire):
    env = LadderLocomotionWarpEnv(2, rung=rung, seed=5, device="cpu", episode_length=3)
    env._disturbance_fire.fill_(fire)
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


def test_step_rung_is_phase_free_and_walk_prior_is_temporary(tmp_path):
    env = LadderLocomotionWarpEnv(2, rung=6, seed=7, device="cpu", episode_length=3)
    assert env.policy_mean_prior(env.observe()) is None
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
    assert moving.action_prior_scale == 0.0
    assert moving.policy_mean_prior(
        moving_obs, torch.zeros((2, moving.act_dim))) is None
    prior = tmp_path / "walk_prior.json"
    prior.write_text(json.dumps({"blend": 0.55, "best": {
        "parameters": [0.1] + [0.0] * 23}}))
    moving.configure_action_prior(prior)
    assert moving.action_prior_scale == 10.0
    moving_target, moving_mask = moving.policy_mean_prior(
        moving_obs, torch.zeros((2, moving.act_dim)))
    assert torch.equal(moving_mask[:, 2::3], torch.ones((2, 4)))
    assert torch.equal(moving_mask[:, 0::3], torch.ones((2, 4)))
    assert torch.equal(moving_mask[:, 1::3], torch.ones((2, 4)))
    assert torch.isfinite(moving_target).all()
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
    moving._constraint_duals.copy_(torch.tensor([3.0]))
    _, slip_pressure_mask = moving.policy_mean_prior(
        moving.observe(), torch.zeros((2, moving.act_dim)), transfer_action=transfer)
    assert torch.allclose(slip_pressure_mask[:, 2::3], torch.ones((2, 4)))
    assert float(moving.action_prior_suppression_pressure) == 3.0
    moving.update_constraint_duals(torch.tensor([0.002]))
    assert float(moving.constraint_duals[0]) > 0.0


def test_constraint_dual_controller_is_scale_normalized():
    env = LadderLocomotionWarpEnv(1, rung=7, seed=9, device="cpu", episode_length=3)

    def response(observed: float) -> float:
        env._constraint_duals.zero_()
        env._constraint_error_square.zero_()
        for _ in range(100):
            env.update_constraint_duals(torch.tensor([observed]))
        return float(env.constraint_duals[0])

    small_violation = response(0.002)
    large_violation = response(0.020)
    # A tenfold transient changes the normalizer, not the long-run controller
    # timescale. Both persistent violations therefore create comparable pressure.
    assert 0.75 <= large_violation / small_violation <= 1.25

    before_recovery = float(env.constraint_duals[0])
    for _ in range(100):
        env.update_constraint_duals(torch.tensor([0.0]))
    assert float(env.constraint_duals[0]) < 0.01 * before_recovery

    for _ in range(100):
        env.update_competence_duals(torch.tensor([0.02]))
    competence_pressure = float(env.competence_duals[0])
    assert competence_pressure > 0.0
    for _ in range(100):
        env.update_competence_duals(torch.tensor([0.12]))
    assert float(env.competence_duals[0]) < 0.01 * competence_pressure


def test_foot_activity_competence_pressure_self_tunes_to_rung_six_gate():
    env = LadderLocomotionWarpEnv(1, rung=6, seed=9, device="cpu", episode_length=3)
    assert env.adaptive_competence_names == ("ladder_foot_activity",)
    assert float(env._competence_targets[0]) == pytest.approx(0.40)

    for _ in range(100):
        env.update_competence_duals(torch.tensor([0.30]))
    shortfall_pressure = float(env.competence_duals[0])
    assert shortfall_pressure > 0.0

    for _ in range(100):
        env.update_competence_duals(torch.tensor([0.45]))
    assert float(env.competence_duals[0]) < 0.01 * shortfall_pressure


def test_rung_six_has_no_hidden_phase_curriculum():
    env = LadderLocomotionWarpEnv(
        128, rung=6, seed=31, device="cpu", episode_length=3)
    mask = torch.ones(env.nworld, dtype=torch.bool)
    assert float(env.reset_phase_randomization_probability) == 0.0
    assert int(torch.count_nonzero(env._task_t)) == 0

    # Neither safety nor competence pressure invents a target phase.
    env._constraint_duals.copy_(torch.tensor([10.0]))
    env._competence_duals.copy_(torch.tensor([10.0]))
    env.reset(mask)
    assert int(torch.count_nonzero(env._task_t)) == 0

    moving = LadderLocomotionWarpEnv(
        128, rung=7, seed=31, device="cpu", episode_length=3)
    assert float(moving.reset_phase_randomization_probability) == 1.0
    moving.reset(mask)
    assert int(torch.count_nonzero(moving._task_t)) > moving.nworld // 2


def test_ladder_disables_hidden_base_reference_gait_reward():
    env = LadderLocomotionWarpEnv(
        2, rung=7, seed=32, device="cpu", episode_length=3)
    _, _, _, info = env.step(
        torch.zeros((2, env.act_dim)), imit_anneal=1.0)
    components = info["reward_components"]
    prior = components.get("motion_prior")
    assert prior is None or not bool(torch.count_nonzero(prior))
    # gait_clock was deleted outright (a permanently zero-weight style term);
    # its absence is the strongest form of "disabled".
    assert "gait_clock" not in components
    for name in ("airtime", "pose_penalty", "clearance", "tracking"):
        assert not bool(torch.count_nonzero(components[name]))


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


def test_step_in_place_reports_duty_without_treating_style_as_catastrophe():
    env = LadderLocomotionWarpEnv(1, rung=6, seed=11, device="cpu", episode_length=100)
    assert env.adaptive_constraint_names == ("cat_slip",)
    assert env.cat_term_keys == ("cat_slip", "cat_orient", "cat_qvel", "cat_body")
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
    assert "cat_duty" not in env.cat_term_keys
    assert "cat_foot_duty" not in env.cat_term_keys


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


def test_extended_resume_cannot_reheat_annealing_schedules():
    args = type("Args", (), {
        "steps": 14_000_000,
        "alpha_frac": 0.60,
        "alpha_start": 0.0,
        "alpha_end": 1.0,
        "imit_anneal_frac": 0.45,
    })()
    assert schedule_progress(12_000_000, args) < 1.0
    assert schedule_progress(12_000_000, args, 1.0) == 1.0
    ent, alpha, imitation = schedules(12_000_000, args, 1.0)
    assert ent == pytest.approx(0.005)
    assert alpha == pytest.approx(1.0)
    assert imitation == pytest.approx(0.0)
    assert action_prior_weight(
        5.0, 0.1, 12_000_000, 14_000_000,
        progress_fraction=1.0) == pytest.approx(0.5)


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
    assert info["ladder_foot_activity"].shape == (2,)
    assert info["ladder_foot_activity_mean"].shape == (2,)
    assert info["ladder_foot_activity_ema"].shape == (2,)
    assert info["ladder_safe_progress"].shape == (2,)
    assert info["ladder_stance_slip_ratio"].shape == (2,)
    assert info["ladder_move_progress"].shape == (2,)
    assert info["fallrate"].shape == (2,)
    expected_activity = ((1.0 - info["cycle_duty_by_leg"])
                         / (1.0 - CAT_FOOT_DUTY_MAX)).clamp(0.0, 1.0).amin(-1)
    assert torch.allclose(info["ladder_foot_activity"], expected_activity)
    components = info["reward_components"]
    ladder_sum = sum(components[name] for name in (
        "ladder_task_outcome", "ladder_task_constraint", "ladder_task_scaffold"))
    assert torch.allclose(ladder_sum, info["ladder_task_reward"])


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
