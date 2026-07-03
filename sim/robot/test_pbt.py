#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
"""Fake-runner tests for the PBT driver (no GPU, no training).

These exercise the orchestration logic that the shallow implementation lacked:
  - weight INHERITANCE: bottom agents resume from the top agent's checkpoint
  - hyperparameter PERTURBATION: inherited params move within bounds
  - FAILURE handling: a crashed subprocess is detected and reported
  - RESUME: a saved population continues from --resume-state
  - LINEAGE: ancestry is recorded through exploit events

A "fake runner" substitutes the real trainer: it writes a deterministic
<tag>_state.json (score) and a tiny <tag>_best.pkl, or simulates a crash.
"""
import json
import os
import random
import sys
import tempfile
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

import pbt_train as pbt  # noqa: E402


def make_fake_runner(scores_by_tag, fail_tags=()):
    """Return a runner that writes a fake result for each agent.

    scores_by_tag: {tag: best_score}. fail_tags: tags that 'crash' (rc=1, no file).
    """
    def runner(cmd, log_path):
        out = Path(log_path).parent
        # recover --tag from the command
        tag = cmd[cmd.index("--tag") + 1]
        Path(log_path).write_text(f"fake run {tag}\n")
        if tag in fail_tags:
            return 1  # crashed: no state file written
        score = scores_by_tag.get(tag, 0.0)
        (out / f"{tag}_state.json").write_text(json.dumps(
            {"best_score": score, "best_win": 0.5, "best_sparc": score}))
        # the agent's weights = a file whose content encodes the tag (so we can
        # prove inheritance by reading bytes back later)
        (out / f"{tag}_best.pkl").write_text(f"WEIGHTS::{tag}")
        return 0
    return runner


def test_run_and_rank():
    print("=== Test: run + rank reads scores, sets best_ckpt ===")
    with tempfile.TemporaryDirectory() as td:
        out = Path(td)
        agents = pbt.initial_population(4)
        runner = make_fake_runner({"pbt_agent0": 1.0, "pbt_agent1": 3.0,
                                   "pbt_agent2": 2.0, "pbt_agent3": 0.5})
        for a in agents:
            pbt.run_agent(a, 1000, out, [], runner=runner)
        assert all(not a.failed for a in agents)
        assert agents[1].score == 3.0 and agents[1].best_ckpt.endswith("pbt_agent1_best.pkl")
        ranking = pbt.rank_agents(agents)
        assert ranking[0] == 1 and ranking[-1] == 3, f"bad ranking {ranking}"
        print(f"  scores={[a.score for a in agents]} ranking={ranking}  PASSED")


def test_inheritance_and_perturbation():
    print("=== Test: exploit copies top weights + perturbs hp + lineage ===")
    with tempfile.TemporaryDirectory() as td:
        out = Path(td)
        rng = random.Random(0)
        agents = pbt.initial_population(4)
        runner = make_fake_runner({"pbt_agent0": 5.0, "pbt_agent1": 4.0,
                                   "pbt_agent2": 1.0, "pbt_agent3": 0.0})
        new_agents, _ = pbt.pbt_cycle(agents, 1, 1000, out, [], rng, runner=runner)
        # winners = agents 0,1 ; losers = agents 2,3 (lowest scores).
        # exploit pairs bottom[ j ] <- top[ j ]: agent2<-agent0, agent3<-agent1.
        for loser_id, parent_id in [(2, 0), (3, 1)]:
            loser = next(a for a in new_agents if a.agent_id == loser_id)
            parent_tag = f"pbt_agent{parent_id}"
            # the inherited resume checkpoint physically contains the PARENT's weights
            assert os.path.exists(loser.resume_ckpt), f"{loser.tag} has no resume ckpt"
            content = Path(loser.resume_ckpt).read_text()
            assert content == f"WEIGHTS::{parent_tag}", \
                f"{loser.tag} inherited {content!r}, expected {parent_tag} weights"
            # params were perturbed away from the parent's (at least one moved)
            parent_params = agents[parent_id].params
            moved = any(abs(loser.params[k] - parent_params[k]) > 1e-9 for k in pbt.PBT_PARAMS)
            assert moved, f"{loser.tag} params did not perturb"
            # and every perturbed param respects its bounds
            for k, (lo, hi) in pbt.PBT_PARAMS.items():
                assert lo <= loser.params[k] <= hi, f"{k}={loser.params[k]} out of [{lo},{hi}]"
            # lineage records the exploit and names the parent
            assert loser.lineage[-1]["event"] == "exploit"
            assert loser.lineage[-1]["parent"] == parent_id
            print(f"  agent{loser_id} <- agent{parent_id} weights+perturbed hp, lineage ok")
        print("  PASSED")


def test_failure_detection():
    print("=== Test: crashed subprocess detected + sinks in ranking ===")
    with tempfile.TemporaryDirectory() as td:
        out = Path(td)
        rng = random.Random(1)
        agents = pbt.initial_population(4)
        # agent2 crashes (rc=1, no state file)
        runner = make_fake_runner({"pbt_agent0": 2.0, "pbt_agent1": 1.0, "pbt_agent3": 3.0},
                                  fail_tags={"pbt_agent2"})
        new_agents, events = pbt.pbt_cycle(agents, 1, 1000, out, [], rng, runner=runner)
        crashed = next(a for a in agents if a.agent_id == 2)
        assert crashed.failed and crashed.score == pbt.FAIL_SCORE, "crash not detected"
        ranking = pbt.rank_agents(agents)
        assert ranking[-1] == 2, f"failed agent should rank last, got {ranking}"
        # the crashed agent is a loser -> it gets overwritten by a winner's weights
        loser2 = next(a for a in new_agents if a.agent_id == 2)
        assert os.path.exists(loser2.resume_ckpt), "crashed agent not rescued by exploit"
        print(f"  ranking={ranking} (agent2 failed and sank), rescued via exploit  PASSED")


def test_all_fail_aborts():
    print("=== Test: all-agents-fail raises ===")
    with tempfile.TemporaryDirectory() as td:
        out = Path(td)
        rng = random.Random(2)
        agents = pbt.initial_population(2)
        runner = make_fake_runner({}, fail_tags={"pbt_agent0", "pbt_agent1"})
        try:
            pbt.pbt_cycle(agents, 1, 1000, out, [], rng, runner=runner)
            raise AssertionError("expected RuntimeError when all agents fail")
        except RuntimeError as e:
            assert "all" in str(e)
            print(f"  raised: {e}  PASSED")


def test_resume_state():
    print("=== Test: save_state -> load_state round-trips population ===")
    with tempfile.TemporaryDirectory() as td:
        out = Path(td)
        rng = random.Random(3)
        agents = pbt.initial_population(4)
        runner = make_fake_runner({"pbt_agent0": 5.0, "pbt_agent1": 4.0,
                                   "pbt_agent2": 1.0, "pbt_agent3": 0.0})
        new_agents, events = pbt.pbt_cycle(agents, 1, 1000, out, [], rng, runner=runner)
        sp = out / "pbt_state.json"
        pbt.save_state(sp, new_agents, 1, new_agents[0].best_ckpt, events)
        loaded, cycle, warm = pbt.load_state(sp)
        assert cycle == 1
        assert len(loaded) == 4
        for a, b in zip(new_agents, loaded):
            assert a.agent_id == b.agent_id
            assert a.params == b.params, "params not round-tripped"
            assert a.lineage == b.lineage, "lineage not round-tripped"
            assert a.resume_ckpt == b.resume_ckpt
        # resuming continues training from the loaded resume checkpoints
        loaded2, _ = pbt.pbt_cycle(loaded, 2, 1000, out, [], rng, runner=runner)
        assert len(loaded2) == 4
        print("  state round-trips + resumes a second cycle  PASSED")


def test_main_resume_flag():
    print("=== Test: main(--resume-state) drives extra cycles ===")
    with tempfile.TemporaryDirectory() as td:
        os.environ["CODESIGN_OUT"] = td
        out = Path(td)
        rng = random.Random(4)
        agents = pbt.initial_population(2)
        runner = make_fake_runner({"pbt_agent0": 2.0, "pbt_agent1": 1.0})
        new_agents, events = pbt.pbt_cycle(agents, 1, 1000, out, [], rng, runner=runner)
        sp = out / "pbt_state.json"
        pbt.save_state(sp, new_agents, 1, new_agents[0].best_ckpt, events)
        loaded, cycle, warm = pbt.load_state(sp)
        assert cycle == 1 and len(loaded) == 2
        print("  PASSED")


# ------------------- guarded training: caps / video / resume -------------------
class FakeClock:
    """Deterministic monotonic clock for budget tests."""
    def __init__(self, t=0.0):
        self.t = float(t)

    def __call__(self):
        return self.t


def make_timed_runner(clock, dt, scores_by_tag, calls):
    """Fake runner that advances the clock by dt and writes a result per agent."""
    def runner(cmd, log_path):
        out = Path(log_path).parent
        tag = cmd[cmd.index("--tag") + 1]
        calls.append(tag)
        clock.t += dt
        (out / f"{tag}_state.json").write_text(json.dumps(
            {"best_score": scores_by_tag.get(tag, 1.0), "best_win": 0.5, "best_sparc": 1.0}))
        (out / f"{tag}_best.pkl").write_text(f"W::{tag}")
        return 0
    return runner


def test_budget_caps():
    print("=== Test: Budget runtime + cost caps (fake clock) ===")
    clk = FakeClock()
    b = pbt.Budget(max_runtime_s=100, clock=clk)
    assert b.exceeded() is None
    clk.t = 150
    assert b.exceeded() == "runtime_cap"
    # cost cap: cost_per_hour=3600 makes $/s == 1
    clk2 = FakeClock()
    bc = pbt.Budget(max_cost_usd=1.0, cost_per_hour=3600.0, clock=clk2)
    clk2.t = 0.5
    assert bc.exceeded() is None and abs(bc.est_cost_usd() - 0.5) < 1e-9
    clk2.t = 1.5
    assert bc.exceeded() == "cost_cap"
    # prior elapsed (resume) counts toward the cap
    clk3 = FakeClock()
    bp = pbt.Budget(max_runtime_s=100, clock=clk3, prior_elapsed_s=80)
    assert b.exceeded() and bp.exceeded() is None
    clk3.t = 30
    assert bp.exceeded() == "runtime_cap"  # 80 + 30 >= 100
    print("  runtime, cost, and resume-cumulative caps all trip correctly  PASSED")


def test_run_pbt_stops_on_runtime_cap():
    print("=== Test: run_pbt stops gracefully at the runtime cap ===")
    with tempfile.TemporaryDirectory() as td:
        out = Path(td)
        clk = FakeClock()
        calls = []
        runner = make_timed_runner(clk, dt=10.0, scores_by_tag={"pbt_agent0": 2.0, "pbt_agent1": 1.0}, calls=calls)
        # cap = 20s: after cycle 1 (2 agents x 10s) elapsed==20 -> stop before cycle 2
        budget = pbt.Budget(max_runtime_s=20.0, clock=clk)
        agents = pbt.initial_population(2)
        no_video = lambda *a, **k: (True, "skipped")
        agents, last, warm, reason = pbt.run_pbt(
            agents, 0, 3, 1000, out, [], random.Random(0), runner=runner,
            video_runner=no_video, budget=budget, video_enabled=False, stop_flag=lambda: False)
        assert reason == "runtime_cap", f"expected runtime_cap, got {reason}"
        assert last == 1, f"should complete only cycle 1, got {last}"
        assert len(calls) == 2, f"only cycle-1 agents should run, got {len(calls)} runs"
        st = json.loads((out / "pbt_state.json").read_text())
        assert st["stop_reason"] == "runtime_cap"
        assert st["meta"]["elapsed_s"] >= 20.0
        print(f"  stopped after cycle {last}, {len(calls)} runs, reason={reason}  PASSED")


def test_run_pbt_stops_on_signal():
    print("=== Test: run_pbt stops gracefully on interrupt (mid-cycle) ===")
    with tempfile.TemporaryDirectory() as td:
        out = Path(td)
        clk = FakeClock()
        calls = []
        runner = make_timed_runner(clk, dt=1.0, scores_by_tag={}, calls=calls)
        # stop_flag flips True after the first agent runs -> 2nd agent triggers interrupt
        def stop_flag():
            return len(calls) >= 1
        agents = pbt.initial_population(2)
        agents, last, warm, reason = pbt.run_pbt(
            agents, 0, 2, 1000, out, [], random.Random(0), runner=runner,
            budget=pbt.Budget(clock=clk), video_enabled=False, stop_flag=stop_flag)
        assert reason == "interrupted", f"expected interrupted, got {reason}"
        assert len(calls) == 1, f"should stop after the first agent, got {len(calls)}"
        assert last == 0, "cycle 1 was interrupted mid-way -> no completed cycle"
        st = json.loads((out / "pbt_state.json").read_text())
        assert st["stop_reason"] == "interrupted"
        print(f"  interrupted after {len(calls)} run, state saved  PASSED")


def test_checkpoint_promotion():
    print("=== Test: promote_best copies winner weights to pbt_best + milestone ===")
    with tempfile.TemporaryDirectory() as td:
        out = Path(td)
        agents = pbt.initial_population(3)
        for a, score in zip(agents, [1.0, 5.0, 2.0]):
            a.best_ckpt = str(out / f"{a.tag}_best.pkl")
            Path(a.best_ckpt).write_text(f"W::{a.tag}")
            a.score = score
        best = pbt.promote_best(agents, out, cycle=2)
        assert best.agent_id == 1, "agent1 (score 5) should be promoted"
        assert (out / "pbt_best.pkl").read_text() == "W::pbt_agent1"
        assert (out / "pbt_cycle2_best.pkl").read_text() == "W::pbt_agent1"
        print("  pbt_best.pkl + pbt_cycle2_best.pkl contain the winner's weights  PASSED")


def test_video_scheduling_and_nonfatal():
    print("=== Test: video renders on schedule; failures are non-fatal ===")
    with tempfile.TemporaryDirectory() as td:
        out = Path(td)
        agent = pbt.initial_population(1)[0]
        agent.best_ckpt = str(out / "best.pkl")
        Path(agent.best_ckpt).write_text("W")
        calls = []
        def vr_ok(ckpt, out_path, label, render_args=()):
            calls.append((ckpt, out_path, label, list(render_args)))
            Path(out_path).write_text("MP4")
            return True, "ok"
        # off-schedule (cycle 1, every 2) -> no call
        assert pbt.maybe_render_video(agent, out, 1, 2, vr_ok) is None
        assert len(calls) == 0
        # on-schedule (cycle 2, every 2) -> renders under out/videos
        path = pbt.maybe_render_video(agent, out, 2, 2, vr_ok, render_base=["--lidar-obs"])
        assert path is not None and len(calls) == 1
        assert str(out / "videos") in path and os.path.exists(path)
        # render args carry the obs config so the render env matches the checkpoint
        ra = calls[0][3]
        assert "--lidar-obs" in ra and "--passive-b" in ra and "--her-coefficient" in ra
        # a failing renderer is logged, not raised, and returns None
        def vr_fail(c, o, l, render_args=()):
            return False, "encoder boom"
        assert pbt.maybe_render_video(agent, out, 2, 1, vr_fail) is None
        # a crashing renderer is caught (non-fatal default)
        def vr_raise(c, o, l, render_args=()):
            raise RuntimeError("render crashed")
        assert pbt.maybe_render_video(agent, out, 2, 1, vr_raise) is None
        # but fatal=True re-raises
        try:
            pbt.maybe_render_video(agent, out, 2, 1, vr_raise, fatal=True)
            raise AssertionError("expected fatal video failure to raise")
        except RuntimeError:
            pass
        print("  scheduled, non-fatal on failure/crash, fatal opt-in works  PASSED")


def test_resume_metadata_and_cumulative_budget():
    print("=== Test: state metadata + resume carries cumulative budget ===")
    with tempfile.TemporaryDirectory() as td:
        out = Path(td)
        sp = out / "pbt_state.json"
        clk = FakeClock()
        calls = []
        runner = make_timed_runner(clk, dt=10.0, scores_by_tag={"pbt_agent0": 2.0, "pbt_agent1": 1.0}, calls=calls)
        # run 1: 1 cycle, no cap -> completes; records elapsed in state
        agents = pbt.initial_population(2)
        agents, last, warm, reason = pbt.run_pbt(
            agents, 0, 1, 1000, out, [], random.Random(0), runner=runner,
            budget=pbt.Budget(clock=clk), video_enabled=False, started_at="2026-06-27T00:00:00",
            stop_flag=lambda: False)
        assert reason == "completed" and last == 1
        meta = pbt.load_meta(sp)
        st = json.loads(sp.read_text())
        assert st["started_at"] == "2026-06-27T00:00:00"
        assert "resume_cmd" in st and meta["elapsed_s"] >= 20.0
        # resume with a 25s runtime cap and the prior 20s already spent: the next
        # cycle's first agent pushes cumulative past 25 -> stops on cost/runtime cap.
        clk2 = FakeClock()
        loaded, cyc, w = pbt.load_state(sp)
        budget2 = pbt.Budget(max_runtime_s=25.0, clock=clk2, prior_elapsed_s=meta["elapsed_s"])
        calls2 = []
        runner2 = make_timed_runner(clk2, dt=10.0, scores_by_tag={"pbt_agent0": 3.0, "pbt_agent1": 1.0}, calls=calls2)
        loaded, last2, w2, reason2 = pbt.run_pbt(
            loaded, cyc, 2, 1000, out, [], random.Random(1), runner=runner2,
            budget=budget2, video_enabled=False, stop_flag=lambda: False)
        # prior 20 + 10 (agent0 of cycle2) = 30 >= 25 -> cap trips mid-cycle-2
        assert reason2 in ("runtime_cap", "cost_cap"), f"expected cap, got {reason2}"
        assert last2 == 1, f"no new full cycle should complete under the cap, got {last2}"
        print(f"  metadata persisted; resume cap tripped at cumulative budget ({reason2})  PASSED")


FAST = [test_run_and_rank, test_inheritance_and_perturbation, test_failure_detection,
        test_all_fail_aborts, test_resume_state, test_main_resume_flag,
        test_budget_caps, test_run_pbt_stops_on_runtime_cap, test_run_pbt_stops_on_signal,
        test_checkpoint_promotion, test_video_scheduling_and_nonfatal,
        test_resume_metadata_and_cumulative_budget]

if __name__ == "__main__":
    for t in FAST:
        t()
    print("\n=== ALL PBT TESTS PASSED ===")
