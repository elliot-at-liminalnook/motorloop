# SPDX-License-Identifier: MIT
"""Population-Based Training (PBT) driver for the adversarial combat policy.

Runs N PPO agents, each with its OWN persistent hyperparameters, checkpoint,
latest benchmark score, and lineage.  Every cycle:

  1. Each agent resumes from ITS OWN checkpoint (``--resume <agent.resume_ckpt>``)
     and trains for ``steps // cycles`` env-steps.
  2. Agents are ranked by their held-out benchmark score (from ``<tag>_state.json``).
  3. EXPLOIT/EXPLORE: each bottom-half agent copies a top-half agent's WEIGHTS
     (the actual ``.pkl`` is copied into the bottom agent's resume path) and
     PERTURBS the inherited hyperparameters.  Lineage records the inheritance.

Guarded for long unattended runs:
  * BUDGET CAPS — ``--max-runtime-min`` / ``--max-cost-usd`` (with ``--cost-per-hour``)
    stop the population GRACEFULLY between agents/cycles, never mid-checkpoint.
  * SIGINT/SIGTERM — caught; the run stops gracefully after the current agent.
  * CHECKPOINTS — after each cycle the best agent's weights are promoted to
    ``pbt_best.pkl`` and a per-cycle milestone ``pbt_cycle{N}_best.pkl``; the full
    population + resume metadata is written to ``pbt_state.json`` every cycle.
  * RESUME — ``--resume-state <pbt_state.json>`` continues a population, carrying
    cumulative elapsed time so budget caps span resumes.  ``stop_reason`` and a
    ``resume_cmd`` hint are recorded for clean restarts.
  * VIDEO — every ``--video-every`` cycles the best checkpoint is rendered to a
    1v1 combat MP4 via ``render_fight_video.py`` (the milestone/render path).
    Render failures are LOGGED, not fatal (unless ``--video-fatal``).

Subprocess failures (non-zero exit OR missing result) are detected and reported;
a failed agent sinks in the ranking and is rescued by exploit/explore.

The orchestration is dependency-light (pure stdlib) and unit-tested with
injectable runner / video-runner / clock (see ``test_pbt.py``) — cap behavior,
checkpoint+video scheduling, state persistence and resume safety are all proven
without a GPU.

Usage:
  python pbt_train.py --agents 4 --cycles 3 --steps 12000000 \
      --max-runtime-min 180 --max-cost-usd 12 --cost-per-hour 1.39 \
      --warm-ckpt <ckpt> -- --lidar-obs --hierarchical ...
  python pbt_train.py --resume-state <OUT>/pbt_state.json --cycles 2 -- ...
"""

from __future__ import annotations

import argparse
import json
import os
import random
import shutil
import signal
import subprocess
import sys
import time
from dataclasses import dataclass, field, asdict
from pathlib import Path

HERE = Path(__file__).resolve().parent
FAIL_SCORE = -1e30  # sentinel for a crashed / missing-result agent
DEFAULT_COST_PER_HOUR = 1.39  # A100 80GB PCIe on RunPod (override with --cost-per-hour)


def default_out() -> Path:
    out = Path(os.environ.get("CODESIGN_OUT", "/root/proj/out"))
    out.mkdir(parents=True, exist_ok=True)
    return out


def now_s() -> float:
    return time.monotonic()


# Hyperparameters PBT perturbs (each agent carries its own value): (min, max).
# NOTE: train_adversarial now defaults --lr-schedule ADAPTIVE_KL, which retunes lr
# from --desired-kl during the run — so PBT's "lr" dimension only sets the INITIAL
# lr. Perturbing it still matters early, but exploit/explore on lr is much weaker
# than pre-2026-07; consider perturbing desired_kl instead if lr diversity stalls.
PBT_PARAMS = {
    "approach_weight": (0.5, 5.0),
    "shaping": (0.2, 1.5),
    "clean_weight": (0.0, 10.0),
    "trade_weight": (0.0, 8.0),
    "fire_shaping": (0.0, 2.0),
    "face_weight": (0.0, 5.0),
    "lr": (1e-4, 1e-3),
    "entropy": (0.005, 0.02),
    "upright_weight": (0.05, 0.5),
    "flee_penalty": (0.0, 2.0),
    "close_bonus": (0.0, 3.0),
    "gate_weight": (0.25, 2.0),
    "rnd_coefficient": (0.0, 0.2),
    "her_coefficient": (0.0, 0.5),
}

# Initial population: spread across the parameter space.
INITIAL_POP = [
    {"approach_weight": 4.0, "shaping": 1.0, "clean_weight": 3.0, "trade_weight": 1.0,
     "fire_shaping": 1.0, "face_weight": 0.5, "lr": 3e-4, "entropy": 0.01,
     "upright_weight": 0.1, "flee_penalty": 0.0, "close_bonus": 0.0,
     "gate_weight": 1.0, "rnd_coefficient": 0.05, "her_coefficient": 0.2},
    {"approach_weight": 2.0, "shaping": 0.6, "clean_weight": 5.0, "trade_weight": 5.0,
     "fire_shaping": 0.5, "face_weight": 1.0, "lr": 3e-4, "entropy": 0.01,
     "upright_weight": 0.3, "flee_penalty": 1.0, "close_bonus": 1.0,
     "gate_weight": 1.0, "rnd_coefficient": 0.05, "her_coefficient": 0.2},
    {"approach_weight": 3.0, "shaping": 1.2, "clean_weight": 2.0, "trade_weight": 2.0,
     "fire_shaping": 0.8, "face_weight": 2.0, "lr": 3e-4, "entropy": 0.015,
     "upright_weight": 0.2, "flee_penalty": 0.5, "close_bonus": 2.0,
     "gate_weight": 1.5, "rnd_coefficient": 0.05, "her_coefficient": 0.35},
    {"approach_weight": 2.5, "shaping": 0.4, "clean_weight": 1.0, "trade_weight": 0.5,
     "fire_shaping": 0.3, "face_weight": 0.0, "lr": 5e-4, "entropy": 0.02,
     "upright_weight": 0.15, "flee_penalty": 0.0, "close_bonus": 0.5,
     "gate_weight": 0.75, "rnd_coefficient": 0.1, "her_coefficient": 0.1},
]


# --------------------------- budget / graceful stop ---------------------------
class BudgetExceeded(Exception):
    """Raised to unwind a cycle when a runtime/cost cap is hit."""
    def __init__(self, reason: str):
        self.reason = reason
        super().__init__(reason)


class Budget:
    """Wall-clock + estimated-cost cap with an injectable clock (testable)."""

    def __init__(self, max_runtime_s: float = 0.0, max_cost_usd: float = 0.0,
                 cost_per_hour: float = DEFAULT_COST_PER_HOUR, clock=now_s,
                 prior_elapsed_s: float = 0.0):
        self.max_runtime_s = float(max_runtime_s)
        self.max_cost_usd = float(max_cost_usd)
        self.cost_per_hour = float(cost_per_hour)
        self._clock = clock
        self._start = clock()
        self._prior = float(prior_elapsed_s)

    def elapsed_s(self) -> float:
        return self._prior + (self._clock() - self._start)

    def est_cost_usd(self) -> float:
        return self.elapsed_s() / 3600.0 * self.cost_per_hour

    def exceeded(self):
        """Return a stop reason string if a cap is hit, else None."""
        if self.max_runtime_s and self.elapsed_s() >= self.max_runtime_s:
            return "runtime_cap"
        if self.max_cost_usd and self.est_cost_usd() >= self.max_cost_usd:
            return "cost_cap"
        return None

    def as_dict(self) -> dict:
        return {"elapsed_s": round(self.elapsed_s(), 1),
                "est_cost_usd": round(self.est_cost_usd(), 4),
                "cost_per_hour": self.cost_per_hour,
                "max_runtime_s": self.max_runtime_s,
                "max_cost_usd": self.max_cost_usd}


_STOP = {"flag": False}


def stop_requested() -> bool:
    return _STOP["flag"]


def install_signal_handlers():
    """Catch SIGINT/SIGTERM so a long run stops gracefully after the current agent."""
    def handler(signum, frame):
        _STOP["flag"] = True
        print(f"\n[PBT] signal {signum} received -> will stop gracefully after the "
              f"current agent (state will be saved)", flush=True)
    for s in (signal.SIGINT, signal.SIGTERM):
        try:
            signal.signal(s, handler)
        except (ValueError, OSError):
            pass  # not in main thread (e.g. under a test harness) — ignore


@dataclass
class Agent:
    """A persistent PBT population member."""
    agent_id: int
    params: dict
    tag: str
    resume_ckpt: str = ""        # checkpoint this agent will warm-start from next run
    best_ckpt: str = ""          # checkpoint this agent produced last run (its weights)
    score: float = FAIL_SCORE    # latest held-out benchmark score
    win: float = 0.0
    sparc: float = 0.0
    failed: bool = False         # last run crashed / produced no result
    lineage: list = field(default_factory=list)  # [{cycle, event, parent}]

    def to_dict(self) -> dict:
        return asdict(self)

    @staticmethod
    def from_dict(d: dict) -> "Agent":
        known = {f for f in Agent.__dataclass_fields__}  # type: ignore[attr-defined]
        return Agent(**{k: v for k, v in d.items() if k in known})


def perturb(value: float, pmin: float, pmax: float, rng: random.Random,
            factor: float = 1.2) -> float:
    """Perturb a hyperparameter by a random factor in [1/factor, factor], clipped."""
    new_val = value * rng.uniform(1.0 / factor, factor)
    return float(min(max(new_val, pmin), pmax))


def perturb_params(params: dict, rng: random.Random, factor: float = 1.2) -> dict:
    """Return a perturbed COPY of a hyperparameter dict (only PBT_PARAMS keys move)."""
    out = dict(params)
    for k, (pmin, pmax) in PBT_PARAMS.items():
        if k in out:
            out[k] = perturb(out[k], pmin, pmax, rng, factor)
    return out


def initial_population(n: int, warm_ckpt: str = "") -> list[Agent]:
    """Build the initial population of ``n`` agents from INITIAL_POP."""
    agents = []
    for i in range(n):
        base = dict(INITIAL_POP[i % len(INITIAL_POP)])
        agents.append(Agent(agent_id=i, params=base, tag=f"pbt_agent{i}",
                            resume_ckpt=warm_ckpt, lineage=[{"cycle": 0, "event": "init"}]))
    return agents


def rank_agents(agents: list[Agent]) -> list[int]:
    """Return agent indices sorted by score, best first (failed agents sink)."""
    return sorted(range(len(agents)), key=lambda i: agents[i].score, reverse=True)


def exploit_explore(agents: list[Agent], cycle: int, rng: random.Random,
                    factor: float = 1.2) -> tuple[list[Agent], list[dict]]:
    """Copy top-half WEIGHTS+params into bottom-half agents, then perturb.

    Returns (new_agents, inheritance_events).  Each bottom agent's
    ``resume_ckpt`` is set to the top agent's ``best_ckpt`` (the actual file is
    copied by the caller via :func:`copy_checkpoint`), its params become a
    perturbation of the top agent's params, and its lineage gains an
    'exploit' record naming the parent.  Top/middle agents are unchanged.
    """
    n = len(agents)
    ranking = rank_agents(agents)
    half = max(1, n // 2)
    top = ranking[:half]
    bottom = ranking[n - (n // 2):] if n > 1 else []
    new_agents = [Agent(agent_id=a.agent_id, params=dict(a.params), tag=a.tag,
                        resume_ckpt=a.best_ckpt or a.resume_ckpt, best_ckpt=a.best_ckpt,
                        score=a.score, win=a.win, sparc=a.sparc, failed=a.failed,
                        lineage=list(a.lineage)) for a in agents]
    events = []
    for j, b_idx in enumerate(bottom):
        t_idx = top[j % len(top)]
        parent = agents[t_idx]
        child = new_agents[b_idx]
        child.resume_ckpt = parent.best_ckpt or parent.resume_ckpt
        child.params = perturb_params(parent.params, rng, factor)
        child.lineage = list(parent.lineage) + [
            {"cycle": cycle, "event": "exploit", "parent": parent.agent_id,
             "parent_tag": parent.tag, "parent_score": parent.score}]
        events.append({"cycle": cycle, "winner": parent.agent_id, "loser": child.agent_id,
                       "winner_tag": parent.tag, "loser_tag": child.tag,
                       "winner_ckpt": parent.best_ckpt})
    return new_agents, events


def copy_checkpoint(src: str, dst: str) -> bool:
    """Copy a checkpoint file src->dst. Returns True if copied."""
    if src and dst and src != dst and os.path.exists(src):
        os.makedirs(os.path.dirname(dst) or ".", exist_ok=True)
        shutil.copy(src, dst)
        return True
    return False


def promote_best(agents: list[Agent], out: Path, cycle: int) -> Agent:
    """Copy the best agent's weights to pbt_best.pkl + a per-cycle milestone."""
    best = agents[rank_agents(agents)[0]]
    if best.best_ckpt and os.path.exists(best.best_ckpt):
        copy_checkpoint(best.best_ckpt, str(out / "pbt_best.pkl"))
        copy_checkpoint(best.best_ckpt, str(out / f"pbt_cycle{cycle}_best.pkl"))
        print(f"[PBT] promoted {best.tag} (score={best.score:.4f}) -> pbt_best.pkl + "
              f"pbt_cycle{cycle}_best.pkl", flush=True)
    return best


def build_cmd(agent: Agent, steps: int, out: Path, extra_args: list[str]) -> list[str]:
    """Assemble the train_adversarial.py command line for one agent."""
    p = agent.params
    cmd = [
        sys.executable, "-u", str(HERE / "train_adversarial.py"),
        "--tag", agent.tag,
        "--steps", str(steps),
        # 512x8x20 = 81,920 env-steps/iter (audit item 3; was 1024x16 = 327,680 -> the
        # "12M steps = 37 iterations" trap). preflight=warn: per-cycle slices are short
        # by design — PBT's cumulative budget is enforced by its own --steps/--max-cost.
        "--envs", "2048", "--batch", "512", "--minibatches", "8", "--unroll", "20",
        "--preflight", "warn",
        "--evals", str(max(2, steps // 500000)),
        "--approach-weight", str(p["approach_weight"]),
        "--shaping", str(p["shaping"]),
        "--clean-weight", str(p["clean_weight"]),
        "--trade-weight", str(p["trade_weight"]),
        "--fire-shaping", str(p["fire_shaping"]),
        "--face-weight", str(p["face_weight"]),
        "--lr", str(p["lr"]),
        "--entropy", str(p["entropy"]),
        "--upright-weight", str(p["upright_weight"]),
        "--flee-penalty", str(p["flee_penalty"]),
        "--close-bonus", str(p["close_bonus"]),
    ]
    if "gate_weight" in p:
        cmd += ["--gate-weight", str(p["gate_weight"])]
    if "rnd_coefficient" in p:
        cmd += ["--rnd-coefficient", str(p["rnd_coefficient"])]
    if "her_coefficient" in p:
        cmd += ["--her-coefficient", str(p["her_coefficient"])]
    if agent.resume_ckpt and os.path.exists(agent.resume_ckpt):
        cmd += ["--resume", agent.resume_ckpt]
    if extra_args:
        cmd += extra_args
    return cmd


def subprocess_runner(cmd: list[str], log_path: Path) -> int:
    """Default runner: launch the training subprocess, stream to a log, return rc."""
    with open(log_path, "w") as lf:
        proc = subprocess.run(cmd, stdout=lf, stderr=subprocess.STDOUT, env={**os.environ})
    return proc.returncode


def read_result(tag: str, out: Path) -> dict | None:
    """Read the held-out result the trainer wrote, or None if it is missing."""
    state_file = out / f"{tag}_state.json"
    if not state_file.exists():
        return None
    try:
        return json.loads(state_file.read_text())
    except (json.JSONDecodeError, OSError):
        return None


def run_agent(agent: Agent, steps: int, out: Path, extra_args: list[str],
              runner=subprocess_runner) -> Agent:
    """Run one agent for ``steps`` and update its score/best_ckpt/failed in place."""
    cmd = build_cmd(agent, steps, out, extra_args)
    log_path = out / f"{agent.tag}.log"
    print(f"[PBT] {agent.tag}: launching (resume={agent.resume_ckpt or 'scratch'}) "
          f"params={ {k: round(v, 4) for k, v in agent.params.items()} }", flush=True)
    rc = runner(cmd, log_path)
    result = read_result(agent.tag, out)
    if rc != 0 or result is None:
        agent.failed = True
        agent.score = FAIL_SCORE
        print(f"[PBT] {agent.tag}: FAILED (rc={rc}, result={'missing' if result is None else 'present'}) "
              f"-> see {log_path}", flush=True)
        return agent
    agent.failed = False
    agent.score = float(result.get("best_score", FAIL_SCORE))
    agent.win = float(result.get("best_win", 0.0))
    agent.sparc = float(result.get("best_sparc", 0.0))
    agent.best_ckpt = str(out / f"{agent.tag}_best.pkl")
    print(f"[PBT] {agent.tag}: score={agent.score:.4f} win={agent.win:.3f} sparc={agent.sparc:.3f}",
          flush=True)
    return agent


# ------------------------------- video render --------------------------------
def subprocess_video_runner(ckpt: str, out_path: str, label: str, render_args=(),
                            steps: int = 200, sep: float = 0.6, timeout: int = 900):
    """Render a combat MP4 from a checkpoint via render_fight_video.py.

    ``render_args`` carries the obs/lidar/her flags so the render env matches the
    trained checkpoint (forwarded verbatim; render_fight_video ignores unknown
    train flags). Returns (ok, message); any failure is captured (never raised)
    so callers can log it without crashing training.
    """
    cmd = [sys.executable, str(HERE / "render_fight_video.py"),
           "--a", ckpt, "--out", out_path,
           "--steps", str(steps), "--sep", str(sep), "--label", label,
           *list(render_args)]
    env = {**os.environ}
    # Force a renderer GL backend (the headless physics environment leaves it empty,
    # the CPU renderer cannot use). Override with PBT_RENDER_GL if egl works.
    env["MUJOCO_GL"] = os.environ.get("PBT_RENDER_GL", "osmesa")
    try:
        proc = subprocess.run(cmd, env=env, timeout=timeout, capture_output=True, text=True)
        msg = (proc.stdout or "") + (proc.stderr or "")
        return proc.returncode == 0, msg.strip()[-300:]
    except Exception as e:  # subprocess timeout, missing file, encoder error, ...
        return False, f"{type(e).__name__}: {e}"


def video_render_args(agent: Agent, render_base) -> list[str]:
    """Flags so the render env matches the checkpoint (passive B + lidar/her dims)."""
    args = list(render_base or [])
    args += ["--her-coefficient", str(agent.params.get("her_coefficient", 0.0)), "--passive-b"]
    return args


def maybe_render_video(agent: Agent, out: Path, cycle: int, video_every: int,
                       video_runner=subprocess_video_runner, enabled: bool = True,
                       fatal: bool = False, render_base=None):
    """Render the best agent's checkpoint to MP4 on schedule (the milestone/render path).

    Returns the output path on success, else None. Failures are LOGGED and
    non-fatal by default (``fatal=True`` re-raises) so a broken encoder never
    kills a long training run.
    """
    if not enabled or video_every <= 0 or (cycle % video_every) != 0:
        return None
    if not agent.best_ckpt or not os.path.exists(agent.best_ckpt):
        print(f"[PBT] video: no checkpoint for {agent.tag}; skipping cycle {cycle}", flush=True)
        return None
    vdir = out / "videos"
    vdir.mkdir(parents=True, exist_ok=True)
    out_path = str(vdir / f"pbt_cycle{cycle:03d}_{agent.tag}_best.mp4")
    render_args = video_render_args(agent, render_base)
    try:
        ok, msg = video_runner(agent.best_ckpt, out_path, f"cycle{cycle} {agent.tag}", render_args)
    except Exception as e:
        if fatal:
            raise
        print(f"[PBT] video: render crashed (caught, non-fatal): {type(e).__name__}: {e}", flush=True)
        return None
    if ok:
        print(f"[PBT] video: rendered cycle {cycle} -> {out_path}", flush=True)
        return out_path
    print(f"[PBT] video: render FAILED for cycle {cycle} (logged, non-fatal): {msg}", flush=True)
    if fatal:
        raise RuntimeError(f"video render failed: {msg}")
    return None


# ------------------------------- state I/O -----------------------------------
def save_state(path: Path, agents: list[Agent], cycle: int, warm_ckpt: str,
               events: list[dict], budget: Budget | None = None,
               stop_reason: str = "in_progress", started_at: str = "") -> None:
    state = {
        "cycle": cycle,
        "warm_ckpt": warm_ckpt,
        "stop_reason": stop_reason,
        "started_at": started_at,
        "meta": budget.as_dict() if budget is not None else {},
        "resume_cmd": f"python pbt_train.py --resume-state {path} --cycles <N> -- <extra args>",
        "agents": [a.to_dict() for a in agents],
        "last_events": events,
        "ranking": [agents[i].agent_id for i in rank_agents(agents)],
    }
    path.write_text(json.dumps(state, indent=2))


def load_state(path) -> tuple[list[Agent], int, str]:
    state = json.loads(Path(path).read_text())
    agents = [Agent.from_dict(d) for d in state["agents"]]
    return agents, int(state.get("cycle", 0)), str(state.get("warm_ckpt", ""))


def load_meta(path) -> dict:
    """Resume metadata: cumulative elapsed time, stop reason, start timestamp."""
    state = json.loads(Path(path).read_text())
    meta = state.get("meta", {}) or {}
    return {"stop_reason": state.get("stop_reason", ""),
            "started_at": state.get("started_at", ""),
            "elapsed_s": float(meta.get("elapsed_s", 0.0))}


# ------------------------------- cycle / loop --------------------------------
def pbt_cycle(agents: list[Agent], cycle: int, steps: int, out: Path,
              extra_args: list[str], rng: random.Random,
              runner=subprocess_runner, budget: Budget | None = None,
              stop_flag=None) -> tuple[list[Agent], list[dict]]:
    """Train every agent (resumed from its own ckpt), rank, then exploit/explore.

    Budget/stop is checked BEFORE each agent so a cap or signal stops cleanly
    between agents (raising :class:`BudgetExceeded`), never mid-checkpoint.
    """
    print(f"\n{'='*64}\n[PBT] CYCLE {cycle} (steps/agent={steps}, agents={len(agents)})\n{'='*64}",
          flush=True)
    for agent in agents:
        reason = (budget.exceeded() if budget is not None else None) \
            or ("interrupted" if (stop_flag and stop_flag()) else None)
        if reason:
            raise BudgetExceeded(reason)
        run_agent(agent, steps, out, extra_args, runner=runner)

    ranking = rank_agents(agents)
    print(f"[PBT] ranking (best->worst): "
          f"{[(agents[i].tag, round(agents[i].score, 3)) for i in ranking]}", flush=True)
    n_failed = sum(1 for a in agents if a.failed)
    if n_failed:
        print(f"[PBT] WARNING: {n_failed}/{len(agents)} agent(s) failed this cycle", flush=True)
    if all(a.failed for a in agents):
        raise RuntimeError(f"[PBT] all {len(agents)} agents failed in cycle {cycle}; aborting")

    new_agents, events = exploit_explore(agents, cycle, rng)
    for ev in events:
        winner = next(a for a in agents if a.agent_id == ev["winner"])
        loser_new = next(a for a in new_agents if a.agent_id == ev["loser"])
        dst = str(out / f"{loser_new.tag}_inherited.pkl")
        if copy_checkpoint(winner.best_ckpt, dst):
            loser_new.resume_ckpt = dst
            print(f"[PBT] exploit: {winner.tag} weights -> {loser_new.tag} "
                  f"(+perturbed hp); resume={dst}", flush=True)
        else:
            print(f"[PBT] exploit: {winner.tag} -> {loser_new.tag} hp only "
                  f"(no checkpoint to copy)", flush=True)
    return new_agents, events


def run_pbt(agents: list[Agent], start_cycle: int, n_cycles: int, steps_per_cycle: int,
            out: Path, extra_args: list[str], rng: random.Random, warm_ckpt: str = "",
            runner=subprocess_runner, video_runner=subprocess_video_runner,
            budget: Budget | None = None, video_every: int = 1, video_enabled: bool = True,
            video_fatal: bool = False, started_at: str = "", stop_flag=None):
    """Drive ``n_cycles`` PBT cycles with budget caps, checkpoints, video, and resume.

    Returns (agents, last_completed_cycle, warm_ckpt, stop_reason). Stops
    GRACEFULLY (state always saved) on a budget cap, a signal, or all-agents-fail.
    Injectable ``runner``/``video_runner``/``budget.clock`` make this GPU-free testable.
    """
    budget = budget or Budget()
    stop_flag = stop_flag if stop_flag is not None else stop_requested
    state_path = out / "pbt_state.json"
    stop_reason = "completed"
    last_cycle = start_cycle
    events: list[dict] = []
    for c in range(start_cycle + 1, start_cycle + n_cycles + 1):
        reason = budget.exceeded() or ("interrupted" if stop_flag() else None)
        if reason:
            stop_reason = reason
            print(f"[PBT] {reason} before cycle {c} "
                  f"(elapsed {budget.elapsed_s()/60:.1f} min, est ${budget.est_cost_usd():.2f}); "
                  f"stopping gracefully", flush=True)
            break
        try:
            agents, events = pbt_cycle(agents, c, steps_per_cycle, out, extra_args, rng,
                                       runner=runner, budget=budget, stop_flag=stop_flag)
        except BudgetExceeded as e:
            stop_reason = e.reason
            print(f"[PBT] {e.reason} during cycle {c}; stopping gracefully "
                  f"(elapsed {budget.elapsed_s()/60:.1f} min, est ${budget.est_cost_usd():.2f})",
                  flush=True)
            break
        except RuntimeError as e:
            stop_reason = "all_failed"
            print(f"[PBT] {e}", flush=True)
            break
        last_cycle = c
        best = promote_best(agents, out, c)
        if best.best_ckpt:
            warm_ckpt = best.best_ckpt
        maybe_render_video(best, out, c, video_every, video_runner,
                           enabled=video_enabled, fatal=video_fatal, render_base=extra_args)
        save_state(state_path, agents, c, warm_ckpt, events, budget=budget,
                   stop_reason="in_progress", started_at=started_at)
        print(f"[PBT] cycle {c} saved -> {state_path} "
              f"(elapsed {budget.elapsed_s()/60:.1f} min, est ${budget.est_cost_usd():.2f}, "
              f"warm_ckpt={warm_ckpt})", flush=True)
    save_state(state_path, agents, last_cycle, warm_ckpt, events, budget=budget,
               stop_reason=stop_reason, started_at=started_at)
    return agents, last_cycle, warm_ckpt, stop_reason


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--agents", type=int, default=4)
    ap.add_argument("--steps", type=int, default=4_000_000, help="TOTAL env-steps per agent across all cycles")
    ap.add_argument("--cycles", type=int, default=3)
    ap.add_argument("--warm-ckpt", default="")
    ap.add_argument("--resume-state", default=None,
                    help="continue a population from a saved pbt_state.json")
    ap.add_argument("--seed", type=int, default=0, help="RNG seed for hyperparameter perturbation")
    # budget caps (graceful stop)
    ap.add_argument("--max-runtime-min", type=float, default=0.0,
                    help="stop gracefully once cumulative runtime (across resumes) reaches this (0=off)")
    ap.add_argument("--max-cost-usd", type=float, default=0.0,
                    help="stop gracefully once estimated spend reaches this (0=off)")
    ap.add_argument("--cost-per-hour", type=float, default=DEFAULT_COST_PER_HOUR,
                    help="GPU $/hr used for the cost cap estimate")
    # video
    ap.add_argument("--video-every", type=int, default=1,
                    help="render the best checkpoint to an MP4 every N cycles (0=off)")
    ap.add_argument("--no-video", action="store_true", help="disable milestone video rendering")
    ap.add_argument("--video-fatal", action="store_true",
                    help="treat a video render failure as fatal (default: log and continue)")
    ap.add_argument("--rnd-feature", choices=["proprio", "tactical"], default="tactical",
                    help="RND novelty feature space forwarded to every agent (audit C19: "
                         "tactical = engagement geometry/contacts/striker; proprio funded "
                         "the joint-jitter exploit)")
    ap.add_argument("extra_args", nargs="*", default=None,
                    help="extra args forwarded to train_adversarial.py (after --)")
    args = ap.parse_args(argv)

    install_signal_handlers()
    out = default_out()
    rng = random.Random(args.seed)
    steps_per_cycle = max(1, args.steps // args.cycles)
    extra_args = list(args.extra_args or [])
    if "--rnd-feature" not in extra_args:
        extra_args += ["--rnd-feature", args.rnd_feature]
    started_at = time.strftime("%Y-%m-%dT%H:%M:%S")
    prior_elapsed = 0.0

    if args.resume_state:
        agents, start_cycle, warm_ckpt = load_state(Path(args.resume_state))
        meta = load_meta(Path(args.resume_state))
        prior_elapsed = meta["elapsed_s"]
        started_at = meta["started_at"] or started_at
        print(f"[PBT] resumed {len(agents)} agents from {args.resume_state} "
              f"(after cycle {start_cycle}, prior elapsed {prior_elapsed/60:.1f} min, "
              f"was '{meta['stop_reason']}')", flush=True)
    else:
        warm_ckpt = args.warm_ckpt
        agents = initial_population(args.agents, warm_ckpt)
        start_cycle = 0

    budget = Budget(max_runtime_s=args.max_runtime_min * 60.0, max_cost_usd=args.max_cost_usd,
                    cost_per_hour=args.cost_per_hour, prior_elapsed_s=prior_elapsed)
    print(f"[PBT] start: agents={len(agents)} cycles={args.cycles} steps/cycle={steps_per_cycle} "
          f"caps(runtime={args.max_runtime_min}min cost=${args.max_cost_usd}@${args.cost_per_hour}/hr) "
          f"video={'off' if args.no_video else f'every {args.video_every} cycle(s)'}", flush=True)

    agents, last_cycle, warm_ckpt, stop_reason = run_pbt(
        agents, start_cycle, args.cycles, steps_per_cycle, out, extra_args, rng,
        warm_ckpt=warm_ckpt, budget=budget, video_every=args.video_every,
        video_enabled=not args.no_video, video_fatal=args.video_fatal,
        started_at=started_at, stop_flag=stop_requested)

    ranked = rank_agents(agents)
    best = agents[ranked[0]]
    print(f"\n[PBT] STOP ({stop_reason}) after cycle {last_cycle}: "
          f"best={best.tag} score={best.score:.4f} win={best.win:.3f} sparc={best.sparc:.3f} "
          f"ckpt={best.best_ckpt}", flush=True)
    print(f"[PBT] elapsed {budget.elapsed_s()/60:.1f} min, est ${budget.est_cost_usd():.2f}; "
          f"best weights -> {out/'pbt_best.pkl'}", flush=True)
    print(f"[PBT] best lineage: {best.lineage}", flush=True)
    if stop_reason in ("runtime_cap", "cost_cap", "interrupted"):
        print(f"[PBT] to continue: python pbt_train.py --resume-state {out/'pbt_state.json'} "
              f"--cycles <N> -- {' '.join(extra_args)}", flush=True)
    return agents


if __name__ == "__main__":
    main()
