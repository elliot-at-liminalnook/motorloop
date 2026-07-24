# SPDX-License-Identifier: MIT
"""Executable, resume-safe 31-rung robot training ladder.

Each rung has a concrete environment or verification command, a held-out metric
gate, and an explicit warm-start family.  A failed gate retries by continuing
the same checkpoint, then stops with the last accepted policy preserved.  This
is the machine-readable source corresponding to Stage 07 in
``notes/modeling-to-rl-pipeline.html``.

Examples:

  # Inspect the exact sequence and gates.
  .venv-warp/bin/python sim/robot/training_ladder.py list

  # CPU plumbing proof; executes every rung with tiny PPO/search budgets.
  .venv-warp/bin/python sim/robot/training_ladder.py run --tiny --to 31

  # Real CUDA sequence (safe to restart with the same --out directory).
  python sim/robot/training_ladder.py run --device cuda --envs 1024 \
      --out /root/proj/out/training_ladder --resume
"""

from __future__ import annotations

import argparse
from dataclasses import asdict, dataclass
import hashlib
import json
import math
import operator
from pathlib import Path
import random
import subprocess
import sys
import time
from typing import Iterable


ROOT = Path(__file__).resolve().parents[2]
TRAINER = ROOT / "sim/robot/train_mesh_warp.py"
SEARCH = ROOT / "sim/robot/warp_search.py"
GAIT_SEARCH = ROOT / "sim/robot/search_ladder_gait.py"
EVALUATOR = ROOT / "sim/robot/warp_eval.py"
VERIFY_BODY = ROOT / "sim/robot/validate_body.py"
LEGACY_WALK_TEACHER = ROOT / "sim/build/gpu/out/walker/walker_step.pt"
CONTROLLER_CONTRACT = "universal256x14:v1"


@dataclass(frozen=True)
class Gate:
    metric: str
    comparison: str
    threshold: float
    role: str = "outcome"

    def passes(self, metrics: dict) -> bool:
        if self.metric not in metrics:
            return False
        fn = {">=": operator.ge, "<=": operator.le}[self.comparison]
        return bool(fn(float(metrics[self.metric]), self.threshold))

    def describe(self) -> str:
        return f"[{self.role}] {self.metric} {self.comparison} {self.threshold:g}"


@dataclass(frozen=True)
class Rung:
    number: int
    tier: str
    slug: str
    name: str
    demand: str
    kind: str
    family: str
    geometry: str | None
    gates: tuple[Gate, ...]
    step_scale: float = 1.0


def g(metric: str, comparison: str, threshold: float) -> Gate:
    return Gate(metric, comparison, threshold)


def c(metric: str, comparison: str, threshold: float) -> Gate:
    """A non-negotiable physical/safety boundary, not a preferred behavior."""
    return Gate(metric, comparison, threshold, role="constraint")


# These remain valuable telemetry, but using them as acceptance criteria would
# define *how* a skill must look instead of whether its external outcome works.
DIAGNOSTIC_ONLY_METRICS = frozenset({
    "diagsync",
    "duty",
    "foot_cycle_duty",
    "ladder_step_action_score",
    "ladder_step_clock",
    "ladder_swing_clearance",
    "ladder_worst_swing_clearance",
})


def merge_candidate_archives(*artifacts: dict) -> list[dict]:
    """Union searched behavior candidates without losing earlier tradeoffs."""
    merged: list[dict] = []
    seen: set[tuple[float, ...]] = set()
    for artifact in artifacts:
        groups = [artifact.get("archive"), artifact.get("pareto"),
                  artifact.get("history"), [artifact.get("best")]]
        for group in groups:
            for candidate in group or []:
                values = candidate.get("parameters") if isinstance(candidate, dict) else None
                if not isinstance(values, list):
                    continue
                key = tuple(float(value) for value in values)
                if key in seen:
                    continue
                seen.add(key)
                merged.append(candidate)
    return merged


RUNGS: tuple[Rung, ...] = (
    Rung(1, "A", "hold_torque", "Hold torque", "deliver derived torque at a loaded joint",
         "verify", "hardware", None, ()),
    Rung(2, "A", "stand_settle", "Stand and settle", "settle from a noisy stance",
         "ppo", "universal", "universal_control",
         (g("up", ">=", 0.85), g("speed", "<=", 0.15),
          c("catrate", "<=", 0.001), c("fallrate", "<=", 0.0001))),
    Rung(3, "A", "balance_push", "Balance under push", "reject alternating lateral impulses",
         "ppo", "universal", "universal_control",
         (g("up", ">=", 0.72), c("catrate", "<=", 0.001),
          c("fallrate", "<=", 0.0001))),
    Rung(4, "A", "pose_vector", "Track a pose vector", "track randomized 12-joint targets",
         "ppo", "universal", "universal_control",
         (g("ladder_pose_score", ">=", 0.45), c("catrate", "<=", 0.001),
          c("fallrate", "<=", 0.0001))),
    Rung(5, "A", "height_control", "Height control", "track randomized crouch/stand height",
         "ppo", "universal", "universal_control",
         (g("ladder_height_score", ">=", 0.45), c("catrate", "<=", 0.001),
          c("fallrate", "<=", 0.0001))),
    Rung(6, "B", "step_in_place", "Step in place", "lift and replace feet without translating",
         "ppo", "universal", "universal_control",
         (g("foot_air_fraction_min", ">=", 0.05),
          g("speed", "<=", 0.20), g("up", ">=", 0.85),
          c("catrate", "<=", 0.001), c("fallrate", "<=", 0.0001))),
    Rung(7, "B", "walk_forward", "Walk forward", "first fixed-speed traveling gait",
         "ppo", "universal", "universal_control",
         (g("xprogress", ">=", 0.10), g("lateral", "<=", 0.15),
          g("up", ">=", 0.85), c("catrate", "<=", 0.001),
          c("fallrate", "<=", 0.0001))),
    Rung(8, "B", "velocity_tracking", "Velocity tracking", "track a forward speed range",
         "ppo", "universal", "universal_control",
         (g("track", ">=", 0.40), g("xprogress", ">=", 0.06),
          c("catrate", "<=", 0.001), c("fallrate", "<=", 0.0001))),
    Rung(9, "B", "turn_in_place", "Turn in place", "track yaw rate at zero translation",
         "ppo", "universal", "universal_control",
         (g("ladder_yaw_score", ">=", 0.35), g("speed", "<=", 0.20),
          c("catrate", "<=", 0.001), c("fallrate", "<=", 0.0001))),
    Rung(10, "B", "omnidirectional", "Omnidirectional", "track planar and yaw commands together",
         "ppo", "universal", "universal_control",
         (g("track", ">=", 0.30), g("xprogress", ">=", 0.05),
          c("catrate", "<=", 0.001), c("fallrate", "<=", 0.0001))),
    Rung(11, "B", "heading_hold", "Heading hold", "translate while regulating heading",
         "ppo", "universal", "universal_control",
         (g("ladder_heading_score", ">=", 0.40), g("xprogress", ">=", 0.05),
          c("catrate", "<=", 0.001), c("fallrate", "<=", 0.0001))),
    Rung(12, "B", "stop_command", "Stop on command", "switch between travel and a true hold",
         "ppo", "universal", "universal_control",
         (g("ladder_stop_score", ">=", 0.30),
          g("ladder_move_progress", ">=", 0.035),
          c("catrate", "<=", 0.001), c("fallrate", "<=", 0.0001))),
    Rung(13, "C", "servo_true", "Servo-true motors", "restore the measured torque-speed droop",
         "ppo", "universal", "universal_control",
         (g("xprogress", ">=", 0.08), c("catrate", "<=", 0.001),
          c("fallrate", "<=", 0.0001))),
    Rung(14, "C", "stumble", "Stumble recovery", "cross a physical trip bar",
         "ppo", "universal", "universal_control",
         (g("xprogress", ">=", 0.08), g("up", ">=", 0.72),
          c("catrate", "<=", 0.001), c("fallrate", "<=", 0.0001)), 1.25),
    Rung(15, "C", "push_recovery", "Push recovery", "resume gait after stronger mid-stride pushes",
         "ppo", "universal", "universal_control",
         (g("xprogress", ">=", 0.07), g("up", ">=", 0.72),
          c("catrate", "<=", 0.001), c("fallrate", "<=", 0.0001)), 1.25),
    Rung(16, "C", "rough_terrain", "Rough terrain", "cross raised tiles with varied friction",
         "ppo", "universal", "universal_control",
         (g("xprogress", ">=", 0.06), g("up", ">=", 0.68),
          c("catrate", "<=", 0.001), c("fallrate", "<=", 0.0001)), 1.5),
    Rung(17, "C", "slope", "Slope walking", "travel on a nine-degree physical incline",
         "ppo", "universal", "universal_control",
         (g("xprogress", ">=", 0.04), g("up", ">=", 0.68),
          c("catrate", "<=", 0.001), c("fallrate", "<=", 0.0001)), 1.25),
    Rung(18, "C", "payload", "Payload carry", "walk with 30 percent extra torso mass",
         "ppo", "universal", "universal_control",
         (g("xprogress", ">=", 0.06), g("up", ">=", 0.68),
          c("catrate", "<=", 0.001), c("fallrate", "<=", 0.0001)), 1.25),
    Rung(19, "D", "return_origin", "Return to origin", "recover from randomized displaced starts",
         "ppo", "universal", "universal_control",
         (g("ladder_goal_hit", ">=", 0.02), g("xprogress", ">=", 0.04),
          c("catrate", "<=", 0.001), c("fallrate", "<=", 0.0001)), 1.5),
    Rung(20, "D", "waypoint", "Go to waypoint", "reach one commanded planar goal",
         "ppo", "universal", "universal_control",
         (g("ladder_goal_hit", ">=", 0.02), g("xprogress", ">=", 0.04),
          c("catrate", "<=", 0.001), c("fallrate", "<=", 0.0001)), 1.5),
    Rung(21, "D", "route", "Route following", "hit an ordered four-waypoint route",
         "ppo", "universal", "universal_control",
         (g("ladder_goal_hit", ">=", 0.015), g("xprogress", ">=", 0.04),
          c("catrate", "<=", 0.001), c("fallrate", "<=", 0.0001)), 1.75),
    Rung(22, "D", "route_correction", "Route correction", "reacquire route after a forced detour",
         "ppo", "universal", "universal_control",
         (g("ladder_goal_hit", ">=", 0.010), g("xprogress", ">=", 0.035),
          c("catrate", "<=", 0.001), c("fallrate", "<=", 0.0001)), 1.75),
    Rung(23, "D", "degraded_lidar", "Degraded-lidar nav",
         "avoid physical obstacles with noisy, dropped, one-step-late 144-ray sensing",
         "ppo", "universal", "universal_control",
         (g("ladder_goal_hit", ">=", 0.010), g("xprogress", ">=", 0.03),
          g("ladder_obstacle_clearance", ">=", 0.02),
          c("catrate", "<=", 0.001), c("fallrate", "<=", 0.0001)), 2.0),
    Rung(24, "E", "approach_dummy", "Approach a dummy", "close on a passive target",
         "ppo", "universal", "universal_control",
         (g("ladder_approach", ">=", 0.0002),
          g("ladder_target_distance", "<=", 0.45),
          c("fallrate", "<=", 0.001)), 1.5),
    Rung(25, "E", "strike_target", "Strike a target", "close and fire the pneumatic striker",
         "ppo", "universal", "universal_control",
         (g("ladder_rod_hit", ">=", 0.0001),
          g("ladder_target_distance", "<=", 0.50),
          c("fallrate", "<=", 0.001)), 1.75),
    Rung(26, "E", "commanded_leg_kick", "Commanded leg kick",
         "switch FL/FR/RL/RR at runtime; selected foot attacks while three legs support",
         "ppo", "universal", "universal_control",
         (g("attack_selected_hit_worst_leg", ">=", 0.0001),
          g("attack_kick_speed_worst_leg", ">=", 0.03),
          g("attack_support_worst_leg", ">=", 0.50),
          g("attack_wrong_hit_worst_leg", "<=", 0.10),
          c("attack_switch_fallrate", "<=", 0.001)), 2.0),
    Rung(27, "E", "pursue_evader", "Pursue an evader", "close on a scripted moving target",
         "ppo", "universal", "universal_control",
         (g("ladder_approach", ">=", 0.0001),
          g("ladder_target_distance", "<=", 0.50),
          c("fallrate", "<=", 0.001)), 1.75),
    Rung(28, "E", "frozen_self", "Beat a frozen self", "fight one frozen prior policy",
         "ppo", "universal", "universal_control",
         (g("ladder_combat_margin", ">=", 0.00005),
          g("ladder_rod_hit", ">=", 0.0001),
          c("fallrate", "<=", 0.002)), 2.0),
    Rung(29, "E", "pfsp", "PFSP league", "train against a growing hall of frozen snapshots",
         "pfsp", "universal", "universal_control",
         (g("ladder_combat_margin", ">=", 0.00005),
          g("ladder_rod_hit", ">=", 0.0001),
          c("fallrate", "<=", 0.002)), 3.0),
    Rung(30, "F", "cross_morphology", "Cross-morphology",
         "one design-conditioned policy across actual compiled bodies",
         "ppo", "universal", "universal_control",
         (g("track", ">=", 0.40), g("xprogress", ">=", 0.05),
          c("catrate", "<=", 0.001), c("fallrate", "<=", 0.0001)), 2.0),
    Rung(31, "F", "codesign_loop", "Co-design loop", "CEM-search body parameters using policy return",
         "search", "universal", None, (g("improvement", ">=", 0.001),), 1.0),
)


def validate_manifest() -> None:
    numbers = [r.number for r in RUNGS]
    if numbers != list(range(1, 32)):
        raise ValueError(f"training ladder must contain exactly contiguous rungs 1..31: {numbers}")
    slugs = [r.slug for r in RUNGS]
    if len(slugs) != len(set(slugs)):
        raise ValueError("training ladder slugs must be unique")
    invalid_roles = [(rung.number, gate.role) for rung in RUNGS for gate in rung.gates
                     if gate.role not in {"outcome", "constraint"}]
    if invalid_roles:
        raise ValueError(f"training ladder has invalid gate roles: {invalid_roles}")
    prescriptive = [(rung.number, gate.metric) for rung in RUNGS for gate in rung.gates
                    if gate.metric in DIAGNOSTIC_ONLY_METRICS]
    if prescriptive:
        raise ValueError(
            "diagnostic/style metrics cannot be promotion gates: "
            f"{prescriptive}")


def selected_rungs(first: int, last: int) -> Iterable[Rung]:
    return (rung for rung in RUNGS if first <= rung.number <= last)


# Walk-first acquisition: every learning rung already shares one "universal"
# policy family, and the commands-only contract makes rung numbering invisible
# to the policy — so acquisition order is a curriculum choice, not an
# identity.  Velocity tracking (with a standing-hold stripe) is learned from
# scratch FIRST; the stand/pose/step rungs then certify as commanded special
# cases of locomotion instead of training a standing attractor that stepping
# must later escape.
WALK_FIRST_ORDER = (1, 8, 10, 9, 11, 12, 7, 2, 3, 4, 5, 6) + tuple(range(13, 32))


def ordered_rungs(first: int, last: int, walk_first: bool) -> Iterable[Rung]:
    if not walk_first:
        yield from selected_rungs(first, last)
        return
    by_number = {rung.number: rung for rung in RUNGS}
    for number in WALK_FIRST_ORDER:
        if first <= number <= last:
            yield by_number[number]


class LadderRunner:
    def __init__(self, args):
        self.args = args
        self.out = Path(args.out).resolve()
        self.out.mkdir(parents=True, exist_ok=True)
        (self.out / "logs").mkdir(exist_ok=True)
        self.state_path = self.out / "ladder_state.json"
        self.regression_matrix_path = self.out / "regression_matrix.json"
        self.manifest_path = self.out / "ladder_manifest.json"
        self.manifest_path.write_text(json.dumps(
            [asdict(rung) for rung in RUNGS],
            indent=2) + "\n")
        self.state = self._load_state() if args.resume else self._fresh_state()
        # Version-one ladder states predate the durable forgetting matrix.  The
        # migration is intentionally additive: accepted checkpoints and their
        # exact baselines remain untouched.
        self.state["version"] = max(int(self.state.get("version", 1)), 2)
        self.state["regression_matrix_path"] = str(self.regression_matrix_path)
        if not isinstance(self.state.get("test_out"), dict):
            self.state["test_out"] = {}
        self.regression_matrix = self._load_regression_matrix()
        if self.state.get("controller_contract") != CONTROLLER_CONTRACT:
            archived = self._quarantine_stale_successor_candidates(0)
            self.state["controller_contract"] = CONTROLLER_CONTRACT
            self.state.setdefault("controller_migrations", []).append({
                "time": time.time(), "contract": CONTROLLER_CONTRACT,
                "archived": archived,
            })
            self.state.pop("failed", None)
            self._save()
        self._failed_on_entry = self.state.get("failed", {}).get("rung")

    @staticmethod
    def _fresh_state() -> dict:
        return {"version": 2, "completed": [], "checkpoints": {}, "metrics": {},
                "attempts": {}, "retention_baselines": {}, "retention_history": [],
                "replay_pressures": {},
                "retention_opponents": {},
                "test_out": {},
                "started": time.time(), "updated": time.time()}

    @staticmethod
    def _fresh_regression_matrix() -> dict:
        return {"schema_version": 1, "updated": time.time(), "candidates": {}}

    def _load_regression_matrix(self) -> dict:
        if self.args.resume and self.regression_matrix_path.exists():
            matrix = json.loads(self.regression_matrix_path.read_text())
            matrix.setdefault("schema_version", 1)
            matrix.setdefault("candidates", {})
            return matrix
        return self._fresh_regression_matrix()

    def _load_state(self) -> dict:
        return json.loads(self.state_path.read_text()) if self.state_path.exists() else self._fresh_state()

    def _save(self) -> None:
        self.state["updated"] = time.time()
        temp = self.state_path.with_suffix(".tmp")
        temp.write_text(json.dumps(self.state, indent=2, sort_keys=True) + "\n")
        temp.replace(self.state_path)

    def _save_regression_matrix(self) -> None:
        """Atomically publish the cross-skill replay table after every check."""
        self.regression_matrix["updated"] = time.time()
        temp = self.regression_matrix_path.with_suffix(".tmp")
        temp.write_text(json.dumps(
            self.regression_matrix, indent=2, sort_keys=True) + "\n")
        temp.replace(self.regression_matrix_path)

    def _regression_candidate_id(self, rung: Rung, checkpoint: str) -> str:
        attempt = int(self.state.get("attempts", {}).get(str(rung.number), 0))
        return f"{Path(checkpoint).stem}__rung_{rung.number:02d}__attempt_{attempt:02d}"

    @staticmethod
    def _checkpoint_sha256(checkpoint: str) -> str | None:
        path = Path(checkpoint)
        if not path.is_file():
            return None
        digest = hashlib.sha256()
        with path.open("rb") as stream:
            for chunk in iter(lambda: stream.read(1024 * 1024), b""):
                digest.update(chunk)
        return digest.hexdigest()

    def _record_regression_task(self, candidate_rung: Rung, checkpoint: str,
                                replayed_rung: Rung, checks: list[dict],
                                passed: bool, kind: str) -> None:
        """Record one cell in candidate-checkpoint x learned-task space."""
        candidate_id = self._regression_candidate_id(candidate_rung, checkpoint)
        candidates = self.regression_matrix.setdefault("candidates", {})
        candidate = candidates.setdefault(candidate_id, {
            "candidate_rung": candidate_rung.number,
            "candidate_name": candidate_rung.name,
            "family": candidate_rung.family,
            "checkpoint": str(checkpoint),
            "checkpoint_sha256": self._checkpoint_sha256(checkpoint),
            "attempt": int(self.state.get("attempts", {}).get(
                str(candidate_rung.number), 0)),
            "tasks": {},
        })
        candidate["recorded"] = time.time()
        candidate["tasks"][str(replayed_rung.number)] = {
            "task_rung": replayed_rung.number,
            "task_name": replayed_rung.name,
            "kind": kind,
            "pass": bool(passed),
            "checks": checks,
        }
        candidate["all_pass"] = bool(candidate["tasks"]) and all(
            task["pass"] for task in candidate["tasks"].values())
        self._save_regression_matrix()

    @staticmethod
    def _current_task_regression_checks(rung: Rung, metrics: dict) -> list[dict]:
        """Represent the new task's fixed-seed gate in the same matrix schema."""
        checks = []
        for gate in rung.gates:
            raw = metrics.get(gate.metric)
            current = float(raw) if isinstance(raw, (int, float)) else None
            margin = (None if current is None else
                      (current - gate.threshold if gate.comparison == ">="
                       else gate.threshold - current))
            checks.append({
                "metric": gate.metric,
                "comparison": gate.comparison,
                "baseline": gate.threshold,
                "baseline_kind": "acceptance_threshold",
                "current": current,
                "gate_threshold": gate.threshold,
                "allowance": 0.0,
                "acceptance_boundary": gate.threshold,
                "margin": margin,
                "regression": None,
                "pass": margin is not None and margin >= 0.0,
            })
        return checks

    @staticmethod
    def _gate_signature() -> str:
        payload = [(rung.number, [asdict(gate) for gate in rung.gates]) for rung in RUNGS]
        return json.dumps(payload, sort_keys=True, separators=(",", ":"))

    def _quarantine_checkpoint(self, number: int, value: str | None) -> None:
        """Preserve an invalidated artifact where it cannot be resumed by name."""
        if not value:
            return
        path = Path(value)
        if not path.exists():
            return
        diagnostics = self.out / "diagnostics"
        diagnostics.mkdir(exist_ok=True)
        stamp = int(time.time())
        destination = diagnostics / f"contract_invalidated_rung_{number:02d}_{stamp}.pt"
        suffix = 1
        while destination.exists():
            destination = diagnostics / (
                f"contract_invalidated_rung_{number:02d}_{stamp}_{suffix}.pt")
            suffix += 1
        path.replace(destination)
        replay = self.out / "replay" / f"rung_{number:02d}.pt"
        for dependency in (replay, replay.with_suffix(replay.suffix + ".json")):
            if dependency.exists():
                dependency.replace(diagnostics / (
                    f"contract_invalidated_rung_{number:02d}_{stamp}_{dependency.name}"))

    def _quarantine_stale_successor_candidates(self, accepted_number: int) -> list[str]:
        """Archive unaccepted candidates whose learned prerequisite just changed.

        A candidate is meaningful only relative to the checkpoint that seeded it.
        If an earlier rung is relearned, resuming a later candidate would silently
        restore behavior learned from the obsolete prerequisite.  Keep those files
        for diagnosis, but remove their canonical names so the later rung must warm
        start from the newly accepted policy.
        """
        diagnostics = self.out / "diagnostics"
        archived: list[str] = []
        stamp = int(time.time())
        for successor in RUNGS:
            if (successor.number <= accepted_number
                    or successor.number in self.state["completed"]):
                continue
            prefix = f"rung_{successor.number:02d}_{successor.slug}"
            for path in sorted(self.out.glob(prefix + "*.pt")):
                diagnostics.mkdir(exist_ok=True)
                destination = diagnostics / (
                    f"stale_dependency_after_{accepted_number:02d}_{path.name}")
                suffix = 1
                while destination.exists():
                    destination = diagnostics / (
                        f"stale_dependency_after_{accepted_number:02d}_"
                        f"{path.stem}_{stamp}_{suffix}{path.suffix}")
                    suffix += 1
                path.replace(destination)
                archived.append(str(destination))
            candidate_dir = self.out / f"{prefix}.candidates"
            if candidate_dir.exists():
                diagnostics.mkdir(exist_ok=True)
                destination = diagnostics / (
                    f"stale_dependency_after_{accepted_number:02d}_{candidate_dir.name}")
                if destination.exists():
                    destination = diagnostics / f"{destination.name}_{stamp}"
                candidate_dir.replace(destination)
                archived.append(str(destination))
            replay = self.out / "replay" / f"rung_{successor.number:02d}.pt"
            for dependency in (replay, replay.with_suffix(replay.suffix + ".json")):
                if dependency.exists():
                    diagnostics.mkdir(exist_ok=True)
                    destination = diagnostics / (
                        f"stale_dependency_after_{accepted_number:02d}_{dependency.name}")
                    dependency.replace(destination)
                    archived.append(str(destination))
            key = str(successor.number)
            for collection in ("attempts", "metrics", "retention_baselines",
                               "checkpoints", "replay_pressures",
                               "retention_opponents"):
                self.state.setdefault(collection, {}).pop(key, None)
        if accepted_number <= 6 and 7 not in self.state["completed"]:
            prior = self.out / "priors" / "rung_07_walk_prior.json"
            if prior.exists():
                diagnostics.mkdir(exist_ok=True)
                destination = diagnostics / (
                    f"stale_dependency_after_{accepted_number:02d}_{prior.name}")
                suffix = 1
                while destination.exists():
                    destination = diagnostics / (
                        f"stale_dependency_after_{accepted_number:02d}_"
                        f"{prior.stem}_{stamp}_{suffix}{prior.suffix}")
                    suffix += 1
                prior.replace(destination)
                archived.append(str(destination))
        if archived:
            self.state.setdefault("dependency_invalidations", []).append({
                "accepted_rung": accepted_number,
                "time": time.time(),
                "archived": archived,
            })
            print(f"DEPENDENCY RESET after rung {accepted_number:02d}: archived "
                  f"{len(archived)} stale successor checkpoint(s)", flush=True)
        return archived

    def _audit_completed_contracts(self) -> None:
        """Reprove accepted checkpoints whenever their gate contract changes.

        Adding a safety metric must not leave an older checkpoint grandfathered
        in merely because it was accepted before that metric existed.  The audit
        also refreshes the exact retention baseline used by later rungs.  If one
        accepted rung no longer passes, that rung and every dependent successor
        are removed from the accepted prefix and will be trained again.
        """
        if self.args.no_gates or self.args.tiny or self.args.dry_run:
            return
        signature = self._gate_signature()
        baselines = self.state.setdefault("retention_baselines", {})
        missing = any(
            gate.metric not in baselines.get(str(rung.number), {})
            for rung in RUNGS if rung.number in self.state["completed"]
            for gate in rung.gates
        )
        if self.state.get("gate_signature") == signature and not missing:
            return
        for rung in RUNGS:
            if rung.number not in self.state["completed"] or not rung.gates:
                continue
            checkpoint = self.state.get("checkpoints", {}).get(str(rung.number))
            if checkpoint and Path(checkpoint).exists() and rung.geometry is not None:
                rc, metrics = self._evaluate_checkpoint(
                    rung, checkpoint, f"contract_audit_task_{rung.number:02d}")
                passed, details = self._gate(rung, metrics)
                passed &= rc == 0
            else:
                metrics, passed = {}, False
                details = ["missing accepted checkpoint"]
            print(f"CONTRACT AUDIT rung {rung.number:02d}: " + "; ".join(details),
                  flush=True)
            if not passed:
                invalid = [number for number in self.state["completed"]
                           if number >= rung.number]
                for number in invalid:
                    self._quarantine_checkpoint(
                        number, self.state.get("checkpoints", {}).get(str(number)))
                self.state["completed"] = [number for number in self.state["completed"]
                                           if number < rung.number]
                for key in ("checkpoints", "metrics", "attempts", "retention_baselines"):
                    values = self.state.setdefault(key, {})
                    for number in invalid:
                        values.pop(str(number), None)
                self.state["failed"] = {
                    "rung": rung.number, "name": rung.name,
                    "reason": "accepted checkpoint failed updated gate contract",
                    "metrics": metrics, "time": time.time(),
                }
                break
            self.state["metrics"][str(rung.number)] = metrics
            baselines[str(rung.number)] = {
                gate.metric: metrics[gate.metric] for gate in rung.gates
            }
        self.state["gate_signature"] = signature
        self._save()

    def _previous_checkpoint(self, rung: Rung) -> str | None:
        """Latest ACCEPTED lineage checkpoint (acceptance order, not number:
        walk-first ordering certifies low-numbered rungs after high ones)."""
        by_number = {other.number: other for other in RUNGS}
        for number in reversed(self.state.get("completed", [])):
            other = by_number.get(number)
            if other is None or other.geometry is None or number == rung.number:
                continue
            if other.family != rung.family:
                continue
            value = self.state["checkpoints"].get(str(number))
            if value and Path(value).exists():
                return value
        return None

    def _run(self, argv: list[str], log: Path, *, dry_run=False) -> int:
        printable = " ".join(argv)
        print(f"COMMAND {printable}", flush=True)
        if dry_run:
            return 0
        with log.open("a") as stream:
            stream.write(f"\nCOMMAND {printable}\n")
            stream.flush()
            return subprocess.run(argv, cwd=ROOT, stdout=stream,
                                  stderr=subprocess.STDOUT).returncode

    def _replay_path(self, rung: Rung) -> Path:
        return self.out / "replay" / f"rung_{rung.number:02d}.pt"

    def _ensure_replay_artifact(self, rung: Rung, checkpoint: str) -> str | None:
        """Build real accepted-task state/action replay with source provenance."""
        if self.args.dry_run or self.args.tiny or rung.geometry is None:
            return None
        output = self._replay_path(rung)
        metadata_path = output.with_suffix(output.suffix + ".json")
        expected_hash = self._checkpoint_sha256(checkpoint)
        opponents = self.state.setdefault("retention_opponents", {}).get(
            str(rung.number), [])
        expected_opponent = opponents[0] if opponents else None
        expected_opponent_hash = (self._checkpoint_sha256(expected_opponent)
                                  if expected_opponent else None)
        if output.exists() and metadata_path.exists():
            metadata = json.loads(metadata_path.read_text())
            if (metadata.get("checkpoint_sha256") == expected_hash
                    and metadata.get("opponent_sha256") == expected_opponent_hash):
                return str(output)
        output.parent.mkdir(parents=True, exist_ok=True)
        log = self.out / "logs" / f"replay_rung_{rung.number:02d}.log"
        argv = [self.args.python, "-u", str(EVALUATOR), "collect-replay",
                *(("--power-model", self.args.power_model)
                  if getattr(self.args, "power_model", "off") != "off" else ()),
                "--geometry", self._geometry(rung), "--rung", str(rung.number),
                "--checkpoint", checkpoint, "--steps", str(self.args.replay_steps),
                "--envs", str(self.args.replay_envs),
                "--seed", str(self.args.retention_seed + 5000 + rung.number),
                "--out", str(output)]
        if opponents:
            argv += ["--opponent", opponents[0]]
        if self.args.device:
            argv += ["--device", self.args.device]
        rc = self._run(argv, log)
        if rc != 0 or not output.exists():
            raise RuntimeError(f"failed to build retention replay for rung {rung.number}")
        self.state.setdefault("replay_pressures", {}).setdefault(str(rung.number), 1.0)
        self._save()
        return str(output)

    def _ensure_completed_replay_artifacts(self) -> None:
        for rung in RUNGS:
            checkpoint = self.state.get("checkpoints", {}).get(str(rung.number))
            if rung.number in self.state["completed"] and checkpoint:
                self._ensure_replay_artifact(rung, checkpoint)

    def _trainer_argv(self, rung: Rung, tag: Path, target_steps: int,
                      *, init_policy: str | None, resume: bool,
                      opponent: str | list[str] | None,
                      anchor_policy: str | None) -> list[str]:
        a = self.args
        if a.tiny:
            envs, horizon, steps = 2, 2, 4
            eval_envs, eval_steps = 2, 2
            diagnostic_eval_seeds, checkpoint_replay_steps = 1, 1
            hidden, epochs, minibatches, preflight = "16,16", 1, 1, "off"
            prediction_horizon, prediction_anchors = 1, 1
        else:
            envs, horizon, steps = a.envs, a.horizon, target_steps
            eval_envs, eval_steps = a.eval_envs, a.eval_steps
            diagnostic_eval_seeds = a.diagnostic_eval_seeds
            checkpoint_replay_steps = a.checkpoint_replay_steps
            hidden, epochs, minibatches, preflight = a.hidden, a.epochs, a.minibatches, "strict"
            prediction_horizon, prediction_anchors = (
                a.prediction_horizon, a.prediction_anchors)
        argv = [a.python, "-u", str(TRAINER), "--geometry", self._geometry(rung),
                "--steps", str(steps), "--envs", str(envs), "--horizon", str(horizon),
                "--episode-length", str(a.episode_length), "--tag", str(tag),
                "--evals", str(a.evals), "--eval-envs", str(eval_envs),
                "--eval-steps", str(eval_steps), "--hidden", hidden,
                "--diagnostic-eval-seeds", str(diagnostic_eval_seeds),
                "--checkpoint-replay-steps", str(checkpoint_replay_steps),
                "--architecture", a.architecture,
                "--prediction-horizon", str(prediction_horizon),
                "--prediction-decoder", a.prediction_decoder,
                "--prediction-anchors", str(prediction_anchors),
                "--prediction-loss-weight", str(a.prediction_loss_weight),
                "--guidance-horizon", str(a.guidance_horizon),
                "--guidance-steps", str(a.guidance_steps),
                "--guidance-interval", str(a.guidance_interval),
                *(("--prediction-lr", str(a.prediction_lr))
                  if a.prediction_lr is not None else ()),
                *(("--power-model", a.power_model)
                  if getattr(a, "power_model", "off") != "off" else ()),
                "--epochs", str(epochs), "--minibatches", str(minibatches),
                "--target-kl", str(a.target_kl),
                "--kl-stop-multiplier", str(a.kl_stop_multiplier),
                "--seed", str(a.seed + rung.number), "--preflight", preflight]
        if rung.geometry in ("ladder_locomotion", "ladder_combat", "universal_control"):
            argv += ["--rung", str(rung.number)]
        if not (a.tiny or a.no_gates):
            trainer_gates = rung.gates
            if rung.number == 26:
                # PPO sees randomized leg commands. Promotion below separately
                # locks every leg and performs an uninterrupted switch test.
                trainer_gates = (
                    Gate("attack_selected_hit", ">=", 0.0001),
                    Gate("attack_kick_speed", ">=", 0.03),
                    Gate("attack_support", ">=", 0.50),
                    Gate("attack_wrong_hit", "<=", 0.10),
                    Gate("fallrate", "<=", 0.001),
                )
            for gate in trainer_gates:
                argv += ["--early-gate",
                         f"{gate.metric},{gate.comparison},{gate.threshold}"]
        if a.device:
            argv += ["--device", a.device]
        if resume:
            argv += ["--resume", str(tag) + ".pt"]
            if a.allow_reward_migration:
                argv += ["--allow-reward-migration"]
        elif init_policy:
            argv += ["--init-policy", init_policy]
        if isinstance(opponent, list):
            for path in opponent:
                argv += ["--opponent-pool", path]
        elif opponent:
            argv += ["--opponent", opponent]
        if anchor_policy and a.distill_weight > 0.0:
            by_number = {item.number: item for item in RUNGS}
            prior = [by_number[number] for number in self.state["completed"]
                     if number != rung.number
                     and by_number[number].family == rung.family]
            indices = [item.number - 1 for item in prior]
            argv += ["--anchor-policy", anchor_policy,
                     "--distill-weight", str(a.distill_weight)]
            if indices:
                argv += ["--anchor-task-indices", ",".join(map(str, indices))]
            for learned in prior:
                replay = self._replay_path(learned)
                if replay.exists():
                    pressure = self.state.setdefault("replay_pressures", {}).get(
                        str(learned.number), 1.0)
                    argv += ["--replay-artifact", f"{replay},{pressure}"]
        prior = self.out / "priors" / "rung_07_walk_prior.json"
        if (rung.number == 7 and self.args.walk_prior_mode != "off"
                and LEGACY_WALK_TEACHER.exists() and prior.exists()):
            argv += ["--transfer-policy", str(LEGACY_WALK_TEACHER),
                     "--transfer-obs-dim", "50"]
            argv += ["--action-prior-json", str(prior)]
        return argv

    def _ensure_walk_prior(self) -> str | None:
        """Search and persist the rung-7 teacher once, then reuse its artifact."""
        prior = self.out / "priors" / "rung_07_walk_prior.json"
        if self.args.tiny or self.args.dry_run:
            return None
        rung6 = self.state.get("checkpoints", {}).get("6")
        if not rung6 or not Path(rung6).exists():
            raise RuntimeError("rung 7 requires an accepted rung-6 checkpoint")
        if not LEGACY_WALK_TEACHER.exists():
            if self.args.walk_prior_mode == "always":
                raise RuntimeError(
                    "--walk-prior-mode=always requires the legacy walk teacher")
            print("WALK PRIOR unavailable: continuing outcome-only from accepted "
                  "rung 6", flush=True)
            return None
        if prior.exists():
            artifact = json.loads(prior.read_text())
            sources = artifact.get("source_checkpoints", {})
            rung6_hash = sources.get("rung6", {}).get("sha256")
            legacy_hash = sources.get("legacy", {}).get("sha256")
            if (rung6_hash == self._checkpoint_sha256(rung6)
                    and legacy_hash == self._checkpoint_sha256(str(LEGACY_WALK_TEACHER))):
                return str(prior)
            diagnostics = self.out / "diagnostics"
            diagnostics.mkdir(exist_ok=True)
            destination = diagnostics / (
                f"stale_walk_prior_{int(time.time())}.json")
            prior.replace(destination)
            print(f"DEPENDENCY RESET: archived gait prior with obsolete sources as "
                  f"{destination.name}", flush=True)
        prior.parent.mkdir(exist_ok=True)
        log = self.out / "logs" / "rung_07_prior_search.log"
        argv = [self.args.python, "-u", str(GAIT_SEARCH),
                "--rung6", rung6, "--legacy", str(LEGACY_WALK_TEACHER),
                "--population", str(self.args.walk_search_population),
                "--repeats", str(self.args.walk_search_repeats),
                "--generations", str(self.args.walk_search_generations),
                "--steps", str(self.args.walk_search_steps),
                "--seed", str(self.args.seed + 7000), "--out", str(prior)]
        if self.args.device:
            argv += ["--device", self.args.device]
        rc = self._run(argv, log)
        if rc != 0 or not prior.exists():
            raise RuntimeError(f"automatic rung-7 prior search failed rc={rc}")
        return str(prior)

    def _retarget_walk_prior(self, observed: dict) -> None:
        """Choose the next teacher from the search Pareto set by gate pressure."""
        path = self.out / "priors" / "rung_07_walk_prior.json"
        if not path.exists():
            return
        artifact = json.loads(path.read_text())
        candidates = (artifact.get("archive") or artifact.get("pareto")
                      or artifact.get("history") or [])
        candidates = [item for item in candidates if item.get("parameters")]
        if not candidates:
            return
        gates = {gate.metric: gate.threshold for gate in RUNGS[6].gates}
        progress_target = gates["xprogress"]
        cat_target = gates["catrate"]
        observed_progress = max(float(observed.get("xprogress", 0.0)), 1.0e-6)
        observed_cat = max(float(observed.get("catrate", 0.0)), 0.0)
        progress_pressure = progress_target / observed_progress
        cat_pressure = observed_cat / max(cat_target, 1.0e-9)
        reaches_speed = [item for item in candidates
                         if float(item.get("x", 0.0)) >= progress_target
                         and float(item.get("fall", 1.0)) <= gates["fallrate"]]
        pool = reaches_speed or candidates
        if cat_pressure >= progress_pressure:
            selected = min(pool, key=lambda item: (
                float(item.get("cat", float("inf"))),
                float(item.get("cat_slip", float("inf"))),
                -float(item.get("safe_x", 0.0))))
            reason = "constraint pressure"
        else:
            selected = max(pool, key=lambda item: (
                float(item.get("safe_x", item.get("x", 0.0))),
                -float(item.get("cat", float("inf")))))
            reason = "progress pressure"
        artifact["best"] = selected
        artifact.setdefault("selection_history", []).append({
            "time": time.time(), "reason": reason,
            "observed_xprogress": observed_progress,
            "observed_catrate": observed_cat,
            "selected_generation": selected.get("generation"),
            "selected_x": selected.get("x"), "selected_cat": selected.get("cat"),
        })
        temp = path.with_suffix(".tmp")
        temp.write_text(json.dumps(artifact, indent=2, sort_keys=True) + "\n")
        temp.replace(path)
        print(f"AUTO-RETARGET rung 07 prior: {reason}; predicted "
              f"x={selected.get('x')} cat={selected.get('cat')}", flush=True)

    def _refine_walk_prior(self, observed: dict) -> None:
        """Rerun constrained search with weights derived from current gate pressure."""
        prior = self.out / "priors" / "rung_07_walk_prior.json"
        rung6 = self.state.get("checkpoints", {}).get("6")
        if (self.args.tiny or self.args.dry_run or not prior.exists()
                or not rung6 or not LEGACY_WALK_TEACHER.exists()):
            return
        gates = {gate.metric: gate.threshold for gate in RUNGS[6].gates}
        progress = max(float(observed.get("xprogress", 0.0)), 1.0e-6)
        catrate = max(float(observed.get("catrate", 0.0)), 0.0)
        progress_pressure = gates["xprogress"] / progress
        cat_pressure = catrate / max(gates["catrate"], 1.0e-9)
        relative_pressure = cat_pressure / max(progress_pressure, 1.0e-6)
        cat_weight = min(12.0, max(1.5, 1.5 * relative_pressure))
        slip_weight = cat_weight / 3.0
        parent = json.loads(prior.read_text())
        index = len(parent.get("selection_history", []))
        candidate = prior.with_name(f"rung_07_walk_refinement_{index:02d}.json")
        log = self.out / "logs" / f"rung_07_prior_refinement_{index:02d}.log"
        argv = [self.args.python, "-u", str(GAIT_SEARCH),
                "--rung6", rung6, "--legacy", str(LEGACY_WALK_TEACHER),
                "--population", str(self.args.walk_search_population),
                "--repeats", str(self.args.walk_search_repeats),
                "--generations", str(max(4, self.args.walk_search_generations // 2)),
                "--steps", str(self.args.walk_search_steps),
                "--seed", str(self.args.seed + 7100 + index),
                "--cat-weight", str(cat_weight),
                "--slip-weight", str(slip_weight),
                "--init-json", str(prior), "--out", str(candidate)]
        if self.args.device:
            argv += ["--device", self.args.device]
        print(f"AUTO-REFINE rung 07 prior: cat_weight={cat_weight:.3f} "
              f"slip_weight={slip_weight:.3f}", flush=True)
        rc = self._run(argv, log)
        if rc == 0 and candidate.exists():
            refined = json.loads(candidate.read_text())
            refined["archive"] = merge_candidate_archives(parent, refined)
            refined["selection_history"] = parent.get("selection_history", []) + [{
                "time": time.time(), "reason": "automatic constrained refinement",
                "cat_weight": cat_weight, "slip_weight": slip_weight,
                "observed_xprogress": progress, "observed_catrate": catrate,
            }]
            temp = prior.with_suffix(".tmp")
            temp.write_text(json.dumps(refined, indent=2, sort_keys=True) + "\n")
            temp.replace(prior)
        else:
            print(f"AUTO-REFINE kept previous prior after rc={rc}", flush=True)

    def _metrics(self, tag: Path) -> dict:
        path = Path(str(tag) + ".stats.json")
        raw = json.loads(path.read_text())
        if not raw.get("evals"):
            raise ValueError(f"no evaluation metrics in {path}")
        return raw["evals"][-1]

    def _geometry(self, rung: Rung) -> str:
        """Resolve a rung's trainer geometry under the observation contract."""
        if (getattr(self.args, "command_observations", False)
                and rung.geometry == "universal_control"):
            return "universal_command"
        return str(rung.geometry)

    @staticmethod
    def _plateau_aborted(tag: Path) -> dict | None:
        """Return the previous attempt's plateau-abort record, if it ended on one."""
        try:
            raw = json.loads(Path(str(tag) + ".stats.json").read_text())
        except (OSError, ValueError, json.JSONDecodeError):
            return None
        record = raw.get("plateau_abort")
        return record if isinstance(record, dict) else None

    @staticmethod
    def _durable_completed_attempts(tag: Path, base_steps: int,
                                    recorded_attempts: int) -> int:
        """Count only attempt boundaries represented by a durable evaluation.

        The ladder records an attempt number after its trainer subprocess exits,
        including an intentional SIGTERM.  If that subprocess was interrupted
        before reaching its target, blindly trusting the counter skips the next
        target on resume.  Evaluation checkpoints are the durable unit: round
        their latest step down to a whole attempt and cap it by the state counter.
        If legacy stats are unavailable, conservatively retry the recorded
        attempt instead of skipping unseen experience.
        """
        stats_path = Path(str(tag) + ".stats.json")
        try:
            raw = json.loads(stats_path.read_text())
            evaluated_steps = [int(row["step"]) for row in raw.get("evals", [])
                               if isinstance(row, dict) and "step" in row]
            if evaluated_steps:
                durable = max(evaluated_steps) // max(1, int(base_steps))
                return min(max(0, int(recorded_attempts)), durable)
        except (OSError, ValueError, TypeError, json.JSONDecodeError):
            # unreadable or legacy stats: same conservative retry as below
            return max(0, int(recorded_attempts) - 1)
        return max(0, int(recorded_attempts) - 1)

    @staticmethod
    def _gate_observation(gate: Gate, metrics: dict) -> tuple[float | None, str, list[float]]:
        """Return the conservative observation represented by an evaluation.

        Trainer evaluations retain the first deterministic seed at the top level
        for backward compatibility, plus all held-out seeds in the diagnostics
        payload.  Promotion must not happen because that first seed happened to
        be easier than the others: lower-bound gates use the minimum seed value,
        while upper-bound gates use the maximum.  A standalone fixed-seed replay
        has no multi-seed payload and therefore uses its direct observation.
        """
        direct = metrics.get(gate.metric)
        seed_metric = (metrics.get("diagnostics", {})
                       .get("multi_seed_evaluation", {})
                       .get("metrics", {})
                       .get(gate.metric, {}))
        raw_values = seed_metric.get("values", []) if isinstance(seed_metric, dict) else []
        values = [float(value) for value in raw_values
                  if isinstance(value, (int, float)) and math.isfinite(float(value))]
        if values:
            observed = min(values) if gate.comparison == ">=" else max(values)
            return observed, f"worst of {len(values)} deterministic seeds", values
        if isinstance(direct, (int, float)) and math.isfinite(float(direct)):
            return float(direct), "direct fixed-seed observation", [float(direct)]
        return None, "missing observation", []

    def _gate(self, rung: Rung, metrics: dict) -> tuple[bool, list[str]]:
        if self.args.no_gates or self.args.tiny:
            return True, ["gates bypassed for smoke/dry validation"]
        details = []
        passed = True
        for gate in rung.gates:
            value, source, values = self._gate_observation(gate, metrics)
            ok = value is not None and gate.passes({gate.metric: value})
            got = "MISSING" if value is None else value
            details.append(
                f"{'PASS' if ok else 'FAIL'} {gate.describe()} "
                f"(got {got}; {source}; values={values})")
            passed &= ok
        return passed, details

    def _evaluate_checkpoint(self, rung: Rung, checkpoint: str,
                             label: str, *, seed: int | None = None,
                             steps: int | None = None,
                             opponent: str | None = None) -> tuple[int, dict]:
        output = self.out / "logs" / f"{label}.json"
        log = self.out / "logs" / f"{label}.log"
        base_seed = seed if seed is not None else self.args.retention_seed + rung.number

        def run_once(suffix: str = "", extra: tuple[str, ...] = ()) -> tuple[int, dict]:
            target = output if not suffix else output.with_name(
                f"{output.stem}_{suffix}{output.suffix}")
            argv = [self.args.python, "-u", str(EVALUATOR), "eval",
                    "--geometry", self._geometry(rung), "--checkpoint", checkpoint,
                    "--episodes", "1", "--steps", str(steps or self.args.retention_steps),
                    "--envs", str(self.args.retention_envs),
                    "--seed", str(base_seed), "--out", str(target),
                    *(("--power-model", self.args.power_model)
                      if getattr(self.args, "power_model", "off") != "off" else ()),
                    *extra]
            if rung.geometry in ("ladder_locomotion", "ladder_combat", "universal_control"):
                argv += ["--rung", str(rung.number)]
            if opponent:
                argv += ["--opponent", opponent]
            if self.args.device:
                argv += ["--device", self.args.device]
            rc = self._run(argv, log)
            metrics = json.loads(target.read_text()) if rc == 0 and target.exists() else {}
            return rc, metrics

        if rung.number != 26:
            return run_once()
        per_leg = {}
        rc = 0
        for leg in ("FL", "FR", "RL", "RR"):
            leg_rc, per_leg[leg] = run_once(leg.lower(), ("--attack-leg", leg))
            rc = rc or leg_rc
        switch_rc, switch = run_once("switch", ("--attack-switch",))
        rc = rc or switch_rc
        combined = {
            "attack_selected_hit_worst_leg": min(
                float(row.get("attack_selected_hit", 0.0)) for row in per_leg.values()),
            "attack_kick_speed_worst_leg": min(
                float(row.get("attack_kick_speed", 0.0)) for row in per_leg.values()),
            "attack_support_worst_leg": min(
                float(row.get("attack_support", 0.0)) for row in per_leg.values()),
            "attack_wrong_hit_worst_leg": max(
                float(row.get("attack_wrong_hit", math.inf)) for row in per_leg.values()),
            "attack_switch_fallrate": float(
                switch.get("attack_switch_fallrate", math.inf)),
            "per_leg": per_leg,
            "switch": switch,
        }
        output.write_text(json.dumps(combined, indent=2, sort_keys=True) + "\n")
        return rc, combined

    def _candidate_paths(self, tag: Path) -> list[str]:
        """Return the best immutable intermediates plus the mutable final policy."""
        archive = Path(str(tag) + ".candidates")
        ranked = []
        for checkpoint in archive.glob("step_*.pt") if archive.exists() else ():
            metadata_path = checkpoint.with_suffix(".json")
            try:
                metadata = json.loads(metadata_path.read_text())
                margin = float(metadata["robust_gates"]["worst_relative_margin"])
            except (FileNotFoundError, KeyError, TypeError, ValueError):
                margin = -math.inf
            ranked.append((margin, str(checkpoint)))
        ranked.sort(reverse=True)
        paths = [path for _, path in ranked[:self.args.candidate_eval_max]]
        final = str(tag) + ".pt"
        if Path(final).exists() and final not in paths:
            paths.append(final)
        return paths

    def _promotion_gate(self, rung: Rung, checkpoint: str,
                        attempt: int, opponent: str | list[str] | None = None
                        ) -> tuple[bool, dict, list[str]]:
        """Use a fresh, rotating full-episode bank unknown to PPO's diagnostics."""
        if self.args.no_gates or self.args.tiny:
            metrics = self._metrics(Path(checkpoint).with_suffix(""))
            return True, metrics, ["promotion bank bypassed"]
        results = []
        details = []
        count = max(1, self.args.promotion_seeds)
        opponents = list(dict.fromkeys(opponent)) if isinstance(opponent, list) else [opponent]
        for opponent_index, opponent_path in enumerate(opponents):
          for index in range(count):
            seed = (self.args.promotion_seed_base + rung.number * 100_003
                    + attempt * 10_007 + opponent_index * 101 + index * 1_009)
            label = (f"promotion_rung_{rung.number:02d}_attempt_{attempt:02d}_"
                     f"opponent_{opponent_index:02d}_seed_{index:02d}_"
                     f"{Path(checkpoint).stem}")
            rc, metrics = self._evaluate_checkpoint(
                rung, checkpoint, label, seed=seed,
                steps=self.args.retention_steps, opponent=opponent_path)
            passed, seed_details = self._gate(rung, metrics)
            details.extend(f"seed {seed}: {item}" for item in seed_details)
            if rc != 0:
                passed = False
            results.append((passed, metrics, seed))
        adverse = {}
        for gate in rung.gates:
            values = [float(metrics[gate.metric]) for _, metrics, _ in results
                      if gate.metric in metrics]
            if values:
                adverse[gate.metric] = (min(values) if gate.comparison == ">="
                                        else max(values))
        passed = len(results) == count * len(opponents) and all(row[0] for row in results)
        adverse["promotion_seed_results"] = [
            {"seed": seed, "pass": ok, "metrics": metrics}
            for ok, metrics, seed in results]
        adverse["promotion_seed_count"] = count
        return passed, adverse, details

    @staticmethod
    def _test_out_margin(rung: Rung, metrics: dict) -> float:
        """Worst dimensionless gate margin in a certification result."""
        margins = []
        for gate in rung.gates:
            value = metrics.get(gate.metric)
            if not isinstance(value, (int, float)) or not math.isfinite(float(value)):
                return -math.inf
            scale = max(abs(float(gate.threshold)), 1.0e-6)
            raw = (float(value) - gate.threshold if gate.comparison == ">="
                   else gate.threshold - float(value))
            margins.append(raw / scale)
        return min(margins) if margins else -math.inf

    def _prepare_test_out_checkpoint(self, rung: Rung, parent: str) -> tuple[str | None, dict]:
        """Clone the prior actor and make the new task exactly inherit it."""
        directory = self.out / "test_out"
        directory.mkdir(exist_ok=True)
        output = directory / f"rung_{rung.number:02d}_{rung.slug}.pt"
        metadata_path = output.with_suffix(output.suffix + ".json")
        parent_hash = self._checkpoint_sha256(parent)
        if output.exists() and metadata_path.exists():
            try:
                metadata = json.loads(metadata_path.read_text())
                if (metadata.get("source_checkpoint_sha256") == parent_hash
                        and metadata.get("source_task_index") == rung.number - 2
                        and metadata.get("target_task_index") == rung.number - 1):
                    return str(output), metadata
            except (OSError, ValueError, TypeError,
                    json.JSONDecodeError) as error:
                print(f"test-out cache unreadable ({error}); regenerating",
                      flush=True)
        log = self.out / "logs" / f"test_out_prepare_rung_{rung.number:02d}.log"
        argv = [self.args.python, "-u", str(EVALUATOR), "inherit-policy",
                "--checkpoint", parent, "--out", str(output),
                "--source-task", str(rung.number - 2),
                "--target-task", str(rung.number - 1)]
        rc = self._run(argv, log, dry_run=self.args.dry_run)
        if rc != 0 or not output.exists() or not metadata_path.exists():
            return None, {"returncode": rc, "source_checkpoint_sha256": parent_hash}
        return str(output), json.loads(metadata_path.read_text())

    def _test_out(self, rung: Rung, *, opponent: str | list[str] | None = None
                  ) -> tuple[bool, dict, str | None]:
        """Certify inherited competence before spending any PPO experience.

        Passing requires a fresh rotating promotion bank, additional margin on
        every current-task gate, and the same full regression replay demanded
        of a trained candidate.  A failed exam simply falls through to PPO.
        """
        if (not self.args.test_out or self.args.no_gates or self.args.tiny
                or self.args.dry_run or not rung.gates
                or self.args.architecture == "mlp"):
            return False, {"test_out": "disabled_or_ineligible"}, None
        tag = self.out / f"rung_{rung.number:02d}_{rung.slug}"
        if (Path(str(tag) + ".pt").exists()
                or str(rung.number) in self.state.get("attempts", {})):
            return False, {"test_out": "existing_training_candidate"}, None
        parent = self._previous_checkpoint(rung)
        if parent is None:
            return False, {"test_out": "no_lineage_predecessor"}, None
        if not getattr(self.args, "command_observations", False):
            # rung-ID conditioning transfers by adjacent-channel inheritance
            previous = next((item for item in RUNGS
                             if item.number == rung.number - 1), None)
            if previous is None or previous.family != rung.family:
                return False, {"test_out": "no_immediate_family_predecessor"}, None
        parent_hash = self._checkpoint_sha256(parent)
        prior = self.state.setdefault("test_out", {}).get(str(rung.number))
        if (isinstance(prior, dict)
                and prior.get("source_checkpoint_sha256") == parent_hash
                and prior.get("gate_signature") == self._gate_signature()):
            if prior.get("decision") == "accept" and Path(
                    str(prior.get("checkpoint", ""))).exists():
                print(f"TEST-OUT reuse accepted rung {rung.number:02d}", flush=True)
                return True, prior["metrics"], prior["checkpoint"]
            if prior.get("decision") == "train":
                print(f"TEST-OUT reuse failed exam for rung {rung.number:02d}; "
                      "continuing to PPO", flush=True)
                return False, prior.get("metrics", {}), None

        if getattr(self.args, "command_observations", False):
            # Command-conditioned policies have no per-rung channels to copy:
            # the exam evaluates the parent directly under the next rung's
            # command distribution, proving generalization on the shared
            # command manifold instead of transfer of a copied embedding.
            candidate, inheritance = parent, {
                "mode": "command_manifold_direct",
                "source_checkpoint_sha256": parent_hash}
        else:
            candidate, inheritance = self._prepare_test_out_checkpoint(rung, parent)
        if candidate is None:
            print(f"TEST-OUT rung {rung.number:02d}: inheritance preparation failed; "
                  "continuing to PPO", flush=True)
            return False, {"test_out": "preparation_failed",
                           "inheritance": inheritance}, None
        promotion_ok, promotion_metrics, promotion_details = self._promotion_gate(
            rung, candidate, 0, opponent)
        margin = self._test_out_margin(rung, promotion_metrics)
        margin_ok = margin >= float(self.args.test_out_margin)
        print("TEST-OUT PROMOTION " + "; ".join(promotion_details), flush=True)
        print(f"TEST-OUT rung {rung.number:02d}: promotion={promotion_ok} "
              f"worst_margin={margin:.4f} required={self.args.test_out_margin:.4f}",
              flush=True)
        fixed_checks = self._current_task_regression_checks(rung, promotion_metrics)
        self._record_regression_task(
            rung, candidate, rung, fixed_checks, promotion_ok and margin_ok,
            "zero_shot_hidden_promotion")
        retention_ok, retention = (self._retention_gate(rung, candidate)
                                   if promotion_ok and margin_ok else (False, []))
        accepted = promotion_ok and margin_ok and retention_ok
        metrics = dict(
            promotion_metrics,
            test_out=True,
            test_out_pass=accepted,
            test_out_worst_relative_margin=margin,
            test_out_required_relative_margin=float(self.args.test_out_margin),
            test_out_inheritance=inheritance,
            retention_pass=retention_ok,
            retention_replays=len(retention),
            selected_candidate=candidate if accepted else None,
        )
        record = {
            "decision": "accept" if accepted else "train",
            "time": time.time(),
            "source_checkpoint": parent,
            "source_checkpoint_sha256": parent_hash,
            "gate_signature": self._gate_signature(),
            "checkpoint": candidate,
            "metrics": metrics,
        }
        self.state.setdefault("test_out", {})[str(rung.number)] = record
        self._save()
        if accepted:
            print(f"TEST-OUT ACCEPT rung {rung.number:02d}: no PPO required", flush=True)
        else:
            print(f"TEST-OUT TRAIN rung {rung.number:02d}: certification did not "
                  "meet every requirement", flush=True)
        return accepted, metrics, candidate if accepted else None

    def _retention_gate(self, rung: Rung, checkpoint: str) -> tuple[bool, list[dict]]:
        """Replay prior same-family tasks and reject catastrophic forgetting."""
        if self.args.no_gates or self.args.tiny or self.args.dry_run:
            return True, []
        by_number = {item.number: item for item in RUNGS}
        previous = [by_number[number] for number in self.state["completed"]
                    if number != rung.number
                    and by_number[number].geometry is not None
                    and by_number[number].family == rung.family]
        if self.args.retention_max > 0:
            previous = previous[-self.args.retention_max:]
        report: list[dict] = []
        all_ok = True
        for learned in previous:
            opponents = self.state.setdefault("retention_opponents", {}).get(
                str(learned.number), []) or [None]
            evaluations = []
            rc = 0
            for opponent_index, opponent in enumerate(opponents):
                label = (f"retention_after_{rung.number:02d}_task_{learned.number:02d}_"
                         f"opponent_{opponent_index:02d}")
                if opponent:
                    opponent_rc, result = self._evaluate_checkpoint(
                        learned, checkpoint, label, opponent=opponent)
                else:
                    opponent_rc, result = self._evaluate_checkpoint(
                        learned, checkpoint, label)
                rc = rc or opponent_rc
                evaluations.append(result)
            current = {}
            for gate in learned.gates:
                values = [float(result[gate.metric]) for result in evaluations
                          if gate.metric in result]
                if values:
                    current[gate.metric] = (min(values) if gate.comparison == ">="
                                            else max(values))
            if len(evaluations) > 1:
                current["retention_opponent_results"] = evaluations
            baseline = self.state.get("retention_baselines", {}).get(
                str(learned.number), self.state["metrics"].get(str(learned.number), {}))
            checks = []
            task_ok = rc == 0
            for gate in learned.gates:
                before = baseline.get(gate.metric)
                after = current.get(gate.metric)
                if before is None or after is None:
                    ok = False
                    allowance = None
                    regression = None
                    acceptance_boundary = None
                    margin = None
                else:
                    before, after = float(before), float(after)
                    allowance = max(abs(before) * self.args.retention_tolerance,
                                    self.args.retention_absolute)
                    regression = ((before - after) if gate.comparison == ">="
                                  else (after - before))
                    acceptance_boundary = (
                        max(gate.threshold, before - allowance)
                        if gate.comparison == ">="
                        else min(gate.threshold, before + allowance))
                    margin = (after - acceptance_boundary
                              if gate.comparison == ">="
                              else acceptance_boundary - after)
                    ok = regression <= allowance and gate.passes(current)
                task_ok &= ok
                checks.append({"metric": gate.metric,
                               "comparison": gate.comparison,
                               "gate_threshold": gate.threshold,
                               "baseline": before,
                               "baseline_kind": "accepted_skill_score",
                               "current": after, "allowance": allowance,
                               "acceptance_boundary": acceptance_boundary,
                               "margin": margin,
                               "regression": regression, "pass": ok})
            all_ok &= task_ok
            pressure_key = str(learned.number)
            old_pressure = float(self.state.setdefault("replay_pressures", {}).get(
                pressure_key, 1.0))
            failed_excesses = []
            for check in checks:
                if not check["pass"] and check["regression"] is not None:
                    failed_excesses.append(max(
                        (float(check["regression"]) - float(check["allowance"]))
                        / max(float(check["allowance"]), 1.0e-6), 0.0))
            if task_ok:
                new_pressure = max(1.0, old_pressure * 0.90)
            else:
                new_pressure = min(10.0, max(old_pressure * 1.5,
                                              1.0 + max(failed_excesses or [0.0])))
            self.state["replay_pressures"][pressure_key] = new_pressure
            report.append({"candidate_rung": rung.number, "replayed_rung": learned.number,
                           "checkpoint": checkpoint, "pass": task_ok, "checks": checks,
                           "replay_pressure_before": old_pressure,
                           "replay_pressure_after": new_pressure})
            self._record_regression_task(
                rung, checkpoint, learned, checks, task_ok, "retention_replay")
            status = "PASS" if task_ok else "FAIL"
            compact = ", ".join(
                f"{check['metric']} {check['baseline']}->{check['current']}"
                for check in checks)
            print(f"RETENTION {status} after {rung.number:02d}, replay "
                  f"{learned.number:02d}: {compact}", flush=True)
        self.state.setdefault("retention_history", []).extend(report)
        self._save()
        return all_ok, report

    def _run_verify(self, rung: Rung) -> tuple[bool, dict, str | None]:
        log = self.out / "logs" / f"rung_{rung.number:02d}_{rung.slug}.log"
        rc = self._run([self.args.python, "-u", str(VERIFY_BODY)], log,
                       dry_run=self.args.dry_run)
        return rc == 0, {"returncode": rc}, None

    def _run_ppo(self, rung: Rung, *, opponent: str | list[str] | None = None,
                 tag_suffix: str = "", warm_override: str | None = None
                 ) -> tuple[bool, dict, str | None]:
        tag = self.out / f"rung_{rung.number:02d}_{rung.slug}{tag_suffix}"
        if rung.number == 7 and self.args.walk_prior_mode == "always":
            self._ensure_walk_prior()
        log = self.out / "logs" / f"{tag.name}.log"
        warm = warm_override or self._previous_checkpoint(rung)
        scratch = warm is None
        base_steps = self.args.scratch_steps if scratch else self.args.steps_per_rung
        base_steps = max(1, int(base_steps * rung.step_scale))
        last_metrics: dict = {}
        candidate_path = Path(str(tag) + ".pt")
        recorded_attempts = int(self.state.get("attempts", {}).get(str(rung.number), 0))
        completed_attempts = (
            self._durable_completed_attempts(tag, base_steps, recorded_attempts)
            if candidate_path.exists() and self.args.resume else 0)
        for offset in range(1, self.args.attempts + 1):
            attempt = completed_attempts + offset
            # First let the accepted physical stepping policy discover forward
            # travel from the outcome itself.  Only a failed attempt authorizes
            # the optional gait teacher as an acquisition fallback; it is never
            # a prerequisite or a permanent definition of walking.
            if (rung.number == 7 and self.args.walk_prior_mode == "fallback"
                    and attempt > 1):
                self._ensure_walk_prior()
            target_steps = base_steps * attempt
            candidate_exists = candidate_path.exists()
            recorded_attempt = str(rung.number) in self.state.get("attempts", {})
            # A process/pod restart may occur between eval checkpoints.  When
            # the ladder itself was launched with --resume, continue that
            # in-rung checkpoint even on this process's first attempt.
            resume = candidate_exists and recorded_attempt and (
                attempt > 1 or self.args.resume)
            argv = self._trainer_argv(rung, tag, target_steps,
                                      init_policy=warm if not resume else None,
                                      resume=resume, opponent=opponent,
                                      anchor_policy=warm)
            plateau = self._plateau_aborted(tag) if resume else None
            if plateau is not None and self.args.plateau_intervention:
                # The previous attempt's gate margin was projected to never
                # cross within its budget.  Buying more identical dynamics
                # would repeat that trajectory: reinject exploration and warm
                # restart the adapted learning rate instead.
                argv += ["--entropy-boost", str(self.args.plateau_entropy_boost),
                         "--learning-rate-restart"]
                print(f"RUNG {rung.number:02d} attempt {attempt}: plateau "
                      f"intervention (entropy x{self.args.plateau_entropy_boost}, "
                      "learning-rate restart) after projected margin stall at "
                      f"step {plateau.get('step')}", flush=True)
            rc = self._run(argv, log, dry_run=self.args.dry_run)
            self.state["attempts"][str(rung.number)] = attempt
            self._save()
            if self.args.dry_run:
                return True, {"dry_run": True}, str(tag) + ".pt"
            if rc != 0:
                print(f"RUNG {rung.number:02d} subprocess failed rc={rc}; see {log}", flush=True)
                return False, {"returncode": rc}, None
            last_metrics = self._metrics(tag)
            passed, details = self._gate(rung, last_metrics)
            print("; ".join(details), flush=True)
            for candidate in self._candidate_paths(tag):
                promotion_ok, promotion_metrics, promotion_details = \
                    self._promotion_gate(rung, candidate, attempt, opponent)
                print("PROMOTION " + "; ".join(promotion_details), flush=True)
                fixed_checks = self._current_task_regression_checks(
                    rung, promotion_metrics)
                self._record_regression_task(
                    rung, candidate, rung, fixed_checks, promotion_ok,
                    "rotating_hidden_promotion")
                if not promotion_ok:
                    continue
                retention_ok, retention = self._retention_gate(rung, candidate)
                last_metrics = dict(
                    promotion_metrics, train_eval=last_metrics,
                    retention_pass=retention_ok, retention_replays=len(retention),
                    selected_candidate=candidate)
                if retention_ok:
                    return True, last_metrics, candidate
                print(f"RUNG {rung.number:02d} learned its new task but regressed an "
                      "earlier skill; continuing the candidate checkpoint", flush=True)
            print(f"RUNG {rung.number:02d} gate failed; continuing its checkpoint "
                  f"toward attempt {attempt + 1}", flush=True)
            if rung.number == 7 and not passed:
                self._retarget_walk_prior(last_metrics)
                self._refine_walk_prior(last_metrics)
        return False, last_metrics, None

    def _run_pfsp(self, rung: Rung) -> tuple[bool, dict, str | None]:
        seed = self._previous_checkpoint(rung)
        if seed is None:
            return False, {"error": "PFSP requires an accepted combat checkpoint"}, None
        hall = [seed]
        final_metrics: dict = {}
        final_ckpt = seed
        rounds = 2 if self.args.tiny else self.args.pfsp_rounds
        for round_index in range(rounds):
            opponent = self._pfsp_opponent(rung, final_ckpt, hall, round_index)
            suffix = f"_round{round_index:02d}"
            passed, final_metrics, checkpoint = self._run_ppo(
                rung, opponent=opponent, tag_suffix=suffix, warm_override=final_ckpt)
            if not passed or checkpoint is None:
                return False, final_metrics, None
            final_ckpt = checkpoint
            hall.append(checkpoint)
        hall_ok, hall_metrics = self._pfsp_hall_gate(rung, final_ckpt, hall[:-1])
        if hall_ok:
            self.state.setdefault("retention_opponents", {})[str(rung.number)] = hall[:-1]
            self._save()
        final_metrics = dict(final_metrics, **hall_metrics,
                             pfsp_hall_size=len(hall))
        return hall_ok, final_metrics, final_ckpt if hall_ok else None

    def _pfsp_hall_gate(self, rung: Rung, candidate: str,
                        hall: list[str]) -> tuple[bool, dict]:
        """Require the final policy to pass against every archived opponent."""
        if self.args.tiny or self.args.no_gates:
            return True, {"pfsp_hall_all_pass": True}
        rows = []
        for index, opponent in enumerate(hall):
            output = self.out / "logs" / f"pfsp_final_opponent_{index:02d}.json"
            log = self.out / "logs" / f"pfsp_final_opponent_{index:02d}.log"
            argv = [self.args.python, "-u", str(EVALUATOR), "eval",
                    "--geometry", self._geometry(rung), "--rung", str(rung.number),
                    "--checkpoint", candidate, "--opponent", opponent,
                    "--episodes", "1", "--steps", str(self.args.retention_steps),
                    "--envs", str(self.args.retention_envs),
                    "--seed", str(self.args.promotion_seed_base + 900_000 + index),
                    "--out", str(output)]
            if self.args.device:
                argv += ["--device", self.args.device]
            rc = self._run(argv, log)
            metrics = json.loads(output.read_text()) if rc == 0 and output.exists() else {}
            passed, _ = self._gate(rung, metrics)
            rows.append({"opponent": opponent, "pass": rc == 0 and passed,
                         "metrics": metrics})
        aggregate = {"pfsp_hall_results": rows,
                     "pfsp_hall_all_pass": all(row["pass"] for row in rows)}
        for gate in rung.gates:
            values = [float(row["metrics"][gate.metric]) for row in rows
                      if gate.metric in row["metrics"]]
            if values:
                aggregate[gate.metric] = (min(values) if gate.comparison == ">="
                                          else max(values))
        return bool(aggregate["pfsp_hall_all_pass"]), aggregate

    def _pfsp_opponent(self, rung: Rung, candidate: str, hall: list[str],
                       round_index: int) -> list[str]:
        """Prioritized fictitious self-play over measured archive matchups.

        Opponents that the current candidate rarely beats receive the largest
        weight. Sampling is seeded, so a resumed run makes the same choice.
        """
        if self.args.tiny:
            return [hall[round_index % len(hall)]]
        rows = []
        for index, opponent in enumerate(hall):
            label = f"pfsp_round_{round_index:02d}_opponent_{index:02d}"
            output = self.out / "logs" / f"{label}.json"
            log = self.out / "logs" / f"{label}.log"
            argv = [self.args.python, "-u", str(EVALUATOR), "eval",
                    "--geometry", self._geometry(rung), "--rung", str(rung.number),
                    "--checkpoint", candidate, "--opponent", opponent,
                    "--episodes", "1", "--steps", str(self.args.retention_steps),
                    "--envs", str(self.args.retention_envs),
                    "--seed", str(self.args.retention_seed + 1000 + round_index),
                    "--out", str(output)]
            if self.args.device:
                argv += ["--device", self.args.device]
            rc = self._run(argv, log)
            result = json.loads(output.read_text()) if rc == 0 and output.exists() else {}
            margin = float(result.get("dealt", 0.0)) - float(result.get("taken", 0.0))
            win_probability = 1.0 / (1.0 + math.exp(-5.0 * margin))
            weight = max(1.0 - win_probability, 0.05) ** 2
            rows.append({"opponent": opponent, "margin": margin,
                         "win_probability": win_probability, "weight": weight})
        weights = [row["weight"] for row in rows]
        choices = random.Random(self.args.seed + round_index).choices(
            range(len(hall)), weights=weights, k=self.args.pfsp_pool_size)
        event = {"round": round_index, "candidate": candidate,
                 "selected_pool": [hall[index] for index in choices], "matchups": rows}
        self.state.setdefault("pfsp_history", []).append(event)
        self._save()
        print(f"PFSP round {round_index}: mixed "
              f"{[Path(hall[index]).name for index in choices]}; "
              f"weights={[round(value, 3) for value in weights]}", flush=True)
        return [hall[index] for index in choices]

    def _run_search(self, rung: Rung) -> tuple[bool, dict, str | None]:
        checkpoint = self._previous_checkpoint(rung)
        if checkpoint is None:
            return False, {"error": "co-design search requires rung 30 checkpoint"}, None
        candidates = 4 if self.args.tiny else self.args.search_candidates
        generations = 1 if self.args.tiny else self.args.search_generations
        steps = 4 if self.args.tiny else self.args.search_steps
        envs = 2 if self.args.tiny else min(self.args.envs, self.args.search_envs)
        rounds = 1 if self.args.tiny else self.args.codesign_rounds
        metrics = {}
        for round_index in range(rounds):
            output = self.out / f"rung_{rung.number:02d}_{rung.slug}_round{round_index:02d}.json"
            log = self.out / "logs" / f"rung_{rung.number:02d}_{rung.slug}_round{round_index:02d}.log"
            argv = [self.args.python, "-u", str(SEARCH), "design",
                    "--candidates", str(candidates), "--generations", str(generations),
                    "--steps", str(steps), "--envs", str(envs),
                    "--repeats", str(self.args.search_repeats),
                    "--heldout-repeats", str(self.args.search_heldout_repeats),
                    "--seed", str(self.args.seed + rung.number + round_index * 100_003),
                    "--checkpoint", checkpoint, "--out", str(output)]
            if self.args.device:
                argv += ["--device", self.args.device]
            rc = self._run(argv, log, dry_run=self.args.dry_run)
            if self.args.dry_run:
                return True, {"dry_run": True}, checkpoint
            metrics = (json.loads(output.read_text())
                       if rc == 0 and output.exists() else {"returncode": rc})
            if rc != 0 or round_index == rounds - 1:
                break
            # Alternate design search with real PPO adaptation of the same
            # universal policy on the discovered morphology plus the original
            # corner/center coverage bank.
            design_bank = [
                [0.0, 0.0, 0.0], [1.0, 0.0, 1.0], [0.0, 1.0, 1.0],
                [1.0, 1.0, 0.0], [0.5, 0.5, 0.5], metrics["parameters"],
            ]
            bank_path = self.out / f"codesign_bank_round{round_index:02d}.json"
            bank_path.write_text(json.dumps(design_bank, indent=2) + "\n")
            adapt_tag = self.out / f"rung_31_codesign_adapt_round{round_index:02d}"
            adapt_log = self.out / "logs" / f"{adapt_tag.name}.log"
            adapt_rung = RUNGS[29]
            adapt_argv = self._trainer_argv(
                adapt_rung, adapt_tag, self.args.codesign_retrain_steps,
                init_policy=checkpoint, resume=False, opponent=None,
                anchor_policy=checkpoint)
            adapt_argv += ["--design-bank-json", str(bank_path)]
            adapt_rc = self._run(adapt_argv, adapt_log)
            if adapt_rc != 0:
                return False, {"error": "co-design policy adaptation failed",
                               "returncode": adapt_rc}, None
            checkpoint = str(adapt_tag) + ".pt"
        passed, details = self._gate(rung, metrics)
        print("; ".join(details), flush=True)
        metrics["codesign_rounds"] = rounds
        metrics["adapted_checkpoint"] = checkpoint
        return rc == 0 and passed, metrics, checkpoint

    def execute(self) -> int:
        self._audit_completed_contracts()
        self._ensure_completed_replay_artifacts()
        for rung in ordered_rungs(self.args.first, self.args.last,
                                  getattr(self.args, 'walk_first', False)):
            if rung.number in self.state["completed"]:
                print(f"SKIP rung {rung.number:02d} {rung.name}: already accepted", flush=True)
                continue
            # A resumed process is actively retrying this rung; do not leave a
            # stale terminal failure in the live status while it runs.
            if self.state.get("failed", {}).get("rung") == rung.number:
                self.state.pop("failed", None)
                self._save()
            print(f"\n=== RUNG {rung.number:02d}/31 · {rung.name} ===\n{rung.demand}", flush=True)
            if rung.kind == "verify":
                passed, metrics, checkpoint = self._run_verify(rung)
            elif rung.kind == "ppo":
                opponent = self._previous_checkpoint(rung) if rung.number == 28 else None
                passed, metrics, checkpoint = self._test_out(rung, opponent=opponent)
                if not passed:
                    passed, metrics, checkpoint = self._run_ppo(rung, opponent=opponent)
                if passed and rung.number == 28 and opponent:
                    self.state.setdefault("retention_opponents", {})["28"] = [opponent]
            elif rung.kind == "pfsp":
                passed, metrics, checkpoint = self._run_pfsp(rung)
            else:
                passed, metrics, checkpoint = self._run_search(rung)
            self.state["metrics"][str(rung.number)] = metrics
            if passed:
                self._quarantine_stale_successor_candidates(rung.number)
                self.state["completed"].append(rung.number)
                if checkpoint:
                    self.state["checkpoints"][str(rung.number)] = checkpoint
                self.state.setdefault("retention_baselines", {})[str(rung.number)] = {
                    gate.metric: metrics.get(gate.metric) for gate in rung.gates
                    if gate.metric in metrics
                }
                self._save()
                if checkpoint:
                    self._ensure_replay_artifact(rung, checkpoint)
                print(f"ACCEPT rung {rung.number:02d}: {rung.name}", flush=True)
            else:
                self.state["failed"] = {"rung": rung.number, "name": rung.name,
                                        "metrics": metrics, "time": time.time()}
                self._save()
                print(f"STOP rung {rung.number:02d}: gate or subprocess failed; "
                      "last accepted checkpoint preserved", flush=True)
                return 2
        self.state.pop("failed", None)
        self.state["finished"] = time.time()
        self._save()
        print(f"\nLADDER COMPLETE: {len(self.state['completed'])}/31 accepted; "
              f"state={self.state_path}", flush=True)
        return 0


def print_manifest() -> None:
    print("#  Tier  Task                     Family       Gate")
    for rung in RUNGS:
        gates = ", ".join(gate.describe() for gate in rung.gates) or "exit code 0"
        print(f"{rung.number:02d} {rung.tier:^5} {rung.name:<24} {rung.family:<12} {gates}")


def make_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    sub.add_parser("list", help="print all 31 rungs and their gates")
    run = sub.add_parser("run", help="execute a sequential, gated training run")
    run.add_argument("--out", default="sim/build/gpu/out/training_ladder")
    run.add_argument("--from", dest="first", type=int, default=1)
    run.add_argument("--to", dest="last", type=int, default=31)
    run.add_argument("--resume", action="store_true")
    run.add_argument("--allow-reward-migration", action="store_true",
                     help=("permit an explicit reward-only checkpoint migration; "
                           "actor/runtime survive while critic/optimizer reset"))
    run.add_argument("--device", default=None)
    run.add_argument("--python", default=sys.executable)
    run.add_argument("--envs", type=int, default=1024)
    run.add_argument("--horizon", type=int, default=64)
    run.add_argument("--episode-length", type=int, default=800)
    run.add_argument("--steps-per-rung", type=int, default=1_000_000)
    run.add_argument("--scratch-steps", type=int, default=8_000_000)
    run.add_argument("--attempts", type=int, default=2)
    run.add_argument("--evals", type=int, default=4)
    run.add_argument("--eval-envs", type=int, default=64)
    run.add_argument("--eval-steps", type=int, default=400)
    run.add_argument("--diagnostic-eval-seeds", type=int, default=3,
                     help="held-out deterministic seeds summarized at every evaluation")
    run.add_argument("--promotion-seeds", type=int, default=5,
                     help="fresh full-episode seeds every candidate must all pass")
    run.add_argument("--promotion-seed-base", type=int, default=2026080100,
                     help="base for rotating promotion banks, separate from PPO diagnostics")
    run.add_argument("--candidate-eval-max", type=int, default=5,
                     help="highest-margin immutable intermediate policies to promote")
    run.add_argument("--test-out", action=argparse.BooleanOptionalAction, default=True,
                     help=("certify inherited behavior on fresh full episodes before PPO; "
                           "use --no-test-out to require training every rung"))
    run.add_argument("--test-out-margin", type=float, default=0.10,
                     help="minimum dimensionless margin on every zero-shot promotion gate")
    run.add_argument("--checkpoint-replay-steps", type=int, default=32,
                     help="fixed rollout length compared immediately across save/reload")
    run.add_argument("--epochs", type=int, default=4)
    run.add_argument("--minibatches", type=int, default=8)
    run.add_argument("--target-kl", type=float, default=0.02,
                     help="whole-rollout PPO KL target for adaptive epoch/lr control")
    run.add_argument("--kl-stop-multiplier", type=float, default=1.5,
                     help="stop remaining PPO epochs above this target-KL multiple")
    # 512,512,512 constructs the exact network the accepted "512,256,128"
    # runs actually trained (FiLM actors are constant-width at hidden[0]);
    # the honest spelling avoids the trainer's taper warning.
    run.add_argument("--hidden", default="512,512,512")
    run.add_argument("--architecture", choices=(
        "mlp", "task_film", "task_film_gru", "predictive_token_gru"),
                     default="task_film")
    run.add_argument("--prediction-horizon", type=int, default=32)
    run.add_argument("--prediction-decoder", choices=("recurrent", "transformer"),
                     default="recurrent")
    run.add_argument("--prediction-anchors", type=int, default=4)
    run.add_argument("--prediction-lr", type=float, default=None,
                     help="constant decoder Adam learning rate forwarded to the "
                          "trainer (default: the trainer's --lr ceiling)")
    run.add_argument("--power-model", choices=("off", "shared_bus"),
                     default="off",
                     help="train and evaluate every locomotion-path rung under "
                          "the shared-bus electrical budget (+shared_bus_v2 "
                          "action semantics); the fused combat layer is not "
                          "yet covered and stays on v1")
    run.add_argument("--walk-first",
                     action=argparse.BooleanOptionalAction, default=False,
                     help="acquire locomotion from scratch first (rung 8 with a standing-hold stripe), then certify stand/pose/step rungs as commanded special cases; requires --command-observations")
    run.add_argument("--command-observations",
                     action=argparse.BooleanOptionalAction, default=False,
                     help="train every universal rung on the commands-only v2 "
                          "observation contract (universal_command geometry): "
                          "rung identity is invisible to the policy and all "
                          "task semantics arrive as explicit commands; see "
                          "notes/universal-command-contract.md")
    run.add_argument("--plateau-intervention",
                     action=argparse.BooleanOptionalAction, default=True,
                     help="after a plateau-aborted attempt, retry with entropy "
                          "reinjection and a learning-rate warm restart instead "
                          "of identical dynamics")
    run.add_argument("--plateau-entropy-boost", type=float, default=1.5,
                     help="entropy-coefficient multiplier for plateau retries "
                          "(capped at the trainer's from-scratch start value)")
    run.add_argument("--prediction-loss-weight", type=float, default=0.25)
    run.add_argument("--guidance-horizon", type=int, default=16)
    run.add_argument("--guidance-steps", type=int, default=2)
    run.add_argument("--guidance-interval", type=int, default=4)
    run.add_argument("--distill-weight", type=float, default=0.05,
                     help="old-policy behavior replay weight between sequential rungs")
    run.add_argument("--seed", type=int, default=20260712)
    run.add_argument("--pfsp-rounds", type=int, default=4)
    run.add_argument("--pfsp-pool-size", type=int, default=8,
                     help="PFSP-weighted frozen opponents mixed within each rollout")
    run.add_argument("--search-candidates", type=int, default=16)
    run.add_argument("--search-generations", type=int, default=4)
    run.add_argument("--search-steps", type=int, default=160)
    run.add_argument("--search-envs", type=int, default=64)
    run.add_argument("--search-repeats", type=int, default=3)
    run.add_argument("--search-heldout-repeats", type=int, default=5)
    run.add_argument("--codesign-rounds", type=int, default=2,
                     help="alternate held-out design search and policy adaptation")
    run.add_argument("--codesign-retrain-steps", type=int, default=500_000)
    run.add_argument("--walk-search-population", type=int, default=64)
    run.add_argument("--walk-search-repeats", type=int, default=4)
    run.add_argument("--walk-search-generations", type=int, default=12)
    run.add_argument("--walk-search-steps", type=int, default=400)
    run.add_argument(
        "--walk-prior-mode", choices=("off", "fallback", "always"),
        default="fallback",
        help="try outcome-only rung 7 first, disable its gait prior, or require it")
    run.add_argument("--no-gates", action="store_true")
    run.add_argument("--retention-tolerance", type=float, default=0.20,
                     help="maximum relative regression from a skill's learned baseline")
    run.add_argument("--retention-absolute", type=float, default=0.02,
                     help="minimum absolute regression allowance for near-zero metrics")
    run.add_argument("--retention-steps", type=int, default=800,
                     help="full fixed-seed episode length for every retention replay")
    run.add_argument("--retention-envs", type=int, default=32)
    run.add_argument("--retention-seed", type=int, default=2026071200)
    run.add_argument("--retention-max", type=int, default=0,
                     help="replay only the latest N prior skills (0 means every prior skill)")
    run.add_argument("--replay-steps", type=int, default=200,
                     help="accepted-policy physics steps stored per prior skill")
    run.add_argument("--replay-envs", type=int, default=16,
                     help="parallel worlds stored in each real retention replay")
    run.add_argument("--tiny", action="store_true")
    run.add_argument("--dry-run", action="store_true")
    return parser


def main(argv=None) -> int:
    validate_manifest()
    args = make_parser().parse_args(argv)
    if args.command == "list":
        print_manifest()
        return 0
    if not 1 <= args.first <= args.last <= 31:
        raise SystemExit("--from/--to must satisfy 1 <= from <= to <= 31")
    if args.test_out_margin < 0.0:
        raise SystemExit("--test-out-margin must be non-negative")
    if getattr(args, "walk_first", False) and not getattr(
            args, "command_observations", False):
        raise SystemExit("--walk-first requires --command-observations: the "
                         "reordered acquisition relies on rung-invisible "
                         "command conditioning")
    return LadderRunner(args).execute()


if __name__ == "__main__":
    raise SystemExit(main())
