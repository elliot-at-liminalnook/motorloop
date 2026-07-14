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


@dataclass(frozen=True)
class Gate:
    metric: str
    comparison: str
    threshold: float

    def passes(self, metrics: dict) -> bool:
        if self.metric not in metrics:
            return False
        fn = {">=": operator.ge, "<=": operator.le}[self.comparison]
        return bool(fn(float(metrics[self.metric]), self.threshold))

    def describe(self) -> str:
        return f"{self.metric} {self.comparison} {self.threshold:g}"


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
         "ppo", "locomotion", "ladder_locomotion",
         (g("up", ">=", 0.85), g("speed", "<=", 0.15),
          g("catrate", "<=", 0.001), g("fallrate", "<=", 0.0001))),
    Rung(3, "A", "balance_push", "Balance under push", "reject alternating lateral impulses",
         "ppo", "locomotion", "ladder_locomotion",
         (g("up", ">=", 0.72), g("catrate", "<=", 0.001),
          g("fallrate", "<=", 0.0001))),
    Rung(4, "A", "pose_vector", "Track a pose vector", "track randomized 12-joint targets",
         "ppo", "locomotion", "ladder_locomotion",
         (g("ladder_pose_score", ">=", 0.45), g("catrate", "<=", 0.001),
          g("fallrate", "<=", 0.0001))),
    Rung(5, "A", "height_control", "Height control", "track randomized crouch/stand height",
         "ppo", "locomotion", "ladder_locomotion",
         (g("ladder_height_score", ">=", 0.45), g("catrate", "<=", 0.001),
          g("fallrate", "<=", 0.0001))),
    Rung(6, "B", "step_in_place", "Step in place", "lift and replace feet without translating",
         "ppo", "locomotion", "ladder_locomotion",
         (g("duty", "<=", 0.95), g("foot_cycle_duty", "<=", 0.95),
          g("speed", "<=", 0.20),
          g("ladder_step_clock", ">=", 0.70),
          g("ladder_swing_clearance", ">=", 0.30),
          g("up", ">=", 0.85), g("catrate", "<=", 0.001),
          g("fallrate", "<=", 0.0001))),
    Rung(7, "B", "walk_forward", "Walk forward", "first fixed-speed traveling gait",
         "ppo", "locomotion", "ladder_locomotion",
         (g("xprogress", ">=", 0.10), g("lateral", "<=", 0.15),
          g("duty", "<=", 0.95), g("up", ">=", 0.85),
          g("ladder_step_clock", ">=", 0.65),
          g("ladder_swing_clearance", ">=", 0.25),
          g("catrate", "<=", 0.001), g("fallrate", "<=", 0.0001))),
    Rung(8, "B", "velocity_tracking", "Velocity tracking", "track a forward speed range",
         "ppo", "locomotion", "ladder_locomotion",
         (g("track", ">=", 0.40), g("xprogress", ">=", 0.06),
          g("duty", "<=", 0.95), g("catrate", "<=", 0.001),
          g("fallrate", "<=", 0.0001))),
    Rung(9, "B", "turn_in_place", "Turn in place", "track yaw rate at zero translation",
         "ppo", "locomotion", "ladder_locomotion",
         (g("ladder_yaw_score", ">=", 0.35), g("speed", "<=", 0.20),
          g("catrate", "<=", 0.001), g("fallrate", "<=", 0.0001))),
    Rung(10, "B", "omnidirectional", "Omnidirectional", "track planar and yaw commands together",
         "ppo", "locomotion", "ladder_locomotion",
         (g("track", ">=", 0.30), g("xprogress", ">=", 0.05),
          g("duty", "<=", 0.95), g("catrate", "<=", 0.001),
          g("fallrate", "<=", 0.0001))),
    Rung(11, "B", "heading_hold", "Heading hold", "translate while regulating heading",
         "ppo", "locomotion", "ladder_locomotion",
         (g("ladder_heading_score", ">=", 0.40), g("xprogress", ">=", 0.05),
          g("duty", "<=", 0.95), g("catrate", "<=", 0.001),
          g("fallrate", "<=", 0.0001))),
    Rung(12, "B", "stop_command", "Stop on command", "switch between travel and a true hold",
         "ppo", "locomotion", "ladder_locomotion",
         (g("ladder_stop_score", ">=", 0.30),
          g("ladder_move_progress", ">=", 0.035),
          g("catrate", "<=", 0.001), g("fallrate", "<=", 0.0001))),
    Rung(13, "C", "servo_true", "Servo-true motors", "restore the measured torque-speed droop",
         "ppo", "locomotion", "ladder_locomotion",
         (g("xprogress", ">=", 0.08), g("duty", "<=", 0.95),
          g("catrate", "<=", 0.001), g("fallrate", "<=", 0.0001))),
    Rung(14, "C", "stumble", "Stumble recovery", "cross a physical trip bar",
         "ppo", "locomotion", "ladder_locomotion",
         (g("xprogress", ">=", 0.08), g("up", ">=", 0.72),
          g("duty", "<=", 0.95), g("catrate", "<=", 0.001),
          g("fallrate", "<=", 0.0001)), 1.25),
    Rung(15, "C", "push_recovery", "Push recovery", "resume gait after stronger mid-stride pushes",
         "ppo", "locomotion", "ladder_locomotion",
         (g("xprogress", ">=", 0.07), g("up", ">=", 0.72),
          g("duty", "<=", 0.95), g("catrate", "<=", 0.001),
          g("fallrate", "<=", 0.0001)), 1.25),
    Rung(16, "C", "rough_terrain", "Rough terrain", "cross raised tiles with varied friction",
         "ppo", "locomotion", "ladder_locomotion",
         (g("xprogress", ">=", 0.06), g("up", ">=", 0.68),
          g("duty", "<=", 0.95), g("catrate", "<=", 0.001),
          g("fallrate", "<=", 0.0001)), 1.5),
    Rung(17, "C", "slope", "Slope walking", "travel on a nine-degree physical incline",
         "ppo", "locomotion", "ladder_locomotion",
         (g("xprogress", ">=", 0.04), g("up", ">=", 0.68),
          g("duty", "<=", 0.95), g("catrate", "<=", 0.001),
          g("fallrate", "<=", 0.0001)), 1.25),
    Rung(18, "C", "payload", "Payload carry", "walk with 30 percent extra torso mass",
         "ppo", "locomotion", "ladder_locomotion",
         (g("xprogress", ">=", 0.06), g("up", ">=", 0.68),
          g("duty", "<=", 0.95), g("catrate", "<=", 0.001),
          g("fallrate", "<=", 0.0001)), 1.25),
    Rung(19, "D", "return_origin", "Return to origin", "recover from randomized displaced starts",
         "ppo", "locomotion", "ladder_locomotion",
         (g("ladder_goal_hit", ">=", 0.02), g("xprogress", ">=", 0.04),
          g("catrate", "<=", 0.001), g("fallrate", "<=", 0.0001)), 1.5),
    Rung(20, "D", "waypoint", "Go to waypoint", "reach one commanded planar goal",
         "ppo", "locomotion", "ladder_locomotion",
         (g("ladder_goal_hit", ">=", 0.02), g("xprogress", ">=", 0.04),
          g("catrate", "<=", 0.001), g("fallrate", "<=", 0.0001)), 1.5),
    Rung(21, "D", "route", "Route following", "hit an ordered four-waypoint route",
         "ppo", "locomotion", "ladder_locomotion",
         (g("ladder_goal_hit", ">=", 0.015), g("xprogress", ">=", 0.04),
          g("catrate", "<=", 0.001), g("fallrate", "<=", 0.0001)), 1.75),
    Rung(22, "D", "route_correction", "Route correction", "reacquire route after a forced detour",
         "ppo", "locomotion", "ladder_locomotion",
         (g("ladder_goal_hit", ">=", 0.010), g("xprogress", ">=", 0.035),
          g("catrate", "<=", 0.001), g("fallrate", "<=", 0.0001)), 1.75),
    Rung(23, "D", "degraded_lidar", "Degraded-lidar nav",
         "avoid physical obstacles with noisy, dropped, one-step-late 144-ray sensing",
         "ppo", "locomotion", "ladder_locomotion",
         (g("ladder_goal_hit", ">=", 0.010), g("xprogress", ">=", 0.03),
          g("ladder_obstacle_clearance", ">=", 0.02),
          g("catrate", "<=", 0.001), g("fallrate", "<=", 0.0001)), 2.0),
    Rung(24, "E", "approach_dummy", "Approach a dummy", "close on a passive target",
         "ppo", "combat", "ladder_combat",
         (g("ladder_approach", ">=", 0.0002),
          g("ladder_target_distance", "<=", 0.45),
          g("fallrate", "<=", 0.001)), 1.5),
    Rung(25, "E", "strike_target", "Strike a target", "close and fire the pneumatic striker",
         "ppo", "combat", "ladder_combat",
         (g("ladder_rod_hit", ">=", 0.0001),
          g("ladder_target_distance", "<=", 0.50),
          g("fallrate", "<=", 0.001)), 1.75),
    Rung(26, "E", "commanded_leg_kick", "Commanded leg kick",
         "switch FL/FR/RL/RR at runtime; selected foot attacks while three legs support",
         "ppo", "combat", "ladder_combat",
         (g("attack_selected_hit", ">=", 0.0001),
          g("attack_kick_speed", ">=", 0.03),
          g("attack_support", ">=", 0.50),
          g("attack_wrong_hit", "<=", 0.10),
          g("fallrate", "<=", 0.001)), 2.0),
    Rung(27, "E", "pursue_evader", "Pursue an evader", "close on a scripted moving target",
         "ppo", "combat", "ladder_combat",
         (g("ladder_approach", ">=", 0.0001),
          g("ladder_target_distance", "<=", 0.50),
          g("fallrate", "<=", 0.001)), 1.75),
    Rung(28, "E", "frozen_self", "Beat a frozen self", "fight one frozen prior policy",
         "ppo", "combat", "ladder_combat",
         (g("ladder_combat_margin", ">=", 0.00005),
          g("ladder_rod_hit", ">=", 0.0001),
          g("fallrate", "<=", 0.002)), 2.0),
    Rung(29, "E", "pfsp", "PFSP league", "train against a growing hall of frozen snapshots",
         "pfsp", "combat", "ladder_combat",
         (g("ladder_combat_margin", ">=", 0.00005),
          g("ladder_rod_hit", ">=", 0.0001),
          g("fallrate", "<=", 0.002)), 3.0),
    Rung(30, "F", "cross_morphology", "Cross-morphology",
         "one design-conditioned policy across actual compiled bodies",
         "ppo", "codesign", "universal",
         (g("track", ">=", 0.40), g("xprogress", ">=", 0.05),
          g("duty", "<=", 0.95), g("catrate", "<=", 0.001),
          g("fallrate", "<=", 0.0001)), 2.0),
    Rung(31, "F", "codesign_loop", "Co-design loop", "CEM-search body parameters using policy return",
         "search", "codesign", None, (g("improvement", ">=", 0.001),), 1.0),
)


def validate_manifest() -> None:
    numbers = [r.number for r in RUNGS]
    if numbers != list(range(1, 32)):
        raise ValueError(f"training ladder must contain exactly contiguous rungs 1..31: {numbers}")
    slugs = [r.slug for r in RUNGS]
    if len(slugs) != len(set(slugs)):
        raise ValueError("training ladder slugs must be unique")


def selected_rungs(first: int, last: int) -> Iterable[Rung]:
    return (rung for rung in RUNGS if first <= rung.number <= last)


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
        self.regression_matrix = self._load_regression_matrix()
        self._failed_on_entry = self.state.get("failed", {}).get("rung")

    @staticmethod
    def _fresh_state() -> dict:
        return {"version": 2, "completed": [], "checkpoints": {}, "metrics": {},
                "attempts": {}, "retention_baselines": {}, "retention_history": [],
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
            key = str(successor.number)
            for collection in ("attempts", "metrics", "retention_baselines",
                               "checkpoints"):
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
        candidates = [other for other in RUNGS if other.number < rung.number
                      and other.family == rung.family]
        for previous in reversed(candidates):
            value = self.state["checkpoints"].get(str(previous.number))
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

    def _trainer_argv(self, rung: Rung, tag: Path, target_steps: int,
                      *, init_policy: str | None, resume: bool,
                      opponent: str | None, anchor_policy: str | None) -> list[str]:
        a = self.args
        if a.tiny:
            envs, horizon, steps = 2, 2, 4
            eval_envs, eval_steps = 2, 2
            diagnostic_eval_seeds, checkpoint_replay_steps = 1, 1
            hidden, epochs, minibatches, preflight = "16,16", 1, 1, "off"
        else:
            envs, horizon, steps = a.envs, a.horizon, target_steps
            eval_envs, eval_steps = a.eval_envs, a.eval_steps
            diagnostic_eval_seeds = a.diagnostic_eval_seeds
            checkpoint_replay_steps = a.checkpoint_replay_steps
            hidden, epochs, minibatches, preflight = a.hidden, a.epochs, a.minibatches, "strict"
        argv = [a.python, "-u", str(TRAINER), "--geometry", str(rung.geometry),
                "--steps", str(steps), "--envs", str(envs), "--horizon", str(horizon),
                "--episode-length", str(a.episode_length), "--tag", str(tag),
                "--evals", str(a.evals), "--eval-envs", str(eval_envs),
                "--eval-steps", str(eval_steps), "--hidden", hidden,
                "--diagnostic-eval-seeds", str(diagnostic_eval_seeds),
                "--checkpoint-replay-steps", str(checkpoint_replay_steps),
                "--architecture", a.architecture,
                "--epochs", str(epochs), "--minibatches", str(minibatches),
                "--target-kl", str(a.target_kl),
                "--kl-stop-multiplier", str(a.kl_stop_multiplier),
                "--seed", str(a.seed + rung.number), "--preflight", preflight]
        if rung.geometry in ("ladder_locomotion", "ladder_combat"):
            argv += ["--rung", str(rung.number)]
        if not (a.tiny or a.no_gates):
            for gate in rung.gates:
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
        if opponent:
            argv += ["--opponent", opponent]
        if anchor_policy and a.distill_weight > 0.0:
            prior = [item for item in RUNGS if item.number in self.state["completed"]
                     and item.number < rung.number and item.family == rung.family]
            if rung.family == "locomotion":
                indices = [item.number - 1 for item in prior]
            elif rung.family == "combat":
                indices = [item.number - 24 for item in prior]
            else:
                indices = []
            argv += ["--anchor-policy", anchor_policy,
                     "--distill-weight", str(a.distill_weight)]
            if indices:
                argv += ["--anchor-task-indices", ",".join(map(str, indices))]
        if rung.number == 7 and LEGACY_WALK_TEACHER.exists():
            argv += ["--transfer-policy", str(LEGACY_WALK_TEACHER),
                     "--transfer-obs-dim", "50"]
            prior = self.out / "priors" / "rung_07_walk_prior.json"
            if prior.exists():
                argv += ["--action-prior-json", str(prior)]
        return argv

    def _ensure_walk_prior(self) -> str | None:
        """Search and persist the rung-7 teacher once, then reuse its artifact."""
        prior = self.out / "priors" / "rung_07_walk_prior.json"
        if self.args.tiny or self.args.dry_run:
            return None
        rung6 = self.state.get("checkpoints", {}).get("6")
        if not rung6 or not Path(rung6).exists() or not LEGACY_WALK_TEACHER.exists():
            raise RuntimeError("rung 7 requires accepted rung-6 and legacy walk teachers")
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
            pass
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
                             label: str) -> tuple[int, dict]:
        output = self.out / "logs" / f"{label}.json"
        log = self.out / "logs" / f"{label}.log"
        argv = [self.args.python, "-u", str(EVALUATOR), "eval",
                "--geometry", str(rung.geometry), "--checkpoint", checkpoint,
                "--episodes", "1", "--steps", str(self.args.retention_steps),
                "--envs", str(self.args.retention_envs),
                "--seed", str(self.args.retention_seed + rung.number),
                "--out", str(output)]
        if rung.geometry in ("ladder_locomotion", "ladder_combat"):
            argv += ["--rung", str(rung.number)]
        if self.args.device:
            argv += ["--device", self.args.device]
        rc = self._run(argv, log)
        metrics = json.loads(output.read_text()) if rc == 0 and output.exists() else {}
        return rc, metrics

    def _retention_gate(self, rung: Rung, checkpoint: str) -> tuple[bool, list[dict]]:
        """Replay prior same-family tasks and reject catastrophic forgetting."""
        if self.args.no_gates or self.args.tiny or self.args.dry_run:
            return True, []
        previous = [item for item in RUNGS
                    if item.number in self.state["completed"]
                    and item.number < rung.number and item.family == rung.family
                    and item.geometry is not None]
        if self.args.retention_max > 0:
            previous = previous[-self.args.retention_max:]
        report: list[dict] = []
        all_ok = True
        for learned in previous:
            rc, current = self._evaluate_checkpoint(
                learned, checkpoint,
                f"retention_after_{rung.number:02d}_task_{learned.number:02d}")
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
            report.append({"candidate_rung": rung.number, "replayed_rung": learned.number,
                           "checkpoint": checkpoint, "pass": task_ok, "checks": checks})
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

    def _run_ppo(self, rung: Rung, *, opponent: str | None = None,
                 tag_suffix: str = "", warm_override: str | None = None
                 ) -> tuple[bool, dict, str | None]:
        tag = self.out / f"rung_{rung.number:02d}_{rung.slug}{tag_suffix}"
        if rung.number == 7:
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
            if candidate_path.exists() and self._failed_on_entry == rung.number else 0)
        for offset in range(1, self.args.attempts + 1):
            attempt = completed_attempts + offset
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
            if passed:
                train_metrics = last_metrics
                if not (self.args.tiny or self.args.no_gates):
                    fixed_rc, fixed_metrics = self._evaluate_checkpoint(
                        rung, str(tag) + ".pt", f"baseline_task_{rung.number:02d}")
                    fixed_ok, fixed_details = self._gate(rung, fixed_metrics)
                    fixed_checks = self._current_task_regression_checks(
                        rung, fixed_metrics)
                    self._record_regression_task(
                        rung, str(tag) + ".pt", rung, fixed_checks,
                        fixed_rc == 0 and fixed_ok, "candidate_task")
                    print("FIXED-SEED " + "; ".join(fixed_details), flush=True)
                    if fixed_rc != 0 or not fixed_ok:
                        print(f"RUNG {rung.number:02d} did not reproduce its pass gate on the "
                              "fixed retention seed", flush=True)
                        if rung.number == 7:
                            self._retarget_walk_prior(fixed_metrics)
                            self._refine_walk_prior(fixed_metrics)
                        continue
                    last_metrics = dict(fixed_metrics, train_eval=train_metrics)
                retention_ok, retention = self._retention_gate(
                    rung, str(tag) + ".pt")
                last_metrics = dict(last_metrics, retention_pass=retention_ok,
                                    retention_replays=len(retention))
                if retention_ok:
                    return True, last_metrics, str(tag) + ".pt"
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
        final_metrics = dict(final_metrics, pfsp_hall_size=len(hall))
        return True, final_metrics, final_ckpt

    def _pfsp_opponent(self, rung: Rung, candidate: str, hall: list[str],
                       round_index: int) -> str:
        """Prioritized fictitious self-play over measured archive matchups.

        Opponents that the current candidate rarely beats receive the largest
        weight. Sampling is seeded, so a resumed run makes the same choice.
        """
        if self.args.tiny:
            return hall[round_index % len(hall)]
        rows = []
        for index, opponent in enumerate(hall):
            label = f"pfsp_round_{round_index:02d}_opponent_{index:02d}"
            output = self.out / "logs" / f"{label}.json"
            log = self.out / "logs" / f"{label}.log"
            argv = [self.args.python, "-u", str(EVALUATOR), "eval",
                    "--geometry", str(rung.geometry), "--rung", str(rung.number),
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
        choice = random.Random(self.args.seed + round_index).choices(
            range(len(hall)), weights=weights, k=1)[0]
        event = {"round": round_index, "candidate": candidate,
                 "selected": hall[choice], "matchups": rows}
        self.state.setdefault("pfsp_history", []).append(event)
        self._save()
        print(f"PFSP round {round_index}: selected {Path(hall[choice]).name}; "
              f"weights={[round(value, 3) for value in weights]}", flush=True)
        return hall[choice]

    def _run_search(self, rung: Rung) -> tuple[bool, dict, str | None]:
        checkpoint = self._previous_checkpoint(rung)
        if checkpoint is None:
            return False, {"error": "co-design search requires rung 30 checkpoint"}, None
        output = self.out / f"rung_{rung.number:02d}_{rung.slug}.json"
        log = self.out / "logs" / f"rung_{rung.number:02d}_{rung.slug}.log"
        candidates = 4 if self.args.tiny else self.args.search_candidates
        generations = 1 if self.args.tiny else self.args.search_generations
        steps = 4 if self.args.tiny else self.args.search_steps
        envs = 2 if self.args.tiny else min(self.args.envs, self.args.search_envs)
        argv = [self.args.python, "-u", str(SEARCH), "design", "--candidates", str(candidates),
                "--generations", str(generations), "--steps", str(steps), "--envs", str(envs),
                "--seed", str(self.args.seed + rung.number), "--checkpoint", checkpoint,
                "--out", str(output)]
        if self.args.device:
            argv += ["--device", self.args.device]
        rc = self._run(argv, log, dry_run=self.args.dry_run)
        if self.args.dry_run:
            return True, {"dry_run": True}, checkpoint
        metrics = json.loads(output.read_text()) if rc == 0 and output.exists() else {"returncode": rc}
        passed, details = self._gate(rung, metrics)
        print("; ".join(details), flush=True)
        return rc == 0 and passed, metrics, checkpoint

    def execute(self) -> int:
        self._audit_completed_contracts()
        for rung in selected_rungs(self.args.first, self.args.last):
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
                passed, metrics, checkpoint = self._run_ppo(rung, opponent=opponent)
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
    run.add_argument("--checkpoint-replay-steps", type=int, default=32,
                     help="fixed rollout length compared immediately across save/reload")
    run.add_argument("--epochs", type=int, default=4)
    run.add_argument("--minibatches", type=int, default=8)
    run.add_argument("--target-kl", type=float, default=0.02,
                     help="whole-rollout PPO KL target for adaptive epoch/lr control")
    run.add_argument("--kl-stop-multiplier", type=float, default=1.5,
                     help="stop remaining PPO epochs above this target-KL multiple")
    run.add_argument("--hidden", default="512,256,128")
    run.add_argument("--architecture", choices=("mlp", "task_film"), default="task_film")
    run.add_argument("--distill-weight", type=float, default=0.05,
                     help="old-policy behavior replay weight between sequential rungs")
    run.add_argument("--seed", type=int, default=20260712)
    run.add_argument("--pfsp-rounds", type=int, default=4)
    run.add_argument("--search-candidates", type=int, default=16)
    run.add_argument("--search-generations", type=int, default=4)
    run.add_argument("--search-steps", type=int, default=160)
    run.add_argument("--search-envs", type=int, default=64)
    run.add_argument("--walk-search-population", type=int, default=64)
    run.add_argument("--walk-search-repeats", type=int, default=4)
    run.add_argument("--walk-search-generations", type=int, default=12)
    run.add_argument("--walk-search-steps", type=int, default=400)
    run.add_argument("--no-gates", action="store_true")
    run.add_argument("--retention-tolerance", type=float, default=0.20,
                     help="maximum relative regression from a skill's learned baseline")
    run.add_argument("--retention-absolute", type=float, default=0.02,
                     help="minimum absolute regression allowance for near-zero metrics")
    run.add_argument("--retention-steps", type=int, default=200)
    run.add_argument("--retention-envs", type=int, default=32)
    run.add_argument("--retention-seed", type=int, default=2026071200)
    run.add_argument("--retention-max", type=int, default=0,
                     help="replay only the latest N prior skills (0 means every prior skill)")
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
    return LadderRunner(args).execute()


if __name__ == "__main__":
    raise SystemExit(main())
