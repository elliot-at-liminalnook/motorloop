<!-- SPDX-License-Identifier: MIT -->
# Documentation home

> **Document status:** Current · **Audience:** All readers · **Last reviewed:** 2026-07-14 · **Canonical for:** Documentation navigation and document ownership

Start with the question you need answered. This page intentionally favors a
small set of maintained entry points over a chronological list of everything
ever written.

## Start here

- [`getting-started.md`](getting-started.md) — the first 15 minutes: project
  scope, non-goals, and safe first commands.
- [`current-status.md`](current-status.md) — what is implemented, verified,
  active, historical, and still unvalidated.
- [`system-architecture.md`](system-architecture.md) — the motor-control and
  robot systems, their real seam, and their sources of truth.
- [`reader-paths.md`](reader-paths.md) — short routes for FPGA, simulation,
  robot-learning, hardware, reuse, and documentation work.
- [`glossary.md`](glossary.md) — project acronyms and evidence vocabulary.

## Run and reproduce

- [`reproduce.md`](reproduce.md) — canonical full-repository setup and
  verification procedure.
- [`blender-agent-workflow.md`](blender-agent-workflow.md) — safe live control,
  screenshot inspection, visual guidance, and validation for Blender/Phobos
  modeling sessions.
- [`pre-gpu-test-entrypoint.md`](pre-gpu-test-entrypoint.md) — canonical robot/RL
  precheck, CUDA verification, and long-run authorization.
- [`training-ladder-runbook.md`](training-ladder-runbook.md) — current 31-rung
  training state, exact resume command, adaptive enforcement, and safe shutdown.
- [`training-objective-contract.md`](training-objective-contract.md) — decides
  what may be an outcome, physical constraint, efficiency cost, temporary
  scaffold, promotion gate, or diagnostic.
- [`sim/README.md`](../sim/README.md) — focused component-bench development and
  manual scenarios.
- [`formal/README.md`](../formal/README.md) — running and interpreting formal
  properties.

Retired GPU commands live in the [`archive`](archive/README.md), not in current
navigation.

## Architecture and contracts

- [`system-architecture.md`](system-architecture.md) — whole repository.
- [`architecture.md`](architecture.md) — lockstep Verilator bench decision.
- [`robot-hardware-contract.md`](robot-hardware-contract.md) — active physical
  robot envelope.
- `rtl/contracts/` — reusable RTL interfaces and timing.
- [`verification-plan.md`](verification-plan.md) — requirement-to-proof/test map.

## Current evidence

- [`current-status.md`](current-status.md) — canonical high-level interpretation.
- [`status-matrix-generated.md`](status-matrix-generated.md) — generated formal
  and ECP5 summary.
- [`formal/proof_report.md`](../formal/proof_report.md) — proof assumptions and
  results.
- [`synth/synth_report.md`](../synth/synth_report.md) — authoritative open ECP5
  fit and timing result.
- [`runpod-warp-validation-2026-07-10.md`](runpod-warp-validation-2026-07-10.md)
  — dated evidence for the current combined launch gate.

## Active robot work

- [`locomotion-status.md`](locomotion-status.md) — short current promotion state
  and combat prerequisite.
- [`training-ladder-runbook.md`](training-ladder-runbook.md) — resumable ladder
  state and regression-preserving handoff.
- [`training-objective-contract.md`](training-objective-contract.md) — current
  outcome-based reward and gate design rules.
- [`universal-command-contract.md`](universal-command-contract.md) —
  commands-only v2 observations: rungs select environments and gates, never
  policy inputs.
- [`predictive-universal-controller.md`](predictive-universal-controller.md) —
  experimental morphology-token GRU, self-supervised future model, and
  confidence-limited prediction-gradient planner.
- [`predictive-transformer-proof-2026-07-15.md`](predictive-transformer-proof-2026-07-15.md)
  — first matched L40S recurrent-versus-Transformer predictor result.
- [`locomotion-bootstrap-teacher-checklist.md`](locomotion-bootstrap-teacher-checklist.md)
  — detailed append-only locomotion work record.
- [`sparc-learning-log.md`](sparc-learning-log.md) — detailed combat-learning log.
- [`codesign-win-exchanges-checklist.md`](codesign-win-exchanges-checklist.md) —
  fight-curriculum execution record.

## Durable findings

- [`training-uplift-results.md`](training-uplift-results.md) — concise verdict
  from the actuator-gear audit and historical walking milestone.
- [`rl-verification-playbook.md`](rl-verification-playbook.md) — checks required
  for model, reward, trajectory, and checkpoint changes.
- [`ethos.md`](ethos.md) — why claims, assumptions, and falsification shape the
  repository.
- [`docs-digest.md`](docs-digest.md) — extracted component facts with provenance
  warnings.

## History and exhaustive inventory

- [`archive/README.md`](archive/README.md) — retired procedures, superseded
  explanations, dated plans, and research snapshots.
- [`document-catalog.md`](document-catalog.md) — generated classification of
  every Markdown file in the repository.

## Maintaining these docs

[`documentation-guide.md`](documentation-guide.md) defines the required status
header, document shapes, canonical ownership, lifecycle states, and checks.

```bash
python3 scripts/check_docs.py
make docs
```
