<!-- SPDX-License-Identifier: MIT -->
# Historical documentation archive

> **Document status:** Reference · **Audience:** Maintainers and researchers tracing decisions · **Last reviewed:** 2026-07-12 · **Canonical for:** Finding superseded plans, retired procedures, and dated research records

Historical files remain at their original paths so old issue, commit, and report
links continue to resolve. Being listed here means “preserved as evidence,” not
“safe to follow as current instructions.” Current navigation starts at
[`../README.md`](../README.md).

## Retired procedures

- [`../gpu-runbook.md`](../gpu-runbook.md) — retired MJX/JAX GPU workflow;
  replaced by [`../pre-gpu-test-entrypoint.md`](../pre-gpu-test-entrypoint.md).
- [`../gpu-pod-setup.md`](../gpu-pod-setup.md) — retired environment setup;
  replaced by `sim/robot/setup_warp_pod.sh` and the canonical entry-point guide.

## Superseded explanations and decisions

- [`../system-tour.md`](../system-tour.md) — readable July 1 narrative, but its
  backend and walking status are historical. Use
  [`../getting-started.md`](../getting-started.md).
- [`../codesign-architecture.md`](../codesign-architecture.md) — early MJX
  co-design understanding bundle.
- [`../gait-feasibility-verdict.md`](../gait-feasibility-verdict.md) — valuable
  mechanism investigation whose hardware baseline was superseded.
- [`../architecture.md`](../architecture.md) remains current for the component
  bench decision, but it is not the whole-system architecture.

## Execution records

- [`../uplift-execution-plan.md`](../uplift-execution-plan.md) — detailed audit
  execution and dated GPU logs; durable verdict in
  [`../training-uplift-results.md`](../training-uplift-results.md).
- Completed `*-checklist.md` files — implementation history retained after their
  result moved into code, tests, or a shorter report.
- Dated `*-report.md` files — evidence for their tested configuration, not
  automatically the current design.

## Research snapshots

- [`../sim-engine-secret-sauce.md`](../sim-engine-secret-sauce.md) — July 2026
  simulator survey; not an active backend runbook.
- [`../sota-training-issues.md`](../sota-training-issues.md) — dated literature
  and project-mapping survey.
- [`../warp-ladder-results.md`](../warp-ladder-results.md) — dated migration and
  throughput evidence.

The generated [`../document-catalog.md`](../document-catalog.md) classifies every
Markdown file and is the exhaustive inventory.
