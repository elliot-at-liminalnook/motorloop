<!-- SPDX-License-Identifier: MIT -->
# Documentation guide

> **Document status:** Current · **Audience:** Contributors and maintainers · **Last reviewed:** 2026-07-12 · **Canonical for:** Documentation structure, lifecycle, and writing conventions

Motorloop documentation is organized by what a reader is trying to do, not by
the order in which the project happened. The curated entry point is
[`notes/README.md`](README.md); the complete inventory is
[`document-catalog.md`](document-catalog.md).

## Information architecture

| Section | Reader question | Typical documents |
| --- | --- | --- |
| Start here | What is this, and where do I begin? | overview, getting started, reader paths |
| Concepts | How does the system fit together? | architecture, ethos, glossary |
| How-to | How do I complete a task? | reproduction and launch procedures |
| Reference | What is the exact interface or contract? | module contracts, status matrices, hardware contracts |
| Decisions | Why was this design chosen? | architecture decision records |
| Results | What was demonstrated, under which assumptions? | reports and validation evidence |
| Active work | What is being investigated now? | bounded checklists and learning logs |
| Archive | What happened before the current design? | superseded plans, retired runbooks, historical surveys |

## Lifecycle vocabulary

Document status is separate from the maturity of the system it describes.

| Document status | Meaning |
| --- | --- |
| **Current** | Maintained guidance or explanation; safe to follow now |
| **Active** | A live work record; useful for current progress but not a stable interface |
| **Reference** | Stable facts, contracts, or background material |
| **Generated** | Mechanically produced evidence; do not edit by hand |
| **Historical** | Preserved evidence from an earlier state; not current guidance |
| **Retired** | Intentionally unsupported procedure; follow its replacement link |

The machine-readable classification for every Markdown file is
[`docs-lifecycle.toml`](../docs-lifecycle.toml). `python3 scripts/check_docs.py`
fails if a file is unclassified, a required status header is missing, a local
link is broken, or a current document sends readers to a retired procedure.

## Required opening

Every current entry page and canonical procedure begins with one compact line:

```markdown
> **Document status:** Current · **Audience:** <reader> · **Last reviewed:** YYYY-MM-DD · **Canonical for:** <scope>
```

Historical and active logs use the same line, name their replacement or summary,
and retain their original dates. A date means “reviewed against the repository,”
not “all claims were rerun on that date.” Generated evidence states its own
generation source.

## Standard shapes

Use the smallest shape that makes the document answerable.

### Explanation or architecture

1. Purpose and scope
2. One diagram or mental model
3. Components and data flow
4. Boundaries and non-goals
5. Evidence and related documents

### How-to or runbook

1. Outcome
2. Prerequisites
3. Canonical command sequence
4. Expected result
5. Failure recovery
6. Related reference material

There must be one canonical runbook per operation. Other pages summarize and
link to it; they do not copy a second command sequence.

### Result or verdict

1. Claim in one sentence
2. Scope and tested configuration
3. Evidence and reproduction command
4. Limitations
5. Date and supersession state

### Execution log

1. Link to a short current summary
2. Current gate and owner-visible next action
3. Dated append-only entries

Do not turn a plan into the permanent explanation of a subsystem. When a plan
produces a durable result, extract a short results page and mark the plan as an
execution record.

## Canonicality rules

- [`current-status.md`](current-status.md) owns high-level maturity claims.
- Generated proof and synthesis reports own volatile counts and timing numbers.
- [`reproduce.md`](reproduce.md) owns full repository reproduction.
- [`pre-gpu-test-entrypoint.md`](pre-gpu-test-entrypoint.md) owns robot/RL launch
  authorization and CUDA-host verification.
- [`robot-hardware-contract.md`](robot-hardware-contract.md) owns the active
  physical design envelope.
- Per-module files under `rtl/contracts/` own reusable RTL interfaces.

When two documents disagree, follow the more specific canonical source. Fix the
other document or mark it historical; do not reconcile contradictions by adding
another status paragraph.

## Writing conventions

- Define an acronym on first use and link to [`glossary.md`](glossary.md) when it
  is central to the page.
- Put the conclusion before the debugging story.
- Use “verified,” “validated,” “measured,” and “demonstrated” only according to
  the maturity definitions in [`current-status.md`](current-status.md).
- Prefer repository-relative links to prose such as “some script in sim.”
- Keep volatile counts in generated reports. Stable pages state whether a gate
  passes and link to the evidence.
- Use diagrams for system boundaries and data flow, not decoration.

## Contributor check

```bash
python3 scripts/check_docs.py
make docs
```

The first command checks source documentation. The second assembles the curated
site and runs MkDocs in strict mode.
