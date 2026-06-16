<!-- SPDX-License-Identifier: MIT -->
# Formal Property Verification

Open-source formal proofs of the RTL's plant-independent safety properties.
While the lockstep bench *observes* invariants across finite simulation, the
formal flow *proves* them for all reachable states — so the headline safety
claims (no shoot-through, dead-time minimum) hold unconditionally, regardless
of the still-unmeasured motor parameters.

**This is verification, not validation.** A proof is a property of the RTL
under stated assumptions; it says nothing about correspondence to real silicon
or the real motor (that needs hardware). See
[`../notes/formal-checklist.md`](../notes/formal-checklist.md) for the full
methodology and design decisions, and **[`proof_report.md`](proof_report.md)
for what is currently proven and under what assumptions.**

## Quick start

```bash
formal/check_formal_toolchain.sh         # yosys + sby + an SMT solver
python3 formal/run_formal.py             # run every proof in the manifest
python3 formal/gen_proof_report.py       # regenerate proof_report.md
python3 formal/run_formal.py --check     # CI gate: all proofs hold, none vacuous
```

Toolchain: the [YosysHQ OSS CAD Suite](https://github.com/YosysHQ/oss-cad-suite-build/releases)
(Yosys + SymbiYosys + smtbmc + boolector/yices/bitwuzla/z3) at `~/oss-cad-suite`
— fully open, no proprietary tools. `run_formal.py` adds it to `PATH`
automatically.

## How it fits together

- `manifest.toml` — the single source of truth: every property, its method
  (`prove`/`bmc`/`cover`), engine, **assumptions**, **non-vacuity covers**, and
  parameter scope. Human-writable, machine-parseable.
- `run_formal.py` — regenerates `rtl/gen/rtl_params.vh`, then for each check
  generates a `.sby`, runs SymbiYosys, and records the status. Safety checks
  with covers are run a second time in cover mode — a proof whose interesting
  states are unreachable is reported **INCOMPLETE (vacuous)**, never PROVEN.
- `gen_proof_report.py` — renders `proof_report.md` from the manifest +
  results: the status banner (PROVEN / BOUNDED / DOCUMENTED / INCOMPLETE /
  FAILED), per-check detail with assumptions and cover status, and two
  completeness views (which modules have proofs; which safety proofs have
  passing covers).
- `bind/` — `*_fv.sv` checker modules `bind`-ed into each DUT (the design stays
  clean; checkers reach internals).
- `lib/` — reusable parameterized checkers (see `lib/README.md`).

## Status vocabulary

| Status | Meaning |
| --- | --- |
| **PROVEN** | unbounded proof converged (k-induction / PDR) — true for all reachable states |
| **BOUNDED(N)** | no counterexample within N cycles — evidence, not a proof |
| **DOCUMENTED** | not machine-proven (e.g. intractable for the open SMT stack); bounded by construction + sim-validated, with the argument recorded |
| **INCOMPLETE** | proof passed but a non-vacuity cover was unreachable (possibly vacuous) |
| **FAILED** | a counterexample exists |

## Adding a new proven HDL block

1. Drop the module in `rtl/`.
2. Write `formal/bind/<module>_fv.sv`: `bind` the relevant `formal/lib/`
   checkers (which bring their own assumptions and covers), plus any bespoke
   `assert`/`cover` properties. Reach internal signals through bind port
   connections (including generate-block paths like `leg[0].off_time_l`).
3. Add a `[[check]]` to `manifest.toml` with the RTL files, `method`,
   `assumptions`, `cover_props`, and `param_scope`.
4. `python3 formal/run_formal.py --only <id> && python3 formal/gen_proof_report.py`
   — the block's guarantees now appear in the proof report.

### Conventions

- **Combinational modules**: use a clocked wrapper `top` (a `*_fv` module that
  instantiates the DUT and samples assertions on a free clock), and declare
  `top = "<module>_fv"` in the manifest. Sequential modules bind into the DUT
  directly (`top` defaults to the module).
- **Bounded-input assumptions**: a unit proof of a module that the design feeds
  a restricted range (e.g. `svpwm`, `circle_limit`) must `assume` that range —
  and the manifest records it. The top-level composition proof discharges such
  assumptions against the upstream logic.
- **Temporal properties**: encode with `$past` and small helper FFs (the open
  Yosys SVA subset does not support full SVA sequences).
- **Every assumption-guarded safety proof ships non-vacuity covers.**
  Structural proofs with no input assumptions (e.g. FSM legality) cannot be
  vacuous and may omit covers — note why in the justification.
