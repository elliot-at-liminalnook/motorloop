<!-- SPDX-License-Identifier: MIT -->
# Codified Circuit Specifications

SPICE netlists for the measurement channels between the power stage and the
MCP3208, used to *derive* simulation parameters instead of hand-computing
them (see `../../notes/derivation-checklist.md`).

Conventions:

- One `.cir` per measurement channel.
- Component values are NEVER hardcoded in a netlist. They live in
  `sim/config/params.toml` under `[circuit.*]` (with per-component
  provenance) and are injected via the generated include
  `sim/build/spice/components.param` — regenerate with
  `python3 sim/scripts/derive_params.py --write-spice-params`.
- Every derived parameter in params.toml carries
  `derived_from = "circuit.<name>"`; `sim/tests/test_derived_params.py`
  re-derives all of them closed-form, and `test_spice_derivations.py`
  cross-checks against ngspice analyses of these netlists.
- ngspice runs are cached in `sim/build/spice/` keyed by netlist+parameter
  hash (`sim/scripts/spice_runner.py`).

When the ZONRI board is measured (Q7), update the `[circuit.*]` component
values and statuses — every downstream parameter then recomputes via
`derive_params.py --update`.
