<!-- SPDX-License-Identifier: MIT -->
# Module contract template

Every reusable leaf IP block ships a `rtl/contracts/<module>.md` following this
exact shape, so the set is uniform and auto-summarizable into the status matrix
(`notes/status-matrix.md`). Copy this file, fill every section, delete the
guidance in _italics_.

---

# `<module>` — one-line purpose

_What it does, in a sentence._

## Interface

| Port | Dir | Width | Signed | Reset | Semantics |
| --- | --- | --- | --- | --- | --- |
| `clk` | in | 1 | — | — | rising-edge clock |
| `rst_n` | in | 1 | — | — | async, active-low reset |
| … | | | | | |

## Clocking & reset

- **Clock domains:** _single `clk` / list them._
- **Reset:** _async active-low `rst_n`; what each output is during/after reset._
- **Latency:** _cycles from input/strobe to output; combinational vs registered._
- **Multi-cycle / false paths:** _none, or list._

## Parameters

| Parameter | Default | Legal range | Meaning |
| --- | --- | --- | --- |
| `…` | `…` | `…` | _width-sized; see the convention in trusted-library-checklist 1.0_ |

## Formal contract

- **Proven** (`formal/manifest.toml` id `<id>`): _the property, verbatim._
- **Assumptions:** _what the proof relies on (reset discipline, input bounds)._
- **Method / scope:** _prove (k-induction) | bmc; config | envelope | generic._
- **Only simulated (not proven):** _what the sim tier covers but formal does not._

## Synthesis fit

- **Device:** _ECP5 LFE5U-85F (or N/A)._
- **Utilisation:** _LUT / FF / DSP / BRAM._
- **Fmax:** _MHz, or "combinational"._

## Reuse notes

- **Language:** _Verilog-2005 / SystemVerilog; tool compatibility._
- **Dependencies:** _none (self-contained) / child modules it instantiates._
- **Gotchas:** _e.g. the table data must be regenerated if `BITS` changes._
