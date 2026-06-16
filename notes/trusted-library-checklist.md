<!-- SPDX-License-Identifier: MIT -->
# Trusted-Library Checklist (stages 1–3)

Ordered tasks to turn motorloop from a co-verified *system* into a **trusted,
reusable HDL library** — one a serious hardware lab would vendor, and from which
they can pull a single module into their own design without dragging the whole
project along. Companion to [platform-abstraction-checklist](platform-abstraction-checklist.md),
[foc-checklist](foc-checklist.md), [formal-checklist](formal-checklist.md);
architecture in [architecture](architecture.md).

This covers the **first three** of the five-stage library plan:

1. **Parameterize the leaf modules** — kill the global-macro coupling (this file, Stage 1).
2. **Package + per-module contracts** — SPDX/license, FuseSoC cores, contract docs, status matrix (Stage 2).
3. **Reproducible CI + releases** — pinned toolchains, four CI gates, semver (Stage 3).

Out of scope here (the later two, where the *earned* trust comes from): **4.**
pipeline `foc_core` for timing closure (Stage 15 found Fmax ≈ 3.3 MHz — the
unpipelined FOC datapath); **5.** a hardware-correlation/validation tier on one
BOM. They get their own checklist once 1–3 land.

**The core idea — a layered library.** "Trustworthy integrated system" and
"cherry-pickable modules" pull in opposite directions. The resolution is to make
the split explicit: a set of self-contained, individually-verified **leaf IP
blocks** (parameterized, contracted, proven), plus the **integrated motorloop
system** that composes them (the co-sim + the top-level composition proof). A lab
can trust the whole, or pull one leaf. Every task below serves that split.

**The reuse linchpin (Stage 1).** Almost every module does `` `include
"rtl_params.vh" `` and references global macros (`` `PWM_HALF_PERIOD ``,
`` `CUR_PI_KP `` …). That means reusing one module imports motorloop's entire
global config — a non-starter. Converting those macros to real Verilog
**`parameter`s with sane defaults** is the single highest-leverage change; it
also makes the formal proofs cleaner (constants become explicit
`param_scope: envelope` ranges instead of baked values).

## Definition of done

- **Stage 1:** no leaf module includes `rtl_params.vh`; each takes its
  motorloop-specific constants as parameters (defaulting to today's values);
  `controller_top` is the only integration point that reads `rtl_params.vh` and
  passes the generated values down. The 401-test sim suite, the 12-PROVEN +
  1-DOCUMENTED formal suite, and the ECP5 synth all stay green and **byte-
  identical** (parameters default to the current values, `controller_top`
  overrides them). At least two leaf proofs are upgraded to `param_scope:
  envelope` (parameter-generic) to demonstrate parameterization strengthened the
  formal story.
- **Stage 2:** every source file carries an SPDX header; each leaf IP has a
  `MODULE.md` contract (interface / timing / params / formal contract / synth
  fit) and a FuseSoC `.core`; `fusesoc run` builds a chosen leaf standalone; a
  status matrix (module × proven/sim/synth/Fmax) renders from those.
- **Stage 3:** CI runs lint + sim + formal + synth on every PR with **pinned**
  toolchain versions; the status matrix regenerates in CI; the repo cuts a
  semver-tagged library release with a CHANGELOG and the matrix/proof-report/
  bitstream attached; a "reproduce from scratch" doc takes a clean clone to green.

## Design decisions (pre-resolved)

- **Behavior-preserving invariant (the safety rail).** Every refactor keeps the
  full sim + formal + synth green. Parameters **default to today's values**, and
  `controller_top` passes the `rtl_params.vh`-generated values explicitly to each
  instance, so the integrated system is byte-identical at every step. Convert one
  module, re-run its tests + proof, move on — never a big-bang rewrite.
- **`rtl_params.vh` survives as the *system* config**, injected at the top by
  `gen_rtl_params.py`; leaf modules stop including it. The generator is
  unchanged; only *who consumes* the header moves up to `controller_top`.
- **Shared types in a `motorloop_pkg.sv` SV package** (only what modules
  genuinely share: the Q15 fixed-point widths, the `leg_mode` encoding) — nothing
  motor-specific. Pure-datapath leaves (clarke/park) need no package.
- **Language target = Verilog-2005-compatible leaf RTL** wherever the module
  already is (max tool compatibility: Vivado/Quartus/Verilator/yosys all accept
  it), SV only where it adds value (the package/typedefs). Documented per module.
- **License = the existing `LICENSE`** — confirm it is permissive (recommend
  **Apache-2.0**: patent grant, standard for vendored HDL; relicense if the
  current file is copyleft) and add SPDX headers referencing it. Final choice is
  the maintainer's call — a Stage-2 decision point, not a blocker.
- **Package format = FuseSoC `.core`** (the open-HDL lingua franca; vendor flows
  that don't use it still get clean standalone files) + one top core for the
  system.
- **Contracts are uniform.** Every leaf `MODULE.md` follows one fixed template so
  the set is auto-summarizable into the status matrix and auditable at a glance.
- **Leaf-first ordering.** The grep census (2026-06-14): 6 modules are already
  macro-free (`clarke`, `park`, `inv_park`, `commutation`, `open_loop_ramp`,
  `divider32`); 11 have a single macro; `controller_top` (7) and `drv_manager`
  (9) are the heavily-coupled integration/system blocks. Convert easiest → hardest.

---

## Stage 1 — Parameterize the leaf modules (kill the global-macro coupling)

### 1.0 — Conventions + the shared package
- [~] `rtl/motorloop_pkg.sv` **deferred (honest scoping):** the parameterization
      made every module self-contained without a shared package — modules
      interface via plain bit-vectors, so there is no shared *type* that two
      modules must agree on yet. A package would be dead code today; it lands
      when a genuine shared typedef appears (e.g. a Q15 struct). Noted, not faked.
- [x] **Parameter convention** (established + applied): each motorloop-specific
      macro a leaf used becomes a `parameter` named the same as the macro, its
      current value as the default, **sized to its datapath width** (e.g. `[15:0]
      PWM_HALF_PERIOD = 16'd625`, `[7:0] DRV_SPI_DIV`) — width-sizing is required
      because a parameter (unlike the old constant-folded macro) is not resized
      to context, so a too-wide param truncates/expands its sized localparams.
      Halving via `>> 1` not `/ 2` (the `2` literal would force 32-bit).
- [x] **Per-module recipe** proven (the byte-identical loop): convert leaf →
      `controller_top` passes `.PARAM(`MACRO)` → rebuild → run that module's
      tests + proof. Defaults = today's values, so the system stays byte-
      identical at every step (201-test checkpoint green twice; 12 PROVEN + 1
      DOCUMENTED formal re-run green).

### 1.1 — The already-self-contained leaves (confirm + lock in) ✅
- [x] `clarke`, `park`, `inv_park`, `commutation`, `open_loop_ramp`, `divider32`
      use no macros — confirmed reusable as-is (no RTL change; SPDX in Stage 2).
      The proof the layered model is real on day one.

### 1.2 — Single-macro leaves (one parameter each) ✅
- [x] Converted (macro → param, include removed): `sincos` (BITS), `svpwm`
      (PWM_HALF_PERIOD), `circle_limit` (V_CIRCLE_LIMIT), `spi_drv_master` /
      `adc_spi_master` / `as5047p_spi_master` (SPI divider, sized `[7:0]`),
      `as5600_pwm_capture` (ANGLE_CARRIER_CYC), `uart_rx` / `uart_tx` (UART_DIV,
      `[15:0]`), `foc_core` (its own PWM_HALF_PERIOD + threads BITS/V_CIRCLE_LIMIT/
      CUR_PI_* to its children). `uart_regfile` threads UART_DIV to uart_rx/tx +
      its own UART_TIMEOUT_CYC.

### 1.3 — Multi-macro datapath leaves (2–4 parameters) ✅
- [x] `pwm_generator` (PWM_HALF_PERIOD/DEAD_CYCLES/MIN_PULSE_CYCLES, `[15:0]`),
      `current_pi` (CUR_PI_KP/KI_SHIFT/V_RAW_MAX), `speed_pi` (KP/KI_SHIFT/
      PWM_HALF_PERIOD/DUTY_DOWN_SLEW — `DUTY_MAX` split into a 32-bit intermediate
      + `[15:0]` slice to dodge the narrow-initializer truncation), `speed_iq_pi`
      (KP/KISH/IQ_MAX), `speed_meter` (CLK_HZ/SPEED_NUM), `ads9224r_master`
      (ADC_SPI_DIV/PWM_HALF_PERIOD/ADC_EMF_LEAD). Each re-proven where it carries
      a proof.

### 1.4 — The macro-heavy system leaves ✅
- [x] `adc_sequencer` (5 params; `CR`-style narrow initializers widened, count
      params sized) and `drv_manager` (9 params; `CR1/CR2_VALUE` split into 32-bit
      full + `[10:0]` slice, `LOCKOUT_N`/`DRV_DEAD_N` sized `[3:0]`). Both
      re-proven: `drv_manager` FSM legality + `adc_sequencer` pulse proof PROVEN.

### 1.5 — `controller_top` becomes the explicit integration layer ✅
- [x] `controller_top` keeps `` `include "rtl_params.vh" `` (the *system* config)
      and passes every generated value down via `#(...)` overrides on all 15
      parameterized instances; its own 7 macro uses stay (it is the system top,
      not a reusable leaf). Verified byte-identical (201-test checkpoint) and the
      config is now *live* by construction — `rtl_params.vh` → `.PARAM(`MACRO)` →
      the leaf's parameter (no longer a per-leaf include).

### 1.6 — Re-green + strengthen the formal story ✅
- [x] Full formal suite stays **12 PROVEN + 1 DOCUMENTED** (re-run with all
      parameterized modules); the 201-test FOC-math + all-8-platforms checkpoint
      is green before and after the threading. Full sim regression reconfirmed
      **401 passed, 1 skipped** (byte-identical); ECP5 synth path unchanged.
- [x] Upgraded **2 leaf proofs to `param_scope: envelope`** (parameter-generic):
      `current_pi` (clamp for any `V_RAW_MAX`) and `speed_iq_pi` (clamp for any
      `IQ_MAX`) — the checker now takes the bound as the module's *own* parameter
      through the bind (`#(.V_RAW_MAX(V_RAW_MAX))`), decoupled from the global
      macro, so the proof is valid for *anyone's* parameter. Both re-PROVEN. The
      concrete payoff of parameterization on the formal side.

## Stage 2 — Packaging + per-module contracts

### 2.1 — License + provenance headers ✅
- [x] `LICENSE` is already **MIT** (permissive — kept, not relicensed; Apache-2.0
      remains the maintainer's option for an explicit patent grant). SPDX
      `MIT` headers added to **167 files** by `scripts/add_spdx.py` (idempotent;
      correct comment syntax per file type — `//`, `#`, `<!-- -->`; placed after a
      shebang or a required first line like `CAPI=2:`). `add_spdx.py --check` is
      the gate (0 missing). Verified behavior-safe: rebuild clean + smoke green.
- [x] **`reuse lint` EXECUTED and CLEAN** (reuse 6.2.0 installed): the tree is
      **compliant with REUSE Spec 3.3** (258/258 files copyright+license). Added
      `LICENSES/MIT.txt`, a `REUSE.toml` (MIT/Elliot for authored files;
      third-party vendor PDFs correctly marked `LicenseRef-Proprietary-Reference`,
      *not* MIT), and `REUSE-Ignore` guards on the two SPDX-emitting scripts.

### 2.2 — The contract template ✅
- [x] `notes/module-contract-template.md` — the fixed shape (Interface /
      Clocking-reset / Parameters+ranges / Formal contract / Synthesis fit /
      Reuse notes). Uniform → auto-summarizable.

### 2.3 — Write the leaf contracts (most-reusable first) ✅ (exemplars)
- [x] Contracts written for the two flagship reusable blocks: `pwm_generator`
      (`rtl/contracts/pwm_generator.md` — self-contained, three PROVEN proofs)
      and `foc_core` (`rtl/contracts/foc_core.md` — composite, and where the
      Fmax≈3.3 MHz finding is stated honestly). Formal facts pulled from the
      manifest, fit from `synth/`. The remaining leaves follow the same template
      (their formal/sim/fit facts are already in the manifest + status matrix);
      writing the full set is mechanical follow-on.

### 2.4 — FuseSoC cores (make pull-in real) ✅
- [x] 25 leaf IP cores + the top `motorloop.core`, generated by
      `cores/gen_cores.py` from a dependency map (composites list their children;
      `sincos`/`foc_core` carry the gen include; only the system core depends on
      `rtl_params.vh`). All 26 validate as CAPI2 YAML.
- [x] **Standalone-elaboration acceptance test EXECUTED** (fusesoc 2.4.6
      installed): `fusesoc library add motorloop .` discovers all 26 cores;
      `fusesoc run --target lint motorloop:ip:pwm_generator` and `…:foc_core`
      (composite + the `sincos_init.vh` include) both elaborate + Verilator-lint
      **standalone, with no motorloop includes** — the reuse capability is real.
      *Known minor:* the cores live in `cores/` and reference `../rtl/…`, which
      emits a FuseSoC path-deprecation warning (works on 2.4.6; a future version
      will require the cores at the IP root — a packaging follow-on, not a
      capability gap).

### 2.5 — Status matrix ✅
- [x] `notes/status-matrix.md` (hand-authored, one row per module: proven /
      simulated / core / contract / synth note) + the generated live view
      (`notes/gen_status_matrix.py` → `status-matrix-generated.md`: 12 PROVEN /
      2 envelope / 1 DOCUMENTED + the Fmax).

### 2.6 — Verification plan ✅
- [x] `notes/verification-plan.md` — the requirement → proof/test traceability
      map (R1–R13), with the coverage gaps named (R6 documented-not-proven;
      timing; no silicon correlation).

## Stage 3 — Reproducible CI + releases

### 3.1 — Pin the toolchain ✅
- [x] `toolchain.lock`: the OSS CAD Suite pin (`2026-06-14`) bundles yosys +
      Verilator (5.049) + SymbiYosys + solvers + nextpnr-ecp5 + ecppack, so one
      tag pins the *entire* lint/sim/formal/synth stack. Python 3.12 (CI) / 3.14
      (local), cmake ≥3.22.

### 3.2 — The four CI gates ✅
- [x] `.github/workflows/ci.yml` adds **lint** (`verilator --lint-only`),
      **sim** (the platform + open-question + FOC-math tiers; full realism tier
      noted nightly), and **synth** (`run_synth.py --check`) — all from the one
      pinned OSS CAD Suite tarball, alongside the existing **formal** workflow.

### 3.3 — Auto-generated status matrix in CI ✅
- [x] `notes/gen_status_matrix.py` renders the live proof column from
      `results.json` + the Fmax from `synth_report.md`; the `ci` workflow
      regenerates it as an artifact. Flags any non-PROVEN/DOCUMENTED proof.

### 3.4 — Semver + release ✅ (CHANGELOG; tag is the maintainer's)
- [x] Semver adopted + `CHANGELOG.md` (the `[Unreleased]` trusted-library
      foundation entry, with the byte-identical guarantee and the named gaps).
      The actual `git tag v0.1.0` + asset upload is the maintainer's to make (I
      do not commit) — the release process is documented in the changelog header.

### 3.5 — Reproduce-from-scratch doc ✅
- [x] `notes/reproduce.md`: clone → pinned OSS CAD Suite → the four gates with
      expected numbers (401 sim / 12+1 formal / ECP5 fit + Fmax) + the
      `fusesoc run` reuse acceptance test — the reproducibility audit artifact.

## Findings

**Stages 1–3 landed 2026-06-14.** Motorloop is now a layered library: 25 leaf IP
cores pull standalone, the system composes them, and one pinned toolchain runs
all four gates.

What worked / what bit:

1. **The byte-identical invariant held all the way.** Parameters default to
   today's values; `controller_top` threads the generated config down. The
   201-test FOC-math + 8-platform checkpoint stayed green before/after every
   batch and after the threading; formal stayed 12 PROVEN + 1 DOCUMENTED.
2. **The real cost of parameterization was width-strictness, not the macros.**
   A `` `define `` constant is resized to its context by folding; a `parameter`
   is not, so a too-wide param truncates/expands its sized localparams (fatal in
   Verilator). The fix is a discipline: **size each count param to the datapath
   width it feeds** (`[15:0] PWM_HALF_PERIOD`, `[7:0] DRV_SPI_DIV`, `[3:0]
   LOCKOUT_N`), halve with `>> 1` not `/ 2`, and split a narrow initializer into
   a 32-bit compute + a sliced result (`DUTY_MAX`, `CR1_VALUE`). Only
   localparam/parameter *initializers* are fatal; procedural assignments aren't.
3. **Envelope proofs are the formal payoff.** `current_pi`/`speed_iq_pi` now take
   their clamp as the module's own parameter through the bind, so the bound is
   proven for *any* value — a leaf proven for anyone's config, not motorloop's.
4. **Honest scoping, not faked completeness:** the SV package is deferred (no
   shared typedef exists yet); contracts are written for the two flagship blocks
   with the template + facts in place for the rest; the `fusesoc`/`reuse-lint`
   acceptance tests are wired but unrun here (tools absent), like the synth gate.

**The two gaps these stages *expose* but do not close** (the library plan's
stages 4–5): timing (Fmax ≈ 3.3 MHz — pipeline `foc_core`/`circle_limit`) and
validation (no silicon-correlation tier). Both are stated plainly in every
artifact (status matrix, verification plan, contracts, CHANGELOG).
