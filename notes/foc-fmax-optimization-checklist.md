<!-- SPDX-License-Identifier: MIT -->
# FOC Fmax optimization checklist — push 41 MHz → ~100+ MHz

Ordered, comprehensive plan to raise the motorloop **system** Fmax past the
current 41.27 MHz by optimizing the FOC IP. Follows the pipelining that already
took `foc_core` from 3.3 → 41 MHz ([[motorloop-robotics-ip]], stage 6.5).

**Read this first — the honest framing.** None of this helps the control loop:
at 41 MHz the FOC loop already has ~15–70× headroom (the math runs once per
~10–50 kHz current sample). Every task here buys **integration margin** (sharing
a 100 MHz+ AXI/SoC clock without a clock-domain crossing) or **credibility**, not
loop bandwidth. So: **gate the whole effort on a concrete target clock**, do the
tasks **in order**, and **stop as soon as the system Fmax clears that target** —
the bottleneck moves after every task, so most runs will end early. Do not let
"a bigger number" become the product.

## Baseline (measured)

- System (`controller_top`) post-route **Fmax = 41.27 MHz** on LFE5U-85F.
- Critical path = **24.23 ns** (10.65 ns logic + **13.58 ns routing**), entirely
  in `speed_iq_pi` (`u_ctrl.u_pi`): the single-cycle 32-bit signed MAC
  `raw = KP*err + (KP*integ>>>KISH)` + the ±IQ_MAX clamp (`rtl/speed_iq_pi.v:36`).
- Note: routing is **56%** of the path at ~5% device utilisation → the **32-bit
  arithmetic is spreading cells**, so *narrowing* widths helps as much as adding
  pipeline registers. This is why Task 2 exists.

## Invariants that must never regress (check after EVERY task)

1. `python3 -m pytest sim/tests -q` → **400 passed / 2 skipped** (run with system
   python3; do *not* source oss-cad-suite — it shadows numpy). The FOC tier is
   tolerance-based and latency-aware, so added pipeline latency is fine.
2. `~/.local/share/cocotb-venv/bin/python -m pytest sim/cocotb/test_cocotb_blocks.py -q`
   → all green (foc_core + any new equivalence TBs).
3. `python3 formal/run_formal.py --check` → all manifest proofs still PROVEN
   (re-prove any leaf you touch — see per-task notes).
4. `python3 formal/check_coverage.py` → every core proven or sim-only.
5. `reuse lint`, `bash rtl/lint/run_verible.sh`, `python3 synth/asic_smoke.py
   --check` → clean.
6. `source ~/oss-cad-suite/environment && python3 synth/run_synth.py` →
   re-measure Fmax; the report's Finding auto-updates.
7. Update `rtl/contracts/foc_core.md` latency line + `notes/robotics-ip-checklist.md`
   Findings if the headline number changes.

---

## Task 0 — Measurement harness + free retiming (enabling; do first)

**Goal.** Make each later task's gain *measured*, not predicted, and grab any
free Fmax from the existing registers before adding new ones.

**0a. Per-module Fmax harness.** New `synth/fmax_module.py <module>`:
- Generate a tiny ring wrapper (`synth/work/ring_<module>.v`) that registers all
  DUT inputs and outputs from one clock (so timed paths are internal reg→reg,
  not I/O-bound). For already-registered modules (`speed_iq_pi`, `foc_core`,
  `circle_limit_seq`) the wrapper just needs input FFs.
- Drive yosys `synth_ecp5` + `nextpnr-ecp5 --85k --freq 200 --report` on the
  wrapper; parse `Max frequency`. Reuse the parsing in `synth/run_synth.py`.
- Print a one-line `module, Fmax, LUT4, MULT18X18D`. This is the instrument used
  to confirm each task and to **find the next bottleneck** (read `nextpnr.log`
  critical path; grep the dominating `u_*` instance, as in the baseline above).

**0b. Register retiming (free, no RTL).** In `synth/synth_ecp5.ys` add `-retime`
to `synth_ecp5` (and/or `abc` with `&dch`/retiming). Re-run `run_synth.py`.
- *Predicted:* 41 → ~45–52 MHz (rebalances the existing FFs across the 32-bit
  MAC; helps the routing-heavy path).
- *Risk:* none to behaviour (retiming is equivalence-preserving); confirm the
  full regression + formal still pass (retiming can rename nets the proofs don't
  depend on). If yosys retiming destabilises a proof, scope it to the synth flow
  only (proofs read the un-retimed RTL anyway).

**Done-when:** `fmax_module.py` reports a number for every FOC block; retiming
either banked or shown to not help (record which in the synth report).

---

## Task 1 — Pipeline `speed_iq_pi` (the current bottleneck; biggest single win)

**Goal.** Remove the 24.23 ns outer-PI path. The outer loop updates once per
*speed* sample (rarer than the current loop), so latency is free — exactly the
`foc_core` sequencer trick.

**Files.** `rtl/speed_iq_pi.v`, `formal/bind/speed_iq_pi_fv.sv`,
`formal/manifest.toml` (depth/notes), maybe a new `sim/cocotb/tb_speed_iq_pi.py`.

**Code changes.**
- Add a small sequencer (`S_IDLE→S_ERR→S_MAC→S_FIN`, `reg [1:0] state`). On
  `update` (and `enable`): latch `target_speed`, `speed`, `reverse`.
- `S_ERR`: register `err_r <= target - measured` (the only sign-extend/subtract).
- `S_MAC`: register `raw_r <= KP*err_r + ((KP*integ)>>>KISH)`; register
  `sat_hi_r`, `sat_lo_r` from `raw_r` (compare in this stage).
- `S_FIN`: the existing integrator update (conditional integration with
  `err_r`/`sat_*_r`) and `iq_target <= clamp(raw_r)`. Integrator advances **once
  per update**, same `err` and sat as today → **bit-identical integ evolution**;
  `iq_target` just appears ~3 cycles later.
- Keep `enable`-low reset of `integ`/`iq_target` and the async reset exactly.

**Verification.**
- Re-prove: the clamp property (`|iq_target| ≤ IQ_MAX`) on the registered
  version — update `speed_iq_pi_fv.sv` to assert at the output reg, bump `depth`
  to cover the walk (~6). It is in the manifest as PROVEN; keep it PROVEN.
- Add `sim/cocotb/tb_speed_iq_pi.py` (latency-aware): drive a speed error, wait
  for `iq_target` to settle, check sign (braking on overspeed) + clamp.
- Full regression (mode-3 closed loop exercises it) + re-synth.

**Predicted:** 41/52 → **~55–70 MHz** (next bottleneck becomes the inner
`current_pi` MAC or `circle_limit_seq` `mag2`). *Risk:* low — behaviour-preserving,
sparse update; mirrors the proven `foc_core` change.

---

## Task 2 — Narrow the PI integer widths (attacks the routing half)

**Goal.** The PIs carry **32-bit** `integ`/`err`/`raw` though the real range is
~16–18 bits (`IQ_MAX=300`, `V_RAW_MAX=2500`, conditional integration + clamp bound
the integrator). Narrowing shrinks the carry chains *and* the cell spread that
caused 56% routing.

**Files.** `rtl/speed_iq_pi.v`, `rtl/current_pi.v`, their `*_fv.sv`, manifest.

**Code changes.**
- Analyse the worst-case `integ` range under conditional integration + clamp
  (write the bound in the header). Pick a width with margin (e.g. `integ`
  `[23:0]` or `[19:0]`); keep products in wide temporaries, slice explicitly to
  avoid `WIDTHTRUNC`. Same for `err`/`raw`.
- Do `current_pi` and `speed_iq_pi` together (same arithmetic shape).

**Verification.**
- **This is the one task that is NOT guaranteed bit-exact** — narrowing changes
  overflow behaviour at extremes. Re-prove both PIs (they are PROVEN,
  parameter-generic — the clamp must still hold at the chosen width) and run the
  full regression; if any FOC scenario shifts, treat it as a real (if tiny)
  behaviour change and **get sign-off before banking it**. Keep generous margin.
- *Risk:* medium (behaviour + two proofs). Only bank if `fmax_module.py` shows a
  worthwhile gain over Task 1 alone.

**Predicted:** **~5–15 MHz** on top of Task 1, mostly by relieving routing.

---

## Task 3 — Pipeline the inner current PIs + split `circle_limit_seq` mag2

**Goal.** After Tasks 1–2, the next arcs are inside `foc_core`: the d/q
`current_pi` MAC (the `S_PI` stage) and `circle_limit_seq`'s `S_MAG`
(`mag2 = d*d+q*q` + the `>VLIM²` compare).

**Files.** `rtl/foc_core.v`, `rtl/circle_limit_seq.v` (and its cocotb equivalence
test stays valid), `rtl/current_pi.v` + `current_pi_fv.sv` if pipelined.

**Code changes.**
- *current_pi:* prefer the Task-2 width narrowing first (cheaper). If still
  limiting, split its MAC by registering `praw` inside the `foc_core` sequencer:
  add an `S_PI2` state so the PI eval spans two cycles. **Care:** the
  `freeze`(sat) feedback must stay consistent — the integrator must not move
  between capturing `vd_raw_r` and the `pi_update` pulse (today's `S_PI →
  S_LIMSTRT → S_LIMWAIT` already holds it; extend the same hold across `S_PI2`).
- *circle_limit_seq:* split `S_MAG` into `S_MUL` (register `mag2`) and `S_CMP`
  (register `sat`, init isqrt). Trivial — already a sparse sequencer; latency
  +1 cycle, still ≪ sample period. Its cocotb equivalence test
  (`tb_circle_limit_seq`) is unaffected (it gates on `done`).

**Verification.** cocotb equivalence test still green (it re-confirms bit-exact
vs combinational `circle_limit`); re-prove `current_pi` if pipelined; full
regression; re-synth. Update `foc_core` contract latency (~10 → ~12–14 clocks).

**Predicted:** **~75–95 MHz** cumulative. *Risk:* medium (the freeze-loop timing
in `foc_core` is the subtle part — preserve the "integ stable until pi_update"
invariant exactly).

---

## Task 4 — Pipelined multiply leaves (`park`/`inv_park`/`svpwm`)

**Goal.** Past ~90 MHz the residual depth is the multiply-add leaves:
`park`/`inv_park` (`(a*c ± b*s)>>>15`, 2 mults + add) and `svpwm`
(`(SQRT3/2)*vbeta` mult + the 3-way min/max tree + clamp). Register between the
MULT18X18D and the add/compare.

**Approach — DO NOT edit the leaves in place.** `test_foc_math` checks
`park`/`inv_park`/`svpwm` **bit-exactly through the *combinational* harness
`rtl/foc_math.v`** ("every assertion exact"), and `svpwm` is **formally PROVEN**.
Editing the leaves would break both. Instead **mirror the `circle_limit_seq`
pattern**:
- New pipelined variants `rtl/park_seq.v`, `rtl/inv_park_seq.v`,
  `rtl/svpwm_seq.v` (each: register the products, then the add/min-max; a fixed
  1–2 cycle latency, `start`/`done` or just `valid`).
- `foc_core` instantiates the `_seq` variants; **`foc_math.v` keeps the
  combinational originals**, so `test_foc_math` and the `svpwm` proof are
  untouched.
- Verify each `_seq` **bit-exact to its combinational original** with a cocotb
  equivalence harness (clone `sim/cocotb/eq_circle_limit.v` +
  `tb_circle_limit_seq.py` → `eq_park.v`/`tb_park_seq.py`, etc.). Add them to
  `sim/cocotb/test_cocotb_blocks.py`.
- Generate `.core`s (`cores/gen_cores.py` `LEAVES`), add to `synth_ecp5.ys` and
  `Bender.yml` (regenerate), declare the `_seq` variants in
  `formal/sim_only.toml` with the "bit-exact to the proven combinational leaf via
  cocotb equivalence" reason (precedent: `circle_limit_seq`).

**Code changes (foc_core).** Expand the sequencer: `S_PARK` → mult/add substages;
`S_INVP` and `S_SVPWM` likewise (wait on each `_seq` `done`, or fixed latency).
Keep the integrator-freeze invariant intact.

**Verification.** New equivalence TBs green; full regression; coverage gate
(3 new cores); `asic_smoke.py` (3 new blocks); reuse/verible; re-synth.

**Predicted:** **~100–130 MHz** cumulative, approaching the floor. *Risk:* high
(most files, most new verification) — only do it if a target clock above ~90 MHz
demands it.

---

## Practical ceiling & stop conditions

- **Ceiling on the -85F (~ -6 speed grade): ~120–150 MHz** for this fixed-point
  datapath — set by MULT18X18D delay (~3–4 ns) + register setup/clk-to-q + the
  global clock network + routing. The ECP5 fabric absolute (~200+ MHz) is not
  reachable for a multiply+add datapath. A faster speed grade lifts this.
- **Stop the moment `run_synth.py` clears your target clock.** Expected order of
  exit: most SoC integrations need ≤ 100 MHz → Tasks 0–1 (±2) usually suffice.
- After each task: re-read `nextpnr.log` critical path to confirm the bottleneck
  actually moved to the block the next task addresses; if it moved **outside the
  FOC** (e.g. `drv_manager`, `adc_sequencer`, UART), stop — further FOC work
  won't raise the *system* Fmax, and that's a different block's checklist.

## Ordering summary

0 (harness+retime) → 1 (speed_iq_pi pipeline) → **measure** → 2 (narrow widths,
sign-off) → **measure** → 3 (inner PI + mag2) → **measure** → 4 (multiply `_seq`
leaves). Tasks 1, 3, 4 are behaviour-preserving (bit-exact / latency-only); Task 2
is the only one that can shift behaviour and needs explicit sign-off.

## Predicted cumulative Fmax

| After task | Predicted system Fmax | Confidence |
| --- | --- | --- |
| 0 (retime) | ~45–52 MHz | med (placement-dependent) |
| 1 (speed_iq_pi) | ~55–70 MHz | med-high |
| 2 (narrow) | ~60–80 MHz | med |
| 3 (inner PI + mag2) | ~75–95 MHz | med |
| 4 (multiply leaves) | ~100–130 MHz | low-med (near floor) |

All numbers are structure-based predictions; `fmax_module.py` + `run_synth.py`
turn each into a measurement. Confirm, don't assume.

## Measured results (executed)

Scope: per the "implement only the tangible-gain parts" directive, the harness
(task 0) was used to do exactly the binding work and skip the no-ops. **System
Fmax: 41.27 → 64.1 MHz (+56%), timing@25 MHz met, all 400 tests green, all 12
formal proofs PROVEN.** Every RTL change is behaviour-preserving (bit-exact /
latency-only); none needed a gain trade-off.

Per-block standalone Fmax (`synth/fmax_module.py`, registered ring):

| Block | Before | After | Change made |
| --- | --- | --- | --- |
| `speed_iq_pi` | 75 | **129** | pipelined the MAC+clamp (task 1; re-proven PROVEN) |
| `svpwm_seq` | — | **95** | new bit-exact sequential SVPWM (task 4; was foc_core's cap) |
| `circle_limit_seq` | 64 | **82** | mag2 + division operand from 18-bit inputs (18×18 hard mults) + S_MUL/S_CMP split (task 3) |
| `foc_core` | 42 | **79** | the above two, integrated |
| `speed_pi` | 48 | **94** | pipelined MAC+clamp+down-slew (surfaced as the *system* cap once FOC was fast) |
| `current_pi` | 100 | 100 | untouched — never binding |
| **system** | **41.3** | **64.1** | — |

What the data changed vs the plan:
- **Task 0b (retiming): no help.** `synth_ecp5 -noabc9 -retime` measured 33 MHz
  (loses more in abc9 mapping than retiming gains); reverted, abc9 default kept.
- **Task 2 (narrow widths): SKIPPED.** The harness showed `current_pi`=100 and
  `speed_iq_pi`=129 — neither binds — so the one non-bit-exact change earned no
  system gain and was not worth its risk (its own gate said so).
- **Task 3 (current_pi pipeline): SKIPPED** (100 MHz, never binding). Only the
  `mag2` split + the 18-bit-operand trick were done (those *were* binding).
- **Task 4: svpwm only.** `svpwm_seq` was the foc_core cap; once fixed the cap
  moved to `circle_limit_seq` then out of foc_core. `park`/`inv_park` were never
  the binding leaf, so no `park_seq`/`inv_park_seq` were built (their 32×32 style
  could be fixed in place bit-exactly *if* they ever bind — they don't here).
- **`speed_pi` (not originally listed):** once the FOC datapath was fast, the
  system cap was the six-step duty PI (47.6 MHz). Same proven pipeline pattern,
  sim-only, so it was the single most tangible remaining system gain → done.

**Stop point (honest).** At 64.1 MHz the critical path is `controller_top`'s
sensored-sector glue (`angle*POLE_PAIRS`, `*6`, the hysteresis FSM,
`controller_top.v:211`) — top-level, continuously-clocked, behaviour-sensitive
interconnect, **not a FOC datapath leaf and not a clean pipeline target**. All
FOC blocks are now fast (foc_core 79, speed_iq_pi 129, circle_limit_seq 82,
svpwm_seq 95). Per the stop rule above, further system Fmax requires reworking
top-level commutation glue — a different block's concern, outside this checklist.
64.1 MHz is >> the FOC loop need (15–70× headroom) and clears a 50 MHz SoC clock.
