<!-- SPDX-License-Identifier: MIT -->
# HDL performance audit — findings + fix checklist

A data-driven Fmax/area audit of the RTL, grounded in the **measured** post-route
critical path (`synth/work/nextpnr.log`), not speculation. Each finding lists its
impact, the evidence, and ordered steps to fix it.

## Framing (so priorities stay honest)

The design **already meets timing at the 25 MHz board clock with 2.5x margin**
(post-route Fmax **64.1 MHz**, `synth/synth_report.md`). Nothing here is *broken* —
these are **headroom** improvements, valuable for two concrete reasons:

1. **Single-clock SoC/AXI integration** — a ~80–100 MHz core lets the LiteX SoC
   run one clock domain (no CDC), the long-stated goal.
2. **Multi-vendor margin** — Gowin/Intel/Xilinx paths run slower than the ECP5,
   so ECP5 headroom is portability insurance.

**One module gates everything.** The FOC datapath is already pipelined to ≥79 MHz
standalone (`foc_core` 79, `speed_iq_pi` 129, `speed_pi` 94, `current_pi` 100,
`svpwm_seq` 95 — `synth/fmax_module.py`). The system sits at 64 MHz purely because
of glue in `controller_top.v`. Fix that and the whole design moves.

## The measured bottleneck (shared evidence)

Post-route critical path (`synth/work/nextpnr.log`, clock `clk`, 64.10 MHz, 7.49 ns
logic + 8.11 ns routing = 15.6 ns):

```
dbg_angle (FF)  ->  elec12 carry-add (controller_top.v:211)
                ->  sector_cand MULT18X18D  = 3.93 ns  <-- largest single term
                ->  pos_in_sector / sensored_sector hysteresis (lines 214,229)
                ->  sensored_sector (FF)
```

The whole chain is the **sensored-sector glue** (`controller_top.v:206–237`):
mechanical angle → electrical angle → sector + hysteresis, combinational from the
`angle` register to the `sensored_sector` register in one clock.

---

## F1 — A constant ×6 was mapped to a DSP, and it is *the* critical-path term
*(easy · bit-exact · high value)*

- **Finding:** `sector_scaled = {20'd0, elec12} * 32'd6` (`controller_top.v:214`)
  infers a `MULT18X18D`. `×6 = (x<<2)+(x<<1)` — a one-level add, no multiplier
  needed. (yosys strength-reduced the `* POLE_PAIRS` ×4 to a shift one line up,
  but grabbed a DSP for the ×6.)
- **Impact:** removes the **3.93 ns** DSP delay *and* its routing detour (DSPs sit
  in fixed columns — several ns of the path's routing is the hop to/from the
  multiplier). Frees one `MULT18X18D` (22→21). Expected **~64 → ~75–80 MHz** alone.
- **Evidence:** the critical path names `u_ctrl.sector_cand_MULT18X18D_P13` (3.93 ns,
  `nextpnr.log`); `synth_report.md` shows `MULT18X18D: 22`.
- **Steps:**
  - [ ] Replace line 214–216 with a narrow shift-add (`elec12` is 12-bit, so
        `×6 ≤ 15 bits`):
        ```verilog
        wire [14:0] sector_scaled = ({3'd0, elec12} << 2) + ({3'd0, elec12} << 1);
        wire [2:0]  sector_cand   = sector_scaled[14:12];
        wire [11:0] pos_in_sector = sector_scaled[11:0];
        ```
  - [ ] Leave `HYST_SCALED = SECTOR_HYST * 6` (line 217) — it's a `localparam`,
        constant-folded, not a multiplier.
  - [ ] `make test` — must stay **byte-identical** (algebraically the same).
  - [ ] `python3 synth/fmax_module.py` on `controller_top` to record the delta;
        `make synth` for the authoritative post-route Fmax.

## F2 — Pipeline the sensored-sector glue to lock in the gain
*(medium effort · behaviour-sensitive · high value)*

- **Finding:** even with the DSP gone, `controller_top.v:206–237` is one long
  combinational chain (angle reg → `elec_raw` add → `elec12` → `sector_scaled` →
  hysteresis compares → `sensored_sector` reg).
- **Impact:** a pipeline register splits the chain and lets the placer break the
  long route; targets the **`foc_core` ~79 MHz cap (~+23%)**. The sector feeds
  commutation, which already tolerates sensor latency, so +1 clock (40 ns @
  25 MHz) is functionally invisible.
- **Evidence:** the path's 8.11 ns routing > 7.49 ns logic — it sprawls across the
  die (`(82,20)→(73,16)→(73,10)→(71,15)` in `nextpnr.log`); registers break it.
- **Steps:**
  - [ ] Register `sector_scaled` (one stage); read `sector_cand`/`pos_in_sector`
        from the registered copy:
        ```verilog
        reg [14:0] sector_scaled_q;
        always @(posedge clk or negedge rst_n)
          if (!rst_n) sector_scaled_q <= 15'd0;
          else        sector_scaled_q <= sector_scaled;
        ```
  - [ ] Run the full suite, **especially six-step (mode 2) + FOC scenario tests** —
        commutation timing shifts by a cycle; confirm the tolerance-based tests
        still pass.
  - [ ] **Re-prove the formal set.** Add the new register to the
        `controller_top_composition` manifest `rtl` list (the known gotcha: a new
        register/submodule left out blackboxes and the safety proof fails).
  - [ ] `make synth` + `make portability` — confirm the new cap and no
        cross-family regression.

## F3 — The FOC angle-extrapolation ×1001 multiply is the *next* bottleneck
*(medium · do after F1/F2)*

- **Finding:** `extrap_counts = (speed_signed_for_extrap * \`EXTRAP_NUM) >>> \`EXTRAP_SH`
  with `EXTRAP_NUM = 1001` (a genuine non-power-of-2 → a DSP) feeds
  **combinationally** into `elec12_foc → theta_e16 → foc_core`
  (`controller_top.v:343–348`).
- **Impact:** once F1/F2 clear the sector path, this multiply-into-FOC-angle is the
  likely new mode-3 critical path. Registering the extrapolated angle keeps the
  controller glue from re-capping the design below `foc_core`.
- **Evidence:** `rtl/gen/rtl_params.vh:45` `EXTRAP_NUM 1001`; the
  `speed_signed_for_extrap` cone already appears in the cross-domain path report
  (`nextpnr.log` line ~311).
- **Steps:**
  - [ ] Register `theta_e16` (or `extrap_counts`) — the extrapolated angle changes
        slowly, so a pipeline cycle is harmless; mirror the F2 latency note.
  - [ ] Validate FOC mode-3 scenario tests (the extrapolation path) stay green;
        re-prove if a new register enters a proven module.
  - [ ] Re-measure Fmax; confirm the cap is now `foc_core`, not controller glue.

## F4 — `test_pin` OR-cone pollutes the timing graph
*(low · synthesis-wrapper only)*

- **Finding:** the worst *routing* path (12.86 ns) is an async cone: a static
  config strap (`angle_spi_mode`) → telemetry logic → `test_pin`, built by
  `synth/board_top.v:110` as an OR-reduction of internal nets "to keep them
  observable." It is a functional **false path** (static input) and lives in the
  **board wrapper, not the core IP**.
- **Impact:** doesn't gate the clocked Fmax, but clutters the timing report and
  can distort placement. Low priority.
- **Evidence:** `nextpnr.log` cross-domain path `angle_spi_mode$tr_io → … →
  test_pin` (1.65 ns logic + 12.86 ns routing).
- **Steps:**
  - [ ] Either register `test_pin`, or add an SDC/constraint marking it a false
        path, so observability doesn't pollute timing.
  - [ ] Confirm the telemetry nets stay un-trimmed (the OR-cone's original purpose).

## F5 — Area / throughput are NOT constraints here
*(context · no action on this board)*

- **Finding:** real post-route utilization is ~4272 LUT4 / 1630 carry / 22 DSP on
  the LFE5U-85F — **~5%**. The portability table's 13k–25k LUTs are pre-P&R generic
  estimates, not real.
- **Impact:** no area work is warranted on this board. The F1 DSP saving matters
  for *timing*, not area (though it helps tighter targets like the Tang Primer 25K).
  The FOC sequentialization (one op/clock) is the correct Fmax/latency trade at
  25 MHz — latency is sub-sample-period.
- **Evidence:** `synth/synth_report.md` (TRELLIS_FF 3045, LUT4 4272, CCU2C 1630,
  MULT18X18D 22) vs `synth/portability_report.md` estimates.
- **Steps:**
  - [ ] None now. Revisit only if targeting a small/cheap FPGA where ~4k LUTs +
        22 DSPs is tight; then F1 (frees a DSP) is the first lever.

---

## Verification recipe (every change)

- [ ] **Bit-exact / cycle-accurate:** `make test` stays byte-identical for F1
      (algebraic); for F2/F3 the duties/sector appear a fixed latency later —
      confirm the tolerance-based scenario tests pass.
- [ ] **Formal:** re-run the proof set; **add any new register to the
      `controller_top_composition` manifest** (blackbox gotcha).
- [ ] **cocotb equivalence** for any block touched.
- [ ] **Synthesis:** `make synth` (authoritative post-route Fmax) + `make
      portability` (no cross-family regression) + `synth/fmax_module.py` for the
      per-module delta.

## Done-when

`controller_top` is no longer the system Fmax cap: post-route Fmax reaches the
`foc_core` ~79 MHz tier (F1+F2, +~23%), the suite + formal + cocotb stay green and
bit-exact, and `synth_report.md` records the new number with the DSP count down by
one (F1).

## What NOT to do

- Don't chase Fmax past the actual need — at 25 MHz the design already passes;
  stop at the `foc_core` tier unless the SoC clock demands more.
- Don't break bit-exactness or a formal proof for marginal MHz.
- Don't fork the RTL per vendor; keep one source byte-identical to the sim.
- Don't pipeline `foc_core` internals further for small gains (documented
  diminishing-returns stop point).

## Implemented (results)
_(to fill in after execution)_
