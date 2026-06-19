<!-- SPDX-License-Identifier: MIT -->
# Open ADS9224R simultaneous current-sense module — design & ship checklist

**The gap (researched):** there is no existing open-source, already-spun ADS9224R
breakout/current-sense board (no KiCad/Gerber/Altium project on the usual GitHub /
EDA channels). The only reference is TI's **ADS9224REVM-PDK** — an evaluation kit
with a complete user guide (schematic, BOM, layout figures, circuit descriptions)
but **no downloadable open EDA source**.

**North star:** spin the missing artifact — a small **open** ADS9224R dual-
simultaneous current-sampling module: KiCad source + Gerbers + BOM + HDL driver +
cocotb tests + **analog provenance**, designed off TI's EVM topology, integrated
with motorloop (it already ships the [`ads9224r_master.v`](../rtl/ads9224r_master.v)
driver, the C++ model, and the part-comparison study that quantifies *why*
simultaneous sampling matters — Q21). This closes motorloop's loop from silicon →
sim → **publishable open hardware**.

## The honest boundary (built into every phase)

This repo's discipline is provenance + "don't claim what you didn't measure"
(vendor-Fmax-where-licensed, litex_sim-headless caveat). The same applies here:
the deliverable is a **designed, simulated, ERC/DRC-clean, orderable** module —
**not fabbed or bench-validated in this environment**. Every electrical value
carries a provenance tag (`datasheet` / `ti-evm-baseline` / `assumed`), exactly
like `sim/config/params.toml`. Physical bring-up is the maintainer/lab step (§10),
clearly marked, never faked.

**License note:** hardware needs an **open-hardware license** (recommend
**CERN-OHL-S**), distinct from the MIT software. Add it to `REUSE.toml` up front.

---

## §1 — Research & golden-reference capture

- [ ] Acquire + archive (gitignored vendor PDFs in `docs/`, the existing pattern;
      record URLs in `notes/` or a `reference` memory): the **ADS9224REVM-PDK user
      guide** (SBAU-series), the **ADS9224R datasheet**, the **THS4551** FDA
      datasheet, the chosen **voltage-reference** datasheet, and (if used) the
      **current-sense-amplifier** datasheet. Cover them in `docs.toml`/REUSE as
      proprietary-reference.
- [ ] Extract the EVM signal chain per channel and record the topology + values:
      shunt/source → **THS4551 fully-differential driver** (with gain) → **RC
      charge-bucket** (R_flt/C_flt) → ADS9224R AIN± ; the **reference** path
      (low-noise series ref + buffer + reservoir cap — the SAR draws charge from
      REF every conversion); supplies/decoupling (AVDD/DVDD); the digital
      interface (CONVST / CS / SCLK / SDO_A / SDO_B / READY).
      *Why the FDA: the SAR cap-DAC is a dynamic load at 3 MSPS — it needs a
      low-impedance, low-distortion driver to settle within the acquisition
      window (TI's stated reason).*
- [ ] **Decision — current-sense front-end** (recommend + record rationale):
  - **Recommended:** low-side shunt → **direct-differential into the THS4551**
    (with gain), matching motorloop's low-side-shunt-sampled-at-PWM-peak topology
    (common mode ≈ ground at the sample instant — no high-CM CSA needed).
  - **Option B:** an **INA240-class CSA** ahead of the FDA for inline/high-side
    shunts (high common-mode rejection). Provide as a populate-option.
- [ ] **Decision — input scaling options:** a gain-resistor table mapping
      (shunt value × FDA gain → full-scale current) so users scale for their
      motor; pick the default to match motorloop's `feedback.current` shunt
      (2 mΩ, `circuit.iout_channel`).
- [ ] **Digital-rail decision:** set DVDD/IO to **3.3 V** to mate the FPGA
      directly (ULX3S is 3.3 V, *not* 5 V-tolerant — see the BOM research); confirm
      ADS9224R IO range covers 3.3 V so **no level translator** is needed.
- [ ] Reconcile with the existing co-sim contract so the board and the model
      agree: list the parameters that MUST match across
      [`ads9224r.cpp`](../sim/cpp/src/ads9224r.cpp),
      [`ads9224r_master.v`](../rtl/ads9224r_master.v) and the board — `vref`,
      codes/A scaling, `ADC_SPI_DIV`/SCLK, CONVST→READY conversion time, the
      CONVST lead before the PWM peak, two's-complement coding.

## §2 — Electrical design as provenance-tracked params (the one-source layer)

- [ ] Add `[circuit.ads9224r_module.*]` tables to `sim/config/params.toml`: FDA
      gain resistors (Rf/Rg), RC charge-bucket (R_flt/C_flt), reference value +
      reservoir cap, decoupling network, shunt option(s), and the derived
      full-scale current per option — each with `status` + `source` + `blocked_by`
      (mirror `[circuit.iout_channel]`).
- [ ] Encode the scaling conversion (shunt × FDA gain × vref × 2^16 → codes/A) in
      the **derivation unit tests** (the repo unit-tests param conversions), and
      assert it equals the FOC fixed-point expectation after `cur_norm_shift`
      renormalization (so the board, the RTL, and the model carry one number).
- [ ] Add the analog **SPICE behavioral model**
      `sim/circuits/ads9224r_frontend.cir` (THS4551 FDA + RC bucket + the ADC
      sample-cap network) — mirror `sim/circuits/iout_channel.cir`. Run an ngspice
      transient: **the input must settle to < 0.5 LSB within the acquisition
      aperture** at the target sample rate. This is the analog-correctness proof
      (and the §8 settling figure).

## §3 — KiCad schematic (generated where possible)

- [ ] New KiCad project `hw/ads9224r-module/`. Extend
      [`gen_kicad_sch.py`](../sim/scripts/gen_kicad_sch.py) (or a sibling
      generator) to render the **passive** network (FDA gain net, RC bucket,
      dividers, decoupling) from `[circuit.ads9224r_module.*]` — the tables stay
      the primary source; the schematic is the generated, reviewable mirror.
- [ ] Create/import KiCad **symbols + footprints** for the active parts not in the
      stock libs: ADS9224R (its datasheet package), THS4551, the reference IC, the
      optional CSA. Pin-map against the datasheets.
- [ ] Define the **FPGA header**: connector + pinout for
      `convst / ncs / sclk / sdo_a / sdo_b / ready` + 3.3 V + grounds, targeting a
      PMOD-style / ULX3S-compatible header; document the mapping to the
      `ads9224r_master.v` ports.
- [ ] **ERC clean**; extend the existing `gen_kicad_sch.py --check` round-trip
      (schematic → SPICE/netlist via `kicad-cli`, already installed at
      `/usr/bin/kicad-cli`) to verify the module's component values + connectivity
      survive the round trip. Make it a gated check.

## §4 — PCB layout

- [ ] Stackup decision: **4-layer** (signal / GND / PWR / signal) for a clean SAR
      ADC. Floorplan per the EVM discipline: FDAs hard against the ADC inputs; the
      reference IC + reservoir cap against the REF pin; solid ground; **Kelvin**
      shunt sensing; length-matched **differential** AIN± routing; keep SCLK and
      digital edges away from the analog front-end.
- [ ] **DRC clean**; differential-pair impedance/length match; decoupling +
      thermal review.
- [ ] Generate manufacturing outputs via `kicad-cli`: Gerbers, drill,
      pick-and-place, fab/assembly drawings, and a 2D/3D **board render** (PNG).

## §5 — BOM & sourcing

- [ ] Structured BOM (`hw/ads9224r-module/bom.csv`): MPNs + DigiKey/Mouser part
      numbers + provenance, reusing the deep-research sourcing format (it already
      verified ADS9224RIRHBR, etc.).
- [ ] A **build-options** table (shunt × FDA gain → full-scale current) so users
      pick a scaling for their motor; flag the populate-options (CSA vs direct).

## §6 — HDL driver alignment + verification

- [ ] Audit [`ads9224r_master.v`](../rtl/ads9224r_master.v) against the **final
      board**: pin names/polarity, SCLK divider vs the board's max SCLK,
      CONVST→READY vs the real conversion time, dual-lane SDO_A/SDO_B framing,
      two's-complement. Update the `ADC_SPI_DIV` / `ADC_EMF_LEAD` params to match.
- [ ] Update the contract
      [`rtl/contracts/ads9224r_master.md`](../rtl/contracts/ads9224r_master.md):
      reference the open board as the physical target; record interface + timing +
      scaling + validation status.
- [ ] Align the C++ model [`ads9224r.cpp`](../sim/cpp/src/ads9224r.cpp) to the
      board's timing/scaling if they differ; keep it byte-identical against the
      RTL where the equivalence is checked.

## §7 — New tests

- [ ] **Block-level cocotb** for `ads9224r_master` against a behavioral ADS9224R
      bus model with the **real datasheet timing** (CONVST pulse, conversion time,
      dual-SDO MSB-first, two's-complement) — add to `sim/cocotb/`. Verify it
      against the C++ model timing (equivalence).
- [ ] **Derivation unit test:** board scaling (shunt × gain × vref → codes/A)
      matches the FOC `cur_norm_shift` expectation (§2).
- [ ] **ngspice front-end settling** test (acquisition-window settle < 0.5 LSB) as
      an automated check, like the existing `.cir` checks.
- [ ] **Integration:** confirm the `zonri_ads9224r` / `ti_reference_hp` platform
      profiles consume the new board params consistently; re-run the
      **part-comparison study** (T3/T4 skew) to confirm the board's simultaneity
      retires Q21 exactly as modeled — the board's *raison d'être*, quantified.
- [ ] `make test` stays green; `reuse lint` clean (incl. the new OHL license).

## §8 — New graphics

- [ ] **Signal-chain block diagram** (shunt → FDA → RC bucket → ADS9224R AIN±,
      reference, FPGA header) + the schematic PDF (`kicad-cli`).
- [ ] **PCB 2D/3D render** (`kicad-cli`).
- [ ] **Simultaneity timing diagram** — CONVST → both channels sampled at the same
      instant (WaveDrom or matplotlib); pair it with the part-comparison
      **T3/T4 skew** figures so the board visibly delivers the Q21 retirement.
- [ ] **Front-end settling plot** (the §2 ngspice transient: input settling inside
      the aperture) — the analog-provenance graphic.
- [ ] Drop these into `figures/` + a `hw/ads9224r-module/` gallery page (mirror
      `figures/comparison/gallery.md`).

## §9 — Packaging, provenance, publication

- [ ] **REUSE/SPDX** on every new source (KiCad files, params additions, scripts,
      `.cir`, BOM); add **CERN-OHL-S** to `REUSE.toml` for the `hw/ads9224r-module/`
      tree; cover the vendor PDFs as proprietary-reference.
- [ ] `hw/ads9224r-module/README.md`: what it is, the EVM-derived provenance, the
      build/scaling options, the FPGA wiring, the HDL-driver link, the **"designed
      + simulated, not yet fabbed/validated"** banner, and ordering instructions
      (Gerbers + BOM).
- [ ] A board **contract/datasheet** (mirror `rtl/contracts/`): interface, scaling,
      provenance, validation status.
- [ ] Wire into the docs site + add to `CITATION.cff` / `.zenodo.json` as a
      **citable open-hardware artifact** ("the first open ADS9224R simultaneous
      current-sense module: KiCad + HDL + cocotb + provenance").

## §10 — Hardware bring-up (MAINTAINER / lab — the honest boundary)

- [ ] Fab + assemble (a board house from the §4 outputs).
- [ ] Bench bring-up: power/reference rails, ADC self-test, **SNR/ENOB at 3 MSPS**,
      and the key claim — **measure the inter-channel skew** (common signal into
      both channels) to confirm simultaneity.
- [ ] Close the loop on a motor; compare to the sim's **Q21 prediction** — at which
      point the part-comparison study becomes a *hardware-validated* result, and
      every `assumed/ti-evm-baseline` value can be promoted to `measured`.

## Done-when

`hw/ads9224r-module/` holds a complete, ERC/DRC-clean, **open** ADS9224R
current-sense module (generated-from-params schematic, layout, Gerbers, BOM) with
analog provenance; the SPICE front-end proves acquisition-window settling; the HDL
driver + cocotb + C++ model agree and re-confirm the Q21 simultaneity win; the
graphics + README + contract + license are published and citable. Hardware bring-up
(§10) is documented as the maintainer step, not faked.

## What NOT to do

- Don't claim a fabricated/validated board — it's designed + simulated here; mark
  §10 as future and tag every value by provenance.
- Don't fork the analog values from the params source — generate the schematic from
  `[circuit.ads9224r_module.*]`, keep one provenance-tracked source.
- Don't license the hardware MIT — use an OHL (CERN-OHL-S); keep the software MIT.
- Don't skip the FDA/reference discipline — a SAR ADC with a weak driver or noisy
  reference loses the ENOB that justifies the part (and the Q21 win).
- Don't let the board and the co-sim model drift — the §6/§7 equivalence + the
  part-comparison re-run are the guardrails.

## Implemented (results)

Design + simulation + provenance + packaging landed (§1–§9); the routed PCB +
fabrication + bench bring-up are the documented interactive/maintainer step
(§4 layout / §10), honestly out of scope for this environment. **All new tests
green; REUSE compliant (372/372); 21 derived params consistent.**

- **§1 research / honesty:** new open question **Q23** (board bring-up);
  `circuit.ads9224r_module.*` values tagged `assumed` / datasheet-typical /
  EVM-topology baseline. Front-end decided: low-side shunt → THS4551 FDA →
  RC bucket → ADS9224R; 3.3 V digital rail (mates ULX3S, no translator).
- **§2 electrical as provenance:** `[circuit.ads9224r_module]` (shunt, FDA Rf/Rg,
  flt R/C, cdac, ref_v, reservoir) + derived `[feedback.current_ads9224r]`
  (fda_gain 20, full_scale ±102.4 A, **320 codes/A**) + `adc.acq_settle_residual_ads9224r`,
  all registered in `derive_params.py` (no orphans). SPICE front-end
  `sim/circuits/ads9224r_frontend.cir` (DC scaling) + `ads9224r_settle.cir`
  (settling); ngspice cross-checks in `test_spice_derivations.py`
  (slope = gain·shunt → 320 codes/A; settling < 0.5 LSB).
- **§3 schematic:** `gen_ads9224r_sch.py` → `hw/ads9224r-module/module.kicad_sch`
  (passive network from params; reuses the refactored `gen_kicad_sch.render_schematic`,
  existing schematic still byte-identical). kicad-cli SPICE round-trip + committed-current
  gated by `test_ads9224r_sch.py`.
- **§5 BOM:** `hw/ads9224r-module/bom.csv` (MPNs + distributor PNs + provenance)
  + the shunt×gain scaling-options table in the README.
- **§6 driver:** `ads9224r_master.v` already matches the board protocol (no RTL
  change); `rtl/contracts/ads9224r_master.md` gains a "Physical target" section
  (pin map + 320 codes/A scaling + board status).
- **§7 tests:** SPICE DC + settling, schematic round-trip, derivation re-derive;
  the existing `test_ads9224r.py` (C++ model) + part-comparison **T3/T4** (Q21 skew)
  still pass — the board's simultaneity rationale, re-confirmed.
- **§8 graphics:** `gen_ads9224r_figures.py` (`make ads9224r`) → `figures/ads9224r-module/`
  (signal_chain, simultaneity, scaling [ngspice], settling [ngspice]) + `gallery.md`.
- **§9 packaging:** `hw/ads9224r-module/README.md` (+ IP/legal hygiene),
  `contract.md` (board datasheet), **CERN-OHL-S-2.0** for the `hw/` tree
  (`LICENSES/` + `REUSE.toml` override; software stays MIT), CITATION keywords.
  `make ads9224r` regenerates schematic + figures.
- **§4 / §10 (NOT done here, by design):** PCB stackup/placement/routing needs
  interactive KiCad; fabrication + bench bring-up (reference IC, ENOB, measured
  skew, motor loop) is the lab step that promotes `assumed` → `measured` (Q23).
