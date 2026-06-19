<!-- SPDX-License-Identifier: MIT -->
# ADS9224R module — simulation validation checklist (Tiers 2–4)

Raise the open ADS9224R module (`hw/ads9224r-module/`,
`notes/ads9224r-open-board-checklist.md`) from "topology math" to
**vendor-model-validated** measurements, *in simulation*, before fab. Extends the
analog provenance already in place (Tier 1: closed-form + behavioral ngspice —
`sim/circuits/ads9224r_{frontend,settle}.cir`, `test_spice_derivations.py`).

## The provenance ceiling (honest, up front)

Simulation cannot produce a `measured` value — that needs the fabbed board
(Q23 / open-board §10). It **can** produce a real tier above "estimate":

| Tier | Method | Status it unlocks |
| --- | --- | --- |
| 1 (have) | closed-form + behavioral B-source ngspice | `assumed` topology is self-consistent |
| **2** | datasheet-parameter device models (op-amp GBW/noise, switched-cap ADC input) | promote select values `assumed → datasheet` |
| **3** | TI **vendor SPICE macromodels** (THS4551 / REF60xx / ADS9224R) | "vendor-model-validated" (cite model + report) |
| **4** | feed the SPICE-extracted nonidealities into the co-sim → FOC loop | **system-level** validated current-loop accuracy |
| (lab) | fabbed board, bench (open-board §10) | `measured` |

Precedent in this repo: `test_ti_vendor_amp_model` already runs TI's DRV8301 amp
macromodel via `spice_runner` (`aux_files=` + `compat="psa"`, skip-if-absent) and
found a real ~40 mV offset — Tier 3 is the same move for this module.

The measurements to gather (each tier sharpens them): **acquisition settling to
<0.5 LSB**, **ENOB/SNR from integrated noise**, **bandwidth**, **THD**, **gain/
offset error**, **reference droop/PSRR**. Acceptance is referenced to the
**ADS9224R datasheet** own SNR/ENOB so the front-end doesn't throttle the part.

TI methodology this follows (capture as references, archive PDFs proprietary):
TIPL 4405 *Amplifier Settling & Charge-Bucket Filter Design*; SBAA277 *Driving a
switched-cap SAR*; SBOA443 *Current-Sense Amplifier Considerations for Driving SAR
ADCs* (directly on-point — current sensing → SAR); SBAA282 *Antialiasing filter
design*; the TI **Analog Engineer's Calculator** (Rfilt/Cfilt range — our 10 Ω /
1.5 nF sits inside its 4.1–32.5 Ω / ~1.1 nF window).

---

## §0 — Shared infrastructure (once, before the tiers)

- [ ] Add a **noise/ENOB helper** to `sim/scripts/` (e.g. `adc_metrics.py`):
      integrate an ngspice `.noise` output-noise density over the Nyquist band →
      total RMS → `SNR = 20·log10(VFS_rms / Vn_rms)`, `ENOB = (SNR − 1.76)/6.02`.
      Combine the front-end noise with the ADC's own quantization+thermal SNR
      (root-sum-square) so ENOB is the *system* number, not just the amp.
- [ ] Confirm `spice_runner.run_netlist` covers the new analyses (`.noise`, `.ac`,
      `.four`/FFT, `.tran` with switches). It already supports `aux_files=` (drop a
      vendor `.LIB`) and `compat=` (PSpice dialect) — used by the DRV8301 test.
- [ ] Decide acceptance thresholds from the **ADS9224R datasheet** (extract its
      SNR/ENOB at the reference; e.g. ~92 dB SNR class → ~15 ENOB) so "front-end
      doesn't degrade the part by >0.X LSB / >0.X bit" is a concrete gate.

---

## §2 — Tier 2: datasheet-parameter models (buildable NOW, no downloads)

Replace ideal B-sources with device models parameterized **from the datasheets**;
honest label = `datasheet-model`.

### §2.1 Capture device parameters into `params.toml` (provenance-tagged)
- [ ] `[circuit.ths4551]` (status `datasheet`, `source` = SLOSxxxx): voltage-noise
      density (≈3.3 nV/√Hz broadband + 1/f corner), current-noise density
      (≈0.5 pA/√Hz), GBW (≈135 MHz), unity BW (150 MHz), slew (220 V/µs), output
      impedance / Cload spec, Vos (≈±175 µV), supply, Iq. **Extract exact values
      from the datasheet** — the listed figures are typical placeholders to verify.
- [ ] `[circuit.ads9224r_adc]`: sample/hold cap `Csh`, acquisition time at 3 MSPS,
      input impedance, datasheet SNR/ENOB, **internal reference 4.096 V + integrated
      reference buffer** (confirm), AVDD = 5 V, IOVDD 1.65–3.6 V (mates 3.3 V FPGA),
      differential full-scale ±VREF. (The current `circuit.ads9224r_module.cdac`
      placeholder gets replaced/anchored by the real `Csh`.)
- [ ] `[circuit.ref6041]` (only if the external-ref option is pursued, §3.4): 4.096 V,
      5 ppm/°C, output noise (~3 µVpp / spectral density), buffer drive.
- [ ] Register any new derived quantities (e.g. acquisition `t_acq` from MSPS) in
      `derive_params.py`; keep the orphan check green (consume or DIRECTLY_CONSUMED).

### §2.2 THS4551 datasheet-model netlist
- [ ] `sim/circuits/ths4551_model.cir` (or a `.subckt`): a one/two-pole op-amp
      macromodel from GBW + a second pole, finite output impedance, and **noise
      sources** (input `vnoise`/`inoise` + resistor Johnson noise). This is the
      reusable amplifier block for the analyses below.

### §2.3 Real switched-capacitor settling (replaces the stiff-source RC)
- [ ] `sim/circuits/ads9224r_acq.cir`: model the ADC input as a **voltage-controlled
      switch + Csh** that disconnects during conversion and reconnects at
      acquisition (the actual charge-kickback), driven by the THS4551 model through
      Rflt/Cflt. `.tran` the acquisition; measure the held-sample error vs the input
      at the end of `t_acq`.
- [ ] **Acceptance:** settles to **< 0.5 LSB** within `t_acq`. This supersedes the
      Tier-1 single-pole estimate (`adc.acq_settle_residual_ads9224r`) with the true
      kickback settling; record both and the delta.
- [ ] Sweep Rflt/Cflt (ngspice `.step`-style param overrides via
      `spice_runner overrides=`) to confirm the chosen 10 Ω / 1.5 nF is near-optimal
      per the TIPL-4405 / Analog-Engineer's-Calculator method; log the trade curve.

### §2.4 Noise → ENOB
- [ ] `sim/circuits/ads9224r_noise.cir`: `.noise` over DC→Nyquist of the
      shunt+FDA+Rflt+reference chain; the §0 helper integrates → input-referred RMS
      → SNR → ENOB.
- [ ] **Acceptance:** front-end ENOB within a stated budget of the ADS9224R
      datasheet ENOB (the amp/reference shouldn't cost more than ~0.X bit).
      Decompose the noise (amp vs resistors vs reference) so the dominant term is
      visible — this drives the §3.4 reference decision.

### §2.5 Bandwidth, distortion, gain/offset
- [ ] `.ac` → closed-loop **bandwidth**; assert ≥ the motor-current bandwidth needed
      (and that the antialiasing pole is placed per SBAA282).
- [ ] `.tran` a full-scale sine + `.four`/FFT → **THD/SFDR** (the nonlinearity the
      B-source can't show); assert below a budget vs the ADC's SFDR.
- [ ] `.dc` sweep with the model's Vos/finite-gain → **gain & offset error** in LSB
      (mirrors what the DRV8301 vendor test found); record for the FOC offset path.

### §2.6 Wire in
- [ ] Add `test_spice_derivations.py` cross-checks for §2.3–§2.5 (assert the
      datasheet-model numbers hit the acceptance gates).
- [ ] Extend `gen_ads9224r_figures.py`: a **noise-spectrum/ENOB** figure and a
      **Rflt/Cflt settling trade** figure (both from live ngspice).
- [ ] **Promote** the values the sim now backs from `assumed → datasheet`
      (`derive_params.py --update`); note the provenance jump in `contract.md`.

---

## §3 — Tier 3: vendor-macromodel cross-check (needs the model files)

### §3.1 Acquire + archive the TI SPICE/TINA models
- [ ] Download from the TI product pages ("Design & development → Simulation
      models"): **THS4551** PSpice model (`.lib`, loads in ngspice), **REF6041**
      (TINA/PSpice), and the **ADS9224R** TINA model / input model if published
      (else use the §2.3 switched-cap input + datasheet Csh).
      Pages: ti.com/product/THS4551, /REF6041, /ADS9224R.
- [ ] Archive under `docs/ti-simulation-models/` (the existing DRV8301 location);
      cover as **`LicenseRef-Proprietary-Reference`** in `REUSE.toml` (vendor IP, not
      MIT) — like the DRV8301 `.LIB` and the vendor PDFs. Record SHA + source URL.

### §3.2 Wire the macromodels into the testbenches
- [ ] Re-point §2.2–§2.5 netlists at the vendor `.subckt` via
      `spice_runner(aux_files={"THS4551.LIB": ...}, compat="psa")`; **skip-if-absent**
      tests (mirror `test_ti_vendor_amp_model`'s `pytest.skip`), so CI stays green
      without the proprietary files but validates when present.

### §3.3 Cross-check vendor vs datasheet-model
- [ ] Re-run settling, noise/ENOB, BW, THD, gain/offset with the macromodels;
      tabulate **vendor vs Tier-2 datasheet-model**. Reconcile deltas (the macromodel
      carries effects the one-pole model omits — e.g. real Zout(f), distortion).
      Where they disagree materially, the vendor number governs.

### §3.4 Internal vs external reference (a sim-decided BOM option)
- [ ] The ADS9224R has an **integrated 4.096 V reference + buffer**, so REF6041 is a
      *precision option*, not a requirement. Simulate **both** reference paths
      (internal vs REF6041) for **noise → ENOB** and **reference droop** (the SAR
      pulls charge from REF each conversion — does the reservoir + buffer hold REF
      within 0.5 LSB across the conversion?).
- [ ] Decide + record: default to the internal reference unless the ENOB/drift delta
      justifies REF6041; update `bom.csv` (REF6041 = optional populate) +
      `README.md` build options + `ref_v`/`ref_reservoir_c` provenance accordingly.

### §3.5 Promote provenance
- [ ] Label every §3 number **vendor-model-validated** (cite the model file + a short
      report); promote the corresponding `params.toml` values' `source` to the
      macromodel cross-check. Update `hw/ads9224r-module/contract.md` "validation
      status".

---

## §4 — Tier 4: close the loop in the co-sim (the system-level measurement)

Turn the component-level SPICE results into a **FOC-loop** measurement, using the
bench's silicon-model ↔ RTL ↔ plant chain (the whole point of motorloop).

### §4.1 Expose the front-end nonidealities on the ADS9224R device model
- [ ] In `sim/cpp/src/ads9224r.{hpp,cpp}`, add (config-driven) the SPICE-extracted
      nonidealities: **acquisition settling residual**, **input-referred noise**
      (RMS, seeded), and **gain/offset error** — defaulting to the
      `datasheet`/vendor-validated values from §2–§3 (thread via `bench_factory`
      from `params.toml`, e.g. `feedback.current_ads9224r.*` + the settling residual).
- [ ] Keep the noiseless path byte-exact (noise off by default for parity runs);
      add cocotb/model equivalence as needed.

### §4.2 System measurement: re-run the part-comparison study
- [ ] With the validated front-end parameters live, re-run `test_part_comparison.py`
      **T3/T4** (the Q21 skew + noise-floor study). Report the **FOC current-loop dq
      error / effective ENOB at the loop**, and confirm the module still **retires
      Q21** by the expected margin over the MCP3208 path — now with a
      *vendor-validated* front-end, not idealized.
- [ ] Optionally add a dedicated `test_ads9224r_module_loop.py` asserting the
      loop-level error stays within budget with the validated noise/settling on.

### §4.3 Report + figure
- [ ] A **system-validation figure** (front-end ENOB/settling → loop dq error) +
      a section in `notes/part-comparison-report.md` (or a module validation report)
      tracing SPICE front-end → device model → control loop. This is the headline
      "validated measurement": *what the module's simulated analog performance costs
      the actual FOC current loop.*

---

## §5 — Provenance, packaging, repro

- [ ] `REUSE.toml`: vendor models under `docs/ti-simulation-models/**` →
      `LicenseRef-Proprietary-Reference`; new `.cir`/scripts MIT; `reuse lint` clean.
- [ ] Update `hw/ads9224r-module/{README,contract}.md` "validation status": which
      values are `datasheet` / vendor-model-validated, which remain `assumed` pending
      hardware (Q23).
- [ ] `Makefile`: extend `make ads9224r` (or add `ads9224r-validate`) to run the new
      ngspice analyses + render the new figures; the skip-if-absent vendor tests keep
      `make test` green without the proprietary `.LIB`s.
- [ ] Update `notes/open-questions.md` Q23: record what sim has now de-risked and
      what *only* the bench can still settle (absolute ENOB, real layout coupling).

## Done-when

The module's **settling (<0.5 LSB), ENOB, bandwidth, THD, gain/offset, and
reference droop** are each backed by an ngspice analysis with an acceptance gate;
the numbers are cross-checked against TI vendor macromodels where the `.LIB`s are
present (skip-if-absent otherwise); the validated front-end parameters flow into
the co-sim and the part-comparison study reports the **loop-level** current
accuracy with Q21 still retired; every value is labelled
datasheet-model / vendor-model-validated, with `measured` reserved for the bench.

## What NOT to do

- Don't call any simulated number `measured` — vendor-model-validated is the
  ceiling here; the fabbed board (Q23/§10) earns `measured`.
- Don't gate CI on the proprietary vendor `.LIB`s — skip-if-absent (the DRV8301
  pattern); ship the datasheet-model tier in CI.
- Don't license the vendor models MIT — `LicenseRef-Proprietary-Reference`, like
  the existing DRV8301 model + datasheets; don't redistribute beyond what TI allows.
- Don't tune Rflt/Cflt to a single corner — verify across the datasheet
  Csh/acquisition spread (TIPL-4405 method), and keep the choice byte-consistent
  with `params.toml`.
- Don't let the SPICE front-end and the co-sim model drift — §4.1 threads the same
  numbers; the part-comparison re-run is the guardrail.

## Key references (archive PDFs as proprietary-reference)

- ADS9224R datasheet — ti.com/lit/ds/symlink/ads9224r.pdf
- THS4551 datasheet — ti.com/lit/ds/symlink/ths4551.pdf
- REF6041 datasheet — ti.com/product/REF6041
- TIPL 4405 Amplifier settling & charge-bucket filter design — training.ti.com
- SBAA277 Driving a switched-cap SAR ADC — ti.com/lit/sbaa277
- SBOA443 Current-sense amplifier considerations for driving SAR ADCs — ti.com/lit/an/sboa443
- SBAA282 Antialiasing filter design — ti.com/lit/pdf/sbaa282
- TINA-TI / PSpice for TI (model host) — ti.com/tool/TINA-TI

## Implemented (results)

Done end-to-end; full write-up + numbers in
[`ads9224r-sim-validation-report.md`](ads9224r-sim-validation-report.md). All new
tests green; derivations consistent; REUSE clean. Honest ceiling held: vendor-
macromodel numbers wait on the portal-gated `.LIB` (skip-if-absent), and
`measured` waits on the bench (Q23).

- **§0/§2 (Tier 2):** `adc_metrics.py` (SNR/ENOB/current-noise helpers). Datasheet
  anchoring from the archived SBAS876C/SBOS778D — corrected Csh 60→**16 pF**,
  tACQ 200→**140 ns**, added `[circuit.ths4551]` + `[circuit.ads9224r_adc]`
  (status `datasheet`). `ads9224r_acq.cir` (real switched-cap settling, **1.6e-7
  < 0.5 LSB**); `ads9224r_noise.cir` (`.noise` → ENOB). **Finding + fix:** the
  bucket alone costs ~1.9 bits to wideband noise → added the antialiasing cap
  `fda_fb_c` (derived `signal_bw_hz` = 295 kHz) → **0.46-bit** cost. Tests in
  `test_spice_derivations.py`; figure `noise.png`.
- **§3 (Tier 3):** `ths4551_vendor.cir` + `test_ads9224r_vendor.py` +
  `spice_runner.THS4551_LIB/REF6041_LIB` + `docs/ti-simulation-models` download
  instructions — skip-if-absent (DRV8301 pattern), cross-checks when a `.LIB`
  lands. **Reference decision (§3.4, datasheet-based):** internal ref default,
  REF6041 optional (drift only) — `bom.csv` U3 → qty 0.
- **§4 (Tier 4):** `test_ads9224r_loop.py` bridges the SPICE front-end to the
  loop — combined **1.79 mA RMS (~15.0 effective bits, 0.0017 % FS)**, balanced
  against the ADC's own noise; tied to the part-comparison Q21 (skew) result.
  Figure `loop_budget.png`.
- **§5:** datasheets archived (`docs/datasheets/ti-ths4551-datasheet.pdf`),
  README updated, REUSE proprietary-reference; `make ads9224r` renders all 6
  figures.
- **Deliberately not done (honest):** the vendor-macromodel run (portal-gated
  `.LIB`) and the C++-model per-sample noise injection (§4.1 deeper integration)
  — the loop budget is computed analytically from the validated front-end instead.
