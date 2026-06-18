<!-- SPDX-License-Identifier: MIT -->
# Open Questions

Consolidated from the docs deep-read ([docs-digest](docs-digest.md)) and the
architecture work ([architecture](architecture.md)), 2026-06-12. Check items
off as they resolve; move design decisions into the "Resolved design
decisions" section of `architecture.md` with their reasoning.

Sequencing note: Q1 (motor) and Q7 (board measurements) unblock the most.
Q10–Q12 are about an hour of desk work. Q3 can wait for simulation results
(Q18) if bench rungs 1–2 are built first.

Simulation parameters affected by an open question are tagged
`blocked_by = "Qn"` in `sim/config/params.toml`; every sim run prints the
list of parameters still resting on unconfirmed assumptions. When resolving a
question here, also flip the matching statuses there.

## Design decisions to make (block design work)

- [ ] **Q1 — Which motor?** No motor is identified anywhere in the project.
  R, L, Ke, J, pole count, and EMF shape parameterize the plant, OC
  thresholds, PWM sanity checks, and expected no-load speed. Resolve: pick the
  motor, then an identification session (DMM for R, LCR/current-step for L,
  drill-spin + scope for Ke and EMF shape, spin-down for J/friction).
- [x] **Q2 — ADC domain: 3.3 V or 5 V?** Resolved 2026-06-12: 3.3 V, with a
  sector-aware PWM-synchronized sampling schedule; TXB0108 off the critical
  path; future FOC-grade sampling via a parallel second ADC if ever needed.
  Full reasoning in `architecture.md`.
- [ ] **Q3 — Angle feedback path: AS5600 PWM, I2C, or check-only?** Latency
  (~1–2 ms through filter + PWM frame ≈ 48 elec deg at 2,000 RPM, 8 poles)
  suggests AS5600 may only support low-speed commutation and
  alignment/calibration. Resolve quantitatively via co-sim (Q18) before
  committing wiring.
- [ ] **Q4 — First control strategy commitment.** Presumably sensored six-step
  before sensorless BEMF integration, but not written down as a decision; it
  determines RTL block order.
- [ ] **Q5 — Bring-up operating point.** Bus voltage (12 V vs 24 V), PWM
  frequency (≥20 kHz for bootstrap health; ~20 kHz pinned by Q2 resolution),
  RTL dead-time value, and the OC_ADJ_SET code computed for 4.6 mΩ clone FETs
  at that operating point. *Supply-sizing evidence (2026-06-12, realism
  S6a): on a 1 A current-limited supply the placeholder motor cannot spin
  up at all — every acceleration transient browns the bus out through PVDD
  UVLO (the slva552 reset, reproduced emergently). Bring-up needs either a
  supply in the 5–20 A class or current-limited startup logic in the
  controller. Also: hard deceleration pumps the bus (+4 V observed, S6b);
  the RTL now carries a duty down-slew guard (rtl.duty_down_slew).*
- [ ] **Q6 — SPI topology.** DRV8301 wants mode 1; MCP3208 wants mode 0,0/1,1.
  Shared bus with per-device mode switching vs two small independent masters
  (simpler RTL; pins are not scarce).

## Hardware facts to measure (bench session before wiring the FPGA)

- [ ] **Q7 — ZONRI board actuals vs TI EVM baseline.** Shunt value (2 mΩ per
  Rev B1, clone revision unknown), EMF and bus divider ratios, DTC resistor,
  whether a logic rail is exposed and usable, pull-up/pull-down states on the
  digital header pins. *Resolution path (2026-06-12): run
  `python3 sim/scripts/derive_params.py --measurement-checklist` for the
  exact component worksheet; update the `[circuit.*]` values/statuses in
  params.toml and run `--update` — every derived parameter recomputes
  mechanically. The EMF `filter_topology` question (~10x cutoff difference)
  is an explicit parameter to resolve by measurement.*
- [ ] **Q8 — Physical FPGA chip marking.** PG138 is now known wrong on paper
  (GW5A-25 exists only in LQ100/LQ144/MG121N); confirm the marking once.
- [ ] **Q9 — Inter-board grounding/harness plan.** Star ground between dock,
  ZONRI board, and ADC; twisted pairs for analog sense lines
  (slva959b/slvaf66 guidance). *Resolution path (2026-06-12): the harness is
  codified in `[circuit.harness]` and the `[disturbance]` amplitudes derive
  from it; the realism scenarios (S8) currently demonstrate ≥2x margin at
  the assumed amplitudes — measure the real harness and update the
  components.*

## Documentation gaps to close (desk work)

- [ ] **Q10 — DRV8301 OC_ADJ_SET power-on default.** Not clearly marked in the
  register table; resolve with an SPI read at first power — it sets first-spin
  trip current.
- [ ] **Q11 — EVM bus-voltage divider math doesn't close.** Extracted 1:6.4
  puts 60 V at ~9.4 V, above any 3.3 V ADC. Re-read schematic p.6 or measure
  on the clone. *Resolution path (2026-06-12): the suspect values are now
  codified in `[circuit.bus_divider]`; fix the components there until they
  match measurement and the derived ratio updates mechanically.*
- [ ] **Q12 — Dock header pin↔FPGA-ball map and core oscillator frequency.**
  Schematic scan resolution defeated transcription (oscillator read as 25 MHz,
  low confidence). Pull the Sipeed wiki pinout table before writing
  constraints.
- [ ] **Q13 — GW5A pin behavior during configuration.** Unresolved in the
  datasheet extraction. Mitigation happens regardless (external pull-downs on
  EN_GATE, ideally INH/INL), which lowers priority. *Modeled (2026-06-12,
  realism S13): with the EN_GATE pull-down (`[circuit.gate_pulldowns]`) the
  power stage stays dead through a worst-case floating-pin window; without
  it, a slowly-drifting EN_GATE plus pin noise produces gate activity — the
  counterfactual that makes the pull-down mandatory.*

- [ ] **Q20 — AS5600 mounting eccentricity and alignment calibration.** A
  magnet off-axis by fractions of a millimeter produces degrees of
  once-per-rev angle error, feeding straight into sensored commutation.
  `[sensor_imperfection]` parameters are assumed until the assembled rotor
  is measured; the bench-side alignment-calibration routine
  (realism-checklist 5.4) is the procedure hardware bring-up reuses.

## Toolchain gaps

- [ ] **Q14 — FPGA build flow for GW5A.** Gowin EDA (proprietary, free
  license, Linux support varies) vs open yosys/Apicula flow, whose Arora-V
  (GW5A) support is much younger than for LittleBee parts — check maturity
  before committing. Plus bitstream loading (openFPGALoader vs Gowin
  programmer).
- [x] **Q15 — Lockstep bench toolchain versions.** Resolved 2026-06-12:
  Verilator 5.032, cmake 4.2.3, ninja 1.13.2, pybind11 3.0.1, pytest 9.0.2,
  g++ 15.2, omc 1.27.0-dev (OpenModelica apt repo supports `resolute`
  directly). `sim/scripts/check_cosim_toolchain.sh` verifies all of it.

## Questions the simulation is meant to answer (open, but planned)

- [ ] **Q16 — Averaged vs switched plant delta**, quantified once on the same
  scenario, then trusted.
- [x] **Q17 — Does the sector-aware 2-conversions-per-period ADC schedule
  support the current loop at the chosen PWM frequency?** Resolved
  2026-06-12 for six-step: yes. The bench verifies (scenario S3) that the
  active-phase current samples arrive every 20 kHz period and the
  floating-phase EMF hold aperture lands inside the PWM off-window across
  duty 0.10–0.95. FOC-grade simultaneous sampling remains out of reach for
  a single MCP3208 (already recorded in `architecture.md`; now **quantified**
  — see Q21: the sequential scheme injects a ~12× larger dq error).
- [ ] **Q18 — Does AS5600-based commutation degrade gracefully or fail as
  speed rises?** Feeds Q3. *Preliminary evidence (2026-06-12, placeholder
  motor params): degrades gracefully — closed-loop six-step is stable and
  settles cleanly at 80 rad/s, but tops out near 120 rad/s mech where the
  sensor's filter+frame latency (~1.5–2 ms ≈ 40–50 elec deg of lag) erodes
  torque. Frame quantization also forced a full-revolution speed
  measurement in the RTL (see simulation-checklist findings). Full sweep
  with identified motor params is post-milestone item P2.* **FOC update
  (2026-06-14, Q22):** for FOC the same latency costs torque that grows with
  speed; with ω·t_latency extrapolation the AS5600 is usable across the
  modeled range. The placeholder motor's voltage-limited base speed
  (~570 rad/s) is far above the sensor ceiling, so the sensor — not the
  voltage limit — bounds FOC speed here.
- [ ] **Q19 — Does the RTL's defensive logic survive the slva552 fault
  scenarios** (silent register reset, nOCTW storms, ADC noise, magnet-loss
  flags)?

## FOC conversion (notes/foc-checklist.md)

- [~] **Q21 — Synchronized phase-current sampling architecture for FOC.**
  FOC needs two phase currents at the *same* instant (the PWM-center, all
  low-sides conducting); the single sequential MCP3208 cannot (~22 µs per
  conversion ≈ a third of a 20 kHz period — Q17, `architecture.md`).
  Candidates: (a) two ADCs on separate SPI buses started simultaneously,
  (b) sequential single-ADC carrying the real inter-sample skew,
  (c) single-shunt DC-bus reconstruction (one sensor, minimum-vector blind
  zones). **Bench-resolved 2026-06-14 (foc-checklist stage 4,
  `test_foc_sampling.py`):** schemes (a) and (b) modeled; (a) recovers both
  currents (dq measurement error ~0.13 A), (b) loses ~half of phase B and
  injects a ~12× larger dq error (~1.5 A) because the second conversion lands
  after that leg's conduction window closes. **Recommendation: simultaneous
  sampling (dual ADC / external S&H) is required.** Scheme (c) characterized
  but not implemented (different sensor topology). *Remaining:* the actual
  board decision + layout.
- [~] **Q22 — Is AS5600 angle latency tolerable for FOC, and does
  extrapolation recover it?** For six-step the sensor frame+filter lag is a
  minor commutation-timing shift; for FOC it rotates the dq frame off-true
  and the error grows with speed (cos(lag) torque loss + cross-coupling).
  **Bench-resolved 2026-06-14 (foc-checklist stage 7,
  `test_foc_latency.py`):** with the raw sensor angle the developed torque
  falls measurably as speed rises (≈10 %+ by 120 rad/s on placeholder
  params); advancing the angle by ω·t_latency recovers it, and is neutral at
  low speed. So the AS5600 is usable for FOC *with* extrapolation across the
  modeled range. *Remaining:* confirm with a real motor's base speed and the
  measured sensor latency (the linear extrapolation overshoots past
  ~300 rad/s, irrelevant for this sensor's range). Extends Q18 / feeds Q3.
- [ ] **Q23 — Open ADS9224R current-sense module bring-up.** The sim resolves
  Q21 (simultaneous sampling wins) but no open ADS9224R board exists to build
  it on (researched — TI's EVM is the only reference, not open EDA). The open
  module (`hw/ads9224r-module/`, `notes/ads9224r-open-board-checklist.md`) is
  designed off the EVM topology (THS4551 FDA per channel + buffered reference +
  RC charge-bucket). *To confirm on hardware:* the reference IC/value, the FDA
  gain + shunt scaling (codes/A), and the acquisition settling / ENOB at the
  conversion rate — then promote the `assumed` `circuit.ads9224r_module.*`
  values to `measured`. Closes the loop on Q21 with a real, citable board.
