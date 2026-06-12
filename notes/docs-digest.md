# Docs Digest: Extracted Facts From docs/

Deep read of the PDF collateral in `docs/`, 2026-06-12. Numbers below are from the
named source documents; TI EVM values are the *reference baseline* for the ZONRI
clone and still need verification on the physical board. Items marked **VERIFY**
were ambiguous in extraction or are clone-dependent.

Register tables and SPI behavior in the DRV8301 section were verified directly
against the datasheet text (SLOS719F pages 21–23). Other sections were extracted
by document readers and spot-checked; treat unusual values with normal suspicion
and re-check the cited page before committing them to RTL or hardware.

---

## 1. DRV8301 Gate Driver (`ti-drv8301-datasheet.pdf`, SLOS719F)

### Electrical / logic interface

- PVDD operating range 6–60 V. Internal buck converter (1.5 A capable) plus
  charge-pump/regulators for GVDD, AVDD, DVDD.
- Digital inputs: VIH = 2.0 V, VIL = 0.8 V → **3.3 V FPGA logic is directly
  compatible**. nFAULT/nOCTW are **open-drain** (need pull-ups; EVM uses ~47 kΩ
  in the VOH spec condition).
- All digital control inputs (INH/INL_x, EN_GATE, DC_CAL, SCLK, SDI, nSCS) have
  internal ~100 kΩ pull-downs.

### PWM modes and dead time

- 6-PWM mode (default): INH_x/INL_x control each gate independently;
  INH=INL=1 → both gates **low** (shoot-through guard).
- 3-PWM mode (Control Reg 1, PWM_MODE=1): INH_x alone controls the half-bridge
  complementarily; INL_x ignored after mode entry.
- **The DRV8301 has a DTC pin (pin 7)**: dead time programmable 50–500 ns via
  0–150 kΩ resistor to GND, linear; DTC shorted to GND = 50 ns minimum.
  Shoot-through prevention is always active regardless of DTC. The RTL should
  still command its own dead time and treat the DTC value as a hardware floor —
  check what resistor the ZONRI board fits on DTC (**VERIFY** on board).
- Timing: input-to-gate propagation ≈ 45 ns typ; min PWM pulse 50 ns;
  error-event→gates-low and →nFAULT ≈ 200 ns.

### EN_GATE sequencing

- After EN_GATE goes high: **wait 5–10 ms** before PWM or SPI (gate drivers and
  SPI not ready earlier).
- EN_GATE low pulse < ~10 µs = "quick reset": clears gate-driver faults and SPI
  status registers without dropping the buck/other blocks. Longer low = full
  shutdown/restart. RTL fault-recovery should implement the quick-reset pulse.

### SPI (verified against SLOS719F pp. 21–23)

- 16-bit frames, MSB first. SDI word: `W0[15] A[14:11] D[10:0]`.
  **W0 = 0 write, W0 = 1 read.** SDO word: `F0[15] A[14:11] D[10:0]` where F0 is
  a frame-fault flag.
- **Mode 1 behavior, not mode 0**: SCLK must be low when nSCS falls and rises;
  SDO shifts out on SCLK **rising** edge, SDI is sampled on SCLK **falling**
  edge. Max SCLK 10 MHz (tCLK ≥ 100 ns).
- Response is pipelined: the reply to frame N arrives in frame N+1. A write
  frame's N+1 response carries Status Register 1. Frames ≠ 16 clocks are frame
  errors (write ignored, F0 set in next response).
- Reading Status Register 1 clears its latched bits (per slva552 usage).

### Register map (verified)

Status Register 1 (0x00, R, defaults 0):

| D10 | D9 | D8 | D7 | D6 | D5 | D4 | D3 | D2 | D1 | D0 |
|---|---|---|---|---|---|---|---|---|---|---|
| FAULT | GVDD_UV | PVDD_UV | OTSD | OTW | FETHA_OC | FETLA_OC | FETHB_OC | FETLB_OC | FETHC_OC | FETLC_OC |

Status Register 2 (0x01, R): D7 = GVDD_OV, D[3:0] = Device ID.

Control Register 1 (0x02, R/W):

| Bits | Field | Values (default first) |
|---|---|---|
| D[1:0] | GATE_CURRENT | 00=1.7 A, 01=0.7 A, 10=0.25 A, 11=reserved |
| D2 | GATE_RESET | 0=normal, 1=reset latched faults (self-clears) |
| D3 | PWM_MODE | 0=6-PWM, 1=3-PWM |
| D[5:4] | OCP_MODE | 00=current limit, 01=OC latch shutdown, 10=report only, 11=OC disabled |
| D[10:6] | OC_ADJ_SET | 5 bits, 0–31 → VDS trip 0.060 V…2.400 V (table 13) |

Control Register 2 (0x03, R/W):

| Bits | Field | Values (default first) |
|---|---|---|
| D[1:0] | OCTW_MODE | 00=report OT and OC, 01=OT only, 10=OC only |
| D[3:2] | GAIN | 00=10 V/V, 01=20, 10=40, 11=80 |
| D4 | DC_CAL_CH1 | 1=short amp 1 inputs, disconnect load |
| D5 | DC_CAL_CH2 | 1=short amp 2 inputs, disconnect load |
| D6 | OC_TOFF | 0=cycle-by-cycle, 1=off-time control |
| D[10:7] | reserved | |

OC_ADJ_SET highlights: code 0 = 0.060 V, 8 = 0.155 V, 16 = 0.403 V,
27 = 1.491 V, 31 = 2.400 V. Codes 28–31 are disallowed for 6–8 V PVDD
operation. Power-on default of OC_ADJ_SET was not clearly marked in the table —
**VERIFY by SPI read on first bring-up** before trusting any trip math.

### Current shunt amplifiers

- Two channels (SO1/SO2). Output = VREF/2 − G·(SNx − SPx); gains 10/20/40/80 V/V
  via SPI. Offset ≈ 4 mV typ, drift 10 µV/°C.
- DC_CAL (pin or SPI bits) shorts the amp inputs and disconnects the load so the
  controller can measure/cancel offset — usable any time, even while switching.

### Protection / faults

- VDS overcurrent sensing on both high- and low-side FETs, threshold from
  OC_ADJ_SET; trip current = VDS_trip / RDS(on) of the *installed* FETs.
- Four OC modes per OCP_MODE above. In current-limit mode OC events show on
  nOCTW as ~64 µs stretched pulses; in latch mode the half-bridge shuts down and
  needs GATE_RESET or EN_GATE toggle.
- nFAULT asserts for: PVDD_UV, GVDD_UV, GVDD_OV (latched, needs full EN_GATE
  cycle), OTSD (latched), OC-latch events. nOCTW reports OTW and OC per
  OCTW_MODE.
- Errata: keep SH_x below 8.5 V when EN_GATE is asserted; a 13–15 µs PVDD
  brownout can hang the device until a full power cycle.

### slva552 (current-limit app note) — RTL consequences

- Real-world failure mode: load transients can dip DVDD/AVDD/GVDD/PVDD through
  UVLO → **all SPI registers silently reset to defaults** (OC threshold, gain,
  mode all revert). Clone boards with weaker decoupling are more exposed.
- RTL requirements derived from this:
  - Periodically re-read Control Registers and compare against intended values;
    rewrite on mismatch (register-refresh/watchdog).
  - Treat nOCTW/nFAULT as asynchronous inputs with synchronizers; nOCTW pulses
    are 64 µs so a slow poll loop is fine, but don't rely on SPI status alone.
  - After any fault recovery, re-initialize both control registers.

### SPI-optional first spin?

Yes, gate drive works with power-on defaults (6-PWM, current-limit mode, gain
10 V/V). But the default OC threshold is at/near the lowest VDS setting —
combined with the clone's 4.6 mΩ FETs that trips around 13 A (0.060 V/4.6 mΩ),
likely fine for first low-power spins, then set OC_ADJ_SET deliberately.

---

## 2. DRV8302 differences (`ti-drv8302-datasheet.pdf`)

Relevant only because the ZONRI PCB supports both assembly variants. DRV8302
replaces SPI with pins: M_PWM (6/3-PWM mode), M_OC (cycle-by-cycle vs latch),
GAIN (10 or 40 V/V only), OC_ADJ (analog threshold), DC_CAL. PVDD 8–60 V. Our
board has a DRV8301 installed → use the SPI path and the DRV8301 silkscreen
table.

---

## 3. Power stage reference: TI HC-EVM (TIDM-THREEPHASE-BLDC-HC-SPI)

Source: `tidr738` schematic, RevD `515502~1.PDF`, BOM `tidr740a`, guides
`tidu317`/`tidu396`/`spruhx4`, history `tidr741`. **All values are TI-EVM
baseline — measure the ZONRI board before relying on any of them.**

- **Shunts: 2 mΩ, 5 W, 1% (Vishay WSR5, R80/R81/R82), one per low-side leg,
  three phases.** Rev B1 (2011-07-14) changed shunts to 2 mΩ; earlier revs
  differ, and the clone could derive from any rev (**VERIFY: measure actual
  shunt value**).
- Current sense: DRV8301 SO1/SO2 cover two phases; the EVM additionally has
  external op-amp stages (OPA365) producing buffered IOUTA/B/C, referenced to
  an onboard **1.65 V reference (REF+ / EX-REF nets)**. Exact gain network
  values not fully extracted (**VERIFY from 515502~1.PDF p.6 / measure**).
- Back-EMF dividers EMF-A/B/C: ≈ 95.3 kΩ : 10 kΩ → **scale ≈ 0.095 (1:10.5)**;
  RC filtering ≈ 1 kΩ + 0.1 µF → fc ≈ 1.6 kHz. At 60 V phase ≈ 5.7 V at the
  divider output — note this exceeds a 3.3 V-referenced MCP3208 input at high
  bus voltage (**plan ADC range accordingly; at our bench voltages it's fine**).
- Bus voltage divider: extraction suggested 53.6 kΩ : 10 kΩ (≈1:6.4), which
  doesn't fit a 60 V design — **VERIFY** (likely additional scaling not captured).
- DC bus capacitance: ~1660 µF bulk (2×330 µF + 1×1000 µF) + ~11 µF ceramic.
- DRV8301 buck on the EVM is configured ≈5 V (33 µH inductor), feeding
  TPS73633 3.3 V LDOs (400 mA) — the clone may or may not populate the same;
  **VERIFY whether the ZONRI board exposes a logic rail before powering
  external logic from it**.
- EVM ratings: 8–60 V input, 60 A peak claims — TI numbers, not the clone's.
- The EVM's external-controller header (tidu317 Table 4) carries exactly the
  signal set on the ZONRI silkscreen (PWM_xH/xL, EN_GATE, DC_CAL, nFAULT,
  nOCTW, SPI, IA/IB/IC-FB, Vhb1-3, VDCBUS) — good confirmation the clone clones
  the HC-EVM interface.

### FETs: clone vs reference

| Param | CRSS052N08N (ZONRI) | CSD18540Q5B (TI ref) |
|---|---|---|
| VDS max | 85 V | 60 V |
| RDS(on) @10 V | 4.6 mΩ | 1.8 mΩ |
| Qg | 55 nC | 41 nC |
| VGS(th) | 2–4 V | 1.5–2.3 V |
| Body diode trr | 60 ns | 82 ns |

Consequences: ~2.6× conduction loss vs TI design; same OC_ADJ_SET code trips at
~2.6× **lower** current on the clone (VDS sensing is RDS(on)-ratiometric) —
conservative for bring-up, must be recomputed for real loads. Plant model
should use CRSS052N08N parameters.

### Bring-up sequence distilled (tidu396/tidu317)

1. Current-limited supply, no motor; verify rails; LEDs.
2. Wait ~2 s after power for DRV8301 init; EN_GATE low until FPGA configured.
3. EN_GATE high → wait 10 ms → SPI read Status 1/2 (clears latched), read/set
   Control 1/2 (OC mode + threshold + gain).
4. DC_CAL cycle, capture ADC offsets (~1.65 V) on IOUT channels.
5. Low-duty open-loop commutation, no motor → scope PWM/gates.
6. Motor attached only after unloaded checks; keep PWM ≥ ~20 kHz (bootstrap
   sag at low frequency); monitor nFAULT/nOCTW continuously.
7. Bus caps hold charge after power-off — wait before touching.

---

## 4. MCP3208 ADC (`microchip-mcp3208-datasheet.pdf`, DS21298)

- Supply 2.7–5.5 V. **fSCLK max: 2.0 MHz @ 5 V, 1.0 MHz @ 2.7 V** — at 3.3 V
  budget **1.0–1.35 MHz** (datasheet doesn't characterize 3.3 V; stay
  conservative).
- Also a **minimum** clock constraint: sample cap droops if a conversion is
  stretched (~1.2 ms total frame budget) — don't pause mid-frame.
- Frame: CS low → start bit(1), SGL/DIFF, D2, D1, D0 → 1.5-clock sample window
  → null bit → 12 data bits MSB-first. Budget ~19–24 clocks with CS overhead;
  CS must go high ≥ 500 ns between conversions. SPI modes 0,0 and 1,1 both
  supported.
- Throughput at 1 MHz SCLK: ≈ 50 kSPS aggregate → **≈ 8 kSPS per channel
  round-robin over 6 channels** (IOUTA/B/C + EMF-A/B/C). Enough for a
  multi-kHz current loop only if you sample a subset per cycle — e.g. 2–3
  channels per PWM period at 20 kHz is not feasible at 1 MHz SCLK
  (3 conversions ≈ 60+ µs > 50 µs period). **Design decision needed:** lower
  PWM-synchronized sampling rate, fewer channels per cycle, or faster
  VDD=5 V ADC domain with level shifting. (Co-sim should model the real
  achievable sampling pattern.)
- VREF 0.25 V–VDD, LSB = VREF/4096. Keep source impedance ≲ 1 kΩ for full
  settling (EVM IOUT op-amp outputs are fine; EMF dividers are ~9 kΩ Thevenin —
  **add buffering or extend sample time; VERIFY behavior**).

## 5. AS5600 angle sensor (`ams-osram-as5600-datasheet.pdf`)

- **Runs natively at 3.3 V** (tie VDD5V+VDD3V3 per figure 6) → no level shifter
  needed for I2C if wired to a 3.3 V master. I2C address 0x36, up to 1 MHz.
- Internal sampling ≈ 150 µs; slow-filter settling 0.286–2.2 ms depending on SF
  setting; angle output is 12-bit (RAW ANGLE vs scaled ANGLE registers).
- OUT pin modes via CONF.OUTS: analog (full/reduced range) or **PWM** with
  carrier 115/230/460/920 Hz (CONF.PWMF). PWM frame = 128-clock-style
  init/data/error framing; at 920 Hz the angle update via PWM is ~1 ms latency —
  fine for commutation sanity checks, marginal for a fast loop at speed.
- STATUS register has magnet too-strong/too-weak/detected flags (MH/ML/MD);
  AGC and MAGNITUDE registers useful for mechanical alignment checks.
- **DANGER: register 0xFF burn commands (ZPOS/config) are one-time OTP.** Never
  write 0xFF during debug; guard it in any I2C driver.
- PGO is a programming-option pin — leave per breakout default.

## 6. TXB0108 level shifter (`ti-txb0108-datasheet.pdf`)

- Auto-direction via edge one-shots + weak keepers; push-pull signals only.
- **Explicitly not for I2C/open-drain.** External resistors on TXB lines must be
  ≥ 50 kΩ — normal I2C pull-ups fight the keepers.
- VA (A-port) 1.2–3.6 V, VB (B-port) 1.65–5.5 V, **VA ≤ VB always**; OE low
  during power sequencing.
- Verdict for this project: fine for SPI (≤ a few MHz here), PWM lines, AS5600
  PWM output; **never for SDA/SCL**. Given everything in the chain is 3.3 V
  (FPGA, DRV8301 logic, MCP3208 at 3.3 V, AS5600 at 3.3 V), the HW-221 may be
  unnecessary entirely — strongest reason to keep it would have been a 5 V
  MCP3208 domain, and that was decided against on 2026-06-12 (see
  [architecture](architecture.md)). The HW-221 stays in the parts bin.

## 7. FPGA: Tang Primer 25K / GW5A (`gowin-gw5a-datasheet-ds1103e.pdf`, Sipeed schematics)

- **Part-number question resolved: GW5A-25 ships in LQ100, LQ144, and MG121N
  only — there is no PG138 variant of the 25K die.** The core module uses the
  MG121N BGA. The `GW5A-LV25PG138C1/I0` string in earlier project notes is
  wrong; correct family string is GW5A-LV25MG121 (C1/I0 = speed/temp grade).
  (Still read the physical chip marking when convenient, per standing note.)
- GW5A-25 resources: 23,040 LUT4/FF, 1,008 Kb BSRAM, 28 DSP multipliers (27×18),
  6 PLLs — far more than a BLDC controller needs.
- Dock I/O (visual schematic read, medium confidence on details, high on
  totals): **two 2×20 40-pin headers (J1/J2) wired directly to FPGA balls, VCC
  pins at 3.3 V, no level shifters or series buffers**, plus PMOD-style
  connectors and buttons/LEDs. The ~21–23 signals this project needs fit on a
  single 40-pin header. **VERIFY exact pin↔ball map against the Sipeed wiki
  pinout table when writing constraints** (schematic scan resolution limited
  pin-by-pin transcription).
- Core module: 3.3 V VCCIO banks, core 0.9 V (LV device); oscillator read as
  25 MHz (**VERIFY — feed it to a PLL regardless**).
- Unresolved: documented pin state **during configuration** (the GW5A datasheet
  extraction did not pin this down). Safety stance regardless: external
  pull-downs on EN_GATE (primary) and ideally INH/INL lines so the power stage
  cannot enable while the FPGA configures; DRV8301's internal 100 kΩ pull-downs
  help only once its supplies are up.

## 8. Control algorithms (TI app notes)

### tida014 — sensorless trapezoidal via BEMF integration

- Six-step commutation @ 20 kHz PWM; floating phase sampled for BEMF
  (sampling positioned relative to PWM to avoid switching edges).
- Integrate floating-phase voltage over the sector; commutate when the
  integral crosses a tunable threshold (≈0.4 PU starting point); reset
  integrator each commutation. Thresholding replaces explicit 30°-delay timing.
- Startup: open-loop align + ramp (example params: 10% duty, 50→500 RPM ramp);
  "advanced startup" hands off to closed loop after N consecutive matches
  between forced commutation and detected BEMF events.
- Control loop runs once per PWM period, ADC end-of-conversion interrupt.

### sprabn7 — InstaSPIN-BLDC lab

- Same flux-integration concept with a GUI tuning flow: watch integrated flux
  (sawtooth), BEMF, phase current; lower threshold = earlier commutation; tune
  until commutation sits at the BEMF inflection. Good template for co-sim
  threshold-sweep experiments.
- Notes on why sensorless is stable at speed and not at low speed.

### spraby9 — sensorless FOC on this exact EVM family

- **Incremental build levels are the best template for the co-sim test plan**:
  (1) SVPWM open-loop waveform check → (2) Clarke/Park verification →
  (3) offset calibration → (4) speed measurement → (5) closed current loops →
  (6) observer open-loop → (7) full closed-loop speed FOC.
- Two low-side shunts, third current reconstructed (ia+ib+ic=0); ADC sampled at
  a fixed point in the PWM period; 10 kHz PWM in the lab.
- Performs DRV8301 SPI setup (gain) and DC_CAL offset capture at startup.

### slva959b / slvaf66 — layout & system (relevant subset for board-to-board wiring)

- Keep ADC/sense returns separate from power returns; star-ground the
  inter-board harness; twisted pairs for analog sense lines between boards.
- Expect inrush into ~1.7 mF bus capacitance — current-limited supply handles
  bench phase; note for any future hard-switch power application.
- Gate resistor / snubber tradeoffs (dv/dt vs ringing) matter if phase-node
  ringing shows up; bootstrap caps need PWM ≥ ~20 kHz to stay topped up.

---

## 9. Cross-cutting implications

For the Verilog controller:

- SPI master must do DRV8301 **mode-1** timing, 16-bit frames, N+1 pipelined
  responses; separate mode-0/1 path for MCP3208 (modes 0,0/1,1) — a shared bus
  is possible but the modes differ; separate CS regardless.
- 6-PWM outputs with RTL-enforced dead time (DTC adds a 50–500 ns hardware
  floor); EN_GATE state machine with 10 ms post-enable wait and the <10 µs
  quick-reset fault-clear pulse; register-refresh watchdog against brownout
  register resets (slva552).
- ADC scheduling is the binding constraint: ~50 kSPS aggregate at 3.3 V — plan
  which channels are sampled in which PWM period.

For the plant model (C++ primary, Modelica oracle — see [architecture](architecture.md)):

- Trapezoidal BEMF shape with correct floating-phase terminal voltage during
  PWM off-time (this is what the BEMF-integration controller actually sees),
  dead-time + body-diode conduction effects, 2 mΩ shunts + amp gain + 1.65 V
  offset for current channels, ≈1:10.5 EMF dividers with ~1.6 kHz RC, 12-bit
  quantization at the modeled sample schedule.
- Motor params (R, L, Ke, J, poles) still needed from the actual motor — not in
  any of these docs.

Open items / discrepancies found while reading:

1. OC_ADJ_SET power-on default not confirmed → SPI read at first bring-up.
2. ZONRI shunt value, EMF/bus divider values, DTC resistor, logic-rail
   availability → measure on the physical board.
3. EVM bus-divider math (1:6.4 at 60 V > 3.3 V ADC) doesn't close → re-read
   schematic p.6 or measure.
4. Dock header pin↔ball map and oscillator frequency → confirm from Sipeed wiki
   table before writing constraints.
5. GW5A pin behavior during configuration unresolved → external pull-down on
   EN_GATE is mandatory either way.
6. ~~MCP3208 at 3.3 V vs sampling budget → decide 5 V ADC domain (with
   TXB0108) vs reduced sample schedule.~~ Resolved 2026-06-12: 3.3 V domain
   with a sector-aware, PWM-synchronized schedule — see
   [architecture](architecture.md).
7. README "Early Hardware Notes" FPGA part caveat can be updated: PG138 does
   not exist for GW5A-25; the part is GW5A-LV25MG121 (confirm physical marking).
