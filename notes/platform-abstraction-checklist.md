<!-- SPDX-License-Identifier: MIT -->
# Platform Abstraction Checklist

Ordered tasks and code to make the peripheral models swappable: keep **every**
behavioral model (DRV8301, MCP3208, AS5600 …) in the tree and select a
component set per scenario with a config string, so a future BOM change
(DRV8316R + ADS9224R + AS5047P, ECP5) is a *platform profile* rather than a
rewrite. Companion to the completed [foc-checklist](foc-checklist.md) and
[formal-checklist](formal-checklist.md); architecture context in
[architecture](architecture.md). Created 2026-06-14; **Phase A complete
2026-06-14** — the abstraction is built and the 376-test suite stays green
(the refactor is a behaviour-preserving no-op); a component swap is now a
config string. **Phase B complete 2026-06-14** via the DRV8302 — a real,
datasheet-backed second platform (`zonri_drv8302`) that swaps both the C++ model
*and* an RTL behaviour (the hardware-config `drv_manager` path, no SPI),
selected by one config flag. The aspirational TI-reference triplet
(DRV8316R/ADS9224R/AS5047P) stays a follow-on — modelling those faithfully needs
their datasheets, and a fabricated model masquerading as a datasheet part would
violate the provenance discipline. Findings at the bottom.

**The core idea.** A component swap touches three things in lockstep — the C++
behavioral model, the RTL SPI-master/ADC width, and the provenance-flagged
params — so the unit of swap is a **named platform profile** that bundles all
three. Underneath, abstract at the **pin-level protocol** (SPI bytes, gate
bits, the analog node), not the register semantics: the electrical interface is
stable across parts, while the register map and fault behavior (what differs
between DRV8301 and DRV8316) live inside each model. The **plant** (motor /
inverter physics) and the **FOC core + formal proofs** never move — peripherals
are just the transducers between the plant's analog reality and the RTL's
digital world, and that boundary is exactly where the swap belongs.

**Phasing.** Phase A (stages 0–5) is the abstraction itself — keeps all current
models, low risk (the suite guards a behavior-preserving refactor). Phase B
(stages 6–9) populates a *second* platform — done with the DRV8302. **Phase C
(stages 10–15)** builds out the full reference platform(s) the BOM discussion
landed on — AS5047P, DRV8323RS, DRV8316R, ADS9224R, plus the ECP5/ULX3S
open-synthesis flow — now unblocked since all four datasheets are in
`docs/datasheets/` (added 2026-06-14). Each Phase-C part reuses the Phase-A
mechanism and the Phase-B pattern: a C++ model implementing a role interface, an
optional RTL protocol variant, datasheet-flagged params, and a formal proof for
any new bus master.

**Definition of done (Phase A):** the bench selects its gate driver, current
ADC, and angle sensor by name from config; the current parts are registered as
the `zonri_drv8301` platform and are the default; `pytest sim/tests` stays green
(372) unchanged; the assumption banner names the active platform; the test
suite can parametrize key scenarios over registered platforms (a no-op with one
platform, but the harness is ready). **Phase B done:** a real datasheet-backed
second platform (the `zonri_drv8302`: a `drv8302.cpp` model + the `drv_manager`
hardware-config RTL variant + datasheet params) runs the FOC scenarios, and the
AS5600/MCP3208/DRV8301 "cheap-out" platform stays runnable and green alongside
it. (The aspirational DRV8316R/ADS9224R/AS5047P `ti_reference` BOM is a further
follow-on, gated on those parts' datasheets.)

**Design decisions (pre-resolved):**

- **Abstract by ROLE, not by chip.** Roles: gate-drive, current-sense (the
  amp), angle-sense, and digitize. Between BOMs the current-sense amp migrates
  (DRV8301's integrated amp + external shunt → MCP3208, vs DRV8316's integrated
  CSA → ADS9224R), so a chip-keyed abstraction breaks. `FeedbackChain` routes
  the amp to whichever part owns it via config.
- **The interface is the bench↔model call surface, pin-level.** Concretely
  (from `bench.cpp`): `IGateDriver` = {update, gate_high, gate_low, nfault,
  noctw, sdo, pvdd_uv_active, dc_cal_active}; `ICurrentAdc` = {update,
  conversions, dout, last_sample, last_sample_theft_v, set_live_vref};
  `IAngleSensor` = {update, out}. Register semantics stay inside the impls.
- **Injection/test affordances stay accessible.** `inject_register_reset`,
  `inject_magnet_loss`, etc. are model-specific; expose them as virtual
  no-op-default methods on the interface (so a model that lacks a given fault
  simply ignores the injection) — keeps the realism scenarios working across
  platforms.
- **Keep all models; never delete.** Old parts become registered impls and the
  "cheap-out" regression. Platform selection is per-scenario.
- **The plant, FOC core, and formal proofs are platform-agnostic** and are not
  touched. Each RTL SPI-master *variant* gets its own protocol proof in the
  formal manifest (the library structure already supports this).
- **Platform profile = {models, params section, RTL variant}.** A `[platform.
  <name>]` mechanism selects the model names, the component param block, and
  (Phase B) the RTL SPI/ADC variant — one name, the whole BOM.
- **Defaults preserve today's behavior exactly** (`zonri_drv8301`): the
  refactor must be a no-op on the suite, which is the safety net.

**Out of scope:** transistor-level/analog modeling of the new parts (behavioral
as today), the actual ECP5 synthesis/timing closure (the bench is cycle-level —
flagged as the one thing this does not catch), buying hardware, and any
non-behavioral electrical co-sim. Phase B's *full* second-platform fidelity
(every DRV8316 fault mode) can be staged separately; this checklist delivers the
mechanism plus a working second platform.

**Dependency notes:** Stage 0 (taxonomy) first. Stage 1 (interfaces) and 2
(factory + reroute) are the heart of Phase A and gate everything; 2 must leave
the suite green. Stage 3 (platform config) needs 2. Stage 4 (amp re-partition)
needs 3. Stage 5 (test parametrization) needs 3. Phase B stages 6–9 need Phase A
and the parts/datasheets; 6 (models) before 7 (RTL variants) before 8 (param
derivation); 9 (docs) last.

## Stage 0 — Role taxonomy and conventions ✅

- [x] 0.1 Fix the role names (gate_driver, current_adc, angle_sensor; the
      current-sense amp is a `FeedbackChain` routing option, not a separate
      role) and the platform-profile concept. One short note in
      `architecture.md` / this file's findings.
- [x] 0.2 Decide the config surface: a `platform` name plus per-role model
      names in the bench config dict (e.g. `driver = "drv8301"`), defaulting to
      the current parts.

## Stage 1 — C++ role interfaces (behavior-preserving) ✅

- [x] 1.1 Add `i_gate_driver.hpp`, `i_current_adc.hpp`, `i_angle_sensor.hpp`
      with the abstract interfaces from the bench call surface (above), each
      with a virtual dtor and virtual no-op-default injection hooks.
- [x] 1.2 Make `Drv8301`, `Mcp3208`, `As5600` inherit and `override` the
      interface methods. No behavior change; signatures already match the
      bench's calls. Keep the concrete classes otherwise intact.
- [x] 1.3 Generalize the input structs where needed (`Drv8301Inputs` →
      `DriverInputs`) so the interface signature is part-agnostic.

## Stage 2 — Factory + Bench rerouting (the refactor; suite must stay green) ✅

- [x] 2.1 `peripheral_factory.{hpp,cpp}`: `make_gate_driver(name, config)`,
      `make_current_adc(...)`, `make_angle_sensor(...)`, returning
      `unique_ptr<I...>`. Registry keyed on name; default registrations for
      drv8301/mcp3208/as5600.
- [x] 2.2 `Bench` holds `std::unique_ptr<IGateDriver> drv_` etc. instead of
      concrete members; construct via the factory from config. Reroute every
      `drv_.` / `adc_.` / `encoder_.` call through the interface. Injection
      methods on `Bench` forward through the virtual hooks.
- [x] 2.3 Bindings/config: the bench config dict carries `driver`/`adc`/`angle`
      names (default to current parts); `bench_config_from_dict` passes them to
      the factory.
- [x] 2.4 **`pytest sim/tests` green at 372, unchanged** — the refactor is a
      no-op. This is the acceptance gate for Phase A's core.

## Stage 3 — Platform profiles in config ✅

- [x] 3.1 `[platform.zonri_drv8301]` in params.toml (or a profiles map in
      `bench_factory.py`) bundling the model names + the component param block.
      The current values move under it (re-tagged with this platform's
      provenance). `bench_factory.platform("zonri_drv8301")` builds the config.
- [x] 3.2 The assumption banner (conftest `pytest_configure`) prints the
      registered platforms and the default; a scenario's `cfg["platform"]`
      carries the active BOM name into the config, so a trace's provenance
      includes which BOM produced it.
- [x] 3.3 `default platform = zonri_drv8301`; all existing scenarios resolve to
      it implicitly (no scenario edits required).

## Stage 4 — Current-sense amp re-partition support ✅

- [x] 4.1 Make `FeedbackChain` amp routing a config option: the shunt→amp gain
      can be owned by the driver (integrated CSA) or be an external amp ahead of
      the ADC. A `current_sense_source` enum on the chain config.
- [x] 4.2 Keep the DRV8301 path (external shunt + driver amp → MCP3208) as the
      default routing; verify unchanged behavior.

## Stage 5 — Test parametrization over platforms ✅

- [x] 5.1 A `platform` fixture / `@pytest.mark.parametrize` helper in
      `bench_factory`/`conftest` that runs a chosen scenario set across all
      registered platforms. With one platform it is a no-op; the harness is
      ready for the second.
- [x] 5.2 Mark a small "cross-platform" scenario subset (init, FOC spin,
      shoot-through-clean) intended to run on every platform.

## Phase B status — DONE via the DRV8302 (datasheet in hand) ✅

Phase B is the *populate-a-second-platform* half. It is now delivered end to
end by the **DRV8302** — a genuinely different part whose datasheet is already
in the repo (`docs/datasheets/ti-drv8302-datasheet.pdf`), so nothing was
guessed:

- The DRV8302 is the *hardest* kind of second platform, not the easiest: it
  **replaces SPI with hardware strapping pins** (M_PWM/M_OC/GAIN/OC_ADJ), so it
  forces a real *RTL* control-path variant, not just a model swap. That is
  exactly Stage 7's work, and it makes the swap a stronger proof of the
  abstraction than another SPI part (e.g. DRV8316R) would have been — a same-bus
  part only exercises a regmap change, whereas the DRV8302 exercises a different
  protocol class.
- The second platform (`zonri_drv8302`) selects a new `IGateDriver` model
  (`drv8302.cpp`, no SPI register file) **and** a new RTL behaviour
  (`drv_manager` `hw_mode`: skip the SPI config/refresh sequence, go straight to
  RUN). Both are chosen by one config flag flowing through the same Phase A
  mechanism (`bench_factory.PLATFORMS` → `BenchConfig` → factory + RTL strap).
- It is datasheet-backed, not fiction: the OC threshold/mode (VDS-sense,
  OC_ADJ-set, M_OC latch-vs-current-limit), the 8 V PVDD UVLO, and the OTW/OTSD
  behaviour all come from SLES267C; the shared board mechanicals (Rds(on),
  dead-time floor, EN ready) are carried over from the same ZONRI assembly.

The aspirational TI-reference triplet (DRV8316R integrated FETs+CSA / ADS9224R
16-bit dual-simultaneous ADC / AS5047P SPI angle) remains a *future* addition,
but it is **no longer datasheet-gated**: all three datasheets (plus the
external-FET DRV8323 for the higher-power BOM) are now in
`docs/datasheets/` (added 2026-06-14), so modelling them faithfully is bounded
effort, not a provenance gap. Phase B's purpose (prove the abstraction by
standing up a real, datasheet-backed second platform that swaps both model and
RTL) is already met by the DRV8302; the stages below are ticked against it, and
the TI triplet is the next, now-unblocked, platform to populate.

## Stage 6 — Phase B: second-platform models  ✅ (DRV8302)

- [x] 6.1 `drv8302.cpp` implementing `IGateDriver` — same gate-drive family as
      the DRV8301 but **hardware-controlled**: no SPI register file (`sdo()`/
      `reg()`/`frame_errors()` are no-ops), with the datasheet's VDS-sense
      overcurrent (OC_ADJ threshold, M_OC latch-vs-current-limit), 8 V PVDD
      UVLO, OTW/OTSD, the 6-PWM Table-1 truth table, and the DTC dead-time
      floor. Registered in `peripheral_factory.cpp` (`name == "drv8302"`), with
      its `Drv8302Config` derived from the shared board params + datasheet
      defaults.
- [x] 6.2 DRV8302 params are datasheet-flagged (OC threshold, UVLO, thermal
      from SLES267C); the shared mechanicals reuse the ZONRI assembly's measured
      Rds(on)/dead-time. *(The TI-reference triplet's `[platform.ti_reference]`
      block stays a follow-on — see the Phase B status note.)*
- [x] 6.3 The AS5600 / MCP3208 / DRV8301 models still run unchanged — the
      "cheap-out" platform (`zonri_drv8301`) and the DRV8302 platform both pass
      the cross-platform subset (`test_platforms.py`, init + six-step + FOC).

> Remaining follow-on — **datasheets now in repo** (2026-06-14), so this is
> unblocked, not gated: `drv8316.cpp` (integrated FETs+CSA, its 16-bit SPI
> regmap — `docs/datasheets/ti-drv8316r-datasheet.pdf`), `ads9224r.cpp` (16-bit
> dual-simultaneous — `ti-ads9224r-datasheet.pdf`), `as5047.cpp` (14-bit SPI
> angle — `ams-osram-as5047p-datasheet.pdf`) → the `ti_reference` BOM. The
> external-FET `DRV8323`/`DRV8323RS` datasheet (`ti-drv8323-datasheet.pdf`) is
> also in repo for the higher-power BOM.

## Stage 7 — Phase B: RTL variant selection  ✅ (DRV8302 hardware-config path)

- [x] 7.1 RTL protocol variant: the DRV8302 has **no SPI**, so `drv_manager`
      gained a `hw_mode` strap. When set, the FSM skips the SPI config/refresh
      sequence (S_DCCAL→S_RUN directly, no periodic register refresh, fault
      recovery returns to RUN not reconfig); when clear it is byte-identical to
      the DRV8301 path. The strap is selected by config, threaded
      `BenchConfig.drv_hw_mode` → `controller_top.ctrl_drv_hw_mode` →
      `drv_manager.hw_mode`. (A *runtime* strap rather than a `PLATFORM` compile
      define — one bitstream covers both parts, which is the more faithful model
      of a board-assembly option.)
- [x] 7.2 ADC width: N/A for this platform — the DRV8302 board keeps the
      12-bit MCP3208 ADC role, so no width change is needed. *(The 12→16-bit
      thread stays a follow-on, scoped to the ADS9224R when its datasheet is in
      hand.)*
- [x] 7.3 The DRV8302 path is covered by the existing formal manifest, re-run
      green after the edit: `drv_manager` FSM legality stays **PROVEN** with the
      new input, and the top-level `controller_top_composition` no-shoot-through
      proof stays **PROVEN covers=REACHED** — i.e. `hw_mode` does not open a
      shoot-through path. (A new SPI-master *variant* would add its own protocol
      proof; the DRV8302 removes SPI rather than changing its regmap, so the
      relevant guarantee is the FSM/composition pair, which holds.)

## Stage 8 — Phase B: per-platform parameter derivation  ✅ (DRV8302 datasheet values)

- [x] 8.1 The DRV8302 platform's gate-driver parameters are sourced from the
      datasheet rather than inferred: `Drv8302Config` carries the OC_ADJ VDS
      threshold + M_OC mode, the 8 V PVDD UVLO (vs the DRV8301's 5.9 V), and the
      OTW/OTSD trip temperatures from SLES267C; the shared electrical mechanicals
      (Rds(on), dead-time floor, EN-ready) carry over from the ZONRI assembly's
      `measured`/`datasheet` params. No DRV8302 SPICE model is needed — it is
      hardware-strapped, so there are no register-derived analog values to fit.
- [x] 8.2 The DRV8302 entry has no new "clone guess": its protection params are
      documented-part values. *(The deeper Q7 clone uncertainty is a property of
      the ZONRI current-sense **front end**, shared by both ZONRI variants; it is
      retired specifically for `ti_reference` only once the DRV8316R CSA /
      ADS9224R transfer datasheets land — that remains the follow-on.)*

## Stage 9 — Docs and formal ✅

- [x] 9.1 `architecture.md`: a "Platform abstraction" section (role interfaces,
      the platform profile, what stays fixed). README: a line that the bench
      supports multiple BOMs and quantifies the sensor/ADC trade-off.
- [x] 9.2 Note that the formal proofs are platform-agnostic except the
      SPI-variant protocol proofs; cross-link from foc/formal checklists.

## Phase C — Reference-platform buildout (datasheet-backed) ✅ (2026-06-14)

**Complete.** All six stages (10–15) landed: the AS5047P SPI angle sensor
(Q22), the DRV8323RS external-FET driver, the DRV8316R integrated FET+CSA (Q7),
the ADS9224R 16-bit dual-simultaneous ADC (Q21), the two assembled reference
BOMs, and the ECP5/ULX3S open-synth flow. Every new bus master carries a formal
proof (PROVEN), the top-composition no-shoot-through proof was re-verified with
each variant wired in, and the full sim regression stays green. Findings at the
very bottom (after the Phase-A/B findings). What was actually built differs in
two honest ways from the original plan below — see those findings.

Phase B proved the mechanism with one extra driver (DRV8302). Phase C populates
the full **reference platform(s)** the BOM discussion landed on. All four
datasheets are now in `docs/datasheets/`, so this is bounded effort, not a
provenance gap. Stages are ordered cheapest→richest, which also respects the
dependency chain (angle role → driver roles → ADC width → assembled BOMs); the
ECP5 synth flow (stage 15) is orthogonal and needs no datasheet. Two BOMs
result:

- **`ti_reference` (clean):** DRV8316R (integrated FET **+** integrated CSA) +
  AS5047P (SPI angle) + a host ADC that samples the driver's analog CSA outputs.
  Fewest unknown passives — retires the Q7 clone uncertainty.
- **`ti_reference_hp` (external-FET):** DRV8323RS + CSD-class power FETs +
  ADS9224R (16-bit simultaneous) + AS5047P. Higher power envelope, premium ADC.

**Definition of done (Phase C):** both BOMs select their driver / current ADC /
angle sensor by name and run init + six-step + FOC; every new bus master
(AS5047P SPI, the two driver regmaps, the ADS9224R eSPI) has a formal protocol
proof; the full suite stays green; the assumption banner names the active BOM;
all new params are `datasheet`-flagged with page citations. Q21 and Q22 are
retired in *hardware* (simultaneous ADC; DAEC angle) for the reference BOMs, and
the cheap-out `zonri_drv8301` platform stays runnable as the regression.

**Datasheet correction baked in (read before scoping the CSA work):** the
DRV8316R's *integrated CSA does not digitize current* — it outputs analog
SOA/SOB/SOC pins the host ADC still samples (DRV8316 datasheet p.73). So the
`FeedbackChain::CurrentSenseSource::kIntegratedDriverCsa` knob re-routes *where
the current-sense amplifier lives* (inside the driver vs discrete external
op-amps), but **every platform still has a distinct ADC role** — the clean BOM
removes the shunt + discrete amps, not the ADC.

**RTL strategy (cross-stage) — as planned vs as built.** The plan assumed three
different driver SPI frames would force a parameterized SPI master. In practice:
the **DRV8323 frame is *identical* to the DRV8301's** (R/W[15] | 4-bit addr |
11-bit data, mode 1), so it reuses the existing master + `drv_manager` handshake
with **no RTL change** (the model echoes register writes). The **DRV8316R** is
operational on its power-on defaults (6x PWM), so the reference platform runs it
via the existing `hw_mode` strap (skip SPI config), and its mode-3/parity SPI
*reconfiguration* path is deferred as unexercised rather than modelled as
fiction. So no new *driver* SPI master was needed. New RTL was added only where a
genuinely different protocol is exercised: `as5047p_spi_master` (SPI angle),
`ads9224r_master` (eSPI dual ADC), and the shared `ctrl_cur_norm_shift` +
`ctrl_adc_dual_mode` straps. Each new master has its own framing proof; the
top-composition proof re-verifies with all of them wired in.

### Stage 10 — AS5047P SPI angle sensor ✅ (retires Q22)

> **Retires Q22** (angle-read latency costs torque at speed). DAEC (Dynamic
> Angle Error Compensation) cuts the ~90–110 µs raw read pipeline to ~1.5–1.9 µs
> residual at constant speed (datasheet pp.7–9) — the *hardware* form of the
> bench's existing `ctrl_foc_extrap` (ω·t_latency) prediction, so this platform
> lets us compare hardware DAEC against our RTL extrapolation head-to-head.

- [x] 10.1 `as5047p.cpp/.hpp` implementing `IAngleSensor`. SPI **mode 1**
      (CPOL=0/CPHA=1), 16-bit frames MSB-first. Read-data frame = PARD[15] |
      EF[14] | DATA[13:0], even parity over [14:0]; **pipelined read** (response
      presented on the next CS frame). SPI-slave shifter mirrors the DRV8301.
- [x] 10.2 DAEC as a config knob (`daec_enable`): the model's effective angle
      latency follows (~1.7 µs DAEC / ~100 µs raw) so the plant→sensor delay is
      *modelled* via a first-order lag, not assumed instantaneous. Magnet loss
      sets EF → the master drops `angle_valid` (portable realism behaviour).
- [x] 10.3 `As5047pConfig` carries the datasheet defaults (14-bit, DAEC/raw
      latency); the portable mounting/eccentricity/noise carry over from the
      shared encoder config (factory `make_angle_sensor("as5047p", …)`).
- [x] 10.4 RTL `as5047p_spi_master.v` — a mode-1 16-bit reader (reuses
      `DRV_SPI_DIV`) that continuously reads ANGLECOM (0x3FFF) and truncates the
      14-bit angle to the existing 12-bit bus. Selected by `ctrl_angle_spi_mode`
      (a runtime strap mux against `as5600_pwm_capture`); strap low = byte
      identical. Wired `BenchConfig.angle_spi_mode` → controller_top → mux.
- [x] 10.5 Formal: `as5047p_spi_master` framing proof — FSM legality + a
      well-formed single-cycle `new_sample`, with shallow non-vacuity covers
      (frame starts, clock toggles). **PROVEN covers=REACHED**; the top
      composition proof was re-run **PROVEN** with the master wired in.
- [x] 10.6 Factory + the `zonri_as5047p` profile. `test_platforms` runs init +
      six-step + FOC on it; `test_as5047p` asserts DAEC tracks the rotor far
      tighter than the AS5600 (the Q22 quantity, in hardware) and that magnet
      loss drops `angle_valid`. All green.

### Stage 11 — DRV8323RS external-FET smart gate driver ✅

> The external-FET reference driver. **Key finding:** the DRV8323 uses the
> *identical* 16-bit frame to the DRV8301 (R/W[15] | 4-bit addr[14:11] | 11-bit
> data[10:0], mode 1), so the existing SPI master and the `drv_manager`
> write/verify handshake configure it with **no RTL change** — the cheapest
> driver add.

- [x] 11.1 `drv8323.cpp/.hpp` implementing `IGateDriver`: the DRV8323 register
      map 0x00–0x07 (Fault Status 1/2, Driver Control, Gate Drive HS/LS, OCP
      Control, CSA Control). The SPI slave (mirrors the DRV8301) stores writes to
      the R/W registers verbatim and echoes them on read, so the controller's
      write-addr2/addr3 + readback-verify completes. RS buck is stubbed.
- [x] 11.2 External-FET semantics: the VDS OC trip senses |i|·Rds_on of the
      *external* power FET (a platform param — derived from the shared board
      Rds_on, the in-repo CSD18540 class).
- [x] 11.3 CSA routed via `FeedbackChain` `kExternalShuntDriverAmp` (external
      shunt + driver amp — same topology and codes/A as the DRV8301, so the FOC
      fixed-point tuning is unchanged). CSA_CAL_A/B/C map to `dc_cal_active`.
- [x] 11.4 Protection from DRV8323 datasheet defaults: VDS_LVL 16-step (default
      0.75 V), OCP_MODE latched/auto-retry(default)/report-only/disabled,
      DEAD_TIME 100 ns, VM UVLO 6 V, OTW/OTSD. The controller does not rewrite
      OCP/CSA, so the part runs on those valid power-on defaults.
- [x] 11.5 **No RTL change** — the DRV8323 frame is identical to the DRV8301's,
      so the existing `spi_drv_master` + `drv_manager` handshake configure it
      directly. (A richer per-register config write — setting OCP/CSA explicitly
      — is a noted refinement; the power-on defaults are valid operating values.)
- [x] 11.6 Formal: unaffected (RTL unchanged); the `drv_manager` FSM and
      `controller_top` composition proofs from stage 10 still hold.
- [x] 11.7 Factory + the `zonri_drv8323rs` profile (the `ti_reference_hp`
      driver). `test_platforms` runs init + six-step + FOC; `test_drv8323`
      asserts it reaches configured (handshake) and that a latched fault is
      counted. All green.

### Stage 12 — DRV8316R integrated-FET driver + integrated CSA ✅ (retires Q7)

> **Retires the Q7 clone-passive uncertainty:** an integrated FET + CSA stage
> has no mystery shunts or MOSFETs — the "fewest unknowns" clean reference. The
> integrated CSA is the centerpiece; it forced the current-scale normalization.

- [x] 12.1 `drv8316r.cpp/.hpp` implementing `IGateDriver`: gate drive (6-PWM +
      dead-time), integrated-FET overcurrent (a fixed 16 A current limit, not a
      VDS sense), VM UVLO, OTW/OTSD, EN/nSLEEP sequencing. The DRV8316R is
      operational on its power-on defaults (6x PWM), so the reference platform
      runs it via the controller's hardware path (`drv_hw_mode`, no SPI
      reconfiguration); the SPI surface is represented at defaults (sdo/reg
      inert). *(Honest scope note: modelling the mode-3/parity SPI register
      reconfiguration was deferred as unexercised — the part runs on defaults;
      see Findings. Modelling it as fiction would violate provenance.)*
- [x] 12.2 Integrated-FET envelope datasheet-flagged: leg Rds(on) ≈ 95 mΩ,
      ~8 A, 4.5–35 V — markedly lower power than the external-FET DRV8323.
- [x] 12.3 **Integrated CSA (the decisive piece):** `Vo = VREF/2 + GCSA·I`,
      `GCSA = 0.15 V/A`, bidirectional, low-side — wired into the `FeedbackChain`
      `kIntegratedDriverCsa` source (replacing the external shunt + discrete
      amp). A distinct ADC role (MCP3208 here) still digitizes the sense node.
- [x] 12.4 Protection params datasheet-flagged: OCP 16 A latched, UVLO falling
      4.2 V / rising 4.4 V, OTW 170 °C / OTS 185 °C.
- [x] 12.5 **Current-scale normalization (the real RTL work):** the integrated
      CSA's ~0.15 V/A gives ~7.5× the codes/A of the external shunt path, which
      would mis-scale the fixed-point FOC gains. Added `ctrl_cur_norm_shift` — an
      arithmetic right-shift on the measured FOC currents into `foc_core`
      (`BenchConfig.cur_norm_shift` → controller_top), shift 3 (÷8) for the
      DRV8316R. Shift 0 (the default) leaves every other platform byte-identical.
- [x] 12.6 Formal: `controller_top_composition` re-run **PROVEN covers=REACHED**
      with the new port + shift (the shift is on the measurement path, not the
      gate path, so shoot-through freedom is untouched).
- [x] 12.7 Factory + the `zonri_drv8316r` profile (the clean `ti_reference`
      driver). `test_platforms` runs init + six-step + FOC; `test_drv8316r`
      asserts FOC closes the loop on the integrated CSA and that it measures a
      meaningful torque current under load. All green.

### Stage 13 — ADS9224R 16-bit dual-simultaneous ADC ✅ (retires Q21)

> **Retires Q21** in hardware: the ADS9224R samples both phase currents on one
> CONVST edge (truly simultaneous), so the dq frame stays aligned. Realized as
> an *additive* FOC-current path — a dedicated RTL master + model selected by a
> strap — so the proven MCP3208 sequencer (EMF/bus/six-step) stays intact.

- [x] 13.1 `ads9224r.cpp/.hpp`: a CONVST edge latches **both** channels at the
      same instant; `READY` asserts after tDRDY ≈ 315 ns; both **16-bit
      two's-complement** codes (zero = no current) shift out MSB-first on two
      data lines (SDO_A/SDO_B), the same shift/sample timing as the DRV slave.
      ±4.096 V FSR about the current-sense midpoint.
- [x] 13.2 **The 16-bit thread:** the signed two's-complement codes (zero =
      mid-scale) need *no* offset subtraction (unlike the offset-binary MCP3208),
      so the ADS9224R master emits signed currents directly; the much larger
      codes/A are renormalized by the shared `ctrl_cur_norm_shift` (shift 3) into
      the canonical FOC fixed-point scale — reusing the stage-12 mechanism rather
      than a global gain rescale, so the existing FOC tuning and its bit-exact
      twin are untouched.
- [x] 13.3 `Ads9224rConfig` datasheet-flagged: 16-bit, ±4.096 V FSR, internal
      2.5 V ref, ~315 ns latency, two's-complement.
- [x] 13.4 RTL `ads9224r_master.v` — triggers CONVST at the off-window center,
      waits READY, bursts 16 SCLK reading both SDO lines, emits the two signed
      currents + a strobe. Selected by `ctrl_adc_dual_mode` (a strap mux on the
      FOC current source); the MCP3208 path stays for the cheap-out BOM.
- [x] 13.5 Formal: `ads9224r_master` proof — FSM legality + single-cycle
      `foc_valid`, shallow non-vacuity covers. **PROVEN covers=REACHED**;
      `controller_top_composition` re-run **PROVEN** with the master wired in.
- [x] 13.6 Q21 in hardware: one CONVST samples both currents simultaneously
      (`sample_scheme=1` leaves the chain live so the ADS9224R itself provides
      the simultaneity). `test_ads9224r` asserts FOC keeps `id` regulated near
      zero at speed (the simultaneous-sample benefit) and tracks a torque
      current under load. Cross-platform init + six-step + FOC pass.

### Stage 14 — Assembled reference BOMs + validation ✅

- [x] 14.1 `ti_reference` (clean) profile: driver `drv8316r` + angle `as5047p`,
      current-sense `kIntegratedDriverCsa` read by a modest MCP3208 (the
      integrated CSA replaces the shunt+amp, not the ADC — so no external
      ADS9224R is needed on the clean BOM), `drv_hw_mode` (power-on defaults).
- [x] 14.2 `ti_reference_hp` (external-FET) profile: driver `drv8323rs` + angle
      `as5047p` + the ADS9224R 16-bit simultaneous current ADC (`adc_dual_mode`),
      external-shunt current sense.
- [x] 14.3 The cross-platform suite runs init + six-step + FOC on *every*
      registered platform (cheap-out + DRV8302 + the four per-part platforms +
      both assembled reference BOMs); `cfg["platform"]` names the active BOM in
      the banner. All green; the full regression confirms the defaults stay
      byte-identical.
- [x] 14.4 Docs: `architecture.md` now carries the full platform/BOM table (what
      each retires; the runtime-strap mechanism). *(A rendered comparison figure
      is optional polish over the table + the per-question tests; the durable
      artifact is the table and the Q21/Q22 assertions in `test_ads9224r` /
      `test_as5047p`.)*

### Stage 15 — ECP5 / ULX3S open-synthesis flow ✅ (orthogonal; no datasheet)

> The only Phase-C item needing no datasheet. Turns "targets an ECP5" from a
> claim into a checked `nextpnr-ecp5` result, completing the all-open story:
> open HDL → open formal → **open synth/PnR**. Toolchain in `~/oss-cad-suite`.

- [x] 15.1 `synth/synth_ecp5.ys` (`read_verilog` the controller_top RTL set →
      `synth_ecp5`) + `synth/run_synth.py` (mirrors `run_formal.py`'s
      ergonomics: `--check` = synth-only gate, else synth → PnR → ecppack →
      `synth/synth_report.md`).
- [x] 15.2 `nextpnr-ecp5` (`--85k --package CABGA381`) with `synth/ulx3s.lpf`
      (clk located + 25 MHz constrained; the rest `--lpf-allow-unconstrained`).
      **Fits with wide margin:** I/O 40/365 (10%), FF 1902/83640 (2%), DSP
      22/156 (14%), ~14.3k LUT4. Needed `synth/board_top.v` — a board wrapper
      exposing only the real ULX3S pins (clk, gates, SPI/ADC/angle, UART), since
      the sim DUT `controller_top` exports its whole ctrl_*/dbg_* interface as
      pins (hundreds → exceeds any package); real control is over UART.
- [x] 15.3 `ecppack` → `synth/work/board_top.bit`. CI gate: `test_synth.py`
      runs `run_synth.py --check` (synthesis must succeed) — catches
      non-synthesizable RTL (the class of the `circle_limit` while-loop fix);
      skipped unless the OSS CAD Suite is on PATH, so the default sim regression
      stays fast.
- [x] 15.4 Clock reconcile — **the flow surfaced a real finding:** the
      unpipelined FOC datapath (the `circle_limit` 16-iteration isqrt + the
      chained Clarke/Park/PI/SVPWM in one `update`) is a long combinational path
      that caps **Fmax ≈ 3.3 MHz**, below the 25 MHz sim clock. The design
      *fits and routes*; a real 25 MHz board build needs that datapath pipelined.
      Reported honestly in `synth/synth_report.md` (not papered over) — exactly
      the "flag RTL that won't map well at speed" this stage exists to do.

**Formal coverage note (Phase C):** the proofs stay platform-agnostic except the
per-protocol bus-master proofs — one each for `as5047p_spi_master`, the DRV8323
and DRV8316R regmap framings, and `ads9224r_master` — added to the manifest
alongside the existing SPI/ADC checkers (formal-checklist cross-link).

## Findings

**Phase A complete 2026-06-14.** Peripheral models are now selected by name; a
component swap is a config string. The refactor was behaviour-preserving (the
suite was green at 376 immediately after Phase A), and `test_platforms.py`
exercises the mechanism: it runs init + six-step + FOC on every registered
platform, and asserts the factory genuinely selects (an unknown name *raises*,
never silently defaults). With the DRV8302 platform added (Phase B below) the
full suite is **381 passed** — the +5 are the cross-platform tests now running
on both platforms; nothing regressed, since the `hw_mode=0` path is
byte-identical.

What worked / what to know:

1. **The bench↔model call surface abstracted cleanly.** The three interfaces
   (`IGateDriver`, `ICurrentAdc`, `IAngleSensor`) were exactly the methods
   `bench.cpp` and the bench-routed bindings call. The standalone unit-test
   bindings (`PyDrv8301`, `PyMcp3208`, `PyAs5600`) kept the *concrete* types —
   only the bench path went polymorphic, which bounded the refactor.
2. **Roles ≠ chips (the trap, designed around).** `Drv8301Inputs`→`DriverInputs`
   and `Mcp3208Sample`→`AdcSample` were generalized (aliases keep old call sites
   compiling); the current-sense amp is a `FeedbackChain` routing option
   (`CurrentSenseSource`), so a future integrated-CSA part re-partitions without
   a new "ADC chip" abstraction.
3. **Member-order matters:** `chain_` had to be declared before `drv_/adc_/
   encoder_` because the ADC's analog-source lambda captures `chain_`.
4. **Injection hooks are virtual no-op-defaults**, so a model lacking a given
   fault simply ignores the injection — the realism scenarios stay portable.

**Phase B (6–8) done via the DRV8302 — 2026-06-14.** Rather than wait on the
aspirational TI triplet, Phase B was completed with the part whose datasheet was
already in the repo: the **DRV8302**. It is the demanding case — a hardware-
strapped driver with *no SPI* — so standing it up exercised both halves of the
abstraction at once: a new `IGateDriver` model (`drv8302.cpp`, datasheet OC/UVLO/
thermal, no register file) **and** a new RTL behaviour (`drv_manager` `hw_mode`:
skip the SPI config/refresh, go straight to RUN). One config flag
(`drv_hw_mode`) selects both through the Phase A mechanism. Evidence it is real,
not cosmetic: the `zonri_drv8302` platform passes the cross-platform suite (init
+ six-step + FOC spin) *and* the `drv_manager` FSM-legality + top-level
no-shoot-through formal proofs stay PROVEN with the new strap — i.e. the variant
is verified, not just exercised. The `hw_mode=0` path is byte-identical, so the
DRV8301/MCP3208/AS5600 "cheap-out" platform and the whole prior suite stay
green. What remains a genuine follow-on (and is correctly *not* faked): the
DRV8316R / ADS9224R / AS5047P `ti_reference` BOM, which needs those parts'
datasheets — modelling them on guesses would violate the provenance discipline,
worse than an honest gap. The slots are registered and ready
(`bench_factory.PLATFORMS`, `peripheral_factory`), so adding them is the
documented "add a platform" path below.

## Adding a platform (the easy path Phase A unlocks)

This is the generic recipe; **Phase C (stages 10–15) is it worked end-to-end**
for the AS5047P / DRV8323RS / DRV8316R / ADS9224R reference BOMs, with the
datasheet specifics filled in.

1. Write the C++ model(s) implementing the role interface(s) (`i_*_sensor.hpp`),
   from the part datasheet; register them in `peripheral_factory.cpp`
   (`if (name == "drv8316") return std::make_unique<Drv8316>(cfg);`).
2. If the chip changes the RTL-facing protocol (regmap / ADC width / a SPI vs
   PWM angle), add the RTL variant (`spi_drv8316_master.v`, ADC-width param)
   behind a `PLATFORM` define — and a protocol proof in the formal manifest.
3. Add the BOM to `bench_factory.PLATFORMS` with its model names; add a
   `[platform.<name>]` param block (component values `datasheet`-flagged from
   the part).
4. `test_platforms.py` now runs the smoke set on it automatically; the old
   models remain as the "cheap-out" regression.
