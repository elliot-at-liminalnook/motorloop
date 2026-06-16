<!-- SPDX-License-Identifier: MIT -->
# Edge-Case Scenario Catalog

Edge cases the system should be tested against, beyond the S0–S13 scenario
suite. Created 2026-06-12; **all entries covered 2026-06-12** — every spec
gap has a recorded policy (architecture.md "Edge-case policies"), every
scenario has a test in `sim/tests/test_edge_cases.py` (E-numbered), and the
chaos fuzz harness (section G) runs four seeded sequences. Coverage notes
and discovered limitations are appended at the bottom.

Original status legend (now historical):

- **testable** — bench machinery exists today, just write the scenario
- **needs-bench** — requires a small bench/model extension first
- **SPEC GAP** — the edge case exposes *undefined behavior*: an RTL design
  decision must be made before a test can assert anything. These are the
  most valuable entries — they are requirements discovered early.

## A. Timing and boundary edges

- **E1. Sector-boundary chatter.** Rotor parked exactly on a commutation
  boundary with angle noise: sector flaps at the noise rate, commutation
  thrashes, speed meter counts phantom edges. Hypothesis: gate churn within
  dead-time limits but speed reads garbage → PI disturbance. Suggest
  hysteresis on the sensored-sector compare. *(testable; likely SPEC GAP:
  sector hysteresis width)*
- **E2. Angle wraparound.** Rotor oscillating across the 4095→0 wrap (and
  the align-offset addition pushing elec12 across the wrap). Off-by-one in
  the modular arithmetic shows up as a 1-sector glitch once per rev.
  *(testable)*
- **E3. Duty extremes and the min-pulse boundary.** duty ∈ {0, 1, MINP−1,
  MINP, HALF−MINP, HALF}: pulse-snap behavior, complementary-gate behavior
  at full-on (low FET never conducts → bootstrap starvation on real
  hardware — not modeled, note it), and dead-time insertion at sliver
  pulses. *(testable)*
- **E4. EMF aperture collision at very high duty.** Above duty ≈ 0.97 the
  off-window is narrower than the sample aperture: the EMF hold lands in
  the on-window and reads the rail. Today the sequencer samples anyway.
  What should it do — skip, flag, or clamp duty? *(SPEC GAP)*
- **E5. Backdriven/reverse rotation.** The interface is
  positive-speed-only; speed_meter counts |edges| direction-blind. An
  overhauling load (or flipped phase wiring) spins the rotor backward:
  measured speed is positive, PI pushes harder, runaway. Needs a direction
  estimate (sector sequence sign) and a policy. *(SPEC GAP)*
- **E6. Mode/parameter changes mid-flight.** mode 2→1→2 while spinning;
  align-offset rewrite at speed (instant sector jump = torque reversal
  transient); target changes mid-SPI-frame, mid-ADC-conversion. Hypothesis:
  the sequencers tolerate it but the transient torque spike trips OC at
  real current scales. *(testable)*
- **E7. Closed-loop target = 0 and target > ceiling.** Is target 0 "active
  hold at zero" or "coast"? (Currently: PI output clamps to 0 duty = coast
  — is that wanted?) Target above the AS5600 latency ceiling: PI saturates
  forever — windup behavior on the way back down. *(half SPEC GAP, half
  testable)*

## B. Fault pile-ups and combinations

- **E8. Fault during fault-handling.** Register reset landing *during* the
  watchdog verify sequence; nFAULT asserting mid-config-write; quick-reset
  pulse colliding with the refresh timer. The drv_manager FSM has untested
  state-pair coverage here. *(testable — fault-injection timing sweep)*
- **E9. Magnet loss during startup ramp / during fault recovery.** Two
  recovery state machines interleaving (angle-valid gate vs drv_manager
  reconfigure). *(testable)*
- **E10. Brownout during the DC_CAL window.** Offsets captured while the
  rail sags are garbage and currently retained forever. Hypothesis: current
  readings permanently biased after one bad init. Likely needs offset
  plausibility bounds or re-cal-on-recovery. *(testable; probable SPEC GAP)*
- **E11. nFAULT line glitch storm.** Spurious fault pulses at rate: each
  triggers the full kill→quick-reset→reconfigure cycle (~ms each).
  Hypothesis: recovery storms starve gate drive; also **fault_count is
  8-bit and wraps**. *(testable; counter-saturation SPEC GAP)*
- **E12. Kitchen sink.** Thermal-hot (rds drifted) + sagging supply +
  disturbances + eccentricity simultaneously: interactions like OC trip
  current falling with rds(T) while the bus sag raises duty demand.
  *(testable)*

## C. Dead and degenerate peripherals

- **E13. DRV SPI unresponsive** (stuck MISO low/high — dead chip, unseated
  jumper). The watchdog rewrites and re-verifies forever; mismatch_count
  (8-bit) wraps; system livelocks silently instead of declaring the
  peripheral dead. Needs an unrecoverable-fault state + telemetry flag.
  *(SPEC GAP — dead-peripheral policy)*
- **E14. ADC absent/stuck.** Codes read 0 (or 4095) forever; nothing in
  the RTL sanity-checks them today (six-step survives since control is
  speed-only — but telemetry and future current limits silently lie).
  Plausibility window + flag. *(SPEC GAP)*
- **E15. AS5600 carrier drift to the validation boundary.** Carrier at
  ±24.9% vs ±25.1% of nominal: capture flips between valid/invalid at the
  boundary; closed loop alternates run/coast. Hysteresis on validity?
  *(testable; small SPEC GAP)*
- **E16. UART desync and partial frames.** A write command with only one
  data byte ever sent: **the regfile FSM waits in S_WHI forever** and every
  subsequent byte is misinterpreted — there is no timeout today. One torn
  frame poisons the link permanently. Needs an inter-byte timeout reset.
  *(SPEC GAP — confirmed by code inspection)*
- **E17. UART flood.** Back-to-back commands at line rate, reads
  interleaved with writes; read response colliding with the next command.
  *(testable)*

## D. Power sequencing

- **E18. Slow supply ramp at power-on.** Bus crawls up through the UVLO
  region while the RTL is already out of reset: EN_GATE asserted into a
  half-alive DRV, ready-timer starts, UV resets registers mid-config.
  Today's scenarios start with a healthy rail. *(testable — supply model
  needs a programmable v_set(t) ramp; small needs-bench item)*
- **E19. Higher-bus operation (24 V).** EMF divider output exceeds the
  3.3 V clamp at speed → EMF channels saturate (BEMF integration would be
  blind); regen pump headroom halves. Run all scenarios at 24 V and catalog
  what saturates. *(testable)*
- **E20. Marginal supply limit cycle.** i_limit set exactly at the
  spin-up boundary: does the system live-lock (brownout → recover → retry →
  brownout) indefinitely? Should the controller detect repeated UV resets
  and back off (retry budget)? *(testable; SPEC GAP — retry policy)*

## E. Mechanical and load edges

- **E21. Hard stall under closed loop.** Rotor blocked at speed: speed→0,
  PI winds to max duty, stall current flows indefinitely (winding cooks —
  thermal model now shows this). No stall detection exists. Policy: stall
  timeout → safe-off + flag? *(SPEC GAP — stall detection)*
- **E22. Load resonance with the speed loop.** Oscillating load torque
  swept through the speed-loop bandwidth and through cogging-order
  harmonics: find the speed bands where ripple amplifies. *(testable —
  needs a sinusoidal load-torque injection, trivial bench add)*
- **E23. 10x inertia / 0.1x inertia.** PI gains are tuned for placeholder
  J; bracket the stability envelope so motor-ID day knows whether retuning
  is mandatory or the margin is wide. *(testable)*
- **E24. 180-degree alignment error.** Magnet glued flipped: the
  calibration sweep (S10b) currently searches ±1 sector around nominal — a
  flipped magnet is 3 sectors away and reads as "all candidates bad".
  Full-circle calibration sweep needed for real bring-up. *(testable —
  extend the routine)*

## F. Numerical and long-run edges

- **E25. Counter wrap audit.** fault_count/mismatch_count (8-bit),
  noctw_count (16-bit), open-loop phase accumulator (by design), speed
  numerator at near-zero period (divide guards), bench cycle counter.
  Enumerate, decide wrap-vs-saturate per counter (saturate is almost always
  right for fault counters), test the boundaries. *(SPEC GAP for
  saturation policy, then testable)*
- **E26. theta grows unbounded.** The plant integrates theta without
  wrapping; sin(pp·theta) loses precision as theta grows (noticeable
  ~1e8 rad — days of sim time, but trivially cheap to fix by wrapping
  theta in the plant). *(needs-bench: wrap + invariance test)*
- **E27. Parameter-range integrator stability.** Real motors can have
  L = 10 µH (τ = L/R ≈ 20 µs): the 1 µs substep is marginal; at 1 µH it is
  unstable. The motor-ID day must not hand the sim parameters it silently
  cannot integrate: add a stability guard (substep vs L/R ratio assert in
  the bench constructor) + a documented valid envelope. *(needs-bench)*

## G. Approach recommendation: fuzz the combinations

The pairwise space of (mode transitions × faults × disturbances × timing)
is too large to enumerate by hand. The bench already has the right
ingredients for **seeded chaos scenarios**: always-on invariant checkers
(shoot-through, dead-time, bounded currents) plus deterministic seeds. A
fuzz harness that drives random-but-seeded sequences of mode changes,
target steps, injections, and glitch windows — asserting only the global
invariants plus "recovers to a sane state within T after the last
perturbation" — would cover B-category combinations far better than
hand-written pairs, and any failing seed replays exactly.

## Suggested order

1. The SPEC GAPS first (E4, E5, E10, E11/E25, E13, E14, E16, E20, E21) —
   each needs a design decision recorded in architecture.md before its
   test can exist; several are cheap RTL fixes (UART timeout, counter
   saturation).
2. The pure-testable entries grouped by run cost (E1–E3, E6–E9, E12, E15,
   E17, E19, E22–E24).
3. The needs-bench items (E18 supply ramp, E22 load injection, E26/E27
   numerics guards).
4. The fuzz harness (G) last — it pays off most once the spec gaps have
   defined what "correct" means under each perturbation.

## Coverage notes (2026-06-12)

All E1–E27 plus the fuzz harness are implemented in
`sim/tests/test_edge_cases.py`; the spec-gap policies live in params.toml
`[rtl]` and are recorded in architecture.md. Notable outcomes:

- **E5 limitation discovered:** direction detection is only valid inside
  the AS5600 tracking envelope (~±120 rad/s mech). Beyond it the sensor
  frame rate aliases the sector sequence and the direction vote flickers —
  at −430 rad/s the loop's behavior is unspecified-but-safe (no
  shoot-through; currents bounded). A backdrive that fast on real hardware
  is a mechanical emergency, not a control regime; the policy covers the
  trackable range and the fuzz invariants cover the rest.
- **E24 surprise:** with an even pole-pair count, a flipped magnet (π
  mechanical) aliases to ZERO electrical offset — the nominal alignment
  works unchanged. The full-circle calibration sweep is still the right
  bring-up procedure (odd-pole-pair motors and arbitrary mounting angles
  need it); the test verifies it finds a working offset.
- **E18 forced an RTL semantic:** a register-verify mismatch *while nFAULT
  is asserted* is a brownout in progress and routes to the fault path; only
  mismatches while otherwise healthy count toward the dead-peripheral
  declaration. Without this split, a slow supply ramp at boot declared the
  DRV dead before the rail was up.
- **E11/E20 interplay:** the lockout (4 recoveries without a healthy
  interval) is what makes nFAULT storms safe — the 8-bit counters can no
  longer be wrapped in practice, and the storm ends in an explicit
  host-acknowledged state.
- **E14/E10:** with an all-dead analog world the six-step loop itself keeps
  running (control is speed-based) — exactly why the stuck-ADC and
  offset-fault telemetry flags exist; nothing else would notice.
- **Recovery convention:** dead/lockout/stall all clear the same way — the
  host holds mode 0 for >100 ms. One acknowledgement gesture, three latched
  conditions. Corollary discovered by E13's test: a host that is *already*
  idle auto-acknowledges dead/lockout into a perpetual retry loop — the
  realistic dead-DRV scenario has the host actively requesting run mode.
- **Partial corruption defeats naive dead-detection** (E13 calibration):
  at moderate glitch rates most SPI frames still verify, so the
  consecutive-failure counter keeps resetting — only near-total corruption
  (~50% per-bit) latches DEAD quickly. Intermittent corruption presents as
  elevated mismatch counts instead; both signatures are now telemetry.
- **Stall detection needed an angle-motion discriminator** (found by E23):
  a feather-light rotor accelerates beyond the AS5600 tracking range,
  aliased speed reads zero at max duty, and the original detector
  false-fired on a spinning motor. A true stall freezes the *angle*; the
  detector now resets unless the angle holds still (~10 ms sampling).
- **E23 envelope result:** with placeholder gains the regulation envelope
  is roughly [0.5x, 10x] of design inertia — 0.5x regulates at +/-1.2
  rad/s, 0.3x is marginal (+/-11), 0.1x limit-cycles (bounded,
  fault-free). Retuning is mandatory below ~0.5x; recorded for motor-ID
  day.
- **A jam is not a load torque** (E21): torque overpowering the motor
  *backdrives* it (that is E5); only a position-constraining effect
  (modeled as dominant Coulomb friction) produces the speed-and-angle
  freeze of a true stall.
- **Verification-discipline lesson:** several "passing" results earlier in
  the day came from background-task summary notifications that did not
  match the output files, plus suites contaminated by concurrent module
  rebuilds. Final results are foreground, serial, and file-free: 29 + 126
  = 155 tests green. Trust synchronous output, never notification
  summaries.
