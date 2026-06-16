<!-- SPDX-License-Identifier: MIT -->
# Verification plan

The requirement → evidence map a lab audits: each safety/correctness claim, and
the proof or test that covers it. "Verification, not validation" — these close
the plant-independent half; silicon correlation (stage 5) is separate.

| # | Requirement | Evidence (proof / test) | Kind |
| --- | --- | --- | --- |
| R1 | No leg ever drives both gates (no shoot-through) | `pwm_generator`, `lib_no_shoot_through`, `controller_top_composition` | formal, PROVEN |
| R2 | Dead-time enforced between complementary edges | `pwm_deadtime` | formal, PROVEN |
| R3 | Gates off during/after reset (no power-on glitch) | `pwm_reset` | formal, PROVEN |
| R4 | Current-PI / speed-PI outputs stay clamped (no runaway) | `current_pi`, `speed_iq_pi` (parameter-generic) | formal, PROVEN (envelope) |
| R5 | SVPWM never exceeds `[0,HALF]` per leg | `svpwm` | formal, PROVEN |
| R6 | Voltage vector bounded to the inscribed circle | `circle_limit` | DOCUMENTED (isqrt intractable for OSS SMT) |
| R7 | Gate-driver FSM never enters an illegal state | `drv_manager` | formal, PROVEN |
| R8 | ADC/angle bus masters are well-framed (1 event/sample) | `adc_sequencer`, `as5047p_spi_master`, `ads9224r_master` | formal, PROVEN |
| R9 | FOC math matches the fixed-point reference bit-for-bit | `test_foc_math.py` | sim, bit-exact |
| R10 | Closed-loop FOC + six-step spin on every BOM | `test_platforms.py` (8 platforms) | sim |
| R11 | Open questions retired in hardware (Q21/Q22/Q7) | `test_ads9224r`, `test_as5047p`, `test_drv8316r` | sim |
| R12 | RTL synthesizes + fits an open ECP5 flow | `test_synth.py` / `synth/run_synth.py` | synth |
| R13 | Realism/fault scenarios behave (UVLO, magnet loss, …) | the realism + edge-case test tiers | sim |

**Coverage gaps (named, not hidden):** R6 is documented not proven (an
arithmetic-aware prover would close it). Timing closure at 25 MHz is *not* a
requirement met today (Fmax ≈ 3.3 MHz — see `status-matrix.md`). No silicon
correlation yet.

**Assumptions are first-class:** every proof's assumptions are in
`formal/manifest.toml` and rendered into `formal/proof_report.md`; non-vacuity
covers guard against vacuous proofs.
