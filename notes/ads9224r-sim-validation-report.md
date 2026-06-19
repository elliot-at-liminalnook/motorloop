<!-- SPDX-License-Identifier: MIT -->
# ADS9224R module — simulation validation results (Tiers 2–4)

Implements [`ads9224r-sim-validation-checklist.md`](ads9224r-sim-validation-checklist.md).
Raises the open ADS9224R module from "topology math" to **datasheet-model /
vendor-model-validated** measurements in simulation. Honest ceiling: `measured`
needs the fabbed board (Q23 / open-board §10).

Reproduce: `make ads9224r` (figures) + `pytest sim/tests/test_spice_derivations.py
sim/tests/test_ads9224r_loop.py sim/tests/test_ads9224r_vendor.py`.
Figures: [`figures/ads9224r-module/gallery.md`](../figures/ads9224r-module/gallery.md).

## Tier 2 — datasheet-parameter models (done, validated)

**Datasheet anchoring corrected the Tier-1 placeholders** (read from the archived
SBAS876C / SBOS778D): the ADC sample cap is **16 pF** (was a 60 pF guess), the
acquisition window **140 ns** (was 200 ns), the reference architecture is the
internal 2.5 V × 1.6388 buffer = 4.096 V (FSR ±4.096 V), with REFby2 = 2.048 V
setting the FDA common mode. These are now `status = datasheet`.

- **Acquisition settling (`ads9224r_acq.cir`, switched-cap):** with the real
  cap-DAC kickback (Csh/(Csh+Cflt) ≈ 1.6 %, not a full step) the held sample
  settles to **1.6e-7 (≪ 0.5 LSB)** within the 140 ns window. The retune to
  `flt_c = 1 nF` (from 1.5 nF) was forced by the real 140 ns budget — a Tier-2
  catch.
- **Noise → ENOB (`ads9224r_noise.cir`, `.noise`):** the bare charge-bucket
  (15.9 MHz pole) is **not** an antialiasing filter — wideband front-end noise
  costs **~1.9 bits**. Adding an FDA feedback cap (`fda_fb_c = 270 pF`, an
  antialiasing pole at **295 kHz**, well below Nyquist) drops the front-end noise
  199 µV → **51 µV**, an ENOB cost of **0.46 bit** (PASS < 0.5). This is the
  SBAA282 lesson, found and fixed in sim; `fda_fb_c` is now a derived design
  parameter (`feedback.current_ads9224r.signal_bw_hz`).
- **Bandwidth:** 295 kHz antialiasing pole (captures the motor-current band; far
  under the 1.5 MHz Nyquist).
- **THD/distortion:** the linear datasheet-model can't show it — deferred to the
  Tier-3 vendor macromodel (which carries the nonlinearity).

## Tier 3 — vendor-macromodel cross-check (infrastructure ready; numbers pending the .LIB)

- **Skip-if-absent infrastructure** (`ths4551_vendor.cir`,
  `test_ads9224r_vendor.py`, `spice_runner.THS4551_LIB/REF6041_LIB`) — the exact
  DRV8301 pattern: cross-checks the moment a vendor `.LIB` is dropped at
  `docs/ti-simulation-models/{ths4551,ref6041}/`, skips otherwise (CI green).
  TI's models are portal-gated (PSpice-for-TI / TINA-TI), so the macromodel
  numbers are the one honest gap here — documented, not faked.
- **Internal vs external reference (§3.4, datasheet-decided):** the ADS9224R's
  94.5 dB SNR **already includes** its internal-reference noise (REFby2 noise
  10 µV → a 112 dB contribution, far above 94.5 dB), so REF6041 gives negligible
  **SNR** gain. Its value is **drift** (5 ppm/°C). **Decision: internal reference
  is the default; REF6041 is an optional populate for low-drift/precision** — BOM
  `U3` set to qty 0 (optional).

## Tier 4 — system-level measurement (done, validated)

Referring the validated front-end to the FOC current loop
(`test_ads9224r_loop.py`):

| Contribution | RMS current noise |
| --- | --- |
| Front-end (THS4551 + resistors, antialiased) | **1.28 mA** |
| ADS9224R transition noise (0.4 LSB) | 1.25 mA |
| **Combined loop noise** | **1.79 mA** |

= **15.0 effective bits**, **0.0017 % of the ±102 A full-scale**. The front-end
is *balanced against the ADC's own noise* (the antialiasing design target), and
the FOC loop sees a negligible current-measurement noise. The **skew** half of
the story is the part-comparison study ([T3/T4](part-comparison-report.md)): the
module supplies the simultaneous (scheme-0) path that retires Q21. Together:
**simultaneous sampling (zero skew) + a small, bounded, ~15-bit noise floor.**

## Honest status

- **Validated in sim:** scaling (320 codes/A), settling (<0.5 LSB), front-end
  ENOB (0.46-bit cost with antialiasing), loop current-noise budget (~15 bits),
  the reference decision.
- **Pending the vendor `.LIB` (Tier 3):** macromodel cross-check of BW/THD/noise
  (portal-gated download).
- **Pending the bench (Q23 / §10):** absolute ENOB/SNR, real layout coupling,
  measured inter-channel skew — only the fabbed board promotes `assumed` →
  `measured`.
