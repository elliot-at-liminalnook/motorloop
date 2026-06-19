<!-- SPDX-License-Identifier: MIT -->
# Motor selection & comparison — results

Implements [`motor-selection-checklist.md`](motor-selection-checklist.md): three
concrete, available motors as provenance-tracked profiles + a head-to-head
comparison. Profiles: `sim/tests/motors.py`; comparison + assertions:
`sim/tests/test_motor_comparison.py`; figures: `make motors`
(`sim/scripts/gen_motor_figures.py`) → `figures/motors/`.

**Frame (honest):** the comparison is **model-based** — closed-form from the
datasheet/estimate params. The DB42/EC 45 carry real datasheet R/L/Kt; the GM2804
is sparser (R + Kv known, L/J estimated). All `align_offset`s and the gimbal's L/J
need the physical motor + the bench motor-ID to become `measured` (Q1). The 12 V
bench under-drives the DB42/EC 45 vs their rated points.

## The three motors

| | Budget **GM2804** | Mid **DB42S03** | Premium **EC 45 flat** |
|---|---|---|---|
| Price | ~$30 | ~$90 | ~$200 |
| Pole pairs | 7 | **4** | 8 |
| R (line-line) | 9 Ω (assumed) | **1.5 Ω (datasheet)** | 0.8 Ω (datasheet*) |
| L (line-line) | ~4 mH (estimate) | **2.1 mH (datasheet)** | 0.56 mH (datasheet*) |
| Kt | 33 mN·m/A | 35 mN·m/A | 27 mN·m/A |
| Provenance | mostly `assumed`/measure | **mostly `datasheet`** | datasheet-typical* |

\* confirm exact against maxon 251601.

## Performance comparison (M1–M8)

- **M1/M2 Torque–speed @ 12 V** (`torque_speed.png`): similar Kt (~27–35 mN·m/A),
  but the gimbal's 9 Ω starves its stall torque to **45 mN·m** vs **277** (DB42)
  and **404** (EC 45) — the low-R motors deliver far more torque at 12 V.
- **M3 Current-loop dynamics** (`dynamics.png`): L/R = **0.44 / 1.40 / 0.70 ms** —
  each motor needs different current-loop tuning.
- **M4 Mechanical dynamics**: acceleration at rated = **2.2k / 12.9k / 9.4k
  rad/s²** — the DB42 accelerates fastest (good Kt/J + rated current).
- **M6 Efficiency** (`efficiency.png`): the gimbal's resistance caps it at
  **~43 %**; the DB42 (**72 %**) and EC 45 (**68 %**) are far better — the single
  biggest practical separator.
- **M8 Motor↔sensor coupling** (`latency_coupling.png`, the headline): angle-
  latency torque loss grows with pole pairs, so with the **AS5600** at speed the
  8-pole EC 45 loses up to **~53 %**, the 7-pole gimbal **~42 %**, but the 4-pole
  DB42 only **~14 %**. The **premium motors demand the AS5047P**; the DB42
  tolerates the cheap sensor. This ties the motor choice directly to the
  sensor-comparison study — they are one coupled decision.

## Implementation notes

- **Pole pairs are build-time.** `gen_rtl_params.py` bakes `POLE_PAIRS`,
  `SPEED_NUM` and `EXTRAP_NUM` from the pole count, so a motor with a different
  pole count needs a regen + re-Verilate — `sim/scripts/build_motor.sh <name>`
  (env-var override, no file mutation). *(Found + fixed a latent bug: the
  `POLE_PAIRS` define had re-read the param directly, desyncing it from
  `SPEED_NUM`/`EXTRAP_NUM` under an override.)*
- **The DB42 (4 pp) = the current build**, so it runs the FOC loop on the existing
  bench with no rebuild — validated (`test_db42_runs_on_current_build`: iq tracks,
  zero shoot-through). The GM2804 (7) and EC 45 (8) need `build_motor.sh` + a bench
  run (on-demand; the analytical comparison covers them rebuild-free).
- The measurable→per-phase conversions reuse the bench's existing relations
  (R=Rll/2, L=Lll/2, Ke=Ke_ll/√3, pp=poles/2) — checked in
  `test_conversions_match_derive_params`.

## Recommendation

- **DB42S03 = the bench's characterized motor.** Datasheet R/L/Kt/J, **4 pp =
  no RTL rebuild**, best efficiency + acceleration. Adopting it promotes the motor
  params `placeholder → datasheet` and retires the data half of Q1.
- **GM2804 = the cheap teaching/sensored motor** (its built-in AS5048A encoder fits
  the angle-sensor work) — but budget on its low efficiency and, at 7 pp, its
  latency sensitivity (pair it with the AS5047P).
- **EC 45 flat = precision/torque** — but at 8 pp it is the most latency-sensitive,
  so it *needs* the AS5047P.

Reproduce: `make motors` + `pytest sim/tests/test_motor_comparison.py`.
