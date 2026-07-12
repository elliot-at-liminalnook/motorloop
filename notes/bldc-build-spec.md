# BLDC Leg-Actuator Build Spec + Control Architecture (2026-07-05)

> **Superseded as the active BOM (2026-07-09).** This remains a BLDC alternative
> study. The selected robot uses twelve Waveshare ST3215-HS bus servos and a 6 lb
> total-mass limit; see [robot-hardware-contract.md](robot-hardware-contract.md).

Reference build spec for the quadruped's leg actuation, replacing the goBILDA
servo path. Grounded in the co-design analysis (`notes/gait-feasibility-verdict.md`,
`sim/robot/codesign*.py`): the servo topped out below the dynamic-gait regime; a
BLDC leg actuator with self-authored FOC clears it with large margin, so selection
is driven by **provenance + availability**, not capability.

Design philosophy: **provenance-first, hierarchical control.** Start with documented
parts so a fault is unambiguously your FOC, not a mystery motor; run the RL policy on
a powerful remote computer at 50 Hz and the hard-real-time motor loops on one FPGA.

---

## 1. Per-joint actuator — LIGHTWEIGHT MIXED CONFIG (1 BLDC + 2 servos per leg)

12 actuated joints: 4 legs × (hip_yaw, pitch, knee). All-BLDC is far too heavy
(9.6 kg of motors). The weight-optimal build is **one BLDC per leg on the hip_yaw,
servos on pitch and knee** — 4 BLDC + 8 servos.

| joint | actuator | reduction | why |
|---|---|---|---|
| hip_yaw | **servo + SEA belt** | 10–15:1 | stride: the belt spring provides the fast stride oscillation (Level B resonance ~3.85 Hz), so a servo here is rescued by the spring — no BLDC needed |
| pitch | servo (torque-class) | 8–15:1 | reach/support — lowest bandwidth |
| **knee / lift** | **BLDC** | 12–16:1 | foot-clearance cadence (the walk's bottleneck) AND the stomp weapon — the BLDC serves both |

**Where the ONE BLDC goes: the knee/lift — NOT the yaw** (`codesign_mixed.py`, which
corrected the initial hand-waved yaw pick). The gait ceiling is set by the SLOWEST
gait-critical axis. Two axes cap cadence: the yaw stride and the knee foot-clearance
(retract+extend each swing). The yaw already carries your **SEA rubber belt**, so a
servo yaw is rescued by the spring (Level B — the belt oscillates the stride near its
resonance without the servo moving fast). That leaves the **knee/lift as the un-sprung
bottleneck** — so the one BLDC belongs there. Bonus: the knee is also the **stomp
weapon**, so a BLDC there gives fast foot-clearance AND a strong/fast strike (24 N·m
peak). Optimizer result for BLDC-on-knee: **Fr 0.084, v 0.54 m/s** — a real walk, ~40×
the servo-only shuffle, at the doorstep of the dynamic regime (BLDC-on-yaw gave only
Fr 0.002, servo-lift-capped).

To cross fully into dynamic (Fr ≥ 0.10): stiffen the SEA belt for a higher stride
resonance, or add a second BLDC on the yaw. The 1-BLDC mixed config is the light
sweet spot; it reaches a fast walk, not a run.

**Which BLDC — go lighter than the D6374.** For a single leg joint driving through a
belt, the torque need is modest (drag a loaded foot ~4–6 N·m), so the big D6374
(0.8 kg) is overkill on mass. Prefer the **ODrive D5312s 330Kv (0.25 kg)** — same
ODrive provenance (published Kt/R/L, thermistor), ~21 N·m continuous / 42 N·m peak at
the output, plenty. Or the **mjbots mj5208 (0.19 kg)** for the absolute lightest. Keep
the D6374 only if you want big peak-torque combat headroom on the yaw and can spend the
mass. Kt = 9.55/Kv.

**Servos (pitch, knee): a torque-class PWM servo**, e.g. REEFS 400:SC V2 (2.86 N·m) or
Savox SB-2274SG (2.45 N·m) — the ones from the servo shortlist. Use a **lower reduction
than the old 20:1 worm** on pitch (say 8–12:1) to trade self-lock/torque for the swing
speed the gait cadence needs; keep the worm only if you want power-off self-lock and
accept a slower walk.

**Encoder: AS5047P** on the BLDC (motor-side, on the D-series dual shaft) for FOC;
optionally a second AS5047P output-side after the SEA belt to observe belt compliance
and true leg angle. The servos use their own internal pots.

### Weight comparison (actuators only)
| config | motors | actuator mass |
|---|---|---|
| all-BLDC (12× D6374) | 12 | ~9.6 kg |
| 1 D6374/leg + 8 servos | 4+8 | ~3.7 kg |
| **1 D5312s/leg + 8 servos** | 4+8 | **~1.5 kg** |
| 1 mj5208/leg + 8 servos | 4+8 | ~1.2 kg |

The mixed light config saves **~8 kg** vs all-BLDC — and lighter actuators feed back
favorably into the Froude/stability co-design (`codesign.py`; Fr scales as 1/L, and
lower mass lowers the COM-shift and torque demands).

---

## 2. Power stage (per motor)

**TI DRV8353RS smart gate driver + external MOSFET half-bridges**, adapting
**TIDA-010956** (documented 85 A / 24–60 V three-phase inverter reference design) as
the board. The DRV8353 is the ideal FPGA split: it takes 6 PWM (or 3-PWM mode with
internal deadtime), drives the FETs, and integrates the current-sense amplifiers +
protection; the FPGA does only the FOC math and PWM. SPI-configurable (gain, deadtime,
fault thresholds); `nFAULT` line per driver.

- Bus: **48 V** (headroom for speed; the co-design used 48 V).
- FET choice: sized to the D6374 (≥90 A peak, ≥50 A continuous), e.g. 60–100 V
  logic-level power MOSFETs on the TIDA-010956 BOM.
- Phase-current sensing: DRV8353 integrated CSAs → a **local ADC per power pod**
  (digitize near the motor; never route analog current over robot cables). See §3.

---

## 3. One FPGA driving all 12 motors

**This is very feasible — the constraint is I/O and ADC bandwidth, not FOC compute.**

### 3.1 Why one FPGA suffices for 12 FOC loops
A FOC step (Clarke → Park → 2× PI current loops → inv-Park → SVPWM) is ~50–200
multiply-adds. At 100 MHz a single pipelined FOC datapath finishes one motor in
~1–2 µs. **Time-multiplex one FOC core across all 12 motors:** 12 × 2 µs = 24 µs,
inside a 50 µs (20 kHz) PWM period with margin. So one shared FOC engine services all
12 — no need for 12 parallel pipelines (though a Zynq/Artix has the DSP slices for
that too if you want per-motor determinism). FOC compute is the cheap part.

### 3.2 Recommended part: AMD/Xilinx Zynq (PS + PL)
A Zynq SoC splits the job cleanly and has the most documented motor-control prior-art:
- **PL (FPGA fabric):** the 12× time-multiplexed FOC engine, SVPWM generators, encoder
  SPI masters, ADC interfaces, hardware safety interlocks (overcurrent/overtemp trip
  in logic, not software). Hard real-time.
- **PS (ARM cores):** the network stack, receiving high-level commands from the remote
  computer (§4), the position/velocity outer loops (1–10 kHz) if you don't put them in
  PL, telemetry, and supervisory safety (watchdog). Soft real-time.

### 3.3 I/O budget (12 motors)
- **PWM:** 6 per motor × 12 = 72 (trivial in PL).
- **Gate-driver SPI:** one shared SPI bus + 12 chip-selects + 12 `nFAULT` for config
  and fault readback.
- **Encoder SPI:** one shared bus + 12 CS (motor-side); +12 CS if output-side encoders.
- **Current + bus ADC:** 3 phase currents + 1 bus V per motor = 48 channels. **Do not
  route analog over cables** — put a small simultaneous-sampling ADC in each power pod
  and stream digital back.

A mid Zynq (e.g., Zynq-7020 / Kria K26) has ample logic + I/O for this.

### 3.4 Topology: central FPGA + distributed "smart power pods"
```
                 ┌──────────────────────── central Zynq board ───────────────────────┐
   remote  ⇄ ETH │  PS(ARM): net stack, outer loops, watchdog                          │
   computer      │  PL(FPGA): 12× FOC engine, SVPWM, encoder SPI, safety interlocks    │
                 └───────┬───────────┬───────────┬─────────── … 12 fast serial links ─┘
                         │LVDS/SPI   │           │
                   ┌─────┴─────┐ ┌───┴────┐  ┌───┴────┐
                   │ power pod │ │power pod│  │power pod│   one per motor, mounted at the leg:
                   │ DRV8353RS │ │  …     │  │  …     │     gate driver + FETs + local ADC
                   │ +FETs+ADC │ │        │  │        │     + AS5047P; digitizes currents,
                   │ +encoder  │ │        │  │        │     streams back over LVDS/SPI.
                   └─────┬─────┘ └────────┘  └────────┘
                       BLDC (D6374)
```
Each pod carries only power + analog + digitization; the FPGA carries all control.
This keeps high-current runs short (pod at the motor), analog local (ADC at the pod),
and only digital + 48 V bus on the robot harness. Links: LVDS pairs or fast SPI, one
per pod (or daisy-chained per leg to cut cable count). If cabling dominates, an
intermediate "leg board" fanning 3 pods to one link to the FPGA is a clean compromise
that still keeps ONE FPGA doing all FOC.

### 3.5 Control-loop rates
- FOC current loop: **20–40 kHz** (PWM rate), in PL.
- Position/velocity loop: **1–10 kHz**, PL or PS.
- High-level command update: **50–100 Hz** from the remote computer (§4) — matches the
  sim `CONTROL_DT = 0.02 s`.

---

## 4. Remote high-level computer → FPGA

The RL policy is heavy (a 512-256-128 net) and non-real-time-friendly; the motor loops
are light but hard-real-time. Split them across the link at the natural seam — the
**50 Hz joint-target boundary that our sim already uses.**

### 4.1 Division of labor
```
  ┌─ REMOTE COMPUTER (Jetson Orin on-board, or desktop GPU over tether) ─┐
  │  • runs the trained RL policy at 50 Hz                               │
  │  • input : observation (joint pos/vel ×12, root quat, root vel,      │
  │            prev action, command)  ← exactly the sim obs (50-D)       │
  │  • output: 12 joint targets (PD position, or torque)  ← the sim      │
  │            action space (12 actions → PD targets)                    │
  │  • also: gait command / operator input, logging, learning           │
  └───────────────────────────┬─────────────────────────────────────────┘
                    50–100 Hz  │  bidirectional, small payload
                     Ethernet  │  down: 12 targets (+mode)   up: obs (~30 floats)
                    (UDP or     │
                     EtherCAT)  ▼
  ┌─ FPGA (Zynq) ───────────────────────────────────────────────────────┐
  │  PS: receive targets @50 Hz, hold-last-on-dropout, run outer loop,   │
  │      assemble the observation from encoders + IMU, send up @50 Hz    │
  │  PL: 12× FOC turning targets → phase voltages @20 kHz; safety trips  │
  └─────────────────────────────────────────────────────────────────────┘
```

### 4.2 The link
- **Physical:** Gigabit Ethernet (tethered robot) — payload is tiny (12 floats down,
  ~30 up), so bandwidth is a non-issue; latency and determinism are what matter.
- **Protocol:** start with **UDP** (simple, low-latency; the policy already tolerates
  50 Hz jitter). Upgrade to **EtherCAT** if you want hard-deterministic timing and easy
  multi-node expansion — EtherCAT is the industrial-robotics standard and the Zynq PS
  has stacks for it.
- **On-board vs tether:** a Jetson Orin on the robot removes the tether-latency risk and
  is the cleanest deployment; a desktop GPU over tether is fine for bring-up and lets
  you keep training in the loop.

### 4.3 Why this seam is the right one — it's the sim contract
Our trained policy's interface **is** this boundary, by construction:
- **Action (down):** 12 values → PD position targets (`walker_warp_env` action space,
  authority-scaled per joint). The FPGA's outer loop turns each target into a
  position/velocity setpoint for the FOC torque loop.
- **Observation (up):** joint pos/vel (×12) from the AS5047P encoders, root
  orientation + velocity from an **IMU** on the torso, previous action, and the command.
  This is the 50-D obs the policy was trained on.

So deployment is: encoders + IMU → obs (FPGA assembles, sends up) → policy on the remote
computer → 12 targets (sends down) → FPGA FOC → BLDCs. The **sim-to-real contract is
already fixed** by the env we trained against; the FPGA is the real-time realization of
the sim's PD-servo + physics step.

### 4.4 Safety across the link (non-negotiable)
- **Watchdog:** if high-level commands stop (link drop, computer crash), the FPGA PS
  detects the missed 50 Hz deadline and commands the PL to a safe state — hold position
  with bounded current, then damp/lower. Never free-run on a stale command.
- **Local trips in PL:** overcurrent and overtemp (thermistor) trip in FPGA logic
  within microseconds, independent of the ARM/software or the remote computer.
- **E-stop:** hardware kill of the 48 V bus, independent of everything above.

---

## 5. Bill of materials — LIGHTWEIGHT MIXED CONFIG (per robot)

| item | part | qty | ~unit | ~total |
|---|---|---:|---:|---:|
| BLDC (yaw, 1/leg) | ODrive D5312s 330Kv | 4 | $129 | $516 |
| BLDC encoder | AS5047P | 4 | ~$8 | ~$32 |
| output-side encoder (opt, after belt) | AS5047P | 4 | ~$8 | ~$32 |
| gate driver (BLDC only) | TI DRV8353RS | 4 | ~$6 | ~$24 |
| power pod (FETs+ADC+PCB) | per TIDA-010956 | 4 | ~$40 | ~$160 |
| servo (pitch + knee) | REEFS 400:SC / Savox SB-2274 | 8 | ~$50 | ~$400 |
| controller | Xilinx Zynq (Kria K26 / 7020 board) | 1 | ~$250–350 | ~$300 |
| high-level computer | Jetson Orin (or tethered desktop) | 1 | ~$500 | ~$500 |
| bus / battery / BEC / wiring | 24–48 V pack + distribution | — | — | ~$250 |
| **estimate** | | | | **~$2.2 k** |

Lighter AND cheaper than all-BLDC (~$3.4 k → ~$2.2 k), and only **4 FOC loops** instead
of 12 — the FPGA barely notices (see §3: one time-multiplexed FOC core did 12; 4 is
trivial), and it also drives the 8 servos with plain PWM. (D-series currents/R are
ODrive-published; bench-characterize any e-skate fallback. Belts, pulleys, brackets,
SEA rubber pulley are mechanical, not listed.)

---

## 6. Open decisions + risks

- **Self-lock is gone.** The worm self-locked (held stance power-off); a BLDC
  backdrives. Options: hold with FOC current (costs power, heats the motor), add a
  small parking brake on the pitch/knee, or accept active holding. Combat "hold when
  hit" argues for a brake or the retained SEA belt (blow absorption).
- **SEA spring is now optional for locomotion** (§ `gait-feasibility-verdict.md`):
  with BLDC speed+torque, a rigid dynamic gait is reachable; keep the belt for energy
  recycling / blow absorption, not necessity. Level B (SLIP) still gives its optimal
  stiffness (k~25, f_nat ~3.5 Hz) if you keep it.
- **Motor mass feeds back into the design.** 12 real motors are heavy; re-run the
  co-design (`codesign.py`) with the actual motor masses before finalizing geometry.
- **Thermal is the real limit.** All torque-at-speed numbers are continuous-current
  (thermal) bounded — validate against YOUR cooling; the thermistor + PL overtemp trip
  enforce it.
- **Mixed-actuator cadence cap (the honest trade).** With pitch + knee as servos, the
  swing-leg cadence is capped by servo speed — so the light config does a **dynamic
  WALK (yaw-BLDC propulsion + servo-positioned lift/clearance), not a run.** That is a
  large step up from the servo-only shuffle and is the right point for a light robot;
  a run would want a second BLDC (e.g. on pitch). Use a lower pitch reduction (8–12:1)
  so the servo lift keeps up with the walk cadence.
- **Yaw vs pitch for the single BLDC** is worth settling in Level C — yaw wins on the
  propulsion + SEA-belt argument, but pitch was the measured cadence bottleneck; the
  mixed-actuator sim can confirm.
- **Level C verification (pre-hardware):** RL on the MIXED-actuator leg model — BLDC
  torque control on yaw (real Kt, reflected inertia J·N², SEA belt compliance) + servo
  PD on pitch/knee. The obs/action contract is unchanged; this de-risks the mixed
  config before buying parts.

---

## 7. Sequenced next steps

1. Refine the motor currents/R from ODrive datasheets; re-run `codesign_bldc.py` with
   the real numbers and the actual per-joint reductions.
2. Re-run `codesign.py` static/dynamic optimization with the real motor masses.
3. Level-C sim: swap the walker env's servo-PD for a BLDC torque model + reflected
   inertia; retrain/verify the policy transfers (the obs/action contract is unchanged).
4. Bring up ONE joint: D6374 + AS5047P + DRV8353RS + Zynq, closed-loop FOC + position,
   validate torque-at-speed against the model on a dyno/bench.
5. Scale to one leg (3 joints), then 12, then the remote-computer 50 Hz loop with the
   trained policy driving it.
