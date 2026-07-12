<!-- SPDX-License-Identifier: MIT -->
# Robot hardware contract

> **Document status:** Current · **Audience:** Robot and hardware developers · **Last reviewed:** 2026-07-12 · **Canonical for:** Active robot physical envelope

Effective 2026-07-09, the modeled robot has two hard constraints:

- maximum complete-robot mass: **6.000 lb = 2.72155422 kg**
- joint actuators: **12 identical Waveshare ST3215-HS bus servos**

## Selected actuator

The model uses the 12 V datasheet point from the
[Waveshare product page](https://www.waveshare.com/st3215-hs-servo-motor.htm) and
[Waveshare wiki](https://www.waveshare.com/wiki/ST3215-HS_Servo_Motor), matching
[RobotShop part RB-Wav-1556](https://www.robotshop.com/products/waveshare-20kgcm-bus-servo-motor-106rpm-high-speed-large-torque-w-360-deg-high-precision-magnetic-encoder):

| property | model value |
| --- | ---: |
| stall torque | 20 kgf.cm = 1.96133 N.m |
| no-load speed | 106 RPM = 11.10029 rad/s |
| mass | 68 g each; 816 g for 12 |
| operating voltage | 6-12.6 V; model point 12 V |
| no-load current | 0.240 A each; 2.88 A for 12 |
| locked-rotor current | 2.4 A each; 28.8 A for 12 |
| position sensing | 12-bit, 360-degree magnetic encoder |
| control | TTL UART serial bus |

Waveshare does not publish continuous-duty torque or output inertia. The simulator
therefore treats 20 kgf.cm as a short-duration stall envelope, applies a linear
torque-speed derating, and keeps output inertia as a named estimate. Both need bench
identification before hardware-correlated training. Supply droop/current sharing is
not yet modeled; the battery and distribution design must be specified before the
model can decide whether all twelve stall envelopes are simultaneously deliverable.

## Mass budget

The twelve servos consume **0.816 kg (1.799 lb)**, leaving **1.905554 kg
(4.201 lb)** for structure, battery, compute, wiring, fasteners and weapon hardware.

The CAD assembly still labels its non-servo masses as placeholders. For conservative
dynamics, `gen_mesh_robot_mjcf.py` preserves their relative distribution and scales
them to fill the legal maximum:

| allocation | mass |
| --- | ---: |
| 12 servos | 0.816000 kg |
| torso/onboard structure | 0.519225 kg |
| four leg structures | 1.360368 kg |
| striker placeholder | 0.025961 kg |
| **compiled total** | **2.721554 kg / 6.000 lb** |

This normalization is a simulation mass envelope, not a manufacturability claim.
The physical BOM must close independently below 6 lb, with scale margin.

## Enforced paths

- `sim/tests/motors.py`: datasheet-backed ST3215-HS profile
- `sim/robot/gen_mesh_robot_mjcf.py`: CAD-derived model and transmitted envelopes
- `sim/robot/robot_design.py`: active walker hardware and mass source
- `sim/robot/robot.toml`: primary parametric model; armed build is 6.000 lb
- `sim/robot/test_mesh_robot_contract.py`, `test_design_alignment.py`, and
  `test_model_contract.py`: compiled mass, count, torque and speed regression gates

The mandatory aggregate entry point before GPU simulation or RL is
`bash scripts/run_pre_gpu_tests.sh --require-gpu`; see
[`notes/pre-gpu-test-entrypoint.md`](pre-gpu-test-entrypoint.md).
