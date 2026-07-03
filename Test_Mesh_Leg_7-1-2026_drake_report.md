# Drake physics analysis — Test_Mesh_Leg_7-1-2026

Multibody analysis of the rigged robotic foot using [Drake](https://drake.mit.edu)
(`scripts/drake_leg_analysis.py`, pinned venv at `build/foot_rig/drake/.venv`,
Python 3.12). The model mirrors `scripts/rig_foot_7_1.py` exactly:

```
world ─weld─ base housing [5]
      └─ leg_swing  revolute Z @ gear axle pin [3]  (±25°, worm-driven 20:1)
          ├─ knee_blade  revolute Z @ knee pin [11]  (−90°…+10°) → blade UPPER length [10]
          │   └─ toe_hinge  revolute @ yellow toe pin [13] → blade LOWER length [18]
          ├─ piston  prismatic along shin @ distal bushing [16] → pushrod [14]
          └─ loop closure: ball constraint pinning [18]'s heel ear to [12] on the piston
```

Masses are the `physics.json` placeholders (leg link 0.55 kg, upper 0.05, lower 0.07,
pushrod 0.08); COMs/inertias are thin-rod estimates. Gravity acts along −Y.

## Results

| Quantity | Value | Notes |
|---|---|---|
| Loop-closure gap, Drake FK vs closed form | 2×10⁻¹⁷ m | model and rig kinematics identical |
| Heel-pin drop at φ = −90° | **−41.1 mm** | matches the animation's closed form |
| Peak dh/dφ (piston per crank angle) | 74.6 mm/rad | at full sweep-out |
| Top dead center | φ = 0° | dh/dφ = 0 → infinite mechanical advantage holding the stowed blade |
| Knee holding torque, peak | **0.129 N·m** at φ = −90° | 3 N·m placeholder motor → ~23× margin |
| Hip (swing) holding torque, peak | **0.99 N·m** at σ = +25°, blade out | |
| Worm-side torque through 20:1 | **0.049 N·m** | 1.2 N·m placeholder motor → ~24× margin |
| Passive drop from φ = −5° (SAP sim) | settles at **−90.0°**, −41.1 mm, in ~0.45 s | joint-limit stop; loop gap ≤ 3 µm throughout |

## Findings

1. **"Lowers by its weight" verified dynamically.** With the kinematic loop closed by
   a ball constraint and gravity on, releasing the blade near vertical sweeps it out
   front and drops the heel pin to the −90° stop — no actuation needed. The knee
   motor's job out-front is *restraint and retraction*, not extension.
2. **Top dead center at blade-vertical.** At φ = 0 the conrod is colinear with the
   piston, so gravity on the piston produces no crank torque: the stowed pose is
   nearly self-holding (only the crank's own imbalance acts, ~−5 mN·m at +10°).
3. **Both placeholder motors are 20×+ oversized for static gravity loads** at these
   placeholder masses. Real sizing will be driven by ground-contact/impact loads,
   not by holding the mechanism's own weight.
4. **Worm self-locking.** A single-start worm at 20:1 is typically non-backdrivable,
   so the hip likely holds the ±25° swing with the motor unpowered; the 0.05 N·m
   worm-side load supports that.

## Plots (`build/foot_rig/drake/`)

- `piston_kinematics.png` — heel-pin drop + dh/dφ across the ROM
- `knee_holding_torque.png` — knee gravity-holding torque vs φ
- `swing_holding_torque.png` — hip holding torque vs σ at three blade poses
- `passive_drop_sim.png` — passive-release time traces
- `results.json` — all numbers, machine-readable

## Caveats

- Masses/COMs/inertias are placeholders; re-run with measured values to firm up torques.
- No ground contact, friction, or impact loads modeled yet — natural next step is a
  stance-phase contact sim (Drake supports this directly on the same model).
- Worm drive modeled as an ideal 20:1 ratio; no mesh efficiency or backlash.
