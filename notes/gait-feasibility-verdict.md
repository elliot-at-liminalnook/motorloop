# Gait feasibility verdict — the mesh leg CAN walk, with a stance re-bias (2026-07-04)

> **Document status:** Historical · **Audience:** Robot and mechanism developers · **Last reviewed:** 2026-07-12 · **Current hardware source:** [`robot-hardware-contract.md`](robot-hardware-contract.md)

> **Hardware baseline superseded (2026-07-09).** The mechanism findings remain
> useful, but the old mass and goBILDA/BLDC numbers below are not the current model.
> Re-run dynamic gait validation for the 6 lb, 12x ST3215-HS contract documented in
> [robot-hardware-contract.md](robot-hardware-contract.md).

> ## CORRECTION (2026-07-04, after the first verdict below)
> The original verdict — "cannot walk, no foot-lift authority" — was **wrong in one
> specific, fixable way**, caught by the user's note that the knee toggle should be
> able to retract. It can. My tests used the *as-designed stance* (knee −50°), and a
> print bug hid the real data. The corrected finding:
>
> - The knee has **~40 mm of foot-height authority** across its range (foot 473.8 mm
>   below torso at knee −90° → 433.2 mm at knee 0°). Retracting the knee lifts the foot.
> - The **as-designed stance sits at knee −50°, near the top of that range**, leaving
>   only **8.5 mm** of upward retraction — which is what every earlier test measured
>   and mistook for a hardware limit. The other 32 mm is spent *downward* on the stomp.
> - **Re-biasing the nominal stance to a deeper knee (~−70°) is stable and level**
>   (measured z=0.448, all feet down) and restores **~18–40 mm of knee-driven foot
>   lift** while keeping ~20 mm of stomp. This is a **free posture/controls change —
>   no new hardware DOF.** The knee servo tracks the retraction command.
>
> **So: no hardware change is required to unlock foot-lift.** The genuine remaining
> challenge is **coordinated COM management during the step** (a hand-coded open-loop
> single-step still rolls the body — my crude COM shift is wrong), which is a *control*
> problem the built RL stack (imitation + reference-state-init + curriculum) is made
> for — not a mechanism limit. What stays true from below: the pitch stability cliff
> (~−12° all-legs), the narrow COM-shift authority (widening stance still helps), and
> that the stride is slow. The path forward is software: re-bias the stance posture,
> regenerate the reference gait around it with real COM coordination, then RL.
>
> The stance-widening lever (lever B) still helps the COM margin and is worth doing in
> the next hardware revision, but it is now an *optimization*, not a prerequisite.

---

*Original verdict (superseded on the foot-lift point by the correction above; the
stance/cliff/COM measurements remain valid):*

**Bottom line: the servo-true mesh robot has no statically-stable walking gait,
and the binding constraint is hardware geometry, not training.** Three RL runs
(40M + 8M + 40M steps) and two reference-gait synthesis attempts all converged on
the same behavior — a planted-foot creep — because that is the only thing this leg
mechanism physically affords. The RL "creep" result was never a training failure.
It is the correct gait for a leg that cannot lift its feet.

All numbers below are measured on the servo-true model (`gen_mesh_robot_mjcf.py`,
belt 6, 8.06 kg) with the P-only servo + torque-speed derating the env uses.
Reproduce via `sim/robot/diagnose_leg_locomotion.py`.

## The five measured constraints

1. **Level stance exists only in a narrow window.** All legs at pitch ≈ −10°,
   knee ≈ −50° settles level at torso z = 0.448 (0.6 mm foot-height spread). This
   is the one stable operating point.

2. **Stability cliff at ~−12° pitch.** Commanding all legs to pitch −15° collapses
   the body to +88° pitch; −17° → +72°. The usable level-stance pitch band is only
   about ±2° wide. The old reference gait's stride bands (front −25→−10°, rear
   −4.6→−20°) ran mostly *past* this cliff — which is why it fell at +93° roll /
   −81° pitch on the open-loop prover.

3. **No foot-lift authority (the killer).** No single leg's own actuators lift its
   foot clear of the ground while the body stays level. Across every strategy —
   knee retract (−50→+10°), knee extend (−50→−90°), pitch lift arc, and combos —
   the foot stays at 4–11 mm (foot radius is 12 mm, so it is *always in contact*),
   and the moves strong enough to shift it tilt the body 10–25° and knock *other*
   feet off instead. The knee's slider-crank "toggle" is a downward *strike* tool;
   it has no upward retraction authority.

4. **Kinematic clearance exists but is unreachable in practice.** Forward
   kinematics says the foot *can* retract 58 mm (at pitch +8°, knee 0) — the
   mechanism is not geometrically incapable. But actuating one leg to that pose
   destabilizes the quadruped: the leg that must swing is also a primary support,
   so lifting it throws the COM and tilts the body.

5. **Yaw is a weak, sideways stride.** ±20° yaw sweeps a foot only 23 mm, and
   almost entirely *laterally* (Δy ≈ 23 mm, Δx ≈ 3 mm). Yaw cannot translate the
   body forward when planted. The design intent ("stride from yaw sweep") does not
   survive contact with the geometry — the corner-mounted yaw axis sweeps an arc
   that is mostly cross-body.

6. **COM-shift authority is 2–3× short.** Cornering a foot for lift needs the COM
   moved ~3–4 cm into the opposite three-foot triangle. The achievable COM shift
   with the available joints is ~1–2 cm, and a hand-built COM-preshift-then-lift
   still rolled the body 37° and dropped two feet.

## Root cause

The leg was designed as a **stomp/strike weapon**: worm-driven pitch (strong, slow,
self-locking) plus a slider-crank blade "foot" driven by the knee for a powered
downward toggle-press. That mechanism has strong *downward* authority and none
*upward* — it cannot retract a foot — and its one strong axis (pitch) couples foot
height to body tilt with a hard stability cliff. It is a fighting leg, not a
walking leg. Walking needs a clean lift DOF and a wider support base; this geometry
provides neither.

## What this redirects

- **Stop spending RL on "walking" for this exact geometry.** No reward tuning or
  reference gait fixes a mechanism with no foot-lift DOF. The creep gait already in
  hand is the honest locomotion this hardware affords.
- **The reference-gait imitation upgrade is moot** for this body — there is no
  passing reference to imitate. `reference_gait.json` should be treated as invalid;
  the warp trainer's imitation hook is correct but has nothing valid to consume.
- **Hardware options, in rough order of leverage:**
  1. *A dedicated foot-lift DOF* (or re-purpose the knee toggle to retract, not
     just strike) — the single biggest unlock; without it, no true stepping.
  2. *Widen and lengthen the stance* (mount spread) — enlarges the support polygon
     so the modest achievable COM shift suffices; also softens the pitch cliff.
  3. *Accept a shuffle/drag locomotor* — if the blade-foot combat role is fixed,
     embrace feet-never-lift locomotion: a low-friction shuffle or a wheel/skid.
     The existing creep policy is a starting point, not a bug.
- **What to re-run only after a hardware change:** the whole ladder (feasibility →
  reference → imitation+RSI RL) is built and validated; it will produce a walking
  policy the moment the mechanism can lift a foot. Nothing in the software stack
  needs to change — only the leg.

## Honest caveats

- This is a *static/quasi-static* verdict plus dynamic spot-checks, not a full
  dynamic trajopt. A sufficiently violent *dynamic* gait (hopping, using momentum
  to unload a leg) was not exhaustively searched and is not obviously impossible —
  but it would demand speeds the 0.76 rad/s pitch axis cannot deliver, so it is a
  remote possibility, not a likely escape.
- Numbers are at belt ratio 6 and the 8.06 kg dieted mass. Belt 4 and heavier
  torso were spot-checked and are strictly worse for the COM-shift margin.

## Physical improvement recommendations (2026-07-04, prototyping phase)

Grounded in measurements on the servo-true model. Ranked by leverage-per-effort.

**Tier 1 — high leverage, cheap to change while prototyping:**

1. **Re-bias nominal stance to a deeper knee (~−70°).** Free (posture, no hardware).
   Restores 18–40 mm of knee foot-lift while keeping ~20 mm of stomp. This is the
   enabling change; everything else builds on it.

2. **Widen + lengthen the stance (move the leg mounts outward).** Measured COM-shift
   authority is ~1–2 cm, but cornering a foot needs ~3–4 cm of shift *into* the
   support triangle. Widening the polygon shrinks the *needed* shift — the cheapest
   way to close that gap. Also softens the pitch stability cliff. Current stance is
   0.20 m long × 0.162 m wide; going to ~0.30 × 0.26 roughly doubles the margin.

3. **Lengthen the yaw-stride lever arm.** Measured: the foot sits only **31 mm from
   the yaw axis**, so a ±30° yaw sweep moves it just **31 mm, and purely laterally**
   (0 mm fore-aft). Yaw is the *fast, strong* axis (2.53 rad/s, 6.35 N·m) — it should
   be the stride engine, but the tiny lever makes each step short. Increasing the
   foot's horizontal offset from the yaw axis (longer lower link, or move the yaw
   pivot inboard) lengthens every step proportionally. This is the single biggest
   win for stride *length*/*speed*.

**Tier 2 — real trade-offs to decide by role priority:**

4. **Worm ratio vs self-locking (the pitch axis).** Measured trade:
   - 20:1 (current): 21.2 N·m, 0.76 rad/s (43°/s), **self-locks** (holds stance
     unpowered — valuable when hit in combat).
   - 10:1: 10.6 N·m, 1.52 rad/s (87°/s), does NOT self-lock (needs a holding brake
     or accepts backdrive).
   All ratios have ample torque (stance needs ~2–4 N·m/leg). The 20:1 self-lock is a
   *combat* asset but caps leg-swing speed — the biggest limit on walk cadence *if*
   striding fore-aft with pitch. Decision: if walk speed matters, drop to ~10:1 + a
   small holding brake; if combat-hold dominates, keep 20:1 and lean on the yaw axis
   for stride (see #5).

5. **Embrace a lateral / crab-walk as the primary gait.** The geometry says the fast
   strong axis (yaw) strides *sideways*, while fore-aft comes only from the slow,
   height-coupled pitch. Fighting that to walk "forward" wastes the good axis. A crab
   gait striding along the yaw-sweep direction uses the 2.53 rad/s yaw + knee-lift,
   leaving pitch for balance — likely this body's fastest, most stable locomotion.

**Tier 3 — minor / fine-tuning:**

6. **Smaller foot (~8–10 mm vs 12 mm).** The foot radius eats into lift clearance; a
   smaller tip clears the ground with less knee retraction (trade vs footing pressure
   and stance stability).
7. **Keep the COM low.** Mount battery/heavy parts low in the torso; already helped by
   the 3.0→2.0 kg torso diet. Lower COM = more tip margin during single-support.

**What NOT to change:** no new actuated DOF is needed — the knee already lifts. The
slider-crank strike mechanism is fine; the loop couplings and dt=0.004 are solid.

## DRAKE PROOF — the three changes certified (2026-07-04)

`sim/robot/walker_improved.py` implements the three approved changes; the geometry
insight that fell out: **splay the legs laterally so the fast yaw axis strides
FORE-AFT** (a foot offset from the vertical yaw axis converts a yaw sweep into
forward travel — and the longer lever both lengthens the stride and widens the
stance at once). Numbers vs the combat geometry:

| metric | combat (as-designed) | improved | change |
|---|---|---|---|
| stance | 200 × 162 mm | 300 × 380 mm | wider + longer |
| foot offset from yaw axis | 31 mm | 110 mm | 3.5× lever |
| fore-aft stride per ±30° yaw | ~0 (was lateral) | 110 mm | forward stride |
| usable foot lift | 8.5 mm (knee −50 bias) | 45 mm (deep-knee −70) | 5× |

`sim/robot/drake_walk_proof.py` — RIGOROUS certificate via Drake's own MultibodyPlant
(mass matrix, gravity generalized forces, contact Jacobians) and a static-equilibrium
QP per single-support keyframe. **Result: the improved crawl is CERTIFIED at all four
keyframes** — COM margin **+46 mm** inside the stance triangle (2.2× the combat
geometry's +21 mm), active yaw holding torque only **0.3 N·m** (the self-locking worm
holds pitch and a braced knee holds the lift with no active power), all within servo
limits. MuJoCo dynamic sanity agrees: single-foot lift keeps the body level (±1°) with
the other three feet planted — which the real combat leg could not do.

**One design spec the proof surfaces:** each stance foot's ~42 N weight share flows
through the vertical knee/lift DOF, so the knee needs a **stance brace/detent rated
≥ ~45 N** (or enough servo force) to bear body weight — analogous to the worm
self-locking the pitch. The lift *servo* (26 N) only has to lift the unloaded swing
leg; the brace bears stance load structurally. This is the one addition to note
alongside the geometry changes: not a new actuator, a load path for the knee at stance.

**Division of labor:** Drake certifies the improved design positively; the real combat
leg's coupled failures (pitch stability cliff, foot-lift-vs-tilt coupling on the actual
slider-crank) were condemned separately in MuJoCo above. Together: the combat leg can't
walk; the three changes make a statically-stable crawl provably feasible.

## RL ON THE CERTIFIED GEOMETRY — feet lift under full physics (2026-07-04)

Warp-path RL (walker_warp_env.py + train_mesh_warp.py --geometry walker) trained on
the certified walker_improved geometry with imitation of reference_gait_walker.json.
GPU (A100), warp physics, ~14M steps in ~70 s each run.

**KEY MILESTONE: the robot lifts its feet and steps under FULL servo-true physics.**
- gate2/slow runs: sustained duty **0.83** (air-time 0.16) at alpha=1.0 for ~1.5M
  steps — the first sustained feet-lifting all session. The combat leg never left the
  ground in 88M steps across three runs.
- Render of the final policy (plain MuJoCo): duty **0.65**, rear feet lifting hard
  (per-foot duty [0.94, 0.99, 0.35, 0.33]), lurches forward at **0.27 m/s** (faster
  than the 0.12 command). It steps and propels.

**Two honest open frontiers (both tractable, neither a wall):**
1. *Reward landscape favors shuffle over clean stepping.* As entropy decays, velocity
   tracking (smooth) out-scores bouncy stepping, so duty drifts 0.83 → 0.96. Boosting
   MESH_IMIT_FEET_W (feet-swing reward) held it at 0.96 vs 0.98 but didn't fully win.
   Needs stronger step incentive (duty penalty / air-time dominance) — reward tuning.
2. *mujoco ↔ mujoco_warp sim-to-sim gap.* The policy is stable in the warp eval (duty
   0.96, no fall) but marginal in plain MuJoCo (falls at 1.6 s) — it tuned to warp's
   float32 contact dynamics. Closing this needs contact-parameter parity or
   domain-randomized contact during training (standard sim-to-real hygiene).

**Verdict:** the three geometry changes are validated end-to-end — from Drake static
certificate through RL that actually lifts feet and moves forward under real physics.
What remains is gait QUALITY (stable sustained walk), not FEASIBILITY. Artifacts:
sim/build/gpu/out/walker/ (walker_step.mp4, .pt checkpoints, run logs).

## BODY DESIGN AS AN OPTIMIZATION PROBLEM (2026-07-05)

`sim/robot/codesign.py` writes the body as an optimization and solves it, combining
the static Drake certificate (Tier 1) with a Level-A dynamic model:

  θ = (half_len, half_wid, yaw_lever, stance_h, lift_range, worm_ratio, belt_ratio)
  maximize  w1·Froude + w2·COM_margin + w3·stomp − w4·mass
  s.t.      COM_margin > 0 (static feasible), stomp ≥ 12 N·m (retain the weapon), bounds

- **STATIC** = the COM-in-support margin drake_walk_proof.py certifies (analytic
  surrogate in the loop; DEFAULTS Drake-certified separately).
- **DYNAMIC (Level A)** = the max Froude number Fr = v²/(gL) the servo torque-speed
  envelope + swing timing admit. v is capped by the slower of yaw-rate
  (≈2v/lev ≤ SERVO_FREE/belt) and pitch-cadence limits.
- **COMBAT** = pitch stall torque through the worm (self-locks at ratio ≥ 15).

**Walk-vs-combat Pareto (the tradeoff, quantified):** the optimizer holds the worm
at 20:1 (self-locking, 21 N·m stomp) until the dynamic weight dominates, then drops
it to the 11:1 combat floor (12 N·m, loses self-lock) to buy ~60% more speed
(0.05 → 0.08 m/s). Geometry wants to be as large as buildable (margin + stride both
grow with size) — the bounds are the operative constraint, i.e. "build the stance as
wide and the lever as long as you practically can."

**THE HEADLINE FINDING — the servo is the ceiling for dynamic gaits.** Across the
ENTIRE design space (loosest ratios, longest lever), the goBILDA servo tops out at
**Fr ≈ 0.027**. Dynamic walking begins at **Fr ≈ 0.10** (v ≈ 0.64 m/s at L = 0.42 m).
Reaching it needs a yaw no-load speed of 7.3 rad/s → a servo **~1.9× faster** than the
goBILDA's 15 rad/s. **No transmission ratio or geometry choice reaches the dynamic
regime with this servo at this scale.** Since Fr scales as 1/L, the levers are: a
higher-speed/power servo class, OR a smaller/lighter robot. Dynamic locomotion is a
**servo-class decision** — the optimization makes that quantitative rather than a hunch.

Level B (SLIP/spring-resonance limit-cycle co-design, optimizing the SEA belt
stiffness) is the natural next rung if a dynamic gait is pursued after a servo change.

**Honest limitations:** the inner-loop static/dynamic models are analytic
necessary-condition surrogates (Drake verifies the static side at in-range geometry;
the analytic optimum pushes to geometry extremes where the IK harness needs widening).
The Froude/limit-cycle criteria are necessary, not sufficient — a chosen design still
needs full-dynamics + RL verification.

## LEVEL B — SLIP spring-resonance co-design + servo shortlist (2026-07-05)

`sim/robot/codesign_slip.py` (SLIP limit-cycle) + `codesign.py` servo_report().

**Level B reverses Level A's speed verdict via series elasticity.** SLIP limit
cycles DO exist for this robot (m 8 kg, L 0.42 m) up to Fr 0.32 — open-loop unstable
at fixed touchdown angle (eig 1.8–4.6) but STABILIZABLE by per-stride touchdown-angle
control (low-bandwidth, servo-friendly; the spring carries the fast bounce). Best at
k~25 (≈4.6 kN/m leg-equiv, f_nat 3.85 Hz), gait v=1.16 m/s. **Energy check: the motor
must inject only ~0.5 W/leg (losses) — well inside even the current goBILDA's ~4 W/leg.**
So with a tuned SEA spring the SPEED wall dissolves: the spring does the fast work,
the motor tops up. (Resonance was detuned, f_nat 3.85 vs stance 3.16 Hz — tune to match.)

**Servo shortlist through both routes** (rigid = Level A, needs ~29 rad/s speed;
sprung = Level B, needs torque + a few W):

| servo | free rad/s | stall N·m | rigid-dynamic? | sprung-dynamic? |
|---|---|---|---|---|
| goBILDA 25-3 (current) | 15 | 1.06 | no | marginal (weak torque) |
| goBILDA 25-4 SuperSpeed | 30 | 0.53 | speed-yes but too WEAK | no |
| REEFS 400:SC V2 | 14 | 2.86 | no | **YES** (torque king) |
| Savox SB-2274SG | 13 | 2.45 | no | **YES** |
| Savox SB-2271/2291, Hitec 9360 | 16–21 | 1.7–2.0 | no | yes |
| DYNAMIXEL XL330 | 44 | 0.22 | speed-yes but far too weak | no |
| CubeMars AK80-9 (BLDC) | 60 | 22 | **YES** (does everything) | yes |

The torque-focused PWM servos DON'T help the rigid route (not faster) but DO enable a
dynamic gait via the SEA spring. Pure-speed hobby servos are a trap (fast, too weak).
CubeMars is the brute-force "real actuator" path (BLDC, not PWM, heavier/costlier).

**Three strategic branches:**
1. *Quasi-static crawl + combat* (current path): the goBILDA already works; a torque
   servo (REEFS/Savox) widens margin and loaded-drag. No spring needed.
2. *Dynamic via SEA spring, keep PWM* (elegant, uses the belt you designed): REEFS
   400:SC or Savox SB-2274 + a tuned SEA belt at k~25 / f_nat ~3.5 Hz resonant with
   the stride. Level B says it's energetically feasible even at these speeds.
3. *Dynamic via BLDC*: CubeMars AK80-9-class — most capable, different control stack.

Caveat: SLIP is a reduced-order necessary-condition model; Level C (full trajopt / RL
on the sprung leg) is the verification rung before committing hardware.

## BLDC MOTOR CO-DESIGN — the dynamic-gait path (2026-07-05)

`sim/robot/codesign_bldc.py` — torque-at-speed under THERMAL (continuous-current)
limits, per the design guidance, at 48 V bus. Kt = 9.55/Kv; output envelope is
voltage+current+thermal-limited through the belt.

**Every BLDC on the shortlist removes the actuator as the constraint.** Continuous
joint torque 16–37 N·m (vs the servo's ~1 N·m) and speed ceilings 5–25× the dynamic
threshold. The choice is mass / cost / thermal / ecosystem, NOT capability.

| motor | belt | τcont | τpeak | mass | $ | note |
|---|---|---|---|---|---|---|
| ODrive D6374 150Kv | 9:1 | 27 | 49 | 0.8 | 119 | **reference — documented, encoder-friendly** |
| Maytech 6374 170Kv | 12:1 | 24 | 40 | 0.8 | 87 | cheapest capable |
| Flipsky 6384 190Kv | 12:1 | 27 | 52 | 1.0 | 129 | cheap hi-torque |
| ODrive D5312s 330Kv | 27:1 | 21 | 42 | 0.2 | 129 | light, compact |
| mjbots mj5208 330Kv | 35:1 | 16 | 53 | 0.2 | 190 | lightest |
| T-Motor V807/V10L | 8–10:1 | 30–37 | 75–80 | 0.7–1.0 | 365–497 | overkill power |

(Froude ceilings shown by the tool are motor CAPABILITY, not gait predictions — they
say the motor could spin the stride joint 5–25× faster than a dynamic walk needs, i.e.
huge headroom; the real gait speed is set by dynamics/control, now unconstrained by
the actuator. Currents marked (est) are class estimates — refine from datasheets.)

**Key consequence: with a BLDC you may not need the SEA spring for dynamics at all.**
The spring (Level B) was the workaround for the weak servo — providing fast bounce so
the slow motor only topped up energy. A BLDC has BOTH speed and torque, so a rigid
(or lightly-sprung) dynamic gait is directly reachable; the SEA belt reverts to an
*efficiency/robustness* choice (energy recycling, blow absorption), not a necessity.

**Recommendation:** the **ODrive D6374 150Kv** (the stated default) is the right
reference — well-documented Kt, dual-shaft for a motor-side encoder, thermistor,
mid mass/cost, ~25× dynamic headroom. Cheap prototyping: Maytech/Flipsky 6374-class.
Lighter robot: ODrive D5312s or mjbots mj5208. Pair with an AS5047P motor-side
encoder + an ~85 A inverter (TIDA-010956-class) matching the 6374 peak current, for
the FPGA FOC. Level C (RL on the BLDC-driven leg) is the pre-hardware verification.

## PROVENANCE + AVAILABILITY STACK (2026-07-05)

`codesign_bldc.provenance_report()` — curated engineering judgment (doc maturity,
datasheet trustworthiness, availability, FOC/FPGA prior-art), weighted toward
provenance. Since capability is a non-issue across all BLDCs, THIS is the real
selection criterion for a self-FOC-on-FPGA build.

**The consistent pattern: "documented-for-robotics" beats "cheap-and-everywhere."**
- MOTOR: ODrive D6374 150Kv / D5312s (4.8) — the ONLY motors here whose vendor
  publishes measured Kt/R/L/thermal, with a dual shaft for the encoder + a thermistor
  (which the whole torque-at-speed-under-thermal analysis depends on). mjbots mj5208
  (4.3) is the open-robotics alternative. The e-skate motors (Maytech/Flipsky, 2.8)
  win availability but their specs are marketing, not measured — characterize yourself.
- ENCODER: AS5047P (5.0) — the universal 14-bit magnetic reference; ODrive/VESC/
  SimpleFOC all use it; stocked at Mouser/Digikey.
- POWER STAGE: TI DRV8353RS + external FETs (5.0), adapting TIDA-010956 (85 A ref
  design). TI = gold-standard docs + EVMs; the SPI-gate-driver split is ideal for
  "FPGA generates PWM, driver does analog + current-sense." VESC (3.8) is open/proven
  but it's a controller — FPGA FOC must bypass its MCU (architecture mismatch).
- FPGA: AMD/Xilinx Zynq class — most documented motor-control prior-art / FOC IP.

**Engineering wisdom (why provenance-first, then cheap):** on a first FPGA-FOC bring-up
you are debugging YOUR control loop — you need trustworthy Kt/R/L to tune the current
loop and a thermistor to hold the thermal envelope, so a fault is unambiguously your
FOC, not a mystery motor. Start documented (ODrive D6374 + AS5047P + DRV8353RS +
TIDA-010956 reference); once the control stack is proven, iterate mechanics on cheap
Maytech/Flipsky 6374s you bench-characterize. Provenance matters MOST early, least late.
