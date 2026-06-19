<!-- SPDX-License-Identifier: MIT -->
# Kinetic-realm expansion: URDF/SDF multibody load + standard viewer (closed-loop)

Expand the plant's "kinetic realm" from a single rigid inertia (`J`, `B`, scalar
`set_load_torque`) to a **real, URDF-described mechanism** — a geared, gravity-
loaded arm — driven by the **RTL FOC closed loop**, recorded to a standard format,
and watched move in a **standard robotics viewer**.

**North star:** the motor's electromagnetic torque drives a URDF multibody load
(simulated by a robotics dynamics engine); the resulting joint motion feeds the
angle-sensor model and closes the loop back to the RTL — and the whole thing is
**recordable + viewable** in a standard tool, with the project's parity/role
discipline intact.

## The honest boundary (built in)

- **Not real-time.** The cycle-accurate RTL co-sim runs slower than wall-clock, so
  visualization is **record → playback** (scrub/loop at chosen speed), not a live
  real-time render. (HIL is the later bridge — §7.)
- **Parity-anchored.** The multibody engine is a *model*; it must reproduce the
  existing C++ 1-DOF plant in the degenerate case (no gear, no gravity, matched
  J/B) before any richer mechanism is trusted — same ethos as the C++/Python/
  Modelica plant parity.
- **Assumed mechanism.** Gear ratio, arm mass/length, gravity, backlash/friction
  are `assumed` params (provenance-tagged) until a real mechanism is measured.

## §0 — Stack decision + the role (decide first)

- [ ] **Dynamics engine (URDF/SDF-native, Python-drivable).** Recommend
      **PyBullet** (pip, URDF-native, built-in GUI for quick live dev view,
      simple step API). Alternatives, same role/structure: **Drake** (SDF/URDF,
      MeshCat viewer, the most principled multibody) or **MuJoCo** (MJCF, best
      contact). Pick one; the `i_mechanical_load` backend is engine-agnostic.
- [ ] **Recording + viewer (the standard-viewer goal).** **URDF** geometry +
      **MCAP** recording + **Foxglove Studio** (web/desktop, reads MCAP + URDF,
      shows the 3D scene *and* the control plots side-by-side — no full ROS
      install). Keep the engine-native GUI (PyBullet) for quick live viewing
      while developing; **RViz2 / rosbag2** is the ROS-native alternative.
- [ ] **The `i_mechanical_load` role** (`sim/cpp/src/i_mechanical_load.hpp` or a
      Python-side role): the closed-loop contract at the **motor shaft** —
      `advance(dt, motor_torque) -> shaft_angle, shaft_velocity`. The existing
      1-DOF C++ plant is the **parity backend**; the URDF engine is the **rich
      backend**. Mirrors `i_gate_driver`/`i_angle_sensor`.

## §1 — Bench coupling (the crux: C++ co-sim ↔ Python URDF engine)

- [ ] **Expose the motor electromagnetic torque** from the bench (a `torque_nm`
      accessor; `three_phase_plant` already computes `torque_n_m`) — the input to
      the external mechanism.
- [ ] **External-mechanical mode** on the bench: a flag that **disables the
      internal 1-DOF mechanical integration** and accepts an externally-imposed
      rotor state each coupling step (`set_rotor_state(theta, omega)` — a
      generalization of `set_speed_clamp`, which only imposes ω). The external
      engine owns *all* mechanics; the bench keeps the electrical + EM-torque.
- [ ] **Coupling contract:** units (N·m / rad / rad·s⁻¹), **sign convention**
      (one flip silently destabilizes — pin it with the parity test), and a
      **coupling step** that is a sub-multiple of the control/PWM period (too
      coarse → co-sim instability, the classic algebraic-loop problem).
- [ ] **Co-sim driver** (`sim/scripts/cosim_kinetic.py`): the loop — step the
      bench a coupling period → read `torque_nm` → step the engine with that
      torque → read joint state → `set_rotor_state(...)` on the bench (its angle
      sensor reads the new motor-shaft angle). Closed loop.

## §2 — The mechanism (URDF)

- [ ] **Author the URDF** (`hw/mechanisms/geared_arm.urdf`): motor shaft →
      gearbox (reduction N) → revolute joint → arm link (mass, length, inertia)
      under gravity. **The angle sensor reads the MOTOR shaft, before the gear**,
      so resolution + latency scale by N — the gear-ratio × sensor coupling
      (an extension of the M8 motor↔sensor finding).
- [ ] **Parameterize** the gear ratio, arm mass/length, gravity from `params.toml`
      (`[mechanism.geared_arm]`, `status="assumed"`); generate the URDF from
      params (the one-source pattern, like the KiCad schematic generator).
- [ ] **A degenerate URDF** (single inertia, gear ratio 1, gravity 0, J matched to
      the C++ plant) for the §1 parity check.

## §3 — The closed-loop scenario

- [ ] **Position-hold against gravity:** command a joint angle; the FOC must hold
      the arm at any angle against the position-dependent gravity load — the
      headline "watch the controller hold the arm" demo.
- [ ] **Move + reversal:** a joint-angle step and a fast reversal through the
      geartrain (reuse the reversal stress pattern); **backlash-on-reversal** if
      the URDF/engine models lash.
- [ ] Run the full chain: RTL FOC ↔ electrical ↔ EM torque ↔ URDF multibody ↔
      joint angle ↔ angle sensor ↔ RTL. Confirm zero shoot-through, stable hold.

## §4 — Recording (standard format)

- [ ] **Extend the trace schema** with the mechanical channels: joint angle/ω
      (commanded + actual), motor-shaft angle, motor torque (+ the existing
      currents/duties).
- [ ] **Emit MCAP** (`sim/scripts/record_kinetic.py`): the channels above + the
      URDF, at a chosen sample rate → `figures/kinetic/<scenario>.mcap`. Decoupled
      so the visualization is replayable and viewer-agnostic.

## §5 — Visualization (the standard viewer)

- [ ] **Foxglove layout** (`figures/kinetic/foxglove_layout.json`): a 3D panel
      (URDF + joint states) **+** synced plots (commanded vs actual joint angle,
      motor torque, phase currents) — the control dashboard. Document opening the
      `.mcap` + URDF in Foxglove.
- [ ] **Engine-native live view** (PyBullet `GUI`): a `--gui` flag on the co-sim
      driver to watch the arm move while developing.
- [ ] **Headless playback render** for the docs/README (like `motorloop.gif`): a
      scripted off-screen render of the arm + a telemetry strip → MP4/GIF, so the
      result is visible without installing a viewer.

## §6 — Tests, parity, integration

- [ ] **Parity test** (`test_kinetic_parity.py`): the degenerate URDF backend
      reproduces the C++ 1-DOF plant (same torque-in → same θ(t)) within
      tolerance — the trust anchor.
- [ ] **Closed-loop tests** (`test_kinetic.py`): position-hold settles + holds
      against gravity (steady-state error bounded), zero shoot-through; the
      **gear-ratio × sensor** coupling — a higher reduction needs the AS5047P more
      (M8-extended), or record "no modeled difference" honestly.
- [ ] **Deps + Make:** `pip install pybullet mcap` (+ Foxglove app, user-side);
      a `make kinetic` target (run co-sim → record MCAP → headless render);
      REUSE/SPDX on all new files; URDF/MCAP covered in `REUSE.toml`.
- [ ] **Report + gallery:** `notes/kinetic-realm-report.md` + `figures/kinetic/`
      (the playback render + a Foxglove screenshot) — the "see it controlled" result.

## §7 — Honest boundary / future (HIL)

- Not real-time (record→playback). Degenerate-parity-validated; mechanism params
  `assumed`. The same `i_mechanical_load` role is the bridge to **hardware-in-the-
  loop** later (RTL on the FPGA, real motor, the mechanism emulated live) via
  FMI-for-real-time / **DCP** — out of scope here, but the contract is designed
  for it.

## Done-when

A URDF geared-gravity-arm runs under the RTL FOC closed loop (position-hold +
reversal, zero shoot-through), the degenerate case is parity-checked against the
C++ plant, the run records to **MCAP + URDF** and opens in **Foxglove** as a 3D +
telemetry dashboard, and a headless playback render lands in `figures/kinetic/`.
`make kinetic` reproduces it.

## What NOT to do

- Don't claim real-time — it's record→playback; say so.
- Don't trust the multibody engine before the degenerate parity check passes.
- Don't bury the geometry in code — generate the URDF from params (one source),
  and let the *same* URDF feed the engine **and** the viewer.
- Don't couple at too coarse a step — keep the co-sim step a sub-multiple of the
  control period and verify stability.
- Don't forget the sensor is on the motor shaft (pre-gear) — the gear ratio scales
  its resolution/latency demand.

## Implemented (results)
_(to fill in after execution)_
