<!-- SPDX-License-Identifier: MIT -->
# RL environment — results (NN learns locomotion over the FOC)

Implements [`rl-environment-checklist.md`](rl-environment-checklist.md): a fast
Gymnasium + MuJoCo + Stable-Baselines3 environment where a neural network learns
locomotion by commanding joints **through a per-joint motor + FOC torque-speed
envelope** (`sim/rl/actuator.py`, reusing the motor profiles). The motor choice
is part of the environment.

Code: `sim/rl/` (`actuator.py`, `foc_env.py`, `train.py`, `eval.py`,
`render_rollout.py`); parity `sim/tests/test_rl_actuator_parity.py`; figures
`sim/scripts/gen_rl_figures.py`. Reproduce: `make rl-train` → `make rl-eval` →
`make rl-figures` (needs `~/rl-venv`; `requirements-rl.txt`).

## Honest frame

- **Fast surrogate, not the cycle-accurate sim** (RL needs millions of steps; the
  RTL co-sim is far too slow — deliberately separate tiers).
- **Reduced inner loop:** the actuator is the FOC torque-speed envelope, *parity-
  checked* against the cycle-accurate bench (below), not the full RTL.
- **CPU demo budget:** 1.5 M steps (~5 min, 16 envs). A *runnable gait*, not a
  SOTA policy. The point is the pipeline + the motor coupling, not the score.
- **Sim-to-real gap** remains; domain-randomization hooks mitigate, the
  verification tier + HIL close it later.

## What was built + measured

- **Task:** HalfCheetah-v5 (6-joint locomotion), NN = high-level policy over the
  FOC; action scaled by each joint's motor torque envelope at its current speed.
- **Training (PPO + VecNormalize):** the policy **learned to run** — raw return
  **3131 ± 43** vs a random baseline of **−233**. (VecNormalize obs+reward and
  small minibatches were essential; without them PPO did not learn.) Video:
  `figures/rl/halfcheetah_db42.mp4` (the headline "watch it run").
- **§5 motor coupling** (`figures/rl/`): the *same trained policy* run under each
  motor envelope gives a **monotonic** ordering — gimbal **3096** < DB42 **3131**
  < maxon **3166**. The spread is modest at gear 2 (the cheetah's joints stay
  mostly within all three motors' capability); `motor_envelope.png` shows the
  mechanism — the gimbal's torque starves far earlier than the servo's, so a
  higher gear or a faster gait would widen the gap. The coupling is real and in
  the expected direction (stronger motor → higher return).
- **§6 parity** (`test_rl_actuator_parity.py`): the reduced FOC actuator matches
  the cycle-accurate bench's achievable current within ~20–30 % in the operating
  regime (both show the back-EMF rolloff); it is *optimistic at extreme speed*
  where the real loop saturates — a documented limitation, which is exactly why
  the verification tier exists.

## The through-line

The motor profiles (`motors.py`) — the same Kt/R/Ke/pole-pairs that set the RTL
controller and drive the cycle-accurate plant — now also **constrain what the
learned policy can do**. So the agent learns under the real motor's limits, the
motor choice is a measurable lever on achievable performance, and the policy is a
candidate to validate against the RTL FOC (and, later, hardware). That continuity
— silicon → controller → plant → learned robot, on one honest set of values — is
the result.

## Engineering notes (the traps hit + fixed)

- Python 3.14 (Ubuntu 26.04) has no torch/mujoco wheels → a `uv`-managed **Python
  3.12** venv (`~/rl-venv`); **CPU torch** (the cu130 build crashed on this no-GPU
  box).
- Headless render needs **OSMesa** (`MUJOCO_GL=osmesa`); gymnasium's own renderer
  crashes there, and **torch tensor construction after a live mujoco context
  corrupts the heap** — so rendering is decoupled (record state in the policy
  process, render torch-free in `render_rollout.py`), and the policy is loaded
  *before* the env is created.

## Still open (the honest boundary)

Per-motor *full training* (not just eval-swap), domain-randomized sim-to-real,
the RTL-in-the-loop policy eval, and HIL/hardware — all flagged in the checklist
as later tiers built on the same actuator contract.
