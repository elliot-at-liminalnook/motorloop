<!-- SPDX-License-Identifier: MIT -->
# Universal command contract (v2 observations)

> **Document status:** Active · **Audience:** Robot-learning contributors · **Last reviewed:** 2026-07-23 · **Canonical for:** Commands-only policy observations, per-rung command signatures, and rung invisibility

The accepted `universal256:...+task31:v1` contract ends every observation with a
31-way one-hot naming the exact ladder rung, and the actor's FiLM conditioning
reads it. That grants the network permission to implement thirty-one loosely
related policies behind one parameter set: rung identities are an orthogonal
basis with no metric, so nothing rewards interpolation between speeds, headings,
or task combinations that lack their own index, and a deployed robot has no
rung index to offer. The v2 contract removes the task ID from the policy
entirely. **Rungs select environment distributions, rewards, and promotion
gates; they are not observable.** Everything the policy must do is stated by
explicit command channels, and two rungs that present identical commands are
thereby asserting that they demand identical behavior.

## Observation layout

`universal256:physical211+actuator_mask14+command31:v2`, served by the
`universal_command` geometry (`UniversalCommandWarpEnv`). Total width, actuator
mask, and action space are unchanged from v1; only the trailing 31 channels
change meaning, and the actor's conditioning input becomes the command block.

| block index | channels | content |
| --- | --- | --- |
| 0–2 | 3 | velocity command (walker three-channel convention) |
| 3 | 1 | velocity command active |
| 4–5 | 2 | commanded heading error (sin, cos) |
| 6 | 1 | heading command active |
| 7 | 1 | height command (always meaningful; nominal stand height when unset) |
| 8–19 | 12 | authority-normalized pose command (zeros = neutral stance) |
| 20–21 | 2 | planar goal (family perception frame: locomotion local goal, combat opponent offset) |
| 22 | 1 | goal active |
| 23–26 | 4 | selected-leg one-hot (masked by engage) |
| 27 | 1 | engage (attack system armed) |
| 28 | 1 | stepping-cadence scaffold value |
| 29 | 1 | stepping-cadence scaffold active |
| 30 | 1 | reserved (always zero) |

Every value is derived from the underlying environment's observation tensor,
never from live simulator state, so terminal observations carry the pre-reset
commands they were experienced under. Activity flags come from a static
per-rung table inside the environment adapter — that table *is* the executable
form of the signatures below. The gait phase clock and kick-phase clock remain
where they were, inside the physical block, as observable retiring scaffolds;
they are deliberately excluded from the command contract.

## Scaffold channels

Channels 28–29 are the one declared **scaffold command**: rung 6 must lift and
replace feet at zero commanded velocity, an acquisition objective that is not
an honest deployment command. Per the training objective contract, the scaffold
is explicit, observable, and retiring: it is active only while the stepping
prerequisite is trained, later rungs never assert it, and no promotion gate may
condition on it. If stepping-in-place ever becomes a genuine product behavior
(commanded readiness stepping), the channel graduates to a permanent command by
a recorded contract change, not by silent reuse.

## Per-rung command signatures

"—" means inactive/zero. Rungs sharing a row assert that identical commands
demand identical behavior; distribution/physics differences live in the
environment, not the observation.

| rungs | signature | notes |
| --- | --- | --- |
| 1 | (not universal) | component-level torque proof, no policy contract |
| 2, 3 | velocity active, value zero | commanded hold; pushes (3) are environment distribution |
| 4 | pose command nonzero, velocity inactive | pose tracking |
| 5 | velocity active zero + height command varying | crouch/stand tracking |
| 6 | velocity active zero + cadence scaffold active | declared scaffold rung |
| 7, 8 | velocity active, forward values | 7's fixed speed is a subset of 8's range |
| 9, 10 | velocity active (yaw rate / full planar+yaw) | turn and omnidirectional tracking |
| 11 | velocity active + heading active | heading regulated while translating |
| 12 | velocity active, value switches to zero mid-episode | true stop; identical signature to 2/3 at zero — deliberately the same demanded behavior |
| 13–18 | velocity active | physics/terrain rungs: servo-true droop, trip bar, pushes, tiles, slope, payload are environment distribution only |
| 19–23 | goal active (+ velocity inactive) | navigation family; degraded lidar (23) is perception distribution |
| 24, 27 | goal active (opponent offset), engage off/env-derived | approach and pursuit |
| 25 | goal active + engage as armed | close and strike |
| 26 | engage + selected-leg one-hot | the commanded-leg contract |
| 28, 29 | goal active + engage as armed | fights; leg selection flows from the attack system when armed |
| 30 | velocity active | cross-morphology tracking; design identity arrives via morphology tokens, not commands |
| 31 | (not universal) | co-design search loop, no policy contract |

## Walk-first acquisition order

`--walk-first` (requires this contract) reorders execution to
`1, 8, 10, 9, 11, 12, 7, 2, 3, 4, 5, 6, 13…`: velocity tracking is acquired
from scratch at rung 8 — whose command distribution holds a fixed stripe of
worlds at v=0 so commanded standing is learned *with* locomotion — and the
stand/pose/step rungs then certify as commanded special cases through the
zero-shot exam. Rationale: a randomly initialized policy starts with nonzero
foot-air, so every stepping-adjacent reward has gradient from the first
update; the classic stand-first order instead builds a standing attractor
that stepping must later escape across a zero-gradient basin (the rung-6
failure of 2026-07-24). Rung numbering is invisible to the policy, so
execution order is pure curriculum; gates are unchanged and warm-start/
retention/replay follow acceptance order rather than numeric order.

## Zero-shot certification under v2

The v1 test-out exam copied the predecessor's task-channel conditioning into
the new rung's channel before evaluating. Under v2 there are no per-rung
channels to inherit: the exam evaluates the parent checkpoint directly under
the next rung's command distribution. This is the same competency-based
allocation with a stronger meaning — passing now demonstrates generalization on
the shared command manifold rather than transfer of a copied embedding.

## Migration rules

- v2 is a contract break by design: `task_conditioning_semantics` becomes
  `command_film_v2` and the trainer refuses to resume v1 checkpoints into it.
- The accepted `task_film_gru` v1 ladder remains the archived baseline.
- The predictive family launches on `--geometry universal_command` from rung 1
  in its own output tree; it must not inherit the rung ID it is meant to
  retire. Trainer-side task-conditioning inheritance is skipped for
  command-conditioned environments.
