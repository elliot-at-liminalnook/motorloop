<!-- SPDX-License-Identifier: MIT -->
# Training objective contract

> **Document status:** Current · **Audience:** Robot-learning contributors · **Last reviewed:** 2026-07-14 · **Canonical for:** Deciding what may be rewarded, constrained, gated, or only observed in the universal-controller ladder

The permanent objective says **what the robot must accomplish**, not how a
successful motion should look. A surprising policy is acceptable when it
achieves the commanded outcome, respects physical limits, remains controllable,
and does not exploit the simulator.

## Four roles that must not be conflated

| Role | Meaning | Lifetime |
| --- | --- | --- |
| Outcome | Externally visible task success: command tracking, reaching, hitting, stopping, or using every foot in the stepping prerequisite | Permanent |
| Physical constraint | A genuine safety or mechanism boundary: fall, planted-foot slip, unsafe orientation, joint speed, violent body motion, collision, current, or thermal limit | Permanent and non-negotiable |
| Efficiency cost | Resource or wear trade-off: energy, impact, heating, action rate, or unnecessary motion | Permanent only when the real system pays that cost; report a Pareto trade-off |
| Scaffold | Imitation, reference gait, phase agreement, symmetry, action demonstration, or other acquisition hint | Temporary, observable, and automatically retiring |

A style signal must never become a promotion gate merely because it correlates
with success in one run. The executable manifest rejects known style-only gates,
including gait-clock agreement, diagonal synchronization, swing timing,
reference-action agreement, and duty-cycle shape.

## Current locomotion contract

Constraints-as-Terminations are restricted to physical failures:

- planted-foot slip;
- unsafe body orientation;
- excessive joint velocity; and
- excessive vertical or roll/pitch body velocity.

Progress, support pattern, duty cycle, phase, and gait family are not
catastrophes. They remain telemetry, and direct task outcomes determine
promotion.

Rung 6, **Step in place**, now requires:

- every foot is physically airborne for at least 5% of the held-out evaluation;
- base speed remains below the in-place limit;
- the robot remains upright; and
- physical constraint and fall rates remain below their limits.

Its dense training signal uses recent per-foot activity, stillness, and
uprightness. A learned competence multiplier temporarily strengthens missing
foot exploration and releases after coverage is sufficient. No reward or gate
specifies foot order, diagonal pairing, cadence, symmetry, phase, or joint
trajectory. Legacy feed-forward actors retain a neutral clock channel for
checkpoint compatibility; the recurrent actor masks it and must organize time
from physical history. Agreement with a clock has no value by itself.

Forward and later locomotion rungs are promoted on command tracking, progress,
direction, stopping, navigation success, balance, and physical safety. A
temporary walking teacher may seed exploration, but its optimizer weight
anneals and yields to safety pressure. It is disabled unless a versioned teacher
artifact is explicitly configured; the accepted retention anchor does not
implicitly authorize imitation. The permanent reference-gait reward is zero by
default. From rung 6 onward, the base walker's fixed gait-clock, airtime,
nominal-pose, and clearance style rewards are also zero.

Rung 7 uses signed progress normalized by its physical acceptance target. The
generic Gaussian velocity-tracking proxy is disabled there: at the gentle
discovery command it gave a nearly stationary policy 91% tracking credit and
supplied 78.4% of absolute reward while direct progress remained unmet. A proxy
that saturates before its outcome passes must be removed or recalibrated, not
countered by increasing an arbitrary competing weight.

New task channels inherit the exact behavior of the preceding accepted channel
before PPO begins. This transfers capability without describing a target gait:
only the new task-encoder column changes, old task outputs remain untouched, and
the new column is free to specialize. Categorical task channels keep frozen
normalization statistics so an unseen one-hot cannot jump to the normalization
clip or drift while the physical sensor statistics continue adapting.

Before PPO, an inherited task may test out with zero gradient updates. This is
allowed only through the ordinary external outcome and physical-constraint
gates on five fresh full episodes, with additional normalized margin, followed
by the complete prior-skill retention matrix. Test-out never introduces a style
gate, weakens a safety boundary, or declares success from training reward. A
failed certification automatically allocates PPO experience to that task.

Commanded-leg attack credits selected-leg hits, physical kick/recovery motion,
wrong-leg avoidance, and support from the other legs whenever they occur. The
old hard-coded strike/recovery phase windows no longer decide whether the same
physical result counts.

The experimental predictive controller expresses the same contract as a masked
future interaction rather than a reference animation. Locomotion requests body
displacement and velocity without foot timing. A selected-leg attack requests a
broad opponent contact region, impact direction, selected-versus-wrong-leg
contact, and aggregate support at any time in the planning horizon. Joint poses,
gait order, and a fixed strike frame remain unspecified. Prediction-gradient
guidance may optimize this intent only while dimensionless held-out forecast
calibration grants it authority; identical-seed guidance-off evaluation must
show whether the extra planning changes real outcomes.

## Diagnostics that protect behavioral freedom

Every evaluation records:

- the distribution and normalized entropy of all 16 four-foot contact patterns;
- the complete 4×4 foot-contact correlation matrix;
- per-leg duty and airborne fractions, including worst-leg use and balance;
- gait-clock, clearance, and diagonal-synchronization scores as diagnostics;
- absolute reward share by outcome, physical constraint, efficiency, and
  scaffold; and
- a warning when scaffold magnitude exceeds real outcome magnitude.

These measurements reveal collapse, asymmetry, or an unexpected gait without
turning those observations into choreography. Video remains required to decide
whether an unfamiliar solution is physically credible.

Adaptive pressure must use the same physical definition and a compatible time
window as held-out evaluation. Exact episode contact coverage drives rung-6
competence; the shorter contact EMA is telemetry only. Scaffold coefficients are
bounded so self-tuning changes emphasis without continuously changing the scale
of the critic's return target. The critic loss is normalized by the rollout
return standard deviation, leaving its optimum unchanged while keeping gradient
clipping meaningful across reward scales.

## Review test for every new signal

Before adding a reward or gate, answer all of the following:

1. What external success or real physical cost does this measure?
2. Could a different-looking behavior satisfy the same requirement safely?
3. Would this signal reject that behavior only because it violates our
   expectation of what the motion should resemble?
4. Can the quantity be normalized by a physical target instead of balanced by
   an arbitrary raw weight?
5. If it is only an exploration hint, what measured competence makes it retire?
6. Which stand-still, dragging, oscillating, falling, reset-farming, asymmetric,
   and actuator-saturating fixtures attempt to exploit it?
7. Is it reported separately by leg, seed, episode, and distribution tail?

If question 3 is yes, the quantity belongs in diagnostics or a temporary
scaffold, not the permanent objective.

## Checkpoint migration

Reward-only migrations preserve the actor, observation normalization, physical
runtime, safety duals, RNG, and monotonic schedule progress. They reset the
critic, optimizer moments, and reward-semantic competence duals. This prevents a
multiplier learned for an old style target from silently carrying that target
into the new objective.
