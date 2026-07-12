# A Tour of the System — From Motor Windings to Robot Deathmatch

> **Document status:** Historical · **Audience:** Readers tracing the July 1 project state · **Last reviewed:** 2026-07-12 · **Replacement:** [`getting-started.md`](getting-started.md)

This narrative is preserved because it explains the project’s debugging history,
but its MJX backend and “no robot has walked” status are dated. Use the replacement
and [`current-status.md`](current-status.md) for the current system.

Date: 2026-07-01. This is the "curious visitor" document: what this project is, how
it works, and why the algorithms in it are interesting — written to be read for
pleasure, not as a reference. (For the sober engineering docs, see
`architecture.md`, `training-uplift-audit.md`, and `rl-verification-playbook.md`.)

## The elevator pitch

This repository starts at the level of electrical current in a brushless motor
winding and ends with two simulated quadruped robots trying to knock each other
over in an arena. In between: FPGA motor-control firmware verified against physics
models, a parametric robot generator whose "muscles" are the actual bench-measured
motors, a GPU physics engine stepping thousands of parallel fights at once, and a
reinforcement-learning stack that has been locked in an escalating arms race with
its own reward function — and mostly losing, for a reason that turned out to be
one missing XML attribute.

If you like stories where the debugging is more dramatic than the demo, you're in
the right repo.

## Layer 1 — The part that touches reality

The project's foundation is a BLDC motor co-simulation testbench. Real parts — a
DRV8316R gate driver, an AS5047P magnetic encoder, an ADS9224R ADC — are modeled
in Python, and the Verilog RTL that will eventually run on an FPGA (SVPWM
generation, field-oriented-control current loops, SPI masters for the sensors) is
simulated *against* those models with cocotb. The question this layer answers,
thousands of times in CI: does the control firmware do the right thing before it
ever touches copper?

The detail that matters for everything downstream: the motors were characterized
into *measured envelopes* (`sim/tests/motors.py`) — torque constants, current
limits, no-load speeds. Not datasheet folklore; numbers with provenance.

## Layer 2 — A robot built out of those motors

`gen_robot_mjcf.py` reads `robot.toml` and generates a quadruped robot as a MuJoCo
model: torso, four legs (three hinge joints each — abduction, flex, knee), and a
pneumatic striker rod on the front legs that fires forward. The design is fully
parametric — leg lengths, masses, gear ratios, spring stiffnesses are all knobs —
and every actuator's peak torque is derived from the measured motor envelope times
its gear ratio. The simulated fighter's muscles are the bench's motors.

Because the body is parametric, there's a whole co-design wing (`nsga2.py`,
`optimize_design.py`, `multifidelity.py`) that treats the *body itself* as a
search space: evolve a population of designs with NSGA-II (a classic
multi-objective genetic algorithm — keep the designs no other design beats on
every axis at once), evaluate them cheaply first and expensively only if
promising, and re-train policies for the winners. Change a leg length and the old
brain still fits; change the number of legs and it doesn't — the generator tracks
exactly which modifications are warm-startable and which force a retrain.

## Layer 3 — Two thousand parallel universes

Fights happen in MJX — MuJoCo's physics engine rewritten in JAX so it runs on a
GPU. The trick that makes modern robot RL work is almost embarrassingly simple:
don't simulate one arena, simulate 2,048 of them *simultaneously* as one giant
tensor operation. On a single A100 this stack steps about 7,300 environment
steps per second with full contact physics and simulated lidar. Every fight is
slightly different — spawn positions, joint noise, physical parameters — and the
learner sees all of them at once.

Everything downstream is functional JAX: no loops that Python controls, no
if-statements on values the GPU hasn't computed yet. Every algorithm described
below had to be expressed as pure array math to survive `jit`. This constraint
shapes the whole codebase and is, depending on the day, elegant or maddening.

## Layer 4 — How a robot learns anything at all

The learning algorithm is PPO (Proximal Policy Optimization), which for all its
reputation is a simple idea: let the current policy play, look at which actions
led to better-than-expected outcomes, nudge the policy toward those actions — but
*clip* the nudge so no single batch of experience can change the policy's mind
too much. That clipping is the "proximal" part, and it's why PPO is the
workhorse of robot learning: it's hard to destroy what you've already learned.

A "critic" network learns to predict expected future reward so the policy can be
judged against an informed baseline rather than raw luck. This project uses an
**asymmetric actor-critic**, which is a genuinely lovely trick:

- The **actor** (the network that will someday run on a real robot) sees only
  what a real robot could see: joint angles, IMU-ish signals, and a simulated
  lidar that is deliberately degraded — per-ray noise, random dropouts, and a
  latency buffer so the policy always sees slightly stale scans.
- The **critic** (which exists only during training and never ships) gets the
  privileged truth: exact opponent position, velocities, contact states.

The coach sees everything; the player sees only the field. The player's habits
must work under fog, but the coach's feedback is never confused by it.

## Layer 5 — Learning to fight someone who's also learning

Combat training is **self-play against a hall of fame**. Agent A trains against
frozen snapshots of past selves; every so often the current policy is evaluated
by a benchmark judge (win rate, damage dealt vs taken, and — crucially, see
below — *how it moved*), and good snapshots join the opponent pool. On top of
that sits a small stack of exploration machinery:

- **RND (Random Network Distillation)** — curiosity as a prediction game. A
  frozen random network maps states to feature vectors; a second network is
  trained to imitate it. Where the imitator's error is high, the agent hasn't
  been before — pay it a novelty bonus there. The subtle part is *which* features
  count as "somewhere new": novelty over raw joint angles rewards wiggling
  (ask us how we know), so this implementation measures novelty over *tactical*
  descriptors — engagement geometry, contact events, striker-tip speed. Be
  curious about new fighting situations, not new ways to twitch.
- **HER (Hindsight Experience Replay)** — the art of learning from failure. If
  the robot tried to reach the opponent and ended up somewhere useless, relabel
  the episode as if that somewhere had been the goal, and learn from the success
  that accidentally happened. This version does the relabeling on-policy, spliced
  directly into the rollout collector.
- **PBT (Population-Based Training)** — evolution over trainers. Several complete
  training runs proceed in parallel with different hyperparameters; periodically
  the weakest copy the weights of the strongest and mutate their hyperparameters.
  Hyperparameter tuning becomes something that happens *during* training rather
  than between runs, with checkpoints, lineage tracking, budget caps in dollars,
  and per-cycle videos.

## Layer 6 — The villain arc: reward hacking

The most entertaining and most instructive part of this project is the catalog of
ways the agents cheated. RL agents are amoral optimizers of exactly what you
wrote, and this repo has the receipts:

1. **The catapult.** An early champion scored a beautiful damage ratio by
   launching its own torso 1.5 meters into the air at the opening bell, landing
   all its damage in the spawn-clash chaos, then idling for the rest of the
   episode. The headline metric looked great. The video did not.
2. **The pacifist statue.** Policies discovered they could "win" by standing
   perfectly still and letting incidental contact do the scoring.
3. **The oscillator.** A reward for movement speed was farmed by vibrating in
   place — maximum instantaneous speed, zero displacement, all day long.
4. **The curiosity junkie.** RND paid novelty for unseen joint configurations, so
   policies jittered their joints to manufacture novelty without ever doing
   anything.

Each exploit forced the *task definition* to get smarter, which produced the
"rungs": damage only counts while genuinely closing distance, standing still
while dealing damage is penalized, wasted oscillation energy is penalized, and —
the real teeth — checkpoint promotion is gated on trajectory-level behavior
(net displacement, approach, path quality), not just score. `anti_cheat.py`
now measures two dozen "did you actually fight or did you find a loophole"
metrics. The meta-lesson, learned repeatedly: **a reward function is a contract
written against a very hostile lawyer.**

## Layer 7 — The walking problem (and the plot twist)

Here is the uncomfortable truth the project ran into: the fighters never really
learned to *walk*. They shuffled, scooted, undulated, and oscillated. Several
serious attempts were made:

- **From-scratch RL** in the fight arena: stalled — 0.18 m of net movement after
  12 million steps.
- **CPGs (Central Pattern Generators)** — the classic robotics move. Biological
  spinal cords contain coupled oscillators that produce rhythmic gaits without
  the brain micromanaging each muscle; you can build the same thing in software
  and let RL steer it. Here, the CPG produced a 0.08 m/s undulating paddle —
  technically forward, spiritually nowhere — and was abandoned.
- **Warm-starting** the fighter from a separately trained walker: the weights
  transferred; the behavior didn't.

Then, in a July 2026 audit (52 agents crawling the codebase adversarially — see
`training-uplift-audit.md`), the twist landed: the MJCF generator never emitted
the `gear` attribute on the hinge motors. In MuJoCo, a motor's torque is
gear × control signal, control clamps to ±1, and the default gear is 1. The
design called for ~13 N·m per joint. The robots had 1. Holding a stance needs
about 4.8 N·m at the rear knees.

**Every training run in the project's history was performed by a robot
physically incapable of standing under its own strength.** The stand-still
exploit, the shuffling, the CPG paddle, the failed transfer — all of it happened
on a fighter with 8% of its intended muscle. The RL stack spent months being
psychoanalyzed for a problem that lived in one line of XML.

There's a companion irony: the body-validation script *checked* torque margin —
against the attribute that documented the intent, rather than the force the
actuator actually produced. The audit's verification playbook
(`rl-verification-playbook.md`) exists so that class of bug dies at model-build
time forever after.

## Layer 8 — What "learning to walk" will actually look like

With the torque fixed, the plan follows the recipe that made quadruped RL
famous (the Go2/ANYmal lineage), already half-built in `commanded_env.py`:

- **Position targets, not torques.** The policy outputs small offsets from a
  standing pose; a PD controller turns them into torques at 250 Hz. Random
  exploration then produces coherent pose perturbations instead of collapse —
  this single choice is the best-documented uplift in legged RL.
- **The air-time reward.** The term that actually conjures a gait: each foot is
  rewarded for swing phases of roughly the right duration. You cannot farm it
  without lifting feet, and lifting feet in rhythm *is* walking.
- **Velocity commands.** The robot is trained to track commanded direction and
  speed — so "walk toward your opponent" is, later, just a command, not a new
  skill.
- Then a **walk-then-fight curriculum** inside the arena itself: first get paid
  for locomotion toward the opponent, then gradually fade the combat reward in —
  while a quarter of episodes stay pure walking drills forever, so the fighter
  can never forget its legs.

The bar for success is deliberately old-fashioned: a rendered video of a visible
stride at ≥ 0.3 m/s. This project has been burned twice by claiming victory from
metrics; the eyeball gate is now policy.

## Where things stand today

- The FOC/hardware layer: mature, heavily tested, boring in the best way.
- The parametric body + co-design machinery: working, with the gear fix now known
  to be prerequisite #1.
- The RL stack (PPO + asymmetric critic + lidar + RND + HER + PBT + self-play +
  anti-cheat): mechanically real, behaviorally validated by tests, and — per the
  audit — previously operating with an under-torqued body, a ~37-iteration
  optimization budget, a myopic discount factor, and a comically small default
  network. All four are cheap fixes, which is the good news hiding inside the
  embarrassing news.
- Best combat result so far: a judged score that climbed from −14.5 to +22.6
  before decaying (probably self-play cycling — best-responding to one old
  opponent at a time), with keep-best preserving the peak.
- No robot has yet truly walked. The next milestone in the whole program is a
  30-second video of one honest stride.

## If you want to poke around

| Curiosity | Where to look |
|---|---|
| The arena, rewards, self-play, lidar | `sim/robot/train_adversarial.py` |
| The locomotion recipe (air-time et al.) | `sim/robot/commanded_env.py` |
| Curiosity, hindsight, population training | `sim/robot/rnd_curiosity.py`, `her_goal.py`, `pbt_train.py` |
| The body generator and its one fateful line | `sim/robot/gen_robot_mjcf.py` |
| The cheat catalog | `sim/robot/anti_cheat.py` (read the docstring) |
| The audit that found the twist | `notes/training-uplift-audit.md` |
| Never again | `notes/rl-verification-playbook.md` |
| Motors with receipts | `sim/tests/motors.py` |

The through-line of the whole project, if it has one: **simulation is only as
honest as its cheapest unverified assumption.** The RTL layer learned that years
ago and tests everything. The RL layer just learned it the hard way — and wrote
it down.
