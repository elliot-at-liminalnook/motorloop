# Training Issues in SOTA Robotics RL (2024–2026): Survey and Project Mapping

Date: 2026-07-03. Method: 8 parallel research threads (60+ web searches, ~50 primary-source
fetches: arXiv, simulator docs, GitHub issues), followed by a citation-verification pass
(12 highest-risk references re-fetched and confirmed against arXiv). Claims that rest on
search snippets rather than fetched sources are marked **[thin]**. Focus: legged locomotion
RL, sim-to-real, and adversarial/multi-robot training.

---

## 1. Sim-to-real transfer

### SOTA approaches

- **DrEureka** (RSS 2024, [arXiv:2406.01967](https://arxiv.org/abs/2406.01967)) — LLM
  writes reward *and* domain-randomization configs jointly. Fixes: manual DR-range tuning
  hell. Demonstrated quadruped-on-yoga-ball transfer.
- **ASAP** (CMU/NVIDIA, RSS 2025, [arXiv:2502.01143](https://arxiv.org/abs/2502.01143)) —
  learns a *delta/residual action model* from real Unitree G1 data, then fine-tunes the
  policy in the corrected sim. Explicitly motivated by "SysID and DR result in overly
  conservative policies that sacrifice agility." Beat SysID, DR, and delta-dynamics
  baselines on agile whole-body tracking.
- **SPI-Active** (CMU, 2025, [arXiv:2505.14266](https://arxiv.org/abs/2505.14266)) —
  massively parallel sampling-based system ID with active exploration (Fisher-information
  maximizing excitation trajectories); 42–63% improvement over baselines. Fixes DR
  over-conservatism by *estimating* parameters instead of randomizing over them.
- **Sim-to-real without dynamics randomization** (ETH, 2025,
  [arXiv:2509.06342](https://arxiv.org/abs/2509.06342)) — physics-grounded PMSM
  actuator/energy model with a minimal parameter set transfers across 13 legged robots
  with **no dynamics randomization** (32% cost-of-transport reduction on ANYmal). Strong
  evidence that good actuator modeling can replace most DR.

**Actuator modeling.** The actuator-network lineage was born on series-elastic actuators
(Hwangbo et al., Science Robotics 2019, [arXiv:1901.08652](https://arxiv.org/abs/1901.08652)):
supervised NN from position-error/velocity history to torque, inserted into sim before
policy training. 2025–26 refinements: **Generalized Actuator Networks** for tendon/muscle-
actuated robots learn the model *without torque sensors*, from joint-position trajectories
only, capturing nonlinearity, friction, and hysteresis
([arXiv:2604.09487](https://arxiv.org/abs/2604.09487), verified); residual actuator
networks with proprioceptive distribution matching
([arXiv:2604.11090](https://arxiv.org/html/2604.11090v1)); **trajectory-based actuator ID
via differentiable simulation** — fits actuator models from encoder data alone by
backprop-through-simulator, validated on a high-gear-ratio actuator with embedded PD
([arXiv:2604.10351](https://arxiv.org/abs/2604.10351), verified). Backlash and
torque-speed saturation treated inside an RL pipeline: Lin et al., CoRL 2025
([arXiv:2502.20396](https://arxiv.org/abs/2502.20396)) **[backlash detail thin]**.

**Contact fidelity.** Sim-to-sim validation is now standard practice because contact-solver
differences (PhysX impulse vs MuJoCo soft-constraint) are not expressible as parameter DR:
**Humanoid-Gym** (2024, [arXiv:2404.05695](https://arxiv.org/abs/2404.05695)) trains in
Isaac Gym and gates on MuJoCo replay before hardware. Joint static friction alone
materially shifts learned gaits ([arXiv:2503.01255](https://arxiv.org/abs/2503.01255)
**[thin]**).

**Teacher-student.** RMA (2021, [arXiv:2107.04034](https://arxiv.org/abs/2107.04034)) →
DreamWaQ (ICRA 2023, [arXiv:2301.10602](https://arxiv.org/abs/2301.10602), removes the
two-phase pipeline via a VAE context estimator) → **Hybrid Internal Model** (ICLR 2024,
[arXiv:2312.11460](https://arxiv.org/abs/2312.11460)): contrastive learning of an implicit
response embedding, fixing (a) information loss when the student imitates a privileged
teacher and (b) infeasibility of explicitly estimating friction/heightmaps from noisy
sensors. **Parkour in the Wild** (ETH, 2025,
[arXiv:2505.11164](https://arxiv.org/abs/2505.11164)) distills multi-expert teachers then
*RL-fine-tunes the student* — direct evidence pure distillation leaves performance on the
table. Real-world fine-tuning canon: Smith et al. 2021
([arXiv:2110.05457](https://arxiv.org/abs/2110.05457)); world-model route: SimDist (2026,
[arXiv:2603.15759](https://arxiv.org/abs/2603.15759), verified).

### Open problems
1. DR ranges are still chosen heuristically; no accepted pre-deployment predictor of
   transfer success.
2. Structural gaps (backlash dead-zones, cogging, non-backdrivability) cannot be fixed by
   randomizing parameters the simulator does not have; residual/delta-action models
   (ASAP-style) are promising but untested on high-backlash gearing.
3. Real-to-sim methods need real data from exactly the agile/contact-rich behaviors that
   are unsafe to collect.

---

## 2. Reward specification and hacking

### SOTA approaches

- **Constraints as Terminations (CaT)** (IROS 2024,
  [arXiv:2403.18765](https://arxiv.org/abs/2403.18765),
  [code](https://github.com/Gepetto/constraints-as-terminations)) — constraint violations
  trigger stochastic termination of future returns instead of weighted penalties; minimal
  PPO change; real Solo-12 parkour. Fixes reward-scale tuning and the
  penalty-overwhelmed-by-task-reward failure — which is *worst under self-play, where
  opponent-relative returns inflate*. Off-policy variant: SoloParkour (CoRL 2024,
  [arXiv:2409.13678](https://arxiv.org/abs/2409.13678)).
- **Relaxed log-barrier style rewards** (2024,
  [arXiv:2409.15780](https://arxiv.org/html/2409.15780v1)) — soft constraints on gait
  timing, foot clearance, body height, joint velocity replace both hand-weighted sums and
  mocap-dependent AMP; produced 4.67 m/s gallop. Lagrangian lineage: PPO-Lag / PID-dual
  variants (empirical study 2025, [arXiv:2510.17564](https://arxiv.org/html/2510.17564v1)).
- **LLM-assisted reward design**: Eureka (2023,
  [arXiv:2310.12931](https://arxiv.org/abs/2310.12931)) → DrEureka (2024, above; adds
  explicit safety instructions because raw generated rewards produce sim-exploiting
  policies) → **RDA: Reward Design Agent** (2026,
  [arXiv:2606.01672](https://arxiv.org/abs/2606.01672), verified) — VLM visually evaluates
  trajectories, not just scalar fitness; plus VLM reward models trained on robot corpora
  ([arXiv:2603.16065](https://arxiv.org/html/2603.16065) **[thin]**).
- **Style/AMP rewards and their exploitation**: AMP discriminators are gameable networks —
  policies find narrow transition distributions the discriminator scores as "real" (mode
  collapse, style-reward saturation). **NEAR** (ICLR 2025,
  [arXiv:2501.14856](https://arxiv.org/abs/2501.14856)) replaces the discriminator with
  annealed energy-based rewards, explicitly motivated by AMP's non-smooth, exploitable
  reward landscape; **APEX** (2025, [arXiv:2505.10022](https://arxiv.org/pdf/2505.10022))
  replaces adversarial training with action priors.

**Documented exploit classes** (locomotion): stand-still/survival farming — directly
documented for the `feet_air_time` term in Isaac Lab
([discussion #1977](https://github.com/isaac-sim/IsaacLab/discussions/1977),
[issue #1955](https://github.com/isaac-sim/IsaacLab/issues/1955)): threshold-based
air-time reward makes stepping net-negative, so policies freeze; stock mitigation gates
the term on commanded-velocity norm. Air-time terms are also farmed by pronking/one-foot
hopping; standard fixes: reward only at touchdown, clamp per-contact air time, pair with
foot-slide penalty. Flip/catapult solver-energy exploits and high-frequency PD oscillation
are handled by orientation/base-contact terminations, action-rate and action-acceleration
penalties, and hard joint-velocity/torque constraints — increasingly as CaT terminations
rather than penalties.

**Eval-time exploit discovery**: **Rethinking Robustness Assessment** (RSS 2024,
[arXiv:2405.12424](https://arxiv.org/abs/2405.12424)) — learned sequential multi-modal
adversarial attacks (command + observation + physical perturbation) break SOTA quadruped
controllers on real hardware; framed as pre-deployment vulnerability discovery. VLM
rollout audits: AHA (2024, [project](https://aha-vlm.github.io/)), manipulation-focused.

### Open problems
1. No shared benchmark or taxonomy of locomotion reward hacks; exploits are
   physics-engine-specific, so fixes do not transfer between engines.
2. Constraint thresholds are still hand-set — constraint hacking replaces reward hacking
   at the margins; Lagrange multipliers oscillate.
3. Nobody audits LLM-generated rewards for hackability before training, and no published
   system searches for reward hacks specifically (attack work targets robustness, not
   "is the score earned legitimately"). Self-play doubles as exploit discovery but nothing
   formalizes harvesting the exploits opponents find.

---

## 3. Sample efficiency and massive parallelization (GPU sim ecosystems)

### Isaac Lab / Isaac Sim (NVIDIA, [repo](https://github.com/isaac-sim/IsaacLab))
Most feature-complete (tiled cameras, warp lidar, terrains, rsl_rl/rl_games, DexPBT).
Documented pitfalls: **training nondeterminism even with fixed seeds**
([issue #275](https://github.com/isaac-sim/IsaacLab/issues/275); officially acknowledged —
GPU scheduling + runtime material randomization; PhysX only deterministic per
hardware/version, [docs](https://isaac-sim.github.io/IsaacLab/main/source/features/reproducibility.html)),
making reward/hyperparameter A/B tests noisy by construction; static GPU buffers that
crash or silently corrupt when undersized; **stale link poses/contact/lidar data after
`env.reset()`** poisoning first-step observations
([known issues](https://isaac-sim.github.io/IsaacLab/main/source/refs/issues.html));
heavy API churn (Isaac Gym → OIGE → Isaac Lab; quaternion convention changes).

### MuJoCo MJX + mujoco_playground + brax
([MJX docs](https://mujoco.readthedocs.io/en/stable/mjx.html); **MuJoCo Playground**, 2025,
[arXiv:2502.08844](https://arxiv.org/abs/2502.08844); [brax](https://github.com/google/brax))
Strengths: bit-deterministic under JAX (unlike PhysX); ~417k steps/s Go1 joystick on one
A100; proven zero-shot sim-to-real on 6 platforms. Documented pitfalls:
- **Contact cost scales with *possible*, not active, contacts** (JAX static shapes).
  Playground's own mitigations: `max_contact_points=6`, `max_geom_pairs=4`, explicit
  `<pair>` culling, Newton solver with 1–2 iterations. For a *two-robot* scene the
  possible robot-robot pair set grows quadratically — enumerate pairs or throughput craters.
- Elliptic-cone gaps in some MJX paths
  ([mujoco#2153](https://github.com/google-deepmind/mujoco/issues/2153)); pyramidal is the
  safe default; no PGS/noslip.
- **`mjx.ray` has no heightfield support**
  ([mujoco#2155](https://github.com/google-deepmind/mujoco/issues/2155),
  [#1821](https://github.com/google-deepmind/mujoco/issues/1821)); no plugin sensors —
  lidar over terrain needs workarounds.
- **brax AutoResetWrapper resets to a *cached* initial state** — per-env init
  randomization is frozen for the whole run ([brax#174](https://github.com/google/brax/issues/174));
  wrapper state leakage ([brax#422](https://github.com/google/brax/issues/422)); check
  truncation-vs-termination bootstrapping and obs-normalization mode in
  [ppo/train.py](https://github.com/google/brax/blob/main/brax/training/agents/ppo/train.py).
  No first-class multi-agent scenes: a 2-robot env is one big model with hand-split rewards.
- **MuJoCo Warp** ([repo](https://github.com/google-deepmind/mujoco_warp),
  [docs](https://mujoco.readthedocs.io/en/latest/mjwarp/)) is the designated successor
  (>100x faster than MJX-JAX on contact-heavy scenes, now powering Playground envs;
  available at MJX HEAD since Aug 2025). Caveats: NVIDIA-only, no autodiff, heightfield
  envs showed training NaNs (being fixed), reward parity with MJX-JAX good but not exact.

### Genesis (Dec 2024, [repo](https://github.com/Genesis-Embodied-AI/Genesis))
Launch claimed 43M FPS; independent analysis found ~150x overstatement on realistic
workloads (1 substep, idle-step hibernation, self-collisions off) — see
[Stone Tao's analysis](https://stoneztao.substack.com/p/the-new-hyped-genesis-simulator-is)
and [Genesis#181](https://github.com/Genesis-Embodied-AI/Genesis/issues/181); the team
later published a [corrected benchmark](https://github.com/zhouxian/genesis-speed-benchmark).
Legged RL works but ecosystem depth (terrains, curricula, sensors, sim-to-real recipes)
trails Isaac Lab and Playground.

### Sample-efficiency levers at scale
Curriculum + asymmetric actor-critic + DR remain the core levers
(Rudin et al. 2021, [arXiv:2109.11978](https://arxiv.org/abs/2109.11978); rsl_rl library
paper 2025, [arXiv:2509.10771](https://arxiv.org/pdf/2509.10771)). 2025 shift to off-policy
at scale: **FastTD3** ([arXiv:2505.22642](https://arxiv.org/abs/2505.22642), verified) —
1000+ envs, 32k batches, distributional critic; solves HumanoidBench in <3 h on one A100
and runs on Playground envs. Caution when porting off-policy: Isaac-style action bounds of
[-100,100] that PPO's Gaussian never explores break tanh-squashed algorithms
([Raffin's SAC-at-scale series](https://araffin.github.io/post/sac-massive-sim/)).
PBT: DexPBT ([arXiv:2305.12127](https://arxiv.org/abs/2305.12127)); population-level
hyperparameter evolution beats single-run PPO/SAC on GPU sims (PBRL benchmark 2024,
[arXiv:2404.03336](https://arxiv.org/abs/2404.03336)).

### Open problems
1. Contact richness vs speed (static-shape penalty in JAX; Warp closes some of the gap).
2. Determinism at scale on PhysX; no ecosystem offers both Isaac-level features and
   MJX-level reproducibility yet.
3. Multi-agent APIs are an afterthought in every GPU sim.

---

## 4. Self-play / multi-agent instabilities

### SOTA approaches

- **AlphaStar league** (Nature 2019,
  [blog](https://deepmind.google/discover/blog/alphastar-grandmaster-level-in-starcraft-ii-using-multi-agent-reinforcement-learning/)) —
  canonical fix for cycling and counter-forgetting: main agents (PFSP vs whole league) +
  main exploiters + league exploiters, with periodic exploiter resets. PFSP weightings:
  f_hard(x)=(1−x)^p to focus on hardest opponents; f_var(x)=x(1−x) for similar-strength
  matches (confirmed in the 2024 self-play survey,
  [arXiv:2408.01072](https://arxiv.org/html/2408.01072v4)). Cheaper exploiters: **Minimax
  Exploiter** (2023/24, [arXiv:2311.17190](https://arxiv.org/abs/2311.17190)) attacks the
  main agent's value function. **OpenAI Five** (2019,
  [arXiv:1912.06680](https://arxiv.org/pdf/1912.06680)): 80% latest / 20% snapshot pool
  with online-updated opponent quality scores — a hall-of-fame with learned priorities.
- **PSRO/NeuPL lineage**: NeuPL-JPSRO (2024,
  [arXiv:2401.05133](https://arxiv.org/abs/2401.05133)) — whole population in one
  conditioned net, converges to coarse correlated equilibria with visuomotor transfer;
  Self-Play PSRO ([arXiv:2207.06541](https://arxiv.org/pdf/2207.06541)) reduces
  exploitability anytime; GEMS (2025, [arXiv:2509.23462](https://arxiv.org/pdf/2509.23462))
  drops the O(n²) meta-game payoff matrix.
- **Robot-vs-robot 2024–26**: DeepMind 1v1 humanoid soccer (Science Robotics 2024,
  [arXiv:2304.13653](https://arxiv.org/abs/2304.13653),
  [journal](https://www.science.org/doi/10.1126/scirobotics.adi8022)) — teacher skills →
  distill → self-play vs snapshot pool; behavior regularization prevented
  hardware-destroying motions. **RoboStriker** (2026,
  [arXiv:2601.22517](https://arxiv.org/abs/2601.22517), verified: hierarchical humanoid
  boxing) — mocap tracker → latent-skill distillation → latent-space neural fictitious
  self-play with AMP warmup and a reservoir strategy buffer; names the two core tensions:
  *physical feasibility vs non-stationary learning* and *strategy evolution vs system
  stability*; evaluated by 8x8 cross-play tournament. **MQE** (IROS 2024,
  [arXiv:2403.16015](https://arxiv.org/abs/2403.16015)) — multi-quadruped competitive
  benchmark incl. sumo (sim-only). Lineage anchor: Bansal et al. 2017 sumo/wrestling
  self-play ([arXiv:1710.03748](https://arxiv.org/abs/1710.03748), delta-uniform opponent
  sampling). No published real-hardware quadruped-vs-quadruped combat RL was found —
  the niche appears open **[absence claim, not exhaustively verifiable]**.
- **Asymmetric information**: privileged critics now have theory — finite-time bounds
  (2025, [arXiv:2501.19116](https://arxiv.org/pdf/2501.19116)) and analysis of when
  privileged signals help (2025, [arXiv:2509.26000](https://arxiv.org/html/2509.26000v1)).
  Giving the critic opponent true state during training is standard CTDE and sound.
- **Exploitability evaluation**: exact exploitability intractable; standard practice is a
  frozen-champion best-response audit — train a fresh RL adversary, report its win margin
  as a lower bound (IJCAI 2022, [arXiv:2004.09677](https://arxiv.org/pdf/2004.09677)) —
  plus cross-play tournament matrices.

### Interactions with the rest of a training stack (thin but important literature)
- **HER inside adversarial play is unsound as-is**: relabeling "the state I reached was my
  goal" is invalid when the outcome was co-authored by an adversary — the relabeled
  transition is off-distribution under any fixed opponent, biasing the critic (stated
  explicitly for RoboCup-style tasks in
  [arXiv:2105.00499](https://arxiv.org/pdf/2105.00499)). Salvageable: HER on the
  *non-adversarial skill layer* only; the adversarial analog of hindsight is subgame/state
  curricula (Subgame Curriculum Learning,
  [arXiv:2310.04796](https://arxiv.org/pdf/2310.04796)). No 2024–26 paper does HER inside
  competitive self-play.
- **RND + self-play**: the opponent already provides an auto-curriculum; a non-stationary
  opponent makes everything look permanently novel, inflating intrinsic reward, and agents
  can farm novelty instead of winning. Practical guidance: anneal RND once win-rate-vs-pool
  is informative, or restrict novelty to opponent-independent state dims
  (**[inference; no combat-specific study exists]**; cf. intrinsic-coefficient scaling,
  [arXiv:2507.19725](https://arxiv.org/html/2507.19725v1)).
- **PBT + self-play**: proven combo (OpenAI Five, TiZero
  [arXiv:2302.07515](https://arxiv.org/pdf/2302.07515)) but PBT's exploit/copy step can
  collapse population diversity exactly when the opponent pool needs it — AlphaStar made
  diversity roles *structural* (exploiters) rather than emergent. Keep the PBT selection
  metric = PFSP-weighted performance vs the whole pool, never vs latest-only.

### Open problems
1. Principled hall-of-fame admission/eviction; ELO is ill-defined under non-transitivity.
2. Exploitability auditing at physics-sim cost (best-response training is expensive).
3. Contact-rich adversarial sim-to-real (two robots striking each other) is untouched.

---

## 5. Closed kinematic chains and nonstandard transmissions

### How the simulators handle loops
- **MuJoCo (CPU)**: loops closed with `equality` constraints (`connect`, `weld`,
  `joint`/`tendon` polynomial couplings); all constraints *soft* with `solref`/`solimp`
  impedances — drift is tunable but real
  ([computation docs](https://mujoco.readthedocs.io/en/stable/computation/index.html)).
- **MJX (JAX backend)**: equality types CONNECT, WELD, JOINT, TENDON are now supported
  (the old [mujoco#1105](https://github.com/google-deepmind/mujoco/issues/1105) gap is
  closed); fixed and spatial tendons supported; joint/site/tendon actuator transmissions
  work ([MJX docs](https://mujoco.readthedocs.io/en/stable/mjx.html)). **MJX-Warp**
  supports all equality and tendon types, all solvers except PGS/noslip, but no autodiff
  ([repo](https://github.com/google-deepmind/mujoco_warp)).
- **Drake**: no loop joints; sanctioned pattern is cut-the-loop + constraint
  (`AddBallConstraint`, `AddWeldConstraint`, `AddDistanceConstraint`,
  `AddCouplerConstraint` — exactly a gear/belt ratio) on MultibodyPlant, **discrete mode
  with SAP solver only**; SAP enforces constraints in a stabilized near-rigid regime
  ([API docs](https://drake.mit.edu/doxygen_cxx/classdrake_1_1multibody_1_1_multibody_plant.html)).
  Drake has a genuine **ScrewJoint** (1-DOF, pitch + damping,
  [docs](https://drake.mit.edu/doxygen_cxx/classdrake_1_1multibody_1_1_screw_joint.html)) —
  a kinematically exact model for worm/screw stages.
- **Isaac/PhysX**: articulations must be trees; the excluded loop-closing D6 joint
  "accumulates the most error"
  ([Isaac Sim docs](https://docs.isaacsim.omniverse.nvidia.com/6.0.0/robot_setup_tutorials/rig_closed_loop_structures.html)).

### RL with loops and exotic drives
- **BRUCE / Mechanical-Intelligence-Aware Curriculum RL** (2025,
  [arXiv:2507.00273](https://arxiv.org/abs/2507.00273), verified) — simulates differential
  pulley, five-bar, and four-bar closed chains *natively* in GPU sim (MJX), curriculum RL,
  zero-shot real deployment beating an MPC baseline; explicitly blames simulator loop
  support for the field's habit of serializing linkages.
- **Closed-chain bipedal RL** (2025, [arXiv:2507.10164](https://arxiv.org/abs/2507.10164),
  verified) — explicit closed-chain dynamics in training; simplified-serial baselines
  significantly worse across terrains.
- **SEA/belt compliance in RL**: standard recipe is motor-rotor-as-extra-joint +
  spring-damper (tendon or joint-equality), randomize stiffness/damping so the policy
  observes the compliance state; when modeling stalls, learn the whole transmission map as
  an actuator network (Hwangbo lineage; GenAN
  [arXiv:2604.09487](https://arxiv.org/abs/2604.09487) for the no-torque-sensor case; PPO
  force control learned directly on an SEA because backlash+stiction defeated sim modeling,
  [arXiv:2304.04911](https://arxiv.org/abs/2304.04911)). Gear-transmission sysID template:
  ROBOTIS-OP3 gear model with friction/backdrivability fit from failed rollouts
  ([arXiv:2204.03897](https://arxiv.org/abs/2204.03897)).
- **Non-backdrivable drives: essentially no RL literature.** Canonical argument against
  them for dynamic legs: MIT Cheetah proprioceptive actuation / Impact Mitigation Factor
  (Wensing et al., IEEE TRO 2017,
  [doi](https://dl.acm.org/doi/abs/10.1109/TRO.2016.2640183)) — high-ratio drives kill
  impact tolerance and force transparency. Worm-gear mechanics exist
  (load-dependent break-in friction, [Shoaib et al. 2023](https://journals.sagepub.com/doi/full/10.1177/16878132231191031));
  **WAVE** (2025, [arXiv:2509.21878](https://arxiv.org/html/2509.21878v1)) pairs a worm
  gear with a series elastic element precisely to decouple impacts — but none involve
  learned policies. Neither MuJoCo nor MJX has a native worm friction cone; practical
  approximation is large `frictionloss` + high `damping` (both MJX-supported), with true
  load-dependent locking checked in Drake (ScrewJoint + joint friction).

### Open problems
1. No published sim-to-real of a learned policy through a self-locking drive.
2. MJX soft equality vs Drake SAP near-rigid constraints will disagree quantitatively on
   loop reaction forces — over-center/toggle holding forces are the predictable worst case.
3. Loop-aware robot description formats are still being invented (URDF+
   [arXiv:2411.19753](https://arxiv.org/abs/2411.19753); extended URDF
   [arXiv:2504.04767](https://arxiv.org/abs/2504.04767)).

---

## 6. Safety and hardware-in-the-loop training

### SOTA approaches

- **CBF-RL** (2025/ICRA 2026, [arXiv:2510.14959](https://arxiv.org/abs/2510.14959)) —
  CBF filtering applied to *training rollouts* so the deployed policy internalizes
  constraints and needs no runtime filter; fixes the filter-fights-the-policy conservatism.
- **Agile But Safe** (RSS 2024, [arXiv:2401.17583](https://arxiv.org/abs/2401.17583)) —
  agile policy + recovery policy, switched by a learned reach-avoid value network; Go1 at
  >3 m/s among obstacles. The dual-policy + learned-value-switch pattern is directly
  reusable for combat (swap "avoid collision" for "avoid losing configurations").
- **Robot-Trains-Robot** (CoRL 2025, [arXiv:2508.12252](https://arxiv.org/abs/2508.12252)) —
  force-aware robot-arm gantry actively supports a humanoid student: protection,
  support-force curriculum, perturbation injection, auto-resets; real-world learning from
  scratch. Predecessor: reset-free multi-task training (Ha et al. 2020,
  [PMLR](https://proceedings.mlr.press/v155/ha21c/ha21c.pdf)).
- **Protective falling**: SafeFall (2025, [arXiv:2511.18509](https://arxiv.org/pdf/2511.18509))
  — fall prediction + RL protective policy minimizing peak impact force with
  component-weighted penalties; FR-Net fall recovery (2025,
  [arXiv:2509.11504](https://arxiv.org/pdf/2509.11504)). For a combat robot, being thrown
  is expected — a protective-reflex layer under the fight policy is the applicable pattern.
- **Hardware-lifetime constraints in training**: thermal-aware locomotion RL (2026,
  [arXiv:2603.01631](https://arxiv.org/abs/2603.01631), verified: 7 min → 27+ min to
  thermal shutdown on A1); actuator-envelope-constrained RL enforcing the speed-dependent
  torque limit, not just peak torque (2023/24,
  [arXiv:2312.17507](https://arxiv.org/html/2312.17507v1)).
- **HIL actuator characterization**: cycloidal QDD with dynamometer-in-the-loop GRU torque
  estimation across friction-dominated regimes, incl. thermal characterization (2024,
  [arXiv:2410.16591](https://arxiv.org/abs/2410.16591), verified) — the clearest published
  bench-to-sim workflow; differentiable-sim actuator ID from encoders only
  ([arXiv:2604.10351](https://arxiv.org/abs/2604.10351)) covers post-assembly re-ID.
- **Conservative on-robot updates**: SLowRL rank-1 (LoRA-style) adaptation on a Go2 with a
  training-time recovery policy (2026, [arXiv:2603.17092](https://arxiv.org/pdf/2603.17092));
  RTR optimizes a single dynamics latent in the real world instead of full weights.

### Open problems
1. Safety for *deliberately adversarial* robot-robot contact is unpublished territory;
   contact-force regulation during intentional impact ("hit hard without spiking gearbox
   torque") has no literature. Nearest neighbors are sim-only (MQE sumo) or attack-style
   robustness audits ([arXiv:2405.12424](https://arxiv.org/abs/2405.12424)).
2. No work couples fall/impact protection with gearbox shock limits; impact torque through
   a non-backdrivable stage has no escape path (series elasticity is the known mitigation,
   cf. WAVE).
3. True HIL-in-the-training-loop (policy queries a physical dyno during RL) is essentially
   unpublished — everything is characterize-then-simulate.

---

## 7. Generalist / foundation policies

### SOTA approaches
- **Cross-embodiment locomotion**: ManyQuadrupeds (ICRA 2024,
  [arXiv:2310.10486](https://arxiv.org/abs/2310.10486)); URMA "One Policy to Run Them All"
  (2024, [arXiv:2409.06366](https://arxiv.org/abs/2409.06366)); Body Transformer
  (CoRL 2024, [arXiv:2408.06316](https://arxiv.org/abs/2408.06316)); **Embodiment Scaling
  Laws** (CoRL 2025, [arXiv:2505.05753](https://arxiv.org/abs/2505.05753)) — ~1,000
  procedural embodiments; embodiment diversity beats more data on fixed embodiments.
- **Distillation from RL specialists**: **HOVER** (ICRA 2025,
  [arXiv:2410.21229](https://arxiv.org/abs/2410.21229)) — oracle teacher → DAgger student
  with command masking, one net covering multiple command modes; **ExBody2** (2024/25,
  [arXiv:2412.13196](https://arxiv.org/abs/2412.13196)) — reports the
  versatility-vs-specialization tradeoff explicitly (fine-tuning some skills degrades
  others: the skill-interference warning).
- **Humanoid foundation efforts**: GR00T N1 (2025,
  [arXiv:2503.14734](https://arxiv.org/abs/2503.14734)) — manipulation-focused VLA;
  marketing-forward, low locomotion relevance.
- **Directly relevant to adversarial play**: **Lifelike Agility and Play** (Tencent
  Robotics X, Nature Machine Intelligence 2024,
  [arXiv:2308.15143](https://arxiv.org/abs/2308.15143)) — VQ motion primitives pretrained
  on dog mocap; a strategic layer then learns a 2-robot chase-tag game with sparse win/loss
  reward + prioritized self-play; zero-shot to a real 14 kg quadruped; beat human
  teleoperators. Pretraining let self-play spend its samples on *tactics*, not motor
  control. The closest published analog to quadruped combat.

### Verdict and open problems
For a single-embodiment project, cross-embodiment generality is mostly irrelevant cost;
the reusable ideas are the HOVER/ExBody2 distillation recipe and the Tencent
primitives-then-self-play staging. Reward design and sim-to-real are explicitly *not*
retired by foundation trends. Open: skill interference in distilled generalists; optimal
pretraining mix; whether generalist pretraining helps competitive play at all (single
strong data point, no systematic study).

---

## 8. Benchmarking and reproducibility

### SOTA approaches
- **Empirical Design in RL** (Patterson et al., JMLR 2024,
  [arXiv:2304.01315](https://arxiv.org/abs/2304.01315)) — the field's how-to-run-RL-
  experiments reference (seed counts, hypothesis testing, experimenter bias); complements
  rliable (NeurIPS 2021) whose uptake in *locomotion sim-to-real* papers remains rare.
  Newer: percentile-based run-variation reporting (2026,
  [arXiv:2606.06746](https://arxiv.org/abs/2606.06746), verified); sequential
  statistically-rigorous policy comparison cutting hardware trials ~70% (2026,
  [arXiv:2603.13616](https://arxiv.org/abs/2603.13616), verified).
- **Evaluation-protocol reform**: "Robot Learning as an Empirical Science" (2024,
  [arXiv:2409.09491](https://arxiv.org/abs/2409.09491)) — success rates reported without
  run counts/initial conditions/criteria; calls for failure narratives; sim-metric ↔
  real-outcome alignment measurement (NVIDIA 2025,
  [arXiv:2508.11117](https://arxiv.org/abs/2508.11117)); RoboArena distributed multi-lab
  eval (2025, [arXiv:2506.18123](https://arxiv.org/abs/2506.18123)).
- **Benchmark fragmentation**: Playground vs legged-gym-lineage vs HumanoidBench
  ([arXiv:2403.10506](https://arxiv.org/abs/2403.10506)) use different rewards,
  terminations, and contact defaults — numbers are not comparable across families; no
  cross-engine leaderboard exists.
- **PPO implementation sensitivity**: the "37 details" lineage
  ([ICLR blog 2022](https://iclr-blog-track.github.io/2022/03/25/ppo-implementation-details/));
  formal hyperparameter-sensitivity metric showing apparent algorithmic gains often come
  from increased tuning dependence (NeurIPS 2024,
  [arXiv:2412.07165](https://arxiv.org/abs/2412.07165)).

### Cross-engine verification (the Drake-sidecar question)
Sim-to-sim validation is mainstream but almost exclusively Isaac(PhysX)-train →
MuJoCo-verify (Humanoid-Gym; PHUMA [arXiv:2510.26236](https://arxiv.org/abs/2510.26236);
Isaac Lab now ships official Newton↔PhysX sim-to-sim docs,
[link](https://isaac-sim.github.io/IsaacLab/main/source/experimental-features/newton-physics-integration/sim-to-sim.html)).
**No paper or repo was found using Drake as the verification engine** — Drake appears in
RL only as a training backend via drake-gym
([repo](https://github.com/ggould-tri/drake-gym)). Which engine *pairs* have usefully
uncorrelated failure modes is uncharacterized — an open research lane the MJX+Drake
combination sits in.

### Open problems
1. No accepted seed/rollout standard for a locomotion claim; distributional reporting not
   enforced.
2. No published head-to-head of rsl_rl PPO vs brax/Playground PPO despite both being
   ecosystem defaults with different advantage normalization, clipping, and minibatching.
3. Rigorous-evaluation work targets manipulation/generalists; legged locomotion evaluation
   is still often a video.

---

## Highest-leverage items for THIS project (ranked)

Stack recap: MuJoCo/MJX + brax PPO on A100; asymmetric actor-critic + simulated lidar; PD
position targets; air-time gait rewards; self-play (hall-of-fame + PFSP); RND; HER; PBT;
reward-hacking history (catapult / stand-still / oscillation); Drake sidecar; hardware with
series-elastic belt yaw, self-locking worm pitch, toggle-press blade foot.

1. **Convert anti-hack penalties to CaT-style terminations**
   ([arXiv:2403.18765](https://arxiv.org/abs/2403.18765)). Penalty terms lose the arms
   race precisely in self-play, where opponent-relative returns inflate and swamp fixed
   penalties. Base-contact, base-height-spike (catapult), joint-velocity, and action-rate
   limits as stochastic-termination constraints is the single most transplantable fix for
   the project's documented exploit history.
2. **Fix the air-time term the way Isaac Lab had to**
   ([#1955](https://github.com/isaac-sim/IsaacLab/issues/1955),
   [#1977](https://github.com/isaac-sim/IsaacLab/discussions/1977)): gate on nonzero
   command/engagement (else stand-still farming), reward only at touchdown with a clamped
   per-contact max (else pronk/one-foot farming), pair with a foot-slide penalty. This
   exact term has a documented stand-still failure mode — the same exploit already seen here.
3. **Audit brax wrappers for two silent bugs**: AutoResetWrapper resets to a *cached*
   init state, freezing per-env initial-condition diversity for the whole run
   ([brax#174](https://github.com/google/brax/issues/174)) — poison for combat diversity
   and PFSP match variety — and truncation-vs-termination bootstrapping in brax PPO.
   Cheap to check, corrupts everything if wrong.
4. **Restrict HER to the non-adversarial skill layer.** Hindsight relabeling inside
   adversarial episodes is statistically unsound (outcome co-authored by the opponent;
   [arXiv:2105.00499](https://arxiv.org/pdf/2105.00499)); the adversarial analog is
   subgame/state curricula ([arXiv:2310.04796](https://arxiv.org/pdf/2310.04796)). If HER
   currently touches self-play transitions, it is plausibly hurting the critic.
5. **Budget two-robot contacts explicitly in MJX**: enumerate `<pair>`s for robot-robot
   collisions, cap `max_contact_points`/`max_geom_pairs`, pyramidal cone + Newton solver
   (Playground recipe, [arXiv:2502.08844](https://arxiv.org/abs/2502.08844)); note
   `mjx.ray` has no heightfield support
   ([mujoco#2155](https://github.com/google-deepmind/mujoco/issues/2155)) — relevant to
   the simulated lidar if arena terrain becomes an hfield. Plan the MuJoCo-Warp migration
   (all equality/tendon types, big contact speedups) but watch its hfield-NaN issue.
6. **Anneal or scope RND once self-play is live.** Non-stationary opponents make states
   look permanently novel, inflating intrinsic reward; the opponent pool is already an
   auto-curriculum. Restrict novelty to opponent-independent dims (own posture/contacts)
   or decay the coefficient on pool win-rate signal (inference from
   [arXiv:2507.19725](https://arxiv.org/html/2507.19725v1); no combat-specific study).
7. **Make the PBT selection metric PFSP-weighted vs the whole hall-of-fame**, never
   vs-latest, or cycling returns through the population axis; PBT copy steps can collapse
   the diversity the opponent pool depends on — AlphaStar's answer was structural roles
   (add a cheap exploiter: Minimax-Exploiter
   [arXiv:2311.17190](https://arxiv.org/abs/2311.17190), or the learned-attack recipe of
   [arXiv:2405.12424](https://arxiv.org/abs/2405.12424) as a checkpoint-promotion gate —
   which doubles as automated reward-hack discovery).
8. **Actuator program for the real robot**: the actuator-net lineage was born on SEAs
   (Hwangbo 2019) — fit one for the belt yaw axis; the BLDC dyno testbench matches the
   published bench-to-sim workflow ([arXiv:2410.16591](https://arxiv.org/abs/2410.16591))
   almost exactly, including thermal characterization (then feed a thermal-aware
   constraint, [arXiv:2603.01631](https://arxiv.org/abs/2603.01631) — worm stages run
   hot). For the worm pitch drive there is **no RL precedent**: treat it as a quasi-static
   position-commanded DOF (PD position targets are the right action space for it), model
   in MJX as `frictionloss`+`damping`, verify locking in Drake's ScrewJoint, and accept
   that Wensing's Impact-Mitigation-Factor argument
   ([TRO 2017](https://dl.acm.org/doi/abs/10.1109/TRO.2016.2640183)) says impacts must be
   absorbed elsewhere (the SEA and the toggle foot, not the worm).
9. **Model the closed chains natively; don't serialize.** BRUCE
   ([arXiv:2507.00273](https://arxiv.org/abs/2507.00273)) and
   [arXiv:2507.10164](https://arxiv.org/abs/2507.10164) both show native-loop GPU training
   works and beats effective-serial models. MJX now supports the needed equality/tendon
   types. Expect the toggle foot's over-center holding force to be the largest
   MJX-soft-constraint vs Drake-SAP discrepancy — sweep `solref`/`solimp` against Drake as
   ground truth; that cross-engine sweep is also a publishable data point, since nobody
   else uses Drake as a verification oracle.
10. **Adopt the two-stage blueprint for combat**: pretrained motion primitives → strategic
    self-play layer (Tencent, Nature MI 2024,
    [arXiv:2308.15143](https://arxiv.org/abs/2308.15143); same staging in RoboStriker
    [arXiv:2601.22517](https://arxiv.org/abs/2601.22517) and DeepMind soccer
    [arXiv:2304.13653](https://arxiv.org/abs/2304.13653)). Every successful physical
    competitive system separates motor-skill acquisition from adversarial strategy and
    regularizes the fight policy toward the skill prior (KL/AMP anchor) — this is also
    the strongest known structural defense against catapult-class hacks, since the prior
    assigns them zero density. Note: no real-hardware quadruped-vs-quadruped combat RL is
    published — this project sits in an open lane.
11. **Evaluation hygiene**: report IQM over multiple seeds for training changes
    ([arXiv:2304.01315](https://arxiv.org/abs/2304.01315)); for checkpoint promotion use
    cross-play matrices + a frozen-champion best-response audit
    ([arXiv:2004.09677](https://arxiv.org/pdf/2004.09677)); MJX's bit-determinism is a
    real advantage here over PhysX ecosystems
    ([IsaacLab#275](https://github.com/isaac-sim/IsaacLab/issues/275)) — keep it by
    pinning JAX/XLA versions.
12. **Skip foundation/generalist models; consider FastTD3 as a cheap experiment.**
    Cross-embodiment payoffs don't apply to one robot; the distillation recipe (HOVER/
    ExBody2) matters only when merging multiple specialist skills. FastTD3
    ([arXiv:2505.22642](https://arxiv.org/abs/2505.22642)) fits a single A100 and enables
    demo-bootstrapping PPO can't — but audit action-space bounds first
    ([SAC-at-scale](https://araffin.github.io/post/sac-massive-sim/)).
