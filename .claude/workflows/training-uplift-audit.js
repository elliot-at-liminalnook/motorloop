export const meta = {
  name: 'training-uplift-audit',
  description: 'Audit repo for biggest algorithmic uplifts to from-scratch RL training; produce ranked top-10',
  phases: [
    { title: 'Map', detail: '6 parallel readers over env/locomotion/meta/body/history/bootstrap code' },
    { title: 'Propose', detail: '7 diverse-lens proposers generate candidate uplifts' },
    { title: 'Consolidate', detail: 'dedup + merge into unique candidate list' },
    { title: 'Judge', detail: 'adversarial per-candidate verification vs the actual codebase + completeness critic' },
    { title: 'Synthesize', detail: 'rank into top 10 with implementation order' },
  ],
}

const ROOT = '/home/elliot/Projects/bldc-cosim-testbench'

const PROJECT_CONTEXT = `
PROJECT: quadruped combat robots ("battlebots") simulated in Brax/MJX. Two small quadrupeds (legs FL/FR/RL/RR,
3 hinge joints each: abd/flex/knee, actuators named like A_FL_abd_m, plus front-leg strikers firing +x) fight in
an arena. GOAL: policies that learn to WALK and then FIGHT from scratch via RL, verified by RENDERED behavior
(visible stride >= ~0.3 m/s crossing the frame; genuine approach + strikes), trainable in single-digit hours on
one A100 80GB.

STACK: brax 0.14.1 PPO (functional JAX, jit/vmap — any addition must be jit-safe, no Python control flow on
traced values), mujoco 3.9 MJX, jax 0.6.2. Throughput ~7,300 env-steps/s at 2048 envs with 128+16 lidar rays;
JIT compile costs 50-250 s per train invocation.

ALREADY IMPLEMENTED (do not re-propose as new): asymmetric actor-critic (dict obs 'state'/'value_state'),
lidar with per-env RNG noise/dropout/latency/frame-stack, RND curiosity (proprio + tactical-descriptor feature
modes, predictor genuinely trained in the training path), on-policy HER (hindsight relabeling via a
generate_unroll patch), PBT (exploit/explore, checkpoints, lineage, budget caps, resume), self-play vs frozen
opponent checkpoints, warm-start with obs/action-space growth, behavior benchmark (displacement/path/closing/
approach metrics), anti-exploit gates (damage only scores while closing, stationary-damage penalty, oscillation
penalty, keep-best behavior gates).

FAILURE HISTORY (root causes matter):
1. Stand-still exploit: policies won via incidental contact without moving — fixed by behavior gates.
2. move_weight (instantaneous-speed reward) was farmed by in-place oscillation.
3. From-scratch locomotion in the fighter env: stuck at ~0.18 m displacement after 12M steps.
4. CPG-PD gait prior: produced only ~0.08 m/s undulation-in-place; user abandoned CPG entirely ("dead end").
   Direction is now PURE RL locomotion.
5. Warm-start from a flat-ground locomotor checkpoint did not behaviorally transfer into the fighter.
6. RND on proprioception was farmed by joint jitter; tactical-feature RND reduced that 6x.
Suspect but unconfirmed: the body/actuation may be marginal for walking (user authorized body adjustment if RL
truly cannot walk it); the fighter env drives hinges with direct torque actions; commanded_env.py contains a
Go2-style velocity-command recipe (PD position-target mode, foot air-time reward, actrate/velz/angxy penalties)
that has NOT yet been validated in pure-pd mode on this body.

THE QUESTION: the biggest ALGORITHMIC uplifts (not infra/devops) to make from-scratch training actually work —
walking first, then combat — ranked by expected uplift x probability of success in THIS stack.
`

const READER_SCHEMA = {
  type: 'object', required: ['summary', 'key_facts', 'weaknesses'], additionalProperties: false,
  properties: {
    summary: { type: 'string', description: 'Dense factual summary, <=600 words' },
    key_facts: { type: 'array', items: { type: 'object', required: ['fact'], additionalProperties: false,
      properties: { fact: { type: 'string' }, where: { type: 'string', description: 'file:line' } } } },
    weaknesses: { type: 'array', items: { type: 'object', required: ['issue'], additionalProperties: false,
      properties: { issue: { type: 'string' }, evidence: { type: 'string' } } } },
  },
}

const CANDIDATE_PROPS = {
  name: { type: 'string' },
  change: { type: 'string', description: 'Concretely what to change/add in this codebase' },
  rationale: { type: 'string', description: 'Why this lifts from-scratch training HERE, tied to the failure history or map' },
  expected_impact: { type: 'string', enum: ['high', 'medium', 'low'] },
  effort: { type: 'string', enum: ['S', 'M', 'L'] },
  evidence: { type: 'string', description: 'Paper/system where this is proven (name + venue/year)' },
  risks: { type: 'string' },
}

const PROPOSAL_SCHEMA = {
  type: 'object', required: ['candidates'], additionalProperties: false,
  properties: { candidates: { type: 'array', items: { type: 'object', additionalProperties: false,
    required: ['name', 'change', 'rationale', 'expected_impact', 'effort', 'evidence'],
    properties: CANDIDATE_PROPS } } },
}

const DEDUP_SCHEMA = {
  type: 'object', required: ['candidates'], additionalProperties: false,
  properties: { candidates: { type: 'array', maxItems: 32, items: { type: 'object', additionalProperties: false,
    required: ['id', 'name', 'change', 'rationale', 'expected_impact', 'effort', 'evidence'],
    properties: { id: { type: 'string' }, sources: { type: 'string', description: 'which lenses proposed it' }, ...CANDIDATE_PROPS } } } },
}

const JUDGE_SCHEMA = {
  type: 'object', additionalProperties: false,
  required: ['verdict', 'impact_score', 'feasibility_score', 'confidence', 'reasoning', 'code_check'],
  properties: {
    verdict: { type: 'string', enum: ['include', 'maybe', 'exclude'] },
    impact_score: { type: 'number', description: '1-10 expected uplift on from-scratch training success' },
    feasibility_score: { type: 'number', description: '1-10 implementability in brax 0.14.1 / MJX / jit-safe JAX' },
    confidence: { type: 'number', description: '0-1' },
    reasoning: { type: 'string' },
    code_check: { type: 'string', description: 'What you actually verified in the repo (files opened, facts confirmed/refuted)' },
  },
}

const CRITIC_SCHEMA = {
  type: 'object', required: ['missing'], additionalProperties: false,
  properties: { missing: { type: 'array', maxItems: 8, items: { type: 'object', additionalProperties: false,
    required: ['name', 'change', 'rationale', 'expected_impact', 'effort', 'evidence'],
    properties: CANDIDATE_PROPS } } },
}

const SYNTH_SCHEMA = {
  type: 'object', required: ['top10', 'honorable_mentions', 'rejected', 'recommended_order'], additionalProperties: false,
  properties: {
    top10: { type: 'array', minItems: 10, maxItems: 10, items: { type: 'object', additionalProperties: false,
      required: ['rank', 'name', 'what', 'why', 'expected_uplift', 'effort', 'implementation_sketch'],
      properties: {
        rank: { type: 'number' }, name: { type: 'string' },
        what: { type: 'string', description: 'The change, concretely, in this codebase' },
        why: { type: 'string', description: 'Grounded in the failure history / evidence' },
        expected_uplift: { type: 'string' }, effort: { type: 'string' },
        implementation_sketch: { type: 'string', description: '2-4 sentences: files touched, mechanism, jit-safety notes' },
        dependencies: { type: 'string', description: 'Which other items this composes with or requires' },
      } } },
    honorable_mentions: { type: 'array', items: { type: 'string' } },
    rejected: { type: 'array', items: { type: 'object', additionalProperties: false, required: ['name', 'reason'],
      properties: { name: { type: 'string' }, reason: { type: 'string' } } } },
    recommended_order: { type: 'string', description: 'Suggested implementation/validation sequence and why' },
  },
}

// ---------- Phase 1: Map ----------
phase('Map')
const READERS = [
  { key: 'combat-env', prompt: `Read ${ROOT}/sim/robot/train_adversarial.py (all ~1830 lines, in chunks). Extract with exact values: (1) every reward term with its weight/formula (damage, closing, gates, penalties, RND/HER coefficients); (2) observation construction for actor and critic (dims, what's included, lidar pipeline); (3) ACTION SPACE — are hinge actions direct torques, position targets, or something else? cite lines; (4) PPO hyperparameters passed to brax ppo.train (lr, entropy, num_envs, batch, unroll, minibatches, updates, discount, network sizes, normalize_observations); (5) episode length, termination/fall conditions, reset/spawn distribution; (6) self-play/opponent mechanics; (7) anything that looks algorithmically weak or missing for learning locomotion from scratch (e.g. no gait-shaping terms, no curriculum on X, reward scale issues).` },
  { key: 'locomotion-recipe', prompt: `Read ${ROOT}/sim/robot/commanded_env.py, ${ROOT}/sim/robot/train_commanded.py, ${ROOT}/sim/robot/eval_commanded.py, ${ROOT}/sim/robot/validate_commanded.py and skim ${ROOT}/sim/robot/cpg_teacher.py. Extract with exact values: reward terms + weights (tracking, air-time, actrate, velz, angxy, progress, etc.), PD gains and control modes (pd vs cpg_pd), command sampling ranges, episode length, termination, PPO hyperparameters used by train_commanded, obs contents. Note what the Go2/ANYmal-style recipe here has vs what it's missing compared to proven quadruped RL recipes (e.g. MuJoCo Playground Go1/Go2, walk-these-ways): domain randomization? terrain? gait-phase observations? default-pose regularization? feet-slip penalty? Note anything mistuned or suspicious.` },
  { key: 'meta-stack', prompt: `Read ${ROOT}/sim/robot/pbt_train.py, ${ROOT}/sim/robot/her_goal.py, ${ROOT}/sim/robot/rnd_curiosity.py, and skim ${ROOT}/sim/robot/selfplay_drive.py, ${ROOT}/sim/robot/coevolve.py, ${ROOT}/sim/robot/combat_rank.py, ${ROOT}/sim/robot/fighter_rank.py. Extract: PBT hyperparameter search space + exploit/explore rules, HER goal definition/relabel mechanics + goal reward, RND architecture/feature spaces/update rule, how self-play opponents are selected/frozen/ranked (uniform? latest? league?), any Elo/ranking machinery. Note algorithmic weaknesses: e.g. opponent-sampling naivety, PBT population size limits, HER goals that don't target locomotion, RND scale/normalization issues.` },
  { key: 'body-physics', prompt: `Read ${ROOT}/sim/robot/gen_robot_mjcf.py, ${ROOT}/sim/robot/robot.toml, and skim ${ROOT}/sim/robot/model.xml (it may be large — extract key numbers only), plus ${ROOT}/notes/battlebot-actuator-spec.md if present. Extract with exact values: total mass and per-link masses, leg segment lengths, joint ranges (abd/flex/knee), actuator type (motor/position/velocity), gear ratios, forcerange/ctrlrange (compute torque-to-weight: max joint torque vs mass*g*leg_length), timestep, frame_skip/control dt, friction, contact solref/solimp, foot geometry. Assess honestly: is this body physically capable of a >=0.3 m/s walking gait? What limits it (torque, joint range, foot size, mass distribution, control rate)? Note any physics params known to hamper MJX RL (soft contacts, high damping, unrealistic armature).` },
  { key: 'failure-history', prompt: `Read ${ROOT}/notes/advanced-rl-implementation.md, ${ROOT}/notes/locomotion-bootstrap-teacher-checklist.md, ${ROOT}/notes/rl-combat-dodge-report.md, ${ROOT}/notes/codesign-fighter-report.md, ${ROOT}/notes/sparc-learning-log.md (skip any that don't exist). Also check ${ROOT}/gpu_artifacts/cpglong_benchmark.jsonl (first + last few lines). Extract: every training attempt documented, its configuration, its outcome, and the diagnosed root cause of failure. Especially: why from-scratch locomotion never produced a gait, what the CPG attempts showed, what warm-start transfer showed, what the judge/benchmark score trajectories looked like (improvement then collapse?), any documented hypotheses that were never tested.` },
  { key: 'bootstrap-ecosystem', prompt: `Skim (read headers, main functions, reward/loss definitions — not every line) these files in ${ROOT}/sim/robot/: train_bootstrap_bc.py, train_residual_locomotion.py, collect_dagger_dataset.py, collect_gait_dataset.py, collect_transition_dataset.py, curriculum_drive.py, search_policy_bias.py, search_policy_router.py, skill_bank.py, return_skill_env.py, train_return_skill.py, anti_cheat.py, adaptive_policy.py. Extract: what teacher/dataset/imitation/curriculum machinery already exists, what state it's in (working? abandoned?), what it produces (datasets? checkpoints?), and whether any of it could be repurposed for bootstrapping from-scratch locomotion (BC pretrain, DAgger, residual learning, skill banks). Note which approaches were clearly abandoned and why if discernible.` },
]
log('Map: 6 readers over env, locomotion recipe, meta-stack, body physics, failure history, bootstrap ecosystem')
const maps = await parallel(READERS.map(r => () => agent(
  `You are auditing part of an RL codebase. Context:\n${PROJECT_CONTEXT}\n\nYOUR ASSIGNMENT:\n${r.prompt}\n\nBe precise: exact numbers, exact file:line citations. Your output feeds improvement proposals — facts only, no recommendations yet. Summary <=600 words.`,
  { label: `read:${r.key}`, phase: 'Map', schema: READER_SCHEMA, effort: 'medium' }
)))
const mapText = READERS.map((r, i) => maps[i] ? `### MAP[${r.key}]\n${JSON.stringify(maps[i])}` : `### MAP[${r.key}]: unavailable`).join('\n\n')
log(`Map complete: ${maps.filter(Boolean).length}/6 readers returned`)

// ---------- Phase 2: Propose ----------
phase('Propose')
const LENSES = [
  { key: 'locomotion-sota', brief: `You are an expert in modern legged-locomotion RL (2022-2025): MuJoCo Playground Go1/Go2 recipes, walk-these-ways, DreamWaQ, rapid motor adaptation, massively-parallel sim (Rudin et al.), gait-phase/periodicity rewards, foot air-time & slip terms, default-pose regularization, symmetry exploitation, terrain/command curricula, domain randomization. Propose what would make THIS body learn a real stride from scratch.` },
  { key: 'ppo-internals', brief: `You are an expert in PPO optimization internals and common silent killers: truncation-vs-termination value bootstrapping bugs, observation/advantage normalization, reward scaling/clipping, entropy & LR schedules, network width/depth (locomotion likes wider), layer norm, action distribution choice (tanh-squashed vs clipped Gaussian), KL-adaptive clipping, value-loss clipping, memory (GRU/LSTM) vs frame-stack, initialization scale of the policy head. Check what brax 0.14.1 PPO actually does (you may inspect the installed brax source under site-packages if available, or reason from its known API in train_adversarial.py's ppo.train call) and propose the highest-leverage fixes/settings for this task.` },
  { key: 'reward-design', brief: `You are an expert in reward design and anti-reward-hacking for RL. This project already got burned three times (stand-still, oscillation farming, RND jitter farming). Propose principled reward structures: potential-based shaping (provably exploit-free), constrained RL / Lagrangian instead of hand-tuned penalty weights, phase-based task decomposition (walk-then-fight), success-grounded sparse+shaped mixes, reward-term curricula (anneal shaping away), automatic penalty balancing. Ground each in what the map shows about the current reward.` },
  { key: 'selfplay-league', brief: `You are an expert in self-play and population game theory: AlphaStar league (main agents + exploiters + past checkpoints), PSRO, fictitious self-play, opponent-sampling distributions (prioritized by win-rate), ELO/TrueSkill gating of checkpoint promotion, curriculum opponents (scripted -> frozen -> live). The current stack fights frozen checkpoints and a passive B. Propose what most improves from-scratch COMBAT emergence once locomotion exists, and what prevents the judge-score collapse seen at 20M+ steps.` },
  { key: 'exploration-curriculum', brief: `You are an expert in exploration and automatic curricula: ALP-GMM, prioritized level replay over spawn/command configs, goal curricula (start close, grow distance), reverse curriculum from contact, reference-state initialization (start episodes from diverse mid-gait states — RSI from DeepMimic), early-termination shaping, adaptive command ranges, success-gated task progression. The core failure here is a hard exploration problem: no gait gradient from standing. Propose the mechanisms that create that gradient.` },
  { key: 'imitation-modelbased', brief: `You are an expert in imitation and model-based bootstrapping: AMP (adversarial motion priors) from even crude reference motion, trajectory optimization (e.g. direct collocation on the MJX model) to synthesize ONE feasible gait cycle as a reference, DeepMimic-style motion tracking then task transfer, BC/DAgger warm-start (repo already has dagger/bc machinery — check the map), residual policy learning on a scripted base, teacher-student distillation. The CPG prior failed; propose bootstraps that are NOT hand-designed oscillators. Note: no motion-capture data exists for this custom body — proposals must synthesize or not need references.` },
  { key: 'embodiment-actionspace', brief: `You are an expert in action-space and embodiment choices for legged RL: position-target PD control vs direct torque (the single best-documented uplift for quadruped RL — check what the fighter env uses per the map), action filtering/low-pass, control frequency choice, default-pose-relative actions, action scaling, symmetry-constrained policies (mirror the two body sides), morphology adjustments within the user's authorization (leg length, torque, foot friction/size, mass) IF the body-physics map shows marginal capability, curriculum on assistive forces (bootstrap harness that fades). Propose the biggest wins here.` },
]
log('Propose: 7 lens agents generating candidates')
const proposals = await parallel(LENSES.map(l => () => agent(
  `${PROJECT_CONTEXT}\n\nCODEBASE MAP (from 6 parallel auditors — treat as ground truth unless you re-verify):\n${mapText}\n\nLENS: ${l.brief}\n\nPropose 6-8 concrete ALGORITHMIC improvements for from-scratch training in THIS codebase. Rules: (a) must be implementable in brax 0.14.1 PPO / MJX / jit-safe JAX; (b) must not already be implemented (check the map and ALREADY IMPLEMENTED list); (c) each needs real published evidence (paper/system, venue/year); (d) tie rationale to this project's specific failure history, not generic benefits; (e) prefer few big wins over many small tweaks. You may open repo files under ${ROOT} to verify details before proposing.`,
  { label: `propose:${l.key}`, phase: 'Propose', schema: PROPOSAL_SCHEMA }
)))
const allCandidates = []
LENSES.forEach((l, i) => {
  if (proposals[i]) proposals[i].candidates.forEach(c => allCandidates.push({ ...c, lens: l.key }))
})
log(`Propose complete: ${allCandidates.length} raw candidates from ${proposals.filter(Boolean).length}/7 lenses`)

// ---------- Phase 3: Consolidate ----------
phase('Consolidate')
const merged = await agent(
  `${PROJECT_CONTEXT}\n\nBelow are ${allCandidates.length} raw improvement candidates from 7 expert lenses. Consolidate them: merge semantic duplicates (keep the strongest/most specific formulation, note all source lenses), drop anything that the ALREADY IMPLEMENTED list covers, drop pure-infra items. Assign each a short id (C01, C02, ...). Output at most 32 unique candidates, keeping ALL distinct ideas — do not drop a unique idea just to shorten the list.\n\nRAW CANDIDATES:\n${JSON.stringify(allCandidates)}`,
  { label: 'consolidate', phase: 'Consolidate', schema: DEDUP_SCHEMA, effort: 'medium' }
)
log(`Consolidated to ${merged.candidates.length} unique candidates`)

// ---------- Phase 4: Judge (+ completeness critic in parallel) ----------
phase('Judge')
const KEY_FILES = `${ROOT}/sim/robot/train_adversarial.py (fighter env+training), ${ROOT}/sim/robot/commanded_env.py + train_commanded.py (locomotion recipe), ${ROOT}/sim/robot/gen_robot_mjcf.py + robot.toml (body), ${ROOT}/sim/robot/pbt_train.py, her_goal.py, rnd_curiosity.py (meta-stack), ${ROOT}/notes/advanced-rl-implementation.md (history)`
const judgeOne = (c) => agent(
  `${PROJECT_CONTEXT}\n\nAdversarially assess ONE proposed training improvement. Your default posture is skepticism — try to find reasons it will NOT deliver here, then weigh honestly.\n\nCANDIDATE ${c.id}: ${JSON.stringify(c)}\n\nKey files if you need to verify claims: ${KEY_FILES}. You MUST open at least the file(s) most relevant to this candidate and confirm or refute its factual premises (e.g. 'the env uses direct torque' — check; 'no domain randomization exists' — check). Score: impact_score = expected uplift on from-scratch walk-then-fight success (10 = likely unblocks the core failure, 1 = cosmetic); feasibility_score = implementability in this jit/vmap brax-0.14.1 stack including effort (10 = drop-in flag, 1 = research project); confidence 0-1. Verdict: include (belongs in a top-10), maybe, exclude. Beware: (a) already implemented; (b) structurally similar to something that already failed here — explain why it would differ; (c) requires data/assets that don't exist; (d) breaks jit-safety.`,
  { label: `judge:${c.id}`, phase: 'Judge', schema: JUDGE_SCHEMA }
).then(v => v ? { ...c, judge: v } : null)

const [judgedMainRaw, crit] = await parallel([
  () => parallel(merged.candidates.map(c => () => judgeOne(c))),
  () => agent(
    `${PROJECT_CONTEXT}\n\nCODEBASE MAP:\n${mapText}\n\nCANDIDATE LIST so far:\n${JSON.stringify(merged.candidates.map(c => ({ id: c.id, name: c.name, change: c.change })))}\n\nYou are a completeness critic. What major, well-evidenced algorithmic uplift for from-scratch quadruped locomotion+combat RL is MISSING from this list? Think systematically across: value-function/bootstrapping correctness, action-space representation, symmetry, curricula, reference-free gait shaping, opponent modeling, network memory/architecture, reward-term scheduling, physics/domain randomization, episode design (reset distributions, early termination), and anything you know from proven quadruped RL systems that no candidate covers. Return only genuinely missing, high-value items (0-8). Empty list is a valid answer.`,
    { label: 'completeness-critic', phase: 'Judge', schema: CRITIC_SCHEMA }
  ),
])
let judged = (judgedMainRaw || []).filter(Boolean)
log(`Judged ${judged.length}/${merged.candidates.length}; critic found ${crit ? crit.missing.length : 0} missing candidates`)

if (crit && crit.missing.length > 0) {
  const extras = crit.missing.map((m, i) => ({ ...m, id: `X${i + 1}`, lens: 'completeness-critic', sources: 'completeness-critic' }))
  const judgedExtras = await parallel(extras.map(c => () => judgeOne(c)))
  judged = judged.concat(judgedExtras.filter(Boolean))
  log(`Judged ${judgedExtras.filter(Boolean).length} critic additions; total pool ${judged.length}`)
}

// ---------- Phase 5: Synthesize ----------
phase('Synthesize')
const finalReport = await agent(
  `${PROJECT_CONTEXT}\n\nBelow is the full pool of adversarially judged improvement candidates (each has judge scores: impact 1-10, feasibility 1-10, confidence 0-1, verdict, and code_check notes on what was verified in the repo).\n\nJUDGED POOL:\n${JSON.stringify(judged)}\n\nProduce the definitive TOP 10 algorithmic improvements, ranked by expected uplift x probability of success in this stack. Rules: (1) prioritize items that attack the ROOT failure — no gait gradient ever emerges from scratch — over marginal tuning; (2) respect judge verdicts but you may overrule with explicit justification; (3) merge candidates that only make sense together into one ranked item, naming both; (4) for each item give a concrete implementation sketch naming actual files (train_adversarial.py, commanded_env.py, gen_robot_mjcf.py, ...) and jit-safety notes; (5) note dependencies/ordering between items; (6) list honorable mentions (real but sub-top-10) and rejected items with one-line honest reasons; (7) recommended_order: the sequence you'd implement + validate them in, with the cheapest decisive experiment first. Be honest about uncertainty — this user has been burned by over-claiming.`,
  { label: 'synthesize-top10', phase: 'Synthesize', schema: SYNTH_SCHEMA, effort: 'high' }
)
log('Synthesis complete')

return {
  top10: finalReport ? finalReport.top10 : null,
  honorable_mentions: finalReport ? finalReport.honorable_mentions : [],
  rejected: finalReport ? finalReport.rejected : [],
  recommended_order: finalReport ? finalReport.recommended_order : '',
  judged_pool: judged.map(c => ({ id: c.id, name: c.name, lens: c.lens, impact: c.judge.impact_score, feasibility: c.judge.feasibility_score, verdict: c.judge.verdict, reasoning: c.judge.reasoning })),
  map_weaknesses: READERS.map((r, i) => maps[i] ? { area: r.key, weaknesses: maps[i].weaknesses } : null).filter(Boolean),
}