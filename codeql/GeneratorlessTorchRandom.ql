/**
 * @name torch random call without an explicit generator in seeded code
 * @description Environments and trainers promise seed-exact determinism
 *              (checkpoint replay, promotion gates, the GPU canary). A
 *              torch.rand/randn/randint/randperm/multinomial call without
 *              generator= draws from global RNG state, which resumes and
 *              replays do not control per-world. Flags live env/training
 *              code only; tests and offline tools are exempt.
 * @kind problem
 * @problem.severity warning
 * @id motorloop/generatorless-torch-random
 */

import python

predicate inScope(File f) {
  // Environments must draw from their per-world env._gen: global-RNG draws
  // there break replay because env state save/restore does not cover them.
  // The trainer (train_mesh_warp) is deliberately OUT of scope — its global
  // torch RNG is seeded and checkpointed in runtime state, so global draws
  // are part of its determinism contract, not a violation of it.
  f.getAbsolutePath().matches("%/sim/robot/%") and
  not f.getAbsolutePath().matches("%/test_%") and
  not f.getAbsolutePath().matches("%/tests/%") and
  (f.getAbsolutePath().matches("%warp_env%") or
   f.getAbsolutePath().matches("%warp_eval%") or
   f.getAbsolutePath().matches("%predictive_control%"))
}

from Call c, Attribute f, string name
where
  f = c.getFunc() and
  f.getName() = name and
  name in ["rand", "randn", "randint", "randperm", "multinomial"] and
  exists(Name torch | torch = f.getObject() and torch.getId() = "torch") and
  not exists(Keyword k | k = c.getANamedArg() and k.getArg() = "generator") and
  inScope(c.getLocation().getFile())
select c,
  "torch." + name + " without generator=: uncontrolled RNG in seed-exact code."
