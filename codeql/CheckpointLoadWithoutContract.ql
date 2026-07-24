/**
 * @name Checkpoint load without a contract check
 * @description A torch.load of a training checkpoint whose enclosing function
 *              neither reads ck["contract"] nor calls
 *              require_compatible_checkpoint validates shapes at best. Shape
 *              equality is not identity: the v1 task-ID and v2 command
 *              observation contracts share every tensor shape by design, so a
 *              contract-blind load silently reinterprets conditioning
 *              channels (the --init-policy/anchor/opponent bypass).
 * @kind problem
 * @problem.severity warning
 * @id motorloop/checkpoint-load-without-contract
 */

import python

predicate checksContract(Function fn) {
  exists(Subscript s, StringLiteral key |
    s.getScope() = fn and key = s.getIndex() and key.getText() = "contract")
  or
  // ck.get("contract") / artifact.get("observation_semantics")
  exists(Call get, Attribute f, StringLiteral key |
    get.getScope() = fn and f = get.getFunc() and f.getName() = "get" and
    key = get.getArg(0) and
    key.getText() in ["contract", "observation_semantics"])
  or
  exists(Call guard, Name callee |
    guard.getScope() = fn and callee = guard.getFunc() and
    callee.getId() = "require_compatible_checkpoint")
  or
  // loaders that delegate: passing expected_semantics/expected_contract on
  exists(Keyword k |
    k.getScope() = fn and
    k.getArg() in ["expected_semantics", "expected_contract"])
}

from Call c, Attribute f, Function fn
where
  f = c.getFunc() and
  f.getName() = "load" and
  exists(Name torch | torch = f.getObject() and torch.getId() = "torch") and
  fn = c.getScope() and
  c.getLocation().getFile().getAbsolutePath().matches("%/sim/robot/%") and
  not c.getLocation().getFile().getAbsolutePath().matches("%/test_%") and
  not checksContract(fn)
select c, "torch.load in " + fn.getName() +
  "() never validates the checkpoint contract; shapes are not identity."
