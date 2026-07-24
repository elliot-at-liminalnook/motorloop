/**
 * @name torch.load without weights_only=True
 * @description torch.load uses unpickling: weights_only=False (or omitted on
 *              older defaults) executes arbitrary code from the checkpoint
 *              file. Every core checkpoint format in this repo loads under
 *              weights_only=True; new load sites must not regress to the
 *              unsafe mode.
 * @kind problem
 * @problem.severity warning
 * @id motorloop/torch-load-unsafe-weights-only
 */

import python

from Call c, Attribute f
where
  f = c.getFunc() and
  f.getName() = "load" and
  exists(Name torch | torch = f.getObject() and torch.getId() = "torch") and
  (
    not exists(Keyword k | k = c.getANamedArg() and k.getArg() = "weights_only")
    or
    exists(Keyword k, NameConstant v |
      k = c.getANamedArg() and k.getArg() = "weights_only" and
      v = k.getValue() and v.toString() = "False")
  )
select c, "torch.load without weights_only=True unpickles arbitrary code."
