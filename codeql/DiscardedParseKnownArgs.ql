/**
 * @name Discarded parse_known_args remainder
 * @description A call to argparse's parse_known_args whose remainder tuple
 *              element is bound to `_` silently swallows unknown flags: a
 *              launcher can "set" knobs that never reach the program. The
 *              PBT population trainer explored a dead search space this way
 *              for months. Either forward the remainder to a strict parser
 *              or use parse_args.
 * @kind problem
 * @problem.severity warning
 * @id motorloop/discarded-parse-known-args
 */

import python

from Assign asgn, Call call, Attribute func, Tuple lhs, Name second
where
  call = asgn.getValue() and
  func = call.getFunc() and
  func.getName() = "parse_known_args" and
  lhs = asgn.getATarget() and
  second = lhs.getElt(1) and
  second.getId() = "_"
select call,
  "parse_known_args remainder is discarded: unknown flags are silently swallowed."
