/**
 * @name Non-recursive glob in a fingerprint/provenance function
 * @description A provenance or hashing function using .glob("*.py") silently
 *              excludes nested packages, so materially different source trees
 *              can share identical provenance (the warplayer/ blind spot).
 *              Use rglob, or justify the boundary in the docstring.
 * @kind problem
 * @problem.severity warning
 * @id motorloop/non-recursive-source-glob
 */

import python

from Call c, Attribute f, Function fn, StringLiteral pattern
where
  f = c.getFunc() and
  f.getName() = "glob" and
  pattern = c.getArg(0) and
  pattern.getText().matches("*.%") and
  fn = c.getScope() and
  (fn.getName().matches("%hash%") or fn.getName().matches("%fingerprint%") or
   fn.getName().matches("%provenance%"))
select c, "Non-recursive glob in " + fn.getName() +
  "(): nested sources will not move this fingerprint."
