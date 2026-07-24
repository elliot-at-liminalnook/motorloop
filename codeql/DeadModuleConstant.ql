/**
 * @name Module constant never loaded anywhere
 * @description An ALL_CAPS module-level constant with no load anywhere in the
 *              codebase — not by name in its own module and not as a module
 *              attribute from outside. The sprint/hop reward graveyard grew
 *              this way: 27 constants only their own definitions referenced.
 * @kind problem
 * @problem.severity recommendation
 * @id motorloop/dead-module-constant
 */

import python

from AssignStmt asgn, Name target, string name
where
  asgn.getScope() instanceof Module and
  target = asgn.getATarget() and
  name = target.getId() and
  name.regexpMatch("[A-Z][A-Z0-9_]{2,}") and
  // no load by bare name anywhere (covers the defining module and star users)
  not exists(Name load |
    load.getId() = name and load.getCtx() instanceof Load) and
  // no load as a module/object attribute (SPEC.NAME, module.NAME) anywhere
  not exists(Attribute attr |
    attr.getName() = name and attr.getCtx() instanceof Load) and
  // no consumption-by-string (getattr / monkeypatch.setattr in tests)
  not exists(StringLiteral s | s.getText() = name)
select asgn, "Constant " + name + " is assigned but never loaded anywhere."
