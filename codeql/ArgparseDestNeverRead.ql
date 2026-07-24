/**
 * @name argparse option whose destination is never read
 * @description An add_argument("--flag") whose derived dest attribute is never
 *              loaded anywhere (neither `args.flag` nor getattr(args, "flag"))
 *              is dead CLI surface: callers can set it, nothing changes. This
 *              is how launchers accumulate knobs that lie. Reads via
 *              vars(args) dumps (config echo) are deliberately not counted as
 *              consumption.
 * @kind problem
 * @problem.severity warning
 * @id motorloop/argparse-dest-never-read
 */

import python

string destOf(Call c) {
  exists(Attribute f, StringLiteral flag |
    f = c.getFunc() and
    f.getName() = "add_argument" and
    flag = c.getArg(0) and
    flag.getText().matches("--%") and
    result = flag.getText().suffix(2).replaceAll("-", "_")
  ) and
  // an explicit dest= overrides the derived name; skip those calls
  not exists(Keyword k | k = c.getANamedArg() and k.getArg() = "dest")
}

predicate isRead(string name) {
  exists(Attribute a | a.getName() = name and a.getCtx() instanceof Load)
  or
  exists(Call g, Name f, StringLiteral s |
    g.getFunc() = f and f.getId() = "getattr" and
    g.getArg(1) = s and s.getText() = name)
}

from Call c, string dest
where
  dest = destOf(c) and
  not isRead(dest)
select c, "Option --" + dest.replaceAll("_", "-") + " is parsed but its value is never read."
