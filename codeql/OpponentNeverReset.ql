/**
 * @name Stored opponent policy called but never state-reset
 * @description A file that calls self._opponent(...) without ever resetting
 *              its recurrent state lets a GRU opponent carry hidden memory
 *              across autoreset episode boundaries — the player's state is
 *              masked on done, the opponent's silently is not.
 * @kind problem
 * @problem.severity warning
 * @id motorloop/opponent-never-reset
 */

import python

predicate callsOpponent(File file, Call c) {
  exists(Attribute f |
    f = c.getFunc() and f.getName() = "_opponent" and
    c.getLocation().getFile() = file)
}

predicate resetsOpponent(File file) {
  // either direct: self._opponent.reset(...), or via a dedicated helper
  exists(Attribute reset, Attribute opp |
    reset.getName() = "reset" and opp = reset.getObject() and
    opp.(Attribute).getName() = "_opponent" and
    reset.getLocation().getFile() = file)
  or
  exists(Call helper, Attribute f |
    f = helper.getFunc() and f.getName() = "_reset_opponent_state" and
    helper.getLocation().getFile() = file)
}

from Call c, File file
where
  callsOpponent(file, c) and
  file = c.getLocation().getFile() and
  not resetsOpponent(file)
select c, "self._opponent is called here but its recurrent state is never reset in this file."
