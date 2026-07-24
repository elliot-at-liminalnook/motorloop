/**
 * @name Host synchronization inside a per-step environment method
 * @description .item()/.cpu()/.numpy()/float(tensor) inside step/observe/
 *              privileged/trajectory_state forces a GPU->host sync every
 *              control step, defeating the captured-graph one-stream
 *              discipline the Warp envs are built around. Batch such reads
 *              at evaluation boundaries instead.
 * @kind problem
 * @problem.severity recommendation
 * @id motorloop/host-sync-in-step-path
 */

import python

predicate hotMethod(Function fn) {
  fn.getName() in ["step", "observe", "privileged", "trajectory_state",
                   "_substep", "_run_physics", "interaction_target"] and
  fn.getLocation().getFile().getAbsolutePath().matches("%warp_env%")
}

from Call c, Attribute f, Function fn, string name
where
  f = c.getFunc() and
  f.getName() = name and
  name in ["item", "numpy"] and
  fn = c.getScope() and
  hotMethod(fn)
select c, "." + name + "() in " + fn.getName() +
  "() forces a host sync every control step."
