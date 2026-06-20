# SPDX-License-Identifier: MIT
"""R3 — validate the unified damage/contact currency: impact FORCE in Newtons.

The checklist's R3 resolves the CPU (150 N) vs MJX (0.05 m penetration) damage mismatch
onto ONE currency: real contact force read from the solver -> `reality_gap.damage_from_force`.
This test validates the ORDERING that currency must preserve on the CPU MuJoCo model
(MJX reads the same `contact` force arrays):
  1. harder hit  > glancing      (drop fast vs slow -> bigger impact N -> more severity)
  2. stable shove > bounce-off   (low restitution sustains contact force; high bounces away)
  3. repeated impacts ACCUMULATE (severity sums over a multi-bounce episode)

Runs on CPU MuJoCo (rl-venv). Skips (not fails) if mujoco absent.
  python test_contact.py
"""

from __future__ import annotations

import os, sys
from pathlib import Path
os.environ.setdefault("MUJOCO_GL", "osmesa")
import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from reality_gap import default_uncertainty, sample_domain_params, damage_from_force  # noqa: E402

DP = sample_domain_params(0, default_uncertainty())


def _scene(restitution=0.0):
    return f"""<mujoco>
  <option timestep="0.001" integrator="implicitfast"/>
  <worldbody>
    <geom name="floor" type="plane" size="0 0 0.1" solref="0.01 1" solimp="0.95 0.99 0.001"/>
    <body name="ball" pos="0 0 0.5">
      <freejoint/>
      <geom name="ball" type="sphere" size="0.05" mass="1.0" solref="0.01 1"
            solimp="0.95 0.99 0.001"/>
    </body>
  </worldbody>
</mujoco>""".replace("solref=\"0.01 1\"",
                     f'solref="0.01 {1.0 - restitution}"')   # lower damping ratio -> bouncier


def _drop(mujoco, drop_v=0.0, restitution=0.0, steps=1200):
    """Drop the ball; return (peak impact N, total severity over the episode, n_impacts)."""
    m = mujoco.MjModel.from_xml_string(_scene(restitution))
    d = mujoco.MjData(m)
    d.qpos[2] = 0.30; d.qvel[2] = -drop_v          # start height + downward kick
    mujoco.mj_forward(m, d)
    peak = 0.0; total_sev = 0.0; in_contact = False; n_imp = 0
    f = np.zeros(6)
    for _ in range(steps):
        mujoco.mj_step(m, d)
        step_force = 0.0
        for c in range(d.ncon):
            mujoco.mj_contactForce(m, d, c, f)
            step_force += abs(f[0])
        if step_force > 1e-6:
            peak = max(peak, step_force)
            total_sev += damage_from_force(step_force, DP)
            if not in_contact:
                n_imp += 1; in_contact = True
        else:
            in_contact = False
    return peak, total_sev, n_imp


def main():
    try:
        import mujoco
    except Exception:
        print("SKIP: mujoco absent — R3 contact validation runs in the rl-venv "
              "($HOME/rl-venv/bin/python test_contact.py).")
        sys.exit(0)

    # 1. harder hit > glancing
    soft_peak, _, _ = _drop(mujoco, drop_v=0.5)
    hard_peak, _, _ = _drop(mujoco, drop_v=5.0)
    harder = hard_peak > soft_peak
    print(f"[R3] impact force  glancing(0.5 m/s)={soft_peak:.0f} N  hard(5 m/s)={hard_peak:.0f} N  "
          f"-> harder hits score more: {harder}")
    print(f"[R3]   severity     glancing={damage_from_force(soft_peak,DP):.2f}  "
          f"hard={damage_from_force(hard_peak,DP):.2f} (unified Newton currency)")

    # 2. stable shove > bounce-off (sustained contact force-time vs a quick bounce)
    _, stable_sev, stable_imp = _drop(mujoco, drop_v=2.0, restitution=0.0)
    _, bounce_sev, bounce_imp = _drop(mujoco, drop_v=2.0, restitution=0.9)
    stable_wins = stable_sev > bounce_sev
    print(f"[R3] sustained severity  stable(e=0)={stable_sev:.1f} (impacts {stable_imp})  "
          f"bouncy(e=0.9)={bounce_sev:.1f} (impacts {bounce_imp}) -> stable shove > bounce: {stable_wins}")

    # 3. repeated impacts accumulate (a bouncy drop registers multiple impacts, summing severity)
    _, multi_sev, multi_imp = _drop(mujoco, drop_v=3.0, restitution=0.85, steps=3000)
    accumulate = multi_imp >= 2 and multi_sev > damage_from_force(hard_peak, DP)
    print(f"[R3] multi-bounce episode: {multi_imp} impacts, accumulated severity {multi_sev:.1f} "
          f"-> repeated impacts accumulate: {accumulate}")

    ok = harder and stable_wins and accumulate
    print(f"\nPROVEN: R3 unified Newton damage currency preserves the orderings (harder>glancing, "
          f"stable>bounce, impacts accumulate) on the real contact solver: {ok}. MJX reads the "
          f"same contact force arrays -> one damage model CPU and GPU.")
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
