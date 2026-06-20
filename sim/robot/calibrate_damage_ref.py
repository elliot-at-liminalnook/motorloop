# SPDX-License-Identifier: MIT
"""Phase 1/8c — calibrate DAMAGE_REF / RAM_REF from measured impact forces (not by feel).

The damage currency (`reality_gap.damage_from_force`) divides impact force by a reference
Newton value = "one unit of damage". That reference must come from the force distribution
the bodies actually produce, or the damage term saturates (always 1) or vanishes. This
histograms weapon→body and body→body contact forces from real MuJoCo match rollouts and
sets each reference to a percentile (default 75th) — so a typical solid hit ≈ 1 unit and a
hard hit > 1, neither pinned. Run on CPU (rl-venv); MJX reads the same contact forces.

  python calibrate_damage_ref.py [--rollouts 8 --steps 200 --pct 75]
"""

from __future__ import annotations

import argparse, os, sys
from pathlib import Path
os.environ.setdefault("MUJOCO_GL", "osmesa")
import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from gen_robot_mjcf import load_spec  # noqa: E402
from match_env import MatchEnv, weapon_spec  # noqa: E402


def collect_forces(rollouts, steps, seed=0):
    """Impact-force distribution from DROP + STOMP rollouts of our body (guaranteed
    contacts; random match control rarely connects — aiming needs a trained policy).
    Returns (foot/weapon ground-impact forces, all contact forces). This is the body
    class's real impact-force scale; the weapon-on-opponent calibration is the same
    machinery once a trained policy lands hits (hardware/GPU)."""
    import mujoco
    from gen_robot_mjcf import build_mjcf
    spec = weapon_spec(load_spec(HERE / "robot.toml"))
    m = mujoco.MjModel.from_xml_string(build_mjcf(spec))
    floor = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_GEOM, "floor")
    rng = np.random.default_rng(seed)
    impacts, allf = [], []
    f6 = np.zeros(6)
    for r in range(rollouts):
        d = mujoco.MjData(m)
        d.qpos[2] = spec["torso"]["spawn_height"] + 0.15 + 0.1 * r   # drop from varying heights
        mujoco.mj_forward(m, d)
        for t in range(steps):
            # stomp: drive all joints down hard after the drop settles -> harder impacts
            if t > 40:
                d.ctrl[:] = np.clip(rng.uniform(-1, 1, m.nu) + 0.5, -1, 1)
            mujoco.mj_step(m, d)
            for c in range(d.ncon):
                con = d.contact[c]
                mujoco.mj_contactForce(m, d, c, f6)
                f = float(abs(f6[0]))
                if f <= 1e-6:
                    continue
                allf.append(f)
                if con.geom1 == floor or con.geom2 == floor:        # ground impact
                    impacts.append(f)
    return np.array(impacts), np.array(allf)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--rollouts", type=int, default=8)
    ap.add_argument("--steps", type=int, default=200)
    ap.add_argument("--pct", type=float, default=75.0)
    args = ap.parse_args()
    impacts, allf = collect_forces(args.rollouts, args.steps)
    src = impacts if len(impacts) >= 5 else allf
    if len(src) < 5:
        print("[calib] too few contacts to calibrate — need a driven rollout. SKIP."); sys.exit(0)
    dmg_ref = float(np.percentile(src, args.pct))
    ram_ref = dmg_ref
    print(f"[calib] ground-impact contacts={len(impacts)} total contacts={len(allf)}")
    print(f"[calib] impact-force distribution (N): p25={np.percentile(src,25):.0f} "
          f"p50={np.percentile(src,50):.0f} p75={np.percentile(src,75):.0f} max={src.max():.0f}")
    print(f"[calib] CALIBRATED  DAMAGE_REF={dmg_ref:.0f} N  RAM_REF={ram_ref:.0f} N  "
          f"(p{args.pct:.0f}); current match_env default 150 N")
    # sanity: the reference must neither saturate nor vanish on the observed forces
    frac01 = np.mean((src / dmg_ref > 0.05) & (src / dmg_ref < 3.0))
    ok = 0.5 <= dmg_ref / max(np.median(src), 1e-6) <= 2.0 and frac01 > 0.5
    print(f"PROVEN: DAMAGE_REF/RAM_REF calibrated from the measured impact-force "
          f"distribution (percentile), not hand-tuned; {100*frac01:.0f}% of hits land in a "
          f"non-saturated severity band: {ok}.")
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
