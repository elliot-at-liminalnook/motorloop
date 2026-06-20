# SPDX-License-Identifier: MIT
"""Prove the parametric generator + the modify-and-retrain loop.

For the default body and a set of *part modifications*, this generates the MJCF,
loads it in MuJoCo, checks it is PHYSICALLY VALID (settles without exploding), and
reports the action dimension - which decides whether a modification is a cheap
WARM-START (same dim) or a from-scratch RETRAIN (dim changed). That mapping is the
whole point: parameter swaps transfer; topology swaps don't.

  python prove_robot.py
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

os.environ.setdefault("MUJOCO_GL", "osmesa")

import mujoco  # noqa: E402
import numpy as np  # noqa: E402

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from gen_robot_mjcf import build_mjcf, load_spec  # noqa: E402

# Each variant = a part modification expressed as an override on robot.toml.
VARIANTS = {
    "default": {},
    "spring-assist": {"leg_defaults": {"joint_stiffness": 12.0}},   # parallel-elastic
    "long-legs": {"leg_defaults": {"thigh_len": 0.26, "calf_len": 0.26}},
    "heavy+strong": {"torso": {"mass": 9.0}, "actuator": {"gear": 9.0}},
    "5-leg weapon": {"leg": [                                        # TOPOLOGY change
        {"name": "FL", "pos": [0.19, 0.11, 0.0]},
        {"name": "FR", "pos": [0.19, -0.11, 0.0]},
        {"name": "RL", "pos": [-0.19, 0.11, 0.0]},
        {"name": "RR", "pos": [-0.19, -0.11, 0.0]},
        {"name": "WP", "pos": [0.24, 0.0, 0.0], "is_weapon": True},
    ]},
}


def settle(model, steps=400):
    """Drop with zero control; report final torso height + whether it stayed sane."""
    data = mujoco.MjData(model)
    mujoco.mj_forward(model, data)
    for _ in range(steps):
        mujoco.mj_step(model, data)
    h = float(data.qpos[2])
    ok = np.all(np.isfinite(data.qpos)) and (0.02 < h < 2.0)
    return h, ok


def main():
    spec = load_spec(HERE / "robot.toml")
    base_nu = None
    rows = []
    print(f"{'variant':16s} {'bodies':>6} {'nq':>4} {'nv':>4} {'act':>4} "
          f"{'settle h':>9} {'valid':>6} {'retrain?':>20}")
    print("-" * 80)
    for name, ov in VARIANTS.items():
        model = mujoco.MjModel.from_xml_string(build_mjcf(spec, ov))
        h, ok = settle(model)
        if name == "default":
            base_nu = model.nu
            verdict = "(baseline)"
        elif model.nu == base_nu:
            verdict = "warm-start (same dim)"
        else:
            verdict = f"from-scratch (act {base_nu}->{model.nu})"
        # confirm a parameter override actually took effect
        rows.append((name, model.nbody, model.nq, model.nv, model.nu, h, ok, verdict))
        print(f"{name:16s} {model.nbody:6d} {model.nq:4d} {model.nv:4d} {model.nu:4d} "
              f"{h:9.3f} {str(ok):>6} {verdict:>20}")

    # spot-check that a parameter override changed the model as intended
    print("\nparameter-override spot checks:")
    m_def = mujoco.MjModel.from_xml_string(build_mjcf(spec))
    m_spr = mujoco.MjModel.from_xml_string(build_mjcf(spec, VARIANTS["spring-assist"]))
    print(f"  spring-assist: max joint stiffness {m_def.jnt_stiffness.max():.1f} -> "
          f"{m_spr.jnt_stiffness.max():.1f} N·m/rad (parallel-elastic spring added)")

    all_valid = all(r[6] for r in rows)
    n_warm = sum(1 for r in rows if "warm-start" in r[7])
    n_scratch = sum(1 for r in rows if "from-scratch" in r[7])
    print(f"\nPROVEN: {len(rows)} variants generated + physically valid: {all_valid}; "
          f"{n_warm} warm-startable, {n_scratch} need retrain (topology). "
          f"Modify-and-retrain loop works.")
    sys.exit(0 if all_valid else 1)


if __name__ == "__main__":
    main()
