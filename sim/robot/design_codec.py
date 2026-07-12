# SPDX-License-Identifier: MIT
"""One design representation for the MuJoCo-Warp co-design stack.

Older workflows grew private design encoders that drifted apart. This module is
the single codec they share so a design means exactly one thing everywhere.

Two design spaces, deliberately distinct because they hit the body differently:

  * FAST (policy-facing, `DESIGN_DIM`=3): normalized [mass, stiffness, damping]
    mapped to fields on a plain MuJoCo model before it is uploaded to Warp.

  * FULL (CEM/topology-facing, 5-D): [thigh_len, calf_len, gear, joint_stiffness,
    torso_mass] that change link geometry / actuator gear → they need `build_mjcf`
    (a new MJModel). This is `optimize_design.PARAMS`.

Keeping the 3-D representation backward-compatible is intentional: a shape change
would resize the policy observation and invalidate every checkpoint.
"""

from __future__ import annotations

import sys
from pathlib import Path
import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

# ----- FAST design (3-D model-field edits) -----
FAST_NAMES = ("mass_scale", "joint_stiffness", "damping_scale")
DESIGN_DIM = 3
# (lo, hi) of the REAL quantity each normalized [0,1] coord maps to.
FAST_RANGE = {"mass_scale": (0.6, 1.0), "joint_stiffness": (0.0, 25.0),
              "damping_scale": (0.5, 2.0)}


def fast_denorm(d) -> dict:
    """Normalized [0,1]^3 -> real fast-design quantities (the same affine maps
    hard-coded in apply_design / _design_model, now in one place)."""
    # Preserve the historical midpoint (d=0.5 -> nominal) while clipping the
    # overweight half of old checkpoints to the legal 6 lb envelope.
    xp = _xp(d)
    raw_mass = 0.6 + 0.8 * d[0]
    mass_s = xp.clamp(raw_mass, max=1.0) if xp is not np else np.minimum(1.0, raw_mass)
    stiff = 25.0 * d[1]
    damp_s = 0.5 + 1.5 * d[2]
    return {"mass_scale": mass_s, "joint_stiffness": stiff, "damping_scale": damp_s}


def apply_fast_mujoco(model, d):
    """Apply a normalized fast design to a mutable ``mujoco.MjModel`` in place."""
    r = fast_denorm(np.asarray(d, dtype=float))
    model.body_mass[:] *= float(r["mass_scale"])
    model.body_inertia[:] *= float(r["mass_scale"])
    model.dof_damping[:] *= float(r["damping_scale"])
    # Free joints are never spring-loaded. Limited hinge/slide joints receive
    # the design stiffness; this happens before the model is uploaded to Warp.
    limited = model.jnt_limited.astype(bool)
    model.jnt_stiffness[limited] = float(r["joint_stiffness"])
    return model


# ----- FULL design (5-D, needs build_mjcf; matches optimize_design.PARAMS) -----
FULL_PARAMS = [
    ("thigh_len", 0.14, 0.28, ("leg_defaults", "thigh_len")),
    ("calf_len", 0.14, 0.28, ("leg_defaults", "calf_len")),
    ("gear", 1.0, 6.0, ("actuator", "gear")),
    ("joint_stiffness", 0.0, 25.0, ("leg_defaults", "joint_stiffness")),
    ("torso_mass", 0.35, 0.5479633165, ("torso", "mass")),
]
FULL_DIM = len(FULL_PARAMS)


def full_norm_to_real(u):
    """Normalized [0,1]^5 -> real units (clipped)."""
    lo = np.array([p[1] for p in FULL_PARAMS]); hi = np.array([p[2] for p in FULL_PARAMS])
    return lo + np.clip(np.asarray(u, float), 0, 1) * (hi - lo)


def full_to_overrides(u_or_x, normalized=True) -> dict:
    """Full design (normalized OR real) -> a build_mjcf override dict (deep-mergeable
    onto robot.toml). This is the one `to_overrides` definition."""
    x = full_norm_to_real(u_or_x) if normalized else np.asarray(u_or_x, float)
    ov: dict = {}
    for v, (_, _, _, (sec, key)) in zip(x, FULL_PARAMS):
        ov.setdefault(sec, {})[key] = float(v)
    return ov


def full_default(spec) -> np.ndarray:
    """The robot.toml default as a real-units full-design vector."""
    return np.array([spec["leg_defaults"]["thigh_len"], spec["leg_defaults"]["calf_len"],
                     spec["actuator"]["gear"], spec["leg_defaults"]["joint_stiffness"],
                     spec["torso"]["mass"]])


def _xp(x):
    try:
        import torch
        if isinstance(x, torch.Tensor):
            return torch
    except ImportError:
        pass
    return np


if __name__ == "__main__":          # CPU self-test (no GPU): consolidation + roundtrips
    from gen_robot_mjcf import build_mjcf, load_spec
    spec = load_spec(HERE / "robot.toml")

    # 1. fast_denorm matches the affine maps the envs used inline
    d = [0.5, 0.5, 0.5]
    r = fast_denorm(d)
    assert abs(r["mass_scale"] - 1.0) < 1e-9 and abs(r["damping_scale"] - 1.25) < 1e-9
    print(f"fast mid-design -> {r}")

    # 2. full override roundtrip lands the requested real values in the spec
    u = np.array([0.3, 0.7, 0.5, 0.2, 0.6])
    ov = full_to_overrides(u)
    x = full_norm_to_real(u)
    assert abs(ov["leg_defaults"]["thigh_len"] - x[0]) < 1e-9
    assert abs(ov["actuator"]["gear"] - x[2]) < 1e-9
    m = build_mjcf(spec, ov)              # builds a valid MJCF
    print(f"full design {np.round(x,3)} -> override -> MJCF built ({len(m)} chars)")

    # 3. default vector roundtrips through overrides unchanged
    x0 = full_default(spec)
    ov0 = full_to_overrides(x0, normalized=False)
    assert abs(ov0["torso"]["mass"] - spec["torso"]["mass"]) < 1e-9
    print(f"default full-design {np.round(x0,3)} consistent with robot.toml")
    print("design_codec self-check OK — one codec for fast(3) + full(5) design views")
