# SPDX-License-Identifier: MIT
"""One design representation, three views (Phase RS consolidation).

The co-design code grew three private design encoders — `mjx_env.apply_design`,
`train_adversarial._design_model`, and `optimize_design.to_overrides` — that drifted
apart. This module is the SINGLE codec they should share so a design means exactly
one thing everywhere (the "don't fork the sources" rule, applied to the design vector).

Two design spaces, deliberately distinct because they hit the body differently:

  * FAST (policy-facing, `DESIGN_DIM`=3): normalized [mass, stiffness, damping] that
    map to pure `mjx.Model` FIELD edits (no XML rebuild) — this is what the universal
    policy carries in its obs and what UniversalEnv randomizes per-env. The whole
    point of Phase 3 is that a fast-design change is a field tweak, not a recompile.

  * FULL (CEM/topology-facing, 5-D): [thigh_len, calf_len, gear, joint_stiffness,
    torso_mass] that change link geometry / actuator gear → they need `build_mjcf`
    (a new MJModel). This is `optimize_design.PARAMS`.

`apply_fast(mx, d)` reproduces the existing `apply_design` byte-for-byte (so the
trained 3-D universal policy keeps working) while being the one definition. Keeping
the codec backward-compatible is intentional: a representation change would resize
the policy obs and invalidate every checkpoint.
"""

from __future__ import annotations

import sys
from pathlib import Path
import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

# ----- FAST design (3-D, mjx field edits; matches mjx_env.apply_design exactly) -----
FAST_NAMES = ("mass_scale", "joint_stiffness", "damping_scale")
DESIGN_DIM = 3
# (lo, hi) of the REAL quantity each normalized [0,1] coord maps to.
FAST_RANGE = {"mass_scale": (0.6, 1.4), "joint_stiffness": (0.0, 25.0),
              "damping_scale": (0.5, 2.0)}


def fast_denorm(d) -> dict:
    """Normalized [0,1]^3 -> real fast-design quantities (the same affine maps
    hard-coded in apply_design / _design_model, now in one place)."""
    mass_s = 0.6 + 0.8 * d[0]
    stiff = 25.0 * d[1]
    damp_s = 0.5 + 1.5 * d[2]
    return {"mass_scale": mass_s, "joint_stiffness": stiff, "damping_scale": damp_s}


def apply_fast(mx, d, hinge_mask=None):
    """Perturb mjx model fields by a normalized fast-design vector. Backend-agnostic
    (numpy or jnp via mx.replace). Reproduces mjx_env.apply_design; if `hinge_mask`
    is given the spring is applied only to hinge joints (the match-scene variant that
    train_adversarial needs, so the free-joint root is never stiffened)."""
    r = fast_denorm(d)
    repl = dict(body_mass=mx.body_mass * r["mass_scale"],
                body_inertia=mx.body_inertia * r["mass_scale"],
                dof_damping=mx.dof_damping * r["damping_scale"])
    if hinge_mask is None:
        repl["jnt_stiffness"] = mx.jnt_stiffness.at[1:].set(r["joint_stiffness"])
    else:
        xp = _xp(mx.jnt_stiffness)
        repl["jnt_stiffness"] = xp.where(hinge_mask, r["joint_stiffness"], mx.jnt_stiffness)
    return mx.replace(**repl)


# ----- FULL design (5-D, needs build_mjcf; matches optimize_design.PARAMS) -----
FULL_PARAMS = [
    ("thigh_len", 0.14, 0.28, ("leg_defaults", "thigh_len")),
    ("calf_len", 0.14, 0.28, ("leg_defaults", "calf_len")),
    ("gear", 4.0, 12.0, ("actuator", "gear")),
    ("joint_stiffness", 0.0, 25.0, ("leg_defaults", "joint_stiffness")),
    ("torso_mass", 3.0, 9.0, ("torso", "mass")),
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
        import jax.numpy as jnp
        if isinstance(x, jnp.ndarray):
            return jnp
    except Exception:
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
