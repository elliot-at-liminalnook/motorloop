# SPDX-License-Identifier: MIT
"""SPARC judging as the objective (tournament_docs/ Judging Guidelines v1.3).

The match objective is to WIN THE DECISION: maximize OUR points, minimize the
OPPONENT's, across the Damage/Control/Aggression criteria (6/6/5 = 17 pts). This is
the single source of truth for every reward + co-evolution fitness - not "just
survive." Two behavioral rules from the guidelines are baked in:
  * Aggression counts ONLY translational movement TOWARD the opponent; fleeing and
    sitting-still score zero (this is exactly why our fleeing policy would also lose
    on the scorecard, not just fail to dodge).
  * Damage is graded RELATIVELY (dealt vs taken), so avoiding damage and dealing it
    are the same currency.

Pure functions, no deps - importable by the combat RL reward, the co-evolution
harness, and the future MJX self-play match.
"""

from __future__ import annotations

DAMAGE_MAX, CONTROL_MAX, AGGRESSION_MAX = 6.0, 6.0, 5.0
TOTAL = DAMAGE_MAX + CONTROL_MAX + AGGRESSION_MAX        # 17


def _c(x: float) -> float:
    return 0.0 if x < 0.0 else 1.0 if x > 1.0 else float(x)


def points(damage: float, control: float, aggression: float) -> float:
    """Category fractions in [0,1] -> SPARC points (max 17)."""
    return DAMAGE_MAX * _c(damage) + CONTROL_MAX * _c(control) + AGGRESSION_MAX * _c(aggression)


def differential(ours: dict, theirs: dict) -> float:
    """our points - opponent points; > 0 means we win the judges' decision."""
    return points(**ours) - points(**theirs)


# --- bridge: a match rollout's measurables -> category fractions ---
def damage_fraction(dealt: float, taken: float) -> float:
    """Relative damage grade: our share of the damage done. 0.5 if nothing landed."""
    tot = dealt + taken
    return 0.5 if tot <= 1e-9 else _c(dealt / tot)


def aggression_fraction(time_closing: float, time_fleeing: float, time_total: float) -> float:
    """Only movement TOWARD the opponent counts; fleeing subtracts (SPARC 1.2.1)."""
    if time_total <= 0.0:
        return 0.0
    return _c((time_closing - time_fleeing) / time_total)


def control_fraction(weapon_denied: float, position_dominance: float) -> float:
    """Controlling the flow without the weapon doing damage (pin, jam, position)."""
    return _c(0.5 * weapon_denied + 0.5 * position_dominance)


def step_reward(dealt: float = 0.0, taken: float = 0.0, closing: float = 0.0,
                fleeing: float = 0.0, control: float = 0.0) -> float:
    """Per-step SPARC-differential reward = OUR points - the OPPONENT's, so the
    policy is optimized to WIN THE DECISION. `dealt`/`taken` are damage events (our
    weapon hits them / theirs hits us); `closing`/`fleeing` are the toward/away
    velocity fractions (aggression credits closing only - fleeing is a penalty, per
    SPARC 1.2.1); `control` is positional dominance. Needs a robot that can deal
    damage (the weapon-leg body) for the `dealt` term to be real.

    Backend-agnostic: scalars (numpy source of truth) OR jnp arrays (the MJX twin)
    flow through identical arithmetic — this IS the single SPARC source the CPU
    match, the co-evolution, and the MJX self-play all share (no fork)."""
    return (DAMAGE_MAX * (dealt - taken)
            + AGGRESSION_MAX * (closing - fleeing)
            + CONTROL_MAX * control)


def step_reward_jax(dealt, taken, closing, fleeing, control, *, xp=None):
    """JAX/numpy twin of `step_reward` that *clamps each fraction to its SPARC range*
    before differencing (the per-step combat envs pass already-saturated fractions;
    this makes the contract explicit and identical across backends). `closing`/
    `fleeing` are non-negative toward/away fractions; `dealt`/`taken` are in [0,1]
    severity units (see `reality_gap.damage_from_force`). `xp` defaults to jnp if
    available else numpy, so the same call works on GPU (traced) and CPU (eager)."""
    if xp is None:
        try:
            import jax.numpy as jnp
            xp = jnp
        except Exception:                       # pragma: no cover - CPU fallback
            import numpy as xp                   # type: ignore
    clip01 = lambda x: xp.clip(x, 0.0, 1.0)
    return (DAMAGE_MAX * (clip01(dealt) - clip01(taken))
            + AGGRESSION_MAX * (clip01(closing) - clip01(fleeing))
            + CONTROL_MAX * clip01(control))
