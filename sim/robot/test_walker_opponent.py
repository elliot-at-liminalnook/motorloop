#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
"""C.2 walker-pursuer smoke: the frozen commanded-env walker (pdval lineage)
actually DRIVES B in the fight arena.

Uses the real pulled pdval checkpoint (sim/build/gpu/out/pdval/pdval.pkl,
obs 53 / act 12) through the real load path (T7 sidecar check included), builds
the walker's commanded-layout obs B-centrically, and asserts B is actuated —
the pursuer that makes standing still LOSE structurally instead of via gates.
"""
from __future__ import annotations

import os
import sys
from pathlib import Path

os.environ.setdefault("MUJOCO_GL", "")
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

import jax
import jax.numpy as jnp
import pytest

import ckpt_meta
import train_adversarial as T

WALKER = HERE.parent / "build" / "gpu" / "out" / "pdval" / "pdval.pkl"

pytestmark = pytest.mark.skipif(not WALKER.exists(),
                                reason="pdval walker checkpoint not pulled locally")

BASE = dict(frame_skip=5, self_collision=False, sep_lo=0.8, sep_hi=1.2,
            engage_obs=True, contact_obs=True)


def test_walker_drives_b():
    ckpt_meta.check_semantics(WALKER, expected_semantics=ckpt_meta.COMMANDED_PD_SEMANTICS,
                              expected_model_hash=None, role="walker opponent (test)")
    infer = T.load_opponent(str(WALKER))
    env = T.AdversarialEnv(opponent="walker", walker_infer=infer, walker_speed=0.3, **BASE)
    s = env.reset(jax.random.PRNGKey(0))
    assert "walker_prev_act" in s.info and s.info["walker_prev_act"].shape == (12,)
    # walker obs layout matches its training contract (53 dims)
    assert env._obs_walker(s.pipeline_state, s.info["walker_prev_act"]).shape == (53,)
    step = jax.jit(env.step)
    b_hinge_ctrl = env._actB[env._hinge_localB]
    for _ in range(5):
        s = step(s, jnp.zeros(env.action_size))
    assert jnp.isfinite(s.reward)
    # B is ACTUATED: the per-substep PD wrote nonzero ctrl on B's hinges...
    assert float(jnp.abs(s.pipeline_state.ctrl[b_hinge_ctrl]).max()) > 1e-3, \
        "walker produced zero ctrl on B — pursuer not wired"
    # ...and its previous action is being tracked (obs feedback loop alive)
    assert float(jnp.abs(s.info["walker_prev_act"]).max()) > 1e-4
    # B's strikers stay unfired (walker has no striker outputs)
    if env._has_striker_b:
        assert float(jnp.abs(s.pipeline_state.ctrl[env._actB[env._B_strike_local]]).max()) == 0.0


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-q"]))
