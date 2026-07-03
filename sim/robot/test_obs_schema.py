#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
"""B.1 obs/action schema tests: the layout contracts that HER/RND/self-play
implicitly depend on, made executable.

  1. flat + lidar obs sizes match the env's declared observation_size after the
     history migration (reset AND step — a mismatch between the two is the bug
     class that breaks brax's scan-carry silently)
  2. the HER goal occupies the LAST 4 dims (her_goal.py's relabel contract)
  3. the critic (value_state) sees exactly [priv(6+e+c) + foot contacts(4)]
     more than the actor — contacts must NOT leak into the actor
  4. PD action mode: a zero-action rollout HOLDS THE STANCE (the single sanity
     property that makes position-target control the right action space: random
     exploration perturbs a stance instead of free-falling)
  5. obsB mirrors A's flat layout including the history block (self-play
     contract for post-B.1 snapshots)
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

import train_adversarial as T

BASE = dict(frame_skip=5, self_collision=False, sep_lo=0.3, sep_hi=0.6)


def test_flat_obs_sizes_and_pd_stance_hold():
    env = T.AdversarialEnv(engage_obs=True, contact_obs=True, **BASE)
    s = env.reset(jax.random.PRNGKey(0))
    assert s.obs.shape == (env.observation_size,), (s.obs.shape, env.observation_size)
    step = jax.jit(env.step)
    z0 = float(s.pipeline_state.xpos[env._At][2])
    for _ in range(50):
        s = step(s, jnp.zeros(env.action_size))
    assert s.obs.shape == (env.observation_size,), "step obs diverged from reset obs"
    z = float(s.pipeline_state.xpos[env._At][2])
    # PD mode, zero action = hold the stance: the torso must NOT collapse.
    assert z > 0.7 * z0, f"zero-action PD rollout collapsed: z {z0:.3f} -> {z:.3f}"
    assert jnp.isfinite(s.reward)


def test_lidar_obs_dict_sizes_and_critic_privilege():
    env = T.AdversarialEnv(engage_obs=True, contact_obs=True, her_coefficient=0.2,
                           lidar=True, lidar_n_rays=8, lidar_n_vertical=2,
                           lidar_frame_stack=2, lidar_noise_sigma=0.0,
                           lidar_dropout_rate=0.0, **BASE)
    sizes = env.observation_size
    s = env.reset(jax.random.PRNGKey(1))
    assert s.obs["state"].shape == (sizes["state"],)
    assert s.obs["value_state"].shape == (sizes["value_state"],)
    # critic extra = privileged tail (6 + 8 engage + 8 contact) + 4 foot contacts
    assert sizes["value_state"] - sizes["state"] == 6 + 8 + 8 + 4
    # HER goal tail is LAST in both actor and critic (her_goal.py relabel contract)
    g = s.info["her_goal"]
    assert jnp.allclose(s.obs["state"][-4:], g)
    assert jnp.allclose(s.obs["value_state"][-4:], g)
    s2 = env.step(s, jnp.zeros(env.action_size))
    assert s2.obs["state"].shape == (sizes["state"],)
    assert jnp.allclose(s2.obs["state"][-4:], g), "goal tail moved after a step"


def test_obsB_mirrors_flat_layout_with_history():
    env = T.AdversarialEnv(engage_obs=True, contact_obs=True, opponent="frozen",
                           **BASE)
    s = env.reset(jax.random.PRNGKey(2))
    b = env._obsB(s.pipeline_state, s.info["design"], info=s.info)
    assert b.shape == (env.obsB_size,), (b.shape, env.obsB_size)
    # A's flat obs (no HER) and B's mirror have identical widths — the self-play
    # symmetry that lets an A-snapshot drive B.
    assert env.obsB_size == env.observation_size


def test_torque_mode_is_still_available():
    env = T.AdversarialEnv(action_mode="torque", history_len=0, **BASE)
    s = env.reset(jax.random.PRNGKey(3))
    s = env.step(s, jnp.zeros(env.action_size))
    assert jnp.isfinite(s.reward)


if __name__ == "__main__":
    import pytest
    raise SystemExit(pytest.main([__file__, "-q"]))


# ---------------------------------------------------------------- V.4 schema
def test_schema_totals_match_observation_size():
    """obs_schema (the described layout) == observation_size (the assembled one),
    across lidar/flat × her on/off × history on/off."""
    import obs_schema as S
    combos = [
        dict(engage_obs=True, contact_obs=True),                          # flat, hist=3
        dict(engage_obs=True, contact_obs=True, her_coefficient=0.2,
             history_len=0),                                              # flat, her, no hist
        dict(engage_obs=True, contact_obs=True, her_coefficient=0.2,
             lidar=True, lidar_n_rays=8, lidar_n_vertical=2,
             lidar_frame_stack=2, lidar_noise_sigma=0.0,
             lidar_dropout_rate=0.0),                                     # lidar, her, hist=3
        dict(lidar=True, lidar_n_rays=8, lidar_n_vertical=2,
             history_len=0, lidar_noise_sigma=0.0, lidar_dropout_rate=0.0),
    ]
    for kw in combos:
        env = T.AdversarialEnv(**{**BASE, **kw})
        a, c = S.actor_slices(env), S.critic_slices(env)
        size = env.observation_size
        if isinstance(size, dict):
            assert S.total(a) == size["state"], (kw, dict(a), size)
            assert S.total(c) == size["value_state"], (kw, dict(c), size)
        else:
            assert S.total(a) == size == S.total(c), (kw, dict(a), size)


def test_schema_her_goal_slice_matches_real_reset():
    """The schema's her_goal slice lands exactly on info['her_goal'], LAST, in
    both heads AND the flat layout (her_goal.relabel_goal_arrays' contract)."""
    import obs_schema as S
    from her_goal import GOAL_DIM

    env = T.AdversarialEnv(engage_obs=True, contact_obs=True, her_coefficient=0.2,
                           lidar=True, lidar_n_rays=8, lidar_n_vertical=2,
                           lidar_noise_sigma=0.0, lidar_dropout_rate=0.0, **BASE)
    s = env.reset(jax.random.PRNGKey(5))
    for head, sl_map in (("state", S.actor_slices(env)),
                         ("value_state", S.critic_slices(env))):
        sl = sl_map["her_goal"]
        assert sl.stop - sl.start == GOAL_DIM
        assert sl.stop == s.obs[head].shape[0], f"{head}: her_goal not LAST"
        assert jnp.allclose(s.obs[head][sl], s.info["her_goal"])

    flat = T.AdversarialEnv(engage_obs=True, contact_obs=True,
                            her_coefficient=0.2, **BASE)
    sf = flat.reset(jax.random.PRNGKey(6))
    sl = S.actor_slices(flat)["her_goal"]
    assert sl.stop == sf.obs.shape[0] and jnp.allclose(sf.obs[sl], sf.info["her_goal"])
