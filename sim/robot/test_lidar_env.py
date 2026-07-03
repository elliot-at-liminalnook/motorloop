#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
"""Behavioral tests for the lidar-enabled AdversarialEnv.

These assert BEHAVIOR, not just shapes — and several FAIL on the previous shallow
implementation (per-env RNG was PRNGKey(0)+timestep; latency was a no-op at
frame_stack==1):

  1. obs dict shapes (state/value_state) for all flag combinations
  2. per-env / per-episode RNG: noisy scans differ across envs and across resets
  3. deterministic CLEAN scans when noise+dropout are off (benchmark determinism)
  4. noisy scan VARIATION when noise is on; stepping advances the noise
  5. dropout marks rays as max-range, and differs across envs
  6. LATENCY changes the actor AND critic obs even at frame_stack==1
  7. frame-stack + latency + goal dims compose with correct dimensions
"""
import os
import sys
from pathlib import Path

os.environ.setdefault("MUJOCO_GL", "osmesa")
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

import jax
import jax.numpy as jnp
import numpy as np

import train_adversarial as ta

LO = ta.LOCO_OBS


def _mk(**kw):
    base = dict(frame_skip=5, sep_lo=0.25, sep_hi=0.70, azimuth=3.14159,
                engage_obs=True, contact_obs=True, lidar=True,
                lidar_n_rays=16, lidar_n_vertical=4, lidar_max_range=2.0)
    base.update(kw)
    return ta.AdversarialEnv(**base)


def _scan_slice(env, obs):
    """Actor-obs lidar scan portion (after loco), length n_total*stack."""
    n = env._lidar_n_total * max(1, env._lidar_stack)
    return np.array(obs["state"][LO:LO + n])


def test_dims_basic():
    print("=== Test: obs dict dims ===")
    env = _mk(lidar_noise_sigma=0.01, lidar_dropout_rate=0.02, lidar_frame_stack=3)
    n_total = 16 + 4
    assert isinstance(env.observation_size, dict)
    # B.1 obs migration: + history block (H frames of hinge qpos/qvel + prev action);
    # critic additionally sees the 4 per-foot contact booleans (privileged).
    hist = env._hist_dim
    assert env.observation_size["state"] == LO + n_total * 3 + hist
    assert env.observation_size["value_state"] == LO + n_total * 3 + 6 + 8 + 8 + hist + 4
    st = env.reset(jax.random.PRNGKey(0))
    assert np.array(st.obs["state"]).shape == (env.observation_size["state"],)
    assert np.array(st.obs["value_state"]).shape == (env.observation_size["value_state"],)
    print(f"  state={env.observation_size['state']} value={env.observation_size['value_state']}  PASSED")


def test_deterministic_clean_scan():
    print("=== Test: clean scan is deterministic (noise+dropout off) ===")
    env = _mk(lidar_noise_sigma=0.0, lidar_dropout_rate=0.0, lidar_frame_stack=1)
    assert env._lidar_stochastic is False
    k = jax.random.PRNGKey(7)
    a = _scan_slice(env, env.reset(k).obs)
    b = _scan_slice(env, env.reset(k).obs)
    assert np.allclose(a, b), "same key -> identical clean scan expected"
    # repeated step from the same state is identical too
    s = env.reset(k)
    s1 = env.step(s, jnp.zeros(env.action_size))
    s2 = env.step(s, jnp.zeros(env.action_size))
    assert np.allclose(np.array(s1.obs["state"]), np.array(s2.obs["state"]))
    # at least some rays hit the opponent (scan < max)
    assert np.sum(a < 0.999) > 0, "expected some lidar hits on the opponent"
    print(f"  clean scan deterministic; {int(np.sum(a < 0.999))} rays hit  PASSED")


def test_per_env_and_episode_rng():
    print("=== Test: noisy scans differ across envs AND episodes ===")
    env = _mk(lidar_noise_sigma=0.03, lidar_dropout_rate=0.0, lidar_frame_stack=1)
    # different reset keys (different envs) -> different noise on the SAME geometry
    keys = jax.random.split(jax.random.PRNGKey(123), 8)
    scans = np.stack([_scan_slice(env, env.reset(k).obs) for k in keys])
    per_ray_std = scans.std(axis=0)
    assert np.mean(per_ray_std) > 1e-4, "scans should differ across envs (per-env RNG)"
    # same env, two different episodes (different keys) -> different noise
    a = _scan_slice(env, env.reset(keys[0]).obs)
    b = _scan_slice(env, env.reset(keys[1]).obs)
    assert not np.allclose(a, b), "different episodes should have different noise"
    # stepping advances the per-env RNG: two consecutive steps differ in noise
    s = env.reset(keys[0])
    s1 = env.step(s, jnp.zeros(env.action_size))
    s2 = env.step(s1, jnp.zeros(env.action_size))
    assert not np.allclose(np.array(s1.obs["state"][LO:]), np.array(s2.obs["state"][LO:])), \
        "noise should advance each step"
    print(f"  cross-env mean per-ray std={np.mean(per_ray_std):.4f}  PASSED")


def test_vmap_per_env_rng():
    print("=== Test: vmapped reset gives per-env noise (training-like) ===")
    env = _mk(lidar_noise_sigma=0.03, lidar_dropout_rate=0.02, lidar_frame_stack=2)
    keys = jax.random.split(jax.random.PRNGKey(321), 64)
    states = jax.jit(jax.vmap(env.reset))(keys)
    scans = np.array(states.obs["state"])[:, LO:LO + env._lidar_n_total * 2]
    assert scans.std(axis=0).mean() > 1e-4, "batched envs must see independent noise"
    print(f"  64-env batched per-ray std={scans.std(axis=0).mean():.4f}  PASSED")


def test_dropout_marks_max_range():
    print("=== Test: dropout sets rays to max range, differs across envs ===")
    clean = _mk(lidar_noise_sigma=0.0, lidar_dropout_rate=0.0, lidar_frame_stack=1)
    drop = _mk(lidar_noise_sigma=0.0, lidar_dropout_rate=0.5, lidar_frame_stack=1)
    k = jax.random.PRNGKey(5)
    c = _scan_slice(clean, clean.reset(k).obs)
    d0 = _scan_slice(drop, drop.reset(k).obs)
    d1 = _scan_slice(drop, drop.reset(jax.random.PRNGKey(6)).obs)
    # dropped rays read 1.0 (max). With 50% dropout many hit-rays become 1.0.
    newly_max = np.sum((c < 0.999) & (d0 > 0.999))
    assert newly_max > 0, "dropout should push some hit rays to max range"
    assert not np.allclose(d0, d1), "dropout pattern should differ across envs"
    print(f"  {int(newly_max)} hit-rays dropped to max; pattern differs across envs  PASSED")


def test_latency_affects_obs_at_stack1():
    print("=== Test: latency changes actor+critic obs even at frame_stack==1 ===")
    no_lat = _mk(lidar_noise_sigma=0.0, lidar_dropout_rate=0.0,
                 lidar_frame_stack=1, lidar_latency_steps=0)
    lat = _mk(lidar_noise_sigma=0.0, lidar_dropout_rate=0.0,
              lidar_frame_stack=1, lidar_latency_steps=2)
    k = jax.random.PRNGKey(9)
    s_no = no_lat.reset(k)
    s_la = lat.reset(k)
    n = no_lat._lidar_n_total
    actor_no = np.array(s_no.obs["state"][LO:LO + n])
    actor_la = np.array(s_la.obs["state"][LO:LO + n])
    crit_no = np.array(s_no.obs["value_state"][LO:LO + n])
    crit_la = np.array(s_la.obs["value_state"][LO:LO + n])
    # With latency=2, the first observed scan is the init history (all max=1.0),
    # while the no-latency env shows real hits. The OLD code left these identical.
    assert not np.allclose(actor_no, actor_la), "latency must change the ACTOR obs at stack 1"
    assert not np.allclose(crit_no, crit_la), "latency must change the CRITIC obs at stack 1"
    assert np.allclose(actor_la, 1.0), "first latency-delayed scan should be the init (max range)"
    # after `latency` steps the observed scan equals the clean reset-time scan
    s = s_la
    for _ in range(2):
        s = lat.step(s, jnp.zeros(lat.action_size))
    observed_now = np.array(s.obs["state"][LO:LO + n])
    assert not np.allclose(observed_now, 1.0), "after latency steps a real scan should surface"
    print(f"  actor/critic differ under latency; delayed scan surfaces after 2 steps  PASSED")


def test_framestack_latency_goal_dims():
    print("=== Test: frame-stack + latency + goal dims compose ===")
    env = _mk(lidar_noise_sigma=0.01, lidar_dropout_rate=0.02, lidar_frame_stack=3,
              lidar_latency_steps=1, her_coefficient=0.5)
    n_total = 16 + 4
    # B.1: + history block in both heads; + 4 foot contacts in the critic only.
    hist = env._hist_dim
    exp_state = LO + n_total * 3 + hist + 4
    exp_value = LO + n_total * 3 + 6 + 8 + 8 + hist + 4 + 4
    assert env.observation_size["state"] == exp_state
    assert env.observation_size["value_state"] == exp_value
    s = env.reset(jax.random.PRNGKey(11))
    assert np.array(s.obs["state"]).shape == (exp_state,)
    assert np.array(s.obs["value_state"]).shape == (exp_value,)
    # goal occupies the last 4 dims of BOTH heads and matches info['her_goal']
    goal = np.array(s.info["her_goal"])
    assert np.allclose(np.array(s.obs["state"][-4:]), goal)
    assert np.allclose(np.array(s.obs["value_state"][-4:]), goal)
    s2 = env.step(s, jnp.zeros(env.action_size))
    assert np.array(s2.obs["state"]).shape == (exp_state,)
    assert np.array(s2.obs["value_state"]).shape == (exp_value,)
    # her_achieved is exposed for the relabel pass
    assert np.array(s2.info["her_achieved"]).shape == (4,)
    print(f"  state={exp_state} value={exp_value}, goal in both heads, her_achieved present  PASSED")


def test_jit_and_backward_compat():
    print("=== Test: jit works + non-lidar backward compat ===")
    env = _mk(lidar_noise_sigma=0.01, lidar_frame_stack=2, lidar_latency_steps=1)

    @jax.jit
    def reset_step(rng):
        s = env.reset(rng)
        s2 = env.step(s, jnp.zeros(env.action_size))
        return s2.obs["state"], s2.obs["value_state"], s2.reward
    a, c, r = reset_step(jax.random.PRNGKey(3))
    assert np.array(a).shape == (env.observation_size["state"],)
    # non-lidar path still returns a flat array
    flat = ta.AdversarialEnv(frame_skip=5, sep_lo=0.25, sep_hi=0.70, azimuth=3.14159,
                             engage_obs=True, contact_obs=True, lidar=False)
    assert not isinstance(flat.observation_size, dict)
    fs = flat.reset(jax.random.PRNGKey(0))
    assert not isinstance(fs.obs, dict)
    print("  jit ok; non-lidar flat obs preserved  PASSED")


def test_obs_combinations():
    print("=== Test: obs dims match across flag combinations ===")
    configs = [
        ("no-lidar/no-HER", dict(lidar=False, her_coefficient=0.0)),
        ("no-lidar/HER", dict(lidar=False, her_coefficient=0.5)),
        ("lidar/HER/stack1/lat0", dict(lidar=True, her_coefficient=0.5, lidar_n_rays=8,
                                       lidar_n_vertical=2, lidar_frame_stack=1)),
        ("lidar/HER/stack3/lat2", dict(lidar=True, her_coefficient=0.5, lidar_n_rays=8,
                                       lidar_n_vertical=2, lidar_frame_stack=3, lidar_latency_steps=2)),
    ]
    for name, cfg in configs:
        env = ta.AdversarialEnv(frame_skip=5, sep_lo=0.25, sep_hi=0.70, azimuth=3.14159,
                                engage_obs=True, contact_obs=True, **cfg)
        s = env.reset(jax.random.PRNGKey(1))
        s2 = env.step(s, jnp.zeros(env.action_size))
        if isinstance(env.observation_size, dict):
            for st in (s, s2):
                assert np.array(st.obs["state"]).shape == (env.observation_size["state"],)
                assert np.array(st.obs["value_state"]).shape == (env.observation_size["value_state"],)
        else:
            for st in (s, s2):
                assert np.array(st.obs).shape == (env.observation_size,)
        print(f"  {name}: ok")
    print("  PASSED")


FAST_TESTS = [test_dims_basic, test_deterministic_clean_scan, test_per_env_and_episode_rng,
              test_vmap_per_env_rng, test_dropout_marks_max_range,
              test_latency_affects_obs_at_stack1, test_framestack_latency_goal_dims,
              test_jit_and_backward_compat, test_obs_combinations]

if __name__ == "__main__":
    for t in FAST_TESTS:
        t()
    print("\n=== ALL LIDAR BEHAVIOR TESTS PASSED ===")
