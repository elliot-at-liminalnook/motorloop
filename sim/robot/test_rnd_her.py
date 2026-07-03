#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
"""Behavioral tests for TRUE RND curiosity and TRUE HER relabeling.

These assert that the features influence learning signals — not that they merely
import.  They FAIL on the previous shallow implementation (RND was fixed-random;
HER did no relabeling):

  RND:
    - the predictor's loss DECREASES when trained on a state
    - novelty DECREASES on a trained/repeated state (and stays high on unseen)
    - the env carries the predictor in info and UPDATES it every step
    - the RND bonus actually changes the env reward
  HER:
    - goal_reward peaks when achieved == goal
    - relabeling rewrites the obs goal to a FUTURE achieved goal
    - relabeling CHANGES the reward by exactly her_coeff*(gr_new - gr_old)
    - fraction=0 is a no-op; the relabeled obs/reward are what PPO would consume
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


# ----------------------------- RND -----------------------------
def test_rnd_predictor_learns():
    print("=== RND: predictor loss decreases, novelty drops on trained state ===")
    from rnd_curiosity import RNDPredictor
    rnd = RNDPredictor(obs_dim=38, hidden_dim=64, output_dim=32, lr=1e-3, seed=0)
    feat = jax.random.normal(jax.random.PRNGKey(1), (16, 38))
    unseen = jax.random.normal(jax.random.PRNGKey(2), (16, 38))
    nov0 = float(jnp.mean(rnd.raw_novelty(feat)))
    nov_unseen0 = float(jnp.mean(rnd.raw_novelty(unseen)))
    losses = [rnd.update(feat) for _ in range(60)]
    nov1 = float(jnp.mean(rnd.raw_novelty(feat)))
    nov_unseen1 = float(jnp.mean(rnd.raw_novelty(unseen)))
    assert losses[-1] < 0.5 * losses[0], f"loss did not decrease: {losses[0]:.4f}->{losses[-1]:.4f}"
    assert nov1 < 0.5 * nov0, f"novelty on trained state did not drop: {nov0:.4f}->{nov1:.4f}"
    # the unseen state stays MORE novel than the trained one (exploration signal)
    assert nov1 < nov_unseen1, "trained state should be less novel than an unseen one"
    print(f"  loss {losses[0]:.4f}->{losses[-1]:.4f}; novelty trained {nov0:.4f}->{nov1:.4f}, "
          f"unseen {nov_unseen0:.4f}->{nov_unseen1:.4f}  PASSED")


def test_rnd_functional():
    print("=== RND: functional make_rnd novelty decreases over updates ===")
    from rnd_curiosity import make_rnd
    rnd = make_rnd(feature_dim=38, hidden_dim=64, output_dim=32, lr=1e-3, key=jax.random.PRNGKey(0))
    feat = jax.random.normal(jax.random.PRNGKey(3), (38,))
    pred, opt = rnd.init_predictor_params, rnd.init_opt_state
    nov0 = float(rnd.novelty(pred, feat))
    for _ in range(50):
        pred, opt, _ = rnd.update(pred, opt, feat)
    nov1 = float(rnd.novelty(pred, feat))
    assert nov1 < 0.5 * nov0, f"functional novelty did not drop: {nov0:.4f}->{nov1:.4f}"
    print(f"  novelty {nov0:.4f}->{nov1:.4f}  PASSED")


def test_rnd_wired_into_env():
    print("=== RND: env carries+updates predictor in info, bonus changes reward ===")
    import train_adversarial as ta
    common = dict(frame_skip=5, sep_lo=0.3, sep_hi=0.6, azimuth=1.0,
                  engage_obs=True, contact_obs=True)
    env = ta.AdversarialEnv(rnd_coefficient=0.1, rnd_hidden_dim=64, rnd_output_dim=32, **common)
    s = env.reset(jax.random.PRNGKey(0))
    assert "rnd_predictor" in s.info and "rnd_opt_state" in s.info, "RND state not in info"
    init_leaves = jax.tree_util.tree_leaves(s.info["rnd_predictor"])
    s1 = env.step(s, jnp.zeros(env.action_size))
    new_leaves = jax.tree_util.tree_leaves(s1.info["rnd_predictor"])
    changed = any(not np.allclose(np.array(a), np.array(b)) for a, b in zip(init_leaves, new_leaves))
    assert changed, "predictor params must change after a step (predictor is training)"
    # same dynamics with rnd off -> reward differs by the (positive) novelty bonus
    env0 = ta.AdversarialEnv(rnd_coefficient=0.0, **common)
    s0 = env0.reset(jax.random.PRNGKey(0))
    r_off = float(env0.step(s0, jnp.zeros(env0.action_size)).reward)
    r_on = float(s1.reward)
    assert r_on > r_off + 1e-6, f"RND bonus should raise reward: off={r_off:.4f} on={r_on:.4f}"
    print(f"  predictor updated in step; reward off={r_off:.4f} -> on={r_on:.4f} (+bonus)  PASSED")


# ----------------------------- HER -----------------------------
def test_goal_reward():
    print("=== HER: goal_reward peaks at achieved==goal ===")
    from her_goal import goal_reward
    g = jnp.array([0.3, 0.5, 0.2, 0.1])
    assert float(goal_reward(g, g)) > 0.99, "reward should be ~1 when achieved==goal"
    far = g + jnp.array([1.0, 1.0, 1.0, 1.0])
    assert float(goal_reward(far, g)) < 0.2, "reward should be small when far"
    print("  PASSED")


def _fake_traj(T=6, B=3, D=12, her_dim=4, seed=0):
    """A goal-conditioned rollout: obs=[features.., goal(4)], with an achieved buffer."""
    k = jax.random.PRNGKey(seed)
    k1, k2, k3, k4 = jax.random.split(k, 4)
    feats = jax.random.normal(k1, (T, B, D - her_dim))
    goal = jax.random.uniform(k2, (B, her_dim))                 # per-episode goal
    goal = jnp.broadcast_to(goal, (T, B, her_dim))
    achieved = jax.random.uniform(k3, (T, B, her_dim))          # achieved at each next state
    obs = jnp.concatenate([feats, goal], axis=-1)
    next_obs = jnp.concatenate([feats + 0.01, goal], axis=-1)
    reward = jax.random.normal(k4, (T, B))
    return obs, next_obs, reward, achieved, goal


def test_her_relabel_rewrites_goal_and_reward():
    print("=== HER: relabel rewrites obs goal to a future achieved goal + adjusts reward ===")
    from her_goal import relabel_goal_arrays, goal_reward
    obs, next_obs, reward, achieved, goal = _fake_traj()
    T, B, _ = obs.shape
    her_coeff, sigma = 0.5, 0.15
    obs2, next2, reward2, info = relabel_goal_arrays(
        obs, next_obs, reward, achieved, goal, jax.random.PRNGKey(7),
        her_coeff, sigma, fraction=1.0)
    obs2, reward2 = np.array(obs2), np.array(reward2)
    mask = np.array(info["mask"]).astype(bool)
    future = np.array(info["future"])
    achieved_np = np.array(achieved)
    # 1) the new goal in obs equals achieved[future] (a future achieved goal, t'>=t)
    for t in range(T):
        for b in range(B):
            if mask[t, b]:
                assert future[t, b] >= t
                assert np.allclose(obs2[t, b, -4:], achieved_np[future[t, b], b], atol=1e-5), \
                    "relabeled obs goal must be the future achieved goal"
    # 2) the reward changed by exactly her_coeff*(gr_new - gr_old)
    gr_old = np.array(goal_reward(achieved, goal, sigma))
    gr_new = np.array(goal_reward(achieved, jnp.array(np.where(mask[..., None],
                       np.array(info["new_goal"]), np.array(goal))), sigma))
    expected = np.array(reward) + her_coeff * (gr_new - gr_old) * mask
    assert np.allclose(reward2, expected, atol=1e-5), "reward delta != her_coeff*(gr_new-gr_old)"
    # the relabel actually MOVED rewards (not a no-op)
    assert np.sum(np.abs(reward2 - np.array(reward)) > 1e-6) > 0, "relabel changed no rewards"
    print(f"  {int(mask.sum())}/{T*B} transitions relabeled; goals+rewards consumed  PASSED")


def test_her_fraction_zero_is_noop():
    print("=== HER: fraction=0 leaves obs and reward unchanged ===")
    from her_goal import relabel_goal_arrays
    obs, next_obs, reward, achieved, goal = _fake_traj(seed=1)
    obs2, next2, reward2, info = relabel_goal_arrays(
        obs, next_obs, reward, achieved, goal, jax.random.PRNGKey(0), 0.5, 0.15, fraction=0.0)
    assert np.allclose(np.array(obs2), np.array(obs))
    assert np.allclose(np.array(reward2), np.array(reward))
    assert int(np.array(info["mask"]).sum()) == 0
    print("  PASSED")


def test_her_install_patches_brax():
    print("=== HER: install_her_relabel patches brax acting.generate_unroll ===")
    try:
        from brax.training import acting
    except Exception as e:
        print(f"  (brax unavailable: {e}) SKIPPED")
        return
    import her_goal
    orig = acting.generate_unroll
    patched = her_goal.install_her_relabel(0.2, sigma=0.15, fraction=0.5)
    assert acting.generate_unroll is patched and patched is not orig, "patch not installed"
    her_goal.uninstall_her_relabel()
    assert acting.generate_unroll is orig, "uninstall did not restore original"
    print("  patch installs + uninstalls cleanly  PASSED")


FAST_TESTS = [test_rnd_predictor_learns, test_rnd_functional, test_rnd_wired_into_env,
              test_goal_reward, test_her_relabel_rewrites_goal_and_reward,
              test_her_fraction_zero_is_noop, test_her_install_patches_brax]

if __name__ == "__main__":
    for t in FAST_TESTS:
        t()
    print("\n=== ALL RND+HER BEHAVIOR TESTS PASSED ===")
