# SPDX-License-Identifier: MIT
"""Hindsight Experience Replay (HER) for on-policy PPO — true goal relabeling.

Brax PPO is on-policy, so classic off-policy HER replay does not apply directly.
This module implements a HER-equivalent that IS compatible: a hindsight
relabeling PASS over each collected rollout window, applied BEFORE PPO computes
rewards/advantages.

For each transition t in an env's unroll, with probability ``fraction`` we relabel
its goal with a goal that was ACHIEVED at some FUTURE step t' >= t in the same
unroll (the "future" HER strategy).  Relabeling does two things that PPO then
consumes verbatim:

  1. it overwrites the goal dimensions in the observation AND next_observation
     (so the policy/value networks are trained on the relabeled goal), and
  2. it recomputes the goal-achievement reward for the relabeled goal
     (``reward += her_coeff * (gr(achieved_t, new_goal) - gr(achieved_t, goal_t))``),

turning a trajectory that "failed" its sampled goal into a success demonstration
for the goal it actually reached.

Wiring: :func:`install_her_relabel` monkeypatches ``brax.training.acting``'s
``generate_unroll`` so every PPO rollout is relabeled.  The env must place
``her_achieved`` (the achieved goal at the next state) and ``her_goal`` (the
active goal) in ``state.info`` so they are collected as transition extras.

The 4D contact-acquisition goal space (distance, bearing, front-alignment,
rod-distance) is shared with :class:`AdversarialEnv`.
"""

from __future__ import annotations

from collections import deque
import random

import jax
import jax.numpy as jnp

# Per-dimension weighting of the goal distance (distance & rod-dist matter most).
GOAL_WEIGHTS = jnp.array([1.0, 0.5, 0.3, 1.0])
GOAL_DIM = 4


def goal_reward(achieved, goal, sigma: float = 0.15, weights=GOAL_WEIGHTS):
    """Gaussian goal-achievement reward in [0,1]; 1 when achieved == goal.

    Batched over leading axes; the goal axis is the last one.
    """
    diff = (achieved - goal) * weights
    dist_sq = jnp.sum(diff ** 2, axis=-1)
    return jnp.exp(-dist_sq / (2.0 * sigma ** 2))


def _future_indices(key, n_steps: int, n_envs: int, fraction: float):
    """Sample, per (t, env), a relabel mask and a FUTURE index t' in [t, T-1]."""
    tk, fk = jax.random.split(key)
    mask = jax.random.bernoulli(tk, fraction, (n_steps, n_envs))
    t_idx = jnp.arange(n_steps)[:, None]              # (T,1)
    span = n_steps - t_idx                            # (T,1) future steps incl. current
    u = jax.random.uniform(fk, (n_steps, n_envs))     # [0,1)
    offset = jnp.floor(u * span).astype(jnp.int32)
    future = jnp.clip(t_idx + offset, 0, n_steps - 1)  # (T,B)
    return mask, future


def relabel_goal_arrays(obs, next_obs, reward, achieved, goal, key,
                        her_coeff: float, sigma: float = 0.15, fraction: float = 0.5,
                        her_dim: int = GOAL_DIM, weights=GOAL_WEIGHTS):
    """Hindsight-relabel a single (T, B, D) goal-conditioned rollout array.

    The goal occupies the LAST ``her_dim`` columns of ``obs``/``next_obs``.
    ``achieved`` is the achieved goal at each transition's NEXT state, ``goal``
    is the active per-episode goal.  Returns (obs', next_obs', reward', info)
    where info carries the boolean ``mask`` and ``new_goal`` (for tests/inspection).
    """
    n_steps, n_envs = reward.shape
    mask, future = _future_indices(key, n_steps, n_envs, fraction)
    idx = jnp.broadcast_to(future[..., None], (n_steps, n_envs, her_dim))
    future_goal = jnp.take_along_axis(achieved, idx, axis=0)   # achieved[t', b]
    new_goal = jnp.where(mask[..., None], future_goal, goal)

    gr_old = goal_reward(achieved, goal, sigma, weights)
    gr_new = goal_reward(achieved, new_goal, sigma, weights)
    reward2 = reward + her_coeff * (gr_new - gr_old) * mask

    def _set_goal(o):
        head = o[..., :-her_dim]
        tail = jnp.where(mask[..., None], new_goal, o[..., -her_dim:])
        return jnp.concatenate([head, tail], axis=-1)

    return _set_goal(obs), _set_goal(next_obs), reward2, {
        "mask": mask, "new_goal": new_goal, "future": future}


def make_her_unroll(orig_unroll, her_coeff: float, sigma: float = 0.15,
                    fraction: float = 0.5, her_dim: int = GOAL_DIM):
    """Wrap brax ``generate_unroll`` to relabel each rollout window in hindsight.

    The wrapped function requests the ``her_achieved``/``her_goal`` extras, relabels
    the returned rollout, then drops those extras so downstream PPO sees the
    standard Transition structure (with relabeled obs/next_obs/reward).
    """
    def unroll(env, env_state, policy, key, unroll_length, extra_fields=()):
        key, rkey = jax.random.split(key)
        fields = tuple(dict.fromkeys(tuple(extra_fields) + ("truncation", "her_achieved", "her_goal")))
        final_state, data = orig_unroll(env, env_state, policy, key, unroll_length,
                                        extra_fields=fields)
        se = data.extras["state_extras"]
        achieved, goal = se["her_achieved"], se["her_goal"]
        obs, next_obs, reward = data.observation, data.next_observation, data.reward

        if isinstance(obs, dict):
            # Asymmetric AC: the goal is the last her_dim of BOTH heads; relabel
            # both with the SAME mask/new_goal (computed from the 'state' head).
            n_steps, n_envs = reward.shape
            mask, future = _future_indices(rkey, n_steps, n_envs, fraction)
            idx = jnp.broadcast_to(future[..., None], (n_steps, n_envs, her_dim))
            future_goal = jnp.take_along_axis(achieved, idx, axis=0)
            new_goal = jnp.where(mask[..., None], future_goal, goal)
            gr_old = goal_reward(achieved, goal, sigma)
            gr_new = goal_reward(achieved, new_goal, sigma)
            reward2 = reward + her_coeff * (gr_new - gr_old) * mask

            def _set(o):
                return jnp.concatenate(
                    [o[..., :-her_dim], jnp.where(mask[..., None], new_goal, o[..., -her_dim:])],
                    axis=-1)

            obs2 = {k: (_set(v) if k in ("state", "value_state") else v) for k, v in obs.items()}
            next2 = {k: (_set(v) if k in ("state", "value_state") else v)
                     for k, v in next_obs.items()}
        else:
            obs2, next2, reward2, _ = relabel_goal_arrays(
                obs, next_obs, reward, achieved, goal, rkey, her_coeff, sigma, fraction, her_dim)

        se2 = {k: v for k, v in se.items() if k not in ("her_achieved", "her_goal")}
        extras2 = {**data.extras, "state_extras": se2}
        # brax Transition is a NamedTuple -> use _replace (not .replace).
        data = data._replace(observation=obs2, next_observation=next2, reward=reward2,
                             extras=extras2)
        return final_state, data

    return unroll


_ORIG_UNROLL = None


def install_her_relabel(her_coeff: float, sigma: float = 0.15, fraction: float = 0.5,
                        her_dim: int = GOAL_DIM):
    """Monkeypatch brax PPO's rollout to apply hindsight relabeling.

    Idempotent: a second call re-wraps the ORIGINAL generate_unroll, not a
    double-wrapped one.  Returns the patched function for inspection/tests.
    """
    global _ORIG_UNROLL
    from brax.training import acting
    if _ORIG_UNROLL is None:
        _ORIG_UNROLL = acting.generate_unroll
    patched = make_her_unroll(_ORIG_UNROLL, her_coeff, sigma, fraction, her_dim)
    acting.generate_unroll = patched
    print(f"HER relabel installed: coeff={her_coeff} sigma={sigma} fraction={fraction} "
          f"goal_dim={her_dim} (on-policy future-goal relabel over each PPO rollout)",
          flush=True)
    return patched


def uninstall_her_relabel():
    """Restore the original generate_unroll (used by tests)."""
    global _ORIG_UNROLL
    if _ORIG_UNROLL is not None:
        from brax.training import acting
        acting.generate_unroll = _ORIG_UNROLL


# --------------------------------------------------------------------------
# Legacy helpers retained for goal extraction / per-episode goal sampling.
# --------------------------------------------------------------------------
class HERGoal:
    """A 4D contact-acquisition goal: [distance, bearing, front, rod_dist]."""
    DIM = GOAL_DIM

    @staticmethod
    def goal_reward(achieved, goal, sigma: float = 0.15):
        return goal_reward(achieved, goal, sigma)


def sample_goal(rng):
    """Sample a random goal in the contact-acquisition space."""
    k1, k2, k3, k4 = jax.random.split(rng, 4)
    dist = jax.random.uniform(k1, (), minval=0.1, maxval=1.0)
    bearing = jax.random.uniform(k2, (), minval=-jnp.pi, maxval=jnp.pi)
    front = jax.random.uniform(k3, (), minval=-1.0, maxval=1.0)
    rod_dist = jax.random.uniform(k4, (), minval=0.05, maxval=0.5)
    return jnp.array([dist, bearing, front, rod_dist])


class HERReplayBuffer:
    """Episode buffer with hindsight relabeling (host-side; for offline analysis).

    Retained for completeness; the PPO training path uses :func:`install_her_relabel`
    (on-policy rollout relabeling), not this buffer.
    """

    def __init__(self, capacity: int = 100000, her_fraction: float = 0.4):
        self.capacity = capacity
        self.her_fraction = her_fraction
        self.buffer = deque(maxlen=capacity)
        self._episode = []

    def add(self, obs, action, reward, next_obs, done, goal, achieved):
        self._episode.append(dict(obs=obs, action=action, reward=reward,
                                  next_obs=next_obs, done=done, goal=goal, achieved=achieved))
        if done:
            self._finish_episode()

    def _finish_episode(self):
        if not self._episode:
            return
        for tr in self._episode:
            self.buffer.append(tr)
        n_her = int(len(self._episode) * self.her_fraction)
        for idx in random.sample(range(len(self._episode)), min(n_her, len(self._episode))):
            future = random.randint(idx, len(self._episode) - 1)
            new_goal = self._episode[future]["achieved"]
            tr = self._episode[idx]
            self.buffer.append(dict(tr, goal=new_goal,
                                    reward=float(goal_reward(tr["achieved"], new_goal))))
        self._episode = []

    def __len__(self):
        return len(self.buffer)
