# SPDX-License-Identifier: MIT
"""Random Network Distillation (RND) intrinsic motivation — TRUE RND.

A fixed random TARGET network f(s) maps a feature vector to a random embedding.
A trained PREDICTOR network g(s) learns to match f(s).  The novelty bonus is the
prediction error ||f(s) - g(s)||^2: high for unfamiliar states, and it DECREASES
as the predictor is trained on a state (the defining property of RND).

This module exposes two interfaces over the SAME networks:

1. :class:`RNDPredictor` — a stateful, host-side trainer used by standalone
   tests and offline batches.  ``update()`` runs a real Adam gradient step on the
   predictor; repeated updates on a state drive its novelty down.

2. :func:`make_rnd` — a PURE FUNCTIONAL interface for wiring RND into a JAX env
   step.  The fixed target params live in a closure (shared, constant); the
   predictor params + optimizer state are returned to the caller to be carried
   in ``state.info`` (per-env) and threaded through ``novelty``/``update``.  This
   is how :class:`AdversarialEnv` trains the predictor INSIDE the rollout, so the
   curiosity bonus genuinely adapts to what the policy has visited.

Both interfaces share :class:`RandomTargetNetwork` / :class:`PredictorNetwork`
so behaviour is identical.
"""

from __future__ import annotations

from typing import NamedTuple, Callable

import jax
import jax.numpy as jnp
from flax import linen as nn
import optax


class RandomTargetNetwork(nn.Module):
    """Fixed random network mapping features to a random target embedding."""
    hidden_dim: int = 256
    output_dim: int = 128

    @nn.compact
    def __call__(self, x):
        x = nn.Dense(self.hidden_dim)(x)
        x = nn.relu(x)
        x = nn.Dense(self.output_dim)(x)
        return x


class PredictorNetwork(nn.Module):
    """Trained network that learns to predict the random target's output."""
    hidden_dim: int = 256
    output_dim: int = 128

    @nn.compact
    def __call__(self, x):
        x = nn.Dense(self.hidden_dim)(x)
        x = nn.relu(x)
        x = nn.Dense(self.hidden_dim)(x)
        x = nn.relu(x)
        x = nn.Dense(self.output_dim)(x)
        return x


def _as_batch(feat):
    """(d,) -> (1,d), keep (n,d). Returns (batched, was_single)."""
    if feat.ndim == 1:
        return feat[None], True
    return feat, False


class RND(NamedTuple):
    """Functional RND handle for env integration.

    ``target_params`` is fixed/shared; ``init_predictor_params`` and
    ``init_opt_state`` seed the per-env carried state.  ``novelty`` and
    ``update`` are pure functions of (predictor_params[, opt_state], feat).
    """
    feature_dim: int
    target_params: dict
    init_predictor_params: dict
    init_opt_state: optax.OptState
    novelty: Callable
    update: Callable


def make_rnd(feature_dim: int, hidden_dim: int = 128, output_dim: int = 64,
             lr: float = 1e-3, key=None) -> RND:
    """Build a functional RND for carrying predictor state in an env's info.

    Returns an :class:`RND` whose ``novelty(predictor_params, feat) -> scalar``
    and ``update(predictor_params, opt_state, feat) -> (params, opt_state, loss)``
    are pure (jit/vmap-safe), so an env can store ``init_predictor_params`` /
    ``init_opt_state`` per env and advance them every step.
    """
    if key is None:
        key = jax.random.PRNGKey(0)
    target = RandomTargetNetwork(hidden_dim=hidden_dim, output_dim=output_dim)
    predictor = PredictorNetwork(hidden_dim=hidden_dim, output_dim=output_dim)
    k1, k2 = jax.random.split(key)
    dummy = jnp.zeros((1, feature_dim))
    target_params = target.init(k1, dummy)
    predictor_params = predictor.init(k2, dummy)
    optimizer = optax.adam(lr)
    opt_state = optimizer.init(predictor_params)

    def novelty(predictor_params, feat):
        f, single = _as_batch(feat)
        t = target.apply(target_params, f)
        p = predictor.apply(predictor_params, f)
        nov = jnp.mean((t - p) ** 2, axis=-1)
        return nov[0] if single else nov

    def update(predictor_params, opt_state, feat):
        f, _ = _as_batch(feat)
        t = jax.lax.stop_gradient(target.apply(target_params, f))

        def loss_fn(pp):
            p = predictor.apply(pp, f)
            return jnp.mean((p - t) ** 2)

        loss, grads = jax.value_and_grad(loss_fn)(predictor_params)
        updates, opt_state = optimizer.update(grads, opt_state, predictor_params)
        predictor_params = optax.apply_updates(predictor_params, updates)
        return predictor_params, opt_state, loss

    return RND(feature_dim=feature_dim, target_params=target_params,
               init_predictor_params=predictor_params, init_opt_state=opt_state,
               novelty=novelty, update=update)


class RNDPredictor:
    """Stateful RND trainer (host-side) — used by standalone tests/offline batches.

    The novelty bonus is the MSE between the fixed target and the trained
    predictor.  ``raw_novelty`` is the unnormalized error (monotone in
    familiarity — use it for assertions); ``novelty`` divides by a running std
    for stable reward scaling.
    """

    def __init__(self, obs_dim: int, hidden_dim: int = 256, output_dim: int = 128,
                 lr: float = 1e-3, obs_start: int = 0, obs_end: int | None = None,
                 seed: int = 0):
        self.obs_dim = obs_dim
        self.obs_start = obs_start
        self.obs_end = obs_end or obs_dim
        self.feature_dim = self.obs_end - self.obs_start

        self._target = RandomTargetNetwork(hidden_dim=hidden_dim, output_dim=output_dim)
        self._predictor = PredictorNetwork(hidden_dim=hidden_dim, output_dim=output_dim)

        key = jax.random.PRNGKey(seed)
        key1, key2 = jax.random.split(key)
        dummy = jnp.zeros((1, self.feature_dim))
        self._target_params = self._target.init(key1, dummy)
        self._predictor_params = self._predictor.init(key2, dummy)

        self._optimizer = optax.adam(lr)
        self._opt_state = self._optimizer.init(self._predictor_params)

        self._rnd_var = jnp.ones(())
        self._rnd_count = 1e-4

        # jit the hot paths
        self._jit_raw = jax.jit(self._raw_novelty_impl)
        self._jit_update = jax.jit(self._update_impl)

    def _extract_features(self, obs):
        if isinstance(obs, dict):
            obs = obs["state"]
        if self.obs_end < obs.shape[-1]:
            return obs[..., self.obs_start:self.obs_end]
        return obs

    def _raw_novelty_impl(self, target_params, predictor_params, feat):
        t = self._target.apply(target_params, feat)
        p = self._predictor.apply(predictor_params, feat)
        return jnp.mean((t - p) ** 2, axis=-1)

    def raw_novelty(self, obs):
        """Unnormalized prediction error (decreases as the predictor learns)."""
        feat = self._extract_features(obs)
        single = feat.ndim == 1
        if single:
            feat = feat[None]
        nov = self._jit_raw(self._target_params, self._predictor_params, feat)
        return nov[0] if single else nov

    def novelty(self, obs):
        """Std-normalized novelty (for reward scaling)."""
        raw = self.raw_novelty(obs)
        return raw / (jnp.sqrt(self._rnd_var) + 1e-8)

    def _update_impl(self, predictor_params, opt_state, target_params, feat):
        t = jax.lax.stop_gradient(self._target.apply(target_params, feat))

        def loss_fn(pp):
            p = self._predictor.apply(pp, feat)
            return jnp.mean((p - t) ** 2)

        loss, grads = jax.value_and_grad(loss_fn)(predictor_params)
        updates, opt_state = self._optimizer.update(grads, opt_state, predictor_params)
        predictor_params = optax.apply_updates(predictor_params, updates)
        return predictor_params, opt_state, loss

    def update(self, obs_batch):
        """One Adam step of the predictor toward the target on this batch."""
        feat = self._extract_features(obs_batch)
        if feat.ndim == 1:
            feat = feat[None]
        # running variance of the raw novelty (for reward normalization)
        raw = self._jit_raw(self._target_params, self._predictor_params, feat)
        self._rnd_var = 0.99 * self._rnd_var + 0.01 * jnp.maximum(jnp.var(raw), 1e-6)
        self._predictor_params, self._opt_state, loss = self._jit_update(
            self._predictor_params, self._opt_state, self._target_params, feat)
        return float(loss)

    def get_state(self):
        return {"predictor_params": self._predictor_params, "opt_state": self._opt_state,
                "rnd_var": self._rnd_var, "rnd_count": self._rnd_count}

    def set_state(self, state):
        self._predictor_params = state["predictor_params"]
        self._opt_state = state["opt_state"]
        self._rnd_var = state["rnd_var"]
        self._rnd_count = state["rnd_count"]
