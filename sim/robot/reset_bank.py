# SPDX-License-Identifier: MIT
"""B.4: banked auto-reset — per-episode reset DIVERSITY that brax silently dropped.

brax's AutoResetWrapper caches `first_pipeline_state`/`first_obs` from the initial
reset and replays that SAME state at every episode boundary, forever. Every
per-episode randomization this project built (spawn separation curriculum, joint
noise, HER goal draws) therefore mostly never ran inside a training run (audit
item 6; test_mechanism_liveness documents the defect against stock brax).

This module provides the sanctioned replacement via brax ppo.train's unused
`wrap_env_fn` hook:

  * a K-entry bank of GENUINELY DISTINCT reset states, built once at wrap time
    by K real env.reset() calls (each with its own key -> its own spawn/noise/
    curriculum draw);
  * ~(1-canonical_frac) of entries are "launch" states: root given a random
    planar velocity U(launch_speed) at a random heading, then integrated one
    zero-action step so qvel/obs are consistent (RSI-lite: the value function
    finally sees moving states near reset, where the stepping gradient lives —
    DeepMimic's reference-state-initialization insight);
  * on done, each env advances its own bank cursor by a stride COPRIME to K, so
    envs orbit the whole bank instead of replaying one state.

Learning state carried in info (per-env RND predictor + Adam, HER goals) is NOT
touched at swap — mirroring stock brax semantics, which is what keeps in-info
learners alive across episode boundaries (the audit's bank-swap caveat).
info_keys.py (V.5) is the registry of which keys DEMAND that survival
(persistent) vs which are episode-scoped; test_info_keys.py enforces that every
key has a declared lifetime.
"""
from __future__ import annotations

import functools

import jax
import jax.numpy as jnp
from brax.envs.wrappers import training as tw

from constants import LAUNCH_SPEED  # V.1: (0.1, 0.5) m/s; DR.5 would certify this range


class BankedAutoResetWrapper(tw.Wrapper):
    def __init__(self, env, bank_key, bank_size=256, canonical_frac=0.3,
                 launch_speed=LAUNCH_SPEED, root_dof=None):
        super().__init__(env)
        self._K = int(bank_size)
        keys = jax.random.split(bank_key, self._K)
        bank = env.reset(keys)                       # vmapped: K distinct reset draws
        n_launch = int(self._K * (1.0 - canonical_frac))
        if n_launch > 0 and root_dof is not None:
            lk1, lk2 = jax.random.split(jax.random.fold_in(bank_key, 1))
            spd = jax.random.uniform(lk1, (self._K,), minval=launch_speed[0],
                                     maxval=launch_speed[1])
            ang = jax.random.uniform(lk2, (self._K,), minval=-jnp.pi, maxval=jnp.pi)
            mask = (jnp.arange(self._K) < n_launch).astype(jnp.float32)
            vx, vy = spd * jnp.cos(ang) * mask, spd * jnp.sin(ang) * mask
            qvel = bank.pipeline_state.qvel
            qvel = qvel.at[:, root_dof].set(vx).at[:, root_dof + 1].set(vy)
            bank = bank.replace(pipeline_state=bank.pipeline_state.replace(qvel=qvel))
            # one zero-action step so physics + obs are consistent with the injected
            # velocity (a mid-motion state, not a doctored still frame)
            bank = env.step(bank, jnp.zeros((self._K,) + self._act_shape(env)))
        self._bank_ps = bank.pipeline_state
        self._bank_obs = bank.obs
        # stride coprime to K -> every env's cursor orbits the full bank
        self._stride = 17 if self._K % 17 else 13

    @staticmethod
    def _act_shape(env):
        a = env.action_size
        return (a,) if isinstance(a, int) else tuple(a)

    def reset(self, rng):
        state = self.env.reset(rng)
        n = state.done.shape[0] if state.done.ndim else 1
        idx = jax.random.randint(jax.random.fold_in(jax.random.PRNGKey(0), 0), (n,), 0, self._K)
        # per-env starting cursor derived from the reset keys so envs start spread out
        idx = jax.vmap(lambda k: jax.random.randint(k, (), 0, self._K))(rng) \
            if rng.ndim == 2 else idx
        state.info["bank_idx"] = idx
        return state

    def step(self, state, action):
        # mirror brax AutoResetWrapper's done/truncation dance, minus the replay
        if "steps" in state.info:
            steps = state.info["steps"]
            steps = jnp.where(state.done, jnp.zeros_like(steps), steps)
            state.info.update(steps=steps)
        state = state.replace(done=jnp.zeros_like(state.done))
        state = self.env.step(state, action)
        idx = (state.info["bank_idx"] + self._stride) % self._K

        def where_done(x, y):
            done = state.done
            if done.ndim > 0:
                done = jnp.reshape(done, [x.shape[0]] + [1] * (len(x.shape) - 1))
            return jnp.where(done, x, y)

        pipeline_state = jax.tree.map(
            where_done, jax.tree.map(lambda b: b[idx], self._bank_ps), state.pipeline_state)
        obs = jax.tree.map(where_done, jax.tree.map(lambda b: b[idx], self._bank_obs), state.obs)
        state.info["bank_idx"] = jnp.where(state.done > 0, idx, state.info["bank_idx"])
        return state.replace(pipeline_state=pipeline_state, obs=obs)


def make_wrap_fn(bank_key, bank_size=256, canonical_frac=0.3, launch_speed=LAUNCH_SPEED,
                 root_dof=None):
    """Drop-in for brax ppo.train's `wrap_env_fn` (replaces training.wrap)."""
    def wrap_for_training(env, episode_length=1000, action_repeat=1, randomization_fn=None):
        env = tw.EpisodeWrapper(env, episode_length, action_repeat)
        if randomization_fn is None:
            env = tw.VmapWrapper(env)
        else:
            env = tw.DomainRandomizationVmapWrapper(env, randomization_fn)
        return BankedAutoResetWrapper(env, bank_key, bank_size=bank_size,
                                      canonical_frac=canonical_frac,
                                      launch_speed=launch_speed, root_dof=root_dof)
    return functools.partial(wrap_for_training)
