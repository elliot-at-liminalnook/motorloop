# SPDX-License-Identifier: MIT
"""MJX (GPU) env for the parametric body — JAX-functional, brax-PPO-compatible.

Drives `mjx` directly (not brax's MJCF loader) so the full generated MJCF is
supported. Reward starts as a stand+forward locomotion baseline (Phase 1); the
SPARC self-play reward arrives with the match env (Phase 4). Obs optionally carries
the design vector (Phase 2 universal policy) when `design` is passed.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
import mujoco
from mujoco import mjx
from brax.envs.base import Env, State


class CodesignEnv(Env):
    def __init__(self, xml: str, frame_skip: int = 5, design: jnp.ndarray | None = None):
        m = mujoco.MjModel.from_xml_string(xml)
        self._mx = mjx.put_model(m)
        self._nu = int(m.nu)
        self._q0 = jnp.array(m.qpos0)
        self._fs = frame_skip
        self._design = None if design is None else jnp.asarray(design)
        d = 0 if self._design is None else int(self._design.shape[0])
        self._obs_size = 2 * self._nu + 11 + d

    @property
    def observation_size(self):
        return self._obs_size

    @property
    def action_size(self):
        return self._nu

    @property
    def backend(self):
        return "mjx"

    def _obs(self, dx):
        base = jnp.concatenate([dx.qpos[7:7 + self._nu], dx.qvel[6:6 + self._nu],
                                dx.qpos[3:7], dx.qvel[0:6], dx.qpos[2:3]])
        if self._design is not None:
            base = jnp.concatenate([base, self._design])
        return base

    def reset(self, rng):
        noise = jax.random.uniform(rng, (self._nu,), minval=-0.05, maxval=0.05)
        qpos = self._q0.at[7:7 + self._nu].add(noise)
        dx = mjx.make_data(self._mx).replace(qpos=qpos)
        dx = mjx.forward(self._mx, dx)
        return State(dx, self._obs(dx), jnp.zeros(()), jnp.zeros(()), {}, {})

    def step(self, state, action):
        action = jnp.clip(action, -1.0, 1.0)
        dx = state.pipeline_state.replace(ctrl=action)
        dx = jax.lax.fori_loop(0, self._fs, lambda i, d: mjx.step(self._mx, d), dx)
        up = 1.0 - 2.0 * (dx.qpos[4] ** 2 + dx.qpos[5] ** 2)     # torso upright
        fwd = dx.qvel[0]                                          # forward progress
        reward = 1.0 + up + fwd - 0.001 * jnp.sum(action ** 2)
        done = jnp.where(dx.qpos[2] < 0.18, 1.0, 0.0)            # fell
        return state.replace(pipeline_state=dx, obs=self._obs(dx),
                             reward=reward, done=done)


# design space (normalized [0,1]^3): mass scale, joint stiffness, damping scale.
DESIGN_DIM = 3


def apply_design(mx, d):
    """Perturb mjx model fields by a normalized design vector (per-env under vmap)."""
    mass_s = 0.6 + 0.8 * d[0]
    stiff = 25.0 * d[1]
    damp_s = 0.5 + 1.5 * d[2]
    return mx.replace(
        body_mass=mx.body_mass * mass_s,
        body_inertia=mx.body_inertia * mass_s,
        jnt_stiffness=mx.jnt_stiffness.at[1:].set(stiff),
        dof_damping=mx.dof_damping * damp_s)


class UniversalEnv(Env):
    """Design-conditioned env: a random body each episode (DR via per-env reset rng),
    the design vector in the obs -> brax PPO learns ONE policy over the design range.
    `fixed_design` pins the body for evaluation (the cheap design-fitness rollout)."""

    def __init__(self, xml: str, frame_skip: int = 5, fixed_design=None):
        m = mujoco.MjModel.from_xml_string(xml)
        self._mx = mjx.put_model(m)
        self._nu = int(m.nu)
        self._q0 = jnp.array(m.qpos0)
        self._fs = frame_skip
        self._fixed = None if fixed_design is None else jnp.asarray(fixed_design)
        self._obs_size = 2 * self._nu + 11 + DESIGN_DIM

    @property
    def observation_size(self): return self._obs_size
    @property
    def action_size(self): return self._nu
    @property
    def backend(self): return "mjx"

    def _obs(self, dx, design):
        return jnp.concatenate([dx.qpos[7:7 + self._nu], dx.qvel[6:6 + self._nu],
                                dx.qpos[3:7], dx.qvel[0:6], dx.qpos[2:3], design])

    def reset(self, rng):
        rng, dr, nr = jax.random.split(rng, 3)
        design = self._fixed if self._fixed is not None else jax.random.uniform(dr, (DESIGN_DIM,))
        qpos = self._q0.at[7:7 + self._nu].add(jax.random.uniform(nr, (self._nu,), minval=-0.05, maxval=0.05))
        dx = mjx.forward(apply_design(self._mx, design), mjx.make_data(self._mx).replace(qpos=qpos))
        return State(dx, self._obs(dx, design), jnp.zeros(()), jnp.zeros(()), {}, {"design": design})

    def reset_with(self, rng, design):
        """Reset with a GIVEN design (eval) — shares the model, no rebuild/recompile."""
        nr, _ = jax.random.split(rng)
        qpos = self._q0.at[7:7 + self._nu].add(jax.random.uniform(nr, (self._nu,), minval=-0.05, maxval=0.05))
        dx = mjx.forward(apply_design(self._mx, design), mjx.make_data(self._mx).replace(qpos=qpos))
        return State(dx, self._obs(dx, design), jnp.zeros(()), jnp.zeros(()), {}, {"design": design})

    def step(self, state, action):
        design = state.info["design"]
        mxd = apply_design(self._mx, design)
        action = jnp.clip(action, -1.0, 1.0)
        dx = state.pipeline_state.replace(ctrl=action)
        dx = jax.lax.fori_loop(0, self._fs, lambda i, d: mjx.step(mxd, d), dx)
        up = 1.0 - 2.0 * (dx.qpos[4] ** 2 + dx.qpos[5] ** 2)
        reward = 1.0 + up + dx.qvel[0] - 0.001 * jnp.sum(action ** 2)
        done = jnp.where(dx.qpos[2] < 0.18, 1.0, 0.0)
        return state.replace(pipeline_state=dx, obs=self._obs(dx, design),
                             reward=reward, done=done)
