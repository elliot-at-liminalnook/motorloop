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
# The maps live in design_codec (the single design codec); apply_design delegates so
# the universal policy, the match env, and CEM all mean the same thing by "design".
from design_codec import DESIGN_DIM, apply_fast as apply_design  # noqa: E402,F401


class UniversalEnv(Env):
    """Design-conditioned env: a random body each episode (DR via per-env reset rng),
    the design vector in the obs -> brax PPO learns ONE policy over the design range.
    `fixed_design` pins the body for evaluation (the cheap design-fitness rollout).

    `reality_gap=True` (checklist R1/R4) draws a calibrated sim "world" per episode from
    `reality_gap.sample_domain_params` (a pre-sampled bank of `n_worlds`), perturbs the
    model (`apply_to_mjx_model`: mass/inertia/damping/friction/contact-softness), and
    scales each command by the speed-dependent torque envelope (`actuator_scale`:
    back-EMF droop + current limit + voltage sag + thermal + gear efficiency) so the
    policy trains against the real motor envelope, not an ideal torque source. The world
    is NOT in the obs (the policy must be robust to it; RS4/RMA later infers it). Default
    OFF — flipping it on is R7 ("re-derive under the calibrated sim"). Latency buffering +
    sensor noise are the next R1/R4 increment (not yet wired)."""

    def __init__(self, xml: str, frame_skip: int = 5, fixed_design=None,
                 reality_gap: bool = False, n_worlds: int = 64):
        m = mujoco.MjModel.from_xml_string(xml)
        self._mx = mjx.put_model(m)
        self._nu = int(m.nu)
        self._q0 = jnp.array(m.qpos0)
        self._fs = frame_skip
        self._fixed = None if fixed_design is None else jnp.asarray(fixed_design)
        self._obs_size = 2 * self._nu + 11 + DESIGN_DIM
        self._rg = bool(reality_gap)
        if self._rg:
            import sys as _sys
            from pathlib import Path as _P
            _sys.path.insert(0, str(_P(__file__).resolve().parent))
            from reality_gap import (sample_domain_params, default_uncertainty,
                                     actuator_scale, apply_to_mjx_model)
            unc = default_uncertainty()
            dps = [sample_domain_params(i, unc) for i in range(n_worlds)]   # numpy source of truth
            self._bank = {k: jnp.asarray([float(dp[k]) for dp in dps])      # stacked -> JAX gather
                          for k in dps[0] if isinstance(dps[0][k], (int, float))}
            self._act_scale = staticmethod(actuator_scale).__func__
            self._apply_dp = staticmethod(apply_to_mjx_model).__func__
            self._nworlds = n_worlds

    @property
    def observation_size(self): return self._obs_size
    @property
    def action_size(self): return self._nu
    @property
    def backend(self): return "mjx"

    def _obs(self, dx, design):
        return jnp.concatenate([dx.qpos[7:7 + self._nu], dx.qvel[6:6 + self._nu],
                                dx.qpos[3:7], dx.qvel[0:6], dx.qpos[2:3], design])

    def _world(self, rng):
        k = jax.random.randint(rng, (), 0, self._nworlds)
        return {f: self._bank[f][k] for f in self._bank}

    def _model(self, design, dp):
        mxd = apply_design(self._mx, design)
        if self._rg:                                  # DR on top of the design (no stiffness clash)
            mxd = self._apply_dp(mxd, dp, hinge_mask=None)
        return mxd

    def _ctrl(self, action, qvel, dp):
        action = jnp.clip(action, -1.0, 1.0)
        if self._rg:                                  # back-EMF torque-speed envelope per joint
            action = action * self._act_scale(qvel[6:6 + self._nu], dp)
        return action

    def _make(self, rng, design):
        nr, wr = jax.random.split(rng)
        dp = self._world(wr) if self._rg else {}
        qpos = self._q0.at[7:7 + self._nu].add(
            jax.random.uniform(nr, (self._nu,), minval=-0.05, maxval=0.05))
        dx = mjx.forward(self._model(design, dp), mjx.make_data(self._mx).replace(qpos=qpos))
        info = {"design": design, "dp": dp} if self._rg else {"design": design}
        return State(dx, self._obs(dx, design), jnp.zeros(()), jnp.zeros(()), {}, info)

    def reset(self, rng):
        rng, dr, mr = jax.random.split(rng, 3)
        design = self._fixed if self._fixed is not None else jax.random.uniform(dr, (DESIGN_DIM,))
        return self._make(mr, design)

    def reset_with(self, rng, design):
        """Reset with a GIVEN design (eval) — shares the model, no rebuild/recompile."""
        return self._make(rng, jnp.asarray(design))

    def step(self, state, action):
        design = state.info["design"]
        dp = state.info["dp"] if self._rg else {}
        mxd = self._model(design, dp)
        action = self._ctrl(action, state.pipeline_state.qvel, dp)
        dx = state.pipeline_state.replace(ctrl=action)
        dx = jax.lax.fori_loop(0, self._fs, lambda i, d: mjx.step(mxd, d), dx)
        up = 1.0 - 2.0 * (dx.qpos[4] ** 2 + dx.qpos[5] ** 2)
        reward = 1.0 + up + dx.qvel[0] - 0.001 * jnp.sum(action ** 2)
        done = jnp.where(dx.qpos[2] < 0.18, 1.0, 0.0)
        return state.replace(pipeline_state=dx, obs=self._obs(dx, design),
                             reward=reward, done=done)
