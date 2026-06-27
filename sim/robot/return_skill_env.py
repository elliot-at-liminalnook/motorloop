# SPDX-License-Identifier: MIT
"""Local return-to-origin skill environment.

This isolates the remaining route problem after waypoint 2:

    start near active WP3 / return handoff -> reach origin without falling

The env wraps ``CommandedEnv`` so it reuses the same robot, CPG/PD scaffold,
contacts, and fall checks, but exposes a low-dimensional action interface:

    [x_correction, y_correction, turn_bias, phase_speed, speed_strength]

The policy steers the CPG command manifold instead of cloning a full 12-D motor
residual schedule.  This is intentionally a local skill; full-route replay must
gate it in later.
"""

from __future__ import annotations

import math
import os
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

from commanded_env import FALL_Z, MIN_UP_Z, VMAX, _build as _build_commanded  # noqa: E402

ACTION_DIM = 5
OBS_DIM = 16
DEFAULT_STARTS = (
    (0.05442, 0.38643),
    (0.06719, 0.38646),
    (0.04125, 0.36983),
    (0.04066, 0.35510),
    (0.02561, 0.33790),
    (0.00388, 0.32333),
    (0.01506, 0.32812),
    (-0.00865, 0.31477),
)


def _bool_env(name: str, default: bool = False) -> bool:
    raw = os.environ.get(name)
    if raw is None:
        return default
    return raw.strip().lower() in ("1", "true", "yes", "on")


def parse_xy_list(text: str) -> np.ndarray:
    starts = []
    for chunk in text.split(";"):
        chunk = chunk.strip()
        if not chunk:
            continue
        parts = [p.strip() for p in chunk.split(",")]
        if len(parts) == 3:
            parts = parts[1:]
        if len(parts) != 2:
            raise ValueError("RETURN_SKILL_STARTS entries must be 'x,y' or 'waypoint,x,y'")
        starts.append((float(parts[0]), float(parts[1])))
    if not starts:
        starts = list(DEFAULT_STARTS)
    return np.asarray(starts, dtype=np.float32)


def _build():
    import jax
    import jax.numpy as jnp
    from brax.envs.base import Env, State

    BaseEnv = _build_commanded()

    class ReturnSkillEnv(Env):
        def __init__(self, xml=None, frame_skip=5):
            self._base = BaseEnv(xml=xml, frame_skip=frame_skip)
            self._target = jnp.asarray([0.0, 0.0], dtype=jnp.float32)
            self._starts = jnp.asarray(parse_xy_list(os.environ.get("RETURN_SKILL_STARTS", "")), dtype=jnp.float32)
            self._start_count = int(self._starts.shape[0])
            self._start_jitter = float(os.environ.get("RETURN_SKILL_START_JITTER", "0.015"))
            self._phase_jitter = float(os.environ.get("RETURN_SKILL_PHASE_JITTER", "0.03"))
            self._vel_jitter = float(os.environ.get("RETURN_SKILL_VEL_JITTER", "0.03"))
            self._radius = float(os.environ.get("RETURN_SKILL_RADIUS", "0.07"))
            self._gain = float(os.environ.get("RETURN_SKILL_GAIN", "2.0"))
            self._cmd_correction_scale = float(os.environ.get("RETURN_SKILL_CMD_CORRECTION_SCALE", "0.75"))
            self._turn_scale = float(os.environ.get("RETURN_SKILL_TURN_SCALE", "0.35"))
            self._phase_scale = float(os.environ.get("RETURN_SKILL_PHASE_SCALE", "0.35"))
            self._speed_min = float(os.environ.get("RETURN_SKILL_SPEED_MIN", "0.25"))
            self._progress_w = float(os.environ.get("RETURN_SKILL_PROGRESS_W", "18.0"))
            self._hit_w = float(os.environ.get("RETURN_SKILL_HIT_W", "4.0"))
            self._dist_w = float(os.environ.get("RETURN_SKILL_DIST_W", "1.5"))
            self._upright_w = float(os.environ.get("RETURN_SKILL_UPRIGHT_W", "0.2"))
            self._away_w = float(os.environ.get("RETURN_SKILL_AWAY_W", "8.0"))
            self._action_w = float(os.environ.get("RETURN_SKILL_ACTION_W", "0.04"))
            self._jerk_w = float(os.environ.get("RETURN_SKILL_JERK_W", "0.06"))
            self._terminate_on_hit = _bool_env("RETURN_SKILL_TERMINATE_ON_HIT", False)
            self._scale = float(max(float(np.max(np.linalg.norm(np.asarray(self._starts), axis=1))), VMAX, 1e-6))

        @property
        def observation_size(self):
            return OBS_DIM

        @property
        def action_size(self):
            return ACTION_DIM

        @property
        def backend(self):
            return self._base.backend

        def _base_cmd(self, pos):
            delta = self._target - pos
            cmd = self._gain * delta
            norm = jnp.linalg.norm(cmd)
            cmd = jnp.where(norm > VMAX, cmd * (VMAX / jnp.maximum(norm, 1e-6)), cmd)
            return cmd

        def _obs(self, state, cmd, prev_cmd, prev_dist, step_count):
            pos = state.pipeline_state.qpos[:2]
            delta = self._target - pos
            dist = jnp.linalg.norm(delta)
            vxy = state.pipeline_state.qvel[:2]
            phase = state.info["phase"]
            progress = prev_dist - dist
            prior_strength = state.info.get("prior_strength", jnp.asarray(1.0, dtype=jnp.float32))
            up = 1.0 - 2.0 * (state.pipeline_state.qpos[4] ** 2 + state.pipeline_state.qpos[5] ** 2)
            return jnp.concatenate([
                delta / self._scale,
                (dist / self._scale).reshape((1,)),
                vxy / jnp.maximum(VMAX, 1e-6),
                jnp.asarray([jnp.sin(phase), jnp.cos(phase)], dtype=jnp.float32),
                prev_cmd / jnp.maximum(VMAX, 1e-6),
                cmd / jnp.maximum(VMAX, 1e-6),
                (step_count.astype(jnp.float32) / 260.0).reshape((1,)),
                (progress / jnp.maximum(VMAX, 1e-6)).reshape((1,)),
                prior_strength.reshape((1,)),
                state.pipeline_state.qpos[2:3],
                up.reshape((1,)),
            ]).astype(jnp.float32)

        def _set_route_info(self, state, cmd, dist):
            info = {
                **state.info,
                "cmd": cmd,
                "remote": jnp.array(True),
                "route_wp": jnp.asarray(3, dtype=jnp.int32),
                "route_prev_dist": dist,
            }
            return state.replace(info=info, obs=state.obs.at[-2:].set(cmd))

        def reset(self, rng):
            rng, base_key, start_key, jitter_key, phase_key, vel_key = jax.random.split(rng, 6)
            state = self._base.reset_with_command(base_key, jnp.zeros(2, dtype=jnp.float32))
            idx = jax.random.randint(start_key, (), 0, self._start_count)
            pos = self._starts[idx]
            if self._start_jitter > 0.0:
                pos = pos + jax.random.uniform(
                    jitter_key, (2,), minval=-self._start_jitter, maxval=self._start_jitter
                )
            qpos = state.pipeline_state.qpos.at[:2].set(pos)
            qvel = state.pipeline_state.qvel
            if self._vel_jitter > 0.0:
                qvel = qvel.at[:2].add(
                    jax.random.uniform(vel_key, (2,), minval=-self._vel_jitter, maxval=self._vel_jitter)
                )
            pipe = state.pipeline_state.replace(qpos=qpos, qvel=qvel)
            state = state.replace(pipeline_state=pipe)
            if self._phase_jitter > 0.0:
                phase_delta = jax.random.uniform(
                    phase_key, (), minval=-self._phase_jitter, maxval=self._phase_jitter
                )
                state = state.replace(info={**state.info, "phase": state.info["phase"] + phase_delta})
            dist = jnp.linalg.norm(self._target - state.pipeline_state.qpos[:2])
            cmd = self._base_cmd(state.pipeline_state.qpos[:2])
            state = self._set_route_info(state, cmd, dist)
            info = {
                **state.info,
                "return_prev_dist": dist,
                "return_step": jnp.zeros((), dtype=jnp.int32),
                "return_prev_action": jnp.zeros(ACTION_DIM, dtype=jnp.float32),
                "return_prev_cmd": cmd,
            }
            metrics = {
                **state.metrics,
                "return_dist": dist,
                "return_progress": jnp.zeros((), dtype=jnp.float32),
                "return_hit": jnp.zeros((), dtype=jnp.float32),
                "return_cmd_x": cmd[0],
                "return_cmd_y": cmd[1],
                "return_strength": jnp.ones((), dtype=jnp.float32),
            }
            state = state.replace(info=info, metrics=metrics)
            return state.replace(obs=self._obs(state, cmd, cmd, dist, info["return_step"]),
                                 reward=jnp.zeros(()), done=jnp.zeros(()))

        def step(self, state: State, action):
            action = jnp.clip(action, -1.0, 1.0)
            pos0 = state.pipeline_state.qpos[:2]
            prev_dist = state.info["return_prev_dist"]
            prev_cmd = state.info["return_prev_cmd"]
            base_cmd = self._base_cmd(pos0)
            cmd = base_cmd + action[:2] * (VMAX * self._cmd_correction_scale)
            perp = jnp.asarray([-base_cmd[1], base_cmd[0]], dtype=jnp.float32)
            perp_norm = jnp.linalg.norm(perp)
            perp = jnp.where(perp_norm > 1e-6, perp / jnp.maximum(perp_norm, 1e-6), perp)
            cmd = cmd + action[2] * self._turn_scale * VMAX * perp
            strength = self._speed_min + (1.0 - self._speed_min) * (0.5 * (action[4] + 1.0))
            cmd = cmd * strength
            cmd_norm = jnp.linalg.norm(cmd)
            cmd = jnp.where(cmd_norm > VMAX, cmd * (VMAX / jnp.maximum(cmd_norm, 1e-6)), cmd)
            state = self._set_route_info(state, cmd, prev_dist)
            base_action = jnp.zeros(self._base.action_size, dtype=jnp.float32)
            next_state = self._base.step(state, base_action)
            if self._phase_scale != 0.0:
                next_state = next_state.replace(info={
                    **next_state.info,
                    "phase": next_state.info["phase"] + action[3] * self._phase_scale,
                })

            pos1 = next_state.pipeline_state.qpos[:2]
            dist1 = jnp.linalg.norm(self._target - pos1)
            progress = prev_dist - dist1
            away = jnp.maximum(0.0, -progress)
            up = 1.0 - 2.0 * (next_state.pipeline_state.qpos[4] ** 2 + next_state.pipeline_state.qpos[5] ** 2)
            hit = dist1 <= self._radius
            action_rate = jnp.sum((action - state.info["return_prev_action"]) ** 2)
            reward = (
                0.1
                + self._progress_w * progress
                - self._away_w * away
                - self._dist_w * dist1
                + self._hit_w * hit.astype(jnp.float32)
                + self._upright_w * up
                - self._action_w * jnp.sum(action ** 2)
                - self._jerk_w * action_rate
            )
            fall = (next_state.pipeline_state.qpos[2] < FALL_Z) | (up < MIN_UP_Z)
            done = fall | (hit & jnp.asarray(self._terminate_on_hit))
            step_count = state.info["return_step"] + jnp.asarray(1, dtype=jnp.int32)
            info = {
                **next_state.info,
                "return_prev_dist": dist1,
                "return_step": step_count,
                "return_prev_action": action,
                "return_prev_cmd": cmd,
            }
            metrics = {
                **next_state.metrics,
                "return_dist": dist1,
                "return_progress": progress,
                "return_hit": hit.astype(jnp.float32),
                "return_cmd_x": cmd[0],
                "return_cmd_y": cmd[1],
                "return_strength": strength,
            }
            next_state = next_state.replace(info=info, metrics=metrics)
            return next_state.replace(
                obs=self._obs(next_state, cmd, prev_cmd, dist1, step_count),
                reward=reward,
                done=done.astype(jnp.float32),
            )

    return ReturnSkillEnv
