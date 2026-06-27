# SPDX-License-Identifier: MIT
"""Route-aware MJX search for transition-compatible CPG priors.

The single-command CPG search can find cardinal gaits that work in isolation,
but those priors may not compose when the controller switches from forward to
left/back/right.  This script scores the actual checkpoint route while carrying
simulator state through the direction changes, then searches all four cardinal
raw CPG vectors jointly.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import pickle
import sys
from pathlib import Path

import jax
import jax.numpy as jnp
import mujoco
import numpy as np
from mujoco import mjx

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import cpg_teacher as cpg_teacher_mod  # noqa: E402
from cpg_teacher import (  # noqa: E402
    DEFAULT_RAW,
    DirectionalCPGParams,
    JOINT_NAMES,
    LEG_NAMES,
    PARAM_DIM,
    cpg_pd_step_target,
    decode_params,
    encode_params,
    params_to_dict,
    transition_phase_delta,
    transition_strength as cpg_transition_strength,
)
from design_codec import DESIGN_DIM, apply_fast  # noqa: E402
from gen_robot_mjcf import build_mjcf, load_spec  # noqa: E402
from search_cpg_gait_mjx import DEFAULT_FAST_DESIGN, load_raw, parse_design, resolve_stance, set_stance  # noqa: E402

OUT = Path(os.environ.get("CODESIGN_OUT", "/root/proj/out"))
OUT.mkdir(parents=True, exist_ok=True)

ROUTE_NAMES = ("forward", "backward", "left", "right")
ROUTE_INDEX = {name: i for i, name in enumerate(ROUTE_NAMES)}


def parse_waypoints(text: str) -> np.ndarray:
    pts = []
    for chunk in text.split(";"):
        chunk = chunk.strip()
        if not chunk:
            continue
        x, y = chunk.split(",")
        pts.append((float(x), float(y)))
    if not pts:
        raise ValueError("at least one waypoint is required")
    return np.asarray(pts, dtype=np.float32)


def infer_segments(waypoints: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    prev = np.vstack([np.zeros((1, 2), dtype=np.float32), waypoints[:-1]])
    delta = waypoints - prev
    lens = np.maximum(np.linalg.norm(delta, axis=1), 1e-6).astype(np.float32)
    dirs = (delta / lens[:, None]).astype(np.float32)
    sides = np.stack([-dirs[:, 1], dirs[:, 0]], axis=1).astype(np.float32)
    seg = []
    for dx, dy in delta:
        if abs(float(dx)) >= abs(float(dy)):
            seg.append(0 if dx >= 0.0 else 1)
        else:
            seg.append(2 if dy >= 0.0 else 3)
    return prev.astype(np.float32), dirs, sides, np.asarray(seg, dtype=np.int32)


def normalize_raw(raw: np.ndarray | None) -> np.ndarray:
    if raw is None:
        return np.asarray(DEFAULT_RAW, dtype=float)
    raw = np.asarray(raw, dtype=float)
    if raw.shape[0] != PARAM_DIM:
        raw = encode_params(decode_params(raw, xp=np))
    return raw


def route_seed(forward: np.ndarray | None, backward: np.ndarray | None,
               left: np.ndarray | None, right: np.ndarray | None) -> np.ndarray:
    return np.stack([
        normalize_raw(forward),
        normalize_raw(backward),
        normalize_raw(left),
        normalize_raw(right),
    ], axis=0)


def load_route(path: str | None) -> np.ndarray | None:
    if not path:
        return None
    d = json.loads(Path(path).read_text())
    if all(k in d for k in ("raw_forward", "raw_backward", "raw_left", "raw_right")):
        return route_seed(
            np.asarray(d["raw_forward"], dtype=float),
            np.asarray(d["raw_backward"], dtype=float),
            np.asarray(d["raw_left"], dtype=float),
            np.asarray(d["raw_right"], dtype=float),
        )
    if "raw_route" in d:
        raw = np.asarray(d["raw_route"], dtype=float)
        if raw.shape != (4, PARAM_DIM):
            raise ValueError(f"{path} raw_route expected shape {(4, PARAM_DIM)}, got {raw.shape}")
        return raw
    raise ValueError(f"{path} is not a route gait JSON")


def parse_mutate_route(text: str) -> np.ndarray:
    names = [x.strip().lower() for x in text.replace(";", ",").split(",") if x.strip()]
    if not names or "all" in names:
        return np.ones((4, PARAM_DIM), dtype=float)
    bad = [x for x in names if x not in ROUTE_INDEX]
    if bad:
        raise ValueError(f"unknown --mutate-route entries {bad}; valid={list(ROUTE_NAMES)} or all")
    mask = np.zeros((4, PARAM_DIM), dtype=float)
    for name in names:
        mask[ROUTE_INDEX[name], :] = 1.0
    return mask


def apply_freeze_route_freq(mask: np.ndarray, text: str) -> np.ndarray:
    names = [x.strip().lower() for x in text.replace(";", ",").split(",") if x.strip()]
    if not names:
        return mask
    out = mask.copy()
    if "all" in names:
        out[:, 0] = 0.0
        return out
    bad = [x for x in names if x not in ROUTE_INDEX]
    if bad:
        raise ValueError(f"unknown --freeze-route-freq entries {bad}; valid={list(ROUTE_NAMES)} or all")
    for name in names:
        out[ROUTE_INDEX[name], 0] = 0.0
    return out


def std_template(scale: float) -> np.ndarray:
    base = np.asarray([
        0.45, 1.00, 1.00, 1.00, 1.00,
        0.16, 0.24, 0.24,
        0.16, 0.24, 0.24,
        0.08, 0.12, 0.12,
        0.08, 0.12, 0.10, 0.08, 0.08,
        0.25, 0.25, 0.25, 0.25,
        0.16, 0.16, 0.16, 0.16,
        0.16, 0.16, 0.16, 0.16,
        0.12, 0.12, 0.12, 0.12,
    ], dtype=float)
    return np.tile(base[None, :], (4, 1)) * float(scale)


def clip_route(raw: np.ndarray) -> np.ndarray:
    out = raw.reshape(4, PARAM_DIM).copy()
    out[:, 0] = np.clip(out[:, 0], 0.4, 4.0)
    out[:, 1:5] = np.mod(out[:, 1:5], 2.0 * math.pi)
    return out.reshape(-1)


class MjxRouteEval:
    def __init__(
        self,
        waypoints: np.ndarray,
        radius: float,
        steps_per_waypoint: int,
        frame_skip: int,
        kp: float,
        kd: float,
        scale: float,
        stand_flex: float,
        stand_knee: float,
        spawn_height: float | None,
        fast_design: tuple[float, float, float] | None,
        max_saturation: float,
        vmax: float,
        gain: float,
        axis_deadband: float,
        command_mode: str,
        cross_w: float,
        backtrack_w: float,
        progress_w: float,
        final_dist_w: float,
        waypoint_bonus_w: float,
        focus_active: int,
        focus_progress_w: float,
        focus_dist_w: float,
        focus_cross_w: float,
    ):
        self.waypoints_np = np.asarray(waypoints, dtype=np.float32)
        prev, dirs, sides, segments = infer_segments(self.waypoints_np)
        self.waypoints = jnp.asarray(self.waypoints_np, dtype=jnp.float32)
        self.prev = jnp.asarray(prev, dtype=jnp.float32)
        self.dirs = jnp.asarray(dirs, dtype=jnp.float32)
        self.sides = jnp.asarray(sides, dtype=jnp.float32)
        self.segments = jnp.asarray(segments, dtype=jnp.int32)
        self.seg_lens = jnp.asarray(np.maximum(np.linalg.norm(self.waypoints_np - prev, axis=1), 1e-6), dtype=jnp.float32)
        self.radius = float(radius)
        self.nwp = int(self.waypoints_np.shape[0])
        self.steps = int(steps_per_waypoint) * self.nwp
        self.frame_skip = int(frame_skip)
        self.kp = float(kp)
        self.kd = float(kd)
        self.scale = float(scale)
        self.max_saturation = float(max_saturation)
        self.vmax = float(vmax)
        self.gain = float(gain)
        self.axis_deadband = float(axis_deadband)
        self.command_mode = str(command_mode)
        self.cross_w = float(cross_w)
        self.backtrack_w = float(backtrack_w)
        self.progress_w = float(progress_w)
        self.final_dist_w = float(final_dist_w)
        self.waypoint_bonus_w = float(waypoint_bonus_w)
        self.focus_active = int(focus_active)
        self.focus_progress_w = float(focus_progress_w)
        self.focus_dist_w = float(focus_dist_w)
        self.focus_cross_w = float(focus_cross_w)

        model = mujoco.MjModel.from_xml_string(build_mjcf(load_spec(HERE / "robot.toml")))
        self.mx = mjx.put_model(model)
        if fast_design is not None:
            self.mx = apply_fast(self.mx, jnp.asarray(fast_design, dtype=jnp.float32))
        self.design = jnp.asarray(
            fast_design if fast_design is not None else tuple(0.0 for _ in range(DESIGN_DIM)),
            dtype=jnp.float32,
        )
        self.q0 = jnp.asarray(set_stance(model, model.qpos0, stand_flex, stand_knee, spawn_height), dtype=jnp.float32)
        self.nu = int(model.nu)
        act_joints = [int(model.actuator_trnid[a, 0]) for a in range(model.nu)]
        self.qadr = jnp.asarray([int(model.jnt_qposadr[j]) for j in act_joints], dtype=jnp.int32)
        self.dadr = jnp.asarray([int(model.jnt_dofadr[j]) for j in act_joints], dtype=jnp.int32)
        self.jrange = jnp.asarray([model.jnt_range[j] for j in act_joints], dtype=jnp.float32)
        self.tmax = jnp.asarray(model.actuator_forcerange[: model.nu, 1], dtype=jnp.float32)
        self.stand = self.q0[self.qadr]
        self.ctrl_dt = self.frame_skip * float(model.opt.timestep)
        name_to_action = {}
        for a, j in enumerate(act_joints):
            name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_JOINT, j) or ""
            name_to_action[name] = a
        rows = []
        for leg in LEG_NAMES:
            rows.append([name_to_action[f"{leg}_{jn}"] for jn in JOINT_NAMES])
        self.cpg_idx = jnp.asarray(rows, dtype=jnp.int32)
        self.eval_one_stats = jax.jit(self.rollout_stats)
        self.eval_one_stats_with_controls = jax.jit(self.rollout_stats_with_controls)
        self.eval_one_trace = jax.jit(self.rollout_trace)
        self.eval_one_dataset = jax.jit(self.rollout_dataset)
        self.eval_command_dataset = jax.jit(self.rollout_command_dataset)

    def obs(self, dx, cmd):
        return jnp.concatenate([
            dx.qpos[7:7 + self.nu],
            dx.qvel[6:6 + self.nu],
            dx.qpos[3:7],
            dx.qvel[0:6],
            dx.qpos[2:3],
            self.design,
            cmd,
        ])

    def command_for_active_segment(self, pos, active):
        target = self.waypoints[active]
        delta = target - pos
        if self.command_mode == "full":
            cmd = self.gain * delta
            norm = jnp.linalg.norm(cmd)
            cmd = jnp.where(norm > self.vmax, cmd * (self.vmax / jnp.maximum(norm, 1e-6)), cmd)
            axis_is_x = jnp.abs(cmd[0]) >= jnp.abs(cmd[1])
            cmd_val = jnp.where(axis_is_x, cmd[0], cmd[1])
            scale = jnp.linalg.norm(cmd) / self.vmax
            x_seg = jnp.where(cmd[0] >= 0.0, 0, 1)
            y_seg = jnp.where(cmd[1] >= 0.0, 2, 3)
            segment = jnp.where(axis_is_x, x_seg, y_seg)
            segment = jnp.where(scale > 1e-6, segment, self.segments[active])
            return cmd, segment.astype(jnp.int32), scale

        direction = self.dirs[active]
        if self.command_mode == "axis-priority":
            axis_is_x = jnp.abs(delta[0]) >= jnp.abs(delta[1])
        else:
            axis_is_x = jnp.abs(direction[0]) >= jnp.abs(direction[1])
        remain = jnp.where(axis_is_x, target[0] - pos[0], target[1] - pos[1])
        cmd_val = jnp.where(jnp.abs(remain) > self.axis_deadband, self.gain * remain, 0.0)
        cmd_val = jnp.clip(cmd_val, -self.vmax, self.vmax)
        cmd = jnp.where(axis_is_x, jnp.stack([cmd_val, jnp.asarray(0.0)]), jnp.stack([jnp.asarray(0.0), cmd_val]))
        scale = jnp.linalg.norm(cmd) / self.vmax
        x_seg = jnp.where(cmd_val >= 0.0, 0, 1)
        y_seg = jnp.where(cmd_val >= 0.0, 2, 3)
        segment = jnp.where(axis_is_x, x_seg, y_seg)
        segment = jnp.where(scale > 1e-6, segment, self.segments[active])
        return cmd, segment.astype(jnp.int32), scale

    def directional_params(self, z_route):
        return DirectionalCPGParams(
            backward=decode_params(z_route[1], xp=jnp),
            forward=decode_params(z_route[0], xp=jnp),
            right=decode_params(z_route[3], xp=jnp),
            left=decode_params(z_route[2], xp=jnp),
        )

    def step_once(self, dx, z_route, phase, cmd, prev_cmd=None, transition_amount=None, transition_controls=None):
        target, motor_action, _ = cpg_pd_step_target(
            self.stand, self.jrange, phase, cmd, jnp.zeros(self.nu, dtype=jnp.float32),
            self.cpg_idx, self.nu, self.vmax, 0.0, self.scale,
            directional=self.directional_params(z_route), prev_command=prev_cmd,
            transition_amount=transition_amount, transition_controls=transition_controls, xp=jnp,
        )
        tau = self.kp * (target - dx.qpos[self.qadr]) - self.kd * dx.qvel[self.dadr]
        ctrl = jnp.clip(tau / jnp.maximum(self.tmax, 1e-6), -1.0, 1.0)
        dx = dx.replace(ctrl=ctrl)
        dx = jax.lax.fori_loop(0, self.frame_skip, lambda _, d: mjx.step(self.mx, d), dx)
        up = 1.0 - 2.0 * (dx.qpos[4] ** 2 + dx.qpos[5] ** 2)
        alive = jnp.logical_and(dx.qpos[2] >= 0.08, up >= 0.3)
        alive = jnp.logical_and(alive, jnp.all(jnp.isfinite(dx.qpos)))
        sat = jnp.mean((jnp.abs(ctrl) > 0.98).astype(jnp.float32))
        return dx, alive, up, sat

    def step_once_with_action(self, dx, z_route, phase, cmd, prev_cmd=None, transition_amount=None, transition_controls=None):
        target, motor_action, _ = cpg_pd_step_target(
            self.stand, self.jrange, phase, cmd, jnp.zeros(self.nu, dtype=jnp.float32),
            self.cpg_idx, self.nu, self.vmax, 0.0, self.scale,
            directional=self.directional_params(z_route), prev_command=prev_cmd,
            transition_amount=transition_amount, transition_controls=transition_controls, xp=jnp,
        )
        tau = self.kp * (target - dx.qpos[self.qadr]) - self.kd * dx.qvel[self.dadr]
        ctrl = jnp.clip(tau / jnp.maximum(self.tmax, 1e-6), -1.0, 1.0)
        dx = dx.replace(ctrl=ctrl)
        dx = jax.lax.fori_loop(0, self.frame_skip, lambda _, d: mjx.step(self.mx, d), dx)
        up = 1.0 - 2.0 * (dx.qpos[4] ** 2 + dx.qpos[5] ** 2)
        alive = jnp.logical_and(dx.qpos[2] >= 0.08, up >= 0.3)
        alive = jnp.logical_and(alive, jnp.all(jnp.isfinite(dx.qpos)))
        sat = jnp.mean((jnp.abs(ctrl) > 0.98).astype(jnp.float32))
        return dx, alive, up, sat, motor_action

    def segment_for_command(self, cmd):
        axis_is_x = jnp.abs(cmd[0]) >= jnp.abs(cmd[1])
        x_seg = jnp.where(cmd[0] >= 0.0, 0, 1)
        y_seg = jnp.where(cmd[1] >= 0.0, 2, 3)
        segment = jnp.where(axis_is_x, x_seg, y_seg)
        return segment.astype(jnp.int32)

    def route_scan(self, z_flat, transition_controls=None):
        z_route = z_flat.reshape((4, PARAM_DIM))
        dx0 = mjx.forward(self.mx, mjx.make_data(self.mx).replace(qpos=self.q0))

        def body(carry, k):
            dx, alive_prev, reached, best_prog, phase, prev_cmd = carry
            active0 = jnp.minimum(reached, self.nwp - 1)
            pos0 = dx.qpos[:2]
            target0 = self.waypoints[active0]
            dist0 = jnp.linalg.norm(target0 - pos0)
            hit = jnp.logical_and(jnp.logical_and(reached < self.nwp, dist0 <= self.radius), alive_prev)
            reached_pre = jnp.minimum(reached + hit.astype(jnp.int32), self.nwp)
            active = jnp.minimum(reached_pre, self.nwp - 1)
            prev = self.prev[active]
            direction = self.dirs[active]
            side = self.sides[active]
            seg_len = self.seg_lens[active]
            rel = pos0 - prev
            seg_prog = jnp.dot(rel, direction)
            cross = jnp.abs(jnp.dot(rel, side))
            best_base = jnp.where(reached_pre != reached, seg_prog, best_prog)
            best_new = jnp.maximum(best_base, seg_prog)
            backtrack = jnp.maximum(0.0, best_new - seg_prog)
            progress_frac = jnp.clip(seg_prog / seg_len, 0.0, 1.0)
            cmd, segment, action_scale = self.command_for_active_segment(pos0, active)
            done = reached_pre >= self.nwp

            transition_amount = cpg_transition_strength(cmd, prev_cmd, self.vmax, xp=jnp)
            dx_new, alive_step, up, sat = self.step_once(
                dx, z_route, phase, cmd, prev_cmd, transition_amount, transition_controls
            )
            do_step = jnp.logical_and(alive_prev, jnp.logical_not(done))
            keep_new = jnp.logical_and(do_step, alive_step)
            dx_safe = jax.tree_util.tree_map(lambda n, o: jnp.where(keep_new, n, o), dx_new, dx)
            alive = jnp.logical_and(alive_prev, jnp.logical_or(alive_step, done))
            base_freq = decode_params(z_route[1], xp=jnp).freq
            phase_new = phase + transition_phase_delta(
                base_freq, self.ctrl_dt, transition_amount, controls=transition_controls, xp=jnp
            )
            phase_safe = jnp.where(do_step, phase_new, phase)
            up_safe = 1.0 - 2.0 * (dx_safe.qpos[4] ** 2 + dx_safe.qpos[5] ** 2)
            sat_safe = jnp.where(do_step, sat, 0.0)
            dist_after = jnp.linalg.norm(self.waypoints[active] - dx_safe.qpos[:2])
            sample = (
                dx_safe.qpos[:2],
                dx_safe.qpos[2],
                up_safe,
                sat_safe,
                alive.astype(jnp.float32),
                reached_pre.astype(jnp.float32),
                active.astype(jnp.float32),
                segment.astype(jnp.float32),
                dist_after,
                cross,
                backtrack,
                progress_frac,
                action_scale,
            )
            return (dx_safe, alive, reached_pre, best_new, phase_safe, cmd), sample

        _, hist = jax.lax.scan(
            body,
            (
                dx0,
                jnp.asarray(True),
                jnp.asarray(0, dtype=jnp.int32),
                jnp.asarray(0.0),
                jnp.asarray(0.0),
                jnp.zeros(2, dtype=jnp.float32),
            ),
            jnp.arange(self.steps),
        )
        return hist

    def stats_from_hist(self, hist):
        xy, z, up, sat, alive, reached, active, segment, dist, cross, backtrack, progress_frac, action_scale = hist
        reached_final = reached[-1]
        active_final = active[-1]
        route_progress = reached_final + jnp.where(reached_final >= self.nwp, 0.0, progress_frac[-1])
        survived = jnp.mean(alive)
        mean_up = jnp.mean(up)
        min_z = jnp.min(z)
        mean_sat = jnp.mean(sat)
        mean_cross = jnp.mean(cross)
        max_cross = jnp.max(cross)
        mean_backtrack = jnp.mean(backtrack)
        final_dist = jnp.where(reached_final >= self.nwp, 0.0, dist[-1])
        focus_enabled = self.focus_active >= 0
        focus_mask = active == self.focus_active
        focus_any = jnp.any(focus_mask)
        focus_count = jnp.maximum(jnp.sum(focus_mask.astype(jnp.float32)), 1.0)
        focus_progress = jnp.where(focus_any, jnp.max(jnp.where(focus_mask, progress_frac, 0.0)), 0.0)
        focus_dist = jnp.where(focus_any, jnp.min(jnp.where(focus_mask, dist, 1e6)), 1e6)
        focus_cross = jnp.where(
            focus_any,
            jnp.sum(jnp.where(focus_mask, cross, 0.0)) / focus_count,
            0.0,
        )
        focus_score = jnp.where(
            focus_enabled,
            self.focus_progress_w * focus_progress
            - self.focus_dist_w * focus_dist
            - self.focus_cross_w * focus_cross,
            0.0,
        )
        score = (
            self.progress_w * route_progress
            + self.waypoint_bonus_w * reached_final
            + 8.0 * survived
            + 0.8 * mean_up
            - self.final_dist_w * final_dist
            - self.cross_w * mean_cross
            - 0.5 * self.cross_w * max_cross
            - self.backtrack_w * mean_backtrack
            - 0.1 * mean_sat
            + focus_score
        )
        score = score - 100.0 * (1.0 - survived)
        score = jnp.where(mean_sat <= self.max_saturation, score,
                          score - 25.0 * (mean_sat - self.max_saturation + 1.0))
        score = jnp.where(jnp.isfinite(score), score, -1e9)
        return jnp.asarray([
            score, reached_final, route_progress, final_dist, survived, mean_up, min_z,
            mean_sat, mean_cross, max_cross, mean_backtrack, xy[-1, 0], xy[-1, 1],
            active_final, focus_progress, focus_dist, focus_cross,
        ])

    def rollout_stats(self, z_flat):
        return self.stats_from_hist(self.route_scan(z_flat))

    def rollout_stats_with_controls(self, z_flat, transition_controls):
        return self.stats_from_hist(self.route_scan(z_flat, transition_controls=transition_controls))

    def rollout_trace(self, z_flat):
        return self.route_scan(z_flat)

    def rollout_dataset(self, z_flat, phase0):
        z_route = z_flat.reshape((4, PARAM_DIM))
        dx0 = mjx.forward(self.mx, mjx.make_data(self.mx).replace(qpos=self.q0))

        def body(carry, k):
            dx, alive_prev, reached, best_prog, phase, prev_cmd = carry
            active0 = jnp.minimum(reached, self.nwp - 1)
            pos0 = dx.qpos[:2]
            target0 = self.waypoints[active0]
            dist0 = jnp.linalg.norm(target0 - pos0)
            hit = jnp.logical_and(jnp.logical_and(reached < self.nwp, dist0 <= self.radius), alive_prev)
            reached_pre = jnp.minimum(reached + hit.astype(jnp.int32), self.nwp)
            active = jnp.minimum(reached_pre, self.nwp - 1)
            target = self.waypoints[active]
            prev = self.prev[active]
            direction = self.dirs[active]
            side = self.sides[active]
            seg_len = self.seg_lens[active]
            rel = pos0 - prev
            seg_prog = jnp.dot(rel, direction)
            cross = jnp.abs(jnp.dot(rel, side))
            best_base = jnp.where(reached_pre != reached, seg_prog, best_prog)
            best_new = jnp.maximum(best_base, seg_prog)
            backtrack = jnp.maximum(0.0, best_new - seg_prog)
            progress_frac = jnp.clip(seg_prog / seg_len, 0.0, 1.0)
            cmd, segment, action_scale = self.command_for_active_segment(pos0, active)
            done = reached_pre >= self.nwp

            obs = self.obs(dx, cmd)
            transition_amount = cpg_transition_strength(cmd, prev_cmd, self.vmax, xp=jnp)
            dx_new, alive_step, up, sat, motor_action = self.step_once_with_action(
                dx, z_route, phase, cmd, prev_cmd, transition_amount
            )
            do_step = jnp.logical_and(alive_prev, jnp.logical_not(done))
            keep_new = jnp.logical_and(do_step, alive_step)
            dx_safe = jax.tree_util.tree_map(lambda n, o: jnp.where(keep_new, n, o), dx_new, dx)
            alive = jnp.logical_and(alive_prev, jnp.logical_or(alive_step, done))
            base_freq = decode_params(z_route[1], xp=jnp).freq
            phase_new = phase + transition_phase_delta(base_freq, self.ctrl_dt, transition_amount, xp=jnp)
            phase_safe = jnp.where(do_step, phase_new, phase)
            up_safe = 1.0 - 2.0 * (dx_safe.qpos[4] ** 2 + dx_safe.qpos[5] ** 2)
            sat_safe = jnp.where(do_step, sat, 0.0)
            dist_after = jnp.linalg.norm(target - dx_safe.qpos[:2])
            sample = (
                obs,
                motor_action,
                pos0,
                dx_safe.qpos[:2],
                dx_safe.qpos[2],
                up_safe,
                sat_safe,
                alive.astype(jnp.float32),
                reached_pre.astype(jnp.float32),
                active.astype(jnp.float32),
                segment.astype(jnp.float32),
                phase,
                cmd,
                target,
                prev,
                dist0,
                dist_after,
                seg_prog,
                progress_frac,
                cross,
                backtrack,
                action_scale,
            )
            return (dx_safe, alive, reached_pre, best_new, phase_safe, cmd), sample

        _, hist = jax.lax.scan(
            body,
            (
                dx0,
                jnp.asarray(True),
                jnp.asarray(0, dtype=jnp.int32),
                jnp.asarray(0.0),
                jnp.asarray(phase0, dtype=jnp.float32),
                jnp.zeros(2, dtype=jnp.float32),
            ),
            jnp.arange(self.steps),
        )
        return hist

    def rollout_command_dataset(self, z_flat, cmds, phase0):
        z_route = z_flat.reshape((4, PARAM_DIM))
        dx0 = mjx.forward(self.mx, mjx.make_data(self.mx).replace(qpos=self.q0))

        def body(carry, cmd):
            dx, alive_prev, phase, prev_cmd = carry
            cmd = jnp.asarray(cmd, dtype=jnp.float32)
            pos0 = dx.qpos[:2]
            cmd_norm = jnp.linalg.norm(cmd)
            direction = jnp.where(cmd_norm > 1e-6, cmd / jnp.maximum(cmd_norm, 1e-6), jnp.asarray([1.0, 0.0]))
            side = jnp.asarray([-direction[1], direction[0]])
            desired_step = cmd * self.ctrl_dt
            target = pos0 + desired_step
            dist0 = jnp.linalg.norm(target - pos0)
            obs = self.obs(dx, cmd)
            segment = self.segment_for_command(cmd)

            transition_amount = cpg_transition_strength(cmd, prev_cmd, self.vmax, xp=jnp)
            dx_new, alive_step, up, sat, motor_action = self.step_once_with_action(
                dx, z_route, phase, cmd, prev_cmd, transition_amount
            )
            keep_new = jnp.logical_and(alive_prev, alive_step)
            dx_safe = jax.tree_util.tree_map(lambda n, o: jnp.where(keep_new, n, o), dx_new, dx)
            alive = jnp.logical_and(alive_prev, alive_step)
            base_freq = decode_params(z_route[1], xp=jnp).freq
            phase_safe = phase + transition_phase_delta(base_freq, self.ctrl_dt, transition_amount, xp=jnp)
            up_safe = 1.0 - 2.0 * (dx_safe.qpos[4] ** 2 + dx_safe.qpos[5] ** 2)
            sat_safe = jnp.where(alive_prev, sat, 0.0)
            delta = dx_safe.qpos[:2] - pos0
            seg_prog = jnp.dot(delta, direction)
            cross = jnp.abs(jnp.dot(delta, side))
            progress_frac = jnp.clip(seg_prog / jnp.maximum(jnp.linalg.norm(desired_step), 1e-6), -1.0, 1.0)
            dist_after = jnp.linalg.norm(target - dx_safe.qpos[:2])
            sample = (
                obs,
                motor_action,
                pos0,
                dx_safe.qpos[:2],
                dx_safe.qpos[2],
                up_safe,
                sat_safe,
                alive.astype(jnp.float32),
                jnp.asarray(0.0),
                segment.astype(jnp.float32),
                segment.astype(jnp.float32),
                phase,
                cmd,
                target,
                pos0,
                dist0,
                dist_after,
                seg_prog,
                progress_frac,
                cross,
                jnp.maximum(0.0, -seg_prog),
                jnp.linalg.norm(cmd) / self.vmax,
            )
            return (dx_safe, alive, phase_safe, cmd), sample

        _, hist = jax.lax.scan(
            body,
            (
                dx0,
                jnp.asarray(True),
                jnp.asarray(phase0, dtype=jnp.float32),
                jnp.zeros(2, dtype=jnp.float32),
            ),
            cmds,
        )
        return hist


def cem(evaler: MjxRouteEval, init_route: np.ndarray, pop: int, gens: int, seed: int,
        elite_frac: float, init_std_scale: float, inject_seed: bool, mutate_mask: np.ndarray):
    rng = np.random.default_rng(seed)
    mean = init_route.reshape(-1).astype(float)
    mask = mutate_mask.reshape(-1)
    std = (std_template(init_std_scale) * mutate_mask).reshape(-1)
    elite_n = max(4, int(pop * elite_frac))
    best_z, best_stats = None, None
    for g in range(gens):
        cand = mean + std * rng.standard_normal((pop, 4 * PARAM_DIM))
        cand = np.asarray([clip_route(c) for c in cand])
        if inject_seed:
            cand[0] = init_route.reshape(-1)
        stats = np.vstack([
            np.asarray(evaler.eval_one_stats(jnp.asarray(c, dtype=jnp.float32)))
            for c in cand
        ])
        stats[:, 0] = np.nan_to_num(stats[:, 0], nan=-1e9, posinf=-1e9, neginf=-1e9)
        order = np.argsort(-stats[:, 0])
        if best_stats is None or stats[order[0], 0] > best_stats[0]:
            best_z = cand[order[0]].copy()
            best_stats = stats[order[0]].copy()
        elites = cand[order[:elite_n]]
        mean = elites.mean(axis=0)
        std = (elites.std(axis=0) + 1e-3) * mask
        row = stats[order[0]]
        print(
            f"[mjx-route] gen {g:02d} best score={row[0]:+.3f} "
            f"reached={row[1]:.0f}/{evaler.nwp} route={row[2]:.2f} "
            f"final_dist={row[3]:.3f} cross={row[8]:.3f}/{row[9]:.3f} "
            f"backtrack={row[10]:.3f} focus={row[14]:.2f}/{row[15]:.3f} surv={row[4]:.2f} "
            f"allbest={best_stats[1]:.0f}/{evaler.nwp}:{best_stats[2]:.2f}",
            flush=True,
        )
    return best_z, best_stats


def raw_summary(route: np.ndarray) -> dict[str, list[float]]:
    return {
        "raw_forward": route[0].tolist(),
        "raw_backward": route[1].tolist(),
        "raw_left": route[2].tolist(),
        "raw_right": route[3].tolist(),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--waypoints", default="0.35,0;0.35,0.35;0,0.35;0,0")
    ap.add_argument("--radius", type=float, default=0.07)
    ap.add_argument("--steps-per-waypoint", type=int, default=260)
    ap.add_argument("--frame-skip", type=int, default=5)
    ap.add_argument("--pop", type=int, default=96)
    ap.add_argument("--gens", type=int, default=18)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--elite-frac", type=float, default=0.2)
    ap.add_argument("--kp", type=float, default=30.0)
    ap.add_argument("--kd", type=float, default=1.0)
    ap.add_argument("--scale", type=float, default=1.0)
    ap.add_argument("--stand-flex", type=float, default=None)
    ap.add_argument("--stand-knee", type=float, default=None)
    ap.add_argument("--spawn-height", type=float, default=None)
    ap.add_argument("--fast-design", default=",".join(str(x) for x in DEFAULT_FAST_DESIGN),
                    help="normalized fast design to match CommandedEnv, or 'raw' to disable")
    ap.add_argument("--max-saturation", type=float, default=0.75)
    ap.add_argument("--vmax", type=float, default=0.35)
    ap.add_argument("--gain", type=float, default=2.0)
    ap.add_argument("--axis-deadband", type=float, default=0.02)
    ap.add_argument("--command-mode", choices=["segment-axis", "axis-priority", "full"], default="segment-axis")
    ap.add_argument(
        "--router",
        choices=["blend", "softmax", "winner", "param_softmax", "param_rbf", "continuous", "param_linear"],
        default="winner",
    )
    ap.add_argument("--gate-power", type=float, default=1.0)
    ap.add_argument("--cross-w", type=float, default=16.0)
    ap.add_argument("--backtrack-w", type=float, default=10.0)
    ap.add_argument("--progress-w", type=float, default=35.0)
    ap.add_argument("--final-dist-w", type=float, default=22.0)
    ap.add_argument("--waypoint-bonus-w", type=float, default=0.0)
    ap.add_argument("--focus-active", type=int, default=-1,
                    help="active waypoint index to emphasize; use 2 for the third leg of the square")
    ap.add_argument("--focus-progress-w", type=float, default=0.0)
    ap.add_argument("--focus-dist-w", type=float, default=0.0)
    ap.add_argument("--focus-cross-w", type=float, default=0.0)
    ap.add_argument("--min-reached", type=int, default=3)
    ap.add_argument("--init-forward", default=None)
    ap.add_argument("--init-backward", default=None)
    ap.add_argument("--init-left", default=None)
    ap.add_argument("--init-right", default=None)
    ap.add_argument("--init-route", default=None,
                    help="route gait JSON with raw_forward/raw_backward/raw_left/raw_right")
    ap.add_argument("--mutate-route", default="all",
                    help="comma-separated route priors to mutate: forward,backward,left,right, or all")
    ap.add_argument("--freeze-route-freq", default="",
                    help="comma-separated route priors whose CPG frequency should not mutate")
    ap.add_argument("--init-std-scale", type=float, default=0.65)
    ap.add_argument("--no-inject-seed", action="store_true")
    ap.add_argument("--tag", default="cpg_route")
    args = ap.parse_args()
    cpg_teacher_mod.CPG_ROUTER = args.router
    cpg_teacher_mod.CPG_GATE_POWER = args.gate_power

    waypoints = parse_waypoints(args.waypoints)
    fast_design = parse_design(args.fast_design)
    stand_flex, stand_knee = resolve_stance(args.stand_flex, args.stand_knee)
    init_route = load_route(args.init_route)
    if init_route is None:
        init_route = route_seed(
            load_raw(args.init_forward),
            load_raw(args.init_backward),
            load_raw(args.init_left),
            load_raw(args.init_right),
        )
    evaler = MjxRouteEval(
        waypoints, args.radius, args.steps_per_waypoint, args.frame_skip,
        args.kp, args.kd, args.scale, stand_flex, stand_knee, args.spawn_height,
        fast_design, args.max_saturation, args.vmax, args.gain, args.axis_deadband, args.command_mode,
        args.cross_w, args.backtrack_w, args.progress_w, args.final_dist_w,
        args.waypoint_bonus_w, args.focus_active, args.focus_progress_w,
        args.focus_dist_w, args.focus_cross_w,
    )
    mutate_mask = apply_freeze_route_freq(parse_mutate_route(args.mutate_route), args.freeze_route_freq)
    best_z, best = cem(
        evaler, init_route, args.pop, args.gens, args.seed, args.elite_frac,
        args.init_std_scale, inject_seed=not args.no_inject_seed, mutate_mask=mutate_mask,
    )
    route = best_z.reshape(4, PARAM_DIM)
    xy, z, up, sat, alive, reached, active, segment, dist, cross, backtrack, progress_frac, action_scale = evaler.eval_one_trace(
        jnp.asarray(best_z, dtype=jnp.float32)
    )
    trace_reached = int(np.asarray(reached)[-1])
    trace_survived = float(np.asarray(alive).mean())
    trace_saturation = float(np.asarray(sat).mean())
    trace_final_dist = float(0.0 if trace_reached >= evaler.nwp else np.asarray(dist)[-1])
    trace_min_z = float(np.asarray(z).min())
    trace_mean_up = float(np.asarray(up).mean())
    trace_mean_cross = float(np.asarray(cross).mean())
    trace_max_cross = float(np.asarray(cross).max())
    trace_mean_backtrack = float(np.asarray(backtrack).mean())
    trace_final_xy = np.asarray(xy)[-1]
    passed = (
        trace_reached >= int(args.min_reached)
        and trace_survived >= 0.999
        and trace_saturation <= args.max_saturation
    )
    deploy_eval_args = []
    if args.command_mode == "segment-axis":
        deploy_eval_args.append("--segment-axis")
    elif args.command_mode == "axis-priority":
        deploy_eval_args.append("--axis-priority")
    final = {
        "tag": args.tag,
        "score": float(best[0]),
        "reached": trace_reached,
        "total_waypoints": int(evaler.nwp),
        "route_progress": float(np.asarray(reached)[-1] + (0.0 if trace_reached >= evaler.nwp else np.asarray(progress_frac)[-1])),
        "final_dist": trace_final_dist,
        "survived_frac": trace_survived,
        "mean_up": trace_mean_up,
        "min_z": trace_min_z,
        "saturation": trace_saturation,
        "mean_cross": trace_mean_cross,
        "max_cross": trace_max_cross,
        "mean_backtrack": trace_mean_backtrack,
        "final_x": float(trace_final_xy[0]),
        "final_y": float(trace_final_xy[1]),
        "active_final": int(np.asarray(active)[-1]),
        "search_stats": {
            "score": float(best[0]),
            "reached": int(best[1]),
            "route_progress": float(best[2]),
            "final_dist": float(best[3]),
            "survived_frac": float(best[4]),
            "saturation": float(best[7]),
            "focus_progress": float(best[14]),
            "focus_dist": float(best[15]),
            "focus_cross": float(best[16]),
        },
        "passes_gate": bool(passed),
        "min_reached": int(args.min_reached),
        "waypoints": waypoints.tolist(),
        "radius": args.radius,
        "steps_per_waypoint": args.steps_per_waypoint,
        "steps": int(evaler.steps),
        "stand_flex": stand_flex,
        "stand_knee": stand_knee,
        "spawn_height": args.spawn_height,
        "kp": args.kp,
        "kd": args.kd,
        "scale": args.scale,
        "fast_design": fast_design,
        "max_saturation": args.max_saturation,
        "vmax": args.vmax,
        "gain": args.gain,
        "axis_deadband": args.axis_deadband,
        "command_mode": args.command_mode,
        "router": args.router,
        "gate_power": args.gate_power,
        "waypoint_bonus_w": args.waypoint_bonus_w,
        "focus_active": args.focus_active,
        "focus_progress_w": args.focus_progress_w,
        "focus_dist_w": args.focus_dist_w,
        "focus_cross_w": args.focus_cross_w,
        "mutate_route": args.mutate_route,
        "freeze_route_freq": args.freeze_route_freq,
        "params": {
            name: params_to_dict(decode_params(route[i], xp=np))
            for i, name in enumerate(ROUTE_NAMES)
        },
        **raw_summary(route),
        "raw_route": route.tolist(),
        "deploy_env": {
            "CMD_CPG_FWD_RAW": ",".join(map(str, route[0].tolist())),
            "CMD_CPG_RAW": ",".join(map(str, route[1].tolist())),
            "CMD_CPG_LEFT_RAW": ",".join(map(str, route[2].tolist())),
            "CMD_CPG_RIGHT_RAW": ",".join(map(str, route[3].tolist())),
            "CMD_CPG_ROUTER": args.router,
            "CMD_CPG_GATE_POWER": str(args.gate_power),
            "CMD_VMAX": str(args.vmax),
            "CMD_RESET_NOISE": "0.0",
        },
        "deploy_eval_args": deploy_eval_args,
    }
    out_json = OUT / f"{args.tag}_route_gait.json"
    out_pkl = OUT / f"{args.tag}_route_gait.pkl"
    out_npz = OUT / f"{args.tag}_route_trace.npz"
    out_json.write_text(json.dumps(final, indent=2))
    pickle.dump(final, open(out_pkl, "wb"))
    np.savez(
        out_npz,
        xy=np.asarray(xy),
        z=np.asarray(z),
        up=np.asarray(up),
        saturation=np.asarray(sat),
        alive=np.asarray(alive),
        reached=np.asarray(reached),
        active=np.asarray(active),
        segment=np.asarray(segment),
        dist=np.asarray(dist),
        cross=np.asarray(cross),
        backtrack=np.asarray(backtrack),
        progress_frac=np.asarray(progress_frac),
        action_scale=np.asarray(action_scale),
        waypoints=waypoints,
        raw_route=np.asarray(route, dtype=np.float32),
    )
    print(json.dumps(final, indent=2), flush=True)
    print(f"saved {out_json}, {out_pkl}, {out_npz}", flush=True)


if __name__ == "__main__":
    main()
