# SPDX-License-Identifier: MIT
"""MJX-native search for a sinusoidal PD gait.

This mirrors search_cpg_gait.py but scores candidates in MJX, the same backend
used by CommandedEnv training/evaluation. The MuJoCo CPU search is still useful
for quick intuition, but it can rank gaits differently enough to mislead the
commanded controller bootstrap.
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
from cpg_teacher import (  # noqa: E402
    DEFAULT_RAW,
    JOINT_NAMES,
    LEG_NAMES,
    PARAM_DIM,
    command_vector,
    cpg_action,
    decode_params,
    encode_params,
    params_to_dict,
)
from design_codec import DESIGN_DIM, apply_fast  # noqa: E402
from gen_robot_mjcf import build_mjcf, load_spec  # noqa: E402

OUT = Path(os.environ.get("CODESIGN_OUT", "/root/proj/out"))
OUT.mkdir(parents=True, exist_ok=True)

STAND_FLEX = -0.4
STAND_KNEE = -1.1
DEFAULT_FAST_DESIGN = (0.5, 0.08, 1.0 / 3.0)


def parse_design(raw: str | None) -> tuple[float, float, float] | None:
    if raw is None or raw.strip().lower() in ("", "none", "raw"):
        return None
    vals = tuple(float(x.strip()) for x in raw.replace(";", ",").split(",") if x.strip())
    if len(vals) != 3:
        raise ValueError("--fast-design expects three comma-separated normalized values")
    return vals


def resolve_stance(stand_flex: float | None, stand_knee: float | None) -> tuple[float, float]:
    leg_defaults = load_spec(HERE / "robot.toml").get("leg_defaults", {})
    flex = float(leg_defaults.get("stand_flex", STAND_FLEX)) if stand_flex is None else float(stand_flex)
    knee = float(leg_defaults.get("stand_knee", STAND_KNEE)) if stand_knee is None else float(stand_knee)
    return flex, knee


def parse_command(raw: str | None, direction: str, speed: float) -> np.ndarray:
    if raw is None or not raw.strip():
        return command_vector(direction, speed)
    vals = [float(x.strip()) for x in raw.replace(";", ",").split(",") if x.strip()]
    if len(vals) != 2:
        raise ValueError("--command expects two comma-separated floats: vx,vy")
    return np.asarray(vals, dtype=float)


def load_raw(path: str | None) -> np.ndarray | None:
    if not path:
        return None
    d = json.loads(Path(path).read_text())
    raw = np.asarray(d["raw"], dtype=float)
    if raw.shape[0] != PARAM_DIM:
        raw = encode_params(decode_params(raw, xp=np))
    return raw


def set_stance(model: mujoco.MjModel, qpos: np.ndarray, stand_flex: float, stand_knee: float,
               spawn_height: float | None) -> np.ndarray:
    qpos = qpos.copy()
    if spawn_height is not None:
        qpos[2] = float(spawn_height)
    for j in range(model.njnt):
        name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_JOINT, j) or ""
        adr = int(model.jnt_qposadr[j])
        if name.endswith("_flex"):
            qpos[adr] = stand_flex
        elif name.endswith("_knee"):
            qpos[adr] = stand_knee
    return qpos


def decode_np(z: np.ndarray) -> dict:
    return params_to_dict(decode_params(np.asarray(z, dtype=float), xp=np))


class MjxGaitEval:
    def __init__(self, command: np.ndarray, steps: int, frame_skip: int, kp: float, kd: float, scale: float,
                 stand_flex: float, stand_knee: float, spawn_height: float | None,
                 fast_design: tuple[float, float, float] | None,
                 track_sigma: float, progress_w: float, lateral_w: float,
                 track_w: float, align_w: float, vel_progress_w: float, vel_lateral_w: float,
                 max_saturation: float):
        command = np.asarray(command, dtype=float)
        target_speed = float(np.linalg.norm(command))
        direction = command / max(target_speed, 1e-9) if target_speed > 1e-9 else np.asarray([1.0, 0.0])
        self.direction = jnp.asarray(direction, dtype=jnp.float32)
        self.side = jnp.asarray([-float(self.direction[1]), float(self.direction[0])], dtype=jnp.float32)
        self.cmd = jnp.asarray(command, dtype=jnp.float32)
        self.target_speed = float(target_speed)
        self.track_sigma = float(track_sigma)
        self.steps = int(steps)
        self.frame_skip = int(frame_skip)
        self.kp = float(kp)
        self.kd = float(kd)
        self.scale = float(scale)
        self.progress_w = float(progress_w)
        self.lateral_w = float(lateral_w)
        self.track_w = float(track_w)
        self.align_w = float(align_w)
        self.vel_progress_w = float(vel_progress_w)
        self.vel_lateral_w = float(vel_lateral_w)
        self.max_saturation = float(max_saturation)
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
        self.eval_pop = jax.jit(jax.vmap(self.rollout_stats))
        self.eval_one_trace = jax.jit(self.rollout_trace)
        self.eval_one_dataset = jax.jit(self.rollout_dataset)

    def decode(self, z):
        return decode_params(z, xp=jnp)

    def cpg_action(self, z, k):
        params = self.decode(z)
        phase = 2.0 * jnp.pi * params.freq * (k.astype(jnp.float32) * self.ctrl_dt)
        return cpg_action(phase, params, self.cpg_idx, self.nu, xp=jnp)

    def obs(self, dx):
        return jnp.concatenate([
            dx.qpos[7:7 + self.nu],
            dx.qvel[6:6 + self.nu],
            dx.qpos[3:7],
            dx.qvel[0:6],
            dx.qpos[2:3],
            self.design,
            self.cmd,
        ])

    def step_once(self, dx, z, k):
        raw = jnp.clip(self.cpg_action(z, k), -1.0, 1.0)
        target = jnp.clip(self.stand + self.scale * raw, self.jrange[:, 0], self.jrange[:, 1])
        tau = self.kp * (target - dx.qpos[self.qadr]) - self.kd * dx.qvel[self.dadr]
        ctrl = jnp.clip(tau / jnp.maximum(self.tmax, 1e-6), -1.0, 1.0)
        dx = dx.replace(ctrl=ctrl)
        dx = jax.lax.fori_loop(0, self.frame_skip, lambda _, d: mjx.step(self.mx, d), dx)
        up = 1.0 - 2.0 * (dx.qpos[4] ** 2 + dx.qpos[5] ** 2)
        alive = jnp.logical_and(dx.qpos[2] >= 0.08, up >= 0.3)
        alive = jnp.logical_and(alive, jnp.all(jnp.isfinite(dx.qpos)))
        sat = jnp.mean((jnp.abs(ctrl) > 0.98).astype(jnp.float32))
        return dx, alive, up, sat

    def rollout_stats(self, z):
        dx0 = mjx.forward(self.mx, mjx.make_data(self.mx).replace(qpos=self.q0))
        start = dx0.qpos[:2]

        def body(carry, k):
            dx, alive_prev = carry
            dx_new, alive_step, up, sat = self.step_once(dx, z, k)
            alive = jnp.logical_and(alive_prev, alive_step)
            dx_safe = jax.tree_util.tree_map(lambda n, o: jnp.where(alive, n, o), dx_new, dx)
            v = dx_safe.qvel[:2]
            speed = jnp.linalg.norm(v)
            vprog = jnp.dot(v, self.direction)
            vlat = jnp.abs(jnp.dot(v, self.side))
            verr = jnp.sum((v - self.cmd) ** 2)
            track = jnp.exp(-verr / self.track_sigma)
            align = vprog / (speed + 1e-6)
            return (dx_safe, alive), (alive.astype(jnp.float32), speed, up, dx_safe.qpos[2],
                                      sat, track, align, vprog, vlat)

        (dx, _), hist = jax.lax.scan(body, (dx0, jnp.asarray(True)), jnp.arange(self.steps))
        alive_hist, speeds, ups, zs, sats, tracks, aligns, vprogs, vlats = hist
        delta = dx.qpos[:2] - start
        progress = jnp.dot(delta, self.direction)
        lateral = jnp.abs(jnp.dot(delta, self.side))
        survived = jnp.mean(alive_hist)
        mean_speed = jnp.mean(speeds)
        mean_up = jnp.mean(ups)
        min_z = jnp.min(zs)
        sat = jnp.mean(sats)
        mean_track = jnp.mean(tracks)
        mean_align = jnp.mean(aligns)
        mean_vprog = jnp.mean(vprogs)
        mean_vlat = jnp.mean(vlats)
        score = (
            4.0 * survived
            + self.progress_w * progress
            - 3.0 * jnp.maximum(0.0, -progress)
            - self.lateral_w * lateral
            + self.track_w * mean_track
            + self.align_w * jnp.clip(mean_align, -1.0, 1.0)
            + self.vel_progress_w * mean_vprog
            - self.vel_lateral_w * mean_vlat
            + 0.4 * mean_speed
            + 0.5 * mean_up
            - 0.1 * sat
        )
        score = score - 40.0 * (1.0 - survived)
        score = jnp.where(survived >= 0.999, score, score - 10.0)
        score = jnp.where(sat <= self.max_saturation, score,
                          score - 20.0 * (sat - self.max_saturation + 1.0))
        score = jnp.where(jnp.isfinite(score), score, -1e9)
        freq = jnp.clip(z[0], 0.4, 4.0)
        return jnp.asarray([
            score, progress, delta[0], delta[1], survived, mean_speed, mean_up, min_z, sat, freq,
            mean_track, mean_align, mean_vprog, mean_vlat
        ])

    def rollout_trace(self, z):
        dx0 = mjx.forward(self.mx, mjx.make_data(self.mx).replace(qpos=self.q0))

        def body(carry, k):
            dx, alive_prev = carry
            dx_new, alive_step, up, sat = self.step_once(dx, z, k)
            alive = jnp.logical_and(alive_prev, alive_step)
            dx_safe = jax.tree_util.tree_map(lambda n, o: jnp.where(alive, n, o), dx_new, dx)
            return (dx_safe, alive), (dx_safe.qpos[:2], dx_safe.qpos[2], up, sat, alive.astype(jnp.float32))

        _, hist = jax.lax.scan(body, (dx0, jnp.asarray(True)), jnp.arange(self.steps))
        return hist

    def rollout_dataset(self, z):
        dx0 = mjx.forward(self.mx, mjx.make_data(self.mx).replace(qpos=self.q0))

        def body(carry, k):
            dx, alive_prev = carry
            obs = self.obs(dx)
            action = jnp.clip(self.cpg_action(z, k), -1.0, 1.0)
            dx_new, alive_step, up, sat = self.step_once(dx, z, k)
            alive = jnp.logical_and(alive_prev, alive_step)
            dx_safe = jax.tree_util.tree_map(lambda n, o: jnp.where(alive, n, o), dx_new, dx)
            sample = (obs, action, dx_safe.qpos[:2], dx_safe.qpos[2], up, sat, alive.astype(jnp.float32))
            return (dx_safe, alive), sample

        _, hist = jax.lax.scan(body, (dx0, jnp.asarray(True)), jnp.arange(self.steps))
        return hist


def cem(evaler: MjxGaitEval, pop: int, gens: int, seed: int, elite_frac: float,
        init_raw: np.ndarray | None = None, inject_raw: list[np.ndarray] | None = None,
        init_std_scale: float = 1.0):
    rng = np.random.default_rng(seed)
    dim = PARAM_DIM
    mean = np.array(init_raw if init_raw is not None else DEFAULT_RAW, dtype=float)
    std = np.array([
        0.8, 1.8, 1.8, 1.8, 1.8,
        0.25, 0.35, 0.35,
        0.25, 0.35, 0.35,
        0.12, 0.18, 0.18,
        0.12, 0.20, 0.18, 0.12, 0.12,
        0.40, 0.40, 0.40, 0.40,
        0.25, 0.25, 0.25, 0.25,
        0.25, 0.25, 0.25, 0.25,
        0.20, 0.20, 0.20, 0.20,
    ], dtype=float)
    if mean.shape[0] != dim or std.shape[0] != dim:
        raise ValueError(f"CEM vectors must match PARAM_DIM={dim}: mean={mean.shape} std={std.shape}")
    std *= float(init_std_scale)
    inject_raw = list(inject_raw or [])
    best_z, best_stats = None, None
    elite_n = max(4, int(pop * elite_frac))
    for g in range(gens):
        cand = mean + std * rng.standard_normal((pop, dim))
        if inject_raw:
            for i, raw in enumerate(inject_raw[:pop]):
                cand[i] = raw
        cand[:, 0] = np.clip(cand[:, 0], 0.4, 4.0)
        cand[:, 1:5] = np.mod(cand[:, 1:5], 2.0 * math.pi)
        stats = np.array(evaler.eval_pop(jnp.asarray(cand, dtype=jnp.float32)), copy=True)
        stats[:, 0] = np.nan_to_num(stats[:, 0], nan=-1e9, posinf=-1e9, neginf=-1e9)
        order = np.argsort(-stats[:, 0])
        if best_stats is None or stats[order[0], 0] > best_stats[0]:
            best_z = cand[order[0]].copy()
            best_stats = stats[order[0]].copy()
        elites = cand[order[:elite_n]]
        mean = elites.mean(axis=0)
        std = elites.std(axis=0) + 1e-3
        row = stats[order[0]]
        print(
            f"[mjx-gait] gen {g:02d} best score={row[0]:+.3f} "
            f"progress={row[1]:+.3f} dx={row[2]:+.3f} dy={row[3]:+.3f} "
            f"track={row[10]:.2f} align={row[11]:+.2f} vprog={row[12]:+.2f} "
            f"surv={row[4]:.2f} speed={row[5]:.2f} allbest={best_stats[1]:+.3f}",
            flush=True,
        )
    return best_z, best_stats


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--direction", choices=["forward", "backward", "left", "right"], default="forward")
    ap.add_argument("--command", default=None, help="explicit vx,vy command; overrides --direction/--speed")
    ap.add_argument("--steps", type=int, default=240)
    ap.add_argument("--frame-skip", type=int, default=5)
    ap.add_argument("--pop", type=int, default=128)
    ap.add_argument("--gens", type=int, default=24)
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
    ap.add_argument("--speed", type=float, default=0.35)
    ap.add_argument("--track-sigma", type=float, default=0.05)
    ap.add_argument("--progress-w", type=float, default=8.0)
    ap.add_argument("--lateral-w", type=float, default=6.0)
    ap.add_argument("--track-w", type=float, default=3.0)
    ap.add_argument("--align-w", type=float, default=1.0)
    ap.add_argument("--vel-progress-w", type=float, default=8.0)
    ap.add_argument("--vel-lateral-w", type=float, default=4.0)
    ap.add_argument("--max-saturation", type=float, default=0.75)
    ap.add_argument("--init-from", default=None, help="JSON gait file whose raw vector initializes CEM mean")
    ap.add_argument("--inject-from", action="append", default=[],
                    help="JSON gait file whose raw vector is injected into every generation")
    ap.add_argument("--init-std-scale", type=float, default=1.0)
    ap.add_argument("--tag", default="cpg_mjx")
    args = ap.parse_args()

    fast_design = parse_design(args.fast_design)
    stand_flex, stand_knee = resolve_stance(args.stand_flex, args.stand_knee)
    command = parse_command(args.command, args.direction, args.speed)
    evaler = MjxGaitEval(
        command, args.steps, args.frame_skip, args.kp, args.kd, args.scale,
        stand_flex, stand_knee, args.spawn_height, fast_design,
        args.track_sigma, args.progress_w, args.lateral_w,
        args.track_w, args.align_w, args.vel_progress_w, args.vel_lateral_w, args.max_saturation
    )
    init_raw = load_raw(args.init_from)
    inject_raw = [r for r in (load_raw(p) for p in args.inject_from) if r is not None]
    best_z, best = cem(evaler, args.pop, args.gens, args.seed, args.elite_frac,
                       init_raw=init_raw, inject_raw=inject_raw, init_std_scale=args.init_std_scale)
    xy, z, up, sat, alive = evaler.eval_one_trace(jnp.asarray(best_z, dtype=jnp.float32))
    params = decode_np(best_z)
    rollout_stats = {
        "score": float(best[0]),
        "progress": float(best[1]),
        "dx": float(best[2]),
        "dy": float(best[3]),
        "survived_frac": float(best[4]),
        "mean_speed": float(best[5]),
        "mean_up": float(best[6]),
        "min_z": float(best[7]),
        "saturation": float(best[8]),
        "freq": float(best[9]),
        "mean_track": float(best[10]),
        "mean_alignment": float(best[11]),
        "mean_velocity_progress": float(best[12]),
        "mean_lateral_velocity": float(best[13]),
    }
    direction = command / max(float(np.linalg.norm(command)), 1e-9) if np.linalg.norm(command) > 1e-9 else np.asarray([1.0, 0.0])
    passes_gate = (
        rollout_stats["survived_frac"] >= 0.999
        and rollout_stats["progress"] > 0.0
        and rollout_stats["saturation"] <= args.max_saturation
    )
    final = {
        **rollout_stats,
        "progress_x": rollout_stats["progress"],
        "target_dx": float(direction[0]),
        "target_dy": float(direction[1]),
        "command": command.tolist(),
        "passes_gate": bool(passes_gate),
        "steps": int(args.steps),
        "direction": args.direction,
        "raw": best_z.tolist(),
        "rollout_stats": rollout_stats,
        "stand_flex": stand_flex,
        "stand_knee": stand_knee,
        "spawn_height": args.spawn_height,
        "kp": args.kp,
        "kd": args.kd,
        "scale": args.scale,
        "fast_design": fast_design,
        "target_speed": float(np.linalg.norm(command)),
        "track_sigma": args.track_sigma,
        "progress_w": args.progress_w,
        "lateral_w": args.lateral_w,
        "track_w": args.track_w,
        "align_w": args.align_w,
        "vel_progress_w": args.vel_progress_w,
        "vel_lateral_w": args.vel_lateral_w,
        "max_saturation": args.max_saturation,
        "params": params,
    }
    out_json = OUT / f"{args.tag}_{args.direction}_gait.json"
    out_pkl = OUT / f"{args.tag}_{args.direction}_gait.pkl"
    out_npz = OUT / f"{args.tag}_{args.direction}_gait_trace.npz"
    out_json.write_text(json.dumps(final, indent=2))
    pickle.dump(final, open(out_pkl, "wb"))
    np.savez(out_npz, xy=np.asarray(xy), z=np.asarray(z), up=np.asarray(up),
             saturation=np.asarray(sat), alive=np.asarray(alive), params=params)
    print(json.dumps(final, indent=2), flush=True)
    print(f"saved {out_json}, {out_pkl}, {out_npz}", flush=True)


if __name__ == "__main__":
    main()
