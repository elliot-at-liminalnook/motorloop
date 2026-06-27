# SPDX-License-Identifier: MIT
"""Search a low-dimensional sinusoidal PD gait for the current robot body.

This is a locomotion bootstrap tool: find an open-loop gait that survives and
translates, then use it as a baseline for learned residual policies.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import pickle
import sys
from pathlib import Path

import mujoco
import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from design_codec import fast_denorm  # noqa: E402
from gen_robot_mjcf import build_mjcf, load_spec  # noqa: E402

OUT = Path(os.environ.get("CODESIGN_OUT", "/root/proj/out"))
OUT.mkdir(parents=True, exist_ok=True)

LEG_NAMES = ("FL", "FR", "RL", "RR")
JOINT_NAMES = ("abd", "flex", "knee")
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


def apply_fast_to_mj_model(model: mujoco.MjModel, design: tuple[float, float, float] | None) -> None:
    if design is None:
        return
    r = fast_denorm(design)
    model.body_mass[:] *= r["mass_scale"]
    model.body_inertia[:] *= r["mass_scale"]
    model.dof_damping[:] *= r["damping_scale"]
    model.jnt_stiffness[1:] = r["joint_stiffness"]


def _set_stance(model: mujoco.MjModel, qpos: np.ndarray, stand_flex: float, stand_knee: float,
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


def direction_vector(name: str) -> np.ndarray:
    table = {
        "forward": np.array([1.0, 0.0], dtype=float),
        "backward": np.array([-1.0, 0.0], dtype=float),
        "left": np.array([0.0, 1.0], dtype=float),
        "right": np.array([0.0, -1.0], dtype=float),
    }
    return table[name]


class GaitEval:
    def __init__(self, direction: np.ndarray, steps: int, frame_skip: int, kp: float, kd: float, scale: float,
                 stand_flex: float, stand_knee: float, spawn_height: float | None,
                 overrides: dict | None, fast_design: tuple[float, float, float] | None,
                 progress_w: float = 8.0, lateral_w: float = 1.5):
        self.direction = np.asarray(direction, dtype=float)
        self.direction = self.direction / max(np.linalg.norm(self.direction), 1e-9)
        self.side = np.array([-self.direction[1], self.direction[0]], dtype=float)
        self.steps = int(steps)
        self.frame_skip = int(frame_skip)
        self.kp = float(kp)
        self.kd = float(kd)
        self.scale = float(scale)
        self.progress_w = float(progress_w)
        self.lateral_w = float(lateral_w)
        self.model = mujoco.MjModel.from_xml_string(build_mjcf(load_spec(HERE / "robot.toml"), overrides))
        apply_fast_to_mj_model(self.model, fast_design)
        self.data = mujoco.MjData(self.model)
        self.q0 = _set_stance(self.model, self.model.qpos0, stand_flex, stand_knee, spawn_height)
        self.torso = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, "torso")
        self.act_joints = [int(self.model.actuator_trnid[a, 0]) for a in range(self.model.nu)]
        self.qadr = np.array([int(self.model.jnt_qposadr[j]) for j in self.act_joints], dtype=int)
        self.dadr = np.array([int(self.model.jnt_dofadr[j]) for j in self.act_joints], dtype=int)
        self.jrange = np.array([self.model.jnt_range[j] for j in self.act_joints], dtype=float)
        self.tmax = np.array(self.model.actuator_forcerange[: self.model.nu, 1], dtype=float)
        self.stand = self.q0[self.qadr].copy()
        self.ctrl_dt = self.frame_skip * float(self.model.opt.timestep)
        self.name_to_action = {}
        for a, j in enumerate(self.act_joints):
            name = mujoco.mj_id2name(self.model, mujoco.mjtObj.mjOBJ_JOINT, j) or ""
            self.name_to_action[name] = a

    def decode(self, z: np.ndarray) -> dict:
        z = np.asarray(z, dtype=float)
        phases = np.mod(z[1:5], 2.0 * math.pi)
        return {
            "freq": float(np.clip(z[0], 0.4, 4.0)),
            "phase": phases,
            "flex_bias": float(np.clip(z[5], -0.7, 0.7)),
            "flex_sin": float(np.clip(z[6], -1.0, 1.0)),
            "flex_cos": float(np.clip(z[7], -1.0, 1.0)),
            "knee_bias": float(np.clip(z[8], -0.7, 0.7)),
            "knee_sin": float(np.clip(z[9], -1.0, 1.0)),
            "knee_cos": float(np.clip(z[10], -1.0, 1.0)),
            "abd_bias": float(np.clip(z[11], -0.4, 0.4)),
            "abd_sin": float(np.clip(z[12], -0.5, 0.5)),
            "abd_cos": float(np.clip(z[13], -0.5, 0.5)),
        }

    def action(self, params: dict, t: float) -> np.ndarray:
        a = np.zeros(self.model.nu, dtype=float)
        omega_t = 2.0 * math.pi * params["freq"] * t
        for li, leg in enumerate(LEG_NAMES):
            ph = omega_t + params["phase"][li]
            s, c = math.sin(ph), math.cos(ph)
            side = 1.0 if leg.endswith("L") else -1.0
            vals = {
                "abd": side * params["abd_bias"] + params["abd_sin"] * s + params["abd_cos"] * c,
                "flex": params["flex_bias"] + params["flex_sin"] * s + params["flex_cos"] * c,
                "knee": params["knee_bias"] + params["knee_sin"] * s + params["knee_cos"] * c,
            }
            for jn in JOINT_NAMES:
                idx = self.name_to_action[f"{leg}_{jn}"]
                a[idx] = vals[jn]
        return np.clip(a, -1.0, 1.0)

    def rollout(self, z: np.ndarray, record: bool = False) -> tuple[float, dict, dict | None]:
        p = self.decode(z)
        mujoco.mj_resetData(self.model, self.data)
        self.data.qpos[:] = self.q0
        self.data.qvel[:] = 0.0
        mujoco.mj_forward(self.model, self.data)
        start = self.data.qpos[:2].copy()
        xs, zs, ups, speeds, ctrls = [], [], [], [], []
        done_step = self.steps
        for k in range(self.steps):
            raw = self.action(p, k * self.ctrl_dt)
            target = np.clip(self.stand + self.scale * raw, self.jrange[:, 0], self.jrange[:, 1])
            tau = self.kp * (target - self.data.qpos[self.qadr]) - self.kd * self.data.qvel[self.dadr]
            ctrl = np.clip(tau / np.maximum(self.tmax, 1e-6), -1.0, 1.0)
            self.data.ctrl[:] = ctrl
            for _ in range(self.frame_skip):
                mujoco.mj_step(self.model, self.data)
            up_z = float(self.data.xmat[self.torso].reshape(3, 3)[2, 2])
            zpos = float(self.data.xpos[self.torso, 2])
            vel = self.data.qvel[:2].copy()
            xs.append(self.data.qpos[:2].copy())
            zs.append(zpos)
            ups.append(up_z)
            speeds.append(float(np.linalg.norm(vel)))
            ctrls.append(ctrl.copy())
            if zpos < 0.08 or up_z < 0.3 or not np.isfinite(self.data.qpos).all():
                done_step = k + 1
                break
        xy = np.array(xs) if xs else start[None, :]
        delta = xy[-1] - start
        progress = float(np.dot(delta, self.direction))
        lateral = abs(float(np.dot(delta, self.side)))
        survived = done_step / self.steps
        mean_speed = float(np.mean(speeds)) if speeds else 0.0
        mean_up = float(np.mean(ups)) if ups else 0.0
        min_z = float(np.min(zs)) if zs else 0.0
        sat = float(np.mean(np.abs(np.array(ctrls)) > 0.98)) if ctrls else 1.0
        # Survive first, then translate in the requested direction without side drift.
        score = (
            4.0 * survived
            + self.progress_w * progress
            - 3.0 * max(0.0, -progress)
            - self.lateral_w * lateral
            + 0.4 * mean_speed
            + 0.5 * mean_up
            - 0.1 * sat
        )
        score = score - 40.0 * (1.0 - survived)
        if survived < 0.999:
            score -= 10.0
        summary = {
            "score": score,
            "progress_x": progress,
            "dx": float(delta[0]),
            "dy": float(delta[1]),
            "target_dx": float(self.direction[0]),
            "target_dy": float(self.direction[1]),
            "survived_frac": survived,
            "steps": int(done_step),
            "mean_speed": mean_speed,
            "mean_up": mean_up,
            "min_z": min_z,
            "saturation": sat,
            "freq": p["freq"],
        }
        trace = None
        if record:
            trace = {"xy": xy, "z": np.asarray(zs), "up": np.asarray(ups), "params": p}
        return score, summary, trace


def cem(evaler: GaitEval, pop: int, gens: int, seed: int, elite_frac: float):
    rng = np.random.default_rng(seed)
    dim = 14
    mean = np.array([
        1.6, 0.0, math.pi, math.pi, 0.0,
        0.0, 0.45, 0.0,
        0.0, -0.35, 0.35,
        0.0, 0.05, 0.0,
    ], dtype=float)
    std = np.array([
        0.8, 1.8, 1.8, 1.8, 1.8,
        0.25, 0.35, 0.35,
        0.25, 0.35, 0.35,
        0.12, 0.18, 0.18,
    ], dtype=float)
    best_z, best = None, {"score": -1e9}
    elite_n = max(4, int(pop * elite_frac))
    for g in range(gens):
        cand = mean + std * rng.standard_normal((pop, dim))
        cand[:, 0] = np.clip(cand[:, 0], 0.4, 4.0)
        cand[:, 1:5] = np.mod(cand[:, 1:5], 2.0 * math.pi)
        rows = []
        for z in cand:
            score, summary, _ = evaler.rollout(z)
            rows.append((score, z, summary))
        rows.sort(key=lambda x: x[0], reverse=True)
        if rows[0][0] > best["score"]:
            best_z = rows[0][1].copy()
            best = dict(rows[0][2])
        elites = np.array([r[1] for r in rows[:elite_n]])
        mean = elites.mean(axis=0)
        std = elites.std(axis=0) + 1e-3
        print(
            f"[gait] gen {g:02d} best score={rows[0][0]:+.3f} "
            f"progress={rows[0][2]['progress_x']:+.3f} surv={rows[0][2]['survived_frac']:.2f} "
            f"speed={rows[0][2]['mean_speed']:.2f} allbest={best['progress_x']:+.3f}",
            flush=True,
        )
    return best_z, best


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--direction", choices=["forward", "backward", "left", "right"], default="backward")
    ap.add_argument("--steps", type=int, default=250)
    ap.add_argument("--frame-skip", type=int, default=5)
    ap.add_argument("--pop", type=int, default=96)
    ap.add_argument("--gens", type=int, default=18)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--elite-frac", type=float, default=0.2)
    ap.add_argument("--kp", type=float, default=18.0)
    ap.add_argument("--kd", type=float, default=0.8)
    ap.add_argument("--scale", type=float, default=0.65)
    ap.add_argument("--progress-w", type=float, default=8.0)
    ap.add_argument("--lateral-w", type=float, default=1.5)
    ap.add_argument("--stand-flex", type=float, default=None)
    ap.add_argument("--stand-knee", type=float, default=None)
    ap.add_argument("--spawn-height", type=float, default=None)
    ap.add_argument("--gear", type=float, default=None)
    ap.add_argument("--torso-mass", type=float, default=None)
    ap.add_argument("--thigh-len", type=float, default=None)
    ap.add_argument("--calf-len", type=float, default=None)
    ap.add_argument("--joint-stiffness", type=float, default=None)
    ap.add_argument("--fast-design", default=",".join(str(x) for x in DEFAULT_FAST_DESIGN),
                    help="normalized fast design to match CommandedEnv, or 'raw' to disable")
    ap.add_argument("--tag", default="cpg")
    args = ap.parse_args()
    direction = direction_vector(args.direction)
    overrides = {}
    if args.gear is not None:
        overrides.setdefault("actuator", {})["gear"] = args.gear
    if args.torso_mass is not None:
        overrides.setdefault("torso", {})["mass"] = args.torso_mass
    leg_over = {}
    if args.thigh_len is not None:
        leg_over["thigh_len"] = args.thigh_len
    if args.calf_len is not None:
        leg_over["calf_len"] = args.calf_len
    if args.joint_stiffness is not None:
        leg_over["joint_stiffness"] = args.joint_stiffness
    if leg_over:
        overrides["leg_defaults"] = leg_over
    fast_design = parse_design(args.fast_design)
    stand_flex, stand_knee = resolve_stance(args.stand_flex, args.stand_knee)
    evaler = GaitEval(direction, args.steps, args.frame_skip, args.kp, args.kd, args.scale,
                      stand_flex, stand_knee, args.spawn_height, overrides or None, fast_design,
                      args.progress_w, args.lateral_w)
    best_z, best = cem(evaler, args.pop, args.gens, args.seed, args.elite_frac)
    _, final, trace = evaler.rollout(best_z, record=True)
    final = {**final, "direction": args.direction, "raw": best_z.tolist(),
             "stand_flex": stand_flex, "stand_knee": stand_knee,
             "spawn_height": args.spawn_height, "kp": args.kp, "kd": args.kd, "scale": args.scale,
             "fast_design": fast_design, "progress_w": args.progress_w, "lateral_w": args.lateral_w,
             "overrides": overrides}
    params = evaler.decode(best_z)
    serializable_params = {
        k: (v.tolist() if hasattr(v, "tolist") else float(v))
        for k, v in params.items()
    }
    final["params"] = serializable_params
    out_json = OUT / f"{args.tag}_{args.direction}_gait.json"
    out_pkl = OUT / f"{args.tag}_{args.direction}_gait.pkl"
    out_npz = OUT / f"{args.tag}_{args.direction}_gait_trace.npz"
    out_json.write_text(json.dumps(final, indent=2))
    pickle.dump(final, open(out_pkl, "wb"))
    np.savez(out_npz, **trace)
    print(json.dumps(final, indent=2), flush=True)
    print(f"saved {out_json}, {out_pkl}, {out_npz}", flush=True)


if __name__ == "__main__":
    main()
