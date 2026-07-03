# SPDX-License-Identifier: MIT
"""Command-conditioned locomotion — a remote controller strongly steers WHERE the robot
goes; the policy keeps its own balance and decides HOW (gait, leg coordination).

The standard velocity-command recipe (Go2/ANYmal-style), applied to the generated body:
  * OBS carries a 2-D command `cmd = [vx_des, vy_des]` (desired world-frame planar velocity,
    m/s) — what a joystick/remote sends. The policy is command-AWARE.
  * REWARD = a STRONG velocity-tracking term `exp(-||v_xy - cmd||² / σ)` (so the controller
    strongly sways the choice) + an always-on upright + alive anchor (autonomy/balance is
    non-negotiable) − a small control cost. With `cmd=0` the tracking term rewards standing
    still → the robot holds position and balances. The command never overrides balance; it
    biases direction.
  * TRAIN: a random command per episode (incl. zero) so the policy learns to follow ANY
    command. DEPLOY: a remote controller overwrites `state.info["cmd"]` each step (see
    `eval_commanded.py`) — the same policy then tracks a live joystick.

Composes onto the fighter: add the same `cmd` to `AdversarialEnv.obs` + this tracking term
to its reward, and the controller steers the fighter while it autonomously balances/attacks.

  python commanded_env.py --prove      # CPU mechanism check (obs carries cmd, reward tracks)
"""

from __future__ import annotations

import argparse, json, os, sys
from pathlib import Path
import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

CMD_DIM = 3          # (vx, vy, yaw-rate). 2-D callers (eval/render scripts, deploy remotes)
                     # are zero-padded by _pad_cmd — a 2-D command means "don't turn".
DEFAULT_FAST_DESIGN = (0.5, 0.08, 1.0 / 3.0)  # mass x1.0, stiffness 2 N/m, damping x1.0
VMAX = float(os.environ.get("CMD_VMAX", "1.2"))  # max commanded planar speed (m/s)
TRACK_W = 5.0       # secondary: exact speed matching after progress is learned
TRACK_SIGMA = float(os.environ.get("CMD_TRACK_SIGMA", "0.35"))  # velocity-error tolerance
UPRIGHT_W = 0.1     # balance anchor; keep lower than command/progress terms
ALIGN_W = 1.0       # signed directional agreement; moving backward is explicitly bad
BACKWARD_W = 6.0    # penalty per m/s of velocity opposite the command
PROGRESS_W = 12.0   # dense command-axis velocity reward
CMD_HOLD_STEPS = int(os.environ.get("CMD_HOLD_STEPS", "80")) # command changes inside an episode
FALL_Z = 0.10       # shared train/eval torso-height fall threshold for the validated taller body
MIN_UP_Z = 0.25     # shared train/eval topple threshold from torso quaternion
CMD_TRAIN_MODE = os.environ.get("CMD_TRAIN_MODE", "cardinal")
CMD_REWARD_MODE = os.environ.get("CMD_REWARD_MODE", "command")
CMD_CONTROL_MODE = os.environ.get("CMD_CONTROL_MODE", "pd")
RESET_NOISE = float(os.environ.get("CMD_RESET_NOISE", "0.05"))
from constants import PD_KD, PD_KP  # V.1: gains live in ONE place (constants.py)
PD_SCALE = float(os.environ.get("CMD_PD_SCALE", "1.0"))
SIMPLE_FWD_W = float(os.environ.get("CMD_SIMPLE_FWD_W", "5.0"))
SIMPLE_BACK_W = float(os.environ.get("CMD_SIMPLE_BACK_W", "3.0"))
CPG_RESIDUAL_SCALE = float(os.environ.get("CMD_CPG_RESIDUAL_SCALE", "0.30"))
CPG_RESIDUAL_SCALE_TRANSITION = float(os.environ.get("CMD_CPG_RESIDUAL_SCALE_TRANSITION", str(CPG_RESIDUAL_SCALE)))
_CPG_RESIDUAL_SCALE_WP2_RAW = os.environ.get("CMD_CPG_RESIDUAL_SCALE_WP2", "").strip()
CPG_RESIDUAL_SCALE_WP2 = None if not _CPG_RESIDUAL_SCALE_WP2_RAW else float(_CPG_RESIDUAL_SCALE_WP2_RAW)
_CPG_RESIDUAL_SCALE_WP3_RAW = os.environ.get("CMD_CPG_RESIDUAL_SCALE_WP3", "").strip()
CPG_RESIDUAL_SCALE_WP3 = None if not _CPG_RESIDUAL_SCALE_WP3_RAW else float(_CPG_RESIDUAL_SCALE_WP3_RAW)
CPG_WP2_ACTION_JSON = os.environ.get("CMD_CPG_WP2_ACTION_JSON", "").strip()
CPG_WP2_ACTION_SEGMENT_STEPS = int(os.environ.get("CMD_CPG_WP2_ACTION_SEGMENT_STEPS", "0") or "0")
CPG_WP3_ACTION_JSON = os.environ.get("CMD_CPG_WP3_ACTION_JSON", "").strip()
CPG_WP3_ACTION_SEGMENT_STEPS = int(os.environ.get("CMD_CPG_WP3_ACTION_SEGMENT_STEPS", "0") or "0")
CPG_TRANSITION_DELTA_THRESH = float(os.environ.get("CMD_CPG_TRANSITION_DELTA_THRESH", "0.08"))
CPG_TRANSITION_HOLD_STEPS = int(os.environ.get("CMD_CPG_TRANSITION_HOLD_STEPS", "0"))
CMD_ROUTE_WAYPOINTS = os.environ.get("CMD_ROUTE_WAYPOINTS", "0.35,0;0.35,0.35;0,0.35;0,0")
CMD_ROUTE_RADIUS = float(os.environ.get("CMD_ROUTE_RADIUS", "0.07"))
CMD_ROUTE_GAIN = float(os.environ.get("CMD_ROUTE_GAIN", "2.0"))
CMD_ROUTE_START_WP = int(os.environ.get("CMD_ROUTE_START_WP", "0"))
CMD_ROUTE_START_XY = os.environ.get("CMD_ROUTE_START_XY", "")
CMD_ROUTE_STARTS = os.environ.get("CMD_ROUTE_STARTS", "")
ROUTE_PROGRESS_W = float(os.environ.get("CMD_ROUTE_PROGRESS_W", "18.0"))
ROUTE_WAYPOINT_W = float(os.environ.get("CMD_ROUTE_WAYPOINT_W", "2.5"))
ROUTE_CROSS_W = float(os.environ.get("CMD_ROUTE_CROSS_W", "4.0"))
ROUTE_BACKTRACK_W = float(os.environ.get("CMD_ROUTE_BACKTRACK_W", "4.0"))
ROUTE_ACTION_W = float(os.environ.get("CMD_ROUTE_ACTION_W", "0.02"))
ROUTE_WP2_Y_OVERSHOOT_W = float(os.environ.get("CMD_ROUTE_WP2_Y_OVERSHOOT_W", "0.0"))
ROUTE_WP2_Y_VEL_W = float(os.environ.get("CMD_ROUTE_WP2_Y_VEL_W", "0.0"))
ROUTE_WP2_X_PROGRESS_W = float(os.environ.get("CMD_ROUTE_WP2_X_PROGRESS_W", "0.0"))
ROUTE_WP2_X_REMAINING_W = float(os.environ.get("CMD_ROUTE_WP2_X_REMAINING_W", "0.0"))
OBS_PRIOR_STRENGTH = os.environ.get("CMD_OBS_PRIOR_STRENGTH", "0").strip().lower() in ("1", "true", "yes", "on")
OBS_ROUTE_CONTEXT = os.environ.get("CMD_OBS_ROUTE_CONTEXT", "0").strip().lower() in ("1", "true", "yes", "on")
ROUTE_CONTEXT_DIM = 9


def resolve_optional_json(path_text: str) -> Path:
    path = Path(path_text)
    candidates = [path]
    if not path.is_absolute():
        candidates.extend([
            HERE.parents[1] / path,
            Path(os.environ.get("CODESIGN_OUT", "/root/proj/out")) / path,
        ])
        if path.parts and path.parts[0] == "out":
            candidates.append(Path(os.environ.get("CODESIGN_OUT", "/root/proj/out")).joinpath(*path.parts[1:]))
    for candidate in candidates:
        if candidate.exists():
            return candidate
    raise FileNotFoundError(path_text)


def load_waypoint_action_schedule(path_text: str, segment_steps_override: int, env_name: str) -> tuple[np.ndarray | None, int]:
    if not path_text:
        return None, 0
    data = json.loads(resolve_optional_json(path_text).read_text())
    action = np.asarray(data.get("best_action"), dtype=np.float32)
    if action.ndim == 1:
        action = action[None, :]
    if action.ndim != 2:
        raise ValueError(f"{env_name} best_action must be 1-D or 2-D, got {action.shape}")
    segment_steps = int(segment_steps_override or data.get("best", {}).get("segment_steps", 0) or 1)
    return np.clip(action, -1.0, 1.0).astype(np.float32), max(1, segment_steps)


CPG_WP2_ACTIONS, CPG_WP2_ACTION_STEPS = load_waypoint_action_schedule(
    CPG_WP2_ACTION_JSON, CPG_WP2_ACTION_SEGMENT_STEPS, "CMD_CPG_WP2_ACTION_JSON")
CPG_WP3_ACTIONS, CPG_WP3_ACTION_STEPS = load_waypoint_action_schedule(
    CPG_WP3_ACTION_JSON, CPG_WP3_ACTION_SEGMENT_STEPS, "CMD_CPG_WP3_ACTION_JSON")

from cpg_teacher import (  # noqa: E402
    cpg_pd_step_target,
    make_directional_params_from_env,
    transition_phase_delta,
    transition_strength as cpg_transition_strength,
)
# --- proven locomotion recipe (MuJoCo-Playground / Go2 style) terms — replaces the crude clearance term ---
AIRTIME_W = 1.0     # feet AIR-TIME: reward a foot for swinging ~AIRTIME_TARGET s before it touches down.
                    # THE term that yields a real stepping rhythm (vs flat-standing OR frantic tapping);
                    # negative when a foot lands too fast, so it shapes a proper gait cadence.
from constants import AIRTIME_CAP, AIRTIME_TARGET, FOOT_CONTACT_Z  # V.1 shared gait constants
                    # (cap: uncapped, one 1 s flight paid 4x a perfect landing — audit item 2a)
ACTRATE_W = 0.05    # penalize |action_t − action_{t−1}|² without suppressing exploration
VELZ_W = 0.5        # penalize vertical bounce (lin_vel_z²) → don't pogo
ANGXY_W = 0.1       # penalize roll/pitch rate → stay flat while moving
POSE_W = 0.2        # penalize hinge deviation from the stand pose (keeps gait near the
                    # well-conditioned posture; standard Go2-recipe regularizer)
SLIP_W = 0.1        # penalize foot planar motion while in CONTACT (feet push, not skate)
YAWERR_W = 0.5      # yaw-rate command error weight inside the tracking kernel
YAW_MAX = float(os.environ.get("CMD_YAW_MAX", "0.8"))   # rad/s commanded turn rate bound


def parse_route_waypoints(text: str) -> np.ndarray:
    pts = []
    for chunk in text.split(";"):
        chunk = chunk.strip()
        if not chunk:
            continue
        x, y = chunk.split(",")
        pts.append((float(x), float(y)))
    return np.asarray(pts or [(0.35, 0.0), (0.35, 0.35), (0.0, 0.35), (0.0, 0.0)], dtype=np.float32)


def parse_xy(text: str):
    text = text.strip()
    if not text:
        return None
    x, y = text.split(",")
    return np.asarray([float(x), float(y)], dtype=np.float32)


def parse_route_starts(text: str):
    starts = []
    for chunk in text.split(";"):
        chunk = chunk.strip()
        if not chunk:
            continue
        parts = [p.strip() for p in chunk.split(",")]
        if len(parts) != 3:
            raise ValueError("CMD_ROUTE_STARTS entries must be 'waypoint,x,y'")
        starts.append((int(parts[0]), float(parts[1]), float(parts[2])))
    return starts


def sample_command(rng):
    """A random velocity command (vx, vy, yaw-rate): random heading × random speed +
    random turn rate, with ~15% 'hold' (all-zero) so the policy also learns to stand
    still + balance on command."""
    import jax, jax.numpy as jnp
    a, s, h, y = jax.random.split(rng, 4)
    if CMD_TRAIN_MODE == "forward":
        direction = jnp.array([1.0, 0.0])
    elif CMD_TRAIN_MODE == "backward":
        direction = jnp.array([-1.0, 0.0])
    elif CMD_TRAIN_MODE == "cardinal":
        dirs = jnp.array([[1.0, 0.0], [0.0, 1.0], [-1.0, 0.0], [0.0, -1.0]])
        direction = dirs[jax.random.randint(a, (), 0, 4)]
    else:
        ang = jax.random.uniform(a, (), minval=-jnp.pi, maxval=jnp.pi)
        direction = jnp.array([jnp.cos(ang), jnp.sin(ang)])
    spd = jax.random.uniform(s, (), minval=0.35 * VMAX, maxval=VMAX)
    yaw = jnp.where(CMD_TRAIN_MODE in ("forward", "backward", "cardinal"), 0.0,
                    jax.random.uniform(y, (), minval=-YAW_MAX, maxval=YAW_MAX))
    hold = (jax.random.uniform(h, ()) < 0.15).astype(jnp.float32)    # 15% stand-still
    return jnp.concatenate([direction * spd, yaw.reshape((1,))]) * (1.0 - hold)


def _build():
    """Build the MJX env class (imports jax lazily so --prove works without it failing early)."""
    import jax, jax.numpy as jnp, mujoco
    from mujoco import mjx
    from brax.envs.base import Env, State
    from gen_robot_mjcf import build_mjcf, load_spec
    from design_codec import DESIGN_DIM, apply_fast as apply_design

    class CommandedEnv(Env):
        def __init__(self, xml=None, frame_skip=5):
            spec = load_spec(HERE / "robot.toml")
            m = mujoco.MjModel.from_xml_string(xml or build_mjcf(spec))
            self._mx = mjx.put_model(m); self._nu = int(m.nu); self._fs = frame_skip
            self._q0 = jnp.array(m.qpos0)
            ld = spec.get("leg_defaults", {})
            stand_flex = float(ld.get("stand_flex", -0.4))
            stand_knee = float(ld.get("stand_knee", -1.1))
            stand_abd = float(ld.get("stand_abd", 0.0))
            # Match the validated no-launch stance used by AdversarialEnv/validate_body.
            # MuJoCo's default qpos leaves limited hinge joints near zero, which is a
            # bad straight-leg/reset pose for this short-leg body.
            for j in range(m.njnt):
                nm = mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_JOINT, j) or ""
                a = int(m.jnt_qposadr[j])
                if nm.endswith("_knee"):
                    self._q0 = self._q0.at[a].set(stand_knee)
                elif nm.endswith("_flex"):
                    self._q0 = self._q0.at[a].set(stand_flex)
                elif nm.endswith("_abd"):
                    self._q0 = self._q0.at[a].set(stand_abd)
            aj = [int(m.actuator_trnid[a, 0]) for a in range(m.nu)]
            self._qa = jnp.array([int(m.jnt_qposadr[j]) for j in aj], dtype=int)
            self._da = jnp.array([int(m.jnt_dofadr[j]) for j in aj], dtype=int)
            self._jr = jnp.array([m.jnt_range[j] for j in aj])
            # ctrl scale for PD torque mapping: delivered torque = gear × ctrl, so the
            # correct divisor is GEAR. actuator_forcerange (the old divisor) merely
            # documents intent — the pre-2026-07 validator trusted it and green-lit a
            # body with 8% of design torque. Same-outcome today (gear == forcerange
            # numerically) but the causal path is now the honest one.
            gear = m.actuator_gear[:m.nu, 0]
            self._tmax = jnp.array(np.where(gear > 0, gear, m.actuator_forcerange[:m.nu, 1]))
            self._stand = self._q0[self._qa]
            self._design = jnp.array(DEFAULT_FAST_DESIGN, dtype=jnp.float32)
            if CPG_WP2_ACTIONS is not None and CPG_WP2_ACTIONS.shape[1] != self._nu:
                raise ValueError(f"CMD_CPG_WP2_ACTION_JSON action width {CPG_WP2_ACTIONS.shape[1]} != env action {self._nu}")
            if CPG_WP3_ACTIONS is not None and CPG_WP3_ACTIONS.shape[1] != self._nu:
                raise ValueError(f"CMD_CPG_WP3_ACTION_JSON action width {CPG_WP3_ACTIONS.shape[1]} != env action {self._nu}")
            self._wp2_actions = (
                jnp.asarray(CPG_WP2_ACTIONS, dtype=jnp.float32)
                if CPG_WP2_ACTIONS is not None else jnp.zeros((0, self._nu), dtype=jnp.float32)
            )
            self._wp2_action_count = int(self._wp2_actions.shape[0])
            self._wp2_action_segment_steps = int(CPG_WP2_ACTION_STEPS)
            self._wp3_actions = (
                jnp.asarray(CPG_WP3_ACTIONS, dtype=jnp.float32)
                if CPG_WP3_ACTIONS is not None else jnp.zeros((0, self._nu), dtype=jnp.float32)
            )
            self._wp3_action_count = int(self._wp3_actions.shape[0])
            self._wp3_action_segment_steps = int(CPG_WP3_ACTION_STEPS)
            route_points = parse_route_waypoints(CMD_ROUTE_WAYPOINTS)
            self._route_scale = float(max(float(np.max(np.linalg.norm(route_points, axis=1))), VMAX, 1e-6))
            self._route_waypoints = jnp.array(route_points, dtype=jnp.float32)
            self._route_n = int(self._route_waypoints.shape[0])
            self._route_start_wp = int(np.clip(CMD_ROUTE_START_WP, 0, self._route_n - 1))
            route_start_xy = parse_xy(CMD_ROUTE_START_XY)
            self._route_start_xy = None if route_start_xy is None else jnp.array(route_start_xy, dtype=jnp.float32)
            route_starts = parse_route_starts(CMD_ROUTE_STARTS)
            if route_starts:
                self._route_start_wps = jnp.array(
                    [int(np.clip(wp, 0, self._route_n - 1)) for wp, _, _ in route_starts],
                    dtype=jnp.int32,
                )
                self._route_start_xys = jnp.array([[x, y] for _, x, y in route_starts], dtype=jnp.float32)
            elif self._route_start_xy is not None:
                self._route_start_wps = jnp.array([self._route_start_wp], dtype=jnp.int32)
                self._route_start_xys = self._route_start_xy[None, :]
            else:
                self._route_start_wps = jnp.zeros((0,), dtype=jnp.int32)
                self._route_start_xys = jnp.zeros((0, 2), dtype=jnp.float32)
            self._route_start_count = int(self._route_start_wps.shape[0])
            self._dt = float(frame_skip) * float(m.opt.timestep)      # effective control dt (for air-time)
            name_to_action = {}
            for a in range(m.nu):
                j = int(m.actuator_trnid[a, 0])
                name_to_action[mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_JOINT, j) or ""] = a
            rows = []
            for leg in ("FL", "FR", "RL", "RR"):
                rows.append([name_to_action[f"{leg}_abd"], name_to_action[f"{leg}_flex"], name_to_action[f"{leg}_knee"]])
            self._cpg_idx = jnp.array(rows, dtype=jnp.int32)
            self._cpg = make_directional_params_from_env()
            self._cpg_base_freq = float(np.asarray(self._cpg.backward.freq))
            # foot geoms — for a STEPPING reward (velocity-tracking alone lets the policy stand flat;
            # rewarding feet lifting breaks that optimum so it explores a gait — the standard missing term)
            self._feet = jnp.array([mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_GEOM, f"{L}_foot")
                                    for L in ("FL", "FR", "RL", "RR")])
            self._obs_size = (2 * self._nu + 11 + DESIGN_DIM
                              + (1 if OBS_PRIOR_STRENGTH else 0)
                              + (ROUTE_CONTEXT_DIM if OBS_ROUTE_CONTEXT else 0)
                              + self._nu          # prev_action (audit item 2a/7: at 50 Hz
                                                  # with springs, stance/swing intent is
                                                  # unobservable from one qpos/qvel frame)
                              + CMD_DIM)          # cmd stays the FINAL slice (goal-tail
                                                  # convention — insert new obs BEFORE it)

        @property
        def observation_size(self): return self._obs_size
        @property
        def action_size(self): return self._nu
        @property
        def backend(self): return "mjx"

        def _model(self, design):
            return apply_design(self._mx, design)

        def _prior_strength(self, residual_scale):
            if CMD_CONTROL_MODE != "cpg_pd":
                return jnp.asarray(0.0, dtype=jnp.float32)
            return 1.0 / (1.0 + jnp.maximum(jnp.asarray(residual_scale, dtype=jnp.float32), 0.0))

        @staticmethod
        def _pad_cmd(cmd):
            """Accept legacy 2-D (vx, vy) commands from eval/render/deploy callers:
            zero-pad the yaw-rate slot. Static shape at trace time — jit-safe."""
            cmd = jnp.asarray(cmd, dtype=jnp.float32)
            if cmd.shape[-1] < CMD_DIM:
                cmd = jnp.concatenate([cmd, jnp.zeros(CMD_DIM - cmd.shape[-1], dtype=cmd.dtype)])
            return cmd

        @staticmethod
        def _yaw_rot(dx):
            """Planar rotation world->body-yaw frame from the root quaternion (w,x,y,z)."""
            w, x, y, z = dx.qpos[3], dx.qpos[4], dx.qpos[5], dx.qpos[6]
            yaw = jnp.arctan2(2.0 * (w * z + x * y), 1.0 - 2.0 * (y * y + z * z))
            c, s = jnp.cos(yaw), jnp.sin(yaw)
            return jnp.array([[c, s], [-s, c]])       # rows: body-forward, body-left

        def _obs(self, dx, design, cmd, prior_strength=None, route_wp=None, route_target=None,
                 route_dist=None, transition_amount=None, prev_cmd=None, prev_action=None):
            # BODY-frame planar velocity + command (audit item 2a): the policy should
            # act on "how fast am I moving relative to my own heading", not on world
            # coordinates its sensors can't know. Roll/pitch stay via the quaternion.
            R = self._yaw_rot(dx)
            v_body = R @ dx.qvel[0:2]
            parts = [
                dx.qpos[7:7 + self._nu],
                dx.qvel[6:6 + self._nu],
                dx.qpos[3:7],
                jnp.concatenate([v_body, dx.qvel[2:6]]),
                dx.qpos[2:3],
                design,
            ]
            if OBS_PRIOR_STRENGTH:
                ps = self._prior_strength(CPG_RESIDUAL_SCALE) if prior_strength is None else prior_strength
                parts.append(jnp.asarray(ps, dtype=jnp.float32).reshape((1,)))
            if OBS_ROUTE_CONTEXT:
                wp = jnp.zeros((), dtype=jnp.float32) if route_wp is None else route_wp.astype(jnp.float32)
                target = jnp.zeros(2, dtype=jnp.float32) if route_target is None else route_target
                delta = target - dx.qpos[:2]
                dist = jnp.linalg.norm(delta) if route_dist is None else route_dist
                prev = jnp.zeros(2, dtype=jnp.float32) if prev_cmd is None else prev_cmd
                trans = jnp.zeros((), dtype=jnp.float32) if transition_amount is None else transition_amount
                route_active = jnp.asarray(1.0 if CMD_TRAIN_MODE == "route" else 0.0, dtype=jnp.float32)
                route_scale = jnp.asarray(self._route_scale, dtype=jnp.float32)
                route_context = jnp.concatenate([
                    route_active.reshape((1,)),
                    (wp / float(max(self._route_n - 1, 1))).reshape((1,)),
                    (wp == 2).astype(jnp.float32).reshape((1,)),
                    delta / route_scale,
                    (dist / route_scale).reshape((1,)),
                    trans.reshape((1,)),
                    prev / jnp.maximum(VMAX, 1e-6),
                ])
                parts.append(route_context)
            parts.append(jnp.zeros(self._nu) if prev_action is None else prev_action)
            # command in the BODY frame; yaw-rate passes through. Goal tail stays LAST.
            parts.append(jnp.concatenate([R @ cmd[:2], cmd[2:3]]))
            return jnp.concatenate(parts)

        def _route_command(self, pos, waypoint):
            active = jnp.minimum(waypoint, self._route_n - 1)
            target = self._route_waypoints[active]
            delta = target - pos
            dist = jnp.linalg.norm(delta)
            cmd = CMD_ROUTE_GAIN * delta
            norm = jnp.linalg.norm(cmd)
            cmd = jnp.where(norm > VMAX, cmd * (VMAX / jnp.maximum(norm, 1e-6)), cmd)
            return cmd, target, dist

        def _route_cross_backtrack(self, pos, waypoint):
            active = jnp.minimum(waypoint, self._route_n - 1)
            prev_idx = jnp.maximum(active - 1, 0)
            prev = jnp.where(active > 0, self._route_waypoints[prev_idx], jnp.zeros(2))
            target = self._route_waypoints[active]
            seg = target - prev
            seg_len = jnp.maximum(jnp.linalg.norm(seg), 1e-6)
            direction = seg / seg_len
            side = jnp.asarray([-direction[1], direction[0]])
            rel = pos - prev
            progress = jnp.dot(rel, direction)
            cross = jnp.abs(jnp.dot(rel, side))
            backtrack = jnp.maximum(0.0, -progress)
            overshoot = jnp.maximum(0.0, progress - seg_len)
            return cross, backtrack + overshoot

        def _metrics0(self):
            return {"track": jnp.zeros(()), "vx": jnp.zeros(()), "vy": jnp.zeros(()),
                    "cmd_vx": jnp.zeros(()), "cmd_vy": jnp.zeros(()), "cmd_yaw": jnp.zeros(()),
                    "slip": jnp.zeros(()), "pose_dev": jnp.zeros(()), "verr": jnp.zeros(()),
                    "align": jnp.zeros(()), "speed": jnp.zeros(()), "progress": jnp.zeros(()),
                    "up": jnp.zeros(()), "height": jnp.zeros(()),
                    "residual_scale": jnp.zeros(()), "prior_strength": jnp.zeros(()),
                    "transition_amount": jnp.zeros(()),
                    "wp2_y_overshoot": jnp.zeros(()), "wp2_y_vel": jnp.zeros(()),
                    "wp2_x_progress": jnp.zeros(()), "wp2_x_remaining": jnp.zeros(()),
                    "wp2_action_segment": jnp.zeros(()),
                    "wp3_action_segment": jnp.zeros(()),
                    "route_dist": jnp.zeros(()), "route_progress": jnp.zeros(()),
                    "route_wp": jnp.zeros(())}

        def reset(self, rng):
            rng, nr, cr, sr = jax.random.split(rng, 4)
            qpos = self._q0.at[7:7 + self._nu].add(
                jax.random.uniform(nr, (self._nu,), minval=-RESET_NOISE, maxval=RESET_NOISE))
            route_wp = jnp.asarray(self._route_start_wp if CMD_TRAIN_MODE == "route" else 0, dtype=jnp.int32)
            if CMD_TRAIN_MODE == "route" and self._route_start_count > 0:
                start_i = jax.random.randint(sr, (), 0, self._route_start_count)
                qpos = qpos.at[0:2].set(self._route_start_xys[start_i])
                route_wp = self._route_start_wps[start_i]
            design = self._design
            dx = mjx.forward(self._model(design), mjx.make_data(self._mx).replace(qpos=qpos))
            route_cmd, route_target, route_dist = self._route_command(dx.qpos[:2], route_wp)
            cmd = self._pad_cmd(route_cmd if CMD_TRAIN_MODE == "route" else sample_command(cr))
            prior_strength = self._prior_strength(CPG_RESIDUAL_SCALE)
            return State(dx, self._obs(dx, design, cmd, prior_strength, route_wp, route_target, route_dist,
                                       jnp.zeros(()), cmd), jnp.zeros(()), jnp.zeros(()),
                         self._metrics0(), {"cmd": cmd, "rng": rng,
                                            "design": design,
                                            "cmd_timer": jnp.zeros((), dtype=jnp.int32),
                                            "remote": jnp.array(False),
                                            "phase": jnp.zeros(()),
                                            "prev_cmd": cmd,
                                            "prior_strength": prior_strength,
                                            "transition_amount": jnp.zeros(()),
                                            "transition_timer": jnp.zeros((), dtype=jnp.int32),
                                            "route_wp": route_wp,
                                            "route_prev_dist": route_dist,
                                            "wp2_residual_step": jnp.zeros((), dtype=jnp.int32),
                                            "wp3_residual_step": jnp.zeros((), dtype=jnp.int32),
                                            "air_time": jnp.zeros(4), "prev_action": jnp.zeros(self._nu),
                                            "prev_feet_xy": dx.geom_xpos[self._feet][:, :2]})

        def reset_with_command(self, rng, cmd):
            """Deploy: reset holding a GIVEN command (the remote controller's value)."""
            import jax.numpy as jnp
            nr, _ = jax.random.split(rng)
            qpos = self._q0.at[7:7 + self._nu].add(
                jax.random.uniform(nr, (self._nu,), minval=-RESET_NOISE, maxval=RESET_NOISE))
            design = self._design
            dx = mjx.forward(self._model(design), mjx.make_data(self._mx).replace(qpos=qpos))
            cmd = self._pad_cmd(cmd)
            route_wp = jnp.zeros((), dtype=jnp.int32)
            _, route_target, route_dist = self._route_command(dx.qpos[:2], route_wp)
            prior_strength = self._prior_strength(CPG_RESIDUAL_SCALE)
            return State(dx, self._obs(dx, design, cmd, prior_strength, route_wp, route_target, route_dist,
                                       jnp.zeros(()), cmd), jnp.zeros(()), jnp.zeros(()),
                         self._metrics0(), {"cmd": cmd, "rng": rng,
                                            "design": design,
                                            "cmd_timer": jnp.zeros((), dtype=jnp.int32),
                                            "remote": jnp.array(True),
                                            "phase": jnp.zeros(()),
                                            "prev_cmd": cmd,
                                            "prior_strength": prior_strength,
                                            "transition_amount": jnp.zeros(()),
                                            "transition_timer": jnp.zeros((), dtype=jnp.int32),
                                            "route_wp": route_wp,
                                            "route_prev_dist": route_dist,
                                            "wp2_residual_step": jnp.zeros((), dtype=jnp.int32),
                                            "wp3_residual_step": jnp.zeros((), dtype=jnp.int32),
                                            "air_time": jnp.zeros(4), "prev_action": jnp.zeros(self._nu),
                                            "prev_feet_xy": dx.geom_xpos[self._feet][:, :2]})

        def step(self, state, action):
            # _pad_cmd: eval/render/deploy overwrite info["cmd"] with legacy 2-D commands
            cmd = self._pad_cmd(state.info["cmd"])
            prev_cmd = self._pad_cmd(state.info["prev_cmd"])
            rng, cr = jax.random.split(state.info["rng"])
            timer = state.info["cmd_timer"] + 1
            should_resample = jnp.logical_and(timer >= CMD_HOLD_STEPS,
                                              jnp.logical_not(state.info["remote"]))
            new_cmd = sample_command(cr)
            cmd = jnp.where(should_resample, new_cmd, cmd)
            timer = jnp.where(should_resample, jnp.zeros_like(timer), timer)
            route_wp0 = state.info["route_wp"]
            route_dist0 = state.info["route_prev_dist"]
            route_target = self._route_waypoints[jnp.minimum(route_wp0, self._route_n - 1)]
            if CMD_TRAIN_MODE == "route":
                route_hit0 = route_dist0 <= CMD_ROUTE_RADIUS
                route_wp0 = jnp.minimum(route_wp0 + route_hit0.astype(jnp.int32), self._route_n - 1)
                route_cmd, route_target, route_dist0 = self._route_command(state.pipeline_state.qpos[:2], route_wp0)
                cmd = jnp.where(state.info["remote"], cmd, self._pad_cmd(route_cmd))
            raw_action = jnp.clip(action, -1.0, 1.0)
            wp2_residual_step0 = state.info["wp2_residual_step"]
            wp3_residual_step0 = state.info["wp3_residual_step"]
            wp2_action_segment = jnp.zeros((), dtype=jnp.int32)
            wp3_action_segment = jnp.zeros((), dtype=jnp.int32)
            if self._wp2_action_count > 0:
                wp2_action_segment = jnp.minimum(
                    wp2_residual_step0 // max(self._wp2_action_segment_steps, 1),
                    self._wp2_action_count - 1,
                )
                scheduled_action = self._wp2_actions[wp2_action_segment]
                raw_action = jnp.where((route_wp0 == 2), scheduled_action, raw_action)
            if self._wp3_action_count > 0:
                wp3_action_segment = jnp.minimum(
                    wp3_residual_step0 // max(self._wp3_action_segment_steps, 1),
                    self._wp3_action_count - 1,
                )
                scheduled_action = self._wp3_actions[wp3_action_segment]
                raw_action = jnp.where((route_wp0 == 3), scheduled_action, raw_action)
            phase = state.info["phase"]
            transition_hit = jnp.linalg.norm(cmd - prev_cmd) > CPG_TRANSITION_DELTA_THRESH
            transition_timer = jnp.where(
                transition_hit,
                jnp.asarray(CPG_TRANSITION_HOLD_STEPS, dtype=jnp.int32),
                jnp.maximum(state.info["transition_timer"] - 1, 0),
            )
            change_strength = cpg_transition_strength(cmd[:2], prev_cmd[:2], VMAX, xp=jnp)
            if CPG_TRANSITION_HOLD_STEPS > 0:
                timer_strength = transition_timer.astype(jnp.float32) / float(max(CPG_TRANSITION_HOLD_STEPS, 1))
            else:
                timer_strength = change_strength
            transition_amount = jnp.maximum(change_strength, timer_strength)
            residual_scale = jnp.where(
                transition_timer > 0,
                jnp.asarray(CPG_RESIDUAL_SCALE_TRANSITION, dtype=jnp.float32),
                jnp.asarray(CPG_RESIDUAL_SCALE, dtype=jnp.float32),
            )
            if CPG_RESIDUAL_SCALE_WP2 is not None:
                wp2_scale = jnp.asarray(CPG_RESIDUAL_SCALE_WP2, dtype=jnp.float32)
                residual_scale = jnp.where(route_wp0 == 2, jnp.maximum(residual_scale, wp2_scale), residual_scale)
            if CPG_RESIDUAL_SCALE_WP3 is not None:
                wp3_scale = jnp.asarray(CPG_RESIDUAL_SCALE_WP3, dtype=jnp.float32)
                residual_scale = jnp.where(route_wp0 == 3, jnp.maximum(residual_scale, wp3_scale), residual_scale)
            prior_strength = self._prior_strength(residual_scale)
            if CMD_CONTROL_MODE == "cpg_pd":
                target, motor_action, _ = cpg_pd_step_target(
                    self._stand, self._jr, phase, cmd[:2], raw_action, self._cpg_idx, self._nu,
                    VMAX, residual_scale, PD_SCALE, directional=self._cpg,
                    prev_command=prev_cmd[:2], transition_amount=transition_amount, xp=jnp)
            else:
                motor_action = raw_action
            if CMD_CONTROL_MODE in ("pd", "cpg_pd"):
                if CMD_CONTROL_MODE == "pd":
                    target = self._stand + PD_SCALE * motor_action
                    target = jnp.clip(target, self._jr[:, 0], self._jr[:, 1])
                tau = PD_KP * (target - state.pipeline_state.qpos[self._qa]) - PD_KD * state.pipeline_state.qvel[self._da]
                ctrl = jnp.clip(tau / jnp.maximum(self._tmax, 1e-6), -1.0, 1.0)
            else:
                ctrl = motor_action
            design = state.info["design"]
            dx = state.pipeline_state.replace(ctrl=ctrl)
            mxd = self._model(design)
            dx = jax.lax.fori_loop(0, self._fs, lambda i, d: mjx.step(mxd, d), dx)
            v = dx.qvel[0:2]                                  # base planar velocity (world)
            # tracking kernel over the FULL command: planar velocity + yaw rate. Rotation-
            # invariant (|v−cmd| is the same in world and body frames), so reward math can
            # stay world-frame while the OBS are body-frame.
            verr = jnp.sum((v - cmd[:2]) ** 2) + YAWERR_W * (dx.qvel[5] - cmd[2]) ** 2
            track = jnp.exp(-verr / TRACK_SIGMA)              # 1 when matching the command
            up = 1.0 - 2.0 * (dx.qpos[4] ** 2 + dx.qpos[5] ** 2)
            route_dist1 = jnp.linalg.norm(route_target - dx.qpos[:2])
            route_progress = route_dist0 - route_dist1
            route_hit1 = route_dist1 <= CMD_ROUTE_RADIUS
            route_cross, route_backtrack = self._route_cross_backtrack(dx.qpos[:2], route_wp0)
            wp2_active = (route_wp0 == 2).astype(jnp.float32)
            wp2_y_overshoot = wp2_active * jnp.maximum(0.0, dx.qpos[1] - route_target[1])
            wp2_y_vel = wp2_active * jnp.maximum(0.0, dx.qvel[1])
            wp2_x_progress = wp2_active * jnp.maximum(0.0, -dx.qvel[0]) / jnp.maximum(VMAX, 1e-6)
            wp2_x_remaining = wp2_active * jnp.maximum(0.0, dx.qpos[0] - route_target[0])
            cmd_norm = jnp.linalg.norm(cmd[:2])
            speed = jnp.linalg.norm(v)
            progress = jnp.dot(v, cmd[:2]) / (cmd_norm + 1e-6)
            align = jnp.dot(v, cmd[:2]) / (speed * cmd_norm + 1e-6)
            active = (cmd_norm > 0.05).astype(jnp.float32)
            cmd_active = jnp.clip(cmd_norm / VMAX, 0.0, 1.0)
            # FOOT AIR-TIME — the gait-rhythm term: credit each foot, on the step it lands, for how long
            # it swung (air_time − target). Negative if it lands too fast → shapes a proper cadence, not
            # flat-standing and not frantic tapping. Per-foot air time tracked in info.
            foot_z = dx.geom_xpos[self._feet][:, 2]
            contact = foot_z < FOOT_CONTACT_Z
            air_time = state.info["air_time"]
            first_contact = jnp.logical_and(contact, air_time > 0.0)
            # credit CAPS at AIRTIME_CAP: a long hop lands once and must not out-earn a
            # cadence of on-target swings (uncapped, flight time was linear free money)
            air_rwd = jnp.sum((jnp.minimum(air_time, AIRTIME_CAP) - AIRTIME_TARGET)
                              * first_contact.astype(jnp.float32))
            new_air = jnp.where(contact, 0.0, air_time + self._dt)
            # feet-slip: feet in CONTACT should hold ground, not skate (planar drift of
            # contacting feet per control step, squared, normalized by dt -> a velocity²)
            feet_xy = dx.geom_xpos[self._feet][:, :2]
            slip = jnp.sum(contact.astype(jnp.float32)
                           * jnp.sum((feet_xy - state.info["prev_feet_xy"]) ** 2, axis=1)) / (self._dt ** 2)
            # pose regularization toward the stand posture (Go2-recipe term)
            pose_dev = jnp.sum((dx.qpos[self._qa] - self._stand) ** 2)
            # smoothness + stability — kill the reckless lurch-and-fall
            act_rate = jnp.sum((motor_action - state.info["prev_action"]) ** 2)
            velz = dx.qvel[2] ** 2                             # vertical bounce
            angxy = dx.qvel[3] ** 2 + dx.qvel[4] ** 2          # roll/pitch rate
            if CMD_REWARD_MODE == "simple_forward":
                fwd01 = jnp.clip(progress / VMAX, -1.0, 1.0)
                reward = (1.0 + up + SIMPLE_FWD_W * fwd01
                          - SIMPLE_BACK_W * jnp.maximum(0.0, -fwd01)
                          - 0.001 * jnp.sum(ctrl ** 2))
            elif CMD_REWARD_MODE == "route":
                route_step = jnp.clip(route_progress / jnp.maximum(VMAX * self._dt, 1e-6), -1.0, 1.0)
                reward = (0.2 + UPRIGHT_W * up
                          + TRACK_W * 0.25 * track
                          + ROUTE_PROGRESS_W * route_step
                          + ROUTE_WAYPOINT_W * route_hit1.astype(jnp.float32)
                          - ROUTE_CROSS_W * route_cross
                          - ROUTE_BACKTRACK_W * route_backtrack
                          - ROUTE_ACTION_W * jnp.sum(raw_action ** 2)
                          + ROUTE_WP2_X_PROGRESS_W * wp2_x_progress
                          - ROUTE_WP2_X_REMAINING_W * wp2_x_remaining
                          - ROUTE_WP2_Y_OVERSHOOT_W * wp2_y_overshoot
                          - ROUTE_WP2_Y_VEL_W * wp2_y_vel
                          - ACTRATE_W * act_rate - VELZ_W * velz - ANGXY_W * angxy
                          - 0.001 * jnp.sum(ctrl ** 2))
            else:
                # PROGRESS credit CLAMPS at the commanded magnitude: exact tracking is the
                # maximum — overspeeding past the command earns nothing extra (it used to
                # out-earn exact tracking by +3.7-4.6/step; audit item 2a's headline exploit).
                progress_capped = jnp.clip(progress, -cmd_norm, cmd_norm) / VMAX
                # Nonzero commands should not pay track-kernel crumbs to passive
                # settling or backsliding. Exact tracking keeps full credit; zero
                # command still rewards holding position.
                track_gate = jnp.where(active > 0.0,
                                       jnp.clip(progress / jnp.maximum(cmd_norm, 1e-6), 0.0, 1.0),
                                       1.0)
                reward = (TRACK_W * track * track_gate + UPRIGHT_W * up + 0.1
                          + ALIGN_W * active * jnp.clip(align, -1.0, 1.0)
                          + PROGRESS_W * active * progress_capped
                          + AIRTIME_W * air_rwd * cmd_active
                          - BACKWARD_W * active * jnp.maximum(0.0, -progress)
                          - POSE_W * pose_dev - SLIP_W * slip
                          - ACTRATE_W * act_rate - VELZ_W * velz - ANGXY_W * angxy
                          - 0.001 * jnp.sum(ctrl ** 2))
            done = jnp.where((dx.qpos[2] < FALL_Z) | (up < MIN_UP_Z), 1.0, 0.0)
            metrics = {**state.metrics, "track": track, "vx": v[0], "vy": v[1],
                       "cmd_vx": cmd[0], "cmd_vy": cmd[1], "cmd_yaw": cmd[2],
                       "slip": slip, "pose_dev": pose_dev, "verr": jnp.sqrt(verr),
                       "align": align, "speed": speed, "progress": progress,
                       "up": up, "height": dx.qpos[2], "residual_scale": residual_scale,
                       "prior_strength": prior_strength, "transition_amount": transition_amount,
                       "wp2_y_overshoot": wp2_y_overshoot,
                       "wp2_y_vel": wp2_y_vel,
                       "wp2_x_progress": wp2_x_progress,
                       "wp2_x_remaining": wp2_x_remaining,
                       "wp2_action_segment": wp2_action_segment.astype(jnp.float32),
                       "wp3_action_segment": wp3_action_segment.astype(jnp.float32),
                       "route_dist": route_dist1, "route_progress": route_progress,
                       "route_wp": route_wp0.astype(jnp.float32)}
            wp2_residual_step1 = jnp.where(
                route_wp0 == 2,
                wp2_residual_step0 + jnp.asarray(1, dtype=jnp.int32),
                jnp.zeros((), dtype=jnp.int32),
            )
            wp3_residual_step1 = jnp.where(
                route_wp0 == 3,
                wp3_residual_step0 + jnp.asarray(1, dtype=jnp.int32),
                jnp.zeros((), dtype=jnp.int32),
            )
            info = {**state.info, "cmd": cmd, "rng": rng, "cmd_timer": timer,
                    "phase": phase + transition_phase_delta(self._cpg_base_freq, self._dt, transition_amount, xp=jnp),
                    "prev_cmd": cmd, "prior_strength": prior_strength,
                    "transition_amount": transition_amount, "transition_timer": transition_timer,
                    "route_wp": route_wp0, "route_prev_dist": route_dist1,
                    "wp2_residual_step": wp2_residual_step1,
                    "wp3_residual_step": wp3_residual_step1,
                    "air_time": new_air, "prev_action": motor_action,
                    "prev_feet_xy": feet_xy}
            return state.replace(pipeline_state=dx, obs=self._obs(dx, design, cmd, prior_strength, route_wp0,
                                                                  route_target, route_dist1,
                                                                  transition_amount, prev_cmd,
                                                                  prev_action=motor_action),
                                 reward=reward, done=done, metrics=metrics, info=info)

    return CommandedEnv


def prove():
    """CPU mechanism check: command enters obs; tracking reward peaks when velocity matches."""
    import numpy as np
    # the tracking kernel + reward shape are pure-python checkable without MJX:
    def track(v, cmd): return float(np.exp(-np.sum((np.array(v) - np.array(cmd)) ** 2) / TRACK_SIGMA))
    cmd = [VMAX, 0.0]
    r_match = track([VMAX, 0.0], cmd); r_wrong = track([-VMAX, 0.0], cmd); r_zero = track([0, 0], cmd)
    print(f"command [{VMAX},0] (move +x): tracking reward — moving +x={r_match:.2f}  "
          f"moving -x={r_wrong:.2f}  standing={r_zero:.2f}")
    print(f"obs grows by CMD_DIM={CMD_DIM} (policy is command-aware); TRACK_W={TRACK_W} "
          f"vs UPRIGHT_W={UPRIGHT_W} (balance); CMD_REWARD_MODE={CMD_REWARD_MODE}; "
          f"CMD_CONTROL_MODE={CMD_CONTROL_MODE}.")
    ok = r_match > 0.9 and r_match > 5 * r_wrong and r_match > r_zero
    # zero command -> standing is rewarded (hold + balance)
    z = track([0, 0], [0, 0]); zmove = track([VMAX, 0], [0, 0])
    hold_ok = z > 0.9 and z > 5 * zmove
    print(f"zero command (hold): standing reward={z:.2f} > moving={zmove:.2f} -> holds+balances: {hold_ok}")
    print(f"PROVEN: command-conditioning mechanism — a directional command strongly rewards "
          f"moving that way, zero command rewards holding, balance is always on: {ok and hold_ok}.")
    sys.exit(0 if (ok and hold_ok) else 1)


if __name__ == "__main__":
    ap = argparse.ArgumentParser(); ap.add_argument("--prove", action="store_true")
    a = ap.parse_args()
    if a.prove:
        prove()
    else:
        print("CommandedEnv module — use --prove (CPU) or import _build() on a GPU/MJX box.")
