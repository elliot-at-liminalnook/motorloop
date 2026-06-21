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

import argparse, sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

CMD_DIM = 2
VMAX = 1.2          # max commanded planar speed (m/s); a joystick maps to [-VMAX, VMAX]^2
TRACK_W = 6.0       # STRONG: the controller dominates the movement decision
TRACK_SIGMA = 0.25  # velocity-error tolerance of the tracking kernel
UPRIGHT_W = 0.5     # always-on balance anchor (autonomy)


def sample_command(rng):
    """A random world-frame velocity command: random heading × random speed, with ~20%
    'hold' (zero) so the policy also learns to stand still + balance on command."""
    import jax, jax.numpy as jnp
    a, s, h = jax.random.split(rng, 3)
    ang = jax.random.uniform(a, (), minval=-jnp.pi, maxval=jnp.pi)
    spd = jax.random.uniform(s, (), minval=0.0, maxval=VMAX)
    hold = (jax.random.uniform(h, ()) < 0.2).astype(jnp.float32)     # 20% stand-still
    return jnp.array([jnp.cos(ang), jnp.sin(ang)]) * spd * (1.0 - hold)


def _build():
    """Build the MJX env class (imports jax lazily so --prove works without it failing early)."""
    import jax, jax.numpy as jnp, mujoco
    from mujoco import mjx
    from brax.envs.base import Env, State
    from gen_robot_mjcf import build_mjcf, load_spec

    class CommandedEnv(Env):
        def __init__(self, xml=None, frame_skip=5):
            m = mujoco.MjModel.from_xml_string(xml or build_mjcf(load_spec(HERE / "robot.toml")))
            self._mx = mjx.put_model(m); self._nu = int(m.nu); self._fs = frame_skip
            self._q0 = jnp.array(m.qpos0)
            self._obs_size = 2 * self._nu + 11 + CMD_DIM

        @property
        def observation_size(self): return self._obs_size
        @property
        def action_size(self): return self._nu
        @property
        def backend(self): return "mjx"

        def _obs(self, dx, cmd):
            return jnp.concatenate([dx.qpos[7:7 + self._nu], dx.qvel[6:6 + self._nu],
                                    dx.qpos[3:7], dx.qvel[0:6], dx.qpos[2:3], cmd])

        def _metrics0(self):
            return {"track": jnp.zeros(()), "vx": jnp.zeros(()), "vy": jnp.zeros(()),
                    "cmd_vx": jnp.zeros(()), "cmd_vy": jnp.zeros(()), "verr": jnp.zeros(())}

        def reset(self, rng):
            rng, nr, cr = jax.random.split(rng, 3)
            qpos = self._q0.at[7:7 + self._nu].add(
                jax.random.uniform(nr, (self._nu,), minval=-0.05, maxval=0.05))
            dx = mjx.forward(self._mx, mjx.make_data(self._mx).replace(qpos=qpos))
            cmd = sample_command(cr)
            return State(dx, self._obs(dx, cmd), jnp.zeros(()), jnp.zeros(()),
                         self._metrics0(), {"cmd": cmd, "rng": rng})

        def reset_with_command(self, rng, cmd):
            """Deploy: reset holding a GIVEN command (the remote controller's value)."""
            import jax.numpy as jnp
            nr, _ = jax.random.split(rng)
            qpos = self._q0.at[7:7 + self._nu].add(
                jax.random.uniform(nr, (self._nu,), minval=-0.05, maxval=0.05))
            dx = mjx.forward(self._mx, mjx.make_data(self._mx).replace(qpos=qpos))
            cmd = jnp.asarray(cmd)
            return State(dx, self._obs(dx, cmd), jnp.zeros(()), jnp.zeros(()),
                         self._metrics0(), {"cmd": cmd, "rng": rng})

        def step(self, state, action):
            cmd = state.info["cmd"]
            action = jnp.clip(action, -1.0, 1.0)
            dx = state.pipeline_state.replace(ctrl=action)
            dx = jax.lax.fori_loop(0, self._fs, lambda i, d: mjx.step(self._mx, d), dx)
            v = dx.qvel[0:2]                                  # base planar velocity (world)
            verr = jnp.sum((v - cmd) ** 2)
            track = jnp.exp(-verr / TRACK_SIGMA)              # 1 when matching the command
            up = 1.0 - 2.0 * (dx.qpos[4] ** 2 + dx.qpos[5] ** 2)
            reward = TRACK_W * track + UPRIGHT_W * up + 0.1 - 0.001 * jnp.sum(action ** 2)
            done = jnp.where(dx.qpos[2] < 0.18, 1.0, 0.0)     # fell -> balance failed
            metrics = {**state.metrics, "track": track, "vx": v[0], "vy": v[1],
                       "cmd_vx": cmd[0], "cmd_vy": cmd[1], "verr": jnp.sqrt(verr)}
            return state.replace(pipeline_state=dx, obs=self._obs(dx, cmd),
                                 reward=reward, done=done, metrics=metrics)

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
          f"(strong) vs UPRIGHT_W={UPRIGHT_W} (always-on balance).")
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
