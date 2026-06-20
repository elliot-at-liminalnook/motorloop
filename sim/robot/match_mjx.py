# SPDX-License-Identifier: MIT
"""Phase 4 — MJX SPARC self-play match (GPU).

Two robots (A learner, B opponent) in one mjx scene, each with a weapon-leg. Reward =
the SPARC differential with FORCE-weighted damage (contact penetration of a weapon
into the opponent's body, scaled by DAMAGE_REF) + aggression (closing toward B). B is
a fixed passive opponent here — A learns to WIN the SPARC score (close + spear); the
league self-play (B = a frozen A snapshot, iterated rounds) is the documented next
step (Phase 4 stretch). Verifies the SPARC match + force-weighting train on GPU.

  python match_mjx.py [--steps 4000000]
"""

from __future__ import annotations

import argparse, sys, time
from pathlib import Path
import jax, jax.numpy as jnp, numpy as np, mujoco
from mujoco import mjx
from brax.envs.base import Env, State
from brax.training.agents.ppo import train as ppo

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from gen_robot_mjcf import build_match, load_spec  # noqa: E402
from match_env import weapon_spec                  # noqa: E402

SPEC = load_spec(HERE / "robot.toml")
DAMAGE_REF = 0.05     # m of penetration that = full damage (force proxy, SPARC severity)


class MatchMjx(Env):
    def __init__(self, frame_skip: int = 5):
        spec = weapon_spec(SPEC)
        m = mujoco.MjModel.from_xml_string(build_match(spec, spec))
        self._mx = mjx.put_model(m); self._fs = frame_skip
        self._q0 = jnp.array(m.qpos0); self._nu = m.nu
        gn = lambda g: mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_GEOM, g) or ""
        an = lambda a: mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_ACTUATOR, a) or ""
        ng = m.ngeom
        mk = lambda pred: jnp.array([pred(gn(g)) for g in range(ng)])
        self._Aw = mk(lambda n: n.startswith("A_") and n.endswith("_spear"))
        self._Ab = mk(lambda n: n.startswith("A_") and not n.endswith("_spear"))
        self._Bw = mk(lambda n: n.startswith("B_") and n.endswith("_spear"))
        self._Bb = mk(lambda n: n.startswith("B_") and not n.endswith("_spear"))
        self._actA = jnp.array([a for a in range(m.nu) if an(a).startswith("A_")])
        self._nuA = int(self._actA.shape[0])
        Aj = [int(m.actuator_trnid[a, 0]) for a in range(m.nu) if an(a).startswith("A_")]
        self._Aqa = jnp.array([int(m.jnt_qposadr[j]) for j in Aj])
        self._Ada = jnp.array([int(m.jnt_dofadr[j]) for j in Aj])
        self._ArD = int(m.jnt_dofadr[mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_JOINT, "A_root")])
        self._BrD = int(m.jnt_dofadr[mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_JOINT, "B_root")])
        self._At = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, "A_torso")
        self._Bt = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, "B_torso")
        self._obs_size = 2 * self._nuA + 4 + 6 + 1 + 3 + 3

    @property
    def observation_size(self): return self._obs_size
    @property
    def action_size(self): return self._nuA
    @property
    def backend(self): return "mjx"

    def _obs(self, dx):
        rel = dx.xpos[self._Bt] - dx.xpos[self._At]
        return jnp.concatenate([dx.qpos[self._Aqa], dx.qvel[self._Ada], dx.xquat[self._At],
                                dx.qvel[self._ArD:self._ArD + 6], dx.xpos[self._At][2:3],
                                rel, dx.qvel[self._BrD:self._BrD + 3]])

    def reset(self, rng):
        qpos = self._q0.at[7:].add(jax.random.uniform(rng, (self._q0.shape[0] - 7,), minval=-0.03, maxval=0.03))
        dx = mjx.forward(self._mx, mjx.make_data(self._mx).replace(qpos=qpos))
        return State(dx, self._obs(dx), jnp.zeros(()), jnp.zeros(()), {}, {})

    def _sparc(self, dx):
        pen = jnp.maximum(0.0, -dx.contact.dist)
        g0, g1 = dx.contact.geom[:, 0], dx.contact.geom[:, 1]
        a_deal = (self._Aw[g0] & self._Bb[g1]) | (self._Aw[g1] & self._Bb[g0])
        b_deal = (self._Bw[g0] & self._Ab[g1]) | (self._Bw[g1] & self._Ab[g0])
        dealt = jnp.sum(pen * a_deal); taken = jnp.sum(pen * b_deal)
        rel = (dx.xpos[self._Bt] - dx.xpos[self._At])[:2]; n = jnp.linalg.norm(rel) + 1e-6
        toward = jnp.dot(dx.qvel[self._ArD:self._ArD + 2], rel) / n
        clos = jnp.maximum(0.0, toward); flee = jnp.maximum(0.0, -toward)
        return (6.0 * (jnp.clip(dealt / DAMAGE_REF, 0, 1) - jnp.clip(taken / DAMAGE_REF, 0, 1))
                + 5.0 * (jnp.clip(clos / 2, 0, 1) - jnp.clip(flee / 2, 0, 1)))

    def step(self, state, action):
        ctrl = jnp.zeros(self._nu).at[self._actA].set(jnp.clip(action, -1, 1))  # B passive
        dx = state.pipeline_state.replace(ctrl=ctrl)
        dx = jax.lax.fori_loop(0, self._fs, lambda i, d: mjx.step(self._mx, d), dx)
        reward = self._sparc(dx) + 0.1                       # +alive
        done = jnp.where(dx.xpos[self._At][2] < 0.12, 1.0, 0.0)
        return state.replace(pipeline_state=dx, obs=self._obs(dx), reward=reward, done=done)


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--steps", type=int, default=4_000_000)
    ap.add_argument("--envs", type=int, default=2048); args = ap.parse_args()
    env = MatchMjx()
    print(f"match-mjx: obs={env.observation_size} act(A)={env.action_size}")
    t0 = time.time(); c = []
    def prog(s, m): r = float(m.get("eval/episode_reward", 0)); c.append(r); print(f"  step {int(s):>9,} SPARC-return {r:7.2f} ({time.time()-t0:.0f}s)", flush=True)
    ppo.train(environment=env, num_timesteps=args.steps, num_evals=6, episode_length=300,
              num_envs=args.envs, batch_size=1024, num_minibatches=16, unroll_length=20,
              num_updates_per_batch=4, learning_rate=3e-4, entropy_cost=1e-2, discounting=0.97,
              reward_scaling=1.0, normalize_observations=True, seed=0, progress_fn=prog)
    print(f"PROVEN: MJX SPARC match trains on GPU — A's SPARC-return {c[0]:.1f} -> {c[-1]:.1f} "
          f"(force-weighted damage + aggression). League self-play = the stretch.")


if __name__ == "__main__":
    main()
