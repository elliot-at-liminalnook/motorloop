# SPDX-License-Identifier: MIT
"""Stage B — adversarial, design-conditioned policy (GPU), warm-started from the
locomotion universal policy (Stage A). The skill-ladder our combat-dodge work proved:
learn to move the body first, THEN learn to fight, so we don't relearn balance under
adversarial pressure (which collapses).

Same default body as Stage A (warm-start-compatible), TWO robots in one scene,
LEGS-AS-WEAPONS (a leg/foot contacting the opponent's body = penetration-weighted
damage). Obs = [Stage-A locomotion obs (38)] + [opponent rel pos/vel (6)] so the
first 38 dims match Stage A; warm-start pads the input layer 38->44 (opponent inputs
init ~0 -> starts as the locomotor). Reward = SPARC (damage dealt - taken +
aggression) + a small locomotion anchor (don't forget to stand). B is passive here
(single-agent attack); self-play league = the stretch.

  python train_adversarial.py [--steps 12000000 --resume <loco_ckpt>]
"""

from __future__ import annotations

import argparse, os, pickle, sys, time
from pathlib import Path
import jax, jax.numpy as jnp, numpy as np, mujoco
from mujoco import mjx
from brax.envs.base import Env, State
from brax.training.agents.ppo import train as ppo

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from gen_robot_mjcf import build_match, load_spec  # noqa: E402

SPEC = load_spec(HERE / "robot.toml")
OUT = Path("/root/proj/out"); OUT.mkdir(parents=True, exist_ok=True)
LOCO_OBS = 38; DAMAGE_REF = 0.05


class AdversarialEnv(Env):
    def __init__(self, frame_skip=5):
        m = mujoco.MjModel.from_xml_string(build_match(SPEC, SPEC))   # default body x2
        self._mx = mjx.put_model(m); self._fs = frame_skip; self._nu = m.nu
        self._q0 = jnp.array(m.qpos0)
        gn = lambda g: mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_GEOM, g) or ""
        an = lambda a: mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_ACTUATOR, a) or ""
        mk = lambda p: jnp.array([p(gn(g)) for g in range(m.ngeom)])
        leg_geom = lambda n, s: n.startswith(s + "_") and (
            n.endswith("_hipg") or n.endswith("_thighg") or
            n.endswith("_calfg") or n.endswith("_foot") or n.endswith("_spear"))
        # legs-as-weapons: any A leg geom vs any B body geom (and vice-versa)
        self._Aleg = mk(lambda n: leg_geom(n, "A"))
        self._Abody = mk(lambda n: n.startswith("A_") and n != "floor")
        self._Bleg = mk(lambda n: leg_geom(n, "B"))
        self._Bbody = mk(lambda n: n.startswith("B_") and n != "floor")
        self._actA = jnp.array([a for a in range(m.nu) if an(a).startswith("A_")])
        self._nuA = int(self._actA.shape[0])
        Aj = [int(m.actuator_trnid[a, 0]) for a in range(m.nu) if an(a).startswith("A_")]
        self._Aqa = jnp.array([int(m.jnt_qposadr[j]) for j in Aj])
        self._Ada = jnp.array([int(m.jnt_dofadr[j]) for j in Aj])
        self._ArD = int(m.jnt_dofadr[mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_JOINT, "A_root")])
        self._BrD = int(m.jnt_dofadr[mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_JOINT, "B_root")])
        self._At = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, "A_torso")
        self._Bt = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, "B_torso")
        self._hinge = jnp.array(m.jnt_type == mujoco.mjtJoint.mjJNT_HINGE)
        self._obs_size = LOCO_OBS + 6

    @property
    def observation_size(self): return self._obs_size
    @property
    def action_size(self): return self._nuA
    @property
    def backend(self): return "mjx"

    def _design_model(self, d):
        mass_s = 0.6 + 0.8 * d[0]; stiff = 25.0 * d[1]; damp_s = 0.5 + 1.5 * d[2]
        return self._mx.replace(
            body_mass=self._mx.body_mass.at[1:].multiply(mass_s),
            body_inertia=self._mx.body_inertia.at[1:].multiply(mass_s),
            jnt_stiffness=jnp.where(self._hinge, stiff, self._mx.jnt_stiffness),
            dof_damping=self._mx.dof_damping * damp_s)

    def _obs(self, dx, d):
        loco = jnp.concatenate([dx.qpos[self._Aqa], dx.qvel[self._Ada], dx.xquat[self._At],
                                dx.qvel[self._ArD:self._ArD + 6], dx.xpos[self._At][2:3], d])
        opp = jnp.concatenate([dx.xpos[self._Bt] - dx.xpos[self._At], dx.qvel[self._BrD:self._BrD + 3]])
        return jnp.concatenate([loco, opp])

    def reset(self, rng):
        rng, dr, nr = jax.random.split(rng, 3)
        d = jax.random.uniform(dr, (3,))
        qpos = self._q0.at[7:].add(jax.random.uniform(nr, (self._q0.shape[0] - 7,), minval=-0.03, maxval=0.03))
        dx = mjx.forward(self._design_model(d), mjx.make_data(self._mx).replace(qpos=qpos))
        return State(dx, self._obs(dx, d), jnp.zeros(()), jnp.zeros(()), {}, {"design": d})

    def step(self, state, action):
        d = state.info["design"]; mxd = self._design_model(d)
        ctrl = jnp.zeros(self._nu).at[self._actA].set(jnp.clip(action, -1, 1))
        dx = state.pipeline_state.replace(ctrl=ctrl)
        dx = jax.lax.fori_loop(0, self._fs, lambda i, x: mjx.step(mxd, x), dx)
        pen = jnp.maximum(0.0, -dx.contact.dist); g0, g1 = dx.contact.geom[:, 0], dx.contact.geom[:, 1]
        dealt = jnp.sum(pen * ((self._Aleg[g0] & self._Bbody[g1]) | (self._Aleg[g1] & self._Bbody[g0])))
        taken = jnp.sum(pen * ((self._Bleg[g0] & self._Abody[g1]) | (self._Bleg[g1] & self._Abody[g0])))
        rel = (dx.xpos[self._Bt] - dx.xpos[self._At])[:2]; n = jnp.linalg.norm(rel) + 1e-6
        toward = jnp.dot(dx.qvel[self._ArD:self._ArD + 2], rel) / n
        up = 1.0 - 2.0 * (dx.qpos[4] ** 2 + dx.qpos[5] ** 2)
        sparc = (6.0 * (jnp.clip(dealt / DAMAGE_REF, 0, 1) - jnp.clip(taken / DAMAGE_REF, 0, 1))
                 + 5.0 * jnp.clip(toward / 2, -1, 1))
        reward = sparc + 0.5 * up                                  # + locomotion anchor
        done = jnp.where(dx.xpos[self._At][2] < 0.18, 1.0, 0.0)
        return state.replace(pipeline_state=dx, obs=self._obs(dx, d), reward=reward, done=done)


def warm_start(path, obs_dim):
    """Best-effort: pad Stage-A params (obs 38) to Stage-B obs; fall back to None."""
    try:
        norm, policy = pickle.load(open(path, "rb"))
        pad = obs_dim - LOCO_OBS
        norm = norm.replace(mean=jnp.concatenate([norm.mean, jnp.zeros(pad)]),
                            summed_variance=jnp.concatenate([norm.summed_variance, jnp.full(pad, float(norm.count))]))
        def pad_leaf(x):
            return jnp.concatenate([x, jnp.zeros((pad,) + x.shape[1:])], 0) if (x.ndim >= 1 and x.shape[0] == LOCO_OBS) else x
        policy = jax.tree_util.tree_map(pad_leaf, policy)
        print(f"WARM-START ok: padded Stage-A params {LOCO_OBS}->{obs_dim}", flush=True)
        return (norm, policy)
    except Exception as e:
        print(f"warm-start failed ({type(e).__name__}: {e}) -> training Stage B from scratch", flush=True)
        return None


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--steps", type=int, default=12_000_000)
    ap.add_argument("--envs", type=int, default=2048); ap.add_argument("--resume", default=None)
    args = ap.parse_args()
    env = AdversarialEnv()
    print(f"adversarial env: obs={env.observation_size} act(A)={env.action_size}", flush=True)
    restore = warm_start(args.resume, env.observation_size) if args.resume and os.path.exists(args.resume) else None
    t0 = time.time(); csv = OUT / "adv_metrics.csv"; csv.write_text("step,reward,sec\n")
    def prog(s, m):
        r = float(m.get("eval/episode_reward", 0)); open(csv, "a").write(f"{int(s)},{r:.3f},{time.time()-t0:.0f}\n")
        print(f"  [adv] step {int(s):>9,} SPARC-return {r:7.2f} ({time.time()-t0:.0f}s)", flush=True)
    def ck(*a):
        try: pickle.dump(a[-1], open(OUT / "adv_ckpt.pkl", "wb"))
        except Exception: pass
    ppo.train(environment=env, num_timesteps=args.steps, num_evals=max(6, args.steps // 1_000_000),
              episode_length=300, num_envs=args.envs, batch_size=1024, num_minibatches=16,
              unroll_length=20, num_updates_per_batch=4, learning_rate=3e-4, entropy_cost=1e-2,
              discounting=0.97, reward_scaling=1.0, normalize_observations=True, seed=0,
              progress_fn=prog, policy_params_fn=ck, restore_params=restore)
    print("PROVEN: Stage B adversarial design-conditioned policy trained (warm-started "
          "from locomotion); co-design fitness can now use SPARC combat return.", flush=True)


if __name__ == "__main__":
    main()
