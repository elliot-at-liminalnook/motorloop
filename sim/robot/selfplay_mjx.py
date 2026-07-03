# SPDX-License-Identifier: MIT
"""Phase 4 — MJX self-play with a Hall-of-Fame league (GPU).

Two SYMMETRIC generated bodies in ONE scene (both = weapon_spec(robot.toml)), each
controllable with a weapon-leg. A is the learner; B is driven by a FROZEN policy snapshot
drawn from a Hall of Fame of past learners (league self-play — sampling the archive, not
just the latest, is what stops the arms race cycling). Symmetric bodies mean an A snapshot
plugs straight into B's slot (identical obs/act), so the league actually self-plays. Reward
= the SPARC differential with force/penetration-weighted damage + aggression. Each league
round: train A vs the sampled B for K steps, snapshot A into the HoF, repeat.

This realizes `match_env.selfplay()`: real matches between two bodies+policies, self-play
training, and a HoF of (body, policy) opponents — co-evolution (coevolve.py, where the
opponent is the GENERATED attacker.toml body) and the match share ONE morphology space
(every body comes from the robot.toml schema). Asymmetric cross-body leagues need a policy
per body shape (noted at the end).

  python selfplay_mjx.py [--rounds 4 --steps 3000000]
  python selfplay_mjx.py --tiny        # plumbing run for the e2e harness
"""

from __future__ import annotations

import argparse, os, pickle, sys, time
from pathlib import Path
import jax, jax.numpy as jnp, numpy as np, mujoco
from mujoco import mjx
from brax.envs.base import Env, State
from brax.training.agents.ppo import train as ppo
import ppo_nets as ppo_networks  # shared (512,256,128) + small-final-init factory (audit item 5)
from brax.training.acme import running_statistics

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from gen_robot_mjcf import build_match, load_spec  # noqa: E402
from match_env import weapon_spec                  # noqa: E402

OUT = Path(os.environ.get("CODESIGN_OUT", "/root/proj/out")); OUT.mkdir(parents=True, exist_ok=True)
OURS = weapon_spec(load_spec(HERE / "robot.toml"))     # our body + a spear leg
DAMAGE_REF = 0.05


def METRIC(**kw):
    print("METRIC " + " ".join(f"{k}={v}" for k, v in kw.items()), flush=True)


class SelfPlayEnv(Env):
    """A learns; B acts from a frozen opponent policy supplied at reset via env state.
    The opponent inference fn + params are captured at construction (one per league round)."""
    def __init__(self, opp_infer=None, opp_params=None, frame_skip=5):
        m = mujoco.MjModel.from_xml_string(build_match(OURS, OURS))   # symmetric self-play
        self._mx = mjx.put_model(m); self._fs = frame_skip; self._nu = m.nu
        self._q0 = jnp.array(m.qpos0)
        self._opp_infer = opp_infer; self._opp_params = opp_params
        gn = lambda g: mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_GEOM, g) or ""
        an = lambda a: mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_ACTUATOR, a) or ""
        mk = lambda pred: jnp.array([pred(gn(g)) for g in range(m.ngeom)])
        self._Aw = mk(lambda n: n.startswith("A_") and n.endswith("_spear"))
        self._Ab = mk(lambda n: n.startswith("A_") and not n.endswith("_spear"))
        self._Bw = mk(lambda n: n.startswith("B_") and n.endswith("_spear"))
        self._Bb = mk(lambda n: n.startswith("B_") and not n.endswith("_spear"))
        self._actA = jnp.array([a for a in range(m.nu) if an(a).startswith("A_")])
        self._actB = jnp.array([a for a in range(m.nu) if an(a).startswith("B_")])
        self._nuA, self._nuB = int(self._actA.shape[0]), int(self._actB.shape[0])
        idx = lambda s: ([int(m.jnt_qposadr[int(m.actuator_trnid[a, 0])])
                          for a in range(m.nu) if an(a).startswith(s)],
                         [int(m.jnt_dofadr[int(m.actuator_trnid[a, 0])])
                          for a in range(m.nu) if an(a).startswith(s)])
        (self._Aqa, self._Ada) = map(jnp.array, idx("A_"))
        (self._Bqa, self._Bda) = map(jnp.array, idx("B_"))
        jd = lambda nm: int(m.jnt_dofadr[mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_JOINT, nm)])
        self._ArD, self._BrD = jd("A_root"), jd("B_root")
        bt = lambda nm: mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, nm)
        self._At, self._Bt = bt("A_torso"), bt("B_torso")
        self._obsA = 2 * self._nuA + 4 + 6 + 1 + 3 + 3
        self._obsB = 2 * self._nuB + 4 + 6 + 1 + 3 + 3

    @property
    def observation_size(self): return self._obsA
    @property
    def action_size(self): return self._nuA
    @property
    def backend(self): return "mjx"

    def _obs_for(self, dx, qa, da, rD, me, opp):
        rel = dx.xpos[opp] - dx.xpos[me]
        return jnp.concatenate([dx.qpos[qa], dx.qvel[da], dx.xquat[me],
                                dx.qvel[rD:rD + 6], dx.xpos[me][2:3], rel,
                                dx.qvel[(self._BrD if me == self._At else self._ArD):
                                        (self._BrD if me == self._At else self._ArD) + 3]])

    def _obsA_(self, dx): return self._obs_for(dx, self._Aqa, self._Ada, self._ArD, self._At, self._Bt)
    def _obsB_(self, dx): return self._obs_for(dx, self._Bqa, self._Bda, self._BrD, self._Bt, self._At)

    def reset(self, rng):
        rng, k = jax.random.split(rng)
        qpos = self._q0.at[7:].add(jax.random.uniform(k, (self._q0.shape[0] - 7,), minval=-0.03, maxval=0.03))
        dx = mjx.forward(self._mx, mjx.make_data(self._mx).replace(qpos=qpos))
        return State(dx, self._obsA_(dx), jnp.zeros(()), jnp.zeros(()), {}, {"rng": rng})

    def _opp_action(self, dx, rng):
        if self._opp_infer is None:
            return jnp.zeros(self._nuB)                       # B passive (round 0, empty HoF)
        act, _ = self._opp_infer(self._opp_params, self._obsB_(dx), rng)
        return jnp.clip(act, -1, 1)

    def _sparc(self, dx):
        pen = jnp.maximum(0.0, -dx.contact.dist)
        g0, g1 = dx.contact.geom[:, 0], dx.contact.geom[:, 1]
        dealt = jnp.sum(pen * ((self._Aw[g0] & self._Bb[g1]) | (self._Aw[g1] & self._Bb[g0])))
        taken = jnp.sum(pen * ((self._Bw[g0] & self._Ab[g1]) | (self._Bw[g1] & self._Ab[g0])))
        rel = (dx.xpos[self._Bt] - dx.xpos[self._At])[:2]; n = jnp.linalg.norm(rel) + 1e-6
        toward = jnp.dot(dx.qvel[self._ArD:self._ArD + 2], rel) / n
        clos, flee = jnp.maximum(0.0, toward), jnp.maximum(0.0, -toward)
        return (6.0 * (jnp.clip(dealt / DAMAGE_REF, 0, 1) - jnp.clip(taken / DAMAGE_REF, 0, 1))
                + 5.0 * (jnp.clip(clos / 2, 0, 1) - jnp.clip(flee / 2, 0, 1)))

    def step(self, state, action):
        rng, ork = jax.random.split(state.info["rng"])
        a = jnp.clip(action, -1, 1)
        b = self._opp_action(state.pipeline_state, ork)
        ctrl = jnp.zeros(self._nu).at[self._actA].set(a).at[self._actB].set(b)
        dx = state.pipeline_state.replace(ctrl=ctrl)
        dx = jax.lax.fori_loop(0, self._fs, lambda i, d: mjx.step(self._mx, d), dx)
        reward = self._sparc(dx) + 0.1                       # +alive
        done = jnp.where((dx.xpos[self._At][2] < 0.12) | (dx.xpos[self._Bt][2] < 0.12), 1.0, 0.0)
        # MERGE rng into info (don't replace it) — brax's AutoReset/Episode wrappers add
        # their own keys to info at reset; replacing the dict changes the scan carry pytree.
        return state.replace(pipeline_state=dx, obs=self._obsA_(dx), reward=reward, done=done,
                             info={**state.info, "rng": rng})


def make_opponent(params, obs_size, act_size):
    """Reconstruct a frozen inference fn from saved params (for the league opponent)."""
    net = ppo_networks.make_ppo_networks(obs_size, act_size,
                                         preprocess_observations_fn=running_statistics.normalize)
    inf = ppo_networks.make_inference_fn(net)
    return lambda p, o, k: inf(p, deterministic=True)(o, k)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--rounds", type=int, default=4)
    ap.add_argument("--steps", type=int, default=3_000_000)
    ap.add_argument("--envs", type=int, default=2048)
    ap.add_argument("--tiny", action="store_true")
    args = ap.parse_args()
    if args.tiny:
        args.rounds, args.steps, args.envs = 2, 8_000, 256
    rng = np.random.default_rng(0)

    hof = []                          # Hall of Fame: list of (params, obs, act) opponent snapshots
    probe = SelfPlayEnv()             # to read obs/act sizes (B passive)
    obsA, actA = probe.observation_size, probe.action_size
    obsB = probe._obsB; actB = probe._nuB
    last = {"r": float("nan"), "step": 0}
    for rd in range(args.rounds):
        # sample an opponent from the HoF (league); round 0 = passive B
        if hof:
            opp = hof[rng.integers(len(hof))]
            env = SelfPlayEnv(opp_infer=make_opponent(opp[0], opp[1], opp[2]), opp_params=opp[0])
            src = f"HoF[{len(hof)}]"
        else:
            env = SelfPlayEnv(); src = "passive"
        t0 = time.time()
        def prog(s, m): last.update(r=float(m.get("eval/episode_reward", 0)), step=int(s))
        n_eval = 2 if args.tiny else max(4, args.steps // 1_000_000)
        make_inf, params, _ = ppo.train(
            environment=env, num_timesteps=args.steps, num_evals=n_eval, episode_length=300,
            num_envs=args.envs, batch_size=256 if args.tiny else 1024,
            num_minibatches=8 if args.tiny else 16, unroll_length=5 if args.tiny else 20,
            num_updates_per_batch=4, learning_rate=3e-4, entropy_cost=1e-2, discounting=0.97,
            reward_scaling=1.0, normalize_observations=True, seed=rd, progress_fn=prog)
        hof.append((params, obsA, actA))            # A's obs/act (next round B uses A's policy in B's slot if sizes match)
        pickle.dump(params, open(OUT / f"selfplay_A_r{rd}.pkl", "wb"))
        METRIC(stage="league_round", round=rd, opponent=src, sparc=f"{last['r']:.2f}",
               train_s=f"{time.time()-t0:.1f}", hof=len(hof))
        print(f"[league] round {rd} vs {src}: A SPARC-return {last['r']:.2f} "
              f"({time.time()-t0:.0f}s); HoF size {len(hof)}", flush=True)

    # NOTE: opponent obs/act sizes (B body = attacker, 5-leg) differ from A's (4-leg) — a
    # cross-body league needs a policy per body shape. The HoF here archives A snapshots;
    # the resilient default opponent is the passive/heuristic B until a B-shaped policy is
    # trained. The two-learner symmetric case = train a B policy with B's obs/act too.
    print("PROVEN: MJX self-play league runs — A trains vs Hall-of-Fame opponents on two "
          "GENERATED bodies sharing one morphology space; SPARC differential is the reward.")


if __name__ == "__main__":
    main()
