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
OUT = Path(os.environ.get("CODESIGN_OUT", "/root/proj/out")); OUT.mkdir(parents=True, exist_ok=True)
LOCO_OBS = 38; DAMAGE_REF = 0.05


def METRIC(**kw):
    print("METRIC " + " ".join(f"{k}={v}" for k, v in kw.items()), flush=True)


class AdversarialEnv(Env):
    def __init__(self, frame_skip=5, shaping=1.0, sep=1.0):
        # shaping = weight on the dense close→strike potential (anneal to 0 to confirm the
        # policy fights without the crutch); sep = start separation (curriculum: smaller=easier).
        self._shaping = float(shaping); self._sep = float(sep)
        m = mujoco.MjModel.from_xml_string(build_match(SPEC, SPEC, sep))   # default body x2, close
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
        # one design codec (design_codec.apply_fast); hinge_mask keeps the spring off
        # the free-joint root. Identical maps to the universal env (worldbody mass 0).
        from design_codec import apply_fast
        return apply_fast(self._mx, d, hinge_mask=self._hinge)

    def _obs(self, dx, d):
        loco = jnp.concatenate([dx.qpos[self._Aqa], dx.qvel[self._Ada], dx.xquat[self._At],
                                dx.qvel[self._ArD:self._ArD + 6], dx.xpos[self._At][2:3], d])
        opp = jnp.concatenate([dx.xpos[self._Bt] - dx.xpos[self._At], dx.qvel[self._BrD:self._BrD + 3]])
        return jnp.concatenate([loco, opp])

    _MET0 = None
    def _metrics0(self):
        return {"dealt": jnp.zeros(()), "taken": jnp.zeros(()), "closing": jnp.zeros(()),
                "fleeing": jnp.zeros(()), "sparc": jnp.zeros(()), "dist": jnp.zeros(())}

    def reset(self, rng):
        rng, dr, nr = jax.random.split(rng, 3)
        d = jax.random.uniform(dr, (3,))
        qpos = self._q0.at[7:].add(jax.random.uniform(nr, (self._q0.shape[0] - 7,), minval=-0.03, maxval=0.03))
        dx = mjx.forward(self._design_model(d), mjx.make_data(self._mx).replace(qpos=qpos))
        return State(dx, self._obs(dx, d), jnp.zeros(()), jnp.zeros(()), self._metrics0(), {"design": d})

    def reset_with(self, rng, design):
        """Reset with a GIVEN design (eval) — for the walker-vs-fighter validation."""
        nr, _ = jax.random.split(rng)
        qpos = self._q0.at[7:].add(jax.random.uniform(nr, (self._q0.shape[0] - 7,), minval=-0.03, maxval=0.03))
        dx = mjx.forward(self._design_model(design), mjx.make_data(self._mx).replace(qpos=qpos))
        return State(dx, self._obs(dx, design), jnp.zeros(()), jnp.zeros(()), self._metrics0(), {"design": design})

    def step(self, state, action):
        d = state.info["design"]; mxd = self._design_model(d)
        ctrl = jnp.zeros(self._nu).at[self._actA].set(jnp.clip(action, -1, 1))
        dx = state.pipeline_state.replace(ctrl=ctrl)
        dx = jax.lax.fori_loop(0, self._fs, lambda i, x: mjx.step(mxd, x), dx)
        pen = jnp.maximum(0.0, -dx.contact.dist); g0, g1 = dx.contact.geom[:, 0], dx.contact.geom[:, 1]
        dealt = jnp.sum(pen * ((self._Aleg[g0] & self._Bbody[g1]) | (self._Aleg[g1] & self._Bbody[g0])))
        taken = jnp.sum(pen * ((self._Bleg[g0] & self._Abody[g1]) | (self._Bleg[g1] & self._Abody[g0])))
        dealt_f = jnp.clip(dealt / DAMAGE_REF, 0, 1); taken_f = jnp.clip(taken / DAMAGE_REF, 0, 1)
        rel = (dx.xpos[self._Bt] - dx.xpos[self._At])[:2]; dist = jnp.linalg.norm(rel); n = dist + 1e-6
        toward = jnp.dot(dx.qvel[self._ArD:self._ArD + 2], rel) / n
        clos = jnp.clip(toward / 2, 0, 1); flee = jnp.clip(-toward / 2, 0, 1)
        up = 1.0 - 2.0 * (dx.qpos[4] ** 2 + dx.qpos[5] ** 2)
        # the real SPARC objective (force/penetration-weighted damage + aggression):
        sparc = 6.0 * (dealt_f - taken_f) + 5.0 * (clos - flee)
        # dense close→strike SHAPING (annealed via self._shaping; legs-as-weapons so getting
        # close + a limb on B already scores): negative-distance potential + a hit accelerator.
        shaped = self._shaping * (-0.25 * dist + 3.0 * dealt_f)
        reward = sparc + shaped + 0.3 * up + 0.1                   # + upright anchor + alive
        done = jnp.where(dx.xpos[self._At][2] < 0.18, 1.0, 0.0)
        metrics = {"dealt": dealt_f, "taken": taken_f, "closing": clos, "fleeing": flee,
                   "sparc": sparc, "dist": dist}
        return state.replace(pipeline_state=dx, obs=self._obs(dx, d), reward=reward, done=done,
                             metrics=metrics)


def warm_start(path, obs_dim):
    """Pad Stage-A params (obs 38, a (normalizer, policy, value) tuple) to Stage-B obs:
    normalizer fields + every net's input-layer kernel grow 38->44 (opponent inputs
    init ~0). Best-effort with a scratch fall-back."""
    try:
        parts = list(pickle.load(open(path, "rb")))      # (normalizer, policy_dict, value_dict, ...)
        norm, nets = parts[0], parts[1:]
        pad = obs_dim - LOCO_OBS
        c = norm.count                                   # brax UInt64 = {hi, lo}: value = hi*2^32 + lo
        cval = float(jnp.asarray(c.hi)) * (2.0 ** 32) + float(jnp.asarray(c.lo))
        nkw = {}
        for fn in ("mean", "std", "summed_variance"):
            v = getattr(norm, fn, None)
            if hasattr(v, "ndim") and v.ndim >= 1 and v.shape[0] == LOCO_OBS:
                # new (opponent) dims start standardized: mean 0, std 1, summed_variance=count (var~1)
                fill = (jnp.zeros(pad) if fn == "mean" else
                        jnp.ones(pad) if fn == "std" else
                        jnp.full((pad,), max(cval, 1.0)))
                nkw[fn] = jnp.concatenate([v, fill])
        norm = norm.replace(**nkw)
        pad_leaf = lambda x: (jnp.concatenate([x, jnp.zeros((pad,) + x.shape[1:])], 0)
                              if (hasattr(x, "ndim") and x.ndim >= 1 and x.shape[0] == LOCO_OBS) else x)
        nets = [jax.tree_util.tree_map(pad_leaf, n) for n in nets]
        print(f"WARM-START ok: padded {LOCO_OBS}->{obs_dim} (count={cval:.0f}, normalizer + {len(nets)} nets)", flush=True)
        return tuple([norm] + nets)
    except Exception as e:
        print(f"warm-start failed ({type(e).__name__}: {e}) -> training Stage B from scratch", flush=True)
        return None


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--steps", type=int, default=12_000_000)
    ap.add_argument("--envs", type=int, default=2048); ap.add_argument("--resume", default=None)
    ap.add_argument("--batch", type=int, default=1024); ap.add_argument("--minibatches", type=int, default=16)
    ap.add_argument("--unroll", type=int, default=20); ap.add_argument("--evals", type=int, default=0)
    ap.add_argument("--shaping", type=float, default=1.0, help="dense close→strike shaping weight (anneal to 0)")
    ap.add_argument("--sep", type=float, default=1.0, help="start separation (curriculum: smaller=easier)")
    ap.add_argument("--tag", default="adv", help="checkpoint/metrics tag")
    ap.add_argument("--tiny", action="store_true", help="lightweight plumbing run (e2e harness)")
    args = ap.parse_args()
    if args.tiny:
        args.steps, args.envs = 8_000, 256
        args.batch, args.minibatches, args.unroll = 256, 8, 5
        args.evals = 2
    n_eval = args.evals or max(6, args.steps // 1_000_000)
    t_env = time.time(); env = AdversarialEnv(shaping=args.shaping, sep=args.sep)
    METRIC(stage="adv_env_build", t_s=f"{time.time()-t_env:.1f}",
           obs=env.observation_size, act=env.action_size)
    print(f"adversarial env: obs={env.observation_size} act(A)={env.action_size}", flush=True)
    restore = warm_start(args.resume, env.observation_size) if args.resume and os.path.exists(args.resume) else None
    METRIC(stage="warm_start", ok=int(restore is not None),
           resume=os.path.basename(args.resume) if args.resume else "none")
    import json
    t0 = time.time(); csv = OUT / "adv_metrics.csv"; csv.write_text("step,reward,sec\n")
    fjson = OUT / "fight_metrics.jsonl"; fjson.write_text("")          # F0: the six trackers
    tm = {"first_eval": None}; last = {"r": float("nan"), "step": 0, "dealt": 0.0, "taken": 0.0}
    def g(m, k): return float(m.get(f"eval/episode_{k}", 0.0))
    def prog(s, m):
        if tm["first_eval"] is None: tm["first_eval"] = time.time() - t0
        r = g(m, "reward"); dealt = g(m, "dealt"); taken = g(m, "taken")
        clos = g(m, "closing"); flee = g(m, "fleeing"); sparc = g(m, "sparc"); dist = g(m, "dist")
        open(csv, "a").write(f"{int(s)},{r:.3f},{time.time()-t0:.0f}\n")
        rec = dict(step=int(s), sec=round(time.time()-t0, 0), reward=round(r, 3), sparc=round(sparc, 3),
                   dealt=round(dealt, 4), taken=round(taken, 4), closing=round(clos, 4),
                   fleeing=round(flee, 4), dist=round(dist, 3), tag=args.tag,
                   shaping=args.shaping, sep=args.sep)
        open(fjson, "a").write(json.dumps(rec) + "\n")
        last.update(r=r, step=int(s), dealt=dealt, taken=taken)
        print(f"  [{args.tag}] step {int(s):>9,} sparc {sparc:6.2f} dealt {dealt:.3f} taken {taken:.3f} "
              f"close {clos:.2f} flee {flee:.2f} dist {dist:.2f} ({time.time()-t0:.0f}s)", flush=True)
    def ck(*a):
        try: pickle.dump(a[-1], open(OUT / f"{args.tag}_ckpt.pkl", "wb"))
        except Exception: pass
    ppo.train(environment=env, num_timesteps=args.steps, num_evals=n_eval,
              episode_length=300, num_envs=args.envs, batch_size=args.batch,
              num_minibatches=args.minibatches, unroll_length=args.unroll, num_updates_per_batch=4,
              learning_rate=3e-4, entropy_cost=1e-2, discounting=0.97, reward_scaling=1.0,
              normalize_observations=True, seed=0, progress_fn=prog, policy_params_fn=ck,
              restore_params=restore)
    train_s = time.time() - t0
    ratio = last["dealt"] / max(last["taken"], 1e-6)
    competent = last["dealt"] > last["taken"] and last["dealt"] > 0.02
    METRIC(stage="fighter_train", train_s=f"{train_s:.1f}", compile_s=f"{tm['first_eval'] or 0:.1f}",
           env_steps=last["step"], throughput=f"{last['step']/max(train_s,1e-6):.0f}",
           final_sparc=f"{last['r']:.2f}", dealt=f"{last['dealt']:.4f}", taken=f"{last['taken']:.4f}",
           dealt_taken_ratio=f"{ratio:.2f}", competent=int(competent), warm=int(restore is not None))
    print(f"FIGHTER: final dealt {last['dealt']:.4f} vs taken {last['taken']:.4f} (ratio {ratio:.2f}); "
          f"competent (dealt>taken & lands hits) = {competent}. Decomposition, not the scalar, "
          f"is the verdict (a survivor has dealt≈0).", flush=True)


if __name__ == "__main__":
    main()
