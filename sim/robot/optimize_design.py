# SPDX-License-Identifier: MIT
"""Co-design: optimize the robot's PARAMETER space (not just its control).

An outer cross-entropy-method (CEM) loop proposes bodies; each is generated
(robot.toml -> MJCF) and scored by a physics-based proxy fitness computed in
MuJoCo. The proxy stands in for the real objective (the RL policy's dodge return)
so the whole co-design loop is provable locally with NO GPU/policy: it rewards a
body that can STAND within its motor torque, RETRACT a foot high (dodge clearance),
and stay LIGHT. On a GPU, swap `proxy_fitness` for the trained policy's return and
this becomes true morphology+control co-optimization (CEM/CMA-ES outer, RL inner).

  python optimize_design.py [--pop 16 --gens 10 --seed 0]
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

os.environ.setdefault("MUJOCO_GL", "osmesa")

import mujoco  # noqa: E402
import numpy as np  # noqa: E402

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from gen_robot_mjcf import build_mjcf, load_spec  # noqa: E402
from gen_mesh_robot_mjcf import MAX_ROBOT_MASS_KG  # noqa: E402

SPEC = load_spec(HERE / "robot.toml")

# continuous design space: (name, lo, hi, where it goes in the spec)
PARAMS = [
    ("thigh_len", 0.14, 0.28, ("leg_defaults", "thigh_len")),
    ("calf_len", 0.14, 0.28, ("leg_defaults", "calf_len")),
    ("gear", 1.0, 6.0, ("actuator", "gear")),
    ("joint_stiffness", 0.0, 25.0, ("leg_defaults", "joint_stiffness")),
    ("torso_mass", 0.35, 0.5479633165, ("torso", "mass")),
]
STAND_TARGET = {"abd": 0.0, "flex": 0.8, "knee": -1.5}    # a crouch stance


def to_overrides(x: np.ndarray) -> dict:
    """Map a design vector (in real units) to a generator override dict."""
    ov: dict = {}
    for v, (_, _, _, (sec, key)) in zip(x, PARAMS):
        ov.setdefault(sec, {})[key] = float(v)
    return ov


def _act_targets(model):
    """Per actuator: (joint qpos addr, dof addr, PD target) from the joint name."""
    out = []
    for a in range(model.nu):
        j = int(model.actuator_trnid[a, 0])
        name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_JOINT, j)
        kind = name.rsplit("_", 1)[-1]
        out.append((model.jnt_qposadr[j], model.jnt_dofadr[j],
                    STAND_TARGET.get(kind, 0.0)))
    return out


def _retract_clearance(model, data):
    """Kinematic max foot-tip height the first leg can reach (dodge clearance)."""
    foot = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM,
                             SPEC["leg"][0]["name"] + "_foot")
    if foot < 0:
        return 0.0
    best = 0.0
    d = SPEC["leg_defaults"]
    for flex in d["flex_range"]:
        for knee in d["knee_range"]:
            mujoco.mj_resetData(model, data)
            data.qpos[2] = SPEC["torso"]["spawn_height"]
            for a in range(model.nu):
                j = int(model.actuator_trnid[a, 0])
                nm = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_JOINT, j)
                if nm.startswith(SPEC["leg"][0]["name"] + "_"):
                    adr = model.jnt_qposadr[j]
                    data.qpos[adr] = flex if nm.endswith("flex") else (
                        knee if nm.endswith("knee") else 0.0)
            mujoco.mj_forward(model, data)
            best = max(best, float(data.geom_xpos[foot][2]))
    return best


def proxy_fitness(x: np.ndarray) -> float:
    """Build the body and score it: stand within torque + retract high - mass."""
    try:
        model = mujoco.MjModel.from_xml_string(build_mjcf(SPEC, to_overrides(x)))
    except Exception:
        return -10.0
    data = mujoco.MjData(model)
    # 1. stand: torque-limited PD hold of the crouch, then measure height + upright
    mujoco.mj_resetData(model, data)
    data.qpos[2] = SPEC["torso"]["spawn_height"]
    mujoco.mj_forward(model, data)
    tg = _act_targets(model)
    for _ in range(250):
        for a, (qa, da, tgt) in enumerate(tg):
            data.ctrl[a] = np.clip(4.0 * (tgt - data.qpos[qa])
                                   - 0.3 * data.qvel[da], -1.0, 1.0)
        mujoco.mj_step(model, data)
    if not np.isfinite(data.qpos).all():
        return -10.0
    h = float(data.qpos[2])
    up = 1.0 - 2.0 * (data.qpos[4] ** 2 + data.qpos[5] ** 2)
    stand = np.clip(h / 0.30, 0, 1) * np.clip(up, 0, 1)
    # 2. dodge clearance: how high the foot retracts above the strike band (0.12 m)
    clear = np.clip((_retract_clearance(model, data) - 0.12) / 0.20, 0, 1)
    # 3. mass penalty (lighter -> faster dodge / less power)
    mass = float(model.body_mass.sum())
    if mass > MAX_ROBOT_MASS_KG + 1e-9:
        return -10.0
    return 2.0 * stand + 1.5 * clear - 0.5 * (mass / MAX_ROBOT_MASS_KG)


def cem(pop, gens, seed, fitness=proxy_fitness):
    """Cross-entropy method in normalized [0,1] space; returns best (x, fitness).
    `fitness(x_real)` is the proxy by default; `--fitness policy` swaps in the trained-
    policy return (codesign_gpu.policy_fitness_direct) — same loop, real objective."""
    rng = np.random.default_rng(seed)
    lo = np.array([p[1] for p in PARAMS]); hi = np.array([p[2] for p in PARAMS])
    denorm = lambda u: lo + np.clip(u, 0, 1) * (hi - lo)
    mean = np.full(len(PARAMS), 0.5); std = np.full(len(PARAMS), 0.30)
    n_elite = max(2, pop // 4)
    best_x, best_f = None, -1e9
    for g in range(gens):
        pcent = np.clip(mean + std * rng.standard_normal((pop, len(PARAMS))), 0, 1)
        fits = np.array([fitness(denorm(u)) for u in pcent])
        elite = pcent[np.argsort(fits)[-n_elite:]]
        mean, std = elite.mean(0), elite.std(0) + 1e-3
        gi = int(np.argmax(fits))
        if fits[gi] > best_f:
            best_f, best_x = fits[gi], denorm(pcent[gi])
        print(f"  gen {g:2d}: best={fits.max():8.3f}  mean={fits.mean():8.3f}  "
              f"(running best {best_f:.3f})")
    return best_x, best_f


def _policy_fitness_factory(budget, restore):
    """Lazy GPU import: a fitness(x_real) -> trained-policy return for that body."""
    import time
    from codesign_gpu import policy_fitness_direct
    def f(x_real):
        t = time.time()
        r = policy_fitness_direct(to_overrides(x_real), K=budget, restore_path=restore)
        print(f"    [policy-fitness] {np.round(x_real,3)} -> return {r:.2f} "
              f"({time.time()-t:.0f}s/candidate)")   # the per-candidate wall-clock #2 removes
        return r
    return f


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pop", type=int, default=16)
    ap.add_argument("--gens", type=int, default=10)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--fitness", choices=["proxy", "policy"], default="proxy",
                    help="proxy = static stand/clearance/mass (CPU, default); "
                         "policy = trained-policy return per candidate (GPU, Phase 2 direct)")
    ap.add_argument("--policy-budget", type=int, default=150_000,
                    help="fine-tune steps per candidate when --fitness policy")
    ap.add_argument("--restore", default=str(Path(os.environ.get("CODESIGN_OUT", ".")) / "baseline.pkl"),
                    help="baseline checkpoint to warm-start each candidate from")
    args = ap.parse_args()

    fitfn = proxy_fitness
    if args.fitness == "policy":
        fitfn = _policy_fitness_factory(args.policy_budget, args.restore)

    default_x = np.array([SPEC["leg_defaults"]["thigh_len"],
                          SPEC["leg_defaults"]["calf_len"],
                          SPEC["actuator"]["gear"],
                          SPEC["leg_defaults"]["joint_stiffness"],
                          SPEC["torso"]["mass"]])
    f0 = fitfn(default_x)
    print(f"default design fitness ({args.fitness}): {f0:.3f}\nCEM (pop {args.pop}, {args.gens} gens):")
    best_x, best_f = cem(args.pop, args.gens, args.seed, fitness=fitfn)
    print("\noptimized design vs default:")
    for (name, lo, hi, _), v0, v1 in zip(PARAMS, default_x, best_x):
        print(f"  {name:16s} {v0:7.3f} -> {v1:7.3f}")
    print(f"\nPROVEN: co-design loop ran; fitness {f0:.3f} -> {best_f:.3f} "
          f"(+{best_f - f0:.3f}) over {args.gens} generations. "
          f"Swap proxy_fitness -> policy dodge-return on GPU for full co-optimization.")
    sys.exit(0 if best_f >= f0 else 1)


if __name__ == "__main__":
    main()
