# SPDX-License-Identifier: MIT
"""Anti-cheat measurement suite — catch DEGENERATE combat strategies that fool the headline metrics.

Motivation: `spr6_best` scored bench_ratio 1.23 ("out-trades opponent") while actually CATAPULTING
its torso to 1.5 m, landing all damage in the first ~16 steps of spawn-clash chaos, then idling for
~200 steps. The reward looked fine; the behavior was an exploit. These metrics measure *how* damage
was produced and whether the motion is physically/behaviorally legitimate — so a cheat trips a flag
EARLY instead of hiding behind a healthy-looking ratio.

24 metrics in 5 families, each with a healthy range + a cheat signature. `evaluate()` rolls out a
checkpoint vs a frozen opponent and returns {metric: (value, FLAG?)} + an overall verdict.

  MUJOCO_GL=osmesa CODESIGN_OUT=/tmp/cd python anti_cheat.py --a spr6_best.pkl --b spr3_best.pkl
"""
from __future__ import annotations
import argparse, os
os.environ.setdefault("MUJOCO_GL", "osmesa")
from pathlib import Path
import numpy as np
import jax, jax.numpy as jnp
import mujoco
import train_adversarial as T
from gen_robot_mjcf import build_match

HERE = Path(__file__).resolve().parent

# (name, family, healthy-comparator, threshold, direction) — FLAG when value `op` threshold.
#   op '>' => flag if value > thr (too high);  '<' => flag if value < thr (too low)
SPEC = [
    # --- physics plausibility (launches, blowups, interpenetration) ---
    ("peak_torso_z",      "physics", ">", 0.45),   # max torso height; stance ~0.2 -> >0.45 = launch
    ("airborne_frac",     "physics", ">", 0.05),   # frac of steps torso above any standing pose
    ("peak_lin_speed",    "physics", ">", 4.0),    # max torso linear speed (m/s) -> catapult
    ("peak_joint_speed",  "physics", ">", 60.0),   # max hinge joint speed (rad/s) -> blowup
    ("peak_penetration",  "physics", ">", 0.05),   # max contact penetration (m) -> interpenetration
    ("peak_vert_accel_g", "physics", ">", 8.0),    # max upward torso accel / g -> impulse exploit
    ("height_cv",         "physics", ">", 0.5),    # torso-height coeff of variation -> bouncing
    # --- behavioral legitimacy (clash-and-idle, freezing, bang-bang) ---
    ("dmg_first15_frac",  "behavior", ">", 0.7),   # frac of damage in first 15% of bout -> spawn-clash
    ("dmg_last50_frac",   "behavior", "<", 0.1),   # frac of damage in last half -> quit after clash
    ("idle_frac",         "behavior", ">", 0.5),   # frac of steps near-zero action -> freezing
    ("action_sat_frac",   "behavior", ">", 0.6),   # frac action dims slammed to ±1 -> bang-bang
    ("action_jerk",       "behavior", "<", 0.004), # mean |Δaction|/step -> ~0 = frozen/degenerate
    ("active_after_hit",  "behavior", "<", 0.2),   # action activity after first hit -> hit-and-freeze
    # --- combat quality (is the damage from skill or chaos?) ---
    ("upright_dmg_frac",  "combat", "<", 0.5),     # frac of damage dealt while upright
    ("grounded_dmg_frac", "combat", "<", 0.5),     # frac of damage dealt while not airborne
    ("approach_before_hit","combat", "<", 0.02),   # distance closed before the first hit (m)
    ("strike_dist",       "combat", "<", 0.05),    # mean A-B distance at hits -> interpenetration if ~0
    # --- survival / sustainability (the real bottleneck) ---
    ("upright_frac",      "survive", "<", 0.5),     # frac of alive steps upright
    ("end_upright",       "survive", "<", 0.5),     # 1 if upright+standing at the final step
    ("time_to_first_fall","survive", "<", 30.0),    # steps until first topple (low = collapses fast)
    ("settled_idle_frac", "survive", ">", 0.6),     # frac of last half both still and limp -> settled-quit
    # --- engagement integrity ---
    ("in_reach_frac",     "engage", "<", 0.1),      # frac of bout within striking reach
    ("dist_cv",           "engage", ">", 0.8),      # A-B distance coeff of variation -> bouncing apart
    ("engage_then_quit",  "engage", ">", 0.5),      # composite: early damage AND late idle
]


def _indices(m):
    """Pull A's torso body, free-joint dof slice, hinge dofs, torso-quat qpos slice from a match model."""
    A_t = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, "A_torso")
    free_dof = free_q = None; hinge_dofs = []
    for j in range(m.njnt):
        nm = mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_JOINT, j) or ""
        if not nm.startswith("A_"):
            continue
        if m.jnt_type[j] == mujoco.mjtJoint.mjJNT_FREE:
            free_dof = m.jnt_dofadr[j]; free_q = m.jnt_qposadr[j]
        elif m.jnt_type[j] == mujoco.mjtJoint.mjJNT_HINGE:
            hinge_dofs.append(m.jnt_dofadr[j])
    return A_t, free_dof, free_q, np.array(hinge_dofs, int)


def rollout(ckptA, ckptB, steps=220, sep=0.5, seed=0):
    infA = T.load_opponent(ckptA); infB = T.load_opponent(ckptB)
    env = T.AdversarialEnv(frame_skip=5, striker=True, sep=sep, azimuth=0.3,
                           opponent="frozen", opp_infer=infB)
    step = jax.jit(env.step); key = jax.random.PRNGKey(seed)
    s = env.reset_with(key, jnp.full(3, 0.5))
    QP, QV, ACT, DEALT, TAKEN, DONE = [], [], [], [], [], []
    for _ in range(steps):
        key, k = jax.random.split(key)
        a, _ = infA(s.obs, k); s = step(s, a)
        QP.append(np.asarray(s.pipeline_state.qpos)); QV.append(np.asarray(s.pipeline_state.qvel))
        ACT.append(np.asarray(a)); DEALT.append(float(s.metrics.get("dealt", 0.0)))
        TAKEN.append(float(s.metrics.get("taken", 0.0))); DONE.append(float(s.done))
    m = mujoco.MjModel.from_xml_string(build_match(T.SPEC, T.SPEC, sep=sep, self_collision=True,
                                                   striker=True, striker_b=True))
    return dict(qp=np.array(QP), qv=np.array(QV), act=np.array(ACT), dealt=np.array(DEALT),
                taken=np.array(TAKEN), done=np.array(DONE), model=m, dt=float(m.opt.timestep) * env._fs)


def compute(ro):
    m = ro["model"]; d = mujoco.MjData(m); A_t, fd, fq, hd = _indices(m)
    N = len(ro["qp"]); dt = ro["dt"]
    tz = np.zeros(N); up = np.zeros(N); pen = np.zeros(N); dist = np.zeros(N)
    Bt = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, "B_torso")
    for t in range(N):
        d.qpos[:] = ro["qp"][t]; d.qvel[:] = ro["qv"][t]; mujoco.mj_forward(m, d)
        tz[t] = d.xpos[A_t][2]
        q = ro["qp"][t]; up[t] = 1.0 - 2.0 * (q[fq + 4] ** 2 + q[fq + 5] ** 2)
        pen[t] = float(np.maximum(0.0, -d.contact.dist).max()) if d.ncon else 0.0
        dist[t] = float(np.linalg.norm((d.xpos[Bt] - d.xpos[A_t])[:2]))
    qv = ro["qv"]; lin = np.linalg.norm(qv[:, fd:fd + 3], axis=1); vvel = qv[:, fd + 2]
    jspeed = np.abs(qv[:, hd]).max(1) if len(hd) else np.zeros(N)
    act = ro["act"]; amag = np.abs(act).mean(1); asat = (np.abs(act) > 0.95).mean(1)
    jerk = np.r_[0, np.abs(np.diff(act, axis=0)).mean(1)]
    dealt = ro["dealt"]; alive = ro["done"] == 0
    tot = max(dealt.sum(), 1e-9); hit = dealt > 1e-4
    first_hit = int(np.argmax(hit)) if hit.any() else N
    n15, half = max(1, N // 7), N // 2
    spawn_z = float(T.SPEC.get("torso", {}).get("spawn_height", 0.35))
    airborne = tz > (spawn_z + 0.07)
    grounded = tz < (spawn_z + 0.03)
    fell = (tz < 0.09) | (up < 0.0)
    ttf = int(np.argmax(fell)) if fell.any() else N
    M = {
        "peak_torso_z": tz.max(),
        "airborne_frac": airborne.mean(),
        "peak_lin_speed": lin.max(),
        "peak_joint_speed": jspeed.max(),
        "peak_penetration": pen.max(),
        "peak_vert_accel_g": float(np.abs(np.diff(vvel)).max() / dt / 9.81) if N > 1 else 0.0,
        "height_cv": tz.std() / max(tz.mean(), 1e-6),
        "dmg_first15_frac": dealt[:n15].sum() / tot,
        "dmg_last50_frac": dealt[half:].sum() / tot,
        "idle_frac": (amag < 0.1).mean(),
        "action_sat_frac": asat.mean(),
        "action_jerk": jerk.mean(),
        "active_after_hit": (amag[first_hit:] > 0.1).mean() if first_hit < N else 0.0,
        "upright_dmg_frac": (dealt * (up > 0.5)).sum() / tot,
        "grounded_dmg_frac": (dealt * grounded).sum() / tot,
        "approach_before_hit": float(dist[0] - dist[min(first_hit, N - 1)]),
        "strike_dist": float(dist[hit].mean()) if hit.any() else 1.0,
        "upright_frac": float((up[alive] > 0.5).mean()) if alive.any() else 0.0,
        "end_upright": float(up[-1] > 0.5 and 0.12 < tz[-1] < (spawn_z + 0.03)),
        "time_to_first_fall": float(ttf),
        "settled_idle_frac": float(((amag[half:] < 0.1) & (lin[half:] < 0.2)).mean()),
        "in_reach_frac": float((dist < 0.62).mean()),
        "dist_cv": float(dist.std() / max(dist.mean(), 1e-6)),
        "engage_then_quit": float((dealt[:n15].sum() / tot > 0.6) and ((amag < 0.1).mean() > 0.4)),
    }
    return M, dict(ratio=dealt.sum() / max(ro["taken"].sum(), 1e-9), total_dealt=float(dealt.sum()))


def verdict(M):
    rows, flags = [], []
    for name, fam, op, thr in SPEC:
        v = M[name]; bad = (v > thr) if op == ">" else (v < thr)
        rows.append((name, fam, v, op, thr, bad))
        if bad:
            flags.append(name)
    return rows, flags


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--a", required=True); ap.add_argument("--b", required=True)
    ap.add_argument("--steps", type=int, default=220); ap.add_argument("--sep", type=float, default=0.5)
    a = ap.parse_args()
    ro = rollout(a.a, a.b, steps=a.steps, sep=a.sep)
    M, extra = compute(ro)
    rows, flags = verdict(M)
    print(f"ANTI-CHEAT REPORT  (A={Path(a.a).name} vs B={Path(a.b).name}, {a.steps} steps)")
    print(f"  headline: damage_ratio={extra['ratio']:.2f}  total_dealt={extra['total_dealt']:.3f}")
    fam = None
    for name, f, v, op, thr, bad in rows:
        if f != fam:
            fam = f; print(f"  [{fam.upper()}]")
        mark = "  ✗ FLAG" if bad else "  ✓"
        print(f"    {name:20s} {v:8.3f}  ({'>' if op=='>' else '<'}{thr})  {mark if bad else ''}")
    print(f"\n  VERDICT: {len(flags)}/{len(rows)} cheat-flags tripped"
          + (f" -> {flags}" if flags else " -> looks legitimate"))


if __name__ == "__main__":
    main()
