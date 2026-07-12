# SPDX-License-Identifier: MIT
"""RS4 — teacher -> student online adaptation (RMA), the biggest currently-absent gap.

Zero-shot transfer needs a policy that ADAPTS to whichever world it is in. RMA does this
in two stages:
  * TEACHER: trained with privileged obs = normal obs + the world EXTRINSICS (the sampled
    dynamics: damping/friction/latency/motor params). It can be optimal because it knows
    the world.
  * STUDENT: deployable — it never sees the extrinsics; instead a HISTORY ENCODER over
    recent (obs, action) infers them online, and the student acts on the inferred
    extrinsics. The adaptation gap = student return vs teacher return on HELD-OUT worlds.

The design-conditioned universal policy (Phase 3) is the *body* half of the extrinsics;
RS4 adds the *dynamics* half and makes it inferred-not-given. [RMA 2107.04034]

This module proves the RMA MECHANISM sim-to-sim on CPU: a controllable 2nd-order plant
with a hidden dynamics parameter z. The teacher is the z-optimal controller (privileged);
the student infers z from a short interaction history (the encoder = the compact RFF net)
and applies the same control law on z_hat. We measure (a) z-recovery and (b) the
adaptation gap vs the privileged teacher and vs a z-blind baseline. The full Warp
teacher/student PPO is the GPU artifact (hooks noted); the mechanism is validated here.
"""

from __future__ import annotations

import sys
from pathlib import Path
import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from actuator_residual import RFFResidual  # noqa: E402

DT = 0.05


def _plant_step(x, v, u, z):
    """2nd-order plant with hidden damping z in [0.5, 4.0]: v' = u - z*v - k*x."""
    k = 6.0
    a = u - z * v - k * x
    v = v + DT * a
    x = x + DT * v
    return x, v


def _control(x, v, z, target):
    """The z-optimal-ish control law (critical-damping feedforward of the known z).
    Teacher uses true z; student uses z_hat; blind baseline uses a fixed mid z."""
    k = 6.0
    return k * (target - x) + z * v               # cancels the plant damping + drives to target


def _rollout(z, controller, target=1.0, steps=60, history=None):
    """Run the plant under `controller(x,v,t)`; return (return, history of [x,v,u])."""
    x, v = 0.0, 0.0; ret = 0.0; hist = []
    for t in range(steps):
        u = controller(x, v, t)
        hist.append([x, v, u])
        x, v = _plant_step(x, v, u, z)
        ret -= (x - target) ** 2 + 0.001 * u ** 2     # track the target cheaply
    return ret, np.array(hist)


def _hist_features(hist):
    """Compact summary of a short interaction history (the RMA encoder input):
    response statistics that reveal the hidden dynamics."""
    x, v, u = hist[:, 0], hist[:, 1], hist[:, 2]
    return np.array([x.mean(), x.std(), v.mean(), v.std(), u.mean(), u.std(),
                     np.mean(x[1:] - x[:-1]), np.mean(np.abs(v))])


def train_student_encoder(n=400, seed=0):
    """Fit the history-encoder z_hat = f(history features) from rollouts of random worlds.
    Probe each world with a FIXED exploratory controller (so the history is comparable),
    then regress the hidden z from the response — this is the deployable adaptation module."""
    rng = np.random.default_rng(seed)
    probe = lambda x, v, t: 1.0 * np.sin(0.6 * t)          # fixed exploration signal
    X, Y = [], []
    for _ in range(n):
        z = rng.uniform(0.5, 4.0)
        _, hist = _rollout(z, probe, steps=40)
        X.append(_hist_features(hist)); Y.append(z)
    enc = RFFResidual(n_features=200, gamma=0.8, ridge=1e-2, seed=1).fit(np.array(X), np.array(Y))
    return enc, probe


def evaluate(enc, probe, n=200, seed=7):
    """Held-out worlds: recover z, then compare teacher / student / z-blind returns."""
    rng = np.random.default_rng(seed)
    z_err, R_teacher, R_student, R_blind = [], [], [], []
    for _ in range(n):
        z = rng.uniform(0.5, 4.0)
        _, hist = _rollout(z, probe, steps=40)            # online probe -> infer z
        z_hat = float(enc.predict(_hist_features(hist)[None])[0])
        z_err.append(abs(z_hat - z))
        R_teacher.append(_rollout(z, lambda x, v, t: _control(x, v, z, 1.0))[0])      # privileged
        R_student.append(_rollout(z, lambda x, v, t: _control(x, v, z_hat, 1.0))[0])  # inferred
        R_blind.append(_rollout(z, lambda x, v, t: _control(x, v, 2.25, 1.0))[0])     # z-blind mid
    return (np.mean(z_err), np.mean(R_teacher), np.mean(R_student), np.mean(R_blind))


if __name__ == "__main__":
    enc, probe = train_student_encoder()
    z_err, Rt, Rs, Rb = evaluate(enc, probe)
    gap = (Rt - Rs); gap_blind = (Rt - Rb)
    print(f"[RS4] held-out z recovery error: {z_err:.3f} (range 0.5..4.0)")
    print(f"[RS4] return  teacher(privileged)={Rt:.3f}  student(inferred)={Rs:.3f}  "
          f"z-blind={Rb:.3f}")
    print(f"[RS4] adaptation gap student->teacher = {gap:.3f}  (z-blind gap = {gap_blind:.3f})")
    # the student (online-inferred z) must close most of the gap a z-blind policy suffers
    ok = z_err < 0.5 and gap < 0.4 * gap_blind + 1e-9
    print(f"PROVEN: RS4 RMA mechanism — a history encoder infers the hidden dynamics online "
          f"and the student nearly matches the privileged teacher, far beating z-blind: {ok}. "
          f"Full Warp teacher/student PPO is the GPU artifact (this validates the mechanism).")
    sys.exit(0 if ok else 1)
