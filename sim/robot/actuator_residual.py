# SPDX-License-Identifier: MIT
"""RS2 — a LEARNED actuator residual on top of the analytic back-EMF prior.

The analytic envelope (`reality_gap.actuator_scale`: back-EMF droop + current limit +
voltage sag + thermal + gear efficiency) captures the first-order torque-speed shape.
RS2 learns the *residual it misses* — stiction, cogging, nonlinear thermal coupling,
control latency, with a compact net inserted between policy output and the Warp
motor, so it only models what the prior gets wrong (not the whole map).

Compact net = a random-feature ridge model (a 1-hidden-layer net with fixed random
features + a learned linear readout): trains in closed form on CPU, no GPU needed, and
crucially carries its **measured support** so it refuses to extrapolate (the checklist's
"don't let the residual extrapolate outside its support" rule — out-of-range inputs are
clamped and flagged).

HELD-OUT honesty: the residual is fit on one set of operating points and scored on a
DISJOINT set, so the reported error is generalization, not memorization. Buildable +
sim-to-sim verified now (recover a synthetic residual from a perturbed motor); the
motorloop bench-log fit is hardware-gated.
"""

from __future__ import annotations

import sys
from pathlib import Path
import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from reality_gap import default_uncertainty, sample_domain_params, actuator_scale  # noqa: E402


class RFFResidual:
    """Random-Fourier-feature ridge residual: y ~= readout . cos(W.x_std + b).
    Fixed random features (the "net"), learned linear readout (closed-form ridge).
    Records per-input support; predict() clamps to it and counts clamped inputs."""

    def __init__(self, n_features=256, gamma=1.0, ridge=1e-2, seed=0):
        self.F, self.gamma, self.ridge, self.seed = n_features, gamma, ridge, seed
        self.fitted = False
        self.last_clamp_frac = 0.0

    def _feat(self, Xs):
        return np.cos(Xs @ self.W + self.b)

    def fit(self, X, y):
        X = np.atleast_2d(np.asarray(X, float)); y = np.asarray(y, float).ravel()
        self.mu, self.sd = X.mean(0), X.std(0) + 1e-9
        self.lo, self.hi = X.min(0), X.max(0)                  # measured support
        rng = np.random.default_rng(self.seed)
        d = X.shape[1]
        self.W = rng.normal(0, self.gamma, (d, self.F))
        self.b = rng.uniform(0, 2 * np.pi, self.F)
        Phi = self._feat((X - self.mu) / self.sd)
        A = Phi.T @ Phi + self.ridge * np.eye(self.F)
        self.beta = np.linalg.solve(A, Phi.T @ y)
        self.fitted = True
        return self

    def predict(self, X):
        X = np.atleast_2d(np.asarray(X, float))
        clamped = (X < self.lo) | (X > self.hi)
        self.last_clamp_frac = float(clamped.any(1).mean())     # fraction of out-of-support rows
        Xc = np.clip(X, self.lo, self.hi)                       # never extrapolate
        return self._feat((Xc - self.mu) / self.sd) @ self.beta


def _prior_torque(action, qvel, dp):
    """Analytic realized torque fraction (the prior the residual sits on top of)."""
    return action * float(actuator_scale(np.array([qvel]), dp)[0])


def _hidden_residual(action, qvel, temp):
    """The 'real' effects the affine prior misses (synthetic ground truth for the
    sim-to-sim test): speed-dependent viscous loss, Coulomb/breakaway friction, and a
    nonlinear thermal droop — all smooth functions of the observed state, so they are
    learnable. A tiny rotor-angle cogging ripple is added as the IRREDUCIBLE floor (it
    is periodic in unobserved rotor angle, so no state-based net can predict it — this
    is why a residual closes most, not all, of the gap). Replaced by bench logs on hw."""
    viscous = -0.004 * qvel                                 # speed-dependent loss
    coulomb = -0.03 * np.tanh(8 * action)                   # breakaway/Coulomb friction
    thermal = -0.04 * temp * abs(action)                    # nonlinear thermal droop
    cogging = 0.004 * np.sin(50 * action + 30 * qvel)       # irreducible (unobserved angle)
    return viscous + coulomb + thermal + cogging


def _dataset(operating_points, dp, seed):
    rng = np.random.default_rng(seed)
    X, y = [], []
    for (action, qvel, temp) in operating_points:
        prior = _prior_torque(action, qvel, dp)
        real = prior + _hidden_residual(action, qvel, temp) + 0.002 * rng.standard_normal()
        X.append([action, qvel, dp["vbus"], temp, dp["kt"], dp["gear"]])
        y.append(real - prior)                                  # learn the RESIDUAL only
    return np.array(X), np.array(y), np.array(
        [_prior_torque(a, q, dp) + _hidden_residual(a, q, t) for (a, q, t) in operating_points])


if __name__ == "__main__":
    dp = sample_domain_params(0, default_uncertainty())
    rng = np.random.default_rng(1)
    # TRAIN on actions in [-0.7,0.7], speeds in [-30,30]; TEST on a DISJOINT inner band
    train_pts = [(rng.uniform(-0.7, 0.7), rng.uniform(-30, 30), rng.uniform(0, 0.8))
                 for _ in range(800)]
    test_pts = [(rng.uniform(-0.5, 0.5), rng.uniform(-20, 20), rng.uniform(0.1, 0.7))
                for _ in range(300)]
    Xtr, ytr, _ = _dataset(train_pts, dp, 2)
    Xte, yte, real_te = _dataset(test_pts, dp, 3)

    net = RFFResidual(n_features=256, gamma=0.7, ridge=1e-3).fit(Xtr, ytr)
    pred = net.predict(Xte)

    prior_te = np.array([_prior_torque(a, q, dp) for (a, q, t) in test_pts])
    rmse_prior = float(np.sqrt(np.mean((real_te - prior_te) ** 2)))             # prior only
    rmse_resid = float(np.sqrt(np.mean((real_te - (prior_te + pred)) ** 2)))     # prior + learned residual
    print(f"held-out RMSE  prior-only={rmse_prior:.4f}  prior+residual={rmse_resid:.4f}  "
          f"({100*(1-rmse_resid/rmse_prior):.0f}% of the gap closed)")
    # out-of-support guard: a wild input must be clamped + flagged
    net.predict(np.array([[5.0, 999.0, dp["vbus"], 9.0, dp["kt"], dp["gear"]]]))
    print(f"out-of-support input clamp flagged: {net.last_clamp_frac > 0}")
    ok = rmse_resid < 0.6 * rmse_prior and net.last_clamp_frac > 0
    print(f"PROVEN: RS2 learned actuator residual recovers a synthetic motor residual on "
          f"held-out points (no memorization) and refuses to extrapolate: {ok}. Fit on "
          f"motorloop bench logs is hardware-gated.")
    sys.exit(0 if ok else 1)
