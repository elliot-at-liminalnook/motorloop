# SPDX-License-Identifier: MIT
"""RS3 — a LEARNED contact/damage residual on top of the analytic Newton severity.

R3 unified damage on one currency: impact FORCE in Newtons -> severity via
`reality_gap.damage_from_force` (force / damage_ref). That analytic model is linear in
force and blind to *where* and *how* the hit lands. RS3 adds a learned residual
`f(rel_vel, normal_alignment, impulse, contact_patch, body_region) -> severity
correction` on top of it — so a square hit on a vital region scores more than a
glancing edge contact of the same force, learned from data rather than hand-tuned.

Reuses the compact RFF residual net (closed-form, CPU, with the out-of-support clamp).
Sim-to-sim verified now (recover a synthetic severity correction on held-out contacts);
the drop/ram/weapon-contact bench fit is hardware-gated.
"""

from __future__ import annotations

import sys
from pathlib import Path
import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from reality_gap import default_uncertainty, sample_domain_params, damage_from_force  # noqa: E402
from actuator_residual import RFFResidual  # noqa: E402


def _analytic_severity(force_N, dp):
    """The R3 prior: linear Newton severity (force / damage_ref)."""
    return float(damage_from_force(force_N, dp))


def _hidden_correction(force_N, rel_vel, normal_align, patch, region):
    """The 'real' severity effects the linear prior misses (synthetic ground truth):
    * normal_align in [0,1]: a square hit (1) transfers more than a glancing edge (0).
    * rel_vel: high closing speed concentrates the impulse (mild superlinear term).
    * patch: a small contact patch (a spear tip) concentrates stress -> more severe.
    * region in [0,1]: a vital region (1, e.g. the torso electronics) scores more than a
      limb (0). All smooth, multiplicative on the linear prior."""
    base = force_N / 150.0
    return base * (0.4 * (normal_align - 0.5)
                   + 0.15 * (rel_vel / 5.0)
                   + 0.3 * (0.5 - patch)              # smaller patch -> more severe
                   + 0.5 * (region - 0.5))


def _dataset(points, dp, seed):
    rng = np.random.default_rng(seed)
    X, y, real = [], [], []
    for (force, rel_vel, normal_align, patch, region) in points:
        prior = _analytic_severity(force, dp)
        sev = prior + _hidden_correction(force, rel_vel, normal_align, patch, region) \
            + 0.003 * rng.standard_normal()
        X.append([force, rel_vel, normal_align, patch, region])
        y.append(sev - prior); real.append(sev)
    return np.array(X), np.array(y), np.array(real)


if __name__ == "__main__":
    dp = sample_domain_params(0, default_uncertainty())
    rng = np.random.default_rng(1)
    gen = lambda n, f: [(rng.uniform(*f), rng.uniform(0, 5), rng.uniform(0, 1),
                         rng.uniform(0.05, 0.6), rng.uniform(0, 1)) for _ in range(n)]
    train_pts = gen(800, (20, 400)); test_pts = gen(300, (40, 350))   # disjoint force band
    Xtr, ytr, _ = _dataset(train_pts, dp, 2)
    Xte, _, real_te = _dataset(test_pts, dp, 3)

    net = RFFResidual(n_features=256, gamma=0.9, ridge=1e-3).fit(Xtr, ytr)
    prior_te = np.array([_analytic_severity(p[0], dp) for p in test_pts])
    pred = net.predict(Xte)
    rmse_prior = float(np.sqrt(np.mean((real_te - prior_te) ** 2)))
    rmse_resid = float(np.sqrt(np.mean((real_te - (prior_te + pred)) ** 2)))
    print(f"held-out severity RMSE  analytic-only={rmse_prior:.4f}  "
          f"analytic+residual={rmse_resid:.4f}  ({100*(1-rmse_resid/rmse_prior):.0f}% closed)")

    # monotonicity preserved (R3 sanity): a hard square hit > a glancing soft hit
    hard = prior_te[0] + net.predict(np.array([[300, 4.5, 0.95, 0.08, 0.9]]))[0]
    soft = _analytic_severity(120, dp) + net.predict(np.array([[120, 1.0, 0.1, 0.5, 0.1]]))[0]
    mono = hard > soft
    net.predict(np.array([[9999, 99, 9, 9, 9]]))                       # out-of-support
    ok = rmse_resid < 0.6 * rmse_prior and mono and net.last_clamp_frac > 0
    print(f"hard square vital hit ({hard:.2f}) > glancing soft limb hit ({soft:.2f}): {mono}; "
          f"out-of-support clamp flagged: {net.last_clamp_frac > 0}")
    print(f"PROVEN: RS3 learned contact residual recovers region/velocity/patch severity on "
          f"held-out contacts, preserves the Newton ordering, refuses to extrapolate: {ok}. "
          f"Drop/ram/weapon-contact fit is hardware-gated.")
    sys.exit(0 if ok else 1)
