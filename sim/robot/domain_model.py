# SPDX-License-Identifier: MIT
"""RS1 — a POSTERIOR over sim worlds, not a fixed randomization range.

Phase R draws each episode's "world" from a fixed measured range (`reality_gap`).
RS1 promotes that to a *posterior* that is **tightened from data**: start at the
calibrated prior, then shrink toward the worlds whose simulated traces match the
(real, eventually) logs — the BayRn / DROPO / BayesSim idea (refs in the checklist).

Representation: a factorized Gaussian over the *normalized* [0,1] world coordinates
of the ranking-relevant parameters (friction, mass, damping, restitution, latency,
motor kt, current limit). `sample_world()` draws one; `update_posterior(worlds,
mismatch)` does an importance/elite update (CEM-style, which is a valid variational
tightening for a factorized Gaussian) toward low-mismatch worlds.

HONEST BOUNDARY (the checklist's rule): the machinery + a **sim-to-sim recovery test**
(recover a hidden world from its own traces) are buildable and verified NOW, here, on
CPU. The real-log update is hardware-gated — `update_posterior` takes a mismatch
callable, which is `reality_gap.score_trace_mismatch` against bench logs once parts exist.
"""

from __future__ import annotations

import dataclasses as dc
import sys
from pathlib import Path
import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from reality_gap import default_uncertainty  # noqa: E402

# ranking-relevant world axes and their REAL ranges (lo, hi); normalized coord in [0,1].
WORLD_AXES = {
    "friction":      (0.6, 1.4),
    "mass_scale":    (0.85, 1.15),
    "joint_damping": (0.25, 1.0),
    "restitution":   (0.0, 0.3),
    "latency_s":     (0.002, 0.012),
    "kt_scale":      (0.9, 1.1),
    "i_limit_scale": (0.8, 1.1),
}
AXES = list(WORLD_AXES)
DIM = len(AXES)


def _lo_hi():
    lo = np.array([WORLD_AXES[a][0] for a in AXES])
    hi = np.array([WORLD_AXES[a][1] for a in AXES])
    return lo, hi


def denorm(u) -> dict:
    """Normalized [0,1]^DIM -> a real-units world dict (keys = WORLD_AXES)."""
    lo, hi = _lo_hi()
    x = lo + np.clip(np.asarray(u, float), 0, 1) * (hi - lo)
    return {a: float(x[i]) for i, a in enumerate(AXES)}


@dc.dataclass
class Posterior:
    """Factorized-Gaussian posterior over normalized world coords."""
    mean: np.ndarray
    std: np.ndarray

    @classmethod
    def prior(cls):
        # centred prior; std 0.28 covers the calibrated range without hard edges
        return cls(mean=np.full(DIM, 0.5), std=np.full(DIM, 0.28))

    def sample_world(self, seed) -> dict:
        rng = np.random.default_rng(seed)
        u = np.clip(self.mean + self.std * rng.standard_normal(DIM), 0, 1)
        return denorm(u)

    def sample_norm(self, n, rng) -> np.ndarray:
        return np.clip(self.mean + self.std * rng.standard_normal((n, DIM)), 0, 1)

    def entropy(self) -> float:
        """Sum of per-axis Gaussian differential entropy (uncertainty scalar)."""
        return float(np.sum(0.5 * np.log(2 * np.pi * np.e * self.std ** 2)))


def update_posterior(post: Posterior, worlds_norm: np.ndarray, mismatch: np.ndarray,
                     elite_frac: float = 0.3, step: float = 0.6) -> Posterior:
    """One variational tightening step toward low-mismatch worlds (BayRn/DROPO-style).
    `worlds_norm` = (N,DIM) candidate normalized worlds; `mismatch` = (N,) parity error
    (lower = closer to the data). Keeps the elite (lowest-mismatch) fraction, moves the
    posterior toward their moments, and never collapses std below a floor."""
    n = len(worlds_norm)
    k = max(2, int(elite_frac * n))
    elite = worlds_norm[np.argsort(mismatch)[:k]]
    em, es = elite.mean(0), elite.std(0)
    mean = (1 - step) * post.mean + step * em
    std = np.maximum((1 - step) * post.std + step * es, 0.02)   # floor: keep some doubt
    return Posterior(mean=mean, std=std)


# ---- sim-to-sim recovery test: a cheap analytic "plant" so the machinery is provable ----
def _norm_coords(world: dict) -> np.ndarray:
    """Real-units world dict -> its normalized [0,1] coords (inverse of denorm)."""
    lo, hi = _lo_hi()
    return np.array([(world[a] - lo[i]) / (hi[i] - lo[i]) for i, a in enumerate(AXES)])


def _plant_trace(world: dict, seed=0) -> np.ndarray:
    """A cheap, deterministic forward model: world params -> a feature trace. Stands in
    for 'run the sim, log the channels' so RS1's inference is testable without MJX. Each
    world axis drives a DISTINCT (orthogonal) Fourier mode of the trace, so the world is
    identifiable from the trace — the recovery test then isolates the *inference*
    machinery, not the well-posedness of this stand-in plant. Real traces won't be this
    clean; that's why the posterior keeps a floor of doubt and the fit is hardware-gated."""
    rng = np.random.default_rng(seed)
    t = np.linspace(0, 1, 48)
    c = _norm_coords(world)                       # 7 coefficients in [0,1]
    modes = [np.ones_like(t), np.sin(2*np.pi*t), np.cos(2*np.pi*t), np.sin(4*np.pi*t),
             np.cos(4*np.pi*t), np.sin(6*np.pi*t), np.cos(6*np.pi*t)]
    y = sum(ci * mi for ci, mi in zip(c, modes))
    return y + 0.01 * rng.standard_normal(t.shape)


def recovery_test(rounds=8, n=64, seed=0, verbose=True):
    """Hide a true world; recover it from its trace via repeated posterior updates.
    Returns (err0, err1, ent0, ent1): param error and entropy before/after — error must
    drop and entropy must shrink (we learned the world)."""
    rng = np.random.default_rng(seed)
    true_u = rng.uniform(0.15, 0.85, DIM)
    true_world = denorm(true_u)
    real = _plant_trace(true_world, seed=1)

    post = Posterior.prior()
    err0 = float(np.linalg.norm(post.mean - true_u)); ent0 = post.entropy()
    for r in range(rounds):
        cand = post.sample_norm(n, rng)
        mism = np.array([float(np.mean((_plant_trace(denorm(u), seed=1) - real) ** 2))
                         for u in cand])
        post = update_posterior(post, cand, mism)
        if verbose:
            print(f"  [recover] round {r}: |mean-true|={np.linalg.norm(post.mean-true_u):.3f} "
                  f"entropy={post.entropy():+.2f}")
    err1 = float(np.linalg.norm(post.mean - true_u)); ent1 = post.entropy()
    return err0, err1, ent0, ent1, true_u, post


if __name__ == "__main__":      # CPU self-test: posterior tightens toward the hidden world
    print(f"world axes ({DIM}): {AXES}")
    err0, err1, ent0, ent1, tu, post = recovery_test()
    print(f"\nparam error  {err0:.3f} -> {err1:.3f}   (recovered the hidden world)")
    print(f"entropy      {ent0:+.2f} -> {ent1:+.2f}   (posterior tightened from traces)")
    ok = err1 < 0.6 * err0 and ent1 < ent0
    print(f"PROVEN: RS1 world posterior is inferable from traces (sim-to-sim): {ok}. "
          f"Swap _plant_trace for reality_gap.score_trace_mismatch vs bench logs (hardware-gated).")
    sys.exit(0 if ok else 1)
