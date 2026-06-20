# SPDX-License-Identifier: MIT
"""R6 + RS6 — rank designs by ROBUST (lower-tail) return, and keep a DIVERSE set of
robust bodies, not one brittle winner.

R6: a design's score across sampled worlds is summarized by **CVaR@20%** (the mean of
    the worst 20% of its returns), so the chosen body refuses to be one that only wins
    in a lucky sim setting. The verify the checklist asks for: the CVaR-optimal design
    *differs* from the mean-optimal one.
RS6: a **MAP-Elites** quality-diversity archive over behavior descriptors (here: mass,
    clearance) where each cell holds the most *robust* (CVaR) design found for that
    niche → a portfolio of robust morphologies. The robust-QD champion differs from the
    mean-NSGA knee.

Everything here is fitness-AGNOSTIC: pass a `score(design, world)->float` and a
`descriptor(design)->(b0,b1)`. CPU-verifiable now with a synthetic score; on GPU pass
the universal-policy rollout return under a sampled world (domain_model.sample_world).
"""

from __future__ import annotations

import numpy as np


def cvar(returns, alpha: float = 0.2) -> float:
    """Mean of the worst `alpha` fraction (lower tail) — the robust score. alpha=1 -> mean."""
    r = np.sort(np.asarray(returns, float))
    k = max(1, int(np.ceil(alpha * len(r))))
    return float(r[:k].mean())


def robust_score(design, score_fn, worlds, alpha=0.2) -> float:
    return cvar([score_fn(design, w) for w in worlds], alpha)


def mean_score(design, score_fn, worlds) -> float:
    return float(np.mean([score_fn(design, w) for w in worlds]))


class MapElites:
    """MAP-Elites over a 2-D behavior descriptor; each cell keeps its most-robust design."""
    def __init__(self, bins=(8, 8), b0_range=(0.0, 1.0), b1_range=(0.0, 1.0)):
        self.bins = bins
        self.r0, self.r1 = b0_range, b1_range
        self.cells: dict = {}        # (i,j) -> (design, robust_fitness, descriptor)

    def _cell(self, b):
        i = int(np.clip((b[0] - self.r0[0]) / (self.r0[1] - self.r0[0]) * self.bins[0], 0, self.bins[0] - 1))
        j = int(np.clip((b[1] - self.r1[0]) / (self.r1[1] - self.r1[0]) * self.bins[1], 0, self.bins[1] - 1))
        return (i, j)

    def add(self, design, fit, desc):
        c = self._cell(desc)
        if c not in self.cells or fit > self.cells[c][1]:
            self.cells[c] = (np.asarray(design), float(fit), tuple(desc))

    @property
    def coverage(self):
        return len(self.cells)

    def best(self):
        return max(self.cells.values(), key=lambda v: v[1]) if self.cells else None

    def elites(self):
        return list(self.cells.values())


def map_elites(score_fn, descriptor_fn, worlds, dim, iters=400, init=40,
               alpha=0.2, sigma=0.15, seed=0, bins=(8, 8)):
    """Fill a QD archive with the most-robust design per behavior niche.
    `score_fn(design, world)`, `descriptor_fn(design)->(b0,b1)`. Returns the archive."""
    rng = np.random.default_rng(seed)
    arch = MapElites(bins=bins)
    for _ in range(init):                                   # random seed designs
        d = rng.uniform(0, 1, dim)
        arch.add(d, robust_score(d, score_fn, worlds, alpha), descriptor_fn(d))
    for _ in range(iters):                                  # mutate elites, niche-compete
        if not arch.cells:
            break
        parent = arch.elites()[rng.integers(arch.coverage)][0]
        child = np.clip(parent + sigma * rng.standard_normal(dim), 0, 1)
        arch.add(child, robust_score(child, score_fn, worlds, alpha), descriptor_fn(child))
    return arch


# ---------- CPU self-test: a score where mean-best and CVaR-best genuinely differ ----------
def _demo_score(design, world):
    """Synthetic combat-return surrogate. design=[d0,d1] in [0,1]; world=[w] in [0,1].
    d0 = a 'gamble' axis: high d0 wins big in benign worlds (w high) but loses badly in
    harsh worlds (w low) — high mean, terrible lower tail. d1 = a 'robust' axis: a flat,
    world-insensitive return. So mean prefers the gamble; CVaR prefers the robust body."""
    d0, d1 = design[0], design[1]
    gamble = d0 * (2.0 * world - 1.0) * 3.0          # +/- with the world, mean ~0 but spiky
    benign = d0 * 1.2                                 # the seductive average
    robust = d1 * 1.0                                 # steady regardless of world
    return benign + gamble + robust


def _demo_descriptor(design):
    return (float(design[0]), float(design[1]))


if __name__ == "__main__":
    rng = np.random.default_rng(0)
    worlds = list(rng.uniform(0, 1, 64))             # 64 sampled worlds

    # 1. R6: scan designs; mean-optimal vs CVaR-optimal must DIFFER
    grid = rng.uniform(0, 1, (400, 2))
    means = np.array([mean_score(d, _demo_score, worlds) for d in grid])
    cvars = np.array([robust_score(d, _demo_score, worlds, 0.2) for d in grid])
    d_mean, d_cvar = grid[int(np.argmax(means))], grid[int(np.argmax(cvars))]
    print(f"[R6] mean-optimal design  d0={d_mean[0]:.2f} d1={d_mean[1]:.2f} "
          f"(mean {means.max():.2f}, its CVaR {robust_score(d_mean,_demo_score,worlds,0.2):.2f})")
    print(f"[R6] CVaR-optimal design  d0={d_cvar[0]:.2f} d1={d_cvar[1]:.2f} "
          f"(CVaR {cvars.max():.2f}, its mean {mean_score(d_cvar,_demo_score,worlds):.2f})")
    differ = abs(d_mean[0] - d_cvar[0]) > 0.25       # the gamble axis separates them
    print(f"[R6] robust-optimal refuses the high-mean gamble (differs from mean-optimal): {differ}")

    # 2. RS6: QD archive of robust designs
    arch = map_elites(_demo_score, _demo_descriptor, worlds, dim=2, iters=600, seed=1)
    champ = arch.best()
    print(f"[RS6] QD archive coverage = {arch.coverage}/64 niches; robust champion "
          f"d0={champ[0][0]:.2f} d1={champ[0][1]:.2f} CVaR={champ[1]:.2f}")
    ok = differ and arch.coverage >= 20 and champ[0][0] < 0.5     # champion is low-gamble (robust)
    print(f"PROVEN: R6 CVaR ranking differs from mean ranking; RS6 QD archive holds a "
          f"diverse set of robust designs: {ok}. Plug in policy-rollout return + "
          f"domain_model.sample_world on GPU.")
    import sys; sys.exit(0 if ok else 1)
