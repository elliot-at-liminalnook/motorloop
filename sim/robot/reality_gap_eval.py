# SPDX-License-Identifier: MIT
"""RS5 + RS8 — evaluate the reality gap by its effect on DESIGN RANKINGS.

RS5 (active experiment selection): don't run a fixed hardware battery — pick the
    measurement that most reduces uncertainty about *which design wins*. We score each
    candidate world-axis by its **information gain on the ranking** (H(winner) -
    H(winner | axis)) over the world posterior, not info gain on the parameter. The
    axis whose value flips the winner scores highest. [the checklist's RS5]

RS8 (flagship): rank N designs three ways — (1) static proxy, (2) nominal-sim return,
    (3) calibrated-ensemble **robust (CVaR)** return — and report which ranking best
    predicts the highest-fidelity signal available (a CPU "oracle" here; reduced
    hardware later). This is the quantitative "how much do calibration + robustness
    change the answer" result, generalizing codesign_validate's walker-vs-fighter rho.

CPU-verifiable now with a synthetic-but-principled score; on GPU pass the universal-
policy rollout return as `score(design, world)` and domain_model worlds.
"""

from __future__ import annotations

import sys
from pathlib import Path
import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from robust_codesign import cvar  # noqa: E402


def _entropy(labels, k):
    p = np.bincount(labels, minlength=k) / max(len(labels), 1)
    p = p[p > 0]
    return float(-(p * np.log(p)).sum())


def ranking_info_gain(designs, worlds_axes, score_fn, axis_names, q=4):
    """Info gain on the WINNER design from learning each world-axis.
    `worlds_axes` = (W, A) sampled worlds (A axes); `score_fn(design_idx, world_row)`.
    Returns {axis_name: info_gain}. Higher = measuring this axis resolves the winner."""
    W = len(worlds_axes); nD = len(designs)
    winners = np.array([int(np.argmax([score_fn(d, w) for d in range(nD)]))
                        for w in worlds_axes])
    H0 = _entropy(winners, nD)
    out = {}
    for a, name in enumerate(axis_names):
        vals = worlds_axes[:, a]
        edges = np.quantile(vals, np.linspace(0, 1, q + 1))
        bins = np.clip(np.digitize(vals, edges[1:-1]), 0, q - 1)
        cond = 0.0
        for b in range(q):
            m = bins == b
            if m.any():
                cond += (m.mean()) * _entropy(winners[m], nD)
        out[name] = H0 - cond
    return out, H0


def rank_correlation(a, b):
    ra, rb = np.argsort(np.argsort(a)), np.argsort(np.argsort(b))
    return float(np.corrcoef(ra, rb)[0, 1])


def rank_three_ways(designs, proxy_fn, nominal_world, worlds, score_fn, oracle_fn, alpha=0.2):
    """RS8: proxy / nominal / robust rankings + their Spearman vs the oracle."""
    proxy = np.array([proxy_fn(d) for d in designs])
    nominal = np.array([score_fn(d, nominal_world) for d in designs])
    robust = np.array([cvar([score_fn(d, w) for w in worlds], alpha) for d in designs])
    oracle = np.array([oracle_fn(d) for d in designs])
    return {
        "proxy_rho": rank_correlation(proxy, oracle),
        "nominal_rho": rank_correlation(nominal, oracle),
        "robust_rho": rank_correlation(robust, oracle),
    }, dict(proxy=proxy, nominal=nominal, robust=robust, oracle=oracle)


# ----------------------------- CPU self-test (synthetic but principled) -----------------------------
if __name__ == "__main__":
    rng = np.random.default_rng(0)

    # RS5: two designs whose winner depends ONLY on the friction axis; restitution is a
    # red herring. The info-gain scorer must rank 'friction' >> 'restitution'.
    AX = ["friction", "restitution"]
    worlds_axes = rng.uniform(0, 1, (300, 2))
    def s_rank(d, w):                       # d=0 likes high friction; d=1 likes low friction
        fric = w[0]
        return (fric if d == 0 else (1 - fric)) + 0.01 * w[1]
    ig, H0 = ranking_info_gain([0, 1], worlds_axes, s_rank, AX)
    print(f"[RS5] winner-entropy H0={H0:.2f} nats; info-gain  friction={ig['friction']:.3f}  "
          f"restitution={ig['restitution']:.3f}")
    rs5_ok = ig["friction"] > 5 * max(ig["restitution"], 1e-3)
    print(f"[RS5] active selection picks the ranking-flipping measurement (friction): {rs5_ok}")

    # RS8: 12 designs. The oracle (high-fidelity truth) reflects performance over a WIDE
    # world distribution incl. harsh tails. proxy = a static guess; nominal = return at the
    # benign nominal world; robust = CVaR over sampled worlds. Robust must track the oracle best.
    nD = 12
    designs = list(range(nD))
    # each design has a benign return and a fragility (how much harsh worlds hurt it)
    benign = rng.uniform(0.5, 1.0, nD)
    fragility = rng.uniform(0.0, 1.2, nD)
    static_guess = benign + 0.1 * rng.standard_normal(nD)          # proxy ~ benign + noise
    worlds = rng.uniform(0, 1, 64)                                 # 0=harsh .. 1=benign
    def score(d, w):  return benign[d] - fragility[d] * (1 - w)    # harsh world (w->0) hurts fragile designs
    nominal_world = 0.9                                            # the (too-benign) nominal sim
    def proxy_fn(d): return static_guess[d]
    def oracle_fn(d): return np.mean([score(d, w) for w in rng.uniform(0, 1, 256)])  # wide truth
    rhos, _ = rank_three_ways(designs, proxy_fn, nominal_world, worlds, score, oracle_fn)
    print(f"[RS8] rank-corr vs oracle:  proxy={rhos['proxy_rho']:+.2f}  "
          f"nominal={rhos['nominal_rho']:+.2f}  robust={rhos['robust_rho']:+.2f}")
    rs8_ok = rhos["robust_rho"] >= rhos["nominal_rho"] and rhos["robust_rho"] >= rhos["proxy_rho"]
    print(f"[RS8] robust (CVaR) ranking predicts the high-fidelity oracle best: {rs8_ok}")

    ok = rs5_ok and rs8_ok
    print(f"PROVEN: RS5 info-gain picks the ranking-flipping test; RS8 robust ranking "
          f"beats proxy/nominal at predicting the oracle: {ok}. On GPU: score=policy "
          f"rollout, oracle=CPU MuJoCo re-rank (hardware later).")
    sys.exit(0 if ok else 1)
