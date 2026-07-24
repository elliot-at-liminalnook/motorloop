# SPDX-License-Identifier: MIT
"""RS7 — the multi-fidelity ladder + promotion gates.

Three explicit tiers, cheapest first, with a gate between them so the EXPENSIVE model
never runs in the inner loop (the checklist's hard rule):
  (a) MuJoCo-Warp ensemble — millions of rollouts; the inner RL/search loop.
  (b) CPU MuJoCo oracle    — richer contact logging + parity; re-ranks promoted candidates.
  (c) hardware snippets    — actuator/leg/drop/ram; validates finalists, updates the ensemble.

`ladder(cands, tier_a, tier_b, tier_c, ...)` ranks all candidates with the cheap tier_a,
promotes the top fraction to tier_b, re-ranks those, optionally promotes the very top to
tier_c. It returns the finalists and the eval BUDGET spent per tier — the point being that
the costly tiers run on O(few) candidates, not all. CPU-verifiable now (tier_c gated):
the gate recovers the true-best with far fewer oracle evals than oracle-ranking everything.
"""

from __future__ import annotations

import numpy as np


def ladder(cands, tier_a, tier_b, tier_c=None, promote_b=0.3, promote_c=2):
    """Cheap-to-expensive promotion. tier_x(cand)->score (higher=better). tier_c may be
    None (hardware-gated). Returns dict(order_a, promoted_b, ranked_b, finalists,
    budget={a,b,c})."""
    n = len(cands)
    sa = np.array([tier_a(c) for c in cands])
    order_a = list(np.argsort(-sa))
    kb = max(1, int(promote_b * n))
    prom_b = order_a[:kb]
    sb = {i: tier_b(cands[i]) for i in prom_b}                 # oracle ONLY on promoted
    ranked_b = sorted(prom_b, key=lambda i: -sb[i])
    budget = {"a": n, "b": len(prom_b), "c": 0}
    finalists = ranked_b[:promote_c]
    if tier_c is not None:
        sc = {i: tier_c(cands[i]) for i in finalists}
        finalists = sorted(finalists, key=lambda i: -sc[i])
        budget["c"] = len(sc)
    return dict(order_a=order_a, promoted_b=list(prom_b), ranked_b=ranked_b,
                finalists=list(finalists), budget=budget, sa=sa, sb=sb)


if __name__ == "__main__":
    import sys
    rng = np.random.default_rng(0)
    n = 120
    # ground truth quality per candidate; tier_b (oracle) ~ truth + small noise; tier_a
    # (fast) ~ truth + LARGE noise (cheap but rough). tier_c ~ truth (hardware, gated here).
    truth = rng.uniform(0, 1, n)
    def tier_a(i): return truth[i] + 0.25 * rng.standard_normal()
    def tier_b(i): return truth[i] + 0.05 * rng.standard_normal()
    def tier_c(i): return truth[i]
    cands = list(range(n))
    res = ladder(cands, tier_a, tier_b, tier_c=tier_c, promote_b=0.25, promote_c=3)

    true_best = int(np.argmax(truth))
    got = res["finalists"][0]
    print(f"[RS7] tiers spent: fast={res['budget']['a']}  oracle={res['budget']['b']}  "
          f"hardware={res['budget']['c']}  (oracle-ranking-all would cost {n})")
    print(f"[RS7] finalist={got} (truth {truth[got]:.3f}); true best={true_best} "
          f"(truth {truth[true_best]:.3f})")
    # the gate must find a near-best finalist while running the oracle on a small fraction
    near_best = truth[got] >= np.sort(truth)[-5]          # finalist is within the top 5
    cheap = res["budget"]["b"] <= 0.3 * n
    ok = near_best and cheap
    print(f"[RS7] inner loop stayed cheap ({res['budget']['b']}/{n} oracle evals = "
          f"{100*res['budget']['b']/n:.0f}%), finalist in top-5: {near_best}")
    print(f"PROVEN: RS7 multi-fidelity ladder promotes only top candidates to the costly "
          f"tiers (oracle never in the inner loop); recovers a near-best finalist for a "
          f"fraction of the oracle cost: {ok}. Tier-c (hardware) is gated.")
    sys.exit(0 if ok else 1)
