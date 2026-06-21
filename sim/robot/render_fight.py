# SPDX-License-Identifier: MIT
"""F0 dashboard — plot the six fight trackers over training from fight_metrics.jsonl.

Reads the per-eval records `train_adversarial.py` streams (sparc, dealt, taken, closing,
fleeing, dist) and renders an ASCII trend per metric per tag, plus the competence verdict
(dealt>taken AND closing>fleeing AND positive SPARC = a fighter, not a survivor). No plot
deps — terminal sparklines so it runs anywhere.

  python render_fight.py [--file out/fight_metrics.jsonl] [--tag f2]
"""

from __future__ import annotations

import argparse, json, os
from pathlib import Path

BARS = " ▁▂▃▄▅▆▇█"


def spark(xs):
    xs = [float(x) for x in xs]
    lo, hi = min(xs), max(xs)
    if hi - lo < 1e-12:
        return BARS[0] * len(xs)
    return "".join(BARS[min(len(BARS) - 1, int((x - lo) / (hi - lo) * (len(BARS) - 1)))] for x in xs)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--file", default=os.environ.get("CODESIGN_OUT", "out") + "/fight_metrics.jsonl")
    ap.add_argument("--tag", default=None)
    a = ap.parse_args()
    p = Path(a.file)
    if not p.exists():
        print(f"no metrics file at {p} — run a fighter training first (make gpu-fighter)."); return
    recs = [json.loads(l) for l in p.read_text().splitlines() if l.strip()]
    if a.tag:
        recs = [r for r in recs if r.get("tag") == a.tag]
    tags = sorted({r.get("tag", "?") for r in recs})
    for tag in tags:
        rs = [r for r in recs if r.get("tag", "?") == tag]
        rs.sort(key=lambda r: r["step"])
        if not rs:
            continue
        steps = [r["step"] for r in rs]
        print(f"\n=== fighter '{tag}'  ({len(rs)} evals, {steps[0]:,}→{steps[-1]:,} steps) ===")
        for k in ("sparc", "dealt", "taken", "closing", "fleeing", "dist"):
            xs = [r.get(k, 0.0) for r in rs]
            print(f"  {k:8s} {spark(xs)}  {xs[0]:+.3f} → {xs[-1]:+.3f}")
        last = rs[-1]
        competent = (last.get("dealt", 0) > last.get("taken", 0) and last.get("dealt", 0) > 0.02
                     and last.get("closing", 0) > last.get("fleeing", 0) and last.get("sparc", -1) > 0)
        verdict = ("COMPETENT FIGHTER (dealt>taken, closing>fleeing, SPARC>0)" if competent else
                   "NOT YET A FIGHTER — " + ("survivor (dealt≈0)" if last.get("dealt", 0) <= 0.02
                   else "flees/loses" if last.get("sparc", 0) <= 0 else "marginal"))
        print(f"  VERDICT: {verdict}")


if __name__ == "__main__":
    main()
