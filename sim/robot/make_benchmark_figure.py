# SPDX-License-Identifier: MIT
"""Plot the HONEST learning curve from the held-out benchmark — the proof that performance keeps
RISING with more training time (the win-exchanges resilience deliverable). Reads one or more
`{tag}_benchmark.jsonl` files (written by `train_adversarial`'s keep-best loop) and plots:
  (1) benchmark SPARC per eval + the BEST-SO-FAR envelope (monotone by construction), vs cum_step;
  (2) benchmark dealt/taken ratio (>1 = winning exchanges) vs cum_step.
Concatenate phases in order to see the whole curriculum chain.

  python make_benchmark_figure.py --tags cval c1 c2 c3 [--out sim/build/gpu/figures]
  python make_benchmark_figure.py --glob 'out/*_benchmark.jsonl'
"""

from __future__ import annotations

import argparse, glob, json, os
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = Path(__file__).resolve().parent
OUT = Path(os.environ.get("CODESIGN_OUT", str(HERE.parents[1] / "sim/build/gpu/out")))


def load(files):
    rows = []
    for f in files:
        for ln in Path(f).read_text().splitlines():
            ln = ln.strip()
            if ln:
                try: rows.append(json.loads(ln))
                except Exception: pass
    rows.sort(key=lambda r: r.get("cum_step", r.get("step", 0)))
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tags", nargs="*", default=None, help="phase tags in order (reads {tag}_benchmark.jsonl)")
    ap.add_argument("--glob", default=None, help="glob for *_benchmark.jsonl")
    ap.add_argument("--src", default=str(OUT), help="dir holding the jsonl files")
    ap.add_argument("--out", default=str(HERE.parents[1] / "sim/build/gpu/figures"))
    a = ap.parse_args()
    if a.tags:
        files = [Path(a.src) / f"{t}_benchmark.jsonl" for t in a.tags]
        files = [f for f in files if f.exists()]
    elif a.glob:
        files = sorted(glob.glob(a.glob))
    else:
        files = sorted(glob.glob(str(Path(a.src) / "*_benchmark.jsonl")))
    if not files:
        print("no benchmark jsonl found"); return
    rows = load(files)
    if not rows:
        print("benchmark jsonl empty"); return
    Path(a.out).mkdir(parents=True, exist_ok=True)
    step = np.array([r.get("cum_step", r.get("step", 0)) for r in rows], float)
    sparc = np.array([r["bench_sparc"] for r in rows], float)
    best = np.array([r["best"] for r in rows], float)
    ratio = np.array([r.get("bench_ratio", 0.0) for r in rows], float)

    fig, ax = plt.subplots(1, 2, figsize=(13, 4.6))
    ax[0].plot(step, sparc, ".-", color="tab:blue", alpha=0.6, label="benchmark SPARC (per eval)")
    ax[0].plot(step, best, "-", color="tab:red", lw=2, label="best-so-far (monotone)")
    ax[0].set_xlabel("cumulative env-steps"); ax[0].set_ylabel("held-out benchmark SPARC")
    ax[0].set_title("Performance keeps rising with training time"); ax[0].legend(); ax[0].grid(alpha=0.3)
    ax[1].axhline(1.0, color="k", lw=0.8, ls="--", label="dealt=taken")
    ax[1].plot(step, ratio, ".-", color="tab:green")
    ax[1].set_xlabel("cumulative env-steps"); ax[1].set_ylabel("benchmark dealt/taken ratio")
    ax[1].set_title("Wins exchanges when ratio > 1"); ax[1].legend(); ax[1].grid(alpha=0.3)
    fig.tight_layout(); p = Path(a.out) / "benchmark_curve.png"; fig.savefig(p, dpi=120); plt.close(fig)
    mono = bool(np.all(np.diff(best) >= -1e-9))
    print(f"rows={len(rows)} final_best={best[-1]:.2f} final_ratio={ratio[-1]:.2f} "
          f"best_monotone={mono} -> {p}")


if __name__ == "__main__":
    main()
