# SPDX-License-Identifier: MIT
"""Multi-panel training DASHBOARD — every held-out benchmark signal over cumulative env-steps, so a
plateau EXPLAINS itself (is it firing? winning cleanly or trading? at what range?). Reads the
`{tag}_benchmark.jsonl` files (basic OR enriched columns; plots whatever is present) in phase order.

  python make_dashboard.py [--tags strk0 cval c1 c2 c3] [--src out] [--out figures/dashboard.png]
  python make_dashboard.py --glob 'out/*_benchmark.jsonl'
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
                except json.JSONDecodeError: pass  # tolerate a truncated in-flight append (writer may be live)
    rows.sort(key=lambda r: r.get("cum_step", r.get("step", 0)))
    return rows


def col(rows, *names):
    """First present column among names -> (xs, ys) with cum_step; None if absent."""
    for n in names:
        if any(n in r for r in rows):
            xs = [r.get("cum_step", r.get("step", 0)) for r in rows if n in r]
            ys = [r[n] for r in rows if n in r]
            return np.array(xs, float), np.array(ys, float)
    return None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tags", nargs="*", default=None)
    ap.add_argument("--glob", default=None)
    ap.add_argument("--src", default=str(OUT))
    ap.add_argument("--out", default=str(HERE.parents[1] / "sim/build/gpu/figures/dashboard.png"))
    ap.add_argument("--table", action="store_true", help="print the per-phase signal table; skip the figure")
    a = ap.parse_args()
    if a.tags:
        files = [Path(a.src) / f"{t}_benchmark.jsonl" for t in a.tags]
        files = [str(f) for f in files if f.exists()]
    else:
        files = sorted(glob.glob(a.glob or str(Path(a.src) / "*_benchmark.jsonl")))
    rows = load(files)
    if not rows:
        print("no benchmark data found"); return

    if a.table:
        # per-file (phase) summary: best SPARC + latest ratio + decomposition if present
        cols = ["best", "bench_ratio", "clean", "trade", "fire", "sparc_close", "sparc_med", "sparc_far"]
        hdr = f"{'phase':8s} {'evals':>5s} {'best':>8s} " + " ".join(f"{c.replace('sparc_',''):>7s}" for c in cols[1:])
        print(hdr)
        for f in files:
            pr = load([f]);
            if not pr: continue
            tag = Path(f).name.replace("_benchmark.jsonl", "")
            best = max(r.get("best", -1e9) for r in pr); last = pr[-1]
            cells = [f"{best:8.2f}"] + [f"{last.get(c, float('nan')):7.3f}" if c in last else f"{'—':>7s}" for c in cols[1:]]
            print(f"{tag:8s} {len(pr):5d} " + " ".join(cells))
        print(f"\nglobal best: {max(r.get('best',-1e9) for r in rows):.2f}  "
              f"(— = enriched columns appear once the enriched kernel runs that phase)")
        return

    # (panel title, [series (label, *colnames)], baseline-y or None)
    panels = [
        ("held-out SPARC + best", [("best", "best"), ("per-eval", "bench_sparc", "sparc")], None),
        ("dealt/taken ratio (>1 wins)", [("ratio", "bench_ratio", "ratio")], 1.0),
        ("clean hit vs trade", [("clean", "clean"), ("trade", "trade")], None),
        ("rod fire rate", [("fire", "fire")], None),
        ("SPARC by range", [("close", "sparc_close"), ("med", "sparc_med"), ("far", "sparc_far")], None),
        ("engagement (dist / closing)", [("dist", "dist", "bench_dist"), ("closing", "closing")], None),
    ]
    present = []
    for title, series, base in panels:
        got = [(lab, col(rows, *cs)) for lab, *cs in series]
        got = [(lab, d) for lab, d in got if d is not None]
        if got:
            present.append((title, got, base))
    n = len(present); ncol = 2; nrow = (n + 1) // 2
    fig, axes = plt.subplots(nrow, ncol, figsize=(13, 3.2 * nrow), squeeze=False)
    for ax, (title, got, base) in zip(axes.flat, present):
        if base is not None:
            ax.axhline(base, color="k", lw=0.8, ls="--")
        for lab, (xs, ys) in got:
            ax.plot(xs, ys, ".-", label=lab, alpha=0.85)
        ax.set_title(title); ax.set_xlabel("cum env-steps"); ax.grid(alpha=0.3); ax.legend(fontsize=8)
    for ax in axes.flat[n:]:
        ax.axis("off")
    fig.suptitle("Fighter training dashboard — held-out signals", y=0.995, fontsize=13)
    fig.tight_layout()
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(a.out, dpi=110); plt.close(fig)
    last = rows[-1]
    print(f"rows={len(rows)} panels={n} -> {a.out}")
    print("latest:", {k: last[k] for k in ("cum_step", "best", "bench_ratio", "clean", "trade", "fire",
                                           "sparc_close", "sparc_med", "sparc_far") if k in last})


if __name__ == "__main__":
    main()
