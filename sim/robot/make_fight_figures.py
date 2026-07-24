# SPDX-License-Identifier: MIT
"""Figures for the fighter-curriculum training from fight_metrics.jsonl.

Reads the per-eval records (sparc, dealt, taken, closing, fleeing, dist) that
train_adversarial streams, concatenates the curriculum phases in order, and renders:
  1. the six metrics over global training steps (phase boundaries marked) — the headline
     "did it learn to fight" curves;
  2. dealt-vs-taken (is it winning the exchange?);
  3. a per-phase summary bar (final dealt/sparc per curriculum rung).
Saves PNGs to <out>/figures/. CPU-only (matplotlib); run locally on the pulled metrics.

  python make_fight_figures.py [--file sim/build/gpu/out/fight_metrics.jsonl] [--tags cval,c1,c2,c3,c4]
"""

from __future__ import annotations

import argparse, re
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = Path(__file__).resolve().parent

# parse the per-eval log lines (robust to fight_metrics.jsonl being truncated per phase):
#   "  [c1] step 1,761,280 sparc -42.97 dealt 0.085 taken 0.085 close 0.05 flee 8.65 dist 39.60 (572s)"
_LINE = re.compile(r"\[(\w+)\] step ([\d,]+) sparc (-?[\d.]+) dealt ([\d.]+) taken ([\d.]+) "
                   r"close (-?[\d.]+) flee (-?[\d.]+) dist (-?[\d.]+)")


def parse_logs(outdir, tags):
    """Reconstruct per-phase metric series from the {tag}.log / curr_{tag}.log / cval.log files."""
    by = {}
    for t in tags:
        recs = []
        for cand in (outdir / f"curr_{t}.log", outdir / f"{t}.log"):
            if cand.exists():
                for ln in cand.read_text(errors="ignore").splitlines():
                    m = _LINE.search(ln)
                    if m and m.group(1) == t:
                        recs.append(dict(tag=t, step=int(m.group(2).replace(",", "")),
                                         sparc=float(m.group(3)), dealt=float(m.group(4)),
                                         taken=float(m.group(5)), closing=float(m.group(6)),
                                         fleeing=float(m.group(7)), dist=float(m.group(8))))
                break
        if recs:
            by[t] = sorted(recs, key=lambda r: r["step"])
    return by


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=str(HERE.parents[1] / "sim/build/gpu/figures"))
    ap.add_argument("--logs", default=str(HERE.parents[1] / "sim/build/gpu/out"),
                    help="dir with the per-phase {tag}.log/curr_{tag}.log files")
    ap.add_argument("--tags", default="cval,c1,c2,c3,c4", help="phase order (comma list)")
    a = ap.parse_args()
    order = [t for t in a.tags.split(",") if t]
    by = parse_logs(Path(a.logs), order)
    if not by:
        print(f"no records for tags {order} in {a.file}"); return
    Path(a.out).mkdir(parents=True, exist_ok=True)

    # concatenate phases on a global step axis; record phase boundaries
    gstep, series = [], {k: [] for k in ("sparc", "dealt", "taken", "closing", "fleeing", "dist")}
    bounds, off = [], 0
    for t in by:
        rs = by[t]
        for r in rs:
            gstep.append(off + r["step"])
            for k in series:
                series[k].append(r.get(k, 0.0))
        off = gstep[-1]; bounds.append((t, off))
    gstep = np.array(gstep) / 1e6  # millions

    # 1. six-metric panel
    fig, ax = plt.subplots(2, 3, figsize=(15, 8))
    for axi, k in zip(ax.flat, ("sparc", "dealt", "taken", "closing", "fleeing", "dist")):
        axi.plot(gstep, series[k], lw=1.5)
        for t, b in bounds:
            axi.axvline(b / 1e6, color="k", ls=":", alpha=0.3)
        axi.set_title(k); axi.set_xlabel("global env-steps (M)"); axi.grid(alpha=0.3)
    fig.suptitle("Fighter curriculum — learning curves (phase boundaries dotted)")
    fig.tight_layout(); fig.savefig(Path(a.out) / "curriculum_metrics.png", dpi=110); plt.close(fig)

    # 2. dealt vs taken
    fig, axi = plt.subplots(figsize=(9, 5))
    axi.plot(gstep, series["dealt"], label="dealt (we land)", color="tab:green")
    axi.plot(gstep, series["taken"], label="taken (we eat)", color="tab:red")
    for t, b in bounds: axi.axvline(b / 1e6, color="k", ls=":", alpha=0.3)
    axi.set_title("Damage exchange: dealt vs taken"); axi.set_xlabel("global env-steps (M)")
    axi.legend(); axi.grid(alpha=0.3)
    fig.tight_layout(); fig.savefig(Path(a.out) / "curriculum_dealt_vs_taken.png", dpi=110); plt.close(fig)

    # 3. per-phase final summary
    fig, axi = plt.subplots(figsize=(9, 5))
    phases = list(by); fd = [by[t][-1].get("dealt", 0) for t in phases]
    fs = [by[t][-1].get("sparc", 0) for t in phases]
    x = np.arange(len(phases))
    axi.bar(x - 0.2, fd, 0.4, label="final dealt", color="tab:green")
    axi2 = axi.twinx(); axi2.bar(x + 0.2, fs, 0.4, label="final SPARC", color="tab:blue", alpha=0.7)
    axi.set_xticks(x); axi.set_xticklabels(phases); axi.set_ylabel("final dealt"); axi2.set_ylabel("final SPARC")
    axi.set_title("Per-phase final dealt + SPARC"); axi.grid(alpha=0.3)
    fig.tight_layout(); fig.savefig(Path(a.out) / "curriculum_per_phase.png", dpi=110); plt.close(fig)

    # text summary
    print("Phase        final_dealt  final_taken  final_sparc  evals")
    for t in by:
        r = by[t][-1]
        print(f"  {t:8s}   {r.get('dealt',0):10.4f}  {r.get('taken',0):11.4f}  {r.get('sparc',0):11.2f}  {len(by[t])}")
    print(f"\nfigures -> {a.out}/  (curriculum_metrics.png, curriculum_dealt_vs_taken.png, curriculum_per_phase.png)")


if __name__ == "__main__":
    main()
