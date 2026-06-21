# SPDX-License-Identifier: MIT
"""Headline co-design figure: does robust calibrated body-ranking beat proxy/nominal, and does
the benefit only appear once CONTACT dynamics matter?

Plots Spearman(ranking, ground-truth) for proxy / nominal / robust, side by side for
LOCOMOTION (where they tied — task too smooth) vs COMBAT (the fair test). Locomotion numbers
come from `fighter_rank.py`; combat from `combat_rank.npz`. Also a combat scatter of each
ranking's score vs true combat performance.

  python make_ranking_figure.py [--combat sim/build/gpu/out/combat_rank.npz \
      --loco-rhos 0.92,0.99,0.98 --combat-rhos auto]
"""

from __future__ import annotations

import argparse
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = Path(__file__).resolve().parent


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--combat", default=str(HERE.parents[1] / "sim/build/gpu/out/combat_rank.npz"))
    ap.add_argument("--loco-rhos", default="0.92,0.99,0.98", help="proxy,nominal,robust on locomotion")
    ap.add_argument("--out", default=str(HERE.parents[1] / "sim/build/gpu/figures"))
    a = ap.parse_args()
    Path(a.out).mkdir(parents=True, exist_ok=True)
    loco = [float(x) for x in a.loco_rhos.split(",")]
    d = np.load(a.combat)
    combat = [float(d["rho_proxy"]), float(d["rho_nominal"]), float(d["rho_robust"])]

    # 1. grouped bar: ranking method × task
    labels = ["proxy", "nominal", "robust(CVaR)"]; x = np.arange(3); w = 0.38
    fig, axi = plt.subplots(figsize=(9, 5))
    axi.bar(x - w/2, loco, w, label="locomotion (smooth → all tie)", color="tab:gray")
    axi.bar(x + w/2, combat, w, label="combat (contact → robust wins)", color="tab:red")
    axi.axhline(0, color="k", lw=0.8)
    axi.set_xticks(x); axi.set_xticklabels(labels)
    axi.set_ylabel("Spearman ρ (ranking vs ground-truth performance)")
    axi.set_title("Robust calibrated body-ranking pays off only when contact dynamics matter")
    axi.legend(); axi.grid(alpha=0.3, axis="y")
    for xi, v in zip(x - w/2, loco): axi.text(xi, v + 0.02*np.sign(v or 1), f"{v:+.2f}", ha="center", fontsize=8)
    for xi, v in zip(x + w/2, combat): axi.text(xi, v + 0.02*np.sign(v or 1), f"{v:+.2f}", ha="center", fontsize=8)
    fig.tight_layout(); fig.savefig(Path(a.out) / "ranking_loco_vs_combat.png", dpi=120); plt.close(fig)

    # 2. combat scatter: each ranking's score vs true combat performance
    fig, ax = plt.subplots(1, 3, figsize=(14, 4.2))
    for axi, k, lab in zip(ax, ("proxy", "nominal", "robust"),
                           (f"proxy ρ={combat[0]:+.2f}", f"nominal ρ={combat[1]:+.2f}", f"robust ρ={combat[2]:+.2f}")):
        axi.scatter(d[k], d["truth"], s=30)
        axi.set_xlabel(f"{k} score"); axi.set_ylabel("true combat perf"); axi.set_title(lab); axi.grid(alpha=0.3)
    fig.suptitle("Combat: which ranking predicts true (wide-world) fight performance?")
    fig.tight_layout(); fig.savefig(Path(a.out) / "ranking_combat_scatter.png", dpi=120); plt.close(fig)

    print(f"locomotion ρ (proxy/nominal/robust): {loco}")
    print(f"combat     ρ (proxy/nominal/robust): {[round(c,3) for c in combat]}")
    print(f"figures -> {a.out}/ (ranking_loco_vs_combat.png, ranking_combat_scatter.png)")


if __name__ == "__main__":
    main()
