# SPDX-License-Identifier: MIT
"""Figures for the command-conditioned controller from cmd_eval.npz: (1) commanded vs
achieved base velocity over time (does it follow the remote command?), (2) the top-down
xy path (does it walk the commanded square?), + a training tracking curve from cmd_train.jsonl.

  python make_command_figure.py [--npz sim/build/gpu/out/cmd_eval.npz]
"""

from __future__ import annotations

import argparse, json
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = Path(__file__).resolve().parent


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--npz", default=str(HERE.parents[1] / "sim/build/gpu/out/cmd_eval.npz"))
    ap.add_argument("--train", default=str(HERE.parents[1] / "sim/build/gpu/out/cmd_train.jsonl"))
    ap.add_argument("--out", default=str(HERE.parents[1] / "sim/build/gpu/figures"))
    a = ap.parse_args()
    Path(a.out).mkdir(parents=True, exist_ok=True)
    d = np.load(a.npz); t = d["t"]

    # 1. commanded vs achieved velocity
    fig, ax = plt.subplots(2, 1, figsize=(11, 7), sharex=True)
    ax[0].plot(t, d["cmd_vx"], "k--", label="commanded vx"); ax[0].plot(t, d["vx"], label="achieved vx")
    ax[1].plot(t, d["cmd_vy"], "k--", label="commanded vy"); ax[1].plot(t, d["vy"], label="achieved vy")
    ax[0].set_ylabel("vx (m/s)"); ax[1].set_ylabel("vy (m/s)"); ax[1].set_xlabel("control step")
    ax[0].set_title("Remote command vs achieved velocity (steerable locomotor)")
    for x in ax: x.legend(); x.grid(alpha=0.3)
    fig.tight_layout(); fig.savefig(Path(a.out) / "command_tracking.png", dpi=110); plt.close(fig)

    # 2. top-down trajectory (the commanded square)
    fig, axi = plt.subplots(figsize=(7, 7))
    sc = axi.scatter(d["x"], d["y"], c=t, cmap="viridis", s=8)
    axi.plot(d["x"], d["y"], lw=0.6, alpha=0.5)
    axi.scatter([d["x"][0]], [d["y"][0]], c="green", s=80, label="start", zorder=5)
    axi.set_aspect("equal"); axi.set_xlabel("x (m)"); axi.set_ylabel("y (m)")
    axi.set_title("Walked path under the remote command square"); axi.legend(); axi.grid(alpha=0.3)
    fig.colorbar(sc, label="control step"); fig.tight_layout()
    fig.savefig(Path(a.out) / "command_trajectory.png", dpi=110); plt.close(fig)

    # 3. training tracking curve (optional)
    tj = Path(a.train)
    if tj.exists():
        rs = [json.loads(l) for l in tj.read_text().splitlines() if l.strip()]
        if rs:
            fig, axi = plt.subplots(figsize=(9, 4))
            axi.plot([r["step"] / 1e6 for r in rs], [r["track"] for r in rs])
            axi.set_xlabel("env-steps (M)"); axi.set_ylabel("velocity-tracking (1=perfect)")
            axi.set_title("Command-following learned over training"); axi.grid(alpha=0.3)
            fig.tight_layout(); fig.savefig(Path(a.out) / "command_training.png", dpi=110); plt.close(fig)

    mv = (np.abs(d["cmd_vx"]) + np.abs(d["cmd_vy"])) > 1e-6
    if mv.sum():
        cmd = np.stack([d["cmd_vx"][mv], d["cmd_vy"][mv]], 1); ach = np.stack([d["vx"][mv], d["vy"][mv]], 1)
        cos = np.sum(cmd * ach, 1) / (np.linalg.norm(cmd, 1) * np.linalg.norm(ach, axis=1) + 1e-6)
        print(f"direction alignment cos = {np.nanmean(cos):+.2f}; figures -> {a.out}/")
    else:
        print(f"figures -> {a.out}/")


if __name__ == "__main__":
    main()
