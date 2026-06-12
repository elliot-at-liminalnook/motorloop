#!/usr/bin/env python3
"""Plot one or more shared-schema trace CSVs (phase currents, speed, and
whatever other known columns are present) for human review.

Usage:
  python3 sim/scripts/plot_trace.py sim/build/bench_closed_loop.csv \
      [more.csv ...] [--output plot.png]
"""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

CURRENT_COLS = ["ia_a", "ib_a", "ic_a"]
PANELS = [
    ("Phase currents [A]", CURRENT_COLS),
    ("Speed [rad/s]", ["omega_rad_s", "speed_rad_s"]),
    ("Mechanics / drive", ["theta_rad", "duty", "sector"]),
    ("ADC codes", ["cur_code", "emf_code", "vbus_code"]),
]


def to_float(value: str) -> float:
    if value == "True":
        return 1.0
    if value == "False":
        return 0.0
    return float(value)


def load_csv(path: Path) -> dict[str, list[float]]:
    with path.open() as f:
        rows = list(csv.DictReader(f))
    return {key: [to_float(r[key]) for r in rows] for key in rows[0]}


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("traces", nargs="+", type=Path)
    parser.add_argument("--output", type=Path, default=None)
    args = parser.parse_args()

    datasets = [(p, load_csv(p)) for p in args.traces]

    panels = [(title, cols) for title, cols in PANELS
              if any(c in data for _, data in datasets for c in cols)]
    fig, axes = plt.subplots(len(panels), 1, figsize=(11, 3 * len(panels)),
                             sharex=True)
    if len(panels) == 1:
        axes = [axes]

    for ax, (title, cols) in zip(axes, panels):
        for path, data in datasets:
            label_prefix = f"{path.stem}: " if len(datasets) > 1 else ""
            for col in cols:
                if col in data:
                    ax.plot(data["time_s"], data[col],
                            label=f"{label_prefix}{col}", linewidth=0.8)
        ax.set_title(title)
        ax.grid(True, alpha=0.3)
        ax.legend(loc="upper right", fontsize=8)
    axes[-1].set_xlabel("time [s]")
    fig.tight_layout()

    output = args.output or args.traces[0].with_suffix(".png")
    fig.savefig(output, dpi=120)
    print(f"wrote {output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
