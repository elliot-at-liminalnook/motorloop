# SPDX-License-Identifier: MIT
"""Fit a command-conditioned CPG weight map from gait/transition datasets.

This does not mutate the gait vectors.  It learns a small linear map:

  [1, cmd_x, cmd_y, speed, prev_cmd_x, prev_cmd_y, command_change, wp2_active]
      -> [forward_w, backward_w, left_w, right_w]

The output JSON can be loaded by `cpg_teacher.py` with:

  CMD_CPG_ROUTER=param_linear CMD_CPG_MAP_JSON=<map.json>
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

import numpy as np

OUT = Path(os.environ.get("CODESIGN_OUT", "/root/proj/out"))

FEATURE_NAMES = (
    "bias",
    "cmd_x_over_vmax",
    "cmd_y_over_vmax",
    "speed_over_vmax",
    "prev_cmd_x_over_vmax",
    "prev_cmd_y_over_vmax",
    "command_change_over_vmax",
    "active_waypoint2",
)

WEIGHT_NAMES = ("forward", "backward", "left", "right")


def command_targets(cmd: np.ndarray, vmax: float) -> np.ndarray:
    gates = np.stack([
        np.clip(cmd[:, 0] / vmax, 0.0, 1.0),
        np.clip(-cmd[:, 0] / vmax, 0.0, 1.0),
        np.clip(cmd[:, 1] / vmax, 0.0, 1.0),
        np.clip(-cmd[:, 1] / vmax, 0.0, 1.0),
    ], axis=1)
    denom = np.maximum(gates.sum(axis=1, keepdims=True), 1e-6)
    return gates / denom


def features(cmd: np.ndarray, prev_cmd: np.ndarray, active: np.ndarray, vmax: float) -> np.ndarray:
    speed = np.clip(np.linalg.norm(cmd, axis=1) / vmax, 0.0, 1.0)
    change = np.clip(np.linalg.norm(cmd - prev_cmd, axis=1) / vmax, 0.0, 1.0)
    active_wp2 = (active.reshape(-1) == 2).astype(np.float32)
    return np.column_stack([
        np.ones(len(cmd), dtype=np.float32),
        cmd[:, 0] / vmax,
        cmd[:, 1] / vmax,
        speed,
        prev_cmd[:, 0] / vmax,
        prev_cmd[:, 1] / vmax,
        change,
        active_wp2,
    ]).astype(np.float32)


def sample_weights(data: dict[str, np.ndarray]) -> np.ndarray:
    n = len(data["command"])
    w = np.ones(n, dtype=np.float32)
    if "alive" in data:
        w *= np.clip(np.asarray(data["alive"], dtype=np.float32).reshape(-1), 0.0, 1.0)
    if "saturation" in data:
        sat = np.asarray(data["saturation"], dtype=np.float32).reshape(-1)
        w *= np.where(sat <= 0.75, 1.0, 0.25)
    if "distance_reduction" in data:
        dr = np.asarray(data["distance_reduction"], dtype=np.float32).reshape(-1)
        w *= np.where(dr > 0.0, 2.0, 0.5)
    if "y_drift" in data and "active" in data:
        yd = np.asarray(data["y_drift"], dtype=np.float32).reshape(-1)
        active = np.asarray(data["active"]).reshape(-1)
        w *= np.where((active == 2) & (yd > 0.0), 0.5, 1.0)
    return np.maximum(w, 1e-4)


def load_npz(path: Path, max_samples: int, rng: np.random.Generator):
    d = np.load(path, allow_pickle=False)
    if "command" not in d:
        return None
    cmd = np.asarray(d["command"], dtype=np.float32)
    if cmd.ndim == 1:
        if cmd.shape != (2,):
            return None
        n = int(np.asarray(d["obs"]).shape[0]) if "obs" in d else 1
        cmd = np.repeat(cmd[None, :], n, axis=0)
    if cmd.ndim != 2 or cmd.shape[1] != 2 or len(cmd) == 0:
        return None
    active = np.asarray(d["active"], dtype=np.int32).reshape(-1) if "active" in d else np.zeros(len(cmd), dtype=np.int32)
    if len(active) != len(cmd):
        active = np.resize(active, len(cmd)).astype(np.int32)
    prev_cmd = np.vstack([cmd[:1], cmd[:-1]]).astype(np.float32)
    data = {"command": cmd, "prev_command": prev_cmd, "active": active}
    for key in ("alive", "saturation", "distance_reduction", "y_drift"):
        if key in d:
            arr = np.asarray(d[key])
            if arr.shape[0] == len(cmd):
                data[key] = arr
    if len(cmd) > max_samples:
        idx = rng.choice(len(cmd), size=max_samples, replace=False)
        data = {k: np.asarray(v)[idx] for k, v in data.items()}
    return data


def fit_ridge(x: np.ndarray, y: np.ndarray, w: np.ndarray, ridge: float) -> np.ndarray:
    sw = np.sqrt(w).reshape(-1, 1)
    xw = x * sw
    yw = y * sw
    reg = ridge * np.eye(x.shape[1], dtype=np.float32)
    reg[0, 0] = ridge * 0.1
    return np.linalg.solve(xw.T @ xw + reg, xw.T @ yw).astype(np.float32)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset-dir", action="append", required=True,
                    help="dataset directory containing .npz files; pass multiple times")
    ap.add_argument("--tag", default="cpg_param_map_v1")
    ap.add_argument("--vmax", type=float, default=0.35)
    ap.add_argument("--ridge", type=float, default=1e-3)
    ap.add_argument("--max-samples-per-file", type=int, default=4096)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    rng = np.random.default_rng(args.seed)
    rows = []
    source_counts = {}
    for root in args.dataset_dir:
        root_path = Path(root)
        for path in sorted(root_path.glob("*.npz")):
            data = load_npz(path, args.max_samples_per_file, rng)
            if data is None:
                continue
            rows.append((path, data))
            source_counts[str(path)] = int(len(data["command"]))
    if not rows:
        raise SystemExit("no usable dataset samples found")

    xs = []
    ys = []
    ws = []
    for _, data in rows:
        cmd = data["command"]
        prev = data["prev_command"]
        active = data["active"]
        xs.append(features(cmd, prev, active, args.vmax))
        ys.append(command_targets(cmd, args.vmax))
        ws.append(sample_weights(data))
    x = np.concatenate(xs, axis=0)
    y = np.concatenate(ys, axis=0)
    w = np.concatenate(ws, axis=0)
    coef = fit_ridge(x, y, w, args.ridge)
    pred = np.maximum(0.0, x @ coef)
    pred = pred / np.maximum(pred.sum(axis=1, keepdims=True), 1e-6)
    mse = float(np.average(np.sum((pred - y) ** 2, axis=1), weights=w))
    total = int(len(x))
    report = {
        "tag": args.tag,
        "map_type": "weighted_ridge_linear_cardinal_weights",
        "feature_names": FEATURE_NAMES,
        "weight_names": WEIGHT_NAMES,
        "vmax": float(args.vmax),
        "ridge": float(args.ridge),
        "coef": coef.tolist(),
        "weighted_mse": mse,
        "total_samples": total,
        "dataset_dirs": args.dataset_dir,
        "source_counts": source_counts,
        "target_weight_mean": np.average(y, axis=0, weights=w).tolist(),
        "pred_weight_mean": np.average(pred, axis=0, weights=w).tolist(),
    }
    OUT.mkdir(parents=True, exist_ok=True)
    out = OUT / f"{args.tag}_cpg_param_map.json"
    out.write_text(json.dumps(report, indent=2))
    print(json.dumps(report, indent=2))
    print(f"saved {out}")


if __name__ == "__main__":
    main()
