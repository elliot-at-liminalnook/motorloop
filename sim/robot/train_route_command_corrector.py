# SPDX-License-Identifier: MIT
"""Train a low-dimensional route command correction head.

Datasets come from ``collect_route_correction_dataset.py`` and include
``head_target``:

    [cmd_dx, cmd_dy, residual_strength, turn_bias, phase_speed_adjust, handoff_timing]

The first two values are normalized command corrections.  This model is meant
to steer the CPG interface directly, not to clone the full 12-D motor residual.
Closed-loop replay remains the promotion gate.
"""

from __future__ import annotations

import argparse
import json
import os
import pickle
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
import optax

OUT = Path(os.environ.get("CODESIGN_OUT", "/root/proj/out"))
HEAD_NAMES = (
    "cmd_dx",
    "cmd_dy",
    "residual_strength",
    "turn_bias",
    "phase_speed_adjust",
    "handoff_timing",
)


def parse_sizes(text: str) -> tuple[int, ...]:
    return tuple(int(x.strip()) for x in text.replace(";", ",").split(",") if x.strip())


def parse_ints(text: str) -> tuple[int, ...]:
    return tuple(int(x.strip()) for x in text.replace(";", ",").split(",") if x.strip())


def dataset_files(path: Path) -> list[Path]:
    if path.is_file():
        return [path]
    return sorted(path.glob("*.npz"))


def load_dataset(path: Path):
    files = dataset_files(path)
    if not files:
        raise FileNotFoundError(f"no .npz files found under {path}")
    xs, ys, seeds, active = [], [], [], []
    for f in files:
        d = np.load(f)
        if "head_target" not in d:
            raise KeyError(f"{f} has no head_target; regenerate with collect_route_correction_dataset.py")
        xs.append(np.asarray(d["features"], dtype=np.float32))
        ys.append(np.asarray(d["head_target"], dtype=np.float32))
        seeds.append(np.asarray(d["seed"], dtype=np.int32))
        active.append(np.asarray(d["active"], dtype=np.int32))
    return {
        "features": np.concatenate(xs, axis=0),
        "target": np.concatenate(ys, axis=0),
        "seed": np.concatenate(seeds, axis=0),
        "active": np.concatenate(active, axis=0),
        "files": [str(f) for f in files],
    }


def init_mlp(key, in_dim: int, out_dim: int, hidden: tuple[int, ...]):
    sizes = (in_dim,) + hidden + (out_dim,)
    keys = jax.random.split(key, len(sizes) - 1)
    params = []
    for k, din, dout in zip(keys, sizes[:-1], sizes[1:]):
        scale = np.sqrt(2.0 / max(din + dout, 1))
        params.append({
            "w": scale * jax.random.normal(k, (din, dout)),
            "b": jnp.zeros((dout,), dtype=jnp.float32),
        })
    return params


def apply_mlp(params, x):
    for layer in params[:-1]:
        x = jnp.tanh(x @ layer["w"] + layer["b"])
    return x @ params[-1]["w"] + params[-1]["b"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", required=True)
    ap.add_argument("--tag", default="route_command_corrector")
    ap.add_argument("--hidden", default="64,64")
    ap.add_argument("--steps", type=int, default=4000)
    ap.add_argument("--batch-size", type=int, default=256)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--val-seed", type=int, default=None)
    ap.add_argument("--val-frac", type=float, default=0.2)
    ap.add_argument("--active-filter", default="")
    ap.add_argument("--active2-weight", type=float, default=1.0)
    ap.add_argument("--active3-weight", type=float, default=1.0)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    data = load_dataset(Path(args.dataset))
    x = data["features"]
    y = data["target"]
    route_seeds = data["seed"]
    active = data["active"]
    sample_w = np.ones((len(x),), dtype=np.float32)
    sample_w *= np.where(active == 2, float(args.active2_weight), 1.0).astype(np.float32)
    sample_w *= np.where(active == 3, float(args.active3_weight), 1.0).astype(np.float32)
    active_filter = parse_ints(args.active_filter)
    if active_filter:
        keep = np.isin(active, list(active_filter))
        x = x[keep]
        y = y[keep]
        route_seeds = route_seeds[keep]
        active = active[keep]
        sample_w = sample_w[keep]
    if len(x) == 0:
        raise ValueError("empty dataset after filtering")

    rng = np.random.default_rng(args.seed)
    if args.val_seed is not None and np.any(route_seeds == args.val_seed):
        val_mask = route_seeds == args.val_seed
    else:
        val_mask = np.zeros((len(x),), dtype=bool)
        val_n = max(1, int(round(len(x) * args.val_frac)))
        val_mask[rng.choice(len(x), size=val_n, replace=False)] = True
    train_mask = ~val_mask
    if not np.any(train_mask) or not np.any(val_mask):
        raise ValueError("empty train or validation split")

    x_train, y_train, w_train = x[train_mask], y[train_mask], sample_w[train_mask]
    x_val, y_val, w_val = x[val_mask], y[val_mask], sample_w[val_mask]
    mean = x_train.mean(axis=0).astype(np.float32)
    std = np.maximum(x_train.std(axis=0), 1e-6).astype(np.float32)
    x_train_n = (x_train - mean) / std
    x_val_n = (x_val - mean) / std

    params = init_mlp(jax.random.PRNGKey(args.seed), x.shape[1], y.shape[1], parse_sizes(args.hidden))
    opt = optax.adam(args.lr)
    opt_state = opt.init(params)

    @jax.jit
    def train_step(params, opt_state, xb, yb, wb):
        def loss_fn(p):
            pred = apply_mlp(p, xb)
            per = jnp.mean((pred - yb) ** 2, axis=-1)
            return jnp.sum(per * wb) / jnp.maximum(jnp.sum(wb), 1e-6)
        loss, grads = jax.value_and_grad(loss_fn)(params)
        updates, opt_state2 = opt.update(grads, opt_state, params)
        return optax.apply_updates(params, updates), opt_state2, loss

    @jax.jit
    def loss_eval(params, xb, yb, wb):
        pred = apply_mlp(params, xb)
        per_dim = jnp.mean((pred - yb) ** 2, axis=0)
        per = jnp.mean((pred - yb) ** 2, axis=-1)
        loss = jnp.sum(per * wb) / jnp.maximum(jnp.sum(wb), 1e-6)
        return loss, per_dim

    hist = []
    n = len(x_train_n)
    for step_i in range(1, args.steps + 1):
        idx = rng.integers(0, n, size=args.batch_size)
        params, opt_state, loss = train_step(
            params,
            opt_state,
            jnp.asarray(x_train_n[idx]),
            jnp.asarray(y_train[idx]),
            jnp.asarray(w_train[idx]),
        )
        if step_i == 1 or step_i % max(args.steps // 10, 1) == 0 or step_i == args.steps:
            tr_n = min(len(x_train_n), 8192)
            tr_loss, tr_dim = loss_eval(params, jnp.asarray(x_train_n[:tr_n]),
                                        jnp.asarray(y_train[:tr_n]), jnp.asarray(w_train[:tr_n]))
            va_loss, va_dim = loss_eval(params, jnp.asarray(x_val_n), jnp.asarray(y_val), jnp.asarray(w_val))
            row = {
                "step": int(step_i),
                "batch_loss": float(loss),
                "train_mse": float(tr_loss),
                "val_mse": float(va_loss),
                "train_dim_mse": {name: float(v) for name, v in zip(HEAD_NAMES, np.asarray(tr_dim))},
                "val_dim_mse": {name: float(v) for name, v in zip(HEAD_NAMES, np.asarray(va_dim))},
            }
            hist.append(row)
            print(
                f"[route-command-corrector] step {step_i:05d} "
                f"train={row['train_mse']:.6f} val={row['val_mse']:.6f}",
                flush=True,
            )

    artifact = {
        "params": params,
        "feature_mean": mean,
        "feature_std": std,
        "hidden": parse_sizes(args.hidden),
        "feature_dim": int(x.shape[1]),
        "head_dim": int(y.shape[1]),
        "head_names": HEAD_NAMES,
        "dataset_files": data["files"],
        "history": hist,
        "val_seed": args.val_seed,
        "active_filter": [int(v) for v in active_filter],
        "active2_weight": float(args.active2_weight),
        "active3_weight": float(args.active3_weight),
    }
    OUT.mkdir(parents=True, exist_ok=True)
    out_pkl = OUT / f"{args.tag}.pkl"
    with open(out_pkl, "wb") as f:
        pickle.dump(artifact, f)
    report = {
        "tag": args.tag,
        "artifact": str(out_pkl),
        "dataset_files": data["files"],
        "samples": int(len(x)),
        "train_samples": int(np.sum(train_mask)),
        "val_samples": int(np.sum(val_mask)),
        "feature_dim": int(x.shape[1]),
        "head_dim": int(y.shape[1]),
        "head_names": list(HEAD_NAMES),
        "val_seed": args.val_seed,
        "active_filter": [int(v) for v in active_filter],
        "history": hist,
        "final": hist[-1] if hist else {},
    }
    out_json = OUT / f"{args.tag}_report.json"
    out_json.write_text(json.dumps(report, indent=2))
    print(json.dumps(report, indent=2), flush=True)
    print(f"saved {out_pkl} and {out_json}", flush=True)


if __name__ == "__main__":
    main()
