# SPDX-License-Identifier: MIT
"""Checkpoint delta/conflict diagnostics for curriculum and self-play skills.

This is a lightweight bridge toward adapter-style continual learning: it treats
checkpoint differences as skill vectors and reports whether a new update is
aligned, orthogonal, or conflicting with reference updates.
"""

from __future__ import annotations

import argparse, json, pickle
from pathlib import Path

import jax
import numpy as np


def _select(params, part: str):
    if part == "policy":
        return params[1] if isinstance(params, (tuple, list)) and len(params) > 1 else params
    if part == "value":
        return params[2] if isinstance(params, (tuple, list)) and len(params) > 2 else params
    return params


def _flat_delta(before, after, part: str) -> np.ndarray:
    b_leaves = jax.tree_util.tree_leaves(_select(before, part))
    a_leaves = jax.tree_util.tree_leaves(_select(after, part))
    chunks = []
    skipped = 0
    for b, a in zip(b_leaves, a_leaves):
        b = np.asarray(b); a = np.asarray(a)
        if b.shape != a.shape or not np.issubdtype(b.dtype, np.number):
            skipped += 1
            continue
        chunks.append((a.astype(np.float64) - b.astype(np.float64)).ravel())
    if not chunks:
        raise ValueError("no matching numeric leaves found")
    if skipped:
        print(f"skipped {skipped} non-matching/non-numeric leaves")
    return np.concatenate(chunks)


def _flat_params(params, part: str) -> np.ndarray:
    chunks = []
    for x in jax.tree_util.tree_leaves(_select(params, part)):
        x = np.asarray(x)
        if np.issubdtype(x.dtype, np.number):
            chunks.append(x.astype(np.float64).ravel())
    return np.concatenate(chunks)


def _cos(a, b):
    return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-12))


def compare(delta, ref, top_frac=0.01):
    nz = (np.abs(delta) > 1e-12) & (np.abs(ref) > 1e-12)
    sign_conflict = float((np.sign(delta[nz]) != np.sign(ref[nz])).mean()) if nz.any() else 0.0
    top = max(1, int(float(top_frac) * len(delta)))
    td = np.argpartition(np.abs(delta), -top)[-top:]
    tr = np.argpartition(np.abs(ref), -top)[-top:]
    overlap = float(len(set(td.tolist()) & set(tr.tolist())) / top)
    return dict(cosine=_cos(delta, ref), sign_conflict_frac=sign_conflict,
                top1pct_overlap=overlap)


def _verdict(cosine, sign_conflict, overlap, aligned_cos, orthogonal_cos,
             conflict_cos, conflict_sign):
    if cosine <= conflict_cos or (sign_conflict >= conflict_sign and overlap >= 0.05):
        return "conflicting"
    if cosine >= aligned_cos and sign_conflict < conflict_sign:
        return "aligned"
    if abs(cosine) <= orthogonal_cos or overlap < 0.01:
        return "orthogonal"
    return "mixed"


def decide(rec, *, tiny_rel, tiny_abs, aligned_cos, orthogonal_cos,
           conflict_cos, conflict_sign):
    if rec["relative_norm"] < tiny_rel or rec["p95_abs"] < tiny_abs:
        return dict(action="prune_or_ignore", rank_hint=0,
                    reason="delta is tiny relative to the base policy")
    refs = rec.get("refs", [])
    if not refs:
        return dict(action="allocate_skill_adapter", rank_hint=4,
                    reason="no reference deltas to share with yet")
    for r in refs:
        r["verdict"] = _verdict(r["cosine"], r["sign_conflict_frac"], r["top1pct_overlap"],
                                aligned_cos, orthogonal_cos, conflict_cos, conflict_sign)
    best = max(refs, key=lambda r: r["cosine"])
    worst = min(refs, key=lambda r: r["cosine"])
    if worst["verdict"] == "conflicting":
        return dict(action="separate_adapter_protect_shared", rank_hint=8,
                    replay_kl="high", best_ref=best["ref"], conflict_ref=worst["ref"],
                    reason="new skill vector conflicts with at least one protected delta")
    if best["verdict"] == "aligned":
        return dict(action="share_or_merge_adapter_capacity", rank_hint=2,
                    replay_kl="normal", best_ref=best["ref"],
                    reason="new skill vector is aligned with an existing skill delta")
    if all(r["verdict"] == "orthogonal" for r in refs):
        return dict(action="allocate_separate_adapter_rank", rank_hint=6,
                    replay_kl="normal", best_ref=best["ref"],
                    reason="new skill vector is mostly orthogonal to existing deltas")
    return dict(action="route_mixture_and_test_ablation", rank_hint=4,
                replay_kl="elevated", best_ref=best["ref"],
                reason="mixed alignment; needs replay and ablation before sharing")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--before", required=True)
    ap.add_argument("--after", required=True)
    ap.add_argument("--part", choices=["policy", "value", "all"], default="policy")
    ap.add_argument("--ref-delta", action="append", default=[],
                    help="previous .npz from this tool; may be passed multiple times")
    ap.add_argument("--out-json", default=None)
    ap.add_argument("--out-npz", default=None)
    ap.add_argument("--top-frac", type=float, default=0.01)
    ap.add_argument("--tiny-rel", type=float, default=1e-4)
    ap.add_argument("--tiny-abs", type=float, default=1e-7)
    ap.add_argument("--aligned-cos", type=float, default=0.15)
    ap.add_argument("--orthogonal-cos", type=float, default=0.05)
    ap.add_argument("--conflict-cos", type=float, default=-0.05)
    ap.add_argument("--conflict-sign", type=float, default=0.55)
    args = ap.parse_args()

    before = pickle.load(open(args.before, "rb"))
    after = pickle.load(open(args.after, "rb"))
    delta = _flat_delta(before, after, args.part)
    base = _flat_params(before, args.part)
    norm = float(np.linalg.norm(delta))
    rec = dict(before=str(args.before), after=str(args.after), part=args.part,
               dim=int(delta.size), delta_norm=norm,
               relative_norm=float(norm / (np.linalg.norm(base) + 1e-12)),
               mean_abs=float(np.mean(np.abs(delta))),
               p95_abs=float(np.quantile(np.abs(delta), 0.95)),
               tiny_frac=float((np.abs(delta) < 1e-6).mean()),
               refs=[])
    for ref_path in args.ref_delta:
        ref = np.load(ref_path)["delta"]
        n = min(len(delta), len(ref))
        rec["refs"].append({"ref": ref_path, **compare(delta[:n], ref[:n], args.top_frac)})
    rec["decision"] = decide(rec, tiny_rel=args.tiny_rel, tiny_abs=args.tiny_abs,
                             aligned_cos=args.aligned_cos, orthogonal_cos=args.orthogonal_cos,
                             conflict_cos=args.conflict_cos, conflict_sign=args.conflict_sign)

    out_npz = Path(args.out_npz) if args.out_npz else Path(args.after).with_suffix(".delta.npz")
    out_json = Path(args.out_json) if args.out_json else Path(args.after).with_suffix(".delta.json")
    np.savez_compressed(out_npz, delta=delta)
    out_json.write_text(json.dumps(rec, indent=2))
    print(json.dumps(rec, indent=2))


if __name__ == "__main__":
    main()
