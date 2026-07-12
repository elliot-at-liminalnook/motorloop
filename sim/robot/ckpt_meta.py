# SPDX-License-Identifier: MIT
"""T7 checkpoint semantics sidecars.

Every checkpoint pickle gets a `<name>.pkl.meta.json` recording what the params
MEAN — the things a shape check cannot see:

  model_hash        sha256 of the MJCF the policy trained on (a policy trained
                    on the 1 N·m gear-bug body is not the same artifact as one
                    trained on the 12.97 N·m body, even at identical shapes)
  action_semantics  how actions become forces ("torque_direct@50hz" today;
                    "pd_target@250hz:<scale>" after the B.1 migration). Shapes
                    matched while torque-trained opponents were silently
                    mis-driven under changed semantics — never again.
  obs_size / obs_version   layout identity for the obs vector/dict
  behavior          last benchmark behavioral numbers at save time (bh_disp
                    etc.) — the base line a warm-start acceptance check rolls
                    against ("plumbing ok" is not transfer)

Loaders keep reading raw pickles (sidecars are additive); the frozen-opponent
path REJECTS on mismatch or missing sidecar — every pre-2026-07 checkpoint is
retired wholesale (gear-bug body), so "no sidecar" means "retired" unless
--allow-legacy-opponent is passed deliberately.
"""
from __future__ import annotations

import hashlib
import json
import time
from pathlib import Path

# Bump when the action pathway changes meaning (B.1 PD migration did).
TORQUE_SEMANTICS = "torque_direct@50hz"
COMMANDED_PD_SEMANTICS = "pd_target@50hz_held:scale=1.0"


def fighter_semantics(action_mode: str, pd_action_scale: float = 0.4) -> str:
    """The fighter env's action semantics string (B.1: per-substep PD is the default)."""
    if action_mode == "pd":
        return f"pd_target@250hz:scale={pd_action_scale:g}"
    return TORQUE_SEMANTICS


def current_model_hash(build_xml: str) -> str:
    return hashlib.sha256(build_xml.encode()).hexdigest()[:16]


def meta_path(pkl_path) -> Path:
    return Path(str(pkl_path) + ".meta.json")


def write_meta(pkl_path, *, action_semantics: str, obs_size, model_hash: str,
               behavior: dict | None = None, extra: dict | None = None) -> None:
    """Best-effort sidecar write — never let bookkeeping kill a training run."""
    try:
        meta = dict(action_semantics=action_semantics,
                    obs_size=obs_size if isinstance(obs_size, (int, dict)) else str(obs_size),
                    model_hash=model_hash, behavior=behavior or {},
                    saved_at=time.time(), **(extra or {}))
        meta_path(pkl_path).write_text(json.dumps(meta, indent=2, default=str))
    except OSError as e:
        print(f"  [ckpt-meta] sidecar write failed for {pkl_path}: {e}", flush=True)


def read_meta(pkl_path) -> dict | None:
    p = meta_path(pkl_path)
    if not p.exists():
        return None
    try:
        return json.loads(p.read_text())
    except (OSError, json.JSONDecodeError):
        return None


def check_semantics(pkl_path, *, expected_semantics: str, expected_model_hash: str | None,
                    role: str = "checkpoint", allow_legacy: bool = False) -> dict | None:
    """Raise unless the sidecar says this checkpoint means what the caller assumes.

    Missing sidecar = pre-2026-07 artifact (gear-bug body) = retired, unless the
    caller explicitly opts into legacy loading.
    """
    meta = read_meta(pkl_path)
    if meta is None:
        if allow_legacy:
            print(f"  [ckpt-meta] {role} '{Path(pkl_path).name}': NO sidecar — legacy "
                  f"(pre-gear-fix) checkpoint loaded by explicit override.", flush=True)
            return None
        raise ValueError(
            f"{role} '{Path(pkl_path).name}' has no .meta.json sidecar. Every checkpoint "
            f"from before the 2026-07 gear fix trained a body with ~8% of design torque "
            f"and is retired; if you REALLY mean to load it, pass the legacy override.")
    if meta.get("action_semantics") != expected_semantics:
        raise ValueError(
            f"{role} '{Path(pkl_path).name}' was trained with action semantics "
            f"'{meta.get('action_semantics')}' but this env drives actions as "
            f"'{expected_semantics}'. Shapes would match; behavior would be garbage. "
            f"Retrain or retire this artifact.")
    if expected_model_hash and meta.get("model_hash") != expected_model_hash:
        msg = (f"{role} '{Path(pkl_path).name}' trained on model "
               f"{meta.get('model_hash')} != current {expected_model_hash}; body physics "
               "changed, so identical tensor shapes do not make it reusable")
        if allow_legacy:
            print(f"  [ckpt-meta] OVERRIDE: {msg}", flush=True)
        else:
            raise ValueError(msg)
    return meta
