# SPDX-License-Identifier: MIT
"""Checkpoint semantics must fail closed before any training state is reused."""

from __future__ import annotations

import json

import pytest

import ckpt_meta


def _write_sidecar(path, **overrides):
    meta = {
        "action_semantics": ckpt_meta.COMMANDED_PD_SEMANTICS,
        "obs_size": 50,
        "model_hash": "model-a",
        "behavior": {},
    }
    meta.update(overrides)
    ckpt_meta.meta_path(path).write_text(json.dumps(meta))


def test_model_hash_mismatch_is_a_hard_failure(tmp_path):
    checkpoint = tmp_path / "policy.pkl"
    checkpoint.write_bytes(b"placeholder")
    _write_sidecar(checkpoint)
    with pytest.raises(ValueError, match="body physics changed"):
        ckpt_meta.check_semantics(
            checkpoint,
            expected_semantics=ckpt_meta.COMMANDED_PD_SEMANTICS,
            expected_model_hash="model-b",
        )


def test_explicit_legacy_override_is_required_for_model_mismatch(tmp_path):
    checkpoint = tmp_path / "policy.pkl"
    checkpoint.write_bytes(b"placeholder")
    _write_sidecar(checkpoint)
    got = ckpt_meta.check_semantics(
        checkpoint,
        expected_semantics=ckpt_meta.COMMANDED_PD_SEMANTICS,
        expected_model_hash="model-b",
        allow_legacy=True,
    )
    assert got["model_hash"] == "model-a"


def test_action_semantics_mismatch_never_loads(tmp_path):
    checkpoint = tmp_path / "policy.pkl"
    checkpoint.write_bytes(b"placeholder")
    _write_sidecar(checkpoint, action_semantics=ckpt_meta.TORQUE_SEMANTICS)
    with pytest.raises(ValueError, match="action semantics"):
        ckpt_meta.check_semantics(
            checkpoint,
            expected_semantics=ckpt_meta.COMMANDED_PD_SEMANTICS,
            expected_model_hash="model-a",
            allow_legacy=True,
        )
