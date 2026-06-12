"""Loader and convention tests for sim/config/params.toml (stage 1.3)."""

from __future__ import annotations

import textwrap

import pytest
import sim_params


def test_config_loads_and_has_unconfirmed_flags(params):
    assert len(params.entries) > 20
    flagged = params.unconfirmed()
    # The project currently rests on unconfirmed assumptions (Q1, Q5, Q7...);
    # if this ever reaches zero, celebrate and update this test.
    assert flagged, "expected unconfirmed parameters at this project stage"
    for entry in flagged:
        assert entry.blocked_by or entry.note, entry.path


def test_banner_lists_every_unconfirmed_param(params):
    banner = params.banner_text()
    for entry in params.unconfirmed():
        assert entry.path in banner


def test_motor_params_are_placeholders_blocked_by_q1(params):
    for name in ("R", "L", "Ke", "Kt", "J", "B"):
        entry = params.entries[f"motor.{name}"]
        assert entry.status == "placeholder"
        assert entry.blocked_by == "Q1"


def _load_snippet(tmp_path, body: str):
    cfg = tmp_path / "params.toml"
    cfg.write_text(
        textwrap.dedent(
            """\
            [meta]
            schema = 1
            statuses = ["measured", "datasheet", "decided", "assumed"]
            unconfirmed_statuses = ["assumed"]
            """
        )
        + textwrap.dedent(body)
    )
    return sim_params.load(cfg)


def test_missing_status_rejected(tmp_path):
    with pytest.raises(sim_params.ParamConfigError, match="missing required"):
        _load_snippet(tmp_path, '[x]\na = { value = 1.0, unit = "V" }\n')


def test_unknown_status_rejected(tmp_path):
    with pytest.raises(sim_params.ParamConfigError, match="not in"):
        _load_snippet(
            tmp_path, '[x]\na = { value = 1.0, unit = "V", status = "vibes" }\n'
        )


def test_unconfirmed_without_reference_rejected(tmp_path):
    with pytest.raises(sim_params.ParamConfigError, match="blocked_by or note"):
        _load_snippet(
            tmp_path, '[x]\na = { value = 1.0, unit = "V", status = "assumed" }\n'
        )


def test_bad_blocked_by_format_rejected(tmp_path):
    with pytest.raises(sim_params.ParamConfigError, match="must look like"):
        _load_snippet(
            tmp_path,
            '[x]\na = { value = 1.0, unit = "V", status = "assumed", blocked_by = "issue-7" }\n',
        )


def test_bare_value_rejected(tmp_path):
    with pytest.raises(sim_params.ParamConfigError, match="bare value"):
        _load_snippet(tmp_path, "[x]\na = 1.0\n")
