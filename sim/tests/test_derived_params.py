# SPDX-License-Identifier: MIT
"""Derivation-as-verification (derivation-checklist stage 1.4): every
parameter with a derived_from reference must re-derive from its circuit
spec; the registry and the specs must stay in sync both directions."""

from __future__ import annotations

import math

import derive_params
import pytest
import sim_params


def test_all_derived_params_rederive(params):
    mismatches, missing = derive_params.check(params)
    assert not missing, f"derived_from without a registered derivation: {missing}"
    assert not mismatches, "\n".join(
        f"{p}: committed {a} vs derived {e}" for p, a, e in mismatches
    )


def test_registry_targets_exist(params):
    orphaned = derive_params.registry_targets_exist(params)
    assert not orphaned, f"derivations targeting missing params: {orphaned}"


def test_no_orphaned_components(params):
    unused = derive_params.unused_components(params)
    assert not unused, f"spec components no derivation consumes: {unused}"


def test_registry_targets_are_tagged(params):
    """Every registry target must carry derived_from in params.toml,
    otherwise --check silently skips it."""
    for d in derive_params.DERIVATIONS:
        entry = params.entries[d.target]
        assert entry.derived_from, f"{d.target} missing derived_from tag"


def test_derivation_uses_reference_real_components(params):
    for d in derive_params.DERIVATIONS:
        for component in d.uses:
            assert component in params.entries, (
                f"{d.target} claims to use missing component {component}"
            )


# ---------------------------------------------------------------------------
# Unit-conversion hand checks (the silent-error traps, verified once)
# ---------------------------------------------------------------------------

def test_kv_to_ke_hand_check():
    # Kv = 1000 RPM/V -> 1000*2pi/60 rad/s per volt -> Ke_ll = 1/that.
    assert math.isclose(derive_params.kv_to_ke_line_line(1000.0),
                        60.0 / (2.0 * math.pi * 1000.0), rel_tol=1e-12)
    # A 100 Kv motor: 9.55 mV*s/rad * 10 = 0.0955 V*s/rad line-to-line.
    assert math.isclose(derive_params.kv_to_ke_line_line(100.0),
                        0.09549296585513721, rel_tol=1e-9)


def test_wye_equivalent_hand_check(params):
    # 1.0 Ohm terminal-to-terminal -> 0.5 Ohm per phase (two phases in
    # series between any terminal pair of the wye equivalent).
    assert params.value("motor.R") == params.value("motor_spec.r_line_line") / 2


def test_ke_phase_peak_hand_check(params):
    # Sinusoidal: line-to-line peak = sqrt(3) * phase peak.
    assert math.isclose(
        params.value("motor.Ke") * math.sqrt(3.0),
        params.value("motor_spec.ke_line_line_peak"), rel_tol=1e-12)


def test_emf_topology_alternative(params, tmp_path):
    """The series_only topology must give the ~10x higher cutoff - the
    ambiguity this layer exists to make explicit."""
    text = params.config_path.read_text().replace(
        'filter_topology = { value = "post_divider"',
        'filter_topology = { value = "series_only"')
    cfg = tmp_path / "params.toml"
    cfg.write_text(text)
    alt = sim_params.load(cfg)

    r_series = alt.value("circuit.emf_channel.r_series")
    c = alt.value("circuit.emf_channel.c_filter")
    expected = 1.0 / (2.0 * math.pi * r_series * c)
    derived = [d for d in derive_params.DERIVATIONS
               if d.target == "feedback.emf.rc_cutoff"][0].fn(alt)
    assert math.isclose(derived, expected, rel_tol=1e-12)
    assert derived > 8 * params.value("feedback.emf.rc_cutoff")


def test_unknown_topology_rejected(params, tmp_path):
    text = params.config_path.read_text().replace(
        'filter_topology = { value = "post_divider"',
        'filter_topology = { value = "mystery"')
    cfg = tmp_path / "params.toml"
    cfg.write_text(text)
    alt = sim_params.load(cfg)
    with pytest.raises(ValueError, match="filter_topology"):
        derive_params.check(alt)


def test_measurement_checklist_complete(params):
    """The Q7/Q1 worksheet (derivation-checklist 4.1) must list every
    unmeasured spec component with what it unblocks."""
    text = derive_params.measurement_checklist(params)
    for e in params.entries.values():
        if (e.path.startswith("circuit.") or e.path.startswith("motor_spec.")) \
                and e.status != "measured":
            assert e.path in text, f"worksheet missing {e.path}"
    assert "unblocks:" in text
    assert "(unused?)" not in text or "filter_topology" in text


def test_loader_rejects_bad_derived_from(tmp_path):
    cfg = tmp_path / "params.toml"
    cfg.write_text(
        '[meta]\n'
        'statuses = ["decided"]\n'
        'unconfirmed_statuses = []\n'
        '[x]\n'
        'a = { value = 1.0, unit = "V", status = "decided", '
        'derived_from = "nonsense.table" }\n'
    )
    with pytest.raises(sim_params.ParamConfigError, match="derived_from"):
        sim_params.load(cfg)


def test_loader_rejects_missing_reference_table(tmp_path):
    cfg = tmp_path / "params.toml"
    cfg.write_text(
        '[meta]\n'
        'statuses = ["decided"]\n'
        'unconfirmed_statuses = []\n'
        '[x]\n'
        'a = { value = 1.0, unit = "V", status = "decided", '
        'derived_from = "circuit.ghost" }\n'
    )
    with pytest.raises(sim_params.ParamConfigError, match="missing table"):
        sim_params.load(cfg)
