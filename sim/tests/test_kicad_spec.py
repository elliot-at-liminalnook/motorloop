"""Derivation-checklist stage 3: the generated KiCad schematic mirror.

The [circuit.*] tables stay the primary source; the schematic is generated
from them and must survive a kicad-cli SPICE-netlist round trip with values
and connectivity intact."""

from __future__ import annotations

import shutil
import subprocess
from pathlib import Path

import gen_kicad_sch
import pytest

PROJECT_ROOT = Path(__file__).resolve().parents[2]

needs_kicad = pytest.mark.skipif(shutil.which("kicad-cli") is None,
                                 reason="kicad-cli not installed")


@pytest.fixture(scope="module")
def exported_netlist(params, tmp_path_factory):
    workdir = tmp_path_factory.mktemp("kicad")
    sch = workdir / "feedback_circuits.kicad_sch"
    sch.write_text(gen_kicad_sch.render_schematic(params))
    netlist = workdir / "feedback_circuits.cir"
    result = subprocess.run(
        ["kicad-cli", "sch", "export", "netlist", "--format", "spice",
         "-o", str(netlist), str(sch)],
        capture_output=True, text=True, timeout=120)
    assert result.returncode == 0, result.stderr
    assert netlist.is_file()
    return netlist


@needs_kicad
def test_schematic_roundtrip_matches_circuit_tables(params, exported_netlist):
    errors = gen_kicad_sch.check(params, exported_netlist)
    assert not errors, "\n".join(errors)


@needs_kicad
def test_committed_schematic_is_current(params):
    """The schematic checked into hw/ must match a fresh render (regenerate
    with gen_kicad_sch.py after changing [circuit.*])."""
    committed = gen_kicad_sch.SCH_PATH
    assert committed.is_file(), "run sim/scripts/gen_kicad_sch.py"
    assert committed.read_text() == gen_kicad_sch.render_schematic(params)


def test_si_value_formatting():
    cases = [(95.3e3, "95.3k"), (10e3, "10k"), (1e3, "1k"),
             (100e-9, "100n"), (53.6e3, "53.6k"), (20e-12, "20p"),
             (2e-3, "2m")]
    for value, expected in cases:
        assert gen_kicad_sch.to_si(value, "") == expected, value
