# SPDX-License-Identifier: MIT
"""Open ADS9224R module schematic (board checklist §3): the generated
passive-network schematic must survive a kicad-cli SPICE-netlist round trip
with values + connectivity intact, and the committed copy must be current."""

from __future__ import annotations

import shutil
import subprocess

import gen_ads9224r_sch
import pytest

needs_kicad = pytest.mark.skipif(shutil.which("kicad-cli") is None,
                                 reason="kicad-cli not installed")


@pytest.fixture(scope="module")
def exported_netlist(params, tmp_path_factory):
    workdir = tmp_path_factory.mktemp("ads9224r_kicad")
    sch = workdir / "module.kicad_sch"
    sch.write_text(gen_ads9224r_sch.render_schematic(params))
    netlist = workdir / "module.cir"
    result = subprocess.run(
        ["kicad-cli", "sch", "export", "netlist", "--format", "spice",
         "-o", str(netlist), str(sch)],
        capture_output=True, text=True, timeout=120)
    assert result.returncode == 0, result.stderr
    assert netlist.is_file()
    return netlist


@needs_kicad
def test_module_schematic_roundtrip(params, exported_netlist):
    errors = gen_ads9224r_sch.check(params, exported_netlist)
    assert not errors, "\n".join(errors)


@needs_kicad
def test_committed_module_schematic_is_current(params):
    """The schematic checked into hw/ must match a fresh render (regenerate
    with gen_ads9224r_sch.py after changing [circuit.ads9224r_module])."""
    committed = gen_ads9224r_sch.SCH_PATH
    assert committed.is_file(), "run sim/scripts/gen_ads9224r_sch.py"
    assert committed.read_text() == gen_ads9224r_sch.render_schematic(params)
