# SPDX-License-Identifier: MIT
"""Tier-3 vendor-macromodel cross-checks for the ADS9224R module
(notes/ads9224r-sim-validation-checklist.md §3).

These run TI's portal-gated SPICE macromodels (THS4551 / REF6041) when the .LIB
files are present, and SKIP otherwise - the same pattern as
test_ti_vendor_amp_model (DRV8301). Present, they promote the Tier-2
datasheet-model numbers to "vendor-model-validated". Absent, CI stays green.
"""

from __future__ import annotations

import math
import shutil

import pytest
import spice_runner

needs_ngspice = pytest.mark.skipif(shutil.which("ngspice") is None,
                                   reason="ngspice not installed")


@needs_ngspice
def test_ths4551_vendor_bandwidth(params):
    """The THS4551 macromodel's closed-loop bandwidth ~= GBW / noise_gain, a
    cross-check of the datasheet GBW used by the Tier-2 noise model."""
    if not spice_runner.THS4551_LIB.is_file():
        pytest.skip("THS4551.LIB not present (docs/ti-simulation-models/ths4551)")
    data = spice_runner.run_netlist(
        "ths4551_vendor", params,
        aux_files={"THS4551.LIB": spice_runner.THS4551_LIB},
        compat="psa")["ths4551_vendor_ac.out"]
    dc_gain_db = data[0][1]
    target = dc_gain_db - 3.0
    fc = None
    for (f1, g1), (f2, g2) in zip(data, data[1:]):
        if g1 >= target >= g2:
            fc = f1 * (f2 / f1) ** ((g1 - target) / (g1 - g2))
            break
    assert fc is not None, "no -3 dB crossing"
    gbw = params.value("circuit.ths4551.gbw")
    noise_gain = 1.0 + (params.value("circuit.ads9224r_module.fda_rf")
                        / params.value("circuit.ads9224r_module.fda_rg"))
    expected_bw = gbw / noise_gain
    # Macromodel BW within 2x of the single-pole estimate (loose: the model
    # carries higher-order poles the datasheet GBW alone does not).
    assert 0.5 * expected_bw < fc < 2.0 * expected_bw, (
        f"vendor BW {fc/1e6:.1f} MHz vs estimate {expected_bw/1e6:.1f} MHz")
    # Closed-loop DC gain matches the configured differential gain (20 -> 26 dB).
    g_lin = 10 ** (dc_gain_db / 20.0)
    assert math.isclose(g_lin, params.value("feedback.current_ads9224r.fda_gain"),
                        rel_tol=0.1)
