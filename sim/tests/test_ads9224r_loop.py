# SPDX-License-Identifier: MIT
"""Tier-4 system-level measurement (ads9224r-sim-validation-checklist §4):
bridge the SPICE-validated front-end to the FOC current loop.

The Tier-2/3 sims validate the analog front-end (noise, settling). Here that
result is referred to the loop: the front-end output-voltage noise becomes a
phase-current measurement noise (/(gain*shunt)), combined RSS with the ADS9224R's
own transition noise, and checked against a current-loop budget. The
part-comparison study (test_part_comparison T3/T4) separately confirms the
*skew* elimination (Q21); together they are the full front-end story:
simultaneous sampling (no skew) + a small, bounded noise floor.
"""

from __future__ import annotations

import shutil

import adc_metrics as M
import pytest
import spice_runner

needs_ngspice = pytest.mark.skipif(shutil.which("ngspice") is None,
                                   reason="ngspice not installed")


def _front_end_current_noise(params):
    """SPICE front-end output noise -> phase-current RMS noise (amps)."""
    spec = spice_runner.run_netlist("ads9224r_noise", params)["ads9224r_noise.out"]
    v_rms = M.integrate_noise_rms([r[0] for r in spec], [r[1] for r in spec])
    gain = params.value("feedback.current_ads9224r.fda_gain")
    shunt = params.value("circuit.ads9224r_module.shunt")
    return M.current_noise_rms_a(v_rms, gain, shunt)


@needs_ngspice
def test_loop_current_noise_budget(params):
    """The validated front-end + the ADC transition noise contribute a small,
    bounded current-measurement noise to the FOC loop (effective bits >= 15,
    and well under 0.1% of full-scale)."""
    i_fe = _front_end_current_noise(params)              # amps RMS (front-end)
    cpa = params.value("feedback.current_ads9224r.codes_per_amp")
    fs = params.value("feedback.current_ads9224r.full_scale_a")

    # ADC's own transition noise (datasheet, in LSB) -> amps.
    adc_lsb = params.value("circuit.ads9224r_adc.transition_noise_lsb")
    i_adc = adc_lsb / cpa
    i_total = M.rss(i_fe, i_adc)                          # combined loop noise

    # As effective bits over the +/- full-scale span.
    eff_bits = M.enob_from_snr(M.snr_db_from_noise(fs, i_total))
    frac_fs = i_total / fs

    assert i_fe < i_adc * 2.0, (
        f"front-end current noise {i_fe*1e3:.2f} mA should be ~<= the ADC's own "
        f"{i_adc*1e3:.2f} mA (the antialiasing-bounded design)")
    assert eff_bits >= 15.0, f"loop effective bits {eff_bits:.2f} < 15"
    assert frac_fs < 1e-3, f"loop current noise {frac_fs*100:.4f}% of FS"


def test_part_comparison_link_documented():
    """Tier 4 ties to the skew result: the part-comparison study (T3/T4) is the
    loop-level skew/ripple measurement; this module supplies the simultaneous
    (scheme-0) path. Guard that the link target exists."""
    import part_compare as PC
    assert PC.SCHEME_SIMULTANEOUS == 0
    assert "ADS9224R" in PC.SCHEME_LABEL[PC.SCHEME_SIMULTANEOUS]
