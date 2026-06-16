# SPDX-License-Identifier: MIT
"""Derivation-checklist stage 2: ngspice cross-checks of the closed-form
derivations, plus the TI vendor-model comparison. Results are cached in
sim/build/spice, so a warm run costs milliseconds."""

from __future__ import annotations

import math
import shutil

import pytest
import spice_runner

needs_ngspice = pytest.mark.skipif(shutil.which("ngspice") is None,
                                   reason="ngspice not installed (stage 0.2)")


def linear_fit(points):
    n = len(points)
    sx = sum(x for x, _ in points)
    sy = sum(y for _, y in points)
    sxx = sum(x * x for x, _ in points)
    sxy = sum(x * y for x, y in points)
    slope = (n * sxy - sx * sy) / (n * sxx - sx * sx)
    intercept = (sy - slope * sx) / n
    return slope, intercept


@needs_ngspice
def test_emf_channel_ac(params):
    """AC analysis: DC gain == divider ratio, -3dB pole == rc_cutoff."""
    data = spice_runner.run_netlist("emf_channel", params)["emf_ac.out"]
    dc_gain = data[0][1]
    assert math.isclose(dc_gain, params.value("feedback.emf.divider_ratio"),
                        rel_tol=1e-4)

    target = dc_gain / math.sqrt(2)
    fc = None
    for (f1, v1), (f2, v2) in zip(data, data[1:]):
        if v1 >= target >= v2:
            frac = (v1 - target) / (v1 - v2)
            fc = f1 * (f2 / f1) ** frac
            break
    assert fc is not None, "no -3dB crossing found"
    assert math.isclose(fc, params.value("feedback.emf.rc_cutoff"),
                        rel_tol=2e-3), (
        f"SPICE pole {fc:.2f} Hz vs derived "
        f"{params.value('feedback.emf.rc_cutoff'):.2f} Hz"
    )

    # Single-pole adequacy: response must track the one-pole model closely
    # through the decade above the cutoff (validates the behavioral chain).
    worst = 0.0
    for f, v in data:
        if fc / 10 <= f <= fc * 10:
            model = dc_gain / math.sqrt(1.0 + (f / fc) ** 2)
            worst = max(worst, abs(v - model) / dc_gain)
    assert worst < 0.01, f"single-pole fit residual {worst:.4f}"


@needs_ngspice
def test_iout_channel_dc(params):
    """DC sweep: slope == gain*shunt, intercept == offset, swing == rails."""
    data = spice_runner.run_netlist("iout_channel", params)["iout_dc.out"]
    lin = [(i, v) for i, v in data if abs(i) < 30]
    slope, intercept = linear_fit(lin)
    expected_slope = (params.value("drv8301.amp_gain")
                      * params.value("feedback.current.shunt"))
    assert math.isclose(slope, expected_slope, rel_tol=1e-6)
    assert math.isclose(intercept, params.value("feedback.current.offset"),
                        rel_tol=1e-6)
    assert math.isclose(max(v for _, v in data), params.value("adc.vref"),
                        rel_tol=1e-6)
    assert min(v for _, v in data) == pytest.approx(0.0, abs=1e-9)


@needs_ngspice
def test_adc_frontend_emf_transient(params):
    """Repeated-sampling transient: validates the per-sample charge-share
    fraction (the derived adc.sample_residual_emf) and shows the
    steady-state reservoir droop the bench reproduces via theft feedback."""
    rows = spice_runner.run_netlist("adc_frontend_emf",
                                    params)["adc_emf_tran.out"]
    ratio = params.value("feedback.emf.divider_ratio")
    v_ideal = 10.0 * ratio  # netlist vphase=10
    v_prev = 1.65           # netlist vprev
    d = params.value("adc.sample_residual_emf")
    t_window = 1.5 / params.value("adc.sclk")

    def held_after_sample(k):
        t_probe = 5e-6 + t_window + k * 50e-6 + 5e-6
        best = min(rows, key=lambda r: abs(r[0] - t_probe))
        return best[1]

    # First sample: pure charge share against the settled reservoir.
    err0 = held_after_sample(0) - v_ideal
    pred0 = d * (v_prev - v_ideal)
    assert math.isclose(err0, pred0, rel_tol=0.05), (
        f"first-sample residual {err0:.3e} vs charge-share {pred0:.3e}"
    )

    # Steady state: theft accumulates against RC recovery. The crude
    # geometric-series estimate overshoots ~20% (it ignores intra-cycle
    # recovery shape), so assert the band rather than the point.
    tail = [held_after_sample(k) - v_ideal for k in range(30, 39)]
    err_ss = sum(tail) / len(tail)
    tau = (params.value("feedback.emf.source_impedance")
           * params.value("circuit.emf_channel.c_filter"))
    amp = math.exp(-50e-6 / tau) / (1.0 - math.exp(-50e-6 / tau))
    pred_ss = pred0 * (1.0 + amp)
    assert 0.5 * pred_ss < err_ss < 1.1 * pred_ss, (
        f"steady-state droop {err_ss:.3e} vs analytic {pred_ss:.3e}"
    )
    # And it is a real, multi-LSB effect - the reason the model exists.
    lsb = params.value("adc.vref") / 4096
    assert abs(err_ss) > 2 * lsb


@needs_ngspice
def test_adc_frontend_bus_transient(params):
    """Unbuffered settling: residual fraction matches the exp formula."""
    rows = spice_runner.run_netlist("adc_frontend_bus",
                                    params)["adc_bus_tran.out"]
    v_ideal = 12.0 * params.value("feedback.bus_voltage.divider_ratio")
    v_prev = 1.65
    t_window = 1.5 / params.value("adc.sclk")
    end = [r for r in rows if r[0] > 1e-6 + t_window + 0.2e-6][0]
    residual = (end[1] - v_ideal) / (v_prev - v_ideal)
    assert math.isclose(residual, params.value("adc.sample_residual_bus"),
                        rel_tol=0.05), (
        f"SPICE residual {residual:.3e} vs derived "
        f"{params.value('adc.sample_residual_bus'):.3e}"
    )


@needs_ngspice
def test_ti_vendor_amp_model(params):
    """Independent cross-check against TI's DRV8301 amp macro (PSpice
    dialect, needs compat mode and a low-impedance network - see netlist
    notes). Gain must match the configured value closely; the model's
    Vos/Ibias produce a ~40 mV output offset, which is exactly what the
    RTL's DC_CAL capture cancels."""
    if not spice_runner.DRV8301_LIB.is_file():
        pytest.skip("TI DRV8301.LIB not present in docs collateral")
    data = spice_runner.run_netlist(
        "drv8301_amp", params,
        aux_files={"DRV8301.LIB": spice_runner.DRV8301_LIB},
        compat="psa")["drv_amp_dc.out"]
    lin = [(vp, vo) for vp, vo in data if abs(vp) < 0.04]
    gain, offset = linear_fit(lin)
    assert math.isclose(gain, params.value("drv8301.amp_gain"),
                        rel_tol=0.02), f"vendor-model gain {gain:.4f}"
    ideal = params.value("feedback.current.offset")
    deviation = abs(offset - ideal)
    assert deviation < 0.08, (
        f"vendor-model output offset {offset:.4f} vs ideal {ideal} - "
        "deviation beyond plausible Vos/Ibias"
    )
    # Record-keeping assertion: the deviation is NOT negligible (> 10 LSB),
    # i.e. DC_CAL is load-bearing.
    lsb = params.value("adc.vref") / 4096
    assert deviation > 10 * lsb
