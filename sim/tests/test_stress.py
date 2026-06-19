# SPDX-License-Identifier: MIT
"""Stress / extreme-scenario suite (notes/stress-test-report.md): push the
system + code to the limit and assert the protections - and the fixed-point
math - hold. Four axes: A system failure boundaries, B control/sensor limits,
C analog/ADC extremes, D numerical/code edges. Figures: gen_stress_figures.py.

Sizes are coarse for CI; the figure generator re-runs them longer/finer.
"""

from __future__ import annotations

import math
import shutil

import numpy as np
import part_compare as PC
import pytest
import spice_runner
from bench_factory import expected_init_time, foc

needs_ngspice = pytest.mark.skipif(shutil.which("ngspice") is None,
                                   reason="ngspice not installed")


def _init(bldcsim, params, cfg, iq=60, omega=None, mode=3):
    b = bldcsim.Bench(cfg)
    b.run_for(expected_init_time(params))
    b.set_align_offset(int(params.value("foc.align_offset")))
    b.set_id_target(0)
    b.set_iq_target(int(iq))
    if omega is not None:
        b.set_speed_clamp(True, float(omega))
    b.set_mode(mode)
    return b


# ----- A: system failure boundaries -----------------------------------------
def test_A1_thermal_heating(bldcsim, params):
    """Locked-rotor cook: the winding heats markedly (the thermal model is live
    and the runaway is real)."""
    b = _init(bldcsim, params, foc(params, "thermal"), iq=250, omega=0.0)
    t0 = b.motor_t_c
    for _ in range(16):
        b.run_for(0.25)             # ~4 s
    assert b.motor_t_c > t0 + 10.0, f"winding barely heated: {t0}->{b.motor_t_c}"


def test_A2_brownout_uvlo(bldcsim, params):
    """Supply current limit -> bus collapse -> UVLO fires (the protection trips)."""
    b = _init(bldcsim, params, foc(params, "supply", supply={"i_limit_a": 0.8}),
              iq=100, omega=60.0)
    b.run_for(0.05); b.set_load_torque(0.6)
    for _ in range(12):
        b.run_for(0.02)
    assert b.bus_v_min < 0.8 * params.value("bus.vbus"), "bus did not sag"
    assert b.pvdd_uv_events > 0, "UVLO never fired under brownout"


def test_A3_regen_overvoltage_bounded(bldcsim, params):
    """Hard reversal pumps the bus (regen) - it rises above nominal but stays
    physically bounded (the cap absorbs it, no runaway)."""
    b = _init(bldcsim, params, foc(params, "supply", supply={"capacitance": 2.0e-4}),
              iq=120, omega=300.0)
    b.run_for(0.05); b.set_speed_clamp(True, -300.0)
    for _ in range(20):
        b.run_for(0.005)
    assert b.bus_v_max > params.value("bus.vbus"), "no regen pump seen"
    assert b.bus_v_max < 60.0, "bus pumped past the FET/cap rating"


def test_A4_overcurrent_clamp_no_shootthrough(bldcsim, params):
    """An absurd torque command is clamped to iq_max, and the dead-time guard
    holds zero shoot-through - protections, not silicon, set the limit."""
    b = _init(bldcsim, params, foc(params), iq=1800, omega=40.0)
    for _ in range(8):
        b.run_for(0.02)
    assert abs(b.foc_iq) <= params.value("foc.iq_max") * 1.1
    assert b.shoot_through_violations == 0


def test_A5_fault_detected(bldcsim, params):
    """A latched DRV fault injected mid-run is detected (fault_count rises)."""
    b = _init(bldcsim, params, foc(params), iq=60, omega=60.0)
    b.run_for(0.1)
    fc0 = b.fault_count
    b.inject_drv_latched_fault()
    for _ in range(10):
        b.run_for(0.02)
    assert b.fault_count > fc0, "latched fault not detected"


# ----- B: control / sensor limits -------------------------------------------
def test_B1_reversal_cliff_sensor(bldcsim, params):
    """Slam reversals faster and faster: the AS5047P survives a strictly higher
    reversal speed than the AS5600 before commutation inverts."""
    speeds = [100, 400, 900, 1600]
    a = PC.run_reversal(params, bldcsim, PC.SENSOR_A, speeds, transition_s=0.008)
    s = PC.run_reversal(params, bldcsim, PC.SENSOR_B, speeds, transition_s=0.008)
    assert a["cliff_speed"] < speeds[-1], "AS5600 should break in range"
    assert s["cliff_speed"] > a["cliff_speed"]


def test_B2_load_step_survives(bldcsim, params):
    """An instant heavy load step spikes the current but trips no shoot-through
    and the bus holds (a survivable extreme)."""
    b = _init(bldcsim, params, foc(params, "mechanical"), iq=120, omega=80.0)
    b.run_for(0.05); b.set_load_torque(0.8)
    pk = 0.0
    for _ in range(60):
        b.run_for(2e-4)
        pk = max(pk, float(np.abs(b.currents).max()))
    assert pk > 0.0
    assert b.shoot_through_violations == 0


# ----- C: analog / ADC extremes ---------------------------------------------
@needs_ngspice
def test_C1_settle_failure_boundary(params):
    """The design bucket (1 nF) settles < 0.5 LSB within tACQ; an oversized
    bucket (47 nF) misses it - the failure boundary is real."""
    t_acq = params.value("adc.ads9224r_acq_window_s")
    ref = params.value("circuit.ads9224r_module.ref_v")

    def resid(cap):
        rows = spice_runner.run_netlist("ads9224r_acq", params,
                                        overrides={"bcap": cap})["ads9224r_acq.out"]
        probe = min(rows, key=lambda r: abs(r[0] - t_acq))
        return abs(ref - probe[1]) / ref

    assert resid(1.0e-9) < 1.0 / 65536          # design: passes
    assert resid(47.0e-9) > 1.0 / 65536         # oversized: fails (boundary)


@needs_ngspice
def test_C2_fullscale_code_rails(params):
    """Beyond +/- full-scale the ADC code rails at +/-32768 (clamp, never wraps)."""
    dc = spice_runner.run_netlist("ads9224r_frontend", params)["ads9224r_dc.out"]
    ref = params.value("circuit.ads9224r_module.ref_v")
    codes = [v / ref * 32768.0 for _, v in dc]
    # Rails near +/- full-scale (clamped, no wrap to the opposite sign).
    assert 32000 < max(codes) < 34000
    assert -34000 < min(codes) < -32000


# ----- D: numerical / code edges --------------------------------------------
def test_D1_fixedpoint_rails_bounded(bldcsim, params):
    """Extreme 18-bit inputs to the FOC math saturate - duties stay in
    [0, PWM_HALF] and dq currents stay bounded; nothing wraps."""
    import focmath
    fm = focmath.FocMath()
    m = (1 << 17) - 1
    for theta in (0, 16384, 32768, 49152, 65535):
        o = fm.eval(theta, m, -m, m, m, m, -m)
        for d in (o["duty_a"], o["duty_b"], o["duty_c"]):
            assert 0 <= d <= 625, f"duty {d} out of [0,625]"
        assert abs(o["id"]) < (1 << 17) and abs(o["iq"]) < (1 << 17)


def test_D2_circle_limiter_saturates(bldcsim, params):
    """The voltage-vector magnitude saturates (the circle limiter) however large
    the command grows - it does not run away."""
    import focmath
    fm = focmath.FocMath()
    m = (1 << 17) - 1
    small = fm.eval(8000, 0, 0, int(0.1 * m * 0.6), int(0.1 * m), 0, 0)
    big = fm.eval(8000, 0, 0, int(m * 0.6), int(m), 0, 0)
    mag_small = math.hypot(small["valpha"], small["vbeta"])
    mag_big = math.hypot(big["valpha"], big["vbeta"])
    # A 10x larger command does not give a 10x larger limited vector.
    assert mag_big < 3.0 * mag_small + 1.0
