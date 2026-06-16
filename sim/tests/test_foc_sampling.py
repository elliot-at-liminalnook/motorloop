# SPDX-License-Identifier: MIT
"""FOC stage 4: phase-current sampling architecture (Q21).

The RTL FOC sequencer samples ch0 (ia) then ch1 (ib) near the PWM-counter
peak, where every low-side shunt conducts. The two samples' simultaneity is a
bench/board property:

  * scheme 0 (dual ADC / external S/H): both currents are frozen at the peak,
    so the sequencer recovers ia AND ib faithfully;
  * scheme 1 (sequential single ADC): the second conversion lands ~22 us
    later, after leg B's low-side conduction window has closed, so ib is lost.

This is the concrete Q21 finding: a single sequential MCP3208 cannot do FOC
current acquisition. Placeholder motor params - this checks the sampling
architecture, not a hardware number.
"""

from __future__ import annotations

import statistics

import pytest

import foc_reference as fr
from bench_factory import foc, expected_init_time, freq_word


def _lsb_per_amp(params):
    return (params.value("drv8301.amp_gain") * params.value(
        "feedback.current.shunt") * 4096.0 / params.value("adc.vref"))


def _collect(bldcsim, params, scheme, f_elec=20.0, duty=0.4, cycles=4):
    """Open-loop forced commutation with FOC sampling forced on; collect
    plant currents and the FOC-decoded currents over a few electrical cycles."""
    clk = params.value("rtl.clock_frequency")
    lpa = _lsb_per_amp(params)
    b = bldcsim.Bench(foc(params, sample_scheme=scheme))
    b.run_for(expected_init_time(params))
    b.set_open_loop(freq_word(f_elec, clk), 1 << 20)
    b.set_duty(int(duty * 625))
    b.set_mode(1)
    b.set_foc_sample(True)
    b.run_for(0.2)  # let currents establish

    samples = []
    period = 1.0 / f_elec
    n = int(cycles * period / 5e-4)
    for _ in range(n):
        b.run_for(5e-4)
        ia, ib, _ = b.currents
        samples.append((ia, ib, b.foc_cur_a / lpa, b.foc_cur_b / lpa))
    assert b.shoot_through_violations == 0
    return samples


def _mean_abs(samples, idx):
    return statistics.mean(abs(s[idx]) for s in samples)


def test_simultaneous_scheme_recovers_both_phases(bldcsim, params):
    """Scheme 0: the FOC-decoded ia and ib both track the plant currents."""
    s = _collect(bldcsim, params, scheme=0)
    plant_ia, plant_ib = _mean_abs(s, 0), _mean_abs(s, 1)
    meas_ia, meas_ib = _mean_abs(s, 2), _mean_abs(s, 3)
    assert plant_ia > 0.5 and plant_ib > 0.5, "currents too small to test"
    assert meas_ia == pytest.approx(plant_ia, rel=0.2), (
        f"ia: meas {meas_ia:.2f} vs plant {plant_ia:.2f}")
    assert meas_ib == pytest.approx(plant_ib, rel=0.2), (
        f"ib: meas {meas_ib:.2f} vs plant {plant_ib:.2f}")


def test_sequential_scheme_loses_second_phase(bldcsim, params):
    """Scheme 1: ia (first conversion, at the peak) is still recovered, but
    ib (second conversion, ~22 us late) collapses - the conduction window has
    closed. This is why a single sequential MCP3208 cannot do FOC (Q21)."""
    s = _collect(bldcsim, params, scheme=1)
    plant_ia, plant_ib = _mean_abs(s, 0), _mean_abs(s, 1)
    meas_ia, meas_ib = _mean_abs(s, 2), _mean_abs(s, 3)
    # Phase A (sampled at the peak) survives in both schemes.
    assert meas_ia == pytest.approx(plant_ia, rel=0.3), (
        f"ia should still track: meas {meas_ia:.2f} vs plant {plant_ia:.2f}")
    # Phase B is badly degraded: only ~half its magnitude survives (the late
    # conversion sometimes still catches conduction, sometimes the offset) -
    # a ~2x current-feedback error, unusable for FOC.
    assert meas_ib < 0.6 * plant_ib, (
        f"ib not degraded as expected: meas {meas_ib:.2f} vs plant "
        f"{plant_ib:.2f}")


def test_scheme_choice_is_the_q21_finding(bldcsim, params):
    """Side-by-side: scheme 0 recovers ib, scheme 1 does not - the bench
    measurement that resolves Q21 toward simultaneous sampling."""
    s0 = _collect(bldcsim, params, scheme=0)
    s1 = _collect(bldcsim, params, scheme=1)
    ib0 = _mean_abs(s0, 3)
    ib1 = _mean_abs(s1, 3)
    assert ib0 > 1.8 * ib1, (
        f"scheme 0 should recover much more ib than scheme 1: "
        f"{ib0:.2f} vs {ib1:.2f}")


def _dq_measurement_error(bldcsim, params, scheme, f_elec=20.0, duty=0.45):
    """Mean magnitude of the dq-frame error between the FOC-sampled currents
    and the plant truth, both projected to the rotor frame at the plant angle
    (amps). This is the torque-command error a scheme injects into the loop
    (4.3): scheme 0 carries only the peak-vs-read sampling ripple; scheme 1
    adds the lost-phase error."""
    clk = params.value("rtl.clock_frequency")
    pp = int(params.value("motor.pole_pairs"))
    lpa = _lsb_per_amp(params)
    b = bldcsim.Bench(foc(params, sample_scheme=scheme))
    b.run_for(expected_init_time(params))
    b.set_open_loop(freq_word(f_elec, clk), 1 << 20)
    b.set_duty(int(duty * 625))
    b.set_mode(1)
    b.set_foc_sample(True)
    b.run_for(0.25)
    errs = []
    for _ in range(300):
        b.run_for(2e-4)
        theta_e = pp * b.theta
        ia, ib, _ = b.currents
        tid, tiq = fr.park(*fr.clarke(ia, ib), theta_e)
        mid, miq = fr.park(*fr.clarke(b.foc_cur_a / lpa, b.foc_cur_b / lpa),
                           theta_e)
        errs.append(((mid - tid) ** 2 + (miq - tiq) ** 2) ** 0.5)
    return statistics.mean(errs)


def test_sampling_error_into_torque_command(bldcsim, params):
    """4.3: the dq measurement error (the torque-command error a scheme would
    inject) is far larger for the sequential scheme than the simultaneous one
    - the quantitative case for dual/simultaneous sampling."""
    err0 = _dq_measurement_error(bldcsim, params, scheme=0)
    err1 = _dq_measurement_error(bldcsim, params, scheme=1)
    assert err1 > 3.0 * err0, (
        f"sequential scheme should inject far more dq error: "
        f"{err1:.2f} A vs {err0:.2f} A")
