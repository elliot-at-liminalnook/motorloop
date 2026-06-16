# SPDX-License-Identifier: MIT
"""ADS9224R dual-simultaneous 16-bit current ADC (platform-abstraction Phase C,
stage 13).

The ADS9224R samples both phase currents on one CONVST edge - truly
simultaneous, retiring Q21 (the MCP3208's sequential inter-channel skew) in
hardware. Simultaneous sampling keeps the measured current vector aligned, so
the d-axis current stays cleanly regulated near zero even at speed (a sequential
skew would rotate the measured vector and bleed into id). These tests confirm
FOC closes the loop on the dual ADC and tracks a torque current under load.
"""

from __future__ import annotations

import statistics

from bench_factory import platform, expected_init_time


def _spin_foc(b, params, target):
    b.run_for(expected_init_time(params))
    b.set_align_offset(int(params.value("foc.align_offset")))
    b.set_id_target(0)
    b.set_foc_speed_loop(True)
    b.set_target_speed(target)
    b.set_mode(3)
    b.run_for(1.2)


def test_ads9224r_simultaneous_keeps_id_clean(bldcsim, params):
    """FOC on the ADS9224R spins to target with the d-axis current regulated
    near zero - the dual-simultaneous sample keeps the dq frame aligned."""
    b = bldcsim.Bench(platform(params, "zonri_ads9224r",
                               motor={"trapezoid_blend": 0.0}))
    _spin_foc(b, params, target=80)
    samples = [(b.omega, b.foc_id) for _ in range(500) if not b.run_for(2e-4)]
    omega = statistics.mean(s[0] for s in samples)
    id_mean = statistics.mean(s[1] for s in samples)
    assert b.shoot_through_violations == 0
    assert abs(omega - 80) < 8.0, f"ADS9224R FOC settled {omega:.1f}"
    # id regulated near zero (the FOC d-axis command is 0); the loop's measured
    # id should sit close to it.
    assert abs(id_mean) < 40, f"id not regulated near 0: {id_mean}"


def test_ads9224r_measures_torque_current(bldcsim, params):
    """Under load the dual ADC reports a meaningful positive q-axis current."""
    b = bldcsim.Bench(platform(params, "zonri_ads9224r",
                               motor={"trapezoid_blend": 0.0}))
    _spin_foc(b, params, target=80)
    b.set_load_torque(0.02)
    b.run_for(0.3)
    iq = statistics.mean([b.foc_iq for _ in range(400) if not b.run_for(2e-4)])
    assert iq > 5, f"dual-ADC torque current implausibly low: {iq}"
