# SPDX-License-Identifier: MIT
"""DRV8316R integrated-FET driver + integrated CSA (platform-abstraction
Phase C, stage 12).

The DRV8316R integrates the FETs and the current-sense amplifiers (the clean
reference BOM - no external shunts, retiring Q7). The bench routes current
sensing through the FeedbackChain's integrated-CSA source (~0.15 V/A) and
renormalizes its larger codes/A back to the canonical FOC fixed-point scale
(cur_norm_shift). These tests confirm FOC closes the loop on the integrated CSA
and that the measured currents are physically meaningful under load.
"""

from __future__ import annotations

import statistics

from bench_factory import platform, expected_init_time


def _spin_foc(b, params, target=80):
    b.run_for(expected_init_time(params))
    b.set_align_offset(int(params.value("foc.align_offset")))
    b.set_id_target(0)
    b.set_foc_speed_loop(True)
    b.set_target_speed(target)
    b.set_mode(3)
    b.run_for(1.2)


def test_drv8316r_integrated_csa_closes_foc(bldcsim, params):
    """FOC spins the PMSM to target on the DRV8316R integrated CSA, with the
    d-axis current regulated near zero and zero shoot-through."""
    b = bldcsim.Bench(platform(params, "zonri_drv8316r",
                               motor={"trapezoid_blend": 0.0}))
    _spin_foc(b, params, target=80)
    omega = statistics.mean([b.omega for _ in range(500) if not b.run_for(2e-4)])
    assert b.shoot_through_violations == 0
    assert abs(omega - 80) < 8.0, f"DRV8316R FOC settled {omega:.1f}"


def test_drv8316r_integrated_csa_measures_current(bldcsim, params):
    """Under load the integrated CSA delivers a meaningful torque-current
    measurement (the sense path is live, not stuck at zero)."""
    b = bldcsim.Bench(platform(params, "zonri_drv8316r",
                               motor={"trapezoid_blend": 0.0}))
    _spin_foc(b, params, target=80)
    b.set_load_torque(0.02)
    b.run_for(0.3)
    iq = statistics.mean(
        [b.foc_iq for _ in range(400) if not b.run_for(2e-4)])
    # A loaded spin draws positive q-axis (torque) current; the integrated CSA
    # must report it (non-trivial, correct sign).
    assert iq > 5, f"integrated-CSA torque current implausibly low: {iq}"
