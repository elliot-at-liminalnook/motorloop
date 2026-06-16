# SPDX-License-Identifier: MIT
"""Cross-platform smoke tier (platform-abstraction stage 5).

Runs a small "does this BOM work at all" scenario set on every registered
platform: init completes, the controller spins the motor (six-step and FOC),
and the shoot-through checker stays clean. With one platform this is a single
pass; it auto-extends to every BOM as `PLATFORMS` grows (Phase B), so the same
RTL gets verified against each component set. Placeholder motor params.
"""

from __future__ import annotations

import statistics

import pytest

from bench_factory import (PLATFORMS, platform, bench_config,
                           expected_init_time, freq_word)


def test_platform_selection_is_live(bldcsim, params):
    """The factory genuinely selects by name: a known set builds, an unknown
    one raises (so a typo fails loudly, never silently to the default)."""
    bldcsim.Bench(bench_config(params))  # defaults
    with pytest.raises(Exception):
        bldcsim.Bench(bench_config(params, driver_name="bogus"))
    with pytest.raises(Exception):
        bldcsim.Bench(bench_config(params, adc_name="bogus"))
    with pytest.raises(KeyError):
        platform(params, "nonexistent_bom")


@pytest.mark.parametrize("name", list(PLATFORMS))
def test_platform_init_and_six_step(bldcsim, params, name):
    """Init reaches RUN and an open-loop spin develops current with zero
    shoot-through on this platform's peripheral set."""
    b = bldcsim.Bench(platform(params, name))
    b.run_for(expected_init_time(params))
    assert b.configured, f"{name}: init did not complete"
    clk = params.value("rtl.clock_frequency")
    b.set_open_loop(freq_word(25.0, clk), 1 << 20)
    b.set_duty(int(0.4 * 625))
    b.set_mode(1)
    b.run_for(0.3)
    assert b.shoot_through_violations == 0
    assert max(abs(i) for i in b.currents) > 0.5, f"{name}: no current developed"


@pytest.mark.parametrize("name", list(PLATFORMS))
def test_platform_foc_spin(bldcsim, params, name):
    """FOC (mode 3) spins the PMSM to target on this platform, id held near 0,
    zero shoot-through."""
    b = bldcsim.Bench(platform(params, name, motor={"trapezoid_blend": 0.0}))
    b.run_for(expected_init_time(params))
    b.set_align_offset(int(params.value("foc.align_offset")))
    b.set_id_target(0)
    b.set_foc_speed_loop(True)
    b.set_target_speed(80)
    b.set_mode(3)
    b.run_for(1.2)
    omega = statistics.mean([b.omega for _ in range(500) if not b.run_for(2e-4)])
    assert b.shoot_through_violations == 0
    assert omega == pytest.approx(80, rel=0.1), f"{name}: settled {omega:.1f}"
