# SPDX-License-Identifier: MIT
"""DRV8323RS external-FET smart gate driver (platform-abstraction Phase C,
stage 11).

The DRV8323 shares the DRV8301's 16-bit SPI frame, so the controller configures
it with the existing write/verify handshake (no RTL change) and the part runs on
its datasheet power-on defaults. These tests confirm the handshake completes
(reaches configured / RUN) and that a latched driver fault is detected and
counted by the manager.
"""

from __future__ import annotations

from bench_factory import platform, expected_init_time, freq_word


def test_drv8323rs_configures_and_spins(bldcsim, params):
    """The DRV8323 register echo completes the controller's write/verify
    handshake (reaches configured) and an open-loop spin develops current."""
    b = bldcsim.Bench(platform(params, "zonri_drv8323rs"))
    b.run_for(expected_init_time(params))
    assert b.configured, "DRV8323RS never reached configured (handshake failed)"
    clk = params.value("rtl.clock_frequency")
    b.set_open_loop(freq_word(25.0, clk), 1 << 20)
    b.set_duty(int(0.4 * 625))
    b.set_mode(1)
    b.run_for(0.3)
    assert b.shoot_through_violations == 0
    assert max(abs(i) for i in b.currents) > 0.5


def test_drv8323rs_latched_fault_detected(bldcsim, params):
    """A latched driver fault (nFAULT low) during RUN is seen by the manager:
    it drops configured and counts the fault."""
    b = bldcsim.Bench(platform(params, "zonri_drv8323rs"))
    b.run_for(expected_init_time(params))
    assert b.configured
    assert b.fault_count == 0
    b.inject_drv_latched_fault()
    b.run_for(2e-3)
    assert b.fault_count > 0, "latched fault not counted"
