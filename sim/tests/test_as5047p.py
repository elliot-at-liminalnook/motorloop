# SPDX-License-Identifier: MIT
"""AS5047P SPI angle sensor (platform-abstraction Phase C, stage 10).

The AS5047P is read over SPI and applies DAEC (Dynamic Angle Error
Compensation): the read-pipeline latency is predicted forward, so the angle the
controller sees tracks the true rotor far more tightly than the high-latency
AS5600 PWM sensor. That is the hardware form of the RTL's omega*t_latency
extrapolation - the resolution of open-question Q22 (angle latency costs torque
at speed). These tests quantify it at the integration level (the 12-bit angle
the controller actually acts on) and confirm the magnet-loss path drops
angle_valid.
"""

from __future__ import annotations

import math
import statistics

from bench_factory import platform, expected_init_time


def _mean_angle_lag_lsb(b, omega_mech, settle_s=0.06, samples=400):
    """Mean signed lag (12-bit LSB) of the controller-visible angle behind the
    true rotor, with the rotor mechanically clamped to omega_mech."""
    b.set_speed_clamp(True, omega_mech)
    b.run_for(settle_s)
    lags = []
    for _ in range(samples):
        b.run_for(2e-4)
        true12 = int((b.theta / (2.0 * math.pi)) * 4096) & 0xFFF
        d = (true12 - b.angle) & 0xFFF
        if d > 2048:
            d -= 4096
        lags.append(d)
    return statistics.mean(lags)


def test_as5047p_daec_tracks_tighter_than_as5600(bldcsim, params):
    """At a fixed rotor speed the AS5047P (DAEC) angle lags the true rotor far
    less than the AS5600 (PWM + slow filter) - the Q22 quantity, in hardware."""
    omega = 60.0

    b47 = bldcsim.Bench(platform(params, "zonri_as5047p"))
    b47.run_for(expected_init_time(params))
    assert b47.angle_valid, "AS5047P never produced a valid angle"
    lag47 = abs(_mean_angle_lag_lsb(b47, omega))

    b56 = bldcsim.Bench(platform(params, "zonri_drv8301"))  # AS5600 PWM
    b56.run_for(expected_init_time(params))
    lag56 = abs(_mean_angle_lag_lsb(b56, omega))

    # DAEC residual is near zero LSB; the AS5600 lags many LSB.
    assert lag47 < lag56, f"AS5047P lag {lag47:.1f} !< AS5600 lag {lag56:.1f}"
    assert lag47 < 5.0, f"AS5047P DAEC lag too large: {lag47:.1f} LSB"
    assert lag56 > 8.0, f"AS5600 lag unexpectedly small: {lag56:.1f} LSB"


def test_as5047p_magnet_loss_drops_valid(bldcsim, params):
    """Magnet loss sets the EF bit; the SPI master's parity/EF check drops
    angle_valid (the realism-scenario portable behaviour)."""
    b = bldcsim.Bench(platform(params, "zonri_as5047p"))
    b.run_for(expected_init_time(params))
    assert b.angle_valid
    b.inject_magnet_loss(True)
    b.run_for(1e-3)
    assert not b.angle_valid, "angle_valid should drop on magnet loss"
    b.inject_magnet_loss(False)
    b.run_for(1e-3)
    assert b.angle_valid, "angle_valid should recover after magnet returns"
