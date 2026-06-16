# SPDX-License-Identifier: MIT
"""Stage 6: closed-loop system scenarios on the lockstep bench.

S0 init sequencing, S1 open-loop spin, S2 closed-loop sensored six-step,
S3 ADC aperture schedule, S4 transients, S5 fault injection, plus the
performance guard. The shoot-through checker runs in every scenario via the
`finished` helper.
"""

from __future__ import annotations

import math

import pytest
from bench_factory import bench_config, expected_init_time, freq_word


@pytest.fixture
def bench(params, bldcsim):
    b = bldcsim.Bench(bench_config(params))
    return b


@pytest.fixture
def align_offset(params):
    return int(params.value("rtl.sensored_align_offset"))


def init_bench(b, params):
    b.run_for(expected_init_time(params))
    assert b.configured, "DRV manager failed to reach RUN"
    return b


def finished(b, params):
    """Global invariants asserted at the end of every scenario."""
    assert b.shoot_through_violations == 0
    assert b.plant_shoot_through_requests == 0
    if b.min_dead_time_s < 1e6:  # any complementary switching observed
        assert b.min_dead_time_s >= params.value("pwm.rtl_dead_time") * 0.99


# ---------------------------------------------------------------------------
# S0 - init sequencing
# ---------------------------------------------------------------------------

def test_s0_init_sequencing(params, bldcsim):
    b = bldcsim.Bench(bench_config(params))

    # Before the EN_GATE ready window, not configured and no gate activity.
    b.run_for(5e-3)
    assert not b.configured
    assert all(abs(i) < 1e-9 for i in b.currents)

    b.run_for(expected_init_time(params))
    assert b.configured

    # SPI config landed in the model's register file with the intended
    # values from params.toml.
    oc_code = int(params.value("rtl.oc_adj_code"))
    gain_code = int(params.value("rtl.amp_gain_code"))
    assert b.drv_reg(2) == oc_code << 6
    assert b.drv_reg(3) == gain_code << 2
    assert b.mismatch_count == 0
    assert b.fault_count == 0

    # DC_CAL offsets captured at the amp midpoint (1.65 V -> code 2048).
    assert abs(b.offset_a - 2048) <= 1
    assert abs(b.offset_b - 2048) <= 1

    # Motor untouched through all of init.
    assert all(abs(i) < 1e-9 for i in b.currents)
    assert abs(b.omega) < 1e-9
    finished(b, params)


# ---------------------------------------------------------------------------
# S1 - open-loop spin
# ---------------------------------------------------------------------------

def test_s1_open_loop_spin(params, bench):
    b = init_bench(bench, params)
    clk = params.value("rtl.clock_frequency")

    f_elec = 30.0
    word = freq_word(f_elec, clk)
    # ramp_increment 1 per 4096 clk -> ~0.85 s ramp to 30 Hz (gentle enough
    # for the placeholder motor's inertia to stay in sync open-loop).
    b.set_open_loop(word, 1)
    b.set_duty(int(0.5 * 625))
    b.set_mode(1)
    b.run_for(1.2)

    pp = int(params.value("motor.pole_pairs"))
    omega_sync = 2 * math.pi * f_elec / pp
    assert b.omega > 0.7 * omega_sync, f"failed to spin up: {b.omega}"

    # Commutation actually cycling through all six sectors.
    trace = b.trace
    assert set(trace["sector"]) >= {0, 1, 2, 3, 4, 5}
    # Currents bounded sanely.
    peak = max(max(abs(v) for v in trace[c]) for c in ("ia_a", "ib_a", "ic_a"))
    assert peak < 15.0, f"implausible current {peak} A"
    assert b.fault_count == 0
    finished(b, params)


# ---------------------------------------------------------------------------
# S2 - closed-loop sensored six-step
# ---------------------------------------------------------------------------

def test_s2_closed_loop_speed(params, bench, align_offset):
    b = init_bench(bench, params)
    target = 80
    b.set_align_offset(align_offset)
    b.set_target_speed(target)
    b.set_mode(2)
    b.run_for(1.2)

    trace = b.trace
    tail = [w for t, w in zip(trace["time_s"], trace["omega_rad_s"])
            if t > b.time_s - 0.3]
    mean_tail = sum(tail) / len(tail)
    # Tolerance covers AS5600 frame-quantized speed measurement (~5%) plus
    # commutation-lag droop at speed.
    assert abs(mean_tail - target) < 0.15 * target, (
        f"settled at {mean_tail:.1f}, target {target}"
    )
    spread = max(tail) - min(tail)
    assert spread < 0.3 * target, f"speed not settled, spread {spread:.1f}"
    assert b.fault_count == 0
    assert b.mismatch_count == 0
    finished(b, params)


# ---------------------------------------------------------------------------
# S3 - ADC schedule: aperture in the off-window across the duty sweep
# ---------------------------------------------------------------------------

def test_s3_adc_aperture_across_duty_sweep(params, bench):
    b = init_bench(bench, params)
    clk = params.value("rtl.clock_frequency")
    half = int(round(clk / (2 * params.value("pwm.frequency"))))

    word = freq_word(20.0, clk)
    b.set_open_loop(word, max(1, word // 4000))
    b.set_mode(1)

    for duty_frac in (0.10, 0.30, 0.50, 0.70, 0.85, 0.95):
        duty = int(duty_frac * half)
        b.set_duty(duty)
        b.clear_trace()
        b.run_for(0.03)
        emf = [s for s in b.adc_samples if 3 <= s["channel"] <= 5]
        cur = [s for s in b.adc_samples if 0 <= s["channel"] <= 2]
        assert len(emf) > 100, f"duty {duty_frac}: too few EMF samples"
        assert len(cur) > 100, f"duty {duty_frac}: too few current samples"
        for s in emf:
            # Hold instant must land in the PWM off-window (counter beyond
            # the duty compare, where the chopping leg's high FET is off).
            assert s["pwm_counter"] > s["duty"], (
                f"duty {duty_frac}: EMF aperture at counter "
                f"{s['pwm_counter']} inside on-window (duty {s['duty']})"
            )
    finished(b, params)


def test_s3_sector_aware_channel_selection(params, bench):
    b = init_bench(bench, params)
    clk = params.value("rtl.clock_frequency")
    # Slow open loop so sectors are long-lived; check channel selection
    # matches the six-step table: low phase carries current, third floats.
    low_for_sector = {0: 1, 1: 2, 2: 2, 3: 0, 4: 0, 5: 1}
    float_for_sector = {0: 2, 1: 1, 2: 0, 3: 2, 4: 1, 5: 0}

    word = freq_word(5.0, clk)
    b.set_open_loop(word, word)  # no ramp: straight to 5 Hz
    b.set_duty(int(0.3 * 625))
    b.set_mode(1)
    b.run_for(0.05)

    trace = b.trace
    samples = b.adc_samples
    # Map each sample to the sector active at its time via the trace.
    times = trace["time_s"]
    sectors = trace["sector"]

    def sector_at(t):
        lo, hi = 0, len(times) - 1
        while lo < hi:
            mid = (lo + hi + 1) // 2
            if times[mid] <= t:
                lo = mid
            else:
                hi = mid - 1
        return sectors[lo]

    checked = 0
    for s in samples:
        sec = sector_at(s["time_s"])
        # Skip samples near sector boundaries (launch/hold race).
        if s["channel"] <= 2:
            expect = low_for_sector[sec]
            if s["channel"] == expect:
                checked += 1
            else:
                # Allow boundary races but they must be rare.
                pass
        elif 3 <= s["channel"] <= 5:
            expect = float_for_sector[sec] + 3
            if s["channel"] == expect:
                checked += 1
    assert checked > 0.9 * len([s for s in samples if s["channel"] <= 5])
    finished(b, params)


# ---------------------------------------------------------------------------
# S4 - transients
# ---------------------------------------------------------------------------

def test_s4_speed_step_and_load_step(params, bench, align_offset):
    b = init_bench(bench, params)
    b.set_align_offset(align_offset)
    b.set_target_speed(60)
    b.set_mode(2)
    b.run_for(0.8)
    w1 = b.omega

    # Speed step up.
    b.set_target_speed(90)
    b.run_for(0.8)
    w2 = b.omega
    assert w2 > w1 + 15, f"speed step did not track: {w1} -> {w2}"

    # Load-torque step: droop allowed, must keep spinning, no faults.
    b.set_load_torque(0.02)
    b.run_for(0.6)
    assert b.omega > 30, f"stalled under load: {b.omega}"
    assert b.fault_count == 0
    finished(b, params)


# ---------------------------------------------------------------------------
# S5 - fault injection
# ---------------------------------------------------------------------------

def test_s5_register_reset_watchdog(params, bench, align_offset):
    """slva552: silent register reset mid-run; the refresh watchdog must
    detect the mismatch and rewrite the configuration."""
    b = init_bench(bench, params)
    b.set_align_offset(align_offset)
    b.set_target_speed(60)
    b.set_mode(2)
    b.run_for(0.4)
    assert b.mismatch_count == 0

    b.inject_drv_register_reset()
    default_cr1 = int(params.value("drv8301.oc_adj_set")) << 6
    assert b.drv_reg(2) == default_cr1  # reset really happened

    # Within one refresh period (+ margin) the watchdog rewrites.
    b.run_for(2.5 * params.value("rtl.drv_refresh_period") + 5e-3)
    assert b.mismatch_count >= 1, "watchdog missed the register reset"
    assert b.drv_reg(2) == int(params.value("rtl.oc_adj_code")) << 6
    assert b.configured
    b.run_for(0.2)
    assert b.omega > 20, "motor did not keep running after rewrite"
    finished(b, params)


def test_s5_latched_fault_recovery(params, bench, align_offset):
    """nFAULT low -> RTL kills gates, quick-reset pulse, reconfigure,
    resume."""
    b = init_bench(bench, params)
    b.set_align_offset(align_offset)
    b.set_target_speed(60)
    b.set_mode(2)
    b.run_for(0.4)
    omega_before = b.omega
    assert omega_before > 30

    b.inject_drv_latched_fault()
    b.run_for(20e-3)
    assert b.fault_count >= 1, "RTL did not register the fault"

    b.run_for(0.5)
    assert b.configured, "did not reconfigure after fault"
    assert b.omega > 20, f"did not resume after fault: {b.omega}"
    finished(b, params)


def test_s5_otw_reported_not_fatal(params, bench, align_offset):
    b = init_bench(bench, params)
    b.set_align_offset(align_offset)
    b.set_target_speed(60)
    b.set_mode(2)
    b.run_for(0.3)
    assert b.noctw_count == 0
    b.inject_drv_otw(True)
    b.run_for(5e-3)
    b.inject_drv_otw(False)
    assert b.noctw_count >= 1, "OTW warning not counted by RTL"
    assert b.fault_count == 0, "OTW must not trigger the fault path"
    b.run_for(0.2)
    assert b.omega > 20
    finished(b, params)


def test_s5_magnet_loss_safe_stop_and_recovery(params, bench, align_offset):
    b = init_bench(bench, params)
    b.set_align_offset(align_offset)
    b.set_target_speed(60)
    b.set_mode(2)
    b.run_for(0.5)
    assert b.omega > 30

    b.inject_magnet_loss(True)
    b.run_for(0.15)
    assert not b.angle_valid, "capture should invalidate on magnet loss"
    # Gates off: currents collapse, motor coasts down on friction.
    assert all(abs(i) < 0.5 for i in b.currents)

    b.inject_magnet_loss(False)
    b.run_for(0.6)
    assert b.angle_valid
    assert b.omega > 20, f"did not restart after magnet recovery: {b.omega}"
    finished(b, params)


def test_s5_adc_noise_robustness(params, bldcsim, align_offset):
    """Closed loop with seeded gaussian noise on every analog channel
    (~6 LSB rms). Speed control must still settle; determinism comes from
    the fixed seed."""
    cfg = bench_config(params, chain={"noise_rms_v": 5e-3,
                                      "noise_seed": 1234})
    b = bldcsim.Bench(cfg)
    b.run_for(expected_init_time(params))
    assert b.configured
    b.set_align_offset(align_offset)
    b.set_target_speed(80)
    b.set_mode(2)
    b.run_for(1.2)

    trace = b.trace
    tail = [w for t, w in zip(trace["time_s"], trace["omega_rad_s"])
            if t > b.time_s - 0.3]
    mean_tail = sum(tail) / len(tail)
    assert abs(mean_tail - 80) < 0.15 * 80, f"noisy settle at {mean_tail:.1f}"
    assert b.fault_count == 0
    finished(b, params)


# ---------------------------------------------------------------------------
# VCD dump toggle (stage 4.1)
# ---------------------------------------------------------------------------

def test_vcd_dump(params, bench, tmp_path):
    b = bench
    vcd_path = tmp_path / "smoke.vcd"
    b.open_vcd(str(vcd_path))
    b.run_for(2e-3)
    b.close_vcd()
    assert vcd_path.is_file()
    assert vcd_path.stat().st_size > 10_000, "VCD suspiciously small"
    header = vcd_path.read_text(errors="ignore")[:2000]
    assert "$var" in header and "clk" in header


# ---------------------------------------------------------------------------
# Performance guard (stage 4.2 / 6.8)
# ---------------------------------------------------------------------------

def test_performance_budget(params, bench, align_offset):
    import time

    b = init_bench(bench, params)
    b.set_align_offset(align_offset)
    b.set_target_speed(80)
    b.set_mode(2)
    start = time.monotonic()
    b.run_for(0.2)
    wall = time.monotonic() - start
    # Budget: >= 1 simulated second per minute of wall clock (checklist 4.2);
    # currently ~5 s/sim-s, so this guards against order-of-magnitude
    # regressions without being flaky on slow machines.
    assert wall < 12.0, f"0.2 s of simulation took {wall:.1f} s"
