# SPDX-License-Identifier: MIT
"""Edge-case scenario coverage (notes/edge-case-scenarios.md, E1-E27).

Each test names the catalog entry it covers. E25 (counter saturation) is
covered jointly by the saturating RTL implementations plus the
lockout/dead-state tests that bound how far the counters can run.
"""

from __future__ import annotations

import math

import pytest
from bench_factory import bench_config, expected_init_time, freq_word, realism


@pytest.fixture
def align_offset(params):
    return int(params.value("rtl.sensored_align_offset"))


def init_bench(b, params):
    b.run_for(expected_init_time(params))
    assert b.configured
    return b


def finished(b):
    assert b.shoot_through_violations == 0
    assert b.plant_shoot_through_requests == 0


def sector_steps(trace):
    """Set of sector step sizes (mod 6) observed in a trace."""
    steps = set()
    prev = None
    for s in trace["sector"]:
        if prev is not None and s != prev:
            steps.add((int(s) - int(prev)) % 6)
        prev = s
    return steps


# ---------------------------------------------------------------------------
# A. Timing and boundary edges
# ---------------------------------------------------------------------------

def test_e1_sector_boundary_chatter(params, bldcsim):
    """Rotor parked with the sensed angle ON a sector boundary plus angle
    noise: the hysteresis must keep the registered sector still."""
    b = bldcsim.Bench(bench_config(
        params, encoder={"angle_noise_lsb": 2.0, "noise_seed": 5}))
    init_bench(b, params)
    # theta = 0 reads angle 0; align chosen so elec12 sits exactly on the
    # sector-0/1 boundary (683).
    b.set_align_offset(683)
    b.set_target_speed(0)
    b.set_mode(2)
    b.clear_trace()
    b.run_for(0.3)
    changes = sum(1 for a, c in zip(b.trace["sector"], b.trace["sector"][1:])
                  if a != c)
    assert changes == 0, f"sector chattered {changes} times at the boundary"
    finished(b)


def test_e2_angle_wraparound_clean_sectors(params, bldcsim, align_offset):
    """Spinning through the 4095->0 wrap repeatedly must only ever produce
    adjacent sector steps."""
    b = bldcsim.Bench(bench_config(params))
    init_bench(b, params)
    b.set_align_offset(align_offset)
    b.set_target_speed(50)
    b.set_mode(2)
    b.run_for(0.6)
    b.clear_trace()
    b.run_for(0.6)  # several mechanical revs through the wrap
    steps = sector_steps(b.trace)
    assert steps <= {1, 5}, f"non-adjacent sector steps observed: {steps}"
    finished(b)


def test_e3_duty_extremes(params, bldcsim):
    """Duty 0 / sliver / full: min-pulse snap, no shoot-through anywhere."""
    half = 625
    b = bldcsim.Bench(bench_config(params))
    init_bench(b, params)
    clk = params.value("rtl.clock_frequency")
    b.set_open_loop(freq_word(5.0, clk), 2**31)
    b.set_mode(1)
    # Sliver duty FIRST, while the rotor has never moved: the min-pulse
    # snap means no drive current at all. (Checked before the other duties
    # because afterwards a spinning rotor legitimately drives braking
    # current through the two ON low-side FETs at duty ~ 0.)
    b.set_duty(1)
    b.run_for(0.05)
    assert all(abs(i) < 0.05 for i in b.currents), (
        f"sub-min-pulse duty produced current: {b.currents}"
    )
    for duty in (0, 2, half // 2, half - 2, half):
        b.set_duty(duty)
        b.run_for(0.02)
        finished(b)
    finished(b)


def test_e4_emf_sample_skipped_at_high_duty(params, bldcsim):
    """Off-window narrower than the aperture: the EMF conversion must be
    skipped, never mis-sampled."""
    half = 625
    margin = int(params.value("rtl.emf_skip_duty_margin"))
    b = bldcsim.Bench(bench_config(params))
    init_bench(b, params)
    clk = params.value("rtl.clock_frequency")
    b.set_open_loop(freq_word(10.0, clk), 2**31)
    b.set_mode(1)
    b.set_duty(half - margin + 4)  # above the skip threshold
    b.run_for(0.02)
    b.clear_trace()
    b.run_for(0.06)
    emf = [s for s in b.adc_samples if 3 <= s["channel"] <= 5]
    cur = [s for s in b.adc_samples if 0 <= s["channel"] <= 2]
    assert not emf, f"EMF sampled despite narrow off-window: {len(emf)}"
    assert len(cur) > 100  # the current slot keeps running

    # Below the threshold the EMF slot resumes and stays in the off-window.
    b.set_duty(half - margin - 8)
    b.run_for(0.02)
    b.clear_trace()
    b.run_for(0.06)
    emf = [s for s in b.adc_samples if 3 <= s["channel"] <= 5]
    assert emf
    assert all(s["pwm_counter"] > s["duty"] for s in emf)
    finished(b)


def test_e5_backdriven_rotor_detected_and_fought(params, bldcsim,
                                                 align_offset):
    """An overhauling load spins the rotor backward: within the sensor's
    tracking range the signed speed path must read it as negative (reverse
    flag) and drive max forward torque instead of believing the target was
    met. (Beyond ~120 rad/s the AS5600 frame rate aliases the sector
    sequence - direction detection is only specified inside the trackable
    envelope; see edge-case-scenarios.md E5 note.)"""
    b = bldcsim.Bench(bench_config(params))
    init_bench(b, params)
    b.set_align_offset(align_offset)
    b.set_target_speed(40)
    b.set_mode(2)
    b.set_load_torque(0.5)   # short shove: backdrive into tracking range
    b.run_for(0.05)
    b.set_load_torque(0.12)  # sustained overhauling load the motor can fight
    b.run_for(0.06)
    assert b.omega < -20, f"expected backdrive, omega={b.omega:.1f}"
    assert b.speed_reverse, "reverse rotation not detected"
    assert b.duty > 550, (
        f"loop must fight the backdrive at max duty, duty={b.duty}"
    )
    # Load drops below the motor's capability: it must arrest the backdrive
    # and recover forward to target.
    b.set_load_torque(0.02)
    b.run_for(1.2)
    assert b.omega > 25, f"did not recover forward: {b.omega:.1f}"
    assert not b.speed_reverse
    finished(b)


def test_e6_mode_transitions_midflight(params, bldcsim, align_offset):
    b = bldcsim.Bench(bench_config(params))
    init_bench(b, params)
    clk = params.value("rtl.clock_frequency")
    b.set_align_offset(align_offset)
    b.set_target_speed(70)
    b.set_mode(2)
    b.run_for(0.7)
    assert b.omega > 50
    # Hot-switch to open loop and back.
    b.set_open_loop(freq_word(30.0, clk), 2**30)
    b.set_duty(int(0.5 * 625))
    b.set_mode(1)
    b.run_for(0.3)
    b.set_mode(2)
    b.run_for(0.7)
    assert b.omega > 50, f"did not re-settle after mode bounce: {b.omega:.1f}"
    # Align rewrite at speed: transient, then recovery.
    b.set_align_offset((align_offset + 341) & 0xFFF)
    b.run_for(0.2)
    b.set_align_offset(align_offset)
    b.run_for(0.5)
    assert b.omega > 50
    assert b.fault_count == 0
    finished(b)


def test_e7_target_zero_and_unreachable(params, bldcsim, align_offset):
    b = bldcsim.Bench(bench_config(params))
    init_bench(b, params)
    b.set_align_offset(align_offset)
    # Unreachable target: saturates at the latency ceiling without faulting.
    b.set_target_speed(500)
    b.set_mode(2)
    b.run_for(1.0)
    ceiling = b.omega
    assert 80 < ceiling < 200, f"ceiling {ceiling:.0f}"
    assert b.fault_count == 0
    # Drop to a sane target: anti-windup must let it come back down quickly.
    b.set_target_speed(60)
    b.run_for(1.2)
    assert abs(b.omega - 60) < 18, f"windup recovery failed: {b.omega:.1f}"
    # Target zero = coast, documented policy: duty collapses to (near)
    # zero - the integrator may hold a sub-slew-step residue - and the
    # rotor decays on friction alone (mech tau = J/B = 2 s, so one second
    # of coasting from ~60 lands near 60*e^-0.5 ~ 36; full stop takes
    # several seconds and is not what this asserts).
    w_before = b.omega
    b.set_target_speed(0)
    b.run_for(1.0)
    assert b.duty <= 40, f"coast policy violated: duty={b.duty}"
    assert b.omega < 0.75 * w_before, (
        f"not coasting: {w_before:.1f} -> {b.omega:.1f}"
    )
    finished(b)


# ---------------------------------------------------------------------------
# B. Fault pile-ups
# ---------------------------------------------------------------------------

def test_e8_fault_timing_sweep_across_refresh(params, bldcsim, align_offset):
    """Latched faults injected at staggered offsets relative to the refresh
    period, including mid-verify: every one must recover."""
    b = bldcsim.Bench(bench_config(params))
    init_bench(b, params)
    b.set_align_offset(align_offset)
    b.set_target_speed(60)
    b.set_mode(2)
    b.run_for(0.5)
    refresh = params.value("rtl.drv_refresh_period")
    for k, frac in enumerate((0.1, 0.45, 0.8, 0.97)):
        b.run_for(frac * refresh)
        b.inject_drv_latched_fault()
        b.run_for(0.45)  # recovery window + healthy time (clears lockout ctr)
        assert b.configured, f"offset {frac}: did not recover"
    assert b.omega > 35
    finished(b)


def test_e9_magnet_loss_during_startup_ramp(params, bldcsim, align_offset):
    b = bldcsim.Bench(bench_config(params))
    init_bench(b, params)
    b.set_align_offset(align_offset)
    b.set_target_speed(70)
    b.set_mode(2)
    b.run_for(0.15)  # mid spin-up
    b.inject_magnet_loss(True)
    b.run_for(0.2)
    assert not b.angle_valid
    b.inject_magnet_loss(False)
    b.run_for(1.0)
    assert b.omega > 50, f"startup did not resume: {b.omega:.1f}"
    finished(b)


def test_e10_dc_cal_offsets_guarded_against_corruption(params, bldcsim):
    """ADC line corrupted through the entire init (including the DC_CAL
    window): the plausibility guard must keep offsets in the sane band and
    raise the offset/stuck telemetry rather than baking garbage."""
    b = bldcsim.Bench(bench_config(params))
    # Per-tick flip probability ~0.5: the ADC data stream is pure noise.
    b.set_line_glitches(0b10, 1.25e7, expected_init_time(params), 4242)
    init_bench(b, params)
    tol = int(params.value("rtl.dc_cal_offset_tol"))
    assert abs(b.offset_a - 2048) <= tol
    assert abs(b.offset_b - 2048) <= tol
    assert b.offset_fault or b.adc_stuck, (
        "corrupted calibration produced no telemetry flag"
    )
    finished(b)


def test_e11_e20_fault_storm_ends_in_lockout(params, bldcsim, align_offset):
    """Repeated faults without a healthy interval must latch lockout (the
    recovery-storm guard) instead of cycling forever; idling clears it."""
    b = bldcsim.Bench(bench_config(params))
    init_bench(b, params)
    b.set_align_offset(align_offset)
    b.set_target_speed(60)
    b.set_mode(2)
    b.run_for(0.4)
    for _ in range(int(params.value("rtl.fault_lockout_threshold")) + 2):
        b.inject_drv_latched_fault()
        b.run_for(0.02)  # well under the healthy-run interval
    b.run_for(0.1)
    assert b.locked_out, "fault storm did not latch lockout"
    assert not b.configured
    assert all(abs(i) < 0.5 for i in b.currents)

    # Host acknowledges by idling >100 ms: lockout clears, system re-inits.
    b.set_mode(0)
    b.run_for(0.15 + expected_init_time(params))
    assert not b.locked_out
    assert b.configured
    b.set_mode(2)
    b.run_for(0.8)
    assert b.omega > 35
    finished(b)


def test_e12_kitchen_sink(params, bldcsim, align_offset):
    b = bldcsim.Bench(realism(params, "supply", "mechanical", "disturbance",
                              "thermal", "sensor",
                              supply={"i_limit_a": 20.0}))
    init_bench(b, params)
    b.set_align_offset(align_offset)
    b.set_target_speed(60)
    b.set_mode(2)
    b.run_for(1.5)
    tail = [w for t, w in zip(b.trace["time_s"], b.trace["omega_rad_s"])
            if t > b.time_s - 0.3]
    mean_tail = sum(tail) / len(tail)
    assert abs(mean_tail - 60) < 0.25 * 60, f"kitchen sink: {mean_tail:.1f}"
    finished(b)


# ---------------------------------------------------------------------------
# C. Dead and degenerate peripherals
# ---------------------------------------------------------------------------

def test_e13_dead_drv_declared_not_hammered(params, bldcsim):
    """DRV SPI returns garbage forever: after DRV_DEAD_N rewrite attempts
    the manager must declare it dead and stop, not livelock."""
    b = bldcsim.Bench(bench_config(params))
    # Per-tick flip probability ~0.5: every readback frame is pure noise
    # (lower rates let most frames through intact and the dead counter
    # keeps resetting - a finding in itself about partial corruption).
    b.set_line_glitches(0b1, 1.25e7, 1e9, 31415)  # drv_miso scrambled
    # Host is actively requesting run mode: mode 0 is the CLEAR gesture, so
    # holding it would auto-acknowledge the dead state in a retry loop.
    b.set_mode(2)
    b.run_for(expected_init_time(params) + 0.1)
    assert b.drv_dead, "manager kept hammering a dead DRV"
    assert not b.configured

    # Hardware 'fixed' (glitches end) + host idles: full retry succeeds.
    b.set_line_glitches(0, 0.0, 0.0, 0)
    b.set_mode(0)
    b.run_for(0.15 + expected_init_time(params))
    assert not b.drv_dead
    assert b.configured
    finished(b)


def test_e14_stuck_adc_flagged(params, bldcsim):
    """All-zero analog world (dead ADC / shorted rails): the stuck flag must
    rise; the speed-based six-step loop itself keeps working."""
    b = bldcsim.Bench(bench_config(
        params,
        chain={"amp_offset_v": 0.0, "emf_divider": 0.0, "bus_divider": 0.0}))
    init_bench(b, params)
    b.run_for(0.05)
    assert b.adc_stuck, "rail-pinned conversions not flagged"
    finished(b)


def test_e15_carrier_validity_hysteresis(params, bldcsim, align_offset):
    """Carrier inside the loose band but outside the strict band: a fresh
    capture must NOT validate (no run/coast flapping from a marginal
    sensor); modest drift inside the strict band must work."""
    nominal = params.value("angle_sensor.pwm_carrier")
    ok = bldcsim.Bench(bench_config(
        params, encoder={"pwm_carrier_hz": nominal * 1.10}))
    init_bench(ok, params)
    ok.set_align_offset(align_offset)
    ok.set_target_speed(50)
    ok.set_mode(2)
    ok.run_for(0.8)
    assert ok.angle_valid and ok.omega > 30

    marginal = bldcsim.Bench(bench_config(
        params, encoder={"pwm_carrier_hz": nominal * 1.22}))
    init_bench(marginal, params)
    marginal.set_align_offset(align_offset)
    marginal.set_target_speed(50)
    marginal.set_mode(2)
    marginal.run_for(0.5)
    assert not marginal.angle_valid, (
        "marginal carrier validated - hysteresis missing"
    )
    assert abs(marginal.omega) < 5  # safely refused to run
    finished(ok)
    finished(marginal)


def test_e16_uart_torn_frame_recovers(params, bldcsim):
    """A truncated write must time out and not poison subsequent frames."""
    from bench_factory import uart_read_frame, uart_write_frame

    b = bldcsim.Bench(bench_config(params))
    init_bench(b, params)
    baud = params.value("rtl.uart_baud")
    b.uart_send([0x82])  # write command for reg 2... and nothing else
    b.run_for(2 * 10 / baud)
    b.run_for(params.value("rtl.uart_byte_timeout") + 1e-3)

    b.uart_send(uart_write_frame(2, 77))
    b.run_for(4 * 10 / baud)
    b.uart_take_received()
    b.uart_send(uart_read_frame(2))
    b.run_for(6 * 10 / baud)
    data = b.uart_take_received()
    assert len(data) >= 2
    assert ((data[-2] << 8) | data[-1]) == 77, (
        "torn frame poisoned the link (E16)"
    )
    finished(b)


def test_e17_uart_flood(params, bldcsim):
    from bench_factory import uart_read_frame, uart_write_frame

    b = bldcsim.Bench(bench_config(params))
    init_bench(b, params)
    baud = params.value("rtl.uart_baud")
    # Writes flood at line rate: every one must land, in order.
    frames = []
    for v in range(10):
        frames += uart_write_frame(2, 100 + v)
    b.uart_take_received()
    b.uart_send(frames)
    b.run_for(len(frames) * 10 / baud + 2e-3)
    # Reads are PACED (protocol contract: the regfile cannot queue commands
    # while transmitting a response - the host waits for each 2-byte reply).
    responses = []
    for _ in range(5):
        b.uart_send(uart_read_frame(2))
        b.run_for(6 * 10 / baud)
        got = b.uart_take_received()
        assert len(got) == 2, f"paced read lost its response: {got}"
        responses.append((got[0] << 8) | got[1])
    assert set(responses) == {109}, (
        f"flooded writes lost data: {responses}"
    )
    finished(b)


# ---------------------------------------------------------------------------
# D. Power sequencing
# ---------------------------------------------------------------------------

def test_e18_slow_supply_ramp_boot(params, bldcsim):
    """Rail crawling up through the UVLO region while the controller boots:
    messy (UV resets, fault retries, possibly a lockout-and-clear cycle) but
    it must converge to configured with zero shoot-through."""
    b = bldcsim.Bench(realism(params, "supply",
                              supply={"i_limit_a": 20.0,
                                      "ramp_time_s": 0.05}))
    b.run_for(expected_init_time(params) + 0.35)
    assert b.pvdd_uv_events >= 1, "boot never swept the UVLO region?"
    assert b.configured, "did not converge after the ramped boot"
    assert b.bus_v > 11.0
    finished(b)


def test_e19_24v_operation(params, bldcsim, align_offset):
    b = bldcsim.Bench(bench_config(params, vbus_v=24.0,
                                   bridge={"vbus_v": 24.0}))
    init_bench(b, params)
    b.set_align_offset(align_offset)
    b.set_target_speed(80)
    b.set_mode(2)
    b.run_for(1.0)
    assert abs(b.omega - 80) < 0.2 * 80, f"24 V settle: {b.omega:.1f}"
    # EMF channels stay inside the rail at this bus voltage (saturation
    # starts near ~34 V phase peak with the 0.095 divider).
    emf = [s for s in b.adc_samples if 3 <= s["channel"] <= 5]
    assert emf and all(s["code"] < 4095 for s in emf)
    finished(b)


# ---------------------------------------------------------------------------
# E. Mechanical and load edges
# ---------------------------------------------------------------------------

def test_e21_stall_detected_safe_off(params, bldcsim, align_offset):
    """A mechanically jammed rotor (Coulomb friction far beyond the motor's
    capability - a load torque would backdrive instead of jamming): duty
    winds to max, speed and angle freeze, the stall detector latches
    safe-off. Idling clears the latch; with the jam still present the
    detector correctly re-fires."""
    b = bldcsim.Bench(bench_config(
        params, motor={"coulomb_friction_nm": 2.0,
                       "stiction_omega_eps": 0.5}))
    init_bench(b, params)
    b.set_align_offset(align_offset)
    b.set_target_speed(60)
    b.set_mode(2)
    b.run_for(0.8)
    assert abs(b.omega) < 1.0  # jammed solid
    assert b.stalled, "stall never detected"
    assert all(abs(i) < 0.5 for i in b.currents), "gates not killed on stall"
    # Host acknowledges by idling: latch clears...
    b.set_mode(0)
    b.run_for(0.05)
    assert not b.stalled
    # ...and with the jam still present, re-running re-detects the stall.
    b.set_mode(2)
    b.run_for(0.8)
    assert b.stalled, "re-stall on a still-jammed rotor not detected"
    finished(b)


def test_e22_load_oscillation_sweep(params, bldcsim, align_offset):
    import statistics
    ripples = {}
    for freq in (2.0, 8.0, 30.0):
        b = bldcsim.Bench(bench_config(params))
        init_bench(b, params)
        b.set_align_offset(align_offset)
        b.set_target_speed(70)
        b.set_mode(2)
        b.run_for(0.8)
        b.set_load_oscillation(0.02, freq)
        b.clear_trace()
        b.run_for(0.8)
        tail = b.trace["omega_rad_s"]
        ripples[freq] = statistics.pstdev(tail)
        assert abs(sum(tail) / len(tail) - 70) < 0.25 * 70
        assert ripples[freq] < 0.3 * 70, (
            f"{freq} Hz load: ripple {ripples[freq]:.1f} unbounded"
        )
        finished(b)


def test_e23_inertia_bracket(params, bldcsim, align_offset):
    """Brackets the regulation envelope for the placeholder PI gains:
    10x J and 0.5x J regulate cleanly; 0.1x J sits OUTSIDE the envelope and
    limit-cycles (sensor latency + slew limit cannot catch the overshoot of
    a feather-light rotor) - it must do so BOUNDED and fault-free. This is
    the bracketing knowledge motor-ID day needs: retune mandatory below
    ~0.5x of the gains' design inertia."""
    import statistics
    j0 = params.value("motor.J")

    heavy = bldcsim.Bench(bench_config(params,
                                       motor={"inertia_kg_m2": 10 * j0}))
    init_bench(heavy, params)
    heavy.set_align_offset(align_offset)
    heavy.set_target_speed(50)
    heavy.set_mode(2)
    heavy.run_for(3.0)
    assert abs(heavy.omega - 50) < 0.25 * 50, f"10x J: {heavy.omega:.1f}"
    finished(heavy)

    half = bldcsim.Bench(bench_config(params,
                                      motor={"inertia_kg_m2": 0.5 * j0}))
    init_bench(half, params)
    half.set_align_offset(align_offset)
    half.set_target_speed(50)
    half.set_mode(2)
    half.run_for(1.2)
    tail = [w for t, w in zip(half.trace["time_s"],
                              half.trace["omega_rad_s"])
            if t > half.time_s - 0.3]
    assert abs(sum(tail) / len(tail) - 50) < 0.2 * 50
    assert statistics.pstdev(tail) < 8
    finished(half)

    # Outside the envelope: a bounded, fault-free limit cycle - not a crash.
    light = bldcsim.Bench(bench_config(params,
                                       motor={"inertia_kg_m2": 0.1 * j0}))
    init_bench(light, params)
    light.set_align_offset(align_offset)
    light.set_target_speed(50)
    light.set_mode(2)
    light.run_for(1.5)
    peak = max(abs(w) for w in light.trace["omega_rad_s"])
    assert peak < 250, f"0.1x J: unbounded excursion {peak:.0f}"
    assert light.fault_count == 0
    assert not light.stalled
    finished(light)


def test_e24_flipped_magnet_full_circle_calibration(params, bldcsim,
                                                    align_offset):
    """Magnet mounted 180 deg off: only a full-circle calibration sweep can
    find the working offset; the found offset must actually spin the motor."""
    flipped = {"mounting_offset_rad": math.pi}

    def run_candidate(cand):
        b = bldcsim.Bench(bench_config(params, encoder=flipped))
        init_bench(b, params)
        b.set_align_offset(cand & 0xFFF)
        b.set_target_speed(60)
        b.set_mode(2)
        b.run_for(0.9)
        tail_w = [w for t, w in zip(b.trace["time_s"],
                                    b.trace["omega_rad_s"])
                  if t > b.time_s - 0.25]
        if sum(tail_w) / len(tail_w) < 40:
            return None, b
        tail = [d for t, d in zip(b.trace["time_s"], b.trace["duty"])
                if t > b.time_s - 0.25]
        return sum(tail) / len(tail), b

    results = {}
    for step in range(12):  # full circle, half-sector steps
        cand = (align_offset + step * 341) & 0xFFF
        score, _ = run_candidate(cand)
        if score is not None:
            results[cand] = score
    assert results, "no working offset found in the full-circle sweep"
    best = min(results, key=results.get)
    # The flipped magnet shifts the working offset by ~half a mech rev in
    # elec counts: pi * pp = 2 full elec revs -> offset unchanged mod 4096!
    # With pp even, a flipped magnet aliases back: best stays near nominal.
    score, bench = run_candidate(best)
    assert score is not None
    tail = [w for t, w in zip(bench.trace["time_s"],
                              bench.trace["omega_rad_s"])
            if t > bench.time_s - 0.25]
    assert sum(tail) / len(tail) > 40
    finished(bench)


# ---------------------------------------------------------------------------
# F. Numerics
# ---------------------------------------------------------------------------

def test_e26_long_run_theta_growth(params, bldcsim):
    """Energy-consistent torque coupling must hold after theta has grown by
    ~1e5 rad (hours-equivalent of unwrapped angle)."""
    cfg = bench_config(params, motor={"inertia_kg_m2": 2e-5})
    plant = bldcsim.ThreePhasePlant(cfg["motor"], cfg["bridge"],
                                    {"current_epsilon_a": 1e-6,
                                     "max_substep_s": 1e-4})
    plant.set_load_torque(-0.05)  # external spin, gates open
    plant.set_gates([False] * 3, [False] * 3)
    plant.advance(300.0)  # ~500 rad/s * 300 s ~ 1.5e5 rad
    assert plant.theta_rad > 1e5
    out = plant.outputs()
    p_emf = sum(e * i for e, i in zip(out["emf_v"], plant.currents_a))
    p_mech = out["torque_n_m"] * plant.omega_rad_s
    assert abs(p_emf - p_mech) < 1e-9 + 1e-9 * abs(p_mech)


def test_e27_integrator_stability_guard(params, bldcsim):
    # L/R = 0.4 us: far below what the 1 us substep can resolve.
    cfg = bench_config(params, motor={"inductance_h": 2e-7})
    with pytest.raises(ValueError, match="max_substep"):
        bldcsim.ThreePhasePlant(cfg["motor"], cfg["bridge"], cfg["plant"])


# ---------------------------------------------------------------------------
# G. Seeded chaos fuzz: random perturbation sequences, global invariants,
# and recovery. Any failing seed replays exactly.
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("seed", [101, 202, 303, 404])
def test_g_chaos_fuzz(params, bldcsim, align_offset, seed):
    import random

    rng = random.Random(seed)
    b = bldcsim.Bench(realism(params, "mechanical", "disturbance",
                              chain={"noise_rms_v": 2e-3,
                                     "noise_seed": seed}))
    init_bench(b, params)
    b.set_align_offset(align_offset)
    clk = params.value("rtl.clock_frequency")

    actions = [
        lambda: b.set_mode(rng.choice([0, 1, 2, 2])),
        lambda: b.set_target_speed(rng.choice([0, 30, 60, 90, 200])),
        lambda: b.set_duty(rng.randrange(0, 626)),
        lambda: b.set_open_loop(freq_word(rng.uniform(5, 35), clk), 1),
        lambda: b.set_align_offset(
            (align_offset + rng.choice([-341, 0, 341])) & 0xFFF),
        lambda: b.inject_drv_latched_fault(),
        lambda: b.inject_drv_register_reset(),
        lambda: b.inject_magnet_loss(rng.random() < 0.5),
        lambda: b.set_load_torque(rng.choice([0.0, 0.01, 0.03])),
        lambda: b.set_line_glitches(rng.randrange(1, 8), 1500.0,
                                    b.time_s + 0.02, rng.randrange(9999)),
    ]
    for _ in range(14):
        rng.choice(actions)()
        b.run_for(rng.uniform(0.03, 0.12))
        # Global invariants hold under ANY perturbation sequence.
        assert b.shoot_through_violations == 0
        assert b.plant_shoot_through_requests == 0
        assert all(abs(i) < 50 for i in b.currents)

    # Cleanup and recovery: the system must come back.
    b.inject_magnet_loss(False)
    b.set_line_glitches(0, 0.0, 0.0, 0)
    b.set_load_torque(0.0)
    b.set_mode(0)
    b.run_for(0.2 + expected_init_time(params))  # clears lockout/stall/dead
    b.set_align_offset(align_offset)
    b.set_target_speed(60)
    b.set_mode(2)
    b.run_for(1.0)
    assert b.configured, f"seed {seed}: never re-configured"
    assert b.omega > 35, f"seed {seed}: no recovery, omega={b.omega:.1f}"
    finished(b)
