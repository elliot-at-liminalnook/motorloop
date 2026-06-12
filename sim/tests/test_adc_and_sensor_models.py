"""Stage 3.3-3.6: MCP3208, AS5600, and feedback-chain models."""

from __future__ import annotations

import math

import pytest


# ---------------------------------------------------------------------------
# MCP3208
# ---------------------------------------------------------------------------

def adc_config(params, **overrides):
    d = {
        "vref_v": params.value("adc.vref"),
        "max_sclk_hz": params.value("adc.sclk"),
        "min_cs_high_s": 500e-9,
    }
    d.update(overrides)
    return d


class AdcHarness:
    """Mode 0,0 SPI master for the MCP3208 model."""

    def __init__(self, adc, sclk_hz=1.0e6, t0=1e-6):
        self.adc = adc
        self.t = t0
        self.half = 0.5 / sclk_hz
        self.cs_n = True
        self.sclk = False
        self.din = False
        self._push()

    def _push(self):
        self.adc.update(self.t, self.cs_n, self.sclk, self.din)

    def idle(self, duration_s):
        self.t += duration_s
        self._push()

    def convert(self, channel, single_ended=True):
        """One full conversion; returns (code, sample_time)."""
        self.cs_n = False
        self.sclk = False
        self.t += self.half
        self._push()

        command = [True, single_ended,
                   bool(channel & 4), bool(channel & 2), bool(channel & 1)]
        code = 0
        # 5 command bits + 1 sample-completion clock + 12 data clocks.
        for n in range(18):
            self.din = command[n] if n < len(command) else False
            self.t += self.half
            self.sclk = True
            self._push()      # rising: device samples DIN
            self.t += self.half
            self.sclk = False
            self._push()      # falling: device shifts DOUT
            if n >= 5:        # null bit on falling #5... data on #6..#17
                if n >= 6:
                    code = (code << 1) | (1 if self.adc.dout else 0)
        self.cs_n = True
        self.t += self.half
        self._push()
        return code, self.adc.last_sample["time_s"]


@pytest.fixture
def adc(params, bldcsim):
    a = bldcsim.Mcp3208(adc_config(params))
    a.set_channels([0.0, 0.4125, 0.825, 1.65, 2.475, 3.3, 1.0, 0.1])
    return a


def test_known_voltages_to_codes(params, adc):
    h = AdcHarness(adc)
    # vref = 3.3: LSB = 3.3/4096; 1.65 V -> 2048, 0.825 -> 1024, 3.3 -> 4095.
    expected = {0: 0, 1: 512, 2: 1024, 3: 2048, 4: 3072, 5: 4095}
    for ch, want in expected.items():
        code, _ = h.convert(ch)
        h.idle(1e-6)
        assert code == want, f"ch{ch}: {code} vs {want}"


def test_sample_instant_is_hold_edge_not_cs(params, adc):
    h = AdcHarness(adc)
    t_cs_fall = h.t
    _, t_sample = h.convert(3)
    # Hold instant: falling edge of the 5th clock after start, well after CS.
    assert t_sample > t_cs_fall + 4 * 2 * h.half
    assert t_sample < t_cs_fall + 8 * 2 * h.half
    assert adc.conversions == 1


def test_sclk_overclock_guard(params, bldcsim):
    a = bldcsim.Mcp3208(adc_config(params, max_sclk_hz=1.0e6))
    a.set_channels([1.0] * 8)
    h = AdcHarness(a, sclk_hz=2.0e6)  # double the configured maximum
    h.convert(0)
    assert a.sclk_too_fast_count > 0


def test_cs_high_time_guard(params, bldcsim):
    a = bldcsim.Mcp3208(adc_config(params))
    a.set_channels([1.0] * 8)
    # Fast harness clock so the total CS-high gap stays under 500 ns
    # (the harness inserts half a clock period before re-asserting CS).
    h = AdcHarness(a, sclk_hz=5.0e6)
    h.convert(0)
    h.idle(100e-9)
    h.convert(1)
    assert a.cs_too_short_count == 1


def test_differential_mode_counted(params, adc):
    h = AdcHarness(adc)
    h.convert(2, single_ended=False)
    assert adc.differential_requests == 1


def test_sample_residual_crosstalk(params, bldcsim):
    """Shared-cap residual (derivation-checklist 2.6): a large residual on
    one channel pulls its sample toward the previous conversion's voltage;
    theft voltage is exposed for the reservoir feedback."""
    d = 0.01
    cfg = adc_config(params)
    cfg["sample_residual"] = [0.0, 0.0, 0.0, d, 0.0, 0.0, 0.0, 0.0]
    a = bldcsim.Mcp3208(cfg)
    a.set_channels([0.0, 0.0, 0.0, 3.3, 0.0, 0.0, 0.0, 0.0])
    h = AdcHarness(a)

    h.convert(0)         # cap now holds 0.0 V
    h.idle(1e-6)
    code, _ = h.convert(3)
    expected_v = 3.3 + d * (0.0 - 3.3)
    expected_code = int(expected_v * 4096 / 3.3)
    assert abs(code - expected_code) <= 1, f"{code} vs {expected_code}"
    assert abs(a.last_sample_theft_v - d * (0.0 - 3.3)) < 1e-12

    # Zero-residual channels are exact regardless of cap history.
    h.idle(1e-6)
    code4, _ = h.convert(4)
    assert code4 == 0


# ---------------------------------------------------------------------------
# AS5600
# ---------------------------------------------------------------------------

def sensor_config(params, **overrides):
    d = {
        "sample_period_s": params.value("angle_sensor.sample_period"),
        "filter_settling_s": params.value("angle_sensor.filter_settling"),
        "pwm_carrier_hz": params.value("angle_sensor.pwm_carrier"),
    }
    d.update(overrides)
    return d


def decode_pwm_frame(sensor, theta, t0, carrier_hz, dt=200e-9):
    """Sample out() through one full frame; return decoded angle12."""
    frame = 1.0 / carrier_hz
    # Align to a frame boundary: wait for a rising edge of out.
    t = t0
    sensor.update(t, theta)
    prev = sensor.out
    t_rise = None
    while t < t0 + 2.5 * frame:
        t += dt
        sensor.update(t, theta)
        if sensor.out and not prev:
            t_rise = t
            break
        prev = sensor.out
    assert t_rise is not None, "no rising edge found"
    high = 0
    total = int(round(frame / dt))
    for n in range(total):
        t += dt
        sensor.update(t, theta)
        if sensor.out:
            high += 1
    # high units = 128 + angle out of 4351.
    units = high / total * 4351.0
    return units - 128.0


def test_pwm_roundtrip_static_angle(params, bldcsim):
    cfg = sensor_config(params)
    sensor = bldcsim.As5600(cfg)
    theta = math.radians(123.4)
    # Let sampling + filter fully settle.
    t = 0.0
    while t < 20e-3:
        t += 50e-6
        sensor.update(t, theta)
    expected12 = theta / (2 * math.pi) * 4096
    decoded = decode_pwm_frame(sensor, theta, t, cfg["pwm_carrier_hz"])
    assert abs(decoded - expected12) < 8, f"{decoded} vs {expected12}"


def test_filter_latency_dominates(params, bldcsim):
    """Step the angle and measure when the filtered value reaches 90%:
    should be on the order of the slow-filter settling time."""
    cfg = sensor_config(params)
    sensor = bldcsim.As5600(cfg)
    t = 0.0
    while t < 10e-3:
        t += 50e-6
        sensor.update(t, 0.0)
    target = math.radians(90)
    t_step = t
    while t < t_step + 20e-3:
        t += 50e-6
        sensor.update(t, target)
        if sensor.filtered_angle_rad > 0.9 * target:
            break
    latency = t - t_step
    settling = cfg["filter_settling_s"]
    assert 0.3 * settling < latency < 3.0 * settling, (
        f"90% latency {latency*1e3:.2f} ms vs settling {settling*1e3:.2f} ms"
    )


def test_magnet_loss_kills_output(params, bldcsim):
    sensor = bldcsim.As5600(sensor_config(params))
    t = 0.0
    while t < 5e-3:
        t += 50e-6
        sensor.update(t, 1.0)
    sensor.inject_magnet_loss(True)
    highs = 0
    while t < 10e-3:
        t += 50e-6
        sensor.update(t, 1.0)
        highs += 1 if sensor.out else 0
    assert highs == 0


# ---------------------------------------------------------------------------
# Feedback chain
# ---------------------------------------------------------------------------

def chain_config(params):
    return {
        "shunt_ohm": params.value("feedback.current.shunt"),
        "amp_gain": params.value("drv8301.amp_gain"),
        "amp_offset_v": params.value("feedback.current.offset"),
        "emf_divider": params.value("feedback.emf.divider_ratio"),
        "emf_rc_cutoff_hz": params.value("feedback.emf.rc_cutoff"),
        "bus_divider": params.value("feedback.bus_voltage.divider_ratio"),
        "rail_v": params.value("adc.vref"),
    }


def make_conducting_plant(params, bldcsim):
    """Locked-rotor plant conducting A(high) -> B(low)."""
    motor = {
        "resistance_ohm": params.value("motor.R"),
        "inductance_h": params.value("motor.L"),
        "ke_v_s_per_rad": params.value("motor.Ke"),
        "inertia_kg_m2": 1e9,
        "damping_n_m_s_per_rad": params.value("motor.B"),
        "pole_pairs": int(params.value("motor.pole_pairs")),
        "trapezoid_blend": 0.0,
        "load_torque_n_m": 0.0,
    }
    bridge = {
        "vbus_v": params.value("bus.vbus"),
        "fet_rds_on_ohm": params.value("inverter.fet_rds_on"),
        "diode_vf_v": params.value("inverter.body_diode_vf"),
    }
    config = {
        "current_epsilon_a": params.value("sim.current_epsilon"),
        "max_substep_s": 1e-6,
    }
    plant = bldcsim.ThreePhasePlant(motor, bridge, config)
    plant.set_gates([True, False, False], [False, True, False])
    plant.advance(20e-3)  # settle to steady conduction
    return plant


def test_current_channels_low_side_only(params, bldcsim):
    chain = bldcsim.FeedbackChain(chain_config(params))
    plant = make_conducting_plant(params, bldcsim)
    chain.update_from_plant(1e-3, plant, params.value("bus.vbus"))

    offset = params.value("feedback.current.offset")
    gain = params.value("drv8301.amp_gain")
    shunt = params.value("feedback.current.shunt")
    ia, ib, _ = plant.currents_a

    # Leg A conducts high side: its low-side shunt sees nothing.
    assert abs(chain.channel(0) - offset) < 1e-9
    # Leg B conducts low side with ib = -ia < 0: reads below offset.
    expected_b = offset + gain * shunt * ib
    assert abs(chain.channel(1) - expected_b) < 1e-9
    assert chain.channel(1) < offset


def test_emf_channel_divider_and_rc(params, bldcsim):
    chain = bldcsim.FeedbackChain(chain_config(params))
    plant = make_conducting_plant(params, bldcsim)
    vbus = params.value("bus.vbus")
    # Step the chain repeatedly so the RC settles on the held plant state.
    for _ in range(200):
        chain.update_from_plant(1e-4, plant, vbus)
    divider = params.value("feedback.emf.divider_ratio")
    out = plant.outputs()
    expected = divider * out["terminal_v"][0]
    assert abs(chain.channel(3) - expected) < 1e-3


def test_bus_voltage_channel(params, bldcsim):
    chain = bldcsim.FeedbackChain(chain_config(params))
    plant = make_conducting_plant(params, bldcsim)
    vbus = params.value("bus.vbus")
    chain.update_from_plant(1e-3, plant, vbus)
    expected = params.value("feedback.bus_voltage.divider_ratio") * vbus
    assert abs(chain.channel(6) - expected) < 1e-9


def test_dc_cal_pins_channel_at_offset(params, bldcsim):
    chain = bldcsim.FeedbackChain(chain_config(params))
    plant = make_conducting_plant(params, bldcsim)
    chain.set_dc_cal(1, True)
    chain.update_from_plant(1e-3, plant, params.value("bus.vbus"))
    offset = params.value("feedback.current.offset")
    assert abs(chain.channel(1) - offset) < 1e-9
