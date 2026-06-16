# SPDX-License-Identifier: MIT
"""Stage 3.1/3.2: DRV8301 behavioral model against datasheet golden values
(register map and SPI behavior verified in notes/docs-digest.md)."""

from __future__ import annotations

import pytest


def drv_config(params, **overrides):
    d = {
        "en_gate_ready_time_s": params.value("drv8301.en_gate_ready_time"),
        "quick_reset_pulse_max_s": params.value(
            "drv8301.quick_reset_pulse_max"),
        "dead_time_floor_s": params.value("inverter.dtc_dead_time_floor"),
        "noctw_pulse_s": 64e-6,
        "rds_on_ohm": params.value("inverter.fet_rds_on"),
        "oc_adj_set_default": int(params.value("drv8301.oc_adj_set")),
    }
    d.update(overrides)
    return d


class DrvHarness:
    """Drives the DRV8301 model pins with a monotonic clock (mode-1 SPI)."""

    def __init__(self, drv, t0=0.0, tick=40e-9):
        self.drv = drv
        self.t = t0
        self.tick = tick
        self.pins = {
            "en_gate": False, "dc_cal": False,
            "inh": [False] * 3, "inl": [False] * 3,
            "nscs": True, "sclk": False, "sdi": False,
        }
        self.currents = [0.0, 0.0, 0.0]

    def step(self, **pin_updates):
        self.pins.update(pin_updates)
        self.t += self.tick
        self.drv.update(self.t, self.pins, self.currents)

    def wait(self, duration_s):
        end = self.t + duration_s
        while self.t < end:
            self.t += min(self.tick * 100, end - self.t)
            self.drv.update(self.t, self.pins, self.currents)

    def spi_frame(self, word16, bits=16):
        """Clock one mode-1 frame; returns the 16-bit response word."""
        response = 0
        self.step(nscs=False, sclk=False)
        for n in range(bits):
            bit = (word16 >> (15 - n)) & 1
            self.step(sdi=bool(bit))
            self.step(sclk=True)        # rising: slave shifts SDO out
            response = (response << 1) | (1 if self.drv.sdo else 0)
            self.step(sclk=False)       # falling: slave samples SDI
        self.step(nscs=True)
        return response

    def spi_write(self, addr, data):
        return self.spi_frame((0 << 15) | (addr << 11) | (data & 0x7FF))

    def spi_read(self, addr):
        self.spi_frame((1 << 15) | (addr << 11))
        # N+1 pipelining: response arrives in the next frame (dummy read).
        return self.spi_frame((1 << 15) | (addr << 11))

    def enable_and_ready(self, params):
        self.step(en_gate=True)
        self.wait(params.value("drv8301.en_gate_ready_time") + 1e-4)


@pytest.fixture
def harness(params, bldcsim):
    return DrvHarness(bldcsim.Drv8301(drv_config(params)))


def test_not_ready_before_en_gate_wait(params, harness):
    h = harness
    h.step(en_gate=True)
    h.wait(1e-3)  # well short of the 10 ms ready time
    assert not h.drv.ready
    # Gates ignore inputs while not ready.
    h.step(inh=[True, False, False], inl=[False, True, False])
    assert h.drv.gate_high == [False] * 3
    # Writes before ready are dropped.
    h.spi_write(3, 0x008)
    assert h.drv.reg(3) == 0
    h.wait(params.value("drv8301.en_gate_ready_time"))
    assert h.drv.ready


def test_spi_write_read_golden_frames(params, harness):
    h = harness
    h.enable_and_ready(params)
    # Write CR2 GAIN=40 V/V (D[3:2] = 10 -> 0x008).
    h.spi_write(3, 0x008)
    assert h.drv.reg(3) == 0x008
    # Read back: response word = addr(0x3) in [14:11], data in [10:0].
    resp = h.spi_read(3)
    assert resp == (0x3 << 11) | 0x008
    assert (resp >> 15) == 0  # no frame fault


def test_spi_write_response_carries_status1(params, harness):
    h = harness
    h.enable_and_ready(params)
    h.spi_write(2, 0x000)
    # Second write's response reflects SR1 (all clear -> 0 with addr 0).
    resp = h.spi_write(2, 0x000)
    assert resp == 0x0000


def test_short_frame_sets_fault_flag_and_ignores_write(params, harness):
    h = harness
    h.enable_and_ready(params)
    h.spi_frame((0 << 15) | (3 << 11) | 0x7FF, bits=15)  # 15 clocks: invalid
    assert h.drv.reg(3) == 0  # write dropped
    resp = h.spi_frame((1 << 15) | (3 << 11))
    assert (resp >> 15) & 1 == 1  # frame-fault flag in next response
    assert h.drv.frame_errors == 1


def test_oc_current_limit_mode_blocks_until_next_cycle(params, harness):
    h = harness
    h.enable_and_ready(params)
    trip_i = h.drv.vds_threshold_v / params.value("inverter.fet_rds_on")

    h.step(inh=[True, False, False], inl=[False, False, False])
    assert h.drv.gate_high[0]
    h.currents = [trip_i * 1.2, 0.0, 0.0]
    h.step()
    assert h.drv.oc_events >= 1
    assert not h.drv.gate_high[0]      # cycle-by-cycle block
    assert not h.drv.noctw             # warning pin pulsing low
    assert h.drv.nfault                # not a latched fault

    h.currents = [0.0, 0.0, 0.0]
    h.step()
    assert not h.drv.gate_high[0]      # still blocked this cycle
    h.step(inh=[False, False, False])  # input deasserts...
    h.step(inh=[True, False, False])   # ...next PWM cycle re-arms
    assert h.drv.gate_high[0]

    # Status bit FETHA_OC (D5) latched until SR1 read, then clears.
    sr1 = h.spi_read(0) & 0x7FF
    assert sr1 & (1 << 5)
    assert sr1 & (1 << 10)  # FAULT summary bit
    sr1_again = h.spi_read(0) & 0x7FF
    assert sr1_again == 0


def test_oc_latch_mode_and_gate_reset(params, harness):
    h = harness
    h.enable_and_ready(params)
    h.spi_write(2, 0x1 << 4)  # OCP_MODE = 01 latch shutdown
    trip_i = h.drv.vds_threshold_v / params.value("inverter.fet_rds_on")

    h.step(inl=[True, False, False], inh=[False, False, False])
    h.currents = [-trip_i * 1.5, 0.0, 0.0]
    h.step()
    assert not h.drv.nfault            # latched fault asserts nFAULT
    assert not h.drv.gate_low[0]
    h.currents = [0.0, 0.0, 0.0]
    h.step(inl=[False, False, False])
    h.step(inl=[True, False, False])
    assert not h.drv.gate_low[0]       # still latched across cycles

    # GATE_RESET (CR1 D2, self-clearing) releases the latch.
    h.spi_write(2, h.drv.reg(2) | (1 << 2))
    h.step()
    assert h.drv.nfault
    assert (h.drv.reg(2) >> 2) & 1 == 0  # bit self-cleared
    h.step(inl=[False, False, False])
    h.step(inl=[True, False, False])
    assert h.drv.gate_low[0]


def test_dead_time_floor_enforced(params, bldcsim):
    floor = 500e-9
    h = DrvHarness(bldcsim.Drv8301(drv_config(params,
                                               dead_time_floor_s=floor)))
    h.enable_and_ready(params)
    h.step(inl=[True, False, False])
    assert h.drv.gate_low[0]
    h.step(inl=[False, False, False])  # GL off
    t_off = h.t
    h.step(inh=[True, False, False])   # immediately request GH
    assert not h.drv.gate_high[0]      # held off by dead-time floor
    while h.t - t_off < floor:
        h.step()
        if h.t - t_off < floor:
            assert not h.drv.gate_high[0]
    h.step()
    assert h.drv.gate_high[0]


def test_shoot_through_input_guard(params, harness):
    h = harness
    h.enable_and_ready(params)
    h.step(inh=[True, False, False], inl=[True, False, False])
    assert not h.drv.gate_high[0] and not h.drv.gate_low[0]


def test_register_reset_injection_slva552(params, harness):
    h = harness
    h.enable_and_ready(params)
    h.spi_write(3, 0x008)
    h.spi_write(2, (5 << 6) | (2 << 4))  # OC_ADJ code 5, report-only
    assert h.drv.reg(3) == 0x008
    h.drv.inject_register_reset()
    default_cr1 = (int(params.value("drv8301.oc_adj_set")) & 0x1F) << 6
    assert h.drv.reg(2) == default_cr1
    assert h.drv.reg(3) == 0
    # Device keeps running (no fault pin indication) - that is the trap.
    assert h.drv.nfault


def test_quick_reset_pulse_clears_faults_without_restart(params, harness):
    h = harness
    h.enable_and_ready(params)
    h.spi_write(2, 0x1 << 4)  # latch mode
    trip_i = h.drv.vds_threshold_v / params.value("inverter.fet_rds_on")
    h.step(inh=[True, False, False])
    h.currents = [trip_i * 2, 0.0, 0.0]
    h.step()
    h.currents = [0.0, 0.0, 0.0]
    assert not h.drv.nfault

    # EN_GATE low pulse shorter than the quick-reset window.
    h.step(en_gate=False, inh=[False, False, False])
    h.wait(2e-6)
    h.step(en_gate=True)
    h.step()
    assert h.drv.nfault     # fault cleared
    assert h.drv.ready      # no full restart


def test_full_en_gate_cycle_resets_registers(params, harness):
    h = harness
    h.enable_and_ready(params)
    h.spi_write(3, 0x008)
    h.step(en_gate=False)
    h.wait(1e-3)  # far beyond the quick-reset window
    h.step(en_gate=True)
    assert not h.drv.ready
    assert h.drv.reg(3) == 0  # full shutdown reset registers


def test_otw_injection_reports_on_noctw_only(params, harness):
    h = harness
    h.enable_and_ready(params)
    h.drv.inject_otw(True)
    h.step()
    assert not h.drv.noctw  # warning
    assert h.drv.nfault     # OTW alone does not assert nFAULT
    sr1 = h.spi_read(0) & 0x7FF
    assert sr1 & (1 << 6)   # OTW bit
