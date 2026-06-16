# SPDX-License-Identifier: MIT
"""cocotb block test for foc_core (robotics-ip-checklist stages 4 + 6.5).

Exercises the PIPELINED FOC datapath standalone and latency-aware:
  * reset / !enable hold the duties at 50% (zero voltage);
  * a zero current / zero target sample leaves the duties at 50% after the walk;
  * a non-zero torque command moves the voltage vector off centre, and the new
    duties appear a FEW CYCLES after `update` (the sequencer latency) and not on
    the same cycle - i.e. the pipeline is real, and it holds between samples.
This complements the cocotb equivalence proof of the sequential limiter
(`tb_circle_limit_seq`) and the in-system FOC regression.
"""

import cocotb
from cocotb.clock import Clock
from cocotb.triggers import RisingEdge

PWM_HALF_PERIOD = 625
CENTER = PWM_HALF_PERIOD >> 1                       # 312, 50% duty
ALL_CENTER = (CENTER << 32) | (CENTER << 16) | CENTER


async def _reset(dut):
    dut.rst_n.value = 0
    dut.enable.value = 0
    dut.update.value = 0
    dut.cur_a.value = 0
    dut.cur_b.value = 0
    dut.theta_e.value = 0
    dut.id_target.value = 0
    dut.iq_target.value = 0
    for _ in range(4):
        await RisingEdge(dut.clk)
    dut.rst_n.value = 1
    await RisingEdge(dut.clk)


async def _sample(dut, *, cur_a=0, cur_b=0, theta=0, idt=0, iqt=0):
    """Latch a current sample (one `update` strobe), return cycles-to-duty-change."""
    before = int(dut.duty3.value)
    dut.cur_a.value = cur_a & 0x3FFFF
    dut.cur_b.value = cur_b & 0x3FFFF
    dut.theta_e.value = theta & 0xFFFF
    dut.id_target.value = idt & 0x3FFFF
    dut.iq_target.value = iqt & 0x3FFFF
    dut.update.value = 1
    await RisingEdge(dut.clk)
    dut.update.value = 0
    changed_at = None
    for k in range(1, 120):
        await RisingEdge(dut.clk)
        if changed_at is None and int(dut.duty3.value) != before:
            changed_at = k
    return changed_at


@cocotb.test()
async def reset_and_enable_hold_centre(dut):
    cocotb.start_soon(Clock(dut.clk, 10, units="ns").start())
    await _reset(dut)
    assert int(dut.duty3.value) == ALL_CENTER, "reset must hold 50% duty"
    # enable low: still centred even if an update is presented
    await _sample(dut, iqt=500)
    assert int(dut.duty3.value) == ALL_CENTER, "!enable must hold 50% duty"


@cocotb.test()
async def zero_sample_stays_centre(dut):
    cocotb.start_soon(Clock(dut.clk, 10, units="ns").start())
    await _reset(dut)
    dut.enable.value = 1
    await RisingEdge(dut.clk)
    await _sample(dut, cur_a=0, cur_b=0, theta=0, idt=0, iqt=0)
    assert int(dut.duty3.value) == ALL_CENTER, "zero voltage -> 50% all legs"
    assert int(dut.dbg_id.value) == 0 and int(dut.dbg_iq.value) == 0


@cocotb.test()
async def torque_command_moves_vector_after_latency(dut):
    cocotb.start_soon(Clock(dut.clk, 10, units="ns").start())
    await _reset(dut)
    dut.enable.value = 1
    await RisingEdge(dut.clk)

    changed_at = await _sample(dut, cur_a=0, cur_b=0, theta=0, idt=0, iqt=300)
    # the duties moved off centre...
    assert int(dut.duty3.value) != ALL_CENTER, "torque command must move the vector"
    # ...the q-axis voltage is non-zero...
    assert int(dut.dbg_vq.value) != 0, "vq should be non-zero under torque"
    # ...and the change took several cycles (pipelined), not the same cycle.
    assert changed_at is not None, "duties never updated"
    assert changed_at >= 4, f"duty changed too soon ({changed_at}); not pipelined"

    # between samples the duties HOLD (no spurious change without a new update).
    held = int(dut.duty3.value)
    for _ in range(40):
        await RisingEdge(dut.clk)
    assert int(dut.duty3.value) == held, "duties must hold between samples"
