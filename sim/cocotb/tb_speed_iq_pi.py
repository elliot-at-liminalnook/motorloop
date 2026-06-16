# SPDX-License-Identifier: MIT
"""cocotb block test for speed_iq_pi (FOC Fmax-optimization checklist, task 1).

Exercises the pipelined outer speed PI standalone and latency-aware: a speed
sample (`update`) produces the clamped q-axis torque command iq* a few cycles
later (the sequencer latency, not the same cycle). Checks the proportional term,
the sign (braking on overspeed), the ±IQ_MAX clamp, and that iq* holds between
samples. Defaults: KP=4, KISH=5, IQ_MAX=300.
"""

import cocotb
from cocotb.clock import Clock
from cocotb.triggers import RisingEdge

IQ_MAX = 300


def _s18(v):
    v &= 0x3FFFF
    return v - 0x40000 if v & 0x20000 else v


async def _reset(dut):
    dut.rst_n.value = 0
    dut.enable.value = 0
    dut.update.value = 0
    dut.target_speed.value = 0
    dut.speed.value = 0
    dut.reverse.value = 0
    for _ in range(4):
        await RisingEdge(dut.clk)
    dut.rst_n.value = 1
    await RisingEdge(dut.clk)


async def _sample(dut, target, speed, reverse=0):
    """One speed sample; return (cycles-to-change, settled iq_target)."""
    before = _s18(int(dut.iq_target.value))
    dut.target_speed.value = target
    dut.speed.value = speed
    dut.reverse.value = reverse
    dut.update.value = 1
    await RisingEdge(dut.clk)
    dut.update.value = 0
    changed_at = None
    for k in range(1, 12):
        await RisingEdge(dut.clk)
        if changed_at is None and _s18(int(dut.iq_target.value)) != before:
            changed_at = k
    return changed_at, _s18(int(dut.iq_target.value))


@cocotb.test()
async def reset_and_disable_zero(dut):
    cocotb.start_soon(Clock(dut.clk, 10, units="ns").start())
    await _reset(dut)
    assert _s18(int(dut.iq_target.value)) == 0
    # enable low: a sample must not move the command
    await _sample(dut, 200, 0)
    assert _s18(int(dut.iq_target.value)) == 0, "!enable must hold iq*=0"


@cocotb.test()
async def proportional_sign_and_clamp(dut):
    cocotb.start_soon(Clock(dut.clk, 10, units="ns").start())
    await _reset(dut)
    dut.enable.value = 1
    await RisingEdge(dut.clk)

    # small positive error: iq* = KP*err = 4*50 = 200 (within clamp), pipelined.
    changed_at, iq = await _sample(dut, 50, 0)
    assert changed_at is not None and changed_at >= 2, (
        f"iq* changed too soon ({changed_at}); not pipelined")
    assert iq == 200, f"KP*err expected 200, got {iq}"

    # overspeed -> braking (negative iq*).
    await _reset(dut)
    dut.enable.value = 1
    await RisingEdge(dut.clk)
    _, iq = await _sample(dut, 0, 50)
    assert iq == -200, f"braking expected -200, got {iq}"

    # large error saturates to +IQ_MAX / -IQ_MAX.
    await _reset(dut)
    dut.enable.value = 1
    await RisingEdge(dut.clk)
    _, iq = await _sample(dut, 1000, 0)
    assert iq == IQ_MAX, f"expected +clamp {IQ_MAX}, got {iq}"
    _, iq = await _sample(dut, 0, 1000)
    assert iq == -IQ_MAX, f"expected -clamp {-IQ_MAX}, got {iq}"


@cocotb.test()
async def integrator_winds_up(dut):
    cocotb.start_soon(Clock(dut.clk, 10, units="ns").start())
    await _reset(dut)
    dut.enable.value = 1
    await RisingEdge(dut.clk)
    # repeated small error: the integral term grows the command past KP*err.
    _, iq1 = await _sample(dut, 20, 0)        # ~KP*20 = 80
    for _ in range(6):
        await _sample(dut, 20, 0)
    _, iqN = await _sample(dut, 20, 0)
    assert iqN > iq1, f"integrator should grow the command ({iqN} !> {iq1})"
    assert iqN <= IQ_MAX
