# SPDX-License-Identifier: MIT
"""cocotb block test for pwm_generator (robotics-ip-checklist stage 4).

Drives the block standalone (no co-sim, no Python plant) and checks the
block-level safety the formal proof also guarantees, dynamically: never both
gates of a leg, gates actually toggle, dead-time is respected on every handoff,
and `kill` forces everything off. Run via sim/cocotb/test_cocotb_blocks.py.
"""

import cocotb
from cocotb.clock import Clock
from cocotb.triggers import RisingEdge

DEAD = 25       # default DEAD_CYCLES


async def _reset(dut):
    dut.rst_n.value = 0
    dut.kill.value = 0
    dut.leg_mode.value = 0
    dut.duty3.value = 0
    for _ in range(4):
        await RisingEdge(dut.clk)
    dut.rst_n.value = 1
    await RisingEdge(dut.clk)


@cocotb.test()
async def shoot_through_free_and_deadtime(dut):
    """No leg drives both gates; complementary handoffs honor the dead-time."""
    cocotb.start_soon(Clock(dut.clk, 10, units="ns").start())
    await _reset(dut)

    # All three legs in PWM at ~48% duty so every leg actually chops.
    duty = 300
    dut.duty3.value = (duty << 32) | (duty << 16) | duty
    dut.leg_mode.value = 0b010101  # PWM on all three legs

    last_gh = [0, 0, 0]
    last_gl = [0, 0, 0]
    gl_off_for = [10**6] * 3
    gh_off_for = [10**6] * 3
    toggles = 0

    for _ in range(4000):
        await RisingEdge(dut.clk)
        gh = int(dut.gate_high.value)
        gl = int(dut.gate_low.value)
        for k in range(3):
            h, l = (gh >> k) & 1, (gl >> k) & 1
            # Safety: never both.
            assert not (h and l), f"shoot-through on leg {k}"
            # Dead-time: a gate may rise only after its complement was off >= DEAD.
            if h and not last_gh[k]:
                assert gl_off_for[k] >= DEAD, (
                    f"leg {k} high rose with low off only {gl_off_for[k]} < {DEAD}")
            if l and not last_gl[k]:
                assert gh_off_for[k] >= DEAD, (
                    f"leg {k} low rose with high off only {gh_off_for[k]} < {DEAD}")
            gl_off_for[k] = 0 if l else gl_off_for[k] + 1
            gh_off_for[k] = 0 if h else gh_off_for[k] + 1
            if h != last_gh[k]:
                toggles += 1
            last_gh[k], last_gl[k] = h, l

    assert toggles > 10, "gates never toggled - vacuous test"


@cocotb.test()
async def kill_forces_gates_off(dut):
    """While kill is asserted, every gate is off."""
    cocotb.start_soon(Clock(dut.clk, 10, units="ns").start())
    await _reset(dut)
    dut.duty3.value = (300 << 32) | (300 << 16) | 300
    dut.leg_mode.value = 0b010101
    dut.kill.value = 1
    for _ in range(200):
        await RisingEdge(dut.clk)
        assert int(dut.gate_high.value) == 0 and int(dut.gate_low.value) == 0, \
            "gate on while killed"
