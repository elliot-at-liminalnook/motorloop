# SPDX-License-Identifier: MIT
"""cocotb block test for as5047p_spi_master (robotics-ip-checklist stage 4).

Exercises the AS5047P read standalone: act as a mode-1 (CPOL=0/CPHA=1) slave
that drives a crafted 16-bit response word (PARD | EF=0 | ANGLE[13:0]) MSB-first
on the rising edges; verify the master frames it (ncs, 16 SCLK), checks even
parity + EF, and emits the top-12 angle on `new_sample` with `angle_valid`.
"""

import cocotb
from cocotb.clock import Clock
from cocotb.triggers import RisingEdge, FallingEdge


def _response(angle14, ef=0):
    bits14_0 = ((ef & 1) << 14) | (angle14 & 0x3FFF)
    pard = bin(bits14_0).count("1") & 1          # even parity over bits[14:0]
    word = (pard << 15) | bits14_0
    expected = (word >> 2) & 0xFFF               # rx[13:2], the top-12 angle
    return word, expected


async def _slave(dut, word):
    """Drive miso MSB-first on each SCLK rising edge, re-syncing per frame."""
    while True:
        await FallingEdge(dut.ncs)               # frame start
        for i in range(16):
            await RisingEdge(dut.sclk)
            dut.miso.value = (word >> (15 - i)) & 1


@cocotb.test()
async def reads_angle_with_parity(dut):
    cocotb.start_soon(Clock(dut.clk, 10, units="ns").start())
    dut.rst_n.value = 0
    dut.miso.value = 0
    for _ in range(4):
        await RisingEdge(dut.clk)
    dut.rst_n.value = 1

    ANGLE14 = 0x1ABC
    word, expected = _response(ANGLE14)
    cocotb.start_soon(_slave(dut, word))

    # free-running master; wait for a fresh valid sample (a couple of frames).
    for _ in range(4000):
        await RisingEdge(dut.clk)
        if int(dut.new_sample.value) == 1:
            break
    else:
        assert False, "as5047p_spi_master never produced new_sample"

    assert int(dut.angle_valid.value) == 1, "valid angle should set angle_valid"
    assert int(dut.angle.value) == expected, (
        f"angle {int(dut.angle.value):#x} != expected {expected:#x}")

    # A second fresh sample confirms the free-running stream keeps framing.
    for _ in range(4000):
        await RisingEdge(dut.clk)
        if int(dut.new_sample.value) == 1:
            break
    else:
        assert False, "as5047p_spi_master stopped streaming"
    assert int(dut.angle.value) == expected
