# SPDX-License-Identifier: MIT
"""cocotb block test for spi_drv_master (robotics-ip-checklist stage 4).

Exercises the SPI protocol standalone: act as a mode-1 slave, verify the master
clocks one 16-bit frame MSB-first on mosi (== tx), returns the slave's response
on rx, and frames it with ncs + a single done pulse.
"""

import cocotb
from cocotb.clock import Clock
from cocotb.triggers import RisingEdge


@cocotb.test()
async def one_frame(dut):
    cocotb.start_soon(Clock(dut.clk, 10, units="ns").start())
    dut.rst_n.value = 0
    dut.start.value = 0
    dut.tx.value = 0
    dut.miso.value = 0
    for _ in range(4):
        await RisingEdge(dut.clk)
    dut.rst_n.value = 1
    await RisingEdge(dut.clk)

    TX = 0xA53C      # what the master should clock out on mosi
    RESP = 0x6F19    # what the slave returns on miso

    dut.tx.value = TX
    dut.start.value = 1
    await RisingEdge(dut.clk)
    dut.start.value = 0

    # Wait for the frame to open (ncs low).
    while int(dut.ncs.value) == 1:
        await RisingEdge(dut.clk)

    # Mode 1 (CPOL=0/CPHA=1): the master launches mosi on the rising edge and
    # samples miso at the trailing edge; the slave presents its bit on the
    # rising edge. So on each rising edge: capture mosi, drive the next miso bit.
    captured = 0
    for i in range(16):
        await RisingEdge(dut.sclk)
        captured = (captured << 1) | int(dut.mosi.value)
        dut.miso.value = (RESP >> (15 - i)) & 1

    while int(dut.done.value) == 0:
        await RisingEdge(dut.clk)

    assert captured == TX, f"master sent {captured:#06x}, expected tx {TX:#06x}"
    assert int(dut.rx.value) == RESP, \
        f"master rx {int(dut.rx.value):#06x}, expected {RESP:#06x}"
    assert int(dut.ncs.value) == 1, "ncs should return high after the frame"
