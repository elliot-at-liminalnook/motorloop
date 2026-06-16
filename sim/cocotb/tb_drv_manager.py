# SPDX-License-Identifier: MIT
"""cocotb block test for drv_manager (robotics-ip-checklist stage 4).

Exercises the bring-up FSM standalone: a cocotb SPI responder echoes register
writes so the write/verify handshake completes, and the manager reaches
`configured` (RUN) with no spurious faults. A second test confirms the hw_mode
(DRV8302) path reaches RUN with no SPI at all. Deep timers are shrunk via build
parameters (see test_cocotb_blocks.py) - the parameterization payoff.
"""

import cocotb
from cocotb.clock import Clock
from cocotb.triggers import RisingEdge


async def spi_responder(dut):
    """Act as the SPI peripheral: store writes, echo {addr, data} on reads, so
    the manager's readback-verify passes."""
    regs = {}
    dut.spi_busy.value = 0
    dut.spi_done.value = 0
    dut.spi_rx.value = 0
    while True:
        await RisingEdge(dut.clk)
        if int(dut.spi_start.value) == 1:
            tx = int(dut.spi_tx.value)
            rw = (tx >> 15) & 1
            addr = (tx >> 11) & 0xF
            data = tx & 0x7FF
            dut.spi_busy.value = 1
            for _ in range(3):
                await RisingEdge(dut.clk)
            if rw == 0:
                regs[addr] = data
                resp = regs.get(0, 0)            # write returns status reg 1
            else:
                resp = (addr << 11) | regs.get(addr, 0)   # read echoes {addr,data}
            dut.spi_rx.value = resp
            dut.spi_done.value = 1
            dut.spi_busy.value = 0
            await RisingEdge(dut.clk)
            dut.spi_done.value = 0


async def _init(dut, hw_mode):
    cocotb.start_soon(Clock(dut.clk, 10, units="ns").start())
    dut.rst_n.value = 0
    dut.nfault_sync.value = 1
    dut.lockout_clear.value = 0
    dut.hw_mode.value = hw_mode
    dut.spi_busy.value = 0
    dut.spi_done.value = 0
    dut.spi_rx.value = 0
    for _ in range(4):
        await RisingEdge(dut.clk)
    dut.rst_n.value = 1


@cocotb.test()
async def reaches_run_via_spi_config(dut):
    await _init(dut, hw_mode=0)
    cocotb.start_soon(spi_responder(dut))
    for _ in range(5000):
        await RisingEdge(dut.clk)
        if int(dut.configured.value) == 1:
            break
    assert int(dut.configured.value) == 1, "never reached configured/RUN"
    assert int(dut.fault_count.value) == 0, "spurious faults during clean bring-up"


@cocotb.test()
async def hw_mode_skips_spi(dut):
    await _init(dut, hw_mode=1)
    saw_spi = 0
    reached = 0
    for _ in range(5000):
        await RisingEdge(dut.clk)
        if int(dut.spi_start.value) == 1:
            saw_spi = 1
        if int(dut.configured.value) == 1:
            reached = 1
            break
    assert reached, "hw_mode never reached configured"
    assert saw_spi == 0, "hw_mode issued SPI (should skip config)"
