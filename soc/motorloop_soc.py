#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
"""Reference SoC: a RISC-V (VexRiscv) LiteX SoC that drives the motorloop
controller over AXI-Lite (tier2-adoption-checklist §2). The controller wrapper
`rtl/soc/motorloop_axil_top.v` is added as an AXI-Lite bus slave; firmware
(`soc/firmware/`) writes the control registers to spin the motor and reads the
telemetry registers. Board: Radiona ULX3S (Lattice ECP5 LFE5U-85F) - the same
device the open synth flow targets (synth/), fully open toolchain
(yosys + nextpnr-ecp5 + ecppack via LiteX's "trellis").

  python3 soc/motorloop_soc.py --build            # gateware -> build/
  python3 soc/motorloop_soc.py --build --load      # + flash to a connected ULX3S
  python3 soc/motorloop_soc.py --sys-clk-freq 50e6  # default 50 MHz (< 64 MHz Fmax)

The register map is axil_regfile's (see rtl/contracts/axil_regfile.md): byte
addr = index*4 from the peripheral base (CSR region "motor").
"""

from migen import Module, Instance, Signal, ClockSignal, ResetSignal, ClockDomain

from litex.build.generic_platform import Pins, Subsignal, IOStandard
from litex_boards.platforms import radiona_ulx3s
from litex_boards.targets.radiona_ulx3s import _CRG
from litex.soc.integration.soc_core import SoCCore
from litex.soc.integration.soc import SoCRegion
from litex.soc.integration.builder import Builder, builder_args, builder_argdict
from litex.soc.interconnect import axi
from litex.build.sim import SimPlatform
from litex.build.sim.config import SimConfig

# motorloop controller pins on a ULX3S GPIO header (gp/gn). Adjust to your wiring;
# these are documented in soc/README.md. The controller is the DRV8301 BOM
# (DRV8301 SPI gate driver, MCP3208 ADC, AS5600 PWM angle).
_motor_io = [
    ("motor", 0,
        Subsignal("inh",       Pins("gp0 gp1 gp2")),
        Subsignal("inl",       Pins("gn0 gn1 gn2")),
        Subsignal("en_gate",   Pins("gp3")),
        Subsignal("dc_cal",    Pins("gn3")),
        Subsignal("drv_sclk",  Pins("gp4")),
        Subsignal("drv_mosi",  Pins("gp5")),
        Subsignal("drv_ncs",   Pins("gp6")),
        Subsignal("drv_miso",  Pins("gn4")),
        Subsignal("nfault",    Pins("gn5")),
        Subsignal("noctw",     Pins("gn6")),
        Subsignal("adc_sclk",  Pins("gp7")),
        Subsignal("adc_mosi",  Pins("gp8")),
        Subsignal("adc_ncs",   Pins("gp9")),
        Subsignal("adc_miso",  Pins("gn7")),
        Subsignal("angle_pwm", Pins("gn8")),
        IOStandard("LVCMOS33")),
]

# RTL: the wrapper + the AXI-Lite slave + the whole controller_top hierarchy.
_RTL = [
    "rtl/soc/motorloop_axil_top.v", "rtl/bus/axil_regfile.v",
    "rtl/controller_top.v", "rtl/pwm_generator.v", "rtl/commutation.v",
    "rtl/spi_drv_master.v", "rtl/adc_spi_master.v", "rtl/divider32.v",
    "rtl/as5600_pwm_capture.v", "rtl/as5047p_spi_master.v", "rtl/ads9224r_master.v",
    "rtl/speed_meter.v", "rtl/speed_pi.v", "rtl/open_loop_ramp.v",
    "rtl/drv_manager.v", "rtl/adc_sequencer.v", "rtl/sincos.v", "rtl/clarke.v",
    "rtl/park.v", "rtl/inv_park.v", "rtl/svpwm.v", "rtl/svpwm_seq.v",
    "rtl/current_pi.v", "rtl/circle_limit.v", "rtl/circle_limit_seq.v",
    "rtl/foc_core.v", "rtl/speed_iq_pi.v", "rtl/uart_regfile.v", "rtl/uart_rx.v",
    "rtl/uart_tx.v",
]


class MotorController(Module):
    """motorloop_axil_top as an AXI4-Lite slave peripheral."""
    def __init__(self, platform, pads, repo_root):
        # Full SoC-width AXI-Lite (the LiteX wishbone<->axilite bridge requires
        # address_width == bus width); the wrapper only needs the low 8 bits
        # (256-byte register region), so slice the address to s_*addr[7:0].
        self.bus = axi.AXILiteInterface(data_width=32, address_width=32)
        b = self.bus
        self.specials += Instance("motorloop_axil_top",
            p_ADDR_W=8,
            i_clk=ClockSignal("sys"), i_rst_n=~ResetSignal("sys"),
            # AXI4-Lite slave <- the LiteX bus (address sliced to the region)
            i_s_awaddr=b.aw.addr[:8], i_s_awvalid=b.aw.valid, o_s_awready=b.aw.ready,
            i_s_wdata=b.w.data,    i_s_wstrb=b.w.strb,     i_s_wvalid=b.w.valid,
            o_s_wready=b.w.ready,  o_s_bresp=b.b.resp,     o_s_bvalid=b.b.valid,
            i_s_bready=b.b.ready,  i_s_araddr=b.ar.addr[:8], i_s_arvalid=b.ar.valid,
            o_s_arready=b.ar.ready, o_s_rdata=b.r.data,    o_s_rresp=b.r.resp,
            o_s_rvalid=b.r.valid,  i_s_rready=b.r.ready,
            # peripheral pins -> board
            o_inh=pads.inh, o_inl=pads.inl, o_en_gate=pads.en_gate,
            o_dc_cal=pads.dc_cal, o_drv_sclk=pads.drv_sclk, o_drv_mosi=pads.drv_mosi,
            o_drv_ncs=pads.drv_ncs, i_drv_miso=pads.drv_miso, i_nfault=pads.nfault,
            i_noctw=pads.noctw, o_adc_sclk=pads.adc_sclk, o_adc_mosi=pads.adc_mosi,
            o_adc_ncs=pads.adc_ncs, i_adc_miso=pads.adc_miso, i_angle_pwm=pads.angle_pwm)
        for f in _RTL:
            platform.add_source(str(repo_root / f))
        platform.add_verilog_include_path(str(repo_root / "rtl"))
        platform.add_verilog_include_path(str(repo_root / "rtl" / "gen"))


class MotorloopSoC(SoCCore):
    def __init__(self, sys_clk_freq=50e6, **kwargs):
        from pathlib import Path
        repo_root = Path(__file__).resolve().parent.parent
        platform = radiona_ulx3s.Platform(device="LFE5U-85F", toolchain="trellis")
        platform.add_extension(_motor_io)
        SoCCore.__init__(self, platform, int(sys_clk_freq),
                         ident="motorloop reference SoC", cpu_type="vexriscv",
                         integrated_rom_size=0x8000,        # BIOS ROM (block RAM)
                         integrated_main_ram_size=0x8000,   # firmware RAM (no SDRAM)
                         **kwargs)
        self.submodules.crg = _CRG(platform, int(sys_clk_freq))
        pads = platform.request("motor")
        self.submodules.motor = MotorController(platform, pads, repo_root)
        # map the controller registers into the CPU's address space
        self.bus.add_slave("motor", self.motor.bus,
                           SoCRegion(origin=0xb0000000, size=0x100,
                                     cached=False))


# --- Verilator sim variant (litex_sim): RISC-V boots over a sim serial with the
# controller mapped as a peripheral; no board pins needed. ------------------------
_sim_io = [
    ("sys_clk", 0, Pins(1)),
    ("sys_rst", 0, Pins(1)),
    ("serial", 0,
        Subsignal("source_valid", Pins(1)), Subsignal("source_ready", Pins(1)),
        Subsignal("source_data",  Pins(8)),
        Subsignal("sink_valid",   Pins(1)), Subsignal("sink_ready",   Pins(1)),
        Subsignal("sink_data",    Pins(8))),
]


class _SimCRG(Module):
    def __init__(self, clk):
        self.clock_domains.cd_sys = ClockDomain()
        self.comb += self.cd_sys.clk.eq(clk)


class _SimPads:
    """Controller peripheral pins as plain signals (no motor in sim)."""
    def __init__(self):
        for n, w in (("inh", 3), ("inl", 3), ("en_gate", 1), ("dc_cal", 1),
                     ("drv_sclk", 1), ("drv_mosi", 1), ("drv_ncs", 1),
                     ("drv_miso", 1), ("nfault", 1), ("noctw", 1),
                     ("adc_sclk", 1), ("adc_mosi", 1), ("adc_ncs", 1),
                     ("adc_miso", 1), ("angle_pwm", 1)):
            setattr(self, n, Signal(w))


class MotorloopSimSoC(SoCCore):
    def __init__(self, repo_root):
        platform = SimPlatform("SIM", _sim_io)
        SoCCore.__init__(self, platform, clk_freq=int(1e6), cpu_type="vexriscv",
                         uart_name="sim", integrated_rom_size=0x10000,
                         integrated_main_ram_size=0x10000,
                         ident="motorloop sim SoC")
        self.submodules.crg = _SimCRG(platform.request("sys_clk"))
        pads = _SimPads()
        self.comb += [pads.nfault.eq(1), pads.noctw.eq(1)]   # no fault -> bring-up
        self.submodules.motor = MotorController(platform, pads, repo_root)
        self.bus.add_slave("motor", self.motor.bus,
                           SoCRegion(origin=0xb0000000, size=0x100, cached=False))


def main():
    import argparse
    from pathlib import Path
    repo_root = Path(__file__).resolve().parent.parent
    parser = argparse.ArgumentParser(description="motorloop reference SoC (ULX3S)")
    parser.add_argument("--build", action="store_true", help="build the gateware")
    parser.add_argument("--load", action="store_true", help="load to the ULX3S")
    parser.add_argument("--sim", action="store_true", help="run in litex_sim (Verilator)")
    parser.add_argument("--sys-clk-freq", default=50e6, type=float)
    builder_args(parser)
    args = parser.parse_args()
    if args.sim:
        soc = MotorloopSimSoC(repo_root)
        builder = Builder(soc, **builder_argdict(args))
        sim_config = SimConfig()
        sim_config.add_clocker("sys_clk", freq_hz=int(1e6))
        sim_config.add_module("serial2console", "serial")
        builder.build(sim_config=sim_config, run=True)
        return
    soc = MotorloopSoC(sys_clk_freq=args.sys_clk_freq)
    builder = Builder(soc, **builder_argdict(args))
    builder.build(run=args.build)
    if args.load:
        prog = soc.platform.create_programmer()
        prog.load_bitstream(builder.get_bitstream_filename(mode="sram"))


if __name__ == "__main__":
    main()
