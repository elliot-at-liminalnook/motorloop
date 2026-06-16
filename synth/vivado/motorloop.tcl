# SPDX-License-Identifier: MIT
# Vivado out-of-context synthesis of controller_top for authoritative Xilinx
# 7-series resource + timing numbers (release-and-portability-checklist §4.3).
# Proprietary tool, NOT in CI - run where Vivado is licensed:
#
#   vivado -mode batch -source synth/vivado/motorloop.tcl
#
# Writes synth/vivado/{utilization,timing}.rpt. Default part xc7a35t (Artix-7,
# e.g. Arty A7-35T); override with -tclargs <part>.
set part xc7a35tcsg324-1
if {$argc > 0} { set part [lindex $argv 0] }

# Same RTL set as the open flows; controller_top is the vendor-neutral top
# (board_top.v is ECP5-specific). foc_math.v is a sim harness, excluded.
foreach f [concat [glob -nocomplain rtl/*.v] [glob -nocomplain rtl/bus/*.v]] {
    if {![string match *foc_math* $f]} { read_verilog $f }
}

synth_design -top controller_top -part $part \
    -include_dirs {rtl rtl/gen} -mode out_of_context

# 25 MHz target (the FPGA bring-up clock); report what's achieved.
create_clock -name clk -period 40.0 [get_ports clk]

report_utilization      -file synth/vivado/utilization.rpt
report_timing_summary   -file synth/vivado/timing.rpt
puts "VIVADO: wrote synth/vivado/utilization.rpt + timing.rpt for part $part"
