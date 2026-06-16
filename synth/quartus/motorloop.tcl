# SPDX-License-Identifier: MIT
# Intel Quartus Prime synthesis + fit of controller_top for authoritative Intel
# resource + Fmax numbers (release-and-portability-checklist §4.3). Proprietary
# tool, NOT in CI - run where Quartus is licensed:
#
#   quartus_sh -t synth/quartus/motorloop.tcl       # map + fit + timing
#
# Writes the fit/timing reports under synth/quartus/db/output_files/. Default a
# Cyclone 10 LP part; edit FAMILY/DEVICE for your board.
load_package flow

set FAMILY "Cyclone 10 LP"
set DEVICE "10CL025YU256C8G"

project_new motorloop -overwrite -directory synth/quartus/db
set_global_assignment -name FAMILY $FAMILY
set_global_assignment -name DEVICE $DEVICE
set_global_assignment -name TOP_LEVEL_ENTITY controller_top
set_global_assignment -name SEARCH_PATH rtl
set_global_assignment -name SEARCH_PATH rtl/gen

# Same RTL set as the open flows (controller_top = vendor-neutral top).
foreach f [glob -nocomplain rtl/*.v rtl/bus/*.v] {
    if {![string match *foc_math* $f]} {
        set_global_assignment -name VERILOG_FILE $f
    }
}
set_global_assignment -name SDC_FILE synth/quartus/motorloop.sdc

execute_module -tool map
execute_module -tool fit
execute_module -tool sta
project_close
puts "QUARTUS: see synth/quartus/db/output_files/ for fit + TimeQuest Fmax"
