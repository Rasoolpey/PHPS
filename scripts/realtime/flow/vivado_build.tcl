# Vivado batch script of the automated flow (stage 3): the block design of chapter 1, step 3, built by commands.
#   vivado -mode batch -source vivado_build.tcl -tclargs <dir> <ip_repo> <vlnv> <top> <fclk_mhz> <probes>
# <probes>: "port:width port:width ..." (the IP's outputs recorded by the ILA, in probe order).
# Writes <dir>/<model>.bit, .hwh, .ltx and prints FLOW: lines (timing, utilization) for flow.sh.
lassign $argv outdir iprepo vlnv top fclk probes
set model [lindex [split $top _] 0]
set proj $outdir/proj

# project for the PYNQ-Z2, with the packaged IP in the catalog
create_project -force ${model}_flow $proj -part xc7z020clg400-1
set_property board_part tul.com.tw:pynq-z2:part0:1.0 [current_project]
set_property ip_repo_paths $iprepo [current_project]
update_ip_catalog

# block design: the processor (board preset), its 125 MHz clock, the IP on AXI-Lite, an ILA on the IP's outputs
create_bd_design system
create_bd_cell -type ip -vlnv xilinx.com:ip:processing_system7:5.5 processing_system7_0
apply_bd_automation -rule xilinx.com:bd_rule:processing_system7 \
    -config {make_external "FIXED_IO, DDR" apply_board_preset "1" Master "Disable" Slave "Disable"} \
    [get_bd_cells processing_system7_0]
set_property CONFIG.PCW_FPGA0_PERIPHERAL_FREQMHZ $fclk [get_bd_cells processing_system7_0]
create_bd_cell -type ip -vlnv $vlnv ${top}_0
apply_bd_automation -rule xilinx.com:bd_rule:axi4 \
    -config [list Clk_master {Auto} Clk_slave {Auto} Clk_xbar {Auto} Master {/processing_system7_0/M_AXI_GP0} \
                  Slave "/${top}_0/s_axi_control" ddr_seg {Auto} intc_ip {New AXI SmartConnect} master_apm {0}] \
    [get_bd_intf_pins ${top}_0/s_axi_control]
create_bd_cell -type ip -vlnv xilinx.com:ip:ila:6.2 ila_0
set cfg [list CONFIG.C_NUM_OF_PROBES [llength $probes] CONFIG.C_DATA_DEPTH 16384 CONFIG.C_EN_STRG_QUAL 1 \
              CONFIG.C_MONITOR_TYPE Native]
set i 0
foreach p $probes {
    lassign [split $p :] port width
    lappend cfg CONFIG.C_PROBE${i}_WIDTH $width
    incr i
}
set_property -dict $cfg [get_bd_cells ila_0]
set i 0
foreach p $probes {
    lassign [split $p :] port width
    connect_bd_net [get_bd_pins ${top}_0/$port] [get_bd_pins ila_0/probe$i]
    incr i
}
connect_bd_net [get_bd_pins processing_system7_0/FCLK_CLK0] [get_bd_pins ila_0/clk]
validate_bd_design
save_bd_design

# wrapper, synthesis, implementation, bitstream
set bd [get_files system.bd]
make_wrapper -files $bd -top
add_files -norecurse [file join [file dirname [get_property DIRECTORY [current_project]]] proj ${model}_flow.gen sources_1 bd system hdl system_wrapper.v]
set_property top system_wrapper [current_fileset]
if {[info exists ::env(IMPL_STRATEGY)] && $::env(IMPL_STRATEGY) ne ""} {   ;# a model's own strategy (flow.cfg)
    set_property strategy $::env(IMPL_STRATEGY) [get_runs impl_1]
    puts "FLOW: implementation strategy $::env(IMPL_STRATEGY)"
}
launch_runs impl_1 -to_step write_bitstream -jobs 12
wait_on_run impl_1
if {[get_property PROGRESS [get_runs impl_1]] != "100%"} { error "implementation or bitstream failed" }

# results
open_run impl_1
set wns [get_property SLACK [get_timing_paths -max_paths 1 -nworst 1 -setup]]
set whs [get_property SLACK [get_timing_paths -max_paths 1 -nworst 1 -hold]]
puts "FLOW: timing after routing at $fclk MHz: WNS $wns ns, WHS $whs ns"
report_utilization -file $outdir/utilization.rpt
foreach k {"Slice LUTs" "Slice Registers" "Block RAM Tile" "DSPs"} {
    set fp [open $outdir/utilization.rpt]
    while {[gets $fp line] >= 0} {
        if {[string match "| $k *" $line]} { puts "FLOW: [string trim $line]"; break }
    }
    close $fp
}
set impl [get_property DIRECTORY [get_runs impl_1]]
file copy -force $impl/system_wrapper.bit $outdir/$model.bit
file copy -force $impl/system_wrapper.ltx $outdir/$model.ltx
file copy -force [lindex [glob $proj/${model}_flow.gen/sources_1/bd/system/hw_handoff/system.hwh] 0] $outdir/$model.hwh
puts "FLOW: bitstream $outdir/$model.bit, hwh $outdir/$model.hwh, probes $outdir/$model.ltx"
