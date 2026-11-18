# Dumps a routed design's placement for a device-view figure (plot_device.py), in batch mode (no GUI).
#   vivado -mode batch -source vivado_device_dump.tcl -tclargs <routed.dcp> <out_dir>
# Writes <out_dir>/sites.csv     every SLICE, DSP, RAMB and PS7 site of the part: name, type, tile, tile grid x/y,
#                                clock region (the tile grid is the device view's layout: one column per tile)
#        <out_dir>/cells.csv     every placed primitive cell: name, site, primitive group
#        <out_dir>/regions.csv   clock regions: name, top-left and bottom-right tile grid x/y
#        <out_dir>/summary.txt   part, utilisation and timing of the checkpoint
set dcp [lindex $argv 0]; set out [lindex $argv 1]
file mkdir $out
open_checkpoint $dcp

set f [open "$out/sites.csv" w]
puts $f "site,type,tile,gx,gy,region"
foreach s [get_sites -filter {SITE_TYPE =~ SLICE* || SITE_TYPE == DSP48E1 || SITE_TYPE =~ RAMB*E1 || SITE_TYPE == PS7}] {
    set t [get_tiles -of_objects $s]
    puts $f "$s,[get_property SITE_TYPE $s],$t,[get_property GRID_POINT_X $t],[get_property GRID_POINT_Y $t],[get_property CLOCK_REGION $s]"
}
close $f

set cells [get_cells -hierarchical -filter {IS_PRIMITIVE && LOC != ""}]
set f [open "$out/cells.csv" w]
puts $f "cell,site,group"
foreach c $cells l [get_property LOC $cells] g [get_property PRIMITIVE_GROUP $cells] {
    puts $f "$c,$l,$g"
}
close $f

set f [open "$out/regions.csv" w]
puts $f "region,x0,y0,x1,y1"
foreach r [get_clock_regions] {
    set a [get_tiles [get_property TOP_LEFT_TILE $r]]; set b [get_tiles [get_property BOTTOM_RIGHT_TILE $r]]
    puts $f "$r,[get_property GRID_POINT_X $a],[get_property GRID_POINT_Y $a],[get_property GRID_POINT_X $b],[get_property GRID_POINT_Y $b]"
}
close $f

set f [open "$out/summary.txt" w]
puts $f "checkpoint $dcp"
puts $f "part [get_property PART [current_design]]"
puts $f "placed primitive cells [llength $cells]"
puts $f "WNS [get_property SLACK [get_timing_paths -max_paths 1 -nworst 1 -setup]] ns"
close $f
report_utilization -hierarchical -hierarchical_depth 2 -file "$out/util_hier.rpt"
puts "device dump written to $out"
