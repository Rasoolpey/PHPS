# Vivado batch script: measure the step interval on the board directly (review 34). Attaches to the ILA of the
# bitstream PYNQ loaded, records EVERY clock cycle (no capture qualifier) from the first rising edge of the step strobe,
# runs the board program, writes the capture as CSV. strobe_interval.py then reads the cycles between strobe edges.
#   vivado -mode batch -source vivado_ila_every_clock.tcl -tclargs <ltx> <strobe probe> <csv> <board script>
lassign $argv ltx strobe csv board_script

open_hw_manager
connect_hw_server -allow_non_jtag
open_hw_target
set dev [lindex [get_hw_devices xc7z020*] 0]
current_hw_device $dev
set_property PROBES.FILE $ltx $dev
set_property FULL_PROBES.FILE $ltx $dev
refresh_hw_device $dev
set ila [lindex [get_hw_ilas -of_objects $dev] 0]

set probe [get_hw_probes -of_objects $ila *${strobe}]
set_property CONTROL.CAPTURE_MODE ALWAYS $ila
set_property CONTROL.DATA_DEPTH 16384 $ila
set_property CONTROL.TRIGGER_POSITION 0 $ila
set_property TRIGGER_COMPARE_VALUE eq1'bR $probe
run_hw_ila $ila
puts "FLOW: ILA armed (every clock, trigger on the rising edge of $strobe)"

set out [exec bash $board_script 2>@1]
foreach line [split $out "\n"] { puts "BOARD: $line" }

wait_on_hw_ila -timeout 1 $ila
set data [upload_hw_ila_data $ila]
write_hw_ila_data -force -csv_file $csv $data
puts "FLOW: every-clock ILA capture written to $csv"
close_hw_target
