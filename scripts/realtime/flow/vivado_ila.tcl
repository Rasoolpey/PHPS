# Vivado batch script of the automated flow (stage 4): attach to the ILA of the bitstream PYNQ loaded, arm it, run the
# board program, capture, and write the capture as CSV. Chapter 1, steps 4 and 5.1, by commands.
#   vivado -mode batch -source vivado_ila.tcl -tclargs <bit> <ltx> <sample_probe> <csv> <board script>
lassign $argv bit ltx sample csv board_script

open_hw_manager
connect_hw_server -allow_non_jtag
open_hw_target
set dev [lindex [get_hw_devices xc7z020*] 0]
current_hw_device $dev
# the bitstream is already loaded by PYNQ (flow.sh, stage 4): only the probe names are given, then the device is read
# (JTAG programming of a Zynq running Linux can fail with "End of startup status: LOW")
set_property PROBES.FILE $ltx $dev
set_property FULL_PROBES.FILE $ltx $dev
refresh_hw_device $dev
set ila [lindex [get_hw_ilas -of_objects $dev] 0]

# record only on the rising edge of the sample probe, starting at its first rising edge (chapter 1, step 5.1)
set probe [get_hw_probes -of_objects $ila *${sample}]
set_property CONTROL.CAPTURE_MODE BASIC $ila
set_property CONTROL.DATA_DEPTH 16384 $ila
set_property CONTROL.TRIGGER_POSITION 0 $ila
set_property TRIGGER_COMPARE_VALUE eq1'bR $probe
set_property CAPTURE_COMPARE_VALUE eq1'bR $probe
foreach p [get_hw_probes -of_objects $ila] { catch { set_property DISPLAY_RADIX SIGNED $p } }
run_hw_ila $ila
puts "FLOW: ILA armed (trigger and capture on the rising edge of $sample)"

# run the board program (it waits for the IP to finish its runs)
set out [exec bash $board_script 2>@1]
foreach line [split $out "\n"] { puts "BOARD: $line" }

wait_on_hw_ila -timeout 1 $ila
set data [upload_hw_ila_data $ila]
write_hw_ila_data -force -csv_file $csv $data
puts "FLOW: ILA capture written to $csv"
close_hw_target
