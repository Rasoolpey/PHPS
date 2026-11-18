"""Load a bitstream on the PYNQ-Z2 (automated flow, stage 4): PYNQ programs the FPGA and sets the PL clocks from the
.hwh beside the .bit.   python3 load_overlay.py <model>.bit"""
import sys
from pynq import Overlay, Clocks

Overlay(sys.argv[1])
print("bitstream %s loaded, FCLK0 = %.3f MHz" % (sys.argv[1], Clocks.fclk0_mhz))
