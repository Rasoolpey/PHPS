# genrou_board.py: written by PHPS.RT.genrou_rt_board_script (src/rt/machines/genrou.jl); do not edit.
# Runs the GENROU IP on the PYNQ-Z2. Copy next to it: genrou.bit and genrou.hwh (same base name), from the
# Vivado project (system_wrapper.bit and system.hwh). Run as root on the board, or in a Jupyter notebook.
#   python3 genrou_board.py                 loads the bitstream (PYNQ also sets the 125 MHz clock), then runs
#   python3 genrou_board.py --no-download   keeps the bitstream already loaded (for the ILA) and
#                                           sets the 125 MHz clock itself, then runs
import sys, time
from pynq import Overlay, Clocks

H = 9.44e-7                     # step size [s]: one step every 118 cycles at 125 MHz
COEFS = [(-1.1799999303800041e-7, 58), (0.0, 58), (0.0, 58), (0.0, 58), (3.1466169742396164e-5, 58), (-3.1466171598900176e-5, 58), (0.0, 58), (0.0, 58), (0.0, 58), (0.0, 58), (-2.3599972152032862e-6, 58), (0.0, 58), (0.0, 58), (0.0, 58), (-1.8879799496319043e-5, 58), (-1.887982177448245e-5, 58), (1.1799999303800041e-7, 58), (-1.769999895570006e-7, 58), (0.0, 58), (1.8565040148013735e-12, 58), (-4.719928524591049e-6, 58), (0.0, 58), (0.0, 58), (0.0, 58), (2.713996797483779e-6, 58), (0.0, 58), (0.0, 58), (-7.5519543296808964e-6, 58), (0.6666666666666666, 40), (0.25, 40), (-0.0, 40), (-4.0, 40), (4.0, 40), (-0.0, 40), (1.0, 40), (1.5733333333333334e-7, 58), (0.0003558796157986518, 58)]
X0 = [0.7928054353286134, 1.0, 0.9149186962695078, 0.8083841972878167, 0.4818669570459969, -0.6494728551489524, 0.7018496983061259, 0.7123250669375606]
V_RE, V_IM, TM, EFD = 1.0, 0.0, 0.7999999999999999, 1.9802636860864182
VDIP, N1, N_ON, N_OFF, DECIM = 0.5, 317797, 105932, 211864, 32
X_JULIA = [0.838369649414165, 0.9975072132158402, 0.8913428745686294, 0.7716305854222995, 0.4598202581731773, -0.6675105705239934]   # final state of the dip case, Julia @hls step (Float64)
NAMES = ["delta", "omega", "Eq'", "psi_d", "Ed'", "psi_q"]

DOWNLOAD = "--no-download" not in sys.argv
ol = Overlay("genrou.bit", download=DOWNLOAD)
if abs(Clocks.fclk0_mhz - 125.0) > 0.01:   # set the IP's clock (PL clock 0) only when
    Clocks.fclk0_mhz = 125.0              # needed: reprogramming it can glitch the clock
                                                      # and stall the ILA, which runs on the same clock
print("bitstream", "loaded by PYNQ" if DOWNLOAD else "kept (already loaded)", "| FCLK0 = %.3f MHz" % Clocks.fclk0_mhz)
ip = ol.genrou_run_0                  # the IP's AXI-Lite registers (offsets from the packaged IP's driver)

def raw(v, F):                        # a value as the raw bits of ap_fixed with F fraction bits (48 bits)
    return int(round(v * 2**F)) & ((1 << 48) - 1)
def w48(addr, v, F):                  # one 48-bit value in a 64-bit slot: low 32 bits, then the high 16
    r = raw(v, F)
    ip.write(addr, r & 0xFFFFFFFF)
    ip.write(addr + 4, r >> 32)
def r48(addr, F=40):
    r = ip.read(addr) | (ip.read(addr + 4) & 0xFFFF) << 32
    if r >> 47:
        r -= 1 << 48
    return r / 2**F

for i, (v, F) in enumerate(COEFS):    # k: 37 words from 0x200
    w48(0x200 + 8 * i, v, F)

def run(nsteps, vdip, n_on, n_off):
    for i, v in enumerate(X0):        # x: 8 words from 0x40 (initial state; final state after the run)
        w48(0x40 + 8 * i, v, 40)
    w48(0x18, V_RE, 40); w48(0x24, V_IM, 40); w48(0x30, TM, 40); w48(0x80, EFD, 40); w48(0x8C, vdip, 40)
    ip.write(0x98, n_on); ip.write(0xA0, n_off); ip.write(0xA8, nsteps); ip.write(0xB0, DECIM)
    t0 = time.perf_counter()
    ip.write(0x00, 1)                 # ap_start
    while not (ip.read(0x00) & 0x2):  # ap_done
        time.sleep(0.001)
    t = time.perf_counter() - t0
    return t, [r48(0x40 + 8 * i) for i in range(8)]

print("run 1: voltage dip to", VDIP, "pu from 0.1 s to 0.2 s,", N1, "steps (0.3 s of simulated time)")
t, x = run(N1, VDIP, N_ON, N_OFF)
print("  wall-clock time %.4f s (simulated %.4f s)" % (t, N1 * H))
print("  %-6s %16s %16s %12s" % ("state", "board", "Julia", "difference"))
for i in range(6):
    print("  %-6s %16.10f %16.10f %12.3e" % (NAMES[i], x[i], X_JULIA[i], x[i] - X_JULIA[i]))

N2 = 10_000_000
print("run 2: operating point,", N2, "steps (%.3f s of simulated time)" % (N2 * H))
t, x = run(N2, 1.0, 0, 0)
print("  wall-clock time %.4f s, %.4f us per step (h = %.4f us)" % (t, t / N2 * 1e6, H * 1e6))
print("  max |x - x0| over the 6 states: %.3e (the operating point must hold)" % max(abs(x[i] - X0[i]) for i in range(6)))
