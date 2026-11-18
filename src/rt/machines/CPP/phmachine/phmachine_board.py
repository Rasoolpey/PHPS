# phmachine_board.py: written by PHPS.RT.phmachine_rt_board_script (src/rt/machines/phmachine.jl); do not edit.
# Runs the PHMachine IP on the PYNQ-Z2. Next to it: phmachine.bit, phmachine.hwh and the IP's register map
# xphmachine_run_hw.h (from the HLS driver; the flow copies it). Run as root on the board.
#   python3 phmachine_board.py [--no-download]   (--no-download keeps the bitstream already loaded, for the ILA)
import sys, time, re, math
from pynq import Overlay, Clocks

H = 3.2e-7
COEFS = [(0.0, 58, 48), (0.0, 58, 48), (0.0, 58, 48), (1.2058021202354663e-7, 58, 48), (-3.0558347922926554e-7, 58, 48), (1.5726596685333716e-7, 58, 48), (2.2560395682852397e-6, 58, 48), (4.326569040057071e-6, 58, 48), (-7.101569771739147e-6, 58, 48), (0.0, 58, 48), (0.0, 58, 48), (0.0, 58, 48), (8.09059469907697e-6, 58, 48), (-1.4993087385107434e-5, 58, 48), (4.92879488346587e-6, 58, 48), (1.7211467740613218e-7, 58, 48), (4.4730145278004674e-7, 58, 48), (-6.614034467194996e-7, 58, 48), (0.0001206371578978474, 58, 48), (7.2732271594322384e-12, 58, 48), (1.3608110124851969e-10, 58, 48), (0.0001206371578978474, 58, 48), (4.880131750095718e-10, 58, 48), (1.0381712850504599e-11, 58, 48), (0.00011274500738116602, 58, 48), (6.79740985720656e-12, 58, 48), (1.2717859751432137e-10, 58, 48), (0.00011274500738116602, 58, 48), (4.56087077826961e-10, 58, 48), (9.702533915190159e-12, 58, 48), (0.0, 58, 48), (0.0001206371394654876, 58, 48), (2.609724970847882e-10, 58, 48), (15.384607889377548, 40, 48), (-8.546860566690157, 40, 48), (-5.81257271850609, 40, 48), (15.384567873417836, 40, 48), (-11.582331587815133, 40, 48), (-2.7150253687532313, 40, 48), (0.0008672688650221971, 50, 48), (0.0008672666092222059, 50, 48), (-0.0005155344838572162, 50, 48), (0.0018559553711474663, 50, 48), (0.0018559505437352186, 50, 48), (0.0009279776855732891, 50, 48), (0.0009279752718676093, 50, 48), (0.9345794392520475, 40, 48), (4.469273838262076e-9, 72, 48), (9.267902750859872e-21, 107, 48), (9.267878644688093e-21, 107, 48), (0.00018095573684677457, 57, 48), (0.0002412743157956948, 57, 48), (6.031857894892717e-5, 57, 48)]   # (value, fraction bits, width)
X0 = [0.549631435553124, -0.7756035822964407, 1.1319269525561708, 0.8642245141948024, -0.6235007806547799, -0.6235007806547799, 1.0, 0.6527195775334746, 0.7575995994617131, 0.549631435553124, -0.7756035822964407, 1.0, 0.9955230012330958, -0.09451959593576476]
VMAG, CM, SROT, VF, TM, VDIP = 1.0171505949811035, 7.276661922323013e-9, 0.00012063715760523772, 0.001191261648273212, 6.4999999995652615, 0.5
N_ON, N_OFF, DECIM, CUTS = 312500, 625000, 32, [312500, 468750, 937500]
X_JULIA = [[0.5496314355524329, -0.7756035822969186, 1.1319269526255598, 0.8642245142200541, -0.6235007806541149, -0.6235007806547799, 1.0, 0.6527195775335727, 0.7575995994616115], [0.5386304370563194, -0.7832834038147093, 1.1259027586312877, 0.8087806818580526, -0.41875432619824937, -0.6052598661777724, 1.0019355487926929, 0.6419623583281089, 0.7667361544167672], [0.41951332325012947, -0.8530323616941194, 1.1142967018575385, 0.8161818258876643, -0.6477547567176885, -0.5902101595762703, 1.001415906551777, 0.5241523757949751, 0.8516244987953805]]
NAMES = ["psi_d", "psi_q", "psi_fd", "psi_1d", "psi_1q", "psi_2q", "omega", "cos th^", "sin th^"]

A = {}                                 # register offsets from the HLS driver header
for m in re.finditer(r"#define XPHMACHINE_RUN_CONTROL_ADDR_(\w+?)_(DATA|BASE)\s+0x([0-9a-fA-F]+)", open("xphmachine_run_hw.h").read()):
    A[m.group(1)] = int(m.group(3), 16)

DOWNLOAD = "--no-download" not in sys.argv
ol = Overlay("phmachine.bit", download=DOWNLOAD)
if abs(Clocks.fclk0_mhz - 125.0) > 0.01:
    Clocks.fclk0_mhz = 125.0
print("bitstream", "loaded by PYNQ" if DOWNLOAD else "kept (already loaded)", "| FCLK0 = %.3f MHz" % Clocks.fclk0_mhz)
ip = ol.phmachine_run_0

def raw(v, F, W):
    return int(round(v * 2**F)) & ((1 << W) - 1)
def w48(addr, v, F=40, W=48):         # a W-bit value with F fraction bits in a 64-bit slot
    r = raw(v, F, W)
    ip.write(addr, r & 0xFFFFFFFF)
    ip.write(addr + 4, r >> 32)
def r48(addr, F=40):
    r = ip.read(addr) | (ip.read(addr + 4) & 0xFFFF) << 32
    if r >> 47:
        r -= 1 << 48
    return r / 2**F

for i, (v, F, W) in enumerate(COEFS):
    w48(A["K"] + 8 * i, v, F, W)
for name, v in (("VMAG", VMAG), ("VF", VF), ("TM", TM), ("VDIP", VDIP)):
    w48(A[name], v)
w48(A["CM"], CM, 58); w48(A["SROT"], SROT, 58)

def run(x, n0, nsteps, n_on, n_off):
    for i, v in enumerate(x):
        w48(A["X"] + 8 * i, v, 45)
    ip.write(A["N0"], n0); ip.write(A["N_ON"], n_on); ip.write(A["N_OFF"], n_off)
    ip.write(A["NSTEPS"], nsteps); ip.write(A["DECIM"], DECIM)
    t0 = time.perf_counter()
    ip.write(0x00, 1)
    while not (ip.read(0x00) & 0x2):
        time.sleep(0.001)
    t = time.perf_counter() - t0
    return t, [r48(A["X"] + 8 * i, 45) for i in range(14)]

print("run 1: voltage dip to", VDIP, "pu for steps", N_ON + 1, "-", N_OFF, "(0.1 s to 0.2 s), compared at", [round(c * H, 3) for c in CUTS], "s")
x, n, ttot = list(X0), 0, 0.0
for g, c in enumerate(CUTS):
    t, x = run(x, n, c - n, N_ON, N_OFF)
    ttot += t
    print("  t = %.3f s: max |board - Julia| over the 9 states %.3e" % (c * H, max(abs(x[i] - X_JULIA[g][i]) for i in range(9))))
    n = c
print("  wall-clock time %.4f s (simulated %.4f s)" % (ttot, CUTS[-1] * H))
print("  %-7s %16s %16s %12s" % ("state", "board", "Julia", "difference"))
for i in range(9):
    print("  %-7s %16.10f %16.10f %12.3e" % (NAMES[i], x[i], X_JULIA[-1][i], x[i] - X_JULIA[-1][i]))

N2 = 10_000_000
print("run 2: operating point,", N2, "steps (%.3f s of simulated time)" % (N2 * H))
t, x = run(X0, 0, N2, -10, -10)
print("  wall-clock time %.4f s, %.4f us per step (h = %.4f us)" % (t, t / N2 * 1e6, H * 1e6))
rel = lambda z: math.atan2(z[8], z[7]) - math.atan2(z[13], z[12])    # midpoint rotor angle relative to the source
dang = (rel(x) - rel(X0) + math.pi) % (2 * math.pi) - math.pi
print("  max |x - x0| over the 7 rotor-frame states: %.3e; relative angle change %.3e rad (the operating point must hold)"
      % (max(abs(x[i] - X0[i]) for i in range(7)), dang))
print("  phasor lengths after %d steps: |c^2 + s^2 - 1| machine %.3e, source %.3e (limit 1e-10)"
      % (N2, abs(x[7]**2 + x[8]**2 - 1), abs(x[12]**2 + x[13]**2 - 1)))
