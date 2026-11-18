# phmachine4_board.py: written by PHPS.RT.phm4_rt_export (src/rt/machines/phmachine4.jl); do not edit.
# Runs the PHMachine4 IP (one build, all ten IEEE-39 machine sets) on the PYNQ-Z2: each set on the dip bench (the port
# source x 0.5 for 0.1-0.2 s, 0.3 s), cut at the checkpoints of the C simulation; the 18 returned state words at every
# checkpoint against the C++ (bit for bit, phmachine4_board_cpp.bin) and Julia (5e-8 relative, phmachine4_board_data.bin).
#   python3 phmachine4_board.py [--no-download]
import sys, time, re
import numpy as np
from pynq import Overlay, Clocks
FST, FSIG, FSM = 45, 37, 56
A = {}
for m in re.finditer(r"#define XPHM4_RUN_CONTROL_ADDR_(\w+?)_(DATA|BASE)\s+0x([0-9a-fA-F]+)", open("xphm4_run_hw.h").read()):
    A[m.group(1)] = int(m.group(3), 16)
DOWNLOAD = "--no-download" not in sys.argv
ol = Overlay("phmachine4.bit", download=DOWNLOAD)
if abs(Clocks.fclk0_mhz - 125.0) > 0.01:
    Clocks.fclk0_mhz = 125.0
print("bitstream", "loaded by PYNQ" if DOWNLOAD else "kept (already loaded)", "| FCLK0 = %.3f MHz" % Clocks.fclk0_mhz)
ip = ol.phm4_run_0
def raw(v, F):
    return int(round(v * 2.0**F)) & ((1 << 48) - 1)
def w48(addr, v, F):
    r = raw(v, F); ip.write(addr, r & 0xFFFFFFFF); ip.write(addr + 4, r >> 32)
def r48(addr):
    r = ip.read(addr) | (ip.read(addr + 4) & 0xFFFF) << 32
    return r - (1 << 48) if r >> 47 else r
D = np.fromfile("phmachine4_board_data.bin", dtype=np.float64)
nset, ncut, non, noff = (int(v) for v in D[:4]); p = 4
cuts = [int(v) for v in D[p:p + ncut]]; p += ncut
C = np.fromfile("phmachine4_board_cpp.bin", dtype=np.int64).reshape(nset, ncut, 20) & ((1 << 48) - 1)   # raw 48-bit words
C = np.where(C >> 47, C - (1 << 48), C)
ok = True; worst = 0.0; nbits = 0; ttot = 0.0
for s in range(nset):
    name = "".join(chr(int(c)) for c in D[p:p + 12]).strip(); p += 12
    nw = int(D[p]); p += 1
    W = D[p:p + 2 * nw].reshape(nw, 2); p += 2 * nw
    x0 = D[p:p + 18]; p += 18
    vmag, cm, srot, vf, tm = D[p:p + 5]; p += 5
    J = D[p:p + 18 * ncut].reshape(ncut, 18); p += 18 * ncut
    for i, (v, F) in enumerate(W):
        if F == 0:
            ip.write(A["K"] + 8 * i, int(v)); ip.write(A["K"] + 8 * i + 4, 0)
        else:
            w48(A["K"] + 8 * i, v, int(F))
    w48(A["VMAG"], vmag, FSIG); w48(A["CM"], cm, FSM); w48(A["SROT"], srot, FSM)
    w48(A["VF"], vf, FSIG); w48(A["TM"], tm, FSIG); w48(A["VDIP"], 0.5, FSIG)
    for i in range(18):
        w48(A["X"] + 8 * i, x0[i], FST)
    ip.write(A["N_ON"], non); ip.write(A["N_OFF"], noff); ip.write(A["DECIM"], 32)
    prev, ex, nb = 0, 0.0, 0
    for g, c in enumerate(cuts):
        ip.write(A["N0"], prev); ip.write(A["NSTEPS"], c - prev)
        t0 = time.perf_counter(); ip.write(0x00, 1)
        while not (ip.read(0x00) & 0x2):
            pass
        ttot += time.perf_counter() - t0
        xr = [r48(A["X"] + 8 * i) for i in range(18)]
        nb += sum(1 for i in range(18) if xr[i] != C[s, g, i])
        ex = max(ex, max(abs(xr[i] / 2.0**FST - J[g, i]) / max(1.0, abs(J[g, i])) for i in range(18)))
        prev = c
    pas = nb == 0 and ex <= 5e-8
    ok = ok and pas; worst = max(worst, ex); nbits += nb
    print("  %-10s %d checkpoints x 18 words: board - C++ %s, board - Julia %.3e: %s" % (name, ncut, "0 LSB" if nb == 0 else "%d words differ" % nb, ex, "PASS" if pas else "FAIL"))
print("  wall-clock %.3f s for %d sets x %d steps (%.4f us per step incl. the AXI-Lite control)" % (ttot, nset, cuts[-1], ttot / (nset * cuts[-1]) * 1e6))
print("board check: %s (board = C++ bit for bit at %d checkpoints x 18 words x %d sets: %s; max board - Julia %.3e, limit 5e-8)" % ("PASS" if ok else "FAIL", ncut, nset, "yes" if nbits == 0 else "no", worst))
