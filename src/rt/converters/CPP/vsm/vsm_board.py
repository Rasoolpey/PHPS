# vsm_board.py: written by PHPS.RT.vsm_board_script (src/rt/converters/vsm.jl); do not edit.
# Runs the VSM controller-call IP on the PYNQ-Z2: the replay of the PHPS simulator's bus-16 fault run (six converters,
# 20,000 calls each; vsm_board_data.bin beside this file), in batches of 256 calls with the states carried by the host.
# Every call's held output against the C++ (C simulation build, bit for bit) and Julia (limit 5e-8); flags against Julia.
#   python3 vsm_board.py [--no-download]
import sys, time, re
import numpy as np
from pynq import Overlay, Clocks
NB, LIM = 256, 5e-8
FST, FTURN, FANG, FSIG = 45, 46, 54, 40
NAMES = ["GFM_VSM_31", "GFM_VSM_33", "GFM_VSM_34_1+GFM_VSM_34_2", "GFM_VSM_36", "GFM_VSM_37", "GFM_VSM_38"]
SETS = [[(0.0, 48), (0.014285714300001473, 48), (0.014285714300001473, 48), (0.0, 48), (0.020408163300000837, 48), (0.10204081630000061, 48), (0.1040616227071709, 48), (0.010828821320448156, 48), (0.0014577259486010519, 48), (0.0, 48), (7.699999999999818, 40), (0.012100000024474866, 40), (0.0, 40), (-69.99999992999983, 40), (5.208110719794604, 40), (1.013676248214324, 40), (8.0, 40), (64.0, 40), (1.1547005383790747, 40), (1.3333333333330302, 40), (0.5, 40), (1.9899999999997817, 40), (0.8152827902267745, 38), (0.6646860280379769, 38), (0.4162464908295078, 38), (0.10786387442567502, 38), (4.8048453117735335, 38), (3.5714285714361083e-6, 55), (0.0012500000000000011, 55), (0.0015707950000000137, 55), (0.0025000000000000022, 55), (0.002999999057557823, 55), (1.2941837269476339e-17, 102), (5.208110719794831, 43), (1.0136762482145514, 43), (3.571428571428573e-6, 64), (0.0, 40), (1.0, 40), (1.0, 40)],
        [(0.0, 48), (0.01249999999999929, 48), (0.01249999999999929, 48), (0.0, 48), (0.015625, 48), (0.078125, 48), (0.0796721798998874, 48), (0.00634765625, 48), (0.0009765625, 48), (0.0, 48), (8.800000000000182, 40), (0.01209999999991851, 40), (0.0, 40), (-80.0, 40), (6.31999999738764, 40), (1.014076060865591, 40), (8.0, 40), (64.0, 40), (1.1547005383790747, 40), (1.3333333333330302, 40), (0.5, 40), (1.9899999999997817, 40), (0.7133724415652978, 38), (0.5089002403838094, 38), (0.31868871959886746, 38), (0.1078638743274496, 38), (6.2757163244205, 38), (3.124999999992717e-6, 55), (0.0012500000000000011, 55), (0.0015707950000000137, 55), (0.0025000000000000022, 55), (0.002999999057557823, 55), (1.2941837269476339e-17, 102), (6.319999997387981, 43), (1.0140760608653636, 43), (3.124999999999981e-6, 64), (0.0, 40), (1.0, 40), (1.0, 40)],
        [(0.0, 48), (0.01666666664999994, 48), (0.01666666664999994, 48), (0.0, 48), (0.027777777775000345, 48), (0.13888888889999862, 48), (0.14163943094348497, 48), (0.02006172839799447, 48), (0.0023148148126850288, 48), (0.0, 48), (6.600000000000364, 40), (0.012099999975362152, 40), (0.0, 40), (-60.000000060000275, 40), (5.079999999559732, 40), (1.042950621399541, 40), (8.0, 40), (64.0, 40), (1.1547005383790747, 40), (1.3333333333330302, 40), (0.5, 40), (1.9899999999997817, 40), (0.9511632554713287, 38), (0.9047115385619691, 38), (0.5665577237741672, 38), (0.10786387421831023, 38), (3.5300904322284623, 38), (4.166666666666208e-6, 55), (0.0012500000000000011, 55), (0.0015707950000000137, 55), (0.0025000000000000022, 55), (0.002999999057557823, 55), (1.2941837269476339e-17, 102), (5.079999999559732, 43), (1.0429506213999957, 43), (4.1666666666666415e-6, 64), (0.0, 40), (1.0, 40), (1.0, 40)],
        [(0.0, 48), (0.014285714300001473, 48), (0.014285714300001473, 48), (0.0, 48), (0.020408163300000837, 48), (0.10204081630000061, 48), (0.1040616227071709, 48), (0.010828821320448156, 48), (0.0014577259486010519, 48), (0.0, 48), (7.699999999999818, 40), (0.012100000024474866, 40), (0.0, 40), (-69.99999992999983, 40), (5.599999998677049, 40), (1.079714196455825, 40), (8.0, 40), (64.0, 40), (1.1547005383790747, 40), (1.3333333333330302, 40), (0.5, 40), (1.9899999999997817, 40), (0.8152827902267745, 38), (0.6646860280379769, 38), (0.4162464908295078, 38), (0.10786387442567502, 38), (4.8048453117735335, 38), (3.5714285714361083e-6, 55), (0.0012500000000000011, 55), (0.0015707950000000137, 55), (0.0025000000000000022, 55), (0.002999999057557823, 55), (1.2941837269476339e-17, 102), (5.59999999867739, 43), (1.079714196455484, 43), (3.571428571428573e-6, 64), (0.0, 40), (1.0, 40), (1.0, 40)],
        [(0.0, 48), (0.014285714300001473, 48), (0.014285714300001473, 48), (0.0, 48), (0.020408163300000837, 48), (0.10204081630000061, 48), (0.1040616227071709, 48), (0.010828821320448156, 48), (0.0014577259486010519, 48), (0.0, 48), (7.699999999999818, 40), (0.012100000024474866, 40), (0.0, 40), (-69.99999992999983, 40), (5.400000002155139, 40), (1.0305978411524848, 40), (8.0, 40), (64.0, 40), (1.1547005383790747, 40), (1.3333333333330302, 40), (0.5, 40), (1.9899999999997817, 40), (0.8152827902267745, 38), (0.6646860280379769, 38), (0.4162464908295078, 38), (0.10786387442567502, 38), (4.8048453117735335, 38), (3.5714285714361083e-6, 55), (0.0012500000000000011, 55), (0.0015707950000000137, 55), (0.0025000000000000022, 55), (0.002999999057557823, 55), (1.2941837269476339e-17, 102), (5.400000002155366, 43), (1.030597841152371, 43), (3.571428571428573e-6, 64), (0.0, 40), (1.0, 40), (1.0, 40)],
        [(0.0, 48), (0.010000000000001563, 48), (0.010000000000001563, 48), (0.0, 48), (0.010000000000001563, 48), (0.05000000000000071, 48), (0.05099019513592751, 48), (0.002600000000001046, 48), (0.0004999999999988347, 48), (0.0, 48), (11.0, 40), (0.01209999999991851, 40), (0.0, 40), (-100.0, 40), (8.300000002592242, 40), (1.0318979597368525, 40), (8.0, 40), (64.0, 40), (1.1547005383790747, 40), (1.3333333333330302, 40), (0.5, 40), (1.9899999999997817, 40), (0.5706979532515106, 38), (0.32569615384636563, 38), (0.20396078054545796, 38), (0.1078638743274496, 38), (9.805806756910897, 38), (2.4999999999886224e-6, 55), (0.0012500000000000011, 55), (0.0015707950000000137, 55), (0.0025000000000000022, 55), (0.002999999057557823, 55), (1.2941837269476339e-17, 102), (8.300000002591787, 43), (1.0318979597365114, 43), (2.5000000000000066e-6, 64), (0.0, 40), (1.0, 40), (1.0, 40)]]
A = {}
for m in re.finditer(r"#define XVSM_RUN_CONTROL_ADDR_(\w+?)_(DATA|BASE)\s+0x([0-9a-fA-F]+)", open("xvsm_run_hw.h").read()):
    A[m.group(1)] = int(m.group(3), 16)
DOWNLOAD = "--no-download" not in sys.argv
ol = Overlay("vsm.bit", download=DOWNLOAD)
if abs(Clocks.fclk0_mhz - 125.0) > 0.01:
    Clocks.fclk0_mhz = 125.0
print("bitstream", "loaded by PYNQ" if DOWNLOAD else "kept (already loaded)", "| FCLK0 = %.3f MHz" % Clocks.fclk0_mhz)
ip = ol.vsm_run_0
mem = ip.mmio.array                                    # the register space as uint32 words
def raw(v, F):
    return int(round(v * 2**F)) & ((1 << 48) - 1)
def w48(addr, v, F):
    r = raw(v, F)
    ip.write(addr, r & 0xFFFFFFFF)
    ip.write(addr + 4, r >> 32)
def r48raw(addr):
    r = ip.read(addr) | (ip.read(addr + 4) & 0xFFFF) << 32
    return r - (1 << 48) if r >> 47 else r
def wvec(base, vals, F):                               # 48-bit words in 64-bit slots, vectorised
    r = np.round(np.asarray(vals, dtype=np.float64) * 2.0**F).astype(np.int64) & ((1 << 48) - 1)
    o = base // 4
    mem[o:o + 2 * len(r):2] = (r & 0xFFFFFFFF).astype(np.uint32)
    mem[o + 1:o + 2 * len(r):2] = (r >> 32).astype(np.uint32)
def rvec(base, n):
    o = base // 4
    lo = mem[o:o + 2 * n:2].astype(np.int64); hi = mem[o + 1:o + 2 * n:2].astype(np.int64) & 0xFFFF
    r = lo | (hi << 32)
    return np.where(r >> 47, r - (1 << 48), r)
def rint(base, n):
    o = base // 4
    return mem[o:o + n].astype(np.int64)

D = np.fromfile("vsm_board_data.bin", dtype=np.float64)
nconv, ncall = int(D[0]), int(D[1]); p = 2
if "--timing" in sys.argv:
    # one batch from inside the fault (calls 2001-2256 of the converter whose virtual impedance acts most, from the
    # Julia state there), for the every-clock capture of the cycles per call on the long path; no checks
    j = int(dict(a.split("=") for a in sys.argv if a.startswith("--conv=")).get("--conv", "3")) - 1
    q = 2 + j * (6 + 10 * ncall); s = int(D[q]); R = D[q + 6:q + 6 + 10 * ncall].reshape(ncall, 10)
    for i, (v, F) in enumerate(SETS[s]):
        w48(A["K"] + 8 * i, v, F)
    S = np.fromfile("vsm_board_state2000.bin", dtype=np.float64).reshape(nconv, 5)[j]
    for i in range(3):
        w48(A["X"] + 8 * i, S[i], FST)
    w48(A["PH_I"], S[3], FTURN); w48(A["PR_I"], S[4], FANG)
    wvec(A["U"], R[2000:2000 + NB, 0:5].reshape(-1), FSIG)
    ip.write(A["NCALL"], NB)
    ip.write(0x00, 1)
    while not (ip.read(0x00) & 0x2):
        pass
    fl = rint(A["FO"], NB)
    print("timing batch: converter %d, calls 2001-%d, engaged in %d of %d calls" % (j + 1, 2000 + NB, int(np.sum(fl & 1)), NB))
    sys.exit(0)
# the replay, the six converters interleaved batch by batch through the one IP (51, 3): each batch loads that converter's
# coefficients and its five state words, runs, and reads back the outputs, the packed flags and the five state words;
# outputs and states against the C++ (bit for bit: the C simulation's board top, vsm_board_bstates.bin), flags against Julia
BS = np.fromfile("vsm_board_bstates.bin", dtype=np.float64).reshape(nconv, -1, 5)
conv = []
for j in range(nconv):
    s = int(D[p]); init = D[p + 1:p + 6]; p += 6
    R = D[p:p + 10 * ncall].reshape(ncall, 10); p += 10 * ncall      # u (5), y C++ (2), y Julia (2), flags Julia
    conv.append(dict(s=s, R=R, st=list(init), Y=[], F=[], dS=0))
FS = (FST, FST, FST, FTURN, FANG)
nb_all = (ncall + NB - 1) // NB
t_run, t_all, nbat = 0.0, 0.0, 0
for b in range(nb_all):
    b0 = b * NB; n = min(NB, ncall - b0)
    for j, c in enumerate(conv):
        ta = time.perf_counter()
        for i, (v, F) in enumerate(SETS[c["s"]]):        # this instance's coefficients
            w48(A["K"] + 8 * i, v, F)
        x, ph, pr = c["st"][:3], c["st"][3], c["st"][4]
        for i in range(3):
            w48(A["X"] + 8 * i, x[i], FST)
        w48(A["PH_I"], ph, FTURN); w48(A["PR_I"], pr, FANG)
        wvec(A["U"], c["R"][b0:b0 + n, 0:5].reshape(-1), FSIG)
        ip.write(A["NCALL"], n)
        t0 = time.perf_counter()
        ip.write(0x00, 1)
        while not (ip.read(0x00) & 0x2):
            pass
        t_run += time.perf_counter() - t0
        c["Y"].append(rvec(A["Y"], 2 * n).reshape(n, 2)); c["F"].append(rint(A["FO"], n))
        rws = [r48raw(A["X"] + 8 * i) for i in range(3)] + [r48raw(A["PH_O"]), r48raw(A["PR_O"])]
        t_all += time.perf_counter() - ta; nbat += 1
        ref = [int(round(v * 2.0**F)) for v, F in zip(BS[j, b], FS)]   # the C++ state after this batch, raw
        c["dS"] = max(c["dS"], max(abs(a - r) for a, r in zip(rws, ref)))
        c["st"] = [rws[i] / 2.0**FS[i] for i in range(5)]
ok, worst_c, worst_j, nflag, worst_s = True, 0, 0.0, 0, 0
ALLY = []                                              # the board's held outputs (raw), every call, for the figures
for j, c in enumerate(conv):
    R = c["R"]; Y = np.vstack(c["Y"]); fl = np.concatenate(c["F"])
    rc = np.round(R[:, 5:7] * 2.0**FSIG).astype(np.int64)
    dc = int(np.max(np.abs(Y - rc)))                    # board vs C++ in units of the last bit
    dj = float(np.max(np.abs(Y / 2.0**FSIG - R[:, 7:9]) / np.maximum(1.0, np.abs(R[:, 7:9]))))
    nf = int(np.sum(fl != R[:, 9].astype(np.int64)))
    pj = dc == 0 and c["dS"] == 0 and dj <= LIM and nf == 0
    print("converter %d %-28s %d calls: outputs board - C++ %d LSB, returned states (5 words, %d batches) board - C++ %d LSB, "
          "board - Julia %.3e (limit %.0e), flag differences %d: %s"
          % (j + 1, NAMES[c["s"]], ncall, dc, nb_all, c["dS"], dj, LIM, nf, "PASS" if pj else "FAIL"))
    ok &= pj; worst_c = max(worst_c, dc); worst_s = max(worst_s, c["dS"]); worst_j = max(worst_j, dj); nflag += nf
    ALLY.append(Y)
np.vstack(ALLY).astype(np.int64).tofile("vsm_board_y.bin")
print("interleaved: %d batches of up to %d calls, the six converters in turn (coefficients and states reloaded every batch)"
      % (nbat, NB))
print("host time: IP runs (start to done, polled) %.3f s = %.2f us per call; whole batches (coefficients, states and inputs "
      "written, run, outputs, flags and states read) %.3f s = %.2f us per call (AXI-Lite from the processor)"
      % (t_run, t_run / (nconv * ncall) * 1e6, t_all, t_all / (nconv * ncall) * 1e6))
print("board check: max |board - Julia| %.3e (limit %.0e), outputs board - C++ %d LSB, returned states board - C++ %d LSB, "
      "flag differences %d: %s" % (worst_j, LIM, worst_c, worst_s, nflag, "PASS" if ok else "FAIL"))
sys.exit(0 if ok else 1)
