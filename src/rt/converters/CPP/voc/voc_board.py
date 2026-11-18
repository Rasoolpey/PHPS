# voc_board.py: written by PHPS.RT.conv_board_script (src/rt/converters/droop.jl); do not edit.
# Runs the voc controller-call IP on the PYNQ-Z2: the replay of the PHPS simulator's bus-16 fault run (six converters,
# 20,000 calls each; voc_board_data.bin beside this file), in batches of 256 calls, the six converters interleaved
# batch by batch with their coefficients and 11 state words reloaded. Every call's held output against the C++ (bit for
# bit) and Julia (5e-8); flags against Julia; the returned state after every batch against the C++ (bit for bit).
#   python3 voc_board.py [--no-download] [--timing]   (--timing: one batch with flag bit 1 set most, for the every-clock capture)
import sys, time, re
import numpy as np
from pynq import Overlay, Clocks
NB, LIM, NS = 256, 5e-8, 11
FSIG = 40
NAMES = ["GFM_VOC_31", "GFM_VOC_33", "GFM_VOC_34_1+GFM_VOC_34_2", "GFM_VOC_36", "GFM_VOC_37", "GFM_VOC_38"]
SETS = [[([0.0, 0.014285714300001473, 0.014285714300001473, 0.0, 0.020408163300000837, 0.10204081630000061, 0.1040616227071709, 0.010828821320448156, 0.0014577259486010519, 0.0], 48), ([7.699999999999818, 0.012100000024474866, 0.0, -69.99999992999983, 0.0, 0.0, 0.0, 0.0, 1.1547005383790747, 1.3333333333330302, 0.5, 1.9899999999997817], 40), ([0.8152827902267745, 0.6646860280379769, 0.4162464908295078, 0.10786387442567502, 4.8048453117735335], 38), ([-0.01884733851120135, -0.018496476835791364, 0.018669691003622146, -0.00035195695912992875], 52), ([0.0005000000000000004], 57), ([3.8278425568138985e-12, 0.0001428571428550001], 59), ([0.9906634958915035, -0.9906048329244967], 47), ([5.207525397033351, 1.9839861715763618, 1.0275395361943538], 42), ([8.79296635503124e-14, -6.750155989720952e-14], 89), ([0.024999999999636202, 0.125, 0.0078125], 40), ([0.002999999057557823], 55), ([1.2941837269476339e-17], 102)],
        [([0.0, 0.01249999999999929, 0.01249999999999929, 0.0, 0.015625, 0.078125, 0.0796721798998874, 0.00634765625, 0.0009765625, 0.0], 48), ([8.800000000000182, 0.01209999999991851, 0.0, -80.0, 0.0, 0.0, 0.0, 0.0, 1.1547005383790747, 1.3333333333330302, 0.5, 1.9899999999997817], 40), ([0.7133724415652978, 0.5089002403838094, 0.31868871959886746, 0.1078638743274496, 6.2757163244205], 38), ([-0.01884733851120135, -0.018496476835791364, 0.018669691003622146, -0.00035195695912992875], 52), ([0.0005000000000000004], 57), ([3.3493624540525957e-12, 0.0001250000000000001], 59), ([0.9906634958915035, -0.9906048329244967], 47), ([6.318653754759453, 1.105354973438125, 1.0283502572201542], 42), ([1.0835776720341528e-13, -3.197442310920451e-14], 89), ([0.024999999999636202, 0.125, 0.0078125], 40), ([0.002999999057557823], 55), ([1.2941837269476339e-17], 102)],
        [([0.0, 0.01666666664999994, 0.01666666664999994, 0.0, 0.027777777775000345, 0.13888888889999862, 0.14163943094348497, 0.02006172839799447, 0.0023148148126850288, 0.0], 48), ([6.600000000000364, 0.012099999975362152, 0.0, -60.000000060000275, 0.0, 0.0, 0.0, 0.0, 1.1547005383790747, 1.3333333333330302, 0.5, 1.9899999999997817], 40), ([0.9511632554713287, 0.9047115385619691, 0.5665577237741672, 0.10786387421831023, 3.5300904322284623], 38), ([-0.01884733851120135, -0.018496476835791364, 0.018669691003622146, -0.00035195695912992875], 52), ([0.0005000000000000004], 57), ([4.465816605403461e-12, 0.00016666666666750006], 59), ([0.9906634958915035, -0.9906048329244967], 47), ([5.079325660779887, 1.659935749370561, 1.0877459986786562], 42), ([8.526512829121202e-14, -9.392486788328824e-14], 89), ([0.024999999999636202, 0.125, 0.0078125], 40), ([0.002999999057557823], 55), ([1.2941837269476339e-17], 102)],
        [([0.0, 0.014285714300001473, 0.014285714300001473, 0.0, 0.020408163300000837, 0.10204081630000061, 0.1040616227071709, 0.010828821320448156, 0.0014577259486010519, 0.0], 48), ([7.699999999999818, 0.012100000024474866, 0.0, -69.99999992999983, 0.0, 0.0, 0.0, 0.0, 1.1547005383790747, 1.3333333333330302, 0.5, 1.9899999999997817], 40), ([0.8152827902267745, 0.6646860280379769, 0.4162464908295078, 0.10786387442567502, 4.8048453117735335], 38), ([-0.01884733851120135, -0.018496476835791364, 0.018669691003622146, -0.00035195695912992875], 52), ([0.0005000000000000004], 57), ([3.8278425568138985e-12, 0.0001428571428550001], 59), ([0.9906634958915035, -0.9906048329244967], 47), ([5.597525186064104, 1.0257772452262088, 1.1657827460273893], 42), ([-9.325873406851315e-14, 3.1530333899354446e-14], 89), ([0.024999999999636202, 0.125, 0.0078125], 40), ([0.002999999057557823], 55), ([1.2941837269476339e-17], 102)],
        [([0.0, 0.014285714300001473, 0.014285714300001473, 0.0, 0.020408163300000837, 0.10204081630000061, 0.1040616227071709, 0.010828821320448156, 0.0014577259486010519, 0.0], 48), ([7.699999999999818, 0.012100000024474866, 0.0, -69.99999992999983, 0.0, 0.0, 0.0, 0.0, 1.1547005383790747, 1.3333333333330302, 0.5, 1.9899999999997817], 40), ([0.8152827902267745, 0.6646860280379769, 0.4162464908295078, 0.10786387442567502, 4.8048453117735335], 38), ([-0.01884733851120135, -0.018496476835791364, 0.018669691003622146, -0.00035195695912992875], 52), ([0.0005000000000000004], 57), ([3.8278425568138985e-12, 0.0001428571428550001], 59), ([0.9906634958915035, -0.9906048329244967], 47), ([5.398621766613132, 0.010129117774340557, 1.0621319101878726], 42), ([-3.907985046680551e-14, 2.3092638912203256e-14], 89), ([0.024999999999636202, 0.125, 0.0078125], 40), ([0.002999999057557823], 55), ([1.2941837269476339e-17], 102)],
        [([0.0, 0.010000000000001563, 0.010000000000001563, 0.0, 0.010000000000001563, 0.05000000000000071, 0.05099019513592751, 0.002600000000001046, 0.0004999999999988347, 0.0], 48), ([11.0, 0.01209999999991851, 0.0, -100.0, 0.0, 0.0, 0.0, 0.0, 1.1547005383790747, 1.3333333333330302, 0.5, 1.9899999999997817], 40), ([0.5706979532515106, 0.32569615384636563, 0.20396078054545796, 0.1078638743274496, 9.805806756910897], 38), ([-0.01884733851120135, -0.018496476835791364, 0.018669691003622146, -0.00035195695912992875], 52), ([0.0005000000000000004], 57), ([2.6794903101867718e-12, 9.99999999999994e-5], 59), ([0.9906634958915035, -0.9906048329244967], 47), ([8.297593214301287, 0.23809791797384605, 1.0648133993083775], 42), ([-7.105427357601002e-15, 3.597122599785507e-14], 89), ([0.024999999999636202, 0.125, 0.0078125], 40), ([0.002999999057557823], 55), ([1.2941837269476339e-17], 102)]]
SW = [("PH", 0, 46), ("PR", 0, 54), ("XW", 0, 40), ("XW", 1, 40), ("XV", 0, 35), ("XV", 1, 35), ("XM", 0, 45), ("XR", 0, 78), ("XR", 1, 78), ("XR", 2, 78), ("XR", 3, 78)]   # the state words: (register, index, fraction bits)
SCALARS = ("PH", "PR", "XM")
A = {}
for m in re.finditer(r"#define XVOC_RUN_CONTROL_ADDR_(\w+?)_(DATA|BASE)\s+0x([0-9a-fA-F]+)", open("xvoc_run_hw.h").read()):
    A[m.group(1)] = int(m.group(3), 16)
def reg(name, io):                                     # a scalar register: NAME_I / NAME_O, or NAME (in and out)
    return A[name + "_" + io] if name + "_" + io in A else A[name]
DOWNLOAD = "--no-download" not in sys.argv
ol = Overlay("voc.bit", download=DOWNLOAD)
if abs(Clocks.fclk0_mhz - 125.0) > 0.01:
    Clocks.fclk0_mhz = 125.0
print("bitstream", "loaded by PYNQ" if DOWNLOAD else "kept (already loaded)", "| FCLK0 = %.3f MHz" % Clocks.fclk0_mhz)
ip = ol.voc_run_0
mem = ip.mmio.array
def raw(v, F):
    return np.round(np.asarray(v, dtype=np.float64) * 2.0**F).astype(np.int64)
def wvec(base, vals, F):
    r = raw(vals, F) & ((1 << 48) - 1)
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
def wcoefs(s):
    a = A["K"]
    for vals, F in SETS[s]:
        wvec(a, vals, F); a += 8 * len(vals)
def wstate(v):
    for (nm, i, F), x in zip(SW, v):
        wvec((reg(nm, "I") if nm in SCALARS else A[nm]) + 8 * i, [x], F)
def rstate():
    return np.array([rvec((reg(nm, "O") if nm in SCALARS else A[nm]) + 8 * i, 1)[0] for (nm, i, F) in SW])

D = np.fromfile("voc_board_data.bin", dtype=np.float64)
nconv, ncall = int(D[0]), int(D[1]); p = 2
BS = np.fromfile("voc_board_bstates.bin", dtype=np.float64).reshape(nconv, -1, NS)   # the C++ state after every batch
conv = []
for j in range(nconv):
    s = int(D[p]); init = D[p + 1:p + 1 + NS]; p += 1 + NS
    R = D[p:p + 10 * ncall].reshape(ncall, 10); p += 10 * ncall   # u (5), y C++ (2), y Julia (2), Julia's packed flags
    conv.append(dict(s=s, R=R, st=init.copy(), Y=[], F=[], dS=0))
FS = np.array([F for (_, _, F) in SW], dtype=np.float64)
nb_all = (ncall + NB - 1) // NB
if "--timing" in sys.argv:   # one batch where flag bit 1 is set most (e.g. the VOC's virtual impedance), from the C++ state
    best = max(((np.sum(c["R"][b * NB:(b + 1) * NB, 9].astype(np.int64) & 1), j, b) for j, c in enumerate(conv)
                for b in range(1, nb_all)))
    cnt, j, b = best; c = conv[j]; b0 = b * NB; n = min(NB, ncall - b0)
    wcoefs(c["s"]); wstate(BS[j, b - 1])
    wvec(A["U"], c["R"][b0:b0 + n, 0:5].reshape(-1), FSIG); ip.write(A["NCALL"], n)
    ip.write(0x00, 1)
    while not (ip.read(0x00) & 0x2):
        pass
    print("timing batch: converter %d, batch %d (%d calls, flag bit 1 set in %d)" % (j + 1, b, n, cnt))
    sys.exit(0)
t_run, t_all, nbat = 0.0, 0.0, 0
for b in range(nb_all):
    b0 = b * NB; n = min(NB, ncall - b0)
    for j, c in enumerate(conv):
        ta = time.perf_counter()
        wcoefs(c["s"]); wstate(c["st"])
        wvec(A["U"], c["R"][b0:b0 + n, 0:5].reshape(-1), FSIG)
        ip.write(A["NCALL"], n)
        t0 = time.perf_counter()
        ip.write(0x00, 1)
        while not (ip.read(0x00) & 0x2):
            pass
        t_run += time.perf_counter() - t0
        c["Y"].append(rvec(A["Y"], 2 * n).reshape(n, 2)); c["F"].append(rint(A["FO"], n))
        rws = rstate()
        t_all += time.perf_counter() - ta; nbat += 1
        ref = np.round(BS[j, b] * 2.0**FS).astype(np.int64)
        c["dS"] = max(c["dS"], int(np.max(np.abs(rws - ref))))
        c["st"] = rws / 2.0**FS
ok, worst_c, worst_j, nflag, worst_s = True, 0, 0.0, 0, 0
ALLY = []
for j, c in enumerate(conv):
    R = c["R"]; Y = np.vstack(c["Y"]); fl = np.concatenate(c["F"])
    dc = int(np.max(np.abs(Y - raw(R[:, 5:7], FSIG))))
    dj = float(np.max(np.abs(Y / 2.0**FSIG - R[:, 7:9]) / np.maximum(1.0, np.abs(R[:, 7:9]))))
    nf = int(np.sum(fl != R[:, 9].astype(np.int64)))
    pj = dc == 0 and c["dS"] == 0 and dj <= LIM and nf == 0
    print("converter %d %-28s %d calls: outputs board - C++ %d LSB, returned states (%d words, %d batches) board - C++ %d LSB, "
          "board - Julia %.3e (limit %.0e), flag differences %d: %s"
          % (j + 1, NAMES[c["s"]], ncall, dc, NS, nb_all, c["dS"], dj, LIM, nf, "PASS" if pj else "FAIL"))
    ok &= pj; worst_c = max(worst_c, dc); worst_s = max(worst_s, c["dS"]); worst_j = max(worst_j, dj); nflag += nf
    ALLY.append(Y)
np.vstack(ALLY).astype(np.int64).tofile("voc_board_y.bin")
print("interleaved: %d batches of up to %d calls, the six converters in turn (coefficients and states reloaded every batch)" % (nbat, NB))
print("host time: IP runs (start to done, polled) %.3f s = %.2f us per call; whole batches (coefficients, states and inputs "
      "written, run, outputs, flags and states read) %.3f s = %.2f us per call (AXI-Lite from the processor)"
      % (t_run, t_run / (nconv * ncall) * 1e6, t_all, t_all / (nconv * ncall) * 1e6))
print("board check: max |board - Julia| %.3e (limit %.0e), outputs board - C++ %d LSB, returned states board - C++ %d LSB, "
      "flag differences %d: %s" % (worst_j, LIM, worst_c, worst_s, nflag, "PASS" if ok else "FAIL"))
sys.exit(0 if ok else 1)
