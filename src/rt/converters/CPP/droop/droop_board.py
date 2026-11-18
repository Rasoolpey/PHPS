# droop_board.py: written by PHPS.RT.conv_board_script (src/rt/converters/droop.jl); do not edit.
# Runs the droop controller-call IP on the PYNQ-Z2: the replay of the PHPS simulator's bus-16 fault run (six converters,
# 20,000 calls each; droop_board_data.bin beside this file), in batches of 256 calls, the six converters interleaved
# batch by batch with their coefficients and 24 state words reloaded. Every call's held output against the C++ (bit for
# bit) and Julia (5e-8); flags against Julia; the returned state after every batch against the C++ (bit for bit).
#   python3 droop_board.py [--no-download]
import sys, time, re
import numpy as np
from pynq import Overlay, Clocks
NB, LIM, NS = 256, 5e-8, 24
FSIG = 40
NAMES = ["GFM_DROOP_31", "GFM_DROOP_33", "GFM_DROOP_34_1+GFM_DROOP_34_2", "GFM_DROOP_36", "GFM_DROOP_37", "GFM_DROOP_38"]
SETS = [[([-0.01884733851120135, -0.018496476835791364, 0.018669691003622146, -0.00035195695912992875], 52), ([0.0005000000000000004, 0.0005000000000000004], 57), ([0.0025000000000000022], 55), ([0.02499999999999991, 0.010000000000000009], 52), ([0.9906634958915035, -0.9906048329244967], 47), ([0.04081632653061362, 0.204081632653061], 48), ([8.399999999999636, 10.5, 0.15000000014970283, 1.1547005383790747, 1.3333333333330302, 0.5, 1.9899999999997817, 0.0], 40), ([5.207525397033351, 1.9839861715763618, 1.0136762482145514], 42), ([0.007142857099999955], 53), ([0.0031115888517690704], 55), ([3.226288754430571e-18], 102), ([2.1428564568270374e-5], 62)],
        [([-0.01884733851120135, -0.018496476835791364, 0.018669691003622146, -0.00035195695912992875], 52), ([0.0005000000000000004, 0.0005000000000000004], 57), ([0.0025000000000000022], 55), ([0.02499999999999991, 0.010000000000000009], 52), ([0.9906634958915035, -0.9906048329244967], 47), ([0.03125, 0.15625], 48), ([9.600000000000364, 12.0, 0.1499999999996362, 1.1547005383790747, 1.3333333333330302, 0.5, 1.9899999999997817, 0.0], 40), ([6.318653754759453, 1.105354973438125, 1.0140760608653636], 42), ([0.006249999999999978], 53), ([0.0031184737782410377], 55), ([3.9493740323935765e-18], 102), ([1.8749994109736566e-5], 62)],
        [([-0.01884733851120135, -0.018496476835791364, 0.018669691003622146, -0.00035195695912992875], 52), ([0.0005000000000000004, 0.0005000000000000004], 57), ([0.0025000000000000022], 55), ([0.02499999999999991, 0.010000000000000009], 52), ([0.9906634958915035, -0.9906048329244967], 47), ([0.055555555555557135, 0.27777777777777857], 48), ([7.199999999999818, 9.0, 0.14999999984956958, 1.1547005383790747, 1.3333333333330302, 0.5, 1.9899999999997817, 0.0], 40), ([5.079325660779887, 1.659935749370561, 1.0429506213999957], 42), ([0.008333333349999972], 53), ([0.0031269821594398783], 55), ([4.709533463674964e-19], 102), ([2.4999992196315284e-5], 62)],
        [([-0.01884733851120135, -0.018496476835791364, 0.018669691003622146, -0.00035195695912992875], 52), ([0.0005000000000000004, 0.0005000000000000004], 57), ([0.0025000000000000022], 55), ([0.02499999999999991, 0.010000000000000009], 52), ([0.9906634958915035, -0.9906048329244967], 47), ([0.04081632653061362, 0.204081632653061], 48), ([8.399999999999636, 10.5, 0.15000000014970283, 1.1547005383790747, 1.3333333333330302, 0.5, 1.9899999999997817, 0.0], 40), ([5.597525186064104, 1.0257772452262088, 1.0797141964553703], 42), ([0.007142857099999955], 53), ([0.0031199459874299285], 55), ([-1.1333479720928911e-18], 102), ([2.1428564568270374e-5], 62)],
        [([-0.01884733851120135, -0.018496476835791364, 0.018669691003622146, -0.00035195695912992875], 52), ([0.0005000000000000004, 0.0005000000000000004], 57), ([0.0025000000000000022], 55), ([0.02499999999999991, 0.010000000000000009], 52), ([0.9906634958915035, -0.9906048329244967], 47), ([0.04081632653061362, 0.204081632653061], 48), ([8.399999999999636, 10.5, 0.15000000014970283, 1.1547005383790747, 1.3333333333330302, 0.5, 1.9899999999997817, 0.0], 40), ([5.398621766613132, 0.010129117774340557, 1.0305978411524848], 42), ([0.007142857099999955], 53), ([0.0031156837726633857], 55), ([-1.2006556414658149e-17], 102), ([2.1428564568270374e-5], 62)],
        [([-0.01884733851120135, -0.018496476835791364, 0.018669691003622146, -0.00035195695912992875], 52), ([0.0005000000000000004, 0.0005000000000000004], 57), ([0.0025000000000000022], 55), ([0.02499999999999991, 0.010000000000000009], 52), ([0.9906634958915035, -0.9906048329244967], 47), ([0.019999999999999574, 0.10000000000000142], 48), ([12.0, 15.0, 0.1499999999996362, 1.1547005383790747, 1.3333333333330302, 0.5, 1.9899999999997817, 0.0], 40), ([8.297593214301287, 0.23809791797384605, 1.0318979597366251], 42), ([0.0050000000000000044], 53), ([0.0031244629166723437], 55), ([2.7600130219796637e-18], 102), ([1.4999995287789166e-5], 62)]]
SW = [("PH", 0, 46), ("PR", 0, 54), ("XP", 0, 42), ("XP", 1, 42), ("XU", 0, 45), ("XS", 0, 40), ("XS", 1, 40), ("XS", 2, 40), ("XS", 3, 40), ("XS", 4, 40), ("XS", 5, 40), ("XV", 0, 35), ("XV", 1, 35), ("XR", 0, 78), ("XR", 1, 78), ("XR", 2, 78), ("XR", 3, 78), ("XR", 4, 78), ("XR", 5, 78), ("XR", 6, 78), ("XR", 7, 78), ("XR", 8, 78), ("XR", 9, 78), ("XR", 10, 78)]   # the state words: (register, index, fraction bits)
SCALARS = ("PH", "PR", "XU")
A = {}
for m in re.finditer(r"#define XDROOP_RUN_CONTROL_ADDR_(\w+?)_(DATA|BASE)\s+0x([0-9a-fA-F]+)", open("xdroop_run_hw.h").read()):
    A[m.group(1)] = int(m.group(3), 16)
def reg(name, io):                                     # a scalar register: NAME_I / NAME_O, or NAME (in and out)
    return A[name + "_" + io] if name + "_" + io in A else A[name]
DOWNLOAD = "--no-download" not in sys.argv
ol = Overlay("droop.bit", download=DOWNLOAD)
if abs(Clocks.fclk0_mhz - 125.0) > 0.01:
    Clocks.fclk0_mhz = 125.0
print("bitstream", "loaded by PYNQ" if DOWNLOAD else "kept (already loaded)", "| FCLK0 = %.3f MHz" % Clocks.fclk0_mhz)
ip = ol.droop_run_0
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

D = np.fromfile("droop_board_data.bin", dtype=np.float64)
nconv, ncall = int(D[0]), int(D[1]); p = 2
BS = np.fromfile("droop_board_bstates.bin", dtype=np.float64).reshape(nconv, -1, NS)   # the C++ state after every batch
conv = []
for j in range(nconv):
    s = int(D[p]); init = D[p + 1:p + 1 + NS]; p += 1 + NS
    R = D[p:p + 10 * ncall].reshape(ncall, 10); p += 10 * ncall   # u (5), y C++ (2), y Julia (2), Julia's packed flags
    conv.append(dict(s=s, R=R, st=init.copy(), Y=[], F=[], dS=0))
FS = np.array([F for (_, _, F) in SW], dtype=np.float64)
nb_all = (ncall + NB - 1) // NB
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
np.vstack(ALLY).astype(np.int64).tofile("droop_board_y.bin")
print("interleaved: %d batches of up to %d calls, the six converters in turn (coefficients and states reloaded every batch)" % (nbat, NB))
print("host time: IP runs (start to done, polled) %.3f s = %.2f us per call; whole batches (coefficients, states and inputs "
      "written, run, outputs, flags and states read) %.3f s = %.2f us per call (AXI-Lite from the processor)"
      % (t_run, t_run / (nconv * ncall) * 1e6, t_all, t_all / (nconv * ncall) * 1e6))
print("board check: max |board - Julia| %.3e (limit %.0e), outputs board - C++ %d LSB, returned states board - C++ %d LSB, "
      "flag differences %d: %s" % (worst_j, LIM, worst_c, worst_s, nflag, "PASS" if ok else "FAIL"))
sys.exit(0 if ok else 1)
