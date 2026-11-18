# gfl_board.py: written by PHPS.RT.gfl_board_export (src/rt/converters/gfl.jl); do not edit.
# Runs the GFL IP (one build, two modes) on the PYNQ-Z2: the 0.3 s event sequences of the six converters (gfl_board_data.bin),
# interleaved per call period (each converter: a mode-0 batch of its current-loop steps, then its call), coefficients and the 26
# state words reloaded every batch. Every u_c and every call's state words and flags against the C++ (bit for bit) and Julia
# (5e-8). --timing: one mode-0 batch of 512 steps (the every-clock capture).
#   python3 gfl_board.py [--no-download] [--timing]
import sys, time, re
import numpy as np
from pynq import Overlay, Clocks
LIM, FSIG = 5e-8, 40
SETS = [[([0.002999999057557823], 55), ([1.2941837269476339e-17], 102), ([0.00011140846016432365, 0.0007957747154594755], 57), ([0.03713616505433848, 0.2652583218167024], 48), ([0.0009999999999998899, 0.02499999999999991], 52), ([5.20811072000015, 1.9825178400001278, 0.003499999999803549, 0.003499999999803549, 0.5, 1.0, 0.5, 10.0, 0.0, 0.0, 0.0010000000002037268, 0.0, 0.01999999999998181, 7.699999999999818, 5.0], 41), ([5.000000000000013e-5], 61), ([0.14285714285714235, 0.14285714285714235], 48), ([0.014285714285716011], 48)],
        [([0.002999999057557823], 55), ([1.2941837269476339e-17], 102), ([0.00011140846016432365, 0.0007957747154594755], 57), ([0.03713616505433848, 0.2652583218167024], 48), ([0.0009999999999998899, 0.02499999999999991], 52), ([6.320000000000164, 1.099059810000199, 0.0039999999999054126, 0.0039999999999054126, 0.5, 1.0, 0.5, 10.0, 0.0, 0.0, 0.0010000000002037268, 0.0, 0.01999999999998181, 8.800000000000182, 5.0], 41), ([5.000000000000013e-5], 61), ([0.125, 0.125], 48), ([0.01249999999999929], 48)],
        [([0.002999999057557823], 55), ([1.2941837269476339e-17], 102), ([0.00011140846016432365, 0.0007957747154594755], 57), ([0.03713616505433848, 0.2652583218167024], 48), ([0.0009999999999998899, 0.02499999999999991], 52), ([5.079999999999927, 1.6576364799998373, 0.003000000000156433, 0.003000000000156433, 0.5, 1.0, 0.5, 10.0, 0.0, 0.0, 0.0010000000002037268, 0.0, 0.01999999999998181, 6.599999999999909, 5.0], 41), ([5.000000000000013e-5], 61), ([0.16666666666666785, 0.16666666666666785], 48), ([0.01666666666666572], 48)],
        [([0.002999999057557823], 55), ([1.2941837269476339e-17], 102), ([0.00011140846016432365, 0.0007957747154594755], 57), ([0.03713616505433848, 0.2652583218167024], 48), ([0.0009999999999998899, 0.02499999999999991], 52), ([5.599999999999909, 1.011750099999972, 0.003499999999803549, 0.003499999999803549, 0.5, 1.0, 0.5, 10.0, 0.0, 0.0, 0.0010000000002037268, 0.0, 0.01999999999998181, 7.699999999999818, 5.0], 41), ([5.000000000000013e-5], 61), ([0.14285714285714235, 0.14285714285714235], 48), ([0.014285714285716011], 48)],
        [([0.002999999057557823], 55), ([1.2941837269476339e-17], 102), ([0.00011140846016432365, 0.0007957747154594755], 57), ([0.03713616505433848, 0.2652583218167024], 48), ([0.0009999999999998899, 0.02499999999999991], 52), ([5.400000000000091, 0.004399259999900096, 0.003499999999803549, 0.003499999999803549, 0.5, 1.0, 0.5, 10.0, 0.0, 0.0, 0.0010000000002037268, 0.0, 0.01999999999998181, 7.699999999999818, 5.0], 41), ([5.000000000000013e-5], 61), ([0.14285714285714235, 0.14285714285714235], 48), ([0.014285714285716011], 48)],
        [([0.002999999057557823], 55), ([1.2941837269476339e-17], 102), ([0.00011140846016432365, 0.0007957747154594755], 57), ([0.03713616505433848, 0.2652583218167024], 48), ([0.0009999999999998899, 0.02499999999999991], 52), ([8.300000000000182, 0.2284162500000093, 0.005000000000109139, 0.005000000000109139, 0.5, 1.0, 0.5, 10.0, 0.0, 0.0, 0.0010000000002037268, 0.0, 0.01999999999998181, 11.0, 5.0], 41), ([5.000000000000013e-5], 61), ([0.10000000000000142, 0.10000000000000142], 48), ([0.010000000000001563], 48)]]
LOOPS = [[([4.5714286736079707e-7, 4.5714286736079707e-7, 2.2857143368039515e-7, 2.2857143368039515e-7], 67), ([0.14285714285714235, 0.14285714285714235, 0.014285714285716011], 48)],
         [([4.000000089407e-7, 4.000000089407e-7, 2.000000044703466e-7, 2.000000044703466e-7], 67), ([0.125, 0.125, 0.01249999999999929], 48)],
         [([5.333333452542621e-7, 5.333333452542621e-7, 2.6666667262713106e-7, 2.6666667262713106e-7], 67), ([0.16666666666666785, 0.16666666666666785, 0.01666666666666572], 48)],
         [([4.5714286736079707e-7, 4.5714286736079707e-7, 2.2857143368039515e-7, 2.2857143368039515e-7], 67), ([0.14285714285714235, 0.14285714285714235, 0.014285714285716011], 48)],
         [([4.5714286736079707e-7, 4.5714286736079707e-7, 2.2857143368039515e-7, 2.2857143368039515e-7], 67), ([0.14285714285714235, 0.14285714285714235, 0.014285714285716011], 48)],
         [([3.2000000715256e-7, 3.2000000715256e-7, 1.6000000357628e-7, 1.6000000357628e-7], 67), ([0.10000000000000142, 0.10000000000000142, 0.010000000000001563], 48)]]
SW = [("PH", 0, 46), ("PR", 0, 54), ("XS", 0, 41), ("XS", 1, 41), ("XS", 2, 41), ("XS", 3, 41), ("XS", 4, 41), ("XS", 5, 41), ("XS", 6, 41), ("XS", 7, 41), ("XS", 8, 41), ("XR", 0, 78), ("XR", 1, 78), ("XR", 2, 78), ("XR", 3, 78), ("XR", 4, 78), ("XR", 5, 78), ("LT", 0, 40), ("LT", 1, 40), ("LT", 2, 40), ("LT", 3, 40), ("LT", 4, 40), ("Z", 0, 45), ("Z", 1, 45), ("ZR", 0, 78), ("ZR", 1, 78)]
A = {}
for m in re.finditer(r"#define XGFL_RUN_CONTROL_ADDR_(\w+?)_(DATA|BASE)\s+0x([0-9a-fA-F]+)", open("xgfl_run_hw.h").read()):
    A[m.group(1)] = int(m.group(3), 16)
def reg(nm, io):
    return A[nm + "_" + io] if nm + "_" + io in A else A[nm]
DOWNLOAD = "--no-download" not in sys.argv
ol = Overlay("gfl.bit", download=DOWNLOAD)
if abs(Clocks.fclk0_mhz - 125.0) > 0.01:
    Clocks.fclk0_mhz = 125.0
print("bitstream", "loaded by PYNQ" if DOWNLOAD else "kept (already loaded)", "| FCLK0 = %.3f MHz" % Clocks.fclk0_mhz)
ip = ol.gfl_run_0
mem = ip.mmio.array
def raw(v, F):
    return np.round(np.asarray(v, dtype=np.float64) * 2.0**F).astype(np.int64)
def wvec(base, vals, F):
    r = raw(vals, F) & ((1 << 48) - 1); o = base // 4
    mem[o:o + 2 * len(r):2] = (r & 0xFFFFFFFF).astype(np.uint32); mem[o + 1:o + 2 * len(r):2] = (r >> 32).astype(np.uint32)
def rvec(base, n):
    o = base // 4
    lo = mem[o:o + 2 * n:2].astype(np.int64); hi = mem[o + 1:o + 2 * n:2].astype(np.int64) & 0xFFFF
    r = lo | (hi << 32)
    return np.where(r >> 47, r - (1 << 48), r)
def wfields(base, fields):
    for vals, F in fields:
        wvec(base, vals, F); base += 8 * len(vals)
FS = np.array([F for (_, _, F) in SW], dtype=np.float64)
def wstate(v):
    for (nm, i, F), x in zip(SW, v):
        wvec((reg(nm, "I") if nm in ("PH", "PR") else A[nm]) + 8 * i, [x], F)
def rstate():
    return np.array([rvec((reg(nm, "O") if nm in ("PH", "PR") else A[nm]) + 8 * i, 1)[0] for (nm, i, F) in SW])
def run(mode, n):
    ip.write(A["MODE"], mode); ip.write(A["N"], n); ip.write(A["DECIM"], 32)
    ip.write(0x00, 1)
    while not (ip.read(0x00) & 0x2):
        pass

D = np.fromfile("gfl_board_data.bin", dtype=np.float64)
nconv = int(D[0]); p = 1; conv = []
for j in range(nconv):
    s, nev = int(D[p]), int(D[p + 1]); init = D[p + 2:p + 28].copy(); p += 28
    steps, calls, order = [], [], []
    for e in range(nev):
        if D[p] == 0:
            steps.append(D[p + 1:p + 7]); order.append(0); p += 7
        else:
            calls.append(D[p + 1:p + 57]); order.append(1); p += 57   # v (2), C++ state (26), C++ flags, Julia state (26), Julia flags
    conv.append(dict(s=s, st=init, steps=np.array(steps), calls=np.array(calls), order=order, ps=0, pc=0, dC=0, dS=0, dJ=0.0, nf=0))
if "--timing" in sys.argv:
    c = conv[0]; wfields(A["K"], SETS[c["s"]]); wfields(A["KQ"], LOOPS[c["s"]]); wstate(c["st"])
    wvec(A["IL"], c["steps"][0:512, 0:2].reshape(-1), FSIG); run(0, 512)
    print("timing batch: 512 current-loop steps of converter 1"); sys.exit(0)
if "--timing-call" in sys.argv:   # a mode-1 batch of 64 calls of converter 1 (the inputs of its first 64 calls; timing only)
    c = conv[0]; wfields(A["K"], SETS[c["s"]]); wfields(A["KQ"], LOOPS[c["s"]]); wstate(c["st"])
    wvec(A["UC"], c["calls"][0:64, 0:2].reshape(-1), FSIG); run(1, 64)
    print("timing batch: 64 calls of converter 1"); sys.exit(0)
t_all = 0.0; nb = 0
for period in range(max(len(c["calls"]) for c in conv) + 1):
    for c in conv:
        # the steps up to the next call (or the end), then that call
        o = c["order"]; i0 = c["ps"] + c["pc"]
        if i0 >= len(o):
            continue
        ta = time.perf_counter()
        wfields(A["K"], SETS[c["s"]]); wfields(A["KQ"], LOOPS[c["s"]])
        n = 0
        while i0 + n < len(o) and o[i0 + n] == 0:
            n += 1
        k = 0
        while k < n:
            m = min(512, n - k); S = c["steps"][c["ps"]:c["ps"] + m]
            wstate(c["st"]); wvec(A["IL"], S[:, 0:2].reshape(-1), FSIG); run(0, m)
            Y = rvec(A["YL"], 2 * m).reshape(m, 2); c["st"] = rstate() / 2.0**FS; nb += 1
            c["dC"] = max(c["dC"], int(np.max(np.abs(Y - raw(S[:, 2:4], FSIG)))))
            c["dJ"] = max(c["dJ"], float(np.max(np.abs(Y / 2.0**FSIG - S[:, 4:6]) / np.maximum(1.0, np.abs(S[:, 4:6])))))
            c["ps"] += m; k += m
        if i0 + n < len(o):
            C = c["calls"][c["pc"]]
            wstate(c["st"]); wvec(A["UC"], C[0:2], FSIG); run(1, 1)
            rs = rstate(); fo = int(mem[A["FO"] // 4]); nb += 1
            c["dS"] = max(c["dS"], int(np.max(np.abs(rs - raw(C[2:28], FS)))))
            c["nf"] += int(fo != int(C[28])) + int(fo != int(C[55]))
            c["st"] = rs / 2.0**FS; c["pc"] += 1
        t_all += time.perf_counter() - ta
ok = True; wC = wS = 0; wJ = 0.0; nfa = 0
for j, c in enumerate(conv):
    pj = c["dC"] == 0 and c["dS"] == 0 and c["dJ"] <= LIM and c["nf"] == 0
    print("converter %d: %d steps, %d calls: u_c board - C++ %d LSB, call states (26 words) board - C++ %d LSB, board - Julia %.3e (limit %.0e), flag differences %d: %s"
          % (j + 1, c["ps"], c["pc"], c["dC"], c["dS"], c["dJ"], LIM, c["nf"], "PASS" if pj else "FAIL"))
    ok &= pj; wC = max(wC, c["dC"]); wS = max(wS, c["dS"]); wJ = max(wJ, c["dJ"]); nfa += c["nf"]
print("interleaved: %d batches (mode 0 and mode 1), the six converters in turn per call period; host time %.1f s" % (nb, t_all))
print("board check: max |board - Julia| %.3e (limit %.0e), u_c board - C++ %d LSB, call states board - C++ %d LSB, flag differences %d: %s"
      % (wJ, LIM, wC, wS, nfa, "PASS" if ok else "FAIL"))
sys.exit(0 if ok else 1)
