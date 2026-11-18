"""Cycles per step on the board, read directly from an every-clock ILA capture (vivado_ila_every_clock.tcl).

    python3 strobe_interval.py <ila.csv> [strobe probe name] [expected cycles]

The strobe toggles once per step, so successive transitions are one step apart and successive rising edges two.
Prints the count, min, max and mean of both intervals in clock cycles; exits 1 if any transition interval differs
from the expected cycles (when given).
"""
import csv
import sys

src = sys.argv[1]
name = sys.argv[2] if len(sys.argv) > 2 else "strobe"
expect = int(sys.argv[3]) if len(sys.argv) > 3 else None
with open(src) as f:
    rows = list(csv.reader(f))
head = rows[0]
col = next((i for i, h in enumerate(head) if h.endswith(name)), None)
if col is None:
    raise SystemExit(f"probe {name} not in the CSV header: {head}")
data = [r for r in rows[1:] if r and r[0].strip().isdigit()]
s = [int(r[col]) & 1 for r in data]                       # 1-bit probe (written as SIGNED: -1 means 1)
edges = [i for i in range(1, len(s)) if s[i] != s[i - 1]]
rising = [i for i in edges if s[i] == 1]
def stats(idx):
    d = [b - a for a, b in zip(idx, idx[1:])]
    return len(d), min(d), max(d), sum(d) / len(d)
n, lo, hi, mean = stats(edges)
print(f"every-clock capture: {len(s)} cycles; strobe transitions {len(edges)}")
print(f"  cycles between transitions (one step): n {n}, min {lo}, max {hi}, mean {mean:.3f}")
n2, lo2, hi2, mean2 = stats(rising)
print(f"  cycles between rising edges (two steps): n {n2}, min {lo2}, max {hi2}, mean {mean2:.3f}")
if expect is not None:
    ok = lo == hi == expect and lo2 == hi2 == 2 * expect
    print(f"  expected {expect} cycles per step: {'PASS' if ok else 'FAIL'}")
    sys.exit(0 if ok else 1)
