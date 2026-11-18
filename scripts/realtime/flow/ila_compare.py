#!/usr/bin/env python3
"""Every ILA sample against Julia (review 39, 2): the flow's stage 4 check.

    python3 ila_compare.py <ila.csv> <ref.csv> [--lags <probe_lags.txt>] <probe:fraction_bits:limit> ...

With --lags (from probe_lags.py, review 39, 1) probe p is compared with the reference column <p>_<lag>: the value
written <lag> iterations before the sample (the ports are written in different pipeline stages).

The ILA capture (Vivado CSV, one row per rising edge of the sample probe) is decoded with each probe's fraction bits
and compared row by row with the reference CSV (written by the model's export: columns named after the probes). Every
captured sample must be within its probe's limit; prints the worst difference per probe; exits 1 on failure.
The capture must hold exactly as many samples as the reference (a truncated capture fails, review 42, 3), and every
compared value must be finite.
"""
import csv, math, sys

ila, ref = sys.argv[1], sys.argv[2]
args = sys.argv[3:]
lags = {}
if args and args[0] == "--lags":
    lags = {l.split()[0]: int(l.split()[1]) for l in open(args[1]) if l.strip()}
    args = args[2:]
specs = [s.split(":") for s in args]
rows = list(csv.reader(open(ila)))
head, radix, data = rows[0], rows[1], [r for r in rows[2:] if r and r[0].strip().isdigit()]
R = list(csv.DictReader(open(ref)))


def col(port):
    return next(i for i, h in enumerate(head) if port in h)


def dec(s, hexa, bits=48):
    v = int(s, 16) if hexa else int(s)
    if hexa and v >= 1 << (bits - 1):
        v -= 1 << bits
    return v


n = len(data)
ok = n > 0 and n == len(R)
print(f"ILA samples {n}, reference samples {len(R)}: {'complete' if ok else 'INCOMPLETE (FAIL)'}")
for port, F, lim in specs:
    c, F, lim = col(port), int(F), float(lim)
    rc = f"{port}_{lags[port]}" if port in lags else port
    hexa = radix[c].strip() == "HEX"
    w, wi = 0.0, -1
    finite = True
    for i in range(min(n, len(R))):
        b, j = dec(data[i][c], hexa) / 2**F, float(R[i][rc])
        if not (math.isfinite(b) and math.isfinite(j)):
            finite = False
            w, wi = math.inf, i
            continue
        e = abs(b - j)
        if e > w:
            w, wi = e, i
    good = finite and w <= lim
    ok = ok and good
    print(f"  {port} (reference {rc}): max |board - Julia| {w:.3e} at sample {wi} (run {R[wi]['run'] if wi >= 0 else '-'}, step "
          f"{R[wi]['step'] if wi >= 0 else '-'}), limit {lim:.1e}: {'PASS' if good else 'FAIL'}")
print(f"ILA check: every sample: {'PASS' if ok else 'FAIL'}")
sys.exit(0 if ok else 1)
