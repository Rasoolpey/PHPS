#!/usr/bin/env python3
"""The step each ILA probe shows when the sample qualifier rises (review 39, 1), from the HLS schedule.

    python3 probe_lags.py <loop verbose.sched.rpt> <II> <sample_probe> <probe> ...

HLS schedules the writes of the ap_none probe ports in different stages of the pipelined step loop. A write in stage s
of iteration j reaches the port in cycle II j + s; the ILA records every probe in the cycle where the sample port
rises (stage s_q of iteration i). So probe p shows the value written in iteration i - L_p, L_p = ceil((s_p - s_q) / II)
(0 when s_p <= s_q). Prints "<probe> <lag> <stage>" per probe; exits 1 if a write is not found.
"""
import math, re, sys

rpt, II, qual, probes = sys.argv[1], int(sys.argv[2]), sys.argv[3], sys.argv[4:]
stage = {}
for line in open(rpt):
    m = re.match(r"ST_(\d+) : Operation \d+ .*\"%\S+ = write void @_ssdm_op_Write\S* (?:i\d+ )?%(\w+),", line)
    if m:
        stage.setdefault(m.group(2), int(m.group(1)))
missing = [p for p in [qual] + probes if p not in stage]
if missing:
    print(f"no write of {missing} in {rpt}"); sys.exit(1)
sq = stage[qual]
print(f"{qual} 0 {sq}")
for p in probes:
    print(f"{p} {max(0, math.ceil((stage[p] - sq) / II))} {stage[p]}")
