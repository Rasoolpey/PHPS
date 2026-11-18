"""Plot an ILA capture (CSV from write_hw_ila_data) of the automated flow (stage 5).

    python3 plot_ila.py <ila.csv> <out.png> <sample_dt> <port:fraction_bits:label> ...

Each listed probe is decoded from its raw integer (signed decimal or hex, as written by Vivado) as value = raw / 2^F
and plotted against time (sample index x sample_dt).
"""
import csv
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

src, out, dt = sys.argv[1], sys.argv[2], float(sys.argv[3])
specs = [s.split(":") for s in sys.argv[4:]]

with open(src) as f:
    rows = list(csv.reader(f))
head = rows[0]
data = [r for r in rows[1:] if r and r[0].strip().lstrip("-").isdigit()]   # skip the radix line


def column(port):
    for i, h in enumerate(head):
        if port in h:
            return i
    raise SystemExit(f"probe {port} not in the CSV header: {head}")


def decode(s, bits=48):
    s = s.strip()
    v = int(s, 16) if any(c in "abcdefABCDEF" for c in s) else int(s)
    if v >= 1 << (bits - 1):                 # hex: two's complement
        v -= 1 << bits
    return v


fig, axes = plt.subplots(len(specs), 1, figsize=(9, 2.6 * len(specs)), sharex=True)
axes = axes if len(specs) > 1 else [axes]
t = [i * dt for i in range(len(data))]
for ax, (port, F, label) in zip(axes, specs):
    c = column(port)
    y = [decode(r[c]) / 2 ** int(F) for r in data]
    ax.plot(t, y, lw=1.2)
    ax.set_ylabel(label.replace("_", " "))
    ax.grid(alpha=0.3)
    print(f"  {port}: {len(y)} samples, min {min(y):.6f}, max {max(y):.6f}")
axes[-1].set_xlabel("time [s] (one ILA sample every %.1f us)" % (dt * 1e6))
fig.suptitle("ILA capture on the PYNQ-Z2")
fig.tight_layout()
fig.savefig(out, dpi=130)
print(f"  figure written to {out}")
