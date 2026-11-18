#!/usr/bin/env python3
"""Device view of a routed design on the XC7Z020, from the dump of vivado_device_dump.tcl (no GUI screenshot).

    python3 plot_device.py <dump_dir> <model> <out.png> [--dark]

The layout is Vivado's tile grid (one column per tile, one row per CLB row, y down as in the Device window); each
slice is half a CLB tile, each DSP48 and RAMB18 half of its 5-row tile, each RAMB36 the whole tile.
Left panel: every used site coloured by the block that owns most of its cells (the model's HLS core; the ILA with its
debug hub; the AXI interconnect, reset and PS glue). Right panel: the core's sites by resource (logic slices,
slices holding LUTRAM, DSP48, block RAM); sites of the other blocks in grey. Columns without sites (the white gap
in the upper half of X1Y1-X1Y2) are the part's configuration column, not missing data (all 13,300 slices are drawn).
"""
import csv, sys
from collections import Counter, defaultdict
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.collections import PatchCollection
from matplotlib.patches import Patch, Rectangle

dump, model, out = sys.argv[1], sys.argv[2], sys.argv[3]
dark = "--dark" in sys.argv
BG, EMPTY, INK, MUTED, EDGE = (("#000000", "#1b2a3a", "#e8e8e8", "#9a9a9a", "#3a4a5a") if dark else
                               ("#ffffff", "#ecebe7", "#1f1f1e", "#6b6a64", "#c9c8c2"))
BLUE, ORANGE, AQUA, YELLOW, MAGENTA, GREEN, GREY = ("#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300",
                                                   "#8f8e88" if not dark else "#5a5a5a")

sites = {r["site"]: r for r in csv.DictReader(open(f"{dump}/sites.csv"))}
regions = list(csv.DictReader(open(f"{dump}/regions.csv")))
core_prefix = f"system_i/{model}_run_0/"


def block(cell):
    if cell.startswith(core_prefix):
        return "core"
    if cell.startswith("system_i/ila_0/") or cell.startswith("dbg_hub/"):
        return "ila"
    return "glue"


owner = defaultdict(Counter); dmem = set()
for r in csv.DictReader(open(f"{dump}/cells.csv")):
    s = r["site"]
    if s not in sites:
        continue                                       # IO, clock buffers: not drawn
    b = block(r["cell"]); owner[s][b] += 1
    if b == "core" and r["group"] == "DMEM":
        dmem.add(s)


def rect(s):
    """the site's rectangle in grid units (x, y, w, h), y down"""
    t, gx, gy = s["type"], int(s["gx"]), int(s["gy"])
    name = s["site"]; xy = name.split("_")[-1]; sx = int(xy[1:xy.index("Y")]); sy = int(xy[xy.index("Y") + 1:])
    if t.startswith("SLICE"):
        return (gx - 0.5 + (0.5 if sx % 2 else 0.0) + 0.04, gy - 0.5 + 0.06, 0.42, 0.88)
    if t == "DSP48E1" or t == "RAMB18E1":              # two sites per 5-row tile; the even Y site is the lower one
        return (gx - 0.45, gy - 4.5 + (0.0 if sy % 2 else 2.5) + 0.1, 0.9, 2.3)
    if t.startswith("RAMB36"):
        return (gx - 0.45, gy - 4.5 + 0.1, 0.9, 4.8)
    return None


def draw(ax, colour_of, title, legend):
    ax.set_facecolor(BG)
    empty, used = [], defaultdict(list)
    for name, s in sites.items():
        if s["type"] == "PS7" or s["type"] == "RAMBFIFO36E1":
            continue
        rc = rect(s)
        if rc is None:
            continue
        c = colour_of(name, s)
        (used[c] if c else empty).append(Rectangle(rc[:2], rc[2], rc[3]))
    ax.add_collection(PatchCollection(empty, facecolor=EMPTY, edgecolor="none", zorder=1))
    for c, ps in used.items():
        ax.add_collection(PatchCollection(ps, facecolor=c, edgecolor="none", zorder=2))
    for r in regions:                                   # clock regions
        x0, y0, x1, y1 = (int(r[k]) for k in ("x0", "y0", "x1", "y1"))
        ax.add_patch(Rectangle((x0 - 0.5, y0 - 0.5), x1 - x0 + 1, y1 - y0 + 1, fill=False, ec=MUTED, lw=0.7, zorder=3))
        ax.text(x0 + 1, y1 - 1, r["region"], color=MUTED, fontsize=7, zorder=4)
    ax.add_patch(Rectangle((9.5, 0.5), 43, 103, facecolor=BG, ec=MUTED, lw=0.7, hatch="////", zorder=3, alpha=1.0))
    ax.text(31, 52, "PS7\n(ARM cores)", color=INK, ha="center", va="center", fontsize=8, zorder=4,
            bbox=dict(fc=BG, ec="none", pad=1.5))
    ax.set_xlim(-1, 188); ax.set_ylim(157, -1); ax.set_aspect("equal")
    ax.set_xticks([]); ax.set_yticks([])
    for sp in ax.spines.values():
        sp.set_visible(False)
    ax.set_title(title, color=INK, fontsize=9, loc="left")
    ax.legend(handles=legend, loc="upper center", bbox_to_anchor=(0.5, -0.01), ncol=2, fontsize=7.5, frameon=False,
              labelcolor=INK)


def main_block(name):
    return owner[name].most_common(1)[0][0] if name in owner else None


BCOL = {"core": BLUE, "ila": ORANGE, "glue": AQUA}
cnt = Counter(); kind = Counter()
for name, s in sites.items():
    b = main_block(name)
    if b:
        t = "slice" if s["type"].startswith("SLICE") else "DSP" if s["type"] == "DSP48E1" else "BRAM"
        cnt[(b, t)] += 1


def ckind(name, s):
    b = main_block(name)
    if not b:
        return None
    if b != "core":
        return GREY
    if s["type"] == "DSP48E1":
        return YELLOW
    if s["type"].startswith("RAMB"):
        return GREEN
    return MAGENTA if name in dmem else BLUE


for name, s in sites.items():
    c = ckind(name, s)
    if c:
        kind[c] += 1

fig, axs = plt.subplots(1, 2, figsize=(13, 5.9), facecolor=BG)
lab = {"core": f"{model} core (HLS)", "ila": "ILA + debug hub", "glue": "AXI interconnect, reset"}
draw(axs[0], lambda n, s: BCOL.get(main_block(n)) if main_block(n) else None, "used sites by block",
     [Patch(fc=BCOL[b], label=f"{lab[b]}: {cnt[(b, 'slice')]} slices, {cnt[(b, 'DSP')]} DSP, {cnt[(b, 'BRAM')]} BRAM")
      for b in ("core", "ila", "glue")] + [Patch(fc=EMPTY, label="unused site")])
draw(axs[1], ckind, f"{model} core by resource",
     [Patch(fc=BLUE, label=f"logic slices ({kind[BLUE]})"), Patch(fc=MAGENTA, label=f"slices with LUTRAM ({kind[MAGENTA]})"),
      Patch(fc=YELLOW, label=f"DSP48 ({kind[YELLOW]})"), Patch(fc=GREEN, label=f"block RAM ({kind[GREEN]})"),
      Patch(fc=GREY, label=f"other blocks ({kind[GREY]})"), Patch(fc=EMPTY, label="unused site")])
summ = dict(l.split(" ", 1) for l in open(f"{dump}/summary.txt").read().strip().split("\n"))
fig.suptitle(f"{model}: routed placement on the {summ['part']} (PYNQ-Z2), WNS {summ['WNS']} at 125 MHz",
             color=INK, fontsize=10)
fig.tight_layout()
fig.savefig(out, dpi=170, facecolor=BG, bbox_inches="tight", pad_inches=0.12)
print(f"{model}: core {cnt[('core', 'slice')]} slices, {cnt[('core', 'DSP')]} DSP, {cnt[('core', 'BRAM')]} BRAM; "
      f"LUTRAM slices {kind[MAGENTA]}; figure written to {out}")
