#!/bin/bash
# C simulation of the DC side (gate 54): export, testbench data (Julia), C simulation (vitis-run --csim).
#   scripts/realtime/csim_dcside.sh      (in tmux rt); work outputs/realtime/flow/dcside/csim, log logs/2a_csim.log; exit 1 on failure
ROOT=$(cd "$(dirname "$0")/../.." && pwd)
CPP=$ROOT/src/rt/converters/CPP/dcside
OUT=$ROOT/outputs/realtime/flow/dcside
mkdir -p "$OUT/csim" "$OUT/logs"
say() { echo "[$(date +%T)] $*"; }
say "1 export and testbench data"
julia --project="$ROOT" -e 'using PHPS; PHPS.RT.dcside_export()' > "$OUT/logs/1_export.log" 2>&1 || { tail -5 "$OUT/logs/1_export.log"; exit 1; }
tail -1 "$OUT/logs/1_export.log"
(cd "$ROOT" && julia --project=. scripts/realtime/dcside_csim_data.jl) > "$OUT/logs/1b_tbdata.log" 2>&1 || { tail -5 "$OUT/logs/1b_tbdata.log"; exit 1; }
grep -E "^replay .*synthetic|^edge|^written" "$OUT/logs/1b_tbdata.log"
printf "part=xc7z020clg400-1\n[hls]\nflow_target=vivado\nsyn.file=$CPP/dcside.cpp\nsyn.top=dcside_run\ntb.file=$CPP/dcside_tb.cpp\nclock=8ns\ncsim.clean=1\nsim.O=1\n" > "$OUT/csim/hls_config.cfg"
XIL=${XIL:-$HOME/tools/Xilinx/2026.1}
source "$XIL/Vitis/settings64.sh" >/dev/null
cd "$OUT/csim"
t0=$(date +%s)
say "2 C simulation"
vitis-run --mode hls --csim --config hls_config.cfg --work_dir work > "$OUT/logs/2a_csim.log" 2>&1; rc=$?
say "C simulation done in $(( $(date +%s) - t0 )) s, exit $rc"
grep -E "^C sim check|ERROR|error:" "$OUT/logs/2a_csim.log" | head -40
exit $rc
