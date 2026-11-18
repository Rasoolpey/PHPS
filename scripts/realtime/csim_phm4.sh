#!/bin/bash
# C simulation of PHMachine4 (all ten sets): export, testbench data (Julia), C simulation (vitis-run --csim).
#   scripts/realtime/csim_phm4.sh      (in tmux rt); work outputs/realtime/flow/phmachine4/csim, log logs/2a_csim.log
ROOT=$(cd "$(dirname "$0")/../.." && pwd)
CPP=$ROOT/src/rt/machines/CPP/phmachine4
OUT=$ROOT/outputs/realtime/flow/phmachine4
mkdir -p "$OUT/csim" "$OUT/logs"
say() { echo "[$(date +%T)] $*"; }
say "1 export and testbench data"
julia --project="$ROOT" -e 'using PHPS; PHPS.RT.phm4_export()' > "$OUT/logs/1_export.log" 2>&1 || { tail -5 "$OUT/logs/1_export.log"; exit 1; }
(cd "$ROOT" && julia --project=. scripts/realtime/phm4_csim_data.jl) > "$OUT/logs/1b_tbdata.log" 2>&1 || { tail -5 "$OUT/logs/1b_tbdata.log"; exit 1; }
grep -E "steps|written" "$OUT/logs/1b_tbdata.log" | tail -12
printf "part=xc7z020clg400-1\n[hls]\nflow_target=vivado\nsyn.file=$CPP/phmachine4.cpp\nsyn.top=phm4_run\ntb.file=$CPP/phmachine4_tb.cpp\nclock=8ns\ncsim.clean=1\nsim.O=1\n" > "$OUT/csim/hls_config.cfg"
XIL=${XIL:-$HOME/tools/Xilinx/2026.1}
source "$XIL/Vitis/settings64.sh" >/dev/null
cd "$OUT/csim"
t0=$(date +%s)
say "2 C simulation"
vitis-run --mode hls --csim --config hls_config.cfg --work_dir work > "$OUT/logs/2a_csim.log" 2>&1; rc=$?
say "C simulation done in $(( $(date +%s) - t0 )) s, exit $rc"
grep -E "^C sim check|ERROR|error:" "$OUT/logs/2a_csim.log" | head -20
exit $rc
