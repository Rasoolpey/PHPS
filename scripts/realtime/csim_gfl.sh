#!/bin/bash
# C simulation of the GFL build (gfl_run, two modes): export, testbench data (Julia), C simulation.
#   scripts/realtime/csim_gfl.sh      (in tmux rt); work outputs/realtime/flow/gfl/csim, log logs/2a_csim.log
ROOT=$(cd "$(dirname "$0")/../.." && pwd); CPP=$ROOT/src/rt/converters/CPP/gfl; OUT=$ROOT/outputs/realtime/flow/gfl
mkdir -p "$OUT/csim" "$OUT/logs"
julia --project="$ROOT" -e 'using PHPS; PHPS.RT.gfl_rt_export()' > "$OUT/logs/1_export.log" 2>&1 || { tail -5 "$OUT/logs/1_export.log"; exit 1; }
(cd "$ROOT" && julia --project=. scripts/realtime/gfl_csim_data.jl) > "$OUT/logs/1b_tbdata.log" 2>&1 || { tail -5 "$OUT/logs/1b_tbdata.log"; exit 1; }
grep -E "steps|written" "$OUT/logs/1b_tbdata.log" | tail -8
printf "part=xc7z020clg400-1\n[hls]\nflow_target=vivado\nsyn.file=$CPP/gfl.cpp\nsyn.top=gfl_run\ntb.file=$CPP/gfl_tb.cpp\nclock=8ns\ncsim.clean=1\nsim.O=1\n" > "$OUT/csim/hls_config.cfg"
source "$HOME/tools/Xilinx/2026.1/Vitis/settings64.sh" >/dev/null; cd "$OUT/csim"
t0=$(date +%s); vitis-run --mode hls --csim --config hls_config.cfg --work_dir work > "$OUT/logs/2a_csim.log" 2>&1; rc=$?
echo "C simulation done in $(( $(date +%s) - t0 )) s, exit $rc"; grep -E "^C sim check|ERROR|error:" "$OUT/logs/2a_csim.log" | head -12; exit $rc
