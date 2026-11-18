#!/bin/bash
# The C simulation of a model on its own (in parallel with Vivado; the flow then runs with SKIP_CSIM=1):
#   scripts/realtime/flow/csim_only.sh <model>
# Work folder outputs/realtime/flow/<model>/csim; log outputs/realtime/flow/<model>/logs/2a_csim.log; exits 1 on failure.
# (no set -u: the Xilinx settings scripts read unset variables)
MODEL=$1
ROOT=$(cd "$(dirname "$0")/../../.." && pwd)
CPP=$(dirname "$(ls "$ROOT"/src/rt/*/CPP/"$MODEL"/flow.cfg | head -1)")
source "$CPP/flow.cfg"
OUT="$ROOT/outputs/realtime/flow/$MODEL"
mkdir -p "$OUT/csim" "$OUT/logs"
printf "part=xc7z020clg400-1\n[hls]\nflow_target=vivado\nsyn.file=$CPP/$MODEL.cpp\nsyn.top=$TOP\ntb.file=$CPP/${MODEL}_tb.cpp\nclock=${CLOCK_NS}ns\ncsim.clean=1\nsim.O=1\n" > "$OUT/csim/hls_config.cfg"
XIL=${XIL:-$HOME/tools/Xilinx/2026.1}
source "$XIL/Vitis/settings64.sh" >/dev/null
cd "$OUT/csim"
t0=$(date +%s)
echo "[$(date +%T)] C simulation of $MODEL (testbench $CPP/${MODEL}_tb.cpp)"
vitis-run --mode hls --csim --config hls_config.cfg --work_dir work > "$OUT/logs/2a_csim.log" 2>&1; rc=$?
echo "[$(date +%T)] C simulation done in $(( $(date +%s) - t0 )) s, exit $rc"
grep -E "^C sim check|sweep|corner|short calls" "$OUT/logs/2a_csim.log" | head -40
exit $rc
