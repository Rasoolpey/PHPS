#!/bin/bash
# C simulation of a converter controller call (droop, voc): export, testbench data (Julia), C simulation (vitis-run --csim).
#   scripts/realtime/csim_conv.sh <droop|voc>      (in tmux rt); work outputs/realtime/flow/<tag>/csim, log logs/2a_csim.log
TAG=${1:?droop or voc}
ROOT=$(cd "$(dirname "$0")/../.." && pwd)
CPP=$ROOT/src/rt/converters/CPP/$TAG
OUT=$ROOT/outputs/realtime/flow/$TAG
mkdir -p "$OUT/csim" "$OUT/logs"
say() { echo "[$(date +%T)] $*"; }
say "1 export and testbench data"
julia --project="$ROOT" -e "using PHPS; PHPS.RT.${TAG}_rt_export()" > "$OUT/logs/1_export.log" 2>&1 || { tail -5 "$OUT/logs/1_export.log"; exit 1; }
(cd "$ROOT" && julia --project=. scripts/realtime/conv_csim_data.jl $TAG) > "$OUT/logs/1b_tbdata.log" 2>&1 || { tail -5 "$OUT/logs/1b_tbdata.log"; exit 1; }
grep -E "^replay|written" "$OUT/logs/1b_tbdata.log" | head -3
printf "part=xc7z020clg400-1\n[hls]\nflow_target=vivado\nsyn.file=$CPP/$TAG.cpp\nsyn.top=${TAG}_run\ntb.file=$CPP/${TAG}_tb.cpp\nclock=8ns\ncsim.clean=1\nsim.O=1\n" > "$OUT/csim/hls_config.cfg"
XIL=${XIL:-$HOME/tools/Xilinx/2026.1}
source "$XIL/Vitis/settings64.sh" >/dev/null
cd "$OUT/csim"
t0=$(date +%s)
say "2 C simulation"
vitis-run --mode hls --csim --config hls_config.cfg --work_dir work > "$OUT/logs/2a_csim.log" 2>&1; rc=$?
say "C simulation done in $(( $(date +%s) - t0 )) s, exit $rc"
grep -E "^C sim check|ERROR|error:" "$OUT/logs/2a_csim.log" | head -20
exit $rc
