#!/bin/bash
# The VSM controller call (review 48 bounded step): export (Julia) and HLS C synthesis of vsm_call (latency, DSP, LUT).
#   scripts/realtime/synth_vsm.sh      (in tmux rt); work outputs/realtime/flow/vsm/hls, log logs/2b_csynth.log
ROOT=$(cd "$(dirname "$0")/../.." && pwd)
CPP=$ROOT/src/rt/converters/CPP/vsm
OUT=$ROOT/outputs/realtime/flow/vsm
mkdir -p "$OUT/hls" "$OUT/logs"
say() { echo "[$(date +%T)] $*"; }
say "1 export: PHPS.RT.vsm_export()"
julia --project="$ROOT" -e 'using PHPS; PHPS.RT.vsm_export()' > "$OUT/logs/1_export.log" 2>&1 || { tail -5 "$OUT/logs/1_export.log"; exit 1; }
tail -1 "$OUT/logs/1_export.log"
cat > "$OUT/hls/hls_config.cfg" <<CFG
part=xc7z020clg400-1
[hls]
flow_target=vivado
syn.file=$CPP/vsm.cpp
syn.top=vsm_call
clock=8ns
CFG
XIL=${XIL:-$HOME/tools/Xilinx/2026.1}
source "$XIL/Vitis/settings64.sh" >/dev/null
cd "$OUT/hls"
t0=$(date +%s)
say "2 hls: v++ -c --mode hls (C synthesis of vsm_call)"
v++ -c --mode hls --config hls_config.cfg --work_dir work > "$OUT/logs/2b_csynth.log" 2>&1; rc=$?
say "C synthesis done in $(( $(date +%s) - t0 )) s, exit $rc"
grep -E "Estimated Fmax|ERROR" "$OUT/logs/2b_csynth.log" | head -20
rpt=$(ls work/hls/syn/report/vsm_call_csynth.rpt 2>/dev/null)
[ -n "$rpt" ] && { grep -E "^\|(Total|Utilization|Available)" "$rpt" | head -3; grep -A6 "Latency (cycles)" "$rpt" | head -8; }
exit $rc
