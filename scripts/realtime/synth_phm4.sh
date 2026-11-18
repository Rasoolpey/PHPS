#!/bin/bash
# The generalised PHMachine (review 47b bounded step): export (Julia) and HLS C synthesis; no C simulation here.
#   outputs/realtime/steam/synth_phm4.sh      (in tmux rt); work outputs/realtime/flow/phmachine4/hls, log logs/2b_csynth.log
ROOT=$(cd "$(dirname "$0")/../.." && pwd)
CPP=$ROOT/src/rt/machines/CPP/phmachine4
OUT=$ROOT/outputs/realtime/flow/phmachine4
mkdir -p "$OUT/hls" "$OUT/logs"
say() { echo "[$(date +%T)] $*"; }
say "1 export: PHPS.RT.phm4_export()"
julia --project="$ROOT" -e 'using PHPS; PHPS.RT.phm4_export()' > "$OUT/logs/1_export.log" 2>&1 || { tail -5 "$OUT/logs/1_export.log"; exit 1; }
cat > "$OUT/hls/hls_config.cfg" <<CFG
part=xc7z020clg400-1
[hls]
flow_target=vivado
syn.file=$CPP/phmachine4.cpp
syn.top=phm4_run
clock=8ns
CFG
XIL=${XIL:-$HOME/tools/Xilinx/2026.1}
source "$XIL/Vitis/settings64.sh" >/dev/null
cd "$OUT/hls"
t0=$(date +%s)
say "2 hls: v++ -c --mode hls (C synthesis of phm4_run)"
v++ -c --mode hls --config hls_config.cfg --work_dir work > "$OUT/logs/2b_csynth.log" 2>&1; rc=$?
say "C synthesis done in $(( $(date +%s) - t0 )) s, exit $rc"
grep -E "Final II|Loop Constraint Status|Estimated Fmax|ERROR" "$OUT/logs/2b_csynth.log" | head -20
rpt=$(ls work/hls/syn/report/phm4_run_csynth.rpt 2>/dev/null)
[ -n "$rpt" ] && grep -E "^\|(Total|Utilization|Available)" "$rpt" | head -3
exit $rc
