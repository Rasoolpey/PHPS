#!/bin/bash
# A converter call (droop, voc, gfl): export (Julia) and HLS C synthesis of the board top <tag>_run, for a list of
# multiplier limits; reports Fmax, resources and the latency of one call (the batch's per-iteration latency).
#   scripts/realtime/synth_conv.sh <tag> [mul_limit ...]      (in tmux rt); work outputs/realtime/flow/<tag>/sweep_<limit>
TAG=${1:?tag}; shift
ROOT=$(cd "$(dirname "$0")/../.." && pwd)
CPP=$ROOT/src/rt/converters/CPP/$TAG
XIL=${XIL:-$HOME/tools/Xilinx/2026.1}
source "$XIL/Vitis/settings64.sh" >/dev/null
julia --project="$ROOT" -e "using PHPS; PHPS.RT.${TAG}_rt_export()" 2>&1 | grep -E "written .*(fit|CPP/$TAG )" | tail -1
for L in "${@:-6}"; do
    W=$ROOT/outputs/realtime/flow/$TAG/sweep_$L${UNC:+_u$UNC}${II:+_ii$II}; mkdir -p "$W"
    sed -e "s/#pragma HLS ALLOCATION operation instances=mul limit=6 /#pragma HLS ALLOCATION operation instances=mul limit=$L /" ${II:+-e "s/#pragma HLS PIPELINE II=[0-9]*/#pragma HLS PIPELINE II=$II/"} "$CPP/$TAG.cpp" > "$W/$TAG.cpp"
    cp "$CPP"/$TAG.h "$CPP"/rt_util.h "$CPP"/${TAG}_rom.h "$W/"
    printf "part=xc7z020clg400-1\n[hls]\nflow_target=vivado\nsyn.file=$W/$TAG.cpp\nsyn.top=${TAG}_run\nclock=8ns\n" > "$W/hls_config.cfg"
    [ -n "${UNC:-}" ] && echo "clock_uncertainty=$UNC" >> "$W/hls_config.cfg"   # UNC=3ns: schedule for a shorter effective period
    (cd "$W" && v++ -c --mode hls --config hls_config.cfg --work_dir work > csynth.log 2>&1)
    rpt=$W/work/hls/syn/report/${TAG}_run_csynth.rpt
    lat=$(grep -A6 "Latency (cycles)" $rpt | grep -E "^\s+\|\s+[0-9]+\|" | head -1 | tr -s ' |' ' ')
    echo "mul $L: $(grep -oE 'Estimated Fmax: [0-9.]+ MHz' $W/csynth.log) | $(grep -E '^\|Total' $rpt 2>/dev/null | head -1) | batch latency min/max (1 / 256 calls): $lat | $(grep -E 'ERROR' $W/csynth.log | head -2)"
done
