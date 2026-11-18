#!/bin/bash
# The DC side: export at II (h = II x 8 ns) and HLS C synthesis of dcside_run, for a list of II values.
#   scripts/realtime/synth_dcside.sh 64 48 32      (in tmux rt); work outputs/realtime/flow/dcside/sweep_<II>
ROOT=$(cd "$(dirname "$0")/../.." && pwd)
CPP=$ROOT/src/rt/converters/CPP/dcside
XIL=${XIL:-$HOME/tools/Xilinx/2026.1}
source "$XIL/Vitis/settings64.sh" >/dev/null
julia --project="$ROOT" -e 'using PHPS; PHPS.RT.dcside_export()' 2>&1 | tail -1
for II in "$@"; do
    W=$ROOT/outputs/realtime/flow/dcside/sweep_$II; mkdir -p "$W"
    sed "s/#pragma HLS PIPELINE II=[0-9]*/#pragma HLS PIPELINE II=$II/" "$CPP/dcside.cpp" > "$W/dcside.cpp"
    cp "$CPP"/dcside.h "$CPP"/rt_util.h "$CPP"/dcside_rom.h "$W/"
    printf "part=xc7z020clg400-1\n[hls]\nflow_target=vivado\nsyn.file=$W/dcside.cpp\nsyn.top=dcside_run\nclock=8ns\n" > "$W/hls_config.cfg"
    (cd "$W" && v++ -c --mode hls --config hls_config.cfg --work_dir work > csynth.log 2>&1)
    rpt=$W/work/hls/syn/report/dcside_run_csynth.rpt
    echo "II $II: $(grep -oE 'Final II = [0-9]+, Depth = [0-9]+, loop .VITIS_LOOP[^ ]*' $W/csynth.log | tail -1) | $(grep -oE 'Estimated Fmax: [0-9.]+ MHz' $W/csynth.log) | $(grep -E '^\|Total' $rpt 2>/dev/null | head -1)"
done
