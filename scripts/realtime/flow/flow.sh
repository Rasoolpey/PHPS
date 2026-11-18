#!/bin/bash
# The automated flow: one model from the Julia package to the figure of its run on the PYNQ-Z2.
#
#   scripts/realtime/flow/flow.sh <model> [first_stage] [last_stage]      e.g.  flow.sh genrou   or   flow.sh genrou 4 5
#
# Stages (each logs to outputs/realtime/flow/<model>/logs/<n>_<name>.log; progress lines with the time):
#   1 export   Julia: the model's C++, coefficients, testbench, board program and flow.cfg
#   2 hls      Vitis HLS: C simulation, C synthesis, package the IP
#   3 vivado   Vivado (batch): project, block design (Zynq, the IP, an ILA), bitstream, timing and utilization
#   4 board    PYNQ-Z2: copy the files, load the bitstream (PYNQ), attach Vivado to the ILA, arm, run, capture to CSV
#   5 figure   plot the ILA capture (PNG) and write the summary
# Run it inside tmux (`tmux attach -t rt`) to watch. Needs: Julia, the AMD tools 2026.1, `ssh pynq`, the board on JTAG.
set -e
MODEL=${1:?give a model name, e.g. genrou}
FIRST=${2:-1}
LAST=${3:-5}
ROOT=$(cd "$(dirname "$0")/../../.." && pwd)
FLOW=$ROOT/scripts/realtime/flow
OUT=$ROOT/outputs/realtime/flow/$MODEL
mkdir -p "$OUT/logs"
say() { echo "[$(date +%H:%M:%S)] $*" | tee -a "$OUT/logs/flow.log"; }
XIL=$HOME/tools/Xilinx/2026.1
BOARD_DIR=/home/xilinx/jupyter_notebooks/${MODEL}_flow
board() { ssh -o BatchMode=yes pynq "cd $BOARD_DIR && sudo -n bash -c 'source /etc/profile.d/xrt_setup.sh >/dev/null 2>&1; source /etc/profile.d/pynq_venv.sh >/dev/null 2>&1; $1'"; }

# ---- 1 export -------------------------------------------------------------------------------------------------------
if [ "$FIRST" -le 1 ] && [ "$LAST" -ge 1 ]; then
    say "1 export: julia --project=. scripts/realtime/export_cpp.jl $MODEL"
    (cd "$ROOT" && julia -t auto --project=. scripts/realtime/export_cpp.jl "$MODEL") > "$OUT/logs/1_export.log" 2>&1 \
        || { say "1 export FAILED, see logs/1_export.log"; tail -20 "$OUT/logs/1_export.log"; exit 1; }
    grep written "$OUT/logs/1_export.log" | tee -a "$OUT/logs/flow.log"
fi
CPP=$(dirname "$(ls "$ROOT"/src/rt/*/CPP/"$MODEL"/flow.cfg | head -1)")
source "$CPP/flow.cfg"
VLNV=porthos:rt:$TOP:$IP_VERSION

# ---- 2 hls ----------------------------------------------------------------------------------------------------------
if [ "$FIRST" -le 2 ] && [ "$LAST" -ge 2 ]; then
    mkdir -p "$OUT/hls"
    cat > "$OUT/hls/hls_config.cfg" <<EOF
part=xc7z020clg400-1
[hls]
flow_target=vivado
package.output.format=ip_catalog
package.output.syn=false
syn.file=$CPP/$MODEL.cpp
syn.top=$TOP
tb.file=$CPP/${MODEL}_tb.cpp
clock=${CLOCK_NS}ns
csim.clean=1
sim.O=1
package.ip.vendor=porthos
package.ip.library=rt
package.ip.version=$IP_VERSION
package.ip.display_name=$IP_NAME
EOF
    [ -n "${HLS_EXTRA:-}" ] && tr ';' '\n' <<< "$HLS_EXTRA" >> "$OUT/hls/hls_config.cfg"   # the model's own HLS settings
    source "$XIL/Vitis/settings64.sh" >/dev/null
    cd "$OUT/hls"
    if [ "${SKIP_CSIM:-0}" = 1 ]; then
        # SKIP_CSIM=1: the C simulation runs separately (scripts/realtime/flow/csim_only.sh, in parallel with Vivado)
        say "2 hls: C simulation skipped here (SKIP_CSIM=1; run it with csim_only.sh)"
    else
    say "2 hls: vitis-run --mode hls --csim --config hls_config.cfg --work_dir work   (C simulation)"
    vitis-run --mode hls --csim --config hls_config.cfg --work_dir work > "$OUT/logs/2a_csim.log" 2>&1 \
        || { say "C simulation FAILED, see logs/2a_csim.log"; tail -20 "$OUT/logs/2a_csim.log"; exit 1; }
    sed -n '/t \[s\]/,/CSim done/p' "$OUT/logs/2a_csim.log" | sed -n '1p;12p;22p;32p' | tee -a "$OUT/logs/flow.log"
    grep -E "^C sim check" "$OUT/logs/2a_csim.log" | tee -a "$OUT/logs/flow.log" || true   # a model's own verdict line
    fi
    say "2 hls: v++ -c --mode hls --config hls_config.cfg --work_dir work   (C synthesis)"
    v++ -c --mode hls --config hls_config.cfg --work_dir work > "$OUT/logs/2b_csynth.log" 2>&1 \
        || { say "C synthesis FAILED, see logs/2b_csynth.log"; tail -20 "$OUT/logs/2b_csynth.log"; exit 1; }
    grep -E "Final II|Loop Constraint Status|Estimated Fmax" "$OUT/logs/2b_csynth.log" | tee -a "$OUT/logs/flow.log"
    rpt=$(ls work/hls/syn/report/${TOP}_csynth.rpt)
    grep -E "^\|(Total|Utilization)" "$rpt" | head -2 | tee -a "$OUT/logs/flow.log"
    say "2 hls: vitis-run --mode hls --package --config hls_config.cfg --work_dir work   (package the IP)"
    vitis-run --mode hls --package --config hls_config.cfg --work_dir work > "$OUT/logs/2c_package.log" 2>&1 \
        || { say "Package FAILED, see logs/2c_package.log"; tail -20 "$OUT/logs/2c_package.log"; exit 1; }
    grep "Created IP archive" "$OUT/logs/2c_package.log" | tee -a "$OUT/logs/flow.log"
    cd "$ROOT"
fi

# ---- 3 vivado -------------------------------------------------------------------------------------------------------
if [ "$FIRST" -le 3 ] && [ "$LAST" -ge 3 ]; then
    source "$XIL/Vivado/settings64.sh" >/dev/null
    mkdir -p "$OUT/vivado"
    cd "$OUT/vivado"
    say "3 vivado: vivado -mode batch -source vivado_build.tcl   (project, block design, bitstream; about 5-15 min)"
    export IMPL_STRATEGY="${IMPL_STRATEGY:-}"
    vivado -mode batch -nojournal -log "$OUT/logs/3_vivado.log" -source "$FLOW/vivado_build.tcl" \
        -tclargs "$OUT/vivado" "$OUT/hls/work/hls/impl/ip" "$VLNV" "$TOP" "$FCLK_MHZ" "$PROBES" > /dev/null 2>&1 \
        || { say "Vivado FAILED, see logs/3_vivado.log"; grep -E "ERROR|CRITICAL" "$OUT/logs/3_vivado.log" | head; exit 1; }
    grep -E "^FLOW:" "$OUT/logs/3_vivado.log" | sed 's/^FLOW: /  /' | tee -a "$OUT/logs/flow.log"
    cd "$ROOT"
    wns=$(grep -oE "WNS -?[0-9.]+" "$OUT/logs/3_vivado.log" | tail -1 | awk '{print $2}')
    if awk "BEGIN{exit !($wns < 0)}"; then say "timing NOT met (WNS $wns ns): stop before the board"; exit 1; fi
fi

# ---- 4 board --------------------------------------------------------------------------------------------------------
if [ "$FIRST" -le 4 ] && [ "$LAST" -ge 4 ]; then
    V=$OUT/vivado
    say "4 board: scp $MODEL.bit $MODEL.hwh ${MODEL}_board.py load_overlay.py -> pynq:$BOARD_DIR"
    ssh -o BatchMode=yes pynq "mkdir -p $BOARD_DIR"
    scp -q "$V/$MODEL.bit" pynq:$BOARD_DIR/$MODEL.bit
    scp -q "$V/$MODEL.hwh" pynq:$BOARD_DIR/$MODEL.hwh
    scp -q "$CPP/${MODEL}_board.py" "$FLOW/load_overlay.py" pynq:$BOARD_DIR/
    # a model's board data (e.g. a replay), when it has one
    for f in "$CPP"/${MODEL}_board_*.bin; do [ -f "$f" ] && scp -q "$f" pynq:$BOARD_DIR/; done
    # the IP's register map (HLS driver header), for board programs that read their register offsets from it
    scp -q "$(ls "$OUT"/hls/work/hls/impl/ip/drivers/${TOP}_v*/src/x${TOP}_hw.h | head -1)" pynq:$BOARD_DIR/
    say "4 board: python3 load_overlay.py $MODEL.bit on the board   (PYNQ loads the bitstream, sets FCLK0 from the .hwh)"
    board "python3 load_overlay.py $MODEL.bit" 2>&1 | tee -a "$OUT/logs/flow.log" | grep -E "^bitstream .* loaded, FCLK0" >/dev/null \
        || { say "4 board: PYNQ did not load the bitstream (power-cycle the board, then rerun stage 4)"; exit 1; }
    tail -1 "$OUT/logs/flow.log"
    source "$XIL/Vivado/settings64.sh" >/dev/null
    # the board program as a small script, so Vivado gets only its path (quotes passed through -tclargs get mangled)
    cat > "$V/run_board.sh" <<EOF
ssh -o BatchMode=yes pynq "cd $BOARD_DIR && sudo -n bash -c 'source /etc/profile.d/xrt_setup.sh >/dev/null 2>&1; source /etc/profile.d/pynq_venv.sh >/dev/null 2>&1; python3 ${MODEL}_board.py $BOARD_ARGS'"
EOF
    say "4 board: vivado -mode batch -source vivado_ila.tcl   (attach to the ILA, arm it, run the board program, capture)"
    cd "$V"
    vivado -mode batch -nojournal -log "$OUT/logs/4_board.log" -source "$FLOW/vivado_ila.tcl" \
        -tclargs "$V/$MODEL.bit" "$V/$MODEL.ltx" "$SAMPLE_PROBE" "$V/ila.csv" "$V/run_board.sh" \
        > /dev/null 2>&1 || { say "Board/ILA FAILED, see logs/4_board.log"; grep -E "ERROR" "$OUT/logs/4_board.log" | head; exit 1; }
    sed -n '/^BOARD:/s/^BOARD: //p' "$OUT/logs/4_board.log" | tee "$OUT/board_output.txt" | tee -a "$OUT/logs/flow.log"
    grep -E "^FLOW:" "$OUT/logs/4_board.log" | sed 's/^FLOW: /  /' | tee -a "$OUT/logs/flow.log"
    cd "$ROOT"
    # failing checks of the board result (review 39, 2; models whose flow.cfg sets them)
    if [ "${STRICT:-0}" = 1 ]; then
        grep -qE "^board check: .*PASS" "$OUT/board_output.txt" || { say "4 board: board check FAILED (board_output.txt)"; exit 1; }
    fi
    if [ -n "${ILA_REF:-}" ]; then
        # the step each probe shows at the sample edge, from the HLS schedule of the step loop (probe_lags.py)
        sched=$(grep -l "_ssdm_op_Write.* %$SAMPLE_PROBE," "$OUT"/hls/work/hls/.autopilot/db/*.verbose.sched.rpt | head -1)
        ports=$(for c in $ILA_CHECK; do echo -n "${c%%:*} "; done)
        python3 "$FLOW/probe_lags.py" "$sched" "$STEP_CYCLES" "$SAMPLE_PROBE" $ports > "$OUT/logs/4_probe_lags.txt" \
            || { say "4 board: probe lags not found in the schedule"; exit 1; }
        say "4 board: probe lags (steps): $(tr '\n' ' ' < "$OUT/logs/4_probe_lags.txt")"
        say "4 board: python3 ila_compare.py ila.csv $ILA_REF   (every ILA sample against Julia)"
        python3 "$FLOW/ila_compare.py" "$V/ila.csv" "$CPP/$ILA_REF" --lags "$OUT/logs/4_probe_lags.txt" $ILA_CHECK \
            > "$OUT/logs/4_ila_compare.txt" 2>&1 && rc=0 || rc=$?
        tee -a "$OUT/logs/flow.log" < "$OUT/logs/4_ila_compare.txt"
        [ $rc -eq 0 ] || { say "4 board: ILA check FAILED"; exit 1; }
    fi
    if [ "${STEP_CYCLES:-0}" -gt 0 ]; then
        say "4 board: vivado -mode batch -source vivado_ila_every_clock.tcl   (every clock: cycles per step, expect $STEP_CYCLES)"
        cd "$V"
        vivado -mode batch -nojournal -log "$OUT/logs/4b_strobe.log" -source "$FLOW/vivado_ila_every_clock.tcl" \
            -tclargs "$V/$MODEL.ltx" strobe "$V/ila_every.csv" "$V/run_board.sh" > /dev/null 2>&1 \
            || { say "every-clock capture FAILED, see logs/4b_strobe.log"; exit 1; }
        cd "$ROOT"
        python3 "$FLOW/strobe_interval.py" "$V/ila_every.csv" strobe "$STEP_CYCLES" > "$OUT/logs/4b_strobe_interval.txt" 2>&1 && rc=0 || rc=$?
        tee -a "$OUT/logs/flow.log" < "$OUT/logs/4b_strobe_interval.txt"
        [ $rc -eq 0 ] || { say "4 board: cycles per step differ from $STEP_CYCLES"; exit 1; }
    fi
fi

# ---- 5 figure -------------------------------------------------------------------------------------------------------
if [ "$FIRST" -le 5 ] && [ "$LAST" -ge 5 ]; then
    say "5 figure: python3 plot_ila.py ila.csv -> ${MODEL}_ila.png"
    python3 "$FLOW/plot_ila.py" "$OUT/vivado/ila.csv" "$OUT/${MODEL}_ila.png" "$SAMPLE_DT" $PLOT | tee -a "$OUT/logs/flow.log"
    say "done: $OUT/${MODEL}_ila.png, board output $OUT/board_output.txt, logs in $OUT/logs"
fi
