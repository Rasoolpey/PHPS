# Real-time implementation: from Julia to the FPGA

Each real-time model is written once in `src/rt/` (module `PHPS.RT`) in a restricted form of Julia: fixed sizes, no
memory allocation, typed signals (`Sig`, `Small`), coefficients prepared on the PC, and no division or trigonometric
unit inside the step. The same function runs in Julia as the double-precision reference, and the `@hls` macro
(`src/rt/translator/julia_to_cpp.jl`) translates it statement by statement into C++ for AMD Vitis HLS. How to write a
new model: `src/rt/TEMPLATE.jl`; a complete example: `src/rt/machines/genrou.jl`.

| Script | What it does |
|---|---|
| `export_cpp.jl <model>` | writes the model's C++, coefficients, test bench, board program and `flow.cfg` into `src/rt/<class>/CPP/<model>/` |
| `flow/flow.sh <model> [first] [last]` | the whole flow in five stages: export, Vitis HLS (C simulation, synthesis, IP), Vivado (block design with the Zynq, the IP and an ILA; bitstream), the board run on a PYNQ-Z2, and the figure of the captured run |
| `flow/csim_only.sh` | C simulation of one model only |
| `csim_*.sh`, `synth_*.sh` | C simulation and synthesis of the converter, DC-side and machine blocks |
| `*_csim_data.jl`, `*_board_data.jl` | the Julia reference runs that the C simulation and the board captures are compared with |
| `flow/*.py`, `flow/*.tcl` | Vivado batch scripts, the ILA capture, and the comparison and plotting of the board's output |

The IEEE 39-bus parameter sets of each block are in `src/rt/<class>/data/*.toml`; the instances of a type (for example
the six DC exciters) are parameter sets loaded at run time, so one bitstream serves them all.

Needs: Julia, AMD Vitis HLS and Vivado (2026.1), and for the board stages a PYNQ-Z2 (Zynq XC7Z020) reachable as
`ssh pynq` with its JTAG cable attached. Outputs go to `outputs/realtime/`.
