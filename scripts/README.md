# scripts/

Command-line entry points. Each needs only the `PHPS` package (and CairoMakie for plots, installed on first use).
Run them from the repository root, for example `julia --project=. scripts/full_model_fault.jl`.

| Script | What it does |
| --- | --- |
| `full_model_fault.jl` | The full physical IEEE 39-bus EMT model (`cases/IEEE39Bus_FullModel`), one fault from its scenario: builds the model, checks the operating point, simulates and stores the run with its energy account in `outputs/ph/full_model/`. `--tstop=`, `--scenario=` |
| `full_model_plots.jl` | Figures of a stored full-model run (bus voltages, speeds, rotor angles, field voltages, turbine power); plotting only, no simulation |
| `ph_ieee39_fault.jl` | IEEE 39-bus as one port-Hamiltonian model (balanced dq, dynamic pi lines, full-winding machines, no controllers): equilibrium, structure checks, a three-phase fault with the implicit midpoint rule, step refinement and an independent BDF run, energy by component |
| `ph_audit.jl` | Port-Hamiltonian audit of a case at its equilibrium: storage, dissipation and port balances |
| `run_base_fault.jl` | RMS: IEEE 39-bus base case, bus-16 fault for 150 ms, with BDF1 and IDA, and plots |
