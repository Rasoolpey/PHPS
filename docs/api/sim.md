# Simulation (RMS)

BDF1 and IDA time integration, results writers and run records. Source: [src/sim/](../../src/sim).

Generated from the docstrings by `scripts/build_api_docs.jl`; do not edit by hand.

| Name | Defined in |
|---|---|
| [`SimResult`](#simresult) | `bdf1.jl` |
| [`simulate_bdf1`](#simulate_bdf1) | `bdf1.jl` |
| [`consistent_voltages!`](#consistent_voltages) | `ida.jl` |
| [`simulate_ida`](#simulate_ida) | `ida.jl` |
| [`csv_columns`](#csv_columns) | `results.jl` |
| [`csv_row`](#csv_row) | `results.jl` |
| [`write_results_csv`](#write_results_csv) | `results.jl` |
| [`write_results_jld2`](#write_results_jld2) | `results.jl` |
| [`run_metadata`](#run_metadata) | `results.jl` |
| [`simulate`](#simulate) | `results.jl` |

---

### SimResult

*[src/sim/bdf1.jl:18](../../src/sim/bdf1.jl#L18)*

    SimResult

A simulation record: logged times `t`, logged states `Y` (one column per time,
`[x; Vd_1, Vq_1, ...]`), the system, solver statistics, and the settings the integrator
actually used (defaults filled in), for `run.json`.

---

### simulate_bdf1

*[src/sim/bdf1.jl:105](../../src/sim/bdf1.jl#L105)*

    simulate_bdf1(sys, y0; dt, duration, log_dt = dt, dv_max = 0.15, newton_tol = 1e-8,
                  max_newton = 40, eps_fd = 1e-7) -> SimResult

Integrate the DAE from `y0 = [x0; V0]` with Python PHPS's fixed-step BDF1 (see the file header).

---

### consistent_voltages!

*[src/sim/ida.jl:13](../../src/sim/ida.jl#L13)*

    consistent_voltages!(V, sys, x, active; ws = DAEWorkspace(sys), tol = 1e-12, maxiter = 50)

Solve KCL `g(x, V) = 0` for the bus voltages with the states `x` held (in place, from the
guess in `V`). Returns the final max-abs residual.

---

### simulate_ida

*[src/sim/ida.jl:42](../../src/sim/ida.jl#L42)*

    simulate_ida(sys, y0; dt, duration, log_dt = dt, rtol = nothing, atol = nothing)
        -> SimResult

Integrate the DAE from `y0 = [x0; V0]` with SUNDIALS IDA, restarting at every fault switch
(see the file header). Records every `log_dt` and at the end.

---

### csv_columns

*[src/sim/results.jl:4](../../src/sim/results.jl#L4)*

    csv_columns(sys) -> Vector{String}

Python PHPS's `simulation_results.csv` header: `t`; for each component its states then its
observables (`"COMP.name"`); then `Vd_Bus<id>`, `Vq_Bus<id>`, `Vterm_Bus<id>` per bus.
`delta_COI` is not written, as in Python PHPS.

---

### csv_row

*[src/sim/results.jl:27](../../src/sim/results.jl#L27)*

    csv_row(sys, t, y) -> Vector{Float64}

One row of `simulation_results.csv` at time `t` and state `y = [x; V]`.

---

### write_results_csv

*[src/sim/results.jl:52](../../src/sim/results.jl#L52)*

    write_results_csv(path, r::SimResult)

`simulation_results.csv` in Python PHPS's format (`%.15e`, comma separated, header line).

---

### write_results_jld2

*[src/sim/results.jl:73](../../src/sim/results.jl#L73)*

    write_results_jld2(path, r::SimResult)

The logged states in binary: `t`, `Y` (one column per time) and the row `names` (the
states, then `Vd_Bus<id>`, `Vq_Bus<id>` per bus).

---

### run_metadata

*[src/sim/results.jl:106](../../src/sim/results.jl#L106)*

    run_metadata(r, scenario; extra...) -> Dict

What `run.json` records: hashes of the case, scenario, source (git commit and a dirty flag)
and Manifest, the Julia and PHPS versions, the solver settings the integrator used
(`r.settings`, then `settings`) and statistics, the wall time.

---

### simulate

*[src/sim/results.jl:131](../../src/sim/results.jl#L131)*

    simulate(scenario_path; outdir = nothing, method = nothing) -> String

Run a scenario: load the case, solve the equilibrium, integrate with the scenario's solver
(`bdf1` or `ida`; `method` overrides it), and write `simulation_results.csv`,
`simulation_results.jld2` and `run.json` into `outdir` (default: `outputs/<scenario
output directory name>`). Returns the path of `run.json`.

