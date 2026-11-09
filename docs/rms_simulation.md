# RMS simulation

The RMS stack is a phasor simulator: the network is algebraic (a Y-bus), and the machines, controllers, loads and
converters are differential equations. It reads the same case files as the EMT simulator and gives the EMT model its
operating point.

## One call

```julia
using PHPS
simulate("cases/IEEE39Bus_PF/bus_fault_bus16_150ms.json")      # -> path of run.json
simulate(path; method = :bdf1, outdir = "outputs/my_run")       # override the solver or the folder
```

`simulate` loads the scenario and its system, solves the equilibrium, integrates with the scenario's solver and writes
`simulation_results.csv`, `simulation_results.jld2` and `run.json` into `outputs/<scenario output directory>`.

## Step by step

```julia
sc   = load_scenario("cases/IEEE39Bus_PF/bus_fault_bus16_150ms.json")
case = load_case(sc.system_path)

eq = solve_equilibrium(case, sc)          # power flow, component initialisation, Newton on the full DAE
eq.sys                                     # the assembled DAESystem
eq.x, eq.V                                 # equilibrium states and bus voltages (Vd, Vq per bus)

y0 = vcat(eq.x, eq.V)
r  = simulate_ida(eq.sys, y0; dt = sc.solver.dt, duration = sc.solver.duration, log_dt = 1e-3)
# or simulate_bdf1(eq.sys, y0; dt, duration)

write_results_csv("outputs/run/simulation_results.csv", r)
write_results_jld2("outputs/run/simulation_results.jld2", r)
```

### Loading

- `load_case(path)` reads a system file and validates it against `cases/schema/system.schema.json`; it returns a
  `Case` with typed tables (`buses`, `pq`, `pv`, `slack`, `lines`, `shunts`), the `components` specs and the
  `connections`.
- `load_scenario(path)` reads a scenario: the system file (resolved relative to the scenario), the solver settings,
  the events (`BusFault`, `LineFault`) and the output folder.
- `component(case, name)`, `param(spec, key)` and `hasparam(spec, key)` read component data; parameter values may be
  expressions such as `"2.0 * M_PI * 60.0"`, read by a safe expression reader (`param_value`), never by `eval`.

### Network and power flow

- `Network(case)`, `ybus(case)`, `ybus_dae(case)`: bus ordering and the admittance matrices.
- `solve_powerflow` (Newton-Raphson, PV/PQ/slack) runs inside `solve_equilibrium`; see the
  [power-flow reference](api/powerflow.md).

### Equilibrium

`solve_equilibrium(case[, scenario]; tol = 1e-13)` solves the power flow, initialises every component from it
(`init_from_phasor`, `converter_init`), and solves the full DAE for `(x*, V*)` by Newton. The slack machine's set-point
is adjusted so the common frequency is exactly nominal. It returns an `EquilibriumResult` with `sys`, `x`, `V`, the
residual and the iteration count.

### Solvers

| Function | Method |
|---|---|
| `simulate_ida(sys, y0; dt, duration, log_dt, rtol, atol)` | SUNDIALS IDA, variable order and step, restarted at every fault switch |
| `simulate_bdf1(sys, y0; dt, duration, log_dt, ...)` | fixed-step backward Euler with Newton on the full DAE |

Both return a `SimResult`: logged times `t`, logged states `Y` (one column per time, `[x; Vd_1, Vq_1, ...]`), the
system, solver statistics and the settings actually used.

### Results

- `write_results_csv(path, r)`: one row per logged time, full precision, with named columns (`csv_columns`).
- `write_results_jld2(path, r)`: the same as arrays, with state names.
- `run_metadata(r, scenario)`: what `run.json` records: hashes of the case, scenario, source (git commit and dirty
  flag) and Manifest, versions, solver settings and statistics, wall time.

## Components

Every RMS model is an `AbstractComponent` with the interface listed in the [README](../README.md#models):
`state_names`, `rhs!`, `outputs!`, `injection`, `hamiltonian`, `modes` and the others. `build_component(case, spec)`
builds one from its case entry. The models: `GENROU_PHTRUE`, `GENSAL_PHTRUE`, `IEEET1_PHTRUE`, `IEEEG1_PHTRUE`,
`IEEEG3_PHTRUE`, `COMPLEXLOAD`, `GFL_PHTRUE`, `GFM_VSM_PHTRUE`, `GFM_DROOP_PHTRUE`, `GFM_VOC_PHTRUE`
([reference](api/components.md)).

## See also

- [API reference](api/README.md): inputs, network, power flow, components, assembly, initialisation, simulation.
- `scripts/run_base_fault.jl`: a complete RMS run with plots.
