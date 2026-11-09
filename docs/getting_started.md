# Getting started

## Install

PHPS needs Julia 1.12. Install it with [juliaup](https://github.com/JuliaLang/juliaup), then, in the repository
folder:

```
julia --project -e "using Pkg; Pkg.instantiate()"
```

This installs the exact package versions recorded in `Manifest.toml`. The first `using PHPS` precompiles the package
and runs a short simulation workload, so that later sessions start quickly; this takes a few minutes once.

## A first EMT run

The IEEE 39-bus full model (`cases/IEEE39Bus_FullModel`) has every component physical: machines with all rotor
windings, DC and brushless exciters, steam units with IAPWS-95 properties, a hydro unit with an elastic penstock.
Its scenario applies a three-phase fault at bus 16 from 0.1 s to 0.2 s.

From the command line:

```
julia --project=. scripts/full_model_fault.jl --tstop=1
julia --project=. scripts/full_model_plots.jl
```

The first command builds the model, checks the operating point, simulates one second and writes the run to
`outputs/ph/full_model/fault_bus16_3ph_t1.0/`. The second draws its figures from the stored run, without simulating
again.

From Julia:

```julia
using PHPS

fm  = phs_full_model("cases/IEEE39Bus_FullModel")
run = phs_simulate(fm.cs, fm.x0, fm.u; tstop = 1.0, faults = fm.faults, save_dt = 1e-3)

run.t                         # saved times (s)
run.V                         # bus voltages, one row per bus (network order)
run.total.scheme_max          # largest running energy-balance error of the system
run.total.throughput          # energy moved through all ports, for scale

s   = fm.cs.s                                        # the physical model inside the controlled one
sig = phs_machine_signals(s, run.X[1:s.n, end])      # speed, angle, torque, field current per machine
phs_write_run("outputs/my_run.jld2", run; model = fm.cs)
```

[EMT simulation](emt_simulation.md) explains each step and its options.

## A first RMS run

```julia
using PHPS

simulate("cases/IEEE39Bus_PF/bus_fault_bus16_150ms.json")   # returns the path of run.json
```

`simulate` loads the scenario and its system, solves the power flow and the equilibrium, integrates with the
scenario's solver and writes `simulation_results.csv`, `simulation_results.jld2` and `run.json` under `outputs/`.
[RMS simulation](rms_simulation.md) shows the same steps one by one.

## Where results go

Everything a run writes goes under `outputs/`, which is not in git. Every run record carries the git commit of the
code that produced it, with a flag if the working tree had changes, so a result can always be traced to its code.

## Tests

```
julia --project -e "using Pkg; Pkg.test()"
```

About 10 000 checks; the suite takes about half an hour, most of it in the EMT tests.

## Next

- [Case files](case_format.md): how a system and a scenario are written.
- [Analysis tools](analysis.md): audits and interval tools on the same models.
- [API reference](api/README.md): every function.
