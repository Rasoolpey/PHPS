# IEEE 39-bus full model

The all-physical, three-phase EMT model of the IEEE 39-bus system, meant as its digital twin.
Load it with `PHPS.phs_full_model(dir)`.

| File | What |
|---|---|
| `system.json` | the network and the components' data: an unchanged copy of `cases/IEEE39Bus_PF/system_phtrue.json` |
| `winding_circuits.json` | PowerFactory's machine winding data after its initialisation (`pf_winding_circuits`) |
| `full_model.json` | the physical build: loads, physical exciters, physical steam units (paths A and C), the hydro unit with its elastic penstock |
| `ieee39_base.json` | a read-only probe of the PowerFactory project (element data as stored there); the zero-sequence network is built from it when unbalanced faults are needed |
| `fault_bus16_3ph.json` | the first fault: three-phase at bus 16 through 0.001 pu, 0.1-0.2 s |

Run (from the repository root): `julia --project=. scripts/full_model_fault.jl [--tstop=1]`, then
`julia --project=. scripts/full_model_plots.jl`.
