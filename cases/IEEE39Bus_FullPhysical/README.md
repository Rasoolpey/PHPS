# IEEE 39-bus full physical model

The IEEE 39-bus system with every device modelled from its physics, all at once, in three-phase EMT. Load it with
`PHPS.phs_full_model(dir)`.

| Generator | Bus | Model |
|---|---|---|
| G1 | 39 | the rest of the grid: a 10 000 MVA machine, constant mechanical power and field voltage |
| G3, G6 | 32, 35 | physical steam units (double reheat, real steam properties) and physical exciters |
| G10 | 30 | the hydropower plant: headrace tunnel, air-cushion surge chamber, Euler-equation Francis turbines, open surge tank, tailrace, PI governor with servomotor |
| six plants | 31, 33, 34 (two units), 36, 37, 38 | droop-controlled converters: battery, DC link, switching SiC bridge, saturating step-up transformer |

The network has travelling-wave lines; the loads are composite loads with induction motors and undervoltage relays;
faults are cleared at each phase's current zero.

| File | What |
|---|---|
| `system.json` | the network and the components' data (the same as `cases/IEEE39Bus_PF_gfm-droop-pf`) |
| `winding_circuits.json` | the four machines' winding data, by bus, on the system base |
| `transformer_cores.json` | the ratings of the converters' step-up transformers with saturating cores |
| `full_model.json` | every physical option the builder switches on |
| `fault_bus16_3ph.json` | three-phase fault at bus 16 through 0.001 pu, 0.1 s to 0.2 s |

Run (from the repository root): `julia --project=. scripts/full_physical_run.jl [tstop_s] [h_us]`. It writes the time
series and the energy account to `outputs/ph/full_physical/`. A 0.3 s run with 1 us steps takes tens of minutes.
