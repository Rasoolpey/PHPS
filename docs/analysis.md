# Analysis tools

The model equations are generic in the number type, so the same code that simulates also linearises (dual numbers)
and encloses (intervals). The tools below work on the RMS model (`DAESystem`, from `solve_equilibrium`); the EMT
model has its own energy account ([EMT simulation](emt_simulation.md#the-run-and-its-energy-account)).

## Handing the EMT model to analysis tools (`src/phsim/export.jl`)

One call gives everything an eigenvalue routine, an optimiser or another language needs about the EMT model at one
state, with every state named. Nothing is re-derived for the purpose: the rates are the simulator's own
(`phs_rhs`), the structure is `phs_structure`, the linearisation is the automatic derivative of the rates.

```julia
using PHPS
am = phs_analysis_model("cases/IEEE39Bus_FullModel")     # or phs_analysis_model(model, x0, u)
phs_export("outputs/analysis/ieee39_full_model.jld2", am)   # one file
phs_export("outputs/analysis/ieee39_full_model_csv", am)    # or a folder of CSV files for MATLAB, Python, ...
```

| Field | What it is |
|---|---|
| `names`, `component`, `kind` | per state: its name, the device it belongs to, and `machine`, `excitation`, `prime_mover`, `converter` or `network` (the network side: nodes, branches, transformers, loads, motors) |
| `in_form` | per state: whether its row is in the port-Hamiltonian form (the signal rows of controllers and the devices listed in `outside` are not) |
| `x0`, `scale`, `rate0` | the state, a typical size of each state, the rates there (zero at an operating point) |
| `A` | the linearisation at `x0` |
| `J`, `R`, `G`, `input`, `effort` | the structure at `x0`: on the rows in the form, rates = `(J - R) effort + G input` (less `Pf vf` where a steam unit has feedthrough) |
| `Q` | derivative of the efforts with respect to the states: the Hessian of the stored energy of the rows in the form |
| `ports`, `outside` | the name of each input; what is outside the form |
| `checks` | the export's own consistency checks |
| `rate`, `energy`, `structure` | functions of a state, for analyses away from `x0` (in memory only) |

`phs_structure(model, x, u)` on its own returns `J`, `R`, `G`, the efforts and the port list at any state.

**On the IEEE 39-bus full model** (519 states; `scripts/analysis/export_full_model.jl`): the model is built in about
60 s and the analysis model in about 30 s. 479 rows are in the form; the 40 outside are the controllers' signal rows
and the brushless exciters. The checks: rates at the operating point 8e-12 of the states' scale, structure against the
rates 8e-13, `J` exactly skew, `R` on the rows in the form symmetric and non-negative to rounding (-6e-17 of its
norm), `Q` symmetric to 2e-8 (scaled by the states' sizes, relative). The CSV folder is 5 MB,
the JLD2 file 9 MB.

### Any device or group of devices as its own port-Hamiltonian system

```julia
sub = phs_subsystem(am, "GENROU_4")                              # one device
sub = phs_subsystem(am, ["GENROU_4", "IEEET1_4", "IEEEG1_4"])    # a group: the machine, its exciter, its steam unit
phs_export("outputs/analysis/unit4", sub)                        # CSV files and equations.md; or a .jld2 file
print(phs_equations(sub))                                        # the equations with the states' names
```

The group is cut out of the assembled model, so its equations are the simulated ones:

    rates = (J - R) efforts + Gp (efforts of the neighbouring states) + G inputs       (rows in the form)

- `Gp` and `ports`: one interconnection port per neighbouring state whose effort acts on the group (a bus voltage for
  a machine, for example), with its value and the power it delivers at the exported state.
- `shared_loss_with`: neighbouring devices that share a loss term with the group. A steam unit and its machine's rotor
  do; the machine alone then has ports that do not conserve power, and the export says which device to add.
- `energy_separable`: zero when the group has a stored energy of its own.
- `varies`: which entries change with the group's state (the numbers written are those at the exported state); the
  functions `sub.structure`, `sub.rate` and `sub.energy` give them at any other state of the group.
- `select` may be a device name, a list of device or state names, a kind (`:machine`), state indices, or a function of
  (name, device, kind).

`equations.md` lists the efforts (linear where the energy is quadratic), then one entry per state with the lossless
exchange, the losses, the ports and the inputs kept apart, then the signal rows linearised, then the ports' powers.

On the full model (`scripts/analysis/export_component.jl`): the machine GENROU_4 alone has 8 states, 29 ports and 2
inputs, and shares loss terms with its steam unit; the whole unit (machine, exciter, steam unit) has 39 states, 35 of
them in the form, 2 ports (the two axes of its bus voltage) and 16 inputs, and every port conserves power. The
right-hand side above reproduces the simulated rates to 3e-13 of the states' scale.

**Through the MCP server.** The tool `export_model` (arguments `model_id`, and `devices` or `kind`) does the same for a
model built with `build_model` and writes to `outputs/mcp/exports/<id>/`; `get_export` reads the summary, the equations,
the states, the ports or the inputs back in bounded pieces. The end-to-end test `mcp/test/e2e.jl` covers both.

**Two analyses that use only the exported content:**

- `scripts/analysis/export_full_model.jl` lists the oscillatory modes between 0.1 and 3 Hz with their damping and the
  devices that take part most in each, from `A`, `names` and `component` alone.
- `scripts/analysis/jump_example.jl` does not load the simulator. It reads the JLD2 file, picks one machine's windings
  and rotor by name and solves a small semidefinite programme on their blocks of `A` and `Q` with JuMP (the weights
  closest to the stored energy that guarantee a given decay rate), then checks the answer by eigenvalues. It needs an environment with JuMP, a semidefinite solver (Hypatia is used) and JLD2. It
  is an illustration of the hand-over, not a study.

**File formats.** The JLD2 file holds plain arrays and strings and needs no part of this package to read. In the CSV
folder, `states.csv` has a header line and indices from 1; the matrix files (`A.csv`, `J.csv`, `R.csv`, `G.csv`,
`Q.csv`, `Pf.csv`) have no header, one row per line, at full precision; `inputs.csv`, `feedthrough.csv`,
`outside.csv` and `checks.csv` hold the rest.

Tested in `test/unit/analysis_export.jl` on two models: the checks above, `A` and `Q` against central differences in
random directions, the efforts against a directional derivative of the stored energy, the labels against the model's
layout, and the numbers of both file formats read back.

## Port-Hamiltonian audits (`src/ph/`)

Each component declares its storage (`hamiltonian`), its ports and its dissipation in its port contract
(`contracts/model_port_contracts.json`). The audits check those declarations against the equations.

| Function | What it checks |
|---|---|
| `total_hamiltonian(sys, x)`, `grad_total_hamiltonian`, `hessian_total_hamiltonian` | the assembled storage, its gradient and Hessian |
| `physical_projection(sys, x, V; contracts)` | which states are held or accounting states, and the physical coordinates that remain |
| `shifted_storage_audit(sys, x, V; projection)` | at an equilibrium: rank and definiteness of the storage Hessian on the physical coordinates, and the sign of its rate |
| `component_power(sys, x, V)`, `network_power` | per component: storage rate against the power through its declared ports |
| `power_audit_samples`, `port_power_audit(sys, samples)` | the same balance over many sampled states, summarised per component and model type |
| `port_model(c, x, u; input, output)` | the linearisation of one component from one port to another (`PortModel`) |
| `frequency_response(m, ws)`, `port_zeros`, `real_part_crossings` | its transfer function on a frequency grid, its zeros, where its real part changes sign |
| `passivity_certificate(m)` | whether the port is passive (positive real) and, if not, the frequency where it fails |
| `open_loops_model`, `multiport_margin`, `loop_margin`, `kyp_riccati` | multi-port models of a component's loops, frequency-domain storage conditions, and a quadratic storage from the KYP inequality |
| `storage_completion` and the `completion_*` functions | a storage function for the closed loop, completed from a fixed physical part (with the JuMP extension) |

## Interval and certificate tools (`src/roa/`)

All interval work uses IntervalArithmetic.jl with outward rounding. A result is reported as proved only when every
check on the box passes.

| Function | What it does |
|---|---|
| `section_model(sys, x0, V0; projection)` | the model on a section transverse to the rotor-angle symmetry, in scaled coordinates |
| `krawczyk(F, J, what, center, r)`, `enclose_equilibrium(m)` | interval Newton (Krawczyk) proof that a unique equilibrium lies in a box |
| `enclose_kcl_branch`, `jacobian_hull`, `centered_hull` | enclosures of the network solution and of the Jacobian over a box |
| `interval_cholesky`, `verified_max_eig`, `verified_min_eig`, `interval_solve` | rigorous definiteness tests, eigenvalue bounds and linear solves on interval matrices |
| `quadratic_candidate(m; P)`, `system_candidate` | Lyapunov candidates: a quadratic form from the Lyapunov equation (or a given `P`), or one built on the physical storage |
| `certify_level(c, level, eq)`, `certify_roa(c)` | all checks at one sublevel set, and the largest level at which they pass (bisection) |
| `certify_roa_taylor(c)`, `taylor_level` | the same with a Taylor-model decay bound |
| `containment_audit(m, E, V)` | every limiter decision and every contract domain clause holds on the box; fails closed |
| `write_certificate(path, record)`, `roa_check(record)` | write a certificate record as JSON; rebuild it from the record and re-prove it independently |

The interval methods of the limiter primitives, the steam tables and the machine fields are in `ivlimit.jl`,
`ivsteam.jl` and `ivfield.jl`; first-order Taylor models in `tm1.jl`.

## Electromechanical dq models (`src/emdq/`)

Reduced models for stability analysis: the machines' electromechanical states with an algebraic or dynamic network
(`emdq_pilot`, `emdq2_*`), with voltage regulators (`emdq_closed_loop_*`), in phase-free coordinates (`emdq_pf_*`),
and views of the system from a point of common coupling (`pcc*`). Each has its vector field, energy and balance
functions, and the reductions and masks used by the interval tools. See the [reference](api/emdq.md).

## JuMP programs (`ext/PHPSJuMPExt/`)

Loaded when JuMP is loaded (`using JuMP` with a solver such as SCS or COSMO): `structured_lyapunov`,
`structured_completion`, `completion_margin`, `minimal_completion`, `completion_rate_feasibility`, `decay_margin`,
`block_lyapunov_conditioning`, `bm_multiplier_feasibility`, `krasovskii_conditioning`. Each solves a semidefinite
program over a fixed sparsity pattern and returns a candidate with the solver's record. A candidate is only a
candidate until the interval tools confirm it.

## See also

- [API reference: port-Hamiltonian audits](api/ph.md), [interval and certificate tools](api/roa.md),
  [electromechanical dq models](api/emdq.md).
- `scripts/ph_audit.jl`: the audit of a case at its equilibrium.
