# EMT simulation

The EMT simulator (`src/phsim/`) models the grid as one port-Hamiltonian system in dq0 coordinates: line and
transformer currents, node charges, every machine winding's flux and every rotor's momentum are states. It is
integrated with the implicit midpoint rule, which keeps each component's energy balance to rounding error, and every
run carries that energy account.

## The steps

```julia
using PHPS

# 1. a case and its equilibrium (the RMS stack gives the operating point)
case = load_case("cases/IEEE39Bus_FullModel/system.json")
eq   = solve_equilibrium(case)

# 2. the EMT model at that operating point, with the case's exciters and governors
cs, x0, u, report = phs_controlled(eq; loads = :series_all)

# 3. a fault and a run
f   = PHFault(16, 0.001, 0.1, 0.2)                  # bus 16, 0.001 pu, from 0.1 s to 0.2 s
run = phs_simulate(cs, collect(x0), u; tstop = 1.0, faults = [f], save_dt = 1e-3)

# 4. store it
phs_write_run("outputs/fault16.jld2", run; model = cs)
```

`phs_full_model(dir)` does steps 1 and 2 for a case folder that has a `full_model.json` (every physical option
switched on as that file declares) and returns `(; cs, x0, u, eq, sc, case, faults, spec)`, with the scenario's
faults already as `PHFault`s.

## Building a model

### `phs_equilibrium`: the network and machines only

`phs_equilibrium(eq; loads, circuits, const_power)` returns the bare port-Hamiltonian model `s::PHSystem` (network,
loads, machines, rotors), its balanced equilibrium `x`, the inputs `u::PHInput` (shaft torques, field voltages,
fault conductances) and a report with the residual of the vector field. Without controllers, the shaft torques and
field voltages stay at their equilibrium values.

### `phs_controlled`: with exciters and governors

`phs_controlled(eq; kw...)` adds the case's exciters and governors as control laws, each found through its `syn`
parameter and initialised at its machine's steady state, and returns `(cs::PHControlled, x0, u, report)`. Its state
is the physical state followed by the controller states (`phs_state_names(cs)` names them). Keywords:

| Keyword | Values | Meaning |
|---|---|---|
| `loads` | `:parallel` (default), `:series`, `:series_all`, `:composite` | constant impedance as parallel R and L, or series R-L (inductor-only nodes reduced exactly); `:composite` adds induction motors and static loads |
| `circuits` | `Dict(bus => winding data)` or `nothing` | full machine winding data (`Ll, Lad, Laq, Lfd, L1d, L1q, L2q, Ra, Rfd, R1d, R1q, R2q, H`, system base); `nothing` derives the windings from the case's GENROU/GENSAL data |
| `lines` | `:lumped` (default), `:travelling` | dynamic pi sections, or travelling-wave lines |
| `dq0` | `(; zero = :physical or :powerfactory, zero_pf_json)` | the zero-sequence network (grounding and transformer connections), needed for faults to ground |
| `dc_exciter` | `(; n, field_ratio)` | a physical self-excited DC shunt exciter on every machine whose IEEET1 has `KE < 0` |
| `brushless` | `(; KC, KD, sat, field_share)` | a physical brushless exciter (alternator and rotating rectifier) on every machine whose IEEET1 has `KE > 0` |
| `steam` | `(; paths, r_v, rh_dT, tau_n, lowflow, cold_reheat, volumes, valve, tables)` | a physical steam unit (IAPWS-95 properties, reheat) in place of each IEEEG1 named in `paths` |
| `hydro` | `(; Tw, Te, N, fp)` | a physical hydro unit with an elastic penstock in place of the IEEEG3; `Tw = :match` matches the case's IEEEG3 at the operating point |
| `converter_Ts`, `converter_dc`, `converter_umax`, `cores` | see the docstring | sampled converter controllers, the converters' DC side, their voltage limit, saturating step-up transformer cores |
| `nsec`, `composite` | integer; named tuple | sections per travelling-wave line; composite-load mix |
| `fixed_units` | `(; steam, hydro_Tw)` | keep turbine hardware from another build (`phs_fixed_units(cs)`), so a model built at another operating point is the same plant |

The values used for the IEEE 39-bus full model are in `cases/IEEE39Bus_FullModel/full_model.json`. Each keyword is
described in full in the docstring (`?phs_controlled`, or the [API reference](api/phsim.md#phs_controlled)).

`ph_system(sys; ...)` is the lower-level constructor that both use: it assembles the model of an RMS `DAESystem`
without solving for an equilibrium, and also takes `relays`.

## Faults

```julia
PHFault(bus, R, t_on, t_off; kind = :abc, Rg = R, clearing = :instant, topen = nothing)
```

| Argument | Meaning |
|---|---|
| `bus`, `R` | faulted bus and fault resistance (pu) |
| `t_on`, `t_off` | fault inception and the clearing command (s) |
| `kind` | `:abc` three-phase, `:abcg` three-phase to ground, `:ag`, `:bg`, `:cg` phase to ground, `:ab`, `:bc`, `:ca` phase to phase, `:abg`, `:bcg`, `:cag` two phases to ground |
| `Rg` | resistance to ground, for the kinds with `g` |
| `clearing` | `:instant`: every phase opens at `t_off`; `:current_zero`: each phase opens at its own first current zero after `t_off` |

Faults to ground need a model built with the zero-sequence network (`dq0`). With `:current_zero`, `phs_simulate`
locates each phase's zero (`phs_resolve_clearing`) unless `topen` gives the opening times.

## Running

```julia
run = phs_simulate(cs, x0, u; tstop, faults = PHFault[], hscale = 1, h = nothing, save_dt = 1e-3,
                   grid = nothing, sample_at = nothing, stops = (), record = (), monitor = nothing, tol = 1e-13)
```

- **Step.** Each segment between switching times uses a fixed step: `h` if given, otherwise `hscale` times
  `phs_step_rule` (20 steps per period of the fastest oscillation of the linearisation, and fast decays damped
  monotonically). Nothing changes a state at an event; only the fault conductance changes.
- **Saving.** Every `save_dt` and at every segment end. `grid` saves exactly on multiples of a time step (for
  comparison with another tool's samples); `sample_at` saves at exactly the given times; `stops` ends segments at
  further times.
- **Records.** `record = ("efd" => (x, u) -> ..., ...)` declares quantities evaluated at every saved time and kept in
  `run.records`. The full state is always kept, so anything else can be computed afterwards from `run.X`.
- **Newton tolerance.** `tol` (scaled residual) is `1e-13` by default; at that value every component's discrete
  energy balance meets `1e-10` of its throughput on the IEEE 39-bus system.

`phs_simulate_bdf` integrates the same model with SUNDIALS CVODE (variable-order BDF) as an independent check.

## The run and its energy account

`phs_simulate` returns a `PHRun`:

| Field | Content |
|---|---|
| `t`, `X`, `V` | saved times, states (one column per time), bus voltages (rows in network order) |
| `names`, `kind` | the components of the energy account: lines, transformers, loads, machines, rotors, faults, physical exciters |
| `energy`, `loss`, `port` | per component and saved time: stored energy, dissipated power, power received through its ports |
| `supply`, `H` | boundary power (shaft and field) and total stored energy |
| `scheme`, `scheme_max` | the discrete energy balance per component, and the largest magnitude its running sum reached |
| `closure`, `simpson`, `throughput`, `moved` | independent checks of the balance, the energy moved through the ports, the variation of storage |
| `total` | the same accumulated quantities for the whole system |
| `segments`, `records` | per-segment step data; the declared records |

At any state, `phs_ledger(cs, x, u)` gives the account by component (`energy`, `loss`, `port`, `rate`, with
`rate = port - loss` for each), and `phs_energy(s, x)` the total stored energy.

## Signals

| Function | Gives, at a state `x` |
|---|---|
| `phs_machine_signals(s, x)` | per machine: speed, rotor angle, electromagnetic torque, field current, stator currents and fluxes |
| `phs_control_signals(cs, x)` | per machine: exciter output `Efd`, turbine power `Pm`, field voltage, power a physical exciter takes from the shaft |
| `phs_converter_signals(cs, x, u)` | per converter: speed, angle, active and reactive power, injected current, DC-side quantities |
| `phs_terminal_voltage(s, x, u)` | the voltage of every generator bus merged with its step-up transformer |
| `phs_phases(z, theta)` | the three phase values of a dq0 phasor at frame angle `theta` |

`s = cs.s` is the physical model inside a controlled one; pass it the physical part of the state, `x[1:s.n]`.

## Stored runs

`phs_write_run(path, run; model, meta, states = true)` writes everything to one JLD2 file as plain arrays and
dictionaries: times, states with their names, bus voltages with bus ids, the full energy account, the records, the git
commit (with a flag for uncommitted changes) and `meta`. `phs_read_run(path)` reads it back as a dictionary; any
JLD2 or HDF5 reader can open it too. With `states = false` the state is left out for a smaller file.

## See also

- [API reference: EMT simulator](api/phsim.md), all functions of `src/phsim/`.
- [API reference: physical steam and shaft models](api/physical.md).
- `scripts/full_model_fault.jl` and `scripts/ph_ieee39_fault.jl` for complete worked runs.
