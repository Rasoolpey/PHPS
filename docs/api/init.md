# Initialisation (RMS)

Component initialisation from the power flow and the equilibrium of the full DAE. Source: [src/init/](../../src/init).

Generated from the docstrings by `scripts/build_api_docs.jl`; do not edit by hand.

| Name | Defined in |
|---|---|
| [`MachineTargets`](#machinetargets) | `components.jl` |
| [`init_from_phasor`](#init_from_phasor) | `components.jl` |
| [`converter_init`](#converter_init) | `components.jl` |
| [`converter_current`](#converter_current) | `components.jl` |
| [`init_from_targets`](#init_from_targets) | `components.jl` |
| [`first_pass`](#first_pass) | `components.jl` |
| [`EquilibriumResult`](#equilibriumresult) | `equilibrium.jl` |
| [`component_io`](#component_io) | `equilibrium.jl` |
| [`solve_equilibrium`](#solve_equilibrium) | `equilibrium.jl` |

---

### MachineTargets

*[src/init/components.jl:9](../../src/init/components.jl#L9)*

    MachineTargets

Steady-state quantities of a machine at its power-flow operating point, used to
initialise its exciter and governor.

---

### init_from_phasor

*[src/init/components.jl:37](../../src/init/components.jl#L37)*

    init_from_phasor(c, V, I) -> (x, targets::MachineTargets)

Machine states at terminal voltage `V` and current `I` (network frame, pu).

---

### converter_init

*[src/init/components.jl:83](../../src/init/components.jl#L83)*

    converter_init(c, V, I) -> (x, params::Dict, targets::MachineTargets)

A grid-forming converter's states at terminal voltage `V` and injected current `I`, and the
parameters Python PHPS's initialisation sets (`Efd0`, `Tm0` and the model's set-points).

---

### converter_current

*[src/init/components.jl:164](../../src/init/components.jl#L164)*

    converter_current(c, x, V) -> ComplexF64

The current a converter injects at bus voltage `V`: (u_out - V) / (j Zseries), with u_out
its Norton source (Python PHPS rebalance_for_bus_voltage).

---

### init_from_targets

*[src/init/components.jl:181](../../src/init/components.jl#L181)*

    init_from_targets(c, t::MachineTargets) -> (x, params::Dict)

Exciter or governor states at its machine's steady state, and the parameters
initialisation sets (set-points and reservoir references).

---

### first_pass

*[src/init/components.jl:214](../../src/init/components.jl#L214)*

    first_pass(case; pf = solve_powerflow(case)) -> (x, init_params, targets)

Python PHPS `Initializer.run` for the ported models: machines from their power-flow phasors (power
split among the machines of a bus as Python PHPS does), then exciters and governors from their
machine's targets, loads at rest with `V0 = Vini =` the power-flow voltage. `x` has the
state layout of [`assemble`](@ref) (`delta_COI = 0`); `init_params` maps component names to
the parameters initialisation sets.

---

### EquilibriumResult

*[src/init/equilibrium.jl:16](../../src/init/equilibrium.jl#L16)*

    EquilibriumResult

`sys` (assembled with the initialised parameters), the equilibrium `x`, `V`, the initial
guess `x_initial` from the first pass, the max-abs residual, and the Newton iterations.

---

### component_io

*[src/init/equilibrium.jl:83](../../src/init/equilibrium.jl#L83)*

    component_io(sys, x, V) -> (inputs, outputs)

Every component's inputs and outputs at `(x, V)`, as the residual evaluates them (after the
step pass).

---

### solve_equilibrium

*[src/init/equilibrium.jl:175](../../src/init/equilibrium.jl#L175)*

    solve_equilibrium(case[, scenario]; tol = 1e-13, maxiter = 20, phps_rounding = true)
        -> EquilibriumResult

Initialise the case (power flow, component initialisation) and solve the full DAE for its
equilibrium `(x*, V*)`. The set-points come from the power flow; the slack machine's
set-point (its governor's `Pref`, or its `Tm0`) is adjusted so the common frequency is
exactly nominal, as the power-flow slack absorbs the losses.

