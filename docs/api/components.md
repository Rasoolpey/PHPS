# Components (RMS)

The component interface, the limiter primitives and the RMS models: machines, exciters, governors, loads, converters. Source: [src/components/](../../src/components).

Generated from the docstrings by `scripts/build_api_docs.jl`; do not edit by hand.

| Name | Defined in |
|---|---|
| [`GFL_PHTRUE`](#gfl_phtrue) | `converters/gfl.jl` |
| [`GFM_DROOP_PHTRUE`](#gfm_droop_phtrue) | `converters/gfm_droop.jl` |
| [`GFM_VOC_PHTRUE`](#gfm_voc_phtrue) | `converters/gfm_voc.jl` |
| [`GFM_VSM_PHTRUE`](#gfm_vsm_phtrue) | `converters/gfm_vsm.jl` |
| [`IEEET1_PHTRUE`](#ieeet1_phtrue) | `exciters/ieeet1.jl` |
| [`IEEEG1_PHTRUE`](#ieeeg1_phtrue) | `governors/ieeeg1.jl` |
| [`IEEEG3_PHTRUE`](#ieeeg3_phtrue) | `governors/ieeeg3.jl` |
| [`AbstractComponent`](#abstractcomponent) | `interface.jl` |
| [`model_type`](#model_type) | `interface.jl` |
| [`state_names`](#state_names) | `interface.jl` |
| [`input_names`](#input_names) | `interface.jl` |
| [`output_names`](#output_names) | `interface.jl` |
| [`nstates`](#nstates) | `interface.jl` |
| [`params`](#params) | `interface.jl` |
| [`param_dict`](#param_dict) | `interface.jl` |
| [`ports`](#ports) | `interface.jl` |
| [`rhs!`](#rhs) | `interface.jl` |
| [`outputs!`](#outputs) | `interface.jl` |
| [`step_outputs!`](#step_outputs) | `interface.jl` |
| [`modes`](#modes) | `interface.jl` |
| [`hamiltonian`](#hamiltonian) | `interface.jl` |
| [`grad_hamiltonian!`](#grad_hamiltonian) | `interface.jl` |
| [`grad_hamiltonian`](#grad_hamiltonian) | `interface.jl` |
| [`lag_states`](#lag_states) | `interface.jl` |
| [`bus`](#bus) | `interface.jl` |
| [`component_role`](#component_role) | `interface.jl` |
| [`injection`](#injection) | `interface.jl` |
| [`norton_admittance`](#norton_admittance) | `interface.jl` |
| [`default_contracts`](#default_contracts) | `interface.jl` |
| [`build_component`](#build_component) | `interface.jl` |
| [`with_params`](#with_params) | `interface.jl` |
| [`COMPLEXLOAD`](#complexload) | `loads/complexload.jl` |
| [`GENROU_PHTRUE`](#genrou_phtrue) | `machines/genrou.jl` |
| [`rotor_structure`](#rotor_structure) | `machines/genrou.jl` |
| [`GENSAL_PHTRUE`](#gensal_phtrue) | `machines/gensal.jl` |
| [`observable_names`](#observable_names) | `observables.jl` |
| [`observable_values!`](#observable_values) | `observables.jl` |
| [`NoModes`](#nomodes) | `primitives.jl` |
| [`ModeLog`](#modelog) | `primitives.jl` |
| [`UndecidedBranch`](#undecidedbranch) | `primitives.jl` |

---

### GFL_PHTRUE

*[src/components/converters/gfl.jl:51](../../src/components/converters/gfl.jl#L51)*

Grid-following converter, minimal RMS form (RMS stack).

---

### GFM_DROOP_PHTRUE

*[src/components/converters/gfm_droop.jl:60](../../src/components/converters/gfm_droop.jl#L60)*

Grid-forming droop converter (RMS stack), the counterpart of PowerFactory's `gfm_droop_pf` model, with a one-way source reservoir.

---

### GFM_VOC_PHTRUE

*[src/components/converters/gfm_voc.jl:55](../../src/components/converters/gfm_voc.jl#L55)*

Grid-forming Andronov-Hopf virtual-oscillator converter (RMS stack), the counterpart of PowerFactory's `gfm_voc_pf` model, with a one-way source reservoir.

---

### GFM_VSM_PHTRUE

*[src/components/converters/gfm_vsm.jl:53](../../src/components/converters/gfm_vsm.jl#L53)*

Grid-forming virtual synchronous machine (RMS stack), the counterpart of PowerFactory's `gfm_vsm_pf` model, with a one-way source reservoir.

---

### IEEET1_PHTRUE

*[src/components/exciters/ieeet1.jl:33](../../src/components/exciters/ieeet1.jl#L33)*

IEEE Type-1 (DC1A) exciter (RMS stack) with a field-supply reservoir account.

---

### IEEEG1_PHTRUE

*[src/components/governors/ieeeg1.jl:39](../../src/components/governors/ieeeg1.jl#L39)*

IEEE steam-turbine governor IEEEG1 (RMS stack) with a steam-reservoir account.

---

### IEEEG3_PHTRUE

*[src/components/governors/ieeeg3.jl:36](../../src/components/governors/ieeeg3.jl#L36)*

IEEE hydro governor IEEEG3 (PowerFactory `pcu_IEEEG3`; RMS stack) with a headrace-reservoir account.

---

### AbstractComponent

*[src/components/interface.jl:20](../../src/components/interface.jl#L20)*

Supertype of every component model.

---

### model_type

*[src/components/interface.jl:23](../../src/components/interface.jl#L23)*

Model type name as used in case files (e.g. `"GENROU_PHTRUE"`).

---

### state_names

*[src/components/interface.jl:25](../../src/components/interface.jl#L25)*

State names, in Python PHPS order.

    state_names(sys) -> Vector{String}

`"COMPONENT.state"` for every differential state, then `"delta_COI"`.

---

### input_names

*[src/components/interface.jl:27](../../src/components/interface.jl#L27)*

Input port names, in Python PHPS order.

---

### output_names

*[src/components/interface.jl:29](../../src/components/interface.jl#L29)*

Output port names, in Python PHPS order.

---

### nstates

*[src/components/interface.jl:32](../../src/components/interface.jl#L32)*

Number of states, inputs and outputs of a component.

---

### params

*[src/components/interface.jl:37](../../src/components/interface.jl#L37)*

Typed parameters of a component.

---

### param_dict

*[src/components/interface.jl:39](../../src/components/interface.jl#L39)*

The processed parameter dictionary the component was built from.

---

### ports

*[src/components/interface.jl:44](../../src/components/interface.jl#L44)*

    ports(c) -> (in = [...], out = [...])

---

### rhs!

*[src/components/interface.jl:49](../../src/components/interface.jl#L49)*

    rhs!(dx, c, x, u[, p]) -> dx

State derivatives. Generic in the element types of `x` and `u`; does not allocate.

---

### outputs!

*[src/components/interface.jl:56](../../src/components/interface.jl#L56)*

    outputs!(y, c, x, u[, p]) -> y

The output kernel: the port outputs the network (KCL) and the other components read.

---

### step_outputs!

*[src/components/interface.jl:63](../../src/components/interface.jl#L63)*

    step_outputs!(y, c, x, u[, p]) -> y

The outputs after Python PHPS's step kernel has run: `outputs!` first, then the entries the step
overwrites (terminal power, dq currents, ...). These are what Python PHPS logs.

---

### modes

*[src/components/interface.jl:76](../../src/components/interface.jl#L76)*

    modes(c, x, u[, p]; kernel = :step) -> Vector{Bool}

The state-dependent branch decisions of the step (or `:out`) kernel, in evaluation order.

---

### hamiltonian

*[src/components/interface.jl:94](../../src/components/interface.jl#L94)*

    hamiltonian(c, x[, p])

Declared storage `H(x)`.

Circuit storage in (E', psi) coordinates: `H (omega-1)^2` plus the d- and q-axis magnetic
energies (see Python PHPS GenRouPHS.hamiltonian).

Kinetic plus reduced-order rotor magnetic storage (Python PHPS GenSal.hamiltonian).

---

### grad_hamiltonian!

*[src/components/interface.jl:102](../../src/components/interface.jl#L102)*

    grad_hamiltonian!(g, c, x[, p]) -> g

---

### grad_hamiltonian

*[src/components/interface.jl:108](../../src/components/interface.jl#L108)*

    grad_hamiltonian(c, x[, p]) -> Vector

---

### lag_states

*[src/components/interface.jl:114](../../src/components/interface.jl#L114)*

    lag_states(c) -> Vector{Tuple{Int,Float64}}

The component's first-order lag states `(j, T)`: `x_j' = (a - x_j) / T`, with `T` the
(guarded) time constant its kernel divides by (measurement lags, filters). The equilibrium
solve settles them exactly on their equations in floating point.

---

### bus

*[src/components/interface.jl:123](../../src/components/interface.jl#L123)*

Bus id of a network-attached component, `nothing` otherwise.

---

### component_role

*[src/components/interface.jl:126](../../src/components/interface.jl#L126)*

    component_role(c) -> Symbol

`:generator`, `:exciter`, `:governor` or `:load`, as Python PHPS's `component_role` (it decides the
wiring refresh after initialisation and the centre-of-inertia members).

---

### injection

*[src/components/interface.jl:134](../../src/components/interface.jl#L134)*

    injection(c, x, V[, p]) -> (I_re, I_im)

Current the component injects into its bus, in the network (RI) frame, for bus voltage
`V = (Vd, Vq)`. For a Norton source this is the Norton current; its admittance is
[`norton_admittance`](@ref) and sits in the Y-bus.

Norton current (network frame), independent of the bus voltage.

Norton current (network frame), independent of the bus voltage.

Correction current into the bus for bus voltage `V = (Vd, Vq)` (depends on V).

Norton current (network frame): state-only, independent of the bus voltage.

Norton current (network frame): state-only, independent of the bus voltage.

Norton current (network frame): state-only, independent of the bus voltage.

The injected current (network frame): the states, independent of the bus voltage; no Norton admittance.

---

### norton_admittance

*[src/components/interface.jl:143](../../src/components/interface.jl#L143)*

    norton_admittance(c) -> Union{Nothing, ComplexF64}

Admittance the component adds to the Y-bus diagonal at its bus, or `nothing`.

---

### default_contracts

*[src/components/interface.jl:160](../../src/components/interface.jl#L160)*

The contract set in `contracts/`, loaded once.

---

### build_component

*[src/components/interface.jl:175](../../src/components/interface.jl#L175)*

    build_component(case, spec) -> AbstractComponent
    build_component(type, name, params::ParamDict) -> AbstractComponent

Build a model from a case entry, with the parameters Python PHPS would use
([`component_params`](@ref)).

---

### with_params

*[src/components/interface.jl:191](../../src/components/interface.jl#L191)*

    with_params(c, overrides) -> component

A copy of `c` with some parameters replaced (e.g. the references set by initialisation).

---

### COMPLEXLOAD

*[src/components/loads/complexload.jl:34](../../src/components/loads/complexload.jl#L34)*

PowerFactory General Load (`ElmLod`/`TypLod`; RMS stack) with a filtered voltage dependence and the out-of-range multiplier `k(V)`. Contract key `ComplexLoad`.

---

### GENROU_PHTRUE

*[src/components/machines/genrou.jl:35](../../src/components/machines/genrou.jl#L35)*

Sixth-order round-rotor synchronous machine (RMS stack), with the field current `i_fd` exposed to the exciter's field-supply account. Built from a case component of type `GENROU_PHTRUE` by `build_component`.

---

### rotor_structure

*[src/components/machines/genrou.jl:185](../../src/components/machines/genrou.jl#L185)*

    rotor_structure(c) -> NamedTuple or nothing

The rotor circuits of a machine as a linear port system, restating its step kernel:
`z' = A z + B_f Efd + B_s [id, iq]` for the rotor states `z = x[states]`, with the magnetic
storage `H_mag = z' Q z / 2` (the declared storage minus `H (omega - 1)^2`) and the
subtransient flux the stator equations use, `(psi_d'', psi_q'') = Psi z`. `nothing` for
models whose rotor is not linear in these terms (GENSAL: saturation). The port-power audit
checks this form against the model's right-hand side at every sample.

---

### GENSAL_PHTRUE

*[src/components/machines/gensal.jl:33](../../src/components/machines/gensal.jl#L33)*

Fifth-order salient-pole synchronous machine (RMS stack; no q-axis transient winding), with the field current `i_fd` exposed.

---

### observable_names

*[src/components/observables.jl:6](../../src/components/observables.jl#L6)*

    observable_names(c) -> Vector{String}

Names of the component's logged signals, in Python PHPS's column order.

---

### observable_values!

*[src/components/observables.jl:13](../../src/components/observables.jl#L13)*

    observable_values!(v, c, x, u, y) -> v

Values of [`observable_names`](@ref) at states `x`, inputs `u` and outputs `y`.

---

### NoModes

*[src/components/primitives.jl:21](../../src/components/primitives.jl#L21)*

Recorder that records nothing (used by `rhs!` and `outputs!`).

---

### ModeLog

*[src/components/primitives.jl:24](../../src/components/primitives.jl#L24)*

Recorder that keeps every state-dependent decision, in evaluation order.

---

### UndecidedBranch

*[src/components/primitives.jl:33](../../src/components/primitives.jl#L33)*

    UndecidedBranch

Thrown by a branch primitive when its operands do not decide the branch (an interval that
straddles the switching surface).

