# Assembly (RMS)

Wiring, the DAE residual and its sparsity pattern. Source: [src/assembly/](../../src/assembly).

Generated from the docstrings by `scripts/build_api_docs.jl`; do not edit by hand.

| Name | Defined in |
|---|---|
| [`DAESystem`](#daesystem) | `dae.jl` |
| [`nalg`](#nalg) | `dae.jl` |
| [`assemble`](#assemble) | `dae.jl` |
| [`DAEWorkspace`](#daeworkspace) | `dae.jl` |
| [`dae_residual!`](#dae_residual) | `dae.jl` |
| [`dae_residual`](#dae_residual) | `dae.jl` |
| [`jacobian_pattern`](#jacobian_pattern) | `sparsity.jl` |
| [`InputSource`](#inputsource) | `wiring.jl` |
| [`resolve_wiring`](#resolve_wiring) | `wiring.jl` |

---

### DAESystem

*[src/assembly/dae.jl:22](../../src/assembly/dae.jl#L22)*

    DAESystem

An assembled system: components, state layout, wiring, network constants and events.
States `x` are in Python PHPS order (components in case order, then `delta_COI`); bus voltages
are `V = [Vd_1, Vq_1, Vd_2, Vq_2, ...]` in network order.

---

### nalg

*[src/assembly/dae.jl:53](../../src/assembly/dae.jl#L53)*

Number of algebraic variables (2 per bus).

---

### assemble

*[src/assembly/dae.jl:76](../../src/assembly/dae.jl#L76)*

    assemble(case[, scenario]; init_params = Dict(), phps_rounding = true) -> DAESystem

Build the DAE of a case, with the bus faults of `scenario`. `init_params` maps component
names to parameters that initialisation sets (until P6, these come from Python PHPS). The
constant-impedance loads use the solved power-flow voltages, as Python PHPS's simulation does.

---

### DAEWorkspace

*[src/assembly/dae.jl:207](../../src/assembly/dae.jl#L207)*

    DAEWorkspace(sys, T = Float64)

Buffers for evaluating the residual of `sys` with element type `T` without allocating.

---

### dae_residual!

*[src/assembly/dae.jl:232](../../src/assembly/dae.jl#L232)*

    dae_residual!(f, g, sys, ws::DAEWorkspace, x, V, fault_active[, logs]) -> (f, g)

Allocation-free form: `fault_active[k]` switches the k-th fault shunt of `sys.faults`.
`logs = (out = Vector{ModeLog}, step = Vector{ModeLog})`, one log per component, records
the branch decisions of the output and step kernels (the ROA containment audit); the
default `nothing` records nothing.

    dae_residual!(f, g, sys, x, V; faults_on = false) -> (f, g)

`f = x'` (differential right-hand side, length `n_diff`) and the algebraic residual `g`
(length `2 nbus`, `[KCL_d, KCL_q]` per bus) at states `x` and bus voltages
`V = [Vd_1, Vq_1, ...]`. With `faults_on`, every fault shunt of the scenario is applied.
Python PHPS's residual is `[x' - f; g]`.

---

### dae_residual

*[src/assembly/dae.jl:344](../../src/assembly/dae.jl#L344)*

    dae_residual(sys, x, V; faults_on = false) -> (f, g)

---

### jacobian_pattern

*[src/assembly/sparsity.jl:14](../../src/assembly/sparsity.jl#L14)*

    jacobian_pattern(sys) -> SparseMatrixCSC{Bool,Int}

Pattern of `d[f; g] / d[x; V]` (rows: `n_diff` derivatives then `2 nbus` KCL residuals;
columns: `n_diff` states then `[Vd_1, Vq_1, ...]`). Valid in every limiter mode and with
the faults on.

---

### InputSource

*[src/assembly/wiring.jl:19](../../src/assembly/wiring.jl#L19)*

    InputSource

Where one component input is read from: `kind`, and the constant `value`, the bus or
component `index`, and the output `port` (1-based), as the kind needs.

---

### resolve_wiring

*[src/assembly/wiring.jl:92](../../src/assembly/wiring.jl#L92)*

    resolve_wiring(case, comps, net; pre_params) -> Vector{Vector{InputSource}}

The source of every input of every component (`comps` in case order). `pre_params` are
the parameter dictionaries before initialisation (read by `PARAM:` wires); the refresh
after initialisation uses the components' own (initialised) parameters.

