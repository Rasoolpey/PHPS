# Inputs and outputs

Case and scenario loaders, JSON schemas, the parameter-expression reader, port contracts and the PowerFactory interface. Source: [src/io/](../../src/io).

Generated from the docstrings by `scripts/build_api_docs.jl`; do not edit by hand.

| Name | Defined in |
|---|---|
| [`CaseConfig`](#caseconfig) | `case.jl` |
| [`BusData`](#busdata) | `case.jl` |
| [`LoadData`](#loaddata) | `case.jl` |
| [`SourceData`](#sourcedata) | `case.jl` |
| [`LineData`](#linedata) | `case.jl` |
| [`ShuntData`](#shuntdata) | `case.jl` |
| [`ComponentSpec`](#componentspec) | `case.jl` |
| [`Wire`](#wire) | `case.jl` |
| [`Case`](#case) | `case.jl` |
| [`load_case`](#load_case) | `case.jl` |
| [`write_case`](#write_case) | `case.jl` |
| [`component`](#component) | `case.jl` |
| [`param`](#param) | `case.jl` |
| [`hasparam`](#hasparam) | `case.jl` |
| [`DomainClause`](#domainclause) | `contracts.jl` |
| [`ContractEntry`](#contractentry) | `contracts.jl` |
| [`ContractSet`](#contractset) | `contracts.jl` |
| [`load_contracts`](#load_contracts) | `contracts.jl` |
| [`contract_key`](#contract_key) | `contracts.jl` |
| [`contract`](#contract) | `contracts.jl` |
| [`parse_param_expr`](#parse_param_expr) | `expr.jl` |
| [`eval_param_expr`](#eval_param_expr) | `expr.jl` |
| [`param_value`](#param_value) | `expr.jl` |
| [`MODEL_TYPES`](#model_types) | `params.jl` |
| [`UnsupportedModelError`](#unsupportedmodelerror) | `params.jl` |
| [`check_model_type`](#check_model_type) | `params.jl` |
| [`component_params`](#component_params) | `params.jl` |
| [`BusFault`](#busfault) | `scenario.jl` |
| [`LineFault`](#linefault) | `scenario.jl` |
| [`OtherEvent`](#otherevent) | `scenario.jl` |
| [`SolverSettings`](#solversettings) | `scenario.jl` |
| [`Scenario`](#scenario) | `scenario.jl` |
| [`load_scenario`](#load_scenario) | `scenario.jl` |
| [`write_scenario`](#write_scenario) | `scenario.jl` |
| [`load_json_input`](#load_json_input) | `scenario.jl` |
| [`validate_json`](#validate_json) | `schema.jl` |

---

### CaseConfig

*[src/io/case.jl:9](../../src/io/case.jl#L9)*

Case-wide settings from the `config` block.

---

### BusData

*[src/io/case.jl:19](../../src/io/case.jl#L19)*

One row of a case's `Bus` table: index, name, base voltage and the stored voltage magnitude and angle.

---

### LoadData

*[src/io/case.jl:30](../../src/io/case.jl#L30)*

A `PQ` entry: constant power consumption (positive = drawn from the network).

---

### SourceData

*[src/io/case.jl:38](../../src/io/case.jl#L38)*

A `PV` or `Slack` entry.

---

### LineData

*[src/io/case.jl:52](../../src/io/case.jl#L52)*

A `Line` entry: pi line or two-winding transformer (tap and phase shift on `bus1`).

---

### ShuntData

*[src/io/case.jl:64](../../src/io/case.jl#L64)*

One row of a case's `Shunt` table: a fixed shunt admittance at a bus.

---

### ComponentSpec

*[src/io/case.jl:74](../../src/io/case.jl#L74)*

A dynamic component: its name, model type and parameters as written in the case.

---

### Wire

*[src/io/case.jl:84](../../src/io/case.jl#L84)*

A signal wire `from -> to` from the `connections` list (`"COMP.port"` strings).

---

### Case

*[src/io/case.jl:90](../../src/io/case.jl#L90)*

    Case

A loaded system file. Tables keep file order; `raw` is the parsed document.

---

### load_case

*[src/io/case.jl:142](../../src/io/case.jl#L142)*

    load_case(path; validate = true) -> Case

Read a system file. With `validate`, the document is first checked against
`cases/schema/system.schema.json`.

---

### write_case

*[src/io/case.jl:189](../../src/io/case.jl#L189)*

    write_case(path, case)

Write the case document. Reading it back gives an identical [`Case`](@ref).

---

### component

*[src/io/case.jl:196](../../src/io/case.jl#L196)*

    component(case, name) -> ComponentSpec

---

### param

*[src/io/case.jl:205](../../src/io/case.jl#L205)*

    param(spec, key[, default]) -> Float64

Numeric parameter of a component (numbers or expression strings).

---

### hasparam

*[src/io/case.jl:214](../../src/io/case.jl#L214)*

Whether the component spec `c` has the parameter `key`.

---

### DomainClause

*[src/io/contracts.jl:10](../../src/io/contracts.jl#L10)*

    DomainClause

One fail-closed domain clause. `check` is one of `state_bounds`, `parameter_conditions`,
`nonbinding_accounting_state`, `single_smooth_mode` or `bus_voltage_squared_above`. Bounds
and conditions are expression strings over the component parameters.

---

### ContractEntry

*[src/io/contracts.jl:33](../../src/io/contracts.jl#L33)*

The port contract of one model type: its status, storage and domain statements (`domain_clauses`).

---

### ContractSet

*[src/io/contracts.jl:44](../../src/io/contracts.jl#L44)*

All port contracts read from `contracts/model_port_contracts.json` by `load_contracts`.

---

### load_contracts

*[src/io/contracts.jl:69](../../src/io/contracts.jl#L69)*

    load_contracts(path = CONTRACTS_PATH) -> ContractSet

Read the port contracts. The schema version must be 2.1-reservoir-corrected.

---

### contract_key

*[src/io/contracts.jl:88](../../src/io/contracts.jl#L88)*

    contract_key(type) -> String

Contract key of a component type. Only `COMPLEXLOAD` differs (`ComplexLoad`).

---

### contract

*[src/io/contracts.jl:95](../../src/io/contracts.jl#L95)*

    contract(set, type) -> ContractEntry

    contract(c) -> ContractEntry

The component's port-contract entry (contracts/model_port_contracts.json).

---

### parse_param_expr

*[src/io/expr.jl:126](../../src/io/expr.jl#L126)*

    parse_param_expr(s::AbstractString, env = Dict{String,Float64}()) -> Float64

Evaluate a parameter expression (numbers, `+ - * /`, parentheses, `M_PI`, `pi`, `M_E`,
`sqrt`, and the names in `env`) with C evaluation order in `Float64`. Never calls `eval`.

```jldoctest
julia> PHPS.parse_param_expr("2.0 * M_PI * 60.0") == 2.0 * π * 60.0
true
```

---

### eval_param_expr

*[src/io/expr.jl:140](../../src/io/expr.jl#L140)*

    eval_param_expr(T, s, env) -> T

`parse_param_expr` in the number type `T`: every literal, constant and `env` value is
converted to `T` (a literal is first read as the nearest Float64, as the model reads its
parameters), and the arithmetic is `T`'s. With an interval type this gives an
outward-rounded enclosure of the Float64 expression's exact value (the ROA containment
audit).

---

### param_value

*[src/io/expr.jl:161](../../src/io/expr.jl#L161)*

    param_value(x) -> Float64

A numeric parameter as stored in case JSON: a number, or an expression string.

---

### MODEL_TYPES

*[src/io/params.jl:9](../../src/io/params.jl#L9)*

Component model types PHPS implements: the final PHTRUE models of Python PHPS plus
`COMPLEXLOAD`. The older models in Python PHPS `components/retired/` are not ported for now; they
stay in PHPS_Opt as reference. Cases that use them still load (`load_case` does not depend
on model types), but every model-level function rejects them.

---

### UnsupportedModelError

*[src/io/params.jl:19](../../src/io/params.jl#L19)*

A component whose model type PHPS does not implement.

---

### check_model_type

*[src/io/params.jl:29](../../src/io/params.jl#L29)*

    check_model_type(spec)

Throw [`UnsupportedModelError`](@ref) unless `spec.type` is in [`MODEL_TYPES`](@ref).

---

### component_params

*[src/io/params.jl:159](../../src/io/params.jl#L159)*

    component_params(case, spec) -> ParamDict

The parameters Python PHPS uses for a component: the case values, normalised to the system base
for synchronous machines (unless `_params_normalized` is set), plus the defaults and derived
values its constructor adds ([`type_defaults!`](@ref)). Parameters set by initialisation
(reservoir references, load `Vini`, `Pref`/`Vref`) keep their defaults here.

---

### BusFault

*[src/io/scenario.jl:9](../../src/io/scenario.jl#L9)*

Three-phase shunt fault `1/(r + jx)` at a bus on `[t_start, t_end)`.

---

### LineFault

*[src/io/scenario.jl:18](../../src/io/scenario.jl#L18)*

Fault at fraction `distance` along a line (from `bus1`), through `1/(r + jx)`.

---

### OtherEvent

*[src/io/scenario.jl:28](../../src/io/scenario.jl#L28)*

An event type PHPS reads but does not interpret yet (kept verbatim).

---

### SolverSettings

*[src/io/scenario.jl:35](../../src/io/scenario.jl#L35)*

The `solver` block of a scenario: method (`:ida`, `:bdf1` or `:rk4`), step, duration and tolerances.

---

### Scenario

*[src/io/scenario.jl:47](../../src/io/scenario.jl#L47)*

    Scenario

A loaded scenario file. `system_path` is resolved relative to the scenario file.

---

### load_scenario

*[src/io/scenario.jl:93](../../src/io/scenario.jl#L93)*

    load_scenario(path; validate = true) -> Scenario

---

### write_scenario

*[src/io/scenario.jl:115](../../src/io/scenario.jl#L115)*

    write_scenario(path, scenario)

Write the scenario document. Reading it back gives an identical [`Scenario`](@ref) (the
`system` entry is written as it was, so keep the file next to its system file).

---

### load_json_input

*[src/io/scenario.jl:123](../../src/io/scenario.jl#L123)*

    load_json_input(path) -> Case or Scenario

Load a system or scenario file, telling them apart by their top-level keys.

---

### validate_json

*[src/io/schema.jl:30](../../src/io/schema.jl#L30)*

    validate_json(kind, obj; file = "<memory>")

Validate a parsed JSON value against the `:system` or `:scenario` schema. Throws
[`SchemaError`](@ref) on failure and returns `nothing` otherwise.

