# Power flow

Newton-Raphson AC power flow. Source: [src/powerflow/](../../src/powerflow).

Generated from the docstrings by `scripts/build_api_docs.jl`; do not edit by hand.

| Name | Defined in |
|---|---|
| [`PowerFlowSpec`](#powerflowspec) | `newton.jl` |
| [`PowerFlowResult`](#powerflowresult) | `newton.jl` |
| [`bus_power`](#bus_power) | `newton.jl` |
| [`solve_powerflow`](#solve_powerflow) | `newton.jl` |

---

### PowerFlowSpec

*[src/powerflow/newton.jl:10](../../src/powerflow/newton.jl#L10)*

    PowerFlowSpec

Power-flow data in network bus order: bus types, net injection specifications
(generation minus load) and the starting point (flat, with PV/slack magnitudes and the slack
angle from the case).

---

### PowerFlowResult

*[src/powerflow/newton.jl:59](../../src/powerflow/newton.jl#L59)*

    PowerFlowResult

---

### bus_power

*[src/powerflow/newton.jl:78](../../src/powerflow/newton.jl#L78)*

    bus_power(Y, V, theta) -> Vector{Complex}

Complex power injected at each bus, `S = V .* conj(Y * V)`.

---

### solve_powerflow

*[src/powerflow/newton.jl:88](../../src/powerflow/newton.jl#L88)*

    solve_powerflow(case; tol = 1e-6, max_iter = 20) -> PowerFlowResult

Solve the power flow. If the case sets `config.skip_pf_solve`, the voltages are taken from
`Bus` `v0`/`a0` instead (Python PHPS `load_bus_overrides`).

