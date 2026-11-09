# Analysis tools

The model equations are generic in the number type, so the same code that simulates also linearises (dual numbers)
and encloses (intervals). The tools below work on the RMS model (`DAESystem`, from `solve_equilibrium`); the EMT
model has its own energy account ([EMT simulation](emt_simulation.md#the-run-and-its-energy-account)).

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
