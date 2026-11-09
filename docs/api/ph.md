# Port-Hamiltonian audits

Storage functions, structure checks, port passivity, dissipativity, power balances and storage completion. Source: [src/ph/](../../src/ph).

Generated from the docstrings by `scripts/build_api_docs.jl`; do not edit by hand.

| Name | Defined in |
|---|---|
| [`PhysicalProjection`](#physicalprojection) | `audit.jl` |
| [`certificate_only`](#certificate_only) | `audit.jl` |
| [`reservoir_states`](#reservoir_states) | `audit.jl` |
| [`physical_projection`](#physical_projection) | `audit.jl` |
| [`shifted_storage_audit`](#shifted_storage_audit) | `audit.jl` |
| [`completion_problem`](#completion_problem) | `completion.jl` |
| [`completion_pattern`](#completion_pattern) | `completion.jl` |
| [`invariant_closure`](#invariant_closure) | `completion.jl` |
| [`StorageCompletion`](#storagecompletion) | `completion.jl` |
| [`storage_completion`](#storage_completion) | `completion.jl` |
| [`completion_energy`](#completion_energy) | `completion.jl` |
| [`completion_energy_state`](#completion_energy_state) | `completion.jl` |
| [`completion_hessian_check`](#completion_hessian_check) | `completion.jl` |
| [`ComponentPort`](#componentport) | `component_ports.jl` |
| [`component_port`](#component_port) | `component_ports.jl` |
| [`port_transfer`](#port_transfer) | `component_ports.jl` |
| [`port_hermitian_scan`](#port_hermitian_scan) | `component_ports.jl` |
| [`port_witness_interval`](#port_witness_interval) | `component_ports.jl` |
| [`polar_transfer`](#polar_transfer) | `component_ports.jl` |
| [`polar_scan`](#polar_scan) | `component_ports.jl` |
| [`injected_current`](#injected_current) | `component_ports.jl` |
| [`MultiPortModel`](#multiportmodel) | `dissipativity.jl` |
| [`open_loops_model`](#open_loops_model) | `dissipativity.jl` |
| [`multiport_margin`](#multiport_margin) | `dissipativity.jl` |
| [`port_margin`](#port_margin) | `dissipativity.jl` |
| [`kyp_riccati`](#kyp_riccati) | `dissipativity.jl` |
| [`port_storage`](#port_storage) | `dissipativity.jl` |
| [`rest_storage`](#rest_storage) | `dissipativity.jl` |
| [`loop_margin`](#loop_margin) | `dissipativity.jl` |
| [`current_correction`](#current_correction) | `exchange.jl` |
| [`kcl_solve`](#kcl_solve) | `exchange.jl` |
| [`rotor_current_coordinates`](#rotor_current_coordinates) | `exchange.jl` |
| [`exchange_one_form`](#exchange_one_form) | `exchange.jl` |
| [`one_form_exactness`](#one_form_exactness) | `exchange.jl` |
| [`one_form_path_test`](#one_form_path_test) | `exchange.jl` |
| [`lossless_variant`](#lossless_variant) | `exchange.jl` |
| [`exchange_curl`](#exchange_curl) | `exchange.jl` |
| [`rotor_energy`](#rotor_energy) | `joint_storage.jl` |
| [`shifted_kinetic_energy`](#shifted_kinetic_energy) | `joint_storage.jl` |
| [`quotient_hessian`](#quotient_hessian) | `joint_storage.jl` |
| [`exchange_dissipation_forms`](#exchange_dissipation_forms) | `joint_storage.jl` |
| [`rotor_gradient_metric`](#rotor_gradient_metric) | `joint_storage.jl` |
| [`network_potential`](#network_potential) | `polar.jl` |
| [`polar_balance`](#polar_balance) | `polar.jl` |
| [`conductance_curl`](#conductance_curl) | `polar.jl` |
| [`PortModel`](#portmodel) | `ports.jl` |
| [`port_model`](#port_model) | `ports.jl` |
| [`transfer`](#transfer) | `ports.jl` |
| [`real_part_crossings`](#real_part_crossings) | `ports.jl` |
| [`passivity_certificate`](#passivity_certificate) | `ports.jl` |
| [`port_zeros`](#port_zeros) | `ports.jl` |
| [`frequency_response`](#frequency_response) | `ports.jl` |
| [`loop_port_model`](#loop_port_model) | `ports.jl` |
| [`component_power`](#component_power) | `power.jl` |
| [`network_power`](#network_power) | `power.jl` |
| [`power_audit_samples`](#power_audit_samples) | `power.jl` |
| [`port_power_audit`](#port_power_audit) | `power.jl` |
| [`ConstantPowerSink`](#constantpowersink) | `sink.jl` |
| [`integrable_loss_variant`](#integrable_loss_variant) | `sink.jl` |
| [`storage_components`](#storage_components) | `storage.jl` |
| [`total_hamiltonian`](#total_hamiltonian) | `storage.jl` |
| [`grad_total_hamiltonian`](#grad_total_hamiltonian) | `storage.jl` |
| [`hessian_total_hamiltonian`](#hessian_total_hamiltonian) | `storage.jl` |
| [`solve_network`](#solve_network) | `storage.jl` |
| [`reduced_field`](#reduced_field) | `storage.jl` |
| [`reduced_jacobian`](#reduced_jacobian) | `storage.jl` |
| [`strain_rotor`](#strain_rotor) | `strain.jl` |
| [`strain_metric`](#strain_metric) | `strain.jl` |
| [`strain_energy`](#strain_energy) | `strain.jl` |
| [`strain_balance`](#strain_balance) | `strain.jl` |
| [`strain_reference`](#strain_reference) | `strain.jl` |
| [`shifted_strain_energy`](#shifted_strain_energy) | `strain.jl` |
| [`strain_decay_forms`](#strain_decay_forms) | `strain.jl` |
| [`state_groups`](#state_groups) | `structure.jl` |
| [`storage_pattern`](#storage_pattern) | `structure.jl` |
| [`section_pattern`](#section_pattern) | `structure.jl` |
| [`reference_section`](#reference_section) | `structure.jl` |
| [`lyapunov_check`](#lyapunov_check) | `structure.jl` |
| [`structured_lyapunov`](#structured_lyapunov) | `structure.jl` |
| [`structured_completion`](#structured_completion) | `structure.jl` |
| [`completion_rate_feasibility`](#completion_rate_feasibility) | `structure.jl` |
| [`completion_margin`](#completion_margin) | `structure.jl` |
| [`minimal_completion`](#minimal_completion) | `structure.jl` |
| [`generalized_decay_rate`](#generalized_decay_rate) | `structure.jl` |
| [`pow2_scaling`](#pow2_scaling) | `structure.jl` |
| [`decay_margin`](#decay_margin) | `structure.jl` |
| [`margin_residuals`](#margin_residuals) | `structure.jl` |
| [`verified_min_eig`](#verified_min_eig) | `structure.jl` |
| [`pattern_certificate`](#pattern_certificate) | `structure.jl` |
| [`verified_lyapunov`](#verified_lyapunov) | `structure.jl` |
| [`rank_couplings`](#rank_couplings) | `structure.jl` |
| [`couple_states!`](#couple_states) | `structure.jl` |
| [`add_coupling!`](#add_coupling) | `structure.jl` |
| [`TerminalModel`](#terminalmodel) | `terminal.jl` |
| [`NetworkModel`](#networkmodel) | `terminal.jl` |
| [`sync_jacobian`](#sync_jacobian) | `terminal.jl` |
| [`terminal_models`](#terminal_models) | `terminal.jl` |
| [`terminal_margins`](#terminal_margins) | `terminal.jl` |

---

### PhysicalProjection

*[src/ph/audit.jl:13](../../src/ph/audit.jl#L13)*

    PhysicalProjection

The physical coordinates of the full state: the one-way reservoir states (from the port
contracts), the identically held states and the `delta_COI` monitor are removed (`keep`);
the COI covector `coi` (weight on each COI source's angle state) defines the invariant
section, whose orthonormal basis in `keep` coordinates is `basis`.

---

### certificate_only

*[src/ph/audit.jl:36](../../src/ph/audit.jl#L36)*

    certificate_only(c) -> Bool

Whether `c` is a stateless element added only by a certificate variant of a system (e.g. the
constant-power sinks of `integrable_loss_variant`), never part of a case.

---

### reservoir_states

*[src/ph/audit.jl:44](../../src/ph/audit.jl#L44)*

    reservoir_states(sys; contracts = load_contracts()) -> Vector{Int}

The one-way accounting states: each component's reservoir state as its port contract
declares it. A component type without a contract is an error (fail closed), except the
stateless certificate-only elements of a variant (`certificate_only`), which have no
reservoir.

---

### physical_projection

*[src/ph/audit.jl:67](../../src/ph/audit.jl#L67)*

    physical_projection(sys, x, V; contracts, probes = 3, scale = 1e-3, seed = 7)

Held states are those whose reduced-field row is exactly zero at `probes` states
`x + scale * randn` (Python PHPS's rule). The COI weights are `sys.coi_weights`.

---

### shifted_storage_audit

*[src/ph/audit.jl:94](../../src/ph/audit.jl#L94)*

    shifted_storage_audit(sys, x, V; projection, eps, rank_tol = 1e-9, rate_tol = 1e-12)
        -> Dict

The local audit of the assembled storage at the equilibrium `(x, V)`:

- `storage_rank`: rank of `S = Hess H` on the kept coordinates (eigenvalues above
  `rank_tol * max(1, max|S|)`), and the eigenvalues of `S` on the section;
- `sym_SA`: the eigenvalues of `M = U' sym(S A) U` on the section (`U` = section basis,
  `A` the exact reduced Jacobian), the number above `rate_tol * max(1, max|eig|)`;
- `exact`: `H_s` and `dH_s/dt` at `x + sign * eps * v` along the top eigenvector `v` of `M`
  (lifted, normalised), with `dH_s/dt` split by storage component.

---

### completion_problem

*[src/ph/completion.jl:39](../../src/ph/completion.jl#L39)*

    completion_problem(m; forms = strain_decay_forms(m)) -> NamedTuple

The data of the storage-completion LMI on the section model `m` (`scale = :none`, at an
equilibrium): `A` (section Jacobian), `H` (Hessian of S'), `Hfix` (`H` plus the field
cross-terms), `own` (the own-unit controller pattern: exciter and governor blocks and their
cross-blocks with their machine), `swing` (every angle and speed pair), `basis` (records with
`kind` = `:load_filter`, `:load_z` or `:delta_omega`, the component index `k` and the
`matrix`), `names`, `units` (machine, exciter and governor component indices) and `field`
(per machine: its exciter's `xe` state, `M B_f`, the rotor state indices).

---

### completion_pattern

*[src/ph/completion.jl:116](../../src/ph/completion.jl#L116)*

    completion_pattern(prob, pairs) -> BitMatrix

The free pattern `own .| swing` plus the full blocks between the listed component pairs
(`(name_a, name_b)`), closed with `invariant_closure`.

---

### invariant_closure

*[src/ph/completion.jl:135](../../src/ph/completion.jl#L135)*

    invariant_closure(mask, names) -> BitMatrix

The smallest superset of `mask` that is invariant under a change of the angle
parametrization of the common-angle section: every non-angle coordinate with a free angle
entry gets all angle entries, and the angle-angle block is free as soon as any entry is.

---

### StorageCompletion

*[src/ph/completion.jl:157](../../src/ph/completion.jl#L157)*

    StorageCompletion

An explicit nonlinear `H_ext` built by `storage_completion`: the physical core (S' with the
fixed field cross-terms), the controller quadratic `Fc` and the repair quadratic `Fr` (section
coordinates) and the repair basis terms with their coefficients.

---

### storage_completion

*[src/ph/completion.jl:173](../../src/ph/completion.jl#L173)*

    storage_completion(prob, sol; T = I) -> StorageCompletion

From a solution `sol` of `structured_completion` on `prob` (in coordinates scaled by the
diagonal `T`: `A_s = T^-1 A T`, `H_s = T H T`), the explicit storage: the free part split
into controller (`prob.own`) and repair entries, the basis coefficients attached to their
terms.

---

### completion_energy

*[src/ph/completion.jl:193](../../src/ph/completion.jl#L193)*

    completion_energy(sc, eta; V0 = sc.m.V0) -> NamedTuple

`(physical, controller, repair, total)` of the explicit storage at section coordinates `eta`
(voltages on the KCL branch through `kcl_solve` from `V0`, converged near `lift(m, eta)`),
generic in the number type.

---

### completion_energy_state

*[src/ph/completion.jl:205](../../src/ph/completion.jl#L205)*

    completion_energy_state(sc, x, V) -> NamedTuple

`completion_energy` at a full state `x` and bus voltages `V` given explicitly (no KCL solve;
the quadratic parts use the section coordinates of `x`), generic in the number type
(intervals and duals included). On the KCL branch it equals `completion_energy`.

---

### completion_hessian_check

*[src/ph/completion.jl:241](../../src/ph/completion.jl#L241)*

    completion_hessian_check(sc, P) -> NamedTuple

The Hessian of `completion_energy(sc, .).total` at the equilibrium against `P` (the LMI's
matrix in unscaled section coordinates): the relative difference, and the gradient norm.

---

### ComponentPort

*[src/ph/component_ports.jl:18](../../src/ph/component_ports.jl#L18)*

    ComponentPort

`x' = A x + B du`, `dy = C x + D du` of one component over `states`, from the inputs `inputs`
to the outputs `outputs` (for `norton = true`, the injected current `I_N - y_N V`).
The element type is Float64 or an interval type.

---

### component_port

*[src/ph/component_ports.jl:46](../../src/ph/component_ports.jl#L46)*

    component_port(c, x, u; inputs, outputs, norton = false, hold = [], contracts) -> ComponentPort

Linearise component `c` at its state `x` and inputs `u` (Float64 or intervals). States named in
`hold` are frozen at their values (left out of the dynamics), which isolates their contribution:
for example, holding a machine's `delta` and `omega` gives its angle-frozen electrical port.

---

### port_transfer

*[src/ph/component_ports.jl:93](../../src/ph/component_ports.jl#L93)*

`G(jw) = C (jw I - A)^{-1} B + D` of a Float64 port.

---

### port_hermitian_scan

*[src/ph/component_ports.jl:97](../../src/ph/component_ports.jl#L97)*

    port_hermitian_scan(P, ws; sign = 1) -> (lam, vecs)

The smallest eigenvalue of `sign * (G(jw) + G(jw)^H)/2` at each `w`, and its eigenvector.

---

### port_witness_interval

*[src/ph/component_ports.jl:113](../../src/ph/component_ports.jl#L113)*

    port_witness_interval(P, w, v) -> Ival

An enclosure of `Re(v^H G(jw) v)` for an interval port `P`, the Float64 frequency `w` and
the Float64 vector `v`. `(jw I - A) X = B` is solved in its real form
`[-A -wI; wI -A] [Xr; Xi] = [B; 0]` by `interval_solve`.

---

### polar_transfer

*[src/ph/component_ports.jl:144](../../src/ph/component_ports.jl#L144)*

    polar_transfer(P, V0, I0, w) -> Matrix{ComplexF64}

The transfer from `(dtheta, drho)` to the injected `(dP, dQ)` at frequency `w`, for a bus port
`P` (inputs `Vd, Vq`, outputs the injected current) at bus voltage `V0 = (Vd, Vq)` and injected
current `I0 = (Id, Iq)`: `dV = [-Vq Vd; Vd Vq] dw`, `dP = I0' dV + V0' dI`,
`dQ = (Id dVq - Iq dVd) + (Vq dId - Vd dIq)`.

---

### polar_scan

*[src/ph/component_ports.jl:161](../../src/ph/component_ports.jl#L161)*

    polar_scan(P, V0, I0, ws) -> (lam, vecs)

The largest eigenvalue of `(G - G^H)/(2i)` (the Hermitian matrix whose quadratic form is
`Im(v^H G v)`) at each `w`, and its eigenvector. A positive value at some `w > 0` excludes every
`C^2` storage at the polar port.

---

### injected_current

*[src/ph/component_ports.jl:179](../../src/ph/component_ports.jl#L179)*

    injected_current(c, x, u) -> (Id, Iq)

The current a Norton component injects into the network at its state `x` and inputs `u`:
its Norton current minus `y_N V`.

---

### MultiPortModel

*[src/ph/dissipativity.jl:11](../../src/ph/dissipativity.jl#L11)*

    MultiPortModel

The rest of the system with several components removed: `x' = A x + B y`, `u = C x`, where
`y` stacks the removed components' outputs (the rest's inputs) and `u` their inputs (the
rest's outputs), on the common-angle section of the remaining states.

---

### open_loops_model

*[src/ph/dissipativity.jl:31](../../src/ph/dissipativity.jl#L31)*

    open_loops_model(sys, x, V, ks; input, output, projection, contracts)
        -> (rest::MultiPortModel, ports::Vector{PortModel}, T::Matrix)

Open the loops at the components `ks` at once (as `loop_port_model` does for one): the rest
of the system from their outputs to their inputs, and each component's own port model.
`T` maps the coordinates `[rest section; component states...]` to the physical (`keep`)
coordinates, so that `T' A T` is the whole system in those coordinates.

---

### multiport_margin

*[src/ph/dissipativity.jl:96](../../src/ph/dissipativity.jl#L96)*

    multiport_margin(rest, n; eta = 0.0, ws) -> NamedTuple

The frequency condition for a storage of the rest with supply
`sum_k y_k u_k - n_k u_k^2 + eta y_k^2` (`u = G y`): the smallest eigenvalue over `ws` of
`Herm(G(jw)) - G(jw)' diag(n) G(jw) + eta I`. `passes` when it is positive (then
`rest_storage(rest, n; eta)` exists, by the KYP lemma in its strict form). As `w -> infinity`
it tends to `eta`. With `eta = 0` it is the classical constant split; `eta > 0` lets the
components pay for directions the rest cannot dissipate (the differential speed directions
at low frequency, which a lossy network makes non-passive).

---

### port_margin

*[src/ph/dissipativity.jl:121](../../src/ph/dissipativity.jl#L121)*

    port_margin(m, d; eta = 0.0, ws) -> NamedTuple

The frequency condition for `port_storage(m, d; eta)`: the smallest over `ws` of
`Re H(jw) + d - eta |H(jw)|^2`, `H = -G`, and its value as `w -> infinity` (`d`).

---

### kyp_riccati

*[src/ph/dissipativity.jl:135](../../src/ph/dissipativity.jl#L135)*

    kyp_riccati(A, B, Q, S, R; strict = 0.0) -> (P, residual)

A quadratic storage `V = x' P x` from the KYP inequality

    K(P) = [A'P + PA + Q + strict*I   PB - S;  (PB - S)'   -R] <= 0,   R > 0,

i.e. `dV/dt <= s(x, w) - strict |x|^2` for the supply `s = -x'Qx + 2 x'S w + w'R w` of
`x' = A x + B w`. `P` is the stabilising solution of the Riccati equation (the Schur complement
of `K` equal to zero), from the stable invariant subspace of its Hamiltonian matrix; it exists
when the frequency condition of the supply holds strictly. `residual` is the largest
eigenvalue of `K(P)`, which must be <= 0 up to rounding.

---

### port_storage

*[src/ph/dissipativity.jl:165](../../src/ph/dissipativity.jl#L165)*

    port_storage(m, d; eta = 0.0, strict = 0.0) -> (P, residual)

The storage of a component at its port with `dV/dt <= u y + d u^2 - eta y^2`, `y = -(C x)`
(a governor: `u` the speed, `y = -Tm`; the port model must have `D = 0`).

---

### rest_storage

*[src/ph/dissipativity.jl:178](../../src/ph/dissipativity.jl#L178)*

    rest_storage(rest, n; eta, strict = 0.0) -> (P, residual)

The storage of the rest of the system with `dV/dt <= sum_k y_k u_k - n_k u_k^2 + eta y_k^2`
(`y` its inputs, the components' outputs; `u = C z` its outputs). `eta > 0` makes the
Riccati equation regular; the components' storages pay it back (`port_storage(...; eta)`),
so it cancels in the sum.

---

### loop_margin

*[src/ph/dissipativity.jl:190](../../src/ph/dissipativity.jl#L190)*

    loop_margin(rest, ports; ws) -> NamedTuple

The most generous diagonal split at the ports: each component pays exactly its own
`Re H_k(jw)` (`H_k = -G_k`) at every frequency. The loop admits a storage split port by port
(with any static or frequency-dependent diagonal multiplier) only if
`Herm(G(jw)) + G(jw)' diag(Re H_k(jw)) G(jw)` is positive definite at every frequency;
returns its smallest eigenvalue over `ws` and where. A negative value rules the port-by-port
split out.

---

### current_correction

*[src/ph/exchange.jl:24](../../src/ph/exchange.jl#L24)*

    current_correction(c) -> Matrix or nothing

`D = -A^-1 B_s` for a machine with a linear `rotor_structure`: `z' = A (z - D i) + B_f Efd`.

---

### kcl_solve

*[src/ph/exchange.jl:34](../../src/ph/exchange.jl#L34)*

    kcl_solve(sys, x, V0; faults_on = false, iterations = 3) -> V

The KCL voltages at `x`, in the element type of `x`, from `V0`, a converged Float64 solution
at the value of `x`: iterations `V <- V - g_V^-1 g(x, V)` with `g_V` the Float64 Jacobian at
`(value(x), V0)`. The value part is already converged, and for the dual parts this map has
zero contraction, so each derivative order is exact after one more step: with duals the
result carries the derivatives of the KCL branch `V(x)` (the implicit function), up to
order `iterations - 1`.

---

### rotor_current_coordinates

*[src/ph/exchange.jl:55](../../src/ph/exchange.jl#L55)*

    rotor_current_coordinates(sys, x, V) -> Dict{String,NamedTuple}

For every machine with a linear rotor: `z`, the stator currents `i = [id, iq]`, `w = z - D i`,
the rotor currents `Qw`, `i_fd_energy = B_f'Qw` (against the runtime `i_fd = B_f'Qz`), the
field supply `Efd i_fd_energy` and the loss `-w' sym(QA) w`.

---

### exchange_one_form

*[src/ph/exchange.jl:113](../../src/ph/exchange.jl#L113)*

    exchange_one_form(m::SectionModel, eta; sys = m.sys, V0 = m.V0) -> Vector

The coefficients of the stator-exchange one-form in section coordinates,
`a(eta) = (dI/deta)' c`, with `I` the stacked stator currents of the linear-rotor machines
on the KCL branch of `sys` (by default the model's; a variant such as `lossless_variant`
can be passed) and `c` the stacked `D'Qw`. `a . eta'` is `w'QD di/dt` summed over the units.
`V0` must be a converged KCL solution near `lift(m, eta)`.

---

### one_form_exactness

*[src/ph/exchange.jl:134](../../src/ph/exchange.jl#L134)*

    one_form_exactness(m, eta; sys = m.sys, V0 = m.V0, pairs = 6, seed = 3) -> NamedTuple

The curl test of `exchange_one_form` at `eta`: for random unit direction pairs `(u, v)`,
`v' Ja u` against `u' Ja v`, with `Ja u` the directional derivative of `a` (nested duals
through the KCL branch). An exact one-form has a symmetric Jacobian, so the two agree;
`curl` is the largest relative difference `|v'Ja u - u'Ja v| / (|v'Ja u| + |u'Ja v|)`.
Also returns `a` and the pairs' values.

---

### one_form_path_test

*[src/ph/exchange.jl:159](../../src/ph/exchange.jl#L159)*

    one_form_path_test(m, eta1; sys = m.sys, V0 = m.V0, order = 8) -> NamedTuple

The integral of the one-form from `eta = 0` to `eta1` along the straight segment and along a
two-leg path (first the coordinates of the first half of the entries, then the rest), by
Gauss-Legendre quadrature of order `order` per leg; equal for an exact form.

---

### lossless_variant

*[src/ph/exchange.jl:186](../../src/ph/exchange.jl#L186)*

    lossless_variant(sys) -> DAESystem

The same system with a lossless network: every Y-bus conductance, the constant-impedance
load conductances, the frequency-dependent active loads and the ComplexLoads' active
power (`P0 = 0`) removed; susceptances, reactive loads and the machines unchanged. For
testing the network identities (step 5), not a model of the case; its KCL solution at a
state of the original system is generally near, not at, the original voltages.

---

### exchange_curl

*[src/ph/exchange.jl:206](../../src/ph/exchange.jl#L206)*

    exchange_curl(m, eta; sys = m.sys, V0 = m.V0) -> Matrix

The curl of the stator-exchange one-form in section coordinates, `Ja - Ja'`, in closed
form: with `c = D'Q(z - D I)`, the part `-(D'QD I)' dI` is exact and `(D'Qz)' dI` is not,
so `Ja - Ja' = (dI/deta)' D'Q (dz/deta) - (dz/deta)' Q D (dI/deta)` (block-diagonal `D`, `Q`
over the units). It vanishes only where the rotor states and the stator currents do not vary
independently on the KCL branch, so no scalar storage has the one-form as its gradient.

---

### rotor_energy

*[src/ph/joint_storage.jl:28](../../src/ph/joint_storage.jl#L28)*

    rotor_energy(sys, x, V0; alpha = 0.0) -> Real

`sum w'Qw/2 - alpha I'D'QD I/2` over the machines with a linear rotor, with the stator
currents on the KCL branch at `x` (`kcl_solve` from the converged `V0`), generic in the
number type of `x`.

---

### shifted_kinetic_energy

*[src/ph/joint_storage.jl:48](../../src/ph/joint_storage.jl#L48)*

    shifted_kinetic_energy(sys, x) -> Real

`sum H (omega - 1)^2` over the synchronous machines (the model's shifted kinetic storage).

---

### quotient_hessian

*[src/ph/joint_storage.jl:63](../../src/ph/joint_storage.jl#L63)*

    quotient_hessian(m, S; eta0 = zeros, tol = 1e-9) -> NamedTuple

The Hessian of `eta -> S(lift(m, eta))` at `eta0` (the Hessian of the Bregman form of `S`
there), its eigenvalues and inertia (`positive`, `negative`, `zero`, counted against
`tol * max|eig|`). `S(x)` must accept dual numbers.

---

### exchange_dissipation_forms

*[src/ph/joint_storage.jl:79](../../src/ph/joint_storage.jl#L79)*

    exchange_dissipation_forms(m; sys = m.sys, V0 = m.V0) -> NamedTuple

The quadratic forms at the reference point of `m` (see the comment at the top of
`src/ph/joint_storage.jl`): `rotor_loss`, `exchange` (symmetrised) and `network_loss`, all
on the section coordinates, and the comparison of the exchange with the rotor and network
dissipation: `null_exchange_max`, the largest eigenvalue of the exchange form on the null
space of the dissipation (positive: directions where no dissipation of these two kinds
dominates it, however small), and `ratio_max`, the largest generalized eigenvalue of the
exchange against the dissipation on its range (> 1: the exchange exceeds that dissipation
in some direction). Controllers carry no storage yet, so their dissipation is not part of
the comparison.

---

### rotor_gradient_metric

*[src/ph/joint_storage.jl:137](../../src/ph/joint_storage.jl#L137)*

    rotor_gradient_metric(rs) -> NamedTuple

The Nishino-Chakrabortty-Ishizaki structure for a linear rotor (`rotor_structure`): per axis
the symmetric `M` with `M B_s = -Psi''` (transposed) and `M A` symmetric, so that with an
internal EMF `E'' = (-psi_q'', psi_d'')` behind `j x''` on a lossless network, whose potential
has `grad_z U_net = Psi''' i`, the rotor dynamics are the gradient flow
`M z' = -grad(U_rot + U_net) + M B_f Efd` of the rotor energy `U_rot = -z' M A z / 2`. Returns
`M` (block-diagonal over the axes), `U = -sym(MA)` (the Hessian of `U_rot`), whether `M > 0`
and `U > 0` (the metric and the convexity), and the residual of `M B_s + Psi''`. The speed
voltage (`E'' = omega psi''` in the model) is outside this structure.

---

### network_potential

*[src/ph/polar.jl:23](../../src/ph/polar.jl#L23)*

    network_potential(sys, V) -> Real

`U_B = -1/2 sum B_ij (Vd_i Vd_j + Vq_i Vq_j)` for the susceptance part of the Y-bus.

---

### polar_balance

*[src/ph/polar.jl:59](../../src/ph/polar.jl#L59)*

    polar_balance(sys, x, V; faults_on = false) -> Dict

At `(x, V)` (KCL satisfied), along the flow `x' = f(x, V)` with `V'` from the KCL branch:
- `lossless_identity_error`: `dU_B/dt` against `sum (P^B theta' + Q^B d ln|V|/dt)`;
- `conductance_rate`: `sum (P^G theta' + Q^G d ln|V|/dt)` and its closed form `-Im(V'^H G V)`;
- per machine with a linear rotor: `P_int`, `Q_int`, the polar supply
  `(P_int phi' + Q_int d ln|E''|/dt)/omega_b`, the transformer power `i . dpsi''/dt / omega_b`,
  the rotor exchange `w'QD di/dt`, and `mismatch = exchange - polar supply`.

---

### conductance_curl

*[src/ph/polar.jl:122](../../src/ph/polar.jl#L122)*

    conductance_curl(sys) -> Matrix

The curl of the conductance one-form in the stacked Cartesian voltages `[Vd_1, Vq_1, ...]`:
`2 (G kron J)`, `J = [0 1; -1 0]`; it vanishes only without conductances.

---

### PortModel

*[src/ph/ports.jl:12](../../src/ph/ports.jl#L12)*

    PortModel

The small-signal model of one component at one port: `x' = A x + B u`, `y = C x + D u`,
over the states `states`, for input `input` and output `output`.

---

### port_model

*[src/ph/ports.jl:33](../../src/ph/ports.jl#L33)*

    port_model(c, x, u; input, output, contracts) -> PortModel

Linearise component `c` at its state `x` and inputs `u` from `input` to `output` (names
from `input_names(c)` and `output_names(c)`; the output as the component's output kernel
computes it). The contract's reservoir state is left out.

---

### transfer

*[src/ph/ports.jl:67](../../src/ph/ports.jl#L67)*

`G(s) = C (sI - A)^{-1} B + D` of a port model.

`G(s) = C (sI - A)^{-1} B` of a multi-port model (square: one row and column per port).

`dI = G(jw) dV` of a unit (the current it injects).

`dI = Y(jw) dV` of the network (the current it absorbs at every bus).

---

### real_part_crossings

*[src/ph/ports.jl:70](../../src/ph/ports.jl#L70)*

    real_part_crossings(m; sign = -1, wmin = 1e-3, wmax = 1e3, n = 60001) -> Vector

The frequencies (rad/s) where `Re(sign * G(jw))` changes sign: sign changes on a
logarithmic grid, each refined by bisection to the resolution of Float64.

---

### passivity_certificate

*[src/ph/ports.jl:100](../../src/ph/ports.jl#L100)*

    passivity_certificate(m; sign = -1, wmin = 1e-3, wmax = 1e3, n = 60001) -> NamedTuple

For `H = sign * G`: `H0 = H(0)`, the smallest `Re H(jw)` on the grid and its frequency, and
`nonpassive = true` when that value is negative by more than `1e-8 * max(1, |H|)` (then no
storage makes the port passive and the positive-real LMI is infeasible; the margin covers
the Float64 evaluation of a well-conditioned small system). `shortage` is `-min Re H`
(the passivity shortage index, 0 when passive on the grid).

---

### port_zeros

*[src/ph/ports.jl:120](../../src/ph/ports.jl#L120)*

    port_zeros(m) -> Vector{ComplexF64}

The finite zeros of `G(s)`: the finite generalized eigenvalues of the pencil
`([A B; C' D], [I 0; 0 0])`, sorted by real part. A zero in the right half-plane rules out
passivity at the port whatever the storage (Python PHPS roadmap section 7, the hydro governor).

---

### frequency_response

*[src/ph/ports.jl:134](../../src/ph/ports.jl#L134)*

    frequency_response(m, ws) -> Vector{ComplexF64}

`G(jw)` at the frequencies `ws` (rad/s): one Hessenberg factorisation of `A`, then a shifted
Hessenberg solve per frequency (exact for any `A`, also with repeated poles).

---

### loop_port_model

*[src/ph/ports.jl:146](../../src/ph/ports.jl#L146)*

    loop_port_model(sys, x, V, k; input, output, projection, contracts) -> PortModel

The rest of the system seen from component `k`: the small-signal transfer from `k`'s
output `output` (as its consumers receive it) to `k`'s input `input`, with `k` removed and
everything else (machines, network, loads, other controllers) kept, at the equilibrium
`(x, V)`. Built from the exact reduced Jacobian on the physical coordinates of `projection`:
`k` must act on the rest only through `output` and the rest on `k` only through `input`
(checked: both coupling blocks have rank one). The model is given on the common-angle
section of the remaining states, where the common-angle mode is removed.

---

### component_power

*[src/ph/power.jl:52](../../src/ph/power.jl#L52)*

    component_power(sys, x, V; contracts = default_contracts(), faults_on = false)
        -> Vector{Dict}

For every component at `(x, V)`: `storage_rate` (`grad H' f` over its own states),
`ports` (name, declared expression, value or `nothing` when the contract gives no
expression), `supply` (the evaluable ports), `complete` (every power port evaluable),
`residual = supply - storage_rate`. Machines also get `kinetic_rate_shift = 2 H omega'`
(the rate of the physical kinetic storage `H omega^2` minus that of the shifted
`H (omega - 1)^2` the model declares) and, against the physical kinetic storage, the split
of the residual into `mechanical_residual = Tm - Pe - d(H omega^2)/dt`,
`magnetic_residual = Efd i_fd - dH_mag/dt` and `terminal_residual = Pe - V . I_norton`.
Machines with a `rotor_structure` (GENROU) also get the rotor port split `field_supply`,
`stator_exchange = [id, iq]' B_s' Q z`, `rotor_loss = -z' sym(QA) z`, and the errors of the
field colocation, of the linear rotor form against the right-hand side, and of the identity
`dH_mag/dt = field_supply + stator_exchange - rotor_loss`.

---

### network_power

*[src/ph/power.jl:141](../../src/ph/power.jl#L141)*

    network_power(sys, x, V; faults_on = false) -> Dict

The network's power balance at `(x, V)` (see the comment at the top of `src/ph/power.jl`):
`injected`, `absorbed` (`norton`, `network`, `fault`, `frequency_loads`), `kcl_residual_power
= V . g` (their difference), and per machine and per ComplexLoad the terminal identities.

---

### power_audit_samples

*[src/ph/power.jl:213](../../src/ph/power.jl#L213)*

    power_audit_samples(sys, x, V; n_random = 8, scale = 1e-2, seed = 11, trajectory = nothing)
        -> Vector{NamedTuple}

The states the audit visits: the equilibrium, `n_random` random perturbations of it
(`scale` relative to each state's magnitude, at least `scale`, reservoirs, held states and
the monitor unchanged, voltages from `solve_network`), and, if given, the states of a
`SimResult` (`trajectory`) with their fault flag (`fault_window`; a record at an event time
is the left limit) and their voltages re-solved on that KCL branch, so every sample
satisfies KCL to round-off.

---

### port_power_audit

*[src/ph/power.jl:262](../../src/ph/power.jl#L262)*

    port_power_audit(sys, samples; contracts = default_contracts()) -> Dict

`component_power` and `network_power` at every sample (from `power_audit_samples`), and a
summary per component and per model type: the range of the residual (`supply - rate`) and
of the storage rate, for machines also the residual against the physical kinetic storage
`H omega^2` (`residual - kinetic_rate_shift`), whether the contract's ports are complete,
and the largest errors of the network, terminal and load identities.

---

### ConstantPowerSink

*[src/ph/sink.jl:22](../../src/ph/sink.jl#L22)*

    ConstantPowerSink

A stateless certificate-only element drawing the constant complex power `P + jQ` at its
bus: injected current `-(P - jQ) V / |V|^2`. Built by `integrable_loss_variant`.

---

### integrable_loss_variant

*[src/ph/sink.jl:72](../../src/ph/sink.jl#L72)*

    integrable_loss_variant(sys, x, V; tol = 1e-12) -> DAESystem

The integrable-loss equivalent of `sys` at its equilibrium `(x, V)` (see the comment at the
top of `src/ph/sink.jl`): `lossless_variant(sys)` plus one `ConstantPowerSink` per bus whose
absorbed power exceeds `tol`. `(x, V)` is an equilibrium of the result as well.

---

### storage_components

*[src/ph/storage.jl:12](../../src/ph/storage.jl#L12)*

    storage_components(sys) -> Vector{Int}

The components that carry storage: those whose Hamiltonian is nonzero at the test state
`x = 0.01` (Python PHPS's `HamiltonianAssembler` rule).

---

### total_hamiltonian

*[src/ph/storage.jl:24](../../src/ph/storage.jl#L24)*

    total_hamiltonian(sys, x) -> Real

`H(x)`, the sum of the storage components' Hamiltonians.

---

### grad_total_hamiltonian

*[src/ph/storage.jl:37](../../src/ph/storage.jl#L37)*

    grad_total_hamiltonian(sys, x) -> Vector

`grad H(x)` on the full state (zero outside the storage components).

---

### hessian_total_hamiltonian

*[src/ph/storage.jl:51](../../src/ph/storage.jl#L51)*

    hessian_total_hamiltonian(sys, x) -> Matrix

`Hess H(x)` (ForwardDiff of the gradient, symmetrised).

---

### solve_network

*[src/ph/storage.jl:61](../../src/ph/storage.jl#L61)*

    solve_network(sys, x, V0; tol = 1e-13, maxiter = 50, faults_on = false) -> V

The bus voltages with KCL `g(x, V) = 0` at the states `x` (with every fault shunt of the
scenario when `faults_on`), by Newton from `V0`.
As in Python PHPS's network solve, one more Newton step is taken once the residual is below
`tol`, so the result is at round-off and does not depend on the stopping rule.

---

### reduced_field

*[src/ph/storage.jl:84](../../src/ph/storage.jl#L84)*

    reduced_field(sys, x, V0) -> (f, V)

The network-reduced field `f(x) = f(x, V(x))` and the voltages `V(x)`, solved from `V0`.

---

### reduced_jacobian

*[src/ph/storage.jl:94](../../src/ph/storage.jl#L94)*

    reduced_jacobian(sys, x, V) -> Matrix

The exact Jacobian of the network-reduced field at `(x, V)` (with `g(x, V) = 0`):
`A = f_x - f_V g_V^{-1} g_x`.

---

### strain_rotor

*[src/ph/strain.jl:33](../../src/ph/strain.jl#L33)*

    strain_rotor(c) -> NamedTuple or nothing

The rotor of a machine in the form the strain-energy structure needs:
`z' = A z + B_f u_f + B_s [id, iq]` with `(psi_d'', psi_q'') = Psi z`, the axes' state blocks,
and the subtransient reactance. GENROU: its `rotor_structure` (`u_f = Efd`). GENSAL: the
linear part of its rotor; its additive saturation enters as `u_f = Efd - Sat (xd - xl)`.
`nothing` for other components.

---

### strain_metric

*[src/ph/strain.jl:72](../../src/ph/strain.jl#L72)*

    strain_metric(sr) -> NamedTuple

For a `strain_rotor`: the block-diagonal metric `M` (per axis `M B_s = -Psi'`, `M A`
symmetric), the Hessian `U = -sym(MA)` of `U_rot = -z' M A z / 2`, whether `M > 0` and
`U > 0`, and the residuals of the two conditions. `rotor_gradient_metric` is the GENROU case.

---

### strain_energy

*[src/ph/strain.jl:183](../../src/ph/strain.jl#L183)*

    strain_energy(sys, x, V) -> Real

`U_ext = U_rot + U_net` (see the comment at the top of `src/ph/strain.jl`) at the states `x`
and bus voltages `V`, generic in the number type.

---

### strain_balance

*[src/ph/strain.jl:210](../../src/ph/strain.jl#L210)*

    strain_balance(sys, x, V) -> Dict

At `(x, V)` (KCL satisfied, healthy network), along the flow with `V'` from the KCL branch:
`rate = dU_ext/dt` (by differentiation) and the terms of its decomposition, `dissipation`
(`-sum z'Mz'`), `field` (`sum Efd B_f'Mz'`), `angle` (`sum Te delta'`), `saturation`,
`speed_rotor`, `speed_network`, `conductance`, `active_load`, `frequency_load`,
`load_lag`; `paper_residual = rate - (dissipation + field + angle)`, `identity_error =
rate - sum of all terms`, `torque_error` (`max |Te - Pe/omega|`, `Te = dU_net/d delta`),
`structure_error` (`max |z' - A z - B_s i - B_f u_f|`) and per machine `speed_rotor`,
`dissipation`, `field`, `saturation`.

---

### strain_reference

*[src/ph/strain.jl:330](../../src/ph/strain.jl#L330)*

    strain_reference(sys, x, V) -> NamedTuple

The equilibrium constants of `shifted_strain_energy`: per strain unit `u_f*` and `Tm*`, and
the rotated conductance and ComplexLoad active currents `c*` (stacked like `V`).

---

### shifted_strain_energy

*[src/ph/strain.jl:353](../../src/ph/strain.jl#L353)*

    shifted_strain_energy(sys, x, V, ref) -> Real

`S' = U_ext + K - sum u_f* B_f'Mz - sum Tm* delta - c*' V` (see above), `ref` from
`strain_reference` at the equilibrium; generic in the number type.

---

### strain_decay_forms

*[src/ph/strain.jl:419](../../src/ph/strain.jl#L419)*

    strain_decay_forms(m; V0 = m.V0) -> NamedTuple

At the reference point of the section model `m` (an equilibrium with omega* = 1, no
frequency-dependent loads), the quadratic forms (section coordinates, symmetrised) of the
second-order terms of dS'/dt listed above (`forms`, a Dict), the Hessian `H` of `S'`, the
norm of its gradient (0 at the equilibrium), the section Jacobian `A`, and `closure`, the
largest entry of `sum(forms) - sym(H A)` relative to `sym(H A)`.

---

### state_groups

*[src/ph/structure.jl:14](../../src/ph/structure.jl#L14)*

    state_groups(sys, projection) -> Vector{NamedTuple}

For each physical coordinate: the component, its role (`:machine`, `:governor`,
`:exciter`, `:load`, `:other`), the unit (the machine it belongs to: itself, or the machine
it feeds), and whether it is a rotor angle.

---

### storage_pattern

*[src/ph/structure.jl:48](../../src/ph/structure.jl#L48)*

    storage_pattern(groups, kind) -> BitMatrix

Allowed nonzero entries of `P` (symmetric):
- `:full` every entry;
- `:unit` within each unit (machine, governor, exciter together), within each other
  component (the loads), and among all rotor angles;
- `:component` within each component only, and among all rotor angles.

---

### section_pattern

*[src/ph/structure.jl:66](../../src/ph/structure.jl#L66)*

    section_pattern(mask, U) -> BitMatrix

The support, in section coordinates, of `Q = U' P U` for `P` on the physical-coordinate
pattern `mask` (`U` from `reference_section`). The reference angle is `-l_z'z / l_ref`, so a
term coupling a state with the reference angle becomes couplings of that state with every
rotor angle; a pattern written directly on the section coordinates would miss them (for the
reference machine's own states in particular). The support is a superset of the image
subspace (its entries are free, not tied to the COI weights).

---

### reference_section

*[src/ph/structure.jl:79](../../src/ph/structure.jl#L79)*

    reference_section(sys, x, V, projection) -> (z, U, As, ref)

The section coordinates: `z` indexes the physical coordinates kept (all but the reference
angle `ref`, the COI source with the largest weight); `U` maps them to the physical
coordinates (identity, plus the row `-l_z'/l_ref` for the reference angle); `As` is the exact
reduced Jacobian in these coordinates (`E A U`, `E` selecting `z`), valid because the section
is invariant.

---

### lyapunov_check

*[src/ph/structure.jl:105](../../src/ph/structure.jl#L105)*

    lyapunov_check(Q, As) -> NamedTuple

The smallest eigenvalue of `Q` and the largest of `As'Q + QAs` (a local Lyapunov function
of the section dynamics when the first is > 0 and the second < 0).

---

### structured_lyapunov

*[src/ph/structure.jl:118](../../src/ph/structure.jl#L118)*

    structured_lyapunov(As, mask; weights = nothing, eps = 1e-3, optimizer, silent = true)
        -> (Q, status)

Solve for `Q` on the pattern `mask` with `Q >= I` and `As'Q + QAs <= -eps I`, minimising
`sum(weights .* |Q|)` (a feasibility problem when `weights` is `nothing`). Implemented in the
JuMP extension (`using JuMP` and a conic solver, e.g. SCS). The result is a candidate: check
it with `lyapunov_check`.

---

### structured_completion

*[src/ph/structure.jl:129](../../src/ph/structure.jl#L129)*

    structured_completion(A, Hfix, mask, basis; optimizer, mu = 0, nonnegative, silent = true)
        -> (P, t, coefficients, F, record, dual_P, dual_rate)

The constrained storage completion: `P = Hfix + F + sum_j k_j basis[j]` with `Hfix` fixed (the
physical core and its fixed cross-terms), `F` free on the pattern `mask` (controller blocks)
and scalar coefficients `k_j` on the structured repair matrices `basis[j]` (`k_j >= 0` where
`nonnegative[j]`); maximises `t` subject to `P >= t I` and `-(A'P + PA) - 2 mu P >= t I`.
`t > 0` is a strict quadratic Lyapunov function with decay rate `mu` in these coordinates
(scale them first, e.g. by powers of two). With `weights` (on the free entries) and
`basis_weights`, it instead minimises the weighted L1 norm of the free entries and the
coefficients subject to `t >= t_min`: the sparsest completion with that margin. Implemented
in the JuMP extension; the result is a candidate, to be checked (`lyapunov_check`,
`verified_lyapunov`).

---

### completion_rate_feasibility

*[src/ph/structure.jl:146](../../src/ph/structure.jl#L146)*

    completion_rate_feasibility(A, Hfix, mask, basis; optimizer, mu, cond_cap, nonnegative)
        -> (feasible, gamma, P, F, coefficients, record, dual_P, dual_cond, dual_rate)

Whether the completion `P = Hfix + F + sum k_j basis[j]` (as in `structured_completion`)
can reach the normalised decay rate `mu` with a bounded condition number: maximise `gamma`
subject to `gamma I <= P <= cond_cap gamma I` and `-(A'P + PA) >= 2 mu P`. `feasible` when
`gamma > 0`; otherwise the solver's duals are the (Float64) infeasibility certificate.
`generalized_decay_rate` gives the rate of a given `P`.

---

### completion_margin

*[src/ph/structure.jl:158](../../src/ph/structure.jl#L158)*

    completion_margin(A, Hfix, mask, basis; optimizer, cond_cap, nonnegative, silent = true)
        -> (feasible, t, alpha, P, F, coefficients, record)

The completion with the largest relative decay margin, the first-order proxy of a certified
region's size: maximise `t` subject to `P = alpha Hfix + F + sum k_j basis[j]` (`alpha >= 0`,
`F` free on `mask`), `I/cond_cap <= P <= I` and `-(A'P + PA) >= t I`. Freeing `alpha` is a
normalisation only: `P/alpha` has the physical core at unit weight and the same pattern, and
a positive multiple certifies the same sets. Implemented in the JuMP extension; the result is
a candidate.

---

### minimal_completion

*[src/ph/structure.jl:171](../../src/ph/structure.jl#L171)*

    minimal_completion(A, Hfix, mask; optimizer, mu, cond_cap, silent = true)
        -> (feasible, F, P, gamma, norm, record, dual_P, dual_cond, dual_rate)

The smallest correction on a pattern: minimise the Frobenius norm of `F` (symmetric, free on
`mask`) subject to `gamma I <= P <= cond_cap gamma I` with `P = Hfix + F` and
`-(A'P + PA) >= 2 mu P`. Work in scaled coordinates (the norm and `cond_cap` are scaling
dependent). `feasible` when the solver converged; the duals are the (Float64) infeasibility
certificate otherwise. Implemented in the JuMP extension; the result is a candidate.

---

### generalized_decay_rate

*[src/ph/structure.jl:183](../../src/ph/structure.jl#L183)*

    generalized_decay_rate(P, A) -> Real

The largest `mu` with `-(A'P + PA) >= 2 mu P` for `P > 0`: half the smallest eigenvalue of
`-(A'P + PA)` against `P`. Invariant under a change of coordinates (a congruence of `P`
with the similarity of `A`).

---

### pow2_scaling

*[src/ph/structure.jl:217](../../src/ph/structure.jl#L217)*

    pow2_scaling(d) -> Diagonal

The diagonal scaling `T` with `T_ii` the power of two nearest to `1/sqrt(d_i)` (e.g. `d` the
diagonal of the dense Lyapunov solution). Scaling by powers of two is exact in floating point.

---

### decay_margin

*[src/ph/structure.jl:225](../../src/ph/structure.jl#L225)*

    decay_margin(As, mask; optimizer, weights = nothing, rate = nothing, silent = true)
        -> NamedTuple

The decay margin `gamma(S)` of the pattern `mask` (see the comment above `pow2_scaling`):
minimise `lambda` over `P` on the pattern with `P >= 0`, `tr P = 1`, `lambda I - (As'P + PAs)
>= 0`. With `weights` (and `rate > 0`), minimise instead `sum(weights .* |P|)` subject to
`As'P + PAs <= -rate I` (the fewest weighted entries that keep a margin `rate`).
Returns `P`, `gamma` (the computed `lambda_max(As'P + PAs)`), the duals `Z` (of the rate
constraint) and `M` (of `P >= 0`), and `record` (solver statuses, time, iterations, and
residuals computed here, not taken from the solver). Implemented in the JuMP extension.
The result is a candidate; `pattern_certificate` and `verified_lyapunov` judge it.

---

### margin_residuals

*[src/ph/structure.jl:240](../../src/ph/structure.jl#L240)*

    margin_residuals(As, mask, P, Z, M) -> Dict

Primal and dual residuals of a `decay_margin` solution, computed in Float64 from the returned
matrices: `P` off the pattern, `lambda_min(P)`, `|tr P - 1|`; `lambda_min(Z)`, `|tr Z - 1|`,
`lambda_min(M)`; the dual stationarity on the pattern (`As Z + Z As' - M` a multiple of the
identity on S) and the duality gap.

---

### verified_min_eig

*[src/ph/structure.jl:263](../../src/ph/structure.jl#L263)*

    verified_min_eig(Y) -> Float64

A rigorous lower bound `c` on the smallest eigenvalue of every symmetric matrix in the
interval matrix `Y` (`-Inf` when the check fails). Method: `X` the approximate eigenvectors of
`mid(Y)`; if the interval matrix `X'(Y - cI)X` is strictly diagonally dominant with a
positive diagonal (Gershgorin), then `X'(Y - cI)X` is positive definite, `X` is nonsingular,
and by Sylvester's law of inertia `Y - cI > 0`.

---

### pattern_certificate

*[src/ph/structure.jl:295](../../src/ph/structure.jl#L295)*

    pattern_certificate(As, mask, Z, M) -> NamedTuple

Checks the dual certificate of `decay_margin` for the pattern `mask`: `L = V sqrt(max(D, 0))`
from the eigen-decomposition of `Z` (so `Z = LL'` exactly, as a real matrix), `Y` equal to
`As Z + Z As'` on the pattern and to `M` off it. Returns the Float64 bound `lambda_min(Y) /
tr Z` and the rigorous one (`L L'`, `As Z + Z As'` and the trace in interval arithmetic,
`verified_min_eig`), for the matrix `As` as given. `infeasible = true` means the rigorous
bound is positive: no quadratic Lyapunov function with this pattern exists for `As`.

---

### verified_lyapunov

*[src/ph/structure.jl:326](../../src/ph/structure.jl#L326)*

    verified_lyapunov(P, As) -> NamedTuple

Rigorous check of a candidate `P` (a strict Lyapunov function of `As` when both bounds are
positive): lower bounds on `lambda_min(P)` and on `lambda_min(-(As'P + PAs))`, in interval
arithmetic (`verified_min_eig`), for the matrices as given.

---

### rank_couplings

*[src/ph/structure.jl:343](../../src/ph/structure.jl#L343)*

    rank_couplings(G, mask, groups; by = :component) -> Vector{NamedTuple}

The blocks missing from the pattern, ranked by the Frobenius norm of `G` over their
off-pattern entries (`G = As Z + Z As' - M` from `decay_margin`'s duals: the rate at which
freeing those entries lowers the margin). `by = :component` groups coordinates by component,
`:unit` by unit, `:state` lists single entries.

---

### couple_states!

*[src/ph/structure.jl:371](../../src/ph/structure.jl#L371)*

    couple_states!(mask, groups, f, g) -> mask

Frees every entry between a coordinate whose group satisfies `f` and one whose group
satisfies `g`, symmetrically (e.g. `f = g -> g.state == "omega"`, `g = g -> g.angle`: all
speed-angle cross-terms).

---

### add_coupling!

*[src/ph/structure.jl:388](../../src/ph/structure.jl#L388)*

    add_coupling!(mask, groups, a, b; by = :component) -> mask

Frees every entry between the coordinates of `a` and those of `b` (component or unit names,
per `by`), symmetrically.

---

### TerminalModel

*[src/ph/terminal.jl:15](../../src/ph/terminal.jl#L15)*

    TerminalModel

A unit (a machine and its controllers) at its terminal bus: `x' = A x + B dV`,
`dI = C x + D dV`, with `dV = (dVd, dVq)` the bus voltage and `dI` the current the unit
injects into the network.

---

### NetworkModel

*[src/ph/terminal.jl:37](../../src/ph/terminal.jl#L37)*

    NetworkModel

The network with the loads, seen from the buses: `x' = A x + B dV`, `dI = C x + D dV`, with
`dV` all bus voltages `(Vd_1, Vq_1, ...)` and `dI` the current it absorbs at each bus.

---

### sync_jacobian

*[src/ph/terminal.jl:51](../../src/ph/terminal.jl#L51)*

    sync_jacobian(sys, x, V) -> J

The Jacobian of `[f; g]` with respect to `[x; V]` in the synchronous frame: the COI
correction `-omega_b (omega_COI - 1)` is taken out of every rotor-angle row.

---

### terminal_models

*[src/ph/terminal.jl:70](../../src/ph/terminal.jl#L70)*

    terminal_models(sys, x, V; projection) -> (units, network)

Cut the system at the machine terminals (see the file header). Every coupling not through a
terminal voltage is checked to be absent.

---

### terminal_margins

*[src/ph/terminal.jl:138](../../src/ph/terminal.jl#L138)*

    terminal_margins(units, net, bus_ids; ws) -> Dict

Frequency tests of the terminal cut (incremental supply dV'dI, power into each part):
- per unit, `min_w eigmin Herm(Y_u(jw))`, `Y_u = -G_u` its absorbing admittance: >= 0 means
  the unit alone is passive at its terminal; the most negative value is its shortage;
- the network, `min_w eigmin Herm(Y_net(jw))`;
- the cut, `min_w eigmin Herm(Y_net(jw) + sum_u E_u Y_u(jw) E_u')`: > 0 at every frequency
  means a storage split unit by unit exists with frequency-dependent terms; < 0 rules
  such a split out.
Each with the frequency where the minimum is reached.

