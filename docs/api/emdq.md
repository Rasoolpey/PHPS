# Electromechanical dq models

Reduced dq pilot models for stability analysis: network, AVR, phase-free forms and point-of-common-coupling views. Source: [src/emdq/](../../src/emdq).

Generated from the docstrings by `scripts/build_api_docs.jl`; do not edit by hand.

| Name | Defined in |
|---|---|
| [`EMDQAVR`](#emdqavr) | `avr.jl` |
| [`emdq_avr`](#emdq_avr) | `avr.jl` |
| [`emdq_avr_energy`](#emdq_avr_energy) | `avr.jl` |
| [`emdq_modulation`](#emdq_modulation) | `avr.jl` |
| [`emdq_closed_loop_rhs!`](#emdq_closed_loop_rhs) | `avr.jl` |
| [`emdq_closed_loop_balance`](#emdq_closed_loop_balance) | `avr.jl` |
| [`emdq_completion_data`](#emdq_completion_data) | `avr.jl` |
| [`completion_farkas_check`](#completion_farkas_check) | `avr.jl` |
| [`emdq_local_obstruction`](#emdq_local_obstruction) | `avr.jl` |
| [`emdq_verify_obstruction`](#emdq_verify_obstruction) | `avr.jl` |
| [`emdq_repair_report`](#emdq_repair_report) | `avr.jl` |
| [`emdq_bm_signature`](#emdq_bm_signature) | `bm.jl` |
| [`emdq_bm_Q`](#emdq_bm_q) | `bm.jl` |
| [`emdq_bm_potential`](#emdq_bm_potential) | `bm.jl` |
| [`emdq_bm_terms`](#emdq_bm_terms) | `bm.jl` |
| [`emdq_bm_balance`](#emdq_bm_balance) | `bm.jl` |
| [`emdq_bm_sharp`](#emdq_bm_sharp) | `bm.jl` |
| [`emdq_bm_rate_check`](#emdq_bm_rate_check) | `bm.jl` |
| [`bm_multiplier_feasibility`](#bm_multiplier_feasibility) | `bm.jl` |
| [`bm_farkas_check`](#bm_farkas_check) | `bm.jl` |
| [`emdq_balance`](#emdq_balance) | `gates.jl` |
| [`emdq_shifted_energy`](#emdq_shifted_energy) | `gates.jl` |
| [`emdq_modulation_output`](#emdq_modulation_output) | `gates.jl` |
| [`emdq_shifted_balance`](#emdq_shifted_balance) | `gates.jl` |
| [`emdq_local_forms`](#emdq_local_forms) | `gates.jl` |
| [`restricted_max_eig`](#restricted_max_eig) | `gates.jl` |
| [`emdq_reduction_check`](#emdq_reduction_check) | `gates.jl` |
| [`emdq_angle_augmentation`](#emdq_angle_augmentation) | `gates.jl` |
| [`emdq_angle_augmentation_rate`](#emdq_angle_augmentation_rate) | `gates.jl` |
| [`emdq_augmented_forms`](#emdq_augmented_forms) | `gates.jl` |
| [`emdq_component_blocks`](#emdq_component_blocks) | `krasovskii.jl` |
| [`emdq_component_mask`](#emdq_component_mask) | `krasovskii.jl` |
| [`emdq_krasovskii_energy`](#emdq_krasovskii_energy) | `krasovskii.jl` |
| [`emdq_krasovskii_balance`](#emdq_krasovskii_balance) | `krasovskii.jl` |
| [`emdq_verify_krasovskii`](#emdq_verify_krasovskii) | `krasovskii.jl` |
| [`block_trace_obstruction`](#block_trace_obstruction) | `krasovskii.jl` |
| [`verified_block_instability`](#verified_block_instability) | `krasovskii.jl` |
| [`block_union_abscissae`](#block_union_abscissae) | `krasovskii.jl` |
| [`block_lyapunov_farkas_check`](#block_lyapunov_farkas_check) | `krasovskii.jl` |
| [`block_lyapunov_conditioning`](#block_lyapunov_conditioning) | `krasovskii.jl` |
| [`krasovskii_conditioning`](#krasovskii_conditioning) | `krasovskii.jl` |
| [`PhysicalNetwork`](#physicalnetwork) | `network.jl` |
| [`physical_network`](#physical_network) | `network.jl` |
| [`pn_incidence`](#pn_incidence) | `network.jl` |
| [`pn_flows`](#pn_flows) | `network.jl` |
| [`pn_energy`](#pn_energy) | `network.jl` |
| [`pn_balance`](#pn_balance) | `network.jl` |
| [`pn_descriptor`](#pn_descriptor) | `network.jl` |
| [`pn_quasistatic_ybus`](#pn_quasistatic_ybus) | `network.jl` |
| [`pn_node_classes`](#pn_node_classes) | `network.jl` |
| [`emdq_pilot_digest`](#emdq_pilot_digest) | `nogo.jl` |
| [`emdq_big_data`](#emdq_big_data) | `nogo.jl` |
| [`emdq_interval_data`](#emdq_interval_data) | `nogo.jl` |
| [`emdq_nogo_certificate`](#emdq_nogo_certificate) | `nogo.jl` |
| [`check_nogo_certificate`](#check_nogo_certificate) | `nogo.jl` |
| [`emdq_radius_masks`](#emdq_radius_masks) | `nogo.jl` |
| [`emdq_verify_lyapunov`](#emdq_verify_lyapunov) | `nogo.jl` |
| [`GridPH`](#gridph) | `pccreduce.jl` |
| [`emdq2_grid_ph`](#emdq2_grid_ph) | `pccreduce.jl` |
| [`grid_ph_to_rotor`](#grid_ph_to_rotor) | `pccreduce.jl` |
| [`grid_ph_from_rotor`](#grid_ph_from_rotor) | `pccreduce.jl` |
| [`grid_ph_JRg`](#grid_ph_jrg) | `pccreduce.jl` |
| [`grid_ph_rhs`](#grid_ph_rhs) | `pccreduce.jl` |
| [`grid_ph_energy`](#grid_ph_energy) | `pccreduce.jl` |
| [`grid_ph_power`](#grid_ph_power) | `pccreduce.jl` |
| [`ReducedGrid`](#reducedgrid) | `pccreduce.jl` |
| [`pcc_L`](#pcc_l) | `pccreduce.jl` |
| [`reduced_grid`](#reduced_grid) | `pccreduce.jl` |
| [`reduced_rhs`](#reduced_rhs) | `pccreduce.jl` |
| [`reduced_project`](#reduced_project) | `pccreduce.jl` |
| [`pcc_qorth`](#pcc_qorth) | `pccreduce.jl` |
| [`pcc_add_pair`](#pcc_add_pair) | `pccreduce.jl` |
| [`pcc_krylov_pool`](#pcc_krylov_pool) | `pccreduce.jl` |
| [`Jet`](#jet) | `pccview.jl` |
| [`jet_flow`](#jet_flow) | `pccview.jl` |
| [`pcc_words`](#pcc_words) | `pccview.jl` |
| [`pcc_coefficient_values`](#pcc_coefficient_values) | `pccview.jl` |
| [`pcc_coefficient_jacobian`](#pcc_coefficient_jacobian) | `pccview.jl` |
| [`pcc_two_input_jacobian`](#pcc_two_input_jacobian) | `pccview.jl` |
| [`pcc_structural_zero`](#pcc_structural_zero) | `pccview.jl` |
| [`pcc_profile`](#pcc_profile) | `pccview.jl` |
| [`EMDQGovernor`](#emdqgovernor) | `phasefree.jl` |
| [`EMDQPhaseFree`](#emdqphasefree) | `phasefree.jl` |
| [`emdq_pf_nstates`](#emdq_pf_nstates) | `phasefree.jl` |
| [`emdq_pf_state_names`](#emdq_pf_state_names) | `phasefree.jl` |
| [`emdq_gov_torque`](#emdq_gov_torque) | `phasefree.jl` |
| [`emdq_gov_rhs`](#emdq_gov_rhs) | `phasefree.jl` |
| [`emdq_gov_mode_margins`](#emdq_gov_mode_margins) | `phasefree.jl` |
| [`emdq_governed`](#emdq_governed) | `phasefree.jl` |
| [`emdq_phasefree`](#emdq_phasefree) | `phasefree.jl` |
| [`emdq_pf_rhs!`](#emdq_pf_rhs) | `phasefree.jl` |
| [`emdq_pf_energy`](#emdq_pf_energy) | `phasefree.jl` |
| [`emdq_pf_balance`](#emdq_pf_balance) | `phasefree.jl` |
| [`emdq_pf_phasor_pairs`](#emdq_pf_phasor_pairs) | `phasefree.jl` |
| [`emdq_pf_rotate`](#emdq_pf_rotate) | `phasefree.jl` |
| [`EMDQPFClosedLoop`](#emdqpfclosedloop) | `phasefree.jl` |
| [`emdq_pf_reduce`](#emdq_pf_reduce) | `phasefree.jl` |
| [`emdq_pf_lift`](#emdq_pf_lift) | `phasefree.jl` |
| [`emdq_pf_reduced_rhs`](#emdq_pf_reduced_rhs) | `phasefree.jl` |
| [`emdq_pf_quotient_data`](#emdq_pf_quotient_data) | `phasefree.jl` |
| [`emdq_pf_masks`](#emdq_pf_masks) | `phasefree.jl` |
| [`emdq_pf_interval_data`](#emdq_pf_interval_data) | `phasefree.jl` |
| [`WindingMachine`](#windingmachine) | `pilot.jl` |
| [`winding_machine`](#winding_machine) | `pilot.jl` |
| [`EMDQPilot`](#emdqpilot) | `pilot.jl` |
| [`EMDQ_STATES`](#emdq_states) | `pilot.jl` |
| [`emdq_state_names`](#emdq_state_names) | `pilot.jl` |
| [`emdq_machine_block`](#emdq_machine_block) | `pilot.jl` |
| [`emdq_machine`](#emdq_machine) | `pilot.jl` |
| [`emdq_rhs!`](#emdq_rhs) | `pilot.jl` |
| [`emdq_energy_parts`](#emdq_energy_parts) | `pilot.jl` |
| [`emdq_energy`](#emdq_energy) | `pilot.jl` |
| [`emdq_power`](#emdq_power) | `pilot.jl` |
| [`emdq_pilot`](#emdq_pilot) | `pilot.jl` |
| [`EMDQSite`](#emdqsite) | `pilot2.jl` |
| [`EMDQPilot2`](#emdqpilot2) | `pilot2.jl` |
| [`emdq2_grid_layout`](#emdq2_grid_layout) | `pilot2.jl` |
| [`emdq2_nc`](#emdq2_nc) | `pilot2.jl` |
| [`emdq2_nstates`](#emdq2_nstates) | `pilot2.jl` |
| [`emdq2_grid_state_names`](#emdq2_grid_state_names) | `pilot2.jl` |
| [`emdq2_grid_rhs!`](#emdq2_grid_rhs) | `pilot2.jl` |
| [`emdq2_rhs!`](#emdq2_rhs) | `pilot2.jl` |
| [`emdq2_grid_energy_parts`](#emdq2_grid_energy_parts) | `pilot2.jl` |
| [`emdq2_grid_energy`](#emdq2_grid_energy) | `pilot2.jl` |
| [`emdq2_energy`](#emdq2_energy) | `pilot2.jl` |
| [`emdq2_grid_power`](#emdq2_grid_power) | `pilot2.jl` |
| [`emdq2_balance`](#emdq2_balance) | `pilot2.jl` |
| [`emdq2_site2_shifted`](#emdq2_site2_shifted) | `pilot2.jl` |
| [`emdq_pilot2`](#emdq_pilot2) | `pilot2.jl` |

---

### EMDQAVR

*[src/emdq/avr.jl:5](../../src/emdq/avr.jl#L5)*

    EMDQAVR(kappa_p, omega_i, u0, y0)

Dimensionless passive PI on the converter's shifted output `y_m`:

    yhat = y_m / y0,   eta' = omega_i yhat,   u = m - m* = u0 (-kappa_p yhat - eta),
    H_AVR = u0 y0 eta^2 / (2 omega_i),

so `u y_m + dH_AVR/dt = -u0 y0 kappa_p yhat^2` exactly (while the modulation limit is inactive).
`u0` is the declared modulation headroom, `y0` the declared output scale.

---

### emdq_avr

*[src/emdq/avr.jl:23](../../src/emdq/avr.jl#L23)*

    emdq_avr(s, xs; kappa_p, omega_i) -> EMDQAVR

The PI with the declared scales of the pilot: headroom `u0 = min(m*, 1 - m*)` (a
two-quadrant converter, `0 <= m <= 1`) and output scale `y0 = v_dc* i_fd*` (the nominal field
power), so `yhat = i_fd/i_fd* - v_dc/v_dc*`.

---

### emdq_avr_energy

*[src/emdq/avr.jl:36](../../src/emdq/avr.jl#L36)*

    emdq_avr_energy(c, eta) -> H_AVR = u0 y0 eta^2 / (2 omega_i) (controller storage).

---

### emdq_modulation

*[src/emdq/avr.jl:39](../../src/emdq/avr.jl#L39)*

    emdq_modulation(s, xs, c, z) -> m = m* + u0 (-kappa_p yhat - eta) for z = [x; eta].

---

### emdq_closed_loop_rhs!

*[src/emdq/avr.jl:46](../../src/emdq/avr.jl#L46)*

    emdq_closed_loop_rhs!(dz, s, xs, c, z) -> dz

The pilot with the PI: `z = [x; eta]` (the plant states, then `eta`), the modulation from
`emdq_modulation`.

---

### emdq_closed_loop_balance

*[src/emdq/avr.jl:63](../../src/emdq/avr.jl#L63)*

    emdq_closed_loop_balance(s, xs, c, z) -> Dict

Gate 1.5: `d/dt (H_s + H_AVR) = -D_inc + r_rot - u0 y0 kappa_p yhat^2` along the closed
loop, every term and `identity_error` against the exact rate.

---

### emdq_completion_data

*[src/emdq/avr.jl:85](../../src/emdq/avr.jl#L85)*

    emdq_completion_data(s, xs, c) -> NamedTuple

The data of the own-unit completion at the equilibrium `[xs; 0]`: the closed-loop Jacobian
`A`, the fixed physical-plus-controller Hessian `Hfix = blockdiag(Hess H_physical,s,
u0 y0/omega_i)`, the declared power-of-two scaling `d` (unit physical curvature per
coordinate: `d_i = 2^round(-log2(Hfix_ii)/2)`), the scaled `As = D^-1 A D`, `Hs = D Hfix D`,
and the patterns: `local_mask` on the machine block (`emdq_machine_block`: `psi_D, psi_Q,
psi_fd, psi_1d, psi_1q, delta, p`, and `psi_2q` for a round rotor) with `eta`,
`excitation_mask`, which adds the `q_dc - psi_fd` and `q_dc - eta` cross-terms (a separate
test), and `port_mask`, which adds only the cross-terms between that block and the bus-2
voltage `{v2_Re, v2_Im}` (the nearest neighbour across the stator port; the capacitor's own
block stays physical). `block` lists the local indices, `eta` the integrator's.

---

### completion_farkas_check

*[src/emdq/avr.jl:124](../../src/emdq/avr.jl#L124)*

    completion_farkas_check(A, H, mask, r; mu, cond_cap) -> NamedTuple

Checks, in Float64, the dual certificate that `completion_rate_feasibility` returns when the
pattern is infeasible. With duals `Y1` (of `P >= g I`), `Y2` (of `P <= cond_cap g I`), `Y3`
(of `-(A'P + PA) >= 2 mu P`), all PSD, and `M = Y1 - Y2 - (A Y3 + Y3 A') - 2 mu Y3`, the
conditions are `M = 0` on the pattern, `cond_cap tr Y2 - tr Y1 = 0` and `<M, H> < 0`: then
every feasible point would give `0 <= <M, H> < 0`. Returns the normalised residuals
(`pattern`, `trace`, relative to `|<M, H>|`), `MH` and the smallest eigenvalues of the duals.

---

### emdq_local_obstruction

*[src/emdq/avr.jl:146](../../src/emdq/avr.jl#L146)*

    emdq_local_obstruction(data; mask = data.local_mask) -> NamedTuple

Solver-free structural test of a correction pattern. Let `R` be the states that `mask`
touches (any free entry in their row). On `K = {v : (A v)_R = 0}` the rate of every
`P = Hfix + F` with `F` free on `mask` is `v' (A'P + PA) v = 2 v' Hfix (A v)`: `F` drops out
because each term `v_i F_ij (A v)_j` has `j` in `R`. If this fixed form has a positive
eigenvalue on `K` (`max_rate > 0`), then `-(A'P + PA)` is not positive semidefinite for any
such `P`: no completion on the pattern decays, at any rate target or condition cap (the
gains enter only through `A`). Returns `max_rate` (scaled coordinates), the direction, `dim_K`
and `R`. `emdq_verify_obstruction` proves the sign in interval arithmetic.

---

### emdq_verify_obstruction

*[src/emdq/avr.jl:169](../../src/emdq/avr.jl#L169)*

    emdq_verify_obstruction(data, w; mask = data.local_mask, rel = 1e-6, abs_A = 0.0,
                            abs_H = 0.0, AsI = nothing, HsI = nothing) -> NamedTuple

With `AsI`, `HsI` (the model-bound enclosures of `emdq_interval_data`) the proof uses them
instead of the boxes below and holds for the pilot model itself.

Interval proof of `emdq_local_obstruction`'s sign for an explicit witness. The witness is
parameterised by its components `w` (Float64, exact) outside `R`: for each `A` the exact
vector of `K` with those components is `v_R = -A_RR^-1 A_RF w`. Every entry of the scaled
`As` and `Hs` is widened by `rel |x|` plus the absolute radius `abs_A` or `abs_H`, so the
conclusion holds for every linearisation and Hessian in that box (choose the absolute radii
above the measured data error: entries that are cancellation noise in Float64 are not covered
by a relative radius). `interval_solve` encloses `v_R` (proving every `A_RR` in the box
nonsingular), and the rate `v' Hs (As v)`, with `(As v)_R = 0` exactly, is enclosed.
`proved` when its lower bound is positive.

---

### emdq_repair_report

*[src/emdq/avr.jl:201](../../src/emdq/avr.jl#L201)*

    emdq_repair_report(data, P) -> NamedTuple

The size and effect of a completed `P` (scaled coordinates, as returned by the LMI) against
`Hs` (scaled `Hfix`): `R = P - Hs`; Frobenius and operator-norm ratios
`|R| / |Hs|`; support (entries above 1e-9 of the largest) and its density in the symmetric
matrix; extremal eigenvalues and condition numbers of `Hs` and `P`; the generalised decay rate
of `P`; the curvature of `Hs`, `R` and `P` on `Hs`'s most negative direction; and the rate
form `sym(P As)` on the slice `{Ddelta = Dp = 0}` (before: `Hs`; after: `P`).

---

### emdq_bm_signature

*[src/emdq/bm.jl:11](../../src/emdq/bm.jl#L11)*

    emdq_bm_signature(s) -> Vector{Float64}

The fixed BM sign hypothesis on `z = [x; eta]`: `-1` on the inductive, flux and momentum
states (machine fluxes, `p`, line and load-inductor currents), `+1` on the capacitive, charge
and position states (bus voltages, `q_dc`, the rotor angle) and on the PI integrator.

---

### emdq_bm_Q

*[src/emdq/bm.jl:27](../../src/emdq/bm.jl#L27)*

    emdq_bm_Q(s, c, z) -> (; Q, sym, skew)

The signed BM differential matrix `Q = Sigma Hess(E_phys + H_AVR)` at `z`, with its symmetric
and skew parts.

---

### emdq_bm_potential

*[src/emdq/bm.jl:56](../../src/emdq/bm.jl#L56)*

    emdq_bm_potential(s, z) -> P(z)

The mixed potential of the pilot, element by element, with the modulation frozen at `m*`
(co-state currents and voltages as functions of the state):
- winding and line contents `1/2 R |i|^2`;
- minus the co-contents of the load conductance `1/2 G3 |v3|^2` and of the DC source
  `(vs - v_dc)^2 / (2 Rs)`;
- the pairings `-I' u*` (stator current with `v2/tap`, field current with `m* v_dc`),
  `-i23 (v2 - v3)`, `-i34 (v3 - V4)`, `-iL v3`;
- the constant shaft source `-Tm omega` and the diagnostic damper content `1/2 Dm (omega - 1)^2`.
The PI integrator does not appear (its coupling is a named term).

---

### emdq_bm_terms

*[src/emdq/bm.jl:85](../../src/emdq/bm.jl#L85)*

    emdq_bm_terms(s, xs, c, z) -> (; gyr, mod, em)

The named non-gradient terms, each in closed form from the element equations (`z = [x; eta]`,
the PI `c`, its reference `xs`), with `K = dI/dPsi` and `dI/d delta` from the machine's
current map (not from the energy Hessian):
- `gyr`: the synchronous-frame gyrations. Flux rows `K (j psi_s; 0)`; capacitors `-j C v`;
  inductors `+j L i`.
- `mod`: the modulation's departure from `m*` through the PI. Flux rows
  `-(m - m*) v_dc K[:, fd]`; the DC link `-(m - m*) i_fd / C_dc`; the PI row `u0 y_m`.
- `em`: the electromechanical coupling. Flux rows `-(omega - 1) dI/d delta`; the momentum
  row `-T_em/(2H)`; the angle row
  `-(1/wb)(grad_Psi T_em . dPsi/dt + dT_em/d delta . d delta/dt) - (dI/d delta)'(R I - u*)`.

---

### emdq_bm_balance

*[src/emdq/bm.jl:136](../../src/emdq/bm.jl#L136)*

    emdq_bm_balance(s, xs, c, z) -> NamedTuple

Gate B0's structural identity: `Q F - grad P - (N_gyr + N_mod + N_em)`, with `Q F` from the
Hessian of the energy and the field, `grad P` by automatic differentiation of the potential,
and the named terms from `emdq_bm_terms`. Returns the residual vector, its norm relative to the
sum of the parts' norms, and the norm of each part.

---

### emdq_bm_sharp

*[src/emdq/bm.jl:154](../../src/emdq/bm.jl#L154)*

    emdq_bm_sharp(s, zs, z, lambda, Mw) -> P#(z)

The BM candidate `lambda P_s + 1/2 grad P_s' Mw grad P_s`, with the potential shifted at the
equilibrium `zs` (unscaled coordinates; `Mw` constant symmetric).

---

### emdq_bm_rate_check

*[src/emdq/bm.jl:167](../../src/emdq/bm.jl#L167)*

    emdq_bm_rate_check(s, xs, c, zs, z, lambda, Mw) -> NamedTuple

The exact rate identity of section 24.4:
`dP#/dt = xdot' Q'(lambda I + Mw Hess P) xdot - dN'(lambda I + Mw Hess P) xdot`, with
`dN = N(z) - N(zs)` (`N` the sum of the named terms). The rate along the flow is the gradient
of `P#` dotted with the field; `identity_error` is relative to the terms' sizes.

---

### bm_multiplier_feasibility

*[src/emdq/bm.jl:189](../../src/emdq/bm.jl#L189)*

    bm_multiplier_feasibility(As, Hs, mask; mu, cond_cap, optimizer, silent = true)
        -> (feasible, lambda, M, S, record, dual_lower, dual_upper, dual_rate)

Gate B1's local LMI in the declared scaled coordinates: find a scalar `lambda` and a
symmetric `M` free on `mask` (zero elsewhere) with `S = lambda Hs + Hs M Hs`,
`I <= S <= cond_cap I` (the conditioning LMI with `alpha = 1`, which the homogeneity in
`(lambda, M)` allows) and `-(As'S + S As) >= 2 mu S`. `Hs` is the scaled Hessian of the mixed
potential at the equilibrium. A converged solve means feasible; a proper `INFEASIBLE`
status comes with the duals for `bm_farkas_check`. Implemented in the JuMP extension; the
result is a candidate.

---

### bm_farkas_check

*[src/emdq/bm.jl:203](../../src/emdq/bm.jl#L203)*

    bm_farkas_check(As, Hs, mask, r; mu, cond_cap) -> NamedTuple

Float64 check of the infeasibility certificate of `bm_multiplier_feasibility`. With PSD duals
`Y1` (of `S >= I`), `Y2` (of `S <= cond_cap I`), `Y3` (of the rate) and
`Z = Y1 - Y2 - (As Y3 + Y3 As') - 2 mu Y3`, a certificate has `<Z, Hs> = 0` (the coefficient of
`lambda`), `(Hs Z Hs) = 0` on the mask (the coefficients of `M`) and
`tr Y1 - cond_cap tr Y2 > 0`. Every feasible point would then give
`0 <= -(tr Y1 - cond_cap tr Y2) < 0`. The residuals are normalised: `<Z, Hs>` by
`|Z| |Hs|`, the masked `Hs Z Hs` by `|Hs Z Hs|`, and the gap by `tr Y1 + cond_cap tr Y2`.
A solver certificate, not an interval proof.

---

### emdq_balance

*[src/emdq/gates.jl:7](../../src/emdq/gates.jl#L7)*

    emdq_balance(s::EMDQPilot, x, m = s.mstar) -> Dict

Raw physical-energy balance (gate 1.2): `rate = grad H' f` against
`sum(supplies) - sum(dissipation)` (`emdq_power`), with every term and `identity_error`.

---

### emdq_shifted_energy

*[src/emdq/gates.jl:27](../../src/emdq/gates.jl#L27)*

    emdq_shifted_energy(s, xs, x) -> H(x) - H(xs) - grad H(xs)' (x - xs)

The equilibrium-shifted (Bregman) energy in the physical coordinates: stator fluxes in the
network frame, rotor fluxes, the rotor angle and momentum, the DC charge and the network
states.

---

### emdq_modulation_output

*[src/emdq/gates.jl:39](../../src/emdq/gates.jl#L39)*

    emdq_modulation_output(s, xs, x) -> y_m

The converter's incremental passive output `y_m = g_m' grad H_s` with
`g_m = df/dm` (field flux `omega_b v_dc`, DC charge `-i_fd`); in closed form
`y_m = v_dc* i_fd - i_fd* v_dc`. Zero for the raw energy (`grad H` instead of `grad H_s`):
the ideal converter is power preserving.

---

### emdq_shifted_balance

*[src/emdq/gates.jl:52](../../src/emdq/gates.jl#L52)*

    emdq_shifted_balance(s::EMDQPilot, xs, x, m = s.mstar) -> Dict

The shifted identity (gates 1.3 and 1.4) at an equilibrium `xs` of modulation `s.mstar`:

    dH_s/dt = -D_inc + r + (m - m*) y_m,

with the incremental dissipation `D_inc` (each resistor's `R |i - i*|^2`, the load's
`G |v - v*|^2`, the diagnostic shaft damping's `Dm (omega - 1)^2`) and the single residual `r = -Re(conj(Delta i_n) j Delta psi_n)` of the
stator's frame rotation (`-j psi_n` in the network-frame stator equation; with the exact
angle and torque pairing `dH/d delta = -Tem/omega_b`, everything else cancels). Returns every
term and `identity_error` against the exact rate `(grad H(x) - grad H(xs))' f(x, m)`.

---

### emdq_local_forms

*[src/emdq/gates.jl:93](../../src/emdq/gates.jl#L93)*

    emdq_local_forms(s::EMDQPilot, xs) -> NamedTuple

Second-order forms at the equilibrium `xs` (modulation `s.mstar`): the Bregman Hessian `Hs`
(= Hess H(xs)), the Jacobian `A`, the rate form `Q = sym(Hs A)` (so `dH_s/dt = dx' Q dx +
O(|dx|^3)`), its parts `D` (incremental dissipation, PSD) and `Rr` (rotation residual) with
`closure = |Q - (-D + Rr)|`, and the linear functional `c` of `y_m` (`y_m = c' dx + O(2)`).

---

### restricted_max_eig

*[src/emdq/gates.jl:148](../../src/emdq/gates.jl#L148)*

    restricted_max_eig(Q, c) -> max eigenvalue of the form `Q` on the hyperplane `c' dx = 0`

Finsler test for gate 1.3: if it is positive, no damping injection through `y_m` alone
(`u = -k y_m`, any `k`) makes `dH_s/dt` negative semidefinite.

---

### emdq_reduction_check

*[src/emdq/gates.jl:159](../../src/emdq/gates.jl#L159)*

    emdq_reduction_check(s::EMDQPilot, x) -> NamedTuple

Singular-limit comparison (gate requirement): with the stator quasi-static in the network
frame (`dpsi_n/dt = 0`, so `psi_n = -j (v_m - Ra i_n)`, the classical consistent pairing of
neglected stator transients with nominal speed in the stator EMF), the rotor derivatives of the
winding model, mapped to `E_q' = Lad psi_fd / Lffd`, `psi_1d`, `psi_q'' = Laq psi_1q / L11q`,
are compared with the Sauer-Pai subtransient model in closed form (expected: round-off).
Also returns the deviation of PHPS's GENSAL_PHTRUE kernel at the same point (no Sauer-Pai
correction in `dE_q'/dt`, speed-scaled EMF `omega psi''`), which vanishes at equilibria.
`x` supplies the rotor states, `delta`, `p` and the bus-2 voltage; only `Ra = 0` is handled.

---

### emdq_angle_augmentation

*[src/emdq/gates.jl:228](../../src/emdq/gates.jl#L228)*

    emdq_angle_augmentation(s, xs, x; sigma, c) -> Z

`Z = sigma (p - p*) sin(delta - delta*) + c (1 - cos(delta - delta*))`, the pilot's analogue
of Jiang's `sigma J (omega - omega*) sin(theta) + (sigma D - J omega*^2)(1 - cos(theta))`
(`p - p* = 2H (omega - 1)`; the kinetic energy `p^2/(4H)` has no angle curvature, so the
`J omega*^2` part of his angle coefficient has no counterpart here).

---

### emdq_angle_augmentation_rate

*[src/emdq/gates.jl:241](../../src/emdq/gates.jl#L241)*

    emdq_angle_augmentation_rate(s, xs, x; sigma, c) -> dZ/dt

Closed form along the pilot flow: `sigma sin(Ddelta) p' + (sigma (p - p*) cos(Ddelta) +
c sin(Ddelta)) omega_b (omega - 1)`, with `p' = Tm + Tem - Dm (omega - 1)`. Its quadratic part
at the equilibrium is `sigma Ddelta (DTem - Dm Domega) + sigma 2H omega_b Domega^2 +
c omega_b Domega Ddelta`: the angle dissipation comes from `sigma Ddelta DTem` (synchronising
torque), at the price of `sigma 2H omega_b Domega^2`, which only shaft damping can dominate.

---

### emdq_augmented_forms

*[src/emdq/gates.jl:260](../../src/emdq/gates.jl#L260)*

    emdq_augmented_forms(s, xs, F, sigma) -> NamedTuple

Local test of `V = H_s + Z` at the equilibrium for one `sigma`, with `c` fixed by the exact
balance (it cancels the angle-speed entry of the rate form, as Jiang's `sigma D` term does;
`c = sigma Dm / omega_b` when the shifted rate has no such entry). `F` from
`emdq_local_forms`. Returns `c`, `HV` (Hessian of V), `QV` (rate form), `min_HV` and the
Finsler value `finsler` (max of `QV` on `ker y_m`).

---

### emdq_component_blocks

*[src/emdq/krasovskii.jl:8](../../src/emdq/krasovskii.jl#L8)*

    emdq_component_blocks(s; partition = :finest) -> Vector{Pair{String,Vector{Int}}}

A partition of the closed-loop state `z = [x; eta]` of the pilot `s`, checked to cover `1:n`
exactly once:
- `:finest` (the finest physical partition): the machine (`emdq_machine_block`), the DC link
  (`q_dc`), the PI integrator (`eta`), each bus capacitor (`v2`, `v3`), each line-current pair
  (`i23`, `i34`) and the load inductor (`iL`);
- `:node_centred`: machine + DC link + PI integrator + terminal capacitor; first line;
  receiving-bus capacitor + load inductor; second line;
- `:edge_centred`: machine + DC link + PI integrator; terminal capacitor + first line;
  receiving-bus capacitor + load inductor; second line.
The two coarse partitions are the predeclared pair of the coarsening experiment
(docs/records/early/STABILITY_INVESTIGATION.md section 22); no other partition is searched.

---

### emdq_component_mask

*[src/emdq/krasovskii.jl:43](../../src/emdq/krasovskii.jl#L43)*

    emdq_component_mask(s; partition = :finest) -> BitMatrix

Full symmetric blocks of `emdq_component_blocks(s; partition)`, zero across blocks.

---

### emdq_krasovskii_energy

*[src/emdq/krasovskii.jl:58](../../src/emdq/krasovskii.jl#L58)*

    emdq_krasovskii_energy(s, xs, c, d, W, z) -> H_K(z) = 1/2 F(z)' W_u F(z)

The Krasovskii storage of the closed loop `F = emdq_closed_loop_rhs(s, xs, c, .)` for a weight
`W` given in the declared scaling `d.d` of `d = emdq_completion_data(s, xs, c)`: with
`D = diag(d)` the scaled field is `D^-1 F`, so `W_u = D^-1 W D^-1` and the Hessian at the
equilibrium is `A_s' W A_s` in scaled coordinates.

---

### emdq_krasovskii_balance

*[src/emdq/krasovskii.jl:71](../../src/emdq/krasovskii.jl#L71)*

    emdq_krasovskii_balance(s, xs, c, d, W, z) -> NamedTuple

The exact rate identity of `H_K` for constant `W`: `dH_K/dt = F' W_u DF F` (no remainder).
Returns the rate along the flow (gradient of `H_K` dotted with `F`), the closed form, and
`identity_error` relative to the size of the closed form's terms.

---

### emdq_verify_krasovskii

*[src/emdq/krasovskii.jl:89](../../src/emdq/krasovskii.jl#L89)*

    emdq_verify_krasovskii(W, AsI; mu) -> NamedTuple

Interval check of a constant weight `W` (scaled coordinates, taken as exact) for every scaled
closed-loop Jacobian in the box `AsI` (the model-bound enclosure of `emdq_interval_data`):
- `V_W = 1/2 dz' W dz`: rigorous lower bounds `W_min` on `lambda_min(W)` and `decay_min` on
  `lambda_min(-(A'W + WA) - 2 mu W)`;
- `H_K = 1/2 F'WF`, Hessian `P_K = A'WA`: rigorous bounds `PK_min <= lambda_min(P_K)`,
  `PK_max >= lambda_max(P_K)` and `PK_cond_max = PK_max / PK_min` over the box.
  `PK_min > 0` proves every `A` in the box nonsingular, and then
  `-(A'P_K + P_K A) - 2 mu P_K = A'(-(A'W + WA) - 2 mu W)A` is positive definite exactly when
  the `V_W` condition holds, so `decay_min > 0` also proves local decay of `H_K` at `mu`.
`proved_VW`, `proved_HK` when the respective bounds are positive.

---

### block_trace_obstruction

*[src/emdq/krasovskii.jl:115](../../src/emdq/krasovskii.jl#L115)*

    block_trace_obstruction(A, blocks; mu) -> Vector{NamedTuple}

Solver-free necessary condition for a block-diagonal Lyapunov matrix. If `W =
blockdiag(W_k) > 0` satisfies `-(A'W + WA) >= 2 mu W`, then each diagonal block gives
`A_kk'W_k + W_k A_kk <= -2 mu W_k`; the congruence by `W_k^(-1/2)` and the trace give
`tr A_kk <= -m_k mu` (`m_k` the block size). So a block with `tr A_kk + m_k mu > 0` excludes
every such `W` (and every Krasovskii weight of that form, by congruence). `A` may be a Float64
or an interval matrix (`AsI` of `emdq_interval_data`); with intervals the lower bound is
rigorous and `obstructs` holds for every matrix in the box. Returns, per block of `blocks`
(`emdq_component_blocks`), its name, size, the bounds of `tr A_kk + m_k mu` and `obstructs`.

---

### verified_block_instability

*[src/emdq/krasovskii.jl:137](../../src/emdq/krasovskii.jl#L137)*

    verified_block_instability(A, ix; mu) -> NamedTuple

Interval proof that the principal submatrix `A[ix, ix]` has an eigenvalue with real part
`> -mu` for every matrix in the box `A` (intervals; Float64 entries are taken as exact). If
`ix` is a union of blocks of a partition, this excludes every block-diagonal `W > 0` on that
partition with `-(A'W + WA) >= 2 mu W`: the `ix` block of that inequality involves only
`A[ix, ix]` and `W[ix, ix] > 0`, so it would force every eigenvalue of `A[ix, ix]` to have real
part `<= -mu`. The proof is the Ostrowski-Schneider inertia theorem: with
`S = A[ix, ix] + mu I` and `X` from the midpoint Lyapunov equation `S_m'X + X S_m = I`, if
`S'X + XS > 0` for every `S` in the box (`verified_min_eig`) then `S` has as many eigenvalues
with positive real part as `X` has positive eigenvalues, and `v'Xv > 0` (enclosed) for a
point vector `v` shows there is one. Returns `proved`, the bounds and `X`'s positive count.

---

### block_union_abscissae

*[src/emdq/krasovskii.jl:169](../../src/emdq/krasovskii.jl#L169)*

    block_union_abscissae(A, blocks) -> Vector{NamedTuple}

Float64 diagnostic: the spectral abscissa of `A[T, T]` for every nonempty union `T` of the
blocks, sorted from the largest. A union with abscissa `>= -mu` violates the necessary
condition of `verified_block_instability` (which proves it on intervals).

---

### block_lyapunov_farkas_check

*[src/emdq/krasovskii.jl:186](../../src/emdq/krasovskii.jl#L186)*

    block_lyapunov_farkas_check(A, mask, r; mu) -> NamedTuple

Float64 check of the infeasibility certificate of `block_lyapunov_conditioning` (minimise
`kappa` subject to `I <= W <= kappa I`, `-(A'W + WA) >= 2 mu W`, `W` symmetric on `mask`). With
duals `Y1` (of `W >= I`), `Y2` (of `W <= kappa I`) and `Y3` (of the rate), all PSD, and
`M = Y1 - Y2 - (A Y3 + Y3 A') - 2 mu Y3`, a certificate has `M = 0` on the mask, `tr Y2 = 0` and
`tr Y1 > 0`: every feasible `W` would give `0 <= <W, M> + kappa tr Y2 - tr Y1 = -tr Y1 < 0`.
Returns the residuals relative to `tr Y1` and the smallest dual eigenvalues. A solver
certificate, not an interval proof.

---

### block_lyapunov_conditioning

*[src/emdq/krasovskii.jl:205](../../src/emdq/krasovskii.jl#L205)*

    block_lyapunov_conditioning(As, mask; mu, optimizer, silent = true)
        -> (feasible, W, kappa, record, dual_lower, dual_upper, dual_rate)

The best-conditioned quadratic Lyapunov matrix on a pattern: minimise `kappa` subject to
`I <= W <= kappa I`, `-(A'W + WA) >= 2 mu W`, `W` symmetric and free on `mask` (zero
elsewhere). The optimum `kappa` is the least achievable `cond(W)`. A converged solve means the
pattern admits decay at `mu` mathematically; comparing `kappa` with a cap is the separate
conditioning decision. A proper `INFEASIBLE` status comes with the duals checked by
`block_lyapunov_farkas_check`. Implemented in the JuMP extension; the result is a candidate.

---

### krasovskii_conditioning

*[src/emdq/krasovskii.jl:218](../../src/emdq/krasovskii.jl#L218)*

    krasovskii_conditioning(As, mask; mu, cond_cap, optimizer, silent = true)
        -> (feasible, W, kappa, gamma, record)

The best-conditioned Krasovskii Hessian on a pattern: minimise `kappa` subject to
`I <= A'WA <= kappa I`, `gamma I <= W <= cond_cap gamma I` (so `cond(W) <= cond_cap`) and
`-(A'W + WA) >= 2 mu W`, `W` symmetric on `mask`. The optimum `kappa` is the least achievable
`cond(A'WA)` among weights obeying the cap. Implemented in the JuMP extension; the result is a
candidate.

---

### PhysicalNetwork

*[src/emdq/network.jl:24](../../src/emdq/network.jl#L24)*

    PhysicalNetwork

The dynamic network of a case: bus order `net`, base frequency `omega_b`; branches
(`from`, `to` as bus rows, complex ratio `a` at `from`, `R`, `L`, case index `branch`); node
capacitances `C` and conductances `G`; shunt inductors (`sh_bus` as bus rows, `Lsh`).

---

### physical_network

*[src/emdq/network.jl:48](../../src/emdq/network.jl#L48)*

    physical_network(case; omega_b = 2pi*60, loads = true, v0 = nothing) -> PhysicalNetwork

The dynamic network of `case`, element by element from the data `ybus(case; loads)` stamps:
each line becomes a series R-L branch with its ideal transformer, its charging `b/2` a node
capacitor at both ends; a shunt `g + jb` becomes a conductance and a capacitor (`b > 0`) or
inductor (`b < 0`); a load `(P, Q)` at voltage `V0` becomes `G = P/V0^2` and a capacitor
(`Q < 0`) or inductor (`Q > 0`) of admittance `|Q|/V0^2`, with `v0` as in `ybus`.

Throws if an element would not be passive (`r < 0`, `x <= 0`, `g < 0`, `P < 0`): such data
have no passive physical realisation in this element class.

---

### pn_incidence

*[src/emdq/network.jl:103](../../src/emdq/network.jl#L103)*

    pn_incidence(p) -> SparseMatrixCSC{ComplexF64}

The complex incidence `B` (buses x branches): `B[from, l] = 1/conj(a_l)`, `B[to, l] = -1`.
Node injections from the branches are `-B i`; the branch driving voltages are `B^H v`.

---

### pn_flows

*[src/emdq/network.jl:115](../../src/emdq/network.jl#L115)*

    pn_flows(p, v, i, ish, iext) -> (; fv, fi, fs)

The right-hand sides of the complex equations: `fv = (C/omega_b) dv/dt`,
`fi = (L/omega_b) di/dt`, `fs = (L_s/omega_b) di_s/dt`. On a node with `C = 0`, `fv = 0` is
the algebraic KCL constraint.

---

### pn_energy

*[src/emdq/network.jl:133](../../src/emdq/network.jl#L133)*

    pn_energy(p, v, i, ish) -> the network's stored energy (pu*s).

---

### pn_balance

*[src/emdq/network.jl:139](../../src/emdq/network.jl#L139)*

    pn_balance(p, v, i, ish, iext) -> NamedTuple

Element-by-element power balances of `docs/records/early/PHYSICAL_NETWORK.md` (Proposition 1), each as
`rate - (port power - loss)`, and the interconnection sum of all internal port powers
(Proposition 2), which is zero for a lossless Kirchhoff interconnection. The element rates
are `Re(conj(state) * flow)`, the time derivative of each element's energy.
`total_error` is `dH/dt - (supply - losses)`.

---

### pn_descriptor

*[src/emdq/network.jl:175](../../src/emdq/network.jl#L175)*

    pn_descriptor(p) -> (; E, J, R, Bp, nx)

The real linear port-Hamiltonian descriptor form `E dx/dt = (J - R) x + Bp u`, `y = Bp' x`,
`H = x' E x / 2`, with `x = (v, i, i_s)` and each complex number stored as `(Re, Im)`, and
`u = i_ext`, `y = v`. `E` is diagonal and positive semidefinite (singular on nodes with
`C = 0`), `J = -J'` exactly (incidence and frame rotation), `R` diagonal and nonnegative
(`G`, branch `R`). Then `dH/dt = -x' R x + y' u` along every solution (Proposition 1).

---

### pn_quasistatic_ybus

*[src/emdq/network.jl:226](../../src/emdq/network.jl#L226)*

    pn_quasistatic_ybus(p) -> Matrix{ComplexF64}

The bus admittance obtained by setting every derivative to zero in the dynamic equations
(Proposition 3): branch rows give `i = (R + jL)^{-1} B^H v`, shunt rows `i_s = v/(j L_s)`,
and the node rows then give `i_ext = Y v` with
`Y = B diag(1/(R + jL)) B^H + diag(G + jC) + sum_s 1/(j L_s) e_s e_s'`.

---

### pn_node_classes

*[src/emdq/network.jl:244](../../src/emdq/network.jl#L244)*

    pn_node_classes(p) -> (; dynamic, algebraic, cutset)

Bus ids by node type: `dynamic` (`C > 0`), `algebraic` (`C = 0`, `G > 0`: the voltage follows
from the currents, index 1), `cutset` (`C = 0`, `G = 0`: only inductive elements meet there,
so an inductive source at the bus, such as a machine stator, makes an inductor cutset).

---

### emdq_pilot_digest

*[src/emdq/nogo.jl:19](../../src/emdq/nogo.jl#L19)*

    emdq_pilot_digest(s::EMDQPilot) -> SHA-256 of every pilot parameter (winding data included).

---

### emdq_big_data

*[src/emdq/nogo.jl:28](../../src/emdq/nogo.jl#L28)*

    emdq_big_data(s, xs, c, d) -> NamedTuple

The scaled closed-loop Jacobian and Hessian recomputed in 256-bit arithmetic at the 256-bit
closed-loop equilibrium (Newton from `[xs; 0]`), with the scaling `d.d` of the Float64 data
`d = emdq_completion_data(s, xs, c)`: `As`, `Hs`, the equilibrium `z`, its residual and its
shift from the Float64 point, and the entrywise data errors `errA`, `errH` of `d`.

---

### emdq_interval_data

*[src/emdq/nogo.jl:57](../../src/emdq/nogo.jl#L57)*

    emdq_interval_data(s, xs, c, d) -> NamedTuple

Model-bound enclosures for the pilot `s` with the PI `c` (the pilot's Float64 parameters, the
controller reference `xs` and the scales `u0`, `y0` are the model's exact data):
- the closed-loop equilibrium: Newton in 256-bit arithmetic for the centre, then the Krawczyk
  test (`krawczyk`) on the closed-loop field, which proves a unique equilibrium in the box `X`
  and encloses it in `Z`;
- the closed-loop Jacobian over `Z` and the Hessian of the physical energy over its plant part
  (ForwardDiff on outward-rounded intervals), with the controller curvature `u0 y0/omega_i`
  enclosed; every exact linearisation at the equilibrium lies in them;
- the same scaled by `d.d` (powers of two, exact): `AsI = D^-1 AI D`, `HsI = D HI D`.

---

### emdq_nogo_certificate

*[src/emdq/nogo.jl:112](../../src/emdq/nogo.jl#L112)*

    emdq_nogo_certificate(eq, s, xs; unit, pattern = :local_mask, kappa_p = 1.0,
                          omega_i = 1.0) -> Dict

Builds and proves a structured-Lyapunov no-go certificate for the pilot `s` (equilibrium
`xs`; `eq` the IEEE-39 equilibrium it came from, for the case fingerprint) with the PI at
`(kappa_p, omega_i)`, which by the gain lemma represents every gain pair. The boxes are
model-bound (`emdq_interval_data`: a Krawczyk enclosure of the closed-loop equilibrium and
interval derivatives over it), so the statement holds for the pilot model itself. The witness
has exact Float64 components `w` outside `R` and `v_R = -A_RR^-1 A_RF w` (enclosed); the rate
`v' Hs (As v)` with `(As v)_R = 0` is enclosed. As a cross-check, the 256-bit recomputation of
the data must lie in the boxes. The record holds the fingerprints (case, pilot, software),
the pattern and `R` (indices and names), the gains and the lemma, the equilibrium enclosure,
the scaling, the boxes (`A_lo`, `A_hi`, `H_lo`, `H_hi`, scaled, as lists of columns), `w`, and
the rate bounds; `check_nogo_certificate` re-proves it from the record alone.

---

### check_nogo_certificate

*[src/emdq/nogo.jl:179](../../src/emdq/nogo.jl#L179)*

    check_nogo_certificate(rec) -> NamedTuple

Re-proves a record of `emdq_nogo_certificate` from its stored data alone (no model code): the
interval boxes `A`, `H`, the pattern's `R` and the witness components `w`. Checks that every
pattern entry lies in `R x R`, encloses `v_R = -A_RR^-1 A_RF w` (`interval_solve`) and the rate
`v' H (A v)` with `(A v)_R = 0`. `proved` when the lower bound is positive.

---

### emdq_radius_masks

*[src/emdq/nogo.jl:200](../../src/emdq/nogo.jl#L200)*

    emdq_radius_masks(s, data) -> Vector{Pair{String,BitMatrix}}

The nested principal blocks of the locality diagnostic: the excitation assembly (machine
block, `q_dc`, `eta`); plus the terminal capacitor (`v2`); plus the first line (`i23`); plus
the next bus and its load (`v3`, `iL`); plus the second line (`i34`), which is every state.
Each mask is a full block (diagonals included).

---

### emdq_verify_lyapunov

*[src/emdq/nogo.jl:225](../../src/emdq/nogo.jl#L225)*

    emdq_verify_lyapunov(data, P; mu, rel = 1e-6, abs_A = 0.0, AsI = nothing) -> NamedTuple

Interval check of a quadratic Lyapunov matrix `P` (scaled coordinates, taken as exact) for
every scaled linearisation in `AsI` (the model-bound enclosure of `emdq_interval_data`) or,
without it, in the box `As +- (rel |x| + abs_A)`: rigorous lower bounds
`P_min` on `lambda_min(P)` and `decay_min` on `lambda_min(-(A'P + PA) - 2 mu P)`
(`verified_min_eig`). `proved` when both are positive.

---

### GridPH

*[src/emdq/pccreduce.jl:10](../../src/emdq/pccreduce.jl#L10)*

    GridPH

The grid side of the closed two-machine pilot in explicit pH form: the pilot `s`, its grid layout
`G`, the rotor-frame index of machine 2's block (`o`, the angle `od`), the constant physical energy
matrix `Q` (zero on the angle), the kept indices `K` (the PCC pair, machine 2's windings, momentum
and DC charge), the other network indices `O` (pairs), and the angle index.

---

### emdq2_grid_ph

*[src/emdq/pccreduce.jl:28](../../src/emdq/pccreduce.jl#L28)*

    emdq2_grid_ph(s) -> GridPH (closed variant B only, machine 2 with two q-axis windings).

---

### grid_ph_to_rotor

*[src/emdq/pccreduce.jl:58](../../src/emdq/pccreduce.jl#L58)*

    grid_ph_to_rotor(P, xg) -> the grid-side state with machine 2's stator flux in its rotor frame.

---

### grid_ph_from_rotor

*[src/emdq/pccreduce.jl:68](../../src/emdq/pccreduce.jl#L68)*

    grid_ph_from_rotor(P, x) -> the pilot's grid-side state (inverse of `grid_ph_to_rotor`).

---

### grid_ph_JRg

*[src/emdq/pccreduce.jl:78](../../src/emdq/pccreduce.jl#L78)*

    grid_ph_JRg(P, x, u, gf) -> (J, R, g)

The structure at the rotor-frame grid state `x`: skew `J(x)`, `R` (with fault conductances
`gf = (g2, g3)` at bus 2 and bus 3) and the inputs `g` (the PCC current `u = (uD, uQ)` into bus 2,
the frame clock, machine 2's shaft torque and DC source).

---

### grid_ph_rhs

*[src/emdq/pccreduce.jl:134](../../src/emdq/pccreduce.jl#L134)*

    grid_ph_rhs(P, x, u, gf) -> the full grid-side vector field `(J - R) Q x + g`.

---

### grid_ph_energy

*[src/emdq/pccreduce.jl:140](../../src/emdq/pccreduce.jl#L140)*

    grid_ph_energy(P, x) -> the grid side's physical energy `1/2 x'Q x`.

---

### grid_ph_power

*[src/emdq/pccreduce.jl:143](../../src/emdq/pccreduce.jl#L143)*

    grid_ph_power(P, x, u, gf) -> NamedTuple

The physical power flows at `x` (all in pu): the PCC supply, machine 2's shaft and DC-source
supplies, and the losses by group (`lines`, `loads`, `fault`, `stator`, `field`, `dampers`,
`source`, the DC series resistor `Rs i_s^2`).

---

### ReducedGrid

*[src/emdq/pccreduce.jl:172](../../src/emdq/pccreduce.jl#L172)*

    ReducedGrid

The projection `x = Phi x_r` of a `GridPH`: `Phi` (n x r_total; identity on the kept indices and
the angle, `V_O` on the other network indices) and the test basis `Wf` (the energy's own,
`W = Q_y V (V'Q_y V)^{-1}`, and 1 on the angle), so `Wf'Phi = I`. `VO` holds the reduced network
basis (columns over `P.O`).

---

### pcc_L

*[src/emdq/pccreduce.jl:187](../../src/emdq/pccreduce.jl#L187)*

    pcc_L(P, v) -> L v: multiplication by j on every phasor pair of the other network indices (v over `P.O`).

---

### reduced_grid

*[src/emdq/pccreduce.jl:196](../../src/emdq/pccreduce.jl#L196)*

    reduced_grid(P, VO) -> ReducedGrid

Builds `Phi` and `Wf` from the network basis `VO` (columns over `P.O`, full column rank).

---

### reduced_rhs

*[src/emdq/pccreduce.jl:221](../../src/emdq/pccreduce.jl#L221)*

    reduced_rhs(Rg, xr, u, gf) -> the reduced vector field `Wf' f(Phi x_r)` (the exact Petrov-Galerkin projection).

---

### reduced_project

*[src/emdq/pccreduce.jl:224](../../src/emdq/pccreduce.jl#L224)*

    reduced_project(Rg, x) -> x_r = Wf' x (the projection orthogonal in the physical energy).

---

### pcc_qorth

*[src/emdq/pccreduce.jl:227](../../src/emdq/pccreduce.jl#L227)*

    pcc_qorth(P, VO, v) -> the part of `v` (over `P.O`) orthogonal in the physical energy to the
    columns of `VO`, normalised in that energy (or `nothing` if negligible).

---

### pcc_add_pair

*[src/emdq/pccreduce.jl:245](../../src/emdq/pccreduce.jl#L245)*

    pcc_add_pair(P, VO, v) -> VO with the L-completed pair of `v` appended (each part orthogonal in
    the physical energy to what is there), or `VO` unchanged if `v` adds nothing.

---

### pcc_krylov_pool

*[src/emdq/pccreduce.jl:257](../../src/emdq/pccreduce.jl#L257)*

    pcc_krylov_pool(P, xs; kmax) -> Vector of candidate directions over `P.O`

The network's own port directions: `A_OO^k B_O` for `k = 0..kmax`, with `A = (J - R) Q` at the
rotor-frame operating point `xs`, `A_OO` its block on the other network indices and `B_O` its
columns from the kept indices (the PCC node and machine 2's stator) into them.

---

### Jet

*[src/emdq/pccview.jl:15](../../src/emdq/pccview.jl#L15)*

    Jet{T}

A truncated power series `sum_k c[k+1] t^k` (degree `length(c) - 1`), with exact arithmetic
on the kept coefficients. Coefficients may themselves be `Jet`s (nested variables) or duals.

---

### jet_flow

*[src/emdq/pccview.jl:82](../../src/emdq/pccview.jl#L82)*

    jet_flow(f!, x0, N) -> Vector{Jet}

The degree-`N` Taylor expansion of the solution of `dx/dt = f(x)`, `x(0) = x0`, by the Picard
recursion `x_{k+1} = f(x)_k / (k + 1)`; exact to rounding. `f!(dx, x)` must be generic.

---

### pcc_words

*[src/emdq/pccview.jl:102](../../src/emdq/pccview.jl#L102)*

    pcc_words(s; K, A = max(K - 8, 0)) -> Vector{NamedTuple}

The words whose coefficients `pcc_coefficient_values` returns, in order: the drift words
`(a = 0, j = 0, b = r)` for `r = 0..K`, then the single-input words `(a, j, b)` with
`a + b + 1 <= K`, `a <= A`; each for the output components `out = 1, 2` (Re, Im of `v2`), with
the word length `len`.

---

### pcc_coefficient_values

*[src/emdq/pccview.jl:121](../../src/emdq/pccview.jl#L121)*

    pcc_coefficient_values(s, xg; K, tau, A = max(K - 8, 0)) -> Vector

The time-scaled coefficients `tau^len c_w(xg)` of the grid side seen from the PCC, for the words
of `pcc_words` (output `v2`, inputs the two components of `u_g`, drift at `u_g = 0`). Generic
in the element type of `xg`.

---

### pcc_coefficient_jacobian

*[src/emdq/pccview.jl:153](../../src/emdq/pccview.jl#L153)*

    pcc_coefficient_jacobian(s, xg; K, tau, A) -> (values, J, words)

The coefficients and their state gradients (ForwardDiff over the Taylor arithmetic).

---

### pcc_two_input_jacobian

*[src/emdq/pccview.jl:164](../../src/emdq/pccview.jl#L164)*

    pcc_two_input_jacobian(s, xg, j; K, tau, A) -> J2

State gradients of the scaled two-input coefficients `tau^(len+1) c_{(j, w)}` for every word
`w` of `pcc_words` (the input letter `j` placed earliest): `c_{(j,w)} = L_{g_j} c_w`, a
derivative of `c_w` along the constant input field. Row `i` belongs to the word `(j, words[i])`.

---

### pcc_structural_zero

*[src/emdq/pccview.jl:220](../../src/emdq/pccview.jl#L220)*

    pcc_structural_zero(w) -> Bool

The words whose coefficients are constant by the structure of the pilot
(`docs/active/PCC_VIEW_N2_FORMULATION.md` section 3.2, confirmed by Codex): the single-input
words `(0^a, j, 0^b)` with `b <= 6` (constant for `a = 0`, zero for `a >= 1`).

---

### pcc_profile

*[src/emdq/pccview.jl:229](../../src/emdq/pccview.jl#L229)*

    pcc_profile(J, words, gradH, scale; K, drop = pcc_structural_zero, rank_tol = 1e-13,
                extra) -> (profile, dropped_max)

For each depth `k = 0..K`: the numerical rank `r` of the codistribution spanned by the
gradients of the coefficients of length `<= k` (columns scaled by `scale`, rows normalised,
singular values `sv` with `gap = sv[r]/sv[r+1]`), the energy test `e_res = |dH - P dH| / |dH|`
(the part of `dH_g` outside that span), the hidden directions `null` (orthonormal kernel
basis, scaled coordinates) and the closure increment `r_{k+1} - r_k`. `extra` holds further
gradients (one per column) tested the same way: `e_extra[i]` is the part of the `i`-th outside
the span (zero for a zero gradient). Only the rows with
`drop(word)` (constant by structure) are left out; `dropped_max` is their largest scaled norm
relative to the largest drift row, a check that they vanish numerically. Works in the element
type of `J` (Float64 by LAPACK, other types by one-sided Jacobi), so the whole analysis can run
in high precision. A numerical rank is evidence, not an exact rank: read it with `gap` and with
the same profile at a higher precision.

---

### EMDQGovernor

*[src/emdq/phasefree.jl:9](../../src/emdq/phasefree.jl#L9)*

    EMDQGovernor

The case's IEEEG1 governor (`IEEEG1_PHTRUE`) in its equilibrium limiter mode: no valve
position, servo rate or non-windup limit active, so its six lag states obey the linear
equations of `ieeeg1.jl`. The steam tank (`x_steam`), a one-way account that never feeds
back, is not modelled. The governor's states carry no physical storage. `Pref` is set so that
its equilibrium torque is the pilot's shaft torque (`emdq_governed`).

---

### EMDQPhaseFree

*[src/emdq/phasefree.jl:39](../../src/emdq/phasefree.jl#L39)*

    EMDQPhaseFree(pilot, C8, G8, L8[, gov])

An `EMDQPilot` whose fixed bus (`pilot.V4` is then unused) is replaced by a node with voltage
`v8`, capacitor `C8`, conductance `G8` and, when `L8` is finite, a parallel inductor (current
`iL8`). With a governor `gov` (`EMDQGovernor`) the shaft torque is the governor's output
instead of the fixed `pilot.Tm`. Plant state: the pilot's `x` (18 or 19 entries), then `v8`,
then `iL8` if present, then the six governor states if present.

---

### emdq_pf_nstates

*[src/emdq/phasefree.jl:60](../../src/emdq/phasefree.jl#L60)*

    emdq_pf_nstates(pf) -> the number of plant states (without the PI integrator).

---

### emdq_pf_state_names

*[src/emdq/phasefree.jl:63](../../src/emdq/phasefree.jl#L63)*

    emdq_pf_state_names(pf) -> plant state names (`emdq_state_names`, `v8`, `iL8`, then the governor's `gov_x0` to `gov_x5`).

---

### emdq_gov_torque

*[src/emdq/phasefree.jl:68](../../src/emdq/phasefree.jl#L68)*

    emdq_gov_torque(g, xg) -> the governor's mechanical torque `K1 x2 + K3 x3 + K5 x4 + K7 x5`.

---

### emdq_gov_rhs

*[src/emdq/phasefree.jl:71](../../src/emdq/phasefree.jl#L71)*

    emdq_gov_rhs(g, xg, omega) -> the governor's six lag derivatives in its linear mode

`err = Pref - 1 - R_base (omega - 1)`, lead-lag `K (1 + s T2)/(1 + s T1)`, servo `1/(1 + s T3)`,
turbine stages `T4` to `T7` (`ieeeg1.jl` with every limit inactive).

---

### emdq_gov_mode_margins

*[src/emdq/phasefree.jl:84](../../src/emdq/phasefree.jl#L84)*

    emdq_gov_mode_margins(g, xg, omega) -> (; position, rate)

How far the state is inside the governor's equilibrium limiter mode: the valve position margin
`min(x1 - PMIN, PMAX - x1)` and the servo rate margin `min(rate - UC, UO - rate)`. Both must
stay positive on any certified region (a domain condition).

---

### emdq_governed

*[src/emdq/phasefree.jl:96](../../src/emdq/phasefree.jl#L96)*

    emdq_governed(pf, zs, p) -> (pfg, zsg, record)

The phase-free pilot `pf` (without a governor; equilibrium `zs`) closed by the case's IEEEG1
governor with parameters `p` (an `Ieeeg1Params`), in its equilibrium limiter mode. The
electromagnetic equilibrium and the shaft torque `T* = pilot.Tm` are kept: every lag state is
`T*` (the turbine fractions sum to one, checked) and `Pref = 1 + T*/K`. The equilibrium must
be strictly inside the limiter mode (checked).

---

### emdq_phasefree

*[src/emdq/phasefree.jl:119](../../src/emdq/phasefree.jl#L119)*

    emdq_phasefree(s, xs, b78) -> (pf, zs, record)

The phase-free variant of the pilot `s` at its equilibrium `xs`. `b78` is the total charging
of the line to the fixed bus (the case value; the pilot kept only its near half, in `C3`).
From the equilibrium values at the fixed bus, `V8 = s.V4` and `S8 = V8 conj(i34)`:
`C8 = b78/2` (the line's remote half-charging), the remaining demand
`S_L = S8 + j C8 |V8|^2`, `G8 = Re S_L / |V8|^2`, and either an inductor
`L8 = |V8|^2 / Im S_L` (when `Im S_L > 0`) or `-Im S_L / |V8|^2` added to `C8`. Passivity
(`G8 >= 0`, `C8 > 0`, `L8 > 0`) is checked, not assumed. `zs` is the plant equilibrium
(`xs`, then `v8 = V8`, then `iL8`), a point of the new pilot's relative equilibrium at
`omega_b`.

---

### emdq_pf_rhs!

*[src/emdq/phasefree.jl:157](../../src/emdq/phasefree.jl#L157)*

    emdq_pf_rhs!(dx, pf, x, m = pf.pilot.mstar) -> dx

The phase-free pilot's field: the pilot's (`emdq_rhs!`) with the fixed voltage in the second
line's equation replaced by `v8`, and the bus-8 node
`dv8 = (wb/C8)(i34 - G8 v8 - iL8 - j C8 v8)`, `diL8 = (wb/L8)(v8 - j L8 iL8)`.

---

### emdq_pf_energy

*[src/emdq/phasefree.jl:193](../../src/emdq/phasefree.jl#L193)*

    emdq_pf_energy(pf, x) -> the physical energy: the pilot's plus `(C8 |v8|^2 + L8 |iL8|^2)/(2 wb)`.

---

### emdq_pf_balance

*[src/emdq/phasefree.jl:202](../../src/emdq/phasefree.jl#L202)*

    emdq_pf_balance(pf, x, m = pf.pilot.mstar) -> NamedTuple

The raw energy balance of the closed pilot: `dE/dt = shaft + dc_source - losses` (every loss
of `emdq_power`, plus `G8 |v8|^2`; there is no boundary bus). With a governor the shaft power
is the governor's torque times the speed; the governor's states have no physical energy.
`identity_error` is relative to the sum of the terms' magnitudes.

---

### emdq_pf_phasor_pairs

*[src/emdq/phasefree.jl:226](../../src/emdq/phasefree.jl#L226)*

    emdq_pf_phasor_pairs(pf) -> the index pairs of every phasor (stator flux, network voltages and currents).

---

### emdq_pf_rotate

*[src/emdq/phasefree.jl:234](../../src/emdq/phasefree.jl#L234)*

    emdq_pf_rotate(pf, z, phi) -> g_phi z

The symmetry: every phasor pair rotated by `phi`, the rotor angle (index 6) shifted by `phi`,
every other state (rotor windings, momentum, `q_dc`, the PI integrator) unchanged. `z` may
carry the PI integrator after the plant states.

---

### EMDQPFClosedLoop

*[src/emdq/phasefree.jl:251](../../src/emdq/phasefree.jl#L251)*

    EMDQPFClosedLoop(pf, xs, c)

The phase-free pilot with the passive PI `c` on `y_m` (references from the original pilot's
equilibrium `xs`): `z = [plant; eta]`.

---

### emdq_pf_reduce

*[src/emdq/phasefree.jl:276](../../src/emdq/phasefree.jl#L276)*

    emdq_pf_reduce(cl, z) -> xi

The quotient coordinates (the rotor's frame): every phasor rotated by `-(delta - delta_ref)`
with `delta_ref = xs[6]`, the rotor angle removed. Invariant under `emdq_pf_rotate`.

---

### emdq_pf_lift

*[src/emdq/phasefree.jl:287](../../src/emdq/phasefree.jl#L287)*

    emdq_pf_lift(cl, xi) -> the representative `z` of `xi` with `delta = delta_ref`.

---

### emdq_pf_reduced_rhs

*[src/emdq/phasefree.jl:290](../../src/emdq/phasefree.jl#L290)*

    emdq_pf_reduced_rhs(cl, xi) -> dxi

The autonomous quotient field: at the representative `z = lift(xi)` (rotation factor one), each
phasor `X` obeys `dY/dt = dX/dt - j (d delta/dt) X`, every scalar keeps its equation, and the
rotor angle's equation is dropped (it gives the phase velocity).

---

### emdq_pf_quotient_data

*[src/emdq/phasefree.jl:308](../../src/emdq/phasefree.jl#L308)*

    emdq_pf_quotient_data(cl, xi) -> NamedTuple

The data of the solver-free structure gate (G2) at a quotient equilibrium `xi`: the
closed-loop quotient Jacobian `A`, the fixed Hessian `Hfix = blockdiag(Hess E (plant part),
u0 y0/omega_i)` (the energy is invariant, so it is a function of `xi`), the declared
power-of-two scaling `d`, `As = D^-1 A D`, `Hs = D Hfix D`, `eta` (last index) and the
quotient state names.

---

### emdq_pf_masks

*[src/emdq/phasefree.jl:332](../../src/emdq/phasefree.jl#L332)*

    emdq_pf_masks(cl) -> Vector{Pair{String,BitMatrix}}

The predeclared patterns carried to the quotient (section 23.3), as masks on the quotient
state: `local` (the machine block without the rotor angle, and `eta`), `port` (local plus
cross-terms to the terminal voltage), and the nested full blocks `radius 1` (excitation
assembly) to `radius 5` (+ second line) and `radius 6` (+ bus 8 and its load: every state).
Radii up to 5 and the two local patterns are proper; radius 6 is the full pilot. With a
governor every pattern is united with the one predeclared joint mechanical block (rotor
momentum `p` and the six governor states, full), and no governor-electrical entry is admitted.

---

### emdq_pf_interval_data

*[src/emdq/phasefree.jl:380](../../src/emdq/phasefree.jl#L380)*

    emdq_pf_interval_data(cl, xi, d) -> NamedTuple

Model-bound enclosures on the quotient, as `emdq_interval_data` for the pilot: a 256-bit
Newton centre, the Krawczyk test on the quotient field (unique quotient equilibrium in the
box, so the orbit is isolated modulo the symmetry), the quotient Jacobian over the enclosure
and the energy Hessian over its plant part, scaled by `d.d`.

---

### WindingMachine

*[src/emdq/pilot.jl:22](../../src/emdq/pilot.jl#L22)*

    WindingMachine

Full winding data of a synchronous machine, per unit on the system base in the reciprocal
`L_ad` system: stator leakage `Ll` (including any merged external leakage), magnetising
`Lad`, `Laq`, rotor leakages `Lfd`, `L1d` (d axis: field and one damper) and `L1q`, `L2q`
(q axis: `nq = 1` damper for a salient-pole machine, `nq = 2` windings for a round rotor),
resistances, inertia `H` and `omega_b`. `Kd`, `Kq` are the inverse inductance matrices of the
d axis (stator, field, damper) and the q axis (stator, then the `nq` rotor windings), motor
convention. With `nq = 1`, `L2q = R2q = 0` are unused.

---

### winding_machine

*[src/emdq/pilot.jl:83](../../src/emdq/pilot.jl#L83)*

    winding_machine(p::GensalParams; x_ext = 0.0, r_ext = 0.0) -> WindingMachine

The explicit map from GENSAL's standard parameters to winding inductances and resistances,
with the exact (Sauer-Pai) definitions: `Lad = xd - xl`, `xd' = xl + Lad Lfd/(Lad + Lfd)`,
`xd'' = xl + 1/(1/Lad + 1/Lfd + 1/L1d)`, `Td0' = (Lad + Lfd)/(omega_b Rfd)`,
`Td0'' = (L1d + Lad Lfd/(Lad + Lfd))/(omega_b R1d)`, `xq'' = xl + Laq L1q/(Laq + L1q)`,
`Tq0'' = (Laq + L1q)/(omega_b R1q)`. With them the Sauer-Pai subtransient model is the exact
reduction of the winding model when the stator transients are neglected
(`emdq_reduction_check`). An external series leakage `x_ext` (a transformer) and resistance
`r_ext` are added to the stator. Throws if a winding parameter is not positive or the machine
data include saturation (not modelled here).

    winding_machine(p::GenrouParams; x_ext = 0.0, r_ext = 0.0) -> WindingMachine

GENROU's round rotor: the d axis as for GENSAL, and on the q axis the same exact
definitions with a transient winding `1q` and a subtransient winding `2q`:
`Laq = xq - xl`, `xq' = xl + Laq L1q/(Laq + L1q)`, `xq'' = xl + 1/(1/Laq + 1/L1q + 1/L2q)`,
`Tq0' = (Laq + L1q)/(omega_b R1q)`, `Tq0'' = (L2q + Laq L1q/(Laq + L1q))/(omega_b R2q)`.
GENROU_PHTRUE has no saturation.

---

### EMDQPilot

*[src/emdq/pilot.jl:131](../../src/emdq/pilot.jl#L131)*

    EMDQPilot

Gate-1 electrical pilot: machine (with the merged transformer) -> ideal ratio `tap` -> bus 2
node (`C2`) -> line (`R23`, `L23`) -> bus 3 node (`C3`, load `G3`, `L3`) -> line (`R34`, `L34`)
-> fixed-voltage boundary `V4`. Shaft torque `Tm` fixed (temporary boundary); DC source
`vs` behind `Rs` feeding the link `Cdc`; nominal modulation `mstar`. `Dm` is a shaft damping
torque `-Dm (omega - 1)` for diagnostics only (0 for the case: the record has `D = 0`; it
stands in for the mechanical damping a prime mover would supply). Units in `pilot.jl`.

---

### EMDQ_STATES

*[src/emdq/pilot.jl:161](../../src/emdq/pilot.jl#L161)*

State names of the pilot with a one-damper q axis, in order (`emdq_state_names`).

---

### emdq_state_names

*[src/emdq/pilot.jl:166](../../src/emdq/pilot.jl#L166)*

    emdq_state_names(m::WindingMachine) -> Vector{String}

The pilot's state names: `EMDQ_STATES`, and for a round rotor (`nq = 2`) the second q-axis
winding `psi_2q` appended as state 19 (so every other index is the same for both machines).

---

### emdq_machine_block

*[src/emdq/pilot.jl:174](../../src/emdq/pilot.jl#L174)*

    emdq_machine_block(m::WindingMachine) -> the machine's flux, angle and momentum indices.

---

### emdq_machine

*[src/emdq/pilot.jl:179](../../src/emdq/pilot.jl#L179)*

    emdq_machine(m::WindingMachine, x) -> NamedTuple

Rotor-frame fluxes and currents of the pilot state `x`: `psid, psiq, id, iq, ifd, i1d,
i1q, i2q` (motor convention; `i2q = 0` for `nq = 1`), the network-frame stator current `in`,
and the motor-convention air-gap torque `Tem = psid iq - psiq id` (the generator torque is
`-Tem`).

---

### emdq_rhs!

*[src/emdq/pilot.jl:210](../../src/emdq/pilot.jl#L210)*

    emdq_rhs!(dx, s::EMDQPilot, x, m = s.mstar) -> dx

The pilot's vector field at modulation `m`. Generic in the element type of `x` and `m`.

---

### emdq_energy_parts

*[src/emdq/pilot.jl:249](../../src/emdq/pilot.jl#L249)*

    emdq_energy_parts(s::EMDQPilot, x) -> NamedTuple

The physical energy by element (pu*s): `magnetic` (all windings, the merged transformer
leakage included), `kinetic` `p^2/(4H)`, `dc_link`, `network` (line inductors, node
capacitors, load inductor).

---

### emdq_energy

*[src/emdq/pilot.jl:270](../../src/emdq/pilot.jl#L270)*

    emdq_energy(s, x) -> the pilot's total physical energy `H(x)`.

---

### emdq_power

*[src/emdq/pilot.jl:273](../../src/emdq/pilot.jl#L273)*

    emdq_power(s::EMDQPilot, x, m = s.mstar) -> (; supplies, dissipation)

Boundary supplies (`shaft = (Tm + Dm) omega`, `dc_source = vs i_s`,
`fixed_bus = -Re(conj(i34) V4)`, the power the fixed bus delivers) and every loss (stator,
field, dampers, source resistance, lines, load conductance, diagnostic shaft damping
`Dm omega^2`). The raw balance is `dH/dt = sum(supplies) - sum(dissipation)` for every `m`
(the converter is lossless).

---

### emdq_pilot

*[src/emdq/pilot.jl:310](../../src/emdq/pilot.jl#L310)*

    emdq_pilot(eq; machine = "GENROU_11", trf = "Trf_1", line = "Line_3", tie = "Line_5",
               mstar = 0.5, source_drop = 0.05, tau_dc = 0.02) -> (pilot, xstar, report)

Gate-1 pilot from the IEEE-39 equilibrium `eq` (`solve_equilibrium`): a GENSAL or GENROU
record (`machine`; the hydro unit by default) with its step-up transformer `trf` merged into
the stator, the
adjacent line `line` (bus 2 to bus 3), bus 3's demand as a constant-impedance RLC surrogate
fitted to its P and Q at the IEEE-39 voltage (passive surrogate, not an identification of the
case load), and the line `tie` (bus 3 to bus 4) to a fixed-voltage boundary at the IEEE-39
voltage of bus 4. The machine's operating point is the IEEE-39 equilibrium: `|V|` at its
bus and the power its transformer carries (the generator output less any load at the
machine bus, which has no node in the merged model; `report.machine_bus_load`), or the
overrides `P`, `V` for operating-point studies; a pilot power flow fixes the rest. Diagnostics only (both 0 for the case): `Dm`, a shaft damping torque, and `ra_add`, an
added stator resistance (behind the machine bus, so the pilot power flow is unchanged).

DC excitation (declared design values, not identified data): nominal modulation `mstar`,
source drop `source_drop` of the link voltage at the operating point, link time constant
`tau_dc = R_s C_dc`. `xstar` is the exact equilibrium with `m = mstar`; `report` holds the
operating point and the RMS-equivalent field voltage.

---

### EMDQSite

*[src/emdq/pilot2.jl:17](../../src/emdq/pilot2.jl#L17)*

    EMDQSite

A machine site of the two-machine pilot: winding data `mach` (transformer merged), the ideal
ratio `tap` at the network side, the fixed shaft torque `Tm`, the DC source `vs` behind `Rs`
feeding the link `Cdc`, the fixed modulation `mstar` and the diagnostic shaft damping `Dm`.

---

### EMDQPilot2

*[src/emdq/pilot2.jl:35](../../src/emdq/pilot2.jl#L35)*

    EMDQPilot2

Two-machine pilot: machine 1 (`m1`) at bus 2 (the PCC); line 2-3 (`R23`, `L23`), bus 3 (`C3`,
load `G3`, `L3`), and, if `fixed_bus`, line 3-4 (`R34`, `L34`) to the fixed voltage `V4`; line
2-25 (`R225`, `L225`), bus 25 (`C25`, load `G25`, `L25`) and machine 2 (`m2`). `C2` is the
whole bus-2 capacitance.

---

### emdq2_grid_layout

*[src/emdq/pilot2.jl:65](../../src/emdq/pilot2.jl#L65)*

    emdq2_grid_layout(s) -> NamedTuple

First (Re) index of each network pair in the grid-side vector `x_g`, `i34 = 0` without the
fixed bus, the offset `m2` of machine 2's block, and the grid-side size `n`.

---

### emdq2_nc

*[src/emdq/pilot2.jl:81](../../src/emdq/pilot2.jl#L81)*

    emdq2_nc(s) -> the number of component (machine 1) states; `x = [x_c; x_g]`.

---

### emdq2_nstates

*[src/emdq/pilot2.jl:84](../../src/emdq/pilot2.jl#L84)*

    emdq2_nstates(s) -> the total number of states.

---

### emdq2_grid_state_names

*[src/emdq/pilot2.jl:87](../../src/emdq/pilot2.jl#L87)*

    emdq2_grid_state_names(s) -> Vector{String}

Names of the grid-side states, in order.

---

### emdq2_grid_rhs!

*[src/emdq/pilot2.jl:155](../../src/emdq/pilot2.jl#L155)*

    emdq2_grid_rhs!(dx, s, xg, ugD, ugQ) -> dx

The grid side's vector field at the PCC current `u_g = ugD + j ugQ` (into the grid at bus 2).
Generic in the element type.

---

### emdq2_rhs!

*[src/emdq/pilot2.jl:201](../../src/emdq/pilot2.jl#L201)*

    emdq2_rhs!(dx, s, x) -> dx

The whole two-machine pilot: machine 1 feeds the grid side at the PCC with
`u_g = -i_n1 / t1`. Generic in the element type.

---

### emdq2_grid_energy_parts

*[src/emdq/pilot2.jl:230](../../src/emdq/pilot2.jl#L230)*

    emdq2_grid_energy_parts(s, xg) -> (; network, magnetic2, kinetic2, dc_link2)

The grid side's physical energy by element (pu*s), as in `emdq_energy_parts`.

---

### emdq2_grid_energy

*[src/emdq/pilot2.jl:247](../../src/emdq/pilot2.jl#L247)*

    emdq2_grid_energy(s, xg) -> the grid side's physical energy `H_g`.

---

### emdq2_energy

*[src/emdq/pilot2.jl:250](../../src/emdq/pilot2.jl#L250)*

    emdq2_energy(s, x) -> the whole pilot's physical energy `H = H_c + H_g`.

---

### emdq2_grid_power

*[src/emdq/pilot2.jl:269](../../src/emdq/pilot2.jl#L269)*

    emdq2_grid_power(s, xg, ugD, ugQ) -> (; supplies, dissipation)

The grid side's four boundary supplies (the PCC `Re(conj(u_g) v2)`, the fixed bus
`-Re(conj(i34) V4)` (zero without it), machine 2's shaft and DC source) and every loss in it.

---

### emdq2_balance

*[src/emdq/pilot2.jl:289](../../src/emdq/pilot2.jl#L289)*

    emdq2_balance(s, x) -> Dict

The raw energy identities: `whole_error` for `dH/dt = supplies - losses` of the whole pilot,
`grid_error` for the grid side with its PCC supply, `pcc_sum` = the component's port power plus
the grid's (zero: the PCC conserves power).

---

### emdq2_site2_shifted

*[src/emdq/pilot2.jl:315](../../src/emdq/pilot2.jl#L315)*

    emdq2_site2_shifted(s, xg, xgs) -> (; r_rot, port, dissipation, rate)

Machine 2's shifted energy `H_m2,s(x) = H_m2(x) - H_m2(x*) - dH_m2(x*)'(x - x*)` (about the
grid-side operating point `xgs`) and its balance along the grid side's dynamics,
`rate = port - dissipation + r_rot`:
- the shifted port supply `port = Re(conj(Delta i_n2) Delta v25) / t2`;
- the incremental losses `dissipation` (stator, field, dampers, DC source, shaft damping);
- the frame-rotation residual `r_rot = -Re(conj(Delta i_n2) j Delta psi_n2)`, as in
  `emdq_shifted_balance`.

`rate` is computed from the energy gradient (Float64 only); the other terms are closed forms,
generic in the element type.

---

### emdq_pilot2

*[src/emdq/pilot2.jl:386](../../src/emdq/pilot2.jl#L386)*

    emdq_pilot2(eq; machine1 = "GENROU_11", trf1 = "Trf_1", line = "Line_3", tie = "Line_5",
                machine2 = "GENROU_9", trf2 = "Trf_11", branch = "Line_4", fixed_bus = true,
                mstar = 0.5, source_drop = 0.05, tau_dc = 0.02) -> (pilot, xstar, report)

The two-machine pilot from the IEEE-39 equilibrium `eq`: the chain of `emdq_pilot` (machine 1
behind `trf1` at bus a, `line` to bus b, `tie` to the fixed bus) and the branch `branch` from
bus a to a load bus c, with `machine2` behind `trf2` at bus c. Bus b's and c's demands are
constant-impedance RLC surrogates at their IEEE-39 voltages; each line's charging is split
between its end nodes. Each machine has `|V|` at its own bus from the IEEE-39 equilibrium;
with the fixed bus both machines send the power their transformers carry there, and without
it (`fixed_bus = false`, a closed two-machine system) machine 2 keeps its IEEE-39 angle as the
reference and takes the balance. Both excitations have the design values of `emdq_pilot`;
`xstar` is the exact equilibrium.

