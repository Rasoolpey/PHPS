# Interval and certificate tools

Interval arithmetic, enclosures, Taylor models, Lyapunov candidates, containment audits and certificate records. Source: [src/roa/](../../src/roa).

Generated from the docstrings by `scripts/build_api_docs.jl`; do not edit by hand.

| Name | Defined in |
|---|---|
| [`AnalyticModel`](#analyticmodel) | `analytic.jl` |
| [`LyapunovCandidate`](#lyapunovcandidate) | `candidate.jl` |
| [`gradient_matrix_hull`](#gradient_matrix_hull) | `candidate.jl` |
| [`system_digest`](#system_digest) | `candidate.jl` |
| [`model_fingerprint`](#model_fingerprint) | `candidate.jl` |
| [`QuadraticCandidate`](#quadraticcandidate) | `candidate.jl` |
| [`quadratic_candidate`](#quadratic_candidate) | `candidate.jl` |
| [`positivity_proof`](#positivity_proof) | `candidate.jl` |
| [`sublevel_half_widths`](#sublevel_half_widths) | `candidate.jl` |
| [`certify_level`](#certify_level) | `certificate.jl` |
| [`certify_roa`](#certify_roa) | `certificate.jl` |
| [`certificate_claim`](#certificate_claim) | `certificate.jl` |
| [`model_assumptions`](#model_assumptions) | `certificate.jl` |
| [`section_record`](#section_record) | `certificate.jl` |
| [`software_record`](#software_record) | `certificate.jl` |
| [`write_certificate`](#write_certificate) | `certificate.jl` |
| [`interval_cholesky`](#interval_cholesky) | `check.jl` |
| [`cholesky_positive_definite`](#cholesky_positive_definite) | `check.jl` |
| [`roa_check`](#roa_check) | `check.jl` |
| [`ContainmentAudit`](#containmentaudit) | `containment.jl` |
| [`box_digest`](#box_digest) | `containment.jl` |
| [`containment_audit`](#containment_audit) | `containment.jl` |
| [`EquilibriumEnclosure`](#equilibriumenclosure) | `enclosure.jl` |
| [`krawczyk`](#krawczyk) | `enclosure.jl` |
| [`enclose_equilibrium`](#enclose_equilibrium) | `enclosure.jl` |
| [`KCLBranch`](#kclbranch) | `enclosure.jl` |
| [`enclose_kcl_branch`](#enclose_kcl_branch) | `enclosure.jl` |
| [`jacobian_hull`](#jacobian_hull) | `enclosure.jl` |
| [`centered_hull`](#centered_hull) | `enclosure.jl` |
| [`ProofFailure`](#prooffailure) | `interval.jl` |
| [`is_interval_type`](#is_interval_type) | `interval.jl` |
| [`interval_solve`](#interval_solve) | `interval.jl` |
| [`verified_max_eig`](#verified_max_eig) | `interval.jl` |
| [`weyl_max_eig`](#weyl_max_eig) | `interval.jl` |
| [`verified_inverse_diagonal`](#verified_inverse_diagonal) | `interval.jl` |
| [`ellipsoid_half_widths`](#ellipsoid_half_widths) | `interval.jl` |
| [`AbstractSectionModel`](#abstractsectionmodel) | `section.jl` |
| [`SectionModel`](#sectionmodel) | `section.jl` |
| [`rotation_action`](#rotation_action) | `section.jl` |
| [`check_rotation_symmetry`](#check_rotation_symmetry) | `section.jl` |
| [`section_model`](#section_model) | `section.jl` |
| [`lift`](#lift) | `section.jl` |
| [`section_coordinates`](#section_coordinates) | `section.jl` |
| [`section_field`](#section_field) | `section.jl` |
| [`state_field`](#state_field) | `section.jl` |
| [`project_field`](#project_field) | `section.jl` |
| [`section_residual`](#section_residual) | `section.jl` |
| [`section_jacobian`](#section_jacobian) | `section.jl` |
| [`SystemCandidate`](#systemcandidate) | `system_candidate.jl` |
| [`reduced_hessian_hull`](#reduced_hessian_hull) | `system_candidate.jl` |
| [`system_candidate`](#system_candidate) | `system_candidate.jl` |
| [`taylor_level`](#taylor_level) | `taylor.jl` |
| [`certify_roa_taylor`](#certify_roa_taylor) | `taylor.jl` |
| [`taylor_recheck`](#taylor_recheck) | `taylor.jl` |
| [`taylor_record_check`](#taylor_record_check) | `taylor.jl` |

---

### AnalyticModel

*[src/roa/analytic.jl:7](../../src/roa/analytic.jl#L7)*

    AnalyticModel(name, h, g, n, V0; description = "")

A closed-form model for the certificate pipeline: `h(eta, v)` (length `n`) and `g(eta, v)`
(length `length(V0)`, at least 1), both generic in the number type; `V0` is the algebraic
unknown near the equilibrium `eta = 0`. Branches must go through the branch primitives
(e.g. `PHPS.clamp_mode(NoModes(), ...)`).

---

### LyapunovCandidate

*[src/roa/candidate.jl:21](../../src/roa/candidate.jl#L21)*

    LyapunovCandidate

Abstract type of the Lyapunov candidates of the ROA pipeline; see the comment at the top of
`src/roa/candidate.jl` for the methods a candidate provides.

---

### gradient_matrix_hull

*[src/roa/candidate.jl:38](../../src/roa/candidate.jl#L38)*

    gradient_matrix_hull(c, xi_box, E, V) -> Matrix{Interval}

The hull with the certificate's box `E = [eta*] + xi_box` and its KCL voltage box `V` also
given (a candidate that depends on the state, such as `SystemCandidate`, needs them). By
default it is `gradient_matrix_hull(c, xi_box)`.

    gradient_matrix_hull(c::QuadraticCandidate, xi_box) -> Matrix{Interval}

`grad V(eta* + xi) = 2P xi` exactly: the thin matrix `2P`.

    gradient_matrix_hull(c::SystemCandidate, xi_box, E, V) -> Matrix{Interval}

The reduced Hessian hull over the certificate's box `E` (with its KCL voltage box `V`): it
contains `int_0^1 Hess Hr(eta* + t xi) dt` for every `xi` in the box, so
`grad V(eta* + xi) = N xi` for some `N` in it.

---

### system_digest

*[src/roa/candidate.jl:71](../../src/roa/candidate.jl#L71)*

    system_digest(sys::DAESystem) -> String

SHA-256 of the healthy-network model: component types, names and parameters (after
initialisation), wiring, Y-bus, load admittances, fixed-voltage buses and the COI constants.
The fault shunts are left out (the certificate is for the healthy network).

---

### model_fingerprint

*[src/roa/candidate.jl:90](../../src/roa/candidate.jl#L90)*

    model_fingerprint(m::SectionModel) -> String

SHA-256 of the model (`system_digest`) and of the section coordinates: reference point, kept
states, section, scale and names.

---

### QuadraticCandidate

*[src/roa/candidate.jl:103](../../src/roa/candidate.jl#L103)*

    QuadraticCandidate

`V(eta* + xi) = xi' P xi` on the section coordinates of `model`, `P` exactly symmetric.
Built by `quadratic_candidate`.

---

### quadratic_candidate

*[src/roa/candidate.jl:118](../../src/roa/candidate.jl#L118)*

    quadratic_candidate(m::AbstractSectionModel; P = nothing) -> QuadraticCandidate

The local quadratic candidate `V_P`. By default `P` solves `A'P + PA = -I` with `A` the
Jacobian of the section field at the reference point (`section_jacobian`); a given `P` (for
instance from an optimiser) is only symmetrised. Either way `P` is a candidate: its
positivity and every decay claim are proved later.

---

### positivity_proof

*[src/roa/candidate.jl:156](../../src/roa/candidate.jl#L156)*

    positivity_proof(c::QuadraticCandidate) -> NamedTuple

`P > 0` proved by `verified_min_eig` (a rigorous lower bound on its smallest eigenvalue).

    positivity_proof(c::SystemCandidate) -> NamedTuple

`V(xi) >= xi' (Hlo/2) xi > 0` on the outer box, `Hlo` proved positive definite
(`verified_min_eig`).

---

### sublevel_half_widths

*[src/roa/candidate.jl:167](../../src/roa/candidate.jl#L167)*

    sublevel_half_widths(c::QuadraticCandidate, level) -> Vector{Float64}

`|xi_i| <= sqrt(level (P^-1)_ii)` on `{xi' P xi <= level}`, with `(P^-1)_ii` bounded above
by an interval enclosure of `P^-1` and the square root rounded up.

    sublevel_half_widths(c::SystemCandidate, level) -> Vector{Float64}

Half-widths of the ellipsoid `{xi' (Hlo/2) xi <= level}`, which contains the connected
component of `{V <= level}` through the equilibrium when it lies inside the outer box less
the equilibrium enclosure's width (checked; otherwise a `ProofFailure`).

---

### certify_level

*[src/roa/certificate.jl:25](../../src/roa/certificate.jl#L25)*

    certify_level(c::LyapunovCandidate, level, eq::EquilibriumEnclosure; contracts,
                  hull = :centered, second_order = :coordinates) -> Dict

All the gates at one level, on one set of boxes. `hull`: the enclosure of the mean-value
matrix, `:centered` (`centered_hull`, with `second_order` passed on) or `:first_order`
(`jacobian_hull`). Returns the row of the certificate record;
`row["certified"]` is the verdict.

---

### certify_roa

*[src/roa/certificate.jl:80](../../src/roa/certificate.jl#L80)*

    certify_roa(c::LyapunovCandidate; level_low = 1e-16, level_high = 1.0, bisections = 40,
                time_limit = 900, contracts, hull = :centered, second_order = :coordinates,
                resolution = 1e-3, io = stdout) -> Dict

The certificate record of `c`: the positivity proof, the equilibrium enclosure, then
`certify_level` at `level_high`, else at `level_low` and a log-bisection between (at most
`bisections` further levels, until the bracket is narrower than `resolution` in log10, and
no new level after `time_limit` seconds). Every tried level
is kept. `verified_valid_level` is the largest certified level found: a certified level that
depends on the search, not a proved maximum. Progress goes to `io` (flushed).

    certify_roa(scenario::AbstractString; kwargs...) -> Dict

The quadratic candidate `V_P` at the equilibrium of a scenario's case (healthy network),
certified by `certify_roa(candidate; kwargs...)`; the record also names the scenario.

---

### certificate_claim

*[src/roa/certificate.jl:165](../../src/roa/certificate.jl#L165)*

    certificate_claim(m) -> String

The theorem a certified level proves, for the model class of `m`.

---

### model_assumptions

*[src/roa/certificate.jl:188](../../src/roa/certificate.jl#L188)*

    model_assumptions(m) -> Vector{String}

What the claim rests on beyond the proof steps, for the model class of `m`.

---

### section_record

*[src/roa/certificate.jl:212](../../src/roa/certificate.jl#L212)*

    section_record(m::SectionModel) -> Dict

The coordinates of a certificate: case and contract fingerprints, the reference point, the
kept states, the section and the scale.

---

### software_record

*[src/roa/certificate.jl:238](../../src/roa/certificate.jl#L238)*

    software_record() -> Dict

Julia, PHPS (version and git commit when the source is a git checkout) and the interval
library versions.

---

### write_certificate

*[src/roa/certificate.jl:263](../../src/roa/certificate.jl#L263)*

    write_certificate(path, record) -> path

Write a certificate record as JSON (non-finite numbers as strings).

---

### interval_cholesky

*[src/roa/check.jl:13](../../src/roa/check.jl#L13)*

    interval_cholesky(A) -> NamedTuple

The interval Cholesky algorithm on the symmetric interval matrix `A` (only the lower
triangle and diagonal are read). `positive_definite = true` proves every symmetric matrix
in `A` positive definite; `min_pivot` is the smallest lower bound of a squared pivot.

---

### cholesky_positive_definite

*[src/roa/check.jl:43](../../src/roa/check.jl#L43)*

    cholesky_positive_definite(A) -> NamedTuple

Proves every symmetric matrix in the interval matrix `A` positive definite by interval
Cholesky: directly, and, when the widening of the elimination defeats that, after the
congruence `B = X A X'` with `X` the inverse of the floating-point Cholesky factor of
mid(A) (`B` is near the identity; `B > 0` for all members implies `A > 0`, since `X` is a
fixed real matrix and `X A X' > 0` forces `X` nonsingular). `method` says which succeeded.

---

### roa_check

*[src/roa/check.jl:66](../../src/roa/check.jl#L66)*

    roa_check(record; contracts = default_contracts()) -> Dict

Rebuild the candidate of a certificate record (from `certify_roa`, with the candidate data
it stores), check the case and contract hashes and the model and candidate fingerprints
against the record, and re-prove its `verified_valid_level`: positivity of `P` and the decay
by interval Cholesky (on `-sym(N'M)` directly, or, when its width defeats it, on the
midpoint shifted by the Collatz bound of the radius), on freshly computed enclosures with
the same hull method. `passed` requires every item.

    roa_check(path::AbstractString; kwargs...) -> Dict

`roa_check` on a certificate record read from its JSON file.

---

### ContainmentAudit

*[src/roa/containment.jl:23](../../src/roa/containment.jl#L23)*

    ContainmentAudit

The result of `containment_audit`: per-category verdicts, the clause records, the branch
decisions of every component on the box, the ranges over which the excluded states are
proved not to feed back (with a bound on their drift rate while the retained state is in the
box), and the SHA-256 of the boxes it ran on.

---

### box_digest

*[src/roa/containment.jl:41](../../src/roa/containment.jl#L41)*

    box_digest(E, V) -> String

SHA-256 of the interval boxes (their bounds), which ties the gates of one certificate to
the same boxes.

---

### containment_audit

*[src/roa/containment.jl:73](../../src/roa/containment.jl#L73)*

    containment_audit(m, E, V; contracts = default_contracts(), spans = (60, 40, 20, 10))
        -> ContainmentAudit

The audit on the section box `E` and the voltage box `V` (see the comment at the top of
`src/roa/containment.jl`). Each reservoir and monitor state gets the widest range
`x0 * [2^-k, 2^k]` (`[-2^k, 2^k]` when `x0 = 0`), `k` in `spans`, over which the retained
rows and KCL are proved independent of it.

---

### EquilibriumEnclosure

*[src/roa/enclosure.jl:6](../../src/roa/enclosure.jl#L6)*

    EquilibriumEnclosure

A box `eta x V` containing exactly one zero of `[h; g]` (the relative equilibrium on the
section and its bus voltages), proved by the Krawczyk test.

---

### krawczyk

*[src/roa/enclosure.jl:18](../../src/roa/enclosure.jl#L18)*

    krawczyk(F, J, what, center, r; maxiter = 12) -> (K, X, record)

The Krawczyk test for `F(w) = 0` around `center`: with `C` an approximate inverse of the
Jacobian at the centre and `X = center + [-r, r]`,
`K(X) = center - C F(center) + (I - C J(X)) (X - center)`. `K` in the interior of `X` proves
that `X` contains exactly one zero, and it lies in `K`. On failure the radius is inflated to
twice the reach of `K` (epsilon inflation), at most `maxiter` times. `F` maps an interval
vector to an interval vector; `J` maps an interval box to an interval Jacobian; `what` names
the problem in errors.

---

### enclose_equilibrium

*[src/roa/enclosure.jl:54](../../src/roa/enclosure.jl#L54)*

    enclose_equilibrium(m::AbstractSectionModel; newton = 4) -> EquilibriumEnclosure

The equilibrium on the section: a few Newton steps from `(0, V0)` in Float64, then the
Krawczyk test on `[h; g]` in the unknowns `[eta; V]`.

---

### KCLBranch

*[src/roa/enclosure.jl:76](../../src/roa/enclosure.jl#L76)*

    KCLBranch

For every `eta` in the box `eta`, KCL `g(eta, V) = 0` has exactly one solution `V(eta)` in
the box `X`, and it lies in `V` (subset of `X`); `g_V` is nonsingular on `eta x X`.

---

### enclose_kcl_branch

*[src/roa/enclosure.jl:89](../../src/roa/enclosure.jl#L89)*

    enclose_kcl_branch(m, E, Vc) -> KCLBranch

The KCL branch over the section box `E`, centred at the voltages `Vc` (Float64): the
parametric Krawczyk test on `V -> g(E, V)`.

---

### jacobian_hull

*[src/roa/enclosure.jl:106](../../src/roa/enclosure.jl#L106)*

    jacobian_hull(m, E, V) -> (M, record)

An interval matrix containing the Jacobian of the network-reduced section field
`eta -> h(eta, v(eta))` at every `eta` in `E` with `v(eta)` in `V` on the KCL branch:
`h_eta + h_V Dv` with `Dv` enclosing `-g_V^-1 g_eta` over `E x V` (`interval_solve`, which
also proves `g_V` nonsingular there). The integral of the Jacobian along any segment in `E`
(the mean-value matrix) lies in the same hull.

---

### centered_hull

*[src/roa/enclosure.jl:140](../../src/roa/enclosure.jl#L140)*

    centered_hull(m, eq, E, V, Xi; second_order = :coordinates, chunk = 1, X = V) -> (M, record)

A tighter enclosure of the mean-value matrix `M(xi) = int_0^1 J(eta* + t xi) dt` for every
equilibrium `eta*` in `eq` and every `xi` in the box `Xi` (with `E` containing `eq.eta + Xi`
and `V` the KCL-branch root box over `E`): the centered form
`M in J(eta*) + [-R, R]`, where `R` bounds half the variation `|d/ds J(p + s xi)|` at every
`p` in `E` along every `xi` in `Xi` (the factor 1/2 is `int_0^1 t dt`). Along a direction
the reduced Jacobian changes by `D = dh_eta + dh_V Dv + h_V dDv`,
`dDv = -g_V^-1 (dg_eta + dg_V Dv)`, with the KCL branch moving as `v' = Dv xi`, and the
derivatives come from nested duals over the box. `second_order = :coordinates` (default)
bounds `R = (1/2) sum_k |xi_k| max |dJ/deta_k|` coordinate by coordinate (171 nested-dual
Jacobians on IEEE-39, about 25 s; `chunk` directions per evaluation, but larger chunks
compile far too slowly); `:direction` uses the single interval direction `[Xi; Dv Xi]` (one
evaluation, about 3 times wider on IEEE-39). The result is intersected entrywise with the first-order hull over `E x V`,
which also encloses `M`.
`J(eta*)` is the first-order hull on the thin equilibrium boxes. The equilibrium voltage box
must lie in the branch's uniqueness box `X` (`KCLBranch.X`): then the equilibrium's voltages
are the branch root `v(eta*)`, which lies in `V` too, and the centre is evaluated on
`eq.V cap V`.

---

### ProofFailure

*[src/roa/interval.jl:12](../../src/roa/interval.jl#L12)*

    ProofFailure

Thrown when a proof step does not go through (a Krawczyk test that does not contract, a
matrix not proved nonsingular, an evaluation that is not well defined on its box). It
means "not proved", never "false"; any other exception is a bug.

---

### is_interval_type

*[src/roa/interval.jl:51](../../src/roa/interval.jl#L51)*

    is_interval_type(T) -> Bool

Whether `T` is an interval, or a (nested) ForwardDiff dual over intervals.

---

### interval_solve

*[src/roa/interval.jl:110](../../src/roa/interval.jl#L110)*

    interval_solve(A, B; iterations = 3) -> (X, beta)

An enclosure `X` of `{A^-1 B : A in [A], B in [B]}` (column by column) for interval
matrices, and the bound `beta >= ||I - C A||_inf` (`C` an approximate inverse of mid(A)).
`beta < 1` proves every `A` in `[A]` nonsingular; otherwise this throws. Method: with
`R = I - C[A]`, every solution satisfies `x = C b + R x`, so `|x| <= |C b| / (1 - beta)`
bounds it, and `X <- (C[B] + R X) cap X` refines the bound (Krawczyk's operator for linear
systems).

---

### verified_max_eig

*[src/roa/interval.jl:153](../../src/roa/interval.jl#L153)*

    verified_max_eig(S) -> Float64

A rigorous upper bound on the largest eigenvalue of every symmetric matrix in the interval
matrix `S` (`Inf` when the check fails); `verified_min_eig` applied to `-S`.

---

### weyl_max_eig

*[src/roa/interval.jl:161](../../src/roa/interval.jl#L161)*

    weyl_max_eig(S) -> NamedTuple

A rigorous upper bound on the largest eigenvalue of every symmetric matrix in the interval
matrix `S`, for wide `S`: by Weyl, `lambda_max(Sc + D) <= lambda_max(Sc) + ||D||_2` with
`Sc` the (Float64, symmetric) midpoint and `|D| <= Rad` entrywise, and
`||D||_2 <= rho(|D|) <= rho(Rad)` (Perron-Frobenius). `lambda_max(Sc)` is bounded by
`verified_max_eig` on the thin matrix, `rho(Rad)` by the Collatz-Wielandt bound
`max_i (Rad v)_i / v_i` for a positive vector `v` near the Perron vector (outward rounded).
Returns the bound and its two parts.

---

### verified_inverse_diagonal

*[src/roa/interval.jl:191](../../src/roa/interval.jl#L191)*

    verified_inverse_diagonal(P) -> Vector{Float64}

Rigorous upper bounds on the diagonal of `P^-1` for a Float64 matrix `P` (which must be
proved positive definite separately): the diagonal of an interval enclosure of `P^-1`.

---

### ellipsoid_half_widths

*[src/roa/interval.jl:203](../../src/roa/interval.jl#L203)*

    ellipsoid_half_widths(level, inv_diag_up) -> Vector{Float64}

Rigorous half-widths of the box enclosing `{xi : xi' P xi <= level}`:
`max |xi_i| = sqrt(level (P^-1)_ii)`, rounded up.

---

### AbstractSectionModel

*[src/roa/section.jl:30](../../src/roa/section.jl#L30)*

    AbstractSectionModel

A system the certificate pipeline works on: section coordinates `eta` (`neta`), algebraic
unknowns `V` (`nvolt`) with a reference value `V0`, and `section_field(m, eta, V) -> (h, g)`
generic in the number type. `SectionModel` (a power system on its common-angle section) and
`AnalyticModel` (closed-form test systems) implement it.

---

### SectionModel

*[src/roa/section.jl:40](../../src/roa/section.jl#L40)*

    SectionModel

The network-reduced dynamics on the common-angle section in scaled coordinates `eta`, with
the bus voltages explicit (see the comment at the top of `src/roa/section.jl`). Built by
`section_model`.

---

### rotation_action

*[src/roa/section.jl:82](../../src/roa/section.jl#L82)*

    rotation_action(c) -> NamedTuple or nothing

How component `c` transforms under the common rotation: `angles`, the states shifted by
theta; `inputs` and `outputs`, the (d, q) pairs in the network frame, turned by theta; every
other state, input and output is invariant. `nothing` when the model type declares no action
(`section_model` then rejects the system).

---

### check_rotation_symmetry

*[src/roa/section.jl:103](../../src/roa/section.jl#L103)*

    check_rotation_symmetry(sys) -> Vector{Int}

Verifies that the assembled system is equivariant under the common rotation, given the
per-model `rotation_action`s: no fixed-voltage bus; every network-frame input pair is wired
to a bus voltage pair or to a network-frame output pair, and every invariant input to an
invariant source (a constant, |V|, a machine's dq-frame voltage, an invariant output);
every KCL injection is a network-frame output pair. The Y-bus, the fault shunts and the
frequency-dependent loads (through the COI speed) are linear or invariant. Returns the
state indices of the rotating angles. Throws `ArgumentError` otherwise.

---

### section_model

*[src/roa/section.jl:172](../../src/roa/section.jl#L172)*

    section_model(sys, x0, V0; projection, scale = :lyapunov) -> SectionModel

The section model at the reference point `(x0, V0)` (normally the equilibrium from
`solve_equilibrium`). `scale`: a vector of powers of two, `:none`, or `:lyapunov` (the
diagonal of the dense Lyapunov solution of the unscaled section Jacobian, rounded to powers
of two by `pow2_scaling`, which makes the candidate `P` well conditioned).

Fails closed unless the rotation symmetry holds (`check_rotation_symmetry`: every model type
declares its `rotation_action`, and the wiring and network respect it) and the rotating
angles are exactly the COI sources' angles, all physical coordinates.

    section_model(eq::EquilibriumResult; kwargs...) -> SectionModel

The section model at the equilibrium of `solve_equilibrium` (its system, state, voltages and
initialisation parameters).

---

### lift

*[src/roa/section.jl:234](../../src/roa/section.jl#L234)*

    lift(m, eta) -> x

The full state at section coordinates `eta` (in `eta`'s number type).

    lift(c::LyapunovCandidate, eta) -> x

The full physical state at section coordinates `eta`.

---

### section_coordinates

*[src/roa/section.jl:256](../../src/roa/section.jl#L256)*

    section_coordinates(m, x) -> eta

The section coordinates of a full state on the section (the inverse of `lift` there; the
states outside `keep` and the reference angle are ignored).

---

### section_field

*[src/roa/section.jl:265](../../src/roa/section.jl#L265)*

    section_field(m, eta, V) -> (h, g)

The projected section field `h` and the KCL residual `g` (see the file comment), in the
promoted number type of `eta` and `V` (Float64, intervals, duals over either). With
`logs = (out = [...], step = [...])`, per-component `ModeLog`s record the branch decisions.

---

### state_field

*[src/roa/section.jl:278](../../src/roa/section.jl#L278)*

    state_field(m, x, V; logs = nothing) -> (f, g)

The unprojected field and KCL residual at a full state `x` (no fault).

---

### project_field

*[src/roa/section.jl:291](../../src/roa/section.jl#L291)*

    project_field(m, f) -> h

The section field from the full field: `[(I - a l'/(l'a)) f_keep]_z ./ s`.

---

### section_residual

*[src/roa/section.jl:310](../../src/roa/section.jl#L310)*

    section_residual(m, w) -> [h; g]

`section_field` on the stacked unknowns `w = [eta; V]`.

---

### section_jacobian

*[src/roa/section.jl:321](../../src/roa/section.jl#L321)*

    section_jacobian(m[, eta, V]) -> Matrix

The Jacobian of the network-reduced section field `eta -> h(eta, V(eta))` at `(eta, V)`
with `g(eta, V) = 0` (default: `eta = 0`, `V = V0`), in Float64:
`h_eta - h_V g_V^-1 g_eta`. For constructing candidates; the proof uses `jacobian_hull`.

---

### SystemCandidate

*[src/roa/system_candidate.jl:24](../../src/roa/system_candidate.jl#L24)*

    SystemCandidate

The nonlinear lift of `H_sys` as a `LyapunovCandidate` on the section model `model`, with the
storage `sc` (`StorageCompletion` on the same section, unscaled), the equilibrium enclosure
`eq`, the outer box half-widths `r`, the lower Hessian bound `Hlo` on the outer box and its
construction record. Built by `system_candidate`.

---

### reduced_hessian_hull

*[src/roa/system_candidate.jl:47](../../src/roa/system_candidate.jl#L47)*

    reduced_hessian_hull(m, sc, E, V) -> (N, record)

An interval matrix containing the Hessian of `eta -> H(lift(eta), v(eta))` (the KCL branch
`v`) at every `eta` in `E`, with `v(eta)` in `V`. It is
`[I; Dv]' Hess(H - lambda' g) [I; Dv]`, with `Dv` enclosing `-g_V^-1 g_eta` and `lambda`
enclosing the solutions of `g_V' lambda = H_V` over `E x V`.

---

### system_candidate

*[src/roa/system_candidate.jl:91](../../src/roa/system_candidate.jl#L91)*

    system_candidate(m, sc; outer_level) -> SystemCandidate

The candidate on the section model `m` (scaled; `sc` is on the same section, unscaled). The
outer box `r` is the ellipsoid box of `xi' (H0/2) xi <= 1.1 outer_level`, with `H0` the Float64
reduced Hessian at the reference point. On `B = [eta*] + [-r, r]` (the equilibrium enclosed by
`enclose_equilibrium`, the KCL branch by `enclose_kcl_branch`) the reduced Hessian hull gives
the lower bound `Hlo`. Levels up to about `outer_level` can then be certified. `Hlo` and the
outer box are proof inputs; `H0` only shapes the box.

---

### taylor_level

*[src/roa/taylor.jl:66](../../src/roa/taylor.jl#L66)*

    taylor_level(c::QuadraticCandidate, level, eq; contracts) -> Dict

The gates of `certify_level` at one level, with the decay proved by the second-order Taylor
bound instead of the mean-value matrix: the sublevel box, the KCL branch, the quadratic
margin `m` (on the thin equilibrium boxes), the cubic bound `B` (the slices `sigma_k` over the
box, the two forms of the ellipsoid bound), `decay = B < m`, and the containment audit on the
same boxes. Returns the row; `row["certified"]` is the verdict.

---

### certify_roa_taylor

*[src/roa/taylor.jl:124](../../src/roa/taylor.jl#L124)*

    certify_roa_taylor(c::QuadraticCandidate; level_low, level_high, bisections = 20,
                       time_limit = 3600, resolution = 0.01, io = stdout) -> Dict

The certificate record with the Taylor decay bound: positivity, the equilibrium enclosure,
then `taylor_level` at `level_high`, else at `level_low` and a log-bisection between them, as in
`certify_roa`. `verified_valid_level` is the largest certified level found (a certified level,
not a proved maximum).

---

### taylor_recheck

*[src/roa/taylor.jl:176](../../src/roa/taylor.jl#L176)*

    taylor_recheck(c::QuadraticCandidate, row) -> NamedTuple

Recomputes a certified row of `certify_roa_taylor` (or `taylor_level`) from the candidate and
the model: the equilibrium enclosure, the boxes, the KCL branch, the quadratic margin and every
slice bound `sigma_k`. It checks that the recomputed values reproduce the recorded ones (the
same boxes by digest) and that the inequality `B < m` holds with them.

---

### taylor_record_check

*[src/roa/taylor.jl:194](../../src/roa/taylor.jl#L194)*

    taylor_record_check(P, row) -> NamedTuple

The arithmetic part of a Taylor row rechecked from the record alone: from `P`, the level and
the recorded slice bounds `sigma`, the half-widths (`verified_inverse_diagonal`), `|P^-1|`
(an interval inverse) and both forms of the cubic bound are recomputed with outward rounding,
and `B < m` is checked against the recorded quadratic margin. The margin and the slices
themselves are rechecked from the model by `taylor_recheck`.

