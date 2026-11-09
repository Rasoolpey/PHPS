# The nonlinear system-level candidate H_sys of the RMS model in the ROA pipeline
# (docs/records/early/STABILITY_INVESTIGATION.md section 25). H_sys is the explicit lift of the storage
# completion (`completion_energy_state`): a system-level mathematical Lyapunov candidate (its
# physical core is audited; its controller and repair parts are certificate terms), not
# physical energy.
#
# The candidate on the section coordinates is the Bregman shift of the reduced lift at the TRUE
# equilibrium eta* (enclosed, never known exactly):
#   Hr(eta) = H(lift(eta), v(eta)) on the KCL branch,
#   V(eta* + xi) = Hr(eta* + xi) - Hr(eta*) - grad Hr(eta*)' xi,
# so grad V(eta* + xi) = (int_0^1 Hess Hr(eta* + t xi) dt) xi: the gradient-matrix hull is an
# enclosure of the reduced Hessian over the box. With Dv = -g_V^-1 g_eta and the adjoint
# lambda solving g_V' lambda = H_V,
#   Hess Hr = [I; Dv]' Hess(H - lambda' g) [I; Dv]     (exact along the branch),
# enclosed with Dv, lambda and the Hessian of the Lagrangian evaluated over the box
# (`interval_solve`, ForwardDiff on intervals).
#
# Positivity and the sublevel set use one fixed outer box B = [eta*] + [-r, r]: a matrix lower
# bound H_lo (midpoint minus a rigorous Weyl radius) of every reduced Hessian on B gives
# V(xi) >= xi' (H_lo/2) xi on B. When the ellipsoid {xi' (H_lo/2) xi <= c} lies inside B, the
# connected component of {V <= c} that contains the equilibrium lies in that ellipsoid. The
# certified set is that component.

"""
    SystemCandidate

The nonlinear lift of `H_sys` as a `LyapunovCandidate` on the section model `model`, with the
storage `sc` (`StorageCompletion` on the same section, unscaled), the equilibrium enclosure
`eq`, the outer box half-widths `r`, the lower Hessian bound `Hlo` on the outer box and its
construction record. Built by `system_candidate`.
"""
struct SystemCandidate <: LyapunovCandidate
    model::SectionModel
    sc::StorageCompletion
    eq::EquilibriumEnclosure
    r::Vector{Float64}
    Hlo::Matrix{Float64}
    construction::Dict{String,Any}
end

Base.show(io::IO, c::SystemCandidate) =
    print(io, "SystemCandidate(", neta(c.model), " coordinates, nonlinear lift of H_sys)")

# the lift on the section coordinates of `m` and explicit voltages
_hsys(c_sc::StorageCompletion, m::SectionModel, eta, V) = completion_energy_state(c_sc, lift(m, eta), V).total

"""
    reduced_hessian_hull(m, sc, E, V) -> (N, record)

An interval matrix containing the Hessian of `eta -> H(lift(eta), v(eta))` (the KCL branch
`v`) at every `eta` in `E`, with `v(eta)` in `V`. It is
`[I; Dv]' Hess(H - lambda' g) [I; Dv]`, with `Dv` enclosing `-g_V^-1 g_eta` and `lambda`
enclosing the solutions of `g_V' lambda = H_V` over `E x V`.
"""
function reduced_hessian_hull(m::SectionModel, sc::StorageCompletion, E::AbstractVector{<:IA.Interval},
                              V::AbstractVector{<:IA.Interval})
    n = neta(m)
    nv = length(V)
    _, rec, Dv = _reduced_jacobian_hull(m, E, V)
    w = vcat(collect(E), collect(V))
    J = ForwardDiff.jacobian(u -> section_residual(m, u), w)
    gV = J[n+1:end, n+1:end]
    HV = ForwardDiff.gradient(u -> _hsys(sc, m, view(u, 1:n), view(u, n+1:n+nv)), w)[n+1:end]
    lam, beta = interval_solve(permutedims(gV), HV)
    L(u) = _hsys(sc, m, view(u, 1:n), view(u, n+1:n+nv)) -
           sum(lam[j] * section_field(m, view(u, 1:n), view(u, n+1:n+nv))[2][j] for j in 1:nv)
    HL = ForwardDiff.hessian(L, w)
    well_defined(HL) || fail("reduced_hessian_hull: the Lagrangian Hessian is not well defined on the box")
    Bm = vcat(ival.(Matrix{Float64}(I, n, n)), Dv)
    N = permutedims(Bm) * HL * Bm
    N = [(N[i, j] + N[j, i]) / 2 for i in 1:n, j in 1:n]
    rec["adjoint_preconditioned_norm"] = beta
    rec["hessian_hull_width_max"] = maximum(IA.diam, N)
    return N, rec
end

# a Float64 matrix below every symmetric matrix of the interval matrix S: the midpoint minus a
# rigorous upper bound of the radius's spectral norm (Weyl), rounded down on the diagonal
function _matrix_lower_bound(S::AbstractMatrix{<:IA.Interval})
    wb = weyl_max_eig(S)                        # bound = lambda_max(mid) + rho, rho rigorous
    Sc = IA.mid.(S)
    Sc = (Sc + Sc') / 2
    rho = wb.radius
    L = copy(Sc)
    for i in axes(L, 1)
        L[i, i] = IA.inf(ival(Sc[i, i]) - ival(rho))
    end
    return L, rho
end

"""
    system_candidate(m, sc; outer_level) -> SystemCandidate

The candidate on the section model `m` (scaled; `sc` is on the same section, unscaled). The
outer box `r` is the ellipsoid box of `xi' (H0/2) xi <= 1.1 outer_level`, with `H0` the Float64
reduced Hessian at the reference point. On `B = [eta*] + [-r, r]` (the equilibrium enclosed by
`enclose_equilibrium`, the KCL branch by `enclose_kcl_branch`) the reduced Hessian hull gives
the lower bound `Hlo`. Levels up to about `outer_level` can then be certified. `Hlo` and the
outer box are proof inputs; `H0` only shapes the box.
"""
function system_candidate(m::SectionModel, sc::StorageCompletion; outer_level::Real)
    n = neta(m)
    eq = enclose_equilibrium(m)
    f(e) = _hsys(sc, m, e, kcl_solve(m.sys, lift(m, e), m.V0))
    H0 = ForwardDiff.hessian(f, IA.mid.(eq.eta))
    H0 = (H0 + H0') / 2
    isposdef(Symmetric(H0)) || throw(ArgumentError("system_candidate: the reduced Hessian is not positive definite"))
    r = sqrt.(1.1 * outer_level .* diag(inv(H0 / 2)))
    B = eq.eta .+ [ival(-v, v) for v in r]
    br = enclose_kcl_branch(m, B, IA.mid.(eq.V))
    N, rec = reduced_hessian_hull(m, sc, B, br.V)
    Hlo, rho = _matrix_lower_bound(N)
    construction = Dict{String,Any}("method" => "nonlinear lift of H_sys, Bregman-shifted at the enclosed equilibrium",
                                    "dimension" => n, "outer_level" => Float64(outer_level),
                                    "outer_half_width_max" => maximum(r), "outer_half_width_min" => minimum(r),
                                    "outer_hull" => rec, "weyl_radius" => rho,
                                    "H0_eig" => [eigmin(Symmetric(H0)), eigmax(Symmetric(H0))],
                                    "Hlo_eig_float64" => [eigmin(Symmetric(Hlo)), eigmax(Symmetric(Hlo))],
                                    "note" => "H0 shapes the outer box only; Hlo is a proof input")
    return SystemCandidate(m, sc, eq, r, Hlo, construction)
end

candidate_model(c::SystemCandidate) = c.model

# the value and gradient at the Float64 centre of the equilibrium enclosure (records only; the
# proofs use the hulls)
function candidate_value(c::SystemCandidate, xi::AbstractVector)
    m = c.model
    e0 = IA.mid.(c.eq.eta)
    f(e) = _hsys(c.sc, m, e, kcl_solve(m.sys, lift(m, e), m.V0))
    g0 = ForwardDiff.gradient(f, e0)
    return f(e0 .+ xi) - f(e0) - dot(g0, xi)
end
function candidate_gradient(c::SystemCandidate, xi::AbstractVector)
    m = c.model
    e0 = IA.mid.(c.eq.eta)
    f(e) = _hsys(c.sc, m, e, kcl_solve(m.sys, lift(m, e), m.V0))
    return ForwardDiff.gradient(f, e0 .+ xi) - ForwardDiff.gradient(f, e0)
end

"""
    positivity_proof(c::SystemCandidate) -> NamedTuple

`V(xi) >= xi' (Hlo/2) xi > 0` on the outer box, `Hlo` proved positive definite
(`verified_min_eig`).
"""
function positivity_proof(c::SystemCandidate)
    lo = verified_min_eig(ival.(c.Hlo ./ 2))
    return (proved = lo > 0, lower_bound = lo,
            method = "V >= xi'(Hlo/2)xi on the outer box (Hlo below the reduced Hessian hull); lambda_min(Hlo/2) >= $lo")
end

"""
    sublevel_half_widths(c::SystemCandidate, level) -> Vector{Float64}

Half-widths of the ellipsoid `{xi' (Hlo/2) xi <= level}`, which contains the connected
component of `{V <= level}` through the equilibrium when it lies inside the outer box less
the equilibrium enclosure's width (checked; otherwise a `ProofFailure`).
"""
function sublevel_half_widths(c::SystemCandidate, level::Real)
    d = get!(() -> verified_inverse_diagonal(c.Hlo ./ 2), c.construction, "inverse_diagonal_upper")
    w = ellipsoid_half_widths(level, d)
    slack = c.r .- IA.diam.(c.eq.eta)
    all(w .< slack) || fail("sublevel: the ellipsoid at level $level leaves the outer box (raise outer_level)")
    return w
end

"""
    gradient_matrix_hull(c::SystemCandidate, xi_box, E, V) -> Matrix{Interval}

The reduced Hessian hull over the certificate's box `E` (with its KCL voltage box `V`): it
contains `int_0^1 Hess Hr(eta* + t xi) dt` for every `xi` in the box, so
`grad V(eta* + xi) = N xi` for some `N` in it.
"""
gradient_matrix_hull(c::SystemCandidate, xi_box::AbstractVector, E::AbstractVector, V::AbstractVector) =
    first(reduced_hessian_hull(c.model, c.sc, E, V))

candidate_fingerprint(c::SystemCandidate) =
    _digest("kind" => "system_lift", "model" => model_fingerprint(c.model), "Fc" => c.sc.Fc, "Fr" => c.sc.Fr,
            "terms" => [string(t.kind, ":", t.k, ":", t.coefficient) for t in c.sc.terms], "r" => c.r, "Hlo" => c.Hlo)

candidate_record(c::SystemCandidate) = Dict{String,Any}(
    "kind" => "nonlinear lift of H_sys (system-level mathematical candidate, not physical energy), Bregman-shifted at the enclosed equilibrium; certified set: the connected component of {V <= c} through the equilibrium",
    "fingerprint" => candidate_fingerprint(c),
    "construction" => Dict(k => v for (k, v) in c.construction if k != "inverse_diagonal_upper"))
