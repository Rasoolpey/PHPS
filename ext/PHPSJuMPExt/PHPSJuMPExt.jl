"""
Design extension, loaded with JuMP: conditioning, hull-aware LMI, PH-informed storage and
OPF-type programs, and the KYP / passivity tests (roadmap P10, II.1, II.2). Output of these
programs is a candidate only; certificates come from `roa/`.
"""
module PHPSJuMPExt

using PHPS
using JuMP
using LinearAlgebra

# A symmetric matrix of affine expressions, free on the pattern `mask` and zero elsewhere.
function _pattern_matrix(model, mask::AbstractMatrix{Bool})
    n = size(mask, 1)
    entries = [(i, j) for j in 1:n for i in 1:j if mask[i, j] || mask[j, i]]
    q = @variable(model, [1:length(entries)])
    Q = zeros(AffExpr, n, n)
    for (k, (i, j)) in enumerate(entries)
        Q[i, j] = q[k]
        Q[j, i] = q[k]
    end
    return Q, q, entries
end

# As'Q + QAs for Q on the pattern `mask`, built entry by entry over the pattern.
function _lyapunov_expr(As::AbstractMatrix, Q::AbstractMatrix, mask::AbstractMatrix{Bool})
    n = size(As, 1)
    cols = [findall(j -> mask[i, j] || mask[j, i], 1:n) for i in 1:n]
    M = zeros(AffExpr, n, n)
    for b in 1:n, a in b:n
        e = AffExpr(0.0)
        for c in cols[b]                         # (As'Q)[a,b] = sum_c As[c,a] Q[c,b]
            As[c, a] == 0 || add_to_expression!(e, As[c, a], Q[c, b])
        end
        for c in cols[a]                         # (Q As)[a,b] = sum_c Q[a,c] As[c,b]
            As[c, b] == 0 || add_to_expression!(e, As[c, b], Q[a, c])
        end
        M[a, b] = e
        M[b, a] = e
    end
    return M
end

_values(Q, q, entries, n) = begin
    V = zeros(n, n)
    for (k, (i, j)) in enumerate(entries)
        V[i, j] = value(q[k])
        V[j, i] = V[i, j]
    end
    V
end

_try(f, default = nothing) = try
    f()
catch
    default
end

# What the solver reported, and the software versions (the solver package's version is
# added by the caller, which loaded it).
function _solver_record(model)
    return Dict{String,Any}(
        "solver" => solver_name(model),
        "solver_version" => _try(() -> string(MOI.get(model, MOI.SolverVersion()))),
        "jump_version" => string(pkgversion(JuMP)),
        "termination_status" => string(termination_status(model)),
        "primal_status" => string(primal_status(model)),
        "dual_status" => string(dual_status(model)),
        "raw_status" => _try(() -> raw_status(model)),
        "solve_time_s" => _try(() -> solve_time(model)),
        "iterations" => _try(() -> barrier_iterations(model)),
        "objective_value" => _try(() -> objective_value(model)),
        "dual_objective_value" => _try(() -> dual_objective_value(model)))
end

_dualmat(c, n) = has_duals(owner_model(c)) ? Matrix(dual(c)) : fill(NaN, n, n)

function PHPS.structured_lyapunov(As::AbstractMatrix, mask::AbstractMatrix{Bool};
                                     weights = nothing, eps::Real = 1e-3, optimizer,
                                     silent::Bool = true)
    n = size(As, 1)
    size(mask) == (n, n) || throw(DimensionMismatch("mask must be $n x $n"))
    model = Model(optimizer)
    silent && set_silent(model)
    Q, q, entries = _pattern_matrix(model, mask)
    cQ = @constraint(model, Symmetric(Q - Matrix(1.0I, n, n)) in PSDCone())
    M = _lyapunov_expr(As, Q, mask)
    cM = @constraint(model, Symmetric(-M - eps * Matrix(1.0I, n, n)) in PSDCone())
    if weights !== nothing
        w = [weights[i, j] for (i, j) in entries]
        pen = findall(>(0), w)
        @variable(model, t[1:length(pen)] >= 0)
        for (r, k) in enumerate(pen)
            @constraint(model, t[r] >= q[k])
            @constraint(model, t[r] >= -q[k])
        end
        @objective(model, Min, sum(w[k] * t[r] for (r, k) in enumerate(pen); init = 0.0))
    end
    optimize!(model)
    Qv = has_values(model) ? _values(Q, q, entries, n) : zeros(n, n)
    return (Q = Qv, status = string(termination_status(model)), record = _solver_record(model),
            dual_Q = _dualmat(cQ, n), dual_rate = _dualmat(cM, n))
end

function PHPS.structured_completion(A::AbstractMatrix, Hfix::AbstractMatrix, mask::AbstractMatrix{Bool},
                                       basis::AbstractVector; optimizer, mu::Real = 0.0,
                                       nonnegative::AbstractVector{Bool} = fill(false, length(basis)),
                                       weights = nothing, basis_weights = nothing, t_min::Real = 0.0,
                                       groups = nothing, group_weights = nothing,
                                       silent::Bool = true)
    n = size(A, 1)
    size(Hfix) == (n, n) && size(mask) == (n, n) || throw(DimensionMismatch("A, Hfix and mask must be $n x $n"))
    model = Model(optimizer)
    silent && set_silent(model)
    F, f, entries = _pattern_matrix(model, mask)
    @variable(model, k[1:length(basis)])
    for (j, nn) in enumerate(nonnegative)
        nn && @constraint(model, k[j] >= 0)
    end
    @variable(model, t)
    P = Hfix .+ F
    for (j, Wj) in enumerate(basis)
        P = P .+ k[j] .* Wj
    end
    # the rate: A'(Hfix + sum k W) + (...)A in closed form, the free part entry by entry
    R0 = A' * Hfix + Hfix * A
    RW = [A' * Wj + Wj * A for Wj in basis]
    RF = _lyapunov_expr(A, F, mask)
    R = R0 .+ RF
    for (j, Rj) in enumerate(RW)
        R = R .+ k[j] .* Rj
    end
    Id = Matrix(1.0I, n, n)
    cP = @constraint(model, Symmetric(P .- t .* Id) in PSDCone())
    cR = @constraint(model, Symmetric(-R .- (2mu) .* P .- t .* Id) in PSDCone())
    if groups !== nothing
        # group sparsity: sum_g w_g ||F_g|| (second-order cones) plus L1 on the basis
        # coefficients, subject to t >= t_min
        @constraint(model, t >= t_min)
        pos = Dict(e => q for (q, e) in enumerate(entries))
        gw = group_weights === nothing ? ones(length(groups)) : collect(float(group_weights))
        @variable(model, sg[1:length(groups)] >= 0)
        for (g, G) in enumerate(groups)
            idx = unique([pos[i <= j ? (i, j) : (j, i)] for (i, j) in G])
            @constraint(model, [sg[g]; f[idx]] in SecondOrderCone())
        end
        bw = basis_weights === nothing ? zeros(length(basis)) : collect(float(basis_weights))
        bpen = findall(>(0), bw)
        @variable(model, sb[1:length(bpen)] >= 0)
        for (r, j) in enumerate(bpen)
            @constraint(model, sb[r] >= k[j])
            @constraint(model, sb[r] >= -k[j])
        end
        @objective(model, Min, sum(gw[g] * sg[g] for g in eachindex(groups); init = 0.0) +
                               sum(bw[j] * sb[r] for (r, j) in enumerate(bpen); init = 0.0))
    elseif weights === nothing
        @objective(model, Max, t)
    else
        # the sparsest completion with margin t >= t_min: weighted L1 of the free entries and
        # of the basis coefficients
        @constraint(model, t >= t_min)
        w = [weights[i, j] for (i, j) in entries]
        pen = findall(>(0), w)
        @variable(model, s[1:length(pen)] >= 0)
        for (r, e) in enumerate(pen)
            @constraint(model, s[r] >= f[e])
            @constraint(model, s[r] >= -f[e])
        end
        bw = basis_weights === nothing ? zeros(length(basis)) : collect(float(basis_weights))
        bpen = findall(>(0), bw)
        @variable(model, sb[1:length(bpen)] >= 0)
        for (r, j) in enumerate(bpen)
            @constraint(model, sb[r] >= k[j])
            @constraint(model, sb[r] >= -k[j])
        end
        @objective(model, Min, sum(w[e] * s[r] for (r, e) in enumerate(pen); init = 0.0) +
                               sum(bw[j] * sb[r] for (r, j) in enumerate(bpen); init = 0.0))
    end
    optimize!(model)
    record = _solver_record(model)
    record["free_entries"] = length(entries)
    record["basis_terms"] = length(basis)
    if !has_values(model)
        return (P = fill(NaN, n, n), t = NaN, coefficients = fill(NaN, length(basis)), F = fill(NaN, n, n),
                record = record, dual_P = fill(NaN, n, n), dual_rate = fill(NaN, n, n))
    end
    Fv = _values(F, f, entries, n)
    kv = value.(k)
    Pv = Hfix .+ Fv .+ sum((kv[j] .* basis[j] for j in eachindex(basis)); init = zeros(n, n))
    return (P = Pv, t = value(t), coefficients = kv, F = Fv, record = record,
            dual_P = _dualmat(cP, n), dual_rate = _dualmat(cR, n))
end

function PHPS.completion_rate_feasibility(A::AbstractMatrix, Hfix::AbstractMatrix, mask::AbstractMatrix{Bool},
                                             basis::AbstractVector; optimizer, mu::Real, cond_cap::Real,
                                             nonnegative::AbstractVector{Bool} = fill(false, length(basis)),
                                             silent::Bool = true)
    n = size(A, 1)
    model = Model(optimizer)
    silent && set_silent(model)
    F, f, entries = _pattern_matrix(model, mask)
    @variable(model, k[1:length(basis)])
    for (j, nn) in enumerate(nonnegative)
        nn && @constraint(model, k[j] >= 0)
    end
    @variable(model, g)
    P = Hfix .+ F
    R = (A' * Hfix + Hfix * A) .+ _lyapunov_expr(A, F, mask)
    for (j, Wj) in enumerate(basis)
        P = P .+ k[j] .* Wj
        R = R .+ k[j] .* (A' * Wj + Wj * A)
    end
    Id = Matrix(1.0I, n, n)
    cP = @constraint(model, Symmetric(P .- g .* Id) in PSDCone())
    cK = @constraint(model, Symmetric((cond_cap * g) .* Id .- P) in PSDCone())
    cR = @constraint(model, Symmetric(-R .- (2mu) .* P) in PSDCone())
    @objective(model, Max, g)
    optimize!(model)
    record = _solver_record(model)
    converged = termination_status(model) in (MOI.OPTIMAL, MOI.ALMOST_OPTIMAL)
    feasible = converged && has_values(model) && value(g) > 0
    Fv = has_values(model) ? _values(F, f, entries, n) : fill(NaN, n, n)
    kv = has_values(model) ? value.(k) : fill(NaN, length(basis))
    Pv = has_values(model) ? Hfix .+ Fv .+ sum((kv[j] .* basis[j] for j in eachindex(basis)); init = zeros(n, n)) :
         fill(NaN, n, n)
    return (feasible = feasible, gamma = has_values(model) ? value(g) : NaN, P = Pv, F = Fv, coefficients = kv,
            record = record, dual_P = _dualmat(cP, n), dual_cond = _dualmat(cK, n), dual_rate = _dualmat(cR, n))
end

function PHPS.completion_margin(A::AbstractMatrix, Hfix::AbstractMatrix, mask::AbstractMatrix{Bool},
                                   basis::AbstractVector; optimizer, cond_cap::Real,
                                   nonnegative::AbstractVector{Bool} = fill(false, length(basis)),
                                   silent::Bool = true)
    n = size(A, 1)
    model = Model(optimizer)
    silent && set_silent(model)
    F, f, entries = _pattern_matrix(model, mask)
    @variable(model, k[1:length(basis)])
    for (j, nn) in enumerate(nonnegative)
        nn && @constraint(model, k[j] >= 0)
    end
    @variable(model, alpha >= 0)
    @variable(model, t)
    P = alpha .* Hfix .+ F
    R = alpha .* (A' * Hfix + Hfix * A) .+ _lyapunov_expr(A, F, mask)
    for (j, Wj) in enumerate(basis)
        P = P .+ k[j] .* Wj
        R = R .+ k[j] .* (A' * Wj + Wj * A)
    end
    Id = Matrix(1.0I, n, n)
    @constraint(model, Symmetric(Id .- P) in PSDCone())
    @constraint(model, Symmetric(P .- (1 / cond_cap) .* Id) in PSDCone())
    @constraint(model, Symmetric(-R .- t .* Id) in PSDCone())
    @objective(model, Max, t)
    optimize!(model)
    record = _solver_record(model)
    record["free_entries"] = length(entries)
    converged = termination_status(model) in (MOI.OPTIMAL, MOI.ALMOST_OPTIMAL)
    ok = converged && has_values(model)
    Fv = ok ? _values(F, f, entries, n) : fill(NaN, n, n)
    kv = ok ? value.(k) : fill(NaN, length(basis))
    av = ok ? value(alpha) : NaN
    Pv = ok ? av .* Hfix .+ Fv .+ sum((kv[j] .* basis[j] for j in eachindex(basis)); init = zeros(n, n)) : fill(NaN, n, n)
    return (feasible = ok && value(t) > 0, t = ok ? value(t) : NaN, alpha = av, P = Pv, F = Fv,
            coefficients = kv, record = record)
end

function PHPS.minimal_completion(A::AbstractMatrix, Hfix::AbstractMatrix, mask::AbstractMatrix{Bool};
                                    optimizer, mu::Real, cond_cap::Real, silent::Bool = true)
    n = size(A, 1)
    size(Hfix) == (n, n) && size(mask) == (n, n) || throw(DimensionMismatch("A, Hfix and mask must be $n x $n"))
    model = Model(optimizer)
    silent && set_silent(model)
    F, f, entries = _pattern_matrix(model, mask)
    @variable(model, g >= 0)
    @variable(model, nF >= 0)
    P = Hfix .+ F
    R = (A' * Hfix + Hfix * A) .+ _lyapunov_expr(A, F, mask)
    Id = Matrix(1.0I, n, n)
    cP = @constraint(model, Symmetric(P .- g .* Id) in PSDCone())
    cK = @constraint(model, Symmetric((cond_cap * g) .* Id .- P) in PSDCone())
    cR = @constraint(model, Symmetric(-R .- (2mu) .* P) in PSDCone())
    # Frobenius norm of F: off-diagonal entries count twice
    wts = [i == j ? 1.0 : sqrt(2.0) for (i, j) in entries]
    @constraint(model, [nF; wts .* f] in SecondOrderCone())
    @objective(model, Min, nF)
    optimize!(model)
    record = _solver_record(model)
    record["free_entries"] = length(entries)
    converged = termination_status(model) in (MOI.OPTIMAL, MOI.ALMOST_OPTIMAL)
    feasible = converged && has_values(model) && value(g) > 0
    Fv = has_values(model) ? _values(F, f, entries, n) : fill(NaN, n, n)
    return (feasible = feasible, F = Fv, P = Hfix .+ Fv, gamma = has_values(model) ? value(g) : NaN,
            norm = has_values(model) ? value(nF) : NaN, record = record,
            dual_P = _dualmat(cP, n), dual_cond = _dualmat(cK, n), dual_rate = _dualmat(cR, n))
end

function PHPS.decay_margin(As::AbstractMatrix, mask::AbstractMatrix{Bool}; optimizer,
                              weights = nothing, rate = nothing, silent::Bool = true)
    n = size(As, 1)
    size(mask) == (n, n) || throw(DimensionMismatch("mask must be $n x $n"))
    model = Model(optimizer)
    silent && set_silent(model)
    P, p, entries = _pattern_matrix(model, mask)
    cP = @constraint(model, Symmetric(P) in PSDCone())
    R = _lyapunov_expr(As, P, mask)
    Id = Matrix(1.0I, n, n)
    if weights === nothing
        @variable(model, lam)
        @constraint(model, sum(P[i, i] for i in 1:n) == 1)
        cR = @constraint(model, Symmetric(lam * Id - R) in PSDCone())
        @objective(model, Min, lam)
    else
        (rate === nothing || rate <= 0) && throw(ArgumentError("weights need a rate > 0"))
        cR = @constraint(model, Symmetric(-R - rate * Id) in PSDCone())
        w = [weights[i, j] for (i, j) in entries]
        pen = findall(>(0), w)
        @variable(model, t[1:length(pen)] >= 0)
        for (r, k) in enumerate(pen)
            @constraint(model, t[r] >= p[k])
            @constraint(model, t[r] >= -p[k])
        end
        @objective(model, Min, sum(w[k] * t[r] for (r, k) in enumerate(pen); init = 0.0))
    end
    optimize!(model)
    record = _solver_record(model)
    record["free_entries"] = length(entries)
    if !has_values(model)
        return (P = fill(NaN, n, n), gamma = NaN, Z = fill(NaN, n, n), M = fill(NaN, n, n),
                record = record)
    end
    Pv = _values(P, p, entries, n)
    Z = _dualmat(cR, n)
    Mv = _dualmat(cP, n)
    gamma = eigmax(Symmetric(As' * Pv + Pv * As)) / tr(Pv)
    if all(isfinite, Z) && weights === nothing
        merge!(record, PHPS.margin_residuals(As, mask, Pv, Z, Mv))
    end
    return (P = Pv, gamma = gamma, Z = Z, M = Mv, record = record)
end

function PHPS.block_lyapunov_conditioning(As::AbstractMatrix, mask::AbstractMatrix{Bool}; mu::Real,
                                             optimizer, silent::Bool = true)
    n = size(As, 1)
    size(mask) == (n, n) || throw(DimensionMismatch("mask must be $n x $n"))
    model = Model(optimizer)
    silent && set_silent(model)
    W, w, entries = _pattern_matrix(model, mask)
    @variable(model, kappa)
    Id = Matrix(1.0I, n, n)
    cL = @constraint(model, Symmetric(W .- Id) in PSDCone())
    cU = @constraint(model, Symmetric(kappa .* Id .- W) in PSDCone())
    cR = @constraint(model, Symmetric(-_lyapunov_expr(As, W, mask) .- (2mu) .* W) in PSDCone())
    @objective(model, Min, kappa)
    optimize!(model)
    record = _solver_record(model)
    record["free_entries"] = length(entries)
    converged = termination_status(model) in (MOI.OPTIMAL, MOI.ALMOST_OPTIMAL)
    return (feasible = converged && has_values(model),
            W = has_values(model) ? _values(W, w, entries, n) : fill(NaN, n, n),
            kappa = has_values(model) ? value(kappa) : NaN, record = record,
            dual_lower = _dualmat(cL, n), dual_upper = _dualmat(cU, n), dual_rate = _dualmat(cR, n))
end

function PHPS.bm_multiplier_feasibility(As::AbstractMatrix, Hs::AbstractMatrix, mask::AbstractMatrix{Bool};
                                           mu::Real, cond_cap::Real, optimizer, silent::Bool = true)
    n = size(As, 1)
    size(Hs) == (n, n) && size(mask) == (n, n) || throw(DimensionMismatch("As, Hs and mask must be $n x $n"))
    model = Model(optimizer)
    silent && set_silent(model)
    W, w, entries = _pattern_matrix(model, mask)
    @variable(model, lambda)
    S = lambda .* Hs .+ Hs * W * Hs
    Id = Matrix(1.0I, n, n)
    cL = @constraint(model, Symmetric(S .- Id) in PSDCone())               # alpha = 1 (homogeneous)
    cU = @constraint(model, Symmetric(cond_cap .* Id .- S) in PSDCone())
    cR = @constraint(model, Symmetric(-(As' * S .+ S * As) .- (2mu) .* S) in PSDCone())
    optimize!(model)
    record = _solver_record(model)
    record["free_entries"] = length(entries)
    converged = termination_status(model) in (MOI.OPTIMAL, MOI.ALMOST_OPTIMAL)
    Mv = has_values(model) ? _values(W, w, entries, n) : fill(NaN, n, n)
    lv = has_values(model) ? value(lambda) : NaN
    return (feasible = converged && has_values(model), lambda = lv, M = Mv,
            S = has_values(model) ? lv .* Hs .+ Hs * Mv * Hs : fill(NaN, n, n), record = record,
            dual_lower = _dualmat(cL, n), dual_upper = _dualmat(cU, n), dual_rate = _dualmat(cR, n))
end

function PHPS.krasovskii_conditioning(As::AbstractMatrix, mask::AbstractMatrix{Bool}; mu::Real,
                                         cond_cap::Real, optimizer, silent::Bool = true)
    n = size(As, 1)
    size(mask) == (n, n) || throw(DimensionMismatch("mask must be $n x $n"))
    model = Model(optimizer)
    silent && set_silent(model)
    W, w, entries = _pattern_matrix(model, mask)
    @variable(model, gamma >= 0)
    @variable(model, kappa)
    Id = Matrix(1.0I, n, n)
    PK = As' * W * As
    @constraint(model, Symmetric(W .- gamma .* Id) in PSDCone())
    @constraint(model, Symmetric((cond_cap * gamma) .* Id .- W) in PSDCone())
    @constraint(model, Symmetric(PK .- Id) in PSDCone())
    @constraint(model, Symmetric(kappa .* Id .- PK) in PSDCone())
    @constraint(model, Symmetric(-_lyapunov_expr(As, W, mask) .- (2mu) .* W) in PSDCone())
    @objective(model, Min, kappa)
    optimize!(model)
    record = _solver_record(model)
    record["free_entries"] = length(entries)
    converged = termination_status(model) in (MOI.OPTIMAL, MOI.ALMOST_OPTIMAL)
    return (feasible = converged && has_values(model),
            W = has_values(model) ? _values(W, w, entries, n) : fill(NaN, n, n),
            kappa = has_values(model) ? value(kappa) : NaN,
            gamma = has_values(model) ? value(gamma) : NaN, record = record)
end

end
