# The PCC coefficient maps of the two-machine pilot (docs/active/PCC_VIEW_N2_FORMULATION.md,
# sections 3-5; step N2 part 3 of ROA_ROUTES_TODO.md).
#
# Words follow N0's convention: c_w = L_{a_1} ... L_{a_l} h, the rightmost operator acts
# first, so the leftmost letter is the earliest in time. With h = v2 and the drift f (u_g = 0):
#   - drift words: c_{0^r}(x) = r! [t^r] h(phi_t(x)), the Taylor coefficients of the free
#     response;
#   - single-input words: c_{(0^a, j, 0^b)}(x) = L_f^a L_{g_j} L_f^b h (x)
#     = a! b! [s^a] [eps^1] [t^b] h(phi_t(phi_s(x) + eps g_j)): flow for s, a kick along the
#     (constant) input field g_j, then flow for t.
# Both are computed exactly (to rounding) by Taylor arithmetic on truncated series (`Jet`),
# nested for s, eps and t, with ForwardDiff duals inside for the state gradients. Time is
# scaled by `tau` (f and g multiplied by tau), so a coefficient of word length l carries tau^l.

"""
    Jet{T}

A truncated power series `sum_k c[k+1] t^k` (degree `length(c) - 1`), with exact arithmetic
on the kept coefficients. Coefficients may themselves be `Jet`s (nested variables) or duals.
"""
struct Jet{T} <: Number
    c::Vector{T}
end

function Jet(x0, N::Int)
    c = Vector{typeof(x0)}(undef, N + 1)
    c[1] = x0
    for k in 2:N+1
        c[k] = zero(x0)
    end
    return Jet(c)
end

Base.zero(a::Jet) = Jet([zero(x) for x in a.c])
Base.:+(a::Jet, b::Jet) = Jet([a.c[k] + b.c[k] for k in eachindex(a.c)])
Base.:-(a::Jet, b::Jet) = Jet([a.c[k] - b.c[k] for k in eachindex(a.c)])
Base.:-(a::Jet) = Jet([-x for x in a.c])
Base.:*(a::Jet, b::Real) = Jet([x * b for x in a.c])
Base.:*(b::Real, a::Jet) = a * b
Base.:/(a::Jet, b::Real) = Jet([x / b for x in a.c])
Base.:+(a::Jet, b::Real) = (c = copy(a.c); c[1] = c[1] + b; Jet(c))
Base.:+(b::Real, a::Jet) = a + b
Base.:-(a::Jet, b::Real) = a + (-b)
Base.:-(b::Real, a::Jet) = (-a) + b

function Base.:*(a::Jet, b::Jet)
    n = length(a.c)
    c = Vector{typeof(a.c[1] * b.c[1])}(undef, n)
    for k in 1:n
        acc = a.c[1] * b.c[k]
        for i in 2:k
            acc += a.c[i] * b.c[k-i+1]
        end
        c[k] = acc
    end
    return Jet(c)
end

# sin and cos together: k s_k = sum_j j u_j c_{k-j}, k c_k = -sum_j j u_j s_{k-j}
function _jet_sincos(u::Jet)
    n = length(u.c)
    s0, c0 = sin(u.c[1]), cos(u.c[1])
    S = Vector{typeof(s0)}(undef, n)
    C = Vector{typeof(c0)}(undef, n)
    S[1], C[1] = s0, c0
    for k in 1:n-1
        ss = u.c[2] * C[k]
        cc = -(u.c[2] * S[k])
        for j in 2:k
            ss += (j * u.c[j+1]) * C[k-j+1]
            cc -= (j * u.c[j+1]) * S[k-j+1]
        end
        S[k+1] = ss / k
        C[k+1] = cc / k
    end
    return Jet(S), Jet(C)
end

Base.sin(u::Jet) = _jet_sincos(u)[1]
Base.cos(u::Jet) = _jet_sincos(u)[2]

"""
    jet_flow(f!, x0, N) -> Vector{Jet}

The degree-`N` Taylor expansion of the solution of `dx/dt = f(x)`, `x(0) = x0`, by the Picard
recursion `x_{k+1} = f(x)_k / (k + 1)`; exact to rounding. `f!(dx, x)` must be generic.
"""
function jet_flow(f!, x0::AbstractVector, N::Int)
    n = length(x0)
    X = [Jet(x0[i], N) for i in 1:n]
    dx = Vector{Any}(undef, n)
    for k in 0:N-1
        f!(dx, X)
        nxt = [dx[i].c[k+1] / (k + 1) for i in 1:n]
        for i in 1:n
            X[i].c[k+2] = nxt[i]
        end
    end
    return X
end

"""
    pcc_words(s; K, A = max(K - 8, 0)) -> Vector{NamedTuple}

The words whose coefficients `pcc_coefficient_values` returns, in order: the drift words
`(a = 0, j = 0, b = r)` for `r = 0..K`, then the single-input words `(a, j, b)` with
`a + b + 1 <= K`, `a <= A`; each for the output components `out = 1, 2` (Re, Im of `v2`), with
the word length `len`.
"""
function pcc_words(s::EMDQPilot2; K::Int, A::Int = max(K - 8, 0))
    w = NamedTuple{(:a, :j, :b, :out, :len),NTuple{5,Int}}[]
    for r in 0:K, o in 1:2
        push!(w, (a = 0, j = 0, b = r, out = o, len = r))
    end
    for j in 1:2, b in 0:K-1, a in 0:min(A, K - 1 - b), o in 1:2
        push!(w, (; a, j, b, out = o, len = a + b + 1))
    end
    return w
end

"""
    pcc_coefficient_values(s, xg; K, tau, A = max(K - 8, 0)) -> Vector

The time-scaled coefficients `tau^len c_w(xg)` of the grid side seen from the PCC, for the words
of `pcc_words` (output `v2`, inputs the two components of `u_g`, drift at `u_g = 0`). Generic
in the element type of `xg`.
"""
function pcc_coefficient_values(s::EMDQPilot2, xg::AbstractVector; K::Int, tau::Real,
                                A::Int = max(K - 8, 0))
    G = emdq2_grid_layout(s)
    gain = tau * s.m2.mach.omega_b / s.C2
    f!(dx, X) = (emdq2_grid_rhs!(dx, s, X, 0, 0);
                 for i in eachindex(dx)
                     dx[i] = dx[i] * tau
                 end;
                 dx)
    vals = Any[]
    Xt = jet_flow(f!, xg, K)
    for r in 0:K, o in 1:2
        push!(vals, Xt[G.v2+o-1].c[r+1] * Float64(factorial(r)))
    end
    Xs = jet_flow(f!, xg, A)
    for j in 1:2
        Z = [Jet([Xs[i], zero(Xs[i]) + (i == G.v2 + j - 1 ? gain : 0.0)]) for i in 1:G.n]
        Y = jet_flow(f!, Z, K - 1)
        for b in 0:K-1, a in 0:min(A, K - 1 - b), o in 1:2
            push!(vals, Y[G.v2+o-1].c[b+1].c[2].c[a+1] * Float64(factorial(a) * factorial(b)))
        end
    end
    return [v for v in vals]
end

"""
    pcc_coefficient_jacobian(s, xg; K, tau, A) -> (values, J, words)

The coefficients and their state gradients (ForwardDiff over the Taylor arithmetic).
"""
function pcc_coefficient_jacobian(s::EMDQPilot2, xg::AbstractVector; K::Int, tau::Real,
                                  A::Int = max(K - 8, 0))
    f = x -> pcc_coefficient_values(s, x; K, tau, A)
    return f(xg), ForwardDiff.jacobian(f, xg), pcc_words(s; K, A)
end

"""
    pcc_two_input_jacobian(s, xg, j; K, tau, A) -> J2

State gradients of the scaled two-input coefficients `tau^(len+1) c_{(j, w)}` for every word
`w` of `pcc_words` (the input letter `j` placed earliest): `c_{(j,w)} = L_{g_j} c_w`, a
derivative of `c_w` along the constant input field. Row `i` belongs to the word `(j, words[i])`.
"""
function pcc_two_input_jacobian(s::EMDQPilot2, xg::AbstractVector, j::Int; K::Int, tau::Real,
                                A::Int = max(K - 8, 0))
    G = emdq2_grid_layout(s)
    gj = zeros(G.n)
    gj[G.v2+j-1] = tau * s.m2.mach.omega_b / s.C2
    f2 = x -> ForwardDiff.derivative(e -> pcc_coefficient_values(s, x .+ e .* gj; K, tau, A), 0.0)
    return ForwardDiff.jacobian(f2, xg)
end

# one-sided Jacobi (Hestenes) SVD in the element type of A: singular values in decreasing
# order and the right singular vectors (all n of them, so the kernel is included); throws if
# the sweeps do not converge
function _jacobi_svd(A::AbstractMatrix{T}) where {T}
    U = Matrix{T}(A)
    m, n = size(U)
    V = Matrix{T}(I, n, n)
    tol = eps(T) * n
    # a column below eps * |A| is rounding noise: its pairs are not rotated (the relative
    # orthogonality test cannot be met for it), and its norm is a zero singular value
    negl = (eps(T) * n)^2 * sum(abs2, U)
    converged = false
    for _ in 1:100
        rotated = false
        for p in 1:n-1, q in p+1:n
            a = sum(abs2, view(U, :, p))
            b = sum(abs2, view(U, :, q))
            (a <= negl || b <= negl) && continue
            c = dot(view(U, :, p), view(U, :, q))
            (abs(c) <= tol * sqrt(a * b) || iszero(c)) && continue
            rotated = true
            z = (b - a) / (2c)
            t = sign(z) / (abs(z) + sqrt(1 + z^2))
            iszero(z) && (t = one(T))
            cs = 1 / sqrt(1 + t^2)
            sn = cs * t
            for M in (U, V), i in axes(M, 1)
                up, uq = M[i, p], M[i, q]
                M[i, p] = cs * up - sn * uq
                M[i, q] = sn * up + cs * uq
            end
        end
        rotated || (converged = true; break)
    end
    converged || error("_jacobi_svd: no convergence in 100 sweeps")
    S = [norm(view(U, :, k)) for k in 1:n]
    perm = sortperm(S; rev = true)
    return S[perm], V[:, perm]
end

"""
    pcc_structural_zero(w) -> Bool

The words whose coefficients are constant by the structure of the pilot
(`docs/active/PCC_VIEW_N2_FORMULATION.md` section 3.2, confirmed by Codex): the single-input
words `(0^a, j, 0^b)` with `b <= 6` (constant for `a = 0`, zero for `a >= 1`).
"""
pcc_structural_zero(w) = w.j != 0 && w.b <= 6

"""
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
"""
function pcc_profile(J::AbstractMatrix, words, gradH::AbstractVector, scale::AbstractVector;
                     K::Int, drop = pcc_structural_zero, rank_tol::Real = 1e-13,
                     extra::AbstractMatrix = zeros(eltype(J), length(gradH), 0))
    T = eltype(J)
    Js = J .* T.(scale)'
    gH = gradH .* T.(scale)
    gH = gH / norm(gH)
    # further functionals to test (columns: gradients); a zero gradient is trivially in the span
    gX = Matrix{T}(extra .* T.(scale))
    for c in eachcol(gX)
        nx = norm(c)
        iszero(nx) || (c ./= nx)
    end
    rown = [norm(view(Js, i, :)) for i in axes(Js, 1)]
    dref = maximum(rown[i] for i in eachindex(words) if words[i].j == 0)
    dz = [rown[i] for i in eachindex(words) if drop(words[i])]
    dropped_max = isempty(dz) ? zero(T) : maximum(dz) / dref
    out = NamedTuple[]
    for k in 0:K
        rows = [i for i in eachindex(words) if words[i].len <= k && !drop(words[i])]
        M = reduce(vcat, [view(Js, i, :)' / rown[i] for i in rows])
        if T == Float64
            F = svd(M; full = true)
            S, Vf = F.S, F.V
        else
            S, Vf = _jacobi_svd(M)
        end
        r = count(>(rank_tol * S[1]), S)
        V = Vf[:, 1:r]
        res = norm(gH - V * (V' * gH))
        gap = r < length(S) ? S[r] / S[r+1] : T(Inf)
        e_extra = [norm(c - V * (V' * c)) for c in eachcol(gX)]
        push!(out, (; k, rank = r, nrows = length(rows), sv = S, gap, e_res = res, e_extra,
                    null = Vf[:, r+1:end]))
    end
    prof = [merge(p, (; closure_increment = k < K ? out[k+2].rank - p.rank : missing))
            for (k, p) in zip(0:K, out)]
    return prof, dropped_max
end
