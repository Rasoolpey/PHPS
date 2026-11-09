# The Krasovskii escape-class experiment on the electromagnetic dq pilots
# (docs/records/early/STABILITY_INVESTIGATION.md section 20, docs/records/early/NOGO_THEOREM.md section 5). For a constant
# weight W the Krasovskii storage H_K = 1/2 F'WF has the equilibrium Hessian P_K = A'WA, and by
# congruence it decays locally at rate mu exactly when W is a quadratic Lyapunov matrix of A at
# that rate. The LMIs are in the JuMP extension (`block_lyapunov_conditioning`,
# `krasovskii_conditioning`); everything here is solver independent.

"""
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
"""
function emdq_component_blocks(s::EMDQPilot; partition::Symbol = :finest)
    n = (s.mach.nq == 2 ? 19 : 18) + 1
    mach = emdq_machine_block(s.mach)
    b = if partition == :finest
        ["machine" => mach, "dc_link" => [8], "pi_integrator" => [n],
         "bus2_capacitor" => [9, 10], "line23" => [11, 12], "bus3_capacitor" => [13, 14],
         "line34" => [15, 16], "load_inductor" => [17, 18]]
    elseif partition == :node_centred
        ["machine+dc_link+pi_integrator+bus2_capacitor" => [mach; 8; n; 9; 10], "line23" => [11, 12],
         "bus3_capacitor+load_inductor" => [13, 14, 17, 18], "line34" => [15, 16]]
    elseif partition == :edge_centred
        ["machine+dc_link+pi_integrator" => [mach; 8; n], "bus2_capacitor+line23" => [9, 10, 11, 12],
         "bus3_capacitor+load_inductor" => [13, 14, 17, 18], "line34" => [15, 16]]
    else
        throw(ArgumentError("unknown partition $partition (:finest, :node_centred, :edge_centred)"))
    end
    sort(reduce(vcat, last.(b))) == 1:n || error("emdq_component_blocks: not a partition of 1:$n")
    return b
end

"""
    emdq_component_mask(s; partition = :finest) -> BitMatrix

Full symmetric blocks of `emdq_component_blocks(s; partition)`, zero across blocks.
"""
function emdq_component_mask(s::EMDQPilot; partition::Symbol = :finest)
    b = emdq_component_blocks(s; partition)
    n = maximum(maximum.(last.(b)))
    m = falses(n, n)
    for (_, ix) in b
        m[ix, ix] .= true
    end
    return m
end

"""
    emdq_krasovskii_energy(s, xs, c, d, W, z) -> H_K(z) = 1/2 F(z)' W_u F(z)

The Krasovskii storage of the closed loop `F = emdq_closed_loop_rhs(s, xs, c, .)` for a weight
`W` given in the declared scaling `d.d` of `d = emdq_completion_data(s, xs, c)`: with
`D = diag(d)` the scaled field is `D^-1 F`, so `W_u = D^-1 W D^-1` and the Hessian at the
equilibrium is `A_s' W A_s` in scaled coordinates.
"""
function emdq_krasovskii_energy(s::EMDQPilot, xs, c::EMDQAVR, d, W::AbstractMatrix, z)
    f = emdq_closed_loop_rhs(s, xs, c, z) ./ d.d
    return dot(f, W, f) / 2
end

"""
    emdq_krasovskii_balance(s, xs, c, d, W, z) -> NamedTuple

The exact rate identity of `H_K` for constant `W`: `dH_K/dt = F' W_u DF F` (no remainder).
Returns the rate along the flow (gradient of `H_K` dotted with `F`), the closed form, and
`identity_error` relative to the size of the closed form's terms.
"""
function emdq_krasovskii_balance(s::EMDQPilot, xs, c::EMDQAVR, d, W::AbstractMatrix, z)
    F(w) = emdq_closed_loop_rhs(s, xs, c, w)
    f = F(z)
    rate = dot(ForwardDiff.gradient(w -> emdq_krasovskii_energy(s, xs, c, d, W, w), z), f)
    Wu = W ./ (d.d * d.d')
    g = ForwardDiff.jacobian(F, z) * f
    closed = dot(f, Wu, g)
    scale = sum(abs, (Wu .* (f * g'))) + eps()
    return (; rate, closed, identity_error = (rate - closed) / scale)
end

"""
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
"""
function emdq_verify_krasovskii(W::AbstractMatrix, AsI::AbstractMatrix; mu::Real)
    WI = ival.((W + W') / 2)
    Q = .-(transpose(AsI) * WI .+ WI * AsI) .- ival(2mu) .* WI
    PK = transpose(AsI) * WI * AsI
    PK = [(PK[i, j] + PK[j, i]) / 2 for i in axes(PK, 1), j in axes(PK, 2)]
    wmin, qmin = verified_min_eig(WI), verified_min_eig(Q)
    pmin, pmax = verified_min_eig(PK), -verified_min_eig(.-PK)
    return (; proved_VW = wmin > 0 && qmin > 0, proved_HK = wmin > 0 && qmin > 0 && pmin > 0,
            W_min = wmin, decay_min = qmin, PK_min = pmin, PK_max = pmax,
            PK_cond_max = pmin > 0 ? pmax / pmin : Inf)
end

"""
    block_trace_obstruction(A, blocks; mu) -> Vector{NamedTuple}

Solver-free necessary condition for a block-diagonal Lyapunov matrix. If `W =
blockdiag(W_k) > 0` satisfies `-(A'W + WA) >= 2 mu W`, then each diagonal block gives
`A_kk'W_k + W_k A_kk <= -2 mu W_k`; the congruence by `W_k^(-1/2)` and the trace give
`tr A_kk <= -m_k mu` (`m_k` the block size). So a block with `tr A_kk + m_k mu > 0` excludes
every such `W` (and every Krasovskii weight of that form, by congruence). `A` may be a Float64
or an interval matrix (`AsI` of `emdq_interval_data`); with intervals the lower bound is
rigorous and `obstructs` holds for every matrix in the box. Returns, per block of `blocks`
(`emdq_component_blocks`), its name, size, the bounds of `tr A_kk + m_k mu` and `obstructs`.
"""
function block_trace_obstruction(A::AbstractMatrix, blocks; mu::Real)
    lo(x) = x isa IA.Interval ? IA.inf(x) : x
    hi(x) = x isa IA.Interval ? IA.sup(x) : x
    map(blocks) do (name, ix)
        t = eltype(A) <: IA.Interval ? sum(A[i, i] for i in ix) + ival(length(ix)) * ival(mu) :
                                       sum(A[i, i] for i in ix) + length(ix) * mu
        (; name, size = length(ix), margin_lo = lo(t), margin_hi = hi(t), obstructs = lo(t) > 0)
    end
end

"""
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
"""
function verified_block_instability(A::AbstractMatrix, ix::AbstractVector{<:Integer}; mu::Real)
    AI = eltype(A) <: IA.Interval ? A[ix, ix] : ival.(A[ix, ix])
    m = length(ix)
    S = [AI[i, j] + (i == j ? ival(mu) : ival(0.0)) for i in 1:m, j in 1:m]
    Sm = IA.mid.(S)
    X = lyap(Matrix(Sm'), -Matrix(1.0I, m, m))
    X = (X + X') / 2
    XI = ival.(X)
    Q = transpose(S) * XI .+ XI * S
    Q = [(Q[i, j] + Q[j, i]) / 2 for i in 1:m, j in 1:m]
    qmin = verified_min_eig(Q)
    E = eigen(Symmetric(X))
    v = ival.(E.vectors[:, end])
    vXv = IA.inf(sum(v[i] * XI[i, j] * v[j] for i in 1:m, j in 1:m))
    return (; proved = qmin > 0 && vXv > 0, lyapunov_min = qmin, vXv_lo = vXv,
            positive_eigs_X = count(>(0), E.values))
end

"""
    block_union_abscissae(A, blocks) -> Vector{NamedTuple}

Float64 diagnostic: the spectral abscissa of `A[T, T]` for every nonempty union `T` of the
blocks, sorted from the largest. A union with abscissa `>= -mu` violates the necessary
condition of `verified_block_instability` (which proves it on intervals).
"""
function block_union_abscissae(A::AbstractMatrix, blocks)
    q = length(blocks)
    out = map(1:(2^q-1)) do code
        T = [k for k in 1:q if (code >> (k - 1)) & 1 == 1]
        ix = reduce(vcat, [blocks[k][2] for k in T])
        (; blocks = [blocks[k][1] for k in T], ix, abscissa = maximum(real, eigvals(Matrix(A[ix, ix]))))
    end
    return sort(out; by = o -> o.abscissa, rev = true)
end

"""
    block_lyapunov_farkas_check(A, mask, r; mu) -> NamedTuple

Float64 check of the infeasibility certificate of `block_lyapunov_conditioning` (minimise
`kappa` subject to `I <= W <= kappa I`, `-(A'W + WA) >= 2 mu W`, `W` symmetric on `mask`). With
duals `Y1` (of `W >= I`), `Y2` (of `W <= kappa I`) and `Y3` (of the rate), all PSD, and
`M = Y1 - Y2 - (A Y3 + Y3 A') - 2 mu Y3`, a certificate has `M = 0` on the mask, `tr Y2 = 0` and
`tr Y1 > 0`: every feasible `W` would give `0 <= <W, M> + kappa tr Y2 - tr Y1 = -tr Y1 < 0`.
Returns the residuals relative to `tr Y1` and the smallest dual eigenvalues. A solver
certificate, not an interval proof.
"""
function block_lyapunov_farkas_check(A::AbstractMatrix, mask::AbstractMatrix{Bool}, r; mu::Real)
    Y1, Y2, Y3 = r.dual_lower, r.dual_upper, r.dual_rate
    M = Y1 - Y2 - (A * Y3 + Y3 * A') - 2mu * Y3
    t1 = tr(Y1)
    return (; trace_Y1 = t1, pattern = maximum(abs, M[mask]) / abs(t1), trace_Y2 = abs(tr(Y2)) / abs(t1),
            min_eig = (eigmin(Symmetric(Y1)), eigmin(Symmetric(Y2)), eigmin(Symmetric(Y3))))
end

"""
    block_lyapunov_conditioning(As, mask; mu, optimizer, silent = true)
        -> (feasible, W, kappa, record, dual_lower, dual_upper, dual_rate)

The best-conditioned quadratic Lyapunov matrix on a pattern: minimise `kappa` subject to
`I <= W <= kappa I`, `-(A'W + WA) >= 2 mu W`, `W` symmetric and free on `mask` (zero
elsewhere). The optimum `kappa` is the least achievable `cond(W)`. A converged solve means the
pattern admits decay at `mu` mathematically; comparing `kappa` with a cap is the separate
conditioning decision. A proper `INFEASIBLE` status comes with the duals checked by
`block_lyapunov_farkas_check`. Implemented in the JuMP extension; the result is a candidate.
"""
function block_lyapunov_conditioning end

"""
    krasovskii_conditioning(As, mask; mu, cond_cap, optimizer, silent = true)
        -> (feasible, W, kappa, gamma, record)

The best-conditioned Krasovskii Hessian on a pattern: minimise `kappa` subject to
`I <= A'WA <= kappa I`, `gamma I <= W <= cond_cap gamma I` (so `cond(W) <= cond_cap`) and
`-(A'W + WA) >= 2 mu W`, `W` symmetric on `mask`. The optimum `kappa` is the least achievable
`cond(A'WA)` among weights obeying the cap. Implemented in the JuMP extension; the result is a
candidate.
"""
function krasovskii_conditioning end
