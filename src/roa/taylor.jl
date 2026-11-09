# The second-order Taylor decay bound for quadratic candidates (docs/records/early/STABILITY_INVESTIGATION.md
# section 25.3). The diagnostic of section 25.2 showed that V_P's level is limited by interval
# dependency in the mean-value decay matrix: every entry of sym(N'M) enclosed over the box,
# then Weyl. This bound keeps the exact quadratic term and bounds the cubic remainder as one
# P-weighted scalar.
#
# With V = xi' P xi, the reduced field h(eta) = h(eta, v(eta)) on the KCL branch, the exact
# enclosed equilibrium eta* and A = Dh(eta*):
#   h(eta* + xi) = A xi + r2(xi),   r2(xi) = int_0^1 (1 - t) D2h(eta* + t xi)[xi, xi] dt,
#   dV/dt = xi'(A'P + PA) xi + 2 xi' P r2(xi).
# Quadratic term: xi'(A'P + PA) xi <= -m |xi|^2 with m from the Jacobian hull on the thin
# equilibrium boxes (the equilibrium uncertainty).
# Remainder: with D_k = dJ/deta_k (the reduced Jacobian's derivative along coordinate k,
# the implicit KCL part included: dDv_k = -g_V^-1 (dg_eta_k + dg_V_k Dv), the directional
# formula g_V V_xx[u,u] = -(g_xx[u,u] + 2 g_xV[u, V_x u] + g_VV[V_x u, V_x u]) at u = e_k),
#   xi' P D2h[xi, xi] = sum_k xi_k xi' sym(P D_k) xi,
# so |2 xi' P r2| <= sup_eta |sum_k xi_k xi' S_k xi| <= |xi|^2 sum_k |xi_k| sigma_k with
# sigma_k >= ||sym(P D_k)||_2 over the box. On the sublevel ellipsoid {xi' P xi <= c},
# sum_k |xi_k| sigma_k <= B = min(sum_k w_k sigma_k, sqrt(c sigma'|P^-1| sigma)) (the box
# half-widths w; Cauchy-Schwarz in the P geometry). So dV/dt <= -(m - B)|xi|^2 < 0 off the
# equilibrium when B < m. The slices D_k are formed one at a time; only sigma_k is kept.

# a rigorous upper bound on ||S||_2 for every symmetric matrix in the interval matrix S: the
# larger of the verified extreme eigenvalue bounds when both are finite, never above the
# outward-rounded Frobenius bound (which also covers a midpoint at or near zero, where the
# eigenvalue bound cannot separate)
function _sym_norm_upper(S::AbstractMatrix{<:IA.Interval})
    frob = IA.sup(sqrt(sum(ival(IA.mag(s))^2 for s in S)))
    e = max(verified_max_eig(S), verified_max_eig(.-S))
    return isfinite(e) ? min(max(e, 0.0), frob) : frob
end

# sigma_k >= ||sym(P D_k)||_2 for every k, with D_k enclosed over E x V on the KCL branch
function _taylor_sigmas(m::AbstractSectionModel, P::AbstractMatrix{Float64}, E, V)
    n = neta(m)
    nv = nvolt(m)
    _, rec, Dv = _reduced_jacobian_hull(m, E, V)
    base = vcat(collect(E), collect(V))
    PI = ival.(P)
    sig = zeros(n)
    h_V = g_V = nothing
    wmax = 0.0
    for k in 1:n
        seeds = [i <= n ? ival(i == k ? 1.0 : 0.0) : Dv[i - n, k] for i in 1:n+nv]
        x = [ForwardDiff.Dual{_CenteredTag}(base[i], seeds[i]) for i in 1:n+nv]
        J = ForwardDiff.jacobian(z -> section_residual(m, z), x)
        well_defined(J) || fail("Taylor bound: the second-order jets are not well defined on the box")
        if h_V === nothing
            Jv = map(ForwardDiff.value, J)
            h_V, g_V = Jv[1:n, n+1:end], Jv[n+1:end, n+1:end]
        end
        dJ = map(y -> ForwardDiff.partials(y)[1], J)
        dh_eta, dh_V = dJ[1:n, 1:n], dJ[1:n, n+1:end]
        dg_eta, dg_V = dJ[n+1:end, 1:n], dJ[n+1:end, n+1:end]
        dDv, _ = interval_solve(g_V, .-(dg_eta .+ dg_V * Dv))
        Dk = dh_eta .+ dh_V * Dv .+ h_V * dDv
        Sk = PI * Dk
        Sk = (Sk .+ permutedims(Sk)) ./ 2
        sig[k] = _sym_norm_upper(Sk)
        wmax = max(wmax, maximum(IA.diam, Dk))
    end
    rec["slice_width_max"] = wmax
    return sig, rec
end

"""
    taylor_level(c::QuadraticCandidate, level, eq; contracts) -> Dict

The gates of `certify_level` at one level, with the decay proved by the second-order Taylor
bound instead of the mean-value matrix: the sublevel box, the KCL branch, the quadratic
margin `m` (on the thin equilibrium boxes), the cubic bound `B` (the slices `sigma_k` over the
box, the two forms of the ellipsoid bound), `decay = B < m`, and the containment audit on the
same boxes. Returns the row; `row["certified"]` is the verdict.
"""
function taylor_level(c::QuadraticCandidate, level::Real, eq::EquilibriumEnclosure;
                      contracts::ContractSet = default_contracts())
    m = candidate_model(c)
    n = neta(m)
    t0 = time()
    row = Dict{String,Any}("level" => Float64(level), "method" => "second-order Taylor bound", "decay" => false,
                           "kcl_branch" => false, "containment" => false, "certified" => false)
    try
        w = sublevel_half_widths(c, level)
        Xi = [ival(-v, v) for v in w]
        E = eq.eta .+ Xi
        row["xi_half_width_max"] = maximum(w)
        row["xi_half_width_min"] = minimum(w)
        br = enclose_kcl_branch(m, E, IA.mid.(eq.V))
        row["kcl_branch"] = true
        all(IA.inf(x) <= IA.inf(e) && IA.sup(e) <= IA.sup(x) for (e, x) in zip(eq.V, br.X)) ||
            fail("the equilibrium voltages are not inside the KCL branch's uniqueness box")
        Mc, _, _ = _reduced_jacobian_hull(m, eq.eta, iintersect(eq.V, br.V))
        PI = ival.(c.P)
        Q = permutedims(Mc) * PI .+ PI * Mc
        Q = (Q .+ permutedims(Q)) ./ 2
        mq = -verified_max_eig(Q)                         # xi'(A'P + PA)xi <= -mq |xi|^2
        row["quadratic_margin"] = mq
        sig, srec = _taylor_sigmas(m, c.P, E, br.V)
        row["sigma"] = sig
        row["sigma_record"] = srec
        Pinv = get!(() -> IA.mag.(first(interval_solve(PI, ival.(Matrix{Float64}(I, n, n))))), c.construction,
                    "abs_inverse_upper")
        Bbox = IA.sup(sum(ival(w[k]) * ival(sig[k]) for k in 1:n))
        Bell = IA.sup(sqrt(ival(level) * sum(ival.(sig) .* (ival.(Pinv) * ival.(sig)))))
        B = min(Bbox, Bell)
        row["cubic_bound_box"] = Bbox
        row["cubic_bound_ellipsoid"] = Bell
        row["cubic_bound"] = B
        row["decay"] = mq > 0 && B < mq
        audit = containment_audit(m, E, br.V; contracts)
        row["containment"] = passed(audit)
        row["containment_categories"] = audit.passed
        row["box_digest"] = box_digest(E, br.V)
        row["same_boxes"] = audit.box_digest == row["box_digest"]
        row["certified"] = row["decay"] && row["kcl_branch"] && row["containment"] && row["same_boxes"]
    catch e
        e isa ProofFailure || e isa UndecidedBranch || e isa IA.InconclusiveBooleanOperation || rethrow()
        row["failure"] = sprint(showerror, e)
    end
    row["time_s"] = time() - t0
    return row
end

"""
    certify_roa_taylor(c::QuadraticCandidate; level_low, level_high, bisections = 20,
                       time_limit = 3600, resolution = 0.01, io = stdout) -> Dict

The certificate record with the Taylor decay bound: positivity, the equilibrium enclosure,
then `taylor_level` at `level_high`, else at `level_low` and a log-bisection between them, as in
`certify_roa`. `verified_valid_level` is the largest certified level found (a certified level,
not a proved maximum).
"""
function certify_roa_taylor(c::QuadraticCandidate; level_low::Real, level_high::Real, bisections::Integer = 20,
                            time_limit::Real = 3600.0, resolution::Real = 0.01,
                            contracts::ContractSet = default_contracts(), io::IO = stdout)
    0 < level_low <= level_high < Inf || throw(ArgumentError("need 0 < level_low <= level_high < Inf"))
    t0 = time()
    say(s) = (println(io, Printf.@sprintf("%6.1f s  ", time() - t0), s); flush(io))
    m = candidate_model(c)
    pos = positivity_proof(c)
    say("candidate positive: $(pos.proved) (lambda_min >= $(pos.lower_bound))")
    eq = enclose_equilibrium(m)
    rows = Dict{String,Any}[]
    function attempt(level)
        r = taylor_level(c, level, eq; contracts)
        push!(rows, r)
        say(Printf.@sprintf("level %.4e: %s (KCL %s, containment %s; margin %s, cubic bound %s) %.1f s", level,
                            r["certified"] ? "CERTIFIED" : "not certified", r["kcl_branch"], r["containment"],
                            string(get(r, "quadratic_margin", "-")), string(get(r, "cubic_bound", "-")), r["time_s"]))
        haskey(r, "failure") && say("    " * r["failure"])
        return r["certified"]
    end
    best = nothing
    if pos.proved
        if attempt(level_high)
            best = level_high
        elseif level_low < level_high && attempt(level_low)
            best = level_low
            lo, hi = log10(level_low), log10(level_high)
            for _ in 1:bisections
                time() - t0 > time_limit && (say("time limit reached"); break)
                hi - lo < resolution && break
                mid = (lo + hi) / 2
                attempt(10.0^mid) ? (best = 10.0^mid; lo = mid) : (hi = mid)
            end
        end
    end
    return Dict{String,Any}(
        "schema_version" => ROA_SCHEMA_VERSION,
        "method" => "second-order Taylor bound: quadratic margin on the thin equilibrium Jacobian hull; cubic remainder sum_k |xi_k| ||sym(P D_k)|| with the reduced Jacobian's coordinate derivatives D_k (implicit KCL part included) over the box; KCL branch by the parametric Krawczyk test; single-mode containment audit on the same boxes",
        "candidate" => candidate_record(c), "positivity" => Dict("proved" => pos.proved, "lower_bound" => pos.lower_bound),
        "equilibrium" => eq.record, "levels" => rows, "verified_valid_level" => best,
        "time_s" => time() - t0)
end

"""
    taylor_recheck(c::QuadraticCandidate, row) -> NamedTuple

Recomputes a certified row of `certify_roa_taylor` (or `taylor_level`) from the candidate and
the model: the equilibrium enclosure, the boxes, the KCL branch, the quadratic margin and every
slice bound `sigma_k`. It checks that the recomputed values reproduce the recorded ones (the
same boxes by digest) and that the inequality `B < m` holds with them.
"""
function taylor_recheck(c::QuadraticCandidate, row::AbstractDict)
    eq = enclose_equilibrium(candidate_model(c))
    r = taylor_level(c, row["level"], eq)
    same = get(r, "box_digest", "") == get(row, "box_digest", "x")
    reprod = same && isapprox(r["quadratic_margin"], row["quadratic_margin"]; rtol = 1e-9) &&
             isapprox(r["cubic_bound"], row["cubic_bound"]; rtol = 1e-9)
    return (; passed = r["certified"] && reprod, same_boxes = same, reproduced = reprod,
            quadratic_margin = get(r, "quadratic_margin", NaN), cubic_bound = get(r, "cubic_bound", NaN))
end

"""
    taylor_record_check(P, row) -> NamedTuple

The arithmetic part of a Taylor row rechecked from the record alone: from `P`, the level and
the recorded slice bounds `sigma`, the half-widths (`verified_inverse_diagonal`), `|P^-1|`
(an interval inverse) and both forms of the cubic bound are recomputed with outward rounding,
and `B < m` is checked against the recorded quadratic margin. The margin and the slices
themselves are rechecked from the model by `taylor_recheck`.
"""
function taylor_record_check(P::AbstractMatrix{Float64}, row::AbstractDict)
    n = size(P, 1)
    level = Float64(row["level"])
    sig = Float64.(row["sigma"])
    w = ellipsoid_half_widths(level, verified_inverse_diagonal(P))
    Pinv = IA.mag.(first(interval_solve(ival.(P), ival.(Matrix{Float64}(I, n, n)))))
    Bbox = IA.sup(sum(ival(w[k]) * ival(sig[k]) for k in 1:n))
    Bell = IA.sup(sqrt(ival(level) * sum(ival.(sig) .* (ival.(Pinv) * ival.(sig)))))
    B = min(Bbox, Bell)
    mq = Float64(row["quadratic_margin"])
    return (; passed = mq > 0 && B < mq, cubic_bound = B, quadratic_margin = mq,
            matches_record = B <= Float64(row["cubic_bound"]) * (1 + 1e-12) + 1e-300)
end
