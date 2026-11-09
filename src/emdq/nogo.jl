# Structured-Lyapunov no-go certificates and the locality diagnostic of the electromagnetic dq
# pilots (TODO.md, "Decision after the two electrical gates").
#
# A no-go certificate for a fixed physical Hessian H, linearisation A and admissible correction
# space S (symmetric matrices on a pattern) is a vector v with
#     v'(F A + A'F) v = 0 for every F in S   and   v'(H A + A'H) v > 0,
# so no P = H + F, F in S, has -(A'P + PA) positive semidefinite (no local decay, at any rate or
# conditioning). The sufficient construction used here: every correction touches only the
# states R (rows/columns of the pattern) and (A v)_R = 0.
#
# Gain lemma (the PI on y_m, `EMDQAVR`). With b = g_m(x*), c = grad y_m(x*), the closed loop is
#     A(kp, wi) = [A_p - kp u0 b c'/y0   -u0 b;   wi c'/y0   0],
# and H = blockdiag(H_p, u0 y0/wi). If eta is in R and wi > 0, the eta row of (A v)_R = 0 gives
# c'v_x = 0, so every kp term vanishes and the other rows of (A v)_R = 0 read
# (A_p v_x)_R - u0 b_R v_eta = 0; the rate sum_{j not in R} v' H_{:,j} (A v)_j never meets
# H_eta,eta and contains no kp. Hence K and the rate on K are the same for every real kp and
# every wi > 0: one interval proof at one gain pair covers the whole gain set.

"""    emdq_pilot_digest(s::EMDQPilot) -> SHA-256 of every pilot parameter (winding data included)."""
function emdq_pilot_digest(s::EMDQPilot)
    M = s.mach
    mv = Float64[getfield(M, f) for f in fieldnames(WindingMachine) if getfield(M, f) isa Real]
    sv = Float64[getfield(s, f) for f in fieldnames(EMDQPilot) if getfield(s, f) isa Real]
    return _digest("machine" => mv, "Kd" => M.Kd, "Kq" => M.Kq, "pilot" => sv,
                   "V4" => [real(s.V4), imag(s.V4)])
end

"""
    emdq_big_data(s, xs, c, d) -> NamedTuple

The scaled closed-loop Jacobian and Hessian recomputed in 256-bit arithmetic at the 256-bit
closed-loop equilibrium (Newton from `[xs; 0]`), with the scaling `d.d` of the Float64 data
`d = emdq_completion_data(s, xs, c)`: `As`, `Hs`, the equilibrium `z`, its residual and its
shift from the Float64 point, and the entrywise data errors `errA`, `errH` of `d`.
"""
function emdq_big_data(s::EMDQPilot, xs, c::EMDQAVR, d)
    nx = length(xs)
    zb = BigFloat.(vcat(xs, 0.0))
    for _ in 1:6
        zb -= ForwardDiff.jacobian(z -> emdq_closed_loop_rhs(s, xs, c, z), zb) \ emdq_closed_loop_rhs(s, xs, c, zb)
    end
    Ab = ForwardDiff.jacobian(z -> emdq_closed_loop_rhs(s, xs, c, z), zb)
    Hb = zeros(BigFloat, nx + 1, nx + 1)
    Hb[1:nx, 1:nx] = ForwardDiff.hessian(z -> emdq_energy(s, z), zb[1:nx])
    Hb[nx+1, nx+1] = d.Hfix[nx+1, nx+1]
    D = Diagonal(BigFloat.(d.d))
    As, Hs = D \ Ab * D, D * Hb * D
    return (; As, Hs, z = zb, residual = Float64(maximum(abs, emdq_closed_loop_rhs(s, xs, c, zb))),
            shift = Float64(maximum(abs, zb - vcat(xs, 0.0))),
            errA = Float64(maximum(abs, As - d.As)), errH = Float64(maximum(abs, Hs - d.Hs)))
end

_columns(M) = [M[:, j] for j in axes(M, 2)]

_interval_box(M, rel, a) = [(r = IA.sup(ival(rel) * ival(abs(x)) + ival(a)); ival(x) + ival(-r, r)) for x in M]

"""
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
"""
function emdq_interval_data(s::EMDQPilot, xs, c::EMDQAVR, d)
    F(z) = emdq_closed_loop_rhs(s, xs, c, z)
    nx = length(xs)
    zb = BigFloat.(vcat(xs, 0.0))
    for _ in 1:6
        zb -= ForwardDiff.jacobian(F, zb) \ F(zb)
    end
    center = Float64.(zb)
    C = inv(ForwardDiff.jacobian(F, center))
    r0 = 4 .* abs.(C * F(center)) .+ 8 .* eps.(max.(abs.(center), 1e-3))
    Z, X, rec = krawczyk(F, B -> ForwardDiff.jacobian(F, B), "pilot equilibrium", center, r0; C)
    AI = ForwardDiff.jacobian(F, Z)
    well_defined(AI) || fail("emdq_interval_data: the Jacobian is not well defined on the box")
    HI = fill(ival(0.0), nx + 1, nx + 1)
    HI[1:nx, 1:nx] = ForwardDiff.hessian(z -> emdq_energy(s, z), Z[1:nx])
    HI[nx+1, nx+1] = ival(c.u0) * ival(c.y0) / ival(c.omega_i)
    well_defined(HI) || fail("emdq_interval_data: the Hessian is not well defined on the box")
    dI = ival.(d.d)
    AsI = [AI[i, j] * dI[j] / dI[i] for i in 1:nx+1, j in 1:nx+1]
    HsI = [HI[i, j] * dI[i] * dI[j] for i in 1:nx+1, j in 1:nx+1]
    rec["equilibrium_width_max"] = maximum(IA.diam, Z)
    rec["A_width_max"] = maximum(IA.diam, AsI)
    rec["H_width_max"] = maximum(IA.diam, HsI)
    return (; Z, X, AI, HI, AsI, HsI, record = rec)
end

# The no-go rate on the witness for interval data: v_R = -A_RR^-1 A_RF w enclosed, (A v)_R = 0
# exactly, rate v' H (A v).
function _obstruction_rate(AI, HI, R, w)
    n = size(AI, 1)
    Fc = setdiff(1:n, R)
    wI = ival.(w)
    vR, beta = interval_solve(AI[R, R], -(AI[R, Fc] * wI))
    v = Vector{Ival}(undef, n)
    v[R] = vR
    v[Fc] = wI
    Av = zeros(Ival, n)
    Av[Fc] = AI[Fc, :] * v
    rate = sum(v[i] * sum(HI[i, j] * Av[j] for j in Fc) for i in 1:n)
    return rate, beta, v
end

"""
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
"""
function emdq_nogo_certificate(eq, s::EMDQPilot, xs; unit::AbstractString,
                               pattern::Symbol = :local_mask, kappa_p::Real = 1.0,
                               omega_i::Real = 1.0)
    c = emdq_avr(s, xs; kappa_p, omega_i)
    d = emdq_completion_data(s, xs, c)
    mask = getfield(d, pattern)
    d.eta in findall(i -> any(mask[i, :]), axes(mask, 1)) ||
        throw(ArgumentError("the gain lemma needs eta in the correction support"))
    iv = emdq_interval_data(s, xs, c, d)
    b = emdq_big_data(s, xs, c, d)
    o = emdq_local_obstruction(d; mask)
    n = size(mask, 1)
    w = o.direction[setdiff(1:n, o.R)]
    w ./= maximum(abs, w)
    rate, beta, _ = _obstruction_rate(iv.AsI, iv.HsI, o.R, w)
    AI, HI = iv.AsI, iv.HsI
    covered = all(IA.inf(AI[i]) <= b.As[i] <= IA.sup(AI[i]) for i in eachindex(AI)) &&
              all(IA.inf(HI[i]) <= b.Hs[i] <= IA.sup(HI[i]) for i in eachindex(HI))
    names = vcat(emdq_state_names(s.mach), "eta")
    entries = [(i, j) for j in 1:n for i in 1:j if mask[i, j]]
    return Dict{String,Any}(
        "kind" => "structured_lyapunov_nogo",
        "statement" => "for the pilot model at its unique equilibrium in the Krawczyk box: no " *
                       "P = H + F with F symmetric on the pattern has -(A'P + PA) positive " *
                       "semidefinite (no local quadratic decay at any rate or condition number)",
        "unit" => unit, "pattern" => string(pattern),
        "pattern_entries" => [[i, j] for (i, j) in entries],
        "R" => o.R, "R_names" => names[o.R], "state_names" => names,
        "case_digest" => system_digest(eq.sys), "pilot_digest" => emdq_pilot_digest(s),
        "software" => software_record(),
        "gains" => Dict("kappa_p" => Float64(kappa_p), "omega_i" => Float64(omega_i),
                        "u0" => c.u0, "y0" => c.y0),
        "gain_lemma" => "eta in R and omega_i > 0: the equilibrium, K and the rate on K are " *
                        "independent of kappa_p (any real) and omega_i (any positive); see " *
                        "src/emdq/nogo.jl",
        "data_source" => "model-bound: Krawczyk enclosure of the closed-loop equilibrium, " *
                         "interval Jacobian and Hessian over it (outward rounded)",
        "equilibrium" => Dict("x_float64" => xs, "Z_lo" => IA.inf.(iv.Z), "Z_hi" => IA.sup.(iv.Z),
                              "krawczyk" => iv.record),
        "scaling" => d.d, "box_width_max" => Dict("A" => iv.record["A_width_max"],
                                                   "H" => iv.record["H_width_max"]),
        "cross_check_256bit" => Dict("boxes_contain_256bit_data" => covered,
                                     "float64_error_A" => b.errA, "float64_error_H" => b.errH),
        "matrix_layout" => "list of columns",
        "A_lo" => _columns(IA.inf.(AI)), "A_hi" => _columns(IA.sup.(AI)),
        "H_lo" => _columns(IA.inf.(HI)), "H_hi" => _columns(IA.sup.(HI)),
        "w" => w, "dim_K" => o.dim_K, "max_rate_float64" => o.max_rate,
        "solve_beta" => beta, "rate_lo" => IA.inf(rate), "rate_hi" => IA.sup(rate),
        "proved" => IA.inf(rate) > 0)
end

"""
    check_nogo_certificate(rec) -> NamedTuple

Re-proves a record of `emdq_nogo_certificate` from its stored data alone (no model code): the
interval boxes `A`, `H`, the pattern's `R` and the witness components `w`. Checks that every
pattern entry lies in `R x R`, encloses `v_R = -A_RR^-1 A_RF w` (`interval_solve`) and the rate
`v' H (A v)` with `(A v)_R = 0`. `proved` when the lower bound is positive.
"""
function check_nogo_certificate(rec::AbstractDict)
    mat(x) = x isa AbstractMatrix ? Matrix{Float64}(x) : reduce(hcat, [Float64.(c) for c in x])
    Alo, Ahi, Hlo, Hhi = mat(rec["A_lo"]), mat(rec["A_hi"]), mat(rec["H_lo"]), mat(rec["H_hi"])
    AI = [ival(Alo[i], Ahi[i]) for i in CartesianIndices(Alo)]
    HI = [ival(Hlo[i], Hhi[i]) for i in CartesianIndices(Hlo)]
    R = Int.(rec["R"])
    all(e -> Int(e[1]) in R && Int(e[2]) in R, rec["pattern_entries"]) ||
        return (; proved = false, reason = "a pattern entry lies outside R x R")
    rate, beta, _ = _obstruction_rate(AI, HI, R, Float64.(rec["w"]))
    return (; proved = IA.inf(rate) > 0, rate_lo = IA.inf(rate), rate_hi = IA.sup(rate), beta,
            reason = "")
end

"""
    emdq_radius_masks(s, data) -> Vector{Pair{String,BitMatrix}}

The nested principal blocks of the locality diagnostic: the excitation assembly (machine
block, `q_dc`, `eta`); plus the terminal capacitor (`v2`); plus the first line (`i23`); plus
the next bus and its load (`v3`, `iL`); plus the second line (`i34`), which is every state.
Each mask is a full block (diagonals included).
"""
function emdq_radius_masks(s::EMDQPilot, data)
    n = size(data.Hs, 1)
    shells = [("excitation assembly (machine, q_dc, eta)", [emdq_machine_block(s.mach); 8; n]),
              ("+ terminal capacitor (v2)", [9, 10]), ("+ first line (i23)", [11, 12]),
              ("+ next bus and load (v3, iL)", [13, 14, 17, 18]),
              ("+ second line (i34): every state", [15, 16])]
    out = Pair{String,BitMatrix}[]
    S = Int[]
    for (lab, ix) in shells
        append!(S, ix)
        m = falses(n, n)
        m[S, S] .= true
        push!(out, lab => m)
    end
    return out
end

"""
    emdq_verify_lyapunov(data, P; mu, rel = 1e-6, abs_A = 0.0, AsI = nothing) -> NamedTuple

Interval check of a quadratic Lyapunov matrix `P` (scaled coordinates, taken as exact) for
every scaled linearisation in `AsI` (the model-bound enclosure of `emdq_interval_data`) or,
without it, in the box `As +- (rel |x| + abs_A)`: rigorous lower bounds
`P_min` on `lambda_min(P)` and `decay_min` on `lambda_min(-(A'P + PA) - 2 mu P)`
(`verified_min_eig`). `proved` when both are positive.
"""
function emdq_verify_lyapunov(data, P::AbstractMatrix; mu::Real, rel::Real = 1e-6, abs_A::Real = 0.0,
                              AsI = nothing)
    Ps = (P + P') / 2
    PI = ival.(Ps)
    AI = AsI === nothing ? _interval_box(data.As, rel, abs_A) : AsI
    Q = .-(transpose(AI) * PI .+ PI * AI) .- ival(2mu) .* PI
    pmin = verified_min_eig(PI)
    qmin = verified_min_eig(Q)
    return (; proved = pmin > 0 && qmin > 0, P_min = pmin, decay_min = qmin)
end
