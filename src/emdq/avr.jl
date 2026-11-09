# Gate 1, step 5 (TODO.md, "Method decision for step 5"): the passive PI on the converter's
# shifted output, the closed-loop identity, the linearisation and the data of the own-unit
# completion. The LMI itself is `minimal_completion` (JuMP extension).

"""
    EMDQAVR(kappa_p, omega_i, u0, y0)

Dimensionless passive PI on the converter's shifted output `y_m`:

    yhat = y_m / y0,   eta' = omega_i yhat,   u = m - m* = u0 (-kappa_p yhat - eta),
    H_AVR = u0 y0 eta^2 / (2 omega_i),

so `u y_m + dH_AVR/dt = -u0 y0 kappa_p yhat^2` exactly (while the modulation limit is inactive).
`u0` is the declared modulation headroom, `y0` the declared output scale.
"""
struct EMDQAVR
    kappa_p::Float64
    omega_i::Float64
    u0::Float64
    y0::Float64
end

"""
    emdq_avr(s, xs; kappa_p, omega_i) -> EMDQAVR

The PI with the declared scales of the pilot: headroom `u0 = min(m*, 1 - m*)` (a
two-quadrant converter, `0 <= m <= 1`) and output scale `y0 = v_dc* i_fd*` (the nominal field
power), so `yhat = i_fd/i_fd* - v_dc/v_dc*`.
"""
function emdq_avr(s::EMDQPilot, xs; kappa_p::Real, omega_i::Real)
    (kappa_p > 0 && omega_i > 0) || throw(ArgumentError("kappa_p and omega_i must be positive"))
    y0 = (xs[8] / s.Cdc) * emdq_machine(s.mach, xs).ifd
    return EMDQAVR(kappa_p, omega_i, min(s.mstar, 1 - s.mstar), y0)
end

"""    emdq_avr_energy(c, eta) -> H_AVR = u0 y0 eta^2 / (2 omega_i) (controller storage)."""
emdq_avr_energy(c::EMDQAVR, eta) = c.u0 * c.y0 * eta^2 / (2 * c.omega_i)

"""    emdq_modulation(s, xs, c, z) -> m = m* + u0 (-kappa_p yhat - eta) for z = [x; eta]."""
function emdq_modulation(s::EMDQPilot, xs, c::EMDQAVR, z)
    nx = length(xs)
    yhat = emdq_modulation_output(s, xs, view(z, 1:nx)) / c.y0
    return s.mstar + c.u0 * (-c.kappa_p * yhat - z[nx+1])
end

"""
    emdq_closed_loop_rhs!(dz, s, xs, c, z) -> dz

The pilot with the PI: `z = [x; eta]` (the plant states, then `eta`), the modulation from
`emdq_modulation`.
"""
function emdq_closed_loop_rhs!(dz, s::EMDQPilot, xs, c::EMDQAVR, z)
    nx = length(xs)
    x = view(z, 1:nx)
    emdq_rhs!(view(dz, 1:nx), s, x, emdq_modulation(s, xs, c, z))
    dz[nx+1] = c.omega_i * emdq_modulation_output(s, xs, x) / c.y0
    return dz
end

emdq_closed_loop_rhs(s::EMDQPilot, xs, c::EMDQAVR, z) =
    emdq_closed_loop_rhs!(similar(z), s, xs, c, z)

"""
    emdq_closed_loop_balance(s, xs, c, z) -> Dict

Gate 1.5: `d/dt (H_s + H_AVR) = -D_inc + r_rot - u0 y0 kappa_p yhat^2` along the closed
loop, every term and `identity_error` against the exact rate.
"""
function emdq_closed_loop_balance(s::EMDQPilot, xs, c::EMDQAVR, z)
    nx = length(xs)
    x = z[1:nx]
    m = emdq_modulation(s, xs, c, z)
    sb = emdq_shifted_balance(s, xs, x, m)
    gH = ForwardDiff.gradient(w -> emdq_energy(s, w), x) - ForwardDiff.gradient(w -> emdq_energy(s, w), xs)
    dz = emdq_closed_loop_rhs(s, xs, c, z)
    rate = dot(gH, dz[1:nx]) + c.u0 * c.y0 * z[nx+1] / c.omega_i * dz[nx+1]
    yhat = sb["y_m"] / c.y0
    damping = c.u0 * c.y0 * c.kappa_p * yhat^2
    inc = sum(v for (k, v) in sb if startswith(k, "inc_loss_"))
    return Dict("rate" => rate, "inc_loss" => inc, "residual_rotation" => sb["residual_rotation"],
                "avr_damping" => damping,
                "identity_error" => rate - (-inc + sb["residual_rotation"] - damping))
end

"""
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
"""
function emdq_completion_data(s::EMDQPilot, xs, c::EMDQAVR)
    nx = length(xs)
    n = nx + 1
    zs = vcat(xs, 0.0)
    A = ForwardDiff.jacobian(z -> emdq_closed_loop_rhs(s, xs, c, z), zs)
    Hp = ForwardDiff.hessian(z -> emdq_energy(s, z), xs)
    Hfix = zeros(n, n)
    Hfix[1:nx, 1:nx] = Hp
    Hfix[n, n] = c.u0 * c.y0 / c.omega_i
    d = [2.0^round(-log2(Hfix[i, i]) / 2) for i in 1:n]
    D = Diagonal(d)
    loc = [emdq_machine_block(s.mach); n]
    local_mask = falses(n, n)
    local_mask[loc, loc] .= true
    excitation_mask = copy(local_mask)
    for (i, j) in ((8, 3), (8, n))
        excitation_mask[i, j] = excitation_mask[j, i] = true
    end
    port_mask = copy(local_mask)
    port_mask[loc, 9:10] .= true
    port_mask[9:10, loc] .= true
    return (; A, Hfix, d, As = D \ A * D, Hs = D * Hfix * D, local_mask, excitation_mask,
            port_mask, block = loc, eta = n, spectral_abscissa = maximum(real, eigvals(A)))
end

"""
    completion_farkas_check(A, H, mask, r; mu, cond_cap) -> NamedTuple

Checks, in Float64, the dual certificate that `completion_rate_feasibility` returns when the
pattern is infeasible. With duals `Y1` (of `P >= g I`), `Y2` (of `P <= cond_cap g I`), `Y3`
(of `-(A'P + PA) >= 2 mu P`), all PSD, and `M = Y1 - Y2 - (A Y3 + Y3 A') - 2 mu Y3`, the
conditions are `M = 0` on the pattern, `cond_cap tr Y2 - tr Y1 = 0` and `<M, H> < 0`: then
every feasible point would give `0 <= <M, H> < 0`. Returns the normalised residuals
(`pattern`, `trace`, relative to `|<M, H>|`), `MH` and the smallest eigenvalues of the duals.
"""
function completion_farkas_check(A::AbstractMatrix, H::AbstractMatrix, mask::AbstractMatrix{Bool},
                                 r; mu::Real, cond_cap::Real)
    Y1, Y2, Y3 = r.dual_P, r.dual_cond, r.dual_rate
    M = Y1 - Y2 - (A * Y3 + Y3 * A') - 2mu * Y3
    MH = dot(M, H)
    sc = abs(MH)
    return (; MH, pattern = maximum(abs, M[mask]) * norm(H) / sc,
            trace = abs(cond_cap * tr(Y2) - tr(Y1)) / sc,
            min_eig = (eigmin(Symmetric(Y1)), eigmin(Symmetric(Y2)), eigmin(Symmetric(Y3))),
            scale = (norm(Y1), norm(Y2), norm(Y3)))
end

"""
    emdq_local_obstruction(data; mask = data.local_mask) -> NamedTuple

Solver-free structural test of a correction pattern. Let `R` be the states that `mask`
touches (any free entry in their row). On `K = {v : (A v)_R = 0}` the rate of every
`P = Hfix + F` with `F` free on `mask` is `v' (A'P + PA) v = 2 v' Hfix (A v)`: `F` drops out
because each term `v_i F_ij (A v)_j` has `j` in `R`. If this fixed form has a positive
eigenvalue on `K` (`max_rate > 0`), then `-(A'P + PA)` is not positive semidefinite for any
such `P`: no completion on the pattern decays, at any rate target or condition cap (the
gains enter only through `A`). Returns `max_rate` (scaled coordinates), the direction, `dim_K`
and `R`. `emdq_verify_obstruction` proves the sign in interval arithmetic.
"""
function emdq_local_obstruction(data; mask::AbstractMatrix{Bool} = data.local_mask)
    n = size(mask, 1)
    R = findall(i -> any(mask[i, :]), 1:n)
    K = nullspace(data.As[R, :])
    size(K, 2) == 0 &&               # K = {0}: no witness of this kind exists
        return (; max_rate = -Inf, direction = zeros(n), dim_K = 0, R)
    S = K' * (data.Hs * data.As + data.As' * data.Hs) * K / 2
    E = eigen(Symmetric(S))
    return (; max_rate = E.values[end], direction = K * E.vectors[:, end], dim_K = size(K, 2), R)
end

"""
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
"""
function emdq_verify_obstruction(data, w::AbstractVector; mask::AbstractMatrix{Bool} = data.local_mask,
                                 rel::Real = 1e-6, abs_A::Real = 0.0, abs_H::Real = 0.0,
                                 AsI = nothing, HsI = nothing)
    n = size(mask, 1)
    R = findall(i -> any(mask[i, :]), 1:n)
    length(w) == n - length(R) || throw(DimensionMismatch("w must have $(n - length(R)) entries"))
    # interval data: the model-bound enclosures (`emdq_interval_data`) when given, otherwise
    # boxes around the Float64 data
    AI = AsI === nothing ? _interval_box(data.As, rel, abs_A) : AsI
    HI = HsI === nothing ? _interval_box(data.Hs, rel, abs_H) : HsI
    rate, beta, _ = _obstruction_rate(AI, HI, R, w)
    return (; proved = IA.inf(rate) > 0, rate_lo = IA.inf(rate), rate_hi = IA.sup(rate),
            beta, rel, abs_A, abs_H, R)
end

"""
    emdq_repair_report(data, P) -> NamedTuple

The size and effect of a completed `P` (scaled coordinates, as returned by the LMI) against
`Hs` (scaled `Hfix`): `R = P - Hs`; Frobenius and operator-norm ratios
`|R| / |Hs|`; support (entries above 1e-9 of the largest) and its density in the symmetric
matrix; extremal eigenvalues and condition numbers of `Hs` and `P`; the generalised decay rate
of `P`; the curvature of `Hs`, `R` and `P` on `Hs`'s most negative direction; and the rate
form `sym(P As)` on the slice `{Ddelta = Dp = 0}` (before: `Hs`; after: `P`).
"""
function emdq_repair_report(data, P::AbstractMatrix)
    Hs, As = data.Hs, data.As
    R = P - Hs
    tol = 1e-9 * maximum(abs, R)
    supp = count(abs.(R) .> tol)
    n = size(P, 1)
    eH, eP = eigvals(Symmetric(Hs)), eigvals(Symmetric(P))
    E = eigen(Symmetric(Hs))
    v = E.vectors[:, 1]
    keep = setdiff(1:n, (6, 7))
    S(M) = (M' * As + As' * M) / 2
    slice(M) = eigmax(Symmetric(S(M)[keep, keep]))
    return (; R, fro_ratio = norm(R) / norm(Hs), op_ratio = opnorm(R) / opnorm(Hs),
            support = supp, density = supp / n^2, min_H = eH[1], max_H = eH[end],
            min_P = eP[1], max_P = eP[end], cond_P = eP[end] / eP[1],
            decay = eP[1] > 0 ? generalized_decay_rate(P, As) : NaN,
            neg_dir = (H = v' * Hs * v, R = v' * R * v, P = v' * P * v),
            slice_rate = (H = slice(Hs), P = slice(P)))
end
