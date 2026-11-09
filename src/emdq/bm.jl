# The Brayton-Moser (BM) gate of the electromagnetic dq pilots (docs/records/early/STABILITY_INVESTIGATION.md
# section 24). The pilot with its PI is an autonomous system conditional on its fixed AC and
# DC source boundaries. Q = Sigma Hess(E) is the signed BM differential matrix (one fixed sign
# hypothesis; not a metric, not necessarily symmetric), so Q F = Sigma d/dt grad E. The mixed
# potential P (with the modulation frozen at m*) and each named non-gradient term N_k are built
# independently, element by element; gate B0 checks Q F - grad P - sum_k N_k = 0.

# the flux states of the machine (stator pair, rotor windings) in pilot indexing
_bm_flux(s::EMDQPilot) = s.mach.nq == 2 ? [1, 2, 3, 4, 5, 19] : [1, 2, 3, 4, 5]

"""
    emdq_bm_signature(s) -> Vector{Float64}

The fixed BM sign hypothesis on `z = [x; eta]`: `-1` on the inductive, flux and momentum
states (machine fluxes, `p`, line and load-inductor currents), `+1` on the capacitive, charge
and position states (bus voltages, `q_dc`, the rotor angle) and on the PI integrator.
"""
function emdq_bm_signature(s::EMDQPilot)
    n = (s.mach.nq == 2 ? 19 : 18) + 1
    sig = ones(n)
    sig[[_bm_flux(s); 7; 11; 12; 15; 16; 17; 18]] .= -1.0
    return sig
end

_bm_energy(s::EMDQPilot, c::EMDQAVR, z) = emdq_energy(s, view(z, 1:length(z)-1)) + emdq_avr_energy(c, z[end])

"""
    emdq_bm_Q(s, c, z) -> (; Q, sym, skew)

The signed BM differential matrix `Q = Sigma Hess(E_phys + H_AVR)` at `z`, with its symmetric
and skew parts.
"""
function emdq_bm_Q(s::EMDQPilot, c::EMDQAVR, z)
    Q = Diagonal(emdq_bm_signature(s)) * ForwardDiff.hessian(w -> _bm_energy(s, c, w), z)
    return (; Q, sym = (Q + Q') / 2, skew = (Q - Q') / 2)
end

# the machine's co-state currents I = (Re i_n, Im i_n, i_fd, i_1d, i_1q[, i_2q]) and the torque
function _bm_currents(s::EMDQPilot, x)
    e = emdq_machine(s.mach, x)
    I = [real(e.in), imag(e.in), e.ifd, e.i1d, e.i1q]
    s.mach.nq == 2 && push!(I, e.i2q)
    return I, e.Tem
end

_bm_R(s::EMDQPilot) = (M = s.mach; R = [M.Ra, M.Ra, M.Rfd, M.R1d, M.R1q]; M.nq == 2 && push!(R, M.R2q); R)

# the applied winding voltages with the modulation frozen at m*: (v2/tap, m* v_dc, 0, ...)
function _bm_ustar(s::EMDQPilot, x)
    v2 = complex(x[9], x[10]) / s.tap
    u = [real(v2), imag(v2), s.mstar * x[8] / s.Cdc, zero(x[1]), zero(x[1])]
    s.mach.nq == 2 && push!(u, zero(x[1]))
    return u
end

"""
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
"""
function emdq_bm_potential(s::EMDQPilot, z)
    x = view(z, 1:length(z)-1)
    I, _ = _bm_currents(s, x)
    R = _bm_R(s)
    v2, i23 = complex(x[9], x[10]), complex(x[11], x[12])
    v3, i34, iL = complex(x[13], x[14]), complex(x[15], x[16]), complex(x[17], x[18])
    vdc = x[8] / s.Cdc
    om = x[7] / (2.0 * s.mach.H)
    dotc(a, b) = real(conj(a) * b)
    P = sum(R .* I .^ 2) / 2 - dot(I, _bm_ustar(s, x))
    P += (s.R23 * abs2(i23) + s.R34 * abs2(i34)) / 2 - s.G3 * abs2(v3) / 2 - (s.vs - vdc)^2 / (2 * s.Rs)
    P += -dotc(i23, v2 - v3) - dotc(i34, v3 - s.V4) - dotc(iL, v3)
    P += -s.Tm * om + s.Dm * (om - 1.0)^2 / 2
    return P
end

"""
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
"""
function emdq_bm_terms(s::EMDQPilot, xs, c::EMDQAVR, z)
    n = length(z)
    x = z[1:n-1]
    M = s.mach
    wb = M.omega_b
    fl = _bm_flux(s)
    F = emdq_closed_loop_rhs(s, xs, c, z)
    Ifun(w) = _bm_currents(s, w)[1]
    Tfun(w) = _bm_currents(s, w)[2]
    JI = ForwardDiff.jacobian(Ifun, x)            # d I / d x
    K = JI[:, fl]                                 # d I / d Psi
    dId = JI[:, 6]                                # d I / d delta
    gT = ForwardDiff.gradient(Tfun, x)
    I, Tem = _bm_currents(s, x)
    m = emdq_modulation(s, xs, c, z)
    vdc = x[8] / s.Cdc
    om = x[7] / (2.0 * M.H)
    T = eltype(z)
    gyr, mod, em = zeros(T, n), zeros(T, n), zeros(T, n)
    gpsi = zeros(T, length(fl))
    gpsi[1], gpsi[2] = -x[2], x[1]                # j psi_s on the stator pair
    gyr[fl] = K * gpsi
    for (a, b, C) in ((9, 10, s.C2), (13, 14, s.C3))           # -j C v
        gyr[a], gyr[b] = C * x[b], -C * x[a]
    end
    for (a, b, L) in ((11, 12, s.L23), (15, 16, s.L34), (17, 18, s.L3))   # +j L i
        gyr[a], gyr[b] = -L * x[b], L * x[a]
    end
    mod[fl] = -(m - s.mstar) * vdc .* K[:, 3]
    mod[8] = -(m - s.mstar) * I[3] / s.Cdc
    mod[n] = c.u0 * emdq_modulation_output(s, xs, x)
    em[fl] = -(om - 1.0) .* dId
    em[7] = -Tem / (2.0 * M.H)
    em[6] = -(dot(gT[fl], F[fl]) + gT[6] * F[6]) / wb - dot(dId, _bm_R(s) .* I .- _bm_ustar(s, x))
    return (; gyr, mod, em)
end

"""
    emdq_bm_balance(s, xs, c, z) -> NamedTuple

Gate B0's structural identity: `Q F - grad P - (N_gyr + N_mod + N_em)`, with `Q F` from the
Hessian of the energy and the field, `grad P` by automatic differentiation of the potential,
and the named terms from `emdq_bm_terms`. Returns the residual vector, its norm relative to the
sum of the parts' norms, and the norm of each part.
"""
function emdq_bm_balance(s::EMDQPilot, xs, c::EMDQAVR, z)
    QF = emdq_bm_Q(s, c, z).Q * emdq_closed_loop_rhs(s, xs, c, z)
    gP = ForwardDiff.gradient(w -> emdq_bm_potential(s, w), z)
    N = emdq_bm_terms(s, xs, c, z)
    r = QF - gP - N.gyr - N.mod - N.em
    scale = norm(QF) + norm(gP) + norm(N.gyr) + norm(N.mod) + norm(N.em)
    return (; residual = r, relative = norm(r) / max(scale, eps()), QF = norm(QF), gradP = norm(gP),
            gyr = norm(N.gyr), mod = norm(N.mod), em = norm(N.em))
end

"""
    emdq_bm_sharp(s, zs, z, lambda, Mw) -> P#(z)

The BM candidate `lambda P_s + 1/2 grad P_s' Mw grad P_s`, with the potential shifted at the
equilibrium `zs` (unscaled coordinates; `Mw` constant symmetric).
"""
function emdq_bm_sharp(s::EMDQPilot, zs, z, lambda, Mw)
    g0 = ForwardDiff.gradient(w -> emdq_bm_potential(s, w), zs)
    Ps = emdq_bm_potential(s, z) - emdq_bm_potential(s, zs) - dot(g0, z - zs)
    gs = ForwardDiff.gradient(w -> emdq_bm_potential(s, w), z) - g0
    return lambda * Ps + dot(gs, Mw * gs) / 2
end

"""
    emdq_bm_rate_check(s, xs, c, zs, z, lambda, Mw) -> NamedTuple

The exact rate identity of section 24.4:
`dP#/dt = xdot' Q'(lambda I + Mw Hess P) xdot - dN'(lambda I + Mw Hess P) xdot`, with
`dN = N(z) - N(zs)` (`N` the sum of the named terms). The rate along the flow is the gradient
of `P#` dotted with the field; `identity_error` is relative to the terms' sizes.
"""
function emdq_bm_rate_check(s::EMDQPilot, xs, c::EMDQAVR, zs, z, lambda, Mw)
    F = emdq_closed_loop_rhs(s, xs, c, z)
    rate = dot(ForwardDiff.gradient(w -> emdq_bm_sharp(s, zs, w, lambda, Mw), z), F)
    Q = emdq_bm_Q(s, c, z).Q
    HP = ForwardDiff.hessian(w -> emdq_bm_potential(s, w), z)
    Nsum(w) = (t = emdq_bm_terms(s, xs, c, w); t.gyr + t.mod + t.em)
    dN = Nsum(z) - Nsum(zs)
    Bm = lambda * I + Mw * HP
    t1 = dot(Q * F, Bm * F)
    t2 = dot(dN, Bm * F)
    return (; rate, formula = t1 - t2, bm_term = t1, remainder_term = t2,
            identity_error = (rate - (t1 - t2)) / (abs(t1) + abs(t2) + abs(rate) + eps()))
end

"""
    bm_multiplier_feasibility(As, Hs, mask; mu, cond_cap, optimizer, silent = true)
        -> (feasible, lambda, M, S, record, dual_lower, dual_upper, dual_rate)

Gate B1's local LMI in the declared scaled coordinates: find a scalar `lambda` and a
symmetric `M` free on `mask` (zero elsewhere) with `S = lambda Hs + Hs M Hs`,
`I <= S <= cond_cap I` (the conditioning LMI with `alpha = 1`, which the homogeneity in
`(lambda, M)` allows) and `-(As'S + S As) >= 2 mu S`. `Hs` is the scaled Hessian of the mixed
potential at the equilibrium. A converged solve means feasible; a proper `INFEASIBLE`
status comes with the duals for `bm_farkas_check`. Implemented in the JuMP extension; the
result is a candidate.
"""
function bm_multiplier_feasibility end

"""
    bm_farkas_check(As, Hs, mask, r; mu, cond_cap) -> NamedTuple

Float64 check of the infeasibility certificate of `bm_multiplier_feasibility`. With PSD duals
`Y1` (of `S >= I`), `Y2` (of `S <= cond_cap I`), `Y3` (of the rate) and
`Z = Y1 - Y2 - (As Y3 + Y3 As') - 2 mu Y3`, a certificate has `<Z, Hs> = 0` (the coefficient of
`lambda`), `(Hs Z Hs) = 0` on the mask (the coefficients of `M`) and
`tr Y1 - cond_cap tr Y2 > 0`. Every feasible point would then give
`0 <= -(tr Y1 - cond_cap tr Y2) < 0`. The residuals are normalised: `<Z, Hs>` by
`|Z| |Hs|`, the masked `Hs Z Hs` by `|Hs Z Hs|`, and the gap by `tr Y1 + cond_cap tr Y2`.
A solver certificate, not an interval proof.
"""
function bm_farkas_check(As::AbstractMatrix, Hs::AbstractMatrix, mask::AbstractMatrix{Bool}, r;
                         mu::Real, cond_cap::Real)
    Y1, Y2, Y3 = r.dual_lower, r.dual_upper, r.dual_rate
    Z = Y1 - Y2 - (As * Y3 + Y3 * As') - 2mu * Y3
    HZH = Hs * Z * Hs
    return (; gap = (tr(Y1) - cond_cap * tr(Y2)) / (tr(Y1) + cond_cap * tr(Y2)),
            lambda_coef = abs(dot(Z, Hs)) / (norm(Z) * norm(Hs)),
            pattern = maximum(abs, HZH[mask]) / norm(HZH),
            min_eig = (eigmin(Symmetric(Y1)), eigmin(Symmetric(Y2)), eigmin(Symmetric(Y3))))
end
