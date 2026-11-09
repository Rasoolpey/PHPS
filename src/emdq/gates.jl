# Gate-1 identities of the electromagnetic dq pilot (docs/records/early/PHYSICAL_EXTENSION.md, "Gates";
# TODO.md, "Upcoming steps"). Every identity is evaluated term by term in closed form and
# compared with the exact rate grad(H)' f computed by automatic differentiation.

_emdq_grad(f, x) = ForwardDiff.gradient(f, x)

"""
    emdq_balance(s::EMDQPilot, x, m = s.mstar) -> Dict

Raw physical-energy balance (gate 1.2): `rate = grad H' f` against
`sum(supplies) - sum(dissipation)` (`emdq_power`), with every term and `identity_error`.
"""
function emdq_balance(s::EMDQPilot, x, m = s.mstar)
    rate = dot(_emdq_grad(z -> emdq_energy(s, z), x), emdq_rhs(s, x, m))
    pw = emdq_power(s, x, m)
    out = Dict{String,Float64}("rate" => rate)
    for (k, v) in pairs(pw.supplies)
        out["supply_" * string(k)] = v
    end
    for (k, v) in pairs(pw.dissipation)
        out["loss_" * string(k)] = v
    end
    out["identity_error"] = rate - (sum(pw.supplies) - sum(pw.dissipation))
    return out
end

"""
    emdq_shifted_energy(s, xs, x) -> H(x) - H(xs) - grad H(xs)' (x - xs)

The equilibrium-shifted (Bregman) energy in the physical coordinates: stator fluxes in the
network frame, rotor fluxes, the rotor angle and momentum, the DC charge and the network
states.
"""
function emdq_shifted_energy(s::EMDQPilot, xs, x)
    gs = _emdq_grad(z -> emdq_energy(s, z), xs)
    return emdq_energy(s, x) - emdq_energy(s, xs) - dot(gs, x - xs)
end

"""
    emdq_modulation_output(s, xs, x) -> y_m

The converter's incremental passive output `y_m = g_m' grad H_s` with
`g_m = df/dm` (field flux `omega_b v_dc`, DC charge `-i_fd`); in closed form
`y_m = v_dc* i_fd - i_fd* v_dc`. Zero for the raw energy (`grad H` instead of `grad H_s`):
the ideal converter is power preserving.
"""
function emdq_modulation_output(s::EMDQPilot, xs, x)
    es, e = emdq_machine(s.mach, xs), emdq_machine(s.mach, x)
    return (xs[8] / s.Cdc) * e.ifd - es.ifd * (x[8] / s.Cdc)
end

"""
    emdq_shifted_balance(s::EMDQPilot, xs, x, m = s.mstar) -> Dict

The shifted identity (gates 1.3 and 1.4) at an equilibrium `xs` of modulation `s.mstar`:

    dH_s/dt = -D_inc + r + (m - m*) y_m,

with the incremental dissipation `D_inc` (each resistor's `R |i - i*|^2`, the load's
`G |v - v*|^2`, the diagnostic shaft damping's `Dm (omega - 1)^2`) and the single residual `r = -Re(conj(Delta i_n) j Delta psi_n)` of the
stator's frame rotation (`-j psi_n` in the network-frame stator equation; with the exact
angle and torque pairing `dH/d delta = -Tem/omega_b`, everything else cancels). Returns every
term and `identity_error` against the exact rate `(grad H(x) - grad H(xs))' f(x, m)`.
"""
function emdq_shifted_balance(s::EMDQPilot, xs, x, m = s.mstar)
    H(z) = emdq_energy(s, z)
    rate = dot(_emdq_grad(H, x) - _emdq_grad(H, xs), emdq_rhs(s, x, m))
    M = s.mach
    es, e = emdq_machine(M, xs), emdq_machine(M, x)
    isd = ((s.vs - x[8] / s.Cdc) - (s.vs - xs[8] / s.Cdc)) / s.Rs
    d23 = complex(x[11] - xs[11], x[12] - xs[12])
    d34 = complex(x[15] - xs[15], x[16] - xs[16])
    D = Dict{String,Float64}(
        "stator" => M.Ra * abs2(e.in - es.in), "field" => M.Rfd * (e.ifd - es.ifd)^2,
        "dampers" => M.R1d * (e.i1d - es.i1d)^2 + M.R1q * (e.i1q - es.i1q)^2 +
                     M.R2q * (e.i2q - es.i2q)^2,
        "source" => s.Rs * isd^2, "lines" => s.R23 * abs2(d23) + s.R34 * abs2(d34),
        "load" => s.G3 * ((x[13] - xs[13])^2 + (x[14] - xs[14])^2),
        "shaft" => s.Dm * ((x[7] - xs[7]) / (2.0 * M.H))^2)
    dpsi = complex(x[1] - xs[1], x[2] - xs[2])
    r = -real(conj(e.in - es.in) * (im * dpsi))
    u = m - s.mstar
    ym = emdq_modulation_output(s, xs, x)
    out = Dict{String,Float64}("rate" => rate, "residual_rotation" => r,
                               "modulation_supply" => u * ym, "y_m" => ym)
    for (k, v) in D
        out["inc_loss_" * k] = v
    end
    out["identity_error"] = rate - (-sum(values(D)) + r + u * ym)
    return out
end

"""
    emdq_local_forms(s::EMDQPilot, xs) -> NamedTuple

Second-order forms at the equilibrium `xs` (modulation `s.mstar`): the Bregman Hessian `Hs`
(= Hess H(xs)), the Jacobian `A`, the rate form `Q = sym(Hs A)` (so `dH_s/dt = dx' Q dx +
O(|dx|^3)`), its parts `D` (incremental dissipation, PSD) and `Rr` (rotation residual) with
`closure = |Q - (-D + Rr)|`, and the linear functional `c` of `y_m` (`y_m = c' dx + O(2)`).
"""
function emdq_local_forms(s::EMDQPilot, xs)
    n = length(xs)
    Hs = ForwardDiff.hessian(z -> emdq_energy(s, z), xs)
    A = ForwardDiff.jacobian(z -> emdq_rhs(s, z), xs)
    Q = (Hs * A + A' * Hs) / 2
    # quadratic parts of the closed-form terms (half their Hessians at xs)
    keys_D = ("inc_loss_stator", "inc_loss_field", "inc_loss_dampers", "inc_loss_source",
              "inc_loss_lines", "inc_loss_load", "inc_loss_shaft")
    Dp = Dict(k => _emdq_piece(s, xs, k) for k in keys_D)
    D = sum(values(Dp))
    Rr = _emdq_piece(s, xs, "residual_rotation")
    c = _emdq_grad(z -> emdq_modulation_output(s, xs, z), xs)
    return (; Hs, A, Q, D, Dparts = Dp, Rr, c, closure = opnorm(Q - (-D + Rr)))
end

# Half the Hessian at xs of one closed-form term of `emdq_shifted_balance` (its quadratic
# form), rebuilt generically (the balance itself returns Float64 Dicts).
function _emdq_piece(s::EMDQPilot, xs, key)
    M = s.mach
    es = emdq_machine(M, xs)
    f = function (x)
        e = emdq_machine(M, x)
        if key == "inc_loss_stator"
            return M.Ra * abs2(e.in - es.in)
        elseif key == "inc_loss_field"
            return M.Rfd * (e.ifd - es.ifd)^2
        elseif key == "inc_loss_dampers"
            return M.R1d * (e.i1d - es.i1d)^2 + M.R1q * (e.i1q - es.i1q)^2 +
                     M.R2q * (e.i2q - es.i2q)^2
        elseif key == "inc_loss_source"
            return s.Rs * ((x[8] - xs[8]) / (s.Cdc * s.Rs))^2
        elseif key == "inc_loss_lines"
            return s.R23 * ((x[11] - xs[11])^2 + (x[12] - xs[12])^2) +
                   s.R34 * ((x[15] - xs[15])^2 + (x[16] - xs[16])^2)
        elseif key == "inc_loss_load"
            return s.G3 * ((x[13] - xs[13])^2 + (x[14] - xs[14])^2)
        elseif key == "inc_loss_shaft"
            return s.Dm * ((x[7] - xs[7]) / (2.0 * M.H))^2
        elseif key == "residual_rotation"
            dpsi = complex(x[1] - xs[1], x[2] - xs[2])
            return -real(conj(e.in - es.in) * (im * dpsi))
        end
        throw(ArgumentError(key))
    end
    return ForwardDiff.hessian(f, xs) / 2
end

"""
    restricted_max_eig(Q, c) -> max eigenvalue of the form `Q` on the hyperplane `c' dx = 0`

Finsler test for gate 1.3: if it is positive, no damping injection through `y_m` alone
(`u = -k y_m`, any `k`) makes `dH_s/dt` negative semidefinite.
"""
function restricted_max_eig(Q::AbstractMatrix, c::AbstractVector)
    N = nullspace(reshape(c, 1, :))
    return eigmax(Symmetric(N' * Q * N))
end

"""
    emdq_reduction_check(s::EMDQPilot, x) -> NamedTuple

Singular-limit comparison (gate requirement): with the stator quasi-static in the network
frame (`dpsi_n/dt = 0`, so `psi_n = -j (v_m - Ra i_n)`, the classical consistent pairing of
neglected stator transients with nominal speed in the stator EMF), the rotor derivatives of the
winding model, mapped to `E_q' = Lad psi_fd / Lffd`, `psi_1d`, `psi_q'' = Laq psi_1q / L11q`,
are compared with the Sauer-Pai subtransient model in closed form (expected: round-off).
Also returns the deviation of PHPS's GENSAL_PHTRUE kernel at the same point (no Sauer-Pai
correction in `dE_q'/dt`, speed-scaled EMF `omega psi''`), which vanishes at equilibria.
`x` supplies the rotor states, `delta`, `p` and the bus-2 voltage; only `Ra = 0` is handled.
"""
function emdq_reduction_check(s::EMDQPilot, x)
    M = s.mach
    M.Ra == 0.0 || throw(ArgumentError("emdq_reduction_check: Ra = 0 only"))
    M.nq == 1 || throw(ArgumentError("emdq_reduction_check: the one-damper q axis (GENSAL) only"))
    wb = M.omega_b
    vm = complex(x[9], x[10]) / s.tap
    xq = copy(x)
    psin = -im * vm
    xq[1], xq[2] = real(psin), imag(psin)
    f = emdq_rhs(s, xq)
    e = emdq_machine(M, xq)
    Lffd, L11q = M.Lfd + M.Lad, M.L1q + M.Laq
    Eqp, psi1d, psiqpp = M.Lad * x[3] / Lffd, x[4], M.Laq * x[5] / L11q
    em = (dEqp = M.Lad * f[3] / Lffd, dpsi1d = f[4], dpsiqpp = M.Laq * f[5] / L11q,
          domega = f[7] / (2.0 * M.H))
    # Sauer-Pai with the stator at nominal speed (generator convention)
    Xl = M.Ll
    Xd, Xq = Xl + M.Lad, Xl + M.Laq
    Xd1 = Xl + M.Lad * M.Lfd / (M.Lad + M.Lfd)
    Xd2 = Xl + 1.0 / (1.0 / M.Lad + 1.0 / M.Lfd + 1.0 / M.L1d)
    Xq2 = Xl + M.Laq * M.L1q / (M.Laq + M.L1q)
    abs(Xd2 - Xq2) < 1e-12 || throw(ArgumentError("subtransient saliency"))
    Td1 = Lffd / (wb * M.Rfd)
    Td2 = (M.L1d + M.Lad * M.Lfd / (M.Lad + M.Lfd)) / (wb * M.R1d)
    Tq2 = L11q / (wb * M.R1q)
    delta = x[6]
    vc = im * conj(_ejd(delta)) * vm
    vd, vq = real(vc), imag(vc)
    kd = (Xd2 - Xl) / (Xd1 - Xl)
    psidpp = kd * Eqp + (1 - kd) * psi1d
    Id = (psidpp - vq) / Xd2
    Iq = (psiqpp + vd) / Xd2
    Efd = M.Lad * (x[8] / s.Cdc) * s.mstar / M.Rfd
    corr = (Xd1 - Xd2) / (Xd1 - Xl)^2 * (psi1d + (Xd1 - Xl) * Id - Eqp)
    sp = (dEqp = (-Eqp - (Xd - Xd1) * (Id - corr) + Efd) / Td1,
          dpsi1d = (-psi1d + Eqp - (Xd1 - Xl) * Id) / Td2,
          dpsiqpp = (-psiqpp - (Xq - Xq2) * Iq) / Tq2,
          domega = (s.Tm - (psidpp * Iq - psiqpp * Id)) / (2.0 * M.H))
    err_sp = maximum(abs(getfield(em, k) - getfield(sp, k)) for k in keys(em))
    # PHPS GENSAL_PHTRUE kernel at the same point (merged leakage added to every reactance;
    # its inputs: network-frame terminal voltage, Tm as power, Efd)
    omega = x[7] / (2.0 * M.H)
    gp = GensalParams(M.H, 0.0, 0.0, Xd, Xq, Xd1, Xd2, Xq2, Td1, Td2, Tq2, Xl, 0.0, 0.0, wb)
    dxp = zeros(5)
    _step!(dxp, nothing, GENSAL_PHTRUE("pilot", 0, gp, ParamDict()), [delta, omega, Eqp, psi1d, psiqpp],
           [real(vm), imag(vm), s.Tm * omega, Efd], gp, NoModes())
    ph = (dEqp = dxp[3], dpsi1d = dxp[4], dpsiqpp = dxp[5], domega = dxp[2])
    dev_phtrue = Dict(string(k) => getfield(ph, k) - getfield(sp, k) for k in keys(sp))
    return (; err_sauer_pai = err_sp, emdq = em, sauer_pai = sp, phtrue = ph, dev_phtrue,
            correction = corr, omega)
end

# ---------------------------------------------------------------------------------------
# Gate 1.3 follow-up (TODO.md, "Method decision"): the periodic speed-angle augmentation of
# Jiang (arXiv:2604.25738, Prop. 2-3), re-derived for the pilot's rotor-angle coordinates.
# A certificate correction from the swing dynamics, not physical stored energy.

"""
    emdq_angle_augmentation(s, xs, x; sigma, c) -> Z

`Z = sigma (p - p*) sin(delta - delta*) + c (1 - cos(delta - delta*))`, the pilot's analogue
of Jiang's `sigma J (omega - omega*) sin(theta) + (sigma D - J omega*^2)(1 - cos(theta))`
(`p - p* = 2H (omega - 1)`; the kinetic energy `p^2/(4H)` has no angle curvature, so the
`J omega*^2` part of his angle coefficient has no counterpart here).
"""
function emdq_angle_augmentation(s::EMDQPilot, xs, x; sigma::Real, c::Real)
    d = x[6] - xs[6]
    return sigma * (x[7] - xs[7]) * sin(d) + c * (1 - cos(d))
end

"""
    emdq_angle_augmentation_rate(s, xs, x; sigma, c) -> dZ/dt

Closed form along the pilot flow: `sigma sin(Ddelta) p' + (sigma (p - p*) cos(Ddelta) +
c sin(Ddelta)) omega_b (omega - 1)`, with `p' = Tm + Tem - Dm (omega - 1)`. Its quadratic part
at the equilibrium is `sigma Ddelta (DTem - Dm Domega) + sigma 2H omega_b Domega^2 +
c omega_b Domega Ddelta`: the angle dissipation comes from `sigma Ddelta DTem` (synchronising
torque), at the price of `sigma 2H omega_b Domega^2`, which only shaft damping can dominate.
"""
function emdq_angle_augmentation_rate(s::EMDQPilot, xs, x; sigma::Real, c::Real)
    M = s.mach
    e = emdq_machine(M, x)
    omega = x[7] / (2.0 * M.H)
    pdot = s.Tm + e.Tem - s.Dm * (omega - 1.0)
    d = x[6] - xs[6]
    return sigma * sin(d) * pdot +
           (sigma * (x[7] - xs[7]) * cos(d) + c * sin(d)) * M.omega_b * (omega - 1.0)
end

"""
    emdq_augmented_forms(s, xs, F, sigma) -> NamedTuple

Local test of `V = H_s + Z` at the equilibrium for one `sigma`, with `c` fixed by the exact
balance (it cancels the angle-speed entry of the rate form, as Jiang's `sigma D` term does;
`c = sigma Dm / omega_b` when the shifted rate has no such entry). `F` from
`emdq_local_forms`. Returns `c`, `HV` (Hessian of V), `QV` (rate form), `min_HV` and the
Finsler value `finsler` (max of `QV` on `ker y_m`).
"""
function emdq_augmented_forms(s::EMDQPilot, xs, F, sigma::Real)
    hz(sg, cc) = ForwardDiff.hessian(z -> emdq_angle_augmentation(s, xs, z; sigma = sg, c = cc), xs)
    qz(sg, cc) = ForwardDiff.hessian(z -> emdq_angle_augmentation_rate(s, xs, z; sigma = sg, c = cc), xs) / 2
    i, j = 6, 7
    Qs, Qc = qz(1.0, 0.0), qz(0.0, 1.0)
    c = -(F.Q[i, j] + sigma * Qs[i, j]) / Qc[i, j]
    HV = F.Hs + hz(sigma, c)
    QV = F.Q + sigma * Qs + c * Qc
    return (; sigma, c, HV, QV, min_HV = eigmin(Symmetric(HV)),
            finsler = restricted_max_eig(QV, F.c))
end
