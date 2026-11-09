# A transformer with a saturating core on a converter's merged branch (docs/active/PH_CONVERTER_MODEL.md,
# sections 19-23; PH_SIMULATOR_TODO.md step S8, problem B).
#
# The branch of a converter merged with its step-up transformer (`system.jl`, file header) becomes a T-circuit referred
# to the converter's side: HV bus -- ideal ratio a -- R2, L2 = L_T/2 -- core node -- R1 = R_T/2, L1 = L_c + L_T/2 --
# source row (the EMF u_c). The core node joins three inductive elements (an inductor-only cutset), reduced by
# minimising the energy over the core flux (T1 row 1). In flux units psi = omega_b x (pu), with the code's direction
# (the branch current runs from the HV bus towards the source row): the HV loop flux psi_H = L2 i_H + lambda, the source
# loop flux psi_S = lambda - L1 i_S, the core current i_m = i_H - i_S, and
#   h(psi_H, psi_S, theta) = min_lambda [ |psi_H - lambda|^2/(2 L2) + |lambda - psi_S|^2/(2 L1) + W(lambda, theta) ],
#   W(lambda, theta) = (2/3) sum_k w(lambda_k),  lambda_k = Re(lambda e^{j(theta - 2 pi k/3)}),  w' = f,
#   i_H = (psi_H - lambda)/L2 = dh/dpsi_H,  i_S = (lambda - psi_S)/L1 = -dh/dpsi_S,  H = h / omega_b,
# with f the per-phase current-flux curve (odd, piecewise linear) and theta the frame clock (theta' = omega_b). The rows
# (states x = psi / omega_b, the frame turning at omega_b):
#   x_H' = v_HV / a - R2 i_H - j omega_b x_H,      x_S' = u_c + R1 i_S - j omega_b x_S.
# Per-phase saturation is evaluated on the zero-sequence-free flux and the magnetising current is projected on the
# balanced subspace (T1 row 3): (2/3) sum_k f(lambda_k) e^{-j(theta - 2 pi k/3)}; the zero-sequence part a balanced
# model cannot carry is dropped. No core loss (no source gives one; PowerFactory's `pfe` is 0): a selected scope.
#
# The step (record section 23): the energy depends on the stationary-frame fluxes only, so with beta = (theta1 -
# theta0)/2 and the rotated ends e^{-j beta} x0, e^{j beta} x1 the energy difference over a step is that of one function
# at frozen theta_m; the rows are written with that exact rotation and the currents are Gonzalez's discrete gradient,
# its numerator taken on the non-linear part of the energy only (exactly zero in the curve's linear part).

"""
    PHCoreCurve

The per-phase current-flux curve of a core on the system base: flux knots `lam` (peak, pu of the winding's rated flux,
starting at 0), current knots `cur` (peak, system base), the slope `di/dlambda` of each segment (`sl`, the last extended
beyond the last knot), the energy `wk` at each knot, `Lm` the linear inductance of the first segment and `wnk` the
non-linear energy (`w - lambda^2/(2 Lm)`) at each knot from the second on. `source` says where the data come from.
"""
struct PHCoreCurve
    lam::Vector{Float64}
    cur::Vector{Float64}
    sl::Vector{Float64}
    wk::Vector{Float64}
    Lm::Float64
    wnk::Vector{Float64}
    source::String
end

"""
    PHCoreCurve(lam, cur, slope_end, scale; source) -> PHCoreCurve

From knots on the transformer's own base (`lam` and `cur` from zero, increasing, positive slopes; `slope_end` the
`di/dlambda` beyond the last knot) to the system base: currents times `scale = S_T / S_b` (the flux base is unchanged:
the winding's rated voltage is the bus's base voltage). Odd extension below zero.
"""
function PHCoreCurve(lam::AbstractVector{<:Real}, cur::AbstractVector{<:Real}, slope_end::Real, scale::Real;
                     source::AbstractString)
    (length(lam) == length(cur) >= 3 && lam[1] == 0 && cur[1] == 0 && all(diff(lam) .> 0) && all(diff(cur) .> 0) &&
     all(isfinite, lam) && all(isfinite, cur) && isfinite(slope_end) && isfinite(scale) && slope_end > 0 && scale > 0) ||
        throw(ArgumentError("PHCoreCurve: knots outside their domain"))
    Λ, I = collect(Float64, lam), scale .* collect(Float64, cur)
    sl = vcat(diff(I) ./ diff(Λ), scale * slope_end)
    n = length(Λ)
    wk = zeros(n)
    for k in 1:n-1
        wk[k+1] = wk[k] + (I[k] + I[k+1]) / 2 * (Λ[k+1] - Λ[k])
    end
    Lm = Λ[2] / I[2]
    # the non-linear part g(s) = f(s) - s/Lm is zero on the first segment; its integral from lam[2]
    wnk = zeros(n)
    g(k, s) = (I[k] - Λ[k] / Lm) + (sl[k] - 1 / Lm) * (s - Λ[k])
    for k in 2:n-1
        wnk[k+1] = wnk[k] + (g(k, Λ[k]) + g(k, Λ[k+1])) / 2 * (Λ[k+1] - Λ[k])
    end
    return PHCoreCurve(Λ, I, sl, wk, Lm, wnk, String(source))
end

"""
    core_curve_gmd(scale) -> PHCoreCurve

The measured curve of the GMD test case (Haddadi et al., IEEE Trans. EMC 2020, doi 10.1109/TEMC.2019.2920271, Table IV,
p. 958; a 300 MVA single-phase shell-form unit's field test, used there for every transformer): current 0.001, 0.01,
0.025, 0.05, 0.1, 0.5, 1, 2 pu at flux 1, 1.075, 1.15, 1.2, 1.23, 1.37, 1.55, 1.86 pu, the fully saturated slope 0.31 pu
(`dlambda/di`) beyond. Taken as peak flux against peak current (the table does not say: an assumption, record section
20), on the transformer's rating, scaled to the system base by `scale = S_T / S_b`.
"""
core_curve_gmd(scale::Real) =
    PHCoreCurve([0, 1, 1.075, 1.15, 1.2, 1.23, 1.37, 1.55, 1.86], [0, 0.001, 0.01, 0.025, 0.05, 0.1, 0.5, 1, 2],
                1 / 0.31, scale; source = "Haddadi et al. 2020, Table IV (GMD test case), peak against peak")

# the primal value of a number, through nested dual numbers
_pv(x::Real) = x
_pv(x::ForwardDiff.Dual) = _pv(ForwardDiff.value(x))
_pv(z::Complex) = complex(_pv(real(z)), _pv(imag(z)))

# the segment of the flux `l` (by its magnitude; 1 to n, the last extends beyond the last knot)
@inline _seg(c::PHCoreCurve, l) = min(searchsortedlast(c.lam, abs(_pv(l))), length(c.lam))

"""    core_f(c, l): the per-phase magnetising current at flux `l` (odd)."""
@inline function core_f(c::PHCoreCurve, l)
    a = abs(l)
    k = _seg(c, a)
    k == 1 && return c.sl[1] * l                 # linear through the origin (its derivative kept at l = 0: T3 1c)
    return sign(_pv(l)) * (c.cur[k] + c.sl[k] * (a - c.lam[k]))
end
"""    core_fp(c, l): the slope `di/dlambda` at `l` (the segment's; right-continuous in |l|)."""
@inline core_fp(c::PHCoreCurve, l) = c.sl[_seg(c, abs(l))]
"""    core_w(c, l): the per-phase energy `w(l) = integral_0^|l| f` (pu flux times pu current)."""
@inline function core_w(c::PHCoreCurve, l)
    a = abs(l)
    k = _seg(c, a)
    d = a - c.lam[k]
    return c.wk[k] + c.cur[k] * d + c.sl[k] * d^2 / 2
end
"""    core_wnl(c, l): `w(l) - l^2/(2 Lm)`, in closed form per segment (exactly zero on the first)."""
@inline function core_wnl(c::PHCoreCurve, l)
    a = abs(l)
    k = _seg(c, a)
    k == 1 && return zero(a)
    d = a - c.lam[k]
    return c.wnk[k] + (c.cur[k] - c.lam[k] / c.Lm) * d + (c.sl[k] - 1 / c.Lm) * d^2 / 2
end

"""
    PHCore

A saturating core on a converter's merged branch (file header): the branch name, the converter's bus id, the branch's
position `pos` in the system's `branches` (its state pair is the HV loop flux `x_H`), the converter's index `conv`, the
first index `xs` of the source loop flux `x_S`, the T-circuit `L1, R1, L2, R2` (system base, `L1` holding the converter's
`Lc`), the transformer's total rating `S` (MVA, all parallel units) and the curve.
"""
struct PHCore
    name::String
    bus::Int
    pos::Int
    conv::Int
    xs::Int
    L1::Float64
    R1::Float64
    L2::Float64
    R2::Float64
    Lc::Float64
    S::Float64
    curve::PHCoreCurve
end

const _CORE_PHASES = (0.0, -2pi / 3, 2pi / 3)      # phase k at theta + this (phs_phases' a, b, c)

# the magnetising current (complex, projected) and the per-phase fluxes of core flux `lam` at clock `th`
@inline function _core_im(c::PHCoreCurve, lam, th)
    i = zero(lam)
    for p in _CORE_PHASES
        e = cis(th + p)
        i += core_f(c, real(lam * e)) * conj(e)
    end
    return 2 * i / 3
end

@inline _core_W(c::PHCoreCurve, lam, th) = 2 * sum(core_w(c, real(lam * cis(th + p))) for p in _CORE_PHASES) / 3
@inline _core_Wnl(c::PHCoreCurve, lam, th) = 2 * sum(core_wnl(c, real(lam * cis(th + p))) for p in _CORE_PHASES) / 3

# the core's linear part: every phase flux within the first segment
@inline _core_linear(c::PHCoreCurve, lam, th) = all(abs(real(lam * cis(th + p))) <= c.lam[2] for p in _CORE_PHASES)

# the residual of the minimisation, r = dPhi/dlambda (complex), and its 2x2 Hessian K (real)
function _core_rK(q::PHCore, lam, psH, psS, th)
    c = q.curve
    r = (lam - psH) / q.L2 + (lam - psS) / q.L1 + _core_im(c, lam, th)
    ks = 1 / q.L1 + 1 / q.L2
    K11, K12, K22 = ks, zero(ks), ks
    for p in _CORE_PHASES
        e = cis(th + p)
        s = 2 * core_fp(c, real(lam * e)) / 3
        ek1, ek2 = real(e), -imag(e)                   # lambda_k = ek1 Re(lambda) + ek2 Im(lambda)
        K11 += s * ek1^2; K12 += s * ek1 * ek2; K22 += s * ek2^2
    end
    return r, K11, K12, K22
end

@inline _core_Phi(q::PHCore, lam, psH, psS, th) =
    abs2(psH - lam) / (2 * q.L2) + abs2(lam - psS) / (2 * q.L1) + _core_W(q.curve, lam, th)

"""
    core_solve(q, psH, psS, th) -> (; lam, iH, iS, laml)

The core flux minimising the branch's energy at the loop fluxes `psH`, `psS` (flux units, `psi = omega_b x`) and the
clock `th`, with the branch currents and the linear minimiser `laml`. In the curve's linear part (every phase flux of the
linear minimiser within the first knot) the linear minimiser itself, exactly. Otherwise a damped semismooth Newton on the
strictly convex objective (T1 row 6), exact once the segments are found; with dual numbers the derivative comes from
one Newton correction at the converged primal solution (the implicit function: K dlam = -dr). That gives first
derivatives; through nested dual numbers the mixed second derivatives are not those of the implicit solve (no supported
path takes them: T3 1b).
"""
function core_solve(q::PHCore, psH, psS, th)
    c = q.curve
    kap = 1 / q.L1 + 1 / q.L2 + 1 / c.Lm
    laml = (psH / q.L2 + psS / q.L1) / kap
    if _core_linear(c, _pv(laml), _pv(th))
        return (; lam = laml, iH = (psH - laml) / q.L2, iS = (laml - psS) / q.L1, laml)
    end
    pH, pS, pt = _pv(psH), _pv(psS), _pv(th)
    lam = _pv(laml)
    scale = abs(pH) / q.L2 + abs(pS) / q.L1 + 1e-300
    for it in 1:60
        r, K11, K12, K22 = _core_rK(q, lam, pH, pS, pt)
        det = K11 * K22 - K12^2
        d1 = -(K22 * real(r) - K12 * imag(r)) / det
        d2 = -(-K12 * real(r) + K11 * imag(r)) / det
        dl = complex(d1, d2)
        abs(r) <= 1e-15 * scale && abs(dl) <= 1e-15 * (abs(lam) + 1) && break
        # backtracking on the objective (convex), then the full step when it does not increase it
        P0, t = _core_Phi(q, lam, pH, pS, pt), 1.0
        slope = real(r) * d1 + imag(r) * d2              # < 0
        while _core_Phi(q, lam + t * dl, pH, pS, pt) > P0 + 1e-4 * t * slope + 1e-15 * abs(P0) && t > 1e-10
            t /= 2
        end
        lam += t * dl
        it == 60 && error("core_solve: no convergence at |psi_H| = $(abs(pH)), |psi_S| = $(abs(pS)) (|r| = $(abs(r)))")
    end
    lamT = if laml isa Complex{Float64} && th isa Float64
        lam
    else                                                  # the implicit derivative: one correction with duals
        r, K11, K12, K22 = _core_rK(q, lam, psH, psS, th)
        rv = _core_rK(q, lam, pH, pS, pt)[1]
        dr = r - rv                                       # the dual part only (the primal residual is converged)
        Kp = [_pv(K11) _pv(K12); _pv(K12) _pv(K22)]
        det = Kp[1, 1] * Kp[2, 2] - Kp[1, 2]^2
        lam + complex(-(Kp[2, 2] * real(dr) - Kp[1, 2] * imag(dr)) / det,
                      -(-Kp[1, 2] * real(dr) + Kp[1, 1] * imag(dr)) / det)
    end
    return (; lam = lamT, iH = (psH - lamT) / q.L2, iS = (lamT - psS) / q.L1, laml)
end

"""    core_energy(q, psH, psS, th) -> h (flux units; the stored energy is h / omega_b)."""
function core_energy(q::PHCore, psH, psS, th)
    s = core_solve(q, psH, psS, th)
    return abs2(psH - s.lam) / (2 * q.L2) + abs2(s.lam - psS) / (2 * q.L1) + _core_W(q.curve, s.lam, th)
end

# the non-linear part of the reduced energy and its gradient in (psH, psS) (record section 23):
# h_nl = kappa |lam - laml|^2 / 2 + W_nl(lam), grad h_nl = ((laml - lam)/L2, (laml - lam)/L1)
function _core_hnl(q::PHCore, psH, psS, th)
    s = core_solve(q, psH, psS, th)
    kap = 1 / q.L1 + 1 / q.L2 + 1 / q.curve.Lm
    d = s.lam - s.laml
    return (; hnl = kap * abs2(d) / 2 + _core_Wnl(q.curve, s.lam, th), gH = -d / q.L2, gS = -d / q.L1, s)
end

"""
    core_dgrad(q, H0, S0, H1, S1, thm) -> (iH, iS)

Gonzalez's discrete gradient of `h(., thm)` between the (rotated) end points `(H0, S0)` and `(H1, S1)` (flux units),
returned as the branch currents it stands for (`(iH, -iS)` is the gradient): the gradient at the midpoint plus the
correction along the increment, its numerator on the non-linear part only (exactly zero in the linear part). With no
measurable increment (below 1e-13 of the ends' size) the midpoint gradient: there the discrete balance holds to that
tolerance instead of exactly (T3 1d).
"""
function core_dgrad(q::PHCore, H0, S0, H1, S1, thm)
    m = _core_hnl(q, (H0 + H1) / 2, (S0 + S1) / 2, thm)
    dH, dS = H1 - H0, S1 - S0
    n2 = abs2(dH) + abs2(dS)
    gH, gS = m.s.iH, -m.s.iS
    if _pv(n2) > (1e-13 * (abs(_pv(H0)) + abs(_pv(S0)) + abs(_pv(H1)) + abs(_pv(S1))))^2
        e0, e1 = _core_hnl(q, H0, S0, thm), _core_hnl(q, H1, S1, thm)
        num = (e1.hnl - e0.hnl) - (real(conj(m.gH) * dH) + real(conj(m.gS) * dS))
        cg = num / n2
        gH += cg * dH
        gS += cg * dS
    end
    return gH, -gS
end

"""
    core_rotation_rate(iH, iS, xH, xS) -> dH/dtheta (x units)

The derivative of the stored energy with respect to the clock at fixed loop fluxes, from the invariance under a common
rotation (record section 23, 23.1): `Re(conj(i_H) j x_H) - Re(conj(i_S) j x_S)`.
"""
core_rotation_rate(iH, iS, xH, xS) = real(conj(iH) * im * xH) - real(conj(iS) * im * xS)
