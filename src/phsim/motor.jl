# Dynamic, voltage-dependent loads (upgrade 2 of 3, REALTIME_TODO section 3; review 74).
#
# 1. The three-phase induction motor `PHMotor` in port-Hamiltonian form: a stator and two rotor cages sharing the
#    magnetising inductance Lm, each a complex flux pair in the network's frame (the rotor is symmetric, so it needs no
#    angle), and the momentum p. Energy coordinates as the machines: lambda = psi / omega_b, i = omega_b K lambda,
#        H = omega_b lambda^H K lambda / 2 + p^2 / (4 Hm),   omega = p / (2 Hm),
#    K the inverse of [Ll+Lm Lm Lm; Lm L1+Lm Lm; Lm Lm L2+Lm]. Synchronous frame, motor convention (i into the motor):
#        lambda_s' = v - Rs i_s - j omega_b lambda_s,
#        lambda_k' = -R_k i_k - j omega_b (1 - omega) lambda_k     (k = 1, 2),
#        p'        = Te - T_L(omega),   Te = omega_b sum_k Im(conj(i_k) lambda_k);
#    dH/dt = Re(v conj(i_s)) - Rs |i_s|^2 - sum_k R_k |i_k|^2 - omega T_L (the -j omega_b terms are lossless,
#    conj(i)^T lambda being real; j omega_b omega lambda_k and Te form a skew pair). Stationary frame: no -j omega_b
#    terms, the rotor's term + j omega_b omega lambda_k. Load torque, T_L omega >= 0 for every omega: Etrq = 2 (fans,
#    pumps) T_L = Tm0 omega |omega|; Etrq = 0 (compressors) T_L = Tm0 omega / sqrt(omega^2 + eps^2).
#    Data: the WECC composite load model's three-phase motors A, B, C (Ls, Lp, Lpp, Tpo, Tppo on the motor base, the
#    stator leakage Ll = 0.8 Lpp), converted to the double-cage circuit by the standard transient / subtransient
#    definitions (`motor_circuit`); the motor base S_m = P_m / LFm.
# 2. The static part: PowerFactory's general-load law of the case (kpv, t1, udmin, udmax) as a resistor modulated by the
#    lag only (review 74c): G(z) = P0 k(Vf) vr^kpv / Vf^2 >= 0, Vf = Vini vr, vr = 1 + z / Vini, z' = (|v| - Vini - z) / t1
#    (`static_conductance`); the lag z a zero-energy signal state. Instantaneously a linear conductance (incrementally
#    passive: the 74b form, G from the instantaneous |v|, was constant power above 1 / t1 and destabilised the network's
#    fast modes); in steady state (|v| = Vf) P = G |v|^2 = P0 k(V) (V / Vini)^kpv, the case's law.

"""WECC three-phase motor data on its own base (pu, s): Rs, Ls, Lp, Lpp, Tpo, Tppo, H, Etrq, Ll."""
struct MotorData
    Rs::Float64
    Ls::Float64
    Lp::Float64
    Lpp::Float64
    Tpo::Float64
    Tppo::Float64
    H::Float64
    etrq::Int
    Ll::Float64
end

# Ma, Wang et al., "Mathematical representation of the WECC composite load model", arXiv 1902.08866, Table III (from
# PSERC S-60, 2017); Ll = 0.8 Lpp (WECC Composite Load Model Specification, April 2021, p. 10)
const WECC_MOTORS = Dict(
    :A => MotorData(0.04, 1.8, 0.10, 0.083, 0.092, 0.0020, 0.05, 0, 0.8 * 0.083),
    :B => MotorData(0.03, 1.8, 0.16, 0.120, 0.100, 0.0026, 1.00, 2, 0.8 * 0.120),
    :C => MotorData(0.03, 1.8, 0.16, 0.120, 0.100, 0.0026, 0.10, 2, 0.8 * 0.120))

"""A motor's double-cage circuit on the system base: resistances, the 3 x 3 inductance matrix and its inverse,
inertia, torque law (Etrq, Tm0, eps) and its base S_m (system pu)."""
struct MotorCircuit
    Rs::Float64
    R1::Float64
    R2::Float64
    L::Matrix{Float64}
    K::Matrix{Float64}
    H::Float64
    etrq::Int
    Tm0::Float64
    eps::Float64
    Sm::Float64
end

"""
    motor_circuit(d, Sm; omega_b, Tm0 = 0.0, eps = 1e-3) -> MotorCircuit

The double-cage circuit of WECC data `d` (motor base) on the system base, S_m the motor base in system pu:
Lm = Ls - Ll, L1 = Lm (Lp - Ll) / (Ls - Lp), L2 = 1 / (1/(Lpp - Ll) - 1/Lm - 1/L1), R1 = (Lm + L1) / (omega_b Tpo),
R2 = (L2 + Lm L1 / (Lm + L1)) / (omega_b Tppo); impedances / S_m, inertia and torque x S_m.
"""
function motor_circuit(d::MotorData, Sm::Real; omega_b::Real, Tm0::Real = 0.0, eps::Real = 1e-3)
    Ll, Lm = d.Ll, d.Ls - d.Ll
    L1 = Lm * (d.Lp - Ll) / (d.Ls - d.Lp)
    L2 = 1 / (1 / (d.Lpp - Ll) - 1 / Lm - 1 / L1)
    (Lm > 0 && L1 > 0 && L2 > 0) || throw(ArgumentError("motor_circuit: no passive double cage (Lm $Lm, L1 $L1, L2 $L2)"))
    R1 = (Lm + L1) / (omega_b * d.Tpo)
    R2 = (L2 + Lm * L1 / (Lm + L1)) / (omega_b * d.Tppo)
    L = [Ll+Lm Lm Lm; Lm L1+Lm Lm; Lm Lm L2+Lm] ./ Sm
    return MotorCircuit(d.Rs / Sm, R1 / Sm, R2 / Sm, L, inv(L), d.H * Sm, d.etrq, Tm0, eps, Sm)
end

with_torque(c::MotorCircuit, Tm0) = MotorCircuit(c.Rs, c.R1, c.R2, c.L, c.K, c.H, c.etrq, Tm0, c.eps, c.Sm)

"""The load torque at speed omega (T_L omega >= 0 for every omega)."""
@inline motor_load_torque(c::MotorCircuit, omega) =
    c.etrq == 2 ? c.Tm0 * omega * abs(omega) : c.Tm0 * omega / sqrt(omega^2 + c.eps^2)

"""Currents (i_s, i_1, i_2) from the fluxes (lambda_s, lambda_1, lambda_2): i = omega_b K lambda."""
@inline function motor_currents(c::MotorCircuit, ls, l1, l2, omega_b)
    K = c.K
    return (omega_b * (K[1, 1] * ls + K[1, 2] * l1 + K[1, 3] * l2),
            omega_b * (K[2, 1] * ls + K[2, 2] * l1 + K[2, 3] * l2),
            omega_b * (K[3, 1] * ls + K[3, 2] * l1 + K[3, 3] * l2))
end

"""
    motor_rates(c, ls, l1, l2, p, v; omega_b, stationary = false) -> (dls, dl1, dl2, dp, (; is, i1, i2, omega, Te, TL))

The motor's equations (file header) at fluxes, momentum and terminal voltage v (network frame).
"""
@inline function motor_rates(c::MotorCircuit, ls, l1, l2, p, v; omega_b, stationary::Bool = false)
    is, i1, i2 = motor_currents(c, ls, l1, l2, omega_b)
    omega = p / (2 * c.H)
    Te = omega_b * (imag(conj(i1) * l1) + imag(conj(i2) * l2))
    TL = motor_load_torque(c, omega)
    if stationary
        dls = v - c.Rs * is
        dl1 = -c.R1 * i1 + im * omega_b * omega * l1
        dl2 = -c.R2 * i2 + im * omega_b * omega * l2
    else
        dls = v - c.Rs * is - im * omega_b * ls
        dl1 = -c.R1 * i1 - im * omega_b * (1 - omega) * l1
        dl2 = -c.R2 * i2 - im * omega_b * (1 - omega) * l2
    end
    return dls, dl1, dl2, Te - TL, (; is, i1, i2, omega, Te, TL)
end

"""Stored energy: magnetic omega_b lambda^H K lambda / 2 (= Re(conj(lambda) . i) / 2) and kinetic p^2 / (4 H)."""
function motor_energy(c::MotorCircuit, ls, l1, l2, p, omega_b)
    is, i1, i2 = motor_currents(c, ls, l1, l2, omega_b)
    return (real(conj(ls) * is) + real(conj(l1) * i1) + real(conj(l2) * i2)) / 2 + p^2 / (4 * c.H)
end

"""
    motor_steady(c, v, slip; omega_b) -> (; is, i1, i2, ls, l1, l2, p, omega, Te, P, Q)

The constant-phasor steady state (synchronous frame) at terminal voltage v and slip: the circuit's equations with all
rates zero, solved for the currents.
"""
function motor_steady(c::MotorCircuit, v::Number, slip::Real; omega_b::Real)
    L, s = c.L, slip
    A = ComplexF64[c.Rs+im*L[1, 1] im*L[1, 2] im*L[1, 3];
                   im*s*L[2, 1] c.R1+im*s*L[2, 2] im*s*L[2, 3];
                   im*s*L[3, 1] im*s*L[3, 2] c.R2+im*s*L[3, 3]]
    i = A \ ComplexF64[v, 0, 0]
    lam = (L * i) ./ omega_b
    omega = 1 - s
    Te = omega_b * (imag(conj(i[2]) * lam[2]) + imag(conj(i[3]) * lam[3]))
    S = v * conj(i[1])
    return (; is = i[1], i1 = i[2], i2 = i[3], ls = lam[1], l1 = lam[2], l2 = lam[3], p = 2 * c.H * omega, omega, Te,
            P = real(S), Q = imag(S))
end

"""
    motor_slip(c, v, P; omega_b) -> slip

The slip at which the motor draws the electrical power P at terminal voltage v: the root on the stable (small-slip)
branch, below the slip of the largest torque; bisection to the last bit.
"""
function motor_slip(c::MotorCircuit, v::Number, P::Real; omega_b::Real)
    Pof(s) = motor_steady(c, v, s; omega_b).P
    Tof(s) = motor_steady(c, v, s; omega_b).Te
    # the slip of the largest torque (the stable branch's end), on a log grid then golden section
    grid = exp10.(range(-5, 0, length = 400))
    k = argmax(Tof.(grid))
    a, b = grid[max(k - 1, 1)], grid[min(k + 1, end)]
    for _ in 1:100
        m1, m2 = a + 0.382 * (b - a), a + 0.618 * (b - a)
        Tof(m1) < Tof(m2) ? (a = m1) : (b = m2)
    end
    smax = (a + b) / 2
    Pof(smax) > P || throw(ArgumentError("motor_slip: P = $P exceeds the motor's largest power at |v| = $(abs(v))"))
    lo, hi = 0.0, smax
    for _ in 1:200
        mid = (lo + hi) / 2
        (mid == lo || mid == hi) && break
        Pof(mid) < P ? (lo = mid) : (hi = mid)
    end
    return (lo + hi) / 2
end

# ---- the static part: PowerFactory's general-load law (complexload.jl) as a conductance ----

"""Static part of a load: P0 (at the reference Vini) with exponent kpv, lag t1 (0: none), band udmin / udmax."""
struct StaticLaw
    P0::Float64
    Vini::Float64
    kpv::Float64
    t1::Float64
    udmin::Float64
    udmax::Float64
end

# PowerFactory's out-of-range multiplier k(r) divided by r^2 and its derivative's pieces: returns (k(r) / r^2,
# d(k(r) / r) / dr) for r >= 0 (no division by r below udmin / 2, where k = 2 r^2 / udmin^2)
@inline function _kv_over_r2(L::StaticLaw, r)
    ud, ux = L.udmin, L.udmax
    if ud > 0 && r < ud / 2
        return 2 / ud^2 + 0 * r, 2 / ud^2 + 0 * r                       # k / r^2;  d(k / r) / dr = d(2 r / ud^2)
    elseif ud > 0 && r <= ud
        d = (r - ud) / ud; k = 1 - 2d^2; dk = -4d / ud
        return k / r^2, (dk * r - k) / r^2
    elseif ux > ud && r >= ux
        k = 1 + (r - ux)^2; dk = 2 * (r - ux)
        return k / r^2, (dk * r - k) / r^2
    else
        return 1 / r^2, -1 / r^2
    end
end

"""
    static_conductance(L, z) -> G

The static part's conductance at the lag state z: G = P0 k(Vf) vr^kpv / Vf^2 with vr = max(1 + z / Vini, 1e-6) (the
case's guard) and Vf = Vini vr; k / Vf^2 = 2 / udmin^2 below udmin / 2 (no division there).
"""
@inline function static_conductance(L::StaticLaw, z)
    vr = max(1 + z / L.Vini, 1e-6)
    return L.P0 * vr^L.kpv * _kv_over_r2(L, L.Vini * vr)[1]
end

"""The lag's rate z' = (r - Vini - z) / t1."""
@inline static_lag_rate(L::StaticLaw, r, z) = (r - L.Vini - z) / L.t1

# ---- in the system (system.jl; `loads = :composite`) ----

"""
    PHMotor

One induction motor of a composite load: its type `kind` (:A, :B, :C), bus row `node`, bus id `bus`, circuit `c`
(system base, with Tm0), the share `F` of the bus's P, its electrical power `Pm` and slip `slip0` at the load flow's
|V|, and the first of its 7 states `offset` (lambda_s, lambda_1, lambda_2 as Re / Im pairs, then p).
"""
struct PHMotor
    name::String
    kind::Symbol
    node::Int
    bus::Int
    c::MotorCircuit
    F::Float64
    Pm::Float64
    slip0::Float64
    offset::Int
    relay::Any          # its undervoltage relay (`MotorRelay`), or nothing (relays are a case-study option, review 76)
    ridx::Int           # the relay's first state (sigma, n1, n2, d1, d2), 0 without a relay
end

# ---- undervoltage relays of the motors (review 76): an optional case study, off by default ----
#
# A motor is an aggregate of identical units; tripping a fraction f leaves (1 - f) of them with the same flux linkages per
# unit and the same speed. The scale sigma (1 at the start) multiplies the currents, the inertia, the load torque and the
# magnetic energy: i = sigma omega_b K lambda, p = 2 sigma H omega, H = sigma (magnetic) + p^2 / (4 sigma H); the flux
# equations do not depend on sigma. A trip sets sigma <- max(0, sigma - f) and p <- p sigma_new / sigma_old (the speed
# kept); the tripped units leave the system with their stored energy. The relay is sampled (period Tr, a numerical
# relay): two levels as the WECC composite load model's (Vtr_i, Ttr_i, Ftr_i; cumulative, once; no reconnection).

"""Two-level undervoltage relay of a motor: level i operates when |v| < vtr_i for ntr_i consecutive samples of Tr."""
struct MotorRelay
    vtr1::Float64
    ftr1::Float64
    n1::Int
    vtr2::Float64
    ftr2::Float64
    n2::Int
    Tr::Float64
end

function MotorRelay(; vtr1 = 0.8, ttr1 = 2.0, ftr1 = 1.0, vtr2 = 0.6, ttr2 = 0.16, ftr2 = 1.0, Tr = 50e-6, kw...)
    # the delays must be whole numbers of samples (review 76, item 2: rounding could shorten a delay)
    for t in (ttr1, ttr2)
        abs(t / Tr - round(t / Tr)) <= 1e-9 * max(1.0, t / Tr) && round(Int, t / Tr) >= 1 ||
            throw(ArgumentError("MotorRelay: the delay $t s is not a whole number of samples of $Tr s"))
    end
    (0 <= ftr1 <= 1 && 0 <= ftr2 <= 1) || throw(ArgumentError("MotorRelay: trip fractions $ftr1, $ftr2 outside [0, 1]"))
    return MotorRelay(vtr1, ftr1, round(Int, ttr1 / Tr), vtr2, ftr2, round(Int, ttr2 / Tr), Tr)
end

"""
    relay_sample!(x, mo, V) -> (sigma_old, sigma_new)

One relay sample of motor `mo` at bus voltage magnitude V: count, operate, rescale the momentum. The simulator calls it
after every step ending on the sampling grid, for every relay from the same voltages of that step's end (the snapshot
before any trip, and before the converters' held outputs change at the same call); a step ending at an event (a fault's
start or end, on the grid) is sampled with that step's inputs, before the event. The tripped units' stored energy
leaves with them: E_removed = (sigma_old - sigma_new) / sigma_old x the motor's energy (clipped trips included).
"""
function relay_sample!(x, mo::PHMotor, V::Real)
    r, i = mo.relay, mo.ridx
    s0 = x[i]
    s1 = s0
    for (lv, vtr, f, nmax) in ((1, r.vtr1, r.ftr1, r.n1), (2, r.vtr2, r.ftr2, r.n2))
        x[i+2+lv] == 0 || continue                     # this level has operated (once only)
        x[i+lv] = V < vtr ? x[i+lv] + 1 : 0.0
        if x[i+lv] >= nmax
            s1 = max(0.0, s1 - f)
            x[i+2+lv] = 1.0
        end
    end
    if s1 != s0
        x[i] = s1
        x[mo.offset+6] = s0 > 0 ? x[mo.offset+6] * (s1 / s0) : 0.0
    end
    return s0, s1
end

_relay_names(mo::PHMotor) = mo.name .* "." .* ["sigma", "n1", "n2", "d1", "d2"]

"""
    PHStatic

The static part of a composite load at bus row `node`: the case's P law `law` (P0 = the static share at Vini = the
load flow's |V|), its fixed reactive element `Qs` (an inductor or a capacitor in the network data) and the lag's state
`zidx`.
"""
struct PHStatic
    name::String
    node::Int
    bus::Int
    law::StaticLaw
    Qs::Float64
    zidx::Int
end

phs_nstates(::PHMotor) = 7

"""|v| with zero partials at v = 0 (ForwardDiff): sqrt(abs2(v)) has none there."""
@inline _vabs(v) = (a = abs2(v); iszero(a) ? zero(a) : sqrt(a))

# the value of a (possibly nested) dual number
_fval(x) = x isa ForwardDiff.Dual ? _fval(ForwardDiff.value(x)) : x
