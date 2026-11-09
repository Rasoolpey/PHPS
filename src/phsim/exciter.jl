# Physical exciters of the port-Hamiltonian grid model (PH_SIMULATOR_TODO.md, package N2;
# docs/active/PH_EXCITER_MODEL.md).
#
# `PHShuntExciter`: a self-excited DC commutator exciter with an ideal armature, on its machine's
# shaft, feeding the machine's field winding (record sections 11.6, 11.7). The device a
# case's IEEET1 with negative `KE` stands for (IEEE Committee Report, IEEE Trans. PAS-87(6),
# 1968, p. 1462); its circuit and its row are those of the IEEE Committee Report, IEEE Trans.
# PAS-100(2), 1981, Appendix A, Fig. A.5 and eq. (A.17), (A.18). All electrical quantities are
# on the machine's field circuit base (`v_f`, `i_fd` of `PHInput`); fluxes in per-unit volts
# times seconds, as the machine's field flux.
#
#   state     phi_f                      the exciter's field flux (its only storage)
#   output    x_E = c phi_f,             c = Lad / (Rfd n TE)
#   armature  v_f = phi_f omega / (n TE)                    generated voltage, ideal armature
#   field     r_F i_f = [1 + n (KE + SE(|x_E|))] phi_f / (n TE),  SE(x) = A exp(B x)
#             phi_f' = v_f + b VR - r_F i_f,                b = n Rfd / Lad
#   shaft     the rotor gives v_f (i_fd + i_f): torque v_f (i_fd + i_f) / omega
#   energy    W_f(phi_f) = integral of i_f dphi_f,  loss r_F i_f^2
#   supplies  the amplifier, b VR i_f (a declared boundary), and the shaft
#
# At omega = 1 the field row is the IEEET1 row `TE x_E' = VR - (KE + SE(x_E)) x_E`, and
# `Efd = v_f Lad / Rfd = omega x_E`.
#
# Signal laws without energy (record section 6): the regulator `KA`, `TA` with its non-windup
# limits (state `xa`; the amplifier's output is `VR = clamp(xa, VRMIN, VRMAX)`), and the
# stabiliser `KF`, `TF` fed by the exciter's output voltage `Efd` (state `xf`).
#
# Two numbers no source gives are parameters without a default (record 11.8, 13.2):
#   `n`     the base of the regulator output over the base of the exciter voltage. It sets how
#           strongly speed acts on the self-excitation and the amplifier's voltage; the steady
#           states and the row at rated speed do not depend on it;
#   `rF`    the shunt field circuit's resistance. It sets the field current, hence the stored
#           energy, the field loss and the field current's share of the shaft torque; the
#           voltages do not depend on it.
# Not in the model: armature resistance and inductance (a declared reduction), armature
# reaction, residual magnetism, hysteresis and coercivity, the amplifier as a device. The case's
# fitted saturation function is used as the field's own curve; since 2026-10-31 (review 74,
# ROA_MATHEMATICAL_FORMULATION.md 12.6.2) `SE` is evaluated at |x_E| (an even function), so the
# field curve `[1 + n (KE + SE(|x_E|))] phi` is odd. Our assumption: the whole lumped
# characteristic is reflected through zero. Read support: the anhysteretic magnetisation of
# iron is odd in the field (Jiles and Atherton, J. Magn. Magn. Mater. 61, 1986, eq. 6, p. 57);
# IEEE Std 421.5-2016's DC1A model has no lower limit on EFD (DC1C reproduces it with
# `EFDmin = -99 pu`, p. 14) and leaves the curve's form to the program (p. 118). Not supported
# by the sources: that `SE`, which also carries load regulation (p. 117), is odd; the transient
# with the machine's field current still positive while `v_f < 0` (power then flows back through
# the ideal armature to the shaft); hysteresis and coercivity near zero flux. Negative flux is
# inside this extended model, not validated physics. The curve is C1 at zero flux (its second
# derivative jumps there). The limiter of `xa` decides by comparison and is
# held over a step (`control.jl`); the clamp of the amplifier's output is a continuous
# function of the state and is not held, so that the vector field and the energy account use
# the same supply.

"""
    PHShuntExciter

A self-excited DC exciter with an ideal armature on its machine's shaft (file header). `KA`,
`TA`, `KE`, `TE`, `KF`, `TF`, `VRMAX`, `VRMIN` and the saturation constants `A`, `B` are the
case's IEEET1 constants; `n` and `rF` are the two numbers no source gives; `c` and `b` are
derived (`x_E = c phi_f`, amplifier volts `b VR`). Local states: `phi_f, xa, xf`.
"""
struct PHShuntExciter
    name::String
    KA::Float64
    TA::Float64
    KE::Float64
    TE::Float64
    KF::Float64
    TF::Float64
    VRMAX::Float64
    VRMIN::Float64
    A::Float64
    B::Float64
    n::Float64
    rF::Float64
    c::Float64
    b::Float64
end

"""
    PHShuntExciter(e::IEEET1_PHTRUE, Rfd, Lad; n, rF)

The physical exciter with the constants of the case's IEEET1 `e`, on a machine with field
resistance `Rfd` and mutual inductance `Lad`. Refused: an input filter (`TR > 0`), a time
constant or gain that is not positive, data that are not finite, a negative saturation
constant, and a field characteristic that is not increasing for every flux
(`1 + n (KE + A) <= 0`: the odd curve's slope is least at zero flux; the refusal asks for a
positive slope everywhere). For a machine that
stands for several identical units, `Rfd` and `Lad` are the combined machine's: `c` and `b` do
not change, and the field current, energy and loss are those of all the exciters together.
"""
function PHShuntExciter(e::IEEET1_PHTRUE, Rfd::Real, Lad::Real; n::Real, rF::Real)
    p = e.p
    (isfinite(p.TR) && abs(p.TR) <= 1e-9) || throw(ArgumentError("PHShuntExciter $(e.name): an input filter (TR > 0) is not modelled"))
    (p.TA > 1e-4 && p.TE > 1e-4 && p.TF > 1e-4 && p.KA > 1e-6) ||
        throw(ArgumentError("PHShuntExciter $(e.name): KA, TA, TE and TF must be positive"))
    all(isfinite, (p.KA, p.TA, p.KE, p.TE, p.KF, p.TF, p.VRMAX, p.VRMIN, p.SAT_A, p.SAT_B, n, rF, Rfd, Lad)) ||
        throw(ArgumentError("PHShuntExciter $(e.name): every parameter must be finite"))
    (n > 0 && rF > 0 && Rfd > 0 && Lad > 0) ||
        throw(ArgumentError("PHShuntExciter $(e.name): n, rF, Rfd and Lad must be positive"))
    (p.SAT_A >= 0 && p.SAT_B >= 0) ||
        throw(ArgumentError("PHShuntExciter $(e.name): the saturation constants must not be negative"))
    p.VRMIN < p.VRMAX || throw(ArgumentError("PHShuntExciter $(e.name): VRMIN must be below VRMAX"))
    # d(r_F i_f)/dphi_f is [1 + n (KE + SE(|x|)(1 + B |x|))] / (n TE) (the curve is odd); with
    # B >= 0 the least value of SE(|x|)(1 + B |x|) is A, at zero flux
    slope = 1 + n * (p.KE + p.SAT_A)
    slope > 0 || throw(ArgumentError("PHShuntExciter $(e.name): with n = $n the field characteristic is " *
                                     "not increasing for every flux"))
    return PHShuntExciter(e.name, p.KA, p.TA, p.KE, p.TE, p.KF, p.TF, p.VRMAX, p.VRMIN, p.SAT_A,
                          p.SAT_B, n, rF, Lad / (Rfd * n * p.TE), n * Rfd / Lad)
end

nstates(::PHShuntExciter) = 3
name(e::PHShuntExciter) = e.name
state_names(::PHShuntExciter) = ["phi_f", "xa", "xf"]

# the exciter's output on the IEEET1 scale, x_E (Efd at rated speed)
@inline _exc_x(e::PHShuntExciter, phi) = e.c * phi
# the generated voltage, which is the machine's field voltage
@inline _exc_vf(e::PHShuntExciter, phi, omega) = phi * omega / (e.n * e.TE)
# the saturation function of the case's IEEET1 at |x| (even; zero when it is disabled: no
# exponential is evaluated then)
@inline _exc_se(e::PHShuntExciter, x) = e.A == 0 ? zero(x) : e.A * exp(e.B * abs(x))
# the voltage the field circuit's resistance takes, r_F i_f: odd in the flux (saturation at |x_E|)
@inline _exc_field_volts(e::PHShuntExciter, phi) =
    (1 + e.n * (e.KE + _exc_se(e, e.c * phi))) * phi / (e.n * e.TE)
# the amplifier's voltage in the field circuit
@inline _exc_amp_volts(e::PHShuntExciter, xa) = e.b * clamp(xa, e.VRMIN, e.VRMAX)

"""The stored energy of the exciter's field, `integral_0^phi i_f dphi` (pu*s); even in `phi`."""
function _exc_energy(e::PHShuntExciter, phi)
    e.A == 0 && return (1 + e.n * e.KE) * phi^2 / (2 * e.n * e.TE * e.rF)
    return ((1 + e.n * e.KE) / 2 + e.n * e.A * _exp_moment(e.B * e.c * abs(phi))) * phi^2 / (e.n * e.TE * e.rF)
end

# (1/x^2) integral_0^x s exp(k s) ds as a function of y = k x: (exp(y)(y - 1) + 1) / y^2. The
# bracket is y exp(y) - expm1(y), which keeps its accuracy down to small y; below 0.01 the
# series sum_k y^k / (k! (k + 2)) is used, truncated beyond y^6 (an error below 1e-18)
@inline _exp_moment(y) = abs(y) < 1e-2 ?
    0.5 + y * (1 / 3 + y * (1 / 8 + y * (1 / 30 + y * (1 / 144 + y * (1 / 840 + y / 5760))))) :
    (y * exp(y) - expm1(y)) / y^2

"""
    _exc_rhs!(dx, e, x, vterm, Vref, omega, ifd, rec) -> dx

The exciter's rows `[phi_f', xa', xf']` at its states `x`, the generator-bus voltage magnitude
`vterm`, the set-point `Vref` and its machine's speed `omega` (the machine's field current
`ifd` does not enter: the armature is ideal). `rec` takes the two decisions of the regulator's
non-windup limit.
"""
function _exc_rhs!(dx, e::PHShuntExciter, x, vterm, Vref, omega, ifd, rec::ModeRecorder)
    phi, xa, xf = x[1], x[2], x[3]
    efd = omega * _exc_x(e, phi)
    verr = Vref - vterm - e.KF * (efd - xf) / e.TF
    dx[1] = _exc_vf(e, phi, omega) + _exc_amp_volts(e, xa) - _exc_field_volts(e, phi)
    dx[2] = nonwindup(rec, xa, (e.KA * verr - xa) / e.TA, e.VRMIN, e.VRMAX)
    dx[3] = (efd - xf) / e.TF
    return dx
end

"""
    _exc_powers(e, phi, xa, omega, ifd)

The powers of the exciter at its field flux `phi`, regulator state `xa`, its machine's speed
and field current: the field voltage `vf` and field current `i_f`, the power the shaft gives
(`shaft = vf (ifd + i_f)`), the amplifier's supply `amp`, the power delivered to the machine's
field `field = vf ifd`, and the loss `r_F i_f^2`. For the energy account: `port`, the net power
the exciter receives; `tport`, the sum of the magnitudes of its port powers; `outside`, what it
receives from outside the model (the amplifier); `slope`, the derivative of its stored energy
with respect to its first state; `flux_rate`, that state's rate.
"""
@inline function _exc_powers(e::PHShuntExciter, phi, xa, omega, ifd)
    vf = _exc_vf(e, phi, omega)
    volts = _exc_field_volts(e, phi)
    i_f = volts / e.rF
    amp = _exc_amp_volts(e, xa) * i_f
    shaft, field = vf * (ifd + i_f), vf * ifd
    return (; vf, i_f, shaft, amp, field, loss = e.rF * i_f^2, port = shaft + amp - field,
            tport = abs(shaft) + abs(amp) + abs(field), outside = amp, slope = i_f,
            flux_rate = vf + _exc_amp_volts(e, xa) - volts)
end

# ----------------------------------------------------------------------------------------------
# `PHBrushlessExciter`: an alternator on the machine's shaft whose rotating diode bridge feeds
# the machine's field winding, with its regulator fed by a permanent-magnet generator on the
# same shaft (docs/active/PH_BRUSHLESS_EXCITER_MODEL.md). The device a case's IEEET1 with
# `KE = 1` is read as (a reading from typical constants, not an identification). The structure
# is that of the Type AC1 model of the IEEE Committee Report, IEEE Trans. PAS-100(2), 1981,
# p. 497, Fig. 6 and Appendix E; the energy account and the dependence on speed are ours.
# `IFD = Lad i_fd` and `EFD = v_f Lad / Rfd` are the machine's field current and voltage in the
# report's per unit.
#
#   state      x_E                       the exciter's field flux; VE = omega x_E
#   field      VFE = (KE + SE(x_E)) x_E + KD IFD,     TE x_E' = VR - VFE
#   rectifier  IN = KC IFD / x_E,    EFD = FEX(IN) omega x_E,    v_f = EFD Rfd / Lad
#   energy     W = g TE integral_0^{x_E} (KE + SE(s)) s ds;  loss g VFE^2
#   powers     P_dc = v_f i_fd;  regulator P_reg = g VR VFE;  to the armature P_tr = g KD IFD (VR - VFE)
#   shaft      the rotor gives P_dc - P_tr + P_reg: no supply from outside the shaft
#
# Signal laws: the regulator `KA`, `TA` with its limits (state `xa`, `VR = clamp(xa)`), and the
# stabiliser `KF`, `TF` fed by `VFE`, the exciter's field current (state `xf`).
# `KC`, `KD` are published typical values of one maker's exciter; `SE` is the alternator's
# no-load saturation `A exp(B x)`, for which no usable data exist; `g`, the power of the field
# circuit per unit of `VR^2`, is a number no source gives. None has a default.
# Outside the model, and to be reported for every run: `x_E <= 0` and `IFD < 0` (IEEE Std
# 421.5-2016, Annex D, p. 119: "If IFD is less than zero or IN is less than zero, the condition
# should be flagged"). The formulas are continued there: zero output for `x_E <= 0`, and no
# loading of the rectifier by a negative field current. `IN >= 1` is inside: "if IN is greater
# than 1 for any reason, the model should set FEX equal to 0" (same page), as `_fex` does. The speed must be positive (the shaft input is a
# power divided by the speed). The regulator's power stage is taken as lossless and reversible:
# with a negative `VR` the regulator's power is negative and goes back to the shaft.

"""
    PHBrushlessExciter

A brushless exciter on its machine's shaft (comment above). `KA`, `TA`, `KE`, `TE`, `KF`, `TF`,
`VRMAX`, `VRMIN` are the case's IEEET1 constants; `KC`, `KD` the rectifier's loading factor and
the demagnetising factor; `A`, `B` the alternator's no-load saturation; `g` the field circuit's
power per unit of `VR^2`; `Lad` and `gain = Rfd/Lad` the machine's. Local states:
`x_E, xa, xf`.
"""
struct PHBrushlessExciter
    name::String
    KA::Float64
    TA::Float64
    KE::Float64
    TE::Float64
    KF::Float64
    TF::Float64
    VRMAX::Float64
    VRMIN::Float64
    A::Float64
    B::Float64
    KC::Float64
    KD::Float64
    g::Float64
    Lad::Float64
    gain::Float64
end

"""
    PHBrushlessExciter(e::IEEET1_PHTRUE, Rfd, Lad; KC, KD, sat, field_share)

The physical exciter with the regulator, stabiliser and time constant of the case's IEEET1 `e`,
on a machine with field resistance `Rfd` and mutual inductance `Lad`. `sat = (A, B)` is the
alternator's no-load saturation function `A exp(B x)` (`(0, 0)` for none; the case's own
saturation constants are a Type 1 curve and are not used). `field_share` is the exciter's field
loss over the power delivered to the machine's field, both at one per unit, at rated speed,
unsaturated and without load: `g = field_share Rfd / (Lad KE)^2`. Refused: an input filter,
`KE` not positive, a gain or time constant that is not positive, data that are not finite or
have the wrong sign, and a field characteristic that is not increasing.
"""
function PHBrushlessExciter(e::IEEET1_PHTRUE, Rfd::Real, Lad::Real; KC::Real, KD::Real, sat, field_share::Real)
    p = e.p
    A, B = Float64(sat[1]), Float64(sat[2])
    all(isfinite, (p.TR, p.KA, p.TA, p.KE, p.TE, p.KF, p.TF, p.VRMAX, p.VRMIN, A, B, KC, KD, field_share, Rfd, Lad)) ||
        throw(ArgumentError("PHBrushlessExciter $(e.name): every parameter must be finite"))
    abs(p.TR) <= 1e-9 || throw(ArgumentError("PHBrushlessExciter $(e.name): an input filter (TR > 0) is not modelled"))
    (p.KE > 0 && p.KA > 1e-6 && p.TA > 1e-4 && p.TE > 1e-4 && p.TF > 1e-4) ||
        throw(ArgumentError("PHBrushlessExciter $(e.name): KE, KA, TA, TE and TF must be positive"))
    (KC >= 0 && KD >= 0 && A >= 0 && B >= 0 && p.KF >= 0 && field_share > 0 && Rfd > 0 && Lad > 0) ||
        throw(ArgumentError("PHBrushlessExciter $(e.name): KC, KD, KF and the saturation constants must not " *
                            "be negative; field_share, Rfd and Lad must be positive"))
    g, gain = field_share * Rfd / (Lad * p.KE)^2, Rfd / Lad
    (isfinite(g) && g > 0 && isfinite(gain) && gain > 0) ||
        throw(ArgumentError("PHBrushlessExciter $(e.name): the field circuit's power scale is not a positive finite number"))
    p.VRMIN < p.VRMAX || throw(ArgumentError("PHBrushlessExciter $(e.name): VRMIN must be below VRMAX"))
    (B > 0 ? p.KE - A * exp(-2.0) : p.KE + A) > 0 ||
        throw(ArgumentError("PHBrushlessExciter $(e.name): the field characteristic is not increasing"))
    return PHBrushlessExciter(e.name, p.KA, p.TA, p.KE, p.TE, p.KF, p.TF, p.VRMAX, p.VRMIN, A, B, KC, KD,
                              g, Lad, gain)
end

nstates(::PHBrushlessExciter) = 3
name(e::PHBrushlessExciter) = e.name
state_names(::PHBrushlessExciter) = ["x_E", "xa", "xf"]

"""
    _fex(IN)

The rectifier's regulation, `EFD / VE` against `IN = KC IFD / VE` (here `KC IFD / x_E`: the
speed cancels; 1981 report, Fig. E.1 and
E.2, with the exact coefficients behind its rounded ones, which join the three segments in
value and slope): mode I for `IN <= sqrt(3)/4`, mode II up to `3/4`, mode III up to 1, and zero
beyond.
"""
@inline function _fex(IN::T) where {T}
    IN <= sqrt(3) / 4 && return 1 - IN / sqrt(3)
    IN < 0.75 && return sqrt(0.75 - IN^2)
    IN < 1 && return sqrt(3) * (1 - IN)
    return zero(T)
end

@inline _exc_se(e::PHBrushlessExciter, x) = e.A == 0 ? zero(x) : e.A * exp(e.B * x)
# the signal proportional to the exciter's field current
@inline _bx_vfe(e::PHBrushlessExciter, x, IFD) = (e.KE + _exc_se(e, x)) * x + e.KD * IFD
# the rectifier's loading and the machine's field voltage in the report's per unit
@inline _bx_in(e::PHBrushlessExciter, x, IFD) = x > 0 ? e.KC * max(IFD, zero(IFD)) / x : oftype(x, Inf)
@inline _bx_efd(e::PHBrushlessExciter, x, IFD, omega) = x > 0 ? _fex(_bx_in(e, x, IFD)) * omega * x : zero(x * omega * IFD)

"""The stored energy of the exciter's field, `g TE integral_0^x (KE + SE(s)) s ds` (pu*s)."""
_exc_energy(e::PHBrushlessExciter, x) =
    e.g * e.TE * (e.KE / 2 + (e.A == 0 ? zero(x) : e.A * _exp_moment(e.B * x))) * x^2

"""
    _exc_rhs!(dx, e::PHBrushlessExciter, x, vterm, Vref, omega, ifd, rec) -> dx

The exciter's rows `[x_E', xa', xf']`. The stabiliser is fed by `VFE`.
"""
function _exc_rhs!(dx, e::PHBrushlessExciter, x, vterm, Vref, omega, ifd, rec::ModeRecorder)
    xE, xa, xf = x[1], x[2], x[3]
    vfe = _bx_vfe(e, xE, e.Lad * ifd)
    verr = Vref - vterm - e.KF * (vfe - xf) / e.TF
    dx[1] = (clamp(xa, e.VRMIN, e.VRMAX) - vfe) / e.TE
    dx[2] = nonwindup(rec, xa, (e.KA * verr - xa) / e.TA, e.VRMIN, e.VRMAX)
    dx[3] = (vfe - xf) / e.TF
    return dx
end

"""
    _exc_powers(e::PHBrushlessExciter, xE, xa, omega, ifd)

The powers of the brushless exciter: the machine's field voltage `vf`, the power delivered to
the machine's field `field`, the regulator's power `reg`, the power the field winding passes to
the armature `tr`, the loss, and the power the shaft gives, `shaft = field - tr + reg`; with
`VFE`, `EFD` and the rectifier's loading `IN`. For the energy account as for the DC exciter:
`port`, `tport`, `outside` (zero: everything comes from the shaft), `slope`, `flux_rate`.
"""
@inline function _exc_powers(e::PHBrushlessExciter, xE, xa, omega, ifd)
    IFD = e.Lad * ifd
    vfe = _bx_vfe(e, xE, IFD)
    vr = clamp(xa, e.VRMIN, e.VRMAX)
    efd = _bx_efd(e, xE, IFD, omega)
    vf = e.gain * efd
    field, reg, tr = vf * ifd, e.g * vr * vfe, e.g * e.KD * IFD * (vr - vfe)
    port = reg - tr                                    # what it receives net: from the shaft, less its output
    shaft = field + port
    return (; vf, shaft, field, reg, tr, loss = e.g * vfe^2, port,
            tport = abs(shaft) + abs(field), outside = zero(shaft),
            slope = e.g * e.TE * (e.KE + _exc_se(e, xE)) * xE, flux_rate = (vr - vfe) / e.TE,
            VFE = vfe, EFD = efd, IN = _bx_in(e, xE, IFD))
end

"""A machine's physical exciter: a `PHShuntExciter` or a `PHBrushlessExciter`."""
const PHPhysicalExciter = Union{PHShuntExciter,PHBrushlessExciter}
