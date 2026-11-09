# Electromagnetic dq pilot of the physical extension (docs/records/early/PHYSICAL_EXTENSION.md, gate 1).
#
# A separate, balanced, averaged electromagnetic model in the synchronous dq frame; the frozen
# RMS case models are not touched. Elements:
#   - a full-winding salient-pole machine: stator flux linkages in the network frame, field,
#     one d-axis and one q-axis damper, rotor angle and momentum; the step-up transformer's
#     leakage is merged into the stator (exact: it is isotropic, so it commutes with the
#     rotation) and its ideal ratio sits at the network side;
#   - its field fed by an ideal averaged converter (v_f = m v_dc, i_dc = m i_f) from a DC link
#     capacitor, which a fixed DC source (a declared chemostat) feeds through R_s;
#   - dynamic pi lines (series R-L state, shunt capacitors in the node states) and a passive
#     RLC load (parallel G, L; C in the node);
#   - a fixed-voltage bus (a declared boundary) and, until the hydro unit is added, a fixed
#     shaft torque (a declared boundary).
#
# Units: per unit on the system base, time in seconds. Energies in pu*s: magnetic and network
# energies carry 1/omega_b (W = psi' L^-1 psi / (2 omega_b)), kinetic p^2/(4H) = H omega^2
# with p = 2 H omega, DC link q_dc^2/(2 C_dc). Machine currents use the motor convention
# (positive into the machine). Phasors: X_dq = Xd + j Xq = j e^{-j delta} X_net, PHPS's
# machine frame; the network frame rotates at nominal speed.

"""
    WindingMachine

Full winding data of a synchronous machine, per unit on the system base in the reciprocal
`L_ad` system: stator leakage `Ll` (including any merged external leakage), magnetising
`Lad`, `Laq`, rotor leakages `Lfd`, `L1d` (d axis: field and one damper) and `L1q`, `L2q`
(q axis: `nq = 1` damper for a salient-pole machine, `nq = 2` windings for a round rotor),
resistances, inertia `H` and `omega_b`. `Kd`, `Kq` are the inverse inductance matrices of the
d axis (stator, field, damper) and the q axis (stator, then the `nq` rotor windings), motor
convention. With `nq = 1`, `L2q = R2q = 0` are unused.
"""
struct WindingMachine
    Ll::Float64
    Lad::Float64
    Laq::Float64
    Lfd::Float64
    L1d::Float64
    L1q::Float64
    L2q::Float64
    Ra::Float64
    Rfd::Float64
    R1d::Float64
    R1q::Float64
    R2q::Float64
    nq::Int
    H::Float64
    omega_b::Float64
    Kd::Matrix{Float64}
    Kq::Matrix{Float64}
end

function WindingMachine(Ll, Lad, Laq, Lfd, L1d, L1q, Ra, Rfd, R1d, R1q, H, omega_b;
                        L2q::Union{Nothing,Real} = nothing, R2q::Real = 0.0)
    Ld = [Ll+Lad Lad Lad; Lad Lfd+Lad Lad; Lad Lad L1d+Lad]
    if L2q === nothing
        Lq = [Ll+Laq Laq; Laq L1q+Laq]
        return WindingMachine(Ll, Lad, Laq, Lfd, L1d, L1q, 0.0, Ra, Rfd, R1d, R1q, 0.0, 1, H,
                              omega_b, inv(Ld), inv(Lq))
    end
    Lq = [Ll+Laq Laq Laq; Laq L1q+Laq Laq; Laq Laq L2q+Laq]
    return WindingMachine(Ll, Lad, Laq, Lfd, L1d, L1q, L2q, Ra, Rfd, R1d, R1q, R2q, 2, H,
                          omega_b, inv(Ld), inv(Lq))
end

# d-axis map shared by GENSAL and GENROU (Sauer-Pai definitions)
function _winding_d_axis(xl, xd, xd1, xd2, Td1, Td2, wb)
    Lad = xd - xl
    Lfd = Lad * (xd1 - xl) / (xd - xd1)
    L1d = 1.0 / (1.0 / (xd2 - xl) - 1.0 / Lad - 1.0 / Lfd)
    Rfd = (Lfd + Lad) / (wb * Td1)
    R1d = (L1d + Lad * Lfd / (Lad + Lfd)) / (wb * Td2)
    return Lad, Lfd, L1d, Rfd, R1d
end

function _check_positive(vals)
    for (k, v) in pairs(vals)
        (isfinite(v) && v > 0) ||
            throw(ArgumentError("winding_machine: $k = $v is not positive (inconsistent data)"))
    end
end

"""
    winding_machine(p::GensalParams; x_ext = 0.0, r_ext = 0.0) -> WindingMachine

The explicit map from GENSAL's standard parameters to winding inductances and resistances,
with the exact (Sauer-Pai) definitions: `Lad = xd - xl`, `xd' = xl + Lad Lfd/(Lad + Lfd)`,
`xd'' = xl + 1/(1/Lad + 1/Lfd + 1/L1d)`, `Td0' = (Lad + Lfd)/(omega_b Rfd)`,
`Td0'' = (L1d + Lad Lfd/(Lad + Lfd))/(omega_b R1d)`, `xq'' = xl + Laq L1q/(Laq + L1q)`,
`Tq0'' = (Laq + L1q)/(omega_b R1q)`. With them the Sauer-Pai subtransient model is the exact
reduction of the winding model when the stator transients are neglected
(`emdq_reduction_check`). An external series leakage `x_ext` (a transformer) and resistance
`r_ext` are added to the stator. Throws if a winding parameter is not positive or the machine
data include saturation (not modelled here).
"""
function winding_machine(p::GensalParams; x_ext::Real = 0.0, r_ext::Real = 0.0)
    (p.A_sat != 0.0 || p.B_sat != 0.0) &&
        throw(ArgumentError("winding_machine: saturation is not modelled in the dq pilot"))
    xl, xq, xq2 = p.xl, p.xq, p.xq_double_prime
    wb = p.omega_b
    Lad, Lfd, L1d, Rfd, R1d = _winding_d_axis(xl, p.xd, p.xd_prime, p.xd_double_prime,
                                              p.Td0_prime, p.Td0_double_prime, wb)
    Laq = xq - xl
    L1q = Laq * (xq2 - xl) / (xq - xq2)
    R1q = (L1q + Laq) / (wb * p.Tq0_double_prime)
    _check_positive((; Lad, Laq, Lfd, L1d, L1q, Rfd, R1d, R1q))
    return WindingMachine(xl + x_ext, Lad, Laq, Lfd, L1d, L1q, p.ra + r_ext, Rfd, R1d, R1q,
                          p.H, wb)
end

"""
    winding_machine(p::GenrouParams; x_ext = 0.0, r_ext = 0.0) -> WindingMachine

GENROU's round rotor: the d axis as for GENSAL, and on the q axis the same exact
definitions with a transient winding `1q` and a subtransient winding `2q`:
`Laq = xq - xl`, `xq' = xl + Laq L1q/(Laq + L1q)`, `xq'' = xl + 1/(1/Laq + 1/L1q + 1/L2q)`,
`Tq0' = (Laq + L1q)/(omega_b R1q)`, `Tq0'' = (L2q + Laq L1q/(Laq + L1q))/(omega_b R2q)`.
GENROU_PHTRUE has no saturation.
"""
function winding_machine(p::GenrouParams; x_ext::Real = 0.0, r_ext::Real = 0.0)
    xl, xq, xq1, xq2 = p.xl, p.xq, p.xq_prime, p.xq_double_prime
    wb = p.omega_b
    Lad, Lfd, L1d, Rfd, R1d = _winding_d_axis(xl, p.xd, p.xd_prime, p.xd_double_prime,
                                              p.Td0_prime, p.Td0_double_prime, wb)
    Laq, L1q, L2q, R1q, R2q = _winding_d_axis(xl, xq, xq1, xq2, p.Tq0_prime, p.Tq0_double_prime, wb)
    _check_positive((; Lad, Laq, Lfd, L1d, L1q, L2q, Rfd, R1d, R1q, R2q))
    return WindingMachine(xl + x_ext, Lad, Laq, Lfd, L1d, L1q, p.ra + r_ext, Rfd, R1d, R1q,
                          p.H, wb; L2q, R2q)
end

"""
    EMDQPilot

Gate-1 electrical pilot: machine (with the merged transformer) -> ideal ratio `tap` -> bus 2
node (`C2`) -> line (`R23`, `L23`) -> bus 3 node (`C3`, load `G3`, `L3`) -> line (`R34`, `L34`)
-> fixed-voltage boundary `V4`. Shaft torque `Tm` fixed (temporary boundary); DC source
`vs` behind `Rs` feeding the link `Cdc`; nominal modulation `mstar`. `Dm` is a shaft damping
torque `-Dm (omega - 1)` for diagnostics only (0 for the case: the record has `D = 0`; it
stands in for the mechanical damping a prime mover would supply). Units in `pilot.jl`.
"""
struct EMDQPilot
    mach::WindingMachine
    tap::Float64
    C2::Float64
    R23::Float64
    L23::Float64
    C3::Float64
    G3::Float64
    L3::Float64
    R34::Float64
    L34::Float64
    V4::ComplexF64
    Tm::Float64
    vs::Float64
    Rs::Float64
    Cdc::Float64
    mstar::Float64
    Dm::Float64
end

"""State names of the pilot with a one-damper q axis, in order (`emdq_state_names`)."""
const EMDQ_STATES = ["psi_D", "psi_Q", "psi_fd", "psi_1d", "psi_1q", "delta", "p", "q_dc",
                     "v2_Re", "v2_Im", "i23_Re", "i23_Im", "v3_Re", "v3_Im", "i34_Re",
                     "i34_Im", "iL_Re", "iL_Im"]

"""
    emdq_state_names(m::WindingMachine) -> Vector{String}

The pilot's state names: `EMDQ_STATES`, and for a round rotor (`nq = 2`) the second q-axis
winding `psi_2q` appended as state 19 (so every other index is the same for both machines).
"""
emdq_state_names(m::WindingMachine) = m.nq == 2 ? vcat(EMDQ_STATES, "psi_2q") : copy(EMDQ_STATES)

"""    emdq_machine_block(m::WindingMachine) -> the machine's flux, angle and momentum indices."""
emdq_machine_block(m::WindingMachine) = m.nq == 2 ? [1:7; 19] : collect(1:7)

@inline _ejd(delta) = complex(cos(delta), sin(delta))

"""
    emdq_machine(m::WindingMachine, x) -> NamedTuple

Rotor-frame fluxes and currents of the pilot state `x`: `psid, psiq, id, iq, ifd, i1d,
i1q, i2q` (motor convention; `i2q = 0` for `nq = 1`), the network-frame stator current `in`,
and the motor-convention air-gap torque `Tem = psid iq - psiq id` (the generator torque is
`-Tem`).
"""
function emdq_machine(m::WindingMachine, x)
    delta = x[6]
    psic = im * conj(_ejd(delta)) * complex(x[1], x[2])
    psid, psiq = real(psic), imag(psic)
    psifd, psi1d, psi1q = x[3], x[4], x[5]
    K, Q = m.Kd, m.Kq
    id = K[1, 1] * psid + K[1, 2] * psifd + K[1, 3] * psi1d
    ifd = K[2, 1] * psid + K[2, 2] * psifd + K[2, 3] * psi1d
    i1d = K[3, 1] * psid + K[3, 2] * psifd + K[3, 3] * psi1d
    if m.nq == 2
        psi2q = x[19]
        iq = Q[1, 1] * psiq + Q[1, 2] * psi1q + Q[1, 3] * psi2q
        i1q = Q[2, 1] * psiq + Q[2, 2] * psi1q + Q[2, 3] * psi2q
        i2q = Q[3, 1] * psiq + Q[3, 2] * psi1q + Q[3, 3] * psi2q
    else
        iq = Q[1, 1] * psiq + Q[1, 2] * psi1q
        i1q = Q[2, 1] * psiq + Q[2, 2] * psi1q
        i2q = zero(iq)
    end
    in_ = -im * _ejd(delta) * complex(id, iq)
    return (; psid, psiq, id, iq, ifd, i1d, i1q, i2q, in = in_, Tem = psid * iq - psiq * id)
end

"""
    emdq_rhs!(dx, s::EMDQPilot, x, m = s.mstar) -> dx

The pilot's vector field at modulation `m`. Generic in the element type of `x` and `m`.
"""
function emdq_rhs!(dx, s::EMDQPilot, x, m = s.mstar)
    M = s.mach
    wb = M.omega_b
    e = emdq_machine(M, x)
    psin = complex(x[1], x[2])
    v2, i23 = complex(x[9], x[10]), complex(x[11], x[12])
    v3, i34, iL = complex(x[13], x[14]), complex(x[15], x[16]), complex(x[17], x[18])
    vdc = x[8] / s.Cdc
    is = (s.vs - vdc) / s.Rs
    dpsi = wb * (v2 / s.tap - M.Ra * e.in - im * psin)
    dx[1], dx[2] = real(dpsi), imag(dpsi)
    dx[3] = wb * (m * vdc - M.Rfd * e.ifd)
    dx[4] = -wb * M.R1d * e.i1d
    dx[5] = -wb * M.R1q * e.i1q
    dx[6] = wb * (x[7] / (2.0 * M.H) - 1.0)
    dx[7] = s.Tm + e.Tem - s.Dm * (x[7] / (2.0 * M.H) - 1.0)
    dx[8] = is - m * e.ifd
    dv2 = (wb / s.C2) * (-e.in / s.tap - i23 - im * s.C2 * v2)
    di23 = (wb / s.L23) * (v2 - v3 - s.R23 * i23 - im * s.L23 * i23)
    dv3 = (wb / s.C3) * (i23 - i34 - s.G3 * v3 - iL - im * s.C3 * v3)
    di34 = (wb / s.L34) * (v3 - s.V4 - s.R34 * i34 - im * s.L34 * i34)
    diL = (wb / s.L3) * (v3 - im * s.L3 * iL)
    dx[9], dx[10] = real(dv2), imag(dv2)
    dx[11], dx[12] = real(di23), imag(di23)
    dx[13], dx[14] = real(dv3), imag(dv3)
    dx[15], dx[16] = real(di34), imag(di34)
    dx[17], dx[18] = real(diL), imag(diL)
    M.nq == 2 && (dx[19] = -wb * M.R2q * e.i2q)
    return dx
end

emdq_rhs(s::EMDQPilot, x, m = s.mstar) =
    emdq_rhs!(similar(x, promote_type(eltype(x), typeof(m))), s, x, m)

"""
    emdq_energy_parts(s::EMDQPilot, x) -> NamedTuple

The physical energy by element (pu*s): `magnetic` (all windings, the merged transformer
leakage included), `kinetic` `p^2/(4H)`, `dc_link`, `network` (line inductors, node
capacitors, load inductor).
"""
function emdq_energy_parts(s::EMDQPilot, x)
    M = s.mach
    wb = M.omega_b
    e = emdq_machine(M, x)
    magnetic = (e.psid * e.id + e.psiq * e.iq + x[3] * e.ifd + x[4] * e.i1d + x[5] * e.i1q +
                (M.nq == 2 ? x[19] * e.i2q : zero(e.iq))) / (2.0 * wb)
    kinetic = x[7]^2 / (4.0 * M.H)
    dc_link = x[8]^2 / (2.0 * s.Cdc)
    network = (s.C2 * (x[9]^2 + x[10]^2) + s.L23 * (x[11]^2 + x[12]^2) +
               s.C3 * (x[13]^2 + x[14]^2) + s.L34 * (x[15]^2 + x[16]^2) +
               s.L3 * (x[17]^2 + x[18]^2)) / (2.0 * wb)
    return (; magnetic, kinetic, dc_link, network)
end

"""    emdq_energy(s, x) -> the pilot's total physical energy `H(x)`."""
emdq_energy(s::EMDQPilot, x) = sum(emdq_energy_parts(s, x))

"""
    emdq_power(s::EMDQPilot, x, m = s.mstar) -> (; supplies, dissipation)

Boundary supplies (`shaft = (Tm + Dm) omega`, `dc_source = vs i_s`,
`fixed_bus = -Re(conj(i34) V4)`, the power the fixed bus delivers) and every loss (stator,
field, dampers, source resistance, lines, load conductance, diagnostic shaft damping
`Dm omega^2`). The raw balance is `dH/dt = sum(supplies) - sum(dissipation)` for every `m`
(the converter is lossless).
"""
function emdq_power(s::EMDQPilot, x, m = s.mstar)
    M = s.mach
    e = emdq_machine(M, x)
    is = (s.vs - x[8] / s.Cdc) / s.Rs
    i23, i34 = complex(x[11], x[12]), complex(x[15], x[16])
    omega = x[7] / (2.0 * M.H)
    supplies = (; shaft = (s.Tm + s.Dm) * omega, dc_source = s.vs * is,
                fixed_bus = -real(conj(i34) * s.V4))
    dissipation = (; stator = M.Ra * abs2(e.in), field = M.Rfd * e.ifd^2,
                   dampers = M.R1d * e.i1d^2 + M.R1q * e.i1q^2 + M.R2q * e.i2q^2,
                   source = s.Rs * is^2,
                   lines = s.R23 * abs2(i23) + s.R34 * abs2(i34),
                   load = s.G3 * (x[13]^2 + x[14]^2), shaft = s.Dm * omega^2)
    return (; supplies, dissipation)
end

# ---------------------------------------------------------------------------------------
# Builder from the IEEE-39 case

function _findreq(pred, v, msg)
    k = findfirst(pred, v)
    k === nothing && throw(ArgumentError(msg))
    return k
end

_case_line(case::Case, idx::AbstractString) =
    case.lines[_findreq(l -> l.idx == idx, case.lines, "no line $idx in the case")]

"""
    emdq_pilot(eq; machine = "GENROU_11", trf = "Trf_1", line = "Line_3", tie = "Line_5",
               mstar = 0.5, source_drop = 0.05, tau_dc = 0.02) -> (pilot, xstar, report)

Gate-1 pilot from the IEEE-39 equilibrium `eq` (`solve_equilibrium`): a GENSAL or GENROU
record (`machine`; the hydro unit by default) with its step-up transformer `trf` merged into
the stator, the
adjacent line `line` (bus 2 to bus 3), bus 3's demand as a constant-impedance RLC surrogate
fitted to its P and Q at the IEEE-39 voltage (passive surrogate, not an identification of the
case load), and the line `tie` (bus 3 to bus 4) to a fixed-voltage boundary at the IEEE-39
voltage of bus 4. The machine's operating point is the IEEE-39 equilibrium: `|V|` at its
bus and the power its transformer carries (the generator output less any load at the
machine bus, which has no node in the merged model; `report.machine_bus_load`), or the
overrides `P`, `V` for operating-point studies; a pilot power flow fixes the rest. Diagnostics only (both 0 for the case): `Dm`, a shaft damping torque, and `ra_add`, an
added stator resistance (behind the machine bus, so the pilot power flow is unchanged).

DC excitation (declared design values, not identified data): nominal modulation `mstar`,
source drop `source_drop` of the link voltage at the operating point, link time constant
`tau_dc = R_s C_dc`. `xstar` is the exact equilibrium with `m = mstar`; `report` holds the
operating point and the RMS-equivalent field voltage.
"""
function emdq_pilot(eq; machine::AbstractString = "GENROU_11", trf::AbstractString = "Trf_1",
                    line::AbstractString = "Line_3", tie::AbstractString = "Line_5",
                    mstar::Real = 0.5, source_drop::Real = 0.05, tau_dc::Real = 0.02,
                    P::Union{Nothing,Real} = nothing, V::Union{Nothing,Real} = nothing,
                    Dm::Real = 0.0, ra_add::Real = 0.0)
    sys = eq.sys
    case = sys.case
    net = sys.net
    k = _findreq(c -> name(c) == machine, sys.comps, "no component $machine")
    c = sys.comps[k]
    c isa Union{GENSAL_PHTRUE,GENROU_PHTRUE} ||
        throw(ArgumentError("$machine is not a GENSAL_PHTRUE or GENROU_PHTRUE"))
    Vb(b) = (i = bus_index(net, b); complex(eq.V[2i-1], eq.V[2i]))
    lt, l23, l34 = _case_line(case, trf), _case_line(case, line), _case_line(case, tie)
    gbus = c.bus
    (lt.bus2 == gbus && l23.bus1 == lt.bus1 && l34.bus1 == l23.bus2) ||
        throw(ArgumentError("emdq_pilot expects the chain machine bus <- $trf -> bus a -> " *
                            "$line -> bus b -> $tie -> boundary bus (bus1/bus2 order)"))
    (lt.phi == 0.0 && lt.b == 0.0) || throw(ArgumentError("$trf: phase shift or charging"))
    b2, b3, b4 = l23.bus1, l23.bus2, l34.bus2
    # the machine sends what its transformer carries at the IEEE-39 equilibrium (a load at the
    # machine bus has no node in the merged model and is folded into this power)
    yt0 = 1.0 / complex(lt.r, lt.x)
    P0 = P === nothing ? real(Vb(gbus) * conj(yt0 * (Vb(gbus) - Vb(lt.bus1) / lt.tap))) : Float64(P)
    Vg = V === nothing ? abs(Vb(gbus)) : Float64(V)
    il = findfirst(l -> l.bus == gbus, case.pq)
    gbus_load = il === nothing ? 0.0im : complex(case.pq[il].p0, case.pq[il].q0)
    ld = _findreq(l -> l.bus == b3, case.pq, "no load at $b3")
    Pl, Ql = case.pq[ld].p0, case.pq[ld].q0
    Ql > 0 || throw(ArgumentError("bus $b3 demand is not inductive; the surrogate needs L"))
    V3c = abs(Vb(b3))
    G3, BL = Pl / V3c^2, Ql / V3c^2
    V4 = Vb(b4)

    yt = 1.0 / complex(lt.r, lt.x)
    y23, y34 = 1.0 / complex(l23.r, l23.x), 1.0 / complex(l34.r, l34.x)
    t = lt.tap
    C2, C3 = l23.b / 2, l23.b / 2 + l34.b / 2
    # pilot power flow: |V| and P at the machine bus; V2, V3 from KCL (linear given theta)
    function flow(theta)
        Vg_c = Vg * complex(cos(theta), sin(theta))
        # KCL 2: yt/t (Vg - V2/t) = j C2 V2 + y23 (V2 - V3)
        # KCL 3: y23 (V2 - V3) = (j C3 + G3 - j BL) V3 + y34 (V3 - V4)
        A = [yt/t^2 + im*C2 + y23 -y23; -y23 y23 + im*C3 + G3 - im*BL + y34]
        rhs = [yt / t * Vg_c, y34 * V4]
        V2, V3 = A \ rhs
        Ig = yt * (Vg_c - V2 / t)
        return Vg_c, V2, V3, Ig
    end
    Pf(th) = real(flow(th)[1] * conj(flow(th)[4])) - P0
    th = angle(Vb(gbus))
    for _ in 1:50
        h = 1e-7
        d = (Pf(th + h) - Pf(th - h)) / (2h)
        step = Pf(th) / d
        th -= step
        abs(step) < 1e-15 && break
    end
    abs(Pf(th)) < 1e-10 || error("emdq_pilot: the pilot power flow did not converge")
    Vg_c, V2, V3, Ig = flow(th)

    # machine (transformer merged): terminal = transformer secondary, v_m = V2/t, i_n = -Ig
    M = winding_machine(c.p; x_ext = lt.x, r_ext = lt.r + ra_add)
    vm, in_ = V2 / t, -Ig
    E = vm - complex(M.Ra, M.Ll + M.Laq) * in_
    delta = angle(E)
    rot = im * conj(_ejd(delta))
    vc, ic = rot * vm, rot * in_
    psic = -im * (vc - M.Ra * ic)
    psid, psiq = real(psic), imag(psic)
    id, iq = real(ic), imag(ic)
    ifd = (psid - (M.Ll + M.Lad) * id) / M.Lad
    ifd > 0 || error("emdq_pilot: field current $ifd is not positive")
    psifd = M.Lad * id + (M.Lfd + M.Lad) * ifd
    psi1d = M.Lad * (id + ifd)
    psi1q = M.Laq * iq
    q_check = psiq - (M.Ll + M.Laq) * iq
    psin = -im * _ejd(delta) * psic
    Tem = psid * iq - psiq * id
    efd = M.Rfd * ifd
    vdc = efd / mstar
    Rs = source_drop * vdc / (mstar * ifd)
    vs = (1 + source_drop) * vdc
    Cdc = tau_dc / Rs
    pilot = EMDQPilot(M, t, C2, l23.r, l23.x, C3, G3, 1.0 / BL, l34.r, l34.x, V4, -Tem,
                      vs, Rs, Cdc, Float64(mstar), Float64(Dm))
    i23, i34 = y23 * (V2 - V3), y34 * (V3 - V4)
    iL = V3 / (im / BL)
    xstar = [real(psin), imag(psin), psifd, psi1d, psi1q, delta, 2.0 * M.H, Cdc * vdc,
             real(V2), imag(V2), real(i23), imag(i23), real(V3), imag(V3), real(i34),
             imag(i34), real(iL), imag(iL)]
    M.nq == 2 && push!(xstar, M.Laq * iq)         # psi_2q (no q-axis rotor current)
    report = (; buses = (gbus, b2, b3, b4), P = P0, Q = imag(Vg_c * conj(Ig)), Vg = Vg_c,
              V2, V3, V4, Ig, efd, ifd, vdc, Efd_rms = M.Lad * efd / M.Rfd,
              q_axis_residual = q_check, Tm = -Tem, machine_bus_load = gbus_load)
    return pilot, xstar, report
end
