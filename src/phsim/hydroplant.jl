# The physical hydropower plant (docs/active/PH_HYDRO_PLANT_MODEL.md; review 110): the 150 MW Torpa plant of Vereide et
# al. (IEEE Trans. Energy Conversion 32(1), 2017) with its waterway, air cushion and open surge tank, its two Francis
# turbines from the Euler turbine equation with the loss components of Giosio et al. (IEEE Trans. Power Systems 32(2),
# 2017; coefficients of Ida 1989), the runner's dimensionless shape from Nielsen's measured high-head runner (IJFMS 8(3),
# 2015), and the plant's PI speed governor. SI units inside; powers and energies to the system base at the interface.
#
#   headrace      (rho L1/A) Q1' = rho g (H_up - H_j) - rho g k1 Q1|Q1|                       (rigid column, friction)
#   junction      H_j = H_c + k_z Qc|Qc|,  Qc = Q1 - Q2 (into the cushion chamber, loss zeta = 60)
#   air cushion   V = V0 (1 - w), p = p0 (V0/V)^gamma (adiabatic), level z_c = z_c0 + w V0 / A_c,
#                 H_c = z_c + (p - p_atm)/(rho g);  V0 w' = Qc
#   conduit       (rho L2/A) Q2' = rho g (H_j - H_t - z_s) - rho g k2 Q2|Q2|   (cushion -> turbines -> open tank)
#   open tank     A_s z_s' = Q2 - Q3
#   tailrace      (rho L3/A) Q3' = rho g (z_s - H_low) - rho g k3 Q3|Q3|
#   turbines      two identical units at one opening: each runner takes Qr = Q2/2;
#                 cu1 = cm1 cot(a1), cm1 = Qr/A1, sin(a1) = y sin(a1R);  cu2 = u2 - cm2 cot(b2), cm2 = Qr/A2;  u = w r
#                 g H_E = u1 cu1 - u2 cu2 (Euler: T w = rho Q2 g H_E),  T = 2 rho Qr (r1 cu1 - r2 cu2)
#                 g H_t = g H_E + E_inc + E_sw + k_f Qr^2,  E_inc = 0.75 dW^2/2, dW = u1 - cu1 - cm1 cot(b1),
#                 E_sw = 1.0 cu2^2/2  (incidence and residual swirl, Giosio eq. 15-16; friction eq. 18)
#                 shaft power P = (T - c_m w) w;  turbine loss rho Q2 (E_inc + E_sw + k_f Qr^2) + c_m w^2 >= 0
#   energy        sum rho L_i Q_i^2/(2A) + (pV - p0V0)/(gamma-1) + p_atm (V - V0) + rho g A_c (z_c^2 - z_c0^2)/2
#                 + rho g A_s (z_s^2 - z_s0^2)/2;  its rate = rho g (H_up Q1 - H_low Q3) - P - friction - junction
#                 - turbine losses, exactly (the air cushion's energy is not quadratic: the midpoint keeps it to O(h^3))
#   governor      e = (1 - omega) - bp (y - y_ref),  y_c = y_ref + Kp e + xi,  xi' = (Kp/Ti) e (frozen at a gate
#                 limit),  y' = clamp((y_c - y)/Ty, -r, r) at the gate's limits  (signal law; Vereide: Kp 4, Ti 6 s,
#                 bp 6 %, closing 10 s; Ki = 6 read as the integral time Ti, review 110)
# States (per plant): xi, y (opening, 1 at the best-efficiency point), q1, w, q2, zs, q3 with q = Q/Q_R (Q_R = 40 m3/s),
# w the fraction of the cushion's air volume filled by water, zs = (z_s - H_low)/H_R. G 10 is driven by `np` identical
# plants (assumption A6, review 110: 2 plants, so that each unit runs inside Torpa's published range of 36-68 MW).
# Every number: source or assumption in the record. Domain: an open gate, forward flow, positive speed.

const HP_RHO = 1000.0
const HP_G = 9.81
const HP_SB = 1e8            # the system base (100 MVA) for powers and energies

"""
    PHHydroPlant

A physical hydropower plant in place of the case's IEEEG3 (file header). Built by `torpa_plant`.
"""
struct PHHydroPlant
    name::String
    np::Int                  # identical plants on the machine
    # waterway
    A::Float64; k1::Float64; k2::Float64; k3::Float64; kz::Float64
    L1::Float64; L2::Float64; L3::Float64
    Hup::Float64; Hlow::Float64
    V0::Float64; p0::Float64; gamma::Float64; patm::Float64; Ac::Float64; zc0::Float64
    As::Float64; zs0::Float64
    # runner (each of the two units)
    r1::Float64; r2::Float64; A1::Float64; A2::Float64; sa::Float64; ca::Float64; cotb1::Float64; cotb2::Float64
    kf::Float64; xi_inc::Float64; xi_sw::Float64; cm::Float64; omegaR::Float64
    # governor
    Kp::Float64; Ti::Float64; bp::Float64; Ty::Float64; rate::Float64; ymax::Float64
    # bases of the states
    QR::Float64; HR::Float64
end

nstates(::PHHydroPlant) = 7
name(g::PHHydroPlant) = g.name
state_names(::PHHydroPlant) = ["xi", "y", "q1", "w", "q2", "zs", "q3"]

"""
    torpa_plant(name; np = 2, kw...) -> PHHydroPlant

The Torpa plant (record section 2), at the cushion's operating pressure `p0` (set by `hydroplant_initialise`).

The runner (assumption A3, reviews 110): Torpa's best-efficiency point is taken as its most efficient published point,
130 MW at 32.9 m3/s (Vereide Table I; 16.45 m3/s per unit), at the net head the waterway leaves there; its hydraulic
efficiency follows from that point with the generator efficiency `etagen` (assumption A8) and the mechanical loss.
The runner's shape is Nielsen's measured high-head runner, made self-consistent from its own dimensions (D1 = 630 mm,
D2 = 349 mm, B1 = 60 mm, guide vane 10 deg, outlet blade 69.6 deg from the meridional direction, review 110 Gemini 2b):
zero outlet swirl and zero inlet incidence at its BEP give the inlet blade angle; the diameter ratio and the inlet
triangle are kept, and the passage areas follow from Torpa's flow (zero incidence, zero swirl at Torpa's BEP), which
is the specific-speed scaling of the passages.
"""
function torpa_plant(nm::AbstractString; np::Integer = 2, p0::Real = NaN, Ac::Real = 450.0, Ty::Real = 0.2,
                     mech::Real = 0.005, etagen::Real = 0.98, L2::Real = 400.0, ymax::Real = 1.4)
    g = HP_G
    A = 35.0; D = sqrt(4A / pi); f = 0.07                          # Vereide: 35 m2 unlined tunnels, friction 0.07
    k(L) = f * L / (2 * g * D * A^2)                               # head loss k Q|Q| (m per (m3/s)^2)
    L1, L3 = 9600.0, 10000.0                                       # headrace 9.6 km, tailrace 10 km
    kz = 60.0 / (2 * g * A^2)                                      # T-junction loss zeta = 60 at A = 35 m2
    omegaR = 2pi * 600 / 60
    # Nielsen's runner made self-consistent (zero swirl and zero incidence at its BEP, from its own dimensions)
    D1m, D2m, B1m, a1R = 0.630, 0.349, 0.060, deg2rad(10.0)
    cota = cot(a1R)
    u2_cm2 = tan(deg2rad(69.6))                                    # zero swirl: u2 = cm2 cot(b2t), b2t = 90 - 69.6 deg
    cm1_u1 = (D2m / D1m) / u2_cm2 / (4 * D1m * B1m / D2m^2)        # cm1/u1 = (u2/u1)(cm2/u2)(cm1/cm2)
    cotb1 = 1 / cm1_u1 - cota                                      # zero incidence: u1 = cm1 (cot a1R + cot b1)
    # Torpa's BEP (the 130 MW point) and its hydraulic efficiency
    QB = 32.9; QrB = QB / 2
    HB = 707.0 - 263.0 - (k(L1) + k(Float64(L2)) + k(L3)) * QB^2  # net head at the BEP
    Pm = mech * 75e6                                               # mechanical loss per unit at rated speed (A4)
    etah = (130e6 / etagen + 2Pm) / (HP_RHO * g * QB * HB)        # hydraulic efficiency at the BEP
    u1 = sqrt(etah * g * HB * (cota + cotb1) / cota)              # zero incidence + Euler at Torpa's BEP
    r1 = u1 / omegaR
    r2 = r1 * D2m / D1m
    A1 = QrB / (u1 * cm1_u1)
    A2 = pi * r2^2
    cotb2 = (omegaR * r2) / (QrB / A2)                             # zero outlet swirl at Torpa's BEP
    kf = (1 - etah) * g * HB / QrB^2                               # friction: the BEP's whole hydraulic loss
    cm = Pm / omegaR^2                                             # mechanical loss, viscous
    return PHHydroPlant(String(nm), Int(np), A, k(L1), k(Float64(L2)), k(L3), kz, L1, Float64(L2), L3,
                        707.0, 263.0, 13000.0, Float64(p0), 1.4, 101325.0, Float64(Ac), 292.3, 1650.0, NaN,
                        r1, r2, A1, A2, sin(a1R), cos(a1R), cotb1, cotb2, kf, 0.75, 1.0, cm, omegaR,
                        4.0, 6.0, 0.06, Float64(Ty), 0.1, Float64(ymax), 40.0, 430.0)
end

# the turbines at opening y, total flow Q2 (m3/s) and turbine speed wt (rad/s): head taken, torque, losses
function _hp_turbine(g::PHHydroPlant, y, Q2, wt)
    s1 = y * g.sa
    _deep(s1) > 0 || throw(DomainError(_deep(y), "PHHydroPlant $(g.name): a closed or negative opening"))
    _deep(Q2) >= 0 || throw(DomainError(_deep(Q2), "PHHydroPlant $(g.name): reverse flow through the turbines"))
    cota = sqrt(1 - s1^2) / s1
    Qr = Q2 / 2
    cm1, cm2 = Qr / g.A1, Qr / g.A2
    u1, u2 = wt * g.r1, wt * g.r2
    cu1 = cm1 * cota
    cu2 = u2 - cm2 * g.cotb2
    gHE = u1 * cu1 - u2 * cu2
    # incidence: the relative flow's tangential component less the blade-conforming one (Giosio eq. 15: positive
    # incidence 0.75 dW^2/2; negative incidence 0.75 (dW cos a1b)^2/2, a1b the guide vane angle at the BEP)
    dW = u1 - cu1 - cm1 * g.cotb1
    Einc = _deep(dW) >= 0 ? g.xi_inc * dW^2 / 2 : g.xi_inc * (dW * g.ca)^2 / 2
    Eloss = Einc + g.xi_sw * cu2^2 / 2 + g.kf * Qr^2
    Ht = (gHE + Eloss) / HP_G
    T = 2 * HP_RHO * Qr * (g.r1 * cu1 - g.r2 * cu2)
    Tm = 2 * g.cm * wt
    return (; Ht, T, Tm, P = (T - Tm) * wt, loss = HP_RHO * Q2 * Eloss + Tm * wt)
end

# the plant's physical quantities at the state x (SI) and machine speed omega (pu)
function _hp_state(g::PHHydroPlant, x, omega)
    Q1, Q2, Q3 = g.QR * x[3], g.QR * x[5], g.QR * x[7]
    w = x[4]
    V = g.V0 * (1 - w)
    p = g.p0 * (g.V0 / V)^g.gamma
    zc = g.zc0 + w * g.V0 / g.Ac
    Hc = zc + (p - g.patm) / (HP_RHO * HP_G)
    Qc = Q1 - Q2
    Hj = Hc + g.kz * Qc * abs(Qc)
    zs = g.Hlow + g.HR * x[6]
    tb = _hp_turbine(g, x[2], Q2, g.omegaR * omega)
    return (; Q1, Q2, Q3, V, p, zc, Hc, Qc, Hj, zs, tb)
end

"""
    _hydro_rhs!(dx, g::PHHydroPlant, x, omega, Pref, rec)

The plant's rows: the governor (xi, y), then the waterway (q1, w, q2, zs, q3). `Pref` is the opening reference.
"""
function _hydro_rhs!(dx, g::PHHydroPlant, x, omega, Pref, rec::ModeRecorder)
    ylim = clamp_mode(rec, x[2], 0.0, g.ymax)
    e = (1.0 - omega) - g.bp * (ylim - Pref)
    yc = Pref + g.Kp * e + x[1]
    dx[1] = nonwindup(rec, ylim, (g.Kp / g.Ti) * e, 0.0, g.ymax)
    v = clamp_mode(rec, (yc - ylim) / g.Ty, -g.rate, g.rate)
    dx[2] = nonwindup(rec, ylim, v, 0.0, g.ymax)
    q = _hp_state(g, x, omega)
    rg = HP_RHO * HP_G
    dx[3] = (rg * (g.Hup - q.Hj) - rg * g.k1 * q.Q1 * abs(q.Q1)) / (HP_RHO * g.L1 / g.A) / g.QR
    dx[4] = q.Qc / g.V0
    dx[5] = (rg * (q.Hj - q.tb.Ht - q.zs) - rg * g.k2 * q.Q2 * abs(q.Q2)) / (HP_RHO * g.L2 / g.A) / g.QR
    dx[6] = (q.Q2 - q.Q3) / g.As / g.HR
    dx[7] = (rg * (q.zs - g.Hlow) - rg * g.k3 * q.Q3 * abs(q.Q3)) / (HP_RHO * g.L3 / g.A) / g.QR
    return dx
end

"""The plants' shaft power on the system base at machine speed `omega` (pu)."""
_hydro_power(g::PHHydroPlant, x, omega) = g.np * _hp_state(g, x, omega).tb.P / HP_SB

# the stored energy (J, per plant, constants removed) and its gradient with respect to the states
function _hp_energy(g::PHHydroPlant, x)
    Q1, Q2, Q3 = g.QR * x[3], g.QR * x[5], g.QR * x[7]
    w = x[4]; V = g.V0 * (1 - w); p = g.p0 * (g.V0 / V)^g.gamma
    zc = g.zc0 + w * g.V0 / g.Ac; zs = g.Hlow + g.HR * x[6]
    kin = HP_RHO / (2 * g.A) * (g.L1 * Q1^2 + g.L2 * Q2^2 + g.L3 * Q3^2)
    air = (p * V - g.p0 * g.V0) / (g.gamma - 1) + g.patm * (V - g.V0)
    grav = HP_RHO * HP_G * (g.Ac * (zc^2 - g.zc0^2) + g.As * (zs^2 - g.zs0^2)) / 2
    return kin + air + grav
end
function _hp_energy_grad(g::PHHydroPlant, x)
    Q1, Q2, Q3 = g.QR * x[3], g.QR * x[5], g.QR * x[7]
    w = x[4]; V = g.V0 * (1 - w); p = g.p0 * (g.V0 / V)^g.gamma
    zc = g.zc0 + w * g.V0 / g.Ac; zs = g.Hlow + g.HR * x[6]
    rg = HP_RHO * HP_G
    dEw = (p - g.patm) * g.V0 + rg * zc * g.V0          # d(air)/dw = (p - p_atm) V0 (dU = -p dV); gravity rg A_c zc dzc
    return (zero(Q1), zero(Q1), HP_RHO * g.L1 / g.A * Q1 * g.QR, dEw, HP_RHO * g.L2 / g.A * Q2 * g.QR,
            rg * g.As * zs * g.HR, HP_RHO * g.L3 / g.A * Q3 * g.QR)
end

"""
    _hydro_powers(g::PHHydroPlant, x, omega) -> NamedTuple

For the energy account (system base, all `np` plants): the waterway's energy, the reservoirs' supply, the shaft power,
the net port, the sum of the port magnitudes, the losses and the energy's rate along the rows.
"""
function _hydro_powers(g::PHHydroPlant, x, omega)
    q = _hp_state(g, x, omega)
    rg = HP_RHO * HP_G
    supply = rg * (g.Hup * q.Q1 - g.Hlow * q.Q3)
    fric = rg * (g.k1 * abs(q.Q1)^3 + g.k2 * abs(q.Q2)^3 + g.k3 * abs(q.Q3)^3) + rg * g.kz * abs(q.Qc)^3
    loss = fric + q.tb.loss
    shaft = q.tb.P
    c = g.np / HP_SB
    return (; energy = c * _hp_energy(g, x), supply = c * supply, shaft = c * shaft, port = c * (supply - shaft),
            tport = c * (rg * (g.Hup * abs(q.Q1) + g.Hlow * abs(q.Q3)) + abs(shaft)), loss = c * loss,
            rate = c * (supply - shaft - loss))
end

"""The energy's rate along `dxu` (the plant's local states' derivatives), system base, all plants."""
function _hydro_energy_rate(g::PHHydroPlant, x, dxu)
    gr = _hp_energy_grad(g, x)
    return g.np / HP_SB * sum(gr[i] * dxu[i] for i in 3:7)
end

"""
    hydroplant_initialise(g, Pm) -> (g, x, Pref)

The plant at rest at shaft power `Pm` per plant... on the system base for all `np` plants: `Pm` is the machine's
turbine power (system base); each plant delivers Pm/np at rated speed. Steady flows equal, the cushion at its measured
level 292.3 m and volume 13 000 m3 (its pressure follows from the heads), the tank at the tailrace's head, the opening
from the turbine's power balance (bisection on the opening, then on the flow). Returns the plant with `p0`, `zs0` set.
"""
function hydroplant_initialise(g::PHHydroPlant, Pm::Real)
    P1 = Pm * HP_SB / g.np                                   # W per plant
    wt = g.omegaR
    # at opening y, the steady flow Q: H_up - k1 Q^2 - k2 Q^2 - (H_low + k3 Q^2) = H_t(y, Q)
    function flow(y)
        r(Q) = g.Hup - g.Hlow - (g.k1 + g.k2 + g.k3) * Q^2 - _hp_turbine(g, y, Q, wt).Ht
        lo, hi = 1e-6, 200.0
        r(hi) < 0 || error("hydroplant_initialise: no steady flow at opening $y")
        for _ in 1:200
            m = (lo + hi) / 2
            r(m) > 0 ? (lo = m) : (hi = m)
        end
        return (lo + hi) / 2
    end
    power(y) = _hp_turbine(g, y, flow(y), wt).P
    ylo, yhi = 1e-3, g.ymax
    power(yhi) >= P1 || error("hydroplant_initialise $(g.name): $(P1/1e6) MW above the plant's maximum $(power(yhi)/1e6) MW")
    for _ in 1:200
        m = (ylo + yhi) / 2
        power(m) < P1 ? (ylo = m) : (yhi = m)
    end
    y = (ylo + yhi) / 2
    Q = flow(y)
    Hc = g.Hup - g.k1 * Q^2                                   # no flow into the chamber: H_j = H_c
    p0 = g.patm + HP_RHO * HP_G * (Hc - g.zc0)
    zs0 = g.Hlow + g.k3 * Q^2
    gp = PHHydroPlant(g.name, g.np, g.A, g.k1, g.k2, g.k3, g.kz, g.L1, g.L2, g.L3, g.Hup, g.Hlow, g.V0, p0, g.gamma,
                      g.patm, g.Ac, g.zc0, g.As, zs0, g.r1, g.r2, g.A1, g.A2, g.sa, g.ca, g.cotb1, g.cotb2, g.kf, g.xi_inc,
                      g.xi_sw, g.cm, g.omegaR, g.Kp, g.Ti, g.bp, g.Ty, g.rate, g.ymax, g.QR, g.HR)
    x = [0.0, y, Q / g.QR, 0.0, Q / g.QR, (zs0 - g.Hlow) / g.HR, Q / g.QR]
    return gp, x, y
end

"""Steady efficiency check (record section 3.1): shaft power, flow and plant efficiency P/(rho g Q H_gross) at `P1` W."""
function hydroplant_steady(g::PHHydroPlant, P1::Real)
    gp, x, _ = hydroplant_initialise(PHHydroPlant(g.name, 1, (getfield(g, f) for f in fieldnames(PHHydroPlant)[3:end])...),
                                     P1 / HP_SB)
    Q = gp.QR * x[5]
    return (; Q, y = x[2], eta = P1 / (HP_RHO * HP_G * Q * (gp.Hup - gp.Hlow)), p0 = gp.p0)
end
