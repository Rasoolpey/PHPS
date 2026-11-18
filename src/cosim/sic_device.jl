# The device-level SiC MOSFET half-bridge leg in port-Hamiltonian form (docs/active/COSIM_DESIGN.md, sections 4 and 8.7;
# review 102). One switch position = one CAB760M12HM3 module position (the converter's three parallel modules: one
# module carrying i / 3, its currents and energies x 3).
#
# Per MOSFET: the capacitances C_gs (constant), C_gd(v_gd), C_ds(v_ds) in the device's two node voltages (v_gs, v_ds),
# v_gd = v_gs - v_ds identically (review 102, row 18); the channel I_ch(v_gs, v_ch) in series with the drift resistance
# R_d; the body diode i_bd(v_sd); the gate resistance R_G to an ideal driver referenced to the Kelvin source. Per leg: the
# commutation loop L_sigma with the package resistances R_u (upper position) and R_l (lower position). The phase current
# i_k (out of the pole) and the DC voltage are held over a step.
#
# Discretisation: per MOSFET a discrete gradient in its node-charge space, phibar = v_mid + delta / |Dq|^2 Dq, with
# delta = Delta H_C - v_mid' Dq computed analytically element by element (no subtraction of nearly equal energies), so
# phibar' Dq = Delta H_C exactly; the loop flux by the midpoint rule; every resistive element at the discrete efforts.
# Each step then satisfies its energy identity exactly (to the Newton residual): `leg_step!` returns the ledger.

# ---------------------------------------------------------------------------------------------------------------------
# piecewise-linear capacitance C(v) with closed-form charge q(v) = int_0^v C and energy E(v) = int_0^v x C(x) dx
# ---------------------------------------------------------------------------------------------------------------------

"""
    PWLCap(x, c)

A capacitance C(v), piecewise linear on the knots x (ascending, containing 0) with values c > 0, constant beyond the
end knots; the charge q(v) = int_0^v C and the energy E(v) = int_0^v x C(x) dx in closed form.
"""
struct PWLCap
    x::Vector{Float64}
    c::Vector{Float64}
    s::Vector{Float64}   # slope of piece k (between x[k] and x[k+1])
    q::Vector{Float64}   # q(x[k])
    e::Vector{Float64}   # E(x[k])
end

@inline _pq(c, s, u) = c * u + s * u^2 / 2                       # int_0^u (c + s t) dt
@inline _pe(x0, c, s, u) = x0 * c * u + (x0 * s + c) * u^2 / 2 + s * u^3 / 3   # int_0^u (x0 + t)(c + s t) dt

function PWLCap(x::AbstractVector, c::AbstractVector)
    x = Float64.(collect(x)); c = Float64.(collect(c))
    n = length(x)
    (n == length(c) && n >= 2 && issorted(x) && allunique(x) && all(>(0), c)) ||
        throw(ArgumentError("PWLCap: knots ascending and distinct, values positive"))
    k0 = findfirst(==(0.0), x)
    k0 === nothing && throw(ArgumentError("PWLCap: the knots must contain 0"))
    s = [(c[k+1] - c[k]) / (x[k+1] - x[k]) for k in 1:n-1]
    q = zeros(n); e = zeros(n)
    for k in k0+1:n
        u = x[k] - x[k-1]
        q[k] = q[k-1] + _pq(c[k-1], s[k-1], u)
        e[k] = e[k-1] + _pe(x[k-1], c[k-1], s[k-1], u)
    end
    for k in k0-1:-1:1
        u = x[k+1] - x[k]
        q[k] = q[k+1] - _pq(c[k], s[k], u)
        e[k] = e[k+1] - _pe(x[k], c[k], s[k], u)
    end
    return PWLCap(x, c, s, q, e)
end

# the piece of v: 0 below the first knot, n at or above the last, else k with x[k] <= v < x[k+1]
@inline function _piece(p::PWLCap, v)
    x = p.x
    v < x[1] && return 0
    v >= x[end] && return length(x)
    lo, hi = 1, length(x)
    while hi - lo > 1
        mid = (lo + hi) >>> 1
        v < x[mid] ? (hi = mid) : (lo = mid)
    end
    return lo
end

function capC(p::PWLCap, v)
    k = _piece(p, v)
    k == 0 && return p.c[1] + zero(v)
    k == length(p.x) && return p.c[end] + zero(v)
    return p.c[k] + p.s[k] * (v - p.x[k])
end

function capq(p::PWLCap, v)
    k = _piece(p, v)
    k == 0 && return p.q[1] + p.c[1] * (v - p.x[1])
    k == length(p.x) && return p.q[end] + p.c[end] * (v - p.x[end])
    return p.q[k] + _pq(p.c[k], p.s[k], v - p.x[k])
end

function capE(p::PWLCap, v)
    k = _piece(p, v)
    k == 0 && return p.e[1] + p.c[1] * (v^2 - p.x[1]^2) / 2
    k == length(p.x) && return p.e[end] + p.c[end] * (v^2 - p.x[end]^2) / 2
    return p.e[k] + _pe(p.x[k], p.c[k], p.s[k], v - p.x[k])
end

"""
    capdelta(p, a, b) = int_a^b (x - (a + b)/2) C(x) dx

Exactly E(b) - E(a) - (a + b)/2 (q(b) - q(a)), summed piece by piece from local formulas (no cancellation): on a piece
[l, r] with C linear of slope s, int (x - m) C = s (r - l)^3 / 12 + ((l + r)/2 - m) C((l + r)/2) (r - l).
"""
function capdelta(p::PWLCap, a, b)
    T = promote_type(typeof(a), typeof(b))
    a == b && return zero(T)
    lo, hi, sg = a < b ? (a, b, 1) : (b, a, -1)
    m = (a + b) / 2
    acc = zero(T)
    x = p.x
    n = length(x)
    l = lo
    k = _piece(p, lo)
    while true
        # the piece's right end (or hi)
        rk = k == n ? hi : (k == 0 ? min(hi, x[1] + zero(T)) : min(hi, x[k+1] + zero(T)))
        mp = (l + rk) / 2
        w = rk - l
        sl = (k == 0 || k == n) ? 0.0 : p.s[k]
        acc += sl * w^3 / 12 + (mp - m) * capC(p, mp) * w
        rk >= hi && break
        l = rk
        k += 1
    end
    return sg * acc
end

# ---------------------------------------------------------------------------------------------------------------------
# the device
# ---------------------------------------------------------------------------------------------------------------------

"""The data of one SiC MOSFET switch position (one module position)."""
struct SiCDevice
    name::String
    T::Float64              # junction temperature (C)
    beta::Float64           # channel: I_sat = beta Vov^3 (A/V^3)
    vth::Float64            # V
    vs::Float64             # softplus width of Vov (V)
    Rd::Float64             # drift resistance (Ohm)
    Cgs::Float64            # F
    cgd::PWLCap             # C_gd(v_gd)
    cds::PWLCap             # C_ds(v_ds)
    bd_v::Vector{Float64}   # body diode: v_sd knots (V) and currents (A), monotone
    bd_i::Vector{Float64}
    gleak::Float64          # S (the diode's leakage; keeps the Newton matrix regular at zero current)
    RGint::Float64          # Ohm
    source::String
end

@inline _softplus(x) = x > 30 ? x : log1p(exp(x))

"""The channel current (drain to source) at gate voltage vgs and channel voltage v: beta [Vov^3 - (Vov - |v|)^3] for
|v| < Vov, beta Vov^3 beyond; odd in v (the third quadrant through the channel, an approximation: review 102, row 15)."""
@inline function chan_current(d::SiCDevice, vgs, v)
    vov = d.vs * _softplus((vgs - d.vth) / d.vs)
    w = abs(v)
    f = w < vov ? d.beta * (vov^3 - (vov - w)^3) : d.beta * vov^3
    return v >= 0 ? f : -f
end

"""The body diode's current (source to drain) at v_sd (piecewise linear, monotone, extrapolated with the last slope;
the leakage gleak v_sd everywhere)."""
@inline function body_current(d::SiCDevice, vsd)
    i = d.gleak * vsd
    vsd <= d.bd_v[1] && return i
    vv, ii = d.bd_v, d.bd_i
    n = length(vv)
    k = 1
    while k < n - 1 && vsd > vv[k+1]
        k += 1
    end
    return i + ii[k] + (ii[k+1] - ii[k]) * (vsd - vv[k]) / (vv[k+1] - vv[k])
end

# one MOSFET's node charges (q_G, q_D) and capacitive storage at (v_gs, v_ds)
@inline _qG(d::SiCDevice, vgs, vds) = d.Cgs * vgs + capq(d.cgd, vgs - vds)
@inline _qD(d::SiCDevice, vgs, vds) = capq(d.cds, vds) - capq(d.cgd, vgs - vds)
@inline dev_energy(d::SiCDevice, vgs, vds) = d.Cgs * vgs^2 / 2 + capE(d.cgd, vgs - vds) + capE(d.cds, vds)

"""
    dev_discrete(d, vgs0, vds0, vgs1, vds1) -> (dqG, dqD, phiG, phiD)

The node-charge increments and the discrete efforts of one MOSFET over a step: phibar' Dq = Delta H_C exactly.
"""
@inline function dev_discrete(d::SiCDevice, vgs0, vds0, vgs1, vds1)
    dqG = _qG(d, vgs1, vds1) - _qG(d, vgs0, vds0)
    dqD = _qD(d, vgs1, vds1) - _qD(d, vgs0, vds0)
    del = capdelta(d.cgd, vgs0 - vds0, vgs1 - vds1) + capdelta(d.cds, vds0, vds1)
    phiG = (vgs0 + vgs1) / 2
    phiD = (vds0 + vds1) / 2
    n2 = dqG^2 + dqD^2
    if n2 > 0
        f = del / n2
        phiG += f * dqG
        phiD += f * dqD
    end
    return dqG, dqD, phiG, phiD
end

# ---------------------------------------------------------------------------------------------------------------------
# the CAB760M12HM3 from its data sheet (Rev. 5, 2025): its curves digitised once (src/cosim/data)
# ---------------------------------------------------------------------------------------------------------------------

const _CAB_DIGITISED = joinpath(@__DIR__, "data", "CAB760M12HM3_rev5_digitised.json")

# the integral of a piecewise-linear table (xs, ys) over [a, b] (constant beyond the ends)
function _pwl_int(xs, ys, a, b)
    f(v) = v <= xs[1] ? ys[1] : v >= xs[end] ? ys[end] :
           (k = searchsortedlast(xs, v); ys[k] + (ys[k+1] - ys[k]) * (v - xs[k]) / (xs[k+1] - xs[k]))
    pts = sort(unique(vcat([a, b], filter(v -> a < v < b, xs))))
    return sum((f(pts[i]) + f(pts[i+1])) / 2 * (pts[i+1] - pts[i]) for i in 1:length(pts)-1)
end

"""
    cab760m12hm3(; T = 25, cgd = :charge, RGext = 1.0) -> (; dev, RG, extraction)

The device data: the cube-law channel from Fig. 4 (one least-squares line of I^(1/3) against V_GS), R_d from the
finite-current channel voltage at 760 A, V_GS = 15 V (R_DS(on) 1.33 mOhm at 25 C; x 1.38 at 125 C, Fig. 3), C_ds = C_oss
- C_rss (Figs. 8-9), the body diode from Fig. 7, and the gate capacitances from Q_GS, Q_GD, Q_G (the table, 25 C) jointly
by one linear solve for (C_gs, C_ox, alpha): `cgd = :charge` scales the small-signal depletion C_rss by alpha so that
Q_GD is met (primary); `cgd = :smallsignal` keeps alpha = 1 (the declared variant). Nothing is fitted to Figs. 11, 15.
"""
function cab760m12hm3(; T::Real = 25, cgd::Symbol = :charge, RGext::Real = 1.0)
    T in (25, 125) || throw(ArgumentError("cab760m12hm3: T = 25 or 125 C (the digitised curves)"))
    cgd in (:charge, :smallsignal) || throw(ArgumentError("cab760m12hm3: cgd = :charge or :smallsignal"))
    D = JSON3.read(read(_CAB_DIGITISED, String))
    tab = D.table
    key = Symbol(string(Int(T), "C"))
    # channel (Fig. 4, V_DS = 20 V): I^(1/3) = a (V_GS - vth), beta = a^3
    tr = D.fig4_transfer_VDS20V[key]
    I = [Float64(p[1]) for p in tr]; V = [Float64(p[2]) for p in tr]
    y = cbrt.(I)
    A = hcat(V, ones(length(V)))
    sl, ic = A \ y
    beta = sl^3
    vth = -ic / sl
    # the 25 C channel for the plateau of the gate-charge extraction (the table's charges are at 25 C)
    tr25 = D.fig4_transfer_VDS20V[Symbol("25C")]
    y25 = cbrt.([Float64(p[1]) for p in tr25]); A25 = hcat([Float64(p[2]) for p in tr25], ones(length(tr25)))
    sl25, ic25 = A25 \ y25
    beta25, vth25 = sl25^3, -ic25 / sl25
    # drift resistance from the finite-current channel voltage (review 102, row 15)
    Ron = tab.Rds_on_25C_760A_mOhm * 1e-3 * (T == 125 ? 1.38 : 1.0)
    Id = 760.0
    vov15 = 15.0 - vth
    vch = vov15 - cbrt(vov15^3 - Id / beta)
    Rd = (Id * Ron - vch) / Id
    # capacitances (Figs. 8-9, 25 C, V_GS = 0; nF -> F)
    cap = D.fig8_9_capacitances_25C_VGS0
    vr = [Float64(p[1]) for p in cap.Crss]; cr = [Float64(p[2]) for p in cap.Crss] .* 1e-9
    vo = [Float64(p[1]) for p in cap.Coss]; co = [Float64(p[2]) for p in cap.Coss] .* 1e-9
    crss(v) = v <= vr[1] ? cr[1] : v >= vr[end] ? cr[end] : (k = searchsortedlast(vr, v); cr[k] + (cr[k+1] - cr[k]) * (v - vr[k]) / (vr[k+1] - vr[k]))
    # C_ds = C_oss - C_rss on the C_oss knots, constant below the first knot (1 V)
    cdsv = [co[k] - crss(vo[k]) for k in eachindex(vo)]
    cds = PWLCap(vcat([-50.0, 0.0], vo), vcat([cdsv[1], cdsv[1]], cdsv))
    # the gate-charge extraction: unknowns (Cgs, Cox, alpha); the plateau V_pl at 760 A (25 C channel), the on-state
    # v_ds at the plateau's end von = R_DS(on) 760 A, the test voltage 800 V; C_gd(v_gd) = alpha C_rss(-v_gd) for
    # v_gd <= -1 V, a linear joint to Cox at +1 V, Cox beyond
    Vt, Vg0, Vg1 = 800.0, -4.0, 15.0
    Vpl = vth25 + cbrt(Id / beta25)
    von = tab.Rds_on_25C_760A_mOhm * 1e-3 * Id
    QGS, QGD, QG = tab.QGS_nC * 1e-9, tab.QGD_nC * 1e-9, tab.QG_nC * 1e-9
    # int of C_gd over v_gd in [lo, hi] = alpha * dep(lo, hi) + Cox * pl(lo, hi) + c1 * ...: the joint (-1, alpha c1) to
    # (1, Cox) contributes alpha c1 (u..)/... ; computed exactly from the linear pieces
    c1 = crss(1.0)
    function gd_parts(lo, hi)        # returns (coefficient of alpha, coefficient of Cox) of int C_gd over [lo, hi]
        a = 0.0; b = 0.0
        # depletion: v_gd <= -1
        l, h = lo, min(hi, -1.0)
        h > l && (a += _pwl_int(vr, cr, -h, -l))
        # joint on [-1, 1]: C = alpha c1 (1 - u) / 2 + Cox (1 + u) / 2 with u = v_gd
        l, h = max(lo, -1.0), min(hi, 1.0)
        if h > l
            a += c1 * ((h - l) - (h^2 - l^2) / 2) / 2
            b += ((h - l) + (h^2 - l^2) / 2) / 2
        end
        l, h = max(lo, 1.0), hi
        h > l && (b += h - l)
        return a, b
    end
    # (1) Q_GS: v_gs -4 -> Vpl at v_ds = 800: Cgs (Vpl + 4) + C_gd part over v_gd in [-4 - 800, Vpl - 800]
    a1, b1 = gd_parts(Vg0 - Vt, Vpl - Vt)
    # (2) Q_G - Q_GS - Q_GD: v_gs Vpl -> 15 at v_ds = von: Cgs (15 - Vpl) + C_gd over [Vpl - von, 15 - von]
    a2, b2 = gd_parts(Vpl - von, Vg1 - von)
    # (3) Q_GD: v_ds 800 -> von at v_gs = Vpl: C_gd over [Vpl - 800, Vpl - von]
    a3, b3 = gd_parts(Vpl - Vt, Vpl - von)
    if cgd === :charge
        M = [Vpl - Vg0 b1 a1; Vg1 - Vpl b2 a2; 0.0 b3 a3]
        Cgs, Cox, alpha = M \ [QGS, QG - QGS - QGD, QGD]
    else
        alpha = 1.0
        M = [Vpl - Vg0 b1; Vg1 - Vpl b2]
        Cgs, Cox = M \ [QGS - a1, QG - QGS - QGD - a2]
    end
    (Cgs > 0 && Cox > 0 && alpha > 0) ||
        throw(ArgumentError("cab760m12hm3: the gate-charge extraction gave Cgs = $Cgs, Cox = $Cox, alpha = $alpha"))
    knee = T == 25 ? 2.5 : 2.3
    # the C_gd knots in v_gd (ascending): -V for the C_rss knots (above 1 V), then -1, 1 and beyond
    dep = [(-v, alpha * crss(v)) for v in reverse(vr) if v > 1.0]
    gx = vcat([p[1] for p in dep], [-1.0, 0.0, 1.0, 20.0])
    gc = vcat([p[2] for p in dep], [alpha * c1, (alpha * c1 + Cox) / 2, Cox, Cox])
    cgdc = PWLCap(gx, gc)
    # body diode (Fig. 7, V_GS = -4 V): the knee at the curve's foot, then the traced points
    bd = D.fig7_body_diode_VGSm4[key]
    bi = [Float64(p[1]) for p in bd]; bv = [Float64(p[2]) for p in bd]
    dev = SiCDevice("Wolfspeed CAB760M12HM3 (one position), T_vj $(T) C, C_gd $(cgd)", Float64(T), beta, vth, 0.05, Rd,
                    Cgs, cgdc, cds, vcat([knee], bv), vcat([0.0], bi), 1e-7, tab.RG_int_Ohm,
                    "data sheet Rev. 5 (2025), digitised: $(basename(_CAB_DIGITISED))")
    # checks from the constructed device (review 103, finding 10): the three charges recomputed from its capacitances;
    # the output capacitance it implies at V_GS = 0 (C_ds + C_gd(-v)) against the data's C_oss, with Q_oss, E_oss at 600 V
    qgd(v) = capq(cgdc, v)
    Q_GS_m = Cgs * (Vpl - Vg0) + qgd(Vpl - Vt) - qgd(Vg0 - Vt)
    Q_GD_m = qgd(Vpl - von) - qgd(Vpl - Vt)
    Q_post_m = Cgs * (Vg1 - Vpl) + qgd(Vg1 - von) - qgd(Vpl - von)
    coss_m(v) = capC(cds, v) + capC(cgdc, -v)
    coss_d(v) = v <= vo[1] ? co[1] : v >= vo[end] ? co[end] : (k = searchsortedlast(vo, v); co[k] + (co[k+1] - co[k]) * (v - vo[k]) / (vo[k+1] - vo[k]))
    vg = range(0.0, 600.0; length = 60001)
    Qoss_m = sum(coss_m((vg[i] + vg[i+1]) / 2) for i in 1:60000) * 0.01
    Eoss_m = sum(((vg[i] + vg[i+1]) / 2) * coss_m((vg[i] + vg[i+1]) / 2) for i in 1:60000) * 0.01
    Qoss_d = sum(coss_d((vg[i] + vg[i+1]) / 2) for i in 1:60000) * 0.01
    Eoss_d = sum(((vg[i] + vg[i+1]) / 2) * coss_d((vg[i] + vg[i+1]) / 2) for i in 1:60000) * 0.01
    checks = (; Q_GS = (Q_GS_m, QGS), Q_GD = (Q_GD_m, QGD), Q_post = (Q_post_m, QG - QGS - QGD),
              Coss = [(v, coss_m(v), coss_d(v)) for v in (1.0, 10.0, 100.0, 600.0)], Qoss600 = (Qoss_m, Qoss_d),
              Eoss600 = (Eoss_m, Eoss_d), Rd_positive = Rd > 0, diode_knots_ascending = issorted(vcat([knee], [Float64(p[2]) for p in D.fig7_body_diode_VGSm4[key]])))
    extraction = (; beta, vth, Rd, Cgs, Cox, alpha, Vpl, von, checks,
                  smallsignal_Crss_dep = _pwl_int(vr, cr, 1.0, Vt - Vpl))
    return (; dev, RG = Float64(RGext) + tab.RG_int_Ohm, extraction)
end
