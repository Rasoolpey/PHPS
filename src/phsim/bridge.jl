# The switching converter bridge with losses (upgrade 3 of 3, REALTIME_TODO section 3; reviews 77-77e).
#
# A two-level three-phase bridge of SiC MOSFET modules (Wolfspeed CAB760M12HM3, 1200 V / 760 A half-bridge, datasheet
# Rev. 5, Sep 2025; three modules in parallel per switch position), the aggregate of the N identical 720 kVA / 300 V AC
# battery-inverter units of a converter (record section 32; synchronised carriers: a coherent-ripple scenario).
#
# Poles (pu, AC phase-peak base; the DC link v_dc on the DC base 2 V_pk): upper on +v_dc - R_u i_k, lower on -v_dc - R_l
# i_k, dead time (both off) -sgn(i_k) (v_dc + v_SD(|i_k|)): the body diode the phase current i_k (out of the leg) flows
# through. Space vector of the poles u = (2/3) sum_k pole_k e^{j a_k}, a_k = 2 pi (k-1)/3 (stationary). Switched
# modulation m_sw = (2/3) sum_k p_k e^{j a_k} with p_k = +1, -1, -sgn(i_k); the DC-link current Re(m_sw conj(i)).
#
# Modulation: SVPWM by carrier comparison, a unit triangle of period T_c = 1 / f_c, regularly sampled at every peak and
# valley (the controllers' call, H = T_c / 2): duties d_k = (1 + r_k + r_0) / 2, r_k = Re(m e^{-j a_k}), r_0 = -(max r +
# min r) / 2, clipped to [0, 1]. Commanded state in a half: high while carrier < d_k; entering the half the right-hand
# limit (after a peak: d_k == 1; after a valley: d_k > 0) against the previous commanded state, then at most one interior
# crossing (falling half: on at (1 - d_k) H; rising half: off at d_k H) (77e). A commanded transition turns the conducting
# switch off at once and the complementary switch on t_d later; a pending turn-on is cancelled when the opposite command
# arrives at or before it (77d).
#
# Losses: conduction through the channel (both directions; position-dependent R_u, R_l); dead-time body-diode drop
# v_SD(|i| / 3) per module (Fig. 7, 125 C); switching energies per transition from Fig. 11 (25 C, 600 V) per module at
# |i| / 3, x 3 modules, scaled to T_vj by the table's ratios at 760 A and to v_dc by (V / 600 V)^1.32: hard turn-off (the
# switch carrying the current) E_off; hard turn-on (taking the current from the opposite body diode) E_on + E_rr; soft
# transitions 0 (directions smoothed by i / sqrt(i^2 + i_eps^2)).
#
# Numerics (fixed h): per step the exact time fractions of every pole state give one averaged modulation m-bar (in the
# synchronous frame the exact average of the rotating vector over each piece); the AC side applies v_dc,mid m-bar + the
# averaged drops, the DC link draws Re(m-bar conj(i_mid)) + the step's switching energy / (h v_dc,mid).

"""A SiC MOSFET half-bridge module's datasheet data (per module, SI units)."""
struct SiCModule
    name::String
    Rds::Float64          # R_DS(on) at T_vj (Ohm)
    Rpkg_u::Float64       # package resistance, upper position (Ohm)
    Rpkg_l::Float64       # lower position
    Ei::Vector{Float64}   # current points (A) of the energy tables
    Eon::Vector{Float64}  # J at 25 C, 600 V
    Eoff::Vector{Float64}
    Err::Vector{Float64}
    kon::Float64          # temperature ratios to T_vj (the table's at 760 A)
    koff::Float64
    krr::Float64
    Vref::Float64         # V
    kv::Float64           # voltage exponent
    vi::Vector{Float64}   # body-diode current points (A) and voltages (V) at T_vj, V_GS = -4 V
    vv::Vector{Float64}
    td::Float64           # dead time (s)
    source::String
    cv::Vector{Float64}   # C_oss: voltage points (V) and capacitance (F) per module (Fig. 9): the dead-time swing
    cc::Vector{Float64}
end
SiCModule(name, Rds, Ru, Rl, Ei, Eon, Eoff, Err, kon, koff, krr, Vref, kv, vi, vv, td, source) =
    SiCModule(name, Rds, Ru, Rl, Ei, Eon, Eoff, Err, kon, koff, krr, Vref, kv, vi, vv, td, source, Float64[], Float64[])

# the charge Q_oss(V) = int_0^V C_oss per module (trapezoids on the table; 0 without a table)
function _qoss(d::SiCModule, V)
    isempty(d.cv) && return zero(V)
    q = zero(V)
    for k in 1:length(d.cv)-1
        a, b = d.cv[k], d.cv[k+1]
        V <= a && break
        hi = min(V, b)
        cb = d.cc[k] + (d.cc[k+1] - d.cc[k]) * (hi - a) / (b - a)
        q += (d.cc[k] + cb) / 2 * (hi - a)
    end
    V > d.cv[end] && (q += d.cc[end] * (V - d.cv[end]))
    return q
end

# Wolfspeed CAB760M12HM3 at T_vj = 125 C (reviews 77b-77c): R_DS(on) 1.33 mOhm x 1.38 (Fig. 3); package 106.5 / 126.3 uOhm;
# E from Fig. 11 (25 C, 600 V; extrapolated linearly to 0 A, clipped at 0), x E(125 C) / E(25 C) at 760 A (table); E_rr
# from Fig. 11 / the table (1.3 mJ at 760 A, 25 C); v_SD from Fig. 7 at 300 dpi (calibrated against the 175 C table
# value); t_d = 300 ns (Fig. 24, 26: t_d(off) + t_f - t_d(on) ~ 225-250 ns, with margin; provisional).
const CAB760M12HM3_125C = SiCModule("Wolfspeed CAB760M12HM3 (Rev. 5, 2025), T_vj 125 C",
    1.33e-3 * 1.38, 106.5e-6, 126.3e-6,
    [0.0, 200.0, 400.0, 600.0, 800.0, 1000.0, 1200.0, 1400.0],
    [1.5, 6.0, 10.5, 15.5, 21.0, 27.0, 34.0, 42.0] .* 1e-3,
    [0.0, 3.0, 7.5, 13.0, 18.5, 25.0, 32.5, 41.0] .* 1e-3,
    [0.3, 0.6, 0.85, 1.1, 1.35, 1.6, 1.85, 2.1] .* 1e-3,
    20.7 / 20.3, 17.5 / 17.9, 3.5 / 1.3, 600.0, 1.32,
    [0.0, 5.0, 200.0, 400.0, 600.0, 760.0, 1000.0], [0.0, 2.4, 3.45, 4.0, 4.55, 4.86, 5.2],
    300e-9, "Wolfspeed CAB760M12HM3 data sheet Rev. 5 (Sep 2025): Fig. 3, 7, 9, 11-14, 24, 26 and the tables",
    [0.0, 15.0, 30.0, 50.0, 100.0, 200.0, 400.0, 600.0, 1200.0],
    [60.0, 23.0, 14.0, 10.6, 7.5, 4.9, 3.55, 2.98, 2.9] .* 1e-9)

# piecewise-linear interpolation, extrapolated with the last slope (AD-safe in x)
function _pwl(xs::Vector{Float64}, ys::Vector{Float64}, x)
    n = length(xs)
    k = 1
    while k < n - 1 && x > xs[k+1]
        k += 1
    end
    return ys[k] + (ys[k+1] - ys[k]) * (x - xs[k]) / (xs[k+1] - xs[k])
end

"""The state of one converter's switching bridge (outside the state vector: its schedule is set at the samples)."""
mutable struct SwitchingBridge
    dev::SiCModule
    npar::Int
    fc::Float64
    H::Float64              # half carrier period = the sampling period
    # per-unit conversion of this converter (system base)
    Ru::Float64             # upper / lower phase resistance (pu)
    Rl::Float64
    Ipk::Float64            # A per unit of system current at one unit (unit phase-peak base / Sr)
    Vpk::Float64            # V per pu of voltage (AC phase-peak base)
    Ej::Float64             # system pu s per joule per unit (Sr / S_unit)
    ieps::Float64           # clamp / direction smoothing current (system pu)
    omega_b::Float64
    stationary::Bool
    theta0::Float64
    # schedule: per leg, the physical pole states' change times and states (:U, :L, :D), sorted; the commanded state
    tchg::Vector{Vector{Float64}}
    schg::Vector{Vector{Symbol}}
    cmd::Vector{Bool}
    thalf::Float64          # start of the current half
    rising::Bool            # the current half rises (it started at a valley)
    active::Bool            # a run is going (false: the averaged lossless bridge, e.g. at the equilibrium)
    a::Float64              # the current step's window
    b::Float64
    uacc::ComplexF64        # the applied bridge voltage integrated over the current half (network frame; the measurement)
    tacc::Float64
    islope::ComplexF64      # the converter current's rate over the current step (network frame; 0: constant)
end

function SwitchingBridge(dev::SiCModule; npar::Int = 3, fc::Real = 10e3, Sr::Real, S_unit::Real = 720e3,
                         V_LL::Real = 300.0, mva_base::Real = 100.0, omega_b::Real, ieps_frac::Real = 0.01)
    Vpk = sqrt(2 / 3) * V_LL
    Ipk_u = S_unit / (1.5 * Vpk)                      # the unit's phase-peak current base (A)
    Zu = Vpk / Ipk_u
    Ru = (dev.Rds + dev.Rpkg_u) / npar / Zu / Sr      # system pu
    Rl = (dev.Rds + dev.Rpkg_l) / npar / Zu / Sr
    Ej = Sr * mva_base * 1e6 / S_unit / (mva_base * 1e6)   # N joules per unit / S_sys = Sr / S_unit
    return SwitchingBridge(dev, npar, Float64(fc), 1 / (2fc), Ru, Rl, Ipk_u / Sr, Vpk, Ej, ieps_frac * Sr,
                           Float64(omega_b), false, 0.0, [Float64[] for _ in 1:3], [Symbol[] for _ in 1:3],
                           falses(3), 0.0, false, false, 0.0, 0.0, 0.0im, 0.0, 0.0im)
end

const _LEG = (0.0, 2pi / 3, -2pi / 3)
const BRIDGE_DROP_PIECE = Ref(false)   # (diagnostic, review 78: the drops at the piece currents instead of the midpoint's)

# the leg duties of a stationary-frame modulation m (SVPWM: min-max zero sequence), clipped
function _duties(m::Complex)
    r = [real(m * cis(-a)) for a in _LEG]
    r0 = -(maximum(r) + minimum(r)) / 2
    return [clamp((1 + rk + r0) / 2, 0.0, 1.0) for rk in r]
end

# the physical state of leg k at time t (the last change at or before t)
function _leg_state(br::SwitchingBridge, k, t)
    ts, ss = br.tchg[k], br.schg[k]
    j = searchsortedlast(ts, t)
    return j == 0 ? ss[1] : ss[j]
end

# a commanded transition of leg k to `high` at tc
function _command!(br::SwitchingBridge, k, high::Bool, tc)
    ts, ss = br.tchg[k], br.schg[k]
    # cancel every pending change at or after tc (a pending turn-on of the other polarity: 77d item 2)
    j = searchsortedfirst(ts, tc)
    resize!(ts, j - 1); resize!(ss, j - 1)
    now = isempty(ss) ? :D : ss[end]
    br.cmd[k] = high
    target = high ? :U : :L
    now == target && return
    if now != :D
        push!(ts, tc); push!(ss, :D)                  # the conducting switch off at once
    end
    push!(ts, tc + br.dev.td); push!(ss, target)      # the complementary switch on t_d later
    return
end

"""
    bridge_start!(br, m_stat, t0; stationary, theta0)

Start a run at t0 (a carrier peak): every leg in the state the comparison gives (no dead time, no energy at t0), and the
first (falling) half's interior edges.
"""
function bridge_start!(br::SwitchingBridge, m_stat::Complex, t0::Real; stationary::Bool, theta0::Real)
    br.stationary, br.theta0 = stationary, theta0
    d = _duties(m_stat)
    for k in 1:3
        s0 = d[k] == 1.0
        br.tchg[k] = [Float64(t0) - 1.0]; br.schg[k] = [s0 ? :U : :L]; br.cmd[k] = s0
    end
    br.thalf, br.rising, br.active = Float64(t0), false, true
    _interior!(br, d)
    br.a, br.b = Float64(t0), Float64(t0) + br.H     # a window until the first step sets its own (the Newton matrix)
    br.uacc, br.tacc = 0.0im, 0.0
    return br
end

# the interior crossings of the current half
function _interior!(br::SwitchingBridge, d)
    for k in 1:3
        0.0 < d[k] < 1.0 || continue
        br.rising ? _command!(br, k, false, br.thalf + d[k] * br.H) : _command!(br, k, true, br.thalf + (1 - d[k]) * br.H)
    end
end

"""
    bridge_sample!(br, m_stat)

The sample at the start of the next half (a peak after a rising half, a valley after a falling one): the right-hand-limit
state against the commanded state (a boundary command if they differ), then the half's interior crossings (77e).
"""
bridge_sample!(br::SwitchingBridge, m_stat::Complex; prune::Bool = true) = bridge_sample_duties!(br, _duties(m_stat); prune)

"""The sample with given leg duties (for tests of the event rules; `bridge_sample!` takes them from the modulation)."""
function bridge_sample_duties!(br::SwitchingBridge, d::AbstractVector; prune::Bool = true)
    br.thalf += br.H
    br.rising = !br.rising
    for k in 1:3
        s0 = br.rising ? d[k] > 0.0 : d[k] == 1.0
        s0 != br.cmd[k] && _command!(br, k, s0, br.thalf)
    end
    _interior!(br, d)
    prune || return br
    # drop change times long past (keep the last one before the half for the state lookup)
    for k in 1:3
        j = searchsortedlast(br.tchg[k], br.thalf - 2br.H)
        if j > 1
            deleteat!(br.tchg[k], 1:j-1); deleteat!(br.schg[k], 1:j-1)
        end
    end
    return br
end

# the average over [ta, tb] (times from the start of the dead time) of min(tau / ts, 1): the fraction of the swing done
@inline function _ramp_avg(ta, tb, ts)
    f(t) = t <= ts ? t^2 / (2ts) : ts / 2 + (t - ts)      # the integral of min(t / ts, 1) from 0
    return (f(tb) - f(ta)) / (tb - ta)
end

bridge_window!(br::SwitchingBridge, a::Real, b::Real) = (br.a = Float64(a); br.b = Float64(b); br.islope = 0.0im; br)

# the frame factor's integral over [ta, tb] / h: network-frame average of a constant stationary vector
@inline function _rot_avg(br::SwitchingBridge, ta, tb, h)
    br.stationary && return (tb - ta) / h
    w = br.omega_b
    return (cis(-(w * tb + br.theta0)) - cis(-(w * ta + br.theta0))) / (-im * w * h)
end

"""
    bridge_average(br, i_net, v_dc) -> (; m, udrop, psw, pcond)

The step's averages over the window [a, b] at the midpoint current i_net (network frame) and DC voltage v_dc (system
pu): the averaged modulation m (network frame; the AC voltage is v_dc m + udrop and the DC current Re(m conj(i))), the
averaged drops udrop (resistances and body diodes), the switching power psw (the window's transitions' energy / h) and
the conduction loss pcond = -Re(udrop conj(i)) (>= 0).
"""
function bridge_average(br::SwitchingBridge, i_net, v_dc)
    a, b = br.a, br.b
    h = b - a
    T = typeof(real(i_net) + v_dc)
    tm = (a + b) / 2
    m = zero(Complex{T}); ud = zero(Complex{T}); esw = zero(T)
    dev = br.dev
    vphys = v_dc * 2 * br.Vpk                          # the DC voltage (V)
    vscale = (vphys / dev.Vref)^dev.kv                 # against the tables' 600 V
    # the phase current of leg k at time t: the midpoint current carried linearly with the step's secant (`islope`, set
    # by the simulator from the Newton iterate), in the stationary frame
    ileg(k, t) = (ist = (i_net + (t - tm) * br.islope) * (br.stationary ? 1.0 : cis(br.omega_b * t + br.theta0));
                  real(ist * cis(-_LEG[k])))
    rotm = br.stationary ? 1.0 + 0im : cis(-(br.omega_b * tm + br.theta0))
    for k in 1:3
        ek = cis(_LEG[k]) * (2 / 3)
        ts, ss = br.tchg[k], br.schg[k]
        ikm = ileg(k, tm)                               # the leg's midpoint current, its direction and diode drop
        sgm = sign(ikm)                                 # (hard: the drop is continuous since v_SD(0) = 0; review 78b, 7)
        vsdm = _pwl(dev.vi, dev.vv, abs(ikm) * br.Ipk / br.npar) / br.Vpk
        # the pieces of the window: the state just before a, then every change in [a, b) (a change exactly at a step
        # boundary belongs to the step that starts there, 77c item 4); each piece's current-dependent terms at the
        # piece's own midpoint current, each transition's energy at the current at its instant
        t = a
        j = searchsortedfirst(ts, a)
        st = j > 1 ? ss[j-1] : ss[1]
        # a dead time carried into the window: its start and the rail it left
        tD0, p0 = st === :D && j > 2 ? (ts[j-1], ss[j-2] === :U ? 1.0 : -1.0) : (a, 0.0)
        while true
            tn = (j <= length(ts) && ts[j] < b) ? ts[j] : b
            if tn > t
                ik = ileg(k, (t + tn) / 2)              # the piece's own current: the clamp's direction (lossless part)
                r = _rot_avg(br, t, tn, h)
                rd = (tn - t) / h * rotm                # the drops: the piece's fraction at the midpoint's rotation
                # the drops (the dissipative part) at the leg's midpoint current, so that the conduction loss
                # -Re(udrop conj(i_mid)) is a sum of non-negative terms (review 78, item 2)
                ikd = BRIDGE_DROP_PIECE[] ? ik : ikm
                if st === :U
                    m += ek * r; ud -= ek * (br.Ru * ikd) * rd
                elseif st === :L
                    m -= ek * r; ud -= ek * (br.Rl * ikd) * rd
                else
                    # dead time (reviews 78-78b): the pole leaves its rail p0 towards the rail pT = -sgn(i) as the phase
                    # current moves the output capacitances' charge, then the body diode clamps it; at zero current it
                    # stays at p0. An effective ramp approximation: progress tau |i| / (2 npar Q_oss(V)) with the present
                    # current (no accumulated-charge state: a current reversal inside the dead time is not followed, and
                    # the capacitors' stored energy is not tracked)
                    pT = ik >= 0 ? -1.0 : 1.0
                    if p0 == pT || isempty(dev.cv)
                        fr = one(T)                     # already at (or, without C_oss data, at once at) the target rail
                    else
                        qsw = 2 * br.npar * _qoss(dev, vphys)
                        tsw = qsw / max(abs(ik) * br.Ipk, 1e-9)
                        fr = _ramp_avg(t - tD0, tn - tD0, tsw)
                    end
                    pav = p0 + (pT - p0) * fr           # the pole's average position (pu of v_dc)
                    m += ek * pav * r
                    # the clamp's diode drop (dissipative) while clamped, at the midpoint current: fraction clamped there
                    if BRIDGE_DROP_PIECE[]                 # (diagnostic: the clamp's drop at the piece current)
                        frp = (p0 == pT || isempty(dev.cv)) ? 1.0 :
                              (tsp = 2 * br.npar * _qoss(dev, vphys) / max(abs(ik) * br.Ipk, 1e-9);
                               max(0.0, (tn - tD0) - max(t - tD0, tsp)) / (tn - t))
                        ud -= ek * (-pT * _pwl(dev.vi, dev.vv, abs(ik) * br.Ipk / br.npar) / br.Vpk * frp) * rd
                    else
                        frm = (p0 == pT || isempty(dev.cv)) ? 1.0 :
                              (tsm = 2 * br.npar * _qoss(dev, vphys) / max(abs(ikm) * br.Ipk, 1e-9);
                               max(0.0, (tn - tD0) - max(t - tD0, tsm)) / (tn - t))
                        ud -= ek * (sgm * vsdm * frm) * rd
                    end
                end
            end
            tn >= b && break
            snew = ss[j]                                # a transition at tn
            ik = ileg(k, tn); sg = ik / sqrt(ik^2 + br.ieps^2); iA = abs(ik) * br.Ipk / br.npar
            eon() = max(_pwl(dev.Ei, dev.Eon, iA), 0.0) * dev.kon + max(_pwl(dev.Ei, dev.Err, iA), 0.0) * dev.krr
            eoff() = max(_pwl(dev.Ei, dev.Eoff, iA), 0.0) * dev.koff
            if st === :U && snew === :D                 # upper off: hard if the current flows out (i > 0)
                esw += (1 + sg) / 2 * eoff()
            elseif st === :L && snew === :D             # lower off: hard if i < 0
                esw += (1 - sg) / 2 * eoff()
            elseif st === :D && snew === :U             # upper on: hard if i > 0 (takes it from the lower diode)
                esw += (1 + sg) / 2 * eon()
            elseif st === :D && snew === :L             # lower on: hard if i < 0
                esw += (1 - sg) / 2 * eon()
            end
            snew === :D && ((tD0, p0) = (tn, st === :U ? 1.0 : -1.0))   # a dead time begins: its start and the rail left
            st = snew; t = tn; j += 1
        end
    end
    psw = esw * br.npar * vscale * br.Ej / h
    pcond = -real(ud * conj(i_net))                    # = sum_k frac (2/3) (R ikm^2 + sgm vsd ikm) >= 0 exactly
    return (; m, udrop = ud, psw, pcond)
end
