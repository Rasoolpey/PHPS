# Physical steam units of the EMT simulator (PH_SIMULATOR_TODO.md, step G1; derivation and data in
# docs/active/PH_STEAM_UNIT_MODEL.md, sections 3, 4, 13, 14, 15).
#
# The device a case's IEEEG1 stands for: the speed governor and lead-lag (signal law) and the
# valve servomotor (IEEEG1's own rows and limiters, states x0, x1 in power units), a control valve
# from a boiler at fixed pressure and enthalpy into a steam chest, and the turbine: storages
# (steam chest, reheaters, crossover, and a small node at every extraction point, Ray 1980 eq.
# (11)) linked by turbine sections (ellipse law, Fuls 2017 eq. (1), Yu 2020 eq. (9); constant
# efficiency, Ray 1980 Appendix 1) and by the reheaters' inlet resistances (the 1991 IEEE working
# group's square-root law, Fig. 12). The last section exhausts to the condenser at fixed pressure.
#
#   storage k:  states M_k, U_k (mass, internal energy), volume V_k; state from IAPWS-95 tables
#               M_k' = sum of inflows - outflows - extraction,  U_k' = sum q h in - out + Q_k
#   valve:      q_v = c_v y beta(eps),   eps = p_chest / p_T,  y = clamp(x1, PMIN, PMAX) / PMAX
#               beta: 1 below eps_cr, Cooke's projected ellipse above (Yu 2020 eq. (1)-(5))
#   section:    q = K sq(p_in^2 - p_out^2) / sqrt(p_in v_in),  h_out = h_in - eta (h_in - h_is)
#               work q (h_in - h_out) to the shaft
#   resistance: q = K_r sq(p_in - p_out), isenthalpic
#   extraction: beta_k times the storage's inflow, leaving at the storage's state
#   heat:       Q_k = k_k (T_w,k - T_k) into a reheater, T_w,k set per initial equilibrium
#
# sq(D) = D / (D^2 + delta^2)^(1/4) is the square root of D for |D| >> delta and linear near zero
# (a declared low-flow regime, PH_STEAM_UNIT_MODEL.md section 15, GPT consultation 6, point 10);
# delta is the parameter `lowflow` times the nominal value of D. It is odd: a flow can reverse,
# and then carries the receiving storage's enthalpy back and does no work (donor cell).
#
# Energy: the unit's stored energy is the sum of the U_k, a sum of states, so the implicit
# midpoint rule keeps its first law exactly at every step (record section 15.1). Supplies: the
# boiler (q_v h_T), the reheaters' heat; outflows: the condenser, the extractions; the shaft power
# goes to the machine. No mechanical loss (record section 3.5).
#
# Units: SI inside; the machine receives the shaft power on the system base, Tm = W / S_b (the
# machine divides by its speed). A unit of the case is `n` published units in parallel (record
# section 6): volumes, flow constants, valve area and heat conductances are multiplied by `n`.

const STEAM_SB = 100.0e6          # W, the system base of IEEE-39
const STEAM_EPS_CR = 0.5457       # superheated steam's critical pressure ratio (Fuls 2017, section 4.1)
const STEAM_VALVE_K = 0.647       # 0.97 x 0.667 (Yu 2020, eq. (5))

# ------------------------------------------------------------------------------------------------
# The published units (record sections 5, 13, 14). Pressures Pa, enthalpies J/kg, temperatures K,
# flows kg/s. A storage's nominal state is given by (p, h), (p, T) or (p, x); `V` its volume (m3)
# for a vessel, `nothing` for a node (sized by the residence time); `heated` for a reheater.

# Vahidi, Bank Tavakoli and Gawlik, IEEE Trans. Power Syst. 22(4), 2007, Fig. 7 (200 MW, single
# reheat), with Vahidi's conversions; the map's unexplained flows as declared extractions (13.6).
function _steam_data_A(; cold_reheat::Real)
    kgf(x) = x * 98.07e3
    th(x) = x * 0.278
    kcal(x) = x * 4.186e3
    cold_reheat in (26.2, 28.22) || throw(ArgumentError("path A: cold_reheat must be 26.2 or 28.22 kgf/cm2 (record 5.1, 13.3)"))
    st = [(name = "chest", kind = :vessel, V = 0.83, p = nothing, h = kcal(822.6), T = nothing, x = nothing, heated = false),
          (name = "hp_ext", kind = :node, V = nothing, p = kgf(41.5), h = kcal(757), T = nothing, x = nothing, heated = false),
          (name = "cold_reheat", kind = :node, V = nothing, p = kgf(cold_reheat), h = kcal(736), T = nothing, x = nothing, heated = false),
          (name = "reheater", kind = :vessel, V = 135.72, p = kgf(24.22), h = kcal(848.2), T = nothing, x = nothing, heated = true),
          (name = "ip_ext1", kind = :node, V = nothing, p = kgf(12.5), h = kcal(803), T = nothing, x = nothing, heated = false),
          (name = "ip_ext2", kind = :node, V = nothing, p = kgf(6.4), h = kcal(758), T = nothing, x = nothing, heated = false),
          (name = "ip_ext3", kind = :node, V = nothing, p = kgf(2.76), h = kcal(711), T = nothing, x = nothing, heated = false),
          (name = "crossover", kind = :vessel, V = 10.26, p = kgf(1.27), h = kcal(674.5), T = nothing, x = nothing, heated = false),
          (name = "lp_ext", kind = :node, V = nothing, p = kgf(0.35), h = kcal(627), T = nothing, x = nothing, heated = false)]
    q_rh = th(543.9)
    sections = [(name = "HP1", up = 1, down = 2, q = th(639.1)),
                (name = "HP2", up = 2, down = 3, q = th(611.2)),
                (name = "IP1", up = 4, down = 5, q = q_rh),
                (name = "IP2", up = 5, down = 6, q = q_rh - th(19.45)),
                (name = "IP3", up = 6, down = 7, q = q_rh - th(19.45 + 25.8)),
                (name = "IP4", up = 7, down = 8, q = q_rh - th(19.45 + 25.8 + 18.4)),
                (name = "LP1", up = 8, down = 9, q = th(435.7)),
                (name = "LP2", up = 9, down = 0, q = th(435.7 - 16.99))]
    resist = [(name = "reheater_inlet", up = 3, down = 4, q = q_rh)]
    # extraction from each storage (kg/s): what leaves besides the downstream links
    ext = [0.0, th(27.9), th(611.2) - q_rh, 0.0, th(19.45), th(25.8), th(18.4),
           q_rh - th(19.45 + 25.8 + 18.4) - th(435.7), th(16.99)]
    return (; path = :A, source = "Vahidi et al. 2007, Fig. 7", throttle = (p = kgf(130), h = kcal(822.6), T = nothing),
            storages = st, sections, resist, ext, p_cond = kgf(0.09), h_exhaust = kcal(595), q_valve = th(639.1))
end

# Zhao et al., Energy 119 (2017) 540-548, Tables 1, 2, 4, and Si et al., Energy Convers. Manag. 147
# (2017) 155-165, Tables 6, 7 (1000 MW, double reheat): design states; the LP exhaust from Si's LP
# work (a declared hybrid calibration, record 14.3). No volumes in the sources: `V_chest`,
# `V_rh1`, `V_rh2`, `V_crossover` are required (record 14.5 gives two readings).
function _steam_data_C(; V_chest::Real, V_rh1::Real, V_rh2::Real, V_crossover::Real)
    bar(x) = x * 1e5
    C(x) = x + 273.15
    st = [(name = "chest", kind = :vessel, V = Float64(V_chest), p = nothing, h = nothing, T = C(600.0), x = nothing, heated = false),
          (name = "vhp_exhaust", kind = :node, V = nothing, p = bar(112.70), h = nothing, T = C(430.3), x = nothing, heated = false),
          (name = "reheater1", kind = :vessel, V = Float64(V_rh1), p = bar(105.4), h = nothing, T = C(610.0), x = nothing, heated = true),
          (name = "hp_ext", kind = :node, V = nothing, p = bar(62.60), h = nothing, T = C(527.1), x = nothing, heated = false),
          (name = "hp_exhaust", kind = :node, V = nothing, p = bar(35.80), h = nothing, T = C(434.0), x = nothing, heated = false),
          (name = "reheater2", kind = :vessel, V = Float64(V_rh2), p = bar(32.14), h = nothing, T = C(610.0), x = nothing, heated = true),
          (name = "ip_ext1", kind = :node, V = nothing, p = bar(18.85), h = nothing, T = C(526.3), x = nothing, heated = false),
          (name = "ip_ext2", kind = :node, V = nothing, p = bar(11.30), h = nothing, T = C(447.0), x = nothing, heated = false),
          (name = "ip_ext3", kind = :node, V = nothing, p = bar(7.82), h = nothing, T = C(393.8), x = nothing, heated = false),
          (name = "crossover", kind = :vessel, V = Float64(V_crossover), p = bar(4.26), h = nothing, T = C(312.5), x = nothing, heated = false),
          (name = "lp_ext1", kind = :node, V = nothing, p = bar(1.34), h = nothing, T = C(191.3), x = nothing, heated = false),
          (name = "lp_ext2", kind = :node, V = nothing, p = bar(0.62), h = nothing, T = C(116.1), x = nothing, heated = false),
          (name = "lp_ext3", kind = :node, V = nothing, p = bar(0.23), h = nothing, T = nothing, x = 0.982, heated = false)]
    q0 = 730.4
    q1 = q0 - 51.0                 # into reheater 1
    q2 = q1 - 56.2                 # HP 2
    q3 = q2 - 40.5                 # into reheater 2
    q4 = q3 - 20.1; q5 = q4 - 51.8; q6 = q5 - 21.2
    q7 = q6 - 31.9                 # into the LPs
    q8 = q7 - 17.2; q9 = q8 - 19.9; q10 = q9 - 21.7
    sections = [(name = "VHP", up = 1, down = 2, q = q0),
                (name = "HP1", up = 3, down = 4, q = q1),
                (name = "HP2", up = 4, down = 5, q = q2),
                (name = "IP1", up = 6, down = 7, q = q3),
                (name = "IP2", up = 7, down = 8, q = q4),
                (name = "IP3", up = 8, down = 9, q = q5),
                (name = "IP4", up = 9, down = 10, q = q6),
                (name = "LP1", up = 10, down = 11, q = q7),
                (name = "LP2", up = 11, down = 12, q = q8),
                (name = "LP3", up = 12, down = 13, q = q9),
                (name = "LP4", up = 13, down = 0, q = q10)]
    resist = [(name = "reheater1_inlet", up = 2, down = 3, q = q1),
              (name = "reheater2_inlet", up = 5, down = 6, q = q3)]
    ext = [0.0, 51.0, 0.0, 56.2, 40.5, 0.0, 20.1, 51.8, 21.2, 31.9, 17.2, 19.9, 21.7]
    return (; path = :C, source = "Zhao et al. 2017, Tables 1, 2, 4; Si et al. 2017, Tables 6, 7",
            throttle = (p = bar(310.0), h = nothing, T = C(600.0)), storages = st, sections, resist, ext,
            p_cond = bar(0.045), h_exhaust = 2409.9e3, q_valve = q0)
end

"""
    steam_volumes_from_time_constants(; tau, path = :C) -> volumes

The volume (m3) a given isothermal time constant `V (drho/dp)_T p0 / q0` implies for each vessel of
the published unit of `path` at its design state (record 14.5, 16.6): for path C `tau = (chest,
rh1, rh2, crossover)`, for path A `tau = (chest, reheater, crossover)`, in seconds. `q0` is the
vessel's outflow at the design point. The chest is taken at the throttle's state, or with `r_v` at the
unit's design chest state (the throttle's enthalpy at `r_v` times its pressure), as `PHSteamUnit`
does for `volumes = :case`; `tables` defaults to `steam_tables()`. A reading, not a source; with the case's IEEEG1 time
constants (`volumes = :case`, record 16.6) the vessels are sized to the case's data.
"""
function steam_volumes_from_time_constants(; tau, path::Symbol = :C, r_v = nothing, tables = nothing)
    if path === :C
        d = _steam_data_C(; V_chest = 1.0, V_rh1 = 1.0, V_rh2 = 1.0, V_crossover = 1.0)
        idx = (1, 3, 6, 10)
        q = (d.q_valve, d.sections[2].q, d.sections[4].q, d.sections[8].q)
    elseif path === :A
        d = _steam_data_A(; cold_reheat = 26.2)
        idx = (1, 4, 8)
        q = (d.q_valve, d.sections[3].q, d.sections[7].q)
    else
        throw(ArgumentError("steam_volumes_from_time_constants: path must be :A or :C"))
    end
    length(tau) == length(idx) && all(t -> isfinite(t) && t > 0, tau) ||
        throw(ArgumentError("steam_volumes_from_time_constants: path $path needs $(length(idx)) positive time constants"))
    tb = tables === nothing ? steam_tables() : tables
    V = Float64[]
    for (k, i) in enumerate(idx)
        s = d.storages[i]
        p = i == 1 ? d.throttle.p : s.p
        ns_ = s.T !== nothing ? _steam_nominal(tb, p; T = s.T) : _steam_nominal(tb, p; h = s.h)
        if i == 1 && r_v !== nothing
            # the chest at the unit's design state: the throttle's enthalpy at r_v times its pressure, as
            # PHSteamUnit places it (consultation 10, point 7)
            thr = d.throttle.h === nothing ? _steam_nominal(tb, d.throttle.p; T = d.throttle.T) :
                  _steam_nominal(tb, d.throttle.p; h = d.throttle.h)
            p = r_v * d.throttle.p
            ns_ = _steam_nominal(tb, p; h = steam_state(tb, thr.rho, thr.u).h)
        end
        T = steam_state(tb, ns_.rho, ns_.u).T
        drdp = 1 / ForwardDiff.derivative(r -> _ps_core(T, r).p, ns_.rho)
        push!(V, tau[k] * q[k] / (p * drdp))
    end
    return path === :C ? (; chest = V[1], rh1 = V[2], rh2 = V[3], crossover = V[4]) :
           (; chest = V[1], reheater = V[2], crossover = V[3])
end

# saturated vapour density at p (kernel, for brackets only)
_sat_rho_g(p) = 1 / steam_sat(steam_tables(), p).v_g

# ------------------------------------------------------------------------------------------------
# A nominal state (rho, u) from (p, h), (p, T) or (p, x), by the IAPWS-95 kernel (vapour) or the
# saturation table (wet); used once, to place the published unit's design point in the tables.
function _steam_nominal(tb::SteamTables, p; h = nothing, T = nothing, x = nothing)
    if x !== nothing
        s = steam_sat(tb, p)
        v = s.v_f + x * (s.v_g - s.v_f)
        return (rho = 1 / v, u = s.u_f + x * (s.u_g - s.u_f))
    end
    # above the saturation table (and above the critical pressure) there is no dome: the density's
    # bracket is the liquid-like limit, and the temperature search starts at the critical one
    supercrit = p >= exp(tb.sat.x0 + (tb.sat.n - 1) * tb.sat.dx)
    rhohi(pp) = supercrit ? 1000.0 : 1.0001 / steam_sat(tb, pp).v_g
    if T === nothing
        if supercrit
            TT = _PS.IAPWS95_TC + 20.0
        else
            s = steam_sat(tb, p)
            # wet, from the enthalpy, with h = u + p v as steam_state has it (consultation 9, point 12)
            hf, hg = s.u_f + p * s.v_f, s.u_g + p * s.v_g
            if h <= hg
                xx = (h - hf) / (hg - hf)
                0 <= xx || throw(DomainError(h, "PHSteamUnit: below the saturated liquid at p = $p"))
                return _steam_nominal(tb, p; x = xx)
            end
            TT = s.T + 1.0
        end
        for _ in 1:100
            r = _vapour_rho(p, TT; rhohi = rhohi(p))
            st = PhysicalSteam.iapws95_properties(TT, r)
            dT = (st.enthalpy - h) / st.cp
            TT -= dT
            abs(dT) < 1e-10 && break
        end
        T = TT
        abs(_ps_core(T, _vapour_rho(p, T; rhohi = rhohi(p))).h - h) <= 1e-9 * abs(h) ||
            error("PHSteamUnit: no state at p = $p, h = $h")
    end
    r = _vapour_rho(p, T; rhohi = rhohi(p))
    return (rho = r, u = _ps_core(T, r).u)
end

# ------------------------------------------------------------------------------------------------
"""
    PHSteamUnit

A physical steam unit on its machine's shaft (file header). IEEEG1's signal law and servo
constants `K, T1, T2, T3, UO, UC, PMAX, PMIN, R_base`; the steam network, already scaled by `n`:
storage `names`, volumes `V`, extraction fractions `beta`, heat conductances `kh` and wall
temperatures `Tw` (set at initialisation); the valve constant `cv` (kg/s at full stroke and
choked flow), the throttle state; sections `(up, down, K, eta, delta)`, resistances
`(up, down, K, delta)`, the valve's `delta`; the condenser pressure; the steam tables. Local states
`x0, x1`, then `M, U` of every storage. With `valve = :characteristic` the servo state `x1` is a
flow demand in power units, as in IEEEG1, and the opening is `vc(x1)`, the inverse of the unit's own
steady characteristic (the turbine control logic's function generator, IEEE working group 1991;
record 16.6); with `valve = :opening` it is `x1 / PMAX`.
"""
struct PHSteamUnit
    name::String
    path::Symbol
    K::Float64
    T1::Float64
    T2::Float64
    T3::Float64
    UO::Float64
    UC::Float64
    PMAX::Float64
    PMIN::Float64
    R_base::Float64
    n::Float64
    names::Vector{String}
    V::Vector{Float64}
    beta::Vector{Float64}
    kh::Vector{Float64}
    Tw::Vector{Float64}
    cv::Float64
    p_T::Float64
    h_T::Float64
    eps_cr::Float64
    delta_v::Float64
    sec_up::Vector{Int}
    sec_down::Vector{Int}
    sec_K::Vector{Float64}
    sec_eta::Vector{Float64}
    sec_delta::Vector{Float64}
    sec_names::Vector{String}
    res_up::Vector{Int}
    res_down::Vector{Int}
    res_K::Vector{Float64}
    res_delta::Vector{Float64}
    p_cond::Float64
    isnode::Vector{Bool}         # the storages that are extraction nodes (sized by tau_n)
    nominal::NamedTuple          # the published unit's design point, scaled: states, flows, powers
    tb::SteamTables
    valve::Symbol                # :characteristic (x1 a flow demand) or :opening (x1 / PMAX the opening)
    vc_d::Vector{Float64}        # the valve characteristic (record 16.6), built by steam_initialise:
    vc_y::Vector{Float64}        #   demand nodes (power units), openings, and slopes dy/dd
    vc_s::Vector{Float64}
    volumes::Dict{String,Any}    # what the vessels' volumes were taken from, and the fit (record 16.6, 16.7)
end

nstates(g::PHSteamUnit) = 2 + 2 * length(g.V)
name(g::PHSteamUnit) = g.name
state_names(g::PHSteamUnit) = vcat(["x0", "x1"], [s * suffix for s in g.names for suffix in (".M", ".U")])

# the square root with a linear low-flow regime, odd in D
@inline _sqd(D, delta) = D / sqrt(sqrt(D^2 + delta^2))

"""
    PHSteamUnit(gov::IEEEG1_PHTRUE; path, r_v, rh_dT, tau_n, lowflow, cold_reheat = nothing,
                volumes = nothing, tables = steam_tables())

The physical steam unit in place of the case's IEEEG1 `gov`: its signal law and servo constants,
and the published unit of `path` (`:A` Vahidi's 200 MW single-reheat unit, `:C` Zhao and Si's
1000 MW double-reheat unit) scaled to `gov`'s `PMAX` (record sections 6, 13.9, 14.8). Numbers no
source gives, all required: `r_v` the valve's pressure ratio at rated flow and full stroke;
`rh_dT` the reheater wall's temperature excess over the steam at the design point (K); `tau_n`
the extraction nodes' residence time at the design point (s); `lowflow` the low-flow regime's
width as a fraction of each square root's nominal argument; for path A `cold_reheat` (26.2 or
28.22 kgf/cm2); for path C `volumes = (; chest, rh1, rh2, crossover)` in m3 of the published unit.
"""
function PHSteamUnit(gov::IEEEG1_PHTRUE; path::Symbol, r_v::Real, rh_dT::Real, tau_n::Real, lowflow::Real,
                     cold_reheat = nothing, volumes = nothing, valve::Symbol = :characteristic,
                     tables::SteamTables = steam_tables())
    p = gov.p
    valve in (:characteristic, :opening) ||
        throw(ArgumentError("PHSteamUnit $(gov.name): valve must be :characteristic or :opening"))
    # the vessels' volumes from the case's own IEEEG1 time constants (record 16.6): path C's chest,
    # reheaters and crossover from T4 to T7, path A's chest, reheater and crossover from T4 to T6
    # `:fit` the volumes fitted to the case's IEEEG1 turbine by its step response (record 16.7, the selected
    # values); `:fit_frequency` by its frequency response (record 16.8, an alternative)
    fit = volumes in (:fit, :fit_frequency)
    objective = volumes === :fit_frequency ? "frequency" : "step"
    fit && valve !== :characteristic &&
        throw(ArgumentError("PHSteamUnit $(gov.name): volumes = :fit needs valve = :characteristic"))
    fit && (volumes = :case)
    vsource = fit ? "fitted to the case's IEEEG1 turbine ($objective response) at initialisation, from the case's time constants" :
              volumes === :case ? (path === :A ? "case IEEEG1 T4-T6" : "case IEEEG1 T4-T7") :
              volumes === nothing ? "published" : "given"
    if volumes === :case
        volumes = path === :C ? steam_volumes_from_time_constants(; tau = (p.T4, p.T5, p.T6, p.T7), path = :C, r_v, tables) :
                  path === :A ? steam_volumes_from_time_constants(; tau = (p.T4, p.T5, p.T6), path = :A, r_v, tables) : volumes
    end
    if path === :A && volumes !== nothing
        (volumes isa NamedTuple && keys(volumes) == (:chest, :reheater, :crossover)) ||
            throw(ArgumentError("PHSteamUnit $(gov.name): path A's volumes are (; chest, reheater, crossover)"))
        all(v -> isfinite(v) && v > 0, values(volumes)) ||
            throw(ArgumentError("PHSteamUnit $(gov.name): path A's volumes must be positive and finite"))
    end
    (0 < r_v < 1 && rh_dT > 0 && tau_n > 0 && lowflow > 0) ||
        throw(ArgumentError("PHSteamUnit $(gov.name): r_v must lie in (0, 1); rh_dT, tau_n, lowflow must be positive"))
    (p.PMAX > p.PMIN >= 0 && p.T1 > 0 && p.T3 > 0 && p.K > 0 && p.UC <= 0 <= p.UO) ||
        throw(ArgumentError("PHSteamUnit $(gov.name): PMAX > PMIN >= 0, T1, T3, K > 0 and UC <= 0 <= UO are required"))
    all(isfinite, (r_v, rh_dT, tau_n, lowflow, p.K, p.T1, p.T2, p.T3, p.UO, p.UC, p.PMAX, p.PMIN, p.R_base)) ||
        throw(ArgumentError("PHSteamUnit $(gov.name): parameters must be finite"))
    if path === :C && volumes !== nothing
        vs = (volumes.chest, volumes.rh1, volumes.rh2, volumes.crossover)
        all(v -> isfinite(v) && v > 0, vs) ||
            throw(ArgumentError("PHSteamUnit $(gov.name): path C's volumes must be positive and finite"))
    end
    d = path === :A ? _steam_data_A(; cold_reheat = cold_reheat === nothing ?
                                        throw(ArgumentError("path A needs cold_reheat")) : cold_reheat) :
        path === :C ? (!(volumes isa NamedTuple) ? throw(ArgumentError("path C needs volumes = (; chest, rh1, rh2, crossover)")) :
                       _steam_data_C(; V_chest = volumes.chest, V_rh1 = volumes.rh1, V_rh2 = volumes.rh2,
                                     V_crossover = volumes.crossover)) :
        throw(ArgumentError("PHSteamUnit: path must be :A or :C"))
    if path === :A && volumes !== nothing
        vA = Dict("chest" => volumes.chest, "reheater" => volumes.reheater, "crossover" => volumes.crossover)
        d = (; d..., storages = [haskey(vA, s.name) ? (; s..., V = Float64(vA[s.name])) : s for s in d.storages])
    end
    tb = tables
    ns = length(d.storages)
    # the throttle and the chest's design pressure
    thr = d.throttle.h === nothing ? _steam_nominal(tb, d.throttle.p; T = d.throttle.T) :
          _steam_nominal(tb, d.throttle.p; h = d.throttle.h)
    sT = steam_state(tb, thr.rho, thr.u)
    p_T, h_T, v_T = d.throttle.p, sT.h, 1 / thr.rho
    # nominal storage states, placed in the tables
    nom = Vector{Any}(undef, ns)
    for (k, s) in enumerate(d.storages)
        pk = k == 1 ? r_v * p_T : s.p
        # the chest holds throttled steam: the throttle's enthalpy at the chest's pressure
        ns_ = k == 1 ? _steam_nominal(tb, pk; h = h_T) :
              s.x !== nothing ? _steam_nominal(tb, pk; x = s.x) :
              s.T !== nothing ? _steam_nominal(tb, pk; T = s.T) : _steam_nominal(tb, pk; h = s.h)
        st = steam_state(tb, ns_.rho, ns_.u)
        nom[k] = (; rho = ns_.rho, u = ns_.u, st...)
    end
    # sections: efficiency and ellipse-law constant at the design point
    nsec = length(d.sections)
    sec_eta, sec_K, sec_delta, W = zeros(nsec), zeros(nsec), zeros(nsec), 0.0
    for (j, sc) in enumerate(d.sections)
        a = nom[sc.up]
        pout = sc.down == 0 ? d.p_cond : nom[sc.down].p
        hout = sc.down == 0 ? d.h_exhaust : nom[sc.down].h
        his = steam_h_isentropic(tb, pout, a.s).h
        sec_eta[j] = (a.h - hout) / (a.h - his)
        0 < sec_eta[j] <= 1 || throw(ArgumentError("PHSteamUnit $(gov.name): section $(sc.name) has efficiency $(sec_eta[j]) at the design point"))
        D = a.p^2 - pout^2
        sec_delta[j] = lowflow * D
        sec_K[j] = sc.q * sqrt(a.p / a.rho) / _sqd(D, sec_delta[j])
        W += sc.q * (a.h - hout)
    end
    # resistances
    nres = length(d.resist)
    res_K, res_delta = zeros(nres), zeros(nres)
    for (r, rs) in enumerate(d.resist)
        D = nom[rs.up].p - nom[rs.down].p
        D > 0 || throw(ArgumentError("PHSteamUnit $(gov.name): $(rs.name) has no pressure drop at the design point"))
        res_delta[r] = lowflow * D
        res_K[r] = rs.q / _sqd(D, res_delta[r])
    end
    # valve: rated flow at full stroke and r_v
    z = (r_v - STEAM_EPS_CR) / (1 - STEAM_EPS_CR)
    Dv = r_v <= STEAM_EPS_CR ? 1.0 : 1 - z^2
    delta_v = lowflow * Dv
    beta_v = r_v <= STEAM_EPS_CR ? 1.0 : _sqd(Dv, delta_v) / _sqd(1.0, delta_v)
    cv = d.q_valve / beta_v
    # inflows at the design point, extraction fractions, heat, node volumes
    qin = zeros(ns)
    qin[1] += d.q_valve
    for sc in d.sections; sc.down > 0 && (qin[sc.down] += sc.q); end
    for rs in d.resist; qin[rs.down] += rs.q; end
    beta = d.ext ./ qin
    qout = qin .- d.ext
    kh, Tw = zeros(ns), zeros(ns)
    for (k, s) in enumerate(d.storages)
        s.heated || continue
        # steady: Q = sum of inflows times (h_k - h_in); the inflow is the resistance's, at the
        # upstream storage's enthalpy
        rs = d.resist[findfirst(r -> r.down == k, d.resist)]
        Q = qin[k] * (nom[k].h - nom[rs.up].h)
        kh[k] = Q / rh_dT
        Tw[k] = nom[k].T + rh_dT
    end
    # a node's residence time is its mass over its whole throughflow, the inflow (the outflow and the
    # extraction together) at the design point (consultation 7, point 9)
    V = [s.kind === :vessel ? s.V : tau_n * qin[k] / nom[k].rho for (k, s) in enumerate(d.storages)]
    # scale to the case's unit: n published units in parallel
    n = p.PMAX * STEAM_SB / W
    nominal = (; W = n * W, Pm = n * W / STEAM_SB, rho = [x.rho for x in nom], u = [x.u for x in nom],
               p = [x.p for x in nom], h = [x.h for x in nom], T = [x.T for x in nom], s = [x.s for x in nom],
               M = n .* V .* [x.rho for x in nom], qin = n .* qin, qout = n .* qout, q_valve = n * d.q_valve,
               sec_q = n .* [sc.q for sc in d.sections], source = d.source)
    return PHSteamUnit(gov.name, path, p.K, p.T1, p.T2, p.T3, p.UO, p.UC, p.PMAX, p.PMIN, p.R_base, n,
                       [s.name for s in d.storages], n .* V, beta, n .* kh, Tw, n * cv, p_T, h_T, STEAM_EPS_CR,
                       delta_v, [sc.up for sc in d.sections], [sc.down for sc in d.sections], n .* sec_K,
                       sec_eta, sec_delta, [sc.name for sc in d.sections],
                       [rs.up for rs in d.resist], [rs.down for rs in d.resist], n .* res_K, res_delta,
                       d.p_cond, [s.kind === :node for s in d.storages], nominal, tb,
                       valve, Float64[], Float64[], Float64[],
                       Dict{String,Any}("source" => vsource, "V" => [s.V for s in d.storages], "fit" => fit, "objective" => objective,
                                        "turbine" => (; T4 = p.T4, T5 = p.T5, T6 = p.T6, T7 = p.T7,
                                                      K1 = p.K1, K3 = p.K3, K5 = p.K5, K7 = p.K7)))
end

# ------------------------------------------------------------------------------------------------
"""
    _steam_eval(g, xs, y) -> NamedTuple

Everything the unit's equations need at its storage states `xs` (the local states without `x0`,
`x1`) and valve stroke `y` (fraction): storage states, the valve, section and resistance flows,
the mass and energy rates `dM`, `dU`, the work `W`, and the boundary powers `boiler`, `heat`,
`cond`, `ext` (W).
"""
function _steam_eval(g::PHSteamUnit, xs::AbstractVector{T}, y; Tw = g.Tw) where {T}
    ns = length(g.V)
    st = [steam_state(g.tb, xs[2k-1] / g.V[k], xs[2k] / xs[2k-1]) for k in 1:ns]
    # the vessels (chest, reheaters, crossover) are single-phase; a vessel state in the dome is outside
    # the model (record section 15.3, consultation 7, point 1)
    for k in 1:ns
        (g.isnode[k] || _deep(st[k].x) >= 1) ||
            throw(DomainError(_deep(st[k].x), "PHSteamUnit $(g.name): the vessel $(g.names[k]) entered the dome"))
    end
    R = promote_type(T, typeof(y), eltype(Tw))
    dM = zeros(R, ns); dU = zeros(R, ns); qin = zeros(R, ns)
    # valve
    eps = st[1].p / g.p_T
    z = (eps - g.eps_cr) / (1 - g.eps_cr)
    # forward only (record section 3.4): no flow once the chest reaches the throttle's pressure
    beta = eps <= g.eps_cr ? one(R) : z >= 1 ? zero(R) : _sqd(1 - z^2, g.delta_v) / _sqd(one(R), g.delta_v)
    qv = g.cv * y * beta
    hv = qv >= 0 ? oftype(st[1].h, g.h_T) : st[1].h
    dM[1] += qv; dU[1] += qv * hv
    qv > 0 && (qin[1] += qv)
    boiler = qv * hv
    # sections
    W = zero(R); cond = zero(R)
    qsec = Vector{R}(undef, length(g.sec_K))
    for j in eachindex(g.sec_K)
        a, b = g.sec_up[j], g.sec_down[j]
        pin = st[a].p
        pout = b == 0 ? oftype(pin, g.p_cond) : st[b].p
        D = pin^2 - pout^2
        if D >= 0 || b == 0
            # forward, with the inlet's p v; none back from the condenser, whose state is not modelled
            # (consultation 7, point 3)
            q = b == 0 && D < 0 ? zero(R) : g.sec_K[j] * _sqd(D, g.sec_delta[j]) / sqrt(pin * g.V[a] / xs[2a-1])
        else
            # reversed: the same law with the donor's (downstream storage's) p v (consultation 7, point 5)
            q = g.sec_K[j] * _sqd(D, g.sec_delta[j]) / sqrt(pout * g.V[b] / xs[2b-1])
        end
        qsec[j] = q
        if q >= 0 && b == 0 && _deep(q) == 0
            continue                                 # nothing reaches the condenser (consultation 8, point 35)
        elseif q >= 0
            hout = st[a].h - g.sec_eta[j] * (st[a].h - steam_h_isentropic(g.tb, pout, st[a].s).h)
            W += q * (st[a].h - hout)
            dM[a] -= q; dU[a] -= q * st[a].h
            if b == 0
                cond += q * hout
            else
                dM[b] += q; dU[b] += q * hout; qin[b] += q
            end
        else                                    # reverse: no work, the receiving side's enthalpy
            hb = b == 0 ? st[a].h : st[b].h
            dM[a] -= q; dU[a] -= q * hb
            b == 0 ? (cond += q * hb) : (dM[b] += q; dU[b] += q * hb)
            qin[a] -= q
        end
    end
    # resistances (isenthalpic)
    qres = Vector{R}(undef, length(g.res_K))
    for r in eachindex(g.res_K)
        a, b = g.res_up[r], g.res_down[r]
        q = g.res_K[r] * _sqd(st[a].p - st[b].p, g.res_delta[r])
        qres[r] = q
        hq = q >= 0 ? st[a].h : st[b].h
        dM[a] -= q; dU[a] -= q * hq; dM[b] += q; dU[b] += q * hq
        q >= 0 ? (qin[b] += q) : (qin[a] -= q)
    end
    # extractions and heat
    ext = zero(R); heat = zero(R)
    for k in 1:ns
        qe = g.beta[k] * qin[k]
        dM[k] -= qe; dU[k] -= qe * st[k].h
        ext += qe * st[k].h
        if g.kh[k] > 0
            Q = g.kh[k] * (Tw[k] - st[k].T)
            dU[k] += Q; heat += Q
        end
    end
    return (; st, qv, beta, qsec, qres, dM, dU, W, boiler, heat, cond, ext)
end

# the valve opening at the governor's demand `d` (power units, within [PMIN, PMAX]). With the valve
# characteristic (record 16.6): cubic Hermite through the steady characteristic's nodes with their
# exact slopes; below the lowest node the opening is proportional to the demand (the valve is choked
# there, its flow proportional to the opening); above the highest it stays at the highest node's
# opening (full opening: the demand is no longer met).
function _steam_opening(g::PHSteamUnit, d)
    g.valve === :opening && return d / g.PMAX
    dd, yy, ss = g.vc_d, g.vc_y, g.vc_s
    isempty(dd) && error("PHSteamUnit $(g.name): the valve characteristic is built by steam_initialise")
    dv = _deep(d)
    dv <= dd[1] && return yy[1] * d / dd[1]
    dv >= dd[end] && return yy[end] + zero(d)
    i = min(searchsortedlast(dd, dv), length(dd) - 1)
    h = dd[i+1] - dd[i]
    t = (d - dd[i]) / h
    t2, t3 = t^2, t^3
    return (2t3 - 3t2 + 1) * yy[i] + (t3 - 2t2 + t) * h * ss[i] + (-2t3 + 3t2) * yy[i+1] + (t3 - t2) * h * ss[i+1]
end

# the valve opening at the servo state x1 (power units), without recording a decision
_steam_stroke(g::PHSteamUnit, x1) = _steam_opening(g, clamp(x1, g.PMIN, g.PMAX))

"""
    _steam_rhs!(dx, g, x, omega, Pref, rec) -> dx

The unit's rows: IEEEG1's lead-lag and servo (its own limiters, recorded by `rec` in the kernel's
order), then the storages' mass and energy rates.
"""
function _steam_rhs!(dx, g::PHSteamUnit, x, omega, Pref, rec::ModeRecorder)
    err = Pref - 1.0 - g.R_base * (omega - 1.0)
    dx[1] = (g.K * err - x[1]) / g.T1
    GV = x[1] * (1.0 - g.T2 / g.T1) + g.K * (g.T2 / g.T1) * err
    x1_lim = clamp_mode(rec, x[2], g.PMIN, g.PMAX)
    srv = (GV - x1_lim) / g.T3
    srv = clamp_mode(rec, srv, g.UC, g.UO)
    srv = nonwindup(rec, x1_lim, srv, g.PMIN, g.PMAX)
    srv = outband_relax(rec, srv, x[2], g.PMIN, g.PMAX, g.T3)
    dx[2] = srv
    e = _steam_eval(g, view(x, 3:length(x)), _steam_opening(g, x1_lim))
    for k in eachindex(g.V)
        dx[2+2k-1] = e.dM[k]
        dx[2+2k] = e.dU[k]
    end
    return dx
end

"""The unit's turbine power on the system base (per case unit): the shaft work over `S_b`."""
_steam_power(g::PHSteamUnit, x) = _steam_eval(g, view(x, 3:length(x)), _steam_stroke(g, x[2])).W / STEAM_SB

"""
    _steam_powers(g, x) -> NamedTuple

For the energy account (per case unit, on the system base): the stored energy `energy = sum U / S_b`
(pu s), `port` the net power received (boiler and heat in, condenser and extractions out, shaft
out), `tport` the sum of the magnitudes of the port powers, `shaft` the turbine power, `rate` the
energy's rate along the unit's own rows, and the parts.
"""
function _steam_powers(g::PHSteamUnit, x)
    e = _steam_eval(g, view(x, 3:length(x)), _steam_stroke(g, x[2]))
    sb = STEAM_SB
    port = (e.boiler + e.heat - e.cond - e.ext - e.W) / sb
    return (; energy = sum(x[2+2k] for k in eachindex(g.V)) / sb, port, shaft = e.W / sb,
            tport = (abs(e.boiler) + abs(e.heat) + abs(e.cond) + abs(e.ext) + abs(e.W)) / sb,
            rate = sum(e.dU) / sb, boiler = e.boiler / sb, heat = e.heat / sb, cond = e.cond / sb,
            ext = e.ext / sb)
end

"""
    steam_initialise(g, Pm; rebuild = true) -> (x, Pref)

The unit's steady state at turbine power `Pm` (system base, per case unit): the stroke, every
storage's mass and energy, and each reheater's wall temperature (written into `g.Tw`) such that
the reheat temperatures are their design values. Newton from the design point scaled by the load;
an error if the stroke would leave `[PMIN, PMAX]` or the solve fails.

With the valve characteristic (`valve = :characteristic`) and `rebuild = true`, the characteristic
is built at this power and, for `volumes = :fit`, the vessels' volumes are fitted at it: the unit's
hardware then depends on `Pm`. With `rebuild = false` the unit's hardware is kept as it is (volumes
and valve characteristic, e.g. from a build at the rated operating point) and only its operating
state is solved at `Pm`: the stroke, the storages, and the wall temperatures that hold the design
reheat temperatures (the boiler's firing, an operating condition). The servo demand is then the
inverse of the existing characteristic at the solved opening; an error if that opening lies outside
the characteristic's range over `[PMIN, PMAX]`.
"""
function steam_initialise(g::PHSteamUnit, Pm::Real; rebuild::Bool = true)
    !rebuild && g.valve === :characteristic && isempty(g.vc_d) &&
        throw(ArgumentError("steam_initialise $(g.name): rebuild = false needs a unit whose valve characteristic is built"))
    Pm > 0 || throw(ArgumentError("steam_initialise $(g.name): the turbine power must be positive (got $Pm)"))
    ns = length(g.V)
    nom = g.nominal
    heated = findall(>(0), g.kh)
    f = Pm / nom.Pm
    # the design data are for the unfitted volumes: a unit whose volumes were fitted (a kept build) starts
    # each storage's mass and energy from the design values times its volume ratio
    vr = haskey(g.volumes, "V_unfitted") ? g.V ./ g.volumes["V_unfitted"] : ones(ns)
    w0 = vcat(f * g.PMAX, [v for k in 1:ns for v in (f * vr[k] * nom.M[k], f * vr[k] * nom.M[k] * nom.u[k])], g.Tw[heated])
    sc = vcat(g.PMAX, [v for k in 1:ns for v in (vr[k] * nom.M[k], vr[k] * nom.M[k] * nom.u[k])], [100.0 for _ in heated])
    function F(w)
        Tw = g.Tw .* one(eltype(w))
        Tw[heated] .= w[2+2ns:end]
        e = _steam_eval(g, view(w, 2:1+2ns), w[1] / g.PMAX; Tw)
        r = similar(w)
        for k in 1:ns
            r[2k] = e.dM[k] / max(nom.qin[k], 1e-9)
            r[2k+1] = e.dU[k] / (max(nom.qin[k], 1e-9) * nom.h[k])
        end
        r[1] = e.W / STEAM_SB - Pm
        for (i, k) in enumerate(heated)
            r[1+2ns+i] = (e.st[k].T - nom.T[k]) / nom.T[k]
        end
        return r
    end
    w = w0 ./ sc
    Fs(v) = F(v .* sc)
    # Newton with backtracking: a step is halved while it leaves the tables or does not reduce the
    # residual (consultation 7, point 21)
    resid(v) = try maximum(abs, Fs(v)) catch err; err isa DomainError ? Inf : rethrow() end
    for it in 1:60
        r = Fs(w)
        nr = maximum(abs, r)
        nr < 1e-12 && break
        it == 60 && error("steam_initialise $(g.name): no steady state at Pm = $Pm (residual $nr)")
        dw = ForwardDiff.jacobian(Fs, w) \ r
        a = 1.0
        while resid(w .- a .* dw) >= nr && a > 1e-6
            a /= 2
        end
        a > 1e-6 || error("steam_initialise $(g.name): the Newton step does not reduce the residual at Pm = $Pm")
        w = w .- a .* dw
    end
    w = w .* sc
    xs = w[2:1+2ns]
    if g.valve === :opening
        g.PMIN <= w[1] <= g.PMAX || throw(ArgumentError("steam_initialise $(g.name): the stroke $(w[1] / g.PMAX) is outside its limits"))
        g.Tw[heated] .= w[2+2ns:end]
        x1 = w[1]
    elseif !rebuild
        # the hardware kept: the servo demand whose opening through the existing characteristic is the
        # solved opening (the characteristic is monotone in the demand over [PMIN, PMAX])
        y = w[1] / g.PMAX
        ylo, yhi = _steam_opening(g, g.PMIN), _steam_opening(g, g.PMAX)
        ylo <= y <= yhi || throw(ArgumentError("steam_initialise $(g.name): the opening $y at Pm = $Pm is outside the " *
                                               "existing characteristic's range [$ylo, $yhi]"))
        lo, hi = Float64(g.PMIN), Float64(g.PMAX)
        for _ in 1:200
            mid = (lo + hi) / 2
            _steam_opening(g, mid) < y ? (lo = mid) : (hi = mid)
            hi - lo <= 1e-14 * max(1.0, abs(hi)) && break
        end
        g.Tw[heated] .= w[2+2ns:end]
        x1 = (lo + hi) / 2
    else
        # the servo state is the flow demand, the turbine power; the opening follows from the valve
        # characteristic, built here at the wall temperatures just found (record 16.6); the opening
        # must lie in [0, 1], the demand in [PMIN, PMAX] (consultation 10, point 5); the wall
        # temperatures are kept only if the characteristic is built (point 14)
        0 < w[1] / g.PMAX <= 1 || throw(ArgumentError("steam_initialise $(g.name): the opening $(w[1] / g.PMAX) is outside (0, 1]"))
        g.PMIN <= Pm <= g.PMAX || throw(ArgumentError("steam_initialise $(g.name): Pm = $Pm is outside [PMIN, PMAX]"))
        # all or nothing: the wall temperatures, the characteristic, the volumes and their record are
        # restored if any step fails (consultation 11, points 5 and 6)
        saved = (copy(g.Tw), copy(g.vc_d), copy(g.vc_y), copy(g.vc_s), copy(g.V), copy(g.volumes))
        try
            g.Tw[heated] .= w[2+2ns:end]
            _steam_valve_characteristic!(g, xs, w[1] / g.PMAX, Pm)
            # the vessels' volumes fitted to the case's IEEEG1 turbine (record 16.7): the masses and
            # energies scale with the volumes at the same steady state
            if g.volumes["fit"]
                f = _steam_fit_volumes!(g, xs, Pm)
                ves = findall(!, g.isnode)
                for (i, k) in enumerate(ves)
                    xs[2k-1] *= f[i]; xs[2k] *= f[i]
                end
            end
        catch
            g.Tw .= saved[1]
            for (v, s) in ((g.vc_d, saved[2]), (g.vc_y, saved[3]), (g.vc_s, saved[4]))
                empty!(v); append!(v, s)
            end
            g.V .= saved[5]
            empty!(g.volumes); merge!(g.volumes, saved[6])
            rethrow()
        end
        x1 = Float64(Pm)
    end
    x = vcat(x1, x1, xs)                    # x0 = x1 = GV at rest
    Pref = 1 + x1 / g.K
    return x, Pref
end

# the step response of IEEEG1's turbine (four lags in series, weights K1, K3, K5, K7) at times `ts`,
# per unit of the demand step
function _ieeeg1_turbine_step(tp, ts)
    f(dx, x, _, _) = (dx[1] = (1 - x[1]) / tp.T4; dx[2] = (x[1] - x[2]) / tp.T5; dx[3] = (x[2] - x[3]) / tp.T6;
                      dx[4] = (x[3] - x[4]) / tp.T7; nothing)
    sol = SciMLBase.solve(SciMLBase.ODEProblem(f, zeros(4), (0.0, ts[end])), Sundials.CVODE_BDF();
                          reltol = 1e-10, abstol = 1e-12, saveat = ts)
    return [tp.K1 * u[1] + tp.K3 * u[2] + tp.K5 * u[3] + tp.K7 * u[4] for u in sol.u]
end

# IEEEG1's turbine as a transfer function at s
_ieeeg1_turbine_tf(tp, s) = (a4 = 1 / (1 + s * tp.T4); a5 = a4 / (1 + s * tp.T5); a6 = a5 / (1 + s * tp.T6);
                             a7 = a6 / (1 + s * tp.T7); tp.K1 * a4 + tp.K3 * a5 + tp.K5 * a6 + tp.K7 * a7)

# Nelder and Mead's simplex search, for the few volume factors of the fit
function _nelder_mead(f, x0; step = 1.0, iters = 300, tol = 1e-8)
    n = length(x0)
    S = [copy(x0)]
    for i in 1:n
        x = copy(x0); x[i] += step; push!(S, x)
    end
    F = [f(x) for x in S]
    converged = false
    shrink!() = for i in 2:n+1
        S[i] = S[1] .+ 0.5 .* (S[i] .- S[1]); F[i] = f(S[i])
    end
    for _ in 1:iters
        p = sortperm(F); S, F = S[p], F[p]
        if F[end] - F[1] < tol * (abs(F[1]) + 1e-12)
            converged = true
            break
        end
        c = sum(S[1:n]) / n
        xr = c .+ (c .- S[end]); fr = f(xr)
        if fr < F[1]
            xe = c .+ 2 .* (c .- S[end]); fe = f(xe)
            if fe < fr
                S[end] = xe; F[end] = fe
            else
                S[end] = xr; F[end] = fr
            end
        elseif fr < F[n]
            S[end] = xr; F[end] = fr
        elseif fr < F[end]
            # outside contraction (consultation 11, point 9)
            xo = c .+ 0.5 .* (xr .- c); fo = f(xo)
            if fo <= fr
                S[end] = xo; F[end] = fo
            else
                shrink!()
            end
        else
            xc = c .+ 0.5 .* (S[end] .- c); fc = f(xc)
            if fc < F[end]
                S[end] = xc; F[end] = fc
            else
                shrink!()
            end
        end
    end
    i = argmin(F)
    return S[i], F[i], converged
end

"""
    _steam_fit_volumes!(g, xs, Pm) -> factors

The vessels' volumes fitted to the case's IEEEG1 turbine (record 16.7, 16.8), at the steady state `xs` at
power `Pm`, the valve through its characteristic, constant speed. The factors on the vessels' volumes
(chest, reheaters, crossover; the nodes keep their residence time) are found by a simplex search in
their logarithms, from the volumes already in `g`, for one of two objectives (`g.volumes["objective"]`):
"step" (`volumes = :fit`, the selected values): the response of the shaft power to a demand step of 1 %
over 10 s against IEEEG1's turbine with the case's `T4` to `T7` and `K1` to `K7`, least squares on a
0.02 s grid; "frequency" (`volumes = :fit_frequency`, an alternative): the linearised demand-to-power
frequency response, mean squared relative complex error on 40 frequencies from 0.02 to 2 Hz. Both
errors are computed before and after for the record, with the response at 0.65 Hz. Accepted if the
step's rms error is at most 5 % of the step (and, for the frequency objective, the frequency error at
most 5 %). Writes the volumes into `g.V`, the factors and the errors into `g.volumes`. (The frequency
objective was tried after the step fit left 10 % too much gain and 9 deg of phase lead at 0.65 Hz on
phase lead at 0.65 Hz on path C: record 16.8.)
"""
function _steam_fit_volumes!(g::PHSteamUnit, xs0::Vector{Float64}, Pm::Real)
    ts = collect(0.02:0.02:10.0)
    tp = g.volumes["turbine"]
    # the target's steady gain is K1 + K3 + K5 + K7; the characteristic's is 1 (consultation 11, point 2)
    abs(tp.K1 + tp.K3 + tp.K5 + tp.K7 - 1) < 1e-6 ||
        error("PHSteamUnit $(g.name): the volume fit needs K1 + K3 + K5 + K7 = 1")
    target = _ieeeg1_turbine_step(tp, ts)
    # a step of 1 % down, or up where down would cross PMIN; normalised by the step taken (point 3)
    d1 = 0.99 * Pm >= g.PMIN ? 0.99 * Pm : 1.01 * Pm
    d1 <= min(g.PMAX, g.vc_d[end]) || error("PHSteamUnit $(g.name): no admissible demand step for the volume fit at $Pm")
    dstep = d1 - Pm
    y1 = Float64(_steam_opening(g, d1))
    ves = findall(!, g.isnode)
    V0 = copy(g.V)
    # the volumes before any fit, for the factors' record (point 7)
    haskey(g.volumes, "V_unfitted") || (g.volumes["V_unfitted"] = copy(g.V))
    function response(lf)
        local fl = exp.(lf)                         # not `fct`: that would share the outer variable
        xs = copy(xs0)
        g.V .= V0
        for (i, k) in enumerate(ves)
            g.V[k] = V0[k] * fl[i]; xs[2k-1] *= fl[i]; xs[2k] *= fl[i]
        end
        rhs!(dx, z, _, _) = (e = _steam_eval(g, z, y1); for k in eachindex(g.V); dx[2k-1] = e.dM[k]; dx[2k] = e.dU[k]; end; nothing)
        sol = try
            SciMLBase.solve(SciMLBase.ODEProblem(rhs!, xs, (0.0, ts[end])), Sundials.CVODE_BDF();
                            reltol = 1e-7, abstol = 1e-8 .* abs.(xs), saveat = ts, maxiters = 10_000_000)
        catch err
            err isa DomainError || rethrow()
            nothing
        end
        (sol === nothing || !SciMLBase.successful_retcode(sol)) && return nothing
        r = try
            [(_steam_eval(g, u, y1).W / STEAM_SB - Pm) / dstep for u in sol.u]
        catch err
            err isa DomainError || rethrow()
            nothing
        end
        return r !== nothing && all(isfinite, r) ? r : nothing
    end
    cost(lf) = (r = response(lf); r === nothing ? 1e9 : sum(abs2, r .- target) / length(ts))
    step_rms(lf) = (r = response(lf); r === nothing ? Inf : sqrt(sum(abs2, r .- target) / length(ts)))
    # the linearised demand-to-power response: G(jw) = C (jw - A)^-1 B + D
    fgrid = exp.(range(log(0.02), log(2.0), length = 40))
    Lt = [_ieeeg1_turbine_tf(tp, 2pi * f * im) for f in fgrid]
    n = length(xs0)
    function freq(lf, fq = fgrid)
        local fl = exp.(lf)                         # not `fct`: that would share the outer variable
        xs = copy(xs0)
        g.V .= V0
        for (i, k) in enumerate(ves)
            g.V[k] = V0[k] * fl[i]; xs[2k-1] *= fl[i]; xs[2k] *= fl[i]
        end
        F(z) = (e = _steam_eval(g, view(z, 1:n), _steam_opening(g, z[end]));
                vcat([v for k in eachindex(g.V) for v in (e.dM[k], e.dU[k])], e.W / STEAM_SB))
        J = try
            ForwardDiff.jacobian(F, vcat(xs, Float64(Pm)))
        catch err
            err isa DomainError || rethrow()
            return nothing
        end
        A, B, C, D = J[1:n, 1:n], J[1:n, end], J[end, 1:n], J[end, end]
        G = try
            [dot(C, (2pi * f * im * I - A) \ B) + D for f in fq]
        catch err                                   # a singular solve (consultation 13, point 7)
            err isa LinearAlgebra.SingularException || err isa LinearAlgebra.LAPACKException || rethrow()
            return nothing
        end
        return all(isfinite, G) ? G : nothing
    end
    fcost(lf) = (G = freq(lf); G === nothing ? 1e9 : sum(abs2, G ./ Lt .- 1) / length(fgrid))
    # every diagnostic is computed here, before anything is written into g (consultation 13, point 6)
    L65 = _ieeeg1_turbine_tf(tp, 2pi * 0.65im)
    rel(G) = G === nothing ? NaN : G[1]
    scost(lf) = (r = response(lf); r === nothing ? 1e9 : sum(abs2, r .- target) / length(ts))
    obj = g.volumes["objective"]
    rms0, fe0, lf, fe1, conv, rms1, worst, G65, G65u = try
        z0 = zeros(length(ves))
        r0, e0 = step_rms(z0), sqrt(fcost(z0))
        lf_, _, conv_ = obj == "step" ? _nelder_mead(scost, z0) : _nelder_mead(fcost, z0; iters = 600)
        Gf = freq(lf_)
        w_ = Gf === nothing ? Inf : maximum(abs.(Gf ./ Lt .- 1))
        (r0, e0, lf_, sqrt(fcost(lf_)), conv_, step_rms(lf_), w_, rel(freq(lf_, [0.65])), rel(freq(z0, [0.65])))
    finally
        g.V .= V0                                   # every exit leaves the entry volumes (point 5)
    end
    # explicit acceptance (point 8): the step's rms error at most 5 % of the step; with the frequency
    # objective also the frequency error at most 5 %
    (rms1 <= 0.05 && (obj == "step" || fe1 <= 0.05)) ||
        error("PHSteamUnit $(g.name): the volume fit is rejected (step rms $rms1, frequency error $fe1)")
    fct = exp.(lf)
    for (i, k) in enumerate(ves)
        g.V[k] = V0[k] * fct[i]
    end
    Vu = g.volumes["V_unfitted"]
    g.volumes["factors"] = Dict(g.names[k] => g.V[k] / Vu[k] for k in ves)      # cumulative (point 7)
    g.volumes["rms_before"] = rms0
    g.volumes["rms"] = rms1
    g.volumes["freq_error_before"] = fe0
    g.volumes["freq_error"] = fe1
    # the worst relative error on the grid, and the response at 0.65 Hz (the grid's main swing in the bus-16
    # fault) at the entry volumes ("before") and the fitted ones
    g.volumes["freq_error_worst"] = worst
    g.volumes["at_0.65Hz"] = Dict("gain_ratio" => abs(G65 / L65), "phase_deg" => rad2deg(angle(G65 / L65)),
                                  "gain_ratio_before" => abs(G65u / L65), "phase_deg_before" => rad2deg(angle(G65u / L65)))
    g.volumes["converged"] = conv
    g.volumes["Pm_fit"] = Float64(Pm)
    g.volumes["step"] = dstep
    return fct
end

# the steady storage states at a fixed opening `y` and the present wall temperatures, by damped Newton
# from `xs0` (continuation); the states only, the power follows
function _steam_steady(g::PHSteamUnit, y::Float64, xs0::Vector{Float64})
    ns = length(g.V)
    nom = g.nominal
    sc = [v for k in 1:ns for v in (nom.M[k], nom.M[k] * nom.u[k])]
    function Fs(v)
        e = _steam_eval(g, v .* sc, y)
        r = similar(v, 2ns)
        for k in 1:ns
            r[2k-1] = e.dM[k] / max(nom.qin[k], 1e-9)
            r[2k] = e.dU[k] / (max(nom.qin[k], 1e-9) * nom.h[k])
        end
        return r
    end
    resid(v) = try maximum(abs, Fs(v)) catch err; err isa DomainError ? Inf : rethrow() end
    v = xs0 ./ sc
    for it in 1:80
        r = Fs(v)
        nr = maximum(abs, r)
        nr < 1e-12 && return v .* sc
        it == 80 && error("PHSteamUnit $(g.name): no steady state at opening $y (residual $nr)")
        dv = ForwardDiff.jacobian(Fs, v) \ r
        a = 1.0
        while resid(v .- a .* dv) >= nr && a > 1e-6
            a /= 2
        end
        a > 1e-6 || error("PHSteamUnit $(g.name): the Newton step does not reduce the residual at opening $y")
        v = v .- a .* dv
    end
end

# the steady power (system base) at opening y with states xs, and its total derivative dP/dy along the
# steady states (implicit function theorem: dxs/dy = -f_x \ f_y)
function _steam_steady_slope(g::PHSteamUnit, xs::Vector{Float64}, y::Float64)
    ns = length(g.V)
    function G(z)
        e = _steam_eval(g, view(z, 1:2ns), z[end])
        return vcat([v for k in 1:ns for v in (e.dM[k], e.dU[k])], e.W / STEAM_SB)
    end
    z = vcat(xs, y)
    J = ForwardDiff.jacobian(G, z)
    fx, fy = J[1:2ns, 1:2ns], J[1:2ns, end]
    Wx, Wy = J[end, 1:2ns], J[end, end]
    return G(z)[end], Wy - dot(Wx, fx \ fy)
end

"""
    _steam_valve_characteristic!(g, xs0, y0, P0)

The valve characteristic of record 16.6: the unit's steady power `P(y)` at fixed throttle state and
the present wall temperatures, on openings `y` from 1 down to where the power is below 30 % of `PMAX`
(steps of 0.02, the initial opening `y0` with power `P0` and states `xs0` as one node, exactly), with
the exact slopes `dP/dy`; stored inverted, as openings over demands with slopes `dy/dd = 1/(dP/dy)`. An
error unless `P` rises strictly with `y` at every node.
"""
function _steam_valve_characteristic!(g::PHSteamUnit, xs0::Vector{Float64}, y0::Float64, P0::Real)
    dy = 0.02
    nodes = Tuple{Float64,Float64,Float64}[]           # (y, P, dP/dy)
    _, s0 = _steam_steady_slope(g, xs0, y0)
    push!(nodes, (y0, Float64(P0), s0))
    # upward to full opening
    xs = copy(xs0)
    y = y0
    while y < 1.0
        y = min(1.0, (floor(y / dy + 1e-9) + 1) * dy)
        y - y0 < dy / 4 && y < 1.0 && continue          # too close to the initial node
        xs = _steam_steady(g, y, xs)
        P, s = _steam_steady_slope(g, xs, y)
        push!(nodes, (y, P, s))
    end
    # downward until the power is below 30 % of PMAX
    xs = copy(xs0)
    y = y0
    P = Float64(P0)
    while P > 0.3 * g.PMAX && y > 2dy
        y = (ceil(y / dy - 1e-9) - 1) * dy
        y0 - y < dy / 4 && continue
        xs = _steam_steady(g, y, xs)
        P, s = _steam_steady_slope(g, xs, y)
        push!(nodes, (y, P, s))
    end
    sort!(nodes; by = first)
    ys, Ps, ss = first.(nodes), [n[2] for n in nodes], last.(nodes)
    (all(>(0), ss) && all(>(0), diff(Ps))) ||
        error("PHSteamUnit $(g.name): the steady characteristic is not strictly increasing")
    # the inverse's slopes; where a pair of them would let the cubic overshoot, both are scaled down
    # (Fritsch and Carlson's condition alpha^2 + beta^2 <= 9), which keeps the interpolant monotone
    # (consultation 10, point 2); with nodes 0.02 apart this is not expected to act
    sl = 1 ./ ss
    nlim = 0
    for i in 1:length(Ps)-1
        D = (ys[i+1] - ys[i]) / (Ps[i+1] - Ps[i])
        a, b = sl[i] / D, sl[i+1] / D
        if a^2 + b^2 > 9
            tau = 3 / sqrt(a^2 + b^2)
            sl[i] = tau * a * D; sl[i+1] = tau * b * D
            nlim += 1
        end
    end
    nlim > 0 && @warn "PHSteamUnit $(g.name): $nlim interval(s) of the valve characteristic limited to stay monotone"
    empty!(g.vc_d); append!(g.vc_d, Ps)
    empty!(g.vc_y); append!(g.vc_y, ys)
    empty!(g.vc_s); append!(g.vc_s, sl)
    return g
end

# ------------------------------------------------------------------------------------------------
# The static-chain reference (the user's "both, compare", 2026-10-14; record section 15): the same
# unit without the extraction nodes. Between two vessels (or a vessel and the condenser) the
# sections and extraction points carry no storage; the chain's inflow is found by shooting: march
# the sections with the ellipse law (the outlet pressure explicit in the flow), take each
# extraction out at its point, and require the march to arrive at the next vessel's pressure (or,
# through a reheater's inlet resistance, at the reheater's). For standalone comparison with the
# node model, not for the grid runs: it solves a one-dimensional root per chain at every evaluation.

"""
    steam_ph(tb, p, h) -> (; rho, u, T, s, x)

The state at pressure `p` and enthalpy `h`: the wet mixture from the saturation table when `h` is
at most the saturated vapour's, otherwise the vapour map inverted by Newton on `(log rho, u)`.
"""
function steam_ph(tb::SteamTables, p, h)
    if p < exp(tb.sat.x0 + (tb.sat.n - 1) * tb.sat.dx)
        s = steam_sat(tb, p)
        # wet, consistently with steam_state's h = u + p v: the quality from u and v
        # (consultation 8, point 29)
        hf, hg = s.u_f + p * s.v_f, s.u_g + p * s.v_g
        if h <= hg
            x = (h - hf) / (hg - hf)
            0 <= x || throw(DomainError(h, "steam_ph: below the saturated liquid at p = $p"))
            v = s.v_f + x * (s.v_g - s.v_f)
            # the same density domain as steam_state (consultation 9, point 9)
            exp(tb.X_min) <= 1 / v <= exp(tb.X_max) ||
                throw(DomainError(1 / v, "steam_ph: the wet state at p = $p, h = $h is outside the tables' densities"))
            return (; rho = 1 / v, u = s.u_f + x * (s.u_g - s.u_f), T = s.T, s = s.s_f + x * (s.s_g - s.s_f), x)
        end
        # a start on the vapour side: the temperature from the enthalpy above saturation with a
        # vapour's heat capacity of about 2.1 kJ/(kg K) (consultation 9, point 8)
        T0 = clamp(s.T + (h - hg) / 2100.0, s.T + 1.0, 990.0)
    else
        T0 = 700.0
    end
    # vapour: unknowns z = (log rho, u), equations p(rho, u) = p, u + p/rho = h; the start's density
    # kept inside the tables (consultation 8, point 31)
    rho0 = clamp(p / (_PS.IAPWS95_R * T0), exp(tb.X_min) * 1.01, exp(tb.X_max) * 0.99)
    z = [log(rho0), h - p / rho0]
    # (a name of its own: a closure assigning to the outer r would share it)
    F(z) = (stz = steam_state(tb, exp(z[1]), z[2]); [stz.p / p - 1, (stz.h - h) / h])
    Fsafe(z) = try
        F(z)
    catch err
        err isa DomainError || rethrow()
        nothing
    end
    r = Fsafe(z)
    r === nothing && throw(DomainError(h, "steam_ph: the start at p = $p, h = $h is outside the tables"))
    for _ in 1:60
        maximum(abs, r) < 1e-13 && break
        dz = ForwardDiff.jacobian(F, z) \ r
        # damped: halve the step until it stays in the tables and lowers the residual
        lam = 1.0
        while true
            zn = z .- lam .* dz
            rn = Fsafe(zn)
            if rn !== nothing && (maximum(abs, rn) < maximum(abs, r) || lam < 1e-3)
                z, r = zn, rn
                break
            end
            lam /= 2
            lam < 1e-6 && error("steam_ph: no state at p = $p, h = $h (step rejected)")
        end
    end
    maximum(abs, F(z)) < 1e-10 || error("steam_ph: no state at p = $p, h = $h")
    r = steam_state(tb, exp(z[1]), z[2])
    return (; rho = exp(z[1]), u = z[2], T = r.T, s = r.s, x = r.x)
end

# the chain leaving vessel `a`: the links and node storages until a vessel or the condenser
function _steam_chain(g::PHSteamUnit, a::Int)
    links = Tuple{Symbol,Int}[]
    k = a
    while true
        j = findfirst(==(k), g.sec_up)
        if j !== nothing
            push!(links, (:sec, j))
            k = g.sec_down[j]
        else
            r = findfirst(==(k), g.res_up)
            r === nothing && error("storage $k has no outgoing link")
            push!(links, (:res, r))
            k = g.res_down[r]
        end
        (k == 0 || !g.isnode[k]) && return links, k
    end
end

# the inverse of the regularised square root: the D with _sqd(D, delta) = y (monotone)
function _sqd_inv(y, delta)
    D = sign(y) * y^2                       # the square law's inverse, a good start
    for _ in 1:100
        f = _sqd(D, delta) - y
        abs(f) <= 1e-14 * max(abs(y), 1e-300) && return D
        df = ForwardDiff.derivative(z -> _sqd(z, delta), D)
        D -= f / df
    end
    abs(_sqd(D, delta) - y) <= 1e-10 * max(abs(y), 1e-300) || error("_sqd_inv: no convergence")
    return D
end

# march the chain from vessel state `st0` with inflow `q`, with the node model's own (regularised)
# laws (consultation 8, point 28): the end pressure, the flows and enthalpies at every point, the
# work. `ok = false` with `low = true` when the flow is too large for the march (a pressure would
# fall below zero or below the tables), `low = false` when it is too small (above the tables).
function _steam_march(g::PHSteamUnit, links, st0, q)
    p, h, s, rho = st0.p, st0.h, st0.s, st0.rho
    W, pts = 0.0, NamedTuple[]
    for (kind, i) in links
        if kind === :sec
            arg = p^2 - _sqd_inv(q * sqrt(p / rho) / g.sec_K[i], g.sec_delta[i])
            arg > 0 || return (; ok = false, low = true)
            pout = sqrt(arg)
            pout >= exp(g.tb.l_min) || return (; ok = false, low = true)
            pout <= exp(g.tb.l_max) || return (; ok = false, low = false)
            hout = h - g.sec_eta[i] * (h - steam_h_isentropic(g.tb, pout, s).h)
            W += q * (h - hout)
            b = g.sec_down[i]
            if b != 0 && g.isnode[b]
                st = steam_ph(g.tb, pout, hout)
                qe = g.beta[b] * q
                push!(pts, (; node = b, p = pout, h = hout, qext = qe))
                q -= qe
                p, h, s, rho = pout, hout, st.s, st.rho
            else
                return (; ok = true, pend = pout, hend = hout, qend = q, W, pts)
            end
        else
            pout = p - _sqd_inv(q / g.res_K[i], g.res_delta[i])
            pout > 0 || return (; ok = false, low = true)
            return (; ok = true, pend = pout, hend = h, qend = q, W, pts)
        end
    end
end

"""
    steam_static_eval(g, xs, y) -> NamedTuple

The static-chain reference at the vessels' states `xs` (mass and energy of every storage that is
not a node, in storage order) and valve stroke `y`: per vessel the mass and energy rates, the work
and the boundary powers, as `_steam_eval` returns them for the node model. The nodes' states are
not states here: each chain's inflow is the root of the march's end pressure against the next
vessel's (or the condenser's) pressure.
"""
function steam_static_eval(g::PHSteamUnit, xs::AbstractVector{Float64}, y)
    ves = findall(!, g.isnode)
    nv = length(ves)
    st = Dict(k => (r = steam_state(g.tb, xs[2i-1] / g.V[k], xs[2i] / xs[2i-1]); (; r..., rho = xs[2i-1] / g.V[k]))
              for (i, k) in enumerate(ves))
    # the same domain as the node model: vessels single-phase (consultation 8, point 34)
    for k in ves
        st[k].x >= 1 || throw(DomainError(st[k].x, "steam_static_eval: the vessel $(g.names[k]) entered the dome"))
    end
    dM, dU = zeros(nv), zeros(nv)
    ix = Dict(k => i for (i, k) in enumerate(ves))
    # valve into the chest
    eps = st[1].p / g.p_T
    z = (eps - g.eps_cr) / (1 - g.eps_cr)
    beta = eps <= g.eps_cr ? 1.0 : z >= 1 ? 0.0 : _sqd(1 - z^2, g.delta_v) / _sqd(1.0, g.delta_v)
    qv = g.cv * y * beta
    dM[ix[1]] += qv; dU[ix[1]] += qv * g.h_T
    boiler, W, cond, ext, heat = qv * g.h_T, 0.0, 0.0, 0.0, 0.0
    qin = zeros(nv); qin[ix[1]] += qv
    for a in ves
        links, b = _steam_chain(g, a)
        pb = b == 0 ? g.p_cond : st[b].p
        # the march's end pressure less the receiver's; a march that fails is too large a flow
        # (negative) or too small (positive); an isentrope outside the tables counts as too small
        function f(q)
            m = try
                _steam_march(g, links, st[a], q)
            catch err
                err isa DomainError || rethrow()
                (; ok = false, low = false)
            end
            return m.ok ? m.pend - pb : (m.low ? -pb : pb)
        end
        # no forward flow when the source is not above the receiver (reverse flow is outside this
        # reference: consultation 8, point 33)
        # (consultation 9, point 10: equal pressures are zero flow, not an error)
        st[a].p == pb && continue
        st[a].p > pb || error("steam_static_eval: the chain from $(g.names[a]) has a reverse pressure difference")
        lo, hi = 0.0, 1.0
        while f(hi) > 0
            lo = hi
            hi *= 2
            hi > 1e7 && error("steam_static_eval: no bracket for the chain from $(g.names[a])")
        end
        q = (lo + hi) / 2
        for _ in 1:200
            fq = f(q)
            abs(fq) < 1e-10 * pb && break
            fq > 0 ? (lo = q) : (hi = q)
            q = (lo + hi) / 2
        end
        m = _steam_march(g, links, st[a], q)
        (m.ok && abs(m.pend - pb) < 1e-8 * pb) || error("steam_static_eval: the chain from $(g.names[a]) did not converge")
        dM[ix[a]] -= q; dU[ix[a]] -= q * st[a].h
        W += m.W
        for pt in m.pts
            ext += pt.qext * pt.h
        end
        if b == 0
            cond += m.qend * m.hend
        else
            dM[ix[b]] += m.qend; dU[ix[b]] += m.qend * m.hend; qin[ix[b]] += m.qend
        end
    end
    # extractions from the vessels themselves, and heat
    for (i, k) in enumerate(ves)
        qe = g.beta[k] * qin[i]
        dM[i] -= qe; dU[i] -= qe * st[k].h; ext += qe * st[k].h
        if g.kh[k] > 0
            Q = g.kh[k] * (g.Tw[k] - st[k].T)
            dU[i] += Q; heat += Q
        end
    end
    return (; dM, dU, W, boiler, heat, cond, ext, qv)
end
