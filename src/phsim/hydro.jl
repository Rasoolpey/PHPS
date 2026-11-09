# The physical hydro unit of the EMT simulator (PH_SIMULATOR_TODO.md, step G2; derivation and data in
# docs/active/PH_HYDRO_UNIT_MODEL.md).
#
# The device a case's IEEEG3 stands for: its governor (IEEEG3's own rows: pilot valve, gate servo with
# rate and position limits, permanent and transient droop), a penstock from a reservoir at fixed level,
# and the turbine, after the IEEE Working Group's nonlinear inelastic model (IEEE Trans. Power Systems
# 7(1), 1992, section 2.1, eq. (2), (4), (5)). Per unit on the turbine's base (flow and gate in units of
# the full-gate flow at the static head, head in units of the static head, power in units of the turbine
# rating, which is the full gate `GMAX` on the system base):
#
#   penstock   Tw q' = 1 - h                  (water inertia; friction f_p q^2 not in the case: 0)
#   turbine    q = G sqrt(h),  h = (q / G)^2,  G = gate / GMAX > 0
#   shaft      P = h q (the ideal turbine: At = 1, q_nl = 0, D = 0, the case's own coefficients),
#              P_sys = GMAX P on the system base; torque on the rotor P_sys / omega
#   energy     the water column's kinetic energy, H = GMAX Tw q^2 / 2 (pu s, system base)
#              H' = GMAX q (1 - h) = GMAX q (reservoir supply) - P_sys (to the shaft), exactly
#
# The gate's limits allow a closed gate (`PMIN = 0` in the case). The turbine law is q = G sqrt(h); written
# as h = (q/G)^2 it is 0/0 at G = 0, but the closed gate with the water column at rest is a regular physical
# state: the seat holds the static head, h = 1, q = 0, no shaft power, the column's energy and its rate zero
# (record 36, both reviewers; the energy book above holds exactly). An open gate with the column at rest has
# h = 0 (the head on the runner before the column accelerates; the start of a reopening). A closed gate with
# water flowing is outside the inelastic model (an unbounded head; a water-hammer model with compressibility
# would be needed), and the unit refuses it (a `DomainError`, as a steam vessel entering the dome does), as it
# refuses reverse flow. The runner is ideal also at zero flow (no shut-off drag; record 36, Codex 4). `Tw` is the water starting time at full
# gate; the value that makes the unit's linearisation at its operating point equal the case's IEEEG3 is
# `hydro_Tw_match` (record section 4).
#
# The ELASTIC water column (option `N > 0`; idea review 64, Codex and Gemini; docs/active/PH_HYDRO_UNIT_MODEL.md
# section "Elastic penstock"): the 1992 report's travelling-wave penstock (section 2.3: H/Q = -Z0 tanh(Te s),
# Tw = Z0 Te) as a lumped ladder of N sections from the reservoir (head 1) to the turbine:
#   (Tw/N) q_k' = h_{k-1} - h_k - (f_p/N) q_k |q_k|,  h_0 = 1          (water inertia, k = 1..N)
#   C_k h_k'    = q_k - q_{k+1},  q_{N+1} = q_t                       (compliance; C_k = Cw/N, the turbine node Cw/(2N))
#   q_t = G sqrt(h_N) for an open gate (h_N > 0), q_t = 0 for a closed gate;  Cw = Te^2 / Tw
#   energy from the reservoir datum: H = GMAX [sum (Tw/N) q_k^2 / 2 + sum C_k (h_k - 1)^2 / 2],
#   H' = GMAX q_t (1 - h_N) - GMAX sum (f_p/N) |q_k|^3 (reservoir supply at the turbine flow less shaft and friction)
# A gate closing on a moving column is then water hammer (the head at the turbine rises), not reverse flow. An open
# gate with a non-positive head at the turbine is outside the turbine model (a DomainError). Te and f_p are labelled
# assumptions (not in the case); N by the reflection-coefficient rule (scripts/opstab/penstock_n_rule.jl).

"""
    PHHydroUnit

A physical hydro unit in place of the case's IEEEG3 (file header): the IEEEG3's parameters (its
governor rows are kept), the full gate `GMAX` on the system base and the water starting time `Tw` at
full gate. Local states `xp, xr, at` (IEEEG3's pilot valve, droop reset and gate) and `q` (the penstock
flow, per unit of full-gate flow).
"""
struct PHHydroUnit
    name::String
    p::Ieeeg3Params
    GMAX::Float64
    Tw::Float64
    Tw_case::Float64         # the case's IEEEG3 water time (full gate), for the record
    Te::Float64              # wave travel time of the elastic penstock (0: the inelastic column)
    N::Int                   # number of lumped sections (0: the inelastic column)
    fp::Float64              # penstock friction coefficient (head loss f_p q|q| over the whole length)
end

nstates(g::PHHydroUnit) = g.N == 0 ? 4 : 3 + 2 * g.N
name(g::PHHydroUnit) = g.name
state_names(g::PHHydroUnit) = g.N == 0 ? ["xp", "xr", "at", "q"] :
    vcat(["xp", "xr", "at"], ["q$k" for k in 1:g.N], ["h$k" for k in 1:g.N])

"""
    PHHydroUnit(gov::IEEEG3_PHTRUE; Tw)

The physical hydro unit with the governor of the case's IEEEG3 `gov` and water starting time `Tw` (s,
at full gate). Refused: a non-positive or non-finite `Tw`, a turbine rating that is not positive, time
constants that are not positive, and coefficients other than the ideal turbine's (`a11 = 0.5`,
`a13 = 1`, `a21 = 1.5`, `a23 = 1`), which are the ones the physical model reproduces (record section 2).
"""
function PHHydroUnit(gov::IEEEG3_PHTRUE; Tw::Real, Te::Real = 0.0, N::Integer = 0, fp::Real = 0.0)
    p, d = gov.p, gov.params
    Tw = Float64(Tw)                                    # finiteness checked after the conversion
    (isfinite(Tw) && Tw > 0) || throw(ArgumentError("PHHydroUnit $(gov.name): Tw must be positive and finite"))
    all(isfinite, (p.PMAX, p.PMIN, p.R_base, p.Tg, p.Tp, p.Tr, p.Sigma, p.Delta, p.UO, p.UC)) ||
        throw(ArgumentError("PHHydroUnit $(gov.name): the governor's parameters must be finite"))
    GMAX = p.PMAX * p.R_base
    (GMAX > 0 && p.PMAX > p.PMIN >= 0) ||
        throw(ArgumentError("PHHydroUnit $(gov.name): PMAX > PMIN >= 0 and a positive rating are required"))
    (p.Tg > 0 && p.Tp > 0 && p.Tr > 0 && p.UC <= 0 <= p.UO) ||
        throw(ArgumentError("PHHydroUnit $(gov.name): Tg, Tp, Tr > 0 and UC <= 0 <= UO are required"))
    ideal = (; a11 = 0.5, a13 = 1.0, a21 = 1.5, a23 = 1.0)
    all(abs(Float64(d[string(k)]) - v) < 1e-12 for (k, v) in pairs(ideal)) ||
        throw(ArgumentError("PHHydroUnit $(gov.name): the case's turbine coefficients are not the ideal turbine's"))
    (N >= 0 && (N == 0 || (isfinite(Te) && Te > 0)) && fp >= 0) ||
        throw(ArgumentError("PHHydroUnit $(gov.name): the elastic penstock needs N >= 1, Te > 0 and f_p >= 0"))
    return PHHydroUnit(gov.name, p, GMAX, Float64(Tw), Float64(d["Tw"]), Float64(Te), Int(N), Float64(fp))
end

"""
    hydro_Tw_match(gov::IEEEG3_PHTRUE, Pm) -> Tw

The water starting time (full gate) with which the physical unit, linearised at turbine power `Pm`
(system base), has the case's IEEEG3 transfer function from gate to power at rated speed: the 1992
report's eq. (7) gives `(1 - G0 Tw s) / (1 + G0 Tw s / 2)` at gate `G0 = Pm / GMAX`, the case's IEEEG3
`(1 - Tw_case s) / (1 + Tw_case s / 2)` (its full-gate coefficients), hence `Tw = Tw_case / G0`.
"""
function hydro_Tw_match(gov::IEEEG3_PHTRUE, Pm::Real)
    GMAX = gov.p.PMAX * gov.p.R_base
    0 < Pm <= GMAX || throw(ArgumentError("hydro_Tw_match $(gov.name): Pm must lie in (0, GMAX]"))
    return Float64(gov.params["Tw"]) * GMAX / Pm
end

# the head at gate state `at` (system base) and flow q; the domain: an open gate, forward flow
function _hydro_head(g::PHHydroUnit, at, q)
    G = at / g.GMAX
    _deep(q) >= 0 || throw(DomainError(_deep(q), "PHHydroUnit $(g.name): reverse flow through the turbine"))
    if _deep(G) == 0
        # the closed gate: the seat holds the static head while the column is at rest (file header)
        _deep(q) == 0 || throw(DomainError(_deep(q), "PHHydroUnit $(g.name): the gate closed with water flowing (outside the inelastic model)"))
        return one(q) * one(G)
    end
    _deep(G) > 0 || throw(DomainError(_deep(G), "PHHydroUnit $(g.name): a negative gate"))
    return (q / G)^2
end

"""
    _hydro_rhs!(dx, g, x, omega, Pref, rec) -> dx

The unit's rows: IEEEG3's governor rows in the kernel's order (its limiters recorded by `rec`), then the
penstock's flow.
"""
function _hydro_rhs!(dx, g::PHHydroUnit, x, omega, Pref, rec::ModeRecorder)
    p = g.p
    GMIN, GMAX = p.PMIN * p.R_base, p.PMAX * p.R_base
    RUP, RDN = p.UO * p.R_base, p.UC * p.R_base
    at_lim = clamp_mode(rec, x[3], GMIN, GMAX)
    e = Pref - p.R_base * (omega - 1.0) - p.Sigma * at_lim - p.Delta * (at_lim - x[2])
    xp_lim = clamp_mode(rec, x[1], RDN, RUP)
    dxp = (e / p.Tg - xp_lim) / p.Tp
    dxp = nonwindup(rec, xp_lim, dxp, RDN, RUP)
    dxp = outband_relax(rec, dxp, x[1], RDN, RUP, p.Tp)
    dx[1] = dxp
    dx[2] = (at_lim - x[2]) / p.Tr
    gate_rate = nonwindup(rec, at_lim, xp_lim, GMIN, GMAX)
    gate_rate = outband_relax(rec, gate_rate, x[3], GMIN, GMAX, p.Tg)
    dx[3] = gate_rate
    if g.N == 0
        dx[4] = (1 - _hydro_head(g, at_lim, x[4])) / g.Tw
    else
        _penstock_rhs!(dx, g, x, at_lim)
    end
    return dx
end

# the elastic penstock's compliance of node k and the turbine flow at gate state `at` and turbine head `hN`
_penstock_C(g::PHHydroUnit, k) = (g.Te^2 / g.Tw) / (k < g.N ? g.N : 2 * g.N)
function _turbine_flow(g::PHHydroUnit, at, hN)
    G = at / g.GMAX
    _deep(G) >= 0 || throw(DomainError(_deep(G), "PHHydroUnit $(g.name): a negative gate"))
    _deep(G) == 0 && return zero(hN) * zero(G)               # closed gate: no flow through the runner
    _deep(hN) > 0 || throw(DomainError(_deep(hN), "PHHydroUnit $(g.name): an open gate with a non-positive head at the turbine (outside the turbine model)"))
    return G * sqrt(hN)
end
function _penstock_rhs!(dx, g::PHHydroUnit, x, at)
    N = g.N; M = g.Tw / N; r = g.fp / N
    q(k) = x[3+k]; h(k) = k == 0 ? one(x[4]) : x[3+N+k]
    qt = _turbine_flow(g, at, h(N))
    for k in 1:N
        dx[3+k] = (h(k - 1) - h(k) - r * q(k) * abs(q(k))) / M
        dx[3+N+k] = (q(k) - (k < N ? q(k + 1) : qt)) / _penstock_C(g, k)
    end
    return dx
end

# the gate within its limits, without recording a decision (as the kernels' output functions do)
_hydro_gate(g::PHHydroUnit, at) = clamp(at, g.p.PMIN * g.p.R_base, g.GMAX)

"""The unit's turbine power on the system base: `GMAX h q`."""
function _hydro_power(g::PHHydroUnit, x)
    g.N == 0 && return g.GMAX * _hydro_head(g, _hydro_gate(g, x[3]), x[4]) * x[4]
    hN = x[3+2*g.N]
    return g.GMAX * hN * _turbine_flow(g, _hydro_gate(g, x[3]), hN)
end

"""
    _hydro_powers(g, x) -> NamedTuple

For the energy account (system base): the water column's energy `GMAX Tw q^2 / 2`, the reservoir's
supply `GMAX q`, the shaft power, the net port (supply less shaft), the sum of the port magnitudes and the
energy's rate along the unit's own row.
"""
function _hydro_powers(g::PHHydroUnit, x)
    if g.N > 0
        N = g.N; hN = x[3+2N]; qt = _turbine_flow(g, _hydro_gate(g, x[3]), hN)
        E = sum((g.Tw / N) * x[3+k]^2 / 2 for k in 1:N) + sum(_penstock_C(g, k) * (x[3+N+k] - 1)^2 / 2 for k in 1:N)
        fric = sum((g.fp / N) * abs(x[3+k])^3 for k in 1:N)
        supply, shaft = g.GMAX * qt, g.GMAX * hN * qt
        return (; energy = g.GMAX * E, supply, shaft, port = supply - shaft, tport = abs(supply) + abs(shaft),
                rate = g.GMAX * (qt * (1 - hN) - fric))
    end
    q = x[4]
    h = _hydro_head(g, _hydro_gate(g, x[3]), q)
    supply, shaft = g.GMAX * q, g.GMAX * h * q
    return (; energy = g.GMAX * g.Tw * q^2 / 2, supply, shaft, port = supply - shaft,
            tport = abs(supply) + abs(shaft), rate = g.GMAX * q * (1 - h))
end

"""
    hydro_initialise(g, Pm) -> (x, Pref)

The unit at rest at turbine power `Pm` (system base) and rated speed: head 1, the gate `at = Pm` (the
ideal turbine at the static head gives `P = GMAX G`), flow `q = Pm / GMAX`, the pilot valve at zero, the
droop reset at the gate; `Pref` from the governor's summing junction, `Pref = Sigma at`.
"""
function hydro_initialise(g::PHHydroUnit, Pm::Real)
    p = g.p
    GMIN = p.PMIN * p.R_base
    (GMIN <= Pm <= g.GMAX && Pm > 0) ||
        throw(ArgumentError("hydro_initialise $(g.name): Pm = $Pm is outside (0, GMAX] or below the gate's minimum"))
    at = Float64(Pm)
    g.N == 0 && return [0.0, at, at, at / g.GMAX], p.Sigma * at
    # the elastic column at rest: q_k = q, h_k = 1 - (k/N) f_p q^2, with q = G / sqrt(1 + f_p G^2) (review 64, Codex 3)
    G = at / g.GMAX; q = G / sqrt(1 + g.fp * G^2)
    x = vcat([0.0, at, at], fill(q, g.N), [1 - (k / g.N) * g.fp * q^2 for k in 1:g.N])
    return x, p.Sigma * at
end
