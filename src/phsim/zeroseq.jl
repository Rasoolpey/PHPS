# The dq0 model (upgrade 4: the zero axis and phase-asymmetric elements; docs/active/PH_SIMULATOR_SPEC.md sections 2, 6;
# review 80). The network quantities are the complex dq (or alpha-beta) phasor x = x_Re + j x_Im and the real zero axis
# x_0, with (spec section 2)
#     x_a = Re(e^{j theta} x) + x_0 / sqrt(2),  x_b = Re(e^{j (theta - a)} x) + x_0 / sqrt(2),  x_c = ... (theta + a),
# a = 2 pi / 3, theta = omega_b t + theta_pw in the synchronous frame (theta_pw - theta0 in the stationary frame), so that the power is
# p = Re(conj(v) i) + v_0 i_0 = (2/3) sum_k v_k i_k. Written P(theta): [x_Re, x_Im, x_0] = P(theta) [x_a, x_b, x_c],
# P^-1 = (3/2) P'.

const _A3 = 2pi / 3

"""P(theta): phase quantities (a, b, c) to (Re, Im, 0)."""
function park(theta)
    c(k) = (theta, theta - _A3, theta + _A3)[k]
    return (2 / 3) .* [cos(c(1)) cos(c(2)) cos(c(3));
                       -sin(c(1)) -sin(c(2)) -sin(c(3));
                       1 / sqrt(2) 1 / sqrt(2) 1 / sqrt(2)]
end

"""
    fault_gabc(kind, R; Rg = R) -> 3 x 3 conductance matrix in phase coordinates (currents out of the phases)

The fault's phase-domain conductance: `:ag`, `:bg`, `:cg` phase to ground through R; `:ab`, `:bc`, `:ca` phase to phase
through R; `:abg`, `:bcg`, `:cag` two phases each through R to a common point, and that point to ground through Rg (the
common point eliminated); `:abcg` every phase to ground through R; `:abc` the three phases through R each to a common
isolated point (the star of three resistors, no ground).
"""
function fault_gabc(kind::Symbol, R::Real; Rg::Real = R)
    R > 0 || throw(ArgumentError("fault_gabc: R = $R"))
    g = 1 / R
    e(k) = (v = zeros(3); v[k] = 1.0; v)
    ph = Dict('a' => 1, 'b' => 2, 'c' => 3)
    s = String(kind)
    if kind in (:ag, :bg, :cg)
        k = ph[s[1]]; return g .* (e(k) * e(k)')
    elseif kind in (:ab, :bc, :ca)
        d = e(ph[s[1]]) .- e(ph[s[2]]); return g .* (d * d')
    elseif kind in (:abg, :bcg, :cag)
        Rg > 0 || throw(ArgumentError("fault_gabc: Rg = $Rg"))
        # node n: g (v1 - vn) + g (v2 - vn) = vn / Rg -> vn = g (v1 + v2) / (2g + gg); the phase currents g (vk - vn)
        gg = 1 / Rg
        u = e(ph[s[1]]) .+ e(ph[s[2]])
        return g .* (e(ph[s[1]]) * e(ph[s[1]])' .+ e(ph[s[2]]) * e(ph[s[2]])') .- (g^2 / (2g + gg)) .* (u * u')
    elseif kind === :abcg
        return g .* Matrix(1.0I, 3, 3)
    elseif kind === :abc
        o = ones(3); return g .* (Matrix(1.0I, 3, 3) .- (o * o') ./ 3)
    end
    throw(ArgumentError("fault_gabc: kind = $kind"))
end

"""
    fault_star(kind, R; Rg) -> (mask, gp, gg)

Every fault kind as a star of per-phase switches (review 82): the phases `mask` (a, b, c), each through the conductance
`gp` to a star node, the node to ground through `gg` (Inf: solid, 0: isolated). ag: 1/R, Inf; ab: 2/R, 0 (b to c through
R); abg: 1/R, 1/R_g; abcg: 1/R, Inf; abc: 1/R, 0.
"""
function fault_star(kind::Symbol, R::Real; Rg::Real = R)
    ph = Dict('a' => 1, 'b' => 2, 'c' => 3)
    s = String(kind)
    m(ps...) = ntuple(p -> p in ps, 3)
    kind in (:ag, :bg, :cg) && return (m(ph[s[1]]), 1 / R, Inf)
    kind in (:ab, :bc, :ca) && return (m(ph[s[1]], ph[s[2]]), 2 / R, 0.0)
    kind in (:abg, :bcg, :cag) && return (m(ph[s[1]], ph[s[2]]), 1 / R, 1 / Rg)
    kind === :abcg && return (m(1, 2, 3), 1 / R, Inf)
    kind === :abc && return (m(1, 2, 3), 1 / R, 0.0)
    throw(ArgumentError("fault_star: kind = $kind"))
end

"""
    fault_gstar(kind, R, closed; Rg) -> 3 x 3

The phase-domain conductance of a fault whose switches `closed` (a, b, c) conduct: G(m) = gp D_m - gp^2 / (gp |m| + gg) m m'
(review 82). With every switch of the kind closed it is `fault_gabc` itself (the existing path, bit for bit); the empty
star and an isolated one-leg star are 0.
"""
function fault_gstar(kind::Symbol, R::Real, closed::NTuple{3,Bool}; Rg::Real = R)
    base, gp, gg = fault_star(kind, R; Rg)
    m = base .& closed
    m == base && return fault_gabc(kind, R; Rg)
    n = count(m)
    (n == 0 || (n == 1 && gg == 0)) && return zeros(3, 3)
    v = Float64[m...]
    G = gp .* Matrix(Diagonal(v))
    isinf(gg) || (G .-= (gp^2 / (gp * n + gg)) .* (v * v'))
    return G
end

"""G_f(theta) = (3/2) P(theta) G_abc P(theta)': the fault in (Re, Im, 0) (spec 6.3; symmetric PSD at every theta)."""
fault_gdq0(Gabc::AbstractMatrix, theta) = (P = park(theta); 1.5 .* (P * Gabc * P'))

"""The fault seen at a bus without a zero-axis node (i_0 = 0): the Schur complement on (Re, Im)."""
function fault_schur(G::AbstractMatrix)
    G[3, 3] > 0 || return G[1:2, 1:2]
    return G[1:2, 1:2] .- (G[1:2, 3] * G[3, 1:2]') ./ G[3, 3]
end

# ---- the zero-sequence data of a case (review 80) ----

"""
    ZeroLine, ZeroShunt

A line's zero-sequence data (R0, X0, B0 total, system pu) and a transformer's zero-sequence shunt to ground at its HV bus
(R, X in system pu), with where they come from.
"""
struct ZeroLine
    idx::String
    from::Int    # bus ids
    to::Int
    r0::Float64
    x0::Float64
    b0::Float64
end
struct ZeroShunt
    idx::String
    bus::Int     # the HV bus id
    r::Float64
    x::Float64
    connection::String
end

# the line ratios (review 79 / 80: the IEEE 39-bus EMTP benchmark v4 as reported by OPAL-RT HYPERSIM, unverified)
const LINE_ZERO_RATIOS = (r = 4.03, x = 2.11, b = 0.65)

"""
    zero_sequence_data(case; mode = :physical, pf_json, ratios = LINE_ZERO_RATIOS, zm0_3limb = 5.0)
        -> (; lines, shunts, mode, notes)

The case's zero-sequence network (review 80): every line from the positive-sequence data and `ratios`; every two-winding
transformer from the PowerFactory probe (`pf_json`: uk0tr, ur0tr, strn, ntnum by name). `:physical`: the generator /
converter step-ups (one side at a generator bus) are YNd, the zero shunt at the HV bus z0 = uk0 + j... (r = ur0, x =
uk0) on the rating; the network transformers YNy0 three-limb, z = z_H + z_M0 = (0.9 + zm0_3limb) uk0 (r = 0.9 ur0), LV
open. `:powerfactory`: every transformer YNy0 as stored: z = (zx0hl_h + zx0hl_n) uk0, LV open. Parallel units divide the
impedances. The machines (isolated stars), converters (three-wire), loads (delta) have no zero axis.
"""
function zero_sequence_data(case; gen_buses, mode::Symbol = :physical, pf_json::AbstractString,
                            ratios = LINE_ZERO_RATIOS, zm0_3limb::Real = 5.0)
    mode in (:physical, :powerfactory) || throw(ArgumentError("zero_sequence_data: mode = $mode"))
    pf = JSON3.read(read(pf_json, String))
    objs = pf.objects
    g(o, a) = (r = get(o.named, Symbol(a), nothing); r === nothing ? get(get(o, :matched, Dict()), Symbol(a), nothing) : r)
    val(o, a) = (r = g(o, a); r === nothing ? nothing : r.value)
    trf = Dict{String,Any}()
    for e in pf.ElmTr2
        nm = replace(split(String(e.full_name), '\\')[end], ".ElmTr2" => "")
        t = objs[Symbol(String(e.type))]
        trf[nm] = (; uk0 = Float64(val(t, "uk0tr")), ur0 = Float64(something(val(t, "ur0tr"), 0.0)),
                   S = Float64(val(t, "strn")), n = Int(something(val(e, "ntnum"), 1)),
                   zh = Float64(something(val(t, "zx0hl_h"), 0.5)), zm = Float64(something(val(t, "zx0hl_n"), 100.0)),
                   vec = String(something(val(t, "vecgrp"), "?")))
    end
    gens = Set(gen_buses)
    raw = Dict(String(r.idx) => r for r in case.raw.Line)       # the case's PowerFactory names and voltage levels
    lines, shunts, notes = ZeroLine[], ZeroShunt[], String[]
    Sb = case.config.mva_base
    for l in case.lines
        rl = get(raw, l.idx, nothing)
        pfname = rl === nothing ? "" : String(get(rl, :_pf, ""))
        Vn1 = rl === nothing ? 1.0 : Float64(get(rl, :Vn1, 1.0)); Vn2 = rl === nothing ? 1.0 : Float64(get(rl, :Vn2, 1.0))
        if haskey(trf, pfname)
            d = trf[pfname]
            sc = Sb / d.S / d.n                         # % on the rating -> system pu, parallel units
            hv = l.bus1 in gens ? l.bus2 : l.bus2 in gens ? l.bus1 : (Vn1 >= Vn2 ? l.bus1 : l.bus2)
            step_up = l.bus1 in gens || l.bus2 in gens
            # z_sc0 = r0 + j x0 on the rating (review 80, item 2): r0 = ur0 / 100, x0 = sqrt((uk0 / 100)^2 - r0^2)
            r0 = d.ur0 / 100; zk = d.uk0 / 100; x0 = sqrt(zk^2 - r0^2)
            if mode === :physical && step_up
                # YNd: from HV z_H + z_L || z_M0 ~ z_sc0 (the zero-sequence magnetising current neglected; the delta's
                # circulating current referred to HV)
                push!(shunts, ZeroShunt(l.idx, hv, r0 * sc, x0 * sc, "YNd (physical baseline): z_sc0"))
            elseif mode === :physical
                # YNy0 three-limb: z_H = 0.9 z_sc0 (PF split) plus the magnetising reactance k |z_sc0|; LV open
                push!(shunts, ZeroShunt(l.idx, hv, 0.9 * r0 * sc, (0.9 * x0 + zm0_3limb * zk) * sc,
                                        "YNy0 three-limb, x_M0 = $(zm0_3limb) |z_sc0| (physical baseline)"))
            else
                push!(shunts, ZeroShunt(l.idx, hv, d.zh * r0 * sc, (d.zh * x0 + d.zm * zk) * sc,
                                        "$(d.vec) as stored, x_M0 = $(d.zm) |z_sc0| (PowerFactory)"))
            end
        elseif l.b > 0 && l.tap == 1
            push!(lines, ZeroLine(l.idx, l.bus1, l.bus2, ratios.r * l.r, ratios.x * l.x, ratios.b * l.b))
        else
            push!(notes, "no zero-sequence data for $(l.idx) ($(pfname)): left open on the zero axis")
        end
    end
    return (; lines, shunts, mode, notes)
end

"""
    phs_phase_voltages(s, x, u; t) -> 3 x nbus matrix (phases a, b, c per bus row; network base, peak per unit)

The phase-to-ground voltages: x_a = Re(e^{j theta} v) + v0 / sqrt(2) (spec section 2), theta = omega_b t + theta_pw in
the synchronous frame (theta_pw - theta0 in the stationary frame; review 81). v0: the bus's zero node, or recovered from a ground fault's
elimination; NaN where the neutral is undetermined (no zero path: e.g. bus 12, review 80b item 4), or `nan0` if given
(an isolated star does not see v0, review 82). Without a zero axis (`zero = :none`) v0 = 0 is taken (the balanced model).
"""
function phs_phase_voltages(s, x, u; t::Real, nan0::Real = NaN)
    ps = _phys(s)
    _set_frame!(u, ps, t)
    up = s isa PHSystem ? u : phs_control_inputs(s, x, u.gf; cf = u.cf)
    xs = view(x, 1:ps.n)
    f = phs_flows(ps, xs, up)
    nb = length(ps.net.net.bus_ids)
    th = _park_angle(ps)
    out = fill(NaN, 3, nb)
    for k in 1:nb
        v = f.v[k]
        isfinite(real(v)) || continue
        v0 = if ps.z0 === nothing
            0.0
        elseif ps.z0.zrow[k] > 0
            f.z0.v0[ps.z0.zrow[k]]
        elseif f.z0 !== nothing && haskey(f.z0.v0bus, k)
            f.z0.v0bus[k]
        else
            Float64(nan0)
        end
        for (j, a) in enumerate((0.0, -_A3, _A3))
            out[j, k] = real(cis(th + a) * v) + v0 / sqrt(2)
        end
    end
    return out
end
