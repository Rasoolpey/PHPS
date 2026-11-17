# The steam unit in exergetic port-Hamiltonian form (PH_TODO.md B4; docs/active/PH_B4_EXERGY.md, revised after
# review 93). Unit level, SI units: the unit's storage rows (M_k, U_k) and its machine rotor's row, whose effort is
# omega and whose rate is the shaft torque. Built element by element from `_steam_eval`'s laws:
#   e = grad H_ex, H_ex = sum_k (U_k - T0 S_k + p0 V_k - mu0 M_k) about the condenser's dead state, e_k = (b_k, a_k),
#     a_k = 1 - T0/T_k, b_k = T0 mu_k/T_k - mu0, mu_k = h_k - T_k s_k (from the table values: section 1);
#   the sections' reversible parts as a skew coupling with the rotor (J), the dissipations as rank-one, energy- and
#   mass-conserving R per element, the throttle and the reheater walls as reservoirs with feedthrough (P, S), the
#   extractions as outflow ports (G), the condenser as a dead-state reservoir (an enlarged environmental sink).
# With exact properties every gamma is >= 0; with the tables the branch gammas can be negative near zero production
# (the Gibbs defect): they are kept (the rows stay exact) and counted in `violations`.

"""
    steam_exergy_structure(g, xs, y, omega) -> NamedTuple

The exergetic PH realisation of the steam unit `g` at its storage states `xs` (the unit's states without the governor's
two), valve stroke `y` and rotor speed `omega` (rad/s, > 0), with the unit's storage rows f = 1:2ns (M_k, U_k) and the
rotor row ns2+1:

    [x_f'; tau] = (J - R) [e; omega] - [P_T; 0] u_T - [P_w; 0] u_w + [G_e; 0] u_e,

with J skew (the sections' reversible parts), R >= 0 for exact properties (every element's rank-one dissipation),
the throttle (u_T = e_T, P_T, S_T), the reheater walls (u_w = 1 - T0/T_w, P_w, S_w) and the extractions (u_e = q_e,
G_e). Returned: `e`, `J`, `R`, `PT`, `ST`, `uT`, `Pw`, `Sw`, `uw`, `Ge`, `ue`, `T0`, `mu0`, the reference residual
`mu0_res` and `mu0_res_wet` (J/kg: the saturation table's vapour end, steam_state's wet condenser states), the elements
(kind, gamma, production, direction w), `violations` (flowing branches with affinity <= 0: gamma < 0, kept; an
undefined gamma, flow with zero affinity, throws), and the
evaluation `ev` of `_steam_eval` the rows are compared with.
"""
function steam_exergy_structure(g::PHSteamUnit, xs::AbstractVector{Float64}, y::Real, omega::Real; Tw = g.Tw)
    omega > 0 || throw(DomainError(omega, "steam_exergy_structure: omega must be positive"))
    tb = g.tb
    ns = length(g.V); nf = 2ns; n = nf + 1; ip = n
    ev = _steam_eval(g, xs, y; Tw)
    st = ev.st
    # the dead state at the condenser: mu0 from the saturation table's liquid end, the end the isentrope's wet law
    # h_f + T (s - s_f) uses, so that an ideal (eta = 1) wet exhaust carries exactly zero flow exergy (the vapour end
    # gave -1.7e-4 to -2.7e-4 J/kg there and a spurious negative production; review 94 test)
    sc = steam_sat(tb, g.p_cond)
    T0 = sc.T
    mu0 = sc.h_f - T0 * sc.s_f
    mu0_res = (sc.h_g - T0 * sc.s_g) - mu0          # the saturation table's vapour end
    # and the flow exergy of steam_state's wet condenser states (h = u + p v there; review 94, Codex 6)
    mu0_res_wet = maximum(abs, (begin
        v = sc.v_f + xq * (sc.v_g - sc.v_f); w_ = steam_state(tb, 1 / v, sc.u_f + xq * (sc.u_g - sc.u_f))
        w_.h - T0 * w_.s - mu0
    end for xq in (0.6, 0.8, 0.9, 0.98)))
    eff(T, h, s) = (T0 * (h - T * s) / T - mu0, 1 - T0 / T)          # (b, a)
    e = zeros(nf)
    for k in 1:ns
        b, a = eff(st[k].T, st[k].h, st[k].s)
        e[2k-1] = b; e[2k] = a
    end
    ee = vcat(e, omega)
    J = zeros(n, n); R = zeros(n, n)
    iM(k) = 2k - 1; iU(k) = 2k
    elements = NamedTuple[]
    violations = NamedTuple[]
    # an isenthalpic branch from donor d to receiver r (storage indices; 0 = not a storage) at enthalpy hd, flow q > 0
    function branch!(kind, d, r, hd, q)
        w = zeros(n)
        d > 0 && (w[iM(d)] -= 1; w[iU(d)] -= hd)
        r > 0 && (w[iM(r)] += 1; w[iU(r)] += hd)
        A = -dot(w, ee)
        # zero flow: gamma = 0 (the equal-pressure limit; f = 0 is reproduced for any gamma >= 0 where w' e = 0, and
        # for every w where q = 0): R is discontinuous there, f is not (review 94)
        gam = q == 0 ? 0.0 : q / A
        isfinite(gam) || throw(DomainError(A, "steam_exergy_structure: $kind with flow $q and zero affinity: no realisation (table defect)"))
        push!(elements, (; kind, gamma = gam, production = q * A, w))
        q > 0 && A <= 0 && push!(violations, (; kind, q, A))
        R .+= gam .* (w * w')
        return nothing
    end
    # the valve: from the throttle reservoir T to the chest (storage 1), at h_T; augmented over (T_M, T_U, chest)
    sT = steam_ph(tb, g.p_T, g.h_T)
    bT, aT = eff(sT.T, g.h_T, sT.s)
    uT = [bT, aT]
    PT = zeros(nf, 2); ST = zeros(2, 2)
    let wT = [-1.0, -g.h_T], wc = zeros(n)
        wc[iM(1)] = 1; wc[iU(1)] = g.h_T
        A = -(dot(wT, uT) + dot(wc, ee))
        gam = ev.qv == 0 ? 0.0 : ev.qv / A                           # a closed valve: gamma = 0
        isfinite(gam) || throw(DomainError(A, "steam_exergy_structure: valve with flow $(ev.qv) and zero affinity"))
        push!(elements, (; kind = :valve, gamma = gam, production = ev.qv * A, w = wc, wT))
        ev.qv > 0 && A <= 0 && push!(violations, (; kind = :valve, q = ev.qv, A))
        R .+= gam .* (wc * wc')
        PT .+= gam .* (wc[1:nf] * wT')                                # R_aug[chest, T]
        ST .+= gam .* (wT * wT')
    end
    # sections
    for j in eachindex(g.sec_K)
        a, b = g.sec_up[j], g.sec_down[j]
        q = ev.qsec[j]
        if q > 0
            ha, sa = st[a].h, st[a].s
            pout = b == 0 ? g.p_cond : st[b].p
            hout = ha - g.sec_eta[j] * (ha - steam_h_isentropic(tb, pout, sa).h)
            Tb = b == 0 ? T0 : st[b].T
            hrev = b == 0 ? mu0 + T0 * sa : st[b].h + Tb * (sa - st[b].s)
            # reversible part: the fluid flux q w_rev, coupled skew with the rotor (fluid rows get J omega)
            wr = zeros(n); wr[iM(a)] = -1; wr[iU(a)] = -ha
            b > 0 && (wr[iM(b)] = 1; wr[iU(b)] = hrev)
            col = (q / omega) .* wr
            J[1:nf, ip] .+= col[1:nf]; J[ip, 1:nf] .-= col[1:nf]
            # dissipative part: L = q (h_out - h_rev) from the rotor to U_b (to the condenser for b = 0)
            L = q * (hout - hrev)
            wd = zeros(n); wd[ip] = -1 / omega
            b > 0 && (wd[iU(b)] = 1)
            gd = L * Tb / T0
            push!(elements, (; kind = :section, gamma = gd, production = L * T0 / Tb, w = wd, wrev = wr))
            gd >= 0 || push!(violations, (; kind = :section, q, A = L))
            R .+= gd .* (wd * wd')
        elseif q < 0
            # reversed: isenthalpic from the downstream storage b at its enthalpy (no flow back from the condenser)
            branch!(:section_reversed, b, a, st[b].h, -q)
        else
            push!(elements, (; kind = :section_zero_flow, gamma = 0.0, production = 0.0, w = zeros(n)))
        end
    end
    # resistances
    for r in eachindex(g.res_K)
        a, b = g.res_up[r], g.res_down[r]
        q = ev.qres[r]
        q >= 0 ? branch!(:resistance, a, b, st[a].h, q) : branch!(:resistance, b, a, st[b].h, -q)
    end
    # extractions: outflow ports at the local state, u_e = beta_k q_in,k (the code's inflows)
    qin = zeros(ns)
    ev.qv > 0 && (qin[1] += ev.qv)
    for j in eachindex(g.sec_K)
        a, b = g.sec_up[j], g.sec_down[j]; q = ev.qsec[j]
        q > 0 && b > 0 && (qin[b] += q)
        q < 0 && (qin[a] -= q)
    end
    for r in eachindex(g.res_K)
        a, b = g.res_up[r], g.res_down[r]; q = ev.qres[r]
        q >= 0 ? (qin[b] += q) : (qin[a] -= q)
    end
    Ge = zeros(nf, ns); ue = zeros(ns)
    for k in 1:ns
        Ge[iM(k), k] = -1; Ge[iU(k), k] = -st[k].h
        ue[k] = g.beta[k] * qin[k]
    end
    # reheater walls: heat reservoirs at T_w, gamma_h = k_h T_w T_k / T0, augmented over (wall, U_k)
    hk = [k for k in 1:ns if g.kh[k] > 0]
    Pw = zeros(nf, length(hk)); Sw = zeros(length(hk), length(hk)); uw = zeros(length(hk))
    for (c, k) in enumerate(hk)
        gh = g.kh[k] * Tw[k] * st[k].T / T0
        uw[c] = 1 - T0 / Tw[k]
        R[iU(k), iU(k)] += gh
        Pw[iU(k), c] = -gh
        Sw[c, c] = gh
        push!(elements, (; kind = :heat, gamma = gh, production = g.kh[k] * (Tw[k] - st[k].T) * T0 * (1 / st[k].T - 1 / Tw[k]), w = (k, c)))
    end
    return (; e, J, R, PT, ST, uT, Pw, Sw, uw, Ge, ue, T0, mu0, mu0_res, mu0_res_wet, elements, violations, ev, nf, ip)
end

"""
    steam_exergy_rows(S, omega) -> (; dx, tau)

The rows of the realisation `S` (`steam_exergy_structure`): the storage rates and the rotor torque.
"""
function steam_exergy_rows(S, omega)
    ee = vcat(S.e, omega)
    v = (S.J .- S.R) * ee
    dx = v[1:S.nf] .- S.PT * S.uT .- S.Pw * S.uw .+ S.Ge * S.ue
    return (; dx, tau = v[S.ip])
end
