# B4 (PH_TODO.md): the steam unit's exergetic PH realisation at unit level (`steam_exergy_structure`; derivation
# docs/active/PH_B4_EXERGY.md; reviews 93, 94). On the full model's 8 steam units, at the rest state, at seeded 0.1 %
# perturbations, with the valve closed, with the reheater walls at the vessel temperature (no heat) and below it (heat
# reversed), and with every section efficiency at 0 and at 1. Checks: the rows (M_k, U_k) and the rotor torque against
# `_steam_eval`; J skew; the full feedthrough block [[R, P], [P', S]] PSD (diagonally scaled); the three boundary
# supplies (throttle q_v eps_T, walls Q (1 - T0/T_w), extractions -q_e eps_k); the exergy balance e' x' + omega tau =
# supplies - the block's quadratic form; no table violations; the dead-state reference residuals reported.

using Test, LinearAlgebra, Random
using PHPS
const _SX = PHPS

function _sx_check(g, xs, y, omega; Tw = g.Tw)
    S = _SX.steam_exergy_structure(g, xs, y, omega; Tw)
    r = _SX.steam_exergy_rows(S, omega)
    ev = S.ev
    dxc = vec(permutedims(hcat(ev.dM, ev.dU)))
    ee = vcat(S.e, omega)
    den = ((abs.(S.J) .+ abs.(S.R)) * abs.(ee))[1:S.nf] .+ abs.(S.PT) * abs.(S.uT) .+ abs.(S.Pw) * abs.(S.uw) .+
          abs.(S.Ge) * abs.(S.ue) .+ 1e-300
    rows = maximum(abs.(r.dx .- dxc) ./ den)
    tau = abs(r.tau - ev.W / omega) / max(abs(ev.W / omega), abs(ev.boiler) / omega, 1e-300)
    # the full feedthrough block over (storage rows + rotor, throttle, walls), P padded with the rotor row
    nT, nw = 2, length(S.uw)
    Pf = vcat(hcat(S.PT, S.Pw), zeros(1, nT + nw))
    Sf = cat(S.ST, S.Sw; dims = (1, 2))
    Pi = [S.R Pf; Pf' Sf]
    d = sqrt.(max.(diag(Pi), 0.0)); d[d .== 0] .= 1
    Pis = Symmetric(Pi ./ (d * d'))
    emin = eigmin(Pis)
    # supplies and the balance
    yT = S.PT' * S.e .+ S.ST * S.uT
    yw = S.Pw' * S.e .+ S.Sw * S.uw
    ye = S.Ge' * S.e
    sT = dot(yT, S.uT); sw = dot(yw, S.uw); se = dot(ye, S.ue)
    sc = abs(ev.boiler) + abs(ev.heat) + abs(ev.ext) + abs(ev.W) + 1e-300
    epsT = let st = _SX.steam_ph(g.tb, g.p_T, g.h_T); g.h_T - S.T0 * st.s - S.mu0 end
    supT = abs(sT - ev.qv * epsT) / sc
    supw = abs(sw - sum((g.kh[k] * (Tw[k] - S.ev.st[k].T) * (1 - S.T0 / Tw[k]) for k in eachindex(g.kh) if g.kh[k] > 0); init = 0.0)) / sc
    supe = abs(se + sum(S.ue[k] * (S.ev.st[k].h - S.T0 * S.ev.st[k].s - S.mu0) for k in eachindex(S.ue))) / sc
    z = vcat(ee, S.uT, S.uw)
    bal = abs(dot(S.e, r.dx) + omega * r.tau - (sT + sw + se - dot(z, Pi * z))) / sc
    return (; rows, tau, skew = maximum(abs, S.J + S.J'), emin, supT, supw, supe, bal, nviol = length(S.violations),
            viol = [(v.kind, v.A / max(v.q, 1e-300)) for v in S.violations],
            mu0_res = S.mu0_res, mu0_res_wet = S.mu0_res_wet)
end

@testset "steam unit: exergetic port-Hamiltonian realisation at unit level (B4)" begin
    sc = load_scenario(joinpath(ROOT, "cases", "IEEE39Bus_PF", "no_fault_phtrue.json"))
    eq = solve_equilibrium(load_case(sc.system_path), sc)
    paths = Dict("IEEEG1_2" => :A, "IEEEG1_5" => :A, "IEEEG1_6" => :A, "IEEEG1_8" => :A, "IEEEG1_9" => :A,
                 "IEEEG1_3" => :C, "IEEEG1_4" => :C, "IEEEG1_7" => :C, "IEEEG1_10" => :C)
    steam = (; paths, r_v = 0.95, rh_dT = 100.0, tau_n = 0.03, lowflow = 1e-3, cold_reheat = 26.2,
             volumes = :fit, valve = :characteristic, tables = steam_tables())
    cs, x0, u, _ = phs_controlled(eq; steam, hydro = (; Tw = :match))
    rng = MersenneTwister(94)
    omega = 2pi * 60
    worst = Dict{Symbol,Float64}(); viols = Any[]; nviol = 0; nout = 0; ncase = 0; mres = 0.0; mwet = 0.0
    units = _SX._steam_units(cs)
    @test length(units) == 8
    for j in units
        g = cs.gov[j]; o = cs.goff[j]
        xs0 = x0[o+2:o+_SX.nstates(g)-1]
        y0 = _SX._steam_stroke(g, x0[o+1])
        gE0 = deepcopy(g); gE0.sec_eta .= 0.0
        gE1 = deepcopy(g); gE1.sec_eta .= 1.0
        Tk = [_SX.steam_state(g.tb, xs0[2k-1] / g.V[k], xs0[2k] / xs0[2k-1]).T for k in eachindex(g.V)]
        cases = Any[(g, xs0, y0, g.Tw), (g, xs0, 0.0, g.Tw), (g, xs0, y0, Tk), (g, xs0, y0, Tk .- 20.0),
                    (gE0, xs0, y0, g.Tw), (gE1, xs0, y0, g.Tw)]
        for _ in 1:5
            push!(cases, (g, xs0 .* (1 .+ 1e-3 .* randn(rng, length(xs0))), y0, g.Tw))
        end
        for (gg, xs, y, Tw) in cases
            q = try
                _sx_check(gg, xs, y, omega; Tw)
            catch err
                err isa DomainError || rethrow()
                nout += 1
                continue
            end
            ncase += 1
            for k in (:rows, :tau, :skew, :supT, :supw, :supe, :bal)
                worst[k] = max(get(worst, k, 0.0), getfield(q, k))
            end
            worst[:emin] = min(get(worst, :emin, 0.0), q.emin)
            nviol += q.nviol; mres = max(mres, abs(q.mu0_res)); mwet = max(mwet, q.mu0_res_wet)
            append!(viols, [(gg === g ? :base : gg.sec_eta[1] == 1 ? :eta1 : :eta0, v...) for v in q.viol])
        end
    end
    @info "steam exergy structure" ncase nout nviol worst mu0_res = mres mu0_res_wet = mwet
    foreach(v -> println("violation: ", v), viols); foreach(k -> println("worst ", k, " = ", worst[k]), sort(collect(keys(worst))))
    @test ncase >= 80 && nout == 0
    @test worst[:rows] <= 1e-12
    @test worst[:tau] <= 1e-12
    @test worst[:skew] == 0
    @test worst[:emin] >= -1e-12
    @test worst[:supT] <= 1e-12 && worst[:supw] <= 1e-12 && worst[:supe] <= 1e-12
    @test worst[:bal] <= 1e-12
    @test nviol == 0
end

# the acceptance items of reviews 93-95: reversals, blocked condenser flow, states near the dry boundary at the nodes,
# per-element conservation, the production signs with the exact IAPWS-95 kernel properties, and the signed Gibbs
# residual of the tabulated storage
function _sx_kernel_state(rho, u)
    # T from u(T, rho) = u with the IAPWS-95 kernel (Newton from the ideal-gas guess), then (h, s)
    T = u / 1500.0 + 273.15
    for _ in 1:60
        k = _SX.PhysicalSteam.iapws95_properties(T, rho)
        dT = (k.internal_energy_specific - u) / (k.cv)
        T -= dT
        abs(dT) < 1e-10 * T && return (; T, s = k.entropy, h = k.enthalpy, p = k.pressure)
    end
    error("kernel state: no convergence at rho = $rho, u = $u")
end

@testset "steam unit: exergetic realisation, acceptance cases (B4, reviews 93-95)" begin
    sc = load_scenario(joinpath(ROOT, "cases", "IEEE39Bus_PF", "no_fault_phtrue.json"))
    eq = solve_equilibrium(load_case(sc.system_path), sc)
    paths = Dict("IEEEG1_2" => :A, "IEEEG1_5" => :A, "IEEEG1_6" => :A, "IEEEG1_8" => :A, "IEEEG1_9" => :A,
                 "IEEEG1_3" => :C, "IEEEG1_4" => :C, "IEEEG1_7" => :C, "IEEEG1_10" => :C)
    steam = (; paths, r_v = 0.95, rh_dT = 100.0, tau_n = 0.03, lowflow = 1e-3, cold_reheat = 26.2,
             volumes = :fit, valve = :characteristic, tables = steam_tables())
    cs, x0, u, _ = phs_controlled(eq; steam, hydro = (; Tw = :match))
    omega = 2pi * 60
    nrev = 0; nblock = 0; nnear = 0; ncons = 0; nkern = 0; worst_rows = 0.0; worst_cons = 0.0; min_kern = Inf
    worst_gibbs = 0.0; nout = 0
    for j in _SX._steam_units(cs)
        g = cs.gov[j]; o = cs.goff[j]; ns = length(g.V)
        xs0 = x0[o+2:o+_SX.nstates(g)-1]; y0 = _SX._steam_stroke(g, x0[o+1])
        variants = Any[]
        # reversals: raise the downstream storage's mass (pressure) of each resistance and of the first internal section
        for r in eachindex(g.res_K)
            b = g.res_down[r]; xs = copy(xs0); xs[2b-1] *= 1.08; xs[2b] *= 1.08 * 1.02; push!(variants, (:rev, xs))
        end
        js = findfirst(jj -> g.sec_down[jj] > 0, eachindex(g.sec_K))
        b = g.sec_down[js]; xs = copy(xs0); xs[2b-1] *= 1.5; xs[2b] *= 1.5 * 1.03; push!(variants, (:rev, xs))
        # blocked condenser flow: the last storage below the condenser's pressure
        jl = findfirst(jj -> g.sec_down[jj] == 0, eachindex(g.sec_K))
        a = g.sec_up[jl]; xs = copy(xs0); xs[2a-1] *= 0.05; xs[2a] *= 0.05; push!(variants, (:block, xs))
        # near the dry boundary at the extraction nodes: energy lowered towards u_g
        for k in 1:ns
            g.isnode[k] || continue
            xs = copy(xs0); xs[2k] *= 0.985; push!(variants, (:near, xs))
        end
        for (kind, xs) in variants
            S = try
                _SX.steam_exergy_structure(g, xs, y0, omega)
            catch err
                err isa DomainError || rethrow()
                nout += 1; continue
            end
            rr = _SX.steam_exergy_rows(S, omega); ev = S.ev
            dxc = vec(permutedims(hcat(ev.dM, ev.dU)))
            ee = vcat(S.e, omega)
            den = ((abs.(S.J) .+ abs.(S.R)) * abs.(ee))[1:S.nf] .+ abs.(S.PT) * abs.(S.uT) .+ abs.(S.Pw) * abs.(S.uw) .+
                  abs.(S.Ge) * abs.(S.ue) .+ 1e-300
            worst_rows = max(worst_rows, maximum(abs.(rr.dx .- dxc) ./ den))
            kind === :rev && (nrev += any(e -> e.kind in (:resistance, :section_reversed) && e.gamma > 0 &&
                                              any(e.w[1:S.nf] .!= 0), S.elements) ? 1 : 0)
            kind === :block && (nblock += ev.qsec[jl] == 0 ? 1 : 0)
            kind === :near && (nnear += 1)
            isempty(S.violations) || @info "violation" kind S.violations
            # per-element conservation: mass and energy (the rotor's work, effort omega) on the retained rows
            for e in S.elements
                hasproperty(e, :w) && e.w isa AbstractVector || continue
                (e.kind === :section && all(iszero, e.w[1:S.nf])) && continue   # into the condenser: sink
                w = e.w; m = sum(w[1:2:S.nf]); en = sum(w[2:2:S.nf]) + omega * w[S.nf+1]
                e.kind === :valve && (m += e.wT[1]; en += e.wT[2])
                worst_cons = max(worst_cons, abs(m), abs(en) / (maximum(abs, w[2:2:S.nf]) + 1e-300)); ncons += 1
            end
            # the production signs with the kernel: for every flowing throttling branch d -> r,
            # (h_d - h_r)/T_r + s_r - s_d > 0 (exact properties)
            for e in S.elements
                e.kind in (:resistance, :section_reversed) && e.gamma != 0 || continue
                d = findfirst(k -> e.w[2k-1] < 0, 1:ns); r = findfirst(k -> e.w[2k-1] > 0, 1:ns)
                (d === nothing || r === nothing) && continue
                kd = _sx_kernel_state(xs[2d-1] / g.V[d], xs[2d] / xs[2d-1])
                kr = _sx_kernel_state(xs[2r-1] / g.V[r], xs[2r] / xs[2r-1])
                hd = -e.w[2d] / 1                                         # the donor's enthalpy carried
                Ak = (hd - kr.h) / kr.T + kr.s - kd.s - (hd - kd.h) / kd.T   # with the kernel's own h at both ends
                min_kern = min(min_kern, Ak); nkern += 1
            end
            # the signed Gibbs residual of the tabulated storage: (grad H_tab - z)' f against the throughput
            Ht(q) = sum(q[2k] - S.T0 * q[2k-1] * _SX.steam_state(g.tb, q[2k-1] / g.V[k], q[2k] / q[2k-1]).s - S.mu0 * q[2k-1] for k in 1:ns)
            gH = _SX.ForwardDiff.gradient(Ht, xs)
            thr = abs(ev.boiler) + abs(ev.heat) + abs(ev.ext) + abs(ev.W) + 1e-300
            worst_gibbs = max(worst_gibbs, abs(dot(gH .- S.e, dxc)) / thr)
        end
    end
    @info "steam acceptance" nrev nblock nnear ncons nkern nout worst_rows worst_cons min_kern worst_gibbs
    @test nout == 0
    @test nrev >= 8 && nblock >= 8 && nnear >= 8          # the cases were reached (non-vacuous)
    @test worst_rows <= 1e-12
    @test worst_cons <= 1e-12
    @test nkern >= 8 && min_kern > 0                      # exact-property production of every flowing branch > 0
    @test worst_gibbs <= 1e-4                             # the tabulated storage's rate off z'f: the Gibbs defect, measured
end
