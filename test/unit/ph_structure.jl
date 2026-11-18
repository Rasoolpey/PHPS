# The port-Hamiltonian structure of every component with a canonical form, assembled by `phs_structure` independently
# of the vector field and checked against it (PH_TODO.md step 1; review 86). Per case and per sampled state:
#   (1) the included rows of the vector field equal (J - R) z + G v (relative 1e-12 of the terms' magnitudes, floor 1e-14);
#   (2) J = -J' exactly; R symmetric (1e-14 of its norm) and positive semidefinite (eigmin >= -1e-12 of its norm);
#   (3) z is the gradient of the included energy (ForwardDiff; relative 1e-11, floor 1e-14 of the largest entry);
#   (4) z' R z equals the physical losses of the included components (the ledger's; the battery in its Norton form,
#       reconciled with the ledger's Thevenin split);
#   (5) the frame port carries no power; the amplifier, reservoir and battery ports carry their physical powers;
#   (6) the boundary (review 86b, 4): the included states are the expected ones (computed here from the model, not from
#       `phs_structure`), no field port on a machine with a DC shunt exciter and no EMF port on an averaged DC side, and
#       the ports' power equals the ledger's boundary power of the included components (internal ports cancel).

using PHPS, Test, LinearAlgebra, ForwardDiff, Random
const _P = PHPS

# the checks at one state; returns the worst values for the record
function _ph_check(m, x, u; label = "")
    S = phs_structure(m, x, u)
    f = phs_rhs(m, x, u)
    I = S.idx
    g = (S.J - S.R) * S.z .+ S.G * S.v
    den = abs.(S.J) * abs.(S.z) .+ abs.(S.R) * abs.(S.z) .+ abs.(S.G) * abs.(S.v) .+ 1e-14
    res = maximum(abs.(f[I] .- g[I]) ./ den[I])
    RI = Symmetric(S.R[I, I])
    nR = opnorm(Matrix(RI)) + 1e-300
    asym = maximum(abs, S.R - S.R') / nR
    emin = eigmin(RI) / nR
    gH = ForwardDiff.gradient(w -> phs_structure_energy(m, w), x)
    grad = maximum(abs.(gH[I] .- S.z[I]) ./ (abs.(gH[I]) .+ 1e-3 * maximum(abs, gH[I])))   # floor 1e-14 of the largest
    y = S.G' * S.z
    fr = findfirst(==("frame"), S.ports)
    frame = abs(y[fr] * S.v[fr]) / (sum(abs.(y .* S.v)) + 1e-300)
    loss = _ph_losses(m, x, u, S)
    bnd = _ph_boundary(m, x, u, S)
    ok = res < 1e-12 && S.J == -S.J' && asym < 1e-14 && emin > -1e-12 && grad < 1e-11 && frame < 1e-12 && loss < 1e-11 &&
         bnd < 1e-11
    ok || @info "phs_structure check failed" label res asym emin grad frame loss bnd
    return (; ok, res, asym, emin, grad, frame, loss, bnd, S)
end

# the expected included states, from the model's own layout
function _ph_expected(s::PHSystem, x)
    out = Set(1:s.n)
    for stc in s.statics; delete!(out, stc.zidx); end
    for mo in s.motors
        mo.ridx > 0 && foreach(k -> delete!(out, k), mo.ridx:mo.ridx+4)
        (mo.ridx > 0 && x[mo.ridx] == 0) && foreach(k -> delete!(out, k), mo.offset:mo.offset+6)
    end
    return out
end
function _ph_expected(cs::PHControlled, x)
    out = _ph_expected(cs.s, x)
    for j in eachindex(cs.exc)
        cs.exc[j] isa PHShuntExciter && push!(out, cs.xoff[j])
    end
    for j in _P._hydro_units(cs)
        foreach(k -> push!(out, k), cs.goff[j]+3:cs.goff[j]+_P.nstates(cs.gov[j])-1)
    end
    for i in _P._dc_converters(cs)
        o = cs.coff[i] + _P._dcidx(cs.conv[i]) - 1
        foreach(k -> push!(out, k), o:o+_P.nbranches(cs.conv[i].dc))
    end
    return out
end

# (6): the boundary
function _ph_boundary(m, x, u, S)
    Set(S.idx) == _ph_expected(m, x) || (@info "phs_structure: unexpected included states" symdiff(Set(S.idx), _ph_expected(m, x)); return Inf)
    y = S.G' * S.z
    pw = sum(y[k] * S.v[k] for k in eachindex(S.v) if S.ports[k] != "frame"; init = 0.0)
    if m isa PHSystem
        L = phs_ledger(m, x, u)
        return max(abs(pw - L.supply), abs(L.interconnection)) / (sum(L.tport) + 1e-300)
    end
    cs = m
    for j in eachindex(cs.exc)
        cs.exc[j] isa PHShuntExciter && any(==("machine$(j).field"), S.ports) && return Inf
    end
    L = phs_ledger(cs, x, u)
    supI = L.supply
    for j in eachindex(cs.exc)                # the brushless exciters' ports are outside (their field and shaft are inputs)
        cs.exc[j] isa PHBrushlessExciter && (supI -= L.port[findfirst(==(_P.name(cs.exc[j]) * ".exciter"), L.names)])
    end
    for j in _P._steam_units(cs)
        supI -= L.port[findfirst(==(_P.name(cs.gov[j]) * ".steam"), L.names)]
    end
    fl = _P.phs_flows(cs.s, x, phs_control_inputs(cs, x, u.gf; cf = u.cf))
    for i in _P._dc_converters(cs)
        g = cs.conv[i]
        any(==("$(cs.s.conv[i].name).emf_re"), S.ports) && return Inf
        xc = view(x, cs.coff[i]:cs.coff[i]+_P.nstates(g)-1)
        q = converter_dc_powers(g, xc, _P._conv_current(fl, cs.s.conv[i]), cs.s.omega_b)
        k = findfirst(==("$(cs.s.conv[i].name).battery"), S.ports)
        pw += g.dc.Eb * q.i_b - y[k] * S.v[k]          # the ledger's Thevenin source power in place of the Norton one
    end
    return max(abs(pw - supI), abs(L.interconnection)) / (sum(L.tport) + 1e-300)
end

# z' R z against the ledger's losses of the included components (the battery reconciled: Norton loss = Thevenin loss
# + (E_b/R_b) s - E_b i_b, s the sum of the DC voltages), relative to the throughput; and the named ports' powers
function _ph_losses(s::PHSystem, x, u, S)
    L = phs_ledger(s, x, u)
    d = S.z' * S.R * S.z
    return abs(d - L.losses) / (sum(L.tport) + 1e-300)
end
function _ph_losses(cs::PHControlled, x, u, S)
    L = phs_ledger(cs, x, u)
    names = L.names
    keep = trues(length(names))
    for j in eachindex(cs.exc)
        cs.exc[j] isa PHBrushlessExciter && (keep[findfirst(==(_P.name(cs.exc[j]) * ".exciter"), names)] = false)
    end
    want = sum(L.loss[keep])
    y = S.G' * S.z
    err = 0.0
    fl = _P.phs_flows(cs.s, x, phs_control_inputs(cs, x, u.gf; cf = u.cf))
    for i in _P._dc_converters(cs)
        g = cs.conv[i]
        xc = view(x, cs.coff[i]:cs.coff[i]+_P.nstates(g)-1)
        q = converter_dc_powers(g, xc, _P._conv_current(fl, cs.s.conv[i]), cs.s.omega_b)
        k = findfirst(==("$(cs.s.conv[i].name).battery"), S.ports)
        sN = y[k] * S.v[k]                                  # (E_b / R_b) s
        want += sN - g.dc.Eb * q.i_b                        # the Norton loss less the Thevenin loss
        err = max(err, abs(y[k] - sum(xc[_P._dcidx(g)+k2] for k2 in 0:_P.nbranches(g.dc))) / abs(y[k]))
    end
    for j in eachindex(cs.exc)                              # the amplifier's power b VR i_f
        cs.exc[j] isa PHShuntExciter || continue
        k = findfirst(==("$(_P.name(cs.exc[j])).amplifier"), S.ports)
        err = max(err, abs(y[k] * S.v[k] - _P._exc_at(cs, x, j).amp) / (abs(_P._exc_at(cs, x, j).amp) + 1e-300))
    end
    for j in _P._hydro_units(cs)                            # the reservoir's supply K q_t
        k = findfirst(==("$(_P.name(cs.gov[j])).reservoir"), S.ports)
        sup = cs.units[j] * _P._hydro_powers(cs.gov[j], _P._gov_states(cs, x, j)).supply
        err = max(err, abs(y[k] * S.v[k] - sup) / (abs(sup) + 1e-12))
    end
    d = S.z' * S.R * S.z
    return max(err, abs(d - want) / (sum(L.tport) + 1e-300))
end

# n seeded states around x0: the included states perturbed by `a` of their scales, the signal states (controllers,
# modulations) by `b`, the steam and brushless rows kept
function _ph_draws(m, x0, u, n; a = 0.2, b = 0.05, seed = 1, keep = Int[])
    rng = MersenneTwister(seed)
    S = phs_structure(m, x0, u)
    sc = _P.phs_scales(m, x0)
    sig = setdiff(1:length(x0), S.idx, keep)
    out = Vector{Vector{Float64}}()
    for _ in 1:n
        x = copy(x0)
        x[S.idx] .+= a .* sc[S.idx] .* randn(rng, length(S.idx))
        x[sig] .+= b .* sc[sig] .* randn(rng, length(sig))
        if m isa PHControlled                     # the hydro flows and heads keep their sign (the turbine's domain)
            for j in _P._hydro_units(m)
                r = m.goff[j]+3:m.goff[j]+_P.nstates(m.gov[j])-1
                x[r] .= x0[r] .* (1 .+ 0.1 .* randn(rng, length(r)))
            end
        end
        push!(out, x)
    end
    return out
end

# the states that must stay at their rest values: steam units and brushless exciters (excluded, with domain limits)
function _ph_frozen(cs::PHControlled)
    k = Int[]
    for j in eachindex(cs.gov)
        cs.gov[j] isa PHSteamUnit && append!(k, cs.goff[j]:cs.goff[j]+_P.nstates(cs.gov[j])-1)
        cs.gov[j] isa PHHydroUnit && append!(k, cs.goff[j]:cs.goff[j]+2)       # the governor and the gate
    end
    for j in eachindex(cs.exc)
        cs.exc[j] isa PHBrushlessExciter && append!(k, cs.xoff[j]:cs.xoff[j]+2)
    end
    return k
end
_ph_frozen(::PHSystem) = Int[]

function _ph_suite(m, x0, u, n; label, kw...)
    draws = _ph_draws(m, x0, u, n; keep = _ph_frozen(m), kw...)
    rs = [_ph_check(m, x, u; label) for x in vcat([x0], draws)]
    worst = (; res = maximum(r.res for r in rs), grad = maximum(r.grad for r in rs), loss = maximum(r.loss for r in rs),
             emin = minimum(r.emin for r in rs))
    @info "phs_structure: $label" n = length(rs) states = length(rs[1].S.idx) worst
    return all(r.ok for r in rs), rs
end

const _NS = 50          # random states per case (plus the rest point and the edge states)

@testset "port-Hamiltonian structure (phs_structure)" begin
    sc = load_scenario(joinpath(ROOT, "cases", "IEEE39Bus_PF", "no_fault_phtrue.json"))
    eq = solve_equilibrium(load_case(sc.system_path), sc)

    @testset "grid model: network, machines, loads; both frames; a symmetric fault" begin
        s, x0, u, _ = phs_equilibrium(eq)
        ok, _ = _ph_suite(s, x0, u, _NS; label = "grid, synchronous frame")
        @test ok
        uf = PHInput{Float64}(u.Tm, u.vf, copy(u.gf), u.const_power, u.uc, u.cf)
        uf.gf[findfirst(==(16), s.net.net.bus_ids)] = 1000.0
        ok, _ = _ph_suite(s, x0, uf, 10; label = "grid, bus-16 fault")
        @test ok
        sst = phs_frame(s, :stationary; theta0 = 0.3)
        xst = _P.phs_rotate_network(s, x0, 0.3)
        ust = PHInput{Float64}(u.Tm, u.vf, u.gf, u.const_power, u.uc, [cis(0.7)])
        ok, _ = _ph_suite(sst, xst, ust, 10; label = "grid, stationary frame")
        @test ok
        # the same matrices as phs_matrices where both apply: perturbed states, varied inputs, both frames, a fault
        rng = MersenneTwister(7)
        same = Float64[]
        for (sm, xm, um) in ((s, x0, u), (s, x0, uf), (sst, xst, ust))
            for xd in _ph_draws(sm, xm, um, 5; seed = 9)
                uv = PHInput{Float64}(um.Tm .* (1 .+ 0.3 .* randn(rng, length(um.Tm))), um.vf .* (1 .+ 0.3 .* randn(rng, length(um.vf))),
                                      um.gf, um.const_power, um.uc, um.cf)
                M, S = phs_matrices(sm, xd, uv), phs_structure(sm, xd, uv)
                push!(same, opnorm(M.J - S.J) / opnorm(M.J), opnorm(M.R - S.R) / opnorm(M.R),
                      maximum(abs, M.z - S.z) / maximum(abs, M.z), maximum(abs, M.g - S.G * S.v) / maximum(abs, M.g))
            end
        end
        @test maximum(same) < 1e-12
    end

    @testset "composite loads: motors (stall, partial and full trips) and static parts" begin
        relays = (; vtr1 = 0.8, ttr1 = 2.0, ftr1 = 1.0, vtr2 = 0.6, ttr2 = 0.16, ftr2 = 1.0, Tr = 50e-6, types = (:A, :B, :C))
        s, x0, u, _ = phs_equilibrium(eq; loads = :composite, composite = (; relays))
        ok, _ = _ph_suite(s, x0, u, _NS; label = "composite loads")
        @test ok
        mo = s.motors
        xe = copy(x0)
        xe[mo[1].offset+6] = 0.0                               # a stalled motor
        xe[mo[2].ridx] = 0.5; xe[mo[2].offset+6] *= 0.5        # half the units of another tripped (speed kept)
        xe[mo[3].ridx] = 0.0; xe[mo[3].offset+6] = 0.0         # every unit of a third tripped
        r = _ph_check(s, xe, u; label = "composite loads, stall and trips")
        @test r.ok
        @test !any(in(r.S.idx), mo[3].offset:mo[3].offset+6)
    end

    @testset "zero axis and asymmetric faults" begin
        dq0 = (; zero = :physical, zero_pf_json = joinpath(ROOT, "cases", "IEEE39Bus_FullModel", "ieee39_base.json"))
        s, x0, u, _ = phs_equilibrium(eq; dq0)
        ok, _ = _ph_suite(s, x0, u, 20; label = "dq0, no fault")
        @test ok
        ran = 0
        for (bus, kind) in ((16, :ag), (16, :ab), (16, :abg), (4, :bcg), (39, :cag))
            uf = PHInput{Float64}(u.Tm, u.vf, copy(u.gf), u.const_power, u.uc, u.cf)
            _P._fault_conductance!(uf.gf, s, [PHFault(bus, 0.01, 0.0, 1.0; kind)], 0.5)
            @test !isempty(s.asym.af)
            ok, _ = _ph_suite(s, x0, uf, 10; label = "dq0, $kind fault at bus $bus")
            @test ok
            ran += 1
            empty!(s.asym.af)
        end
        @test ran == 5
        # the model refuses a fault at a generator bus whose terminal the series merge removed
        @test_throws ArgumentError _P._fault_conductance!(copy(u.gf), s, [PHFault(30, 0.01, 0.0, 1.0; kind = :ag)], 0.5)
        empty!(s.asym.af)
    end

    @testset "full IEEE-39 model: DC shunt exciters, hydro unit (steam and brushless excluded)" begin
        m = phs_full_model(joinpath(ROOT, "cases", "IEEE39Bus_FullModel"))
        cs, x0, u = m.cs, m.x0, m.u
        ok, rs = _ph_suite(cs, x0, u, _NS; label = "full model")
        @test ok
        S = rs[1].S
        @test any(occursin("steam unit", e) for e in S.excluded) && any(occursin("brushless", e) for e in S.excluded)
        # edge states: exciter flux signs and saturation, regulator limits, hydro friction
        shunt = [j for j in eachindex(cs.exc) if cs.exc[j] isa PHShuntExciter]
        for (k, (sf, xa)) in enumerate(((-0.5, 0.0), (2.5, 1e3), (1.0, -1e3), (0.0, 0.0)))
            xe = copy(x0)
            for j in shunt
                xe[cs.xoff[j]] *= sf
                xe[cs.xoff[j]+1] += xa
            end
            r = _ph_check(cs, xe, u; label = "full model, exciter edge $k")
            @test r.ok
        end
        j = only(_P._hydro_units(cs))
        g = cs.gov[j]
        cs2 = deepcopy(cs)
        cs2.gov[j] = PHHydroUnit(g.name, g.p, g.GMAX, g.Tw, g.Tw_case, g.Te, g.N, 0.02)
        ok, _ = _ph_suite(cs2, x0, u, 10; label = "full model, penstock friction")
        @test ok
        # the ledger's hydro rate along a given dx (review 86 item 12; 86b, 1): flows and heads away from rest
        r = cs.goff[j]+3:cs.goff[j]+_P.nstates(g)-1
        ch = findfirst(==(_P.name(g) * ".hydro"), phs_ledger(cs, x0, u).names)
        rng = MersenneTwister(3)
        for (cm, xd) in ((cs, _ph_draws(cs, x0, u, 3; seed = 4, keep = _ph_frozen(cs))[1]), (cs2, _ph_draws(cs2, x0, u, 3; seed = 5, keep = _ph_frozen(cs2))[2]))
            @test maximum(abs, xd[r] .- x0[r]) > 1e-3                         # nonuniform flows, heads off rest
            dxr = randn(rng, length(xd))
            Ld = phs_ledger(cm, xd, u; dx = dxr)
            E(w) = cm.units[j] * _P._hydro_powers(cm.gov[j], _P._gov_states(cm, w, j)).energy
            exact = ForwardDiff.gradient(E, xd)' * dxr
            @test Ld.rate[ch] ≈ exact rtol = 1e-12
            L1 = phs_ledger(cm, xd, u; dx = phs_rhs(cm, xd, u))
            @test L1.rate[ch] ≈ L1.port[ch] - L1.loss[ch] rtol = 1e-10
        end
        L2 = phs_ledger(cs2, _ph_draws(cs2, x0, u, 1; seed = 6, keep = _ph_frozen(cs2))[1], u)
        @test L2.loss[ch] > 0                                                  # friction is a loss
        # closure: the gate shut on a moving column (water hammer), and the refused open gate at a non-positive head
        xc = copy(x0); xc[cs.goff[j]+2] = 0.0
        rc = _ph_check(cs, xc, u; label = "full model, gate closed on a moving column")
        @test rc.ok
        xn = copy(x0); xn[cs.goff[j]+2+2g.N] = -0.1
        @test_throws DomainError phs_structure(cs, xn, u)
    end

    @testset "inelastic hydro column" begin
        cs, x0, u, _ = phs_controlled(eq; hydro = (; Tw = :match))
        @test any(g -> g isa PHHydroUnit && g.N == 0, cs.gov)
        ok, _ = _ph_suite(cs, x0, u, 20; label = "inelastic hydro")
        @test ok
        # closure (review 86b, 5): the closed gate with the column at rest, the open gate with the column at rest, and
        # the refused closed gate with water flowing
        j = only(_P._hydro_units(cs)); o = cs.goff[j]; g = cs.gov[j]
        for (at, q) in ((0.0, 0.0), (0.1 * g.GMAX, 0.0))
            xe = copy(x0); xe[o+2] = at; xe[o+3] = q
            r = _ph_check(cs, xe, u; label = "inelastic hydro, gate $at, flow $q")
            @test r.ok
        end
        xe = copy(x0); xe[o+2] = 0.0; xe[o+3] = 0.2
        @test_throws DomainError phs_structure(cs, xe, u)
    end

    @testset "full IEEE-39 model with the steam units in exergetic form (B4, steam = :exergy)" begin
        # the steam units' storage rows in the form (docs/active/PH_B4_EXERGY.md): their turbines coupled skew with the
        # rotors, their dissipation in R, the throttle and the reheater walls as feedthrough ports, the extractions as
        # ports. Checked: the included rows of the vector field, J skew, the full block [[R, Pf], [Pf', Sf]] PSD, the
        # gradient (steam rows to the tables' Gibbs defect, measured; the rest to round-off), and the balance
        # z' x' = y' v + yf' vf - [z; vf]' Pi [z; vf] with the code's own vector field
        m = phs_full_model(joinpath(ROOT, "cases", "IEEE39Bus_FullModel"))
        cs, x0, u = m.cs, m.x0, m.u
        srows = reduce(vcat, [collect(cs.goff[j]+2:cs.goff[j]+_P.nstates(cs.gov[j])-1) for j in _P._steam_units(cs)])
        rng = MersenneTwister(93)
        worst = (res = 0.0, emin = 0.0, grad = 0.0, grad_M = 0.0, grad_U = 0.0, bal = 0.0, supply = 0.0)
        mrows = reduce(vcat, [collect(cs.goff[j]+2:2:cs.goff[j]+_P.nstates(cs.gov[j])-1) for j in _P._steam_units(cs)])
        urows = mrows .+ 1
        for w in 0:5
            x = copy(x0)
            if w > 0
                S0 = phs_structure(cs, x0, u; steam = :exergy)
                sc = _P.phs_scales(cs, x0)
                oth = setdiff(S0.idx, srows)
                x[oth] .+= 0.05 .* sc[oth] .* randn(rng, length(oth))
                x[srows] .*= 1 .+ 1e-3 .* randn(rng, length(srows))
            end
            S = phs_structure(cs, x, u; steam = :exergy)
            @test all(r -> r in S.idx, srows)
            @test !any(occursin("steam unit", e) for e in S.excluded)
            f = phs_rhs(cs, x, u)
            I = S.idx
            g = (S.J - S.R) * S.z .+ S.G * S.v .- S.Pf * S.vf
            den = abs.(S.J) * abs.(S.z) .+ abs.(S.R) * abs.(S.z) .+ abs.(S.G) * abs.(S.v) .+ abs.(S.Pf) * abs.(S.vf) .+ 1e-14
            res = maximum(abs.(f[I] .- g[I]) ./ den[I])
            @test S.J == -S.J'
            @test isempty(S.steam_violations)        # no flowing branch with a non-positive affinity (table defect)
            Pi = [S.R[I, I] S.Pf[I, :]; S.Pf[I, :]' S.Sf]
            @test Pi == Pi'
            d = sqrt.(max.(diag(Pi), 0.0)); d[d .== 0] .= 1
            emin = eigmin(Symmetric(Pi ./ (d * d')))
            gH = ForwardDiff.gradient(q -> phs_structure_energy(cs, q; steam = :exergy), x)
            oth = setdiff(I, srows)
            grad = maximum(abs.(gH[oth] .- S.z[oth]) ./ (abs.(gH[oth]) .+ 1e-3 * maximum(abs, gH[oth])))
            # componentwise, M rows and U rows each against their own scale (the Gibbs defect, about 7e-6 relative,
            # times the efforts' factors T0/T <= 1 and (u delta_u + p v delta_v); PH_B4_GATE.md section 2)
            grad_M = maximum(abs.(gH[mrows] .- S.z[mrows])) / maximum(abs, S.z[mrows])
            grad_U = maximum(abs.(gH[urows] .- S.z[urows])) / maximum(abs, S.z[urows])
            # the physical steam supplies, independently: c (q_v eps_T + sum Q_k (1 - T0/T_w,k)) per unit
            supT = 0.0; sup_ref = 0.0
            for j in _P._steam_units(cs)
                g = cs.gov[j]; o = cs.goff[j]; c = cs.units[j] / _P.STEAM_SB
                ev = _P._steam_eval(g, x[o+2:o+_P.nstates(g)-1], _P._steam_stroke(g, x[o+1]))
                scs = _P.steam_sat(g.tb, g.p_cond); T0 = scs.T; mu0 = scs.h_f - T0 * scs.s_f
                sT = _P.steam_ph(g.tb, g.p_T, g.h_T).s
                sup_ref += c * (ev.qv * (g.h_T - T0 * sT - mu0) +
                                sum((g.kh[k] * (g.Tw[k] - ev.st[k].T) * (1 - T0 / g.Tw[k]) for k in eachindex(g.kh) if g.kh[k] > 0); init = 0.0))
            end
            yfv = (S.Pf' * S.z .+ S.Sf * S.vf)' * S.vf
            supply = abs(yfv - sup_ref) / abs(sup_ref)
            yv = (S.G' * S.z)' * S.v
            yf = S.Pf' * S.z .+ S.Sf * S.vf
            zz = vcat(S.z[I], S.vf)
            bal = abs(S.z[I]' * f[I] - (yv + yf' * S.vf - zz' * Pi * zz)) / (abs(yv) + abs(yf' * S.vf) + abs(zz' * Pi * zz) + 1e-300)
            worst = (res = max(worst.res, res), emin = min(worst.emin, emin), grad = max(worst.grad, grad),
                     grad_M = max(worst.grad_M, grad_M), grad_U = max(worst.grad_U, grad_U), bal = max(worst.bal, bal),
                     supply = max(worst.supply, supply))
        end
        @info "phs_structure with steam = :exergy" worst
        @test worst.res <= 1e-12
        @test worst.emin >= -1e-12
        @test worst.grad <= 1e-11
        @test worst.grad_M <= 1e-4 && worst.grad_U <= 1e-4   # the tables' Gibbs defect, measured (not a derived bound)
        @test worst.supply <= 1e-12
        @test worst.bal <= 1e-12
    end
    @testset "converters: ideal sources and averaged DC sides (cores where the case has them)" begin
        Ts = 5e-5
        dcs = (; H = 5e-3, battery = :sossan, soc = "40-50")
        cores = Dict(33 => 800.0, 36 => 700.0, 37 => 700.0, 38 => 1000.0)
        for (dir, kw) in (("IEEE39Bus_PF_gfm-vsm-pf", (;)),
                          ("IEEE39Bus_PF_gfm-vsm-pf", (; converter_Ts = Ts, converter_dc = merge(dcs, (; mode = :held)))),
                          ("IEEE39Bus_PF_gfm-droop-pf", (; converter_Ts = Ts, droop_dc_reject = true,
                                                          converter_dc = merge(dcs, (; mode = :held)), cores)),
                          ("IEEE39Bus_PF_gfm-voc-pf", (; converter_Ts = Ts, voc_dc_reject = true,
                                                        converter_dc = merge(dcs, (; mode = :held)))),
                          ("IEEE39Bus_PF_gfl-pf", (; converter_Ts = Ts, gfl_advance = 1.0,
                                                    converter_dc = merge(dcs, (; mode = :continuous)), cores)))
            scc = load_scenario(joinpath(ROOT, "cases", dir, "ph_fault_bus16.json"))
            eqc = solve_equilibrium(load_case(scc.system_path), scc)
            cs, x0, u, _ = phs_controlled(eqc; loads = :series_all, kw...)
            label = "$dir $(keys(kw))"
            ok, _ = _ph_suite(cs, x0, u, 20; label)
            @test ok
            # DC charging and discharging, and saturated cores
            for (k, fdc) in enumerate((0.9, 1.1))
                xe = copy(x0)
                for i in _P._dc_converters(cs)
                    xe[cs.coff[i]+_P._dcidx(cs.conv[i])-1] *= fdc
                end
                for q in cs.s.cores
                    xe[cs.s.bidx[q.pos]:cs.s.bidx[q.pos]+1] .*= 1.6; xe[q.xs:q.xs+1] .*= 1.6
                end
                r = _ph_check(cs, xe, u; label = "$label, edge $k")
                @test r.ok
            end
        end
    end
    @testset "converters: the switching SiC bridge over a step (B3)" begin
        # the droop case with switching bridges (bridge.jl): the bridges started at the rest point, then windows of one
        # step h placed across the first half carrier (pole changes and dead times inside), with a current slope, so the
        # step's m-bar, drops and switching sink are those of a running step
        Ts = 5e-5; h = 0.5e-6
        dcs = (; H = 5e-3, battery = :sossan, soc = "40-50", mode = :held, bridge = :switching)
        cores = Dict(33 => 800.0, 36 => 700.0, 37 => 700.0, 38 => 1000.0)
        scc = load_scenario(joinpath(ROOT, "cases", "IEEE39Bus_PF_gfm-droop-pf", "ph_fault_bus16.json"))
        eqc = solve_equilibrium(load_case(scc.system_path), scc)
        cs, x0, u, _ = phs_controlled(eqc; loads = :series_all, converter_Ts = Ts, droop_dc_reject = true,
                                      converter_dc = dcs, cores)
        @test any(_P._has_sw, cs.conv)
        _P._bridge_start!(cs, x0, u, 0.0)
        br1 = first(g.dc.sw for g in cs.conv if _P._has_sw(g))
        # windows around the first leg's changes (each change and the dead time after it) and in between
        tc = filter(t -> 0 < t < Ts - 2h, br1.tchg[1])
        starts = unique(vcat([t - 0.3h for t in tc], [t + 0.5h for t in tc], [0.3Ts, 0.7Ts]))
        rng = MersenneTwister(5)
        nsw = 0; ncond = 0
        for (w, a) in enumerate(starts)
            _P._bridge_window!(cs, a, a + h)
            for g in cs.conv
                _P._has_sw(g) && (g.dc.sw.islope = 1e3 * randn(rng, ComplexF64))
            end
            S = phs_structure(cs, x0, u)
            @test !any(occursin("switching bridge", e) for e in S.excluded)
            ok, rs = _ph_suite(cs, x0, u, 5; label = "droop, switching bridge, window $w at $(round(a * 1e6; digits = 3)) us",
                               seed = w)
            @test ok
            # non-vacuous: windows with a switching event (p_sw > 0) and with conduction losses
            fl = _P.phs_flows(cs.s, x0, phs_control_inputs(cs, x0, u.gf; cf = u.cf))
            for i in _P._dc_converters(cs)
                g = cs.conv[i]; xc = view(x0, cs.coff[i]:cs.coff[i]+_P.nstates(g)-1)
                r = _P._sw_terms(g, xc, _P._conv_current(fl, cs.s.conv[i]))
                nsw += r.psw > 0; ncond += r.pcond > 0
            end
        end
        _P._bridge_stop!(cs)
        @test length(starts) >= 4
        @test nsw > 0 && ncond > 0
        @info "switching-bridge windows" windows = length(starts) with_switching = nsw with_conduction = ncond
        # a window starting exactly at a pole change (the change belongs to it)
        _P._bridge_start!(cs, x0, u, 0.0)
        _P._bridge_window!(cs, tc[1], tc[1] + h)
        ok, _ = _ph_suite(cs, x0, u, 3; label = "droop, switching bridge, window at a pole change", seed = 99)
        @test ok
        _P._bridge_stop!(cs)
        # a short run with switching bridges: the simulator's energy account closes (the step balance with the bridge's
        # booked losses; review 91, 8)
        rr = phs_simulate(cs, x0, u; tstop = 2Ts, h)
        # two measures (review 91b): `closure` = summed energy increments less the endpoint storage (round-off of the
        # stored energies: absolute, since over 1e-4 s the throughput is only ~2e-3 pu s); `scheme` = summed increments less
        # h (port power - losses) at the midpoints, the step balance with the booked supply and losses (switching and
        # conduction included), which measures the Newton solve. Its reference: the same run with averaged bridges.
        @test maximum(abs, rr.closure) <= 1e-12
        kdc = findall(==(:converter_dc), rr.kind)
        @test !isempty(kdc)
        csa, x0a, ua, _ = phs_controlled(eqc; loads = :series_all, converter_Ts = Ts, droop_dc_reject = true,
                                         converter_dc = merge(dcs, (; bridge = :averaged)), cores)
        ra = phs_simulate(csa, x0a, ua; tstop = 2Ts, h)
        sw_dc, av_dc = maximum(rr.scheme_max[kdc]), maximum(ra.scheme_max[findall(==(:converter_dc), ra.kind)])
        @test rr.total.scheme_max <= 2 * ra.total.scheme_max + 1e-13      # the solve, not the bridge, sets it
        @test sw_dc <= 1e-11                                              # the converters' own step balance (pu s)
        @info "switching-bridge run, 2 Ts = 1e-4 s" closure = maximum(abs, rr.closure) scheme_total = rr.total.scheme_max scheme_dc = sw_dc averaged_scheme_total = ra.total.scheme_max averaged_scheme_dc = av_dc throughput = maximum(abs, rr.throughput)
        # the GFL (continuous DC side) with switching bridges and cores
        dcg = (; H = 5e-3, battery = :sossan, soc = "40-50", mode = :continuous, bridge = :switching)
        scg = load_scenario(joinpath(ROOT, "cases", "IEEE39Bus_PF_gfl-pf", "ph_fault_bus16.json"))
        eqg = solve_equilibrium(load_case(scg.system_path), scg)
        cg, xg, ug, _ = phs_controlled(eqg; loads = :series_all, converter_Ts = Ts, gfl_advance = 1.0, converter_dc = dcg, cores)
        _P._bridge_start!(cg, xg, ug, 0.0)
        bg = first(g.dc.sw for g in cg.conv if _P._has_sw(g))
        tg = filter(t -> 0 < t < Ts - 2h, bg.tchg[1])
        for (w, a) in enumerate(unique(vcat([t - 0.3h for t in tg], [0.5Ts])))
            _P._bridge_window!(cg, a, a + h)
            for g in cg.conv
                _P._has_sw(g) && (g.dc.sw.islope = 1e3 * randn(rng, ComplexF64))
            end
            ok, _ = _ph_suite(cg, xg, ug, 3; label = "GFL, switching bridge, window $w", seed = 100 + w)
            @test ok
        end
        _P._bridge_stop!(cg)
    end
end
