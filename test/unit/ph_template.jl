# The component template, T1 and T2 (PH_TODO.md section 4; docs/active/PH_COMPONENT_TEMPLATE.md section 8): the
# hydro unit's physical subsystem declared once (`HydroPhysicalTemplate`), its generated forms against today's
# hand-written code. Elastic (the full model's 28-section penstock, with and without friction) and inelastic; one
# unit and two in parallel; random admissible states, gates between the limits, speeds off nominal.
# Checks: the generated rows against `_penstock_rhs!` (elastic) or (1 - head)/Tw (inelastic); the rotor torque against
# units * `_hydro_power` / omega; the energy, loss and supply against `_hydro_powers`; the gradient against
# `phs_structure`'s efforts and the local J, R against `phs_structure`'s hydro block (the hand-assembled form); the
# balance H' = supply - loss + the power received through the internal ports.

using Test, LinearAlgebra, Random
using PHPS
const _PT = PHPS

function _tpl_hydro_case(cs, x, u, j; units = cs.units[j], omega_scale = 1.0)
    g = cs.gov[j]; o = cs.goff[j]; mch = cs.s.machines[j]; ip = _PT._ip(mch)
    xg = x[o:o+_PT.nstates(g)-1]
    at = xg[3]
    om = omega_scale * x[ip] / (2 * mch.w.H)
    c = _PT.HydroPhysicalTemplate(g, units, _PT._hydro_gate(g, at))   # the live clamp, as the outputs use
    xl = xg[4:end]
    r = _PT.tpl_rows(c, xl, [om])
    acc = _PT.tpl_account(c, xl, [om])
    # the hand-written rows
    dxh = zeros(length(xg))
    if g.N == 0
        dxh[4] = (1 - _PT._hydro_head(g, _PT._hydro_gate(g, at), xg[4])) / g.Tw
    else
        _PT._penstock_rhs!(dxh, g, xg, _PT._hydro_gate(g, at))
    end
    pw = _PT._hydro_powers(g, xg)
    den = (abs.(r.blk.J) .+ abs.(r.blk.R)) * abs.(vcat(r.z, om)) .+ abs.(r.blk.G) * abs.(r.blk.v)
    rows = maximum(abs.(r.dx .- dxh[4:end]) ./ (den[1:end-1] .+ 1e-300))
    tq = units * _PT._hydro_power(g, xg) / om
    torque = abs(r.flows[1] - tq) / abs(tq)
    sc = units * pw.tport
    energy = abs(acc.energy - units * pw.energy) / (units * pw.energy + 1e-300)
    loss = abs(acc.loss - units * pw.loss) / sc
    supply = abs(acc.supply - units * pw.supply) / sc
    bal = abs(dot(r.z, r.dx) - (acc.supply - acc.loss + acc.port_power)) / sc
    return (; rows, torque, energy, loss, supply, bal, skew = maximum(abs, r.blk.J + r.blk.J'),
            emin = eigmin(Symmetric(r.blk.R)), r, c, xl, om, ip, o, g)
end

@testset "component template: the hydro unit's physical subsystem (T1, T2)" begin
    rng = MersenneTwister(87)
    worst = Dict{Symbol,Float64}()
    function acc!(q)
        for k in (:rows, :torque, :energy, :loss, :supply, :bal, :skew)
            worst[k] = max(get(worst, k, 0.0), getfield(q, k))
        end
        worst[:emin] = min(get(worst, :emin, 0.0), q.emin)
    end
    # elastic: the full model's hydro unit
    m = phs_full_model(joinpath(ROOT, "cases", "IEEE39Bus_FullModel"))
    cs, x0, u = m.cs, m.x0, m.u
    j = only(_PT._hydro_units(cs))
    g = cs.gov[j]
    @test g.N > 0
    cases = 0
    for units in (cs.units[j], 2.0), w in 0:15
        x = copy(x0); o = cs.goff[j]
        if w > 0
            r = o+3:o+_PT.nstates(g)-1
            x[r] .*= 1 .+ 0.1 .* randn(rng, length(r))
            x[o+2] = clamp(x[o+2] * (1 + 0.2 * randn(rng)), 0.05 * g.GMAX, g.GMAX)
        end
        q = _tpl_hydro_case(cs, x, u, j; units, omega_scale = w == 0 ? 1.0 : 1 + 0.02 * randn(rng))
        acc!(q); cases += 1
    end
    # with friction
    cs2 = deepcopy(cs)
    cs2.gov[j] = PHHydroUnit(g.name, g.p, g.GMAX, g.Tw, g.Tw_case, g.Te, g.N, 0.02)
    for w in 0:5
        x = copy(x0); o = cs.goff[j]
        w > 0 && (r = o+3:o+_PT.nstates(g)-1; x[r] .*= 1 .+ 0.1 .* randn(rng, length(r)))
        q = _tpl_hydro_case(cs2, x, u, j)
        acc!(q); cases += 1
    end
    # the generated local structure against phs_structure's hydro block (the hand-assembled form) at the rest point
    S = phs_structure(cs, x0, u)
    q = _tpl_hydro_case(cs, x0, u, j)
    rows_g = vcat(collect(q.o+3:q.o+_PT.nstates(g)-1), q.ip)
    @test maximum(abs, S.J[rows_g, rows_g] .- q.r.blk.J) <= 1e-12 * maximum(abs, q.r.blk.J)
    @test maximum(abs, S.z[q.o+3:q.o+_PT.nstates(g)-1] .- q.r.z) <= 1e-12 * maximum(abs, q.r.z)
    # inelastic
    sc = load_scenario(joinpath(ROOT, "cases", "IEEE39Bus_PF", "no_fault_phtrue.json"))
    eq = solve_equilibrium(load_case(sc.system_path), sc)
    ci, xi, ui, _ = phs_controlled(eq; hydro = (; Tw = :match))
    ji = only(_PT._hydro_units(ci))
    @test ci.gov[ji].N == 0
    for w in 0:10
        x = copy(xi); o = ci.goff[ji]; gi = ci.gov[ji]
        if w > 0
            x[o+3] *= 1 + 0.1 * randn(rng)
            x[o+2] = clamp(x[o+2] * (1 + 0.2 * randn(rng)), 0.05 * gi.GMAX, gi.GMAX)
        end
        q = _tpl_hydro_case(ci, x, ui, ji)
        acc!(q); cases += 1
    end
    Si = phs_structure(ci, xi, ui)
    q = _tpl_hydro_case(ci, xi, ui, ji)
    rows_i = [q.o + 3, q.ip]
    @test maximum(abs, Si.J[rows_i, rows_i] .- q.r.blk.J) <= 1e-12 * maximum(abs, q.r.blk.J)
    # the domain is the component's own
    xe = copy(xi); xe[ci.goff[ji]+2] = 0.0; xe[ci.goff[ji]+3] = 0.2
    @test_throws DomainError _tpl_hydro_case(ci, xe, ui, ji)
    # review 97: Jacobians through the template (states, speed, gate) with dual numbers, against finite differences
    q0 = _tpl_hydro_case(cs, x0, u, j)
    cf(xl, om, at) = _PT.tpl_rows(_PT.HydroPhysicalTemplate(q0.g, cs.units[j], at), xl, [om]).dx
    Jx = ForwardDiff.jacobian(xl -> cf(xl, q0.om, q0.c.at), q0.xl)
    Jw = ForwardDiff.derivative(om -> cf(q0.xl, om, q0.c.at), q0.om)
    Ja = ForwardDiff.derivative(at -> cf(q0.xl, q0.om, at), q0.c.at)
    hfd = 1e-7
    Jxf = reduce(hcat, [(cf(q0.xl .+ hfd .* (1:length(q0.xl) .== k), q0.om, q0.c.at) .-
                         cf(q0.xl .- hfd .* (1:length(q0.xl) .== k), q0.om, q0.c.at)) ./ (2hfd) for k in 1:length(q0.xl)])
    @test maximum(abs, Jx .- Jxf) <= 1e-5 * maximum(abs, Jx)
    @test all(isfinite, Jw) && all(isfinite, Ja)
    # multiplicity validated
    @test_throws ArgumentError _PT.HydroPhysicalTemplate(g, 0.0, 1.0)
    @test_throws ArgumentError _PT.HydroPhysicalTemplate(g, -1.0, 1.0)
    # named port powers: the reservoir's supply and the rotor's received power
    pp = _PT.tpl_port_powers(q0.c, q0.xl, [q0.om])
    @test pp["reservoir"] ≈ cs.units[j] * _PT._hydro_powers(g, x0[cs.goff[j]:cs.goff[j]+_PT.nstates(g)-1]).supply rtol = 1e-13
    @test pp["rotor"] ≈ -cs.units[j] * _PT._hydro_power(g, x0[cs.goff[j]:cs.goff[j]+_PT.nstates(g)-1]) rtol = 1e-13
    @info "template: hydro physical subsystem" cases worst
    @test cases == 49
    @test worst[:rows] <= 1e-13
    @test worst[:torque] <= 1e-13
    @test worst[:energy] <= 1e-13
    @test worst[:loss] <= 1e-13
    @test worst[:supply] <= 1e-13
    @test worst[:bal] <= 1e-12
    @test worst[:skew] == 0
    @test worst[:emin] >= 0
end

@testset "component template: a saturating core (T3: nonlinear storage, K3, the frame clock)" begin
    # the droop converter case with cores: each cored branch as a CoreTemplate; its generated rows against the
    # simulator's vector field on the branch's two loop rows, its gradient against the system's (phs_gradient) on those
    # rows, and the frame port's power zero (rotation invariance); at the rest point, at states driven into saturation,
    # and at random clock angles
    Ts = 5e-5
    dcs = (; H = 5e-3, battery = :sossan, soc = "40-50", mode = :held)
    cores = Dict(33 => 800.0, 36 => 700.0, 37 => 700.0, 38 => 1000.0)
    scc = load_scenario(joinpath(ROOT, "cases", "IEEE39Bus_PF_gfm-droop-pf", "ph_fault_bus16.json"))
    eqc = solve_equilibrium(load_case(scc.system_path), scc)
    cs, x0, u, _ = phs_controlled(eqc; loads = :series_all, converter_Ts = Ts, droop_dc_reject = true,
                                  converter_dc = merge(dcs, (; bridge = :averaged)), cores)
    s = cs.s
    @test !isempty(s.cores)
    rng = MersenneTwister(33)
    worst = (rows = 0.0, grad = 0.0, frame = 0.0); n = 0
    for w in 0:6
        x = copy(x0)
        if w > 0                                   # saturation and other clock angles
            for q in s.cores
                f = w <= 3 ? 1.0 + 0.4 * w : 1.0
                x[s.bidx[q.pos]:s.bidx[q.pos]+1] .*= f; x[q.xs:q.xs+1] .*= f
            end
            w > 3 && (x[s.iclk] += 2pi * rand(rng))
        end
        f = phs_rhs(cs, x, u)
        fl = _PT.phs_flows(s, x, phs_control_inputs(cs, x, u.gf; cf = u.cf))
        zs = _PT.phs_gradient(s, view(x, 1:s.n))
        pn = s.net
        for q in s.cores
            l = s.branches[q.pos]; r = s.bidx[q.pos]; rs = q.xs
            c = _PT.CoreTemplate(q, s.omega_b, complex(pn.a[l]))
            xl = [x[r], x[r+1], x[rs], x[rs+1], x[s.iclk]]
            vf, vt = fl.v[pn.from[l]], fl.v[pn.to[l]]
            res = _PT.tpl_rows(c, xl, [real(vf), imag(vf), real(vt), imag(vt)])
            fh = [f[r], f[r+1], f[rs], f[rs+1]]
            den = ((abs.(res.blk.J) .+ abs.(res.blk.R)) * abs.(vcat(res.z, real(vf), imag(vf), real(vt), imag(vt))))[1:4] .+
                  (abs.(res.blk.G) * [1.0])[1:4] .+ 1e-300
            worst = (rows = max(worst.rows, maximum(abs.(res.dx[1:4] .- fh) ./ den)),
                     grad = max(worst.grad, maximum(abs.(res.z[1:4] .- [zs[r], zs[r+1], zs[rs], zs[rs+1]])) / maximum(abs, res.z[1:4])),
                     frame = max(worst.frame, abs(dot(res.z, res.blk.G[1:5, 1])) / (sum(abs.(res.z .* res.blk.G[1:5, 1])) + 1e-300)))
            n += 1
        end
    end
    @info "template: saturating core" n worst
    @test n >= 20
    @test worst.rows <= 1e-12
    @test worst.grad <= 1e-12
    @test worst.frame <= 1e-12
end


@testset "component template: a reduced inductor-only node (T3: K2)" begin
    # the IEEE-39 grid with series R-L loads: every inductor-only node without a machine as a CutTemplate declared from
    # its unreduced inductors; the generic reduction against ph_system's M, Rm; the generated rows against the simulator;
    # the gradient against phs_gradient; the node current reconstructed against Kirchhoff
    sc = load_scenario(joinpath(ROOT, "cases", "IEEE39Bus_PF", "no_fault_phtrue.json"))
    eq = solve_equilibrium(load_case(sc.system_path), sc)
    s, x0, u, _ = phs_equilibrium(eq; loads = :series_all)
    cuts = [c for c in s.cuts if c.machine == 0]
    @test !isempty(cuts)
    pn = s.net
    rng = MersenneTwister(12)
    worst = (M = 0.0, rows = 0.0, grad = 0.0); n = 0
    for w in 0:5
        x = copy(x0)
        w > 0 && (x[1:s.n] .*= 1 .+ 0.02 .* randn(rng, s.n))
        f = phs_rhs(s, x, u)
        fl = _PT.phs_flows(s, x, u)
        zs = _PT.phs_gradient(s, view(x, 1:s.n))
        for cut in cuts
            ls = [s.branches[j] for j in cut.pos]
            c = _PT.CutTemplate(pn.L[ls], pn.R[ls], cut.LL, cut.RL, cut.gamma, cut.c, s.omega_b; stationary = s.stationary)
            red = _PT.tpl_reduce(c)
            rows = vcat([[s.bidx[j], s.bidx[j] + 1] for j in cut.pos]...)
            xl = x[rows]
            ext = vcat([[real(fl.v[k]), imag(fl.v[k])] for k in cut.far]...)
            r = _PT.tpl_rows(c, xl, ext)
            den = ((abs.(r.blk.J) .+ abs.(r.blk.R)) * abs.(vcat(r.z, ext)))[1:length(xl)] .+ 1e-300
            # the eliminated node current against the simulator's (review 98, Codex 3)
            ci = findfirst(cc -> cc === cut, s.cuts)
            @test abs(_PT.tpl_reconstruct(c, xl).iL - fl.iload[ci]) <= 1e-12 * (abs(fl.iload[ci]) + 1e-12)
            # the named port powers add up to the account's (complex pairs, review 98, Codex 5)
            pp = _PT.tpl_port_powers(c, xl, ext)
            @test abs(sum(values(pp)) - _PT.tpl_account(c, xl, ext).port_power) <= 1e-12 * (abs(_PT.tpl_account(c, xl, ext).port_power) + 1e-12)
            worst = (M = max(worst.M, maximum(abs, red.M .- cut.M) / maximum(abs, cut.M), maximum(abs, red.Rm .- cut.Rm) / maximum(abs, cut.Rm)),
                     rows = max(worst.rows, maximum(abs.(r.dx .- f[rows]) ./ den)),
                     grad = max(worst.grad, maximum(abs.(r.z .- zs[rows])) / maximum(abs, r.z)))
            n += 1
        end
    end
    @info "template: reduced node" n worst
    @test n >= 6
    @test worst.M <= 1e-14
    @test worst.rows <= 1e-12
    @test worst.grad <= 1e-12
end

@testset "component template: the hydro governor's execution (T2, second half)" begin
    # the unit's rows from the two declarations (signal law + physical part) against the hand-written `_hydro_rhs!`,
    # with the decision sequence (ModeLog) compared: at the rest point, with the pilot valve and the gate driven to and
    # beyond their limits (engaged, released), with evolving governor states and speeds off nominal; and the
    # derivative of the rows with respect to the governor's states and the speed (dual numbers) through both
    m = phs_full_model(joinpath(ROOT, "cases", "IEEE39Bus_FullModel"))
    cs, x0 = m.cs, m.x0
    j = only(_PT._hydro_units(cs)); g = cs.gov[j]; o = cs.goff[j]; p = g.p
    xg0 = x0[o:o+_PT.nstates(g)-1]
    Pref = p.Sigma * _PT._hydro_gate(g, xg0[3])          # the rest point's reference
    GMIN, GMAX = p.PMIN * p.R_base, p.PMAX * p.R_base
    RUP, RDN = p.UO * p.R_base, p.UC * p.R_base
    gov = _PT.HydroGovernorSignal(g)
    rng = MersenneTwister(97)
    worst = 0.0; ndec = 0; nlim = 0; ncase = 0
    states = Any[(xg0, 1.0)]
    for _ in 1:10
        xg = copy(xg0); xg[1] += 0.05 * randn(rng); xg[2] *= 1 + 0.1 * randn(rng); xg[3] *= 1 + 0.1 * randn(rng)
        push!(states, (xg, 1 + 0.02 * randn(rng)))
    end
    for (a1, a3) in ((RUP * 1.2, xg0[3]), (RDN * 1.2, xg0[3]), (0.0, GMAX * 1.05), (0.0, GMIN - 0.01), (RUP * 1.2, GMAX * 1.05))
        xg = copy(xg0); xg[1] = a1; xg[3] = a3; push!(states, (xg, 1.0)); nlim += 1
    end
    for (xg, om) in states
        r1 = _PT.ModeLog(); r2 = _PT.ModeLog()
        dxh = _PT._hydro_rhs!(zeros(length(xg)), g, xg, om, Pref, r1)
        dxt = try
            _PT.tpl_unit_rows(gov, 1.0, xg, om, Pref, r2)
        catch err
            err isa DomainError || rethrow(); continue
        end
        @test r1.decisions == r2.decisions
        worst = max(worst, maximum(abs.(dxt .- dxh)) / (maximum(abs, dxh) + 1))   # (all rates 0 at rest)
        ndec += length(r1.decisions); ncase += 1
    end
    # derivatives through the composition (the governor's states and the speed)
    jt = ForwardDiff.jacobian(z -> _PT.tpl_unit_rows(gov, 1.0, vcat(z[1:3], xg0[4:end]), z[4], Pref, _PT.NoModes()), vcat(xg0[1:3], 1.0))
    jh = ForwardDiff.jacobian(z -> _PT._hydro_rhs!(zeros(eltype(z), length(xg0)), g, vcat(z[1:3], xg0[4:end]), z[4], Pref, _PT.NoModes()), vcat(xg0[1:3], 1.0))
    @info "template: hydro governor execution" ncase nlim ndec worst
    @test ncase >= 14 && nlim == 5
    @test worst <= 1e-13          # round-off: the physical rows as (J - R) z, another order of operations
    @test maximum(abs, jt .- jh) <= 1e-12 * maximum(abs, jh)
end

@testset "component template: the specialised evaluation equals the reference (T4)" begin
    m = phs_full_model(joinpath(ROOT, "cases", "IEEE39Bus_FullModel"))
    cs, x0 = m.cs, m.x0
    j = only(_PT._hydro_units(cs)); g = cs.gov[j]; o = cs.goff[j]
    xg = x0[o:o+_PT.nstates(g)-1]; mch = cs.s.machines[j]; om = x0[_PT._ip(mch)] / (2 * mch.w.H)
    rng = MersenneTwister(4)
    for w in 0:5
        xl = xg[4:end] .* (w == 0 ? 1.0 : (1 .+ 0.05 .* randn(rng, length(xg) - 3)))
        c = _PT.HydroPhysicalTemplate(g, cs.units[j], _PT._hydro_gate(g, xg[3]))
        ref = _PT.tpl_rows(c, xl, [om])
        dx = zeros(length(xl)); fl = _PT.tpl_rows!(dx, c, xl, [om])
        @test maximum(abs, dx .- ref.dx) <= 1e-14 * (maximum(abs, ref.dx) + 1)
        @test abs(fl[1] - ref.flows[1]) <= 1e-14 * abs(ref.flows[1])
        @test maximum(abs, _PT.tpl_gradient(c, xl) .- ForwardDiff.gradient(q -> _PT.tpl_energy(c, q), xl)) <= 1e-12 * maximum(abs, ref.z)
        ch = _PT.TplCache(c); dx2 = zeros(length(xl)); tq = _PT.tpl_rows!(dx2, ch, c, xl, om)
        # the same refusals as the reference (review 98): non-positive speed, reverse elastic flow
        @test_throws DomainError _PT.tpl_rows!(dx2, ch, c, xl, 0.0)
        xr = copy(xl); xr[1] = -abs(xr[1]) - 0.1
        @test_throws DomainError _PT.tpl_rows!(dx2, ch, c, xr, om)
        @test_throws DomainError _PT.tpl_rows(c, xr, [om])
        @test maximum(abs, dx2 .- ref.dx) <= 1e-14 * (maximum(abs, ref.dx) + 1)
        @test abs(tq - ref.flows[1]) <= 1e-14 * abs(ref.flows[1])
    end
end


@testset "component template: the cached evaluation of the inelastic column (T4, N = 0)" begin
    sc = load_scenario(joinpath(ROOT, "cases", "IEEE39Bus_PF", "no_fault_phtrue.json"))
    eq = solve_equilibrium(load_case(sc.system_path), sc)
    ci, xi, ui, _ = phs_controlled(eq; hydro = (; Tw = :match))
    ji = only(_PT._hydro_units(ci)); gi = ci.gov[ji]; o = ci.goff[ji]
    @test gi.N == 0
    om = 1.0; rng = MersenneTwister(5)
    for w in 0:5
        xg = xi[o:o+3] .* (w == 0 ? 1.0 : (1 .+ 0.05 .* randn(rng, 4)))
        c = _PT.HydroPhysicalTemplate(gi, 1.0, _PT._hydro_gate(gi, xg[3]))
        ref = _PT.tpl_rows(c, xg[4:4], [om])
        ch = _PT.TplCache(c); dx = zeros(1); tq = _PT.tpl_rows!(dx, ch, c, xg[4:4], om)
        @test abs(dx[1] - ref.dx[1]) <= 1e-14 * (abs(ref.dx[1]) + 1)
        @test abs(tq - ref.flows[1]) <= 1e-14 * abs(ref.flows[1])
        @test maximum(abs, _PT.tpl_gradient(c, xg[4:4]) .- _PT.tpl_gradient_ad(c, xg[4:4])) <= 1e-13 * abs(ref.z[1])
    end
end

@testset "component template: the generic assembler and the interconnection as data (T1)" begin
    # the full model's hydro unit placed by data (its rows, its rotor port), assembled generically; its entries in the
    # global J and R against phs_structure's on every row and column the unit touches
    m = phs_full_model(joinpath(ROOT, "cases", "IEEE39Bus_FullModel"))
    cs, x0, u = m.cs, m.x0, m.u
    j = only(_PT._hydro_units(cs)); g = cs.gov[j]; o = cs.goff[j]; mch = cs.s.machines[j]; ip = _PT._ip(mch)
    S = phs_structure(cs, x0, u)
    hrows = collect(o+3:o+_PT.nstates(g)-1)
    pl = [_PT.TplPlacement(_PT.HydroPhysicalTemplate(g, cs.units[j], _PT._hydro_gate(g, x0[o+2])), hrows, [ip])]
    A = _PT.tpl_assemble(pl, length(x0), x0, S.z)
    @test maximum(abs, A.J[hrows, :] .- S.J[hrows, :]) <= 1e-12 * maximum(abs, A.J[hrows, :])
    @test maximum(abs, A.J[:, hrows] .- S.J[:, hrows]) <= 1e-12 * maximum(abs, A.J[:, hrows])
    @test maximum(abs, A.R[hrows, hrows] .- S.R[hrows, hrows]) <= 1e-12 * (maximum(abs, A.R[hrows, hrows]) + 1e-300)
    # the reservoir's column at the rotor row
    k = findfirst(==("$(_PT.name(g)).reservoir"), S.ports)
    @test abs(A.G[ip, 1] - S.G[ip, k]) <= 1e-12 * abs(S.G[ip, k])
    @test A.J == -A.J'
end

@testset "component template: the DC shunt exciter (T5)" begin
    # each shunt exciter of the full model from its two declarations against the hand-written `_exc_rhs!` (rows and
    # the limiter's decisions), at the rest point, off it, with the regulator beyond its limits and at negative flux;
    # the generated account (energy, loss, supply, port powers) against `_exc_powers`; the analytic gradient against
    # AD; derivatives through the composition
    m = phs_full_model(joinpath(ROOT, "cases", "IEEE39Bus_FullModel"))
    cs, x0 = m.cs, m.x0
    js = [j for j in eachindex(cs.exc) if cs.exc[j] isa PHShuntExciter]
    @test !isempty(js)
    rng = MersenneTwister(99)
    worst = 0.0; wacc = 0.0; wden = 0.0; ncase = 0; ndec = 0
    flag0 = _PT.EXC_FROM_TEMPLATE[]
    _PT.EXC_FROM_TEMPLATE[] = false
    try
        for j in js
            e = cs.exc[j]; o = cs.xoff[j]; xe0 = x0[o:o+2]; sig = _PT.ShuntExciterSignal(e)
            states = Any[(xe0, 1.0, 1.0, 0.8)]
            for _ in 1:8
                push!(states, (xe0 .* (1 .+ 0.1 .* randn(rng, 3)), 1 + 0.02 * randn(rng), 1 + 0.05 * randn(rng), 0.8 + 0.2 * randn(rng)))
            end
            push!(states, ([xe0[1], e.VRMAX * 1.3, xe0[3]], 1.0, 0.5, 0.8), ([xe0[1], e.VRMIN - 0.5, xe0[3]], 1.0, 1.5, 0.8),
                  ([-0.5 * xe0[1], xe0[2], xe0[3]], 0.98, 1.0, -0.1))
            for (xe, om, vt, ifd) in states
                r1 = _PT.ModeLog(); r2 = _PT.ModeLog()
                dh = _PT._exc_rhs!(zeros(3), e, xe, vt, cs.Vref[j], om, ifd, r1)
                dt = _PT.tpl_unit_rows!(zeros(3), sig, xe, vt, cs.Vref[j], om, ifd, r2)
                @test r1.decisions == r2.decisions
                worst = max(worst, maximum(abs.(dt .- dh)) / (maximum(abs, dh) + 1))
                ndec += length(r1.decisions); ncase += 1
                # the generated account and port powers against the hand-written powers
                c = _PT.ShuntExciterTemplate(e, xe[2])
                q = _PT._exc_powers(e, xe[1], xe[2], om, ifd)
                a = _PT.tpl_account(c, [xe[1]], [om, ifd]); pp = _PT.tpl_port_powers(c, [xe[1]], [om, ifd])
                sc = q.tport + q.loss + 1e-300
                wacc = max(wacc, abs(a.energy - _PT._exc_energy(e, xe[1])) / (abs(a.energy) + 1e-300),
                           abs(a.loss - q.loss) / sc, abs(a.supply - q.amp) / sc,
                           abs(pp["rotor"] - q.shaft) / sc, abs(pp["field"] + q.field) / sc, abs(pp["amplifier"] - q.amp) / sc,
                           abs(a.port_power - (q.shaft - q.field)) / sc)
                @test abs(_PT.tpl_gradient(c, [xe[1]])[1] - _PT.tpl_gradient_ad(c, [xe[1]])[1]) <= 1e-12 * (abs(q.i_f) + 1e-300)
                rd = _PT.tpl_rows(c, [xe[1]], [om, ifd])
                wden = max(wden, abs(rd.dx[1] - dt[1]) / (maximum(abs, dh) + 1),
                           abs(rd.flows[1] + q.shaft / om) / (abs(q.shaft / om) + 1e-300), abs(rd.flows[2] - q.vf) / (abs(q.vf) + 1e-300))
            end
            jt = ForwardDiff.jacobian(z -> _PT.tpl_unit_rows!(zeros(eltype(z), 3), sig, z[1:3], 1.0, cs.Vref[j], z[4], 0.8, _PT.NoModes()), vcat(xe0, 1.0))
            jh = ForwardDiff.jacobian(z -> _PT._exc_rhs!(zeros(eltype(z), 3), e, z[1:3], 1.0, cs.Vref[j], z[4], 0.8, _PT.NoModes()), vcat(xe0, 1.0))
            @test maximum(abs, jt .- jh) <= 1e-12 * maximum(abs, jh)
        end
    finally
        _PT.EXC_FROM_TEMPLATE[] = flag0
    end
    @info "template: DC shunt exciter" nexc = length(js) ncase ndec worst wacc wden
    @test worst <= 1e-13
    @test wden <= 1e-14           # the dense block against the entry path; its flows (rotor torque, field voltage)
    @test wacc <= 1e-13
end

@testset "component template: the converter's physical DC side (T5)" begin
    # the DC rows from the declaration against the hand-written `_dc_rows!`: the droop case with the battery's RC
    # branches (held modulation; averaged and switching bridges over windows of a step) and the GFL (continuous
    # modulation), at the rest point and with the link charged and discharged, the branches perturbed and random
    # currents; the generated balance (supply - loss + port power) and the storage's rate against `converter_dc_powers`
    # and `_dc_energy_rate`; derivatives through the composition
    Ts = 5e-5; h = 0.5e-6
    rng = MersenneTwister(101)
    worst = 0.0; wbal = 0.0; wden = 0.0; wsw = 0.0; ncase = 0; nsw = 0
    flag0 = _PT.DC_FROM_TEMPLATE[]
    _PT.DC_FROM_TEMPLATE[] = false
    try
        for (dir, mode, bridge) in (("IEEE39Bus_PF_gfm-droop-pf", :held, :averaged), ("IEEE39Bus_PF_gfm-droop-pf", :held, :switching),
                                    ("IEEE39Bus_PF_gfl-pf", :continuous, :averaged))
            dcs = (; H = 5e-3, battery = :sossan, soc = "40-50", mode, bridge)
            scc = load_scenario(joinpath(ROOT, "cases", dir, "ph_fault_bus16.json"))
            eqc = solve_equilibrium(load_case(scc.system_path), scc)
            kw = dir == "IEEE39Bus_PF_gfl-pf" ? (; gfl_advance = 1.0) : (; droop_dc_reject = true)
            cs, x0, u, _ = phs_controlled(eqc; loads = :series_all, converter_Ts = Ts, converter_dc = dcs, kw...)
            wb = cs.s.omega_b
            if bridge === :switching
                _PT._bridge_start!(cs, x0, u, 0.0)
                _PT._bridge_window!(cs, 0.3Ts, 0.3Ts + h)
            end
            for i in _PT._dc_converters(cs)
                g = cs.conv[i]; xc0 = x0[cs.coff[i]:cs.coff[i]+_PT.nstates(g)-1]; j = _PT._dcidx(g)
                nb = _PT.nbranches(g.dc)
                for w in 0:6
                    xc = copy(xc0)
                    if w > 0
                        xc[j] *= w <= 2 ? (0.9, 1.1)[w] : 1 + 0.05 * randn(rng)
                        xc[j+1:j+nb] .*= 1 .+ 0.2 .* randn(rng, nb)
                    end
                    ic = (0.5 + 0.5 * rand(rng)) * cis(2pi * rand(rng))
                    dh = _PT._dc_rows!(zeros(length(xc)), g, xc, ic, wb)
                    dt = _PT.tpl_dc_rows!(zeros(length(xc)), g, xc, ic, wb)
                    rg = j:j+nb
                    # the scale of the rows' terms: the Norton current through the largest omega_b / cap (the rows are its
                    # difference with y / Rb, which cancels near rest)
                    tsc = maximum(wb ./ _PT._dc_caps(g.dc)) * abs(g.dc.Eb) / g.dc.Rb
                    worst = max(worst, maximum(abs.(dt[rg] .- dh[rg])) / tsc)
                    ncase += 1
                    # the complete declaration (with the switching bridge's Rac): its dense rows against the entry path,
                    # and (review 99) the switching port's flow and its losses each against the bridge's own terms
                    c = _PT.dc_side_template(g, xc, ic, wb)
                    rd = _PT.tpl_rows(c, xc[rg], [real(ic), imag(ic)])
                    wden = max(wden, maximum(abs.(rd.dx .- dt[rg])) / tsc)
                    if _PT._sw_on(g)
                        r = _PT._sw_terms(g, xc, ic)
                        ux = xc[j] * r.m + r.udrop
                        e = vcat(rd.z, real(ic), imag(ic))
                        psc = abs(ux) * abs(ic) + abs(r.pcond) + abs(r.psw) + 1e-300
                        wsw = max(wsw, abs(complex(rd.flows[1], rd.flows[2]) - ux) / (abs(ux) + 1e-300),
                                  abs(dot([real(ic), imag(ic)], c.Rac * [real(ic), imag(ic)]) - r.pcond) / psc,
                                  abs(c.Gsw * xc[j]^2 - r.psw) / psc)
                        nsw += 1
                        continue
                    end
                    a = _PT.tpl_account(c, xc[rg], [real(ic), imag(ic)])
                    q = _PT.converter_dc_powers(g, xc, ic, wb)
                    sc = q.tport + q.loss
                    wbal = max(wbal, abs((a.supply - a.loss + a.port_power) - (q.port - q.loss)) / sc,
                               abs(a.energy - q.energy) / q.energy,
                               abs(_PT.tpl_rate(c, xc[rg], dh[rg]) - _PT._dc_energy_rate(g, xc, dh, wb)) / sc)
                end
                if !_PT._sw_on(g)
                    jt = ForwardDiff.jacobian(z -> _PT.tpl_dc_rows!(zeros(eltype(z), length(xc0)), g, z, 0.7 + 0.2im, wb), xc0)
                    jh = ForwardDiff.jacobian(z -> _PT._dc_rows!(zeros(eltype(z), length(xc0)), g, z, 0.7 + 0.2im, wb), xc0)
                    @test maximum(abs, jt .- jh) <= 1e-12 * maximum(abs, jh)
                end
            end
        end
    finally
        _PT.DC_FROM_TEMPLATE[] = flag0
    end
    @info "template: converter DC side" ncase nsw worst wbal wden wsw
    @test ncase >= 21 && nsw >= 1
    @test wden <= 1e-15           # the dense block against the entry path
    @test wsw <= 1e-12            # the switching port: m v + udrop, i' Rac i = p_cond, Gsw v^2 = p_sw
    @test worst <= 1e-15          # relative to the terms (round-off: the Norton form against (Eb - y) / Rb)
    @test wbal <= 1e-13
end

@testset "component template: the composite load's induction motor (T5)" begin
    # motors A (compressor torque law), B and C (fans and pumps) on the system base, both frames: the rates from the
    # declaration (the entry path and the dense reference) against the hand-written `motor_rates`, at the steady state,
    # at random fluxes and speeds (stalled, reversed, over speed); the generated account against the header's balance
    # (Re(v conj i_s) - the copper losses - omega T_L); derivatives through the entry path
    wb = 2pi * 60; rng = MersenneTwister(103)
    worst = 0.0; wref = 0.0; wacc = 0.0; ncase = 0
    flag0 = _PT.MOTOR_FROM_TEMPLATE[]
    _PT.MOTOR_FROM_TEMPLATE[] = false
    try
        for X in (:A, :B, :C), stat in (false, true)
            c0 = _PT.motor_circuit(_PT.WECC_MOTORS[X], 0.3; omega_b = wb)
            st = _PT.motor_steady(c0, 1.0 + 0im, 0.02; omega_b = wb)
            c = _PT.with_torque(c0, 0.8 * abs(st.Te))
            t = _PT.MotorTemplate(c, wb; stationary = stat)
            pts = Any[(st.ls, st.l1, st.l2, st.p, 1.0 + 0im)]
            for om in (0.0, -0.3, 1.2, 0.5, 0.97)
                push!(pts, (st.ls * cis(randn(rng)), st.l1 * (1 + 0.2 * randn(rng, ComplexF64)), st.l2 * (1 + 0.2 * randn(rng, ComplexF64)),
                            2 * c.H * om, (0.9 + 0.2 * rand(rng)) * cis(randn(rng))))
            end
            for (ls, l1, l2, p, v) in pts
                h = _PT.motor_rates(c, ls, l1, l2, p, v; omega_b = wb, stationary = stat)
                g = _PT.tpl_motor_rates(t, ls, l1, l2, p, v)
                hv = [real(h[1]), imag(h[1]), real(h[2]), imag(h[2]), real(h[3]), imag(h[3]), h[4]]
                gv = [real(g[1]), imag(g[1]), real(g[2]), imag(g[2]), real(g[3]), imag(g[3]), g[4]]
                # the scale of the terms: the frame's rotation omega_b |lambda| and the torque
                sc = wb * maximum(abs, (ls, l1, l2)) + abs(h[5].Te) + abs(h[5].TL) + abs(v)
                worst = max(worst, maximum(abs.(gv .- hv)) / sc)
                x = [real(ls), imag(ls), real(l1), imag(l1), real(l2), imag(l2), p]
                r = _PT.tpl_rows(t, x, [real(v), imag(v)])
                wref = max(wref, maximum(abs.(r.dx .- gv)) / sc, maximum(abs.(r.flows .+ [real(h[5].is), imag(h[5].is)])) / (abs(h[5].is) + 1))
                a = _PT.tpl_account(t, x, [real(v), imag(v)])
                loss = c.Rs * abs2(h[5].is) + c.R1 * abs2(h[5].i1) + c.R2 * abs2(h[5].i2) + h[5].omega * h[5].TL
                pin = real(v * conj(h[5].is))
                asc = abs(pin) + loss + 1e-12
                wacc = max(wacc, abs(a.loss - loss) / asc, abs(a.port_power - pin) / asc, abs(a.supply) / asc,
                           abs(a.energy - _PT.motor_energy(c, ls, l1, l2, p, wb)) / a.energy,
                           abs(_PT.tpl_rate(t, x, hv) - (pin - loss)) / (asc + wb * abs(a.energy)))
                ncase += 1
            end
            # derivatives through the entry path against the hand-written rates
            fh(z) = (q = _PT.motor_rates(c, complex(z[1], z[2]), complex(z[3], z[4]), complex(z[5], z[6]), z[7], 1.0 + 0im; omega_b = wb, stationary = stat);
                     [real(q[1]), imag(q[1]), real(q[2]), imag(q[2]), real(q[3]), imag(q[3]), q[4]])
            ft(z) = (q = _PT.tpl_motor_rates(t, complex(z[1], z[2]), complex(z[3], z[4]), complex(z[5], z[6]), z[7], 1.0 + 0im);
                     [real(q[1]), imag(q[1]), real(q[2]), imag(q[2]), real(q[3]), imag(q[3]), q[4]])
            z0 = [real(st.ls), imag(st.ls), real(st.l1), imag(st.l1), real(st.l2), imag(st.l2), st.p]
            jh = ForwardDiff.jacobian(fh, z0); jt = ForwardDiff.jacobian(ft, z0)
            @test maximum(abs, jt .- jh) <= 1e-12 * maximum(abs, jh)
        end
    finally
        _PT.MOTOR_FROM_TEMPLATE[] = flag0
    end
    @info "template: induction motor" ncase worst wref wacc
    @test ncase == 36
    @test worst <= 1e-14
    @test wref <= 1e-14
    @test wacc <= 1e-12
end

@testset "component template: the cores' loop rows at the step's efforts (T5)" begin
    # the droop case with cores: the simulator's loop rows from the template (flag on) against the hand-written rows
    # (flag off) on the whole vector field, and the entry path against the dense block evaluated at the given efforts
    # (tpl_rows_at), at the rest point, in saturation and at other clock angles
    Ts = 5e-5
    dcs = (; H = 5e-3, battery = :sossan, soc = "40-50", mode = :held)
    cores = Dict(33 => 800.0, 36 => 700.0, 37 => 700.0, 38 => 1000.0)
    scc = load_scenario(joinpath(ROOT, "cases", "IEEE39Bus_PF_gfm-droop-pf", "ph_fault_bus16.json"))
    eqc = solve_equilibrium(load_case(scc.system_path), scc)
    cs, x0, u, _ = phs_controlled(eqc; loads = :series_all, converter_Ts = Ts, droop_dc_reject = true,
                                  converter_dc = merge(dcs, (; bridge = :averaged)), cores)
    s = cs.s; pn = s.net
    rng = MersenneTwister(34)
    wfull = 0.0; wref = 0.0; wfr = 0.0; n = 0
    flag0 = _PT.CORE_FROM_TEMPLATE[]
    try
        for w in 0:6
            x = copy(x0)
            if w > 0
                for q in s.cores
                    fm = w <= 3 ? 1.0 + 0.4 * w : 1.0
                    x[s.bidx[q.pos]:s.bidx[q.pos]+1] .*= fm; x[q.xs:q.xs+1] .*= fm
                end
                w > 3 && (x[s.iclk] += 2pi * rand(rng))
            end
            _PT.CORE_FROM_TEMPLATE[] = false; fh = phs_rhs(cs, x, u)
            _PT.CORE_FROM_TEMPLATE[] = true;  ft = phs_rhs(cs, x, u)
            wfull = max(wfull, maximum(abs.(ft .- fh) ./ (abs.(fh) .+ 1e-6 * maximum(abs, fh))))
            fl = _PT.phs_flows(s, x, phs_control_inputs(cs, x, u.gf; cf = u.cf))
            for (k, q) in enumerate(s.cores)
                l = s.branches[q.pos]
                c = _PT.CoreTemplate(q, s.omega_b, complex(pn.a[l]))
                xH, xS = _PT._cx(x, s.bidx[q.pos]), _PT._cx(x, q.xs)
                iH, iS = _PT._core_currents(s, k, x)
                vf, vt = fl.v[pn.from[l]], fl.v[pn.to[l]]
                dH, dS = _PT.tpl_core_loops(c, xH, xS, iH, iS, vf, vt)
                z = [real(iH), imag(iH), -real(iS), -imag(iS), _PT.core_rotation_rate(iH, iS, xH, xS)]
                r = _PT.tpl_rows_at(c, [real(xH), imag(xH), real(xS), imag(xS), x[s.iclk]], z, [real(vf), imag(vf), real(vt), imag(vt)])
                sc = s.omega_b * max(abs(xH), abs(xS)) + abs(vf) + abs(vt)
                wref = max(wref, maximum(abs.(r.dx[1:4] .- [real(dH), imag(dH), real(dS), imag(dS)])) / sc)
                # the frame port's power at the step's efforts with the clock effort (review 99): zero
                G5 = _PT.tpl_block(c, [real(xH), imag(xH), real(xS), imag(xS), x[s.iclk]], zeros(4)).G[1:5, 1]
                wfr = max(wfr, abs(dot(z, G5)) / (sum(abs.(z .* G5)) + 1e-300))
                n += 1
            end
        end
    finally
        _PT.CORE_FROM_TEMPLATE[] = flag0
    end
    @info "template: core loops (T5)" n wfull wref wfr
    @test n >= 28
    @test wfr <= 1e-12
    @test wfull <= 1e-14
    @test wref <= 1e-15
end

@testset "component template: the synchronous machine (T5)" begin
    # the full model's machines (and the base case with :series_all loads, whose reduced nodes carry machines), both
    # frames: the whole vector field with the template's rows (flag on) against the hand-written rows (flag off); per
    # machine the dense block's rows against the entry path, its port flow against the bus's received current
    # -inet / conj(a) (no cut), its gradient against phs_gradient, its rotation and port couplings skew; random states
    rng = MersenneTwister(107)
    flag0 = _PT.MACHINE_FROM_TEMPLATE[]
    wfull = 0.0; wden = 0.0; wport = 0.0; wgrad = 0.0; ncut = 0; n = 0
    sc = load_scenario(joinpath(ROOT, "cases", "IEEE39Bus_PF", "no_fault_phtrue.json"))
    eqb = solve_equilibrium(load_case(sc.system_path), sc)
    sb, xb, ub, _ = phs_equilibrium(eqb; loads = :series_all)
    fm = phs_full_model(joinpath(ROOT, "cases", "IEEE39Bus_FullModel"))
    models = Any[(fm.cs, fm.x0, fm.u, false), (sb, xb, ub, false)]
    sst = phs_frame(sb, :stationary; theta0 = 0.3)
    push!(models, (sst, _PT.phs_rotate_network(sb, xb, 0.3), PHInput{Float64}(ub.Tm, ub.vf, ub.gf, ub.const_power, ub.uc, [cis(0.7)]), true))
    try
        for (mdl, x0, u, stat) in models
            s = mdl isa PHSystem ? mdl : mdl.s
            for w in 0:4
                x = copy(x0)
                w > 0 && (x .+= 1e-2 .* randn(rng, length(x)) .* abs.(x))
                _PT.MACHINE_FROM_TEMPLATE[] = false; fh = phs_rhs(mdl, x, u)
                _PT.MACHINE_FROM_TEMPLATE[] = true;  ft = phs_rhs(mdl, x, u)
                # relative to the rows' terms: omega_b omega on the angles (at rest the vector field itself is ~1e-13)
                wfull = max(wfull, maximum(abs.(ft .- fh)) / (maximum(abs, fh) + s.omega_b))
                up = mdl isa PHSystem ? u : phs_control_inputs(mdl, x, u.gf; cf = u.cf)
                fl = _PT.phs_flows(s, x, up)
                zs = _PT.phs_gradient(s, view(x, 1:s.n))
                for (j, m) in enumerate(s.machines)
                    cut = m.cut == 0 ? nothing : s.cuts[m.cut]
                    rows = collect(m.offset:m.offset+_PT.phs_nstates(m)-1)
                    cut === nothing || append!(rows, (m.tidx, m.tidx + 1))
                    cut === nothing || (ncut += 1)
                    t = _PT.MachineTemplate(m, cut, stat ? up.cf[1] : 1.0 + 0im; const_power = up.const_power)
                    vb = m.cut == 0 ? fl.v[m.node] : fl.v[cut.far[1]]
                    r = _PT.tpl_rows(t, x[rows], [real(vb), imag(vb)], up.vf[j], up.Tm[j])
                    dfast = zeros(length(x)); _PT.tpl_machine_rows!(dfast, s, m, x, fl.mc[j], vb, up.vf[j], up.Tm[j], up.const_power)
                    sc_r = maximum(abs, r.blk.J * vcat(r.z, real(vb), imag(vb))) + maximum(abs, r.blk.R * vcat(r.z, real(vb), imag(vb))) + 1
                    wden = max(wden, maximum(abs.(r.dx .- dfast[rows])) / sc_r)
                    wgrad = max(wgrad, maximum(abs.(r.z .- zs[rows])) / maximum(abs, r.z))
                    if cut === nothing
                        inj = -fl.mc[j].inet / conj(m.a)
                        wport = max(wport, abs(complex(r.flows[1], r.flows[2]) - inj) / (abs(inj) + 1e-12))
                    end
                    @test r.blk.J == -r.blk.J'
                    @test r.blk.R == r.blk.R' && minimum(eigvals(Symmetric(r.blk.R))) >= -1e-14
                    n += 1
                end
            end
        end
    finally
        _PT.MACHINE_FROM_TEMPLATE[] = flag0
    end
    @info "template: synchronous machine" n ncut wfull wden wport wgrad
    @test n >= 100 && ncut >= 1
    @test wfull <= 1e-15
    @test wden <= 1e-15
    @test wport <= 1e-14
    @test wgrad <= 1e-14
end

@testset "component template: the network's nodes, branches, shunts and reduced nodes (T5)" begin
    # the base case with :series_all loads (reduced nodes) in both frames and the full model: the whole vector field
    # with the template's network rows against the hand-written rows; per component the dense block's rows (at the
    # system's gradient and port voltages) against the row functions, and the gradient against phs_gradient
    rng = MersenneTwister(109)
    flag0 = _PT.NETWORK_FROM_TEMPLATE[]
    wfull = 0.0; wden = 0.0; wgrad = 0.0; cnt = Dict(:node => 0, :branch => 0, :shunt => 0, :cut => 0)
    sc = load_scenario(joinpath(ROOT, "cases", "IEEE39Bus_PF", "no_fault_phtrue.json"))
    eqb = solve_equilibrium(load_case(sc.system_path), sc)
    sb, xb, ub, _ = phs_equilibrium(eqb; loads = :series_all)
    fm = phs_full_model(joinpath(ROOT, "cases", "IEEE39Bus_FullModel"))
    sst = phs_frame(sb, :stationary; theta0 = 0.3)
    models = Any[(sb, xb, ub), (sst, _PT.phs_rotate_network(sb, xb, 0.3), PHInput{Float64}(ub.Tm, ub.vf, ub.gf, ub.const_power, ub.uc, [cis(0.7)])),
                 (fm.cs, fm.x0, fm.u)]
    try
        for (mdl, x0, u) in models
            s = mdl isa PHSystem ? mdl : mdl.s; pn = s.net; wb = s.omega_b; st = s.stationary
            for w in 0:3
                x = copy(x0)
                w > 0 && (x .+= 1e-2 .* randn(rng, length(x)) .* abs.(x))
                _PT.NETWORK_FROM_TEMPLATE[] = false; fh = phs_rhs(mdl, x, u)
                _PT.NETWORK_FROM_TEMPLATE[] = true;  ft = phs_rhs(mdl, x, u)
                wfull = max(wfull, maximum(abs.(ft .- fh)) / (maximum(abs, fh) + wb))
                up = mdl isa PHSystem ? u : phs_control_inputs(mdl, x, u.gf; cf = u.cf)
                fl = _PT.phs_flows(s, x, up); v = fl.v
                zs = _PT.phs_gradient(s, view(x, 1:s.n))
                pr(c) = [real(c), imag(c)]
                for k in s.dyn
                    g = pn.G[k] + s.twG[k] + up.gf[k] + _PT._gstat(s, fl, k)
                    t = _PT.NodeTemplate(pn.C[k], g, wb; stationary = st)
                    r = s.qidx[k]; xl = x[r:r+1]
                    b = _PT.tpl_block(t, xl, Float64[])
                    dd = (b.J .- b.R) * _PT.tpl_gradient(t, xl)
                    ref = _PT.tpl_node_row(pn.C[k], g, st, v[k], zero(v[k]))
                    wden = max(wden, maximum(abs.(dd .- pr(ref))) / (abs(v[k]) * (pn.C[k] + g) + 1e-12))
                    wgrad = max(wgrad, maximum(abs.(_PT.tpl_gradient(t, xl) .- zs[r:r+1])) / (abs(v[k]) + 1e-12))
                    cnt[:node] += 1
                end
                for (j, l) in enumerate(s.branches)
                    s.bcut[j] == 0 || continue
                    any(q -> q.pos == j, s.cores) && continue
                    t = _PT.BranchTemplate(pn.R[l], pn.L[l], pn.a[l], wb; stationary = st)
                    r = s.bidx[j]; xl = x[r:r+1]
                    vf, vt = v[pn.from[l]], v[pn.to[l]]
                    rr = _PT.tpl_rows(t, xl, vcat(pr(vf), pr(vt)))
                    ref = _PT.tpl_branch_row(pn.R[l], pn.L[l], pn.a[l], st, fl.ib[j], vf, vt)
                    wden = max(wden, maximum(abs.(rr.dx .- pr(ref))) / (abs(vf) + abs(vt) + abs(fl.ib[j]) * (pn.R[l] + pn.L[l])))
                    # the flows into its nodes: -conj(1/a) i at the from end, +i at the to end
                    wden = max(wden, maximum(abs.(rr.flows .- vcat(pr(-fl.ib[j] / conj(pn.a[l])), pr(fl.ib[j])))) / (abs(fl.ib[j]) + 1e-12))
                    wgrad = max(wgrad, maximum(abs.(rr.z .- zs[r:r+1])) / (abs(fl.ib[j]) + 1e-12))
                    cnt[:branch] += 1
                end
                for (j, k) in enumerate(pn.sh_bus)
                    s.sidx[j] > 0 || continue
                    t = _PT.ShuntTemplate(s.Rsh[j], pn.Lsh[j], wb; stationary = st)
                    r = s.sidx[j]; xl = x[r:r+1]
                    rr = _PT.tpl_rows(t, xl, pr(v[k]))
                    ref = _PT.tpl_shunt_row(s.Rsh[j], pn.Lsh[j], st, fl.ish[j], v[k])
                    wden = max(wden, maximum(abs.(rr.dx .- pr(ref))) / (abs(v[k]) + abs(fl.ish[j]) * (s.Rsh[j] + pn.Lsh[j])))
                    cnt[:shunt] += 1
                end
                for q in s.cuts
                    q.machine == 0 || continue
                    L = [pn.L[s.branches[j]] for j in q.pos]; Rb = [pn.R[s.branches[j]] for j in q.pos]
                    t = _PT.CutTemplate(L, Rb, q.LL, q.RL, q.gamma, q.c, wb; stationary = st)
                    xl = vcat([x[s.bidx[j]:s.bidx[j]+1] for j in q.pos]...)
                    rr = _PT.tpl_rows(t, xl, vcat([pr(v[q.far[a]]) for a in eachindex(q.pos)]...))
                    ref = vcat([pr(_PT.tpl_cut_row(q, a, q.pos, fl.ib, v, st)) for a in eachindex(q.pos)]...)
                    wden = max(wden, maximum(abs.(rr.dx .- ref)) / (maximum(abs, ref) + maximum(abs, xl) * wb))
                    cnt[:cut] += 1
                end
            end
        end
    finally
        _PT.NETWORK_FROM_TEMPLATE[] = flag0
    end
    @info "template: network" cnt wfull wden wgrad
    @test all(>(0), values(cnt))
    @test wfull <= 1e-15
    @test wden <= 1e-14
    @test wgrad <= 1e-14
end

@testset "component template: the ledger entries of the shunt exciters and the hydro unit (T5)" begin
    # the full model's controlled ledger with the shunt exciters' and the hydro unit's entries from the template
    # against the hand-written entries: energy, loss, port, rate (own rows and along a given dx), tport, the supply
    fm = phs_full_model(joinpath(ROOT, "cases", "IEEE39Bus_FullModel"))
    cs, x0, u = fm.cs, fm.x0, fm.u
    rng = MersenneTwister(113)
    flag0 = _PT.LEDGER_FROM_TEMPLATE[]
    worst = 0.0; n = 0
    try
        for w in 0:4
            x = copy(x0)
            w > 0 && (x .+= 1e-3 .* randn(rng, length(x)) .* abs.(x))
            for dx in (nothing, phs_rhs(cs, x, u) .* (1 .+ 0.1 .* randn(rng, length(x))))
                _PT.LEDGER_FROM_TEMPLATE[] = false; Lh = phs_ledger(cs, x, u; dx)
                _PT.LEDGER_FROM_TEMPLATE[] = true;  Lt = phs_ledger(cs, x, u; dx)
                @test Lh.names == Lt.names
                for f in (:energy, :loss, :port, :rate, :tport)
                    a, b = getfield(Lh, f), getfield(Lt, f)
                    worst = max(worst, maximum(abs.(a .- b) ./ (abs.(Lh.tport) .+ abs.(Lh.energy) .+ 1e-12)))
                end
                worst = max(worst, abs(Lh.supply - Lt.supply) / (sum(abs, Lh.tport) + 1e-12))
                n += 1
            end
        end
    finally
        _PT.LEDGER_FROM_TEMPLATE[] = flag0
    end
    @info "template: ledger entries" n worst
    @test n == 10
    @test worst <= 1e-13
end

@testset "component template: the whole model by the generic assembler (T1)" begin
    # every component placed as data and assembled generically, the algebraic voltages eliminated: J, R, G v and the
    # gradient against phs_structure's hand-assembled form; the base case with :series_all loads (reduced nodes, machines
    # at reduced nodes) in both frames, the composite loads (54 motors), the droop converters with cores (EMFs as inputs,
    # the shared clock), at the rest point and perturbed states
    rng = MersenneTwister(127)
    sc = load_scenario(joinpath(ROOT, "cases", "IEEE39Bus_PF", "no_fault_phtrue.json"))
    eqb = solve_equilibrium(load_case(sc.system_path), sc)
    sb, xb, ub, _ = phs_equilibrium(eqb; loads = :series_all)
    sst = phs_frame(sb, :stationary; theta0 = 0.3)
    sm, xm, um, _ = phs_equilibrium(eqb; loads = :composite)
    # the droop converters with saturating cores (the cores' shared frame clock), as a PHSystem with the EMFs as inputs
    dcs = (; H = 5e-3, battery = :sossan, soc = "40-50", mode = :held)
    cores = Dict(33 => 800.0, 36 => 700.0, 37 => 700.0, 38 => 1000.0)
    scc = load_scenario(joinpath(ROOT, "cases", "IEEE39Bus_PF_gfm-droop-pf", "ph_fault_bus16.json"))
    eqc = solve_equilibrium(load_case(scc.system_path), scc)
    csd, xd, ud, _ = phs_controlled(eqc; loads = :series_all, converter_Ts = 5e-5, droop_dc_reject = true, cores)
    upd = phs_control_inputs(csd, xd, ud.gf; cf = ud.cf)
    worst = (J = 0.0, R = 0.0, Gv = 0.0, z = 0.0); n = 0; wframe = 0.0
    for (s, x0, u) in ((csd.s, xd[1:csd.s.n], upd), (sb, xb, ub), (sst, _PT.phs_rotate_network(sb, xb, 0.3), PHInput{Float64}(ub.Tm, ub.vf, ub.gf, ub.const_power, ub.uc, [cis(0.7)])),
                       (sm, xm, um))
        for w in 0:2
            x = copy(x0)
            w > 0 && (x .+= 1e-2 .* randn(rng, length(x)) .* abs.(x))
            S = phs_structure(s, x, u)
            A = _PT.tpl_assemble_system(s, x, u)
            rel(a, b) = maximum(abs.(a .- b)) / (maximum(abs, b) + 1e-300)
            worst = (J = max(worst.J, rel(A.J, S.J)), R = max(worst.R, rel(A.R, S.R)), Gv = max(worst.Gv, rel(A.Gv, S.G * S.v)),
                     z = max(worst.z, rel(A.z, S.z)))         # the whole gradient, the shared clock's included (review 101)
            @test A.J ≈ -A.J' atol = 1e-12
            # the frame port's power is zero: the cores' loop terms against the clock's summed rotation rate
            fc = findall(==("frame"), A.sources)
            pf = sum(A.v[k] * dot(A.G[:, k], A.z) for k in fc); sf = sum(abs(A.v[k]) * sum(abs.(A.G[:, k] .* A.z)) for k in fc)
            wframe = max(wframe, abs(pf) / (sf + 1e-300))
            # the machine through the generic three-argument interface (review 101): its inputs in the declaration
            m1 = s.machines[1]; cut1 = m1.cut == 0 ? nothing : s.cuts[m1.cut]
            t1 = _PT.MachineTemplate(m1, cut1, s.stationary ? u.cf[1] : 1.0 + 0im; const_power = u.const_power, vf = u.vf[1], Tm = u.Tm[1])
            rows1 = vcat(collect(m1.offset:m1.offset+_PT.phs_nstates(m1)-1), cut1 === nothing ? Int[] : [m1.tidx, m1.tidx + 1])
            e1 = randn(rng, 2)
            @test _PT.tpl_rows(t1, x[rows1], e1).dx ≈ _PT.tpl_rows(t1, x[rows1], e1, u.vf[1], u.Tm[1]).dx rtol = 1e-15
            @test isfinite(_PT.tpl_account(t1, x[rows1], e1).loss)
            n += 1
        end
    end
    # repeated ports (review 101): a reduced node with both far ports at one bus against the same block assembled with
    # distinct ports and those merged (P' A P), J, R and the source columns
    ct = _PT.CutTemplate([0.1, 0.2], [0.01, 0.02], 0.3, 0.03, [1.0, -1.0], [1.0, 0.5], 377.0)
    xc = randn(rng, 4)
    Ar = _PT.tpl_assemble([_PT.TplPlacement(ct, [1, 2, 3, 4], [5, 6, 5, 6])], 6, vcat(xc, 0, 0), zeros(6))
    Ad = _PT.tpl_assemble([_PT.TplPlacement(ct, [1, 2, 3, 4], [5, 6, 7, 8])], 8, vcat(xc, 0, 0, 0, 0), zeros(8))
    Pm = zeros(8, 6); for k in 1:4; Pm[k, k] = 1; end; Pm[5, 5] = Pm[7, 5] = 1; Pm[6, 6] = Pm[8, 6] = 1
    @test Ar.J ≈ Pm' * Ad.J * Pm atol = 1e-15
    @test Ar.R ≈ Pm' * Ad.R * Pm atol = 1e-15
    @test any(!iszero, Ar.J[1:4, 5:6])
    @info "template: whole-model assembler" n worst wframe
    @test wframe <= 1e-12
    @test n == 12
    @test worst.J <= 1e-12 && worst.R <= 1e-12 && worst.Gv <= 1e-12 && worst.z <= 1e-12
end
