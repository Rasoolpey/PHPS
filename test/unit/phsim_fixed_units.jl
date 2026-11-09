# Fixed turbine hardware across operating points (ROA_ROUTES_TODO.md Y5 step 1; review 19, 2026-10-21):
# phs_controlled(...; fixed_units) with phs_fixed_units, and steam_initialise(...; rebuild = false). A model built at
# another operating point with the units of a first build keeps their hardware (vessel volumes, valve characteristic,
# hydro water time) and solves only the operating state; built without them it is a different plant.

@testset "fixed turbine hardware across operating points" begin
    sc = load_scenario(joinpath(ROOT, "cases", "IEEE39Bus_PF", "no_fault_phtrue.json"))
    case = load_case(sc.system_path)
    eq = solve_equilibrium(case, sc)
    paths = Dict("IEEEG1_2" => :A, "IEEEG1_5" => :A, "IEEEG1_6" => :A, "IEEEG1_8" => :A, "IEEEG1_9" => :A,
                 "IEEEG1_3" => :C, "IEEEG1_4" => :C, "IEEEG1_7" => :C, "IEEEG1_10" => :C)
    steam = (; paths, r_v = 0.95, rh_dT = 100.0, tau_n = 0.03, lowflow = 1e-3, cold_reheat = 26.2,
             volumes = :fit, valve = :characteristic, tables = steam_tables())
    hydro = (; Tw = :match)
    c1, x1, u1, _ = phs_controlled(eq; steam, hydro)
    fu = phs_fixed_units(c1)
    @test length(fu.steam) == 8                                            # units 5 and 6 share a machine
    @test haskey(fu, :hydro_Tw) && fu.hydro_Tw > 0
    hw(g) = (copy(g.V), copy(g.vc_d), copy(g.vc_y), copy(g.vc_s), copy(g.Tw))
    hw0 = Dict(k => hw(g) for (k, g) in fu.steam)

    # the same operating point with the fixed units reproduces the first build
    c2, x2, u2, _ = phs_controlled(eq; steam, hydro, fixed_units = fu)
    @test x2 ≈ x1 rtol = 1e-9
    @test maximum(abs.(phs_rhs(c2, x2, u2)) ./ phs_scales(c2, x2)) < 1e-9
    @test all(hw(fu.steam[k]) == hw0[k] for k in keys(fu.steam))           # the given units are not changed

    # another operating point: every demand and PV dispatch times 0.8
    case08 = Case(case.path, case.raw, case.description, case.config, case.buses,
                  [PHPS.LoadData(l.idx, l.bus, 0.8 * l.p0, 0.8 * l.q0) for l in case.pq],
                  [PHPS.SourceData(g.idx, g.bus, 0.8 * g.p0, g.q0, g.v0, g.a0, g.pmax, g.pmin, g.qmax, g.qmin) for g in case.pv],
                  case.slack, case.lines, case.shunts, case.components, case.connections)
    eq08 = solve_equilibrium(case08, sc)
    cf, xf, uf, _ = phs_controlled(eq08; steam, hydro, fixed_units = fu)     # fixed hardware
    cr, xr, ur, _ = phs_controlled(eq08; steam, hydro)                       # rebuilt at 0.8: another plant
    @test maximum(abs.(phs_rhs(cf, xf, uf)) ./ phs_scales(cf, xf)) < 1e-9
    gf = Dict(g.name => g for g in cf.gov if g isa PHSteamUnit)
    gr = Dict(g.name => g for g in cr.gov if g isa PHSteamUnit)
    @test all(hw(gf[k])[1:4] == hw0[k][1:4] for k in keys(fu.steam))        # volumes and characteristic kept
    @test all(hw(fu.steam[k]) == hw0[k] for k in keys(fu.steam))            # the given units (walls too) untouched
    @test any(gf[k].Tw != fu.steam[k].Tw for k in keys(fu.steam))           # the walls adapt to the new power (firing)
    @test any(gr[k].V != hw0[k][1] || gr[k].vc_d != hw0[k][2] for k in keys(fu.steam))   # the rebuild changes them
    hf = [g for g in cf.gov if g isa PHHydroUnit][1]
    hr = [g for g in cr.gov if g isa PHHydroUnit][1]
    @test hf.Tw == fu.hydro_Tw
    @test hr.Tw != fu.hydro_Tw
    # the turbine powers follow the new operating point
    @test phs_control_signals(cf, xf).Pm ≈ phs_control_signals(cr, xr).Pm rtol = 1e-8

    # a missing unit in fixed_units is refused (no silent mixture of kept and refitted hardware)
    fpart = (; steam = Dict(k => g for (k, g) in fu.steam if k != first(keys(fu.steam))), hydro_Tw = fu.hydro_Tw)
    @test_throws ArgumentError phs_controlled(eq08; steam, hydro, fixed_units = fpart)
    # the default rebuild on an already fitted unit (the start scaled by its volume ratio) still finds its steady state
    gk, gu = first(fu.steam)
    gre = deepcopy(gu)
    jg = findfirst(g -> g isa PHSteamUnit && g.name == gk, c1.gov)
    Pm1 = phs_control_signals(c1, x1).Pm[jg] / c1.s.machines[jg].units
    xre, _ = steam_initialise(gre, Pm1)
    @test PHPS._steam_power(gre, xre) ≈ Pm1 rtol = 1e-9
    # an operating point outside the kept characteristic's range is refused
    @test_throws Exception steam_initialise(deepcopy(gu), 1.2 * gu.PMAX; rebuild = false)
    # refusal: rebuild = false on a unit whose characteristic was never built
    fresh = deepcopy(first(values(fu.steam)))
    empty!(fresh.vc_d); empty!(fresh.vc_y); empty!(fresh.vc_s)
    @test_throws ArgumentError steam_initialise(fresh, 1.0; rebuild = false)
end
