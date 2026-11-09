# The hydro unit at a closed gate (src/phsim/hydro.jl header; record 36): the seat holds the static head while the water
# column is at rest (h = 1, no shaft power, the column's energy rate zero, the energy book exact); an open gate with the
# column at rest has h = 0; a closed gate with water flowing and reverse flow stay refused.

@testset "hydro unit: the closed gate with the column at rest" begin
    sc = load_scenario(joinpath(ROOT, "cases", "IEEE39Bus_PF", "no_fault_phtrue.json"))
    eq = solve_equilibrium(load_case(sc.system_path), sc)
    g3 = first(c for c in eq.sys.comps if c isa IEEEG3_PHTRUE)
    un = PHHydroUnit(g3; Tw = hydro_Tw_match(g3, 2.5))
    x, Pref = hydro_initialise(un, 2.5)
    dx = similar(x)
    # closed gate, column at rest: a rest point of the column (q' = 0), no power, the account exact
    xc = [0.0, 0.0, 0.0, 0.0]
    @test PHPS._hydro_head(un, 0.0, 0.0) == 1
    PHPS._hydro_rhs!(dx, un, xc, 1.0, Pref, PHPS.NoModes())
    @test dx[4] == 0
    @test PHPS._hydro_power(un, xc) == 0
    pw = PHPS._hydro_powers(un, xc)
    @test pw.energy == 0 && pw.shaft == 0 && pw.supply == 0 && pw.rate == 0 && pw.port == pw.rate
    # differentiable there (the continuation's Jacobians): the head is constant on the closed seat
    @test ForwardDiff.derivative(q -> PHPS._hydro_head(un, 0.0, q), 0.0) == 0
    # open gate, column at rest: no head on the runner yet, the column accelerates
    @test PHPS._hydro_head(un, 0.1 * un.GMAX, 0.0) == 0
    PHPS._hydro_rhs!(dx, un, [0.0, 0.0, 0.1 * un.GMAX, 0.0], 1.0, Pref, PHPS.NoModes())
    @test dx[4] ≈ 1 / un.Tw
    # still refused: a closed gate with water flowing, reverse flow
    @test_throws DomainError PHPS._hydro_head(un, 0.0, 0.29)
    @test_throws DomainError PHPS._hydro_head(un, 2.5, -0.01)
    # unchanged on an open gate: the steady state h = 1 at q = G
    @test PHPS._hydro_head(un, 2.5, 2.5 / un.GMAX) ≈ 1 rtol = 1e-15
end
