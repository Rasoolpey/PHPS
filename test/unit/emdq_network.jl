# Physical network (docs/records/early/PHYSICAL_NETWORK.md): structure of the descriptor form, the element
# balances and the lossless interconnection (Propositions 1 and 2), the quasi-static Y-bus
# (Proposition 3) and the node classes, on IEEE-39.

@testset "physical network" begin
    case = load_case(joinpath(ROOT, "cases", "IEEE39Bus_PF", "system_phtrue.json"))
    p = physical_network(case)
    n, m, ns = pn_sizes(p)
    @test (n, m, ns) == (39, 46, 18)

    d = pn_descriptor(p)
    @test d.J == -d.J'                               # exact, by construction
    @test isdiag(d.R) && all(>=(0), diag(d.R))
    @test isdiag(d.E) && all(>=(0), diag(d.E))

    rng = MersenneTwister(11)
    for _ in 1:5
        v = complex.(1 .+ 0.1 .* randn(rng, n), 0.3 .* randn(rng, n))
        i = complex.(randn(rng, m), randn(rng, m))
        ish = complex.(randn(rng, ns), randn(rng, ns))
        iext = complex.(randn(rng, n), randn(rng, n))
        b = pn_balance(p, v, i, ish, iext)
        @test maximum(abs, b.node_err) < 1e-12
        @test maximum(abs, b.line_err) < 1e-12
        @test maximum(abs, b.sh_err) < 1e-12
        @test abs(b.interconnection) < 1e-12
        @test abs(b.total_error) < 1e-11
        # the network is linear: shifted flows are the flows of the increments (Corollary, §4)
        v0, i0, s0, u0 = v .+ 0.1, i .- 0.2im, ish .* 0.5, iext .+ 0.3
        f, f0 = pn_flows(p, v, i, ish, iext), pn_flows(p, v0, i0, s0, u0)
        fd = pn_flows(p, v - v0, i - i0, ish - s0, iext - u0)
        @test maximum(abs, (f.fv - f0.fv) - fd.fv) < 1e-12
        @test maximum(abs, (f.fi - f0.fi) - fd.fi) < 1e-12
    end

    Y = Matrix(ybus(case; loads = true))
    @test maximum(abs, pn_quasistatic_ybus(p) - Y) < 1e-11

    cls = pn_node_classes(p)
    @test length(cls.dynamic) == 28
    @test cls.algebraic == [12, 20, 31]
    @test cls.cutset == [30, 32, 33, 34, 35, 36, 37, 38]

    # data without a passive realisation are refused
    bad = deepcopy(case)
    l = bad.lines[1]
    bad.lines[1] = PHPS.LineData(l.idx, l.bus1, l.bus2, -0.01, l.x, l.b, l.tap, l.phi)
    @test_throws ArgumentError physical_network(bad)
end
