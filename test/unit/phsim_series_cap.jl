# A series capacitor on a line (the case's `xc`, system.jl header): on IEEE-39 (PowerFactory-matched) with Line_33
# (26-29) 50 % compensated, the structure identities with the capacitor's charge pair, the equilibrium, a fault run's
# discrete energy balance (the capacitor's storage is quadratic: midpoint is exact), the subsynchronous network mode of
# the linearization, and the limit xc -> 0 (the uncompensated line).

@testset "series capacitor on a line" begin
    sc = load_scenario(joinpath(ROOT, "cases", "IEEE39Bus_PF", "no_fault_phtrue.json"))
    case0 = load_case(sc.system_path)
    # the case with Line_33's series capacitor `xc` (x stays the line's own reactance)
    function compensated(xc)
        c = deepcopy(case0)
        k = findfirst(l -> l.idx == "Line_33", c.lines)
        l = c.lines[k]
        c.lines[k] = PHPS.LineData(l.idx, l.bus1, l.bus2, l.r, l.x, l.b, l.tap, l.phi, xc)
        return c
    end
    xline = case0.lines[findfirst(l -> l.idx == "Line_33", case0.lines)].x
    eq = solve_equilibrium(compensated(0.5 * xline), sc)
    s, x0, u, rep = phs_equilibrium(eq)
    eq0 = solve_equilibrium(case0, sc)
    s0, x00, u0, _ = phs_equilibrium(eq0)
    cx = PHPS._cx

    @testset "assembly, the quasi-static network and the equilibrium" begin
        @test length(s.cpos) == 1 && s.net.branch[s.branches[s.cpos[1]]] == "Line_33"
        @test s.n == s0.n + 2 && s.cidx[1] == s0.n + 1
        @test phs_state_names(s)[s.cidx[1]:s.cidx[1]+1] == ["qc_Line_33_Re", "qc_Line_33_Im"]
        # the load flow sees x - xc: the RMS network's series admittance of the line
        Y = ybus(eq.sys.case)
        i, j = eq.sys.net.index[26], eq.sys.net.index[29]
        @test -Y[i, j] ≈ 1 / complex(0.0057, 0.5 * xline) rtol = 1e-12
        @test rep.residual < 1e-12 && rep.dV < 1e-8
        # at rest the capacitor's voltage is -j xc i (in phase quadrature with the line current)
        jl = s.cpos[1]
        il = phs_flows(s, x0, u).ib[jl]
        @test PHPS._scap_v(s, x0, 1) ≈ -im * 0.5 * xline * il rtol = 1e-9
        @test_throws ArgumentError phs_equilibrium(eq; lines = :travelling)
    end

    @testset "structure identities with the capacitor" begin
        rng = MersenneTwister(7)
        scl = phs_scales(s, x0)
        for trial in 1:3
            y = x0 .+ 0.2 .* scl .* randn(rng, s.n)
            uu = PHInput(u.Tm .* (1 .+ 0.3 .* randn(rng, length(u.Tm))), u.vf .* (1 .+ 0.3 .* randn(rng, length(u.vf))),
                         zeros(length(u.gf)))
            trial == 3 && (uu.gf[bus_index(s.net.net, 29)] = 500.0)          # a fault conductance at the line's end
            f, M, L = phs_rhs(s, y, uu), phs_matrices(s, y, uu), phs_ledger(s, y, uu)
            den = abs.(M.J) * abs.(M.z) .+ abs.(M.R) * abs.(M.z) .+ abs.(M.g)
            @test maximum(abs.(f .- ((M.J - M.R) * M.z .+ M.g)) ./ den) < 1e-12
            @test M.J == -M.J'
            @test eigmin(Symmetric(M.R)) > -1e-12 * opnorm(M.R)
            @test M.z ≈ ForwardDiff.gradient(w -> phs_energy(s, w), y) rtol = 1e-12
            bal = abs.(L.rate .- (L.port .- L.loss)) ./ (abs.(L.rate) .+ abs.(L.port) .+ abs.(L.loss) .+ floatmin())
            @test maximum(bal) < 1e-12
            @test abs(L.dH - (L.supply - L.losses)) < 1e-12 * (abs(L.supply) + L.losses)
            @test L.H ≈ phs_energy(s, y) rtol = 1e-13
            st = phs_structure(s, y, uu)
            @test maximum(abs, st.J .+ st.J') == 0 && eigmin(Symmetric(st.R)) > -1e-12 * opnorm(st.R)
            @test maximum(abs, (st.J - st.R) * st.z .+ st.G * st.v .- f) < 1e-12 * maximum(abs, f)
            T = PHPS.tpl_assemble_system(s, y, uu)
            @test T.J ≈ st.J atol = 1e-12
            @test T.R ≈ st.R atol = 1e-12
        end
    end

    @testset "fault run: the discrete energy balance" begin
        r = phs_simulate(s, x0, u; tstop = 0.006, faults = [PHFault(29, 0.001, 0.001, 0.003)],
                         tol = 1e-14)
        @test all(r.scheme_max .<= 1e-10 .* r.throughput)
        @test r.total.scheme_max < 1e-10 * r.total.throughput
        k = findfirst(==("Line_33"), r.names)
        @test abs(r.closure[k]) <= 1e-10 * r.moved[k] + 1e-12 * r.throughput[k]
    end

    @testset "the subsynchronous network mode" begin
        A = ForwardDiff.jacobian(y -> phs_rhs(s, y, u), x0)
        E = eigen(A)
        ci = s.cidx[1]
        part = [(abs2(E.vectors[ci, k]) + abs2(E.vectors[ci+1, k])) / sum(abs2, E.vectors[:, k]) for k in eachindex(E.values)]
        k = argmax(part)
        f = abs(imag(E.values[k])) / 2pi
        # in the dq frame the abc resonance f_er appears at f0 - f_er and f0 + f_er (60 Hz here)
        @test real(E.values[k]) < 0
        @test (20 < f < 40) || (80 < f < 100)
        # the fixed-input model (no governors) drifts slowly with or without the capacitor (a real eigenvalue near
        # 0.01 /s); the capacitor does not add to it
        A0 = ForwardDiff.jacobian(y -> phs_rhs(s0, y, u0), x00)
        @test maximum(real, E.values) <= maximum(real, eigvals(A0)) + 1e-6
    end

    @testset "xc -> 0: the uncompensated line" begin
        eqe = solve_equilibrium(compensated(1e-9 * xline), sc)
        se, xe, ue, repe = phs_equilibrium(eqe)
        @test repe.residual < 1e-12
        @test xe[1:s0.n] ≈ x00 rtol = 1e-7
        f = PHFault(29, 0.001, 0.001, 0.003)
        re = phs_simulate(se, xe, ue; tstop = 0.004, faults = [f], save_dt = 0.004, tol = 1e-13)
        r0 = phs_simulate(s0, x00, u0; tstop = 0.004, faults = [f], save_dt = 0.004, tol = 1e-13)
        @test re.X[1:s0.n, end] ≈ r0.X[:, end] rtol = 1e-6
    end
end
