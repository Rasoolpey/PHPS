# Two-machine pilot and PCC coefficient maps (docs/active/PCC_VIEW_N2_FORMULATION.md): the
# equilibrium and energy identities of both variants, the real-form machine currents against
# emdq_machine, and the Taylor-series coefficients against direct Lie derivatives.

@testset "emdq two-machine pilot and PCC coefficients" begin
    sc = load_scenario(joinpath(ROOT, "cases", "IEEE39Bus_PF", "no_fault_phtrue.json"))
    eq = solve_equilibrium(load_case(sc.system_path), sc)
    Random.seed!(7)
    for fb in (true, false)
        s, xs, rep = emdq_pilot2(eq; fixed_bus = fb)
        @test length(xs) == emdq2_nstates(s) == (fb ? 33 : 31)
        @test maximum(abs, emdq2_rhs(s, xs)) < 1e-9
        fb && @test rep.P2 ≈ rep.P2_case atol = 1e-9
        for _ in 1:3
            b = emdq2_balance(s, xs .+ 0.02 .* randn(length(xs)))
            @test abs(b["whole_error"]) < 1e-12
            @test abs(b["grid_error"]) < 1e-12
            @test abs(b["pcc_sum"]) < 1e-12
        end
    end

    # machine 2's shifted balance: rate = port - dissipation + r_rot
    let (s, xs, _) = emdq_pilot2(eq)
        xgs = xs[emdq2_nc(s)+1:end]
        for _ in 1:3
            t = emdq2_site2_shifted(s, xgs .+ 0.02 .* randn(length(xgs)), xgs)
            @test t.rate ≈ t.port - t.dissipation + t.r_rot rtol = 1e-9
        end
    end

    # the real-form currents equal emdq_machine's (GENSAL: the same state order)
    p1, x1, _ = emdq_pilot(eq)
    x = x1 .+ 0.1 .* randn(length(x1))
    e = emdq_machine(p1.mach, x)
    r = PHPS._site_currents(p1.mach, x, 0)
    @test r.inD + im * r.inQ ≈ e.in atol = 1e-12
    @test r.Tem ≈ e.Tem atol = 1e-12
    @test r.ifd ≈ e.ifd atol = 1e-12

    # Taylor arithmetic: sin, cos and a mixed word on a toy system with a nonconstant one
    u = Jet([0.3, 1.0, 0.0, 0.0])
    @test sin(u).c[2] ≈ cos(0.3) && sin(u).c[3] ≈ -sin(0.3) / 2 && cos(u).c[4] ≈ sin(0.3) / 6
    ftoy!(dx, x) = (dx[1] = x[2]; dx[2] = -sin(x[1]) + x[1] * x[2]; dx)
    x0 = [0.4, -0.7]
    Xs = jet_flow(ftoy!, x0, 1)
    Z = [Jet([Xs[i], zero(Xs[i]) + (i == 2 ? 1.0 : 0.0)]) for i in 1:2]
    Y = jet_flow(ftoy!, Z, 1)
    @test Y[2].c[2].c[2].c[2] ≈ x0[2]          # L_f L_g L_f x2 = x2 for g = e2

    # the pilot's coefficients against nested ForwardDiff Lie derivatives
    s, xs, _ = emdq_pilot2(eq)
    G = emdq2_grid_layout(s)
    xg = xs[emdq2_nc(s)+1:end] .+ 0.05 .* randn(G.n)
    tau = 1e-3
    f(x) = (dx = similar(x, promote_type(eltype(x), Float64)); emdq2_grid_rhs!(dx, s, x, 0, 0); tau .* dx)
    g = zeros(G.n)
    g[G.v2] = tau * s.m2.mach.omega_b / s.C2
    L(φ, X) = x -> ForwardDiff.jacobian(φ, x) * X(x)
    c1 = L(x -> x[G.v2:G.v2+1], f)
    c2 = L(c1, f)
    c3 = L(c2, f)
    cg = L(c2, x -> g)
    vals = pcc_coefficient_values(s, xg; K = 4, tau, A = 1)
    w = pcc_words(s; K = 4, A = 1)
    pick(a, j, b) = [vals[i] for i in eachindex(w) if (w[i].a, w[i].j, w[i].b) == (a, j, b)]
    @test pick(0, 0, 3) ≈ c3(xg) rtol = 1e-12
    @test pick(0, 1, 2) ≈ cg(xg) rtol = 1e-12

    # the explicit pH form of B's grid side and the identities of the reduction
    # (docs/active/PCC_REDUCTION_DERIVATION.md sections 1 and 3)
    let (sB, xsB, _) = emdq_pilot2(eq; fixed_bus = false)
        P = emdq2_grid_ph(sB)
        xg = xsB[emdq2_nc(sB)+1:end]
        xr0 = grid_ph_to_rotor(P, xg)
        @test grid_ph_from_rotor(P, xr0) ≈ xg atol = 1e-14
        fpil(xp, u) = (dx = similar(xp, promote_type(eltype(xp), eltype(u))); emdq2_grid_rhs!(dx, sB, xp, u[1], u[2]); dx)
        for _ in 1:3
            xp = xg .+ 0.05 .* randn(length(xg))
            u = 0.3 .* randn(2)
            fref = ForwardDiff.jacobian(z -> grid_ph_to_rotor(P, z), xp) * fpil(xp, u)
            x = grid_ph_to_rotor(P, xp)
            @test grid_ph_rhs(P, x, u) ≈ fref rtol = 1e-12
            @test grid_ph_energy(P, x) ≈ emdq2_grid_energy(sB, xp) rtol = 1e-13
            J, R, _ = grid_ph_JRg(P, x, u, (0.5, 0.2))
            @test maximum(abs, J + J') == 0.0
            @test minimum(eigvals(Symmetric(R))) >= 0
        end
        # the pointwise physical ledger, with faults at both buses:
        # (Q x)' f = pcc + shaft + dc_source - sum of losses
        for gf in ((0.0, 0.0), (100.0, 0.0), (0.0, 100.0))
            x = grid_ph_to_rotor(P, xg .+ 0.05 .* randn(length(xg)))
            u = 0.3 .* randn(2)
            pw = grid_ph_power(P, x, u, gf)
            lhs = dot(P.Q * x, grid_ph_rhs(P, x, u, gf))
            rhs = pw.pcc + pw.shaft + pw.dc_source - (pw.lines + pw.loads + pw.fault + pw.stator +
                                                        pw.field + pw.dampers + pw.source)
            @test lhs ≈ rhs rtol = 1e-10 atol = 1e-10
        end
        # a reduced basis containing the operating point's pair and one more pair
        yO = xr0[P.O]
        VO = pcc_add_pair(P, zeros(length(P.O), 0), yO)
        VO = pcc_add_pair(P, VO, randn(length(P.O)))
        Rg = reduced_grid(P, VO)
        @test Rg.Wf' * Rg.Phi ≈ I atol = 1e-12
        xr = reduced_project(Rg, xr0) .+ 0.01 .* randn(size(Rg.Phi, 2))
        x = Rg.Phi * xr
        z = P.Q * x
        @test Rg.Wf * (Rg.Phi' * z) ≈ z rtol = 1e-12                      # 3.1 effort
        u = [0.4, -0.2]
        gf = (1.0, 0.0)
        J, R, g = grid_ph_JRg(P, x, u, gf)
        Jr, Rr = Rg.Wf' * J * Rg.Wf, Rg.Wf' * R * Rg.Wf
        @test reduced_rhs(Rg, xr, u, gf) ≈ (Jr - Rr) * (Rg.Phi' * z) + Rg.Wf' * g rtol = 1e-12   # 3.2
        @test (Rg.Phi' * z)' * Rr * (Rg.Phi' * z) ≈ z' * R * z rtol = 1e-12   # 3.3 losses
        @test maximum(abs, reduced_rhs(Rg, reduced_project(Rg, xr0), [-real(PHPS._site_currents(sB.m1.mach, xsB, 0).inD) / sB.m1.tap,
                                                                        -real(PHPS._site_currents(sB.m1.mach, xsB, 0).inQ) / sB.m1.tap])) < 1e-8   # 3.5
        xo = xr0 .+ 0.05 .* randn(length(xr0))
        xp0 = reduced_project(Rg, xo)
        @test grid_ph_energy(P, xo) ≈ grid_ph_energy(P, Rg.Phi * xp0) + grid_ph_energy(P, xo - Rg.Phi * xp0) rtol = 1e-12   # 3.6
    end

    # the generic one-sided Jacobi SVD used for the high-precision profile
    Mt = randn(9, 5) * Diagonal([1.0, 1e-3, 1e-6, 1e-9, 0.0])
    Sj, Vj = PHPS._jacobi_svd(big.(Mt))
    @test Float64.(Sj[1:4]) ≈ svdvals(Mt)[1:4] rtol = 1e-8
    @test abs(Sj[5]) < 1e-60
    @test norm(big.(Mt) * Vj[:, 5]) < 1e-60
end
