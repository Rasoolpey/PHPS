# Component port passivity (docs/active/COMPONENT_PASSIVITY.md): a machine's bus port fails the
# positive-real test, proved in interval arithmetic at the enclosed IEEE-39 equilibrium; an
# exciter has relative degree 2 and a real part of both signs; the validated root of the
# converters' virtual-impedance bisection encloses the Float64 bisection.

@testset "component ports" begin
    sc = load_scenario(joinpath(ROOT, "cases", "IEEE39Bus_PF", "no_fault_phtrue.json"))
    eq = solve_equilibrium(load_case(sc.system_path), sc)
    sys = eq.sys
    ins, _ = component_io(sys, eq.x, eq.V)
    m = section_model(eq)
    enc = enclose_equilibrium(m)
    xb = lift(m, enc.eta)
    ub, _ = component_io(sys, xb, enc.V)
    rng(k) = sys.offsets[k]:(sys.offsets[k] + PHPS.nstates(sys.comps[k]) - 1)
    k_of(n) = findfirst(c -> PHPS.name(c) == n, sys.comps)

    # machine GENROU_2: bus port with Efd, Tm held; supply = power absorbed
    k = k_of("GENROU_2")
    c = sys.comps[k]
    P = component_port(c, eq.x[rng(k)], ins[k]; inputs = ["Vd", "Vq"], outputs = ["Id", "Iq"],
                       norton = true)
    ws = 10 .^ range(0, 2; length = 401)
    lam, vecs = port_hermitian_scan(P, ws; sign = -1)
    j = argmin(lam)
    @test lam[j] < -30
    Pi = component_port(c, xb[rng(k)], ub[k]; inputs = ["Vd", "Vq"], outputs = ["Id", "Iq"],
                        norton = true)
    q = -port_witness_interval(Pi, ws[j], vecs[j])
    @test PHPS.IA.sup(q) < 0                            # proved: no storage makes it passive

    @testset "polar machine port" begin
        x0, u0 = eq.x[rng(k)], ins[k]
        V0 = u0[1:2]
        I0 = injected_current(c, x0, u0)
        In = complex(PHPS.injection(c, x0, V0)...)
        @test complex(I0...) ≈ In - norton_admittance(c) * complex(V0...)

        # Independently linearise the nonlinear equations in polar coordinates and
        # compute power as V * conj(I), including the loaded operating-point current.
        n = length(x0)
        w0 = [atan(V0[2], V0[1]), log(norm(V0))]
        function polar_equations(z)
            x = z[1:n]
            V = exp(z[n+2]) * cis(z[n+1])
            u = vcat(real(V), imag(V), u0[3:end])
            dx = rhs!(similar(x), c, x, u)
            In = complex(PHPS.injection(c, x, u[1:2])...)
            S = V * conj(In - norton_admittance(c) * V)
            return vcat(dx, real(S), imag(S))
        end
        J = ForwardDiff.jacobian(polar_equations, vcat(x0, w0))
        reference = ComponentPort("polar reference", P.states, ["theta", "rho"], ["P", "Q"],
                                  J[1:n, 1:n], J[1:n, n+1:end],
                                  J[n+1:end, 1:n], J[n+1:end, n+1:end])
        for w in (0.2, 8.51, 100.0)
            @test polar_transfer(P, V0, I0, w) ≈ port_transfer(reference, w) rtol = 1e-11 atol = 1e-11
        end

        # A component without a Norton stamp returns its declared current unchanged.
        kl = k_of("CLOAD_12")
        cl = sys.comps[kl]
        @test norton_admittance(cl) === nothing
        @test injected_current(cl, eq.x[rng(kl)], ins[kl]) ≈
              collect(PHPS.injection(cl, eq.x[rng(kl)], ins[kl][1:2]))
    end

    # exciter IEEET1_2: relative degree 2; Re G takes both signs (neither sign is passive)
    k = k_of("IEEET1_2")
    E = component_port(sys.comps[k], eq.x[rng(k)], ins[k]; inputs = ["Vterm"], outputs = ["Efd"])
    @test E.D[1, 1] == 0 && (E.C * E.B)[1, 1] == 0 && (E.C * E.A * E.B)[1, 1] != 0
    re = [real(port_transfer(E, w)[1, 1]) for w in 10 .^ range(-3, 3; length = 601)]
    @test minimum(re) < 0 < maximum(re)

    # validated virtual-impedance root (regime off) encloses the Float64 bisection
    args = (0.2, 0.01, 5.0, 0.0, 0.05, 1.0, 1.0)            # du, Zseries, i_lim, r_vi, x_vi, kpx, kpr; root 3.33
    i64 = PHPS.vi_bisection(args...)
    iiv = PHPS.vi_bisection(PHPS.ival(args[1]), args[2:end]...)
    @test abs(i64 - PHPS.IA.mid(iiv)) < 1e-12 * i64
    @test_throws PHPS.UndecidedBranch PHPS.vi_bisection(PHPS.ival(0.2), 0.01, 1.0, 0.0,
                                                               0.05, 1.0, 1.0)
end

@testset "polar supply and scan signs" begin
    V0 = [0.8, 0.6]
    eye = Matrix{Float64}(I, 2, 2)
    static_port(D) = ComponentPort("static", String[], ["Vd", "Vq"], ["Id", "Iq"],
                                   zeros(0, 0), zeros(0, 2), zeros(2, 0), D)
    ws = [0.25, 1.0, 4.0]

    # A shunt susceptance has Q = -b |V|^2 and a symmetric, real polar response.
    b = 0.7
    Dc = [0.0 -b; b 0.0]
    capacitor = static_port(Dc)
    @test polar_transfer(capacitor, V0, Dc * V0, 1.0) ≈ [0.0 0.0; 0.0 -2b * dot(V0, V0)] atol = 1e-14
    lc, _ = polar_scan(capacitor, V0, Dc * V0, ws)
    @test maximum(abs, lc) < 1e-14

    # Conductance has P = -g |V|^2: its real, asymmetric response gives both signs
    # in the polar supply, so taking entrywise imaginary parts would miss the failure.
    g = 0.3
    Dr = -g .* eye
    resistor = static_port(Dr)
    @test polar_transfer(resistor, V0, Dr * V0, 1.0) ≈ [0.0 -2g * dot(V0, V0); 0.0 0.0] atol = 1e-14
    lr, _ = polar_scan(resistor, V0, Dr * V0, ws)
    @test lr ≈ fill(g * dot(V0, V0), length(ws))

    # The polar response is diag(2, -1)/(1 + iw); the largest supply eigenvalue
    # is +w/(1 + w^2), even though the first diagonal entry has negative imaginary part.
    dynamic = ComponentPort("opposite signs", ["x1", "x2"], ["Vd", "Vq"], ["Id", "Iq"],
                            -eye, eye, [0.0 2.0; 1.0 0.0], zeros(2, 2))
    ld, vectors = polar_scan(dynamic, [1.0, 0.0], zeros(2), ws)
    @test ld ≈ ws ./ (1 .+ ws .^ 2)
    for (w, value, v) in zip(ws, ld, vectors)
        @test norm(v) ≈ 1.0
        @test imag(dot(v, polar_transfer(dynamic, [1.0, 0.0], zeros(2), w) * v)) ≈ value
    end
end
