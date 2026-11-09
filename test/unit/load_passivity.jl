# Load passivity (docs/records/early/LOAD_PASSIVITY.md) on the code's COMPLEXLOAD: the static law is shifted
# passive in the band (Proposition L1); the filtered law's incremental admittance matches the
# closed form and fails the positive-real test at every frequency above zero (Proposition L2);
# CLOAD_12's static law has negative supply outside the band (Proposition L3).

@testset "load passivity" begin
    case = load_case(joinpath(ROOT, "cases", "IEEE39Bus_PF", "system_phtrue.json"))
    function load(name; t1 = nothing)
        spec = component(case, name)
        pd = PHPS.component_params(case, spec)
        t1 === nothing || (pd["t1"] = t1)
        return build_component(spec.type, spec.name, pd)
    end
    function drawn(c, z, V)
        p = c.p
        Id, Iq, _, _ = PHPS._load_law(PHPS.NoModes(), p, [z], V)
        return [p.G_load_c * V[1] + p.Q_base_c * V[2] - Id,
                p.G_load_c * V[2] - p.Q_base_c * V[1] - Iq]
    end
    function zdot(c, z, V)
        dx = zeros(eltype(V), 1)
        PHPS._step!(dx, nothing, c, [z], V, c.p, PHPS.NoModes())
        return dx[1]
    end
    hmin(Y) = eigmin(Hermitian((Y + Y') / 2))

    for name in ("CLOAD_4", "CLOAD_12", "CLOAD_24")
        c = load(name)
        p = c.p
        V = p.Vini_c
        w0 = [0.0, V, 0.0]
        J = ForwardDiff.jacobian(w -> drawn(c, w[1], w[2:3]), w0)
        g = ForwardDiff.gradient(w -> zdot(c, w[1], w[2:3]), w0)
        Y(ω) = J[:, 2:3] + J[:, 1] * transpose(g[2:3]) / (im * ω - g[1])
        for ω in (0.1, 1 / p.t1_c, 1e3)
            F = 1 / (1 + im * ω * p.t1_c)
            P, Q = p.P0_c, p.Q0_c
            Ycf = [P*(F-1) Q; Q*(1-2F) P] ./ V^2
            @test maximum(abs, Y(ω) - Ycf) < 1e-12
            @test hmin(Y(ω)) < 0                        # not positive real
        end
        @test hmin(J[:, 2:3] + J[:, 1] * transpose(g[2:3]) / (-g[1])) > -1e-12   # DC
        # static law: positive semidefinite at the equilibrium, nonnegative supply in the band
        cs = load(name; t1 = 0.0)
        @test hmin(ForwardDiff.jacobian(v -> drawn(cs, 0.0, v), [V, 0.0])) > -1e-12
        Is = drawn(cs, 0.0, [V, 0.0])
        for r in range(0.8, 1.2; length = 21), θ in range(-pi, pi; length = 73)
            Vv = [r * cos(θ), r * sin(θ)]
            @test dot(Vv - [V, 0.0], drawn(cs, 0.0, Vv) - Is) > -1e-10
        end
    end

    # CLOAD_12 outside the band: a counterexample to shifted passivity of the static law
    c12 = load("CLOAD_12"; t1 = 0.0)
    Vs = [c12.p.Vini_c, 0.0]
    Vb = [1.8 * cos(0.5236), 1.8 * sin(0.5236)]
    s = dot(Vb - Vs, drawn(c12, 0.0, Vb) - drawn(c12, 0.0, Vs))
    @test s < -0.1
end
