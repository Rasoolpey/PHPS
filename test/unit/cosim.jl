# Co-simulation through PH ports and the device-level SiC switch (docs/active/COSIM_DESIGN.md; review 102).

@testset "cosim: piecewise-linear capacitance" begin
    p = PHPS.PWLCap([-10.0, -1.0, 0.0, 2.0, 50.0], [5e-9, 4e-9, 3e-9, 1e-9, 0.2e-9])
    for (a, b) in ((-30.0, 70.0), (-0.5, 0.25), (1.0, 1.0 + 1e-7), (45.0, 3.0), (0.0, 0.0))
        n = 100000; xs = range(a, b; length = n + 1); h = (b - a) / n
        qn = sum(PHPS.capC(p, (xs[i] + xs[i+1]) / 2) for i in 1:n) * h
        @test isapprox(PHPS.capq(p, b) - PHPS.capq(p, a), qn; rtol = 1e-6, atol = 1e-24)
        dE = PHPS.capE(p, b) - PHPS.capE(p, a); dq = PHPS.capq(p, b) - PHPS.capq(p, a)
        @test isapprox(PHPS.capdelta(p, a, b), dE - (a + b) / 2 * dq; rtol = 1e-9, atol = 1e-22)
        @test PHPS.capdelta(p, a, b) == -PHPS.capdelta(p, b, a) || a == b
    end
    @test_throws ArgumentError PHPS.PWLCap([1.0, 2.0], [1e-9, 1e-9])        # no knot at 0
end

@testset "cosim: the device's discrete gradient and the leg step" begin
    c = PHPS.cab760m12hm3()
    d = c.dev
    @test c.extraction.alpha > 1 && c.extraction.Cgs > 0 && c.extraction.Cox > 0
    rng = Random.Xoshiro(2)
    worst = 0.0
    for _ in 1:2000
        a = (rand(rng) * 20 - 5, rand(rng) * 800 - 10); b = (rand(rng) * 20 - 5, rand(rng) * 800 - 10)
        dqG, dqD, pG, pD = PHPS.dev_discrete(d, a..., b...)
        dH = PHPS.dev_energy(d, b...) - PHPS.dev_energy(d, a...)
        worst = max(worst, abs(pG * dqG + pD * dqD - dH) / (abs(pG * dqG) + abs(pD * dqD)))
    end
    @test worst < 1e-12
    leg = PHPS.SiCLeg(d, c.RG, 8.4e-9, 106.5e-6, 126.3e-6)
    s = PHPS.leg_equilibrium(leg, 600.0, 300.0, false)
    @test isapprox(s.vds_l, -300 * 1.33e-3; rtol = 0.02)                        # R_DS(on) 1.33 mOhm (third quadrant)
    sol = PHPS.SiCLegSolver(leg)
    for (Vu, Vl) in ((-4.0, -4.0), (15.0, -4.0))                                # dead time, then the upper on
        for _ in 1:400
            ok, l = PHPS.leg_step!(sol, s, 0.25e-9, 600.0, 300.0, Vu, Vl)
            @test ok
            # the floor: Delta H from energy differences rounds at eps x the stored energy (about 5e-4 J here)
            Es = PHPS.dev_energy(d, s.vgs_u, s.vds_u) + PHPS.dev_energy(d, s.vgs_l, s.vds_l)
            # and the solve's own term: the residual tolerance (A) x the efforts (<= 1000 V) x h (review 102, row 10)
            @test abs(PHPS.defect(l)) <= 1e-12 * (abs(l.E_dc) + abs(l.E_ac) + abs(l.E_gate)) + 16 * eps() * Es +
                                            PHPS.LEG_TOL * 1000 * 0.25e-9
        end
    end
end

@testset "cosim: window-held coupling (group A)" begin
    slow = PHPS.SlowRC(1.0, 0.1, PHPS.QuarticStorage(1e-3, 5e3))
    fast = PHPS.FastRL(0.05, PHPS.QuarticStorage(1e-5, 1e5), 0.0)
    # P3: N = 1 equals the monolithic step
    q, l, qm, lm = 0.0, 0.0, 0.0, 0.0
    a = 0.0
    for _ in 1:200
        q, l, a, _, led, _ = PHPS.cosim_window(slow, fast, q, l, 1e-5, [1e-5]; a0 = a)
        qm, lm, _, _ = PHPS.monolithic_step(slow, fast, qm, lm, 1e-5)
        @test abs(led.closure) <= 1e-15 * (abs(led.Esup) + abs(led.D))
    end
    @test abs(q - qm) <= 1e-14 * abs(qm) && abs(l - lm) <= 1e-14 * abs(lm)
    # P1: an irregular fast mesh closes exactly
    rng = Random.Xoshiro(5)
    for _ in 1:200
        n = rand(rng, 1:20); w = rand(rng, n) .+ 0.05; hs = 1e-5 .* w ./ sum(w)
        q, l, a, _, led, _ = PHPS.cosim_window(slow, fast, q, l, 1e-5, hs; a0 = a)
        @test abs(led.closure) <= 1e-14 * (abs(led.Esup) + abs(led.D))
        @test abs(led.defect) <= 1e-12 * abs(led.Esup)
    end
    # P4: feed-through on both sides: the explicit exchange diverges, the window solve converges
    s4 = PHPS.SlowRC(1.0, 1.0, nothing); f4 = PHPS.FastRL(0.05, PHPS.QuarticStorage(1e-5, 0.0), 2.0)
    q4, l4, a4 = 0.0, 0.0, 1.0
    for _ in 1:500
        q4, l4, a4, _, _, _ = PHPS.cosim_window(s4, f4, q4, l4, 1e-5, fill(1e-6, 10); a0 = a4)
    end
    @test isapprox(l4, 1e-5 / (0.05 * 3 + 1); rtol = 1e-6)
    qe, le, ae, be = 0.0, 0.0, 1.0, 0.0
    for _ in 1:60
        qe, le, ae, be, _ = PHPS.explicit_vi_window(s4, f4, qe, le, 1e-5, fill(1e-6, 10), ae, be)
        abs(ae) > 1e3 && break                      # diverged (beyond this the fast side's Newton fails, as it should)
    end
    @test abs(ae) > 1e3
    # P10: the fast side in C, the same arithmetic
    if Sys.which("gcc") !== nothing
        fc = PHPS.FastRLc(fast, PHPS.build_c_fast())
        for _ in 1:50
            n = rand(rng, 1:20); w = rand(rng, n) .+ 0.05; hs = 1e-5 .* w ./ sum(w); l0 = 1e-5 * randn(rng); av = randn(rng)
            rj = PHPS.fast_window(fast, l0, av, hs); rc = PHPS.fast_window_c(fc, l0, av, hs)
            @test rj[1] == rc[1] && rj[2] == rc[2]
        end
    end
end

@testset "cosim: the bridge's modulation methods" begin
    m = 0.9 * cis(0.3)
    for mod in (:svpwm, :spwm, :dpwm1, :dpwmmin)
        d = PHPS._duties(m, mod)
        r = [real(m * cis(-a)) for a in PHPS._LEG]
        # the line-to-line duty differences carry the reference in every method
        @test isapprox(d[1] - d[2], (r[1] - r[2]) / 2; atol = 1e-14)
        @test isapprox(d[2] - d[3], (r[2] - r[3]) / 2; atol = 1e-14)
        mod in (:dpwm1, :dpwmmin) && @test any(x -> x == 0.0 || x == 1.0, d)
    end
    @test_throws ArgumentError PHPS._duties(m, :none)
end

@testset "cosim: the deferred fine exit (review 105)" begin
    P = PHPS
    c = P.cab760m12hm3(; T = 25); leg = P.SiCLeg(c.dev, c.RG, 8.4e-9, 0.0, 0.0)
    s = P.leg_equilibrium(leg, 600.0, -400.0, false)
    nf = fieldcount(P.SiCStepLedger)
    lspike = P.SiCStepLedger([i == 12 ? 1.0 : 0.0 for i in 1:nf]...)    # a gate current at the first step (iGu)
    lquiet = P.SiCStepLedger(zeros(nf)...)
    for defer in (false, true)
        f = P.FineTracker(; defer); P.fine_command!(f, 0.0)
        t = 0.0; texit = NaN
        for k in 1:8000                                                  # 2 us of 0.25 ns steps
            t += f.h_f
            P.fine_update!(f, t, f.h_f, s, k == 1 ? lspike : lquiet, s.vds_u, s.vds_l)
            (!f.active && isnan(texit)) && (texit = t)
        end
        if defer
            @test f.active && f.pending == 1 && f.exits == 0                # still fine until the window ends
            P.fine_window_end!(f)
            @test !f.active && f.pending == 0 && f.exits == 1
            P.fine_command!(f, t); @test f.active && f.pending == 0
        else
            @test !f.active && f.exits == 1 && 1.5e-6 <= texit < 1.6e-6     # immediate (the double-pulse test)
        end
    end
end
