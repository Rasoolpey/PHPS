# The saturating transformer core on a converter's merged branch (docs/active/PH_CONVERTER_MODEL.md, sections 20-24;
# src/phsim/transformer.jl) and the grid-forming droop controller (sections 15-17; src/phsim/converter.jl): the curve,
# the reduced energy and its derivatives, the discrete gradient, the T-circuit's three expressions of the core voltage,
# a DC test, and the droop case with cores (equilibrium, refusals, continuous and discrete balances).

@testset "saturating transformer core and the droop controller" begin
    cx = PHPS._cx
    wb = 2pi * 60

    @testset "the curve (GMD test case, Haddadi et al. 2020, Table IV)" begin
        c1, c8 = core_curve_gmd(1.0), core_curve_gmd(8.0)
        Λ = [0, 1, 1.075, 1.15, 1.2, 1.23, 1.37, 1.55, 1.86]
        I = [0, 0.001, 0.01, 0.025, 0.05, 0.1, 0.5, 1, 2]
        for k in eachindex(Λ)
            @test PHPS.core_f(c1, Λ[k]) ≈ I[k] atol = 1e-15
            @test PHPS.core_f(c1, -Λ[k]) ≈ -I[k] atol = 1e-15               # odd
            @test PHPS.core_f(c8, Λ[k]) ≈ 8 * I[k] atol = 1e-14             # the system base: currents times S_T / S_b
            k > 1 && @test PHPS.core_f(c1, Λ[k] - 1e-12) ≈ I[k] atol = 1e-9  # continuous at the knots
        end
        @test PHPS.core_fp(c1, 2.5) ≈ 1 / 0.31 && PHPS.core_f(c1, 2.5) ≈ 2 + (2.5 - 1.86) / 0.31   # the saturated slope beyond
        @test c1.Lm == 1000 && c8.Lm == 125
        # w' = f, w(0) = 0, w_nl = w - l^2/(2 Lm) in closed form, zero on the first segment
        for l in (-2.3, -1.4, -0.7, 0.3, 0.99, 1.1, 1.21, 1.5, 1.9, 2.4)
            h = 1e-6
            @test (PHPS.core_w(c8, l + h) - PHPS.core_w(c8, l - h)) / 2h ≈ PHPS.core_f(c8, l) rtol = 1e-6
            @test PHPS.core_wnl(c8, l) ≈ PHPS.core_w(c8, l) - l^2 / (2 * c8.Lm) atol = 1e-12
        end
        @test PHPS.core_wnl(c8, 0.999) == 0 && PHPS.core_w(c8, 0.0) == 0
        @test_throws ArgumentError PHPS.PHCoreCurve([0, 1, 0.5], [0, 1, 2], 1.0, 1.0; source = "")
    end

    # a core on a test branch (system base; not the case's)
    mkcore(curve; L1 = 0.0196, L2 = 0.0071, R1 = 0.00035, R2 = 0.00035) =
        PHCore("test", 0, 1, 1, 3, L1, R1, L2, R2, 0.0125, 800.0, curve)
    q = mkcore(core_curve_gmd(8.0))
    rng = MersenneTwister(23)

    @testset "the reduced energy: minimiser, envelope, implicit derivative, rotation" begin
        # a linear curve (f = l / Lm): W = |lambda|^2/(2 Lm) whatever the clock, i_m = lambda / Lm (T1 row 2)
        Lm = 50.0
        ql = mkcore(PHPS.PHCoreCurve([0, 1, 2], [0, 1 / Lm, 2 / Lm], 1 / Lm, 1.0; source = "linear"))
        for _ in 1:5
            lam, th = complex(randn(rng), randn(rng)), 2pi * rand(rng)
            @test PHPS._core_W(ql.curve, lam, th) ≈ abs2(lam) / (2Lm)
            @test PHPS._core_im(ql.curve, lam, th) ≈ lam / Lm
        end
        # T1 row 3: phase fluxes (1.2, -0.6, -0.6); the retained phase currents are f(lambda_k) less their mean
        c1 = core_curve_gmd(1.0)
        imc = PHPS._core_im(c1, 1.2 + 0im, 0.0)
        raw = [PHPS.core_f(c1, 1.2), PHPS.core_f(c1, -0.6), PHPS.core_f(c1, -0.6)]
        @test collect(phs_phases(imc, 0.0)) ≈ raw .- sum(raw) / 3 atol = 1e-14
        @test phs_phases(imc, 0.0)[1] ≈ 0.0337 atol = 1e-4
        # at random states, unsaturated and deep in saturation
        for amp in (0.6, 1.0, 1.15, 1.4, 1.9), _ in 1:3
            lam0 = amp * cis(2pi * rand(rng))
            iS0 = 6 * cis(2pi * rand(rng))
            iH0 = iS0 + 0.5 * randn(rng) * cis(2pi * rand(rng))
            pH, pS, th = q.L2 * iH0 + lam0, lam0 - q.L1 * iS0, 2pi * rand(rng)
            c = core_solve(q, pH, pS, th)
            # Kirchhoff's current law at the core node: i_H = i_S + i_m
            @test abs(c.iH - c.iS - PHPS._core_im(q.curve, c.lam, th)) < 1e-12 * (1 + abs(c.iH))
            # the minimum: no direction lowers the objective
            P = PHPS._core_Phi(q, c.lam, pH, pS, th)
            for d in (1e-4, 1e-4im, -1e-4, -1e-4im)
                @test PHPS._core_Phi(q, c.lam + d, pH, pS, th) >= P - 1e-15
            end
            # the envelope: the energy's gradient is (i_H, -i_S); with the implicit derivative through ForwardDiff
            gr = ForwardDiff.gradient(z -> core_energy(q, complex(z[1], z[2]), complex(z[3], z[4]), z[5]),
                                      [real(pH), imag(pH), real(pS), imag(pS), th])
            @test gr[1:4] ≈ [real(c.iH), imag(c.iH), -real(c.iS), -imag(c.iS)] rtol = 1e-9
            # the clock: dh/dtheta from the rotation invariance (record 23.1), in flux units
            @test gr[5] ≈ PHPS.core_rotation_rate(c.iH, c.iS, pH, pS) rtol = 1e-7 atol = 1e-12
            # (23.1): a common rotation of the loop fluxes is a turn of the clock
            phi = 0.37
            @test core_energy(q, pH * cis(phi), pS * cis(phi), th - phi) ≈ core_energy(q, pH, pS, th) rtol = 1e-13
            # the implicit derivative of i_S against central differences (away from a knot)
            dH, dS, dth = complex(randn(rng), randn(rng)), complex(randn(rng), randn(rng)), randn(rng)
            iSt(t) = core_solve(q, pH + t * dH, pS + t * dS, th + t * dth).iS
            ad = ForwardDiff.derivative(iSt, 0.0)
            hh = 1e-7
            fd = (iSt(hh) - iSt(-hh)) / 2hh
            # a knot crossed within the difference shows as a jump of the one-sided slopes; skip those samples
            onesided = abs((iSt(hh) - iSt(0.0)) / hh - (iSt(0.0) - iSt(-hh)) / hh)
            onesided < 1e-4 * abs(ad) + 1e-8 && @test abs(ad - fd) < 1e-5 * (1 + abs(ad))
            # convexity along a random line (the reduced energy is convex at a fixed clock)
            hm = core_energy(q, pH, pS, th)
            @test (core_energy(q, pH + dH, pS + dS, th) + core_energy(q, pH - dH, pS - dS, th)) / 2 >= hm - 1e-12
        end
        # one-sided derivatives at a knot: the solve's derivative is the one of the segment it lies on
        lamk = 1.15 + 0im                             # phase a exactly at a knot when theta = 0: the loop fluxes
        pH = q.L2 * (5.0 + PHPS._core_im(q.curve, lamk, 0.0)) + lamk   # whose minimiser is lamk (KCL there)
        pS = lamk - q.L1 * 5.0
        c = core_solve(q, pH, pS, 0.0)
        e = 1e-7
        imag_(t) = (cc = core_solve(q, pH + t, pS + t, 0.0); cc.iH - cc.iS)   # the magnetising current
        up = (imag_(e) - imag_(0.0)) / e
        dn = (imag_(0.0) - imag_(-e)) / e
        sup = ForwardDiff.derivative(imag_, 0.0)
        @test abs(up - dn) > 1e-3                     # a kink
        @test min(abs(sup - up), abs(sup - dn)) < 1e-4
    end

    @testset "the discrete gradient (record section 23)" begin
        for amp in (0.8, 1.05, 1.3, 1.8), _ in 1:4
            lam0 = amp * cis(2pi * rand(rng))
            iS0 = 6 * cis(2pi * rand(rng))
            pH0, pS0 = q.L2 * (iS0 + 0.01) + lam0, lam0 - q.L1 * iS0
            dH, dS = 0.02 * complex(randn(rng), randn(rng)), 0.02 * complex(randn(rng), randn(rng))
            pH1, pS1, thm = pH0 + dH, pS0 + dS, 2pi * rand(rng)
            iH, iS = core_dgrad(q, pH0, pS0, pH1, pS1, thm)
            dh = core_energy(q, pH1, pS1, thm) - core_energy(q, pH0, pS0, thm)
            @test abs(real(conj(iH) * dH) - real(conj(iS) * dS) - dh) < 1e-13 * (1 + core_energy(q, pH0, pS0, thm))
            # symmetric in its two ends
            iHr, iSr = core_dgrad(q, pH1, pS1, pH0, pS0, thm)
            @test abs(iHr - iH) < 1e-11 && abs(iSr - iS) < 1e-11
            # consistent: the midpoint gradient up to the increment
            m = core_solve(q, (pH0 + pH1) / 2, (pS0 + pS1) / 2, thm)
            @test abs(iH - m.iH) < 1 && abs(iS - m.iS) < 1
        end
        # in the linear part the correction is exactly zero: the midpoint gradient itself
        lam0 = 0.7 + 0.1im
        pH0, pS0 = q.L2 * 5 + lam0, lam0 - q.L1 * 5
        iH, iS = core_dgrad(q, pH0, pS0, pH0 + 0.01, pS0 + 0.01im, 0.4)
        m = core_solve(q, pH0 + 0.005, pS0 + 0.005im, 0.4)
        @test iH == m.iH && iS == m.iS
    end

    @testset "a DC voltage on the converter side: the core saturates, the HV side's DC decays" begin
        # one branch alone: source u = 1 + D e^{-j theta} (a DC voltage D in the phases), HV side on a stiff 1 pu
        # bus; the step of record section 23 (exact rotation, discrete gradient); larger resistances than the case's
        # so that the DC settles within the test
        qd = mkcore(core_curve_gmd(1.0); L1 = 0.15, L2 = 0.075, R1 = 0.05, R2 = 0.05)
        D, h, T = 0.05, 1e-4, 0.4
        u(th) = 1.0 + D * cis(-th)
        b = cis(wb * h / 2)
        z = [complex(-1im * 1.0), complex(-1im * 1.0)] ./ wb           # x_H, x_S at the no-load flux
        th = 0.0
        acc = 0.0
        iSs, iHs, ts = ComplexF64[], ComplexF64[], Float64[]
        E(z, th) = core_energy(qd, wb * z[1], wb * z[2], th) / wb
        for k in 1:round(Int, T / h)
            thm = th + wb * h / 2
            um, vm = u(thm), 1.0 + 0im
            function F(y)
                z1 = [complex(y[1], y[2]), complex(y[3], y[4])]
                iH, iS = core_dgrad(qd, wb * z[1] / b, wb * z[2] / b, wb * z1[1] * b, wb * z1[2] * b, thm)
                zm = (z .+ z1) ./ 2
                rH = z1[1] * b - z[1] / b - h * (vm - qd.R2 * iH)
                rS = z1[2] * b - z[2] / b - h * (um + qd.R1 * iS)
                return [real(rH), imag(rH), real(rS), imag(rS)], iH, iS
            end
            y = [real(z[1]), imag(z[1]), real(z[2]), imag(z[2])]
            for _ in 1:30
                r = F(y)[1]
                maximum(abs, r) < 1e-15 && break
                J = reduce(hcat, [(F(y .+ 1e-9 .* (1:4 .== j))[1] .- F(y .- 1e-9 .* (1:4 .== j))[1]) ./ 2e-9 for j in 1:4])
                y .-= J \ r
            end
            r, iH, iS = F(y)
            E0 = E(z, th)
            z = [complex(y[1], y[2]), complex(y[3], y[4])]
            th += wb * h
            # the discrete balance of the step: the stored energy's change is the supply less the losses
            acc = max(acc, abs(E(z, th) - E0 - h * (real(vm * conj(iH)) - real(um * conj(iS)) - qd.R2 * abs2(iH) -
                                                    qd.R1 * abs2(iS))))
            push!(iSs, iS * cis(th)); push!(iHs, iH * cis(th)); push!(ts, k * h)  # stationary frame
        end
        @test acc < 1e-14
        # the DC parts over the last cycle: the source carries D / R1 (the core is a short circuit at DC), the HV side none
        n = round(Int, (1 / 60) / h)
        dcS, dcH = abs(sum(iSs[end-n+1:end]) / n), abs(sum(iHs[end-n+1:end]) / n)
        @test abs(dcS - D / qd.R1) < 0.05 * D / qd.R1
        @test dcH < 0.02 * dcS
        # and the core went into saturation getting there
        c = core_solve(qd, wb * z[1], wb * z[2], th)
        @test maximum(abs(real(c.lam * cis(th + p))) for p in PHPS._CORE_PHASES) > 1.37
    end

    # ---- the droop case (PowerFactory's GFM_droop placement) with the cores ----
    scd = load_scenario(joinpath(ROOT, "cases", "IEEE39Bus_PF_gfm-droop-pf", "ph_fault_bus16.json"))
    eqd = solve_equilibrium(load_case(scd.system_path), scd)
    Ts = 5e-5
    cores = Dict(33 => 800.0, 36 => 700.0, 37 => 700.0, 38 => 1000.0)

    @testset "the droop controller without cores (sections 15-17)" begin
        cs, x0, u, rep = phs_controlled(eqd; loads = :series_all, converter_Ts = Ts)
        @test [cv.bus for cv in cs.s.conv] == [31, 33, 34, 36, 37, 38] && all(g -> g isa PHDroop, cs.conv)
        @test rep.residual < 1e-12 && all(c -> !c.engaged && c.divider < 1e-9 && c.emf_vs_terminal < 1e-12, rep.converters)
        # the combined units: bus 34 is two converters, its gains per unit of the combined rating
        g34, p34 = cs.conv[3], cs.s.conv[3].comp.p
        @test g34.Zs ≈ p34.Zseries / 2 && g34.i_con_lim ≈ 2 * p34.i_con_lim && g34.kpx ≈ p34.kpx / 4 && g34.mp ≈ p34.mp / 2
        # the limiter's three parts on hand-made states (controller frame; record section 16)
        g = cs.conv[2]
        xc = [0.3, g.p_set, g.q_set, 1.05, 1.0, 0.0, 0.5 * g.i_lim, 0.0]
        d = droop_output(g, xc)
        @test !d.boost && !d.stage1 && !d.stage2 && d.uc ≈ 1.05 * cis(0.3)
        xc[4] = 1.4                                    # the EMF far from the measured voltage: Stage 1 clips the difference
        d = droop_output(g, xc)
        @test d.stage1 && !d.stage2 && abs(d.uc - d.vmeas) ≈ g.Zs * g.i_con_lim
        xc[4], xc[7] = 1.05, 1.1 * g.i_lim             # above i_lim: the boost (kpr + j kpx)(|i| - i_lim) i, then Stage 1
        d = droop_output(g, xc)
        upre = 1.05 * cis(0.3) - complex(g.kpr, g.kpx) * 0.1 * g.i_lim * (1.1 * g.i_lim) * cis(0.3)
        dv = upre - d.vmeas
        @test d.boost && !d.stage2 && d.uc ≈ d.vmeas + dv * min(1, g.Zs * g.i_con_lim / abs(dv))
        xc[7] = 1.2 * g.i_con_lim                      # above i_con_lim: Stage 2 scales the difference by i_con_lim/|i|
        d = droop_output(g, xc)
        @test d.stage2
        # a call: the output from the states before it, then the Euler step of the eight states
        o = cs.coff[2]
        xd = copy(x0); xd[o] += 0.2; xd[o+3] *= 1.1
        upd = phs_control_inputs(cs, xd, u.gf)
        fd = PHPS.phs_flows(cs.s, xd, upd)
        vk, ic = PHPS._conv_vk(cs.s, cs.s.conv[2], xd, fd), PHPS._conv_current(fd, cs.s.conv[2])
        rates = PHPS.droop_rates!(zeros(8), g, view(xd, o:o+7), vk, ic, cs.s.omega_b)
        xa = PHPS._sample!(cs, copy(xd), u)
        @test complex(xa[o+8], xa[o+9]) ≈ droop_output(g, view(xd, o:o+7)).uc
        @test xa[o:o+7] ≈ xd[o:o+7] .+ Ts .* rates
        # a short fault run: the discrete balance of every component
        r = phs_simulate(cs, x0, u; tstop = 4Ts, h = Ts / 4, save_dt = Ts, faults = [PHFault(16, 0.001, Ts, 3Ts)], tol = 1e-14)
        @test all(r.scheme_max .<= 1e-10 .* r.throughput .+ 1e-14)
    end

    @testset "the droop limiter's DC rejection (section 25)" begin
        cs, x0, u, rep = phs_controlled(eqd; loads = :series_all, converter_Ts = Ts, droop_dc_reject = true)
        @test all(g -> g.dcr && nstates(g) == 14, cs.conv) && rep.residual < 1e-12 && !any(c -> c.engaged, rep.converters)
        @test_throws ArgumentError phs_controlled(eqd; loads = :series_all, droop_dc_reject = true)   # sampled only
        g = cs.conv[2]
        # the filter: gain 1 at 0 Hz in the controller frame, 0 at -omega_n (a DC in the phases), stable
        H(f) = (zeta = cis(2pi * f * Ts); g.dcc * (1 - g.dg / (zeta - g.dlam)))
        @test abs(H(0) - 1) < 1e-14 && abs(H(-g.omega_n / 2pi)) < 1e-12 && abs(g.dlam) < 1
        # driven by a lagged signal that is a constant plus a DC in the phases (a -omega_n component in the frame):
        # the output converges to the constant
        Z(k) = 1.02 + 0.3 * cis(-g.omega_n * k * Ts)
        V = Z(0) / (1 - g.dlam)
        y = 0.0im
        for k in 0:2000
            y = g.dcc * (Z(k) - g.dg * V)
            V = g.dlam * V + Z(k)
        end
        @test abs(y - 1.02) < 1e-6
        # a call: the output from the filtered signals, the filter states V <- lam V + Z with the lags before their step
        o = cs.coff[2]
        xd = copy(x0); xd[o+4] += 0.05; xd[o+6] += 0.4
        xa = PHPS._sample!(cs, copy(xd), u)
        @test complex(xa[o+10], xa[o+11]) ≈ g.dlam * complex(xd[o+10], xd[o+11]) + complex(xd[o+4], xd[o+5])
        @test complex(xa[o+12], xa[o+13]) ≈ g.dlam * complex(xd[o+12], xd[o+13]) + complex(xd[o+6], xd[o+7])
        @test complex(xa[o+8], xa[o+9]) ≈ droop_output(g, view(xd, o:o+13)).uc
        # with the filter at rest the output is the unfiltered controller's
        gn = cs.conv[2]
        @test droop_output(gn, view(x0, o:o+13)).uc ≈ u.uc[2]
        r = phs_simulate(cs, x0, u; tstop = 4Ts, h = Ts / 4, save_dt = Ts, faults = [PHFault(16, 0.001, Ts, 3Ts)], tol = 1e-14)
        @test all(r.scheme_max .<= 1e-10 .* r.throughput .+ 1e-14)
    end

    @testset "the droop case with cores at buses 33, 36, 37, 38" begin
        cs, x0, u, rep = phs_controlled(eqd; loads = :series_all, converter_Ts = Ts, cores)
        s = cs.s
        @test length(s.cores) == 4 && s.iclk == s.n && phs_state_names(s)[end] == "clock"
        @test [q.bus for q in s.cores] == [33, 36, 37, 38] && all(cv -> (cv.core > 0) == (cv.bus in keys(cores)), s.conv)
        q33 = s.cores[1]
        @test q33.L1 ≈ q33.Lc + q33.L2 && q33.R1 == q33.R2 && q33.curve.Lm ≈ 1000 / 8
        # the operating point: rest residual, controllers at rest, converter powers as without cores (the
        # magnetising currents are about 0.1 % of the ratings)
        @test rep.residual < 1e-12 && all(c -> !c.engaged, rep.converters)
        csn, _, _, repn = phs_controlled(eqd; loads = :series_all, converter_Ts = Ts)
        for (c, cn) in zip(rep.converters, repn.converters)
            @test abs(c.p_set - cn.p_set) < 1e-3 * abs(cn.p_set)
        end
        @test all(c -> 0.98 < c.lam < 1.06, rep.core_flux)       # three of four just above the first knot
        # refusals: a core inside the reduced cutset at bus 20, a bus without a converter, continuous controllers
        @test_throws ArgumentError phs_controlled(eqd; loads = :series_all, converter_Ts = Ts, cores = Dict(34 => 600.0))
        @test_throws ArgumentError phs_controlled(eqd; loads = :series_all, converter_Ts = Ts, cores = Dict(16 => 600.0))
        @test_throws ArgumentError phs_controlled(eqd; loads = :series_all, cores)
        # away from the operating point, in saturation: the T-circuit's three expressions of the core voltage agree
        # along the vector field (the reduced rows against the unreduced circuit), and every component balances
        x1 = copy(x0)
        for q in s.cores
            for i in (s.bidx[q.pos], q.xs)
                x1[i] *= 1.3; x1[i+1] *= 1.3
            end
        end
        x1[s.iclk] = 0.7
        up = phs_control_inputs(cs, x1, u.gf)
        f1, d1 = PHPS.phs_flows(s, x1, up), PHPS.phs_rhs(s, x1, up)
        for q in s.cores
            l = s.branches[q.pos]
            xH, xS, th = cx(x1, s.bidx[q.pos]), cx(x1, q.xs), x1[s.iclk]
            dH, dS = cx(d1, s.bidx[q.pos]), cx(d1, q.xs)
            sol(t) = core_solve(q, wb * (xH + t * dH), wb * (xS + t * dS), th + t * wb)
            c = sol(0.0)
            dlam = ForwardDiff.derivative(t -> sol(t).lam, 0.0)
            diH = ForwardDiff.derivative(t -> sol(t).iH, 0.0)
            diS = ForwardDiff.derivative(t -> sol(t).iS, 0.0)
            vm_core = dlam / wb + im * c.lam
            vm_hv = f1.v[s.net.from[l]] / s.net.a[l] - q.R2 * c.iH - q.L2 * diH / wb - im * q.L2 * c.iH
            vm_src = f1.v[s.net.to[l]] + q.R1 * c.iS + q.L1 * diS / wb + im * q.L1 * c.iS
            @test abs(vm_core - vm_hv) < 1e-10 && abs(vm_core - vm_src) < 1e-10
            @test maximum(abs(real(c.lam * cis(th + p))) for p in PHPS._CORE_PHASES) > 1.15
        end
        L = phs_ledger(cs, x1, u)
        @test maximum(abs, L.rate .- (L.port .- L.loss)) < 1e-9 * max(1, maximum(L.tport))
        @test abs(L.interconnection) < 1e-9 * sum(L.tport)
        @test phs_energy(cs, x1) ≈ L.H
        # the terminal voltage of a cored converter, recovered along the vector field: u_c + j L_c i_S + (L_c/omega_b) i_S'
        cv = s.conv[2]
        q = s.cores[cv.core]
        diS = ForwardDiff.derivative(t -> core_solve(q, wb * (cx(x1, s.bidx[q.pos]) + t * cx(d1, s.bidx[q.pos])),
                                                       wb * (cx(x1, q.xs) + t * cx(d1, q.xs)), x1[s.iclk] + t * wb).iS, 0.0)
        @test PHPS._conv_vk(s, cv, x1, f1) ≈ up.uc[2] + im * cv.Lc * f1.isrc[cv.pos] + cv.Lc * diS / wb
        # the step rule sees the cores short-circuited too
        rule = phs_step_rule(cs, x0, u)
        @test rule.w_max >= maximum(abs ∘ imag, eigvals(PHPS._jacobian(cs, x0, u)))
        @test s.step[] === nothing
        # the discrete balance in saturation, with a fault: every component, and the cored branches' summed
        # discrete-gradient increments against their energy function (closure)
        r = phs_simulate(cs, x1, u; tstop = 4Ts, h = Ts / 4, save_dt = Ts, faults = [PHFault(16, 0.001, Ts, 3Ts)], tol = 1e-14)
        @test all(r.scheme_max .<= 1e-10 .* r.throughput .+ 1e-14)
        kc = [findfirst(==(s.net.branch[s.branches[q.pos]]), r.names) for q in s.cores]
        @test all(abs.(r.closure[kc]) .<= 1e-11 .* r.throughput[kc] .+ 1e-15)
        # the clock follows the elapsed time
        @test r.X[s.iclk, end] ≈ x1[s.iclk] + wb * 4Ts
    end
end
