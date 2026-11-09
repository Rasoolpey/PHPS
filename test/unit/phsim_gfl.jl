# The grid-following (GFL) converter on the EMT model (docs/active/PH_CONVERTER_MODEL.md, sections 36-40; src/phsim/converter.jl,
# PHGfl; src/components/converters/gfl.jl, the minimal RMS GFL_PHTRUE): PowerFactory's GFL placement, the RMS start, the rest
# point, the combined units, a call against the DLL's formulas written out here, the static generator's EMF, and the discrete
# balance of short fault runs, without and with the saturating cores. (The call-by-call check against the DLL's own code is
# scripts/ph_gfl_dll_harness.jl with pf/dsl/harness/gfl_harness.cpp; the static generator's current loop against PowerFactory
# is scripts/ph_gfl_sgtest_port.jl: both need files outside the repository's tests.)

@testset "the GFL converter (sections 36-40)" begin
    scg = load_scenario(joinpath(ROOT, "cases", "IEEE39Bus_PF_gfl-pf", "ph_fault_bus16.json"))
    caseg = load_case(scg.system_path)
    eqg = solve_equilibrium(caseg, scg)
    Ts = 5e-5
    cores = Dict(33 => 800.0, 36 => 700.0, 37 => 700.0, 38 => 1000.0)

    @testset "the minimal RMS component: the start at the dialog's set-points" begin
        gfls = [(k, c) for (k, c) in enumerate(eqg.sys.comps) if c isa GFL_PHTRUE]
        @test length(gfls) == 7 && all(((k, c),) -> nstates(c) == 2 && norton_admittance(c) === nothing, gfls)
        for (k, c) in gfls
            o, i = eqg.sys.offsets[k], eqg.sys.net.index[c.bus]
            S = complex(eqg.V[2i-1], eqg.V[2i]) * conj(complex(eqg.x[o], eqg.x[o+1]))
            @test S ≈ complex(c.p.p_ref, c.p.q_ref) atol = 1e-6
        end
        d = copy(param_dict(gfls[1][2]))
        d["T_gs"] = 0.01
        @test_throws ArgumentError GFL_PHTRUE("x", d)
    end

    @testset "rest point, combined units, refusals" begin
        cs, x0, u, rep = phs_controlled(eqg; loads = :series_all, converter_Ts = Ts)
        @test [cv.bus for cv in cs.s.conv] == [31, 33, 34, 36, 37, 38] && all(g -> g isa PHGfl && nstates(g) == 17, cs.conv)
        @test rep.residual < 1e-12
        @test all(c -> c.emf < 1e-12 && c.emf_vs_terminal < 1e-12 && c.reference_vs_current < 1e-6, rep.converters)
        # bus 34 is two units: the currents, powers and ramps times two, the reactance and the PI's gains halved
        g34, p34 = cs.conv[3], cs.s.conv[3].comp.p
        @test g34.i_max ≈ 2 * p34.i_max && g34.x1 ≈ p34.Zseries / 2 && g34.Kd ≈ p34.Kd / 2 && g34.Kq ≈ p34.Kq / 2
        @test g34.p_ref ≈ 2 * p34.p_ref && g34.q_ref ≈ 2 * p34.q_ref && g34.r_p_max ≈ 2 * p34.r_p_max && g34.ki_P == p34.ki_P
        # in the study case's per unit (on each rating): i_max 1.1, uk 0.1, Kd 1
        @test all(g -> g.i_max * g.x1 ≈ 0.11 && g.Kd / g.x1 ≈ 10 && g.ki_P == 1 && g.Ts == Ts, cs.conv)
        @test_throws ArgumentError phs_controlled(eqg; loads = :series_all)                       # sampled only
        @test_throws ArgumentError phs_controlled(eqg; loads = :series_all, converter_Ts = Ts, converter_umax = 1.15)
        @test_throws ArgumentError phs_controlled(eqg; loads = :series_all, converter_Ts = Ts,
                                                  converter_dc = (; H = 5e-3, battery = :sossan, soc = "40-50", mode = :held))
        # at rest the static generator's integrators hold still (to the equilibrium's tolerance) and the held frame turns at -omega_b
        dx = phs_rhs(cs, x0, u)
        for i in eachindex(cs.conv)
            o = cs.coff[i]
            @test maximum(abs, dx[o+14:o+15]) < 1e-5
            @test complex(dx[o+10], dx[o+11]) ≈ -im * cs.s.omega_b * complex(x0[o+10], x0[o+11]) atol = 1e-12
            @test all(==(0), dx[o:o+9]) && all(==(0), dx[o+12:o+13]) && dx[o+16] == 0
        end
        # one call at rest: the angle steps by Ts (omega_n - omega_b), nothing else of the DLL's moves; the outputs latched
        xa = PHPS._sample!(cs, copy(x0), u)
        for (i, g) in enumerate(cs.conv)
            o = cs.coff[i]
            @test xa[o] - x0[o] ≈ Ts * (g.omega_n - cs.s.omega_b) atol = 1e-15
            @test maximum(abs, xa[o+1:o+9] .- x0[o+1:o+9]) < 1e-12
            @test complex(xa[o+10], xa[o+11]) ≈ cis(x0[o]) && xa[o+12] == x0[o+8] && xa[o+13] == x0[o+9]
        end
    end

    @testset "a call, against the DLL's formulas (section 36)" begin
        cs, x0, u, rep = phs_controlled(eqg; loads = :series_all, converter_Ts = Ts)
        g = cs.conv[2]
        xc = collect(x0[cs.coff[2]:cs.coff[2]+16])
        xc[2] = 1e-3; xc[3] = 0.97; xc[4] = 1.002; xc[7] *= 1.1; xc[8] *= 0.9; xc[9] *= 1.05
        for vk in (0.95 * cis(xc[1] + 0.07), 0.2 * cis(xc[1] - 0.3), 0.05 * cis(xc[1]))   # gate 1, 0.5, 0
            v = vk * cis(-xc[1])
            vd, vq = real(v), imag(v)
            om = g.omega_n + g.kp_pll * vq + g.ki_pll * xc[2]
            eP = xc[5] - vd * xc[9]                       # the droops are off in this case
            eQ = xc[6] + vd * xc[10]
            idr, iqr = g.kp_P * eP + g.ki_P * xc[7], -(g.kp_Q * eQ + g.ki_Q * xc[8])
            m = hypot(idr, iqr)
            sat = m > g.i_max
            sat && (idr *= g.i_max / m; iqr *= g.i_max / m)
            gate = clamp((vd - 0.1) / 0.2, 0, 1)
            c = gfl_call(g, xc, vk)
            @test c.omega ≈ om && c.gate ≈ gate && c.sat == sat
            @test c.d[7] ≈ (sat ? 0 : gate) * eP && c.d[8] ≈ (sat ? 0 : gate) * eQ
            @test c.d[9] ≈ (gate * idr - xc[9]) / g.T_cc && c.d[10] ≈ (gate * iqr - xc[10]) / g.T_cc
            @test c.d[3] ≈ (abs(vk) - xc[3]) / g.T_meas && c.d[4] ≈ (om / g.omega_n - xc[4]) / g.T_meas
            xn = gfl_sample!(copy(xc), g, vk, cs.s.omega_b)
            @test xn[1] ≈ xc[1] + Ts * (om - cs.s.omega_b) && xn[2] ≈ xc[2] + Ts * vq
            @test xn[9] ≈ xc[9] + Ts * c.d[9] && complex(xn[11], xn[12]) ≈ cis(xc[1]) && xn[13] == xc[9]
            @test xn[17] ≈ om / g.omega_n                 # the monitor: the call's omega, held
        end
        # the gate's band and the clamp of phi
        @test PHPS._gfl_gate(0.3) == 1 && PHPS._gfl_gate(0.1) == 0 && PHPS._gfl_gate(0.2) ≈ 0.5
        xp = copy(xc); xp[2] = g.phi_max
        @test gfl_sample!(xp, g, 0.9 * cis(xc[1] + 0.2), cs.s.omega_b)[2] == g.phi_max
    end

    @testset "the exact boundaries: gate, dead band, limit, ramp, the start (G3 item 6)" begin
        cs, x0, u, rep = phs_controlled(eqg; loads = :series_all, converter_Ts = Ts)
        g = cs.conv[1]
        # the gate and the dead band at their edges (droop.h: >= high is 1, <= low is 0; |x| <= db is 0, shifted outside)
        @test PHPS._gfl_gate(0.3) === 1.0 && PHPS._gfl_gate(0.1) === 0.0 && PHPS._gfl_gate(0.3 - 1e-12) < 1
        @test PHPS._gfl_deadband(0.02, 0.02) == 0 && PHPS._gfl_deadband(-0.02, 0.02) == 0
        @test PHPS._gfl_deadband(0.025, 0.02) ≈ 0.005 && PHPS._gfl_deadband(-0.025, 0.02) ≈ -0.005
        # the limit at exactly i_max does not act (mag > i_max, strict); just above it scales onto i_max
        xc = collect(x0[cs.coff[1]:cs.coff[1]+16])
        vk = cis(xc[1])                                    # vd = 1, vq = 0
        xc[9], xc[10] = xc[5], -xc[6]                      # e_P = e_Q = 0
        xc[7], xc[8] = g.i_max / g.ki_P, 0.0               # id_raw = i_max, iq_raw = 0
        c = gfl_call(g, xc, vk)
        @test !c.sat && c.idc == g.i_max && c.d[7] == 0
        xc[7] *= 1 + 1e-9
        c = gfl_call(g, xc, vk)
        @test c.sat && hypot(c.idc, c.iqc) ≈ g.i_max && c.d[7] == 0 && c.d[8] == 0   # frozen on the limit
        # the ramp: at most r_max per second; the last step lands on the set-point exactly
        xr = copy(xc)
        xr[5] = g.p_ref - 0.5 * g.r_p_max * Ts
        @test gfl_call(g, xr, vk).d[5] ≈ 0.5 * g.r_p_max && gfl_sample!(copy(xr), g, vk, cs.s.omega_b)[5] ≈ g.p_ref
        xr[5] = g.p_ref + 10.0
        @test gfl_call(g, xr, vk).d[5] ≈ -g.r_p_max
        # the start under the gate (|v| = 0.2: gate 0.5) and under the limit (a set-point above i_max)
        x2 = gfl_initial(g, 0.2 * cis(0.3), 0.0im, 0.0im)
        @test x2[9] ≈ 0.5 * g.p_ref / 0.2 * min(1, g.i_max / hypot(g.p_ref, g.q_ref) * 0.2) && x2[13] == x2[9]
        gbig = PHGfl(g.name, g.kp_pll, g.ki_pll, g.omega_n, g.phi_max, g.i_max, g.T_cc, g.kp_P, g.ki_P, g.kp_Q, g.ki_Q, g.k_droop_p,
                     g.f_db, g.k_droop_q, g.v_db, g.r_p_max, g.r_q_max, g.T_meas, g.Kqf, 3 * g.i_max, 0.0, g.x1, g.Kd, g.Td, g.Kq, g.Tq, Ts)
        x3 = gfl_initial(gbig, 1.0 + 0im, 0.0im, 0.0im)
        @test hypot(x3[9], x3[10]) ≈ gbig.i_max && x3[7] ≈ x3[9] / gbig.ki_P
    end

    @testset "one unit against two identical units (G3 item 5)" begin
        # bus 34's two units of 300 MVA merged into one of 600 MVA: currents, powers, i_max, ramps doubled, Zseries and the PI's
        # gains halved; the EMT model must be the same converter (to the RMS equilibrium's tolerance)
        obj = JSON3.read(read(scg.system_path, String))          # ordered: the case's component order matters
        c1 = JSON3.read(JSON3.write(obj.components.GFL_34_1), Dict{String,Any})
        p = c1["params"]
        # (p_override, q_override: Python PHPS's per-unit dispatch, which the first pass takes for a bus's only unit)
        for k in ("i_max", "p_ref", "q_ref", "r_p_max", "r_q_max", "k_droop_p", "k_droop_q", "Kqf", "p_override", "q_override")
            p[k] *= 2
        end
        for k in ("Zseries", "Kd", "Kq")
            p[k] /= 2
        end
        c1["_plant_Sn_MVA"] = 600.0
        comps = join(("\"$k\":" * (k === :GFL_34_1 ? JSON3.write(c1) : JSON3.write(v)) for (k, v) in pairs(obj.components)
                      if k !== :GFL_34_2), ",")
        conns = JSON3.write([c for c in obj.connections if !occursin("GFL_34_2", JSON3.write(c))])
        body = join(("\"$k\":" * (k === :components ? "{" * comps * "}" : k === :connections ? conns : JSON3.write(v))
                     for (k, v) in pairs(obj)), ",")
        dir = mktempdir()
        write(joinpath(dir, "system_gfl_pf.json"), "{" * body * "}")
        cp(joinpath(dirname(scg.system_path), "ph_fault_bus16.json"), joinpath(dir, "ph_fault_bus16.json"))
        sc1 = load_scenario(joinpath(dir, "ph_fault_bus16.json"))
        eq1 = solve_equilibrium(load_case(sc1.system_path), sc1)
        csa, xa, ua, repa = phs_controlled(eqg; loads = :series_all, converter_Ts = Ts)
        csb, xb, ub, repb = phs_controlled(eq1; loads = :series_all, converter_Ts = Ts)
        @test csb.s.conv[3].units == 1 && csa.s.conv[3].units == 2 && length(xa) == length(xb)
        ga, gb = csa.conv[3], csb.conv[3]
        @test all(f -> getfield(ga, f) ≈ getfield(gb, f), (:i_max, :p_ref, :q_ref, :r_p_max, :x1, :Kd, :Kq, :ki_P, :Td))
        @test maximum(abs, xa .- xb) < 1e-6
        ra = phs_simulate(csa, xa, ua; tstop = 40Ts, h = Ts / 4, save_dt = Ts, faults = [PHFault(16, 0.001, 5Ts, 30Ts)], tol = 1e-12)
        rb = phs_simulate(csb, xb, ub; tstop = 40Ts, h = Ts / 4, save_dt = Ts, faults = [PHFault(16, 0.001, 5Ts, 30Ts)], tol = 1e-12)
        sa, sb = phs_converter_signals(csa, ra.X[:, end], ua), phs_converter_signals(csb, rb.X[:, end], ub)
        @test maximum(abs, ra.X .- rb.X) < 1e-5 && maximum(abs, sa.p .- sb.p) < 1e-5 && maximum(abs, sa.i .- sb.i) < 1e-5
    end

    @testset "the angle advance (the variant gfl_pf_adv, section 42)" begin
        cs, x0, u, rep = phs_controlled(eqg; loads = :series_all, converter_Ts = Ts, gfl_advance = 1.0)
        @test all(g -> g.advance == 1.0, cs.conv) && rep.residual < 1e-12 && all(c -> c.emf < 1e-12, rep.converters)
        for (i, g) in enumerate(cs.conv)
            o = cs.coff[i]
            # the start's frame as the first call latches it (omega = omega_n), the integrators holding u_c = E0
            @test complex(x0[o+10], x0[o+11]) ≈ cis(x0[o] + g.omega_n * Ts)
            xc = collect(x0[o:o+16])
            vk = 0.98 * cis(xc[1] + 0.03)
            c = gfl_call(g, xc, vk)
            xn = gfl_sample!(copy(xc), g, vk, cs.s.omega_b)
            @test complex(xn[11], xn[12]) ≈ cis(xc[1] + c.omega * Ts)          # the call's omega, rad/s
            @test xn[1] ≈ xc[1] + Ts * (c.omega - cs.s.omega_b)                 # the PLL's angle itself not advanced
        end
        @test_throws ArgumentError PHGfl(cs.s.conv[1]; Ts, advance = Inf)
        @test PHGfl(cs.s.conv[1]; Ts).advance == 0                              # the original DLL by default
    end

    @testset "the physical DC side under the GFL (section 43)" begin
        dcd = (; H = 5e-3, battery = :sossan, soc = "40-50", mode = :continuous)
        cs, x0, u, rep = phs_controlled(eqg; loads = :series_all, converter_Ts = Ts, gfl_advance = 1.0, converter_dc = dcd)
        @test all(g -> g.dc !== nothing && nstates(g) == 21 && PHPS._dcidx(g) == 18, cs.conv)
        @test rep.residual < 1e-12 && all(c -> c.bridge < 1e-12 && c.m0 < 2 / sqrt(3), rep.converters)
        @test_throws ArgumentError phs_controlled(eqg; loads = :series_all, converter_Ts = Ts, converter_dc = merge(dcd, (; mode = :held)))
        # at rest: every DC and branch rate zero, the bridge applying E0, the battery holding the bridge power
        dx = phs_rhs(cs, x0, u)
        upd = phs_control_inputs(cs, x0, u.gf)
        f = PHPS.phs_flows(cs.s, x0, upd)
        for (i, g) in enumerate(cs.conv)
            o = cs.coff[i]
            @test maximum(abs, dx[o+17:o+20]) < 1e-12
            xc = view(x0, o:o+20)
            ic = PHPS._conv_current(f, cs.s.conv[i])
            q = converter_dc_powers(g, xc, ic, cs.s.omega_b)
            @test abs(q.port - q.loss) < 1e-12 * q.tport
            @test q.P_x ≈ real(gfl_bridge(g, xc, ic) * conj(ic))
        end
        # a clipped state: the DC voltage lowered until the static generator's EMF exceeds m_max v_dc; the bridge applies the limit,
        # the bridge power is the same on both sides, the DC storage rate equals the port less the losses
        g, o = cs.conv[2], cs.coff[2]
        xc = collect(x0[o:o+20])
        ic = PHPS._conv_current(f, cs.s.conv[2])
        xc[18] = 0.8 * abs(gfl_emf(g, xc, ic)) / g.dc.m_max
        ux = gfl_bridge(g, xc, ic)
        @test abs(ux) ≈ g.dc.m_max * xc[18] && abs(PHPS._modulation(g, xc, ic)) ≈ g.dc.m_max
        @test angle(ux) ≈ angle(gfl_emf(g, xc, ic))
        q = converter_dc_powers(g, xc, ic, cs.s.omega_b)
        @test q.P_x ≈ real(ux * conj(ic))
        dxc = zeros(21)
        PHPS._dc_rows!(dxc, g, xc, ic, cs.s.omega_b)
        @test PHPS._dc_energy_rate(g, xc, dxc, cs.s.omega_b) ≈ q.port - q.loss rtol = 1e-12
        # unclipped, the AC side is the ideal source's: a short fault run with and without the DC side, the AC states equal to the
        # solver's tolerance (G8 item 4)
        csi, xi, ui, _ = phs_controlled(eqg; loads = :series_all, converter_Ts = Ts, gfl_advance = 1.0)
        ri = phs_simulate(csi, xi, ui; tstop = 40Ts, h = Ts / 4, save_dt = Ts, faults = [PHFault(16, 0.001, 5Ts, 30Ts)], tol = 1e-13)
        rp = phs_simulate(cs, x0, u; tstop = 40Ts, h = Ts / 4, save_dt = Ts, faults = [PHFault(16, 0.001, 5Ts, 30Ts)], tol = 1e-13)
        nac = length(xi)                                   # the ideal-source model's states: the AC side and the controllers
        keep = vcat([collect(csi.coff[i]:csi.coff[i]+16) for i in eachindex(csi.conv)]...)
        ac = setdiff(1:nac, keep)
        @test maximum(abs, ri.X[ac, :] .- rp.X[ac, :]) < 1e-8
        sp = phs_converter_signals(cs, rp.X[:, end], u)
        @test all(sp.mreq .< 2 / sqrt(3)) && all(isfinite, sp.vdc)
        # the discrete balance with the cores
        csc, xcc, uc_, _ = phs_controlled(eqg; loads = :series_all, converter_Ts = Ts, gfl_advance = 1.0, converter_dc = dcd, cores)
        r = phs_simulate(csc, xcc, uc_; tstop = 4Ts, h = Ts / 4, save_dt = Ts, faults = [PHFault(16, 0.001, Ts, 3Ts)], tol = 1e-14)
        # (as for the VOC: the battery's slow branch stores hundreds of pu s against a throughput of ~1e-3 here, so its energy
        # difference is exact only to a few ulps of the stored energy; bus 34's DC side measured 2.5e-13)
        @test all(k -> r.scheme_max[k] <= 1e-10 * r.throughput[k] + 16 * eps(maximum(abs, r.energy[k, :])) + 1e-14, eachindex(r.names))
    end

    @testset "the static generator's EMF (TechRef 7.1, eqs. 81-85)" begin
        cs, x0, u, rep = phs_controlled(eqg; loads = :series_all, converter_Ts = Ts)
        g = cs.conv[1]
        xc = collect(x0[cs.coff[1]:cs.coff[1]+16])
        ch = cis(0.4)
        xc[11], xc[12], xc[13], xc[14], xc[15], xc[16] = real(ch), imag(ch), 5.1, -1.9, 0.97, 0.03
        ic = complex(4.8, -2.6)
        # with Kd = Kq: u_c = K i*_h - (K - j x1) i + z c_h, i*_h = i*_r c_h (record section 38)
        @test g.Kd == g.Kq
        @test gfl_emf(g, xc, ic) ≈ g.Kd * complex(5.1, -1.9) * ch - (g.Kd - im * g.x1) * ic + complex(0.97, 0.03) * ch
        dx = gfl_rates!(zeros(17), g, xc, ic, cs.s.omega_b)
        @test complex(dx[15], dx[16]) ≈ g.Kd / g.Td * (complex(5.1, -1.9) - ic * conj(ch))
    end

    @testset "short fault runs: discrete balance (without and with the cores)" begin
        for cr in (nothing, cores)
            cs, x0, u, rep = phs_controlled(eqg; loads = :series_all, converter_Ts = Ts, cores = cr)
            r = phs_simulate(cs, x0, u; tstop = 4Ts, h = Ts / 4, save_dt = Ts, faults = [PHFault(16, 0.001, Ts, 3Ts)], tol = 1e-14)
            @test all(r.scheme_max .<= 1e-10 .* r.throughput .+ 1e-14)
            sg = phs_converter_signals(cs, r.X[:, end], u)
            @test all(isfinite, sg.omega) && all(isfinite, sg.uc) && all(sg.uc .> 0.5)
        end
    end
end
