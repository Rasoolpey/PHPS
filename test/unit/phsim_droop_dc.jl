# The droop converter on the measured battery, DC link and bridge (docs/active/PH_CONVERTER_MODEL.md, sections 45-46): PowerFactory's
# GFM_droop placement with the DC-rejecting limiter (gfm_droop_pf_dcr), the DC side of the VOC's Part A (Sossan et al. 2016) with the
# modulation held at the calls: the rest, a call's held modulation, a clipped state's energy identity, the continuous mode against the
# ideal source, and the discrete balance with the cores.

@testset "the droop's physical DC side (section 45)" begin
    scd = load_scenario(joinpath(ROOT, "cases", "IEEE39Bus_PF_gfm-droop-pf", "ph_fault_bus16.json"))
    eqd = solve_equilibrium(load_case(scd.system_path), scd)
    Ts = 5e-5
    cores = Dict(33 => 800.0, 36 => 700.0, 37 => 700.0, 38 => 1000.0)
    dcd = (; H = 5e-3, battery = :sossan, soc = "40-50", mode = :held)
    cs, x0, u, rep = phs_controlled(eqd; loads = :series_all, converter_Ts = Ts, droop_dc_reject = true, converter_dc = dcd)
    @test all(g -> g isa PHDroop && g.dc !== nothing && nstates(g) == 18 && PHPS._dcidx(g) == 15, cs.conv)
    @test state_names(cs.conv[1])[9:10] == ["m_Re", "m_Im"] && state_names(cs.conv[1])[15] == "v_dc"
    @test rep.residual < 1e-12 && all(c -> c.bridge < 1e-12 && c.m0 < 2 / sqrt(3), rep.converters)
    @test_throws ArgumentError phs_controlled(eqd; loads = :series_all, droop_dc_reject = true, converter_dc = dcd)   # sampled only

    @testset "the rest and a call" begin
        dx = phs_rhs(cs, x0, u)
        upd = phs_control_inputs(cs, x0, u.gf)
        f = PHPS.phs_flows(cs.s, x0, upd)
        for (i, g) in enumerate(cs.conv)
            o = cs.coff[i]
            @test maximum(abs, dx[o+14:o+17]) < 1e-12                      # the DC link and the battery's branches at rest
            @test all(==(0), dx[o:o+7]) && all(==(0), dx[o+10:o+13])       # the controller's states held between calls
            @test complex(dx[o+8], dx[o+9]) ≈ -im * cs.s.omega_b * complex(x0[o+8], x0[o+9]) atol = 1e-12   # held in alpha-beta
            xc = view(x0, o:o+17)
            ic = PHPS._conv_current(f, cs.s.conv[i])
            q = converter_dc_powers(g, xc, ic, cs.s.omega_b)
            @test abs(q.port - q.loss) < 1e-12 * q.tport
            @test q.P_x ≈ real(PHPS._bridge_voltage(g, xc) * conj(ic))
        end
        # a call: the held modulation is the controller's command over the DC voltage sampled at the call (clipped radially); the
        # DC states are not stepped by a call
        g, o = cs.conv[2], cs.coff[2]
        xd = copy(x0)
        xd[o+14] *= 0.97                                     # the DC voltage off its rest
        upd2 = phs_control_inputs(cs, xd, u.gf)
        f2 = PHPS.phs_flows(cs.s, xd, upd2)
        ucmd = droop_output(g, view(xd, o:o+17)).uc
        xa = PHPS._sample!(cs, copy(xd), u)
        @test complex(xa[o+8], xa[o+9]) ≈ PHPS._radial_clip(ucmd / xd[o+14], g.dc.m_max)
        @test xa[o+14:o+17] == xd[o+14:o+17]
    end

    @testset "a clipped state: the energy identity" begin
        g, o = cs.conv[3], cs.coff[3]
        upd = phs_control_inputs(cs, x0, u.gf)
        f = PHPS.phs_flows(cs.s, x0, upd)
        ic = PHPS._conv_current(f, cs.s.conv[3])
        xc = collect(x0[o:o+17])
        xc[15] = 0.8 * abs(droop_output(g, xc).uc) / g.dc.m_max
        h = PHPS._held_from(g, droop_output(g, xc).uc, xc)
        @test abs(h) ≈ g.dc.m_max                            # the held modulation at its limit
        xc[9], xc[10] = real(h), imag(h)
        q = converter_dc_powers(g, xc, ic, cs.s.omega_b)
        @test q.P_x ≈ real(PHPS._bridge_voltage(g, xc) * conj(ic))
        dxc = zeros(18)
        PHPS._dc_rows!(dxc, g, xc, ic, cs.s.omega_b)
        @test PHPS._dc_energy_rate(g, xc, dxc, cs.s.omega_b) ≈ q.port - q.loss rtol = 1e-12
    end

    @testset "the continuous mode against the ideal source (G10 item 1)" begin
        csc, xc0, uc0, _ = phs_controlled(eqd; loads = :series_all, converter_Ts = Ts, droop_dc_reject = true,
                                          converter_dc = merge(dcd, (; mode = :continuous)))
        csi, xi, ui, _ = phs_controlled(eqd; loads = :series_all, converter_Ts = Ts, droop_dc_reject = true)
        ri = phs_simulate(csi, xi, ui; tstop = 40Ts, h = Ts / 4, save_dt = Ts, faults = [PHFault(16, 0.001, 5Ts, 30Ts)], tol = 1e-13)
        rc = phs_simulate(csc, xc0, uc0; tstop = 40Ts, h = Ts / 4, save_dt = Ts, faults = [PHFault(16, 0.001, 5Ts, 30Ts)], tol = 1e-13)
        keep = vcat([collect(csi.coff[i]:csi.coff[i]+nstates(csi.conv[i])-1) for i in eachindex(csi.conv)]...)
        ac = setdiff(1:length(xi), keep)
        @test maximum(abs, ri.X[ac, :] .- rc.X[ac, :]) < 1e-8
        sc_ = phs_converter_signals(csc, rc.X[:, end], uc0)
        @test all(sc_.m .< 2 / sqrt(3))                      # unclipped throughout this short run
    end

    @testset "short fault runs: the discrete balance (held, with the cores)" begin
        csk, xk, uk, _ = phs_controlled(eqd; loads = :series_all, converter_Ts = Ts, droop_dc_reject = true, converter_dc = dcd, cores)
        r = phs_simulate(csk, xk, uk; tstop = 4Ts, h = Ts / 4, save_dt = Ts, faults = [PHFault(16, 0.001, Ts, 3Ts)], tol = 1e-14)
        # (the battery's slow branch: a few ulps of its stored energy, as for the VOC and the GFL)
        @test all(k -> r.scheme_max[k] <= 1e-10 * r.throughput[k] + 16 * eps(maximum(abs, r.energy[k, :])) + 1e-14, eachindex(r.names))
        sg = phs_converter_signals(csk, r.X[:, end], uk)
        @test all(isfinite, sg.vdc) && all(sg.vdc .> 0.5) && all(sg.m .<= 2 / sqrt(3) + 1e-12)
    end
end
