# The grid-forming virtual-oscillator (VOC) controller on the EMT model (docs/active/PH_CONVERTER_MODEL.md, sections 28-32;
# src/phsim/converter.jl, PHVoc): PowerFactory's GFM_VOC placement, the rest point, the combined units, a call against the
# DLL's formulas written out here, the DC-rejection filter (the DLL variant gfm_voc_pf_dcr) and the discrete balance of
# short fault runs, without and with the saturating cores; part A, the measured battery, DC link and bridge under it.

@testset "the VOC controller (sections 28-32)" begin
    scv = load_scenario(joinpath(ROOT, "cases", "IEEE39Bus_PF_gfm-voc-pf", "ph_fault_bus16.json"))
    eqv = solve_equilibrium(load_case(scv.system_path), scv)
    Ts = 5e-5
    cores = Dict(33 => 800.0, 36 => 700.0, 37 => 700.0, 38 => 1000.0)

    @testset "rest point, combined units, refusals" begin
        cs, x0, u, rep = phs_controlled(eqv; loads = :series_all, converter_Ts = Ts)
        @test [cv.bus for cv in cs.s.conv] == [31, 33, 34, 36, 37, 38] && all(g -> g isa PHVoc && nstates(g) == 4, cs.conv)
        @test rep.residual < 1e-12 && all(c -> !c.engaged && c.divider < 1e-9 && c.emf_vs_terminal < 1e-12, rep.converters)
        @test all(c -> c.rest_rate < 1e-12, rep.converters)           # V_nom = |v0| and i* = i: dv = 0
        # V_nom from the load flow equals the study case's (record section 29) to 1e-6
        @test [c.V_nom for c in rep.converters] ≈ [1.013676, 1.014076, 1.042951, 1.079714, 1.030598, 1.031898] atol = 1e-6
        # bus 34 is two units: the gains per unit of the combined rating
        g34, p34 = cs.conv[3], cs.s.conv[3].comp.p
        @test g34.eta ≈ p34.eta / 2 && g34.Zs ≈ p34.Zseries / 2 && g34.i_lim ≈ 2 * p34.i_lim && g34.kpx ≈ p34.kpx / 4
        @test g34.p_set ≈ 2 * p34.p_set rtol = 1e-6
        # in the study case's per unit (converter base, eta 20 on each rating): eta * rating / S_b = 20
        @test all(i -> cs.conv[i].eta * (0.1 / cs.conv[i].Zs) ≈ 20, eachindex(cs.conv))
        @test_throws ArgumentError phs_controlled(eqv; loads = :series_all)                      # sampled only
        @test_throws ArgumentError phs_controlled(eqv; loads = :series_all, converter_Ts = Ts, converter_umax = 1.15)
        # one call at rest: only the DLL's rotation at omega_n against the frame's omega_b
        xa = PHPS._sample!(cs, copy(x0), u)
        for (i, g) in enumerate(cs.conv)
            o = cs.coff[i]
            v0 = complex(x0[o], x0[o+1])
            @test complex(xa[o], xa[o+1]) ≈ cis((g.omega_n - cs.s.omega_b) * Ts) * v0 atol = 1e-12
            @test complex(xa[o+2], xa[o+3]) ≈ u.uc[i] atol = 1e-12
        end
    end

    @testset "a call, against the DLL's formulas (section 28)" begin
        cs, x0, u, rep = phs_controlled(eqv; loads = :series_all, converter_Ts = Ts)
        k, wb = 2, cs.s.omega_b
        g, o = cs.conv[k], cs.coff[k]
        xd = copy(x0)
        xd[o] *= 1.02; xd[o+1] += 0.05                     # the oscillator off its rest
        upd = phs_control_inputs(cs, xd, u.gf)
        fd = PHPS.phs_flows(cs.s, xd, upd)
        vk, ic = PHPS._conv_vk(cs.s, cs.s.conv[k], xd, fd), PHPS._conv_current(fd, cs.s.conv[k])
        v = complex(xd[o], xd[o+1])
        # current_ref.h: i* = (v_a P + v_b Q, v_b P - v_a Q)/|u|^2 on the terminal voltage
        ir = complex(real(vk) * g.p_set + imag(vk) * g.q_set, imag(vk) * g.p_set - real(vk) * g.q_set) / abs2(vk)
        e = ir - ic
        Je = complex(cos(angle(g.J)) * real(e) - sin(angle(g.J)) * imag(e), sin(angle(g.J)) * real(e) + cos(angle(g.J)) * imag(e))
        dv = g.xi * (g.V_nom^2 - abs2(v)) * v + g.eta * Je
        xa = PHPS._sample!(cs, copy(xd), u)
        @test complex(xa[o], xa[o+1]) ≈ cis((g.omega_n - wb) * Ts) * (v + Ts * dv) rtol = 1e-12
        # the held output: the divider on v at the measured voltage (below i_lim: u_c = v, r_vi = x_vi = 0)
        d = vsm_divider(g, v, 0.0, vk)
        @test complex(xa[o+2], xa[o+3]) ≈ d.uc && (d.on || d.uc ≈ v)
        # the guard: below |u| = 1e-3 the reference is zero
        @test voc_rates(g, view(xd, o:o+3), 1e-4 + 0im, ic).iref == 0
        # above i_lim the virtual impedance engages and the output leaves v (kpr, kpx > 0)
        dd = vsm_divider(g, v, 0.0, v - im * g.Zs * 1.5 * g.i_lim * cis(0.4))
        @test dd.on && abs(dd.ip) < 1.5 * g.i_lim && abs(dd.uc - v) > 0
    end

    @testset "the DC rejection of the oscillator's current (section 30)" begin
        cs, x0, u, rep = phs_controlled(eqv; loads = :series_all, converter_Ts = Ts, voc_dc_reject = true)
        @test all(g -> g.dcr && nstates(g) == 7, cs.conv) && all(c -> c.rest_rate < 1e-12, rep.converters)
        g = cs.conv[2]
        H(f) = (zeta = cis(2pi * f * Ts); g.dcc * (1 - g.dg / (zeta - g.dlam)))
        @test abs(H(0) - 1) < 1e-14 && abs(H(-g.omega_n / 2pi)) < 1e-12 && abs(g.dlam) < 1
        # a current that is the operating current plus a DC in the phases (in the frame of v, turning at omega_n, a
        # component at -omega_n): the current the oscillator reads converges to the operating current
        o = cs.coff[2]
        xc = collect(x0[o:o+6])
        v = complex(xc[1], xc[2])
        i0 = PHPS._conv_current(PHPS.phs_flows(cs.s, x0, phs_control_inputs(cs, x0, u.gf)), cs.s.conv[2])
        vk = PHPS._conv_vk(cs.s, cs.s.conv[2], x0, PHPS.phs_flows(cs.s, x0, phs_control_inputs(cs, x0, u.gf)))
        local r
        for kk in 0:3000
            iab = i0 + 0.3 * cis(-g.omega_n * kk * Ts) * cis(0.7)   # a DC in alpha-beta seen in a frame turning at omega_n
            xc[1], xc[2] = real(v), imag(v)                           # v held (the oscillator frozen for this test)
            r = voc_rates(g, xc, vk, iab)
            V = g.dlam * complex(xc[5], xc[6]) + r.Z
            xc[5], xc[6] = real(V), imag(V)
        end
        @test abs(r.ifilt - i0) < 1e-6
        # a call: the filter's update V <- lam V + Z, Z the current in the frame of v with the cached |v|; the cache updated
        xd = copy(x0); xd[o] *= 1.03; xd[o+6] = 0.9 * abs(complex(x0[o], x0[o+1]))
        upd = phs_control_inputs(cs, xd, u.gf)
        fd = PHPS.phs_flows(cs.s, xd, upd)
        ic = PHPS._conv_current(fd, cs.s.conv[2])
        vd = complex(xd[o], xd[o+1])
        Z = ic * conj(vd) / xd[o+6]
        xa = PHPS._sample!(cs, copy(xd), u)
        @test complex(xa[o+4], xa[o+5]) ≈ g.dlam * complex(xd[o+4], xd[o+5]) + Z
        @test xa[o+6] ≈ abs(vd)
        # the oscillator reads c (Z - g V) v/|v|cached
        ifil = g.dcc * (Z - g.dg * complex(xd[o+4], xd[o+5])) * vd / xd[o+6]
        @test voc_rates(g, view(xd, o:o+6), PHPS._conv_vk(cs.s, cs.s.conv[2], xd, fd), ic).ifilt ≈ ifil
    end

    @testset "short fault runs: discrete balance (without and with the cores)" begin
        for (dcr, cr) in ((false, nothing), (true, nothing), (true, cores))
            cs, x0, u, rep = phs_controlled(eqv; loads = :series_all, converter_Ts = Ts, voc_dc_reject = dcr, cores = cr)
            r = phs_simulate(cs, x0, u; tstop = 4Ts, h = Ts / 4, save_dt = Ts, faults = [PHFault(16, 0.001, Ts, 3Ts)], tol = 1e-14)
            @test all(r.scheme_max .<= 1e-10 .* r.throughput .+ 1e-14)
            sg = phs_converter_signals(cs, r.X[:, end], u)
            @test all(isfinite, sg.omega) && all(isfinite, sg.umag) && all(sg.umag .> 0.5)
        end
    end

    @testset "part A: the measured battery, DC link and bridge under the VOC (section 32)" begin
        dcd = (; H = 5e-3, battery = :sossan, soc = "40-50", mode = :held)
        cs, x0, u, rep = phs_controlled(eqv; loads = :series_all, converter_Ts = Ts, voc_dc_reject = true, converter_dc = dcd)
        @test length(PHPS._dc_converters(cs)) == 6 && all(g -> nstates(g) == 7 + 4, cs.conv)
        g, o = cs.conv[2], cs.coff[2]
        q, j = g.dc, PHPS._dcidx(g)
        # the data: Table I's 40-50 % column per unit of the rating on the DC base of the 300 V side, then the system base
        Vb, Zb = 2 * sqrt(2 / 3) * 300, (2 * sqrt(2 / 3) * 300)^2 / 720e3
        @test Vb ≈ 489.898 atol = 1e-3
        @test Zb ≈ 1 / 3
        @test q.Eb ≈ 652.9 / Vb && q.Rb * q.Sr ≈ 0.015 / Zb && q.R[1] * q.Sr ≈ 0.090 / Zb && q.R[3] * q.Sr ≈ 2.4e-4 / Zb
        @test [q.R[k] * q.cb[k] / cs.s.omega_b for k in 1:3] ≈ [0.090 * 13996, 0.009 * 2482, 2.4e-4 * 2959.7]   # tau_k (s)
        @test q.Sr ≈ 0.1 / cs.s.conv[2].Lc && q.c * q.v_dc0^2 / (2 * cs.s.omega_b) ≈ 5e-3 * q.Sr
        # the operating point: the battery holds the bridge power with every branch at its DC equilibrium
        ic0 = PHPS._conv_current(PHPS.phs_flows(cs.s, x0, phs_control_inputs(cs, x0, u.gf)), cs.s.conv[2])
        Px = real(u.uc[2] * conj(ic0))
        ib = PHPS._battery_current(g, view(x0, o:o+nstates(g)-1))
        @test ib * x0[o-1+j] ≈ Px rtol = 1e-12
        @test [x0[o-1+j+k] for k in 1:3] ≈ q.R .* ib
        @test q.Eb - (q.Rb + sum(q.R)) * ib ≈ x0[o-1+j]
        d0 = PHPS.phs_rhs(cs, x0, u)
        @test maximum(abs, d0[o-1+j:o-1+j+3]) < 1e-10
        @test abs(PHPS._bridge_voltage(g, view(x0, o:o+nstates(g)-1)) - u.uc[2]) < 1e-12
        # the energy account at a disturbed state: each DC side's rate = port - loss, along the vector field too
        xd = copy(x0); xd[o-1+j] *= 0.97; xd[o-1+j+3] += 0.02
        L = phs_ledger(cs, xd, u)
        @test maximum(abs, L.rate .- (L.port .- L.loss)) < 1e-10 && abs(L.dH - (L.supply - L.losses)) < 1e-10
        Lf = phs_ledger(cs, xd, u; dx = PHPS.phs_rhs(cs, xd, u))
        kdc = findall(==(:converter_dc), PHPS._ckind(cs))
        @test length(kdc) == 6 && maximum(abs, Lf.rate[kdc] .- (Lf.port[kdc] .- Lf.loss[kdc])) < 1e-10
        @test phs_energy(cs, xd) ≈ L.H
        # a call: the held modulation is the command over the DC voltage sampled at the call, clipped radially
        xa = PHPS._sample!(cs, copy(xd), u)
        upd = phs_control_inputs(cs, xd, u.gf)
        fd = PHPS.phs_flows(cs.s, xd, upd)
        ucmd = vsm_divider(g, complex(xd[o], xd[o+1]), 0.0, PHPS._conv_vk(cs.s, cs.s.conv[2], xd, fd)).uc
        @test complex(xa[o+2], xa[o+3]) ≈ PHPS._radial_clip(ucmd / xd[o-1+j], q.m_max)
        @test xa[o-1+j:o-1+j+3] == xd[o-1+j:o-1+j+3]          # the DC states are not stepped by a call
        # a short fault run: the discrete balance with the cores
        csc, x0c, uc_, _ = phs_controlled(eqv; loads = :series_all, converter_Ts = Ts, voc_dc_reject = true, converter_dc = dcd, cores)
        r = phs_simulate(csc, x0c, uc_; tstop = 4Ts, h = Ts / 4, save_dt = Ts, faults = [PHFault(16, 0.001, Ts, 3Ts)], tol = 1e-14)
        # the slow branch (tau 1260 s) stores ~500 pu s against a throughput of ~1e-3 here: its energy difference is exact only to
        # a few ulps of the stored energy (1-4 ulps measured), so the bound allows that rounding
        @test all(k -> r.scheme_max[k] <= 1e-10 * r.throughput[k] + 16 * eps(maximum(abs, r.energy[k, :])) + 1e-14, eachindex(r.names))
        sg = phs_converter_signals(csc, r.X[:, end], uc_)
        @test all(isfinite, sg.vdc) && all(sg.vdc .> 0.5) && all(sg.m .<= q.m_max + 1e-12)
        @test_throws ArgumentError phs_controlled(eqv; loads = :series_all, converter_Ts = Ts,
                                                  converter_dc = (; H = 5e-3, battery = :sossan, soc = "50-60"))
    end
end
