# Port-Hamiltonian grid model (docs/active/PH_SIMULATOR_SPEC.md) on IEEE-39: assembly and its
# reductions, the equilibrium, the structure identities of spec section 8 at random states, the
# regression against the existing balanced code (network rows against `pn_flows`, machine rows
# against the pilot's `emdq_rhs` through the change of coordinates), and a short fault run with
# its discrete energy balance, order and an independent BDF solution.

@testset "port-Hamiltonian IEEE-39 model" begin
    sc = load_scenario(joinpath(ROOT, "cases", "IEEE39Bus_PF", "no_fault_phtrue.json"))
    eq = solve_equilibrium(load_case(sc.system_path), sc)
    s, x0, u, rep = phs_equilibrium(eq)
    pn = s.net
    wb = s.omega_b
    ids = pn.net.bus_ids
    cx = PHPS._cx

    @testset "assembly and reductions" begin
        @test s.n == 247 == length(phs_state_names(s))
        @test length(s.dyn) == 28
        @test ids[s.alg] == [12, 20, 31]
        @test sort(ids[s.merged]) == [30, 32, 33, 34, 35, 36, 37, 38]
        @test length(s.branches) == 46 - 8
        @test length(s.machines) == 10
        twin = s.machines[findfirst(m -> m.units == 2, s.machines)]
        @test twin.name == "GENROU_5+GENROU_6" && twin.trf == "Trf_8" && ids[twin.node] == 20
        # n identical units in parallel: flux linkages equal, currents and energy add
        p = eq.sys.comps[findfirst(c -> PHPS.name(c) == "GENROU_5", eq.sys.comps)].p
        w1, w2 = winding_machine(p), winding_machine(PHPS._parallel(p, 2))
        @test w2.Kd ≈ 2 .* w1.Kd && w2.Kq ≈ 2 .* w1.Kq
        @test w2.H == 2 * w1.H && w2.Rfd ≈ w1.Rfd / 2 && w2.R1q ≈ w1.R1q / 2
        @test sum(s.Cown) + sum(s.charge) ≈ sum(pn.C)
    end

    @testset "equilibrium" begin
        @test rep.residual < 1e-12
        @test rep.dV < 1e-8 && rep.dVgen < 1e-8        # the RMS (and PowerFactory) load flow
        @test all(>(0), rep.vf) && all(>(0), rep.Tm)
        twin = findfirst(m -> m.units == 2, s.machines)
        @test rep.Tm[twin] ≈ 5.08 rtol = 1e-6            # two units of 2.54 pu
        @test all(m -> x0[PHPS._ip(m)] == 2 * m.w.H, s.machines)
    end

    rng = MersenneTwister(5)
    scl = phs_scales(s, x0)
    draw() = (x0 .+ 0.2 .* scl .* randn(rng, s.n),
              PHInput(u.Tm .* (1 .+ 0.3 .* randn(rng, 10)), u.vf .* (1 .+ 0.3 .* randn(rng, 10)),
                      zeros(length(u.gf))))

    @testset "structure identities (R3, R4, R7, R9)" begin
        for trial in 1:5
            y, uu = draw()
            trial > 2 && (uu.gf[rand(rng, vcat(s.dyn, s.alg))] = 800.0)
            f, M, L = phs_rhs(s, y, uu), phs_matrices(s, y, uu), phs_ledger(s, y, uu)
            den = abs.(M.J) * abs.(M.z) .+ abs.(M.R) * abs.(M.z) .+ abs.(M.g)
            @test maximum(abs.(f .- ((M.J - M.R) * M.z .+ M.g)) ./ den) < 1e-12
            @test M.J == -M.J'
            @test maximum(abs, M.R - M.R') < 1e-14 * opnorm(M.R)
            @test eigmin(Symmetric(M.R)) > -1e-12 * opnorm(M.R)
            @test M.z ≈ ForwardDiff.gradient(w -> phs_energy(s, w), y) rtol = 1e-12
            @test all(M.g[PHPS._idelta(m)] == -wb for m in s.machines)      # the frame clock
            bal = abs.(L.rate .- (L.port .- L.loss)) ./ (abs.(L.rate) .+ abs.(L.port) .+ abs.(L.loss) .+ floatmin())
            @test maximum(bal) < 1e-12
            @test abs(L.interconnection) < 1e-12 * sum(abs, L.port)
            @test abs(sum(L.port) - L.supply) < 1e-12 * sum(L.tport)   # the ports themselves cancel
            @test all(L.tport .>= abs.(L.port) .- 1e-12 .* L.tport)
            @test abs(L.dH - (L.supply - L.losses)) < 1e-12 * (abs(L.supply) + L.losses)
            @test L.H ≈ phs_energy(s, y) rtol = 1e-13
            @test L.dH ≈ dot(M.z, f) rtol = 1e-10
        end
    end

    @testset "network rows equal pn_flows" begin
        y, uu = draw()
        f, fl = phs_rhs(s, y, uu), phs_flows(s, y, uu)
        nb, mb, _ = pn_sizes(pn)
        v = copy(fl.v)
        v[s.merged] .= 0
        i, iext = zeros(ComplexF64, mb), zeros(ComplexF64, nb)
        for (j, l) in enumerate(s.branches)
            i[l] = wb * cx(y, s.bidx[j]) / pn.L[l]
        end
        for (j, m) in enumerate(s.machines)
            if m.trf == ""
                iext[m.node] -= fl.mc[j].inet
            else                                   # the merged transformer carries the stator current
                i[findfirst(==(m.trf), pn.branch)] = fl.mc[j].inet
                iext[bus_index(pn.net, m.bus)] -= fl.mc[j].inet
            end
        end
        ish = [wb * cx(y, s.sidx[j]) / pn.Lsh[j] for j in eachindex(pn.Lsh)]
        ref = pn_flows(pn, v, i, ish, iext)
        @test all(k -> isapprox(ref.fv[k], cx(f, s.qidx[k]); rtol = 1e-11, atol = 1e-13), s.dyn)
        @test maximum(abs, ref.fv[s.alg]) < 1e-11         # the eliminated rows: KCL holds
        @test maximum(abs, ref.fv[s.merged]) < 1e-12      # the merged rows: stator current = transformer current
        @test all(j -> isapprox(ref.fi[s.branches[j]], cx(f, s.bidx[j]); rtol = 1e-11, atol = 1e-13),
                  eachindex(s.branches))
        @test all(j -> isapprox(ref.fs[j], cx(f, s.sidx[j]); rtol = 1e-11, atol = 1e-13), eachindex(ish))
    end

    @testset "machine rows equal the pilot through the change of coordinates" begin
        # the pilot stores the stator flux in the network frame. One salient-pole machine (the
        # record named GENROU_11 is a GENSAL) and one round rotor with its second q winding.
        for (mname, kw) in (("GENROU_11", (;)),
                            ("GENROU_7", (; machine = "GENROU_7", trf = "Trf_9", line = "Line_28", tie = "Line_29")))
            pilot, xs, _ = emdq_pilot(eq; kw...)
            j = findfirst(m -> m.name == mname, s.machines)
            m = s.machines[j]
            @test m.w.Kd == pilot.mach.Kd && m.w.Kq == pilot.mach.Kq && m.a == pilot.tap
            @test m.w.nq == (mname == "GENROU_7" ? 2 : 1)
            for _ in 1:4
                xp = xs .+ 0.05 .* randn(rng, length(xs)) .* max.(abs.(xs), 0.1)
                mod = 0.2 + 0.6 * rand(rng)
                dxp = emdq_rhs(pilot, xp, mod)
                y, uu = draw()
                k2 = m.node
                q2 = pn.C[k2] * complex(xp[9], xp[10]) / wb
                y[s.qidx[k2]], y[s.qidx[k2]+1] = real(q2), imag(q2)
                delta = xp[6]
                rot = im * complex(cos(delta), -sin(delta))
                psic = rot * complex(xp[1], xp[2])
                o, ip, id = m.offset, PHPS._ip(m), PHPS._idelta(m)
                y[o], y[o+1] = real(psic) / wb, imag(psic) / wb
                y[o+2], y[o+3], y[o+4] = xp[3] / wb, xp[4] / wb, xp[5] / wb
                m.w.nq == 2 && (y[o+5] = xp[19] / wb)
                y[ip], y[id] = xp[7], delta
                uu.Tm[j], uu.vf[j] = pilot.Tm, mod * xp[8] / pilot.Cdc
                f = phs_rhs(s, y, uu)
                dpsic = rot * complex(dxp[1], dxp[2]) - im * dxp[6] * psic
                @test complex(f[o], f[o+1]) ≈ dpsic / wb rtol = 1e-11
                @test f[o+2:o+4] ≈ dxp[3:5] ./ wb rtol = 1e-11
                m.w.nq == 2 && @test f[o+5] ≈ dxp[19] / wb rtol = 1e-11
                @test f[ip] ≈ dxp[7] rtol = 1e-11 atol = 1e-12
                @test f[id] ≈ dxp[6] rtol = 1e-11 atol = 1e-12
            end
        end
    end

    @testset "recovered generator-bus voltage satisfies the unreduced stator equation" begin
        # away from equilibrium: the machine's own stator flux (without the merged leakage), with
        # the recovered voltage as its terminal voltage, obeys the stator equation of the
        # unreduced model, and the transformer carries the same current (the eliminated KCL row)
        y, uu = draw()
        f, fl, vg = phs_rhs(s, y, uu), phs_flows(s, y, uu), phs_terminal_voltage(s, y, uu)
        @test sort(collect(keys(vg))) == sort(ids[s.merged])
        for (j, m) in enumerate(s.machines)
            m.trf == "" && continue
            w, c, o = m.w, fl.mc[j], m.offset
            K, Q = w.Kd, w.Kq
            did = wb * (K[1, 1] * f[o] + K[1, 2] * f[o+2] + K[1, 3] * f[o+3])
            diq = wb * (Q[1, 1] * f[o+1] + Q[1, 2] * f[o+4] + (w.nq == 2 ? Q[1, 3] * f[o+5] : 0.0))
            lam_own = complex(y[o], y[o+1]) - m.x_ext * complex(c.id, c.iq) / wb
            dlam_own = complex(f[o], f[o+1]) - m.x_ext * complex(did, diq) / wb
            vdq = im * conj(c.rot) * vg[m.bus]
            rhs_own = vdq - (w.Ra - m.r_ext) * complex(c.id, c.iq) - im * wb * c.omega * lam_own
            @test dlam_own ≈ rhs_own rtol = 1e-10 atol = 1e-12
        end
    end

    @testset "the choices made to simulate what PowerFactory EMT simulates" begin
        # winding data given directly (here: the model's own, so the result must not change),
        # loads as series R-L where the bus has capacitance, turbine power held constant
        circ = Dict(m.bus => (; Ll = m.w.Ll - m.x_ext, Lad = m.w.Lad, Laq = m.w.Laq, Lfd = m.w.Lfd,
                              L1d = m.w.L1d, L1q = m.w.L1q, L2q = m.w.nq == 2 ? m.w.L2q : nothing,
                              Ra = m.w.Ra - m.r_ext, Rfd = m.w.Rfd, R1d = m.w.R1d, R1q = m.w.R1q,
                              R2q = m.w.R2q, H = m.w.H) for m in s.machines)
        sc_, xc, uc, repc = phs_equilibrium(eq; circuits = circ)
        @test xc ≈ x0 rtol = 1e-12
        @test all(sc_.machines[j].w.Kd ≈ s.machines[j].w.Kd for j in 1:10)

        s2, x2, u2, rep2 = phs_equilibrium(eq; loads = :series, const_power = true)
        @test rep2.residual < 1e-12 && rep2.dV < 1e-8 && rep2.dVgen < 1e-8   # same operating point
        @test u2.const_power
        @test u2.Tm ≈ u.Tm rtol = 1e-9                     # power = torque at speed 1
        @test count(>(0), s2.Rsh) == 15                    # every inductive load at a bus with capacitance
        @test ids[s2.alg] == [12, 20, 31]                  # the three that stay parallel
        @test all(s2.net.G[k] == 0 for k in s2.dyn if k in s2.net.sh_bus)
        # each series load draws the case's P and Q at the equilibrium voltage
        fl = phs_flows(s2, x2, u2)
        pq = Dict(b => complex(P, Q) for (b, P, Q) in PHPS.load_power_by_bus(eq.sys.case, s2.net.net))
        for (j, k) in enumerate(s2.net.sh_bus)
            s2.Rsh[j] > 0 || continue
            i = wb * cx(x2, s2.sidx[j]) / s2.net.Lsh[j]
            @test fl.v[k] * conj(i) ≈ pq[ids[k]] rtol = 1e-8
        end
        scl2 = phs_scales(s2, x2)
        for _ in 1:3
            y = x2 .+ 0.2 .* scl2 .* randn(rng, s2.n)
            uu = PHInput(u2.Tm .* (1 .+ 0.3 .* randn(rng, 10)), u2.vf .* (1 .+ 0.3 .* randn(rng, 10)),
                         zeros(length(u2.gf)), true)
            f, M, L = phs_rhs(s2, y, uu), phs_matrices(s2, y, uu), phs_ledger(s2, y, uu)
            den = abs.(M.J) * abs.(M.z) .+ abs.(M.R) * abs.(M.z) .+ abs.(M.g)
            @test maximum(abs.(f .- ((M.J - M.R) * M.z .+ M.g)) ./ den) < 1e-12
            @test M.J == -M.J' && eigmin(Symmetric(M.R)) > -1e-12 * opnorm(M.R)
            bal = abs.(L.rate .- (L.port .- L.loss)) ./ (abs.(L.rate) .+ abs.(L.port) .+ abs.(L.loss) .+ floatmin())
            @test maximum(bal) < 1e-12
            @test abs(L.dH - (L.supply - L.losses)) < 1e-12 * (abs(L.supply) + L.losses)
            @test L.supply ≈ sum(uu.Tm) + sum(uu.vf[j] * phs_flows(s2, y, uu).mc[j].ifd for j in 1:10) rtol = 1e-12
            # constant power: the torque on each rotor is the power over its speed
            m1 = s2.machines[1]
            @test M.g[PHPS._ip(m1)] ≈ uu.Tm[1] / (y[PHPS._ip(m1)] / (2 * m1.w.H)) rtol = 1e-14
        end
        # a run this short has almost no throughput, and one step's defect is bounded by the
        # Newton tolerance times the stored momentum; the tolerance is tightened to keep the
        # limit of 1e-10 of throughput meaningful
        r = phs_simulate(s2, x2, u2; tstop = 0.004, faults = [PHFault(16, 0.001, 0.001, 0.002)], grid = 1e-4,
                         tol = 1e-14)
        @test r.t ≈ collect(0:1e-4:0.004) atol = 1e-12       # saved exactly on the grid
        @test all(r.scheme_max .<= 1e-10 .* r.throughput)
        @test_throws ArgumentError phs_simulate(s2, x2, u2; tstop = 0.00405, grid = 1e-4)
        # sampled at arbitrary times: the same trajectory, read off inside the steps
        ts_ = [0.0, 0.00073, 0.001, 0.0010004, 0.0019999, 0.002, 0.0031]
        ra = phs_simulate(s2, x2, u2; tstop = 0.004, faults = [PHFault(16, 0.001, 0.001, 0.002)], sample_at = ts_)
        @test ra.t == ts_ && ra.X[:, 1] == x2
        @test ra.X[:, 3] ≈ r.X[:, 11] rtol = 1e-6       # 0.001 s, a grid point of the other run
        @test ra.X[:, 6] ≈ r.X[:, 21] rtol = 1e-6       # 0.002 s
        @test ra.H ≈ [phs_energy(s2, ra.X[:, k]) for k in eachindex(ts_)] rtol = 1e-13
        @test ra.total.scheme_max < 1e-10 * ra.total.throughput
    end

    @testset "series loads at the buses without capacitance: the inductor-only nodes reduced" begin
        # `loads = :series_all`: buses 12, 20 and 31 keep no voltage and their load inductor no
        # state (van der Schaft 2013, eq. 4.6-4.8; Mayer and Wasynczuk 1991, eq. 33-38)
        s2, x2, u2, _ = phs_equilibrium(eq; loads = :series, const_power = true)
        s3, x3, u3, rep3 = phs_equilibrium(eq; loads = :series_all, const_power = true)
        pn3 = s3.net
        @test s3.n == s2.n - 6 == length(phs_state_names(s3))
        @test isempty(s3.alg) && ids[s3.cut] == [12, 20, 31]
        @test count(>(0), s3.Rsh) == 18 && count(==(0), s3.sidx) == 3
        @test [c.machine == 0 ? "" : s3.machines[c.machine].name for c in s3.cuts] ==
              ["", "GENROU_5+GENROU_6", "GENROU_2"]
        @test [pn3.branch[s3.branches[c.pos]] for c in s3.cuts] == [["Trf_4", "Trf_5"], ["Trf_6"], ["Trf_2"]]
        @test rep3.residual < 1e-12 && rep3.dV < 1e-8 && rep3.dVgen < 1e-8   # the same operating point
        @test u3.Tm ≈ u2.Tm rtol = 1e-9
        @test u3.vf ≈ u2.vf rtol = 1e-9
        # each of the three loads draws the case's P and Q at the recovered voltage
        fl3, vc3 = phs_flows(s3, x3, u3), phs_cut_voltage(s3, x3, u3)
        pq = Dict(b => complex(P, Q) for (b, P, Q) in PHPS.load_power_by_bus(eq.sys.case, pn3.net))
        @test all(vc3[ids[c.node]] * conj(fl3.iload[ci]) ≈ pq[ids[c.node]] for (ci, c) in enumerate(s3.cuts))
        @test maximum(abs, fl3.inj[s3.cut]) < 1e-12        # Kirchhoff's current law there

        scl3 = phs_scales(s3, x3)
        draw3() = (x3 .+ 0.2 .* scl3 .* randn(rng, s3.n),
                   PHInput(u3.Tm .* (1 .+ 0.3 .* randn(rng, 10)), u3.vf .* (1 .+ 0.3 .* randn(rng, 10)),
                           zeros(length(u3.gf)), true))
        for trial in 1:3
            y, uu = draw3()
            trial == 3 && (uu.gf[rand(rng, s3.dyn)] = 800.0)
            f, M, L = phs_rhs(s3, y, uu), phs_matrices(s3, y, uu), phs_ledger(s3, y, uu)
            den = abs.(M.J) * abs.(M.z) .+ abs.(M.R) * abs.(M.z) .+ abs.(M.g)
            @test maximum(abs.(f .- ((M.J - M.R) * M.z .+ M.g)) ./ den) < 1e-12
            @test M.J == -M.J'
            @test maximum(abs, M.R - M.R') < 1e-14 * opnorm(M.R)
            @test eigmin(Symmetric(M.R)) > -1e-12 * opnorm(M.R)
            @test M.z ≈ ForwardDiff.gradient(w -> phs_energy(s3, w), y) rtol = 1e-11
            # the branches and the machine windings at a reduced node balance with the voltage
            # recovered from the load: evidence, not construction
            bal = abs.(L.rate .- (L.port .- L.loss)) ./ (abs.(L.rate) .+ abs.(L.port) .+ abs.(L.loss) .+ floatmin())
            @test maximum(bal) < 1e-12
            @test abs(L.interconnection) < 1e-12 * sum(abs, L.port)
            @test abs(L.dH - (L.supply - L.losses)) < 1e-12 * (abs(L.supply) + L.losses)
            @test L.H ≈ phs_energy(s3, y) rtol = 1e-13
            @test all(>=(0), L.energy) && all(>=(0), L.loss)
        end

        # The unreduced model, element by element, in its own variables: every inductance matrix
        # below is rebuilt here from the network and winding data, and the currents and their
        # rates are solved from the reduced fluxes and the vector field.
        rel(a, b) = abs(a - b) / (abs(a) + abs(b))
        for _ in 1:3
            y, uu = draw3()
            f, fl, vc = phs_rhs(s3, y, uu), phs_flows(s3, y, uu), phs_cut_voltage(s3, y, uu)
            v = copy(fl.v)
            for (b, vb) in vc
                v[findfirst(==(b), ids)] = vb
            end
            branch_rhs(l, i) = v[pn3.from[l]] / pn3.a[l] - v[pn3.to[l]] - pn3.R[l] * i - im * pn3.L[l] * i
            for k in s3.cut
                js = [j for (j, l) in enumerate(s3.branches) if pn3.from[l] == k || pn3.to[l] == k]
                ls = s3.branches[js]
                gam = [pn3.from[l] == k ? 1 / real(pn3.a[l]) : -1.0 for l in ls]
                sh = findfirst(==(k), pn3.sh_bus)
                RL, LL = s3.Rsh[sh], pn3.Lsh[sh]
                mj = findfirst(m -> m.node == k, s3.machines)
                if mj === nothing
                    M = Diagonal(pn3.L[ls]) + LL * gam * gam'
                    i = wb * (M \ [cx(y, s3.bidx[j]) for j in js])
                    di = wb * (M \ [cx(f, s3.bidx[j]) for j in js])
                    @test i ≈ fl.ib[js] rtol = 1e-12
                    @test all(rel(pn3.L[l] * di[a] / wb, branch_rhs(l, i[a])) < 1e-11 for (a, l) in enumerate(ls))
                    @test rel(-LL * dot(gam, di) / wb, v[k] + complex(RL, LL) * dot(gam, i)) < 1e-11   # the load
                else
                    m = s3.machines[mj]
                    w, o, t, l = m.w, m.offset, s3.bidx[js[1]], ls[1]
                    a, g, nq = real(m.a), gam[1], m.w.nq
                    al, be, ka = LL / a^2, g * LL / a, pn3.L[l] + g^2 * LL
                    Ld, Lq = inv(w.Kd), inv(w.Kq)        # the machine's own windings
                    Ldx = [Ld zeros(3); zeros(1, 3) ka]
                    Lqx = [Lq zeros(nq + 1); zeros(1, nq + 1) ka]
                    Ldx[1, 1] += al; Ldx[1, 4] = Ldx[4, 1] = be
                    Lqx[1, 1] += al; Lqx[1, end] = Lqx[end, 1] = be
                    fd(z) = [z[o], z[o+2], z[o+3], z[t]]
                    fq(z) = nq == 2 ? [z[o+1], z[o+4], z[o+5], z[t+1]] : [z[o+1], z[o+4], z[t+1]]
                    id, iq = wb * (Ldx \ fd(y)), wb * (Lqx \ fq(y))
                    did, diq = wb * (Ldx \ fd(f)), wb * (Lqx \ fq(f))
                    c = fl.mc[mj]
                    @test [id[1], iq[1], id[2], id[4], iq[end]] ≈ [c.id, c.iq, c.ifd, c.iTd, c.iTq] rtol = 1e-11
                    delta, ddelta = y[PHPS._idelta(m)], f[PHPS._idelta(m)]
                    omega = y[PHPS._ip(m)] / (2 * w.H)
                    rot = cis(delta)
                    idq, iTdq = complex(id[1], iq[1]), complex(id[4], iq[end])
                    didq, diTdq = complex(did[1], diq[1]), complex(did[4], diq[end])
                    # the machine's own stator flux obeys its own stator equation at the node voltage
                    lam = complex(y[o], y[o+1]) - (al * idq + be * iTdq) / wb
                    dlam = complex(f[o], f[o+1]) - (al * didq + be * diTdq) / wb
                    @test lam ≈ complex((Ld*id[1:3])[1], (Lq*iq[1:nq+1])[1]) / wb rtol = 1e-11
                    @test rel(dlam, im * conj(rot) * v[k] / a - w.Ra * idq - im * wb * omega * lam) < 1e-11
                    ps = phs_machine_signals(s3, y)[mj]
                    @test [ps.psid, ps.psiq] ≈ wb .* [real(lam), imag(lam)] .- m.x_ext .* [id[1], iq[1]] rtol = 1e-11
                    # the branch, in the network frame
                    iT, diT = -im * rot * iTdq, -im * rot * (diTdq + im * ddelta * iTdq)
                    @test iT ≈ fl.ib[js[1]] rtol = 1e-12
                    @test rel(pn3.L[l] * diT / wb, branch_rhs(l, iT)) < 1e-11
                    # the load, by Kirchhoff's current law
                    inet, dinet = -im * rot * idq, -im * rot * (didq + im * ddelta * idq)
                    iL, diL = -g * iT - inet / a, -g * diT - dinet / a
                    @test rel(LL * diL / wb, v[k] - complex(RL, LL) * iL) < 1e-11
                    # the torque on the rotor is the machine's own
                    tq = wb * (real(lam) * iq[1] - imag(lam) * id[1])
                    @test f[PHPS._ip(m)] - uu.Tm[mj] / omega ≈ tq rtol = 1e-10
                    @test ps.torque ≈ tq rtol = 1e-10
                end
            end
        end

        # a fault at a reduced node is refused, as an event and as an input; elsewhere the
        # discrete energy balance holds
        @test_throws ArgumentError phs_simulate(s3, x3, u3; tstop = 0.01, faults = [PHFault(20, 0.001, 0.0, 0.01)])
        ug = PHInput(u3.Tm, u3.vf, zeros(length(u3.gf)), true)
        ug.gf[s3.cut[1]] = 800.0
        @test_throws ArgumentError phs_rhs(s3, x3, ug)
        faults = [PHFault(16, 0.001, 0.001, 0.002)]
        r = phs_simulate(s3, x3, u3; tstop = 0.004, faults, grid = 1e-4, tol = 1e-14)
        @test all(r.scheme_max .<= 1e-10 .* r.throughput)
        @test [phs_energy(s3, r.X[:, k]) for k in eachindex(r.t)] ≈ r.H rtol = 1e-13
        k20 = findfirst(==(20), ids)
        @test all(isfinite, r.V)                    # the three recovered voltages included
        @test abs(r.V[k20, 1]) ≈ abs(rep3.V_rms[k20]) rtol = 1e-8

        # The reduced model is the limit of the circuit with a stray capacitor at the three buses
        # (the unreduced code path: a series load at a bus with a voltage state). An independent
        # solver on both; the difference falls in proportion to the capacitance.
        sig(sx, X) = (q = phs_machine_signals(sx, X);
                      vcat([z.speed for z in q], [z.delta for z in q], [z.psid for z in q], [z.psiq for z in q],
                           [z.ifd for z in q], [z.torque for z in q]))
        _, Xb = phs_simulate_bdf(s3, x3, u3; tstop = 0.004, faults, saveat = [0.004])
        dist = map((1e-3, 1e-4)) do C
            sC, xC, uC, repC = phs_equilibrium(eq; loads = :series_all, const_power = true, stray_capacitance = C)
            @test sC.n == s2.n + 6 && isempty(sC.cut) && isempty(sC.alg) && repC.residual < 1e-12
            _, XC = phs_simulate_bdf(sC, xC, uC; tstop = 0.004, faults, saveat = [0.004])
            maximum(abs.(sig(sC, XC[:, 1]) .- sig(s3, Xb[:, 1])))
        end
        @test dist[2] < 2e-3 && 8 < dist[1] / dist[2] < 12
        # and the midpoint solution of the reduced model converges to that independent solver's:
        # fourfold per halving of the step, the extrapolated value agreeing with it
        qh = [sig(s3, phs_simulate(s3, x3, u3; tstop = 0.004, faults, hscale = hs, save_dt = 0.004).X[:, end])
              for hs in (0.5, 0.25)]
        eh = [maximum(abs.(q .- sig(s3, Xb[:, 1]))) for q in qh]
        @test 3.5 < eh[1] / eh[2] < 4.5
        @test maximum(abs.((4 .* qh[2] .- qh[1]) ./ 3 .- sig(s3, Xb[:, 1]))) < 1e-5

        # the controllers read the recovered voltages
        cs3, xc3, uc3, repc3 = phs_controlled(eq; loads = :series_all)
        @test repc3.residual_controlled < 1e-11 && repc3.input_change < 1e-12
        y = xc3 .+ 0.05 .* phs_scales(cs3, xc3) .* randn(rng, cs3.n)
        fc = phs_rhs(cs3, y, uc3)
        up = phs_control_inputs(cs3, y, uc3.gf)
        vg, vc = phs_terminal_voltage(cs3.s, y[1:s3.n], up), phs_cut_voltage(cs3.s, y[1:s3.n], up)
        for (j, m) in enumerate(cs3.s.machines)
            cs3.exc[j] === nothing && continue
            rr = cs3.xoff[j]:cs3.xoff[j]+4
            vt = abs(haskey(vg, m.bus) ? vg[m.bus] : get(vc, m.bus, phs_flows(cs3.s, y, up).v[m.node]))
            ifd = phs_flows(cs3.s, y, up).mc[j].ifd
            @test fc[rr] ≈ rhs!(zeros(5), cs3.exc[j], y[rr], [vt, cs3.Vref[j], 0.0, m.w.Lad * ifd]) rtol = 1e-9 atol = 1e-9
        end
    end

    @testset "exciters and governors as control laws" begin
        cs, xc, uc, repc = phs_controlled(eq)
        nm = length(cs.s.machines)
        @test cs.s.n == 247 && cs.n == 247 + 9 * 5 + 8 * 7 + 5 == length(phs_state_names(cs))
        @test count(!isnothing, cs.exc) == 9 && count(g -> g isa IEEEG1_PHTRUE, cs.gov) == 8
        @test count(g -> g isa IEEEG3_PHTRUE, cs.gov) == 1
        @test cs.exc[1] === nothing && cs.gov[1] === nothing            # GENROU_1 has no controller
        twin = findfirst(==(2), cs.units)
        @test repc.controllers[twin].exciter == "IEEET1_5" && repc.controllers[twin].governor == "IEEEG1_5"
        # the equilibrium is unchanged by attaching them, and they sit at rest
        @test repc.residual_controlled < 1e-11 && repc.input_change < 1e-12
        @test xc[1:247] ≈ x0 rtol = 1e-12
        sig = phs_control_signals(cs, xc)
        fl = phs_flows(cs.s, xc, uc)
        @test sig.Pm ≈ uc.Tm rtol = 1e-12
        @test sig.Efd ≈ [m.w.Lad * fl.mc[j].ifd for (j, m) in enumerate(cs.s.machines)] rtol = 1e-12
        # with governors the common-speed drift of the uncontrolled model is gone
        @test maximum(real, eigvals(PHPS._jacobian(s, x0, u))) > 5e-3
        @test maximum(real, eigvals(PHPS._jacobian(cs, xc, uc))) < 1e-8

        scl_c = phs_scales(cs, xc)
        for trial in 1:3
            y = xc .+ 0.1 .* scl_c .* randn(rng, cs.n)
            trial == 3 && (y[cs.xoff[2]+1] = 5.0)                       # an exciter regulator beyond its limit
            f = phs_rhs(cs, y, uc)
            up = phs_control_inputs(cs, y, uc.gf)
            # the physical rows are the physical model's, at the inputs the controllers set
            @test f[1:247] ≈ phs_rhs(cs.s, y[1:247], up) rtol = 1e-13
            @test up.const_power
            vg = phs_terminal_voltage(cs.s, y[1:247], up)
            flc = phs_flows(cs.s, y, up)
            for (j, m) in enumerate(cs.s.machines)
                # the controller rows are the unchanged RMS kernels at their machine's signals
                if cs.exc[j] !== nothing
                    r = cs.xoff[j]:cs.xoff[j]+4
                    vt = abs(haskey(vg, m.bus) ? vg[m.bus] : flc.v[m.node])
                    ref = rhs!(zeros(5), cs.exc[j], y[r], [vt, cs.Vref[j], 0.0, m.w.Lad * flc.mc[j].ifd])
                    @test f[r] ≈ ref rtol = 1e-9 atol = 1e-9
                    @test up.vf[j] ≈ y[r[3]] * m.w.Rfd / m.w.Lad rtol = 1e-14
                end
                if cs.gov[j] !== nothing
                    r = cs.goff[j]:cs.goff[j]+nstates(cs.gov[j])-1
                    ref = rhs!(zeros(length(r)), cs.gov[j], y[r], [flc.mc[j].omega, cs.Pref[j], 0.0])
                    @test f[r] ≈ ref rtol = 1e-12 atol = 1e-12
                end
            end
            # limiter decisions recorded at a state and replayed there give the same vector field
            modes = phs_modes(cs, y, uc)
            @test PHPS._frozen_rhs!(zeros(cs.n), cs, y, uc, modes) == f
            trial == 3 && @test any(modes[2, 1])                        # the limit is active
            # the energy account is the physical model's with those inputs
            L = phs_ledger(cs, y, uc)
            @test L.supply ≈ sum(up.Tm) + sum(up.vf[j] * flc.mc[j].ifd for j in 1:nm) rtol = 1e-12
            @test abs(L.dH - (L.supply - L.losses)) < 1e-11 * (abs(L.supply) + L.losses)
            @test L.H ≈ phs_energy(cs, y) rtol = 1e-13
        end
        # a held decision changes the vector field: the frozen field differs once the state moves
        y = copy(xc)
        modes0 = phs_modes(cs, y, uc)
        y[cs.xoff[2]+1] = 5.0
        @test PHPS._frozen_rhs!(zeros(cs.n), cs, y, uc, modes0) != phs_rhs(cs, y, uc)

        r = phs_simulate(cs, xc, uc; tstop = 0.006, faults = [PHFault(16, 0.001, 0.002, 0.004)], tol = 1e-14)
        @test all(r.scheme_max .<= 1e-10 .* r.throughput)
        @test size(r.X, 1) == cs.n
        @test phs_control_signals(cs, r.X[:, end]).Efd[2] > sig.Efd[2]    # the exciter responds to the dip
        @test r.X[1:247, 1] == x0
    end

    @testset "physical self-excited DC exciters, ideal armature (PH_EXCITER_MODEL.md, 11.7)" begin
        c0, xc0, uc0, _ = phs_controlled(eq)
        nm = length(c0.s.machines)
        shunt = ["IEEET1_2", "IEEET1_3", "IEEET1_4", "IEEET1_7", "IEEET1_9", "IEEET1_11"]
        sig0 = phs_control_signals(c0, xc0)
        for (nv, fr) in ((1.0, 50.0), (4.0, 20.0))
            # the base ratio n and the field ratio are the two numbers no source gives; unit 2
            # admits no n above 2.82 (1 + n (KE + A) > 0, the odd field curve)
            nd = Dict(k => (k == "IEEET1_2" ? 1.0 : nv) for k in shunt)
            cs, xs, us, reps = phs_controlled(eq; dc_exciter = (; n = nd, field_ratio = fr))
            js = PHPS._physical_exciters(cs)
            @test sort(reps.physical_exciters) == sort(shunt)
            @test count(e -> e isa IEEET1_PHTRUE, cs.exc) == 3            # the KE = 1 units keep the law
            @test cs.n == 247 + 6 * 3 + 3 * 5 + 8 * 7 + 5 == length(phs_state_names(cs))
            # the operating point of the physical model is unchanged, and everything is at rest
            @test reps.residual_controlled < 1e-11 && reps.input_change < 1e-12
            @test xs[1:247] ≈ x0 rtol = 1e-12
            @test cs.Vref ≈ c0.Vref rtol = 1e-13
            sig = phs_control_signals(cs, xs)
            @test sig.Efd ≈ sig0.Efd rtol = 1e-12
            # the turbines supply the excitation: their power rises by what the exciters take
            @test sig.Pm .- sig0.Pm ≈ sig.Pexc atol = 1e-13
            @test all(>(0), sig.Pexc[js]) && all(iszero, sig.Pexc[setdiff(1:nm, js)])
            L0 = phs_ledger(cs, xs, us)
            ke = findall(==(:exciter), L0.kind)
            @test L0.names[ke] == [cs.exc[j].name * ".exciter" for j in js] && length(L0.names) == length(L0.energy)
            @test all(>(0), L0.energy[ke]) && all(>(0), L0.loss[ke])
            @test abs(L0.dH) < 1e-12 * L0.supply
            # the field takes about 1/field_ratio of the machine's field current (more with saturation)
            fl0 = phs_flows(cs.s, xs, us)
            for j in js
                e = cs.exc[j]
                q = PHPS._exc_at(cs, xs, j)
                xE = e.c * xs[cs.xoff[j]]
                @test q.i_f / fl0.mc[j].ifd ≈ (1 + e.n * (e.KE + e.A * exp(e.B * xE))) / fr rtol = 1e-12
                @test e.rF == fr * cs.s.machines[j].w.Rfd && e.n == nd[e.name]
            end

            scl_c = phs_scales(cs, xs)
            for trial in 1:3
                y = xs .+ 0.1 .* scl_c .* randn(rng, cs.n)
                if trial == 3                # a regulator beyond its limit and still asked for more
                    y[cs.xoff[js[1]]+1] = cs.exc[js[1]].VRMAX + 0.5
                    y[cs.xoff[js[1]]+2] += 1000.0
                end
                f = phs_rhs(cs, y, us)
                up = phs_control_inputs(cs, y, us.gf)
                @test f[1:247] ≈ phs_rhs(cs.s, y[1:247], up) rtol = 1e-13
                vg = phs_terminal_voltage(cs.s, y[1:247], up)
                flc = phs_flows(cs.s, y, up)
                Ly = phs_ledger(cs, y, us)
                amp = 0.0
                for (a, j) in enumerate(js)
                    e, m, o = cs.exc[j], cs.s.machines[j], cs.xoff[j]
                    phi, xa, xf = y[o], y[o+1], y[o+2]
                    om, ifd = flc.mc[j].omega, flc.mc[j].ifd
                    xE = e.c * phi
                    VR = clamp(xa, e.VRMIN, e.VRMAX)
                    SE = e.A * exp(e.B * xE)
                    # the generated voltage follows the speed: Efd = omega x_E
                    @test up.vf[j] ≈ phi * om / (e.n * e.TE) rtol = 1e-14
                    @test up.vf[j] * m.w.Lad / m.w.Rfd ≈ om * xE rtol = 1e-13
                    # the field row is the IEEET1 row with the speed acting on the self-excitation
                    @test e.c * f[o] ≈ (VR + (om - 1) * xE / e.n - (e.KE + SE) * xE) / e.TE rtol = 1e-11 atol = 1e-12
                    # regulator and stabiliser: the IEEET1 kernel's rows, fed by Efd = omega x_E
                    vt = abs(haskey(vg, m.bus) ? vg[m.bus] : flc.v[m.node])
                    ref = rhs!(zeros(5), c0.exc[j], [0.0, xa, om * xE, xf, 1.0], [vt, cs.Vref[j], 0.0, 0.0])
                    @test f[o+1] ≈ ref[2] rtol = 1e-9 atol = 1e-9
                    @test f[o+2] ≈ ref[4] rtol = 1e-12 atol = 1e-12
                    # at rated speed the field row is the kernel's exciter row
                    refx = rhs!(zeros(5), c0.exc[j], [0.0, xa, xE, xf, 1.0], [vt, cs.Vref[j], 0.0, 0.0])
                    @test e.c * (f[o] - phi * (om - 1) / (e.n * e.TE)) ≈ refx[3] rtol = 1e-11 atol = 1e-12
                    # the shaft gives v_f (i_fd + i_f); the stored energy's slope is the field current
                    i_f = (1 + e.n * (e.KE + SE)) * phi / (e.n * e.TE * e.rF)
                    @test PHPS._turbine_power(cs, y, j) - up.Tm[j] ≈ up.vf[j] * (ifd + i_f) atol = 1e-12
                    h = 1e-6 * abs(phi)
                    @test (PHPS._exc_energy(e, phi + h) - PHPS._exc_energy(e, phi - h)) / 2h ≈ i_f rtol = 1e-8
                    # the exciter's own balance: rate = port - loss
                    k = ke[a]
                    @test Ly.rate[k] ≈ Ly.port[k] - Ly.loss[k] rtol = 1e-11 atol = 1e-13 * Ly.tport[k]
                    @test Ly.port[k] ≈ up.vf[j] * i_f + e.b * VR * i_f rtol = 1e-12
                    @test Ly.loss[k] ≈ e.rF * i_f^2 rtol = 1e-12
                    amp += e.b * VR * i_f
                end
                # the whole model: supplies are the turbines, the amplifiers, and the field
                # voltages of the machines without a physical exciter
                turb = sum(PHPS._turbine_power(cs, y, j) for j in 1:nm)
                other = sum(up.vf[j] * flc.mc[j].ifd for j in setdiff(1:nm, js))
                @test Ly.supply ≈ turb + amp + other rtol = 1e-12
                @test abs(Ly.dH - (Ly.supply - Ly.losses)) < 1e-11 * (abs(Ly.supply) + Ly.losses)
                @test Ly.H ≈ phs_energy(cs, y) rtol = 1e-13
                modes = phs_modes(cs, y, us)
                @test PHPS._frozen_rhs!(zeros(cs.n), cs, y, us, modes) == f
                @test all(length(modes[j, 1]) == 2 for j in js)           # the non-windup limit only
                # the limit holds the regulator's rate, and the amplifier's output is clamped
                trial == 3 && @test modes[js[1], 1][1] && f[cs.xoff[js[1]]+1] == 0
            end

            # a short fault: the discrete balance holds for every component, exciters included;
            # the exciters' energy is not quadratic, so its closure is the scheme's error
            r = phs_simulate(cs, xs, us; tstop = 0.006, faults = [PHFault(16, 0.001, 0.002, 0.004)], tol = 1e-14)
            kr = findall(==(:exciter), r.kind)
            @test length(kr) == 6 && size(r.energy, 1) == length(r.names)
            @test all(r.scheme_max .<= 1e-10 .* r.throughput)
            @test r.total.scheme_max < 1e-10 * r.total.throughput
            @test all(abs.(r.closure[kr]) .<= 1e-6 .* r.moved[kr])
            @test [phs_energy(cs, r.X[:, k]) for k in eachindex(r.t)] ≈ r.H rtol = 1e-13
            @test r.X[1:247, 1] == x0
        end
        # refused: a base ratio at which unit 2's field characteristic is not increasing, a
        # missing entry, a field resistance that is not positive
        @test_throws ArgumentError phs_controlled(eq; dc_exciter = (; n = 3.0, field_ratio = 50.0))
        @test_throws KeyError phs_controlled(eq; dc_exciter = (; n = Dict("IEEET1_2" => 1.0), field_ratio = 50.0))
        @test_throws ArgumentError phs_controlled(eq; dc_exciter = (; n = 1.0, field_ratio = -1.0))

        # the exciter by itself: the stored energy's slope is the field current on both sides of
        # the point where the energy formula changes branch, with and without saturation
        e0 = first(e for e in c0.exc if e !== nothing && e.name == "IEEET1_3")
        ifield(e, phi) = PHPS._exc_field_volts(e, phi) / e.rF
        for (A, B) in ((e0.p.SAT_A, e0.p.SAT_B), (e0.p.SAT_A, 0.0), (0.0, e0.p.SAT_B), (0.0, 0.0))
            ed = IEEET1_PHTRUE("x", PHPS.Ieeet1Params(0.0, e0.p.KA, e0.p.TA, e0.p.KE, e0.p.TE, e0.p.KF, e0.p.TF,
                                                          e0.p.VRMAX, e0.p.VRMIN, A, B, 1.0, 1.0, 0.0), e0.params)
            e = PHShuntExciter(ed, 6e-4, 1.6; n = 2.0, rF = 0.03)
            @test PHPS._exc_energy(e, 0.0) == 0
            ys = B > 0 ? (1e-5, 9.99e-3, 1.01e-2, 0.3, 3.0, -1e-5, -9.99e-3, -1.01e-2, -3.0) : (0.5, -0.5)
            for y in ys
                phi = B > 0 ? y / (B * e.c) : y * 1e-3
                h = 1e-4 * abs(phi)
                # Simpson's rule over [phi - h, phi + h] against the energy difference
                quad = (ifield(e, phi - h) + 4 * ifield(e, phi) + ifield(e, phi + h)) * h / 3
                @test PHPS._exc_energy(e, phi + h) - PHPS._exc_energy(e, phi - h) ≈ quad rtol = 1e-9
                @test PHPS._exc_energy(e, phi) > 0
            end
            # the energy from zero by quadrature
            phi = 2e-3
            N = 2000
            quad = sum(ifield(e, phi * (k - 0.5) / N) for k in 1:N) * phi / N
            @test PHPS._exc_energy(e, phi) ≈ quad rtol = 1e-6
            # no exponential is evaluated without saturation, however large the flux
            A == 0 && @test isfinite(PHPS._exc_field_volts(e, 1e6))
            # the field curve is odd and its energy even (saturation at |x_E|, 12.6.2); at zero flux
            # the derivative by automatic differentiation (at +0.0 and -0.0) is the analytic slope,
            # and the energy's slope is the field current; at negative flux the stored energy's rate
            # is the net port power less the loss
            for phi in (1e-4, 2e-3, 0.05)
                @test PHPS._exc_field_volts(e, -phi) == -PHPS._exc_field_volts(e, phi)
                @test PHPS._exc_energy(e, -phi) == PHPS._exc_energy(e, phi)
                for (xa, w, ifd) in ((0.3, 1.0, 1.2), (-0.7, 0.98, 0.4))
                    pw = PHPS._exc_powers(e, -phi, xa, w, ifd)
                    @test pw.slope * pw.flux_rate ≈ pw.port - pw.loss rtol = 1e-12
                end
            end
            for z in (0.0, -0.0)
                @test ForwardDiff.derivative(p -> PHPS._exc_field_volts(e, p), z) ≈
                      (1 + e.n * (e.KE + e.A)) / (e.n * e.TE) rtol = 1e-14
                @test ForwardDiff.derivative(p -> PHPS._exc_energy(e, p), z) == ifield(e, z)
            end
        end
        bad(; kw...) = PHShuntExciter(e0, 6e-4, 1.6; n = 1.0, rF = 0.03, kw...)
        @test bad() isa PHShuntExciter
        @test_throws ArgumentError bad(n = Inf)
        @test_throws ArgumentError bad(rF = 0.0)
        @test_throws ArgumentError PHShuntExciter(e0, 0.0, 1.6; n = 1.0, rF = 0.03)
        @test_throws ArgumentError PHShuntExciter(e0, 6e-4, -1.0; n = 1.0, rF = 0.03)
    end

    @testset "physical brushless exciters (PH_BRUSHLESS_EXCITER_MODEL.md)" begin
        c0, xc0, uc0, _ = phs_controlled(eq)
        nm = length(c0.s.machines)
        sig0 = phs_control_signals(c0, xc0)
        # the rectifier's regulation: the report's three segments with exact coefficients join in
        # value and in slope, and stay within its rounding
        fex = PHPS._fex
        @test fex(0.0) == 1 && fex(sqrt(3) / 4) ≈ 0.75 && fex(0.75) ≈ sqrt(3) / 4 && fex(1.0) == 0 && fex(2.5) == 0
        for b in (sqrt(3) / 4, 0.75)
            @test fex(b - 1e-9) ≈ fex(b + 1e-9) atol = 1e-8
            @test (fex(b) - fex(b - 1e-5)) / 1e-5 ≈ (fex(b + 1e-5) - fex(b)) / 1e-5 atol = 1e-4
        end
        @test abs(fex(0.3) - (1 - 0.58 * 0.3)) < 2e-3 && fex(0.6) ≈ sqrt(0.75 - 0.36) && abs(fex(0.9) - 1.732 * 0.1) < 1e-4
        @test all(fex(a) > fex(a + 0.01) for a in 0.0:0.01:0.98)

        brushless_names = ["IEEET1_10", "IEEET1_5", "IEEET1_8"]
        for (sat, share) in (((0.0, 0.0), 0.02), ((0.02, 0.4), 0.1))
            bl = (; KC = 0.2, KD = 0.38, sat, field_share = share)
            cs, xs, us, reps = phs_controlled(eq; brushless = bl)
            js = PHPS._physical_exciters(cs)
            @test sort(reps.physical_exciters) == brushless_names
            @test all(cs.exc[j] isa PHBrushlessExciter for j in js) && count(e -> e isa IEEET1_PHTRUE, cs.exc) == 6
            @test cs.n == 247 + 3 * 3 + 6 * 5 + 8 * 7 + 5 == length(phs_state_names(cs))
            # the operating point of the physical model is unchanged, and everything is at rest
            @test reps.residual_controlled < 1e-11 && reps.input_change < 1e-12
            @test xs[1:247] ≈ x0 rtol = 1e-12
            sig = phs_control_signals(cs, xs)
            @test sig.Efd ≈ sig0.Efd rtol = 1e-12
            @test sig.Pm .- sig0.Pm ≈ sig.Pexc atol = 1e-13
            @test all(>(0), sig.Pexc[js]) && all(iszero, sig.Pexc[setdiff(1:nm, js)])
            fl0 = phs_flows(cs.s, xs, us)
            for j in js
                e, m = cs.exc[j], cs.s.machines[j]
                q = PHPS._exc_at(cs, xs, j)
                # steady state at rated speed: EFD = IFD, the first mode, IN the same at any load
                @test q.EFD ≈ m.w.Lad * fl0.mc[j].ifd rtol = 1e-12
                @test q.IN ≈ 0.2 / (1 + 0.2 / sqrt(3)) rtol = 1e-12
                @test xs[cs.xoff[j]] ≈ (1 + 0.2 / sqrt(3)) * q.EFD rtol = 1e-12
                @test xs[cs.xoff[j]+1] == xs[cs.xoff[j]+2] == q.VFE && abs(q.tr) < 1e-18
                @test e.g == share * m.w.Rfd / m.w.Lad^2 && e.KE == 1
                # the field loss over the output, without saturation and load effects, is the share
                sat == (0.0, 0.0) && @test e.g * 1.0^2 / (e.gain * 1.0 * 1.0 / m.w.Lad) ≈ share rtol = 1e-13
            end
            L0 = phs_ledger(cs, xs, us)
            ke = findall(==(:exciter), L0.kind)
            @test L0.names[ke] == [cs.exc[j].name * ".exciter" for j in js]
            @test all(>(0), L0.energy[ke]) && all(>(0), L0.loss[ke]) && abs(L0.dH) < 1e-12 * L0.supply

            scl_c = phs_scales(cs, xs)
            for trial in 1:3
                y = xs .+ 0.1 .* scl_c .* randn(rng, cs.n)
                if trial == 3
                    y[cs.xoff[js[1]]+1] = cs.exc[js[1]].VRMAX + 0.5
                    y[cs.xoff[js[1]]+2] += 1000.0
                end
                f = phs_rhs(cs, y, us)
                up = phs_control_inputs(cs, y, us.gf)
                @test f[1:247] ≈ phs_rhs(cs.s, y[1:247], up) rtol = 1e-13
                vg = phs_terminal_voltage(cs.s, y[1:247], up)
                flc = phs_flows(cs.s, y, up)
                Ly = phs_ledger(cs, y, us)
                for (a, j) in enumerate(js)
                    e, m, o = cs.exc[j], cs.s.machines[j], cs.xoff[j]
                    xE, xa, xf = y[o], y[o+1], y[o+2]
                    om, ifd = flc.mc[j].omega, flc.mc[j].ifd
                    IFD = m.w.Lad * ifd
                    VR = clamp(xa, e.VRMIN, e.VRMAX)
                    SE = sat[1] * exp(sat[2] * xE)
                    VFE = (1 + SE) * xE + 0.38 * IFD
                    @test xE > 0 && IFD > 0
                    # the field circuit, and the rectified voltage following the speed
                    @test f[o] ≈ (VR - VFE) / e.TE rtol = 1e-12 atol = 1e-13
                    @test up.vf[j] * m.w.Lad / m.w.Rfd ≈ fex(0.2 * IFD / xE) * om * xE rtol = 1e-13
                    # regulator and stabiliser: the IEEET1 kernel's rows, fed by VFE
                    vt = abs(haskey(vg, m.bus) ? vg[m.bus] : flc.v[m.node])
                    ref = rhs!(zeros(5), c0.exc[j], [0.0, xa, VFE, xf, 1.0], [vt, cs.Vref[j], 0.0, 0.0])
                    @test f[o+1] ≈ ref[2] rtol = 1e-9 atol = 1e-9
                    @test f[o+2] ≈ ref[4] rtol = 1e-12 atol = 1e-12
                    # the powers, and all of them from the shaft
                    Pdc, Preg, Ptr = up.vf[j] * ifd, e.g * VR * VFE, e.g * 0.38 * IFD * (VR - VFE)
                    @test PHPS._turbine_power(cs, y, j) - up.Tm[j] ≈ Pdc - Ptr + Preg atol = 1e-12
                    h = 1e-6 * xE
                    @test (PHPS._exc_energy(e, xE + h) - PHPS._exc_energy(e, xE - h)) / 2h ≈
                          e.g * e.TE * (1 + SE) * xE rtol = 1e-8
                    k = ke[a]
                    @test Ly.rate[k] ≈ Ly.port[k] - Ly.loss[k] rtol = 1e-10 atol = 1e-13 * Ly.tport[k]
                    @test Ly.port[k] ≈ Preg - Ptr rtol = 1e-10 atol = 1e-15
                    @test Ly.loss[k] ≈ e.g * VFE^2 rtol = 1e-12
                end
                # the whole model: no supply but the turbines and the field voltages of the
                # machines without a physical exciter
                turb = sum(PHPS._turbine_power(cs, y, j) for j in 1:nm)
                other = sum(up.vf[j] * flc.mc[j].ifd for j in setdiff(1:nm, js))
                @test Ly.supply ≈ turb + other rtol = 1e-12
                @test abs(Ly.dH - (Ly.supply - Ly.losses)) < 1e-11 * (abs(Ly.supply) + Ly.losses)
                @test Ly.H ≈ phs_energy(cs, y) rtol = 1e-13
                modes = phs_modes(cs, y, us)
                @test PHPS._frozen_rhs!(zeros(cs.n), cs, y, us, modes) == f
                @test all(length(modes[j, 1]) == 2 for j in js)
                trial == 3 && @test modes[js[1], 1][1] && f[cs.xoff[js[1]]+1] == 0
            end

            r = phs_simulate(cs, xs, us; tstop = 0.006, faults = [PHFault(16, 0.001, 0.002, 0.004)], tol = 1e-14)
            kr = findall(==(:exciter), r.kind)
            @test length(kr) == 3
            @test all(r.scheme_max .<= 1e-10 .* r.throughput) && r.total.scheme_max < 1e-10 * r.total.throughput
            @test all(abs.(r.closure[kr]) .<= 1e-6 .* r.moved[kr])
            @test [phs_energy(cs, r.X[:, k]) for k in eachindex(r.t)] ≈ r.H rtol = 1e-13
        end

        # both kinds together: no exciter is a control law, and the only supplies are the
        # turbines, the DC exciters' amplifiers and the field voltage of the unit without an exciter
        shunt = ["IEEET1_2", "IEEET1_3", "IEEET1_4", "IEEET1_7", "IEEET1_9", "IEEET1_11"]
        cs, xs, us, reps = phs_controlled(eq; dc_exciter = (; n = Dict(k => 1.0 for k in shunt), field_ratio = 50.0),
                                          brushless = (; KC = 0.2, KD = 0.38, sat = (0.0, 0.0), field_share = 0.02))
        @test count(e -> e isa IEEET1_PHTRUE, cs.exc) == 0 && length(reps.physical_exciters) == 9
        @test reps.residual_controlled < 1e-11 && xs[1:247] ≈ x0
        y = xs .+ 0.05 .* phs_scales(cs, xs) .* randn(rng, cs.n)
        Ly = phs_ledger(cs, y, us)
        up = phs_control_inputs(cs, y, us.gf)
        flc = phs_flows(cs.s, y, up)
        amp = sum(PHPS._exc_at(cs, y, j).amp for j in eachindex(cs.exc) if cs.exc[j] isa PHShuntExciter)
        @test Ly.supply ≈ sum(PHPS._turbine_power(cs, y, j) for j in 1:nm) + amp + up.vf[1] * flc.mc[1].ifd rtol = 1e-12
        @test cs.exc[1] === nothing
        @test abs(Ly.dH - (Ly.supply - Ly.losses)) < 1e-11 * (abs(Ly.supply) + Ly.losses)

        # refused: an operating point outside the rectifier's first mode, a share that is not
        # positive, a missing number
        ok = (; KC = 0.2, KD = 0.38, sat = (0.0, 0.0), field_share = 0.02)
        @test_throws ArgumentError phs_controlled(eq; brushless = (; ok..., KC = 2.0))
        @test_throws ArgumentError phs_controlled(eq; brushless = (; ok..., field_share = 0.0))
        @test_throws UndefKeywordError phs_controlled(eq; brushless = (; KC = 0.2, KD = 0.38, sat = (0.0, 0.0)))
        # the continuation outside the model: no output without flux, and none beyond IN = 1
        e = first(cs.exc[j] for j in eachindex(cs.exc) if cs.exc[j] isa PHBrushlessExciter)
        @test PHPS._bx_efd(e, 0.0, 2.0, 1.0) == 0 && PHPS._bx_efd(e, -1.0, 2.0, 1.0) == 0
        @test PHPS._bx_efd(e, 1.0, 5.0, 1.0) == 0 && PHPS._bx_in(e, 1.0, 5.0) == 1.0
        @test PHPS._bx_efd(e, 2.0, -1.0, 1.0) == 2.0                 # a negative field current: continued, outside the model
        # the derivatives automatic differentiation gives (the integrator's Jacobian uses them)
        # against finite differences, in each of the rectifier's three modes
        for IN in (0.2, 0.6, 0.9)
            xE, om = 2.3, 1.01
            IFD = IN * xE / e.KC
            ifd = IFD / e.Lad
            for (fn, z) in ((x -> PHPS._exc_powers(e, x, 1.5, om, ifd).vf, xE),
                            (x -> PHPS._exc_powers(e, x, 1.5, om, ifd).shaft, xE),
                            (i -> PHPS._exc_powers(e, xE, 1.5, om, i).vf, ifd),
                            (i -> PHPS._exc_powers(e, xE, 1.5, om, i).shaft, ifd),
                            (w -> PHPS._exc_powers(e, xE, 1.5, w, ifd).vf, om))
                h = 1e-6 * z
                @test ForwardDiff.derivative(fn, z) ≈ (fn(z + h) - fn(z - h)) / 2h rtol = 1e-7
            end
            @test PHPS._exc_powers(e, xE, 1.5, om, ifd).IN ≈ IN
        end
    end

    @testset "physical steam units (PH_STEAM_UNIT_MODEL.md, sections 3, 13, 14, 15)" begin
        # local: the enclosing testset's s, u, x0 and others must not be overwritten
        local P, tb, kvap, sat8, gv, volC, base, g, un, nom, ns, xs, e, x, Pref, dx, f, J, er, xr, g2, paths,
              c0, xc0, uc0, cs, xs, us, reps, ks, scl_s, L0, kk, y, Ly, modes, r, kr, rho, Tg, T, k, rr, p
        P = PHPS.PhysicalSteam
        tb = steam_tables()
        # the kernel's vapour density at (p, T), for references
        function kvap(p, T)
            rho = p / (P.IAPWS95_R * T)
            for _ in 1:200
                k = P.iapws95_properties(T, rho)
                d = (k.pressure - p) / k.pressure_density
                rho -= d
                abs(d) < 1e-14 * rho && return rho
            end
            error("no density")
        end
        # tables against the kernel: saturation at the release's Table 8 points
        for (T, p, rl, rv, hl, hv, sl, sv, tol) in ((450.0, 0.932203564e6, 0.890341250e3, 0.481200360e1, 0.749161585e6,
                                                     0.277441078e7, 0.210865845e4, 0.660921221e4, 2e-7),
                                                    (625.0, 0.169082693e8, 0.567090385e3, 0.118290280e3, 0.168626976e7,
                                                     0.255071625e7, 0.380194683e4, 0.518506121e4, 5e-5))
            sat8 = steam_sat(tb, p)
            @test sat8.T ≈ T rtol = tol
            @test 1 / sat8.v_f ≈ rl rtol = tol
            @test 1 / sat8.v_g ≈ rv rtol = tol
            @test sat8.h_f ≈ hl rtol = tol
            @test sat8.h_g ≈ hv rtol = tol
            @test sat8.s_f ≈ sl rtol = tol
            @test sat8.s_g ≈ sv rtol = tol
        end
        # the vapour map and the isentrope at random states
        for _ in 1:150
            rho = exp(log(0.01) + rand(rng) * (log(150.0) - log(0.01)))
            Tg = PHPS._interp(tb.gas, 1, log(rho), "")
            T = Tg + 2.0 + rand(rng) * (990.0 - Tg - 2.0)
            k = P.iapws95_properties(T, rho)
            rr = steam_state(tb, rho, k.internal_energy_specific)
            # measured worst on this grid: T and p 4e-6 near the dome at a few kPa, h 8e-8, s 7e-9
            @test rr.T ≈ T rtol = 1e-5
            @test rr.p ≈ k.pressure rtol = 1e-5
            @test rr.h ≈ k.enthalpy rtol = 3e-7
            @test rr.s ≈ k.entropy rtol = 3e-8
            @test rr.h == k.internal_energy_specific + rr.p / rho        # enforced, not tabulated
            @test rr.x == 1
        end
        for _ in 1:100
            p = exp(log(1000.0) + rand(rng) * (log(19.0e6) - log(1000.0)))
            T = steam_sat(tb, p).T + 1.0 + rand(rng) * 400.0
            T > 999.0 && continue
            k = P.iapws95_properties(T, kvap(p, T))
            @test steam_h_isentropic(tb, p, k.entropy).h ≈ k.enthalpy rtol = 3e-7
        end
        # Gibbs relation in the vapour map: ds/du at fixed rho is 1/T, ds/dv at fixed u is p/T
        let rho = 7.4, u = 2.95e6
            g(z) = steam_state(tb, z[1], z[2]).s
            d = ForwardDiff.gradient(g, [rho, u])
            r = steam_state(tb, rho, u)
            @test d[2] ≈ 1 / r.T rtol = 1e-5
            @test -d[1] * rho^2 ≈ r.p / r.T rtol = 1e-5                 # ds/dv = -rho^2 ds/drho
        end
        # a wet state: the lever rule, h = u + p v, and its derivative against finite differences
        let rho = 0.2, u = 2.2e6
            r = steam_state(tb, rho, u)
            @test 0 < r.x < 1
            @test r.h ≈ u + r.p / rho rtol = 1e-6
            g(z) = steam_state(tb, z[1], z[2]).p
            d = ForwardDiff.gradient(g, [rho, u])
            for (i, hh) in ((1, 1e-6 * rho), (2, 1.0))
                e = zeros(2); e[i] = hh
                @test d[i] ≈ (g([rho, u] .+ e) - g([rho, u] .- e)) / 2hh rtol = 1e-6
            end
        end
        @test_throws DomainError steam_state(tb, 500.0, 3.0e6)
        # the dry boundary (consultation 8, point 12): across it the state is continuous, and on
        # either side ds/du at fixed density is 1/T (no plateau)
        for rho in (0.2, 3.0, 40.0)
            lg = PHPS._sat_l_of_vg(tb, 1 / rho)
            ub = PHPS._sat_ug(tb, lg)
            a, b = steam_state(tb, rho, ub * (1 - 1e-9)), steam_state(tb, rho, ub * (1 + 1e-9))
            @test a.x < 1 + 1e-6 && b.x == 1
            @test a.p ≈ b.p rtol = 1e-5
            @test a.T ≈ b.T rtol = 1e-5
            @test a.s ≈ b.s rtol = 1e-6
            for uu in (ub * (1 - 1e-7), ub * (1 + 1e-7), ub * (1 + 1e-4))
                d = ForwardDiff.derivative(z -> steam_state(tb, rho, z).s, uu)
                @test d ≈ 1 / steam_state(tb, rho, uu).T rtol = 1e-3
            end
        end
        # steam_ph inverts steam_state, on both sides of the dome, and refuses the liquid
        for (pp, hh) in ((5.0e4, 2.4e6), (5.0e4, 2.8e6), (2.0e6, 3.2e6), (1.0e3, 2.6e6))
            r = PHPS.steam_ph(tb, pp, hh)
            q = steam_state(tb, r.rho, r.u)
            @test q.p ≈ pp rtol = 1e-8
            @test q.h ≈ hh rtol = 1e-8
        end
        @test_throws DomainError PHPS.steam_ph(tb, 5.0e4, 1.0e5)
        # a hot vapour far above saturation (consultation 9, point 8), and the saturated liquid,
        # whose density is outside the tables (point 9)
        let rho = 1.0e5 / (PHPS._PS.IAPWS95_R * 990.0), c = PHPS._ps_core(990.0, rho)
            q0 = steam_state(tb, rho, c.u)
            r = PHPS.steam_ph(tb, q0.p, q0.h)
            @test r.rho ≈ rho rtol = 1e-6
            @test r.T ≈ 990.0 rtol = 1e-6
        end
        let s = steam_sat(tb, 1.0e6)
            @test_throws DomainError PHPS.steam_ph(tb, 1.0e6, s.u_f + 1.0e6 * s.v_f)
        end
        @test_throws DomainError steam_h_isentropic(tb, 30.0e6, 6000.0)

        # the units, standalone
        gv = Dict(PHPS.name(c) => c for c in eq.sys.comps if c isa IEEEG1_PHTRUE)
        volC = steam_volumes_from_time_constants(; tau = (0.25, 7.5, 7.5, 0.4))
        base = (; r_v = 0.95, rh_dT = 100.0, tau_n = 0.03, lowflow = 1e-3, tables = tb)
        for (gname, path, kw) in (("IEEEG1_2", :A, (; cold_reheat = 26.2)), ("IEEEG1_3", :C, (; volumes = volC)))
            g = gv[gname]
            un = PHSteamUnit(g; path, base..., kw...)
            nom = un.nominal
            @test all(0 .< un.sec_eta .<= 1)
            @test nom.Pm ≈ g.p.PMAX rtol = 1e-12
            ns = length(un.V)
            # the design point: at rest at full stroke, and the first law to rounding
            xs = [v for k in 1:ns for v in (nom.M[k], nom.M[k] * nom.u[k])]
            e = PHPS._steam_eval(un, xs, 1.0)
            @test maximum(abs.(e.dM) ./ nom.qin) < 1e-6
            @test e.W ≈ nom.W rtol = 1e-9
            @test abs(sum(e.dU) - (e.boiler + e.heat - e.cond - e.ext - e.W)) < 1e-12 * e.boiler
            # initialised at 70 % of PMAX: at rest, the reheat temperatures at design, the power
            x, Pref = steam_initialise(un, 0.7 * g.p.PMAX)
            dx = similar(x)
            PHPS._steam_rhs!(dx, un, x, 1.0, Pref, PHPS.NoModes())
            @test maximum(abs.(dx) ./ max.(1.0, abs.(x))) < 1e-10
            @test PHPS._steam_power(un, x) ≈ 0.7 * g.p.PMAX rtol = 1e-12
            for k in findall(>(0), un.kh)
                @test steam_state(tb, x[2+2k-1] / un.V[k], x[2+2k] / x[2+2k-1]).T ≈ nom.T[k] rtol = 1e-10
            end
            @test x[1] == x[2] && Pref ≈ 1 + x[2] / g.p.K
            # the Jacobian automatic differentiation gives, against finite differences
            f(z) = (d = similar(z); PHPS._steam_rhs!(d, un, z, 1.0, Pref, PHPS.NoModes()); d)
            J = ForwardDiff.jacobian(f, x)
            for i in 1:length(x)
                hh = 1e-6 * abs(x[i])
                e1 = zeros(length(x)); e1[i] = hh
                col = (f(x .+ e1) .- f(x .- e1)) ./ 2hh
                @test col ≈ J[:, i] rtol = 1e-6 atol = 1e-9 * maximum(abs, J[:, i])
            end
            # a negative pressure difference reverses a flow without work (donor cell): the chest
            # at a fifth of its mass, below the first node's pressure
            xr = copy(x); xr[3] *= 0.2; xr[4] *= 0.2
            er = PHPS._steam_eval(un, view(xr, 3:length(xr)), PHPS._steam_stroke(un, x[2]))
            @test er.qsec[1] < 0
            @test abs(sum(er.dU) - (er.boiler + er.heat - er.cond - er.ext - er.W)) < 1e-12 * abs(er.boiler)
            # entropy production at the initial state (consultation 7, point 25): every turbine
            # section, the valve and every resistance produce entropy
            e0 = PHPS._steam_eval(un, view(x, 3:length(x)), PHPS._steam_stroke(un, x[2]))
            for j in eachindex(un.sec_eta)
                a, b = un.sec_up[j], un.sec_down[j]
                pout = b == 0 ? un.p_cond : e0.st[b].p
                hout = e0.st[a].h - un.sec_eta[j] * (e0.st[a].h - steam_h_isentropic(tb, pout, e0.st[a].s).h)
                @test PHPS.steam_ph(tb, pout, hout).s >= e0.st[a].s
            end
            @test PHPS.steam_ph(tb, e0.st[1].p, un.h_T).s >= PHPS.steam_ph(tb, un.p_T, un.h_T).s
            for r_ in eachindex(un.res_K)
                a, b = un.res_up[r_], un.res_down[r_]
                @test PHPS.steam_ph(tb, e0.st[b].p, e0.st[a].h).s >= e0.st[a].s
            end
            # the valve passes nothing once the chest is at the throttle's pressure
            xv = copy(x); xv[3] *= 2.0; xv[4] *= 2.0
            ev = PHPS._steam_eval(un, view(xv, 3:length(xv)), 1.0)
            @test ev.st[1].p >= un.p_T
            @test ev.qv == 0
            # nothing comes back from the condenser: the last node below the condenser's pressure
            kl = un.sec_up[findfirst(==(0), un.sec_down)]
            xc = copy(x); xc[2+2kl-1] *= 0.05; xc[2+2kl] *= 0.05
            ec = PHPS._steam_eval(un, view(xc, 3:length(xc)), PHPS._steam_stroke(un, x[2]))
            if ec.st[kl].p < un.p_cond
                @test ec.cond == 0
            end
            # a vessel driven into the dome is refused (the crossover, cooled)
            kco = findfirst(==("crossover"), un.names)
            xd = copy(x); xd[2+2kco] = xd[2+2kco-1] * 1.5e6
            @test_throws DomainError PHPS._steam_eval(un, view(xd, 3:length(xd)), 0.5)
            # the static-chain reference (no nodes) is at rest at the node model's steady state
            ves = findall(!, un.isnode)
            xsv = [v for k in ves for v in (x[2+2k-1], x[2+2k])]
            es = PHPS.steam_static_eval(un, xsv, PHPS._steam_stroke(un, x[2]))
            @test maximum(abs.(es.dM) ./ nom.qin[ves]) < 1e-6
            # the reference reaches the extraction points through the (p, h) inversion of the tables,
            # the node model through the nodes' (rho, u): they agree to the tables' error (1e-6 seen)
            @test es.W ≈ PHPS._steam_power(un, x) * PHPS.STEAM_SB rtol = 1e-5
        end
        @test_throws ArgumentError steam_initialise(PHSteamUnit(gv["IEEEG1_2"]; path = :A, base..., cold_reheat = 26.2), 0.0)
        @test_throws ArgumentError PHSteamUnit(gv["IEEEG1_3"]; path = :C, base..., volumes = (; volC..., rh1 = -1.0))
        g2 = gv["IEEEG1_2"]
        @test_throws ArgumentError PHSteamUnit(g2; path = :A, base...)
        @test_throws ArgumentError PHSteamUnit(g2; path = :A, base..., cold_reheat = 27.0)
        @test_throws ArgumentError PHSteamUnit(g2; path = :C, base...)
        @test_throws ArgumentError PHSteamUnit(g2; path = :A, base..., r_v = 1.0, cold_reheat = 26.2)
        @test_throws ArgumentError PHSteamUnit(g2; path = :B, base..., cold_reheat = 26.2)
        @test_throws ArgumentError PHSteamUnit(g2; path = :A, base..., cold_reheat = 26.2, volumes = volC)
        @test_throws ArgumentError PHSteamUnit(g2; path = :A, base..., cold_reheat = 26.2, valve = :lift)
        # the valve characteristic (record 16.6): the servo state is the flow demand; the steady power
        # at the opening the map gives is the demand, between the nodes too; the map rises; the
        # opening mode keeps x1 / PMAX and the same plant
        let un = PHSteamUnit(gv["IEEEG1_3"]; path = :C, base..., volumes = :case)
            x, _ = steam_initialise(un, 0.9 * un.PMAX)
            @test x[2] == 0.9 * un.PMAX
            @test un.volumes["source"] == "case IEEEG1 T4-T7"
            @test all(diff(un.vc_d) .> 0) && all(diff(un.vc_y) .> 0) && all(un.vc_s .> 0)
            xs_ = x[3:end]
            for d in (0.45, 0.62, 0.8, 0.93) .* un.PMAX
                d < un.vc_d[end] || continue
                y = PHPS._steam_opening(un, d)
                xs_ = PHPS._steam_steady(un, y, xs_)
                # the map's interpolation error between nodes 0.02 apart in opening: 9e-6 seen near the
                # valve's choking point, 1e-8 elsewhere
                @test PHPS._steam_eval(un, xs_, y).W / PHPS.STEAM_SB ≈ d rtol = 2e-5
            end
            @test all(diff([PHPS._steam_stroke(un, d) for d in range(0.0, un.PMAX, length = 2001)]) .>= 0)
            uo = PHSteamUnit(gv["IEEEG1_3"]; path = :C, base..., volumes = :case, valve = :opening)
            xo, _ = steam_initialise(uo, 0.9 * uo.PMAX)
            @test PHPS._steam_stroke(uo, xo[2]) == xo[2] / uo.PMAX
            @test PHPS._steam_stroke(uo, xo[2]) ≈ PHPS._steam_stroke(un, x[2]) rtol = 1e-9
            @test xo[3:end] ≈ x[3:end] rtol = 1e-9
        end
        # path A from the case's time constants; volumes scale with the time constants
        let ua = PHSteamUnit(g2; path = :A, base..., cold_reheat = 26.2, volumes = :case)
            x, Pref = steam_initialise(ua, 0.8 * ua.PMAX)
            dx = similar(x)
            PHPS._steam_rhs!(dx, ua, x, 1.0, Pref, PHPS.NoModes())
            @test maximum(abs.(dx) ./ max.(1.0, abs.(x))) < 1e-10
            @test ua.volumes["V"][1] ≈ steam_volumes_from_time_constants(; tau = (0.6, 0.5, 0.8), path = :A, r_v = 0.95).chest
            @test ua.volumes["source"] == "case IEEEG1 T4-T6"
            # the chest's volume at its own design state differs from the one at the throttle's
            @test ua.volumes["V"][1] != steam_volumes_from_time_constants(; tau = (0.6, 0.5, 0.8), path = :A).chest
        end
        let ua = PHSteamUnit(g2; path = :A, base..., cold_reheat = 26.2, volumes = :case), vh = copy(ua.Tw)
            steam_initialise(ua, 0.8 * ua.PMAX)
            Tw1 = copy(ua.Tw)
            # a failed initialisation leaves the wall temperatures and the characteristic as they were
            @test_throws Exception steam_initialise(ua, 1.5 * ua.PMAX)
            @test ua.Tw == Tw1
        end
        # the volumes fitted to the case's IEEEG1 turbine (record 16.7): the fit improves on the reading of
        # the time constants, the unit is at rest after it, the opening mode refuses it
        let uf = PHSteamUnit(g2; path = :A, base..., cold_reheat = 26.2, volumes = :fit)
            x, Pref = steam_initialise(uf, 0.875 * uf.PMAX)
            dx = similar(x)
            PHPS._steam_rhs!(dx, uf, x, 1.0, Pref, PHPS.NoModes())
            @test maximum(abs.(dx) ./ max.(1.0, abs.(x))) < 1e-10
            @test uf.volumes["rms"] < 0.03 && uf.volumes["rms"] < uf.volumes["rms_before"] / 5
            # the fitted volumes are the unit's after initialisation, and its states are at them (a closure
            # once reset the factors to one: record 16.8)
            ves = findall(!, uf.isnode)
            @test [uf.V[k] / uf.volumes["V_unfitted"][k] for k in ves] ≈ [uf.volumes["factors"][uf.names[k]] for k in ves]
            @test all(abs(uf.volumes["factors"][uf.names[k]] - 1) > 0.05 for k in ves)
            @test uf.volumes["objective"] == "step"
        end
        # the frequency objective (record 16.8): within a few per cent of IEEEG1's over 0.02-2 Hz, and at the
        # grid's swing frequency
        let uq = PHSteamUnit(g2; path = :A, base..., cold_reheat = 26.2, volumes = :fit_frequency)
            x, Pref = steam_initialise(uq, 0.875 * uq.PMAX)
            dx = similar(x)
            PHPS._steam_rhs!(dx, uq, x, 1.0, Pref, PHPS.NoModes())
            @test maximum(abs.(dx) ./ max.(1.0, abs.(x))) < 1e-10
            @test uq.volumes["objective"] == "frequency"
            @test uq.volumes["freq_error"] < 0.05
            @test abs(uq.volumes["at_0.65Hz"]["gain_ratio"] - 1) < 0.03
            @test abs(uq.volumes["at_0.65Hz"]["phase_deg"]) < 3
            ves = findall(!, uq.isnode)
            @test [uq.V[k] / uq.volumes["V_unfitted"][k] for k in ves] ≈ [uq.volumes["factors"][uq.names[k]] for k in ves]
            @test PHPS._steam_power(uq, x) ≈ 0.875 * uq.PMAX rtol = 1e-12
        end
        @test_throws ArgumentError PHSteamUnit(g2; path = :A, base..., cold_reheat = 26.2, volumes = :fit, valve = :opening)
        # the grid option reaches both paths (a :fit once reached path C only)
        @test PHPS._steam_volumes_for(:fit, :A) === :fit && PHPS._steam_volumes_for(:fit, :C) === :fit
        @test PHPS._steam_volumes_for(:case, :A) === :case
        for (path, tau) in ((:A, (0.6, 0.5, 0.8)), (:C, (0.6, 0.5, 0.8, 1.0)))
            v1 = steam_volumes_from_time_constants(; tau, path)
            v2 = steam_volumes_from_time_constants(; tau = 2 .* tau, path)
            @test collect(values(v2)) ≈ 2 .* collect(values(v1)) rtol = 1e-12
        end

        # in the grid model: the user's assignment, A for units 2, 5, 6, 8, 9 and C for 3, 4, 7, 10
        paths = Dict("IEEEG1_2" => :A, "IEEEG1_5" => :A, "IEEEG1_6" => :A, "IEEEG1_8" => :A, "IEEEG1_9" => :A,
                     "IEEEG1_3" => :C, "IEEEG1_4" => :C, "IEEEG1_7" => :C, "IEEEG1_10" => :C)
        c0, xc0, uc0, _ = phs_controlled(eq)
        cs, xs, us, reps = phs_controlled(eq; steam = (; paths, base..., cold_reheat = 26.2, volumes = volC))
        ks = PHPS._steam_units(cs)
        @test length(ks) == 8 && length(reps.steam_units) == 8                # units 5 and 6 share a machine
        @test count(g -> g isa IEEEG1_PHTRUE, cs.gov) == 0
        @test reps.input_change < 1e-12
        @test xs[1:247] ≈ xc0[1:247] rtol = 1e-12
        scl_s = phs_scales(cs, xs)
        @test maximum(abs.(phs_rhs(cs, xs, us)) ./ scl_s) < 1e-10
        @test phs_control_signals(cs, xs).Pm ≈ phs_control_signals(c0, xc0).Pm rtol = 1e-12
        L0 = phs_ledger(cs, xs, us)
        kk = findall(==(:steam), L0.kind)
        @test length(kk) == 8 && all(>(0), L0.energy[kk]) && all(iszero, L0.loss[kk])
        @test abs(L0.dH) < 1e-12 * L0.supply
        for trial in 1:2
            y = xs .+ 0.02 .* scl_s .* randn(rng, cs.n)
            # the storages perturbed physically: each mass by a few per cent, its specific energy by
            # 0.2 % (independent 2 % changes of M and U would move a hot reheater past the tables'
            # 1000 K)
            for j in ks
                o = cs.goff[j]
                for kq in eachindex(cs.gov[j].V)
                    iM, iU = o + 2kq, o + 2kq + 1
                    y[iU] = y[iM] * (xs[iU] / xs[iM]) * (1 + 0.002 * randn(rng))
                end
            end
            Ly = phs_ledger(cs, y, us)
            # the first law of each unit: its energy's rate is its port, exactly
            @test Ly.rate[kk] ≈ Ly.port[kk] rtol = 1e-10
            @test abs(Ly.dH - (Ly.supply - Ly.losses)) < 1e-11 * (abs(Ly.supply) + Ly.losses)
            @test Ly.H ≈ phs_energy(cs, y) rtol = 1e-13
            modes = phs_modes(cs, y, us)
            @test PHPS._frozen_rhs!(zeros(cs.n), cs, y, us, modes) == phs_rhs(cs, y, us)
        end
        r = phs_simulate(cs, xs, us; tstop = 0.004, faults = [PHFault(16, 0.001, 0.001, 0.002)], tol = 1e-14)
        kr = findall(==(:steam), r.kind)
        @test all(r.scheme_max[kr] .<= 1e-10 .* r.throughput[kr])
        # the units' energy is a sum of states: the midpoint rule keeps it exactly (section 15.1)
        @test all(abs.(r.closure[kr]) .<= 1e-10 .* r.moved[kr] .+ 1e-12 .* r.throughput[kr])
        @test [phs_energy(cs, r.X[:, k]) for k in eachindex(r.t)] ≈ r.H rtol = 1e-13
    end

    @testset "physical hydro unit (PH_HYDRO_UNIT_MODEL.md, sections 3, 4)" begin
        local g3 = first(c for c in eq.sys.comps if c isa IEEEG3_PHTRUE)
        local Pm = 2.5
        local Tw = hydro_Tw_match(g3, Pm)
        @test Tw ≈ 0.75 * 8.5 / 2.5 rtol = 1e-6
        local un = PHHydroUnit(g3; Tw)
        local x, Pref = hydro_initialise(un, Pm)
        @test x ≈ [0.0, Pm, Pm, Pm / un.GMAX] && Pref ≈ un.p.Sigma * Pm
        local dx = similar(x)
        PHPS._hydro_rhs!(dx, un, x, 1.0, Pref, PHPS.NoModes())
        @test maximum(abs, dx) < 1e-14
        @test PHPS._hydro_power(un, x) ≈ Pm rtol = 1e-14
        # the energy balance, exactly, away from rest: H' = reservoir supply - shaft power
        local xa = [0.01, 2.4, 2.6, 0.31]
        PHPS._hydro_rhs!(dx, un, xa, 1.002, Pref, PHPS.NoModes())
        local q = PHPS._hydro_powers(un, xa)
        @test un.GMAX * un.Tw * xa[4] * dx[4] ≈ q.supply - q.shaft rtol = 1e-13
        @test q.rate ≈ q.port rtol = 1e-13
        # the linearisation from gate to power at rated speed is the case's IEEEG3 (1992 report, eq. (7))
        local F(z) = [(1 - PHPS._hydro_head(un, z[2], z[1])) / un.Tw]
        local J = ForwardDiff.jacobian(F, [x[4], x[3]])
        local dP = ForwardDiff.gradient(z -> PHPS._hydro_power(un, [0.0, 0.0, z[2], z[1]]), [x[4], x[3]])
        for f in (0.02, 0.2, 0.65, 2.0)
            local s_ = 2pi * f * im
            @test dP[1] * J[1, 2] / (s_ - J[1, 1]) + dP[2] ≈ (1 - 0.75s_) / (1 + 0.375s_) rtol = 1e-6
        end
        # every Jacobian column against finite differences, away from rest
        local fh(z) = (d = similar(z); PHPS._hydro_rhs!(d, un, z, 1.001, Pref, PHPS.NoModes()); d)
        local Jh = ForwardDiff.jacobian(fh, xa)
        for i in 1:4
            local hh = 1e-6 * max(abs(xa[i]), 1e-3)
            local e1 = zeros(4); e1[i] = hh
            @test (fh(xa .+ e1) .- fh(xa .- e1)) ./ 2hh ≈ Jh[:, i] rtol = 1e-6 atol = 1e-9 * maximum(abs, Jh[:, i])
        end
        # refusals: a closed gate with flow, reverse flow, a non-positive water time, power out of range
        @test_throws DomainError PHPS._hydro_rhs!(dx, un, [0.0, 2.5, 0.0, 0.29], 1.0, Pref, PHPS.NoModes())
        @test_throws DomainError PHPS._hydro_rhs!(dx, un, [0.0, 2.5, 2.5, -0.01], 1.0, Pref, PHPS.NoModes())
        @test_throws ArgumentError PHHydroUnit(g3; Tw = 0.0)
        @test_throws ArgumentError hydro_initialise(un, 9.0)
        # in the grid: at rest, the first law of the unit, a short fault run with the energy identities
        local cs, xs, us, reps = phs_controlled(eq; hydro = (; Tw = :match))
        @test length(reps.hydro_units) == 1 && reps.hydro_units[1].Tw ≈ Tw
        @test maximum(abs.(phs_rhs(cs, xs, us)) ./ phs_scales(cs, xs)) < 1e-10
        local L0 = phs_ledger(cs, xs, us)
        local kh = findall(==(:hydro), L0.kind)
        @test length(kh) == 1
        @test L0.energy[kh[1]] ≈ un.GMAX * Tw * (Pm / un.GMAX)^2 / 2 rtol = 1e-8    # the grid's power is 2.5 to 1e-10
        local y = copy(xs); y[cs.goff[findfirst(g -> g isa PHHydroUnit, cs.gov)]+3] *= 1.03
        local Ly = phs_ledger(cs, y, us)
        @test Ly.rate[kh] ≈ Ly.port[kh] rtol = 1e-10
        @test Ly.H ≈ phs_energy(cs, y) rtol = 1e-13
        # the ledger against the assembled vector field (consultation H2, point 5): the unit's rate is
        # GMAX Tw q q' along the grid's own dx, and the total is grad(H) . dx
        local jh = findfirst(g -> g isa PHHydroUnit, cs.gov)
        local oh = cs.goff[jh]
        local dxy = phs_rhs(cs, y, us)
        local Ld = phs_ledger(cs, y, us; dx = dxy)
        @test Ld.rate[kh[1]] ≈ cs.units[jh] * cs.gov[jh].GMAX * cs.gov[jh].Tw * y[oh+3] * dxy[oh+3] rtol = 1e-12
        @test Ld.dH ≈ dot(ForwardDiff.gradient(z -> phs_energy(cs, z), y), dxy) rtol = 1e-9
        # the limiter decisions replayed give the same vector field
        local md = phs_modes(cs, y, us)
        @test PHPS._frozen_rhs!(zeros(cs.n), cs, y, us, md) == phs_rhs(cs, y, us)
        # the governor rows are IEEEG3's own, inside and beyond the limits (consultation H2, point 6)
        local p3 = PHPS.params(g3)
        for (xg, w) in (([0.01, 2.4, 2.6, 0.31], 1.002), ([1.0, 2.0, 2.6, 0.31], 0.99), ([-1.0, 3.0, 2.6, 0.31], 1.01),
                        ([0.0, 2.5, 9.0, 0.31], 1.0), ([0.05, 2.5, 8.5, 0.31], 0.98))
            local d1 = zeros(4); local d2 = zeros(5)
            local l1 = PHPS.ModeLog(); local l2 = PHPS.ModeLog()
            try
                PHPS._hydro_rhs!(d1, un, xg, w, Pref, l1)
            catch err
                err isa DomainError || rethrow()
            end
            PHPS._step!(d2, nothing, g3, vcat(xg[1:3], 2.5, 1000.0), [w, Pref, 0.0], p3, l2)
            @test d1[1:3] ≈ d2[1:3] rtol = 1e-14 atol = 1e-15
            @test l1.decisions == l2.decisions[1:length(l1.decisions)]
        end
        @test_throws ArgumentError PHHydroUnit(g3; Tw = big"1e400")
        local r = phs_simulate(cs, xs, us; tstop = 0.004, faults = [PHFault(16, 0.001, 0.001, 0.002)], tol = 1e-14)
        local kr = findall(==(:hydro), r.kind)
        @test all(r.scheme_max[kr] .<= 1e-10 .* r.throughput[kr])
        @test all(abs.(r.closure[kr]) .<= 1e-10 .* r.moved[kr] .+ 1e-12 .* r.throughput[kr])
    end

    @testset "grid-forming converters, the DLL's EMT path (PH_CONVERTER_MODEL.md, sections 3, 4)" begin
        scv = load_scenario(joinpath(ROOT, "cases", "IEEE39Bus_PF_gfm-vsm-pf", "ph_fault_bus16.json"))
        eqv = solve_equilibrium(load_case(scv.system_path), scv)
        cs, xv, uv, rv = phs_controlled(eqv; loads = :series_all)
        sv = cs.s
        # PowerFactory's placement: bus 31 a branch of its own at an inductor-only node, the others
        # merged with their step-up transformers; bus 34's two units one converter
        @test [cv.bus for cv in sv.conv] == [31, 33, 34, 36, 37, 38]
        @test [cv.merged for cv in sv.conv] == [false, true, true, true, true, true]
        @test sv.conv[3].units == 2 && sv.conv[3].Lc ≈ sv.conv[3].comp.p.Zseries / 2
        @test sv.cutof[sv.conv[1].k] > 0
        # the equilibrium: the RMS case's voltages, u_c = E = v_k + j L_c i_c, the divider off
        @test rv.residual < 1e-12 && rv.residual_controlled < 1e-12 && rv.dVgen < 1e-8 && rv.dV < 1e-8
        @test all(c -> c.emf_vs_terminal < 1e-12 && c.divider == 0 && !c.engaged, rv.converters)
        # the power each converter delivers is the RMS case's (its p_set)
        for (i, cv) in enumerate(sv.conv)
            @test cs.conv[i].p_set ≈ cv.units * cv.comp.p.p_set rtol = 1e-6
        end
        # away from the equilibrium: the terminal voltage recovered along the vector field is
        # alpha u_c + w, and the DLL's divider at that terminal voltage returns the same u_c
        local rng = MersenneTwister(31)
        x1 = copy(xv)
        x1[1:sv.n] .+= 0.02 .* randn(rng, sv.n) .* max.(abs.(xv[1:sv.n]), 1e-3)
        for (i, o) in enumerate(cs.coff)
            x1[o] += 0.3; x1[o+1] += 0.01; x1[o+3] *= 1.2      # into the virtual impedance's range
        end
        up = phs_control_inputs(cs, x1, uv.gf)
        f1, d1 = PHPS.phs_flows(sv, x1, up), PHPS.phs_rhs(sv, x1, up)
        engaged = 0
        for (i, cv) in enumerate(sv.conv)
            g = cs.conv[i]
            vk = PHPS._conv_terminal(sv, cv, f1, d1)
            w = PHPS._conv_w(sv, cv, x1, f1)
            @test abs(vk - (cv.alpha * up.uc[i] + w)) < 1e-12 * max(1, abs(vk))
            E = PHPS._vsm_emf(view(x1, cs.coff[i]:cs.coff[i]+3))
            ur, ui = PHPS._vi_divider(PHPS.NoModes(), real(E), imag(E), real(vk), imag(vk), g.Zs, g.i_lim,
                                         g.r_vi, g.x_vi, g.kpr, g.kpx)
            @test abs(complex(ur, ui) - up.uc[i]) < 1e-9
            engaged += vsm_divider(g, E, cv.alpha, w).on
        end
        @test engaged >= 1                                     # the check covered the engaged branch
        # the energy account: every component balances, the converters' EMFs are the supply
        L = phs_ledger(cs, x1, uv)
        @test maximum(abs, L.rate .- (L.port .- L.loss)) < 1e-9 * max(1, maximum(L.tport))
        @test abs(L.interconnection) < 1e-9 * sum(L.tport)
        @test abs(L.dH - (L.supply - L.losses)) < 1e-9 * sum(L.tport)
        # the divider's derivative: finite where E = v_k, and the implicit one across the threshold
        g, cv = cs.conv[2], sv.conv[2]
        w0 = complex(0.3, 0.8)
        Ez = w0 / (1 - cv.alpha)                               # (1 - alpha) E - w = 0
        J = ForwardDiff.jacobian(z -> (q = vsm_divider(g, complex(z[1], z[2]), cv.alpha, w0).uc; [real(q), imag(q)]),
                                 [real(Ez), imag(Ez)])
        @test all(isfinite, J)
        for scale in (0.5, 1.3, 3.0)                           # below, just above and well above i_lim
            D = scale * g.i_lim * g.Zs
            Eh = (w0 + D) / (1 - cv.alpha)
            F(z) = (q = vsm_divider(g, complex(z[1], z[2]), cv.alpha, w0).uc; [real(q), imag(q)])
            Jd = ForwardDiff.jacobian(F, [real(Eh), imag(Eh)])
            h = 1e-6
            Jf = hcat((F([real(Eh) + h, imag(Eh)]) - F([real(Eh) - h, imag(Eh)])) / 2h,
                      (F([real(Eh), imag(Eh) + h]) - F([real(Eh), imag(Eh) - h])) / 2h)
            @test maximum(abs, Jd - Jf) < 1e-5
        end
        # the predicted current solves the eliminated magnitude equation
        Eh = (w0 + 2.0 * g.i_lim * g.Zs) / (1 - cv.alpha)
        dd = vsm_divider(g, Eh, cv.alpha, w0)
        b = 1 - cv.alpha
        ov = abs(dd.ip) - g.i_lim
        @test dd.on && abs(abs(dd.ip) * abs(complex(b * (g.r_vi + g.kpr * ov), b * (g.x_vi + g.kpx * ov) + g.Zs)) -
                           abs(b * Eh - w0)) < 1e-9
        # refusals: parameters outside the DLL build this port covers
        function bad(; kw...)
            c = sv.conv[2]
            P = PHPS.GfmVsmParams
            p = P((get(kw, f, getfield(c.comp.p, f)) for f in fieldnames(P))...)
            return PHConverter(c.name, c.bus, c.k, c.row, c.pos, c.merged, c.Lc, c.alpha, c.units,
                               PHPS.GFM_VSM_PHTRUE(c.comp.name, c.comp.bus, p, c.comp.params))
        end
        @test_throws ArgumentError PHVsm(bad(kpr = -1.0); p_set = 1, u_set = 1)
        @test_throws ArgumentError PHVsm(bad(Tv = 0.0); p_set = 1, u_set = 1)
        @test_throws ArgumentError PHVsm(bad(K_field = 1.0); p_set = 1, u_set = 1)
        @test_throws ArgumentError PHVsm(bad(pf_frame = 1.0); p_set = 1, u_set = 1)
        @test_throws ArgumentError phs_matrices(sv, xv[1:sv.n], phs_control_inputs(cs, xv, uv.gf))
        # alpha = 0: with parallel loads bus 31 has a conductance, its voltage does not depend on u_c
        cp, xp, up0, rp = phs_controlled(eqv)
        @test cp.s.conv[1].alpha == 0 && !cp.s.conv[1].merged && rp.residual_controlled < 1e-12
        xq = copy(xp); xq[cp.coff[1]] += 0.2
        uq = phs_control_inputs(cp, xq, up0.gf)
        fq, dq = PHPS.phs_flows(cp.s, xq, uq), PHPS.phs_rhs(cp.s, xq, uq)
        @test abs(PHPS._conv_terminal(cp.s, cp.s.conv[1], fq, dq) - PHPS._conv_w(cp.s, cp.s.conv[1], xq, fq)) < 1e-12
        @test abs(fq.v[cp.s.conv[1].k] - PHPS._conv_w(cp.s, cp.s.conv[1], xq, fq)) < 1e-12
        # an operating point with the virtual impedance engaged is refused (i_lim below the current)
        dir = mktempdir()
        js = JSON3.read(read(scv.system_path, String), Dict{String,Any})
        js["components"]["GFM_VSM_31"]["params"]["i_lim"] = 1.0
        write(joinpath(dir, "system.json"), JSON3.write(js))
        jsc = JSON3.read(read(joinpath(ROOT, "cases", "IEEE39Bus_PF_gfm-vsm-pf", "ph_fault_bus16.json"), String), Dict{String,Any})
        jsc["system"] = "system.json"
        write(joinpath(dir, "scenario.json"), JSON3.write(jsc))
        local scl = load_scenario(joinpath(dir, "scenario.json"))
        @test_throws ArgumentError phs_controlled(solve_equilibrium(load_case(scl.system_path), scl); loads = :series_all)
        # the DLL's sampled execution (Ts = 50 us; record sections 10, 11)
        local Ts = 5e-5
        css, xs0, us0, rs = phs_controlled(eqv; loads = :series_all, converter_Ts = Ts)
        @test PHPS._sample_period(css) == Ts && all(g -> nstates(g) == 6, css.conv)
        @test_throws ArgumentError phs_simulate(css, xs0, us0; tstop = 2Ts, grid = Ts)
        @test_throws ArgumentError phs_simulate_bdf(css, xs0, us0; tstop = 2Ts, saveat = [Ts])
        # from the continuous equilibrium: not a rest point of the sampled system (the hold lags the output
        # by half a sample on average; PowerFactory settles 100 ms before t = 0 for the same reason), but
        # close to one; between calls the held output keeps its magnitude and turns at -omega_b
        r0 = phs_simulate(css, xs0, us0; tstop = 4Ts, h = Ts / 4, save_dt = Ts / 4, tol = 1e-14)
        for (i, o) in enumerate(css.coff)
            @test maximum(abs, r0.X[o+1, :] .- 1) < 1e-5
            for k in 2:4                                   # t = h, 2h, 3h: before the first call at Ts
                @test abs(complex(r0.X[o+4, k], r0.X[o+5, k]) - us0.uc[i] * cis(-css.s.omega_b * r0.t[k])) < 1e-7
            end
        end
        # a call: the held output is the DLL's divider (Python PHPS's mirror) at the measured terminal voltage, from the
        # states before the call; the states then take the Euler step
        xd = copy(xs0)
        for o in css.coff
            xd[o] += 0.4; xd[o+1] += 0.01                      # into the virtual impedance's range
        end
        upd = phs_control_inputs(css, xd, us0.gf)
        fd = PHPS.phs_flows(css.s, xd, upd)
        xa = PHPS._sample!(css, copy(xd), us0)
        for (i, cv) in enumerate(css.s.conv)
            g, o = css.conv[i], css.coff[i]
            vk = cv.alpha * upd.uc[i] + PHPS._conv_w(css.s, cv, xd, fd)
            E = xd[o+3] * cis(xd[o])
            ur, ui = PHPS._vi_divider(PHPS.NoModes(), real(E), imag(E), real(vk), imag(vk), g.Zs, g.i_lim,
                                         g.r_vi, g.x_vi, g.kpr, g.kpx)
            @test abs(complex(xa[o+4], xa[o+5]) - complex(ur, ui)) < 1e-9
            p = real(vk * conj(-fd.ib[cv.pos]))
            @test xa[o+1] ≈ xd[o+1] + Ts * ((g.p_set - p) - g.Dp * (xd[o+2] - 1)) / g.Ta
            @test xa[o] ≈ xd[o] + Ts * (xd[o+1] * g.omega_n - css.s.omega_b)
        end
        # both ways of saving give the state after the call at a call instant; the energy account holds across
        # the calls (the Simpson diagnostic with the step's own end, C3 finding 1)
        ra = phs_simulate(css, xd, us0; tstop = 4Ts, h = Ts / 4, sample_at = collect(0:Ts:4Ts), tol = 1e-14)
        rb = phs_simulate(css, xd, us0; tstop = 4Ts, h = Ts / 4, save_dt = Ts, tol = 1e-14)
        @test ra.t ≈ rb.t && maximum(abs, ra.X .- rb.X) < 1e-12
        @test all(rb.scheme_max .<= 1e-10 .* rb.throughput .+ 1e-14)
        @test abs(rb.total.simpson) < 1e-6 * rb.total.throughput
        # the physical converter (S8.3, record section 12): battery, DC link, averaged bridge
        dcd = (; H = 5e-3, headroom = 1.15, drop = 0.05, G = 0.0, m_max = 2 / sqrt(3), uk = 0.1)
        @test_throws ArgumentError phs_controlled(eqv; loads = :series_all, converter_dc = (; dcd..., mode = :held))
        @test_throws ArgumentError phs_controlled(eqv; loads = :series_all, converter_Ts = Ts,
                                                  converter_dc = (; dcd..., mode = :held, headroom = 0.9))
        @test_throws ArgumentError phs_controlled(eqv; loads = :series_all, converter_Ts = Ts,
                                                  converter_dc = (; dcd..., mode = :other))
        for mode in (:held, :continuous)
            cp2, xp2, up2, _ = phs_controlled(eqv; loads = :series_all, converter_Ts = Ts, converter_dc = (; dcd..., mode))
            @test length(PHPS._dc_converters(cp2)) == 6 && count(==(:converter_dc), PHPS._ckind(cp2)) == 6
            # at the operating point: the DC rows rest, the bridge applies the equilibrium EMF, the selected data
            d2 = phs_rhs(cp2, xp2, up2)
            for (i, o) in enumerate(cp2.coff)
                g, q = cp2.conv[i], cp2.conv[i].dc
                @test abs(d2[o+6]) < 1e-10
                @test abs(PHPS._bridge_voltage(g, view(xp2, o:o+6)) - up2.uc[i]) < 1e-12
                @test q.c * q.v_dc0^2 / (2 * cp2.s.omega_b) ≈ 5e-3 * q.Sr
                @test q.Sr ≈ 0.1 / cp2.s.conv[i].Lc && q.v_dc0 ≈ 1.15 / q.m_max
            end
            # away from it: every component balances, the batteries are the converters' supply, the bridge cancels
            x2 = copy(xp2)
            for o in cp2.coff
                x2[o] += 0.3; x2[o+6] *= 0.97
            end
            L2 = phs_ledger(cp2, x2, up2)
            @test maximum(abs, L2.rate .- (L2.port .- L2.loss)) < 1e-10 && abs(L2.interconnection) < 1e-10
            @test abs(L2.dH - (L2.supply - L2.losses)) < 1e-10
            @test phs_energy(cp2, x2) ≈ L2.H
            # along the vector field itself (an independent check of the DC row, P2 finding 1)
            Lf = phs_ledger(cp2, x2, up2; dx = phs_rhs(cp2, x2, up2))
            kdc = findall(==(:converter_dc), Lf.kind)
            @test maximum(abs, Lf.rate[kdc] .- (Lf.port[kdc] .- Lf.loss[kdc])) < 1e-10
            @test_throws DomainError PHPS._bridge_voltage(cp2.conv[1], [0, 1, 1, 1, 1.0, 0, -0.1])
            # in saturation (a command beyond the bridge's range): the bridge holds |m| = m_max, the discrete balance holds
            # (reverse power, the battery charging, occurs in the 1 s fault runs, record section 12)
            x4 = copy(xp2)
            for o in cp2.coff
                x4[o+3] *= 1.6
            end
            r4 = phs_simulate(cp2, x4, up2; tstop = 3Ts, h = Ts / 4, save_dt = Ts, tol = 1e-14)
            @test all(r4.scheme_max .<= 1e-10 .* r4.throughput .+ 1e-14)
            @test all(phs_converter_signals(cp2, r4.X[:, end], up2).m .≈ cp2.conv[1].dc.m_max)
            # a call: the modulation is the command over the DC voltage sampled at the call, clipped radially
            if mode === :held
                o, g, cv = cp2.coff[2], cp2.conv[2], cp2.s.conv[2]
                x3 = copy(x2); x3[o+3] *= 1.6                 # a command beyond the bridge's range
                up3 = phs_control_inputs(cp2, x3, up2.gf)
                f3 = PHPS.phs_flows(cp2.s, x3, up3)
                vk = cv.alpha * up3.uc[2] + PHPS._conv_w(cp2.s, cv, x3, f3)
                ucmd = vsm_divider(g, x3[o+3] * cis(x3[o]), 0.0, vk).uc
                xa3 = PHPS._sample!(cp2, copy(x3), up2)
                mnew = complex(xa3[o+4], xa3[o+5])
                @test abs(mnew) ≈ min(abs(ucmd) / x3[o+6], g.dc.m_max)
                @test abs(angle(mnew / ucmd)) < 1e-12
            end
            # a short fault run: the discrete balance of every component, the DC sides included
            r2 = phs_simulate(cp2, xp2, up2; tstop = 4Ts, h = Ts / 4, save_dt = Ts,
                              faults = [PHFault(16, 0.001, Ts, 3Ts)], tol = 1e-14)
            @test all(r2.scheme_max .<= 1e-10 .* r2.throughput .+ 1e-14)
        end
        # the controller's output-voltage limit (a control law, the DLL variant gfm_vsm_pf_vlim; record section 14)
        @test_throws ArgumentError phs_controlled(eqv; loads = :series_all, converter_umax = 1.15)
        @test_throws ArgumentError phs_controlled(eqv; loads = :series_all, converter_Ts = Ts, converter_umax = 1.0)
        csu, xsu, usu, _ = phs_controlled(eqv; loads = :series_all, converter_Ts = Ts, converter_umax = 1.15)
        @test all(g -> g.u_max == 1.15, csu.conv)
        xu = copy(xsu)
        for o in csu.coff
            xu[o+3] *= 1.6                                  # a command beyond the limit
        end
        upu = phs_control_inputs(csu, xu, usu.gf)
        fu = PHPS.phs_flows(csu.s, xu, upu)
        xau = PHPS._sample!(csu, copy(xu), usu)
        for (i, cv) in enumerate(csu.s.conv)
            g, o = csu.conv[i], csu.coff[i]
            vk = cv.alpha * upu.uc[i] + PHPS._conv_w(csu.s, cv, xu, fu)
            ucmd = vsm_divider(g, xu[o+3] * cis(xu[o]), 0.0, vk).uc
            held = complex(xau[o+4], xau[o+5])
            @test abs(held) ≈ min(abs(ucmd), 1.15) && abs(angle(held / ucmd)) < 1e-12
        end
        # below the limit the limited controller is the unlimited one
        ru = phs_simulate(csu, xsu, usu; tstop = 4Ts, h = Ts / 4, save_dt = Ts, tol = 1e-14)
        rnl = phs_simulate(css, xs0, us0; tstop = 4Ts, h = Ts / 4, save_dt = Ts, tol = 1e-14)
        @test maximum(abs, ru.X .- rnl.X) < 1e-12
        # with continuous normalisation and the bridge unclipped, the physical converter is the ideal source
        cpc, xpc, upc, _ = phs_controlled(eqv; loads = :series_all, converter_Ts = Ts, converter_dc = (; dcd..., mode = :continuous))
        rc = phs_simulate(cpc, xpc, upc; tstop = 4Ts, h = Ts / 4, save_dt = Ts, tol = 1e-14)
        ri = phs_simulate(css, xs0, us0; tstop = 4Ts, h = Ts / 4, save_dt = Ts, tol = 1e-14)
        @test maximum(abs, rc.X[1:cpc.s.n, :] .- ri.X[1:css.s.n, :]) < 1e-9
        # a short fault run: the discrete energy balance of every component
        r = phs_simulate(cs, xv, uv; tstop = 0.004, faults = [PHFault(16, 0.001, 0.001, 0.002)], tol = 1e-14)
        @test all(r.scheme_max .<= 1e-10 .* r.throughput .+ 1e-14)
        @test maximum(abs, r.X[cs.coff[1]+1, :] .- 1) > 0     # the converters moved
    end

    @testset "declared records, and a run written to a file and read back" begin
        cs, xc, uc, _ = phs_controlled(eq)
        rec = ("efd" => (x, u) -> phs_control_signals(cs, x).Efd,
               "speed_1" => (x, u) -> phs_machine_signals(cs.s, x)[1].speed)
        r = phs_simulate(cs, xc, uc; tstop = 0.004, faults = [PHFault(16, 0.001, 0.001, 0.002)], save_dt = 1e-3,
                         record = rec, tol = 1e-14)
        nt = length(r.t)
        # a record of vectors is a matrix with a column per saved time, a record of numbers a vector
        @test size(r.records["efd"]) == (length(cs.s.machines), nt) && length(r.records["speed_1"]) == nt
        # each column is the quantity at the saved state
        for k in (1, nt ÷ 2, nt)
            @test r.records["efd"][:, k] == phs_control_signals(cs, r.X[:, k]).Efd
            @test r.records["speed_1"][k] == phs_machine_signals(cs.s, r.X[:, k])[1].speed
        end
        # nothing declared: no records, and the run is the same
        r0 = phs_simulate(cs, xc, uc; tstop = 0.004, faults = [PHFault(16, 0.001, 0.001, 0.002)], save_dt = 1e-3,
                          tol = 1e-14)
        @test isempty(r0.records) && r0.X == r.X

        path = joinpath(mktempdir(), "run.jld2")
        @test phs_write_run(path, r; model = cs, meta = Dict("what" => "a test", :n => 3)) == path
        d = phs_read_run(path)
        @test d["t"] == r.t && d["X"] == r.X && d["V"] == r.V && d["energy"] == r.energy
        @test d["records"]["efd"] == r.records["efd"] && d["records"]["speed_1"] == r.records["speed_1"]
        @test d["state_names"] == phs_state_names(cs) && length(d["bus_ids"]) == size(r.V, 1)
        @test d["names"] == r.names && d["kind"] == String.(r.kind) && d["supply"] == r.supply
        @test d["total"]["scheme_max"] == r.total.scheme_max && length(d["segments"]) == length(r.segments)
        @test d["meta"]["what"] == "a test" && d["meta"]["n"] == 3
        @test d["git_commit"] isa String && d["git_dirty"] isa Bool
        # without the state, everything else is kept
        path2 = joinpath(mktempdir(), "small.jld2")
        phs_write_run(path2, r; model = cs, states = false)
        d2 = phs_read_run(path2)
        @test size(d2["X"]) == (0, nt) && d2["records"]["efd"] == r.records["efd"] && d2["V"] == r.V
    end

    @testset "three phases from the dq phasor" begin
        # the inverse of the specification's dq0 transformation, with the zero axis at zero
        z, th = complex(0.7, -0.4), 1.234
        abc = collect(phs_phases(z, th))
        al = 2pi / 3
        P = (2 / 3) .* [cos(th) cos(th - al) cos(th + al); -sin(th) -sin(th - al) -sin(th + al);
                        1/sqrt(2) 1/sqrt(2) 1/sqrt(2)]
        @test P * abc ≈ [real(z), imag(z), 0.0] atol = 1e-15
        @test (2 / 3) * (abc[1] + cis(al) * abc[2] + cis(2al) * abc[3]) ≈ cis(th) * z   # the space vector
        @test sum(abc) ≈ 0 atol = 1e-15
        # per-unit three-phase power on the three-phase base: (2/3) sum of v i = Re(conj(v) i)
        zi = complex(-0.3, 0.9)
        @test (2 / 3) * sum(abc .* collect(phs_phases(zi, th))) ≈ real(conj(z) * zi)
    end

    @testset "fault run: energy balance, order, independent solver" begin
        faults = [PHFault(16, 0.001, 0.004, 0.008)]
        r1, r2, r4 = [phs_simulate(s, x0, u; tstop = 0.012, faults, hscale = hs, save_dt = 0.004)
                      for hs in (1.0, 0.5, 0.25)]
        @test [g.fault for g in r1.segments] == [false, true, false]
        @test r1.segments[2].h < r1.segments[1].h          # the fault's decay sets the step
        @test r1.t[end] ≈ 0.012 && r1.X[:, 1] == x0
        @test abs(r1.V[findfirst(==(16), ids), 3]) < 0.2   # at 0.008 s, the end of the fault
        # scheme identity (spec section 8): system and every component, at its worst along the run
        @test r1.total.scheme_max < 1e-10 * r1.total.throughput
        @test all(r1.scheme_max .<= 1e-10 .* r1.throughput)
        @test all(r1.throughput .> 0)
        # the summed increments equal the change of each component's energy function: the
        # energies are quadratic forms
        emax = vec(maximum(abs, r1.energy, dims = 2))
        @test all(abs.(r1.closure) .<= 1e-12 .* (emax .+ r1.moved))
        # accumulated defect, components with storage
        st = [k for k in eachindex(r1.kind) if r1.kind[k] != :fault]
        @test all(abs.(r1.simpson[st]) .<= 1e-2 .* max.(r1.moved[st], 1e-9 .* r1.throughput[st]))
        @test abs(r1.total.simpson) < 1e-2 * r1.total.moved
        # the fault dissipates and the rotors gain: energy by component adds up at every saved time
        @test [phs_energy(s, r1.X[:, k]) for k in eachindex(r1.t)] ≈ r1.H rtol = 1e-13
        @test all(>=(0), r1.loss)
        # second order
        d1 = maximum(abs.(r1.X[:, end] .- r2.X[:, end]) ./ scl)
        d2 = maximum(abs.(r2.X[:, end] .- r4.X[:, end]) ./ scl)
        @test 1.8 < log2(d1 / d2) < 2.2
        # independent BDF solution of the same model: the midpoint error falls fourfold per
        # halving towards it, and the extrapolated midpoint solution agrees with it
        _, Xb = phs_simulate_bdf(s, x0, u; tstop = 0.012, faults, saveat = [0.012])
        e1, e2, e4 = [maximum(abs.(q.X[:, end] .- Xb[:, 1]) ./ scl) for q in (r1, r2, r4)]
        @test e1 < 1e-3
        @test 3.5 < e1 / e2 < 4.5 && 3.5 < e2 / e4 < 4.5
        @test maximum(abs.((4 .* r4.X[:, end] .- r2.X[:, end]) ./ 3 .- Xb[:, 1]) ./ scl) < 1e-6
        # faults the reduced model cannot take
        @test_throws ArgumentError phs_simulate(s, x0, u; tstop = 0.01, faults = [PHFault(30, 0.001, 0.0, 0.01)])
        @test_throws ArgumentError phs_simulate(s, x0, u; tstop = 0.01, faults = [PHFault(16, 0.0, 0.0, 0.01)])
    end
end
