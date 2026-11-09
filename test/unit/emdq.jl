# Electromagnetic dq pilot (gate 1 of docs/records/early/PHYSICAL_EXTENSION.md): the winding map, the
# equilibrium, the exact raw and shifted energy identities, the singular limit, and the
# recorded gate-1.3 finding (indefinite Bregman Hessian, positive rate outside ker y_m).

@testset "emdq pilot (gate 1)" begin
    sc = load_scenario(joinpath(ROOT, "cases", "IEEE39Bus_PF", "no_fault_phtrue.json"))
    eq = solve_equilibrium(load_case(sc.system_path), sc)
    s, xs, rep = emdq_pilot(eq)
    M = s.mach

    # winding map: the Sauer-Pai definitions reproduce the record
    xd1 = M.Ll + M.Lad * M.Lfd / (M.Lad + M.Lfd)
    xd2 = M.Ll + 1 / (1 / M.Lad + 1 / M.Lfd + 1 / M.L1d)
    @test xd1 - 0.0181 ≈ 0.031 rtol = 1e-6
    @test xd2 - 0.0181 ≈ 0.025 rtol = 1e-6
    @test (M.Lfd + M.Lad) / (M.omega_b * M.Rfd) ≈ 10.2 rtol = 1e-6
    @test isposdef(Symmetric(inv(M.Kd))) && isposdef(Symmetric(inv(M.Kq)))

    # gate 1.1: exact equilibrium at m*, case dispatch at the machine bus
    @test maximum(abs, emdq_rhs(s, xs)) < 1e-9
    @test rep.P ≈ 2.5
    @test abs(rep.Vg) ≈ 1.0475 rtol = 1e-7   # the case value is 1.04750001
    @test abs(rep.q_axis_residual) < 1e-14
    @test 0 < s.mstar < 1

    rng = MersenneTwister(3)
    for _ in 1:4
        x = xs .+ 0.05 .* randn(rng, length(xs)) .* max.(abs.(xs), 0.1)
        # gate 1.2: raw balance, any modulation (the converter is lossless)
        @test abs(emdq_balance(s, x, 0.3)["identity_error"]) < 1e-13
        # gates 1.3/1.4: shifted identity with the single rotation residual and u y_m
        sb = emdq_shifted_balance(s, xs, x, 0.6)
        @test abs(sb["identity_error"]) < 1e-13
        # y_m is the shifted output g_m' grad H_s; the raw output g_m' grad H is zero
        g = ForwardDiff.gradient(z -> emdq_energy(s, z), x)
        gs = ForwardDiff.gradient(z -> emdq_energy(s, z), xs)
        gm = ForwardDiff.derivative(m -> emdq_rhs(s, x, m), 0.4)
        @test abs(dot(gm, g)) < 1e-12
        @test dot(gm, g - gs) ≈ sb["y_m"] rtol = 1e-10 atol = 1e-14
        # singular limit: exact Sauer-Pai reduction
        @test emdq_reduction_check(s, x).err_sauer_pai < 1e-12
    end
    @test maximum(abs, values(emdq_reduction_check(s, xs).dev_phtrue)) < 1e-12

    # gate 1.3 finding at the case operating point (recorded in TODO.md)
    F = emdq_local_forms(s, xs)
    @test F.closure < 1e-12
    @test count(<(0), eigvals(Symmetric(F.Hs))) == 1
    @test restricted_max_eig(F.Q, F.c) > 1.0
end

@testset "emdq angle augmentation (route b)" begin
    sc = load_scenario(joinpath(ROOT, "cases", "IEEE39Bus_PF", "no_fault_phtrue.json"))
    eq = solve_equilibrium(load_case(sc.system_path), sc)
    s, xs, _ = emdq_pilot(eq; Dm = 2.5)
    rng = MersenneTwister(11)
    for _ in 1:3
        x = xs .+ 0.05 .* randn(rng, length(xs)) .* max.(abs.(xs), 0.1)
        zr = dot(ForwardDiff.gradient(z -> emdq_angle_augmentation(s, xs, z; sigma = 0.3, c = 0.7), x),
                 emdq_rhs(s, x))
        @test zr ≈ emdq_angle_augmentation_rate(s, xs, x; sigma = 0.3, c = 0.7) atol = 1e-12
        @test abs(emdq_balance(s, x)["identity_error"]) < 1e-13          # with shaft damping
        @test abs(emdq_shifted_balance(s, xs, x, 0.6)["identity_error"]) < 1e-13
    end
    F = emdq_local_forms(s, xs)
    a = emdq_augmented_forms(s, xs, F, 1e-3)
    @test a.c ≈ 1e-3 * s.Dm / s.mach.omega_b rtol = 1e-8   # cancels the angle-speed entry
    @test abs(a.QV[6, 7]) < 1e-12
    # structural obstruction: positive rate on {Ddelta = Dp = 0} and ker y_m
    keep = setdiff(1:length(xs), (6, 7))
    Ek = Matrix{Float64}(I, length(xs), length(xs))[:, keep]
    N = Ek * nullspace(reshape(F.c' * Ek, 1, :))
    @test eigmax(Symmetric(N' * F.Q * N)) > 1.0
    @test N' * a.QV * N ≈ N' * F.Q * N
end

@testset "emdq AVR closed loop and the local-block obstruction (step 5)" begin
    sc = load_scenario(joinpath(ROOT, "cases", "IEEE39Bus_PF", "no_fault_phtrue.json"))
    eq = solve_equilibrium(load_case(sc.system_path), sc)
    s, xs, _ = emdq_pilot(eq)
    c = emdq_avr(s, xs; kappa_p = 1.0, omega_i = 1.0)
    @test c.u0 == 0.5
    rng = MersenneTwister(4)
    for _ in 1:3
        z = vcat(xs .+ 0.02 .* randn(rng, 18) .* max.(abs.(xs), 0.1), 0.1 * randn(rng))
        @test abs(emdq_closed_loop_balance(s, xs, c, z)["identity_error"]) < 1e-13   # gate 1.5
    end
    @test maximum(abs, emdq_closed_loop_rhs(s, xs, c, vcat(xs, 0.0))) < 1e-9
    # the structural obstruction of the local block, the same for any gains
    rates = [emdq_local_obstruction(emdq_completion_data(s, xs, emdq_avr(s, xs; kappa_p = k, omega_i = w))).max_rate
             for (k, w) in ((1.0, 1.0), (0.01, 100.0), (100.0, 0.1))]
    @test all(>(100.0), rates)
    @test maximum(rates) - minimum(rates) < 1e-6 * maximum(rates)
    d = emdq_completion_data(s, xs, c)
    o = emdq_local_obstruction(d)
    # on K, any local correction leaves the rate unchanged
    F = zeros(19, 19)
    L = [1:7; 19]
    F[L, L] = Symmetric(randn(rng, 8, 8))
    v = o.direction
    rate(P) = v' * (d.As' * P + P * d.As) * v / 2
    @test rate(d.Hs + F) ≈ rate(d.Hs) rtol = 1e-9
    @test rate(d.Hs) ≈ o.max_rate rtol = 1e-9
end

@testset "emdq obstruction: interval proof and the stator-port pattern" begin
    sc = load_scenario(joinpath(ROOT, "cases", "IEEE39Bus_PF", "no_fault_phtrue.json"))
    eq = solve_equilibrium(load_case(sc.system_path), sc)
    s, xs, _ = emdq_pilot(eq)
    d = emdq_completion_data(s, xs, emdq_avr(s, xs; kappa_p = 1.0, omega_i = 1.0))
    @test count(d.port_mask) == 64 + 2 * 2 * 8        # local block and its bus-2 cross-terms
    @test !any(d.port_mask[9:10, 9:10])               # the capacitor's own block stays physical
    for M in (d.local_mask, d.port_mask)
        o = emdq_local_obstruction(d; mask = M)
        @test o.max_rate > 50
        w = o.direction[setdiff(1:19, o.R)]
        w ./= maximum(abs, w)
        p = emdq_verify_obstruction(d, w; mask = M, rel = 1e-6, abs_A = 1e-10, abs_H = 1e-12)
        @test p.proved && p.beta < 1
        # a correction on the pattern does not change the rate along the witness
        rng = MersenneTwister(9)
        F = Symmetric(randn(rng, 19, 19)) .* M
        v = o.direction
        @test v' * (d.As' * F + F * d.As) * v ≈ 0 atol = 1e-8 * norm(F)
    end
end

@testset "emdq round-rotor pilot (gate 2, part 1: GENROU_2)" begin
    sc = load_scenario(joinpath(ROOT, "cases", "IEEE39Bus_PF", "no_fault_phtrue.json"))
    eq = solve_equilibrium(load_case(sc.system_path), sc)
    s, xs, rep = emdq_pilot(eq; machine = "GENROU_2", trf = "Trf_2", line = "Line_11", tie = "Line_13")
    M = s.mach
    @test M.nq == 2 && length(xs) == 19 && emdq_state_names(M)[19] == "psi_2q"
    # the q-axis map reproduces the record (system base; Trf_2 leakage 0.025 merged)
    xq1 = M.Ll + M.Laq * M.L1q / (M.Laq + M.L1q)
    xq2 = M.Ll + 1 / (1 / M.Laq + 1 / M.L1q + 1 / M.L2q)
    gp = eq.sys.comps[findfirst(c -> PHPS.name(c) == "GENROU_2", eq.sys.comps)].p
    @test xq1 - 0.025 ≈ gp.xq_prime rtol = 1e-6
    @test xq2 - 0.025 ≈ gp.xq_double_prime rtol = 1e-6
    @test (M.L1q + M.Laq) / (M.omega_b * M.R1q) ≈ gp.Tq0_prime rtol = 1e-6
    @test (M.L2q + M.Laq * M.L1q / (M.Laq + M.L1q)) / (M.omega_b * M.R2q) ≈ gp.Tq0_double_prime rtol = 1e-6
    @test isposdef(Symmetric(inv(M.Kq)))
    @test maximum(abs, emdq_rhs(s, xs)) < 1e-9
    @test rep.machine_bus_load ≈ 0.092 + 0.046im
    rng = MersenneTwister(21)
    for _ in 1:3
        x = xs .+ 0.05 .* randn(rng, 19) .* max.(abs.(xs), 0.1)
        @test abs(emdq_balance(s, x, 0.3)["identity_error"]) < 1e-13
        @test abs(emdq_shifted_balance(s, xs, x, 0.6)["identity_error"]) < 1e-13
    end
    F = emdq_local_forms(s, xs)
    @test F.closure < 1e-12
    @test count(<(0), eigvals(Symmetric(F.Hs))) == 1
    @test restricted_max_eig(F.Q, F.c) > 1.0
    c = emdq_avr(s, xs; kappa_p = 1.0, omega_i = 1.0)
    z = vcat(xs .+ 0.02 .* randn(rng, 19) .* max.(abs.(xs), 0.1), 0.1)
    @test abs(emdq_closed_loop_balance(s, xs, c, z)["identity_error"]) < 1e-13
    d = emdq_completion_data(s, xs, c)
    @test d.block == [1:7; 19; 20] && d.eta == 20
    for M_ in (d.local_mask, d.port_mask)
        o = emdq_local_obstruction(d; mask = M_)
        w = o.direction[setdiff(1:20, o.R)]
        w ./= maximum(abs, w)
        @test emdq_verify_obstruction(d, w; mask = M_, abs_A = 1e-10, abs_H = 1e-12).proved
    end
end

@testset "emdq no-go certificate, gain lemma and locality radii" begin
    sc = load_scenario(joinpath(ROOT, "cases", "IEEE39Bus_PF", "no_fault_phtrue.json"))
    eq = solve_equilibrium(load_case(sc.system_path), sc)
    s, xs, _ = emdq_pilot(eq)
    rec = emdq_nogo_certificate(eq, s, xs; unit = "GENROU_11", pattern = :local_mask)
    @test rec["proved"] && rec["cross_check_256bit"]["boxes_contain_256bit_data"]
    @test rec["equilibrium"]["krawczyk"]["equilibrium_width_max"] < 1e-10      # model-bound data
    @test rec["box_width_max"]["A"] < 1e-8
    @test rec["rate_lo"] > 0
    back = check_nogo_certificate(rec)                     # from the record alone
    @test back.proved && back.rate_lo ≈ rec["rate_lo"]
    bad = copy(rec)
    bad["pattern_entries"] = vcat(rec["pattern_entries"], [[9, 11]])   # an entry outside R x R
    @test !check_nogo_certificate(bad).proved
    # gain lemma: the witness rate does not depend on the PI gains
    y0 = emdq_avr(s, xs; kappa_p = 1.0, omega_i = 1.0).y0
    r = [emdq_local_obstruction(emdq_completion_data(s, xs, EMDQAVR(k, w, 0.5, y0))).max_rate
         for (k, w) in ((1.0, 1.0), (-30.0, 1e-3), (1e3, 1e3))]
    @test maximum(r) - minimum(r) < 1e-10 * maximum(r)
    d = emdq_completion_data(s, xs, emdq_avr(s, xs; kappa_p = 1.0, omega_i = 1.0))
    # docs/records/early/NOGO_THEOREM.md, Lemma 2: the unscaled witness is the same vector for every gain
    # pair (kp real, wi != 0) with the same rate, and has no integrator component
    o = emdq_local_obstruction(d)
    v, R = d.d .* o.direction, o.R
    rho = v' * d.Hfix * d.A * v
    for (k, w) in ((-50.0, 1e-4), (0.0, 3.0), (1e4, 1e4))
        dk = emdq_completion_data(s, xs, EMDQAVR(k, w, 0.5, y0))
        @test norm((dk.A * v)[R]) < 1e-12 * opnorm(dk.A) * norm(v)
        @test v' * dk.Hfix * dk.A * v ≈ rho rtol = 1e-12
    end
    @test abs(o.direction[d.eta]) < 1e-12
    # Corollary 4: a Krasovskii term 1/2 f'Wf with W on R x R (Hessian A'WA) vanishes on the
    # witness, in value and in rate
    W = zeros(19, 19)
    W[R, R] = Symmetric(randn(MersenneTwister(3), length(R), length(R)))
    PK = d.A' * W * d.A
    @test abs(v' * PK * v) < 1e-12 * opnorm(PK) * norm(v)^2
    @test abs(v' * PK * d.A * v) < 1e-12 * opnorm(PK) * opnorm(d.A) * norm(v)^2
    # nested radii: full blocks, the last is every state; the first four are obstructed
    radii = emdq_radius_masks(s, d)
    @test length(radii) == 5 && all(radii[end][2])
    @test all(count(radii[k][2]) < count(radii[k+1][2]) for k in 1:4)
    for k in 1:4
        @test emdq_local_obstruction(d; mask = radii[k][2]).max_rate > 1
    end
    @test emdq_local_obstruction(d; mask = radii[5][2]).dim_K == 0
end

@testset "emdq Krasovskii gate: block-diagonal weights (section 20, item 2)" begin
    sc = load_scenario(joinpath(ROOT, "cases", "IEEE39Bus_PF", "no_fault_phtrue.json"))
    eq = solve_equilibrium(load_case(sc.system_path), sc)
    s, xs, _ = emdq_pilot(eq)
    blocks = emdq_component_blocks(s)
    mask = emdq_component_mask(s)
    @test length(blocks) == 8 && count(mask[i, j] for j in 1:19 for i in 1:j) == 45
    # the two predeclared coarse partitions (section 22): four blocks each, no trace obstruction
    for (p, nfree) in ((:node_centred, 82), (:edge_centred, 68))
        mp = emdq_component_mask(s; partition = p)
        @test length(emdq_component_blocks(s; partition = p)) == 4
        @test count(mp[i, j] for j in 1:19 for i in 1:j) == nfree
    end
    @test_throws ArgumentError emdq_component_blocks(s; partition = :other)
    c = emdq_avr(s, xs; kappa_p = 1.0, omega_i = 1.0)
    d = emdq_completion_data(s, xs, c)
    # gate question 2: the exact rate identity of H_K for a constant weight
    rng = MersenneTwister(11)
    W = zeros(19, 19)
    for (_, ix) in blocks
        B = randn(rng, length(ix), length(ix))
        W[ix, ix] = B * B' + I
    end
    z = vcat(xs .+ 0.02 .* randn(rng, 18) .* max.(abs.(xs), 0.1), 0.1)
    @test abs(emdq_krasovskii_balance(s, xs, c, d, W, z).identity_error) < 1e-13
    # the block-trace obstruction: the integrator, the bus-2 capacitor and the load inductor
    # have tr A_kk = 0 exactly, for any gains (the gains enter only the psi_fd, q_dc, eta rows)
    for (k, w) in ((1.0, 1.0), (-50.0, 1e-4), (1e4, 1e4))
        dk = emdq_completion_data(s, xs, EMDQAVR(k, w, 0.5, c.y0))
        t = block_trace_obstruction(dk.As, blocks; mu = 1e-3)
        # (a negative kappa_p also anti-damps the DC link)
        @test ["pi_integrator", "bus2_capacitor", "load_inductor"] ⊆ [b.name for b in t if b.obstructs]
        @test dk.A[19, 19] == 0 && tr(dk.A[9:10, 9:10]) == 0 && tr(dk.A[17:18, 17:18]) == 0
    end
    iv = emdq_interval_data(s, xs, c, d)               # model-bound: the same three blocks
    tI = block_trace_obstruction(iv.AsI, blocks; mu = 1e-3)
    @test [b.name for b in tI if b.obstructs] == ["pi_integrator", "bus2_capacitor", "load_inductor"]
    for p in (:node_centred, :edge_centred)
        @test !any(b.obstructs for b in block_trace_obstruction(iv.AsI, emdq_component_blocks(s; partition = p); mu = 1e-3))
    end
    # section 22: the union-spectral test. With Ra = 0 the stator-flux rows inside the
    # edge-centred first block (and the finest machine block) are the frame rotation alone,
    # for any gains, so that block has the eigenvalues +-j omega_b exactly
    e1 = emdq_component_blocks(s; partition = :edge_centred)[1][2]
    @test s.mach.Ra == 0 && all(d.A[1:2, setdiff(e1, [1, 2])] .== 0) && d.A[1, 2] == -d.A[2, 1] && d.A[1, 1] == 0
    @test verified_block_instability(iv.AsI, e1; mu = 1e-3).proved
    @test verified_block_instability(iv.AsI, emdq_machine_block(s.mach); mu = 1e-3).proved
    n1 = emdq_component_blocks(s; partition = :node_centred)[1][2]
    @test block_union_abscissae(d.As, emdq_component_blocks(s; partition = :node_centred))[1].abscissa > 1
    @test verified_block_instability(iv.AsI, n1; mu = 1e-3).proved      # machine on its capacitor
    @test !verified_block_instability(iv.AsI, [11, 12]; mu = 1e-3).proved   # a resistive line is stable
    @test verified_block_instability([0.0 2.0; -2.0 0.0], [1, 2]; mu = 1e-3).proved
    @test !verified_block_instability(-Matrix(1.0I, 3, 3), 1:3; mu = 1e-3).proved
    # the analytic dual certificate of the integrator block passes the Farkas check
    e = zeros(19); e[19] = 1
    r = (; dual_lower = 2e-3 * e * e', dual_upper = zeros(19, 19), dual_rate = e * e')
    f = block_lyapunov_farkas_check(d.As, mask, r; mu = 1e-3)
    @test f.trace_Y1 > 0 && f.pattern == 0 && f.trace_Y2 == 0
    # the interval check of V_W and H_K on a synthetic Hurwitz matrix
    A = [-1.0 2.0 0.0; -2.0 -1.0 1.0; 0.0 0.0 -3.0]
    Wl = lyap(Matrix(A'), Matrix(1.0I, 3, 3))          # A'W + WA = -I
    v = emdq_verify_krasovskii(Wl, PHPS.ival.(A); mu = 1e-3)
    ep = eigvals(Symmetric(A' * Wl * A))
    @test v.proved_VW && v.proved_HK && v.PK_min <= ep[1] && v.PK_max >= ep[end]
    @test !emdq_verify_krasovskii(Wl, PHPS.ival.(-A); mu = 1e-3).proved_VW
end

@testset "emdq phase-free GENROU_2 pilot (section 23, gate G1)" begin
    sc = load_scenario(joinpath(ROOT, "cases", "IEEE39Bus_PF", "no_fault_phtrue.json"))
    eq = solve_equilibrium(load_case(sc.system_path), sc)
    s, xs, _ = emdq_pilot(eq; machine = "GENROU_2", trf = "Trf_2", line = "Line_11", tie = "Line_13")
    b78 = eq.sys.case.lines[findfirst(l -> l.idx == "Line_13", eq.sys.case.lines)].b
    pf, zs, br = emdq_phasefree(s, xs, b78)
    # the derived bus-8 node is passive and reproduces the boundary current at the equilibrium
    @test br.inductive && pf.G8 >= 0 && pf.C8 > 0 && pf.L8 > 0
    @test emdq_pf_nstates(pf) == 23 && length(zs) == 23
    @test maximum(abs, emdq_pf_rhs(pf, zs)) < 1e-9
    c = emdq_avr(s, xs; kappa_p = 1.0, omega_i = 1.0)
    cl = EMDQPFClosedLoop(pf, xs, c)
    rng = MersenneTwister(31)
    z = vcat(zs .+ 0.03 .* randn(rng, 23) .* max.(abs.(zs), 0.1), 0.1)
    phi = 0.7
    # the rotational symmetry, the quotient identity and the raw energy balance (no boundary)
    lhs = emdq_pf_closed_loop_rhs(cl, emdq_pf_rotate(pf, z, phi))
    rhs = emdq_pf_rotate(pf, emdq_pf_closed_loop_rhs(cl, z), phi)
    rhs[6] -= phi
    @test lhs ≈ rhs rtol = 1e-12
    dred = ForwardDiff.jacobian(w -> emdq_pf_reduce(cl, w), z) * emdq_pf_closed_loop_rhs(cl, z)
    @test dred ≈ emdq_pf_reduced_rhs(cl, emdq_pf_reduce(cl, z)) rtol = 1e-12
    @test emdq_pf_reduce(cl, emdq_pf_rotate(pf, z, phi)) ≈ emdq_pf_reduce(cl, z) rtol = 1e-12
    @test abs(emdq_pf_balance(pf, z[1:23], 0.4).identity_error) < 1e-13
    # G1 fails: the quotient (transverse) equilibrium is unstable, a speed mode
    xi = emdq_pf_reduce(cl, vcat(zs, 0.0))
    d = emdq_pf_quotient_data(cl, xi)
    @test maximum(abs, emdq_pf_reduced_rhs(cl, xi)) < 1e-9
    @test 0.02 < d.spectral_abscissa < 0.03
    E = eigen(d.A)
    v = abs.(E.vectors[:, argmax(real.(E.values))] ./ d.d)
    @test d.names[argmax(v)] == "p"
    @test verified_block_instability(d.As, 1:size(d.A, 1); mu = 0.0).proved
    @test length(emdq_pf_masks(cl)) == 8 && all(emdq_pf_masks(cl)[end][2])
end

@testset "emdq phase-free GENROU_2 pilot with IEEEG1_2 (section 23.6, gates G1 and G2)" begin
    sc = load_scenario(joinpath(ROOT, "cases", "IEEE39Bus_PF", "no_fault_phtrue.json"))
    eq = solve_equilibrium(load_case(sc.system_path), sc)
    s, xs, _ = emdq_pilot(eq; machine = "GENROU_2", trf = "Trf_2", line = "Line_11", tie = "Line_13")
    b78 = eq.sys.case.lines[findfirst(l -> l.idx == "Line_13", eq.sys.case.lines)].b
    pf0, zs0, _ = emdq_phasefree(s, xs, b78)
    gc = eq.sys.comps[findfirst(c_ -> PHPS.name(c_) == "IEEEG1_2", eq.sys.comps)]
    pf, zs, gr = emdq_governed(pf0, zs0, gc.p)
    # same torque, inside the equilibrium limiter mode, the same electromagnetic equilibrium
    @test gr.Tm == s.Tm && gr.margins.position > 0 && gr.margins.rate > 0
    @test emdq_pf_nstates(pf) == 29 && zs[1:23] == zs0
    @test maximum(abs, emdq_pf_rhs(pf, zs)) < 1e-9
    c = emdq_avr(s, xs; kappa_p = 1.0, omega_i = 1.0)
    cl = EMDQPFClosedLoop(pf, xs, c)
    rng = MersenneTwister(37)
    z = vcat(zs .+ 0.03 .* randn(rng, 29) .* max.(abs.(zs), 0.1), 0.1)
    lhs = emdq_pf_closed_loop_rhs(cl, emdq_pf_rotate(pf, z, 1.1))
    rhs = emdq_pf_rotate(pf, emdq_pf_closed_loop_rhs(cl, z), 1.1)
    rhs[6] -= 1.1
    @test lhs ≈ rhs rtol = 1e-12
    @test abs(emdq_pf_balance(pf, z[1:29], 0.55).identity_error) < 1e-13   # shaft power from the governor
    # G1: the governed transverse equilibrium is stable
    xi = emdq_pf_reduce(cl, vcat(zs, 0.0))
    d = emdq_pf_quotient_data(cl, xi)
    @test d.spectral_abscissa < -1e-3
    # the masks: one joint mechanical block, no governor-electrical entries
    names = d.names
    g = findall(startswith.(names, "gov_"))
    ip = findfirst(==("p"), names)
    for (_, M) in emdq_pf_masks(cl)
        @test all(M[g, vcat(ip, g)]) && !any(M[g, setdiff(1:length(names), vcat(ip, g))])
    end
    # G2: the local pattern with the mechanical block is obstructed on the quotient
    o = emdq_local_obstruction(d; mask = emdq_pf_masks(cl)[1][2])
    @test o.dim_K > 0 && o.max_rate > 0.1
end

@testset "emdq Brayton-Moser gate B0 and the structural B1 (section 24)" begin
    sc = load_scenario(joinpath(ROOT, "cases", "IEEE39Bus_PF", "no_fault_phtrue.json"))
    eq = solve_equilibrium(load_case(sc.system_path), sc)
    s, xs, _ = emdq_pilot(eq; machine = "GENROU_2", trf = "Trf_2", line = "Line_11", tie = "Line_13")
    c = emdq_avr(s, xs; kappa_p = 1.0, omega_i = 1.0)
    zs = vcat(xs, 0.0)
    sig = emdq_bm_signature(s)
    @test sig[[1, 2, 3, 7, 11, 17, 19]] == fill(-1.0, 7) && sig[[6, 8, 9, 13, 20]] == fill(1.0, 5)
    q = emdq_bm_Q(s, c, zs)
    @test q.Q ≈ q.sym + q.skew && norm(q.skew) > 0.1 * norm(q.Q)        # the flux-angle coupling
    @test q.skew[6, 1] != 0 && all(q.skew[9:18, 9:18] .== 0)           # network blocks are symmetric
    rng = MersenneTwister(43)
    z = vcat(xs .+ 0.03 .* randn(rng, 19) .* max.(abs.(xs), 0.1), 0.1)
    # B0: Q F - grad P - sum N_k = 0 with every part built independently
    @test emdq_bm_balance(s, xs, c, z).relative < 1e-14
    @test emdq_bm_balance(s, xs, c, zs).relative < 1e-14
    @test emdq_bm_balance(s, xs, c, z).em > 1 && emdq_bm_balance(s, xs, c, z).gyr > 1
    Mw = zeros(20, 20)
    for (_, ix) in emdq_component_blocks(s)
        B = randn(rng, length(ix), length(ix))
        Mw[ix, ix] = (B + B') / 2
    end
    @test abs(emdq_bm_rate_check(s, xs, c, zs, z, 0.7, 1e-3 .* Mw).identity_error) < 1e-12
    # the structural B1 failure: Hess P has exactly zero rows at p and eta, so S(lambda, M)
    # annihilates them for every multiplier
    HP = ForwardDiff.hessian(w -> emdq_bm_potential(s, w), zs)
    @test all(HP[7, :] .== 0) && all(HP[20, :] .== 0)
    S = 0.3 .* HP .+ HP * Mw * HP
    @test norm(S[:, 7]) == 0 && norm(S[:, 20]) == 0
end
