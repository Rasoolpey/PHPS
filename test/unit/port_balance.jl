# The power books at the machines' ports (E1, record 30): with S_k taken at the high-voltage bus (as
# scripts/opstab/demand_common.jl `port_power`), a merged step-up transformer lies inside generator k's port. At every
# state the machines' port powers equal what the network components absorb (`phs_network_mask`): active power from the
# ledger (losses and rates from the vector field) and active and reactive power from the terminal sums, each checked
# separately; and each machine's own balance holds port by port, so no error can cancel between generators.

@testset "port power balance: merged step-up transformers inside the ports" begin
    sc = load_scenario(joinpath(ROOT, "cases", "IEEE39Bus_PF", "no_fault_phtrue.json"))
    eq = solve_equilibrium(load_case(sc.system_path), sc)
    s, x0, u, _ = phs_equilibrium(eq; loads = :series_all, const_power = true)
    pn = s.net
    ids = pn.net.bus_ids
    merged = [m.trf for m in s.machines if m.trf != ""]

    # S_k at the high-voltage bus, and the complex power every network element absorbs at its terminals
    function books(x, uu)
        f = phs_flows(s, x, uu)
        vc = phs_cut_voltage(s, x, uu)
        V(k) = (v = f.v[k]; isfinite(real(v)) ? v : vc[ids[k]])
        S = [-V(m.node) * conj(f.mc[j].inet) / m.a for (j, m) in enumerate(s.machines)]
        Se = ComplexF64[]
        for (j, l) in enumerate(s.branches)
            push!(Se, (V(pn.from[l]) / pn.a[l] - V(pn.to[l])) * conj(f.ib[j]))
        end
        for (j, k) in enumerate(pn.sh_bus)
            s.sidx[j] > 0 && push!(Se, V(k) * conj(f.ish[j]))
        end
        for (ci, c) in enumerate(s.cuts)
            push!(Se, V(c.node) * conj(f.iload[ci]))
        end
        for k in s.dyn
            push!(Se, f.v[k] * conj(f.inj[k]))
        end
        for k in s.alg
            push!(Se, V(k) * conj((pn.G[k] + uu.gf[k]) * V(k)))
        end
        return S, Se, f, V
    end

    L0 = phs_ledger(s, x0, u)
    net0 = phs_network_mask(s, L0)
    @testset "ownership" begin
        @test length(merged) == 8 && allunique(merged)
        @test all(m -> m.trf == "" || ids[m.node] != m.bus, s.machines)     # a merged machine's port is another bus
        for t in merged
            k = findfirst(==(t), L0.names)
            @test L0.kind[k] === :transformer && !net0[k]
        end
        @test count(net0) == count(k -> k in (:line, :transformer, :load, :fault), L0.kind) - 8
        @test_throws ArgumentError phs_network_mask(s, (; names = replace(L0.names, merged[1] => "none"), kind = L0.kind))
    end

    rng = MersenneTwister(30)
    scl = phs_scales(s, x0)
    states = vcat([("equilibrium", x0)], [("random $r", x0 .+ 0.05 .* scl .* randn(rng, s.n)) for r in 1:3])
    for (nm, x) in states
        @testset "$nm" begin
            L = phs_ledger(s, x, u)
            net = phs_network_mask(s, L)
            S, Se, f, V = books(x, u)
            vg = phs_terminal_voltage(s, x, u)
            P = real.(S)
            # each machine: what it delivers at the high-voltage bus = field + shaft input - its own losses and
            # storage rate (its merged transformer, windings and rotor); the windings' and rotor's rates are evidence
            for (j, m) in enumerate(s.machines)
                own = [findfirst(==(c), L.names) for c in (m.trf == "" ? [m.name, m.name * ".rotor"] :
                                                           [m.trf, m.name, m.name * ".rotor"])]
                c = f.mc[j]
                pin = u.vf[j] * c.ifd + PHPS._shaft_torque(u, j, c.omega) * c.omega
                sc_ = abs(P[j]) + abs(pin) + sum(abs, L.rate[own]) + sum(L.loss[own])
                @test abs(P[j] - (pin - sum(L.rate[own] .+ L.loss[own]))) < 1e-10 * sc_
                # the merged transformer alone (its rate cancels in the sum above; review 31): the active power it
                # takes in at the high-voltage bus less what it passes at the recovered generator-bus voltage (from
                # the vector field, `phs_terminal_voltage`) is its loss plus its storage rate
                if m.trf != ""
                    k = own[1]
                    ptr = real((V(m.node) / m.a - vg[m.bus]) * conj(c.inet))
                    @test abs(ptr - (L.rate[k] + L.loss[k])) < 1e-10 * (abs(P[j]) + abs(ptr) + abs(L.rate[k]) + L.loss[k])
                end
            end
            # the network side, active power from the ledger
            T = sum(abs, P) + sum(abs, L.rate[net]) + sum(L.loss[net])
            @test abs(sum(P) - sum(L.rate[net] .+ L.loss[net])) < 1e-10 * T
            # the network side from the terminal sums, active and reactive checked separately
            @test abs(sum(P) - sum(real, Se)) < 1e-10 * (sum(abs, P) + sum(abs ∘ real, Se))
            Q = imag.(S)
            @test abs(sum(Q) - sum(imag, Se)) < 1e-10 * (sum(abs, Q) + sum(abs ∘ imag, Se))
            imer = [findfirst(==(t), L.names) for t in merged]
            if nm == "equilibrium"
                # counting the merged transformers as network components misses by their losses (about 0.14 pu)
                @test abs(sum(L.rate[net])) < 1e-10 * T
                @test all(abs(L.rate[k]) < 1e-10 * T for k in imer)
                @test sum(L.loss[imer]) > 0.05
                old = sum(P) - sum(L.loss[net]) - sum(L.loss[imer])
                @test abs(old + sum(L.loss[imer])) < 1e-10 * T
            else
                # away from rest every merged transformer's storage is excited, the three without resistance too
                @test all(abs(L.rate[k]) > 1e-8 for k in imer)
            end
        end
    end

    @testset "with the controllers (PHControlled), away from rest" begin
        cs, xc, uc, _ = phs_controlled(eq; loads = :series_all)
        x = xc .+ 0.05 .* phs_scales(cs, xc) .* randn(rng, length(xc))
        L = phs_ledger(cs, x, uc)
        net = phs_network_mask(cs, L)
        @test net == vcat(phs_network_mask(cs.s, phs_ledger(cs.s, x[1:cs.s.n], phs_control_inputs(cs, x, uc.gf))),
                          falses(length(L.kind) - length(cs.s.ckind)))
        uin = phs_control_inputs(cs, x, uc.gf)
        f = phs_flows(cs.s, x[1:cs.s.n], uin)
        vc = phs_cut_voltage(cs.s, x[1:cs.s.n], uin)
        P = [real(-(isfinite(real(f.v[m.node])) ? f.v[m.node] : vc[ids[m.node]]) * conj(f.mc[j].inet) / m.a)
             for (j, m) in enumerate(cs.s.machines)]
        T = sum(abs, P) + sum(abs, L.rate[net]) + sum(L.loss[net])
        @test abs(sum(L.rate[net])) > 1e-6 * T                  # the network's storage is excited
        @test abs(sum(P) - sum(L.rate[net] .+ L.loss[net])) < 1e-10 * T
    end
end
