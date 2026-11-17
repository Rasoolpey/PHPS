# The model handed to analysis tools (`phs_analysis_model`, `phs_export`; docs/analysis.md):
#   (1) the export's own checks are at rounding level (operating point, structure against the rates, J skew, R >= 0);
#   (2) the linearisation A agrees with central differences of the rates in random directions;
#   (3) Q (derivative of the efforts) agrees with central differences of the efforts in random directions and is symmetric
#       on the rows in the form; the efforts are the gradient of the stored energy there (a directional derivative of
#       the energy, independent of the structure routine);
#   (4) every state has a name, a device and a kind; the kinds partition the states;
#   (5) both file formats give back the same numbers (JLD2; CSV);
#   (6) any device or group as its own system: its matrices are the model's blocks, its right-hand side with its ports
#       reproduces its rates, its functions agree with the export at x0, and its equations name every state.

using PHPS, Test, LinearAlgebra, Random
const PHS_NS = PHPS.phs_nstates

@testset "analysis model and its export (phs_analysis_model, phs_export)" begin
    sc = load_scenario(joinpath(ROOT, "cases", "IEEE39Bus_PF", "no_fault_phtrue.json"))
    eq = solve_equilibrium(load_case(sc.system_path), sc)
    s, xs, us, _ = phs_equilibrium(eq)
    cs, xc, uc, _ = phs_controlled(eq; hydro = (; Tw = :match))
    rng = MersenneTwister(7)
    for (label, m, x0, u) in (("grid model", s, xs, us), ("controlled model", cs, xc, uc))
        am = phs_analysis_model(m, x0, u)
        N = length(x0)
        @testset "$label" begin
            @test am.checks.operating_point < 1e-8
            @test am.checks.structure < 1e-10
            @test am.checks.J_skew == 0
            @test am.checks.R_min > -1e-12
            @test am.checks.R_asymmetry < 1e-12
            @test count(am.in_form) > 0 && all(>(0), am.scale)
            @test am.checks.Q_asymmetry < 1e-6
            @test length(am.names) == N == length(am.component) == length(am.kind) == length(am.in_form)
            @test all(!isempty, am.component)
            @test all(k -> k in (:network, :machine, :excitation, :prime_mover, :converter), am.kind)
            @test count(==(:machine), am.kind) > 0 && count(==(:network), am.kind) > 0
            mach = m isa PHControlled ? m.s.machines : m.machines
            @test all(mc -> all(==(mc.name), am.component[mc.offset:mc.offset+PHS_NS(mc)-1]), mach)
            m isa PHControlled && @test count(==(:excitation), am.kind) + count(==(:prime_mover), am.kind) == N - m.s.n
            for _ in 1:3
                d = randn(rng, N) .* am.scale; d ./= norm(d ./ am.scale); h = 1e-6
                fd = (am.rate(x0 .+ h .* d) .- am.rate(x0 .- h .* d)) ./ 2h
                @test maximum(abs.(am.A * d .- fd) ./ am.scale) < 1e-5 * maximum(abs.(fd) ./ am.scale) + 1e-9
                zd = (am.structure(x0 .+ h .* d).z .- am.structure(x0 .- h .* d).z) ./ 2h
                @test norm(am.Q * d .- zd) < 1e-4 * norm(zd) + 1e-9
                di = d .* am.in_form                                   # a direction on the rows in the form
                dH = (am.energy(x0 .+ h .* di) - am.energy(x0 .- h .* di)) / 2h
                @test abs(dH - dot(am.effort, di)) < 1e-6 * (sum(abs.(am.effort .* di)) + 1e-12)
            end
            dir = mktempdir()
            f = phs_export(joinpath(dir, "m.jld2"), am); back = PHPS.JLD2.load(f)
            @test back["A"] == am.A && back["J"] == am.J && back["R"] == am.R && back["Q"] == am.Q
            @test back["names"] == am.names && back["kind"] == String.(am.kind)
            c = phs_export(joinpath(dir, "csv"), am)
            rd(n) = reduce(vcat, [parse.(Float64, split(l, ","))' for l in eachline(joinpath(c, n))])
            @test rd("A.csv") == am.A && rd("J.csv") == am.J && rd("Q.csv") == am.Q
            @test countlines(joinpath(c, "states.csv")) == N + 1
            rows = [split(l, ",") for l in Iterators.drop(eachline(joinpath(c, "states.csv")), 1)]
            @test [parse(Float64, r[6]) for r in rows] == am.x0 && [strip(r[2], '"') for r in rows] == am.names
            @test back["outside"] == am.outside && countlines(joinpath(c, "outside.csv")) == length(am.outside) + 1
            @test isfile(joinpath(c, "feedthrough.csv")) == (am.vf !== nothing)
            phs_export(c, merge(am, (; Q = nothing)))                  # an export without Q into the same folder
            @test !isfile(joinpath(c, "Q.csv")) && isfile(joinpath(c, "A.csv"))
            # any device or group as its own system (phs_subsystem), and its written-out equations
            dev = am.component[findfirst(==(:machine), am.kind)]
            for sel in (dev, [dev, am.names[1]], :machine, (nm, cp, kd) -> kd == :network)
                sub = phs_subsystem(am, sel)
                @test sub.checks.structure < 1e-10 && sub.checks.J_skew == 0 && sub.checks.R_min > -1e-12
                @test sub.J == am.J[sub.index, sub.index] && sub.A == am.A[sub.index, sub.index] && sub.names == am.names[sub.index]
                st = sub.structure(sub.x0)
                @test st.J - st.R ≈ sub.J - sub.R && st.Gp ≈ sub.Gp && st.effort ≈ sub.effort
                @test sub.rate(sub.x0) ≈ sub.rate0 atol = 1e-12
            end
            xkeep = copy(am.x0)
            sub = phs_subsystem(am, dev)
            sub.rate(sub.x0 .* 1.001); sub.structure(sub.x0 .* 1.001)
            @test am.x0 == xkeep && sub.x0 == xkeep[sub.index]              # the model's state is not touched by the group's functions
            @test sub.checks.varies_samples == 2 && any(sub.varies[:, 1:length(sub.names)]) && any(sub.varies[:, length(sub.names)+1:end])   # a machine's internal couplings depend on its fluxes, its ports on its angle
            @test sub.J == am.J[sub.index, sub.index]                       # still the model's block after the sampling
            @test all(==(dev), sub.component) && !isempty(sub.ports) && size(sub.Gp) == (length(sub.names), length(sub.ports))
            eqs = phs_equations(sub)
            @test all(nm -> occursin("d/dt " * nm, eqs), sub.names) && occursin("## Ports", eqs)
            whole = phs_subsystem(am, collect(1:N))
            @test isempty(whole.ports) && whole.checks.shared_loss_entries == 0 && whole.checks.energy_separable == 0
            @test_throws ArgumentError phs_subsystem(am, "no such device")
            cs2 = phs_export(joinpath(dir, "sub"), sub)
            @test isfile(joinpath(cs2, "equations.md")) && rd("../sub/Gp.csv") == sub.Gp && countlines(joinpath(cs2, "ports.csv")) == length(sub.ports) + 1
            fb = PHPS.JLD2.load(phs_export(joinpath(dir, "sub.jld2"), sub))
            @test fb["Gp"] == sub.Gp && fb["equations"] == eqs && fb["index"] == sub.index
        end
    end
end
