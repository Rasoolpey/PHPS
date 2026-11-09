# IEEE-39 as one port-Hamiltonian model (PH_SIMULATOR_TODO.md, docs/active/PH_SIMULATOR_SPEC.md):
# balanced dq, every line a dynamic pi line, every machine a full-winding machine, shaft torques
# and field voltages fixed (no exciter or governor yet). From the PHPS folder:
#
#   julia --project=. scripts/ph_ieee39_fault.jl [bus] [R_fault_pu] [t_stop]
#
# Assembles the model, finds its equilibrium, checks the structure at random states, runs a
# three-phase fault (default: bus 16, 0.001 pu, applied at 0.1 s for 0.1 s, to 1 s) with the
# implicit midpoint rule, repeats a shorter window at half and a quarter of the step and with an
# independent BDF solver, and writes report.json and figures to outputs/ph/ieee39_bus<bus>/.

println("loading PHPS...")
flush(stdout)
using PHPS
using LinearAlgebra, Random, Printf
using ForwardDiff

root = normpath(joinpath(@__DIR__, ".."))
fbus = length(ARGS) >= 1 ? parse(Int, ARGS[1]) : 16
Rf = length(ARGS) >= 2 ? parse(Float64, ARGS[2]) : 0.001
tstop = length(ARGS) >= 3 ? parse(Float64, ARGS[3]) : 1.0
t_on, t_off = 0.1, 0.2
outdir = joinpath(root, "outputs", "ph", "ieee39_bus$(fbus)")
mkpath(outdir)

sc = load_scenario(joinpath(root, "cases", "IEEE39Bus_PF", "no_fault_phtrue.json"))
eq = solve_equilibrium(load_case(sc.system_path), sc)
s, x0, u, rep = phs_equilibrium(eq)
ids = s.net.net.bus_ids
@printf("model: %d states, %d machines, %d components; equilibrium residual %.1e, |V - V_rms| <= %.1e pu\n",
        s.n, length(s.machines), length(s.cnames), rep.residual, max(rep.dV, rep.dVgen))

# -- structure at random states (spec section 8, R3, R4, R7, R9) --------------------------------
rng = MersenneTwister(1)
scl = phs_scales(s, x0)
chk = Dict(k => 0.0 for k in ("rhs_vs_structure", "J_skew", "R_symmetry", "R_min_eig",
                              "component_balance", "interconnection", "total_balance",
                              "energy_sum", "gradient"))
for _ in 1:20
    y = x0 .+ 0.2 .* scl .* randn(rng, s.n)
    uu = PHInput(u.Tm .* (1 .+ 0.3 .* randn(rng, length(u.Tm))),
                 u.vf .* (1 .+ 0.3 .* randn(rng, length(u.vf))), copy(u.gf))
    uu.gf[rand(rng, vcat(s.dyn, s.alg))] = 500.0
    f, M, L = phs_rhs(s, y, uu), phs_matrices(s, y, uu), phs_ledger(s, y, uu)
    den = abs.(M.J) * abs.(M.z) .+ abs.(M.R) * abs.(M.z) .+ abs.(M.g)
    chk["rhs_vs_structure"] = max(chk["rhs_vs_structure"], maximum(abs.(f .- ((M.J - M.R) * M.z .+ M.g)) ./ den))
    chk["J_skew"] = max(chk["J_skew"], maximum(abs, M.J + M.J'))
    chk["R_symmetry"] = max(chk["R_symmetry"], maximum(abs, M.R - M.R'))
    chk["R_min_eig"] = max(chk["R_min_eig"], -eigmin(Symmetric(M.R)) / opnorm(M.R))
    chk["component_balance"] = max(chk["component_balance"],
        maximum(abs.(L.rate .- (L.port .- L.loss)) ./ (abs.(L.rate) .+ abs.(L.port) .+ abs.(L.loss) .+ floatmin())))
    chk["interconnection"] = max(chk["interconnection"], abs(L.interconnection) / sum(abs, L.port))
    chk["total_balance"] = max(chk["total_balance"], abs(L.dH - (L.supply - L.losses)) / (abs(L.supply) + L.losses))
    chk["energy_sum"] = max(chk["energy_sum"], abs(L.H - phs_energy(s, y)) / L.H)
    chk["gradient"] = max(chk["gradient"],
        maximum(abs, ForwardDiff.gradient(w -> phs_energy(s, w), y) .- M.z) / maximum(abs, M.z))
end
println("structure at 20 random states (relative):")
for k in sort!(collect(keys(chk)))
    @printf("  %-20s %.2e\n", k, chk[k])
end

ev = eigvals(PHPS._jacobian(s, x0, u))
unstable = sort!([real(e) for e in ev if real(e) > 1e-6], rev = true)
@printf("linearisation at the equilibrium: %d eigenvalues, largest real part %.4g /s, %d with positive real part\n",
        length(ev), maximum(real, ev), length(unstable))

# -- the fault run -----------------------------------------------------------------------------
faults = [PHFault(fbus, Rf, t_on, t_off)]
println("fault run: bus $fbus, R = $Rf pu, $t_on to $t_off s, to $tstop s ...")
flush(stdout)
r = phs_simulate(s, x0, u; tstop, faults)
for g in r.segments
    @printf("  segment %.3f-%.3f s: h = %.3e s, %d steps (w_max %.3g rad/s, fastest decay %.3g /s)%s\n",
            g.t0, g.t1, g.h, g.steps, g.w_max, g.r_max, g.fault ? ", fault on" : "")
end
@printf("  %d Newton iterations, %d Jacobians, worst scaled residual %.1e, %.0f s\n",
        r.newton.iterations, r.newton.jacobians, r.newton.worst_residual, r.wall)

store = [k for k in eachindex(r.names) if r.kind[k] != :fault]
scheme_rel = r.scheme_max ./ r.throughput
over = [r.names[k] for k in eachindex(r.names) if r.scheme_max[k] > 1e-10 * r.throughput[k]]
closure_rel = abs.(r.closure) ./ (vec(maximum(abs, r.energy, dims = 2)) .+ r.moved .+ floatmin())
kc = argmax(closure_rel)
simp_rel = abs.(r.simpson[store]) ./ max.(r.moved[store], 1e-9 .* r.throughput[store])
ks, kp = argmax(scheme_rel), store[argmax(simp_rel)]
@printf("energy: H(0) = %.4f, H(end) = %.4f pu*s; throughput %.3f, moved %.3f\n", r.H[1], r.H[end],
        r.total.throughput, r.total.moved)
@printf("  scheme identity: system %.2e of throughput; worst component %s %.2e of its throughput\n",
        r.total.scheme_max / r.total.throughput, r.names[ks], scheme_rel[ks])
println("  components over 1e-10 of their throughput: ", isempty(over) ? "none" : join(over, ", "))
@printf("  closure (summed increments against the energy function): worst component %s %.2e of its energy\n",
        r.names[kc], closure_rel[kc])
@printf("  accumulated defect (Simpson): system %.2e of moved energy; worst component %s %.2e\n",
        abs(r.total.simpson) / r.total.moved, r.names[kp], maximum(simp_rel))

# -- refinement and the independent solver, on a shorter window --------------------------------
tref = min(tstop, 0.3)
tcmp = collect(0.05:0.05:tref)                       # compared at exact segment ends
at_times(run::PHRun, ts) = reduce(hcat, [run.X[:, findfirst(y -> abs(y - t) < 1e-12, run.t)] for t in ts])
println("refinement on [0, $tref] s: h, h/2, h/4 ...")
flush(stdout)
runs = [phs_simulate(s, x0, u; tstop = tref, faults, stops = tcmp, hscale = hs, save_dt = 0.05)
        for hs in (1.0, 0.5, 0.25)]
Xr = [at_times(q, tcmp) for q in runs]
serr(A, B) = maximum(abs.(A .- B) ./ scl)
d1, d2 = serr(Xr[1], Xr[2]), serr(Xr[2], Xr[3])
order = log2(d1 / d2)
defects = [abs(q.total.simpson) for q in runs]
@printf("  scaled state difference: h vs h/2 %.3e, h/2 vs h/4 %.3e -> observed order %.2f\n", d1, d2, order)
@printf("  accumulated defect: %.3e, %.3e, %.3e (ratios %.2f, %.2f)\n", defects..., defects[1] / defects[2],
        defects[2] / defects[3])
println("independent BDF (CVODE, reltol 1e-10) on the same window ...")
flush(stdout)
tb = time()
_, Xb = phs_simulate_bdf(s, x0, u; tstop = tref, faults, saveat = tcmp)
dbdf = [serr(X, Xb) for X in Xr]
dextra = serr((4 .* Xr[3] .- Xr[2]) ./ 3, Xb)
worst_state = phs_state_names(s)[argmax(vec(maximum(abs.(Xr[1] .- Xb) ./ scl, dims = 2)))]
@printf("  midpoint vs BDF, scaled: h %.3e, h/2 %.3e, h/4 %.3e (ratios %.2f, %.2f)\n", dbdf...,
        dbdf[1] / dbdf[2], dbdf[2] / dbdf[3])
@printf("  extrapolated midpoint vs BDF %.3e; worst state %s (%.0f s)\n", dextra, worst_state, time() - tb)

# -- the voltage at the faulted bus just after the fault is removed -----------------------------
# The fault current still flows in the line inductances when its path is removed and charges
# the node capacitance. The peak is checked under refinement and against BDF on its own window.
kf = findfirst(==(fbus), ids)
tpk = collect(t_off:2e-5:min(t_off + 0.01, tref))
vmag(X) = [abs(phs_flows(s, X[:, k], u).v[kf]) for k in axes(X, 2)]
vpk = [vmag(at_times(phs_simulate(s, x0, u; tstop = tpk[end], faults, stops = tpk, hscale = hs,
                                  save_dt = 1.0), tpk)) for hs in (1.0, 0.5, 0.25)]
pk = maximum.(vpk)
_, Xpk = phs_simulate_bdf(s, x0, u; tstop = tpk[end], faults, saveat = tpk)
vpk_bdf = vmag(Xpk)
pk_bdf = maximum(vpk_bdf)
@printf("peak |V| at bus %d within 10 ms after the fault is removed: %.4f, %.4f, %.4f pu (h, h/2, h/4); BDF %.4f pu\n",
        fbus, pk..., pk_bdf)

# -- report ------------------------------------------------------------------------------------
kinds = [:line, :transformer, :load, :machine, :rotor]
group(M, kd) = vec(sum(M[[k for k in eachindex(r.kind) if r.kind[k] == kd], :], dims = 1))
report = Dict{String,Any}(
    "case" => sc.system_path, "fault" => Dict("bus" => fbus, "R_pu" => Rf, "t_on" => t_on, "t_off" => t_off),
    "t_stop" => tstop, "states" => s.n,
    "reductions" => Dict("merged_generator_buses" => ids[s.merged], "eliminated_buses" => ids[s.alg],
                         "machines" => [Dict("name" => m.name, "bus" => m.bus, "at_bus" => ids[m.node],
                                             "transformer" => m.trf, "units" => m.units) for m in s.machines]),
    "equilibrium" => Dict("residual" => rep.residual, "max_dV_vs_rms_pu" => max(rep.dV, rep.dVgen),
                          "Tm" => rep.Tm, "vf" => rep.vf,
                          "largest_real_part_per_s" => maximum(real, ev), "positive_real_parts" => unstable),
    "structure_checks_relative" => chk,
    "segments" => [Dict(String(k) => v for (k, v) in pairs(g)) for g in r.segments],
    "newton" => Dict(String(k) => v for (k, v) in pairs(r.newton)),
    "energy" => Dict("H0" => r.H[1], "H_end" => r.H[end], "throughput" => r.total.throughput,
                     "moved" => r.total.moved, "scheme_system_rel_throughput" => r.total.scheme_max / r.total.throughput,
                     "scheme_worst_component" => r.names[ks], "scheme_worst_rel_throughput" => scheme_rel[ks],
                     "scheme_components_over_tolerance" => over,
                     "closure_worst_component" => r.names[kc], "closure_worst_rel_energy" => closure_rel[kc],
                     "simpson_system_rel_moved" => abs(r.total.simpson) / r.total.moved,
                     "simpson_worst_component" => r.names[kp], "simpson_worst_rel_moved" => maximum(simp_rel)),
    "refinement" => Dict("window_s" => tref, "h_vs_h2" => d1, "h2_vs_h4" => d2, "order" => order,
                         "simpson_defects" => defects, "midpoint_vs_bdf" => dbdf,
                         "extrapolated_vs_bdf" => dextra, "worst_state" => worst_state),
    "peak_voltage_after_clearing_pu" => Dict("midpoint_h_h2_h4" => pk, "bdf" => pk_bdf),
    "wall_s" => r.wall)
PHPS.write_json(joinpath(outdir, "report.json"), report)
println("report -> ", joinpath(outdir, "report.json"))

# -- figures -----------------------------------------------------------------------------------
if Base.find_package("CairoMakie") === nothing
    println("installing CairoMakie into Julia's default environment (first time only)...")
    run(`$(Base.julia_cmd()) --startup-file=no -e "using Pkg; Pkg.add(\"CairoMakie\")"`)
end
println("drawing the figures...")
flush(stdout)
@eval using CairoMakie
t = r.t
ref = s.machines[1]
dref = r.X[PHPS._idelta(ref), :]
title = "IEEE-39 port-Hamiltonian model, three-phase fault at bus $fbus (R = $Rf pu), no exciter or governor"

fig = Figure(size = (1200, 900))
ax1 = Axis(fig[1, 1], title = "rotor angle relative to $(ref.name)", xlabel = "time [s]", ylabel = "deg")
ax2 = Axis(fig[1, 2], title = "rotor speed", xlabel = "time [s]", ylabel = "pu")
for m in s.machines
    lines!(ax1, t, rad2deg.(r.X[PHPS._idelta(m), :] .- dref), label = m.name)
    lines!(ax2, t, r.X[PHPS._ip(m), :] ./ (2 * m.w.H), label = m.name)
end
ax3 = Axis(fig[2, 1], title = "bus voltage magnitude (sampled every 1 ms; see ph_clearing.png)",
           xlabel = "time [s]", ylabel = "pu")
for b in unique([fbus, 4, 16, 26, 39, 30])
    lines!(ax3, t, abs.(r.V[findfirst(==(b), ids), :]), label = "bus $b")
end
axislegend(ax3, position = :rb, labelsize = 10)
ax4 = Axis(fig[2, 2], title = "electrical torque (generator convention)", xlabel = "time [s]", ylabel = "pu")
for m in s.machines
    o = m.offset
    cur = [PHPS._machine_currents(m, view(r.X, :, k)) for k in eachindex(t)]
    lines!(ax4, t, [-s.omega_b * (r.X[o, k] * cur[k].iq - r.X[o+1, k] * cur[k].id) for k in eachindex(t)], label = m.name)
end
for ax in (ax1, ax2, ax3, ax4)
    vspan!(ax, t_on, t_off, color = (:red, 0.1))
end
Legend(fig[1:2, 3], ax1, labelsize = 10)
Label(fig[0, :], title, fontsize = 15)
save(joinpath(outdir, "ph_trajectories.png"), fig)

fig = Figure(size = (1200, 900))
labels = Dict(:line => "lines", :transformer => "transformers", :load => "loads", :machine => "machine windings",
              :rotor => "rotors (kinetic)")
ax1 = Axis(fig[1, 1], title = "stored energy by group, change from t = 0", xlabel = "time [s]", ylabel = "pu*s")
for kd in kinds
    e = group(r.energy, kd)
    lines!(ax1, t, e .- e[1], label = labels[kd])
end
axislegend(ax1, position = :lt, labelsize = 10)
ax2 = Axis(fig[1, 2], title = "electromagnetic energy by group", xlabel = "time [s]", ylabel = "pu*s")
for kd in (:line, :transformer, :load, :machine)
    lines!(ax2, t, group(r.energy, kd), label = labels[kd])
end
axislegend(ax2, position = :rt, labelsize = 10)
ax3 = Axis(fig[2, 1], title = "power: boundary supply and losses", xlabel = "time [s]", ylabel = "pu")
lines!(ax3, t, r.supply, label = "shaft and field supply", color = :black)
for kd in (:line, :transformer, :load, :machine, :fault)
    lines!(ax3, t, group(r.loss, kd), label = "loss in " * get(labels, kd, "the fault"))
end
axislegend(ax3, position = :rt, labelsize = 10)
ax4 = Axis(fig[2, 2], title = "zoom: first 20 ms of the fault, electromagnetic energy", xlabel = "time [s]", ylabel = "pu*s")
zoom = findall(x -> t_on - 0.005 <= x <= t_on + 0.02, t)
for kd in (:line, :transformer, :load, :machine)
    lines!(ax4, t[zoom], group(r.energy, kd)[zoom], label = labels[kd])
end
for ax in (ax1, ax2, ax3, ax4)
    vspan!(ax, t_on, min(t_off, ax === ax4 ? t_on + 0.02 : t_off), color = (:red, 0.1))
end
Label(fig[0, :], title, fontsize = 15)
save(joinpath(outdir, "ph_energy.png"), fig)

fig = Figure(size = (900, 500))
ax = Axis(fig[1, 1], title = "voltage magnitude at bus $fbus after the fault conductance is removed at $t_off s " *
                             "(sampled every 20 microseconds)",
          xlabel = "time after removal [ms]", ylabel = "pu (positive-sequence magnitude, 1 pu = nominal)")
lines!(ax, 1e3 .* (tpk .- t_off), vpk[3], label = "implicit midpoint, h/4", color = :black)
lines!(ax, 1e3 .* (tpk .- t_off), vpk_bdf, label = "independent BDF", color = :darkorange, linestyle = :dash)
axislegend(ax, position = :rt)
save(joinpath(outdir, "ph_clearing.png"), fig)
println("figures -> ", outdir)
