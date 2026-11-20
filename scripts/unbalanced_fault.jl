# An unbalanced short circuit on the IEEE 39-bus full model (cases/IEEE39Bus_FullModel) built with the zero-axis
# network (ph_system's `zero = :physical`): phase to ground, phase to phase, two phases to ground, ... The phase-to-ground
# voltages a, b, c (with the zero-sequence part) are recorded at chosen buses while the run goes, then drawn.
# Run: julia --project=. scripts/unbalanced_fault.jl [--kind=ag] [--bus=16] [--R=0.001] [--on=0.1] [--off=0.2]
#      [--clearing=current_zero] [--tstop=0.4] [--save_dt=1e-4] [--buses=16,17,33] [--replot]
#      kind: ag, bg, cg (phase to ground), ab, bc, ca (phase to phase), abg, bcg, cag (two phases to ground), abcg, abc
# Writes outputs/ph/full_model/unbalanced_<kind>_bus<bus>/ (run.jld2, phases.png).
using PHPS, JLD2, CairoMakie, Printf
root = normpath(joinpath(@__DIR__, ".."))
opt(key, default) = (a = filter(x -> startswith(x, "--$key="), ARGS); isempty(a) ? default : split(a[1], "="; limit = 2)[2])
kind = Symbol(opt("kind", "ag")); bus = parse(Int, opt("bus", "16")); R = parse(Float64, opt("R", "0.001"))
ton, toff = parse(Float64, opt("on", "0.1")), parse(Float64, opt("off", "0.2"))
clearing = Symbol(opt("clearing", "current_zero"))
tstop, save_dt = parse(Float64, opt("tstop", "0.4")), parse(Float64, opt("save_dt", "1e-4"))
buses = parse.(Int, split(opt("buses", "16,17,33"), ","))
dir = joinpath(root, "cases", "IEEE39Bus_FullModel")

out = joinpath(root, "outputs", "ph", "full_model", "unbalanced_$(kind)_bus$(bus)")
if "--replot" in ARGS                                     # draw again from the saved run, without simulating
    p = JLD2.load(joinpath(out, "phases.jld2")); t, ph, buses = p["t"], p["phases"], p["buses"]
    d = JLD2.load(joinpath(out, "run.jld2")); V, ids = d["V"], d["bus_ids"]
else
    t0 = time()
    fm = phs_full_model(dir; dq0 = (; zero = :physical, zero_pf_json = joinpath(dir, "ieee39_base.json")))
    cs, x0, u = fm.cs, fm.x0, fm.u
    ps = PHPS._phys(cs)
    ids = collect(Int, ps.net.net.bus_ids)
    rows = [something(findfirst(==(b), ids), 0) for b in buses]
    any(==(0), rows) && error("not a network bus: ", buses[rows .== 0])
    @printf("full model with the zero axis built in %.0f s: %d states\n", time() - t0, length(x0))
    fault = PHFault(bus, R, ton, toff; kind, clearing)
    # the frame angle is omega_b t + theta_pw when a record is evaluated (the simulator sets it), so t is recovered from it
    tnow() = (ps.asym.theta[] - ps.asym.theta_pw) / ps.omega_b
    # a bus with no zero-sequence path to ground (a generator terminal on the delta side of its step-up transformer) has
    # an undetermined neutral: v0 = 0 is taken there (`nan0`), which keeps the line-to-line voltages right
    record = ("phases" => (x, u) -> vec(PHPS.phs_phase_voltages(cs, x, u; t = tnow(), nan0 = 0.0)[:, rows]),)
    t1 = time()
    r = phs_simulate(cs, x0, u; tstop, faults = [fault], save_dt, record)
    @printf("run: %.2f s of %s fault at bus %d in %.0f s wall\n", tstop, kind, bus, time() - t1)
    mkpath(out)
    t, ph, V = r.t, r.records["phases"], r.V            # phases: 3 * length(buses) rows (a, b, c per bus), one column per time
    phs_write_run(joinpath(out, "run.jld2"), r; model = cs, meta = Dict("fault_bus" => bus, "kind" => string(kind), "R" => R,
                  "fault_on" => ton, "fault_off" => toff, "clearing" => string(clearing), "buses" => buses))
    JLD2.jldsave(joinpath(out, "phases.jld2"); t, phases = ph, buses)
end
# a bus that is not a node of the network (a machine terminal reduced into its machine) has no recorded phases: they are
# taken from the run's recovered bus voltage phasor, without a zero-sequence part (theta_pw = 0: theta = omega_b t)
fromV = Int[]
for (i, b) in enumerate(buses)
    all(isnan, ph[3i-2, :]) || continue
    k = findfirst(==(b), ids); push!(fromV, b)
    for j in eachindex(t)
        ph[3i-2:3i, j] .= phs_phases(V[k, j], 2pi * 60 * t[j])
    end
end
isempty(fromV) || println("bus ", join(fromV, ", "), ": phases from the bus voltage phasor (no zero-sequence part)")

names = Dict(:ag => "phase a to ground", :bg => "phase b to ground", :cg => "phase c to ground", :ab => "phase a to phase b",
             :bc => "phase b to phase c", :ca => "phase c to phase a", :abg => "phases a, b to ground",
             :bcg => "phases b, c to ground", :cag => "phases c, a to ground", :abcg => "three phases to ground",
             :abc => "three phases")
fig = Figure(size = (1400, 330 * length(buses) + 60), fontsize = 14)
Label(fig[0, 1], "phase-to-ground voltages, $(names[kind]) fault at bus $bus ($(ton)-$(toff) s, R = $R pu, $(clearing) clearing)";
      fontsize = 17, font = :bold, tellwidth = false)
cols = (:royalblue, :darkorange, :seagreen)
sel = findall(x -> 0.05 <= x, t)
for (i, b) in enumerate(buses)
    ax = Axis(fig[i, 1], title = b in fromV ? "bus $b (from its voltage phasor, no zero-sequence part)" : "bus $b", xlabel = i == length(buses) ? "time (s)" : "", ylabel = "pu (peak phase)")
    vspan!(ax, ton, toff; color = (:red, 0.08))
    for (p, n) in enumerate(("a", "b", "c"))
        lines!(ax, t[sel], ph[3(i-1)+p, sel]; color = cols[p], label = "phase $n")
    end
    i == 1 && axislegend(ax; position = :rt, orientation = :horizontal)
end
save(joinpath(out, "phases.png"), fig; px_per_unit = 2)
println("written to ", relpath(out, root))
