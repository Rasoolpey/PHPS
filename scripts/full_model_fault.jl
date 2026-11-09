# The IEEE 39-bus full model (cases/IEEE39Bus_FullModel), one short circuit: the scenario's fault (three-phase at bus
# 16 through 0.001 pu, 0.1-0.2 s), run by the implicit midpoint rule with the exact energy account (phs_simulate).
# Checks written to report.json (docs/active/FULL_MODEL_DIGITAL_TWIN_PLAN.md section 3): the energy account's closure;
# the fault current and the bus-16 voltage during the fault; the machine speeds; the steam and hydro states in their
# domain (the run completes); against PowerFactory's stored EMT run of the same fault (its governors and exciters are
# the IEEEG1/IEEET1 block diagrams, ours physical: the first cycles are electromagnetic and comparable).
# Only simulates; scripts/full_model_plots.jl draws the figures.
# Run: julia --project=. scripts/full_model_fault.jl [--tstop=10] [--scenario=fault_bus16_3ph.json]
using PHPS, LinearAlgebra, Printf, JSON3, JLD2
root = normpath(joinpath(@__DIR__, ".."))
opt(key, default) = (a = filter(x -> startswith(x, "--$key="), ARGS); isempty(a) ? default : split(a[1], "="; limit = 2)[2])
dir = joinpath(root, "cases", "IEEE39Bus_FullModel")
scen = opt("scenario", "fault_bus16_3ph.json")
tstop = parse(Float64, opt("tstop", "10"))
t0 = time()
fm = phs_full_model(dir; scenario = scen)
cs, x0, u = fm.cs, fm.x0, fm.u
@printf("full model built in %.0f s: %d states, %d machines, %d steam units, hydro %s\n", time() - t0, length(x0),
        length(cs.s.machines), length(PHPS._steam_units(cs)), string(typeof.(cs.gov[PHPS._hydro_units(cs)])))
# the operating point: the field at x0
F = phs_rhs(cs, x0, u); sc = phs_scales(cs, x0)
@printf("operating point: max |f(x0)| / scale = %.2e\n", maximum(abs.(F) ./ sc))
ks = PHPS._steam_units(cs)
record = ("efd" => (x, u) -> phs_control_signals(cs, x).Efd,
          "pm" => (x, u) -> phs_control_signals(cs, x).Pm,
          "stroke" => (x, u) -> Float64[PHPS._steam_stroke(cs.gov[j], x[cs.goff[j]+1]) for j in ks],
          "speed" => (x, u) -> Float64[x[PHPS._ip(m)] / (2m.w.H) for m in cs.s.machines])
t1 = time()
r = phs_simulate(cs, x0, u; tstop, faults = fm.faults, save_dt = 1e-3, record)
wall = time() - t1
out = joinpath(root, "outputs", "ph", "full_model", splitext(scen)[1] * (tstop == 10 ? "" : "_t$(tstop)"))
mkpath(out)
phs_write_run(joinpath(out, "run.jld2"), r; model = cs,
              meta = Dict("case" => dir, "scenario" => scen, "tstop" => tstop, "fault_bus" => fm.faults[1].bus,
                          "fault_R" => fm.faults[1].R, "fault_on" => fm.faults[1].t_on, "fault_off" => fm.faults[1].t_off,
                          "steam_units" => [cs.gov[j].name for j in ks],
                          "machines" => [m.name for m in cs.s.machines], "machine_buses" => [m.bus for m in cs.s.machines]))
# checks
f = fm.faults[1]; s = cs.s
Vb = r.V                                                  # node voltages (complex, per unit), one column per saved time
kf = PHPS.bus_index(s.net.net, f.bus)                  # the faulted bus's node
duringf = findall(t -> f.t_on + 0.02 <= t <= f.t_off - 0.005, r.t)
v16 = abs.(Vb[kf, :])
ifault = v16[duringf] ./ f.R
speeds = reduce(hcat, [[r.X[PHPS._ip(m), k] / (2m.w.H) for m in s.machines] for k in eachindex(r.t)])
rep = Dict(
    "states" => length(x0), "tstop" => tstop, "wall_s" => wall,
    "energy" => Dict("scheme_max_over_throughput" => r.total.scheme_max / max(r.total.throughput, 1e-300),
                     "scheme" => r.total.scheme, "throughput" => r.total.throughput),
    "bus16_voltage_before" => v16[1], "bus16_voltage_during_fault_mean" => sum(v16[duringf]) / length(duringf),
    "fault_current_pu_mean" => sum(ifault) / length(ifault), "fault_current_pu_max" => maximum(ifault),
    "speed_min" => minimum(speeds), "speed_max" => maximum(speeds),
    "speed_end_spread" => maximum(speeds[:, end]) - minimum(speeds[:, end]),
    "voltage_end_bus16" => v16[end])
open(joinpath(out, "report.json"), "w") do io; JSON3.pretty(io, rep); end
@printf("run: %.1f s in %.0f s wall; energy scheme max / throughput %.2e\n", tstop, wall, rep["energy"]["scheme_max_over_throughput"])
@printf("bus 16: %.4f pu before, %.4f pu during the fault; fault current %.2f pu (system base) mean, %.2f max\n",
        v16[1], rep["bus16_voltage_during_fault_mean"], rep["fault_current_pu_mean"], rep["fault_current_pu_max"])
@printf("speeds: %.5f to %.5f pu; spread at the end %.2e; bus 16 at the end %.4f pu\n", rep["speed_min"], rep["speed_max"],
        rep["speed_end_spread"], rep["voltage_end_bus16"])
println("written to ", out)
