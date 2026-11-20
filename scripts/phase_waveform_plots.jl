# Three-phase waveforms of a stored EMT run: the instantaneous phase voltages a, b, c at chosen buses, from the run's
# bus-voltage phasors (phs_phases at the frame angle omega_b t), around the fault.
# Run: julia --project=. scripts/phase_waveform_plots.jl <run folder or run.jld2> [out.png] [--buses=16,17,33] [--from=0.05] [--to=0.35]
using PHPS, JLD2, CairoMakie, Printf
opt(key, default) = (a = filter(x -> startswith(x, "--$key="), ARGS); isempty(a) ? default : split(a[1], "="; limit = 2)[2])
pos = filter(a -> !startswith(a, "--"), ARGS)
isempty(pos) && error("usage: phase_waveform_plots.jl <run folder or run.jld2> [out.png] [--buses=16,17,33]")
f = isdir(pos[1]) ? joinpath(pos[1], "run.jld2") : pos[1]
out = length(pos) >= 2 ? pos[2] : joinpath(dirname(f), "phases.png")
d = JLD2.load(f); t = d["t"]; V = d["V"]; ids = d["bus_ids"]; m = get(d, "meta", Dict{String,Any}())
buses = parse.(Int, split(opt("buses", "16,17,33"), ","))
t1, t2 = parse(Float64, opt("from", "0.05")), parse(Float64, opt("to", "0.35"))
wb = 2pi * 60
sel = findall(x -> t1 <= x <= t2, t)
fon, foff = get(m, "fault_on", NaN), get(m, "fault_off", NaN)
title = opt("title", "phase voltages, three-phase fault at bus $(get(m, "fault_bus", "?")) ($(fon)-$(foff) s)")
fig = Figure(size = (1400, 330 * length(buses) + 60), fontsize = 14)
Label(fig[0, 1], title; fontsize = 17, font = :bold, tellwidth = false)
cols = (:royalblue, :darkorange, :seagreen)
for (r, b) in enumerate(buses)
    k = findfirst(==(b), ids); k === nothing && error("bus $b is not in the run")
    abc = [phs_phases(V[k, i], wb * t[i]) for i in sel]
    ax = Axis(fig[r, 1], title = "bus $b", xlabel = r == length(buses) ? "time (s)" : "", ylabel = "pu (peak phase)")
    isfinite(fon) && vspan!(ax, fon, foff; color = (:red, 0.08))
    for (p, name) in enumerate(("a", "b", "c"))
        lines!(ax, t[sel], getindex.(abc, p); color = cols[p], label = "phase $name")
    end
    r == 1 && axislegend(ax; position = :rt, orientation = :horizontal)
end
save(out, fig; px_per_unit = 2)
println("saved ", out)
