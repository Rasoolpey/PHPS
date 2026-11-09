# Figures of the IEEE 39-bus full model's fault run (scripts/full_model_fault.jl), from the saved run only.
#   fault.png: bus voltages (the faulted bus and its neighbours); machine speeds (Hz from 60); field voltages; turbine
#              powers; steam valve strokes; the energy account's step balance against its throughput
# Run: julia --project=. scripts/full_model_plots.jl [outputs/ph/full_model/fault_bus16_3ph]
using JLD2, CairoMakie, Printf
root = normpath(joinpath(@__DIR__, ".."))
dir = isempty(ARGS) ? joinpath(root, "outputs", "ph", "full_model", "fault_bus16_3ph") : abspath(ARGS[1])
d = JLD2.load(joinpath(dir, "run.jld2")); m = d["meta"]; t = d["t"]; rec = d["records"]
fb = m["fault_bus"]; ids = d["bus_ids"]
fig = Figure(size = (1500, 1000), fontsize = 13)
ax = Axis(fig[1, 1], title = "bus voltages (bus $fb faulted, $(m["fault_on"])-$(m["fault_off"]) s, R = $(m["fault_R"]) pu)", xlabel = "time (s)", ylabel = "pu")
for b in unique([fb, 15, 17, 14, 19, 24])
    k = findfirst(==(b), ids); k === nothing && continue
    lines!(ax, t, abs.(d["V"][k, :]), label = "bus $b")
end
axislegend(ax, position = :rb)
ax = Axis(fig[1, 2], title = "machine speeds", xlabel = "time (s)", ylabel = "Hz from 60")
for (i, n) in enumerate(m["machines"]); lines!(ax, t, 60 .* (rec["speed"][i, :] .- 1), label = replace(n, "GENROU_" => "G")); end
axislegend(ax, position = :rt, nbanks = 2)
ax = Axis(fig[2, 1], title = "field voltages", xlabel = "time (s)", ylabel = "pu")
for (i, n) in enumerate(m["machines"]); lines!(ax, t, rec["efd"][i, :]); end
ax = Axis(fig[2, 2], title = "turbine powers", xlabel = "time (s)", ylabel = "pu (system base)")
for (i, n) in enumerate(m["machines"]); lines!(ax, t, rec["pm"][i, :]); end
ax = Axis(fig[3, 1], title = "steam valve strokes", xlabel = "time (s)", ylabel = "power units")
for (i, n) in enumerate(m["steam_units"]); lines!(ax, t, rec["stroke"][i, :], label = n); end
ax = Axis(fig[3, 2], title = @sprintf("energy account: largest step balance / throughput = %.1e", d["total"]["scheme_max"] / max(d["total"]["throughput"], 1e-300)),
          xlabel = "time (s)", ylabel = "stored energy H (pu s)")
lines!(ax, t, d["H"])
save(joinpath(dir, "fault.png"), fig)
println("saved ", joinpath(dir, "fault.png"))
