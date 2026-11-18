# The IEEE 39-bus full physical model (cases/IEEE39Bus_FullPhysical): build it, check its operating point, run the
# scenario's fault by the implicit midpoint rule with the energy account, and store the run. Progress (simulated time
# and wall time) is appended to progress.log every 5 ms of simulated time. Only simulates; plots read run.jld2.
#   julia --project=. scripts/full_physical_run.jl [tstop_s = 0.3] [h_us = 1.0]
using PHPS, Printf, JLD2
const P = PHPS
root = normpath(joinpath(@__DIR__, ".."))
tstop = length(ARGS) >= 1 ? parse(Float64, ARGS[1]) : 0.3
h = (length(ARGS) >= 2 ? parse(Float64, ARGS[2]) : 1.0) * 1e-6
dir = joinpath(root, "cases", "IEEE39Bus_FullPhysical")
out = joinpath(root, "outputs", "ph", "full_physical", "fault_bus16_3ph_t$(tstop)"); mkpath(out)
plog = joinpath(out, "progress.log"); rm(plog; force = true)
say(s) = (println(s); open(io -> println(io, s), plog, "a"))
t0 = time()
fm = phs_full_model(dir)
cs, x0, u = fm.cs, fm.x0, fm.u
say(@sprintf("built in %.0f s: %d states, %d machines, %d steam units, %d hydropower plants; faults %s",
             time() - t0, length(x0), length(cs.s.machines), length(P._steam_units(cs)), length(P._hydro_plants(cs)),
             string(fm.faults)))
dx = similar(x0); P.phs_rhs!(dx, cs, x0, u)
sc = phs_scales(cs, x0)
say(@sprintf("operating point: max |f(x0)| / scale = %.2e", maximum(abs.(dx) ./ sc)))
nextt = Ref(0.0); t1 = time()
monitor = (t, x, u) -> (t >= nextt[] && (say(@sprintf("t = %.4f s, wall %.0f s", t, time() - t1)); nextt[] += 5e-3); nothing)
ts = filter(<=(tstop), sort(unique(round.(vcat(collect(0:1e-5:tstop), collect(0.0995:1e-6:0.1025),
                                               collect(0.1995:1e-6:0.2075)); digits = 9))))
r = phs_simulate(cs, x0, u; tstop, faults = fm.faults, h, sample_at = ts, monitor)
wall = time() - t1
say(@sprintf("run %.3f s at h = %.2f us: wall %.0f s; closure / gross %.2e", tstop, h * 1e6, wall,
             maximum(abs, r.closure) / maximum(abs, r.throughput)))
phs_write_run(joinpath(out, "run.jld2"), r; model = cs,
              meta = Dict("case" => dir, "tstop" => tstop, "h" => h, "wall" => wall, "faults" => string(fm.faults)))
say("written " * joinpath(out, "run.jld2"))
