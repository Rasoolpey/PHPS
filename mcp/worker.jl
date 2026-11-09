# The PHPS MCP simulation worker: one persistent Julia process that holds the simulator and runs one job at a time
# (build a model, run a simulation). It talks to the MCP server (server.jl) over a local socket, one JSON object per
# line; its own standard output and error go to outputs/mcp/worker.log (set by the server), so nothing it or the
# simulator prints can reach the protocol stream. Design: docs/mcp/MCP_DESIGN.md, review 01.
#
# Messages in:  {"cmd": "build"|"run", "job": id, ...}, {"cmd": "cancel", "job": id}
# Messages out: {"type": "ready"}, {"type": "progress", "job", ...}, {"type": "done", "job", "result"},
#               {"type": "failed", "job", "error"}

using Sockets, Dates, LinearAlgebra
import JSON3
LinearAlgebra.BLAS.set_num_threads(1)
include(joinpath(@__DIR__, "common.jl"))
using PHPS

const SOCK = connect(ip"127.0.0.1", parse(Int, ARGS[1]))
const SEND = ReentrantLock()
send(d) = lock(SEND) do
    println(SOCK, JSON3.write(d)); flush(SOCK)
end
const CANCELLED = Set{String}()
const CLOCK = ReentrantLock()
cancelled(job) = lock(() -> job in CANCELLED, CLOCK)
const JOBS = Channel{Any}(Inf)

struct Cancelled <: Exception
    msg::String
end

# The reader runs on its own thread so a cancel arrives while a simulation runs.
Threads.@spawn begin
    try
        for line in eachline(SOCK)
            m = JSON3.read(line, Dict{String,Any})
            m["cmd"] == "cancel" ? lock(() -> push!(CANCELLED, m["job"]), CLOCK) : put!(JOBS, m)
        end
    catch e
        @error "reader stopped" exception = e
    end
    exit(0)                                            # the server went away
end

# ---- building ---------------------------------------------------------------------------------------------------------

const STEAM_TABLES = Ref{Any}(nothing)
steam_tables_once() = (STEAM_TABLES[] === nothing && (STEAM_TABLES[] = steam_tables()); STEAM_TABLES[])

function windings()
    wc = JSON3.read(read(joinpath(BASE_CASE, "winding_circuits.json"), String))
    fl(v) = v === nothing ? nothing : Float64(v)
    return Dict{Int,NamedTuple}(parse(Int, String(b)) =>
        (; Ll = fl(v.Ll), Lad = fl(v.Lad), Laq = fl(v.Laq), Lfd = fl(v.Lfd), L1d = fl(v.L1d), L1q = fl(v.L1q),
         L2q = fl(v.L2q), Ra = fl(v.Ra), Rfd = fl(v.Rfd), R1d = fl(v.R1d), R1q = fl(v.R1q), R2q = fl(v.R2q), H = fl(v.H))
        for (b, v) in pairs(wc.machines))
end

"""Build the controlled EMT model of a stored specification (the derived case in the model's folder)."""
function build(spec::Dict, mdir::AbstractString)
    raw = JSON3.read(read(joinpath(mdir, "system.json"), String), Dict{String,Any})
    fm = JSON3.read(read(joinpath(BASE_CASE, "full_model.json"), String))
    case = load_case(joinpath(mdir, "system.json"))
    eq = solve_equilibrium(case)
    circuits = spec["machine_data"] == "powerfactory_windings" ? windings() : nothing
    dq0 = spec["unbalance"] == "none" ? (;) :
          (; zero = Symbol(spec["unbalance"]), zero_pf_json = joinpath(BASE_CASE, "ieee39_base.json"))
    kw = Dict{Symbol,Any}(:loads => Symbol(spec["loads"]), :circuits => circuits, :lines => Symbol(spec["lines"]),
                          :dq0 => dq0)
    comps = raw["components"]
    if spec["dc_exciter"]
        shunt = [k for (k, c) in comps if startswith(c["type"], "IEEET1") && c["params"]["KE"] < 0]
        dc = fm.exciters.dc_shunt
        kw[:dc_exciter] = (; n = Dict(k => Float64(dc.n) for k in shunt), field_ratio = Float64(dc.field_ratio))
    end
    if spec["brushless_exciter"]
        bl = fm.exciters.brushless
        kw[:brushless] = (; KC = Float64(bl.KC), KD = Float64(bl.KD), sat = (Float64(bl.sat[1]), Float64(bl.sat[2])),
                          field_share = Float64(bl.field_share))
    end
    if spec["steam"]
        st = fm.steam
        kw[:steam] = (; paths = Dict(String(k) => Symbol(v) for (k, v) in pairs(st.paths)), r_v = Float64(st.r_v),
                      rh_dT = Float64(st.rh_dT), tau_n = Float64(st.tau_n), lowflow = Float64(st.lowflow),
                      cold_reheat = Float64(st.cold_reheat), volumes = Symbol(st.volumes), valve = Symbol(st.valve),
                      tables = steam_tables_once())
    end
    if spec["hydro"]
        hy = fm.hydro
        kw[:hydro] = (; Tw = hy.Tw isa AbstractString ? Symbol(hy.Tw) : Float64(hy.Tw), Te = Float64(hy.Te),
                      N = Int(hy.N), fp = Float64(hy.fp))
    end
    cs, x0, u, rep = phs_controlled(eq; kw...)
    return cs, collect(x0), u, rep
end

"""The operating point and the model's description, in engineering quantities."""
function summarise(cs, x0, u, rep, spec)
    s = cs.s
    xs = x0[1:s.n]
    up = phs_control_inputs(cs, x0, u.gf; cf = u.cf)
    V = PHPS._bus_voltages(s, xs, up)
    ids = collect(Int, PHPS._phys(cs).net.net.bus_ids)
    vt = phs_terminal_voltage(s, xs, up)
    ms = phs_machine_signals(s, xs)
    cg = phs_control_signals(cs, x0)
    gens = [Dict("machine" => m.name, "bus" => m.bus,
                 "terminal_voltage_pu" => abs(get(vt, m.bus, V[findfirst(==(m.bus), ids)])),
                 "air_gap_power_pu" => -ms[j].torque * ms[j].speed, "turbine_power_pu" => cg.Pm[j],
                 "field_voltage_Efd_pu" => cg.Efd[j], "speed_pu" => ms[j].speed,
                 "rotor_angle_deg" => rad2deg(ms[j].delta))
            for (j, m) in enumerate(s.machines)]
    buses = [Dict("bus" => ids[k], "voltage_pu" => abs(V[k]), "angle_deg" => rad2deg(angle(V[k])))
             for k in eachindex(ids) if isfinite(abs(V[k]))]
    ctrl = [Dict("machine" => s.machines[j].name,
                 "exciter" => cs.exc[j] === nothing ? "none (fixed field voltage)" : string(nameof(typeof(cs.exc[j]))),
                 "governor" => cs.gov[j] === nothing ? "none (fixed turbine power)" : string(nameof(typeof(cs.gov[j]))))
            for j in eachindex(s.machines)]
    return Dict{String,Any}(
        "states" => cs.n, "machines" => length(s.machines), "buses_in_network" => length(ids),
        "controllers" => ctrl, "generators" => gens, "bus_voltages" => buses,
        "equilibrium_residual" => Float64(get(rep, :residual_controlled, NaN)),
        "base" => "per unit on the 100 MVA system base; voltages on the bus base; angles in degrees in the synchronous frame",
        "state_names" => phs_state_names(cs), "energy_components" => PHPS._cnames(cs))
end

function job_build(m)
    spec = m["spec"]; mid = m["model_id"]
    mdir = under_out("models", mid)
    t0 = time()
    cs, x0, u, rep = build(spec, mdir)
    summ = summarise(cs, x0, u, rep, spec)
    summ["build_seconds"] = round(time() - t0, digits = 1)
    atomic_write(joinpath(mdir, "model.json"), JSON3.write(summ))
    return Dict("model_id" => mid, "build_seconds" => summ["build_seconds"], "states" => summ["states"])
end

# ---- running ----------------------------------------------------------------------------------------------------------

function standard_records(cs)
    n = cs.s.n
    sig(x) = phs_machine_signals(cs.s, view(x, 1:n))
    function vt(x, u)
        up = phs_control_inputs(cs, x, u.gf; cf = u.cf)
        V = PHPS._bus_voltages(cs.s, x[1:n], up)
        ids = PHPS._phys(cs).net.net.bus_ids
        d = phs_terminal_voltage(cs.s, x[1:n], up)
        return [abs(get(d, m.bus, V[findfirst(==(m.bus), ids)])) for m in cs.s.machines]
    end
    return ("speed" => (x, u) -> [q.speed for q in sig(x)],
            "rotor_angle" => (x, u) -> [q.delta for q in sig(x)],
            "air_gap_power" => (x, u) -> [-q.torque * q.speed for q in sig(x)],
            "terminal_voltage" => vt,
            "field_voltage" => (x, u) -> phs_control_signals(cs, collect(x)).Efd,
            "turbine_power" => (x, u) -> phs_control_signals(cs, collect(x)).Pm)
end

function job_run(m)
    job = m["job"]; mid = m["model_id"]; rid = m["run_id"]
    mdir = under_out("models", mid); rdir = under_out("runs", rid)
    spec = JSON3.read(read(joinpath(mdir, "spec.json"), String), Dict{String,Any})
    p = m["params"]
    t0 = time()
    cs, x0, u, _ = build(spec, mdir)                    # a fresh model for every run (review 01, point 3)
    tbuild = time() - t0
    faults = [PHFault(f["bus"], f["R"], f["t_on"], f["t_off"]; kind = Symbol(f["kind"]), Rg = f["Rg"],
                      clearing = Symbol(f["clearing"])) for f in p["faults"]]
    tstop = Float64(p["tstop"]); deadline = t0 + 60 * Float64(p["max_wall_minutes"])
    last = Ref(0.0); tprev = Ref(-1.0); pass = Ref(1)
    monitor = function (t, x, uu)
        t < tprev[] - 1e-12 && (pass[] += 1)
        tprev[] = t
        cancelled(job) && throw(Cancelled("cancelled by the user"))
        tn = time()
        tn > deadline && throw(Cancelled("stopped at the wall-time limit of $(p["max_wall_minutes"]) min"))
        if tn - last[] > 2
            last[] = tn
            send(Dict("type" => "progress", "job" => job, "t" => t, "tstop" => tstop, "pass" => pass[],
                      "wall" => tn - t0))
        end
        return nothing
    end
    kw = Dict{Symbol,Any}(:tstop => tstop, :faults => faults, :save_dt => Float64(p["save_dt"]),
                          :hscale => Float64(p["hscale"]), :tol => Float64(p["tol"]), :monitor => monitor,
                          :record => standard_records(cs))
    p["h"] === nothing || (kw[:h] = Float64(p["h"]))
    r = phs_simulate(cs, x0, u; kw...)
    meta = Dict{String,Any}("model_id" => mid, "spec" => JSON3.write(spec), "params" => JSON3.write(p),
                            "machines" => [mm.name for mm in cs.s.machines],
                            "machine_buses" => [mm.bus for mm in cs.s.machines],
                            "build_seconds" => tbuild, "created" => string(now()))
    mkpath(rdir)
    tmp = joinpath(rdir, "run.tmp.jld2")
    phs_write_run(tmp, r; model = cs, meta)
    mv(tmp, joinpath(rdir, "run.jld2"); force = true)
    wall = time() - t0
    steps = sum(g.steps for g in r.segments)
    info = Dict{String,Any}("run_id" => rid, "model_id" => mid, "tstop" => tstop, "wall_seconds" => round(wall, digits = 1),
                            "build_seconds" => round(tbuild, digits = 1), "steps" => steps,
                            "simulation_seconds" => round(r.wall, digits = 1), "passes" => pass[],
                            "faults" => p["faults"], "states" => cs.n, "preset" => spec["preset"])
    atomic_write(joinpath(rdir, "run.json"), JSON3.write(info))
    # timing history for estimates
    hist = joinpath(OUT, "timing.jsonl")
    open(io -> println(io, JSON3.write(Dict("states" => cs.n, "preset" => spec["preset"], "lines" => spec["lines"],
                                            "tstop" => tstop, "sim_wall" => r.wall, "build" => tbuild,
                                            "clearing" => any(f -> f["clearing"] == "current_zero", p["faults"])))),
         hist, "a")
    return info
end

# ---- the loop ---------------------------------------------------------------------------------------------------------

function errtext(e)
    e isa Cancelled && return e.msg
    e isa ArgumentError && return e.msg
    e isa ErrorException && return e.msg
    return sprint(showerror, e)
end

send(Dict("type" => "ready", "threads" => Threads.nthreads()))
for m in JOBS
    job = m["job"]
    if cancelled(job)
        send(Dict("type" => "failed", "job" => job, "error" => "cancelled before it started")); continue
    end
    send(Dict("type" => "started", "job" => job))
    try
        res = m["cmd"] == "build" ? job_build(m) : m["cmd"] == "run" ? job_run(m) :
              throw(ArgumentError("unknown command $(m["cmd"])"))
        send(Dict("type" => "done", "job" => job, "result" => res))
    catch e
        @error "job failed" job exception = (e, catch_backtrace())
        send(Dict("type" => "failed", "job" => job, "error" => errtext(e)))
    end
end
