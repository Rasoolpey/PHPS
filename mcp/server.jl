# The PHPS MCP server: lets an AI assistant drive the PHPS EMT simulator through a fixed set of tools. This process
# speaks the Model Context Protocol on standard input/output and loads no simulator code; builds and runs go to one
# persistent worker process (worker.jl) whose output goes to outputs/mcp/worker.log. Stored runs are read here.
# Design and reviews: docs/mcp/MCP_DESIGN.md, docs/mcp/reviews/01_*.
#
# Start (what .mcp.json does):  julia --project=mcp --startup-file=no mcp/server.jl

using ModelContextProtocol, Sockets, Dates, Statistics
import JSON3, JLD2
include(joinpath(@__DIR__, "common.jl"))

const PROTO_IN, PROTO_OUT = stdin, stdout
mkpath(MODELS); mkpath(RUNS)

# ---- the worker -------------------------------------------------------------------------------------------------------

mutable struct Job
    id::String
    kind::String                 # build | run
    target::String               # model id or run id
    status::String               # queued | running | done | failed | cancelled
    created::Float64
    started::Float64
    finished::Float64
    progress::Dict{String,Any}
    result::Any
    error::String
end

const JOBS = Dict{String,Job}()
const JL = ReentrantLock()
const WORKER = Ref{Any}(nothing)        # (; proc, conn) when connected
const WSTATE = Ref("stopped")           # stopped | starting | ready
const WSEND = ReentrantLock()

function start_worker()
    WSTATE[] = "starting"
    srv = listen(ip"127.0.0.1", 0)
    port = getsockname(srv)[2]
    log = open(joinpath(OUT, "worker.log"), "a")
    println(log, "\n==== worker start ", now()); flush(log)
    cmd = `$(Base.julia_cmd()) --project=$(@__DIR__) --startup-file=no -t 2 $(joinpath(@__DIR__, "worker.jl")) $port`
    proc = run(pipeline(cmd; stdin = devnull, stdout = log, stderr = log); wait = false)
    conn = accept(srv)
    close(srv)
    WORKER[] = (; proc, conn)
    @async try
        for line in eachline(conn)
            on_message(JSON3.read(line, Dict{String,Any}))
        end
    catch e
        @warn "worker connection ended" exception = e
    finally
        WSTATE[] = "stopped"; WORKER[] = nothing
        lock(JL) do
            for j in values(JOBS)
                j.status in ("queued", "running") || continue
                j.status = "failed"; j.finished = time()
                j.error = "the simulation worker stopped (see outputs/mcp/worker.log)"
            end
        end
    end
end

function on_message(m)
    t = m["type"]
    t == "ready" && (WSTATE[] = "ready"; return)
    lock(JL) do
        j = get(JOBS, m["job"], nothing)
        j === nothing && return
        if t == "started"
            j.status = "running"; j.started = time()
        elseif t == "progress"
            j.progress = m
        elseif t == "done"
            j.status = "done"; j.finished = time(); j.result = m["result"]
        elseif t == "failed"
            j.status = occursin("cancel", m["error"]) || occursin("wall-time limit", m["error"]) ? "cancelled" : "failed"
            j.finished = time(); j.error = m["error"]
        end
    end
end

function ensure_worker(wait_s = 600)
    WSTATE[] == "stopped" && @async start_worker()
    t0 = time()
    while WSTATE[] != "ready" && time() - t0 < wait_s
        sleep(0.2)
    end
    WSTATE[] == "ready" || throw(ErrorException("the simulation worker is still starting (loading the simulator); try again in a minute"))
end

function wsend(d)
    w = WORKER[]
    w === nothing && throw(ErrorException("the simulation worker is not running"))
    lock(WSEND) do
        println(w.conn, JSON3.write(d)); flush(w.conn)
    end
end

function submit(kind, target, msg)
    ensure_worker()
    id = newid("j")
    lock(JL) do
        JOBS[id] = Job(id, kind, target, "queued", time(), NaN, NaN, Dict{String,Any}(), nothing, "")
    end
    wsend(merge(msg, Dict("cmd" => kind, "job" => id)))
    return id
end

function wait_job(id, wait_s, ctx)
    t0 = time()
    while time() - t0 < wait_s
        j = lock(() -> JOBS[id], JL)
        j.status in ("done", "failed", "cancelled") && break
        if ctx !== nothing && !isempty(j.progress)
            p = j.progress
            send_progress(ctx, (p["pass"] - 1) * p["tstop"] + p["t"];
                          message = "simulated $(round(p["t"], digits = 3)) s of $(p["tstop"]) s (pass $(p["pass"]))")
        end
        sleep(0.5)
    end
    return lock(() -> JOBS[id], JL)
end

# ---- results ----------------------------------------------------------------------------------------------------------

text(s) = TextContent(text = s)
fail(s) = CallToolResult(content = [Dict{String,Any}("type" => "text", "text" => s)], is_error = true)
fmt(x; d = 4) = x isa Real ? string(round(x, sigdigits = d)) : string(x)

const RUNCACHE = Dict{String,Any}()
function loadrun(rid)
    checkid(rid)
    path = under_out("runs", rid, "run.jld2")
    isfile(path) || throw(ArgumentError("no stored run $rid (see list_runs)"))
    get!(() -> JLD2.load(path), RUNCACHE, rid)
end

const UNITS = Dict("speed" => "pu of rated speed", "rotor_angle" => "degrees, synchronous frame",
                   "air_gap_power" => "pu on 100 MVA", "terminal_voltage" => "pu", "field_voltage" => "pu (Efd)",
                   "turbine_power" => "pu on 100 MVA", "voltage" => "pu (bus voltage magnitude)",
                   "energy" => "pu s on 100 MVA (stored energy)")

"""A named signal of a stored run as (values, unit)."""
function signal(R, name; refmachine = nothing)
    meta = R["meta"]; machines = String.(meta["machines"])
    k = findlast('.', name)
    k === nothing && throw(ArgumentError("signal $name: use <machine>.<quantity>, bus<k>.voltage or <component>.energy (see list_signals)"))
    obj, q = name[1:k-1], name[k+1:end]
    if q == "voltage" && startswith(obj, "bus")
        b = parse(Int, obj[4:end]); i = findfirst(==(b), R["bus_ids"])
        i === nothing && throw(ArgumentError("bus $b is not a network bus of this run"))
        return abs.(R["V"][i, :]), UNITS["voltage"]
    elseif q == "energy"
        i = findfirst(==(obj), R["names"])
        i === nothing && throw(ArgumentError("no energy component $obj (see list_signals)"))
        return R["energy"][i, :], UNITS["energy"]
    elseif haskey(R["records"], q)
        j = findfirst(==(obj), machines)
        j === nothing && throw(ArgumentError("no machine $obj; machines: $(join(machines, ", "))"))
        v = R["records"][q][j, :]
        if q == "rotor_angle"
            if refmachine !== nothing
                r = findfirst(==(refmachine), machines)
                r === nothing && throw(ArgumentError("no reference machine $refmachine"))
                v = v .- R["records"][q][r, :]
            end
            v = rad2deg.(v)
        end
        return v, UNITS[q]
    elseif q == "state"
        i = findfirst(==(obj), R["state_names"])
        i === nothing && throw(ArgumentError("no state $obj"))
        return R["X"][i, :], "state units"
    end
    throw(ArgumentError("unknown quantity $q in $name (see list_signals)"))
end

fault_times(R) = sort(unique(Float64[t for f in JSON3.read(R["meta"]["params"], Dict{String,Any})["faults"] for t in (f["t_on"], f["t_off"])]))

"""Indices that keep each signal's minimum and maximum in every time bucket, the ends and the fault instants."""
function envelope(t, vals, idx, npts)
    length(idx) <= npts && return idx
    nb = max(1, npts ÷ (2 * max(1, length(vals))))
    keep = Set([first(idx), last(idx)])
    edges = round.(Int, range(1, length(idx) + 1, length = nb + 1))
    for b in 1:nb, v in vals
        r = idx[edges[b]:edges[b+1]-1]
        isempty(r) && continue
        push!(keep, r[argmin(v[r])], r[argmax(v[r])])
    end
    return sort(collect(keep))
end

function digest(rid)
    R = loadrun(rid)
    t = R["t"]; ids = R["bus_ids"]; V = abs.(R["V"])
    ft = fault_times(R)
    ext(f, mask) = (any(mask) ? f(i -> isfinite(V[i]) && mask[i[2]] ? V[i] : (f === findmin ? Inf : -Inf), CartesianIndices(V)) : (NaN, nothing))
    vtxt(lbl, (v, k)) = k === nothing ? "" : "$lbl $(fmt(v)) pu at bus $(ids[k[1]]), t = $(fmt(t[k[2]])) s."
    lines = String[]
    if !isempty(ft)
        during = (first(ft) .< t .< last(ft)); after = t .> last(ft)
        push!(lines, vtxt("During the fault: lowest bus voltage", ext(findmin, during)))
        push!(lines, vtxt("After clearing: highest bus voltage", ext(findmax, after)) * " " * vtxt("Lowest bus voltage", ext(findmin, after)))
        push!(lines, "Bus voltages at the end (lowest, highest): $(fmt(minimum(filter(isfinite, V[:, end])))) pu, $(fmt(maximum(filter(isfinite, V[:, end])))) pu.")
    end
    sp = R["records"]["speed"]; dev = abs.(sp .- 1)
    dmax, kk = findmax(dev)
    ms = String.(R["meta"]["machines"])
    tot = R["total"]
    append!(lines, [
             "Largest speed deviation: $(fmt(100 * dmax)) % at $(ms[kk[1]]), t = $(fmt(t[kk[2]])) s.",
             "Final speeds (pu): " * join(["$(ms[j]) $(fmt(sp[j, end]; d = 6))" for j in eachindex(ms)], ", "),
             "Energy balance: largest accumulated error $(fmt(tot["scheme_max"] / max(tot["throughput"], 1e-300); d = 3)) of the energy exchanged (see energy_audit)."])
    return join(filter(!isempty, lines), "\n")
end

function jobtext(j::Job)
    s = "Job $(j.id) ($(j.kind) $(j.target)): $(j.status)"
    if j.status == "running" && !isempty(j.progress)
        p = j.progress
        s *= "; simulated $(fmt(p["t"])) s of $(p["tstop"]) s, pass $(p["pass"]), wall $(round(Int, p["wall"])) s"
    elseif j.status == "running" && j.kind == "build"
        s *= "; building (the first build compiles the simulator and can take several minutes)"
    end
    j.status in ("failed", "cancelled") && (s *= ".\nReason: " * j.error)
    if j.status == "done"
        r = j.result
        if j.kind == "build"
            s *= ".\nModel $(r["model_id"]) built in $(r["build_seconds"]) s, $(r["states"]) states. Use operating_point or describe."
        else
            s *= ".\nRun $(r["run_id"]) finished in $(r["wall_seconds"]) s ($(r["build_seconds"]) s building, $(r["simulation_seconds"]) s simulating, $(r["steps"]) steps).\n" * digest(r["run_id"])
        end
    end
    return s
end

# ---- the tools --------------------------------------------------------------------------------------------------------

"""Wrap a handler: argument errors and simulator refusals come back as plain error results."""
guard(f) = (args, ctx) -> try
    f(args, ctx)
catch e
    e isa ArgumentError || e isa ErrorException ? fail(e isa ArgumentError ? e.msg : e.msg) : fail(sprint(showerror, e))
end

schema(props; required = String[]) = Dict{String,Any}("type" => "object", "properties" => props, "required" => required)
P(type, desc; kw...) = merge(Dict{String,Any}("type" => type, "description" => desc), Dict{String,Any}(String(k) => v for (k, v) in kw))

const CASES_TEXT = """
IEEE 39-bus New England system (the only case in this version), as an electromagnetic-transient (EMT) model: every
line, transformer and machine winding dynamic, the network in the synchronous dq frame, optionally with the zero
sequence for unbalanced faults. Data: cases/IEEE39Bus_FullModel (the case system, PowerFactory's machine winding data,
the physical-unit options).

Presets for build_model:
- ieee: IEEE block-diagram controllers (IEEET1 exciters, IEEEG1 steam and IEEEG3 hydro governors).
- full_physical: physical DC shunt and brushless exciters, steam units with steam-table thermodynamics, hydro unit
  with an elastic penstock (the all-physical IEEE 39-bus model).
Switches (override the preset): dc_exciter, brushless_exciter, steam, hydro; loads (series_all, parallel, series,
composite = induction motors plus a static part); lines (lumped pi, travelling = distributed Bergeron lines);
unbalance (none, physical = zero-sequence network with effective grounding, powerfactory = PowerFactory's grounding);
machine_data (powerfactory_windings, case_genrou); load_scale; parameter_changes.
Typical cost: 30 to 90 s of computer time per simulated second; the first build after a server start compiles the
simulator (several minutes).
"""

function t_build(a, ctx)
    spec = resolve_build(a)
    raw = JSON3.read(read(joinpath(BASE_CASE, "system.json"), String), Dict{String,Any})
    notes = apply_changes!(raw, spec)
    if get(a, "dry_run", false) == true
        return text("Build request is valid.\nResolved options: $(JSON3.write(spec))\n" * join(notes, "\n") *
                    "\nExpected time: a few seconds to a minute once compiled; the first build after a server start compiles the simulator (several minutes); the first steam build creates the steam tables (about 40 s).")
    end
    mid = newid("m")
    mdir = under_out("models", mid)
    atomic_write(joinpath(mdir, "system.json"), JSON3.write(raw))
    atomic_write(joinpath(mdir, "spec.json"), JSON3.write(spec))
    id = submit("build", mid, Dict{String,Any}("spec" => spec, "model_id" => mid))
    j = wait_job(id, Float64(get(a, "wait_seconds", 20)), ctx)
    return text(jobtext(j) * (isempty(notes) ? "" : "\nNotes:\n" * join(notes, "\n")) *
                (j.status in ("queued", "running") ? "\nCall job_status with job_id $(j.id) (wait_seconds up to 60) to follow it." : ""))
end

function model_info(mid)
    checkid(mid)
    p = under_out("models", mid, "model.json")
    isfile(p) || throw(ArgumentError("model $mid is not built (or does not exist); see list_models"))
    JSON3.read(read(p, String), Dict{String,Any})
end

function t_opoint(a, ctx)
    M = model_info(a["model_id"])
    io = IOBuffer()
    println(io, "Operating point of model $(a["model_id"]) ($(M["states"]) states). ", M["base"], ".")
    println(io, "\nGenerators: machine, bus, terminal voltage, air-gap power, turbine power, field voltage Efd, speed, rotor angle")
    for g in M["generators"]
        println(io, "$(g["machine"]), bus $(g["bus"]): $(fmt(g["terminal_voltage_pu"]; d = 5)) pu, $(fmt(g["air_gap_power_pu"])) pu, ",
                "$(fmt(g["turbine_power_pu"])) pu, Efd $(fmt(g["field_voltage_Efd_pu"])) pu, $(fmt(g["speed_pu"]; d = 7)) pu, $(fmt(g["rotor_angle_deg"])) deg")
    end
    println(io, "\nBus voltages (magnitude pu, angle deg):")
    println(io, join(["$(b["bus"]): $(fmt(b["voltage_pu"]; d = 5)) / $(fmt(b["angle_deg"]))" for b in M["bus_voltages"]], "; "))
    println(io, "\nEquilibrium residual: $(fmt(M["equilibrium_residual"]; d = 3)) (scaled).")
    return text(String(take!(io)))
end

function t_describe(a, ctx)
    mid = checkid(a["model_id"]); M = model_info(mid)
    raw = JSON3.read(read(under_out("models", mid, "system.json"), String), Dict{String,Any})
    comps = raw["components"]
    c = get(a, "component", nothing)
    if c === nothing
        io = IOBuffer()
        println(io, "Components of model $mid (name, type, bus or controlled machine):")
        for k in sort(collect(keys(comps)))
            p = comps[k]["params"]
            println(io, "$k: $(comps[k]["type"])", haskey(p, "bus") ? ", bus $(p["bus"])" : "", haskey(p, "syn") ? ", controls $(p["syn"])" : "")
        end
        println(io, "\nControllers in the built model:")
        for x in M["controllers"]
            println(io, "$(x["machine"]): exciter $(x["exciter"]), governor $(x["governor"])")
        end
        return text(String(take!(io)))
    end
    haskey(comps, c) || throw(ArgumentError("no component $c"))
    p = comps[c]["params"]
    states = [s for s in M["state_names"] if startswith(s, String(c) * ".")]
    return text("$c ($(comps[c]["type"])) parameters (case data, per unit on the component's base unless named otherwise):\n" *
                join(["$k = $(p[k])" for k in sort(collect(keys(p)))], ", ") *
                "\nStates in the model: " * (isempty(states) ? "none under this name (physical units are named after their machine)" : join(states, ", ")))
end

function estimate(spec, tstop, faults)
    hist = joinpath(OUT, "timing.jsonl")
    rates = Float64[]
    builds = Float64[]
    if isfile(hist)
        for l in eachline(hist)
            h = JSON3.read(l)
            h.preset == spec["preset"] && h.lines == spec["lines"] && (push!(rates, h.sim_wall / h.tstop); push!(builds, h.build))
        end
    end
    cz = any(f -> f["clearing"] == "current_zero", faults) ? (2.0, 5.0) : (1.0, 1.0)
    if isempty(rates)
        lo, hi = spec["preset"] == "full_physical" ? (60.0, 150.0) : (30.0, 90.0)
        b = (5.0, 120.0); src = "no measured history for these options yet: rough prior"
    else
        lo, hi = minimum(rates), maximum(rates); b = (minimum(builds), maximum(builds))
        src = "from $(length(rates)) earlier runs with the same options"
    end
    return (lo * tstop * cz[1] + b[1], hi * tstop * cz[2] + b[2], src)
end

function t_run(a, ctx)
    mid = checkid(a["model_id"]); M = model_info(mid)
    spec = JSON3.read(read(under_out("models", mid, "spec.json"), String), Dict{String,Any})
    raw = JSON3.read(read(under_out("models", mid, "system.json"), String), Dict{String,Any})
    buses = Set(Int(b["idx"]) for b in raw["Bus"])
    tstop = Float64(get(a, "tstop", 0.0))
    0 < tstop <= 120 || throw(ArgumentError("tstop must be in (0, 120] s"))
    faults = resolve_faults(get(a, "faults", Any[]), spec, tstop, buses)
    save_dt = Float64(something(get(a, "save_dt", nothing), 1e-3))
    tstop / save_dt <= 200_000 || throw(ArgumentError("too many saved samples (tstop / save_dt > 200000); increase save_dt"))
    maxw = Float64(something(get(a, "max_wall_minutes", nothing), 10.0))
    lo, hi, src = estimate(spec, tstop, faults)
    est = "Estimated wall time $(round(Int, lo))-$(round(Int, hi)) s ($src)."
    if hi > 60 * maxw && get(a, "allow_long", false) != true
        return fail(est * " This exceeds max_wall_minutes = $maxw. Ask the user before starting it; then call again with allow_long = true and max_wall_minutes large enough.")
    end
    params = Dict{String,Any}("tstop" => tstop, "faults" => faults, "save_dt" => save_dt,
                              "h" => get(a, "h", nothing), "hscale" => Float64(something(get(a, "hscale", nothing), 1.0)),
                              "tol" => Float64(something(get(a, "tol", nothing), 1e-13)), "max_wall_minutes" => maxw)
    get(a, "dry_run", false) == true && return text("Run request is valid. $est\nResolved: $(JSON3.write(params))")
    rid = newid("r")
    id = submit("run", rid, Dict{String,Any}("model_id" => mid, "run_id" => rid, "params" => params))
    j = wait_job(id, Float64(something(get(a, "wait_seconds", nothing), 20)), ctx)
    return text(est * "\n" * jobtext(j) *
                (j.status in ("queued", "running") ? "\nCall job_status with job_id $(j.id) and wait_seconds (up to 60) to follow it." : ""))
end

function t_status(a, ctx)
    id = checkid(a["job_id"])
    haskey(JOBS, id) || throw(ArgumentError("no job $id in this server session"))
    j = wait_job(id, min(60.0, Float64(something(get(a, "wait_seconds", nothing), 0))), ctx)
    return text(jobtext(j))
end

function t_cancel(a, ctx)
    id = checkid(a["job_id"])
    j = get(JOBS, id, nothing)
    j === nothing && throw(ArgumentError("no job $id"))
    j.status in ("done", "failed", "cancelled") && return text("Job $id already ended: $(j.status).")
    if get(a, "force", false) == true
        w = WORKER[]; w === nothing || kill(w.proc)
        return text("The worker was stopped; every queued or running job ends as failed. It restarts with the next build or run.")
    end
    wsend(Dict("cmd" => "cancel", "job" => id))
    j = wait_job(id, 15.0, nothing)
    return text(jobtext(j) * (j.status == "running" ? "\nStill running (it stops at the next step it takes); use force = true to stop the worker." : ""))
end

function t_list_runs(a, ctx)
    rows = String[]
    for d in sort(readdir(RUNS); rev = true)
        f = joinpath(RUNS, d, "run.json")
        isfile(f) || continue
        r = JSON3.read(read(f, String))
        fs = join(["$(x.kind) at bus $(x.bus) $(x.t_on)-$(x.t_off) s" for x in r.faults], "; ")
        push!(rows, "$(r.run_id): model $(r.model_id) ($(r.preset), $(r.states) states), $(r.tstop) s, faults: $(isempty(fs) ? "none" : fs), wall $(r.wall_seconds) s")
    end
    models = String[]
    for d in sort(readdir(MODELS); rev = true)
        f = joinpath(MODELS, d, "spec.json")
        isfile(f) || continue
        s = JSON3.read(read(f, String))
        built = isfile(joinpath(MODELS, d, "model.json")) ? "built" : "not built"
        push!(models, "$d: $(s.preset), loads $(s.loads), lines $(s.lines), unbalance $(s.unbalance), exciters dc=$(s.dc_exciter) brushless=$(s.brushless_exciter), steam=$(s.steam), hydro=$(s.hydro) ($built)")
    end
    return text("Models:\n" * (isempty(models) ? "none" : join(models, "\n")) * "\n\nRuns:\n" * (isempty(rows) ? "none" : join(rows, "\n")))
end

function t_signals(a, ctx)
    R = loadrun(a["run_id"])
    ms = String.(R["meta"]["machines"])
    q = ["speed", "rotor_angle", "air_gap_power", "terminal_voltage", "field_voltage", "turbine_power"]
    s = "Signals of run $(a["run_id"]) (saved every $(fmt(R["t"][2] - R["t"][1])) s, $(length(R["t"])) samples):\n" *
        "Per machine (" * join(ms, ", ") * "): " * join(["<machine>.$x [$(UNITS[x])]" for x in q], ", ") *
        " (rotor_angle can be taken relative to a reference machine in get_signals).\n" *
        "Per bus: bus<k>.voltage [pu], k in " * join(R["bus_ids"], ", ") * ".\n" *
        "Stored energy per component: <component>.energy [pu s], components: " * join(R["names"], ", ") * "."
    get(a, "include_raw_states", false) == true && (s *= "\nRaw states (as <name>.state): " * join(R["state_names"], ", "))
    return text(s)
end

function t_get(a, ctx)
    R = loadrun(a["run_id"])
    t = R["t"]
    t0 = Float64(something(get(a, "t_from", nothing), t[1])); t1 = Float64(something(get(a, "t_to", nothing), t[end]))
    idx = findall(x -> t0 - 1e-12 <= x <= t1 + 1e-12, t)
    isempty(idx) && throw(ArgumentError("no samples between $t0 and $t1 s"))
    names = String.(a["signals"]); isempty(names) && throw(ArgumentError("give at least one signal"))
    ref = get(a, "reference_machine", nothing)
    sv = [signal(R, n; refmachine = ref) for n in names]
    io = IOBuffer()
    println(io, "Run $(a["run_id"]), $(fmt(t0)) s to $(fmt(t1)) s, $(length(idx)) samples.")
    ref === nothing || println(io, "Rotor angles relative to $ref.")
    println(io, "Summary (initial, minimum at time, maximum at time, final):")
    for (n, (v, u)) in zip(names, sv)
        w = v[idx]; i1, i2 = argmin(w), argmax(w)
        println(io, "$n [$u]: $(fmt(w[1]; d = 6)), min $(fmt(w[i1]; d = 6)) at $(fmt(t[idx[i1]])) s, max $(fmt(w[i2]; d = 6)) at $(fmt(t[idx[i2]])) s, final $(fmt(w[end]; d = 6))")
    end
    npts = min(500, Int(something(get(a, "max_points", nothing), 150)))
    keep = envelope(t, [v for (v, _) in sv], idx, npts)
    for ft in fault_times(R)                                   # the switching instants
        k = argmin(abs.(t[idx] .- ft)); push!(keep, idx[k])
    end
    keep = sort(unique(keep))
    println(io, "\nTable ($(length(keep)) rows; each signal's minimum and maximum kept in every time bucket, and the fault switching instants):")
    println(io, "t_s,", join(names, ","))
    for k in keep
        println(io, fmt(t[k]; d = 7), ",", join([fmt(v[k]; d = 7) for (v, _) in sv], ","))
    end
    return text(String(take!(io)))
end

function t_audit(a, ctx)
    R = loadrun(a["run_id"])
    tot = R["total"]
    thr = R["throughput"]; sm = R["scheme_max"]
    ok = findall(>(0), thr)
    rel = sm[ok] ./ thr[ok]
    ord = ok[sortperm(rel; rev = true)][1:min(5, length(ok))]
    clo = abs.(R["closure"]) ./ (vec(maximum(abs, R["energy"]; dims = 2)) .+ R["moved"] .+ floatmin())
    oc = sortperm(clo; rev = true)[1:min(5, length(clo))]
    io = IOBuffer()
    println(io, "Energy account of run $(a["run_id"]).")
    println(io, "What it checks: at every step, each component's change of stored energy must equal the power it received at its terminals minus its losses, times the step. The error is accumulated over the run and compared with the energy the component exchanged (its throughput).")
    println(io, "System: largest accumulated error $(fmt(tot["scheme_max"] / max(tot["throughput"], 1e-300); d = 3)) of the energy exchanged; integration-rule defect $(fmt(abs(tot["simpson"]) / max(tot["moved"], 1e-300); d = 3)) of the energy moved.")
    println(io, "Worst components (accumulated error / energy exchanged): " *
                join(["$(R["names"][k]) $(fmt(sm[k] / thr[k]; d = 3))" for k in ord], ", "))
    println(io, "Largest closure (sum of step increments against the energy function, relative to the energy moved; zero to rounding for quadratic storage, the midpoint rule's small error for saturating storage): " *
                join(["$(R["names"][k]) $(fmt(clo[k]; d = 3))" for k in oc], ", "))
    println(io, "Project acceptance level: accumulated error at most 1e-10 of the energy exchanged, per component.")
    return text(String(take!(io)))
end

const MAKIE = Ref(false)
function t_plot(a, ctx)
    rids = String.(a["run_ids"]); names = String.(a["signals"])
    (isempty(rids) || isempty(names)) && throw(ArgumentError("give run_ids and signals"))
    if !MAKIE[]
        @eval Main using CairoMakie
        MAKIE[] = true
    end
    png = Base.invokelatest(plot_png, rids, names, a)
    return [ImageContent(data = read(png), mime_type = "image/png"),
            text("Saved: $png (read from the stored runs; nothing was simulated).")]
end

function plot_png(rids, names, a)
    M = Main.CairoMakie
    ref = get(a, "reference_machine", nothing)
    series = [(r, n, signal(loadrun(r), n; refmachine = ref)...) for r in rids for n in names]
    units = unique([s[4] for s in series])
    fig = M.Figure(size = (1000, 330 * length(units)))
    t0 = get(a, "t_from", nothing); t1 = get(a, "t_to", nothing)
    for (i, u) in enumerate(units)
        ax = M.Axis(fig[i, 1]; xlabel = "time (s)", ylabel = u, title = i == 1 ? String(something(get(a, "title", nothing), "")) : "")
        for (r, n, v, uu) in series
            uu == u || continue
            t = loadrun(r)["t"]
            M.lines!(ax, t, v; label = length(rids) > 1 ? "$n ($r)" : n)
        end
        (t0 !== nothing || t1 !== nothing) && M.xlims!(ax, t0, t1)
        M.axislegend(ax; position = :rb, labelsize = 10)
    end
    dir = under_out("runs", rids[1], "plots"); mkpath(dir)
    png = joinpath(dir, "plot-" * string(round(Int, time())) * ".png")
    M.save(png, fig)
    return png
end

function t_delete(a, ctx)
    rid = checkid(a["run_id"])
    d = under_out("runs", rid)
    isdir(d) || throw(ArgumentError("no run $rid"))
    rm(d; recursive = true); delete!(RUNCACHE, rid)
    return text("Run $rid deleted.")
end

tool(name, desc, props, req, f; ann = nothing) =
    MCPTool(name = name, description = desc, input_schema = schema(props; required = req), handler = guard(f),
            annotations = ann)

const RO = Dict{String,Any}("readOnlyHint" => true)

const TOOLS = [
    tool("list_cases", "The power-system cases and model options this server offers.", Dict{String,Any}(), String[],
         (a, c) -> text(CASES_TEXT); ann = RO),
    tool("build_model", "Build an EMT model of the IEEE 39-bus system with the chosen options and solve its operating point. Waits up to wait_seconds, then returns a job id to follow with job_status. dry_run = true only checks the request.",
         Dict{String,Any}(
             "preset" => P("string", "ieee (block-diagram controllers) or full_physical (physical exciters, steam and hydro units)"; enum = ["ieee", "full_physical"]),
             "dc_exciter" => P("boolean", "physical self-excited DC exciters (overrides the preset)"),
             "brushless_exciter" => P("boolean", "physical brushless exciters (overrides the preset)"),
             "steam" => P("boolean", "physical steam units (overrides the preset)"),
             "hydro" => P("boolean", "physical hydro unit with elastic penstock (overrides the preset)"),
             "loads" => P("string", "load model"; enum = collect(LOADS)),
             "lines" => P("string", "lumped pi lines or travelling-wave (Bergeron) lines"; enum = collect(LINES)),
             "unbalance" => P("string", "zero-sequence network for unbalanced faults: none, physical grounding, PowerFactory grounding"; enum = collect(UNBALANCE)),
             "machine_data" => P("string", "machine circuits from PowerFactory's winding data or from the case's GENROU data"; enum = collect(MACHINE_DATA)),
             "load_scale" => P("number", "scale every load's P and Q before the operating point is solved (0.5-1.5)"),
             "parameter_changes" => P("object", "{component: {parameter: value}}, e.g. {\"IEEET1_2\": {\"KA\": 50}}; see describe"),
             "dry_run" => P("boolean", "only check the request"),
             "wait_seconds" => P("number", "how long to wait for the build before returning (default 20)")),
         String[], t_build),
    tool("operating_point", "The operating point of a built model: generator terminal voltage, power, field voltage, speed, rotor angle; bus voltages.",
         Dict{String,Any}("model_id" => P("string", "model id")), ["model_id"], t_opoint; ann = RO),
    tool("describe", "List a model's components, or one component's parameters and states.",
         Dict{String,Any}("model_id" => P("string", "model id"), "component" => P("string", "component name, e.g. IEEET1_2")),
         ["model_id"], t_describe; ann = RO),
    tool("run_simulation", "Simulate a built model with faults. Returns the wall-time estimate; runs above max_wall_minutes need allow_long = true, which must only be set after the user agrees. Waits up to wait_seconds, then returns a job id.",
         Dict{String,Any}(
             "model_id" => P("string", "model id"), "tstop" => P("number", "end time in s"),
             "faults" => P("array", "bus faults: {bus, kind (abc, abcg, ag, bg, cg, ab, bc, ca, abg, bcg, cag; a/b/c = phases, g = ground), R (pu, > 0, default 0.001), Rg, t_on, t_off (s), clearing (instant or current_zero)}";
                           items = Dict{String,Any}("type" => "object")),
             "save_dt" => P("number", "saving interval in s (default 0.001)"),
             "h" => P("number", "fixed step in s (default: from the model's fastest dynamics)"),
             "hscale" => P("number", "scale of the automatic step (default 1)"),
             "tol" => P("number", "Newton tolerance (default 1e-13)"),
             "max_wall_minutes" => P("number", "hard limit on wall time (default 10)"),
             "allow_long" => P("boolean", "start even if the estimate exceeds max_wall_minutes (only after the user agrees)"),
             "dry_run" => P("boolean", "only check the request and estimate the time"),
             "wait_seconds" => P("number", "how long to wait before returning (default 20)")),
         ["model_id", "tstop"], t_run),
    tool("job_status", "State of a build or run job; waits up to wait_seconds (max 60) for it to finish.",
         Dict{String,Any}("job_id" => P("string", "job id"), "wait_seconds" => P("number", "seconds to wait (0-60)")),
         ["job_id"], t_status; ann = RO),
    tool("cancel_job", "Stop a job. force = true stops the whole simulation worker (it restarts on the next request).",
         Dict{String,Any}("job_id" => P("string", "job id"), "force" => P("boolean", "stop the worker process")),
         ["job_id"], t_cancel; ann = Dict{String,Any}("destructiveHint" => true)),
    tool("list_runs", "Built models and stored runs.", Dict{String,Any}(), String[], t_list_runs; ann = RO),
    tool("list_signals", "The signals stored in a run, by name.",
         Dict{String,Any}("run_id" => P("string", "run id"), "include_raw_states" => P("boolean", "also list every state")),
         ["run_id"], t_signals; ann = RO),
    tool("get_signals", "Signals of a stored run: a summary (initial, minimum, maximum, final with times) and a bounded table that keeps peaks and the fault instants.",
         Dict{String,Any}("run_id" => P("string", "run id"),
                          "signals" => P("array", "names from list_signals, e.g. GENROU_3.speed, bus16.voltage"; items = Dict{String,Any}("type" => "string")),
                          "t_from" => P("number", "start time in s"), "t_to" => P("number", "end time in s"),
                          "max_points" => P("integer", "table rows (default 150, at most 500)"),
                          "reference_machine" => P("string", "take rotor angles relative to this machine")),
         ["run_id", "signals"], t_get; ann = RO),
    tool("energy_audit", "The energy balance of a stored run: per component and for the system, and what the numbers mean.",
         Dict{String,Any}("run_id" => P("string", "run id")), ["run_id"], t_audit; ann = RO),
    tool("plot", "Plot signals of one or more stored runs (one panel per unit) and return the image.",
         Dict{String,Any}("run_ids" => P("array", "run ids"; items = Dict{String,Any}("type" => "string")),
                          "signals" => P("array", "signal names"; items = Dict{String,Any}("type" => "string")),
                          "t_from" => P("number", "start time in s"), "t_to" => P("number", "end time in s"),
                          "reference_machine" => P("string", "rotor angles relative to this machine"),
                          "title" => P("string", "figure title")),
         ["run_ids", "signals"], t_plot; ann = RO),
    tool("delete_run", "Delete a stored run and its plots.", Dict{String,Any}("run_id" => P("string", "run id")), ["run_id"],
         t_delete; ann = Dict{String,Any}("destructiveHint" => true)),
]

const INSTRUCTIONS = """
PHPS is an electromagnetic-transient simulator of power systems in port-Hamiltonian form. Typical use: list_cases,
build_model, operating_point, run_simulation (with faults), then get_signals, energy_audit and plot on the stored run.
Runs are slow (tens of seconds of computer time per simulated second): start with short runs (0.3-1 s); never set
allow_long without asking the user. Report results in power-engineering terms with units and bases.
"""

server = mcp_server(name = "phps", version = "0.1.0", tools = TOOLS, instructions = INSTRUCTIONS,
                    description = "PHPS EMT power-system simulator")
@async try
    start_worker()                               # warm up in the background; the first build still compiles
catch e
    @warn "worker start failed" exception = e
end
start!(server; transport = StdioTransport(input = PROTO_IN, output = PROTO_OUT))
w = WORKER[]
w === nothing || kill(w.proc)
