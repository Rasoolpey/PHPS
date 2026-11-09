# Shared by the PHPS MCP server (server.jl) and its simulation worker (worker.jl): paths, ids, atomic writes, the
# build options and their checks. Loads no simulator code.

const ROOT = normpath(joinpath(@__DIR__, ".."))
const OUT = joinpath(ROOT, "outputs", "mcp")
const MODELS = joinpath(OUT, "models")
const RUNS = joinpath(OUT, "runs")
const BASE_CASE = joinpath(ROOT, "cases", "IEEE39Bus_FullModel")

const ID_RE = r"^[A-Za-z0-9_-]{1,64}$"
checkid(id) = (id isa AbstractString && occursin(ID_RE, id)) ? String(id) :
              throw(ArgumentError("invalid id $(repr(id)): letters, digits, '_' and '-' only, at most 64"))

"""A path under `outputs/mcp`, refusing anything that would leave it."""
function under_out(parts...)
    p = normpath(joinpath(OUT, parts...))
    startswith(p, normpath(OUT)) || throw(ArgumentError("path leaves outputs/mcp"))
    return p
end

newid(prefix) = prefix * "-" * string(round(Int, time() * 1000), base = 36) * "-" * string(rand(UInt16), base = 36)

"""Write `bytes_or_string` to `path` atomically (a temporary file, then a rename)."""
function atomic_write(path, data)
    mkpath(dirname(path))
    tmp = path * ".tmp"
    open(io -> write(io, data), tmp, "w")
    mv(tmp, path; force = true)
    return path
end

# ---- build options -------------------------------------------------------------------------------------------------

const LOADS = ("series_all", "parallel", "series", "composite")
const LINES = ("lumped", "travelling")
const UNBALANCE = ("none", "physical", "powerfactory")
const MACHINE_DATA = ("powerfactory_windings", "case_genrou")
const PROTECTED = Set(["bus", "idx", "name", "syn", "gen", "Sn", "Vn", "omega_b", "V0"])

"""
    resolve_build(args) -> Dict

The build specification from the tool arguments: a preset (`ieee` = IEEE block-diagram exciters and governors,
`full_physical` = every physical unit, as cases/IEEE39Bus_FullModel) and explicit switches that override it.
Throws `ArgumentError` with a plain message on anything not supported.
"""
function resolve_build(a)
    get_(k, d) = haskey(a, k) && a[k] !== nothing ? a[k] : d
    preset = String(get_("preset", "ieee"))
    preset in ("ieee", "full_physical") || throw(ArgumentError("preset must be \"ieee\" or \"full_physical\""))
    phys = preset == "full_physical"
    spec = Dict{String,Any}(
        "case" => "IEEE39",
        "preset" => preset,
        "machine_data" => String(get_("machine_data", "powerfactory_windings")),
        "loads" => String(get_("loads", "series_all")),
        "lines" => String(get_("lines", "lumped")),
        "dc_exciter" => Bool(get_("dc_exciter", phys)),
        "brushless_exciter" => Bool(get_("brushless_exciter", phys)),
        "steam" => Bool(get_("steam", phys)),
        "hydro" => Bool(get_("hydro", phys)),
        "unbalance" => String(get_("unbalance", "none")),
        "load_scale" => Float64(get_("load_scale", 1.0)),
        "parameter_changes" => Dict{String,Any}(String(k) => Dict{String,Any}(String(p) => v for (p, v) in pairs(d))
                                                for (k, d) in pairs(get_("parameter_changes", Dict()))))
    spec["machine_data"] in MACHINE_DATA || throw(ArgumentError("machine_data must be one of $(join(MACHINE_DATA, ", "))"))
    spec["loads"] in LOADS || throw(ArgumentError("loads must be one of $(join(LOADS, ", "))"))
    spec["lines"] in LINES || throw(ArgumentError("lines must be one of $(join(LINES, ", "))"))
    spec["unbalance"] in UNBALANCE || throw(ArgumentError("unbalance must be one of $(join(UNBALANCE, ", "))"))
    (isfinite(spec["load_scale"]) && 0.5 <= spec["load_scale"] <= 1.5) ||
        throw(ArgumentError("load_scale must be between 0.5 and 1.5 (all loads scaled before the operating point is solved)"))
    return spec
end

"""Check and apply parameter changes and the load scale to the case JSON (a Dict). Returns notes for the user."""
function apply_changes!(raw::Dict, spec::Dict)
    notes = String[]
    comps = raw["components"]
    for (comp, ch) in spec["parameter_changes"]
        haskey(comps, comp) || throw(ArgumentError("no component named $comp (see `describe`)"))
        p = comps[comp]["params"]
        for (k, v) in ch
            haskey(p, k) || throw(ArgumentError("$comp has no parameter $k; its parameters: $(join(sort(collect(keys(p))), ", "))"))
            k in PROTECTED && throw(ArgumentError("$comp.$k describes the network connection or rating and cannot be changed here"))
            v isa Real && isfinite(v) || throw(ArgumentError("$comp.$k must be a finite number"))
            old = p[k]
            old isa Real || throw(ArgumentError("$comp.$k is not a number"))
            (old != 0 && sign(v) != sign(old) && v != 0) &&
                push!(notes, "$comp.$k changes sign ($old -> $v); check this is intended")
            p[k] = Float64(v)
            push!(notes, "$comp.$k: $old -> $v")
            startswith(comps[comp]["type"], "GEN") && spec["machine_data"] == "powerfactory_windings" && k != "D" &&
                push!(notes, "$comp.$k has no effect with machine_data = powerfactory_windings (the machine circuits come from PowerFactory's winding data); use machine_data = case_genrou")
        end
    end
    s = spec["load_scale"]
    if s != 1.0
        for l in raw["PQ"]
            l["p0"] *= s; l["q0"] *= s
        end
        for (_, c) in comps
            c["type"] == "COMPLEXLOAD" || continue
            c["params"]["P0"] *= s; c["params"]["Q0"] *= s
        end
        push!(notes, "all loads (P and Q) scaled by $s before the operating point is solved")
    end
    return notes
end

# ---- faults -----------------------------------------------------------------------------------------------------------

const KINDS = ("abc", "abcg", "ag", "bg", "cg", "ab", "bc", "ca", "abg", "bcg", "cag")
const GROUND = Set(["abcg", "ag", "bg", "cg", "abg", "bcg", "cag"])

"""Check the faults of a run request against the model's options; returns them as Dicts with every field set."""
function resolve_faults(fs, spec::Dict, tstop::Float64, buses)
    out = Dict{String,Any}[]
    for (i, f) in enumerate(fs)
        g(k, d) = haskey(f, k) && f[k] !== nothing ? f[k] : d
        bus = Int(g("bus", 0))
        bus in buses || throw(ArgumentError("fault $i: bus $bus is not in the network"))
        kind = String(g("kind", "abc"))
        kind in KINDS || throw(ArgumentError("fault $i: kind must be one of $(join(KINDS, ", ")) (a = phase a, g = ground)"))
        kind in GROUND && spec["unbalance"] == "none" &&
            throw(ArgumentError("fault $i: a ground fault needs the zero-sequence network; build the model with unbalance = physical or powerfactory"))
        R = Float64(g("R", 0.001)); Rg = Float64(g("Rg", R))
        (R > 0 && Rg > 0) || throw(ArgumentError("fault $i: R and Rg must be positive (per unit on the system base; 0.001 is a near-bolted fault)"))
        ton, toff = Float64(g("t_on", 0.1)), Float64(g("t_off", 0.2))
        0 <= ton < toff <= tstop || throw(ArgumentError("fault $i: need 0 <= t_on < t_off <= tstop"))
        clr = String(g("clearing", "instant"))
        clr in ("instant", "current_zero") || throw(ArgumentError("fault $i: clearing must be instant or current_zero"))
        push!(out, Dict{String,Any}("bus" => bus, "kind" => kind, "R" => R, "Rg" => Rg, "t_on" => ton, "t_off" => toff,
                                    "clearing" => clr))
    end
    return out
end
