# The model handed to analysis tools (docs/analysis.md): everything an optimiser, an eigenvalue routine or
# another language needs about the model at one state, with every state named, in one call and one file.
#   phs_analysis_model(m, x0, u)  or  phs_analysis_model(case_folder)   ->  named tuple (below)
#   phs_export(path, am)          ->  a .jld2 file, or a folder of CSV files (states.csv, A.csv, J.csv, ...)
# Nothing here is a new model: the rates are `phs_rhs`, the structure is `phs_structure`, the linearisation is the
# automatic derivative of the rates. The export reports its own consistency checks (`am.checks`).

"""
    phs_analysis_model(m, x0, u; hessian = true, steam = :exergy) -> NamedTuple
    phs_analysis_model(dir; scenario = "fault_bus16_3ph.json", kwargs...)

The model `m` (a `PHSystem` or a `PHControlled`; or the full model of the case folder `dir`) at the state `x0`, for
analysis outside the simulator. Fields:

| Field | What it is |
|---|---|
| `names`, `component`, `kind` | per state: its name, the device it belongs to, and `:machine`, `:excitation`, `:prime_mover`, `:converter` or `:network` (the network side: nodes, branches, transformers, loads, motors) |
| `in_form` | per state: whether its row is in the port-Hamiltonian form (signal rows of controllers and the devices in `outside` are not) |
| `x0`, `scale`, `rate0` | the state, a typical size of each state, and the rates at `x0` (zero at an operating point) |
| `A` | the linearisation: derivative of the rates with respect to the states at `x0` (automatic differentiation) |
| `J`, `R`, `G`, `input`, `effort` | the structure at `x0`: on the rows in the form, rates = `(J - R) effort + G input` (less `Pf * vf` where a steam unit has feedthrough) |
| `Q` | derivative of the efforts with respect to the states: the Hessian of the stored energy of the rows in the form (central differences; `nothing` if `hessian = false`) |
| `ports`, `outside` | the name of each input, and what is outside the form |
| `checks` | consistency of this export (see below) |
| `rate`, `energy`, `structure` | functions of a state, for analyses away from `x0` |

`checks`: `operating_point` (largest rate over its state's scale), `structure` (largest difference between the rates
and the structure's right-hand side on the rows in the form, over the scale), `J_skew` (largest entry of `J + J'`),
`R_asymmetry` and `R_min` (asymmetry and smallest eigenvalue of `R` on those rows, over its norm; in the model's
coordinates), `Q_asymmetry` (scaled by the states' sizes, relative). With no row in the form the structure checks are zero.
"""
function phs_analysis_model(m::Union{PHSystem,PHControlled}, x0::AbstractVector, u::PHInput; hessian::Bool = true,
                            steam::Symbol = :exergy)
    x0 = collect(Float64, x0); N = length(x0)
    names = String.(phs_state_names(m)); sc = phs_scales(m, x0)
    pos = filter(>(0), sc)                     # a state whose size is zero at x0 still needs a positive one: the smallest
    isempty(pos) && throw(ArgumentError("phs_analysis_model: every state scale is zero"))
    sc = [v > 0 ? v : minimum(pos) for v in sc]   # positive size (the states have mixed units, so no common floor)
    kind, component = _analysis_labels(m, names)
    structure = m isa PHControlled ? (x -> phs_structure(m, x, u; steam)) : (x -> phs_structure(m, x, u))
    rate = x -> phs_rhs(m, x, u)
    energy = m isa PHControlled ? (x -> phs_structure_energy(m, x; steam)) : (x -> phs_structure_energy(m, x))
    f0 = rate(x0)
    A = ForwardDiff.jacobian(rate, x0)
    st = structure(x0)
    idx = collect(st.idx); in_form = falses(N); in_form[idx] .= true
    feed = hasproperty(st, :Pf) && !isempty(st.vf)
    rhs = (st.J - st.R) * st.z + st.G * st.v - (feed ? st.Pf * st.vf : zeros(N))
    Q = nothing; qasym = NaN
    if hessian
        Q = zeros(N, N)
        for j in 1:N
            h = 1e-5 * sc[j]
            xp = copy(x0); xp[j] += h; xm = copy(x0); xm[j] -= h
            Q[:, j] = (structure(xp).z .- structure(xm).z) ./ 2h
        end
        D = Diagonal(sc); qasym = maximum(abs, D * (Q .- Q') * D) / max(maximum(abs, D * Q * D), floatmin())
    end
    RI = Symmetric(Matrix(st.R[idx, idx]))
    checks = (; operating_point = maximum(abs.(f0) ./ sc),
              structure = isempty(idx) ? 0.0 : maximum(abs.(rhs[idx] .- f0[idx]) ./ sc[idx]),
              J_skew = maximum(abs, st.J + st.J'),
              R_asymmetry = isempty(idx) ? 0.0 : maximum(abs, st.R[idx, idx] - st.R[idx, idx]') / max(opnorm(Matrix(RI)), floatmin()),
              R_min = isempty(idx) ? 0.0 : eigmin(RI) / max(opnorm(Matrix(RI)), floatmin()),
              Q_asymmetry = qasym)
    return (; names, component, kind, in_form, x0, scale = sc, rate0 = f0, A, J = Matrix(st.J), R = Matrix(st.R),
            G = Matrix(st.G), input = collect(st.v), effort = collect(st.z), Q,
            Pf = feed ? Matrix(st.Pf) : nothing, vf = feed ? collect(st.vf) : nothing,
            ports = string.(st.ports), outside = string.(st.excluded), checks, rate, energy, structure)
end
function phs_analysis_model(dir::AbstractString; scenario::AbstractString = "fault_bus16_3ph.json", kwargs...)
    fm = phs_full_model(dir; scenario)
    return phs_analysis_model(fm.cs, fm.x0, fm.u; kwargs...)
end

# kind and device of every state, from the model's own layout: the blocks of the machines, excitation units, prime
# movers and converters carry their unit's name; every other state belongs to the network side (nodes, branches,
# transformers, loads, motors) and is named after its element, without the axis suffix (`phi_Line_17_Im` -> `phi_Line_17`)
function _analysis_labels(m::Union{PHSystem,PHControlled}, names)
    N = length(names); kind = fill(:network, N)
    component = [(endswith(n, "_Re") || endswith(n, "_Im")) ? String(n[1:end-3]) : String(n) for n in names]
    block!(k, name, off, len) = (kind[off:off+len-1] .= k; component[off:off+len-1] .= name)
    s = m isa PHControlled ? m.s : m
    for mc in s.machines
        block!(:machine, mc.name, mc.offset, phs_nstates(mc))
    end
    if m isa PHControlled
        for j in eachindex(s.machines)
            for (c, off, k) in ((m.exc[j], m.xoff[j], :excitation), (m.gov[j], m.goff[j], :prime_mover))
                c === nothing || block!(k, c.name, off, nstates(c))
            end
        end
        for (c, off) in zip(m.conv, m.coff)
            block!(:converter, c.name, off, nstates(c))
        end
    end
    return kind, component
end

"""
    phs_export(path, am) -> path

Write the analysis model `am` (`phs_analysis_model`). A path ending in `.jld2` gives one JLD2 file (arrays and
strings; readable without this package). Any other path is taken as a folder and filled with plain CSV files for other
tools: `states.csv` (a header line, then index from 1, name, component, kind, in_form, x0, scale, rate0, effort), one
file per matrix without a header (`A.csv`, `J.csv`, `R.csv`, `G.csv`, and `Q.csv`, `Pf.csv` if present), `inputs.csv`
(port, value), `feedthrough.csv` (the values `vf` that `Pf` multiplies, if present), `outside.csv` and `checks.csv`.
Files of an earlier export into the same folder that this one does not write are removed.
"""
function phs_export(path::AbstractString, am)
    if endswith(lowercase(path), ".jld2")
        mkpath(dirname(abspath(path)))
        d = Dict{String,Any}("names" => am.names, "component" => am.component, "kind" => String.(am.kind),
                             "in_form" => collect(am.in_form), "x0" => am.x0, "scale" => am.scale, "rate0" => am.rate0,
                             "A" => am.A, "J" => am.J, "R" => am.R, "G" => am.G, "input" => am.input,
                             "effort" => am.effort, "ports" => am.ports, "outside" => am.outside,
                             "checks" => Dict(String(k) => v for (k, v) in pairs(am.checks)))
        am.Q === nothing || (d["Q"] = am.Q)
        am.Pf === nothing || (d["Pf"] = am.Pf; d["vf"] = am.vf)
        if hasproperty(am, :Gp)                      # a subsystem (`phs_subsystem`): its ports and its equations
            merge!(d, Dict("index" => am.index, "Gp" => am.Gp, "port_effort" => am.port_effort, "port_power" => am.port_power,
                           "sources" => am.sources, "feedthrough" => am.feedthrough, "A_outside" => am.A_outside,
                           "outside_states" => am.outside_states, "varies" => collect(am.varies), "equations" => phs_equations(am)))
        end
        JLD2.save(path, d)
        return path
    end
    mkpath(path)
    for f in ("Q.csv", "Pf.csv", "feedthrough.csv", "Gp.csv", "ports.csv", "A_outside.csv", "outside_states.csv", "equations.md")   # optional files: none may survive from an earlier export
        rm(joinpath(path, f); force = true)
    end
    num(v) = repr(Float64(v))
    q(s) = "\"" * replace(String(s), "\"" => "\"\"") * "\""
    open(joinpath(path, "states.csv"), "w") do io
        println(io, "index,name,component,kind,in_form,x0,scale,rate0,effort")
        for i in eachindex(am.names)
            println(io, join((i, q(am.names[i]), q(am.component[i]), am.kind[i], am.in_form[i] ? 1 : 0, num(am.x0[i]),
                              num(am.scale[i]), num(am.rate0[i]), num(am.effort[i])), ","))
        end
    end
    sub = hasproperty(am, :Gp)
    for (nm, M) in (("A", am.A), ("J", am.J), ("R", am.R), ("G", am.G), ("Q", am.Q), ("Pf", am.Pf),
                    ("Gp", sub ? am.Gp : nothing), ("A_outside", sub ? am.A_outside : nothing))
        M === nothing && continue
        open(joinpath(path, nm * ".csv"), "w") do io
            for i in axes(M, 1)
                println(io, join((num(M[i, j]) for j in axes(M, 2)), ","))
            end
        end
    end
    open(joinpath(path, "inputs.csv"), "w") do io
        println(io, "port,value")
        src = sub ? am.sources : am.ports
        for k in eachindex(src); println(io, q(src[k]), ",", num(am.input[k])); end
    end
    if sub
        open(joinpath(path, "ports.csv"), "w") do io
            println(io, "port_effort_of,effort,power_into_the_group")
            for k in eachindex(am.ports); println(io, q(am.ports[k]), ",", num(am.port_effort[k]), ",", num(am.port_power[k])); end
        end
        open(joinpath(path, "outside_states.csv"), "w") do io
            println(io, "state")
            for s in am.outside_states; println(io, q(s)); end
        end
        open(io -> phs_equations(io, am), joinpath(path, "equations.md"), "w")
        if am.feedthrough !== nothing && any(!iszero, am.feedthrough)
            open(joinpath(path, "feedthrough.csv"), "w") do io
                println(io, "value")
                for v in am.feedthrough; println(io, num(v)); end
            end
        end
    end
    if am.vf !== nothing
        open(joinpath(path, "feedthrough.csv"), "w") do io
            println(io, "value")
            for v in am.vf; println(io, num(v)); end
        end
    end
    open(joinpath(path, "outside.csv"), "w") do io
        println(io, "outside_the_form")
        for s in am.outside; println(io, q(s)); end
    end
    open(joinpath(path, "checks.csv"), "w") do io
        println(io, "check,value")
        for (k, v) in pairs(am.checks); println(io, k, ",", num(v)); end
    end
    return path
end

# ---------------------------------------------------------------------------------------------------------------------
# Any device, or any group of devices, as its own port-Hamiltonian system with its ports to the rest of the model.

"""
    phs_subsystem(am, select) -> NamedTuple

The part of the analysis model `am` (`phs_analysis_model`) made of the chosen states, as a system of its own:

    rates = (J - R) effort + Gp * (efforts of the neighbouring states) + G * input      (rows in the form)

`select` is a device name or a vector of device or state names (`"GENROU_4"`, `["GENROU_4", "IEEET1_4"]`), a kind
(`:machine`), a vector of state indices, or a function `(name, component, kind) -> Bool`.

Fields, beside those of `am` restricted to the chosen states (`names`, `component`, `kind`, `in_form`, `x0`, `scale`,
`rate0`, `effort`, `J`, `R`, `Q`, `A`):

| Field | What it is |
|---|---|
| `index` | the chosen states' indices in `am` |
| `Gp`, `ports`, `port_effort`, `port_power` | the interconnection ports: one per neighbouring state whose effort acts on the group, its name, its effort at `x0`, and the power it delivers to the group there |
| `G`, `sources`, `input` | the external inputs that act on the group |
| `feedthrough` | the term subtracted on the group's rows where a steam unit has feedthrough (zero otherwise) |
| `A_outside`, `outside_states` | for the linearisation: how the group's rates depend on states outside it |
| `varies` | for each entry of `[J - R, Gp]`: whether it changes with the group's state (sampled; the numbers are those at `x0`) |
| `shared_loss_with` | the neighbouring devices that share a loss term with the group (a steam unit with its machine's rotor, for example): include them for a group whose ports all conserve power |
| `checks` | `structure` (the rates against the right-hand side above, rows in the form, over the scale), `J_skew`, `R_min`, `shared_loss_entries` (the number of loss entries shared with neighbours: zero when every port conserves power), `varies_samples` (how many of the two sampled states `varies` rests on), `energy_separable` (the part of the group's efforts that depends on outside states, scaled and relative: zero when the group has an energy of its own) |
| `rate`, `energy`, `structure` | functions of the group's state, the rest of the model held at `x0` |
"""
function phs_subsystem(am, select)
    N = length(am.names); S = _analysis_select(am, select)
    isempty(S) && throw(ArgumentError("phs_subsystem: nothing selected by $(repr(select))"))
    Rm = setdiff(1:N, S)
    feed0 = am.Pf === nothing ? zeros(N) : am.Pf * am.vf
    part(st) = (; M = st.J - st.R, z = st.z, G = st.G, v = st.v,
                feed = (hasproperty(st, :Pf) && !isempty(st.vf)) ? st.Pf * st.vf : zeros(N))
    M = am.J - am.R
    pc = [k for k in Rm if any(!iszero, view(M, S, k))]
    sc_ = [k for k in axes(am.G, 2) if any(!iszero, view(am.G, S, k))]
    Gp = M[S, pc]; z = am.effort
    rhs = M[S, S] * z[S] + Gp * z[pc] + am.G[S, sc_] * am.input[sc_] - feed0[S]
    form = findall(am.in_form[S])
    oc = [k for k in Rm if any(!iszero, view(am.A, S, k))]
    embed(xs) = (x = Vector{promote_type(eltype(xs), Float64)}(am.x0); x[S] = xs; x)     # a copy: the model's x0 is never touched
    structure = xs -> begin
        p = part(am.structure(collect(Float64, embed(xs))))
        (; J = (p.M[S, S] - p.M[S, S]') / 2, R = -(p.M[S, S] + p.M[S, S]') / 2, Gp = p.M[S, pc], effort = p.z[S],
         port_effort = p.z[pc], G = p.G[S, sc_], input = p.v[sc_], feedthrough = p.feed[S])
    end
    # which entries of [J - R, Gp] change with the group's state (two sampled states near x0)
    varies = falses(length(S), length(S) + length(pc)); nsamp = 0
    B0 = hcat(M[S, S], Gp)
    for sgn in (1.0, -1.0)
        xs = am.x0[S] .+ sgn * 1e-6 .* am.scale[S] .* (1 .+ (1:length(S)) ./ (2 * length(S)))
        st = try structure(xs) catch; nothing end          # a sample outside the model's domain is skipped and counted
        st === nothing && continue
        nsamp += 1
        B1 = hcat(st.J - st.R, st.Gp)
        varies .|= abs.(B1 .- B0) .> 1e-9 .* abs.(B0) .+ 1e-300          # relative to the entry itself (the entries span many decades)
    end
    RS = Symmetric((am.R[S, S][form, form] + am.R[S, S][form, form]') / 2)
    Ds = Diagonal(am.scale[S])
    checks = (; structure = isempty(form) ? 0.0 : maximum(abs.(rhs[form] .- am.rate0[S][form]) ./ am.scale[S][form]),
              J_skew = maximum(abs, am.J[S, S] + am.J[S, S]'),
              R_min = isempty(form) ? 0.0 : eigmin(RS) / max(opnorm(Matrix(RS)), floatmin()),
              shared_loss_entries = Float64(count(!iszero, view(am.R, S, Rm))),
              varies_samples = Float64(nsamp),
              energy_separable = (am.Q === nothing || isempty(Rm)) ? (am.Q === nothing ? NaN : 0.0) :
                                 maximum(abs, Ds * am.Q[S, Rm] * Diagonal(am.scale[Rm])) / max(maximum(abs, Ds * am.Q[S, S] * Ds), floatmin()))
    return (; index = S, names = am.names[S], component = am.component[S], kind = am.kind[S], in_form = am.in_form[S],
            x0 = am.x0[S], scale = am.scale[S], rate0 = am.rate0[S], effort = z[S], A = am.A[S, S], J = am.J[S, S],
            R = am.R[S, S], Q = am.Q === nothing ? nothing : am.Q[S, S], Gp, ports = am.names[pc], port_effort = z[pc],
            port_power = [dot(z[S], view(Gp, :, k)) * z[pc[k]] for k in eachindex(pc)], G = am.G[S, sc_],
            sources = am.ports[sc_], input = am.input[sc_], feedthrough = feed0[S], A_outside = am.A[S, oc],
            outside_states = am.names[oc], varies, checks,
            shared_loss_with = unique(am.component[[k for k in Rm if any(!iszero, view(am.R, S, k))]]), rate = xs -> am.rate(embed(xs))[S],
            energy = xs -> am.energy(embed(xs)), structure,
            Pf = nothing, vf = nothing, outside = String[n for n in am.names[S][.!am.in_form[S]]])
end

function _analysis_select(am, select)
    N = length(am.names)
    select isa AbstractVector{<:Integer} && return collect(Int, select)
    select isa Function && return [i for i in 1:N if select(am.names[i], am.component[i], am.kind[i])]
    select isa Symbol && return findall(==(select), am.kind)
    want = select isa AbstractString ? [String(select)] : String.(collect(select))
    miss = [w for w in want if !(w in am.component) && !(w in am.names)]
    isempty(miss) || throw(ArgumentError("phs_subsystem: no device or state named $(join(miss, ", "))"))
    return [i for i in 1:N if am.component[i] in want || am.names[i] in want]
end

"""
    phs_equations(sub) -> String
    phs_equations(io, sub; digits = 6)

The equations of a subsystem (`phs_subsystem`) written out with the names of its states, as Markdown: the efforts
(the stored energy's derivatives, linear where the energy is quadratic), one line per state for the rows in the
port-Hamiltonian form (the lossless exchange, the losses, the ports and the inputs kept apart), the linearised rows of
the signal states, and the power through every port. The numbers are those at the state `x0` of the export; an entry
that changes with the group's state is marked `*`.
"""
function phs_equations(io::IO, sub; digits::Int = 6)
    f(v) = Printf.format(Printf.Format("%.$(digits)g"), abs(v))
    sg(v) = (v < 0 ? "-" : "") * f(v)
    term(v, what, star) = (v < 0 ? " - " : " + ") * f(v) * (star ? "*" : "") * " " * what
    n = length(sub.names); form = findall(sub.in_form); sig = findall(.!sub.in_form)
    tolM = 1e-12 * max(maximum(abs, sub.J; init = 0.0), maximum(abs, sub.R; init = 0.0), floatmin())
    println(io, "# Port-Hamiltonian equations of: ", join(unique(sub.component), ", "), "\n")
    println(io, n, " states, ", length(form), " in the port-Hamiltonian form, ", length(sub.ports), " interconnection ports, ",
            length(sub.sources), " external inputs. Numbers at the exported state; `*` marks an entry that changes with the group's state.\n")
    println(io, "Form of the rows below: `d/dt x = (J - R) e + Gp u_port + G u`, with `e` the efforts (derivatives of the stored energy).\n")
    println(io, "## Efforts\n")
    for i in form
        s = "e[" * sub.names[i] * "] = " * sg(sub.effort[i])
        if sub.Q !== nothing
            tq = 1e-9 * maximum(abs, view(sub.Q, i, :); init = 0.0)
            for j in 1:n
                abs(sub.Q[i, j]) > tq && abs(sub.Q[i, j]) > 0 && (s *= term(sub.Q[i, j], "(" * sub.names[j] * (sub.x0[j] < 0 ? " + " : " - ") * f(sub.x0[j]) * ")", false))
            end
        end
        println(io, "- `", s, "`")
    end
    println(io, "\n## Rows in the port-Hamiltonian form\n")
    for i in form
        lossless = ""; loss = ""; prt = ""; inp = ""
        for j in 1:n
            abs(sub.J[i, j]) > tolM && (lossless *= term(sub.J[i, j], "e[" * sub.names[j] * "]", sub.varies[i, j]))
            abs(sub.R[i, j]) > tolM && (loss *= term(-sub.R[i, j], "e[" * sub.names[j] * "]", sub.varies[i, j]))
        end
        for k in eachindex(sub.ports)
            abs(sub.Gp[i, k]) > tolM && (prt *= term(sub.Gp[i, k], "e[" * sub.ports[k] * "]", sub.varies[i, n+k]))
        end
        for k in eachindex(sub.sources)
            abs(sub.G[i, k]) > 0 && (inp *= term(sub.G[i, k], "u[" * sub.sources[k] * "]", false))
        end
        abs(sub.feedthrough[i]) > 0 && (inp *= term(-sub.feedthrough[i], "(feedthrough)", false))
        println(io, "- `d/dt ", sub.names[i], " =`")
        isempty(lossless) || println(io, "  - exchange: `", lossless, "`")
        isempty(loss) || println(io, "  - losses: `", loss, "`")
        isempty(prt) || println(io, "  - ports: `", prt, "`")
        isempty(inp) || println(io, "  - inputs: `", inp, "`")
    end
    if !isempty(sig)
        println(io, "\n## Signal rows (not in the form), linearised at the exported state\n")
        for i in sig
            s = ""
            ta = 1e-12 * max(maximum(abs, view(sub.A, i, :); init = 0.0), maximum(abs, view(sub.A_outside, i, :); init = 0.0))
            for j in 1:n; abs(sub.A[i, j]) > ta && (s *= term(sub.A[i, j], "Δ" * sub.names[j], false)); end
            for k in eachindex(sub.outside_states); abs(sub.A_outside[i, k]) > ta && (s *= term(sub.A_outside[i, k], "Δ" * sub.outside_states[k], false)); end
            println(io, "- `d/dt ", sub.names[i], " =", s, "`")
        end
    end
    println(io, "\n## Ports: the neighbouring effort and the power it delivers to the group at the exported state\n")
    println(io, "| Port (effort of) | Effort | Power into the group |\n|---|---|---|")
    for k in eachindex(sub.ports)
        println(io, "| ", sub.ports[k], " | ", Printf.format(Printf.Format("%.$(digits)g"), sub.port_effort[k]), " | ", Printf.format(Printf.Format("%.$(digits)g"), sub.port_power[k]), " |")
    end
    isempty(sub.shared_loss_with) || println(io, "\nThe group shares loss terms with: ", join(sub.shared_loss_with, ", "),
                                             ". Its ports to them do not conserve power; include them for a group whose ports all do.")
    println(io, "\n## Checks\n")
    for (k, v) in pairs(sub.checks); println(io, "- ", k, ": ", Printf.format(Printf.Format("%.2e"), v)); end
    return nothing
end
phs_equations(sub; kwargs...) = (io = IOBuffer(); phs_equations(io, sub; kwargs...); String(take!(io)))
