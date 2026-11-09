# Exciters and governors on the port-Hamiltonian grid model, as control laws
# (docs/active/PH_SIMULATOR_SPEC.md, section 3.1; user decision of 2026-10-13: control laws now,
# physical exciters and prime movers later).
#
# The controllers are the existing RMS kernels, unchanged: IEEET1_PHTRUE, IEEEG1_PHTRUE and
# IEEEG3_PHTRUE with their limiters. Here they are signal laws. Each reads signals of its
# machine and sets one of the machine's boundary inputs:
#   exciter   Vterm (magnitude of the generator-bus voltage) -> Efd,  v_f = Efd Rfd / Lad
#   governor  omega -> turbine power P_m, shaft torque P_m / omega
# `v_f = Efd Rfd/Lad` makes `Efd = Lad i_fd` at rest, the per-unit excitation of the RMS models
# and of PowerFactory. No controller energy is included in the Hamiltonian, and a controller is
# not a component of the energy account; its states are carried beside the physical ones.
# Excitation and turbine power are external supplies to the physical model,
# `sum(P_m + v_f i_fd)` through the shaft and field ports. This says nothing about the energy a
# real exciter or turbine holds, and it is not a passivity or stability statement about the
# controlled system.
#
# State: the physical states of the `PHSystem`, then each machine's exciter states, then each
# machine's governor states. A machine without a controller keeps its fixed input.
#
# Limiters. The kernels' limiters decide by comparisons (`components/primitives.jl`), and a
# non-windup limit switches a rate to zero: the vector field jumps there. An implicit step
# cannot iterate across a jump, so the integrator takes every such decision once, at the start
# of a step, and holds it for that step (`phs_modes`, `ModeReplay`). This is an explicit
# treatment of the switching, not the implicit midpoint rule applied to the kernels as they are:
# a limiter engages or releases at a step boundary, a held branch can carry a state slightly
# past its limit within the step, and the order of accuracy at a switching is not that of the
# smooth segments. How large those effects are has to be measured by halving the step
# (`docs/active/PH_PF_EMT_COMPARISON.md`). The output kernels are not frozen: they are
# continuous, and the physical inputs they set must be the same in the vector field and in the
# energy account.
#
# Physical exciters (`exciter.jl`, PH_SIMULATOR_TODO.md package N2). A machine's exciter may be
# a physical component in place of the IEEET1 law: a `PHShuntExciter` or a `PHBrushlessExciter`
# holds its own energy (the field flux), sets the field voltage from that flux and the machine's
# speed, and takes its power from the machine's shaft, so the turbine supplies the excitation. It is then a
# component of the energy account, after the physical model's components; the supplies of the
# whole model are the turbine powers, the field voltages of machines without a physical exciter
# and the amplifiers of the physical exciters. `cs.u0.Tm` is the turbine power of a machine
# without a governor.
#
# Physical steam units (`steam.jl`, PH_SIMULATOR_TODO.md step G1): a machine's governor may be a
# `PHSteamUnit` in place of the IEEEG1 law: IEEEG1's own signal law and servo, and a steam
# turbine with its own storages (mass and internal energy) whose shaft work is the machine's
# turbine power. It is then a component of the energy account (`<name>.steam`); its supplies are
# the boiler and the reheaters' heat, less what goes to the condenser and the extractions, and the
# turbine power becomes an internal port.
#
# Physical hydro units (`hydro.jl`, step G2): a machine's IEEEG3 may be a `PHHydroUnit`: IEEEG3's
# governor rows, a penstock with the water column's kinetic energy and the ideal turbine; component
# `<name>.hydro` of the energy account, its supply the reservoir, its shaft power an internal port.

"""
Recorder that replays stored decisions in place of the computed ones: the kernel runs with its
branches held as they were when `decisions` was recorded.
"""
mutable struct ModeReplay <: ModeRecorder
    decisions::Vector{Bool}
    i::Int
end
@inline _record!(r::ModeReplay, b::Bool) = (r.i += 1; r.decisions[r.i])

"""
    PHControlled

A `PHSystem` with exciters and governors. Per machine: `exc` (the unchanged IEEET1 component as
a control law, a physical `PHShuntExciter` or `PHBrushlessExciter`, or `nothing`) and `gov` (the unchanged RMS
component as a control law, a physical `PHSteamUnit`, or `nothing`), their first state indices `xoff`, `goff` (0 if none),
their set-points `Vref`, `Pref`, the factor `efd_gain = Rfd/Lad` from `Efd` to the field voltage
and the number of identical units the governor's output is multiplied by. `u0` holds the fixed
inputs (turbine power, field voltage) used where a machine has no controller. `conv`, `coff`: the
converters' controllers (`PHVsm`), which set the converters' EMFs, and their first state indices.
"""
struct PHControlled
    s::PHSystem
    exc::Vector{Union{Nothing,IEEET1_PHTRUE,PHShuntExciter,PHBrushlessExciter}}
    gov::Vector{Union{Nothing,IEEEG1_PHTRUE,IEEEG3_PHTRUE,PHSteamUnit,PHHydroUnit}}
    xoff::Vector{Int}
    goff::Vector{Int}
    Vref::Vector{Float64}
    Pref::Vector{Float64}
    efd_gain::Vector{Float64}
    units::Vector{Int}
    u0::PHInput{Float64}
    n::Int
    conv::Vector{Union{PHVsm,PHDroop,PHVoc,PHGfl,PHEmfTrace}}   # each converter's controller (`converter.jl`), in `s.conv`'s order
    coff::Vector{Int}             # and its first state index
end

"""The grid model or the grid model with its controllers."""
const PHModel = Union{PHSystem,PHControlled}

_phys(s::PHSystem) = s
_phys(cs::PHControlled) = cs.s

"""
    phs_frame(cs::PHControlled, frame; theta0 = 0.0) -> PHControlled

The controlled model with its physical model in `frame` (`phs_frame(::PHSystem, ...)`); the controllers' states are
rotor or scalar quantities and are unchanged.
"""
phs_frame(cs::PHControlled, frame::Symbol; theta0::Real = 0.0) =
    PHControlled(phs_frame(cs.s, frame; theta0), (getfield(cs, f) for f in fieldnames(PHControlled)[2:end])...)

"""    phs_rotate_network(cs::PHControlled, x, theta): the physical model's network pairs rotated; controller states unchanged."""
phs_rotate_network(cs::PHControlled, x::AbstractVector, theta::Real) =
    vcat(phs_rotate_network(cs.s, x[1:cs.s.n], theta), x[cs.s.n+1:end])
_nstates(m::PHModel) = m.n

# the machines with a physical exciter, and the components of the energy account
_physical_exciters(cs::PHControlled) = [j for j in eachindex(cs.exc) if cs.exc[j] isa PHPhysicalExciter]
_cnames(s::PHSystem) = s.cnames
_ckind(s::PHSystem) = s.ckind
# the volumes option for one path (phs_controlled's docstring)
function _steam_volumes_for(v, path::Symbol)
    v === nothing && return nothing
    v isa Symbol && return v                # :case, :fit: every path (a :fit once fell to path C only)
    v isa NamedTuple && haskey(v, :A) && haskey(v, :C) && return v[path]
    v isa NamedTuple && (haskey(v, :A) || haskey(v, :C)) &&
        throw(ArgumentError("steam volumes per path need both A and C (each nothing, :case or a named tuple)"))
    return path === :C ? v : nothing        # path C's named tuple alone: path A keeps Vahidi's
end

_steam_units(cs::PHControlled) = [j for j in eachindex(cs.gov) if cs.gov[j] isa PHSteamUnit]
_dc_converters(cs::PHControlled) = [i for i in eachindex(cs.conv) if cs.conv[i] isa Union{PHVsm,PHVoc,PHGfl,PHDroop} && cs.conv[i].dc !== nothing]
_hydro_units(cs::PHControlled) = [j for j in eachindex(cs.gov) if cs.gov[j] isa PHHydroUnit]
_cnames(cs::PHControlled) = vcat(cs.s.cnames, [name(cs.exc[j]) * ".exciter" for j in _physical_exciters(cs)],
                                 [name(cs.gov[j]) * ".steam" for j in _steam_units(cs)],
                                 [name(cs.gov[j]) * ".hydro" for j in _hydro_units(cs)],
                                 [cs.conv[i].name * ".dc" for i in _dc_converters(cs)])
_ckind(cs::PHControlled) = vcat(cs.s.ckind, [:exciter for _ in _physical_exciters(cs)],
                                [:steam for _ in _steam_units(cs)], [:hydro for _ in _hydro_units(cs)],
                                [:converter_dc for _ in _dc_converters(cs)])
_gov_states(cs::PHControlled, x, j) = view(x, cs.goff[j]:cs.goff[j]+nstates(cs.gov[j])-1)

# the powers of machine j's physical exciter at the state `x`
function _exc_at(cs::PHControlled, x, j::Int)
    e, m, o = cs.exc[j], cs.s.machines[j], cs.xoff[j]
    return _exc_powers(e, x[o], x[o+1], x[_ip(m)] / (2 * m.w.H), _winding_currents(m, x).ifd)
end

# the turbine power of machine j: its governor's output, or the fixed input
_turbine_power(cs::PHControlled, x::AbstractVector{T}, j::Int) where {T} =
    cs.gov[j] === nothing ? T(cs.u0.Tm[j]) :
    cs.gov[j] isa PHSteamUnit ? cs.units[j] * _steam_power(cs.gov[j], _gov_states(cs, x, j)) :
    cs.gov[j] isa PHHydroUnit ? cs.units[j] * _hydro_power(cs.gov[j], _gov_states(cs, x, j)) :
    cs.units[j] * _controller_output(cs.gov[j], view(x, cs.goff[j]:cs.goff[j]+nstates(cs.gov[j])-1))

# generator-bus voltage of machine j: the bus itself (recovered if its node was reduced), or
# recovered from the merged transformer
function _gen_voltage(s::PHSystem, j::Int, f, dx)
    m = s.machines[j]
    vn = _node_voltage(s, m.node, f, dx)
    m.trf == "" && return vn
    c, r = f.mc[j], _winding_currents(m, dx)
    if s.stationary                    # the rotation e^{j (delta + theta_f)} turns at delta' + omega_b; no frame term
        dinet = -im * c.rot * (complex(r.id, r.iq) + im * (dx[_idelta(m)] + s.omega_b) * complex(c.id, c.iq))
        return vn / m.a - m.r_ext * c.inet - m.x_ext * dinet / s.omega_b
    end
    dinet = -im * c.rot * (complex(r.id, r.iq) + im * dx[_idelta(m)] * complex(c.id, c.iq))
    return vn / m.a - complex(m.r_ext, m.x_ext) * c.inet - m.x_ext * dinet / s.omega_b
end

# the output of a controller from its states (the output kernels do not read their inputs)
function _controller_output(c, xc::AbstractVector{T}) where {T}
    y = zeros(T, 1)
    outputs!(y, c, xc, zeros(T, ninputs(c)))
    return y[1]
end

"""
    phs_control_inputs(cs, x, gf; cf = nothing) -> PHInput

The inputs of the physical model at the state `x`: shaft power and field voltage per machine
(the fixed `cs.u0` values where there is no controller), with the fault conductances `gf` and
the constant-power shaft law. The shaft power is the turbine power, less what a physical
exciter takes from the shaft; a physical exciter's field voltage is its generated voltage. `cf` is the frame phasor
of the stationary frame (`PHInput.cf`), passed on unchanged; ignored in the synchronous frame.
"""
function phs_control_inputs(cs::PHControlled, x::AbstractVector{T}, gf::Vector{Float64}; cf = nothing) where {T}
    nm = length(cs.s.machines)
    Tm, vf = Vector{T}(undef, nm), Vector{T}(undef, nm)
    for j in 1:nm
        Tm[j] = _turbine_power(cs, x, j)
        e = cs.exc[j]
        if e === nothing
            vf[j] = T(cs.u0.vf[j])
        elseif e isa PHPhysicalExciter
            q = _exc_at(cs, x, j)
            vf[j] = q.vf
            Tm[j] -= q.shaft
        else
            vf[j] = cs.efd_gain[j] * _controller_output(e, view(x, cs.xoff[j]:cs.xoff[j]+nstates(e)-1))
        end
    end
    if cs.s.stationary                 # (no converters in the stationary frame: `phs_frame`)
        return PHInput{T}(Tm, vf, gf, true, Complex{T}[], cf === nothing ? Complex{T}[1] : Complex{T}.(cf))
    end
    isempty(cs.conv) && return PHInput{T}(Tm, vf, gf, true)
    f = phs_flows(cs.s, x, PHInput{T}(Tm, vf, gf, true, zeros(Complex{T}, length(cs.conv))))
    uc = [_conv_uc(cs, x, f, i) for i in eachindex(cs.conv)]
    return PHInput{T}(Tm, vf, gf, true, uc)
end

# converter i's EMF at the state `x` (flows `f` with any EMFs: they enter `w` not at all)
function _conv_uc(cs::PHControlled, x, f, i::Int)
    g = cs.conv[i]
    xc = view(x, cs.coff[i]:cs.coff[i]+nstates(g)-1)
    g isa PHEmfTrace && return _trace_at(g, xc[1])
    if _sw_on(g)                                    # the switching bridge (upgrade 3): v_dc m-bar + the averaged drops
        r = _sw_terms(g, xc, _conv_current(f, cs.s.conv[i]))
        return xc[_dcidx(g)] * r.m + r.udrop
    end
    g isa PHDroop && return g.Ts > 0 ? _bridge_voltage(g, xc) : droop_output(g, xc).uc
    g isa PHVoc && return _bridge_voltage(g, xc)
    g isa PHGfl && return gfl_bridge(g, xc, _conv_current(f, cs.s.conv[i]))
    g.Ts > 0 && return _bridge_voltage(g, xc)
    return vsm_divider(g, _vsm_emf(xc), cs.s.conv[i].alpha, _conv_w(cs.s, cs.s.conv[i], x, f)).uc
end

# converter i at the state `x`, with flows `f` there (any EMFs: they enter `w` not at all): its
# controller states, the divider's result and the eliminated terminal voltage's `w`
function _vsm_at(cs::PHControlled, x, f, i::Int)
    g, cv = cs.conv[i], cs.s.conv[i]
    xc = view(x, cs.coff[i]:cs.coff[i]+nstates(g)-1)
    w = cv.core == 0 ? _conv_w(cs.s, cv, x, f) : complex(NaN, NaN)   # no affine form with a cored transformer
    if g isa PHEmfTrace
        uc = _trace_at(g, xc[1])
        return (; uc, ip = zero(uc), on = false, w, xc)
    end
    if g isa PHDroop                               # the droop: its output from its states (held when sampled)
        o = droop_output(g, xc)
        uc = g.Ts > 0 ? _bridge_voltage(g, xc) : o.uc
        return (; uc, ip = zero(uc), on = o.boost || o.stage1 || o.stage2, w, xc)
    end
    if g isa PHGfl                                 # the GFL: the static generator's EMF from its states and the current
        uc = gfl_bridge(g, xc, _conv_current(f, cv))
        return (; uc, ip = zero(uc), on = false, w, xc)
    end
    if g isa PHVoc                                 # the VOC (sampled): the held output; the divider at v_k for the record
        uc = _bridge_voltage(g, xc)
        vk = cv.core == 0 ? cv.alpha * uc + w : _conv_vk(cs.s, cv, x, f)
        d = vsm_divider(g, complex(xc[1], xc[2]), 0.0, vk)
        return (; uc, ip = d.ip, on = d.on, w, xc)
    end
    if g.Ts > 0                                    # sampled: the held output; the divider at v_k for the record
        uc = _bridge_voltage(g, xc)
        vk = cv.core == 0 ? cv.alpha * uc + w : _conv_vk(cs.s, cv, x, f)
        d = vsm_divider(g, _vsm_emf(xc), 0.0, vk)
        return (; uc, ip = d.ip, on = d.on, w, xc)
    end
    d = vsm_divider(g, _vsm_emf(xc), cv.alpha, w)
    return (; d..., w, xc)
end

# The vector field. `rec(j, kind)` gives the recorder for machine j's exciter (kind 1) or
# governor (kind 2): `NoModes()` to decide every limiter at `x`, a `ModeLog` to record the
# decisions, a `ModeReplay` to hold recorded ones.
# `qs`: the indices of states made quasi-steady (an RMS model's algebraic rows). Their rates are zero on the model's
# constraint, so where the controllers reconstruct a terminal voltage from rates (a merged transformer, a reduced node)
# those rates are taken as zero; the rows themselves are still returned (the constraint residuals). On the constraint
# this is the same field; it keeps an interval evaluation from carrying the residuals' width into the voltages.
function _controlled_rhs!(dx, cs::PHControlled, x::AbstractVector{T}, u::PHInput, rec; qs = nothing) where {T}
    s = cs.s
    up = phs_control_inputs(cs, x, u.gf; cf = u.cf)
    phs_rhs!(dx, s, x, up)
    dq = qs === nothing ? nothing : dx[qs]
    qs === nothing || (dx[qs] .= zero(eltype(dx)))
    f = phs_flows(s, x, up)
    for (i, cv) in enumerate(s.conv)              # the converters' controllers
        r = cs.coff[i]:cs.coff[i]+nstates(cs.conv[i])-1
        cs.conv[i] isa PHEmfTrace && (dx[r[1]] = 1; continue)
        if cs.conv[i] isa PHGfl                    # the DLL's states held, the frame turning, the static generator's PI
            gfl_rates!(view(dx, r), cs.conv[i], view(x, r), _conv_current(f, cv), s.omega_b)
            cs.conv[i].dc === nothing || _dc_rows!(view(dx, r), cs.conv[i], view(x, r), _conv_current(f, cv), s.omega_b)
            continue
        end
        if cs.conv[i] isa PHVoc                    # sampled: the oscillator and the filter held, the output held in alpha-beta
            dx[r[1:2]] .= 0
            dx[r[3]], dx[r[4]] = s.omega_b * x[r[4]], -s.omega_b * x[r[3]]
            cs.conv[i].dcr && (dx[r[5:7]] .= 0)
            cs.conv[i].dc === nothing || _dc_rows!(view(dx, r), cs.conv[i], view(x, r), _conv_current(f, cv), s.omega_b)
            continue
        end
        if cs.conv[i] isa PHDroop
            g = cs.conv[i]
            if g.Ts > 0                            # sampled: held states, the output held in alpha-beta
                dx[r[1:8]] .= 0
                dx[r[9]], dx[r[10]] = s.omega_b * x[r[10]], -s.omega_b * x[r[9]]
                g.dcr && (dx[r[11:14]] .= 0)       # the DC-rejection states change at the calls only
                g.dc === nothing || _dc_rows!(view(dx, r), g, view(x, r), _conv_current(f, cv), s.omega_b)
            else
                vk = cv.alpha * up.uc[i] + _conv_w(s, cv, x, f)
                droop_rates!(view(dx, r), g, view(x, r), vk, _conv_current(f, cv), s.omega_b)
            end
            continue
        end
        if cs.conv[i].Ts > 0                       # sampled: states held, the output held in alpha-beta
            dx[r[1:4]] .= 0
            dx[r[5]], dx[r[6]] = s.omega_b * x[r[6]], -s.omega_b * x[r[5]]
            g = cs.conv[i]
            g.dc === nothing ||                    # the DC link and the battery: its branches, conductance, bridge
                _dc_rows!(view(dx, r), g, view(x, r), _conv_current(f, cv), s.omega_b)
            continue
        end
        vk = cv.alpha * up.uc[i] + _conv_w(s, cv, x, f)
        vsm_rhs!(view(dx, r), cs.conv[i], view(x, r), vk, _conv_current(f, cv))
    end
    for (j, m) in enumerate(s.machines)
        c = f.mc[j]
        if cs.exc[j] !== nothing
            e = cs.exc[j]
            r = cs.xoff[j]:cs.xoff[j]+nstates(e)-1
            vterm = abs(_gen_voltage(s, j, f, dx))
            if e isa PHPhysicalExciter
                _exc_rhs!(view(dx, r), e, view(x, r), vterm, cs.Vref[j], c.omega, c.ifd, rec(j, 1))
            else
                _step!(view(dx, r), nothing, e, view(x, r), T[vterm, cs.Vref[j], 0, m.w.Lad * c.ifd],
                       params(e), rec(j, 1))
            end
        end
        if cs.gov[j] isa PHSteamUnit
            g = cs.gov[j]
            r = cs.goff[j]:cs.goff[j]+nstates(g)-1
            _steam_rhs!(view(dx, r), g, view(x, r), c.omega, cs.Pref[j], rec(j, 2))
        elseif cs.gov[j] isa PHHydroUnit
            g = cs.gov[j]
            r = cs.goff[j]:cs.goff[j]+nstates(g)-1
            _hydro_rhs!(view(dx, r), g, view(x, r), c.omega, cs.Pref[j], rec(j, 2))
        elseif cs.gov[j] !== nothing
            g = cs.gov[j]
            r = cs.goff[j]:cs.goff[j]+nstates(g)-1
            _step!(view(dx, r), nothing, g, view(x, r), T[c.omega, cs.Pref[j], 0], params(g), rec(j, 2))
        end
    end
    qs === nothing || (dx[qs] = dq)
    return dx
end

phs_rhs!(dx, cs::PHControlled, x::AbstractVector, u::PHInput; qs = nothing) =
    _controlled_rhs!(dx, cs, x, u, (j, k) -> NoModes(); qs)

phs_rhs(cs::PHControlled, x::AbstractVector{T}, u::PHInput; qs = nothing) where {T} =
    phs_rhs!(Vector{T}(undef, cs.n), cs, x, u; qs)

"""
    phs_modes(cs, x, u) -> Matrix{Vector{Bool}}

Every limiter decision of the controllers' step kernels at the state `x`: entry `[j, 1]` for
machine `j`'s exciter, `[j, 2]` for its governor (empty where there is none), in the order the
kernel takes them.
"""
function phs_modes(cs::PHControlled, x::AbstractVector{Float64}, u::PHInput)
    logs = [ModeLog() for _ in eachindex(cs.exc), _ in 1:2]
    _controlled_rhs!(Vector{Float64}(undef, cs.n), cs, x, u, (j, k) -> logs[j, k])
    return map(l -> l.decisions, logs)
end

# the vector field with the limiter decisions held as recorded in `modes`; every recorded
# decision must be consumed, in the kernel's own order
function _frozen_rhs!(dx, cs::PHControlled, x, u::PHInput, modes)
    replay = [ModeReplay(modes[j, k], 0) for j in axes(modes, 1), k in 1:2]
    _controlled_rhs!(dx, cs, x, u, (j, k) -> replay[j, k])
    all(r -> r.i == length(r.decisions), replay) ||
        error("the controllers took a different number of limiter decisions than were recorded")
    return dx
end
_frozen_rhs!(dx, s::PHSystem, x, u::PHInput, modes) = phs_rhs!(dx, s, x, u)
_freeze(cs::PHControlled, x, u::PHInput) = phs_modes(cs, x, u)
_freeze(s::PHSystem, x, u::PHInput) = nothing

phs_network_mask(cs::PHControlled, L) = phs_network_mask(cs.s, L)

"""
    phs_ledger(cs, x, u; dx = nothing)

The energy account at `x`: the physical model's, with the inputs the controllers set there,
and after its components one per physical exciter (`<name>.exciter`): the stored energy of its
field, the loss in its field circuit, the net power it receives (from the shaft and, for a DC
exciter, from its amplifier, less what it delivers to the machine's field) and the rate of its
energy along `dx` (the vector field if `dx` is not given). `supply` is then the turbine powers,
the field powers of machines without a physical exciter and the DC exciters' amplifier powers.
"""
function phs_ledger(cs::PHControlled, x::AbstractVector{Float64}, u::PHInput; dx = nothing)
    L = phs_ledger(cs.s, view(x, 1:cs.s.n), phs_control_inputs(cs, x, u.gf; cf = u.cf);
                   dx = dx === nothing ? nothing : view(dx, 1:cs.s.n))
    js = _physical_exciters(cs)
    ks = _steam_units(cs)
    hs = _hydro_units(cs)
    ds = _dc_converters(cs)
    isempty(js) && isempty(ks) && isempty(hs) && isempty(ds) && return L
    energy, loss, port, rate, tport = copy(L.energy), copy(L.loss), copy(L.port), copy(L.rate), copy(L.tport)
    supply = L.supply
    for j in js
        e, o = cs.exc[j], cs.xoff[j]
        q = _exc_at(cs, x, j)
        # the field flux's rate: along `dx`, or the exciter's own row (which holds no decision)
        dphi = dx === nothing ? q.flux_rate : dx[o]
        push!(energy, _exc_energy(e, x[o]))
        push!(loss, q.loss)
        push!(port, q.port)
        push!(rate, q.slope * dphi)
        push!(tport, q.tport)
        supply += q.port
    end
    for j in ks
        # one component per steam unit, for all the identical units of the machine: stored energy
        # sum U / S_b, no loss term (its irreversibility is entropy, not energy), the net port
        # (boiler and heat in, condenser and extractions out, shaft out)
        g, o, nu = cs.gov[j], cs.goff[j], cs.units[j]
        xs = _gov_states(cs, x, j)
        q = _steam_powers(g, xs)
        push!(energy, nu * q.energy)
        push!(loss, 0.0)
        push!(port, nu * q.port)
        # the energy's rate: along dx (the U rows), or the unit's own rows
        push!(rate, dx === nothing ? nu * q.rate :
                    nu * sum(dx[o+1+2k] for k in eachindex(g.V)) / STEAM_SB)
        push!(tport, nu * q.tport)
        supply += nu * q.port
    end
    for j in hs
        # one component per hydro unit: the water column's kinetic energy, no loss (the ideal turbine),
        # the net port (the reservoir's supply less the shaft power)
        g, o, nu = cs.gov[j], cs.goff[j], cs.units[j]
        q = _hydro_powers(g, _gov_states(cs, x, j))
        push!(energy, nu * q.energy)
        push!(loss, 0.0)
        push!(port, nu * q.port)
        push!(rate, dx === nothing ? nu * q.rate : nu * g.GMAX * g.Tw * x[o+3] * dx[o+3])
        push!(tport, nu * q.tport)
        supply += nu * q.port
    end
    if !isempty(ds)
        # one component per converter's physical DC side: the capacitor's energy, the battery's resistance and the
        # conductance as losses, the net port (the battery's source less the bridge power, which the network's
        # account counts as its supply)
        fl = phs_flows(cs.s, x, phs_control_inputs(cs, x, u.gf; cf = u.cf))
        for i in ds
            g, o, cv = cs.conv[i], cs.coff[i], cs.s.conv[i]
            xc = view(x, o:o+nstates(g)-1)
            q = converter_dc_powers(g, xc, _conv_current(fl, cv), cs.s.omega_b)
            push!(energy, q.energy)
            push!(loss, q.loss)
            push!(port, q.port)
            push!(rate, dx === nothing ? q.port - q.loss : _dc_energy_rate(g, xc, view(dx, o:o+nstates(g)-1), cs.s.omega_b))
            push!(tport, q.tport)
            supply += q.port
        end
    end
    return (; names = _cnames(cs), kind = _ckind(cs), energy, loss, port, rate, tport, supply,
            interconnection = L.interconnection, H = sum(energy), dH = sum(rate), losses = sum(loss))
end

phs_energy(cs::PHControlled, x::AbstractVector) =
    phs_energy(cs.s, view(x, 1:cs.s.n)) +
    sum(_exc_energy(cs.exc[j], x[cs.xoff[j]]) for j in _physical_exciters(cs); init = zero(eltype(x))) +
    sum(cs.units[j] * sum(x[cs.goff[j]+1+2k] for k in eachindex(cs.gov[j].V)) / STEAM_SB
        for j in _steam_units(cs); init = zero(eltype(x))) +
    sum(cs.units[j] * _hydro_powers(cs.gov[j], _gov_states(cs, x, j)).energy for j in _hydro_units(cs);
        init = zero(eltype(x))) +
    sum(_dc_energy(cs.conv[i], view(x, cs.coff[i]:cs.coff[i]+nstates(cs.conv[i])-1), cs.s.omega_b) for i in _dc_converters(cs);
        init = zero(eltype(x)))

function phs_scales(cs::PHControlled, x::AbstractVector{Float64})
    sc = ones(cs.n)
    sc[1:cs.s.n] .= phs_scales(cs.s, x[1:cs.s.n])
    for j in eachindex(cs.exc), (c, off) in ((cs.exc[j], cs.xoff[j]), (cs.gov[j], cs.goff[j]))
        c === nothing && continue
        r = off:off+nstates(c)-1
        sc[r] .= max(1.0, maximum(abs, view(x, r)))
        # a physical exciter's field flux is a small number: scale it by itself
        c isa PHShuntExciter && x[off] != 0 && (sc[off] = abs(x[off]))
        # a steam unit's masses and energies: each by itself (kilograms to gigajoules)
        if c isa PHSteamUnit
            sc[off] = sc[off+1] = max(1.0, abs(x[off]), abs(x[off+1]))
            for i in off+2:off+nstates(c)-1
                sc[i] = abs(x[i]) > 0 ? abs(x[i]) : 1.0
            end
        end
    end
    for (g, o) in zip(cs.conv, cs.coff)
        r = o:o+nstates(g)-1
        sc[r] .= max.(1.0, abs.(view(x, r)))
    end
    return sc
end

function _bus_voltages(cs::PHControlled, x, u::PHInput)
    return _bus_voltages(cs.s, x[1:cs.s.n], phs_control_inputs(cs, x, u.gf; cf = u.cf))
end

"""
    phs_control_signals(cs, x) -> (; Efd, Pm, vf, Pexc)

Per machine at the state `x`: the exciter output `Efd` (per unit, `Lad i_fd` at rest), the
turbine power `Pm` (system base, all units), the field voltage `vf` and the power a physical
exciter takes from the shaft, `Pexc` (zero otherwise). `Efd` is `vf Lad/Rfd` for a machine
without an exciter.
"""
function phs_control_signals(cs::PHControlled, x::AbstractVector{Float64})
    up = phs_control_inputs(cs, x, cs.u0.gf)
    Efd = [up.vf[j] * m.w.Lad / m.w.Rfd for (j, m) in enumerate(cs.s.machines)]
    Pexc = [cs.exc[j] isa PHPhysicalExciter ? _exc_at(cs, x, j).shaft : 0.0 for j in eachindex(cs.exc)]
    return (; Efd, Pm = up.Tm .+ Pexc, vf = up.vf, Pexc)
end

"""    phs_state_names(cs) -> the physical state names, then `<controller>.<state>`."""
function phs_state_names(cs::PHControlled)
    names = Vector{String}(undef, cs.n)
    names[1:cs.s.n] .= phs_state_names(cs.s)
    for j in eachindex(cs.exc), (c, off) in ((cs.exc[j], cs.xoff[j]), (cs.gov[j], cs.goff[j]))
        c === nothing && continue
        names[off:off+nstates(c)-1] .= name(c) .* "." .* state_names(c)
    end
    for (g, o) in zip(cs.conv, cs.coff)
        names[o:o+nstates(g)-1] .= g.name .* "." .* state_names(g)
    end
    return names
end

"""
    phs_fixed_units(cs::PHControlled) -> (; steam, hydro_Tw)

The turbine hardware of a built model, for `phs_controlled(...; fixed_units)`: every physical steam
unit by its IEEEG1's name, and the water time of its physical hydro unit (the entry omitted if there is
none; an error if there is more than one).
"""
function phs_fixed_units(cs::PHControlled)
    steam = Dict{String,PHSteamUnit}(g.name => g for g in cs.gov if g isa PHSteamUnit)
    hs = [g for g in cs.gov if g isa PHHydroUnit]
    length(hs) <= 1 || throw(ArgumentError("phs_fixed_units: more than one hydro unit; give their water times per unit"))
    return isempty(hs) ? (; steam) : (; steam, hydro_Tw = hs[1].Tw)
end


"""
    phs_controlled(eq::EquilibriumResult; loads = :parallel, circuits = nothing,
                   dc_exciter = nothing, brushless = nothing) -> (cs, x, u, report)

The model of `phs_equilibrium` (with the constant-power shaft law) and the case's exciters and
governors attached as control laws, at its equilibrium. Each controller is the case's own
component, found through its `syn` parameter and initialised by `init_from_targets` at its
machine's steady state (`Efd = Lad i_fd`, turbine power, generator-bus voltage); that also gives
its set-point. Identical units on one bus share one exciter and one governor, the governor's
output multiplied by the number of units; there must be exactly one controller per unit and
their parameters must be equal (the units are already required to be identical, they see the
same signals, and `Efd = Lad i_fd` does not depend on how many units are combined). `u`
carries the fault conductances for `phs_simulate`. `report` adds to `phs_equilibrium`'s the
residual of the whole vector field, each machine's controller names and the names of the
physical exciters.

With `dc_exciter = (; n, field_ratio)` every machine whose IEEET1 has a negative `KE` (a
self-excited shunt field, by the 1968 committee report) gets a physical `PHShuntExciter` in
place of the control law. Both numbers are ones no source gives and have no default: `n`, the
base of the regulator output over the base of the exciter voltage (a number, or a dictionary by
IEEET1 name), and `field_ratio`, the shunt field circuit's resistance over the machine's field
resistance. The operating point of the physical model is unchanged: the exciter starts at the
flux that gives the same field voltage, and the turbine power is raised by what the exciter
takes from the shaft.

With `brushless = (; KC, KD, sat, field_share)` every machine whose IEEET1 has a positive `KE`
gets a physical `PHBrushlessExciter` (an alternator with a rotating rectifier, its regulator
fed from the shaft) in place of the control law; the keywords are those of its constructor and
none has a default. Its operating point is in the rectifier's first mode, which needs
`KC / (1 + KC/sqrt(3)) <= sqrt(3)/4`.

With `steam = (; paths, r_v, rh_dT, tau_n, lowflow, cold_reheat, volumes, tables)` every machine
whose IEEEG1 is named in `paths` (a dictionary from the IEEEG1's name to `:A` or `:C`) gets a
physical `PHSteamUnit` in place of the IEEEG1 law (docs/active/PH_STEAM_UNIT_MODEL.md): IEEEG1's
signal law and servo, and the published steam unit of that path scaled to the IEEEG1's `PMAX`. The
other keywords are the constructor's, none with a default (`cold_reheat` is needed if any unit is on
path A, `volumes` if any is on path C; `tables` defaults to `steam_tables()`). `volumes` is
`:case` (every vessel from its IEEEG1's time constants, record 16.6), `:fit` (fitted to the case's
IEEEG1 turbine, record 16.7), or `(; A = ..., C = ...)` per
path (each `nothing`, `:case` or the constructor's named tuple), or, as before, path C's named tuple
alone (path A then keeps Vahidi's volumes). `valve` (`:characteristic` by default, or `:opening`) is
the constructor's. The unit is initialised at the machine's turbine power (per identical unit):
storages and reheater wall temperatures in steady state; the servo state is the flow demand (the
opening with `valve = :opening`), `Pref` from it.

With `hydro = (; Tw)` every machine with an IEEEG3 gets a physical `PHHydroUnit` in its place
(docs/active/PH_HYDRO_UNIT_MODEL.md): IEEEG3's governor rows, a penstock with the water starting time `Tw`
(s, at full gate) and the ideal turbine; `Tw = :match` takes the value with which the unit's
linearisation at its operating point equals the case's IEEEG3 (`hydro_Tw_match`).

With `fixed_units = (; steam = Dict(IEEEG1 name => PHSteamUnit), hydro_Tw = value)` (either entry may be
left out; `phs_fixed_units` collects both from a built model) the turbines' hardware is taken as given
instead of being built at this operating point: each named steam unit is a copy of the given unit
(its volumes and valve characteristic kept, only its operating state solved at the new power:
`steam_initialise(...; rebuild = false)`), and the hydro unit uses the given water time. Without it,
`volumes = :fit`, the valve characteristic and `Tw = :match` are all set at this operating point, so a
model built at another operating point is a different plant.
"""
function phs_controlled(eq::EquilibriumResult; loads::Symbol = :parallel, circuits = nothing,
                        dc_exciter = nothing, brushless = nothing, steam = nothing, hydro = nothing,
                        emf_trace = nothing, converter_Ts::Real = 0.0, converter_dc = nothing,
                        converter_umax::Real = Inf, cores = nothing, droop_dc_reject::Bool = false,
                        droop_dcr_on::Symbol = :both, voc_dc_reject::Bool = false, gfl_advance::Real = 0.0,
                        fixed_units = nothing, lines::Symbol = :lumped, nsec::Int = 4, composite = (;), dq0 = (;))
    s, xp, u0, rep = phs_equilibrium(eq; loads, circuits, const_power = true, cores, lines, nsec, composite, dq0)
    (isempty(s.cores) || converter_Ts > 0) ||
        throw(ArgumentError("phs_controlled: transformer cores need the sampled converter controllers (converter_Ts > 0): " *
                            "the continuous controllers read the terminal voltage inside their rows (record section 21, T1 row 4)"))
    comps = eq.sys.comps
    nm = length(s.machines)
    f = phs_flows(s, xp, u0)
    dxp = phs_rhs(s, xp, u0)
    exc = Vector{Union{Nothing,IEEET1_PHTRUE,PHShuntExciter,PHBrushlessExciter}}(nothing, nm)
    gov = Vector{Union{Nothing,IEEEG1_PHTRUE,IEEEG3_PHTRUE,PHSteamUnit,PHHydroUnit}}(nothing, nm)
    Pturb = copy(u0.Tm)                       # turbine power: the shaft power plus the exciter's
    xoff, goff = zeros(Int, nm), zeros(Int, nm)
    Vref, Pref = zeros(nm), zeros(nm)
    x = collect(xp)
    names = Vector{NamedTuple}(undef, nm)
    # the numeric parameters that define a controller, without its name and its machine's
    numeric(c) = Dict(k => param_value(v) for (k, v) in param_dict(c) if !(k in ("idx", "name", "syn")))
    function of_units(T, units)
        found = [c for c in comps if c isa T && get(param_dict(c), "syn", nothing) in units]
        isempty(found) && return nothing
        # exactly one controller on each unit
        sort!([String(param_dict(c)["syn"]) for c in found]) == sort(units) ||
            throw(ArgumentError("phs_controlled: the controllers $(name.(found)) do not pair one to one " *
                                "with the units $units"))
        all(c -> typeof(c) == typeof(found[1]) && numeric(c) == numeric(found[1]), found) ||
            throw(ArgumentError("phs_controlled: the units $units have different controllers"))
        return found[1]
    end
    for (j, m) in enumerate(s.machines)
        units = String.(split(m.name, "+"))
        c = f.mc[j]
        efd0, vt0 = m.w.Lad * c.ifd, abs(_gen_voltage(s, j, f, dxp))
        e = of_units(IEEET1_PHTRUE, units)
        if e !== nothing && dc_exciter !== nothing && e.p.KE < 0
            n = dc_exciter.n isa Real ? dc_exciter.n : dc_exciter.n[name(e)]
            pe = PHShuntExciter(e, m.w.Rfd, m.w.Lad; n, rF = dc_exciter.field_ratio * m.w.Rfd)
            # at rest and at rated speed: Efd = x_E, and the regulator output the row needs
            vr = (pe.KE + _exc_se(pe, efd0)) * efd0
            pe.VRMIN <= vr <= pe.VRMAX ||
                throw(ArgumentError("phs_controlled: $(name(e)) needs a regulator output of $vr at " *
                                    "the operating point, outside its limits"))
            exc[j], Vref[j] = pe, vt0 + vr / pe.KA
            xoff[j] = length(x) + 1
            append!(x, [efd0 / pe.c, vr, efd0])
            Pturb[j] += _exc_powers(pe, efd0 / pe.c, vr, c.omega, c.ifd).shaft
        elseif e !== nothing && brushless !== nothing && e.p.KE > 0
            pe = PHBrushlessExciter(e, m.w.Rfd, m.w.Lad; brushless...)
            # in steady state at rated speed EFD = IFD (= efd0); the rectifier is in its first
            # mode there, EFD = x_E - KC IFD / sqrt(3)
            x0 = efd0 * (1 + pe.KC / sqrt(3))
            _bx_in(pe, x0, efd0) <= sqrt(3) / 4 ||
                throw(ArgumentError("phs_controlled: $(name(e)) would not rest in the rectifier's first mode"))
            vr = _bx_vfe(pe, x0, efd0)
            pe.VRMIN <= vr <= pe.VRMAX ||
                throw(ArgumentError("phs_controlled: $(name(e)) needs a regulator output of $vr at " *
                                    "the operating point, outside its limits"))
            exc[j], Vref[j] = pe, vt0 + vr / pe.KA
            xoff[j] = length(x) + 1
            append!(x, [x0, vr, vr])
            Pturb[j] += _exc_powers(pe, x0, vr, c.omega, c.ifd).shaft
        elseif e !== nothing
            xe, pe = init_from_targets(e, MachineTargets(efd0, u0.Tm[j] / m.units, vt0, efd0, 0.0, 0.0, 0.0, 0.0))
            exc[j], Vref[j] = with_params(e, pe), pe["Vref"]
            xoff[j] = length(x) + 1
            append!(x, xe)
        end
        t = MachineTargets(efd0, Pturb[j] / m.units, vt0, efd0, 0.0, 0.0, 0.0, 0.0)
        g = of_units(Union{IEEEG1_PHTRUE,IEEEG3_PHTRUE}, units)
        if g !== nothing && steam !== nothing && g isa IEEEG1_PHTRUE && haskey(steam.paths, name(g))
            fixed = fixed_units !== nothing && haskey(fixed_units, :steam)
            fixed && !haskey(fixed_units.steam, name(g)) &&
                throw(ArgumentError("phs_controlled: fixed_units.steam has no unit for $(name(g)); a fixed plant needs every " *
                                    "steam unit (a missing one would be rebuilt at this operating point)"))
            ps = fixed ? deepcopy(fixed_units.steam[name(g)]) :
                 PHSteamUnit(g; path = steam.paths[name(g)], r_v = steam.r_v, rh_dT = steam.rh_dT,
                             tau_n = steam.tau_n, lowflow = steam.lowflow,
                             cold_reheat = get(steam, :cold_reheat, nothing),
                             volumes = _steam_volumes_for(get(steam, :volumes, nothing), steam.paths[name(g)]),
                             valve = get(steam, :valve, :characteristic),
                             tables = get(steam, :tables, steam_tables()))
            xg, pref = steam_initialise(ps, Pturb[j] / m.units; rebuild = !fixed)
            gov[j], Pref[j] = ps, pref
            goff[j] = length(x) + 1
            append!(x, xg)
        elseif g !== nothing && hydro !== nothing && g isa IEEEG3_PHTRUE
            Pm_j = Pturb[j] / m.units
            Tw = fixed_units !== nothing && get(fixed_units, :hydro_Tw, nothing) !== nothing ? Float64(fixed_units.hydro_Tw) :
                 hydro.Tw === :match ? hydro_Tw_match(g, Pm_j) : Float64(hydro.Tw)
            ph = PHHydroUnit(g; Tw, Te = get(hydro, :Te, 0.0), N = get(hydro, :N, 0), fp = get(hydro, :fp, 0.0))
            xg, pref = hydro_initialise(ph, Pm_j)
            gov[j], Pref[j] = ph, pref
            goff[j] = length(x) + 1
            append!(x, xg)
        elseif g !== nothing
            xg, pg = init_from_targets(g, t)
            gov[j], Pref[j] = with_params(g, pg), pg["Pref"]
            goff[j] = length(x) + 1
            append!(x, xg)
        end
        names[j] = (; machine = m.name, exciter = e === nothing ? "" : name(e),
                    governor = g === nothing ? "" : name(g))
    end
    # the converters' controllers at the equilibrium (record, section 3): u_c = E there, the
    # set-points from the terminal voltage and the converter current, speeds 1
    conv, coff, conv_init = Union{PHVsm,PHDroop,PHVoc,PHGfl,PHEmfTrace}[], Int[], NamedTuple[]
    for (i, cv) in enumerate(s.conv)
        E0 = u0.uc[i]
        if emf_trace !== nothing && haskey(emf_trace, cv.bus)
            # a diagnostic: the converter's EMF prescribed (`PHEmfTrace`), its clock from 0
            tr = emf_trace[cv.bus]
            push!(conv, PHEmfTrace(cv.name, collect(Float64, tr.t), collect(ComplexF64, tr.uc)))
            push!(coff, length(x) + 1)
            push!(x, 0.0)
            push!(conv_init, (; name = cv.name, bus = cv.bus, trace_vs_equilibrium = abs(tr.uc[1] - E0)))
            continue
        end
        vk, ic = _conv_terminal(s, cv, xp, f, dxp), _conv_current(f, cv)
        if cv.comp isa GFL_PHTRUE
            isinf(converter_umax) || throw(ArgumentError("phs_controlled: the output limit is not implemented for the GFL"))
            # the physical DC side (record section 43): the battery holding the bridge power Re(E0 conj(ic)) at the operating point
            dc = converter_dc === nothing ? nothing : _converter_dc(cv, E0, ic, s.omega_b; converter_dc...)
            # the DLL's Model_Initialize at the terminal voltage, the static generator's integrators holding u_c = E0 (record
            # section 38); the set-points are the dialog's (the case's p_ref, q_ref), not captured from the equilibrium
            g = PHGfl(cv; Ts = converter_Ts, advance = gfl_advance, dc)
            push!(conv, g)
            push!(coff, length(x) + 1)
            append!(x, gfl_initial(g, vk, ic, E0))
            xc = view(x, coff[end]:coff[end]+nstates(g)-1)
            push!(conv_init, (; name = cv.name, bus = cv.bus, merged = cv.merged, alpha = cv.alpha, p_ref = g.p_ref,
                              q_ref = g.q_ref, i = abs(ic), vk = abs(vk), emf_vs_terminal = abs(E0 - (vk + im * cv.Lc * ic)),
                              emf = abs(gfl_emf(g, xc, ic) - E0),
                              setpoint_vs_power = abs(vk * conj(ic) - complex(g.p_ref, g.q_ref)),
                              reference_vs_current = abs(complex(xc[13], xc[14]) - ic * cis(-xc[1])),
                              bridge = abs(gfl_bridge(g, xc, ic) - E0), m0 = dc === nothing ? NaN : abs(E0) / dc.v_dc0))
            continue
        end
        if cv.comp isa GFM_VOC_PHTRUE
            isinf(converter_umax) ||
                throw(ArgumentError("phs_controlled: the output limit is not implemented for the VOC"))
            dc = converter_dc === nothing ? nothing : _converter_dc(cv, E0, ic, s.omega_b; converter_dc...)
            # the DLL's Model_Initialize (record section 28): v0 = u0 + (r_vi + j (Zs + x_vi)) i0 = E0 + (r_vi + j x_vi) i0,
            # the set-points from the measured power, V_nom = |v0| when the case takes it from the load flow
            pv, nu = cv.comp.p, cv.units
            S0 = vk * conj(ic)
            v0 = E0 + complex(pv.r_vi, pv.x_vi) / nu * ic
            abs(v0) > 1e-6 || throw(ArgumentError("phs_controlled: VOC converter $(cv.name) has |v0| = $(abs(v0))"))
            vnom = _p(cv.comp.params, "vnom_from_lf", cv.name) > 0.5 ? abs(v0) : pv.V_nom
            g = PHVoc(cv; p_set = real(S0), q_set = imag(S0), V_nom = vnom, Ts = converter_Ts, dc_reject = voc_dc_reject, dc)
            push!(conv, g)
            push!(coff, length(x) + 1)
            h = (dc !== nothing && dc.mode === :held) ? E0 / dc.v_dc0 : E0     # the held output, or the held modulation
            append!(x, [real(v0), imag(v0), real(h), imag(h)])
            if g.dcr                               # the filter at rest: V = Z / (1 - lam), so Y = Z; the cached |v0|
                V0 = ic * conj(v0) / abs(v0) / (1 - g.dlam)
                append!(x, [real(V0), imag(V0), abs(v0)])
            end
            dc === nothing || append!(x, _dc_initial(dc))
            xc = view(x, coff[end]:coff[end]+nstates(g)-1)
            d = vsm_divider(g, v0, 0.0, vk)
            dv = voc_rates(g, xc, vk, ic).dv
            (!d.on && abs(d.uc - E0) <= 1e-9 * abs(E0)) ||
                throw(ArgumentError("phs_controlled: VOC converter $(cv.name) does not rest with its virtual impedance off"))
            push!(conv_init, (; name = cv.name, bus = cv.bus, merged = cv.merged, alpha = cv.alpha, p_set = g.p_set,
                              q_set = g.q_set, V_nom = g.V_nom, i = abs(ic), vk = abs(vk),
                              emf_vs_terminal = abs(E0 - (vk + im * cv.Lc * ic)), divider = abs(d.uc - E0), engaged = false,
                              rest_rate = abs(dv)))
            continue
        end
        if cv.comp isa GFM_DROOP_PHTRUE
            isinf(converter_umax) || throw(ArgumentError("phs_controlled: the output limit is not implemented for droop"))
            # the physical DC side (record section 45): the battery holding the bridge power at the operating point
            dc = converter_dc === nothing ? nothing : _converter_dc(cv, E0, ic, s.omega_b; converter_dc...)
            S0, th0 = vk * conj(ic), angle(E0)
            g = PHDroop(cv; p_set = real(S0), q_set = imag(S0), u_set = abs(E0), Ts = converter_Ts, dc_reject = droop_dc_reject,
                        dcr_on = droop_dcr_on, dc)
            push!(conv, g)
            push!(coff, length(x) + 1)
            vm0, im0 = vk * cis(-th0), ic * cis(-th0)
            append!(x, [th0, real(S0), imag(S0), abs(E0), real(vm0), imag(vm0), real(im0), imag(im0)])
            if g.Ts > 0                            # the held output: the command, or (:held) the modulation at the DC voltage
                h = (dc !== nothing && dc.mode === :held) ? E0 / dc.v_dc0 : E0
                append!(x, [real(h), imag(h)])
            end
            if g.dcr                               # the filter at rest: V = Z / (1 - lam), so Y = Z
                Vv0, Vi0 = vm0 / (1 - g.dlam), im0 / (1 - g.dlam)
                append!(x, [real(Vv0), imag(Vv0), real(Vi0), imag(Vi0)])
            end
            dc === nothing || append!(x, _dc_initial(dc))
            d = droop_output(g, view(x, coff[end]:coff[end]+nstates(g)-1))
            (!d.boost && !d.stage1 && !d.stage2 && abs(d.uc - E0) <= 1e-9 * abs(E0)) ||
                throw(ArgumentError("phs_controlled: droop converter $(cv.name) does not rest with its limiters off"))
            push!(conv_init, (; name = cv.name, bus = cv.bus, merged = cv.merged, alpha = cv.alpha, p_set = g.p_set,
                              q_set = g.q_set, u_set = g.u_set, i = abs(ic), vk = abs(vk),
                              emf_vs_terminal = abs(E0 - (vk + im * cv.Lc * ic)), divider = abs(d.uc - E0), engaged = false,
                              bridge = g.Ts > 0 ? abs(_bridge_voltage(g, view(x, coff[end]:coff[end]+nstates(g)-1)) - E0) : 0.0,
                              m0 = dc === nothing ? NaN : abs(E0) / dc.v_dc0))
            continue
        end
        dc = converter_dc === nothing ? nothing : _converter_dc(cv, E0, ic, s.omega_b; converter_dc...)
        g = PHVsm(cv; p_set = real(vk * conj(ic)), u_set = abs(E0), Ts = converter_Ts, dc, u_max = converter_umax)
        abs(E0) <= converter_umax ||
            throw(ArgumentError("phs_controlled: converter $(cv.name) rests above its output-voltage limit"))
        push!(conv, g)
        push!(coff, length(x) + 1)
        append!(x, [angle(E0), 1.0, 1.0, abs(E0)])
        if g.Ts > 0                                # the held output: the command, or the modulation; the DC voltage
            h = (dc !== nothing && dc.mode === :held) ? E0 / dc.v_dc0 : E0
            append!(x, [real(h), imag(h)])
            dc === nothing || append!(x, _dc_initial(dc))
        end
        d = cv.core == 0 ? vsm_divider(g, E0, cv.alpha, _conv_w(s, cv, xp, f)) : vsm_divider(g, E0, 0.0, vk)
        # the equilibrium is the RMS case's with the divider off (u_c = E); an operating point with the
        # virtual impedance engaged would need its own construction (consultation C2, finding 6)
        (!d.on && abs(d.uc - E0) <= 1e-9 * abs(E0)) ||
            throw(ArgumentError("phs_controlled: converter $(cv.name) rests with its virtual impedance engaged " *
                                "(|i_p| = $(abs(d.ip)), i_lim = $(g.i_lim)); not supported"))
        push!(conv_init, (; name = cv.name, bus = cv.bus, merged = cv.merged, alpha = cv.alpha,
                          p_set = g.p_set, u_set = g.u_set, i = abs(ic), vk = abs(vk),
                          emf_vs_terminal = abs(E0 - (vk + im * cv.Lc * ic)), divider = abs(d.uc - E0),
                          engaged = d.on))
    end
    cs = PHControlled(s, exc, gov, xoff, goff, Vref, Pref, [m.w.Rfd / m.w.Lad for m in s.machines],
                      [m.units for m in s.machines],
                      PHInput{Float64}(Pturb, copy(u0.vf), u0.gf, true, copy(u0.uc)), length(x), conv, coff)
    up = phs_control_inputs(cs, x, u0.gf)
    report = (; rep..., residual_controlled = phs_rest_residual(cs, phs_rhs(cs, x, u0)),
              input_change = max(maximum(abs, up.Tm .- u0.Tm), maximum(abs.(up.vf .- u0.vf) ./ u0.vf)),
              controllers = names, converters = conv_init, physical_exciters = [name(cs.exc[j]) for j in _physical_exciters(cs)],
              steam_units = [(name = name(cs.gov[j]), path = cs.gov[j].path, n = cs.gov[j].n)
                             for j in _steam_units(cs)],
              hydro_units = [(name = name(cs.gov[j]), Tw = cs.gov[j].Tw, Tw_case = cs.gov[j].Tw_case,
                              gate = cs.gov[j].GMAX) for j in _hydro_units(cs)])
    return cs, x, u0, report
end

# the converters' sampling period (all sampled converters share it; 0 if none) and their call at `x`
function _sample_period(cs::PHControlled)
    T = unique([g.Ts for g in cs.conv if g isa Union{PHVsm,PHDroop,PHVoc,PHGfl} && g.Ts > 0])
    Tr = _sample_period(cs.s)                         # the motors' relays (review 76): on the same grid
    Tr > 0 && push!(T, Tr)
    T = unique(T)
    length(T) <= 1 || throw(ArgumentError("converters and relays with different sampling periods $T"))
    return isempty(T) ? 0.0 : T[1]
end

# switching bridges (upgrade 3): start the PWM at a run's first instant t0 (a carrier peak; the sampling grid is the
# controllers' Ts = the half carrier period), each step's window, back to the averaged bridge after the run
function _bridge_start!(cs::PHControlled, x, u::PHInput, t0)
    any(_has_sw, cs.conv) || return nothing
    up = phs_control_inputs(cs, x, u.gf; cf = u.cf)
    f = phs_flows(cs.s, x, up)
    for (i, cv) in enumerate(cs.s.conv)
        g = cs.conv[i]
        _has_sw(g) || continue
        br = g.dc.sw
        abs(g.Ts - br.H) <= 1e-12 * br.H ||
            throw(ArgumentError("switching bridge of $(cv.name): the controller's Ts = $(g.Ts) s must be the half carrier period $(br.H) s"))
        abs(t0 / br.H - round(t0 / br.H)) < 1e-9 || throw(ArgumentError("switching bridge: the run must start on the carrier grid"))
        xc = view(x, cs.coff[i]:cs.coff[i]+nstates(g)-1)
        m = _modulation(g, xc, _conv_current(f, cv))
        bridge_start!(br, cs.s.stationary ? m : m * cis(cs.s.omega_b * t0 + cs.s.theta0), t0;
                      stationary = cs.s.stationary, theta0 = cs.s.theta0)
    end
    return nothing
end
_bridge_window!(cs::PHControlled, a, b) = foreach(g -> _has_sw(g) && bridge_window!(g.dc.sw, a, b), cs.conv)
_bridge_stop!(cs::PHControlled) = foreach(g -> _has_sw(g) && (g.dc.sw.active = false), cs.conv)
# each Newton iteration: the converter currents' secant over the step (x0 to the iterate x1), for the pieces' currents
function _bridge_slope!(cs::PHControlled, x0, x1, u::PHInput, h)
    any(_sw_on, cs.conv) || return nothing
    f0 = phs_flows(cs.s, x0, phs_control_inputs(cs, x0, u.gf; cf = u.cf))
    f1 = phs_flows(cs.s, x1, phs_control_inputs(cs, x1, u.gf; cf = u.cf))
    for (i, cv) in enumerate(cs.s.conv)
        _sw_on(cs.conv[i]) || continue
        cs.conv[i].dc.sw.islope = (_conv_current(f1, cv) - _conv_current(f0, cv)) / h
    end
    return nothing
end
# after each converged step: the applied bridge voltage of the step (its midpoint value is the step average), for the
# controllers' averaged terminal-voltage measurement
function _bridge_accumulate!(cs::PHControlled, xm, u::PHInput, h)
    any(_sw_on, cs.conv) || return nothing
    up = phs_control_inputs(cs, xm, u.gf; cf = u.cf)
    for (i, g) in enumerate(cs.conv)
        _sw_on(g) || continue
        g.dc.sw.uacc += h * up.uc[i]; g.dc.sw.tacc += h
    end
    return nothing
end

function _sample!(cs::PHControlled, x, u::PHInput)
    up = phs_control_inputs(cs, x, u.gf; cf = u.cf)
    f = phs_flows(cs.s, x, up)
    for (i, cv) in enumerate(cs.s.conv)
        g = cs.conv[i]
        (g isa Union{PHVsm,PHDroop,PHVoc,PHGfl} && g.Ts > 0) || continue
        vk = if _sw_on(g) && g.dc.sw.tacc > 0
            # a switching bridge: the controller measures the terminal voltage averaged over the elapsed PWM half period
            # (PWM-synchronous averaging; the instantaneous terminal voltage of an L interface jumps with every switch
            # state, and at a carrier peak or valley SVPWM applies a zero vector)
            vv = copy(f.v); vv[cv.row] = g.dc.sw.uacc / g.dc.sw.tacc
            _conv_vk(cs.s, cv, x, merge(f, (; v = vv)))
        else
            _conv_vk(cs.s, cv, x, f)
        end
        xc = view(x, cs.coff[i]:cs.coff[i]+nstates(g)-1)
        g isa PHGfl ? gfl_sample!(xc, g, vk, cs.s.omega_b) :
        g isa PHVoc ? voc_sample!(xc, g, vk, _conv_current(f, cv), cs.s.omega_b) :
        g isa PHDroop ? droop_sample!(xc, g, vk, _conv_current(f, cv), cs.s.omega_b) :
                        vsm_sample!(xc, g, vk, _conv_current(f, cv), cs.s.omega_b)
    end
    for (i, cv) in enumerate(cs.s.conv)              # switching bridges (upgrade 3): the next half's PWM from the new command
        g = cs.conv[i]
        _sw_on(g) || continue
        xc = view(x, cs.coff[i]:cs.coff[i]+nstates(g)-1)
        br = g.dc.sw
        t = br.thalf + br.H
        m = _modulation(g, xc, _conv_current(f, cv))
        bridge_sample!(br, cs.s.stationary ? m : m * cis(cs.s.omega_b * t + cs.s.theta0))
        br.uacc, br.tacc = 0.0im, 0.0
    end
    for mo in cs.s.motors                             # the motors' relays (review 76), from the same end-of-step voltages
        mo.ridx > 0 && relay_sample!(x, mo, abs(f.v[mo.node]))
    end
    return x
end

# A converter's physical DC side from the selected data (record section 12): the rating from PowerFactory's `uk`
# (`S_r / S_b = uk / Zs`), the nominal DC voltage `headroom / m_max`, the capacitor holding `H` seconds of the rating
# at it, the battery resistance giving the relative drop `drop` at rated power, and the battery voltage that holds
# the DC voltage at its nominal value at the operating point (bridge power `Re(E0 conj(ic))`).
#
# With `battery = :sossan` the battery is the measured BESS of Sossan et al. 2016 (record sections 31-32, part A): `E` behind
# `R_s` and three RC branches, Table I's column `soc`, per unit of the converter's rating on the DC base of its 300 V AC side
# (same C-rate and per-unit impedances: a similarity assumption), the battery directly on the link (a selected topology); the
# DC voltage is then where the battery holds the operating power, on the high root of `(1 + R_t G) v^2 - E v + R_t P = 0`
# (`R_t` the battery's total resistance, every branch at its DC equilibrium).
function _converter_dc(cv::PHConverter, E0, ic, omega_b; H::Real, headroom = nothing, drop = nothing, G::Real = 0.0,
                       m_max::Real = 2 / sqrt(3), mode::Symbol = :held, uk::Real = 0.1, battery::Symbol = :thevenin,
                       soc::String = "40-50", bridge::Symbol = :averaged, device = CAB760M12HM3_125C, fc::Real = 10e3,
                       mva_base::Real = 100.0)
    if battery === :sossan
        q = _converter_dc_sossan(cv, E0, ic, omega_b; H, G, m_max, mode, uk, soc)
        bridge === :averaged && return q
        bridge === :switching || throw(ArgumentError("converter_dc: bridge = $bridge (:averaged or :switching)"))
        # upgrade 3 (bridge.jl, reviews 77-77e): the N units' switching SiC bridge on the converter's rating
        sw = SwitchingBridge(device; Sr = q.Sr, fc, omega_b, mva_base)
        return PHConverterDC(q.c, q.Rb, q.Eb, q.G, q.m_max, q.mode, q.Sr, q.H, q.v_dc0, q.R, q.cb, q.source, sw)
    end
    bridge === :averaged || throw(ArgumentError("converter_dc: the switching bridge needs battery = :sossan (the unit's data)"))
    battery === :thevenin || throw(ArgumentError("converter_dc: battery = $battery (:thevenin or :sossan)"))
    (headroom isa Real && drop isa Real && H > 0 && headroom > 0 && 0 < drop < 1 && uk > 0) ||
        throw(ArgumentError("converter_dc: H = $H, headroom = $headroom, drop = $drop, uk = $uk"))
    Pr = uk / cv.Lc                                    # the rating on the system base
    v0 = headroom / m_max
    c = 2 * omega_b * H * Pr / v0^2
    Rb = drop * v0^2 / Pr
    Px = real(E0 * conj(ic))
    Eb = v0 + Rb * (G * v0 + Px / v0)
    abs(E0) / v0 <= m_max ||
        throw(ArgumentError("converter_dc: $(cv.name) needs |m| = $(abs(E0) / v0) > m_max = $m_max at its operating point"))
    return PHConverterDC(c, Rb, Eb, Float64(G), Float64(m_max), mode, Pr, Float64(H), v0)
end

# Sossan, Namor, Cherkaoui, Paolone, IEEE TSTE 7(4) 2016, doi 10.1109/TSTE.2016.2600103: Table I (p. 9), the 720 kVA / 500 kWh
# lithium-titanate BESS's model per state-of-charge range: E (V), R_s, R_1, C_1, R_2, C_2, R_3, C_3 (ohm, F); the 0.3/20 kV
# transformer (p. 10) gives its AC side, 300 V.
const SOSSAN_BESS = (S = 720e3, V_LL = 300.0, source = "Sossan et al. 2016, IEEE TSTE, Table I (p. 9)",
    table = Dict("0-20"   => (592.2, 0.029, 0.095, 8930.0, 0.04, 909.0, 2.5e-3, 544.2),
                 "20-40"  => (625.0, 0.021, 0.075, 9809.0, 0.009, 2139.0, 4.9e-5, 789.0),
                 "40-50"  => (652.9, 0.015, 0.090, 13996.0, 0.009, 2482.0, 2.4e-4, 2959.7),
                 "60-80"  => (680.2, 0.014, 0.079, 9499.0, 0.009, 2190.0, 6.8e-4, 100.2),
                 "80-100" => (733.2, 0.013, 0.199, 11234.0, 0.010, 2505.0, 6.0e-4, 6177.3)))

function _converter_dc_sossan(cv::PHConverter, E0, ic, omega_b; H::Real, G::Real, m_max::Real, mode::Symbol, uk::Real,
                              soc::String)
    haskey(SOSSAN_BESS.table, soc) || throw(ArgumentError("converter_dc: soc = $soc; Table I has $(sort(collect(keys(SOSSAN_BESS.table))))"))
    (H > 0 && uk > 0 && G >= 0) || throw(ArgumentError("converter_dc: H = $H, uk = $uk, G = $G"))
    Pr = uk / cv.Lc                                    # the rating on the system base
    Vb = 2 * sqrt(2 / 3) * SOSSAN_BESS.V_LL            # the DC base of the 300 V AC side (section 12)
    Zb = Vb^2 / SOSSAN_BESS.S
    E, Rs, R1, C1, R2, C2, R3, C3 = SOSSAN_BESS.table[soc]
    # per unit of the rating, then the system base (resistances divided by the rating, times its inverse for capacitances):
    # (cb/omega_b) v' = i with time in seconds: cb = omega_b tau / R
    Rsys(R) = (R / Zb) / Pr
    R = [Rsys(R1), Rsys(R2), Rsys(R3)]
    cb = [omega_b * R1 * C1, omega_b * R2 * C2, omega_b * R3 * C3] ./ R
    Eb, Rb = E / Vb, Rsys(Rs)
    Px = real(E0 * conj(ic))
    Rt = Rb + sum(R)
    disc = Eb^2 - 4 * (1 + Rt * G) * Rt * Px
    disc > 0 || throw(ArgumentError("converter_dc: $(cv.name): the battery cannot hold P = $Px (E = $Eb, R_t = $Rt)"))
    v0 = (Eb + sqrt(disc)) / (2 * (1 + Rt * G))
    c = 2 * omega_b * H * Pr / v0^2
    abs(E0) / v0 <= m_max ||
        throw(ArgumentError("converter_dc: $(cv.name) needs |m| = $(abs(E0) / v0) > m_max = $m_max at its operating point"))
    return PHConverterDC(c, Rb, Eb, Float64(G), Float64(m_max), mode, Pr, Float64(H), v0, R, cb,
                         SOSSAN_BESS.source * ", column $soc")
end
