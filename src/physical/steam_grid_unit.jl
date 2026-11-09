# The physical steam component as the mechanical source of one machine in an assembled
# grid. Derivation, scope and evidence: docs/active/STEAM_GRID_CONSTRAINT_BLOCK.md.
#
# The grid DAE is the unchanged assembled system x' = f(x,V), 0 = g(x,V) (KCL at every
# bus). The driven machine's Tm input must be a constant source c0 in that system. The
# kernel's swing row is affine in Tm with slope 1/(2*H*omega), so the steam supply enters
# as the correction (Tm_steam - c0)/(2*H*omega) on that one row; no RMS code is changed.
# Composite: X = (x, S, M), algebraic unknowns (V, y) with y the nine steam unknowns.

"""
    case_without_component(path, name) -> Case

Load the case at `path` with component `name` and every wire to or from it removed. A
machine input that loses its wire becomes an undriven constant at initialisation.
"""
function case_without_component(path::AbstractString, name::AbstractString)
    raw = JSON3.read(read(path, String), Dict{String,Any})
    haskey(raw["components"], name) || throw(ArgumentError("case has no component $name"))
    delete!(raw["components"], name)
    raw["connections"] = [w for w in raw["connections"]
                          if !startswith(w["from"], name * ".") && !startswith(w["to"], name * ".")]
    file = joinpath(mktempdir(), "system.json")
    open(io -> JSON3.write(io, raw), file, "w")
    return load_case(file)
end

"""
    SteamGridUnit(sys, machine_name, vessel, flow_law; exhaust_pressure, ambient_temperature,
                  efficiency, mechanical_loss_fraction, minimum_speed_pu, friction_pu=0.0,
                  pole_pairs=1)

The steam vessel, valve and turbine on the rotor of machine `machine_name` of the
assembled system `sys`. The machine must be a `GENROU_PHTRUE` whose `Tm` input is a
constant source (assemble a case without its governor). The power base is the system
base of the case; the shared rotor has `J*Omega_b^2 = 2*H*S_b` with the kernel's
system-base `H`.

Composite differential state `X = [x; S; M]` with `x` the states of `sys`; composite
algebraic unknowns `(V, y)` with `V` the bus voltages of `sys` and `y` the nine steam
unknowns of `STEAM_ALGEBRAIC_NAMES`.
"""
struct SteamGridUnit{S<:PhysicalSteam.SteamTurbineShaft}
    sys::DAESystem
    machine::Int
    steam::S
    power_base::Float64
    mechanical_base_speed::Float64
    constant_power::Float64
end

function SteamGridUnit(sys::DAESystem, machine_name::AbstractString, vessel::PhysicalSteam.SteamVessel,
        flow_law::PhysicalSteam.AbstractSteamFlowLaw; exhaust_pressure, ambient_temperature,
        efficiency, mechanical_loss_fraction, minimum_speed_pu, friction_pu = 0.0,
        pole_pairs::Integer = 1)
    k = findfirst(c -> name(c) == machine_name, sys.comps)
    k === nothing && throw(ArgumentError("system has no component $machine_name"))
    machine = sys.comps[k]
    machine isa GENROU_PHTRUE || throw(ArgumentError("$machine_name is not a GENROU_PHTRUE"))
    source = sys.sources[k][findfirst(==("Tm"), input_names(machine))]
    source.kind === SRC_CONST ||
        throw(ArgumentError("$machine_name.Tm is driven by another component; assemble the case without its governor"))
    Sb = 1e6 * sys.case.config.mva_base
    shaft = SteamMachineShaft(vessel, flow_law, machine; power_base = Sb, pole_pairs, exhaust_pressure,
        ambient_temperature, efficiency, mechanical_loss_fraction, minimum_speed_pu, friction_pu)
    return SteamGridUnit(sys, k, shaft.steam, Sb, shaft.mechanical_base_speed, source.value)
end

"""Number of composite differential states and of composite algebraic unknowns."""
steam_grid_sizes(u::SteamGridUnit) = (differential = u.sys.n_diff + 2, algebraic = nalg(u.sys) + 9)

# Rotor state of the steam component from the machine speed in X.
function _steam_grid_rotor_state(u::SteamGridUnit, X)
    n = u.sys.n_diff
    length(X) == n + 2 || throw(DimensionMismatch("composite state must have $(n + 2) entries"))
    omega = X[u.sys.offsets[u.machine] + 1]
    return [X[n + 1], X[n + 2], u.steam.inertia * u.mechanical_base_speed * omega]
end

"""The nine steam unknowns from the implemented steam path at composite state `X`."""
steam_grid_algebraic(u::SteamGridUnit, X; admission) =
    PhysicalSteam.steam_algebraic_state(u.steam, _steam_grid_rotor_state(u, X); admission).y

"""
    steam_grid_dae(u, X, V, y; admission, heat and feed keywords...) -> (F, G, report)

The composite semi-explicit DAE `X' = F(X,V,y)`, `0 = G(X,V,y)`, root-free in all three
arguments. `F` is the grid right-hand side with the machine's speed row corrected by
`(Tm_steam - c0)/(2*H*omega)`, followed by the vessel entropy and mass rates. `G` is the
grid's KCL residual followed by the nine steam constraints. `Tm_steam` is the net shaft
power (after rotor friction) divided by the power base.
"""
function steam_grid_dae(u::SteamGridUnit, X, V, y; admission,
        heat_source_temperature = u.steam.ambient_temperature, heat_conductance = 0.0,
        feed_flow = 0.0, feed_enthalpy = nothing, feed_entropy = nothing)
    sys = u.sys
    n = sys.n_diff
    rotor = _steam_grid_rotor_state(u, X)
    f, g = dae_residual(sys, view(X, 1:n), V)
    row = sys.offsets[u.machine] + 1
    omega = X[row]
    dynamics = PhysicalSteam.steam_constraint_dynamics(u.steam, rotor, y; heat_source_temperature,
        heat_conductance, feed_flow, feed_enthalpy, feed_entropy, load_torque = 0.0)
    speed = u.mechanical_base_speed * omega
    shaft_power = (1 - u.steam.mechanical_loss_fraction) * dynamics.turbine_power
    friction_power = u.steam.friction * speed^2
    Tm = (shaft_power - friction_power) / u.power_base
    constraints = PhysicalSteam.steam_constraint_residual(u.steam, rotor, y; admission)
    T = promote_type(eltype(f), typeof(Tm), eltype(constraints))
    F = Vector{T}(undef, n + 2)
    copyto!(F, 1, f, 1, n)
    F[row] += (Tm - u.constant_power) / (2 * sys.comps[u.machine].p.H * omega)
    F[n + 1] = dynamics.dx[1]
    F[n + 2] = dynamics.dx[2]
    G = Vector{T}(undef, length(g) + 9)
    copyto!(G, 1, g, 1, length(g))
    copyto!(G, length(g) + 1, constraints, 1, 9)
    return F, G, (; mechanical_power_pu = Tm, shaft_power, friction_power, heat_power = dynamics.heat_power)
end

"""
    steam_grid_jacobians(u, X, V, y; kwargs...)

Jacobian blocks of the composite DAE by ForwardDiff, and the reduced linearisation
`A = F_X - F_z*G_z^{-1}*G_X` with `z = (V, y)`. AD results through the steam kernel must
be cross-checked (docs/active/STEAM_MACHINE_SHAFT_COUPLING.md, section 8); the audit
compares every block with central differences.
"""
function steam_grid_jacobians(u::SteamGridUnit, X, V, y; kwargs...)
    FX = ForwardDiff.jacobian(v -> steam_grid_dae(u, v, V, y; kwargs...)[1], X)
    GX = ForwardDiff.jacobian(v -> steam_grid_dae(u, v, V, y; kwargs...)[2], X)
    FV = ForwardDiff.jacobian(v -> steam_grid_dae(u, X, v, y; kwargs...)[1], V)
    GV = ForwardDiff.jacobian(v -> steam_grid_dae(u, X, v, y; kwargs...)[2], V)
    Fy = ForwardDiff.jacobian(v -> steam_grid_dae(u, X, V, v; kwargs...)[1], y)
    Gy = ForwardDiff.jacobian(v -> steam_grid_dae(u, X, V, v; kwargs...)[2], y)
    Gz = hcat(GV, Gy)
    reduced = FX - hcat(FV, Fy) * (Gz \ GX)
    return (; FX, GX, FV, GV, Fy, Gy, Gz, reduced)
end

"""
    grid_rotation_direction(sys, x, V) -> (dx, dV)

Generator of a common rotation of every rotor angle and every bus voltage phasor: one
radian on the angle state of each swing source, and `(-Vq, Vd)` at each bus.
"""
function grid_rotation_direction(sys::DAESystem, x, V)
    dx = zeros(length(x))
    for k in sys.coi_members
        dx[sys.offsets[k]] = 1.0
    end
    dV = zeros(length(V))
    for i in 1:nbus(sys)
        dV[2i - 1] = -V[2i]
        dV[2i] = V[2i - 1]
    end
    return dx, dV
end
