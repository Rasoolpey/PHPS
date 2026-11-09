# Rigid-shaft coupling of the physical steam component (PhysicalSteam, SI units) to the
# unchanged GENROU_PHTRUE kernel (per unit on the system base). Derivation, scope and
# evidence: docs/active/STEAM_MACHINE_SHAFT_COUPLING.md.
#
# One shared rotor. Its kinetic energy is S_b*H*omega^2, with S_b the power base in W and
# H the kernel's system-base inertia constant, so J*Omega_b^2 = 2*H*S_b. The kernel input
# Tm is the actual net shaft power on that base. Terminal voltage and field voltage are
# prescribed boundary inputs: no network, governor or exciter is attached here, and the
# machine's electromagnetic storage is not part of the energy reported.

"""
    SteamMachineShaft(vessel, flow_law, machine; power_base, pole_pairs=1, exhaust_pressure,
                      ambient_temperature, efficiency, mechanical_loss_fraction,
                      minimum_speed_pu, friction_pu=0.0)

Steam vessel, valve and turbine on one rigid shaft with a `GENROU_PHTRUE` machine. The
state is `[S, M, delta, omega, Eq', psi_d, Ed', psi_q]`: vessel entropy (J/K) and mass (kg)
followed by the machine's own states. `power_base` is in W. `friction_pu` is the rotor
friction power at rated speed divided by `power_base`; the friction power is
`friction_pu*power_base*omega^2`.

The pole-pair number sets the mechanical base speed `omega_b/pole_pairs` and the SI
inertia, but it cancels from the dynamics and from every reported power.
"""
struct SteamMachineShaft{S<:PhysicalSteam.SteamTurbineShaft}
    steam::S
    machine::GENROU_PHTRUE
    power_base::Float64
    pole_pairs::Int
    mechanical_base_speed::Float64
    friction_pu::Float64
end

function SteamMachineShaft(vessel::PhysicalSteam.SteamVessel, flow_law::PhysicalSteam.AbstractSteamFlowLaw,
        machine::GENROU_PHTRUE; power_base, pole_pairs::Integer = 1, exhaust_pressure,
        ambient_temperature, efficiency, mechanical_loss_fraction, minimum_speed_pu,
        friction_pu = 0.0)
    Sb = float(power_base)
    isfinite(Sb) && Sb > 0 || throw(DomainError(Sb, "power base must be finite and positive (W)"))
    pole_pairs >= 1 || throw(DomainError(pole_pairs, "pole-pair number must be at least one"))
    isfinite(friction_pu) && friction_pu >= 0 ||
        throw(DomainError(friction_pu, "per-unit friction must be finite and nonnegative"))
    isfinite(minimum_speed_pu) && minimum_speed_pu > 0 ||
        throw(DomainError(minimum_speed_pu, "per-unit minimum speed must be finite and positive"))
    base_speed = machine.p.omega_b / pole_pairs
    inertia = 2 * machine.p.H * Sb / base_speed^2
    steam = PhysicalSteam.SteamTurbineShaft(vessel, flow_law; inertia, exhaust_pressure,
        ambient_temperature, minimum_speed = minimum_speed_pu * base_speed, efficiency,
        mechanical_loss_fraction, friction = friction_pu * Sb / base_speed^2)
    return SteamMachineShaft(steam, machine, Sb, Int(pole_pairs), base_speed, float(friction_pu))
end

function _steam_rotor_state(c::SteamMachineShaft, z)
    length(z) == 8 || throw(DimensionMismatch("coupled state must be [S, M, delta, omega, Eq', psi_d, Ed', psi_q]"))
    return [z[1], z[2], c.steam.inertia * c.mechanical_base_speed * z[4]]
end

"""Thermal and mechanical energy `U(S,M) + S_b*H*omega^2` in J. No electromagnetic storage."""
steam_machine_energy(c::SteamMachineShaft, z) =
    PhysicalSteam.steam_energy(c.steam, _steam_rotor_state(c, z))

"""
    steam_machine_balance(c, z; admission, terminal_voltage, field_voltage, kwargs...)

Coupled rates and the thermal-mechanical power balance. `terminal_voltage` is the
network-frame pair `(V_Re, V_Im)` in per unit.

The state derivative `dz` uses the unchanged machine kernel with
`Tm = (shaft power - friction power)/power_base` (formulation B). `omega_rate_rotor` is the
same speed derivative obtained from the physical rotor with load torque
`power_base*(Te + D*(omega-1))/Omega_b` (formulation A); the two are equal identically.

`power.electrical_out` is the kernel-defined terminal power `power_base*(vd*id+vq*iq)`
withdrawn from the shaft. `power.damping_exchange` is `power_base*D*omega*(omega-1)`: a
signed exchange, not a dissipation about rest.
"""
function steam_machine_balance(c::SteamMachineShaft, z; admission, terminal_voltage, field_voltage,
        heat_source_temperature = c.steam.ambient_temperature, heat_conductance = 0.0,
        feed_flow = 0.0, feed_enthalpy = nothing, feed_entropy = nothing,
        entropy_tolerance = 0.0)
    xs = _steam_rotor_state(c, z)
    p, Sb, base_speed = c.machine.p, c.power_base, c.mechanical_base_speed
    omega = z[4]
    # Steam side with the shared rotor unloaded: its momentum rate is tau_t - b*Omega.
    steam = PhysicalSteam.steam_balance(c.steam, xs; admission, heat_source_temperature,
        heat_conductance, feed_flow, feed_enthalpy, feed_entropy, load_torque = 0.0,
        entropy_tolerance)
    shaft_power = steam.turbine.shaft_power
    friction_power = steam.power.friction_heat_out
    Tm = (shaft_power - friction_power) / Sb
    xm = z[3:8]
    um = [terminal_voltage[1], terminal_voltage[2], Tm, field_voltage]
    T = promote_type(eltype(xm), eltype(um))
    dxm = zeros(T, 6)
    ym = zeros(T, 10)
    _outputs!(ym, c.machine, xm, um, p, NoModes())
    _step!(dxm, ym, c.machine, xm, um, p, NoModes())
    terminal_power = ym[4]
    electrical_torque = terminal_power / omega
    load_torque = Sb * (electrical_torque + p.D * (omega - 1)) / base_speed
    omega_rate_rotor = (steam.dx[3] - load_torque) / (c.steam.inertia * base_speed)
    dz = vcat(steam.dx[1], steam.dx[2], dxm)
    power = (heat_in = steam.power.heat_in, feed_in = steam.power.feed_in,
        exhaust_out = steam.power.exhaust_out, mechanical_heat_out = steam.power.mechanical_heat_out,
        friction_heat_out = friction_power, electrical_out = Sb * terminal_power,
        damping_exchange = Sb * p.D * omega * (omega - 1))
    vessel = steam.state.vessel
    energy_rate = vessel.T * dz[1] + vessel.mu * dz[2] + 2 * Sb * p.H * omega * dxm[2]
    energy_supply = power.heat_in + power.feed_in - power.exhaust_out - power.mechanical_heat_out -
        power.friction_heat_out - power.electrical_out - power.damping_exchange
    all(isfinite, dz) || throw(DomainError(dz, "nonfinite coupled rate"))
    return (; steam, dz, mechanical_power_pu = Tm, shaft_power, terminal_power_pu = terminal_power,
        electrical_torque_pu = electrical_torque, load_torque, omega_rate_kernel = dxm[2],
        omega_rate_rotor, machine_outputs = ym, power, energy_rate, energy_supply,
        energy_residual = energy_rate - energy_supply)
end

"""Coupled right-hand side; the caller owns solver events and guard enforcement."""
function steam_machine_rhs!(dz, c::SteamMachineShaft, z; kwargs...)
    length(dz) == 8 || throw(DimensionMismatch("coupled derivative must have eight entries"))
    copyto!(dz, steam_machine_balance(c, z; kwargs...).dz)
    return dz
end

"""
    steam_machine_shift(c, z, zstar)

Bregman shift of the thermal-mechanical energy about `zstar` in the physical storage
coordinates `(S, M, l)`, `l = J*Omega_b*omega`:

    W = E(z) - E(zstar) - grad E(zstar) . (x - xstar),   grad E = (T, mu, Omega).

Its rotor part is `power_base*H*(omega - omegastar)^2`. `W` contains neither the rotor
angle nor any electromagnetic storage, so it is not a storage function of the eight-state
system. Returns `W` (J), the effort differences and the storage-coordinate difference.
"""
function steam_machine_shift(c::SteamMachineShaft, z, zstar)
    x, xstar = _steam_rotor_state(c, z), _steam_rotor_state(c, zstar)
    state, reference = PhysicalSteam.steam_state(c.steam, x), PhysicalSteam.steam_state(c.steam, xstar)
    efforts = [state.vessel.T, state.vessel.mu, state.speed]
    reference_efforts = [reference.vessel.T, reference.vessel.mu, reference.speed]
    difference = x .- xstar
    shift = state.total_energy - reference.total_energy - sum(reference_efforts .* difference)
    return (; shift, effort_difference = efforts .- reference_efforts, state_difference = difference,
        rotor_shift = c.power_base * c.machine.p.H * (z[4] - zstar[4])^2)
end
