"""
Standalone physical vessel primitives in SI units. They are not RMS case models.
The perfect-gas provider is for validation, not calibrated steam data. See
`docs/active/STEAM_RESOURCE_PH_FORMULATION.md` for the model and its open obligations.
"""
module PhysicalSteam

using ForwardDiff
using LinearAlgebra: dot

export AbstractSteamEOS, PerfectGasEOS, SteamVessel, internal_energy,
       thermodynamic_state, stream_enthalpy, stream_entropy, perfect_gas_state,
       vessel_energy, vessel_state, vessel_ports, vessel_rhs!, vessel_balance,
       heat_exchange, throttle_stream, turbine_stage, reserve_admission_flow,
       choked_orifice_flow, AbstractSteamFlowLaw, LinearSteamConductance,
       SteamTurbineShaft, pressure_flow, steam_state, steam_energy,
       turbine_graph_ports, steam_rhs!, steam_balance,
       iapws95_helmholtz_parts, iapws95_properties, IAPWS95EOS, IAPWS95ConsistencyError,
       iapws95_state, iapws95_entropy_density_state, iapws95_stream_state,
       STEAM_ALGEBRAIC_NAMES, STEAM_CONSTRAINT_NAMES, steam_algebraic_state,
       steam_constraint_residual, steam_constraint_dynamics,
       steam_constraint_jacobian, steam_domain_margins,
       vessel_shifted_energy, steam_shifted_rate, steam_local_rate_form

abstract type AbstractSteamEOS end

_positive(v, label) = isfinite(v) && v>0 ? v : throw(DomainError(v, "$label must be finite and positive"))
_nonnegative(v, label) = isfinite(v) && v>=0 ? v : throw(DomainError(v, "$label must be finite and nonnegative"))
_finite(v, label) = isfinite(v) ? v : throw(DomainError(v, "$label must be finite"))

"""Validation EOS: calorically perfect gas with explicit SI parameters and references."""
struct PerfectGasEOS{T<:Real} <: AbstractSteamEOS
    cv::T
    R::T
    T_ref::T
    rho_ref::T
    s_ref::T
    function PerfectGasEOS(;cv, R, T_ref=300.0, rho_ref=1.0, s_ref=0.0)
        vals = promote(float(cv), float(R), float(T_ref), float(rho_ref), float(s_ref))
        for (value, label) in zip(vals[1:4], ("cv", "R", "T_ref", "rho_ref"))
            _positive(value, label)
        end
        _finite(vals[5], "s_ref")
        new{typeof(vals[1])}(vals...)
    end
end

"""
    internal_energy(eos, S, M, V)

Fundamental relation U in J, with S in J/K, M in kg and V in m^3. Providers must
support ForwardDiff and declare a physically admissible phase/domain. Additional
providers should throw DomainError outside their material domain.
"""
function internal_energy end
function internal_energy(eos::PerfectGasEOS, S, M, V)
    _finite(S, "entropy"); _positive(M, "mass"); _positive(V, "volume")
    T = eos.T_ref*exp((S/M-eos.s_ref)/eos.cv)*(M/(V*eos.rho_ref))^(eos.R/eos.cv)
    _positive(T, "temperature")
    _finite(eos.cv*M*T, "internal energy")
end

"""Intensive quantities derived from the provider's U, rather than independent fitted maps."""
function thermodynamic_state(eos::AbstractSteamEOS, S, M, V)
    _finite(S, "entropy"); _positive(M, "mass"); _positive(V, "volume")
    z = collect(promote(S, M, V))
    U = internal_energy(eos, z...)
    # One direction per pass. With the default three, Dual inputs of exactly six
    # partials gave wrong efforts through the IAPWS-95 provider at Julia's default
    # optimisation level (docs/active/STEAM_MACHINE_SHAFT_COUPLING.md, section 8).
    # This avoids that shape; the cause is not established.
    energy = v->internal_energy(eos, v...)
    g = ForwardDiff.gradient(energy, z, ForwardDiff.GradientConfig(energy, z, ForwardDiff.Chunk{1}()))
    T, mu, p = g[1], g[2], -g[3]
    _finite(U, "internal energy"); _positive(T, "temperature")
    _finite(mu, "chemical potential"); _positive(p, "pressure")
    s = S/M
    h = mu+T*s
    _finite(s, "specific entropy"); _finite(h, "specific enthalpy")
    (;U, T, mu, p, s, h)
end

"""Stream h(s,p) in J/kg. A provider must use the same material law as U."""
function stream_enthalpy end
"""Stream s(h,p) in J/(kg K). A provider must use the same material law as U."""
function stream_entropy end

function stream_enthalpy(eos::PerfectGasEOS, s, p)
    _finite(s, "specific entropy"); _positive(p, "stream pressure")
    cp = eos.cv+eos.R
    T = eos.T_ref*exp((s-eos.s_ref)/cp)*(p/(eos.rho_ref*eos.R*eos.T_ref))^(eos.R/cp)
    _positive(T, "stream temperature")
    _finite(cp*T, "stream enthalpy")
end
function stream_entropy(eos::PerfectGasEOS, h, p)
    _positive(h, "perfect-gas enthalpy"); _positive(p, "stream pressure")
    T = h/(eos.cv+eos.R)
    s = eos.s_ref+eos.cv*log(T/eos.T_ref)-eos.R*log(p/(eos.R*T*eos.rho_ref))
    _finite(s, "stream entropy")
end

"""Construct [S,M] for the validation gas from mass, temperature and fixed volume."""
function perfect_gas_state(eos::PerfectGasEOS; mass, temperature, volume)
    _positive(mass, "mass"); _positive(temperature, "temperature"); _positive(volume, "volume")
    s = eos.s_ref+eos.cv*log(temperature/eos.T_ref)-eos.R*log(mass/(volume*eos.rho_ref))
    [mass*s, mass]
end

struct SteamVessel{E<:AbstractSteamEOS,T<:Real}
    eos::E
    volume::T
    function SteamVessel(eos::AbstractSteamEOS; volume)
        _positive(volume, "volume")
        new{typeof(eos),typeof(float(volume))}(eos, float(volume))
    end
end
function _state(v::SteamVessel, x)
    length(x)==2 || throw(DimensionMismatch("vessel state must be [S,M]"))
    thermodynamic_state(v.eos, x[1], x[2], v.volume)
end
vessel_state(v::SteamVessel, x) = _state(v, x)
vessel_energy(v::SteamVessel, x) = _state(v, x).U

"""Modulated energetic pH ports: inputs (Q/T,qfeed,qout), outputs (T,hfeed,-h)."""
function vessel_ports(v::SteamVessel, x; feed_enthalpy=0.0)
    st = _state(v, x)
    _finite(feed_enthalpy, "feed enthalpy")
    G = [one(st.T) (feed_enthalpy-st.mu)/st.T -st.s;
         zero(st.T) one(st.T) -one(st.T)]
    e = [st.T, st.mu]
    (;J=zeros(eltype(G),2,2), R=zeros(eltype(G),2,2), G, effort=e, output=transpose(G)*e)
end

function _feed_state(feed_flow,feed_enthalpy,feed_entropy)
    _nonnegative(feed_flow,"feed mass flow")
    if feed_flow>0 && (isnothing(feed_enthalpy) || isnothing(feed_entropy))
        throw(ArgumentError("positive feed flow requires explicit feed enthalpy and entropy"))
    end
    hf=isnothing(feed_enthalpy) ? 0.0 : _finite(feed_enthalpy,"feed enthalpy")
    sf=isnothing(feed_entropy) ? 0.0 : _finite(feed_entropy,"feed entropy")
    hf,sf
end

function _rates(v, x, heat_power, feed_flow, feed_enthalpy, feed_entropy, outflow, entropy_tolerance)
    st = _state(v, x)
    _finite(heat_power, "heat power")
    _nonnegative(feed_flow, "feed mass flow"); _nonnegative(outflow, "discharge mass flow")
    feed_enthalpy,feed_entropy=_feed_state(feed_flow,feed_enthalpy,feed_entropy)
    _nonnegative(entropy_tolerance, "entropy production tolerance")
    sigma_mix = feed_flow*((feed_enthalpy-st.mu)/st.T-feed_entropy)
    _finite(sigma_mix, "mixing entropy production")
    sigma_mix >= -entropy_tolerance ||
        throw(DomainError(sigma_mix, "feed violates mixing entropy admissibility (W/K)"))
    Sdot = heat_power/st.T+feed_flow*(feed_enthalpy-st.mu)/st.T-outflow*st.s
    Mdot = feed_flow-outflow
    _finite(Sdot, "entropy rate"); _finite(Mdot, "mass rate")
    (;st, Sdot, Mdot, sigma_mix,feed_enthalpy,feed_entropy)
end

"""
    vessel_rhs!(dx, vessel, x; heat_power=0, feed_flow=0, feed_enthalpy=nothing,
                feed_entropy=nothing, outflow=0, entropy_tolerance=0)

Physical first-law dynamics in SI. Feed entropy production is checked, not clipped.
Any numerical tolerance is explicit (W/K); its default is zero. The caller owns the
flow/heat laws and domain/guard enforcement. This prototype may allocate.
"""
function vessel_rhs!(dx, v::SteamVessel, x; heat_power=0.0, feed_flow=0.0,
                     feed_enthalpy=nothing, feed_entropy=nothing, outflow=0.0, entropy_tolerance=0.0)
    length(dx)==2 || throw(DimensionMismatch("vessel derivative must have two entries"))
    r = _rates(v, x, heat_power, feed_flow, feed_enthalpy, feed_entropy, outflow, entropy_tolerance)
    dx[1], dx[2] = r.Sdot, r.Mdot
    dx
end

"""Report signed rates, port powers and mixing production without an unowned residual supply."""
function vessel_balance(v::SteamVessel, x; heat_power=0.0, feed_flow=0.0,
                        feed_enthalpy=nothing, feed_entropy=nothing, outflow=0.0, entropy_tolerance=0.0)
    r = _rates(v, x, heat_power, feed_flow, feed_enthalpy, feed_entropy, outflow, entropy_tolerance)
    energy_rate = r.st.T*r.Sdot+r.st.mu*r.Mdot
    feed_power, discharge_power = feed_flow*r.feed_enthalpy, outflow*r.st.h
    supplied_power = heat_power+feed_power-discharge_power
    entropy_transport = heat_power/r.st.T+feed_flow*r.feed_entropy-outflow*r.st.s
    (;state=r.st, dx=[r.Sdot,r.Mdot], heat_power, feed_power, discharge_power,
      energy_rate, supplied_power,
      energy_residual=energy_rate-supplied_power,
      entropy_transport, sigma_mix=r.sigma_mix,
      entropy_residual=r.Sdot-entropy_transport-r.sigma_mix)
end

function heat_exchange(T, source_T, conductance)
    _positive(T, "vessel temperature"); _positive(source_T, "source temperature")
    _nonnegative(conductance, "thermal conductance")
    heat_power = conductance*(source_T-T)
    sigma_heat = heat_power*(1/T-1/source_T)
    (;heat_power, sigma_heat)
end

function _stream_entropy(eos,h,p,inlet_entropy,tolerance)
    _nonnegative(tolerance,"stream entropy consistency tolerance (J/(kg K))")
    mapped=stream_entropy(eos,h,p)
    supplied=isnothing(inlet_entropy) ? mapped : _finite(inlet_entropy,"inlet entropy")
    residual=mapped-supplied
    abs(residual)<=tolerance || throw(DomainError(residual,"inconsistent connected stream entropy (J/(kg K))"))
    supplied,residual
end

"""Static isenthalpic throttle. An explicit connected entropy is checked against the EOS map."""
function throttle_stream(eos::AbstractSteamEOS; flow, enthalpy, inlet_pressure, outlet_pressure,
                         inlet_entropy=nothing,entropy_state_tolerance=1e-8)
    _nonnegative(flow, "mass flow")
    _positive(inlet_pressure, "inlet pressure"); _positive(outlet_pressure, "outlet pressure")
    flow>0 && outlet_pressure>inlet_pressure &&
        throw(DomainError(outlet_pressure, "positive throttle flow requires outlet pressure <= inlet pressure"))
    sin,entropy_state_residual=_stream_entropy(eos,enthalpy,inlet_pressure,inlet_entropy,entropy_state_tolerance)
    sout = stream_entropy(eos, enthalpy, outlet_pressure)
    (;enthalpy, inlet_entropy=sin, outlet_entropy=sout, sigma_valve=flow*(sout-sin),entropy_state_residual)
end

"""Static adiabatic expansion; reports raw EOS entropy production, with no division by flow."""
function turbine_stage(eos::AbstractSteamEOS; flow, inlet_enthalpy, inlet_pressure,
                       exhaust_pressure, efficiency, mechanical_loss_fraction=0.0,
                       inlet_entropy=nothing,entropy_state_tolerance=1e-8)
    _nonnegative(flow, "mass flow")
    _positive(inlet_pressure, "inlet pressure"); _positive(exhaust_pressure, "exhaust pressure")
    exhaust_pressure<=inlet_pressure || throw(DomainError(exhaust_pressure, "expansion requires exhaust pressure <= inlet pressure"))
    isfinite(efficiency) && 0<=efficiency<=1 || throw(DomainError(efficiency, "efficiency must be in [0,1]"))
    isfinite(mechanical_loss_fraction) && 0<=mechanical_loss_fraction<=1 ||
        throw(DomainError(mechanical_loss_fraction, "mechanical loss fraction must be in [0,1]"))
    sin,entropy_state_residual=_stream_entropy(eos,inlet_enthalpy,inlet_pressure,inlet_entropy,entropy_state_tolerance)
    his = exhaust_pressure==inlet_pressure ? inlet_enthalpy : stream_enthalpy(eos, sin, exhaust_pressure)
    his<=inlet_enthalpy || throw(DomainError(his, "EOS gives a negative ideal expansion enthalpy drop"))
    hout = inlet_enthalpy-efficiency*(inlet_enthalpy-his)
    sout = stream_entropy(eos, hout, exhaust_pressure)
    power = flow*efficiency*(inlet_enthalpy-his)
    mechanical_loss = mechanical_loss_fraction*power
    shaft_power = power-mechanical_loss
    (;ideal_exhaust_enthalpy=his, exhaust_enthalpy=hout, exhaust_entropy=sout,
      power, mechanical_loss, shaft_power, sigma_turbine=flow*(sout-sin),entropy_state_residual)
end

"""Illustrative reserve-limited admission law; not a calibrated valve/swallowing model."""
function reserve_admission_flow(; mass, pressure, exhaust_pressure, admission,
                                reserve_mass, mass_scale, nominal_flow)
    _positive(mass, "mass"); _positive(pressure, "pressure"); _positive(exhaust_pressure, "exhaust pressure")
    _finite(admission, "admission"); _positive(reserve_mass, "reserve mass")
    _positive(mass_scale, "mass scale"); _nonnegative(nominal_flow, "nominal flow")
    reserve = max(mass-reserve_mass, zero(mass))
    nominal_flow*clamp(admission, 0, 1)*reserve/(mass_scale+reserve)*max(1-exhaust_pressure/pressure, 0)
end

"""
    choked_orifice_flow(eos::PerfectGasEOS; pressure, temperature, back_pressure,
                        area, discharge_coefficient=1, admission=1)

Ideal-gas choked discharge (kg/s). Throws outside the sonic pressure-ratio branch;
there is no implicit continuation, pressure clipping or subcritical flow model.
"""
function choked_orifice_flow(eos::PerfectGasEOS; pressure, temperature, back_pressure,
                             area, discharge_coefficient=1.0, admission=1.0)
    _positive(pressure, "pressure"); _positive(temperature, "temperature")
    _nonnegative(back_pressure, "back pressure"); _nonnegative(area, "orifice area")
    _nonnegative(discharge_coefficient, "discharge coefficient"); _finite(admission, "admission")
    gamma = 1+eos.R/eos.cv
    critical = (2/(gamma+1))^(gamma/(gamma-1))
    back_pressure/pressure<=critical ||
        throw(DomainError(back_pressure/pressure, "pressure ratio is outside the choked-flow branch"))
    discharge_coefficient*area*clamp(admission, 0, 1)*pressure*sqrt(gamma/(eos.R*temperature))*
        (2/(gamma+1))^((gamma+1)/(2*(gamma-1)))
end

include("steam_turbine_shaft.jl")
include("iapws95.jl")
include("iapws95_provider.jl")
include("steam_composition.jl")
include("steam_shifted_balance.jl")

end # module PhysicalSteam
