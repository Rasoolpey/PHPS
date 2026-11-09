# Own open-component physical model; SI units, not an RMS case model.
# See docs/active/STEAM_TURBINE_SHAFT_INTERCONNECTION.md for its derivation.
abstract type AbstractSteamFlowLaw end

"""
Synthetic quasi-static valve/turbine conductances in kg/(s Pa).
No compressible-flow derivation, choke branch, speed dependence or calibration.
"""
struct LinearSteamConductance{T<:Real} <: AbstractSteamFlowLaw
    valve::T
    turbine::T
    function LinearSteamConductance(;valve, turbine)
        Cv,Ct=promote(float(valve),float(turbine))
        _positive(Cv,"valve conductance"); _positive(Ct,"turbine conductance")
        new{typeof(Cv)}(Cv,Ct)
    end
end

"""Solve the declared pressure/flow branch. Providers must also declare its domain."""
function pressure_flow end
function pressure_flow(law::LinearSteamConductance, state; admission, exhaust_pressure, speed)
    _positive(state.p,"vessel pressure"); _positive(exhaust_pressure,"exhaust pressure")
    _positive(speed,"shaft speed"); _finite(admission,"admission")
    0<=admission<=1 || throw(DomainError(admission,"admission must be in [0,1]"))
    state.p>=exhaust_pressure || throw(DomainError(state.p,"reverse pressure is outside the declared branch"))
    denom=admission*law.valve+law.turbine
    _positive(denom,"combined conductance")
    alpha=admission*law.valve/denom
    pressure_drop=state.p-exhaust_pressure
    inlet_pressure=exhaust_pressure+alpha*pressure_drop
    mass_flow=alpha*law.turbine*pressure_drop
    _finite(inlet_pressure,"turbine inlet pressure"); _nonnegative(mass_flow,"mass flow")
    (;inlet_pressure,mass_flow,constraint_derivative=-denom)
end

"""
Open vessel/valve/turbine/rotor in SI; state [S,M,l]. The power-to-torque
law is valid only for speed>minimum_speed. Sources and load remain boundaries.
"""
struct SteamTurbineShaft{V<:SteamVessel,F<:AbstractSteamFlowLaw,T<:Real}
    vessel::V
    flow_law::F
    inertia::T
    exhaust_pressure::T
    ambient_temperature::T
    minimum_speed::T
    efficiency::T
    mechanical_loss_fraction::T
    friction::T
    function SteamTurbineShaft(vessel::SteamVessel,flow_law::AbstractSteamFlowLaw;
            inertia,exhaust_pressure,ambient_temperature,minimum_speed,
            efficiency,mechanical_loss_fraction,friction)
        vals=promote(float(inertia),float(exhaust_pressure),float(ambient_temperature),
            float(minimum_speed),float(efficiency),float(mechanical_loss_fraction),float(friction))
        for (value,label) in zip(vals[1:4],("inertia","exhaust pressure","bath temperature","minimum speed"))
            _positive(value,label)
        end
        isfinite(vals[5]) && 0<=vals[5]<=1 || throw(DomainError(vals[5],"efficiency must be in [0,1]"))
        isfinite(vals[6]) && 0<=vals[6]<=1 || throw(DomainError(vals[6],"mechanical loss fraction must be in [0,1]"))
        _nonnegative(vals[7],"rotor friction")
        new{typeof(vessel),typeof(flow_law),typeof(vals[1])}(vessel,flow_law,vals...)
    end
end

function steam_state(model::SteamTurbineShaft,x)
    length(x)==3 || throw(DimensionMismatch("coupled state must be [S,M,l]"))
    st=vessel_state(model.vessel,view(x,1:2))
    _finite(x[3],"rotor momentum")
    speed=x[3]/model.inertia
    speed>model.minimum_speed || throw(DomainError(speed,"torque law requires speed>minimum_speed"))
    st.p>=model.exhaust_pressure || throw(DomainError(st.p,"reverse pressure is outside the declared branch"))
    kinetic_energy=x[3]^2/(2model.inertia)
    _finite(kinetic_energy,"rotor energy")
    total_energy=_finite(st.U+kinetic_energy,"total energy")
    (;vessel=st,momentum=x[3],speed,kinetic_energy,total_energy)
end
steam_energy(model::SteamTurbineShaft,x)=steam_state(model,x).total_energy

"""
State/input-modulated energetic Dirac graph after flow closure. This does not
determine flow, pressure or entropy; those are separate constitutive equations.
"""
function turbine_graph_ports(;flow,inlet_enthalpy,exhaust_enthalpy,speed,
                              ambient_temperature,mechanical_loss_fraction)
    _nonnegative(flow,"mass flow"); _positive(speed,"speed")
    _positive(ambient_temperature,"bath temperature")
    _finite(inlet_enthalpy,"inlet enthalpy"); _finite(exhaust_enthalpy,"exhaust enthalpy")
    chi=mechanical_loss_fraction
    isfinite(chi) && 0<=chi<=1 || throw(DomainError(chi,"loss fraction must be in [0,1]"))
    ns=(1-chi)*flow/speed
    nh=chi*flow/ambient_temperature
    z=zero(ns+nh)
    B=[z z ns nh; z z -ns -nh; -ns ns z z; -nh nh z z]
    efforts=[inlet_enthalpy,exhaust_enthalpy,speed,ambient_temperature]
    flows=B*efforts
    all(isfinite,flows) || throw(DomainError(flows,"nonfinite energetic graph flow"))
    (;B,efforts,flows,powers=efforts.*flows)
end

"""
Report coupled dynamics and all physical power/entropy contributions. Heat uses
Q=k*(Tf-T). Numerical entropy tolerance is explicit in W/K; nothing is clipped.
Availability uses the declared common bath temperature and chemical-potential
reference. The report is not a positivity, invariance or stability certificate.
"""
function steam_balance(model::SteamTurbineShaft,x;admission,
        heat_source_temperature=model.ambient_temperature,heat_conductance=0.0,
        feed_flow=0.0,feed_enthalpy=nothing,feed_entropy=nothing,load_torque=0.0,
        entropy_tolerance=0.0,stream_entropy_tolerance=1e-8,ambient_chemical_potential=0.0)
    state=steam_state(model,x)
    st=state.vessel
    _finite(load_torque,"load torque"); _finite(ambient_chemical_potential,"ambient chemical potential")
    _nonnegative(entropy_tolerance,"entropy tolerance")
    feed_enthalpy,feed_entropy=_feed_state(feed_flow,feed_enthalpy,feed_entropy)
    closure=pressure_flow(model.flow_law,st;admission,exhaust_pressure=model.exhaust_pressure,speed=state.speed)
    q,p1=closure.mass_flow,closure.inlet_pressure
    heat=heat_exchange(st.T,heat_source_temperature,heat_conductance)
    valve=throttle_stream(model.vessel.eos;flow=q,enthalpy=st.h,inlet_pressure=st.p,outlet_pressure=p1,
        inlet_entropy=st.s,entropy_state_tolerance=stream_entropy_tolerance)
    turbine=turbine_stage(model.vessel.eos;flow=q,inlet_enthalpy=st.h,inlet_pressure=p1,
        exhaust_pressure=model.exhaust_pressure,efficiency=model.efficiency,
        mechanical_loss_fraction=model.mechanical_loss_fraction,inlet_entropy=valve.outlet_entropy,
        entropy_state_tolerance=stream_entropy_tolerance)
    graph=turbine_graph_ports(;flow=q,inlet_enthalpy=st.h,exhaust_enthalpy=turbine.exhaust_enthalpy,
        speed=state.speed,ambient_temperature=model.ambient_temperature,
        mechanical_loss_fraction=model.mechanical_loss_fraction)
    vessel=vessel_balance(model.vessel,view(x,1:2);heat_power=heat.heat_power,feed_flow,
        feed_enthalpy,feed_entropy,outflow=q,entropy_tolerance)
    torque=-graph.flows[3]
    graph_residuals=(inlet_mass=graph.flows[1]-q,outlet_mass=graph.flows[2]+q,
        power=sum(graph.powers),mechanical_heat=-model.ambient_temperature*graph.flows[4]-turbine.mechanical_loss,
        shaft_power=state.speed*torque-turbine.shaft_power)
    momentum_rate=torque-load_torque-model.friction*state.speed
    dx=[vessel.dx[1],vessel.dx[2],momentum_rate]
    friction_power=model.friction*state.speed^2
    rejected_heat=turbine.mechanical_loss+friction_power
    entropy_productions=(mixing=vessel.sigma_mix,heat_exchange=heat.sigma_heat,
        valve=valve.sigma_valve,turbine=turbine.sigma_turbine,
        mechanical_loss=turbine.mechanical_loss/model.ambient_temperature,
        rotor_friction=friction_power/model.ambient_temperature)
    for sigma in values(entropy_productions)
        isfinite(sigma) && sigma>=-entropy_tolerance || throw(DomainError(sigma,"negative/nonfinite entropy production (W/K)"))
    end
    sigma_total=sum(values(entropy_productions))
    power=(heat_in=heat.heat_power,feed_in=feed_flow*feed_enthalpy,
        exhaust_out=q*turbine.exhaust_enthalpy,mechanical_heat_out=turbine.mechanical_loss,
        friction_heat_out=friction_power,load_out=load_torque*state.speed)
    energy_rate=st.T*dx[1]+st.mu*dx[2]+state.speed*dx[3]
    energy_supply=power.heat_in+power.feed_in-power.exhaust_out-
        power.mechanical_heat_out-power.friction_heat_out-power.load_out
    entropy_supply=heat.heat_power/heat_source_temperature+feed_flow*feed_entropy-
        q*turbine.exhaust_entropy-rejected_heat/model.ambient_temperature+sigma_total
    Ta,mua=model.ambient_temperature,ambient_chemical_potential
    availability_rate=energy_rate-Ta*dx[1]-mua*dx[2]
    availability_supply=(1-Ta/heat_source_temperature)*heat.heat_power+
        feed_flow*(feed_enthalpy-Ta*feed_entropy-mua)-
        q*(turbine.exhaust_enthalpy-Ta*turbine.exhaust_entropy-mua)-power.load_out-Ta*sigma_total
    all(isfinite,dx) && all(isfinite,values(power)) || throw(DomainError(dx,"nonfinite coupled rate/power"))
    all(isfinite,(energy_rate,energy_supply,entropy_supply,availability_rate,availability_supply)) ||
        throw(DomainError("nonfinite composite balance"))
    (;state,closure,heat,valve,turbine,graph,graph_residuals,dx,torque,power,entropy_productions,sigma_total,
      energy_rate,energy_supply,energy_residual=energy_rate-energy_supply,
      entropy_supply,entropy_residual=dx[1]-entropy_supply,
      availability_rate,availability_supply,availability_residual=availability_rate-availability_supply)
end

"""Standalone RHS; the caller owns solver events and all branch/guard enforcement."""
function steam_rhs!(dx,model::SteamTurbineShaft,x;kwargs...)
    length(dx)==3 || throw(DimensionMismatch("coupled derivative must have three entries"))
    balance=steam_balance(model,x;kwargs...)
    copyto!(dx,balance.dx)
    dx
end
