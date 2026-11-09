# Composition of the restricted IAPWS-95 chart with the vessel/valve/turbine/rotor
# model, written as a semi-explicit DAE x'=f(x,y,u), 0=g(x,y,u). Derivation, scope
# and evidence: docs/active/STEAM_COMPONENT_MANIFOLD.md.
#
# The constraint map uses only the guarded forward (T,rho) chart, so it contains no
# root solve. The implemented nested inverses supply a solution, not the definition.
# Declared scope: IAPWS95EOS with the synthetic LinearSteamConductance law.
using LinearAlgebra: Diagonal, det, svdvals

const SteamIAPWSComposition=SteamTurbineShaft{<:SteamVessel{IAPWS95EOS},<:LinearSteamConductance}

"""Order of the algebraic unknowns `y`. Units: K, kg/s, Pa, then (kg/m^3, K) pairs."""
const STEAM_ALGEBRAIC_NAMES=(:T,:q,:p1,:rho1,:T1,:rho_i,:T_i,:rho_o,:T_o)
"""Order of the constraints `g`. Units: J/(kg K), kg/s, kg/s, Pa, J/kg, Pa, J/(kg K), Pa, J/kg."""
const STEAM_CONSTRAINT_NAMES=(:vessel_entropy,:valve_flow,:swallowing_flow,
    :valve_outlet_pressure,:valve_enthalpy,:isentropic_pressure,:isentropic_entropy,
    :exhaust_pressure,:exhaust_enthalpy)

function _composition_lengths(x,y)
    length(x)==3 || throw(DimensionMismatch("coupled state must be [S,M,l]"))
    length(y)==9 || throw(DimensionMismatch("algebraic vector must have nine entries"))
end

# Root-free constraint map with every parameter explicit, so that it can be
# differentiated in any of them.
function _steam_constraint(eos,volume,Cv,Ct,pc,eta,x,y,admission)
    _composition_lengths(x,y)
    S,M=x[1],x[2]
    T,q,p1,rho1,T1,rhoi,Ti,rhoo,To=y
    v=iapws95_state(eos;temperature=T,density=M/volume)
    a1=iapws95_state(eos;temperature=T1,density=rho1)
    ai=iapws95_state(eos;temperature=Ti,density=rhoi)
    ao=iapws95_state(eos;temperature=To,density=rhoo)
    [v.entropy-S/M,
     q-admission*Cv*(v.pressure-p1),
     q-Ct*(p1-pc),
     a1.pressure-p1,
     a1.enthalpy-v.enthalpy,
     ai.pressure-pc,
     ai.entropy-a1.entropy,
     ao.pressure-pc,
     ao.enthalpy-(a1.enthalpy-eta*(a1.enthalpy-ai.enthalpy))]
end

"""
    steam_constraint_residual(model, x, y; admission)

Constraint map `g(x,y,u)` of the composed component in the forward chart; see
`STEAM_CONSTRAINT_NAMES` for the order and units. Evaluating it outside the
restricted chart raises `DomainError`.
"""
steam_constraint_residual(model::SteamIAPWSComposition,x,y;admission)=
    _steam_constraint(model.vessel.eos,model.vessel.volume,model.flow_law.valve,
        model.flow_law.turbine,model.exhaust_pressure,model.efficiency,x,y,admission)

"""
    steam_algebraic_state(model, x; admission)

The nine algebraic unknowns from the implemented path: vessel inverse through the
energy gradient, linear pressure/flow closure and the nested stream inverses. The
turbine inlet enthalpy passed on is the vessel enthalpy, as in `steam_balance`.
"""
function steam_algebraic_state(model::SteamIAPWSComposition,x;admission)
    state=steam_state(model,x)
    st=state.vessel
    eos,pc=model.vessel.eos,model.exhaust_pressure
    closure=pressure_flow(model.flow_law,st;admission,exhaust_pressure=pc,speed=state.speed)
    outlet=iapws95_stream_state(eos;pressure=closure.inlet_pressure,enthalpy=st.h)
    ideal=iapws95_stream_state(eos;pressure=pc,entropy=outlet.entropy)
    exhaust_enthalpy=st.h-model.efficiency*(st.h-ideal.enthalpy)
    exhaust=iapws95_stream_state(eos;pressure=pc,enthalpy=exhaust_enthalpy)
    y=[st.T,closure.mass_flow,closure.inlet_pressure,outlet.density,outlet.temperature,
       ideal.density,ideal.temperature,exhaust.density,exhaust.temperature]
    (;y,vessel=st,speed=state.speed,closure,valve_outlet=outlet,isentropic=ideal,exhaust)
end

"""
    steam_constraint_dynamics(model, x, y; kwargs...)

Differential part `f(x,y,u)` evaluated from the forward chart only. It takes the
turbine enthalpy drop from the valve-outlet and exhaust states in `y`.
"""
function steam_constraint_dynamics(model::SteamIAPWSComposition,x,y;
        heat_source_temperature=model.ambient_temperature,heat_conductance=0.0,
        feed_flow=0.0,feed_enthalpy=nothing,feed_entropy=nothing,load_torque=0.0)
    _composition_lengths(x,y)
    eos=model.vessel.eos
    S,M,l=x
    T,q=y[1],y[2]
    hf,_=_feed_state(feed_flow,feed_enthalpy,feed_entropy)
    v=iapws95_state(eos;temperature=T,density=M/model.vessel.volume)
    a1=iapws95_state(eos;temperature=y[5],density=y[4])
    ao=iapws95_state(eos;temperature=y[9],density=y[8])
    s=S/M
    mu=v.enthalpy-T*s
    speed=l/model.inertia
    heat_power=heat_conductance*(heat_source_temperature-T)
    work=q*(a1.enthalpy-ao.enthalpy)
    dx=[heat_power/T+feed_flow*(hf-mu)/T-q*s,
        feed_flow-q,
        (1-model.mechanical_loss_fraction)*work/speed-load_torque-model.friction*speed]
    (;dx,heat_power,turbine_power=work,exhaust_enthalpy=ao.enthalpy,exhaust_entropy=ao.entropy)
end

"""
    steam_constraint_jacobian(model, x, y; admission)

Analytic `dg/dy` and `dg/dx` assembled from the chart's property partials. In the
order of `STEAM_ALGEBRAIC_NAMES`, `dg/dy` is block lower triangular with diagonal
blocks of sizes 1,2,2,2,2. `blocks` holds their determinants,

    cv/T,  -(a*Cv+Ct),  p_rho*cp,  p_rho*cp/T,  p_rho*cp,

evaluated at the vessel, the flow closure, the valve outlet, the isentropic
endpoint and the exhaust. The scaled matrix divides rows by
`(R, q_ref, q_ref, p1, R*T, pc, R, pc, R*T)` and multiplies columns by
`(T, q_ref, p, rho1, T1, rho_i, T_i, rho_o, T_o)` with `q_ref=Ct*p`. Singular
values and the condition number refer to that declared scaling only.
"""
function steam_constraint_jacobian(model::SteamIAPWSComposition,x,y;admission)
    _composition_lengths(x,y)
    eos,V=model.vessel.eos,model.vessel.volume
    Cv,Ct,pc,eta=model.flow_law.valve,model.flow_law.turbine,model.exhaust_pressure,model.efficiency
    S,M=x[1],x[2]
    T,q,p1,rho1,T1,rhoi,Ti,rhoo,To=y
    v=iapws95_state(eos;temperature=T,density=M/V)
    a1=iapws95_state(eos;temperature=T1,density=rho1)
    ai=iapws95_state(eos;temperature=Ti,density=rhoi)
    ao=iapws95_state(eos;temperature=To,density=rhoo)
    J=zeros(typeof(v.pressure),9,9)
    J[1,1]=v.entropy_temperature
    J[2,1]=-admission*Cv*v.pressure_temperature; J[2,2]=1; J[2,3]=admission*Cv
    J[3,2]=1; J[3,3]=-Ct
    J[4,3]=-1; J[4,4]=a1.pressure_density; J[4,5]=a1.pressure_temperature
    J[5,1]=-v.enthalpy_temperature; J[5,4]=a1.enthalpy_density; J[5,5]=a1.enthalpy_temperature
    J[6,6]=ai.pressure_density; J[6,7]=ai.pressure_temperature
    J[7,4]=-a1.entropy_density; J[7,5]=-a1.entropy_temperature
    J[7,6]=ai.entropy_density; J[7,7]=ai.entropy_temperature
    J[8,8]=ao.pressure_density; J[8,9]=ao.pressure_temperature
    J[9,4]=-(1-eta)*a1.enthalpy_density; J[9,5]=-(1-eta)*a1.enthalpy_temperature
    J[9,6]=-eta*ai.enthalpy_density; J[9,7]=-eta*ai.enthalpy_temperature
    J[9,8]=ao.enthalpy_density; J[9,9]=ao.enthalpy_temperature
    Jx=zeros(eltype(J),9,3)
    Jx[1,1]=-1/M; Jx[1,2]=v.entropy_density/V+S/M^2
    Jx[2,2]=-admission*Cv*v.pressure_density/V
    Jx[5,2]=-v.enthalpy_density/V
    blocks=(vessel=v.cv/T,flow=-(admission*Cv+Ct),
        valve_outlet=a1.pressure_density*a1.cp,
        isentropic=ai.pressure_density*ai.cp/Ti,
        exhaust=ao.pressure_density*ao.cp)
    qref=Ct*v.pressure
    row_scales=[IAPWS95_R,qref,qref,p1,IAPWS95_R*T,pc,IAPWS95_R,pc,IAPWS95_R*T]
    column_scales=[T,qref,v.pressure,rho1,T1,rhoi,Ti,rhoo,To]
    scaled=Diagonal(inv.(row_scales))*J*Diagonal(column_scales)
    block(range)=det(scaled[range,range])
    scaled_blocks=(vessel=scaled[1,1],flow=block(2:3),valve_outlet=block(4:5),
        isentropic=block(6:7),exhaust=block(8:9))
    singular_values=svdvals(scaled)
    (;jacobian=J,state_jacobian=Jx,blocks,scaled_blocks,row_scales,column_scales,scaled,
      singular_values,sigma_min=minimum(singular_values),sigma_max=maximum(singular_values),
      condition=maximum(singular_values)/minimum(singular_values))
end

# Margins are (name, class, value, scale, strict). A value below -64 eps*scale is a
# violation; a negative value inside that band cannot be told from roundoff. Strict
# margins (mass, speed) are open conditions and are violated at zero.
const _STEAM_MARGIN_ULPS=64
_margin(name,class,value,scale;strict=false)=(;name,class,value=float(value),scale=float(abs(scale)),strict)
_margin_violated(m)=m.strict ? !(m.value>0) :
    !(m.value>= -_STEAM_MARGIN_ULPS*eps(Float64)*max(m.scale,floatmin(Float64)))
_margin_marginal(m)=m.value<0 && !_margin_violated(m)
_margin_on_face(m)=!m.strict && m.class!==:physical &&
    abs(m.value)<=_STEAM_MARGIN_ULPS*eps(Float64)*max(m.scale,floatmin(Float64))

function _face_margins!(entries,label,T,rho)
    push!(entries,_margin(Symbol(label,:_T_low),:validity,T-IAPWS95_T_BOUNDS[1],T))
    push!(entries,_margin(Symbol(label,:_T_high),:validity,IAPWS95_T_BOUNDS[2]-T,T))
    push!(entries,_margin(Symbol(label,:_rho_low),:validity,rho-IAPWS95_RHO_BOUNDS[1],rho))
    push!(entries,_margin(Symbol(label,:_rho_high),:validity,IAPWS95_RHO_BOUNDS[2]-rho,rho))
end

# Staged feasibility of one stream inversion: pressure image, then target image on
# the fixed-pressure interval, then the endpoint itself. A failed stage supplies no
# endpoint coordinates, so later margins are reported as unavailable, not guessed.
function _stream_margins!(entries,unavailable,failures,eos,label,pressure,kind,target)
    plow=iapws95_state(eos;temperature=IAPWS95_T_BOUNDS[1],density=IAPWS95_RHO_BOUNDS[1]).pressure
    phigh=iapws95_state(eos;temperature=IAPWS95_T_BOUNDS[2],density=IAPWS95_RHO_BOUNDS[2]).pressure
    low=_margin(Symbol(label,:_pressure_image_low),:validity,pressure-plow,pressure)
    high=_margin(Symbol(label,:_pressure_image_high),:validity,phigh-pressure,pressure)
    push!(entries,low,high)
    later=(Symbol(label,:_target_image),Symbol(label,:_faces))
    if low.value<0 || high.value<0
        append!(unavailable,later)
        return nothing
    end
    image(T)=begin
        st=iapws95_state(eos;temperature=T,density=_iapws95_density_at_pressure(eos,T,pressure))
        kind===:entropy ? st.entropy : st.enthalpy
    end
    try
        interval=_iapws95_pressure_interval(eos,pressure)
        tlow=_margin(Symbol(label,:_target_image_low),:validity,target-image(interval.lower),target)
        thigh=_margin(Symbol(label,:_target_image_high),:validity,image(interval.upper)-target,target)
        push!(entries,tlow,thigh)
        if tlow.value<0 || thigh.value<0
            push!(unavailable,later[2])
            return nothing
        end
        st=kind===:entropy ? iapws95_stream_state(eos;pressure,entropy=target) :
                             iapws95_stream_state(eos;pressure,enthalpy=target)
        _face_margins!(entries,label,st.temperature,st.density)
        st
    catch err
        err isa DomainError || err isa ErrorException || rethrow()
        push!(failures,(;stage=Symbol(label),message=sprint(showerror,err)))
        append!(unavailable,later)
        nothing
    end
end

"""
    steam_domain_margins(model, x; admission, heat_source_temperature, heat_conductance,
                         feed_flow, feed_enthalpy, feed_entropy)

Signed margins of the composed domain without relying on exception text. Classes:
`:validity` (faces and images of the restricted material chart, for the vessel,
valve-outlet, isentropic and exhaust states), `:branch` (limits of the declared
flow and torque laws: `p>=pc`, `Omega>Omega_min`) and `:physical` (mass, admission
range, nonnegative flow and entropy productions). A margin that needs a state
which could not be computed is listed in `unavailable`. `failures` lists numerical
errors raised where the image margins declared the inversion feasible.

`status` is `:excluded` (a margin is violated beyond 64 eps of its scale),
`:numerical_failure`, `:marginal` (a margin is negative within that band, so the
sign is not resolved), `:indeterminate` (a required margin is unavailable although
nothing computed is violated, for example a positive feed without its enthalpy and
entropy), `:boundary` (a validity or branch margin is zero within the band, so the
point lies on a face) or `:admissible` (every margin was computed and none is
violated or on a validity or branch face). Only `:admissible` and `:excluded`
predict what the guarded implementation does with a primal evaluation. Admission
0 and 1 are admissible; derivatives with respect to admission are not available
from the guarded code there.

Feed thermodynamic realisability is an external assumption: only the supplied
numbers and the mixing inequality are checked.
"""
function steam_domain_margins(model::SteamIAPWSComposition,x;admission,
        heat_source_temperature=model.ambient_temperature,heat_conductance=0.0,
        feed_flow=0.0,feed_enthalpy=nothing,feed_entropy=nothing)
    length(x)==3 || throw(DimensionMismatch("coupled state must be [S,M,l]"))
    eos,V,pc=model.vessel.eos,model.vessel.volume,model.exhaust_pressure
    S,M,l=float.(x)
    entries=NamedTuple[]; unavailable=Symbol[]; failures=NamedTuple[]
    finish()=begin
        violated=[m.name for m in entries if _margin_violated(m)]
        marginal=[m.name for m in entries if _margin_marginal(m)]
        on_face=[m.name for m in entries if _margin_on_face(m)]
        classes=unique([m.class for m in entries if _margin_violated(m)])
        status=!isempty(violated) ? :excluded : !isempty(failures) ? :numerical_failure :
               !isempty(marginal) ? :marginal : !isempty(unavailable) ? :indeterminate :
               !isempty(on_face) ? :boundary : :admissible
        (;status,classes,violated,marginal,on_face,unavailable,failures,entries)
    end
    speed=l/model.inertia
    push!(entries,_margin(:entropy_finite,:physical,isfinite(S) ? 1.0 : -Inf,1.0))
    push!(entries,_margin(:mass,:physical,isfinite(M) ? M : -Inf,1.0;strict=true))
    push!(entries,_margin(:admission_low,:physical,isfinite(admission) ? admission : -Inf,1.0))
    push!(entries,_margin(:admission_high,:physical,isfinite(admission) ? 1-admission : -Inf,1.0))
    push!(entries,_margin(:speed,:branch,isfinite(speed) ? speed-model.minimum_speed : -Inf,
        model.minimum_speed;strict=true))
    feed_given=!(isnothing(feed_enthalpy) || isnothing(feed_entropy))
    push!(entries,_margin(:feed_flow,:physical,feed_flow,1.0))
    push!(entries,_margin(:heat_conductance,:physical,heat_conductance,1.0))
    push!(entries,_margin(:heat_source_temperature,:physical,heat_source_temperature,1.0;strict=true))
    downstream=(:vessel_faces,:pressure_drop,:valve_outlet,:isentropic,:exhaust,:productions)
    if !(isfinite(M) && M>0 && isfinite(S))
        append!(unavailable,downstream)
        return finish()
    end
    rho,s=M/V,S/M
    push!(entries,_margin(:vessel_rho_low,:validity,rho-IAPWS95_RHO_BOUNDS[1],rho))
    push!(entries,_margin(:vessel_rho_high,:validity,IAPWS95_RHO_BOUNDS[2]-rho,rho))
    if !(IAPWS95_RHO_BOUNDS[1]<=rho<=IAPWS95_RHO_BOUNDS[2])
        append!(unavailable,downstream)
        return finish()
    end
    slow=iapws95_state(eos;temperature=IAPWS95_T_BOUNDS[1],density=rho).entropy
    shigh=iapws95_state(eos;temperature=IAPWS95_T_BOUNDS[2],density=rho).entropy
    push!(entries,_margin(:vessel_entropy_image_low,:validity,s-slow,s))
    push!(entries,_margin(:vessel_entropy_image_high,:validity,shigh-s,s))
    vessel=try
        slow<=s<=shigh ? iapws95_entropy_density_state(eos,s,rho) : nothing
    catch err
        err isa DomainError || err isa ErrorException || rethrow()
        push!(failures,(;stage=:vessel,message=sprint(showerror,err)))
        nothing
    end
    if isnothing(vessel)
        append!(unavailable,downstream)
        return finish()
    end
    T,p,h=vessel.temperature,vessel.pressure,vessel.enthalpy
    push!(entries,_margin(:vessel_T_low,:validity,T-IAPWS95_T_BOUNDS[1],T))
    push!(entries,_margin(:vessel_T_high,:validity,IAPWS95_T_BOUNDS[2]-T,T))
    push!(entries,_margin(:pressure_drop,:branch,p-pc,p))
    if feed_given
        mu=h-T*s
        push!(entries,_margin(:sigma_mix,:physical,feed_flow*((feed_enthalpy-mu)/T-feed_entropy),
            abs(feed_flow*s)))
    elseif feed_flow>0
        push!(unavailable,:sigma_mix)
    end
    if heat_source_temperature>0
        Q=heat_conductance*(heat_source_temperature-T)
        push!(entries,_margin(:sigma_heat,:physical,Q*(1/T-1/heat_source_temperature),abs(Q/T)))
    end
    if p<pc || !(0<=admission<=1) || !(speed>0)
        append!(unavailable,downstream[3:end])
        return finish()
    end
    closure=pressure_flow(model.flow_law,(;p);admission,exhaust_pressure=pc,speed)
    q,p1=closure.mass_flow,closure.inlet_pressure
    push!(entries,_margin(:mass_flow,:physical,q,model.flow_law.turbine*p))
    outlet=_stream_margins!(entries,unavailable,failures,eos,:valve_outlet,p1,:enthalpy,h)
    if isnothing(outlet)
        append!(unavailable,downstream[4:end])
        return finish()
    end
    push!(entries,_margin(:sigma_valve,:physical,q*(outlet.entropy-s),abs(q*s)))
    ideal=_stream_margins!(entries,unavailable,failures,eos,:isentropic,pc,:entropy,outlet.entropy)
    if isnothing(ideal)
        append!(unavailable,downstream[5:end])
        return finish()
    end
    hout=h-model.efficiency*(h-ideal.enthalpy)
    exhaust=_stream_margins!(entries,unavailable,failures,eos,:exhaust,pc,:enthalpy,hout)
    if isnothing(exhaust)
        push!(unavailable,:sigma_turbine)
        return finish()
    end
    push!(entries,_margin(:sigma_turbine,:physical,q*(exhaust.entropy-outlet.entropy),abs(q*s)))
    finish()
end
