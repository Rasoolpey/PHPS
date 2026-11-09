# Shifted (Bregman) energy of the steam vessel about a steady point, and its exact
# rate by port. Derivation, scope and evidence: docs/active/STEAM_SHIFTED_ENERGY_BALANCE.md.
#
# W(x) = U(x) - U(x*) - T*(S-S*) - mu*(M-M*),   W' = (T-T*) S' + (mu-mu*) M'.
# The identity for W' holds at every x. With x* steady for the same boundary inputs it
# splits into a heat, a feed and a discharge contribution; nothing here assumes a sign.

"""
    vessel_shifted_energy(vessel, x, xstar)

Bregman shift of the vessel energy `U(S,M;V)` about `xstar = (S*, M*)`, in J, with the
effort differences `(T-T*, mu-mu*)`. Positive near `xstar` where the fixed-volume
Hessian of `U` is positive definite, that is where `cv > 0` and `p_rho > 0`.
"""
function vessel_shifted_energy(v::SteamVessel,x,xstar)
    state,reference=vessel_state(v,x),vessel_state(v,xstar)
    shift=state.U-reference.U-reference.T*(x[1]-xstar[1])-reference.mu*(x[2]-xstar[2])
    (;shift,effort_difference=[state.T-reference.T,state.mu-reference.mu],state,reference)
end

"""
    steam_shifted_rate(model, x, xstar; admission, heat and feed keywords...)

Exact rate of the vessel's shifted energy at coupled state `x = [S, M, l]` about
`xstar`, for the boundary inputs given: `rate = (T-T*)*S' + (mu-mu*)*M'`.

`contributions` splits it by port, each as the effort difference times the difference of
that port's entropy and mass flows between `x` and `xstar`:

    heat       (T-T*)*(Q/T - Q*/T*)                  = -k*Tf*(T-T*)^2/(T*T*)
    feed       (T-T*)*qf*((hf-mu)/T - (hf-mu*)/T*)
    discharge  -(T-T*)*(q*s - q**s*) - (mu-mu*)*(q - q*)

The heat term has the closed form shown and is never positive. The other two have no
definite sign. `reference_term = (T-T*)*S'(x*) + (mu-mu*)*M'(x*)` vanishes when `xstar`
is steady for these inputs, and `rate = heat + feed + discharge + reference_term`.
"""
function steam_shifted_rate(model::SteamTurbineShaft,x,xstar;admission,
        heat_source_temperature=model.ambient_temperature,heat_conductance=0.0,
        feed_flow=0.0,feed_enthalpy=nothing,feed_entropy=nothing,entropy_tolerance=0.0)
    inputs=(;admission,heat_source_temperature,heat_conductance,feed_flow,feed_enthalpy,
        feed_entropy,entropy_tolerance)
    current=steam_balance(model,x;inputs...)
    reference=steam_balance(model,xstar;inputs...)
    hf,_=_feed_state(feed_flow,feed_enthalpy,feed_entropy)
    v,r=current.state.vessel,reference.state.vessel
    dT,dmu=v.T-r.T,v.mu-r.mu
    q,qr=current.closure.mass_flow,reference.closure.mass_flow
    shift=v.U-r.U-r.T*(x[1]-xstar[1])-r.mu*(x[2]-xstar[2])
    heat=dT*(current.heat.heat_power/v.T-reference.heat.heat_power/r.T)
    feed=dT*feed_flow*((hf-v.mu)/v.T-(hf-r.mu)/r.T)
    discharge=-dT*(q*v.s-qr*r.s)-dmu*(q-qr)
    reference_term=dT*reference.dx[1]+dmu*reference.dx[2]
    rate=dT*current.dx[1]+dmu*current.dx[2]
    (;shift,rate,contributions=(;heat,feed,discharge),reference_term,
      heat_closed_form=-heat_conductance*heat_source_temperature*dT^2/(v.T*r.T),
      effort_difference=[dT,dmu],balance=current,reference)
end

"""
    steam_local_rate_form(model, xstar; admission, heat_source_temperature, heat_conductance)

Second-order part of the shifted rate at a steady point of the IAPWS-95 vessel with the
linear conductance law, in the deviations `(dT, dp)` of vessel temperature and pressure:

    W'_2 = -[(k + q*cp)/T]*dT^2 - (q*theta/T)*dT*dp - kappa*v*dp^2,

with `q` the steady mass flow, `kappa = a*Cv*Ct/(a*Cv+Ct)` the discharge conductance,
`v = 1/rho` and `theta = (dh/dp)_T = v*(1 - beta*T)` the isothermal throttling
coefficient. It uses the steady conditions `qf = q` and `Q = q*(h - hf)`, a fixed exhaust
pressure and fixed admission. `matrix` is the symmetric `A` with `W'_2 = -d'*A*d`;
the rate form is negative definite exactly when `margin = 4*kappa*v*(k+q*cp)/T -
(q*theta/T)^2 > 0`. `state_margin = 4*cp*T - p*v*(1-beta*T)^2` is a property of the
vessel state alone: where it is positive, `margin > 0` for every `k >= 0`, every exhaust
pressure with `0 <= pc < p` and every positive conductance.
"""
function steam_local_rate_form(model::SteamTurbineShaft{<:SteamVessel{IAPWS95EOS},<:LinearSteamConductance},
        xstar;admission,heat_source_temperature=model.ambient_temperature,heat_conductance=0.0)
    state=steam_state(model,xstar)
    closure=pressure_flow(model.flow_law,state.vessel;admission,exhaust_pressure=model.exhaust_pressure,speed=state.speed)
    chart=iapws95_state(model.vessel.eos;temperature=state.vessel.T,density=xstar[2]/model.vessel.volume)
    law=model.flow_law
    kappa=admission*law.valve*law.turbine/(admission*law.valve+law.turbine)
    q,T,v=closure.mass_flow,chart.temperature,1/chart.density
    theta=chart.enthalpy_density/chart.pressure_density
    thermal=(heat_conductance+q*chart.cp)/T
    cross=q*theta/T
    matrix=[thermal cross/2; cross/2 kappa*v]
    (;matrix,thermal,cross,pressure=kappa*v,margin=4*kappa*v*thermal-cross^2,
      state_margin=iapws95_shift_state_margin(chart),conductance=kappa,mass_flow=q,
      throttling=theta,expansivity_product=1-theta/v,
      # d(T,p)/d(S,M) at fixed volume, for moving the form to storage coordinates.
      coordinate_jacobian=_vessel_Tp_jacobian(chart,xstar[2],model.vessel.volume))
end

"""`4*cp*T - p*v*(1-beta*T)^2` at a chart state, in J/kg; see `steam_local_rate_form`."""
iapws95_shift_state_margin(chart)=4*chart.cp*chart.temperature-
    chart.pressure/chart.density*(chart.density*chart.enthalpy_density/chart.pressure_density)^2

# Jacobian of (T, p) with respect to (S, M) at fixed volume, from the chart partials.
function _vessel_Tp_jacobian(chart,M,V)
    T,rho=chart.temperature,chart.density
    b=chart.pressure_temperature/rho-chart.entropy
    T_S=T/(M*chart.cv); T_M=T*b/(M*chart.cv)
    [T_S T_M; chart.pressure_temperature*T_S chart.pressure_temperature*T_M+chart.pressure_density/V]
end
