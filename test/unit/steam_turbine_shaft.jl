using PHPS.PhysicalSteam

@testset "Coupled physical resource turbine shaft" begin
    eos=PerfectGasEOS(cv=1500.0,R=500.0)
    vessel=SteamVessel(eos;volume=2.0)
    law=LinearSteamConductance(valve=2e-7,turbine=3e-7)
    base=(inertia=20.0,exhaust_pressure=225000.0,ambient_temperature=300.0,
        minimum_speed=10.0,efficiency=0.85,mechanical_loss_fraction=0.05,friction=0.0)
    make_model(;kwargs...)=SteamTurbineShaft(vessel,law;base...,kwargs...)
    model=make_model()
    x=vcat(perfect_gas_state(eos;mass=10.0,temperature=600.0,volume=2.0),7000.0)
    st=PhysicalSteam.steam_state(model,x)
    @test st.speed ≈ 350.0
    @test st.kinetic_energy ≈ 1.225e6
    @test steam_energy(model,x) ≈ 10.225e6
    @test ForwardDiff.gradient(z->steam_energy(model,z),x) ≈ [st.vessel.T,st.vessel.mu,350.0]
    b=steam_balance(model,x;admission=0.7)
    @test 225000 < b.closure.inlet_pressure < st.vessel.p
    @test b.closure.mass_flow ≈ 0.7*law.valve*(st.vessel.p-b.closure.inlet_pressure)
    @test b.closure.mass_flow ≈ law.turbine*(b.closure.inlet_pressure-225000)
    @test b.torque*st.speed ≈ b.turbine.shaft_power
    @test dot(ForwardDiff.gradient(z->steam_energy(model,z),x),b.dx) ≈ b.energy_supply
    @test abs(b.energy_residual) < 1e-8
    @test abs(b.entropy_residual) < 1e-10
    @test abs(b.availability_residual) < 1e-8
    @test minimum(values(b.entropy_productions)) >= 0
    @test b.graph.B == -transpose(b.graph.B)
    @test b.graph.flows ≈ [b.closure.mass_flow,-b.closure.mass_flow,
        -b.torque,-b.turbine.mechanical_loss/model.ambient_temperature]
    @test abs(sum(b.graph.powers)) < 1e-8
    @test maximum(abs,values(b.graph_residuals)) < 1e-8
    @test b.valve.inlet_entropy == st.vessel.s
    @test abs(b.valve.entropy_state_residual) <= 1e-8
    @test_throws DomainError throttle_stream(eos;flow=0.1,enthalpy=st.vessel.h,
        inlet_pressure=st.vessel.p,outlet_pressure=225000.0,inlet_entropy=st.vessel.s+1.0)
    @test steam_rhs!(zeros(3),model,x;admission=0.7) ≈ b.dx
    @test all(isfinite,ForwardDiff.jacobian(z->steam_rhs!(similar(z),model,z;admission=0.7),x))

    loaded=make_model(;inertia=20.0,exhaust_pressure=225000.0,friction=0.08)
    hf=1.1st.vessel.h
    sf=stream_entropy(eos,hf,1.2st.vessel.p)
    for source_T in (400.0,800.0)
        r=steam_balance(loaded,x;admission=0.7,heat_source_temperature=source_T,
            heat_conductance=100.0,feed_flow=0.04,feed_enthalpy=hf,feed_entropy=sf,
            load_torque=100.0,ambient_chemical_potential=12345.0)
        @test r.power.friction_heat_out ≈ 0.08*350^2
        @test r.power.load_out ≈ 35000.0
        @test abs(r.energy_residual)<1e-8
        @test abs(r.entropy_residual)<1e-10
        @test abs(r.availability_residual)<1e-8
        @test minimum(values(r.entropy_productions))>=0
        @test dot(ForwardDiff.gradient(z->steam_energy(loaded,z)-300z[1]-12345z[2],x),r.dx) ≈ r.availability_supply
    end
    closed=steam_balance(model,x;admission=0.0)
    @test closed.closure.inlet_pressure == model.exhaust_pressure
    @test closed.closure.mass_flow == 0
    @test iszero(closed.dx)
    @test iszero(closed.graph.B)
    equal_pressure=make_model(;inertia=20.0,exhaust_pressure=st.vessel.p)
    @test iszero(steam_balance(equal_pressure,x;admission=0.5).dx)
    for eta in (0.0,1.0), chi in (0.0,1.0)
        edge=make_model(;inertia=20.0,exhaust_pressure=225000.0,
            efficiency=eta,mechanical_loss_fraction=chi)
        r=steam_balance(edge,x;admission=1.0,entropy_tolerance=1e-10)
        @test minimum(values(r.entropy_productions))>=-1e-10
        @test abs(r.energy_residual)<1e-8
        @test r.torque>=0
        (eta==0 || chi==1) && @test r.torque==0
    end
    @test_throws DomainError LinearSteamConductance(valve=0,turbine=1)
    @test_throws DomainError LinearSteamConductance(valve=1,turbine=Inf)
    overflow_law=LinearSteamConductance(valve=1e308,turbine=1e308)
    @test_throws DomainError pressure_flow(overflow_law,st.vessel;admission=1.0,
        exhaust_pressure=225000.0,speed=350.0)
    @test_throws DomainError make_model(;inertia=0,exhaust_pressure=1e5)
    @test_throws DomainError make_model(;inertia=20,exhaust_pressure=1e5,efficiency=1.1)
    @test_throws DomainError make_model(;inertia=20,exhaust_pressure=1e5,friction=-1)
    @test_throws DomainError make_model(;inertia=20,exhaust_pressure=1e5,mechanical_loss_fraction=-1)
    @test_throws DimensionMismatch PhysicalSteam.steam_state(model,x[1:2])
    @test_throws DimensionMismatch steam_rhs!(zeros(2),model,x;admission=0.7)
    @test_throws DomainError PhysicalSteam.steam_state(model,[x[1],x[2],0.0])
    @test_throws DomainError PhysicalSteam.steam_state(model,[x[1],x[2],200.0])
    reverse=make_model(;inertia=20,exhaust_pressure=2e6)
    @test_throws DomainError PhysicalSteam.steam_state(reverse,x)
    @test_throws DomainError steam_balance(model,x;admission=-0.1)
    @test_throws DomainError steam_balance(model,x;admission=1.1)
    @test_throws DomainError steam_balance(model,x;admission=0.7,heat_source_temperature=0)
    @test_throws DomainError steam_balance(model,x;admission=0.7,load_torque=NaN)
    @test_throws ArgumentError steam_balance(model,x;admission=0.7,feed_flow=0.1)
    @test_throws DomainError steam_balance(model,x;admission=0.7,feed_flow=0.1,
        feed_enthalpy=st.vessel.h,feed_entropy=stream_entropy(eos,st.vessel.h,st.vessel.p/2))
end
