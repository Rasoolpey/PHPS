using PHPS.PhysicalSteam

@testset "Standalone physical steam-resource primitives" begin
    eos = PerfectGasEOS(cv=1500.0, R=500.0)
    vessel = SteamVessel(eos; volume=2.0)
    x = perfect_gas_state(eos; mass=10.0, temperature=600.0, volume=2.0)
    st = vessel_state(vessel, x)
    s = 1500log(600/300)-500log(10/2)
    @test st.T ≈ 600.0
    @test st.p ≈ 1.5e6
    @test st.U ≈ 9e6
    @test st.s ≈ s
    @test st.h ≈ 1.2e6
    @test st.mu ≈ 1.2e6-600s
    @test vessel_energy(vessel,x) ≈ st.U
    @test ForwardDiff.gradient(z->internal_energy(eos,z...),[x[1],x[2],2.0]) ≈ [st.T,st.mu,-st.p]
    @test stream_entropy(eos,stream_enthalpy(eos,st.s,st.p),st.p) ≈ st.s
    @test stream_enthalpy(eos,st.s,st.p) ≈ st.h

    # Independently specified stream state; reject incompatible feed, not its residual.
    hfeed = 2000.0*700.0
    sfeed = 1500log(700/300)-500log(1.8e6/(500*700))
    rates = (heat_power=1000.0,feed_flow=0.1,feed_enthalpy=hfeed,feed_entropy=sfeed,outflow=0.2)
    dx = vessel_rhs!(zeros(2),vessel,x;rates...)
    @test dx[2] ≈ -0.1
    @test dot(ForwardDiff.gradient(z->vessel_energy(vessel,z),x),dx) ≈ 1000+0.1hfeed-0.2st.h
    b = vessel_balance(vessel,x;rates...)
    @test b.dx ≈ dx
    @test abs(b.energy_residual) < 1e-8
    @test abs(b.entropy_residual) < 1e-10
    @test b.sigma_mix > 0
    pt = vessel_ports(vessel,x;feed_enthalpy=hfeed)
    @test iszero(pt.J) && iszero(pt.R)
    @test pt.output ≈ [st.T,hfeed,-st.h]
    @test pt.G*[1000/st.T,0.1,0.2] ≈ dx
    @test dot(pt.output,[1000/st.T,0.1,0.2]) ≈ b.energy_rate
    @test b.heat_power+b.feed_power-b.discharge_power ≈ b.energy_rate
    @test all(isfinite,ForwardDiff.jacobian(z->vessel_rhs!(similar(z),vessel,z;rates...),x))
    low_pressure_entropy = stream_entropy(eos,st.h,st.p/2)
    @test_throws DomainError vessel_rhs!(zeros(2),vessel,x;feed_flow=0.1,
        feed_enthalpy=st.h,feed_entropy=low_pressure_entropy)

    @test_throws DomainError PerfectGasEOS(cv=-1,R=500)
    @test_throws DomainError PerfectGasEOS(cv=1500,R=NaN)
    @test_throws DomainError SteamVessel(eos;volume=0)
    @test_throws DomainError vessel_state(vessel,[0.0,0.0])
    @test_throws DomainError vessel_state(vessel,[Inf,1.0])
    @test_throws DimensionMismatch vessel_state(vessel,[1.0])
    @test_throws DimensionMismatch vessel_rhs!(zeros(3),vessel,x)
    @test_throws DomainError vessel_rhs!(zeros(2),vessel,x;outflow=-0.1)
    @test_throws DomainError vessel_rhs!(zeros(2),vessel,x;heat_power=NaN)
    @test_throws DomainError vessel_rhs!(zeros(2),vessel,x;entropy_tolerance=-1)
    @test_throws ArgumentError vessel_rhs!(zeros(2),vessel,x;feed_flow=0.1)
    @test_throws DomainError stream_entropy(eos,0.0,1e5)
    @test_throws DomainError stream_enthalpy(eos,1.0,0.0)

    heat = heat_exchange(st.T,800.0,100.0)
    cool = heat_exchange(st.T,400.0,100.0)
    @test heat.heat_power > 0 && heat.sigma_heat > 0
    @test cool.heat_power < 0 && cool.sigma_heat > 0
    @test_throws DomainError heat_exchange(st.T,0.0,100.0)
    @test_throws DomainError heat_exchange(st.T,800.0,-1.0)
    valve = throttle_stream(eos;flow=0.2,enthalpy=st.h,inlet_pressure=st.p,outlet_pressure=0.8st.p)
    @test valve.enthalpy == st.h
    @test valve.sigma_valve > 0
    @test_throws DomainError throttle_stream(eos;flow=0.2,enthalpy=st.h,inlet_pressure=st.p,outlet_pressure=1.2st.p)
    @test throttle_stream(eos;flow=0.0,enthalpy=st.h,inlet_pressure=st.p,outlet_pressure=1.2st.p).sigma_valve == 0

    for eta in (0.0,0.85,1.0), q in (0.0,0.2)
        t = turbine_stage(eos;flow=q,inlet_enthalpy=st.h,inlet_pressure=0.8st.p,
            exhaust_pressure=0.2st.p,efficiency=eta,mechanical_loss_fraction=0.1)
        @test all(isfinite,values(t))
        @test t.shaft_power+t.mechanical_loss+q*t.exhaust_enthalpy ≈ q*st.h
        @test t.sigma_turbine >= -1e-10
        @test t.power >= 0 && t.shaft_power >= 0
        q==0 && @test t.shaft_power == 0
    end
    @test_throws DomainError turbine_stage(eos;flow=0.2,inlet_enthalpy=st.h,
        inlet_pressure=st.p,exhaust_pressure=0.2st.p,efficiency=1.1)
    @test_throws DomainError turbine_stage(eos;flow=0.2,inlet_enthalpy=st.h,
        inlet_pressure=st.p,exhaust_pressure=1.2st.p,efficiency=0.8)
    degenerate = turbine_stage(eos;flow=0.2,inlet_enthalpy=st.h,inlet_pressure=st.p,
        exhaust_pressure=st.p,efficiency=1.0)
    @test degenerate.power == 0

    guard = (pressure=2e5,exhaust_pressure=1e5,admission=1.0,reserve_mass=1.0,mass_scale=2.0,nominal_flow=0.1)
    @test reserve_admission_flow(;mass=1.0,guard...) == 0
    @test reserve_admission_flow(;mass=0.5,guard...) == 0
    @test reserve_admission_flow(;mass=3.0,guard...) > 0
    @test reserve_admission_flow(;mass=3.0,guard...,admission=-1) == 0
    @test reserve_admission_flow(;mass=3.0,guard...,exhaust_pressure=2e5) == 0

    flowargs = (pressure=st.p,temperature=st.T,back_pressure=1e5,area=2e-4)
    q0 = choked_orifice_flow(eos;flowargs...)
    @test q0 > 0
    @test choked_orifice_flow(eos;flowargs...,area=4e-4) ≈ 2q0
    @test choked_orifice_flow(eos;flowargs...,temperature=4st.T) ≈ q0/2
    @test choked_orifice_flow(eos;flowargs...,admission=0) == 0
    @test_throws DomainError choked_orifice_flow(eos;flowargs...,back_pressure=0.9st.p)
    @test_throws DomainError choked_orifice_flow(eos;flowargs...,area=-1.0)
end
