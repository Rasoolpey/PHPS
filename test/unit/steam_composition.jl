using PHPS.PhysicalSteam

@testset "IAPWS95 composed vessel-valve-turbine-rotor constraint map" begin
    eos=IAPWS95EOS()
    # Synthetic SI fixtures inside the restricted chart; not plant calibration.
    function fixture(;T0=1100.0,rho0=50.0,V=2.0,ratio=0.5,efficiency=0.85)
        # `local`: the testset body reuses these names, and a closure would rebind them.
        local chart=iapws95_state(eos;temperature=T0,density=rho0)
        local shaft=SteamTurbineShaft(SteamVessel(eos;volume=V),
            LinearSteamConductance(valve=1e-6,turbine=1e-6);inertia=20.0,
            exhaust_pressure=ratio*chart.pressure,ambient_temperature=300.0,minimum_speed=10.0,
            efficiency,mechanical_loss_fraction=0.05,friction=0.02)
        shaft,[rho0*V*chart.entropy,rho0*V,20.0*350.0],chart
    end
    @test length(STEAM_ALGEBRAIC_NAMES)==9 && length(STEAM_CONSTRAINT_NAMES)==9

    for (kwargs,a) in (((;),0.7),((;T0=900.0,rho0=20.0,ratio=0.6),0.3),
                       ((;T0=1150.0,rho0=5.0,ratio=0.8),1.0),((;efficiency=0.0),0.7),
                       ((;),0.0))
        model,x,_=fixture(;kwargs...)
        sol=steam_algebraic_state(model,x;admission=a)
        y=sol.y
        jac=steam_constraint_jacobian(model,x,y;admission=a)
        g=steam_constraint_residual(model,x,y;admission=a)
        @test maximum(abs,g./jac.row_scales)<=1e-11
        # Analytic Jacobians against AD of the root-free map, including sparsity.
        Jy=ForwardDiff.jacobian(z->steam_constraint_residual(model,x,z;admission=a),y)
        Jx=ForwardDiff.jacobian(z->steam_constraint_residual(model,z,y;admission=a),x)
        @test (Jy.==0)==(jac.jacobian.==0)
        @test all(isapprox.(jac.jacobian,Jy;rtol=1e-11))
        @test (Jx.==0)==(jac.state_jacobian.==0)
        @test all(isapprox.(jac.state_jacobian,Jx;rtol=1e-11))
        # Block lower triangular: nothing above the diagonal blocks 1,2,2,2,2.
        starts=(1,2,4,6,8); stops=(1,3,5,7,9)
        @test all(iszero(jac.jacobian[i,j]) for k in 1:5 for i in starts[k]:stops[k] for j in stops[k]+1:9)
        @test det(jac.jacobian) ≈ prod(values(jac.blocks)) rtol=1e-10
        @test jac.blocks.vessel>0 && jac.blocks.flow<0
        @test jac.blocks.valve_outlet>0 && jac.blocks.isentropic>0 && jac.blocks.exhaust>0
        @test det(jac.jacobian[4:5,4:5]) ≈ jac.blocks.valve_outlet rtol=1e-11
        @test det(jac.jacobian[6:7,6:7]) ≈ jac.blocks.isentropic rtol=1e-11
        @test det(jac.jacobian[8:9,8:9]) ≈ jac.blocks.exhaust rtol=1e-11
        @test jac.sigma_min>0.05 && jac.condition<1e3
        @test jac.condition ≈ cond(jac.scaled) rtol=1e-10
        # Forward-chart dynamics agree with the implemented right-hand side.
        f=steam_constraint_dynamics(model,x,y;load_torque=100.0)
        b=steam_balance(model,x;admission=a,load_torque=100.0)
        @test f.dx ≈ b.dx rtol=1e-11 atol=1e-9
        m=steam_domain_margins(model,x;admission=a)
        @test m.status===:admissible && isempty(m.unavailable) && isempty(m.failures)
        # Conditional orderings of the stream states (hold under H1-H3).
        @test y[4]<=x[2]/model.vessel.volume*(1+1e-12)
        @test y[6]<=y[4]*(1+1e-12) && y[7]<=y[5]*(1+1e-12)
        @test y[7]<=y[9]*(1+1e-12) && y[8]<=y[6]*(1+1e-12)
    end

    # Implicit-function derivative of the root-free map against AD through the
    # implemented nested solves and their Dual lifts, in an interior neighbourhood.
    model,x,_=fixture()
    y=steam_algebraic_state(model,x;admission=0.7).y
    Jy=ForwardDiff.jacobian(z->steam_constraint_residual(model,x,z;admission=0.7),y)
    Jx=ForwardDiff.jacobian(z->steam_constraint_residual(model,z,y;admission=0.7),x)
    lifted=ForwardDiff.jacobian(z->steam_algebraic_state(model,z;admission=0.7).y,x)
    implicit=-(Jy\Jx)
    @test all(isapprox.(lifted[:,1:2],implicit[:,1:2];rtol=1e-9))
    @test all(iszero,lifted[:,3]) && all(iszero,implicit[:,3])
    # Admission sensitivity through the explicit-parameter map.
    PS=PHPS.PhysicalSteam
    ga=ForwardDiff.derivative(a->steam_constraint_residual(model,x,y;admission=a),0.7)
    dyda=ForwardDiff.derivative(a->steam_algebraic_state(model,x;admission=a).y,0.7)
    @test all(isapprox.(dyda,-(Jy\ga);rtol=1e-9,atol=1e-12))
    gpc=ForwardDiff.derivative(pc->PS._steam_constraint(eos,model.vessel.volume,
        model.flow_law.valve,model.flow_law.turbine,pc,model.efficiency,x,y,0.7),model.exhaust_pressure)
    @test all(isfinite,Jy\gpc)

    # Domain margins classify exclusions without reading exception text.
    st=iapws95_state(eos;temperature=1100.0,density=50.0)
    excluded(m,class)=m.status===:excluded && class in m.classes
    cold=fixture(;ratio=0.02)[1]
    m=steam_domain_margins(cold,x;admission=0.7)
    @test excluded(m,:validity) && :isentropic_target_image_low in m.violated
    @test :isentropic_faces in m.unavailable
    @test_throws DomainError steam_balance(cold,x;admission=0.7)
    starved=SteamTurbineShaft(model.vessel,model.flow_law;inertia=20.0,exhaust_pressure=100.0,
        ambient_temperature=300.0,minimum_speed=10.0,efficiency=0.85,
        mechanical_loss_fraction=0.05,friction=0.02)
    m=steam_domain_margins(starved,x;admission=1e-9)
    @test excluded(m,:validity) && :valve_outlet_pressure_image_low in m.violated
    @test_throws DomainError steam_balance(starved,x;admission=1e-9)
    reverse_pressure=fixture(;ratio=1.1)[1]
    m=steam_domain_margins(reverse_pressure,x;admission=0.7)
    @test excluded(m,:branch) && m.violated==[:pressure_drop]
    @test_throws DomainError steam_balance(reverse_pressure,x;admission=0.7)
    slow=[x[1],x[2],20.0*10.0]
    m=steam_domain_margins(model,slow;admission=0.7)
    @test excluded(m,:branch) && m.violated==[:speed]
    @test_throws DomainError steam_balance(model,slow;admission=0.7)
    for a in (-0.1,1.2)
        m=steam_domain_margins(model,x;admission=a)
        @test excluded(m,:physical) && m.classes==[:physical]
        @test_throws DomainError steam_balance(model,x;admission=a)
    end
    dense=[x[1],101.0*2.0,x[3]]
    m=steam_domain_margins(model,dense;admission=0.7)
    @test excluded(m,:validity) && m.violated==[:vessel_rho_high] && :vessel_faces in m.unavailable
    @test_throws DomainError steam_balance(model,dense;admission=0.7)
    chilled=[x[2]*(iapws95_state(eos;temperature=700.0,density=50.0).entropy-10.0),x[2],x[3]]
    m=steam_domain_margins(model,chilled;admission=0.7)
    @test excluded(m,:validity) && m.violated==[:vessel_entropy_image_low]
    @test_throws DomainError steam_balance(model,chilled;admission=0.7)
    # Feed: the mixing inequality is checked; realisability of the feed state is not.
    bad=(feed_flow=0.5,feed_enthalpy=st.enthalpy,feed_entropy=st.entropy+100.0)
    m=steam_domain_margins(model,x;admission=0.7,bad...)
    @test excluded(m,:physical) && :sigma_mix in m.violated
    @test_throws DomainError steam_balance(model,x;admission=0.7,bad...)
    good=(feed_flow=0.5,feed_enthalpy=st.enthalpy,feed_entropy=st.entropy)
    @test steam_domain_margins(model,x;admission=0.7,good...).status===:admissible
    # A positive feed without its properties cannot be assessed; it is not admissible.
    m=steam_domain_margins(model,x;admission=0.7,feed_flow=0.5)
    @test m.status===:indeterminate && :sigma_mix in m.unavailable
    @test_throws ArgumentError steam_balance(model,x;admission=0.7,feed_flow=0.5)
    m=steam_domain_margins(model,[NaN,x[2],x[3]];admission=0.7)
    @test m.status===:excluded && :entropy_finite in m.violated
    m=steam_domain_margins(model,x;admission=0.7,heat_source_temperature=0.0)
    @test m.status===:excluded && :heat_source_temperature in m.violated
    @test_throws DomainError steam_balance(model,x;admission=0.7,heat_source_temperature=0.0)
    # Forward-chart dynamics with heat, feed and load all active.
    feed=iapws95_state(eos;temperature=1150.0,density=60.0)
    driven=(heat_source_temperature=1300.0,heat_conductance=2e4,feed_flow=3.0,
        feed_enthalpy=feed.enthalpy,feed_entropy=feed.entropy,load_torque=100.0)
    @test steam_constraint_dynamics(model,x,y;driven...).dx ≈
          steam_balance(model,x;admission=0.7,driven...).dx rtol=1e-11
    @test steam_domain_margins(model,[x[1],0.0,x[3]];admission=0.7).status===:excluded
    # A point on a validity face is reported as such; its outcome is not predicted.
    corner,xc,_=fixture(;T0=700.0,rho0=0.001,ratio=1.0)
    m=steam_domain_margins(corner,xc;admission=0.0)
    @test m.status in (:boundary,:marginal) && isempty(m.violated)
    @test :vessel_rho_low in m.on_face
    @test_throws DimensionMismatch steam_constraint_residual(model,x,y[1:8];admission=0.7)
    # Declared scope: the composition methods exist for the IAPWS-95 provider only.
    gas=SteamTurbineShaft(SteamVessel(PerfectGasEOS(cv=1500.0,R=500.0);volume=2.0),model.flow_law;
        inertia=20.0,exhaust_pressure=225000.0,ambient_temperature=300.0,minimum_speed=10.0,
        efficiency=0.85,mechanical_loss_fraction=0.05,friction=0.0)
    @test_throws MethodError steam_algebraic_state(gas,x;admission=0.7)
end
