using PHPS.PhysicalSteam

@testset "Shifted energy of the steam vessel about a steady point" begin
    eos=IAPWS95EOS()
    V=600.0; J=18292.7; wb=2pi*60
    # Synthetic SI fixtures; a pressurised cooler feed and a hotter heat source.
    function steady(T0,rho0,admission,ratio)
        chart=iapws95_state(eos;temperature=T0,density=rho0)
        model=SteamTurbineShaft(SteamVessel(eos;volume=V),LinearSteamConductance(valve=3e-4,turbine=3e-4);
            inertia=J,exhaust_pressure=ratio*chart.pressure,ambient_temperature=300.0,minimum_speed=0.1wb,
            efficiency=0.85,mechanical_loss_fraction=0.05,friction=0.0)
        x=[rho0*V*chart.entropy,rho0*V,J*wb]
        free=steam_balance(model,x;admission)
        q=free.closure.mass_flow
        feed=iapws95_state(eos;temperature=T0-100.0,density=1.4rho0)
        @assert feed.pressure>chart.pressure
        inputs=(admission=admission,heat_source_temperature=T0+200.0,
            heat_conductance=q*(free.state.vessel.h-feed.enthalpy)/200.0,feed_flow=q,
            feed_enthalpy=feed.enthalpy,feed_entropy=feed.entropy)
        model,x,chart,inputs,q,free
    end
    function hessian(chart,M)
        T=chart.temperature; b=chart.pressure_temperature/chart.density-chart.entropy
        [T/(M*chart.cv) T*b/(M*chart.cv); T*b/(M*chart.cv) (chart.pressure_density+T/chart.cv*b^2)/M]
    end
    for (T0,rho0,admission,ratio) in ((1100.0,50.0,0.7,0.5),(900.0,5.0,0.4,0.7))
        model,x,chart,inputs,q,free=steady(T0,rho0,admission,ratio)
        z0=x[1:2]
        # The shift vanishes at the steady point, with zero effort differences and zero rate.
        at=steam_shifted_rate(model,x,x;inputs...)
        @test at.shift==0 && at.rate==0 && all(iszero,at.effort_difference)
        @test abs(at.reference.dx[1])<=1e-10*q*abs(chart.entropy) && abs(at.reference.dx[2])<=1e-10*q
        @test vessel_shifted_energy(model.vessel,z0,z0).shift==0

        point=[x[1]*1.004,x[2]*0.997,x[3]]
        r=steam_shifted_rate(model,point,x;inputs...)
        shifted=vessel_shifted_energy(model.vessel,point[1:2],z0)
        @test r.shift ≈ shifted.shift rtol=1e-12
        @test r.shift>0
        @test r.effort_difference ≈ shifted.effort_difference rtol=1e-12
        # The rate is the effort difference paired with the state rate, and it splits by port.
        balance=steam_balance(model,point;inputs...)
        @test r.rate ≈ dot(r.effort_difference,balance.dx[1:2]) rtol=1e-12
        @test r.rate ≈ sum(values(r.contributions))+r.reference_term rtol=1e-9
        @test abs(r.reference_term)<=1e-9*sum(abs,values(r.contributions))
        @test r.contributions.heat ≈ r.heat_closed_form rtol=1e-9
        @test r.contributions.heat<=0
        # Independent route: the rate as a directional derivative of the shift along the flow.
        step=1e-4
        ahead=vessel_shifted_energy(model.vessel,point[1:2].+step.*balance.dx[1:2],z0).shift
        behind=vessel_shifted_energy(model.vessel,point[1:2].-step.*balance.dx[1:2],z0).shift
        @test (ahead-behind)/(2step) ≈ r.rate rtol=1e-6

        # Local form: closed form in (dT, dp) against Hess(U)*A in storage coordinates.
        rates(z)=steam_balance(model,[z[1],z[2],x[3]];inputs...).dx[1:2]
        steps=1e-6 .*abs.(z0)
        A=hcat([(rates(z0.+steps[j].*(1:2 .==j)).-rates(z0.-steps[j].*(1:2 .==j)))./(2steps[j]) for j in 1:2]...)
        P=hessian(chart,z0[2])
        Q=(P*A+A'*P)/2
        form=steam_local_rate_form(model,x;admission,heat_source_temperature=inputs.heat_source_temperature,
            heat_conductance=inputs.heat_conductance)
        closed=-form.coordinate_jacobian'*form.matrix*form.coordinate_jacobian
        @test maximum(abs,Q.-closed)<=1e-5*maximum(abs,Q)
        @test isposdef(Symmetric(P)) && isposdef(Symmetric(form.matrix))
        @test form.margin>0 && form.state_margin>0
        @test form.margin ≈ 4det(form.matrix) rtol=1e-12
        @test form.mass_flow ≈ q rtol=1e-12
        @test form.conductance ≈ q/(chart.pressure-model.exhaust_pressure) rtol=1e-9
        # The throttling coefficient against a finite difference of h at fixed T.
        up=iapws95_state(eos;temperature=T0,density=rho0*(1+1e-6)); down=iapws95_state(eos;temperature=T0,density=rho0*(1-1e-6))
        @test form.throttling ≈ (up.enthalpy-down.enthalpy)/(up.pressure-down.pressure) rtol=1e-6
        # Coordinate Jacobian d(T,p)/d(S,M) against differences of the vessel state.
        Tp(z)=(st=thermodynamic_state(eos,z[1],z[2],V); [st.T,st.p])
        Jfd=hcat([(Tp(z0.+steps[j].*(1:2 .==j)).-Tp(z0.-steps[j].*(1:2 .==j)))./(2steps[j]) for j in 1:2]...)
        @test all(isapprox.(form.coordinate_jacobian,Jfd;rtol=1e-6))
        # Small deviations decay and agree with the quadratic form.
        for angle in (0.3,1.7,3.9,5.2)
            d=1e-4 .*[cos(angle),sin(angle)].*abs.(z0)
            near=steam_shifted_rate(model,[x[1]+d[1],x[2]+d[2],x[3]],x;inputs...)
            @test near.shift>0 && near.rate<0
            @test near.rate ≈ dot(d,Q*d) rtol=1e-2
        end
    end
    # A state margin of the vessel alone: positive at sampled chart states.
    PS=PHPS.PhysicalSteam
    @test all(PS.iapws95_shift_state_margin(iapws95_state(eos;temperature=T,density=rho))>0
              for T in (700.0,900.0,1200.0), rho in (0.001,1.0,100.0))
    # The local form is declared for the IAPWS-95 provider only.
    gas=SteamTurbineShaft(SteamVessel(PerfectGasEOS(cv=1500.0,R=500.0);volume=2.0),
        LinearSteamConductance(valve=2e-7,turbine=3e-7);inertia=20.0,exhaust_pressure=225000.0,
        ambient_temperature=300.0,minimum_speed=10.0,efficiency=0.85,mechanical_loss_fraction=0.05,friction=0.0)
    xg=vcat(perfect_gas_state(gas.vessel.eos;mass=10.0,temperature=600.0,volume=2.0),7000.0)
    @test_throws MethodError steam_local_rate_form(gas,xg;admission=0.7)
    # The shifted rate itself works for any provider.
    rg=steam_shifted_rate(gas,[xg[1]*1.001,xg[2],xg[3]],xg;admission=0.7)
    @test rg.rate ≈ sum(values(rg.contributions))+rg.reference_term rtol=1e-9
end
