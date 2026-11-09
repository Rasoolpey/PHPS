using PHPS.PhysicalSteam

@testset "Restricted IAPWS95 constitutive chart" begin
    eos=IAPWS95EOS()
    for (T,rho) in ((725.0,0.01),(900.0,52.615),(1150.0,95.0),
                    (700.0,0.001),(1200.0,100.0))
        st=iapws95_state(eos;temperature=T,density=rho)
        inverse=iapws95_entropy_density_state(eos,st.entropy,rho)
        @test inverse.temperature ≈ T rtol=1e-13
        @test abs(inverse.entropy_residual)<=1e-10
        M=2.0; V=M/rho; S=M*st.entropy
        thermo=thermodynamic_state(eos,S,M,V)
        @test thermo.T ≈ T rtol=1e-12
        @test thermo.p ≈ st.pressure rtol=1e-12
        @test thermo.mu ≈ st.chemical_potential rtol=1e-12
        gradient=ForwardDiff.gradient([st.entropy,rho]) do z
            iapws95_entropy_density_state(eos,z...).temperature
        end
        @test gradient ≈ [T/st.cv,-st.entropy_density*T/st.cv] rtol=1e-12
        # Fixed-volume Hessian follows from du=T ds+p/rho^2 d rho.
        b=st.pressure_temperature/rho-st.entropy
        expected=[T/st.cv T/st.cv*b;
            T/st.cv*b st.pressure_density+T/st.cv*b^2]/M
        hessian=ForwardDiff.hessian(z->internal_energy(eos,z[1],z[2],V),[S,M])
        @test hessian ≈ expected rtol=1e-12
        @test det(hessian)>0
    end
    st=iapws95_state(eos;temperature=900.0,density=52.615)
    vessel=SteamVessel(eos;volume=2/52.615)
    x=[2st.entropy,2.0]
    ports=vessel_ports(vessel,x;feed_enthalpy=st.enthalpy)
    @test ports.output ≈ [900.0,st.enthalpy,-st.enthalpy] rtol=1e-12
    for (T,rho) in ((699.0,1.0),(1201.0,1.0),(900.0,0.0009),(900.0,101.0))
        @test_throws DomainError iapws95_state(eos;temperature=T,density=rho)
    end
    @test_throws DomainError iapws95_entropy_density_state(eos,st.entropy,0.0)
    @test_throws DomainError iapws95_entropy_density_state(eos,Inf,52.615)
    @test_throws DomainError iapws95_entropy_density_state(eos,
        iapws95_state(eos;temperature=700.0,density=1.0).entropy-1.0,1.0)
    @test_throws DomainError iapws95_entropy_density_state(eos,
        iapws95_state(eos;temperature=1200.0,density=1.0).entropy+1.0,1.0)
    @test_throws DomainError internal_energy(eos,1.0,0.0,1.0)
    @test_throws DomainError internal_energy(eos,1.0,1.0,0.0)
end

@testset "Restricted IAPWS95 stream inverses and guards" begin
    eos=IAPWS95EOS()
    for (T,rho) in ((725.0,0.01),(900.0,52.615),(1150.0,95.0),
                    (700.0,0.001),(1200.0,100.0),(900.0,0.001),(900.0,100.0))
        st=iapws95_state(eos;temperature=T,density=rho)
        hs=iapws95_stream_state(eos;pressure=st.pressure,entropy=st.entropy)
        sh=iapws95_stream_state(eos;pressure=st.pressure,enthalpy=st.enthalpy)
        @test hs.temperature ≈ T rtol=1e-12
        @test hs.density ≈ rho rtol=1e-12
        @test sh.temperature ≈ T rtol=1e-12
        @test sh.density ≈ rho rtol=1e-12
        @test stream_enthalpy(eos,st.entropy,st.pressure) ≈ st.enthalpy rtol=1e-12
        @test stream_entropy(eos,st.enthalpy,st.pressure) ≈ st.entropy rtol=1e-12
        gh=ForwardDiff.gradient(z->stream_enthalpy(eos,z...),[st.entropy,st.pressure])
        gs=ForwardDiff.gradient(z->stream_entropy(eos,z...),[st.enthalpy,st.pressure])
        @test gh[1] ≈ T rtol=1e-12
        @test gh[2] ≈ inv(rho) rtol=1e-12
        @test gs[1] ≈ inv(T) rtol=1e-12
        @test gs[2] ≈ -inv(rho*T) rtol=1e-12
    end
    st=iapws95_state(eos;temperature=900.0,density=52.615)
    jacdet=st.pressure_density*st.entropy_temperature-st.pressure_temperature*st.entropy_density
    expected=[st.pressure_density -st.entropy_density;
        -st.entropy_density -st.entropy_temperature/st.density^2]/jacdet
    hessian=ForwardDiff.hessian(z->stream_enthalpy(eos,z...),[st.entropy,st.pressure])
    @test all(isapprox.(hessian,expected;rtol=1e-11))
    @test_throws ArgumentError iapws95_stream_state(eos;pressure=st.pressure)
    @test_throws ArgumentError iapws95_stream_state(eos;pressure=st.pressure,entropy=st.entropy,enthalpy=st.enthalpy)
    @test_throws DomainError stream_enthalpy(eos,st.entropy,100.0)
    @test_throws DomainError stream_enthalpy(eos,st.entropy,1e9)
    @test_throws DomainError stream_enthalpy(eos,-1e6,st.pressure)
    @test_throws DomainError stream_entropy(eos,1e9,st.pressure)
    @test_throws DomainError throttle_stream(eos;flow=1.0,enthalpy=st.enthalpy,
        inlet_pressure=st.pressure,outlet_pressure=100.0)
    cool=iapws95_state(eos;temperature=800.0,density=20.0)
    @test_throws DomainError turbine_stage(eos;flow=1.0,inlet_enthalpy=cool.enthalpy,
        inlet_pressure=cool.pressure,exhaust_pressure=0.1cool.pressure,efficiency=0.85)
end

@testset "Restricted IAPWS95 implicit lifts, residual partials and nested AD" begin
    eos=IAPWS95EOS()
    entropy_forward(z)=iapws95_properties(z[1],z[2]).entropy
    for (T,rho) in ((725.0,0.01),(900.0,52.615),(1150.0,95.0))
        st=iapws95_state(eos;temperature=T,density=rho)
        s=st.entropy
        # The lifted entropy residual must be stationary to first order.
        rg=ForwardDiff.gradient(z->iapws95_entropy_density_state(eos,z...).entropy_residual,[s,rho])
        @test abs(rg[1])<=1e-12
        @test abs(rg[2])<=1e-12*abs(st.entropy_density)
        # Nested AD through the lifted root against the implicit-function formula
        # built only from the forward chart's Hessian (no root in the reference).
        H=ForwardDiff.hessian(entropy_forward,[T,rho])
        sT,srho=st.entropy_temperature,st.entropy_density
        sTT,sTr,srr=H[1,1],H[1,2],H[2,2]
        Ts,Tr=1/sT,-srho/sT
        Tsr=-(sTT*Ts*Tr+sTr*Ts)/sT
        expected=[-sTT*Ts^2/sT Tsr;
                  Tsr -(sTT*Tr^2+2sTr*Tr+srr)/sT]
        lifted=ForwardDiff.hessian(z->iapws95_entropy_density_state(eos,z...).temperature,[s,rho])
        @test all(isapprox.(lifted,expected;rtol=1e-9,atol=1e-12*maximum(abs,expected)))
        # Different Dual tags on the two arguments, both differentiation orders.
        m_sr=ForwardDiff.derivative(a->ForwardDiff.derivative(
            b->iapws95_entropy_density_state(eos,s+a,rho+b).temperature,0.0),0.0)
        m_rs=ForwardDiff.derivative(b->ForwardDiff.derivative(
            a->iapws95_entropy_density_state(eos,s+a,rho+b).temperature,0.0),0.0)
        @test m_sr ≈ Tsr rtol=1e-9
        @test m_rs ≈ Tsr rtol=1e-9
        # Stream lifts: pressure and target residuals stationary to first order,
        # measured against the scale of the terms that must cancel.
        for (key,value) in ((:entropy,st.entropy),(:enthalpy,st.enthalpy))
            state=z->key===:entropy ? iapws95_stream_state(eos;pressure=z[2],entropy=z[1]) :
                                      iapws95_stream_state(eos;pressure=z[2],enthalpy=z[1])
            z0=[value,st.pressure]
            gT=ForwardDiff.gradient(z->state(z).temperature,z0)
            grho=ForwardDiff.gradient(z->state(z).density,z0)
            gp=ForwardDiff.gradient(z->state(z).pressure_residual,z0)
            gv=ForwardDiff.gradient(z->state(z).target_residual,z0)
            vT=key===:entropy ? st.entropy_temperature : st.enthalpy_temperature
            vr=key===:entropy ? st.entropy_density : st.enthalpy_density
            pscale=abs.(st.pressure_density.*grho).+abs.(st.pressure_temperature.*gT).+[0.0,1.0]
            vscale=abs.(vr.*grho).+abs.(vT.*gT).+[1.0,0.0]
            @test all(abs.(gp).<=1e-11 .*pscale)
            @test all(abs.(gv).<=1e-11 .*vscale)
        end
    end
    st=iapws95_state(eos;temperature=900.0,density=52.615)
    jacdet=st.pressure_density*st.entropy_temperature-st.pressure_temperature*st.entropy_density
    h_sp=-st.entropy_density/jacdet
    m1=ForwardDiff.derivative(a->ForwardDiff.derivative(
        b->stream_enthalpy(eos,st.entropy+a,st.pressure+b),0.0),0.0)
    m2=ForwardDiff.derivative(b->ForwardDiff.derivative(
        a->stream_enthalpy(eos,st.entropy+a,st.pressure+b),0.0),0.0)
    @test m1 ≈ h_sp rtol=1e-10
    @test m2 ≈ h_sp rtol=1e-10
end

@testset "Restricted IAPWS95 bracket and interval edge cases" begin
    PS=PHPS.PhysicalSteam
    linear(x)=(2.0+3.0*(x-1.0),3.0)
    # Singleton bracket: accept only the single image value (16-eps budget).
    @test PS._iapws95_root(linear,2.0,1.0,1.0,"singleton")==1.0
    @test PS._iapws95_root(linear,2.0+8eps(2.0),1.0,1.0,"singleton")==1.0
    @test_throws DomainError PS._iapws95_root(linear,2.0+1e-9,1.0,1.0,"singleton")
    @test_throws ArgumentError PS._iapws95_root(linear,2.0,2.0,1.0,"reversed")
    # Near-singleton bracket whose image is non-increasing by roundoff only.
    hi=nextfloat(1.0,2)
    roundoff(x)=(x==1.0 ? 2.0+4eps(2.0) : 2.0,3.0)
    @test PS._iapws95_root(roundoff,2.0,1.0,hi,"near") in (1.0,hi)
    @test PS._iapws95_root(roundoff,2.0+4eps(2.0),1.0,hi,"near") in (1.0,hi)
    @test_throws DomainError PS._iapws95_root(roundoff,3.0,1.0,hi,"near")
    # The same non-increasing image on a wide bracket is a numerical failure.
    wide(x)=(x==1.0 ? 2.1 : 2.0,3.0)
    @test_throws ErrorException PS._iapws95_root(wide,2.05,1.0,2.0,"wide")
    # An ordinary increasing bracket is unchanged.
    @test PS._iapws95_root(linear,3.5,1.0,2.0,"linear") ≈ 1.5 rtol=1e-15
    # Crossed fixed-pressure interval: merged only within the ulp budget and
    # only if feasible at the merged temperature; otherwise empty.
    T=900.0
    yes(t)=true; no(t)=false
    merged=PS._iapws95_reconcile_interval(T+4eps(T),T,yes)
    @test merged.collapsed && merged.lower==merged.upper && merged.crossing==4eps(T)
    @test !PS._iapws95_reconcile_interval(T,T+1.0,yes).collapsed
    @test_throws DomainError PS._iapws95_reconcile_interval(T+4eps(T),T,no)
    @test_throws DomainError PS._iapws95_reconcile_interval(T+1e-6,T,yes)
    # All four chart-corner pressures invert exactly; any collapse is reported.
    eos=IAPWS95EOS()
    for (Tc,rhoc) in ((700.0,0.001),(1200.0,100.0),(700.0,100.0),(1200.0,0.001))
        st=iapws95_state(eos;temperature=Tc,density=rhoc)
        hs=iapws95_stream_state(eos;pressure=st.pressure,entropy=st.entropy)
        @test hs.temperature ≈ Tc rtol=1e-12
        @test hs.density ≈ rhoc rtol=1e-12
        lo,up=hs.feasible_temperature_interval
        @test lo<=up
        @test hs.interval_collapsed==false || hs.interval_crossing>0
    end
end

@testset "Restricted IAPWS95 coupled vessel-valve-turbine-rotor fixture" begin
    eos=IAPWS95EOS()
    # Synthetic SI fixture inside the restricted chart; not plant calibration.
    T0,rho0,V=1100.0,50.0,2.0
    M=rho0*V
    st=iapws95_state(eos;temperature=T0,density=rho0)
    vessel=SteamVessel(eos;volume=V)
    law=LinearSteamConductance(valve=1e-6,turbine=1e-6)
    make(pc)=SteamTurbineShaft(vessel,law;inertia=20.0,exhaust_pressure=pc,
        ambient_temperature=300.0,minimum_speed=10.0,efficiency=0.85,
        mechanical_loss_fraction=0.05,friction=0.02)
    model=make(0.5st.pressure)
    x=[M*st.entropy,M,20.0*350.0]
    b=steam_balance(model,x;admission=0.7,load_torque=100.0)
    q=b.closure.mass_flow
    @test q>0
    scale=q*abs(st.enthalpy)
    @test abs(b.energy_residual)<=1e-11*scale
    @test abs(b.entropy_residual)<=1e-11*q*abs(st.entropy)
    @test abs(b.availability_residual)<=1e-11*scale
    @test abs(b.graph_residuals.power)<=1e-11*scale
    @test abs(b.graph_residuals.mechanical_heat)<=1e-11*scale
    @test abs(b.graph_residuals.shaft_power)<=1e-11*scale
    @test abs(b.graph_residuals.inlet_mass)<=1e-12*q
    @test abs(b.graph_residuals.outlet_mass)<=1e-12*q
    @test b.entropy_productions.valve>0
    @test b.entropy_productions.turbine>0
    @test abs(b.valve.entropy_state_residual)<=1e-8
    @test abs(b.turbine.entropy_state_residual)<=1e-8
    @test b.closure.inlet_pressure<st.pressure
    exhaust=iapws95_stream_state(eos;pressure=model.exhaust_pressure,enthalpy=b.turbine.exhaust_enthalpy)
    @test 700.0<exhaust.temperature<b.state.vessel.T
    # Nested AD through the vessel inverse inside the coupled storage.
    g=ForwardDiff.gradient(z->steam_energy(model,z),x)
    @test g ≈ [b.state.vessel.T,b.state.vessel.mu,b.state.speed] rtol=1e-11
    J=ForwardDiff.jacobian(z->steam_balance(model,z;admission=0.7,load_torque=100.0).dx,x)
    @test all(isfinite,J)
    # Explicit outlet guards: an isentropic turbine endpoint below 700 K, and a
    # valve outlet below the chart's pressure image. Both reject; no extrapolation.
    cold=make(0.02st.pressure)
    err=try
        steam_balance(cold,x;admission=0.7)
        nothing
    catch e
        e
    end
    @test err isa DomainError && occursin("stream temperature",err.msg)
    starved=make(100.0)
    err=try
        steam_balance(starved,x;admission=1e-9)
        nothing
    catch e
        e
    end
    @test err isa DomainError && occursin("chart image",err.msg)
end
