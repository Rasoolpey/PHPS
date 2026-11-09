using PHPS.PhysicalSteam

@testset "Steam unit as the mechanical source of a machine in the assembled grid" begin
    eos=IAPWS95EOS()
    path=joinpath(pkgdir(PHPS),"cases","Bench3_RMS","system_phtrue.json")
    chart=iapws95_state(eos;temperature=1100.0,density=50.0)
    # Synthetic steam scale; one factor on conductances and volume, not plant data.
    build(sys,scale)=SteamGridUnit(sys,"GENROU_1",SteamVessel(eos;volume=600.0*scale),
        LinearSteamConductance(valve=3e-4*scale,turbine=3e-4*scale);exhaust_pressure=0.5chart.pressure,
        ambient_temperature=300.0,efficiency=0.85,mechanical_loss_fraction=0.05,minimum_speed_pu=0.1)
    vessel(scale)=[50.0*600.0*scale*chart.entropy,50.0*600.0*scale]

    governed=solve_equilibrium(load_case(path))
    # The machine's mechanical input must be free: with its governor attached it is not.
    @test_throws ArgumentError build(governed.sys,1.0)
    @test_throws ArgumentError case_without_component(path,"NOPE")
    baseline=solve_equilibrium(case_without_component(path,"IEEEG1_1"))
    sys=baseline.sys
    @test sys.n_diff==governed.sys.n_diff-7
    @test baseline.V ≈ governed.V atol=1e-9
    @test_throws ArgumentError SteamGridUnit(sys,"GENROU_3",SteamVessel(eos;volume=600.0),
        LinearSteamConductance(valve=3e-4,turbine=3e-4);exhaust_pressure=0.5chart.pressure,
        ambient_temperature=300.0,efficiency=0.85,mechanical_loss_fraction=0.05,minimum_speed_pu=0.1)

    n=sys.n_diff; nV=PHPS.nalg(sys)
    k=findfirst(c->PHPS.name(c)=="GENROU_1",sys.comps)
    row=sys.offsets[k]+1
    H=sys.comps[k].p.H
    probe=build(sys,1.0)
    X1=vcat(baseline.x,vessel(1.0))
    reference=steam_grid_dae(probe,X1,baseline.V,steam_grid_algebraic(probe,X1;admission=0.7);admission=0.7)[3]
    scale=probe.constant_power/reference.mechanical_power_pu
    unit=build(sys,scale)
    @test steam_grid_sizes(unit)==(differential=n+2,algebraic=nV+9)
    @test unit.power_base==1e8
    @test unit.steam.inertia*unit.mechanical_base_speed^2 ≈ 2H*unit.power_base rtol=1e-14
    X=vcat(baseline.x,vessel(scale)); V=copy(baseline.V)
    y=steam_grid_algebraic(unit,X;admission=0.7)
    rotor=PHPS._steam_grid_rotor_state(unit,X)
    free=steam_balance(unit.steam,rotor;admission=0.7)
    q=free.closure.mass_flow
    feed=iapws95_state(eos;temperature=1000.0,density=70.0)
    inputs=(admission=0.7,heat_source_temperature=1300.0,
        heat_conductance=q*(free.state.vessel.h-feed.enthalpy)/(1300.0-free.state.vessel.T),
        feed_flow=q,feed_enthalpy=feed.enthalpy,feed_entropy=feed.entropy)

    # Composite equilibrium: the grid rows, KCL, the steam rates and the steam constraints.
    F,G,report=steam_grid_dae(unit,X,V,y;inputs...)
    @test report.mechanical_power_pu ≈ unit.constant_power rtol=1e-12
    @test report.shaft_power ≈ free.turbine.shaft_power rtol=1e-12
    @test maximum(abs,F[1:n])<=1e-10 && maximum(abs,G[1:nV])<=1e-10
    @test abs(F[n+1])<=1e-10*q*abs(free.state.vessel.s) && abs(F[n+2])<=1e-10*q
    scales=steam_constraint_jacobian(unit.steam,rotor,y;admission=0.7).row_scales
    @test maximum(abs,G[nV+1:end]./scales)<=1e-10
    @test steam_domain_margins(unit.steam,rotor;inputs...).status===:admissible
    # Without the steam correction the grid rows are the assembled system's own.
    f,g=dae_residual(sys,baseline.x,V)
    @test G[1:nV]==g
    @test all(F[j]==f[j] for j in 1:n if j!=row)

    # Away from equilibrium and at nonunit speed, the corrected rows equal the assembled
    # system evaluated with the steam power as its constant mechanical input.
    Xp=copy(X); Xp[row]=0.97; Xp[row-1]+=0.1; Xp[n+1]*=1.05; Xp[n+2]*=1.05
    Vp=0.98.*V
    yp=steam_grid_algebraic(unit,Xp;admission=0.75)
    Fp,_,rp=steam_grid_dae(unit,Xp,Vp,yp;inputs...,admission=0.75)
    @test abs(rp.mechanical_power_pu-unit.constant_power)>1e-3
    sources=deepcopy(sys.sources)
    sources[k][3]=PHPS.const_source(rp.mechanical_power_pu)
    direct=PHPS.DAESystem((s===:sources ? sources : getfield(sys,s) for s in fieldnames(PHPS.DAESystem))...)
    fp,_=dae_residual(direct,Xp[1:n],Vp)
    @test maximum(abs,Fp[1:n].-fp)<=1e-12
    # Steam rates do not depend on the load side.
    @test Fp[n+1:n+2]==steam_grid_dae(unit,Xp,V,yp;inputs...,admission=0.75)[1][n+1:n+2]

    # Algebraic Jacobian: block diagonal, grid block unchanged, steam block as derived.
    jac=steam_grid_jacobians(unit,X,V,y;inputs...)
    @test all(iszero,jac.GV[nV+1:end,:]) && all(iszero,jac.Gy[1:nV,:])
    gv=ForwardDiff.jacobian(v->dae_residual(sys,baseline.x,v)[2],V)
    @test jac.GV[1:nV,:]==gv
    @test minimum(svdvals(gv))>1.0
    analytic=steam_constraint_jacobian(unit.steam,rotor,y;admission=0.7)
    @test all(isapprox.(jac.Gy[nV+1:end,:],analytic.jacobian;rtol=1e-10))
    # One-way coupling with the declared laws.
    @test all(iszero,jac.FX[n+1:n+2,1:n]) && all(iszero,jac.FV[n+1:n+2,:]) && all(iszero,jac.GX[nV+1:end,1:n])
    # AD against central differences for the speed row with respect to the steam state.
    for j in (n+1,n+2)
        step=1e-6*abs(X[j]); e=zeros(n+2); e[j]=step
        ya,yb=steam_grid_algebraic(unit,X.+e;admission=0.7),steam_grid_algebraic(unit,X.-e;admission=0.7)
        total=(steam_grid_dae(unit,X.+e,V,ya;inputs...)[1][row]-steam_grid_dae(unit,X.-e,V,yb;inputs...)[1][row])/(2step)
        chain=jac.FX[row,j]-dot(jac.Fy[row,:],jac.Gy[nV+1:end,:]\jac.GX[nV+1:end,j])
        @test chain ≈ total rtol=1e-5
    end

    # Rotation: a null direction of the full and reduced Jacobians, not of dg/dV alone.
    dx,dV=grid_rotation_direction(sys,baseline.x,V)
    v=vcat(dx,0.0,0.0)
    @test maximum(abs,jac.FX*v+jac.FV*dV)<=1e-9
    @test maximum(abs,jac.GX*v+jac.GV*dV)<=1e-9
    @test maximum(abs,jac.reduced*v)<=1e-9
    @test maximum(abs,gv*dV)>1.0
    zero_columns=count(j->all(iszero,jac.reduced[:,j]),1:n+2)
    @test count(e->abs(e)<1e-6,eigvals(jac.reduced))==zero_columns+1
    @test_throws DimensionMismatch steam_grid_dae(unit,X[1:n],V,y;inputs...)
end
