using PHPS.PhysicalSteam

@testset "Steam component on a rigid shaft with the GENROU kernel" begin
    eos=IAPWS95EOS()
    case=load_case(joinpath(pkgdir(PHPS),"cases","Bench3_RMS","system_phtrue.json"))
    genrou=build_component(case,component(case,"GENROU_1"))
    Sb=1e6*case.config.mva_base
    chart=iapws95_state(eos;temperature=1100.0,density=50.0)
    V=600.0
    # Synthetic steam scale (about 2 pu shaft power); not plant calibration.
    function build(;pole_pairs=1,friction_pu=0.0,machine=genrou)
        SteamMachineShaft(SteamVessel(eos;volume=V),LinearSteamConductance(valve=3e-4,turbine=3e-4),
            machine;power_base=Sb,pole_pairs,exhaust_pressure=0.5chart.pressure,
            ambient_temperature=300.0,efficiency=0.85,mechanical_loss_fraction=0.05,
            minimum_speed_pu=0.1,friction_pu)
    end
    function damped(machine,D)
        p=machine.p
        entries=[k===:D ? D : getfield(p,k) for k in fieldnames(typeof(p))]
        typeof(machine)(machine.name,machine.bus,typeof(p)(entries...),machine.params)
    end
    shaft=build()
    H=genrou.p.H
    @test genrou.p.D==0 && genrou.p.ra==0
    @test shaft.steam.inertia*shaft.mechanical_base_speed^2 ≈ 2H*Sb rtol=1e-14
    @test shaft.mechanical_base_speed==genrou.p.omega_b
    @test build(;pole_pairs=3).mechanical_base_speed ≈ genrou.p.omega_b/3

    z=[50V*chart.entropy,50V,0.6,0.98,1.0,0.9,0.2,-0.4]
    inputs=(admission=0.7,terminal_voltage=(0.97,0.1),field_voltage=2.0)
    b=steam_machine_balance(shaft,z;inputs...)
    vessel=b.steam.state.vessel
    @test steam_machine_energy(shaft,z) ≈ vessel.U+Sb*H*z[4]^2 rtol=1e-14
    # The kernel receives the actual net shaft power on the power base.
    @test b.mechanical_power_pu ≈ b.shaft_power/Sb rtol=1e-14
    direct=rhs!(zeros(6),genrou,z[3:8],[0.97,0.1,b.mechanical_power_pu,2.0])
    @test b.dz[3:8]==direct
    @test b.dz[1:2]==b.steam.dx[1:2]
    @test b.terminal_power_pu ≈ b.electrical_torque_pu*z[4]
    @test b.power.electrical_out ≈ Sb*b.terminal_power_pu
    @test steam_machine_rhs!(zeros(8),shaft,z;inputs...)==b.dz
    # Kinetic energy rate from the kernel's swing: shaft power less terminal power.
    @test 2Sb*H*z[4]*b.omega_rate_kernel ≈ b.shaft_power-b.power.electrical_out rtol=1e-12
    @test abs(b.energy_residual)<=1e-12*(abs(b.shaft_power)+abs(b.power.electrical_out))

    # Rotor-owned swing equals the kernel's swing, with friction and with damping,
    # and nothing depends on the pole-pair number.
    for friction_pu in (0.0,0.01), D in (0.0,2.0)
        machine=damped(genrou,D)
        one=steam_machine_balance(build(;friction_pu,machine),z;inputs...)
        three=steam_machine_balance(build(;pole_pairs=3,friction_pu,machine),z;inputs...)
        @test one.omega_rate_rotor ≈ one.omega_rate_kernel rtol=1e-12
        @test three.omega_rate_rotor ≈ three.omega_rate_kernel rtol=1e-12
        @test all(isapprox.(one.dz,three.dz;rtol=1e-12,atol=0))
        @test one.power.friction_heat_out ≈ friction_pu*Sb*z[4]^2 atol=1e-6
        @test one.power.damping_exchange ≈ Sb*D*z[4]*(z[4]-1) atol=1e-6
        @test one.mechanical_power_pu ≈ (one.shaft_power-one.power.friction_heat_out)/Sb rtol=1e-14
        @test abs(one.energy_residual)<=1e-12*(abs(one.shaft_power)+abs(one.power.electrical_out))
    end
    # Below synchronous speed the damping term injects power: it is not a dissipation.
    @test steam_machine_balance(build(;machine=damped(genrou,2.0)),z;inputs...).power.damping_exchange<0

    # Bregman shift in (S,M,l).
    zstar=[50V*chart.entropy,50V,0.5,1.0,1.1,1.0,0.3,-0.4]
    @test steam_machine_shift(shaft,zstar,zstar).shift==0
    shifted=steam_machine_shift(shaft,z,zstar)
    @test shifted.shift ≈ Sb*H*(z[4]-1)^2 rtol=1e-9      # same vessel state: rotor part only
    @test shifted.rotor_shift ≈ Sb*H*(z[4]-1)^2
    moved=copy(zstar); moved[1]*=1.001; moved[2]*=0.999; moved[4]=1.01
    w=steam_machine_shift(shaft,moved,zstar)
    @test w.shift>w.rotor_shift>0
    reference=PhysicalSteam.steam_state(shaft.steam,PHPS._steam_rotor_state(shaft,zstar))
    @test w.shift ≈ steam_machine_energy(shaft,moved)-steam_machine_energy(shaft,zstar)-
        dot([reference.vessel.T,reference.vessel.mu,reference.speed],w.state_difference) rtol=1e-9
    # The angle and the electromagnetic states do not enter the shift.
    turned=copy(moved); turned[3]+=1.0; turned[5:8].*=1.3
    @test steam_machine_shift(shaft,turned,zstar).shift==w.shift

    # Eight seed directions: AD against central differences (see the AD hazard in the record).
    f(v)=steam_machine_balance(shaft,v;inputs...).dz
    scale=[abs(z[1]),z[2],1.0,1.0,1.0,1.0,1.0,1.0]
    step=1e-6
    fd=hcat([(f(z.+step*scale[j].*(1:8 .==j)).-f(z.-step*scale[j].*(1:8 .==j)))./(2step*scale[j]) for j in 1:8]...)
    ad=ForwardDiff.jacobian(f,z)
    # Row-scaled, so that small rows are not hidden behind the largest derivative.
    rows=[maximum(abs,fd[i,:].*scale) for i in 1:8]
    @test maximum(abs,Diagonal(inv.(rows))*(ad.-fd)*Diagonal(scale))<=1e-5

    @test_throws DomainError build(;pole_pairs=0)
    @test_throws DomainError SteamMachineShaft(SteamVessel(eos;volume=V),
        LinearSteamConductance(valve=3e-4,turbine=3e-4),genrou;power_base=0.0,exhaust_pressure=1e6,
        ambient_temperature=300.0,efficiency=0.85,mechanical_loss_fraction=0.05,minimum_speed_pu=0.1)
    @test_throws DimensionMismatch steam_machine_balance(shaft,z[1:7];inputs...)
    slow=copy(z); slow[4]=0.05
    @test_throws DomainError steam_machine_balance(shaft,slow;inputs...)
    @test_throws DomainError steam_machine_balance(shaft,z;inputs...,admission=1.5)
end
