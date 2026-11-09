using PHPS.PhysicalSteam

@testset "Published IAPWS95 raw Helmholtz kernel" begin
    st=iapws95_properties(500.0,0.435)
    @test st.pressure ≈ 99967.9423 rtol=1e-8
    @test st.cv ≈ 1508.17541 rtol=1e-8
    @test st.sound_speed ≈ 548.314253 rtol=1e-8
    @test st.entropy ≈ 7944.88271 rtol=1e-8
    @test st.enthalpy ≈ st.internal_energy_specific+st.pressure/st.density
    @test st.cp>st.cv>0
    @test st.chemical_potential ≈ st.enthalpy-st.temperature*st.entropy
    hi=iapws95_properties(900.0,52.615)
    @test hi.pressure ≈ 20.0000690e6 rtol=1e-8
    @test hi.pressure_density>0
    @test hi.entropy_temperature ≈ hi.cv/900
    @test hi.cp-hi.cv ≈ 900hi.pressure_temperature^2/(hi.density^2*hi.pressure_density)
    jac=ForwardDiff.jacobian([52.615,900.0]) do z
        state=iapws95_properties(z[2],z[1])
        [state.pressure,state.entropy,state.enthalpy]
    end
    @test all(isfinite,jac)
    @test det(jac[[1,3],:]) ≈ 900det(jac[1:2,:])
    axis=iapws95_properties(900.0,322.0)
    @test isfinite(axis.cv) && isfinite(axis.cp)
    @test_throws DomainError iapws95_properties(647.096,322.0)
    @test_throws DomainError iapws95_properties(0.0,1.0)
    @test_throws DomainError iapws95_helmholtz_parts(-1.0,1.0)
end
