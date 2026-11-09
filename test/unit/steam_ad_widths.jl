using PHPS.PhysicalSteam

# Regression for a wrong value from nested AD: with Dual inputs of exactly six partials
# the efforts from `thermodynamic_state` were wrong through the IAPWS-95 provider at
# Julia's default optimisation level (docs/active/STEAM_MACHINE_SHAFT_COUPLING.md §8).
# The efforts and their derivatives must not depend on the outer derivative width.
@testset "Thermodynamic efforts under nested AD of every outer width" begin
    primal(x)=x isa ForwardDiff.Dual ? primal(ForwardDiff.value(x)) : x
    padded(v,n,k=0)=ForwardDiff.Dual{:width}(v,ntuple(i->i==k ? 1.0 : 0.0,n)...)
    part(x,k)=ForwardDiff.partials(x)[k]
    eos=IAPWS95EOS()
    for (T0,rho0,V) in ((1100.0,50.0,600.0),(900.0,5.0,2.0))
        st=iapws95_state(eos;temperature=T0,density=rho0)
        M=rho0*V; S=M*st.entropy
        plain=thermodynamic_state(eos,S,M,V)
        # The gradient efforts against the forward chart.
        @test plain.T ≈ T0 rtol=1e-12
        @test plain.mu ≈ st.chemical_potential rtol=1e-12
        @test plain.p ≈ st.pressure rtol=1e-12
        # Second derivatives of U(S,M,V) from the chart's closed forms.
        b=st.pressure_temperature/rho0-st.entropy
        T_S=T0/(M*st.cv); T_M=T0*b/(M*st.cv)
        mu_M=(st.pressure_density+T0/st.cv*b^2)/M
        p_S=st.pressure_temperature*T_S
        p_M=st.pressure_temperature*T_M+st.pressure_density/V
        p_V=-st.pressure_temperature*p_S-st.pressure_density*M/V^2
        expected=(T=(T_S,T_M,-p_S),mu=(T_M,mu_M,-p_M),p=(p_S,p_M,p_V))
        for n in 1:12
            zero_seeded=thermodynamic_state(eos,padded(S,n),padded(M,n),V)
            @test primal(zero_seeded.T) ≈ plain.T rtol=1e-13
            @test primal(zero_seeded.mu) ≈ plain.mu rtol=1e-13
            @test primal(zero_seeded.p) ≈ plain.p rtol=1e-13
            # Seed S, M and (from width three) V; any further directions stay empty.
            seeded=thermodynamic_state(eos,padded(S,n,1),padded(M,n,2),n>=3 ? padded(V,n,3) : V)
            @test primal(seeded.T) ≈ plain.T rtol=1e-13
            for (effort,derivatives) in pairs(expected), k in 1:min(n,3)
                @test part(getproperty(seeded,effort),k) ≈ derivatives[k] rtol=1e-9
            end
            @test all(iszero(part(getproperty(seeded,effort),k)) for effort in (:T,:mu,:p) for k in 4:n)
        end
        # Both stream inversions at and around the width that failed.
        for n in (5,6,7)
            h=iapws95_stream_state(eos;pressure=padded(st.pressure,n,2),entropy=padded(st.entropy,n,1))
            @test primal(h.temperature) ≈ T0 rtol=1e-12
            @test part(h.enthalpy,1) ≈ T0 rtol=1e-10
            @test part(h.enthalpy,2) ≈ inv(rho0) rtol=1e-10
            s=iapws95_stream_state(eos;pressure=padded(st.pressure,n,2),enthalpy=padded(st.enthalpy,n,1))
            @test primal(s.temperature) ≈ T0 rtol=1e-12
            @test part(s.entropy,1) ≈ inv(T0) rtol=1e-10
            @test part(s.entropy,2) ≈ -inv(rho0*T0) rtol=1e-10
            @test all(iszero(part(q,k)) for q in (h.enthalpy,s.entropy) for k in 3:n)
        end
    end
    # Two-level shapes straight into the property kernel. With m = 3 inner directions
    # over n = 6 outer ones the kernel returned a wrong primal entropy and enthalpy at
    # Julia's default optimisation level. Its consistency guard must make that loud:
    # the known shape either agrees or raises; every other shape must agree.
    nested(v,m,n)=(inner=padded(v,n); ForwardDiff.Dual{:inner}(inner,ntuple(_->zero(inner),m)...))
    reference=iapws95_properties(1100.0,50.0)
    leaves=(:pressure,:internal_energy_specific,:entropy,:enthalpy,:cv,:cp,:chemical_potential)
    for m in 1:3, n in (5,6,7)
        outcome=try
            iapws95_properties(nested(1100.0,m,n),nested(50.0,m,n))
        catch err
            err
        end
        if outcome isa Exception
            @test (m,n)==(3,6) && outcome isa IAPWS95ConsistencyError
        else
            @test all(isapprox(primal(getproperty(outcome,f)),getproperty(reference,f);rtol=1e-12) for f in leaves)
        end
    end
    # The guard itself: a deliberately inconsistent result is rejected, a consistent one passes.
    PS=PHPS.PhysicalSteam
    @test PS._iapws95_check_primal(reference,reference,Float64)===nothing
    @test_throws IAPWS95ConsistencyError PS._iapws95_check_primal(
        merge(reference,(;entropy=reference.entropy+1.0)),reference,Float64)
    @test_throws IAPWS95ConsistencyError PS._iapws95_check_primal(
        merge(reference,(;enthalpy=NaN)),reference,Float64)
    # A nonfinite reference is rejected too, and the echoed inputs are covered.
    @test_throws IAPWS95ConsistencyError PS._iapws95_check_primal(
        reference,merge(reference,(;cv=Inf)),Float64)
    @test_throws IAPWS95ConsistencyError PS._iapws95_check_primal(
        merge(reference,(;density=reference.density*1.001)),reference,Float64)
    @test_throws IAPWS95ConsistencyError PS._iapws95_check_primal(
        merge(reference,(;reduced=(;delta=reference.reduced.delta,tau=reference.reduced.tau*1.001))),reference,Float64)
    gas=PerfectGasEOS(cv=1500.0,R=500.0)
    x=perfect_gas_state(gas;mass=10.0,temperature=600.0,volume=2.0)
    plain=thermodynamic_state(gas,x[1],x[2],2.0)
    for n in 1:12
        seeded=thermodynamic_state(gas,padded(x[1],n,1),padded(x[2],n,min(2,n)),2.0)
        @test primal(seeded.T) ≈ plain.T rtol=1e-13
        @test primal(seeded.mu) ≈ plain.mu rtol=1e-13
        @test primal(seeded.p) ≈ plain.p rtol=1e-13
    end
end
