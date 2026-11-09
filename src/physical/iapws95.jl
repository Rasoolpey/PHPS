# International Association for the Properties of Water and Steam, R6-95(2018),
# Revised Release on IAPWS-95, equations (1)-(6), Tables 1-3, pp. 3-4,9-11.
# https://iapws.org/technical-guidance/release/IAPWS-95.download
# The release permits reproduction with attribution. This is our transcription.
# Raw Helmholtz kernel only: no phase selection, inversion or SteamEOS provider.
const IAPWS95_TC=647.096
const IAPWS95_RHOC=322.0
const IAPWS95_R=461.51805 # release value, J/(kg K); do not replace with newer R.

# Table 2: (c,d,t,n); c=0 means the first undamped polynomial sum.
const IAPWS95_TERMS=(
    (0,1,-0.5,0.012533547935523),
    (0,1,0.875,7.8957634722828),
    (0,1,1.0,-8.7803203303561),
    (0,2,0.5,0.31802509345418),
    (0,2,0.75,-0.26145533859358),
    (0,3,0.375,-0.0078199751687981),
    (0,4,1.0,0.0088089493102134),
    (1,1,4.0,-0.66856572307965),
    (1,1,6.0,0.20433810950965),
    (1,1,12.0,-0.000066212605039687),
    (1,2,1.0,-0.19232721156002),
    (1,2,5.0,-0.25709043003438),
    (1,3,4.0,0.16074868486251),
    (1,4,2.0,-0.040092828925807),
    (1,4,13.0,0.00000039343422603254),
    (1,5,9.0,-0.0000075941377088144),
    (1,7,3.0,0.00056250979351888),
    (1,9,4.0,-0.000015608652257135),
    (1,10,11.0,0.0000000011537996422951),
    (1,11,4.0,0.00000036582165144204),
    (1,13,13.0,-0.0000000000013251180074668),
    (1,15,1.0,-0.00000000062639586912454),
    (2,1,7.0,-0.10793600908932),
    (2,2,1.0,0.017611491008752),
    (2,2,9.0,0.22132295167546),
    (2,2,10.0,-0.40247669763528),
    (2,3,10.0,0.58083399985759),
    (2,4,3.0,0.0049969146990806),
    (2,4,7.0,-0.031358700712549),
    (2,4,10.0,-0.74315929710341),
    (2,5,10.0,0.47807329915480),
    (2,6,6.0,0.020527940895948),
    (2,6,10.0,-0.13636435110343),
    (2,7,10.0,0.014180634400617),
    (2,9,1.0,0.0083326504880713),
    (2,9,2.0,-0.029052336009585),
    (2,9,3.0,0.038615085574206),
    (2,9,4.0,-0.020393486513704),
    (2,9,8.0,-0.0016554050063734),
    (2,10,6.0,0.0019955571979541),
    (2,10,9.0,0.00015870308324157),
    (2,12,8.0,-0.000016388568342530),
    (3,3,16.0,0.043613615723811),
    (3,4,22.0,0.034994005463765),
    (3,4,23.0,-0.076788197844621),
    (3,5,23.0,0.022446277332006),
    (4,14,10.0,-0.000062689710414685),
    (6,3,50.0,-0.00000000055711118565645),
    (6,6,44.0,-0.19905718354408),
    (6,6,46.0,0.31777497330738),
    (6,6,50.0,-0.11841182425981))

"""
IAPWS-95 ideal and residual dimensionless Helmholtz parts at delta=rho/rhoc,
tau=Tc/T. Raw formula, not a phase/domain certificate. Exact critical point is
excluded; fractional critical terms are not globally smooth there.
"""
function iapws95_helmholtz_parts(delta,tau)
    _positive(delta,"reduced density"); _positive(tau,"inverse reduced temperature")
    delta==1 && tau==1 && throw(DomainError((delta,tau),"critical point excluded"))
    ideal=log(delta)-8.3204464837497+6.6832105275932*tau+3.00632*log(tau)
    for (n,gamma) in ((0.012436,1.28728967),(0.97315,3.53734222),
                      (1.27950,7.74073708),(0.96956,9.24437796),(0.24873,27.5075105))
        ideal+=n*log(-expm1(-gamma*tau))
    end
    residual=zero(delta+tau)
    for (c,d,t,n) in IAPWS95_TERMS
        term=n*delta^d*tau^t
        residual+=c==0 ? term : term*exp(-delta^c)
    end
    for (t,n,beta,gamma) in ((0,-31.306260323435,150,1.21),
                             (1,31.546140237781,150,1.21),
                             (4,-2521.3154341695,250,1.25))
        residual+=n*delta^3*tau^t*exp(-20*(delta-1)^2-beta*(tau-gamma)^2)
    end
    # Algebraically identical to powers of (delta-1)^2 in eq. (6); writing the
    # powers directly in |delta-1| avoids nested-AD 0^negative*0 artifacts.
    distance=abs(delta-1)
    theta=1-tau+0.32*distance^(1/0.3)
    Delta=theta^2+0.2*distance^7
    for (n,b,C,D) in ((-0.14874640856724,0.85,28,700),(0.31806110878444,0.95,32,800))
        residual+=n*Delta^b*delta*exp(-C*(delta-1)^2-D*(tau-1)^2)
    end
    (;ideal,residual)
end

"""
Raised when `iapws95_properties` is evaluated on ForwardDiff Dual numbers and the primal
part of a returned value differs from the same kernel evaluated on the Float64 primal
inputs. `field` names the first leaf that disagrees.

This guards against a known wrong result: for inputs of type
`Dual{.,Dual{.,Float64,6},3}` at Julia's default optimisation level (1.12.7, Windows
x86-64, LLVM 18.1.7) the kernel returned a wrong primal entropy and enthalpy although
every Helmholtz derivative leaf was correct; at `-O1` and `-O0` it was right. See
docs/active/STEAM_MACHINE_SHAFT_COUPLING.md, section 8. The check covers primal values
only; it does not verify the derivative parts.
"""
struct IAPWS95ConsistencyError <: Exception
    field::Symbol
    value::Float64
    reference::Float64
    temperature::Float64
    density::Float64
    input_type::String
end
Base.showerror(io::IO,e::IAPWS95ConsistencyError)=print(io,
    "IAPWS95 nested-AD consistency: primal ",e.field," = ",e.value,
    " differs from the Float64 kernel value ",e.reference," at T = ",e.temperature,
    " K, rho = ",e.density," kg/m^3 for input type ",e.input_type,
    ". The derivative parts of this evaluation cannot be trusted.")

_iapws95_deep(x)=x
_iapws95_deep(x::ForwardDiff.Dual)=_iapws95_deep(ForwardDiff.value(x))

const _IAPWS95_CONSISTENCY_RTOL=1e-9
const _IAPWS95_CHECKED_FIELDS=(:pressure,:internal_energy_specific,:entropy,:enthalpy,:cv,:cp,
    :sound_speed,:chemical_potential,:pressure_density,:pressure_temperature,
    :entropy_temperature,:entropy_density,:enthalpy_temperature,:enthalpy_density)

# Absolute floors for the leaves that can be small against their natural scale:
# energies against R*T, entropy against R, the chemical potential against |h|+|T s|
# and the density derivative of the enthalpy (zero for an ideal gas) against p_rho/rho.
function _iapws95_floor(field::Symbol,ref)
    field===:internal_energy_specific || field===:enthalpy ? IAPWS95_R*ref.temperature :
    field===:entropy ? IAPWS95_R :
    field===:chemical_potential ? abs(ref.enthalpy)+abs(ref.temperature*ref.entropy) :
    field===:enthalpy_density ? abs(ref.pressure_density/ref.density) : 0.0
end

@noinline function _iapws95_check_primal(st,ref,input_type)
    fail(field,value,reference)=throw(IAPWS95ConsistencyError(field,value,reference,
        ref.temperature,ref.density,string(input_type)))
    agree(value,reference,floor)=isfinite(value) && isfinite(reference) &&
        abs(value-reference)<=_IAPWS95_CONSISTENCY_RTOL*max(abs(value),abs(reference),floor)
    # Echoed inputs and reduced coordinates.
    for field in (:temperature,:density)
        value,reference=_iapws95_deep(getproperty(st,field)),getproperty(ref,field)
        agree(value,reference,0.0) || fail(field,value,reference)
    end
    for field in (:delta,:tau)
        value,reference=_iapws95_deep(getproperty(st.reduced,field)),getproperty(ref.reduced,field)
        agree(value,reference,0.0) || fail(Symbol(:reduced_,field),value,reference)
    end
    for field in _IAPWS95_CHECKED_FIELDS
        value,reference=_iapws95_deep(getproperty(st,field)),getproperty(ref,field)
        agree(value,reference,_iapws95_floor(field,ref)) || fail(field,value,reference)
    end
    # Dimensionless Helmholtz leaves, with an absolute floor of one.
    for field in (:ideal,:residual,:gradient_ideal,:gradient_residual,:hessian_ideal,:hessian_residual)
        values,references=getproperty(st.helmholtz,field),getproperty(ref.helmholtz,field)
        for (value,reference) in zip(values,references)
            agree(_iapws95_deep(value),reference,1.0) || fail(Symbol(:helmholtz_,field),_iapws95_deep(value),reference)
        end
    end
    nothing
end

"""
Raw IAPWS-95 properties at (T K, rho kg/m^3); energy/entropy outputs are SI.
Derivatives come from the transcribed Helmholtz formula using ForwardDiff.
This evaluates a supplied state and does not choose a stable phase or invert
state/stream coordinates. Unstable computed heat capacity/compressibility or
sound-speed branches are rejected; that is not a full phase-selection test.

For Float64-backed Dual inputs the primal part of every returned leaf is compared with
the Float64 evaluation, and `IAPWS95ConsistencyError` is raised on disagreement.
"""
function iapws95_properties(temperature,density)
    st=_iapws95_properties_unchecked(temperature,density)
    temperature isa ForwardDiff.Dual || density isa ForwardDiff.Dual || return st
    T0,rho0=_iapws95_deep(temperature),_iapws95_deep(density)
    T0 isa Float64 && rho0 isa Float64 || return st
    _iapws95_check_primal(st,_iapws95_properties_unchecked(T0,rho0),typeof(temperature+density))
    st
end

function _iapws95_properties_unchecked(temperature,density)
    _positive(temperature,"temperature"); _positive(density,"density")
    delta=density/IAPWS95_RHOC
    tau=IAPWS95_TC/temperature
    z=collect(promote(delta,tau))
    phi0(v)=iapws95_helmholtz_parts(v[1],v[2]).ideal
    phir(v)=iapws95_helmholtz_parts(v[1],v[2]).residual
    parts=iapws95_helmholtz_parts(z...)
    g0=ForwardDiff.gradient(phi0,z)
    gr=ForwardDiff.gradient(phir,z)
    H0=ForwardDiff.hessian(phi0,z)
    Hr=ForwardDiff.hessian(phir,z)
    phi_tau=g0[2]+gr[2]
    phi_tautau=H0[2,2]+Hr[2,2]
    D=1+2delta*gr[1]+delta^2*Hr[1,1]
    N=1+delta*gr[1]-delta*tau*Hr[1,2]
    R=IAPWS95_R
    pressure=density*R*temperature*(1+delta*gr[1])
    internal_energy_specific=R*temperature*tau*phi_tau
    entropy=R*(tau*phi_tau-parts.ideal-parts.residual)
    enthalpy=R*temperature*(1+tau*phi_tau+delta*gr[1])
    cv=-R*tau^2*phi_tautau
    _positive(cv,"computed isochoric heat capacity")
    _positive(D,"computed compressibility denominator")
    cp=cv+R*N^2/D
    sound_speed_squared=R*temperature*(D-N^2/(tau^2*phi_tautau))
    _positive(pressure,"computed pressure"); _positive(sound_speed_squared,"computed squared sound speed")
    sound_speed=sqrt(sound_speed_squared)
    pressure_density=R*temperature*D
    pressure_temperature=density*R*N
    entropy_temperature=cv/temperature
    entropy_density=-pressure_temperature/density^2
    enthalpy_temperature=cv+pressure_temperature/density
    enthalpy_density=pressure_density/density-temperature*pressure_temperature/density^2
    (;temperature,density,pressure,internal_energy_specific,entropy,enthalpy,cv,cp,sound_speed,
      chemical_potential=enthalpy-temperature*entropy,pressure_density,pressure_temperature,
      entropy_temperature,entropy_density,enthalpy_temperature,enthalpy_density,
      reduced=(;delta,tau),helmholtz=(;parts...,gradient_ideal=g0,gradient_residual=gr,
          hessian_ideal=H0,hessian_residual=Hr))
end
