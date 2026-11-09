"""
Restricted IAPWS-95 provider: 700 <= T <= 1200 K and .001 <= rho <= 100 kg/m^3.
These fixed validity bounds exclude saturation, wet steam and the critical region.
They are numerical chart boundaries, not physical invariant boundaries. Positive
cv/compressibility is checked at evaluations; whole-box regularity is not certified.
"""
struct IAPWS95EOS <: AbstractSteamEOS end

const IAPWS95_T_BOUNDS=(700.0,1200.0)
const IAPWS95_RHO_BOUNDS=(0.001,100.0)

_iapws95_primal(x)=x
_iapws95_primal(x::ForwardDiff.Dual)=_iapws95_primal(ForwardDiff.value(x))

function _iapws95_bound(value,bounds,label)
    base=_iapws95_primal(value)
    isfinite(base) && bounds[1]<=base<=bounds[2] ||
        throw(DomainError(base,"$label outside restricted IAPWS95 chart $bounds"))
    value
end

"""Forward thermodynamic chart, with explicit high-temperature validity guards."""
function iapws95_state(::IAPWS95EOS;temperature,density)
    _iapws95_bound(temperature,IAPWS95_T_BOUNDS,"temperature")
    _iapws95_bound(density,IAPWS95_RHO_BOUNDS,"density")
    iapws95_properties(temperature,density)
end

# Monotone safeguarded Newton, performed on primal inputs only. No physical
# trajectory is projected onto a face. Numerical budgets, each relative to the
# named scale:
#  * interior root residual: 8 eps of the target scale;
#  * endpoint acceptance: 16 eps of the larger of target and endpoint image, twice
#    the interior budget, so a value returned at interior tolerance by one solve is
#    recognised as the same face value by a later solve;
#  * near-singleton bracket: width at most 32 eps of the larger endpoint. Only on
#    such a bracket may the two image values contradict the increasing order, and
#    then only by roundoff. A non-increasing image on a wider bracket is a numerical
#    or regularity failure (error), not a chart-domain violation (DomainError).
# Order: singleton bracket, orientation, image rejection, endpoints, interior solve.
const _IAPWS95_NEAR_SINGLETON_ULPS=32

_iapws95_near_singleton(lo,hi)=hi-lo<=_IAPWS95_NEAR_SINGLETON_ULPS*eps(max(abs(lo),abs(hi)))

function _iapws95_root(evaluate,target,lower,upper,label)
    target=float(target)
    isfinite(target) || throw(DomainError(target,"nonfinite $label target"))
    lo,hi=promote(target*0+lower,target*0+upper)
    lo<=hi || throw(ArgumentError("$label bracket bounds are reversed"))
    vlo,dlo=evaluate(lo)
    _positive(dlo,"$label lower derivative")
    lower_tolerance=16eps(typeof(target))*max(one(target),abs(target),abs(vlo))
    if lo==hi
        abs(vlo-target)<=lower_tolerance && return lo
        throw(DomainError(target,"$label outside singleton chart image"))
    end
    vhi,dhi=evaluate(hi)
    _positive(dhi,"$label upper derivative")
    upper_tolerance=16eps(typeof(target))*max(one(target),abs(target),abs(vhi))
    if !(vlo<vhi)
        _iapws95_near_singleton(lo,hi) ||
            error("$label image is not increasing on a non-degenerate bracket")
        abs(vlo-target)<=lower_tolerance && return lo
        abs(vhi-target)<=upper_tolerance && return hi
        throw(DomainError(target,"$label outside near-singleton chart image"))
    end
    target<vlo-lower_tolerance && throw(DomainError(target,"$label below chart image"))
    target>vhi+upper_tolerance && throw(DomainError(target,"$label above chart image"))
    abs(vlo-target)<=lower_tolerance && return lo
    abs(vhi-target)<=upper_tolerance && return hi
    x=(lo+hi)/2
    tolerance=8eps(typeof(target))*max(one(target),abs(target))
    for _ in 1:100
        value,derivative=evaluate(x)
        _positive(derivative,"$label derivative")
        residual=value-target
        abs(residual)<=tolerance && return x
        residual<0 ? (lo=x) : (hi=x)
        candidate=x-residual/derivative
        next=isfinite(candidate) && lo<candidate<hi ? candidate : (lo+hi)/2
        if next==x || next==lo || next==hi
            abs(residual)<=16eps(typeof(target))*max(one(target),abs(target),abs(value)) && return x
            error("$label inversion stagnated with residual $residual")
        end
        x=next
    end
    error("$label inversion did not converge")
end

"""
Invert entropy at fixed density on the restricted chart. Numerical roots are
lifted through two Newton corrections for nested ForwardDiff; accuracy depends
on the primal residual. Returned entropy_residual is in J/(kg K). Derivatives at
validity faces are those of the smooth interior extension, not face invariance.
"""
function iapws95_entropy_density_state(eos::IAPWS95EOS,entropy,density)
    _finite(entropy,"specific entropy")
    _iapws95_bound(density,IAPWS95_RHO_BOUNDS,"density")
    s0,rho0=_iapws95_primal(entropy),_iapws95_primal(density)
    evaluate(T)=begin
        st=iapws95_state(eos;temperature=T,density=rho0)
        st.entropy,st.entropy_temperature
    end
    root=_iapws95_root(evaluate,s0,IAPWS95_T_BOUNDS...,"entropy/density")
    temperature=root+zero(entropy+density)
    if entropy isa ForwardDiff.Dual || density isa ForwardDiff.Dual
        # Evaluate the smooth formula during lifting, then restore the converged
        # primal value. This avoids a numerical ulp crossing at a chart face.
        for _ in 1:2
            st=iapws95_properties(temperature,density)
            temperature-=(st.entropy-entropy)/st.entropy_temperature
        end
        temperature+=root-_iapws95_primal(temperature)
    end
    st=iapws95_state(eos;temperature,density)
    (;st...,entropy_residual=st.entropy-entropy)
end

function internal_energy(eos::IAPWS95EOS,S,M,V)
    _finite(S,"entropy"); _positive(M,"mass"); _positive(V,"volume")
    st=iapws95_entropy_density_state(eos,S/M,M/V)
    _finite(M*st.internal_energy_specific,"internal energy")
end

function _iapws95_density_at_pressure(eos,T,p)
    evaluate(rho)=begin
        st=iapws95_state(eos;temperature=T,density=rho)
        st.pressure,st.pressure_density
    end
    _iapws95_root(evaluate,p,IAPWS95_RHO_BOUNDS...,"pressure/density")
end

# Crossed fixed-pressure boundary temperatures. Each end is a root of
# p(T,rho_face)=p at an 8-eps pressure residual, or an endpoint accepted at 16 eps.
# At fixed density dT=dp/p_T, and p/p_T is of order T on this gas-like chart, so
# each end carries O(16) ulps of T. A crossing within
# _IAPWS95_INTERVAL_CROSSING_ULPS ulps of T may be roundoff: it is merged and
# reported only if the merged temperature passes both density-face pressure checks
# at the endpoint tolerance. Wider or infeasible crossings are an empty interval.
# A merged singleton still admits a target only within its image tolerance.
const _IAPWS95_INTERVAL_CROSSING_ULPS=32

function _iapws95_reconcile_interval(lower,upper,feasible)
    lower<=upper && return (;lower,upper,collapsed=false,crossing=zero(lower))
    crossing=lower-upper
    crossing<=_IAPWS95_INTERVAL_CROSSING_ULPS*eps(max(abs(lower),abs(upper))) ||
        throw(DomainError(crossing,"empty fixed-pressure chart interval (K)"))
    merged=(lower+upper)/2
    feasible(merged) ||
        throw(DomainError(crossing,"crossed fixed-pressure interval is infeasible at its merged temperature (K)"))
    (;lower=merged,upper=merged,collapsed=true,crossing)
end

# Under p_T>0 and p_rho>0 throughout the box, the feasible fixed-pressure
# temperatures form this interval. Runtime checks and grid sampling are evidence,
# not a proof that these inequalities hold between every evaluated point.
function _iapws95_pressure_interval(eos,p)
    _positive(p,"stream pressure")
    Tmin,Tmax=IAPWS95_T_BOUNDS
    rhomin,rhomax=IAPWS95_RHO_BOUNDS
    boundary(T,rho)=begin
        st=iapws95_state(eos;temperature=T,density=rho)
        _positive(st.pressure_temperature,"pressure-temperature derivative")
        st.pressure,st.pressure_temperature
    end
    lowmin=first(boundary(Tmin,rhomin)); highmax=first(boundary(Tmax,rhomax))
    lowmin<=p<=highmax || throw(DomainError(p,"pressure outside restricted IAPWS95 chart image"))
    lower=p>first(boundary(Tmin,rhomax)) ?
        _iapws95_root(T->boundary(T,rhomax),p,Tmin,Tmax,"pressure/lower temperature") : Tmin
    upper=p<first(boundary(Tmax,rhomin)) ?
        _iapws95_root(T->boundary(T,rhomin),p,Tmin,Tmax,"pressure/upper temperature") : Tmax
    tolerance=16eps(typeof(float(p)))*max(one(float(p)),abs(p))
    feasible(T)=first(boundary(T,rhomin))<=p+tolerance && first(boundary(T,rhomax))>=p-tolerance
    _iapws95_reconcile_interval(lower,upper,feasible)
end

"""
Restricted stream inverse with exactly one of entropy or enthalpy and pressure.
Primal coordinates use nested monotone solves; two coupled Dual Newton corrections
lift derivatives, then restore the converged primal pair. Returned residuals are
Pa and J/(kg K) or J/kg. No saturation/phase extension or fallback is provided.
Derivatives at validity faces (density at a chart bound, or temperature at an end
of the fixed-pressure interval) are those of the smooth interior extension; they
do not state that a face is invariant. A fixed-pressure interval whose computed
ends crossed by roundoff is merged only after a feasibility check and is reported
by `interval_collapsed` and `interval_crossing` (K). It never admits a target that
fails the singleton image tolerance.
"""
function iapws95_stream_state(eos::IAPWS95EOS;pressure,entropy=nothing,enthalpy=nothing)
    xor(isnothing(entropy),isnothing(enthalpy)) ||
        throw(ArgumentError("supply exactly one of entropy or enthalpy"))
    isentropy=!isnothing(entropy)
    target=isentropy ? _finite(entropy,"stream entropy") : _finite(enthalpy,"stream enthalpy")
    _positive(pressure,"stream pressure")
    p0,target0=_iapws95_primal(pressure),_iapws95_primal(target)
    interval=_iapws95_pressure_interval(eos,p0)
    lower,upper=interval.lower,interval.upper
    evaluate(T)=begin
        rho=_iapws95_density_at_pressure(eos,T,p0)
        st=iapws95_state(eos;temperature=T,density=rho)
        _positive(st.pressure_temperature,"pressure-temperature derivative")
        isentropy ? (st.entropy,st.cp/T) : (st.enthalpy,st.cp)
    end
    T0=_iapws95_root(evaluate,target0,lower,upper,"stream temperature")
    rho0=_iapws95_density_at_pressure(eos,T0,p0)
    temperature=T0+zero(target+pressure)
    density=rho0+zero(target+pressure)
    if target isa ForwardDiff.Dual || pressure isa ForwardDiff.Dual
        for _ in 1:2
            st=iapws95_properties(temperature,density)
            rp=st.pressure-pressure
            rv=isentropy ? st.entropy-target : st.enthalpy-target
            vr=isentropy ? st.entropy_density : st.enthalpy_density
            vt=isentropy ? st.entropy_temperature : st.enthalpy_temperature
            determinant=st.pressure_density*vt-st.pressure_temperature*vr
            _positive(determinant,"stream inverse Jacobian determinant")
            density-=(vt*rp-st.pressure_temperature*rv)/determinant
            temperature-=(st.pressure_density*rv-vr*rp)/determinant
        end
        temperature+=T0-_iapws95_primal(temperature)
        density+=rho0-_iapws95_primal(density)
    end
    st=iapws95_state(eos;temperature,density)
    target_residual=isentropy ? st.entropy-target : st.enthalpy-target
    (;st...,pressure_residual=st.pressure-pressure,target_residual,
      feasible_temperature_interval=(lower,upper),
      interval_collapsed=interval.collapsed,interval_crossing=interval.crossing)
end

stream_enthalpy(eos::IAPWS95EOS,s,p)=iapws95_stream_state(eos;pressure=p,entropy=s).enthalpy
stream_entropy(eos::IAPWS95EOS,h,p)=iapws95_stream_state(eos;pressure=p,enthalpy=h).entropy
