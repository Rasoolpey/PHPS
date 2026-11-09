# Interval methods of the limiter primitives (the window certificate, docs/active/ROA_MATHEMATICAL_FORMULATION.md 13.31,
# milestone 1). Methods for interval types only (an Interval, or a ForwardDiff dual over intervals); the Float64 and
# Float64-dual paths of primitives.jl are unchanged.
#
# On a box where a limiter's branch is decided, the branch is taken. Where the box straddles the limit:
#   - continuous limiters (clamp, clamp_mode, outband_relax: piecewise linear and continuous in their argument): the
#     value is the exact interval image and the derivative factor of the limited argument is the hull [0, 1] of the
#     two branches' factors. For a continuous, piecewise C1 function the mean-value inclusion holds with the hull of
#     the pieces' derivatives (generalised mean-value theorem), so a centred form built from these duals encloses;
#   - the non-windup integrator (the rate jumps to zero at a limit it would cross): the value is the hull of the two
#     branches (an enclosure of the set-valued field), the partials the hull of both, and the site is COUNTED in
#     IVLIMIT_JUMPS: a centred form across it is not justified by smoothness, and the caller must add the jump.
# The counters IVLIMIT_STRADDLE and IVLIMIT_JUMPS record how often each case was met (reset by the caller).

const IVLIMIT_STRADDLE = Ref(0)
const IVLIMIT_JUMPS = Ref(0)
const _IvR = IA.Interval{Float64}
const _IvAny = Union{_IvR,ForwardDiff.Dual{<:Any,_IvR}}

_ival(x::_IvR) = x
_ival(x::ForwardDiff.Dual{<:Any,_IvR}) = ForwardDiff.value(x)
_ival(x::Real) = IA.interval(x)
_mkconst(::Type{T}, c) where {T<:_IvR} = IA.interval(c)
_mkconst(::Type{ForwardDiff.Dual{Tg,_IvR,N}}, c) where {Tg,N} = ForwardDiff.Dual{Tg}(IA.interval(c), ntuple(_ -> IA.interval(0.0), N))
_scale_partials(x::_IvR, f) = x
_scale_partials(x::ForwardDiff.Dual{Tg,_IvR,N}, f) where {Tg,N} = ForwardDiff.Dual{Tg}(ForwardDiff.value(x), ForwardDiff.partials(x).values .* f)
_with_val(x::_IvR, v) = v
_with_val(x::ForwardDiff.Dual{Tg,_IvR,N}, v) where {Tg,N} = ForwardDiff.Dual{Tg}(v, ForwardDiff.partials(x))
_iclamp(V, lo, hi) = IA.interval(clamp(IA.inf(V), lo, hi), clamp(IA.sup(V), lo, hi))

# clamp to [lo, hi] (continuous): decided, or the exact image with derivative factor [0, 1]
function _ivclamp(x::T, lo::Real, hi::Real) where {T<:_IvAny}
    V = _ival(x)
    (IA.inf(V) >= lo && IA.sup(V) <= hi) && return x
    IA.inf(V) >= hi && return _mkconst(T, hi)
    IA.sup(V) <= lo && return _mkconst(T, lo)
    IVLIMIT_STRADDLE[] += 1
    return _with_val(_scale_partials(x, IA.interval(0.0, 1.0)), _iclamp(V, lo, hi))
end
Base.clamp(x::ForwardDiff.Dual{Tg,_IvR}, lo::Real, hi::Real) where {Tg} = _ivclamp(x, lo, hi)
Base.clamp(x::_IvR, lo::Real, hi::Real) = _ivclamp(x, lo, hi)
clamp_mode(rec::NoModes, v::T, lo, hi) where {T<:_IvAny} = _ivclamp(v, lo, hi)

# outband_relax: rate + (hi - s)/Tc where s > hi, + (lo - s)/Tc where s < lo (continuous in s)
function outband_relax(rec::NoModes, rate::R, s::S, lo, hi, Tc) where {R<:_IvAny,S<:_IvAny}
    V = _ival(s)
    if IA.sup(V) > hi
        if IA.inf(V) >= hi
            rate = rate + (hi - s) / Tc
        else
            IVLIMIT_STRADDLE[] += 1
            term = _scale_partials((hi - s) / Tc, IA.interval(0.0, 1.0))
            rate = rate + _with_val(term, IA.interval(min(0.0, (hi - IA.sup(V)) / Tc), 0.0))
        end
    end
    if IA.inf(V) < lo
        if IA.sup(V) <= lo
            rate = rate + (lo - s) / Tc
        else
            IVLIMIT_STRADDLE[] += 1
            term = _scale_partials((lo - s) / Tc, IA.interval(0.0, 1.0))
            rate = rate + _with_val(term, IA.interval(0.0, max(0.0, (lo - IA.inf(V)) / Tc)))
        end
    end
    return rate
end
outband_relax(rec::NoModes, rate::Real, s::S, lo, hi, Tc) where {S<:_IvAny} = outband_relax(rec, s * 0 + rate, s, lo, hi, Tc)

# nonwindup: the rate is frozen at a limit it would cross (discontinuous): decided, or the hull of both, counted
function _ivhull(a::T, b::T) where {T<:_IvR}
    return IA.hull(a, b)
end
function _ivhull(a::ForwardDiff.Dual{Tg,_IvR,N}, b::ForwardDiff.Dual{Tg,_IvR,N}) where {Tg,N}
    return ForwardDiff.Dual{Tg}(IA.hull(ForwardDiff.value(a), ForwardDiff.value(b)),
                                ntuple(i -> IA.hull(ForwardDiff.partials(a)[i], ForwardDiff.partials(b)[i]), N))
end
function nonwindup(rec::NoModes, s, rate::T, lo, hi) where {T<:_IvAny}
    S, R = _ival(s), _ival(rate)
    z = zero(rate)
    for (atlim, push) in ((IA.inf(S) >= hi, IA.inf(R) > 0), (IA.sup(S) <= lo, IA.sup(R) < 0))
        # definitely frozen
        atlim && push && return z
    end
    up_possible = IA.sup(S) >= hi && IA.sup(R) > 0
    dn_possible = IA.inf(S) <= lo && IA.inf(R) < 0
    (up_possible || dn_possible) || return rate
    IVLIMIT_JUMPS[] += 1
    return _ivhull(rate, z)
end
