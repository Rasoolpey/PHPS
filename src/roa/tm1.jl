# First-order Taylor models (the window certificate, docs/active/ROA_MATHEMATICAL_FORMULATION.md 13.31.3; reviews 90).
#
# A TM1 x stands for the set { c + a' e + d : e in [-1, 1]^n, d in r }: its value c at the box centre, a linear form a
# in the box's normalised coordinates e (shared by every quantity of one evaluation, so correlations are kept), and one
# remainder interval r. A constant has an empty a. Plain interval evaluation encloses each intermediate quantity
# separately and loses the correlations (13.31.3: steam efforts enclosed 2e4-1e5 times too wide); a TM1 keeps them to
# first order.
#
# Arithmetic: + - * exactly in the linear part, the second-order and remainder products bounded by intervals; every
# Float64 rounding of c and a is covered by adding 4 eps (|c| + sum |a|) to r. Nonlinear functions by the mean-value
# rule (review 90): with X = range(x), f(c) enclosed by Fc and f' over X enclosed by Fp, m = mid(Fp),
#   f(x) in mid(Fc) + m a'e + [ (Fc - mid(Fc)) + m r + (Fp - m) (range of x - c) ],
# valid for any f with a mean-value inclusion over X (C1; or continuous and piecewise C1 with Fp the hull of the
# pieces' derivatives). Comparisons are decided on the range of the difference, or refused (TMUndecided): a branch is
# never guessed. Functions with their own interval methods (the steam tables and inner solves) are lifted by the same
# rule from those methods' enclosures of the value at the centre and of the gradient over the range.

struct TMUndecided <: Exception
    msg::String
end
Base.showerror(io::IO, e::TMUndecided) = print(io, "TMUndecided: ", e.msg)

struct TM1 <: Real
    c::Float64
    a::Vector{Float64}
    r::IA.Interval{Float64}
end
const _TMEPS = 4 * eps(Float64)
TM1(c::Real) = TM1(Float64(c), Float64[], IA.interval(0.0))
TM1(x::TM1) = x
"""the box coordinate i of dimension n: centre c, half-width w"""
tm1_var(c::Real, w::Real, i::Int, n::Int) = (a = zeros(n); a[i] = w; TM1(Float64(c), a, IA.interval(0.0)))
_rho(x::TM1) = sum(abs, x.a; init = 0.0)
"""the range of x - c (the linear part's box plus the remainder)"""
_dev(x::TM1) = (ρ = _rho(x) * (1 + _TMEPS); IA.interval(-ρ, ρ) + x.r)
tm1_range(x::TM1) = x.c + _dev(x)
tm1_range(x::Real) = IA.interval(x)
_rnd(c, a) = (e = _TMEPS * (abs(c) + sum(abs, a; init = 0.0)); e == 0 ? IA.interval(0.0) : IA.interval(-e, e))

# linear combination alpha a1 + beta a2 of coefficient vectors that may be empty (constants)
function _lin(α, a1::Vector{Float64}, β, a2::Vector{Float64})
    isempty(a1) && return β .* a2
    isempty(a2) && return α .* a1
    n1, n2 = length(a1), length(a2)
    n1 == n2 && return α .* a1 .+ β .* a2
    # different lengths: symbols added later (e.g. a remainder turned into new symbols) are zero in the shorter one
    n = max(n1, n2); out = zeros(n)
    @inbounds for i in 1:n1; out[i] += α * a1[i]; end
    @inbounds for i in 1:n2; out[i] += β * a2[i]; end
    return out
end
"""x with its remainder interval turned into a new symbol (index k, of a total n): the same set, so subsequent linear
maps keep the cancellation of that part exactly"""
function tm1_symbolise(x::TM1, k::Int, n::Int)
    a = zeros(n); a[1:length(x.a)] .= x.a
    m = IA.mid(x.r)
    rad = max(IA.sup(x.r) - m, m - IA.inf(x.r)) * (1 + _TMEPS) + eps(Float64) * abs(m)   # covers x.r - m, rounded up
    a[k] += rad
    c = x.c + m
    return TM1(c, a, _rnd(c, Float64[]))                                                 # the rounding of c + m
end

Base.zero(::Type{TM1}) = TM1(0.0); Base.zero(::TM1) = TM1(0.0)
Base.one(::Type{TM1}) = TM1(1.0); Base.one(::TM1) = TM1(1.0)
Base.convert(::Type{TM1}, x::Real) = TM1(x)
Base.convert(::Type{TM1}, x::TM1) = x
Base.promote_rule(::Type{TM1}, ::Type{<:AbstractFloat}) = TM1
Base.promote_rule(::Type{TM1}, ::Type{<:Integer}) = TM1
Base.promote_rule(::Type{TM1}, ::Type{Bool}) = TM1
Base.promote_rule(::Type{TM1}, ::Type{<:Rational}) = TM1
Base.float(x::TM1) = x
Base.isfinite(x::TM1) = isfinite(x.c) && all(isfinite, x.a) && IA.isbounded(x.r)
Base.isnan(x::TM1) = false
Base.show(io::IO, x::TM1) = print(io, "TM1(", x.c, " ± ", _rho(x), " + ", x.r, "; range ", tm1_range(x), ")")

Base.:+(x::TM1) = x
Base.:-(x::TM1) = TM1(-x.c, -x.a, -x.r)
function Base.:+(x::TM1, y::TM1)
    c = x.c + y.c; a = _lin(1.0, x.a, 1.0, y.a)
    TM1(c, a, x.r + y.r + _rnd(c, a))
end
Base.:-(x::TM1, y::TM1) = x + (-y)
Base.:+(x::TM1, y::Real) = (c = x.c + y; TM1(c, x.a, x.r + _rnd(c, x.a)))
Base.:+(y::Real, x::TM1) = x + y
Base.:-(x::TM1, y::Real) = x + (-y)
Base.:-(y::Real, x::TM1) = (-x) + y
function Base.:*(x::TM1, s::Real)
    c = x.c * s; a = x.a .* s
    TM1(c, a, x.r * s + _rnd(c, a))
end
Base.:*(s::Real, x::TM1) = x * s
Base.:*(x::TM1, b::Bool) = b ? x : zero(x)
Base.:*(b::Bool, x::TM1) = x * b
function Base.:*(x::TM1, y::TM1)
    x === y && return _tmsquare(x)
    c = x.c * y.c
    a = _lin(x.c, y.a, y.c, x.a)
    ρx, ρy = _rho(x) * (1 + _TMEPS), _rho(y) * (1 + _TMEPS)
    Lx, Ly = IA.interval(-ρx, ρx), IA.interval(-ρy, ρy)
    r = Lx * Ly + x.c * y.r + y.c * x.r + Lx * y.r + Ly * x.r + x.r * y.r
    TM1(c, a, r + _rnd(c, a) + IA.interval(-1.0, 1.0) * (_TMEPS * (abs(x.c) * _rho(y) + abs(y.c) * _rho(x))))
end
# a square: (c + L + r)^2 = c^2 + 2 c L + [L^2 + 2 (c + L) r + r^2], with L^2 in [0, rho^2] (never negative)
function _tmsquare(x::TM1)
    c = x.c * x.c; a = (2 * x.c) .* x.a
    ρ = _rho(x) * (1 + _TMEPS); L = IA.interval(-ρ, ρ)
    r = IA.interval(0.0, ρ * ρ) + 2 * (x.c + L) * x.r + x.r^2
    TM1(c, a, r + _rnd(c, a) + IA.interval(-1.0, 1.0) * (_TMEPS * 2 * abs(x.c) * _rho(x)))
end
Base.:/(x::TM1, y::TM1) = x * inv(y)
Base.:/(x::TM1, y::Real) = _divr(x, y)
_divr(x::TM1, y::Real) = (Y = IA.interval(y); TM1(x.c / y, x.a ./ y, x.r / Y + IA.interval(-1, 1) * (_TMEPS * (abs(x.c / y) + sum(abs, x.a ./ y; init = 0.0)))))
Base.:/(y::Real, x::TM1) = y * inv(x)
Base.:+(x::TM1, r::IA.Interval{Float64}) = TM1(x.c, x.a, x.r + r)

# the mean-value rule: Fc encloses f(c), Fp encloses f' over range(x)
function _mv(x::TM1, Fc::IA.Interval{Float64}, Fp::IA.Interval{Float64})
    IA.isbounded(Fc) && IA.isbounded(Fp) || throw(TMUndecided("unbounded enclosure in the mean-value rule"))
    m = IA.mid(Fp); c = IA.mid(Fc); a = m .* x.a
    TM1(c, a, (Fc - c) + m * x.r + (Fp - m) * _dev(x) + _rnd(0.0, a))   # c = mid(Fc) is exact; Fc - c is outward
end
# the multivariate rule: Fc encloses g(c), G[j] encloses dg/dx_j over the range box
function _mvn(xs, Fc::IA.Interval{Float64}, G)
    IA.isbounded(Fc) && all(IA.isbounded, G) || throw(TMUndecided("unbounded enclosure in the mean-value rule"))
    c = IA.mid(Fc); a = Float64[]; r = Fc - c
    for (x, g) in zip(xs, G)
        m = IA.mid(g)
        a = _lin(1.0, a, m, x.a)
        r = r + m * x.r + (g - m) * _dev(x)
    end
    TM1(c, a, r + _rnd(0.0, a))
end
_I(c) = IA.interval(c)
Base.inv(x::TM1) = (X = tm1_range(x); 0 in X && throw(TMUndecided("inverse of a range containing zero: $X")); _mv(x, 1 / _I(x.c), -1 / X^2))
Base.exp(x::TM1) = _mv(x, exp(_I(x.c)), exp(tm1_range(x)))
Base.log(x::TM1) = (X = tm1_range(x); IA.inf(X) > 0 || throw(TMUndecided("log of $X")); _mv(x, log(_I(x.c)), 1 / X))
Base.sqrt(x::TM1) = (X = tm1_range(x); IA.inf(X) > 0 || throw(TMUndecided("sqrt of $X")); _mv(x, sqrt(_I(x.c)), 1 / (2 * sqrt(X))))
Base.sin(x::TM1) = _mv(x, sin(_I(x.c)), cos(tm1_range(x)))
Base.cos(x::TM1) = _mv(x, cos(_I(x.c)), -sin(tm1_range(x)))
Base.sincos(x::TM1) = (sin(x), cos(x))
Base.atan(x::TM1) = _mv(x, atan(_I(x.c)), 1 / (1 + tm1_range(x)^2))
Base.:^(x::TM1, n::Integer) = n == 0 ? one(x) : n == 1 ? x : n == 2 ? _tmsquare(x) : n < 0 ? inv(x^(-n)) : x * x^(n - 1)
function Base.:^(x::TM1, p::Real)
    isinteger(p) && return x^Int(p)
    X = tm1_range(x); IA.inf(X) > 0 || throw(TMUndecided("power of $X"))
    _mv(x, _I(x.c)^p, p * X^(p - 1))
end
Base.:^(x::TM1, y::TM1) = exp(y * log(x))
Base.hypot(x::TM1, y::TM1) = sqrt(x * x + y * y)
Base.hypot(x::TM1, y::Real) = sqrt(x * x + y * y)
Base.hypot(x::Real, y::TM1) = sqrt(x * x + y * y)
Base.abs2(x::TM1) = _tmsquare(x)
function Base.abs(x::TM1)
    X = tm1_range(x)
    IA.inf(X) >= 0 && return x
    IA.sup(X) <= 0 && return -x
    return _mv(x, abs(_I(x.c)), IA.interval(-1.0, 1.0))
end
function Base.sign(x::TM1)
    X = tm1_range(x)
    IA.inf(X) > 0 && return TM1(1.0)
    IA.sup(X) < 0 && return TM1(-1.0)
    throw(TMUndecided("sign of $X"))
end

# comparisons: decided on the range of the difference, else refused
function _cmp(x, y)
    D = tm1_range(x - y)
    return IA.inf(D), IA.sup(D)
end
function Base.:<(x::TM1, y::TM1)
    lo, hi = _cmp(x, y)
    hi < 0 && return true
    lo >= 0 && return false
    throw(TMUndecided("x < y on $(lo)..$(hi)"))
end
function Base.:<=(x::TM1, y::TM1)
    lo, hi = _cmp(x, y)
    hi <= 0 && return true
    lo > 0 && return false
    throw(TMUndecided("x <= y on $(lo)..$(hi)"))
end
Base.:<(x::TM1, y::Real) = x < TM1(y); Base.:<(x::Real, y::TM1) = TM1(x) < y
Base.:<=(x::TM1, y::Real) = x <= TM1(y); Base.:<=(x::Real, y::TM1) = TM1(x) <= y
function Base.:(==)(x::TM1, y::TM1)
    lo, hi = _cmp(x, y)
    lo == 0 == hi && return true
    (lo > 0 || hi < 0) && return false
    throw(TMUndecided("x == y on $(lo)..$(hi)"))
end
Base.:(==)(x::TM1, y::Real) = x == TM1(y); Base.:(==)(x::Real, y::TM1) = TM1(x) == y
Base.isless(x::TM1, y::TM1) = x < y
Base.isless(x::TM1, y::Real) = x < y; Base.isless(x::Real, y::TM1) = x < y
Base.isless(x::TM1, y::AbstractFloat) = x < y; Base.isless(x::AbstractFloat, y::TM1) = x < y
function Base.min(x::TM1, y::TM1)
    lo, hi = _cmp(x, y)
    hi <= 0 && return x
    lo >= 0 && return y
    return (x + y - abs(x - y)) * 0.5
end
function Base.max(x::TM1, y::TM1)
    lo, hi = _cmp(x, y)
    hi <= 0 && return y
    lo >= 0 && return x
    return (x + y + abs(x - y)) * 0.5
end
Base.min(x::TM1, y::Real) = min(x, TM1(y)); Base.min(x::Real, y::TM1) = min(TM1(x), y)
Base.max(x::TM1, y::Real) = max(x, TM1(y)); Base.max(x::Real, y::TM1) = max(TM1(x), y)
for f in (:floor, :ceil, :round, :trunc)
    @eval function Base.$f(::Type{T}, x::TM1) where {T<:Integer}
        X = tm1_range(x); a, b = $f(T, IA.inf(X)), $f(T, IA.sup(X))
        a == b || throw(TMUndecided(string($(string(f)), " of ", X)))
        return a
    end
end

# limiters (the same semantics as ivlimit.jl): decided, or the mean-value rule with slope [0, 1]; the non-windup jump
# as the hull of both values (counted)
function _tmclamp(x::TM1, lo::Real, hi::Real)
    X = tm1_range(x)
    (IA.inf(X) >= lo && IA.sup(X) <= hi) && return x
    IA.inf(X) >= hi && return TM1(hi)
    IA.sup(X) <= lo && return TM1(lo)
    IVLIMIT_STRADDLE[] += 1
    return _mv(x, IA.interval(clamp(x.c, lo, hi)), IA.interval(0.0, 1.0))
end
Base.clamp(x::TM1, lo::Real, hi::Real) = _tmclamp(x, lo, hi)
clamp_mode(rec::NoModes, v::TM1, lo, hi) = _tmclamp(v, lo, hi)
function outband_relax(rec::NoModes, rate, s::TM1, lo, hi, Tc)
    rate = TM1(0.0) + rate
    rate = rate + (_tmclamp(s, -Inf, hi) - s) / Tc            # (hi - s)/Tc where s > hi, else 0
    rate = rate + (_tmclamp(s, lo, Inf) - s) / Tc              # (lo - s)/Tc where s < lo, else 0
    return rate
end
function nonwindup(rec::NoModes, s, rate::TM1, lo, hi)
    S, R = tm1_range(TM1(0.0) + s), tm1_range(rate)
    (IA.inf(S) >= hi && IA.inf(R) > 0) && return TM1(0.0)
    (IA.sup(S) <= lo && IA.sup(R) < 0) && return TM1(0.0)
    up = IA.sup(S) >= hi && IA.sup(R) > 0
    dn = IA.inf(S) <= lo && IA.inf(R) < 0
    (up || dn) || return rate
    IVLIMIT_JUMPS[] += 1
    return rate * 0.5 + IA.interval(-1.0, 1.0) * (0.5 * IA.mag(R))   # the hull of {rate, 0}
end
_deep(x::TM1) = x

# the steam property functions, lifted from their interval methods (ivsteam.jl): the value at the centre on thin
# intervals, the gradient over the range box from duals over intervals
function _tmlift2(f, x::TM1, y::TM1, fields)
    c = f(IA.interval(x.c), IA.interval(y.c))
    X, Y = tm1_range(x), tm1_range(y)
    d = f(ForwardDiff.Dual{Nothing}(X, IA.interval(1.0), IA.interval(0.0)), ForwardDiff.Dual{Nothing}(Y, IA.interval(0.0), IA.interval(1.0)))
    vals = map(fields) do k
        dk = getfield(d, k)
        if dk isa ForwardDiff.Dual
            p = ForwardDiff.partials(dk)
            _mvn((x, y), _ivwrap(getfield(c, k)), (p[1], p[2]))
        else
            TM1(Float64(IA.mid(_ivwrap(getfield(c, k)))), Float64[], _ivwrap(getfield(c, k)) - IA.mid(_ivwrap(getfield(c, k))))
        end
    end
    return NamedTuple{fields}(Tuple(vals))
end
_ivwrap(v::IA.Interval{Float64}) = v
_ivwrap(v::Real) = IA.interval(v)
const TM1_STEAM_DIRECT = Ref(0)
const TM1_STEAM_LIFTED = Ref(0)
# The lift is used throughout: running the tables directly in TM1 arithmetic was tried and is far worse (13.31.4: each
# Hermite basis function's second-order part, scaled by table values near 1e6, goes to the remainder although the parts
# cancel in the sum; remainders 1e5 times the lift's).
function steam_state(tb::SteamTables, rho::TM1, u::TM1)
    TM1_STEAM_LIFTED[] += 1
    return _tmlift2((r, v) -> steam_state(tb, r, v), rho, u, (:T, :p, :h, :s, :x))
end
function steam_h_isentropic(tb::SteamTables, p::TM1, s::TM1)
    TM1_STEAM_LIFTED[] += 1
    return _tmlift2((a, b) -> steam_h_isentropic(tb, a, b), p, s, (:h, :T, :x))
end
steam_h_isentropic(tb::SteamTables, p::Real, s::TM1) = steam_h_isentropic(tb, TM1(p), s)
function _steam_opening(g::PHSteamUnit, d::TM1)
    c = _steam_opening(g, IA.interval(d.c))
    D = _steam_opening(g, ForwardDiff.Dual{Nothing}(tm1_range(d), IA.interval(1.0)))
    _mv(d, _ivwrap(c), ForwardDiff.partials(D)[1])
end

# The voltage of a reduced node with a machine: the load current in the network frame is the same Kirchhoff sum
# rotated once, -j e^{j delta} i_L(rotor frame) (an exact identity, checked to 1e-15 at stored and off-constraint
# states, 13.32), instead of the sum of separately rotated large currents (f.iload), whose rotation remainders do not
# cancel in TM1 (G2's terminal voltage: remainder 10.7 against 1e-2 with the identity).
function _cut_voltage(s::PHSystem, ci::Int, f, dx::AbstractVector{TM1})
    c = s.cuts[ci]
    iL, diL = _cut_load_rate(s, ci, f, dx)
    c.machine > 0 || return c.LL * diL / s.omega_b + complex(c.RL, c.LL) * f.iload[ci]
    m, q = s.machines[c.machine], f.mc[c.machine]
    diL = -im * q.rot * (diL + im * dx[_idelta(m)] * iL)
    return c.LL * diL / s.omega_b + complex(c.RL, c.LL) * (-im * q.rot * iL)
end

# The brushless rectifier's regulation curve (exciter.jl `_fex`): continuous, four pieces (mode I to IN = sqrt(3)/4,
# II to 3/4, III to 1, zero beyond). Inside one piece the formula in TM1; across pieces the mean-value rule with the
# hull of the pieces' slopes over the parts of the range they cover (-1/sqrt 3; -IN/sqrt(3/4 - IN^2); -sqrt 3; 0).
function _fex(IN::TM1)
    R = tm1_range(IN); lo, hi = IA.inf(R), IA.sup(R)
    b1, b2 = sqrt(3) / 4, 0.75
    hi <= b1 && return 1 - IN / sqrt(3)
    (lo > b1 && hi < b2) && return sqrt(0.75 - IN^2)
    (lo >= b2 && hi < 1) && return sqrt(3) * (1 - IN)
    lo >= 1 && return TM1(0.0)
    IVLIMIT_STRADDLE[] += 1
    c = IN.c
    Fc = c <= b1 ? 1 - IA.interval(c) / sqrt(IA.interval(3.0)) : c < b2 ? sqrt(0.75 - IA.interval(c)^2) :
         c < 1 ? sqrt(IA.interval(3.0)) * (1 - IA.interval(c)) : IA.interval(0.0)
    S = IA.emptyinterval(Float64)
    lo <= b1 && (S = IA.hull(S, -1 / sqrt(IA.interval(3.0))))
    if hi > b1 && lo < b2
        X = IA.interval(max(lo, b1), min(hi, b2 - 1e-12))
        S = IA.hull(S, -X / sqrt(0.75 - X^2))
    end
    (hi >= b2 && lo < 1) && (S = IA.hull(S, -sqrt(IA.interval(3.0))))
    hi >= 1 && (S = IA.hull(S, IA.interval(0.0)))
    return _mv(IN, Fc, S)
end
