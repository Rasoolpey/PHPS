# Interval methods for the steam unit's table lookups and inner solves (the practical-return proof, step P0;
# docs/active/ROA_MATHEMATICAL_FORMULATION.md section 7; the survey docs/research/interval_field_survey_2026-10-24.md).
#
# Only methods for interval arguments (an Interval, or a ForwardDiff dual over intervals) are added: the Float64 and
# Float64-dual paths of steamprops.jl and steam.jl are unchanged. Every method here encloses: its result contains the
# value (and, for a dual, the derivative) of the Float64 code at every point of the box, or it throws ProofFailure
# ("not proved", never "false").
#
#   - A table lookup takes the hull, over every cell the argument meets, of that cell's own Hermite polynomial on the
#     argument restricted to the cell (review 49 item 4). The tables are C1 across cells, so the hull of the cellwise
#     derivative enclosures encloses the derivative too.
#   - The saturation pressure of saturated vapour at a specific volume (`_sat_l_of_vg`) is enclosed at the two ends of
#     the volume box by interval Newton, and the map is proved monotone on their hull.
#   - The wet state's pressure (`_wet_state`) is enclosed by interval Newton on its residual over the box of (v, u);
#     for a dual, its derivative by the implicit function theorem over that enclosure.
#   - The valve characteristic (`_steam_opening`) takes the hull over its pieces and cells.
# Phase and branch decisions in the unchanged code compare intervals: decided on the whole box, or an
# InconclusiveBooleanOperation, which the callers report as "not proved".

const _IvD = Union{IA.Interval{Float64},ForwardDiff.Dual{<:Any,IA.Interval{Float64}}}

_ivalue(x::IA.Interval) = x
_ivalue(x::ForwardDiff.Dual) = ForwardDiff.value(x)
_ivalue(x::Real) = ival(x)
_with_value(x::IA.Interval, v::IA.Interval) = v
_with_value(x::ForwardDiff.Dual{T}, v::IA.Interval) where {T} = ForwardDiff.Dual{T}(v, ForwardDiff.partials(x))
# the hull of two enclosures, as a fresh interval (IA.hull lowers the decoration to trv by the interval standard); an
# input that is not defined and continuous on its box makes the hull fail
function _ihull(a::IA.Interval, b::IA.Interval)
    (IA.decoration(a) >= IA.dac && IA.decoration(b) >= IA.dac) || fail("ivsteam: hull of an enclosure that is not well defined")
    return ival(min(IA.inf(a), IA.inf(b)), max(IA.sup(a), IA.sup(b)))
end
_ihull(a::ForwardDiff.Dual{T}, b::ForwardDiff.Dual{T}) where {T} =
    ForwardDiff.Dual{T}(_ihull(ForwardDiff.value(a), ForwardDiff.value(b)),
                        ForwardDiff.Partials(map(_ihull, ForwardDiff.partials(a).values, ForwardDiff.partials(b).values)))
_ihull(a::NamedTuple{K}, b::NamedTuple{K}) where {K} = NamedTuple{K}(map(_ihull, values(a), values(b)))
_ihull(a, b::ForwardDiff.Dual) = _ihull(convert(typeof(b), a), b)
_ihull(a::ForwardDiff.Dual, b) = _ihull(a, convert(typeof(a), b))
_ihull(a::Tuple, b::Tuple) = map(_ihull, a, b)
# a fresh interval [lo, hi] (decoration com)
_iclip(v::IA.Interval, lo::Float64, hi::Float64) = (l = max(IA.inf(v), lo); h = min(IA.sup(v), hi);
                                                     l <= h || fail("ivsteam: empty restriction"); ival(l, h))
_inside_open(k::IA.Interval, x::IA.Interval) = IA.inf(x) < IA.inf(k) && IA.sup(k) < IA.sup(x)

# the cells of a uniform grid that x meets, each with the local coordinate t restricted to that cell
function _icells(x, x0::Float64, dx::Float64, n::Int, what; below = 1e-9)
    x isa _IvD || (x = ival(x))
    tt = (_ivalue(x) - x0) / dx
    lo, hi = IA.inf(tt), IA.sup(tt)
    (isfinite(lo) && isfinite(hi) && lo >= -below && hi <= n - 1 + 1e-9) ||
        fail("steam tables: $what not proved inside the tabulated range [$x0, $(x0 + (n - 1) * dx)]")
    i0, i1 = clamp(floor(Int, lo), 0, n - 2), clamp(floor(Int, hi), 0, n - 2)
    return map(i0:i1) do i
        t = (x - x0) / dx - i
        tlo = i == 0 ? -below : 0.0
        thi = i == n - 2 ? 1.0 + 1e-9 : 1.0
        (i + 1, _with_value(t, _iclip(_ivalue(t), tlo, thi)))
    end
end

# Each cell's polynomial is a cubic (bicubic) in the cell coordinate t = (x - x0)/dx - i, so it equals its own Taylor
# expansion of order 3 about the midpoint tm of the coordinate box: P(t) = sum_{d<=3} P^(d)(tm) (t - tm)^d / d!, exactly.
# The coefficients are evaluated at the thin point tm in interval arithmetic, and every term carries the small factor
# (t - tm). This avoids the dependency of the term-by-term evaluation, which cancels large table values against each
# other and overestimates by orders of magnitude. For a dual t the same polynomial gives exact partials.
# The Hermite basis (h00, h10, h01, h11) and its derivatives of order d = 0..3 at a thin point:
function _hbasis(t, d)
    d == 0 && return (2t^3 - 3t^2 + 1, t^3 - 2t^2 + t, -2t^3 + 3t^2, t^3 - t^2)
    d == 1 && return (6t^2 - 6t, 3t^2 - 4t + 1, -6t^2 + 6t, 3t^2 - 2t)
    d == 2 && return (12t - 6, 6t - 4, -12t + 6, 6t - 2)
    return (12 + 0t, 6 + 0t, -12 + 0t, 6 + 0t)
end
const _FACT = (1.0, 1.0, 2.0, 6.0)
# P^(d)(tm) of a 1-D cell, d = 0..3, in the cell coordinate
_cub_coeffs(tm, V0, V1, D0, D1, dx) = ntuple(4) do d
    h = _hbasis(ival(tm), d - 1)
    h[1] * V0 + h[3] * V1 + dx * (h[2] * D0 + h[4] * D1)
end
# the cubic with coefficients c (derivatives at tm) at t: sum c_d (t - tm)^d / d!; and its t-derivative
_taylor3(c, t, tm) = (dt = t - tm; c[1] + dt * (c[2] + dt * (c[3] / 2 + dt * (c[4] / 6))))
_dtaylor3(c, t, tm) = (dt = t - tm; c[2] + dt * (c[3] + dt * (c[4] / 2)))

function _interp(tab::Hermite1, k::Int, x::_IvD, what)
    r = nothing
    for (i, t) in _icells(x, tab.x0, tab.dx, tab.n, what)
        tm = IA.mid(_ivalue(t))
        c = _cub_coeffs(tm, tab.V[k, i], tab.V[k, i+1], tab.D[k, i], tab.D[k, i+1], tab.dx)
        v = _taylor3(c, t, tm)
        r = r === nothing ? v : _ihull(r, v)
    end
    return r
end

function _interp_all(tab::Hermite1, x::_IvD, what)
    r = nothing
    m = size(tab.V, 1)
    for (i, t) in _icells(x, tab.x0, tab.dx, tab.n, what)
        tm = IA.mid(_ivalue(t))
        cs = ntuple(k -> _cub_coeffs(tm, tab.V[k, i], tab.V[k, i+1], tab.D[k, i], tab.D[k, i+1], tab.dx), m)
        val = ntuple(k -> _taylor3(cs[k], t, tm), m)
        der = ntuple(k -> _dtaylor3(cs[k], t, tm) / tab.dx, m)          # d/dx = (1/dx) d/dt
        r = r === nothing ? (val, der) : (_ihull(r[1], val), _ihull(r[2], der))
    end
    return r
end

# a bicubic cell: the 16 mixed derivatives d^(d+e) P / dt^d ds^e, d, e = 0..3, at (tm, sm) in the cell coordinates,
# then the full tensor expansion sum_{d,e<=3} (all mixed terms): exact, since the bicubic has degree 3 in each variable
function _interp2_iv(tab::Hermite2, k::Int, x, y, what, below)
    r = nothing
    for (i, t) in _icells(x, tab.x0, tab.dx, tab.nx, what * " (first coordinate)")
        tm = IA.mid(_ivalue(t))
        ht = ntuple(d -> _hbasis(ival(tm), d - 1), 4)
        for (j, s) in _icells(y, tab.y0, tab.dy, tab.ny, what * " (second coordinate)"; below)
            sm = IA.mid(_ivalue(s))
            hs = ntuple(e -> _hbasis(ival(sm), e - 1), 4)
            dt, ds = t - tm, s - sm
            # explicit powers: dt^0 through ForwardDiff differentiates as 0 * dt^-1, which is undefined on a box
            # containing dt = 0
            pt = (one(dt), dt, dt * dt, dt * dt * dt); ps = (one(ds), ds, ds * ds, ds * ds * ds)
            v = zero(dt * ds)
            for d in 1:4, e in 1:4
                a, b = ht[d], hs[e]           # (h00, h10, h01, h11) derivatives of order d-1, e-1
                c = zero(a[1])
                for p in 0:1, q in 0:1
                    ii, jj = i + p, j + q
                    A, C = p == 0 ? (a[1], tab.dx * a[2]) : (a[3], tab.dx * a[4])
                    B, D = q == 0 ? (b[1], tab.dy * b[2]) : (b[3], tab.dy * b[4])
                    c += A * B * tab.F[k, ii, jj] + C * B * tab.Fx[k, ii, jj] + A * D * tab.Fy[k, ii, jj] +
                         C * D * tab.Fxy[k, ii, jj]
                end
                v += c / (_FACT[d] * _FACT[e]) * pt[d] * ps[e]
            end
            r = r === nothing ? v : _ihull(r, v)
        end
    end
    return r
end

_interp(tab::Hermite2, k::Int, x::_IvD, y::_IvD, what; below = 1e-9) = _interp2_iv(tab, k, x, y, what, below)
_interp(tab::Hermite2, k::Int, x::_IvD, y::Real, what; below = 1e-9) = _interp2_iv(tab, k, x, y, what, below)
_interp(tab::Hermite2, k::Int, x::Real, y::_IvD, what; below = 1e-9) = _interp2_iv(tab, k, x, y, what, below)

# l = log p of saturated vapour with specific volume vv (a Float64), enclosed by interval Newton around the Float64 root
function _sat_l_enclose(tb::SteamTables, vv::Float64)
    l = _sat_l_of_vg(tb, vv)
    l === nothing && fail("steam tables: saturated-vapour volume $vv outside the table")
    f = _interp(tb.sat, 3, ival(l), "saturation (log Pa)") - vv
    del = 1e-12 * max(1.0, abs(l))
    for _ in 1:30
        L = ival(l - del, l + del)
        _, der = _interp_all(tb.sat, L, "saturation (log Pa)")
        0 in der[3] && fail("steam tables: v_g not proved monotone near l = $l")
        N = ival(l) - f / der[3]
        _inside_open(N, L) && return N
        del = 4 * max(del, IA.mag(N - l))
    end
    fail("steam tables: no enclosure of the saturation pressure at v_g = $vv")
end

# over a box of specific volumes: v_g decreases with l, so l_g(v) lies between its values at the box's ends
function _sat_l_of_vg(tb::SteamTables, v::IA.Interval{Float64})
    L = _ihull(_sat_l_enclose(tb, IA.sup(v)), _sat_l_enclose(tb, IA.inf(v)))
    _, der = _interp_all(tb.sat, L, "saturation (log Pa)")
    IA.sup(der[3]) < 0 || fail("steam tables: v_g not proved decreasing on $L")
    return L
end

# the wet state over a box of (v, u): its pressure by interval Newton on the residual R(l; v, u), derivatives by the
# implicit function theorem (R_l over the enclosure, R_u = 1, R_v = -(u_g - u_f)/(v_g - v_f))
function _wet_state(tb::SteamTables, v::_IvD, u::_IvD, lg; strict::Bool = true)
    V, U = _ivalue(v), _ivalue(u)
    vm, um = IA.mid(V), IA.mid(U)
    lgm = _sat_l_of_vg(tb, vm)
    lgm === nothing && fail("steam tables: no dry boundary at v = $vm")
    lt = log(_wet_state(tb, vm, um, lgm).p)
    f, _ = _wet_resid(tb, ival(lt), V, U)
    del = 1e-12 * max(1.0, abs(lt))
    L = nothing
    for _ in 1:30
        Lb = ival(lt - del, lt + del)
        _, d = _wet_resid(tb, Lb, V, U)
        0 in d && fail("steam_state: the wet residual's slope is not proved nonzero near l = $lt")
        N = ival(lt) - f / d
        if _inside_open(N, Lb)
            L = N
            break
        end
        del = 4 * max(del, IA.mag(N - lt))
    end
    L === nothing && fail("steam_state: no enclosure of the wet-state pressure (v = $vm, u = $um)")
    ll = _wet_lift(tb, L, v, u, V, U)
    s, _ = _interp_all(tb.sat, ll, "wet state (log Pa)")
    x = (v - s[2]) / (s[3] - s[2])
    # the quality's range check of the Float64 code; when the box also holds vapour points (strict = false), the wet
    # branch is evaluated beyond the dry boundary on a superset, so only the lower end is checked
    IA.inf(_ivalue(x)) >= -1e-9 || fail("steam_state: wet-state quality not proved >= 0")
    strict && (IA.sup(_ivalue(x)) <= 1 + 1e-6 || fail("steam_state: wet-state quality not proved <= 1"))
    p = exp(ll)
    return (; T = s[1], p, h = u + p * v, s = s[8] + x * (s[9] - s[8]), x)
end
_wet_lift(tb, L, v::IA.Interval, u::IA.Interval, V, U) = L
function _wet_lift(tb, L, v::ForwardDiff.Dual{T}, u::ForwardDiff.Dual{T}, V, U) where {T}
    s, d = _interp_all(tb.sat, L, "wet state (log Pa)")
    _, Rl = _wet_resid(tb, L, V, U)
    Rv = -(s[5] - s[4]) / (s[3] - s[2])
    pv, pu = ForwardDiff.partials(v), ForwardDiff.partials(u)
    return ForwardDiff.Dual{T}(L, ForwardDiff.Partials(map((a, b) -> -(Rv * a + b) / Rl, pv.values, pu.values)))
end

# A cubic Hermite cell on [d0, d0 + h] with values y0, y1 and slopes s0, s1, in t = (d - d0)/h:
#   p(t) = (2t^3 - 3t^2 + 1) y0 + (t^3 - 2t^2 + t) h s0 + (-2t^3 + 3t^2) y1 + (t^3 - t^2) h s1,
#   dp/dd = q(t) = A t^2 + B t + C,  A = (6 y0 + 3 h s0 - 6 y1 + 3 h s1)/h,  B = (-6 y0 - 4 h s0 + 6 y1 - 2 h s1)/h,
#   C = s0  (q(0) = s0, q(1) = s1). All in interval arithmetic, so the ranges are rigorous.
_herm_p(t, y0, y1, s0, s1, h) = (2t^3 - 3t^2 + 1) * y0 + (t^3 - 2t^2 + t) * h * s0 + (-2t^3 + 3t^2) * y1 + (t^3 - t^2) * h * s1
function _herm_q(y0, y1, s0, s1, h)
    Y0, Y1, S0, S1, Hh = ival(y0), ival(y1), ival(s0), ival(s1), ival(h)
    return ((6Y0 + 3Hh * S0 - 6Y1 + 3Hh * S1) / Hh, (-6Y0 - 4Hh * S0 + 6Y1 - 2Hh * S1) / Hh, S0)
end
# the range of q over [ta, tb] (ta <= tb in [0, 1]): the ends, and the vertex value C - B^2/(4A) when the vertex can
# lie inside
function _herm_slope_range(ta::Float64, tb::Float64, y0, y1, s0, s1, h)
    A, B, C = _herm_q(y0, y1, s0, s1, h)
    q(t) = (A * t + B) * t + C
    r = _ihull(q(ival(ta)), q(ival(tb)))
    if !IA.in_interval(0.0, A)
        tv = -B / (2A)
        IA.sup(tv) >= ta && IA.inf(tv) <= tb && (r = _ihull(r, C - B^2 / (4A)))
    else
        r = _ihull(r, q(ival(ta, tb)))                               # A may vanish: the naive form, still an enclosure
    end
    return r
end
# the cell is non-decreasing when q >= 0 on [0, 1] (proved: the lower bound of its range)
_herm_monotone(y0, y1, s0, s1, h) = IA.inf(_herm_slope_range(0.0, 1.0, y0, y1, s0, s1, h)) >= 0
# the exact range of a non-decreasing cell over the demands Dc (inside the cell): the value from its two ends, the
# slope by _herm_slope_range; for a dual, the slope times the demand's partials
function _herm_range(dc, Dc::IA.Interval, d0, h, y0, y1, s0, s1)
    T = (Dc - ival(d0)) / ival(h)
    ta, tb = max(IA.inf(T), 0.0), min(IA.sup(T), 1.0)
    v = ival(IA.inf(_herm_p(ival(ta), y0, y1, s0, s1, h)), IA.sup(_herm_p(ival(tb), y0, y1, s0, s1, h)))
    dc isa ForwardDiff.Dual || return v
    sl = _herm_slope_range(ta, tb, y0, y1, s0, s1, h)
    return typeof(dc)(v, sl * ForwardDiff.partials(dc))
end

# the valve characteristic over a box of demands: the hull over its pieces (proportional below the lowest node, cubic
# Hermite cells between nodes, constant above the highest)
function _steam_opening(g::PHSteamUnit, d::_IvD)
    g.valve === :opening && return d / g.PMAX
    dd, yy, ss = g.vc_d, g.vc_y, g.vc_s
    isempty(dd) && error("PHSteamUnit $(g.name): the valve characteristic is built by steam_initialise")
    D = _ivalue(d)
    # the characteristic is only C0 at its end nodes: a derivative evaluation must not straddle them (review 50 item 8)
    d isa ForwardDiff.Dual && ((IA.inf(D) <= dd[1] <= IA.sup(D)) || (IA.inf(D) <= dd[end] <= IA.sup(D))) &&
        fail("valve characteristic: a derivative evaluation over a box that straddles an end node")
    r = nothing
    put!(v) = (r = r === nothing ? v : _ihull(r, v))
    if IA.inf(D) <= dd[1]
        dl = _with_value(d, _iclip(D, -Inf, dd[1]))
        put!(yy[1] * dl / dd[1])
    end
    if IA.sup(D) >= dd[end]
        put!(yy[end] + zero(d))
    end
    lo, hi = max(IA.inf(D), dd[1]), min(IA.sup(D), dd[end])
    if lo <= hi
        n = length(dd)
        i0 = min(searchsortedlast(dd, lo), n - 1); i1 = min(searchsortedlast(dd, hi), n - 1)
        for i in max(i0, 1):max(i1, 1)
            Dc = IA.inf(D) <= dd[i+1] && IA.sup(D) >= dd[i] ? _iclip(D, dd[i], dd[i+1]) : continue
            dc = _with_value(d, Dc)
            h = dd[i+1] - dd[i]
            if _herm_monotone(yy[i], yy[i+1], ss[i], ss[i+1], h)
                # a non-decreasing cell: the exact range of the value (its two ends) and of the slope (a quadratic in
                # t), instead of the naive form, whose t, t^2, t^3 lose their correlation (fault envelope, 12.4.1)
                put!(_herm_range(dc, Dc, dd[i], h, yy[i], yy[i+1], ss[i], ss[i+1]))
                continue
            end
            t = (dc - dd[i]) / h
            t2, t3 = t^2, t^3
            put!((2t3 - 3t2 + 1) * yy[i] + (t3 - 2t2 + t) * h * ss[i] + (-2t3 + 3t2) * yy[i+1] + (t3 - t2) * h * ss[i+1])
        end
    end
    r === nothing && fail("valve characteristic: no piece met by $D")
    return r
end

# The equilibrium state over a box of (rho, u). The Float64 code decides the phase by
#   vapour if u >= u_g(X) (1 + 1e-6)  [X = log rho, the gas table], else
#   vapour if u >= u_g,sat(l_g(v))     [l_g the saturated-vapour pressure at v = 1/rho], else wet.
# Over a box, a decided phase takes its branch; an undecided one encloses both: the vapour branch over the part of the
# box with u >= min(u_g(X)(1 + 1e-6), u_g,sat(l_g)) and the wet branch over the part with u <= u_g,sat(l_g), each
# a superset of the points that take that branch, and the result is their hull. The branches agree on the dry boundary
# to the tables' interpolation error, so the hull is not much wider than either.
const IVSTEAM_HULL_ACROSS_DRY = Ref(false)
const IVSTEAM_DRY_CROSSINGS = Ref(0)
function steam_state(tb::SteamTables, rho::_IvD, u::_IvD)
    X = log(rho)
    ug = _interp(tb.gas, 2, X, "density (log kg/m3)")
    U, UG = _ivalue(u), _ivalue(ug) * (1 + 1e-6)
    IA.inf(U) >= IA.sup(UG) && return _vapour_state(tb, rho, u, X, ug)
    v = 1 / rho
    V = _ivalue(v)
    vlo, vhi = _interp(tb.sat, 3, tb.sat.x0, ""), _interp(tb.sat, 3, tb.sat.x0 + (tb.sat.n - 1) * tb.sat.dx, "")
    if IA.sup(V) < vhi || IA.inf(V) > vlo                            # beyond the table's pressures: vapour
        return _vapour_state(tb, rho, u, X, ug)
    end
    (vhi <= IA.inf(V) && IA.sup(V) <= vlo) || fail("steam_state: the box straddles the saturation table's volume range")
    L = _sat_l_of_vg(tb, V)
    UGS = _ivalue(_sat_ug(tb, L))
    IA.inf(U) >= IA.sup(UGS) && return _vapour_state(tb, rho, u, X, ug)  # at or beyond the dry boundary
    wet_only = IA.sup(U) < IA.inf(UGS) && IA.sup(U) < IA.inf(UG)
    # the field is not C1 across the dry boundary (the two branches meet to the tables' interpolation error only), so a
    # derivative evaluation, which a mean-value form relies on, must not straddle it (review 50 item 8)
    # with IVSTEAM_HULL_ACROSS_DRY[] (the window certificate, 13.31) the derivative is the hull of both branches and the
    # crossing is COUNTED in IVSTEAM_DRY_CROSSINGS: the caller adds the jump between the branches to its bound
    if !wet_only && u isa ForwardDiff.Dual
        IVSTEAM_HULL_ACROSS_DRY[] || fail("steam_state: a derivative evaluation over a box that straddles the dry boundary")
        IVSTEAM_DRY_CROSSINGS[] += 1
    end
    wet = _wet_state(tb, v, wet_only ? u : _with_value(u, _iclip(U, -Inf, IA.sup(UGS))), L; strict = wet_only)
    wet_only && return wet
    uv = _with_value(u, _iclip(U, min(IA.inf(UG), IA.inf(UGS)), Inf))
    return _ihull(_vapour_state(tb, rho, uv, X, ug), wet)
end
