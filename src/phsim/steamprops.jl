# Steam properties for the physical steam units of the EMT simulator, from IAPWS-95 through
# tables (docs/active/PH_STEAM_UNIT_MODEL.md, section 15.4).
#
# The simulator steps at microseconds and differentiates its whole vector field, so the steam
# unit cannot invert IAPWS-95 at every evaluation. Everything it needs is tabulated once, on
# first use, from the repository's IAPWS-95 kernel (`PhysicalSteam.iapws95_properties`, release
# IAPWS R6-95(2018)), with exact derivatives at the grid points, and read back by cubic or bicubic
# Hermite interpolation:
#
#   saturation, as functions of l = log(p / Pa):  T, v_f, v_g, u_f, u_g, h_f, h_g, s_f, s_g
#   saturated vapour against density, X = log(rho / (kg/m3)):  T_g, u_g
#   the vapour map (X, xi) -> (T, p, s), h = u + p/rho exactly, with u = u_g(X) + xi (u_top(X) - u_g(X)) and u_top the
#       internal energy at T_TOP: the saturation boundary is the grid line xi = 0
#   the isentrope (l, eta) -> (h, T), with s = s_g(l) + eta (s_top(l) - s_g(l))
#
# Inside the dome a state is the equilibrium mixture (`steam_state`) and an isentropic end point
# is h_f + T_sat (s - s_f) (`steam_h_isentropic`). The tables are IAPWS-95 to their interpolation
# error, which the tests measure against the kernel; separately tabulated maps are not exactly
# thermodynamically consistent with each other. Outside the tables' range a lookup raises a
# DomainError; nothing is extrapolated.
#
# Inner solves are done on primal values and then lifted: two Newton corrections in the caller's
# number type give exact first and second derivatives for ForwardDiff (as the IAPWS-95 provider
# does). The tables themselves are plain Float64 arrays; lookups are generic in the number type.

const _PS = PhysicalSteam
const STEAM_T_TOP = 1000.0            # K, upper edge of the vapour tables
const STEAM_P_CRIT = 22.064e6         # Pa, IAPWS-95 critical pressure
const STEAM_T_TRIPLE = 273.16
const STEAM_P_TRIPLE = 611.657

_deep(x) = x
_deep(x::ForwardDiff.Dual) = _deep(ForwardDiff.value(x))

# ------------------------------------------------------------------------------------------------
# Kernel evaluations without the positivity checks (Newton iterates may pass through metastable
# states), from the Helmholtz parts: pressure, Gibbs energy, internal energy, enthalpy, entropy.
function _ps_core(T, rho)
    d = rho / _PS.IAPWS95_RHOC
    t = _PS.IAPWS95_TC / T
    pt = _PS.iapws95_helmholtz_parts(d, t)
    dr = ForwardDiff.derivative(dd -> _PS.iapws95_helmholtz_parts(dd, t).residual, d)
    phi_tau = ForwardDiff.derivative(tt -> (q = _PS.iapws95_helmholtz_parts(d, tt); q.ideal + q.residual), t)
    R = _PS.IAPWS95_R
    p = rho * R * T * (1 + d * dr)
    u = R * T * t * phi_tau
    h = R * T * (1 + t * phi_tau + d * dr)
    s = R * (t * phi_tau - pt.ideal - pt.residual)
    g = R * T * (1 + pt.ideal + pt.residual + d * dr)
    return (; p, u, h, s, g)
end

# ------------------------------------------------------------------------------------------------
# Saturation at pressure p: (T, rho_l, rho_v) with equal pressure and Gibbs energy. Primal solve by
# Newton from a guess, then two corrections in the caller's type.
function _sat_residual(x, p)
    l = _ps_core(x[1], x[2]); v = _ps_core(x[1], x[3])
    RT = _PS.IAPWS95_R * x[1]
    return [(l.p - p) / (x[2] * RT), (v.p - p) / (x[3] * RT), (l.g - v.g) / RT]
end

function _sat_newton(p, guess; iters = 60, tol = 1e-13)
    x = collect(guess)
    for _ in 1:iters
        r = _sat_residual(x, p)
        J = ForwardDiff.jacobian(y -> _sat_residual(y, p), x)
        x = x .- J \ r
        maximum(abs, r) < tol && break
    end
    maximum(abs, _sat_residual(x, p)) < 1e-10 || error("steam saturation: no convergence at p = $p Pa")
    abs(x[2] - x[3]) > 1e-3 * x[2] || error("steam saturation: trivial solution at p = $p Pa")
    # the physical branches: positive temperature, the liquid denser than the vapour
    (x[1] > 0 && x[2] > x[3] > 0) || error("steam saturation: not a liquid-vapour pair at p = $p Pa")
    # both phases mechanically stable: pressure increasing with density
    for r in (x[2], x[3])
        ForwardDiff.derivative(z -> _ps_core(x[1], z).p, r) > 0 || error("steam saturation: an unstable phase at p = $p Pa")
    end
    return x
end

# lift a primal solution x0 of r(x, a) = 0 to the type of the parameter a (two Newton corrections)
function _lift(r, x0, a)
    x = convert(Vector{typeof(a * one(eltype(x0)))}, x0)
    for _ in 1:2
        J = ForwardDiff.jacobian(y -> r(y, _deep(a)), _deep.(x))
        x = x .- J \ r(x, a)
    end
    return x
end

# ------------------------------------------------------------------------------------------------
# Hermite interpolation on uniform grids. 1-D: values V[k, i] and derivatives D[k, i] at
# x0 + (i - 1) dx. 2-D: values and the x, y and mixed derivatives at the grid points.
struct Hermite1
    x0::Float64
    dx::Float64
    n::Int
    V::Matrix{Float64}
    D::Matrix{Float64}
end

struct Hermite2
    x0::Float64
    dx::Float64
    nx::Int
    y0::Float64
    dy::Float64
    ny::Int
    F::Array{Float64,3}       # [output, ix, iy]
    Fx::Array{Float64,3}
    Fy::Array{Float64,3}
    Fxy::Array{Float64,3}
end

@inline _h00(t) = (1 + 2t) * (1 - t)^2
@inline _h10(t) = t * (1 - t)^2
@inline _h01(t) = t^2 * (3 - 2t)
@inline _h11(t) = t^2 * (t - 1)

@inline function _cell(x, x0, dx, n, what; below = 1e-9)
    xp = _deep(x)
    tt = (xp - x0) / dx
    (isfinite(tt) && -below <= tt <= n - 1 + 1e-9) ||
        throw(DomainError(xp, "steam tables: $what outside the tabulated range [$x0, $(x0 + (n - 1) * dx)]"))
    i = clamp(floor(Int, tt), 0, n - 2)
    return i + 1, (x - x0) / dx - i
end

function _interp(tab::Hermite1, k::Int, x, what)
    i, t = _cell(x, tab.x0, tab.dx, tab.n, what)
    return _h00(t) * tab.V[k, i] + _h01(t) * tab.V[k, i+1] +
           tab.dx * (_h10(t) * tab.D[k, i] + _h11(t) * tab.D[k, i+1])
end

function _interp(tab::Hermite2, k::Int, x, y, what; below = 1e-9)
    i, t = _cell(x, tab.x0, tab.dx, tab.nx, what * " (first coordinate)")
    j, s = _cell(y, tab.y0, tab.dy, tab.ny, what * " (second coordinate)"; below)
    a = (_h00(t), _h01(t)); b = (_h00(s), _h01(s))
    c = (tab.dx * _h10(t), tab.dx * _h11(t)); d = (tab.dy * _h10(s), tab.dy * _h11(s))
    r = zero(a[1] * b[1])
    @inbounds for p in 0:1, q in 0:1
        ii, jj = i + p, j + q
        r += a[p+1] * b[q+1] * tab.F[k, ii, jj] + c[p+1] * b[q+1] * tab.Fx[k, ii, jj] +
             a[p+1] * d[q+1] * tab.Fy[k, ii, jj] + c[p+1] * d[q+1] * tab.Fxy[k, ii, jj]
    end
    return r
end

# all outputs of a 1-D table and their derivatives at x (one cell lookup)
@inline _dh00(t) = 6t * (t - 1)
@inline _dh10(t) = (1 - t) * (1 - 3t)
@inline _dh01(t) = 6t * (1 - t)
@inline _dh11(t) = t * (3t - 2)
function _interp_all(tab::Hermite1, x, what)
    i, t = _cell(x, tab.x0, tab.dx, tab.n, what)
    a0, a1, b0, b1 = _h00(t), _h01(t), tab.dx * _h10(t), tab.dx * _h11(t)
    c0, c1, d0, d1 = _dh00(t) / tab.dx, _dh01(t) / tab.dx, _dh10(t), _dh11(t)
    m = size(tab.V, 1)
    val = ntuple(k -> a0 * tab.V[k, i] + a1 * tab.V[k, i+1] + b0 * tab.D[k, i] + b1 * tab.D[k, i+1], m)
    der = ntuple(k -> c0 * tab.V[k, i] + c1 * tab.V[k, i+1] + d0 * tab.D[k, i] + d1 * tab.D[k, i+1], m)
    return val, der
end

# ------------------------------------------------------------------------------------------------
# The tables.
"""
Tabulated IAPWS-95 water and steam properties (saturation and gas branches) used by the physical steam units.
"""
struct SteamTables
    sat::Hermite1          # of l = log p: T, v_f, v_g, u_f, u_g, h_f, h_g, s_f, s_g
    gas::Hermite1          # of X = log rho: T_g, u_g, u_top
    vap::Hermite2          # of (X, xi): T, p, s
    ise::Hermite2          # of (l, eta): h, T
    istop::Hermite1        # of l: s_top (entropy at T_TOP and p)
    l_min::Float64
    l_max::Float64
    X_min::Float64
    X_max::Float64
end

const _STEAM_TABLES = Ref{Union{Nothing,SteamTables}}(nothing)
const _STEAM_TABLES_GRID = Ref{Any}(nothing)
# raised whenever the tables' construction changes, so that a cached file of an earlier version is
# never read (the file name carries it)
const STEAM_TABLES_VERSION = "v2"

"""
    steam_tables(; rebuild = false) -> SteamTables

The tables of this file, built from IAPWS-95 on first use and kept for the session; with `cache`
they are also written to `outputs/cache/` and read from there by later sessions (`rebuild`
forces a new build). The grid sizes are keywords: `nl` saturation points in log pressure, `nX` by
`nxi` for the vapour map, `nl`-spaced pressures by `neta` for the isentrope.
"""
function steam_tables(; rebuild::Bool = false, cache::Bool = true, nl = 121, nX = 91, nxi = 61, neta = 61)
    grid = (nl, nX, nxi, neta)
    _STEAM_TABLES[] === nothing || rebuild || _STEAM_TABLES_GRID[] != grid || return _STEAM_TABLES[]
    path = joinpath(PHPS_ROOT, "outputs", "cache",
                    "steam_tables_$(STEAM_TABLES_VERSION)_$(nl)_$(nX)_$(nxi)_$(neta).jld2")
    # the tables and their grid are published together, after everything that can fail
    # (consultation 9, point 4)
    if cache && !rebuild && isfile(path)
        tb = JLD2.load(path, "tables")
    else
        tb = _build_steam_tables(; nl, nX, nxi, neta)
        cache && (mkpath(dirname(path)); JLD2.jldsave(path; tables = tb))
    end
    _STEAM_TABLES[], _STEAM_TABLES_GRID[] = tb, grid
    return tb
end

# saturation properties at a pressure (generic), from the lifted solve
function _sat_props(p, guess)
    x0 = _sat_newton(_deep(p), guess)
    x = _lift((y, a) -> _sat_residual(y, a), x0, p)
    l = _ps_core(x[1], x[2]); v = _ps_core(x[1], x[3])
    return (x[1], 1 / x[2], 1 / x[3], l.u, v.u, l.h, v.h, l.s, v.s), x0
end

# saturated vapour at a vapour density: unknowns (T, rho_l) with rho_v fixed
function _gas_residual(y, rhov)
    l = _ps_core(y[1], y[2]); v = _ps_core(y[1], rhov)
    RT = _PS.IAPWS95_R * y[1]
    return [(l.p - v.p) / (y[2] * RT), (l.g - v.g) / RT]
end

# vapour density at (p, T) on the vapour side: safeguarded Newton between a small density and the
# saturated vapour's (or the critical density above it)
function _vapour_rho(p, T; rhohi)
    lo, hi = 1e-9, rhohi
    _ps_core(T, hi).p >= p || error("steam tables: no vapour density below $hi kg/m3 at p = $p, T = $T")
    rho = min(p / (_PS.IAPWS95_R * T), 0.999 * hi)
    for _ in 1:300
        c = _ps_core(T, rho)
        f = c.p - p
        # converged: return this iterate (testing before the update; an update after a collapsed
        # bracket would bisect away from the root)
        abs(f) <= 1e-12 * p && return rho
        f > 0 ? (hi = rho) : (lo = rho)
        dp = ForwardDiff.derivative(r -> _ps_core(T, r).p, rho)
        step = rho - f / dp
        rho = (dp > 0 && lo < step < hi) ? step : (lo + hi) / 2
        if hi - lo <= 4eps(hi)
            abs(_ps_core(T, rho).p - p) <= 1e-9 * p && return rho
            error("steam tables: the density bracket collapsed without a root at p = $p, T = $T")
        end
    end
    error("steam tables: no vapour density at p = $p, T = $T")
end

function _build_steam_tables(; nl = 241, nX = 201, nxi = 121, neta = 121)
    # saturation on l = log p, from just above the triple point to 21 MPa, by continuation
    l_min, l_max = log(700.0), log(21.0e6)
    dl = (l_max - l_min) / (nl - 1)
    satV, satD = zeros(9, nl), zeros(9, nl)
    guess = [273.16, 999.793, 0.00485458]
    pp = STEAM_P_TRIPLE
    for i in 1:nl
        p = exp(l_min + (i - 1) * dl)
        while pp < p                                   # continuation from the last solution
            pp = min(p, pp * 1.15)
            guess = _sat_newton(pp, guess)
        end
        f = l -> collect(_sat_props(exp(l), guess)[1])
        l = l_min + (i - 1) * dl
        satV[:, i] = f(l)
        satD[:, i] = ForwardDiff.derivative(f, l)
        guess = _sat_newton(p, guess)
    end
    sat = Hermite1(l_min, dl, nl, satV, satD)
    # saturated vapour against log density
    X_min, X_max = log(0.004), log(160.0)
    dX = (X_max - X_min) / (nX - 1)
    gasV, gasD = zeros(3, nX), zeros(3, nX)
    for i in 1:nX
        X = X_min + (i - 1) * dX
        rv = exp(X)
        # start from the saturation table: the pressure whose v_g is 1/rv
        lg = l_min
        for _ in 1:60
            vg = _interp(sat, 3, lg, "")
            dvg = ForwardDiff.derivative(z -> _interp(sat, 3, z, ""), lg)
            lg = clamp(lg - (vg - 1 / rv) / dvg, l_min, l_max)
        end
        T0 = _interp(sat, 1, lg, ""); rl0 = 1 / _interp(sat, 2, lg, "")
        y0 = [T0, rl0]
        for _ in 1:60
            r = _gas_residual(y0, rv)
            maximum(abs, r) < 1e-13 && break
            y0 = y0 .- ForwardDiff.jacobian(y -> _gas_residual(y, rv), y0) \ r
        end
        maximum(abs, _gas_residual(y0, rv)) < 1e-10 || error("steam tables: no saturated vapour at density $rv")
        (y0[1] > 0 && y0[2] > rv) || error("steam tables: not a liquid-vapour pair at vapour density $rv")
        f = X -> begin
            y = _lift((yy, a) -> _gas_residual(yy, exp(a)), y0, X)
            rhov = exp(X)
            [y[1], _ps_core(y[1], rhov).u, _ps_core(oftype(y[1], STEAM_T_TOP), rhov).u]
        end
        gasV[:, i] = f(X)
        gasD[:, i] = ForwardDiff.derivative(f, X)
    end
    gas = Hermite1(X_min, dX, nX, gasV, gasD)
    # the vapour map on (X, xi): T solved from u(T, rho) = u on [T_g, T_TOP]
    dxi = 1.0 / (nxi - 1)
    F, Fx, Fy, Fxy = (zeros(3, nX, nxi) for _ in 1:4)
    function vap_node(X, xi)
        rho = exp(X)
        Tg = _interp(gas, 1, X, ""); ug = _interp(gas, 2, X, ""); ut = _interp(gas, 3, X, "")
        u = ug + xi * (ut - ug)
        # primal solve
        Tp, up, rp = _deep(Tg), _deep(u), _deep(rho)
        T = Tp + _deep(xi) * (STEAM_T_TOP - Tp)
        for _ in 1:100
            c = _ps_core(T, rp)
            du = ForwardDiff.derivative(z -> _ps_core(z, rp).u, T)
            dT = (c.u - up) / du
            T -= dT
            abs(dT) < 1e-12 * T && break
        end
        abs(_ps_core(T, rp).u - up) < 1e-9 * abs(up) || error("steam tables: no vapour temperature at rho = $rp, u = $up")
        # lift
        Tl = T * one(u)
        for _ in 1:2
            c = _ps_core(Tl, rho)
            du = ForwardDiff.derivative(z -> _ps_core(z, _deep(rho)).u, _deep(Tl))
            Tl = Tl - (c.u - u) / du
        end
        c = _ps_core(Tl, rho)
        return [Tl, c.p, c.s]
    end
    for i in 1:nX, j in 1:nxi
        X, xi = X_min + (i - 1) * dX, (j - 1) * dxi
        g = z -> vap_node(z[1], z[2])
        F[:, i, j] = g([X, xi])
        J = ForwardDiff.jacobian(g, [X, xi])
        Fx[:, i, j] = J[:, 1]; Fy[:, i, j] = J[:, 2]
        Fxy[:, i, j] = ForwardDiff.derivative(b -> ForwardDiff.derivative(a -> vap_node(a, b), X), xi)
    end
    vap = Hermite2(X_min, dX, nX, 0.0, dxi, nxi, F, Fx, Fy, Fxy)
    # the isentrope on (l, eta), up to 20 MPa (no section ends higher; saturation coordinates do not
    # exist above the critical pressure): T solved from s(T, rho(p, T)) = s, with ds/dT|p = cp/T
    l_hi = log(20.0e6)
    ni = ceil(Int, (l_hi - l_min) / dl - 1e-9) + 1          # the last pressure at or above 20 MPa
    l_hi = l_min + (ni - 1) * dl
    l_hi <= l_max || error("steam tables: the isentrope would exceed the saturation table")
    deta = 1.0 / (neta - 1)
    G, Gx, Gy, Gxy = (zeros(2, ni, neta) for _ in 1:4)
    topV, topD = zeros(1, ni), zeros(1, ni)
    # vapour density at (p, T), T at or above saturation, p below critical: bracketed on the vapour
    # side by the saturated vapour's density at p; lifted to the caller's type
    function dens(p, T, rhog)
        r0 = _vapour_rho(_deep(p), _deep(T); rhohi = 1.0001 * _deep(rhog))
        r = r0 * one(p * T)
        for _ in 1:2
            dp = ForwardDiff.derivative(z -> _ps_core(_deep(T), z).p, _deep(r))
            r = r - (_ps_core(T, r).p - p) / dp
        end
        return r
    end
    cp_over_T(T, rho) = (st = _PS.iapws95_properties(_deep(T), _deep(rho)); st.cp / st.temperature)
    function ise_node(l, eta)
        p = exp(l)
        Ts = _interp(sat, 1, l, ""); sg = _interp(sat, 9, l, "")
        rhog = 1 / _interp(sat, 3, l, "")
        Ttop = oftype(Ts, STEAM_T_TOP)
        st = _ps_core(Ttop, dens(p, Ttop, rhog)).s
        s = sg + eta * (st - sg)
        sp, pp_, rg = _deep(s), _deep(p), _deep(rhog)
        lo, hi = _deep(Ts), STEAM_T_TOP
        T = lo + _deep(eta) * (hi - lo)
        done = false
        for _ in 1:300
            rho = _vapour_rho(pp_, T; rhohi = 1.0001 * rg)
            f = _ps_core(T, rho).s - sp
            # converged: keep this iterate (tested before the update, as in _vapour_rho)
            abs(f) < 1e-11 * max(1.0, abs(sp)) && (done = true; break)
            f > 0 ? (hi = T) : (lo = T)
            step = T - f / cp_over_T(T, rho)
            T = lo <= step <= hi ? step : (lo + hi) / 2
            if hi - lo <= 4eps(hi)
                done = abs(_ps_core(T, _vapour_rho(pp_, T; rhohi = 1.0001 * rg)).s - sp) < 1e-9 * max(1.0, abs(sp))
                break
            end
        end
        done || error("steam tables: no isentrope temperature at p = $pp_, s = $sp")
        Tl = T * one(s)
        for _ in 1:2
            rho = dens(p, Tl, rhog)
            Tl = Tl - (_ps_core(Tl, rho).s - s) / cp_over_T(Tl, rho)
        end
        rho = dens(p, Tl, rhog)
        return [_ps_core(Tl, rho).h, Tl], st
    end
    for i in 1:ni
        l = l_min + (i - 1) * dl
        topV[1, i] = ise_node(l, 0.0)[2]
        topD[1, i] = ForwardDiff.derivative(z -> ise_node(z, 0.0)[2], l)
        for j in 1:neta
            eta = (j - 1) * deta
            g = z -> ise_node(z[1], z[2])[1]
            G[:, i, j] = g([l, eta])
            J = ForwardDiff.jacobian(g, [l, eta])
            Gx[:, i, j] = J[:, 1]; Gy[:, i, j] = J[:, 2]
            Gxy[:, i, j] = ForwardDiff.derivative(b -> ForwardDiff.derivative(a -> ise_node(a, b)[1], l), eta)
        end
    end
    ise = Hermite2(l_min, dl, ni, 0.0, deta, neta, G, Gx, Gy, Gxy)
    istop = Hermite1(l_min, dl, ni, topV, topD)
    return SteamTables(sat, gas, vap, ise, istop, l_min, l_hi, X_min, X_max)
end

# ------------------------------------------------------------------------------------------------
# Lookups (generic in the number type).

"""
    steam_sat(tb, p) -> (; T, v_f, v_g, u_f, u_g, h_f, h_g, s_f, s_g)

Saturation properties at pressure `p` (Pa), from the table.
"""
function steam_sat(tb::SteamTables, p)
    l = log(p)
    v = ntuple(k -> _interp(tb.sat, k, l, "saturation pressure (log Pa)"), 9)
    return (; T = v[1], v_f = v[2], v_g = v[3], u_f = v[4], u_g = v[5], h_f = v[6], h_g = v[7], s_f = v[8], s_g = v[9])
end

"""
    steam_state(tb, rho, u) -> (; T, p, h, s, x)

The equilibrium state at density `rho` (kg/m3) and specific internal energy `u` (J/kg):
superheated vapour from the vapour table, or inside the dome the wet mixture (quality `x`); `x`
is 1 for vapour. Only the vapour side of the dome is covered (densities up to 160 kg/m3).
"""
function steam_state(tb::SteamTables, rho, u)
    X = log(rho)
    ug = _interp(tb.gas, 2, X, "density (log kg/m3)")
    # clearly vapour: the vapour map
    _deep(u) >= _deep(ug) * (1 + 1e-6) && return _vapour_state(tb, rho, u, X, ug)
    # near or inside the dome the decision is made on the saturation table alone, so that the
    # dry boundary is one curve (consultation 7, point 12): the pressure l_g at which the saturated
    # vapour has this specific volume, and its energy u_g there
    v = 1 / rho
    lg = _sat_l_of_vg(tb, _deep(v))
    lg === nothing && return _vapour_state(tb, rho, u, X, ug)        # beyond the table's pressures
    if _deep(u) >= _sat_ug(tb, lg)
        return _vapour_state(tb, rho, u, X, ug)                       # at or beyond the dry boundary
    end
    return _wet_state(tb, v, u, lg)
end

# the vapour map at (rho, u). Between the saturation table's dry boundary (which decides the phase)
# and the vapour map's (which differ by the tables' interpolation error, about 1e-7 of u) the map's
# first cell is continued a little below its edge: the same cubic, smooth in value and derivative,
# no plateau (consultation 8, point 12); further below, an error
const _STEAM_BELOW_EDGE = 0.01      # in cells of the vapour map's second coordinate
function _vapour_state(tb::SteamTables, rho, u, X, ug)
    ut = _interp(tb.gas, 3, X, "density (log kg/m3)")
    xi = (u - ug) / (ut - ug)
    T = _interp(tb.vap, 1, X, xi, "vapour state"; below = _STEAM_BELOW_EDGE)
    p = _interp(tb.vap, 2, X, xi, "vapour state"; below = _STEAM_BELOW_EDGE)
    s = _interp(tb.vap, 3, X, xi, "vapour state"; below = _STEAM_BELOW_EDGE)
    return (; T, p, h = u + p / rho, s, x = one(T))
end

# saturated vapour's energy at l
_sat_ug(tb::SteamTables, l) = _interp(tb.sat, 5, l, "saturation (log Pa)")

# the l = log p at which the saturated vapour's specific volume is v (v_g decreases with l), or
# nothing outside the table; Newton with a bracket
function _sat_l_of_vg(tb::SteamTables, v)
    lo, hi = tb.sat.x0, tb.sat.x0 + (tb.sat.n - 1) * tb.sat.dx
    vlo, vhi = _interp(tb.sat, 3, lo, ""), _interp(tb.sat, 3, hi, "")
    (vhi <= v <= vlo) || return nothing
    l = (lo + hi) / 2
    for _ in 1:200
        (val, der) = _interp_all(tb.sat, l, "saturation (log Pa)")
        f = val[3] - v
        abs(f) <= 1e-13 * v && return l
        f > 0 ? (lo = l) : (hi = l)
        st = l - f / der[3]
        l = lo < st < hi ? st : (lo + hi) / 2
        hi - lo < 1e-14 && break
    end
    abs(_interp(tb.sat, 3, l, "") - v) <= 1e-9 * v || error("steam tables: no saturation pressure for v_g = $v")
    return l
end

# R and dR/dl at l for specific volume v and energy u (table indices: 2 v_f, 3 v_g, 4 u_f, 5 u_g)
@inline function _wet_resid(tb::SteamTables, l, v, u)
    s, d = _interp_all(tb.sat, l, "wet state (log Pa)")
    dv = s[3] - s[2]
    x = (v - s[2]) / dv
    dx = (-d[2] * dv - (v - s[2]) * (d[3] - d[2])) / dv^2
    du = s[5] - s[4]
    return u - s[4] - x * du, -d[4] - dx * du - x * (d[5] - d[4])
end

function _wet_state(tb::SteamTables, v, u, lg)
    vp, up = _deep(v), _deep(u)
    # bracket: the table's lowest pressure, and the dry boundary lg at this specific volume (quality
    # 1 there, so the residual u - u_g(lg) < 0); safeguarded Newton
    lo = tb.sat.x0
    hi = lg
    flo = _wet_resid(tb, lo, vp, up)[1]
    fhi = _wet_resid(tb, hi, vp, up)[1]
    tol0 = 1e-10 * max(1.0, abs(up))
    # a root at an end of the bracket is accepted (consultation 7, point 11)
    if abs(flo) < tol0 || abs(fhi) < tol0
        l = abs(flo) < abs(fhi) ? lo : hi
        @goto lifted
    end
    flo * fhi < 0 || throw(DomainError((vp, up), "steam_state: no wet state at this density and energy"))
    l = (lo + hi) / 2
    fprev = Inf
    tol = 1e-10 * max(1.0, abs(up))
    converged = false
    for k in 1:200
        f, d = _wet_resid(tb, l, vp, up)
        if abs(f) < tol
            converged = true
            break
        end
        (f > 0) == (flo > 0) ? (lo = l) : (hi = l)
        st = l - f / d
        # Newton inside the bracket while it at least halves the residual; bisection otherwise
        l = (lo < st < hi && abs(f) < 0.5 * fprev) || k == 1 && lo < st < hi ? st : (lo + hi) / 2
        fprev = abs(f)
        hi - lo < 1e-15 && (converged = abs(_wet_resid(tb, l, vp, up)[1]) < 1e3 * tol; break)
    end
    converged || error("steam_state: the wet-state solve did not converge (v = $vp, u = $up)")
    @label lifted
    # lift to the caller's number type: two Newton corrections with the primal slope
    ll = l * one(v * u)
    for _ in 1:2
        f = _wet_resid(tb, ll, v, u)[1]
        ll = ll - f / _wet_resid(tb, _deep(ll), vp, up)[2]
    end
    s, _ = _interp_all(tb.sat, ll, "wet state (log Pa)")
    x = (v - s[2]) / (s[3] - s[2])
    # the quality, against the separately interpolated dry boundaries (consultation 7, point 12)
    -1e-9 <= _deep(x) <= 1 + 1e-6 || throw(DomainError(_deep(x), "steam_state: wet-state quality outside [0, 1]"))
    p = exp(ll)
    # h = u + p v exactly, as on the vapour side (consultation 7, point 13); s from the lever rule
    return (; T = s[1], p, h = u + p * v, s = s[8] + x * (s[9] - s[8]), x)
end

"""
    steam_h_isentropic(tb, p, s) -> (; h, T, x)

The enthalpy at pressure `p` and entropy `s`: from the isentrope table above the dome, the
equilibrium mixture `h_f + T_sat (s - s_f)` inside it. `x` is the quality (1 for vapour).
"""
function steam_h_isentropic(tb::SteamTables, p, s)
    l = log(p)
    (tb.l_min - 1e-9 <= _deep(l) <= tb.l_max + 1e-9) ||
        throw(DomainError(_deep(p), "steam_h_isentropic: pressure outside the table (700 Pa to 20 MPa)"))
    sat = steam_sat(tb, p)
    if s >= sat.s_g
        st = _interp(tb.istop, 1, l, "isentrope (log Pa)")
        eta = (s - sat.s_g) / (st - sat.s_g)
        return (; h = _interp(tb.ise, 1, l, eta, "isentrope"), T = _interp(tb.ise, 2, l, eta, "isentrope"), x = one(s))
    end
    s >= sat.s_f || throw(DomainError(_deep(s), "steam_h_isentropic: entropy below saturated liquid"))
    x = (s - sat.s_f) / (sat.s_g - sat.s_f)
    return (; h = sat.h_f + sat.T * (s - sat.s_f), T = sat.T, x)
end
