# Fault clearing at current zero (reviews 82-82d): each phase switch of a `clearing = :current_zero` fault opens at its
# own first current zero after t_off. The zeros are located by reruns of the same simulation: a run with tstop = t
# reproduces the final run's segments before t exactly (later breakpoints do not change earlier segments), so the
# phase current at the end of that run is the final run's left-limit current at an opening at t.
#
# Statuses of an opening: :zero (its own left-limit current within the tolerance), :removed (it fell out of the star by
# topology with a nonzero left limit), :grid (delayed to the admissible event grid: a nonzero interruption), :failed (the
# refinement missed the tolerance; the best time is used), :never (no zero before tstop: stays closed).

struct _ClearingStop <: Exception end

_with_open(f::PHFault, p::Int, t::Float64) =
    PHFault(f.bus, f.R, f.t_on, f.t_off, f.kind, f.Rg, f.clearing, ntuple(q -> q == p ? t : f.topen[q], 3))

# the phase currents into fault f with the switches `m` closed, at state x with the inputs u in force, time t
function _fault_phase_currents(s, x, u, f::PHFault, m::NTuple{3,Bool}, t)
    base, _, gg = fault_star(f.kind, f.R; Rg = f.Rg)
    any(base .& m) || return zeros(3)
    k = bus_index(_phys(s).net.net, f.bus)
    v = phs_phase_voltages(s, x, u; t, nan0 = gg == 0 ? 0.0 : NaN)[:, k]   # an isolated star does not see v0
    G = (base .& m) == base ? fault_gabc(f.kind, f.R; Rg = f.Rg) : fault_gstar(f.kind, f.R, m; Rg = f.Rg)
    return G * v
end

# the admissible event spacing: 0 (anywhere) for lumped lines without sampled controllers
function _event_grid(s, h, grid)
    Tsp = _sample_period(s)
    Tsp > 0 && return Float64(Tsp)
    grid !== nothing && return Float64(grid)
    isempty(_phys(s).tw) || return Float64(h)
    return 0.0
end

"""
    phs_resolve_clearing(s, x0, u; faults, tstop, kw...) -> (faults, report)

Locate the opening of every phase of the `clearing = :current_zero` faults (reviews 82-82d), in time order over all
faults: a detection run (the unresolved phases closed) stops at the first step end where a phase current changes sign
after max(t_off, the last located opening); every phase crossing in that step is refined by a safeguarded regula falsi
(Illinois) on runs to tstop = t, to |i| <= 1e-9 + 1e-7 x the phase's peak; the earliest is opened, and every other
phase of that fault whose current after the opening is within its tolerance opens with it. On an event grid (travelling-
wave lines, sampled controllers, `grid`) the phase opens at the first grid point at or after the detection step's end
(status :grid, its chopped left-limit current and an estimate Delta x the largest step secant slope within continuous
pieces reported, the jumps at fault switchings listed separately; no estimate with sampled controllers, whose inputs
jump at every sample). `cache = false` re-runs every evaluation (a check that the cache never changes the result). `kw` are
`phs_simulate`'s stepping keywords (`stops, hscale, h, grid, tol, per_period, maxiter`). The report has one entry per
phase opening.
"""
function phs_resolve_clearing(s, x0, u; faults, tstop::Real, stops = (), hscale = 1, h = nothing, grid = nothing,
                              tol = 1e-13, per_period = 20, maxiter = 40, rtol = 1e-7, atol = 1e-9, maxit = 12,
                              verbose::Bool = false, cache::Bool = true)
    fl = collect(PHFault, faults)
    Δ = _event_grid(s, h, grid)
    Δ > 0 && h === nothing && _sample_period(s) == 0 && grid === nothing &&
        throw(ArgumentError("phs_resolve_clearing: travelling-wave lines need h"))
    report = NamedTuple[]
    nruns = Ref(0)
    tlast = -Inf                                       # the latest opening located so far
    runkw = (; stops, hscale, h, grid, tol, per_period, maxiter, _resolve = false)
    tolof(pk) = atol + rtol * pk
    # the left-limit currents of fault fi at the end of a run to tstop = t (the run's last step end)
    # (runs are deterministic: one run per time and topology; the key carries every fault's opening times, so a result
    # from before an opening is never reused after it, review 83, 1)
    usecache = cache
    cache_ = Dict{Any,Any}()
    ckey(t) = (Float64(t), Tuple(f.topen for f in fl))
    evaluate(t) = usecache ? get!(() -> _evaluate(t), cache_, ckey(t)) : _evaluate(t)
    function _evaluate(t)
        nruns[] += 1
        out = Ref{Any}(nothing)
        mon = (tt, x, ub) -> (tt >= t - 1e-12 * max(1.0, t) &&
                              (out[] = [(_fault_phase_currents(s, x, ub, f, _closed(f, t; left = true), t),
                                         x, ub) for f in fl]); nothing)
        phs_simulate(s, x0, u; tstop = t, faults = fl, monitor = mon, sample_at = [Float64(t)], runkw...)
        out[] === nothing && error("phs_resolve_clearing: the run to $t s gave no last step")
        return out[]
    end
    while true
        U = [(fi, p) for (fi, f) in enumerate(fl) if f.clearing === :current_zero
             for p in 1:3 if fault_star(f.kind, f.R; Rg = f.Rg)[1][p] && isnan(f.topen[p])]
        isempty(U) && break
        tstart = Dict(fi => max(fl[fi].t_off, tlast) for (fi, _) in U)
        # ---- detection: the first step end with a sign change (or an exact zero at the start) ----
        prev = Dict{Tuple{Int,Int},Tuple{Float64,Float64}}()
        peak = Dict{Tuple{Int,Int},Float64}(c => 0.0 for c in U)
        hist = Dict{Tuple{Int,Int},Vector{Tuple{Float64,Float64}}}(c => Tuple{Float64,Float64}[] for c in U)
        found = Tuple{Int,Int,Float64,Float64,Float64,Float64}[]   # fi, p, ta, ia, tb, ib
        tgrid = Ref(Inf)                               # (grid: run on to the first grid point at or after the crossing)
        mon = function (t, x, ub)
            for fi in unique(first.(U))
                f = fl[fi]
                t >= f.t_on || continue
                i3 = _fault_phase_currents(s, x, ub, f, _closed(f, t), t)
                for (gi, p) in U
                    gi == fi || continue
                    c = (fi, p); i = i3[p]
                    peak[c] = max(peak[c], abs(i))
                    t >= tstart[fi] - 1e-12 || continue
                    push!(hist[c], (t, i))
                    if isfinite(tgrid[])
                        continue
                    elseif !haskey(prev, c)
                        abs(i) <= tolof(peak[c]) && push!(found, (fi, p, t, i, t, i))
                    else
                        ta, ia = prev[c]
                        (ia * i < 0 || i == 0) && push!(found, (fi, p, ta, ia, t, i))
                    end
                    prev[c] = (t, i)
                end
            end
            if !isempty(found) && !isfinite(tgrid[])
                Δ == 0 && throw(_ClearingStop())
                tb = maximum(c[5] for c in found)
                tgrid[] = min(ceil(tb / Δ - 1e-9) * Δ, Float64(tstop))
            end
            isfinite(tgrid[]) && t >= tgrid[] - 1e-12 && throw(_ClearingStop())
            return nothing
        end
        nruns[] += 1
        stopped = try
            phs_simulate(s, x0, u; tstop, faults = fl, monitor = mon, sample_at = [Float64(tstop)], runkw...)
            false
        catch e
            e isa _ClearingStop || rethrow()
            true
        end
        if !stopped || isempty(found)                  # no zero before tstop: these phases stay closed
            for (fi, p) in U
                fl[fi] = _with_open(fl[fi], p, Inf)
                push!(report, (; fault = fi, bus = fl[fi].bus, kind = fl[fi].kind, phase = "abc"[p], topen = Inf,
                               status = :never, ileft = NaN, peak = peak[(fi, p)], tol = tolof(peak[(fi, p)]),
                               runs = nruns[], estimate = NaN, jumps = Tuple{Float64,Float64}[],
                               note = "no current zero before tstop"))
            end
            break
        end
        # ---- the opening time of every crossing found in that step ----
        cands = NamedTuple[]
        # refined in the order of the linear estimate of their zero; a crossing whose own current is already within its
        # tolerance at an earlier refined zero (the ab partner) is not refined again
        lin(c) = c[3] == c[5] ? c[3] : c[3] + (c[5] - c[3]) * c[4] / (c[4] - c[6])
        for (fi, p, ta, ia, tb, ib) in sort(found; by = lin)
            c = (fi, p); tl = tolof(peak[c])
            if Δ == 0 && any(cd -> cd.status === :zero && haskey(cache_, ckey(cd.t)) &&
                                   abs(cache_[ckey(cd.t)][fi][1][p]) <= tl, cands)
                continue
            end
            if Δ > 0                                   # the first admissible grid point at or after the crossing
                ts_ = tgrid[]
                h_ = hist[c]
                # the secant slopes within continuous pieces only: a step that contains a fault switching time (any
                # fault's t_on, t_off or opening) is a jump, listed separately (review 83, 2). With sampled controllers
                # the inputs jump at every sample: no estimate is given there
                sw = [t for g in fl for t in (g.t_on, g.t_off, g.topen...) if isfinite(t)]
                sl, jumps = Float64[], Tuple{Float64,Float64}[]
                for j in 2:length(h_)
                    (t0_, i0_), (t1_, i1_) = h_[j-1], h_[j]
                    t0_ >= ta - 1e-12 || continue
                    if any(t -> t0_ + 1e-12 < t <= t1_ + 1e-12, sw)
                        push!(jumps, (t1_, i1_ - i0_))
                    else
                        push!(sl, abs(i1_ - i0_) / (t1_ - t0_))
                    end
                end
                est = _sample_period(s) > 0 ? NaN : Δ * (isempty(sl) ? 0.0 : maximum(sl))
                push!(cands, (; fi, p, t = ts_, status = :grid, estimate = est, jumps, tl))
                continue
            end
            if ta == tb                                # exactly zero at the start of eligibility
                push!(cands, (; fi, p, t = ta, status = :zero, estimate = NaN, tl)); continue
            end
            fe(t) = evaluate(t)[fi][1][p]
            a, b = ta, tb
            fa, fb = fe(a), fe(b)
            hdet = tb - ta; lo = tstart[fi]; hi = Float64(tstop)
            widen = 0
            while fa * fb > 0 && widen < 3                # re-evaluated end points no longer bracket: widen, clamped
                widen += 1
                a = max(lo, a - hdet); b = min(hi, b + hdet)
                fa, fb = fe(a), fe(b)
            end
            best = abs(fa) <= abs(fb) ? (a, fa) : (b, fb)
            if fa * fb <= 0
                it = 0
                while abs(best[2]) > tl && it < maxit
                    it += 1
                    cc = (fb * a - fa * b) / (fb - fa)
                    (cc <= min(a, b) || cc >= max(a, b)) && (cc = (a + b) / 2)
                    fc = fe(cc)
                    abs(fc) < abs(best[2]) && (best = (cc, fc))
                    if fc * fb < 0
                        a, fa = b, fb
                    else
                        fa /= 2
                    end
                    b, fb = cc, fc
                end
            end
            push!(cands, (; fi, p, t = best[1], status = abs(best[2]) <= tl ? :zero : :failed, estimate = NaN, tl))
        end
        # ---- open the earliest; its fault's other phases open with it if their current after the opening is zero ----
        e = cands[argmin([c.t for c in cands])]
        res = evaluate(e.t)
        f = fl[e.fi]
        il, x_, ub_ = res[e.fi]
        mb = _closed(f, e.t; left = true)
        f = _with_open(f, e.p, e.t)
        opened = [(e.p, e.status, il[e.p], e.tl, e.estimate, get(e, :jumps, Tuple{Float64,Float64}[]))]
        while true                                     # phases that fall out of the star with it
            ma = mb .& ntuple(q -> !any(o -> o[1] == q, opened), 3)
            ir = _fault_phase_currents(s, x_, ub_, f, ma, e.t)
            more = false
            for q in 1:3
                (ma[q] && fault_star(f.kind, f.R; Rg = f.Rg)[1][q] && isnan(f.topen[q])) || continue
                tq = tolof(get(peak, (e.fi, q), 0.0))
                abs(ir[q]) <= tq || continue
                st = abs(il[q]) <= tq ? :zero : (e.status === :grid ? :grid : :removed)
                f = _with_open(f, q, e.t); push!(opened, (q, st, il[q], tq, NaN, Tuple{Float64,Float64}[])); more = true
            end
            more || break
        end
        fl[e.fi] = f
        tlast = e.t
        for (q, st, i, tq, est, jmp) in opened
            push!(report, (; fault = e.fi, bus = f.bus, kind = f.kind, phase = "abc"[q], topen = e.t, status = st,
                           ileft = i, peak = get(peak, (e.fi, q), NaN), tol = tq, runs = nruns[], estimate = est, jumps = jmp,
                           note = st === :failed ? "refinement missed the tolerance" :
                                  st === :removed ? "fell out of the star with a nonzero current" :
                                  st === :grid ? "delayed to the event grid (nonzero interruption)" : ""))
            verbose && println("  clearing: fault $(e.fi) bus $(f.bus) $(f.kind) phase $("abc"[q]) opens at $(e.t) s ($st, ",
                               "left-limit current $(i), runs $(nruns[]))")
        end
        any(o -> o[2] === :failed, opened) &&
            @warn "phs_resolve_clearing: fault $(e.fi) at bus $(f.bus): the current-zero refinement missed the tolerance"
    end
    return fl, report
end
