# Integration of the port-Hamiltonian grid model (docs/active/PH_SIMULATOR_SPEC.md, section 7).
#
# Primary integrator: the implicit midpoint rule with a fixed step in each segment between
# events,
#   x1 - x0 = h f((x0 + x1)/2),
# solved by a simplified Newton iteration. The energy is quadratic in the model's coordinates,
# so H(x1) - H(x0) = grad H(xm)' (x1 - x0) and the discrete balance
#   H(x1) - H(x0) = h (supply(xm) - losses(xm))
# holds for the whole system and for every component, up to round-off and the Newton
# tolerance (the "scheme identity"). The account keeps the two parts of that statement apart:
#   - `scheme`: the step's energy increment grad H_c(xm)' (x1 - x0) less h (port - loss)(xm).
#     It is grad H_c(xm)' F with F the Newton residual of the step, so it measures the solve
#     and the component balance at the midpoint;
#   - `closure`: the sum of those increments less H_c(x_end) - H_c(x_0) from the energy
#     function. It is zero up to rounding exactly when H_c is quadratic.
# It is a statement about energy, not about accuracy: the
# trajectory is second order in h and is checked by refinement and against an independent
# BDF solution (`phs_simulate_bdf`).
#
# Saturating transformer cores (`transformer.jl`; record section 23) make the energy non-quadratic and dependent on the
# frame clock. Their branches' rows are written with the exact frame rotation at the frozen angle theta_m and their
# currents are a discrete gradient (`_step_residual!`); every other row of the step sees those currents. The balance then
# holds exactly again, and for a cored branch `closure` compares the summed discrete-gradient increments with its
# energy function. The clock is set from the elapsed time, not iterated. The step's currents reach the rows through
# `PHSystem.step`, a shared mutable reference set only around a step's residual and its account: one evaluation of a
# system at a time (no threads on one system; T3 risks).
#
# The method does not damp fast modes. The step of a segment is taken from the spectrum of
# that segment's linearisation (`phs_step_rule`): it resolves the fastest oscillation and the
# fastest decay, the latter because a fault conductance on a node capacitor decays much faster
# than anything oscillates.

"""
    PHFault(bus, R, t_on, t_off; kind = :abc, Rg = R, clearing = :instant, topen = nothing)

A fault of resistance `R` (pu) at bus `bus` from `t_on`. `clearing = :instant`: every phase opens at `t_off` (whatever its
current). `clearing = :current_zero` (review 82): the fault is a star of per-phase switches (`fault_star`), and each phase
opens at its own first current zero after `t_off` (the arc kept at R until then; extinction of the shunt fault with
successful recovery, no restrike). `phs_simulate` locates the zeros (`phs_resolve_clearing`) unless `topen` (each phase's
opening time) is given; NaN in `topen` = not yet located (closed).
"""
struct PHFault
    bus::Int
    R::Float64
    t_on::Float64
    t_off::Float64
    kind::Symbol       # :abc (the three phases to an isolated star: today's symmetric fault, dq only), :abcg, :ag, :bg,
                       # :cg, :ab, :bc, :ca, :abg, :bcg, :cag (zeroseq.jl, `fault_gabc`; upgrade 4)
    Rg::Float64        # the ground resistance of a two-phase-to-ground fault
    clearing::Symbol   # :instant or :current_zero (review 82)
    topen::NTuple{3,Float64}   # phase a, b, c: the switch conducts on [t_on, topen); NaN: not yet located (closed)
end
function PHFault(bus, R, t_on, t_off; kind::Symbol = :abc, Rg::Real = R, clearing::Symbol = :instant, topen = nothing)
    clearing in (:instant, :current_zero) || throw(ArgumentError("PHFault: clearing = $clearing"))
    tt = topen !== nothing ? ntuple(p -> Float64(topen[p]), 3) :
         clearing === :instant ? (Float64(t_off), Float64(t_off), Float64(t_off)) : (NaN, NaN, NaN)
    return PHFault(bus, Float64(R), Float64(t_on), Float64(t_off), kind, Float64(Rg), clearing, tt)
end
# the switches that conduct at t (right limit: t < topen; left limit: t <= topen), NaN = closed
_closed(f::PHFault, t; left::Bool = false) =
    ntuple(p -> (left ? f.t_on < t : f.t_on <= t) && (isnan(f.topen[p]) || (left ? t <= f.topen[p] : t < f.topen[p])), 3)
_unresolved(f::PHFault) = f.clearing === :current_zero &&
    any(p -> fault_star(f.kind, f.R; Rg = f.Rg)[1][p] && isnan(f.topen[p]), 1:3)

"""
    phs_scales(s, x) -> Vector{Float64}

A magnitude for every state, used to scale residuals: the largest absolute value within its
group (node charges, branch fluxes, shunt fluxes, each machine's fluxes), the momentum itself,
and 1 for an angle.
"""
function phs_scales(s::PHSystem, x::AbstractVector{Float64})
    sc = ones(s.n)
    nd, nbk, ns = 2 * length(s.dyn), 2 * length(s.branches), 2 * count(>(0), s.sidx)
    for r in (1:nd, nd+1:nd+nbk, nd+nbk+1:nd+nbk+ns)
        isempty(r) || (sc[r] .= maximum(abs, view(x, r)))
    end
    for m in s.machines
        r = m.offset:_ip(m)-1
        sc[r] .= maximum(abs, view(x, r))
        sc[_ip(m)] = abs(x[_ip(m)])
    end
    if !isempty(s.cores)                            # the source loop fluxes with the branch fluxes; the clock is set
        r = reduce(vcat, [[q.xs, q.xs + 1] for q in s.cores])
        sc[r] .= max(maximum(abs, view(x, r)), maximum(abs, view(x, nd+1:nd+nbk)))
    end
    for mo in s.motors                              # composite loads (review 74b): flux 1 / omega_b, momentum 2H, lag 1
        sc[mo.offset:mo.offset+5] .= 1 / s.omega_b
        sc[mo.offset+6] = 2 * mo.c.H
    end
    if s.z0 !== nothing                             # the zero axis (zero at rest): the dq groups' scales
        z = s.z0
        qn = nd > 0 ? maximum(abs, view(x, 1:nd)) : 1.0
        qb = nbk > 0 ? maximum(abs, view(x, nd+1:nd+nbk)) : 1.0
        for q in z.dyn0; sc[z.qidx0[q]] = qn; end
        for idx in z.bidx0; sc[idx] = qb; end
        for idx in z.sidx0; sc[idx] = qb; end
    end
    return sc
end

# ---- saturating cores in a step (record section 23) ----

# the step's discrete-gradient currents of every core, from the step's ends: the rotated ends e^{-j beta} x0,
# e^{j beta} x1 with beta = (theta1 - theta0)/2 from the stored clock, Gonzalez's discrete gradient at theta_m; the
# rotated increments `dxt` and each core's energy increment `inc` (pu s); nothing without cores
function _core_context(ps::PHSystem, x0, x1, h)
    isempty(ps.cores) && return nothing
    wb, ic = ps.omega_b, ps.iclk
    beta = (x1[ic] - x0[ic]) / 2
    thm = x0[ic] + beta
    em, ep = cis(-beta), cis(beta)
    ibar = Tuple{Any,Any}[]
    dxt = Tuple{Any,Any}[]
    inc = Any[]
    for q in ps.cores
        H0, S0 = em * _cx(x0, ps.bidx[q.pos]), em * _cx(x0, q.xs)
        H1, S1 = ep * _cx(x1, ps.bidx[q.pos]), ep * _cx(x1, q.xs)
        iH, iS = core_dgrad(q, wb * H0, wb * S0, wb * H1, wb * S1, thm)
        push!(ibar, (iH, iS))
        push!(dxt, (H1 - H0, S1 - S0))
        push!(inc, real(conj(iH) * (H1 - H0)) - real(conj(iS) * (S1 - S0)))
    end
    return (; ibar, dxt, inc, h, beta)
end

"""
    _step_residual!(F, fm, s, x0, x1, u, frozen, h; clock = nothing) -> the step's core context (or nothing)

The residual of one step: `x1 - x0 - h f(x_m)` (the implicit midpoint rule), and with saturating cores the cored
branches' rows in the exact-rotation form `e^{j beta} x1 - e^{-j beta} x0 - h (f + j omega_b x)(x_m)` evaluated with the
step's discrete-gradient currents, which every other row of the step uses too (record section 23). The clock's row is
`x1 - clock` (the clock is set from the elapsed time).
"""
function _step_residual!(F, fm, s::PHModel, x0, x1, u, frozen, h; clock = nothing)
    ps = _phys(s)
    xm = (x0 .+ x1) ./ 2
    if isempty(ps.cores)
        _frozen_rhs!(fm, s, xm, u, frozen)
        @. F = x1 - x0 - h * fm
        return nothing
    end
    ctx = _core_context(ps, x0, x1, h)
    ps.step[] = ctx
    try
        _frozen_rhs!(fm, s, xm, u, frozen)
    finally
        ps.step[] = nothing
    end
    @. F = x1 - x0 - h * fm
    wb = ps.omega_b
    for (k, q) in enumerate(ps.cores)
        for (i, d) in ((ps.bidx[q.pos], ctx.dxt[k][1]), (q.xs, ctx.dxt[k][2]))
            r = d - h * (complex(fm[i], fm[i+1]) + im * wb * complex(xm[i], xm[i+1]))
            F[i], F[i+1] = real(r), imag(r)
        end
    end
    F[ps.iclk] = clock === nothing ? x1[ps.iclk] - x0[ps.iclk] - h * wb : x1[ps.iclk] - clock
    return ctx
end

# the segments of every core's phase fluxes at the state `x` (the Newton matrix is refreshed when they change)
function _core_segments(ps::PHSystem, x)
    out = Int[]
    for q in ps.cores
        c = core_solve(q, ps.omega_b * _cx(x, ps.bidx[q.pos]), ps.omega_b * _cx(x, q.xs), x[ps.iclk])
        for p in _CORE_PHASES
            push!(out, _seg(q.curve, real(c.lam * cis(x[ps.iclk] + p))))
        end
    end
    return out
end

_jacobian(s::PHModel, x, u) = ForwardDiff.jacobian(y -> phs_rhs(s, y, u), x)
# the Jacobian of the vector field a step actually solves: limiter decisions held
_jacobian(s::PHModel, x, u, frozen) =
    ForwardDiff.jacobian(y -> _frozen_rhs!(similar(y, _nstates(s)), s, y, u, frozen), x)

"""
    phs_step_rule(s, x, u; per_period = 20) -> (; h, w_max, r_max)

The step the midpoint rule needs at `x`: `2 pi / (per_period w_max)` to resolve the fastest
oscillation (`w_max`, the largest imaginary part of an eigenvalue of the linearisation) and
`1 / r_max` so that the fastest decay (`r_max`, the largest magnitude of a real part) is damped
monotonically; the smaller of the two.
"""
function phs_step_rule(s::PHModel, x::AbstractVector{Float64}, u::PHInput; per_period::Real = 20)
    ev = eigvals(_jacobian(s, x, u))
    w_max, r_max = maximum(abs ∘ imag, ev), maximum(abs ∘ real, ev)
    ps = _phys(s)
    if !isempty(ps.cores)
        # with saturating cores: also every core short-circuited, the stiffest the reduced branch can be (the leakage
        # bounds it; T2, ideas): a heuristic bound, not a proved one for the coupled system (T3 3d); refinement checks it
        ps.step[] = :short
        evs = try
            eigvals(_jacobian(s, x, u))
        finally
            ps.step[] = nothing
        end
        w_max, r_max = max(w_max, maximum(abs ∘ imag, evs)), max(r_max, maximum(abs ∘ real, evs))
    end
    return (; h = min(2pi / (per_period * w_max), 1 / r_max), w_max, r_max)
end

"""
    PHRun

A trajectory and its energy account. Saved every `save_dt` and at every segment end: `t`, states
`X`, bus voltages `V` (rows in network order, generator buses recovered), component `energy`,
`loss` and `port` power, boundary `supply`, total `H`. Accumulated over the run, per component
(`names`, `kind`) and for the system (`total`): `scheme` (the discrete balance, see the file
header) with `scheme_max` the largest magnitude its running sum reached, `closure` (summed
increments against the energy function), `simpson` (the accumulated defect with Simpson's rule
on the powers at step ends and midpoints, one-sided at events), `throughput` (integral of the
sum of the absolute port powers) and `moved` (variation of the stored energy). `segments` lists each segment's `(t0, t1, h, steps, w_max, r_max)`.
`records` holds the quantities declared with `phs_simulate(...; record)`, by name: a vector,
or a matrix with one column per saved time. `phs_write_run` writes all of it to a file.
"""
struct PHRun
    t::Vector{Float64}
    X::Matrix{Float64}
    V::Matrix{ComplexF64}
    names::Vector{String}
    kind::Vector{Symbol}
    energy::Matrix{Float64}
    loss::Matrix{Float64}
    port::Matrix{Float64}
    supply::Vector{Float64}
    H::Vector{Float64}
    scheme::Vector{Float64}
    scheme_max::Vector{Float64}
    closure::Vector{Float64}
    simpson::Vector{Float64}
    throughput::Vector{Float64}
    moved::Vector{Float64}
    total::NamedTuple
    segments::Vector{NamedTuple}
    newton::NamedTuple
    wall::Float64
    records::Dict{String,Any}
end

Base.show(io::IO, r::PHRun) =
    print(io, "PHRun(", length(r.t), " saved points to t = ", r.t[end], " s, ",
          sum(g.steps for g in r.segments), " steps)")

function _bus_voltages(s::PHSystem, x, u)
    f, dx = phs_flows(s, x, u), phs_rhs(s, x, u)
    v = f.v
    for (b, vg) in _terminal_voltages(s, f, dx, x)
        v[bus_index(s.net.net, b)] = vg
    end
    for (ci, c) in enumerate(s.cuts)
        v[c.node] = _cut_voltage(s, ci, f, dx)
    end
    return v[1:nbus(s.net.net)]               # the bus rows (without the converters' source rows)
end

# switching bridges (upgrade 3): start at a run's first instant (a carrier peak), and each step's window
_bridge_start!(s, x, u, t0) = nothing
_bridge_window!(s, a, b) = nothing
_bridge_stop!(s) = nothing
_bridge_accumulate!(s, xm, u, h) = nothing
_bridge_slope!(s, x0, x1, u, h) = nothing

# sampled controllers: their period (0: none) and their call at the state `x`; the motors' undervoltage relays (review
# 76, a case-study option) are sampled devices of the physical system
function _sample_period(s::PHSystem)
    T = unique([mo.relay.Tr for mo in s.motors if mo.ridx > 0])
    length(T) <= 1 || throw(ArgumentError("motor relays with different sampling periods $T"))
    return isempty(T) ? 0.0 : T[1]
end
function _sample!(s::PHSystem, x, u)
    any(mo -> mo.ridx > 0, s.motors) || return x
    v = phs_flows(s, x, u).v
    for mo in s.motors
        mo.ridx > 0 && relay_sample!(x, mo, abs(v[mo.node]))
    end
    return x
end

function _fault_conductance!(gf, s::PHSystem, faults, t)
    fill!(gf, 0.0)
    empty!(s.asym.af)
    acc = Dict{Int,Any}()                          # phase-asymmetric faults summed per bus in phase coordinates (80b, 5)
    for f in faults
        base = fault_star(f.kind, f.R; Rg = f.Rg)[1]
        m = base .& _closed(f, t)                  # the switches conducting at t (review 82; :instant: all until t_off)
        any(m) || continue
        full = m == base
        if f.kind !== :abc || !full                # (a partly open abc star is asymmetric: b to c through 2R)
            k = bus_index(s.net.net, f.bus)
            k in s.merged && throw(ArgumentError("a fault at generator bus $(f.bus) needs the terminal the series merge removed"))
            s.cutof[k] > 0 && throw(ArgumentError("an asymmetric fault at the reduced inductor-only bus $(f.bus)"))
            ground = f.kind in (:ag, :bg, :cg, :abg, :bcg, :cag, :abcg)
            if ground
                s.z0 === nothing && throw(ArgumentError("a ground fault ($(f.kind)) needs the zero axis (ph_system(...; zero))"))
                k in s.z0.removed && throw(ArgumentError("a ground fault at bus $(f.bus): its zero node is inductor-only " *
                                                         "(clearing would interrupt an inductor current; review 80b, 4)"))
            end
            G = full ? fault_gabc(f.kind, f.R; Rg = f.Rg) : fault_gstar(f.kind, f.R, m; Rg = f.Rg)
            a = get(acc, k, (G = zeros(3, 3), ground = false))
            acc[k] = (G = a.G .+ G, ground = a.ground || ground)
            continue
        end
        k = bus_index(s.net.net, f.bus)
        k in s.merged && throw(ArgumentError("a fault at generator bus $(f.bus) needs the " *
                                             "terminal the series merge removed"))
        s.cutof[k] > 0 && throw(ArgumentError("a fault at bus $(f.bus) would turn its reduced " *
                                              "inductor-only node into one with a conductance; " *
                                              "the number of states would change at the event"))
        f.R > 0 || throw(ArgumentError("a fault needs a resistance greater than zero"))
        gf[k] += 1 / f.R
    end
    for (k, a) in acc
        push!(s.asym.af, (row = k, Gabc = a.G, ground = a.ground))
    end
    return gf
end

_breakpoints(faults, tstop, stops = ()) =
    sort!(unique!(vcat(0.0, Float64(tstop), [Float64(t) for t in stops if 0 < t < tstop],
                       [t for f in faults for t in (f.t_on, f.t_off, f.topen...) if isfinite(t) && 0 < t < tstop])))

"""
    phs_simulate(s, x0, u; tstop, faults = PHFault[], stops = (), hscale = 1, h = nothing,
                 save_dt = 1e-3, grid = nothing, sample_at = nothing, monitor = nothing,
                 record = (), tol = 1e-13, per_period = 20) -> PHRun

Integrate the model (a `PHSystem`, or a `PHControlled` with its exciters and governors) from
`x0` with the implicit midpoint rule. For a controlled model the energy account is the physical
model's, with the inputs the controllers set at each evaluated state, and one more component
per physical exciter. A physical exciter's field energy is not quadratic (it saturates), so
for that component `closure` is not zero to rounding: it is the midpoint rule's energy error,
of second order in the step. Each segment between fault
switching times uses a fixed step: `h` if given, otherwise `hscale` times `phs_step_rule` at the
segment's start, shortened so that a whole number of steps fits. Nothing changes a state at an
event; the fault conductance changes. `stops` are further times at which a segment ends, so
that the trajectory is saved exactly there (a fixed step reaches no other prescribed time).
With `grid` given (seconds), every segment's step is shortened to divide `grid` and the
trajectory is saved exactly at the multiples of `grid`, for comparison with another tool's
samples; segment ends must then be multiples of `grid`. With `sample_at` given (increasing
times), the trajectory is saved at exactly those times and no others: within the step that
contains a time, the state is the cubic Hermite interpolant of the step's two ends and the
vector field there (the step is a few microseconds, so this adds nothing visible), and the
energy account is evaluated at that state. The accumulated accounts are unaffected.
`monitor(t, x, u)`, if given, is called after every step with the time, the state and the inputs
in force, for diagnostics that must see every step and not only the saved ones. `record` declares
quantities to keep while the run goes, as pairs `name => f` with `f(x, u)` a number or a vector
computed from the state and the inputs in force; each is evaluated at every saved time (once per
saved time, not per step) and kept in `r.records[name]`. The state itself is always kept, so a
quantity not declared can still be computed afterwards from `r.X`. The Newton iteration of a step stops when the residual,
scaled by `phs_scales`, is below `tol` (at `1e-12` the discrete balance of some rotors misses
`1e-10` of their throughput on IEEE-39; at `1e-13` every component meets it).
"""
function phs_simulate(s::PHModel, x0::AbstractVector{Float64}, u::PHInput{Float64};
                      tstop::Real, faults::AbstractVector{PHFault} = PHFault[], stops = (),
                      hscale::Real = 1, h::Union{Nothing,Real} = nothing, save_dt::Real = 1e-3,
                      grid::Union{Nothing,Real} = nothing,
                      sample_at::Union{Nothing,AbstractVector{<:Real}} = nothing, monitor = nothing,
                      record = (), tol::Real = 1e-13,
                      per_period::Real = 20, maxiter::Integer = 40, _resolve::Bool = true)
    if _resolve && any(_unresolved, faults)       # current-zero clearing: locate the openings first (review 82)
        fr, rep = phs_resolve_clearing(s, x0, u; faults, tstop, stops, hscale, h, grid, tol, per_period, maxiter)
        r = phs_simulate(s, x0, u; tstop, faults = fr, stops, hscale, h, save_dt, grid, sample_at, monitor, record, tol,
                         per_period, maxiter, _resolve = false)
        r.records["clearing"] = rep
        return r
    end
    wall = time()
    phys = _phys(s)
    ps = phys
    clock0 = ps.iclk > 0 ? Float64(x0[ps.iclk]) : 0.0   # the clock at t = 0: theta = clock0 + omega_b t
    n, nc = _nstates(s), length(_cnames(s))
    ub = PHInput{Float64}(u.Tm, u.vf, zeros(length(u.gf)), u.const_power, u.uc)
    _set_frame!(ub, ps, 0.0)                       # (stationary frame: the frame of t = 0 for anything before the loop)
    x = collect(x0)
    sc = phs_scales(s, x)
    ts, Xs, Vs = Float64[], Vector{Float64}[], Vector{ComplexF64}[]
    Es, Ls, Ps, Ss, Hs = Vector{Float64}[], Vector{Float64}[], Vector{Float64}[], Float64[], Float64[]
    rec = [(String(first(p)), last(p), Any[]) for p in record]     # name, function, values
    # (in the stationary frame every evaluation sees the frame phasor of its own time: `_set_frame!`, a no-op in the
    # synchronous frame)
    save!(t, L, xs = x) = (_set_frame!(ub, ps, t); push!(ts, t); push!(Xs, copy(xs)); push!(Vs, _bus_voltages(s, xs, ub));
                           push!(Es, L.energy); push!(Ls, L.loss); push!(Ps, L.port);
                           push!(Ss, L.supply); push!(Hs, L.H);
                           foreach(((_, f, vals),) -> push!(vals, f(xs, ub)), rec))
    samples = sample_at === nothing ? Float64[] : collect(Float64, sample_at)
    issorted(samples) || throw(ArgumentError("phs_simulate: sample_at must be increasing"))
    isamp = 1
    xold, f0, f1, xi = similar(x), similar(x), similar(x), similar(x)
    scheme, scheme_max, incsum, simpson, thr, moved = (zeros(nc) for _ in 1:6)
    E_start = Float64[]
    tot = Dict(:scheme => 0.0, :scheme_max => 0.0, :simpson => 0.0, :throughput => 0.0,
               :moved => 0.0, :switched => 0.0)
    motorc = findall(==(:motor), _ckind(s))           # the motors' components (relay trips: switching jumps)
    segments = NamedTuple[]
    iters, njac, worst, nfloor = 0, 0, 0.0, 0
    bp = _breakpoints(faults, tstop, stops)
    x1, xm, F, fm, dprev = similar(x), similar(x), similar(x), similar(x), zeros(n)
    # travelling-wave lines (tline.jl; reviews 71-72c): one fixed step for the run, every event on its grid; the lines'
    # histories start at their steady state at the initial bus voltages and carry across segments
    twrun = !isempty(ps.tw)
    twL, twS = TLine[], TLineState[]
    # the zero (ground) mode of the travelling-wave lines (upgrade 4): its own lines and histories on the zero nodes; the
    # history currents of both modes handled as one vector [twJ; twJ0]
    z0 = ps.z0
    tw0 = z0 !== nothing && !isempty(z0.tw0)
    twL0, twS0z = TLine[], TLineState[]
    nJ = length(ps.twJ)
    Jget() = tw0 ? vcat(ps.twJ, ComplexF64.(z0.twJ0)) : copy(ps.twJ)
    Jset!(J) = (ps.twJ .= view(J, 1:nJ); tw0 && (z0.twJ0 .= real.(view(J, nJ+1:length(J)))); nothing)
    if twrun
        h === nothing && throw(ArgumentError("phs_simulate: travelling-wave lines need a fixed step h"))
        ps.stationary && throw(ArgumentError("phs_simulate: travelling-wave lines are formulated in the synchronous frame"))
        for b in bp
            abs(b / h - round(b / h)) < 1e-6 ||
                throw(ArgumentError("phs_simulate: the event time $b s is not on the step grid of h = $h s"))
        end
        tw_steady_history!(ps, x, ub)               # the lines' steady history at the initial state (any builder)
        v0 = phs_flows(ps, x, ub).v
        for t in ps.tw
            L = TLine(t.r, t.x, t.b; omega_b = ps.omega_b, h = Float64(h), nsec = ps.nsec)
            push!(twL, L); push!(twS, tline_state(L, v0[t.from], v0[t.to]))
        end
        if tw0                                      # the zero mode at rest (v0 = 0 at the start: no asymmetry before)
            for t in z0.tw0
                L = TLine(t.r, t.x, t.b; omega_b = ps.omega_b, h = Float64(h), nsec = ps.nsec, rot = false)
                push!(twL0, L); push!(twS0z, tline_state(L, 0.0im, 0.0im))
            end
        end
    end
    twS0 = twrun ? sum(tline_storage(L, st) for (L, st) in zip(vcat(twL, twL0), vcat(twS, twS0z))) : 0.0   # the lines' storage at the start
    twk = Ref(0)                                   # steps so far: every 1000th step's identity is checked independently
                                                   # (full storage sums); `lines.worst` is over those steps
    twnorton!() = (fill!(ps.twJ, 0);
                   for (t, L, st) in zip(ps.tw, twL, twS)
                       _, J1, J2 = tline_norton(L, st); ps.twJ[t.from] += J1; ps.twJ[t.to] += J2
                   end;
                   tw0 && (fill!(z0.twJ0, 0.0);
                           for (t, L, st) in zip(z0.tw0, twL0, twS0z)
                               _, J1, J2 = tline_norton(L, st); z0.twJ0[t.from] += real(J1); z0.twJ0[t.to] += real(J2)
                           end))
    twnorton!()
    twnext() = (Jn = zeros(ComplexF64, nJ + (tw0 ? length(z0.twJ0) : 0));
                for (t, L, st) in zip(ps.tw, twL, twS)
                    _, J1, J2 = tline_norton(L, st); Jn[t.from] += J1; Jn[t.to] += J2
                end;
                tw0 && for (t, L, st) in zip(z0.tw0, twL0, twS0z)
                    _, J1, J2 = tline_norton(L, st); Jn[nJ+t.from] += real(J1); Jn[nJ+t.to] += real(J2)
                end; Jn)
    twJk, twJn = Jget(), Jget()                    # this step's history currents (midpoint) and the next step's
    twJp = Jget()                                  # the previous step's (the start: the steady history)
    twJof(t, tn) = t < tn + hq_ref[] / 2 ?         # the history current at time t of the step starting at tn: linear
        twJp .+ ((t - (tn - hq_ref[] / 2)) / hq_ref[]) .* (twJk .- twJp) :   # between the neighbouring midpoints
        twJk .+ ((t - (tn + hq_ref[] / 2)) / hq_ref[]) .* (twJn .- twJk)
    hq_ref = Ref(0.0)
    twacc = (; supply = Ref(0.0), dS = Ref(0.0), loss = Ref(0.0), dinterp = Ref(0.0), flow = Ref(0.0), worst = Ref(0.0),
             boundary = Ref(0.0))
    try                                            # (switching bridges are switched off again however the run ends)
    _bridge_start!(s, x, ub, bp[1])                # switching bridges: the PWM from the first instant (a carrier peak)
    for q in 1:length(bp)-1
        t0, t1 = bp[q], bp[q+1]
        _fault_conductance!(ub.gf, phys, faults, t0)
        _set_frame!(ub, ps, t0)
        rule = phs_step_rule(s, x, ub; per_period)
        hq = h === nothing ? hscale * rule.h : Float64(h)
        nsteps = max(1, ceil(Int, (t1 - t0) / hq - 1e-9))
        hq = twrun ? Float64(h) : (t1 - t0) / nsteps      # travelling-wave lines: h exactly (events on its grid)
        twrun && (nsteps = round(Int, (t1 - t0) / hq))
        stride = max(1, round(Int, save_dt / hq))
        if grid !== nothing
            stride = max(1, ceil(Int, grid / (h === nothing ? hscale * rule.h : Float64(h)) - 1e-9))
            ncell = (t1 - t0) / grid
            abs(ncell - round(ncell)) < 1e-9 ||
                throw(ArgumentError("phs_simulate: the segment $t0 to $t1 s is not a multiple of grid = $grid"))
            if twrun                               # the lines were built for h: the grid must be a multiple of it (review 73b)
                abs(grid / h - round(grid / h)) < 1e-9 ||
                    throw(ArgumentError("phs_simulate: grid = $grid s is not a multiple of h = $h s (travelling-wave lines)"))
                stride = round(Int, grid / h)
            else
                nsteps = stride * round(Int, ncell)
                hq = (t1 - t0) / nsteps
            end
        end
        # a sampled controller (`_sample_period > 0`): the step divides the period, and the controller's
        # call (`_sample!`) is made after every step that ends on a multiple of it
        Tsp = _sample_period(s)
        msub = 0
        Tsp > 0 && grid !== nothing &&
            throw(ArgumentError("phs_simulate: grid with sampled controllers is not supported (C3, finding 3)"))
        if Tsp > 0
            ncell = (t1 - t0) / Tsp
            abs(ncell - round(ncell)) < 1e-6 && abs(t0 / Tsp - round(t0 / Tsp)) < 1e-6 ||
                throw(ArgumentError("phs_simulate: the segment $t0 to $t1 s is not on the sampling grid of $Tsp s"))
            msub = max(1, ceil(Int, Tsp / hq - 1e-9))
            nsteps = msub * round(Int, ncell)
            twrun && abs(msub * Float64(h) - Tsp) > 1e-12 &&
                throw(ArgumentError("phs_simulate: h = $h s does not divide the sampling period $Tsp s"))
            hq = twrun ? Float64(h) : (t1 - t0) / nsteps
        end
        A = _jacobian(s, x, ub)
        lu_ = lu(I - (hq / 2) .* A)
        njac += 1
        L0 = phs_ledger(s, x, ub)                  # one-sided: this segment's inputs
        if q == 1
            E_start = copy(L0.energy)
            if sample_at === nothing
                save!(t0, L0)
            else
                while isamp <= length(samples) && samples[isamp] <= t0 + 1e-15
                    save!(samples[isamp], L0)
                    isamp += 1
                end
            end
        end
        fill!(dprev, 0.0)
        cored = !isempty(ps.cores)
        segJ = cored ? _core_segments(ps, x) : Int[]   # the cores' segments where the Newton matrix was taken
        for k in 1:nsteps
            @. x1 = x + dprev
            clock = cored ? clock0 + ps.omega_b * (t0 + k * hq) : nothing
            cored && (x1[ps.iclk] = clock)
            it = 0
            twrun && twnorton!()                   # the lines' history currents of this step (midpoint t0 + (k - 1/2) h)
            _bridge_window!(s, t0 + (k - 1) * hq, t0 + k * hq)   # switching bridges: this step's window
            _set_frame!(ub, ps, t0 + (k - 1) * hq)
            frozen = _freeze(s, x, ub)             # limiter decisions, held for this step
            _set_frame!(ub, ps, t0 + (k - 1) * hq + hq / 2)    # the step's midpoint
            ctx = nothing
            exactmode = false                      # the step residual's own Jacobian, once the contraction stalled
            res_prev = Inf
            while true
                @. xm = (x + x1) / 2
                _bridge_slope!(s, x, x1, ub, hq)   # switching bridges: the currents' secant of this iterate
                ctx = _step_residual!(F, fm, s, x, x1, ub, frozen, hq; clock)
                res = 0.0
                @inbounds for i in 1:n
                    res = max(res, abs(F[i]) / sc[i])
                end
                if res <= tol
                    worst = max(worst, res)
                    break
                end
                # with cores: a step that stopped contracting at the rounding floor of the core solves (within 100 tol)
                # is accepted and counted (`newton.floor`); the scheme identity reports what it costs
                if cored && it >= 6 && res <= 100 * tol && res >= res_prev / 2
                    worst = max(worst, res)
                    nfloor += 1
                    break
                end
                res_prev = res
                it += 1
                it > maxiter && error("phs_simulate: Newton did not converge at t = " *
                                      "$(t0 + (k - 1) * hq) (scaled residual $res)")
                cored && it >= 12 && (exactmode = true)
                if exactmode && (it - 12) % 4 == 0
                    # a slow contraction with cores: the step residual's own Jacobian (T2 finding 6), refreshed every
                    # four iterations while the contraction stays slow (T3 3c)
                    lu_ = lu(ForwardDiff.jacobian(y -> (Fy = similar(y); _step_residual!(Fy, similar(y), s, x, y, ub,
                                                                                          frozen, hq; clock); Fy), x1))
                    njac += 1
                elseif !exactmode && (it % 8 == 0 || (cored && _core_segments(ps, xm) != segJ))   # slow, or a segment changed
                    lu_ = lu(I - (hq / 2) .* _jacobian(s, xm, ub, frozen))
                    njac += 1
                    cored && (segJ = _core_segments(ps, xm))
                end
                x1 .-= lu_ \ F
            end
            iters += it
            if twrun                                # the lines' step with the converged midpoint bus voltages
                twJp .= twJk; twJk .= Jget(); hq_ref[] = hq
                ps.step[] = ctx                     # (with cores: the step's discrete-gradient currents, as the residual)
                fm_ = try phs_flows(ps, xm, ub) finally ps.step[] = nothing end
                vm = fm_.v
                pb = hq * sum(real(conj(vm[q]) * (ps.twG[q] * vm[q] - twJk[q])) for q in eachindex(ps.twG) if ps.twG[q] != 0)
                vm0 = tw0 ? fm_.z0.v0 : Float64[]
                tw0 && (pb += hq * sum(vm0[q] * (z0.G0[q] * vm0[q] - real(twJk[nJ+q])) for q in eachindex(vm0) if z0.G0[q] != 0))
                psum = 0.0; pabs = 0.0
                twk[] += 1; full = twk[] % 1000 == 0
                for (t, L, st, vf, vt) in Iterators.flatten(((zip(ps.tw, twL, twS, (vm[t.from] for t in ps.tw), (vm[t.to] for t in ps.tw))),
                                                             tw0 ? zip(z0.tw0, twL0, twS0z, (complex(vm0[t.from]) for t in z0.tw0),
                                                                       (complex(vm0[t.to]) for t in z0.tw0)) : ()))
                    r = tline_step!(L, st, vf, vt; full)
                    sup = hq * real(vf * conj(r.i1) + vt * conj(r.i2))
                    psum += sup; pabs += hq * (abs(vf * r.i1) + abs(vt * r.i2))
                    twacc.supply[] += sup; twacc.dS[] += r.dS; twacc.loss[] += r.loss; twacc.dinterp[] += r.dinterp
                    twacc.flow[] += r.flow
                    full && (twacc.worst[] = max(twacc.worst[], abs(sup - r.dS - r.loss - r.dinterp) / max(r.flow, 1e-300)))
                end
                twJn .= twnext()                    # the next midpoint's: quantities between midpoints interpolate linearly
                # the lumped system's boundary power to the lines (its ledger) against the lines' own supply
                twacc.boundary[] = max(twacc.boundary[], abs(pb - psum) / max(pabs, 1e-300))
            end
            @. dprev = x1 - x
            @. F = dprev / hq                      # the step's increment as a rate
            _bridge_accumulate!(s, xm, ub, hq)     # switching bridges: the step's applied voltage (the measurement)
            ps.step[] = ctx                        # the step's currents in the step's account (record section 23)
            Lm, Li = try
                phs_ledger(s, xm, ub), phs_ledger(s, xm, ub; dx = F)   # energy increments along x1 - x0
            finally
                ps.step[] = nothing
            end
            tn = t0 + (k - 1) * hq
            # a step ending on a controller call: its end is saved after the call (below), as PowerFactory
            # records the outputs of that step; the interior samples here (C3, finding 2)
            call = msub > 0 && k % msub == 0
            tlast = call ? tn + hq - 1e-12 : tn + hq + 1e-15
            if isamp <= length(samples) && samples[isamp] <= tlast
                xold .= x
                _set_frame!(ub, ps, tn)
                twrun && Jset!((twJp .+ twJk) ./ 2)         # the step's start: halfway between its two midpoints
                phs_rhs!(f0, s, xold, ub)
                _set_frame!(ub, ps, tn + hq)
                twrun && Jset!((twJk .+ twJn) ./ 2)
                phs_rhs!(f1, s, x1, ub)
                while isamp <= length(samples) && samples[isamp] <= tlast
                    th = clamp((samples[isamp] - tn) / hq, 0.0, 1.0)
                    h00, h10 = (1 + 2th) * (1 - th)^2, th * (1 - th)^2
                    h01, h11 = th^2 * (3 - 2th), th^2 * (th - 1)
                    @. xi = h00 * xold + h10 * hq * f0 + h01 * x1 + h11 * hq * f1
                    _set_frame!(ub, ps, samples[isamp])
                    twrun && Jset!(twJof(samples[isamp], tn))
                    save!(samples[isamp], phs_ledger(s, xi, ub), xi)
                    isamp += 1
                end
            end
            x .= x1
            _set_frame!(ub, ps, tn + hq)
            twrun && Jset!((twJk .+ twJn) ./ 2)       # the step's end lies halfway between the two midpoints
            L1 = phs_ledger(s, x, ub)
            dH = 0.0
            @inbounds for c in 1:nc
                dE = hq * Li.rate[c]
                dH += dE
                p0, pm, p1 = L0.port[c] - L0.loss[c], Lm.port[c] - Lm.loss[c], L1.port[c] - L1.loss[c]
                scheme[c] += dE - hq * pm
                scheme_max[c] = max(scheme_max[c], abs(scheme[c]))
                incsum[c] += dE
                simpson[c] += dE - hq * (p0 + 4pm + p1) / 6
                thr[c] += hq * (L0.tport[c] + 4 * Lm.tport[c] + L1.tport[c]) / 6
                moved[c] += abs(dE)
            end
            tot[:scheme] += dH - hq * (Lm.supply - Lm.losses)
            tot[:scheme_max] = max(tot[:scheme_max], abs(tot[:scheme]))
            tot[:simpson] += dH - hq * ((L0.supply - L0.losses) + 4 * (Lm.supply - Lm.losses) +
                                         (L1.supply - L1.losses)) / 6
            tot[:moved] += abs(dH)
            # a controller call: it changes only controller states, after the step's account above (made
            # with the step's own end, C3 finding 1); the next step starts from the account after it
            if call
                Epre = L1.energy
                _sample!(s, x, ub)
                L1 = phs_ledger(s, x, ub)
                # a relay trip (review 76) removes the tripped motor units with their stored energy: a switching jump,
                # booked in the motor's account (no other component's energy changes at a sample)
                for c in motorc
                    dj = L1.energy[c] - Epre[c]
                    dj == 0 && continue
                    incsum[c] += dj
                    tot[:switched] += dj
                end
                while isamp <= length(samples) && samples[isamp] <= tn + hq + 1e-15
                    save!(samples[isamp], L1)
                    isamp += 1
                end
            end
            L0 = L1
            sample_at === nothing && (k % stride == 0 || k == nsteps) && save!(t0 + k * hq, L1)
            _set_frame!(ub, ps, t0 + k * hq)
            monitor === nothing || monitor(t0 + k * hq, x, ub)
        end
        push!(segments, (; t0, t1, h = hq, steps = nsteps, w_max = rule.w_max, r_max = rule.r_max,
                         fault = any(>(0), ub.gf)))
    end
    finally
        _bridge_stop!(s)                           # switching bridges: back to the averaged bridge outside a run
    end
    tot[:throughput] = sum(thr)
    total = (; scheme = tot[:scheme], scheme_max = tot[:scheme_max], simpson = tot[:simpson],
             throughput = tot[:throughput], moved = tot[:moved],
             switched = tot[:switched],               # energy removed by relay trips (motor units leaving the system)
             # travelling-wave lines: energy the buses gave the lines, their storage change, resistor losses and
             # interpolation dissipation, and the worst per-line step identity relative to its gross wave energy
             lines = twrun ? (; supply = twacc.supply[], dS = twacc.dS[], loss = twacc.loss[], interp = twacc.dinterp[],
                               worst = twacc.worst[], storage = (Sf = sum(tline_storage(L, st) for (L, st) in zip(vcat(twL, twL0), vcat(twS, twS0z)))),
                               # independent: the full storage at the ends against the summed step changes
                               storage_closure = (Sf - twS0) - twacc.dS[],
                               boundary = twacc.boundary[]) :
                     nothing)
    closure = incsum .- (Es[end] .- E_start)
    # a record of numbers is a vector, a record of vectors a matrix with one column per saved time
    records = Dict{String,Any}(n => (isempty(v) || v[1] isa Number) ? collect(Float64, v) :
                                    reduce(hcat, [collect(Float64, y) for y in v]) for (n, _, v) in rec)
    return PHRun(ts, reduce(hcat, Xs), reduce(hcat, Vs), _cnames(s), _ckind(s), reduce(hcat, Es),
                 reduce(hcat, Ls), reduce(hcat, Ps), Ss, Hs, scheme, scheme_max, closure, simpson,
                 thr, moved, total,
                 segments, (; iterations = iters, jacobians = njac, worst_residual = worst, floor = nfloor),
                 time() - wall, records)
end

"""
    phs_simulate_bdf(s, x0, u; tstop, faults = PHFault[], saveat, reltol = 1e-10, abstol = 1e-12)
        -> (t, X)

The same model integrated by SUNDIALS CVODE (variable-order BDF), restarted at every fault
switching time: an independent trajectory for cross-checking `phs_simulate`. In the stationary frame every
right-hand side sees the frame phasor of its own time; a caller that evaluates ledgers or voltages from the returned
states must set the frame of each saved time itself (`_set_frame!`). `abstol` multiplies
`phs_scales`. BDF damps energy numerically; no energy claim is made for this solution.
"""
function phs_simulate_bdf(s::PHModel, x0::AbstractVector{Float64}, u::PHInput{Float64};
                          tstop::Real, faults::AbstractVector{PHFault} = PHFault[],
                          saveat::AbstractVector{<:Real}, reltol::Real = 1e-10,
                          abstol::Real = 1e-12)
    isempty(_phys(s).tw) || throw(ArgumentError("phs_simulate_bdf: travelling-wave lines (their history is not integrated here)"))
    any(_unresolved, faults) &&
        throw(ArgumentError("phs_simulate_bdf: current-zero clearing needs located openings (phs_resolve_clearing)"))
    _sample_period(s) > 0 &&
        throw(ArgumentError("phs_simulate_bdf: sampled controllers are not supported (C3, finding 4)"))
    ub = PHInput{Float64}(u.Tm, u.vf, zeros(length(u.gf)), u.const_power, u.uc)
    f!(dx, x, p, t) = (_set_frame!(ub, _phys(s), t); phs_rhs!(dx, s, x, ub); nothing)
    atol = abstol .* phs_scales(s, collect(x0))
    x = collect(x0)
    bp = _breakpoints(faults, tstop)
    ts, cols = Float64[], Vector{Float64}[]
    for q in 1:length(bp)-1
        t0, t1 = bp[q], bp[q+1]
        _fault_conductance!(ub.gf, _phys(s), faults, t0)
        sv = [Float64(t) for t in saveat if t0 < t <= t1 || (q == 1 && t == t0)]
        prob = SciMLBase.ODEProblem(f!, x, (t0, t1))
        sol = SciMLBase.solve(prob, Sundials.CVODE_BDF(); reltol, abstol = atol,
                              saveat = unique(vcat(sv, t1)), maxiters = 10_000_000)
        SciMLBase.successful_retcode(sol) || error("phs_simulate_bdf: CVODE stopped at $(sol.t[end])")
        for t in sv
            j = findfirst(y -> abs(y - t) <= 1e-12, sol.t)
            push!(ts, t)
            push!(cols, copy(sol.u[j]))
        end
        x = copy(sol.u[end])
    end
    return ts, reduce(hcat, cols)
end

"""
    phs_write_run(path, r::PHRun; model = nothing, meta = Dict(), states = true) -> path

Everything a run produced, to a JLD2 file, as plain arrays and dictionaries so that it can be
read without this code (`phs_read_run`, or any JLD2/HDF5 reader): the saved times `t`; the state
`X` (one column per saved time) with the state names of `model`; the bus voltages `V` with the
bus ids; the energy account per component (`energy`, `loss`, `port`, with `names`, `kind`) and for
the system (`supply`, `H`); the accumulated balances (`scheme`, `scheme_max`, `closure`,
`simpson`, `throughput`, `moved`, `total`); the segments and the Newton statistics; every record
declared with `phs_simulate(...; record)`; the git commit of the code that ran it, with a flag
if the working tree had changes; and `meta`, whatever the caller needs to say what was run.
Anything not recorded can be computed afterwards from `X` with the same model, without
simulating again. With `states = false` the state is left out (a smaller file; then only the
records, the voltages and the account remain).
"""
function phs_write_run(path::AbstractString, r::PHRun; model = nothing, meta = Dict{String,Any}(),
                       states::Bool = true)
    mkpath(dirname(abspath(path)))
    commit, dirty = _git_state(PHPS_ROOT)
    nt2dict(x) = Dict{String,Any}(String(k) => v for (k, v) in pairs(x))
    JLD2.jldsave(path;
                 t = r.t, X = states ? r.X : zeros(0, length(r.t)),
                 state_names = model === nothing ? String[] : phs_state_names(model),
                 V = r.V, bus_ids = model === nothing ? Int[] : collect(Int, _phys(model).net.net.bus_ids),
                 names = r.names, kind = String.(r.kind), energy = r.energy, loss = r.loss, port = r.port,
                 supply = r.supply, H = r.H, scheme = r.scheme, scheme_max = r.scheme_max,
                 closure = r.closure, simpson = r.simpson, throughput = r.throughput, moved = r.moved,
                 total = nt2dict(r.total), segments = [nt2dict(g) for g in r.segments],
                 newton = nt2dict(r.newton), wall = r.wall, records = r.records,
                 git_commit = commit === nothing ? "" : String(commit), git_dirty = dirty === true,
                 meta = Dict{String,Any}(string(k) => v for (k, v) in meta))
    return path
end

"""    phs_read_run(path) -> Dict{String,Any}: what `phs_write_run` wrote, by name."""
phs_read_run(path::AbstractString) = JLD2.load(path)
