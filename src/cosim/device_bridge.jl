# The device-level fast side of a converter's switching bridge (COSIM_DESIGN.md sections 3, 5, 8.3, 8.4, 8.9; review 102).
#
# Attached to a `SwitchingBridge` (its field `dl`): the bridge's PWM scheduler gives the gate commands (fixed at the
# carrier samples), its table model stays the Newton surrogate (the derivatives), and for the step's values each
# Newton residual evaluation runs the three device legs over the step's window from the committed state, with the
# slow side's midpoint DC voltage and current held (window-held port coupling). The returned EMF and DC current are the
# window averages of the legs' discrete port efforts and flows, so the grid's port energy equals the legs' port energy
# exactly (E_j (3/2) V_pk I_b = 1). The accepted step's candidate (the one of the final passing residual) is committed
# by `_bridge_commit!` after the step.

"""The accumulated ledger of a device-level bridge (J per converter unit, i.e. one 720 kVA unit of N)."""
mutable struct DeviceLedger
    E_dc::Float64; E_ac::Float64; E_gate::Float64; dH::Float64
    D_ch::Float64; D_rd::Float64; D_bd::Float64; D_rg::Float64; D_pkg::Float64
    defect_abs::Float64            # sum of |per-step defects|
    gross::Float64
    table_psw::Float64             # the table model's losses along the same steps (J per unit): switching, conduction
    table_pcond::Float64
    time::Float64
    steps_fine::Int; steps_coarse::Int; retries::Int
    windows::Int; evals::Int       # committed windows; fast window runs (Newton evaluations)
    port_identity::Float64         # max |Re(ebar conj i) h - E_j E_ac| / |E_j E_ac|
    nchg::Int                      # scheduled pole-state changes (gate events) in the measured windows
    max_ratio::Float64             # the slow Newton's largest residual ratio (contraction) seen with this bridge
    nfallback::Int                 # steps that switched to the actual interface Jacobian
    slivers::Int                   # gate states shorter than 1 ps merged into the next segment (measured windows)
    guess_retries::Int             # leg steps that succeeded only after the channel guesses were reset (whole run)
end
DeviceLedger() = DeviceLedger(zeros(14)..., 0, 0, 0, 0, 0, 0.0, 0, 0.0, 0, 0, 0)
dissipation(l::DeviceLedger) = l.D_ch + l.D_rd + l.D_bd + l.D_rg + l.D_pkg

"""A device's switching edge in progress (data-sheet definitions, Figs. 30-31): 1 turn-on waiting for 10 % of the
current, 2 integrating until 10 % of V_dc; 3 turn-off waiting for 10 % of V_dc, 4 integrating until 10 % of the current."""
mutable struct EdgeTrack
    mode::Int
    Iref::Float64
    E::Float64
    tstart::Float64
    vprev::Float64      # the previous step's v and i (the crossings interpolated within their steps, as in group B)
    iprev::Float64
end
EdgeTrack() = EdgeTrack(0, 0.0, 0.0, 0.0, 0.0, 0.0)
Base.copy(e::EdgeTrack) = EdgeTrack(e.mode, e.Iref, e.E, e.tstart, e.vprev, e.iprev)
function Base.copyto!(a::EdgeTrack, b::EdgeTrack)
    a.mode, a.Iref, a.E, a.tstart, a.vprev, a.iprev = b.mode, b.Iref, b.E, b.tstart, b.vprev, b.iprev
    return a
end

mutable struct DeviceLevelBridge
    mode::Symbol                     # :device (the fast side runs) or :record (the table model, its losses recorded)
    t_meas0::Float64                 # the ledger accumulates from this time on (a settling start before it)
    leg::SiCLeg
    npar::Int
    Von::Float64
    Voff::Float64
    solver::SiCLegSolver
    st::Vector{SiCLegState}          # committed
    ft::Vector{FineTracker}
    cst::Vector{SiCLegState}         # candidate
    cft::Vector{FineTracker}
    et::Vector{EdgeTrack}            # per leg: upper, lower (committed: 6) and their candidates
    cet::Vector{EdgeTrack}
    cevents::Vector{NTuple{6,Float64}}   # the candidate window's completed hard edges: (t, kind 1 on / 2 off, I_unit A, E_dev J, V, leg)
    events::Vector{NTuple{6,Float64}}    # committed (in the measured interval)
    abandoned::Int
    init::Bool
    fresh::Bool
    has_cand::Bool
    committed::Bool                  # the candidate was committed (its averages stay valid within its window)
    jac::Matrix{Float64}             # the actual interface Jacobian (the fallback), and whether the Newton matrix uses it
    use_actual::Bool
    force_fallback::Bool             # (a test: the fallback at every step that needs a second iteration)
    cand_key::NTuple{5,Float64}      # (a, b, v_dc, Re i, Im i)
    cand_val::NamedTuple
    cand_led::Vector{Float64}        # E_dc, E_ac, E_gate, dH, D_ch, D_rd, D_bd, D_rg, D_pkg, |defect|, gross (J per unit)
    cand_steps::NTuple{4,Int}
    cand_ident::Float64
    led::DeviceLedger
    trace::Union{Nothing,Vector{NamedTuple}}   # per committed window (optional): t, losses, edge counts
    wave::Any                                  # optional waveform recording (`record_waves!`), or nothing
end

"""
    DeviceLevelBridge(; T = 125, cgd = :charge, RGext = 1.0, Lsigma = 8.4e-9, h_f = 0.25e-9, npar = 3, trace = false)

The fast side for one converter: three legs of CAB760M12HM3 positions (`cab760m12hm3`), each module in its own
8.4 nH loop (the declared topology, COSIM_DESIGN.md 8.9).
"""
function DeviceLevelBridge(; T::Real = 125, cgd::Symbol = :charge, RGext::Real = 1.0, Lsigma::Real = 8.4e-9,
                           h_f::Real = 0.25e-9, npar::Int = 3, trace::Bool = false, mode::Symbol = :device,
                           t_meas0::Real = 0.0)
    mode in (:device, :record) || throw(ArgumentError("DeviceLevelBridge: mode = :device or :record"))
    c = cab760m12hm3(; T, cgd, RGext)
    leg = SiCLeg(c.dev, c.RG, Float64(Lsigma), 106.5e-6, 126.3e-6)
    z() = SiCLegState(0, 0, 0, 0, 0, 0, 0)
    f() = FineTracker(; h_f = Float64(h_f), L = Float64(Lsigma), defer = true)
    return DeviceLevelBridge(mode, Float64(t_meas0), leg, npar, 15.0, -4.0, SiCLegSolver(leg), [z() for _ in 1:3], [f() for _ in 1:3],
                             [z() for _ in 1:3], [f() for _ in 1:3], [EdgeTrack() for _ in 1:6], [EdgeTrack() for _ in 1:6],
                             NTuple{6,Float64}[], NTuple{6,Float64}[], 0, false, false, false, false, zeros(3, 3), false,
                             false, ntuple(_ -> NaN, 5),
                             (; m = 0.0im, udrop = 0.0im, psw = 0.0, pcond = 0.0), zeros(11), (0, 0, 0, 0), 0.0,
                             DeviceLedger(), trace ? NamedTuple[] : nothing, nothing)
end

"""A run starts (`bridge_start!`): the legs are initialised at the first window from its held values."""
function device_start!(d::DeviceLevelBridge)
    d.init = false; d.fresh = false; d.has_cand = false; d.committed = false; d.use_actual = false
    d.led = DeviceLedger(); empty!(d.events); d.abandoned = 0
    d.trace === nothing || empty!(d.trace)
    d.cand_key = ntuple(_ -> NaN, 5)
    return d
end

# the gate voltages (upper referenced to the pole, lower to DC-) of a scheduled leg state
@inline _gates(d::DeviceLevelBridge, st::Symbol) = st === :U ? (d.Von, d.Voff) : st === :L ? (d.Voff, d.Von) : (d.Voff, d.Voff)

# a hard edge's integral by the data sheet's definitions (Figs. 30-31) at the die boundary: v = phibar_D, i = the drain
# current; a completed edge is pushed to the candidate's events (t, kind 1 on / 2 off / 3 abandoned, I A, E J, V_dc) per unit
function _edge_step!(d::DeviceLevelBridge, e::EdgeTrack, v, i, t, h, Vdc, evs)
    np = d.npar
    if e.mode != 0 && t - e.tstart > 5e-6
        push!(evs, (t, 3.0, e.Iref * np, 0.0, Vdc)); e.mode = 0
    elseif e.mode == 1                            # turn-on: waiting for i rising through 10 % of I_ref
        thr = 0.1 * e.Iref
        if i >= thr
            f = e.iprev < thr ? (i - thr) / (i - e.iprev) : 1.0      # the part of the step after the crossing
            e.mode = 2; e.E = f * h * v * i
        end
    elseif e.mode == 2                            # until v falls through 10 % of V_dc
        thr = 0.1 * Vdc
        if v <= thr
            g = e.vprev > thr ? (e.vprev - thr) / (e.vprev - v) : 0.0  # the part of the step before the crossing
            e.E += g * h * v * i
            push!(evs, (t, 1.0, e.Iref * np, e.E * np, Vdc)); e.mode = 0
        else
            e.E += h * v * i
        end
    elseif e.mode == 3                            # turn-off: waiting for v rising through 10 % of V_dc
        thr = 0.1 * Vdc
        if v >= thr
            f = e.vprev < thr ? (v - thr) / (v - e.vprev) : 1.0
            e.mode = 4; e.E = f * h * v * i
        end
    elseif e.mode == 4                            # until i falls through 10 % of I_ref
        thr = 0.1 * e.Iref
        if i <= thr
            g = e.iprev > thr ? (e.iprev - thr) / (e.iprev - i) : 0.0
            e.E += g * h * v * i
            push!(evs, (t, 2.0, e.Iref * np, e.E * np, Vdc)); e.mode = 0
        else
            e.E += h * v * i
        end
    end
    e.vprev = v; e.iprev = i
    return nothing
end

# a scheduled transition old -> new of a leg at t with the module's held current ik: start the hard edge's integral
function _edge_start!(eu::EdgeTrack, el::EdgeTrack, old::Symbol, new::Symbol, ik, t)
    abs(ik) < 10.0 && return nothing             # (no hard edge below 10 A per module: thresholds at the noise)
    if old === :U && new === :D && ik > 0        # upper off, forward current: hard turn-off
        eu.mode, eu.Iref, eu.E, eu.tstart = 3, ik, 0.0, t
    elseif old === :L && new === :D && ik < 0    # lower off
        el.mode, el.Iref, el.E, el.tstart = 3, -ik, 0.0, t
    elseif old === :D && new === :U && ik > 0    # upper on, taking the current from the lower diode: hard turn-on
        eu.mode, eu.Iref, eu.E, eu.tstart = 1, ik, 0.0, t
    elseif old === :D && new === :L && ik < 0    # lower on
        el.mode, el.Iref, el.E, el.tstart = 1, -ik, 0.0, t
    end
    return nothing
end

# advance one leg over [ta, tb) with the gates constant; fine steps while the tracker is active; a coarse step failing
# in Newton is split (the identity holds for any step)
function _leg_segment!(d::DeviceLevelBridge, s::SiCLegState, ft::FineTracker, ta, tb, Vdc, ik, Vu, Vl, acc, cnt,
                       eu::EdgeTrack, el::EdgeTrack, evs, rows = nothing)
    t = ta
    # segments below 1 ps are skipped: a gate change a rounding error from a window boundary (e.g. 2.9e-16 s) would
    # otherwise take a step at which the absolute residual tolerance cannot be met
    while tb - t > 1e-12
        # fine steps of h_f; a remainder below 1.5 h_f merged into the last step (no tiny steps from the summed times)
        h = ft.active ? (tb - t < 1.5 * ft.h_f ? tb - t : ft.h_f) : tb - t
        vdu0, vdl0 = s.vds_u, s.vds_l
        ok, l = leg_step!(d.solver, s, h, Vdc, ik, Vu, Vl)
        while !ok
            h /= 4
            if h < 1e-13
                s0 = (s.vgs_u, s.vds_u, s.vgs_l, s.vds_l, s.iL, s.vch_u, s.vch_l)
                open(joinpath(tempdir(), "device_bridge_failure.txt"), "w") do io
                    println(io, (; t, ta, tb, Vdc, ik, Vu, Vl, state = s0, fine = ft.active, iters = d.solver.iters,
                                 res = d.solver.maxres))
                end
                error("device bridge: the leg's Newton failed at t = $t (state $s0, Vdc $Vdc, ik $ik, gates $Vu $Vl; " *
                      "dumped to $(joinpath(tempdir(), "device_bridge_failure.txt")))")
            end
            cnt[3] += 1
            ok, l = leg_step!(d.solver, s, h, Vdc, ik, Vu, Vl)
        end
        t += h
        acc[1] += l.E_dc; acc[2] += l.E_ac; acc[3] += l.E_gate; acc[4] += l.dH
        acc[5] += l.D_ch; acc[6] += l.D_rd; acc[7] += l.D_bd; acc[8] += l.D_rg; acc[9] += l.D_pkg
        acc[10] += abs(defect(l)); acc[11] += abs(l.E_dc) + abs(l.E_ac) + abs(l.E_gate)
        acc[12] += h * l.phip; acc[13] += h * l.iLm
        ft.active ? (cnt[1] += 1) : (cnt[2] += 1)
        fine_update!(ft, t, h, s, l, vdu0, vdl0)
        _edge_step!(d, eu, l.phiDu, l.idu, t, h, Vdc, evs)
        _edge_step!(d, el, l.phiDl, l.idl, t, h, Vdc, evs)
        # (the recorded values are the step's discrete efforts and currents: the ones the loss integrals use)
        rows === nothing || push!(rows, (t, l.phiDu, l.phiDl, l.idu, l.idl, ik, Vu, Vl, Vdc))
    end
    return nothing
end

"""
    device_average!(d, br, i_net, v_dc) -> (; m, udrop, psw, pcond)

The step's averages for the bridge's interface (`bridge_average`): from the window run (COSIM_DESIGN.md 8.9),
m = ebar / v_dc, udrop = 0, psw = the net bridge power (DC power - Re(ebar conj i)); pcond = 0. The first call of a
fresh Newton residual (the flag set by `_bridge_slope!`) consumes the flag on every path and runs the fast side (or
returns the candidate if its values are the candidate's); any other call returns the candidate within its window, or the
table model's averages in a new window: no fast run outside the Newton residuals (review 103, finding 1).
"""
function device_average!(d::DeviceLevelBridge, br::SwitchingBridge, i_net, v_dc)
    key = (br.a, br.b, Float64(v_dc), Float64(real(i_net)), Float64(imag(i_net)))
    if d.mode === :record                         # the table model; the fresh residual's values kept for the record
        r = bridge_average_table(br, i_net, v_dc)
        if d.fresh
            d.fresh = false
            d.cand_key = key; d.cand_val = r; d.has_cand = true; d.committed = false
        end
        return r
    end
    if d.fresh
        d.fresh = false
        d.has_cand && !d.committed && key == d.cand_key && return d.cand_val
        return _device_window!(d, br, i_net, v_dc, true).val
    end
    d.has_cand && key == d.cand_key && return d.cand_val
    d.has_cand && key[1] == d.cand_key[1] && key[2] == d.cand_key[2] && return d.cand_val
    return bridge_average_table(br, i_net, v_dc)
end

# the legs' start states for a run of the window: the committed states, or (before the first commit) the conduction
# states at this run's held values, staged in the run (the transaction: review 103, finding 3)
function _start_states(d::DeviceLevelBridge, br::SwitchingBridge, Vdc, ist, a)
    if d.init
        return [copy(d.st[k]) for k in 1:3], [copy(d.ft[k]) for k in 1:3]
    end
    sts = SiCLegState[]; fts = FineTracker[]
    for k in 1:3
        ik = br.Ipk * real(ist * cis(-_LEG[k])) / d.npar
        stt = _leg_state(br, k, a)
        if stt === :D                             # (not at a run's start: bridge_start! begins without a dead time)
            ts, ss = br.tchg[k], br.schg[k]; j = searchsortedlast(ts, a)
            stt = j > 1 ? ss[j-1] : :L
        end
        push!(sts, leg_equilibrium(d.leg, Vdc, ik, stt === :U; Von = d.Von, Voff = d.Voff))
        push!(fts, FineTracker(; h_f = d.ft[k].h_f, L = d.leg.L, defer = true))
    end
    return sts, fts
end

"""
    _device_window!(d, br, i_net, v_dc, store) -> (; val, y)

One run of the fast side over the bridge's window [a, b) from the committed (or staged initial) state with the held
values. With `store` the run becomes the candidate; without, nothing of the bridge changes (finite-difference runs).
`y` = (Re m, Im m, psw).
"""
const _DL_DIAG = Ref{Any}(nothing)               # set to a Vector to record the windows of each slow step

function _device_window!(d::DeviceLevelBridge, br::SwitchingBridge, i_net, v_dc, store::Bool)
    a, b = br.a, br.b
    h = b - a
    tm = (a + b) / 2
    rot = br.stationary ? 1.0 + 0im : cis(br.omega_b * tm + br.theta0)
    Vdc = v_dc * 2 * br.Vpk                       # V
    ist = i_net * rot
    sts, fts = _start_states(d, br, Vdc, ist, a)
    ets = [copy(e) for e in d.et]
    evs = NTuple{5,Float64}[]
    evk = Int[]                                                    # the leg of each edge in evs
    recw = d.wave !== nothing && store && b > d.wave.t0 && a < d.wave.t1
    wrows = recw ? NTuple{9,Float64}[] : nothing
    sv = 0.0im
    acc = zeros(13)
    cnt = [0, 0, 0, 0]
    for k in 1:3
        sk, fk = sts[k], fts[k]
        eu, el = ets[2k-1], ets[2k]
        ik = br.Ipk * real(ist * cis(-_LEG[k])) / d.npar          # the module's held phase current (A)
        accl = zeros(13)
        ts, ss = br.tchg[k], br.schg[k]
        # change times within 1 ps of a window boundary snap to it (a change just before b belongs to the next window,
        # where it counts as at its start); two changes closer than 1 ps: the middle state is skipped, its sliver merged
        # into the next segment: every leg covers exactly [a, b) (review 104, Gemini finding 7)
        snap(tc) = abs(tc - a) <= 1e-12 ? a : (abs(tc - b) <= 1e-12 ? b : tc)
        j = searchsortedlast(ts, a + 1e-12)
        stt = j == 0 ? ss[1] : ss[j]
        if j > 0 && snap(ts[j]) == a                               # a change at the window's start belongs to it
            fine_command!(fk, a)
            j > 1 && _edge_start!(eu, el, ss[j-1], stt, ik, a)
        end
        t = a
        jn = j + 1
        while true
            tn = (jn <= length(ts) && snap(ts[jn]) < b) ? max(snap(ts[jn]), t) : b
            Vu, Vl = _gates(d, stt)
            if tn - t > 1e-12
                _leg_segment!(d, sk, fk, t, tn, Vdc, ik, Vu, Vl, accl, cnt, eu, el, evs, recw && k == d.wave.leg ? wrows : nothing)
                t = tn
            elseif tn < b
                cnt[4] += 1                                        # a sliver: the next state starts at t
            end
            tn >= b && break
            _edge_start!(eu, el, stt, ss[jn], ik, t)
            stt = ss[jn]; jn += 1
            fine_command!(fk, t)
        end
        b - t > 1e-12 && error("device bridge: leg $k covered [a, $t) of [a, $b)")
        fine_window_end!(fk)                                       # a pending fine exit takes effect at b (review 105)
        append!(evk, fill(k, length(evs) - length(evk)))
        acc .+= accl
        vbar = accl[12] / h - Vdc / 2                              # the pole's discrete effort average, from the DC midpoint
        sv += cis(_LEG[k]) * vbar
    end
    np = d.npar
    led = acc[1:11] .* np                                          # J per unit (the module's x npar)
    ebar = (2 / (3 * br.Vpk)) * conj(rot) * sv                     # system pu (network frame)
    Pdc = Vdc * acc[13] * np / h                                   # W per unit
    psw = br.Ej * Pdc - real(ebar * conj(i_net))                   # net bridge power, system pu
    val = (; m = ebar / v_dc, udrop = zero(ebar), psw, pcond = 0.0)
    # the Newton-failure diagnostic (2026-11-14): every evaluated window of the current step, the inputs, the outputs and
    # the fast side's discrete decisions (fine / coarse steps, subdivisions, slivers, guess resets), when switched on
    if _DL_DIAG[] !== nothing && store
        dg = _DL_DIAG[]
        (!isempty(dg) && dg[end].a != a) && empty!(dg)
        push!(dg, (; a, b, v_dc = Float64(v_dc), i_net = ComplexF64(i_net), m = ComplexF64(val.m), psw = Float64(psw),
                   cnt = copy(cnt), nretry = d.solver.nretry, ident = abs(br.Ej * led[2] - real(ebar * conj(i_net)) * h)))
    end
    y = (real(val.m), imag(val.m), psw)
    store || return (; val, y)
    for k in 1:3
        copyto!(d.cst[k], sts[k]); copyto!(d.cft[k], fts[k])
    end
    for e in 1:6
        copyto!(d.cet[e], ets[e])
    end
    empty!(d.cevents); append!(d.cevents, [(e..., Float64(k)) for (e, k) in zip(evs, evk)])
    if d.wave !== nothing
        empty!(d.wave.crow); recw && append!(d.wave.crow, wrows)
    end
    eac = br.Ej * led[2]                                           # the port identity (J per unit -> pu s)
    d.cand_ident = abs(real(ebar * conj(i_net)) * h - eac) / max(abs(eac), 1e-300)
    d.cand_key = (a, b, Float64(v_dc), Float64(real(i_net)), Float64(imag(i_net)))
    d.cand_val = val
    d.cand_led .= led
    d.cand_steps = (cnt[1], cnt[2], cnt[3], cnt[4])
    d.has_cand = true
    d.committed = false
    d.led.evals += 1
    return (; val, y)
end

"""
    device_jacobian!(d, br)

The actual interface Jacobian of the fast side at the candidate's held values: d(Re m, Im m, psw) / d(v_dc, Re i, Im i)
by forward differences (three runs that change nothing of the bridge). Used by the Newton fallback (design 8.3).
"""
function device_jacobian!(d::DeviceLevelBridge, br::SwitchingBridge)
    _, _, v, ir, ii = d.cand_key
    y0 = (real(d.cand_val.m), imag(d.cand_val.m), d.cand_val.psw)
    sc = max(abs(complex(ir, ii)), 1e-2)
    for (j, (dv, dr, di)) in enumerate(((1e-7 * max(abs(v), 1.0), 0.0, 0.0), (0.0, 1e-7 * sc, 0.0), (0.0, 0.0, 1e-7 * sc)))
        y = _device_window!(d, br, complex(ir + dr, ii + di), v + dv, false).y
        del = dv + dr + di
        for r in 1:3
            d.jac[r, j] = (y[r] - y0[r]) / del
        end
    end
    return d.jac
end

"""The bridge's averages for dual-number inputs with the actual interface Jacobian (`use_actual`): the candidate's values,
the partials J times the inputs' partials."""
function device_dual(d::DeviceLevelBridge, br::SwitchingBridge, i_net, v_dc)
    T = typeof(real(i_net) + v_dc)
    v = convert(T, v_dc); ir = convert(T, real(i_net)); ii = convert(T, imag(i_net))
    pv, pr, pii = ForwardDiff.partials(v), ForwardDiff.partials(ir), ForwardDiff.partials(ii)
    J = d.jac
    y0 = (real(d.cand_val.m), imag(d.cand_val.m), d.cand_val.psw)
    mk(r) = T(y0[r], J[r, 1] * pv + J[r, 2] * pr + J[r, 3] * pii)
    m = complex(mk(1), mk(2))
    return (; m, udrop = zero(m), psw = mk(3), pcond = zero(T))
end

"""Commit the accepted step's candidate (after the step; `_bridge_commit!`), with the table model's losses along it."""
function device_commit!(d::DeviceLevelBridge, br::SwitchingBridge)
    d.has_cand && d.committed && d.cand_key[1] == br.a && d.cand_key[2] == br.b && return d    # (idempotent)
    # the accepted step's candidate: the one of the final passing residual, in this step's window (review 103, 2)
    (d.has_cand && d.cand_key[1] == br.a && d.cand_key[2] == br.b) ||
        error("device bridge: no candidate for the accepted step [$(br.a), $(br.b)) (candidate $(d.cand_key))")
    d.use_actual = false
    d.init = true
    if d.mode === :device
        for k in 1:3
            copyto!(d.st[k], d.cst[k]); copyto!(d.ft[k], d.cft[k])
        end
        for e in 1:6
            copyto!(d.et[e], d.cet[e])
        end
    end
    nchg = sum(count(t -> br.a <= t < br.b, br.tchg[k]) for k in 1:3)
    if br.b <= d.t_meas0                          # settling: commit the state, do not accumulate
        d.committed = true
        return d
    end
    L = d.led
    L.nchg += nchg
    d.wave === nothing || append!(d.wave.rows, d.wave.crow)
    if d.mode === :device
        for ev in d.cevents
            ev[2] == 3.0 ? (d.abandoned += 1) : push!(d.events, ev)
        end
    end
    c = d.mode === :device ? d.cand_led : zeros(11)
    L.E_dc += c[1]; L.E_ac += c[2]; L.E_gate += c[3]; L.dH += c[4]
    L.D_ch += c[5]; L.D_rd += c[6]; L.D_bd += c[7]; L.D_rg += c[8]; L.D_pkg += c[9]
    L.defect_abs += c[10]; L.gross += c[11]
    h = br.b - br.a
    # the table model at the same held values (the replay of COSIM_DESIGN.md 8.6): pu power x h / E_j -> J per unit
    _, _, v, ir, ii = d.cand_key
    rt = bridge_average_table(br, complex(ir, ii), v)
    L.table_psw += rt.psw * h / br.Ej
    L.table_pcond += rt.pcond * h / br.Ej
    L.time += h
    L.steps_fine += d.cand_steps[1]; L.steps_coarse += d.cand_steps[2]; L.retries += d.cand_steps[3]
    L.slivers += d.cand_steps[4]
    L.guess_retries = d.solver.nretry
    L.windows += 1
    L.port_identity = max(L.port_identity, d.cand_ident)
    if d.trace !== nothing
        push!(d.trace, (; t = br.b, i = complex(ir, ii), v_dc = v, m = d.cand_val.m, D = sum(c[5:9]), dH = c[4],
                        psw_tab = rt.psw * h / br.Ej, pcond_tab = rt.pcond * h / br.Ej))
    end
    d.committed = true
    return d
end

_bridge_commit!(s) = nothing
_bridge_commit!(cs::PHControlled) =
    foreach(g -> (_has_sw(g) && g.dc.sw.dl !== nothing && device_commit!(g.dc.sw.dl, g.dc.sw)), cs.conv)

# the Newton safeguard (design 8.3; review 103, finding 4): the residual ratio is measured at every iteration; above 0.5,
# the device bridges' actual interface Jacobians replace their surrogates in the Newton matrix (refreshed by the caller)
_bridge_fallback!(s, it, res, res_prev) = false
function _bridge_fallback!(cs::PHControlled, it, res, res_prev)
    fb = false
    for g in cs.conv
        (_has_sw(g) && g.dc.sw.dl !== nothing && g.dc.sw.dl.mode === :device) || continue
        d = g.dc.sw.dl
        (it >= 1 && isfinite(res_prev) && res_prev > 0) || continue
        ratio = res / res_prev
        d.led.max_ratio = max(d.led.max_ratio, ratio)
        if (ratio > 0.5 || d.force_fallback) && !d.use_actual && d.has_cand
            device_jacobian!(d, g.dc.sw)
            d.use_actual = true
            d.led.nfallback += 1
            fb = true
        elseif d.use_actual && ratio > 0.5 && d.has_cand && it % 4 == 0
            device_jacobian!(d, g.dc.sw)                           # still slow after the fallback: refreshed (review 105)
            d.led.nfallback += 1
            fb = true
        end
    end
    return fb
end

"""Record one leg's per-step waveforms (discrete drain efforts, drain currents, the held phase current, the drivers, V_dc)
over [t0, t1) of the committed steps."""
record_waves!(d::DeviceLevelBridge, t0, t1; leg::Int = 1) =
    (d.wave = (; t0 = Float64(t0), t1 = Float64(t1), leg, rows = NTuple{9,Float64}[], crow = NTuple{9,Float64}[]); d)
