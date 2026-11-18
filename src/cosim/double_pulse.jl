# The fine-step rule and the virtual double-pulse test (COSIM_DESIGN.md 8.7, 4.5; experiment group B).

"""
    FineTracker(; h_f = 0.25e-9, tmin = 1.5e-6, tmax = 20e-6, block = 120, Ering = 1e-5, Coss = 3e-9, L = 8.4e-9)

Fine mode after a gate command until all of: tmin elapsed; |i_G| < 1 % of its peak since the command; over the last
block of fine steps (about one 35 MHz ringing period at h_f), the ringing energy estimated from the variations,
L (Delta i_L / 2)^2 / 2 + Coss ((Delta v_ds,u / 2)^2 + (Delta v_ds,l / 2)^2) / 2, below Ering; or tmax elapsed.
(Revised 2026-11-13 before the validation runs: the amplitude limits 2 A / 2 V / 0.1 V/ns of COSIM_DESIGN.md 8.7 never
close a fine interval in a loop with only the data sheet's resistances, where the ringing decays with 2 L / R of about
10 us; the energy limit 10 uJ is about 0.05 % of an edge's energy at 760 A. Stated in COSIM_DESIGN.md 8.10.)
"""
mutable struct FineTracker
    h_f::Float64
    tmin::Float64
    tmax::Float64
    block::Int
    active::Bool
    t_cmd::Float64
    igpk::Float64
    n::Int
    vu_lo::Float64; vu_hi::Float64
    vl_lo::Float64; vl_hi::Float64
    i_lo::Float64; i_hi::Float64
    dvmax::Float64
    Ering::Float64
    Coss::Float64
    L::Float64
    exits::Int                 # fine intervals closed by the criteria
    capped::Int                # closed by tmax
    defer::Bool                # the exit takes effect at the end of the window (`fine_window_end!`), not mid-window
    pending::Int               # with defer: 0 none, 1 the criteria passed, 2 tmax reached, in this window
end
# the block is 30 ns of physical time (about one ringing period) whatever h_f (review 103, finding 11)
# defer (the device-level bridge in the grid, review 105): the exit decided mid-window from the iterate's held values
# made the window map jump between Newton iterates (a two-cycle at the 1.5e-7 floor, one 30 ns block apart); deferred,
# the fine / coarse pattern of a window is fixed by the committed state and the gate schedule, and fine mode can only
# last longer (by at most the rest of the window)
FineTracker(; h_f = 0.25e-9, tmin = 1.5e-6, tmax = 20e-6, block = max(4, round(Int, 30e-9 / h_f)), Ering = 1e-5,
            Coss = 3e-9, L = 8.4e-9, defer::Bool = false) =
    FineTracker(h_f, tmin, tmax, block, false, -Inf, 0.0, 0, Inf, -Inf, Inf, -Inf, Inf, -Inf, 0.0, Ering, Coss, L, 0, 0,
                defer, 0)
Base.copy(f::FineTracker) = (g = FineTracker(); copyto!(g, f))
function Base.copyto!(a::FineTracker, b::FineTracker)
    for nm in fieldnames(FineTracker)
        setfield!(a, nm, getfield(b, nm))
    end
    return a
end

function _fine_reset_block!(f::FineTracker)
    f.n = 0
    f.vu_lo, f.vu_hi, f.vl_lo, f.vl_hi, f.i_lo, f.i_hi, f.dvmax = Inf, -Inf, Inf, -Inf, Inf, -Inf, 0.0
end

"""A gate command at t: fine mode (re)starts."""
function fine_command!(f::FineTracker, t)
    f.active = true; f.t_cmd = t; f.igpk = 0.0; f.pending = 0
    _fine_reset_block!(f)
    return f
end

"""After a fine step ending at t with the new state s, ledger l and the previous v_ds values: update, maybe exit."""
function fine_update!(f::FineTracker, t, h, s::SiCLegState, l::SiCStepLedger, vdu0, vdl0)
    (f.active && f.pending == 0) || return f
    ig = max(abs(l.iGu), abs(l.iGl))
    f.igpk = max(f.igpk, ig)
    f.n += 1
    f.vu_lo = min(f.vu_lo, s.vds_u); f.vu_hi = max(f.vu_hi, s.vds_u)
    f.vl_lo = min(f.vl_lo, s.vds_l); f.vl_hi = max(f.vl_hi, s.vds_l)
    f.i_lo = min(f.i_lo, l.iLm); f.i_hi = max(f.i_hi, l.iLm)
    f.dvmax = max(f.dvmax, ig)                       # (the block's largest gate current)
    el = t - f.t_cmd
    if el >= f.tmax
        f.defer ? (f.pending = 2) : (f.active = false; f.capped += 1)
        return f
    end
    if f.n >= f.block
        Er = f.L * ((f.i_hi - f.i_lo) / 2)^2 / 2 + f.Coss * (((f.vu_hi - f.vu_lo) / 2)^2 + ((f.vl_hi - f.vl_lo) / 2)^2) / 2
        quiet = el >= f.tmin && f.dvmax < 0.01 * f.igpk && Er < f.Ering
        if quiet
            f.defer ? (f.pending = 1) : (f.active = false; f.exits += 1)
        else
            _fine_reset_block!(f)
        end
    end
    return f
end

"""With defer: a pending exit takes effect at the end of the window (fine mode ends there)."""
function fine_window_end!(f::FineTracker)
    f.pending == 0 && return f
    f.pending == 1 ? (f.exits += 1) : (f.capped += 1)
    f.active = false; f.pending = 0
    return f
end

"""
    double_pulse(dev, RG; Vdd = 600, I = 760, Lload = 13.7e-6, Lsigma = 8.4e-9, Ru = 106.5e-6, Rl = 126.3e-6,
                 h_f = 0.25e-9, h_c = 5e-9, toff = 5e-6, ttail = 3e-6, Lint = 2.45e-9, Ering = 1e-5) -> NamedTuple

The virtual double-pulse test at the data sheet's conditions: the lower position is the device under test, the upper
position's gate at -4 V (its body diode freewheels), the load inductor between DC+ and the pole. The first pulse ramps
the load current to I, the turn-off (E_off), toff of freewheeling, the turn-on (E_on).
- E_on, E_off by the data sheet's definitions (Figs. 30-31) at the die boundary (the die's v_ds, the drain-terminal
  current), the threshold crossings interpolated linearly inside their steps (review 103, finding 13); and at the
  terminal boundary (the pole effort + Lint i di with the currents at the crossings). An edge whose window does not
  close returns NaN.
- Timings by Figs. 30-31 (with the ideal step driver, the terminal V_GS jumps at the command: a declared limit).
- Ledgers: the leg's (exact) and the whole harness's: the load inductor is advanced explicitly with each step's
  discrete pole effort, which adds L_load (Delta i)^2 / 2 per step outside the leg (review 103, finding 14): reported.
"""
function double_pulse(dev::SiCDevice, RG::Real; Vdd = 600.0, I = 760.0, Lload = 13.7e-6, Lsigma = 8.4e-9,
                      Ru = 106.5e-6, Rl = 126.3e-6, h_f = 0.25e-9, h_c = 5e-9, toff = 5e-6, ttail = 3e-6,
                      Lint = 2.45e-9, Von = 15.0, Voff = -4.0, Ering = 1e-5, t_rise = 0.0, record::Bool = false)
    leg = SiCLeg(dev, Float64(RG), Lsigma, Ru, Rl)
    sol = SiCLegSolver(leg)
    # both gates off, no current; the load inductor holds the pole at DC+: the lower position blocks Vdd
    s = SiCLegState(Voff, 0.0, Voff, Float64(Vdd), 0.0, 0.0, Float64(Vdd))
    il = 0.0                                                              # load current (A), DC+ -> pole
    ft = FineTracker(; h_f, L = Lsigma, Ering)
    t = 0.0
    Vl = Voff
    acc = zeros(7)                   # leg: dH, Edc, Eac, Eg, D, |defect| sum, gross
    harness = zeros(3)               # load inductor: Delta(L i^2 / 2), source work into it, the explicit-update defect
    rec = record ? NamedTuple[] : nothing
    phase = :pulse1
    t_cmd_off = NaN; t_cmd_on = NaN
    # an edge integral: (state 0 waiting / 1 open / 2 closed, die energy, terminal energy, i at start, i at stop)
    eoff = [0.0, 0.0, 0.0, 0.0, 0.0]; eon = [0.0, 0.0, 0.0, 0.0, 0.0]
    Ioff = NaN; Ion = NaN
    vgs_term(l) = Vstep_last[] - (RG - dev.RGint) * l.iGl
    Vstep_last = Ref(Voff)
    tim = Dict{Symbol, Float64}()
    dvdt = Dict(:off => 0.0, :on => 0.0); didt = Dict(:off => 0.0, :on => 0.0)
    win = Tuple{Float64, Float64, Float64}[]   # (t, v_ds, i_S) for the 5 ns sliding slopes
    function slopes!(key, tt, v, i)
        push!(win, (tt, v, i))
        while win[end][1] - win[1][1] > 5e-9
            popfirst!(win)
        end
        if win[end][1] - win[1][1] >= 4.5e-9           # (the span after popping: 4.75-5 ns at h_f = 0.25 ns)
            dt = win[end][1] - win[1][1]
            dvdt[key] = max(dvdt[key], abs(win[end][2] - win[1][2]) / dt)
            didt[key] = max(didt[key], abs(win[end][3] - win[1][3]) / dt)
        end
    end
    # one step of an edge integral: `xs` the start variable (crossing upwards through `a`), `xe` the end variable
    # (crossing downwards through `b`), previous and present values; the crossing step's fraction by linear interpolation
    function edge!(e, xs0, xs1, a, xe0, xe1, b, h, v, i, pp, i0, i1)
        if e[1] == 0.0 && xs1 >= a && xs0 < a
            f = (xs1 - a) / (xs1 - xs0)                 # the part of the step after the start crossing
            e[1] = 1.0; e[2] = f * h * v * i; e[3] = f * h * pp * i; e[4] = i0 + (i1 - i0) * (1 - f)
            if xe1 <= b && xe0 > b                      # (both crossings in one step: not expected at these steps)
                e[1] = 2.0; e[5] = i1
            end
        elseif e[1] == 1.0
            if xe1 <= b && xe0 > b
                g = (xe0 - b) / (xe0 - xe1)             # the part of the step before the end crossing
                e[2] += g * h * v * i; e[3] += g * h * pp * i; e[5] = i0 + (i1 - i0) * g; e[1] = 2.0
            else
                e[2] += h * v * i; e[3] += h * pp * i
            end
        end
        return nothing
    end
    vds_prev = Float64(Vdd); iS_prev = 0.0
    Vl = Von; fine_command!(ft, 0.0)
    while true
        tnext = phase === :pulse1 ? Inf : phase === :free ? t_cmd_off + toff : phase === :pulse2 ? t_cmd_on + ttail : Inf
        h = ft.active ? h_f : h_c
        t + h > tnext && (h = tnext - t)
        vdu0, vdl0 = s.vds_u, s.vds_l
        # (a diagnostic, not the validated model: a driver ramp of t_rise from the last command, at the step's midpoint)
        tc = phase === :pulse2 ? t_cmd_on : (phase === :off || phase === :free) ? t_cmd_off : 0.0
        Vstep = (t_rise > 0 && t + h / 2 - tc < t_rise) ? (Vl == Von ? Voff : Von) + (Vl - (Vl == Von ? Voff : Von)) * (t + h / 2 - tc) / t_rise : Vl
        ok, l = leg_step!(sol, s, h, Vdd, -il, Voff, Vstep)
        ok || error("double_pulse: the leg's Newton failed at t = $t (h = $h)")
        Vstep_last[] = Vstep
        t += h
        acc[1] += l.dH; acc[2] += l.E_dc; acc[3] += l.E_ac; acc[4] += l.E_gate; acc[5] += dissipation(l)
        acc[6] += abs(defect(l)); acc[7] += abs(l.E_dc) + abs(l.E_ac) + abs(l.E_gate)
        il1 = il + h * (Vdd - l.phip) / Lload
        harness[1] += Lload * (il1^2 - il^2) / 2
        harness[2] += h * (Vdd - l.phip) * il           # the work into the inductor booked with the held current
        harness[3] += Lload * (il1 - il)^2 / 2          # their difference: the explicit update's defect
        il = il1
        fine_update!(ft, t, h, s, l, vdu0, vdl0)
        vds = l.phiDl; iS = l.idl
        record && push!(rec, (; t, vds, iS, vgs = vgs_term(l), il, h))
        if phase === :pulse1
            if il >= I
                Ioff = il; Vl = Voff; t_cmd_off = t; phase = :off; fine_command!(ft, t)
                tim[:off_cmd] = t
            end
        elseif phase === :off || phase === :free
            # E_off: from v_ds rising through 10 % Vdd to i_S falling through 10 % Ioff
            edge!(eoff, vds_prev, vds, 0.1Vdd, iS_prev, iS, 0.1Ioff, h, vds, iS, l.phip, iS_prev, iS)
            slopes!(:off, t, vds, iS)
            vg = vgs_term(l)
            haskey(tim, :vg90_off) || (vg <= Voff + 0.9 * (Von - Voff) && (tim[:vg90_off] = t))
            haskey(tim, :i90_off) || (iS <= 0.9Ioff && (tim[:i90_off] = t))
            haskey(tim, :i10_off) || (iS <= 0.1Ioff && (tim[:i10_off] = t))
            phase === :off && (phase = :free)
            if t >= t_cmd_off + toff - 1e-15
                Ion = il; Vl = Von; t_cmd_on = t; phase = :pulse2; fine_command!(ft, t); empty!(win)
                tim[:on_cmd] = t
            end
        elseif phase === :pulse2
            # E_on: from i_S rising through 10 % Ion to v_ds falling through 10 % Vdd
            edge!(eon, iS_prev, iS, 0.1Ion, vds_prev, vds, 0.1Vdd, h, vds, iS, l.phip, iS_prev, iS)
            slopes!(:on, t, vds, iS)
            vg = vgs_term(l)
            haskey(tim, :vg10_on) || (vg >= Voff + 0.1 * (Von - Voff) && (tim[:vg10_on] = t))
            haskey(tim, :i10_on) || (iS >= 0.1Ion && (tim[:i10_on] = t))
            haskey(tim, :i90_on) || (iS >= 0.9Ion && (tim[:i90_on] = t))
            t >= t_cmd_on + ttail - 1e-15 && break
        end
        vds_prev, iS_prev = vds, iS
    end
    closed(e) = e[1] == 2.0
    E_off_die = closed(eoff) ? eoff[2] : NaN; E_on_die = closed(eon) ? eon[2] : NaN
    E_off_term = closed(eoff) ? eoff[3] + Lint * (eoff[5]^2 - eoff[4]^2) / 2 : NaN
    E_on_term = closed(eon) ? eon[3] + Lint * (eon[5]^2 - eon[4]^2) / 2 : NaN
    timing = (td_on = get(tim, :i10_on, NaN) - get(tim, :vg10_on, NaN), tr = get(tim, :i90_on, NaN) - get(tim, :i10_on, NaN),
              td_off = get(tim, :i90_off, NaN) - get(tim, :vg90_off, NaN), tf = get(tim, :i10_off, NaN) - get(tim, :i90_off, NaN))
    closure = (dH = acc[1], E_dc = acc[2], E_ac = acc[3], E_gate = acc[4], D = acc[5],
               residual = acc[1] - (acc[2] - acc[3] + acc[4] - acc[5]), sum_abs_defect = acc[6], gross = acc[7],
               load_dH = harness[1], load_work = harness[2], load_defect = harness[3])
    return (; I_off = Ioff, I_on = Ion, E_off_die, E_on_die, E_off_term, E_on_term, timing, dvdt_on = dvdt[:on],
            dvdt_off = dvdt[:off], didt_on = didt[:on], didt_off = didt[:off], closure,
            fine = (exits = ft.exits, capped = ft.capped), rec)
end
