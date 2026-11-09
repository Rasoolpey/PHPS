# The grid-forming converter's controller on the EMT model (docs/active/PH_CONVERTER_MODEL.md,
# sections 2 to 4; PH_SIMULATOR_TODO.md step S8.2c): the EMT path of PowerFactory's gfm_vsm_pf DLL
# (v1.4, VI-only build), ported from its source, as a signal law. Its output is the EMF u_c of the
# converter's source row (`system.jl`, file header); it holds no energy and is not a component of
# the energy account.
#
# The DLL's laws, in the simulator's frame turning at rated speed (record, section 2: every
# operation is a rotation, a product Re(u conj(i)) or a magnitude, so only the angle's rate changes):
#   p         = Re(v_k conj(i_c))                        (terminal voltage, measured current; unfiltered)
#   Ta omega' = (p_set - p) - Dp (omega_f - 1)
#   omega_f'  = omega_c (omega - omega_f)
#   theta'    = (omega - 1) omega_n
#   u_mag'    = (u_set - u_mag) / Tv
#   E         = u_mag e^{j theta}
#   |i_p| |r_eff + j(Zs + x_eff)| = |E - v_k|,  r_eff = r_vi + kpr max(0, |i_p| - i_lim), x_eff alike
#   u_c       = E - (r_eff + j x_eff) i_p,  i_p = (E - v_k) / (r_eff + j(Zs + x_eff))
# The divider reads the terminal voltage, which depends on u_c: v_k = alpha u_c + w (`system.jl`).
# Eliminated first (consultation C1, findings 5 and 6):
#   ((1 - alpha)(r_eff + j x_eff) + j Zs) i_p = (1 - alpha) E - w,
# a scalar equation for |i_p| of `vi_bisection`'s form with the virtual-impedance parameters times
# (1 - alpha) > 0, monotone, so one root; then i_p and u_c. The map is continuous in the state, so
# no decision is recorded (finding 7): `|i_p| > i_lim` is taken at the root itself.
#
# Identical units on one bus are one converter of their combined rating (system base): Zs, r_vi,
# x_vi divided by n, kpr and kpx by n^2, i_lim, p_set, Ta and Dp multiplied by n.
#
# Sampled execution (`Ts > 0`; record section 10). PowerFactory calls the DLL at its declared
# `FixedStepBaseSampleTime` (50 us, gfm_vsm_pf.cpp line 546), not at the network's step: at each call
# it reads the terminal voltage and current, computes the output from the states as they are
# (`Model_Outputs`), then advances the states by explicit Euler (`Model_Update`); the output (alpha-beta)
# is held until the next call. Here: the four states are held between calls, and two more states hold
# the output, u_c' = -j omega_b u_c (a constant alpha-beta value in the frame turning at omega_b); at a
# call `vsm_sample!` sets u_c from the DLL's divider with the measured terminal voltage (no elimination:
# the voltage is that of the network with the held output) and makes the Euler step, with the angle in
# the turning frame theta <- theta + Ts (omega omega_n - omega_b).

"""
    PHConverterDC

The physical DC side and bridge of a converter (record section 12; S8.3), system base: the battery `Eb` behind `Rb`, the
DC-link capacitor `c` (`(c/omega_b) v_dc' = i_b - G v_dc - Re(m conj(i))`, `i` the injected current), the averaged lossless
bridge `u_x = m v_dc` with the radial limit `|m| <= m_max` (SVPWM `2/sqrt(3)` on the DC base `2 sqrt(2/3) V_LL`), and how
the modulation follows the controller: `:held` (sampled with the controller, `m = clip(u_c / v_dc)` at a call, held in
the alpha-beta frame) or `:continuous` (the command `u_c` held, `m = clip(u_c / v_dc)` at every instant: while unclipped
the AC side is the ideal source's). `Sr` (the rating, system per unit), `H` (s), `v_dc0` are the selected data
it was made from.

The battery may carry RC branches in series behind `Rb` (record section 32, part A: the measured BESS of Sossan et al. 2016,
`E` behind `R_s` and three RC branches): `R` and `cb` (each branch's resistance and its capacitance in the DC link's
convention, `(cb_k/omega_b) v_k' = i_b - v_k/R_k`), so `i_b = (Eb - sum v_k - v_dc)/Rb`. With none it is `Eb` behind `Rb`.
"""
struct PHConverterDC
    c::Float64
    Rb::Float64
    Eb::Float64
    G::Float64
    m_max::Float64
    mode::Symbol
    Sr::Float64
    H::Float64
    v_dc0::Float64
    R::Vector{Float64}           # the battery's RC branches (none: Eb behind Rb)
    cb::Vector{Float64}
    source::String               # where the battery's data come from
    sw::Any                      # the switching bridge (bridge.jl, upgrade 3; `SwitchingBridge`), or nothing: averaged
end
PHConverterDC(c, Rb, Eb, G, m_max, mode, Sr, H, v_dc0) =
    PHConverterDC(c, Rb, Eb, G, m_max, mode, Sr, H, v_dc0, Float64[], Float64[], "selected: Eb behind Rb (section 12)", nothing)
PHConverterDC(c, Rb, Eb, G, m_max, mode, Sr, H, v_dc0, R, cb, source) =
    PHConverterDC(c, Rb, Eb, G, m_max, mode, Sr, H, v_dc0, R, cb, source, nothing)

# the switching bridge in a run (bridge.jl): its step averages at the injected current ic, or nothing (averaged bridge)
@inline _has_sw(g) = hasproperty(g, :dc) && g.dc !== nothing && g.dc.sw !== nothing
@inline _sw_on(g) = _has_sw(g) && g.dc.sw.active
_sw_terms(g, xc, ic) = bridge_average(g.dc.sw, ic, _vdc_checked(xc[_dcidx(g)]))
nbranches(q::PHConverterDC) = length(q.R)

"""
    PHVsm

The controller of one converter (all its identical units, system base): `Ta`, `Dp`, `omega_c`,
`omega_n`, `Tv`, the series reactance `Zs`, the virtual impedance's `i_lim`, `r_vi`, `x_vi`, `kpr`,
`kpx`, and the set-points `p_set`, `u_set`. States `theta, omega, omega_f, u_mag`; sampled (`Ts > 0`) also
the held output (the command, or with a physical DC side `dc` in mode `:held` the modulation) and with `dc` the DC-link
voltage.
"""
struct PHVsm
    name::String
    Ta::Float64
    Dp::Float64
    omega_c::Float64
    omega_n::Float64
    Tv::Float64
    Zs::Float64
    i_lim::Float64
    r_vi::Float64
    x_vi::Float64
    kpr::Float64
    kpx::Float64
    p_set::Float64
    u_set::Float64
    Ts::Float64                  # 0: continuous; > 0: the DLL's sampled execution at this period
    dc::Union{Nothing,PHConverterDC}
    u_max::Float64               # the output-voltage limit of the controller (a control law; Inf: none), section 14
end

const _PHVSM_STATES = ["theta", "omega", "omega_f", "u_mag"]
nstates(g::PHVsm) = g.Ts > 0 ? (g.dc === nothing ? 6 : 7 + nbranches(g.dc)) : 4
state_names(g::PHVsm) = g.Ts == 0 ? _PHVSM_STATES :
    g.dc === nothing ? vcat(_PHVSM_STATES, ["uc_Re", "uc_Im"]) :
    vcat(_PHVSM_STATES, g.dc.mode === :held ? ["m_Re", "m_Im"] : ["uc_Re", "uc_Im"], _dc_state_names(g.dc))
_dc_state_names(q::PHConverterDC) = vcat(["v_dc"], ["v_b$k" for k in 1:nbranches(q)])
name(g::PHVsm) = g.name

function PHVsm(cv::PHConverter; p_set::Real, u_set::Real, Ts::Real = 0.0, dc = nothing, u_max::Real = Inf)
    (isfinite(Ts) && Ts >= 0) || throw(ArgumentError("PHVsm: Ts = $Ts"))
    u_max > 0 || throw(ArgumentError("PHVsm: u_max = $u_max"))
    (isinf(u_max) || Ts > 0) ||
        throw(ArgumentError("PHVsm: an output-voltage limit needs the sampled controller (Ts > 0): with the continuous " *
                            "controller the limit sits in its algebraic loop"))
    dc === nothing || Ts > 0 ||
        throw(ArgumentError("PHVsm: a physical DC side needs the sampled controller (Ts > 0); with the continuous " *
                            "controller the clip sits in its algebraic loop (record section 12, P1 finding 2)"))
    if dc !== nothing
        (dc.mode in (:held, :continuous) && all(isfinite, (dc.c, dc.Rb, dc.G, dc.m_max, dc.Eb)) &&
         dc.c > 0 && dc.Rb > 0 && dc.G >= 0 && dc.m_max > 0 && dc.Eb > 0) ||
            throw(ArgumentError("PHVsm: $(cv.name): DC data outside their domain: $dc"))
    end
    p, n = cv.comp.p, cv.units
    for (k, v) in (("Ta", p.Ta), ("omega_c", p.omega_c), ("omega_n", p.omega_n), ("Tv", p.Tv),
                   ("Zseries", p.Zseries), ("i_lim", p.i_lim))
        (isfinite(v) && v > 0) || throw(ArgumentError("PHVsm: $(cv.name) needs $k > 0 (it is $v)"))
    end
    for (k, v) in (("Dp", p.Dp), ("r_vi", p.r_vi), ("x_vi", p.x_vi), ("kpr", p.kpr), ("kpx", p.kpx))
        (isfinite(v) && v >= 0) || throw(ArgumentError("PHVsm: $(cv.name) needs $k >= 0 (it is $v)"))
    end
    (p.f_set == 1 && p.K_field == 0 && p.D_direct == 0 && p.n_vi <= 1 && p.pf_frame <= 0.5) ||
        throw(ArgumentError("PHVsm: $(cv.name) uses a feature the DLL's VI-only build does not have " *
                            "(f_set = 1, K_field = 0, D_direct = 0, n_vi = 1, pf_frame = 0 needed)"))
    p.Zseries / n ≈ cv.Lc || error("PHVsm: $(cv.name)'s reactance differs from its branch's")
    return PHVsm(cv.name, p.Ta * n, p.Dp * n, p.omega_c, p.omega_n, p.Tv, p.Zseries / n, p.i_lim * n,
                 p.r_vi / n, p.x_vi / n, p.kpr / n^2, p.kpx / n^2, Float64(p_set), Float64(u_set), Float64(Ts), dc,
                 Float64(u_max))
end

# a magnitude whose derivative is taken as zero where it is zero (finding 11: there the result does
# not depend on the magnitude's derivative, the virtual impedance being off below i_lim > 0)
@inline _safe_abs(z) = (m2 = abs2(z); m2 == 0 ? zero(m2) : sqrt(m2))

"""
    vsm_divider(g, E, alpha, w) -> (; uc, ip, on)

The predictive virtual-impedance divider with the terminal voltage `v_k = alpha u_c + w`
eliminated (file header): the EMF `uc`, the predicted current `ip` and whether the virtual
impedance is engaged (`|ip| > i_lim`).
"""
function vsm_divider(g, E, alpha, w)              # g: PHVsm or PHVoc (the same divider, section 28)
    b = 1 - alpha
    D = b * E - w
    i = vi_bisection(_safe_abs(D), g.Zs, g.i_lim, b * g.r_vi, b * g.x_vi, b * g.kpx, b * g.kpr)
    on = i > g.i_lim
    ov = on ? i - g.i_lim : zero(i)
    ze = complex(g.r_vi + g.kpr * ov, g.x_vi + g.kpx * ov)
    ip = D / (b * ze + im * g.Zs)
    return (; uc = E - ze * ip, ip, on)
end

# the converter's EMF E = u_mag e^{j theta} from its states
@inline _vsm_emf(xc) = xc[4] * cis(xc[1])

"""
    vsm_rhs!(dx, g, xc, vk, ic)

The controller's rows at its states `xc`, terminal voltage `vk` and converter current `ic`
(injected into the network).
"""
function vsm_rhs!(dx, g::PHVsm, xc, vk, ic)
    p = real(vk * conj(ic))
    dx[1] = (xc[2] - 1) * g.omega_n
    dx[2] = ((g.p_set - p) - g.Dp * (xc[3] - 1)) / g.Ta
    dx[3] = g.omega_c * (xc[2] - xc[3])
    dx[4] = (g.u_set - xc[4]) / g.Tv
    return dx
end

"""
    phs_converter_signals(cs, x, u) -> (; omega, theta, p, q, i, v, uc, ip, on, icr, ici, ...)

Per converter at the state `x` (inputs `u` for the fault conductances), system base, all units:
speed, angle (rad, simulator frame), active and reactive power `v_k conj(i_c)`, the injected current `icr + j ici`
(network frame), current and
terminal-voltage magnitudes, the EMF's magnitude, the predicted current's magnitude and whether
the virtual impedance is engaged. For a sampled converter (`Ts > 0`) the EMF is the held output, and the
predicted current and engagement are the divider at the present state, not the call that made the held
output; its `theta` is the angle prepared for the next call (C3, findings 6, 10). With a physical DC side also
`vdc`, the modulation's magnitude `m` and the battery current `ib` (NaN otherwise).
"""
function phs_converter_signals(cs, x::AbstractVector{Float64}, u)
    s = cs.s
    up = phs_control_inputs(cs, x, u.gf)
    f = phs_flows(s, x, up)
    n = length(cs.conv)
    out = (; omega = zeros(n), theta = zeros(n), p = zeros(n), q = zeros(n), i = zeros(n), v = zeros(n),
           uc = zeros(n), ip = zeros(n), on = zeros(n), vdc = fill(NaN, n), m = fill(NaN, n), ib = fill(NaN, n),
           umag = fill(NaN, n), icr = zeros(n), ici = zeros(n), mreq = fill(NaN, n))
    for (k, cv) in enumerate(s.conv)
        d = _vsm_at(cs, x, f, k)
        vk, ic = cv.core == 0 ? cv.alpha * d.uc + d.w : _conv_vk(s, cv, x, f), _conv_current(f, cv)
        S = vk * conj(ic)
        out.omega[k], out.theta[k] = cs.conv[k] isa PHEmfTrace ? (NaN, angle(d.uc)) :
                                     cs.conv[k] isa PHDroop ? (droop_omega(cs.conv[k], d.xc), d.xc[1]) :
                                     cs.conv[k] isa PHVoc ? (voc_omega(cs.conv[k], d.xc, vk, ic), angle(complex(d.xc[1], d.xc[2]))) :
                                     cs.conv[k] isa PHGfl ? (gfl_omega(cs.conv[k], d.xc), d.xc[1]) :
                                     (d.xc[2], d.xc[1])
        out.p[k], out.q[k], out.i[k], out.v[k] = real(S), imag(S), abs(ic), abs(vk)
        out.icr[k], out.ici[k] = real(ic), imag(ic)
        out.uc[k], out.ip[k], out.on[k] = abs(d.uc), abs(d.ip), d.on
        cs.conv[k] isa Union{PHEmfTrace,PHGfl} ||
            (out.umag[k] = cs.conv[k] isa PHVoc ? abs(complex(d.xc[1], d.xc[2])) : d.xc[4])   # the VOC: |v|
        g = cs.conv[k]
        if g isa Union{PHVsm,PHVoc,PHGfl,PHDroop} && g.dc !== nothing
            out.vdc[k], out.m[k] = d.xc[_dcidx(g)], abs(_modulation(g, d.xc, ic))
            out.ib[k] = _battery_current(g, d.xc)
            # the modulation the GFL's static generator asks for, before the bridge's limit (record section 43, G8 items 2, 5)
            g isa PHGfl && (out.mreq[k] = abs(gfl_emf(g, d.xc, ic)) / _vdc_checked(d.xc[_dcidx(g)]))
            # the droop: the command its next call will make, over the DC voltage now (record section 45, G10 item 5)
            g isa PHDroop && (out.mreq[k] = abs(droop_output(g, d.xc).uc) / _vdc_checked(d.xc[_dcidx(g)]))
        end
    end
    return out
end

"""
    PHEmfTrace

A diagnostic in place of a converter's controller: the EMF prescribed as a recorded trace `uc` at the
times `t` (network frame, system base), linearly interpolated; one state, a clock (`clock' = 1`). Used to
drive the network with another simulator's converter commands, which separates the network and the
machines from the controller (consultation C2, next checks).
"""
struct PHEmfTrace
    name::String
    t::Vector{Float64}
    uc::Vector{ComplexF64}
end
nstates(::PHEmfTrace) = 1
state_names(::PHEmfTrace) = ["clock"]
name(g::PHEmfTrace) = g.name

function _trace_at(g::PHEmfTrace, tau)
    tv = ForwardDiff.value(tau)
    k = clamp(searchsortedlast(g.t, tv), 1, length(g.t) - 1)
    a = (tau - g.t[k]) / (g.t[k+1] - g.t[k])
    return (1 - a) * g.uc[k] + a * g.uc[k+1]
end

"""
    vsm_sample!(xc, g, vk, ic, omega_b)

One call of the DLL (sampled execution): with the controller states `xc` (six, `g.Ts > 0`), the measured
terminal voltage `vk` and converter current `ic`, the new held output (the divider at `vk`, from the states
as they are) and the explicit Euler step of the states over `g.Ts`.
"""
function vsm_sample!(xc, g::PHVsm, vk, ic, omega_b)
    E = _vsm_emf(xc)
    uc = vsm_divider(g, E, 0.0, vk).uc            # alpha = 0, w = v_k: the divider at the measured voltage
    isinf(g.u_max) || (uc = _radial_clip(uc, g.u_max))   # the output-voltage limit (gfm_vsm_pf_vlim, section 14)
    d = vsm_rhs!(zeros(promote_type(eltype(xc), typeof(real(vk))), 4), g, xc, vk, ic)
    xc[1] += g.Ts * (xc[2] * g.omega_n - omega_b)  # the alpha-beta angle's step, less the frame's
    xc[2] += g.Ts * d[2]
    xc[3] += g.Ts * d[3]
    xc[4] += g.Ts * d[4]
    # the held quantity: the command, or the modulation from the DC voltage sampled at this call
    h = _held_from(g, uc, xc)
    xc[5], xc[6] = real(h), imag(h)
    return xc
end

# the held quantity from a call's command: the command, or (physical DC side, :held) the modulation from the DC voltage sampled
# at the call, clipped radially
_held_from(g, uc, xc) = (g.dc !== nothing && g.dc.mode === :held) ? _radial_clip(uc / _vdc_checked(xc[_dcidx(g)]), g.dc.m_max) : uc

# the radial limit |z| <= U (direction kept)
@inline _radial_clip(z, U) = abs2(z) <= U^2 ? z : z * (U / sqrt(abs2(z)))

# the DC voltage, inside the model's domain v_dc > 0 (a Newton trial outside it stops the run: P2 finding 1)
@inline function _vdc_checked(v)
    ForwardDiff.value(v) > 0 || throw(DomainError(ForwardDiff.value(v), "the DC-link voltage left v_dc > 0"))
    return v
end

# where a sampled controller keeps its held quantity and its DC voltage (followed by the battery's branch voltages)
_held(g::PHVsm, xc) = complex(xc[5], xc[6])
_dcidx(g::PHVsm) = 7

# the voltage the bridge (or the ideal source) applies, from a sampled converter's states
function _bridge_voltage(g, xc)
    h = _held(g, xc)
    g.dc === nothing && return h
    v = _vdc_checked(xc[_dcidx(g)])
    g.dc.mode === :held && return h * v
    return _radial_clip(h, g.dc.m_max * v)
end

# the modulation in force (physical DC side); with the injected current `ic` for a controller whose EMF depends on it (the GFL)
_modulation(g, xc, ic) = _modulation(g, xc)
_modulation(g, xc) = g.dc.mode === :held ? _held(g, xc) : _bridge_voltage(g, xc) / _vdc_checked(xc[_dcidx(g)])

# the battery's current: Eb behind Rb and its RC branches, into the DC link
function _battery_current(g, xc)
    q, j = g.dc, _dcidx(g)
    return (q.Eb - sum(xc[j+k] for k in 1:nbranches(q); init = zero(eltype(xc))) - xc[j]) / q.Rb
end

"""
    converter_dc_powers(g, xc, ic, omega_b) -> (; i_b, P_x, energy, loss, port, tport)

The physical DC side's account at its states `xc` and injected AC current `ic` (system base): the battery current,
the bridge power `P_x = v_dc Re(m conj(ic)) = Re(u_x conj(ic))`, the stored energy `(c v_dc^2 + sum cb_k v_k^2) / (2 omega_b)`
(the DC link and the battery's branches), the losses `Rb i_b^2 + sum v_k^2 / R_k + G v_dc^2` and the net port `Eb i_b - P_x`.
"""
function converter_dc_powers(g, xc, ic, omega_b)
    q, j = g.dc, _dcidx(g)
    v = xc[j]
    i_b = _battery_current(g, xc)
    nb = nbranches(q)
    energy = _dc_energy(g, xc, omega_b)
    loss = q.Rb * i_b^2 + sum(xc[j+k]^2 / q.R[k] for k in 1:nb; init = zero(v)) + q.G * v^2
    if _sw_on(g)                                 # switching bridge: P_x the AC terminal power; its conduction and
        r = _sw_terms(g, xc, ic)                 # switching losses are this component's
        P_x = real((v * r.m + r.udrop) * conj(ic))
        loss += r.pcond + r.psw
    else
        P_x = v * real(_modulation(g, xc, ic) * conj(ic))
    end
    return (; i_b, P_x, energy, loss, port = q.Eb * i_b - P_x, tport = abs(q.Eb * i_b) + abs(P_x))
end

# the DC side's rows: the link (c/omega_b) v' = i_b - G v - Re(m conj(ic)), the branches (cb_k/omega_b) v_k' = i_b - v_k/R_k
function _dc_rows!(dx, g, xc, ic, omega_b)
    q, j = g.dc, _dcidx(g)
    i_b = _battery_current(g, xc)
    if _sw_on(g)                                 # switching bridge: the step's averaged modulation and the switching sink
        r = _sw_terms(g, xc, ic)
        dx[j] = omega_b * (i_b - q.G * xc[j] - real(r.m * conj(ic)) - r.psw / xc[j]) / q.c
    else
        dx[j] = omega_b * (i_b - q.G * xc[j] - real(_modulation(g, xc, ic) * conj(ic))) / q.c
    end
    for k in 1:nbranches(q)
        dx[j+k] = omega_b * (i_b - xc[j+k] / q.R[k]) / q.cb[k]
    end
    return dx
end

# the DC side's stored energy: the link and the battery's branches
_dc_energy(g, xc, omega_b) = (g.dc.c * xc[_dcidx(g)]^2 +
    sum(g.dc.cb[k] * xc[_dcidx(g)+k]^2 for k in 1:nbranches(g.dc); init = zero(eltype(xc)))) / (2 * omega_b)

# the DC side's states at the operating point: the link at v_dc0, each branch carrying the battery's current there
_dc_initial(q::PHConverterDC) = vcat(q.v_dc0, q.R .* ((q.Eb - q.v_dc0) / (q.Rb + sum(q.R; init = 0.0))))

# the DC side's energy rate along dx
_dc_energy_rate(g, xc, dxc, omega_b) =
    (g.dc.c * xc[_dcidx(g)] * dxc[_dcidx(g)] +
     sum(g.dc.cb[k] * xc[_dcidx(g)+k] * dxc[_dcidx(g)+k] for k in 1:nbranches(g.dc); init = zero(eltype(xc)))) / omega_b

# ---------------------------------------------------------------------------------------------------------------
# The grid-forming droop controller (record section 16; S8.4): the EMT path of PowerFactory's DLL variant
# gfm_droop_pf_emt (gfm_droop_pf with its measurement lags in the controller's rotating frame), with r_vi = x_vi = 0 and
# the adaptive features off. In the simulator's frame (theta the angle in it):
#   p + j q = v_k conj(i);  omega = 1 - mp (p_f - p_set);  u_ref = u_set - mq (q_f - q_set)
#   v_meas = v_m e^{j theta}, i_meas = i_m e^{j theta} (filter states in the controller's frame)
#   du = (kpr + j kpx)(|i_m| - i_lim) i_meas above i_lim;  u_con = u_mag e^{j theta} - du
#   Stage 1: |u_con - v_meas| <= Zs i_con_lim;  Stage 2: above i_con_lim, u_con = v_meas + (u_con - v_meas) i_con_lim/|i_m|
#   p_f' = (p - p_f)/Tf_p, q_f' = (q - q_f)/Tf_q, u_mag' = (u_ref - u_mag)/Tv,
#   v_m' = (v_k e^{-j theta} - v_m)/T_V, i_m' = (i e^{-j theta} - i_m)/T_I, theta' = omega omega_n - omega_b
# The output depends on the states only (no algebraic loop). Identical units combined as for the VSM: Zs/n, i_lim n,
# i_con_lim n, kpr/n^2, kpx/n^2, mp/n, mq/n (the droop gains per unit of the combined rating), p_set, q_set n.
#
# DC rejection (`dc_reject`, sampled only; record section 25; the DLL variant gfm_droop_pf_dcr): the limiter (boost, Stage 1,
# Stage 2) reads the lagged voltage and current through a discrete filter in the controller's frame, one complex state each,
#   Y_k = c (Z_k - g V_k),  V_{k+1} = lam V_k + Z_k,  lam = e^{-(a + j wn) Ts}, g = e^{-j wn Ts} - lam,
#   c = (1 - lam)/(1 - e^{-j wn Ts}), a = wn,
# with gain 1 at 0 Hz in that frame and 0 at -wn (a DC in the phases): the EMF no longer copies the terminal voltage's DC.
# ---------------------------------------------------------------------------------------------------------------

"""
    PHDroop

The droop controller of one converter (all its identical units, system base). States `theta, p_f, q_f, u_mag, vm_d,
vm_q, im_d, im_q` (the filter states in the controller's frame), sampled (`Ts > 0`) also the held output `uc_Re, uc_Im`.
"""
struct PHDroop
    name::String
    mp::Float64
    mq::Float64
    Tf_p::Float64
    Tf_q::Float64
    omega_n::Float64
    Tv::Float64
    T_V::Float64
    T_I::Float64
    Zs::Float64
    i_lim::Float64
    i_con_lim::Float64
    kpr::Float64
    kpx::Float64
    p_set::Float64
    q_set::Float64
    u_set::Float64
    Ts::Float64
    dcr::Bool                    # DC rejection of the limiter's inputs (section 25)
    dlam::ComplexF64             # its lam, g, c (c renamed dcc: `dc` is the physical DC side, as for the VOC)
    dg::ComplexF64
    dcc::ComplexF64
    dcr_on::Symbol               # which inputs are filtered: :both (the variant gfm_droop_pf_dcr), or :voltage, :current,
                                 # :boost, :stage2 (ablations: consultation D4; the current for the boost or Stage 2 only)
    dc::Union{Nothing,PHConverterDC}   # the physical DC side (battery, link, bridge; record section 45), or the ideal source
end

const _PHDROOP_STATES = ["theta", "p_f", "q_f", "u_mag", "vm_d", "vm_q", "im_d", "im_q"]
nstates(g::PHDroop) = g.Ts > 0 ? (g.dcr ? 14 : 10) + (g.dc === nothing ? 0 : 1 + nbranches(g.dc)) : 8
function state_names(g::PHDroop)
    g.Ts == 0 && return _PHDROOP_STATES
    held = (g.dc !== nothing && g.dc.mode === :held) ? ["m_Re", "m_Im"] : ["uc_Re", "uc_Im"]
    v = g.dcr ? vcat(_PHDROOP_STATES, held, ["dcrV_d", "dcrV_q", "dcrI_d", "dcrI_q"]) : vcat(_PHDROOP_STATES, held)
    return g.dc === nothing ? v : vcat(v, _dc_state_names(g.dc))
end
# the held quantity (the command, or with a physical DC side in mode :held the modulation) and the DC voltage's index
_held(g::PHDroop, xc) = complex(xc[9], xc[10])
_dcidx(g::PHDroop) = g.dcr ? 15 : 11

# the DC-rejection filter's constants at the call period Ts (a = wn)
function _dcr_constants(wn, Ts)
    q = cis(-wn * Ts)
    lam = exp(-(wn + im * wn) * Ts)
    return lam, q - lam, (1 - lam) / (1 - q)
end
name(g::PHDroop) = g.name

function PHDroop(cv::PHConverter; p_set::Real, q_set::Real, u_set::Real, Ts::Real = 0.0, dc_reject::Bool = false,
                 dcr_on::Symbol = :both, dc = nothing)
    dcr_on in (:both, :voltage, :current, :boost, :stage2) || throw(ArgumentError("PHDroop: dcr_on = $dcr_on"))
    p, n = cv.comp.p, cv.units
    (isfinite(Ts) && Ts >= 0) || throw(ArgumentError("PHDroop: Ts = $Ts"))
    (!dc_reject || Ts > 0) || throw(ArgumentError("PHDroop: the DC rejection is defined for the sampled controller (Ts > 0)"))
    for (k, v) in (("Tf_p", p.Tf_p), ("Tf_q", p.Tf_q), ("omega_n", p.omega_n), ("Tv", p.Tv), ("Zseries", p.Zseries),
                   ("i_lim", p.i_lim), ("i_con_lim", p.i_con_lim), ("Tmeas", p.Tmeas), ("Tmeas_i", p.Tmeas_i))
        (isfinite(v) && v > 0) || throw(ArgumentError("PHDroop: $(cv.name) needs $k > 0 (it is $v)"))
    end
    for (k, v) in (("mp", p.mp), ("mq", p.mq), ("kpr", p.kpr), ("kpx", p.kpx))
        (isfinite(v) && v >= 0) || throw(ArgumentError("PHDroop: $(cv.name) needs $k >= 0 (it is $v)"))
    end
    (p.r_vi == 0 && p.x_vi == 0 && p.adapt_droop <= 0.5 && p.adapt_vi <= 0.5 && p.f_set == 1 && p.pf_frame <= 0.5) ||
        throw(ArgumentError("PHDroop: $(cv.name) uses a feature this port does not cover (r_vi = x_vi = 0, adaptive " *
                            "droop and virtual impedance off, f_set = 1, pf_frame = 0 needed)"))
    p.vi_mode > 0.5 ||
        throw(ArgumentError("PHDroop: $(cv.name) has vi_mode = $(p.vi_mode); only the paper form (vi_mode > 0.5) is ported " *
                            "(the legacy branch multiplies the boost by 2 i_meas - i_lag2: D2 finding 3)"))
    Ts < min(p.Tf_p, p.Tf_q, p.Tv, p.Tmeas, p.Tmeas_i) ||
        throw(ArgumentError("PHDroop: $(cv.name): the sample period must be below every filter time constant " *
                            "(the DLL's filter skips Euler otherwise: D2 finding 5)"))
    p.Zseries / n ≈ cv.Lc || error("PHDroop: $(cv.name)'s reactance differs from its branch's")
    if dc !== nothing                            # the physical DC side (record section 45): the sampled controller only
        Ts > 0 || throw(ArgumentError("PHDroop: $(cv.name): a physical DC side needs the sampled controller (Ts > 0)"))
        (dc.mode in (:held, :continuous) && all(isfinite, (dc.c, dc.Rb, dc.G, dc.m_max, dc.Eb)) && dc.c > 0 && dc.Rb > 0 &&
         dc.G >= 0 && dc.m_max > 0 && dc.Eb > 0 && length(dc.R) == length(dc.cb) && all(>(0), dc.R) && all(>(0), dc.cb)) ||
            throw(ArgumentError("PHDroop: $(cv.name): DC data outside their domain: $dc"))
    end
    lam, gg, cc = dc_reject ? _dcr_constants(p.omega_n, Ts) : (0.0im, 0.0im, 0.0im)
    return PHDroop(cv.name, p.mp / n, p.mq / n, p.Tf_p, p.Tf_q, p.omega_n, p.Tv, p.Tmeas, p.Tmeas_i, p.Zseries / n,
                   p.i_lim * n, p.i_con_lim * n, p.kpr / n^2, p.kpx / n^2, Float64(p_set), Float64(q_set),
                   Float64(u_set), Float64(Ts), dc_reject, lam, gg, cc, dcr_on, dc)
end

"""
    droop_output(g, xc) -> (; uc, vmeas, imeas, boost, stage1, stage2)

The droop controller's output (the DLL's `Model_Outputs`) from its states alone.
"""
function droop_output(g::PHDroop, xc)
    rot = cis(xc[1])
    vm_c, im_c = complex(xc[5], xc[6]), complex(xc[7], xc[8])
    im_s = im_c                                      # the current Stage 2 reads
    if g.dcr                                         # the limiter's inputs DC-rejected (section 25)
        g.dcr_on in (:both, :voltage) && (vm_c = g.dcc * (vm_c - g.dg * complex(xc[11], xc[12])))
        im_f = g.dcc * (im_c - g.dg * complex(xc[13], xc[14]))
        g.dcr_on in (:both, :current, :stage2) && (im_s = im_f)
        g.dcr_on in (:both, :current, :boost) && (im_c = im_f)
    end
    vmeas = vm_c * rot
    imeas = im_c * rot
    ia = _safe_abs(im_c)
    boost = ia > g.i_lim
    du = boost ? complex(g.kpr, g.kpx) * (ia - g.i_lim) * imeas : zero(imeas)
    u = xc[4] * rot - du
    d = u - vmeas
    dm, dmax = _safe_abs(d), g.Zs * g.i_con_lim
    stage1 = dm > dmax
    stage1 && (u = vmeas + d * (dmax / dm))
    ias = _safe_abs(im_s)
    stage2 = ias > g.i_con_lim
    stage2 && (u = vmeas + (u - vmeas) * (g.i_con_lim / ias))
    return (; uc = u, vmeas, imeas, boost, stage1, stage2)
end

"""
    droop_rates!(dx, g, xc, vk, ic, omega_b)

The droop controller's rates (the first eight states) at the terminal voltage `vk` and the injected current `ic`.
"""
function droop_rates!(dx, g::PHDroop, xc, vk, ic, omega_b)
    S = vk * conj(ic)
    omega = 1 - g.mp * (xc[2] - g.p_set)
    uref = g.u_set - g.mq * (xc[3] - g.q_set)
    rinv = cis(-xc[1])
    vc, icc = vk * rinv, ic * rinv
    dx[1] = omega * g.omega_n - omega_b
    dx[2] = (real(S) - xc[2]) / g.Tf_p
    dx[3] = (imag(S) - xc[3]) / g.Tf_q
    dx[4] = (uref - xc[4]) / g.Tv
    dx[5], dx[6] = real(vc - complex(xc[5], xc[6])) / g.T_V, imag(vc - complex(xc[5], xc[6])) / g.T_V
    dx[7], dx[8] = real(icc - complex(xc[7], xc[8])) / g.T_I, imag(icc - complex(xc[7], xc[8])) / g.T_I
    return dx
end

droop_omega(g::PHDroop, xc) = 1 - g.mp * (xc[2] - g.p_set)

"""
    droop_sample!(xc, g, vk, ic, omega_b)

One call of the DLL (sampled): the output from the states before the call, held; then the explicit Euler step.
"""
function droop_sample!(xc, g::PHDroop, vk, ic, omega_b)
    uc = droop_output(g, xc).uc
    d = droop_rates!(zeros(promote_type(eltype(xc), typeof(real(vk))), 8), g, xc, vk, ic, omega_b)
    if g.dcr                                         # V <- lam V + Z, with the lagged signals before their step
        Vv = g.dlam * complex(xc[11], xc[12]) + complex(xc[5], xc[6])
        Vi = g.dlam * complex(xc[13], xc[14]) + complex(xc[7], xc[8])
        xc[11], xc[12], xc[13], xc[14] = real(Vv), imag(Vv), real(Vi), imag(Vi)
    end
    for k in 1:8
        xc[k] += g.Ts * d[k]
    end
    h = _held_from(g, uc, xc)                    # the command, or (physical DC side, :held) the modulation at the sampled v_dc
    xc[9], xc[10] = real(h), imag(h)
    return xc
end

# ---------------------------------------------------------------------------------------------------------------
# The Andronov-Hopf virtual-oscillator controller (record sections 28-32; S8.5): the EMT path of PowerFactory's DLL
# gfm_voc_pf v1.2 (legacy AHO mode, pvoc_mode = 0), sampled at the DLL's call period only. At a call, with the measured
# terminal voltage u and injected current i (simulator frame, system base, all units of the bus combined):
#   i*   = (p_set - j q_set) / conj(u)   if |u| >= 1e-3, else 0        (current_ref.h; i_ref_max = 0, no circular limit)
#   dv   = xi (V_nom^2 - |v|^2) v + eta e^{j phi} (i* - i_f)           (aho_oscillator.h)
#   u_c  = the predictive divider on v and u (section 2; vsm_divider), held to the next call
#   v   <- e^{j (omega_n - omega_b) Ts} (v + Ts dv)                    (the DLL's Euler step, then its exact rotation at
#                                                                        omega_n, seen in the frame turning at omega_b)
# i_f = i, or with `dc_reject` (the DLL variant gfm_voc_pf_dcr, section 30) the current through the droop's DC-rejection
# filter in the frame of v: Z = i conj(v)/m, i_f = c (Z - g V) v/m, V <- lam V + Z, where m is |v| at the previous call (the
# DLL's cached aho.mag(), section 32: its Park transform uses v_k/|v_{k-1}|, kept as a state). Identical units combined:
# eta/n, Zs/n, r_vi/n, x_vi/n, kpr/n^2, kpx/n^2, i_lim n, p_set and q_set n.
# ---------------------------------------------------------------------------------------------------------------

"""
    PHVoc

The virtual-oscillator controller of one converter (all its identical units, system base), sampled (`Ts > 0`). States
`v_Re, v_Im` (the oscillator's voltage, simulator frame), `uc_Re, uc_Im` (the held output, turning at `-omega_b`; with a
physical DC side in mode `:held` the held modulation `m_Re, m_Im`), with `dc_reject` the filter state `dcrV_d, dcrV_q` and
`vmag_prev` (|v| at the previous call), and with a physical DC side `dc` (`PHConverterDC`, record section 32 part A) the DC-link
voltage and the battery's branch voltages. Between calls `v` holds the value for the
next call (GPT V3, items 4-5): the speed and the divider's predicted current recorded between calls are taken at that state, not at
the call that made the held output (as for the sampled VSM, record section 10).
"""
struct PHVoc
    name::String
    xi::Float64
    eta::Float64
    V_nom::Float64
    omega_n::Float64
    J::ComplexF64                # e^{j phi}
    Zs::Float64
    i_lim::Float64
    r_vi::Float64
    x_vi::Float64
    kpr::Float64
    kpx::Float64
    p_set::Float64
    q_set::Float64
    Ts::Float64
    dcr::Bool                    # the oscillator reads a DC-rejected current (section 30)
    dlam::ComplexF64             # the filter's lam, g, c
    dg::ComplexF64
    dcc::ComplexF64
    dc::Union{Nothing,PHConverterDC}   # the physical DC side (battery, link, bridge), or the ideal source
end

nstates(g::PHVoc) = 4 + (g.dcr ? 3 : 0) + (g.dc === nothing ? 0 : 1 + nbranches(g.dc))
state_names(g::PHVoc) = vcat(["v_Re", "v_Im"], (g.dc !== nothing && g.dc.mode === :held) ? ["m_Re", "m_Im"] : ["uc_Re", "uc_Im"],
                             g.dcr ? ["dcrV_d", "dcrV_q", "vmag_prev"] : String[],
                             g.dc === nothing ? String[] : _dc_state_names(g.dc))
name(g::PHVoc) = g.name
_held(g::PHVoc, xc) = complex(xc[3], xc[4])
_dcidx(g::PHVoc) = g.dcr ? 8 : 5

function PHVoc(cv::PHConverter; p_set::Real, q_set::Real, V_nom::Real, Ts::Real, dc_reject::Bool = false, dc = nothing)
    p, n = cv.comp.p, cv.units
    (isfinite(Ts) && Ts > 0) ||
        throw(ArgumentError("PHVoc: $(cv.name): only the sampled controller is ported (Ts > 0, the DLL's call period; it is $Ts)"))
    for (k, v) in (("eta", p.eta), ("omega_n", p.omega_n), ("Zseries", p.Zseries), ("i_lim", p.i_lim), ("V_nom", V_nom))
        (isfinite(v) && v > 0) || throw(ArgumentError("PHVoc: $(cv.name) needs $k > 0 (it is $v)"))
    end
    for (k, v) in (("xi", p.xi), ("r_vi", p.r_vi), ("x_vi", p.x_vi), ("kpr", p.kpr), ("kpx", p.kpx))
        (isfinite(v) && v >= 0) || throw(ArgumentError("PHVoc: $(cv.name) needs $k >= 0 (it is $v)"))
    end
    all(isfinite, (p_set, q_set, p.phi)) || throw(ArgumentError("PHVoc: $(cv.name): p_set, q_set and phi must be finite"))
    (p.pvoc_mode <= 0.5 && p.i_ref_max == 0 && p.pf_frame <= 0.5) ||
        throw(ArgumentError("PHVoc: $(cv.name) uses a feature this port does not cover (pvoc_mode = 0, i_ref_max = 0, " *
                            "pf_frame = 0 needed)"))
    p.Zseries / n ≈ cv.Lc || error("PHVoc: $(cv.name)'s reactance differs from its branch's")
    if dc !== nothing
        (dc.mode in (:held, :continuous) && all(isfinite, (dc.c, dc.Rb, dc.G, dc.m_max, dc.Eb)) && dc.c > 0 && dc.Rb > 0 &&
         dc.G >= 0 && dc.m_max > 0 && dc.Eb > 0 && length(dc.R) == length(dc.cb) && all(>(0), dc.R) && all(>(0), dc.cb)) ||
            throw(ArgumentError("PHVoc: $(cv.name): DC data outside their domain: $dc"))
    end
    lam, gg, cc = dc_reject ? _dcr_constants(p.omega_n, Ts) : (0.0im, 0.0im, 0.0im)
    return PHVoc(cv.name, p.xi, p.eta / n, Float64(V_nom), p.omega_n, cis(p.phi), p.Zseries / n, p.i_lim * n, p.r_vi / n,
                 p.x_vi / n, p.kpr / n^2, p.kpx / n^2, Float64(p_set), Float64(q_set), Float64(Ts), dc_reject, lam, gg, cc, dc)
end

"""
    voc_rates(g, xc, vk, ic) -> (; dv, iref, ifilt, Z)

The oscillator's derivative (without the rotation at `omega_n`, applied in the step) at the states `xc`, the measured
terminal voltage `vk` and injected current `ic`; the current reference, the current the oscillator reads, and with
`dc_reject` the filter's input `Z` (the current in the frame of `v`).
"""
function voc_rates(g::PHVoc, xc, vk, ic)
    v = complex(xc[1], xc[2])
    iref = abs2(vk) < 1e-6 ? zero(vk) : (g.p_set - im * g.q_set) / conj(vk)
    ifilt, Z = ic, zero(ic)
    if g.dcr
        m = xc[7]
        rot = m > 1e-6 ? v / m : zero(v)           # the DLL's (v_alpha, v_beta)/|v| with the cached magnitude
        Z = ic * conj(rot)
        ifilt = g.dcc * (Z - g.dg * complex(xc[5], xc[6])) * rot
    end
    dv = g.xi * (g.V_nom^2 - abs2(v)) * v + g.eta * g.J * (iref - ifilt)
    return (; dv, iref, ifilt, Z)
end

"""
    voc_sample!(xc, g, vk, ic, omega_b)

One call of the DLL: the held output from the divider on the oscillator's voltage before the call and the measured `vk`
(with a physical DC side in mode `:held`, the modulation: the command over the DC voltage sampled at the call, clipped); then
the oscillator's Euler step with the DLL's rotation, and the filter's update.
"""
function voc_sample!(xc, g::PHVoc, vk, ic, omega_b)
    v = complex(xc[1], xc[2])
    uc = vsm_divider(g, v, 0.0, vk).uc
    r = voc_rates(g, xc, vk, ic)
    vn = cis((g.omega_n - omega_b) * g.Ts) * (v + g.Ts * r.dv)
    if g.dcr
        V = g.dlam * complex(xc[5], xc[6]) + r.Z
        xc[5], xc[6], xc[7] = real(V), imag(V), abs(v)
    end
    h = _held_from(g, uc, xc)
    xc[1], xc[2], xc[3], xc[4] = real(vn), imag(vn), real(h), imag(h)
    return xc
end

# the oscillator's speed (per unit of omega_n) at the states and the measured signals: the DLL's monitor `omega`
# (omega_n + Im(dv conj v)/|v|^2, rad/s in EMT) over omega_n
function voc_omega(g::PHVoc, xc, vk, ic)
    v = complex(xc[1], xc[2])
    return 1 + imag(voc_rates(g, xc, vk, ic).dv * conj(v)) / (abs2(v) * g.omega_n)
end

# ---------------------------------------------------------------------------------------------------------------
# The grid-following converter (record sections 36-39; S8.6): PowerFactory's DLL gfl_pf v3.1 on its EMT path, sampled at the
# DLL's call period, and the static generator's EMT current-source model with its built-in current controller (TechRef
# StaticGenerator 7.1, eqs. 81-85), which makes the EMF on the converter's source row. System base, the bus's identical units
# combined (currents, powers, i_max, the ramps, the droop gains, Kqf times n; Zseries, Kd, Kq divided by n; the gains kp_P,
# ki_P, kp_Q, ki_Q, the PLL's and the time constants unchanged: section 36).
#
# The DLL (ten states, held between calls; at a call, from the terminal voltage u only, every right-hand side at the states
# before the call): vd + j vq = u e^{-j theta};  omega = omega_n + kp_pll vq + ki_pll phi;  the measurement lags v_f, f_f; the
# droops; the dead-beat ramps p_r, q_r; P* = p_r + dP, Q* = q_r + dQ;  p = vd id_l, q = -vd iq_l;  the outer PI
# id_raw = kp_P e_P + ki_P gamma_P, iq_raw = -(kp_Q e_Q + ki_Q gamma_Q) + Kqf vq;  the circular limit i_max;  the gate
# clamp((vd - 0.1)/0.2, 0, 1);  the anti-windup (frozen on the limit, faded with the gate);  the lag T_cc. Its outputs, latched
# at the call from the states before the step and held in the stationary frame as PowerFactory holds them (section 39):
# the frame c_h = e^{j theta} (turning at -omega_b between calls in the simulator's frame) and the reference i*_r = id_l + j iq_l.
# Then the Euler step, theta in the simulator's frame: theta <- theta + Ts (omega - omega_b).
#
# The static generator (continuous; two states z = z_d + j z_q, its PI integrators in the held frame), with the injected
# current i (a state of the network) and i_h = i conj(c_h), e = i*_r - i_h:
#     z_d' = (Kd/Td) e_d,  z_q' = (Kq/Tq) e_q,   u_c = ((Kd e_d + z_d - x1 i_hq) + j (Kq e_q + z_q + x1 i_hd)) c_h,
# x1 = Zseries (the branch's Lc). u_c depends on states only: no algebraic loop.
# ---------------------------------------------------------------------------------------------------------------

"""
    PHGfl

The grid-following controller of one converter (all its identical units, system base), sampled (`Ts > 0`). States: the DLL's
`theta` (the PLL angle in the simulator's frame), `phi`, `v_f`, `f_f`, `p_r`, `q_r`, `gamma_P`, `gamma_Q`, `id_l`, `iq_l`; the held
frame `ch_Re, ch_Im` and reference `ir_d, ir_q`; the static generator's PI integrators `z_d, z_q` (record section 38); the DLL's
monitor `omega` (per unit of `omega_n`) from the last call, held as the DLL holds its outputs (record section 40: recomputed from
the state between calls it reads the voltage after the call's frame reset, 9e-4 pu off the PLL's frequency).
"""
struct PHGfl
    name::String
    kp_pll::Float64
    ki_pll::Float64
    omega_n::Float64
    phi_max::Float64
    i_max::Float64
    T_cc::Float64                # the DLL's effective floors: max(T_cc, 1e-6), max(T_meas, 1e-4) (G1 item 1)
    kp_P::Float64
    ki_P::Float64
    kp_Q::Float64
    ki_Q::Float64
    k_droop_p::Float64
    f_db::Float64
    k_droop_q::Float64
    v_db::Float64
    r_p_max::Float64
    r_q_max::Float64
    T_meas::Float64
    Kqf::Float64
    p_ref::Float64
    q_ref::Float64
    x1::Float64                  # the static generator's series reactance (= the branch's Lc)
    Kd::Float64
    Td::Float64
    Kq::Float64
    Tq::Float64
    Ts::Float64
    advance::Float64             # the output angle advanced by advance * omega_k * Ts (record section 42; 0: the DLL as it is)
    dc::Union{Nothing,PHConverterDC}   # the physical DC side (battery, link, bridge; record section 43), or the ideal source
end
PHGfl(name::String, v::Vararg{Real,26}) = PHGfl(name, v..., 0.0, nothing)
PHGfl(name::String, v::Vararg{Real,27}) = PHGfl(name, v..., nothing)

const _PHGFL_STATES = ["theta", "phi", "v_f", "f_f", "p_r", "q_r", "gamma_P", "gamma_Q", "id_l", "iq_l",
                       "ch_Re", "ch_Im", "ir_d", "ir_q", "z_d", "z_q", "omega_call"]
nstates(g::PHGfl) = 17 + (g.dc === nothing ? 0 : 1 + nbranches(g.dc))
state_names(g::PHGfl) = g.dc === nothing ? _PHGFL_STATES : vcat(_PHGFL_STATES, _dc_state_names(g.dc))
_dcidx(g::PHGfl) = 18
name(g::PHGfl) = g.name

function PHGfl(cv::PHConverter; Ts::Real, advance::Real = 0.0, dc = nothing)
    p, n = cv.comp.p, cv.units
    (isfinite(Ts) && Ts > 0) ||
        throw(ArgumentError("PHGfl: $(cv.name): only the sampled controller is ported (Ts > 0, the DLL's call period; it is $Ts)"))
    for (k, v) in (("kp_pll", p.kp_pll), ("omega_n", p.omega_n), ("i_max", p.i_max), ("T_cc", p.T_cc), ("T_meas", p.T_meas),
                   ("r_p_max", p.r_p_max), ("r_q_max", p.r_q_max), ("Zseries", p.Zseries), ("Kd", p.Kd), ("Td", p.Td),
                   ("Kq", p.Kq), ("Tq", p.Tq))
        (isfinite(v) && v > 0) || throw(ArgumentError("PHGfl: $(cv.name) needs $k > 0 (it is $v)"))
    end
    for (k, v) in (("ki_pll", p.ki_pll), ("phi_max", p.phi_max), ("kp_P", p.kp_P), ("ki_P", p.ki_P), ("kp_Q", p.kp_Q),
                   ("ki_Q", p.ki_Q), ("k_droop_p", p.k_droop_p), ("f_db", p.f_db), ("k_droop_q", p.k_droop_q), ("v_db", p.v_db))
        (isfinite(v) && v >= 0) || throw(ArgumentError("PHGfl: $(cv.name) needs $k >= 0 (it is $v)"))
    end
    p.Kqf == 0 || throw(ArgumentError("PHGfl: $(cv.name) has Kqf = $(p.Kqf); its sign is not settled (record section 36)"))
    (p.ki_P > 1e-9 && p.ki_Q > 1e-9) ||
        throw(ArgumentError("PHGfl: $(cv.name): ki_P and ki_Q must be positive (the DLL's start is no equilibrium otherwise)"))
    isfinite(advance) || throw(ArgumentError("PHGfl: advance = $advance"))
    if dc !== nothing                            # the bridge follows the static generator's continuous EMF (record section 43)
        dc.mode === :continuous ||
            throw(ArgumentError("PHGfl: $(cv.name): the physical DC side needs mode :continuous (the EMF is not held; section 43)"))
        (all(isfinite, (dc.c, dc.Rb, dc.G, dc.m_max, dc.Eb)) && dc.c > 0 && dc.Rb > 0 && dc.G >= 0 && dc.m_max > 0 && dc.Eb > 0 &&
         length(dc.R) == length(dc.cb) && all(>(0), dc.R) && all(>(0), dc.cb)) ||
            throw(ArgumentError("PHGfl: $(cv.name): DC data outside their domain: $dc"))
    end
    p.Zseries / n ≈ cv.Lc || error("PHGfl: $(cv.name)'s reactance differs from its branch's")
    return PHGfl(cv.name, p.kp_pll, p.ki_pll, p.omega_n, p.phi_max, p.i_max * n, max(p.T_cc, 1e-6), p.kp_P, p.ki_P, p.kp_Q,
                 p.ki_Q, p.k_droop_p * n, p.f_db, p.k_droop_q * n, p.v_db, p.r_p_max * n, p.r_q_max * n, max(p.T_meas, 1e-4),
                 p.Kqf * n, p.p_ref * n, p.q_ref * n, p.Zseries / n, p.Kd / n, p.Td, p.Kq / n, p.Tq, Float64(Ts), Float64(advance), dc)
end

# droop.h: the shifted dead band and the de-sync gate
@inline _gfl_deadband(x, db) = x > db ? x - db : (x < -db ? x + db : zero(x))
@inline _gfl_gate(vd) = vd >= 0.3 ? one(vd) : (vd <= 0.1 ? zero(vd) : (vd - 0.1) / 0.2)

"""
    gfl_call(g, xc, vk) -> (; d, vd, vq, omega, sat, gate, idc, iqc)

The DLL's `Model_Outputs` at the states `xc` and the terminal voltage `vk` (simulator frame): the rates `d` of the ten DLL
states (`d[1]` is the PLL's `omega` in rad/s; the angle's step in the simulator's frame subtracts `omega_b`) and the monitors.
"""
function gfl_call(g::PHGfl, xc, vk)
    T = promote_type(eltype(xc), typeof(real(vk)))
    d = zeros(T, 10)
    v = vk * cis(-xc[1])
    vd, vq = real(v), imag(v)
    omega = g.omega_n + g.kp_pll * vq + g.ki_pll * xc[2]
    d[1], d[2] = omega, vq
    d[3] = (abs(vk) - xc[3]) / g.T_meas
    d[4] = (omega / g.omega_n - xc[4]) / g.T_meas
    dQ = -g.k_droop_q * _gfl_deadband(xc[3] - 1, g.v_db)
    dP = -g.k_droop_p * _gfl_deadband(xc[4] - 1, g.f_db)
    d[5] = clamp(g.p_ref - xc[5], -g.r_p_max * g.Ts, g.r_p_max * g.Ts) / g.Ts
    d[6] = clamp(g.q_ref - xc[6], -g.r_q_max * g.Ts, g.r_q_max * g.Ts) / g.Ts
    eP = (xc[5] + dP) - vd * xc[9]
    eQ = (xc[6] + dQ) + vd * xc[10]
    idr = g.kp_P * eP + g.ki_P * xc[7]
    iqr = -(g.kp_Q * eQ + g.ki_Q * xc[8]) + g.Kqf * vq
    m = sqrt(idr^2 + iqr^2)
    sat = m > g.i_max && m > 0
    sat && (idr *= g.i_max / m; iqr *= g.i_max / m)
    gate = _gfl_gate(vd)
    aw = sat ? zero(gate) : gate
    d[7], d[8] = aw * eP, aw * eQ
    d[9] = (gate * idr - xc[9]) / g.T_cc
    d[10] = (gate * iqr - xc[10]) / g.T_cc
    return (; d, vd, vq, omega, sat, gate, idc = gate * idr, iqc = gate * iqr)
end

"""
    gfl_sample!(xc, g, vk, omega_b)

One call of the DLL: the held frame (with `g.advance > 0`, its angle advanced by `advance * omega * Ts`, record section 42) and
reference latched from the states before the call, then the Euler step (`phi`
clamped to `+-phi_max`, the angle in the simulator's frame). The static generator's integrators are not touched.
"""
function gfl_sample!(xc, g::PHGfl, vk, omega_b)
    c = gfl_call(g, xc, vk)
    h = cis(xc[1] + g.advance * c.omega * g.Ts)  # the DLL's output angle, advanced in the variant (record section 42)
    xc[11], xc[12], xc[13], xc[14] = real(h), imag(h), xc[9], xc[10]
    xc[17] = c.omega / g.omega_n
    xc[1] += g.Ts * (c.omega - omega_b)
    xc[2] = clamp(xc[2] + g.Ts * c.d[2], -g.phi_max, g.phi_max)
    for k in 3:10
        xc[k] += g.Ts * c.d[k]
    end
    return xc
end

"""
    gfl_emf(g, xc, ic) -> u_c

The static generator's EMF (TechRef 7.1, eqs. 81-85) at the states `xc` and the injected current `ic`.
"""
function gfl_emf(g::PHGfl, xc, ic)
    ch = complex(xc[11], xc[12])
    ih = ic * conj(ch)
    ed, eq = xc[13] - real(ih), xc[14] - imag(ih)
    return complex(g.Kd * ed + xc[15] - g.x1 * imag(ih), g.Kq * eq + xc[16] + g.x1 * real(ih)) * ch
end

"""
    gfl_rates!(dx, g, xc, ic, omega_b)

The rows between calls: the DLL's states and the held reference constant, the held frame turning at `-omega_b`, the static
generator's PI integrators on the error in the held frame.
"""
function gfl_rates!(dx, g::PHGfl, xc, ic, omega_b)
    for k in (1:10..., 13, 14, 17)
        dx[k] = 0
    end
    dx[11], dx[12] = omega_b * xc[12], -omega_b * xc[11]
    ih = ic * conj(complex(xc[11], xc[12]))
    dx[15] = g.Kd / g.Td * (xc[13] - real(ih))
    dx[16] = g.Kq / g.Tq * (xc[14] - imag(ih))
    return dx
end

# the voltage the converter applies: the static generator's EMF, or with a physical DC side the bridge's, `u_x = m v_dc` with the
# radial limit |m| <= m_max (the bridge follows the EMF continuously; record section 43)
function gfl_bridge(g::PHGfl, xc, ic)
    u = gfl_emf(g, xc, ic)
    g.dc === nothing && return u
    return _radial_clip(u, g.dc.m_max * _vdc_checked(xc[_dcidx(g)]))
end
_modulation(g::PHGfl, xc, ic) = gfl_bridge(g, xc, ic) / _vdc_checked(xc[_dcidx(g)])

# the DLL's monitor `omega` (per unit of omega_n): its value at the last call, held (record section 40)
gfl_omega(g::PHGfl, xc) = xc[17]

"""
    gfl_initial(g, vk, ic, E0) -> x

The DLL's `Model_Initialize` at the terminal voltage `vk` (simulator frame, t = 0): `theta = arg vk`, `phi = 0`, `v_f = |vk|`,
`f_f = 1`, the ramps at the set-points, `id0 = P*/|vk|`, `iq0 = -Q*/|vk|` (limited), the outer PI pre-charged, the lag at the gated
command; the held frame and reference as the first call would latch them; the static generator's integrators such that
`u_c = E0` at `ic` (G2 item 7: the DLL's start, not the sampled system's settled orbit).
"""
function gfl_initial(g::PHGfl, vk, ic, E0)
    th, vm = angle(vk), abs(vk)
    vs = vm > 0.01 ? vm : 1.0
    dQ = -g.k_droop_q * _gfl_deadband(vm - 1, g.v_db)
    id0, iq0 = g.p_ref / vs, -(g.q_ref + dQ) / vs
    m = sqrt(id0^2 + iq0^2)
    m > g.i_max && (id0 *= g.i_max / m; iq0 *= g.i_max / m)
    gate = _gfl_gate(vm)
    x = zeros(nstates(g))
    x[1:10] .= (th, 0.0, vm, 1.0, g.p_ref, g.q_ref, id0 / g.ki_P, -iq0 / g.ki_Q, gate * id0, gate * iq0)
    x[17] = 1.0                                  # Model_Initialize: pll.omega = omega_n
    ch = cis(th + g.advance * g.omega_n * g.Ts)   # as the first call would latch it (omega = omega_n at the start)
    x[11], x[12], x[13], x[14] = real(ch), imag(ch), x[9], x[10]
    ih = ic * conj(ch)
    ed, eq = x[13] - real(ih), x[14] - imag(ih)
    u = E0 * conj(ch)
    x[15] = real(u) - g.Kd * ed + g.x1 * imag(ih)
    x[16] = imag(u) - g.Kq * eq - g.x1 * real(ih)
    g.dc === nothing || (x[18:end] .= _dc_initial(g.dc))
    return x
end
