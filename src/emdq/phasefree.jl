# The phase-free pilot of the stationary-frame orbital test (docs/records/early/STABILITY_INVESTIGATION.md
# section 23). The fixed-voltage boundary bus of an `EMDQPilot` is replaced by a passive node
# derived from the equilibrium, so the closed pilot has the rotational symmetry (every phasor
# pair rotated together with a shift of the rotor angle) and its synchronous solution is a
# free S^1 orbit. This is a new closed pilot, not the infinite-bus pilot. The quotient by the
# symmetry (every phasor in the rotor's frame, the rotor angle removed) is an autonomous system
# whose equilibrium is the orbit.

"""
    EMDQGovernor

The case's IEEEG1 governor (`IEEEG1_PHTRUE`) in its equilibrium limiter mode: no valve
position, servo rate or non-windup limit active, so its six lag states obey the linear
equations of `ieeeg1.jl`. The steam tank (`x_steam`), a one-way account that never feeds
back, is not modelled. The governor's states carry no physical storage. `Pref` is set so that
its equilibrium torque is the pilot's shaft torque (`emdq_governed`).
"""
struct EMDQGovernor
    K::Float64
    T1::Float64
    T2::Float64
    T3::Float64
    T4::Float64
    T5::Float64
    T6::Float64
    T7::Float64
    K1::Float64
    K3::Float64
    K5::Float64
    K7::Float64
    PMAX::Float64
    PMIN::Float64
    UO::Float64
    UC::Float64
    R_base::Float64
    Pref::Float64
end

"""
    EMDQPhaseFree(pilot, C8, G8, L8[, gov])

An `EMDQPilot` whose fixed bus (`pilot.V4` is then unused) is replaced by a node with voltage
`v8`, capacitor `C8`, conductance `G8` and, when `L8` is finite, a parallel inductor (current
`iL8`). With a governor `gov` (`EMDQGovernor`) the shaft torque is the governor's output
instead of the fixed `pilot.Tm`. Plant state: the pilot's `x` (18 or 19 entries), then `v8`,
then `iL8` if present, then the six governor states if present.
"""
struct EMDQPhaseFree
    pilot::EMDQPilot
    C8::Float64
    G8::Float64
    L8::Float64
    gov::Union{Nothing,EMDQGovernor}
end
EMDQPhaseFree(pilot::EMDQPilot, C8::Real, G8::Real, L8::Real) = EMDQPhaseFree(pilot, C8, G8, L8, nothing)

_pf_nx(pf::EMDQPhaseFree) = pf.pilot.mach.nq == 2 ? 19 : 18
_pf_has_L(pf::EMDQPhaseFree) = isfinite(pf.L8)
_pf_ngrid(pf::EMDQPhaseFree) = _pf_nx(pf) + 2 + (_pf_has_L(pf) ? 2 : 0)
"""    emdq_pf_nstates(pf) -> the number of plant states (without the PI integrator)."""
emdq_pf_nstates(pf::EMDQPhaseFree) = _pf_ngrid(pf) + (pf.gov === nothing ? 0 : 6)

"""    emdq_pf_state_names(pf) -> plant state names (`emdq_state_names`, `v8`, `iL8`, then the governor's `gov_x0` to `gov_x5`)."""
emdq_pf_state_names(pf::EMDQPhaseFree) =
    vcat(emdq_state_names(pf.pilot.mach), ["v8_Re", "v8_Im"], _pf_has_L(pf) ? ["iL8_Re", "iL8_Im"] : String[],
         pf.gov === nothing ? String[] : ["gov_x$k" for k in 0:5])

"""    emdq_gov_torque(g, xg) -> the governor's mechanical torque `K1 x2 + K3 x3 + K5 x4 + K7 x5`."""
emdq_gov_torque(g::EMDQGovernor, xg) = g.K1 * xg[3] + g.K3 * xg[4] + g.K5 * xg[5] + g.K7 * xg[6]

"""
    emdq_gov_rhs(g, xg, omega) -> the governor's six lag derivatives in its linear mode

`err = Pref - 1 - R_base (omega - 1)`, lead-lag `K (1 + s T2)/(1 + s T1)`, servo `1/(1 + s T3)`,
turbine stages `T4` to `T7` (`ieeeg1.jl` with every limit inactive).
"""
function emdq_gov_rhs(g::EMDQGovernor, xg, omega)
    err = g.Pref - 1.0 - g.R_base * (omega - 1.0)
    GV = xg[1] * (1.0 - g.T2 / g.T1) + g.K * (g.T2 / g.T1) * err
    return [(g.K * err - xg[1]) / g.T1, (GV - xg[2]) / g.T3, (xg[2] - xg[3]) / g.T4,
            (xg[3] - xg[4]) / g.T5, (xg[4] - xg[5]) / g.T6, (xg[5] - xg[6]) / g.T7]
end

"""
    emdq_gov_mode_margins(g, xg, omega) -> (; position, rate)

How far the state is inside the governor's equilibrium limiter mode: the valve position margin
`min(x1 - PMIN, PMAX - x1)` and the servo rate margin `min(rate - UC, UO - rate)`. Both must
stay positive on any certified region (a domain condition).
"""
function emdq_gov_mode_margins(g::EMDQGovernor, xg, omega)
    rate = emdq_gov_rhs(g, xg, omega)[2]
    return (; position = min(xg[2] - g.PMIN, g.PMAX - xg[2]), rate = min(rate - g.UC, g.UO - rate))
end

"""
    emdq_governed(pf, zs, p) -> (pfg, zsg, record)

The phase-free pilot `pf` (without a governor; equilibrium `zs`) closed by the case's IEEEG1
governor with parameters `p` (an `Ieeeg1Params`), in its equilibrium limiter mode. The
electromagnetic equilibrium and the shaft torque `T* = pilot.Tm` are kept: every lag state is
`T*` (the turbine fractions sum to one, checked) and `Pref = 1 + T*/K`. The equilibrium must
be strictly inside the limiter mode (checked).
"""
function emdq_governed(pf::EMDQPhaseFree, zs, p)
    pf.gov === nothing || throw(ArgumentError("emdq_governed: the pilot already has a governor"))
    Ts = pf.pilot.Tm
    ksum = p.K1 + p.K3 + p.K5 + p.K7
    abs(ksum - 1) < 1e-12 || error("emdq_governed: turbine fractions sum to $ksum, not 1")
    g = EMDQGovernor(p.K, p.T1, p.T2, p.T3, p.T4, p.T5, p.T6, p.T7, p.K1, p.K3, p.K5, p.K7,
                     p.PMAX, p.PMIN, p.UO, p.UC, p.R_base, 1.0 + Ts / p.K)
    xg = fill(Ts, 6)
    mm = emdq_gov_mode_margins(g, xg, 1.0)
    (mm.position > 0 && mm.rate > 0) || error("emdq_governed: the equilibrium is not inside the limiter mode")
    pfg = EMDQPhaseFree(pf.pilot, pf.C8, pf.G8, pf.L8, g)
    return pfg, vcat(zs, xg), (; Tm = Ts, Pref = g.Pref, margins = mm, droop_torque_per_speed = g.K * g.R_base)
end

"""
    emdq_phasefree(s, xs, b78) -> (pf, zs, record)

The phase-free variant of the pilot `s` at its equilibrium `xs`. `b78` is the total charging
of the line to the fixed bus (the case value; the pilot kept only its near half, in `C3`).
From the equilibrium values at the fixed bus, `V8 = s.V4` and `S8 = V8 conj(i34)`:
`C8 = b78/2` (the line's remote half-charging), the remaining demand
`S_L = S8 + j C8 |V8|^2`, `G8 = Re S_L / |V8|^2`, and either an inductor
`L8 = |V8|^2 / Im S_L` (when `Im S_L > 0`) or `-Im S_L / |V8|^2` added to `C8`. Passivity
(`G8 >= 0`, `C8 > 0`, `L8 > 0`) is checked, not assumed. `zs` is the plant equilibrium
(`xs`, then `v8 = V8`, then `iL8`), a point of the new pilot's relative equilibrium at
`omega_b`.
"""
function emdq_phasefree(s::EMDQPilot, xs, b78::Real)
    V8 = s.V4
    i34 = complex(xs[15], xs[16])
    S8 = V8 * conj(i34)
    C8 = b78 / 2
    SL = S8 + im * C8 * abs2(V8)
    G8 = real(SL) / abs2(V8)
    if imag(SL) > 0
        L8 = abs2(V8) / imag(SL)
    else
        C8 += -imag(SL) / abs2(V8)
        L8 = Inf
    end
    (G8 >= 0 && C8 > 0 && L8 > 0) ||
        error("emdq_phasefree: the derived bus-8 load is not passive (G8 $G8, C8 $C8, L8 $L8)")
    pf = EMDQPhaseFree(s, C8, G8, L8)
    zs = vcat(xs, real(V8), imag(V8))
    if isfinite(L8)
        iL8 = V8 / (im * L8)
        append!(zs, [real(iL8), imag(iL8)])
    end
    rec = (; V8, S8, S_load = SL, C8, G8, L8, inductive = isfinite(L8))
    return pf, zs, rec
end

"""
    emdq_pf_rhs!(dx, pf, x, m = pf.pilot.mstar) -> dx

The phase-free pilot's field: the pilot's (`emdq_rhs!`) with the fixed voltage in the second
line's equation replaced by `v8`, and the bus-8 node
`dv8 = (wb/C8)(i34 - G8 v8 - iL8 - j C8 v8)`, `diL8 = (wb/L8)(v8 - j L8 iL8)`.
"""
function emdq_pf_rhs!(dx, pf::EMDQPhaseFree, x, m = pf.pilot.mstar)
    s = pf.pilot
    wb = s.mach.omega_b
    nx = _pf_nx(pf)
    emdq_rhs!(view(dx, 1:nx), s, view(x, 1:nx), m)
    v8 = complex(x[nx+1], x[nx+2])
    corr = (wb / s.L34) * (s.V4 - v8)            # replaces -V4 by -v8 in di34
    dx[15] += real(corr)
    dx[16] += imag(corr)
    i34 = complex(x[15], x[16])
    iL8 = _pf_has_L(pf) ? complex(x[nx+3], x[nx+4]) : zero(v8)
    dv8 = (wb / pf.C8) * (i34 - pf.G8 * v8 - iL8 - im * pf.C8 * v8)
    dx[nx+1], dx[nx+2] = real(dv8), imag(dv8)
    if _pf_has_L(pf)
        diL8 = (wb / pf.L8) * (v8 - im * pf.L8 * iL8)
        dx[nx+3], dx[nx+4] = real(diL8), imag(diL8)
    end
    if pf.gov !== nothing                         # the shaft torque from the governor
        ng = _pf_ngrid(pf)
        xg = view(x, ng+1:ng+6)
        omega = x[7] / (2.0 * s.mach.H)
        dx[7] += emdq_gov_torque(pf.gov, xg) - s.Tm
        dx[ng+1:ng+6] .= emdq_gov_rhs(pf.gov, xg, omega)
    end
    return dx
end

emdq_pf_rhs(pf::EMDQPhaseFree, x, m = pf.pilot.mstar) = emdq_pf_rhs!(similar(x), pf, x, m)

"""    emdq_pf_energy(pf, x) -> the physical energy: the pilot's plus `(C8 |v8|^2 + L8 |iL8|^2)/(2 wb)`."""
function emdq_pf_energy(pf::EMDQPhaseFree, x)
    nx = _pf_nx(pf)
    wb = pf.pilot.mach.omega_b
    e = emdq_energy(pf.pilot, view(x, 1:nx)) + pf.C8 * (x[nx+1]^2 + x[nx+2]^2) / (2wb)
    _pf_has_L(pf) && (e += pf.L8 * (x[nx+3]^2 + x[nx+4]^2) / (2wb))
    return e
end

"""
    emdq_pf_balance(pf, x, m = pf.pilot.mstar) -> NamedTuple

The raw energy balance of the closed pilot: `dE/dt = shaft + dc_source - losses` (every loss
of `emdq_power`, plus `G8 |v8|^2`; there is no boundary bus). With a governor the shaft power
is the governor's torque times the speed; the governor's states have no physical energy.
`identity_error` is relative to the sum of the terms' magnitudes.
"""
function emdq_pf_balance(pf::EMDQPhaseFree, x, m = pf.pilot.mstar)
    nx = _pf_nx(pf)
    p = emdq_power(pf.pilot, x[1:nx], m)
    load8 = pf.G8 * (x[nx+1]^2 + x[nx+2]^2)
    shaft = p.supplies.shaft
    if pf.gov !== nothing
        ng = _pf_ngrid(pf)
        shaft = (emdq_gov_torque(pf.gov, x[ng+1:ng+6]) + pf.pilot.Dm) * x[7] / (2.0 * pf.pilot.mach.H)
    end
    supply = shaft + p.supplies.dc_source
    loss = sum(values(p.dissipation)) + load8
    rate = dot(ForwardDiff.gradient(w -> emdq_pf_energy(pf, w), x), emdq_pf_rhs(pf, x, m))
    scale = abs(supply) + abs(loss) + abs(rate)
    return (; rate, supply, loss, load8, identity_error = (rate - (supply - loss)) / scale)
end

"""    emdq_pf_phasor_pairs(pf) -> the index pairs of every phasor (stator flux, network voltages and currents)."""
function emdq_pf_phasor_pairs(pf::EMDQPhaseFree)
    nx = _pf_nx(pf)
    pairs_ = [(1, 2), (9, 10), (11, 12), (13, 14), (15, 16), (17, 18), (nx + 1, nx + 2)]
    _pf_has_L(pf) && push!(pairs_, (nx + 3, nx + 4))
    return pairs_
end

"""
    emdq_pf_rotate(pf, z, phi) -> g_phi z

The symmetry: every phasor pair rotated by `phi`, the rotor angle (index 6) shifted by `phi`,
every other state (rotor windings, momentum, `q_dc`, the PI integrator) unchanged. `z` may
carry the PI integrator after the plant states.
"""
function emdq_pf_rotate(pf::EMDQPhaseFree, z, phi)
    w = collect(z)
    c, s_ = cos(phi), sin(phi)
    for (a, b) in emdq_pf_phasor_pairs(pf)
        w[a], w[b] = c * z[a] - s_ * z[b], s_ * z[a] + c * z[b]
    end
    w[6] = z[6] + phi
    return w
end

"""
    EMDQPFClosedLoop(pf, xs, c)

The phase-free pilot with the passive PI `c` on `y_m` (references from the original pilot's
equilibrium `xs`): `z = [plant; eta]`.
"""
struct EMDQPFClosedLoop
    pf::EMDQPhaseFree
    xs::Vector{Float64}
    c::EMDQAVR
end

function emdq_pf_closed_loop_rhs(cl::EMDQPFClosedLoop, z)
    pf, c = cl.pf, cl.c
    s = pf.pilot
    np = emdq_pf_nstates(pf)
    nx = _pf_nx(pf)
    yhat = emdq_modulation_output(s, cl.xs, view(z, 1:nx)) / c.y0
    m = s.mstar + c.u0 * (-c.kappa_p * yhat - z[np+1])
    dz = similar(z)
    emdq_pf_rhs!(view(dz, 1:np), pf, view(z, 1:np), m)
    dz[np+1] = c.omega_i * yhat
    return dz
end

"""
    emdq_pf_reduce(cl, z) -> xi

The quotient coordinates (the rotor's frame): every phasor rotated by `-(delta - delta_ref)`
with `delta_ref = xs[6]`, the rotor angle removed. Invariant under `emdq_pf_rotate`.
"""
function emdq_pf_reduce(cl::EMDQPFClosedLoop, z)
    w = emdq_pf_rotate(cl.pf, z, -(z[6] - cl.xs[6]))
    return w[[1:5; 7:length(w)]]
end

"""    emdq_pf_lift(cl, xi) -> the representative `z` of `xi` with `delta = delta_ref`."""
emdq_pf_lift(cl::EMDQPFClosedLoop, xi) = vcat(xi[1:5], [zero(xi[1]) + cl.xs[6]], xi[6:end])

"""
    emdq_pf_reduced_rhs(cl, xi) -> dxi

The autonomous quotient field: at the representative `z = lift(xi)` (rotation factor one), each
phasor `X` obeys `dY/dt = dX/dt - j (d delta/dt) X`, every scalar keeps its equation, and the
rotor angle's equation is dropped (it gives the phase velocity).
"""
function emdq_pf_reduced_rhs(cl::EMDQPFClosedLoop, xi)
    z = emdq_pf_lift(cl, xi)
    dz = emdq_pf_closed_loop_rhs(cl, z)
    dd = dz[6]
    for (a, b) in emdq_pf_phasor_pairs(cl.pf)
        dz[a] += dd * z[b]
        dz[b] -= dd * z[a]
    end
    return dz[[1:5; 7:length(dz)]]
end

"""
    emdq_pf_quotient_data(cl, xi) -> NamedTuple

The data of the solver-free structure gate (G2) at a quotient equilibrium `xi`: the
closed-loop quotient Jacobian `A`, the fixed Hessian `Hfix = blockdiag(Hess E (plant part),
u0 y0/omega_i)` (the energy is invariant, so it is a function of `xi`), the declared
power-of-two scaling `d`, `As = D^-1 A D`, `Hs = D Hfix D`, `eta` (last index) and the
quotient state names.
"""
function emdq_pf_quotient_data(cl::EMDQPFClosedLoop, xi)
    n = length(xi)
    A = ForwardDiff.jacobian(w -> emdq_pf_reduced_rhs(cl, w), xi)
    Ep(w) = emdq_pf_energy(cl.pf, emdq_pf_lift(cl, vcat(w, zero(eltype(w))))[1:end-1])
    Hfix = zeros(n, n)
    Hfix[1:n-1, 1:n-1] = ForwardDiff.hessian(Ep, xi[1:n-1])
    Hfix[n, n] = cl.c.u0 * cl.c.y0 / cl.c.omega_i
    # declared power-of-two scaling; states without physical curvature (the governor's) get 1
    d = [Hfix[i, i] > 0 ? 2.0^round(-log2(Hfix[i, i]) / 2) : 1.0 for i in 1:n]
    D = Diagonal(d)
    names = vcat(emdq_pf_state_names(cl.pf), "eta")[[1:5; 7:end]]
    return (; A, Hfix, d, As = D \ A * D, Hs = D * Hfix * D, eta = n, names,
            spectral_abscissa = maximum(real, eigvals(A)))
end

"""
    emdq_pf_masks(cl) -> Vector{Pair{String,BitMatrix}}

The predeclared patterns carried to the quotient (section 23.3), as masks on the quotient
state: `local` (the machine block without the rotor angle, and `eta`), `port` (local plus
cross-terms to the terminal voltage), and the nested full blocks `radius 1` (excitation
assembly) to `radius 5` (+ second line) and `radius 6` (+ bus 8 and its load: every state).
Radii up to 5 and the two local patterns are proper; radius 6 is the full pilot. With a
governor every pattern is united with the one predeclared joint mechanical block (rotor
momentum `p` and the six governor states, full), and no governor-electrical entry is admitted.
"""
function emdq_pf_masks(cl::EMDQPFClosedLoop)
    names = vcat(emdq_pf_state_names(cl.pf), "eta")[[1:5; 7:end]]
    n = length(names)
    ix(v...) = [findfirst(==(x), names) for x in v]
    mach = ix("psi_D", "psi_Q", "psi_fd", "psi_1d", "psi_1q", "p")
    cl.pf.pilot.mach.nq == 2 && append!(mach, ix("psi_2q"))
    eta = n
    loc = vcat(mach, eta)
    full(S) = (m = falses(n, n); m[S, S] .= true; m)
    out = Pair{String,BitMatrix}[]
    push!(out, "local (machine block, eta)" => full(loc))
    pm = full(loc)
    v2 = ix("v2_Re", "v2_Im")
    pm[loc, v2] .= true
    pm[v2, loc] .= true
    push!(out, "port (local + terminal-voltage cross-terms)" => pm)
    shells = [("radius 1: excitation assembly (machine, q_dc, eta)", vcat(loc, ix("q_dc"))),
              ("radius 2: + terminal capacitor", v2), ("radius 3: + first line", ix("i23_Re", "i23_Im")),
              ("radius 4: + next bus and load", ix("v3_Re", "v3_Im", "iL_Re", "iL_Im")),
              ("radius 5: + second line", ix("i34_Re", "i34_Im")),
              ("radius 6: + bus 8 and its load (every electrical state)", setdiff(1:n, vcat(loc, ix("q_dc"), v2,
                  ix("i23_Re", "i23_Im", "v3_Re", "v3_Im", "iL_Re", "iL_Im", "i34_Re", "i34_Im"),
                  findall(startswith.(names, "gov_")))))]
    S = Int[]
    for (lab, s_) in shells
        append!(S, s_)
        push!(out, lab => full(unique(S)))
    end
    gov = startswith.(names, "gov_")
    all(out[end][2][.!gov, .!gov]) || error("emdq_pf_masks: radius 6 is not every electrical state")
    if any(gov)
        mech = vcat(ix("p"), findall(gov))
        out = [lab * " + mechanical block" => (m2 = copy(m); m2[mech, mech] .= true; m2) for (lab, m) in out]
    end
    return out
end

"""
    emdq_pf_interval_data(cl, xi, d) -> NamedTuple

Model-bound enclosures on the quotient, as `emdq_interval_data` for the pilot: a 256-bit
Newton centre, the Krawczyk test on the quotient field (unique quotient equilibrium in the
box, so the orbit is isolated modulo the symmetry), the quotient Jacobian over the enclosure
and the energy Hessian over its plant part, scaled by `d.d`.
"""
function emdq_pf_interval_data(cl::EMDQPFClosedLoop, xi, d)
    G(w) = emdq_pf_reduced_rhs(cl, w)
    n = length(xi)
    zb = BigFloat.(xi)
    for _ in 1:6
        zb -= ForwardDiff.jacobian(G, zb) \ G(zb)
    end
    center = Float64.(zb)
    C = inv(ForwardDiff.jacobian(G, center))
    r0 = 4 .* abs.(C * G(center)) .+ 8 .* eps.(max.(abs.(center), 1e-3))
    Z, X, rec = krawczyk(G, B -> ForwardDiff.jacobian(G, B), "phase-free quotient equilibrium", center, r0; C)
    AI = ForwardDiff.jacobian(G, Z)
    well_defined(AI) || fail("emdq_pf_interval_data: the Jacobian is not well defined on the box")
    Ep(w) = emdq_pf_energy(cl.pf, emdq_pf_lift(cl, vcat(w, zero(eltype(w))))[1:end-1])
    HI = fill(ival(0.0), n, n)
    HI[1:n-1, 1:n-1] = ForwardDiff.hessian(Ep, Z[1:n-1])
    HI[n, n] = ival(cl.c.u0) * ival(cl.c.y0) / ival(cl.c.omega_i)
    well_defined(HI) || fail("emdq_pf_interval_data: the Hessian is not well defined on the box")
    dI = ival.(d.d)
    AsI = [AI[i, j] * dI[j] / dI[i] for i in 1:n, j in 1:n]
    HsI = [HI[i, j] * dI[i] * dI[j] for i in 1:n, j in 1:n]
    rec["equilibrium_width_max"] = maximum(IA.diam, Z)
    rec["A_width_max"] = maximum(IA.diam, AsI)
    rec["H_width_max"] = maximum(IA.diam, HsI)
    return (; Z, X, AsI, HsI, center, record = rec)
end
