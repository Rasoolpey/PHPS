# The component template (PH_TODO.md section 4, T1 and T2; docs/active/PH_COMPONENT_TEMPLATE.md sections 4 and 8).
#
# Evaluation paths (review 99): the dense reference (`tpl_block` + `tpl_rows`) is generated from the declaration; the
# simulator's fast paths (`tpl_rows!`, `tpl_motor_rates`, `tpl_core_loops`, the hydro cache) are hand-lowered
# specialisations of the same block, sharing its entries where they can, and each is tested against the dense reference;
# they are not mechanically generated (that is the lowering step still to come, T6).
#
# A component declares its physical part once: its states, its storage H_c, its local structure (J_c, R_c, G_c over
# its own rows and the efforts of its internal ports), its source ports (B) and its domain. Everything else is
# generated from that declaration: the vector field of its rows, the flows it sends through its internal ports, its
# energy and gradient, its loss and its port powers. The interconnection (which internal port joins which row of
# another component) is data. Today's hand-written code is the reference the generated forms are checked against
# (`test/unit/ph_template.jl`); the direction is reversed only when every component passes (T5).

"""
    PHTemplateComponent

A component in the template. It implements:

- `tpl_states(c) -> Vector{String}`: the names of its physical states (local order);
- `tpl_energy(c, x) -> H`: its storage, generic in the number type (the gradient is ForwardDiff's unless
  `tpl_gradient` is given);
- `tpl_block(c, x, ext) -> (; J, R, G, v, ports, sources)`: its structure over the local rows `[x; ext]`, with `ext`
  the efforts at its internal ports (in the order of `ports`), `J` skew, `R` symmetric PSD (admissibility conditions
  checked by the component), `G` the source columns with inputs `v` (names `sources`);
- `tpl_domain(c, x)`: throws a `DomainError` outside the admissible states.
"""
abstract type PHTemplateComponent end

tpl_gradient(c::PHTemplateComponent, x) = tpl_gradient_ad(c, x)
"""`tpl_gradient_ad(c, x)`: the storage's gradient by automatic differentiation (the reference; a component's own
`tpl_gradient` is checked against it)."""
tpl_gradient_ad(c::PHTemplateComponent, x) = ForwardDiff.gradient(q -> tpl_energy(c, q), x)

# optional entries (review 97): their defaults
"""`tpl_classes(c)`: each state's class (`:S` physical storage, `:shared` a state shared through the interconnection,
e.g. the frame clock); default all `:S`."""
tpl_classes(c::PHTemplateComponent) = fill(:S, length(tpl_states(c)))
"""`tpl_step(c)`: the step rule of its storage (`:midpoint` for quadratic storage, `:discrete_gradient` otherwise)."""
tpl_step(::PHTemplateComponent) = :midpoint
"""`tpl_rate(c, x, dx)`: the storage's rate along any direction `dx`, z' dx (for the ledger along a step)."""
tpl_rate(c::PHTemplateComponent, x, dx) = dot(tpl_gradient(c, x), dx)
# the number type of an evaluation: matrices follow the states, efforts and modulations (ForwardDiff through them)
_tpl_T(args...) = promote_type(map(a -> a isa AbstractArray ? eltype(a) : typeof(a), args)...)

"""
    tpl_rows(c, x, ext) -> (; dx, flows, z, blk)

The generated vector field of the component's rows and the flows it sends through its internal ports:
`[dx; flows] = (J - R) [z; ext] + G v`, `z` the gradient of its storage.
"""
function tpl_rows(c::PHTemplateComponent, x, ext)
    tpl_domain(c, x)
    blk = tpl_block(c, x, ext)
    z = tpl_gradient(c, x)
    w = (blk.J .- blk.R) * vcat(z, ext) .+ blk.G * blk.v
    n = length(x)
    return (; dx = w[1:n], flows = w[n+1:end], z, blk)
end

"""
    tpl_port_powers(c, x, ext) -> Dict{String,Any}

The named port powers: each source's `(G' e) v` and each internal port's received power `-effort' flow`.
"""
function tpl_port_powers(c::PHTemplateComponent, x, ext)
    r = tpl_rows(c, x, ext)
    e = vcat(r.z, ext)
    out = Dict{String,Any}()
    for (k, nm) in enumerate(r.blk.sources)
        out[nm] = (r.blk.G[:, k]' * e) * r.blk.v[k]
    end
    # each internal port's coordinates (`port_dims`, default 1 each; complex pairs 2)
    dims = hasproperty(r.blk, :port_dims) ? r.blk.port_dims : ones(Int, length(r.blk.ports))
    o = 0
    for (k, nm) in enumerate(r.blk.ports)
        rg = o+1:o+dims[k]
        out[String(nm)] = -dot(ext[rg], r.flows[rg])
        o += dims[k]
    end
    return out
end

"""
    tpl_account(c, x, ext) -> (; energy, loss, supply, port_power)

The generated energy account: the storage, the loss `[z; ext]' R [z; ext]`, the sources' power `(G' [z; ext])' v`, and
the power received through the internal ports, `port_power = -ext' flows`, so that H' = supply - loss + port_power
(for the hydro unit `port_power` is minus the shaft power it gives the rotor).
"""
function tpl_account(c::PHTemplateComponent, x, ext)
    r = tpl_rows(c, x, ext)
    e = vcat(r.z, ext)
    return (; energy = tpl_energy(c, x), loss = dot(e, r.blk.R * e), supply = dot(r.blk.G' * e, r.blk.v),
            port_power = -dot(ext, r.flows))
end

# ------------------------------------------------------------------------------------------------
# T2 pilot: the hydro unit's physical subsystem (penstock, ideal turbine, reservoir), elastic or inelastic, with
# `units` identical units in parallel; the gate is the governor's modulation (a signal), the rotor its internal port.

"""
    HydroPhysicalTemplate(g::PHHydroUnit, units, at)

The hydro unit's physical part as a template component: states the penstock's flows (and, elastic, the heads at its
nodes), storage `units GMAX (sum (Tw/N) q_k^2/2 + sum C_k (h_k - 1)^2/2)` (inelastic `units GMAX Tw q^2/2`), the ladder
in `J`, the friction in `R`, the ideal turbine as a skew coupling of the turbine head (the flow, inelastic) with the
rotor (internal port `:rotor`, effort omega), the reservoir's head (1 pu) as the source. `at` is the gate (the
governor's state, clamped within its limits).
"""
struct HydroPhysicalTemplate{T<:Real} <: PHTemplateComponent
    g::PHHydroUnit
    units::Float64
    at::T             # the gate as the caller applies it: the rows take the governor's recorded clamp decision,
                      # the outputs the live clamp (review 97, Codex 6); this component does not clamp
    function HydroPhysicalTemplate(g::PHHydroUnit, units::Real, at::T) where {T<:Real}
        (isfinite(units) && units > 0) || throw(ArgumentError("HydroPhysicalTemplate: units must be positive and finite"))
        return new{T}(g, Float64(units), at)
    end
end

tpl_states(c::HydroPhysicalTemplate) = c.g.N == 0 ? ["q"] :
    vcat(["q$k" for k in 1:c.g.N], ["h$k" for k in 1:c.g.N])

function tpl_energy(c::HydroPhysicalTemplate, x)
    g = c.g; K = c.units * g.GMAX
    g.N == 0 && return K * g.Tw * x[1]^2 / 2
    N = g.N
    return K * (sum((g.Tw / N) * x[k]^2 / 2 for k in 1:N) + sum(_penstock_C(g, k) * (x[N+k] - 1)^2 / 2 for k in 1:N))
end

function tpl_domain(c::HydroPhysicalTemplate, x)
    g = c.g
    at = c.at
    if g.N == 0
        _hydro_head(g, at, x[1])                 # forward flow, an open gate with flow
    else
        _turbine_flow(g, at, x[2g.N])            # a positive head at an open gate
        all(k -> x[k] >= 0, 1:g.N) || throw(DomainError(minimum(x[1:g.N]), "HydroPhysicalTemplate: reverse flow"))
    end
    return nothing
end

function tpl_block(c::HydroPhysicalTemplate, x, ext)
    g = c.g; K = c.units * g.GMAX
    om = ext[1]
    _deep(om) > 0 || throw(DomainError(_deep(om), "HydroPhysicalTemplate: the rotor speed must be positive"))
    at = c.at
    T = _tpl_T(x, ext, at)
    if g.N == 0
        n = 1; ip = 2
        J = zeros(T, 2, 2); R = zeros(T, 2, 2); G = zeros(T, 2, 1)
        h = _hydro_head(g, at, x[1])
        b = h / (g.Tw * om)                      # q' gets -h/Tw, the rotor gets K h q / omega
        J[1, ip] -= b; J[ip, 1] += b
        G[1, 1] = 1 / g.Tw
        return (; J, R, G, v = [1.0], ports = [:rotor], sources = ["reservoir"])
    end
    N = g.N; M = g.Tw / N; r = g.fp / N
    n = 2N; ip = n + 1
    iq(k) = k; ih(k) = N + k
    J = zeros(T, n + 1, n + 1); R = zeros(T, n + 1, n + 1); G = zeros(T, n + 1, 1)
    for k in 1:N
        a = 1 / (K * M * _penstock_C(g, k))      # q_k' gets -(h_k - 1)/M, h_k' gets q_k / C_k
        J[iq(k), ih(k)] -= a; J[ih(k), iq(k)] += a
        if k < N
            J[ih(k), iq(k + 1)] -= a; J[iq(k + 1), ih(k)] += a
        end
        R[iq(k), iq(k)] += r * abs(x[iq(k)]) / (K * M^2)
    end
    hN = x[ih(N)]; qt = _turbine_flow(g, at, hN)
    CN = _penstock_C(g, N)
    a = qt / (CN * om)                           # h_N' gets -q_t/C_N, the rotor gets q_t (h_N - 1) K / omega
    J[ih(N), ip] -= a; J[ip, ih(N)] += a
    G[ip, 1] = K * qt / om                       # the reservoir's head at the turbine: K q_t / omega on the rotor
    # the reservoir's head 1 at the penstock's inlet is the datum of the shifted storage (h_k - 1): no other column
    return (; J, R, G, v = [1.0], ports = [:rotor], sources = ["reservoir"])
end

# ------------------------------------------------------------------------------------------------
# T3: a saturating core (nonlinear storage by a constitutive solve, K3; the frame clock as a shared state).

"""
    CoreTemplate(q::PHCore, omega_b, a)

A cored transformer branch (`transformer.jl`) as a template component: states the HV and source loop fluxes
(x_H, x_S, complex pairs) and the frame clock theta (a shared state, the system's one clock: the interconnection
identifies it across cores). Storage: the branch energy at the core flux minimising it, `core_energy / omega_b`
(the K3 constitutive solve `core_solve`, its derivative by the implicit function). Structure: J couples the HV loop with
the from-node's voltage over the ratio `a` and the source loop with the to-node's voltage (internal ports
`:v_from`, `:v_to`, complex pairs); R = diag(R2, R2, R1, R1, 0); the frame rotation is a source port of zero power
(G: -j omega_b x on the loops, omega_b on the clock).
"""
struct CoreTemplate <: PHTemplateComponent
    q::PHCore
    wb::Float64
    a::ComplexF64
end

tpl_states(::CoreTemplate) = ["xH_re", "xH_im", "xS_re", "xS_im", "clock"]
tpl_classes(::CoreTemplate) = [:S, :S, :S, :S, :shared]
tpl_step(::CoreTemplate) = :discrete_gradient    # at the rotated end points with shared currents (transformer.jl)
# the gradient explicitly from the constitutive solve (K3: the minimiser's currents and the energy's dependence on the
# clock), not by nested dual numbers, which do not give the solve's mixed derivatives (core_solve's docstring)
function tpl_gradient(c::CoreTemplate, x)
    xH, xS = complex(x[1], x[2]), complex(x[3], x[4])
    r = core_solve(c.q, c.wb * xH, c.wb * xS, x[5])
    return [real(r.iH), imag(r.iH), -real(r.iS), -imag(r.iS), core_rotation_rate(r.iH, r.iS, xH, xS)]
end
tpl_energy(c::CoreTemplate, x) =
    core_energy(c.q, c.wb * complex(x[1], x[2]), c.wb * complex(x[3], x[4]), x[5]) / c.wb
tpl_domain(::CoreTemplate, x) = nothing

function tpl_block(c::CoreTemplate, x, ext)
    q, wb = c.q, c.wb
    n = 5; ne = 4; N = n + ne
    T = _tpl_T(x, ext)
    J = zeros(T, N, N); R = zeros(T, N, N); G = zeros(T, N, 1)
    R[1, 1] = q.R2; R[2, 2] = q.R2; R[3, 3] = q.R1; R[4, 4] = q.R1
    # the HV loop driven by v_from / a (attach!'s coefficient 1/a), the source loop by v_to (coefficient 1): a complex
    # coefficient as a real 2x2 block, rows (r, r+1) driven by coef * v, the node receiving -conj(coef) * effort
    function cblk!(r, col, coef)
        J[r, col] += real(coef); J[r, col+1] -= imag(coef); J[r+1, col] += imag(coef); J[r+1, col+1] += real(coef)
    end
    for (r, col, coef) in ((1, n + 1, 1 / c.a), (3, n + 3, 1.0 + 0im))
        cblk!(r, col, coef)
        # the node's row receives -conj(coef) * the loop's effort (skew)
        cc = -conj(coef)
        J[col, r] += real(cc); J[col, r+1] -= imag(cc); J[col+1, r] += imag(cc); J[col+1, r+1] += real(cc)
    end
    gH, gS = -im * wb * complex(x[1], x[2]), -im * wb * complex(x[3], x[4])
    G[1, 1] = real(gH); G[2, 1] = imag(gH); G[3, 1] = real(gS); G[4, 1] = imag(gS); G[5, 1] = wb
    return (; J, R, G, v = [1.0], ports = [:v_from, :v_to], port_dims = [2, 2], sources = ["frame"])
end

# ------------------------------------------------------------------------------------------------
# T3: a reduced inductor-only node (K2), derived from its unreduced inductors by a generic constrained-coordinate
# reduction (van der Schaft 2013, section 4.1.1), independently of `ph_system`'s formulas.

"""
    CutTemplate(L, R, LL, RL, gamma, c, omega_b; stationary = false)

The branches `L`, `R` meeting at a node where only inductors meet, with the node's own inductor `LL`, `RL` (a series
R-L load), Kirchhoff's weights `gamma` (the node's current i_L = gamma' i) and the far-node coefficients `c`. The K2
reduction is declared, not hand-written: with the admissible currents i_full = T i, T = [I; gamma'], the reduced
inductance and resistance are `M = T' diag(L, LL) T` and `Rm = T' diag(R, RL) T` (`tpl_reduce`; T = [I; -gamma'], the
node receiving -gamma_j i_j as in `ph_system`, so the node's inductor carries i_L = -gamma' i), the states the
reduced fluxes Phi = M i / omega_b (complex pairs, branch by branch), the storage omega_b Phi' inv(M) Phi / 2; J the
frame's -j M (none in the stationary frame), R = Rm, the far nodes' voltages the internal ports (coefficient c).
The eliminated quantities are reconstructed by `tpl_reconstruct`: the node's current i_L = -gamma' i.
"""
struct CutTemplate <: PHTemplateComponent
    L::Vector{Float64}
    R::Vector{Float64}
    LL::Float64
    RL::Float64
    gamma::Vector{Float64}
    c::Vector{Float64}
    wb::Float64
    stationary::Bool
end
CutTemplate(L, R, LL, RL, gamma, c, wb; stationary = false) = CutTemplate(L, R, LL, RL, gamma, c, wb, stationary)

"""`tpl_reduce(c::CutTemplate) -> (; M, Rm, T)`: the K2 reduction by the admissible-current basis T = [I; -gamma']."""
function tpl_reduce(c::CutTemplate)
    n = length(c.L)
    T = vcat(Matrix(1.0I, n, n), -permutedims(c.gamma))
    return (; M = T' * Diagonal(vcat(c.L, c.LL)) * T, Rm = T' * Diagonal(vcat(c.R, c.RL)) * T, T)
end
tpl_states(c::CutTemplate) = vcat([["Phi$(k)_re", "Phi$(k)_im"] for k in eachindex(c.L)]...)
tpl_domain(::CutTemplate, x) = nothing
function tpl_energy(c::CutTemplate, x)
    Mi = inv(tpl_reduce(c).M); n = length(c.L)
    return c.wb * sum(Mi[a, b] * (x[2a-1] * x[2b-1] + x[2a] * x[2b]) for a in 1:n, b in 1:n) / 2
end
"""`tpl_reconstruct(c, x) -> (; i, iL)`: the branch currents and the eliminated node current."""
function tpl_reconstruct(c::CutTemplate, x)
    n = length(c.L); Mi = inv(tpl_reduce(c).M)
    ph = [complex(x[2a-1], x[2a]) for a in 1:n]
    i = c.wb .* (Mi * ph)
    return (; i, iL = -sum(c.gamma .* i))
end
function tpl_block(c::CutTemplate, x, ext)
    n = length(c.L); N = 2n + 2n
    red = tpl_reduce(c)
    T = _tpl_T(x, ext)
    J = zeros(T, N, N); R = zeros(T, N, N); G = zeros(T, N, 0)
    for a in 1:n, b in 1:n
        if !c.stationary                          # -j M on the efforts (the currents): the frame's rotation
            J[2a-1, 2b] += red.M[a, b]; J[2a, 2b-1] -= red.M[a, b]
        end
        R[2a-1, 2b-1] += red.Rm[a, b]; R[2a, 2b] += red.Rm[a, b]
    end
    for a in 1:n                                  # the far node's voltage drives branch a with c_a (real)
        r, col = 2a - 1, 2n + 2a - 1
        J[r, col] += c.c[a]; J[r+1, col+1] += c.c[a]
        J[col, r] -= c.c[a]; J[col+1, r+1] -= c.c[a]
    end
    return (; J, R, G, v = zeros(T, 0), ports = [Symbol("v_far$a") for a in 1:n], port_dims = fill(2, n), sources = String[])
end

# ------------------------------------------------------------------------------------------------
# T2, second half: a signal law in the template (the governor's execution). A signal component declares its states
# (no physical storage, not counted in H), its law as an ODE (`:ode`) or a sampled map (`:sampled`), the limiter
# decisions it records through the branch primitives (in the hand-written kernels' order, so that the midpoint
# rule's held decisions replay), and the modulations it hands to physical components.

"""
    PHSignalComponent

A signal law: `sig_states(c)`, `sig_execution(c)` (`:ode` or `:sampled`), and
`sig_rows!(dx, c, x, inputs, rec) -> modulations` (the rates of its states and the modulations it sets, every
state-dependent decision recorded by `rec`).
"""
abstract type PHSignalComponent end

"""
    HydroGovernorSignal(g::PHHydroUnit)

IEEEG3's governor as a template signal law: states (xp, xr, at), an ODE with the kernel's limiter decisions (pilot
valve and gate rate limits with non-windup and out-of-band relaxation, gate position limits); inputs (omega, Pref);
modulation the gate within its limits as decided (`at_lim`), which the physical part takes for its rows.
"""
struct HydroGovernorSignal <: PHSignalComponent
    g::PHHydroUnit
end
sig_states(::HydroGovernorSignal) = ["xp", "xr", "at"]
sig_execution(::HydroGovernorSignal) = :ode
function sig_rows!(dx, c::HydroGovernorSignal, x, inputs, rec::ModeRecorder)
    p = c.g.p
    omega, Pref = inputs
    GMIN, GMAX = p.PMIN * p.R_base, p.PMAX * p.R_base
    RUP, RDN = p.UO * p.R_base, p.UC * p.R_base
    at_lim = clamp_mode(rec, x[3], GMIN, GMAX)
    e = Pref - p.R_base * (omega - 1.0) - p.Sigma * at_lim - p.Delta * (at_lim - x[2])
    xp_lim = clamp_mode(rec, x[1], RDN, RUP)
    dxp = (e / p.Tg - xp_lim) / p.Tp
    dxp = nonwindup(rec, xp_lim, dxp, RDN, RUP)
    dxp = outband_relax(rec, dxp, x[1], RDN, RUP, p.Tp)
    dx[1] = dxp
    dx[2] = (at_lim - x[2]) / p.Tr
    gate_rate = nonwindup(rec, at_lim, xp_lim, GMIN, GMAX)
    gate_rate = outband_relax(rec, gate_rate, x[3], GMIN, GMAX, p.Tg)
    dx[3] = gate_rate
    return (; at = at_lim)
end

"""
    tpl_unit_rows(gov::HydroGovernorSignal, units, x, omega, Pref, rec) -> dx

The hydro unit's rows generated from its two declarations: the governor's signal law (recording its decisions) and
the physical part at the gate the governor decided.
"""
function tpl_unit_rows(gov::HydroGovernorSignal, units, x, omega, Pref, rec::ModeRecorder)
    T = _tpl_T(x, omega, Pref)
    dx = zeros(T, length(x))
    m = sig_rows!(view(dx, 1:3), gov, view(x, 1:3), (omega, Pref), rec)
    phys = HydroPhysicalTemplate(gov.g, units, m.at)
    r = tpl_rows(phys, x[4:end], [omega])
    dx[4:end] .= r.dx
    return dx
end

# ------------------------------------------------------------------------------------------------
# T4: specialised evaluation. The dense reference path (tpl_block + ForwardDiff) is kept for checking; a component may
# add an analytic gradient and a sparse structure, and `tpl_rows!` evaluates into buffers.

# the hydro unit's analytic gradient (the shifted quadratic storage)
function tpl_gradient(c::HydroPhysicalTemplate, x)
    g = c.g; K = c.units * g.GMAX
    g.N == 0 && return [K * g.Tw * x[1]]
    N = g.N; M = g.Tw / N
    return vcat([K * M * x[k] for k in 1:N], [K * _penstock_C(g, N == 0 ? 1 : k) * (x[N+k] - 1) for k in 1:N])
end

"""
    tpl_sparse(c::HydroPhysicalTemplate, x, ext) -> (; A, G, v)

The hydro block as one sparse matrix A = J - R over [x; ext] (triplets), for the specialised evaluation.
"""
function tpl_sparse(c::HydroPhysicalTemplate, x, ext)
    g = c.g; K = c.units * g.GMAX; om = ext[1]; at = c.at
    _deep(om) > 0 || throw(DomainError(_deep(om), "HydroPhysicalTemplate: the rotor speed must be positive"))
    T = _tpl_T(x, ext, at)
    if g.N == 0
        h = _hydro_head(g, at, x[1]); b = h / (g.Tw * om)
        A = sparse([1, 2], [2, 1], T[-b, b], 2, 2)
        return (; A, G = sparse([1], [1], T[1 / g.Tw], 2, 1), v = T[1])
    end
    N = g.N; M = g.Tw / N; r = g.fp / N; n = 2N; ip = n + 1
    I_ = Int[]; J_ = Int[]; V = T[]
    put!(i, j, v) = (push!(I_, i); push!(J_, j); push!(V, v))
    for k in 1:N
        a = 1 / (K * M * _penstock_C(g, k))
        put!(k, N + k, -a); put!(N + k, k, a)
        if k < N
            put!(N + k, k + 1, -a); put!(k + 1, N + k, a)
        end
        put!(k, k, -r * abs(x[k]) / (K * M^2))
    end
    hN = x[2N]; qt = _turbine_flow(g, at, hN); CN = _penstock_C(g, N)
    a = qt / (CN * om)
    put!(2N, ip, -a); put!(ip, 2N, a)
    return (; A = sparse(I_, J_, V, n + 1, n + 1), G = sparse([ip], [1], T[K * qt / om], n + 1, 1), v = T[1])
end

"""
    tpl_rows!(dx, c::HydroPhysicalTemplate, x, ext) -> flows

The specialised evaluation: the analytic gradient and the sparse block (`tpl_sparse`), the rows into `dx`.
"""
function tpl_rows!(dx, c::HydroPhysicalTemplate, x, ext)
    tpl_domain(c, x)
    sp = tpl_sparse(c, x, ext)
    e = vcat(tpl_gradient(c, x), ext)
    w = sp.A * e .+ sp.G * sp.v
    n = length(x)
    dx .= view(w, 1:n)
    return w[n+1:end]
end

# an evaluation cache: the sparsity pattern built once, the entries written in place each call (no allocation)
struct TplCache
    A::SparseMatrixCSC{Float64,Int}
    pos::Vector{Int}          # triplet k -> its place in A.nzval
    V::Vector{Float64}
    e::Vector{Float64}
    w::Vector{Float64}
end

# the hydro unit's entries, in a fixed order (elastic: per section the ladder pair, the next pair, the friction; then
# the turbine pair; inelastic: the turbine pair), written into V; returns the turbine's G entry
function _hydro_entries!(V, c::HydroPhysicalTemplate, x, om)
    g = c.g; K = c.units * g.GMAX; at = c.at
    if g.N == 0
        b = _hydro_head(g, at, x[1]) / (g.Tw * om)
        V[1] = -b; V[2] = b
        return 1 / g.Tw
    end
    N = g.N; M = g.Tw / N; r = g.fp / N; t = 0
    for k in 1:N
        a = 1 / (K * M * _penstock_C(g, k))
        V[t+=1] = -a; V[t+=1] = a
        if k < N
            V[t+=1] = -a; V[t+=1] = a
        end
        V[t+=1] = -r * abs(x[k]) / (K * M^2)
    end
    qt = _turbine_flow(g, at, x[2N]); a = qt / (_penstock_C(g, N) * om)
    V[t+=1] = -a; V[t+=1] = a
    return K * qt / om
end

"""`TplCache(c::HydroPhysicalTemplate)`: the pattern of the hydro unit's A = J - R."""
function TplCache(c::HydroPhysicalTemplate)
    g = c.g
    I_ = Int[]; J_ = Int[]
    if g.N == 0
        append!(I_, (1, 2)); append!(J_, (2, 1)); n = 2
    else
        N = g.N; ip = 2N + 1; n = ip
        for k in 1:N
            push!(I_, k, N + k); push!(J_, N + k, k)
            if k < N
                push!(I_, N + k, k + 1); push!(J_, k + 1, N + k)
            end
            push!(I_, k); push!(J_, k)
        end
        push!(I_, 2N, ip); push!(J_, ip, 2N)
    end
    A = sparse(I_, J_, ones(length(I_)), n, n)
    pos = [findfirst(p -> A.rowval[p] == I_[k], A.colptr[J_[k]]:A.colptr[J_[k]+1]-1) + A.colptr[J_[k]] - 1 for k in eachindex(I_)]
    return TplCache(A, pos, zeros(length(I_)), zeros(n), zeros(n))
end

"""
    tpl_rows!(dx, cache, c::HydroPhysicalTemplate, x, om) -> torque

The cached evaluation: the entries written in place, the analytic gradient into the effort buffer, one sparse
product; no allocation. Returns the flow at the rotor port (the torque).
"""
function tpl_rows!(dx, ch::TplCache, c::HydroPhysicalTemplate, x, om::Real)
    om > 0 || throw(DomainError(om, "HydroPhysicalTemplate: the rotor speed must be positive"))
    tpl_domain(c, x)                             # the same refusals as the reference path (review 98)
    g = c.g; K = c.units * g.GMAX
    Gt = _hydro_entries!(ch.V, c, x, om)
    nz = ch.A.nzval
    fill!(nz, 0.0)
    @inbounds for k in eachindex(ch.pos)
        nz[ch.pos[k]] += ch.V[k]
    end
    e = ch.e
    if g.N == 0
        e[1] = K * g.Tw * x[1]; e[2] = om
        mul!(ch.w, ch.A, e)
        dx[1] = ch.w[1] + Gt
        return ch.w[2]
    end
    N = g.N; M = g.Tw / N
    @inbounds for k in 1:N
        e[k] = K * M * x[k]; e[N+k] = K * _penstock_C(g, k) * (x[N+k] - 1)
    end
    e[2N+1] = om
    mul!(ch.w, ch.A, e)
    @inbounds for k in 1:2N
        dx[k] = ch.w[k]
    end
    return ch.w[2N+1] + Gt
end

# the cache of each hydro unit, built once per task (its pattern depends only on N; buffers are written per call, so a
# cache is never shared between tasks); review 98: not rebuilt per call
function _hydro_tpl_cache(g::PHHydroUnit)
    d = get!(() -> IdDict{PHHydroUnit,TplCache}(), task_local_storage(), :_hydro_tpl_cache)::IdDict{PHHydroUnit,TplCache}
    return get!(() -> TplCache(HydroPhysicalTemplate(g, 1.0, g.GMAX)), d, g)
end

# ------------------------------------------------------------------------------------------------
# T1, the interconnection as data and the generic assembler: each component is placed by mapping its own rows and its
# internal ports to global rows; the assembler adds every placed block into global J, R, G (the interconnection's
# power-conserving coupling is in the components' port blocks, which are skew across the placement).

"""`TplPlacement(comp, rows, ports)`: a component's own states at global `rows`; its internal ports (in `tpl_block`'s
order, `port_dims` coordinates each) at the global rows `ports`, whose efforts are those rows' gradient entries."""
struct TplPlacement
    comp::PHTemplateComponent
    rows::Vector{Int}
    ports::Vector{Int}
end

"""
    tpl_assemble(placements, n, x, z; inputs) -> (; J, R, G, v, sources)

The generic assembler: for each placement, its block at its own rows (x[rows]) and port efforts (z[ports]) added
into the global n x n `J`, `R`, and its source columns appended to `G` (inputs `v`, names `sources`; `inputs[k]` the
inputs of placement k where the caller sets them). A shared state's source (the frame clock of several cores) enters
once (review 99).
"""
function tpl_assemble(pl::AbstractVector{TplPlacement}, n::Int, x, z; inputs = Dict{Int,Any}())
    T = promote_type(eltype(x), eltype(z))
    J = zeros(T, n, n); R = zeros(T, n, n); Gc = Vector{Vector{T}}(); v = T[]; src = String[]
    shared = Set{Int}()                    # rows of shared states (`tpl_classes` :shared): their sources enter once
    for (pi, p) in enumerate(pl)
        blk = tpl_block(p.comp, x[p.rows], z[p.ports])
        bv = haskey(inputs, pi) ? inputs[pi] : blk.v       # the inputs the caller sets (e.g. a machine's field, shaft)
        idx = vcat(p.rows, p.ports)
        # scalar scatter-addition: a coordinate may repeat (two ports at one bus; review 101), and indexed broadcast
        # assignment would overwrite instead of accumulate
        @inbounds for (b, gb) in enumerate(idx), (a, ga) in enumerate(idx)
            J[ga, gb] += blk.J[a, b]
            R[ga, gb] += blk.R[a, b]
        end
        cls = tpl_classes(p.comp)
        for k in 1:size(blk.G, 2)
            col = zeros(T, n)
            for (a, ga) in enumerate(idx)
                col[ga] += blk.G[a, k]
            end
            for (a, r) in enumerate(p.rows)
                cls[a] === :shared || continue
                r in shared ? (col[r] -= blk.G[a, k]) : (iszero(blk.G[a, k]) || push!(shared, r))
            end
            push!(Gc, col); push!(v, bv[k]); push!(src, blk.sources[k])
        end
    end
    G = isempty(Gc) ? zeros(T, n, 0) : reduce(hcat, Gc)
    return (; J, R, G, v, sources = src)
end

# ------------------------------------------------------------------------------------------------
# T5, the DC shunt exciter (`exciter.jl`): its physical part (the field flux, the ideal armature as a gyrator between
# the rotor and the machine's field, the field resistance, the amplifier as a source) and its signal laws (the
# regulator with its non-windup limit, the stabiliser) as template declarations.

"""
    ShuntExciterTemplate(e::PHShuntExciter, xa)

The shunt exciter's physical part: state the field flux `phi_f`, storage `W_f(phi_f)` (gradient the field current
`i_f`), internal ports `:rotor` (effort omega) and `:field` (effort the machine's field current `i_fd`). The ideal
armature is the skew coupling `kappa = phi_f / (n TE)` of omega with the flux row and with the machine's field row
(`v_f = kappa omega`), and of the rotor with `-(i_f + i_fd)` (the shaft power `v_f (i_f + i_fd)` over omega); the
field resistance `r_F` is `R`; the amplifier's voltage `b clamp(xa)` is the source. `xa` is the regulator's state.
"""
struct ShuntExciterTemplate{T<:Real} <: PHTemplateComponent
    e::PHShuntExciter
    xa::T
end
tpl_states(::ShuntExciterTemplate) = ["phi_f"]
tpl_energy(c::ShuntExciterTemplate, x) = _exc_energy(c.e, x[1])
tpl_gradient(c::ShuntExciterTemplate, x) = [_exc_field_volts(c.e, x[1]) / c.e.rF]
tpl_step(c::ShuntExciterTemplate) = c.e.A == 0 ? :midpoint : :discrete_gradient
function tpl_domain(c::ShuntExciterTemplate, x)
    isfinite(_deep(x[1])) || throw(DomainError(x[1], "ShuntExciterTemplate: the field flux must be finite"))
    return nothing
end
# the block's entries: the armature's coupling kappa, the field resistance, the amplifier's voltage
@inline _shunt_entries(c::ShuntExciterTemplate, phi) = (phi / (c.e.n * c.e.TE), c.e.rF, _exc_amp_volts(c.e, c.xa))
function tpl_block(c::ShuntExciterTemplate, x, ext)
    T = _tpl_T(x, ext, c.xa)
    kap, rF, va = _shunt_entries(c, x[1])
    J = zeros(T, 3, 3)                          # local rows [phi_f; omega; i_fd]
    J[1, 2] = kap; J[2, 1] = -kap
    J[3, 2] = kap; J[2, 3] = -kap
    R = zeros(T, 3, 3); R[1, 1] = rF
    G = zeros(T, 3, 1); G[1, 1] = 1
    return (; J, R, G, v = T[va], ports = [:rotor, :field], sources = ["amplifier"])
end
"""`tpl_rows!(dx, c::ShuntExciterTemplate, phi, omega)`: the flux row from the same entries, no allocation."""
@inline function tpl_rows!(dx, c::ShuntExciterTemplate, phi::Float64, omega::Float64)
    isfinite(phi) || throw(DomainError(phi, "ShuntExciterTemplate: the field flux must be finite"))
    kap, rF, va = _shunt_entries(c, phi)
    dx[1] = kap * omega - rF * (_exc_field_volts(c.e, phi) / rF) + va
    return dx
end

"""
    ShuntExciterSignal(e::PHShuntExciter)

The shunt exciter's signal laws: states (xa, xf); inputs (vterm, Vref, Efd); the regulator `KA`, `TA` with its
non-windup limit (recorded) fed by the error with the stabiliser's feedback, the stabiliser `KF`, `TF` fed by `Efd`;
modulation the regulator's state `xa`, from which the physical part takes the amplifier's voltage.
"""
struct ShuntExciterSignal <: PHSignalComponent
    e::PHShuntExciter
end
sig_states(::ShuntExciterSignal) = ["xa", "xf"]
sig_execution(::ShuntExciterSignal) = :ode
function sig_rows!(dx, c::ShuntExciterSignal, x, inputs, rec::ModeRecorder)
    e = c.e
    vterm, Vref, efd = inputs
    xa, xf = x[1], x[2]
    verr = Vref - vterm - e.KF * (efd - xf) / e.TF
    dx[1] = nonwindup(rec, xa, (e.KA * verr - xa) / e.TA, e.VRMIN, e.VRMAX)
    dx[2] = (efd - xf) / e.TF
    return (; xa)
end

"""
    tpl_unit_rows!(dx, sig::ShuntExciterSignal, x, vterm, Vref, omega, ifd, rec) -> dx

The shunt exciter's rows `[phi_f', xa', xf']` from its two declarations: the signal laws (recording the limiter's
decisions) and the physical part at the regulator state they hand over. Float64 takes the entry path; other number
types the dense reference.
"""
function tpl_unit_rows!(dx, sig::ShuntExciterSignal, x, vterm, Vref, omega, ifd, rec::ModeRecorder)
    e = sig.e
    phi = x[1]
    efd = omega * _exc_x(e, phi)
    m = sig_rows!(view(dx, 2:3), sig, view(x, 2:3), (vterm, Vref, efd), rec)
    phys = ShuntExciterTemplate(e, m.xa)
    if phi isa Float64 && omega isa Float64 && m.xa isa Float64
        tpl_rows!(view(dx, 1:1), phys, phi, omega)
    else
        dx[1] = tpl_rows(phys, [phi], [omega, ifd]).dx[1]
    end
    return dx
end

# ------------------------------------------------------------------------------------------------
# T5, the converter's physical DC side (`converter.jl`): the DC link and the battery's RC branches, the battery in its
# Norton form, the bridge as the modulated coupling with the AC side (averaged, or the switching bridge over a step:
# m-bar, its drops Rac, its switching sink Gsw; `bridge.jl`).

"""
    DCSideTemplate(q::PHConverterDC, omega_b, m; Gsw = 0, Rac = zeros(2, 2))

The DC side as a template component: states the link voltage and the battery's branch voltages, storage
`(c v^2 + sum cb_k v_k^2) / (2 omega_b)`; internal port `:ac` (two coordinates, effort the injected current
(Re ic, Im ic), flow the bridge's AC voltage `m v - Rac ic`); `J` the modulated coupling of the link with the port;
`R` the battery's resistance `w w' / Rb` (w = omega_b ./ caps), the link's conductance and the switching sink on the
link, each branch's resistance, `Rac` on the port; source the battery's Norton current `Eb / Rb` (column `w`).
`m` is the modulation in force (the controller's, or the switching step's m-bar).

Ledger attribution (review 99): the Norton form books the battery as supply `y Eb / Rb` and loss `y^2 / Rb`
(y = w' z, the battery's terminal voltage); the physical battery's `Eb i_b` and `Rb i_b^2` differ from these by the
same amount, so the net `Eb i_b - Rb i_b^2` and every balance are equal, the attribution is not (at open circuit both
Norton terms are Eb^2 / Rb, the physical ones zero). The simulator's ledger stays the physical one
(`converter_dc_powers`); a template ledger with the physical attribution needs the battery as a resistive boundary
port (feedthrough S = 1/Rb with its cross terms), not yet declared.
"""
struct DCSideTemplate{T<:Real} <: PHTemplateComponent
    q::PHConverterDC
    wb::Float64
    m::Complex{T}
    Gsw::T
    Rac::Matrix{T}
end
DCSideTemplate(q::PHConverterDC, wb, m; Gsw = zero(real(m)), Rac = zeros(typeof(real(m)), 2, 2)) =
    (T = promote_type(typeof(real(m)), typeof(Gsw), eltype(Rac));
     DCSideTemplate{T}(q, Float64(wb), Complex{T}(m), T(Gsw), Matrix{T}(Rac)))
tpl_states(c::DCSideTemplate) = vcat("v_dc", ["v_b$k" for k in 1:nbranches(c.q)])
_dc_caps(q::PHConverterDC) = vcat(q.c, q.cb)
tpl_energy(c::DCSideTemplate, x) = sum(_dc_caps(c.q) .* x .^ 2) / (2 * c.wb)
tpl_gradient(c::DCSideTemplate, x) = _dc_caps(c.q) .* x ./ c.wb
function tpl_domain(c::DCSideTemplate, x)
    _deep(x[1]) > 0 || throw(DomainError(_deep(x[1]), "DCSideTemplate: the DC-link voltage left v_dc > 0"))
    return nothing
end
function tpl_block(c::DCSideTemplate, x, ext)
    q = c.q; n = length(x); T = _tpl_T(x, ext, real(c.m), c.Gsw, c.Rac)
    wv = c.wb ./ _dc_caps(q)
    J = zeros(T, n + 2, n + 2); R = zeros(T, n + 2, n + 2)
    J[1, n+1] = -wv[1] * real(c.m); J[n+1, 1] = wv[1] * real(c.m)
    J[1, n+2] = -wv[1] * imag(c.m); J[n+2, 1] = wv[1] * imag(c.m)
    R[1:n, 1:n] .= wv * wv' ./ q.Rb
    R[1, 1] += (q.G + c.Gsw) * wv[1]^2
    for k in 1:nbranches(q)
        R[1+k, 1+k] += wv[1+k]^2 / q.R[k]
    end
    R[n+1:n+2, n+1:n+2] .= c.Rac
    G = zeros(T, n + 2, 1); G[1:n, 1] .= wv
    return (; J, R, G, v = T[q.Eb / q.Rb], ports = [:ac], port_dims = [2], sources = ["battery"])
end
"""`tpl_rows!(dx, c::DCSideTemplate, x, ic)`: the DC rows from the same entries, no allocation (Float64)."""
function tpl_rows!(dx, c::DCSideTemplate{Float64}, x, ic::ComplexF64)
    q = c.q; wb = c.wb; nb = nbranches(q)
    x[1] > 0 || throw(DomainError(x[1], "DCSideTemplate: the DC-link voltage left v_dc > 0"))
    w1 = wb / q.c
    y = w1 * (q.c * x[1] / wb)                   # sum_b w_b z_b: the battery's terminal voltage
    @inbounds for k in 1:nb
        y += (wb / q.cb[k]) * (q.cb[k] * x[1+k] / wb)
    end
    z1 = q.c * x[1] / wb
    dx[1] = -w1 * (real(c.m) * real(ic) + imag(c.m) * imag(ic)) - w1 * y / q.Rb - (q.G + c.Gsw) * w1^2 * z1 + w1 * q.Eb / q.Rb
    @inbounds for k in 1:nb
        wk = wb / q.cb[k]
        dx[1+k] = -wk * y / q.Rb - wk^2 * (q.cb[k] * x[1+k] / wb) / q.R[k] + wk * q.Eb / q.Rb
    end
    return dx
end

"""
    dc_side_template(g, xc, ic, omega_b) -> DCSideTemplate

The converter's complete DC-side declaration at its states and injected current: the modulation in force, and with the
switching bridge the step's m-bar, its sink `Gsw = p_sw / v^2` and its drops `Rac` (`bridge_structure`) on the AC port.
The DC rows do not depend on `Rac` (it acts on the port only), so `tpl_dc_rows!` omits it; the port's flow and the
account use this one.
"""
function dc_side_template(g, xc, ic, omega_b)
    q, j = g.dc, _dcidx(g)
    _sw_on(g) || return DCSideTemplate(q, omega_b, _modulation(g, xc, ic))
    r = _sw_terms(g, xc, ic)
    b = bridge_structure(g.dc.sw, r, ic, xc[j])
    return DCSideTemplate(q, omega_b, b.m; Gsw = b.Gsw, Rac = b.Rac)
end

"""
    tpl_dc_rows!(dx, g, xc, ic, omega_b) -> dx

The converter's DC rows (in its local state vector) from the DC side's declaration, at the modulation in force: the
controller's (`_modulation`), or with the switching bridge the step's m-bar and its sink `Gsw = p_sw / v^2`.
"""
function tpl_dc_rows!(dx, g, xc, ic, omega_b)
    q, j = g.dc, _dcidx(g)
    rg = j:j+nbranches(q)
    if _sw_on(g)
        r = _sw_terms(g, xc, ic)
        c = DCSideTemplate(q, omega_b, r.m; Gsw = r.psw / xc[j]^2)
    else
        c = DCSideTemplate(q, omega_b, _modulation(g, xc, ic))
    end
    if c isa DCSideTemplate{Float64} && eltype(xc) === Float64 && ic isa ComplexF64
        tpl_rows!(view(dx, rg), c, view(xc, rg), ic)
    else
        dx[rg] .= tpl_rows(c, collect(xc[rg]), [real(ic), imag(ic)]).dx
    end
    return dx
end

# ------------------------------------------------------------------------------------------------
# T5, the composite load's induction motor (`motor.jl`): a stator and two rotor cages sharing Lm, the momentum, the load
# torque; the terminal voltage its internal port.

"""
    MotorTemplate(c::MotorCircuit, omega_b; stationary = false)

The motor as a template component: states (lambda_s, lambda_1, lambda_2) as complex pairs and the momentum p, storage
`omega_b lambda^H K lambda / 2 + p^2 / (4 H)` (gradient the currents i = omega_b K lambda and omega). Internal port
`:v` (two coordinates, effort the terminal voltage, flow -i_s into the node). `J`: the terminal's identity coupling with
the stator; the frame's rotation -j L on the three flux pairs (synchronous frame: -j omega_b lambda = -j L i; none in
the stationary frame); the rotor's j omega_b lambda_k against omega (the pair that gives Te). `R`: Rs, R1, R2 on the
pairs and the load torque's `T_L(omega) / omega >= 0` on the momentum. No sources.
"""
struct MotorTemplate <: PHTemplateComponent
    c::MotorCircuit
    wb::Float64
    stationary::Bool
end
MotorTemplate(c::MotorCircuit, wb; stationary::Bool = false) = MotorTemplate(c, Float64(wb), stationary)
tpl_states(::MotorTemplate) = ["lam_s_Re", "lam_s_Im", "lam_1_Re", "lam_1_Im", "lam_2_Re", "lam_2_Im", "p"]
_motor_lams(x) = (complex(x[1], x[2]), complex(x[3], x[4]), complex(x[5], x[6]))
tpl_energy(c::MotorTemplate, x) = motor_energy(c.c, _motor_lams(x)..., x[7], c.wb)
function tpl_gradient(c::MotorTemplate, x)
    is, i1, i2 = motor_currents(c.c, _motor_lams(x)..., c.wb)
    return [real(is), imag(is), real(i1), imag(i1), real(i2), imag(i2), x[7] / (2 * c.c.H)]
end
tpl_domain(::MotorTemplate, x) = (all(q -> isfinite(_deep(q)), x) || throw(DomainError(x, "MotorTemplate: the states must be finite")); nothing)
# the load torque's resistance T_L(omega) / omega (bounded at omega = 0 for both laws)
@inline _motor_rL(c::MotorCircuit, om) = c.etrq == 2 ? c.Tm0 * abs(om) : c.Tm0 / sqrt(om^2 + c.eps^2)
function tpl_block(c::MotorTemplate, x, ext)
    m = c.c; T = _tpl_T(x, ext)
    N = 9; ip = 7; pv = 8
    J = zeros(T, N, N); R = zeros(T, N, N)
    J[1, pv] = 1; J[2, pv+1] = 1; J[pv, 1] = -1; J[pv+1, 2] = -1           # the terminal drives the stator
    if !c.stationary                                                        # the frame: -j L i on the three pairs
        for a in 1:3, b in 1:3
            ra, rb = 2a - 1, 2b - 1
            J[ra, rb+1] += m.L[a, b]; J[ra+1, rb] -= m.L[a, b]
        end
    end
    for k in 2:3                                                            # the rotor: j omega_b lambda_k against omega
        l = complex(x[2k-1], x[2k]) * im * c.wb
        J[2k-1, ip] += real(l); J[2k, ip] += imag(l)
        J[ip, 2k-1] -= real(l); J[ip, 2k] -= imag(l)
    end
    for (k, r) in enumerate((m.Rs, m.R1, m.R2))
        R[2k-1, 2k-1] = r; R[2k, 2k] = r
    end
    R[ip, ip] = _motor_rL(m, x[7] / (2 * m.H))
    return (; J, R, G = zeros(T, N, 0), v = T[], ports = [:v], port_dims = [2], sources = String[])
end
"""`tpl_motor_rates(c, ls, l1, l2, p, v)`: the motor's rates from the block's entries (no allocation), in
`motor_rates`'s return form."""
@inline function tpl_motor_rates(t::MotorTemplate, ls, l1, l2, p, v)
    m = t.c; wb = t.wb
    is, i1, i2 = motor_currents(m, ls, l1, l2, wb)
    omega = p / (2 * m.H)
    # the frame's J entries: -j (L i) on each pair (= -j omega_b lambda)
    fs = t.stationary ? zero(is) : -im * (m.L[1, 1] * is + m.L[1, 2] * i1 + m.L[1, 3] * i2)
    f1 = t.stationary ? zero(i1) : -im * (m.L[2, 1] * is + m.L[2, 2] * i1 + m.L[2, 3] * i2)
    f2 = t.stationary ? zero(i2) : -im * (m.L[3, 1] * is + m.L[3, 2] * i1 + m.L[3, 3] * i2)
    g1, g2 = im * wb * l1, im * wb * l2                                    # the rotor's columns
    Te = -(real(g1) * real(i1) + imag(g1) * imag(i1) + real(g2) * real(i2) + imag(g2) * imag(i2))
    TL = _motor_rL(m, omega) * omega
    dls = v + fs - m.Rs * is
    dl1 = f1 + g1 * omega - m.R1 * i1
    dl2 = f2 + g2 * omega - m.R2 * i2
    return dls, dl1, dl2, Te - TL, (; is, i1, i2, omega, Te, TL)
end

# ------------------------------------------------------------------------------------------------
# T5, the saturating transformer core (`transformer.jl`, the T3 declaration `CoreTemplate`): the simulator evaluates its
# loops at the efforts of the step (the discrete-gradient currents while a step is evaluated, the minimiser's otherwise:
# `_core_currents`), so the rows are generated at given efforts, `(J - R) [z; ext] + G v` with z handed in.

"""
    tpl_rows_at(c, x, z, ext) -> (; dx, flows)

The generated rows at the structure of the state `x` and the given efforts `z` (e.g. a step's discrete gradient) and
port efforts `ext`: `[dx; flows] = (J(x) - R(x)) [z; ext] + G(x) v`.
"""
function tpl_rows_at(c::PHTemplateComponent, x, z, ext)
    tpl_domain(c, x)
    blk = tpl_block(c, x, ext)
    w = (blk.J .- blk.R) * vcat(z, ext) .+ blk.G * blk.v
    n = length(x)
    return (; dx = w[1:n], flows = w[n+1:end])
end

"""
    tpl_core_loops(c::CoreTemplate, xH, xS, iH, iS, vfrom, vto) -> (dH, dS)

The core's two loop rows from its block's entries at the efforts (i_H, -i_S) and the port voltages (no allocation):
the HV loop `v_from / a - R2 i_H - j omega_b x_H`, the source loop `v_to + R1 i_S - j omega_b x_S` (the frame column).
"""
@inline function tpl_core_loops(c::CoreTemplate, xH, xS, iH, iS, vfrom, vto)
    q = c.q
    dH = vfrom / c.a - q.R2 * iH + (-im * c.wb * xH)
    dS = vto - q.R1 * (-iS) + (-im * c.wb * xS)
    return dH, dS
end

# ------------------------------------------------------------------------------------------------
# T5, the synchronous machine (`system.jl`, `PHMachine`): its windings, rotor and angle; at a reduced node also the
# branch winding the reduction adds (`PHCut`).

"""
    MachineTemplate(m::PHMachine, cut, cf; field = true, const_power = false)

The machine as a template component. States, in local order: the machine's own (lam_d, lam_q, lam_fd, lam_1d, lam_1q[,
lam_2q], p, delta), then at a reduced node (`cut` the `PHCut`, else `nothing`) the branch pair (lam_Td, lam_Tq).
Storage `omega_b (lam_d' Kd lam_d + lam_q' Kq lam_q) / 2 + p^2 / (4 H)` (no delta), gradient the winding currents,
omega and 0. Internal port `:v` (two coordinates): the terminal bus voltage (no cut) or the far bus's (cut), with the
coefficient `kappa = j e^{-j delta} conj(cf) / a` (cut: `c_1 j e^{-j delta} conj(cf)`) on the stator (cut: branch)
pair, the bus receiving `-conj(kappa) i`. `J`: the rotation `omega_b omega` of every pair (the skew pairs with p that
give the torque) and delta's `omega_b` against p. `R`: Ra (cut: the 2 x 2 Rm over the stator and branch pairs), Rfd,
R1d, R1q[, R2q]. Sources: the field voltage on lam_fd (`field = false` for a machine whose DC shunt exciter couples
instead), the shaft torque on p (coefficient 1/omega with `const_power`), the frame's `-omega_b` on delta (zero power).
`cf` is the frame phasor (1 in the synchronous frame); `vf`, `Tm` the field voltage and the shaft input (the block's
`v`; the five-argument `tpl_rows` overrides them).
"""
struct MachineTemplate <: PHTemplateComponent
    m::PHMachine
    cut::Union{Nothing,PHCut}
    cf::ComplexF64
    field::Bool
    const_power::Bool
    vf::Float64          # the sources' inputs (review 101: the generic interface needs them in the declaration)
    Tm::Float64
end
MachineTemplate(m::PHMachine, cut, cf = 1.0 + 0im; field::Bool = true, const_power::Bool = false, vf::Real = 0.0, Tm::Real = 0.0) =
    MachineTemplate(m, cut, ComplexF64(cf), field, const_power, Float64(vf), Float64(Tm))
_mt_n0(c::MachineTemplate) = phs_nstates(c.m)
_mt_ip(c::MachineTemplate) = c.m.w.nq == 2 ? 7 : 6
function tpl_states(c::MachineTemplate)
    nm = ["lam_d", "lam_q", "lam_fd", "lam_1d", "lam_1q"]
    c.m.w.nq == 2 && push!(nm, "lam_2q")
    push!(nm, "p", "delta")
    c.cut === nothing || push!(nm, "lam_Td", "lam_Tq")
    return nm
end
# the d- and q-axis flux vectors in the order of Kd, Kq
function _mt_axes(c::MachineTemplate, x)
    t = _mt_n0(c) + 1
    ld = c.cut === nothing ? [x[1], x[3], x[4]] : [x[1], x[3], x[4], x[t]]
    lq = c.m.w.nq == 2 ? [x[2], x[5], x[6]] : [x[2], x[5]]
    c.cut === nothing || push!(lq, x[t+1])
    return ld, lq
end
function tpl_energy(c::MachineTemplate, x)
    ld, lq = _mt_axes(c, x); w = c.m.w
    return w.omega_b * (dot(ld, c.m.Kd * ld) + dot(lq, c.m.Kq * lq)) / 2 + x[_mt_ip(c)]^2 / (4 * w.H)
end
function tpl_gradient(c::MachineTemplate, x)
    ld, lq = _mt_axes(c, x); w = c.m.w; wb = w.omega_b
    id, iq = wb .* (c.m.Kd * ld), wb .* (c.m.Kq * lq)
    z = zeros(eltype(id), length(x))
    z[1], z[3], z[4] = id[1], id[2], id[3]
    z[2], z[5] = iq[1], iq[2]
    w.nq == 2 && (z[6] = iq[3])
    ip = _mt_ip(c); z[ip] = x[ip] / (2 * w.H)
    if c.cut !== nothing
        t = _mt_n0(c) + 1
        z[t], z[t+1] = id[4], iq[end]
    end
    return z
end
tpl_domain(::MachineTemplate, x) = (all(q -> isfinite(_deep(q)), x) || throw(DomainError(x, "MachineTemplate: the states must be finite")); nothing)
# the port's coefficient at the angle delta
_mt_kappa(c::MachineTemplate, delta) =
    (crot = complex(cos(delta), -sin(delta)) * conj(c.cf); c.cut === nothing ? im * crot / c.m.a : c.cut.c[1] * im * crot)
function tpl_block(c::MachineTemplate, x, ext)
    m, w = c.m, c.m.w; wb = w.omega_b
    T = _tpl_T(x, ext)
    n = length(x); N = n + 2; ip = _mt_ip(c); idl = ip + 1
    J = zeros(T, N, N); R = zeros(T, N, N)
    rot!(r) = (J[r, ip] += wb * x[r+1]; J[ip, r] -= wb * x[r+1]; J[r+1, ip] -= wb * x[r]; J[ip, r+1] += wb * x[r])
    rot!(1)
    J[idl, ip] += wb; J[ip, idl] -= wb
    k = _mt_kappa(c, x[idl])
    r = c.cut === nothing ? 1 : _mt_n0(c) + 1
    c.cut === nothing || rot!(r)
    # the port: rows (r, r+1) driven by kappa v, the bus receiving -conj(kappa) i (skew)
    J[r, n+1] += real(k); J[r, n+2] -= imag(k); J[r+1, n+1] += imag(k); J[r+1, n+2] += real(k)
    cc = -conj(k)
    J[n+1, r] += real(cc); J[n+1, r+1] -= imag(cc); J[n+2, r] += imag(cc); J[n+2, r+1] += real(cc)
    if c.cut === nothing
        R[1, 1] = R[2, 2] = w.Ra
    else
        q = c.cut; t = r
        R[1, 1] = R[2, 2] = q.Rm[1, 1]; R[t, t] = R[t+1, t+1] = q.Rm[2, 2]
        R[1, t] = R[t, 1] = R[2, t+1] = R[t+1, 2] = q.Rm[1, 2]
    end
    R[3, 3], R[4, 4], R[5, 5] = w.Rfd, w.R1d, w.R1q
    w.nq == 2 && (R[6, 6] = w.R2q)
    src = String[]; cols = Vector{Vector{T}}()
    if c.field
        g = zeros(T, N); g[3] = 1; push!(cols, g); push!(src, "field")
    end
    g = zeros(T, N); g[ip] = c.const_power ? 2 * w.H / x[ip] : 1; push!(cols, g); push!(src, "shaft")
    g = zeros(T, N); g[idl] = -wb; push!(cols, g); push!(src, "frame")
    return (; J, R, G = reduce(hcat, cols), v = T.(tpl_block_inputs(c, c.vf, c.Tm)), ports = [:v], port_dims = [2], sources = src)
end
"""`tpl_block_inputs(c, vf, Tm)`: the inputs of the sources in `tpl_block`'s order (field voltage, shaft torque or
power, the frame's 1)."""
tpl_block_inputs(c::MachineTemplate, vf, Tm) = c.field ? [vf, Tm, one(Tm)] : [Tm, one(Tm)]
# with the inputs: the generated rows (the block's `v` is set by the caller's inputs)
function tpl_rows(c::MachineTemplate, x, ext, vf, Tm)
    tpl_domain(c, x)
    blk = tpl_block(c, x, ext)
    z = tpl_gradient(c, x)
    w = (blk.J .- blk.R) * vcat(z, ext) .+ blk.G * tpl_block_inputs(c, vf, Tm)
    n = length(x)
    return (; dx = w[1:n], flows = w[n+1:end], z, blk)
end

"""
    tpl_machine_rows!(dx, s, m, x, c, v, vf, Tm, const_power)

The machine's rows into the global `dx` from the block's entries (no allocation): `c` its currents (`_machine_currents`,
with the frame's rotation in `c.rot`), `v` the port's bus voltage (terminal, or the far bus at a reduced node), `vf` the
field voltage, `Tm` the shaft input.
"""
@inline function tpl_machine_rows!(dx, s, m::PHMachine, x, c, v, vf, Tm, const_power::Bool)
    w = m.w; o = m.offset; wb = w.omega_b; om = c.omega
    k = m.cut == 0 ? im * conj(c.rot) / m.a : s.cuts[m.cut].c[1] * im * conj(c.rot)   # the port's coefficient
    kv = k * v
    if m.cut == 0
        dx[o] = real(kv) - w.Ra * c.id + wb * x[o+1] * om
        dx[o+1] = imag(kv) - w.Ra * c.iq - wb * x[o] * om
    else
        q, t = s.cuts[m.cut], m.tidx
        dx[o] = -q.Rm[1, 1] * c.id - q.Rm[1, 2] * c.iTd + wb * x[o+1] * om
        dx[o+1] = -q.Rm[1, 1] * c.iq - q.Rm[1, 2] * c.iTq - wb * x[o] * om
        dx[t] = real(kv) - q.Rm[2, 1] * c.id - q.Rm[2, 2] * c.iTd + wb * x[t+1] * om
        dx[t+1] = imag(kv) - q.Rm[2, 1] * c.iq - q.Rm[2, 2] * c.iTq - wb * x[t] * om
    end
    dx[o+2] = vf - w.Rfd * c.ifd
    dx[o+3] = -w.R1d * c.i1d
    dx[o+4] = -w.R1q * c.i1q
    w.nq == 2 && (dx[o+5] = -w.R2q * c.i2q)
    ip = _ip(m)
    # the rotor: the shaft source and the skew partners of every pair's rotation (-J' entries times the currents)
    tq = -wb * x[o+1] * c.id + wb * x[o] * c.iq
    m.tidx > 0 && (tq += -wb * x[m.tidx+1] * c.iTd + wb * x[m.tidx] * c.iTq)
    dx[ip] = (const_power ? Tm / om : Tm) + tq
    dx[_idelta(m)] = wb * om - wb
    return dx
end

# ------------------------------------------------------------------------------------------------
# T5, the network (`system.jl`): the nodes (capacitance and conductance), the branches (series R-L behind an ideal
# ratio), the shunt inductors and the reduced inductor-only nodes without a machine (the T3 declaration `CutTemplate`).
# Each is a complex pair; the interconnection is the sum of the attached components' port flows at a node (`inj`).

# a complex coefficient k applied to a pair, as J entries of rows (r, r+1) against columns (col, col+1); the partner
# rows receive -conj(k) (skew)
function _tpl_cskew!(J, r, col, k)
    J[r, col] += real(k); J[r, col+1] -= imag(k); J[r+1, col] += imag(k); J[r+1, col+1] += real(k)
    cc = -conj(k)
    J[col, r] += real(cc); J[col, r+1] -= imag(cc); J[col+1, r] += imag(cc); J[col+1, r+1] += real(cc)
    return J
end
# -j X on a pair's own efforts (the frame's rotation, a skew 2 x 2 block)
_tpl_frame!(J, r, X) = (J[r, r+1] += X; J[r+1, r] -= X; J)

"""
    NodeTemplate(C, g, wb; stationary = false)

A network node: state the charge pair q (v = omega_b q / C), storage omega_b abs2(q) / (2C); `J` the frame's -j C,
`R` the conductance `g` (the bus's own, the travelling-wave lines' and the fault's, and the static loads'); no ports of
its own: the attached components' blocks put their couplings on its rows (`inj`, their received currents).
"""
struct NodeTemplate <: PHTemplateComponent
    C::Float64
    g::Float64
    wb::Float64
    stationary::Bool
end
NodeTemplate(C, g, wb; stationary::Bool = false) = NodeTemplate(C, g, wb, stationary)
tpl_states(::NodeTemplate) = ["q_re", "q_im"]
tpl_energy(c::NodeTemplate, x) = c.wb * (x[1]^2 + x[2]^2) / (2 * c.C)
tpl_gradient(c::NodeTemplate, x) = c.wb .* x ./ c.C
tpl_domain(::NodeTemplate, x) = nothing
function tpl_block(c::NodeTemplate, x, ext)
    T = _tpl_T(x, ext); J = zeros(T, 2, 2); R = zeros(T, 2, 2)
    c.stationary || _tpl_frame!(J, 1, c.C)
    R[1, 1] = R[2, 2] = c.g
    return (; J, R, G = zeros(T, 2, 0), v = T[], ports = Symbol[], port_dims = Int[], sources = String[])
end
"""`tpl_node_row(c, v, inj)`: the node's row (complex) from its block at the voltage v, plus the interconnection `inj`."""
@inline tpl_node_row(C, g, stationary::Bool, v, inj) = inj - g * v + (stationary ? zero(v) : -im * C * v)

"""
    BranchTemplate(R, L, a, wb; stationary = false)

A series R-L branch behind an ideal ratio `a` at its from end: state the flux pair (i = omega_b x / L), storage
omega_b abs2(x) / (2L); `J` the frame's -j L and the ports `:v_from` (coefficient 1/a) and `:v_to` (coefficient -1),
the nodes receiving -conj(1/a) i and +i; `R` = R.
"""
struct BranchTemplate <: PHTemplateComponent
    R::Float64
    L::Float64
    a::ComplexF64
    wb::Float64
    stationary::Bool
end
BranchTemplate(R, L, a, wb; stationary::Bool = false) = BranchTemplate(R, L, ComplexF64(a), wb, stationary)
tpl_states(::BranchTemplate) = ["phi_re", "phi_im"]
tpl_energy(c::BranchTemplate, x) = c.wb * (x[1]^2 + x[2]^2) / (2 * c.L)
tpl_gradient(c::BranchTemplate, x) = c.wb .* x ./ c.L
tpl_domain(::BranchTemplate, x) = nothing
function tpl_block(c::BranchTemplate, x, ext)
    T = _tpl_T(x, ext); J = zeros(T, 6, 6); R = zeros(T, 6, 6)
    c.stationary || _tpl_frame!(J, 1, c.L)
    _tpl_cskew!(J, 1, 3, 1 / c.a); _tpl_cskew!(J, 1, 5, -1.0 + 0im)
    R[1, 1] = R[2, 2] = c.R
    return (; J, R, G = zeros(T, 6, 0), v = T[], ports = [:v_from, :v_to], port_dims = [2, 2], sources = String[])
end
"""`tpl_branch_row(R, L, a, stationary, i, vfrom, vto)`: the branch's row from its block's entries."""
@inline tpl_branch_row(R, L, a, stationary::Bool, i, vfrom, vto) =
    vfrom / a - vto - R * i + (stationary ? zero(i) : -im * L * i)

"""
    ShuntTemplate(R, L, wb; stationary = false)

A shunt R-L branch at a node: as `BranchTemplate` with the port `:v` (coefficient 1), the node receiving -i.
"""
struct ShuntTemplate <: PHTemplateComponent
    R::Float64
    L::Float64
    wb::Float64
    stationary::Bool
end
ShuntTemplate(R, L, wb; stationary::Bool = false) = ShuntTemplate(R, L, wb, stationary)
tpl_states(::ShuntTemplate) = ["phi_re", "phi_im"]
tpl_energy(c::ShuntTemplate, x) = c.wb * (x[1]^2 + x[2]^2) / (2 * c.L)
tpl_gradient(c::ShuntTemplate, x) = c.wb .* x ./ c.L
tpl_domain(::ShuntTemplate, x) = nothing
function tpl_block(c::ShuntTemplate, x, ext)
    T = _tpl_T(x, ext); J = zeros(T, 4, 4); R = zeros(T, 4, 4)
    c.stationary || _tpl_frame!(J, 1, c.L)
    _tpl_cskew!(J, 1, 3, 1.0 + 0im)
    R[1, 1] = R[2, 2] = c.R
    return (; J, R, G = zeros(T, 4, 0), v = T[], ports = [:v], port_dims = [2], sources = String[])
end
@inline tpl_shunt_row(R, L, stationary::Bool, i, v) = v - R * i + (stationary ? zero(i) : -im * L * i)

"""`tpl_cut_row(q::PHCut, a, ib, v, stationary)`: branch a's row of a reduced node without a machine, from the
`CutTemplate` block's entries: `c_a v_far - j (M i)_a - (Rm i)_a` (the frame's -j M on the currents)."""
@inline function tpl_cut_row(q, a::Int, pos, ib, v, stationary::Bool)
    d = q.c[a] * v[q.far[a]]
    for (b, jb) in enumerate(pos)
        d -= q.Rm[a, b] * ib[jb]
        stationary || (d -= im * q.M[a, b] * ib[jb])
    end
    return d
end

# ------------------------------------------------------------------------------------------------
# T5, the ledger through the template: a component's entry in the energy account (its energy, loss, net port, rate and
# the sum of its port powers' magnitudes) generated from its declaration (`tpl_rows`, `tpl_port_powers`).

"""
    tpl_ledger_entry(c, x, ext; dxl = nothing) -> (; energy, loss, port, rate, tport)

The component's ledger entry: `energy` its storage, `loss` `[z; ext]' R [z; ext]`, `port` the sources' power plus the
power received through its internal ports, `rate` along `dxl` (its local rows' rates; default its own rows, then
`port - loss`), `tport` the sum of the port powers' magnitudes.
"""
function tpl_ledger_entry(c::PHTemplateComponent, x, ext; dxl = nothing)
    a = tpl_account(c, x, ext)
    pp = tpl_port_powers(c, x, ext)
    port = a.supply + a.port_power
    rate = dxl === nothing ? port - a.loss : tpl_rate(c, x, dxl)
    return (; energy = a.energy, loss = a.loss, port, rate, tport = sum(abs, values(pp); init = zero(port)))
end

# ------------------------------------------------------------------------------------------------
# T1, the whole model as data: every component of a `PHSystem` placed by its rows and ports, assembled generically, the
# buses without a voltage state as algebraic columns (their conductance only), eliminated as `phs_structure` does.

"""`AlgebraicNodeTemplate(g)`: a bus without a voltage state: no storage, its conductance `g` on its algebraic pair."""
struct AlgebraicNodeTemplate <: PHTemplateComponent
    g::Float64
end
tpl_states(::AlgebraicNodeTemplate) = String[]
tpl_energy(::AlgebraicNodeTemplate, x) = zero(eltype(x))
tpl_gradient(::AlgebraicNodeTemplate, x) = eltype(x)[]
tpl_domain(::AlgebraicNodeTemplate, x) = nothing
function tpl_block(c::AlgebraicNodeTemplate, x, ext)
    T = _tpl_T(x, ext); R = zeros(T, 2, 2); R[1, 1] = R[2, 2] = c.g
    return (; J = zeros(T, 2, 2), R, G = zeros(T, 2, 0), v = T[], ports = [:v], port_dims = [2], sources = String[])
end

"""
    tpl_placements(s::PHSystem, x, u) -> (; pl, inputs, n, N, wcol)

The model as data: one placement per node (state or algebraic), branch, core, reduced node, shunt inductor, machine
and motor, each at its rows and with its ports at the buses' rows (`s.qidx`) or algebraic columns (`wcol`, past the n
states), a converter's source row as an input column (its EMF). Refused: travelling-wave lines, the zero axis,
phase-asymmetric faults, the controlled model's DC sides and exciter couplings (placed by `phs_structure(::PHControlled)`
only, not yet here).
"""
function tpl_placements(s::PHSystem, x, u)
    (isempty(s.tw) && s.z0 === nothing && isempty(_fault_matrices(s))) ||
        throw(ArgumentError("tpl_placements: travelling-wave lines, the zero axis and asymmetric faults are not placed yet"))
    pn, wb, st, n = s.net, s.omega_b, s.stationary, s.n
    f = phs_flows(s, x, u)
    wcol = Dict{Int,Int}(); nw = 0
    for k in s.alg; wcol[k] = n + nw + 1; nw += 2; end
    # the converters' source rows: their voltage is the input u_c (input columns past the algebraic ones)
    ccol = Dict{Int,Int}(); nc = 0
    for cv in s.conv; ccol[cv.row] = n + nw + nc + 1; nc += 2; end
    bus(k) = (c = haskey(ccol, k) ? ccol[k] : s.qidx[k] > 0 ? s.qidx[k] : wcol[k]; [c, c + 1])
    pair(r) = [r, r + 1]
    gstat(k) = s.sl[k] == 0 ? 0.0 : f.gs[s.sl[k]]
    pl = TplPlacement[]; inputs = Dict{Int,Any}()
    for k in s.dyn
        push!(pl, TplPlacement(NodeTemplate(pn.C[k], pn.G[k] + u.gf[k] + gstat(k), wb; stationary = st), pair(s.qidx[k]), Int[]))
    end
    for k in s.alg
        push!(pl, TplPlacement(AlgebraicNodeTemplate(pn.G[k] + u.gf[k] + gstat(k)), Int[], bus(k)))
    end
    cored = Dict(q.pos => q for q in s.cores)
    for (j, l) in enumerate(s.branches)
        (s.bcut[j] == 0 && !haskey(cored, j)) || continue
        push!(pl, TplPlacement(BranchTemplate(pn.R[l], pn.L[l], pn.a[l], wb; stationary = st), pair(s.bidx[j]),
                               vcat(bus(pn.from[l]), bus(pn.to[l]))))
    end
    for q in s.cores
        l = s.branches[q.pos]
        push!(pl, TplPlacement(CoreTemplate(q, wb, ComplexF64(pn.a[l])), vcat(pair(s.bidx[q.pos]), pair(q.xs), s.iclk),
                               vcat(bus(pn.from[l]), bus(pn.to[l]))))
    end
    for c in s.cuts
        c.machine == 0 || continue
        L = [pn.L[s.branches[j]] for j in c.pos]; Rb = [pn.R[s.branches[j]] for j in c.pos]
        push!(pl, TplPlacement(CutTemplate(L, Rb, c.LL, c.RL, c.gamma, c.c, wb; stationary = st),
                               vcat([pair(s.bidx[j]) for j in c.pos]...), vcat([bus(c.far[a]) for a in eachindex(c.pos)]...)))
    end
    for (j, k) in enumerate(pn.sh_bus)
        s.sidx[j] > 0 || continue
        push!(pl, TplPlacement(ShuntTemplate(s.Rsh[j], pn.Lsh[j], wb; stationary = st), pair(s.sidx[j]), bus(k)))
    end
    for (j, m) in enumerate(s.machines)
        cut = m.cut == 0 ? nothing : s.cuts[m.cut]
        rows = collect(m.offset:m.offset+phs_nstates(m)-1)
        cut === nothing || append!(rows, pair(m.tidx))
        t = MachineTemplate(m, cut, st ? u.cf[1] : 1.0 + 0im; const_power = u.const_power, vf = u.vf[j], Tm = u.Tm[j])
        push!(pl, TplPlacement(t, rows, cut === nothing ? bus(m.node) : bus(cut.far[1])))
    end
    for mo in s.motors
        q = _motor_state(mo, x, wb)
        (q.on && q.sg == 1) || throw(ArgumentError("tpl_placements: a motor with tripped units (sigma < 1) is not placed yet"))
        push!(pl, TplPlacement(MotorTemplate(mo.c, wb; stationary = st), collect(mo.offset:mo.offset+6), bus(mo.node)))
    end
    uc = zeros(nc)
    for (i, cv) in enumerate(s.conv)
        uc[ccol[cv.row]-n-nw], uc[ccol[cv.row]-n-nw+1] = real(u.uc[i]), imag(u.uc[i])
    end
    return (; pl, inputs, n, N = n + nw + nc, nw, uc, wcol, ccol)
end

"""
    tpl_assemble_system(s::PHSystem, x, u) -> (; J, R, Gv, z)

The whole model's structure from the placements (`tpl_placements`) by the generic assembler, the algebraic voltages
eliminated (the Schur complement of `J - R` on their columns), as `phs_structure` builds it by hand: J, R over the
states, `Gv` the sources' sum G v, `z` the states' gradient from each component's declaration.
"""
function tpl_assemble_system(s::PHSystem, x, u)
    P = tpl_placements(s, x, u)
    n, N = P.n, P.N
    xe = vcat(collect(x[1:n]), zeros(N - n))
    z = zeros(N)
    for p in P.pl
        isempty(p.rows) && continue
        zl = tpl_gradient(p.comp, x[p.rows])
        cls = tpl_classes(p.comp)
        for (a, r) in enumerate(p.rows)
            # a shared state's derivative adds over the components that share it (the clock: the cores' rotation
            # rates; review 101), an own state's is its component's
            cls[a] === :shared ? (z[r] += zl[a]) : (z[r] = zl[a])
        end
    end
    A = tpl_assemble(P.pl, N, xe, z; inputs = P.inputs)
    M = A.J .- A.R
    S, W, C = 1:n, n+1:n+P.nw, n+P.nw+1:N
    Mr = P.nw > 0 ? M[S, S] .- M[S, W] * (M[W, W] \ M[W, S]) : M[S, S]
    # the sources: G v, and the converters' EMFs through their columns (after the elimination: the algebraic
    # voltages also respond to them)
    Gv = (A.G * A.v)[S]
    if !isempty(C)
        Kc = P.nw > 0 ? M[S, C] .- M[S, W] * (M[W, W] \ M[W, C]) : M[S, C]
        Gv = Gv .+ Kc * P.uc
    end
    return (; J = (Mr .- Mr') ./ 2, R = .-(Mr .+ Mr') ./ 2, Gv, z = z[S], G = A.G[S, :], v = A.v, sources = A.sources)
end
