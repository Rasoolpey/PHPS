# Physical network of the component-first programme (docs/records/early/PHYSICAL_NETWORK.md): every case
# branch as a dynamic dq pi line, every loss inside an element, and a lossless Kirchhoff
# interconnection. A separate model; the frozen RMS network (`ybus`) is not touched.
#
# Elements, in the synchronous dq frame (phasors rotate at nominal speed; t in seconds,
# per unit on the system base, X = L at nominal frequency):
#   - node k: capacitor C_k (half the charging of every incident line, capacitive shunts and
#     loads) and conductance G_k (shunt g, load P/V0^2);
#   - branch l: series R_l, L_l from bus `from` to bus `to`, with an ideal transformer of
#     complex ratio a_l = tap e^{j phi} at `from` (the Y-bus convention of `_stamp_line!`);
#   - shunt inductor s at bus b_s: inductive shunts and the inductive part Q/V0^2 of a load;
#   - an external current port i_ext at every bus (machines, converters), supply Re(conj(v) i_ext).
#
# Equations (complex):
#   (C_k/omega_b) dv_k/dt = i_ext,k - (B i)_k - G_k v_k - sum_{s at k} i_s - j C_k v_k
#   (L_l/omega_b) di_l/dt = (B^H v)_l - R_l i_l - j L_l i_l
#   (L_s/omega_b) di_s/dt = v_{b_s} - j L_s i_s
# with B[from, l] = 1/conj(a_l), B[to, l] = -1. A node with C_k = 0 is an algebraic KCL row.
#
# Energies in pu*s: H = (sum C|v|^2 + sum L|i|^2 + sum L_s|i_s|^2) / (2 omega_b).
# Loads are constant-impedance surrogates at the case voltages (the `ybus(case; loads = true)`
# stamp), not the ComplexLoad dynamics.

"""
    PhysicalNetwork

The dynamic network of a case: bus order `net`, base frequency `omega_b`; branches
(`from`, `to` as bus rows, complex ratio `a` at `from`, `R`, `L`, case index `branch`); node
capacitances `C` and conductances `G`; shunt inductors (`sh_bus` as bus rows, `Lsh`).
"""
struct PhysicalNetwork
    net::Network
    omega_b::Float64
    from::Vector{Int}
    to::Vector{Int}
    a::Vector{ComplexF64}
    R::Vector{Float64}
    L::Vector{Float64}
    branch::Vector{String}
    C::Vector{Float64}
    G::Vector{Float64}
    sh_bus::Vector{Int}
    Lsh::Vector{Float64}
end

pn_sizes(p::PhysicalNetwork) = (nbus(p.net), length(p.R), length(p.Lsh))

"""
    physical_network(case; omega_b = 2pi*60, loads = true, v0 = nothing) -> PhysicalNetwork

The dynamic network of `case`, element by element from the data `ybus(case; loads)` stamps:
each line becomes a series R-L branch with its ideal transformer, its charging `b/2` a node
capacitor at both ends; a shunt `g + jb` becomes a conductance and a capacitor (`b > 0`) or
inductor (`b < 0`); a load `(P, Q)` at voltage `V0` becomes `G = P/V0^2` and a capacitor
(`Q < 0`) or inductor (`Q > 0`) of admittance `|Q|/V0^2`, with `v0` as in `ybus`.

Throws if an element would not be passive (`r < 0`, `x <= 0`, `g < 0`, `P < 0`): such data
have no passive physical realisation in this element class.
"""
function physical_network(case::Case; omega_b::Real = 2pi * 60, loads::Bool = true,
                          v0 = nothing)
    net = Network(case)
    n = nbus(net)
    C, G = zeros(n), zeros(n)
    from, to, a, R, L, names = Int[], Int[], ComplexF64[], Float64[], Float64[], String[]
    for l in case.lines
        (haskey(net.index, l.bus1) && haskey(net.index, l.bus2)) || continue
        l.r >= 0 || throw(ArgumentError("physical_network: line $(l.idx) has r = $(l.r) < 0"))
        l.x > 0 || throw(ArgumentError("physical_network: line $(l.idx) has x = $(l.x) <= 0; " *
                                       "a series inductor needs x > 0"))
        l.b >= 0 || throw(ArgumentError("physical_network: line $(l.idx) has b = $(l.b) < 0"))
        i, j = net.index[l.bus1], net.index[l.bus2]
        push!(from, i); push!(to, j); push!(R, l.r); push!(L, l.x); push!(names, l.idx)
        push!(a, l.tap * complex(cos(l.phi), sin(l.phi)))
        C[i] += l.b / 2
        C[j] += l.b / 2
    end
    sh_bus, Lsh = Int[], Float64[]
    function shunt!(k, g, b, what)
        g >= 0 || throw(ArgumentError("physical_network: $what at bus $(net.bus_ids[k]) has " *
                                      "negative conductance $g (not passive)"))
        G[k] += g
        if b > 0
            C[k] += b
        elseif b < 0
            push!(sh_bus, k)
            push!(Lsh, -1 / b)
        end
    end
    for sh in case.shunts
        haskey(net.index, sh.bus) && shunt!(net.index[sh.bus], sh.g, sh.b, "shunt")
    end
    if loads
        for (bus, P, Q) in load_power_by_bus(case, net)
            (P == 0 && Q == 0) && continue
            V02 = max(_load_v0(case, net, bus, v0)^2, 1e-12)
            shunt!(net.index[bus], P / V02, -Q / V02, "load")
        end
    end
    return PhysicalNetwork(net, Float64(omega_b), from, to, a, R, L, names, C, G, sh_bus, Lsh)
end

"""
    pn_incidence(p) -> SparseMatrixCSC{ComplexF64}

The complex incidence `B` (buses x branches): `B[from, l] = 1/conj(a_l)`, `B[to, l] = -1`.
Node injections from the branches are `-B i`; the branch driving voltages are `B^H v`.
"""
function pn_incidence(p::PhysicalNetwork)
    n, m, _ = pn_sizes(p)
    return sparse(vcat(p.from, p.to), vcat(1:m, 1:m),
                  vcat(1.0 ./ conj.(p.a), fill(-1.0 + 0im, m)), n, m)
end

"""
    pn_flows(p, v, i, ish, iext) -> (; fv, fi, fs)

The right-hand sides of the complex equations: `fv = (C/omega_b) dv/dt`,
`fi = (L/omega_b) di/dt`, `fs = (L_s/omega_b) di_s/dt`. On a node with `C = 0`, `fv = 0` is
the algebraic KCL constraint.
"""
function pn_flows(p::PhysicalNetwork, v, i, ish, iext)
    B = pn_incidence(p)
    fv = iext .- B * i .- p.G .* v .- im .* p.C .* v
    for (s, k) in enumerate(p.sh_bus)
        fv[k] -= ish[s]
    end
    fi = B' * v .- p.R .* i .- im .* p.L .* i
    fs = v[p.sh_bus] .- im .* p.Lsh .* ish
    return (; fv, fi, fs)
end

"""
    pn_energy(p, v, i, ish) -> the network's stored energy (pu*s).
"""
pn_energy(p::PhysicalNetwork, v, i, ish) =
    (sum(p.C .* abs2.(v)) + sum(p.L .* abs2.(i)) + sum(p.Lsh .* abs2.(ish))) / (2 * p.omega_b)

"""
    pn_balance(p, v, i, ish, iext) -> NamedTuple

Element-by-element power balances of `docs/records/early/PHYSICAL_NETWORK.md` (Proposition 1), each as
`rate - (port power - loss)`, and the interconnection sum of all internal port powers
(Proposition 2), which is zero for a lossless Kirchhoff interconnection. The element rates
are `Re(conj(state) * flow)`, the time derivative of each element's energy.
`total_error` is `dH/dt - (supply - losses)`.
"""
function pn_balance(p::PhysicalNetwork, v, i, ish, iext)
    f = pn_flows(p, v, i, ish, iext)
    B = pn_incidence(p)
    Bv = B' * v
    Bi = B * i
    ish_at = zeros(ComplexF64, length(v))
    for (s, k) in enumerate(p.sh_bus)
        ish_at[k] += ish[s]
    end
    # port powers into each element from the interconnection
    p_node = real.(conj.(v) .* (-Bi .- ish_at))
    p_line = real.(conj.(i) .* Bv)
    p_sh = real.(conj.(ish) .* v[p.sh_bus])
    supply = real.(conj.(v) .* iext)
    loss_node = p.G .* abs2.(v)
    loss_line = p.R .* abs2.(i)
    node_err = real.(conj.(v) .* f.fv) .- (supply .+ p_node .- loss_node)
    line_err = real.(conj.(i) .* f.fi) .- (p_line .- loss_line)
    sh_err = real.(conj.(ish) .* f.fs) .- p_sh
    interconnection = sum(p_node) + sum(p_line) + sum(p_sh)
    rate = sum(real.(conj.(v) .* f.fv)) + sum(real.(conj.(i) .* f.fi)) +
           sum(real.(conj.(ish) .* f.fs))
    total_error = rate - (sum(supply) - sum(loss_node) - sum(loss_line))
    return (; node_err, line_err, sh_err, interconnection, rate, supply = sum(supply),
            losses = sum(loss_node) + sum(loss_line), total_error)
end

"""
    pn_descriptor(p) -> (; E, J, R, Bp, nx)

The real linear port-Hamiltonian descriptor form `E dx/dt = (J - R) x + Bp u`, `y = Bp' x`,
`H = x' E x / 2`, with `x = (v, i, i_s)` and each complex number stored as `(Re, Im)`, and
`u = i_ext`, `y = v`. `E` is diagonal and positive semidefinite (singular on nodes with
`C = 0`), `J = -J'` exactly (incidence and frame rotation), `R` diagonal and nonnegative
(`G`, branch `R`). Then `dH/dt = -x' R x + y' u` along every solution (Proposition 1).
"""
function pn_descriptor(p::PhysicalNetwork)
    n, m, ns = pn_sizes(p)
    nx = 2 * (n + m + ns)
    E, J, R = zeros(nx, nx), zeros(nx, nx), zeros(nx, nx)
    vi(k) = 2k - 1                       # Re index of node k; Im is +1
    ii(l) = 2n + 2l - 1
    si(s) = 2n + 2m + 2s - 1
    # complex coefficient c acting on a complex variable, as a 2x2 real block
    function cblock!(M, r, c_, z)
        M[r, c_] += real(z); M[r, c_+1] -= imag(z)
        M[r+1, c_] += imag(z); M[r+1, c_+1] += real(z)
    end
    w = p.omega_b
    for k in 1:n
        E[vi(k), vi(k)] = E[vi(k)+1, vi(k)+1] = p.C[k] / w
        R[vi(k), vi(k)] = R[vi(k)+1, vi(k)+1] = p.G[k]
        cblock!(J, vi(k), vi(k), -im * p.C[k])
    end
    Bc = pn_incidence(p)
    for l in 1:m
        E[ii(l), ii(l)] = E[ii(l)+1, ii(l)+1] = p.L[l] / w
        R[ii(l), ii(l)] = R[ii(l)+1, ii(l)+1] = p.R[l]
        cblock!(J, ii(l), ii(l), -im * p.L[l])
        for k in (p.from[l], p.to[l])
            b = Bc[k, l]
            cblock!(J, vi(k), ii(l), -b)        # node row: -B i
            cblock!(J, ii(l), vi(k), conj(b))   # branch row: +B^H v
        end
    end
    for (s, k) in enumerate(p.sh_bus)
        E[si(s), si(s)] = E[si(s)+1, si(s)+1] = p.Lsh[s] / w
        cblock!(J, si(s), si(s), -im * p.Lsh[s])
        cblock!(J, vi(k), si(s), -1.0 + 0im)
        cblock!(J, si(s), vi(k), 1.0 + 0im)
    end
    Bp = zeros(nx, 2n)
    for k in 1:n
        Bp[vi(k), 2k-1] = Bp[vi(k)+1, 2k] = 1.0
    end
    return (; E, J, R, Bp, nx)
end

"""
    pn_quasistatic_ybus(p) -> Matrix{ComplexF64}

The bus admittance obtained by setting every derivative to zero in the dynamic equations
(Proposition 3): branch rows give `i = (R + jL)^{-1} B^H v`, shunt rows `i_s = v/(j L_s)`,
and the node rows then give `i_ext = Y v` with
`Y = B diag(1/(R + jL)) B^H + diag(G + jC) + sum_s 1/(j L_s) e_s e_s'`.
"""
function pn_quasistatic_ybus(p::PhysicalNetwork)
    n, _, _ = pn_sizes(p)
    B = Matrix(pn_incidence(p))
    Y = B * Diagonal(1.0 ./ complex.(p.R, p.L)) * B' + Diagonal(complex.(p.G, p.C))
    for (s, k) in enumerate(p.sh_bus)
        Y[k, k] += 1 / (im * p.Lsh[s])
    end
    return Y
end

"""
    pn_node_classes(p) -> (; dynamic, algebraic, cutset)

Bus ids by node type: `dynamic` (`C > 0`), `algebraic` (`C = 0`, `G > 0`: the voltage follows
from the currents, index 1), `cutset` (`C = 0`, `G = 0`: only inductive elements meet there,
so an inductive source at the bus, such as a machine stator, makes an inductor cutset).
"""
function pn_node_classes(p::PhysicalNetwork)
    ids = p.net.bus_ids
    dynamic = ids[p.C .> 0]
    algebraic = ids[(p.C .== 0) .& (p.G .> 0)]
    cutset = ids[(p.C .== 0) .& (p.G .== 0)]
    return (; dynamic, algebraic, cutset)
end
