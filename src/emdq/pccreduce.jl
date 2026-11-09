# Structure-preserving truncation of the grid seen from the PCC, on the closed two-machine pilot
# (docs/active/PCC_REDUCTION_DERIVATION.md). The grid side of `EMDQPilot2` (variant B) in explicit
# port-Hamiltonian form with machine 2 in its rotor frame,
#     x' = (J(x) - R) Q x + g,      H = 1/2 x'Q x   (Q zero on the rotor angle),
# and its reduction x = Phi x_r with the energy's own test space W = Q_y V (V'Q_y V)^{-1}: the PCC
# pair, all of machine 2 and its angle kept, the other network pairs reduced by an L-closed V_O
# (L = multiplication by j on every pair). Every identity of the derivation's section 3 is tested
# in test/unit/emdq_pilot2.jl. The physical energy is used as it is; nothing is fitted.

"""
    GridPH

The grid side of the closed two-machine pilot in explicit pH form: the pilot `s`, its grid layout
`G`, the rotor-frame index of machine 2's block (`o`, the angle `od`), the constant physical energy
matrix `Q` (zero on the angle), the kept indices `K` (the PCC pair, machine 2's windings, momentum
and DC charge), the other network indices `O` (pairs), and the angle index.
"""
struct GridPH
    s::EMDQPilot2
    G::NamedTuple
    o::Int
    od::Int
    Q::Matrix{Float64}
    K::Vector{Int}
    O::Vector{Int}
end

"""    emdq2_grid_ph(s) -> GridPH (closed variant B only, machine 2 with two q-axis windings)."""
function emdq2_grid_ph(s::EMDQPilot2)
    s.fixed_bus && throw(ArgumentError("emdq2_grid_ph: the closed variant B only"))
    G = emdq2_grid_layout(s)
    M = s.m2.mach
    M.nq == 2 || throw(ArgumentError("emdq2_grid_ph: machine 2 needs nq = 2"))
    wb = M.omega_b
    n = G.n
    o = G.m2
    od = o + 5 + M.nq
    Q = zeros(n, n)
    for (k, c) in _gph_caps(s, G)
        Q[k, k] = Q[k+1, k+1] = c / wb
    end
    for (k, l) in _gph_inds(s, G)
        Q[k, k] = Q[k+1, k+1] = l / wb
    end
    dI, qI = [o + 1, o + 3, o + 4], [o + 2, o + 5, o + 6]
    Q[dI, dI] = M.Kd ./ wb
    Q[qI, qI] = M.Kq ./ wb
    Q[od+1, od+1] = 1 / (2 * M.H)
    Q[od+2, od+2] = 1 / s.m2.Cdc
    K = vcat([G.v2, G.v2 + 1], collect(o+1:o+6), [od + 1, od + 2])
    O = sort(vcat([[k, k + 1] for k in (G.i23, G.v3, G.iL3, G.i225, G.v25, G.iL25)]...))
    return GridPH(s, G, o, od, Q, K, O)
end

_gph_caps(s, G) = ((G.v2, s.C2), (G.v3, s.C3), (G.v25, s.C25))
_gph_inds(s, G) = ((G.i23, s.L23), (G.iL3, s.L3), (G.i225, s.L225), (G.iL25, s.L25))

"""    grid_ph_to_rotor(P, xg) -> the grid-side state with machine 2's stator flux in its rotor frame."""
function grid_ph_to_rotor(P::GridPH, xg)
    x = copy(xg)
    o, od = P.o, P.od
    c, sn = cos(xg[od]), sin(xg[od])
    x[o+1] = sn * xg[o+1] - c * xg[o+2]
    x[o+2] = c * xg[o+1] + sn * xg[o+2]
    return x
end

"""    grid_ph_from_rotor(P, x) -> the pilot's grid-side state (inverse of `grid_ph_to_rotor`)."""
function grid_ph_from_rotor(P::GridPH, x)
    xg = copy(x)
    o, od = P.o, P.od
    c, sn = cos(x[od]), sin(x[od])
    xg[o+1] = sn * x[o+1] + c * x[o+2]
    xg[o+2] = -c * x[o+1] + sn * x[o+2]
    return xg
end

"""
    grid_ph_JRg(P, x, u, gf) -> (J, R, g)

The structure at the rotor-frame grid state `x`: skew `J(x)`, `R` (with fault conductances
`gf = (g2, g3)` at bus 2 and bus 3) and the inputs `g` (the PCC current `u = (uD, uQ)` into bus 2,
the frame clock, machine 2's shaft torque and DC source).
"""
function grid_ph_JRg(P::GridPH, x, u, gf = (0.0, 0.0))
    s, G, o, od = P.s, P.G, P.o, P.od
    M, st = s.m2.mach, s.m2
    wb, t2 = M.omega_b, st.tap
    n = G.n
    T = promote_type(eltype(x), eltype(u), typeof(gf[1]))
    J, R, g = zeros(T, n, n), zeros(T, n, n), zeros(T, n)
    cpl!(r, c, z) = (J[r, c] += real(z); J[r, c+1] -= imag(z); J[r+1, c] += imag(z); J[r+1, c+1] += real(z))
    caps = Dict(_gph_caps(s, G))
    inds = Dict(_gph_inds(s, G))
    for (k, c) in caps
        cpl!(k, k, -im * wb^2 / c)
    end
    for (k, l) in inds
        cpl!(k, k, -im * wb^2 / l)
    end
    kirch!(node, br, sg) = (w = wb^2 / (caps[node] * inds[br]); cpl!(node, br, complex(-sg * w)); cpl!(br, node, complex(sg * w)))
    kirch!(G.v2, G.i23, 1); kirch!(G.v3, G.i23, -1)
    kirch!(G.v3, G.iL3, 1)
    kirch!(G.v2, G.i225, 1); kirch!(G.v25, G.i225, -1)
    kirch!(G.v25, G.iL25, 1)
    for (k, l, r) in ((G.i23, s.L23, s.R23), (G.i225, s.L225, s.R225))
        R[k, k] = R[k+1, k+1] = wb^2 * r / l^2
    end
    for (k, c, gg) in ((G.v2, s.C2, gf[1]), (G.v3, s.C3, s.G3 + gf[2]), (G.v25, s.C25, s.G25))
        R[k, k] = R[k+1, k+1] = wb^2 * gg / c^2
    end
    dl = x[od]
    a = (wb^2 / (s.C25 * t2)) * im * complex(cos(dl), -sin(dl))
    cpl!(o + 1, G.v25, a)
    cpl!(G.v25, o + 1, -conj(a))
    J[o+1, od+1] += wb * x[o+2]; J[od+1, o+1] -= wb * x[o+2]
    J[o+2, od+1] -= wb * x[o+1]; J[od+1, o+2] += wb * x[o+1]
    J[od, od+1] += wb; J[od+1, od] -= wb
    J[o+3, od+2] += wb * st.mstar; J[od+2, o+3] -= wb * st.mstar
    R[o+1, o+1] = R[o+2, o+2] = wb^2 * M.Ra
    R[o+3, o+3] = wb^2 * M.Rfd
    R[o+4, o+4] = wb^2 * M.R1d
    R[o+5, o+5] = wb^2 * M.R1q
    R[o+6, o+6] = wb^2 * M.R2q
    R[od+2, od+2] = 1 / st.Rs
    g[G.v2] += (wb / s.C2) * u[1]
    g[G.v2+1] += (wb / s.C2) * u[2]
    g[od] = -wb
    g[od+1] = st.Tm
    g[od+2] = st.vs / st.Rs
    return J, R, g
end

"""    grid_ph_rhs(P, x, u, gf) -> the full grid-side vector field `(J - R) Q x + g`."""
function grid_ph_rhs(P::GridPH, x, u, gf = (0.0, 0.0))
    J, R, g = grid_ph_JRg(P, x, u, gf)
    return (J - R) * (P.Q * x) + g
end

"""    grid_ph_energy(P, x) -> the grid side's physical energy `1/2 x'Q x`."""
grid_ph_energy(P::GridPH, x) = dot(x, P.Q * x) / 2

"""
    grid_ph_power(P, x, u, gf) -> NamedTuple

The physical power flows at `x` (all in pu): the PCC supply, machine 2's shaft and DC-source
supplies, and the losses by group (`lines`, `loads`, `fault`, `stator`, `field`, `dampers`,
`source`, the DC series resistor `Rs i_s^2`).
"""
function grid_ph_power(P::GridPH, x, u, gf = (0.0, 0.0))
    s, G, o, od = P.s, P.G, P.o, P.od
    M, st = s.m2.mach, s.m2
    wb = M.omega_b
    z = P.Q * x
    sq(k) = x[k]^2 + x[k+1]^2
    zR(k, r) = r * (z[k]^2 + z[k+1]^2)
    is = (st.vs - x[od+2] / st.Cdc) / st.Rs
    return (; pcc = u[1] * x[G.v2] + u[2] * x[G.v2+1],
            shaft = st.Tm * z[od+1], dc_source = st.vs * is,
            lines = s.R23 * sq(G.i23) + s.R225 * sq(G.i225),
            loads = s.G3 * sq(G.v3) + s.G25 * sq(G.v25),
            fault = gf[1] * sq(G.v2) + gf[2] * sq(G.v3),
            stator = zR(o + 1, wb^2 * M.Ra),
            field = wb^2 * M.Rfd * z[o+3]^2,
            dampers = wb^2 * (M.R1d * z[o+4]^2 + M.R1q * z[o+5]^2 + M.R2q * z[o+6]^2),
            source = st.Rs * is^2)
end

# ---------------------------------------------------------------------------------------
# The reduction

"""
    ReducedGrid

The projection `x = Phi x_r` of a `GridPH`: `Phi` (n x r_total; identity on the kept indices and
the angle, `V_O` on the other network indices) and the test basis `Wf` (the energy's own,
`W = Q_y V (V'Q_y V)^{-1}`, and 1 on the angle), so `Wf'Phi = I`. `VO` holds the reduced network
basis (columns over `P.O`).
"""
struct ReducedGrid
    P::GridPH
    VO::Matrix{Float64}
    Phi::Matrix{Float64}
    Wf::Matrix{Float64}
end

"""    pcc_L(P, v) -> L v: multiplication by j on every phasor pair of the other network indices (v over `P.O`)."""
function pcc_L(P::GridPH, v::AbstractVector)
    w = similar(v)
    for k in 1:2:length(v)
        w[k], w[k+1] = -v[k+1], v[k]
    end
    return w
end

"""
    reduced_grid(P, VO) -> ReducedGrid

Builds `Phi` and `Wf` from the network basis `VO` (columns over `P.O`, full column rank).
"""
function reduced_grid(P::GridPH, VO::AbstractMatrix)
    n = P.G.n
    nk = length(P.K)
    r = nk + size(VO, 2) + 1
    Phi = zeros(n, r)
    for (j, k) in enumerate(P.K)
        Phi[k, j] = 1.0
    end
    Phi[P.O, nk+1:nk+size(VO, 2)] = VO
    Phi[P.od, r] = 1.0
    y = setdiff(1:n, [P.od])
    Vy = Phi[y, 1:r-1]
    Qy = P.Q[y, y]
    Wy = Qy * Vy / (Vy' * Qy * Vy)
    Wf = zeros(n, r)
    Wf[y, 1:r-1] = Wy
    Wf[P.od, r] = 1.0
    return ReducedGrid(P, Matrix{Float64}(VO), Phi, Wf)
end

"""    reduced_rhs(Rg, xr, u, gf) -> the reduced vector field `Wf' f(Phi x_r)` (the exact Petrov-Galerkin projection)."""
reduced_rhs(Rg::ReducedGrid, xr, u, gf = (0.0, 0.0)) = Rg.Wf' * grid_ph_rhs(Rg.P, Rg.Phi * xr, u, gf)

"""    reduced_project(Rg, x) -> x_r = Wf' x (the projection orthogonal in the physical energy)."""
reduced_project(Rg::ReducedGrid, x) = Rg.Wf' * x

"""
    pcc_qorth(P, VO, v) -> the part of `v` (over `P.O`) orthogonal in the physical energy to the
    columns of `VO`, normalised in that energy (or `nothing` if negligible).
"""
function pcc_qorth(P::GridPH, VO::AbstractMatrix, v::AbstractVector; tol = 1e-10)
    QO = P.Q[P.O, P.O]
    w = copy(v)
    for _ in 1:2
        for j in axes(VO, 2)
            c = VO[:, j]
            w -= c * (dot(c, QO * w) / dot(c, QO * c))
        end
    end
    nw = sqrt(dot(w, QO * w))
    nv = sqrt(dot(v, QO * v))
    return nw <= tol * max(nv, 1e-300) ? nothing : w / nw
end

"""
    pcc_add_pair(P, VO, v) -> VO with the L-completed pair of `v` appended (each part orthogonal in
    the physical energy to what is there), or `VO` unchanged if `v` adds nothing.
"""
function pcc_add_pair(P::GridPH, VO::AbstractMatrix, v::AbstractVector)
    a = pcc_qorth(P, VO, v)
    a === nothing && return VO
    V1 = hcat(VO, a)
    b = pcc_qorth(P, V1, pcc_L(P, a))
    return b === nothing ? V1 : hcat(V1, b)
end

"""
    pcc_krylov_pool(P, xs; kmax) -> Vector of candidate directions over `P.O`

The network's own port directions: `A_OO^k B_O` for `k = 0..kmax`, with `A = (J - R) Q` at the
rotor-frame operating point `xs`, `A_OO` its block on the other network indices and `B_O` its
columns from the kept indices (the PCC node and machine 2's stator) into them.
"""
function pcc_krylov_pool(P::GridPH, xs; kmax::Int = 6)
    J, R, _ = grid_ph_JRg(P, xs, (0.0, 0.0))
    A = (J - R) * P.Q
    AOO = A[P.O, P.O]
    BO = A[P.O, P.K]
    pool = Vector{Vector{Float64}}()
    X = BO
    for _ in 0:kmax
        for j in axes(X, 2)
            norm(X[:, j]) > 0 && push!(pool, X[:, j] / norm(X[:, j]))
        end
        X = AOO * X
    end
    return pool
end
