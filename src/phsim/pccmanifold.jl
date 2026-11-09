# Truncation of the whole grid seen from one machine's PCC, with the block energy-consistent
# projection (docs/active/PCC_MANIFOLD_REDUCTION_DERIVATION.md; ROA_ROUTES_TODO.md H0).
#
# The state splits into the PCC machine (kept), its bus's voltage pair (kept, the port), the other
# machines' angles (zero energy) and every other energy-bearing grid state `y`. The flat map is
# x = (x_G, v2, delta* + Vd theta, Phi a); the test map W = blockdiag(I, I, Wd, Q_y Phi (Phi'Q_y Phi)^-1).
# The reduced model is xi' = W' f(phi(xi)): port-Hamiltonian with the physical energy of the
# reconstructed state, every loss and source power exact at it (derivation section 3).

"""
    PCCSplit

Index sets of a `PHSystem` seen from the machine `machine` (its index in `s.machines`): `G` that
machine's states, `V2` its attachment bus's voltage pair, `D` the other machines' angles, `Y` every
other state (all energy-bearing). Over `Y`: `pos` maps a state index to its position, `isnet` marks
network phasor pairs (rotated by the common rotation), `netpairs` their first positions, `P` the
positions of the other machines' momenta (in `D` order) and `H` their inertia constants.
"""
struct PCCSplit
    machine::Int
    G::Vector{Int}
    V2::Vector{Int}
    D::Vector{Int}
    Y::Vector{Int}
    pos::Dict{Int,Int}
    isnet::BitVector
    netpairs::Vector{Int}
    P::Vector{Int}
    H::Vector{Float64}
end

function pcc_split(s::PHSystem, machine::Int)
    isempty(s.conv) && isempty(s.cores) ||
        throw(ArgumentError("pcc_split: converters and cores are not handled (rotor-frame states only)"))
    m0 = s.machines[machine]
    m0.cut == 0 || throw(ArgumentError("pcc_split: the PCC machine sits at a reduced node"))
    s.qidx[m0.node] > 0 || throw(ArgumentError("pcc_split: the PCC bus has no voltage state"))
    G = collect(m0.offset:_idelta(m0))
    V2 = [s.qidx[m0.node], s.qidx[m0.node] + 1]
    others = [m for (j, m) in enumerate(s.machines) if j != machine]
    D = [_idelta(m) for m in others]
    Y = setdiff(1:s.n, vcat(G, V2, D))
    pos = Dict(k => j for (j, k) in enumerate(Y))
    mach = falses(s.n)
    for m in s.machines
        mach[m.offset:_idelta(m)] .= true
        m.cut == 0 || (mach[m.tidx:m.tidx+1] .= true)       # a machine's branch winding: rotor frame
    end
    isnet = falses(length(Y))
    netpairs = Int[]
    for i in 1:2:s.n
        (mach[i] || i in V2) && continue
        isnet[pos[i]] = isnet[pos[i+1]] = true
        push!(netpairs, pos[i])
    end
    P = [pos[_ip(m)] for m in others]
    H = [m.w.H for m in others]
    return PCCSplit(machine, G, V2, D, Y, pos, isnet, netpairs, P, H)
end

"""    pcc_Q(s) -> Q with H(x) = x'Qx/2 exactly (the model's energy is quadratic without cores)."""
pcc_Q(s::PHSystem) = reduce(hcat, [phs_gradient(s, Float64.(1:s.n .== i)) for i in 1:s.n])

"""    pcc_rot(sp, v, g) -> v over `Y` with every network pair rotated by `g` (rotor-frame entries fixed)."""
function pcc_rot(sp::PCCSplit, v::AbstractVector, g::Real)
    w = Vector{promote_type(eltype(v), typeof(g))}(v)
    c, sn = cos(g), sin(g)
    for a in sp.netpairs
        w[a], w[a+1] = c * v[a] - sn * v[a+1], sn * v[a] + c * v[a+1]
    end
    return w
end

"""    pcc_split3(sp, v) -> [rotor-frame part, network part, j * network part] of `v` over `Y`."""
pcc_split3(sp::PCCSplit, v::AbstractVector) =
    [v .* .!sp.isnet, v .* sp.isnet, pcc_rot(sp, v .* sp.isnet, pi / 2)]

"""
    pcc_qappend(QY, Phi, cols; tol) -> (Phi, added)

Appends each column of `cols` orthonormalised in the physical energy `QY` against `Phi`, unless its
remainder is below `tol` of its own energy norm (then skipped). `added` counts the new columns.
"""
function pcc_qappend(QY::AbstractMatrix, Phi::AbstractMatrix, cols; tol = 1e-8)
    Phi = Matrix{Float64}(Phi)
    added = 0
    for c in cols
        nc = sqrt(max(dot(c, QY * c), 0.0))
        nc > 0 || continue
        w = copy(c)
        for _ in 1:2, j in axes(Phi, 2)
            w -= Phi[:, j] * dot(Phi[:, j], QY * w)
        end
        nw = sqrt(max(dot(w, QY * w), 0.0))
        nw > tol * nc || continue
        Phi = hcat(Phi, w / nw)
        added += 1
    end
    return Phi, added
end

"""    pcc_protected(sp, QY, x0) -> the operating point's rotation-closed span (three columns over `Y`)."""
pcc_protected(sp::PCCSplit, QY, x0) = first(pcc_qappend(QY, zeros(length(sp.Y), 0), pcc_split3(sp, x0[sp.Y])))

"""
    pcc_pod(sp, QY, Phi0, snaps) -> (modes, energies)

Proposal directions from snapshot columns `snaps` (over `Y`): each projected onto the physical-energy
complement of `Phi0`, the state covariance rotation-averaged (rotor-frame block kept, network block
`(C + L C L')/2`, cross blocks zero), then the eigenvectors in the physical energy:
`(QY^1/2 C QY^1/2) u = lambda u`, `phi = QY^-1/2 u`. Sorted by `lambda`, largest first. Network modes
come in pairs (`v`, `j v`).
"""
function pcc_pod(sp::PCCSplit, QY::AbstractMatrix, Phi0::AbstractMatrix, snaps::AbstractMatrix)
    Z = snaps - Phi0 * (Phi0' * (QY * snaps))
    C = Z * Z'
    r, nn = .!sp.isnet, sp.isnet
    Cb = zeros(size(C))
    Cb[r, r] = C[r, r]
    Lm = zeros(length(sp.Y), length(sp.Y))                 # j on network pairs, as a matrix over Y
    for a in sp.netpairs
        Lm[a+1, a] = 1.0; Lm[a, a+1] = -1.0
    end
    CN = zeros(size(C)); CN[nn, nn] = C[nn, nn]
    Cb .+= (CN .+ Lm * CN * Lm') ./ 2
    E = eigen(Symmetric(QY))
    Qh = E.vectors * Diagonal(sqrt.(max.(E.values, 0.0))) * E.vectors'
    Qhi = E.vectors * Diagonal(1 ./ sqrt.(E.values)) * E.vectors'
    F = eigen(Symmetric(Qh * Cb * Qh))
    o = sortperm(F.values; rev = true)
    return Qhi * F.vectors[:, o], F.values[o]
end

"""
    PCCReduced

A reduced model: the split `sp`, the energy-bearing basis `Phi` (over `Y`, orthonormal in the physical
energy), the angle basis `Vd` and its left inverse `Wd`, the full trial and test matrices `V`, `W`
(`W'V = I`) and the operating angles `dstar`. Coordinates `xi = (x_G, v2, theta, a)`.
"""
struct PCCReduced
    sp::PCCSplit
    Phi::Matrix{Float64}
    Vd::Matrix{Float64}
    Wd::Matrix{Float64}
    V::Matrix{Float64}
    W::Matrix{Float64}
    dstar::Vector{Float64}
end

"""
    pcc_reduced(s, sp, Q, Phi, x0; Vd_extra) -> PCCReduced

Builds the map from `Phi`. The angle basis is the kinematic image of the kept momenta plus the
common direction (derivation 3.7), with any extra angle columns `Vd_extra`; orthonormal, so
`Wd = Vd`.
"""
function pcc_reduced(s::PHSystem, sp::PCCSplit, Q::AbstractMatrix, Phi::AbstractMatrix, x0::AbstractVector;
                     Vd_extra = zeros(length(sp.D), 0))
    nD = length(sp.D)
    K = hcat(Diagonal(1 ./ (2 .* sp.H)) * Phi[sp.P, :], ones(nD), Vd_extra)
    F = svd(K)
    k = count(>(1e-10 * F.S[1]), F.S)
    Vd = F.U[:, 1:k]
    QY = Q[sp.Y, sp.Y]
    Wy = QY * Phi / Symmetric(Phi' * QY * Phi)
    nG, m = length(sp.G), size(Phi, 2)
    r = nG + 2 + k + m
    V, W = zeros(s.n, r), zeros(s.n, r)
    for (j, i) in enumerate(sp.G)
        V[i, j] = W[i, j] = 1.0
    end
    for (j, i) in enumerate(sp.V2)
        V[i, nG+j] = W[i, nG+j] = 1.0
    end
    V[sp.D, nG+3:nG+2+k] = Vd; W[sp.D, nG+3:nG+2+k] = Vd
    V[sp.Y, nG+3+k:r] = Phi;   W[sp.Y, nG+3+k:r] = Wy
    return PCCReduced(sp, Matrix(Phi), Vd, Vd, V, W, x0[sp.D])
end

pcc_nred(R::PCCReduced) = size(R.V, 2)

"""    pcc_phi(R, xi) -> the reconstructed full state."""
function pcc_phi(R::PCCReduced, xi::AbstractVector)
    x = R.V * xi
    x[R.sp.D] .+= R.dstar
    return x
end

"""    pcc_project(R, x) -> xi (the projection orthogonal in the physical energy; angles by Wd)."""
function pcc_project(R::PCCReduced, x::AbstractVector)
    y = copy(x)
    y[R.sp.D] .-= R.dstar
    return R.W' * y
end

"""    pcc_rhs(R, s, xi, u) -> W' f(phi(xi)), the reduced vector field."""
pcc_rhs(R::PCCReduced, s::PHSystem, xi::AbstractVector, u::PHInput) = R.W' * phs_rhs(s, pcc_phi(R, xi), u)

"""
    pcc_power(s, sp, x) -> the power the PCC machine delivers into its bus (per unit), from the
    stator-bus coupling of `J` and the efforts (the pairing the reduction keeps exactly).
"""
function pcc_power(s::PHSystem, sp::PCCSplit, x::AbstractVector)
    m = s.machines[sp.machine]
    z = phs_gradient(s, x)
    c = im * complex(cos(x[_idelta(m)]), -sin(x[_idelta(m)])) / m.a
    # J[V2, stator] is the 2x2 real block of -conj(c)
    cc = -conj(c)
    B = [real(cc) -imag(cc); imag(cc) real(cc)]
    return dot(z[sp.V2], B * z[m.offset:m.offset+1])
end
