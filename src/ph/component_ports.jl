# Port passivity of single components, multi-input and multi-output, with interval
# verification (docs/active/COMPONENT_PASSIVITY.md).
#
# A component's own equations are linearised at a point (x, u) for chosen inputs and outputs;
# every other input is held at its value, and the contract's reservoir state (which acts on
# no output) is left out. For a component attached to a bus with a Norton admittance y_N in
# the Y-bus, `norton = true` turns its Norton current into the current it injects,
# I = I_N - y_N V. The linearisation is generic in the number type: with interval (x, u) it
# encloses the Jacobians of every point of the box.
#
# A multi-port with supply sign * Re(dU^H dY) is passive for some storage only if
# sign * (G(jw) + G(jw)^H) is positive semidefinite at every w where G is defined. One
# frequency w and one vector v with sign * Re(v^H G(jw) v) < 0 is therefore a certificate
# that no storage, quadratic or not, makes the port passive (for a C^2 storage of the
# nonlinear model: by the linearisation argument of docs/records/early/LOAD_PASSIVITY.md, Proposition L2).
# `port_witness_interval` evaluates that number in interval arithmetic.

"""
    ComponentPort

`x' = A x + B du`, `dy = C x + D du` of one component over `states`, from the inputs `inputs`
to the outputs `outputs` (for `norton = true`, the injected current `I_N - y_N V`).
The element type is Float64 or an interval type.
"""
struct ComponentPort{T}
    component::String
    states::Vector{String}
    inputs::Vector{String}
    outputs::Vector{String}
    A::Matrix{T}
    B::Matrix{T}
    C::Matrix{T}
    D::Matrix{T}
end

Base.show(io::IO, P::ComponentPort) =
    print(io, "ComponentPort(", P.component, ": ", join(P.inputs, ","), " -> ",
          join(P.outputs, ","), ", ", length(P.states), " states)")

function _nonreservoir(c::AbstractComponent, contracts::ContractSet)
    snames = state_names(c)
    res = get(contract(contracts, model_type(c)).raw, :reservoir, nothing)
    return [j for (j, s) in enumerate(snames) if !(res isa JSON3.Object && s == String(res[:state]))]
end

"""
    component_port(c, x, u; inputs, outputs, norton = false, hold = [], contracts) -> ComponentPort

Linearise component `c` at its state `x` and inputs `u` (Float64 or intervals). States named in
`hold` are frozen at their values (left out of the dynamics), which isolates their contribution:
for example, holding a machine's `delta` and `omega` gives its angle-frozen electrical port.
"""
function component_port(c::AbstractComponent, x::AbstractVector, u::AbstractVector;
                        inputs::AbstractVector{<:AbstractString},
                        outputs::AbstractVector{<:AbstractString}, norton::Bool = false,
                        hold::AbstractVector{<:AbstractString} = String[],
                        contracts::ContractSet = default_contracts())
    snames = state_names(c)
    all(s -> s in snames, hold) || throw(ArgumentError("$(name(c)): unknown state in hold = $hold"))
    keep = [j for j in _nonreservoir(c, contracts) if !(snames[j] in hold)]
    function index_of(s, names, what)
        k = findfirst(==(s), names)
        k === nothing && throw(ArgumentError("$(name(c)) has no $what $s"))
        return k
    end
    ki = [index_of(s, input_names(c), "input") for s in inputs]
    ko = [index_of(s, output_names(c), "output") for s in outputs]
    n, m = length(keep), length(ki)
    T0 = promote_type(eltype(x), eltype(u))
    function F(z)
        T = promote_type(eltype(z), T0)
        xx = collect(T, x)
        xx[keep] .= view(z, 1:n)
        uu = collect(T, u)
        uu[ki] .= view(z, n+1:n+m)
        dx = rhs!(similar(xx), c, xx, uu)
        y = outputs!(zeros(T, noutputs(c)), c, xx, uu)
        return vcat(dx[keep], y[ko])
    end
    J = ForwardDiff.jacobian(F, vcat(collect(T0, x[keep]), collect(T0, u[ki])))
    A, B = J[1:n, 1:n], J[1:n, n+1:end]
    C, D = J[n+1:end, 1:n], J[n+1:end, n+1:end]
    if norton
        (inputs == ["Vd", "Vq"] && outputs == ["Id", "Iq"]) ||
            throw(ArgumentError("norton = true needs inputs Vd, Vq and outputs Id, Iq"))
        y = norton_admittance(c)
        y === nothing || (D = D .- [real(y) -imag(y); imag(y) real(y)])
    end
    return ComponentPort(name(c), state_names(c)[keep], collect(String, inputs),
                         collect(String, outputs), A, B, C, D)
end

"""`G(jw) = C (jw I - A)^{-1} B + D` of a Float64 port."""
port_transfer(P::ComponentPort{Float64}, w::Real) =
    isempty(P.states) ? complex.(P.D) : P.C * ((im * w * I - P.A) \ complex.(P.B)) + P.D

"""
    port_hermitian_scan(P, ws; sign = 1) -> (lam, vecs)

The smallest eigenvalue of `sign * (G(jw) + G(jw)^H)/2` at each `w`, and its eigenvector.
"""
function port_hermitian_scan(P::ComponentPort{Float64}, ws; sign::Real = 1)
    lam, vecs = Float64[], Vector{ComplexF64}[]
    for w in ws
        G = port_transfer(P, w)
        E = eigen(Hermitian(sign .* (G + G') ./ 2))
        push!(lam, E.values[1])
        push!(vecs, E.vectors[:, 1])
    end
    return lam, vecs
end

"""
    port_witness_interval(P, w, v) -> Ival

An enclosure of `Re(v^H G(jw) v)` for an interval port `P`, the Float64 frequency `w` and
the Float64 vector `v`. `(jw I - A) X = B` is solved in its real form
`[-A -wI; wI -A] [Xr; Xi] = [B; 0]` by `interval_solve`.
"""
function port_witness_interval(P::ComponentPort, w::Real, v::AbstractVector{<:Complex})
    A, B, C, D = ival.(P.A), ival.(P.B), ival.(P.C), ival.(P.D)
    n, m = size(A, 1), size(B, 2)
    if n == 0
        Gr, Gi = D, zeros(Ival, size(D))
    else
        W = ival(Float64(w)) .* ival.(Matrix{Float64}(I, n, n))
        M = [-A -W; W -A]
        X, _ = interval_solve(M, vcat(B, zeros(Ival, n, m)))
        Gr = C * X[1:n, :] + D
        Gi = C * X[n+1:end, :]
    end
    vr, vi = ival.(real.(v)), ival.(imag.(v))
    return dot(vr, Gr * vr) + dot(vi, Gr * vi) - dot(vr, Gi * vi) + dot(vi, Gi * vr)
end

# -- the polar bus port (MASTER_TODO W2, polar-port programme) ----------------------------
#
# Input w = (theta, rho = ln|V|) of the bus voltage, output the injected power (P, Q), supply
# -(P theta' + Q rho'). At every bus the injected powers of all components sum to zero, so these
# supplies cancel in any interconnection. With w as a position-type input the frequency lemma
# applies to the system with input w' (transfer G/s): a C^2 storage S(x, w) >= 0 with
# dS/dt <= -dy' dw' requires Im(v^H G(jw) v) <= 0 for all v and every w > 0.

"""
    polar_transfer(P, V0, I0, w) -> Matrix{ComplexF64}

The transfer from `(dtheta, drho)` to the injected `(dP, dQ)` at frequency `w`, for a bus port
`P` (inputs `Vd, Vq`, outputs the injected current) at bus voltage `V0 = (Vd, Vq)` and injected
current `I0 = (Id, Iq)`: `dV = [-Vq Vd; Vd Vq] dw`, `dP = I0' dV + V0' dI`,
`dQ = (Id dVq - Iq dVd) + (Vq dId - Vd dIq)`.
"""
function polar_transfer(P::ComponentPort{Float64}, V0::AbstractVector, I0::AbstractVector, w::Real)
    Vd, Vq = V0
    Id, Iq = I0
    Tin = [-Vq Vd; Vd Vq]
    A1 = [Id Iq; -Iq Id]
    A2 = [Vd Vq; Vq -Vd]
    return (A1 + A2 * port_transfer(P, w)) * Tin
end

"""
    polar_scan(P, V0, I0, ws) -> (lam, vecs)

The largest eigenvalue of `(G - G^H)/(2i)` (the Hermitian matrix whose quadratic form is
`Im(v^H G v)`) at each `w`, and its eigenvector. A positive value at some `w > 0` excludes every
`C^2` storage at the polar port.
"""
function polar_scan(P::ComponentPort{Float64}, V0, I0, ws)
    lam, vecs = Float64[], Vector{ComplexF64}[]
    for w in ws
        G = polar_transfer(P, V0, I0, w)
        E = eigen(Hermitian((G - G') ./ (2im)))
        push!(lam, E.values[end])
        push!(vecs, E.vectors[:, end])
    end
    return lam, vecs
end

"""
    injected_current(c, x, u) -> (Id, Iq)

The current a Norton component injects into the network at its state `x` and inputs `u`:
its Norton current minus `y_N V`.
"""
function injected_current(c::AbstractComponent, x::AbstractVector, u::AbstractVector)
    y = outputs!(zeros(eltype(x), noutputs(c)), c, x, u)
    jd = findfirst(==("Id"), output_names(c))
    jq = findfirst(==("Iq"), output_names(c))
    I = complex(y[jd], y[jq])
    yN = norton_admittance(c)
    yN === nothing || (I -= yN * complex(u[1], u[2]))
    return [real(I), imag(I)]
end
