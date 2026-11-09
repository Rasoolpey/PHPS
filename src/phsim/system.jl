# The port-Hamiltonian grid model of PH_SIMULATOR_TODO.md (docs/active/PH_SIMULATOR_SPEC.md):
# an electromagnetic-transient (EMT) model, with stator flux, line current and capacitor voltage
# dynamics. A whole case as one port-Hamiltonian system, balanced dq, built from the physical network
# (`emdq/network.jl`) and the full-winding machine (`emdq/pilot.jl`). A separate model; the
# RMS kernels and their parity are not touched.
#
# States are energy variables scaled so that the gradient of the energy is the plain effort:
#   node k with capacitance:  q_k   = C_k v_k / omega_b      (complex, two reals)
#   branch l, shunt inductor: phi   = L i / omega_b          (complex, two reals)
#   machine, rotor frame:     lam   = psi / omega_b for (d, q, fd, 1d, 1q[, 2q]), momentum p,
#                             angle delta (no energy)
# Energy (pu*s), quadratic and independent of delta:
#   H = sum omega_b |q|^2/(2C) + sum omega_b |phi|^2/(2L) + sum_m [omega_b lam' K lam/2 + p^2/(4H)]
#
# Equations (complex; B the incidence of `pn_incidence`, motor convention for the machines):
#   q_k'   = inj_k - (G_k + g_fault,k) v_k - j C_k v_k
#   inj_k  = -(B i)_k - sum_{s at k} i_s - sum_{m at k} i_net,m / conj(a_m)
#   v_k    = inj_k / (G_k + g_fault,k)                    at a node with conductance and no C
#   phi_l' = (B^H v)_l - R_l i_l - j L_l i_l
#   phi_s' = v_b - R_s i_s - j L_s i_s        (R_s > 0 only for a series R-L load)
#   v_dq   = j e^{-j delta} v_k / a_m,   i_net = -j e^{j delta} (i_d + j i_q)
#   lam_d' = v_d - Ra i_d + omega_b omega lam_q,   lam_q' = v_q - Ra i_q - omega_b omega lam_d
#   lam_fd' = v_f - Rfd i_fd,   lam_1d' = -R1d i_1d,   lam_1q' = -R1q i_1q,   lam_2q' = -R2q i_2q
#   p'     = T_m + omega_b (lam_d i_q - lam_q i_d),   delta' = omega_b (omega - 1)
# With `const_power` the shaft input is a power: T_m = P_m / omega.
#
# Exact reductions made at assembly (spec section 5), each recorded in `PHSystem`:
#   - series merge: a generator bus where only the stator and one transformer meet is removed;
#     the transformer's r and x join the stator and the machine sits at the transformer's other
#     bus through the ideal ratio a;
#   - a node with conductance and no capacitance has its voltage eliminated by its KCL row;
#   - identical machines on one bus are one machine of their combined rating (`units`), the
#     choice the PowerFactory reference makes for G 05. Exact while the units have equal states;
#   - with `loads = :series_all` only: a node where only inductors meet (`PHCut`, below).
# Boundaries: every shaft torque (or turbine power) and field voltage is a fixed input.
#
# Two choices exist only to simulate what PowerFactory's EMT model simulates, for comparison
# with it; the defaults are the model's own:
#   - `loads = :series`: an inductive load at a bus with capacitance is a resistor and an
#     inductor in series (PowerFactory general load, EMT, eq. 35-36), not in parallel. At a bus
#     without capacitance it stays parallel: in series it would make an inductor cutset.
#     `loads = :series_all` makes that one a series branch too and reduces the cutset;
#   - `circuits`: winding data taken as given (PowerFactory's own equivalent-circuit values) in
#     place of the Sauer-Pai map from standard data.
#
# A node where only inductors meet (a series R-L load `R_L, L_L` at a bus k without capacitance
# or conductance, with the branches and at most one machine there). Unreduced, v_k is a
# multiplier and KCL a constraint on inductor currents. Write branch j at the node as
#   (L_j/omega_b) i_j' = gamma_j v_k + c_j v_far - R_j i_j - j L_j i_j,
# gamma_j = 1/a_j on the branch's ratio side and -1 on its other side; the node receives
# -gamma_j i_j (real ratios). The load's current is eliminated by KCL, i_L = sum of what the
# node receives, and v_k by the load's own equation (van der Schaft 2013, section 4.1.1,
# eq. 4.6-4.8: states G_perp x, energy the total energy on the constraint):
#   - without a machine: the branch states become Phi = M i / omega_b,
#       M = diag(L_j) + L_L gamma gamma',    Rm = diag(R_j) + R_L gamma gamma',
#       Phi' = c .* v_far - Rm i - j omega_b Phi;
#   - with a machine (ratio a) and one branch T (b = gamma_T): the stator pair and the branch
#     pair, both in the rotor frame, become
#       Lam   = lam + [(L_L/a^2) i + (b L_L/a) i_T] / omega_b,
#       Phi_T = [(b L_L/a) i + (L_T + b^2 L_L) i_T] / omega_b,
#       Lam_d'   = -Rm11 i_d - Rm12 i_Td + omega_b omega Lam_q,          (q alike)
#       Phi_Td'  = Re(c j e^{-j delta} v_far) - Rm12 i_d - Rm22 i_Td + omega_b omega Phi_Tq,
#       Rm = [Ra + R_L/a^2, R_L b/a; R_L b/a, R_T + b^2 R_L],
#     so the machine has one more winding on each axis, attached at the far bus; the sum of both
#     pairs' flux_d i_q - flux_q i_d is the machine's own torque. This is the system of Mayer
#     and Wasynczuk (IEEE Trans. Power Syst. 6(3), 1991, eq. 33-38) in energy coordinates.
# The energy stays quadratic and free of delta; v_k is recovered from the load's equation with
# the current rates of the vector field (`phs_cut_voltage`).
#
# Converters (GFM_VSM_PHTRUE; docs/active/PH_CONVERTER_MODEL.md, section 3): PowerFactory's static
# generator in its voltage-source model, an EMF u_c behind the series inductance L_c = Zseries. Each
# converter bus gets a source row past the bus rows, whose voltage is the input `PHInput.uc`, and a
# branch ending at that row: at a bus where only the converter and its step-up transformer meet,
# the transformer itself, with L_c added to its inductance (the series merge; the bus is removed);
# elsewhere a branch of its own from the bus. The branch current flows towards the source row, so
# the converter injects i_c = -i. The terminal voltage is v_k = u_c + j L_c i + (L_c/omega_b) i',
# affine in u_c: v_k = alpha u_c + w(x), with alpha a constant (`PHConverter`).

"""
    PHMachine

One machine of a `PHSystem`: winding data `w` (stator leakage and resistance include the merged
transformer `trf`, `x_ext`, `r_ext`), terminal bus row `node` and ideal ratio `a` (the stator sees
`v_node / a`), original generator bus id `bus`, number of identical units represented, and the
first state index `offset`. Local states: `lam_d, lam_q, lam_fd, lam_1d, lam_1q[, lam_2q], p,
delta`. `Kd`, `Kq` are the inverse inductance matrices its currents are computed with: `w`'s,
or, at a reduced node (`cut`, the index in the system's `cuts`; 0 otherwise), those extended by
the branch winding, whose flux pair is the state `tidx`.
"""
struct PHMachine
    name::String
    w::WindingMachine
    node::Int
    a::ComplexF64
    bus::Int
    trf::String
    x_ext::Float64
    r_ext::Float64
    units::Int
    offset::Int
    cut::Int
    tidx::Int
    Kd::Matrix{Float64}
    Kq::Matrix{Float64}
end

phs_nstates(m::PHMachine) = m.w.nq == 2 ? 8 : 7
_ip(m::PHMachine) = m.offset + (m.w.nq == 2 ? 6 : 5)       # momentum
_idelta(m::PHMachine) = _ip(m) + 1

"""
    PHCut

A node where only inductors meet, reduced as the file header describes: bus row `node`, the
series load `RL`, `LL` (the shunt inductor `shunt` of the network, which has no state), the
branches at the node (`pos`, positions in the system's `branches`; `gamma`, `c` the coefficients
of the node's and the far bus's voltage in each branch's row; `far` the far bus rows), and
`machine` (index in the system's `machines`, 0 if none). Without a machine `M`, `Rm` are the
inductance and resistance matrices over the branches and `Minv = inv(M)`; with one they are the
2 by 2 matrices over (stator, branch) that the reduction adds: `M = [L_L/a^2 b L_L/a; b L_L/a
L_T + b^2 L_L]`, `Rm` as in the file header.
"""
struct PHCut
    node::Int
    shunt::Int
    RL::Float64
    LL::Float64
    pos::Vector{Int}
    gamma::Vector{Float64}
    c::Vector{Float64}
    far::Vector{Int}
    machine::Int
    M::Matrix{Float64}
    Minv::Matrix{Float64}
    Rm::Matrix{Float64}
end

"""
    PHConverter

One converter of a `PHSystem` (file header): its bus id `bus` and bus row `k` (removed if
`merged`), its source row `row`, the position `pos` of its branch in the system's `branches`, the
series inductance `Lc`, the constant `alpha` of its terminal voltage `v_k = alpha u_c + w`, the
number of identical units it stands for, the case's component `comp` (one unit) and the index `core` of its
transformer's saturating core in the system's `cores` (0 if none; then `alpha` is NaN: the terminal voltage is no longer
affine in `u_c` with a constant, record section 21, T1 row 4).
"""
struct PHConverter
    name::String
    bus::Int
    k::Int
    row::Int
    pos::Int
    merged::Bool
    Lc::Float64
    alpha::Float64
    units::Int
    comp::Union{GFM_VSM_PHTRUE,GFM_DROOP_PHTRUE,GFM_VOC_PHTRUE,GFL_PHTRUE}
    core::Int
end
PHConverter(name, bus, k, row, pos, merged, Lc, alpha, units, comp) =
    PHConverter(name, bus, k, row, pos, merged, Lc, alpha, units, comp, 0)

"""
    PHSystem

A case as one port-Hamiltonian system. `net` is the full element list; `dyn` and `alg` are the
bus rows with a voltage state and with an eliminated voltage; `merged` the bus rows removed by
the series merge; `qidx[k]` the first state of bus row `k` (0 if it has none); `branches` the
kept branches with `bidx` their first states and `charge` their total charging; `sidx` the first
states of the shunt inductors (0 for the load of a reduced node) and `Rsh` their series
resistances (zero unless the load is a series R-L branch); `Cown` the node capacitance not owned
by a line (loads, shunts); `cut` the bus rows of the reduced inductor-only nodes, `cuts` their
data, `cutof[k]` the index in `cuts` of bus row `k` and `bcut[j]` that of branch position `j`
(0 if none); `cnames`, `ckind` the components of the energy account (`phs_ledger`), in its
order; `conv` the converters, whose source rows follow the bus rows; `cores` the saturating transformer cores
(`transformer.jl`), whose source loop fluxes follow the machines' states, and `iclk` the frame clock's state after them
(0 without cores); `step` holds the step's discrete-gradient currents while `phs_simulate` evaluates a step (`nothing`
otherwise; `:short` evaluates every core short-circuited, for the step rule's bound). `stationary` selects the frame
of the network states: `false` (the default) the synchronous frame, `true` the stationary frame
(`docs/realtime/docs/STEP0_CONTRACTS.md` A3-A4; made by `phs_frame`), with `theta0` the frame angle at t = 0.
"""
struct PHSystem
    net::PhysicalNetwork
    omega_b::Float64
    dyn::Vector{Int}
    alg::Vector{Int}
    merged::Vector{Int}
    qidx::Vector{Int}
    branches::Vector{Int}
    bidx::Vector{Int}
    charge::Vector{Float64}
    sidx::Vector{Int}
    Cown::Vector{Float64}
    Rsh::Vector{Float64}
    cut::Vector{Int}
    cuts::Vector{PHCut}
    cutof::Vector{Int}
    bcut::Vector{Int}
    machines::Vector{PHMachine}
    n::Int
    cnames::Vector{String}
    ckind::Vector{Symbol}
    conv::Vector{PHConverter}
    cores::Vector{PHCore}
    iclk::Int
    step::Base.RefValue{Any}
    tw::Vector{Any}            # travelling-wave lines (tline.jl; `lines = :travelling`): (; name, from, to, r, x, b, Zc, G)
    twG::Vector{Float64}       # their Norton conductance at each bus row (a port relation, not a load: kept apart from G)
    twJ::Vector{ComplexF64}    # their history current into each bus row, set by the simulator before each step
    nsec::Int                  # sections per travelling-wave line
    motors::Vector{PHMotor}    # composite loads (`loads = :composite`, motor.jl): the induction motors, states after the cores'
    statics::Vector{PHStatic}  # their static parts (the case's P law as a conductance), lag states after the motors'
    sl::Vector{Int}            # index in `statics` of each bus row (0 without one)
    z0::Any                    # the zero-axis network (zeroseq.jl, upgrade 4; nothing: none, the model as before)
    asym::Any                  # phase-asymmetric faults in force (`af`: (row, G_abc, ground)) and the frame angle theta
    stationary::Bool
    theta0::Float64
end

"""
    PHInput

The boundary inputs: shaft torque `Tm` and field voltage `vf` per machine (generator torque
positive, reciprocal per unit `vf = Rfd i_fd` at rest), and the fault conductance `gf` per bus
row (0 without a fault). With `const_power`, `Tm` is the turbine power and the torque is
`Tm / omega` (what PowerFactory does with a turbine power input and no governor). `cf[1]` is the frame phasor
e^{j theta_f}: read only in the stationary frame, where the simulators set it to e^{j (omega_b t + theta0)} for the
time of each evaluation (`_set_frame!`); 1 otherwise and unused.
"""
struct PHInput{T}
    Tm::Vector{T}
    vf::Vector{T}
    gf::Vector{Float64}
    const_power::Bool
    uc::Vector{Complex{T}}        # each converter's EMF, network frame (docs/active/PH_CONVERTER_MODEL.md)
    cf::Vector{Complex{T}}        # the frame phasor e^{j theta_f} (stationary frame only), in the input's number type
end

PHInput{T}(Tm::Vector{T}, vf::Vector{T}, gf::Vector{Float64}, const_power::Bool, uc::Vector{Complex{T}}) where {T} =
    PHInput{T}(Tm, vf, gf, const_power, uc, Complex{T}[1])
PHInput(Tm::Vector{T}, vf::Vector{T}, gf::Vector{Float64}, const_power::Bool, uc::Vector{Complex{T}}) where {T} =
    PHInput{T}(Tm, vf, gf, const_power, uc)
PHInput{T}(Tm::Vector{T}, vf::Vector{T}, gf::Vector{Float64}, const_power::Bool) where {T} =
    PHInput{T}(Tm, vf, gf, const_power, Complex{T}[])
PHInput(Tm::Vector{T}, vf::Vector{T}, gf::Vector{Float64}, const_power::Bool) where {T} =
    PHInput{T}(Tm, vf, gf, const_power)
PHInput(Tm::Vector{T}, vf::Vector{T}, gf::Vector{Float64}) where {T} = PHInput{T}(Tm, vf, gf, false)

# the shaft torque at speed `omega`: the input itself, or the input as a power
@inline _shaft_torque(u::PHInput, j, omega) = u.const_power ? u.Tm[j] / omega : u.Tm[j]

# n identical machines in parallel on the system base: every impedance divided by n, inertia
# and damping multiplied by n, time constants unchanged
_parallel(p::GenrouParams, n::Int) =
    GenrouParams(p.H * n, p.D * n, p.ra / n, p.xd / n, p.xq / n, p.xd_prime / n, p.xq_prime / n,
                 p.xd_double_prime / n, p.xq_double_prime / n, p.Td0_prime, p.Tq0_prime,
                 p.Td0_double_prime, p.Tq0_double_prime, p.xl / n, p.omega_b)
_parallel(p::GensalParams, n::Int) =
    GensalParams(p.H * n, p.D * n, p.ra / n, p.xd / n, p.xq / n, p.xd_prime / n,
                 p.xd_double_prime / n, p.xq_double_prime / n, p.Td0_prime, p.Td0_double_prime,
                 p.Tq0_double_prime, p.xl / n, p.A_sat, p.B_sat, p.omega_b)

# the inverse inductance matrices of a machine with one more winding on each axis, coupled to
# the stator by `M = [alpha beta; beta kappa]` (added to the stator's own entry, and the new
# winding's own inductance)
function _extended_K(w::WindingMachine, M::Matrix{Float64})
    ext(K) = (n = size(K, 1); L = zeros(n + 1, n + 1); L[1:n, 1:n] .= inv(K); L[1, 1] += M[1, 1];
              L[1, n+1] = L[n+1, 1] = M[1, 2]; L[n+1, n+1] = M[2, 2]; inv(Symmetric(L)))
    return Matrix(ext(w.Kd)), Matrix(ext(w.Kq))
end

"""
    ph_system(sys::DAESystem; v0 = nothing, loads = :parallel, circuits = nothing,
              stray_capacitance = 0.0) -> PHSystem

Assemble the port-Hamiltonian model of the case behind `sys`: its `physical_network` (loads as
constant-impedance elements at the voltages `v0`, in network order) and one full-winding machine
(`winding_machine`, the Sauer-Pai map) per GENROU or GENSAL record, with the reductions listed in
the file header. `loads = :series`, `loads = :series_all` and `circuits` (a `Dict` from generator
bus id to winding data on the system base, all units of the bus together: `Ll, Lad, Laq, Lfd,
L1d, L1q, L2q, Ra, Rfd, R1d, R1q, R2q, H`, with `L2q = nothing` for one q-axis rotor winding)
are the comparison choices of the file header. `stray_capacitance > 0` (with `:series_all`)
puts that capacitance at every capacitance-free bus with an inductive load in place of reducing
the node: a different circuit, kept only as a reference the reduced model must be the limit of.
Throws when a reduction's condition does not hold (different machines on one bus, a generator
bus with more than its transformer, a machine with damping `D != 0`, an inductor-only node with
a phase-shifting ratio, with two machines, or whose branches do not end at buses with
capacitance).

`cores` (a dictionary from a converter's bus id to its step-up transformer's total rating in MVA, all parallel units)
gives those transformers a saturating core (`transformer.jl`; record sections 20-23): the leakage `R_T + j X_T` split
0.5/0.5 around the core, the GMD curve (`core_curve_gmd`) on the rating. Only a converter merged with its transformer.
"""
function ph_system(sys::DAESystem; v0 = nothing, loads::Symbol = :parallel, circuits = nothing,
                   stray_capacitance::Real = 0.0, cores = nothing, lines::Symbol = :lumped, nsec::Int = 4,
                   mix = (A = 0.10, B = 0.10, C = 0.10), composite_exclude = [39], LFm::Real = 0.8,
                   motor_eps::Real = 1e-3, relays = nothing, zero::Symbol = :none, zero_pf_json = nothing,
                   zm0_3limb::Real = 5.0, theta_pw::Real = 0.0)
    loads in (:parallel, :series, :series_all, :composite) || throw(ArgumentError("ph_system: loads = $loads"))
    lines in (:lumped, :travelling) || throw(ArgumentError("ph_system: lines = $lines"))
    case = sys.case
    pn = physical_network(case; omega_b = sys.omega_b, v0)
    nb, m, ns = pn_sizes(pn)
    # travelling-wave lines (tline.jl, reviews 71-72c): every line (b > 0, no ratio) leaves the lumped network: its b/2
    # shares of node capacitance go, its branch is dropped below, and each end becomes the Norton conductance 1 / Z
    # (Z = Zc + R / (4 nsec)) plus the history current twJ set by the simulator
    tw, twG, istw = Any[], zeros(nb), falses(m)
    if lines == :travelling
        cl = Dict(String(l.idx) => l for l in case.lines)
        for l in 1:m
            c = get(cl, pn.branch[l], nothing)
            (c === nothing || c.b <= 0 || pn.a[l] != 1) && continue
            i, j = pn.from[l], pn.to[l]
            pn.C[i] -= c.b / 2; pn.C[j] -= c.b / 2
            abs(pn.C[i]) < 1e-12 && (pn.C[i] = 0.0); abs(pn.C[j]) < 1e-12 && (pn.C[j] = 0.0)
            Zc = sqrt(c.x / c.b); G = 1 / (Zc + c.r / (4nsec))
            twG[i] += G; twG[j] += G; istw[l] = true
            push!(tw, (; name = pn.branch[l], from = i, to = j, r = c.r, x = c.x, b = c.b, Zc, G))
        end
    end
    # composite loads (motor.jl; reviews 74-74b): at every load bus not in `composite_exclude`, the induction motors A, B,
    # C (shares `mix` of P, each on its base P_m / LFm, Tm0 from the slip that draws P_m at the load flow's |V|) and the
    # static rest: the case's P law as a conductance, Q as a fixed inductor or capacitor carrying Q0 less the motors' Q
    mdat, sdat = NamedTuple[], NamedTuple[]
    slb = falses(nb)
    if loads == :composite
        clds = Dict(bus(c) => c for c in sys.comps if c isa COMPLEXLOAD)
        for (b, P, Q) in load_power_by_bus(case, pn.net)
            (b in composite_exclude || (P == 0 && Q == 0)) && continue
            k = bus_index(pn.net, b)
            cl = get(clds, b, nothing)
            cl === nothing && throw(ArgumentError("ph_system: the composite load at bus $b has no COMPLEXLOAD record"))
            V0 = _load_v0(case, pn.net, b, v0); V02 = max(V0^2, 1e-12)
            Qm, Pmt = 0.0, 0.0
            for X in (:A, :B, :C)
                F = Float64(getfield(mix, X)); F > 0 || continue
                Pm = F * P
                c0 = motor_circuit(WECC_MOTORS[X], Pm / LFm; omega_b = sys.omega_b, eps = motor_eps)
                sl0 = motor_slip(c0, V0 + 0im, Pm; omega_b = sys.omega_b)
                st = motor_steady(c0, V0 + 0im, sl0; omega_b = sys.omega_b)
                Tm0 = c0.etrq == 2 ? st.Te / (st.omega * abs(st.omega)) : st.Te * sqrt(st.omega^2 + motor_eps^2) / st.omega
                push!(mdat, (; name = "motor$(X)_bus$b", kind = X, node = k, bus = b, c = with_torque(c0, Tm0), F, Pm,
                             slip = sl0))
                Qm += st.Q; Pmt += Pm
            end
            Ps, Qs = P - Pmt, Q - Qm
            Ps >= 0 || throw(ArgumentError("ph_system: the motors at bus $b draw more than its P ($Pmt > $P)"))
            p = cl.params
            law = StaticLaw(Ps, V0, Float64(p["kpv"]), Float64(p["t1"]), Float64(p["udmin"]), Float64(p["udmax"]))
            (law.t1 > 0 && law.udmin > 0) ||
                throw(ArgumentError("ph_system: a static part without a lag or a low-voltage band (bus $b) is not implemented"))
            # the load's constant-impedance elements (`physical_network`) replaced: G goes, the susceptance -Q / V0^2
            # becomes -Qs / V0^2
            pn.G[k] -= P / V02
            abs(pn.G[k]) < 1e-12 && (pn.G[k] = 0.0)
            if Q > 0
                j = findall(==(k), pn.sh_bus)
                length(j) == 1 || throw(ArgumentError("ph_system: bus $b has $(length(j)) shunt inductors"))
                if Qs > 0
                    pn.Lsh[j[1]] = V02 / Qs
                else
                    deleteat!(pn.sh_bus, j[1]); deleteat!(pn.Lsh, j[1])
                    pn.C[k] -= Qs / V02
                end
            else
                pn.C[k] += Qm / V02
            end
            slb[k] = true
            push!(sdat, (; name = "static_bus$b", node = k, bus = b, law, Qs))
        end
        nb, m, ns = pn_sizes(pn)
    end
    Rsh = zeros(ns)
    cutload = Int[]                              # bus rows whose series load leaves an inductor-only node
    if loads in (:series, :series_all)
        # an inductive load: G parallel L becomes R series L with the same P and Q at V0 and
        # nominal frequency, Z = V0^2 / (P - jQ). At a bus without capacitance only with
        # `:series_all`
        for (b, P, Q) in load_power_by_bus(case, pn.net)
            k = bus_index(pn.net, b)
            Q > 0 || continue
            if pn.C[k] == 0 && twG[k] == 0             # (with travelling-wave lines' conductance it is an algebraic node)
                loads == :series_all || continue
                stray_capacitance > 0 ? (pn.C[k] = stray_capacitance) : push!(cutload, k)
            end
            j = findall(==(k), pn.sh_bus)
            length(j) == 1 || throw(ArgumentError("ph_system: bus $b has $(length(j)) shunt inductors"))
            V02 = max(_load_v0(case, pn.net, b, v0)^2, 1e-12)
            z = V02 / complex(P, -Q)
            pn.G[k] -= P / V02
            abs(pn.G[k]) < 1e-12 && (pn.G[k] = 0.0)
            pn.Lsh[j[1]], Rsh[j[1]] = imag(z), real(z)
        end
    end
    gens = [c for c in sys.comps if c isa Union{GENROU_PHTRUE,GENSAL_PHTRUE}]
    gen_buses = unique([bus(c) for c in gens])
    hasgen = falses(nb)
    for b in gen_buses
        hasgen[bus_index(pn.net, b)] = true
    end
    inc = [Int[] for _ in 1:nb]                  # branches at each bus row (travelling-wave lines are not branches)
    for l in 1:m
        istw[l] && continue
        push!(inc[pn.from[l]], l)
        push!(inc[pn.to[l]], l)
    end

    merged, dropped = Int[], copy(istw)
    mdata = NamedTuple[]
    # winding data of the machine(s) at bus b with the merged series x, r: the Sauer-Pai map of
    # the (combined) standard data, or the given circuit
    function wind(b, pp, x_ext, r_ext)
        circuits === nothing && return winding_machine(pp; x_ext, r_ext)
        c = circuits[b]
        return WindingMachine(c.Ll + x_ext, c.Lad, c.Laq, c.Lfd, c.L1d, c.L1q, c.Ra + r_ext, c.Rfd,
                              c.R1d, c.R1q, c.H, sys.omega_b; L2q = c.L2q,
                              R2q = c.L2q === nothing ? 0.0 : c.R2q)
    end
    for b in gen_buses
        k = bus_index(pn.net, b)
        group = [c for c in gens if bus(c) == b]
        p = group[1].p
        all(c -> c.p == p, group) ||
            throw(ArgumentError("ph_system: machines at bus $b differ; the general cutset " *
                                "reduction (spec section 5) is not implemented"))
        p.D == 0 || throw(ArgumentError("ph_system: $(name(group[1])) has D = $(p.D); a shaft " *
                                        "damping has no element here"))
        units = length(group)
        pp = units == 1 ? p : _parallel(p, units)
        nm = join(name.(group), "+")
        if pn.C[k] == 0 && pn.G[k] == 0 && twG[k] == 0 && !slb[k] && !(k in cutload)
            (length(inc[k]) == 1 && !(k in pn.sh_bus)) ||
                throw(ArgumentError("ph_system: generator bus $b has no capacitance or " *
                                    "conductance and is not a single-transformer terminal"))
            l = inc[k][1]
            pn.to[l] == k || throw(ArgumentError("ph_system: $(pn.branch[l]) has the machine " *
                                                 "on its ratio side"))
            hasgen[pn.from[l]] && throw(ArgumentError("ph_system: $(pn.branch[l]) joins two " *
                                                      "generator buses"))
            dropped[l] = true
            push!(merged, k)
            push!(mdata, (; name = nm, w = wind(b, pp, pn.L[l], pn.R[l]),
                          node = pn.from[l], a = pn.a[l], bus = b, trf = pn.branch[l],
                          x_ext = pn.L[l], r_ext = pn.R[l], units))
        else
            push!(mdata, (; name = nm, w = wind(b, pp, 0.0, 0.0), node = k, a = 1.0 + 0im, bus = b,
                          trf = "", x_ext = 0.0, r_ext = 0.0, units))
        end
    end

    # the converters (file header): a source row per converter bus and the branch that ends there
    convs = [c for c in sys.comps if c isa Union{GFM_VSM_PHTRUE,GFM_DROOP_PHTRUE,GFM_VOC_PHTRUE,GFL_PHTRUE}]
    cdata = NamedTuple[]
    for b in unique([bus(c) for c in convs])
        k = bus_index(pn.net, b)
        hasgen[k] && throw(ArgumentError("ph_system: bus $b has a machine and a converter"))
        group = [c for c in convs if bus(c) == b]
        p = group[1].p
        all(c -> c.p == p, group) ||
            throw(ArgumentError("ph_system: the converters at bus $b differ; only identical units are combined"))
        (isfinite(p.Zseries) && p.Zseries > 0) ||
            throw(ArgumentError("ph_system: converter $(name(group[1])) needs a positive Zseries"))
        p.ra == 0 || throw(ArgumentError("ph_system: converter $(name(group[1])) has ra = $(p.ra); the " *
                                         "voltage-source model here has no series resistance (Pcu = 0)"))
        units = length(group)
        Lc = p.Zseries / units
        nm = join(name.(group), "+")
        push!(pn.C, 0.0); push!(pn.G, 0.0); push!(inc, Int[]); push!(twG, 0.0)
        row = length(pn.C)
        merge = pn.C[k] == 0 && pn.G[k] == 0 && twG[k] == 0 && !slb[k] && !(k in cutload) && length(inc[k]) == 1 && !(k in pn.sh_bus)
        if merge
            l = inc[k][1]
            pn.to[l] == k || throw(ArgumentError("ph_system: $(pn.branch[l]) has the converter on its ratio side"))
            (hasgen[pn.from[l]] || pn.from[l] > nb) &&
                throw(ArgumentError("ph_system: $(pn.branch[l]) joins a converter to a generator or converter bus"))
            pn.to[l] = row
            pn.L[l] += Lc
            pn.branch[l] *= "+" * nm
            inc[k] = Int[]
            push!(merged, k)
        else
            push!(pn.from, k); push!(pn.to, row); push!(pn.a, 1.0 + 0im); push!(pn.R, 0.0); push!(pn.L, Lc)
            push!(pn.branch, "conv_" * nm)
            push!(dropped, false); push!(istw, false)
            l = length(pn.R)
            push!(inc[k], l)
        end
        push!(inc[row], l)
        push!(cdata, (; name = nm, bus = b, k, row, l, merged = merge, Lc, units, comp = group[1]))
    end
    m = length(pn.R)

    dyn = [k for k in 1:nb if pn.C[k] > 0]
    alg = [k for k in 1:nb if pn.C[k] == 0 && (pn.G[k] + twG[k] > 0 || slb[k])]
    cut = [k for k in 1:nb if pn.C[k] == 0 && pn.G[k] + twG[k] == 0 && !slb[k] && !(k in merged)]
    issubset(cut, cutload) ||
        throw(ArgumentError("ph_system: a bus without capacitance or conductance is not a " *
                            "generator terminal (an inductor cutset the model does not reduce)"))
    qidx = zeros(Int, nb)
    for (j, k) in enumerate(dyn)
        qidx[k] = 2j - 1
    end
    branches = [l for l in 1:m if !dropped[l]]
    bidx = [2 * length(dyn) + 2j - 1 for j in eachindex(branches)]
    # the load inductor of a reduced node has no state
    sidx, nsk = zeros(Int, ns), 0
    for s in 1:ns
        pn.sh_bus[s] in cut && continue
        nsk += 1
        sidx[s] = 2 * length(dyn) + 2 * length(branches) + 2nsk - 1
    end
    # total charging of each kept branch, and the node capacitance that no line owns
    b_of = Dict(l.idx => l.b for l in case.lines)
    # (a converter's own branch has none; a transformer merged with a converter keeps its own)
    charge = [startswith(pn.branch[l], "conv_") ? 0.0 : b_of[first(split(pn.branch[l], '+'))]
              for l in branches]
    Cown = copy(pn.C)
    for (j, l) in enumerate(branches)
        Cown[pn.from[l]] -= charge[j] / 2
        Cown[pn.to[l]] -= charge[j] / 2
    end
    Cown .= ifelse.(Cown .< 1e-12, 0.0, Cown)

    # the reduced inductor-only nodes
    cuts, cutof, bcut = PHCut[], zeros(Int, nb), zeros(Int, length(branches))
    for k in cut
        b = pn.net.bus_ids[k]
        sh = findall(==(k), pn.sh_bus)
        length(sh) == 1 || throw(ArgumentError("ph_system: bus $b has $(length(sh)) shunt inductors"))
        pos = [j for (j, l) in enumerate(branches) if pn.from[l] == k || pn.to[l] == k]
        mach = [j for (j, d) in enumerate(mdata) if d.node == k]
        length(mach) <= 1 || throw(ArgumentError("ph_system: two machines at the inductor-only bus $b"))
        isempty(pos) && throw(ArgumentError("ph_system: no branch at the inductor-only bus $b"))
        (isempty(mach) || length(pos) == 1) ||
            throw(ArgumentError("ph_system: a machine and $(length(pos)) branches at the inductor-only bus $b"))
        gamma, c, far = Float64[], Float64[], Int[]
        for j in pos
            l = branches[j]
            imag(pn.a[l]) == 0 || throw(ArgumentError("ph_system: $(pn.branch[l]) at the inductor-only " *
                                                      "bus $b has a phase-shifting ratio"))
            a = real(pn.a[l])
            atfrom = pn.from[l] == k
            push!(gamma, atfrom ? 1 / a : -1.0)
            push!(c, atfrom ? -1.0 : 1 / a)
            push!(far, atfrom ? pn.to[l] : pn.from[l])
            (pn.C[far[end]] > 0 || far[end] > nb || twG[far[end]] > 0) ||   # a capacitance, a converter's source, or a
                throw(ArgumentError("ph_system: $(pn.branch[l]) joins the inductor-only " *    # travelling-wave line's
                                    "bus $b to a bus without capacitance"))                    # conductance (algebraic)
        end
        RL, LL = Rsh[sh[1]], pn.Lsh[sh[1]]
        if isempty(mach)
            M = Matrix(Diagonal(pn.L[branches[pos]])) .+ LL .* (gamma * gamma')
            Rm = Matrix(Diagonal(pn.R[branches[pos]])) .+ RL .* (gamma * gamma')
            push!(cuts, PHCut(k, sh[1], RL, LL, pos, gamma, c, far, 0, M, inv(M), Rm))
        else
            d, l = mdata[mach[1]], branches[pos[1]]
            imag(d.a) == 0 || throw(ArgumentError("ph_system: the machine at the inductor-only bus $b " *
                                                  "sits behind a phase-shifting ratio"))
            a, g = real(d.a), gamma[1]
            M = [LL/a^2 g*LL/a; g*LL/a pn.L[l]+g^2*LL]
            Rm = [d.w.Ra+RL/a^2 g*RL/a; g*RL/a pn.R[l]+g^2*RL]
            push!(cuts, PHCut(k, sh[1], RL, LL, pos, gamma, c, far, mach[1], M, inv(M), Rm))
        end
        cutof[k] = length(cuts)
        bcut[pos] .= length(cuts)
    end

    off = 2 * (length(dyn) + length(branches) + nsk) + 1
    machines = PHMachine[]
    for (j, d) in enumerate(mdata)
        ci = cutof[d.node]
        Kd, Kq = ci == 0 ? (d.w.Kd, d.w.Kq) : _extended_K(d.w, cuts[ci].M)
        mc = PHMachine(d.name, d.w, d.node, d.a, d.bus, d.trf, d.x_ext, d.r_ext, d.units, off, ci,
                       ci == 0 ? 0 : bidx[cuts[ci].pos[1]], Kd, Kq)
        push!(machines, mc)
        off += phs_nstates(mc)
    end
    # the components of the energy account, in the order `phs_ledger` fills them
    cnames, ckind = String[], Symbol[]
    for l in branches
        push!(cnames, pn.branch[l])
        push!(ckind, startswith(pn.branch[l], "Trf") ? :transformer :
                     startswith(pn.branch[l], "conv_") ? :converter : :line)
    end
    for k in vcat(dyn, alg, cut)
        (_has_load(pn, Cown, k) || slb[k]) || continue
        push!(cnames, "load_bus$(pn.net.bus_ids[k])")
        push!(ckind, :load)
    end
    push!(cnames, "faults")
    push!(ckind, :fault)
    for mc in machines
        mc.trf == "" || (push!(cnames, mc.trf); push!(ckind, :transformer))
        push!(cnames, mc.name, mc.name * ".rotor")
        push!(ckind, :machine, :rotor)
    end
    for d in mdat                                     # composite loads' motors: windings and shaft with its load torque
        push!(cnames, d.name)
        push!(ckind, :motor)
    end
    # each converter's alpha = 1 + (L_c/omega_b) d(i')/d(u_c), from its branch's row (file header)
    conv = PHConverter[]
    cl = [d.l for d in cdata]
    for d in cdata
        pos = findfirst(==(d.l), branches)
        ci = bcut[pos]
        alpha = if ci == 0
            1 - d.Lc / pn.L[d.l]
        else
            q = cuts[ci]
            # (with a machine the node's branch currents are winding currents in the rotor frame; the
            # formulas here do not hold there: consultation C2, finding 2)
            q.machine == 0 ||
                throw(ArgumentError("ph_system: converter $(d.name) at an inductor-only node with a machine"))
            count(j -> branches[j] in cl, q.pos) == 1 ||
                throw(ArgumentError("ph_system: two converters at the inductor-only bus $(pn.net.bus_ids[q.node])"))
            a = findfirst(==(pos), q.pos)
            1 + q.c[a] * d.Lc * q.Minv[a, a]
        end
        # alpha = 0 at a bus with capacitance or conductance (the terminal voltage does not depend on u_c)
        0 <= alpha < 1 || error("ph_system: converter $(d.name) has alpha = $alpha")
        push!(conv, PHConverter(d.name, d.bus, d.k, d.row, pos, d.merged, d.Lc, alpha, d.units, d.comp))
    end
    # the saturating cores (`cores`): the source loop flux of each after the machines' states, then the frame clock
    pcores = PHCore[]
    if cores !== nothing && !isempty(cores)
        for (b, S) in sort!(collect(cores); by = first)
            i = findfirst(cv -> cv.bus == b, conv)
            i === nothing && throw(ArgumentError("ph_system: a core at bus $b, which has no converter"))
            cv = conv[i]
            cv.merged || throw(ArgumentError("ph_system: a core at bus $b: the converter is not merged with its " *
                                             "transformer (bus 31's load; record section 21, T1 row 4)"))
            l = branches[cv.pos]
            bcut[cv.pos] == 0 ||
                throw(ArgumentError("ph_system: a core at bus $b: its transformer ends at the reduced inductor-only bus " *
                                    "$(pn.net.bus_ids[cuts[bcut[cv.pos]].node]); a core inside a reduced cutset needs the " *
                                    "two inductor-only nodes reduced together (not implemented; record section 24)"))
            (S > 0 && isfinite(S)) || throw(ArgumentError("ph_system: the transformer at bus $b has a rating of $S MVA"))
            LT, RT = pn.L[l] - cv.Lc, pn.R[l]
            LT > 0 || throw(ArgumentError("ph_system: the transformer at bus $b has no leakage inductance"))
            push!(pcores, PHCore(pn.branch[l], b, cv.pos, i, off, cv.Lc + LT / 2, RT / 2, LT / 2, RT / 2, cv.Lc,
                                 Float64(S), core_curve_gmd(S / case.config.mva_base)))
            off += 2
            conv[i] = PHConverter(cv.name, cv.bus, cv.k, cv.row, cv.pos, cv.merged, cv.Lc, NaN, cv.units, cv.comp,
                                  length(pcores))
        end
    end
    iclk = isempty(pcores) ? 0 : off
    isempty(pcores) || (off += 1)
    # composite loads: the motors' states, then the static parts' lag states
    # (with `relays`, review 76: the protected motors' relay states after the lags: sigma, n1, n2, d1, d2)
    relay = relays === nothing ? nothing : MotorRelay(; relays...)
    rtypes = relays === nothing ? () : get(relays, :types, (:A, :B, :C))
    roff = off + 7 * length(mdat) + length(sdat)
    motors = PHMotor[]
    for d in mdat
        prot = relay !== nothing && d.kind in rtypes
        push!(motors, PHMotor(d.name, d.kind, d.node, d.bus, d.c, d.F, d.Pm, d.slip, off, prot ? relay : nothing,
                              prot ? roff : 0))
        prot && (roff += 5)
        off += 7
    end
    statics, sl = PHStatic[], zeros(Int, length(pn.C))
    for d in sdat
        push!(statics, PHStatic(d.name, d.node, d.bus, d.law, d.Qs, off))
        sl[d.node] = length(statics)
        off += 1
    end
    off = roff
    # the zero-axis network (upgrade 4, reviews 80-80b): states after everything else
    z0 = nothing
    if zero !== :none
        zero_pf_json === nothing && throw(ArgumentError("ph_system: zero = $zero needs zero_pf_json (the PowerFactory probe)"))
        zd = zero_sequence_data(case; gen_buses, mode = zero, pf_json = zero_pf_json, zm0_3limb)
        z0, off = _zero_network(pn, zd, off; lines, nsec)
        push!(cnames, "zero_axis"); push!(ckind, :zero)
    end
    asym = (; af = Any[], theta = Ref(0.0), theta_pw = Float64(theta_pw))
    return PHSystem(pn, pn.omega_b, dyn, alg, merged, qidx, branches, bidx, charge, sidx, Cown,
                    Rsh, cut, cuts, cutof, bcut, machines, off - 1, cnames, ckind, conv, pcores, iclk, Ref{Any}(nothing),
                    tw, twG, zeros(ComplexF64, length(twG)), nsec, motors, statics, sl, z0, asym, false, 0.0)
end

# the zero-axis network (review 80b, items 1, 4): zero nodes = bus rows touched by a zero element; lumped lines: R0, L0
# branches with C0 = B0/2 at each end; travelling-wave lines: the ground mode as its own Bergeron line (Norton G0 at both
# ends, no C0); transformer zero shunts (R, L to ground) at their HV buses; a node with no capacitance and no line
# conductance whose only elements are shunts is inductor-only: its shunts carry no current (no path) and are removed
# with it (ground faults there refused)
function _zero_network(pn, zd, off; lines::Symbol, nsec::Int)
    nrow = length(pn.C)
    zrow = zeros(Int, nrow); nodes = Int[]
    addn(k) = (zrow[k] == 0 && (push!(nodes, k); zrow[k] = length(nodes)); zrow[k])
    C0, G0 = Float64[], Float64[]
    grow!() = (while length(C0) < length(nodes); push!(C0, 0.0); push!(G0, 0.0); end)
    br, tw0, sh = NamedTuple[], NamedTuple[], NamedTuple[]
    for zl in zd.lines
        i, j = addn(bus_index(pn.net, zl.from)), addn(bus_index(pn.net, zl.to)); grow!()
        if lines === :travelling
            Zc = sqrt(zl.x0 / zl.b0); Gn = 1 / (Zc + zl.r0 / (4nsec))
            G0[i] += Gn; G0[j] += Gn
            push!(tw0, (; name = zl.idx, from = i, to = j, r = zl.r0, x = zl.x0, b = zl.b0, Zc, G = Gn))
        else
            C0[i] += zl.b0 / 2; C0[j] += zl.b0 / 2
            push!(br, (; name = zl.idx, from = i, to = j, R = zl.r0, L = zl.x0))
        end
    end
    for z in zd.shunts
        k = addn(bus_index(pn.net, z.bus)); grow!()
        push!(sh, (; name = z.idx, node = k, R = z.r, L = z.x, connection = z.connection))
    end
    # inductor-only nodes: no C0, no G0, no line branch
    hasbr = falses(length(nodes))
    for b in br; hasbr[b.from] = true; hasbr[b.to] = true; end
    isolated = [k for k in eachindex(nodes) if C0[k] == 0 && G0[k] == 0 && !hasbr[k]]
    removed = [nodes[k] for k in isolated]
    keep = setdiff(eachindex(nodes), isolated)
    remap = zeros(Int, length(nodes)); remap[keep] = 1:length(keep)
    nodes = nodes[keep]; C0 = C0[keep]; G0 = G0[keep]
    zrow = zeros(Int, nrow); for (q, k) in enumerate(nodes); zrow[k] = q; end
    br = [merge(b, (; from = remap[b.from], to = remap[b.to])) for b in br]
    tw0 = [merge(t, (; from = remap[t.from], to = remap[t.to])) for t in tw0]
    shk = [merge(z, (; node = remap[z.node])) for z in sh if remap[z.node] > 0]
    dyn0 = [q for q in eachindex(nodes) if C0[q] > 0]
    alg0 = [q for q in eachindex(nodes) if C0[q] == 0]
    all(q -> G0[q] > 0, alg0) || throw(ArgumentError("ph_system: a zero node without capacitance or conductance remains"))
    qidx0 = zeros(Int, length(nodes))
    for q in dyn0; qidx0[q] = off; off += 1; end
    bidx0 = Int[]; for _ in br; push!(bidx0, off); off += 1; end
    sidx0 = Int[]; for _ in shk; push!(sidx0, off); off += 1; end
    z0 = (; mode = zd.mode, nodes, zrow, C0, G0, dyn0, alg0, qidx0, br, bidx0, sh = shk, sidx0, tw0,
          twJ0 = zeros(length(nodes)), removed, notes = zd.notes, data = zd)
    return z0, off
end

_has_load(pn::PhysicalNetwork, Cown, k) = pn.G[k] > 0 || Cown[k] > 0 || k in pn.sh_bus

"""    phs_state_names(s) -> Vector{String}"""
function phs_state_names(s::PHSystem)
    ids = s.net.net.bus_ids
    names = String[]
    for k in s.dyn
        push!(names, "q_$(ids[k])_Re", "q_$(ids[k])_Im")
    end
    for (j, l) in enumerate(s.branches)
        ci = s.bcut[j]
        if ci == 0
            push!(names, "phi_$(s.net.branch[l])_Re", "phi_$(s.net.branch[l])_Im")
        elseif s.cuts[ci].machine == 0        # the reduced flux of the node's branches
            push!(names, "Phi_$(s.net.branch[l])_Re", "Phi_$(s.net.branch[l])_Im")
        else                                  # a machine's branch winding, in its rotor frame
            mn = s.machines[s.cuts[ci].machine].name
            push!(names, "$(mn).Phi_$(s.net.branch[l])_d", "$(mn).Phi_$(s.net.branch[l])_q")
        end
    end
    for (j, k) in enumerate(s.net.sh_bus)
        s.sidx[j] > 0 || continue
        push!(names, "phi_sh$(j)_bus$(ids[k])_Re", "phi_sh$(j)_bus$(ids[k])_Im")
    end
    for m in s.machines
        st = m.cut == 0 ? ["lam_d", "lam_q"] : ["Lam_d", "Lam_q"]
        loc = m.w.nq == 2 ? vcat(st, ["lam_fd", "lam_1d", "lam_1q", "lam_2q", "p", "delta"]) :
              vcat(st, ["lam_fd", "lam_1d", "lam_1q", "p", "delta"])
        append!(names, m.name .* "." .* loc)
    end
    for q in s.cores                          # the branch's own pair is the HV loop flux x_H
        push!(names, "phiS_$(q.name)_Re", "phiS_$(q.name)_Im")
    end
    s.iclk > 0 && push!(names, "clock")
    for mo in s.motors
        append!(names, mo.name .* "." .* ["lam_s_Re", "lam_s_Im", "lam_1_Re", "lam_1_Im", "lam_2_Re", "lam_2_Im", "p"])
    end
    for st in s.statics
        push!(names, st.name * ".z")
    end
    for mo in s.motors
        mo.ridx > 0 && append!(names, _relay_names(mo))
    end
    if s.z0 !== nothing
        z = s.z0
        for q in z.dyn0; push!(names, "q0_$(ids[z.nodes[q]])"); end
        for b in z.br; push!(names, "phi0_$(b.name)"); end
        for h in z.sh; push!(names, "phi0_sh_$(h.name)"); end
    end
    return names
end

# the currents (i_H, i_S) of core `k` at the state `x`: the step's discrete-gradient currents while a step is
# evaluated (`s.step`), every core short-circuited for the step rule's bound, or the minimisation at the state
function _core_currents(s::PHSystem, k::Int, x)
    q, st, wb = s.cores[k], s.step[], s.omega_b
    st isa NamedTuple && return st.ibar[k]
    xH, xS = _cx(x, s.bidx[q.pos]), _cx(x, q.xs)
    st === :short && return (wb * xH / q.L2, -wb * xS / q.L1)
    c = core_solve(q, wb * xH, wb * xS, x[s.iclk])
    return (c.iH, c.iS)
end

@inline _cx(x, i) = complex(x[i], x[i+1])

# the winding currents as the linear map of the flux entries of `y` (the state, or a rate of
# it): stator, field, dampers and, at a reduced node, the branch winding `T`
@inline function _winding_currents(m::PHMachine, y)
    w = m.w
    wb = w.omega_b
    o = m.offset
    K, Q = m.Kd, m.Kq
    ld, lq, lfd, l1d, l1q = y[o], y[o+1], y[o+2], y[o+3], y[o+4]
    id = wb * (K[1, 1] * ld + K[1, 2] * lfd + K[1, 3] * l1d)
    ifd = wb * (K[2, 1] * ld + K[2, 2] * lfd + K[2, 3] * l1d)
    i1d = wb * (K[3, 1] * ld + K[3, 2] * lfd + K[3, 3] * l1d)
    if w.nq == 2
        l2q = y[o+5]
        iq = wb * (Q[1, 1] * lq + Q[1, 2] * l1q + Q[1, 3] * l2q)
        i1q = wb * (Q[2, 1] * lq + Q[2, 2] * l1q + Q[2, 3] * l2q)
        i2q = wb * (Q[3, 1] * lq + Q[3, 2] * l1q + Q[3, 3] * l2q)
    else
        iq = wb * (Q[1, 1] * lq + Q[1, 2] * l1q)
        i1q = wb * (Q[2, 1] * lq + Q[2, 2] * l1q)
        i2q = zero(iq)
    end
    iTd, iTq = zero(id), zero(iq)
    if m.tidx > 0
        td, tq, n = y[m.tidx], y[m.tidx+1], w.nq + 2
        id += wb * K[1, 4] * td
        ifd += wb * K[2, 4] * td
        i1d += wb * K[3, 4] * td
        iTd = wb * (K[4, 1] * ld + K[4, 2] * lfd + K[4, 3] * l1d + K[4, 4] * td)
        iq += wb * Q[1, n] * tq
        i1q += wb * Q[2, n] * tq
        iTq = wb * (Q[n, 1] * lq + Q[n, 2] * l1q + Q[n, n] * tq)
        if w.nq == 2
            i2q += wb * Q[3, n] * tq
            iTq += wb * Q[n, 3] * y[o+5]
        end
    end
    return (; id, iq, ifd, i1d, i1q, i2q, iTd, iTq)
end

# currents and speed of one machine from its states
@inline function _machine_currents(m::PHMachine, x)
    c = _winding_currents(m, x)
    delta = x[_idelta(m)]
    rot = complex(cos(delta), sin(delta))                 # e^{j delta}
    inet = -im * rot * complex(c.id, c.iq)
    return (; c..., omega = x[_ip(m)] / (2 * m.w.H), rot, inet)
end

# the same in the stationary frame: the terminal rotation is e^{j (delta + theta_f)}, cf = e^{j theta_f}
@inline function _machine_currents(m::PHMachine, x, cf::Complex)
    c = _winding_currents(m, x)
    delta = x[_idelta(m)]
    rot = complex(cos(delta), sin(delta)) * cf            # e^{j (delta + theta_f)}
    inet = -im * rot * complex(c.id, c.iq)
    return (; c..., omega = x[_ip(m)] / (2 * m.w.H), rot, inet)
end

# the electromagnetic torque on the rotor (motor convention): omega_b (flux_d i_q - flux_q i_d)
# over the stator pair and, at a reduced node, the branch pair. The added inductances are the
# same on both axes, so this is the machine's own torque
@inline function _torque(m::PHMachine, x, c)
    t = x[m.offset] * c.iq - x[m.offset+1] * c.id
    m.tidx > 0 && (t += x[m.tidx] * c.iTq - x[m.tidx+1] * c.iTd)
    return m.w.omega_b * t
end

# the machine's own stator flux linkage per second (pu): without the merged transformer's
# leakage and without what a reduced node adds
@inline function _own_stator_flux(s::PHSystem, m::PHMachine, x, c)
    wb = s.omega_b
    psid, psiq = wb * x[m.offset] - m.x_ext * c.id, wb * x[m.offset+1] - m.x_ext * c.iq
    if m.cut > 0
        M = s.cuts[m.cut].M
        psid -= M[1, 1] * c.id + M[1, 2] * c.iTd
        psiq -= M[1, 1] * c.iq + M[1, 2] * c.iTq
    end
    return psid, psiq
end

"""
    phs_machine_signals(s, x) -> Vector of (; speed, delta, torque, ifd, id, iq, psid, psiq)

Per machine at the state `x`: speed, rotor angle, electromagnetic torque (motor convention,
system base), field current, stator currents, and the machine's own stator flux linkages (pu;
without the merged transformer's leakage and without what a reduced node adds).
"""
function phs_machine_signals(s::PHSystem, x::AbstractVector)
    return map(s.machines) do m
        c = _machine_currents(m, x)
        psid, psiq = _own_stator_flux(s, m, x, c)
        (; speed = c.omega, delta = x[_idelta(m)], torque = _torque(m, x, c), ifd = c.ifd, id = c.id,
         iq = c.iq, psid, psiq)
    end
end

# the zero axis's stored energy: omega_b q0^2 / (2 C0) + omega_b phi^2 / (2 L) (lines' and shunts' fluxes)
function _zero_energy(s::PHSystem, x)
    z = s.z0
    z === nothing && return zero(eltype(x))
    wb = s.omega_b; H = zero(eltype(x))
    for q in z.dyn0; H += wb * x[z.qidx0[q]]^2 / (2 * z.C0[q]); end
    for (b, idx) in zip(z.br, z.bidx0); H += wb * x[idx]^2 / (2 * b.L); end
    for (h, idx) in zip(z.sh, z.sidx0); H += wb * x[idx]^2 / (2 * h.L); end
    return H
end

# composite loads (motor.jl): a motor's fluxes, momentum and currents from the state
# (with a relay the scale sigma of the remaining units, review 76: currents scaled, omega = p / (2 sigma H); sigma = 0: gone)
@inline function _motor_state(mo::PHMotor, x, wb)
    o = mo.offset
    ls, l1, l2 = _cx(x, o), _cx(x, o + 2), _cx(x, o + 4)
    sg = mo.ridx > 0 ? x[mo.ridx] : one(eltype(x))
    is, i1, i2 = motor_currents(mo.c, ls, l1, l2, wb)
    on = !(_fval(sg) == 0)
    omega = on ? x[o+6] / (2 * sg * mo.c.H) : zero(sg)
    return (; ls, l1, l2, p = x[o+6], is = sg * is, i1 = sg * i1, i2 = sg * i2, omega, sg, on)
end

_static_z(st::PHStatic, x) = x[st.zidx]

# a static part's conductance: modulated by its lag only (review 74c)
_static_G(st::PHStatic, x) = static_conductance(st.law, _static_z(st, x))

# the static conductance at bus row k (0 without a static part)
@inline _gstat(s::PHSystem, f, k) = s.sl[k] == 0 ? zero(eltype(f.gs)) : f.gs[s.sl[k]]

"""
    phs_flows(s, x, u) -> (; v, inj, mc, ib, ish, iload)

Bus voltages `v` (network frame; `NaN` at a merged bus, see `phs_terminal_voltage`, and at a
reduced node, see `phs_cut_voltage`), the current `inj` each bus receives from its branches,
shunt inductors and machines (and, at a reduced node, gives to its load, so that it is zero
there), the machine quantities `mc`, the branch currents `ib` (network frame, by position in
`s.branches`), the shunt inductor currents `ish` and the load current `iload` of each reduced
node.
"""
function phs_flows(s::PHSystem, x::AbstractVector{T}, u::PHInput) where {T}
    pn = s.net
    wb = s.omega_b
    nb = length(pn.C)
    v = fill(complex(T(NaN), T(NaN)), nb)
    inj = zeros(Complex{T}, nb)
    for k in s.dyn
        v[k] = wb * _cx(x, s.qidx[k]) / pn.C[k]
    end
    length(u.uc) == length(s.conv) ||
        throw(ArgumentError("phs_flows: $(length(s.conv)) converters and $(length(u.uc)) EMFs in the input"))
    for (i, cv) in enumerate(s.conv)
        v[cv.row] = u.uc[i]
    end
    mc = s.stationary ? [_machine_currents(m, x, u.cf[1]) for m in s.machines] : [_machine_currents(m, x) for m in s.machines]
    ib = Vector{Complex{T}}(undef, length(s.branches))
    for (j, l) in enumerate(s.branches)
        s.bcut[j] == 0 && (ib[j] = wb * _cx(x, s.bidx[j]) / pn.L[l])
    end
    for c in s.cuts
        if c.machine == 0
            for (a, j) in enumerate(c.pos)
                i = zero(Complex{T})
                for (b, jb) in enumerate(c.pos)
                    i += c.Minv[a, b] * _cx(x, s.bidx[jb])
                end
                ib[j] = wb * i
            end
        else
            q = mc[c.machine]
            ib[c.pos[1]] = -im * q.rot * complex(q.iTd, q.iTq)
        end
    end
    # the current each branch delivers at its far (`to`) end: the branch current, or a cored branch's source-side
    # current (the HV side carries the core's magnetising current too)
    isrc = copy(ib)
    for k in eachindex(s.cores)
        iH, iS = _core_currents(s, k, x)
        ib[s.cores[k].pos], isrc[s.cores[k].pos] = iH, iS
    end
    for (j, l) in enumerate(s.branches)
        inj[pn.from[l]] -= ib[j] / conj(pn.a[l])
        inj[pn.to[l]] += isrc[j]
    end
    ish = zeros(Complex{T}, length(pn.sh_bus))
    for (j, k) in enumerate(pn.sh_bus)
        s.sidx[j] > 0 || continue
        ish[j] = wb * _cx(x, s.sidx[j]) / pn.Lsh[j]
        inj[k] -= ish[j]
    end
    for (j, m) in enumerate(s.machines)
        inj[m.node] -= mc[j].inet / conj(m.a)
    end
    for k in eachindex(s.twJ)                        # travelling-wave lines: the history currents
        s.twJ[k] == 0 || (inj[k] += s.twJ[k])
    end
    mo = [_motor_state(mm, x, wb) for mm in s.motors]   # composite loads' motors: the stator current into the motor
    for (j, mm) in enumerate(s.motors)
        inj[mm.node] -= mo[j].is
    end
    iload = [inj[c.node] for c in s.cuts]            # Kirchhoff's current law at the reduced node
    for (ci, c) in enumerate(s.cuts)
        u.gf[c.node] == 0 ||
            throw(ArgumentError("phs_flows: a fault conductance at bus $(pn.net.bus_ids[c.node]), whose " *
                                "inductor-only node was reduced; the reduced equations have no place for it"))
        inj[c.node] -= iload[ci]
    end
    gs = T[_static_G(st, x) for st in s.statics]    # each static part's conductance (its lag's)
    F = _fault_matrices(s)                           # phase-asymmetric faults at this evaluation's theta (row => G)
    for k in s.alg
        haskey(F, k) && continue                     # (solved with the fault below)
        v[k] = inj[k] / (pn.G[k] + s.twG[k] + u.gf[k] + (s.sl[k] == 0 ? 0 : gs[s.sl[k]]))
    end
    z = s.z0
    if z === nothing && isempty(F)
        return (; v, inj, mc, ib, ish, iload, isrc, mo, gs, z0 = nothing)
    end
    return (; v, inj, mc, ib, ish, iload, isrc, mo, gs, z0 = _zero_flows!(v, s, x, u, inj, gs, F))
end

# the faults in force as (Re, Im, 0) conductances at the current frame angle: row => (G 3 x 3, ground)
function _fault_matrices(s::PHSystem)
    isempty(s.asym.af) && return Dict{Int,Any}()
    return Dict{Int,Any}(f.row => (G = fault_gdq0(f.Gabc, _park_angle(s)), ground = f.ground) for f in s.asym.af)
end

# the dq0 transform's angle (review 81, item 1): phase a is Re(e^{j theta} x) + x0 / sqrt(2) of the network quantity. In
# the synchronous frame theta = omega_b t + theta_pw; in the stationary frame (x_stat = x_sync e^{j (omega_b t + theta0)})
# theta = theta_pw - theta0, a constant
@inline _park_angle(s::PHSystem) = s.stationary ? s.asym.theta_pw - s.theta0 : s.asym.theta[]

# the zero axis and the asymmetric faults (review 80b): zero-node voltages, injections, branch / shunt currents; the
# fault currents (dq and zero) and the voltages of the algebraic buses carrying a fault; v0 at buses without a zero node
# recovered from the fault (NaN where undetermined)
function _zero_flows!(v, s::PHSystem, x::AbstractVector{T}, u, inj, gs, F) where {T}
    pn, wb, z = s.net, s.omega_b, s.z0
    nz = z === nothing ? 0 : length(z.nodes)
    v0 = zeros(T, nz); inj0 = zeros(T, nz)
    i0br = T[]; i0sh = T[]
    if z !== nothing
        for q in z.dyn0
            v0[q] = wb * x[z.qidx0[q]] / z.C0[q]
        end
        for (b, idx) in zip(z.br, z.bidx0)
            i = wb * x[idx] / b.L; push!(i0br, i)
            inj0[b.from] -= i; inj0[b.to] += i
        end
        for (h, idx) in zip(z.sh, z.sidx0)
            i = wb * x[idx] / h.L; push!(i0sh, i)
            inj0[h.node] -= i
        end
        inj0 .+= z.twJ0
    end
    fdq = Dict{Int,Complex{T}}(); f0 = Dict{Int,T}(); v0bus = Dict{Int,T}()
    isalg0 = q -> z !== nothing && q > 0 && z.C0[q] == 0
    for (k, fk) in F
        G = fk.G
        q = z === nothing ? 0 : z.zrow[k]
        algdq = s.qidx[k] == 0
        gsc = pn.G[k] + s.twG[k] + u.gf[k] + (s.sl[k] == 0 ? zero(T) : gs[s.sl[k]])
        if q == 0
            # no zero node: i_0 = 0; v0 eliminated (ground faults: the Schur complement; phase faults: no coupling)
            Gr = fk.ground ? fault_schur(G) : G[1:2, 1:2]
            if algdq
                M = [gsc+Gr[1, 1] Gr[1, 2]; Gr[2, 1] gsc+Gr[2, 2]]
                w = M \ [real(inj[k]), imag(inj[k])]
                v[k] = complex(w[1], w[2])
            end
            vd = [real(v[k]), imag(v[k])]
            id = Gr * vd
            fdq[k] = complex(id[1], id[2])
            v0bus[k] = fk.ground && G[3, 3] > 0 ? -(G[3, 1] * vd[1] + G[3, 2] * vd[2]) / G[3, 3] : T(NaN)
        else
            # unknowns: the dq pair if the bus is algebraic, v0 if its zero node is algebraic; the rest known
            a0 = isalg0(q)
            if algdq || a0
                vd = [real(v[k]), imag(v[k])]
                if algdq && a0
                    M = [gsc+G[1, 1] G[1, 2] G[1, 3]; G[2, 1] gsc+G[2, 2] G[2, 3]; G[3, 1] G[3, 2] z.G0[q]+G[3, 3]]
                    w = M \ [real(inj[k]), imag(inj[k]), inj0[q]]
                    v[k] = complex(w[1], w[2]); v0[q] = w[3]
                elseif algdq
                    M = [gsc+G[1, 1] G[1, 2]; G[2, 1] gsc+G[2, 2]]
                    w = M \ ([real(inj[k]), imag(inj[k])] .- G[1:2, 3] .* v0[q])
                    v[k] = complex(w[1], w[2])
                else
                    v0[q] = (inj0[q] - G[3, 1] * vd[1] - G[3, 2] * vd[2]) / (z.G0[q] + G[3, 3])
                end
            end
            w3 = G * [real(v[k]), imag(v[k]), v0[q]]
            fdq[k] = complex(w3[1], w3[2]); f0[k] = w3[3]; v0bus[k] = v0[q]
        end
    end
    if z !== nothing
        for q in z.alg0                             # algebraic zero nodes without a fault
            k = z.nodes[q]
            haskey(F, k) && continue
            v0[q] = inj0[q] / z.G0[q]
        end
    end
    return (; v0, inj0, i0br, i0sh, fdq, f0, v0bus)
end

# the load current of reduced node `ci` and its rate along `d`, in the frame its states are
# kept in: the network frame without a machine, the machine's rotor frame with one (where the
# rate is that of the rotor-frame components)
function _cut_load_rate(s::PHSystem, ci::Int, f, d)
    c = s.cuts[ci]
    if c.machine == 0
        di = zero(f.iload[ci]) + zero(eltype(d))
        for (a, _) in enumerate(c.pos), (b, jb) in enumerate(c.pos)
            di -= c.gamma[a] * s.omega_b * c.Minv[a, b] * _cx(d, s.bidx[jb])
        end
        return f.iload[ci], di
    end
    m, q = s.machines[c.machine], f.mc[c.machine]
    r = _winding_currents(m, d)
    g, a = c.gamma[1], real(m.a)
    return -(g * complex(q.iTd, q.iTq) + complex(q.id, q.iq) / a),
           -(g * complex(r.iTd, r.iTq) + complex(r.id, r.iq) / a)
end

# voltage of reduced node `ci` from its load's own equation,
# v = (L_L/omega_b) i_L' + R_L i_L + j L_L i_L, with the current rates along `dx`
function _cut_voltage(s::PHSystem, ci::Int, f, dx)
    c = s.cuts[ci]
    iL, diL = _cut_load_rate(s, ci, f, dx)
    if c.machine > 0                                  # from the rotor frame to the network frame
        m, q = s.machines[c.machine], f.mc[c.machine]
        # the rotation's rate: delta' in the synchronous frame, delta' + omega_b in the stationary frame
        diL = s.stationary ? -im * q.rot * (diL + im * (dx[_idelta(m)] + s.omega_b) * iL) :
                             -im * q.rot * (diL + im * dx[_idelta(m)] * iL)
    end
    s.stationary && return c.LL * diL / s.omega_b + c.RL * f.iload[ci]   # no frame term
    return c.LL * diL / s.omega_b + complex(c.RL, c.LL) * f.iload[ci]
end

# the voltage of bus row `k`: its state or eliminated value, or recovered at a reduced node
@inline _node_voltage(s::PHSystem, k::Int, f, dx) =
    s.cutof[k] == 0 ? f.v[k] : _cut_voltage(s, s.cutof[k], f, dx)

# the rate di/dt of converter `cv`'s branch current (towards its source row) along `dx`
function _conv_rate(s::PHSystem, cv::PHConverter, dx)
    j, wb = cv.pos, s.omega_b
    ci = s.bcut[j]
    ci == 0 && return wb * _cx(dx, s.bidx[j]) / s.net.L[s.branches[j]]
    q = s.cuts[ci]
    a = findfirst(==(j), q.pos)
    return wb * sum(q.Minv[a, b] * _cx(dx, s.bidx[jb]) for (b, jb) in enumerate(q.pos))
end

# the converter's terminal voltage, v_k = u_c + j L_c i + (L_c/omega_b) di/dt, with the rate along `dx`
function _conv_terminal(s::PHSystem, cv::PHConverter, f, dx)
    cv.core == 0 || throw(ArgumentError("_conv_terminal: converter $(cv.name) has a cored transformer; pass the state"))
    return f.v[cv.row] + im * cv.Lc * f.ib[cv.pos] + cv.Lc * _conv_rate(s, cv, dx) / s.omega_b
end

# the same with the state `x`, for any converter: with a cored transformer the current is the source loop's i_S and its
# rate the directional derivative of i_S(x_H, x_S, theta) along `dx` (the implicit derivative of the core flux)
function _conv_terminal(s::PHSystem, cv::PHConverter, x, f, dx)
    cv.core == 0 && return _conv_terminal(s, cv, f, dx)
    q, wb = s.cores[cv.core], s.omega_b
    xH, xS, th = _cx(x, s.bidx[q.pos]), _cx(x, q.xs), x[s.iclk]
    dH, dS, dth = _cx(dx, s.bidx[q.pos]), _cx(dx, q.xs), dx[s.iclk]
    di = ForwardDiff.derivative(t -> core_solve(q, wb * (xH + t * dH), wb * (xS + t * dS), th + t * dth).iS, 0.0)
    return f.v[cv.row] + im * cv.Lc * f.isrc[cv.pos] + cv.Lc * di / wb
end

# the converter's terminal voltage at the state `x` with flows `f` there (the EMF `f.v[cv.row]`): `alpha u_c + w`, or
# with a cored transformer from the branch's own rows (the HV voltage and the EMF are in `f`)
function _conv_vk(s::PHSystem, cv::PHConverter, x, f)
    cv.core == 0 && return cv.alpha * f.v[cv.row] + _conv_w(s, cv, x, f)
    q, pn, wb = s.cores[cv.core], s.net, s.omega_b
    l, j = s.branches[q.pos], q.pos
    d = zeros(eltype(x), length(x))
    dH = f.v[pn.from[l]] / pn.a[l] - q.R2 * f.ib[j] - im * wb * _cx(x, s.bidx[j])
    dS = f.v[pn.to[l]] + q.R1 * f.isrc[j] - im * wb * _cx(x, q.xs)
    d[s.bidx[j]], d[s.bidx[j]+1], d[q.xs], d[q.xs+1], d[s.iclk] = real(dH), imag(dH), real(dS), imag(dS), wb
    return _conv_terminal(s, cv, x, f, d)
end

# the current a converter injects into the network
@inline _conv_current(f, cv::PHConverter) = -f.isrc[cv.pos]

# w of v_k = alpha u_c + w: the terminal voltage with the converter's own EMF set to zero, from its
# branch's row alone (the flows `f` at the state `x`; no other converter's EMF enters that row,
# `ph_system` makes sure)
function _conv_w(s::PHSystem, cv::PHConverter, x, f)
    pn, wb, j = s.net, s.omega_b, cv.pos
    vz(k) = k == cv.row ? zero(f.v[k]) : f.v[k]
    ci = s.bcut[j]
    if ci == 0
        l = s.branches[j]
        di = wb * (vz(pn.from[l]) / pn.a[l] - pn.R[l] * f.ib[j] - im * pn.L[l] * f.ib[j]) / pn.L[l]
    else
        q = s.cuts[ci]
        a = findfirst(==(j), q.pos)
        di = zero(f.ib[j])
        for (b, jb) in enumerate(q.pos)
            d = q.c[b] * vz(q.far[b]) - im * wb * _cx(x, s.bidx[jb])
            for (e, je) in enumerate(q.pos)
                d -= q.Rm[b, e] * f.ib[je]
            end
            di += wb * q.Minv[a, b] * d
        end
    end
    return im * cv.Lc * f.ib[j] + cv.Lc * di / wb
end

"""
    phs_rhs!(dx, s, x, u) -> dx

The vector field of the model (file header). Generic in the element type of `x`.
"""
function phs_rhs!(dx, s::PHSystem, x::AbstractVector, u::PHInput)
    pn = s.net
    wb = s.omega_b
    f = phs_flows(s, x, u)
    v = f.v
    for k in s.dyn
        g = pn.G[k] + s.twG[k] + u.gf[k] + _gstat(s, f, k)
        d = s.stationary ? f.inj[k] - g * v[k] : f.inj[k] - g * v[k] - im * pn.C[k] * v[k]
        f.z0 === nothing || (d -= get(f.z0.fdq, k, zero(d)))     # a phase-asymmetric fault at the bus
        dx[s.qidx[k]], dx[s.qidx[k]+1] = real(d), imag(d)
    end
    for (j, l) in enumerate(s.branches)
        s.bcut[j] == 0 || continue
        i = f.ib[j]
        d = s.stationary ? v[pn.from[l]] / pn.a[l] - v[pn.to[l]] - pn.R[l] * i :
                           v[pn.from[l]] / pn.a[l] - v[pn.to[l]] - pn.R[l] * i - im * pn.L[l] * i
        dx[s.bidx[j]], dx[s.bidx[j]+1] = real(d), imag(d)
    end
    for q in s.cores                                   # a cored branch: the HV and source loops (transformer.jl)
        l, j = s.branches[q.pos], q.pos
        dH = v[pn.from[l]] / pn.a[l] - q.R2 * f.ib[j] - im * wb * _cx(x, s.bidx[j])
        dS = v[pn.to[l]] + q.R1 * f.isrc[j] - im * wb * _cx(x, q.xs)
        dx[s.bidx[j]], dx[s.bidx[j]+1] = real(dH), imag(dH)
        dx[q.xs], dx[q.xs+1] = real(dS), imag(dS)
    end
    s.iclk > 0 && (dx[s.iclk] = wb)
    for c in s.cuts                                    # the branches of a node without a machine
        c.machine == 0 || continue
        for (a, j) in enumerate(c.pos)
            d = s.stationary ? c.c[a] * v[c.far[a]] :
                               c.c[a] * v[c.far[a]] - im * wb * _cx(x, s.bidx[j])
            for (b, jb) in enumerate(c.pos)
                d -= c.Rm[a, b] * f.ib[jb]
            end
            dx[s.bidx[j]], dx[s.bidx[j]+1] = real(d), imag(d)
        end
    end
    for (j, k) in enumerate(pn.sh_bus)
        s.sidx[j] > 0 || continue
        i = f.ish[j]
        d = s.stationary ? v[k] - s.Rsh[j] * i : v[k] - s.Rsh[j] * i - im * pn.Lsh[j] * i
        dx[s.sidx[j]], dx[s.sidx[j]+1] = real(d), imag(d)
    end
    for (j, m) in enumerate(s.machines)
        w, c, o = m.w, f.mc[j], m.offset
        if m.cut == 0
            vdq = im * conj(c.rot) * v[m.node] / m.a
            dx[o] = real(vdq) - w.Ra * c.id + wb * c.omega * x[o+1]
            dx[o+1] = imag(vdq) - w.Ra * c.iq - wb * c.omega * x[o]
        else                                           # the stator and branch pairs of a reduced node
            q, t = s.cuts[m.cut], m.tidx
            vT = q.c[1] * im * conj(c.rot) * v[q.far[1]]
            dx[o] = -q.Rm[1, 1] * c.id - q.Rm[1, 2] * c.iTd + wb * c.omega * x[o+1]
            dx[o+1] = -q.Rm[1, 1] * c.iq - q.Rm[1, 2] * c.iTq - wb * c.omega * x[o]
            dx[t] = real(vT) - q.Rm[2, 1] * c.id - q.Rm[2, 2] * c.iTd + wb * c.omega * x[t+1]
            dx[t+1] = imag(vT) - q.Rm[2, 1] * c.iq - q.Rm[2, 2] * c.iTq - wb * c.omega * x[t]
        end
        dx[o+2] = u.vf[j] - w.Rfd * c.ifd
        dx[o+3] = -w.R1d * c.i1d
        dx[o+4] = -w.R1q * c.i1q
        w.nq == 2 && (dx[o+5] = -w.R2q * c.i2q)
        dx[_ip(m)] = _shaft_torque(u, j, c.omega) + _torque(m, x, c)
        dx[_idelta(m)] = wb * (c.omega - 1)
    end
    for (j, mm) in enumerate(s.motors)                 # composite loads' motors (motor.jl)
        q, o = f.mo[j], mm.offset
        mm.ridx > 0 && (dx[mm.ridx:mm.ridx+4] .= 0)    # relay states: changed only at the samples
        if !q.on                                       # every unit tripped: the motor has left the system
            dx[o:o+6] .= 0
            continue
        end
        # the unit's equations (sigma-free fluxes), the momentum of the remaining units
        dls, dl1, dl2, dp, _ = motor_rates(mm.c, q.ls, q.l1, q.l2, q.p / q.sg, v[mm.node]; omega_b = wb,
                                           stationary = s.stationary)
        dx[o], dx[o+1], dx[o+2], dx[o+3], dx[o+4], dx[o+5] = real(dls), imag(dls), real(dl1), imag(dl1), real(dl2), imag(dl2)
        dx[o+6] = q.sg * dp
    end
    for st in s.statics                                # their static parts' lags
        dx[st.zidx] = static_lag_rate(st.law, _vabs(v[st.node]), _static_z(st, x))
    end
    z = s.z0
    if z !== nothing                                   # the zero axis (no rotation term in either frame)
        zf = f.z0
        for q in z.dyn0
            k = z.nodes[q]
            dx[z.qidx0[q]] = zf.inj0[q] - z.G0[q] * zf.v0[q] - get(zf.f0, k, zero(zf.v0[q]))
        end
        for (j, (b, idx)) in enumerate(zip(z.br, z.bidx0))
            dx[idx] = zf.v0[b.from] - zf.v0[b.to] - b.R * zf.i0br[j]
        end
        for (j, (h, idx)) in enumerate(zip(z.sh, z.sidx0))
            dx[idx] = zf.v0[h.node] - h.R * zf.i0sh[j]
        end
    end
    return dx
end

"""
    phs_rhs(s, x, u) -> dx

The vector field of the EMT model `s` at state `x` and input `u` (allocating form of `phs_rhs!`).
"""
phs_rhs(s::PHSystem, x::AbstractVector{T}, u::PHInput{S}) where {T,S} =
    phs_rhs!(Vector{promote_type(T, S)}(undef, s.n), s, x, u)

# the magnetic energy of a machine's windings (with the branch winding of a reduced node):
# half the sum of flux times current
@inline function _magnetic_energy(m::PHMachine, x, c)
    o = m.offset
    e = x[o] * c.id + x[o+1] * c.iq + x[o+2] * c.ifd + x[o+3] * c.i1d + x[o+4] * c.i1q
    m.w.nq == 2 && (e += x[o+5] * c.i2q)
    m.tidx > 0 && (e += x[m.tidx] * c.iTd + x[m.tidx+1] * c.iTq)
    return e / 2
end

"""    phs_energy(s, x) -> the total stored energy `H(x)` (pu*s)."""
function phs_energy(s::PHSystem, x::AbstractVector)
    pn = s.net
    wb = s.omega_b
    H = zero(eltype(x))
    for k in s.dyn
        H += wb * abs2(_cx(x, s.qidx[k])) / (2 * pn.C[k])
    end
    cored = Set(q.pos for q in s.cores)
    for (j, l) in enumerate(s.branches)
        (s.bcut[j] == 0 && !(j in cored)) && (H += wb * abs2(_cx(x, s.bidx[j])) / (2 * pn.L[l]))
    end
    for q in s.cores
        H += core_energy(q, wb * _cx(x, s.bidx[q.pos]), wb * _cx(x, q.xs), x[s.iclk]) / wb
    end
    for c in s.cuts                                    # omega_b Phi' inv(M) Phi / 2
        c.machine == 0 || continue
        for (a, j) in enumerate(c.pos), (b, jb) in enumerate(c.pos)
            H += wb * c.Minv[a, b] * real(conj(_cx(x, s.bidx[j])) * _cx(x, s.bidx[jb])) / 2
        end
    end
    for j in eachindex(pn.sh_bus)
        s.sidx[j] > 0 && (H += wb * abs2(_cx(x, s.sidx[j])) / (2 * pn.Lsh[j]))
    end
    for m in s.machines
        H += _magnetic_energy(m, x, _machine_currents(m, x)) + x[_ip(m)]^2 / (4 * m.w.H)
    end
    for mo in s.motors
        q = _motor_state(mo, x, wb)
        q.on && (H += q.sg * motor_energy(mo.c, q.ls, q.l1, q.l2, q.p / q.sg, wb))
    end
    H += _zero_energy(s, x)
    return H
end

"""
    phs_gradient(s, x) -> grad H(x)

The efforts: bus voltages, branch and shunt currents, machine currents, speeds, and zero on every
angle.
"""
function phs_gradient(s::PHSystem, x::AbstractVector)
    pn = s.net
    wb = s.omega_b
    z = zeros(eltype(x), s.n)
    for k in s.dyn
        z[s.qidx[k]], z[s.qidx[k]+1] = wb * x[s.qidx[k]] / pn.C[k], wb * x[s.qidx[k]+1] / pn.C[k]
    end
    for (j, l) in enumerate(s.branches)
        s.bcut[j] == 0 || continue
        z[s.bidx[j]], z[s.bidx[j]+1] = wb * x[s.bidx[j]] / pn.L[l], wb * x[s.bidx[j]+1] / pn.L[l]
    end
    for k in eachindex(s.cores)                        # (i_H, -i_S), and the energy's dependence on the clock (23.1)
        q = s.cores[k]
        xH, xS = _cx(x, s.bidx[q.pos]), _cx(x, q.xs)
        c = core_solve(q, wb * xH, wb * xS, x[s.iclk])
        z[s.bidx[q.pos]], z[s.bidx[q.pos]+1] = real(c.iH), imag(c.iH)
        z[q.xs], z[q.xs+1] = -real(c.iS), -imag(c.iS)
        z[s.iclk] += core_rotation_rate(c.iH, c.iS, xH, xS)
    end
    for c in s.cuts
        c.machine == 0 || continue
        for (a, j) in enumerate(c.pos), (b, jb) in enumerate(c.pos)
            z[s.bidx[j]] += wb * c.Minv[a, b] * x[s.bidx[jb]]
            z[s.bidx[j]+1] += wb * c.Minv[a, b] * x[s.bidx[jb]+1]
        end
    end
    for j in eachindex(pn.sh_bus)
        s.sidx[j] > 0 || continue
        z[s.sidx[j]], z[s.sidx[j]+1] = wb * x[s.sidx[j]] / pn.Lsh[j], wb * x[s.sidx[j]+1] / pn.Lsh[j]
    end
    for m in s.machines
        c, o = _machine_currents(m, x), m.offset
        z[o], z[o+1], z[o+2], z[o+3], z[o+4] = c.id, c.iq, c.ifd, c.i1d, c.i1q
        m.w.nq == 2 && (z[o+5] = c.i2q)
        m.tidx > 0 && ((z[m.tidx], z[m.tidx+1]) = (c.iTd, c.iTq))
        z[_ip(m)] = c.omega
    end
    if s.z0 !== nothing                                # the zero axis: v0 and the currents
        zz = s.z0
        for q in zz.dyn0; z[zz.qidx0[q]] = wb * x[zz.qidx0[q]] / zz.C0[q]; end
        for (b, idx) in zip(zz.br, zz.bidx0); z[idx] = wb * x[idx] / b.L; end
        for (h, idx) in zip(zz.sh, zz.sidx0); z[idx] = wb * x[idx] / h.L; end
    end
    for mo in s.motors                                 # currents and speed; the lags have no energy
        q, o = _motor_state(mo, x, wb), mo.offset
        z[o], z[o+1], z[o+2], z[o+3], z[o+4], z[o+5] = real(q.is), imag(q.is), real(q.i1), imag(q.i1), real(q.i2), imag(q.i2)
        z[o+6] = q.omega
    end
    return z
end

"""
    phs_matrices(s, x, u) -> (; J, R, g, z)

The structure of the model at `x`, built independently of `phs_rhs!`: skew `J(x)`, symmetric
positive semidefinite `R(x)`, the input vector `g` (shaft torques, which depend on the speed
when the shaft input is a power, field voltages and the frame clock `-omega_b` on every angle) and the gradient `z`, with `x' = (J - R) z + g`. `R` holds the
element resistances and conductances and, for the eliminated nodes, `N diag(1/G) N'` with `N`
the incidence of the elements at such a node. At a reduced inductor-only node `J` holds the
frame rotation `-j M` of its coupled branches (or the speed-voltage pair of a machine's branch
winding) and `R` the matrix `Rm`.
"""
function phs_matrices(s::PHSystem, x::AbstractVector{Float64}, u::PHInput)
    isempty(s.tw) || throw(ArgumentError("phs_matrices: travelling-wave lines: their history input is not represented"))
    isempty(s.motors) && isempty(s.statics) || throw(ArgumentError("phs_matrices: composite loads are not represented"))
    s.z0 === nothing || throw(ArgumentError("phs_matrices: the zero axis is not represented"))
    isempty(s.asym.af) || throw(ArgumentError("phs_matrices: phase-asymmetric faults are not represented"))
    isempty(s.conv) || throw(ArgumentError("phs_matrices: converters are not implemented here"))
    pn = s.net
    wb = s.omega_b
    n = s.n
    J, R, g = zeros(n, n), zeros(n, n), zeros(n)
    acol = Dict(k => 2j - 1 for (j, k) in enumerate(s.alg))
    N = zeros(n, 2 * length(s.alg))
    # a complex coefficient acting on a complex variable, as a 2x2 real block
    function cblock!(M, r, c, zc)
        M[r, c] += real(zc); M[r, c+1] -= imag(zc)
        M[r+1, c] += imag(zc); M[r+1, c+1] += real(zc)
    end
    # element rows `r` driven by `coef * v_k`; the node receives `-conj(coef) * effort`
    function attach!(r, k, coef)
        if s.qidx[k] > 0
            cblock!(J, r, s.qidx[k], coef)
            cblock!(J, s.qidx[k], r, -conj(coef))
        else
            cblock!(N, r, acol[k], coef)
        end
    end
    for k in s.dyn
        q = s.qidx[k]
        s.stationary || cblock!(J, q, q, -im * pn.C[k])          # the frame term (none in the stationary frame)
        R[q, q] = R[q+1, q+1] = pn.G[k] + u.gf[k]
    end
    for (j, l) in enumerate(s.branches)
        s.bcut[j] == 0 || continue
        r = s.bidx[j]
        s.stationary || cblock!(J, r, r, -im * pn.L[l])
        R[r, r] = R[r+1, r+1] = pn.R[l]
        attach!(r, pn.from[l], 1 / pn.a[l])
        attach!(r, pn.to[l], -1.0 + 0im)
    end
    for c in s.cuts
        c.machine == 0 || continue
        for (a, j) in enumerate(c.pos)
            r = s.bidx[j]
            for (b, jb) in enumerate(c.pos)
                rb = s.bidx[jb]
                s.stationary || cblock!(J, r, rb, -im * c.M[a, b])
                R[r, rb] += c.Rm[a, b]
                R[r+1, rb+1] += c.Rm[a, b]
            end
            attach!(r, c.far[a], complex(c.c[a]))
        end
    end
    for (j, k) in enumerate(pn.sh_bus)
        s.sidx[j] > 0 || continue
        r = s.sidx[j]
        s.stationary || cblock!(J, r, r, -im * pn.Lsh[j])
        R[r, r] = R[r+1, r+1] = s.Rsh[j]
        attach!(r, k, 1.0 + 0im)
    end
    for (j, m) in enumerate(s.machines)
        w, o, ip, id = m.w, m.offset, _ip(m), _idelta(m)
        delta = x[id]
        # the terminal rotation's conjugate e^{-j delta} (stationary frame: e^{-j (delta + theta_f)})
        crot = s.stationary ? conj(complex(cos(delta), sin(delta)) * u.cf[1]) : complex(cos(delta), -sin(delta))
        J[o, ip] += wb * x[o+1]; J[ip, o] -= wb * x[o+1]
        J[o+1, ip] -= wb * x[o]; J[ip, o+1] += wb * x[o]
        J[id, ip] += wb; J[ip, id] -= wb
        if m.cut == 0
            attach!(o, m.node, im * crot / m.a)
            R[o, o] = R[o+1, o+1] = w.Ra
        else
            q, t = s.cuts[m.cut], m.tidx
            attach!(t, q.far[1], q.c[1] * im * crot)
            J[t, ip] += wb * x[t+1]; J[ip, t] -= wb * x[t+1]
            J[t+1, ip] -= wb * x[t]; J[ip, t+1] += wb * x[t]
            R[o, o] = R[o+1, o+1] = q.Rm[1, 1]
            R[t, t] = R[t+1, t+1] = q.Rm[2, 2]
            R[o, t] = R[t, o] = R[o+1, t+1] = R[t+1, o+1] = q.Rm[1, 2]
        end
        R[o+2, o+2], R[o+3, o+3], R[o+4, o+4] = w.Rfd, w.R1d, w.R1q
        w.nq == 2 && (R[o+5, o+5] = w.R2q)
        g[o+2] = u.vf[j]
        g[ip] = _shaft_torque(u, j, x[ip] / (2 * w.H))
        g[id] = -wb
    end
    for (j, k) in enumerate(s.alg)
        gk = pn.G[k] + u.gf[k]
        c = 2j - 1
        R .+= (N[:, c] * N[:, c]' .+ N[:, c+1] * N[:, c+1]') ./ gk
    end
    return (; J, R, g, z = phs_gradient(s, x))
end

"""
    phs_ledger(s, x, u; dx = nothing) -> NamedTuple

The energy account at `x`, by component. `names`, `kind` (`:line`, `:transformer`, `:load`,
`:machine`, `:rotor`, and one `:fault` entry for all fault conductances) and, per component: stored `energy`, `loss`, the power `port` it
receives through its ports (network ports; for a machine also the field, less the air-gap power
it passes to its rotor; for a rotor the shaft and the air-gap power), and the `rate` of its
energy from the vector field. Each satisfies `rate = port - loss`. `tport` is the sum of the
absolute values of its individual port powers (the integrand of its throughput). With `dx` given,
`rate` is `grad H_c(x)' dx` for that `dx` in place of the vector field; ports and losses are
unchanged. Every component energy is a quadratic form, so with `x` a step's midpoint and
`dx = (x1 - x0)/h` this is the exact energy increment of the step over `h`, free of the
cancellation in `H_c(x1) - H_c(x0)`.

Which balances are independent evidence: the series branches, shunt inductors, rotors and the
windings together with their merged transformer. The merged transformer's port power is defined
as its rate plus its loss, a fault's and a conductance's as its loss, and the capacitors at a
node share that node's balance in proportion to their capacitance. At a reduced inductor-only
node the voltage is recovered from the load's equation, so the load's balance holds by
construction, and those of the node's branches and of its machine's windings are evidence.
`supply` is the boundary power (all shaft and field power), `interconnection` the sum of the
internal port powers (zero), `H` the total energy and `dH` its rate.

Ownership: a line owns its series inductor and half its charging at each end; a load owns its
conductance, its shunt inductor and its capacitor; a merged transformer owns `x_ext`, `r_ext` of
the stator they share; a machine owns its windings; a rotor owns the kinetic energy and converts
between the air-gap power `Tem omega` and the shaft. At a reduced node each element still owns
its own inductor and resistor: their energies and losses are computed from the currents.
"""
function phs_ledger(s::PHSystem, x::AbstractVector{Float64}, u::PHInput; dx = nothing)
    pn = s.net
    wb = s.omega_b
    f = phs_flows(s, x, u)
    v = f.v
    dxf = phs_rhs(s, x, u)                    # the vector field: ports that need a rate use it
    for ci in eachindex(s.cuts)
        v[s.cuts[ci].node] = _cut_voltage(s, ci, f, dxf)
    end
    stepinc = dx !== nothing && s.step[] isa NamedTuple   # a step's account: cored branches take the step's increment
    dx = dx === nothing ? dxf : dx            # what the rates are taken along
    nc = length(s.cnames)
    energy, loss, port, rate, tport = zeros(nc), zeros(nc), zeros(nc), zeros(nc), zeros(nc)
    c_ = Ref(0)
    add!(e, ls, pt, rt, tp) = (c = (c_[] += 1); energy[c] = e; loss[c] = ls; port[c] = pt;
                               rate[c] = rt; tport[c] = tp)
    # per unit of capacitance at each node with a state: stored energy, power received, rate
    ecap, pcap, rcap = zeros(length(v)), zeros(length(v)), zeros(length(v))
    for k in s.dyn
        ecap[k] = abs2(v[k]) / (2 * wb)
        ifk = f.z0 === nothing ? zero(v[k]) : get(f.z0.fdq, k, zero(v[k]))   # a phase-asymmetric fault's dq current
        pcap[k] = real(conj(v[k]) * (f.inj[k] - (pn.G[k] + s.twG[k] + u.gf[k] + _gstat(s, f, k)) * v[k] - ifk)) / pn.C[k]
        rcap[k] = real(conj(v[k]) * _cx(dx, s.qidx[k])) / pn.C[k]
    end
    sh_e, sh_p, sh_r, sh_a, sh_l = (zeros(length(v)) for _ in 1:5)
    # at a reduced node: the rate of each branch's own inductor energy along `dx`, and the load
    rser = zeros(length(s.branches))
    for (ci, c) in enumerate(s.cuts)
        if c.machine == 0
            for (a, j) in enumerate(c.pos)
                di = sum(wb * c.Minv[a, b] * _cx(dx, s.bidx[jb]) for (b, jb) in enumerate(c.pos))
                rser[j] = pn.L[s.branches[j]] * real(conj(f.ib[j]) * di) / wb
            end
        else
            m, q = s.machines[c.machine], f.mc[c.machine]
            r = _winding_currents(m, dx)
            rser[c.pos[1]] = pn.L[s.branches[c.pos[1]]] * (q.iTd * r.iTd + q.iTq * r.iTq) / wb
        end
        iL, diL = _cut_load_rate(s, ci, f, dx)
        k = c.node
        sh_e[k] = c.LL * abs2(iL) / (2 * wb)
        sh_l[k] = c.RL * abs2(iL)
        sh_p[k] = real(conj(f.iload[ci]) * v[k])
        sh_a[k] = abs(sh_p[k])
        sh_r[k] = c.LL * real(conj(iL) * diL) / wb
    end
    inter = sum(sh_p)
    corek = Dict(q.pos => k for (k, q) in enumerate(s.cores))
    st = s.step[]
    for (j, l) in enumerate(s.branches)
        i = f.ib[j]
        fr, to, c = pn.from[l], pn.to[l], s.charge[j] / 2
        if haskey(corek, j)
            # a cored branch (transformer.jl), one component: its stored energy, both leakage resistances, the power
            # in at the HV end less the power out to the EMF; the rate along `dx`, or in a step the discrete-gradient
            # increment (record section 23), whose sum `closure` compares with the energy function
            k = corek[j]
            q, iS = s.cores[k], f.isrc[j]
            xH, xS = _cx(x, s.bidx[j]), _cx(x, q.xs)
            e = core_energy(q, wb * xH, wb * xS, x[s.iclk]) / wb
            pin, pout = real(conj(i) * v[fr] / pn.a[l]), real(conj(iS) * v[to])
            pser = pin - pout
            inter += pser
            rs = stepinc ? st.inc[k] / st.h :
                 real(conj(i) * _cx(dx, s.bidx[j])) - real(conj(iS) * _cx(dx, q.xs)) +
                 core_rotation_rate(i, iS, xH, xS) * dx[s.iclk]
            add!(e + c * (ecap[fr] + ecap[to]), q.R2 * abs2(i) + q.R1 * abs2(iS), pser + c * (pcap[fr] + pcap[to]),
                 rs + c * (rcap[fr] + rcap[to]), abs(pin) + abs(pout) + c * (abs(pcap[fr]) + abs(pcap[to])))
            continue
        end
        pser = real(conj(i) * (v[fr] / pn.a[l] - v[to]))
        inter += pser
        rs = s.bcut[j] == 0 ? real(conj(i) * _cx(dx, s.bidx[j])) : rser[j]
        add!(pn.L[l] * abs2(i) / (2 * wb) + c * (ecap[fr] + ecap[to]), pn.R[l] * abs2(i),
             pser + c * (pcap[fr] + pcap[to]), rs + c * (rcap[fr] + rcap[to]),
             abs(pser) + c * (abs(pcap[fr]) + abs(pcap[to])))
    end
    for (j, k) in enumerate(pn.sh_bus)
        s.sidx[j] > 0 || continue
        i = f.ish[j]
        sh_e[k] += pn.Lsh[j] * abs2(i) / (2 * wb)
        sh_l[k] += s.Rsh[j] * abs2(i)
        sh_p[k] += real(conj(i) * v[k])
        sh_a[k] += abs(real(conj(i) * v[k]))
        sh_r[k] += real(conj(i) * _cx(dx, s.sidx[j]))
        inter += real(conj(i) * v[k])
    end
    for k in vcat(s.dyn, s.alg, s.cut)
        inter += real(conj(v[k]) * f.inj[k])
        (_has_load(pn, s.Cown, k) || s.sl[k] > 0) || continue
        gl = (pn.G[k] + _gstat(s, f, k)) * abs2(v[k])      # (with a composite load's static conductance)
        add!(sh_e[k] + s.Cown[k] * ecap[k], gl + sh_l[k],
             sh_p[k] + s.Cown[k] * pcap[k] + gl, sh_r[k] + s.Cown[k] * rcap[k],
             sh_a[k] + s.Cown[k] * abs(pcap[k]) + gl)
    end
    pf = sum(u.gf[k] * abs2(v[k]) for k in vcat(s.dyn, s.alg))
    if f.z0 !== nothing                               # phase-asymmetric faults: the full [v; v0]' G_f [v; v0]
        for (k, ifk) in f.z0.fdq
            pf += real(conj(v[k]) * ifk) + (haskey(f.z0.f0, k) ? f.z0.v0bus[k] * f.z0.f0[k] : 0.0)
        end
    end
    add!(0.0, pf, pf, 0.0, pf)
    supply = 0.0
    for (j, m) in enumerate(s.machines)
        w, c, o = m.w, f.mc[j], m.offset
        i2 = c.id^2 + c.iq^2
        pstat = real(conj(v[m.node] / m.a) * c.inet)
        inter += pstat
        tem = _torque(m, x, c)
        emag = _magnetic_energy(m, x, c)
        rmag = c.id * dx[o] + c.iq * dx[o+1] + c.ifd * dx[o+2] + c.i1d * dx[o+3] + c.i1q * dx[o+4] +
               (w.nq == 2 ? c.i2q * dx[o+5] : 0.0)
        etr = m.x_ext * i2 / (2 * wb)
        # rate of the merged transformer's energy: x_ext/omega_b * (id did/dt + iq diq/dt),
        # along `dx` for the rate and along the vector field for the power it passes on
        function rtr_of(d)
            rr = _winding_currents(m, d)
            return m.x_ext * (c.id * rr.id + c.iq * rr.iq) / wb
        end
        rtr, rtrf = rtr_of(dx), rtr_of(dxf)
        if m.cut > 0
            # the magnetic energy of all windings holds the branch's and the load's inductors
            q = s.cuts[m.cut]
            jT = q.pos[1]
            rmag += c.iTd * dx[m.tidx] + c.iTq * dx[m.tidx+1]
            emag -= pn.L[s.branches[jT]] * (c.iTd^2 + c.iTq^2) / (2 * wb) + sh_e[q.node]
            rmag -= rser[jT] + sh_r[q.node]
        end
        ltr = m.r_ext * i2
        pout = pstat - (rtrf + ltr)            # what the transformer passes to the windings
        lown = (w.Ra - m.r_ext) * i2 + w.Rfd * c.ifd^2 + w.R1d * c.i1d^2 + w.R1q * c.i1q^2 +
               w.R2q * c.i2q^2
        pfield = u.vf[j] * c.ifd
        pshaft = _shaft_torque(u, j, c.omega) * c.omega
        supply += pfield + pshaft
        if m.trf != ""
            # the transformer passes what it receives less its own storage rate and loss
            add!(etr, ltr, rtrf + ltr, rtr, abs(pstat) + abs(pout))
        end
        # windings: stator power after the transformer, field power, less the air-gap power
        add!(emag - etr, lown, pout + pfield - tem * c.omega, rmag - rtr,
             abs(pout) + abs(pfield) + abs(tem * c.omega))
        add!(x[_ip(m)]^2 / (4 * w.H), 0.0, pshaft + tem * c.omega, c.omega * dx[_ip(m)],
             abs(pshaft) + abs(tem * c.omega))
    end
    # composite loads' motors (motor.jl): windings and shaft; the load torque's work T_L omega is the motor's output
    # (booked with its losses: it leaves the electrical system)
    for (j, mm) in enumerate(s.motors)
        q, o, c = f.mo[j], mm.offset, mm.c
        q.on || (add!(0.0, 0.0, 0.0, 0.0, 0.0); continue)   # tripped: the motor has left the system
        sg = q.sg                                      # (the remaining units: resistances / sigma, torque x sigma)
        pin = real(conj(q.is) * v[mm.node])
        inter += pin
        TL = sg * motor_load_torque(c, q.omega)
        lw = (c.Rs * abs2(q.is) + c.R1 * abs2(q.i1) + c.R2 * abs2(q.i2)) / sg
        rt = real(conj(q.is) * _cx(dx, o)) + real(conj(q.i1) * _cx(dx, o + 2)) + real(conj(q.i2) * _cx(dx, o + 4)) +
             q.omega * dx[o+6]
        add!(sg * motor_energy(c, q.ls, q.l1, q.l2, q.p / sg, wb), lw + TL * q.omega, pin, rt,
             abs(pin) + lw + abs(TL * q.omega))
    end
    # the zero axis (upgrade 4): one component: its capacitors, the lines' zero branches, the transformer zero shunts;
    # losses R0 i0^2; its port: the power it receives from the faults' zero terminals (-v0 i_f0) and from the zero-mode
    # travelling-wave lines (-v0 (G0 v0 - J0), a boundary like the positive mode's)
    if s.z0 !== nothing
        z, zf = s.z0, f.z0
        e0 = _zero_energy(s, x)
        l0 = sum(b.R * i^2 for (b, i) in zip(z.br, zf.i0br); init = 0.0) + sum(h.R * i^2 for (h, i) in zip(z.sh, zf.i0sh); init = 0.0)
        pz = -sum((zf.v0bus[k] * i for (k, i) in zf.f0); init = 0.0)
        ptw0 = 0.0
        for q in eachindex(z.nodes)
            z.G0[q] == 0 && continue
            ptw0 += zf.v0[q] * (z.G0[q] * zf.v0[q] - z.twJ0[q])
        end
        r0 = sum(zf.v0[q] * dx[z.qidx0[q]] for q in z.dyn0; init = 0.0) +
             sum(i * dx[idx] for (i, idx) in zip(zf.i0br, z.bidx0); init = 0.0) +
             sum(i * dx[idx] for (i, idx) in zip(zf.i0sh, z.sidx0); init = 0.0)
        add!(e0, l0, pz - ptw0, r0, abs(pz) + abs(ptw0) + l0)
        inter += ptw0; supply -= ptw0                 # (the zero-mode lines' boundary, as the positive mode's)
    end
    # travelling-wave lines: the power a bus gives its lines, Re(conj(v) (twG v - twJ)), leaves the lumped system through
    # a boundary (the lines' own account is tline.jl's: storage, resistor losses, interpolation dissipation)
    for k in eachindex(s.twG)
        s.twG[k] == 0 && continue
        pl = real(conj(v[k]) * (s.twG[k] * v[k] - s.twJ[k]))
        inter += pl
        supply -= pl
    end
    # a converter's EMF is a boundary supply: the power its source row gives its branch
    for cv in s.conv
        pr = real(conj(v[cv.row]) * f.inj[cv.row])
        inter += pr
        supply -= pr
    end
    c_[] == nc || error("phs_ledger: filled $(c_[]) of $nc components")
    return (; names = s.cnames, kind = s.ckind, energy, loss, port, rate, tport, supply,
            interconnection = inter, H = sum(energy), dH = sum(rate), losses = sum(loss))
end

"""
    phs_network_mask(s, L) -> BitVector

The components of the ledger `L` (`phs_ledger` of `s`, a `PHSystem` or `PHControlled`) on the network side of the
machines' ports, where a machine's power is `-v_node conj(i_net) / a` at its terminal bus row (the high-voltage bus
after the series merge): lines, network transformers, loads and the fault entry. A machine's merged step-up
transformer is not among them: it lies inside the port, between that bus and the windings (record 30). So at any
state the machines' port active powers sum to the masked components' `rate + loss`. Throws if a merged transformer
is not exactly one `:transformer` entry of `L`, or with converters or saturating cores (not covered).
"""
function phs_network_mask(s::PHSystem, L)
    (isempty(s.conv) && isempty(s.cores)) ||
        throw(ArgumentError("phs_network_mask: converters and transformer cores are not covered"))
    # (added at the realtime merge: the motors, the zero axis and travelling-wave lines came later and are not covered)
    (isempty(s.motors) && s.z0 === nothing && isempty(s.tw)) ||
        throw(ArgumentError("phs_network_mask: composite-load motors, the zero axis and travelling-wave lines are not covered"))
    merged = [m.trf for m in s.machines if m.trf != ""]
    allunique(merged) || throw(ArgumentError("phs_network_mask: a merged transformer assigned twice"))
    for t in merged
        k = findall(==(t), L.names)
        (length(k) == 1 && L.kind[k[1]] === :transformer) ||
            throw(ArgumentError("phs_network_mask: merged transformer $t is not one :transformer entry of the ledger"))
    end
    mset = Set(merged)
    return BitVector([k in (:line, :transformer, :load, :fault) && !(nm in mset) for (nm, k) in zip(L.names, L.kind)])
end

# the generator-bus voltage of every merged machine, from flows `f` whose reduced-node voltages
# are those along `dx` (the state `x` is needed for a converter with a cored transformer)
function _terminal_voltages(s::PHSystem, f, dx, x = nothing)
    wb = s.omega_b
    out = Dict{Int,ComplexF64}()
    for (j, m) in enumerate(s.machines)
        m.trf == "" && continue
        c = f.mc[j]
        r = _winding_currents(m, dx)
        # d/dt of i_net = -j e^{j delta} i_dq (stationary frame: e^{j (delta + theta_f)}, rotating at delta' + omega_b;
        # and no frame term j x_ext i)
        if s.stationary
            dinet = -im * c.rot * (complex(r.id, r.iq) + im * (dx[_idelta(m)] + wb) * complex(c.id, c.iq))
            out[m.bus] = _node_voltage(s, m.node, f, dx) / m.a - m.r_ext * c.inet - m.x_ext * dinet / wb
            continue
        end
        dinet = -im * c.rot * (complex(r.id, r.iq) + im * dx[_idelta(m)] * complex(c.id, c.iq))
        out[m.bus] = _node_voltage(s, m.node, f, dx) / m.a - complex(m.r_ext, m.x_ext) * c.inet -
                     m.x_ext * dinet / wb
    end
    for cv in s.conv                          # the converter buses removed by the merge
        cv.merged && (out[cv.bus] = cv.core == 0 ? _conv_terminal(s, cv, f, dx) : _conv_terminal(s, cv, x, f, dx))
    end
    return out
end

"""
    phs_terminal_voltage(s, x, u) -> Dict(bus id => complex voltage)

The voltage of every generator bus removed by the series merge, recovered from the transformer
it was merged with: `v_g = v/a - (r_ext + j x_ext) i - (x_ext/omega_b) di/dt`, with `di/dt` from
the vector field. This is the eliminated variable of the unreduced model.
"""
phs_terminal_voltage(s::PHSystem, x::AbstractVector{Float64}, u::PHInput) =
    _terminal_voltages(s, phs_flows(s, x, u), phs_rhs(s, x, u), x)

"""
    phs_frame(s, frame; theta0 = 0.0) -> PHSystem

The same model with its network states in `frame`: `:synchronous` (rotating at omega_b; today's model) or
`:stationary` (`docs/realtime/docs/STEP0_CONTRACTS.md` A3-A4): every frame term `-j omega_b x` of a network-frame
state is removed, and each machine's terminal rotation becomes e^{j (delta + theta_f)} with
theta_f = omega_b t + theta0. Rotor-frame states (machine windings, a reduced node's machine winding) and the angles
delta keep their meaning. Converters and saturating cores are not available in the stationary frame yet (their
controllers and the cores' frame clock need their own derivation): the model is refused. Initial states:
`phs_rotate_network`.
"""
function phs_frame(s::PHSystem, frame::Symbol; theta0::Real = 0.0)
    frame in (:synchronous, :stationary) || throw(ArgumentError("phs_frame: frame = $frame"))
    st = frame === :stationary
    if st
        isempty(s.conv) || throw(ArgumentError("phs_frame: the stationary frame has no converters yet " *
                                               "($(join([c.name for c in s.conv], ", ")))"))
        isempty(s.cores) || throw(ArgumentError("phs_frame: the stationary frame has no saturating cores yet"))
    end
    return PHSystem((getfield(s, f) for f in fieldnames(PHSystem)[1:end-2])..., st, st ? Float64(theta0) : 0.0)
end

# the network-frame complex pairs of the state (node charges, kept branch fluxes, a reduced node's branch fluxes
# without a machine, shunt fluxes): the states the frame rotates; rotor-frame pairs are not among them
function _network_pairs(s::PHSystem)
    pairs = Int[]
    for k in s.dyn
        push!(pairs, s.qidx[k])
    end
    for (j, _) in enumerate(s.branches)
        (s.bcut[j] == 0 || s.cuts[s.bcut[j]].machine == 0) && push!(pairs, s.bidx[j])
    end
    for j in eachindex(s.net.sh_bus)
        s.sidx[j] > 0 && push!(pairs, s.sidx[j])
    end
    for mo in s.motors                                 # composite loads' motors: all three flux pairs are in the network frame
        push!(pairs, mo.offset, mo.offset + 2, mo.offset + 4)
    end
    return sort!(pairs)
end

"""
    phs_rotate_network(s, x, theta) -> state

`x` with every network-frame pair multiplied by e^{j theta} (rotor-frame states and angles unchanged): the state of
the stationary-frame model at time t from the synchronous one, with theta = omega_b t + theta0; `-theta` goes back.
Only the first `s.n` entries (the physical model) are touched; controller states follow unchanged.
"""
function phs_rotate_network(s::PHSystem, x::AbstractVector, theta::Real)
    y = copy(x)
    c, sn = cos(theta), sin(theta)
    for i in _network_pairs(s)
        y[i], y[i+1] = c * x[i] - sn * x[i+1], sn * x[i] + c * x[i+1]
    end
    return y
end

# the frame phasor of the input for an evaluation at time t: e^{j (omega_b t + theta0)} in the stationary frame;
# nothing (no change, no operation) in the synchronous frame
@inline _set_frame!(u::PHInput, s::PHSystem, t::Real) =
    (s.stationary && (u.cf[1] = cis(s.omega_b * t + s.theta0));
     s.asym.theta[] = s.omega_b * t + s.asym.theta_pw;     # the dq0 transform's angle (upgrade 4), synchronous frame
     u)

"""
    phs_cut_voltage(s, x, u) -> Dict(bus id => complex voltage)

The voltage of every reduced inductor-only node, recovered from its load's own equation,
`v = (L_L/omega_b) di_L/dt + (R_L + j L_L) i_L`, with `di_L/dt` from the vector field. This is
the eliminated multiplier of the unreduced model.
"""
function phs_cut_voltage(s::PHSystem, x::AbstractVector{Float64}, u::PHInput)
    f, dx = phs_flows(s, x, u), phs_rhs(s, x, u)
    return Dict(s.net.net.bus_ids[c.node] => _cut_voltage(s, ci, f, dx) for (ci, c) in enumerate(s.cuts))
end

"""
    phs_phases(z, theta) -> (a, b, c)

The three phase values of the network-frame phasor `z` at the frame angle
`theta = omega_b t + theta_0`: `Re(e^{j theta} z)` and the same a third of a turn behind and
ahead (the inverse dq0 transformation of the specification, section 2, in per unit of the peak
phase value). The balanced model has no zero-axis component, so nothing is added.
"""
phs_phases(z, theta) = (real(cis(theta) * z), real(cis(theta - 2pi / 3) * z), real(cis(theta + 2pi / 3) * z))

"""
    phs_equilibrium(eq::EquilibriumResult; loads = :parallel, circuits = nothing,
                    const_power = false, stray_capacitance = 0.0) -> (s, x, u, report)

The model of `eq`'s case and its balanced equilibrium. Start: the bus voltages of the RMS
equilibrium, with the loads as constant impedances at those voltages, so that the network's
quasi-static limit is the case's admittance matrix; branch and shunt currents from them; each
machine's angle, fluxes and field voltage by the steady-state construction of `emdq_pilot`.
Then, with every rotor angle, speed (1) and field voltage held, the remaining equations are
linear in the remaining states and are solved exactly, and each shaft torque is set to balance
its air-gap torque (with `const_power`, the turbine power, the same number at speed 1). The
keywords are those of `ph_system` and `PHInput`. `report` holds the full residual of the vector field, the largest change of
a bus voltage from the RMS equilibrium (reduced nodes included), and the recovered
generator-bus voltages.

With `cores` (`ph_system`) each cored branch starts from the linear T-circuit at the operating point (the core's first
segment), the frame clock at 0 and held; the clock's row (its rate is `omega_b`) is not part of the residual. If a core's
phase fluxes there exceed the curve's first knot the operating point is not a fixed point of the model (its steady state
is periodic: record section 21, T1 row 7); the rows are then solved at the clock's initial value only (an instantaneous
rest, not a periodic operating solution: T3 2d) and `report.core_flux` says how far the fluxes are.
"""
function phs_equilibrium(eq::EquilibriumResult; loads::Symbol = :parallel, circuits = nothing,
                         const_power::Bool = false, stray_capacitance::Real = 0.0, cores = nothing,
                         lines::Symbol = :lumped, nsec::Int = 4, composite = (;), dq0 = (;))
    sys = eq.sys
    nb = nbus(sys.net)
    V = [complex(eq.V[2i-1], eq.V[2i]) for i in 1:nb]
    s = ph_system(sys; v0 = abs.(V), loads, circuits, stray_capacitance, cores, lines, nsec, composite..., dq0...)
    pn = s.net
    wb = s.omega_b
    # each converter's EMF on its source row: the RMS kernel's output at its equilibrium state
    # (u_out = E there: the virtual impedance is zero below the current limit)
    Ec = ComplexF64[]
    for cv in s.conv
        p = cv.comp.p
        kc = findfirst(c -> c === cv.comp, sys.comps)
        o = sys.offsets[kc]
        if cv.comp isa GFM_VSM_PHTRUE
            (p.r_vi == 0 && p.x_vi == 0 && p.K_field == 0 && p.n_vi <= 1) ||
                throw(ArgumentError("phs_equilibrium: converter $(cv.name): only r_vi = x_vi = 0, K_field = 0, n_vi = 1"))
            push!(Ec, eq.x[o+3] * cis(eq.x[o]))
        elseif cv.comp isa GFL_PHTRUE                # the static generator's voltage source behind its reactance (record
            i = sys.net.index[cv.bus]                # section 38): E0 = V + j Zseries I, I the unit's injected current
            push!(Ec, V[i] + im * p.Zseries * complex(eq.x[o], eq.x[o+1]))
        else                                       # a droop or VOC converter: its output from the Norton injection
            (p.r_vi == 0 && p.x_vi == 0) ||
                throw(ArgumentError("phs_equilibrium: converter $(cv.name): only r_vi = x_vi = 0"))
            Id, Iq = injection(cv.comp, eq.x[o:o+nstates(cv.comp)-1], nothing)
            push!(Ec, im * p.Zseries * complex(Id, Iq))
        end
    end
    V = vcat(V, Ec)
    # the current every bus needs from its machines at these voltages: Kirchhoff's current law
    # over all elements of the case (the merged transformers included), at nominal frequency
    Iext = complex.(pn.G, pn.C) .* V
    ibr(l) = (V[pn.from[l]] / pn.a[l] - V[pn.to[l]]) / complex(pn.R[l], pn.L[l])
    # travelling-wave lines: the same bus voltages (the load flow's), each line's own steady state (constant phasors) at
    # them; their history currents twJ = twG v - i_line held at those values (the new equilibrium: the machines supply
    # what the travelling-wave network needs at these voltages, so their outputs differ slightly from the load flow's)
    fill!(s.twJ, 0)
    for t in s.tw
        L = TLine(t.r, t.x, t.b; omega_b = wb, h = 1e-9, nsec = s.nsec)   # (the steady state does not depend on h)
        i1, i2 = tline_steady(L, V[t.from], V[t.to])
        Iext[t.from] += i1; Iext[t.to] += i2
        s.twJ[t.from] += t.G * V[t.from] - i1; s.twJ[t.to] += t.G * V[t.to] - i2
    end
    for l in eachindex(pn.R)
        any(t -> t.name == pn.branch[l] && t.from == pn.from[l] && t.to == pn.to[l], s.tw) && continue
        i = ibr(l)
        Iext[pn.from[l]] += i / conj(pn.a[l])
        Iext[pn.to[l]] -= i
    end
    for (j, k) in enumerate(pn.sh_bus)
        Iext[k] += V[k] / complex(s.Rsh[j], pn.Lsh[j])
    end
    # composite loads: each static part at rest (z = |V| - Vini) and each motor at the slip that draws its P_m at the
    # load flow's |V| (the steady state is linear in V at a fixed slip)
    for st in s.statics
        k = st.node
        Iext[k] += static_conductance(st.law, abs(V[k]) - st.law.Vini) * V[k]
    end
    msteady = [motor_steady(mm.c, V[mm.node], mm.slip0; omega_b = wb) for mm in s.motors]
    for (mm, q) in zip(s.motors, msteady)
        Iext[mm.node] += q.is
    end
    x = zeros(s.n)
    setc!(i, c) = (x[i] = real(c); x[i+1] = imag(c))
    for k in s.dyn
        setc!(s.qidx[k], pn.C[k] * V[k] / wb)
    end
    for (j, l) in enumerate(s.branches)
        s.bcut[j] == 0 && setc!(s.bidx[j], pn.L[l] * ibr(l) / wb)
    end
    for c in s.cuts                                  # Phi = M i / omega_b
        c.machine == 0 || continue
        ic = [ibr(s.branches[j]) for j in c.pos]
        for (a, j) in enumerate(c.pos)
            setc!(s.bidx[j], sum(c.M[a, b] * ic[b] for b in eachindex(ic)) / wb)
        end
    end
    for (j, k) in enumerate(pn.sh_bus)
        s.sidx[j] > 0 && setc!(s.sidx[j], pn.Lsh[j] * (V[k] / complex(s.Rsh[j], pn.Lsh[j])) / wb)
    end
    for q in s.cores                                 # the linear T-circuit at nominal frequency
        l = s.branches[q.pos]
        v2, E = V[pn.from[l]] / pn.a[l], V[pn.to[l]]
        Z2, Z1, Zm = complex(q.R2, q.L2), complex(q.R1, q.L1), im * q.curve.Lm
        vm = (v2 / Z2 + E / Z1) / (1 / Z2 + 1 / Z1 + 1 / Zm)
        iH, iS, lam = (v2 - vm) / Z2, (vm - E) / Z1, -im * vm
        setc!(s.bidx[q.pos], (q.L2 * iH + lam) / wb)
        setc!(q.xs, (lam - q.L1 * iS) / wb)
    end
    for (mm, q) in zip(s.motors, msteady)
        o = mm.offset
        setc!(o, q.ls); setc!(o + 2, q.l1); setc!(o + 4, q.l2)
        x[o+6] = q.p
    end
    for st in s.statics
        x[st.zidx] = abs(V[st.node]) - st.law.Vini
    end
    for mm in s.motors                               # relays (review 76): every unit in, counts and levels at 0
        mm.ridx > 0 && (x[mm.ridx] = 1.0)
    end
    nm = length(s.machines)
    Tm, vf = zeros(nm), zeros(nm)
    for (j, m) in enumerate(s.machines)
        w, o = m.w, m.offset
        g = bus_index(pn.net, m.bus)
        vt = V[m.node] / m.a
        inet = m.trf == "" ? -Iext[g] : (vt - V[g]) / complex(m.r_ext, m.x_ext)
        E = vt - complex(w.Ra, w.Ll + w.Laq) * inet
        delta = angle(E)
        rot = im * complex(cos(delta), -sin(delta))
        vc, ic = rot * vt, rot * inet
        psic = -im * (vc - w.Ra * ic)
        psid, psiq, id, iq = real(psic), imag(psic), real(ic), imag(ic)
        ifd = (psid - (w.Ll + w.Lad) * id) / w.Lad
        ifd > 0 || error("phs_equilibrium: field current $ifd of $(m.name) is not positive")
        x[o], x[o+1] = psid / wb, psiq / wb
        x[o+2] = (w.Lad * id + (w.Lfd + w.Lad) * ifd) / wb
        x[o+3] = w.Lad * (id + ifd) / wb
        x[o+4] = w.Laq * iq / wb
        w.nq == 2 && (x[o+5] = w.Laq * iq / wb)
        if m.cut > 0                                 # the pairs of a reduced node
            q = s.cuts[m.cut]
            iT = rot * ibr(s.branches[q.pos[1]])
            x[o] += real(q.M[1, 1] * ic + q.M[1, 2] * iT) / wb
            x[o+1] += imag(q.M[1, 1] * ic + q.M[1, 2] * iT) / wb
            setc!(m.tidx, (q.M[2, 1] * ic + q.M[2, 2] * iT) / wb)
        end
        x[_ip(m)] = 2 * w.H
        x[_idelta(m)] = delta
        vf[j] = w.Rfd * ifd
        Tm[j] = -(psid * iq - psiq * id)
    end
    u = PHInput{Float64}(Tm, vf, zeros(nb), const_power, Ec)   # at speed 1 the torque and the power are equal
    # hold angles and momenta; the other rows are linear in the other states
    held = reduce(vcat, [[_ip(m), _idelta(m)] for m in s.machines]; init = Int[])
    s.iclk > 0 && push!(held, s.iclk)
    for mm in s.motors                               # the relays' discrete states
        mm.ridx > 0 && append!(held, mm.ridx:mm.ridx+4)
    end
    free = setdiff(1:s.n, held)
    sub(y) = (xx = Vector{eltype(y)}(x); xx[free] .= y; phs_rhs(s, xx, u)[free])
    residual0 = maximum(abs, sub(x[free]); init = 0.0)   # before the refinement (the built state's rest residual)
    get(ENV, "PHS_EQ_DEBUG", "") == "1" && (r0 = sub(x[free]); nm0 = phs_state_names(s)[free];
        for i in sortperm(abs.(r0); rev = true)[1:8]; println("  rest residual ", nm0[i], " ", r0[i]); end)
    # travelling-wave lines: the load flow's voltages do not satisfy KCL with the lines' own steady state, so the operating
    # point is the model's own: the rest state with each line's history at its steady state of the current bus voltages,
    # by alternating the rest solve (history held) and the history update until both agree (reviews 71, item 1)
    twl = [TLine(t.r, t.x, t.b; omega_b = wb, h = 1e-9, nsec = s.nsec) for t in s.tw]
    twdiff = 0.0
    if isempty(s.tw)
        for _ in 1:(isempty(s.cores) ? 3 : 30)
            r = sub(x[free])
            maximum(abs, r) < 1e-14 && break
            x[free] .-= ForwardDiff.jacobian(sub, x[free]) \ r
        end
    else
        # one Newton solve over the free states and the history currents together (alternating the two diverges: the
        # history fed back through the algebraic buses has a gain near 1); the residual is the rest field and
        # J - J_ss(v), J_ss the lines' steady state at the bus voltages v; Jacobian by central differences
        jr = findall(!iszero, s.twG); nf = length(free); nj = length(jr)
        function Faug(y)
            xx = copy(x); xx[free] .= y[1:nf]
            s.twJ[jr] .= complex.(y[nf+1:nf+nj], y[nf+nj+1:end])
            r1 = phs_rhs(s, xx, u)[free]
            vv = phs_flows(s, xx, u).v
            Jss = zeros(ComplexF64, length(s.twJ))
            for (t, L) in zip(s.tw, twl)
                i1, i2 = tline_steady(L, vv[t.from], vv[t.to])
                Jss[t.from] += t.G * vv[t.from] - i1; Jss[t.to] += t.G * vv[t.to] - i2
            end
            d = s.twJ[jr] .- Jss[jr]
            return vcat(r1, real.(d), imag.(d))
        end
        y = vcat(x[free], real.(s.twJ[jr]), imag.(s.twJ[jr]))
        for _ in 1:25
            r = Faug(y)
            maximum(abs, r) < 1e-13 && break
            Jm = zeros(length(r), length(y))
            for c in eachindex(y)
                hc = 1e-7 * max(1.0, abs(y[c])); yp = copy(y); ym = copy(y); yp[c] += hc; ym[c] -= hc
                Jm[:, c] = (Faug(yp) .- Faug(ym)) ./ (2hc)
            end
            y .-= Jm \ r
        end
        r = Faug(y); twdiff = maximum(abs, r[nf+1:end])
        x[free] .= y[1:nf]
    end
    f = phs_flows(s, x, u)
    for (j, m) in enumerate(s.machines)
        Tm[j] = -_torque(m, x, f.mc[j])
    end
    dx = phs_rhs(s, x, u)
    vt = _terminal_voltages(s, f, dx, x)
    dV = maximum(abs(_node_voltage(s, k, f, dx) - V[k]) for k in vcat(s.dyn, s.alg, s.cut))
    dVg = isempty(vt) ? 0.0 : maximum(abs(vt[b] - V[bus_index(pn.net, b)]) for b in keys(vt))
    core_flux = [(; bus = q.bus, lam = abs(core_solve(q, wb * _cx(x, s.bidx[q.pos]), wb * _cx(x, q.xs), x[s.iclk]).lam),
                  knot = q.curve.lam[2]) for q in s.cores]
    report = (; residual = phs_rest_residual(s, dx), residual0, twdiff, dV, dVgen = dVg, V_rms = V,
              Tm = copy(Tm), vf = copy(vf), core_flux)
    return s, x, u, report
end

"""
    tw_steady_history!(s, x, u) -> the largest change of twJ in the last iteration

Travelling-wave lines: set `s.twJ` to the lines' steady-state history at the bus voltages of the state `x` (constant
phasors), solving J = J_ss(v(x, J)) (the algebraic buses' voltages depend on J); linear, Newton with a difference
Jacobian. Used at the start of a run so it does not depend on how the model was built.
"""
function tw_steady_history!(s::PHSystem, x::AbstractVector{Float64}, u::PHInput)
    isempty(s.tw) && return 0.0
    twl = [TLine(t.r, t.x, t.b; omega_b = s.omega_b, h = 1e-9, nsec = s.nsec) for t in s.tw]
    jr = findall(!iszero, s.twG); nj = length(jr)
    function F(y)
        s.twJ[jr] .= complex.(y[1:nj], y[nj+1:end])
        vv = phs_flows(s, x, u).v
        Jss = zeros(ComplexF64, length(s.twJ))
        for (t, L) in zip(s.tw, twl)
            i1, i2 = tline_steady(L, vv[t.from], vv[t.to])
            Jss[t.from] += t.G * vv[t.from] - i1; Jss[t.to] += t.G * vv[t.to] - i2
        end
        d = s.twJ[jr] .- Jss[jr]
        return vcat(real.(d), imag.(d))
    end
    y = vcat(real.(s.twJ[jr]), imag.(s.twJ[jr])); ch = Inf
    for _ in 1:6
        r = F(y); maximum(abs, r) < 1e-14 && break
        Jm = zeros(2nj, 2nj)
        for c in 1:2nj
            hc = 1e-6; yp = copy(y); ym = copy(y); yp[c] += hc; ym[c] -= hc
            Jm[:, c] = (F(yp) .- F(ym)) ./ (2hc)
        end
        dy = Jm \ r; y .-= dy; ch = maximum(abs, dy)
    end
    F(y)
    return ch
end

"""    phs_rest_residual(s, dx) -> the largest entry of the vector field `dx` at rest, the frame clock's row left out."""
function phs_rest_residual(s, dx)
    ps = _phys(s)
    ps.iclk == 0 && return maximum(abs, dx)
    return maximum(abs(dx[i]) for i in eachindex(dx) if i != ps.iclk)
end

"""
    phs_core_signals(s, x) -> Vector of (; bus, lam, phase, im, iH, iS, segment, E_Lc, E_leak, E_core)

Per saturating core at the state `x`: the core flux (complex, network frame, flux units), the three phase fluxes, the
magnetising current (`iH - iS`), the HV and source-side branch currents, the highest segment of the curve a phase is
on, and the stored energy split by owner (pu s): the converter's `Lc`, the two leakage halves and the core. The pieces
are energies at the state only; their step increments have no balance of their own (record section 23).
"""
function phs_core_signals(s::PHSystem, x::AbstractVector)
    wb = s.omega_b
    return map(s.cores) do q
        th = x[s.iclk]
        c = core_solve(q, wb * _cx(x, s.bidx[q.pos]), wb * _cx(x, q.xs), th)
        ph = Tuple(real(c.lam * cis(th + p)) for p in _CORE_PHASES)
        (; bus = q.bus, lam = c.lam, phase = ph, im = c.iH - c.iS, iH = c.iH, iS = c.iS,
         segment = maximum(_seg(q.curve, l) for l in ph),
         E_Lc = q.Lc * abs2(c.iS) / (2 * wb), E_leak = ((q.L1 - q.Lc) * abs2(c.iS) + q.L2 * abs2(c.iH)) / (2 * wb),
         E_core = _core_W(q.curve, c.lam, th) / wb)
    end
end
phs_core_signals(cs, x::AbstractVector) = phs_core_signals(_phys(cs), view(x, 1:_phys(cs).n))
