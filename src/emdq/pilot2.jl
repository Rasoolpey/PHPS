# Two-machine electromagnetic dq pilot for the PCC view (docs/active/PCC_VIEW_N2_FORMULATION.md,
# step N2 of ROA_ROUTES_TODO.md). The one-machine pilot (`pilot.jl`) with a second branch from
# its PCC (bus 2): a line to a load bus (bus 25) and a second full-winding machine behind its
# step-up transformer (merged into its stator, as for machine 1). The fixed-voltage bus at the
# far end of the first chain is kept (`fixed_bus = true`) or removed, which closes the system.
#
# Same elements, units and conventions as `pilot.jl` (motor convention, network frame at
# nominal speed, energies in pu*s with 1/omega_b on the magnetic and network energies), with
# fixed modulation on both converter excitations. Written in real arithmetic, so that the same
# code evaluates on Float64, BigFloat, ForwardDiff duals and the truncated Taylor series of
# `pccview.jl`.
#
# State order: machine 1, then the grid side x_g = [network; machine 2]. A machine block is
# [psi_D, psi_Q, psi_fd, psi_1d, psi_1q, (psi_2q for nq = 2), delta, p, q_dc]; the network is
# [v2, i23, v3, (i34 with the fixed bus), iL3, i225, v25, iL25], each a (Re, Im) pair.

"""
    EMDQSite

A machine site of the two-machine pilot: winding data `mach` (transformer merged), the ideal
ratio `tap` at the network side, the fixed shaft torque `Tm`, the DC source `vs` behind `Rs`
feeding the link `Cdc`, the fixed modulation `mstar` and the diagnostic shaft damping `Dm`.
"""
struct EMDQSite
    mach::WindingMachine
    tap::Float64
    Tm::Float64
    vs::Float64
    Rs::Float64
    Cdc::Float64
    mstar::Float64
    Dm::Float64
end

"""
    EMDQPilot2

Two-machine pilot: machine 1 (`m1`) at bus 2 (the PCC); line 2-3 (`R23`, `L23`), bus 3 (`C3`,
load `G3`, `L3`), and, if `fixed_bus`, line 3-4 (`R34`, `L34`) to the fixed voltage `V4`; line
2-25 (`R225`, `L225`), bus 25 (`C25`, load `G25`, `L25`) and machine 2 (`m2`). `C2` is the
whole bus-2 capacitance.
"""
struct EMDQPilot2
    m1::EMDQSite
    m2::EMDQSite
    C2::Float64
    R23::Float64
    L23::Float64
    C3::Float64
    G3::Float64
    L3::Float64
    R34::Float64
    L34::Float64
    V4::ComplexF64
    R225::Float64
    L225::Float64
    C25::Float64
    G25::Float64
    L25::Float64
    fixed_bus::Bool
end

_site_nstates(M::WindingMachine) = 7 + M.nq

"""
    emdq2_grid_layout(s) -> NamedTuple

First (Re) index of each network pair in the grid-side vector `x_g`, `i34 = 0` without the
fixed bus, the offset `m2` of machine 2's block, and the grid-side size `n`.
"""
function emdq2_grid_layout(s::EMDQPilot2)
    names = s.fixed_bus ? (:v2, :i23, :v3, :i34, :iL3, :i225, :v25, :iL25) :
                          (:v2, :i23, :v3, :iL3, :i225, :v25, :iL25)
    idx = Dict(n => 2k - 1 for (k, n) in enumerate(names))
    m2 = 2 * length(names)
    return (; v2 = idx[:v2], i23 = idx[:i23], v3 = idx[:v3], i34 = get(idx, :i34, 0),
            iL3 = idx[:iL3], i225 = idx[:i225], v25 = idx[:v25], iL25 = idx[:iL25], m2,
            n = m2 + _site_nstates(s.m2.mach))
end

"""    emdq2_nc(s) -> the number of component (machine 1) states; `x = [x_c; x_g]`."""
emdq2_nc(s::EMDQPilot2) = _site_nstates(s.m1.mach)

"""    emdq2_nstates(s) -> the total number of states."""
emdq2_nstates(s::EMDQPilot2) = emdq2_nc(s) + emdq2_grid_layout(s).n

"""
    emdq2_grid_state_names(s) -> Vector{String}

Names of the grid-side states, in order.
"""
function emdq2_grid_state_names(s::EMDQPilot2)
    net = s.fixed_bus ? ["v2", "i23", "v3", "i34", "iL3", "i225", "v25", "iL25"] :
                        ["v2", "i23", "v3", "iL3", "i225", "v25", "iL25"]
    out = String[]
    for n in net
        push!(out, n * "_Re", n * "_Im")
    end
    m = ["psi_D2", "psi_Q2", "psi_fd2", "psi_1d2", "psi_1q2"]
    s.m2.mach.nq == 2 && push!(m, "psi_2q2")
    return vcat(out, m, ["delta2", "p2", "q_dc2"])
end

# rotor-frame fluxes and currents of the machine block at offset `o` (real form of
# `emdq_machine`): psi_d + j psi_q = j e^{-j delta} psi_n, i_n = -j e^{j delta} (i_d + j i_q)
@inline function _site_currents(M::WindingMachine, x, o)
    nq = M.nq
    psiD, psiQ, psifd, psi1d, psi1q = x[o+1], x[o+2], x[o+3], x[o+4], x[o+5]
    delta = x[o+5+nq]
    c, sn = cos(delta), sin(delta)
    psid = sn * psiD - c * psiQ
    psiq = c * psiD + sn * psiQ
    K, Q = M.Kd, M.Kq
    id = K[1, 1] * psid + K[1, 2] * psifd + K[1, 3] * psi1d
    ifd = K[2, 1] * psid + K[2, 2] * psifd + K[2, 3] * psi1d
    i1d = K[3, 1] * psid + K[3, 2] * psifd + K[3, 3] * psi1d
    if nq == 2
        psi2q = x[o+6]
        iq = Q[1, 1] * psiq + Q[1, 2] * psi1q + Q[1, 3] * psi2q
        i1q = Q[2, 1] * psiq + Q[2, 2] * psi1q + Q[2, 3] * psi2q
        i2q = Q[3, 1] * psiq + Q[3, 2] * psi1q + Q[3, 3] * psi2q
    else
        psi2q = 0 * psi1q
        iq = Q[1, 1] * psiq + Q[1, 2] * psi1q
        i1q = Q[2, 1] * psiq + Q[2, 2] * psi1q
        i2q = 0 * iq
    end
    inD = sn * id + c * iq
    inQ = sn * iq - c * id
    return (; psid, psiq, psifd, psi1d, psi1q, psi2q, id, iq, ifd, i1d, i1q, i2q, inD, inQ,
            Tem = psid * iq - psiq * id)
end

# the machine block's equations (`emdq_rhs!` lines 224-231 and 242) at terminal voltage
# (vD, vQ) on the network side of the ratio
@inline function _site_rhs!(dx, st::EMDQSite, x, o, vD, vQ, e)
    M = st.mach
    wb = M.omega_b
    od = o + 5 + M.nq
    dx[o+1] = wb * (vD / st.tap - M.Ra * e.inD + x[o+2])
    dx[o+2] = wb * (vQ / st.tap - M.Ra * e.inQ - x[o+1])
    vdc = x[od+2] / st.Cdc
    is = (st.vs - vdc) / st.Rs
    dx[o+3] = wb * (st.mstar * vdc - M.Rfd * e.ifd)
    dx[o+4] = -wb * M.R1d * e.i1d
    dx[o+5] = -wb * M.R1q * e.i1q
    M.nq == 2 && (dx[o+6] = -wb * M.R2q * e.i2q)
    w = x[od+1] / (2.0 * M.H)
    dx[od] = wb * (w - 1)
    dx[od+1] = st.Tm + e.Tem - st.Dm * (w - 1)
    dx[od+2] = is - st.mstar * e.ifd
    return dx
end

"""
    emdq2_grid_rhs!(dx, s, xg, ugD, ugQ) -> dx

The grid side's vector field at the PCC current `u_g = ugD + j ugQ` (into the grid at bus 2).
Generic in the element type.
"""
function emdq2_grid_rhs!(dx, s::EMDQPilot2, x, ugD, ugQ)
    G = emdq2_grid_layout(s)
    wb = s.m2.mach.omega_b
    e2 = _site_currents(s.m2.mach, x, G.m2)
    t2 = s.m2.tap
    v2D, v2Q = x[G.v2], x[G.v2+1]
    i23D, i23Q = x[G.i23], x[G.i23+1]
    v3D, v3Q = x[G.v3], x[G.v3+1]
    iL3D, iL3Q = x[G.iL3], x[G.iL3+1]
    i225D, i225Q = x[G.i225], x[G.i225+1]
    v25D, v25Q = x[G.v25], x[G.v25+1]
    iL25D, iL25Q = x[G.iL25], x[G.iL25+1]
    # node: C dv/dt = wb (I - j C v); branch: L di/dt = wb (dv - R i - j L i)
    dx[G.v2] = (wb / s.C2) * (ugD - i23D - i225D) + wb * v2Q
    dx[G.v2+1] = (wb / s.C2) * (ugQ - i23Q - i225Q) - wb * v2D
    dx[G.i23] = (wb / s.L23) * (v2D - v3D - s.R23 * i23D) + wb * i23Q
    dx[G.i23+1] = (wb / s.L23) * (v2Q - v3Q - s.R23 * i23Q) - wb * i23D
    I3D = i23D - s.G3 * v3D - iL3D
    I3Q = i23Q - s.G3 * v3Q - iL3Q
    if s.fixed_bus
        i34D, i34Q = x[G.i34], x[G.i34+1]
        I3D -= i34D
        I3Q -= i34Q
        dx[G.i34] = (wb / s.L34) * (v3D - real(s.V4) - s.R34 * i34D) + wb * i34Q
        dx[G.i34+1] = (wb / s.L34) * (v3Q - imag(s.V4) - s.R34 * i34Q) - wb * i34D
    end
    dx[G.v3] = (wb / s.C3) * I3D + wb * v3Q
    dx[G.v3+1] = (wb / s.C3) * I3Q - wb * v3D
    dx[G.iL3] = (wb / s.L3) * v3D + wb * iL3Q
    dx[G.iL3+1] = (wb / s.L3) * v3Q - wb * iL3D
    dx[G.i225] = (wb / s.L225) * (v2D - v25D - s.R225 * i225D) + wb * i225Q
    dx[G.i225+1] = (wb / s.L225) * (v2Q - v25Q - s.R225 * i225Q) - wb * i225D
    dx[G.v25] = (wb / s.C25) * (i225D - e2.inD / t2 - s.G25 * v25D - iL25D) + wb * v25Q
    dx[G.v25+1] = (wb / s.C25) * (i225Q - e2.inQ / t2 - s.G25 * v25Q - iL25Q) - wb * v25D
    dx[G.iL25] = (wb / s.L25) * v25D + wb * iL25Q
    dx[G.iL25+1] = (wb / s.L25) * v25Q - wb * iL25D
    _site_rhs!(dx, s.m2, x, G.m2, v25D, v25Q, e2)
    return dx
end

"""
    emdq2_rhs!(dx, s, x) -> dx

The whole two-machine pilot: machine 1 feeds the grid side at the PCC with
`u_g = -i_n1 / t1`. Generic in the element type.
"""
function emdq2_rhs!(dx, s::EMDQPilot2, x)
    nc = emdq2_nc(s)
    n = length(x)
    e1 = _site_currents(s.m1.mach, x, 0)
    t1 = s.m1.tap
    emdq2_grid_rhs!(view(dx, nc+1:n), s, view(x, nc+1:n), -e1.inD / t1, -e1.inQ / t1)
    _site_rhs!(dx, s.m1, x, 0, x[nc+1], x[nc+2], e1)
    return dx
end

emdq2_rhs(s::EMDQPilot2, x) = emdq2_rhs!(similar(x), s, x)

function _site_energy(st::EMDQSite, x, o)
    M = st.mach
    e = _site_currents(M, x, o)
    od = o + 5 + M.nq
    magnetic = (e.psid * e.id + e.psiq * e.iq + e.psifd * e.ifd + e.psi1d * e.i1d +
                e.psi1q * e.i1q + e.psi2q * e.i2q) / (2.0 * M.omega_b)
    return (; magnetic, kinetic = x[od+1]^2 / (4.0 * M.H), dc_link = x[od+2]^2 / (2.0 * st.Cdc))
end

_sq(x, k) = x[k]^2 + x[k+1]^2

"""
    emdq2_grid_energy_parts(s, xg) -> (; network, magnetic2, kinetic2, dc_link2)

The grid side's physical energy by element (pu*s), as in `emdq_energy_parts`.
"""
function emdq2_grid_energy_parts(s::EMDQPilot2, x)
    G = emdq2_grid_layout(s)
    wb = s.m2.mach.omega_b
    net = s.C2 * _sq(x, G.v2) + s.L23 * _sq(x, G.i23) + s.C3 * _sq(x, G.v3) +
          s.L3 * _sq(x, G.iL3) + s.L225 * _sq(x, G.i225) + s.C25 * _sq(x, G.v25) +
          s.L25 * _sq(x, G.iL25)
    s.fixed_bus && (net += s.L34 * _sq(x, G.i34))
    m = _site_energy(s.m2, x, G.m2)
    return (; network = net / (2.0 * wb), magnetic2 = m.magnetic, kinetic2 = m.kinetic,
            dc_link2 = m.dc_link)
end

"""    emdq2_grid_energy(s, xg) -> the grid side's physical energy `H_g`."""
emdq2_grid_energy(s::EMDQPilot2, x) = sum(emdq2_grid_energy_parts(s, x))

"""    emdq2_energy(s, x) -> the whole pilot's physical energy `H = H_c + H_g`."""
function emdq2_energy(s::EMDQPilot2, x)
    nc = emdq2_nc(s)
    return sum(_site_energy(s.m1, x, 0)) + emdq2_grid_energy(s, view(x, nc+1:length(x)))
end

function _site_power(st::EMDQSite, x, o)
    M = st.mach
    e = _site_currents(M, x, o)
    od = o + 5 + M.nq
    is = (st.vs - x[od+2] / st.Cdc) / st.Rs
    w = x[od+1] / (2.0 * M.H)
    supplies = (; shaft = (st.Tm + st.Dm) * w, dc_source = st.vs * is)
    dissipation = (; stator = M.Ra * (e.inD^2 + e.inQ^2), field = M.Rfd * e.ifd^2,
                   dampers = M.R1d * e.i1d^2 + M.R1q * e.i1q^2 + M.R2q * e.i2q^2,
                   source = st.Rs * is^2, shaft = st.Dm * w^2)
    return supplies, dissipation
end

"""
    emdq2_grid_power(s, xg, ugD, ugQ) -> (; supplies, dissipation)

The grid side's four boundary supplies (the PCC `Re(conj(u_g) v2)`, the fixed bus
`-Re(conj(i34) V4)` (zero without it), machine 2's shaft and DC source) and every loss in it.
"""
function emdq2_grid_power(s::EMDQPilot2, x, ugD, ugQ)
    G = emdq2_grid_layout(s)
    sup2, dis2 = _site_power(s.m2, x, G.m2)
    bus = s.fixed_bus ? -(x[G.i34] * real(s.V4) + x[G.i34+1] * imag(s.V4)) : zero(x[1])
    supplies = (; pcc = ugD * x[G.v2] + ugQ * x[G.v2+1], fixed_bus = bus,
                shaft2 = sup2.shaft, dc_source2 = sup2.dc_source)
    lines = s.R23 * _sq(x, G.i23) + s.R225 * _sq(x, G.i225) +
            (s.fixed_bus ? s.R34 * _sq(x, G.i34) : zero(x[1]))
    dissipation = (; lines, loads = s.G3 * _sq(x, G.v3) + s.G25 * _sq(x, G.v25),
                   stator2 = dis2.stator, field2 = dis2.field, dampers2 = dis2.dampers,
                   source2 = dis2.source, shaft2 = dis2.shaft)
    return (; supplies, dissipation)
end

"""
    emdq2_balance(s, x) -> Dict

The raw energy identities: `whole_error` for `dH/dt = supplies - losses` of the whole pilot,
`grid_error` for the grid side with its PCC supply, `pcc_sum` = the component's port power plus
the grid's (zero: the PCC conserves power).
"""
function emdq2_balance(s::EMDQPilot2, x)
    nc = emdq2_nc(s)
    n = length(x)
    dx = emdq2_rhs(s, x)
    rate = dot(ForwardDiff.gradient(z -> emdq2_energy(s, z), x), dx)
    xg = x[nc+1:n]
    e1 = _site_currents(s.m1.mach, x, 0)
    t1 = s.m1.tap
    ugD, ugQ = -e1.inD / t1, -e1.inQ / t1
    gp = emdq2_grid_power(s, xg, ugD, ugQ)
    sup1, dis1 = _site_power(s.m1, x, 0)
    pc = (e1.inD * xg[1] + e1.inQ * xg[2]) / t1
    rate_g = dot(ForwardDiff.gradient(z -> emdq2_grid_energy(s, z), xg), dx[nc+1:n])
    whole = rate - (sum(sup1) + sum(gp.supplies) - gp.supplies.pcc - sum(dis1) - sum(gp.dissipation))
    return Dict("rate" => rate, "whole_error" => whole,
                "grid_error" => rate_g - (sum(gp.supplies) - sum(gp.dissipation)),
                "pcc_sum" => pc + gp.supplies.pcc)
end

"""
    emdq2_site2_shifted(s, xg, xgs) -> (; r_rot, port, dissipation, rate)

Machine 2's shifted energy `H_m2,s(x) = H_m2(x) - H_m2(x*) - dH_m2(x*)'(x - x*)` (about the
grid-side operating point `xgs`) and its balance along the grid side's dynamics,
`rate = port - dissipation + r_rot`:
- the shifted port supply `port = Re(conj(Delta i_n2) Delta v25) / t2`;
- the incremental losses `dissipation` (stator, field, dampers, DC source, shaft damping);
- the frame-rotation residual `r_rot = -Re(conj(Delta i_n2) j Delta psi_n2)`, as in
  `emdq_shifted_balance`.

`rate` is computed from the energy gradient (Float64 only); the other terms are closed forms,
generic in the element type.
"""
function emdq2_site2_shifted(s::EMDQPilot2, x, xs)
    G = emdq2_grid_layout(s)
    st = s.m2
    M = st.mach
    o = G.m2
    od = o + 5 + M.nq
    e, es = _site_currents(M, x, o), _site_currents(M, xs, o)
    dD, dQ = e.inD - es.inD, e.inQ - es.inQ
    pD, pQ = x[o+1] - xs[o+1], x[o+2] - xs[o+2]
    r_rot = dD * pQ - dQ * pD
    port = (dD * (x[G.v25] - xs[G.v25]) + dQ * (x[G.v25+1] - xs[G.v25+1])) / st.tap
    dis = (st.vs - x[od+2] / st.Cdc) / st.Rs - (st.vs - xs[od+2] / st.Cdc) / st.Rs
    dw = (x[od+1] - xs[od+1]) / (2.0 * M.H)
    dissipation = M.Ra * (dD^2 + dQ^2) + M.Rfd * (e.ifd - es.ifd)^2 +
                  M.R1d * (e.i1d - es.i1d)^2 + M.R1q * (e.i1q - es.i1q)^2 +
                  M.R2q * (e.i2q - es.i2q)^2 + st.Rs * dis^2 + st.Dm * dw^2
    rate = if eltype(x) == Float64
        Hm(z) = sum(_site_energy(st, z, o))
        dx = similar(x)
        emdq2_grid_rhs!(dx, s, x, 0.0, 0.0)
        dot(ForwardDiff.gradient(Hm, x) - ForwardDiff.gradient(Hm, xs), dx)
    else
        nothing
    end
    return (; r_rot, port, dissipation, rate)
end

# ---------------------------------------------------------------------------------------
# Builder from the IEEE-39 case

# machine state at terminal voltage vm and motor-convention current in_ (`emdq_pilot` lines
# 393-416), with its DC excitation design values
function _site_point(M::WindingMachine, tap, vm, in_, mstar, source_drop, tau_dc)
    E = vm - complex(M.Ra, M.Ll + M.Laq) * in_
    delta = angle(E)
    rot = im * conj(_ejd(delta))
    vc, ic = rot * vm, rot * in_
    psic = -im * (vc - M.Ra * ic)
    psid, psiq = real(psic), imag(psic)
    id, iq = real(ic), imag(ic)
    ifd = (psid - (M.Ll + M.Lad) * id) / M.Lad
    ifd > 0 || error("emdq_pilot2: field current $ifd is not positive")
    psifd = M.Lad * id + (M.Lfd + M.Lad) * ifd
    psi1d = M.Lad * (id + ifd)
    psin = -im * _ejd(delta) * psic
    efd = M.Rfd * ifd
    vdc = efd / mstar
    Rs = source_drop * vdc / (mstar * ifd)
    vs = (1 + source_drop) * vdc
    Cdc = tau_dc / Rs
    xm = [real(psin), imag(psin), psifd, psi1d, M.Laq * iq]
    M.nq == 2 && push!(xm, M.Laq * iq)
    append!(xm, [delta, 2.0 * M.H, Cdc * vdc])
    site = EMDQSite(M, tap, -(psid * iq - psiq * id), vs, Rs, Cdc, Float64(mstar), 0.0)
    return site, xm, (; q_axis_residual = psiq - (M.Ll + M.Laq) * iq, ifd, vdc)
end

"""
    emdq_pilot2(eq; machine1 = "GENROU_11", trf1 = "Trf_1", line = "Line_3", tie = "Line_5",
                machine2 = "GENROU_9", trf2 = "Trf_11", branch = "Line_4", fixed_bus = true,
                mstar = 0.5, source_drop = 0.05, tau_dc = 0.02) -> (pilot, xstar, report)

The two-machine pilot from the IEEE-39 equilibrium `eq`: the chain of `emdq_pilot` (machine 1
behind `trf1` at bus a, `line` to bus b, `tie` to the fixed bus) and the branch `branch` from
bus a to a load bus c, with `machine2` behind `trf2` at bus c. Bus b's and c's demands are
constant-impedance RLC surrogates at their IEEE-39 voltages; each line's charging is split
between its end nodes. Each machine has `|V|` at its own bus from the IEEE-39 equilibrium;
with the fixed bus both machines send the power their transformers carry there, and without
it (`fixed_bus = false`, a closed two-machine system) machine 2 keeps its IEEE-39 angle as the
reference and takes the balance. Both excitations have the design values of `emdq_pilot`;
`xstar` is the exact equilibrium.
"""
function emdq_pilot2(eq; machine1::AbstractString = "GENROU_11", trf1::AbstractString = "Trf_1",
                     line::AbstractString = "Line_3", tie::AbstractString = "Line_5",
                     machine2::AbstractString = "GENROU_9", trf2::AbstractString = "Trf_11",
                     branch::AbstractString = "Line_4", fixed_bus::Bool = true,
                     mstar::Real = 0.5, source_drop::Real = 0.05, tau_dc::Real = 0.02)
    sys = eq.sys
    case = sys.case
    net = sys.net
    comp(nm) = (k = _findreq(c -> name(c) == nm, sys.comps, "no component $nm");
                c = sys.comps[k];
                c isa Union{GENSAL_PHTRUE,GENROU_PHTRUE} ||
                    throw(ArgumentError("$nm is not a GENSAL_PHTRUE or GENROU_PHTRUE"));
                c)
    c1, c2 = comp(machine1), comp(machine2)
    Vb(b) = (i = bus_index(net, b); complex(eq.V[2i-1], eq.V[2i]))
    lt1, lt2 = _case_line(case, trf1), _case_line(case, trf2)
    l23, l34, l225 = _case_line(case, line), _case_line(case, tie), _case_line(case, branch)
    (lt1.bus2 == c1.bus && l23.bus1 == lt1.bus1 && l34.bus1 == l23.bus2 &&
     l225.bus1 == lt1.bus1 && lt2.bus1 == l225.bus2 && lt2.bus2 == c2.bus) ||
        throw(ArgumentError("emdq_pilot2 expects machine1 <- $trf1 -> bus a -> $line -> bus b " *
                            "-> $tie -> boundary, and bus a -> $branch -> bus c <- $trf2 -> " *
                            "machine2 (bus1/bus2 order)"))
    for lt in (lt1, lt2)
        (lt.phi == 0.0 && lt.b == 0.0) || throw(ArgumentError("$(lt.idx): phase shift or charging"))
    end
    for c in (c1, c2)
        findfirst(l -> l.bus == c.bus, case.pq) === nothing ||
            throw(ArgumentError("emdq_pilot2: a load at machine bus $(c.bus) is not modelled"))
    end
    b2, b3, b4, b25 = l23.bus1, l23.bus2, l34.bus2, l225.bus2
    function surrogate(b)
        ld = _findreq(l -> l.bus == b, case.pq, "no load at $b")
        Pl, Ql = case.pq[ld].p0, case.pq[ld].q0
        Ql > 0 || throw(ArgumentError("bus $b demand is not inductive; the surrogate needs L"))
        V = abs(Vb(b))
        return Pl / V^2, Ql / V^2
    end
    G3, BL3 = surrogate(b3)
    G25, BL25 = surrogate(b25)
    V4 = fixed_bus ? Vb(b4) : 0.0im
    yt1, yt2 = 1.0 / complex(lt1.r, lt1.x), 1.0 / complex(lt2.r, lt2.x)
    y23, y34, y225 = 1.0 / complex(l23.r, l23.x), 1.0 / complex(l34.r, l34.x),
                     1.0 / complex(l225.r, l225.x)
    t1, t2 = lt1.tap, lt2.tap
    C2 = (l23.b + l225.b) / 2
    C3 = l23.b / 2 + (fixed_bus ? l34.b / 2 : 0.0)
    C25 = l225.b / 2
    ptrf(c, lt) = real(Vb(c.bus) * conj(1.0 / complex(lt.r, lt.x) * (Vb(c.bus) - Vb(lt.bus1) / lt.tap)))
    P1, P2 = ptrf(c1, lt1), ptrf(c2, lt2)
    Vg1, Vg2 = abs(Vb(c1.bus)), abs(Vb(c2.bus))
    th2_ref = angle(Vb(c2.bus))
    yb = fixed_bus ? y34 : 0.0im
    function flow(th1, th2)
        Vg1c, Vg2c = Vg1 * _ejd(th1), Vg2 * _ejd(th2)
        A = [yt1/t1^2+im*C2+y23+y225 -y23 -y225;
             -y23 y23+im*C3+G3-im*BL3+yb 0;
             -y225 0 y225+yt2/t2^2+im*C25+G25-im*BL25]
        V2, V3, V25 = A \ [yt1 / t1 * Vg1c, yb * V4, yt2 / t2 * Vg2c]
        Ig1, Ig2 = yt1 * (Vg1c - V2 / t1), yt2 * (Vg2c - V25 / t2)
        return (; Vg1c, Vg2c, V2, V3, V25, Ig1, Ig2)
    end
    Pm(r) = (real(r.Vg1c * conj(r.Ig1)), real(r.Vg2c * conj(r.Ig2)))
    th = [angle(Vb(c1.bus)), th2_ref]
    h = 1e-7
    F2(t) = collect(Pm(flow(t[1], t[2]))) .- [P1, P2]
    F1(t) = Pm(flow(t, th2_ref))[1] - P1
    for _ in 1:50
        if fixed_bus
            J = hcat((F2(th .+ [h, 0]) - F2(th .- [h, 0])) / 2h, (F2(th .+ [0, h]) - F2(th .- [0, h])) / 2h)
            step = J \ F2(th)
        else
            step = [F1(th[1]) / ((F1(th[1] + h) - F1(th[1] - h)) / 2h), 0.0]
        end
        th .-= step
        maximum(abs, step) < 1e-15 && break
    end
    r = flow(th[1], th[2])
    P1r, P2r = Pm(r)
    (abs(P1r - P1) < 1e-10 && (!fixed_bus || abs(P2r - P2) < 1e-10)) ||
        error("emdq_pilot2: the pilot power flow did not converge")
    M1 = winding_machine(c1.p; x_ext = lt1.x, r_ext = lt1.r)
    M2 = winding_machine(c2.p; x_ext = lt2.x, r_ext = lt2.r)
    s1, xm1, rep1 = _site_point(M1, t1, r.V2 / t1, -r.Ig1, mstar, source_drop, tau_dc)
    s2, xm2, rep2 = _site_point(M2, t2, r.V25 / t2, -r.Ig2, mstar, source_drop, tau_dc)
    pilot = EMDQPilot2(s1, s2, C2, l23.r, l23.x, C3, G3, 1.0 / BL3, l34.r, l34.x, V4,
                       l225.r, l225.x, C25, G25, 1.0 / BL25, fixed_bus)
    i23, i225 = y23 * (r.V2 - r.V3), y225 * (r.V2 - r.V25)
    iL3, iL25 = r.V3 / (im / BL3), r.V25 / (im / BL25)
    pairs = [r.V2, i23, r.V3]
    fixed_bus && push!(pairs, y34 * (r.V3 - V4))
    append!(pairs, [iL3, i225, r.V25, iL25])
    xnet = reduce(vcat, [[real(z), imag(z)] for z in pairs])
    xstar = vcat(xm1, xnet, xm2)
    report = (; buses = (; machine1 = c1.bus, pcc = b2, b3, fixed = fixed_bus ? b4 : nothing,
                         b25, machine2 = c2.bus),
              P1 = P1r, P2 = P2r, P1_case = P1, P2_case = P2, V2 = r.V2, V3 = r.V3, V25 = r.V25,
              Vg1 = r.Vg1c, Vg2 = r.Vg2c, machine1 = rep1, machine2 = rep2)
    return pilot, xstar, report
end
