# The port-Hamiltonian structure of a model, assembled from its components (PH_TODO.md step 1; review 86).
#
# `phs_structure(m, x, u)` returns, at the state `x`, the matrices of
#     x_I' = (J(x) - R(x)) z(x) + G(x) v,      y = G' z,
# on the included rows `I` (the physical states of the components that have a port-Hamiltonian form), with `z` the
# gradient of the included energy `phs_structure_energy` and `v` the port inputs (their names in `ports`). Everything
# else is outside `I` and named in `excluded`: controllers (signal laws, whose outputs enter as inputs `v` or as
# modulations in `J` and `R`), the static loads' lags and the relays' counters (signal states), and the components
# without a canonical form yet (steam units, brushless exciters, travelling-wave lines; the switching bridge is in
# the form over each step, B3, and the lines have their own discrete realisation, B2).
#
# The assembly is independent of `phs_rhs!`: each component adds its storage effort, its interconnection, its
# dissipation and its ports, from its own constitutive data. The network is first built unreduced, as the
# port-Hamiltonian DAE of docs/presentation/porthos_component_models.tex eq. (ph-dae): the voltages without storage
# (buses without capacitance, zero nodes without capacitance, the zero voltage of a grounded fault at a bus without a
# zero node) are efforts `w` of zero-storage nodes with their conductances. They are eliminated exactly,
#     [A B; C D] [z; w],  0 = C z + D w  ->  M = A - B D^{-1} C,  J = (M - M')/2,  R = -(M + M')/2,
# which is the descriptor form's reduction: R is the Schur complement of the conductances (positive semidefinite
# when they are) plus B_J G_ww^{-1} B_J', and J stays skew. The reduced inductor-only nodes (Mayer-Wasynczuk) enter
# as in `phs_matrices`. Signals are evaluated at `x` with no limiter decision held (`NoModes`).

# the complex coefficient zc on a complex variable, as a 2 x 2 real block of M at (r, c)
@inline function _cblk!(M, r, c, zc)
    M[r, c] += real(zc); M[r, c+1] -= imag(zc)
    M[r+1, c] += imag(zc); M[r+1, c+1] += real(zc)
end
# a skew pair between a complex row pair r and a real state c: row r gets a * z_c, row c gets -Re(conj(a) z_r)
@inline function _cskew!(J, r, c, a)
    J[r, c] += real(a); J[r+1, c] += imag(a)
    J[c, r] -= real(a); J[c, r+1] -= imag(a)
end
# the load torque over the speed, without dividing by the speed (T_L omega >= 0)
@inline _tl_over_omega(c::MotorCircuit, omega) =
    c.etrq == 2 ? c.Tm0 * abs(omega) : c.Tm0 / sqrt(omega^2 + c.eps^2)

"""
    phs_structure_energy(m, x) -> H_I(x)

The energy of the included components of `phs_structure`: the grid model's `phs_energy`, the DC shunt exciters'
fields, the hydro units' water columns and the converters' averaged DC sides (the steam units and brushless exciters
are excluded).
"""
phs_structure_energy(s::PHSystem, x::AbstractVector) = phs_energy(s, view(x, 1:s.n))
function phs_structure_energy(cs::PHControlled, x::AbstractVector; steam::Symbol = :exclude)
    H = phs_energy(cs.s, view(x, 1:cs.s.n))
    if steam === :exergy                         # c H_ex from the tables (B4)
        for j in _steam_units(cs)
            g, o = cs.gov[j], cs.goff[j]
            sc = steam_sat(g.tb, g.p_cond); T0 = sc.T; mu0 = sc.h_f - T0 * sc.s_f; p0 = g.p_cond
            for k in eachindex(g.V)
                M, U = x[o+2k], x[o+2k+1]
                st = steam_state(g.tb, M / g.V[k], U / M)
                H += cs.units[j] / STEAM_SB * (U - T0 * M * st.s + p0 * g.V[k] - mu0 * M)
            end
        end
    end
    for j in eachindex(cs.exc)
        cs.exc[j] isa PHShuntExciter && (H += _exc_energy(cs.exc[j], x[cs.xoff[j]]))
    end
    for j in _hydro_units(cs)
        H += cs.units[j] * _hydro_powers(cs.gov[j], _gov_states(cs, x, j)).energy
    end
    for i in _dc_converters(cs)
        H += _dc_energy(cs.conv[i], view(x, cs.coff[i]:cs.coff[i]+nstates(cs.conv[i])-1), cs.s.omega_b)
    end
    return H
end

"""
    phs_structure(m, x, u) -> (; J, R, G, v, z, idx, ports, excluded)

The port-Hamiltonian structure at `x` (file header), with matrices over all states of `m` (zero outside the
included rows `idx`). `m` is a `PHSystem` (then `u` is its input) or a `PHControlled` (then `u` carries the fault
conductances and the frame phasor; the controllers set the rest). `ports[k]` names input `v[k]`; `excluded` names
what is outside `idx`.
"""
function phs_structure(s::PHSystem, x::AbstractVector{Float64}, u::PHInput)
    st = _structure_core(s, x, u)
    return _structure_finish(st, length(x))
end

function phs_structure(cs::PHControlled, x::AbstractVector{Float64}, u::PHInput; steam::Symbol = :exclude)
    steam in (:exclude, :exergy) || throw(ArgumentError("phs_structure: steam = $steam (:exclude or :exergy)"))
    s = cs.s
    up = phs_control_inputs(cs, x, u.gf; cf = u.cf)
    f = phs_flows(s, x, up)
    # the converters whose DC side is in the form: the bridge as J(m), not the EMF input
    dcin = Dict{Int,Any}()
    for i in _dc_converters(cs)
        g = cs.conv[i]
        xc = view(x, cs.coff[i]:cs.coff[i]+nstates(g)-1)
        ic = _conv_current(f, s.conv[i])
        if _sw_on(g)                       # the switching bridge over the step (B3): m-bar, Rac, Gsw
            sw = bridge_structure(g.dc.sw, _sw_terms(g, xc, ic), ic, xc[_dcidx(g)])
            dcin[i] = (; g, o = cs.coff[i] + _dcidx(g) - 1, m = sw.m, xc, sw)
        else
            dcin[i] = (; g, o = cs.coff[i] + _dcidx(g) - 1, m = _modulation(g, xc, ic), xc, sw = nothing)
        end
    end
    shunt = [j for j in eachindex(cs.exc) if cs.exc[j] isa PHShuntExciter]
    # the shaft inputs: a DC shunt exciter's shaft power and a hydro unit's turbine are couplings in J, not inputs
    Tm = copy(up.Tm)
    for j in shunt
        Tm[j] += _exc_at(cs, x, j).shaft
    end
    for j in _hydro_units(cs)
        Tm[j] -= cs.units[j] * _hydro_power(cs.gov[j], _gov_states(cs, x, j))
    end
    if steam === :exergy                         # the steam units' shafts: their turbines are couplings in J (B4)
        for j in _steam_units(cs)
            Tm[j] -= cs.units[j] * _steam_power(cs.gov[j], _gov_states(cs, x, j))
        end
    end
    up = PHInput{Float64}(Tm, up.vf, up.gf, up.const_power, up.uc, up.cf)
    st = _structure_core(s, x, up; n = cs.n, dcin, shunt)
    J, R, G, v, z, idx, ports, excluded = st.J, st.R, st.G, st.v, st.z, st.idx, st.ports, st.excluded
    wb = s.omega_b
    addport!(name, val) = (push!(ports, name); push!(v, val); G = hcat(G, zeros(cs.n)); size(G, 2))
    # DC shunt exciters with their machine's field and rotor: phi_f' = v_f + b VR - r_F i_f, v_f = kappa omega,
    # lambda_fd' = v_f - Rfd i_fd, p' gets -kappa (i_fd + i_f) (the shaft power v_f (i_fd + i_f) over omega)
    for j in shunt
        e, mch, o = cs.exc[j], s.machines[j], cs.xoff[j]
        phi, ip = x[o], _ip(mch)
        q = _exc_at(cs, x, j)
        kap = phi / (e.n * e.TE)
        J[o, ip] += kap; J[ip, o] -= kap
        J[mch.offset+2, ip] += kap; J[ip, mch.offset+2] -= kap
        R[o, o] += e.rF
        z[o] = q.i_f
        c = addport!("$(name(e)).amplifier", _exc_amp_volts(e, x[o+1])); G[o, c] = 1.0
        push!(idx, o)
        push!(excluded, "$(name(e)).xa", "$(name(e)).xf")
    end
    # hydro units: the penstock (ladder J, friction R) from the reservoir datum, the ideal turbine as a skew coupling
    # of the turbine head with the rotor, and the reservoir's head (1 pu) as the input whose output is the flow
    for j in _hydro_units(cs)
        g, nu, o = cs.gov[j], cs.units[j], cs.goff[j]
        mch = s.machines[j]; ip = _ip(mch)
        om = x[ip] / (2 * mch.w.H)
        at = _hydro_gate(g, x[o+2])
        K = nu * g.GMAX
        c = addport!("$(name(g)).reservoir", 1.0)
        if g.N == 0
            iq = o + 3
            qv = x[iq]; h = _hydro_head(g, at, qv)
            z[iq] = K * g.Tw * qv
            b = h / (g.Tw * om)
            J[iq, ip] -= b; J[ip, iq] += b
            G[iq, c] = 1 / g.Tw
            push!(idx, iq)
        else
            N, M, r = g.N, g.Tw / g.N, g.fp / g.N
            iq(k) = o + 2 + k; ih(k) = o + 2 + N + k
            for k in 1:N
                Ck = _penstock_C(g, k)
                z[iq(k)] = K * M * x[iq(k)]
                z[ih(k)] = K * Ck * (x[ih(k)] - 1)
                a = 1 / (K * M * Ck)                  # q_k' gets -(h_k - 1)/M, h_k' gets q_k / C_k
                J[iq(k), ih(k)] -= a; J[ih(k), iq(k)] += a
                if k < N                              # h_k' gets -q_{k+1}/C_k, q_{k+1}' gets (h_k - 1)/M
                    J[ih(k), iq(k + 1)] -= a; J[iq(k + 1), ih(k)] += a
                end
                R[iq(k), iq(k)] += r * abs(x[iq(k)]) / (K * M^2)
                push!(idx, iq(k), ih(k))
            end
            hN = x[ih(N)]; qt = _turbine_flow(g, at, hN)
            CN = _penstock_C(g, N)
            a = qt / (CN * om)                        # h_N' gets -q_t/C_N, p' gets q_t (h_N - 1) K / omega
            J[ih(N), ip] -= a; J[ip, ih(N)] += a
            G[ip, c] = K * qt / om                    # the reservoir's head at the turbine: K q_t / omega on the rotor
        end
        push!(excluded, "$(name(g)).governor")
    end
    # converters' averaged DC sides: link and battery branches in the battery's Norton form (E_b/R_b a current source)
    for (i, d) in dcin
        g, o = d.g, d.o
        q = g.dc
        nb = nbranches(q)
        rows = [o; [o + k for k in 1:nb]]
        caps = [q.c; q.cb]
        wv = wb ./ caps
        for (a, ra) in enumerate(rows)
            z[ra] = caps[a] * x[ra] / wb
            for (b, rb) in enumerate(rows)
                R[ra, rb] += wv[a] * wv[b] / q.Rb
            end
        end
        R[o, o] += q.G * wv[1]^2
        for k in 1:nb
            R[o+k, o+k] += wv[k+1]^2 / q.R[k]
        end
        c = addport!("$(cs.conv[i].name).battery", q.Eb / q.Rb)
        for (a, ra) in enumerate(rows)
            G[ra, c] = wv[a]
        end
        # the bridge: u_c = m v_dc = m (omega_b / c) z_dc on every row the source row's voltage drives
        for (r, coef) in st.ucrows[i]
            _cskew!(J, r, o, coef * d.m * wv[1])
        end
        if d.sw !== nothing
            # the drops: the rows driven by coef_r u_c get -coef_r Rac i_c, with i_c = sum_r' conj(coef_r') (z_r' + j z_r'+1):
            # R += B Rac B' over the driven rows (B the real 2 x 2 blocks of multiplication by coef); the switching sink
            # Gsw v_dc on the link
            cm(c) = [real(c) -imag(c); imag(c) real(c)]
            for (r1, c1) in st.ucrows[i], (r2, c2) in st.ucrows[i]
                R[r1:r1+1, r2:r2+1] .+= cm(c1) * d.sw.Rac * cm(c2)'
            end
            R[o, o] += d.sw.Gsw * wv[1]^2
        end
        append!(idx, rows)
    end
    # steam units in exergetic form (B4; docs/active/PH_B4_EXERGY.md, section 2b): the unit-level realisation
    # (`steam_exergy_structure`, SI) mapped with A = diag(I on the steam rows, c on the rotor), c = units / STEAM_SB:
    # z = (c e, omega) at the rotor effort x_p / (2H); J unchanged; R -> A R A' / c = [R_ff/c R_fp; R_pf c R_pp]; the
    # throttle and walls as feedthrough ports (P/c, S/c, inputs c u); the extractions as ports (G, u = q_e, output
    # -c eps_k); the elements' violations (flowing branches with affinity <= 0, table defect) returned
    Pf = zeros(cs.n, 0); Sf = zeros(0, 0); vf = Float64[]; fports = String[]; steam_violations = NamedTuple[]
    if steam === :exergy
        for j in _steam_units(cs)
            g, o, mch = cs.gov[j], cs.goff[j], s.machines[j]
            ip = _ip(mch)
            om = x[ip] / (2 * mch.w.H)
            ns = length(g.V)
            rows = o+2:o+1+2ns
            c = cs.units[j] / STEAM_SB
            S = steam_exergy_structure(g, x[rows], _steam_stroke(g, x[o+1]), om)
            append!(steam_violations, [merge(v, (; unit = name(g))) for v in S.violations])
            nf = S.nf
            z[rows] .= c .* S.e
            J[rows, ip] .+= S.J[1:nf, nf+1]; J[ip, rows] .+= S.J[nf+1, 1:nf]
            R[rows, rows] .+= S.R[1:nf, 1:nf] ./ c
            R[rows, ip] .+= S.R[1:nf, nf+1]; R[ip, rows] .+= S.R[nf+1, 1:nf]
            R[ip, ip] += c * S.R[nf+1, nf+1]
            # feedthrough: the throttle (2 efforts) and the reheater walls
            np_ = 2 + length(S.uw)
            Pnew = zeros(cs.n, np_)
            Pnew[rows, :] .= hcat(S.PT, S.Pw) ./ c
            Pf = hcat(Pf, Pnew)
            Snew = cat(S.ST, S.Sw; dims = (1, 2)) ./ c
            Sf = cat(Sf, Snew; dims = (1, 2))
            append!(vf, c .* vcat(S.uT, S.uw))
            append!(fports, ["$(name(g)).throttle_M", "$(name(g)).throttle_U"],
                    ["$(name(g)).wall_$(g.names[k])" for k in eachindex(g.kh) if g.kh[k] > 0])
            # the extractions: outflow ports at the local state
            for k in 1:ns
                col = addport!("$(name(g)).extraction_$(g.names[k])", S.ue[k])
                G[rows, col] .= S.Ge[:, k]
            end
            append!(idx, rows)
        end
    end
    # every other state of the controllers and units is outside the form
    for j in eachindex(cs.exc)
        e = cs.exc[j]
        e === nothing || e isa PHShuntExciter || push!(excluded, e isa PHBrushlessExciter ?
            "$(name(e)) (brushless exciter: no canonical form; its field voltage and shaft power are inputs)" :
            "$(name(e)) (signal exciter)")
    end
    for j in eachindex(cs.gov)
        g = cs.gov[j]
        g isa PHHydroPlant && (push!(excluded, "$(name(g)) (hydropower plant: Euler turbine, no canonical form here)"); continue)
        g === nothing || g isa PHHydroUnit || push!(excluded, g isa PHSteamUnit ?
            (steam === :exergy ? "$(name(g)).governor" :
             "$(name(g)) (steam unit: excluded by default; steam = :exergy gives its exergetic form)") :
            "$(name(g)) (signal governor)")
    end
    for i in eachindex(cs.conv)
        push!(excluded, "$(cs.conv[i].name).controller")
    end
    return _structure_finish((; J, R, G, v, z, idx, ports, excluded, Pf, Sf, vf, fports, steam_violations), cs.n)
end

function _structure_finish(st, n)
    idx = sort(unique(st.idx))
    out = setdiff(1:n, idx)
    J, R, G = st.J, st.R, st.G
    (all(iszero, J[out, :]) && all(iszero, J[:, out]) && all(iszero, R[out, :]) && all(iszero, R[:, out]) &&
     all(iszero, G[out, :])) || throw(ErrorException("phs_structure: an excluded state carries structure"))
    Pf = hasproperty(st, :Pf) ? st.Pf : zeros(n, 0)
    all(iszero, Pf[out, :]) || throw(ErrorException("phs_structure: an excluded state carries a feedthrough"))
    z = st.z; z[out] .= 0
    # the feedthrough ports (steam = :exergy): x_I' = [(J - R) z + G v - Pf vf]_I, yf = Pf' z + Sf vf, with
    # [[R, Pf], [Pf', Sf]] >= 0 (empty otherwise)
    return (; J, R, G, v = st.v, z, idx, ports = st.ports, excluded = st.excluded, Pf,
            Sf = hasproperty(st, :Sf) ? st.Sf : zeros(0, 0), vf = hasproperty(st, :vf) ? st.vf : Float64[],
            fports = hasproperty(st, :fports) ? st.fports : String[],
            steam_violations = hasproperty(st, :steam_violations) ? st.steam_violations : NamedTuple[])
end

# the grid model's part, unreduced over (states, algebraic voltages), then reduced
function _structure_core(s::PHSystem, x, u; n = s.n, dcin = Dict{Int,Any}(), shunt = Int[])
    isempty(s.tw) || throw(ArgumentError("phs_structure: travelling-wave lines have no canonical form (PH_TODO.md B2)"))
    pn, wb = s.net, s.omega_b
    f = phs_flows(s, x, u)
    excluded = String[]
    # the algebraic voltages: bus rows without a voltage state, zero nodes without capacitance, the zero voltage of a
    # grounded fault at a bus without a zero node
    F = _fault_matrices(s)
    z0 = s.z0
    wcol = Dict{Int,Int}(); nw = 0
    for k in s.alg; wcol[k] = n + nw + 1; nw += 2; end
    w0col = Dict{Int,Int}()
    if z0 !== nothing
        for q in z0.alg0; w0col[q] = n + nw + 1; nw += 1; end
    end
    wfcol = Dict{Int,Int}()
    for (k, fk) in F
        (z0 === nothing || z0.zrow[k] == 0) && fk.ground && fk.G[3, 3] > 0 && (wfcol[k] = n + nw + 1; nw += 1)
    end
    N = n + nw
    J, R = zeros(N, N), zeros(N, N)
    ports, v = String[], Float64[]
    Gcols = Vector{Vector{Float64}}()
    newport!(name, val) = (push!(ports, name); push!(v, val); push!(Gcols, zeros(N)); length(Gcols))
    idx = Int[]
    ucrows = Dict(i => Tuple{Int,ComplexF64}[] for i in eachindex(s.conv))
    convof = Dict(cv.row => i for (i, cv) in enumerate(s.conv))
    ucol = Dict{Int,Int}()
    for (i, cv) in enumerate(s.conv)
        haskey(dcin, i) && continue
        ucol[i] = newport!("$(cv.name).emf_re", real(u.uc[i]))
        newport!("$(cv.name).emf_im", imag(u.uc[i]))
    end
    # an element's complex row pair r driven by coef * v_k; the node k receives -conj(coef) * effort
    function attach!(r, k, coef)
        if haskey(convof, k)                     # the converter's source row: its EMF is an input (or the bridge)
            i = convof[k]
            push!(ucrows[i], (r, ComplexF64(coef)))
            haskey(ucol, i) || return
            c = ucol[i]
            g1, g2 = Gcols[c], Gcols[c+1]
            g1[r] += real(coef); g1[r+1] += imag(coef)
            g2[r] -= imag(coef); g2[r+1] += real(coef)
        else
            col = s.qidx[k] > 0 ? s.qidx[k] : wcol[k]
            _cblk!(J, r, col, coef)
            _cblk!(J, col, r, -conj(coef))
        end
    end
    fr = newport!("frame", 1.0)                   # the frame's rotation: the angles' clock, the cores' rotation
    # buses
    gstat(k) = s.sl[k] == 0 ? 0.0 : f.gs[s.sl[k]]
    for k in s.dyn
        q = s.qidx[k]
        s.stationary || _cblk!(J, q, q, -im * pn.C[k])
        R[q, q] += pn.G[k] + u.gf[k] + gstat(k); R[q+1, q+1] += pn.G[k] + u.gf[k] + gstat(k)
        push!(idx, q, q + 1)
    end
    for k in s.alg
        c = wcol[k]
        R[c, c] += pn.G[k] + u.gf[k] + gstat(k); R[c+1, c+1] += pn.G[k] + u.gf[k] + gstat(k)
    end
    # branches (a cored branch: below)
    cored = Dict(q.pos => q for q in s.cores)
    for (j, l) in enumerate(s.branches)
        (s.bcut[j] == 0 && !haskey(cored, j)) || continue
        r = s.bidx[j]
        s.stationary || _cblk!(J, r, r, -im * pn.L[l])
        R[r, r] += pn.R[l]; R[r+1, r+1] += pn.R[l]
        attach!(r, pn.from[l], 1 / pn.a[l])
        attach!(r, pn.to[l], -1.0 + 0im)
        push!(idx, r, r + 1)
    end
    # series capacitors: the frame's -j C on the charge pair, coupled to their branch (-vc on the branch, +i on the pair)
    for (a, j) in enumerate(s.cpos)
        rc, rb = s.cidx[a], s.bidx[j]
        s.stationary || _cblk!(J, rc, rc, -im / _scap_x(s, a))
        _cblk!(J, rb, rc, -1.0 + 0im); _cblk!(J, rc, rb, 1.0 + 0im)
        push!(idx, rc, rc + 1)
    end
    # saturating cores (transformer.jl; the chapter's eq. tr:canonical): J = 0 within the branch, R = diag(R2, R1),
    # the HV loop driven by v_from / a, the source loop by v_to, the rotation and the clock a port of zero power
    for q in s.cores
        l, j = s.branches[q.pos], q.pos
        r, rs = s.bidx[j], q.xs
        R[r, r] += q.R2; R[r+1, r+1] += q.R2
        R[rs, rs] += q.R1; R[rs+1, rs+1] += q.R1
        attach!(r, pn.from[l], 1 / pn.a[l])
        attach!(rs, pn.to[l], 1.0 + 0im)
        gH, gS = -im * wb * _cx(x, r), -im * wb * _cx(x, rs)
        Gcols[fr][r] += real(gH); Gcols[fr][r+1] += imag(gH)
        Gcols[fr][rs] += real(gS); Gcols[fr][rs+1] += imag(gS)
        push!(idx, r, r + 1, rs, rs + 1)
    end
    if s.iclk > 0
        Gcols[fr][s.iclk] += wb
        push!(idx, s.iclk)
    end
    # the branches of a reduced inductor-only node without a machine
    for c in s.cuts
        c.machine == 0 || continue
        for (a, j) in enumerate(c.pos)
            r = s.bidx[j]
            for (b, jb) in enumerate(c.pos)
                rb = s.bidx[jb]
                s.stationary || _cblk!(J, r, rb, -im * c.M[a, b])
                R[r, rb] += c.Rm[a, b]; R[r+1, rb+1] += c.Rm[a, b]
            end
            attach!(r, c.far[a], complex(c.c[a]))
            push!(idx, r, r + 1)
        end
    end
    # shunt inductors
    for (j, k) in enumerate(pn.sh_bus)
        s.sidx[j] > 0 || continue
        r = s.sidx[j]
        s.stationary || _cblk!(J, r, r, -im * pn.Lsh[j])
        R[r, r] += s.Rsh[j]; R[r+1, r+1] += s.Rsh[j]
        attach!(r, k, 1.0 + 0im)
        push!(idx, r, r + 1)
    end
    # machines: windings, rotor and angle; field voltage and shaft power as inputs (a DC shunt exciter's field
    # voltage and shaft power are its coupling instead: `phs_structure(::PHControlled)`)
    for (j, m) in enumerate(s.machines)
        w, o, ip, id = m.w, m.offset, _ip(m), _idelta(m)
        delta = x[id]
        crot = s.stationary ? conj(complex(cos(delta), sin(delta)) * u.cf[1]) : complex(cos(delta), -sin(delta))
        J[o, ip] += wb * x[o+1]; J[ip, o] -= wb * x[o+1]
        J[o+1, ip] -= wb * x[o]; J[ip, o+1] += wb * x[o]
        J[id, ip] += wb; J[ip, id] -= wb
        if m.cut == 0
            attach!(o, m.node, im * crot / m.a)
            R[o, o] += w.Ra; R[o+1, o+1] += w.Ra
        else
            q, t = s.cuts[m.cut], m.tidx
            attach!(t, q.far[1], q.c[1] * im * crot)
            J[t, ip] += wb * x[t+1]; J[ip, t] -= wb * x[t+1]
            J[t+1, ip] -= wb * x[t]; J[ip, t+1] += wb * x[t]
            R[o, o] += q.Rm[1, 1]; R[o+1, o+1] += q.Rm[1, 1]
            R[t, t] += q.Rm[2, 2]; R[t+1, t+1] += q.Rm[2, 2]
            R[o, t] += q.Rm[1, 2]; R[t, o] += q.Rm[1, 2]; R[o+1, t+1] += q.Rm[1, 2]; R[t+1, o+1] += q.Rm[1, 2]
            push!(idx, t, t + 1)
        end
        R[o+2, o+2] += w.Rfd; R[o+3, o+3] += w.R1d; R[o+4, o+4] += w.R1q
        w.nq == 2 && (R[o+5, o+5] += w.R2q)
        omega = x[ip] / (2 * w.H)
        if !(j in shunt)
            c = newport!("machine$(j).field", u.vf[j]); Gcols[c][o+2] = 1.0
        end
        c = newport!("machine$(j).shaft", u.Tm[j])
        Gcols[c][ip] = u.const_power ? 1 / omega : 1.0
        Gcols[fr][id] -= wb
        append!(idx, o:o+phs_nstates(m)-1)
    end
    # composite loads: induction motors (sigma, the relay's remaining fraction, frozen) and static conductances
    for mo in s.motors
        q, o = _motor_state(mo, x, wb), mo.offset
        mo.ridx > 0 && push!(excluded, "$(mo.name).relay (sampled signal states)")
        q.on || (push!(excluded, "$(mo.name) (every unit tripped)"); continue)
        sg, c = q.sg, mo.c
        rows = (o, o + 2, o + 4)
        attach!(o, mo.node, 1.0 + 0im)
        for (a, Ra) in enumerate((c.Rs, c.R1, c.R2))
            R[rows[a], rows[a]] += Ra / sg; R[rows[a]+1, rows[a]+1] += Ra / sg
        end
        if !s.stationary
            for a in 1:3, b in 1:3
                _cblk!(J, rows[a], rows[b], -im * c.L[a, b] / sg)
            end
        end
        for (a, lk) in ((2, q.l1), (3, q.l2))
            _cskew!(J, rows[a], o + 6, im * wb * lk)
        end
        R[o+6, o+6] += sg * _tl_over_omega(c, q.omega)
        append!(idx, o:o+6)
    end
    for stc in s.statics
        push!(excluded, "$(stc.name).lag (signal state; its conductance is in R)")
    end
    # the zero axis: no rotation; its algebraic nodes eliminated with their conductance
    if z0 !== nothing
        zc(q) = z0.qidx0[q] > 0 ? z0.qidx0[q] : w0col[q]
        for q in z0.dyn0
            R[z0.qidx0[q], z0.qidx0[q]] += z0.G0[q]
            push!(idx, z0.qidx0[q])
        end
        for q in z0.alg0
            R[w0col[q], w0col[q]] += z0.G0[q]
        end
        for (b, r) in zip(z0.br, z0.bidx0)
            R[r, r] += b.R
            J[r, zc(b.from)] += 1; J[zc(b.from), r] -= 1
            J[r, zc(b.to)] -= 1; J[zc(b.to), r] += 1
            push!(idx, r)
        end
        for (h, r) in zip(z0.sh, z0.sidx0)
            R[r, r] += h.R
            J[r, zc(h.node)] += 1; J[zc(h.node), r] -= 1
            push!(idx, r)
        end
    end
    # phase-asymmetric faults: the dq0 conductance on (v_d, v_q, v_0) of the bus; without a zero node i_0 = 0
    for (k, fk) in F
        G3 = fk.G
        cdq = s.qidx[k] > 0 ? s.qidx[k] : wcol[k]
        if z0 !== nothing && z0.zrow[k] > 0
            q = z0.zrow[k]
            cols = (cdq, cdq + 1, z0.qidx0[q] > 0 ? z0.qidx0[q] : w0col[q])
            for a in 1:3, b in 1:3
                R[cols[a], cols[b]] += G3[a, b]
            end
        elseif haskey(wfcol, k)
            cols = (cdq, cdq + 1, wfcol[k])
            for a in 1:3, b in 1:3
                R[cols[a], cols[b]] += G3[a, b]
            end
        else
            for a in 1:2, b in 1:2
                R[cdq+a-1, cdq+b-1] += G3[a, b]
            end
        end
    end
    # the reduction of the algebraic voltages
    Gf = isempty(Gcols) ? zeros(N, 0) : reduce(hcat, Gcols)
    Mf = J .- R
    S, W = 1:n, n+1:N
    if nw > 0
        D = Mf[W, W]
        all(iszero, Gf[W, :]) || throw(ErrorException("phs_structure: an input enters an algebraic row"))
        Mr = Mf[S, S] .- Mf[S, W] * (D \ Mf[W, S])
    else
        Mr = Mf[S, S]
    end
    Jr = (Mr .- Mr') ./ 2
    Rr = .-(Mr .+ Mr') ./ 2
    z = zeros(n)
    z[1:s.n] .= phs_gradient(s, view(x, 1:s.n))
    return (; J = Jr, R = Rr, G = Gf[S, :], v, z, idx, ports, excluded, ucrows)
end
