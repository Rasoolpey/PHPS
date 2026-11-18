# =====================================================================================================================
# The network block (buses, lines, transformers, series R-L loads) of the 464-state EMT model, real-time style
# =====================================================================================================================
#
# Record: docs/realtime/models/NETWORK.md; reviews 35 (step method) and 35b (architecture). Loaded by src/rt/RT.jl;
# `julia --project=. scripts/realtime/export_cpp.jl network` generates the board top (network_gen.jl, @hls), the C++,
# the ROM tables, the testbench, the board program and flow.cfg in src/rt/network/CPP/network/.
#
# Model:   the IEEE-39 network of `PHSystem` (src/phsim/system.jl) in the stationary frame; per axis (alpha = beta, no
#          cross terms) x' = A x + B u with 79 states (28 node charges q = C v / omega_b, 51 inductor fluxes
#          phi = L i / omega_b); data src/rt/network/data/ieee39_network.toml (scripts/realtime/network_extract.jl,
#          BigFloat). Inputs: the currents the machines inject at 9 buses. Fault bank: bus 16 to ground, 0.001 pu.
# Step:    the exact implicit midpoint in nodal form, folding F3 (NETWORK.md section 3; review 35b):
#            P1  rq  = 2^m_rq (G1 x + B1 um),  rp = 2^m_rp G2 x,  the bench's phasor rotated (two more rows)
#            P2  dq  = 2^m_dq S^-1 rq,         x_q += dq
#            P3  x_p += rp + F dq
#          G1 = hA_q + E hA_p, G2 = Dinv hA_p, E = -M_qp Dinv, S = M_qq + E M_pq, F = -Dinv M_pq, M = I - hA/2,
#          prepared in BigFloat per bank and rounded once; the increments stored scaled by powers of two in the
#          states' 48-bit format, so every vector keeps its full precision (exact bookkeeping in Julia and C++).
# Engine:  8 lanes x 2 axes = 16 multipliers, one coefficient per lane per cycle serving both axes; every row owned
#          by one lane (row ownership), the lanes' schedules in ROM tables; operands in banked LUTRAM (bank = the
#          writing lane); three pipelined phase loops (II = 1) per step, the multipliers shared by ALLOCATION.
# =====================================================================================================================

import TOML, Random
using LinearAlgebra: I, Diagonal, Symmetric, eigvals, norm

const NETWORK_RT_DIR = joinpath(@__DIR__)
const NETWORK_RT_LANES = 8
const NETWORK_RT_CLOCK_NS = 8
const NETWORK_RT_STEP_CYCLES = 108            # h = cycles x 8 ns (measured on the board, every-clock ILA; NETWORK.md 7)
const NETWORK_RT_GROUPS = 2                   # lane groups, each with its own read port of every bank (review 37)
const NETWORK_RT_SEEDS = 5000                 # the bounded schedule search (review 37, 5)
const NETWORK_RT_RAW_D = 12                   # iterations from a bank write to a read of it (certified against the
                                              # synthesised schedule, review 36b; scripts/realtime/network_hwcert.py)
const NETWORK_RT_H = "$(NETWORK_RT_STEP_CYCLES * NETWORK_RT_CLOCK_NS)e-9"

# number types of the @hls code (Float64 / Int in Julia; the C++ formats are chosen at export, NETWORK.md 5)
const X = Float64; const C = Float64
const P = Float64; const Acc = Float64
const Ib = Int

# ---------------------------------------------------------------------------------------------------------------------
# 1. Data and preparation (BigFloat)
# ---------------------------------------------------------------------------------------------------------------------
"""
    network_rt_data(; path) -> NamedTuple

The network of the 464-state IEEE-39 model from `src/rt/network/data/ieee39_network.toml` (BigFloat at the caller's
precision): A (pre-fault, one axis), the fault row and its change, Q, the injection rows and phasors, z0, names.
"""
function network_rt_data(; path = joinpath(NETWORK_RT_DIR, "data", "ieee39_network.toml"))
    t = TOML.parsefile(path)
    N = t["n"]
    A = zeros(BigFloat, N, N); for (i, j, v) in t["A"]; A[i, j] = parse(BigFloat, v); end
    Q = zeros(BigFloat, N, N); for (i, j, v) in t["Q"]; Q[i, j] = parse(BigFloat, v); end
    f = t["fault"]
    inj = complex.(parse.(BigFloat, t["inj_re"]), parse.(BigFloat, t["inj_im"]))
    z0 = complex.(parse.(BigFloat, t["z0_re"]), parse.(BigFloat, t["z0_im"]))
    return (; N, A, Q, frow = f["row"], dA = parse(BigFloat, f["dA"]), rows = Vector{Int}(t["inj_rows"]), inj, z0,
            names = Vector{String}(t["names"]), nodes = Vector{Int}(t["nodes"]), wb = BigFloat(t["omega_b"]))
end

"""
    network_rt_prepare(d, h) -> NamedTuple

The F3 matrices for both banks (pre-fault, fault) in BigFloat, the prepared injection phasors
hI' = (B1 or B2) I (1 + e^{j wb h}) / 2 per row, and the bench's rotation e^{j wb h}.
"""
function network_rt_prepare(d, h::BigFloat)
    q = d.nodes; p = setdiff(1:d.N, q)
    NI = length(d.rows)
    B = zeros(BigFloat, d.N, NI); for (c, r) in enumerate(d.rows); B[r, c] = 1; end
    half = (1 + cis(d.wb * h)) / 2
    banks = map((false, true)) do fault
        A = copy(d.A); fault && (A[d.frow, d.frow] += d.dA)
        M = I - (h / 2) * A
        Dinv = inv(M[p, p]); E = -M[q, p] * Dinv; S = M[q, q] + E * M[p, q]
        F = -Dinv * M[p, q]; hA = h * A; hB = h * B
        G1 = hA[q, :] + E * hA[p, :]; G2 = Dinv * hA[p, :]
        B1 = hB[q, :] + E * hB[p, :]; B2 = Dinv * hB[p, :]
        P = h * (M \ A); W = h * (M \ B)
        (; A, G1, G2, B1, B2, Sinv = inv(S), F, P, W)
    end
    pre, flt = banks
    maximum(abs, pre.G2 - flt.G2) == 0 && maximum(abs, pre.F - flt.F) == 0 ||
        error("the fault changes G2 or F: the bank layout of NETWORK.md 4 does not hold")
    maximum(abs, pre.B2) == 0 || error("the injections reach the inductor rows directly (B2 != 0)")
    I1 = pre.B1 * (d.inj .* half)                    # per node row: the prepared injection phasor (complex)
    return (; q, p, pre, flt, I1, half, a1 = cos(d.wb * h) - 1, b = sin(d.wb * h))
end

# ---------------------------------------------------------------------------------------------------------------------
# 2. The exact reference (Float64 from the BigFloat matrices): ranges, check (b), energy ledger
# ---------------------------------------------------------------------------------------------------------------------
# step n (n >= 1) covers t_{n-1} .. t_n; its input is the interval average of the endpoint injections,
# um = I e^{j wb t_{n-1}} (1 + e^{j wb h}) / 2; the fault bank applies to steps n_on + 1 .. n_off.
function _network_rt_exact(d, pr, h, nsteps; non, noff, ranges = false, keep = Int[])
    P = (Float64.(pr.pre.P), Float64.(pr.flt.P)); W = (Float64.(pr.pre.W), Float64.(pr.flt.W))
    G1 = (Float64.(pr.pre.G1), Float64.(pr.flt.G1)); G2 = Float64.(pr.pre.G2)
    B1 = (Float64.(pr.pre.B1), Float64.(pr.flt.B1)); Sv = (Float64.(pr.pre.Sinv), Float64.(pr.flt.Sinv))
    inj = ComplexF64.(d.inj .* pr.half); wbh = Float64(d.wb) * h
    z = ComplexF64.(d.z0)
    mx = zeros(4)                                    # max |x|, |rq|, |rp|, |dq| (real and imaginary parts)
    m(v) = max(maximum(abs ∘ real, v), maximum(abs ∘ imag, v))
    traj = Dict{Int,Vector{ComplexF64}}(0 => copy(z))   # the states after the steps in `keep`
    for n in 1:nsteps
        b = non < n <= noff ? 2 : 1
        um = inj .* cis(wbh * (n - 1))
        if ranges
            rq = G1[b] * z + B1[b] * um; rp = G2 * z; dq = Sv[b] * rq
            mx .= max.(mx, (m(z), m(rq), m(rp), m(dq)))
        end
        z = z + P[b] * z + W[b] * um
        n in keep && (traj[n] = z)
    end
    return traj, mx
end

# ---------------------------------------------------------------------------------------------------------------------
# 3. Formats and scales (powers of two; NETWORK.md 5)
# ---------------------------------------------------------------------------------------------------------------------
_ifmt(m) = Int(floor(log2(Float64(m)))) + 2          # ap_fixed<48, I>: range +-2^(I-1) > m (m < 2^(I-1) <= 2m)
_q48(v, I) = (g = BigFloat(2)^(-(48 - I)); Float64(round(BigFloat(v) / g, RoundNearest) * g))

function network_rt_formats(mx)
    xmax = mx[1]
    Ix = Int(floor(log2(xmax))) + 3                  # states: range 2^(Ix-1) in [2, 4) x max |x|
    sc(v) = Int(floor(log2(xmax / v)))               # the scaled vector's max <= max |x|
    m_rq, m_rp, m_dq = sc(mx[2]), sc(mx[3]), sc(mx[4])
    sig = Int(floor(log2(xmax)))                     # phasor (cos, sin) stored as 2^sig (c, s): <= max |x|
    return (; Ix, m_rq, m_rp, m_dq, sig)
end

# ---------------------------------------------------------------------------------------------------------------------
# 4. The lane schedule (row ownership, slots, ROM tables) and its certificate (review 35b, 6)
# ---------------------------------------------------------------------------------------------------------------------
# P1 rows: the 28 rq rows (entries G1 and the injection's two phasor entries), the 51 rp rows (G2), the 2 phasor
# rows; P2 rows: the 28 dq rows (S^-1); P3 rows: the 51 dp rows (F, initialised with rp). Source codes: x bank b,
# slot s (b = 9: the phasor bank: slot 1 = 2^sig (c, s) alpha/beta, slot 2 = 2^sig (-s, c)).
function _lpt!(load, w)                              # the lane with the smallest load takes the row
    l = argmin(load); load[l] += w; return l
end

function network_rt_schedule(d, pr, f; grid = true, groups = 2, seed = 0)
    L = NETWORK_RT_LANES
    q, p = pr.q, pr.p
    nq, np_ = length(q), length(p)
    two(v) = BigFloat(2)^v
    # one coefficient format C = ap_fixed<48, 1> for every phase (the shared multiplier, review 36 follow-up); every
    # row kind k gets its own scale 2^c_k so that its largest coefficient lies in [0.5, 1): full precision for each
    # kind; the writes compensate with exact power-of-two shifts (scale2, wiring). Kinds: q (rq rows: G1 and the
    # injection entries), p (rp rows: G2), K (phasor rows), s (dq rows: S^-1), f (P3 rows: F).
    cs(m) = -Int(floor(log2(Float64(m)))) - 1
    Ib1 = pr.I1 / two(f.sig)                         # the injection coefficient per unit of the stored phasor
    cq = cs(maximum(abs, vcat(vec(pr.pre.G1), vec(pr.flt.G1), abs.(real.(Ib1)), abs.(imag.(Ib1)))))
    cpk = cs(maximum(abs, pr.pre.G2))
    cK = cs(max(abs(pr.a1), abs(pr.b)))
    css = cs(maximum(abs, vcat(vec(pr.pre.Sinv), vec(pr.flt.Sinv))))
    cf = cs(maximum(abs, pr.pre.F))
    Ic = 1
    G1s = [b.G1 * two(cq) for b in (pr.pre, pr.flt)]; Is = Ib1 * two(cq)
    G2s = pr.pre.G2 * two(cpk)
    Ss = [b.Sinv * two(css) for b in (pr.pre, pr.flt)]
    Fs = pr.pre.F * two(cf)
    rot = (pr.a1 * two(cK), pr.b * two(cK))
    # the writes: stored = acc x 2^shift (acc = true value x 2^(c_k + operand scale))
    sh = (; q = f.m_rq - cq, p = f.m_rp - cpk, K = -cK, dq = f.m_dq - css - f.m_rq, x2 = -(css + f.m_rq),
          init = cf + f.m_dq - f.m_rp, x3 = -(cf + f.m_dq))
    # the pattern is always the rounded one (entries below the grid are dropped); grid = false keeps the values
    # unrounded (Float64 of the BigFloat value), for check (b): the algebra and the schedule without coefficient rounding
    r1(v) = (w = _q48(v, Ic); grid || w == 0 ? w : Float64(v))
    r2 = r1; r3 = r1
    # rows: (entries, dest); an entry (coef pre, coef fault, source bank or :x / :rq / :dq column)
    P1 = Vector{Any}(); P2 = Vector{Any}(); P3 = Vector{Any}()
    for (i, r) in enumerate(q)
        e = [(r1(G1s[1][i, j]), r1(G1s[2][i, j]), (:x, j)) for j in 1:d.N if r1(G1s[1][i, j]) != 0 || r1(G1s[2][i, j]) != 0]
        if Is[i] != 0
            push!(e, (r1(real(Is[i])), r1(real(Is[i])), (:ph, 1)), (r1(imag(Is[i])), r1(imag(Is[i])), (:ph, 2)))
        end
        push!(P1, (e, (:rq, i)))
        push!(P2, ([(r2(Ss[1][i, j]), r2(Ss[2][i, j]), (:rq, j)) for j in 1:nq if r2(Ss[1][i, j]) != 0 || r2(Ss[2][i, j]) != 0], (:xq, i)))
    end
    for (k, r) in enumerate(p)
        push!(P1, ([(r1(G2s[k, j]), r1(G2s[k, j]), (:x, j)) for j in 1:d.N if r1(G2s[k, j]) != 0], (:rp, k)))
        e = [(r3(Fs[k, j]), r3(Fs[k, j]), (:dq, j)) for j in 1:nq if r3(Fs[k, j]) != 0]
        isempty(e) && push!(e, (0.0, 0.0, (:dq, 1)))   # a row without F entries still writes x_p += rp
        push!(P3, (e, (:xp, k)))
    end
    a, bb = r1(rot[1]), r1(rot[2])
    push!(P1, ([(a, a, (:ph, 1)), (bb, bb, (:ph, 2))], (:ph, 1)))      # slot 1: 2^sig (c, s) <- rotated
    push!(P1, ([(-bb, -bb, (:ph, 1)), (a, a, (:ph, 2))], (:ph, 2)))    # slot 2: 2^sig (-s, c)
    any(r -> isempty(r[1]), vcat(P1, P2)) && error("a P1 or P2 row has no entries")
    # ownership (review 36b, one combined loop): a dq row's lane owns its x_q slot and its dq bank slot; a p row's lane
    # owns rp (P1), the P3 row and its x_p slot; rq rows and the phasor rows go to any lane. Balanced on the total load.
    lt = zeros(Int, L)
    own_p = zeros(Int, np_)
    for k in sortperm([length(P1[nq + k][1]) + length(P3[k][1]) for k in 1:np_], rev = true)
        own_p[k] = _lpt!(lt, length(P1[nq + k][1]) + length(P3[k][1]))
    end
    own_dq = zeros(Int, nq)
    for i in sortperm([length(P2[i][1]) for i in 1:nq], rev = true); own_dq[i] = _lpt!(lt, length(P2[i][1])); end
    own_rq = zeros(Int, nq)
    for i in sortperm([length(P1[i][1]) for i in 1:nq], rev = true); own_rq[i] = _lpt!(lt, length(P1[i][1])); end
    own_ph = [_lpt!(lt, 2), 0]; own_ph[2] = _lpt!(lt, 2)
    # slots
    xs = zeros(Int, d.N); xl = zeros(Int, d.N); cntx = zeros(Int, L)    # x bank (= writer lane) and slot
    for (i, r) in enumerate(q); l = own_dq[i]; cntx[l] += 1; xl[r] = l; xs[r] = cntx[l]; end
    for (k, r) in enumerate(p); l = own_p[k]; cntx[l] += 1; xl[r] = l; xs[r] = cntx[l]; end
    qs = zeros(Int, nq); c = zeros(Int, L); for i in 1:nq; c[own_rq[i]] += 1; qs[i] = c[own_rq[i]]; end; SQ1 = maximum(c)
    ds = zeros(Int, nq); c = zeros(Int, L); for i in 1:nq; c[own_dq[i]] += 1; ds[i] = c[own_dq[i]]; end; SQ2 = maximum(c)
    ps = zeros(Int, np_); c = zeros(Int, L); for k in 1:np_; c[own_p[k]] += 1; ps[k] = c[own_p[k]]; end; SP = maximum(c)
    SX = max(maximum(cntx), 2)
    # every row as (kind, lane, entries, producer key, x slot written or 0); an entry (cp, cf, source key)
    rows = Any[]
    for i in 1:nq; push!(rows, (:rq, own_rq[i], P1[i][1], (:rq, i))); end
    for k in 1:np_; push!(rows, (:rp, own_p[k], P1[nq + k][1], (:rp, k))); end
    push!(rows, (:ph, own_ph[1], P1[nq + np_ + 1][1], (:ph, 1))); push!(rows, (:ph, own_ph[2], P1[nq + np_ + 2][1], (:ph, 2)))
    for i in 1:nq; push!(rows, (:dq, own_dq[i], P2[i][1], (:dq, i))); end
    for k in 1:np_; push!(rows, (:p3, own_p[k], P3[k][1], (:p3, k))); end
    xrow(r) = r[1] === :dq ? q[r[4][2]] : r[1] === :p3 ? p[r[4][2]] : 0      # the x row a row writes (0: none)
    readers = Dict{Int,Vector{Int}}()                                          # x row -> rows reading it (P1)
    for (ri, r) in enumerate(rows), e in r[3]
        e[3][1] === :x && push!(get!(readers, e[3][2], Int[]), ri)
    end
    # list scheduling, cycle by cycle (one accumulator per lane: rows contiguous on their lane). An operand produced
    # in this step is ready D iterations after its write; a row that writes x row j ends at least one iteration after
    # the last P1 read of x row j (reads happen in earlier stages; one iteration margin, review 36b, 3); a P3 row's
    # init (rp) is read at its first entry.
    D = NETWORK_RT_RAW_D
    wt = Dict{Any,Int}()                                                      # producer -> write iteration
    rd = Dict{Int,Int}()                                                      # x row -> last read iteration so far
    done = falses(length(rows)); started = falses(length(rows))
    prio = Dict(:rq => 1, :dq => 2, :rp => 3, :ph => 4, :p3 => 5)
    ready(e) = e[3][1] in (:x, :ph) ? 0 : haskey(wt, e[3]) ? wt[e[3]] + D : typemax(Int) ÷ 2
    function plan(ri, t)                       # the entries in ready order and the earliest start, or nothing
        r = rows[ri]
        es = sort(r[3]; by = ready)            # stable: same order for every coefficient set (grid or not)
        s0 = maximum(k -> ready(es[k]) - (k - 1), eachindex(es))
        r[1] === :p3 && (s0 = max(s0, haskey(wt, (:rp, r[4][2])) ? wt[(:rp, r[4][2])] + D : typemax(Int) ÷ 2))
        j = xrow(r)
        if j > 0
            all(x -> done[x], get(readers, j, Int[])) || return nothing
            s0 = max(s0, get(rd, j, -1) + 1 - (length(es) - 1))
        end
        return s0 <= t ? (es, s0) : nothing
    end
    # one read address per bank per cycle (review 37): the 25 banks (1-8 x by writer lane, 9 the phasor, 10-17 rq,
    # 18-25 dq) each read one slot per cycle, lanes needing the same value share it. Inside a row the entry order is
    # chosen per cycle (a ready entry whose bank is free or already reads the same slot); otherwise the lane stalls
    # (a no-op inside the row: coefficient 0, the accumulator keeps its value).
    function bankof(sk)
        sk[1] === :x && return (xl[sk[2]], xs[sk[2]])
        sk[1] === :ph && return (9, sk[2])
        sk[1] === :rq && return (9 + own_rq[sk[2]], qs[sk[2]])
        return (17 + own_dq[sk[2]], ds[sk[2]])
    end
    NB = 25
    # lane groups (review 37, 5): group g = lanes (g-1)L/G+1 .. gL/G reads its own port of every bank (one bank copy
    # per group), so a lane selects among 25 bank outputs; one address per bank and group per cycle
    grp(l) = (l - 1) ÷ (L ÷ groups) + 1
    # seed > 0: a bounded search (review 37, 5): random tie-breaks in the lane order and among the issuable entries,
    # deterministic per seed; the export keeps the seed with the smallest T
    rng = Random.Xoshiro(seed)
    function pick(rem, t, claim, l)
        ok = findall(e -> ready(e) <= t && (b = bankof(e[3]); c = get(claim, (grp(l), b[1]), 0); c == 0 || c == b[2]), rem)
        isempty(ok) && return nothing
        return seed == 0 || rand(rng) < 0.5 ? ok[1] : ok[rand(rng, eachindex(ok))]
    end
    load = zeros(Int, L); for r in rows; load[r[2]] += length(r[3]); end   # entries left per lane
    prog = [Any[] for _ in 1:L]
    claims = Dict{Tuple{Int,Int},Int}[]                                      # per cycle: (group, bank) -> slot
    cur = Vector{Any}(nothing, L)
    t = 0
    while !all(done)
        t > 2000 && error("the list scheduler did not finish")
        claim = Dict{Tuple{Int,Int},Int}(); push!(claims, claim)
        for l in sortperm(load .+ (seed == 0 ? zeros(L) : 3 .* rand(rng, L)), rev = true)   # most loaded lane first
            if cur[l] === nothing
                best = nothing
                for ri in eachindex(rows)
                    (started[ri] || rows[ri][2] != l) && continue
                    pl = plan(ri, t); pl === nothing && continue
                    pick(pl[1], t, claim, l) === nothing && continue
                    key = (prio[rows[ri][1]], pl[2], ri)
                    (best === nothing || key < best[1]) && (best = (key, ri, pl[1]))
                end
                best === nothing || (cur[l] = (best[2], copy(best[3]), 0); started[best[2]] = true)
            end
            k = cur[l] === nothing ? nothing : pick(cur[l][2], t, claim, l)
            if k === nothing
                push!(prog[l], nothing)                                          # no-op, or a stall inside a row
            else
                ri, rem, n = cur[l]
                e = rem[k]; deleteat!(rem, k); load[l] -= 1
                b = bankof(e[3]); claim[(grp(l), b[1])] = b[2]
                push!(prog[l], (ri, e, n == 0, isempty(rem)))
                e[3][1] === :x && (rd[e[3][2]] = max(get(rd, e[3][2], -1), t))
                if isempty(rem)
                    wt[rows[ri][4]] = t; done[ri] = true; cur[l] = nothing
                else
                    cur[l] = (ri, rem, n + 1)
                end
            end
        end
        t += 1
    end
    T = t
    # ROM entries: (cp, cf, source bank 1-25, first, last, init from rp, write kind 0 none 1 rq 2 rp 3 phasor
    # 4 dq + x_q 5 x_p, dest slots (rq, rp, dq, phasor), x slot, init slot). The read addresses are per bank and
    # group (tables BA1, BA2: [bank, cycle]); a bank no lane of the group reads, and the rp / x-copy addresses of
    # entries that do not use them, point at a parking slot (one past the last, never written, zero)
    PX, PQ, PD, PP = SX + 1, SQ1 + 1, SQ2 + 1, SP + 1
    park(b) = b <= 9 ? PX : b <= 17 ? PQ : PD
    G = groups
    BA = [[get(claims[t], (g, b), park(b)) for b in 1:NB, t in 1:T] for g in 1:G]
    function rom_entry(x)
        x === nothing && return (0.0, 0.0, 1, false, false, false, 0, 1, 1, 1, 1, PX, PP)
        ri, e, first, last = x
        r = rows[ri]; b = bankof(e[3])[1]; id = r[4][2]
        wk, dsl, xsl, isl = r[1] === :rq ? (1, qs[id], PX, PP) : r[1] === :rp ? (2, ps[id], PX, PP) :
                            r[1] === :ph ? (3, id, PX, PP) : r[1] === :dq ? (4, ds[id], xs[q[id]], PP) : (5, 1, xs[p[id]], ps[id])
        last || (xsl = PX)
        (first && r[1] === :p3) || (isl = PP)
        # one destination address per array (rq, rp, dq, phasor): each address register drives one array only
        dst = (wk == 1 ? dsl : 1, wk == 2 ? dsl : 1, wk == 4 ? dsl : 1, wk == 3 ? dsl : 1)   # (DQ, DP, DD, DN)
        return (e[1], e[2], b, first, last, r[1] === :p3, last ? wk : 0, dst..., xsl, isl)
    end
    progr = [[rom_entry(prog[l][t]) for t in 1:T] for l in 1:L]
    # certificate, recomputed from the programs (review 36b, 3, 6): every entry once; rows contiguous per lane; RAW
    # distance >= D; x written after its last read (one iteration); every slot written once
    nent = sum(r -> length(r[3]), rows)
    count(x -> x !== nothing, vcat(prog...)) == nent || error("certificate: the programs hold a different number of entries")
    wr = Dict{Any,Int}(); rdx = Dict{Int,Int}(); wx = Dict{Int,Int}()
    for l in 1:L, t in 1:T                                                     # writes and reads by iteration
        x = prog[l][t]; x === nothing && continue
        ri, e, first, last = x; r = rows[ri]
        e[3][1] === :x && (rdx[e[3][2]] = max(get(rdx, e[3][2], -1), t))
        if last
            haskey(wr, r[4]) && error("certificate: $(r[4]) written twice")
            wr[r[4]] = t
            j = xrow(r); j > 0 && (haskey(wx, j) && error("certificate: x row $j written twice"); wx[j] = t)
        end
    end
    for l in 1:L
        open = 0
        for t in 1:T
            x = prog[l][t]; x === nothing && continue
            ri, e, first, last = x
            first && (open == 0 || error("certificate: lane $l starts a row inside another"); open = ri)
            open == ri || error("certificate: lane $l mixes rows")
            if e[3][1] in (:rq, :dq)
                haskey(wr, e[3]) && t >= wr[e[3]] + D || error("certificate: lane $l reads $(e[3]) at $t, written at $(get(wr, e[3], -1)) (D = $D)")
            end
            if first && rows[ri][1] === :p3
                t >= wr[(:rp, rows[ri][4][2])] + D || error("certificate: lane $l reads rp too early")
            end
            last && (open = 0)
        end
        open == 0 || error("certificate: lane $l leaves a row open")
    end
    for (j, tw) in wx
        tw >= get(rdx, j, -1) + 1 || error("certificate: x row $j written at $tw before its last read at $(rdx[j])")
    end
    length(wr) == length(rows) || error("certificate: not every row was written")
    # review 37, 4: each row's entries exactly once (multiset), and the ROMs give every entry its operand: the lane's
    # bank (RS) read at the group's address for that cycle (BA) is the entry's source; one address per bank and group
    got = [String[] for _ in rows]
    for l in 1:L, t in 1:T
        x = prog[l][t]; x === nothing && continue
        ri, e = x; push!(got[ri], repr(e))
        b = progr[l][t][3]
        (b, BA[grp(l)][b, t]) == bankof(e[3]) || error("certificate: lane $l cycle $t reads bank $b slot $(BA[grp(l)][b, t]), not $(bankof(e[3]))")
    end
    for (ri, r) in enumerate(rows)
        sort(got[ri]) == sort(repr.(r[3])) || error("certificate: row $ri does not hold its entries exactly once")
    end
    for t in 1:T, g in 1:G, b in 1:NB                                          # claimed slots are real, unclaimed park
        sl = BA[g][b, t]
        any(l -> grp(l) == g && prog[l][t] !== nothing && bankof(prog[l][t][2][3]) == (b, sl), 1:L) || sl == park(b) ||
            error("certificate: bank $b group $g cycle $t reads slot $sl for no lane")
    end
    allunique([(own_rq[i], qs[i]) for i in 1:nq]) && allunique([(own_dq[i], ds[i]) for i in 1:nq]) &&
        allunique([(xl[r], xs[r]) for r in 1:d.N]) && allunique([(own_p[k], ps[k]) for k in 1:np_]) ||
        error("two rows share a bank slot")
    return (; progr, BA, G, seed, T, D, lb = cld(nent, L), Ic, sh, cq, own_ph, SX, SQ1, SQ2, SP, xl, xs, own_rq, own_dq, own_p,
            nentries = (sum(r -> length(r[1]), P1), sum(r -> length(r[1]), P2), sum(r -> length(r[1]), P3)), nent,
            idle = count(x -> x === nothing, vcat(prog...)))
end

# registers the ROM tables of a schedule (hls_rom!) under the names the generated code reads
const NETWORK_RT_ROMS = ("CP", "CF", "RS", "F", "E", "IK", "WK", "DQ", "DP", "DD", "DN", "W", "I")
function _network_rt_roms!(s)
    L = NETWORK_RT_LANES
    for (i, nm) in enumerate(NETWORK_RT_ROMS)
        hls_rom!(Symbol("NET_$nm"), [s.progr[l][t][i] for l in 1:L, t in 1:s.T])
    end
    for g in 1:s.G
        hls_rom!(Symbol("NET_BA$g"), s.BA[g])
    end
end

# ---------------------------------------------------------------------------------------------------------------------
# 5. The generated board top (network_gen.jl, @hls)
# ---------------------------------------------------------------------------------------------------------------------
function network_rt_generate(d, f, s, outrows; path = joinpath(NETWORK_RT_DIR, "network_gen.jl"))
    N = d.N; L = NETWORK_RT_LANES; sh = s.sh
    # the states' bank and slot (copy in and out by small loops: constant-index statements made a large address
    # multiplexer on the AXI-Lite state memory, the worst routed path of build 3)
    hls_rom!(:NET_CB, [s.xl[r] for r in 1:N]); hls_rom!(:NET_CS, [s.xs[r] for r in 1:N])
    cin = ["    z0::X = 0                                        # the parking slots hold zero (read, never written)",
           "    for l in 1:9",
           "        xa[l, $(s.SX + 1)] = z0",
           "        xb[l, $(s.SX + 1)] = z0",
           "    end",
           "    for l in 1:$L",
           "        xo[l, $(s.SX + 1)] = z0",
           "        yo[l, $(s.SX + 1)] = z0",
           "        qa[l, $(s.SQ1 + 1)] = z0",
           "        qb[l, $(s.SQ1 + 1)] = z0",
           "        da[l, $(s.SQ2 + 1)] = z0",
           "        db[l, $(s.SQ2 + 1)] = z0",
           "        pa[l, $(s.SP + 1)] = z0",
           "        pb[l, $(s.SP + 1)] = z0",
           "    end",
           "    for r in 1:$N                                    # copy in: the states to their banks (and owner copies)",
           "        xa[cbk[r], csl[r]] = x[r]",
           "        xb[cbk[r], csl[r]] = x[$N + r]",
           "        xo[cbk[r], csl[r]] = x[r]",
           "        yo[cbk[r], csl[r]] = x[$N + r]",
           "    end"]
    cout = ["    for r in 1:$N                                    # copy out",
            "        x[r] = xa[cbk[r], csl[r]]",
            "        x[$N + r] = xb[cbk[r], csl[r]]",
            "    end"]
    o1, o2 = outrows
    roms = ("rcp", "rcf", "rs", "rf", "re", "rik", "rwk", "rdq", "rdp", "rdd", "rdn", "rw", "ri", "rb1", "rb2")
    s.G == 2 || error("the generator is written for two lane groups")
    part(v, dim = "dim=1 ") = "    @pragma \"HLS ARRAY_PARTITION variable=$v $(dim)complete\""
    lut(v) = "    @pragma \"HLS BIND_STORAGE variable=$v type=ram_1wnr impl=lutram\""
    # the dependence on every bank array is false: the export's certificate shows that every read of a value written
    # in this step comes >= D iterations after its write and every x slot is written after its last read (review 36b)
    deps = join(("            @pragma \"HLS DEPENDENCE variable=$v type=inter dependent=false\"" for v in
                 ("xa", "xb", "xo", "yo", "qa", "qb", "da", "db", "pa", "pb")), "\n")
    src = """
    # network_gen.jl: written by PHPS.RT.network_rt_generate (src/rt/network/network.jl); do not edit.
    # The board top of the network block for h = $(NETWORK_RT_H) s (NETWORK.md 4-5): three phase loops per step on
    # 8 lanes x 2 axes; one shared multiply function (16 multipliers) serves all three; the lanes' programs are the
    # ROM tables NET_* (network_rom.h); formats in the export (types); scale2(v, k) = v 2^k exactly (wiring in C++).

    # the 16 products of one cycle: 8 lanes x (alpha, beta), one coefficient per lane (shared by all phase loops)
    @hls function network_mac!(c::C[$L], oa::X[$L], ob::X[$L], pa::P[$L], pb::P[$L])
        @pragma "HLS INLINE off"
        @pragma "HLS PIPELINE II=1"
        for l in 1:$L
            @pragma "HLS UNROLL"
            pa[l] = c[l] * oa[l]
            pb[l] = c[l] * ob[l]
        end
    end

    @hls function network_run!(x::X[$(2N)], ph::X[2], n0::Int, n_on::Int, n_off::Int, nsteps::Int, decim::Int,
                               v1_out::Out{Sig}, v2_out::Out{Sig}, strobe::Out{Bool}, sample::Out{Bool})
        @pragma "HLS INTERFACE mode=s_axilite port=return"
        @pragma "HLS INTERFACE mode=s_axilite port=x storage_impl=bram"
        @pragma "HLS INTERFACE mode=s_axilite port=ph"
        @pragma "HLS INTERFACE mode=s_axilite port=n0"
        @pragma "HLS INTERFACE mode=s_axilite port=n_on"
        @pragma "HLS INTERFACE mode=s_axilite port=n_off"
        @pragma "HLS INTERFACE mode=s_axilite port=nsteps"
        @pragma "HLS INTERFACE mode=s_axilite port=decim"
        @pragma "HLS INTERFACE mode=ap_none port=v1_out"
        @pragma "HLS INTERFACE mode=ap_none port=v2_out"
        @pragma "HLS INTERFACE mode=ap_none port=strobe"
        @pragma "HLS INTERFACE mode=ap_none port=sample"
        xa::X[9, $(s.SX + 1)]                                # x operand banks, alpha (bank = writing lane; 9: the phasor)
    $(part("xa"))
    $(lut("xa"))
        xb::X[9, $(s.SX + 1)]                                # beta
    $(part("xb"))
    $(lut("xb"))
        xo::X[$L, $(s.SX + 1)]                               # the owner's copy of its x slots (read-modify-write), so
    $(part("xo"))                                        # the x banks have one write and the lanes' reads only
    $(lut("xo"))
        yo::X[$L, $(s.SX + 1)]
    $(part("yo"))
    $(lut("yo"))
        qa::X[$L, $(s.SQ1 + 1)]                              # rq banks (written in P1, read in P2)
    $(part("qa"))
    $(lut("qa"))
        qb::X[$L, $(s.SQ1 + 1)]
    $(part("qb"))
    $(lut("qb"))
        da::X[$L, $(s.SQ2 + 1)]                              # dq banks (written in P2, read in P3)
    $(part("da"))
    $(lut("da"))
        db::X[$L, $(s.SQ2 + 1)]
    $(part("db"))
    $(lut("db"))
        pa::X[$L, $(s.SP + 1)]                               # rp, lane-local (written in P1, the init of P3)
    $(part("pa"))
    $(lut("pa"))
        pb::X[$L, $(s.SP + 1)]
    $(part("pb"))
    $(lut("pb"))
        pn::X[$L, 2]                                     # the phasor rows' increments, lane-local
    $(part("pn", "dim=0 "))
        pm::X[$L, 2]
    $(part("pm", "dim=0 "))
        cc::C[$L]                                        # one cycle's coefficients, operands and products
    $(part("cc", ""))
        va::X[$L]
    $(part("va", ""))
        vb::X[$L]
    $(part("vb", ""))
        pa_::P[$L]
    $(part("pa_", ""))
        pb_::P[$L]
    $(part("pb_", ""))
        aa::Acc[$L]                                      # accumulators (>= 16 guard bits below X's grid)
    $(part("aa", ""))
        ab::Acc[$L]
    $(part("ab", ""))
        zero::Acc = 0
        oa1::X[25]                                       # the 25 bank outputs (1-8 x, 9 phasor, 10-17 rq, 18-25 dq)
    $(part("oa1", ""))                                   # of each lane group (lanes 1-4: port 1, 5-8: port 2)
        ob1::X[25]
    $(part("ob1", ""))
        oa2::X[25]
    $(part("oa2", ""))
        ob2::X[25]
    $(part("ob2", ""))
        rcp::C[$L, $(s.T)] = rom(:NET_CP)               # the lanes' program, one entry per lane and cycle: the
        rcf::C[$L, $(s.T)] = rom(:NET_CF)               # coefficient (pre-fault, fault), the source bank (1-25),
        rs::Ib[$L, $(s.T)] = rom(:NET_RS)               # first / last entry of a row, the init from rp, the write
        rb1::Ib[25, $(s.T)] = rom(:NET_BA1)             # kind (1 rq, 2 rp, 3 phasor, 4 dq and x_q, 5 x_p) and its
        rb2::Ib[25, $(s.T)] = rom(:NET_BA2)             # slots; per bank and group one read address (rb1, rb2;
        rf::Bool[$L, $(s.T)] = rom(:NET_F)               # unused: the parking slot)
        re::Bool[$L, $(s.T)] = rom(:NET_E)
        rik::Bool[$L, $(s.T)] = rom(:NET_IK)
        rwk::Ib[$L, $(s.T)] = rom(:NET_WK)
        rdq::Ib[$L, $(s.T)] = rom(:NET_DQ)
        rdp::Ib[$L, $(s.T)] = rom(:NET_DP)
        rdd::Ib[$L, $(s.T)] = rom(:NET_DD)
        rdn::Ib[$L, $(s.T)] = rom(:NET_DN)
        rw::Ib[$L, $(s.T)] = rom(:NET_W)
        ri::Ib[$L, $(s.T)] = rom(:NET_I)
        cbk::Ib[$N] = rom(:NET_CB)                       # bank and slot of each state (copy in and out)
        csl::Ib[$N] = rom(:NET_CS)
    $(join((part(v) for v in roms), "\n"))
    $(join(cin, "\n"))
        xa[9, 1] = ph[1]                                 # the phasor bank: slot 1 (c, s), slot 2 (-s, c), x 2^sig
        xa[9, 2] = -ph[2]
        xb[9, 1] = ph[2]
        xb[9, 2] = ph[1]
        st::Bool = false
        cnt::Int = 0
        for i in 1:nsteps
            @pragma "HLS ALLOCATION function instances=network_mac limit=1"
            n::Int = n0 + i
            fault::Bool = false                          # the fault bank, latched for the whole step
            if n > n_on && n <= n_off
                fault = true
            end
            for t in 1:$(s.T)                            # the step: rq, rp, phasor, dq (x_q), x_p rows interleaved
                @pragma "HLS PIPELINE II=1"
    $(deps)
                for b in 1:9                             # one read per bank and group (review 37)
                    @pragma "HLS UNROLL"
                    oa1[b] = xa[b, rb1[b, t]]
                    ob1[b] = xb[b, rb1[b, t]]
                    oa2[b] = xa[b, rb2[b, t]]
                    ob2[b] = xb[b, rb2[b, t]]
                end
                for b in 1:$L
                    @pragma "HLS UNROLL"
                    oa1[9 + b] = qa[b, rb1[9 + b, t]]
                    ob1[9 + b] = qb[b, rb1[9 + b, t]]
                    oa2[9 + b] = qa[b, rb2[9 + b, t]]
                    ob2[9 + b] = qb[b, rb2[9 + b, t]]
                    oa1[17 + b] = da[b, rb1[17 + b, t]]
                    ob1[17 + b] = db[b, rb1[17 + b, t]]
                    oa2[17 + b] = da[b, rb2[17 + b, t]]
                    ob2[17 + b] = db[b, rb2[17 + b, t]]
                end
                for l in 1:$L
                    @pragma "HLS UNROLL"
                    cc[l] = rcp[l, t]
                    if fault
                        cc[l] = rcf[l, t]
                    end
                    if l <= $(L ÷ 2)
                        va[l] = oa1[rs[l, t]]
                        vb[l] = ob1[rs[l, t]]
                    else
                        va[l] = oa2[rs[l, t]]
                        vb[l] = ob2[rs[l, t]]
                    end
                end
                network_mac!(cc, va, vb, pa_, pb_)
                for l in 1:$L
                    @pragma "HLS UNROLL"
                    if rf[l, t]
                        if rik[l, t]
                            aa[l] = scale2(pa[l, ri[l, t]], $(sh.init)) + pa_[l]
                            ab[l] = scale2(pb[l, ri[l, t]], $(sh.init)) + pb_[l]
                        else
                            aa[l] = zero + pa_[l]
                            ab[l] = zero + pb_[l]
                        end
                    else
                        aa[l] = aa[l] + pa_[l]
                        ab[l] = ab[l] + pb_[l]
                    end
                    if rwk[l, t] == 1
                        qa[l, rdq[l, t]] = scale2(aa[l], $(sh.q))
                        qb[l, rdq[l, t]] = scale2(ab[l], $(sh.q))
                    elseif rwk[l, t] == 2
                        pa[l, rdp[l, t]] = scale2(aa[l], $(sh.p))
                        pb[l, rdp[l, t]] = scale2(ab[l], $(sh.p))
                    elseif rwk[l, t] == 3
                        pn[l, rdn[l, t]] = scale2(aa[l], $(sh.K))
                        pm[l, rdn[l, t]] = scale2(ab[l], $(sh.K))
                    elseif rwk[l, t] == 4
                        da[l, rdd[l, t]] = scale2(aa[l], $(sh.dq))
                        db[l, rdd[l, t]] = scale2(ab[l], $(sh.dq))
                        ya2::X = xo[l, rw[l, t]] + scale2(aa[l], $(sh.x2))
                        yb2::X = yo[l, rw[l, t]] + scale2(ab[l], $(sh.x2))
                        xo[l, rw[l, t]] = ya2
                        yo[l, rw[l, t]] = yb2
                        xa[l, rw[l, t]] = ya2
                        xb[l, rw[l, t]] = yb2
                    elseif rwk[l, t] == 5
                        ya3::X = xo[l, rw[l, t]] + scale2(aa[l], $(sh.x3))
                        yb3::X = yo[l, rw[l, t]] + scale2(ab[l], $(sh.x3))
                        xo[l, rw[l, t]] = ya3
                        yo[l, rw[l, t]] = yb3
                        xa[l, rw[l, t]] = ya3
                        xb[l, rw[l, t]] = yb3
                    end
                end
            end
            xa[9, 1] = xa[9, 1] + pn[$(s.own_ph[1]), 1]       # the phasor for the next step (P2, P3 do not read it)
            xa[9, 2] = xa[9, 2] + pn[$(s.own_ph[2]), 2]
            xb[9, 1] = xb[9, 1] + pm[$(s.own_ph[1]), 1]
            xb[9, 2] = xb[9, 2] + pm[$(s.own_ph[2]), 2]
            st = !st
            cnt += 1
            if cnt >= decim
                cnt = 0
                sample[] = true
            else
                sample[] = false
            end
            v1_out[] = xa[$(s.xl[o1]), $(s.xs[o1])]      # alpha charge of $(d.names[o1]) (v = Q x), to the ILA
            v2_out[] = xa[$(s.xl[o2]), $(s.xs[o2])]      # alpha charge of $(d.names[o2])
            strobe[] = st
        end
    $(join(cout, "\n"))
        ph[1] = xa[9, 1]
        ph[2] = xb[9, 1]
    end
    """
    write(path, src)
    println("written $path")
    return path
end

# ---------------------------------------------------------------------------------------------------------------------
# 6. Export: formats, schedule, generated code, checks, C++, testbench, board program
# ---------------------------------------------------------------------------------------------------------------------
_network_rt_cuts(h, nseg, non, noff) =
    sort(unique(vcat([round(Int, g * 0.01 / h) for g in 1:nseg], non-1:non+2, noff-1:noff+2)))

# the generated network_run!, called in the latest world (it is defined by including network_gen.jl at export)
_netrun(args...) = Base.invokelatest(Base.invokelatest(getglobal, @__MODULE__, :network_run!), args...)

# the Julia run of the generated network_run! (Float64), cut at `cuts`: states (both axes) and phasor after each cut
function _network_rt_julia(x0, ph0, cuts; non, noff)
    x = copy(x0); ph = copy(ph0); rec = [(copy(x), copy(ph))]
    r = (Ref(0.0), Ref(0.0), Ref(false), Ref(false)); n = 0
    for c in cuts
        _netrun(x, ph, n, non, noff, c - n, 32, r...)
        push!(rec, (copy(x), copy(ph))); n = c
    end
    return rec
end

"""
    network_rt_export(; h = NETWORK_RT_H, dir = src/rt/network/CPP/network)

Prepares the step for h (BigFloat), runs the exact reference (ranges), chooses the formats, builds and certifies the
lane schedule, generates network_gen.jl and includes it, runs the checks fixed in NETWORK.md 6 (the Julia @hls run
against the exact midpoint step, the energy ledger, continuous = segmented, ranges) and writes the C++, the ROM
header, the testbench, the board program and flow.cfg.
"""
function network_rt_export(; h = NETWORK_RT_H, dir = joinpath(NETWORK_RT_DIR, "CPP", "network"))
    setprecision(BigFloat, 256) do
        hb = parse(BigFloat, h); hh = Float64(hb)
        d = network_rt_data()
        pr = network_rt_prepare(d, hb)
        non = round(Int, 0.1 / hh); noff = round(Int, 0.2 / hh); nend = round(Int, 0.3 / hh)
        println("h = $hh s: fault steps $(non + 1) .. $noff, run of $nend steps")
        cuts = _network_rt_cuts(hh, 30, non, noff)
        traj, mx = _network_rt_exact(d, pr, hh, nend; non, noff, ranges = true, keep = Set(cuts))
        f = network_rt_formats(mx)
        println("ranges (max over the run, both axes): |x| $(mx[1]), |rq| $(mx[2]), |rp| $(mx[3]), |dq| $(mx[4]); ",
                "formats: X <48, $(f.Ix)>, scales 2^$(f.m_rq), 2^$(f.m_rp), 2^$(f.m_dq), phasor 2^$(f.sig)")
        # the bounded schedule search (review 37, 5): seeds 0..NETWORK_RT_SEEDS, the first with the smallest T
        s = network_rt_schedule(d, pr, f; groups = NETWORK_RT_GROUPS)
        for seed in 1:NETWORK_RT_SEEDS
            T = try network_rt_schedule(d, pr, f; groups = NETWORK_RT_GROUPS, seed).T catch; typemax(Int) end
            T < s.T && (s = network_rt_schedule(d, pr, f; groups = NETWORK_RT_GROUPS, seed))
        end
        println("schedule search: $(NETWORK_RT_GROUPS) lane groups, seeds 0-$(NETWORK_RT_SEEDS): T = $(s.T) at seed $(s.seed)")
        println("schedule: entries per phase $(s.nentries) per axis, one loop of T = $(s.T) iterations against the bound ",
                "$(s.lb) (idle lane slots $(s.idle)), D = $(s.D); slots x $(s.SX), rq $(s.SQ1), dq $(s.SQ2), rp $(s.SP); ",
                "C <48, $(s.Ic)> for all phases; write shifts $(s.sh)")
        _network_rt_roms!(s)
        o1 = findfirst(==("node bus 16"), d.names); o2 = findfirst(==("node bus 39"), d.names)
        outrows = (o1, o2)
        network_rt_generate(d, f, s, outrows)
        Base.include(@__MODULE__, joinpath(NETWORK_RT_DIR, "network_gen.jl"))
        # accumulator formats: exact for every product and sum (NETWORK.md 5)
        # products truncated to the accumulator's format (wiring; the add stays as wide as the accumulator), which
        # holds 64 products and keeps >= 16 guard bits below X's grid
        # (after a write's shift 2^sh the accumulator's grid must stay >= 16 bits below X's grid: width from the
        # largest left shift)
        Ip = s.Ic + f.Ix; Ia = Ip + 5
        Wa = Ia + (48 - f.Ix) + 16 + max(0, maximum(collect(s.sh)))
        types = [:Sig => "ap_fixed<48, 8, AP_RND_CONV>", :X => "ap_fixed<48, $(f.Ix), AP_RND_CONV>",
                 :C => "ap_fixed<48, $(s.Ic), AP_RND_CONV>", :P => "ap_fixed<$Wa, $Ia>", :Acc => "ap_fixed<$Wa, $Ia>",
                 :Ib => "ap_uint<6>"]
        println("accumulator ap_fixed<$Wa, $Ia>; write shifts $(s.sh)")
        # the initial state on the X grid and the phasor (2^sig, 0)
        x0 = vcat(Float64.(real.(d.z0)), Float64.(imag.(d.z0)))
        x0 = map(v -> _q48(v, f.Ix), x0); ph0 = [2.0^f.sig, 0.0]
        Qf = Float64.(d.Q); N = d.N
        eff(x) = vcat(Qf * x[1:N], Qf * x[N+1:2N])
        effd(rec) = maximum(g -> (ze = traj[cuts[g]]; maximum(abs, eff(rec[g + 1][1]) - eff(vcat(real.(ze), imag.(ze))))),
                            eachindex(cuts))
        # check (b): the generated code with UNROUNDED coefficients against the exact midpoint step (the algebra and
        # the schedule); limit 1e-9 pu on the efforts e = Q x (node voltages, branch currents), fixed before results
        _network_rt_roms!(network_rt_schedule(d, pr, f; grid = false, groups = s.G, s.seed))
        recu = _network_rt_julia(x0, ph0, cuts; non, noff)
        db = effd(recu)
        println("check (b), unrounded coefficients: max |e_hls - e_exact| over $(length(cuts)) checkpoints $db pu (limit 1e-9)")
        db <= 1e-9 || error("check (b) failed: the generated step differs from the exact midpoint step by $db")
        # the rounded coefficients (the build): their effect on the trajectory, limit 5e-8 pu (as the C simulation's)
        _network_rt_roms!(s)
        rec = _network_rt_julia(x0, ph0, cuts; non, noff)
        dr = effd(rec)
        phl = maximum(r -> abs((r[2][1]^2 + r[2][2]^2) / 4.0^f.sig - 1), rec)
        xm = maximum(r -> maximum(abs, r[1]), rec)
        println("rounded coefficients: max |e_hls - e_exact| $dr pu (limit 5e-8); phasor length max |c^2 + s^2 - 1| $phl ",
                "(limit 1e-10); max |x| at the checkpoints $xm (X range $(2.0^(f.Ix - 1)))")
        dr <= 5e-8 || error("the rounded coefficients move the trajectory by $dr")
        phl <= 1e-10 || error("the phasor length drifts by $phl")
        xm < 2.0^(f.Ix - 2) || error("a state reaches $xm: too close to the X range")
        # the energy ledger of the build, every step of the run: D = e_m' (x1 - x0) - h e_m' (A x_m + B u_m) per axis,
        # e_m = Q x_m; limit: sum |D| <= 1e-9 H0 (NETWORK.md 6)
        Af = Float64.(d.A); Afl = copy(Af); Afl[d.frow, d.frow] += Float64(d.dA)
        H(x) = (x[1:N]' * Qf * x[1:N] + x[N+1:2N]' * Qf * x[N+1:2N]) / 2
        xs_ = copy(x0); ps_ = copy(ph0); r4 = (Ref(0.0), Ref(0.0), Ref(false), Ref(false))
        H0 = H(x0); dsum = 0.0; dabs = 0.0; dmax = 0.0
        Isr = map(v -> complex(_q48(real(v), s.Ic), _q48(imag(v), s.Ic)), pr.I1 .* (2.0^s.cq / 2.0^f.sig))
        for n in 1:nend
            xo_ = copy(xs_); po_ = copy(ps_)
            _netrun(xs_, ps_, n - 1, non, noff, 1, 32, r4...)
            A_ = non < n <= noff ? Afl : Af
            psi = complex(po_[1], po_[2])                # the input the step applied: h B u = 2^-m_rq Is psi (rows q)
            hu = zeros(ComplexF64, N); hu[pr.q] = Isr .* psi ./ 2.0^s.cq
            Dn = 0.0
            for (ax, part) in ((1:N, real), (N+1:2N, imag))
                xm_ = (xo_[ax] + xs_[ax]) / 2; em = Qf * xm_
                Dn += em' * (xs_[ax] - xo_[ax]) - (hh * (em' * (A_ * xm_)) + em' * part.(hu))
            end
            dsum += Dn; dabs += abs(Dn); dmax = max(dmax, abs(Dn))
        end
        println("energy ledger over $nend steps: H0 $H0, sum D $dsum, sum |D| $dabs (limit $(1e-9 * H0)), max |D| $dmax")
        dabs <= 1e-9 * H0 || error("the energy ledger defect $dabs exceeds 1e-9 H0")
        xc = copy(x0); pc = copy(ph0)
        _netrun(xc, pc, 0, non, noff, cuts[end], 32, Ref(0.0), Ref(0.0), Ref(false), Ref(false))
        xc == rec[end][1] && pc == rec[end][2] || error("the continuous Julia run differs from the segmented one")
        mkpath(dir)
        hls_write_roms(joinpath(dir, "network_rom.h"),
                       vcat([Symbol("NET_$nm") for nm in NETWORK_RT_ROMS], [Symbol("NET_BA$g") for g in 1:s.G], [:NET_CB, :NET_CS]);
                       header = "the lanes' programs of the network block, h = $hh s (network.jl section 4)")
        hls_write_util(dir)
        hls_write(joinpath(dir, "network.cpp"), [:network_mac!, :network_run!]; types, local_includes = ["rt_util.h", "network_rom.h"],
                  header = "the IEEE-39 network block, one run of steps of h = $hh s; from src/rt/network/network_gen.jl")
        network_rt_testbench(joinpath(dir, "network_tb.cpp"), x0, ph0, rec, cuts; h = hh, non, noff, N = d.N, Q = Qf)
        network_rt_board_script(joinpath(dir, "network_board.py"), x0, ph0, f; h = hh, non, noff, N = d.N, Q = Qf)
        network_rt_cosim_tb(joinpath(dir, "network_cosim_tb.cpp"), x0, ph0; non, noff, N = d.N)
        flow_config(joinpath(dir, "flow.cfg"); model = "network", top = "network_run", clock_ns = NETWORK_RT_CLOCK_NS,
                    ip_version = "1.0", ip_name = "IEEE-39 network real-time step",
                    probes = ["v1_out" => 48, "v2_out" => 48, "strobe" => 1, "sample" => 1], sample_probe = "sample",
                    plot = ["v1_out" => (40, "q_alpha bus 16 [pu s]"), "v2_out" => (40, "q_alpha bus 39 [pu s]")],
                    sample_dt = 32 * hh, board_args = "--no-download",
                    # HLS must not partition the x banks into registers on its own (it did, for throughput: 99:1
                    # operand multiplexers; NETWORK.md 6)
                    hls_extra = ["syn.array_partition.throughput_driven=off"],
                    # high-fanout bank write addresses: post-route physical optimisation replicates their drivers
                    impl_strategy = "Performance_ExplorePostRoutePhysOpt")
        return (; f, s, cuts)
    end
end

# ---------------------------------------------------------------------------------------------------------------------
# 7. Testbench and board program
# ---------------------------------------------------------------------------------------------------------------------
function network_rt_testbench(path, x0, ph0, rec, cuts; h, non, noff, N, Q)
    fs(v) = repr(Float64(v))
    rowsQ = [join(map(fs, Q[i, :]), ", ") for i in 1:N]
    open(path, "w") do io
        print(io, """
        // $(basename(path)): written by PHPS.RT.network_rt_testbench (src/rt/network/network.jl); do not edit
        // C simulation: the network at its operating point driven by the machines' equilibrium currents (rotated on
        // chip), the fault at bus 16 for steps $(non + 1)-$noff, $(cuts[end]) steps of h = $h s, cut at $(length(cuts))
        // checkpoints (every 0.01 s and around both edges); at each, the C++ efforts e = Q x (node voltages, branch
        // currents, both axes) next to the same @hls code run in Julia (Float64). Limits (NETWORK.md 6, fixed before
        // any result): efforts 1e-8 pu, phasor length 1e-10, a continuous run equal to the segmented one bit for bit.
        // Returns 1 if any fails.
        #include <cstdio>
        #include <cmath>
        #include "network.h"

        const int N = $N, NCUT = $(length(cuts)), N_ON = $non, N_OFF = $noff;
        const int CUTS[NCUT] = {$(join(cuts, ", "))};
        const double H = $(fs(h)), LIM_E = 1e-8, LIM_PH = 1e-10, PH0SQ = $(fs(ph0[1]^2));
        const double X0[2 * N] = {$(join(map(fs, x0), ", "))};
        const double PH0[2] = {$(fs(ph0[1])), $(fs(ph0[2]))};
        const double Q[N][N] = {$(join(("{" * r * "}" for r in rowsQ), ",\n"))};
        const double JX[NCUT + 1][2 * N] = {
        $(join(("    {" * join(map(fs, r[1]), ", ") * "}" for r in rec), ",\n"))
        };

        static double maxeff(const X *x, const double *j) {   // max |Q (x - j)| over both axes
            double m = 0;
            for (int a = 0; a < 2; a++)
                for (int i = 0; i < N; i++) {
                    double e = 0;
                    for (int k = 0; k < N; k++) e += Q[i][k] * (x[a * N + k].to_double() - j[a * N + k]);
                    m = fmax(m, fabs(e));
                }
            return m;
        }

        int main() {
            static X x[2 * N], xc[2 * N];
            X ph[2], phc[2];
            for (int i = 0; i < 2 * N; i++) { x[i] = X0[i]; xc[i] = X0[i]; }
            for (int i = 0; i < 2; i++) { ph[i] = PH0[i]; phc[i] = PH0[i]; }
            Sig v1, v2; bool st, sa;
            double we = 0, wp = 0;
            printf("%8s %8s %14s %12s %12s\\n", "step", "t [s]", "v1 [pu]", "max|de|", "|ph|^2-1");
            int prev = 0;
            for (int g = 0; g < NCUT; g++) {
                network_run(x, ph, prev, N_ON, N_OFF, CUTS[g] - prev, 32, v1, v2, st, sa);
                prev = CUTS[g];
                double de = maxeff(x, JX[g + 1]);
                double c = ph[0].to_double(), s = ph[1].to_double(), dp = fabs((c * c + s * s) / PH0SQ - 1);
                printf("%8d %8.5f %14.10f %12.3e %12.3e\\n", CUTS[g], CUTS[g] * H, v1.to_double(), de, dp);
                we = fmax(we, de); wp = fmax(wp, dp);
            }
            network_run(xc, phc, 0, N_ON, N_OFF, CUTS[NCUT - 1], 32, v1, v2, st, sa);
            int same = (phc[0] == ph[0]) && (phc[1] == ph[1]);
            for (int i = 0; i < 2 * N; i++) same = same && (xc[i] == x[i]);
            int fail = !(we <= LIM_E && wp <= LIM_PH && same);
            printf("C sim check: max|de| %.3e (limit %.0e), max|ph^2-1| %.3e (limit %.0e), continuous = segmented %s: %s\\n",
                   we, LIM_E, wp, LIM_PH, same ? "yes" : "NO", fail ? "FAIL" : "PASS");
            return fail;
        }
        """)
    end
    println("written $path")
    return path
end

# C/RTL co-simulation testbench (review 36b, 4): from the Julia states 20 steps before the fault and 20 steps before its
# clearance, 40 steps through each edge; prints every state and the phasor as raw 48-bit integers after every step,
# so the C simulation's and the co-simulation's outputs can be compared bit for bit (diff).
function network_rt_cosim_tb(path, x0, ph0; non, noff, N, pre = 20, len = 40)
    starts = [non - pre, noff - pre]
    rec = _network_rt_julia(x0, ph0, starts; non, noff)
    fs(v) = repr(Float64(v))
    open(path, "w") do io
        print(io, """
        // $(basename(path)): written by PHPS.RT.network_rt_cosim_tb (src/rt/network/network.jl); do not edit
        // Co-simulation testbench: $len single steps from step $(starts[1]) (fault on after step $non) and from step
        // $(starts[2]) (fault off after step $noff), Julia states as the start; prints every state and the phasor as raw
        // integers after each step. Compare the C simulation's output with the co-simulation's (diff): bit for bit.
        #include <cstdio>
        #include "network.h"
        const int N = $N, N_ON = $non, N_OFF = $noff, LEN = $len;
        const int START[2] = {$(starts[1]), $(starts[2])};
        const double XS[2][2 * N] = {$(join(("{" * join(map(fs, r[1]), ", ") * "}" for r in rec[2:end]), ",\n    "))};
        const double PS[2][2] = {$(join(("{" * fs(r[2][1]) * ", " * fs(r[2][2]) * "}" for r in rec[2:end]), ", "))};
        int main() {
            FILE *o = fopen("network_cosim_out.txt", "w");
            for (int c = 0; c < 2; c++) {
                X x[2 * N], ph[2];
                for (int i = 0; i < 2 * N; i++) x[i] = XS[c][i];
                ph[0] = PS[c][0]; ph[1] = PS[c][1];
                Sig v1, v2; bool st, sa;
                for (int k = 0; k < LEN; k++) {
                    network_run(x, ph, START[c] + k, N_ON, N_OFF, 1, 32, v1, v2, st, sa);
                    fprintf(o, "%d", START[c] + k + 1);
                    for (int i = 0; i < 2 * N; i++) fprintf(o, " %s", x[i].to_string(16).c_str());
                    fprintf(o, " %s %s\\n", ph[0].to_string(16).c_str(), ph[1].to_string(16).c_str());
                }
            }
            fclose(o);
            printf("cosim testbench: %d steps written to network_cosim_out.txt\\n", 2 * LEN);
            return 0;
        }
        """)
    end
    println("written $path")
    return path
end

function network_rt_board_script(path, x0, ph0, f; h, non, noff, N, Q, decim = 32)
    cuts = [non, round(Int, 0.15 / h), round(Int, 0.3 / h)]
    rec = _network_rt_julia(x0, ph0, cuts; non, noff)
    N2 = 10_000_000
    rec2 = _network_rt_julia(x0, ph0, [N2]; non = -10, noff = -10)
    fs(v) = repr(Float64(v)); Fx = 48 - f.Ix
    open(path, "w") do io
        print(io, """
        # $(basename(path)): written by PHPS.RT.network_rt_board_script (src/rt/network/network.jl); do not edit.
        # Runs the network IP on the PYNQ-Z2. Next to it: network.bit, network.hwh and xnetwork_run_hw.h.
        #   python3 network_board.py [--no-download]
        import sys, time, re
        from pynq import Overlay, Clocks

        H, N, FX = $(fs(h)), $N, $Fx
        X0 = [$(join(map(fs, x0), ", "))]
        PH0 = [$(fs(ph0[1])), $(fs(ph0[2]))]
        N_ON, N_OFF, DECIM, CUTS = $non, $noff, $decim, $cuts
        X_JULIA = [$(join(("[" * join(map(fs, r[1]), ", ") * "]" for r in rec[2:end]), ", "))]
        X2_JULIA = [$(join(map(fs, rec2[end][1]), ", "))]
        PH2_JULIA = [$(fs(rec2[end][2][1])), $(fs(rec2[end][2][2]))]
        Q = [$(join(("[" * join(map(fs, Q[i, :]), ", ") * "]" for i in 1:N), ", "))]

        A = {}
        for m in re.finditer(r"#define XNETWORK_RUN_CONTROL_ADDR_(\\w+?)_(DATA|BASE)\\s+0x([0-9a-fA-F]+)", open("xnetwork_run_hw.h").read()):
            A[m.group(1)] = int(m.group(3), 16)

        DOWNLOAD = "--no-download" not in sys.argv
        ol = Overlay("network.bit", download=DOWNLOAD)
        if abs(Clocks.fclk0_mhz - $(1000 / NETWORK_RT_CLOCK_NS)) > 0.01:
            Clocks.fclk0_mhz = $(1000 / NETWORK_RT_CLOCK_NS)
        print("bitstream", "loaded by PYNQ" if DOWNLOAD else "kept (already loaded)", "| FCLK0 = %.3f MHz" % Clocks.fclk0_mhz)
        ip = ol.network_run_0

        def w48(addr, v, F=FX):
            r = int(round(v * 2**F)) & ((1 << 48) - 1)
            ip.write(addr, r & 0xFFFFFFFF)
            ip.write(addr + 4, r >> 32)
        def r48(addr, F=FX):
            r = ip.read(addr) | (ip.read(addr + 4) & 0xFFFF) << 32
            if r >> 47:
                r -= 1 << 48
            return r / 2**F
        def eff(x, j):
            return max(abs(sum(Q[i][k] * (x[a * N + k] - j[a * N + k]) for k in range(N))) for a in range(2) for i in range(N))

        def run(x, ph, n0, nsteps, n_on, n_off):
            for i, v in enumerate(x):
                w48(A["X"] + 8 * i, v)
            for i, v in enumerate(ph):
                w48(A["PH"] + 8 * i, v)
            ip.write(A["N0"], n0); ip.write(A["N_ON"], n_on); ip.write(A["N_OFF"], n_off)
            ip.write(A["NSTEPS"], nsteps); ip.write(A["DECIM"], DECIM)
            t0 = time.perf_counter()
            ip.write(0x00, 1)
            while not (ip.read(0x00) & 0x2):
                time.sleep(0.001)
            t = time.perf_counter() - t0
            return t, [r48(A["X"] + 8 * i) for i in range(2 * N)], [r48(A["PH"] + 8 * i) for i in range(2)]

        print("run 1: fault at bus 16 for steps", N_ON + 1, "-", N_OFF, "(0.1 s to 0.2 s), compared at", [round(c * H, 3) for c in CUTS], "s")
        x, ph, n, ttot, worst = list(X0), list(PH0), 0, 0.0, 0.0
        for g, c in enumerate(CUTS):
            t, x, ph = run(x, ph, n, c - n, N_ON, N_OFF)
            ttot += t
            e = eff(x, X_JULIA[g]); worst = max(worst, e)
            print("  t = %.3f s: max |board - Julia| over the efforts (node voltages, branch currents) %.3e pu" % (c * H, e))
            n = c
        print("  host elapsed time %.4f s (simulated %.4f s)" % (ttot, CUTS[-1] * H))

        N2 = $N2
        print("run 2: no fault,", N2, "steps (%.3f s of simulated time)" % (N2 * H))
        t, x, ph = run(X0, PH0, 0, N2, -10, -10)
        print("  host elapsed time %.4f s, %.4f us per step (h = %.4f us)" % (t, t / N2 * 1e6, H * 1e6))
        print("  max |board - Julia| over the efforts after %d steps: %.3e pu" % (N2, eff(x, X2_JULIA)))
        e2 = eff(x, X2_JULIA); pl = abs((ph[0]**2 + ph[1]**2) / PH0[0]**2 - 1)
        print("  phasor: board (%.12f, %.12f), Julia (%.12f, %.12f); length |c^2 + s^2 - 1| %.3e" %
              (ph[0], ph[1], PH2_JULIA[0], PH2_JULIA[1], pl))
        # limits (NETWORK.md 6, fixed before results): efforts 5e-8 pu in both runs, phasor length 1e-10
        ok = worst <= 5e-8 and e2 <= 5e-8 and pl <= 1e-10
        print("board check: run 1 max %.3e, run 2 %.3e (limit 5e-8), phasor length %.3e (limit 1e-10): %s"
              % (worst, e2, pl, "PASS" if ok else "FAIL"))
        sys.exit(0 if ok else 1)
        """)
    end
    println("written $path")
    return path
end
