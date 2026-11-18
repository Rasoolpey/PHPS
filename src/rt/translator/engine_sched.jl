# The shared engine's scheduling core (review 46b: the connectivity experiment). Input: the frame graph of hls_graph.jl;
# a machine (unit classes, counts, latencies, accumulator contexts); output: a modulo schedule at II = P, the frame
# distances L (f history) and d (seeds), the MAC chains with their accumulator widths, and a binding of every operand
# delivery under a fixed producer -> register-file connectivity.
#
# Machine model (46b, findings 2, 3, 5):
#   MAC  issue one term per cycle; term = s (a [+- c]) b or an addend; the product pipeline LM cycles, the accumulator
#        add one cycle (a chain's terms may issue on consecutive cycles); a chain is owned by one MAC and holds one of
#        its NCTX accumulator contexts from its first product's arrival (first issue + LM) to its result's read-out
#        (last issue + LM + 1); the result (rounded once) is ready then
#   ALU, IALU, SPC  one op per instance per cycle, latencies LA, LI, LS;  MEM  one read port per bank, latency LMEM
#   STATE  NST ports (reads of x_n, remainders, f history, seeds, omega; writes of f_n, x_n+1, remainders, governor, P)
#   files  one per unit input (MAC a, b, c; ALU a, b, p; IALU a, b, c; SPC a, b; MEM address; STATE data), one write
#          per cycle, depth DEPTH, the write counter reset at each frame start (frame-relative addresses); constants come
#          from a per-unit constant file (no write traffic)
# The connectivity: a set of producers per file (F at most), shared by every instance (derived from all instances'
# schedules and fixed); a producer can hold a result HOLD extra cycles to wait for a free write port.

struct EngineMachine
    P::Int
    n::Dict{Symbol,Int}         # :MAC :ALU :IALU :SPC :STATE (MEM: one per bank)
    lat::Dict{Symbol,Int}       # :MAC (product pipeline) :ALU :IALU :SPC :MEM :STATE
    nctx::Int
    depth::Int
    hold::Int
end
EngineMachine(; P = 110, MAC = 21, ALU = 15, IALU = 7, SPC = 3, STATE = 2, LM = 6, LA = 2, LI = 1, LS = 3, LMEM = 2,
              LST = 2, nctx = 4, depth = 32, hold = 1) =
    EngineMachine(P, Dict(:MAC => MAC, :ALU => ALU, :IALU => IALU, :SPC => SPC, :STATE => STATE),
                  Dict(:MAC => LM, :ALU => LA, :IALU => LI, :SPC => LS, :MEM => LMEM, :STATE => LST), nctx, depth, hold)

const _ES_ALU = Set([:add, :sub, :neg, :cvt, :shiftc, :cmp, :sel])
const _ES_IALU = Set([:iadd, :isub, :imul, :ineg, :icmp, :isel, :floor, :pand, :por, :pnot, :psel])
const _ES_SPC = Set([:norm, :vshift, :sidx, :cell])

# the formats with :exact resolved to the storing consumer's (the chain's destination)
function es_formats(G::HGraph)
    n = length(G.op); cons = [Int[] for _ in 1:n]
    for i in 1:n, a in G.args[i]; push!(cons[a], i); end
    fm = copy(G.fmt)
    for _ in 1:30
        ch = false
        for i in n:-1:1
            fm[i] === :exact || continue
            for c in cons[i]
                f = fm[c]
                if !(f in (:exact, :pair, :bool, :mem)) && G.op[c] !== :cmp
                    fm[i] = f; ch = true; break
                end
            end
        end
        ch || break
    end
    return fm, cons
end

"""
    es_ops(G) -> ops

The machine operations of the frame graph: MAC chains (terms, root, width), ALU / IALU / SPC ops, MEM reads (bank),
STATE reads (inputs) and writes (`outs`, the values the frame stores). Each op: class, the values it reads (node ids
of producing ops' results), the node its result is (or nothing).
"""
function es_ops(G::HGraph, outs; I = nothing)
    n = length(G.op); fm, cons = es_formats(G)
    exact(i) = G.fmt[i] === :exact
    isadd(i) = G.op[i] in (:add, :sub)
    absorbed = falses(n)
    for i in 1:n
        isadd(i) || continue
        absorbed[i] = any(a -> (G.op[a] === :mul && exact(a)) || (absorbed[a] && exact(a)), G.args[i])
    end
    pre = falses(n)
    for i in 1:n
        G.op[i] === :mul || continue
        for a in G.args[i]
            isadd(a) && exact(a) && !absorbed[a] && length(cons[a]) == 1 && (pre[a] = true)
        end
    end
    # the value a node delivers: proj -> its parent op's output; const -> a constant (no producer)
    src(a) = G.op[a] === :proj ? G.args[a][1] : a
    isconst(a) = G.op[a] === :const
    ops = NamedTuple[]
    opof = Dict{Int,Int}()                   # node -> the op producing it
    # chain roots: a mul or absorbed add whose result is stored (not exact) or read by a non-absorbing consumer
    inchain(i) = (G.op[i] === :mul && exact(i)) || (absorbed[i] && exact(i))
    function terms!(i, acc)
        if G.op[i] === :mul
            push!(acc, i)
        elseif absorbed[i]
            for a in G.args[i]
                inchain(a) ? terms!(a, acc) : push!(acc, -a)          # -a: an addend term (one operand)
            end
        end
        return acc
    end
    Iof(f) = I === nothing ? 0 : get(I, f, 0)
    for i in 1:n
        o = G.op[i]
        o in (:const, :proj) && continue
        pre[i] && continue
        if o === :mul || absorbed[i]
            inchain(i) && all(c -> inchain(c) || (absorbed[c] && G.op[c] in (:add, :sub)), cons[i]) && !isempty(cons[i]) && continue
            ts = terms!(i, Int[])
            reads = Int[]; widths = Int[]; tr = Int[]
            for t in ts
                n0 = length(reads)
                if t > 0
                    for a in G.args[t]
                        if pre[a]
                            for b in G.args[a]; isconst(b) || push!(reads, src(b)); end
                        else
                            isconst(a) || push!(reads, src(a))
                        end
                    end
                    # the exact product's width: two 48-bit words (+1 for a pre-add)
                    push!(widths, 96 + (any(a -> pre[a], G.args[t]) ? 1 : 0))
                else
                    isconst(-t) || push!(reads, src(-t))
                    push!(widths, 48)
                end
                push!(tr, length(reads) - n0)
            end
            # the accumulator: aligned at the finest term binary point; integer bits of the largest term + log2(terms)
            fr = Int[]; ib = Int[]
            for t in ts
                if t > 0
                    a, b = G.args[t]; Fa = 48 - Iof(fm[a]); Fb = 48 - Iof(fm[b])
                    push!(fr, Fa + Fb); push!(ib, Iof(fm[a]) + Iof(fm[b]) + (any(x -> pre[x], G.args[t]) ? 1 : 0))
                else
                    push!(fr, 48 - Iof(fm[-t])); push!(ib, Iof(fm[-t]))
                end
            end
            wacc = maximum(fr) + maximum(ib) + ceil(Int, log2(max(length(ts), 1))) + 1
            push!(ops, (; cls = :MAC, node = i, reads, nterms = length(ts), termreads = tr,
                        wacc, fmt = fm[i]))
            opof[i] = length(ops)
        elseif o === :in
            nm = G.info[i][1]                 # the state banks: X (states, remainders), F (f history), G (the rest)
            bk = nm === :x ? :X : nm === :cc ? :C : nm === :fa ? :FA : nm === :fb ? :FB : :G
            push!(ops, (; cls = Symbol(:ST_, bk, :R), node = i, reads = Int[], nterms = 0, termreads = Int[], wacc = 0, fmt = fm[i]))
            opof[i] = length(ops)
        else
            # a ROM of <= 512 words has one copy per body (cheap); larger ROMs and the run-time tables are shared
            mb = o === :mem ? (G.info[i][1] in keys(HLS_ROMS) && length(HLS_ROMS[G.info[i][1]]) <= 512 ?
                               Symbol(G.info[i][1], :_, G.tag[i]) : G.info[i][1]) : :none
            cls = o === :mem ? Symbol(:MEM_, mb) : o in _ES_ALU ? :ALU : o in _ES_IALU ? :IALU : o in _ES_SPC ? :SPC :
                  error("engine: op $o")
            reads = Int[src(a) for a in G.args[i] if !isconst(a)]
            push!(ops, (; cls, node = i, reads, nterms = 0, termreads = Int[], wacc = 0, fmt = fm[i]))
            opof[i] = length(ops)
        end
    end
    # the frame's stored values: STATE writes
    flat(v) = v isa AbstractArray ? reduce(vcat, map(flat, v); init = Any[]) : v isa Tuple ? reduce(vcat, map(flat, collect(v)); init = Any[]) : Any[v]
    for (fld, v0) in pairs(outs), v in flat(v0)
        v isa GV || continue
        bk = fld === :f ? :FA : fld === :xo ? :X : fld === :cco ? :C : :G     # f_n into the history ring (bank FA;
        # FB holds the same ring's second tap: written alongside, one write port each)
        push!(ops, (; cls = Symbol(:ST_, bk, :W), node = 0, reads = Int[src(v.id)], nterms = 0, termreads = Int[], wacc = 0, fmt = :Sig))
    end
    # operand reads -> op indices (the producing op)
    for (k, op) in enumerate(ops)
        for r in op.reads; haskey(opof, r) || error("engine: node $r ($(G.op[r])) has no producing op"); end
    end
    return ops, opof
end

"""
    es_schedule(G, outs, m; I) -> NamedTuple

Modulo list scheduling at II = m.P: critical-path priority, MAC chains on one instance with accumulator contexts, one
read port per bank, STATE ports. Returns per op the issue time(s), instance, ready time; L and d; the step readiness.
"""
function es_schedule(G::HGraph, outs, m::EngineMachine; I = nothing, stepfirst = Set{Int}())
    ops, opof = es_ops(G, outs; I)
    P = m.P; no = length(ops)
    pred = [unique(Int[opof[r] for r in op.reads]) for op in ops]
    succ = [Int[] for _ in 1:no]; for k in 1:no, p in pred[k]; push!(succ[p], k); end
    lat(op) = op.cls === :MAC ? op.nterms - 1 + m.lat[:MAC] + 1 : startswith(String(op.cls), "ST_") ? m.lat[:STATE] :
              startswith(String(op.cls), "MEM_") ? m.lat[:MEM] : m.lat[op.cls]
    # heights (longest path to a sink)
    h = zeros(Int, no)
    for k in no:-1:1; h[k] = lat(ops[k]) + (isempty(succ[k]) ? 0 : maximum(h[s] for s in succ[k])); end
    # resource tables (mod P)
    rt = Dict{Symbol,Vector{Int}}()
    cap(c) = startswith(String(c), "MEM_") || startswith(String(c), "ST_") ? 1 : m.n[c]    # one port per bank
    tab(c) = get!(() -> zeros(Int, P), rt, c)
    macissue = [zeros(Bool, P) for _ in 1:m.n[:MAC]]
    macctx = [zeros(Int, P) for _ in 1:m.n[:MAC]]
    t_issue = Vector{Vector{Int}}(undef, no); inst = zeros(Int, no); ready = zeros(Int, no)
    npred = [length(p) for p in pred]
    readyat = zeros(Int, no)
    heap = [(k) for k in 1:no if npred[k] == 0]
    done = 0
    while !isempty(heap)
        # pick the ready op of largest priority (the step's ops first, then height)
        j = argmax([(k in stepfirst ? 10^7 : 0) + h[k] for k in heap]); k = heap[j]; deleteat!(heap, j)
        op = ops[k]; t0 = readyat[k]
        if op.cls === :MAC
            # the terms issue on consecutive free cycles of one MAC (operands: all of the chain's reads ready at t0)
            best = nothing
            for u in 1:m.n[:MAC]
                t = t0; ts = Int[]
                while length(ts) < op.nterms
                    macissue[u][mod(t, P)+1] || push!(ts, t); t += 1
                    t - t0 > 4P && break
                end
                length(ts) < op.nterms && continue
                tend = ts[end] + m.lat[:MAC] + 1
                # the accumulator context: from the first product's arrival to the result's read-out
                ok = all(macctx[u][mod(c, P)+1] < m.nctx for c in ts[1]+m.lat[:MAC]:tend)
                ok || continue
                (best === nothing || tend < best[3]) && (best = (u, ts, tend))
            end
            best === nothing && error("engine: no MAC for a chain of $(op.nterms) terms at P = $P")
            u, ts, tend = best
            for c in ts; macissue[u][mod(c, P)+1] = true; end
            for c in ts[1]+m.lat[:MAC]:tend; macctx[u][mod(c, P)+1] += 1; end
            t_issue[k] = ts; inst[k] = u; ready[k] = tend
        else
            c = op.cls; tb = tab(c); t = t0
            while tb[mod(t, P)+1] >= cap(c)
                t += 1
                t - t0 > P && error("engine: class $c full at P = $P")
            end
            tb[mod(t, P)+1] += 1
            t_issue[k] = [t]; inst[k] = tb[mod(t, P)+1]; ready[k] = t + lat(op)
        end
        done += 1
        for s in succ[k]
            readyat[s] = max(readyat[s], ready[k] + 1); npred[s] -= 1          # + 1: the write into the reader's file
            npred[s] == 0 && push!(heap, s)
        end
    end
    done == no || error("engine: $(no - done) ops not scheduled (a cycle?)")
    return (; ops, opof, pred, succ, t_issue, inst, ready, lat)
end

"""    es_frame_report(G, outs, s, m): L, d, step readiness, unit occupancy, MAC widths."""
function es_frame_report(G, outs, s, m)
    P = m.P
    rdy(v) = (k = s.opof[v.id]; s.ready[k])
    issue_of_input(name) = [s.t_issue[s.opof[i]][1] for i in eachindex(G.op) if G.op[i] === :in && G.info[i][1] === name]
    tf = maximum(rdy(v) for v in outs.f if v isa GV)
    tx = maximum(rdy(v) for v in vcat(outs.xo, outs.cco, outs.go) if v isa GV)
    ts = maximum(rdy(v) for v in outs.seeds if v isa GV)
    rfa = minimum(issue_of_input(:fa)); rx = minimum(vcat(issue_of_input(:x), issue_of_input(:cc), issue_of_input(:gx)))
    rsd = minimum(issue_of_input(:seed))
    L = cld(tf + m.lat[:STATE] - rfa, P)       # f_n written by tf (+ the write), read in frame n + L at rfa
    d = cld(ts + m.lat[:STATE] - rsd, P)
    xok = tx + m.lat[:STATE] <= P + rx
    return (; tf, tx, ts, L, d, xok, rx, wacc = maximum(op.wacc for op in s.ops if op.cls === :MAC))
end

# ---------------------------------------------------------------------------------------------------------------------
# Binding under a fixed connectivity (46b, findings 2, 4, 5): every operand delivery (producer op -> consumer op's input
# file) needs a write into that file in [ready, ready + hold] before the consumer reads it, through a write port free
# in that cycle (one per file, mod P), from a producer adjacent to the file. The adjacency `adj` (file -> producers)
# is shared by all instances: grown up to F per file while learning, then frozen. A value already in the file (still
# live) is reused. Lifetimes longer than P are hazards under frame-relative addresses (counted).
# Files: (class, instance, port); producers: (class, instance).
_es_ports(op) = op.cls === :MAC ? 3 : op.cls in (:ALU, :IALU) ? 3 : op.cls === :SPC ? 2 : 1

function es_bind(s, m::EngineMachine; F = 4, adj = Dict{Tuple,Set{Tuple}}(), learn = true)
    P = m.P; ops = s.ops; no = length(ops)
    # instances: MAC as scheduled; ALU / IALU / SPC chosen per cycle; banks and state ports single
    inst = copy(s.inst)
    producer(k) = (ops[k].cls, inst[k])
    # deliveries: (time, consumer op, term, port candidates, value op)
    dl = Tuple{Int,Int,Vector{Int},Int}[]          # (time, consumer, ports, value)
    for k in 1:no
        op = ops[k]
        if op.cls === :MAC
            pos = 1
            for (ti, nr) in enumerate(op.termreads)
                t = s.t_issue[k][ti]
                for j in 1:nr
                    v = s.opof[op.reads[pos]]; pos += 1
                    push!(dl, (t, k, nr == 3 ? [j == 1 ? 1 : j == 2 ? 3 : 2] : [1, 2], v))   # a, b swap freely
                end
            end
        else
            for (j, r) in enumerate(op.reads); push!(dl, (s.t_issue[k][1], k, [j], s.opof[r])); end
        end
    end
    sort!(dl, by = first)
    wport = Dict{Tuple,Vector{Bool}}()                # file -> write port use per cycle (mod P)
    live = Dict{Tuple,Vector{Tuple{Int,Int,Int}}}()   # file -> (value, write time, last read)
    fails = 0; reused = 0; written = 0; long = 0
    # instance choice for the per-cycle classes: the free instance whose files accept most of the op's operands
    used = Dict{Tuple{Symbol,Int},Set{Int}}()         # (class, cycle mod P) -> instances taken
    chosen = falses(no)
    for c in (:ALU, :IALU, :SPC), k in 1:no
        ops[k].cls === c || continue
        t = s.t_issue[k][1]; key = (c, mod(t, P)); tk = get!(() -> Set{Int}(), used, key)
        best = 0; bscore = -1
        for u in 1:m.n[c]
            u in tk && continue
            sc = 0
            for (j, r) in enumerate(ops[k].reads)
                f = (c, u, j); pk = s.opof[r]
                pr = (ops[pk].cls, inst[pk])
                sc += pr in get(adj, f, Set{Tuple}()) ? 2 : (learn && length(get(adj, f, Set{Tuple}())) < F ? 1 : 0)
            end
            sc > bscore && (best = u; bscore = sc)
        end
        push!(tk, best); inst[k] = best; chosen[k] = true
    end
    for (t, k, ports, v) in dl
        tr = s.ready[v]; pr = producer(v); done = false
        for pt in ports
            f = (ops[k].cls, inst[k], pt)
            # already in the file and live?
            L = get!(() -> Tuple{Int,Int,Int}[], live, f)
            ix = findfirst(e -> e[1] == v, L)
            if ix !== nothing
                L[ix] = (v, L[ix][2], max(L[ix][3], t)); reused += 1; done = true; break
            end
            A = get!(() -> Set{Tuple}(), adj, f)
            (pr in A) || (learn && length(A) < F) || continue
            W = get!(() -> falses(P), wport, f)
            for w in tr:min(tr + m.hold, t - 1)
                if !W[mod(w, P)+1]
                    W[mod(w, P)+1] = true; push!(A, pr); push!(L, (v, w, t)); written += 1; done = true
                    t - w >= P && (long += 1)
                    break
                end
            end
            done && break
        end
        done || (fails += 1)
    end
    # peak occupancy per file (values live from write to last read, overlapped mod P across frames)
    peak = 0
    for (f, L) in live
        occ = zeros(Int, P)
        for (v, w, r) in L, c in w:r; occ[mod(c, P)+1] += 1; end
        peak = max(peak, maximum(occ))
    end
    nfiles = length(adj); fanin = isempty(adj) ? 0 : maximum(length, values(adj))
    return (; fails, reused, written, long, peak, nfiles, fanin, deliveries = length(dl), adj)
end
