# The flat operation graph of @hls code (the shared engine's compiler, step 1; review 46, finding 1): a symbolic
# interpreter over the recorded source (HLS_DEFS), not a trace of one run. Per call it binds the arguments (numbers and
# the coefficient struct as compile-time constants, input arrays as graph inputs, kept tables as run-time memories or as
# constants), unrolls the loops (constant bounds), inlines the @hls calls, turns if / elseif with a run-time condition
# into both branches and predicated selections (if-conversion), folds everything that is a compile-time constant, and
# records one node per hardware primitive:
#   :in      a graph input (an element of an input array)        :const   a numeric constant used by a node (immediate)
#   :mul :add :sub :neg  arithmetic (fmt the stored format; :exact = an intermediate inside a larger expression)
#   :cvt     a format conversion (alignment, rounding)            :shiftc  x 2^k, k constant
#   :cmp     a numeric comparison -> predicate                    :sel     a numeric selection by a predicate
#   :pand :por :pnot :psel   predicate logic                      :floor   the integer part (ifloor)
#   :iadd :isub :imul :icmp :isel   integer and address arithmetic (index registers)
#   :mem     a table read (info = the memory; one address operand, constant offsets folded into the instruction)
#   intrinsics (special units, the census of finding 1): :norm (a = m 2^k: m and k), :vshift (x 2^n, n an integer
#   register), :sidx (a seed / ROM index from m's leading bits), :cell (a window cell: index, coordinate, inside flag)
# Values: Int, Bool, Float64 (compile-time), GV (a node), arrays of values (locals and outputs), MemRef (a table).

struct GV
    id::Int
    fmt::Symbol
end
struct MemRef
    name::Symbol
    kind::Symbol                # :rom or :keep (run-time table)
    data::Array
end
mutable struct HGraph
    op::Vector{Symbol}
    args::Vector{Vector{Int}}
    fmt::Vector{Symbol}
    info::Vector{Any}
    inputs::Dict{Any,Int}       # (array name, index) -> node
    consts::Dict{Tuple{Float64,Symbol},Int}
    tag::Vector{Symbol}         # the body / item label of every node (for the census)
    curtag::Symbol
end
HGraph() = HGraph(Symbol[], Vector{Int}[], Symbol[], Any[], Dict{Any,Int}(), Dict{Tuple{Float64,Symbol},Int}(), Symbol[], :none)
function gnode!(g::HGraph, op::Symbol, args::Vector{Int}, fmt::Symbol, info = nothing)
    push!(g.op, op); push!(g.args, args); push!(g.fmt, fmt); push!(g.info, info); push!(g.tag, g.curtag)
    return GV(length(g.op), fmt)
end
gconst!(g::HGraph, v::Real, fmt::Symbol) =
    GV(get!(() -> gnode!(g, :const, Int[], fmt, Float64(v)).id, g.consts, (Float64(v), fmt)), fmt)

# the number formats of the code (Int and Bool are integers and predicates)
const _HG_INT = Set([:Int]); const _HG_BOOL = Set([:Bool])
_hgfmt(T::Symbol) = T in _HG_INT ? :int : T in _HG_BOOL ? :bool : T

# intrinsics: @hls functions realised by a special unit (their ladders are not expanded)
const HG_INTRINSICS = Dict{Symbol,Symbol}()
const HG_CONSTKEEPS = Set{Symbol}()     # kept tables that hold per-instance coefficients (compile-time constants)

mutable struct HGEnv
    vars::Dict{Symbol,Any}
    types::Dict{Symbol,Symbol}
    ret::Any
end
HGEnv() = HGEnv(Dict{Symbol,Any}(), Dict{Symbol,Symbol}(), nothing)
_hgcopyval(v) = v isa Array ? copy(v) : v
function _hgcopy(e::HGEnv)
    # arrays are shared by reference between caller and callee; a branch copy must copy them (elements are immutable)
    seen = IdDict{Any,Any}()
    vars = Dict{Symbol,Any}(k => (v isa Array ? get!(() -> copy(v), seen, v) : v) for (k, v) in e.vars)
    return HGEnv(vars, copy(e.types), e.ret), seen
end

isconstv(v) = v isa Real
_numop(g, op, a, b, fmt = :exact) = gnode!(g, op, Int[_hgid(g, a, fmt), _hgid(g, b, fmt)], fmt)
_hgid(g, v::GV, _) = v.id
_hgid(g, v::Bool, _) = gconst!(g, Int(v), :bool).id
_hgid(g, v::Integer, _) = gconst!(g, v, :int).id
_hgid(g, v::Real, fmt) = gconst!(g, v, fmt === :int || fmt === :bool ? :exact : fmt).id
_isint(v) = v isa Integer || (v isa GV && v.fmt === :int)
_isbool(v) = v isa Bool || (v isa GV && v.fmt === :bool)

# coerce a value into a declared format (an assignment): an :exact node takes the format (the op stores it there)
function _hgcoerce(g, v, T::Symbol)
    f = _hgfmt(T)
    f === :int && return v isa Real ? Int(v) : v
    f === :bool && return v
    v isa Real && return Float64(v)
    v isa GV || error("hls_graph: cannot store $(typeof(v)) as $T")
    v.fmt === f && return v
    if v.fmt === :exact
        g.fmt[v.id] = f
        return GV(v.id, f)
    end
    return gnode!(g, :cvt, [v.id], f)
end

function hg_call(g::HGraph, name::Symbol, args::Vector)
    haskey(HG_INTRINSICS, name) && return _hg_intrinsic(g, HG_INTRINSICS[name], name, args)
    ex = HLS_DEFS[name]
    sig = ex.args[1]; call = sig.head === :(::) ? sig.args[1] : sig
    rtype = sig.head === :(::) ? sig.args[2] : nothing
    env = HGEnv()
    for (a, v) in zip(call.args[2:end], args)
        if a isa Expr && a.head === :(::)
            nm, T = a.args[1], a.args[2]
            if T isa Symbol && (v isa GV || v isa Real)
                env.types[nm] = T
                v = _hgcoerce(g, v, T)
            elseif T isa Expr && T.head === :ref
                env.types[nm] = T.args[1]                          # an array argument: its element format
            end
            env.vars[nm] = v
        else
            env.vars[a] = v
        end
    end
    _hgexec(g, ex.args[2], env)
    r = env.ret
    return rtype === nothing || r === nothing ? r : _hgcoerce(g, r, rtype)
end

function _hgexec(g, s, env::HGEnv)
    s isa LineNumberNode && return
    s isa Expr || return
    h = s.head
    if h === :block
        for x in s.args; _hgexec(g, x, env); env.ret === nothing || return; end
    elseif h === :macrocall
        return                                                     # @pragma
    elseif h === :return
        env.ret = _hgev(g, s.args[1], env)
    elseif h === :(::)                                             # z::Sig[4] (a new array) or x::T
        v, t = s.args
        if t isa Expr && t.head === :ref
            dims = Tuple(_hgev(g, d, env) for d in t.args[2:end])
            env.vars[v] = Array{Any}(nothing, dims...); env.types[v] = t.args[1]
        else
            env.types[v] = t
        end
    elseif h === :(=)
        lhs, rhs = s.args
        if lhs isa Expr && lhs.head === :(::)
            v, t = lhs.args
            if t isa Expr && t.head === :ref                       # v::T[...] = rom(:N) / keep(:N) / (a, b)
                if rhs isa Expr && rhs.head === :call && rhs.args[1] in (:rom, :keep)
                    nm = rhs.args[2].value
                    data = rhs.args[1] === :rom ? HLS_ROMS[nm] : HLS_KEEPS[nm]
                    env.vars[v] = rhs.args[1] === :keep && nm in HG_CONSTKEEPS ? copy(data) :
                                  MemRef(nm, rhs.args[1], data)
                else
                    vals = rhs isa Expr && rhs.head in (:tuple, :vect) ? rhs.args : [rhs]
                    env.vars[v] = Any[_hgcoerce(g, _hgev(g, x, env), t.args[1]) for x in vals]
                end
                env.types[v] = t.args[1]
            else
                env.types[v] = t
                env.vars[v] = _hgcoerce(g, _hgev(g, rhs, env), t)
            end
        elseif lhs isa Symbol
            val = _hgev(g, rhs, env)
            env.vars[lhs] = haskey(env.types, lhs) ? _hgcoerce(g, val, env.types[lhs]) : val
        elseif lhs isa Expr && lhs.head === :ref
            arr = _hgev(g, lhs.args[1], env)
            arr isa Array || error("hls_graph: a store into $(lhs.args[1]) (a table) in the evaluated mode")
            idx = map(i -> _hgev(g, i, env), lhs.args[2:end])
            all(isconstv, idx) || error("hls_graph: a store at a run-time index into $(lhs.args[1])")
            T = get(env.types, lhs.args[1] isa Symbol ? lhs.args[1] : :_, nothing)
            val = _hgev(g, rhs, env)
            arr[idx...] = T isa Symbol ? _hgcoerce(g, val, T) : val
        else
            error("hls_graph: assignment to $lhs")
        end
    elseif h in (:+=, :-=, :*=)
        op = h === :+= ? :+ : h === :-= ? :- : :*
        _hgexec(g, Expr(:(=), s.args[1], Expr(:call, op, s.args[1], s.args[2])), env)
    elseif h === :for
        it, rng = s.args[1].args
        r = _hgev(g, rng, env)
        for i in r
            env.vars[it] = i
            _hgexec(g, s.args[2], env)
        end
    elseif h === :if || h === :elseif
        cx = s.args[1]
        cx isa Expr && cx.head === :block && (cx = last(filter(x -> !(x isa LineNumberNode), cx.args)))
        c = _hgev(g, cx, env)
        c isa GV && c.fmt !== :bool && error("hls_graph: a condition that is not a predicate")
        if c isa Bool
            c ? _hgexec(g, s.args[2], env) : (length(s.args) >= 3 && _hgexec(g, s.args[3], env))
        else
            e1, m1 = _hgcopy(env); _hgexec(g, s.args[2], e1)
            e2, m2 = _hgcopy(env); length(s.args) >= 3 && _hgexec(g, s.args[3], e2)
            _hgmerge!(g, env, c, e1, m1, e2, m2)
        end
    elseif h === :call
        _hgev(g, s, env)
    else
        error("hls_graph: statement $(h)")
    end
end

_hgsame(a, b) = a === b || (a isa Real && b isa Real && a == b && typeof(a) == typeof(b))
function _hgselv(g, c, a, b)
    _hgsame(a, b) && return a
    (a === nothing || b === nothing) && return a === nothing ? b : a       # a value defined in one branch only
    if _isbool(a) && _isbool(b)
        return gnode!(g, :psel, Int[c.id, _hgid(g, a, :bool), _hgid(g, b, :bool)], :bool)
    elseif _isint(a) && _isint(b)
        return gnode!(g, :isel, Int[c.id, _hgid(g, a, :int), _hgid(g, b, :int)], :int)
    end
    f = a isa GV ? a.fmt : b isa GV ? b.fmt : :exact
    return gnode!(g, :sel, Int[c.id, _hgid(g, a, f), _hgid(g, b, f)], f)
end
function _hgmerge!(g, env, c, e1, m1, e2, m2)
    for k in union(keys(e1.vars), keys(e2.vars))
        a = get(e1.vars, k, nothing); b = get(e2.vars, k, nothing)
        orig = get(env.vars, k, nothing)
        if orig isa Array
            # the branch copies of this array: merge element by element into the original (shared by reference)
            A = get(m1, orig, orig); B = get(m2, orig, orig)
            for i in eachindex(orig); orig[i] = _hgselv(g, c, A[i], B[i]); end
        elseif !(a isa Array)
            env.vars[k] = _hgselv(g, c, a, b)
        end
    end
    merge!(env.types, e1.types, e2.types)
end

const _HG_CMP = Set([:<, :<=, :>, :>=, :(==), :!=])
# an index expression as (run-time part, constant offset): `b + 19`, `la - 6047`, `ix + 1` -> the node and the offset
function _hgaddr(g, e, env)
    if e isa Expr && e.head === :call && e.args[1] in (:+, :-) && length(e.args) >= 3
        vs = map(x -> _hgev(g, x, env), e.args[2:end])
        sg = [1; fill(e.args[1] === :- ? -1 : 1, length(vs) - 1)]
        dyn = [(v, s_) for (v, s_) in zip(vs, sg) if !isconstv(v)]
        if length(dyn) == 1 && dyn[1][2] == 1
            off = sum((s_ * v for (v, s_) in zip(vs, sg) if isconstv(v)); init = 0)
            return (dyn[1][1], off)                       # (the index register, the instruction's offset)
        end
    end
    v = _hgev(g, e, env)
    return v isa GV ? (v, 0) : v
end
function _hgev(g, e, env::HGEnv)
    e isa Symbol && return e === :true ? true : e === :false ? false : haskey(env.vars, e) ? env.vars[e] :
                           e === :pi ? Float64(pi) : error("hls_graph: unknown name $e")
    e isa Real && return e
    e isa QuoteNode && return e.value
    e isa Expr || error("hls_graph: expression $e")
    h = e.head
    if h === :&& || h === :||
        a = _hgev(g, e.args[1], env)
        if a isa Bool
            (h === :&& && !a) && return false
            (h === :|| && a) && return true
            return _hgev(g, e.args[2], env)
        end
        b = _hgev(g, e.args[2], env)
        b isa Bool && return h === :&& ? (b ? a : false) : (b ? true : a)
        return gnode!(g, h === :&& ? :pand : :por, Int[a.id, b.id], :bool)
    elseif h === :ref
        arr = _hgev(g, e.args[1], env)
        if arr isa MemRef                    # a table address: constant terms go into the instruction (no adder)
            idx = map(i -> _hgaddr(g, i, env), e.args[2:end])
        else
            idx = map(i -> _hgev(g, i, env), e.args[2:end])
        end
        if arr isa Array
            all(isconstv, idx) || error("hls_graph: a run-time index into a local array $(e.args[1])")
            v = arr[idx...]
            v === nothing && error("hls_graph: $(e.args[1])$(idx) read before it is written")
            return v
        elseif arr isa MemRef
            all(isconstv, idx) && arr.kind === :rom && return Float64(arr.data[idx...])
            # a table read at the row-major (C++) word address  stride x register + offset  (1-based indices of the
            # source; the offset holds every constant part, the register at most one position)
            dims = size(arr.data); strides = [prod(dims[k+1:end]; init = 1) for k in eachindex(dims)]
            off = 0; reg = nothing; stride = 0
            for (k, ix) in enumerate(idx)
                if isconstv(ix)
                    off += (ix - 1) * strides[k]
                else
                    reg === nothing || error("hls_graph: two run-time index parts into $(arr.name)")
                    reg, o = ix; off += (o - 1) * strides[k]; stride = strides[k]
                end
            end
            T = e.args[1] isa Symbol ? get(env.types, e.args[1], :mem) : :mem    # the word format of the table
            T isa Symbol || (T = :mem)
            reg === nothing && return gnode!(g, :mem, Int[], T, (arr.name, 0, off))              # a fixed address
            return gnode!(g, :mem, Int[reg.id], T, (arr.name, stride, off))
        elseif arr isa Symbol
            error("hls_graph: $(e.args[1])")
        end
        error("hls_graph: indexing $(typeof(arr))")
    elseif h === :.
        s = _hgev(g, e.args[1], env)
        return getfield(s, e.args[2].value)
    elseif h === :call
        f = e.args[1]
        if f === :(:)
            return _hgev(g, e.args[2], env):_hgev(g, e.args[3], env)
        elseif f in (:+, :-, :*)
            vs = map(x -> _hgev(g, x, env), e.args[2:end])
            if f === :- && length(vs) == 1
                v = vs[1]; return v isa Real ? -v : gnode!(g, _isint(v) ? :ineg : :neg, Int[v.id], v isa GV ? v.fmt : :exact)
            end
            acc = vs[1]
            for b in vs[2:end]; acc = _hgarith(g, f, acc, b); end
            return acc
        elseif f in _HG_CMP
            a = _hgev(g, e.args[2], env); b = _hgev(g, e.args[3], env)
            a isa Real && b isa Real && return getfield(Base, f)(a, b)
            op = _isint(a) && _isint(b) ? :icmp : :cmp
            return gnode!(g, op, Int[_hgid(g, a, :int), _hgid(g, b, :int)], :bool, f)
        elseif f === :!
            a = _hgev(g, e.args[2], env)
            return a isa Bool ? !a : gnode!(g, :pnot, Int[a.id], :bool)
        elseif f === :scale2
            x = _hgev(g, e.args[2], env); k = _hgev(g, e.args[3], env)
            x isa Real && return x * 2.0^k
            k == 0 && return x
            return gnode!(g, :shiftc, Int[x.id], :exact, k)
        elseif f === :ifloor
            x = _hgev(g, e.args[2], env)
            return x isa Real ? floor(Int, x) : gnode!(g, :floor, Int[x.id], :int)
        elseif f in (:rom, :keep)
            error("hls_graph: $(f) outside a declaration")
        elseif haskey(HLS_DEFS, f)
            return hg_call(g, f, Any[_hgev(g, x, env) for x in e.args[2:end]])
        end
        error("hls_graph: call to $f")
    end
    error("hls_graph: expression head $h")
end

function _hgarith(g, f, a, b)
    if a isa Real && b isa Real
        return f === :+ ? a + b : f === :- ? a - b : a * b
    end
    if _isint(a) && _isint(b)
        f === :* && (a isa Integer && a == 1) && return b
        f === :* && (b isa Integer && b == 1) && return a
        (f === :+ || f === :-) && b isa Integer && b == 0 && return a
        f === :+ && a isa Integer && a == 0 && return b
        op = f === :+ ? :iadd : f === :- ? :isub : :imul
        return gnode!(g, op, Int[_hgid(g, a, :int), _hgid(g, b, :int)], :int)
    end
    op = f === :+ ? :add : f === :- ? :sub : :mul
    if f === :* && (a isa Real || b isa Real)
        c = a isa Real ? a : b; x = a isa Real ? b : a
        c == 1 && return x
        c == 0 && return 0.0
        c > 0 && c == 2.0^round(log2(c)) &&
            return gnode!(g, :shiftc, Int[x.id], :exact, Int(round(log2(c))))
    end
    (f === :+ || f === :-) && b isa Real && b == 0 && return a
    return _numop(g, op, a, b)
end

# ---------------------------------------------------------------------------------------------------------------------
# the special units (finding 1: a ladder is one normalisation, not ~10 compares)
function _hg_intrinsic(g, kind::Symbol, name::Symbol, args)
    if kind === :norm                                    # (a, mo) -> k; mo[1] = m
        a, mo = args
        a isa Real && return hg_fold_intrinsic(name, args)
        n = gnode!(g, :norm, Int[a.id], :pair, name)
        mo[1] = gnode!(g, :proj, Int[n.id], :Rq, 1)
        return gnode!(g, :proj, Int[n.id], :int, 2)
    elseif kind === :vshift                              # (x, n) -> x 2^(+-n) with n an integer register
        x, n = args
        if n isa Integer
            return hg_fold_intrinsic(name, Any[x, n]; g)
        end
        return gnode!(g, :vshift, Int[x.id, n.id], :exact, name)
    elseif kind === :sidx                                # (m) -> integer index from the leading bits
        m = args[1]
        m isa Real && return getfield(@__MODULE__, name)(m)
        return gnode!(g, :sidx, Int[m.id], :int, name)
    elseif kind === :cell                                # (tt, window constants..., ci, ct) -> ok
        tt = args[1]; ci = args[end-1]; ct = args[end]
        tt isa Real && return getfield(@__MODULE__, name)(args...)        # a compile-time cell (the terminal's p_cond)
        n = gnode!(g, :cell, Int[tt.id], :pair, (name, args[2:end-2]...))
        ci[1] = gnode!(g, :proj, Int[n.id], :int, 1)
        ct[1] = gnode!(g, :proj, Int[n.id], :Wt, 2)
        return gnode!(g, :proj, Int[n.id], :bool, 3)
    end
    error("hls_graph: intrinsic $kind")
end
# an intrinsic with compile-time arguments: run the Julia function itself (a constant), or a constant shift of a node
function hg_fold_intrinsic(name, args; g = nothing)
    x, n = args[1], args[2]
    if x isa GV
        # a shift by a compile-time amount: the function's own ladder picks one scale2; find it from x = 1
        r = getfield(@__MODULE__, name)(1.0, n)
        k = round(Int, log2(r))
        return k == 0 ? x : gnode!(g, :shiftc, Int[x.id], :exact, k)
    end
    return getfield(@__MODULE__, name)(args...)
end

# ---------------------------------------------------------------------------------------------------------------------
"""    hg_census(g; tags = nothing): counts of the graph's nodes per op (memory reads per memory), optionally per tag."""
function hg_census(g::HGraph)
    c = Dict{Symbol,Int}(); m = Dict{Symbol,Int}(); inner = 0
    for i in eachindex(g.op)
        o = g.op[i]
        o === :proj && continue
        c[o] = get(c, o, 0) + 1
        o === :mem && (m[g.info[i]] = get(m, g.info[i], 0) + 1)
        g.fmt[i] === :exact && o in (:mul, :add, :sub, :neg, :shiftc, :vshift) && (inner += 1)
    end
    return (; ops = c, mems = m, inner)
end

# ---------------------------------------------------------------------------------------------------------------------
# Range instrumentation (review 47c, 1): an @hls function rewritten to record every value it assigns (every typed
# declaration, plain assignment, update `+=` (so a loop accumulator's partial sums), and element store) with its declared
# type: `hls_instrument(:f)` defines `f_rng(rec, args...)`, which runs as `f` and updates `rec[(T, name)] = max |value|`.
# The C++ formats are then checked against the recorded maxima (each assignment is where the C++ rounds / wraps).
# ---------------------------------------------------------------------------------------------------------------------
function _hgins(e, types::Dict{Symbol,Any})
    e isa Expr || return e
    h = e.head
    if h === :block
        return Expr(:block, map(x -> _hgins(x, types), e.args)...)
    elseif h === :for || h === :while                                # the header (i = 1:4) is not an assignment
        return Expr(h, e.args[1], map(x -> _hgins(x, types), e.args[2:end])...)
    elseif h === :if || h === :elseif
        return Expr(h, map(x -> _hgins(x, types), e.args)...)
    elseif h === :(::) && length(e.args) == 2 && e.args[2] isa Expr && e.args[2].head === :ref
        types[e.args[1]] = e.args[2].args[1]                        # z::Sig[4]: element type
        return e
    elseif h === :(=) || h in (:+=, :-=, :*=)
        lhs = e.args[1]
        if lhs isa Expr && lhs.head === :(::)
            v, T = lhs.args
            if T isa Expr && T.head === :ref
                types[v] = T.args[1]; return e                       # an array declaration with values / rom / keep
            end
            types[v] = T
            return Expr(:block, e, :(_hg_rec!(__rec, $(QuoteNode(T)), $(QuoteNode(v)), $v)))
        elseif lhs isa Symbol
            T = get(types, lhs, :Any)
            return Expr(:block, e, :(_hg_rec!(__rec, $(QuoteNode(T)), $(QuoteNode(lhs)), $lhs)))
        elseif lhs isa Expr && lhs.head === :ref
            a = lhs.args[1]; T = get(types, a, :Any)
            return Expr(:block, e, :(_hg_rec!(__rec, $(QuoteNode(T)), $(QuoteNode(a)), $lhs)))
        end
    end
    return e
end
_hg_rec!(rec, T, v, x::Real) = (k = (T, v); rec[k] = max(get(rec, k, 0.0), abs(Float64(x))); rec[(:val, v)] = Float64(x); nothing)
_hg_rec!(rec, T, v, x) = nothing
"""    hls_instrument(name): defines `<name>_rng(rec, args...)` (the @hls function recording its assigned values' maxima)."""
function hls_instrument(name::Symbol; mod = @__MODULE__, deep = false)
    ex = HLS_DEFS[name]
    sig = ex.args[1]; call = sig.head === :(::) ? sig.args[1] : sig
    types = Dict{Symbol,Any}()
    args = map(call.args[2:end]) do a
        if a isa Expr && a.head === :(::)
            T = a.args[2]
            types[a.args[1]] = T isa Expr && T.head === :ref ? T.args[1] : T
            T isa Expr && T.head === :ref ? Expr(:(::), a.args[1], _arrjulia(T)) : a
        else
            a
        end
    end
    body = _hls_julia(_hgins(ex.args[2], types))
    if deep          # calls of other @hls functions go to their instrumented copies (ranges of nested intermediates)
        callees = Set{Symbol}()
        _deepcall(e) = e
        function _deepcall(e::Expr)
            a2 = map(_deepcall, e.args)
            if e.head === :call && a2[1] isa Symbol && a2[1] !== name && haskey(HLS_DEFS, a2[1]) &&
               HLS_DEFS[a2[1]] isa Expr && HLS_DEFS[a2[1]].head === :function
                push!(callees, a2[1])
                return Expr(:call, Symbol(replace(String(a2[1]), "!" => ""), :_rng), :__rec, a2[2:end]...)
            end
            return Expr(e.head, a2...)
        end
        body = _deepcall(body)
        foreach(c -> hls_instrument(c; mod, deep = true), callees)
    end
    fname = Symbol(replace(String(name), "!" => ""), :_rng)
    f = Expr(:function, Expr(:call, fname, :__rec, args...), body)
    Core.eval(mod, f)
    return getfield(mod, fname)
end
