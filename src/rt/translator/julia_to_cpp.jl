# Julia -> C++ by translating the SOURCE CODE (not a trace): a function or struct written with `@hls` runs in Julia as
# written, and `hls_write` turns it into C++ for Vitis HLS line by line: the same statements, the same loops (kept as
# loops, so HLS can reuse hardware), the same names and the same comments. The C++ is as long as the Julia.
#
# The subset (what a model written with @hls may use):
#   types      the names of number types, declared once per model as C++ typedefs at export (e.g. Sig, Small);
#              in Julia they are Float64 for now (const Sig = Float64)
#   arguments  x::Sig (a number), x::Sig[6] (an array, changed in place), P::Small[4, 4], k::SomeStruct
#   locals     v::Sig = expr        a new variable (always with its type)
#              z::Sig[4]            a new array;  v::Sig[3] = (a, b, c)  an array with values
#              v = expr, x[i] = expr, v += expr (-=, *=)
#   control    for i in 1:4 ... end (constant bounds), if / elseif / else, return
#   expressions + - * / (C++ order), comparisons, &&, ||, !, integer literals, x[i], x[i, j], k.field,
#              sin, cos, sqrt, abs, exp -> hls::sin, ...; calls to other @hls functions
#   constants  a float literal or pi only as the value of a new variable (twopi::Sig = 2pi)
#   pragmas    @pragma "HLS ARRAY_PARTITION variable=x complete" -> #pragma HLS ... (nothing in Julia)
#   ROM tables v::Cf[8, 40] = rom(:NAME)  a constant table registered with hls_rom!(:NAME, array): in Julia the
#              array itself, in C++ `static const Cf v[8][40] = NAME;` with NAME defined by hls_write_roms
#   kept tables v::Tw[6048] = keep(:NAME)  a table that keeps its words between calls (run-time data, e.g. an instance's
#              table window, written by a load mode of the top function): registered with hls_keep!(:NAME, array); in
#              Julia that array itself (mutable), in C++ `static Tw v[6048];`
#   outputs    v::Out{Sig} (an output port): in Julia a Ref, written with v[] = expr; in C++ volatile Sig &v, written
#              v = expr (volatile: every write reaches the port, also inside a loop)
#   integers   Int, Bool -> int, bool; calls to other @hls functions as statements (genrou_step!(x, u, k));
#              ifloor(x) the integer part of x >= 0 (a ROM index; rt_util.h)
# Indices are 1-based in Julia and 0-based in C++ (converted: x[2 + j] -> x[j + 2] with j counted from 0).

const Sig = Float64                 # the signal type (in C++: chosen at export, e.g. ap_fixed<48, 8>)
const Small = Float64               # the small-coefficient type (e.g. ap_fixed<48, -10>)
const Tiny = Float64                # a very small quantity of its own (e.g. ap_fixed<48, -30>), declared per model
const Coef = Float64                # coefficients of signal size (e.g. ap_fixed<42, 8>), declared per model

const Out{T} = Base.RefValue{T}       # an output port: Julia Ref (write with v[] = ...), C++ reference (T &v)

const HLS_DEFS = Dict{Symbol,Any}()  # name -> the @hls source (an Expr with its line numbers)
const HLS_ROMS = Dict{Symbol,Any}()  # name -> a constant table (Array) for `v::T[...] = rom(:NAME)`
const HLS_KEEPS = Dict{Symbol,Any}() # name -> a kept table (Array) for `v::T[...] = keep(:NAME)`

"""    hls_rom!(name, array): registers a constant table for `rom(:name)` in @hls code."""
hls_rom!(name::Symbol, a::AbstractArray) = (HLS_ROMS[name] = a)
rom(name::Symbol) = HLS_ROMS[name]
"""    hls_keep!(name, array): registers a kept table for `keep(:name)` in @hls code (C++: a static array)."""
hls_keep!(name::Symbol, a::AbstractArray) = (HLS_KEEPS[name] = a)
keep(name::Symbol) = HLS_KEEPS[name]
"""    scale2(x, k): x 2^k, exact (k a literal integer); in C++ `scale2<k>(x)` (HLS_UTIL_H): widen, then shift: wiring."""
scale2(x, k::Integer) = x * 2.0^k
"""    ifloor(x): the integer part of x >= 0 (a ROM index); in C++ `ifloor(x)` = x.to_int() (truncation = floor for x >= 0)."""
ifloor(x) = (x >= 0 || error("ifloor: negative argument $x"); floor(Int, x))
const HLS_UTIL_H = """
// rt_util.h: written by PHPS.RT (src/rt/translator/julia_to_cpp.jl); do not edit
#pragma once
#include <ap_fixed.h>
// x 2^K exactly: the format is widened by |K| bits first, so the shift loses nothing (no multiplier: wiring)
template<int K, int W, int I, ap_q_mode Q, ap_o_mode O, int N>
ap_fixed<W + (K < 0 ? -K : K), I + (K > 0 ? K : 0), Q, O, N> scale2(ap_fixed<W, I, Q, O, N> x) {
    ap_fixed<W + (K < 0 ? -K : K), I + (K > 0 ? K : 0), Q, O, N> y = x;
    if (K > 0) y <<= K; else if (K < 0) y >>= -K;
    return y;
}
// the integer part of x >= 0 (a ROM index): truncation, which is the floor for x >= 0
template<int W, int I, ap_q_mode Q, ap_o_mode O, int N>
int ifloor(ap_fixed<W, I, Q, O, N> x) { return x.to_int(); }
"""
"""    hls_write_util(dir): writes rt_util.h (scale2) into `dir`."""
hls_write_util(dir) = (p = joinpath(dir, "rt_util.h"); write(p, HLS_UTIL_H); println("written $p"); p)
_isrom(r) = r isa Expr && r.head === :call && r.args[1] === :rom
_iskeep(r) = r isa Expr && r.head === :call && r.args[1] === :keep

"""
    @pragma "HLS ..."

An instruction for Vitis HLS inside an @hls function: written to the C++ as `#pragma HLS ...`; does nothing in Julia.
"""
macro pragma(s)
    s isa String || error("@pragma: expects a string, e.g. @pragma \"HLS ARRAY_PARTITION variable=x complete\"")
    return nothing
end

_isarrtype(t) = t isa Expr && t.head === :ref                 # Sig[6], Small[4, 4]
_arrjulia(t) = length(t.args) == 2 ? :(AbstractVector) : :(AbstractMatrix)
_newarr(t) = length(t.args) == 2 ? :(Vector{$(t.args[1])}(undef, $(t.args[2]))) :
                                   :(Matrix{$(t.args[1])}(undef, $(t.args[2]), $(t.args[3])))

# the Julia code of an @hls body: array declarations become allocations, everything else stays
function _hls_julia(e)
    e isa Expr || return e
    if e.head === :(::) && length(e.args) == 2 && _isarrtype(e.args[2])           # z::Sig[4]
        return :($(e.args[1]) = $(_newarr(e.args[2])))
    elseif e.head === :(=) && e.args[1] isa Expr && e.args[1].head === :(::) && _isarrtype(e.args[1].args[2])
        v, t = e.args[1].args[1], e.args[1].args[2]                                  # v::Sig[3] = (a, b, c)
        (_isrom(e.args[2]) || _iskeep(e.args[2])) &&                                  # v::Cf[8, 40] = rom(:NAME) / keep(:NAME)
            return :($v = $(e.args[2])::Array{$(t.args[1]), $(length(t.args) - 1)})   # (typed: the table is concrete)
        vals = e.args[2] isa Expr && e.args[2].head in (:tuple, :vect) ? e.args[2].args : [e.args[2]]
        return :($v = $(t.args[1])[$(map(_hls_julia, vals)...)])
    end
    return Expr(e.head, map(_hls_julia, e.args)...)
end

"""
    @hls function f(...) ... end
    @hls struct S ... end

Defines `f` (or `S`) in Julia as written and records its source for `hls_write` (C++ for Vitis HLS). See the subset
at the top of src/rt/translator/julia_to_cpp.jl.
"""
macro hls(ex)
    ex isa Expr && ex.head in (:function, :struct) || error("@hls: expects a function or a struct")
    if ex.head === :struct
        name = ex.args[2] isa Symbol ? ex.args[2] : ex.args[2].args[1]
        body = Expr(:block, map(ex.args[3].args) do f
            f isa Expr && f.head === :(::) && _isarrtype(f.args[2]) ?
                Expr(:(::), f.args[1], length(f.args[2].args) == 2 ? :(Vector{$(f.args[2].args[1])}) :
                                                                     :(Matrix{$(f.args[2].args[1])})) : f
        end...)
        jl = Expr(:struct, ex.args[1], ex.args[2], body)
    else
        sig = ex.args[1]
        call = sig.head === :(::) ? sig.args[1] : sig
        name = call.args[1]
        args = map(call.args[2:end]) do a
            a isa Expr && a.head === :(::) && _isarrtype(a.args[2]) ? Expr(:(::), a.args[1], _arrjulia(a.args[2])) : a
        end
        newcall = Expr(:call, name, args...)
        jl = Expr(:function, sig.head === :(::) ? Expr(:(::), newcall, sig.args[2]) : newcall, _hls_julia(ex.args[2]))
    end
    return esc(quote
        $jl
        HLS_DEFS[$(QuoteNode(name))] = $(QuoteNode(ex))
    end)
end

# ------------------------------------------------------------------------------------------------ C++ writer
_cppname(s) = replace(string(s), "!" => "")
const _HLS_FUNS = Dict(:sin => "hls::sin", :cos => "hls::cos", :sqrt => "hls::sqrt", :abs => "hls::abs",
                       :exp => "hls::exp", :min => "hls::min", :max => "hls::max")
const _PREC = Dict(:|| => 1, :&& => 2, :(==) => 3, :!= => 3, :< => 4, :<= => 4, :> => 4, :>= => 4,
                   :+ => 5, :- => 5, :* => 6, :/ => 6)

mutable struct _CppCtx
    loops::Vector{Symbol}           # loop variables (0-based in C++)
    io::IO
    ind::Int
    line::Union{Nothing,LineNumberNode}
    used_line::Int
    srclines::Dict{String,Vector{String}}
end

# the comment written after the Julia code on a source line (for the C++ line)
function _comment(c::_CppCtx)
    ln = c.line
    (ln === nothing || ln.file === nothing || ln.line == c.used_line) && return ""
    f = string(ln.file); isfile(f) || return ""
    L = get!(() -> readlines(f), c.srclines, f)
    ln.line <= length(L) || return ""
    c.used_line = ln.line
    m = match(r"^[^#\"]*#\s?(.*)$", L[ln.line])
    return m === nothing ? "" : "  // " * strip(m[1])
end

# an index as a linear form of loop variables and constants: Dict(var => coef), const; nothing if not linear
function _linear(e)
    e isa Integer && return (Dict{Symbol,Int}(), Int(e))
    e isa Symbol && return (Dict(e => 1), 0)
    if e isa Expr && e.head === :call
        op = e.args[1]; xs = [_linear(a) for a in e.args[2:end]]
        any(isnothing, xs) && return nothing
        if op === :+ || (op === :- && length(xs) == 2)
            s = op === :- ? -1 : 1
            d = copy(xs[1][1]); c = xs[1][2]
            for (k, (dk, ck)) in enumerate(xs[2:end])
                for (v, a) in dk; d[v] = get(d, v, 0) + s * a; end
                c += s * ck
            end
            return (d, c)
        elseif op === :* && length(xs) == 2 && (isempty(xs[1][1]) || isempty(xs[2][1]))
            (k, (d, c)) = isempty(xs[1][1]) ? (xs[1][2], xs[2]) : (xs[2][2], xs[1])
            return (Dict(v => k * a for (v, a) in d), k * c)
        end
    end
    return nothing
end

# a 1-based Julia index -> the 0-based C++ index (loop variables are 0-based in C++)
function _cidx(e, c::_CppCtx)
    lin = _linear(e)
    if lin !== nothing
        d, k = lin
        k -= 1
        for (v, a) in d
            v in c.loops && (k += a)                       # v_julia = v_cpp + 1
        end
        terms = String[]
        for (v, a) in sort(collect(d); by = first)
            a == 0 && continue
            push!(terms, a == 1 ? string(v) : a == -1 ? "-$v" : "$a * $v")
        end
        isempty(terms) && return string(k)
        s = join(terms, " + ")
        return k == 0 ? s : k > 0 ? "$s + $k" : "$s - $(-k)"
    end
    return "$(_cexpr(e, c)) - 1"
end

function _cexpr(e, c::_CppCtx, prec::Int = 0)
    if e isa Symbol
        e === :pi && error("@hls: pi only as the value of a new variable (piv::Sig = pi)")
        return e in c.loops ? "($e + 1)" : string(e)
    elseif e isa Bool
        return e ? "true" : "false"
    elseif e isa Integer
        return string(e)
    elseif e isa AbstractFloat
        error("@hls: float literal $e in an expression; declare it as a variable (half::Sig = 0.5)")
    elseif e isa QuoteNode
        return string(e.value)
    elseif e isa Expr
        h = e.head
        if h === :ref
            length(e.args) == 1 && return _cexpr(e.args[1], c, 10)                  # v[]: an output port
            return _cexpr(e.args[1], c, 10) * join("[" * _cidx(i, c) * "]" for i in e.args[2:end])
        elseif h === :.
            return _cexpr(e.args[1], c, 10) * "." * string(e.args[2] isa QuoteNode ? e.args[2].value : e.args[2])
        elseif h in (:&&, :||)
            p = _PREC[h]
            s = join((_cexpr(a, c, p + 1) for a in e.args), " $(h) ")
            return p < prec ? "($s)" : s
        elseif h === :comparison
            parts = [string(_cexpr(e.args[i], c, 5), " ", e.args[i+1], " ", _cexpr(e.args[i+2], c, 5)) for i in 1:2:length(e.args)-2]
            s = join(parts, " && ")
            return 2 < prec ? "($s)" : s
        elseif h === :call
            f = e.args[1]; xs = e.args[2:end]
            if f in (:+, :-, :*, :/) && length(xs) >= 2
                p = _PREC[f]
                s = _cexpr(xs[1], c, p) * join(" $f " * _cexpr(x, c, p + 1) for x in xs[2:end])
                return p < prec ? "($s)" : s
            elseif f in (:-, :+) && length(xs) == 1
                return string(f, _cexpr(xs[1], c, 9))
            elseif f === :! && length(xs) == 1
                return "!" * _cexpr(xs[1], c, 9)
            elseif haskey(_PREC, f) && length(xs) == 2
                p = _PREC[f]
                s = "$(_cexpr(xs[1], c, p + 1)) $f $(_cexpr(xs[2], c, p + 1))"
                return p < prec ? "($s)" : s
            elseif f === :scale2 && length(xs) == 2 && xs[2] isa Integer             # exact x 2^k: wiring
                return "scale2<$(xs[2])>(" * _cexpr(xs[1], c) * ")"
            else
                fn = get(_HLS_FUNS, f, _cppname(f))
                return fn * "(" * join((_cexpr(x, c) for x in xs), ", ") * ")"
            end
        end
    end
    error("@hls: cannot translate the expression $e")
end

# a constant expression (numbers and pi only) -> its value, or nothing
function _constval(e)
    ok(x) = x isa Number || x === :pi || (x isa Expr && x.head === :call && x.args[1] in (:+, :-, :*, :/) && all(ok, x.args[2:end]))
    ok(e) || return nothing
    v = Core.eval(Base, e)
    return v
end
_numlit(v) = v isa Integer ? string(v) : string(Float64(v))

_out(c::_CppCtx, s::AbstractString) = println(c.io, " "^(4 * c.ind), s, _comment(c))

function _cstmt(e, c::_CppCtx)
    if e isa LineNumberNode
        c.line = e; return
    end
    e isa Expr || error("@hls: cannot translate the statement $e")
    h = e.head
    if h === :block
        foreach(x -> _cstmt(x, c), e.args)
    elseif h === :(::) && _isarrtype(e.args[2])                                     # z::Sig[4]
        t = e.args[2]
        _out(c, "$(_ctypename(t.args[1])) $(e.args[1])" * join("[$(n)]" for n in t.args[2:end]) * ";")
    elseif h === :(=)
        l, r = e.args
        if l isa Expr && l.head === :(::)
            v, t = l.args
            if _isarrtype(t) && _isrom(r)                                           # v::Cf[8, 40] = rom(:NAME)
                nm = r.args[2] isa QuoteNode ? r.args[2].value : r.args[2]
                _out(c, "static const $(_ctypename(t.args[1])) $v" * join("[$(n)]" for n in t.args[2:end]) * " = $nm;")
            elseif _isarrtype(t) && _iskeep(r)                                      # v::Tw[6048] = keep(:NAME)
                _out(c, "static $(_ctypename(t.args[1])) $v" * join("[$(n)]" for n in t.args[2:end]) * ";")
            elseif _isarrtype(t)                                                    # v::Sig[3] = (a, b, c)
                vals = r isa Expr && r.head in (:tuple, :vect) ? r.args : [r]
                _out(c, "$(t.args[1]) $v" * join("[$(n)]" for n in t.args[2:end]) * " = {" *
                        join((_cexpr(x, c) for x in vals), ", ") * "};")
            else                                                                    # v::Sig = expr
                cv = _constval(r)
                _out(c, "$(_ctypename(t)) $v = " * (cv === nothing ? _cexpr(r, c) : _numlit(cv)) * ";")
            end
        elseif l isa Expr && l.head === :tuple                                      # a::T, b::T = e1, e2
            r isa Expr && r.head === :tuple && length(r.args) == length(l.args) || error("@hls: tuple assignment needs a tuple of the same length")
            for (a, b) in zip(l.args, r.args); _cstmt(Expr(:(=), a, b), c); end
        else
            _out(c, "$(_cexpr(l, c)) = $(_cexpr(r, c));")
        end
    elseif h in (:+=, :-=, :*=, :/=)
        _out(c, "$(_cexpr(e.args[1], c)) $(h) $(_cexpr(e.args[2], c));")
    elseif h === :for
        spec = e.args[1]
        (spec.head === :(=) && spec.args[2] isa Expr && spec.args[2].head === :call && spec.args[2].args[1] === :(:)) ||
            error("@hls: for loops must be `for i in a:b`")
        v = spec.args[1]; a, b = spec.args[2].args[2], spec.args[2].args[3]
        lo = a isa Integer ? string(a - 1) : "$(_cexpr(a, c)) - 1"
        hi = b isa Integer ? string(b) : _cexpr(b, c)
        _out(c, "for (int $v = $lo; $v < $hi; $v++) {")
        push!(c.loops, v); c.ind += 1
        _cstmt(e.args[2], c)
        c.ind -= 1; pop!(c.loops)
        println(c.io, " "^(4 * c.ind), "}")
    elseif h in (:if, :elseif)
        cond = e.args[1] isa Expr && e.args[1].head === :block ? e.args[1].args[end] : e.args[1]
        _out(c, (h === :if ? "if (" : "} else if (") * _cexpr(cond, c) * ") {")
        c.ind += 1; _cstmt(e.args[2], c); c.ind -= 1
        if length(e.args) == 3
            if e.args[3] isa Expr && e.args[3].head === :elseif
                _cstmt(e.args[3], c)                                                # the chain's last branch
                h === :if && println(c.io, " "^(4 * c.ind), "}")                     # closes the whole chain
                return
            end
            println(c.io, " "^(4 * c.ind), "} else {")
            c.ind += 1; _cstmt(e.args[3], c); c.ind -= 1
        end
        h === :if && println(c.io, " "^(4 * c.ind), "}")
    elseif h === :return
        _out(c, isempty(e.args) || e.args[1] === nothing ? "return;" : "return $(_cexpr(e.args[1], c));")
    elseif h === :local
        foreach(x -> _cstmt(x, c), e.args)
    elseif h === :call                                                              # a call as a statement
        _out(c, _cexpr(e, c) * ";")
    elseif h === :macrocall && e.args[1] === Symbol("@pragma")
        _out(c, "#pragma " * e.args[end])
    else
        error("@hls: cannot translate the statement $e")
    end
end

_ctypename(t) = t === :Int ? "int" : t === :Bool ? "bool" : string(t)

# one argument of a C++ function
function _carg(a)
    a isa Expr && a.head === :(::) || error("@hls: every argument needs a type ($a)")
    v, t = a.args
    _isarrtype(t) && return "$(_ctypename(t.args[1])) $v" * join("[$(n)]" for n in t.args[2:end])
    # an output port: volatile, so that HLS keeps every write (without it HLS keeps only the last write of a loop:
    # the port would change once, at the end of the run)
    t isa Expr && t.head === :curly && t.args[1] === :Out && return "volatile $(_ctypename(t.args[2])) &$v"
    haskey(HLS_DEFS, t) && return "const $t &$v"                                    # an @hls struct
    return "$(_ctypename(t)) $v"
end

function _csignature(ex::Expr)
    sig = ex.args[1]
    call = sig.head === :(::) ? sig.args[1] : sig
    ret = sig.head === :(::) ? _ctypename(sig.args[2]) : "void"
    return "$ret $(_cppname(call.args[1]))(" * join(map(_carg, call.args[2:end]), ", ") * ")"
end

function _cfunction(io::IO, ex::Expr)
    c = _CppCtx(Symbol[], io, 1, nothing, 0, Dict{String,Vector{String}}())
    println(io, _csignature(ex), " {")
    _cstmt(ex.args[2], c)
    println(io, "}\n")
end

function _cstruct(io::IO, ex::Expr)
    name = ex.args[2] isa Symbol ? ex.args[2] : ex.args[2].args[1]
    c = _CppCtx(Symbol[], io, 1, nothing, 0, Dict{String,Vector{String}}())
    println(io, "struct $name {")
    for f in ex.args[3].args
        if f isa LineNumberNode
            c.line = f
        elseif f isa Expr && f.head === :(::)
            _out(c, _carg(f) * ";")
        end
    end
    println(io, "};\n")
end

"""
    hls_write(path, names; types, includes = ["ap_fixed.h", "hls_math.h"], header = "")

Writes the C++ of the @hls structs and functions `names` (in this order): `path` (`<model>.cpp`, the functions) and
beside it `<model>.h` (the includes, one typedef per entry of `types` (Julia type name => C++ type, e.g.
:Sig => "ap_fixed<48, 8>"), the structs and the function prototypes), which the .cpp and a testbench include.
"""
function hls_write(path::AbstractString, names; types, includes = ["ap_fixed.h", "hls_math.h"], header = "",
                   local_includes = String[])
    mkpath(dirname(path))
    hpath = splitext(path)[1] * ".h"
    banner(io, f) = (println(io, "// ", basename(f), ": written by PHPS.RT.hls_write from the Julia model (@hls); do not edit");
                     isempty(header) || foreach(l -> println(io, "// ", l), split(header, '\n')))
    open(hpath, "w") do io
        banner(io, hpath)
        println(io, "#pragma once")
        foreach(f -> println(io, "#include <$f>"), includes)
        println(io)
        for (k, v) in types
            println(io, "typedef $v $k;")
        end
        isempty(local_includes) || println(io)
        foreach(f -> println(io, "#include \"$f\""), local_includes)
        println(io)
        for n in names
            ex = HLS_DEFS[n]
            ex.head === :struct ? _cstruct(io, ex) : println(io, _csignature(ex), ";")
        end
    end
    open(path, "w") do io
        banner(io, path)
        println(io, "#include \"", basename(hpath), "\"\n")
        for n in names
            ex = HLS_DEFS[n]
            ex.head === :function && _cfunction(io, ex)
        end
    end
    println("written $hpath ($(countlines(hpath)) lines)")
    println("written $path ($(countlines(path)) lines)")
    return path
end

# a Julia value as a C++ initializer: numbers, vectors, matrices (row by row), structs (field by field)
_cinit(v::Bool) = v ? "true" : "false"
_cinit(v::Integer) = string(v)
_cinit(v::Real) = string(Float64(v))
_cinit(v::AbstractVector) = "{" * join(map(_cinit, v), ", ") * "}"
_cinit(v::AbstractMatrix) = "{" * join((_cinit(v[i, :]) for i in axes(v, 1)), ", ") * "}"
_cinit(v) = "{" * join((_cinit(getfield(v, f)) for f in fieldnames(typeof(v))), ", ") * "}"

"""
    hls_write_roms(path, names; header = "")

Writes `#define NAME {...}` for each registered table (hls_rom!) in `names`, row by row: the initialisers of the
`static const` tables that `v::T[...] = rom(:NAME)` declares in the C++.
"""
function hls_write_roms(path::AbstractString, names; header = "")
    mkpath(dirname(path))
    open(path, "w") do io
        println(io, "// ", basename(path), ": written by PHPS.RT.hls_write_roms; do not edit")
        isempty(header) || foreach(l -> println(io, "// ", l), split(header, '\n'))
        println(io, "#pragma once")
        for n in names
            println(io, "#define $n ", _cinit(HLS_ROMS[n]))
        end
    end
    println("written $path")
    return path
end

"""
    hls_write_const(path, name, value; header = "")

Writes `const <type> <name> = {...};` for the @hls struct value `value` to `path` (a header for the C++ model).
"""
function hls_write_const(path::AbstractString, name::AbstractString, value; header = "")
    mkpath(dirname(path))
    open(path, "w") do io
        println(io, "// ", basename(path), ": written by PHPS.RT.hls_write_const; do not edit")
        isempty(header) || foreach(l -> println(io, "// ", l), split(header, '\n'))
        println(io, "const $(nameof(typeof(value))) $name = ", _cinit(value), ";")
    end
    println("written $path")
    return path
end

"""
    flow_config(path; model, top, clock_ns, ip_version, ip_name, probes, sample_probe, plot, sample_dt, board_args)

Writes `path` (flow.cfg, shell variables) with a model's settings for the automated flow scripts/realtime/flow/flow.sh:
the HLS top function and clock, the IP version and name, the ILA probes (`port => width`), the probe whose rising edge
the ILA records, the probes to plot (`port => (fraction bits, label)`), the time between ILA samples, and the arguments
of the board program.
"""
function flow_config(path; model, top, clock_ns, ip_version, ip_name, probes, sample_probe, plot, sample_dt, board_args,
                     hls_extra = String[], impl_strategy = "", step_cycles = 0, ila_ref = "", ila_check = String[],
                     strict = false)
    open(path, "w") do io
        println(io, "# flow.cfg: written by the model's export (PHPS.RT); settings for scripts/realtime/flow/flow.sh")
        println(io, "MODEL=$model")
        println(io, "TOP=$top")
        println(io, "CLOCK_NS=$clock_ns")
        println(io, "FCLK_MHZ=$(1000 ÷ clock_ns)")
        println(io, "IP_VERSION=$ip_version")
        println(io, "IP_NAME=\"$ip_name\"")
        println(io, "PROBES=\"", join(("$n:$w" for (n, w) in probes), " "), "\"")
        println(io, "SAMPLE_PROBE=$sample_probe")
        println(io, "PLOT=\"", join(("$n:$F:$(replace(l, ' ' => '_'))" for (n, (F, l)) in plot), " "), "\"")
        println(io, "SAMPLE_DT=$sample_dt")
        println(io, "BOARD_ARGS=\"$board_args\"")
        println(io, "HLS_EXTRA=\"", join(hls_extra, ";"), "\"")       # extra lines for hls_config.cfg (flow.sh)
        println(io, "IMPL_STRATEGY=\"$impl_strategy\"")                # Vivado implementation strategy ("": default)
        println(io, "STEP_CYCLES=$step_cycles")                         # > 0: the every-clock ILA must show this per step
        println(io, "ILA_REF=\"$ila_ref\"")                             # Julia's values of every ILA sample (beside flow.cfg)
        println(io, "ILA_CHECK=\"", join(ila_check, " "), "\"")           # probe:fraction_bits:limit for ila_compare.py
        println(io, "STRICT=$(strict ? 1 : 0)")                         # 1: the board program's "board check: ... PASS" required
    end
    println("written $path")
    return path
end
