# =====================================================================================================================
# TEMPLATE: a component model for the FPGA, written in Julia and translated to C++ (Vitis HLS)
# =====================================================================================================================
# This file is a guide to copy. It is NOT loaded by the package. Complete example: machines/genrou.jl.
#
# HOW TO START A NEW MODEL
#   1. Copy this file to src/rt/<class>/<model>.jl (same class and file name as in src/components/).
#   2. Replace `mymodel` / `MyModel` by the model's name.
#   3. Write sections 1-4, keeping the equations of src/components/<class>/<model>.jl.
#   4. Add `include("<class>/<model>.jl")` to src/rt/RT.jl (the model file itself has no include or using lines).
#   5. Export: `julia --project=. scripts/realtime/export_cpp.jl mymodel` writes src/rt/<class>/CPP/<model>/.
#
# HOW IT WORKS
#   Functions and structs written with @hls run in Julia as written, and are translated line by line to C++: the same
#   statements, loops, names and comments (src/rt/translator/julia_to_cpp.jl). What an @hls function may use:
#     types      Sig (signals) and Small (small coefficients); the C++ types are chosen at export (section 4)
#     arguments  x::Sig (a number), x::Sig[6] (an array, changed in place), P::Small[4, 4], k::MyModelCoefs
#     locals     v::Sig = expr  (a new variable, always with its type);  z::Sig[4];  v::Sig[3] = (a, b, c)
#                v = expr,  x[i] = expr,  v += expr
#     control    for i in 1:4 ... end,  if / elseif / else,  return
#     maths      + - * /, comparisons, &&, ||, !, sin, cos, sqrt, abs, exp, integer literals
#     constants  a float or pi only as the value of a new variable: half::Sig = 0.5, twopi::Sig = 2pi
#     pragmas    @pragma "HLS ARRAY_PARTITION variable=x complete" (an HLS instruction; nothing in Julia)
#   Rules for a fast step (scripts/realtime/README.md section 2.4, learned on GENROU):
#     1. no division by a signal: Newton steps (r = r (2 - x r)) from a seed; division by a constant is a coefficient
#     2. no sin/cos units: carry the cos and sin of a slowly changing angle as states and rotate them each step
#     3. give every product that feeds another multiplication its own variable (t::Sig = a * b): C++ keeps the full
#        width of a * b in a * b * c, which makes 96 x 48-bit multipliers
#     4. partition the small arrays: @pragma "HLS ARRAY_PARTITION variable=x complete"
#     5. unroll small loops with independent iterations: @pragma "HLS UNROLL" inside the loop
#     6. round to nearest: types ap_fixed<W, I, AP_RND_CONV>; AMD's default (truncation) makes the states drift
#   Coefficients (matrices, 1/T, ...) are computed once on the PC in section 2 and passed in a struct (section 3):
#   the step does only the per-step arithmetic.
# =====================================================================================================================


# ---------------------------------------------------------------------------------------------------------------------
# 1. The model (in words and equations)
# ---------------------------------------------------------------------------------------------------------------------
# Source:    src/components/<class>/<model>.jl (and its reference: TechRef file, pages, equation numbers)
# States x:  names, units                      Inputs u: names, units
# Equations: x' = f(x, u)
# Step:      h = 1 us; the discretisation (e.g. implicit midpoint for the linear part, as genrou.jl)


# ---------------------------------------------------------------------------------------------------------------------
# 2. Parameters and coefficients (on the PC; BigFloat)
# ---------------------------------------------------------------------------------------------------------------------
"""
    mymodel_rt_params(; a, b, ...) -> NamedTuple

The parameters in BigFloat; pass decimal strings (`a = "0.05"`) so they are parsed exactly.
"""
function mymodel_rt_params(; kw...)
    p(v) = v isa AbstractString ? parse(BigFloat, v) : BigFloat(v)
    return NamedTuple{keys(kw)}(map(p, values(kw)))
end


# ---------------------------------------------------------------------------------------------------------------------
# 3. The coefficient struct and THE STEP FUNCTION (these become the C++)
# ---------------------------------------------------------------------------------------------------------------------
@hls struct MyModelCoefs
    A::Small[2, 2]      # example: a step matrix
    b::Sig              # example: a gain
end

# x = (...), u = (...); one step of h, x changed in place
@hls function mymodel_step!(x::Sig[2], u::Sig[1], k::MyModelCoefs)
    z::Sig[2]
    for i in 1:2
        d::Sig = 0
        for j in 1:2
            d += k.A[i, j] * x[j]
        end
        z[i] = x[i] + d + k.b * u[1]
    end
    for i in 1:2
        x[i] = z[i]
    end
end


# ---------------------------------------------------------------------------------------------------------------------
# 4. Export (C++ types, the coefficient values)
# ---------------------------------------------------------------------------------------------------------------------
const MYMODEL_RT_TYPES = [:Sig => "ap_fixed<48, 8, AP_RND_CONV>",       # signals (round to nearest)
                          :Small => "ap_fixed<48, -10, AP_RND_CONV>"]   # small coefficients

"Writes src/rt/<class>/CPP/mymodel/mymodel.cpp and mymodel_coefs.h."
function mymodel_rt_export(; dir = joinpath(@__DIR__, "CPP", "mymodel"))
    hls_write(joinpath(dir, "mymodel.cpp"), [:MyModelCoefs, :mymodel_step!]; types = MYMODEL_RT_TYPES)
    k = MyModelCoefs([0.0 0.0; 0.0 0.0], 0.0)              # the coefficients from section 2
    hls_write_const(joinpath(dir, "mymodel_coefs.h"), "MYMODEL_COEFS", k)
end
