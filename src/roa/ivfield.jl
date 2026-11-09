# Interval evaluation of the controlled field and its Jacobian, beyond the steam lookups (ivsteam.jl): the practical-
# return proof, step P0 (docs/active/ROA_MATHEMATICAL_FORMULATION.md section 7). Methods for interval types only.

# ForwardDiff multiplies a dual by a Bool (complex products with im reach it) with signbit to choose the sign of a
# zero; on intervals that comparison is undecidable, and the sign of zero carries no meaning
@inline Base.:*(d::ForwardDiff.Dual{T,IA.Interval{Float64}}, x::Bool) where {T} = x ? d : zero(d)
@inline Base.:*(x::Bool, d::ForwardDiff.Dual{T,IA.Interval{Float64}}) where {T} = d * x
