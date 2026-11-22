# Listing 2 of the paper, second part (exactly as printed). Run in an environment with JLD2, JuMP and Hypatia,
# from the PHPS folder after listing2_export.jl: julia --project=@roa_sdp scripts/paper/listing2_jump.jl
using JLD2, JuMP, Hypatia, LinearAlgebra  # without
m = load("model.jld2"); nm = m["names"]
w = findall(i -> m["component"][i] == "GENROU_4" &&
            !endswith(nm[i], ".delta"), eachindex(nm))
A, Q = m["A"][w, w], Symmetric(m["Q"][w, w])
c0 = -maximum(real, eigvals(A))
model = Model(Hypatia.Optimizer)
@variable(model, P[1:7, 1:7], Symmetric)
@variable(model, κ)
@constraint(model, Symmetric(P - Q) in PSDCone())
@constraint(model, Symmetric(κ*Q - P) in PSDCone())
@constraint(model, Symmetric(-(A'*P + P*A) - c0*P)
                   in PSDCone())
@objective(model, Min, κ); optimize!(model)
println(termination_status(model), "  ", value(κ))   # 824.6
