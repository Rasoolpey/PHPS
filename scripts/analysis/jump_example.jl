# An optimisation tool working on the exported model alone (docs/analysis.md). This script does not load
# the simulator: it reads outputs/analysis/ieee39_full_model.jld2 (written by export_full_model.jl) and uses JuMP.
# Example question (an illustration of the hand-over, not a study): for the windings and the rotor of one machine
# (stator and rotor flux linkages and the momentum; the rotor angle and every other state held), find the weights
# P = P' closest in shape to the stored energy (smallest kappa with Q_ww <= P <= kappa Q_ww) for which the weighted
# energy decays at c0 or faster:  A_ww' P + P A_ww <= -c0 P.  Q_ww is the stored-energy Hessian of those states.
# States are picked by name from the file.
# Run (an environment with JuMP, Hypatia and JLD2): julia --project=@roa_sdp scripts/analysis/jump_example.jl [machine [c0]]
using JLD2, JuMP, Hypatia, LinearAlgebra, Printf
root = normpath(joinpath(@__DIR__, "..", ".."))
m = load(joinpath(root, "outputs", "analysis", "ieee39_full_model.jld2"))
unit = isempty(ARGS) ? "GENROU_4" : ARGS[1]
w = [i for i in eachindex(m["names"]) if m["component"][i] == unit && !endswith(m["names"][i], ".delta")]
isempty(w) && error("no states found for $unit; devices in the file: " * join(unique(m["component"][m["kind"] .== "machine"]), ", "))
haskey(m, "Q") || error("the file has no Q: export with hessian = true")
A = m["A"][w, w]; Q = Symmetric((m["Q"][w, w] + m["Q"][w, w]') / 2); n = length(w)
eigmin(Q) > 0 || error("the stored energy does not see every chosen state (smallest eigenvalue of Q: $(eigmin(Q)))")
maximum(real, eigvals(A)) < 0 || error("this block alone is not stable; no weights exist")
@printf("%s: %d states (%s); slowest decay of this block alone %.3f /s\n", unit, n, join(last.(split.(m["names"][w], ".")), ", "), -maximum(real, eigvals(A)))
c0 = length(ARGS) > 1 ? parse(Float64, ARGS[2]) : 0.5 * (-2 * maximum(real, eigvals(A)))     # half of the fastest rate any weights can guarantee
model = Model(Hypatia.Optimizer); set_silent(model)
@variable(model, P[1:n, 1:n], Symmetric); @variable(model, kappa)
@constraint(model, Symmetric(P - Q) in PSDCone()); @constraint(model, Symmetric(kappa .* Q - P) in PSDCone())
@constraint(model, Symmetric(-(A' * P + P * A) - c0 .* P) in PSDCone())
@objective(model, Min, kappa); optimize!(model)
termination_status(model) in (MOI.OPTIMAL, MOI.ALMOST_OPTIMAL) || error("the solver did not return a solution: $(termination_status(model))")
Pv = Symmetric(value.(P)); Ph = sqrt(Pv); Qh = sqrt(Q)
@printf("solver: %s; kappa = %.3f (1 would mean the stored energy itself decays at that rate)\n", termination_status(model), value(kappa))
@printf("checks without the solver: decay rate of P %.3f /s (asked: %.3f); P between %.3f and %.3f of the stored energy; the stored energy alone: %.3f /s\n",
        -eigmax(Symmetric(Ph \ (A' * Pv + Pv * A) / Ph)), c0, eigmin(Symmetric(Qh \ Pv / Qh)), eigmax(Symmetric(Qh \ Pv / Qh)), -eigmax(Symmetric(Qh \ (A' * Q + Q * A) / Qh)))
