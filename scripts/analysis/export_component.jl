# Any device, or a group of devices, exported as its own port-Hamiltonian system (docs/analysis.md).
# Run: julia --project=. scripts/analysis/export_component.jl [device ...]
#      default: the machine GENROU_4 alone, then the whole unit (machine, exciter IEEET1_4, steam unit IEEEG1_4); writes outputs/analysis/<names>/ (CSV files
#      and equations.md) and prints the checks.
using PHPS, LinearAlgebra, Printf
root = normpath(joinpath(@__DIR__, "..", ".."))
out = joinpath(root, "outputs", "analysis")
am = phs_analysis_model(joinpath(root, "cases", "IEEE39Bus_FullModel"))
groups = isempty(ARGS) ? [["GENROU_4"], ["GENROU_4", "IEEET1_4", "IEEEG1_4"]] : [String.(ARGS)]
for g in groups
    t0 = time(); sub = phs_subsystem(am, g); dir = joinpath(out, join(g, "+"))
    phs_export(dir, sub); phs_export(dir * ".jld2", sub)
    @printf("\n%s: %d states (%d in the form), %d ports, %d inputs; %.2f s; written to %s\n", join(g, " + "), length(sub.names), count(sub.in_form),
            length(sub.ports), length(sub.sources), time() - t0, relpath(dir, root))
    for (k, v) in pairs(sub.checks); @printf("  %-17s %.2e\n", k, v); end
    @printf("  entries of [J - R, Gp] that change with the group's state: %d of %d non-zero\n", count(sub.varies), count(!iszero, hcat(sub.J - sub.R, sub.Gp)))
    isempty(sub.shared_loss_with) ? println("  every port conserves power") : println("  shares loss terms with: ", join(sub.shared_loss_with, ", "), " (include them for power-conserving ports)")
end
println("\nfirst lines of the equations of the first group:\n"); foreach(println, Iterators.take(eachline(joinpath(out, join(groups[1], "+"), "equations.md")), 30))
