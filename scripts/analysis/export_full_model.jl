# Export of the IEEE 39-bus full model for analysis tools (docs/analysis.md), and a first analysis that
# uses nothing but the exported file's content: the oscillatory modes with the devices that take part in each.
# Run: julia --project=. scripts/analysis/export_full_model.jl      (writes outputs/analysis/ieee39_full_model.jld2 and
#      the CSV folder outputs/analysis/ieee39_full_model_csv/)
using PHPS, LinearAlgebra, Printf
root = normpath(joinpath(@__DIR__, "..", ".."))
out = joinpath(root, "outputs", "analysis"); mkpath(out)
t0 = time(); fm = phs_full_model(joinpath(root, "cases", "IEEE39Bus_FullModel")); t1 = time()
am = phs_analysis_model(fm.cs, fm.x0, fm.u); t2 = time()
phs_export(joinpath(out, "ieee39_full_model.jld2"), am); phs_export(joinpath(out, "ieee39_full_model_csv"), am); t3 = time()
N = length(am.x0)
@printf("model built in %.0f s; analysis model in %.0f s; files written in %.0f s\n", t1 - t0, t2 - t1, t3 - t2)
@printf("%d states: %s\n", N, join(["$(count(==(k), am.kind)) $(k)" for k in unique(am.kind)], ", "))
@printf("%d rows in the port-Hamiltonian form, %d outside (%s)\n", count(am.in_form), N - count(am.in_form), join(am.outside, "; "))
println("checks of the export:"); for (k, v) in pairs(am.checks); @printf("  %-16s %.2e\n", k, v); end

# modes of the linearisation, with the devices that take part (participation of state i in mode k: |v_ik w_ik|)
F = eigen(am.A); W = inv(F.vectors)
osc = [k for k in eachindex(F.values) if imag(F.values[k]) > 2pi * 0.1 && imag(F.values[k]) < 2pi * 3]
println("\noscillatory modes between 0.1 and 3 Hz (frequency, damping ratio, the three devices that take part most):")
comps = unique(am.component)
for k in sort(osc; by = k -> imag(F.values[k]))
    lam = F.values[k]; p = abs.(F.vectors[:, k] .* W[k, :]); p ./= sum(p)
    pc = [sum(p[am.component .== c]) for c in comps]; o = sortperm(pc; rev = true)[1:3]
    @printf("  %.3f Hz, %5.1f %%: %s\n", imag(lam) / 2pi, -100 * real(lam) / abs(lam), join(["$(comps[i]) $(round(100 * pc[i]; digits = 0))%" for i in o], ", "))
end
@printf("\nslowest mode %.4f /s; modes with positive real part: %d; zero modes (|lambda| < 1e-6): %d\n", maximum(real, F.values[abs.(F.values) .> 1e-6]),
        count(real.(F.values) .> 1e-6), count(abs.(F.values) .<= 1e-6))
