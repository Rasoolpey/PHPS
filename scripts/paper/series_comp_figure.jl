# Figure of the series-compensation example from the stored results (outputs/paper/series_comp_mode.json, written by
# scripts/paper/series_comp_mode.jl; nothing is simulated here). (a) The subsynchronous network mode of the linearization
# against the compensation of Line 26-29: abc frequency and damping ratio. (b) The amplitude spectrum of the line's
# phase-a current after the fault is cleared (0.155-0.45 s, Hann window, evaluated directly at each frequency), with the
# abc frequency the eigenvalues predict at 50 % compensation.
# Run: julia --project=. scripts/paper/series_comp_figure.jl
using JSON3, CairoMakie, Printf
root = normpath(joinpath(@__DIR__, "..", ".."))
o = joinpath(root, "outputs", "paper")
d = JSON3.read(read(joinpath(o, "series_comp_mode.json"), String))
set_theme!(Theme(fontsize = 8, linewidth = 1.0, Axis = (; xgridvisible = false, ygridvisible = false, xtickalign = 1,
                                                       ytickalign = 1, spinewidth = 0.6, xticksize = 3, yticksize = 3)))
cols = Makie.wong_colors()
fig = Figure(size = (516, 200))
a1 = Axis(fig[1, 1], xlabel = "compensation xc / x of Line 26-29 [%]", ylabel = "abc frequency [Hz]",
          title = "(a) subsynchronous network mode (eigenvalues)")
ks = [100 * s.k for s in d.sweep]
scatterlines!(a1, ks, [s.f_abc for s in d.sweep], color = cols[1], markersize = 5)
a1r = Axis(fig[1, 1], yaxisposition = :right, ylabel = "damping ratio", ytickcolor = cols[2], yticklabelcolor = cols[2],
           ylabelcolor = cols[2])
hidespines!(a1r); hidexdecorations!(a1r)
scatterlines!(a1r, ks, [s.zeta_low for s in d.sweep], color = cols[2], marker = :rect, markersize = 4)
# (b) spectrum of the phase-a current after the clearing
t, ia = collect(Float64, d.run.t), collect(Float64, d.run.i_a)
k = findall(x -> 0.155 <= x <= 0.45, t)
tt, y = t[k], ia[k] .- sum(ia[k]) / length(k)
T = tt[end] - tt[1]
w = 0.5 .- 0.5 .* cos.(2pi .* (tt .- tt[1]) ./ T)
dtk = vcat(diff(tt), tt[end] - tt[end-1])                     # quadrature weights (the saved grid need not be uniform)
fs = range(1, 120, length = 1191)
amp = [2 * abs(sum(w .* y .* cis.(-2pi * f .* tt) .* dtk)) / sum(w .* dtk) for f in fs]
fpred = d.modes_50[1].f_abc
fpk = fs[argmax([f < 50 ? a : 0.0 for (f, a) in zip(fs, amp)])]
a2 = Axis(fig[1, 2], xlabel = "frequency [Hz]", ylabel = "amplitude [pu]", yscale = log10,
          title = "(b) Line 26-29 phase-a current after clearing")
lines!(a2, collect(fs), max.(amp, 1e-6), color = cols[1])
vlines!(a2, [fpred], color = :black, linestyle = :dash, linewidth = 0.8)
text!(a2, fpred - 2, 3e-4; text = @sprintf("eigenvalue\n%.1f Hz", fpred), fontsize = 7, align = (:right, :bottom))
text!(a2, 62, maximum(amp); text = "60 Hz", fontsize = 7, align = (:left, :top))
@printf("spectrum peak below 50 Hz: %.1f Hz; eigenvalue prediction %.2f Hz\n", fpk, fpred)
save(joinpath(o, "series_comp_mode.pdf"), fig, pt_per_unit = 1)
println("figure ", joinpath(o, "series_comp_mode.pdf"))
