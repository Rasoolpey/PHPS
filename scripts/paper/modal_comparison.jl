# "The same model" against PowerFactory's modal analysis (TPWRS paper): every eigenvalue of the linearization of PHPS's
# EMT model of the comparison case (IEEE 39-bus, exciters and governors in service; the model of ringing_modes.jl,
# 347 states) against PowerFactory's ComMod eigenvalues of the same case after RMS and after EMT initial conditions
# (pf/api/modal_ieee39_ringing.py -> outputs/pf/modal_ieee39/eig_rms.csv, eig_ins.csv), with the frequencies of the
# ringing that both EMT programs show after the bus-16 fault is applied and removed (outputs/paper/ringing_modes.json).
# (1) the electromechanical modes (0.1-3 Hz) of both programs side by side; (2) a map frequency (log) against damping
# ratio. Without the PowerFactory files, only PHPS is drawn and a notice is printed.
# Run: julia --project=. scripts/paper/modal_comparison.jl  ->  outputs/paper/modal_comparison.pdf, .json
using PHPS, LinearAlgebra, ForwardDiff, JSON3, Printf, CairoMakie
root = normpath(joinpath(@__DIR__, "..", ".."))
sc = load_scenario(joinpath(root, "cases", "IEEE39Bus_PF", "ph_fault_bus16_10s.json"))
eq = solve_equilibrium(load_case(sc.system_path), sc)
wc = JSON3.read(read(joinpath(root, "cases", "IEEE39Bus_FullModel", "winding_circuits.json"), String))
fl(v) = v === nothing ? nothing : Float64(v)
circuits = Dict(parse(Int, String(b)) => (; Ll = fl(v.Ll), Lad = fl(v.Lad), Laq = fl(v.Laq), Lfd = fl(v.Lfd), L1d = fl(v.L1d),
                L1q = fl(v.L1q), L2q = fl(v.L2q), Ra = fl(v.Ra), Rfd = fl(v.Rfd), R1d = fl(v.R1d), R1q = fl(v.R1q),
                R2q = fl(v.R2q), H = fl(v.H)) for (b, v) in pairs(wc.machines))
cs, x0, u, rep = phs_controlled(eq; loads = :series_all, circuits)
lam = eigvals(ForwardDiff.jacobian(x -> phs_rhs(cs, x, u), x0))
osc(l) = imag(l) > 1e-6
ph = [(f = imag(l) / 2pi, zeta = -real(l) / abs(l)) for l in lam if osc(l)]
@printf("PHPS: %d states, %d oscillatory modes, largest frequency %.0f Hz, %d with 100 Hz-5 kHz\n", cs.n, length(ph),
        maximum(m.f for m in ph), count(m -> 100 <= m.f <= 5000, ph))

# PowerFactory's ComMod eigenvalues, if present
pfdir = joinpath(root, "outputs", "pf", "modal_ieee39")
function readpf(sim)
    f = joinpath(pfdir, "eig_$sim.csv")
    isfile(f) || return nothing
    rows = [parse.(Float64, split(l, ',')) for l in readlines(f)[2:end] if !isempty(strip(l))]   # sigma, omega, f_hz, zeta
    return [(f = r[3], zeta = r[4]) for r in rows if r[2] > 1e-6]
end
pfr, pfe = readpf("rms"), readpf("ins")
pfr === nothing && println("no PowerFactory eigenvalues in $pfdir: run pf/api/modal_ieee39_ringing.py on the PowerFactory machine")
for (lab, p) in (("PowerFactory, RMS initial conditions", pfr), ("PowerFactory, EMT initial conditions", pfe))
    p === nothing && continue
    @printf("%s: %d oscillatory modes, largest frequency %.1f Hz, %d with 100 Hz-5 kHz\n", lab, length(p),
            maximum(m.f for m in p), count(m -> 100 <= m.f <= 5000, p))
end
# (1) electromechanical modes side by side
em(p) = sort([m for m in p if 0.1 <= m.f <= 3.0], by = m -> m.f)
if pfr !== nothing
    println("electromechanical modes (0.1-3 Hz), PowerFactory RMS -> nearest PHPS:")
    for m in em(pfr)
        n = ph[argmin([abs(q.f - m.f) for q in ph])]
        @printf("  %.3f Hz / %.3f  ->  %.3f Hz / %.3f\n", m.f, m.zeta, n.f, n.zeta)
    end
end
# ringing frequencies observed in both EMT programs (0.2-12 ms after the events, below 4 kHz)
rj = JSON3.read(read(joinpath(root, "outputs", "paper", "ringing_modes.json"), String))
obs = Float64[]
for key in (:ringing, :ringing_fault)
    haskey(rj, key) || continue
    for (b, v) in pairs(rj[key])
        b === :window_s && continue
        append!(obs, [m.f for m in v.powerfactory if 100 <= m.f <= 4000])
    end
end

# (2) the map
set_theme!(Theme(fontsize = 8, linewidth = 1.0, Axis = (; xgridvisible = false, ygridvisible = false, xtickalign = 1,
                                                       ytickalign = 1, spinewidth = 0.6, xticksize = 3, yticksize = 3)))
cols = Makie.wong_colors()
fig = Figure(size = (516, 210))
ax = Axis(fig[1, 1], xscale = log10, yscale = log10, xlabel = "frequency [Hz]", ylabel = "damping ratio",
          xticks = ([0.1, 1, 10, 100, 1000], ["0.1", "1", "10", "100", "1000"]),
          title = "Eigenvalues of the comparison case: PHPS (linearized EMT model) and PowerFactory (modal analysis)")
vlines!(ax, obs, color = (:gray75, 0.5), linewidth = 0.6)
lines!(ax, [NaN], [NaN], color = :gray70, label = "ringing in both EMT runs (fitted)")
z(m) = clamp(m.zeta, 1e-4, 1.0)
k = [m for m in ph if m.f >= 0.05]
scatter!(ax, [m.f for m in k], z.(k), color = cols[1], marker = :xcross, markersize = 5, label = "PHPS, EMT model")
pfr === nothing || scatter!(ax, [m.f for m in pfr if m.f >= 0.05], [z(m) for m in pfr if m.f >= 0.05], color = :white,
                            strokecolor = cols[2], strokewidth = 0.9, markersize = 6, label = "PowerFactory, RMS initial conditions")
pfe === nothing || scatter!(ax, [m.f for m in pfe if m.f >= 0.05], [z(m) for m in pfe if m.f >= 0.05], color = cols[3],
                            marker = :utriangle, markersize = 4, label = "PowerFactory, EMT initial conditions")
xlims!(ax, 0.05, 5000); ylims!(ax, 1e-4, 1.2)
Legend(fig[1, 2], ax, framevisible = false, labelsize = 7, patchsize = (10, 6))
o = joinpath(root, "outputs", "paper"); mkpath(o)
save(joinpath(o, "modal_comparison.pdf"), fig, pt_per_unit = 1)
open(joinpath(o, "modal_comparison.json"), "w") do io
    JSON3.pretty(io, Dict("phps" => ph, "pf_rms" => pfr, "pf_emt" => pfe, "observed_ringing_hz" => obs, "states" => cs.n))
end
println("written to ", joinpath(o, "modal_comparison.pdf"))
