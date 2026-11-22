# Candidate figure for the PHPS paper (review 09): the exported structure of the full physical model and one generating
# unit cut out of it. Left: the sparsity of J - R over the 519 states, ordered as in the model, with the device kind of
# every state along the axes; the unit's rows and its neighbour columns marked. Right: the unit's rows alone, its own
# block (J skew, R) and its port columns G_p = (J - R)_SN. Drawn from the stored export (nothing is simulated).
# Run: julia --project=. scripts/paper/export_structure_fig.jl [outdir]
using CairoMakie, JLD2, Printf
root = normpath(joinpath(@__DIR__, "..", ".."))
out = isempty(ARGS) ? joinpath(root, "outputs", "paper", "figures") : ARGS[1]
mkpath(out)
set_theme!(Theme(fontsize = 8, Axis = (; xgridvisible = false, ygridvisible = false, spinewidth = 0.6,
                                        xticksize = 2, yticksize = 2)))
m = load(joinpath(root, "outputs", "analysis", "ieee39_full_model.jld2"))
J, R, names, comp, kind = m["J"], m["R"], m["names"], m["component"], m["kind"]
n = size(J, 1); M = J .- R
unit = ["GENROU_4", "IEEET1_4", "IEEEG1_4"]
S = findall(in(unit), comp)
N = [k for k in 1:n if !(k in S) && any(!iszero, view(M, S, k))]
kinds = ["network", "machine", "excitation", "prime_mover"]
klab = Dict("network" => "network", "machine" => "machines", "excitation" => "exciters", "prime_mover" => "prime movers")
kcol = Dict("network" => RGBf(0.0, 0.49, 0.49), "machine" => RGBf(0.25, 0.25, 0.25),
            "excitation" => RGBf(0.27, 0.43, 0.16), "prime_mover" => RGBf(0.77, 0.12, 0.12))
Jc, Rc, Pc = RGBf(0.10, 0.35, 0.75), RGBf(0.55, 0.16, 0.55), RGBf(0.0, 0.49, 0.49)

fig = Figure(size = (516, 235), figure_padding = (2, 6, 2, 2))
# ---- left: the whole model
ax = Axis(fig[1, 1], aspect = DataAspect(), yreversed = true, title = @sprintf("J − R of the full model, %d states", n),
          xlabel = "column (effort)", ylabel = "row (state)", xticks = [1, 100, 200, 300, 400, 519],
          yticks = [1, 100, 200, 300, 400, 519])
# the kind of every state along the axes
for k in 1:n
    c = kcol[kind[k]]
    poly!(ax, Rect(k - 0.5, -9, 1, 7), color = c, strokewidth = 0)
    poly!(ax, Rect(-9, k - 0.5, 7, 1), color = c, strokewidth = 0)
end
# the unit's rows and its neighbour columns
for k in S; poly!(ax, Rect(0.5, k - 0.5, n, 1), color = (:orange, 0.18), strokewidth = 0); end
for k in N; poly!(ax, Rect(k - 0.5, 0.5, 1, n), color = (Pc, 0.25), strokewidth = 0); end
iJ = [(j, i) for i in 1:n, j in 1:n if J[i, j] != 0 && R[i, j] == 0]
iR = [(j, i) for i in 1:n, j in 1:n if R[i, j] != 0]
scatter!(ax, Point2f.(iJ), color = Jc, markersize = 1.3, strokewidth = 0)
scatter!(ax, Point2f.(iR), color = Rc, markersize = 1.3, strokewidth = 0)
xlims!(ax, -10, n + 1); ylims!(ax, n + 1, -10)
# ---- right: the unit as a system of its own
cols_ = vcat(S, N)
ax2 = Axis(fig[1, 2], yreversed = true, aspect = DataAspect(),
           title = @sprintf("unit 4 cut out: %d states, %d ports", length(S), length(N)),
           xlabel = "effort", ylabel = "state of the unit")
for (a, i) in enumerate(S), (b, j) in enumerate(cols_)
    v = M[i, j]; v == 0 && continue
    c = j in N ? Pc : (R[i, j] != 0 ? Rc : Jc)
    poly!(ax2, Rect(b - 0.5, a - 0.5, 1, 1), color = c, strokewidth = 0)
end
vlines!(ax2, [length(S) + 0.5], color = :black, linewidth = 0.6)
# the devices of the unit along the rows
# the devices of the unit along the rows, the two blocks along the columns
yt, yl = Float64[], String[]
let off = 0
    for d in unit
        k = count(==(d), comp[S])
        push!(yt, off + k / 2 + 0.5); push!(yl, Dict("GENROU" => "machine", "IEEET1" => "DC exciter", "IEEEG1" => "steam unit")[split(d, "_")[1]])
        off + k < length(S) && hlines!(ax2, [off + k + 0.5], color = :gray70, linewidth = 0.4)
        off += k
    end
end
ax2.yticks = (yt, yl); ax2.yticklabelsize = 6.5; ax2.yticksize = 0
ax2.xticks = ([length(S) / 2 + 0.5, length(S) + length(N) / 2 + 0.5], ["own block: J (skew), R", "G_p"])
ax2.xticklabelsize = 6.5; ax2.xticksize = 0
xlims!(ax2, 0.5, length(cols_) + 0.5); ylims!(ax2, length(S) + 0.5, 0.5)
ax2.ylabel = ""; ax2.xlabel = ""
# legend
Legend(fig[2, 1:2], vcat([PolyElement(color = kcol[k]) for k in kinds],
                         [MarkerElement(marker = :rect, color = Jc), MarkerElement(marker = :rect, color = Rc),
                          PolyElement(color = (Pc, 0.6)), PolyElement(color = (:orange, 0.35))]),
       vcat([klab[k] for k in kinds], ["J only", "R (with or without J)", "ports of the unit", "rows of the unit"]),
       orientation = :horizontal, framevisible = false, labelsize = 6.5, patchsize = (7, 7), nbanks = 2, colgap = 6,
       tellheight = true)
colsize!(fig.layout, 2, Relative(0.40)); rowgap!(fig.layout, 2)
save(joinpath(out, "export_structure.pdf"), fig, pt_per_unit = 1)
@printf("export_structure.pdf: %d states, unit rows %d, ports %d (%s)\n", n, length(S), length(N), join(names[N], ", "))
