# Candidate figures for the PHPS paper (review 09), drawn from stored results only (nothing is simulated here):
#   step_size.pdf      outputs/paper/step_size_study.json (scripts/paper/step_size_study.jl)
#   pf_angles_zoom.pdf outputs/ph/pf_compare_ph_fault_bus16_10s_controllers_series_all/traces.jld2
#                      (scripts/ph_pf_compare.jl ... --save-traces)
#   energy_account.pdf outputs/ph/full_model/fault_bus16_3ph_t0.3/run.jld2 (scripts/full_model_fault.jl --tstop=0.3)
#   ringing_modes.pdf  outputs/paper/ringing_modes.json (scripts/paper/ringing_modes.jl) and the traces above
# Run: julia --project=. scripts/paper/paper_figures.jl [outdir]  (default: outputs/paper/figures)
using CairoMakie, JSON3, JLD2, Printf
root = normpath(joinpath(@__DIR__, "..", ".."))
out = isempty(ARGS) ? joinpath(root, "outputs", "paper", "figures") : ARGS[1]
mkpath(out)
set_theme!(Theme(fontsize = 8, linewidth = 1.0, Axis = (; xgridvisible = false, ygridvisible = false,
                                                       xtickalign = 1, ytickalign = 1, spinewidth = 0.6,
                                                       xticksize = 3, yticksize = 3)))
const SINGLE, DOUBLE = 252, 516            # IEEE column and page widths in pt (pt_per_unit = 1)
cols = Makie.wong_colors()

# ---------------------------------------------------------------- 1. step size: accuracy and energy defect
let f = joinpath(root, "outputs", "paper", "step_size_study.json")
    if isfile(f)
        d = JSON3.read(read(f, String))
        rules = [("midpoint", "implicit midpoint", cols[1], :circle), ("trapezoidal", "trapezoidal", cols[2], :rect),
                 ("backward_euler", "backward Euler", cols[3], :utriangle)]
        fig = Figure(size = (DOUBLE, 205))
        hx = ([1.25, 2.5, 5, 10, 20], ["1.25", "2.5", "5", "10", "20"])
        a1 = Axis(fig[1, 1], xscale = log10, yscale = log10, xlabel = "step h [μs]", xticks = hx,
                  yticks = LogTicks(-6:-1), ylabel = "state error at 20 ms (scaled)", title = "accuracy")
        a2 = Axis(fig[1, 2], xscale = log10, yscale = log10, xlabel = "step h [μs]", xticks = hx,
                  yticks = LogTicks(-13:2:-3), ylabel = "|energy defect| / energy exchanged",
                  title = "discrete energy balance of each rule")
        for (key, lab, c, mk) in rules
            r = [x for x in d.rows if x.rule == key]
            h = [1e6 * x.h for x in r]
            scatterlines!(a1, h, [max(x.error_final, 1e-16) for x in r], color = c, marker = mk,
                          markersize = key == "midpoint" ? 9 : 5, strokewidth = key == "midpoint" ? 1 : 0,
                          strokecolor = c, markercolor = key == "midpoint" ? :white : c, label = lab)
            scatterlines!(a2, h, [max(x.defect_rel, 1e-17) for x in r], color = c, marker = mk, markersize = 5, label = lab)
            # the same rule with the powers at the step's midpoint (identical to the solid line for the midpoint rule)
            haskey(r[1], :defect_mid_rel) && key != "midpoint" &&
                lines!(a2, h, [max(x.defect_mid_rel, 1e-17) for x in r], color = c, linestyle = :dash)
        end
        lines!(a2, [NaN], [NaN], color = :gray40, linestyle = :dash, label = "powers at the midpoint")
        # slope guides
        hh = [1.25, 20.0]
        e0 = minimum(x.error_final for x in d.rows if x.rule == "midpoint")
        lines!(a1, hh, e0 .* (hh ./ hh[1]) .^ 2 ./ 3, color = :gray60, linestyle = :dash)
        text!(a1, 6.0, e0 * 16 / 3; text = "order 2", fontsize = 7, color = :gray40, align = (:left, :top))
        e1 = minimum(x.error_final for x in d.rows if x.rule == "backward_euler")
        lines!(a1, hh, e1 .* (hh ./ hh[1]) .* 3, color = :gray60, linestyle = :dot)
        text!(a1, 6.0, e1 * 3 * 4.8; text = "order 1", fontsize = 7, color = :gray40, align = (:left, :bottom))
        Legend(fig[2, 1:2], a2, orientation = :horizontal, framevisible = false, labelsize = 7, patchsize = (14, 6),
               tellheight = true, padding = (0, 0, 0, 0))
        rowgap!(fig.layout, 2)
        save(joinpath(out, "step_size.pdf"), fig, pt_per_unit = 1)
        println("step_size.pdf")
    else
        println("skipped step_size.pdf: no ", f)
    end
end

# ---------------------------------------------------------------- 2. PowerFactory comparison with zoomed views
let f = joinpath(root, "outputs", "ph", "pf_compare_ph_fault_bus16_10s_controllers_series_all", "traces.jld2")
    if isfile(f)
        d = load(f)
        t, ton, toff = d["t"], d["t_on"], d["t_off"]
        gs = [g for g in d["machines"] if g != d["ref"]]
        fig = Figure(size = (DOUBLE, 300))
        ax = Axis(fig[1, 1:3], xlabel = "time [s]", ylabel = "rotor angle rel. to G 01 [deg]",
                  title = "PowerFactory EMT (black) and PHPS (coloured, dashed)")
        for (k, g) in enumerate(gs)
            lines!(ax, t, d["pf"][g]["angle"], color = :black, linewidth = 0.9)
            lines!(ax, t, d["ph"][g]["angle"], color = cols[mod1(k, 7)], linestyle = :dash, linewidth = 0.9)
        end
        vspan!(ax, ton, toff, color = (:red, 0.12))
        # zoom 1: the fault and the first swing; zoom 2: the worst machine at its largest difference
        worst = gs[argmax([maximum(abs.(d["ph"][g]["angle"] .- d["pf"][g]["angle"])) for g in gs])]
        dif = d["ph"][worst]["angle"] .- d["pf"][worst]["angle"]
        iw = argmax(abs.(dif))
        z1 = findall(x -> 0.05 <= x <= 0.6, t)
        a1 = Axis(fig[2, 1], xlabel = "time [s]", ylabel = "[deg]", title = "fault and first swing")
        for (k, g) in enumerate(gs)
            lines!(a1, t[z1], d["pf"][g]["angle"][z1], color = :black, linewidth = 0.9)
            lines!(a1, t[z1], d["ph"][g]["angle"][z1], color = cols[mod1(k, 7)], linestyle = :dash, linewidth = 0.9)
        end
        vspan!(a1, ton, toff, color = (:red, 0.12))
        w = max(1, iw - 2000):min(length(t), iw + 2000)
        a2 = Axis(fig[2, 2], xlabel = "time [s]", ylabel = "[deg]", title = "$worst near its largest difference")
        lines!(a2, t[w], d["pf"][worst]["angle"][w], color = :black)
        lines!(a2, t[w], d["ph"][worst]["angle"][w], color = cols[mod1(findfirst(==(worst), gs), 7)], linestyle = :dash)
        a3 = Axis(fig[2, 3], xlabel = "time [s]", ylabel = "PHPS − PF [deg]", title = "difference, every machine")
        for (k, g) in enumerate(gs)
            lines!(a3, t, d["ph"][g]["angle"] .- d["pf"][g]["angle"], color = cols[mod1(k, 7)], linewidth = 0.7)
        end
        rowsize!(fig.layout, 1, Relative(0.52))
        save(joinpath(out, "pf_angles_zoom.pdf"), fig, pt_per_unit = 1)
        @printf("pf_angles_zoom.pdf (worst %s, largest difference %.4f deg at %.3f s)\n", worst, dif[iw], t[iw])
    else
        println("skipped pf_angles_zoom.pdf: no ", f)
    end
end

# ---------------------------------------------------------------- 3. the energy account through the fault
let f = joinpath(root, "outputs", "ph", "full_model", "fault_bus16_3ph_t0.3", "run.jld2")
    if isfile(f)
        d = load(f)
        t, E, kind = d["t"], d["energy"], d["kind"]
        groups = [("network", ["line", "transformer", "load"]), ("machine windings", ["machine"]),
                  ("exciter fields", ["exciter"]), ("hydro unit", ["hydro"])]
        fig = Figure(size = (SINGLE, 250))
        ax = Axis(fig[1, 1], ylabel = "ΔE [pu·s]", title = "energy of the device groups")
        axr = Axis(fig[1, 1], yaxisposition = :right, ylabel = "ΔE rotors [pu·s]", ylabelcolor = cols[3], yticklabelcolor = cols[3])
        hidespines!(axr); hidexdecorations!(axr)
        for (k, (lab, ks)) in enumerate(groups)
            c = findall(in(ks), kind)
            isempty(c) && continue
            y = vec(sum(E[c, :], dims = 1)); lines!(ax, t, y .- y[1], color = cols[k > 2 ? k + 1 : k], label = lab)
        end
        c = findall(==("rotor"), kind); y = vec(sum(E[c, :], dims = 1))
        lines!(axr, t, y .- y[1], color = cols[3], linewidth = 1.4)
        lines!(ax, [NaN], [NaN], color = cols[3], linewidth = 1.4, label = "rotors (right axis)")
        vspan!(ax, 0.1, 0.2, color = (:red, 0.10))
        axislegend(ax, position = :lt, framevisible = false, labelsize = 6.5, patchsize = (10, 5), rowgap = -2)
        tot = d["total"]
        res = tot["residual_t"]; thr = tot["throughput"]
        a2 = Axis(fig[2, 1], yscale = log10, xlabel = "time [s]", ylabel = "residual / exchanged",
                  title = "accumulated account of the whole system")
        lines!(a2, t, max.(abs.(res) ./ thr, 1e-18), color = :black)
        hlines!(a2, [tot["scheme_max"] / thr], color = :gray50, linestyle = :dash)
        vspan!(a2, 0.1, 0.2, color = (:red, 0.10))
        cl = abs(sum(d["closure"])) / thr
        text!(a2, 0.005, tot["scheme_max"] / thr; text = @sprintf("largest %.1e", tot["scheme_max"] / thr), fontsize = 6.5,
              align = (:left, :bottom), color = :gray35)
        text!(a2, 0.295, 2e-18; text = @sprintf("closure at the end %.1e", cl), fontsize = 6.5, align = (:right, :bottom))
        ylims!(a2, 1e-18, 1e-12)
        linkxaxes!(ax, a2); hidexdecorations!(ax, grid = false, ticks = false)
        save(joinpath(out, "energy_account.pdf"), fig, pt_per_unit = 1)
        println("energy_account.pdf")
    else
        println("skipped energy_account.pdf: no ", f)
    end
end

# ---------------------------------------------------------------- 4. work-precision: error and energy defect against time
let f = joinpath(root, "outputs", "paper", "work_precision.json"), g = joinpath(root, "outputs", "paper", "step_size_study.json")
    if isfile(f) && isfile(g)
        d = JSON3.read(read(f, String)); e = JSON3.read(read(g, String))
        defect(rule, h) = first(x.defect_rel for x in e.rows if x.rule == rule && abs(x.h - h) < 1e-12)
        sv = [("midpoint", "midpoint (fixed step)", cols[1], :circle), ("trapezoidal", "trapezoidal (fixed step)", cols[2], :rect),
              ("backward_euler", "backward Euler (fixed step)", cols[3], :utriangle),
              ("phs_simulate", "PHPS integrator with its account", cols[4], :diamond),
              ("cvode_bdf", "CVODE, variable-order BDF", cols[5], :xcross)]
        fig = Figure(size = (DOUBLE, 225))
        tx = ([0.05, 0.1, 0.2, 0.5, 1, 2], ["0.05", "0.1", "0.2", "0.5", "1", "2"])
        a1 = Axis(fig[1, 1], xscale = log10, yscale = log10, xlabel = "computing time [s]", ylabel = "state error at 20 ms (scaled)",
                  title = "accuracy against cost", yticks = LogTicks(-9:1:-1), xticks = tx)
        a2 = Axis(fig[1, 2], xscale = log10, yscale = log10, xlabel = "computing time [s]",
                  ylabel = "|energy defect| / energy exchanged", title = "energy balance against cost", yticks = LogTicks(-15:2:-1),
                  xticks = tx)
        # the reference's own error (midpoint at h/8: about 1/64 of the midpoint error at h)
        fl = minimum(x.error for x in d.rows if x.solver == "midpoint") / 64
        hlines!(a1, [fl], color = :gray55, linestyle = :dash, linewidth = 0.7)
        text!(a1, 2.2, fl * 1.25; text = "reference's own error", fontsize = 6.5, color = :gray40, align = (:right, :bottom))
        text!(a2, 2.2, 3e-8; text = "CVODE (BDF): no energy balance;\nits steps damp energy numerically", fontsize = 6.5,
              color = cols[5], align = (:right, :center))
        for (key, lab, c, mk) in sv
            r = sort([x for x in d.rows if x.solver == key], by = x -> x.wall_s)
            isempty(r) && continue
            scatterlines!(a1, [x.wall_s for x in r], [max(x.error, 1e-12) for x in r], color = c, marker = mk, markersize = 5, label = lab)
            if key in ("midpoint", "trapezoidal", "backward_euler")
                scatterlines!(a2, [x.wall_s for x in r], [max(defect(key, x.param), 1e-16) for x in r], color = c, marker = mk, markersize = 5)
            elseif key == "phs_simulate"
                scatterlines!(a2, [x.wall_s for x in r], [max(x.energy_defect_rel, 1e-16) for x in r], color = c, marker = mk, markersize = 5)
            end
        end
        Legend(fig[2, 1:2], a1, orientation = :horizontal, framevisible = false, labelsize = 7, patchsize = (12, 6),
               tellheight = true, nbanks = 2)
        rowgap!(fig.layout, 2)
        save(joinpath(out, "work_precision.pdf"), fig, pt_per_unit = 1)
        println("work_precision.pdf")
    else
        println("skipped work_precision.pdf: no ", f)
    end
end

# ---------------------------------------------------------------- 5. diagnostic resolution of the energy account
let f = joinpath(root, "outputs", "paper", "robustness_study.json")
    if isfile(f)
        d = JSON3.read(read(f, String))
        fig = Figure(size = (SINGLE, 200))
        ax = Axis(fig[1, 1], xscale = log10, yscale = log10, xlabel = "step h [μs]",
                  ylabel = "|account error| / energy exchanged", xticks = ([25, 50, 100, 200], ["25", "50", "100", "200"]),
                  yticks = LogTicks(-15:2:-3))
        for (key, lab, c, mk) in (("midpoint", "implicit midpoint (PHPS)", cols[1], :circle),
                                  ("trapezoidal", "trapezoidal (EMT rule)", cols[2], :rect),
                                  ("backward_euler", "backward Euler", cols[3], :utriangle))
            r = [x for x in d.rows if x.rule == key]
            scatterlines!(ax, [1e6 * x.h for x in r], [max(x.defect_rel, 1e-16) for x in r], color = c, marker = mk,
                          markersize = 5, label = lab)
        end
        hlines!(ax, [5.65e-6], color = :gray45, linestyle = :dash, linewidth = 0.8)
        text!(ax, 195, 5.65e-6 / 2.2; text = "hydro-ledger error found\nwith the account (VII-D)", fontsize = 6.5,
              color = :gray35, align = (:right, :top))
        ylims!(ax, 1e-16, 1e-3)
        axislegend(ax, position = :rc, framevisible = false, labelsize = 6.5, patchsize = (10, 5), rowgap = -2)
        save(joinpath(out, "account_resolution.pdf"), fig, pt_per_unit = 1)
        println("account_resolution.pdf")
    end
end

# ---------------------------------------------------------------- 6. "the same model": network ringing and eigenvalues
# Two events of the PowerFactory comparison case, one row each: the fault applied at bus 16 (the faulted network) and the
# fault removed (the network without the fault). Left: the bus-17 voltage magnitude, PowerFactory EMT and PHPS. Right:
# the frequencies of the ringing fitted by matrix pencil 0.2-12 ms after the event at four buses (outputs/paper/
# ringing_modes.json, scripts/paper/ringing_modes.jl), against the eigenfrequencies of the linearization of PHPS's own EMT
# model of that network (dq frame; the voltage magnitude does not depend on the frame). Only 0-4 kHz is shown: near the
# 5 kHz Nyquist limit of the 100 us traces the fits are aliased (the 4.40 kHz component at bus 18 after the removal is
# the 5.60 kHz eigenvalue folded) and are not used.
let f = joinpath(root, "outputs", "paper", "ringing_modes.json"),
    ft = joinpath(root, "outputs", "ph", "pf_compare_ph_fault_bus16_10s_controllers_series_all", "traces.jld2")
    if isfile(f) && isfile(ft)
        d = JSON3.read(read(f, String)); tr = load(ft)
        haskey(d, :ringing_fault) || error("rerun scripts/paper/ringing_modes.jl (no fault-on fits in $f)")
        t, ids = tr["t"], tr["bus_ids"]
        fmax = 4000.0
        fig = Figure(size = (DOUBLE, 330))
        buses = ["bus15", "bus16", "bus17", "bus18"]
        i17 = findfirst(==(17), ids)
        rows = ((tr["t_on"], "applied", d.ringing_fault, d.eigen_modes_fault, "faulted network"),
                (tr["t_off"], "removed", d.ringing, d.eigen_modes, "network without the fault"))
        local a2
        for (r, (t0, ev, rg, em, net)) in enumerate(rows)
            a1 = Axis(fig[r, 1], xlabel = r == 2 ? "time after the event [ms]" : "", ylabel = "|V| bus 17 [pu]",
                      title = "($(Char('a' + 2(r-1)))) fault $ev at bus 16")
            w = findall(x -> t0 - 1e-3 <= x <= t0 + 1.2e-2, t)
            lines!(a1, 1e3 .* (t[w] .- t0), tr["pfV"][i17, w], color = :gray30, linewidth = 1.6)
            lines!(a1, 1e3 .* (t[w] .- t0), tr["phV"][i17, w], color = cols[1], linewidth = 0.8, linestyle = :dash)
            a2 = Axis(fig[r, 2], xlabel = r == 2 ? "frequency [kHz]" : "",
                      title = "($(Char('b' + 2(r-1)))) fitted ringing and eigenvalues of the $net",
                      yticks = (0:length(buses), vcat(["eigenvalues"], ["bus " * b[4:end] for b in buses])),
                      limits = ((0, 4), (-0.6, length(buses) + 0.6)))
            ef = [m.f_hz for m in em if m.f_hz < fmax]
            vlines!(a2, ef ./ 1e3, color = (:gray70, 0.6), linewidth = 0.5)
            scatter!(a2, ef ./ 1e3, zeros(length(ef)), marker = :vline, markersize = 8, color = :black,
                     label = "eigenvalue of PHPS's linearization")
            for (k, b) in enumerate(buses)
                pf = [m.f for m in rg[b].powerfactory if m.f < fmax]
                ph = [m.f for m in rg[b].phps if m.f < fmax]
                scatter!(a2, pf ./ 1e3, fill(k + 0.12, length(pf)), marker = :circle, markersize = 6, color = :white,
                         strokecolor = :gray20, strokewidth = 0.8, label = k == 1 ? "fit, PowerFactory EMT" : nothing)
                scatter!(a2, ph ./ 1e3, fill(k - 0.12, length(ph)), marker = :xcross, markersize = 5, color = cols[1],
                         label = k == 1 ? "fit, PHPS EMT" : nothing)
            end
            if r == 2
                lines!(a2, [NaN], [NaN], color = :gray30, linewidth = 1.6, label = "|V|, PowerFactory EMT")
                lines!(a2, [NaN], [NaN], color = cols[1], linewidth = 0.8, linestyle = :dash, label = "|V|, PHPS")
            end
        end
        Legend(fig[3, 1:2], a2, orientation = :horizontal, framevisible = false, labelsize = 7, patchsize = (10, 6),
               tellheight = true, padding = (0, 0, 0, 0))
        colsize!(fig.layout, 1, Relative(0.36)); rowgap!(fig.layout, 4)
        save(joinpath(out, "ringing_modes.pdf"), fig, pt_per_unit = 1)
        println("ringing_modes.pdf")
    else
        println("skipped ringing_modes.pdf: needs ", f, " and ", ft)
    end
end
