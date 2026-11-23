# Numerical ringing after switching (TPWRS review): does the implicit midpoint rule of PHPS alternate sign step to step
# (the trapezoidal "chatter" of EMT programs) after a fault is applied or removed? Midpoint and trapezoidal share the
# stability function R(z) = (1 + z/2)/(1 - z/2), A-stable but not L-stable: a real stiff mode z = -h r with h r > 2 gives
# R < 0, so its deviation alternates sign and decays only by |R| per step (R -> -1 as h r -> inf). Backward Euler,
# R = 1/(1 - z), never alternates. PHPS's default step (`phs_step_rule`, h <= 1/r_max at each segment's start) keeps
# R(-h r_max) >= 1/3 > 0, so the question is what happens at a typical EMT step, which PHPS allows (`h = ...`).
# Model: the IEEE-39 grid model (cases/IEEE39Bus_PF, all machine windings and the network, fixed inputs, no
# controllers), a fault at bus 16 through Rf from t_on to t_off. Three one-step rules on the same vector field
# `phs_rhs`, the same Newton tolerance and steps (the pattern of scripts/paper/step_size_study.jl). Every step's efforts
# (grad H: bus voltages, branch/shunt/machine currents) are compared with a midpoint reference at a step that keeps
# R > 0 for every mode. The alternating part of the error, alt_k = (e_k - (e_{k-1} + e_{k+1})/2)/2 (exact for a
# (-1)^k component; it also picks up part of a resolved oscillation near the grid's Nyquist frequency, so it is reported
# beside the plain error), is measured in windows after the application and after the removal of the fault.
# Run: julia --project=. scripts/paper/numerical_ringing.jl [--R=0.001]  ->  outputs/paper/numerical_ringing.json, .pdf
using PHPS, LinearAlgebra, Printf, JSON3, ForwardDiff
root = normpath(joinpath(@__DIR__, "..", ".."))
opt(key, default) = (a = filter(x -> startswith(x, "--$key="), ARGS); isempty(a) ? default : split(a[1], "="; limit = 2)[2])
Rf = parse(Float64, opt("R", "0.001"))
sc_ = load_scenario(joinpath(root, "cases", "IEEE39Bus_PF", "no_fault_phtrue.json"))
eq = solve_equilibrium(load_case(sc_.system_path), sc_)
s, x0, u, _ = phs_equilibrium(eq)
n = length(x0)
names = phs_state_names(s)
kf = PHPS.bus_index(s.net.net, 16)
t_on, t_off, tstop = 0.001, 0.006, 0.009
uon = PHInput{Float64}(u.Tm, u.vf, (g = zeros(length(u.gf)); g[kf] = 1 / Rf; g), u.const_power, u.uc)
uoff = PHInput{Float64}(u.Tm, u.vf, zeros(length(u.gf)), u.const_power, u.uc)
sc = phs_scales(s, x0)
f(x, uu) = phs_rhs(s, x, uu)
A(x, uu) = ForwardDiff.jacobian(y -> phs_rhs(s, y, uu), x)
grad(x) = phs_gradient(s, x)
sl(x, uu) = (L = phs_ledger(s, x, uu); (sup = L.supply, loss = L.losses))

# stiffness of the two topologies and the per-step factor of the stiffest real modes for each rule
stiff = Dict{String,Any}()
for (lab, uu) in (("no fault", uoff), ("fault", uon))
    ev = eigvals(A(x0, uu))
    r = sort(abs.(real.(ev)), rev = true)
    rule = phs_step_rule(s, x0, uu)
    stiff[lab] = Dict("r_max" => r[1], "r_2" => r[2], "r_3" => r[3], "h_rule" => rule.h, "w_max" => rule.w_max)
end
Rmid(z) = (1 + z / 2) / (1 - z / 2)
Rbe(z) = 1 / (1 - z)

# integrate one rule at step h; returns the efforts at every step (columns), the midpoint-averaged efforts, the times,
# the accumulated energy defect of the rule's own balance and the energy exchanged
function integrate(rule::Symbol, h; tol = 1e-13, energy = true)
    x = copy(x0); t = 0.0
    nsteps = round(Int, tstop / h)
    Z = zeros(n, nsteps + 1); Z[:, 1] = grad(x)
    Zm = zeros(n, nsteps)                      # efforts at the step's midpoint state (x_k + x_{k+1})/2
    ts = collect(0:nsteps) .* h
    defect = 0.0; thr = 0.0; k = 0
    for (ta, tb, uu) in ((0.0, t_on, uoff), (t_on, t_off, uon), (t_off, tstop, uoff))
        nst = round(Int, (tb - ta) / h)
        abs(nst * h - (tb - ta)) < 1e-12 || error("step $h does not divide the segment")
        Jm = A(x, uu)
        F = lu(rule === :backward_euler ? I - h .* Jm : I - (h / 2) .* Jm)
        x1 = similar(x); r = similar(x)
        for _ in 1:nst
            f0 = rule === :trapezoidal ? f(x, uu) : nothing
            x1 .= x .+ h .* (rule === :trapezoidal ? f0 : f(x, uu))
            for it in 1:60
                if rule === :midpoint
                    r .= x1 .- x .- h .* f((x .+ x1) ./ 2, uu)
                elseif rule === :trapezoidal
                    r .= x1 .- x .- (h / 2) .* (f0 .+ f(x1, uu))
                else
                    r .= x1 .- x .- h .* f(x1, uu)
                end
                maximum(abs.(r) ./ sc) < tol && break
                it == 30 && (Jm = A(rule === :midpoint ? (x .+ x1) ./ 2 : x1, uu);
                             F = lu(rule === :backward_euler ? I - h .* Jm : I - (h / 2) .* Jm))
                it == 60 && error("$rule h = $h: Newton did not converge")
                x1 .-= F \ r
            end
            if energy
                dH = phs_energy(s, x1) - phs_energy(s, x)
                if rule === :midpoint
                    q = sl((x .+ x1) ./ 2, uu); defect += dH - h * (q.sup - q.loss); thr += h * (abs(q.sup) + q.loss)
                elseif rule === :trapezoidal
                    q0 = sl(x, uu); q1 = sl(x1, uu)
                    defect += dH - h * ((q0.sup - q0.loss) + (q1.sup - q1.loss)) / 2
                    thr += h * (abs(q0.sup) + q0.loss + abs(q1.sup) + q1.loss) / 2
                else
                    q = sl(x1, uu); defect += dH - h * (q.sup - q.loss); thr += h * (abs(q.sup) + q.loss)
                end
            end
            k += 1
            Zm[:, k] = grad((x .+ x1) ./ 2)
            x .= x1
            Z[:, k + 1] = grad(x)
        end
    end
    return (; Z, Zm, ts, defect, thr)
end

href = 0.3125e-6                                      # h r_max = 0.25 with the fault: R > 0.77 for every real mode
t0 = time(); ref = integrate(:midpoint, href; energy = false)
@printf("reference: midpoint, h = %.4g us, %.0f s\n", 1e6href, time() - t0)
zscale = vec(maximum(abs.(ref.Z), dims = 2)) .+ 1e-6  # each effort's largest value over the run (pu)

# windows after the events (relative to the event time) and a late window before the next event
windows = [("fault applied, 0-0.5 ms", t_on, t_on + 5e-4), ("fault applied, 2-5 ms", t_on + 2e-3, t_off),
           ("fault removed, 0-0.5 ms", t_off, t_off + 5e-4), ("fault removed, 1-3 ms", t_off + 1e-3, tstop)]
steps = [1.25e-6, 5e-6, 1e-5, 2.5e-5, 5e-5]
rows = Any[]
traces = Dict{String,Any}()
iv16 = findfirst(==("q_16_Re"), names)               # the faulted bus's voltage (real part, network frame)
for rule in (:midpoint, :trapezoidal, :backward_euler), h in steps
    t1 = time()
    a = integrate(rule, h)
    m = round(Int, h / href)
    E = a.Z .- ref.Z[:, 1:m:end]                       # error of the efforts at the rule's own instants
    alt = similar(E); alt .= 0
    alt[:, 2:end-1] = (E[:, 2:end-1] .- (E[:, 1:end-2] .+ E[:, 3:end]) ./ 2) ./ 2
    rel(M, cols) = maximum(abs.(M[:, cols]) ./ zscale)           # largest over efforts, relative to their peak
    wrow = Dict{String,Any}()
    for (lab, ta, tb) in windows
        cols = findall(t -> ta + 1.5h <= t <= tb - 1.5h, a.ts)   # interior instants (the detector needs neighbours)
        isempty(cols) && continue
        j = argmax(vec(maximum(abs.(alt[:, cols]) ./ zscale, dims = 2)))
        wrow[lab] = Dict("err" => rel(E, cols), "alt" => rel(alt, cols), "alt_state" => names[j],
                         "alt_v16" => maximum(abs.(alt[iv16, cols])), "err_v16" => maximum(abs.(E[iv16, cols])))
    end
    # midpoint-averaged efforts (what the midpoint rule evaluates the powers at) against the reference midway
    Em = a.Zm .- ref.Z[:, (1 + m ÷ 2):m:end][:, 1:size(a.Zm, 2)]
    cols = findall(t -> t_on <= t <= t_on + 5e-4, a.ts[1:end-1] .+ h / 2)
    errmid = m % 2 == 0 ? rel(Em, cols) : NaN
    z = -h * stiff["fault"]["r_max"]; z0 = -h * stiff["no fault"]["r_max"]
    Rf_ = rule === :backward_euler ? Rbe(z) : Rmid(z); R0 = rule === :backward_euler ? Rbe(z0) : Rmid(z0)
    push!(rows, Dict("rule" => String(rule), "h" => h, "R_fault" => Rf_, "R_nofault" => R0, "windows" => wrow,
                     "err_midpoint_avg_after_fault" => errmid, "energy_defect_rel" => abs(a.defect) / a.thr))
    if h in (5e-5, 1.25e-6)
        traces["$(rule)_$(round(Int, 1e6h))us"] = Dict("t" => a.ts, "v16re" => a.Z[iv16, :])
    end
    w1 = wrow["fault applied, 0-0.5 ms"]; w2 = wrow["fault applied, 2-5 ms"]; w3 = wrow["fault removed, 0-0.5 ms"]
    @printf("%-14s h = %5.2f us: R(fault) %+.3f R(no fault) %+.3f | applied: err %.2e alt %.2e (%s), 2-5 ms alt %.2e | removed: err %.2e alt %.2e | energy %.1e | %.0f s\n",
            rule, 1e6h, Rf_, R0, w1["err"], w1["alt"], w1["alt_state"], w2["alt"], w3["err"], w3["alt"],
            abs(a.defect) / a.thr, time() - t1)
end
traces["reference"] = Dict("t" => ref.ts[1:8:end], "v16re" => ref.Z[iv16, 1:8:end])

out = joinpath(root, "outputs", "paper"); mkpath(out)
open(joinpath(out, "numerical_ringing.json"), "w") do io
    JSON3.pretty(io, Dict("model" => "IEEE39Bus_PF grid model (no controllers)", "states" => n, "fault_bus" => 16,
                          "fault_R" => Rf, "t_on" => t_on, "t_off" => t_off, "tstop" => tstop, "h_ref" => href,
                          "stiffness" => stiff, "rows" => rows, "traces" => traces))
end
println("written to ", joinpath(out, "numerical_ringing.json"))

# diagnostic figure: the faulted bus's voltage (real part) right after the fault is applied, at 50 us
using CairoMakie
fig = Figure(size = (520, 200))
for (j, (ta, lab)) in enumerate(((t_on, "fault applied"), (t_off, "fault removed")))
    ax = Axis(fig[1, j], xlabel = "time after the event [μs]", ylabel = "v₁₆ (Re) [pu]", title = lab)
    r = traces["reference"]; k = findall(t -> ta - 1e-4 <= t <= ta + 6e-4, r["t"])
    lines!(ax, 1e6 .* (r["t"][k] .- ta), r["v16re"][k], color = :gray30, linewidth = 2, label = "reference")
    for (key, c, lb) in (("midpoint_50us", :dodgerblue, "midpoint 50 μs"), ("trapezoidal_50us", :orange, "trapezoidal 50 μs"),
                         ("backward_euler_50us", :green, "backward Euler 50 μs"))
        d = traces[key]; k = findall(t -> ta - 1e-4 <= t <= ta + 6e-4, d["t"])
        scatterlines!(ax, 1e6 .* (d["t"][k] .- ta), d["v16re"][k], color = c, markersize = 4, label = lb)
    end
    j == 2 && axislegend(ax, position = :rb, framevisible = false, labelsize = 7)
end
save(joinpath(out, "numerical_ringing.pdf"), fig)
println("figure ", joinpath(out, "numerical_ringing.pdf"))
