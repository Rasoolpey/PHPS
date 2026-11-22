# Step-size study for the PHPS paper (review 09, figure "energy error and accuracy against the step"): the IEEE-39 grid
# model (cases/IEEE39Bus_PF, machines with all windings and the network, fixed inputs, no controllers), a three-phase
# fault at bus 16 through `R` pu from 5 ms to 10 ms, 20 ms simulated, integrated with three one-step implicit rules
# written here on the same vector field `phs_rhs`, the same Newton tolerance and the same steps:
#   midpoint      x1 - x0 = h f((x0 + x1)/2)
#   trapezoidal   x1 - x0 = h (f(x0) + f(x1))/2         (the rule of EMT programs)
#   backward Euler x1 - x0 = h f(x1)
# For each rule and step: (1) the trajectory error at the end and its largest value over the saved grid, against a
# midpoint reference at a step 8 times smaller (scaled by `phs_scales`, max norm); (2) the energy defect of the rule's
# own discrete balance, accumulated over the run,
#   midpoint      sum [H(x1) - H(x0) - h (S - L)(x_m)]
#   trapezoidal   sum [H(x1) - H(x0) - h ((S - L)(x0) + (S - L)(x1))/2]
#   backward Euler sum [H(x1) - H(x0) - h (S - L)(x1)]
# with S the supply and L the losses of the energy account (`phs_ledger`), relative to the energy exchanged
# sum h (|S| + L). The midpoint defect is the solver's; the others are the rules' own. Each rule's defect is also
# given with the powers at the step's midpoint (`defect_mid`): for a linear device the trapezoidal and midpoint steps
# coincide, so this separates what the trajectory does (state-dependent J, numerical dissipation) from the quadrature.
# Run: julia --project=. scripts/paper/step_size_study.jl [--R=0.01]   ->  outputs/paper/step_size_study.json
using PHPS, LinearAlgebra, Printf, JSON3, ForwardDiff
root = normpath(joinpath(@__DIR__, "..", ".."))
opt(key, default) = (a = filter(x -> startswith(x, "--$key="), ARGS); isempty(a) ? default : split(a[1], "="; limit = 2)[2])
Rf = parse(Float64, opt("R", "0.01"))
sc_ = load_scenario(joinpath(root, "cases", "IEEE39Bus_PF", "no_fault_phtrue.json"))
eq = solve_equilibrium(load_case(sc_.system_path), sc_)
s, x0, u, _ = phs_equilibrium(eq)
n = length(x0)
kf = PHPS.bus_index(s.net.net, 16)
t_on, t_off, tstop = 0.005, 0.010, 0.020
uon = PHInput{Float64}(u.Tm, u.vf, (g = zeros(length(u.gf)); g[kf] = 1 / Rf; g), u.const_power, u.uc)
uoff = PHInput{Float64}(u.Tm, u.vf, zeros(length(u.gf)), u.const_power, u.uc)
inputs(t) = t_on - 1e-12 <= t < t_off - 1e-12 ? uon : uoff
sc = phs_scales(s, x0)
f(x, uu) = phs_rhs(s, x, uu)
A(x, uu) = ForwardDiff.jacobian(y -> phs_rhs(s, y, uu), x)
sl(x, uu) = (L = phs_ledger(s, x, uu); (sup = L.supply, loss = L.losses))

# one rule over [0, tstop] at step h; segments split at the fault's times; returns the end state, the states at the
# 1 ms grid, the accumulated energy defect and the energy exchanged
function integrate(rule::Symbol, h; energy = true, tol = 1e-13)
    x = copy(x0); grid = Dict{Int,Vector{Float64}}(0 => copy(x)); defect = 0.0; dmid = 0.0; thr = 0.0; iters = 0
    for (ta, tb) in ((0.0, t_on), (t_on, t_off), (t_off, tstop))
        uu = inputs(ta + h / 2)
        nst = round(Int, (tb - ta) / h)
        abs(nst * h - (tb - ta)) < 1e-12 || error("step $h does not divide the segment")
        Jm = A(x, uu)
        M = rule === :backward_euler ? I - h .* Jm : I - (h / 2) .* Jm
        F = lu(M)
        x1 = similar(x)
        for k in 1:nst
            f0 = rule === :trapezoidal ? f(x, uu) : nothing
            x1 .= x .+ h .* (rule === :trapezoidal ? f0 : f(x, uu))      # explicit predictor
            r = similar(x)
            for it in 1:60
                if rule === :midpoint
                    r .= x1 .- x .- h .* f((x .+ x1) ./ 2, uu)
                elseif rule === :trapezoidal
                    r .= x1 .- x .- (h / 2) .* (f0 .+ f(x1, uu))
                else
                    r .= x1 .- x .- h .* f(x1, uu)
                end
                iters += 1
                maximum(abs.(r) ./ sc) < tol && break
                it == 30 && (Jm = A(rule === :midpoint ? (x .+ x1) ./ 2 : x1, uu);
                             F = lu(rule === :backward_euler ? I - h .* Jm : I - (h / 2) .* Jm))
                it == 60 && error("$rule h = $h: Newton did not converge at t = $(ta + k * h)")
                x1 .-= F \ r
            end
            if energy
                dH = phs_energy(s, x1) - phs_energy(s, x)
                qm = sl((x .+ x1) ./ 2, uu); dmid += dH - h * (qm.sup - qm.loss)   # every rule, powers at the midpoint
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
            x .= x1
            t = ta + k * h
            ms = round(Int, t * 1e3)
            abs(t - ms * 1e-3) < 1e-12 && (grid[ms] = copy(x))
        end
    end
    return (; x, grid, defect, dmid, thr, iters)
end

steps = [1.25e-6, 2.5e-6, 5e-6, 1e-5, 2e-5]
href = steps[1] / 8
@printf("IEEE-39 grid model, %d states; fault at bus 16 through %.3g pu, %.0f-%.0f ms; reference midpoint at h = %.3g us\n",
        n, Rf, 1e3t_on, 1e3t_off, 1e6href)
t0 = time(); ref = integrate(:midpoint, href; energy = false)
@printf("reference: %.0f s\n", time() - t0)
err(a) = (; final = maximum(abs.(a.x .- ref.x) ./ sc),
          max = maximum(maximum(abs.(a.grid[k] .- ref.grid[k]) ./ sc) for k in keys(ref.grid)))
rows = Any[]
for rule in (:midpoint, :trapezoidal, :backward_euler), h in steps
    t1 = time()
    a = integrate(rule, h)
    e = err(a)
    push!(rows, Dict("rule" => String(rule), "h" => h, "error_final" => e.final, "error_max" => e.max,
                     "energy_defect" => a.defect, "energy_exchanged" => a.thr, "defect_rel" => abs(a.defect) / a.thr,
                     "defect_mid" => a.dmid, "defect_mid_rel" => abs(a.dmid) / a.thr,
                     "newton_iterations" => a.iters, "wall_s" => time() - t1))
    @printf("%-15s h = %6.3g us: error %.2e (max %.2e); energy defect %+.3e (%.2e of the exchanged); powers at the midpoint %.2e; %.0f s\n",
            rule, 1e6h, e.final, e.max, a.defect, abs(a.defect) / a.thr, abs(a.dmid) / a.thr, time() - t1)
end
out = joinpath(root, "outputs", "paper"); mkpath(out)
open(joinpath(out, "step_size_study.json"), "w") do io
    JSON3.pretty(io, Dict("model" => "IEEE39Bus_PF grid model (no controllers)", "states" => n, "fault_bus" => 16,
                          "fault_R" => Rf, "t_on" => t_on, "t_off" => t_off, "tstop" => tstop, "h_ref" => href, "rows" => rows))
end
println("written to ", joinpath(out, "step_size_study.json"))
