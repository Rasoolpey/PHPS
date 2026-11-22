# Work-precision study for the PHPS paper (review 09): accuracy and energy defect against computing time, on the setup
# of scripts/paper/step_size_study.jl (IEEE-39 grid model, three-phase fault at bus 16 through 0.01 pu, 5-10 ms, 20 ms).
# Solvers:
#   midpoint, trapezoidal, backward Euler   fixed steps, written here on `phs_rhs` (as in step_size_study.jl), timed
#                                           without the energy bookkeeping
#   phs_simulate                            the simulator's own midpoint integrator with its energy account, at the
#                                           same steps (its time includes the account)
#   CVODE (phs_simulate_bdf)                SUNDIALS variable-order BDF with adaptive steps, at several tolerances
# Error: the state at 20 ms against a midpoint reference at h = 1.25 us / 8 (max norm, scaled by `phs_scales`).
# Time: the smallest of two timed runs after a warm-up. The energy defects of the fixed-step rules are those of
# step_size_study.json (the bookkeeping does not change the trajectory); phs_simulate's is its own account.
# Run: julia --project=. scripts/paper/work_precision.jl   ->  outputs/paper/work_precision.json
using PHPS, LinearAlgebra, Printf, JSON3, ForwardDiff
root = normpath(joinpath(@__DIR__, "..", ".."))
Rf = 0.01
sc_ = load_scenario(joinpath(root, "cases", "IEEE39Bus_PF", "no_fault_phtrue.json"))
eq = solve_equilibrium(load_case(sc_.system_path), sc_)
s, x0, u, _ = phs_equilibrium(eq)
kf = PHPS.bus_index(s.net.net, 16)
t_on, t_off, tstop = 0.005, 0.010, 0.020
uon = PHInput{Float64}(u.Tm, u.vf, (g = zeros(length(u.gf)); g[kf] = 1 / Rf; g), u.const_power, u.uc)
uoff = PHInput{Float64}(u.Tm, u.vf, zeros(length(u.gf)), u.const_power, u.uc)
sc = phs_scales(s, x0)
f(x, uu) = phs_rhs(s, x, uu)
A(x, uu) = ForwardDiff.jacobian(y -> phs_rhs(s, y, uu), x)
function integrate(rule::Symbol, h; tol = 1e-13)
    x = copy(x0)
    for (ta, tb) in ((0.0, t_on), (t_on, t_off), (t_off, tstop))
        uu = t_on - 1e-12 <= ta + h / 2 < t_off - 1e-12 ? uon : uoff
        nst = round(Int, (tb - ta) / h)
        Jm = A(x, uu); F = lu(rule === :backward_euler ? I - h .* Jm : I - (h / 2) .* Jm)
        x1 = similar(x); r = similar(x)
        for k in 1:nst
            f0 = rule === :trapezoidal ? f(x, uu) : nothing
            x1 .= x .+ h .* (rule === :trapezoidal ? f0 : f(x, uu))
            for it in 1:60
                rule === :midpoint ? (r .= x1 .- x .- h .* f((x .+ x1) ./ 2, uu)) :
                rule === :trapezoidal ? (r .= x1 .- x .- (h / 2) .* (f0 .+ f(x1, uu))) :
                                        (r .= x1 .- x .- h .* f(x1, uu))
                maximum(abs.(r) ./ sc) < tol && break
                it == 30 && (Jm = A(rule === :midpoint ? (x .+ x1) ./ 2 : x1, uu);
                             F = lu(rule === :backward_euler ? I - h .* Jm : I - (h / 2) .* Jm))
                it == 60 && error("$rule h = $h did not converge")
                x1 .-= F \ r
            end
            x .= x1
        end
    end
    return x
end
best(fun) = (fun(); minimum(@elapsed(fun()) for _ in 1:2))     # warm-up, then the best of two
steps = [1.25e-6, 2.5e-6, 5e-6, 1e-5, 2e-5]
println("reference..."); flush(stdout)
xref = integrate(:midpoint, steps[1] / 8)
err(x) = maximum(abs.(x .- xref) ./ sc)
faults = [PHFault(16, Rf, t_on, t_off)]
rows = Any[]
for rule in (:midpoint, :trapezoidal, :backward_euler), h in steps
    xe = integrate(rule, h); w = best(() -> integrate(rule, h))
    push!(rows, Dict("solver" => String(rule), "param" => h, "error" => err(xe), "wall_s" => w))
    @printf("%-16s h = %6.3g us: error %.2e, %.3f s\n", rule, 1e6h, err(xe), w); flush(stdout)
end
for h in steps
    r = phs_simulate(s, x0, u; tstop, faults, h, tol = 1e-13)
    w = best(() -> phs_simulate(s, x0, u; tstop, faults, h, tol = 1e-13))
    e = err(r.X[:, end]); bal = abs(r.total.scheme - sum(r.closure)) / r.total.throughput
    abs(r.t[end] - tstop) < 1e-12 || error("phs_simulate did not save the end")
    push!(rows, Dict("solver" => "phs_simulate", "param" => h, "error" => e, "wall_s" => w, "energy_defect_rel" => bal))
    @printf("%-16s h = %6.3g us: error %.2e, %.3f s, account %.2e\n", "phs_simulate", 1e6h, e, w, bal); flush(stdout)
end
for rt in (1e-4, 1e-5, 1e-6, 1e-7, 1e-8, 1e-9, 1e-10, 1e-11)
    sol = phs_simulate_bdf(s, x0, u; tstop, faults, saveat = [tstop], reltol = rt, abstol = rt / 100)
    xe = sol[2][:, end]
    w = best(() -> phs_simulate_bdf(s, x0, u; tstop, faults, saveat = [tstop], reltol = rt, abstol = rt / 100))
    push!(rows, Dict("solver" => "cvode_bdf", "param" => rt, "error" => err(xe), "wall_s" => w))
    @printf("%-16s rtol = %.0e: error %.2e, %.3f s\n", "CVODE (BDF)", rt, err(xe), w); flush(stdout)
end
out = joinpath(root, "outputs", "paper"); mkpath(out)
open(joinpath(out, "work_precision.json"), "w") do io
    JSON3.pretty(io, Dict("model" => "IEEE39Bus_PF grid model (no controllers)", "fault_R" => Rf, "t_on" => t_on,
                          "t_off" => t_off, "tstop" => tstop, "h_ref" => steps[1] / 8, "rows" => rows))
end
println("written to ", joinpath(out, "work_precision.json"))
