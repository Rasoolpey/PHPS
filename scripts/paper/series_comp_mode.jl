# "The same model" with a series-compensated line (TPWRS paper, review 10): IEEE-39 (PowerFactory-matched, exciters and
# governors in service, PowerFactory's winding data) with Line_33 (26-29) compensated by a series capacitor (the case's
# `xc`). The linearization of PHPS's own EMT model has the subsynchronous network mode of the compensated line; an EMT
# run of a fault at bus 29 shows it in the line current. A quasi-static (RMS) network has no capacitor state: the line
# is x - xc there and the mode does not exist.
#   1. the controlled model at 50 % compensation: the modes with the largest participation of the capacitor's charge
#      pair, in the dq frame (f) and as the abc resonance f0 -/+ f;
#   2. a sweep of the compensation (20-70 %): the mode's abc frequency and damping;
#   3. an EMT run (fault at bus 29 through 0.001 pu, 0.1-0.15 s, to 0.45 s): Line_33's current, the matrix-pencil fit of
#      the post-clearing ringing (dq frame and phase a) against the eigenvalues.
# Inputs: cases/IEEE39Bus_PF/system_phtrue_sercomp.json (xc = 0.5 x on Line_33), ph_fault_bus29_sercomp.json,
# cases/IEEE39Bus_FullModel/winding_circuits.json. Output: outputs/paper/series_comp_mode.json, series_comp_mode.pdf.
using PHPS, LinearAlgebra, ForwardDiff, JSON3, Printf
root = normpath(joinpath(@__DIR__, "..", ".."))
const F0 = 60.0
sc = load_scenario(joinpath(root, "cases", "IEEE39Bus_PF", "ph_fault_bus29_sercomp.json"))
case_c = load_case(sc.system_path)
wc = JSON3.read(read(joinpath(root, "cases", "IEEE39Bus_FullModel", "winding_circuits.json"), String))
fl(v) = v === nothing ? nothing : Float64(v)
circuits = Dict(parse(Int, String(b)) => (; Ll = fl(v.Ll), Lad = fl(v.Lad), Laq = fl(v.Laq), Lfd = fl(v.Lfd), L1d = fl(v.L1d),
                L1q = fl(v.L1q), L2q = fl(v.L2q), Ra = fl(v.Ra), Rfd = fl(v.Rfd), R1d = fl(v.R1d), R1q = fl(v.R1q),
                R2q = fl(v.R2q), H = fl(v.H)) for (b, v) in pairs(wc.machines))
kl = findfirst(l -> l.idx == "Line_33", case_c.lines)
xline = case_c.lines[kl].x
with_xc(c0, xc) = (c = deepcopy(c0); l = c.lines[kl];
                   c.lines[kl] = PHPS.LineData(l.idx, l.bus1, l.bus2, l.r, l.x, l.b, l.tap, l.phi, xc); c)

# the controlled model and the capacitor's modes
function cap_modes(case; nmode = 2)
    eq = solve_equilibrium(case, sc)
    cs, x0, u, rep = phs_controlled(eq; loads = :series_all, circuits)
    ps = PHPS._phys(cs)
    A = ForwardDiff.jacobian(y -> phs_rhs(cs, y, u), x0)
    E = eigen(A)
    ci = isempty(ps.cidx) ? 0 : ps.cidx[1]
    # participation factors |v_ik w_ki| (right and left eigenvectors, normalised over the states): the capacitor pair's
    W = inv(E.vectors)
    pf(k) = (p = abs.(E.vectors[:, k] .* W[k, :]); p ./ sum(p))
    part = ci == 0 ? zeros(length(E.values)) : [(q = pf(k); q[ci] + q[ci+1]) for k in eachindex(E.values)]
    ks = [k for k in sortperm(part; rev = true) if imag(E.values[k]) > 0][1:nmode]
    modes = [(f = imag(E.values[k]) / 2pi, sigma = real(E.values[k]), zeta = -real(E.values[k]) / abs(E.values[k]),
              part = part[k]) for k in ks]
    return (; cs, x0, u, rep, ps, modes, lam = E.values)
end

println("1. the controlled model, Line_33 50 % compensated")
m50 = cap_modes(case_c)
@printf("   %d states (%d physical, capacitor pair at %d); rest residual %.1e\n", m50.cs.n, m50.ps.n, m50.ps.cidx[1],
        m50.rep.residual_controlled)
for m in m50.modes
    fabc = m.f < F0 ? F0 - m.f : m.f - F0
    @printf("   dq %.2f Hz, damping ratio %.4f (sigma %.2f /s), capacitor participation %.3f  ->  abc %.2f Hz\n",
            m.f, m.zeta, m.sigma, m.part, fabc)
end
# the uncompensated model has no mode near these frequencies with a capacitor (the line has none): its modes in 10-110 Hz
m0 = let eq0 = solve_equilibrium(with_xc(case_c, 0.0), sc)
    cs0, x00, u0, _ = phs_controlled(eq0; loads = :series_all, circuits)
    eigvals(ForwardDiff.jacobian(y -> phs_rhs(cs0, y, u0), x00))
end
band(l) = sort([z for z in l if 2pi * 10 < imag(z) < 2pi * 110], by = imag)
@printf("   modes 10-110 Hz (dq), compensated: %s\n", join([@sprintf("%.1f/%.3f", imag(z) / 2pi, -real(z) / abs(z)) for z in band(m50.lam)], ", "))
@printf("   modes 10-110 Hz (dq), uncompensated: %s\n", join([@sprintf("%.1f/%.3f", imag(z) / 2pi, -real(z) / abs(z)) for z in band(m0)], ", "))

println("2. compensation sweep")
sweep = Any[]
for k in (0.2, 0.3, 0.4, 0.5, 0.6, 0.7)
    mm = cap_modes(with_xc(case_c, k * xline))
    lo = mm.modes[argmin([m.f for m in mm.modes])]
    hi = mm.modes[argmax([m.f for m in mm.modes])]
    push!(sweep, Dict("k" => k, "f_dq_low" => lo.f, "zeta_low" => lo.zeta, "f_dq_high" => hi.f, "zeta_high" => hi.zeta,
                      "f_abc" => F0 - lo.f))
    @printf("   %2d %%: dq %.2f Hz (zeta %.4f) and %.2f Hz (zeta %.4f); abc %.2f Hz\n", round(Int, 100k), lo.f, lo.zeta,
            hi.f, hi.zeta, F0 - lo.f)
end

println("3. EMT run: fault at bus 29, 0.1-0.15 s")
ps, wb = m50.ps, m50.ps.omega_b
jl = ps.cpos[1]; l = ps.branches[jl]
tstop = 0.45
t0 = time()
r = phs_simulate(m50.cs, m50.x0, m50.u; tstop, faults = [PHFault(29, 0.001, 0.1, 0.15)], save_dt = 1e-4)
@printf("   %.0f s wall; energy account: worst step residual / throughput %.1e, Line_33 closure %.1e of its moved energy\n",
        time() - t0, maximum(r.scheme_max ./ max.(r.throughput, floatmin())),
        (k = findfirst(==("Line_33"), r.names); abs(r.closure[k]) / r.moved[k]))
t = r.t
iline = [wb * complex(r.X[ps.bidx[jl], c], r.X[ps.bidx[jl]+1, c]) / ps.net.L[l] for c in eachindex(t)]
vcap = [wb * ps.net.Xs[l] * complex(r.X[ps.cidx[1], c], r.X[ps.cidx[1]+1, c]) for c in eachindex(t)]
ia = [real(cis(wb * t[c]) * iline[c]) for c in eachindex(t)]           # phase a (synchronous frame at omega_b t)
function pencil(y, dt, p)
    N = length(y); L = N ÷ 2
    Y = [y[i + j - 1] for i in 1:N-L, j in 1:L+1]
    _, _, V = svd(Y); V1 = V[1:end-1, 1:p]; V2 = V[2:end, 1:p]
    s = log.(complex.(eigvals(pinv(V1) * V2))) ./ dt
    return sort([(f = imag(x) / 2pi, zeta = -real(x) / abs(x)) for x in s if imag(x) > 2pi * 5], by = m -> m.f)
end
w = findall(x -> 0.155 <= x <= 0.40, t)
dt = t[2] - t[1]
yd = real.(iline[w]) .- sum(real.(iline[w])) / length(w)
fit_dq = pencil(yd[1:4:end], 4dt, 8)
fit_a = pencil(ia[w][1:2:end], 2dt, 8)
@printf("   Line_33 current, dq real part, matrix pencil: %s\n", join([@sprintf("%.2f Hz/%.3f", m.f, m.zeta) for m in fit_dq], ", "))
@printf("   Line_33 current, phase a, matrix pencil: %s\n", join([@sprintf("%.2f Hz/%.3f", m.f, m.zeta) for m in fit_a], ", "))

o = joinpath(root, "outputs", "paper"); mkpath(o)
open(joinpath(o, "series_comp_mode.json"), "w") do io
    JSON3.pretty(io, Dict("line" => "Line_33", "x" => xline, "xc" => ps.net.Xs[l], "states" => m50.cs.n,
        "modes_50" => [Dict("f_dq" => m.f, "zeta" => m.zeta, "sigma" => m.sigma, "participation" => m.part,
                            "f_abc" => m.f < F0 ? F0 - m.f : m.f - F0) for m in m50.modes],
        "band_compensated" => [Dict("f" => imag(z) / 2pi, "zeta" => -real(z) / abs(z)) for z in band(m50.lam)],
        "band_uncompensated" => [Dict("f" => imag(z) / 2pi, "zeta" => -real(z) / abs(z)) for z in band(m0)],
        "sweep" => sweep, "fit_dq" => fit_dq, "fit_phase_a" => fit_a,
        "run" => Dict("t" => t, "i_re" => real.(iline), "i_im" => imag.(iline), "i_a" => ia, "vc_abs" => abs.(vcap),
                      "scheme_rel_max" => maximum(r.scheme_max ./ max.(r.throughput, floatmin())))))
end
println("written to ", joinpath(o, "series_comp_mode.json"))

# ---------------------------------------------------------------------------------------------- figure
using CairoMakie
set_theme!(Theme(fontsize = 8, linewidth = 1.0, Axis = (; xgridvisible = false, ygridvisible = false, xtickalign = 1,
                                                       ytickalign = 1, spinewidth = 0.6, xticksize = 3, yticksize = 3)))
cols = Makie.wong_colors()
fig = Figure(size = (516, 200))
a1 = Axis(fig[1, 1], xlabel = "compensation xc / x of Line 26-29 [%]", ylabel = "abc frequency [Hz]",
          title = "(a) subsynchronous network mode (eigenvalues)")
ks = [100 * d["k"] for d in sweep]
scatterlines!(a1, ks, [d["f_abc"] for d in sweep], color = cols[1], markersize = 5)
a1r = Axis(fig[1, 1], yaxisposition = :right, ylabel = "damping ratio", ytickcolor = cols[2], yticklabelcolor = cols[2],
           ylabelcolor = cols[2])
hidespines!(a1r); hidexdecorations!(a1r)
scatterlines!(a1r, ks, [d["zeta_low"] for d in sweep], color = cols[2], marker = :rect, markersize = 4)
a2 = Axis(fig[1, 2], xlabel = "time [s]", ylabel = "Line 26-29 current, phase a [pu]", title = "(b) EMT run, fault at bus 29")
k2 = findall(x -> 0.05 <= x <= tstop, t)
lines!(a2, t[k2], ia[k2], color = cols[1], linewidth = 0.6)
vspan!(a2, [0.1], [0.15], color = (:gray70, 0.3))
save(joinpath(o, "series_comp_mode.pdf"), fig, pt_per_unit = 1)
println("figure ", joinpath(o, "series_comp_mode.pdf"))
