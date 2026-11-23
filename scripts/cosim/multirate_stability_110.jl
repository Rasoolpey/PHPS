# Multirate EMT stability: the window coupling through power ports (TII paper, Prop. "exact interface energy") against
# the MATE multirate interface of paper 110 (Sinkar, Dey, Zhao, Gole, "Stability assessment of multi-rate EMT
# simulations", IEEE Trans. Power Del. 40 (2025) 3017-3030), on their Test Case I (Fig. 6 there):
#   slow side (step dT): source 93 kV, 60 Hz, -- 16.4 ohm -- v2 -- (50 kohm || 0.147 H) -- v3
#   link v3 = v4 (zero impedance)
#   fast side (step dt): v4 -- 0.0413 H -- v5, and at v5 5 ohm and 10 uF to ground.
# Both schemes use the same one-step rule (trapezoidal = implicit midpoint on this linear network) and the same steps;
# they differ only in the interface:
#   MATE (110, Sec. III-B, after its ref. [6]): non-iterative; the slow Thevenin voltage is linearly interpolated over the
#     large step for the fast side, the fast link current is averaged over the large step for the slow side.
#   window (TII): over a window T = dT the fast side holds the slow side's port voltage a (its midpoint value) and
#     returns the step-weighted average b of its midpoint port current; a and b are solved together (implicit; linear
#     here, so one solve), so the port energy T a b booked by the two sides is the same number.
# Checks: (1) MATE reproduces 110's eigenvalues of the period map (dt = 5 us: -0.589 at dT = 5 us, -0.987486 at 40 us,
# -1.00869 at 45 us); (2) the largest eigenvalue modulus of both period maps over dT = 5..500 us; (3) a 0.5 s run at
# dT = 45 us (v3, against a single-rate reference at 1 us) and the energy balance of the window scheme per window;
# (4) the accuracy of both against a single-rate reference at 1 us.
# Run: julia --project=. scripts/cosim/multirate_stability_110.jl  ->  outputs/cosim/multirate_110.json, .pdf
using LinearAlgebra, Printf, JSON3

const R1, R2, L1 = 16.4, 50_000.0, 0.147          # slow side
const L2, R3, C = 0.0413, 5.0, 10e-6              # fast side
const EPK, F0 = 93e3, 60.0
E(t) = EPK * cos(2pi * F0 * t)

# ------------------------------------------------------------------------------------------------ MATE (paper 110)
# state carried from one large step to the next: (hL1, hL2, hC, VthS(t)); histories of the trapezoidal companions
struct Mate
    N::Int; dT::Float64; dt::Float64
    YS::Matrix{Float64}; YF::Matrix{Float64}; BS::Vector{Float64}; BF::Vector{Float64}; Zth::Float64
    GL1::Float64; GL2::Float64; GC::Float64
end
function Mate(N, dt)
    dT = N * dt
    GL1, GL2, GC = dT / (2L1), dt / (2L2), 2C / dt
    G1, G2 = 1 / R1, 1 / R2
    YS = [G1+G2+GL1 -(G2 + GL1); -(G2 + GL1) G2+GL1]       # nodes v2, v3
    YF = [GL2 -GL2; -GL2 GL2+1/R3+GC]                       # nodes v4, v5
    BS = YS \ [0.0, 1.0]                                    # P_S = e(v3)
    BF = YF \ [-1.0, 0.0]                                   # P_F = -e(v4): the link current enters v4
    Zth = BS[2] + (-BF[1])
    return Mate(N, dT, dt, YS, YF, BS, BF, Zth, GL1, GL2, GC)
end
# one large step from t; z = (hL1, hL2, hC, VthS_t); returns the new z, v5 at the end of the step and v3 (MATE's slow
# node voltage, computed with the link current averaged over the step)
_v5_last = 0.0
function mate_step(m::Mate, z, t, src::Bool)
    hL1, hL2, hC, VthS0 = z
    Es = src ? E(t + m.dT) : 0.0
    eS = m.YS \ [Es / R1 - hL1, hL1]
    VthS1 = eS[2]
    isum = 0.0
    for k in 1:m.N
        VthS = VthS0 + k / m.N * (VthS1 - VthS0)
        eF = m.YF \ [-hL2, hL2 - hC]
        ia = (VthS + (-eF[1])) / m.Zth
        vF = eF .- m.BF .* ia
        iL2 = m.GL2 * (vF[1] - vF[2]) + hL2
        hL2 = iL2 + m.GL2 * (vF[1] - vF[2])
        iCc = m.GC * vF[2] + hC
        hC = -iCc - m.GC * vF[2]
        isum += ia
        global _v5_last = vF[2]
    end
    iavg = isum / m.N
    vS = eS .- m.BS .* iavg
    iL1 = m.GL1 * (vS[1] - vS[2]) + hL1
    hL1 = iL1 + m.GL1 * (vS[1] - vS[2])
    return [hL1, hL2, hC, VthS1], _v5_last, vS[2]
end
function period_map(step, n)
    M = zeros(n, n)
    for j in 1:n
        e = zeros(n); e[j] = 1.0
        M[:, j] = step(e)
    end
    return M
end
mate_map(m::Mate) = period_map(z -> mate_step(m, z, 0.0, false)[1], 4)

# ------------------------------------------------------------------------------------------------ window coupling
# state (i1, i2, v5): the inductor currents and the capacitor voltage (energy H = L1 i1^2/2 + L2 i2^2/2 + C v5^2/2)
struct Window
    N::Int; dT::Float64; dt::Float64
end
Window(N::Int, dt::Float64) = Window(N, N * dt, dt)
energy(x) = L1 * x[1]^2 / 2 + L2 * x[2]^2 / 2 + C * x[3]^2 / 2
# slow side, one midpoint step of size T with the port current b held: returns i1+, its port voltage a (midpoint value)
function slow(w::Window, i1, b, Em)
    T = w.dT
    i1p = (i1 * (L1 / T - R2 / 2) + R2 * b) / (L1 / T + R2 / 2)
    a = Em - R1 * b - R2 * (b - (i1 + i1p) / 2)
    return i1p, a
end
# fast side, N midpoint steps of size dt with the port voltage a held: returns (i2, v5)+, the average port current b and
# the energy dissipated in R3
function fast(w::Window, i2, v5, a)
    h = w.dt; bsum = 0.0; diss = 0.0
    # [L2/h + 0, 1/2; -1/2, C/h + 1/(2 R3)] [i2+; v5+] = rhs
    A = [L2/h 0.5; -0.5 C/h+1/(2R3)]
    for _ in 1:w.N
        rhs = [L2 / h * i2 + a - v5 / 2, C / h * v5 + i2 / 2 - v5 / (2R3)]
        i2p, v5p = A \ rhs
        im, vm = (i2 + i2p) / 2, (v5 + v5p) / 2
        bsum += h * im; diss += h * vm^2 / R3
        i2, v5 = i2p, v5p
    end
    return i2, v5, bsum / w.dT, diss
end
# one window from t: solve a, b together (linear: b(a) = b0 + g a, a(b) = a0 + z b)
function window_step(w::Window, x, t, src::Bool)
    Em = src ? E(t + w.dT / 2) : 0.0
    _, _, b0, _ = fast(w, x[2], x[3], 0.0); _, _, b1, _ = fast(w, x[2], x[3], 1.0); g = b1 - b0
    _, a0 = slow(w, x[1], 0.0, Em); _, a1 = slow(w, x[1], 1.0, Em); zs = a1 - a0
    b = (b0 + g * a0) / (1 - g * zs)
    # two refinement steps on the interface residual b - b_fast(a_slow(b)) (the superposition rounds)
    for _ in 1:2
        _, a = slow(w, x[1], b, Em)
        bf = fast(w, x[2], x[3], a)[3]
        b += (bf - b) / (1 - g * zs)
    end
    i1p, a = slow(w, x[1], b, Em)
    i2p, v5p, bf, dfast = fast(w, x[2], x[3], a)
    xn = [i1p, i2p, v5p]
    T = w.dT
    dslow = T * (R1 * b^2 + R2 * (b - (x[1] + i1p) / 2)^2)
    supply = T * Em * b
    port_slow, port_fast = T * a * b, T * a * bf        # energy out of the slow side, into the fast side
    resid = (energy(xn) - energy(x)) - (supply - dslow - dfast) # the balance of the whole window
    return xn, (; resid, port_slow, port_fast, supply, a, moved = abs(port_slow) + abs(supply) + dslow + dfast)
end
window_map(w::Window) = period_map(x -> window_step(w, x, 0.0, false)[1], 3)

# ------------------------------------------------------------------------------------------------ 1. reproduction
dt = 5e-6
println("1. MATE period map: the eigenvalue on 110's critical branch (most negative real part; paper 110: -0.589,")
println("   -0.987486, -1.00869) and the largest modulus")
repro = Dict{String,Any}()
for N in (1, 8, 9)
    lam = eigvals(mate_map(Mate(N, dt)))
    l = lam[argmin(real.(lam))]
    @printf("   dT = %3d us, dt = 5 us: critical %s; largest |lambda| %.6f\n", round(Int, N * dt * 1e6),
            string(round(l, digits = 6)), maximum(abs, lam))
    repro["dT_$(round(Int, N * dt * 1e6))us"] = Dict("re" => real(l), "im" => imag(l), "abs" => abs(l))
end

# ------------------------------------------------------------------------------------------------ 2. sweep
Ns = 1:100
sweep = [(dT = N * dt, mate = maximum(abs, eigvals(mate_map(Mate(N, dt)))),
          window = maximum(abs, eigvals(window_map(Window(N, dt))))) for N in Ns]
println("2. largest |lambda| of the period map, dt = 5 us")
for s in sweep[[1, 8, 9, 10, 20, 40, 100]]
    @printf("   dT = %4.0f us: MATE %.6f   window %.6f\n", s.dT * 1e6, s.mate, s.window)
end
firstbad = findfirst(s -> s.mate > 1, sweep)
@printf("   MATE unstable from dT = %.0f us; window: largest |lambda| over the sweep %.6f\n",
        firstbad === nothing ? NaN : sweep[firstbad].dT * 1e6, maximum(s.window for s in sweep))

# ------------------------------------------------------------------------------------------------ 3. runs at 45 us
function run_mate(N, dt, tend)
    m = Mate(N, dt); z = zeros(4); t = 0.0
    # VthS at t = 0 from the zero histories and E(0)
    z[4] = (m.YS \ [E(0.0) / R1, 0.0])[2]
    ts, v3, v5 = Float64[], Float64[], Float64[]
    while t < tend - 1e-12
        z, v5e, v = mate_step(m, z, t, true)
        t += m.dT
        push!(ts, t); push!(v3, v); push!(v5, v5e)
    end
    return ts, v3, v5
end
function run_window(N, dt, tend)
    w = Window(N, dt); x = zeros(3); t = 0.0
    ts, v3, res, port, sup = Float64[], Float64[], Float64[], Float64[], Float64[]   # sup: energy moved per window
    te, v5 = Float64[], Float64[]
    while t < tend - 1e-12
        x, r = window_step(w, x, t, true)
        t += w.dT
        push!(ts, t - w.dT / 2); push!(v3, r.a); push!(res, r.resid); push!(port, r.port_slow - r.port_fast)
        push!(sup, r.moved); push!(te, t); push!(v5, x[3])
    end
    return ts, v3, res, port, sup, te, v5
end
tend = 0.5
tm, vm, v5m = run_mate(9, dt, tend)
tw, vw, rw, pw, sw, tew, v5w = run_window(9, dt, tend)
tr, vr, _, _, _, ter, v5r = run_window(1, 1e-6, tend)                 # single-rate reference (one 1 us step per window)
println("3. 0.5 s at dT = 45 us, dt = 5 us")
@printf("   MATE: max |v3| %.3g V (reference %.3g V)\n", maximum(abs, filter(isfinite, vm)), maximum(abs, vr))
@printf("   window: max |v3| %.3g V; energy residual per window relative to the energy moved in it: max %.2e\n",
        maximum(abs, vw), maximum(abs.(rw) ./ sw))
@printf("   window: port energy booked by the slow side minus the fast side, relative: max %.2e\n",
        maximum(abs.(pw) ./ sw))
# accuracy: v5 (a state both schemes compute) at the end of each large step against the single-rate 1 us reference at
# the same instant, over 0.1-0.5 s (after the start-up transient from rest), relative to the reference's peak there
v5r_at(t) = v5r[round(Int, t / 1e-6)]
function relerr(te, v5)
    k = findall(t -> 0.1 <= t <= tend, te)
    pk = maximum(abs(v5r_at(te[i])) for i in k)
    return maximum(abs(v5[i] - v5r_at(te[i])) for i in k) / pk
end
acc = Dict{String,Float64}()
acc["window_45_5"] = relerr(tew, v5w)
r1 = run_window(1, 45e-6, tend); acc["single_45"] = relerr(r1[6], r1[7])
r2 = run_window(1, 5e-6, tend);  acc["single_5"] = relerr(r2[6], r2[7])
r3 = run_window(8, dt, tend);    acc["window_40_5"] = relerr(r3[6], r3[7])
r4 = run_mate(8, dt, tend);      acc["mate_40_5"] = relerr(r4[1], r4[3])
pk(ts, v) = maximum(abs(v[i]) for i in eachindex(ts) if 0.1 <= ts[i] <= tend)
@printf("   v3 amplitude over 0.1-0.5 s: reference %.1f kV, window (45/5 us) %.1f kV, MATE (40/5 us, stable) %.1f kV\n",
        pk(tr, vr) / 1e3, pk(tw, vw) / 1e3, pk(r4[1], r4[2]) / 1e3)
println("4. largest v5 error over 0.1-0.5 s against the single-rate 1 us reference, relative to its peak:")
for k in ("single_5", "window_40_5", "mate_40_5", "window_45_5", "single_45")
    @printf("   %-12s %.3g\n", k, acc[k])
end
o = joinpath(@__DIR__, "..", "..", "outputs", "cosim"); mkpath(o)
open(joinpath(o, "multirate_110.json"), "w") do io
    JSON3.pretty(io, Dict("reproduction" => repro,
        "sweep" => [Dict("dT" => s.dT, "mate" => s.mate, "window" => s.window) for s in sweep],
        "run45" => Dict("mate_t" => tm, "mate_v3" => vm, "window_t" => tw, "window_v3" => vw, "window_resid" => rw,
                        "window_port" => pw, "ref_t" => tr[1:10:end], "ref_v3" => vr[1:10:end]),
        "accuracy" => acc, "mate40_t" => r4[1], "mate40_v3" => r4[2], "window_resid_rel_max" => maximum(abs.(rw) ./ sw)))
end
println("written to ", joinpath(o, "multirate_110.json"))

# ------------------------------------------------------------------------------------------------ figure
using CairoMakie
set_theme!(Theme(fontsize = 8, linewidth = 1.0, Axis = (; xgridvisible = false, ygridvisible = false, xtickalign = 1,
                                                       ytickalign = 1, spinewidth = 0.6, xticksize = 3, yticksize = 3)))
cols = Makie.wong_colors()
fig = Figure(size = (252, 118), figure_padding = (2, 6, 2, 2))
a1 = Axis(fig[1, 1], xscale = log10, xlabel = "large step ΔT [μs] (small step 5 μs)", ylabel = "largest |λ|",
          xticks = ([5, 10, 20, 50, 100, 200, 500], ["5", "10", "20", "50", "100", "200", "500"]))
hlines!(a1, [1.0], color = :gray60, linestyle = :dash, linewidth = 0.7)
lines!(a1, [1e6 * x.dT for x in sweep], [x.mate for x in sweep], color = cols[2], label = "interpolated interface (MATE)")
lines!(a1, [1e6 * x.dT for x in sweep], [x.window for x in sweep], color = cols[1], label = "window coupling")
scatter!(a1, [45.0], [1.00869], color = :black, marker = :circle, markersize = 4, label = "reported by Sinkar et al.")
axislegend(a1, position = :lt, framevisible = false, labelsize = 6.5, patchsize = (12, 5), rowgap = 1, padding = (2, 2, 2, 2))
save(joinpath(o, "multirate_110.pdf"), fig, pt_per_unit = 1)
println("figure ", joinpath(o, "multirate_110.pdf"))
