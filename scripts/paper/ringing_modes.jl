# "The same model" (review 10): the modes of the linearization of PHPS's own EMT model against the ringing of the bus
# voltages after the fault is removed, in PowerFactory EMT and in PHPS, on the comparison case of the paper (IEEE 39-bus,
# exciters and governors in service, three-phase fault at bus 16, 0.1-0.2 s). A quasi-static (RMS) modal analysis has no
# network or stator states and cannot represent these modes.
# Inputs: cases/IEEE39Bus_PF/ph_fault_bus16_10s.json, the PowerFactory winding data of cases/IEEE39Bus_FullModel, and the
# stored traces outputs/ph/pf_compare_ph_fault_bus16_10s_controllers_series_all/traces.jld2 (ph_pf_compare.jl
# --save-traces). Output: outputs/paper/ringing_modes.json.
using PHPS, LinearAlgebra, ForwardDiff, JLD2, JSON3, Printf
root = normpath(joinpath(@__DIR__, "..", ".."))
sc = load_scenario(joinpath(root, "cases", "IEEE39Bus_PF", "ph_fault_bus16_10s.json"))
eq = solve_equilibrium(load_case(sc.system_path), sc)
wc = JSON3.read(read(joinpath(root, "cases", "IEEE39Bus_FullModel", "winding_circuits.json"), String))
fl(v) = v === nothing ? nothing : Float64(v)
circuits = Dict(parse(Int, String(b)) => (; Ll = fl(v.Ll), Lad = fl(v.Lad), Laq = fl(v.Laq), Lfd = fl(v.Lfd), L1d = fl(v.L1d),
                L1q = fl(v.L1q), L2q = fl(v.L2q), Ra = fl(v.Ra), Rfd = fl(v.Rfd), R1d = fl(v.R1d), R1q = fl(v.R1q),
                R2q = fl(v.R2q), H = fl(v.H)) for (b, v) in pairs(wc.machines))
cs, x0, u, rep = phs_controlled(eq; loads = :series_all, circuits)
@printf("model: %d states (%d physical); residual at x0 %.1e\n", cs.n, cs.s.n, rep.residual_controlled)
A = ForwardDiff.jacobian(x -> phs_rhs(cs, x, u), x0)
lam = eigvals(A)
fast = sort([l for l in lam if imag(l) > 2pi * 100 && imag(l) < 2pi * 5000], by = l -> imag(l))
mode(l) = (f = imag(l) / 2pi, zeta = -real(l) / abs(l), sigma = real(l))
@printf("%d modes between 100 Hz and 5 kHz (dq frame)\n", length(fast))
# the ringing after the removal: matrix pencil on the bus-voltage magnitude deviation (needs the stored PowerFactory
# traces; without them only the eigenvalues are reported)
tf = joinpath(root, "outputs", "ph", "pf_compare_ph_fault_bus16_10s_controllers_series_all", "traces.jld2")
res = Dict{String,Any}()
if !isfile(tf)
    println("no stored traces at ", tf, ": ringing fit skipped")
else
d = load(tf)
t, toff, ids = d["t"], d["t_off"], d["bus_ids"]
function pencil(y, dt, p)
    N = length(y); L = N ÷ 2
    Y = [y[i + j - 1] for i in 1:N-L, j in 1:L+1]
    U, S, V = svd(Y); V1 = V[1:end-1, 1:p]; V2 = V[2:end, 1:p]
    z = eigvals(pinv(V1) * V2)
    s = log.(complex.(z)) ./ dt
    return [(f = abs(imag(x)) / 2pi, zeta = -real(x) / abs(x)) for x in s if imag(x) > 2pi * 50]
end
w = findall(x -> toff + 2e-4 <= x <= toff + 1.2e-2, t)
res["window_s"] = [toff + 2e-4, toff + 1.2e-2]
dt = t[w[2]] - t[w[1]]
for b in (18, 16, 15, 17)
    i = findfirst(==(b), ids)
    out = Dict{String,Any}()
    for (src, V) in (("powerfactory", d["pfV"]), ("phps", d["phV"]))
        y = V[i, w]; y = y .- (y[end] + y[end-1]) / 2
        out[src] = sort(pencil(y, dt, 16), by = m -> m.f)
    end
    res["bus$b"] = out
    @printf("bus %d ringing (matrix pencil, 0.2-12 ms after removal):\n  PF   %s\n  PHPS %s\n", b,
            join([@sprintf("%.0f Hz/%.2f", m.f, m.zeta) for m in out["powerfactory"]], ", "),
            join([@sprintf("%.0f Hz/%.2f", m.f, m.zeta) for m in out["phps"]], ", "))
end
end
@printf("PHPS eigenvalues (Hz/damping ratio): %s\n", join([@sprintf("%.0f/%.3f", mode(l).f, mode(l).zeta) for l in fast], ", "))
o = joinpath(root, "outputs", "paper"); mkpath(o)
open(joinpath(o, "ringing_modes.json"), "w") do io
    JSON3.pretty(io, Dict("eigen_modes" => [Dict("f_hz" => mode(l).f, "zeta" => mode(l).zeta, "sigma" => mode(l).sigma) for l in fast],
                          "ringing" => res, "states" => cs.n))
end
println("written to ", joinpath(o, "ringing_modes.json"))
