# PHMachine4's board data (src/rt/machines/CPP/phmachine4/phmachine4_board_data.bin): the ten machine sets (coefficient
# words, initial state, bench), the checkpoints of the C simulation and Julia's states there (the same @hls code, Float64);
# then copies the C simulation's C++ checkpoints (outputs/realtime/flow/phmachine4/phm4_cpp_states.bin) beside it as
# phmachine4_board_cpp.bin. Same checkpoints and order as scripts/realtime/phm4_csim_data.jl.
#   julia --project=. scripts/realtime/phm4_board_data.jl      (after the C simulation)
using PHPS
const RT = PHPS.RT
root = pwd()
cpp = joinpath(root, "src", "rt", "machines", "CPP", "phmachine4")
h = setprecision(() -> BigFloat(RT.PHM4_STEP_CYCLES) * RT.PHM4_CLOCK_NS / 10^9, BigFloat, 256)
hh = Float64(h)
names = RT.phm4_names()
nst = round(Int, 0.3 / hh); non = round(Int, 0.1 / hh); noff = round(Int, 0.2 / hh)
cuts = sort(unique(vcat([round(Int, t / hh) for t in 0.01:0.01:0.3], non - 1, non, non + 1, non + 2, noff - 1, noff, noff + 1, noff + 2, nst)))
D = Float64[length(names), length(cuts), non, noff]; append!(D, cuts)
for nm in names
    d = RT.phm4_data(nm)
    k = setprecision(() -> RT.phm4_coefstruct(d, h), BigFloat, 256)
    x0, b = RT.phm4_initial(d, hh)
    append!(D, Float64.(codeunits(rpad(nm, 12)[1:12])))
    w = RT.phm4_coef_words(k); push!(D, length(w)); foreach(((v, F),) -> push!(D, v, F), w)
    append!(D, x0); append!(D, (b.vmag, b.cm, b.srot, b.vf, b.tm))
    x = copy(x0); n = 0
    for c in cuts
        RT.phm4_bench!(x, k, b, n, c - n; non, noff); append!(D, x); n = c
    end
    println(nm, ": ", length(w), " coefficient words, omega at 0.3 s ", x[9])
end
write(joinpath(cpp, "phmachine4_board_data.bin"), D)
src = joinpath(root, "outputs", "realtime", "flow", "phmachine4", "phm4_cpp_states.bin")
nb = filesize(src); nb == 8 * 20 * length(cuts) * length(names) || error("$src: $nb bytes, expected $(8 * 20 * length(cuts) * length(names))")
cp(src, joinpath(cpp, "phmachine4_board_cpp.bin"); force = true)
println("written phmachine4_board_data.bin ($(length(D)) words) and phmachine4_board_cpp.bin")
