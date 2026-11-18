# The GFL board run's data (src/rt/converters/CPP/gfl/gfl_board_data.bin, git-ignored; the flow copies it): the C simulation's
# event sequences (outputs/realtime/converters/gfl_tb.bin), the C++ results (gfl_cpp_trace.bin: u_c every step;
# gfl_cpp_bstates.bin: the 26 words and the flags after every call) and Julia's.
#   julia --project=. scripts/realtime/gfl_board_data.jl      (after scripts/realtime/csim_gfl.sh)
# Layout (Float64): nconv; per converter: set, nev, the state (26); per event: type, then (step) i (2), y C++ (2), y Julia (2)
# or (call) v (2), the C++ state (26), C++ flags, the Julia state (26), Julia flags.
using Printf
od = joinpath(pwd(), "outputs", "realtime", "converters")
io = open(joinpath(od, "gfl_tb.bin")); nconv = Int(read(io, Int32))
T = reinterpret(Float64, read(joinpath(od, "gfl_cpp_trace.bin"))); BS = reinterpret(Float64, read(joinpath(od, "gfl_cpp_bstates.bin")))
out = Float64[nconv]; pt = 0; pb = 0
for j in 1:nconv
    s = read(io, Int32); nev = Int(read(io, Int32)); init = read!(io, Vector{Float64}(undef, 26))
    append!(out, [s, nev]); append!(out, init)
    for e in 1:nev
        t = read(io, Int32)
        if t == 0
            E = read!(io, Vector{Float64}(undef, 4))
            append!(out, [0.0, E[1], E[2], T[pt+1], T[pt+2], E[3], E[4]]); global pt += 2
        else
            E = read!(io, Vector{Float64}(undef, 29))
            append!(out, [1.0, E[1], E[2]]); append!(out, BS[pb+1:pb+27]); append!(out, E[3:29]); global pb += 27
        end
    end
end
close(io)
(pt == length(T) && pb == length(BS)) || error("trace / states do not match the events")
path = joinpath(pwd(), "src", "rt", "converters", "CPP", "gfl", "gfl_board_data.bin")
write(path, out)
@printf("written %s (%.1f MB)\n", path, filesize(path) / 2^20)
