# A converter call's board data (src/rt/converters/CPP/<tag>/<tag>_board_data.bin and <tag>_board_bstates.bin, git-ignored;
# the flow copies them to the board): the replay's inputs, the C++ results (the C simulation's trace: run
# scripts/realtime/csim_conv.sh <tag> first) and Julia's.
#   julia --project=. scripts/realtime/conv_board_data.jl <droop|voc>
# Layout (Float64): nconv, ncall; per converter: set (0-based), the state (NS words); per call: u (5), y C++ (2),
# y Julia (2), Julia's packed flags.
using Printf
tag = ARGS[1]
root = pwd()
od = joinpath(root, "outputs", "realtime", "converters")
io = open(joinpath(od, "$(tag)_tb.bin"))
nconv = Int(read(io, Int32))
T = reshape(reinterpret(Float64, read(joinpath(od, "$(tag)_cpp_trace.bin"))), 2, :)
NS = Dict("droop" => 24, "voc" => 11)[tag]; NR = 2 + NS + 1
out = Float64[]; off = 0; ncall = 0
for j in 1:nconv
    s = read(io, Int32); nc = Int(read(io, Int32)); global ncall = nc
    x0 = read!(io, Vector{Float64}(undef, NS))
    M = reshape(read!(io, Vector{Float64}(undef, (5 + NR) * nc)), 5 + NR, nc)
    append!(out, [s; x0])
    for c in 1:nc
        append!(out, M[1:5, c]); append!(out, T[:, off + c]); append!(out, M[6:7, c]); push!(out, M[5 + NR, c])
    end
    global off += nc
end
close(io)
off == size(T, 2) || error("trace length $(size(T, 2)) != $off")
path = joinpath(root, "src", "rt", "converters", "CPP", tag, "$(tag)_board_data.bin")
write(path, vcat(Float64[nconv, ncall], out))
cp(joinpath(od, "$(tag)_cpp_bstates.bin"), replace(path, "_board_data.bin" => "_board_bstates.bin"); force = true)
@printf("written %s (%d converters x %d calls, %.1f MB) and %s_board_bstates.bin\n", path, nconv, ncall, filesize(path) / 2^20, tag)
