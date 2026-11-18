# The DC side's board data (src/rt/converters/CPP/dcside/dcside_board_data.bin and dcside_board_bstates.bin, git-ignored;
# the flow copies them to the board): the C simulation's replays (outputs/realtime/converters/dcside_tb.bin), the C++
# results (dcside_cpp_trace.bin, dcside_cpp_bstates.bin: run scripts/realtime/csim_dcside.sh first) and Julia's.
#   julia --project=. scripts/realtime/dcside_board_data.jl
# Layout (Float64): nrep; per replay: set (0-based), steps, x (5), xc (4); then per replay, per step: u (4), y C++ (3),
# i_b C++, flags C++ (rejected + 2 clipped), y Julia (3), i_b Julia, Julia's packed flags (+ 4 sticky).
using Printf
root = pwd()
od = joinpath(root, "outputs", "realtime", "converters")
cpp = joinpath(root, "src", "rt", "converters", "CPP", "dcside")
const NR = 14
io = open(joinpath(od, "dcside_tb.bin"))
nrep = Int(read(io, Int32))
T = reshape(reinterpret(Float64, read(joinpath(od, "dcside_cpp_trace.bin"))), 6, :)   # y (3), yb, fo, v per step
hdr = Float64[nrep]; body = Vector{Matrix{Float64}}()
off = 0
for j in 1:nrep
    s = read(io, Int32); ns = Int(read(io, Int32))
    x0 = read!(io, Vector{Float64}(undef, 5))
    R = reshape(read!(io, Vector{Float64}(undef, (4 + NR) * ns)), 4 + NR, ns)    # u (4), the Julia row (x 5, xc 4, y 3, yb, packed)
    C = T[:, off+1:off+ns]; global off += ns
    append!(hdr, [s, ns]); append!(hdr, x0); append!(hdr, zeros(4))
    push!(body, vcat(R[1:4, :], C[1:5, :], R[4+10:4+14, :]))
end
close(io)
off == size(T, 2) || error("the trace ($(size(T, 2)) steps) does not match the replays ($off)")
path = joinpath(cpp, "dcside_board_data.bin")
open(path, "w") do f
    write(f, hdr)
    foreach(B -> write(f, B), body)                  # column-major: one step's 14 values contiguous
end
cp(joinpath(od, "dcside_cpp_bstates.bin"), joinpath(cpp, "dcside_board_bstates.bin"); force = true)
@printf("written %s (%d replays, %d steps, %.1f MB) and dcside_board_bstates.bin\n", path, nrep, off, filesize(path) / 2^20)
