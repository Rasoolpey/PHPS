# The VSM board run's data (src/rt/converters/CPP/vsm/vsm_board_data.bin, git-ignored; the flow copies it to the board):
# the six converters' replay inputs (alpha-beta), the C++ results (the C simulation's trace,
# outputs/realtime/converters/vsm_cpp_trace.bin: run scripts/realtime/csim_vsm.sh first) and Julia's (@hls, Float64).
#   julia --project=. scripts/realtime/vsm_board_data.jl
# Layout (Float64): nconv, ncall; per converter: set (0-based), x (3), phi, remainder; per call: u (5), y C++ (2),
# y Julia (2), Julia's flags packed (engaged + 2 u_max clip + 4 m clip + 8 v_dc domain).
using PHPS, JLD2, TOML, Printf
const RT = PHPS.RT
root = pwd()
od = joinpath(root, "outputs", "realtime", "converters")
OB = TOML.parsefile(joinpath(root, "src", "rt", "converters", "data", "ieee39_vsm.toml"))["omega_b"]
ds = [RT.vsm_data(i) for i in 1:RT.vsm_ninstances()]
R = load(joinpath(od, "vsm_replay.jld2"))
before, vk, ic, tc = R["before"], R["vk"], R["ic"], R["tcall"]
ncall, nconv = size(vk)
cpp = reshape(reinterpret(Float64, read(joinpath(od, "vsm_cpp_trace.bin"))), 6, ncall, nconv)
out = Float64[nconv, ncall]
st2000 = Float64[]
for j in 1:nconv
    s = findfirst(e -> e["name"] == R["names"][j], ds)
    k = RT.vsm_coefs(ds[s])
    x = before[1, j, 2:4]; ph = [mod((before[1, j, 1] + OB * tc[1]) / 2pi, 1.0)]; pr = [0.0]
    append!(out, [s - 1; x; ph; pr])
    y = zeros(2); fl = zeros(Int, 4)
    for c in 1:ncall
        c == 2001 && append!(st2000, [x; ph; pr])     # the Julia state before call 2001 (the board's --timing batch)
        r = cis(OB * tc[c])
        u = [real(vk[c, j] * r), imag(vk[c, j] * r), real(ic[c, j] * r), imag(ic[c, j] * r), before[c, j, 7]]
        RT.vsm_call!(x, ph, pr, u, y, fl, k)
        append!(out, u); append!(out, cpp[1:2, c, j]); append!(out, y); push!(out, fl[1] + 2fl[2] + 4fl[3] + 8fl[4])
    end
end
path = joinpath(root, "src", "rt", "converters", "CPP", "vsm", "vsm_board_data.bin")
write(path, out)
write(replace(path, "vsm_board_data.bin" => "vsm_board_state2000.bin"), st2000)
cp(joinpath(od, "vsm_cpp_bstates.bin"), replace(path, "vsm_board_data.bin" => "vsm_board_bstates.bin"); force = true)   # the C++ states after every batch
@printf("written %s (%d converters x %d calls, %.1f MB)\n", path, nconv, ncall, filesize(path) / 2^20)
