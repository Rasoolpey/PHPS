# The GFL build's C simulation data (one build, two modes; reviews 64, 64 merge 4): per converter the 0.3 s sequence of the
# current loop's inputs (dcside_replay_gfl.jld2: the injected current at each step's midpoint, alpha-beta, applied at the
# build's h) with the calls in between (each after the step that ends on a multiple of Ts; the call's terminal voltage from
# gfl_replay.jld2, alpha-beta), and the Julia @hls results of every step and call; the coefficient header; the testbench:
# (1) the sequence through gfl_loop / gfl_call (C++) against Julia, (2) the same through the board top gfl_run in mode-0
# batches (up to 512 steps, broken at the calls) and mode-1 batches (one call), all 26 state words through double between
# batches and modes: bit for bit with (1). Limit 5e-8 (absolute + relative; vectors relative to their magnitude).
#   julia --project=. scripts/realtime/gfl_csim_data.jl      (after PHPS.RT.gfl_export(): the same types and h)
using PHPS, JLD2, Printf, TOML
const RT = PHPS.RT
root = pwd()
cpp = joinpath(root, "src", "rt", "converters", "CPP", "gfl")
od = joinpath(root, "outputs", "realtime", "converters")
bin = joinpath(od, "gfl_tb.bin")
HB = big(RT.GFL_II) * 8 / big(10)^9
setprecision(() -> RT.gfl_types!(HB), BigFloat, 256)
n = RT.gfl_ninstances(); ds = [RT.gfl_data(i) for i in 1:n]
ks = [setprecision(() -> RT.gfl_coefs(d), BigFloat, 256) for d in ds]
kl = [setprecision(() -> RT.gfl_loop(d, HB), BigFloat, 256) for d in ds]
OB = TOML.parsefile(joinpath(root, "src", "rt", "converters", "data", "ieee39_gfl.toml"))["omega_b"]
open(joinpath(cpp, "gfl_coefs_all.h"), "w") do io
    println(io, "// gfl_coefs_all.h: written by scripts/realtime/gfl_csim_data.jl; do not edit (the loop at h = $(Float64(HB)) s)")
    for i in 1:n
        println(io, "const GflCoefs GFL_COEFS_$i = ", RT._cinit(ks[i]), ";  // ", ds[i]["name"])
        println(io, "const GflLoop GFL_LOOP_$i = ", RT._cinit(kl[i]), ";")
    end
    println(io, "const GflCoefs *GFL_SETS[$n] = {", join(("&GFL_COEFS_$i" for i in 1:n), ", "), "};")
    println(io, "const GflLoop *GFL_LOOPS[$n] = {", join(("&GFL_LOOP_$i" for i in 1:n), ", "), "};")
end
flat(s) = reduce(vcat, s)                            # 26 words: ph, pr, xs 9, xr 6, lt 5, z 2, zr 2
D = load(joinpath(od, "dcside_replay_gfl.jld2")); C = load(joinpath(od, "gfl_replay.jld2"))
io = open(bin, "w")
write(io, Int32(length(D["names"])))
nsat = 0
for (q, nm) in enumerate(D["names"])
    i = findfirst(e -> e["name"] == nm, ds); jc = findfirst(==(nm), C["names"]); Dq = D["data"][q]
    s = RT.gfl_state(Float64.(ds[i]["x0"]), 0.0)
    init = flat(s)
    t = 0.0; ev = Vector{Vector{Float64}}()       # events: [0, i (2), u_c Julia (2)] or [1, v (2), the state after (26), flags]
    for r in axes(Dq, 1)
        tm = t + Dq[r, 1] / 2; iab = complex(Dq[r, 10], Dq[r, 11]) * cis(OB * tm)
        y = zeros(2); RT.gfl_loop!(s[6], s[7], s[5], [real(iab), imag(iab)], y, kl[i])
        push!(ev, [0.0, real(iab), imag(iab), y[1], y[2]])
        t += Dq[r, 1]
        c = round(Int, t / 5e-5)
        if abs(t / 5e-5 - c) < 1e-6 && c <= size(C["vk"], 1)
            v = C["vk"][c, jc] * cis(C["clk"][c]); fl = zeros(Int, 4)
            RT.gfl_call!(s[1], s[2], s[3], s[4], s[5], [real(v), imag(v)], fl, ks[i])
            global nsat += fl[1]
            push!(ev, vcat(1.0, real(v), imag(v), flat(s), fl[1] + 2fl[2] + 4fl[3] + 8fl[4]))
        end
    end
    write(io, Int32(i - 1), Int32(length(ev))); write(io, init)
    for e in ev
        write(io, Int32(e[1])); write(io, e[2:end])
    end
    @printf("%-18s %d steps, %d calls\n", nm, count(e -> e[1] == 0, ev), count(e -> e[1] == 1, ev))
end
close(io)
@printf("written %s (%.1f MB); calls with the circular limit: %d\n", bin, filesize(bin) / 2^20, nsat)
open(joinpath(cpp, "gfl_tb.cpp"), "w") do io
    print(io, """
// gfl_tb.cpp: written by scripts/realtime/gfl_csim_data.jl; do not edit
// C simulation of the GFL build: (1) the 0.3 s sequence (current-loop steps at h = $(Float64(HB)) s, the calls in between)
// through gfl_loop / gfl_call against Julia (5e-8; u_c every step, the 26 state words at each call: x + e compared for the
// compensated ones), (2) the same through the board top gfl_run in mode-0 batches (<= 512 steps, broken at the calls) and
// mode-1 batches, all state words through double between batches: bit for bit with (1). Returns 1 on failure.
#include <cstdio>
#include <cmath>
#include <cstdint>
#include <vector>
#include "gfl.h"
#include "gfl_coefs_all.h"
static const char *BIN = "$bin";
static const char *TRACE = "$(joinpath(od, "gfl_cpp_trace.bin"))";     // u_c every step (the board top's)
static const char *BSTATES = "$(joinpath(od, "gfl_cpp_bstates.bin"))"; // after every call: the 26 words and the flags
static const double LIM = 5e-8;
struct S { Turn ph[1]; Ang pr[1]; Pf xs[9]; Rm xr[6]; Sig lt[5]; Zs z[2]; Rm zr[2]; };
static void load(S &s, const double *v) { s.ph[0] = v[0]; s.pr[0] = v[1]; for (int i = 0; i < 9; i++) s.xs[i] = v[2 + i];
    for (int i = 0; i < 6; i++) s.xr[i] = v[11 + i]; for (int i = 0; i < 5; i++) s.lt[i] = v[17 + i];
    for (int i = 0; i < 2; i++) { s.z[i] = v[22 + i]; s.zr[i] = v[24 + i]; } }
static void save(const S &s, double *v) { v[0] = s.ph[0].to_double(); v[1] = s.pr[0].to_double();
    for (int i = 0; i < 9; i++) v[2 + i] = s.xs[i].to_double(); for (int i = 0; i < 6; i++) v[11 + i] = s.xr[i].to_double();
    for (int i = 0; i < 5; i++) v[17 + i] = s.lt[i].to_double(); for (int i = 0; i < 2; i++) { v[22 + i] = s.z[i].to_double(); v[24 + i] = s.zr[i].to_double(); } }
static double rd(double a, double b) { return std::fabs(a - b) / std::fmax(1.0, std::fabs(b)); }
static double wrapd(double a) { double r = std::fmod(a + 0.5, 1.0); if (r < 0) r += 1.0; return std::fabs(r - 0.5); }
static double cmpst(const double *a, const double *b) {   // the phase; xs (+ xr where compensated); lt (ch as a vector); z + zr
    double w = wrapd(a[0] - b[0]);
    const int xe[9] = {-1, 11, 12, -1, -1, 13, 14, 15, 16};
    for (int i = 0; i < 9; i++) w = std::fmax(w, rd(a[2 + i] + (xe[i] >= 0 ? a[xe[i]] : 0), b[2 + i] + (xe[i] >= 0 ? b[xe[i]] : 0)));
    w = std::fmax(w, std::hypot(a[17] - b[17], a[18] - b[18]));
    for (int i = 19; i < 22; i++) w = std::fmax(w, rd(a[i], b[i]));
    for (int i = 0; i < 2; i++) w = std::fmax(w, rd(a[22 + i] + a[24 + i], b[22 + i] + b[24 + i]));
    return w;
}
int main() {
    FILE *f = std::fopen(BIN, "rb"); if (!f) { std::printf("C sim check: cannot open %s\\n", BIN); return 1; }
    FILE *tr = std::fopen(TRACE, "wb");
    FILE *bs = std::fopen(BSTATES, "wb");
    int32_t nconv; std::fread(&nconv, 4, 1, f);
    bool ok = true; double wall = 0;
    static Sig IL[512][2], YL[512][2], UC[64][2]; static int FO[64];
    std::vector<int> SET(nconv); std::vector<std::vector<int>> TYP(nconv); std::vector<std::vector<std::vector<double>>> EV(nconv);
    std::vector<std::vector<double>> INIT(nconv, std::vector<double>(26)), YC(nconv), STC(nconv); std::vector<std::vector<int>> FC(nconv);
    std::vector<double> W(nconv, 0); std::vector<int> FD(nconv, 0);
    // (1) every converter's sequence through the functions, against Julia
    for (int j = 0; j < nconv; j++) {
        int32_t set, nev; std::fread(&set, 4, 1, f); std::fread(&nev, 4, 1, f); SET[j] = set;
        std::fread(INIT[j].data(), 8, 26, f);
        TYP[j].resize(nev); EV[j].resize(nev);
        for (int e = 0; e < nev; e++) { int32_t t; std::fread(&t, 4, 1, f); TYP[j][e] = t; EV[j][e].resize(t == 0 ? 4 : 29); std::fread(EV[j][e].data(), 8, EV[j][e].size(), f); }
        const GflCoefs &k = *GFL_SETS[set]; const GflLoop &kq = *GFL_LOOPS[set];
        S s; load(s, INIT[j].data()); double v[26];
        for (int e = 0; e < nev; e++) {
            const std::vector<double> &E = EV[j][e];
            if (TYP[j][e] == 0) {
                Sig i2[2] = {E[0], E[1]}, y2[2];
                gfl_loop(s.z, s.zr, s.lt, i2, y2, kq);
                YC[j].push_back(y2[0].to_double()); YC[j].push_back(y2[1].to_double());
                W[j] = std::fmax(W[j], std::hypot(y2[0].to_double() - E[2], y2[1].to_double() - E[3]) / std::fmax(1.0, std::hypot(E[2], E[3])));
            } else {
                Sig u2[2] = {E[0], E[1]}; int fl[4];
                gfl_call(s.ph, s.pr, s.xs, s.xr, s.lt, u2, fl, k);
                save(s, v); for (int i = 0; i < 26; i++) STC[j].push_back(v[i]);
                int pk = fl[0] + 2 * fl[1] + 4 * fl[2] + 8 * fl[3]; FC[j].push_back(pk);
                if (pk != (int)E[28]) FD[j]++;
                W[j] = std::fmax(W[j], cmpst(v, &E[2]));
            }
        }
    }
    // (2) the board top, the six converters interleaved per call period (each: a mode-0 batch of its steps up to its next
    // call, then that call), coefficients and the 26 words reloaded for every batch (66, 2)
    std::vector<S> B(nconv); for (int j = 0; j < nconv; j++) load(B[j], INIT[j].data());
    std::vector<size_t> pe(nconv, 0), py(nconv, 0), pc(nconv, 0); std::vector<bool> same(nconv, true);
    Sig uo; bool sto, sao; double v[26]; bool more = true;
    while (more) {
        more = false;
        for (int j = 0; j < nconv; j++) {
            int nev = (int)TYP[j].size(); if ((int)pe[j] >= nev) continue; more = true;
            const GflCoefs &k = *GFL_SETS[SET[j]]; const GflLoop &kq = *GFL_LOOPS[SET[j]];
            int e = (int)pe[j];
            while (e < nev && TYP[j][e] == 0) {
                int m = 0; while (e + m < nev && TYP[j][e + m] == 0 && m < 512) { IL[m][0] = EV[j][e + m][0]; IL[m][1] = EV[j][e + m][1]; m++; }
                gfl_run(k, kq, 0, B[j].ph, B[j].pr, B[j].xs, B[j].xr, B[j].lt, B[j].z, B[j].zr, UC, FO, IL, YL, m, 32, uo, sto, sao);
                for (int q = 0; q < m; q++) {
                    if (YL[q][0].to_double() != YC[j][py[j]] || YL[q][1].to_double() != YC[j][py[j] + 1]) same[j] = false;
                    py[j] += 2;
                }
                save(B[j], v); load(B[j], v); e += m;
            }
            if (e < nev) {
                UC[0][0] = EV[j][e][0]; UC[0][1] = EV[j][e][1];
                gfl_run(k, kq, 1, B[j].ph, B[j].pr, B[j].xs, B[j].xr, B[j].lt, B[j].z, B[j].zr, UC, FO, IL, YL, 1, 32, uo, sto, sao);
                save(B[j], v);
                for (int i = 0; i < 26; i++) if (v[i] != STC[j][26 * pc[j] + i]) same[j] = false;
                if (FO[0] != FC[j][pc[j]]) same[j] = false;
                pc[j]++; load(B[j], v); e++;
            }
            pe[j] = e;
        }
    }
    // the traces for the board, in converter order
    for (int j = 0; j < nconv; j++) {
        if (tr) std::fwrite(YC[j].data(), 8, YC[j].size(), tr);
        if (bs) for (size_t c = 0; c < FC[j].size(); c++) { std::fwrite(&STC[j][26 * c], 8, 26, bs); double fo = FC[j][c]; std::fwrite(&fo, 8, 1, bs); }
        bool p = W[j] <= LIM && FD[j] == 0 && same[j];
        std::printf("C sim check: GFL %d (set %d), %zu steps, %zu calls: max diff %.3e (limit %.0e), flag differences %d, functions = board top interleaved across the six (u_c, flags, 26 words) %s: %s\\n",
                    j + 1, SET[j] + 1, YC[j].size() / 2, FC[j].size(), W[j], LIM, FD[j], same[j] ? "bit for bit" : "NO", p ? "PASS" : "FAIL");
        ok = ok && p; wall = std::fmax(wall, W[j]);
    }
    std::fclose(f); if (tr) std::fclose(tr); if (bs) std::fclose(bs);
    std::printf("C sim check: %s (max %.3e)\\n", ok ? "ALL PASS" : "FAIL", wall);
    return ok ? 0 : 1;
}
""")
end
println("written ", joinpath(cpp, "gfl_tb.cpp"), " and gfl_coefs_all.h")
