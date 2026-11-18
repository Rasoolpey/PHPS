# The DC side's C simulation data (REALTIME_TODO START HERE item 1; gate 54): the replays of all 24 DC sides at the
# build's h (106,000 steps each, states carried), three synthetic replays (a rejection then recovery, held: the sticky flag
# across batches; the continuous mode's clip in and out with a rejection; a two-branch instance: padding), the edge /
# random set, with the Julia @hls results (Float64) of every step, in one binary file; the coefficient header of all sets;
# the testbench. Then scripts/realtime/csim_dcside.sh runs the C simulation.
#   julia --project=. scripts/realtime/dcside_csim_data.jl      (after PHPS.RT.dcside_export(): the same types and h)
using PHPS, JLD2, Random, Printf
const RT = PHPS.RT
root = pwd()
cpp = joinpath(root, "src", "rt", "converters", "CPP", "dcside")
od = joinpath(root, "outputs", "realtime", "converters")
bin = joinpath(od, "dcside_tb.bin")

# h and the types exactly as dcside_export
h = setprecision(() -> big(RT.DCSIDE_II) * 8 / big(10)^9, BigFloat, 256)
setprecision(() -> RT.dcside_types!(h), BigFloat, 256)
n = RT.dcside_ninstances()
ds = [RT.dcside_data(i) for i in 1:n]
d2 = copy(ds[1]); d2["R"] = ds[1]["R"][1:2]; d2["cb"] = ds[1]["cb"][1:2]; d2["name"] *= "_two_branches"   # set 25: padding
sets = vcat(ds, [d2])
const ks = [setprecision(() -> RT.dcside_coefs(d, h), BigFloat, 256) for d in sets]
ih = 1                                               # a held set (GFM_VSM_31)
ic_ = findfirst(d -> d["mode"] == "continuous", ds)  # a continuous set (GFL_31)

open(joinpath(cpp, "dcside_coefs_all.h"), "w") do io
    println(io, "// dcside_coefs_all.h: written by scripts/realtime/dcside_csim_data.jl; do not edit")
    println(io, "// the 24 DC sides of ieee39_dcside.toml at h = $(Float64(h)) s and set 25 = set 1 with two branches (padding)")
    for (i, k) in enumerate(ks)
        println(io, "const DcCoefs DCSIDE_COEFS_$i = ", RT._cinit(k), ";  // ", sets[i]["name"], " (", sets[i]["mode"], ")")
    end
    println(io, "const DcCoefs *DCSIDE_SETS[$(length(ks))] = {", join(("&DCSIDE_COEFS_$i" for i in eachindex(ks)), ", "), "};")
end

# one Julia step; the result row: x (5), xc (4), y (3), yb, packed flags (rejected this step + 2 clipped + 4 sticky)
const NR = 14
# the replay columns 10-13: i_c and a (m in the held mode, the bridge command u_c in the continuous mode; dcside_extract.jl)
function step!(s, x, xc, fl, u)
    y = zeros(3); yb = zeros(1)
    fl[2] = 0; fl[3] = 0
    RT.dcside_step!(x, xc, u, y, yb, fl, ks[s])
    return vcat(x, xc, y, yb, fl[3] + 2fl[2] + 4fl[1])
end
st5(x4) = vcat(x4, 1 / x4[1])

# a replay: (set, x0 (5), inputs (4 x N)) -> the Julia rows (NR x N)
function replay(s, x0, U)
    x = copy(x0); xc = zeros(4); fl = zeros(Int, 3)
    J = zeros(NR, size(U, 2))
    for c in axes(U, 2)
        J[:, c] = step!(s, x, xc, fl, U[:, c])
    end
    return J
end
paths(J) = (rej = count(==(1), Int.(J[14, :]) .& 1), clip = count(!=(0), Int.(J[14, :]) .& 2), sticky = Int(J[14, end]) >> 2)

io = open(bin, "w")
reps = Any[]
for tag in ("vsm", "droop", "voc", "gfl")
    R = load(joinpath(od, "dcside_replay_$tag.jld2"))
    for (q, nm) in enumerate(R["names"])
        s = findfirst(e -> e["case"] == tag && e["name"] == nm, ds); D = R["data"][q]
        U = permutedims(D[:, 10:13])                     # i_c and a per step (columns of U)
        push!(reps, (s, nm, st5(D[1, 2:5]), U, 0))
    end
end
# synthetic 1 (held, set 1): from v = v_min, P_c = 120 pushes v_mid below the admitted interval -> rejected; then 2999
# ordinary steps (the link recovers); the sticky flag must stay set through every later batch
let d = ds[ih], x0 = st5(vcat(0.5, Float64.(d["x0"][2:4])))
    U = zeros(4, 3000); U[:, 1] = [120.0, 0.0, 1.0, 0.0]
    for c in 2:3000; U[:, c] = [5.0 + sin(c / 50), 1.0 + 0.5cos(c / 70), 0.9, 0.1]; end
    push!(reps, (ih, "synthetic: rejection at v_min, then recovery (held)", x0, U, 1))
end
# synthetic 2 (continuous, GFL_31): rejection first (v = v_min, P_c > 0, |u_c| small), then |u_c| swept through m_max v
# (clipped and unclipped in turn), P_c of both signs
let d = ds[ic_], x0 = st5(vcat(0.5, Float64.(d["x0"][2:4]))), mm = d["m_max"]
    U = zeros(4, 3000); U[:, 1] = [120.0, 0.0, 0.5, 0.0]
    for c in 2:3000
        a = mm * 1.08 * (1 + 0.12sin(2pi * c / 400)) * cis(0.3 + c / 900)
        i = (8 + 4sin(2pi * c / 1300)) * cis(0.3 + c / 900 + 0.4 + (c > 1500 ? pi : 0.0))
        U[:, c] = [real(i), imag(i), real(a), imag(a)]
    end
    push!(reps, (ic_, "synthetic: rejection, then the clip in and out, P_c of both signs (continuous)", x0, U, 1))
end
# synthetic 3 (padding, set 25 = set 1 with two branches): the first 3000 inputs of set 1's replay, the third branch 0
let x0 = st5(vcat(Float64.(ds[ih]["x0"][1:3]), 0.0))
    push!(reps, (n + 1, "synthetic: two branches (padding, the third stays 0)", x0, reps[1][4][:, 1:3000], 1))
end

write(io, Int32(length(reps)))
for (s, nm, x0, U, syn) in reps
    J = replay(s, x0, U)
    p = paths(J)
    @printf("replay %-60s set %2d: %6d steps, rejected %d, clipped %d, sticky at the end %d\n", first(nm, 60), s, size(U, 2), p.rej, p.clip, p.sticky)
    syn == 1 && s == n + 1 && (J[4, :] == zeros(size(J, 2)) || error("padding: the third branch moved in Julia"))
    write(io, Int32(s - 1), Int32(size(U, 2))); write(io, x0)
    write(io, vec(vcat(U, J)))                       # per step: u (4), the Julia row (NR)
end

# the edge and random set: (set, x (5), xc (4), sticky, u (4), tag) -> one step; tag 1 = a decision boundary (Float64 and
# the fixed point may fall on opposite sides: flag differences allowed there, outputs compared only where they agree)
edges = Any[]
rng = MersenneTwister(5454)
for (s, d) in enumerate(sets)
    k = ks[s]; x0 = st5(Float64.(d["x0"])); cont = d["mode"] == "continuous"; mm = d["m_max"]
    a0 = cont ? 1.0 : 0.9
    E(x, u, tag; xc = zeros(4), st = 0) = push!(edges, (s, x, xc, st, u, tag))
    alf(x) = x[1] + (sum(k.pv .* x[1:4]) + k.qv[1]) / 2      # alpha of the step (z = 1 / alpha: admitted)
    x0[5] = 1 / alf(x0)
    E(x0, [0.0, 0, 0, 0], 0)                         # zero command and current
    E(x0, [-8.0, 1.0, a0, 0.1], 0)                   # regeneration
    E(x0, [8.0, 1.0, a0, 0.1], 0; st = 1)            # sticky carried in
    for vv in (0.3, 0.5, 0.75, 1.5, 1.99, 2.5)       # the link voltage at and outside the domain
        xv = vcat(vv, x0[2:4], 0.0); xv[5] = 1 / alf(xv)
        E(xv, [2.0, 0.0, 0.5, 0.0], vv in (0.5, 1.99) ? 1 : 0)
    end
    al = alf(x0)
    for e in (-1e-2, -1.01e-3, -1e-3, -0.99e-3, 0.0, 0.99e-3, 1e-3, 1.01e-3, 1e-2)   # the reciprocal's admission
        E(vcat(x0[1:4], (1 + e) / al), [3.0, 0.5, a0, 0.1], abs(e) == 1e-3 ? 1 : 0)
    end
    if cont                                           # the clip boundary at the candidate's own v_mid
        ic = 2.0 + 0.5im; dirn = cis(0.3); sb = mm * al
        for _ in 1:60
            pcb = sb * real(dirn * conj(ic))
            sb = mm * (al + sqrt(al^2 + 2 * k.rv[1] * pcb)) / 2      # 4 rho P_c, rho = r_1 / 2
        end
        for e in (-1e-6, -1e-9, -2.0^-40, 0.0, 2.0^-40, 1e-9, 1e-6)
            a = sb * (1 + e) * dirn
            E(x0, [real(ic), imag(ic), real(a), imag(a)], abs(e) <= 1e-9 ? 1 : 0)
        end
    end
    for _ in 1:300                                   # random states and inputs
        vv = 0.45 + 1.6rand(rng)
        x = vcat(vv, x0[2:4] .* (1 .+ 0.2 .* (2 .* rand(rng, 3) .- 1)), 0.0)
        x[5] = (1 + 3e-3 * (2rand(rng) - 1)) / alf(x)
        xc = 1e-14 .* (2 .* rand(rng, 4) .- 1)
        i = 30rand(rng) * cispi(2rand(rng)); a = (cont ? 2.5 : 1.2) * rand(rng) * cispi(2rand(rng))
        E(x, [real(i), imag(i), real(a), imag(a)], 0; xc, st = rand(rng) < 0.1 ? 1 : 0)
    end
end
write(io, Int32(length(edges)))
cnt = Dict{Symbol,Int}(:rej => 0, :clip => 0, :unclip => 0, :held => 0)
for (s, x, xc, st, u, tag) in edges
    fl = [st, 0, 0]
    J = step!(s, copy(x), copy(xc), fl, u)
    f = Int(J[14])
    sets[s]["mode"] == "continuous" ? (f & 1 == 1 ? cnt[:rej] += 1 : f & 2 == 2 ? cnt[:clip] += 1 : cnt[:unclip] += 1) :
                                      (f & 1 == 1 ? cnt[:rej] += 1 : cnt[:held] += 1)
    write(io, Int32(s - 1), Int32(tag), Int32(st)); write(io, x); write(io, xc); write(io, u); write(io, J)
end
close(io)
@printf("edge / random: %d steps over %d sets: held accepted %d, continuous unclipped %d, clipped %d, rejected %d; tagged %d\n",
        length(edges), length(sets), cnt[:held], cnt[:unclip], cnt[:clip], cnt[:rej], count(e -> e[6] == 1, edges))
@printf("written %s (%.1f MB)\n", bin, filesize(bin) / 2^20)

# the testbench
NB = RT.DCSIDE_NB
open(joinpath(cpp, "dcside_tb.cpp"), "w") do io
    print(io, """
// dcside_tb.cpp: written by scripts/realtime/dcside_csim_data.jl; do not edit
// C simulation of the DC side (src/rt/converters/dcside.jl) against the same @hls code in Julia (Float64), gate 54:
// (1) the replays (the 24 DC sides, 106,000 steps each at the build's h, and three synthetic ones), states carried in the
// C++ types from the first step; (2) the same in segments of 1, 7, 64, 333, 1000 steps (cycled), the state (x, z, the
// remainders, the sticky flag) passed through double / int at every boundary: bit for bit with (1); (3) the board top
// dcside_run in batches of $NB steps, the state through double between batches: outputs, packed flags and the returned
// state (x, xc, flg) bit for bit with (1); (4) the edge / random set. Limit (fixed before results, as the other blocks):
// 5e-8 absolute + relative on the outputs (u_x, w, i_b), the states, x + xc and z. No flag differences in the replays;
// in the edge set only at the tagged decision boundaries. Returns 1 on failure.
#include <cstdio>
#include <cmath>
#include <cstdint>
#include <vector>
#include "dcside.h"
#include "dcside_coefs_all.h"

static const char *BIN = "$bin";
static const char *TRACE = "$(joinpath(od, "dcside_cpp_trace.bin"))";     // the board top's results, every step (y 3, yb, fo, v)
static const char *BSTATES = "$(joinpath(od, "dcside_cpp_bstates.bin"))"; // the returned state after every batch (x 5, xc 4, flg)
static const double LIM = 5e-8;
static const int NR = $NR;

struct State { St x[5]; Small xc[4]; int fl[3]; };
static void load(State &s, const double *v, int st) { for (int i = 0; i < 5; i++) s.x[i] = v[i]; for (int i = 0; i < 4; i++) s.xc[i] = v[5 + i]; s.fl[0] = st; s.fl[1] = 0; s.fl[2] = 0; }
static void save(const State &s, double *v) { for (int i = 0; i < 5; i++) v[i] = s.x[i].to_double(); for (int i = 0; i < 4; i++) v[5 + i] = s.xc[i].to_double(); }
static double rd(double a, double b) { return std::fabs(a - b) / std::fmax(1.0, std::fabs(b)); }

// one step; out = the C++ row (as Julia's: x 5, xc 4, y 3, yb, packed flags); returns the worst difference to J
static double step(DcCoefs &k, State &s, const double *uin, const double *J, double *out) {
    Sig u[4], y[3]; Pw yb[1];
    for (int i = 0; i < 4; i++) u[i] = uin[i];
    s.fl[1] = 0; s.fl[2] = 0;
    dcside_core(s.x, s.xc, u, y, yb, s.fl, k.pv, k.qv, k.rv, k.pb, k.qb, k.rbr, k.kq, k.kc, k.kg);
    save(s, out);
    for (int i = 0; i < 3; i++) out[9 + i] = y[i].to_double();
    out[12] = yb[0].to_double();
    out[13] = s.fl[2] + 2 * s.fl[1] + 4 * s.fl[0];
    double w = 0;
    for (int i = 0; i < 5; i++) w = std::fmax(w, std::fabs(out[i] - J[i]));                       // x, z
    for (int i = 0; i < 4; i++) w = std::fmax(w, std::fabs((out[i] + out[5 + i]) - (J[i] + J[5 + i])));   // x + xc
    for (int i = 9; i < 13; i++) w = std::fmax(w, rd(out[i], J[i]));                              // u_x, w, i_b
    return w;
}

int main() {
    FILE *f = std::fopen(BIN, "rb");
    if (!f) { std::printf("C sim check: cannot open %s\\n", BIN); return 1; }
    FILE *tr = std::fopen(TRACE, "wb");
    FILE *bs = std::fopen(BSTATES, "wb");
    int32_t nrep; std::fread(&nrep, 4, 1, f);
    bool ok = true; double wall = 0; long nall = 0;
    static Sig U[$NB][4], Y[$NB][3]; static Pw YB[$NB]; static int FO[$NB];
    for (int j = 0; j < nrep; j++) {
        int32_t set, ns; std::fread(&set, 4, 1, f); std::fread(&ns, 4, 1, f);
        double init[9] = {0}; std::fread(init, 8, 5, f);
        std::vector<double> rec((4 + NR) * (size_t)ns);
        std::fread(rec.data(), 8, rec.size(), f);
        DcCoefs k = *DCSIDE_SETS[set];
        // (1) continuous
        State s; load(s, init, 0); int fd = 0; double w = 0;
        std::vector<double> rc(NR * (size_t)ns), bst(10 * (size_t)ns);
        for (int c = 0; c < ns; c++) {
            const double *r = &rec[(4 + NR) * (size_t)c];
            w = std::fmax(w, step(k, s, r, r + 4, &rc[NR * (size_t)c]));
            if ((int)rc[NR * (size_t)c + 13] != (int)r[4 + 13]) fd++;
        }
        // (2) segmented: the state through double / int at every boundary
        State t; load(t, init, 0); bool same = true; int c = 0, seg = 0; const int L[5] = {1, 7, 64, 333, 1000};
        double v[10], o2[NR];
        while (c < ns) {
            int e = c + L[seg++ % 5]; if (e > ns) e = ns;
            for (; c < e; c++) {
                step(k, t, &rec[(4 + NR) * (size_t)c], &rec[(4 + NR) * (size_t)c + 4], o2);
                for (int i = 0; i < NR; i++) if (o2[i] != rc[NR * (size_t)c + i]) same = false;
            }
            save(t, v); int st = t.fl[0]; load(t, v, st);
        }
        // (3) the board top in batches, the state through double between batches, the sticky flag through int
        State b; load(b, init, 0); bool samer = true, sames = true; int flg[1] = {0};
        St vo; bool sto, sao;
        for (int c0 = 0; c0 < ns; c0 += $NB) {
            int nb = ns - c0 < $NB ? ns - c0 : $NB;
            for (int q = 0; q < nb; q++) for (int i = 0; i < 4; i++) U[q][i] = rec[(4 + NR) * (size_t)(c0 + q) + i];
            dcside_run(k, b.x, b.xc, U, Y, YB, FO, flg, nb, 100, vo, sto, sao);
            for (int q = 0; q < nb; q++) {
                const double *J = &rc[NR * (size_t)(c0 + q)];
                if (Y[q][0].to_double() != J[9] || Y[q][1].to_double() != J[10] || Y[q][2].to_double() != J[11] ||
                    YB[q].to_double() != J[12] || FO[q] != ((int)J[13] & 3)) samer = false;
                if (tr) { double o[6] = {Y[q][0].to_double(), Y[q][1].to_double(), Y[q][2].to_double(), YB[q].to_double(), (double)FO[q], J[0]};
                          std::fwrite(o, 8, 6, tr); }   // J[0]: v after the step (the continuous C++ run, = the board top's)
            }
            save(b, v); v[9] = flg[0];
            const double *J = &rc[NR * (size_t)(c0 + nb - 1)];
            for (int i = 0; i < 9; i++) if (v[i] != J[i]) sames = false;
            if (flg[0] != ((int)J[13] >> 2)) sames = false;
            if (bs) std::fwrite(v, 8, 10, bs);
            load(b, v, 0);
        }
        save(s, v);
        bool fin = true; for (int i = 0; i < 9; i++) if (v[i] != rc[NR * (size_t)(ns - 1) + i]) fin = false;
        same = same && samer && sames && fin;
        bool pj = w <= LIM && same && fd == 0;
        std::printf("C sim check: replay %2d (set %2d), %6d steps: max diff %.3e (limit %.0e), flag differences %d, continuous = segmented = board top in batches (outputs, flags, returned states) %s: %s\\n",
                    j + 1, set + 1, ns, w, LIM, fd, same ? "bit for bit" : "NO", pj ? "PASS" : "FAIL");
        ok = ok && pj; wall = std::fmax(wall, w); nall += ns;
    }
    int32_t ne; std::fread(&ne, 4, 1, f);
    double we = 0; int fde = 0, fdu = 0;
    for (int q = 0; q < ne; q++) {
        int32_t set, tag, st; std::fread(&set, 4, 1, f); std::fread(&tag, 4, 1, f); std::fread(&st, 4, 1, f);
        double init[9], u[4], J[NR], o[NR]; std::fread(init, 8, 9, f); std::fread(u, 8, 4, f); std::fread(J, 8, NR, f);
        DcCoefs k = *DCSIDE_SETS[set];
        State s; load(s, init, st);
        double w = step(k, s, u, J, o);
        bool fdiff = (int)o[13] != (int)J[13];
        if (fdiff) { if (tag) fde++; else fdu++; }
        if (!fdiff) we = std::fmax(we, w);
    }
    std::fclose(f);
    if (tr) std::fclose(tr);
    if (bs) std::fclose(bs);
    bool pe = we <= LIM && fdu == 0;
    std::printf("C sim check: %d edge / random steps over 25 sets: max diff %.3e (limit %.0e; where the flags agree), flag differences %d at boundary cases, %d elsewhere: %s\\n",
                ne, we, LIM, fde, fdu, pe ? "PASS" : "FAIL");
    ok = ok && pe;
    std::printf("C sim check: %s (%ld replay steps; max %.3e)\\n", ok ? "ALL PASS" : "FAIL", nall, std::fmax(wall, we));
    return ok ? 0 : 1;
}
""")
end
println("written ", joinpath(cpp, "dcside_tb.cpp"), " and dcside_coefs_all.h")
