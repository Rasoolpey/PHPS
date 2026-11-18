# The VSM call's C simulation data (review 49, 5): the replay of all six converters (20,000 calls each, states carried)
# and the edge / random set, with the Julia @hls results, in one binary file; the coefficient header of all sets; the
# testbench. Then scripts/realtime/csim_vsm.sh runs the C simulation.
#   julia --project=. scripts/realtime/vsm_csim_data.jl
using PHPS, JLD2, Random, TOML, Printf
const RT = PHPS.RT
root = pwd()
cpp = joinpath(root, "src", "rt", "converters", "CPP", "vsm")
bin = joinpath(root, "outputs", "realtime", "converters", "vsm_tb.bin")
OB = TOML.parsefile(joinpath(root, "src", "rt", "converters", "data", "ieee39_vsm.toml"))["omega_b"]
n = RT.vsm_ninstances()
ds = [RT.vsm_data(i) for i in 1:n]
du = copy(ds[1]); du["u_max"] = 1.15; du["name"] *= "_umax1.15"      # set 7: the u_max clip active
sets = vcat(ds, [du])
ks = [RT.vsm_coefs(d) for d in sets]

# the coefficient header: all seven sets
open(joinpath(cpp, "vsm_coefs_all.h"), "w") do io
    println(io, "// vsm_coefs_all.h: written by scripts/realtime/vsm_csim_data.jl; do not edit")
    println(io, "// the six VSM converters of ieee39_vsm.toml and set 7 = set 1 with u_max = 1.15")
    for (i, k) in enumerate(ks)
        println(io, "const VsmCoefs VSM_COEFS_$i = ", RT._cinit(k), ";  // ", sets[i]["name"])
    end
    println(io, "const VsmCoefs *VSM_SETS[$(length(ks))] = {", join(("&VSM_COEFS_$i" for i in eachindex(ks)), ", "), "};")
end

call!(s, x, ph, pr, u) = (y = zeros(2); fl = zeros(Int, 4); RT.vsm_call!(x, ph, pr, u, y, fl, ks[s]); vcat(y, x, ph, pr, Float64.(fl)))

io = open(bin, "w")
R = load(joinpath(root, "outputs", "realtime", "converters", "vsm_replay.jld2"))
before, vk, ic, tc = R["before"], R["vk"], R["ic"], R["tcall"]
ncall, nconv = size(vk)
write(io, Int32(nconv))
for j in 1:nconv
    s = findfirst(e -> e["name"] == R["names"][j], ds)
    x = before[1, j, 2:4]; ph = [mod((before[1, j, 1] + OB * tc[1]) / 2pi, 1.0)]; pr = [0.0]
    write(io, Int32(s - 1), Int32(ncall)); write(io, vcat(x, ph, pr))
    for c in 1:ncall
        r = cis(OB * tc[c])
        u = [real(vk[c, j] * r), imag(vk[c, j] * r), real(ic[c, j] * r), imag(ic[c, j] * r), before[c, j, 7]]
        write(io, u); write(io, call!(s, x, ph, pr, u))
    end
end
# the edge and random set: (set, initial state, input) -> one call
edges = Any[]
rng = MersenneTwister(4949)
for (s, d) in enumerate(sets)
    k = ks[s]; x0 = Float64.(d["x0"])
    th = sqrt(Float64(RT.vsm_coefs_big(d).kg[2]))
    base(phi) = ([x0[2], x0[3], x0[4]], [phi], [0.0])
    E(phi, um) = um * cispi(2phi)
    # D = 0, the threshold, phase wrap and bin edges, v_dc out of domain
    p0 = mod(x0[1] / 2pi, 1.0)
    push!(edges, (s, base(p0)..., (v = E(p0, x0[4]); [real(v), imag(v), 1.0, 0.0, x0[7]]), 0))
    for e in (-1e-9, -1e-12, 0.0, 1e-12, 1e-9)
        v = E(p0, x0[4]) - th * (1 + e) * cis(0.7)
        push!(edges, (s, base(p0)..., [real(v), imag(v), 2.0, 0.0, x0[7]], 1))
    end
    for p in (0.0, prevfloat(1.0), 0.5, 1 / 1024, prevfloat(1 / 1024), 511 / 1024, 1023 / 1024)
        v = 0.6 * cispi(2p - 0.3 / pi)
        push!(edges, (s, base(p)..., [real(v), imag(v), 3.0, 0.0, x0[7]], 0))
    end
    for vd in (0.3, 0.5, 1.99, 2.5)
        push!(edges, (s, base(p0)..., [0.9cospi(2p0), 0.9sinpi(2p0), 1.0, 0.0, vd], vd in (0.5, 1.99) ? 1 : 0))
    end
    for _ in 1:500                                   # random states and inputs (VI off / engaged / bolted fault)
        p = rand(rng); um = 0.9 + 0.3rand(rng)
        x = [1 + 0.03(2rand(rng) - 1), 1 + 0.03(2rand(rng) - 1), um]
        q = rand(rng)
        v = q < 1 / 3 ? E(p, um) - 0.12rand(rng) * cispi(2rand(rng)) : q < 2 / 3 ? 1.3rand(rng) * cispi(2rand(rng)) :
            0.05rand(rng) * cispi(2rand(rng))
        i = 15rand(rng) * cispi(2rand(rng))
        push!(edges, (s, x, [p], [0.0], [real(v), imag(v), real(i), imag(i), 0.8 + 0.4rand(rng)], 0))
    end
end
write(io, Int32(length(edges)))
for (s, x, ph, pr, u, tag) in edges
    write(io, Int32(s - 1), Int32(tag)); write(io, vcat(x, ph, pr)); write(io, u)
    write(io, call!(s, copy(x), copy(ph), copy(pr), u))
end
close(io)
@printf("written %s: %d converters x %d calls, %d edge / random calls (%.1f MB)\n", bin, nconv, ncall, length(edges), filesize(bin) / 2^20)

# the testbench
open(joinpath(cpp, "vsm_tb.cpp"), "w") do io
    print(io, """
// vsm_tb.cpp: written by scripts/realtime/vsm_csim_data.jl; do not edit
// C simulation of the VSM controller call (src/rt/converters/vsm.jl) against the same @hls code in Julia (Float64),
// review 49, 5: (1) the replay of the six converters (20,000 calls each, states carried in the C++ types), (2) the same
// replay in segments with the states passed through double at every boundary (as the processor will): bit for bit,
// (3) the edge / random set. Limit (fixed before results, as the other blocks): 5e-8 absolute + relative on the held
// output and the states, phase modulo one turn. Flags are counted where they differ. Returns 1 on failure.
#include <cstdio>
#include <cmath>
#include <cstdint>
#include <vector>
#include "vsm.h"
#include "vsm_coefs_all.h"

static const char *BIN = "$bin";
static const char *TRACE = "$(replace(bin, "vsm_tb.bin" => "vsm_cpp_trace.bin"))";   // the C++ replay, every call (for the figures)
static const char *BSTATES = "$(replace(bin, "vsm_tb.bin" => "vsm_cpp_bstates.bin"))";  // the board top's returned states after every batch (for the board)
static const double LIM = 5e-8;

static double wrapd(double a) { double r = std::fmod(a + 0.5, 1.0); if (r < 0) r += 1.0; return std::fabs(r - 0.5); }

struct State { St x[3]; Turn ph[1]; Ang pr[1]; };
static void load(State &s, const double *v) { for (int i = 0; i < 3; i++) s.x[i] = v[i]; s.ph[0] = v[3]; s.pr[0] = v[4]; }
static void save(const State &s, double *v) { for (int i = 0; i < 3; i++) v[i] = s.x[i].to_double(); v[3] = s.ph[0].to_double(); v[4] = s.pr[0].to_double(); }

// one call; returns the worst difference to the Julia result J (11 values); counts flag differences
static double call(const VsmCoefs &k, State &s, const double *uin, const double *J, int *fdiff, double *out, int *pk = nullptr) {
    Sig u[5], y[2]; int fl[4];
    for (int i = 0; i < 5; i++) u[i] = uin[i];
    vsm_call(s.x, s.ph, s.pr, u, y, fl, k);
    double w = 0;
    for (int i = 0; i < 2; i++) w = std::fmax(w, std::fabs(y[i].to_double() - J[i]) / std::fmax(1.0, std::fabs(J[i])));
    for (int i = 0; i < 3; i++) w = std::fmax(w, std::fabs(s.x[i].to_double() - J[2 + i]));
    w = std::fmax(w, wrapd(s.ph[0].to_double() - J[5]));
    w = std::fmax(w, std::fabs(s.pr[0].to_double() - J[6]));   // the phase remainder (50, 3)
    for (int i = 0; i < 4; i++) if (fl[i] != (int)J[7 + i]) (*fdiff)++;
    if (out) { out[0] = y[0].to_double(); out[1] = y[1].to_double(); }
    if (pk) *pk = fl[0] + 2 * fl[1] + 4 * fl[2] + 8 * fl[3];
    return w;
}

int main() {
    FILE *f = std::fopen(BIN, "rb");
    if (!f) { std::printf("C sim check: cannot open %s\\n", BIN); return 1; }
    int32_t nconv; std::fread(&nconv, 4, 1, f);
    FILE *tr = std::fopen(TRACE, "wb");
    FILE *bs = std::fopen(BSTATES, "wb");
    bool ok = true; double wall = 0;
    for (int j = 0; j < nconv; j++) {
        int32_t set, ncall; std::fread(&set, 4, 1, f); std::fread(&ncall, 4, 1, f);
        double init[5]; std::fread(init, 8, 5, f);
        std::vector<double> rec(16 * (size_t)ncall);
        std::fread(rec.data(), 8, rec.size(), f);
        const VsmCoefs &k = *VSM_SETS[set];
        // (1) continuous
        State s; load(s, init); int fd = 0; double w = 0;
        std::vector<double> yc(2 * (size_t)ncall), stc(5 * (size_t)ncall); std::vector<int> pfc(ncall);
        for (int c = 0; c < ncall; c++) {
            w = std::fmax(w, call(k, s, &rec[16 * c], &rec[16 * c + 5], &fd, &yc[2 * c], &pfc[c]));
            double *st = &stc[5 * c]; save(s, st);
            if (tr) { std::fwrite(&yc[2 * c], 8, 2, tr); std::fwrite(st, 8, 4, tr); }   // m (2), omega, omega_f, u_mag, phi
        }
        double endc[5]; save(s, endc);
        // (2) segmented: segments of 1, 7, 64, 333, ... calls, the states through double at each boundary
        State t; load(t, init); int fd2 = 0; bool same = true; int c = 0, seg = 0; const int L[5] = {1, 7, 64, 333, 1000};
        double v[5]; double y2[2];
        while (c < ncall) {
            int e = c + L[seg++ % 5]; if (e > ncall) e = ncall;
            for (; c < e; c++) {
                call(k, t, &rec[16 * c], &rec[16 * c + 5], &fd2, y2);
                if (y2[0] != yc[2 * c] || y2[1] != yc[2 * c + 1]) same = false;
            }
            save(t, v); load(t, v);
        }
        double ends[5]; save(t, ends);
        for (int i = 0; i < 5; i++) if (ends[i] != endc[i]) same = false;
        // (4) the board top vsm_run: the replay in batches of 256 calls, the states through double between batches
        State b; load(b, init); bool samer = true, sames = true, samef = true;
        static Sig U[256][5], Y[256][2]; static int FO[256]; Sig mo; bool sto, sao;
        for (int c0 = 0; c0 < ncall; c0 += 256) {
            int nb = ncall - c0 < 256 ? ncall - c0 : 256;
            for (int c = 0; c < nb; c++) for (int i = 0; i < 5; i++) U[c][i] = rec[16 * (c0 + c) + i];
            vsm_run(k, b.x, b.ph, b.pr, U, Y, FO, nb, mo, sto, sao);
            for (int c = 0; c < nb; c++) {
                if (Y[c][0].to_double() != yc[2 * (c0 + c)] || Y[c][1].to_double() != yc[2 * (c0 + c) + 1]) samer = false;
                if (FO[c] != pfc[c0 + c]) samef = false;             // the packed flags (51, 3)
            }
            save(b, v);
            for (int i = 0; i < 5; i++) if (v[i] != stc[5 * (c0 + nb - 1) + i]) sames = false;   // all five returned words
            if (bs) std::fwrite(v, 8, 5, bs);
            load(b, v);
        }
        same = same && samer && sames && samef;
        bool pj = w <= LIM && same && fd == 0;        // no flag difference allowed in the replay (50, 3)
        std::printf("C sim check: converter %d (set %d), %d calls: max diff %.3e (limit %.0e), flag differences %d, continuous = segmented = board top in batches (outputs, flags, returned states) %s: %s\\n",
                    j + 1, set + 1, ncall, w, LIM, fd, same ? "bit for bit" : "NO", pj ? "PASS" : "FAIL");
        ok = ok && pj; wall = std::fmax(wall, w);
    }
    int32_t ne; std::fread(&ne, 4, 1, f);
    double we = 0; int fde = 0, fdu = 0;
    for (int q = 0; q < ne; q++) {
        int32_t set, tag; std::fread(&set, 4, 1, f); std::fread(&tag, 4, 1, f);
        double init[5], u[5], J[11]; std::fread(init, 8, 5, f); std::fread(u, 8, 5, f); std::fread(J, 8, 11, f);
        State s; load(s, init); int fq = 0;
        we = std::fmax(we, call(*VSM_SETS[set], s, u, J, &fq, nullptr));
        if (tag) fde += fq; else fdu += fq;        // differences allowed only at the tagged boundary cases (50, 3)
    }
    std::fclose(f);
    if (tr) std::fclose(tr);
    if (bs) std::fclose(bs);
    bool pe = we <= LIM && fdu == 0;
    std::printf("C sim check: %d edge / random calls over 7 sets: max diff %.3e (limit %.0e), flag differences %d at boundary cases, %d elsewhere: %s\\n",
                ne, we, LIM, fde, fdu, pe ? "PASS" : "FAIL");
    ok = ok && pe;
    std::printf("C sim check: %s (max %.3e)\\n", ok ? "ALL PASS" : "FAIL", std::fmax(wall, we));
    return ok ? 0 : 1;
}
""")
end
println("written ", joinpath(cpp, "vsm_tb.cpp"), " and vsm_coefs_all.h")
