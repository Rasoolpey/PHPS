# A converter controller call's C simulation data (droop, VOC): the replay of all six converters (20,000 calls each,
# states carried) and the edge / random set, with the Julia @hls results, in one binary file; the coefficient header of all
# sets; the testbench (continuous, segmented with the state through double, the board top in batches; the edge set).
#   julia --project=. scripts/realtime/conv_csim_data.jl <droop|voc>     (after PHPS.RT.<tag>_rt_export(): the same types)
using PHPS, JLD2, Random, Printf
const P = PHPS
const RT = PHPS.RT
tag = ARGS[1]
root = pwd()
cpp = joinpath(root, "src", "rt", "converters", "CPP", tag)
od = joinpath(root, "outputs", "realtime", "converters")
bin = joinpath(od, "$(tag)_tb.bin")
cx(v) = complex(v[1], v[2])

# ---- the specs: state words (C++ field, type, count), the pairs compared as x + e (indices into the flat state, 0-based
# remainder index or -1), the phase word, the call, the package state -> ours, the replay's v_dc column
if tag == "droop"
    setprecision(() -> RT.droop_types!(), BigFloat, 256)
    nI = RT.droop_ninstances(); ds = [RT.droop_data(i) for i in 1:nI]
    ks = [setprecision(() -> RT.droop_coefs(d), BigFloat, 256) for d in ds]
    FIELDS = [("ph", "Turn", 1), ("pr", "Ang", 1), ("xp", "Pf", 2), ("xu", "St", 1), ("xs", "Sig", 6), ("xv", "Vb", 2), ("xr", "Rm", 11)]
    COMP = [(2 + i, 13 + i) for i in 0:10]           # the eleven filter states as x + e
    VECS = [(3, 4), (5, 6), (7, 8), (9, 10)]          # within COMP (0-based): vm, im, V_v, V_i compared as vectors
    callf = (s, u, y, fl, k) -> RT.droop_call!(s..., u, y, fl, k)
    state = RT.droop_state; VDC = 15; NB = RT.DROOP_NB; CS = "DroopCoefs"; CALL = "droop_call"; RUN = "droop_run"
elseif tag == "voc"
    setprecision(() -> RT.voc_types!(), BigFloat, 256)
    nI = RT.voc_ninstances(); ds = [RT.voc_data(i) for i in 1:nI]
    ks = [setprecision(() -> RT.voc_coefs(d), BigFloat, 256) for d in ds]
    FIELDS = [("ph", "Turn", 1), ("pr", "Ang", 1), ("xw", "Sig", 2), ("xv", "Vb", 2), ("xm", "St", 1), ("xr", "Rm", 4)]
    COMP = [(2, 7), (3, 8), (4, 9), (5, 10), (6, -1)]  # w and V as x + e, m
    VECS = [(0, 1), (2, 3)]                           # w and V compared as vectors
    callf = (s, u, y, fl, k) -> RT.voc_call!(s..., u, y, fl, k)
    state = RT.voc_state; VDC = 8; NB = RT.VOC_NB; CS = "VocCoefs"; CALL = "voc_call"; RUN = "voc_run"
end
NS = sum(f[3] for f in FIELDS)
NR = 2 + NS + 1
off = cumsum([0; [f[3] for f in FIELDS]])
flat(s) = reduce(vcat, s)
unflat(v) = Tuple([v[off[i]+1:off[i+1]] for i in eachindex(FIELDS)])
function call!(si, s, u)
    y = zeros(2); fl = zeros(Int, 5)
    callf(s, u, y, fl, ks[si])
    return vcat(y, flat(s), sum(fl[b] << (b - 1) for b in 1:5))
end

open(joinpath(cpp, "$(tag)_coefs_all.h"), "w") do io
    println(io, "// $(tag)_coefs_all.h: written by scripts/realtime/conv_csim_data.jl; do not edit")
    for (i, k) in enumerate(ks)
        println(io, "const $CS $(uppercase(tag))_COEFS_$i = ", RT._cinit(k), ";  // ", ds[i]["name"])
    end
    println(io, "const $CS *$(uppercase(tag))_SETS[$nI] = {", join(("&$(uppercase(tag))_COEFS_$i" for i in 1:nI), ", "), "};")
end

io = open(bin, "w")
R = load(joinpath(od, "$(tag)_replay.jld2"))
before, vk, ic, clk = R["before"], R["vk"], R["ic"], R["clk"]
ncall, nconv = size(vk)
write(io, Int32(nconv))
cnt = zeros(Int, 5)
for j in 1:nconv
    si = findfirst(e -> e["name"] == R["names"][j], ds)
    s = state(before[1, j, :], clk[1])
    write(io, Int32(si - 1), Int32(ncall)); write(io, flat(s))
    M = zeros(5 + NR, ncall)
    for c in 1:ncall
        r = cis(clk[c])
        u = [real(vk[c, j] * r), imag(vk[c, j] * r), real(ic[c, j] * r), imag(ic[c, j] * r), before[c, j, VDC]]
        M[1:5, c] = u; M[6:end, c] = call!(si, s, u)
        f = Int(M[end, c]); for b in 1:5; cnt[b] += (f >> (b - 1)) & 1; end
    end
    write(io, M)
end
@printf("replay: %d converters x %d calls; flags (bits 1-5) set %s times\n", nconv, ncall, string(cnt))

# the edge / random set: states drawn as the checks' (a), the package state -> ours at a random frame angle; tag 1 at
# decision boundaries (from the checks: the boost / Stage / threshold / clip at equality +-1e-12)
edges = Any[]
rng = MersenneTwister(5858)
for (si, d) in enumerate(ds)
    x0 = Float64.(d["x0"]); lam = cx(d["dlam"]); dcc = cx(d["dcc"])
    E(xc, vkk, icc, tg; c0 = 0.0) = (r = cis(c0); push!(edges, (si, state(xc, c0),
        [real(vkk * r), imag(vkk * r), real(icc * r), imag(icc * r), xc[VDC]], tg)))
    for s in 1:500
        xc = copy(x0); c0 = 2pi * rand(rng) * 50
        if tag == "droop"
            xc[1] = 2pi * rand(rng); xc[2] = d["p_set"] + 3 * (2rand(rng) - 1); xc[3] = d["q_set"] + 2 * (2rand(rng) - 1)
            xc[4] = 0.9 + 0.2rand(rng)
            vm = (0.85 + 0.3rand(rng)) * cis(0.3 * (2rand(rng) - 1)); q = rand(rng)
            imf = (q < 0.4 ? 0.8d["i_lim"] * rand(rng) : q < 0.8 ? d["i_lim"] * (0.9 + 1.5rand(rng)) : 2d["i_con_lim"] * rand(rng)) * cispi(2rand(rng))
            Vv = vm / (1 - lam) * (1 + 0.1 * (2rand(rng) - 1))
            Vi = imf / dcc / (1 - lam) * (1 + 0.1 * (2rand(rng) - 1)); imv = imf / dcc + cx(d["dg"]) * Vi
            xc[5], xc[6], xc[7], xc[8] = real(vm), imag(vm), real(imv), imag(imv)
            xc[11], xc[12], xc[13], xc[14] = real(Vv), imag(Vv), real(Vi), imag(Vi)
            xc[15] = 0.8 + 0.4rand(rng)
            E(xc, (q < 0.5 ? 1.0 + 0.2(2rand(rng) - 1) : 1.3rand(rng)) * cispi(2rand(rng)), 25rand(rng) * cispi(2rand(rng)), 0; c0)
        else
            v = (0.85 + 0.35rand(rng)) * cispi(2rand(rng)); xc[1], xc[2] = real(v), imag(v)
            xc[7] = abs(v) * (1 + 0.02 * (2rand(rng) - 1)); q = rand(rng)
            vkk = q < 0.4 ? v - 0.1rand(rng) * cispi(2rand(rng)) : q < 0.8 ? (0.1 + 1.2rand(rng)) * cispi(2rand(rng)) :
                  (0.03 + 0.1rand(rng)) * cispi(2rand(rng))
            icc = 25rand(rng) * cispi(2rand(rng))
            V = icc * conj(v / xc[7]) / (1 - lam) * (1 + 0.1 * (2rand(rng) - 1)); xc[5], xc[6] = real(V), imag(V)
            xc[8] = 0.8 + 0.4rand(rng)
            E(xc, vkk, icc, 0; c0)
        end
    end
    # the decision boundaries (57, 2): at equality and at adjacent representable inputs (one Sig unit 2^-40 along each axis)
    lsb = 2.0^-40
    adj(z) = (z, z + lsb, z - lsb, z + im * lsb, z - im * lsb)
    if tag == "droop"
        xb = copy(x0); xb[11:14] .= 0.0; vk0 = cis(xb[1] + 0.1); ic0 = 3.0 + 1.0im
        setw(z, a, b, w) = (q = w / dcc; z[a] = real(q); z[b] = imag(q); z)
        for lim in (d["i_lim"], d["i_con_lim"]), w in adj(lim * cis(0.4))            # boost, Stage 2: |imf| = limit
            E(setw(copy(xb), 7, 8, w), vk0, ic0, 1)
        end
        zi = setw(copy(xb), 7, 8, 1.0 + 0im)
        for w in adj(xb[4] - d["Zs"] * d["i_con_lim"] * cis(0.7))                     # Stage 1: |u_mag - vmf| = dmax
            E(setw(copy(zi), 5, 6, w), vk0, ic0, 1)
        end
        zc = setw(copy(zi), 5, 6, xb[4] - 0.05); uc = abs(P.droop_output(P.PHDroop(d["name"], d["mp"], d["mq"], d["Tf_p"], d["Tf_q"],
            d["omega_n"], d["Tv"], d["T_V"], d["T_I"], d["Zs"], d["i_lim"], d["i_con_lim"], d["kpr"], d["kpx"], d["p_set"], d["q_set"],
            d["u_set"], d["Ts"], d["dcr"], lam, cx(d["dg"]), dcc, Symbol(d["dcr_on"]), nothing), zc).uc)
        for e in (0.0, lsb, -lsb, 2lsb, -2lsb)                                         # the m clip: v_dc = |u_c| / m_max
            z = copy(zc); z[VDC] = uc / d["m_max"] + e; E(z, vk0, ic0, 1)
        end
    else
        xb = copy(x0); vb = complex(xb[1], xb[2])
        gp = P.PHVoc(d["name"], d["xi"], d["eta"], d["V_nom"], d["omega_n"], cx(d["J"]), d["Zs"], d["i_lim"], d["r_vi"], d["x_vi"],
                     d["kpr"], d["kpx"], d["p_set"], d["q_set"], d["Ts"], d["dcr"], lam, cx(d["dg"]), dcc, nothing)
        th = sqrt(Float64(RT.voc_coefs_big(d).kg[2]))
        for w in adj(vb - th * cis(0.7))                                               # the VI engagement: |D| = threshold
            E(xb, w, 2.0 + 0im, 1)
        end
        vk = 0.95 * vb * cis(0.05); uc = abs(P.vsm_divider(gp, vb, 0.0, vk).uc)
        for e in (0.0, lsb, -lsb, 2lsb, -2lsb)                                         # the m clip
            z = copy(xb); z[VDC] = uc / d["m_max"] + e; E(z, vk, 1.0 + 0im, 1)
        end
        for a in (RT.VOC_VKMIN, RT.VOC_VKMIN + lsb, RT.VOC_VKMIN - lsb, RT.VOC_VK0 + lsb, RT.VOC_VK0 - lsb, 0.0)   # the floors, zero voltage
            E(xb, a * vb / abs(vb), 1.0 + 0im, 1)
        end
        for m in (RT.VOC_MMIN, RT.VOC_MMIN + 2.0^-45, RT.VOC_MMIN - 2.0^-45)            # the m floor (St unit 2^-45)
            z = copy(xb); z[7] = m; E(z, 0.9vb, 1.0 + 0im, 1)
        end
        z = copy(xb); z[1] = z[2] = 0.0; E(z, 0.9vb, 1.0 + 0im, 1)                        # w = 0
    end
    for p in (0.0, 1 - 1e-15, 0.5, 1 / 1024, 1 / 1024 - 1e-15, 511 / 1024 + 1e-15, 1023 / 1024)   # phase edges
        E(x0, 0.9 * cis(0.2), 3.0 + 1.0im, 0; c0 = 2pi * p)
    end
    for vd in (0.3, 0.5, 1.99, 2.5)
        z = copy(x0); z[VDC] = vd; E(z, 0.9 * cis(0.2), 3.0 + 1.0im, vd in (0.5, 1.99) ? 1 : 0)
    end
end
write(io, Int32(length(edges)))
for (si, s, u, tg) in edges
    write(io, Int32(si - 1), Int32(tg)); write(io, flat(s)); write(io, u)
    write(io, call!(si, deepcopy(s), u))
end
close(io)
@printf("written %s: %d converters x %d calls, %d edge / random calls (%.1f MB)\n", bin, nconv, ncall, length(edges), filesize(bin) / 2^20)

ld = join(("    " * join((f[3] == 1 ? "s.$(f[1])[0] = v[$(off[i])];" : "for (int q = 0; q < $(f[3]); q++) s.$(f[1])[q] = v[$(off[i]) + q];")
          for (i, f) in enumerate(FIELDS)), "\n    "))
sv = join(("    " * join((f[3] == 1 ? "v[$(off[i])] = s.$(f[1])[0].to_double();" : "for (int q = 0; q < $(f[3]); q++) v[$(off[i]) + q] = s.$(f[1])[q].to_double();")
          for (i, f) in enumerate(FIELDS)), "\n    "))
args(s) = join(("$s.$(f[1])" for f in FIELDS), ", ")
open(joinpath(cpp, "$(tag)_tb.cpp"), "w") do io
    print(io, """
// $(tag)_tb.cpp: written by scripts/realtime/conv_csim_data.jl; do not edit
// C simulation of the $tag controller call (src/rt/converters/$tag.jl) against the same @hls code in Julia (Float64):
// (1) the replay of the six converters (states carried in the C++ types), (2) the same in segments (1, 7, 64, 333, 1000
// calls, cycled) with the state through double at every boundary: bit for bit, (3) the board top $RUN in batches of $NB
// calls, the state through double between batches: outputs, packed flags and the returned state bit for bit, (4) the
// edge / random set. Limit (fixed before results, as every block): 5e-8 absolute + relative on the held output, the
// phase (modulo one turn) and the states (the compensated ones as x + e). No flag differences in the replay; in the edge
// set only at the tagged boundary cases. Returns 1 on failure.
#include <cstdio>
#include <cmath>
#include <cstdint>
#include <vector>
#include "$tag.h"
#include "$(tag)_coefs_all.h"

static const char *BIN = "$bin";
static const char *TRACE = "$(joinpath(od, "$(tag)_cpp_trace.bin"))";
static const char *BSTATES = "$(joinpath(od, "$(tag)_cpp_bstates.bin"))";
static const double LIM = 5e-8;
static const int NS = $NS, NR = $NR, NB = $NB;

struct State { $(join(("$(f[2]) $(f[1])[$(f[3])];" for f in FIELDS), " ")) };
static void load(State &s, const double *v) {
$ld
}
static void save(const State &s, double *v) {
$sv
}
static double wrapd(double a) { double r = std::fmod(a + 0.5, 1.0); if (r < 0) r += 1.0; return std::fabs(r - 0.5); }
static double rd(double a, double b) { return std::fabs(a - b) / std::fmax(1.0, std::fabs(b)); }
static const int CX[$(length(COMP))] = {$(join((c[1] for c in COMP), ", "))}, CE[$(length(COMP))] = {$(join((c[2] for c in COMP), ", "))};
static double call(const $CS &k, State &s, const double *uin, const double *J, double *out) {
    Sig u[5], y[2]; int fl[5];
    for (int i = 0; i < 5; i++) u[i] = uin[i];
    $CALL($(args("s")), u, y, fl, k);
    out[0] = y[0].to_double(); out[1] = y[1].to_double();
    save(s, out + 2);
    out[NR - 1] = fl[0] + 2 * fl[1] + 4 * fl[2] + 8 * fl[3] + 16 * fl[4];
    double w = std::fmax(rd(out[0], J[0]), rd(out[1], J[1]));
    w = std::fmax(w, wrapd(out[2] - J[2]));                       // the phase
    double a[$(length(COMP))], b[$(length(COMP))]; bool inv[$(length(COMP))] = {false};
    for (int i = 0; i < $(length(COMP)); i++) {
        a[i] = out[2 + CX[i]] + (CE[i] >= 0 ? out[2 + CE[i]] : 0.0); b[i] = J[2 + CX[i]] + (CE[i] >= 0 ? J[2 + CE[i]] : 0.0);
    }
    const int VA[$(length(VECS))] = {$(join((v[1] for v in VECS), ", "))}, VB[$(length(VECS))] = {$(join((v[2] for v in VECS), ", "))};
    for (int q = 0; q < $(length(VECS)); q++) {         // vector states: relative to the vector's magnitude
        double dx = a[VA[q]] - b[VA[q]], dy = a[VB[q]] - b[VB[q]], m = std::sqrt(b[VA[q]] * b[VA[q]] + b[VB[q]] * b[VB[q]]);
        w = std::fmax(w, std::sqrt(dx * dx + dy * dy) / std::fmax(1.0, m)); inv[VA[q]] = inv[VB[q]] = true;
    }
    for (int i = 0; i < $(length(COMP)); i++) if (!inv[i]) w = std::fmax(w, rd(a[i], b[i]));
    return w;
}

int main() {
    FILE *f = std::fopen(BIN, "rb");
    if (!f) { std::printf("C sim check: cannot open %s\\n", BIN); return 1; }
    FILE *tr = std::fopen(TRACE, "wb");
    FILE *bs = std::fopen(BSTATES, "wb");
    int32_t nconv; std::fread(&nconv, 4, 1, f);
    bool ok = true; double wall = 0;
    static Sig U[$NB][5], Y[$NB][2]; static int FO[$NB];
    for (int j = 0; j < nconv; j++) {
        int32_t set, nc; std::fread(&set, 4, 1, f); std::fread(&nc, 4, 1, f);
        double init[NS]; std::fread(init, 8, NS, f);
        std::vector<double> rec((5 + NR) * (size_t)nc);
        std::fread(rec.data(), 8, rec.size(), f);
        const $CS &k = *$(uppercase(tag))_SETS[set];
        State s; load(s, init); int fd = 0; double w = 0;
        std::vector<double> rc(NR * (size_t)nc);
        for (int c = 0; c < nc; c++) {
            const double *r = &rec[(5 + NR) * (size_t)c];
            w = std::fmax(w, call(k, s, r, r + 5, &rc[NR * (size_t)c]));
            if ((int)rc[NR * (size_t)c + NR - 1] != (int)r[5 + NR - 1]) fd++;
        }
        State t; load(t, init); bool same = true; int c = 0, seg = 0; const int L[5] = {1, 7, 64, 333, 1000};
        double v[NS], o2[NR];
        while (c < nc) {
            int e = c + L[seg++ % 5]; if (e > nc) e = nc;
            for (; c < e; c++) {
                call(k, t, &rec[(5 + NR) * (size_t)c], &rec[(5 + NR) * (size_t)c + 5], o2);
                for (int i = 0; i < NR; i++) if (o2[i] != rc[NR * (size_t)c + i]) same = false;
            }
            save(t, v); load(t, v);
        }
        State b; load(b, init); bool samer = true, sames = true; Sig mo; bool sto, sao;
        for (int c0 = 0; c0 < nc; c0 += NB) {
            int nb = nc - c0 < NB ? nc - c0 : NB;
            for (int q = 0; q < nb; q++) for (int i = 0; i < 5; i++) U[q][i] = rec[(5 + NR) * (size_t)(c0 + q) + i];
            $RUN(k, $(args("b")), U, Y, FO, nb, mo, sto, sao);
            for (int q = 0; q < nb; q++) {
                const double *J = &rc[NR * (size_t)(c0 + q)];
                if (Y[q][0].to_double() != J[0] || Y[q][1].to_double() != J[1] || FO[q] != (int)J[NR - 1]) samer = false;
                if (tr) { double o[2] = {Y[q][0].to_double(), Y[q][1].to_double()}; std::fwrite(o, 8, 2, tr); }
            }
            save(b, v);
            for (int i = 0; i < NS; i++) if (v[i] != rc[NR * (size_t)(c0 + nb - 1) + 2 + i]) sames = false;
            if (bs) std::fwrite(v, 8, NS, bs);
            load(b, v);
        }
        same = same && samer && sames;
        bool pj = w <= LIM && same && fd == 0;
        std::printf("C sim check: converter %d (set %d), %d calls: max diff %.3e (limit %.0e), flag differences %d, continuous = segmented = board top in batches (outputs, flags, returned states) %s: %s\\n",
                    j + 1, set + 1, nc, w, LIM, fd, same ? "bit for bit" : "NO", pj ? "PASS" : "FAIL");
        ok = ok && pj; wall = std::fmax(wall, w);
    }
    int32_t ne; std::fread(&ne, 4, 1, f);
    double we = 0; int fde = 0, fdu = 0;
    for (int q = 0; q < ne; q++) {
        int32_t set, tg; std::fread(&set, 4, 1, f); std::fread(&tg, 4, 1, f);
        double init[NS], u[5], J[NR], o[NR]; std::fread(init, 8, NS, f); std::fread(u, 8, 5, f); std::fread(J, 8, NR, f);
        State s; load(s, init);
        double w = call(*$(uppercase(tag))_SETS[set], s, u, J, o);
        bool fdiff = (int)o[NR - 1] != (int)J[NR - 1];
        if (fdiff) { if (tg) fde++; else fdu++; }
        we = std::fmax(we, w);                         // the error counts at the boundaries too (57, 2): the maps are continuous there
    }
    std::fclose(f);
    if (tr) std::fclose(tr);
    if (bs) std::fclose(bs);
    bool pe = we <= LIM && fdu == 0;
    std::printf("C sim check: %d edge / random calls over %d sets: max diff %.3e (limit %.0e), flag differences %d at boundary cases, %d elsewhere: %s\\n",
                ne, $nI, we, LIM, fde, fdu, pe ? "PASS" : "FAIL");
    ok = ok && pe;
    std::printf("C sim check: %s (max %.3e)\\n", ok ? "ALL PASS" : "FAIL", std::fmax(wall, we));
    return ok ? 0 : 1;
}
""")
end
println("written ", joinpath(cpp, "$(tag)_tb.cpp"), " and $(tag)_coefs_all.h")
