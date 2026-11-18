// dcside_tb.cpp: written by scripts/realtime/dcside_csim_data.jl; do not edit
// C simulation of the DC side (src/rt/converters/dcside.jl) against the same @hls code in Julia (Float64), gate 54:
// (1) the replays (the 24 DC sides, 106,000 steps each at the build's h, and three synthetic ones), states carried in the
// C++ types from the first step; (2) the same in segments of 1, 7, 64, 333, 1000 steps (cycled), the state (x, z, the
// remainders, the sticky flag) passed through double / int at every boundary: bit for bit with (1); (3) the board top
// dcside_run in batches of 512 steps, the state through double between batches: outputs, packed flags and the returned
// state (x, xc, flg) bit for bit with (1); (4) the edge / random set. Limit (fixed before results, as the other blocks):
// 5e-8 absolute + relative on the outputs (u_x, w, i_b), the states, x + xc and z. No flag differences in the replays;
// in the edge set only at the tagged decision boundaries. Returns 1 on failure.
#include <cstdio>
#include <cmath>
#include <cstdint>
#include <vector>
#include "dcside.h"
#include "dcside_coefs_all.h"

static const char *BIN = "/home/pvm8318/Documents/PHPS.jl/outputs/realtime/converters/dcside_tb.bin";
static const char *TRACE = "/home/pvm8318/Documents/PHPS.jl/outputs/realtime/converters/dcside_cpp_trace.bin";     // the board top's results, every step (y 3, yb, fo, v)
static const char *BSTATES = "/home/pvm8318/Documents/PHPS.jl/outputs/realtime/converters/dcside_cpp_bstates.bin"; // the returned state after every batch (x 5, xc 4, flg)
static const double LIM = 5e-8;
static const int NR = 14;

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
    if (!f) { std::printf("C sim check: cannot open %s\n", BIN); return 1; }
    FILE *tr = std::fopen(TRACE, "wb");
    FILE *bs = std::fopen(BSTATES, "wb");
    int32_t nrep; std::fread(&nrep, 4, 1, f);
    bool ok = true; double wall = 0; long nall = 0;
    static Sig U[512][4], Y[512][3]; static Pw YB[512]; static int FO[512];
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
        for (int c0 = 0; c0 < ns; c0 += 512) {
            int nb = ns - c0 < 512 ? ns - c0 : 512;
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
        std::printf("C sim check: replay %2d (set %2d), %6d steps: max diff %.3e (limit %.0e), flag differences %d, continuous = segmented = board top in batches (outputs, flags, returned states) %s: %s\n",
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
    std::printf("C sim check: %d edge / random steps over 25 sets: max diff %.3e (limit %.0e; where the flags agree), flag differences %d at boundary cases, %d elsewhere: %s\n",
                ne, we, LIM, fde, fdu, pe ? "PASS" : "FAIL");
    ok = ok && pe;
    std::printf("C sim check: %s (%ld replay steps; max %.3e)\n", ok ? "ALL PASS" : "FAIL", nall, std::fmax(wall, we));
    return ok ? 0 : 1;
}
