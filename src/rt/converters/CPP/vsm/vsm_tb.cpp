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

static const char *BIN = "/home/pvm8318/Documents/PHPS.jl/outputs/realtime/converters/vsm_tb.bin";
static const char *TRACE = "/home/pvm8318/Documents/PHPS.jl/outputs/realtime/converters/vsm_cpp_trace.bin";   // the C++ replay, every call (for the figures)
static const char *BSTATES = "/home/pvm8318/Documents/PHPS.jl/outputs/realtime/converters/vsm_cpp_bstates.bin";  // the board top's returned states after every batch (for the board)
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
    if (!f) { std::printf("C sim check: cannot open %s\n", BIN); return 1; }
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
        std::printf("C sim check: converter %d (set %d), %d calls: max diff %.3e (limit %.0e), flag differences %d, continuous = segmented = board top in batches (outputs, flags, returned states) %s: %s\n",
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
    std::printf("C sim check: %d edge / random calls over 7 sets: max diff %.3e (limit %.0e), flag differences %d at boundary cases, %d elsewhere: %s\n",
                ne, we, LIM, fde, fdu, pe ? "PASS" : "FAIL");
    ok = ok && pe;
    std::printf("C sim check: %s (max %.3e)\n", ok ? "ALL PASS" : "FAIL", std::fmax(wall, we));
    return ok ? 0 : 1;
}
