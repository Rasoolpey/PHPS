// voc_tb.cpp: written by scripts/realtime/conv_csim_data.jl; do not edit
// C simulation of the voc controller call (src/rt/converters/voc.jl) against the same @hls code in Julia (Float64):
// (1) the replay of the six converters (states carried in the C++ types), (2) the same in segments (1, 7, 64, 333, 1000
// calls, cycled) with the state through double at every boundary: bit for bit, (3) the board top voc_run in batches of 256
// calls, the state through double between batches: outputs, packed flags and the returned state bit for bit, (4) the
// edge / random set. Limit (fixed before results, as every block): 5e-8 absolute + relative on the held output, the
// phase (modulo one turn) and the states (the compensated ones as x + e). No flag differences in the replay; in the edge
// set only at the tagged boundary cases. Returns 1 on failure.
#include <cstdio>
#include <cmath>
#include <cstdint>
#include <vector>
#include "voc.h"
#include "voc_coefs_all.h"

static const char *BIN = "/home/pvm8318/Documents/PHPS.jl/outputs/realtime/converters/voc_tb.bin";
static const char *TRACE = "/home/pvm8318/Documents/PHPS.jl/outputs/realtime/converters/voc_cpp_trace.bin";
static const char *BSTATES = "/home/pvm8318/Documents/PHPS.jl/outputs/realtime/converters/voc_cpp_bstates.bin";
static const double LIM = 5e-8;
static const int NS = 11, NR = 14, NB = 256;

struct State { Turn ph[1]; Ang pr[1]; Sig xw[2]; Vb xv[2]; St xm[1]; Rm xr[4]; };
static void load(State &s, const double *v) {
    s.ph[0] = v[0];s.pr[0] = v[1];for (int q = 0; q < 2; q++) s.xw[q] = v[2 + q];for (int q = 0; q < 2; q++) s.xv[q] = v[4 + q];s.xm[0] = v[6];for (int q = 0; q < 4; q++) s.xr[q] = v[7 + q];
    
}
static void save(const State &s, double *v) {
    v[0] = s.ph[0].to_double();v[1] = s.pr[0].to_double();for (int q = 0; q < 2; q++) v[2 + q] = s.xw[q].to_double();for (int q = 0; q < 2; q++) v[4 + q] = s.xv[q].to_double();v[6] = s.xm[0].to_double();for (int q = 0; q < 4; q++) v[7 + q] = s.xr[q].to_double();
    
}
static double wrapd(double a) { double r = std::fmod(a + 0.5, 1.0); if (r < 0) r += 1.0; return std::fabs(r - 0.5); }
static double rd(double a, double b) { return std::fabs(a - b) / std::fmax(1.0, std::fabs(b)); }
static const int CX[5] = {2, 3, 4, 5, 6}, CE[5] = {7, 8, 9, 10, -1};
static double call(const VocCoefs &k, State &s, const double *uin, const double *J, double *out) {
    Sig u[5], y[2]; int fl[5];
    for (int i = 0; i < 5; i++) u[i] = uin[i];
    voc_call(s.ph, s.pr, s.xw, s.xv, s.xm, s.xr, u, y, fl, k);
    out[0] = y[0].to_double(); out[1] = y[1].to_double();
    save(s, out + 2);
    out[NR - 1] = fl[0] + 2 * fl[1] + 4 * fl[2] + 8 * fl[3] + 16 * fl[4];
    double w = std::fmax(rd(out[0], J[0]), rd(out[1], J[1]));
    w = std::fmax(w, wrapd(out[2] - J[2]));                       // the phase
    double a[5], b[5]; bool inv[5] = {false};
    for (int i = 0; i < 5; i++) {
        a[i] = out[2 + CX[i]] + (CE[i] >= 0 ? out[2 + CE[i]] : 0.0); b[i] = J[2 + CX[i]] + (CE[i] >= 0 ? J[2 + CE[i]] : 0.0);
    }
    const int VA[2] = {0, 2}, VB[2] = {1, 3};
    for (int q = 0; q < 2; q++) {         // vector states: relative to the vector's magnitude
        double dx = a[VA[q]] - b[VA[q]], dy = a[VB[q]] - b[VB[q]], m = std::sqrt(b[VA[q]] * b[VA[q]] + b[VB[q]] * b[VB[q]]);
        w = std::fmax(w, std::sqrt(dx * dx + dy * dy) / std::fmax(1.0, m)); inv[VA[q]] = inv[VB[q]] = true;
    }
    for (int i = 0; i < 5; i++) if (!inv[i]) w = std::fmax(w, rd(a[i], b[i]));
    return w;
}

int main() {
    FILE *f = std::fopen(BIN, "rb");
    if (!f) { std::printf("C sim check: cannot open %s\n", BIN); return 1; }
    FILE *tr = std::fopen(TRACE, "wb");
    FILE *bs = std::fopen(BSTATES, "wb");
    int32_t nconv; std::fread(&nconv, 4, 1, f);
    bool ok = true; double wall = 0;
    static Sig U[256][5], Y[256][2]; static int FO[256];
    for (int j = 0; j < nconv; j++) {
        int32_t set, nc; std::fread(&set, 4, 1, f); std::fread(&nc, 4, 1, f);
        double init[NS]; std::fread(init, 8, NS, f);
        std::vector<double> rec((5 + NR) * (size_t)nc);
        std::fread(rec.data(), 8, rec.size(), f);
        const VocCoefs &k = *VOC_SETS[set];
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
            voc_run(k, b.ph, b.pr, b.xw, b.xv, b.xm, b.xr, U, Y, FO, nb, mo, sto, sao);
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
        std::printf("C sim check: converter %d (set %d), %d calls: max diff %.3e (limit %.0e), flag differences %d, continuous = segmented = board top in batches (outputs, flags, returned states) %s: %s\n",
                    j + 1, set + 1, nc, w, LIM, fd, same ? "bit for bit" : "NO", pj ? "PASS" : "FAIL");
        ok = ok && pj; wall = std::fmax(wall, w);
    }
    int32_t ne; std::fread(&ne, 4, 1, f);
    double we = 0; int fde = 0, fdu = 0;
    for (int q = 0; q < ne; q++) {
        int32_t set, tg; std::fread(&set, 4, 1, f); std::fread(&tg, 4, 1, f);
        double init[NS], u[5], J[NR], o[NR]; std::fread(init, 8, NS, f); std::fread(u, 8, 5, f); std::fread(J, 8, NR, f);
        State s; load(s, init);
        double w = call(*VOC_SETS[set], s, u, J, o);
        bool fdiff = (int)o[NR - 1] != (int)J[NR - 1];
        if (fdiff) { if (tg) fde++; else fdu++; }
        we = std::fmax(we, w);                         // the error counts at the boundaries too (57, 2): the maps are continuous there
    }
    std::fclose(f);
    if (tr) std::fclose(tr);
    if (bs) std::fclose(bs);
    bool pe = we <= LIM && fdu == 0;
    std::printf("C sim check: %d edge / random calls over %d sets: max diff %.3e (limit %.0e), flag differences %d at boundary cases, %d elsewhere: %s\n",
                ne, 6, we, LIM, fde, fdu, pe ? "PASS" : "FAIL");
    ok = ok && pe;
    std::printf("C sim check: %s (max %.3e)\n", ok ? "ALL PASS" : "FAIL", std::fmax(wall, we));
    return ok ? 0 : 1;
}
