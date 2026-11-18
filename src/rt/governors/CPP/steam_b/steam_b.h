// steam_b.h: written by PHPS.RT.hls_write from the Julia model (@hls); do not edit
// Body B of the steam unit (11 sections at II 10); src/rt/governors/steam.jl
#pragma once
#include <ap_fixed.h>
#include <hls_math.h>

typedef ap_fixed<48, 5, AP_RND_CONV> St;
typedef ap_fixed<48, 8, AP_RND_CONV> Sig;
typedef ap_fixed<48, 2, AP_RND_CONV> Wt;
typedef ap_fixed<48, 8, AP_RND_CONV> Tw;
typedef ap_fixed<48, 10, AP_RND_CONV, AP_SAT> Tc;
typedef ap_fixed<48, 4, AP_RND_CONV> Rq;
typedef ap_fixed<48, 1, AP_RND_CONV> Lw;
typedef ap_fixed<48, -9, AP_RND_CONV> Rs;
typedef ap_fixed<48, 8, AP_RND_CONV> Cf;
typedef ap_fixed<48, 3, AP_RND_CONV> Dq;
typedef ap_ufixed<18, 2> Sd;

#include "rt_util.h"
#include "steam_b_rom.h"

struct SteamBCoefs {
    int ia[11];  // the inlet storage (0-based, storage order)
    int ib[11];  // the outlet storage (terminal: the inlet's, not used)
    int tm[11];  // 1: into the condenser
    int wo[11];  // 1: the item has a wet end
    int ep[11];  // p~ = p / 2^ep
    Sig pm[11];  // the pressure limit 1.375 x 2^ep (1e6 Pa)
    Cf kq[11];  // Kq_j
    Dq d2[11];  // delta~_j^2
    Wt et[11];  // the section's isentropic efficiency
    int llo[11];  // the l window (as body A's cells)
    int lhi[11];
    Cf lflo[11];
    Cf lfhi[11];
    Cf ltlo[11];
    Cf lthi[11];
    Wt lcl[11];
    Wt lch[11];
    int elo[11];  // the eta window
    int ehi[11];
    Cf eflo[11];
    Cf efhi[11];
    Cf etlo[11];
    Cf ethi[11];
    Wt ecl[11];
    Wt ech[11];
    Sig pc;  // p_cond (1e6 Pa)
    Cf lpc;  // log p_cond (Pa)
    Cf sx0;  // the l axis: x0, 1 / dx (saturation, top isentrope and isentrope map share it)
    Cf sidx;
    Cf ey0;  // the eta axis: y0, 1 / dy
    Cf eidy;
};

int sta_j(Rq m);
Rq sta_rcp(Rq m, Rq w, Rq e);
bool sta_cell(Tc tt, int lo, int hi, Cf flo, Cf fhi, Cf tlo, Cf thi, Wt cl, Wt ch, int ci[1], Wt ct[1]);
void sta_basis(Wt t, Wt w[4]);
int stb_norm(Sig a, Rq mo[1]);
int stb_nq(Dq z, Rq mo[1]);
int stb_nr(Dq x, Rq mo[1]);
int stb_jq(Rq m);
int stb_jr(Rq m);
Rq stb_qrt(Rq m, Rq y0);
Rq stb_rsq(Rq m, Rq y0);
Dq stb_ps(Sig x, int e);
Sig stb_sh(Sig x, int n);
Tw stb_v2(Tw F[2717], Tw X[2717], Tw Y[2717], Tw Z[2717], int b, Wt a0, Wt a1, Wt c0, Wt c1, Wt b0, Wt b1, Wt d0, Wt d1);
void steam_b_run(const SteamBCoefs &k, int mode, int la, Tw lw, Sig p[13], Sig hh[13], Sig v[13], Sig s[13], Sig q[11], Sig e1[11], Sig e2[11], int fw[11], int fl[11]);
