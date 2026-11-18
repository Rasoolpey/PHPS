// steam_a.h: written by PHPS.RT.hls_write from the Julia model (@hls); do not edit
// Body A of the steam unit (12 dry storages at II 9); src/rt/governors/steam.jl
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

#include "rt_util.h"
#include "steam_a_rom.h"

struct SteamACoefs {
    Cf vt[12];  // V_k / Mref_k
    Cf cx[12];  // log Mref_k - log V_k
    int xlo[12];  // the X window: first and last cell (0-based table cells)
    int xhi[12];
    Cf xflo[12];  // first cell, last cell + 1 (as numbers)
    Cf xfhi[12];
    Cf xtlo[12];  // in the window: xtlo <= t <= xthi (the table edge's tolerance included)
    Cf xthi[12];
    Wt xcl[12];  // the coordinate's clamp outside the window: [xcl, xch]
    Wt xch[12];
    int ylo[12];  // the same for xi
    int yhi[12];
    Cf yflo[12];
    Cf yfhi[12];
    Cf ytlo[12];
    Cf ythi[12];
    Wt ycl[12];
    Wt ych[12];
    Cf gx0;  // the X axis (gas table and vapour map): x0, 1 / dx
    Cf gidx;
    Cf vy0;  // the xi axis: y0, 1 / dy
    Cf vidy;
    Cf eps1;  // 1 + 1e-6 (the clear test)
};

int sta_norm(Sig a, Rq mo[1]);
Sig sta_sh(Sig x, int k);
int sta_j(Rq m);
Rq sta_rcp(Rq m, Rq w, Rq e);
bool sta_cell(Tc tt, int lo, int hi, Cf flo, Cf fhi, Cf tlo, Cf thi, Wt cl, Wt ch, int ci[1], Wt ct[1]);
void sta_basis(Wt t, Wt w[4]);
Tw sta_v2(Tw F[6048], Tw X[6048], Tw Y[6048], Tw Z[6048], int b, Wt a0, Wt a1, Wt c0, Wt c1, Wt b0, Wt b1, Wt d0, Wt d1);
void steam_a_run(const SteamACoefs &k, int mode, int la, Tw lw, St M[12], St U[12], Sig p[12], Sig hh[12], Sig v[12], Sig s[12], int fl[12], Wt wo[12][8], int co[12][2]);
