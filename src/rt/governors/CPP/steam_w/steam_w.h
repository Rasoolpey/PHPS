// steam_w.h: written by PHPS.RT.hls_write from the Julia model (@hls); do not edit
// Body W of the steam unit (the wet-capable slot, II 110 on 3 multipliers); src/rt/governors/steam.jl
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
typedef ap_fixed<48, 6, AP_RND_CONV> Tx;

#include "rt_util.h"
#include "steam_w_rom.h"

struct SteamWCoefs {
    Cf vt;  // V / Mref
    Cf cx;  // log Mref - log V
    int xlo;  // the X window (as body A)
    int xhi;
    Cf xflo;
    Cf xfhi;
    Cf xtlo;
    Cf xthi;
    Wt xcl;
    Wt xch;
    int ylo;  // the xi window
    int yhi;
    Cf yflo;
    Cf yfhi;
    Cf ytlo;
    Cf ythi;
    Wt ycl;
    Wt ych;
    Cf gx0;  // the X and xi axes, the clear test (as body A)
    Cf gidx;
    Cf vy0;
    Cf vidy;
    Cf eps1;
    Cf sx0;  // the saturation table's l axis: x0, 1 / dx, the in-table limits of t (tolerances included)
    Cf sidx;
    Cf stlo;
    Cf sthi;
    Sig vlo;  // v_g at l_max and at l_min (v / 16)
    Sig vhi;
    Cf l2e;  // log2 e; 11 - log2 1e6
    Cf l2c;
};

int sta_norm(Sig a, Rq mo[1]);
Sig sta_sh(Sig x, int k);
int sta_j(Rq m);
Rq sta_rcp(Rq m, Rq w, Rq e);
bool sta_cell(Tc tt, int lo, int hi, Cf flo, Cf fhi, Cf tlo, Cf thi, Wt cl, Wt ch, int ci[1], Wt ct[1]);
void sta_basis(Wt t, Wt w[4]);
Tw sta_v2(Tw F[6048], Tw X[6048], Tw Y[6048], Tw Z[6048], int b, Wt a0, Wt a1, Wt c0, Wt c1, Wt b0, Wt b1, Wt d0, Wt d1);
int stb_norm(Sig a, Rq mo[1]);
Sig stb_sh(Sig x, int n);
bool stw_cell(Tc tt, Cf tlo, Cf thi, int ci[1], Wt ct[1]);
void stw_dbasis(Wt t, Wt w[3]);
Sig stw_srcp(Sig a, int nk[1]);
Sig stw_exp(Sig l, Cf l2e, Cf l2c);
void steam_w_eval(const SteamWCoefs &k, Tw VF[504], Tw VX[504], Tw VY[504], Tw VZ[504], St Mi, St Ui, Sig lg0, Sig l0, Sig o[8], int fo[2]);
void steam_w_run(const SteamWCoefs &k, int mode, int la, Tw lw, St Mi, St Ui, Sig lg0, Sig l0, Sig o[8], int fo[2]);
