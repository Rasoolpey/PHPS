// hydro.h: written by PHPS.RT.hls_write from the Julia model (@hls); do not edit
// the hydro unit (PHHydroUnit), one run of steps of h = 1.92e-7 s; from src/rt/governors/hydro.jl
#pragma once
#include <ap_fixed.h>
#include <hls_math.h>

typedef ap_fixed<48, 6, AP_RND_CONV> St;
typedef ap_fixed<48, 8, AP_RND_CONV> Sig;
typedef ap_fixed<48, -7, AP_RND_CONV> Small;
typedef ap_fixed<48, -4, AP_RND_CONV> Inc;
typedef ap_fixed<48, 8, AP_RND_CONV> Coef;
typedef ap_fixed<48, -20, AP_RND_CONV> Bt;
typedef ap_fixed<48, 4, AP_RND_CONV> Rq;
typedef ap_fixed<48, 8, AP_RND_CONV> Rr;
typedef ap_fixed<48, 11, AP_RND_CONV> Cw;
typedef ap_fixed<48, 13, AP_RND_CONV> Gc;
typedef ap_fixed<48, 8, AP_RND_CONV> Pw;

#include "rt_util.h"
#include "hydro_rom.h"

struct HydroCoefs {
    Small P[16][9];  // bank = 1 + 8 A_in + 4 X_in + 2 NX + NA: h M J, row by row
    Small W[16][3];  // (omega - 1)
    Small K[16][3];  // Pref
    Small EX[16][3];  // the clamped pilot valve xpc
    Small EA[16][3];  // the clamped gate atc
    Coef kc[11];  // 1 Pref, 2 R_base, 3 Sigma + Delta, 4 Delta, 5 Tg, 6 RUP, 7 RDN, 8 GMAX, 9 GMIN, 10 0.05 GMAX, 11 1.05 GMAX
    Bt kb[1];  // b = h / (2 Tw)
    Cw kw[1];  // c at GMIN, (GMAX / GMIN)^2 (0 when GMIN = 0: never valid)
};

Rr hy_recip(Sig a);
int hy_region(St v, Coef lo, Coef hi);
int hy_freeze(St vl, int pos, int neg, Coef lo, Coef hi);
void hydro_step(St x[4], Small xc[4], Sig u[1], Pw y[1], int md[3], Small kP[16][9], Small kW[16][3], Small kK[16][3], Small kEX[16][3], Small kEA[16][3], Coef kc[11], Bt kb[1], Cw kw[1]);
void hydro_run(const HydroCoefs &k, St x[4], Small xc[4], Sig us[4], Small du[4], int n_on, int n_mid, int n_off, int n0, int nsteps, int decim, Pw y[1], int flg[1], volatile St &at_out, volatile Pw &p_out, volatile bool &strobe, volatile bool &sample);
