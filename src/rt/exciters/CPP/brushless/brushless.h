// brushless.h: written by PHPS.RT.hls_write from the Julia model (@hls); do not edit
// the brushless exciter (unsaturated PHBrushlessExciter), one run of steps of h = 1.92e-7 s; from src/rt/exciters/brushless.jl
#pragma once
#include <ap_fixed.h>
#include <hls_math.h>

typedef ap_fixed<48, 6, AP_RND_CONV> St;
typedef ap_fixed<48, 8, AP_RND_CONV> Sig;
typedef ap_fixed<48, -7, AP_RND_CONV> Small;
typedef ap_fixed<48, -4, AP_RND_CONV> Inc;
typedef ap_fixed<48, 8, AP_RND_CONV> Coef;
typedef ap_fixed<48, -8, AP_RND_CONV> Tiny;
typedef ap_fixed<48, 4, AP_RND_CONV> Rq;
typedef ap_fixed<48, 2, AP_RND_CONV> Pw;
typedef ap_fixed<48, 10, AP_RND_CONV> Wd;

#include "rt_util.h"
#include "brushless_rom.h"

struct BrushlessCoefs {
    Small P[4][9];  // banks 1 free-L, 2 free-sat, 3 frozen-L, 4 frozen-sat: (I - hA/2)^-1 h A, row by row
    Small W[4][6];  // inputs (vterm, IFD), row by row
    Small c[4][3];  // the constant (Vref) term
    Small wmax[2][3];  // sat banks (free, frozen): w VRMAX
    Small wmin[2][3];  // and w VRMIN
    Coef kc[15];  // 1 KA, 2 c1 = -KA KF KE / TF, 3 c3 = KA KF / TF, 4 KA KF KD / TF, 5 KE, 6 KD, 7 Vref, 8 VRMAX,
    Tiny kt[3];  // 1 Rfd / Lad, 2 g, 3 g KD
};

Sig bx_sqrt(Sig a);
void brushless_step(St x[3], Sig u[3], Pw y[2], int md[3], Small kP[4][9], Small kW[4][6], Small kk[4][3], Small kwx[2][3], Small kwn[2][3], Coef kc[15], Tiny kt[3]);
void brushless_run(const BrushlessCoefs &k, St x[3], Sig us[4][3], Small du[4][3], int n_on, int n_mid, int n_off, int n0, int nsteps, int decim, Pw y[2], volatile St &xa_out, volatile Pw &vf_out, volatile bool &strobe, volatile bool &sample);
