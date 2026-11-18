// core.h: written by PHPS.RT.hls_write from the Julia model (@hls); do not edit
// the saturating transformer core (PHCore): the discrete-gradient currents; from src/rt/converters/core.jl
#pragma once
#include <ap_fixed.h>
#include <hls_math.h>

typedef ap_fixed<48, 8, AP_RND_CONV> Sig;
typedef ap_fixed<48, 3, AP_RND_CONV> St;
typedef ap_fixed<48, 2, AP_RND_CONV> Turn;
typedef ap_fixed<48, -6, AP_RND_CONV> Ang;
typedef ap_fixed<48, 10, AP_RND_CONV> Coef;
typedef ap_fixed<48, 4, AP_RND_CONV> Rq;
typedef ap_fixed<48, 2, AP_RND_CONV> Rc;
typedef ap_fixed<48, 4, AP_RND_CONV> Fx;
typedef ap_fixed<48, 20, AP_RND_CONV> Kd;
typedef ap_fixed<48, 7, AP_RND_CONV> Wd;
typedef ap_fixed<48, 48, AP_RND_CONV> Pu;
typedef ap_fixed<48, 1, AP_RND_CONV> Pd;
typedef ap_fixed<48, 2, AP_RND_CONV> Dl;
typedef ap_fixed<48, 0, AP_RND_CONV> Zc;
typedef ap_fixed<48, -7, AP_RND_CONV> Small;
typedef ap_fixed<48, 2, AP_RND_CONV> Rn;
typedef ap_fixed<48, 3, AP_RND_CONV> Ry;
typedef ap_fixed<128, 40, AP_RND_CONV> Wx;

#include "rt_util.h"
#include "core_rom.h"

struct CoreCoefs {
    Coef kl[5];  // 1/L2, 1/L1, kappa' = 1/L1 + 1/L2, kappa = kappa' + 1/Lm, kappa / 2
    Pd kc[2];  // c_H = 1 / (L2 kappa), c_S = 1 / (L1 kappa): lambda_l = c_H psi_H + c_S psi_S (<48,1>: up to 0.73)
    Fx lam[10];  // the knots (flux; lam[10] a cap above any flux)
    Sig cur[9];  // the knot currents
    Sig sl[9];  // the slopes
    Sig g1[9];  // cur_s - lam_s / Lm
    Sig g2[9];  // (sl_s - 1/Lm) / 2
};

Wx cr_shift(Wx x, int m);
void core_solve(Fx psi[4], Rc ce[6], Fx lo[4], Fx am[3], int sg[3], const CoreCoefs &k);
void core_unit(Fx z[8], Turn th[1], Sig y[4], int fl[2], const CoreCoefs &k);
void core_run(const CoreCoefs &k, Fx z[256][8], Turn th[256], Sig y[256][4], int fo[256], int n, int decim, volatile Sig &i_out, volatile bool &strobe, volatile bool &sample);
