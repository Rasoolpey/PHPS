// shunt.h: written by PHPS.RT.hls_write from the Julia model (@hls); do not edit
// the DC shunt exciter (saturated PHShuntExciter), one run of steps of h = 2.16e-7 s; from src/rt/exciters/shunt.jl
#pragma once
#include <ap_fixed.h>
#include <hls_math.h>

typedef ap_fixed<48, 6, AP_RND_CONV> St;
typedef ap_fixed<48, 8, AP_RND_CONV> Sig;
typedef ap_fixed<48, -7, AP_RND_CONV> Small;
typedef ap_fixed<48, -4, AP_RND_CONV> Inc;
typedef ap_fixed<48, 8, AP_RND_CONV> Coef;
typedef ap_fixed<48, -8, AP_RND_CONV> Tiny;
typedef ap_fixed<48, 6, AP_RND_CONV> Targ;
typedef ap_fixed<48, 10, AP_RND_CONV> Frac8;
typedef ap_fixed<48, 3, AP_RND_CONV> Mant;
typedef ap_fixed<48, 4, AP_RND_CONV> Sat;
typedef ap_fixed<48, 2, AP_RND_CONV> Pw;
typedef ap_fixed<48, 10, AP_RND_CONV> Wd;

#include "rt_util.h"
#include "shunt_rom.h"

struct ShuntCoefs {
    Small P[4][9];  // banks 1 free-W, 2 free-sat, 3 frozen-W, 4 frozen-sat: h M J, row by row
    Small W[4][3];  // the vterm column h M G
    Small c[4][3];  // the constant (Vref) term
    Small wmax[2][3];  // sat banks (free, frozen): h M e VRMAX
    Small wmin[2][3];  // and h M e VRMIN
    Small Q[4][3];  // the speed vector h M g
    Small S[4][3];  // the saturation vector -h M e1 / TE
    Coef kc[10];  // 1 KA Vref, 2 KA, 3 c3 = KA KF / TF, 4 VRMAX, 5 VRMIN, 6 Bl = B log2(e), 7 Al = log2(A) + 16,
    Tiny kt[1];  // 1 Rfd / Lad
};

void sh_exp(Targ tp, Mant mk[1], int kk[1]);
Sat sh_se(St yv, Coef kc[10]);
int sh_decide(St xa, Sig r, Coef vmax, Coef vmin);
int sh_select(Inc gw, Inc gh, Inc gl, St umax, St umin);
void shunt_step(St x[5], Sig u[3], Pw y[2], int md[2], Sat se, Small kP[4][9], Small kW[4][3], Small kk[4][3], Small kwx[2][3], Small kwn[2][3], Small kQ[4][3], Small kS[4][3], Coef kc[10], Tiny kt[1]);
void shunt_run(const ShuntCoefs &k, St x[5], Sig us[4][3], Small du[4][3], int n_on, int n_mid, int n_off, int n0, int nsteps, int decim, Pw y[2], volatile St &xa_out, volatile Pw &vf_out, volatile bool &strobe, volatile bool &sample);
