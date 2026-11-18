// dcside.h: written by PHPS.RT.hls_write from the Julia model (@hls); do not edit
// the converters' DC side (PHConverterDC), one step of h = 6.08e-7 s; from src/rt/converters/dcside.jl
#pragma once
#include <ap_fixed.h>
#include <hls_math.h>

typedef ap_fixed<48, 8, AP_RND_CONV> Sig;
typedef ap_fixed<48, 3, AP_RND_CONV> St;
typedef ap_fixed<48, -14, AP_RND_CONV> Small;
typedef ap_fixed<48, 10, AP_RND_CONV> Coef;
typedef ap_fixed<48, 4, AP_RND_CONV> Rq;
typedef ap_fixed<48, 10, AP_RND_CONV> Pw;
typedef ap_fixed<48, -5, AP_RND_CONV> Lk;
typedef ap_fixed<48, -13, AP_RND_CONV> Lr;
typedef ap_fixed<48, -17, AP_RND_CONV> Br;
typedef ap_fixed<48, -22, AP_RND_CONV> Brr;
typedef ap_fixed<48, 22, AP_RND_CONV> Rinv;
typedef ap_fixed<48, -6, AP_RND_CONV> Inc;
typedef ap_fixed<48, -18, AP_RND_CONV> Ib;
typedef ap_fixed<48, -3, AP_RND_CONV> Ep;
typedef ap_fixed<48, 3, AP_RND_CONV> Fs;
typedef ap_ufixed<18, 2> Sd;

#include "rt_util.h"
#include "dcside_rom.h"

struct DcCoefs {
    Lk pv[4];  // the link row of P
    Lk qv[1];  // the link row of q
    Lr rv[1];  // the link row of r
    Br pb[3][4];  // the branch rows of P
    Br qb[3];  // the branch rows of q
    Brr rbr[3];  // the branch rows of r over rho (r_b / rho: the branches' increments from delta = rho w)
    Rinv kq[1];  // 1 / rho (w = delta / rho, an output only)
    Coef kc[3];  // 1 / Rb, Eb, unused (the battery current i_b = (Eb - sum v_k - v) / Rb)
    Sig kg[6];  // m_max, m_max^2, v_min, v_max, the mode (1 continuous, 0 held), 1 / m_max
};

Sig vs_rsqrt(Sig a);
void dcside_core(St x[5], Small xc[4], Sig u[4], Sig y[3], Pw yb[1], int fl[3], Lk pv[4], Lk qv[1], Lr rv[1], Br pb[3][4], Br qb[3], Brr rbr[3], Rinv kq[1], Coef kc[3], Sig kg[6]);
void dcside_run(const DcCoefs &k, St x[5], Small xc[4], Sig u[512][4], Sig y[512][3], Pw yb[512], int fo[512], int flg[1], int nstep, int decim, volatile St &v_out, volatile bool &strobe, volatile bool &sample);
