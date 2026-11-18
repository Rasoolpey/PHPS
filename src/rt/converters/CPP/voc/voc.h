// voc.h: written by PHPS.RT.hls_write from the Julia model (@hls); do not edit
// the VOC converter's controller call (PHVoc, sampled, DC rejection); from src/rt/converters/voc.jl
#pragma once
#include <ap_fixed.h>
#include <hls_math.h>

typedef ap_fixed<48, 8, AP_RND_CONV> Sig;
typedef ap_fixed<48, 3, AP_RND_CONV> St;
typedef ap_fixed<48, 2, AP_RND_CONV> Turn;
typedef ap_fixed<48, -7, AP_RND_CONV> Small;
typedef ap_fixed<48, -6, AP_RND_CONV> Ang;
typedef ap_fixed<48, 10, AP_RND_CONV> Coef;
typedef ap_fixed<48, 0, AP_RND_CONV> Zc;
typedef ap_fixed<48, 4, AP_RND_CONV> Rq;
typedef ap_fixed<48, 2, AP_RND_CONV> Rc;
typedef ap_fixed<48, -54, AP_RND_CONV> Tq;
typedef ap_fixed<48, 6, AP_RND_CONV> Pf;
typedef ap_fixed<48, -41, AP_RND_CONV> Pl;
typedef ap_fixed<48, 13, AP_RND_CONV> Vb;
typedef ap_fixed<48, -4, AP_RND_CONV> Gn;
typedef ap_fixed<48, -9, AP_RND_CONV> Gx;
typedef ap_fixed<48, -11, AP_RND_CONV> Ge;
typedef ap_fixed<48, 11, AP_RND_CONV> Ir;
typedef ap_fixed<48, 1, AP_RND_CONV> Dcf;
typedef ap_fixed<48, -2, AP_RND_CONV> Df;
typedef ap_fixed<48, 8, AP_RND_CONV> Dg;
typedef ap_fixed<48, -30, AP_RND_CONV> Rm;
typedef ap_fixed<48, 2, AP_RND_CONV> Rn;
typedef ap_fixed<48, 3, AP_RND_CONV> Ry;
typedef ap_fixed<48, 0, AP_RND_CONV> Dw;
typedef ap_ufixed<18, 2> Sd;
typedef ap_fixed<112, 16, AP_RND_CONV> Wv;

#include "rt_util.h"
#include "voc_rom.h"

struct VocCoefs {
    Zc kz[10];  // the VSM's divider: r_vi, b = Zs + x_vi, Zs, x_vi, kpr, kpx, c1, A2, ab, unused
    Sig kg[12];  // the VSM's: i_lim, thr2, y0 (re, im), unused x 4, m_max, m_max^2, v_min, v_max
    Coef kc[5];  // the VSM's chord start: beta, beta^2, 4 c1, gam0, 1 / (2 c1)
    Gn kl[4];  // lam - 1 (re, im), dg (re, im)
    Gx kx[1];  // Ts xi
    Ge ke[2];  // Ts eta J (re, im)
    Dcf kd[2];  // dcc (re, im)
    Pf kp[3];  // p_set, q_set, V_nom^2
    Pl kq[2];  // p_set, q_set: what their Pf rounding drops
    Sig kv[3];  // |v_k| floor 0.025, m floor 1/8, zero-voltage floor 2^-7
    Small ks[1];  // Ts f_n
    Tq kt[1];  // Ts f_n - its Small rounding
};

Sig vs_rsqrt(Sig a);
Sig vs_recip0(Sig a);
Sig vs_recipv(Sig v);
void dr_mag(Sig xr, Sig xi, Rn on[2], Ry oy[2], int kk[1]);
void voc_call(Turn ph[1], Ang pr[1], Sig xw[2], Vb xv[2], St xm[1], Rm xr[4], Sig u[5], Sig y[2], int fl[5], const VocCoefs &k);
void voc_run(const VocCoefs &k, Turn ph[1], Ang pr[1], Sig xw[2], Vb xv[2], St xm[1], Rm xr[4], Sig u[256][5], Sig y[256][2], int fo[256], int ncall, volatile Sig &m_out, volatile bool &strobe, volatile bool &sample);
