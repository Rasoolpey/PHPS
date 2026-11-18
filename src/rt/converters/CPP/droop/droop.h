// droop.h: written by PHPS.RT.hls_write from the Julia model (@hls); do not edit
// the droop converter's controller call (PHDroop, sampled, DC rejection); from src/rt/converters/droop.jl
#pragma once
#include <ap_fixed.h>
#include <hls_math.h>

typedef ap_fixed<48, 8, AP_RND_CONV> Sig;
typedef ap_fixed<48, 3, AP_RND_CONV> St;
typedef ap_fixed<48, 2, AP_RND_CONV> Turn;
typedef ap_fixed<48, -7, AP_RND_CONV> Small;
typedef ap_fixed<48, -6, AP_RND_CONV> Ang;
typedef ap_fixed<48, 10, AP_RND_CONV> Coef;
typedef ap_fixed<48, 4, AP_RND_CONV> Rq;
typedef ap_fixed<48, 2, AP_RND_CONV> Rc;
typedef ap_fixed<48, -54, AP_RND_CONV> Tq;
typedef ap_fixed<48, 0, AP_RND_CONV> Zc;
typedef ap_fixed<48, 6, AP_RND_CONV> Pf;
typedef ap_fixed<48, 13, AP_RND_CONV> Vb;
typedef ap_fixed<48, -4, AP_RND_CONV> Gn;
typedef ap_fixed<48, -9, AP_RND_CONV> Ga;
typedef ap_fixed<48, -7, AP_RND_CONV> Gu;
typedef ap_fixed<48, -4, AP_RND_CONV> Gv;
typedef ap_fixed<48, -5, AP_RND_CONV> Gm;
typedef ap_fixed<48, -14, AP_RND_CONV> Km;
typedef ap_fixed<48, 1, AP_RND_CONV> Dcf;
typedef ap_fixed<48, -2, AP_RND_CONV> Df;
typedef ap_fixed<48, 3, AP_RND_CONV> Dh;
typedef ap_fixed<48, 8, AP_RND_CONV> Dg;
typedef ap_fixed<48, -30, AP_RND_CONV> Rm;
typedef ap_fixed<48, 2, AP_RND_CONV> Rn;
typedef ap_fixed<48, 3, AP_RND_CONV> Ry;
typedef ap_fixed<48, 2, AP_RND_CONV> Fr;
typedef ap_ufixed<18, 2> Sd;

#include "rt_util.h"
#include "droop_rom.h"

struct DroopCoefs {
    Gn kl[4];  // lam - 1 (re, im); dg (re, im)
    Ga ka[2];  // aP, aQ = Ts / (Tf_p, Tf_q)
    Gu ku[1];  // aU = Ts / Tv
    Gv kv[2];  // aV, aI = Ts / (T_V, T_I)
    Dcf kd[2];  // dcc (re, im)
    Zc kk[2];  // kpr, kpx
    Sig kg[8];  // i_lim, i_con_lim, dmax = Zs i_con_lim, m_max, m_max^2, v_min, v_max, unused
    Pf kp[3];  // p_set, q_set, u_set
    Gm km[1];  // mq
    Small ks[1];  // c0 = Ts f_n (1 + mp p_set) (f_n = omega_n / 2 pi): the phase increment's constant part
    Tq kt[1];  // c0 - its Small rounding (the second word: a rounded p_set would drift the phase ~2e-18 turn per call)
    Km kpm[1];  // Ts f_n mp (the increment is c0 - (Ts f_n mp) p_f)
};

Sig vs_rsqrt(Sig a);
Sig vs_recipv(Sig v);
void dr_mag(Sig xr, Sig xi, Rn on[2], Ry oy[2], int kk[1]);
void droop_call(Turn ph[1], Ang pr[1], Pf xp[2], St xu[1], Sig xs[6], Vb xv[2], Rm xr[11], Sig u[5], Sig y[2], int fl[5], const DroopCoefs &k);
void droop_run(const DroopCoefs &k, Turn ph[1], Ang pr[1], Pf xp[2], St xu[1], Sig xs[6], Vb xv[2], Rm xr[11], Sig u[256][5], Sig y[256][2], int fo[256], int ncall, volatile Sig &m_out, volatile bool &strobe, volatile bool &sample);
