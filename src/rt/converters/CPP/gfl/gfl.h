// gfl.h: written by PHPS.RT.hls_write from the Julia model (@hls); do not edit
// the GFL converter (PHGfl): the sampled call and the h-paced current loop, h = 3.2e-8 s; from src/rt/converters/gfl.jl
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
typedef ap_fixed<48, 7, AP_RND_CONV> Pf;
typedef ap_fixed<48, -30, AP_RND_CONV> Rm;
typedef ap_fixed<48, 2, AP_RND_CONV> Rn;
typedef ap_fixed<48, 3, AP_RND_CONV> Ry;
typedef ap_fixed<48, 0, AP_RND_CONV> Zc;
typedef ap_fixed<48, -4, AP_RND_CONV> Gl;
typedef ap_fixed<48, -9, AP_RND_CONV> Gp;
typedef ap_fixed<48, 0, AP_RND_CONV> Go;
typedef ap_fixed<48, -19, AP_RND_CONV> Gc;
typedef ap_fixed<48, 3, AP_RND_CONV> Zs;
typedef ap_fixed<48, -13, AP_RND_CONV> Gt;
typedef ap_fixed<48, -4, AP_RND_CONV> Ai;
typedef ap_ufixed<18, 2> Sd;

#include "rt_util.h"
#include "gfl_rom.h"

struct GflCoefs {
    Small kc[1];  // c1 = Ts f_n (the phase increment's constant)
    Tq kt[1];  // c1 - its Small rounding
    Gp kp[2];  // c2 = Ts kp_pll / 2 pi, c3 = Ts ki_pll / 2 pi
    Go ko[2];  // kp_pll / omega_n, ki_pll / omega_n (omega / omega_n - 1)
    Gl kl[2];  // Ts / T_meas, Ts / T_cc
    Pf kg[15];  // p_ref, q_ref, r_p Ts, r_q Ts, kp_P, ki_P, kp_Q, ki_Q, Kqf, k_droop_p, f_db, k_droop_q, v_db, i_max, phi_max
    Gt kts[1];  // Ts
    Zc kz[2];  // Kd, Kq (x1 below)
    Zc kx[1];  // x1
};

struct GflLoop {
    Gc kh[4];  // h Kd / Td, h Kq / Tq, their halves
    Zc kz[3];  // Kd, Kq, x1
};

void dr_mag(Sig xr, Sig xi, Rn on[2], Ry oy[2], int kk[1]);
void gfl_call(Turn ph[1], Ang pr[1], Pf xs[9], Rm xr[6], Sig lt[5], Sig u[2], int fl[4], const GflCoefs &k);
void gfl_loop(Zs z[2], Rm zr[2], Sig lt[5], Sig i[2], Sig y[2], const GflLoop &kq);
void gfl_run(const GflCoefs &k, const GflLoop &kq, int mode, Turn ph[1], Ang pr[1], Pf xs[9], Rm xr[6], Sig lt[5], Zs z[2], Rm zr[2], Sig uc[64][2], int fo[64], Sig il[512][2], Sig yl[512][2], int n, int decim, volatile Sig &u_out, volatile bool &strobe, volatile bool &sample);
