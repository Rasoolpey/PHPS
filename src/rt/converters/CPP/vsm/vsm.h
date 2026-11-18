// vsm.h: written by PHPS.RT.hls_write from the Julia model (@hls); do not edit
// the VSM converter's controller call (PHVsm, sampled); from src/rt/converters/vsm.jl
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
typedef ap_fixed<48, 5, AP_RND_CONV> Pq;
typedef ap_fixed<48, -16, AP_RND_CONV> Ka;
typedef ap_ufixed<18, 2> Sd;

#include "rt_util.h"
#include "vsm_rom.h"

struct VsmCoefs {
    Zc kz[10];  // r_vi, b = Zs + x_vi, Zs, x_vi, kpr, kpx, c1, A2 = c1^2, ab = r_vi kpr + b kpx, unused
    Sig kg[12];  // i_lim, thr2 = (i_lim |Z0|)^2, y0 = 1 / Z0 (re, im), p_set, u_set, u_max, umax2, m_max, mmax2, v_min, v_max
    Coef kc[5];  // beta = c0' + c1 i_lim, beta2, 4 c1, gam0 = c0' i_lim, inv2c1 = 1 / (2 c1)
    Small ks[5];  // Ts / Ta, Ts Dp / Ta, Ts omega_c, Ts / Tv, Ts f_n  (f_n = omega_n / 2 pi)
    Tq kt[1];  // Ts f_n - its Small rounding (the phase increment's second word)
    Pq kx[2];  // p_set, u_set (finer than Sig: omega and u_mag settle on them; 49, 3)
    Ka ka[1];  // Ts / Ta (~3.6e-6: its own fine grid; its rounding otherwise drifts the phase, 49, 3)
    Sig kf[3];  // flags as words (1 or 0): u_max set, the DC side (held m), the VI grows (c1 > 0)
};

Sig vs_rsqrt(Sig a);
Sig vs_recip0(Sig a);
Sig vs_recipv(Sig v);
void vsm_call(St x[3], Turn ph[1], Ang pr[1], Sig u[5], Sig y[2], int fl[4], const VsmCoefs &k);
void vsm_run(const VsmCoefs &k, St x[3], Turn ph[1], Ang pr[1], Sig u[256][5], Sig y[256][2], int fo[256], int ncall, volatile Sig &m_out, volatile bool &strobe, volatile bool &sample);
