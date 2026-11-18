// phmachine4.h: written by PHPS.RT.hls_write from the Julia model (@hls); do not edit
// PHMachine, the type with the branch winding (all ten generators); src/rt/machines/phmachine4.jl
#pragma once
#include <ap_fixed.h>
#include <hls_math.h>

typedef ap_fixed<96, 0, AP_RND_CONV> W4y;
typedef ap_fixed<96, 14, AP_RND_CONV> W4c;
typedef ap_fixed<48, 11, AP_RND_CONV> Sig;
typedef ap_fixed<48, 3, AP_RND_CONV> St;
typedef ap_fixed<48, -8, AP_RND_CONV> Small;
typedef ap_fixed<48, 9, AP_RND_CONV> Coef;
typedef ap_fixed<48, -9, AP_RND_CONV> Ang;
typedef ap_fixed<48, -2, AP_RND_CONV> Mid;
typedef ap_fixed<48, -24, AP_RND_CONV> Rate;
typedef ap_fixed<48, -59, AP_RND_CONV> Beta;
typedef ap_fixed<48, -55, AP_RND_CONV> Beta2;

struct PHM4Coefs {
    Small Pd[4][4];  // d group step matrix (stator, fd, 1d, T)
    Small Pq[4][4];  // q group (stator, 1q, 2q, T)
    Small cdv[4];  // the port voltage's column (kappa h omega_b M^-1 e_pw), d and q
    Small cqv[4];
    Small wf[4];  // the field voltage's column (d)
    Small c1d[4];  // the speed-coupling columns of the pairs (h omega_b M^-1 e_1, e_4)
    Small c4d[4];
    Small c1q[4];
    Small c4q[4];
    Coef a1d[4];  // currents of windings 1 and 4 at (psi + y / 2) from psi: k_i + P' k_i / 2
    Coef a4d[4];
    Coef a1q[4];
    Coef a4q[4];
    Mid alphad1;  // k_i . cdv / 2, k_i . cqv / 2: the same currents per unit port voltage
    Mid alphad4;
    Mid alphaq1;
    Mid alphaq4;
    Mid bf1;  // k_i . wf / 2 (field)
    Mid bf4;
    Mid kd11;  // G / 2 (cross term G): t3 = omega_n S / 2 by terms
    Mid kd14;
    Mid kd44;
    Mid kq11;
    Mid kq14;
    Mid kq44;
    Mid kp1d;  // the port winding's current per unit psi^ of each pair (speed coupling)
    Mid kp4d;
    Mid kp1q;
    Mid kp4q;
    Coef kappa;  // 1 / a (ordinary) or c (cut)
    int pw;  // the port winding: 1 or 4
    Rate hq;  // h / 2H
    Beta bd11;  // (hq^2 / 4) G (cross term 2 G): q = hq - bs
    Beta bd14;
    Beta bd44;
    Beta bq11;
    Beta bq14;
    Beta bq44;
    Ang kap1;
    Ang kap2;
    Ang wbh2;
};

void phm4_step(St x[18], Sig u[4], Sig y[2], bool hold, bool hold_next, const PHM4Coefs &k);
void phm4_run(const PHM4Coefs &k, St x[18], Sig vmag, Small cm, Small srot, Sig vf, Sig tm, Sig vdip, int n0, int n_on, int n_off, int nsteps, int decim, volatile Sig &omega_out, volatile Sig &ia_out, volatile Sig &ib_out, volatile bool &strobe, volatile bool &sample);
