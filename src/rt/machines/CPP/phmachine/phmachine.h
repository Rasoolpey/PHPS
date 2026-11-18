// phmachine.h: written by PHPS.RT.hls_write from the Julia model (@hls); do not edit
// PHMachine (GENROU_3), one step of h; from src/rt/machines/phmachine.jl
#pragma once
#include <ap_fixed.h>
#include <hls_math.h>

typedef ap_fixed<48, 8, AP_RND_CONV> Sig;
typedef ap_fixed<48, 3, AP_RND_CONV> St;
typedef ap_fixed<48, -10, AP_RND_CONV> Small;
typedef ap_fixed<48, 8, AP_RND_CONV> Coef;
typedef ap_fixed<48, -9, AP_RND_CONV> Ang;
typedef ap_fixed<48, -2, AP_RND_CONV> Mid;
typedef ap_fixed<48, -24, AP_RND_CONV> Rate;
typedef ap_fixed<48, -59, AP_RND_CONV> Beta;
typedef ap_fixed<48, -55, AP_RND_CONV> Beta2;

struct PHMachineCoefs {
    Small Pd[3][3];  // d group step matrix (d, fd, 1d)
    Small Pq[3][3];  // q group step matrix (q, 1q, 2q)
    Small cd[3];  // d group: column of the stator voltage v_d (and of the speed coupling)
    Small cq[3];  // q group: column of the stator voltage v_q (and of the speed coupling)
    Small cdv[3];  // cd / a: the column of the network voltage's d component (the stator sees v / a)
    Small cqv[3];  // cq / a
    Small wf[3];  // d group: column of the field voltage v_f
    Coef ad[3];  // kd + Pd' kd / 2: i_d of (psi + y / 2) from psi
    Coef aq[3];  // kq + Pq' kq / 2
    Mid alphad;  // kd . cdv / 2: i_d of (psi + y / 2) per unit v_d
    Mid alphaq;  // kq . cqv / 2
    Mid bf;  // kd . wf / 2: i_d of (psi + y / 2) per unit v_f
    Mid kcd;  // kd . cd: stator current of the speed coupling per unit psi_q^
    Mid kcq;  // kq . cq
    Mid kcd2;  // kd . cd / 2
    Mid kcq2;  // kq . cq / 2
    Coef ainv;  // 1 / a (the injected current is i_net / a)
    Rate hq;  // h / 2H
    Beta betad;  // (h / 2H)(h / 8H) kd . cd: q = hq - betad psi_q^2 - betaq psi_d^2
    Beta betaq;  // (h / 2H)(h / 8H) kq . cq
    Ang kap1;  // 1.5 h omega_b: the midpoint angle's increment per unit omega_m (normal next step)
    Ang kap2;  // 2 h omega_b (the next step holds)
    Ang wbh2;  // h omega_b / 2
};

void phmachine_step(St x[14], Sig u[4], Sig y[2], bool hold, bool hold_next, const PHMachineCoefs &k);
void phmachine_run(const PHMachineCoefs &k, St x[14], Sig vmag, Small cm, Small srot, Sig vf, Sig tm, Sig vdip, int n0, int n_on, int n_off, int nsteps, int decim, volatile Sig &omega_out, volatile Sig &ia_out, volatile Sig &ib_out, volatile bool &strobe, volatile bool &sample);
