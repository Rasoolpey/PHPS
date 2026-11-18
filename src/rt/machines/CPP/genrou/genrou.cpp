// genrou.cpp: written by PHPS.RT.hls_write from the Julia model (@hls); do not edit
// GENROU, one step of h; from src/rt/machines/genrou.jl
#include "genrou.h"

void genrou_step(Sig x[8], Sig u[4], const GenrouCoefs &k) {
    #pragma HLS ARRAY_PARTITION variable=x complete
    #pragma HLS ARRAY_PARTITION variable=u complete
    Sig c = x[6];
    Sig s = x[7];
    Sig vd = u[0] * s - u[1] * c;  // Park transform
    Sig vq = u[0] * c + u[1] * s;
    Sig pdd = k.kd * x[2] + (1 - k.kd) * x[3];  // subtransient fluxes
    Sig pqq = -k.kq * x[4] + (1 - k.kq) * x[5];
    Sig rd = vd + x[1] * pqq;
    Sig rq = vq - x[1] * pdd;
    Sig id = k.S[0] * rd + k.S[1] * rq;  // stator currents
    Sig iq = k.S[2] * rd + k.S[3] * rq;
    Sig Pe = vd * id + vq * iq;  // terminal power
    Sig v[3] = {u[3], id, iq};  // rotor fluxes: implicit midpoint, currents held
    Sig z[4];
    for (int i = 0; i < 4; i++) {
        Sig d = 0;
        for (int j = 0; j < 4; j++) {
            d += k.P[i][j] * x[j + 2];
        }
        for (int j = 0; j < 3; j++) {
            d += k.W[i][j] * v[j];
        }
        z[i] = x[i + 2] + d;
    }
    Sig r0 = 2 - x[1];  // 1/omega: seed 2 - omega, then three Newton steps
    Sig t1 = x[1] * r0;  // r = r (2 - omega r); every product rounded to Sig, so the
    Sig r1 = r0 * (2 - t1);  // multipliers stay 48 x 48 bits; straight lines (no loop), so
    Sig t2 = x[1] * r1;  // HLS can run them alongside the stator and rotor
    Sig r2 = r1 * (2 - t2);
    Sig t3 = x[1] * r2;
    Sig r = r2 * (2 - t3);
    Sig f = (u[2] - Pe) * r;  // (Tm - Pe) / omega
    Sig w1 = 1 + k.a1 * (x[1] - 1) + k.a2 * f;  // speed
    Small dd = k.wb_h * ((x[1] + w1) / 2 - 1);  // angle increment of the step (trapezoid)
    Sig cd = 1 - dd * dd / 2;  // cos and sin of the increment (series; error below the last bit)
    Sig c1 = c * cd - s * dd;  // rotate (cos delta, sin delta)
    Sig s1 = s * cd + c * dd;
    Sig g = (3 - (c1 * c1 + s1 * s1)) / 2;  // keep cos^2 + sin^2 = 1
    Sig d1 = x[0] + dd;  // delta, kept in [-pi, pi)
    Sig piv = 3.141592653589793;
    Sig twopi = 6.283185307179586;
    if (d1 >= piv) {
        d1 -= twopi;
    }
    if (d1 < -piv) {
        d1 += twopi;
    }
    x[0] = d1;
    x[1] = w1;
    for (int i = 0; i < 4; i++) {
        #pragma HLS UNROLL
        x[i + 2] = z[i];
    }
    x[6] = c1 * g;
    x[7] = s1 * g;
}

void genrou_run(const GenrouCoefs &k, Sig x[8], Sig vre, Sig vim, Sig tm, Sig efd, Sig vdip, int n_on, int n_off, int nsteps, int decim, volatile Sig &delta_out, volatile Sig &omega_out, volatile Sig &eq_out, volatile Sig &psid_out, volatile bool &strobe, volatile bool &sample) {
    #pragma HLS INTERFACE mode=s_axilite port=return
    #pragma HLS INTERFACE mode=s_axilite port=k
    #pragma HLS INTERFACE mode=s_axilite port=x
    #pragma HLS INTERFACE mode=s_axilite port=vre
    #pragma HLS INTERFACE mode=s_axilite port=vim
    #pragma HLS INTERFACE mode=s_axilite port=tm
    #pragma HLS INTERFACE mode=s_axilite port=efd
    #pragma HLS INTERFACE mode=s_axilite port=vdip
    #pragma HLS INTERFACE mode=s_axilite port=n_on
    #pragma HLS INTERFACE mode=s_axilite port=n_off
    #pragma HLS INTERFACE mode=s_axilite port=nsteps
    #pragma HLS INTERFACE mode=s_axilite port=decim
    #pragma HLS INTERFACE mode=ap_none port=delta_out
    #pragma HLS INTERFACE mode=ap_none port=omega_out
    #pragma HLS INTERFACE mode=ap_none port=eq_out
    #pragma HLS INTERFACE mode=ap_none port=psid_out
    #pragma HLS INTERFACE mode=ap_none port=strobe
    #pragma HLS INTERFACE mode=ap_none port=sample
    Sig xs[8];  // the states, kept in registers during the run
    #pragma HLS ARRAY_PARTITION variable=xs complete
    for (int i = 0; i < 8; i++) {
        #pragma HLS UNROLL
        xs[i] = x[i];
    }
    Sig u[4];
    #pragma HLS ARRAY_PARTITION variable=u complete
    bool st = false;
    int cnt = 0;  // steps since the last ILA sample
    for (int n = 0; n < nsteps; n++) {
        #pragma HLS PIPELINE II=118
        Sig vs = 1;  // bus voltage factor: vdip during steps n_on+1 .. n_off
        if ((n + 1) > n_on && (n + 1) <= n_off) {
            vs = vdip;
        }
        u[0] = vre * vs;
        u[1] = vim * vs;
        u[2] = tm;
        u[3] = efd;
        genrou_step(xs, u, k);
        st = !st;
        cnt += 1;  // sample = 1 for one step every decim steps (the ILA records
        if (cnt >= decim) {  // only then: with decim = 32, 16k samples cover 0.5 s)
            cnt = 0;
            sample = true;
        } else {
            sample = false;
        }
        delta_out = xs[0];  // to the ILA
        omega_out = xs[1];
        eq_out = xs[2];
        psid_out = xs[3];
        strobe = st;
    }
    for (int i = 0; i < 8; i++) {
        #pragma HLS UNROLL
        x[i] = xs[i];
    }
}

