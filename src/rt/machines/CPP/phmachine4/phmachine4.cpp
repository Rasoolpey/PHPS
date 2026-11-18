// phmachine4.cpp: written by PHPS.RT.hls_write from the Julia model (@hls); do not edit
// PHMachine, the type with the branch winding (all ten generators); src/rt/machines/phmachine4.jl
#include "phmachine4.h"

void phm4_step(St x[18], Sig u[4], Sig y[2], bool hold, bool hold_next, const PHM4Coefs &k) {
    #pragma HLS ARRAY_PARTITION variable=x complete
    #pragma HLS ARRAY_PARTITION variable=u complete
    #pragma HLS ARRAY_PARTITION variable=y complete
    Sig pd1 = (3 * x[0] - x[11]) / 2;  // predicted midpoint fluxes of the pairs (windings 1 and 4)
    Sig pq1 = (3 * x[4] - x[12]) / 2;
    Sig pd4 = (3 * x[3] - x[13]) / 2;
    Sig pq4 = (3 * x[7] - x[14]) / 2;
    if (hold) {  // the step after an input edge: no extrapolation
        pd1 = x[0];
        pq1 = x[4];
        pd4 = x[3];
        pq4 = x[7];
    }
    Sig mpd1 = -pd1;
    Sig mpd4 = -pd4;
    St c = x[9];  // (cos, sin) of the step's midpoint angle
    St s = x[10];
    St gn = (3 - (c * c + s * s)) / 2;
    St cn = c * gn;
    St sn = s * gn;
    Sig sixth = 0.16666666666666666;
    Sig cn6 = cn * sixth;
    Sig sn6 = sn * sixth;
    Ang kap = k.kap1;
    Sig asum = x[15];
    if (hold_next) {
        kap = k.kap2;
        asum = x[8] + x[15];
    }
    Ang b = k.wbh2 * asum;
    Sig vd = s * u[0] - c * u[1];  // Park: j e^{-j theta} v (kappa in the coefficients)
    Sig vq = c * u[0] + s * u[1];
    Sig xd[4] = {x[0], x[1], x[2], x[3]};
    Sig xq[4] = {x[4], x[5], x[6], x[7]};
    Ang yd[4];
    Ang yq[4];
    for (int i = 0; i < 4; i++) {
        W4y sd = k.cdv[i] * vd + k.wf[i] * u[2];  // summed exactly on a wide word, rounded once to Ang (each rounded
        W4y sq = k.cqv[i] * vq;  // partial sum cost an adder and a rounder: the routing congested)
        for (int j = 0; j < 4; j++) {
            sd += k.Pd[i][j] * xd[j];
            sq += k.Pq[i][j] * xq[j];
        }
        yd[i] = sd;
        yq[i] = sq;
    }
    W4c a1w = k.bf1 * u[2];  // currents of windings 1, 4 at (psi + y / 2) without v terms (wide sums)
    W4c a4w = k.bf4 * u[2];
    W4c b1w = 0;
    W4c b4w = 0;
    for (int j = 0; j < 4; j++) {
        a1w += k.a1d[j] * xd[j];
        a4w += k.a4d[j] * xd[j];
        b1w += k.a1q[j] * xq[j];
        b4w += k.a4q[j] * xq[j];
    }
    Sig D1 = a1w;  // rounded once
    Sig D4 = a4w;
    Sig Q1 = b1w;
    Sig Q4 = b4w;
    Sig Dp = D1;  // the port winding's
    Sig Qp = Q1;
    Mid adp = k.alphad1;
    Mid aqp = k.alphaq1;
    if (k.pw == 4) {
        Dp = D4;
        Qp = Q4;
        adp = k.alphad4;
        aqp = k.alphaq4;
    }
    Sig ibd = Dp + adp * vd;
    Sig ibq = Qp + aqp * vq;
    Sig e1 = k.alphad1 * u[0];  // the torque's voltage terms, the phasor entering once:
    Sig e4 = k.alphad4 * u[0];  // X = u1 A_q - u2 A_d, Y = -u2 A_q - u1 A_d
    Sig f1 = k.alphad1 * u[1];
    Sig f4 = k.alphad4 * u[1];
    Sig g1 = k.alphaq1 * u[0];
    Sig g4 = k.alphaq4 * u[0];
    Sig h1 = k.alphaq1 * u[1];
    Sig h4 = k.alphaq4 * u[1];
    Sig X = pq1 * e1 + pq4 * e4 - pd1 * h1 - pd4 * h4;
    Sig Y = mpd1 * g1 + mpd4 * g4 - pq1 * f1 - pq4 * f4;
    Sig q11 = pq1 * pq1;
    Sig q14 = pq1 * pq4;
    Sig q44 = pq4 * pq4;
    Sig d11 = pd1 * pd1;
    Sig d14 = pd1 * pd4;
    Sig d44 = pd4 * pd4;
    Mid w11 = x[8] * k.kd11;  // omega_n G / 2
    Mid w14 = x[8] * k.kd14;
    Mid w44 = x[8] * k.kd44;
    Mid v11 = x[8] * k.kq11;
    Mid v14 = x[8] * k.kq14;
    Mid v44 = x[8] * k.kq44;
    Sig T1 = pq1 * D1 + pq4 * D4 - pd1 * Q1 - pd4 * Q4 + s * X + c * Y;
    Sig t3 = w11 * q11 + w14 * q14 + w44 * q44 + v11 * d11 + v14 * d14 + v44 * d44;
    Sig f = u[3] - T1 - t3;
    Beta2 bs = k.bd11 * q11 + k.bd14 * q14 + k.bd44 * q44 + k.bq11 * d11 + k.bq14 * d14 + k.bq44 * d44;
    Rate q = k.hq - bs;
    Small dw = q * f;
    Sig wm = x[8] + dw / 2;
    Sig ird = k.kp1d * pq1 + k.kp4d * pq4;  // the port winding's speed-coupling current per omega_m
    Sig irq = k.kp1q * mpd1 + k.kp4q * mpd4;
    Sig hwm = wm / 2;
    Sig idm = ibd + hwm * ird;
    Sig iqm = ibq + hwm * irq;
    Sig t4 = s * idm + c * iqm;  // -j e^{j theta} (i_d + j i_q), times kappa
    Sig t5 = s * iqm - c * idm;
    y[0] = t4 * k.kappa;
    y[1] = t5 * k.kappa;
    Ang dl = kap * wm - b;
    Small dl2 = dl * dl;
    Small hdl2 = dl2 / 2;
    Small sdl = sn6 * dl;
    Small cdl = cn6 * dl;
    St c1 = cn - cn * hdl2 - sn * dl + sdl * dl2;
    St s1 = sn - sn * hdl2 + cn * dl - cdl * dl2;
    Mid ecd[4];
    Mid ecq[4];
    for (int i = 0; i < 4; i++) {
        ecd[i] = k.c1d[i] * pq1 + k.c4d[i] * pq4;
        ecq[i] = k.c1q[i] * mpd1 + k.c4q[i] * mpd4;
    }
    x[11] = x[0];
    x[12] = x[4];
    x[13] = x[3];
    x[14] = x[7];
    for (int i = 0; i < 4; i++) {
        x[i] = x[i] + yd[i] + ecd[i] * wm;
        x[i + 4] = x[i + 4] + yq[i] + ecq[i] * wm;
    }
    x[8] = x[8] + dw;
    x[9] = c1;
    x[10] = s1;
    x[15] = wm;
    if (hold_next) {
        x[15] = x[8];
    }
}

void phm4_run(const PHM4Coefs &k, St x[18], Sig vmag, Small cm, Small srot, Sig vf, Sig tm, Sig vdip, int n0, int n_on, int n_off, int nsteps, int decim, volatile Sig &omega_out, volatile Sig &ia_out, volatile Sig &ib_out, volatile bool &strobe, volatile bool &sample) {
    #pragma HLS INTERFACE mode=s_axilite port=return
    #pragma HLS INTERFACE mode=s_axilite port=k
    #pragma HLS INTERFACE mode=s_axilite port=x
    #pragma HLS INTERFACE mode=s_axilite port=vmag
    #pragma HLS INTERFACE mode=s_axilite port=cm
    #pragma HLS INTERFACE mode=s_axilite port=srot
    #pragma HLS INTERFACE mode=s_axilite port=vf
    #pragma HLS INTERFACE mode=s_axilite port=tm
    #pragma HLS INTERFACE mode=s_axilite port=vdip
    #pragma HLS INTERFACE mode=s_axilite port=n0
    #pragma HLS INTERFACE mode=s_axilite port=n_on
    #pragma HLS INTERFACE mode=s_axilite port=n_off
    #pragma HLS INTERFACE mode=s_axilite port=nsteps
    #pragma HLS INTERFACE mode=s_axilite port=decim
    #pragma HLS INTERFACE mode=ap_none port=omega_out
    #pragma HLS INTERFACE mode=ap_none port=ia_out
    #pragma HLS INTERFACE mode=ap_none port=ib_out
    #pragma HLS INTERFACE mode=ap_none port=strobe
    #pragma HLS INTERFACE mode=ap_none port=sample
    St xs[18];
    #pragma HLS ARRAY_PARTITION variable=xs complete
    for (int i = 0; i < 18; i++) {
        #pragma HLS UNROLL
        xs[i] = x[i];
    }
    Sig u[4];
    #pragma HLS ARRAY_PARTITION variable=u complete
    Sig y[2];
    #pragma HLS ARRAY_PARTITION variable=y complete
    bool st = false;
    int cnt = 0;
    for (int i = 0; i < nsteps; i++) {
        #pragma HLS PIPELINE II=43
        #pragma HLS ALLOCATION operation instances=mul limit=16
        int n = n0 + (i + 1);
        Sig vs = vmag;
        if (n > n_on && n <= n_off) {
            vs = vmag * vdip;
        }
        bool hold = false;
        if (n == n_on + 1) {
            hold = true;
        }
        if (n == n_off + 1) {
            hold = true;
        }
        bool hold_next = false;
        if (n == n_on) {
            hold_next = true;
        }
        if (n == n_off) {
            hold_next = true;
        }
        u[0] = vs * xs[16];
        u[1] = vs * xs[17];
        u[2] = vf;
        u[3] = tm;
        phm4_step(xs, u, y, hold, hold_next, k);
        St sc1 = xs[16] - xs[16] * cm - xs[17] * srot;  // the bench's source phasor on the state grid (St): on Sig <48,11>
        St ss1 = xs[17] - xs[17] * cm + xs[16] * srot;  // (widened for GENROU_1's currents) its 7.3e-12 rounding random-
        St gs = (3 - (sc1 * sc1 + ss1 * ss1)) / 2;  // walked ~4e-9 in angle over 0.3 s of the dip bench
        xs[16] = sc1 * gs;
        xs[17] = ss1 * gs;
        st = !st;
        cnt += 1;
        if (cnt >= decim) {
            cnt = 0;
            sample = true;
        } else {
            sample = false;
        }
        omega_out = xs[8];
        ia_out = y[0];
        ib_out = y[1];
        strobe = st;
    }
    for (int i = 0; i < 18; i++) {
        #pragma HLS UNROLL
        x[i] = xs[i];
    }
}

