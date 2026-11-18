// phmachine.cpp: written by PHPS.RT.hls_write from the Julia model (@hls); do not edit
// PHMachine (GENROU_3), one step of h; from src/rt/machines/phmachine.jl
#include "phmachine.h"

void phmachine_step(St x[14], Sig u[4], Sig y[2], bool hold, bool hold_next, const PHMachineCoefs &k) {
    #pragma HLS ARRAY_PARTITION variable=x complete
    #pragma HLS ARRAY_PARTITION variable=u complete
    #pragma HLS ARRAY_PARTITION variable=y complete
    Sig ph_d = (3 * x[0] - x[9]) / 2;  // predicted stator fluxes at the step's midpoint
    Sig ph_q = (3 * x[1] - x[10]) / 2;
    if (hold) {  // the step after an input edge: no extrapolation
        ph_d = x[0];
        ph_q = x[1];
    }
    Sig mph_d = -ph_d;
    St c = x[7];  // (cos, sin) of the step's midpoint angle theta^_n (on the St
    St s = x[8];  // grid, as the state)
    St gn = (3 - (c * c + s * s)) / 2;  // the rotation's source renormalised (off the speed path)
    St cn = c * gn;
    St sn = s * gn;
    Sig sixth = 0.16666666666666666;
    Sig cn6 = cn * sixth;
    Sig sn6 = sn * sixth;
    Ang kap = k.kap1;  // next midpoint: Delta = kap omega_m - b (PHMACHINE.md 7.2)
    Sig asum = x[11];  // b = h omega_b a_n / 2
    if (hold_next) {  // the next step holds: a_{n+1} = omega_{n+1} = 2 omega_m - omega_n
        kap = k.kap2;
        asum = x[6] + x[11];  // b = h omega_b (omega_n + a_n) / 2
    }
    Ang b = k.wbh2 * asum;
    Sig vd = s * u[0] - c * u[1];  // Park transform at the midpoint: j e^{-j theta} v (the 1/a
    Sig vq = c * u[0] + s * u[1];  // is in the coefficients cdv, cqv)
    Sig xd[3] = {x[0], x[2], x[3]};  // d group (d, fd, 1d), q group (q, 1q, 2q)
    Sig xq[3] = {x[1], x[4], x[5]};
    Ang yd[3];  // the step's increment at frozen speed: y = P psi + W u, summed
    Ang yq[3];  // in the fine Ang (in Sig the sums' rounding biased psi_fd by
    for (int i = 0; i < 3; i++) {  // one Sig grid per step: C sim run 1, review 33b)
        Ang sd = k.cdv[i] * vd + k.wf[i] * u[2];
        Ang sq = k.cqv[i] * vq;
        for (int j = 0; j < 3; j++) {
            sd += k.Pd[i][j] * xd[j];
            sq += k.Pq[i][j] * xq[j];
        }
        yd[i] = sd;
        yq[i] = sq;
    }
    Sig D = k.bf * u[2];  // stator currents of (psi + y/2) without the v_dq terms
    Sig Q = 0;
    for (int j = 0; j < 3; j++) {
        D += k.ad[j] * xd[j];
        Q += k.aq[j] * xq[j];
    }
    Sig ibd = D + k.alphad * vd;  // stator currents of (psi + y/2)
    Sig ibq = Q + k.alphaq * vq;
    Sig a1d = k.alphad * u[0];  // the torque with the midpoint phasor entering once:
    Sig a2d = k.alphad * u[1];  // T1 = ph_q D - ph_d Q + s X + c Y
    Sig a1q = k.alphaq * u[0];
    Sig a2q = k.alphaq * u[1];
    Sig X = ph_q * a1d - ph_d * a2q;
    Sig Y = mph_d * a1q - ph_q * a2d;
    Sig phq2 = ph_q * ph_q;
    Sig phd2 = ph_d * ph_d;
    Mid wkd = x[6] * k.kcd2;  // omega_n / 2 times K c
    Mid wkq = x[6] * k.kcq2;
    Sig T1 = ph_q * D - ph_d * Q + s * X + c * Y;  // a' K (psi + y/2)
    Sig t3 = wkd * phq2 + wkq * phd2;  // omega_n / 2 a' K r
    Sig f = u[3] - T1 - t3;  // T_m - a' K (psi + y/2 + omega_0 r / 2)
    Beta2 bs = k.betad * phq2 + k.betaq * phd2;
    Rate q = k.hq - bs;  // h / 2H / s with 1/s = 1 - (s - 1)
    Small dw = q * f;  // speed increment
    Sig wm = x[6] + dw / 2;  // midpoint speed
    Sig ird = k.kcd * ph_q;  // midpoint stator currents, to the network frame
    Sig irq = k.kcq * mph_d;
    Sig hwm = wm / 2;
    Sig idm = ibd + hwm * ird;
    Sig iqm = ibq + hwm * irq;
    Sig t4 = s * idm + c * iqm;  // i_net = -j e^{j theta} (i_d + j i_q), over a
    Sig t5 = s * iqm - c * idm;
    y[0] = t4 * k.ainv;
    y[1] = t5 * k.ainv;
    Ang dl = kap * wm - b;  // rotate the midpoint phasor to the next midpoint
    Small dl2 = dl * dl;
    Small hdl2 = dl2 / 2;
    Small sdl = sn6 * dl;
    Small cdl = cn6 * dl;
    St c1 = cn - cn * hdl2 - sn * dl + sdl * dl2;  // series of cos to Delta^2, sin to Delta^3 (St: review 33c, 4)
    St s1 = sn - sn * hdl2 + cn * dl - cdl * dl2;
    Mid ecd[3];  // the speed coupling's columns times psi^, before omega_m
    Mid ecq[3];
    for (int i = 0; i < 3; i++) {
        ecd[i] = k.cd[i] * ph_q;
        ecq[i] = k.cq[i] * mph_d;
    }
    x[9] = x[0];  // predictor history
    x[10] = x[1];
    x[0] = x[0] + yd[0] + ecd[0] * wm;  // new fluxes: psi + y + omega_m r
    x[2] = x[2] + yd[1] + ecd[1] * wm;
    x[3] = x[3] + yd[2] + ecd[2] * wm;
    x[1] = x[1] + yq[0] + ecq[0] * wm;
    x[4] = x[4] + yq[1] + ecq[1] * wm;
    x[5] = x[5] + yq[2] + ecq[2] * wm;
    x[6] = x[6] + dw;
    x[7] = c1;
    x[8] = s1;
    x[11] = wm;  // a_{n+1}
    if (hold_next) {
        x[11] = x[6];
    }
}

void phmachine_run(const PHMachineCoefs &k, St x[14], Sig vmag, Small cm, Small srot, Sig vf, Sig tm, Sig vdip, int n0, int n_on, int n_off, int nsteps, int decim, volatile Sig &omega_out, volatile Sig &ia_out, volatile Sig &ib_out, volatile bool &strobe, volatile bool &sample) {
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
    St xs[14];  // the states, kept in registers during the run
    #pragma HLS ARRAY_PARTITION variable=xs complete
    for (int i = 0; i < 14; i++) {
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
        #pragma HLS PIPELINE II=40
        #pragma HLS ALLOCATION operation instances=mul limit=16
        int n = n0 + (i + 1);
        Sig vs = vmag;
        if (n > n_on && n <= n_off) {
            vs = vmag * vdip;
        }
        bool hold = false;  // the step after an edge of the dip holds the predictor
        if (n == n_on + 1) {
            hold = true;
        }
        if (n == n_off + 1) {
            hold = true;
        }
        bool hold_next = false;  // and the step before it prepares the held midpoint angle
        if (n == n_on) {
            hold_next = true;
        }
        if (n == n_off) {
            hold_next = true;
        }
        u[0] = vs * xs[12];
        u[1] = vs * xs[13];
        u[2] = vf;
        u[3] = tm;
        phmachine_step(xs, u, y, hold, hold_next, k);
        Sig sc1 = xs[12] - xs[12] * cm - xs[13] * srot;  // the source phasor to the next step's midpoint
        Sig ss1 = xs[13] - xs[13] * cm + xs[12] * srot;
        Sig gs = (3 - (sc1 * sc1 + ss1 * ss1)) / 2;
        xs[12] = sc1 * gs;
        xs[13] = ss1 * gs;
        st = !st;
        cnt += 1;
        if (cnt >= decim) {
            cnt = 0;
            sample = true;
        } else {
            sample = false;
        }
        omega_out = xs[6];  // to the ILA
        ia_out = y[0];
        ib_out = y[1];
        strobe = st;
    }
    for (int i = 0; i < 14; i++) {
        #pragma HLS UNROLL
        x[i] = xs[i];
    }
}

