// hydro.cpp: written by PHPS.RT.hls_write from the Julia model (@hls); do not edit
// the hydro unit (PHHydroUnit), one run of steps of h = 1.92e-7 s; from src/rt/governors/hydro.jl
#include "hydro.h"

Rr hy_recip(Sig a) {
    static const Rq seed[1024] = HY_SEED;
    Rq two = 2;
    Rq m = 0;
    int k = 0;
    Sig c8 = 8;
    Sig c4 = 4;
    Sig c2 = 2;
    Sig c1 = 1;
    Sig ch = 0.5;
    Sig cq = 0.25;
    Sig ce = 0.125;
    Sig c16 = 0.0625;
    Sig c32 = 0.03125;
    Sig c64 = 0.015625;
    if (a >= c8) {
        m = scale2<-4>(a);
        k = 4;
    } else if (a >= c4) {
        m = scale2<-3>(a);
        k = 3;
    } else if (a >= c2) {
        m = scale2<-2>(a);
        k = 2;
    } else if (a >= c1) {
        m = scale2<-1>(a);
        k = 1;
    } else if (a >= ch) {
        m = a;
        k = 0;
    } else if (a >= cq) {
        m = scale2<1>(a);
        k = -1;
    } else if (a >= ce) {
        m = scale2<2>(a);
        k = -2;
    } else if (a >= c16) {
        m = scale2<3>(a);
        k = -3;
    } else if (a >= c32) {
        m = scale2<4>(a);
        k = -4;
    } else if (a >= c64) {
        m = scale2<5>(a);
        k = -5;
    } else {
        m = scale2<6>(a);
        k = -6;
    }
    int j = ifloor(scale2<11>(m)) - 1024;
    if (j < 0) {
        j = 0;
    }
    if (j > 1023) {
        j = 1023;
    }
    Rq y = seed[j];
    for (int it = 0; it < 2; it++) {
        Rq t = m * y;
        Rq u = two - t;
        y = y * u;
    }
    Rr r = y;
    if (k == 4) {
        r = scale2<-4>(y);
    } else if (k == 3) {
        r = scale2<-3>(y);
    } else if (k == 2) {
        r = scale2<-2>(y);
    } else if (k == 1) {
        r = scale2<-1>(y);
    } else if (k == -1) {
        r = scale2<1>(y);
    } else if (k == -2) {
        r = scale2<2>(y);
    } else if (k == -3) {
        r = scale2<3>(y);
    } else if (k == -4) {
        r = scale2<4>(y);
    } else if (k == -5) {
        r = scale2<5>(y);
    } else if (k == -6) {
        r = scale2<6>(y);
    }
    return r;
}

int hy_region(St v, Coef lo, Coef hi) {
    int rg = 0;
    if (v > hi) {
        rg = 1;
    } else if (v < lo) {
        rg = 2;
    }
    return rg;
}

int hy_freeze(St vl, int pos, int neg, Coef lo, Coef hi) {
    int fz = 0;
    if (vl >= hi && pos == 1) {
        fz = 1;
    } else if (vl <= lo && neg == 1) {
        fz = 1;
    }
    return fz;
}

void hydro_step(St x[4], Small xc[4], Sig u[1], Pw y[1], int md[3], Small kP[16][9], Small kW[16][3], Small kK[16][3], Small kEX[16][3], Small kEA[16][3], Coef kc[11], Bt kb[1], Cw kw[1]) {
    Sig om = u[0];
    Sig one = 1;
    Sig om1 = om - one;
    int ar = hy_region(x[2], kc[8], kc[7]);
    int xr_ = hy_region(x[0], kc[6], kc[5]);
    St atl = x[2];  // St: a state one LSB off its limit stays off it (C++ corners)
    Coef atc = 0;
    int ain = 1;
    if (ar == 1) {
        atl = kc[7];
        atc = kc[7];
        ain = 0;
    } else if (ar == 2) {
        atl = kc[8];
        atc = kc[8];
        ain = 0;
    }
    St xpl = x[0];
    Coef xpc = 0;
    int xin = 1;
    if (xr_ == 1) {
        xpl = kc[5];
        xpc = kc[5];
        xin = 0;
    } else if (xr_ == 2) {
        xpl = kc[6];
        xpc = kc[6];
        xin = 0;
    }
    Sig e1 = kc[1] * om1;
    Sig e2 = kc[2] * atl;
    Sig e3 = kc[3] * x[1];
    Sig e4 = kc[4] * xpl;
    Sig ea = kc[0] - e1;
    Sig eb = e3 - e2;
    Sig ec = ea - e4;
    Sig s = ec + eb;
    int sp = 0;
    int sn = 0;
    if (s > 0) {
        sp = 1;
    } else if (s < 0) {
        sn = 1;
    }
    int xq = 0;
    int xn_ = 0;
    if (xpl > 0) {
        xq = 1;
    } else if (xpl < 0) {
        xn_ = 1;
    }
    int nx = hy_freeze(xpl, sp, sn, kc[6], kc[5]);
    int na = hy_freeze(atl, xq, xn_, kc[8], kc[7]);
    int bk = 1 + 8 * ain + 4 * xin + 2 * nx + na;
    Inc d[3];
    #pragma HLS ARRAY_PARTITION variable=d complete
    for (int i = 0; i < 3; i++) {
        #pragma HLS UNROLL
        Inc l1 = kP[bk - 1][3 * i] * x[0];
        Inc l2 = kP[bk - 1][3 * i + 1] * x[1];
        Inc l3 = kP[bk - 1][3 * i + 2] * x[2];
        Inc l4 = kW[bk - 1][i] * om1;
        Inc l5 = kEX[bk - 1][i] * xpc;
        Inc l6 = kEA[bk - 1][i] * atc;
        Inc la = l1 + l2;
        Inc lb = l3 + l4;
        Inc lc = l5 + l6;
        Inc ld = lc + kK[bk - 1][i];
        Inc le = la + lb;
        d[i] = le + ld;
    }
    St atm = x[2] + scale2<-1>(d[2]);
    int inr = 0;
    if (atm >= kc[8] && atm <= kc[7]) {
        inr = 1;
    }
    Sig ro = kc[7];
    if (ain == 1 || inr == 1) {
        ro = atm;
    }
    Rr r = hy_recip(ro);
    Sig g1 = kc[7] * r;
    Cw c = g1 * g1;
    Cw crow = c;  // the row: held clamp
    if (ar == 1) {
        crow = 1;
    } else if (ar == 2) {
        crow = kw[0];
    }
    Cw cP = c;  // the output: the gate reclamped at the midpoint (the package's)
    if (inr == 0) {
        if (atm > kc[7]) {
            cP = 1;
        } else {
            cP = kw[0];
        }
    }
    St q = x[3];
    Small bc = kb[0] * crow;
    St q2 = q * q;
    Small bq2 = bc * q2;
    Small aa = kb[0] - bq2;
    Small bq = bc * q;
    Small dd = scale2<1>(bq);
    Small ad = aa * dd;
    Small dl = aa - ad;
    St qm = q + dl;
    St qm2 = qm * qm;
    St qm3 = qm2 * qm;
    Gc gc = kc[7] * cP;
    y[0] = gc * qm3;
    int fl = 0;
    St atr = atm;
    if (ain == 0) {
        atr = atc;
    }
    if (atr < kc[9] || atr > kc[10]) {
        fl = 1;
    }
    Cw hr = crow * qm2;
    Cw hp = cP * qm2;
    Cw four = 4;
    if (hr > four || hp > four) {
        fl = 1;
    }
    if (inr == 1 && atm < kc[9]) {
        fl = 1;
    }
    St zero = 0;
    St qmax = 1.5;
    if (qm <= zero || qm > qmax) {
        fl = 1;
    }
    Sig omin = 0.5;
    if (om < omin) {
        fl = 1;
    }
    int cr = 0;
    if (ain == 1 && inr == 0) {
        cr = 1;
    } else if (ain == 0 && inr == 1) {
        cr = 2;
    }
    Inc dq = scale2<1>(dl);
    for (int i = 0; i < 4; i++) {
        #pragma HLS UNROLL
        Inc di = dq;
        if ((i + 1) < 4) {
            di = d[i];
        }
        Inc ei = di + xc[i];
        St xn = x[i] + ei;
        xc[i] = x[i] + ei - xn;
        x[i] = xn;
    }
    md[0] = bk;
    md[1] = fl;
    md[2] = cr;
}

void hydro_run(const HydroCoefs &k, St x[4], Small xc[4], Sig us[4], Small du[4], int n_on, int n_mid, int n_off, int n0, int nsteps, int decim, Pw y[1], int flg[1], volatile St &at_out, volatile Pw &p_out, volatile bool &strobe, volatile bool &sample) {
    #pragma HLS INTERFACE mode=s_axilite port=return
    #pragma HLS INTERFACE mode=s_axilite port=k
    #pragma HLS INTERFACE mode=s_axilite port=x
    #pragma HLS INTERFACE mode=s_axilite port=xc
    #pragma HLS INTERFACE mode=s_axilite port=us
    #pragma HLS INTERFACE mode=s_axilite port=du
    #pragma HLS INTERFACE mode=s_axilite port=n_on
    #pragma HLS INTERFACE mode=s_axilite port=n_mid
    #pragma HLS INTERFACE mode=s_axilite port=n_off
    #pragma HLS INTERFACE mode=s_axilite port=n0
    #pragma HLS INTERFACE mode=s_axilite port=nsteps
    #pragma HLS INTERFACE mode=s_axilite port=decim
    #pragma HLS INTERFACE mode=s_axilite port=y
    #pragma HLS INTERFACE mode=s_axilite port=flg
    #pragma HLS INTERFACE mode=ap_none port=at_out
    #pragma HLS INTERFACE mode=ap_none port=p_out
    #pragma HLS INTERFACE mode=ap_none port=strobe
    #pragma HLS INTERFACE mode=ap_none port=sample
    St xs[4];
    #pragma HLS ARRAY_PARTITION variable=xs complete
    Small cs[4];  // the remainders of the compensated summation
    #pragma HLS ARRAY_PARTITION variable=cs complete
    for (int i = 0; i < 4; i++) {
        #pragma HLS UNROLL
        xs[i] = x[i];
        cs[i] = xc[i];
    }
    Sig u[1];
    #pragma HLS ARRAY_PARTITION variable=u complete
    Pw yy[1];
    #pragma HLS ARRAY_PARTITION variable=yy complete
    int md[3];
    #pragma HLS ARRAY_PARTITION variable=md complete
    Small kP[16][9];  // the coefficients in registers for the run
    #pragma HLS ARRAY_PARTITION variable=kP complete dim=0
    Small kW[16][3];
    #pragma HLS ARRAY_PARTITION variable=kW complete dim=0
    Small kK[16][3];
    #pragma HLS ARRAY_PARTITION variable=kK complete dim=0
    Small kEX[16][3];
    #pragma HLS ARRAY_PARTITION variable=kEX complete dim=0
    Small kEA[16][3];
    #pragma HLS ARRAY_PARTITION variable=kEA complete dim=0
    Coef kc[11];
    #pragma HLS ARRAY_PARTITION variable=kc complete
    Bt kb[1];
    Cw kw[1];
    for (int b = 0; b < 16; b++) {
        for (int j = 0; j < 9; j++) {
            kP[b][j] = k.P[b][j];
        }
        for (int j = 0; j < 3; j++) {
            kW[b][j] = k.W[b][j];
            kK[b][j] = k.K[b][j];
            kEX[b][j] = k.EX[b][j];
            kEA[b][j] = k.EA[b][j];
        }
    }
    for (int j = 0; j < 11; j++) {
        kc[j] = k.kc[j];
    }
    kb[0] = k.kb[0];
    kw[0] = k.kw[0];
    int fs = flg[0];
    bool st = false;
    int cnt = 0;
    St pat = xs[2];
    Pw ppw = 0;
    bool psa = false;
    bool pst = false;
    for (int i = 0; i < nsteps; i++) {
        #pragma HLS PIPELINE II=24
        #pragma HLS ALLOCATION operation instances=mul limit=12
        at_out = pat;
        p_out = ppw;
        sample = psa;
        strobe = pst;
        int m1 = n0 + (i + 1) - 1;
        int sg = 1;
        int s0 = 0;
        if (m1 >= n_on) {
            sg = 2;
            s0 = n_on;
        }
        if (m1 >= n_mid) {
            sg = 3;
            s0 = n_mid;
        }
        if (m1 >= n_off) {
            sg = 4;
            s0 = n_off;
        }
        int ns = m1 - s0;
        u[0] = us[sg - 1] + du[sg - 1] * ns;
        hydro_step(xs, cs, u, yy, md, kP, kW, kK, kEX, kEA, kc, kb, kw);
        if (md[1] == 1) {
            fs = 1;
        }
        st = !st;
        cnt += 1;
        if (cnt >= decim) {
            cnt = 0;
            psa = true;
        } else {
            psa = false;
        }
        pat = xs[2];
        ppw = yy[0];
        pst = st;
    }
    for (int i = 0; i < 4; i++) {
        #pragma HLS UNROLL
        x[i] = xs[i];
        xc[i] = cs[i];
    }
    y[0] = yy[0];
    flg[0] = fs;
}

