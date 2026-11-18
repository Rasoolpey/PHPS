// brushless.cpp: written by PHPS.RT.hls_write from the Julia model (@hls); do not edit
// the brushless exciter (unsaturated PHBrushlessExciter), one run of steps of h = 1.92e-7 s; from src/rt/exciters/brushless.jl
#include "brushless.h"

Sig bx_sqrt(Sig a) {
    #pragma HLS INLINE off
    #pragma HLS ALLOCATION operation instances=mul limit=1
    static const Rq seed[256] = BX_SEED;
    Rq quarter = 0.25;
    Rq half = 0.5;
    Rq three = 3;
    Sig four = 4;
    Sig sixteen = 16;
    Sig onesig = 1;
    Rq m = 0;  // a >= 1 is scaled down exactly before it enters Rq (+-8)
    int e = 0;
    if (a >= sixteen) {  // a < 64: a / 4^3
        m = scale2<-6>(a);
        e = 3;
    } else if (a >= four) {
        m = scale2<-4>(a);
        e = 2;
    } else if (a >= onesig) {
        m = scale2<-2>(a);
        e = 1;
    } else {
        m = a;
    }
    for (int i = 0; i < 6; i++) {  // a >= 2^-12: at most six multiplications by 4
        if (m < quarter) {
            m = scale2<2>(m);
            e -= 1;
        }
    }
    if (m < quarter) {  // below the range (a < 2^-12; flagged by the export's range check)
        m = quarter;
    }
    int ix = 0;  // the seed's index: octave and the top 7 bits below the leading one
    if (m >= half) {
        ix = ifloor(scale2<8>(m));  // m in [1/2, 1): 128 .. 255
    } else {
        ix = ifloor(scale2<9>(m)) - 128;  // m in [1/4, 1/2): 0 .. 127
    }
    Rq y = seed[ix];
    for (int it = 0; it < 2; it++) {
        Rq y2 = y * y;
        Rq my2 = m * y2;
        Rq t = three - my2;
        Rq yt = y * t;
        y = scale2<-1>(yt);
    }
    Rq r = m * y;  // sqrt(m) in [1/2, 1)
    for (int i = 0; i < 3; i++) {
        if (e > 0) {
            r = scale2<1>(r);
            e -= 1;
        }
    }
    for (int i = 0; i < 6; i++) {
        if (e < 0) {
            r = scale2<-1>(r);
            e += 1;
        }
    }
    return r;
}

void brushless_step(St x[3], Sig u[3], Pw y[2], int md[3], Small kP[4][9], Small kW[4][6], Small kk[4][3], Small kwx[2][3], Small kwn[2][3], Coef kc[15], Tiny kt[3]) {
    Sig vt = u[0];
    Sig om = u[1];
    Sig ifd = u[2];
    Sig IFD = kc[9] * ifd;
    Sig kdI = kc[5] * IFD;
    Sig ve = kc[6] - vt;
    Sig r1 = kc[0] * ve;  // KA (Vref - vterm)
    Sig r2 = kc[3] * IFD;  // KA KF KD / TF IFD
    Sig r3 = kc[1] * x[0];  // -KA KF KE / TF x_E: one level of products on the loop path
    Sig r4 = kc[2] * x[2];  // KA KF / TF xf
    Sig ra = r1 - r2;
    Sig rb = r3 + r4;
    Sig r = ra + rb - x[1];
    int nb = 0;
    if (x[1] >= kc[7] && r > 0) {
        nb = 1;
    } else if (x[1] <= kc[8] && r < 0) {
        nb = 2;
    }
    int fz = 0;
    if (nb > 0) {
        fz = 1;
    }
    int bL = 1 + 2 * fz;  // the bank of candidate L, and of the sat candidates
    int bS = 2 + 2 * fz;
    Sig uu[2] = {vt, IFD};
    #pragma HLS ARRAY_PARTITION variable=uu complete
    Inc dL[3];
    #pragma HLS ARRAY_PARTITION variable=dL complete
    Inc dS[3];
    #pragma HLS ARRAY_PARTITION variable=dS complete
    for (int i = 0; i < 3; i++) {
        #pragma HLS UNROLL
        Inc l1 = kP[bL - 1][3 * i] * x[0];
        Inc l2 = kP[bL - 1][3 * i + 1] * x[1];
        Inc l3 = kP[bL - 1][3 * i + 2] * x[2];
        Inc l4 = kW[bL - 1][2 * i] * uu[0];
        Inc l5 = kW[bL - 1][2 * i + 1] * uu[1];
        Inc s1 = kP[bS - 1][3 * i] * x[0];
        Inc s2 = kP[bS - 1][3 * i + 1] * x[1];
        Inc s3 = kP[bS - 1][3 * i + 2] * x[2];
        Inc s4 = kW[bS - 1][2 * i] * uu[0];
        Inc s5 = kW[bS - 1][2 * i + 1] * uu[1];
        Inc la = l1 + l2;
        Inc lb = l3 + kk[bL - 1][i];
        Inc lc = l4 + l5;
        Inc ld = la + lb;
        dL[i] = ld + lc;
        Inc sa = s1 + s2;
        Inc sb = s3 + kk[bS - 1][i];
        Inc sc = s4 + s5;
        Inc sd = sa + sb;
        dS[i] = sd + sc;
    }
    Sig xamL = x[1] + scale2<-1>(dL[1]);
    Inc dH2 = dS[1] + kwx[fz][1];
    Inc dW2 = dS[1] + kwn[fz][1];
    Sig xamH = x[1] + scale2<-1>(dH2);
    Sig xamW = x[1] + scale2<-1>(dW2);
    Inc d[3];
    #pragma HLS ARRAY_PARTITION variable=d complete
    Sig vrm = xamL;
    int cs = 0;
    if (xamL >= kc[8] && xamL <= kc[7]) {
        cs = 0;
    } else if (xamH >= kc[7]) {
        cs = 1;
    } else if (xamW <= kc[8]) {
        cs = 2;
    } else {
        cs = 3;
    }
    for (int i = 0; i < 3; i++) {
        #pragma HLS UNROLL
        if (cs == 1) {
            d[i] = dS[i] + kwx[fz][i];
        } else if (cs == 2) {
            d[i] = dS[i] + kwn[fz][i];
        } else {
            d[i] = dL[i];
        }
    }
    if (cs == 1) {
        vrm = kc[7];
    } else if (cs == 2) {
        vrm = kc[8];
    }
    Sig xEm = x[0] + scale2<-1>(d[0]);
    Sig kem = kc[4] * xEm;
    Sig vfem = kem + kdI;
    Sig kcI = 0;
    if (IFD > 0) {
        kcI = kc[10] * IFD;
    }
    Sig t1 = kc[13] * xEm;
    Sig t2 = kc[14] * xEm;
    int rm = 0;
    if (xEm <= 0) {
        rm = 0;
    } else if (kcI <= t1) {
        rm = 1;
    } else if (kcI < t2) {
        rm = 2;
    } else if (kcI < xEm) {
        rm = 3;
    } else {
        rm = 4;
    }
    Coef al = 1;
    Coef be = kc[11];
    if (rm == 3) {
        al = kc[12];
        be = kc[12];
    }
    Sig lx = al * xEm;
    Sig lk = be * kcI;
    Sig lin = lx - lk;
    Sig xe2 = xEm * xEm;  // mode II: sqrt((3/4) x_E^2 - (KC IFD)^2)
    Sig kc2 = kcI * kcI;
    Sig q2 = kc[14] * xe2;
    Sig ar = q2 - kc2;
    Sig rad = 0;
    if (rm == 2) {
        rad = bx_sqrt(ar);
    }
    Sig ef = 0;
    if (rm == 1 || rm == 3) {
        ef = lin;
    } else if (rm == 2) {
        ef = rad;
    }
    Sig efd = om * ef;
    Pw vf = kt[0] * efd;
    Pw p1 = vf * ifd;
    Wd pa = vrm * kem;
    Wd pb = kdI * vfem;
    Wd pc = pa + pb;
    Pw p2 = kt[1] * pc;
    y[0] = vf;
    y[1] = p1 + p2;
    for (int i = 0; i < 3; i++) {
        #pragma HLS UNROLL
        x[i] = x[i] + d[i];
    }
    md[0] = nb;
    md[1] = cs;
    md[2] = rm;
}

void brushless_run(const BrushlessCoefs &k, St x[3], Sig us[4][3], Small du[4][3], int n_on, int n_mid, int n_off, int n0, int nsteps, int decim, Pw y[2], volatile St &xa_out, volatile Pw &vf_out, volatile bool &strobe, volatile bool &sample) {
    #pragma HLS INTERFACE mode=s_axilite port=return
    #pragma HLS INTERFACE mode=s_axilite port=k
    #pragma HLS INTERFACE mode=s_axilite port=x
    #pragma HLS INTERFACE mode=s_axilite port=us
    #pragma HLS INTERFACE mode=s_axilite port=du
    #pragma HLS INTERFACE mode=s_axilite port=n_on
    #pragma HLS INTERFACE mode=s_axilite port=n_mid
    #pragma HLS INTERFACE mode=s_axilite port=n_off
    #pragma HLS INTERFACE mode=s_axilite port=n0
    #pragma HLS INTERFACE mode=s_axilite port=nsteps
    #pragma HLS INTERFACE mode=s_axilite port=decim
    #pragma HLS INTERFACE mode=s_axilite port=y
    #pragma HLS INTERFACE mode=ap_none port=xa_out
    #pragma HLS INTERFACE mode=ap_none port=vf_out
    #pragma HLS INTERFACE mode=ap_none port=strobe
    #pragma HLS INTERFACE mode=ap_none port=sample
    St xs[3];  // the states, in registers during the run
    #pragma HLS ARRAY_PARTITION variable=xs complete
    for (int i = 0; i < 3; i++) {
        #pragma HLS UNROLL
        xs[i] = x[i];
    }
    Sig u[3];
    #pragma HLS ARRAY_PARTITION variable=u complete
    Pw yy[2];
    #pragma HLS ARRAY_PARTITION variable=yy complete
    int md[3];
    #pragma HLS ARRAY_PARTITION variable=md complete
    Small kP[4][9];  // the coefficients in registers for the run (read once over
    #pragma HLS ARRAY_PARTITION variable=kP complete dim=0
    Small kW[4][6];
    #pragma HLS ARRAY_PARTITION variable=kW complete dim=0
    Small kk[4][3];
    #pragma HLS ARRAY_PARTITION variable=kk complete dim=0
    Small kwx[2][3];
    #pragma HLS ARRAY_PARTITION variable=kwx complete dim=0
    Small kwn[2][3];
    #pragma HLS ARRAY_PARTITION variable=kwn complete dim=0
    Coef kc[15];
    #pragma HLS ARRAY_PARTITION variable=kc complete
    Tiny kt[3];
    #pragma HLS ARRAY_PARTITION variable=kt complete
    for (int b = 0; b < 4; b++) {
        for (int j = 0; j < 9; j++) {
            kP[b][j] = k.P[b][j];
        }
        for (int j = 0; j < 6; j++) {
            kW[b][j] = k.W[b][j];
        }
        for (int j = 0; j < 3; j++) {
            kk[b][j] = k.c[b][j];
        }
    }
    for (int b = 0; b < 2; b++) {
        for (int j = 0; j < 3; j++) {
            kwx[b][j] = k.wmax[b][j];
            kwn[b][j] = k.wmin[b][j];
        }
    }
    for (int j = 0; j < 15; j++) {
        kc[j] = k.kc[j];
    }
    for (int j = 0; j < 3; j++) {
        kt[j] = k.kt[j];
    }
    bool st = false;
    int cnt = 0;
    St pxa = xs[1];  // the probes' registers: the previous step's values, written to
    Pw pvf = 0;  // all four ports together at the start of each iteration, so
    bool psa = false;  // the ILA sees one step in every probe (review 39, 1)
    bool pst = false;
    for (int i = 0; i < nsteps; i++) {
        #pragma HLS PIPELINE II=24
        xa_out = pxa;
        vf_out = pvf;
        sample = psa;
        strobe = pst;
        int m1 = n0 + (i + 1) - 1;  // steps before this one
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
        for (int j = 0; j < 3; j++) {
            #pragma HLS UNROLL
            u[j] = us[sg - 1][j] + du[sg - 1][j] * ns;
        }
        brushless_step(xs, u, yy, md, kP, kW, kk, kwx, kwn, kc, kt);
        st = !st;
        cnt += 1;
        if (cnt >= decim) {
            cnt = 0;
            psa = true;
        } else {
            psa = false;
        }
        pxa = xs[1];  // to the ILA in the next iteration
        pvf = yy[0];
        pst = st;
    }
    for (int i = 0; i < 3; i++) {
        #pragma HLS UNROLL
        x[i] = xs[i];
    }
    y[0] = yy[0];
    y[1] = yy[1];
}

