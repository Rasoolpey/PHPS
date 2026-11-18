// shunt.cpp: written by PHPS.RT.hls_write from the Julia model (@hls); do not edit
// the DC shunt exciter (saturated PHShuntExciter), one run of steps of h = 2.16e-7 s; from src/rt/exciters/shunt.jl
#include "shunt.h"

void sh_exp(Targ tp, Mant mk[1], int kk[1]) {
    static const Mant ex2[256] = SH_EXP2;
    static const Mant lc[3] = SH_EXPC;
    int k = ifloor(tp);
    Targ f = tp - k;
    Frac8 f8 = scale2<8>(f);
    int j = ifloor(f8);
    Frac8 r8 = f8 - j;
    Targ r = scale2<-8>(r8);
    Mant one = 1;  // Estrin: (1 + l1 r) + r^2 (l2 + l3 r), two product levels
    Mant r2 = r * r;
    Mant q1 = lc[0] * r;
    Mant q3 = lc[2] * r;
    Mant lo = one + q1;
    Mant hi = lc[1] + q3;
    Mant q2 = r2 * hi;
    Mant pl = lo + q2;
    mk[0] = ex2[j] * pl;
    kk[0] = k;
}

Sat sh_se(St yv, Coef kc[10]) {
    #pragma HLS INLINE off
    #pragma HLS ALLOCATION operation instances=mul limit=3
    static const Sat p2[18] = SH_POW2;
    Targ zero = 0;
    Targ top = 17.999999;
    Targ t1 = kc[5] * yv;
    Targ tp = t1 + kc[6];
    if (tp < zero) {
        tp = zero;
    }
    if (tp > top) {
        tp = top;
    }
    Mant mk[1];
    int kk[1];
    sh_exp(tp, mk, kk);
    Sat se = mk[0] * p2[kk[0] + 1 - 1];
    return se;
}

int sh_decide(St xa, Sig r, Coef vmax, Coef vmin) {
    int nb = 0;
    if (xa >= vmax && r > 0) {
        nb = 1;
    } else if (xa <= vmin && r < 0) {
        nb = 2;
    }
    return nb;
}

int sh_select(Inc gw, Inc gh, Inc gl, St umax, St umin) {
    int cs = 3;
    if (gw >= umin && gw <= umax) {
        cs = 0;
    } else if (gh >= umax) {
        cs = 1;
    } else if (gl <= umin) {
        cs = 2;
    }
    return cs;
}

void shunt_step(St x[5], Sig u[3], Pw y[2], int md[2], Sat se, Small kP[4][9], Small kW[4][3], Small kk[4][3], Small kwx[2][3], Small kwn[2][3], Small kQ[4][3], Small kS[4][3], Coef kc[10], Tiny kt[1]) {
    Sig vt = u[0];
    Sig om = u[1];
    Sig ifd = u[2];
    Sig one = 1;
    Wd zero = 0;
    Sig om1 = om - one;
    St ys = x[4];
    Sig by = kc[7] * ys;
    Sig sb = se * by;
    Sig np = se + sb;  // N'* = SE (1 + B y*)
    Wd sby = sb * ys;
    Wd n0 = zero - sby;  // N0 = N* - N'* y* = -SE B y*^2
    Sig r1 = kc[1] * vt;
    Sig ra = kc[0] - r1;  // KA (Vref - vterm)
    Sig cw = kc[2] * om;  // KA KF omega / TF (input only)
    Sig r3 = cw * x[0];  // one level of products on the loop path
    Sig r4 = kc[2] * x[2];
    Sig tq = ra - x[1];  // ready before the products
    St vmx = kc[3] - x[1];  // the clamp test's bounds (sh_select), from x_n
    St vmn = kc[4] - x[1];
    St umax = scale2<1>(vmx);
    St umin = scale2<1>(vmn);
    Sig rb = r4 - r3;
    Sig r = tq + rb;  // two adds after the products (sweep 5)
    int nb = sh_decide(x[1], r, kc[3], kc[4]);
    int fz = 0;
    if (nb > 0) {
        fz = 1;
    }
    int bW = 1 + 2 * fz;  // the bank of candidate W, and of the constant-VR candidates
    int bS = 2 + 2 * fz;
    Inc kK[4][3];
    #pragma HLS ARRAY_PARTITION variable=kK complete dim=0
    Small gB[4];
    #pragma HLS ARRAY_PARTITION variable=gB complete
    Small hB[4][2];
    #pragma HLS ARRAY_PARTITION variable=hB complete dim=0
    for (int b = 0; b < 4; b++) {
        #pragma HLS UNROLL
        Small aa[3];
        #pragma HLS ARRAY_PARTITION variable=aa complete
        for (int i = 0; i < 3; i++) {
            #pragma HLS UNROLL
            Inc l4 = kW[b][i] * vt;
            Inc l5 = kS[b][i] * n0;
            Inc lc = l4 + l5;
            kK[b][i] = lc + kk[b][i];
            Small qa = kQ[b][i] * om1;
            Small pa = kS[b][i] * np;
            aa[i] = qa + pa;
        }
        Small eb = scale2<-1>(aa[0]);
        Small e2 = eb * eb;
        Small gb = eb + e2;
        gB[b] = gb;
        for (int i = 0; i < 2; i++) {
            #pragma HLS UNROLL
            Small ag = aa[i + 1] * gb;
            Small ta = aa[i + 1] + ag;
            hB[b][i] = scale2<-1>(ta);
        }
    }
    Inc kH[3];
    #pragma HLS ARRAY_PARTITION variable=kH complete
    Inc kL[3];
    #pragma HLS ARRAY_PARTITION variable=kL complete
    for (int i = 0; i < 3; i++) {
        #pragma HLS UNROLL
        kH[i] = kK[bS - 1][i] + kwx[fz][i];
        kL[i] = kK[bS - 1][i] + kwn[fz][i];
    }
    Inc dW[3];
    #pragma HLS ARRAY_PARTITION variable=dW complete
    Inc dH[3];
    #pragma HLS ARRAY_PARTITION variable=dH complete
    Inc dL[3];
    #pragma HLS ARRAY_PARTITION variable=dL complete
    for (int i = 0; i < 3; i++) {
        #pragma HLS UNROLL
        Inc l1 = kP[bW - 1][3 * i] * x[0];
        Inc l2 = kP[bW - 1][3 * i + 1] * x[1];
        Inc l3 = kP[bW - 1][3 * i + 2] * x[2];
        Inc s1 = kP[bS - 1][3 * i] * x[0];
        Inc s2 = kP[bS - 1][3 * i + 1] * x[1];
        Inc s3 = kP[bS - 1][3 * i + 2] * x[2];
        Inc la = l1 + l2;
        Inc lb = l3 + kK[bW - 1][i];
        dW[i] = la + lb;
        Inc sa = s1 + s2;
        Inc sh = s3 + kH[i];
        Inc sl = s3 + kL[i];
        dH[i] = sa + sh;
        dL[i] = sa + sl;
    }
    Small gW = gB[bW - 1];
    Small gS = gB[bS - 1];
    Small hW2 = hB[bW - 1][0];
    Small hW3 = hB[bW - 1][1];
    Small hS2 = hB[bS - 1][0];
    Small hS3 = hB[bS - 1][1];
    St RW = x[0] + scale2<-1>(dW[0]);
    St RH = x[0] + scale2<-1>(dH[0]);
    St RL = x[0] + scale2<-1>(dL[0]);
    Inc rgW = RW * gW;
    Inc rgH = RH * gS;
    Inc rgL = RL * gS;
    Inc mW2 = hW2 * RW;
    Inc mH2 = hS2 * RH;
    Inc mL2 = hS2 * RL;
    Inc mW3 = hW3 * RW;
    Inc mH3 = hS3 * RH;
    Inc mL3 = hS3 * RL;
    Inc fW = dW[0] + scale2<1>(rgW);
    Inc fH = dH[0] + scale2<1>(rgH);
    Inc fL = dL[0] + scale2<1>(rgL);
    Inc gW2 = dW[1] + scale2<1>(mW2);
    Inc gH2 = dH[1] + scale2<1>(mH2);
    Inc gL2 = dL[1] + scale2<1>(mL2);
    int cs = sh_select(gW2, gH2, gL2, umax, umin);
    Inc gW3 = dW[2] + scale2<1>(mW3);
    Inc gH3 = dH[2] + scale2<1>(mH3);
    Inc gL3 = dL[2] + scale2<1>(mL3);
    Inc d1 = fW;
    Inc d2 = gW2;
    Inc d3 = gW3;
    if (cs == 1) {
        d1 = fH;
        d2 = gH2;
        d3 = gH3;
    } else if (cs == 2) {
        d1 = fL;
        d2 = gL2;
        d3 = gL3;
    }
    St ym = x[0] + scale2<-1>(d1);  // y_m for the outputs
    Sig oy = om * ym;
    Pw vf = kt[0] * oy;
    Wd npy = np * ym;
    Wd nm = n0 + npy;  // N(y_m), linearised
    Sig ky = kc[8] * ym;
    Wd fs = ky + nm;
    Sig ifm = kc[9] * fs;  // i_f
    Sig it = ifd + ifm;
    y[0] = vf;
    y[1] = vf * it;
    x[4] = x[3];
    x[3] = x[0];
    x[0] = x[0] + d1;
    x[1] = x[1] + d2;
    x[2] = x[2] + d3;
    md[0] = nb;
    md[1] = cs;
}

void shunt_run(const ShuntCoefs &k, St x[5], Sig us[4][3], Small du[4][3], int n_on, int n_mid, int n_off, int n0, int nsteps, int decim, Pw y[2], volatile St &xa_out, volatile Pw &vf_out, volatile bool &strobe, volatile bool &sample) {
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
    St xs[5];  // the states, in registers during the run
    #pragma HLS ARRAY_PARTITION variable=xs complete
    for (int i = 0; i < 5; i++) {
        #pragma HLS UNROLL
        xs[i] = x[i];
    }
    Sig u[3];
    #pragma HLS ARRAY_PARTITION variable=u complete
    Pw yy[2];
    #pragma HLS ARRAY_PARTITION variable=yy complete
    int md[2];
    #pragma HLS ARRAY_PARTITION variable=md complete
    Small kP[4][9];  // the coefficients in registers for the run (brushless synth 1)
    #pragma HLS ARRAY_PARTITION variable=kP complete dim=0
    Small kW[4][3];
    #pragma HLS ARRAY_PARTITION variable=kW complete dim=0
    Small kk[4][3];
    #pragma HLS ARRAY_PARTITION variable=kk complete dim=0
    Small kQ[4][3];
    #pragma HLS ARRAY_PARTITION variable=kQ complete dim=0
    Small kS[4][3];
    #pragma HLS ARRAY_PARTITION variable=kS complete dim=0
    Small kwx[2][3];
    #pragma HLS ARRAY_PARTITION variable=kwx complete dim=0
    Small kwn[2][3];
    #pragma HLS ARRAY_PARTITION variable=kwn complete dim=0
    Coef kc[10];
    #pragma HLS ARRAY_PARTITION variable=kc complete
    Tiny kt[1];
    #pragma HLS ARRAY_PARTITION variable=kt complete
    for (int b = 0; b < 4; b++) {
        for (int j = 0; j < 9; j++) {
            kP[b][j] = k.P[b][j];
        }
        for (int j = 0; j < 3; j++) {
            kW[b][j] = k.W[b][j];
            kk[b][j] = k.c[b][j];
            kQ[b][j] = k.Q[b][j];
            kS[b][j] = k.S[b][j];
        }
    }
    for (int b = 0; b < 2; b++) {
        for (int j = 0; j < 3; j++) {
            kwx[b][j] = k.wmax[b][j];
            kwn[b][j] = k.wmin[b][j];
        }
    }
    for (int j = 0; j < 10; j++) {
        kc[j] = k.kc[j];
    }
    kt[0] = k.kt[0];
    Sat seh[2];  // SE(y_{n-1}), SE(y_{n-2}) for the first two steps: one call site,
    #pragma HLS ARRAY_PARTITION variable=seh complete
    for (int j = 0; j < 2; j++) {
        seh[j] = sh_se(xs[j + 3], kc);
    }
    Sat se1 = seh[0];
    Sat se2 = seh[1];
    bool st = false;
    int cnt = 0;
    St pxa = xs[1];  // the probes' registers (review 39, 1)
    Pw pvf = 0;
    bool psa = false;
    bool pst = false;
    for (int i = 0; i < nsteps; i++) {
        #pragma HLS PIPELINE II=27
        #pragma HLS ALLOCATION operation instances=mul limit=9
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
        Sat sn = sh_se(xs[0], kc);  // SE(y_n), for the step two iterations later
        shunt_step(xs, u, yy, md, se2, kP, kW, kk, kwx, kwn, kQ, kS, kc, kt);
        se2 = se1;
        se1 = sn;
        st = !st;
        cnt += 1;
        if (cnt >= decim) {
            cnt = 0;
            psa = true;
        } else {
            psa = false;
        }
        pxa = xs[1];
        pvf = yy[0];
        pst = st;
    }
    for (int i = 0; i < 5; i++) {
        #pragma HLS UNROLL
        x[i] = xs[i];
    }
    y[0] = yy[0];
    y[1] = yy[1];
}

