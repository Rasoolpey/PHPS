// dcside.cpp: written by PHPS.RT.hls_write from the Julia model (@hls); do not edit
// the converters' DC side (PHConverterDC), one step of h = 6.08e-7 s; from src/rt/converters/dcside.jl
#include "dcside.h"

Sig vs_rsqrt(Sig a) {
    #pragma HLS INLINE off
    #pragma HLS ALLOCATION operation instances=mul limit=1
    static const Sd seed[4096] = VS_RSQ;
    Rq quarter = 0.25;
    Rq half = 0.5;
    Rq three = 3;
    Sig four = 4;
    Sig sixteen = 16;
    Sig onesig = 1;
    Sig lo = 0.00390625;
    Sig aa = a;
    if (aa < lo) {  // outside the range (flagged by the range check): clamped
        aa = lo;
    }
    Rq m = 0;
    int e = 0;
    if (aa >= sixteen) {
        m = scale2<-6>(aa);
        e = 3;
    } else if (aa >= four) {
        m = scale2<-4>(aa);
        e = 2;
    } else if (aa >= onesig) {
        m = scale2<-2>(aa);
        e = 1;
    } else {
        m = aa;
    }
    for (int i = 0; i < 4; i++) {  // a >= 2^-8: at most four multiplications by 4
        #pragma HLS UNROLL
        if (m < quarter) {
            m = scale2<2>(m);
            e -= 1;
        }
    }
    int ix = 0;
    if (m >= half) {
        ix = ifloor(scale2<12>(m));  // [1/2, 1): 2048 .. 4095
    } else {
        ix = ifloor(scale2<13>(m)) - 2048;  // [1/4, 1/2): 0 .. 2047
    }
    if (ix > 4095) {  // a at the range's top edge (m rounded to 1)
        ix = 4095;
    }
    Rq y = seed[ix];
    for (int it = 0; it < 2; it++) {
        Rq y2 = y * y;
        Rq my2 = m * y2;
        Rq t = three - my2;
        Rq yt = y * t;
        y = scale2<-1>(yt);
    }
    Sig r = y;  // 1 / sqrt(m) in (1, 2]; then 2^-e, shifted from the finer Rq word in
    if (e == 3) {  // one step, so the result is rounded to Sig once (49c)
        r = scale2<-3>(y);
    } else if (e == 2) {
        r = scale2<-2>(y);
    } else if (e == 1) {
        r = scale2<-1>(y);
    } else if (e == -1) {
        r = scale2<1>(y);
    } else if (e == -2) {
        r = scale2<2>(y);
    } else if (e == -3) {
        r = scale2<3>(y);
    } else if (e == -4) {
        r = scale2<4>(y);
    }
    return r;
}

void dcside_core(St x[5], Small xc[4], Sig u[4], Sig y[3], Pw yb[1], int fl[3], Lk pv[4], Lk qv[1], Lr rv[1], Br pb[3][4], Br qb[3], Brr rbr[3], Rinv kq[1], Coef kc[3], Sig kg[6]) {
    #pragma HLS ARRAY_PARTITION variable=x complete
    #pragma HLS ARRAY_PARTITION variable=xc complete
    #pragma HLS ARRAY_PARTITION variable=u complete
    #pragma HLS ARRAY_PARTITION variable=y complete
    #pragma HLS ARRAY_PARTITION variable=fl complete
    Sig half = 0.5;
    St two = 2;
    Sig one = 1;
    Ep lim = 0.01;  // |eps| admitted for the series (certified <= 3.5e-4)
    Sig pr1 = u[2] * u[0];
    Sig pr2 = u[3] * u[1];
    Sig pc = pr1 + pr2;  // Re(a conj(i_c))
    Lr rh = scale2<-1>(rv[0]);  // rho = r_1 / 2
    Inc kap = rh * pc;  // kappa = rho P_c (held: delta = kappa)
    Sig a1 = u[2] * u[2];
    Sig a2 = u[3] * u[3];
    Sig uu2 = a1 + a2;  // |u_c|^2
    Sig rU = vs_rsqrt(uu2);  // 1 / |u_c| (clamped at 2^-8 for |u_c| ~ 0: the clipped path needs |u_c| > m_max v_min)
    Sig ua = uu2 * rU;  // |u_c|
    Sig aa = ua * kg[5];  // |u_c| / m_max: unclipped iff aa <= v_mid
    Sig wc0 = kg[0] * pc;
    Sig wc = wc0 * rU;  // m_max P_c / |u_c|
    Inc dlc = rh * wc;  // the clipped candidate's delta
    Sig sc0 = kg[0] * rU;  // m_max / |u_c| (the clipped bridge voltage's factor, times v_mid)
    Inc pv1 = pv[0] * x[0];  // the increments without w: each product its own variable,
    Inc pv2 = pv[1] * x[1];  // summed as a tree (no serial chain on the recurrence)
    Inc pv3 = pv[2] * x[2];
    Inc pv4 = pv[3] * x[3];
    Inc sa = pv1 + pv2;
    Inc sb = pv3 + pv4;
    Inc sc1 = sa + sb;
    Inc dv0 = sc1 + qv[0];
    Ib db0[3];
    #pragma HLS ARRAY_PARTITION variable=db0 complete
    for (int i = 0; i < 3; i++) {
        #pragma HLS UNROLL
        Ib b1 = pb[i][0] * x[0];
        Ib b2 = pb[i][1] * x[1];
        Ib b3 = pb[i][2] * x[2];
        Ib b4 = pb[i][3] * x[3];
        Ib ba = b1 + b2;
        Ib bb2 = b3 + b4;
        Ib bc = ba + bb2;
        db0[i] = bc + qb[i];
    }
    St vn = x[0];
    St al = vn + scale2<-1>(dv0);  // v_mid when w = 0
    St lo = scale2<-1>(vn + kg[2]);  // the admitted v_mid interval
    St hi = scale2<-1>(vn + kg[3]);
    St z0 = x[4];  // ~ 1 / alpha of the last step: two Newton steps to 1 / alpha
    St n1 = al * z0;
    St m1 = two - n1;
    St z1 = z0 * m1;
    St n2 = al * z1;
    St m2 = two - n2;
    St ai = z1 * m2;
    St gz = n1 - one;  // the carried reciprocal's admission (53, 4): |1 - alpha z| <= 1e-3, so
    if (gz < 0) {  // two Newton steps leave <= 1e-12 relative (else the unclipped
        gz = -gz;  // candidate is not admitted)
    }
    St zl = 0.001;
    bool okz = gz <= zl;
    Inc dl = kap;  // delta (held)
    int clp = 0;
    int acc = 1;
    St vm = al;
    if (kg[4] > half) {
        Inc t = kap * ai;  // kappa / alpha
        Ep ep = t * ai;  // eps (Ep <48,-3>: the 0.01 admission and 5 eps representable, 53, 1; 53c)
        Ep ee = ep * ep;  // eps^2; the series in Estrin form with the constants as shifts:
        Ep e5 = scale2<2>(ep) + ep;  // 5 eps
        Ep e14 = scale2<4>(ee) - scale2<1>(ee);  // 14 eps^2
        Ep e5h = scale2<-1>(e5);  // (halved: the terms stay near 1 in Fs <48,3>, a half unit 1.4e-14)
        Ep e14h = scale2<-1>(e14);
        Fs g1 = one - e5h;  // 1 - 5 eps / 2
        Fs g2 = g1 + e14h;  // (2 - 5 eps + 14 eps^2) / 2
        Ep g3h = ee * g2;
        Ep g3 = scale2<1>(g3h);  // eps^2 (2 - 5 eps + 14 eps^2)
        Fs g4 = one - ep;
        Fs f = g4 + g3;  // 1 - eps + 2 eps^2 - 5 eps^3 + 14 eps^4 (truncation 42 eps^5)
        Inc du = t * f;
        St vu = al + du;
        St vc = al + dlc;
        Ep epa = ep;
        if (epa < 0) {
            epa = -epa;
        }
        bool oku = epa <= lim && (okz && (vu >= lo && (vu <= hi && aa <= vu)));
        bool okc = vc >= lo && (vc <= hi && aa > vc);
        if (oku) {
            dl = du;
            vm = vu;
        } else if (okc) {
            dl = dlc;
            vm = vc;
            clp = 1;
        } else {
            acc = 0;
        }
    } else {
        vm = al + kap;
        if (vm < lo || vm > hi) {
            acc = 0;
        }
    }
    if (acc == 1) {
        Inc dv = dv0 + scale2<1>(dl);  // r_1 w = 2 delta
        Inc ev = dv + xc[0];
        St xv = x[0] + ev;
        xc[0] = x[0] - xv + ev;  // (x - x_new) + e: exact in Float64 (Julia) and fixed point (C++)
        x[0] = xv;
        Sig bm = 0;  // the branches' midpoint sum (for i_b)
        for (int i = 0; i < 3; i++) {
            #pragma HLS UNROLL
            Ib di = db0[i] + rbr[i] * dl;
            Ib ei = di + xc[i + 1];
            St xn = x[i + 1] + ei;
            bm += x[i + 1] + scale2<-1>(di);
            xc[i + 1] = x[i + 1] - xn + ei;
            x[i + 1] = xn;
        }
        x[4] = ai;
        Sig w = dl * kq[0];  // w = delta / rho
        if (kg[4] > half) {
            if (clp == 1) {
                Sig sc = sc0 * vm;  // m_max v_mid / |u_c|
                y[0] = u[2] * sc;
                y[1] = u[3] * sc;
            } else {
                y[0] = u[2];
                y[1] = u[3];
            }
            y[2] = w;
        } else {
            y[0] = u[2] * vm;
            y[1] = u[3] * vm;
            y[2] = pc;
        }
        Sig t1 = bm + vm;
        Sig t2 = kc[1] - t1;  // Eb - sum v_k - v (small: no large cancellation)
        yb[0] = kc[0] * t2;  // i_b at the midpoint
    } else {
        fl[0] = 1;
        Sig bs = x[1] + x[2] + x[3] + vn;
        Sig t3 = kc[1] - bs;
        yb[0] = kc[0] * t3;
        if (kg[4] > half) {
            St q1 = vn * ai;  // 1 / v_n by two Newton steps from 1 / alpha
            St q2 = two - q1;
            St zv1 = ai * q2;
            St q3 = vn * zv1;
            St q4 = two - q3;
            St zv = zv1 * q4;
            if (aa <= vn) {
                y[0] = u[2];
                y[1] = u[3];
                y[2] = pc * zv;
            } else {
                Sig sn = sc0 * vn;
                y[0] = u[2] * sn;
                y[1] = u[3] * sn;
                y[2] = wc;
            }
        } else {
            y[0] = u[2] * vn;
            y[1] = u[3] * vn;
            y[2] = pc;
        }
    }
    fl[1] = clp;
    fl[2] = 1 - acc;
}

void dcside_run(const DcCoefs &k, St x[5], Small xc[4], Sig u[512][4], Sig y[512][3], Pw yb[512], int fo[512], int flg[1], int nstep, int decim, volatile St &v_out, volatile bool &strobe, volatile bool &sample) {
    #pragma HLS INTERFACE mode=s_axilite port=return
    #pragma HLS INTERFACE mode=s_axilite port=k
    #pragma HLS INTERFACE mode=s_axilite port=x
    #pragma HLS INTERFACE mode=s_axilite port=xc
    #pragma HLS INTERFACE mode=s_axilite port=u
    #pragma HLS INTERFACE mode=s_axilite port=y
    #pragma HLS INTERFACE mode=s_axilite port=yb
    #pragma HLS INTERFACE mode=s_axilite port=fo
    #pragma HLS INTERFACE mode=s_axilite port=flg
    #pragma HLS INTERFACE mode=s_axilite port=nstep
    #pragma HLS INTERFACE mode=s_axilite port=decim
    #pragma HLS INTERFACE mode=ap_none port=v_out
    #pragma HLS INTERFACE mode=ap_none port=strobe
    #pragma HLS INTERFACE mode=ap_none port=sample
    St xs[5];
    #pragma HLS ARRAY_PARTITION variable=xs complete
    Small cs[4];
    #pragma HLS ARRAY_PARTITION variable=cs complete
    for (int i = 0; i < 4; i++) {
        #pragma HLS UNROLL
        cs[i] = xc[i];
    }
    for (int i = 0; i < 5; i++) {
        #pragma HLS UNROLL
        xs[i] = x[i];
    }
    Lk pv[4];
    #pragma HLS ARRAY_PARTITION variable=pv complete
    Lk qv[1];
    Lr rv[1];
    Br pb[3][4];
    #pragma HLS ARRAY_PARTITION variable=pb complete dim=0
    Br qb[3];
    #pragma HLS ARRAY_PARTITION variable=qb complete
    Brr rbr[3];
    #pragma HLS ARRAY_PARTITION variable=rbr complete
    Rinv kq[1];
    Coef kc[3];
    #pragma HLS ARRAY_PARTITION variable=kc complete
    Sig kg[6];
    #pragma HLS ARRAY_PARTITION variable=kg complete
    for (int j = 0; j < 4; j++) {
        pv[j] = k.pv[j];
        for (int i = 0; i < 3; i++) {
            pb[i][j] = k.pb[i][j];
        }
    }
    qv[0] = k.qv[0];
    rv[0] = k.rv[0];
    kq[0] = k.kq[0];
    for (int i = 0; i < 3; i++) {
        qb[i] = k.qb[i];
        rbr[i] = k.rbr[i];
        kc[i] = k.kc[i];
    }
    for (int i = 0; i < 6; i++) {
        kg[i] = k.kg[i];
    }
    Sig uu[4];
    #pragma HLS ARRAY_PARTITION variable=uu complete
    Sig yy[3];
    #pragma HLS ARRAY_PARTITION variable=yy complete
    Pw bb[1];
    int ff[3];
    #pragma HLS ARRAY_PARTITION variable=ff complete
    ff[0] = flg[0];  // the sticky rejection flag, carried across calls (53, 6)
    ff[2] = 0;
    bool st = false;
    int cnt = 0;
    for (int n = 0; n < nstep; n++) {
        #pragma HLS PIPELINE II=76
        #pragma HLS LOOP_TRIPCOUNT min=1 max=512
        for (int i = 0; i < 4; i++) {
            uu[i] = u[n][i];
        }
        dcside_core(xs, cs, uu, yy, bb, ff, pv, qv, rv, pb, qb, rbr, kq, kc, kg);
        y[n][0] = yy[0];
        y[n][1] = yy[1];
        y[n][2] = yy[2];
        yb[n] = bb[0];
        fo[n] = ff[2] + 2 * ff[1];  // this step: rejected + 2 clipped
        st = !st;
        cnt += 1;
        if (cnt >= decim) {
            cnt = 0;
            sample = true;
        } else {
            sample = false;
        }
        v_out = xs[0];
        strobe = st;
    }
    for (int i = 0; i < 4; i++) {
        #pragma HLS UNROLL
        xc[i] = cs[i];
    }
    for (int i = 0; i < 5; i++) {
        #pragma HLS UNROLL
        x[i] = xs[i];
    }
    flg[0] = ff[0];
}

